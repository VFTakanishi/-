from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from podcast_clipper import clip_selector, config
from podcast_clipper.clip_selector import (
    CandidateOutput,
    CandidateSegmentOutput,
    Stage1Output,
    Stage2Output,
)
from podcast_clipper.models import RawClipCandidate, RawUsedSegment, Transcript, TranscriptSegment, TranscriptWord


@pytest.fixture(autouse=True)
def _forbid_real_anthropic_client(monkeypatch):
    """Every test in this module must go through a mocked client (e.g. via
    clip_selector._client, or by mocking rank_and_finalize/run_stage1
    higher up) -- never a real anthropic.Anthropic(). Poisoning the
    constructor turns an accidental real API call into an immediate, loud
    test failure instead of a silent live call to Anthropic.
    """

    def _forbidden(*args, **kwargs):
        raise AssertionError("real anthropic.Anthropic() must not be instantiated in tests")

    monkeypatch.setattr(clip_selector.anthropic, "Anthropic", _forbidden)


def _segment(i, start):
    return TranscriptSegment(
        id=i, start=start, end=start + 2.0, text=f"segment {i}",
        words=[TranscriptWord(start=start, end=start + 2.0, text=f"seg{i}")],
    )


def _long_transcript(minutes=25):
    segments = [_segment(i, start=i * 20.0) for i in range(int(minutes * 60 / 20))]
    return Transcript(video_id="vid1", language="ja", segments=segments)


def test_build_chunks_covers_whole_transcript_with_overlap(monkeypatch):
    monkeypatch.setattr(config, "CHUNK_MINUTES", 10.0)
    monkeypatch.setattr(config, "CHUNK_OVERLAP_MINUTES", 1.0)
    transcript = _long_transcript(minutes=25)

    chunks = clip_selector._build_chunks(transcript.segments)

    assert len(chunks) >= 3  # 25 min / (10 min window, 1 min stride overlap) needs multiple chunks
    # every segment must be covered by at least one chunk
    covered_ids = {s.id for _, segs in chunks for s in segs}
    assert covered_ids == {s.id for s in transcript.segments}


def test_usable_segments_returns_all_when_op_exclusion_unset(monkeypatch):
    monkeypatch.setattr(config, "OP_EXCLUSION_SECONDS", None)
    transcript = _long_transcript(minutes=2)
    assert clip_selector._usable_segments(transcript) == transcript.segments


def test_usable_segments_excludes_only_when_explicitly_configured(monkeypatch):
    monkeypatch.setattr(config, "OP_EXCLUSION_SECONDS", 30.0)
    transcript = _long_transcript(minutes=2)
    usable = clip_selector._usable_segments(transcript)
    assert all(s.start >= 30.0 for s in usable)
    assert len(usable) < len(transcript.segments)


def _raw_candidate(start_id, end_id, role="hook", opening_hook_strength=80):
    return RawClipCandidate(
        hook_type="story",
        segments=[RawUsedSegment(role=role, start_segment_id=start_id, end_segment_id=end_id)],
        hook_text="h", opening_hook_strength=opening_hook_strength, title="t", description="d",
        score=80, reasoning="r", caveats="",
    )


def test_select_candidates_retries_once_on_out_of_range_duration(monkeypatch):
    transcript = _long_transcript(minutes=1)  # 3 short segments

    monkeypatch.setattr(config, "MAX_STAGE2_RETRIES", 1)
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 20.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 50.0)

    monkeypatch.setattr(
        clip_selector, "run_stage1",
        lambda t, title, force_refresh=False: [{"chunk_index": 0, "candidates": [_raw_candidate(0, 0)]}],
    )

    call_count = {"n": 0}

    def fake_rank_and_finalize(all_candidates, transcript, video_title, feedback=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # too short: 2s segment duration, way below 20s hard minimum
            return [_raw_candidate(0, 0), _raw_candidate(0, 0), _raw_candidate(0, 0)]
        # second call (after feedback) returns a longer, in-range candidate
        return [_raw_candidate(0, 2), _raw_candidate(0, 2), _raw_candidate(0, 2)]

    monkeypatch.setattr(clip_selector, "rank_and_finalize", fake_rank_and_finalize)

    result = clip_selector.select_candidates(transcript, "タイトル")

    assert call_count["n"] == 2  # exactly one retry, per config.MAX_STAGE2_RETRIES=1
    assert len(result) == 3


def test_select_candidates_retries_when_opening_hook_is_weak(monkeypatch):
    """Stage2 must not settle for a candidate whose *actual spoken* opening
    is weak, even if its overall score/duration look fine -- it should
    retry with feedback and prefer a candidate whose real transcript
    opening is strong.
    """
    transcript = _long_transcript(minutes=1)  # segments 0/1/2, 20s apart
    # give segment 0 a literal weak "warm-up" opening the prefix list catches
    transcript.segments[0].text = "今回はトランプ関税について話していきます"

    monkeypatch.setattr(config, "MAX_STAGE2_RETRIES", 1)
    monkeypatch.setattr(
        clip_selector, "run_stage1",
        lambda t, title, force_refresh=False: [
            {"chunk_index": 0, "candidates": [_raw_candidate(0, 0), _raw_candidate(2, 2)]}
        ],
    )

    call_count = {"n": 0}

    def fake_rank_and_finalize(all_candidates, transcript, video_title, feedback=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # weak spoken opening (segment 0's literal "warm-up" text)
            return [_raw_candidate(0, 0) for _ in range(3)]
        # after feedback: a strong opening (segment 2, untouched text)
        return [_raw_candidate(2, 2) for _ in range(3)]

    monkeypatch.setattr(clip_selector, "rank_and_finalize", fake_rank_and_finalize)

    result = clip_selector.select_candidates(transcript, "タイトル")

    assert call_count["n"] == 2  # exactly one retry
    assert all(c.segments[0].start_segment_id == 2 for c in result)


def test_select_candidates_forces_first_segment_role_to_hook(monkeypatch):
    """segments[0].role must always be "hook" in the final candidates, even
    if Claude tagged it differently -- this is a labeling/consistency fix,
    not a semantic re-decision, so it's corrected mechanically rather than
    triggering a retry.
    """
    transcript = _long_transcript(minutes=1)

    monkeypatch.setattr(
        clip_selector, "run_stage1",
        lambda t, title, force_refresh=False: [
            {"chunk_index": 0, "candidates": [_raw_candidate(0, 0)]}
        ],
    )
    monkeypatch.setattr(
        clip_selector, "rank_and_finalize",
        lambda all_candidates, transcript, video_title, feedback=None: [
            _raw_candidate(0, 0, role="context") for _ in range(3)
        ],
    )

    result = clip_selector.select_candidates(transcript, "タイトル")

    assert all(c.segments[0].role == "hook" for c in result)


def test_select_candidates_caches_and_skips_recompute(monkeypatch):
    from podcast_clipper import cache

    transcript = _long_transcript(minutes=1)
    cache.save_stage2("vid1", [_raw_candidate(0, 0)] * 3)

    monkeypatch.setattr(
        clip_selector, "run_stage1", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run"))
    )

    result = clip_selector.select_candidates(transcript, "タイトル")
    assert len(result) == 3


# --- Structured Outputs model tests (schema conformance is now guaranteed
# --- by Claude's Structured Outputs at the API level; Pydantic is the local
# --- enforcement of the constraints the API strips from the sent schema,
# --- e.g. numeric ranges and array length -- see clip_selector.py's
# --- CandidateOutput/Stage1Output/Stage2Output docstrings). No JSON-repair
# --- code exists any more -- a malformed response is expected to raise a
# --- pydantic.ValidationError (or be rejected by the SDK before it ever
# --- reaches this code), not be silently patched up.


def _valid_segment_kwargs():
    return {"role": "hook", "start_segment_id": 0, "end_segment_id": 0}


def _valid_candidate_kwargs():
    return {
        "hook_type": "story",
        "segments": [_valid_segment_kwargs()],
        "hook_text": "h", "opening_hook_strength": 80,
        "title": "t", "description": "d", "score": 80, "reasoning": "r", "caveats": "",
    }


# A: Stage1 may return 0 candidates
def test_stage1_output_accepts_zero_candidates():
    assert Stage1Output(candidates=[]).candidates == []


# B: Stage1 may return 1-3 candidates
def test_stage1_output_accepts_one_to_three_candidates():
    for n in (1, 2, 3):
        out = Stage1Output(candidates=[CandidateOutput(**_valid_candidate_kwargs()) for _ in range(n)])
        assert len(out.candidates) == n


def test_stage1_output_rejects_more_than_three_candidates():
    with pytest.raises(ValidationError):
        Stage1Output(candidates=[CandidateOutput(**_valid_candidate_kwargs()) for _ in range(4)])


# C: Stage2 must return exactly 3 candidates
def test_stage2_output_requires_exactly_three_candidates():
    Stage2Output(candidates=[CandidateOutput(**_valid_candidate_kwargs()) for _ in range(3)])
    with pytest.raises(ValidationError):
        Stage2Output(candidates=[CandidateOutput(**_valid_candidate_kwargs()) for _ in range(2)])
    with pytest.raises(ValidationError):
        Stage2Output(candidates=[CandidateOutput(**_valid_candidate_kwargs()) for _ in range(4)])


# D: a candidate's segments must have 1-3 items
def test_candidate_output_segments_length_bounds():
    for n in (1, 2, 3):
        kwargs = _valid_candidate_kwargs()
        kwargs["segments"] = [_valid_segment_kwargs() for _ in range(n)]
        CandidateOutput(**kwargs)

    kwargs = _valid_candidate_kwargs()
    kwargs["segments"] = []
    with pytest.raises(ValidationError):
        CandidateOutput(**kwargs)

    kwargs = _valid_candidate_kwargs()
    kwargs["segments"] = [_valid_segment_kwargs() for _ in range(4)]
    with pytest.raises(ValidationError):
        CandidateOutput(**kwargs)


# E: opening_hook_strength must be within 0-100
def test_candidate_output_opening_hook_strength_bounds():
    for value in (0, 100):
        kwargs = _valid_candidate_kwargs()
        kwargs["opening_hook_strength"] = value
        CandidateOutput(**kwargs)

    for value in (-1, 101):
        kwargs = _valid_candidate_kwargs()
        kwargs["opening_hook_strength"] = value
        with pytest.raises(ValidationError):
            CandidateOutput(**kwargs)


# F: score must be within 0-100
def test_candidate_output_score_bounds():
    for value in (0, 100):
        kwargs = _valid_candidate_kwargs()
        kwargs["score"] = value
        CandidateOutput(**kwargs)

    for value in (-1, 101):
        kwargs = _valid_candidate_kwargs()
        kwargs["score"] = value
        with pytest.raises(ValidationError):
            CandidateOutput(**kwargs)


# G: an invalid hook_type must be rejected
def test_candidate_output_rejects_invalid_hook_type():
    kwargs = _valid_candidate_kwargs()
    kwargs["hook_type"] = "not_a_real_hook_type"
    with pytest.raises(ValidationError):
        CandidateOutput(**kwargs)


# H: an invalid segment role must be rejected
def test_candidate_segment_output_rejects_invalid_role():
    with pytest.raises(ValidationError):
        CandidateSegmentOutput(role="not_a_real_role", start_segment_id=0, end_segment_id=0)


# I: a wrongly-typed segment id must be rejected
def test_candidate_segment_output_rejects_wrongly_typed_segment_id():
    with pytest.raises(ValidationError):
        CandidateSegmentOutput(role="hook", start_segment_id=["not", "an", "int"], end_segment_id=0)


class _FakeMessages:
    def __init__(self, parsed_output):
        self._parsed_output = parsed_output

    def parse(self, **kwargs):
        return SimpleNamespace(parsed_output=self._parsed_output)


class _FakeClient:
    def __init__(self, parsed_output):
        self.messages = _FakeMessages(parsed_output)


# J: Stage1 output converts correctly into RawClipCandidate
def test_extract_candidates_for_chunk_converts_structured_output(monkeypatch):
    parsed = Stage1Output(candidates=[CandidateOutput(**_valid_candidate_kwargs())])
    monkeypatch.setattr(clip_selector, "_client", lambda: _FakeClient(parsed))

    segments = [_segment(0, start=0.0)]
    result = clip_selector.extract_candidates_for_chunk(segments, "タイトル")

    assert len(result) == 1
    assert isinstance(result[0], RawClipCandidate)
    assert result[0].hook_type == "story"
    assert result[0].segments[0].role == "hook"


# K: Stage2 output converts correctly into RawClipCandidate, exactly 3
def test_rank_and_finalize_converts_structured_output(monkeypatch):
    parsed = Stage2Output(candidates=[CandidateOutput(**_valid_candidate_kwargs()) for _ in range(3)])
    monkeypatch.setattr(clip_selector, "_client", lambda: _FakeClient(parsed))

    transcript = _long_transcript(minutes=1)
    result = clip_selector.rank_and_finalize([_raw_candidate(0, 0)], transcript, "タイトル")

    assert len(result) == 3
    assert all(isinstance(c, RawClipCandidate) for c in result)


# P: the real Anthropic client must never be constructed, even by the safety
# net's own escape hatch -- confirms _forbid_real_anthropic_client actually
# poisons the constructor rather than silently no-oping.
def test_forbid_real_anthropic_client_fixture_actually_blocks_construction():
    with pytest.raises(AssertionError):
        clip_selector._client()
