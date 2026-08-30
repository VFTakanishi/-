import json

import pytest

from podcast_clipper import clip_selector, config
from podcast_clipper.models import (
    MalformedCandidateError,
    RawClipCandidate,
    RawUsedSegment,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)


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


# --- diagnostics for the 2026-08-30 "string indices must be integers, not
# --- 'str'" incident: a malformed item from Claude's real tool_use
# --- response reproduced that exact error with no traceback captured. These
# --- pin that the parsing function now raises a diagnosable error naming
# --- the stage, index, and actual type/value instead. No retry/repair
# --- behavior is added -- this is diagnostics only.


def test_raw_candidate_from_tool_input_raises_diagnosable_error_when_candidate_is_str():
    with pytest.raises(MalformedCandidateError) as exc_info:
        clip_selector._raw_candidate_from_tool_input(
            "oops not a dict", stage="Stage1", candidate_index=2
        )
    message = str(exc_info.value)
    assert "Stage1" in message
    assert "candidates[2]" in message
    assert "str" in message
    assert "oops not a dict" in message


def test_raw_candidate_from_tool_input_raises_diagnosable_error_when_segment_is_str():
    malformed = {
        "hook_type": "story",
        "segments": [{"role": "hook", "start_segment_id": 0, "end_segment_id": 0}, "bad segment"],
        "hook_text": "h", "opening_hook_strength": 80,
        "title": "t", "description": "d", "score": 80, "reasoning": "r", "caveats": "",
    }
    with pytest.raises(MalformedCandidateError) as exc_info:
        clip_selector._raw_candidate_from_tool_input(
            malformed, stage="Stage2", candidate_index=0
        )
    message = str(exc_info.value)
    assert "Stage2" in message
    assert "candidates[0].segments[1]" in message
    assert "str" in message
    assert "bad segment" in message


# --- 2026-08-30 follow-up incident: real-machine diagnosis (job JSON's
# --- error_traceback) confirmed the actual cause was
# --- `block.input["candidates"]` arriving as a JSON-*encoded string*
# --- (e.g. the literal text "[{...}, ...]") instead of an already-parsed
# --- list, so `for c in ...["candidates"]` iterated the string character
# --- by character and candidates[0] came out as "[". These tests use only
# --- fixtures/mocks -- the autouse fixture above makes any accidental real
# --- Anthropic call fail loudly instead of silently reaching the network.


def _valid_candidate_dict():
    return {
        "hook_type": "story",
        "segments": [{"role": "hook", "start_segment_id": 0, "end_segment_id": 0}],
        "hook_text": "h", "opening_hook_strength": 80,
        "title": "t", "description": "d", "score": 80, "reasoning": "r", "caveats": "",
    }


# A: candidates is a normal list[dict] -> unaffected, still passes straight through
def test_normalize_json_array_accepts_plain_list():
    payload = [_valid_candidate_dict()]
    assert clip_selector._normalize_json_array(payload, context="ctx") == payload


# B: candidates is a JSON-encoded string of a valid array -> parsed successfully
def test_normalize_json_array_decodes_valid_json_string_once():
    payload = [_valid_candidate_dict(), _valid_candidate_dict()]
    result = clip_selector._normalize_json_array(json.dumps(payload), context="ctx")
    assert result == payload


# C: candidates == "[" (the exact real-machine shape) -> diagnosable error
def test_normalize_json_array_reproduces_the_reported_incident_shape():
    with pytest.raises(MalformedCandidateError) as exc_info:
        clip_selector._normalize_json_array("[", context="Stage1 candidates")
    message = str(exc_info.value)
    assert "Stage1 candidates" in message
    assert "isn't valid JSON" in message


# D: candidates is a JSON string but decodes to a dict, not a list -> error
def test_normalize_json_array_rejects_json_string_decoding_to_non_list():
    with pytest.raises(MalformedCandidateError) as exc_info:
        clip_selector._normalize_json_array(json.dumps({"not": "a list"}), context="ctx")
    message = str(exc_info.value)
    assert "ctx" in message
    assert "dict" in message


def test_normalize_json_array_rejects_non_list_non_str():
    with pytest.raises(MalformedCandidateError):
        clip_selector._normalize_json_array(123, context="ctx")


# E: candidate dict is fine, but its "segments" arrived as a JSON string -> parsed
def test_raw_candidate_from_tool_input_accepts_segments_as_json_string():
    payload = _valid_candidate_dict()
    payload["segments"] = json.dumps(payload["segments"])
    result = clip_selector._raw_candidate_from_tool_input(payload, stage="Stage1", candidate_index=0)
    assert len(result.segments) == 1
    assert result.segments[0].role == "hook"


# F: "segments" is a broken JSON string -> diagnosable error
def test_raw_candidate_from_tool_input_raises_for_malformed_segments_json_string():
    payload = _valid_candidate_dict()
    payload["segments"] = "not json at all ["
    with pytest.raises(MalformedCandidateError) as exc_info:
        clip_selector._raw_candidate_from_tool_input(payload, stage="Stage2", candidate_index=1)
    message = str(exc_info.value)
    assert "Stage2 candidates[1].segments" in message
    assert "isn't valid JSON" in message


# G: "segments" is a JSON string that decodes to something other than a list -> error
def test_raw_candidate_from_tool_input_raises_when_segments_json_decodes_to_non_list():
    payload = _valid_candidate_dict()
    payload["segments"] = json.dumps({"role": "hook"})
    with pytest.raises(MalformedCandidateError) as exc_info:
        clip_selector._raw_candidate_from_tool_input(payload, stage="Stage1", candidate_index=0)
    assert "segments" in str(exc_info.value)


# H: the candidate item itself is a JSON-encoded object string -- must NOT be
# silently json.loads()'d; the tolerance is scoped to the candidates/segments
# *arrays* only, never to an individual item.
def test_raw_candidate_from_tool_input_does_not_decode_a_json_string_candidate_itself():
    single_candidate_as_json_string = json.dumps(_valid_candidate_dict())
    with pytest.raises(MalformedCandidateError) as exc_info:
        clip_selector._raw_candidate_from_tool_input(
            single_candidate_as_json_string, stage="Stage1", candidate_index=0
        )
    assert "got str" in str(exc_info.value)


class _FakeToolUseBlock:
    def __init__(self, type_, name, input_):
        self.type = type_
        self.name = name
        self.input = input_


class _FakeResponse:
    def __init__(self, content):
        self.content = content


def test_extract_candidates_for_chunk_handles_candidates_as_json_string(monkeypatch):
    """Stage1 end to end, with the Anthropic client fully mocked (no real
    API call): the exact real-machine shape (`candidates` returned as a
    JSON-encoded string) must resolve to real RawClipCandidate objects
    instead of crashing.
    """
    fake_block = _FakeToolUseBlock(
        "tool_use", "submit_chunk_candidates",
        {"candidates": json.dumps([_valid_candidate_dict()])},
    )

    class _FakeMessages:
        def create(self, **kwargs):
            return _FakeResponse([fake_block])

    class _FakeClient:
        messages = _FakeMessages()

    monkeypatch.setattr(clip_selector, "_client", lambda: _FakeClient())

    segments = [_segment(0, start=0.0)]
    result = clip_selector.extract_candidates_for_chunk(segments, "タイトル")

    assert len(result) == 1
    assert isinstance(result[0], RawClipCandidate)
    assert result[0].hook_type == "story"


def test_rank_and_finalize_handles_candidates_as_json_string(monkeypatch):
    """Stage2 end to end, with the Anthropic client fully mocked (no real
    API call): same JSON-string `candidates` shape as Stage1.
    """
    three_candidates = [_valid_candidate_dict() for _ in range(3)]
    fake_block = _FakeToolUseBlock(
        "tool_use", "submit_final_candidates", {"candidates": json.dumps(three_candidates)}
    )

    class _FakeMessages:
        def create(self, **kwargs):
            return _FakeResponse([fake_block])

    class _FakeClient:
        messages = _FakeMessages()

    monkeypatch.setattr(clip_selector, "_client", lambda: _FakeClient())

    transcript = _long_transcript(minutes=1)
    result = clip_selector.rank_and_finalize([_raw_candidate(0, 0)], transcript, "タイトル")

    assert len(result) == 3
    assert all(isinstance(c, RawClipCandidate) for c in result)
