import pytest
from pydantic import ValidationError

from podcast_clipper import cache, clip_selector, config
from podcast_clipper.clip_selector import (
    Stage1CandidateOutput,
    Stage1Output,
    Stage1SegmentOutput,
    Stage2RankingOutput,
)
from podcast_clipper.models import RawClipCandidate, RawUsedSegment, Transcript, TranscriptSegment, TranscriptWord


@pytest.fixture(autouse=True)
def _forbid_real_anthropic_client(monkeypatch):
    """Every test in this module must go through a mocked structured_output
    call -- never a real anthropic.Anthropic(). clip_selector.py no longer
    imports anthropic itself (that lives in structured_output.py, the
    dedicated API boundary module -- see tests/test_structured_output.py
    for its own contract tests), so this reaches through to poison the
    constructor there. Turns an accidental real API call into an
    immediate, loud test failure instead of a silent live call to
    Anthropic.
    """

    def _forbidden(*args, **kwargs):
        raise AssertionError("real anthropic.Anthropic() must not be instantiated in tests")

    monkeypatch.setattr(clip_selector.structured_output.anthropic, "Anthropic", _forbidden)


def _segment(i, start, text=None):
    text = text if text is not None else f"segment {i}"
    return TranscriptSegment(
        id=i, start=start, end=start + 2.0, text=text,
        words=[TranscriptWord(start=start, end=start + 2.0, text=text)],
    )


def _long_transcript(minutes=25):
    segments = [_segment(i, start=i * 20.0) for i in range(int(minutes * 60 / 20))]
    return Transcript(video_id="vid1", language="ja", segments=segments)


def test_build_chunks_covers_whole_transcript_with_overlap(monkeypatch):
    monkeypatch.setattr(config, "CHUNK_MINUTES", 10.0)
    monkeypatch.setattr(config, "CHUNK_OVERLAP_MINUTES", 1.0)
    transcript = _long_transcript(minutes=25)

    chunks = clip_selector._build_chunks(transcript.segments)

    assert len(chunks) >= 3
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


def _raw_candidate(start_id, end_id, role="hook", opening_hook_strength=80, score=80):
    return RawClipCandidate(
        hook_type="story",
        segments=[RawUsedSegment(role=role, start_segment_id=start_id, end_segment_id=end_id)],
        hook_text="h", opening_hook_strength=opening_hook_strength, title="", description="",
        score=score, reasoning="", caveats="",
    )


# --- _filter_local_quality (item G) --------------------------------------


def test_filter_local_quality_keeps_strong_candidates(monkeypatch):
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _long_transcript(minutes=1)
    candidates = [_raw_candidate(0, 2, opening_hook_strength=90)]

    kept = clip_selector._filter_local_quality(candidates, transcript)
    assert len(kept) == 1


def test_filter_local_quality_drops_out_of_range_duration(monkeypatch):
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 20.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 50.0)
    transcript = _long_transcript(minutes=1)
    # single 2-second segment -- far below the 20s hard minimum
    candidates = [_raw_candidate(0, 0, opening_hook_strength=90)]

    kept = clip_selector._filter_local_quality(candidates, transcript)
    assert kept == []


def test_filter_local_quality_drops_weak_opening_hook_strength(monkeypatch):
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    monkeypatch.setattr(config, "MIN_OPENING_HOOK_STRENGTH", 60)
    transcript = _long_transcript(minutes=1)
    candidates = [_raw_candidate(0, 2, opening_hook_strength=10)]

    kept = clip_selector._filter_local_quality(candidates, transcript)
    assert kept == []


def test_filter_local_quality_drops_literal_weak_opening_text(monkeypatch):
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _long_transcript(minutes=1)
    transcript.segments[0].text = "今回はトランプ関税について話していきます"
    candidates = [_raw_candidate(0, 2, opening_hook_strength=90)]

    kept = clip_selector._filter_local_quality(candidates, transcript)
    assert kept == []


def test_filter_local_quality_drops_nonexistent_segment_ids(monkeypatch):
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _long_transcript(minutes=1)
    candidates = [_raw_candidate(9999, 9999, opening_hook_strength=90)]

    kept = clip_selector._filter_local_quality(candidates, transcript)
    assert kept == []


def test_filter_local_quality_forces_first_segment_role_to_hook(monkeypatch):
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _long_transcript(minutes=1)
    candidates = [_raw_candidate(0, 2, role="context", opening_hook_strength=90)]

    kept = clip_selector._filter_local_quality(candidates, transcript)
    assert len(kept) == 1
    assert kept[0].segments[0].role == "hook"


# --- ending completeness: clips must not end mid-utterance ----------------


def _transcript_with_gap(gap_sec, texts):
    segments = []
    t = 0.0
    for i, text in enumerate(texts):
        segments.append(
            TranscriptSegment(
                id=i, start=t, end=t + 2.0, text=text,
                words=[TranscriptWord(start=t, end=t + 2.0, text=text)],
            )
        )
        t += 2.0 + gap_sec
    return Transcript(video_id="vidX", language="ja", segments=segments)


def test_ends_with_terminal_punctuation_true_for_sentence_final_marker():
    assert clip_selector._ends_with_terminal_punctuation("これで終わりです。") is True


def test_ends_with_terminal_punctuation_false_without_a_marker():
    # No dictionary of Japanese sentence-ending words/particles is
    # consulted -- lacking a terminal punctuation mark is treated as
    # "not confidently complete" regardless of what the text actually
    # says, and the gap-based structural check decides the rest.
    assert clip_selector._ends_with_terminal_punctuation("それはこうなので") is False
    assert clip_selector._ends_with_terminal_punctuation("普通の単語") is False


def test_extend_to_natural_ending_leaves_natural_endings_unchanged():
    # C: already ends naturally -- no extension needed, same object back.
    transcript = _transcript_with_gap(0.3, ["これで結論です。", "次のトピックです。"])
    raw = _raw_candidate(0, 0)
    result = clip_selector.extend_to_natural_ending(raw, transcript)
    assert result is raw


def test_extend_to_natural_ending_accepts_when_nothing_to_extend_into():
    # E (variant): no terminal punctuation, but it's the last transcript
    # segment -- nothing to extend into, so it's accepted as-is (never
    # returns None; there is no candidate to reject to).
    transcript = _transcript_with_gap(0.3, ["冒頭の発言です。", "それが起きた理由としては、こういうことが考えられるので"])
    raw = _raw_candidate(0, 1)
    result = clip_selector.extend_to_natural_ending(raw, transcript)
    assert result.segments[-1].end_segment_id == 1


def test_extend_to_natural_ending_accepts_on_real_pause():
    # E: no terminal punctuation, but the next segment is far enough away
    # (a real VAD-detected pause) that it's treated as an intentional
    # stopping point rather than forced across the gap.
    transcript = _transcript_with_gap(
        5.0, ["冒頭の発言です。", "それが起きた理由としては、こういうことが考えられるので", "全く別の話題です。"]
    )
    raw = _raw_candidate(0, 1)
    result = clip_selector.extend_to_natural_ending(raw, transcript)
    assert result.segments[-1].end_segment_id == 1


def test_extend_to_natural_ending_extends_into_continuing_segment():
    # D: no terminal punctuation, and the next segment is a close-in-time
    # continuation -- extends purely on the structural gap signal, with
    # no dictionary lookup on the text at all.
    transcript = _transcript_with_gap(
        0.3, ["冒頭の発言です。", "それが起きた理由としては、こういうことが考えられるので", "そのあたりも確認する必要があります。"]
    )
    raw = _raw_candidate(0, 1)
    result = clip_selector.extend_to_natural_ending(raw, transcript)
    assert result.segments[-1].end_segment_id == 2


def test_extend_to_natural_ending_extends_even_without_a_known_continuation_word():
    # Same as above but the trailing text matches no particular
    # suffix/particle at all -- proves the decision is driven by the gap,
    # not by matching against a fixed word list.
    transcript = _transcript_with_gap(
        0.3, ["冒頭の発言です。", "それについてはこう考えられます", "というのが今回の結論です。"]
    )
    raw = _raw_candidate(0, 1)
    result = clip_selector.extend_to_natural_ending(raw, transcript)
    assert result.segments[-1].end_segment_id == 2


def test_extend_to_natural_ending_stops_at_extension_budget(monkeypatch):
    monkeypatch.setattr(config, "MAX_END_EXTENSION_SEGMENTS", 1)
    # Three unpunctuated segments in a row with short gaps -- extending
    # fully would need 2 hops, but the budget only allows 1.
    transcript = _transcript_with_gap(
        0.3, ["冒頭の発言です。", "それについて一つ目の話ですが", "さらに二つ目の話ですが", "これで結論です。"]
    )
    raw = _raw_candidate(0, 1)
    result = clip_selector.extend_to_natural_ending(raw, transcript)
    assert result.segments[-1].end_segment_id == 2  # only one hop taken, budget exhausted


def test_filter_local_quality_rejects_candidate_when_natural_ending_exceeds_hard_max(monkeypatch):
    # F: reaching a natural ending would exceed DURATION_HARD_MAX_SEC --
    # the candidate is rejected rather than cut off mid-utterance to fit
    # (Stage1 has other candidates to fall back on here).
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 5.0)
    transcript = _transcript_with_gap(
        0.3, ["冒頭の発言です。", "それが起きた理由としては、こういうことが考えられるので", "そのあたりも確認する必要があります。"]
    )
    raw = _raw_candidate(0, 1, opening_hook_strength=90)

    kept = clip_selector._filter_local_quality([raw], transcript)
    assert kept == []


def test_filter_local_quality_keeps_candidate_when_extension_stays_within_hard_max(monkeypatch):
    # A: fresh (non-cached) candidate is extended.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _transcript_with_gap(
        0.3, ["冒頭の発言です。", "それが起きた理由としては、こういうことが考えられるので", "そのあたりも確認する必要があります。"]
    )
    raw = _raw_candidate(0, 1, opening_hook_strength=90)

    kept = clip_selector._filter_local_quality([raw], transcript)
    assert len(kept) == 1
    assert kept[0].segments[-1].end_segment_id == 2


# --- B (most important): a cache hit must go through the same correction -


def test_select_candidates_applies_ending_correction_to_cached_candidates(monkeypatch):
    """The exact scenario that slipped through before this fix: a Stage2
    result cached under the *old* (pre-extension) behavior -- its last
    segment ends mid-utterance, matching the real clip_c2 incident -- must
    still come out corrected on a cache hit, without any Claude API call,
    without discarding/recomputing the cache. (C: cache candidate, stays
    within hard bounds after extension -> accepted.)
    """
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _transcript_with_gap(
        0.3, ["冒頭の発言です。", "それが起きた理由としては、こういうことが考えられるので", "そのあたりも確認する必要があります。"]
    )
    stale_cached_candidate = _raw_candidate(0, 1, opening_hook_strength=90)
    cache.save_stage2(transcript.video_id, [stale_cached_candidate] * 3)

    result = clip_selector.select_candidates(transcript, "タイトル")

    assert len(result) == 3
    for c in result:
        assert c.segments[-1].end_segment_id == 2  # extended past the stale mid-utterance cutoff


def test_select_candidates_raises_when_all_cached_candidates_exceed_hard_max_after_extension(monkeypatch):
    """D: a cached candidate is never kept just because there's no
    substitute -- reaching a natural ending past DURATION_HARD_MAX_SEC
    makes it ineligible exactly as it would during fresh selection, never
    truncated mid-utterance to fit. All 3 cached candidates are equally
    over-length here, so none remain eligible and select_candidates must
    raise rather than silently return them anyway or call the API again.
    """
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 5.0)
    transcript = _transcript_with_gap(
        0.3, ["冒頭の発言です。", "それが起きた理由としては、こういうことが考えられるので", "そのあたりも確認する必要があります。"]
    )
    stale_cached_candidate = _raw_candidate(0, 1, opening_hook_strength=90)
    cache.save_stage2(transcript.video_id, [stale_cached_candidate] * 3)

    with pytest.raises(RuntimeError, match="有効な"):
        clip_selector.select_candidates(transcript, "タイトル")


def test_select_candidates_raises_when_only_some_cached_candidates_remain_eligible(monkeypatch):
    """F: 2 of 3 cached candidates stay within hard bounds after
    extension, 1 doesn't -- the absolute condition is exactly 3
    candidates, so select_candidates must not silently return the 2
    still-valid ones; it must raise instead of shortchanging the UI.
    """
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 5.0)
    transcript = _transcript_with_gap(
        0.3, ["冒頭の発言です。", "それが起きた理由としては、こういうことが考えられるので", "そのあたりも確認する必要があります。"]
    )
    good = _raw_candidate(0, 0, opening_hook_strength=90)  # ends naturally, short -> stays valid
    bad = _raw_candidate(0, 1, opening_hook_strength=90)  # extends past segment 2 -> exceeds hard max
    cache.save_stage2(transcript.video_id, [good, good, bad])

    with pytest.raises(RuntimeError, match="有効な"):
        clip_selector.select_candidates(transcript, "タイトル")


def test_select_candidates_applies_the_same_duration_rule_on_fresh_and_cached_paths(monkeypatch):
    """Proves fresh and cache share one rule, not two: with identical
    transcript/bounds, a fresh Stage2 selection and a cache hit both
    extend to the same natural ending and both stay eligible under the
    same hard-duration check (fresh via _filter_local_quality before
    Stage2, cache via _finalize_candidates after).
    """
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _transcript_with_gap(
        0.3, ["冒頭の発言です。", "それが起きた理由としては、こういうことが考えられるので", "そのあたりも確認する必要があります。"]
    )
    candidate = _raw_candidate(0, 1, opening_hook_strength=90)

    monkeypatch.setattr(clip_selector, "run_stage1", lambda *a, **k: [candidate] * 4)
    monkeypatch.setattr(clip_selector, "rank_candidates", lambda id_map, t, title: list(id_map.keys()))
    fresh_result = clip_selector.select_candidates(transcript, "タイトル")

    cache_transcript = Transcript(
        video_id=transcript.video_id + "-cache", language="ja", segments=transcript.segments
    )
    cache.save_stage2(cache_transcript.video_id, [candidate] * 3)
    cached_result = clip_selector.select_candidates(cache_transcript, "タイトル")

    assert len(fresh_result) == 3
    assert len(cached_result) == 3
    assert all(c.segments[-1].end_segment_id == 2 for c in fresh_result)
    assert all(c.segments[-1].end_segment_id == 2 for c in cached_result)


# --- select_candidates: no automatic retry (item H) ----------------------


def test_select_candidates_raises_without_calling_stage2_when_too_few_filtered(monkeypatch):
    transcript = _long_transcript(minutes=1)
    monkeypatch.setattr(clip_selector, "run_stage1", lambda *a, **k: [_raw_candidate(0, 0)])  # too short -> filtered out
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 20.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 50.0)
    monkeypatch.setattr(
        clip_selector, "rank_candidates",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Stage2 must not be called")),
    )

    with pytest.raises(RuntimeError, match="ローカル品質フィルタ"):
        clip_selector.select_candidates(transcript, "タイトル")


def test_select_candidates_raises_when_stage2_returns_too_few_ids(monkeypatch):
    transcript = _long_transcript(minutes=1)
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    candidates = [_raw_candidate(0, 2), _raw_candidate(0, 2), _raw_candidate(0, 2)]
    monkeypatch.setattr(clip_selector, "run_stage1", lambda *a, **k: candidates)
    monkeypatch.setattr(clip_selector, "rank_candidates", lambda id_map, t, title: list(id_map.keys())[:1])

    with pytest.raises(RuntimeError, match="Stage2ランキング"):
        clip_selector.select_candidates(transcript, "タイトル")


def test_select_candidates_happy_path(monkeypatch):
    transcript = _long_transcript(minutes=1)
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    candidates = [_raw_candidate(0, 2, score=s) for s in (10, 20, 30, 40)]
    monkeypatch.setattr(clip_selector, "run_stage1", lambda *a, **k: candidates)
    monkeypatch.setattr(clip_selector, "rank_candidates", lambda id_map, t, title: list(id_map.keys()))

    result = clip_selector.select_candidates(transcript, "タイトル")
    assert len(result) == config.NUM_CANDIDATES


def test_select_candidates_caches_and_skips_recompute(monkeypatch):
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _long_transcript(minutes=1)
    cache.save_stage2("vid1", [_raw_candidate(0, 0)] * 3)

    monkeypatch.setattr(
        clip_selector, "run_stage1", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run"))
    )

    result = clip_selector.select_candidates(transcript, "タイトル")
    assert len(result) == 3


# --- Structured Outputs models (schema is minimal: AI only ever produces
# --- segment_id ranges + a few scores, never display text) --------------


def test_stage1_candidate_output_field_set_excludes_ai_authored_text():
    # N: Claude structurally cannot produce hook_text/title/description/
    # reasoning/caveats any more -- they're not even fields on the model.
    fields = set(Stage1CandidateOutput.model_fields)
    assert fields == {"hook_type", "segments", "opening_hook_strength", "score"}
    assert "hook_text" not in fields
    assert "title" not in fields
    assert "description" not in fields
    assert "reasoning" not in fields
    assert "caveats" not in fields


def test_stage2_ranking_output_field_set_is_just_ranked_ids():
    # D/E: Stage2 output is nothing but an ordered list of candidate ids.
    assert set(Stage2RankingOutput.model_fields) == {"ranked_candidate_ids"}


def _valid_segment_kwargs():
    return {"role": "hook", "start_segment_id": 0, "end_segment_id": 0}


def _valid_candidate_kwargs():
    return {
        "hook_type": "story", "segments": [_valid_segment_kwargs()],
        "opening_hook_strength": 80, "score": 80,
    }


def test_stage1_output_accepts_zero_to_three_candidates():
    assert Stage1Output(candidates=[]).candidates == []
    for n in (1, 2, 3):
        out = Stage1Output(candidates=[Stage1CandidateOutput(**_valid_candidate_kwargs()) for _ in range(n)])
        assert len(out.candidates) == n
    with pytest.raises(ValidationError):
        Stage1Output(candidates=[Stage1CandidateOutput(**_valid_candidate_kwargs()) for _ in range(4)])


def test_stage1_candidate_output_segments_length_bounds():
    for n in (1, 2, 3):
        kwargs = _valid_candidate_kwargs()
        kwargs["segments"] = [_valid_segment_kwargs() for _ in range(n)]
        Stage1CandidateOutput(**kwargs)
    for n in (0, 4):
        kwargs = _valid_candidate_kwargs()
        kwargs["segments"] = [_valid_segment_kwargs() for _ in range(n)]
        with pytest.raises(ValidationError):
            Stage1CandidateOutput(**kwargs)


def test_stage1_candidate_output_opening_hook_strength_and_score_bounds():
    for field in ("opening_hook_strength", "score"):
        for value in (0, 100):
            kwargs = _valid_candidate_kwargs()
            kwargs[field] = value
            Stage1CandidateOutput(**kwargs)
        for value in (-1, 101):
            kwargs = _valid_candidate_kwargs()
            kwargs[field] = value
            with pytest.raises(ValidationError):
                Stage1CandidateOutput(**kwargs)


def test_stage1_candidate_output_rejects_invalid_hook_type():
    kwargs = _valid_candidate_kwargs()
    kwargs["hook_type"] = "not_a_real_hook_type"
    with pytest.raises(ValidationError):
        Stage1CandidateOutput(**kwargs)


def test_stage1_segment_output_rejects_invalid_role():
    with pytest.raises(ValidationError):
        Stage1SegmentOutput(role="not_a_real_role", start_segment_id=0, end_segment_id=0)


def test_stage1_segment_output_rejects_wrongly_typed_segment_id():
    with pytest.raises(ValidationError):
        Stage1SegmentOutput(role="hook", start_segment_id=["not", "an", "int"], end_segment_id=0)


def test_stage1_candidate_output_rejects_unknown_fields():
    # extra="forbid" -> additionalProperties: false in the schema sent to
    # Claude, and the same strictness applies locally.
    kwargs = _valid_candidate_kwargs()
    kwargs["hook_text"] = "should not be accepted"
    with pytest.raises(ValidationError):
        Stage1CandidateOutput(**kwargs)


# --- _deterministic_hook_text (item M) ------------------------------------


def test_deterministic_hook_text_uses_real_transcript_text():
    segments = [_segment(0, start=0.0, text="これは実際の発言です")]
    assert clip_selector._deterministic_hook_text(0, segments) == "これは実際の発言です"


def test_deterministic_hook_text_truncates_by_character_count_only(monkeypatch):
    monkeypatch.setattr(config, "HOOK_TEXT_MAX_CHARS", 5)
    segments = [_segment(0, start=0.0, text="abcdefghij")]
    result = clip_selector._deterministic_hook_text(0, segments)
    assert result == "abcde…"


# --- extract_candidates_for_chunk / rank_candidates: real wiring ---------
# (the Structured Outputs API boundary itself -- stop_reason handling,
# max_retries=0, request-body contract -- is tested in
# tests/test_structured_output.py; here we only verify clip_selector.py's
# own use of that boundary: prompt/input construction and output
# conversion)


def test_extract_candidates_for_chunk_converts_structured_output(monkeypatch):
    output = Stage1Output(candidates=[Stage1CandidateOutput(**_valid_candidate_kwargs())])
    monkeypatch.setattr(
        clip_selector.structured_output, "call",
        lambda schema_model, **kwargs: output,
    )

    segments = [_segment(0, start=0.0, text="強い発言です")]
    result = clip_selector.extract_candidates_for_chunk(segments, "タイトル")

    assert len(result) == 1
    assert isinstance(result[0], RawClipCandidate)
    assert result[0].hook_type == "story"
    assert result[0].hook_text == "強い発言です"
    assert result[0].title == ""
    assert result[0].description == ""
    assert result[0].reasoning == ""
    assert result[0].caveats == ""


def test_rank_candidates_returns_ranked_known_ids_only(monkeypatch):
    transcript = _long_transcript(minutes=1)
    id_map = {"s1_c000": _raw_candidate(0, 2), "s1_c001": _raw_candidate(0, 2)}
    output = Stage2RankingOutput(ranked_candidate_ids=["s1_c001", "unknown_id", "s1_c000", "s1_c001"])
    monkeypatch.setattr(clip_selector.structured_output, "call", lambda schema_model, **kwargs: output)

    ranked = clip_selector.rank_candidates(id_map, transcript, "タイトル")
    assert ranked == ["s1_c001", "s1_c000"]  # unknown id dropped, duplicate id de-duped, order preserved


def test_rank_candidates_does_not_send_full_transcript(monkeypatch):
    # F: Stage2 only sees a compact summary of the filtered candidates --
    # an unreferenced transcript segment's distinctive text must never
    # appear in what gets sent to the API.
    transcript = _long_transcript(minutes=5)
    transcript.segments[-1].text = "この文言はどの候補にも含まれない特徴的な発言マーカーXYZ123"
    id_map = {"s1_c000": _raw_candidate(0, 2)}

    captured = {}

    def _spy(schema_model, *, stage, system_prompt, user_content, max_tokens):
        captured["user_content"] = user_content
        return Stage2RankingOutput(ranked_candidate_ids=["s1_c000"])

    monkeypatch.setattr(clip_selector.structured_output, "call", _spy)
    clip_selector.rank_candidates(id_map, transcript, "タイトル")

    assert "マーカーXYZ123" not in captured["user_content"]


def test_select_candidates_calls_stage2_at_most_once(monkeypatch):
    transcript = _long_transcript(minutes=1)
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    candidates = [_raw_candidate(0, 2) for _ in range(3)]
    monkeypatch.setattr(clip_selector, "run_stage1", lambda *a, **k: candidates)

    call_count = {"n": 0}

    def _fake_rank_candidates(id_map, t, title):
        call_count["n"] += 1
        return list(id_map.keys())

    monkeypatch.setattr(clip_selector, "rank_candidates", _fake_rank_candidates)
    clip_selector.select_candidates(transcript, "タイトル")
    assert call_count["n"] == 1
