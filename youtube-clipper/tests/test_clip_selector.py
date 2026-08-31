import pytest
from pydantic import ValidationError

from podcast_clipper import boundary, cache, clip_selector, config
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


def test_min_opening_hook_strength_default_is_80():
    """Real-machine validation showed the old default of 60 let through
    explanatory/abstract openings that read as a weak Shorts hook (see
    prompts/extract_candidates.md's 70-79 band). Raised to 80 so only
    openings scored as "clearly makes you want to keep watching" or
    stronger clear the local filter.
    """
    assert config.MIN_OPENING_HOOK_STRENGTH == 80


def test_filter_local_quality_drops_opening_hook_strength_of_79(monkeypatch):
    """79 sits in the prompt's 70-79 ("explanatory/abstract, weak hook")
    band and must be rejected under the default threshold."""
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _long_transcript(minutes=1)
    candidates = [_raw_candidate(0, 2, opening_hook_strength=79)]

    kept = clip_selector._filter_local_quality(candidates, transcript)
    assert kept == []


def test_filter_local_quality_passes_opening_hook_strength_of_80(monkeypatch):
    """80 is the minimum score the prompt calls "clearly makes you want to
    keep watching" and must clear the local filter when other conditions
    (duration, natural opening text) are satisfied."""
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _long_transcript(minutes=1)
    candidates = [_raw_candidate(0, 2, opening_hook_strength=80)]

    kept = clip_selector._filter_local_quality(candidates, transcript)
    assert len(kept) == 1


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


# --- prompt content: strengthened hook-scoring rubric ---------------------


def _extract_candidates_prompt_text():
    return (clip_selector._PROMPTS_DIR / "extract_candidates.md").read_text(encoding="utf-8")


def _rank_and_finalize_prompt_text():
    return (clip_selector._PROMPTS_DIR / "rank_and_finalize.md").read_text(encoding="utf-8")


def test_extract_candidates_prompt_has_90_80_70_scoring_bands():
    text = _extract_candidates_prompt_text()
    assert "90" in text and "100" in text
    assert "80" in text and "89" in text
    assert "70" in text and "79" in text


def test_extract_candidates_prompt_does_not_rate_abstract_explanation_as_strong_hook():
    """Pins that the real-machine-observed weak opening ("弱点を直すとか改善す
    ると次の弱点というのが生まれてくるので...") is explicitly called out as an
    example that must NOT be scored as a strong hook, so this specific
    real-world failure can't silently regress if the prompt is edited
    again later.
    """
    text = _extract_candidates_prompt_text()
    assert "弱点を直すとか改善すると" in text
    assert "だと思っています" in text or "と思っています" in text


def test_rank_and_finalize_prompt_independently_evaluates_stage1_hook_score():
    """Stage2 must not blindly trust Stage1's opening_hook_strength -- it
    has to re-evaluate the actual first utterance itself."""
    text = _rank_and_finalize_prompt_text()
    assert "鵜呑みにしないでください" in text or "鵜呑み" in text


# --- prompt content: Stage1 widened to recall-oriented search (max 6) -----


def test_extract_candidates_prompt_allows_up_to_six_candidates():
    text = _extract_candidates_prompt_text()
    assert "最大6件" in text
    assert "最大3件" not in text


def test_extract_candidates_prompt_requires_scanning_whole_chunk():
    """Stage1 must not stop after finding candidates early in the chunk --
    it has to read to the end before finalizing its candidate list, so a
    stronger later utterance isn't missed."""
    text = _extract_candidates_prompt_text()
    assert "冒頭から末尾まで全体を読んで" in text


def test_extract_candidates_prompt_forbids_using_up_slots_on_the_first_half():
    text = _extract_candidates_prompt_text()
    assert "前半で見つかった強そうな発話だけで候補枠を使い切り" in text
    assert "後半" in text


def test_extract_candidates_prompt_forbids_padding_weak_candidates_to_fill_six():
    text = _extract_candidates_prompt_text()
    assert "件数を埋める必要はありません" in text


def test_extract_candidates_prompt_forbids_near_duplicate_candidates():
    text = _extract_candidates_prompt_text()
    assert "複数枠に並べないでください" in text


def test_extract_candidates_prompt_states_stage1_is_recall_not_final_selection():
    """Documents the Stage1/Stage2 role split: Stage1 casts a wide net,
    Stage2 (seeing the pooled candidates from every chunk) picks the final
    best-3."""
    text = _extract_candidates_prompt_text()
    assert "最終ベスト3を決める係ではありません" in text or "最終的に使う3本を選び切る係ではありません" in text


def test_rank_and_finalize_prompt_states_it_picks_the_final_best_three():
    text = _rank_and_finalize_prompt_text()
    assert "最終的に採用すべきベスト3" in text


# --- prompt content: junction safety (cut-point naturalness) --------------


def test_extract_candidates_prompt_documents_junction_safety():
    text = _extract_candidates_prompt_text()
    assert "カット接続の自然さ" in text
    assert "すべての隣接ペア" in text


def test_extract_candidates_prompt_includes_bad_junction_example():
    text = _extract_candidates_prompt_text()
    assert "車を冷やしますっていうのであれば" in text
    assert "連続周回をする場合は" in text


def test_extract_candidates_prompt_requires_extending_before_non_chronological_jump():
    text = _extract_candidates_prompt_text()
    assert "いきなり別のsegmentへ飛ばないこと" in text


def test_extract_candidates_prompt_documents_limited_exact_repeat():
    text = _extract_candidates_prompt_text()
    assert "最大2回まで" in text
    assert "3回以上は禁止" in text


def test_rank_and_finalize_prompt_documents_junction_safety():
    text = _rank_and_finalize_prompt_text()
    assert "カット接続の自然さ" in text
    assert "連続周回をする場合は" in text


# --- prompt content: start_anchor_text trim + segment reordering ----------


def test_extract_candidates_prompt_documents_start_anchor_text():
    text = _extract_candidates_prompt_text()
    assert "start_anchor_text" in text
    assert "完全一致" in text


def test_extract_candidates_prompt_forbids_mid_word_anchor_starts():
    text = _extract_candidates_prompt_text()
    assert "単語の途中" in text
    assert "word境界に一致する必要がある" in text


def test_extract_candidates_prompt_allows_reordering_segments():
    """The most important change this round: segments no longer need to
    be in transcript-chronological order."""
    text = _extract_candidates_prompt_text()
    assert "時系列順である必要はありません" in text or "時系列順である必要はない" in text


def test_extract_candidates_prompt_reorder_forbids_fabrication():
    text = _extract_candidates_prompt_text()
    assert "発話を作文しない" in text
    assert "因果関係を逆転させない" in text


def test_extract_candidates_prompt_scores_hook_strength_post_trim_and_reorder():
    """opening_hook_strength must be scored against what actually plays
    first after anchor trim / reordering, not the raw untrimmed segment or
    the pre-reorder chronological first segment (item 16)."""
    text = _extract_candidates_prompt_text()
    assert "トリム後のテキスト" in text
    assert "並び替え後に実際に最初に来るsegment" in text


def test_extract_candidates_prompt_includes_real_machine_examples():
    text = _extract_candidates_prompt_text()
    assert "これも私の愛車である86はスープラをベースに作られています" in text
    assert "ZN6-86であったり" in text
    assert "冷却効率を上げるために重量を増やすというのはアンチパターンになる" in text


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


def test_extend_to_natural_ending_uses_lenient_gap_for_confirmed_continuation(monkeypatch):
    # C: a confirmed continuation ending ("...ので") must not be accepted
    # as complete just because a pause exceeds the base 0.8s gap threshold
    # -- but the new, still-bounded, lenient threshold for *confirmed*
    # continuation text should bridge it rather than stopping early.
    monkeypatch.setattr(config, "END_EXTENSION_MAX_GAP_SEC", 0.8)
    monkeypatch.setattr(config, "END_EXTENSION_CONTINUATION_MAX_GAP_SEC", 1.5)
    transcript = _transcript_with_gap(
        1.2, ["冒頭の発言です。", "整備士に出会うことが大切じゃないかなと思うので", "そのあたりも確認する必要があります。"]
    )
    raw = _raw_candidate(0, 1)
    result = clip_selector.extend_to_natural_ending(raw, transcript)
    assert result.segments[-1].end_segment_id == 2


def test_extend_to_natural_ending_ambiguous_text_still_uses_base_gap(monkeypatch):
    # The lenient threshold applies only to text matching a confirmed
    # continuation marker -- an ambiguous ending (no punctuation, no
    # continuation suffix) at the same 1.2s gap must still stop at the
    # base 0.8s threshold, exactly as before this change.
    monkeypatch.setattr(config, "END_EXTENSION_MAX_GAP_SEC", 0.8)
    monkeypatch.setattr(config, "END_EXTENSION_CONTINUATION_MAX_GAP_SEC", 1.5)
    transcript = _transcript_with_gap(
        1.2, ["冒頭の発言です。", "それについてはこう考えられます", "というのが今回の結論です。"]
    )
    raw = _raw_candidate(0, 1)
    result = clip_selector.extend_to_natural_ending(raw, transcript)
    assert result.segments[-1].end_segment_id == 1


# --- has_confident_natural_ending: a pause alone must never mean complete -


def test_has_confident_natural_ending_true_for_terminal_punctuation():
    transcript = _transcript_with_gap(0.3, ["これで結論です。", "次のトピックです。"])
    raw = _raw_candidate(0, 0)
    assert clip_selector.has_confident_natural_ending(raw, transcript) is True


def test_has_confident_natural_ending_false_for_confirmed_continuation_suffix():
    transcript = _transcript_with_gap(
        0.3, ["冒頭の発言です。", "整備士に出会うことが大切じゃないかなと思うので"]
    )
    raw = _raw_candidate(0, 1)
    assert clip_selector.has_confident_natural_ending(raw, transcript) is False


def test_has_confident_natural_ending_true_for_ambiguous_non_continuation_text():
    # No terminal punctuation, but also not a confirmed continuation
    # marker -- accepted as a natural stopping point (rule 4).
    transcript = _transcript_with_gap(0.3, ["冒頭の発言です。", "普通の単語"])
    raw = _raw_candidate(0, 1)
    assert clip_selector.has_confident_natural_ending(raw, transcript) is True


def test_has_confident_natural_ending_false_when_no_next_segment_to_bridge():
    # Item M test 3: "...ので" + no next segment at all must never be
    # treated as complete just because there's nothing left to extend
    # into -- the candidate must be flagged ineligible, not accepted.
    transcript = _transcript_with_gap(
        0.3, ["冒頭の発言です。", "整備士に出会うことが大切じゃないかなと思うので"]
    )
    raw = _raw_candidate(0, 1)
    extended = clip_selector.extend_to_natural_ending(raw, transcript)
    assert clip_selector.has_confident_natural_ending(extended, transcript) is False


def test_filter_local_quality_drops_confirmed_continuation_with_no_viable_extension():
    # The same scenario wired through the actual pre-Stage2 filter.
    transcript = _transcript_with_gap(
        0.3, ["冒頭の発言です。", "整備士に出会うことが大切じゃないかなと思うので"]
    )
    raw = _raw_candidate(0, 1, opening_hook_strength=90)
    kept = clip_selector._filter_local_quality([raw], transcript)
    assert kept == []


def test_select_candidates_raises_when_cached_candidates_are_confirmed_continuation_with_no_extension(monkeypatch):
    # Item M test 7: the identical "pause != complete" rule must apply to
    # the cache-hit path (_finalize_candidates), not just the fresh path.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _transcript_with_gap(
        0.3, ["冒頭の発言です。", "整備士に出会うことが大切じゃないかなと思うので"]
    )
    stale_cached_candidate = _raw_candidate(0, 1, opening_hook_strength=90)
    cache.save_stage2(transcript.video_id, [stale_cached_candidate] * 3)

    with pytest.raises(RuntimeError, match="有効な"):
        clip_selector.select_candidates(transcript, "タイトル")


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


# --- overlap safety net: segments no longer required to be chronological -


def _reorder_transcript():
    # 3 segments, chronological order 0 -> 1 -> 2, each ending without
    # terminal punctuation (so extension is tempted to keep walking).
    return Transcript(
        video_id="vidR",
        language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=2.0, text="真冬のサーキットで走ります",
                words=[TranscriptWord(start=0.0, end=2.0, text="真冬のサーキットで走ります")],
            ),
            TranscriptSegment(
                id=1, start=2.3, end=4.0, text="車を冷やしますというのであれば",
                words=[TranscriptWord(start=2.3, end=4.0, text="車を冷やしますというのであれば")],
            ),
            TranscriptSegment(
                id=2, start=4.3, end=6.0, text="重量を増やすのはアンチパターンになる",
                words=[TranscriptWord(start=4.3, end=6.0, text="重量を増やすのはアンチパターンになる")],
            ),
        ],
    )


def test_has_overlapping_segments_true_for_overlapping_ranges():
    transcript = _reorder_transcript()
    raw = RawClipCandidate(
        hook_type="story",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=1, end_segment_id=2),
            RawUsedSegment(role="context", start_segment_id=0, end_segment_id=1),
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )
    assert clip_selector._has_overlapping_segments(raw, transcript) is True


def test_has_overlapping_segments_false_for_disjoint_reordered_ranges():
    transcript = _reorder_transcript()
    raw = RawClipCandidate(
        hook_type="story",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=2, end_segment_id=2),
            RawUsedSegment(role="context", start_segment_id=0, end_segment_id=1),
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )
    assert clip_selector._has_overlapping_segments(raw, transcript) is False


def test_extend_to_natural_ending_does_not_walk_into_another_segments_range():
    """A reordered candidate (hook = chronologically-later segment 2,
    context = chronologically-earlier segments 0-1) has no terminal
    punctuation anywhere, so the last-played segment (context, ending at
    segment 1) would normally keep extending forward -- but segment 2 is
    already used by this same candidate's hook. Extension must stop
    before segment 1 -> 2, exactly as if segment 1 were the end of the
    transcript, rather than reusing content segment 2 already plays.
    """
    transcript = _reorder_transcript()
    raw = RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=2, end_segment_id=2),
            RawUsedSegment(role="context", start_segment_id=0, end_segment_id=1),
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )
    extended = clip_selector.extend_to_natural_ending(raw, transcript)
    assert extended.segments[-1].end_segment_id == 1
    assert clip_selector._has_overlapping_segments(extended, transcript) is False


def _overlap_transcript_with_clean_endings():
    # Every segment ends with terminal punctuation, so extend_to_natural_
    # ending never kicks in -- isolates the overlap check itself as the
    # reason a candidate is dropped, independent of ending-completeness.
    return Transcript(
        video_id="vidO",
        language="ja",
        segments=[
            TranscriptSegment(
                id=i, start=i * 2.0, end=i * 2.0 + 1.5, text=f"文{i}です。",
                words=[TranscriptWord(start=i * 2.0, end=i * 2.0 + 1.5, text=f"文{i}です。")],
            )
            for i in range(3)
        ],
    )


def test_filter_local_quality_drops_candidates_with_overlapping_segments(monkeypatch):
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _overlap_transcript_with_clean_endings()
    raw = RawClipCandidate(
        hook_type="story",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=1),
            RawUsedSegment(role="context", start_segment_id=1, end_segment_id=2),
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )
    kept = clip_selector._filter_local_quality([raw], transcript)
    assert kept == []


def test_finalize_candidates_drops_overlapping_segments(monkeypatch):
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _overlap_transcript_with_clean_endings()
    raw = RawClipCandidate(
        hook_type="story",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=1),
            RawUsedSegment(role="context", start_segment_id=1, end_segment_id=2),
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )
    # Padded with valid, non-overlapping candidates so the overlap itself
    # (not NUM_CANDIDATES underflow) is what's being isolated.
    good = [_raw_candidate(0, 0) for _ in range(config.NUM_CANDIDATES)]
    finalized = clip_selector.finalize_candidates([raw] + good, transcript)
    assert all(not clip_selector._has_overlapping_segments(c, transcript) for c in finalized)
    assert len(finalized) == config.NUM_CANDIDATES


# --- junction safety net: cut points between segments must read naturally
# (real-machine feedback: "車を冷やしますっていうのであれば" hard-cut into an
# unrelated "連続周回をする場合は" -- every check up to this point (per-
# segment, hook, final ending) passed, but the A->B cut itself was broken
# Japanese) ------------------------------------------------------------


def _junction_transcript():
    # Reconstructs the real-machine candidate 3 scenario: a chronological
    # run (0: context intro, 1: unfinished "...のであれば", 2: the real
    # conclusion that follows it) plus an unrelated, distant segment (3)
    # that must never be spliced onto segment 1's unfinished ending.
    return Transcript(
        video_id="vidJ",
        language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=2.0, text="真冬のサーキットで2、3周しかアタックをしません",
                words=[TranscriptWord(start=0.0, end=2.0, text="真冬のサーキットで2、3周しかアタックをしません")],
            ),
            TranscriptSegment(
                id=1, start=2.3, end=4.0, text="車を冷やしますっていうのであれば",
                words=[TranscriptWord(start=2.3, end=4.0, text="車を冷やしますっていうのであれば")],
            ),
            TranscriptSegment(
                id=2, start=4.3, end=6.0,
                text="冷却効率を上げるために重量を増やすというのはアンチパターンになるかなと思います",
                words=[
                    TranscriptWord(start=4.3, end=4.9, text="冷却効率を"),
                    TranscriptWord(start=4.9, end=5.3, text="上げるために"),
                    TranscriptWord(start=5.3, end=6.0, text="重量を増やすというのはアンチパターンになるかなと思います"),
                ],
            ),
            TranscriptSegment(
                id=3, start=20.0, end=22.0, text="連続周回をする場合は違う話になります",
                words=[TranscriptWord(start=20.0, end=22.0, text="連続周回をする場合は違う話になります")],
            ),
        ],
    )


def _junction_candidate(second_start_id, second_end_id, second_role="context"):
    return RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=1),
            RawUsedSegment(role=second_role, start_segment_id=second_start_id, end_segment_id=second_end_id),
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )


def test_validate_candidate_junctions_A_rejects_bad_non_chronological_junction():
    # A: unfinished "...のであれば" hard-cut into an unrelated condition.
    transcript = _junction_transcript()
    bad = _junction_candidate(3, 3)
    assert clip_selector._validate_candidate_junctions(bad, transcript) is False


def test_validate_candidate_junctions_B_allows_chronological_continuation():
    # B: same unfinished ending, but the next segment is literally the
    # real transcript continuation (segment 2 follows segment 1).
    transcript = _junction_transcript()
    good = _junction_candidate(2, 2)
    assert clip_selector._validate_candidate_junctions(good, transcript) is True


def test_validate_candidate_junctions_C_allows_complete_then_independent_jump():
    # C: a non-chronological jump is fine when the first segment is
    # complete and the second reads independently.
    transcript = _junction_transcript()
    raw = RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=2, end_segment_id=2),
            RawUsedSegment(role="context", start_segment_id=0, end_segment_id=0),
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )
    assert clip_selector._validate_candidate_junctions(raw, transcript) is True


def test_validate_candidate_junctions_D_rejects_context_dependent_hook():
    # D: candidate 2 regression -- "これのクラッチ交換の際に..." never
    # establishes what "これ" refers to within this candidate.
    transcript = Transcript(
        video_id="vidD",
        language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=3.0,
                text="これのクラッチ交換の際にメタルクラッチを入れるとミッションが壊れやすくなるっていうのはよく言われてます",
                words=[TranscriptWord(
                    start=0.0, end=3.0,
                    text="これのクラッチ交換の際にメタルクラッチを入れるとミッションが壊れやすくなるっていうのはよく言われてます",
                )],
            ),
        ],
    )
    raw = RawClipCandidate(
        hook_type="surprising_fact",
        segments=[RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0)],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )
    assert clip_selector._validate_candidate_junctions(raw, transcript) is False


def test_validate_candidate_junctions_E_allows_anchor_trimmed_independent_opening():
    # E: the same underlying segment, but start_anchor_text drops "これの"
    # and starts at the car models actually named -- independent, allowed.
    transcript = Transcript(
        video_id="vidE",
        language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=5.0,
                text="よくある話が私も乗っているZN6-86であったりあとはBRZあとGR86メタルクラッチを入れるとミッションが壊れやすくなるっていうのはよく言われてます",
                words=[
                    TranscriptWord(start=0.0, end=0.4, text="よくある話が"),
                    TranscriptWord(start=0.4, end=0.8, text="私も乗っている"),
                    TranscriptWord(start=0.8, end=1.2, text="ZN6-86であったり"),
                    TranscriptWord(
                        start=1.2, end=5.0,
                        text="あとはBRZあとGR86メタルクラッチを入れるとミッションが壊れやすくなるっていうのはよく言われてます",
                    ),
                ],
            ),
        ],
    )
    raw = RawClipCandidate(
        hook_type="surprising_fact",
        segments=[
            RawUsedSegment(
                role="hook", start_segment_id=0, end_segment_id=0,
                start_anchor_text="ZN6-86であったり",
            )
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )
    assert clip_selector._validate_candidate_junctions(raw, transcript) is True


def test_validate_candidate_junctions_F_candidate1_anchor_regression():
    # F: candidate 1 regression -- "これも私の愛車である86は..." trimmed via
    # start_anchor_text="86は" resolves to an independent opening.
    transcript = Transcript(
        video_id="vidF",
        language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=3.0,
                text="これも私の愛車である86はスープラをベースに作られています",
                words=[
                    TranscriptWord(start=0.0, end=0.3, text="これも"),
                    TranscriptWord(start=0.3, end=0.6, text="私の"),
                    TranscriptWord(start=0.6, end=0.9, text="愛車である"),
                    TranscriptWord(start=0.9, end=1.2, text="86は"),
                    TranscriptWord(start=1.2, end=3.0, text="スープラをベースに作られています"),
                ],
            ),
        ],
    )
    raw = RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0, start_anchor_text="86は")
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )
    assert clip_selector._validate_candidate_junctions(raw, transcript) is True
    resolved = boundary.resolve_candidate(raw, transcript, candidate_id="c1")
    assert resolved.segments[0].text.startswith("86は")


def test_confirmed_continuation_catches_kedomo_ending():
    # G: root cause of the real-machine ending bug -- the suffix list
    # didn't previously recognize "けれないんですけども"-style casual
    # continuations (only "けど"/"けれども", not "けども").
    assert clip_selector._ends_with_confirmed_continuation("〜良いかもしれないんですけども") is True
    assert clip_selector._segment_ending_is_confident("〜良いかもしれないんですけども") is False


def test_has_confident_natural_ending_rejects_kedomo_final_segment():
    # G: a candidate whose last segment ends in "...けども" must fail the
    # final-ending check, same as any other unfinished ending.
    transcript = _junction_transcript()
    raw = _raw_candidate(3, 3)  # segment 3 is a placeholder; override text
    transcript.segments[3].text = "〜良いかもしれないんですけども"
    assert clip_selector.has_confident_natural_ending(raw, transcript) is False


def test_overlap_H_allows_hook_payoff_exact_repeat_with_context_between():
    transcript = _junction_transcript()
    raw = RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=2, end_segment_id=2),
            RawUsedSegment(role="context", start_segment_id=0, end_segment_id=1),
            RawUsedSegment(role="payoff", start_segment_id=2, end_segment_id=2),
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )
    assert clip_selector._has_overlapping_segments(raw, transcript) is False


def test_overlap_I_rejects_partial_overlap_even_with_hook_payoff_roles():
    transcript = _junction_transcript()
    raw = RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=1, end_segment_id=2),
            RawUsedSegment(role="context", start_segment_id=0, end_segment_id=0),
            RawUsedSegment(role="payoff", start_segment_id=2, end_segment_id=2),
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )
    assert clip_selector._has_overlapping_segments(raw, transcript) is True


def test_overlap_J_rejects_same_segment_reused_three_times():
    transcript = _junction_transcript()
    raw = RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=2, end_segment_id=2),
            RawUsedSegment(role="answer", start_segment_id=2, end_segment_id=2),
            RawUsedSegment(role="payoff", start_segment_id=2, end_segment_id=2),
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )
    assert clip_selector._has_overlapping_segments(raw, transcript) is True


def test_overlap_rejects_adjacent_hook_repeat():
    transcript = _junction_transcript()
    raw = RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=2, end_segment_id=2),
            RawUsedSegment(role="payoff", start_segment_id=2, end_segment_id=2),
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )
    assert clip_selector._has_overlapping_segments(raw, transcript) is True


def test_overlap_rejects_repeat_with_wrong_second_role():
    transcript = _junction_transcript()
    raw = RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=2, end_segment_id=2),
            RawUsedSegment(role="context", start_segment_id=0, end_segment_id=1),
            RawUsedSegment(role="context", start_segment_id=2, end_segment_id=2),
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )
    assert clip_selector._has_overlapping_segments(raw, transcript) is True


def test_is_candidate_junction_safe_combines_both_checks():
    transcript = _junction_transcript()
    bad_junction = _junction_candidate(3, 3)
    assert clip_selector.is_candidate_junction_safe(bad_junction, transcript) is False

    good = RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=2, end_segment_id=2),
            RawUsedSegment(role="context", start_segment_id=0, end_segment_id=1),
            RawUsedSegment(role="payoff", start_segment_id=2, end_segment_id=2),
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )
    assert clip_selector.is_candidate_junction_safe(good, transcript) is True


def test_filter_local_quality_extends_internal_junction_before_rejecting(monkeypatch):
    # Internal (non-last) segment ending unfinished, followed by a
    # non-chronological jump: _extend_internal_junctions must try
    # extending it to the real transcript continuation first. Extending
    # segment 0-1's unfinished "...のであれば" ending reaches segment 2
    # (the real conclusion) *before* it would hit segment 3 (blocked,
    # since the candidate's other segment already uses it) -- turning an
    # unsafe (1 -> 3) jump into a safe, chronological (2 -> 3) one, rather
    # than rejecting the candidate outright.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _junction_transcript()
    raw = RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=1),
            RawUsedSegment(role="payoff", start_segment_id=3, end_segment_id=3),
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )
    kept = clip_selector._filter_local_quality([raw], transcript)
    assert len(kept) == 1
    assert kept[0].segments[0].end_segment_id == 2
    assert kept[0].segments[-1].start_segment_id == 3


# --- diagnostic evaluator: evaluate_local_candidate / diagnose_local_filter
# (real-machine incident: "Stage1を再解析しましたが...候補が0件しかありません
# でした" with no visibility into *why* -- this makes the reason visible,
# API 0, without changing which candidates pass or fail) ------------------


def test_evaluate_local_candidate_A_hook_strength_below_80(monkeypatch):
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _long_transcript(minutes=1)
    raw = _raw_candidate(0, 2, opening_hook_strength=79)

    result = clip_selector.evaluate_local_candidate(raw, transcript)
    assert result.accepted is False
    assert result.reason == "hook_strength_below_80"


def test_evaluate_local_candidate_B_duration_too_short(monkeypatch):
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 20.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 50.0)
    transcript = _long_transcript(minutes=1)
    raw = _raw_candidate(0, 0, opening_hook_strength=90)  # single ~2s segment

    result = clip_selector.evaluate_local_candidate(raw, transcript)
    assert result.accepted is False
    assert result.reason == "duration_too_short"


def test_evaluate_local_candidate_C_duration_too_long(monkeypatch):
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 20.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 50.0)
    transcript = _long_transcript(minutes=5)  # plenty of segments (20s apart)
    raw = _raw_candidate(0, 5, opening_hook_strength=90)  # spans >100s, far more than 50s

    result = clip_selector.evaluate_local_candidate(raw, transcript)
    assert result.accepted is False
    assert result.reason == "duration_too_long"


def test_evaluate_local_candidate_D_kedomo_ending_is_incomplete(monkeypatch):
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    # Last (and only) transcript segment -- nothing left to extend into.
    transcript = _transcript_with_gap(0.3, ["冒頭の発言です。", "〜良いかもしれないんですけども"])
    raw = _raw_candidate(0, 1, opening_hook_strength=90)

    result = clip_selector.evaluate_local_candidate(raw, transcript)
    assert result.accepted is False
    assert result.reason == "incomplete_final_ending"


def _unfixable_bad_junction_transcript():
    # Unlike _junction_transcript (where the gap to the real continuation
    # is small enough that _extend_internal_junctions can bridge it, per
    # test_filter_local_quality_extends_internal_junction_before_
    # rejecting), here the gap to segment 1 (the next transcript segment,
    # NOT used by the candidate below) is deliberately too large
    # (END_EXTENSION_CONTINUATION_MAX_GAP_SEC default 1.5s) for extension
    # to bridge at all -- so the unfinished "...のであれば" ending truly
    # cannot be fixed. Segment 2 (a distant, unrelated segment) sits at
    # transcript index 2, so jumping straight to it from segment 0 is a
    # genuine non-chronological jump (index 2 != index 0 + 1), unlike a
    # 2-segment transcript where the next segment is always "adjacent" by
    # list position regardless of its actual time gap.
    return Transcript(
        video_id="vidUnfixable",
        language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=2.0, text="車を冷やしますっていうのであれば",
                words=[TranscriptWord(start=0.0, end=2.0, text="車を冷やしますっていうのであれば")],
            ),
            TranscriptSegment(
                id=1, start=20.0, end=22.0, text="別の話題の説明です。",
                words=[TranscriptWord(start=20.0, end=22.0, text="別の話題の説明です。")],
            ),
            TranscriptSegment(
                id=2, start=40.0, end=42.0, text="連続周回をする場合は違う話になります",
                words=[TranscriptWord(start=40.0, end=42.0, text="連続周回をする場合は違う話になります")],
            ),
        ],
    )


def test_evaluate_local_candidate_E_bad_junction_reports_jump_prev_incomplete(monkeypatch):
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _unfixable_bad_junction_transcript()
    bad = RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0),
            RawUsedSegment(role="context", start_segment_id=2, end_segment_id=2),
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )

    result = clip_selector.evaluate_local_candidate(bad, transcript)
    assert result.accepted is False
    assert result.reason == "unsafe_junction"
    assert result.junction_reason == "jump_prev_incomplete"


def test_evaluate_local_candidate_F_context_dependent_hook(monkeypatch):
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = Transcript(
        video_id="vidF2",
        language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=3.0,
                text="これのクラッチ交換の際にメタルクラッチを入れるとミッションが壊れやすくなるっていうのはよく言われてます",
                words=[TranscriptWord(
                    start=0.0, end=3.0,
                    text="これのクラッチ交換の際にメタルクラッチを入れるとミッションが壊れやすくなるっていうのはよく言われてます",
                )],
            ),
        ],
    )
    raw = _raw_candidate(0, 0, opening_hook_strength=90)

    result = clip_selector.evaluate_local_candidate(raw, transcript)
    assert result.accepted is False
    assert result.reason == "context_dependent_opening"
    assert result.junction_reason == "hook_context_dependent"


def test_evaluate_local_candidate_G_candidate1_anchor_not_rejected_as_weak_or_context_dependent(monkeypatch):
    # G: start_anchor_text="86は" must not itself cause a
    # weak_opening_prefix or context_dependent_opening rejection -- the
    # candidate should be fully accepted.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = Transcript(
        video_id="vidG",
        language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=3.0,
                text="これも私の愛車である86はスープラをベースに作られています。",
                words=[
                    TranscriptWord(start=0.0, end=0.3, text="これも"),
                    TranscriptWord(start=0.3, end=0.6, text="私の"),
                    TranscriptWord(start=0.6, end=0.9, text="愛車である"),
                    TranscriptWord(start=0.9, end=1.2, text="86は"),
                    TranscriptWord(start=1.2, end=3.0, text="スープラをベースに作られています。"),
                ],
            ),
        ],
    )
    raw = RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0, start_anchor_text="86は")
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )

    result = clip_selector.evaluate_local_candidate(raw, transcript)
    assert result.reason == "accepted"
    assert result.accepted is True


def test_evaluate_local_candidate_H_accepted_candidate(monkeypatch):
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _long_transcript(minutes=1)
    raw = _raw_candidate(0, 2, opening_hook_strength=90)

    result = clip_selector.evaluate_local_candidate(raw, transcript)
    assert result.accepted is True
    assert result.reason == "accepted"


def test_evaluate_local_candidate_I_matches_filter_local_quality_exactly(monkeypatch):
    # I: production accept/reject (_filter_local_quality) and the
    # diagnostic evaluator must never disagree -- they share one
    # implementation (evaluate_local_candidate).
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _unfixable_bad_junction_transcript()
    candidates = [
        RawClipCandidate(
            hook_type="strong_take",
            segments=[
                RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0),
                RawUsedSegment(role="context", start_segment_id=2, end_segment_id=2),
            ],
            hook_text="h", opening_hook_strength=90, title="", description="",
            score=90, reasoning="", caveats="",
        ),  # unsafe_junction
        RawClipCandidate(
            hook_type="strong_take",
            segments=[RawUsedSegment(role="hook", start_segment_id=1, end_segment_id=1)],
            hook_text="h", opening_hook_strength=79, title="", description="",
            score=79, reasoning="", caveats="",
        ),  # hook_strength_below_80
        RawClipCandidate(
            hook_type="strong_take",
            segments=[RawUsedSegment(role="hook", start_segment_id=1, end_segment_id=1)],
            hook_text="h", opening_hook_strength=90, title="", description="",
            score=90, reasoning="", caveats="",
        ),  # accepted
    ]

    filtered = clip_selector._filter_local_quality(candidates, transcript)
    evaluations = clip_selector._evaluate_all_local_candidates(candidates, transcript)
    accepted_via_evaluations = [e.candidate for e in evaluations if e.accepted]

    assert len(filtered) == len(accepted_via_evaluations) == 1
    assert filtered[0].segments == accepted_via_evaluations[0].segments


def test_refresh_stage1_and_candidates_J_error_includes_diagnostic_summary(monkeypatch):
    # J: the exact real-machine failure path -- diagnostic counts and
    # per-candidate detail must be embedded in the RuntimeError text
    # (which becomes job.error, already rendered to the user).
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _long_transcript(minutes=1)
    candidates = [_raw_candidate(0, 0, opening_hook_strength=50)]
    monkeypatch.setattr(clip_selector, "run_stage1", lambda *a, **k: candidates)

    with pytest.raises(RuntimeError) as exc_info:
        clip_selector.refresh_stage1_and_candidates(transcript, "タイトル")

    message = str(exc_info.value)
    assert "【診断】" in message
    assert "Stage1候補: 1件" in message
    assert "hook強度不足" in message


def test_diagnose_local_filter_K_makes_zero_api_calls(monkeypatch):
    # K: relies on this module's autouse _forbid_real_anthropic_client
    # fixture (poisons anthropic.Anthropic()) plus an explicit guard on
    # run_stage1/extract_candidates_for_chunk -- diagnose_local_filter
    # must never reach either.
    transcript = _long_transcript(minutes=1)
    cache.save_stage1_chunk(transcript.video_id, 0, [_raw_candidate(0, 0, opening_hook_strength=90)])

    def _forbidden(*a, **k):
        raise AssertionError("diagnose_local_filter must never call the Stage1 API")

    monkeypatch.setattr(clip_selector, "run_stage1", _forbidden)
    monkeypatch.setattr(clip_selector, "extract_candidates_for_chunk", _forbidden)

    evaluations = clip_selector.diagnose_local_filter(transcript)
    assert len(evaluations) >= 1


def test_diagnose_local_filter_L_raises_clearly_without_cache():
    transcript = _long_transcript(minutes=1)
    transcript.video_id = "vid-never-cached-for-diagnosis"

    with pytest.raises(RuntimeError, match="Stage1候補キャッシュ"):
        clip_selector.diagnose_local_filter(transcript)


def test_candidate_schema_version_still_8():
    # L: this round adds diagnostics only -- no Stage1 output shape
    # change, so the schema version must stay exactly where the previous
    # round (junction safety) left it.
    assert config.CANDIDATE_SCHEMA_VERSION == 8


# --- repair-before-reject: real-machine incident (4/4 Stage1 candidates
# rejected under schema v8) -- deterministic, API-0 repair tried before a
# candidate is finally rejected, always re-judged by the identical
# evaluate_local_candidate no separate lenient path -------------------


def test_repair_A_candidate1_auto_opening_trim(monkeypatch):
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = Transcript(
        video_id="repairA", language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=3.0,
                text="これも私の愛車である86はハイグリップタイヤでサーキットを走ります。",
                words=[
                    TranscriptWord(start=0.0, end=0.3, text="これも"),
                    TranscriptWord(start=0.3, end=0.6, text="私の"),
                    TranscriptWord(start=0.6, end=0.9, text="愛車である"),
                    TranscriptWord(start=0.9, end=1.2, text="86は"),
                    TranscriptWord(start=1.2, end=3.0, text="ハイグリップタイヤでサーキットを走ります。"),
                ],
            ),
        ],
    )
    # Stage1 did not set start_anchor_text this time (the real-machine
    # failure mode) -- repair must discover the trim locally.
    candidate = RawClipCandidate(
        hook_type="strong_take",
        segments=[RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0)],
        hook_text="h", opening_hook_strength=85, title="", description="",
        score=85, reasoning="", caveats="",
    )

    result = clip_selector.evaluate_local_candidate_with_repair(candidate, transcript)
    assert result.accepted is True
    assert result.repair_method == "opening_trim"
    assert result.original_reason == "context_dependent_opening"
    resolved = boundary.resolve_candidate(result.candidate, transcript, candidate_id="c1")
    assert resolved.segments[0].text.startswith("86は")


def test_repair_B_no_word_timestamps_keeps_rejection(monkeypatch):
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = Transcript(
        video_id="repairB", language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=3.0,
                text="これも私の愛車である86はハイグリップタイヤでサーキットを走ります。",
                words=[],  # no word timestamps at all
            ),
        ],
    )
    candidate = RawClipCandidate(
        hook_type="strong_take",
        segments=[RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0)],
        hook_text="h", opening_hook_strength=85, title="", description="",
        score=85, reasoning="", caveats="",
    )

    result = clip_selector.evaluate_local_candidate_with_repair(candidate, transcript)
    assert result.accepted is False
    assert result.reason == "context_dependent_opening"
    assert result.repair_method is None


def test_repair_C_sequential_prefix_trim_to_ZN6_86():
    # C: "よくある話が" alone is not itself a mechanical reject trigger
    # (it's in neither WEAK_OPENING_PREFIXES nor CONTEXT_DEPENDENT_
    # OPENING_PREFIXES), so this candidate is accepted outright without
    # needing repair -- this test instead pins _try_opening_trim_repair's
    # sequential chaining directly: "よくある話が" then "私も乗っている"
    # both get cleared in one pass, landing on "ZN6-86であったり...".
    transcript = Transcript(
        video_id="repairC", language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=5.0,
                text="よくある話が私も乗っているZN6-86であったりあとはBRZあとGR86です。",
                words=[
                    TranscriptWord(start=0.0, end=0.4, text="よくある話が"),
                    TranscriptWord(start=0.4, end=0.8, text="私も乗っている"),
                    TranscriptWord(start=0.8, end=4.5, text="ZN6-86であったりあとはBRZあとGR86"),
                    TranscriptWord(start=4.5, end=5.0, text="です。"),
                ],
            ),
        ],
    )
    candidate = RawClipCandidate(
        hook_type="surprising_fact",
        segments=[RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0)],
        hook_text="h", opening_hook_strength=83, title="", description="",
        score=83, reasoning="", caveats="",
    )

    trimmed = clip_selector._try_opening_trim_repair(candidate, transcript)
    assert trimmed is not None
    resolved = boundary.resolve_candidate(trimmed, transcript, candidate_id="c1")
    assert resolved.segments[0].text.startswith("ZN6-86であったり")
    assert "よくある話が" not in resolved.segments[0].text
    assert "私も乗っている" not in resolved.segments[0].text


def _candidate2_transcript():
    return Transcript(
        video_id="repairD", language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=5.0,
                text="よくある話が私も乗っているZN6-86であったりあとはBRZあとGR86です。",
                words=[
                    TranscriptWord(start=0.0, end=0.4, text="よくある話が"),
                    TranscriptWord(start=0.4, end=0.8, text="私も乗っている"),
                    TranscriptWord(start=0.8, end=4.5, text="ZN6-86であったりあとはBRZあとGR86"),
                    TranscriptWord(start=4.5, end=5.0, text="です。"),
                ],
            ),
            TranscriptSegment(
                id=1, start=5.3, end=8.0,
                text="これのクラッチ交換の際にメタルクラッチを入れるとミッションが壊れやすくなるっていうのはよく言われてます。",
                words=[TranscriptWord(
                    start=5.3, end=8.0,
                    text="これのクラッチ交換の際にメタルクラッチを入れるとミッションが壊れやすくなるっていうのはよく言われてます。",
                )],
            ),
        ],
    )


def test_repair_D_candidate2_prepend_previous_segment(monkeypatch):
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _candidate2_transcript()
    candidate = RawClipCandidate(
        hook_type="surprising_fact",
        segments=[RawUsedSegment(role="hook", start_segment_id=1, end_segment_id=1)],
        hook_text="h", opening_hook_strength=83, title="", description="",
        score=83, reasoning="", caveats="",
    )

    result = clip_selector.evaluate_local_candidate_with_repair(candidate, transcript)
    assert result.accepted is True
    assert result.repair_method == "prepend_previous_and_trim"
    assert result.original_reason == "context_dependent_opening"
    resolved = boundary.resolve_candidate(result.candidate, transcript, candidate_id="c1")
    assert resolved.segments[0].text.startswith("ZN6-86であったり")


def test_repair_E_prepend_previous_still_context_dependent_keeps_rejection(monkeypatch):
    # E: the previous segment doesn't actually resolve the dangling
    # reference (still starts with a context-dependent word after trim) --
    # repair must not paper over a genuinely unresolved opening.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = Transcript(
        video_id="repairE", language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=3.0, text="それについては後で話します。",
                words=[TranscriptWord(start=0.0, end=3.0, text="それについては後で話します。")],
            ),
            TranscriptSegment(
                id=1, start=3.3, end=6.0,
                text="これのクラッチ交換の際にメタルクラッチを入れるとミッションが壊れやすくなるっていうのはよく言われてます。",
                words=[TranscriptWord(
                    start=3.3, end=6.0,
                    text="これのクラッチ交換の際にメタルクラッチを入れるとミッションが壊れやすくなるっていうのはよく言われてます。",
                )],
            ),
        ],
    )
    candidate = RawClipCandidate(
        hook_type="surprising_fact",
        segments=[RawUsedSegment(role="hook", start_segment_id=1, end_segment_id=1)],
        hook_text="h", opening_hook_strength=83, title="", description="",
        score=83, reasoning="", caveats="",
    )

    result = clip_selector.evaluate_local_candidate_with_repair(candidate, transcript)
    assert result.accepted is False
    assert result.reason == "context_dependent_opening"
    assert result.repair_method is None
    assert "prepend_previous_and_trim" in result.attempted_repair_methods


def _candidate3_transcript():
    return Transcript(
        video_id="repairF", language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=25.0,
                text="冷却不足という弱点はなくなるんですけども今度は違う問題が出てきます。",
                words=[TranscriptWord(start=0.0, end=25.0, text="x")],
            ),
            TranscriptSegment(
                id=1, start=25.3, end=45.0,
                text="真冬のサーキットで2、3周しかアタックをしません。それが理由です。",
                words=[TranscriptWord(start=25.3, end=45.0, text="x")],
            ),
            TranscriptSegment(
                id=2, start=45.3, end=55.8,
                text="冷却効率を上げるために重量を増やすというのはアンチパターンになると思います。",
                words=[TranscriptWord(start=45.3, end=55.8, text="x")],
            ),
        ],
    )


def _candidate3_raw():
    return RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0),
            RawUsedSegment(role="context", start_segment_id=1, end_segment_id=1),
            RawUsedSegment(role="answer", start_segment_id=2, end_segment_id=2),
        ],
        hook_text="h", opening_hook_strength=80, title="", description="",
        score=80, reasoning="", caveats="",
    )


def test_repair_F_candidate3_drops_context_segment_under_50s(monkeypatch):
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 20.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 50.0)
    transcript = _candidate3_transcript()
    candidate = _candidate3_raw()

    result = clip_selector.evaluate_local_candidate_with_repair(candidate, transcript)
    assert result.accepted is True
    assert result.repair_method == "drop_context_segment"
    assert result.original_reason == "duration_too_long"
    assert 20.0 <= result.duration_sec <= 50.0
    assert [s.role for s in result.candidate.segments] == ["hook", "answer"]


def test_repair_G_drop_segment_still_unsafe_keeps_rejection(monkeypatch):
    # G: dropping either non-hook segment leaves the unfinished hook
    # jumping to something non-chronological (segment id=1 is a filler,
    # never referenced by the candidate itself, sitting between hook and
    # both context/answer -- so neither drop variant can become
    # "chronologically adjacent" the way test F's clean drop can). The gap
    # from the hook to id=1 (5.0s) is deliberately larger than both
    # END_EXTENSION_MAX_GAP_SEC and END_EXTENSION_CONTINUATION_MAX_GAP_SEC,
    # so _extend_internal_junctions never bridges the hook into id=1 --
    # otherwise the hook would absorb id=1 and become index-adjacent to
    # id=2, which would make that junction chronological (and therefore
    # safe) for the wrong reason.
    # Dropping must not be accepted just because duration now fits.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 20.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 50.0)
    transcript = Transcript(
        video_id="repairG", language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=25.0, text="車を冷やしますっていうのであれば",
                words=[TranscriptWord(start=0.0, end=25.0, text="x")],
            ),
            TranscriptSegment(
                id=1, start=30.0, end=32.7, text="無関係な話題です。",
                words=[TranscriptWord(start=30.0, end=32.7, text="x")],
            ),
            TranscriptSegment(
                id=2, start=45.3, end=48.0, text="別の話題の説明です。",
                words=[TranscriptWord(start=45.3, end=48.0, text="x")],
            ),
            TranscriptSegment(
                id=3, start=100.3, end=125.3, text="連続周回をする場合は違う話になります",
                words=[TranscriptWord(start=100.3, end=125.3, text="x")],
            ),
        ],
    )
    # Total duration is 25.0 + 2.7 + 25.0 = 52.7s -> duration_too_long.
    # Dropping "context" (id=2) leaves hook(0-25)+answer(25.0) = 50.0s, which
    # fits, but the hook ends in a confirmed continuation suffix ("...れば")
    # so it is not "confidently complete", and jumping straight to id=3 is a
    # non-chronological jump -- unsafe. Dropping "answer" (id=3) leaves
    # hook(0-25)+context(2.7)=27.7s, which also fits, but is the same
    # non-chronological jump from the same not-confidently-complete hook.
    # Neither variant may be accepted just because duration now fits.
    candidate = RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0),
            RawUsedSegment(role="context", start_segment_id=2, end_segment_id=2),
            RawUsedSegment(role="answer", start_segment_id=3, end_segment_id=3),
        ],
        hook_text="h", opening_hook_strength=80, title="", description="",
        score=80, reasoning="", caveats="",
    )

    result = clip_selector.evaluate_local_candidate_with_repair(candidate, transcript)
    assert result.accepted is False
    assert "drop_context_segment" in result.attempted_repair_methods
    assert "drop_non_context_segment" in result.attempted_repair_methods


def test_repair_H_never_hard_cuts_mid_sentence_at_50s(monkeypatch):
    # H: drop-segment repair only ever removes *whole* segments -- it must
    # never truncate a segment's own start/end to force it under 50s.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 20.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 50.0)
    transcript = _candidate3_transcript()
    candidate = _candidate3_raw()

    variants = clip_selector.generate_local_repair_variants(candidate, transcript, "duration_too_long")
    for _, variant in variants:
        for orig_seg, new_seg in zip(candidate.segments, variant.segments):
            if new_seg.start_segment_id == orig_seg.start_segment_id:
                # A kept segment's own range must be byte-identical to
                # what Stage1 chose -- never partially trimmed.
                assert new_seg.end_segment_id == orig_seg.end_segment_id
        # Every segment in the variant must be one of the original
        # segments verbatim -- never a new, narrower range.
        original_ranges = {(s.start_segment_id, s.end_segment_id) for s in candidate.segments}
        for s in variant.segments:
            assert (s.start_segment_id, s.end_segment_id) in original_ranges


def _candidate4_transcript():
    return Transcript(
        video_id="repairI", language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=5.0,
                text="冷却効率を上げるために重量を増やすというのはアンチパターンになると思います。",
                words=[TranscriptWord(start=0.0, end=5.0, text="x")],
            ),
            TranscriptSegment(
                id=1, start=5.3, end=25.0,
                text="真冬のサーキットで2、3周しかアタックをしませんという話なんですけども",
                words=[TranscriptWord(start=5.3, end=25.0, text="x")],
            ),
        ],
    )


def _candidate4_raw():
    return RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0),
            RawUsedSegment(role="context", start_segment_id=1, end_segment_id=1),
        ],
        hook_text="h", opening_hook_strength=83, title="", description="",
        score=83, reasoning="", caveats="",
    )


def test_repair_I_candidate4_hook_repeat_payoff(monkeypatch):
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 20.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 50.0)
    transcript = _candidate4_transcript()
    candidate = _candidate4_raw()

    result = clip_selector.evaluate_local_candidate_with_repair(candidate, transcript)
    assert result.accepted is True
    assert result.repair_method == "hook_repeat_payoff"
    assert result.original_reason == "incomplete_final_ending"
    assert [s.role for s in result.candidate.segments] == ["hook", "context", "payoff"]
    assert result.candidate.segments[-1].start_segment_id == result.candidate.segments[0].start_segment_id


def test_repair_J_hook_repeat_payoff_over_50s_keeps_rejection(monkeypatch):
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 20.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 30.0)  # tight ceiling
    transcript = _candidate4_transcript()
    candidate = _candidate4_raw()

    result = clip_selector.evaluate_local_candidate_with_repair(candidate, transcript)
    assert result.accepted is False
    assert "hook_repeat_payoff" in result.attempted_repair_methods


def test_repair_K_hook_repeat_payoff_respects_exact_repeat_limit(monkeypatch):
    # K: the underlying overlap rule (max 2 uses of the same source range,
    # hook then answer/payoff) must still hold for a repair-generated
    # repeat -- a candidate that already contains a hook/payoff exact
    # repeat must not get a *third* use appended.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _candidate4_transcript()
    candidate = RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0),
            RawUsedSegment(role="context", start_segment_id=1, end_segment_id=1),
            RawUsedSegment(role="payoff", start_segment_id=0, end_segment_id=0),
        ],
        hook_text="h", opening_hook_strength=83, title="", description="",
        score=83, reasoning="", caveats="",
    )
    # context's own ending is still unfinished ("...けども") -- would
    # normally prompt another hook_repeat_payoff attempt.
    variants = clip_selector.generate_local_repair_variants(candidate, transcript, "incomplete_final_ending")
    for _, variant in variants:
        ev = clip_selector.evaluate_local_candidate(variant, transcript)
        assert not (ev.accepted and clip_selector._has_overlapping_segments(ev.candidate, transcript) is False and
                    sum(1 for s in ev.candidate.segments if s.start_segment_id == 0) > 2), (
            "must never allow the hook's source range to be used 3+ times"
        )


def test_repair_L_always_routes_through_evaluate_local_candidate(monkeypatch):
    # L: every repair variant must be judged by the exact same evaluator
    # production uses -- verified by cross-checking generate_local_repair_
    # variants' outputs against a direct evaluate_local_candidate call.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = Transcript(
        video_id="repairL", language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=3.0,
                text="これも私の愛車である86はスープラをベースに作られています。",
                words=[
                    TranscriptWord(start=0.0, end=0.3, text="これも"),
                    TranscriptWord(start=0.3, end=0.6, text="私の"),
                    TranscriptWord(start=0.6, end=0.9, text="愛車である"),
                    TranscriptWord(start=0.9, end=1.2, text="86は"),
                    TranscriptWord(start=1.2, end=3.0, text="スープラをベースに作られています。"),
                ],
            ),
        ],
    )
    candidate = RawClipCandidate(
        hook_type="strong_take",
        segments=[RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0)],
        hook_text="h", opening_hook_strength=85, title="", description="",
        score=85, reasoning="", caveats="",
    )
    with_repair = clip_selector.evaluate_local_candidate_with_repair(candidate, transcript)
    _, only_variant = clip_selector.generate_local_repair_variants(candidate, transcript, "context_dependent_opening")[0]
    direct = clip_selector.evaluate_local_candidate(only_variant, transcript)
    assert with_repair.accepted == direct.accepted
    assert with_repair.candidate.segments == direct.candidate.segments


def test_repair_M_variant_count_is_bounded():
    assert clip_selector._MAX_LOCAL_REPAIR_VARIANTS <= 8


def test_repair_N_diagnostic_summary_shows_repair_method(monkeypatch):
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = Transcript(
        video_id="repairN", language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=3.0,
                text="これも私の愛車である86はスープラをベースに作られています。",
                words=[
                    TranscriptWord(start=0.0, end=0.3, text="これも"),
                    TranscriptWord(start=0.3, end=0.6, text="私の"),
                    TranscriptWord(start=0.6, end=0.9, text="愛車である"),
                    TranscriptWord(start=0.9, end=1.2, text="86は"),
                    TranscriptWord(start=1.2, end=3.0, text="スープラをベースに作られています。"),
                ],
            ),
        ],
    )
    candidate = RawClipCandidate(
        hook_type="strong_take",
        segments=[RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0)],
        hook_text="h", opening_hook_strength=85, title="", description="",
        score=85, reasoning="", caveats="",
    )
    evaluations = clip_selector._evaluate_all_local_candidates([candidate], transcript)
    summary = clip_selector._format_diagnostic_summary(evaluations)
    assert "opening_trim" in summary
    assert "original_reject=context_dependent_opening" in summary
    assert "→ accepted" in summary


def test_repair_Q_zero_api_calls_via_repair(monkeypatch):
    # Q: repair generation/evaluation is pure local computation -- relies
    # on this module's autouse _forbid_real_anthropic_client fixture, plus
    # an explicit guard that structured_output.call is never reached.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)

    def _forbidden(*a, **k):
        raise AssertionError("repair must never call the Anthropic API")

    monkeypatch.setattr(clip_selector.structured_output, "call", _forbidden)

    transcript = Transcript(
        video_id="repairQ", language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=3.0,
                text="これも私の愛車である86はスープラをベースに作られています。",
                words=[
                    TranscriptWord(start=0.0, end=0.3, text="これも"),
                    TranscriptWord(start=0.3, end=0.6, text="私の"),
                    TranscriptWord(start=0.6, end=0.9, text="愛車である"),
                    TranscriptWord(start=0.9, end=1.2, text="86は"),
                    TranscriptWord(start=1.2, end=3.0, text="スープラをベースに作られています。"),
                ],
            ),
        ],
    )
    candidate = RawClipCandidate(
        hook_type="strong_take",
        segments=[RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0)],
        hook_text="h", opening_hook_strength=85, title="", description="",
        score=85, reasoning="", caveats="",
    )
    result = clip_selector.evaluate_local_candidate_with_repair(candidate, transcript)
    assert result.accepted is True


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


def test_select_candidates_cache_hit_rewrites_cache_with_finalized_candidates(monkeypatch):
    """The Stage2 cache on disk must be normalized to the finalized
    (ending-corrected/duration-validated) state the moment a cache hit
    succeeds -- otherwise web.py's render path, which reads
    cache.load_stage2 directly, would still see the stale pre-correction
    candidates even though the UI already showed the corrected ones.
    """
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _transcript_with_gap(
        0.3, ["冒頭の発言です。", "それが起きた理由としては、こういうことが考えられるので", "そのあたりも確認する必要があります。"]
    )
    stale_cached_candidate = _raw_candidate(0, 1, opening_hook_strength=90)
    cache.save_stage2(transcript.video_id, [stale_cached_candidate] * 3)

    result = clip_selector.select_candidates(transcript, "タイトル")
    assert all(c.segments[-1].end_segment_id == 2 for c in result)

    # Re-reading the cache from scratch (a fresh load, simulating what
    # web._run_render would see) must return the already-finalized state,
    # not the original stale end_segment_id=1.
    reloaded = cache.load_stage2(transcript.video_id)
    assert all(c.segments[-1].end_segment_id == 2 for c in reloaded)


def test_select_candidates_cache_hit_does_not_rewrite_cache_when_insufficient_valid(monkeypatch):
    """When a cache hit doesn't have enough eligible candidates after
    finalization, select_candidates must raise *without* touching the
    on-disk cache -- never overwrite it with a known-insufficient result,
    and never silently discard the original (potentially still-useful for
    diagnosis, or for a future local-rule change) cached data.
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

    # The cache must be completely untouched -- still the original 3
    # stale candidates, not overwritten with an empty/partial result.
    reloaded = cache.load_stage2(transcript.video_id)
    assert len(reloaded) == 3
    assert all(c.segments[-1].end_segment_id == 1 for c in reloaded)


def test_select_candidates_fresh_path_saves_finalized_candidates_to_cache(monkeypatch):
    """The fresh (non-cache-hit) path must also save the finalized
    (post-extension) candidates to disk, not the raw Stage2 picks --
    cache.save_stage2 must run *after* finalize_candidates, never before.
    """
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _transcript_with_gap(
        0.3, ["冒頭の発言です。", "それが起きた理由としては、こういうことが考えられるので", "そのあたりも確認する必要があります。"]
    )
    candidate = _raw_candidate(0, 1, opening_hook_strength=90)
    monkeypatch.setattr(clip_selector, "run_stage1", lambda *a, **k: [candidate] * 4)
    monkeypatch.setattr(clip_selector, "rank_candidates", lambda id_map, t, title: list(id_map.keys()))

    result = clip_selector.select_candidates(transcript, "タイトル")
    assert all(c.segments[-1].end_segment_id == 2 for c in result)

    reloaded = cache.load_stage2(transcript.video_id)
    assert all(c.segments[-1].end_segment_id == 2 for c in reloaded)


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


def test_stage1_output_accepts_zero_to_six_candidates():
    """Stage1's per-chunk cap was widened 3 -> config.STAGE1_MAX_CANDIDATES_
    PER_CHUNK (6): Stage1's job is recall (cast a wide net of candidates
    that could plausibly clear MIN_OPENING_HOOK_STRENGTH), not picking the
    final best-3 -- that narrowing still happens via the local quality
    filter + Stage2 ranking, not by capping Stage1's search breadth.
    """
    assert config.STAGE1_MAX_CANDIDATES_PER_CHUNK == 6
    assert Stage1Output(candidates=[]).candidates == []
    for n in range(1, 7):
        out = Stage1Output(candidates=[Stage1CandidateOutput(**_valid_candidate_kwargs()) for _ in range(n)])
        assert len(out.candidates) == n
    with pytest.raises(ValidationError):
        Stage1Output(candidates=[Stage1CandidateOutput(**_valid_candidate_kwargs()) for _ in range(7)])


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
    assert clip_selector._deterministic_hook_text(0, None, segments) == "これは実際の発言です"


def test_deterministic_hook_text_truncates_by_character_count_only(monkeypatch):
    monkeypatch.setattr(config, "HOOK_TEXT_MAX_CHARS", 5)
    segments = [_segment(0, start=0.0, text="abcdefghij")]
    result = clip_selector._deterministic_hook_text(0, None, segments)
    assert result == "abcde…"


def test_deterministic_hook_text_reflects_anchor_trim():
    """hook_text (the UI's "冒頭の実音声") must match what boundary.py
    actually resolves as the opening -- so a weak self-introduction lead-in
    like "これも私の愛車である" doesn't show in the UI when start_anchor_text
    has already trimmed it out of the rendered clip's real opening.
    """
    segments = [
        TranscriptSegment(
            id=0, start=0.0, end=3.0,
            text="これも私の愛車である86はスープラをベースに作られています",
            words=[
                TranscriptWord(start=0.0, end=0.3, text="これも"),
                TranscriptWord(start=0.3, end=0.6, text="私の"),
                TranscriptWord(start=0.6, end=0.9, text="愛車である"),
                TranscriptWord(start=0.9, end=1.2, text="86は"),
                TranscriptWord(start=1.2, end=3.0, text="スープラをベースに作られています"),
            ],
        )
    ]
    result = clip_selector._deterministic_hook_text(0, "86は", segments)
    assert result.startswith("86は")
    assert "これも私の愛車である" not in result


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


def test_extract_candidates_for_chunk_carries_anchor_through_to_hook_text(monkeypatch):
    # E: the UI's "冒頭の実音声" (hook_text) must reflect the same anchor
    # trim boundary.py applies when resolving the real edit points -- both
    # must show "86は..." and never the raw untrimmed "これも私の愛車である...".
    kwargs = _valid_candidate_kwargs()
    kwargs["segments"] = [
        {"role": "hook", "start_segment_id": 0, "end_segment_id": 0, "start_anchor_text": "86は"}
    ]
    output = Stage1Output(candidates=[Stage1CandidateOutput(**kwargs)])
    monkeypatch.setattr(
        clip_selector.structured_output, "call",
        lambda schema_model, **kwargs: output,
    )

    chunk_segments = [
        TranscriptSegment(
            id=0, start=0.0, end=3.0,
            text="これも私の愛車である86はスープラをベースに作られています",
            words=[
                TranscriptWord(start=0.0, end=0.3, text="これも"),
                TranscriptWord(start=0.3, end=0.6, text="私の"),
                TranscriptWord(start=0.6, end=0.9, text="愛車である"),
                TranscriptWord(start=0.9, end=1.2, text="86は"),
                TranscriptWord(start=1.2, end=3.0, text="スープラをベースに作られています"),
            ],
        )
    ]
    result = clip_selector.extract_candidates_for_chunk(chunk_segments, "タイトル")

    raw = result[0]
    assert raw.segments[0].start_anchor_text == "86は"
    assert raw.hook_text.startswith("86は")
    assert "これも私の愛車である" not in raw.hook_text

    # UI (hook_text) and render (boundary-resolved opening) must agree.
    transcript = Transcript(video_id="vidE", language="ja", segments=chunk_segments)
    resolved = boundary.resolve_candidate(raw, transcript, candidate_id="c1")
    assert resolved.segments[0].text.startswith(raw.hook_text.rstrip("…"))


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


# --- opening trim: _opening_text/_looks_like_weak_opening must agree with
# --- boundary.py's mechanical trim (item M: 8-11) -------------------------


def _segment_with_words(i, start, *word_texts):
    """A TranscriptSegment whose word list is `word_texts`, laid out
    back-to-back from `start` (0.5s each), with .text set to their exact
    concatenation -- mirrors how boundary._apply_opening_trim expects word
    spans to line up with segment.text (see test_boundary.py).
    """
    words = []
    t = start
    for wt in word_texts:
        words.append(TranscriptWord(start=t, end=t + 0.5, text=wt))
        t += 0.5
    return TranscriptSegment(id=i, start=start, end=t, text="".join(word_texts), words=words)


def test_opening_text_returns_post_trim_text():
    # _opening_text must see what render/UI will actually show (post-trim),
    # never the raw untrimmed transcript text.
    transcript = Transcript(
        video_id="vid1", language="ja",
        segments=[
            _segment_with_words(0, 0.0, "このように", "弱点を", "直すと"),
            _segment_with_words(1, 5.0, "次の弱点が生まれます。"),
        ],
    )
    raw = _raw_candidate(0, 1, opening_hook_strength=90)
    opening = clip_selector._opening_text(raw, transcript)
    assert opening.startswith("弱点を")
    assert "このように" not in opening


def test_looks_like_weak_opening_no_longer_rejects_once_trimmed(monkeypatch):
    # Once boundary.py trims "このように" off the front, _opening_text
    # sees "弱点を直すと..." -- not a weak-opening prefix -- so the
    # candidate must survive _filter_local_quality (items J/K: the reject
    # check and the trim mechanism must agree, never disagree).
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = Transcript(
        video_id="vid1", language="ja",
        segments=[
            _segment_with_words(0, 0.0, "このように", "弱点を", "直すと次の弱点が生まれます。"),
        ],
    )
    raw = _raw_candidate(0, 0, opening_hook_strength=90)
    kept = clip_selector._filter_local_quality([raw], transcript)
    assert len(kept) == 1


def test_looks_like_weak_opening_still_rejects_when_trim_leaves_weak_text(monkeypatch):
    # If, after trimming one known prefix, the opening still starts with
    # another weak prefix from models.WEAK_OPENING_PREFIXES, the reject
    # check must still fire -- trimming only removes the mechanically
    # known lead-in word(s) at the very front, it does not launder an
    # opening that is weak all the way through.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = Transcript(
        video_id="vid1", language="ja",
        segments=[
            _segment_with_words(0, 0.0, "このように", "今回は", "本題に入ります。"),
        ],
    )
    raw = _raw_candidate(0, 0, opening_hook_strength=90)
    kept = clip_selector._filter_local_quality([raw], transcript)
    assert kept == []


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


# --- refresh_candidates_only: low-cost re-selection from cache only ------
# (reuses the already-cached Transcript + Stage1 chunk cache, never calls
# the Stage1 API, calls Stage2 ranking at most once -- see web.py's
# /api/jobs/{id}/refresh-candidates, the "候補だけ再選定" UI action)


def test_refresh_candidates_only_calls_stage2_exactly_once_with_enough_stage1_cache(monkeypatch):
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _long_transcript(minutes=1)
    candidates = [_raw_candidate(0, 2, opening_hook_strength=90) for _ in range(3)]
    cache.save_stage1_chunk(transcript.video_id, 0, candidates)

    call_count = {"n": 0}

    def _fake_rank_candidates(id_map, t, title):
        call_count["n"] += 1
        return list(id_map.keys())

    monkeypatch.setattr(clip_selector, "rank_candidates", _fake_rank_candidates)
    monkeypatch.setattr(
        clip_selector, "extract_candidates_for_chunk",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Stage1 API must not be called")),
    )

    result = clip_selector.refresh_candidates_only(transcript, "タイトル")

    assert len(result) == 3
    assert call_count["n"] == 1
    # The finalized result is saved to the same Stage2 cache
    # select_candidates uses, so a subsequent render sees it too.
    reloaded = cache.load_stage2(transcript.video_id)
    assert len(reloaded) == 3


def test_refresh_candidates_only_raises_without_stage2_call_when_stage1_cache_incomplete(monkeypatch):
    # No Stage1 chunk cache saved at all -- refresh_candidates_only must
    # never call the Stage1 API to fill the gap, and must never reach
    # Stage2 ranking either (Anthropic API calls = 0 for this failure).
    transcript = _long_transcript(minutes=1)
    monkeypatch.setattr(
        clip_selector, "rank_candidates",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Stage2 must not be called")),
    )

    with pytest.raises(RuntimeError, match="Stage1候補キャッシュ"):
        clip_selector.refresh_candidates_only(transcript, "タイトル")


def test_refresh_candidates_only_raises_without_stage2_call_when_too_few_pass_local_filter(monkeypatch):
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 20.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 50.0)
    transcript = _long_transcript(minutes=1)
    # single 2-second segment candidates -- far below the 20s hard minimum,
    # so none survive _filter_local_quality and Stage2 must never be
    # reached (Anthropic API calls = 0 for this failure too).
    candidates = [_raw_candidate(0, 0, opening_hook_strength=90) for _ in range(3)]
    cache.save_stage1_chunk(transcript.video_id, 0, candidates)

    monkeypatch.setattr(
        clip_selector, "rank_candidates",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Stage2 must not be called")),
    )

    with pytest.raises(RuntimeError, match="ローカル品質フィルタ"):
        clip_selector.refresh_candidates_only(transcript, "タイトル")


# --- refresh_stage1_and_candidates: mid-cost re-analysis (Stage1 rebuilt) -
# (Transcript is reused, never re-transcribed; every Stage1 chunk is
# regenerated via the Stage1 API regardless of existing chunk cache; Stage2
# ranking runs at most once -- see web.py's /api/jobs/{id}/refresh-stage1,
# the "Stage1からやり直す" UI action)


def test_refresh_stage1_and_candidates_ignores_existing_stage1_cache(monkeypatch):
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _long_transcript(minutes=1)
    # A stale cached chunk result that would fail the current quality
    # filter (weak opening_hook_strength) -- refresh_stage1_and_candidates
    # must never reuse this, only what a fresh Stage1 call returns.
    stale_bad = [_raw_candidate(0, 0, opening_hook_strength=10)]
    cache.save_stage1_chunk(transcript.video_id, 0, stale_bad)

    fresh_good = [_raw_candidate(0, 2, opening_hook_strength=90) for _ in range(3)]
    call_count = {"n": 0}

    def _fake_extract(chunk_segments, video_title):
        call_count["n"] += 1
        return fresh_good

    monkeypatch.setattr(clip_selector, "extract_candidates_for_chunk", _fake_extract)
    monkeypatch.setattr(clip_selector, "rank_candidates", lambda id_map, t, title: list(id_map.keys()))

    result = clip_selector.refresh_stage1_and_candidates(transcript, "タイトル")

    assert len(result) == 3
    assert call_count["n"] == 1  # exactly one chunk for a 1-minute transcript
    # The stale, weak-opening candidate must be gone: Stage1 was fully
    # regenerated, not reused from the existing (now-outdated) cache --
    # and the chunk cache on disk is overwritten with the new result.
    reloaded_chunk = cache.load_stage1_chunk(transcript.video_id, 0)
    assert all(c.opening_hook_strength == 90 for c in reloaded_chunk)
    # The finalized Stage2 result is saved too.
    reloaded_stage2 = cache.load_stage2(transcript.video_id)
    assert len(reloaded_stage2) == 3


def test_refresh_stage1_and_candidates_keeps_earlier_chunk_success_on_later_failure(monkeypatch):
    monkeypatch.setattr(config, "CHUNK_MINUTES", 10.0)
    monkeypatch.setattr(config, "CHUNK_OVERLAP_MINUTES", 1.0)
    transcript = _long_transcript(minutes=25)
    chunks = clip_selector._build_chunks(clip_selector._usable_segments(transcript))
    assert len(chunks) >= 2  # sanity: this test needs at least 2 chunks

    good = [_raw_candidate(0, 2, opening_hook_strength=90)]
    call_count = {"n": 0}

    def _fake_extract(chunk_segments, video_title):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return good
        raise RuntimeError("simulated Stage1 API failure on chunk 2")

    monkeypatch.setattr(clip_selector, "extract_candidates_for_chunk", _fake_extract)
    monkeypatch.setattr(
        clip_selector, "rank_candidates",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Stage2 must not be called")),
    )

    with pytest.raises(RuntimeError, match="simulated Stage1 API failure"):
        clip_selector.refresh_stage1_and_candidates(transcript, "タイトル")

    # No retry on the failing chunk -- extract_candidates_for_chunk was
    # called exactly once per chunk attempted (chunk 0 succeeded, chunk 1
    # failed once and the exception propagated straight through).
    assert call_count["n"] == 2
    # The first chunk's freshly-generated result must still be on disk --
    # a later chunk's failure never discards an earlier chunk's
    # already-paid-for result.
    assert cache.load_stage1_chunk(transcript.video_id, 0) is not None


def test_refresh_stage1_and_candidates_raises_without_stage2_call_when_too_few_pass_local_filter(monkeypatch):
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 20.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 50.0)
    transcript = _long_transcript(minutes=1)
    # Freshly "regenerated" Stage1 candidates that still don't clear the
    # current quality filter -- Stage2 must never be reached (Anthropic
    # API calls = 0 for this failure).
    weak_candidates = [_raw_candidate(0, 0, opening_hook_strength=90)]

    monkeypatch.setattr(clip_selector, "extract_candidates_for_chunk", lambda *a, **k: weak_candidates)
    monkeypatch.setattr(
        clip_selector, "rank_candidates",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Stage2 must not be called")),
    )

    with pytest.raises(RuntimeError, match="現在の品質基準を満たす候補"):
        clip_selector.refresh_stage1_and_candidates(transcript, "タイトル")


def test_refresh_stage1_and_candidates_does_not_overwrite_stage2_cache_when_finalize_fails(monkeypatch):
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _long_transcript(minutes=1)

    # Existing Stage2 cache from a prior successful run -- must survive
    # completely untouched if this refresh's final gate fails.
    old_stage2 = [_raw_candidate(0, 2, opening_hook_strength=90)] * 3
    cache.save_stage2(transcript.video_id, old_stage2)

    fresh_candidates = [_raw_candidate(0, 2, opening_hook_strength=90) for _ in range(3)]
    monkeypatch.setattr(clip_selector, "extract_candidates_for_chunk", lambda *a, **k: fresh_candidates)
    # Stage2 ranking runs (costs 1 API call) but returns only 2 valid ids --
    # _rank_finalize_and_cache must raise before ever calling
    # cache.save_stage2, leaving the old cache exactly as it was.
    monkeypatch.setattr(clip_selector, "rank_candidates", lambda id_map, t, title: list(id_map.keys())[:2])

    with pytest.raises(RuntimeError, match="有効な候補ID"):
        clip_selector.refresh_stage1_and_candidates(transcript, "タイトル")

    reloaded = cache.load_stage2(transcript.video_id)
    assert len(reloaded) == 3
    assert all(c.segments[0].end_segment_id == 2 for c in reloaded)  # untouched old cache
