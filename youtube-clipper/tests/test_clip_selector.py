import pytest
from pydantic import ValidationError

from podcast_clipper import boundary, cache, clip_selector, config
from podcast_clipper.clip_selector import (
    Stage1MaterialOutput,
    Stage1MaterialSegmentOutput,
    Stage1Output,
    Stage2CandidateOutput,
    Stage2Output,
    Stage2SegmentOutput,
)
from podcast_clipper.models import (
    RawClipCandidate,
    RawMaterial,
    RawMaterialSegment,
    RawUsedSegment,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)


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


def _raw_material(start_id, end_id, material_type="hook", usefulness_score=80):
    return RawMaterial(
        material_type=material_type,
        segments=[RawMaterialSegment(start_segment_id=start_id, end_segment_id=end_id)],
        usefulness_score=usefulness_score,
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
    assert "前半で見つかった素材だけで枠を使い切り" in text
    assert "後半" in text


def test_extract_candidates_prompt_forbids_padding_weak_candidates_to_fill_six():
    text = _extract_candidates_prompt_text()
    assert "件数を埋める必要はありません" in text


def test_extract_candidates_prompt_forbids_near_duplicate_candidates():
    text = _extract_candidates_prompt_text()
    assert "複数枠に並べないでください" in text


def test_extract_candidates_prompt_states_stage1_is_recall_not_final_selection():
    """Documents the Stage1/Stage2 role split: Stage1 casts a wide net of
    single-purpose materials, Stage2 (seeing the pooled materials from
    every chunk) assembles and picks the final best-3."""
    text = _extract_candidates_prompt_text()
    assert "完成したShorts候補を組み立てる係ではありません" in text


def test_extract_candidates_prompt_allows_single_purpose_material():
    """Stage1/Stage2 redesign: a material no longer needs to bundle a
    hook+reason+example into one candidate to be useful -- a standalone
    hook-only material (segments of length 1) is explicitly allowed."""
    text = _extract_candidates_prompt_text()
    assert "素材は完成したShorts構成である必要はありません" in text
    assert "hookになり得る発話" in text


def test_extract_candidates_prompt_does_not_force_duration_target():
    """Stage1/Stage2 redesign: Stage1 must not skip/shape candidates to
    fit 20-50s -- that's now Stage2's responsibility."""
    text = _extract_candidates_prompt_text()
    assert "無理に収めようとしないでください" in text
    assert "尺の最終調整は後段のStage2の責務です" in text


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


def test_rank_and_finalize_prompt_documents_junction_safety():
    text = _rank_and_finalize_prompt_text()
    assert "カット接続の自然さ" in text
    assert "連続周回をする場合は" in text


def test_rank_and_finalize_prompt_allows_recombining_across_materials():
    """The core Stage1/Stage2 redesign: Stage2 may build a final candidate
    out of segments drawn from *different* materials, not just pick or
    exclude a whole Stage1 candidate as-is."""
    text = _rank_and_finalize_prompt_text()
    assert "異なる素材のsegmentを組み合わせて" in text


def test_rank_and_finalize_prompt_forbids_fabricated_segment_ids():
    text = _rank_and_finalize_prompt_text()
    assert "実在する`segment_id`の組み合わせのみを使うこと" in text
    assert "存在しない`segment_id`を作文しないこと" in text


def test_rank_and_finalize_prompt_documents_duration_target():
    text = _rank_and_finalize_prompt_text()
    assert "20〜50秒" in text
    assert "50秒を超える設計は絶対に出さないこと" in text


def test_rank_and_finalize_prompt_documents_end_anchor_text():
    """end_anchor_text is the one genuinely new field vs Stage1's own
    schema -- the prompt must explain it symmetrically to start_anchor_
    text, including the same word-boundary/no-fabrication constraints."""
    text = _rank_and_finalize_prompt_text()
    assert "end_anchor_text" in text
    assert "start_anchor_text" in text
    assert "word境界に一致する必要がある" in text


def test_rank_and_finalize_prompt_never_forces_three_designs():
    text = _rank_and_finalize_prompt_text()
    assert "無理に3件埋めず" in text


# --- prompt content: start_anchor_text trim + segment reordering ----------


def test_extract_candidates_prompt_documents_start_anchor_text():
    text = _extract_candidates_prompt_text()
    assert "start_anchor_text" in text
    assert "完全一致" in text


def test_extract_candidates_prompt_forbids_mid_word_anchor_starts():
    text = _extract_candidates_prompt_text()
    assert "単語の途中" in text
    assert "word境界に一致する必要がある" in text


def test_extract_candidates_prompt_no_longer_reorders_within_stage1():
    """Stage1/Stage2 material-contract fix: materials are single-purpose
    (no internal role structure), so the old "reorder segments across
    roles within one Stage1 candidate" mechanism no longer applies --
    that responsibility now belongs entirely to Stage2 (which recombines
    segments across *different* materials, see rank_and_finalize.md)."""
    text = _extract_candidates_prompt_text()
    assert "候補内の並び替え" not in text
    assert "候補内でのsegmentの再利用" not in text


def test_extract_candidates_prompt_scores_hook_usefulness_post_trim():
    """usefulness_score for a hook material must be scored against what
    actually plays first after anchor trim, not the raw untrimmed
    segment text."""
    text = _extract_candidates_prompt_text()
    assert "トリム後のテキスト" in text


def test_extract_candidates_prompt_scopes_hook_criteria_to_hook_material_type():
    """The root-cause fix: the strong-opening criteria/reject list must be
    explicitly scoped to material_type: hook only, and reason/example/
    context/payoff materials must have their own, lighter quality bar --
    otherwise a reason material like the fuel-cut example would be
    rejected by hook-strength rules before ever reaching Stage2."""
    text = _extract_candidates_prompt_text()
    assert "この基準は`material_type: hook`の素材にのみ適用されます" in text
    assert "reason/example/context/payoff素材の基準" in text
    assert "穏やかな説明調であること自体、あるいはhookとしては弱いことは、これらのmaterial_typeでは不合格理由にしないでください" in text


def test_extract_candidates_prompt_documents_material_type_enum():
    text = _extract_candidates_prompt_text()
    for material_type in ("hook", "reason", "example", "context", "payoff"):
        assert f"`{material_type}`" in text
    assert "material_type" in text


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
    # (which becomes job.error, already rendered to the user). Weak hook
    # strength is no longer screened at the material-prefilter stage (only
    # the final local gate on Stage2's designed output still checks it --
    # see evaluate_local_candidate's hook_strength_below_80), so this
    # simulates Stage2 designing its final candidate straight from the
    # weak material unchanged.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _long_transcript(minutes=1)
    materials = [_raw_material(0, 0)]
    monkeypatch.setattr(clip_selector, "run_stage1", lambda *a, **k: materials)
    monkeypatch.setattr(
        clip_selector, "design_final_candidates",
        lambda materials, t, title: [_raw_candidate(0, 0, opening_hook_strength=50)],
    )

    with pytest.raises(RuntimeError) as exc_info:
        clip_selector.refresh_stage1_and_candidates(transcript, "タイトル")

    message = str(exc_info.value)
    assert "【診断】" in message
    assert "評価対象候補: 1件" in message
    assert "hook強度不足" in message


def test_diagnose_local_filter_K_makes_zero_api_calls(monkeypatch):
    # K: relies on this module's autouse _forbid_real_anthropic_client
    # fixture (poisons anthropic.Anthropic()) plus an explicit guard on
    # run_stage1/extract_candidates_for_chunk -- diagnose_local_filter
    # must never reach either.
    transcript = _long_transcript(minutes=1)
    cache.save_stage1_chunk(transcript.video_id, 0, [_raw_material(0, 0, usefulness_score=90)])

    def _forbidden(*a, **k):
        raise AssertionError("diagnose_local_filter must never call the Stage1 API")

    monkeypatch.setattr(clip_selector, "run_stage1", _forbidden)
    monkeypatch.setattr(clip_selector, "extract_candidates_for_chunk", _forbidden)

    evaluations = clip_selector.diagnose_local_filter(transcript)
    assert len(evaluations) >= 1
    assert all(isinstance(e, clip_selector.MaterialUsabilityResult) for e in evaluations)
    assert evaluations[0].usable is True
    assert evaluations[0].reason is None


def test_diagnose_local_filter_L_raises_clearly_without_cache():
    transcript = _long_transcript(minutes=1)
    transcript.video_id = "vid-never-cached-for-diagnosis"

    with pytest.raises(RuntimeError, match="Stage1素材キャッシュ"):
        clip_selector.diagnose_local_filter(transcript)


def test_candidate_schema_version_still_11():
    # This round redesigns Stage1 from candidate-shaped output
    # (Stage1CandidateOutput: hook_type/opening_hook_strength/score
    # required on every item) into a genuine material contract
    # (Stage1MaterialOutput: material_type/usefulness_score) -- a real
    # Structured Outputs schema change on the Stage1 side, so the version
    # was bumped once more (10->11); it must not drift further within
    # this round.
    assert config.CANDIDATE_SCHEMA_VERSION == 11


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

    trimmed, skip_reason = clip_selector._try_opening_trim_repair(candidate, transcript)
    assert trimmed is not None
    assert skip_reason is None
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
    assert result.repair_method == "prepend_previous_1"
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
    assert "prepend_previous_1" in result.attempted_repair_methods


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
    # H: no duration_too_long repair variant ever references a transcript
    # segment_id range Stage1 didn't already choose -- every kept segment
    # is one of the original (start_segment_id, end_segment_id) pairs
    # verbatim, never a new, narrower id range. This still holds after
    # body_opening_trim/body_ending_trim were added: those only ever
    # narrow *playback* within an existing segment id via start_anchor_
    # text/end_anchor_text (verified against real word timestamps,
    # landing only on already-confidently-complete clause boundaries --
    # see models.find_natural_end_trim_points/find_sequential_removable_
    # prefix_word), never by referencing a different/narrower segment_id
    # range or guessing a cut point -- so this test's id-range assertions
    # (which don't inspect anchor fields at all) remain the right
    # invariant: no variant, of any method, ever fabricates a segment_id
    # range Stage1 didn't choose.
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
    # Raised from 8 to 12 when body_opening_trim/body_ending_trim were
    # added (duration_too_long alone can now generate more small, already-
    # safe variants) -- still a small, fixed bound, never unbounded.
    assert clip_selector._MAX_LOCAL_REPAIR_VARIANTS <= 12


# --- body_opening_trim / body_ending_trim: real-machine incident -------
# --- (7 Stage1 candidates, only 2 passed the local filter; 4 of the ------
# --- other 5 failed on duration_too_long, and the only existing repair --
# --- (dropping a whole non-hook segment) swung duration far more than --
# --- needed -- e.g. 51.0s -> 20.7s -- so the result then failed some ----
# --- other check a smaller trim would never have triggered) -------------


def _duration_repair_hook_words():
    return [TranscriptWord(start=0.0, end=5.0, text="結論から言うと、ギアを入れてアクセルオフの方が燃費がいいです。")]


def test_repair_A_body_ending_trim_rescues_barely_over_50s_candidate(monkeypatch):
    # A: a candidate a little over 50s, whose body has one independent,
    # confidently-complete trailing sentence -- the smallest possible
    # trim (drop only that sentence) is enough, so body_ending_trim must
    # win over the far larger drop_context_segment.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 20.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 50.0)
    hook_words = _duration_repair_hook_words()
    body_words = [
        TranscriptWord(start=5.3, end=46.3, text="その理由は減速時の燃料カットが働くからです。"),
        TranscriptWord(start=46.3, end=51.3, text="以上が本日の内容でした。"),
    ]
    transcript = Transcript(
        video_id="repairEndTrimA", language="ja",
        segments=[
            TranscriptSegment(id=0, start=0.0, end=5.0, text=hook_words[0].text, words=hook_words),
            TranscriptSegment(
                id=1, start=5.3, end=51.3, text="".join(w.text for w in body_words), words=body_words,
            ),
        ],
    )
    candidate = RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0),
            RawUsedSegment(role="context", start_segment_id=1, end_segment_id=1),
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )

    original = clip_selector.evaluate_local_candidate(candidate, transcript)
    assert original.reason == "duration_too_long"

    result = clip_selector.evaluate_local_candidate_with_repair(candidate, transcript)
    assert result.accepted is True
    assert result.repair_method == "body_ending_trim"
    assert result.original_reason == "duration_too_long"
    assert 20.0 <= result.duration_sec <= 50.0
    resolved = boundary.resolve_candidate(result.candidate, transcript, candidate_id="c1")
    assert "以上が本日の内容でした。" not in resolved.segments[1].text


def test_repair_B_body_opening_trim_rescues_barely_over_50s_candidate(monkeypatch):
    # B: a candidate a little over 50s, whose non-hook segment opens on a
    # known removable weak lead-in ("よくある話が") -- shaving only that
    # off is enough, reusing find_sequential_removable_prefix_word (the
    # same primitive opening_trim already uses for the hook) on a
    # non-hook segment for the first time.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 20.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 50.0)
    hook_words = _duration_repair_hook_words()
    body_words = [
        TranscriptWord(start=5.3, end=10.3, text="よくある話が"),
        TranscriptWord(start=10.3, end=53.3, text="その理由は減速時の燃料カットが働くからです。"),
    ]
    transcript = Transcript(
        video_id="repairEndTrimB", language="ja",
        segments=[
            TranscriptSegment(id=0, start=0.0, end=5.0, text=hook_words[0].text, words=hook_words),
            TranscriptSegment(
                id=1, start=5.3, end=53.3, text="".join(w.text for w in body_words), words=body_words,
            ),
        ],
    )
    candidate = RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0),
            RawUsedSegment(role="context", start_segment_id=1, end_segment_id=1),
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )

    original = clip_selector.evaluate_local_candidate(candidate, transcript)
    assert original.reason == "duration_too_long"

    result = clip_selector.evaluate_local_candidate_with_repair(candidate, transcript)
    assert result.accepted is True
    assert result.repair_method == "body_opening_trim"
    assert result.original_reason == "duration_too_long"
    assert 20.0 <= result.duration_sec <= 50.0
    resolved = boundary.resolve_candidate(result.candidate, transcript, candidate_id="c1")
    assert resolved.segments[1].text.startswith("その理由は")
    assert "よくある話が" not in resolved.segments[1].text


def test_repair_C_body_ending_trim_tries_a_larger_cut_when_the_smallest_is_not_enough(monkeypatch):
    # C: whole-segment drop would make this candidate too short (real-
    # machine shape: "丸ごと削除では短くなりすぎる"), but the body has THREE
    # independent sentences -- the smallest end-trim (dropping only the
    # last) still leaves it too long, so a larger end-trim (dropping the
    # last two) must be tried next, smallest-to-largest.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 20.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 50.0)
    hook_words = _duration_repair_hook_words()
    body_words = [
        TranscriptWord(start=5.3, end=35.3, text="その理由は減速時の燃料カットが働くからです。"),
        TranscriptWord(start=35.3, end=60.3, text="他にもいくつか要因があります。"),
        TranscriptWord(start=60.3, end=70.3, text="以上が本日の内容でした。"),
    ]
    transcript = Transcript(
        video_id="repairEndTrimC", language="ja",
        segments=[
            TranscriptSegment(id=0, start=0.0, end=5.0, text=hook_words[0].text, words=hook_words),
            TranscriptSegment(
                id=1, start=5.3, end=70.3, text="".join(w.text for w in body_words), words=body_words,
            ),
        ],
    )
    candidate = RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0),
            RawUsedSegment(role="context", start_segment_id=1, end_segment_id=1),
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )

    result = clip_selector.evaluate_local_candidate_with_repair(candidate, transcript)
    assert result.accepted is True
    assert result.repair_method == "body_ending_trim"
    assert 20.0 <= result.duration_sec <= 50.0
    # The smallest end-trim variant (drop only the 3rd sentence) must have
    # been tried and failed *for being too long* before the larger one
    # (drop the 2nd and 3rd) succeeded -- confirms the smallest-change-
    # first ordering, not a lucky first guess.
    end_trim_attempts = [a for a in result.repair_attempts if a.method == "body_ending_trim"]
    assert len(end_trim_attempts) == 2
    assert end_trim_attempts[0].reject_reason == "duration_too_long"
    assert end_trim_attempts[1].accepted is True
    resolved = boundary.resolve_candidate(result.candidate, transcript, candidate_id="c1")
    assert resolved.segments[1].text == "その理由は減速時の燃料カットが働くからです。"


def test_repair_duration_too_long_never_guesses_without_a_natural_boundary(monkeypatch):
    # D/E: the body is a single, real, continuous sentence with no earlier
    # complete-sentence boundary at all (real-machine shape: the only
    # "reason" content spans the whole excess duration) -- there is no
    # safe place to cut, so body_ending_trim/body_opening_trim must both
    # decline (never approximate a cut point, never sacrifice the only
    # semantic content to hit a duration target), and since the whole-
    # segment drop also fails (it would leave only the 5s hook, under the
    # 20s floor), the candidate stays correctly rejected -- never silently
    # accepted with a mid-sentence cut, never silently missing its only
    # reason-bearing content.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 20.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 50.0)
    hook_words = _duration_repair_hook_words()
    body_words = [
        TranscriptWord(
            start=5.3, end=51.3, text="その理由は減速時の燃料カットが働くことで燃費が向上するからです。",
        ),
    ]
    transcript = Transcript(
        video_id="repairEndTrimD", language="ja",
        segments=[
            TranscriptSegment(id=0, start=0.0, end=5.0, text=hook_words[0].text, words=hook_words),
            TranscriptSegment(
                id=1, start=5.3, end=51.3, text="".join(w.text for w in body_words), words=body_words,
            ),
        ],
    )
    candidate = RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0),
            RawUsedSegment(role="context", start_segment_id=1, end_segment_id=1),
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )

    result = clip_selector.evaluate_local_candidate_with_repair(candidate, transcript)
    assert result.accepted is False
    by_method = {a.method: a for a in result.repair_attempts}
    assert by_method["body_opening_trim"].generated is False
    assert by_method["body_opening_trim"].generation_skip_reason == "no_removable_prefix"
    assert by_method["body_ending_trim"].generated is False
    assert by_method["body_ending_trim"].generation_skip_reason == "no_natural_end_trim_point"


def test_repair_F_duration_too_long_body_without_word_timestamps_never_guesses(monkeypatch):
    # F: no word-timestamp data at all on the over-long body segment --
    # body_opening_trim/body_ending_trim must both decline cleanly (no
    # crash, no guessed cut point), exactly like every other word-
    # timestamp-dependent repair in this module.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 20.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 50.0)
    hook_words = _duration_repair_hook_words()
    transcript = Transcript(
        video_id="repairEndTrimF", language="ja",
        segments=[
            TranscriptSegment(id=0, start=0.0, end=5.0, text=hook_words[0].text, words=hook_words),
            TranscriptSegment(
                id=1, start=5.3, end=56.3,
                text="その理由は減速時の燃料カットが働くことで燃費が向上するからです。以上が本日の内容でした。",
                words=[],
            ),
        ],
    )
    candidate = RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0),
            RawUsedSegment(role="context", start_segment_id=1, end_segment_id=1),
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )

    result = clip_selector.evaluate_local_candidate_with_repair(candidate, transcript)
    assert result.accepted is False
    by_method = {a.method: a for a in result.repair_attempts}
    assert by_method["body_opening_trim"].generated is False
    assert by_method["body_ending_trim"].generated is False


def test_repair_G_speech_restart_in_another_segment_still_rejects_after_duration_fixed(monkeypatch):
    # G: real-machine "candidate2" shape -- duration_too_long is fixable
    # (body_ending_trim gets it back under 50s), but a separate segment
    # still contains a speech restart. evaluate_local_candidate re-runs
    # every check on the repaired variant (item 6: never a bespoke, more
    # lenient judgment for a repaired candidate), so this must stay
    # rejected overall, with the diagnostic showing *why* the repaired
    # variant itself still failed.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 20.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 50.0)
    hook_words = _duration_repair_hook_words()
    context_words = [
        TranscriptWord(start=5.3, end=45.3, text="その理由は減速時の燃料カットが働くからです。"),
        TranscriptWord(start=45.3, end=50.3, text="以上が補足です。"),
    ]
    answer_words = [
        TranscriptWord(start=50.6, end=51.0, text="ホンダも取扱説明書の中に、"),
        TranscriptWord(start=51.0, end=51.4, text="走行中にNレンジで下るというのは、"),
        TranscriptWord(start=51.4, end=51.8, text="Nレンジにすると、"),
        TranscriptWord(start=51.8, end=52.2, text="エンジンブレーキが効かなくなって、"),
        TranscriptWord(start=52.2, end=52.6, text="思わぬ事故の原因になるので、"),
        TranscriptWord(start=52.6, end=53.0, text="急な坂道では注意が必要です。"),
    ]
    transcript = Transcript(
        video_id="repairEndTrimG", language="ja",
        segments=[
            TranscriptSegment(id=0, start=0.0, end=5.0, text=hook_words[0].text, words=hook_words),
            TranscriptSegment(
                id=1, start=5.3, end=50.3, text="".join(w.text for w in context_words), words=context_words,
            ),
            TranscriptSegment(
                id=2, start=50.6, end=53.0, text="".join(w.text for w in answer_words), words=answer_words,
            ),
        ],
    )
    candidate = RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0),
            RawUsedSegment(role="context", start_segment_id=1, end_segment_id=1),
            RawUsedSegment(role="answer", start_segment_id=2, end_segment_id=2),
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )

    result = clip_selector.evaluate_local_candidate_with_repair(candidate, transcript)
    assert result.accepted is False
    end_trim_attempt = next(a for a in result.repair_attempts if a.method == "body_ending_trim")
    assert end_trim_attempt.generated is True
    assert 20.0 <= end_trim_attempt.duration_sec <= 50.0  # duration itself was fixed
    assert end_trim_attempt.reject_reason == "speech_restart"
    assert end_trim_attempt.disfluency_detail == "answer: Nレンジ"


def test_repair_K_end_trim_overshoot_into_too_short_does_not_get_rescued(monkeypatch):
    # K: the only available end-trim point overshoots past the 20s floor
    # -- that variant must be rejected via the ordinary duration_too_short
    # check, exactly like any other variant, never silently accepted for
    # having "fixed" duration_too_long.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 20.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 50.0)
    hook_words = _duration_repair_hook_words()
    body_words = [
        TranscriptWord(start=5.3, end=15.3, text="先に短い結論だけ言います。"),
        TranscriptWord(start=15.3, end=55.3, text="細かい追加の話が続きます。"),
    ]
    transcript = Transcript(
        video_id="repairEndTrimK", language="ja",
        segments=[
            TranscriptSegment(id=0, start=0.0, end=5.0, text=hook_words[0].text, words=hook_words),
            TranscriptSegment(
                id=1, start=5.3, end=55.3, text="".join(w.text for w in body_words), words=body_words,
            ),
        ],
    )
    candidate = RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0),
            RawUsedSegment(role="context", start_segment_id=1, end_segment_id=1),
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )

    result = clip_selector.evaluate_local_candidate_with_repair(candidate, transcript)
    assert result.accepted is False
    end_trim_attempt = next(a for a in result.repair_attempts if a.method == "body_ending_trim")
    assert end_trim_attempt.generated is True
    assert end_trim_attempt.reject_reason == "duration_too_short"


def test_repair_body_ending_trim_never_touches_the_hook_segment(monkeypatch):
    # Item 4: the hook is never end-trimmed by this repair, even when it
    # is the only segment with a natural internal boundary -- only its
    # own start is ever mechanically trimmed (opening_trim/disfluency_
    # trim, for unrelated reasons). A duration_too_long candidate whose
    # hook itself contains multiple sentences must still fail to end-trim
    # if no *non-hook* segment has one.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 20.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 50.0)
    hook_words = [
        TranscriptWord(start=0.0, end=40.0, text="結論から言うと、ギアを入れてアクセルオフの方が燃費がいいです。"),
        TranscriptWord(start=40.0, end=51.0, text="以上が本日の結論です。"),
    ]
    transcript = Transcript(
        video_id="repairEndTrimHookProtected", language="ja",
        segments=[
            TranscriptSegment(id=0, start=0.0, end=51.0, text="".join(w.text for w in hook_words), words=hook_words),
        ],
    )
    candidate = RawClipCandidate(
        hook_type="strong_take",
        segments=[RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0)],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )

    result = clip_selector.evaluate_local_candidate_with_repair(candidate, transcript)
    assert result.accepted is False
    by_method = {a.method: a for a in result.repair_attempts}
    # Single-segment candidate -- nothing non-hook to trim or drop at all.
    assert by_method["body_ending_trim"].generation_skip_reason == "no_natural_end_trim_point"
    assert by_method["body_opening_trim"].generation_skip_reason == "no_removable_prefix"
    assert by_method["drop_context_segment"].generation_skip_reason == "single_segment_candidate"


def test_repair_hook_payoff_exact_repeat_coexists_with_body_ending_trim(monkeypatch):
    # J regression: the existing, intentionally-allowed hook->context->
    # payoff exact-repeat structure must keep working once body_opening_
    # trim/body_ending_trim are tried first for duration_too_long -- a
    # trim on the (non-hook, non-payoff) context segment must not disturb
    # the hook/payoff exact-repeat relationship at all.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 20.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 50.0)
    hook_words = [TranscriptWord(start=0.0, end=5.0, text="冷却効率を上げるために重量を増やすというのはアンチパターンになると思います。")]
    context_words = [
        TranscriptWord(start=5.3, end=42.3, text="真冬のサーキットで2、3周しかアタックをしません。"),
        TranscriptWord(start=42.3, end=53.3, text="以上が背景の説明です。"),
    ]
    transcript = Transcript(
        video_id="repairEndTrimHookRepeat", language="ja",
        segments=[
            TranscriptSegment(id=0, start=0.0, end=5.0, text=hook_words[0].text, words=hook_words),
            TranscriptSegment(
                id=1, start=5.3, end=53.3, text="".join(w.text for w in context_words), words=context_words,
            ),
        ],
    )
    candidate = RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0),
            RawUsedSegment(role="context", start_segment_id=1, end_segment_id=1),
            RawUsedSegment(role="payoff", start_segment_id=0, end_segment_id=0),
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )

    result = clip_selector.evaluate_local_candidate_with_repair(candidate, transcript)
    assert result.accepted is True
    assert result.repair_method == "body_ending_trim"
    assert [s.role for s in result.candidate.segments] == ["hook", "context", "payoff"]
    assert result.candidate.segments[0].start_segment_id == result.candidate.segments[-1].start_segment_id


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


# --- round 7: real-machine incident -- "候補2" got repaired into "あとGR86
# もそうだと思うんですけども..." which the diagnostic reported as accepted,
# but is still a context-dependent opening (an enumeration continuation
# whose first item, "ZN6-86であったり...", is further back and not in the
# clip). Fixes: (1) models.CONTEXT_DEPENDENT_OPENING_PREFIXES now also
# catches leading "あと"/"それから"/"さらに", so a repair landing on one is
# rejected like any other dependent opening; (2) the previous-segment
# prepend repair now tries up to 3 real, chronologically-preceding
# segments (prepend_previous_1/2/3) instead of only ever the immediately
# preceding one, so it can actually reach the real antecedent; (3) full
# per-repair-variant diagnostics (RepairAttemptDiagnostic) so a real-
# machine report shows *why* each attempted method failed (or was never
# even generated), not just its name. -----------------------------------


def _candidate2_round7_transcript():
    return Transcript(
        video_id="repairR", language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=1.2,
                text="よくある話が私も乗っているZN6-86であったりです。",
                words=[
                    TranscriptWord(start=0.0, end=0.4, text="よくある話が"),
                    TranscriptWord(start=0.4, end=0.8, text="私も乗っている"),
                    TranscriptWord(start=0.8, end=1.2, text="ZN6-86であったりです。"),
                ],
            ),
            TranscriptSegment(
                id=1, start=1.5, end=2.5,
                text="あとGR86もそうだと思うんですけども。",
                words=[TranscriptWord(start=1.5, end=2.5, text="あとGR86もそうだと思うんですけども。")],
            ),
            TranscriptSegment(
                id=2, start=2.8, end=5.0,
                text="これのクラッチ交換の際にメタルクラッチを入れるとミッションが壊れやすくなるっていうのはよく言われてます。",
                words=[TranscriptWord(
                    start=2.8, end=5.0,
                    text="これのクラッチ交換の際にメタルクラッチを入れるとミッションが壊れやすくなるっていうのはよく言われてます。",
                )],
            ),
        ],
    )


def _candidate2_round7_raw():
    return RawClipCandidate(
        hook_type="surprising_fact",
        segments=[RawUsedSegment(role="hook", start_segment_id=2, end_segment_id=2)],
        hook_text="h", opening_hook_strength=83, title="", description="",
        score=83, reasoning="", caveats="",
    )


def test_repair_R_ato_prefix_is_context_dependent():
    # R: "あとGR86も..." must itself be detected as a context-dependent
    # opening -- the negative signal a repair result is re-checked
    # against -- not accepted as independent Japanese just because it
    # happens to be grammatically well-formed.
    assert clip_selector._looks_context_dependent_opening("あとGR86もそうだと思うんですけども。") is True
    # A mid-sentence "あと" must never trigger this -- only a leading one.
    assert clip_selector._looks_context_dependent_opening("ZN6-86であったりあとはBRZです。") is False


def test_repair_S_candidate2_prepend_1_still_rejected_then_prepend_2_accepted(monkeypatch):
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _candidate2_round7_transcript()
    candidate = _candidate2_round7_raw()

    result = clip_selector.evaluate_local_candidate_with_repair(candidate, transcript)
    assert result.accepted is True
    assert result.repair_method == "prepend_previous_2"
    assert result.original_reason == "context_dependent_opening"

    # prepend_previous_1 must have been attempted and rejected -- landing
    # on "あとGR86..." is not silently skipped, it's tried and fails.
    attempt_1 = next(a for a in result.repair_attempts if a.method == "prepend_previous_1")
    assert attempt_1.generated is True
    assert attempt_1.accepted is False
    assert attempt_1.reject_reason == "context_dependent_opening"
    assert attempt_1.junction_reason == "hook_context_dependent"

    resolved = boundary.resolve_candidate(result.candidate, transcript, candidate_id="c1")
    assert resolved.segments[0].text.startswith("ZN6-86であったり")


def test_repair_T_minimal_change_preferred_when_multiple_would_accept(monkeypatch):
    # T: when a shallower repair already resolves the dependent opening,
    # a deeper (larger-change) lookback must never be preferred over it --
    # generate_local_repair_variants/_build_repair_candidates deliberately
    # orders shallow-to-deep so "first accepted wins" already implements
    # this without a separate scoring step.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = Transcript(
        video_id="repairT", language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=1.0, text="今日の天気の話をします。",
                words=[TranscriptWord(start=0.0, end=1.0, text="x")],
            ),
            TranscriptSegment(
                id=1, start=1.3, end=2.3, text="ZN6-86であったりGR86です。",
                words=[TranscriptWord(start=1.3, end=2.3, text="x")],
            ),
            TranscriptSegment(
                id=2, start=2.6, end=5.0,
                text="これのクラッチ交換の際にメタルクラッチを入れるとミッションが壊れやすくなるっていうのはよく言われてます。",
                words=[TranscriptWord(start=2.6, end=5.0, text="x")],
            ),
        ],
    )
    candidate = RawClipCandidate(
        hook_type="surprising_fact",
        segments=[RawUsedSegment(role="hook", start_segment_id=2, end_segment_id=2)],
        hook_text="h", opening_hook_strength=83, title="", description="",
        score=83, reasoning="", caveats="",
    )

    result = clip_selector.evaluate_local_candidate_with_repair(candidate, transcript)
    assert result.accepted is True
    assert result.repair_method == "prepend_previous_1"
    # prepend_previous_2 was never even evaluated -- repair stops at the
    # first accepted (shallowest) variant rather than searching further.
    assert "prepend_previous_2" not in result.attempted_repair_methods


def test_repair_U_prepend_lookback_never_exceeds_3_segments(monkeypatch):
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 1000.0)
    segments = [
        TranscriptSegment(
            id=i, start=float(i), end=i + 0.8, text=f"話題{i}についてです。",
            words=[TranscriptWord(start=float(i), end=i + 0.8, text="x")],
        )
        for i in range(5)
    ] + [
        TranscriptSegment(
            id=5, start=5.0, end=5.8, text="これのクラッチ交換については以上です。",
            words=[TranscriptWord(start=5.0, end=5.8, text="x")],
        )
    ]
    transcript = Transcript(video_id="repairU", language="ja", segments=segments)
    candidate = RawClipCandidate(
        hook_type="surprising_fact",
        segments=[RawUsedSegment(role="hook", start_segment_id=5, end_segment_id=5)],
        hook_text="h", opening_hook_strength=83, title="", description="",
        score=83, reasoning="", caveats="",
    )

    built = clip_selector._build_repair_candidates(candidate, transcript, "context_dependent_opening")
    methods = [r.method for r in built]
    assert "prepend_previous_1" in methods
    assert "prepend_previous_2" in methods
    assert "prepend_previous_3" in methods
    assert not any(m.startswith("prepend_previous_4") for m in methods)
    assert clip_selector._MAX_PREPEND_LOOKBACK_SEGMENTS == 3


def test_repair_V_prepend_lookback_respects_time_cap(monkeypatch):
    # V: a segment technically exists far enough back, but pulling it in
    # would reach well beyond a reasonable "the antecedent is nearby"
    # window -- reported as skipped, not silently generated.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 1000.0)
    transcript = Transcript(
        video_id="repairV", language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=1.0, text="遠い昔の話題です。",
                words=[TranscriptWord(start=0.0, end=1.0, text="x")],
            ),
            TranscriptSegment(
                id=1, start=30.0, end=31.0, text="近い話題です。",
                words=[TranscriptWord(start=30.0, end=31.0, text="x")],
            ),
            TranscriptSegment(
                id=2, start=31.3, end=32.3, text="これのクラッチ交換については以上です。",
                words=[TranscriptWord(start=31.3, end=32.3, text="x")],
            ),
        ],
    )
    candidate = RawClipCandidate(
        hook_type="surprising_fact",
        segments=[RawUsedSegment(role="hook", start_segment_id=2, end_segment_id=2)],
        hook_text="h", opening_hook_strength=83, title="", description="",
        score=83, reasoning="", caveats="",
    )

    built = clip_selector._build_repair_candidates(candidate, transcript, "context_dependent_opening")
    by_method = {r.method: r for r in built}
    assert by_method["prepend_previous_1"].candidate is not None
    assert by_method["prepend_previous_2"].candidate is None
    assert by_method["prepend_previous_2"].skip_reason == "lookback_exceeds_time_cap"


def test_repair_W_candidate3_diagnostic_shows_per_variant_reject_reason(monkeypatch):
    # W: candidate3's real-machine report showed "repair_tried=
    # drop_context_segment → reject" with no further detail -- the
    # diagnostic must now show *why* each attempted drop variant still
    # failed (never just the method name).
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 20.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 50.0)
    transcript = Transcript(
        video_id="repairW", language="ja",
        segments=[
            TranscriptSegment(id=0, start=0.0, end=25.0, text="車を冷やしますっていうのであれば", words=[TranscriptWord(start=0.0, end=25.0, text="x")]),
            TranscriptSegment(id=1, start=30.0, end=32.7, text="無関係な話題です。", words=[TranscriptWord(start=30.0, end=32.7, text="x")]),
            TranscriptSegment(id=2, start=45.3, end=48.0, text="別の話題の説明です。", words=[TranscriptWord(start=45.3, end=48.0, text="x")]),
            TranscriptSegment(id=3, start=100.3, end=125.3, text="連続周回をする場合は違う話になります", words=[TranscriptWord(start=100.3, end=125.3, text="x")]),
        ],
    )
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
    # body_opening_trim/body_ending_trim are tried first (smallest change
    # first) but can't even be built here -- these transcript segments'
    # word lists are single placeholder "x" words with no real clause
    # structure -- so they show up as not-generated, not absent.
    assert len(result.repair_attempts) == 4
    by_method = {a.method: a for a in result.repair_attempts}
    assert by_method["body_opening_trim"].generated is False
    assert by_method["body_ending_trim"].generated is False
    assert by_method["drop_context_segment"].generated is True
    assert by_method["drop_context_segment"].reject_reason is not None
    assert by_method["drop_non_context_segment"].reject_reason == "unsafe_junction"
    assert by_method["drop_non_context_segment"].junction_reason == "jump_prev_incomplete"

    summary = clip_selector._format_diagnostic_summary([result])
    assert "repair_attempts:" in summary
    assert "drop_context_segment → reject=" in summary
    assert "drop_non_context_segment → reject=unsafe_junction junction=jump_prev_incomplete" in summary


def _candidate4_round7_transcript():
    return Transcript(
        video_id="repairX", language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=5.0,
                text="冷却効率を上げるために重量を増やすというのはアンチパターンになると思います。",
                words=[TranscriptWord(start=0.0, end=5.0, text="x")],
            ),
            TranscriptSegment(
                id=1, start=5.3, end=25.0,
                text="真冬のサーキットで2、3周しかアタックをしません。それが理由です。",
                words=[TranscriptWord(start=5.3, end=25.0, text="x")],
            ),
            TranscriptSegment(
                id=2, start=25.3, end=29.0,
                text="冷却不足という弱点はなくなるんですけども",
                words=[TranscriptWord(start=25.3, end=29.0, text="x")],
            ),
        ],
    )


def _candidate4_round7_raw():
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


def test_repair_X_candidate4_3segment_replace_final_with_hook_payoff(monkeypatch):
    # X: real-machine bug -- the old _try_hook_repeat_payoff_repair
    # guard (2 <= len(segments) < 3) silently refused any 3-segment
    # candidate, which is exactly candidate4's real shape
    # (hook+context+answer, answer incomplete) -- so no repair was even
    # attempted and the diagnostic showed nothing. The 3-segment case now
    # replaces the incomplete final segment with an exact hook repeat
    # instead of only ever appending (which would exceed the 3-segment
    # ceiling).
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 20.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 50.0)
    transcript = _candidate4_round7_transcript()
    candidate = _candidate4_round7_raw()

    result = clip_selector.evaluate_local_candidate_with_repair(candidate, transcript)
    assert result.accepted is True
    assert result.repair_method == "replace_incomplete_final_with_hook_payoff"
    assert result.original_reason == "incomplete_final_ending"
    assert [s.role for s in result.candidate.segments] == ["hook", "context", "payoff"]
    assert result.candidate.segments[-1].start_segment_id == result.candidate.segments[0].start_segment_id
    assert len(result.candidate.segments) == 3


def test_repair_Y_replace_final_over_duration_cap_keeps_rejection(monkeypatch):
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 20.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 30.0)  # tight ceiling
    transcript = _candidate4_round7_transcript()
    candidate = _candidate4_round7_raw()

    result = clip_selector.evaluate_local_candidate_with_repair(candidate, transcript)
    assert result.accepted is False
    attempt = next(a for a in result.repair_attempts if a.method == "replace_incomplete_final_with_hook_payoff")
    assert attempt.generated is True
    assert attempt.reject_reason == "duration_too_long"


def test_repair_Z_replace_final_not_generated_when_hook_ending_not_confident(monkeypatch):
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = Transcript(
        video_id="repairZ", language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=5.0,
                text="冷却効率を上げるために重量を増やすというのはアンチパターンになりますので",
                words=[TranscriptWord(start=0.0, end=5.0, text="x")],
            ),
            TranscriptSegment(
                id=1, start=5.3, end=25.0,
                text="真冬のサーキットで2、3周しかアタックをしません。それが理由です。",
                words=[TranscriptWord(start=5.3, end=25.0, text="x")],
            ),
            TranscriptSegment(
                id=2, start=25.3, end=29.0,
                text="冷却不足という弱点はなくなるんですけども",
                words=[TranscriptWord(start=25.3, end=29.0, text="x")],
            ),
        ],
    )
    candidate = RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0),
            RawUsedSegment(role="context", start_segment_id=1, end_segment_id=1),
            RawUsedSegment(role="answer", start_segment_id=2, end_segment_id=2),
        ],
        hook_text="h", opening_hook_strength=80, title="", description="",
        score=80, reasoning="", caveats="",
    )

    result = clip_selector.evaluate_local_candidate_with_repair(candidate, transcript)
    assert result.accepted is False
    attempt = next(
        a for a in result.repair_attempts if a.method == "replace_incomplete_final_with_hook_payoff"
    )
    assert attempt.generated is False
    assert attempt.generation_skip_reason == "hook_ending_not_confident"


def test_repair_AA_single_segment_hook_repeat_payoff_not_generated(monkeypatch):
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 1000.0)
    transcript = Transcript(
        video_id="repairAA", language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=5.0, text="今日は天気がとても良いので",
                words=[TranscriptWord(start=0.0, end=5.0, text="x")],
            ),
        ],
    )
    candidate = RawClipCandidate(
        hook_type="strong_take",
        segments=[RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0)],
        hook_text="h", opening_hook_strength=83, title="", description="",
        score=83, reasoning="", caveats="",
    )

    result = clip_selector.evaluate_local_candidate_with_repair(candidate, transcript)
    assert result.accepted is False
    assert result.reason == "incomplete_final_ending"
    attempt = next(a for a in result.repair_attempts if a.method == "hook_repeat_payoff")
    assert attempt.generated is False
    assert attempt.generation_skip_reason == "single_segment_candidate"


def test_repair_AB_evaluated_variant_count_never_exceeds_cap(monkeypatch):
    # AB: the _MAX_LOCAL_REPAIR_VARIANTS bound applies to *evaluated*
    # (evaluate_local_candidate-called) variants -- lower it to 1 and
    # confirm at most 1 evaluate_local_candidate call happens for repair,
    # with the rest reported as skipped (generation_skip_reason=
    # "repair_variant_cap_reached") rather than silently dropped.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 20.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 50.0)
    monkeypatch.setattr(clip_selector, "_MAX_LOCAL_REPAIR_VARIANTS", 1)
    transcript = Transcript(
        video_id="repairAB", language="ja",
        segments=[
            TranscriptSegment(id=0, start=0.0, end=25.0, text="車を冷やしますっていうのであれば", words=[TranscriptWord(start=0.0, end=25.0, text="x")]),
            TranscriptSegment(id=1, start=30.0, end=32.7, text="無関係な話題です。", words=[TranscriptWord(start=30.0, end=32.7, text="x")]),
            TranscriptSegment(id=2, start=45.3, end=48.0, text="別の話題の説明です。", words=[TranscriptWord(start=45.3, end=48.0, text="x")]),
            TranscriptSegment(id=3, start=100.3, end=125.3, text="連続周回をする場合は違う話になります", words=[TranscriptWord(start=100.3, end=125.3, text="x")]),
        ],
    )
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
    assert len(result.attempted_repair_methods) <= 1
    capped = [a for a in result.repair_attempts if a.generation_skip_reason == "repair_variant_cap_reached"]
    assert len(capped) >= 1


def test_repair_AC_round7_repairs_make_zero_api_calls(monkeypatch):
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)

    def _forbidden(*a, **k):
        raise AssertionError("repair must never call the Anthropic API")

    monkeypatch.setattr(clip_selector.structured_output, "call", _forbidden)

    transcript = _candidate2_round7_transcript()
    candidate = _candidate2_round7_raw()
    result = clip_selector.evaluate_local_candidate_with_repair(candidate, transcript)
    assert result.accepted is True

    transcript4 = _candidate4_round7_transcript()
    candidate4 = _candidate4_round7_raw()
    config.DURATION_HARD_MIN_SEC = 20.0
    config.DURATION_HARD_MAX_SEC = 50.0
    result4 = clip_selector.evaluate_local_candidate_with_repair(candidate4, transcript4)
    assert result4.accepted is True


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
    material = _raw_material(0, 1)
    candidate = _raw_candidate(0, 1, opening_hook_strength=90)
    monkeypatch.setattr(clip_selector, "run_stage1", lambda *a, **k: [material] * 4)
    monkeypatch.setattr(
        clip_selector, "design_final_candidates",
        lambda materials, t, title: [candidate for _ in materials],
    )

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
    material = _raw_material(0, 1)
    candidate = _raw_candidate(0, 1, opening_hook_strength=90)

    monkeypatch.setattr(clip_selector, "run_stage1", lambda *a, **k: [material] * 4)
    monkeypatch.setattr(
        clip_selector, "design_final_candidates",
        lambda materials, t, title: [candidate for _ in materials],
    )
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


# --- _material_is_usable: the lightweight pre-Stage2 material filter -----
# (Stage1 materials are recall-priority raw ingredient, not final
# candidates -- duration/ending-completeness/junction-safety/hook-strength
# are deliberately NOT checked here; only referential integrity and
# actively-broken speech are, since those are disqualifying no matter
# which role a piece of material eventually plays)


def test_material_is_usable_accepts_a_short_incomplete_fragment(monkeypatch):
    # D/E baseline: a material that would never pass the old full gate
    # (too short, no confident ending) must still be usable -- Stage1
    # no longer needs to produce anything duration/ending-shaped.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 20.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 50.0)
    transcript = _long_transcript(minutes=1)
    material = _raw_material(0, 0)  # a single 2-second segment
    assert clip_selector._material_is_usable(material, transcript) is True


def test_material_is_usable_rejects_disfluent_material():
    # D: a material containing a word-search/self-correction marker must
    # never reach Stage2, regardless of which role it might play.
    transcript = Transcript(
        video_id="vidMaterialDisfluent", language="ja",
        segments=[_segment(0, start=0.0, text="ちょっと表現が難しいんですけども、要するにこうです。")],
    )
    material = _raw_material(0, 0)
    assert clip_selector._material_is_usable(material, transcript) is False


def test_material_is_usable_rejects_speech_restart_material():
    # E: the real-machine restart pattern must never reach Stage2 either.
    words = [
        TranscriptWord(start=0.0, end=0.5, text="ホンダも取扱説明書の中に、"),
        TranscriptWord(start=0.5, end=1.0, text="走行中にNレンジで下るというのは、"),
        TranscriptWord(start=1.0, end=1.5, text="Nレンジにすると、"),
        TranscriptWord(start=1.5, end=2.0, text="エンジンブレーキが効かなくなって、"),
        TranscriptWord(start=2.0, end=2.5, text="思わぬ事故の原因になるので、"),
        TranscriptWord(start=2.5, end=3.0, text="急な坂道では注意が必要です。"),
    ]
    transcript = Transcript(
        video_id="vidMaterialRestart", language="ja",
        segments=[TranscriptSegment(id=0, start=0.0, end=3.0, text="".join(w.text for w in words), words=words)],
    )
    material = _raw_material(0, 0)
    assert clip_selector._material_is_usable(material, transcript) is False


def test_material_is_usable_rejects_invalid_segment_reference():
    transcript = _long_transcript(minutes=1)
    material = _raw_material(999, 999)  # segment_id doesn't exist
    assert clip_selector._material_is_usable(material, transcript) is False


def test_material_rejection_reason_names_the_reason():
    # diagnose_local_filter's MaterialUsabilityResult surfaces this reason
    # string directly -- confirm the raw helper names each rejection case,
    # not just a boolean.
    transcript = _long_transcript(minutes=1)
    assert clip_selector._material_rejection_reason(_raw_material(999, 999), transcript) == "invalid_segment_reference"
    disfluent_transcript = Transcript(
        video_id="vidMaterialRejectionReason", language="ja",
        segments=[_segment(0, start=0.0, text="ちょっと表現が難しいんですけども、要するにこうです。")],
    )
    assert clip_selector._material_rejection_reason(_raw_material(0, 0), disfluent_transcript) == "speech_disfluency"
    assert clip_selector._material_rejection_reason(_raw_material(0, 0), transcript) is None


def test_material_is_usable_accepts_reason_material_with_low_usefulness_score():
    # B: a calm, low-scoring-as-a-hook reason material must still be
    # usable -- the prefilter never looks at usefulness_score, only
    # referential integrity/disfluency/restart. This is the exact property
    # the old design broke: a reason material scored low would previously
    # have been judged against hook-strength standards and discarded.
    transcript = _long_transcript(minutes=1)
    material = _raw_material(0, 0, material_type="reason", usefulness_score=40)
    assert clip_selector._material_is_usable(material, transcript) is True


def test_material_is_usable_rejects_disfluent_reason_material():
    # D: material_type has no bearing on the disfluency check -- a reason
    # material with a word-search/self-correction marker is rejected
    # exactly like a hook material would be.
    transcript = Transcript(
        video_id="vidReasonDisfluent", language="ja",
        segments=[_segment(0, start=0.0, text="ちょっと表現が難しいんですけども、要するにこうです。")],
    )
    material = _raw_material(0, 0, material_type="reason", usefulness_score=70)
    assert clip_selector._material_is_usable(material, transcript) is False


def test_material_is_usable_rejects_speech_restart_reason_material():
    # E: same for the speech-restart check.
    words = [
        TranscriptWord(start=0.0, end=0.5, text="ホンダも取扱説明書の中に、"),
        TranscriptWord(start=0.5, end=1.0, text="走行中にNレンジで下るというのは、"),
        TranscriptWord(start=1.0, end=1.5, text="Nレンジにすると、"),
        TranscriptWord(start=1.5, end=2.0, text="エンジンブレーキが効かなくなって、"),
        TranscriptWord(start=2.0, end=2.5, text="思わぬ事故の原因になるので、"),
        TranscriptWord(start=2.5, end=3.0, text="急な坂道では注意が必要です。"),
    ]
    transcript = Transcript(
        video_id="vidReasonRestart", language="ja",
        segments=[TranscriptSegment(id=0, start=0.0, end=3.0, text="".join(w.text for w in words), words=words)],
    )
    material = _raw_material(0, 0, material_type="reason", usefulness_score=70)
    assert clip_selector._material_is_usable(material, transcript) is False


def test_stage1_stage2_recombine_hook_material_with_weak_hook_reason_material(monkeypatch):
    # Item 7 -- the single most important test for this round's fix
    # ("これが通らなければ今回の再設計は成立していない"): chunk A has only a hook
    # material ("Xの方がYより燃費が良い"); chunk B has a material that would be
    # weak as a Shorts hook but is an important reason explanation
    # ("減速時には一定条件で燃料噴射が止まるためです"). Both must survive Stage1's
    # material prefilter regardless of the reason material's weak-hook
    # shape, and Stage2 must be able to combine them into one
    # semantically-complete finished candidate that clears the unchanged
    # local hard gates.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    monkeypatch.setattr(config, "NUM_CANDIDATES", 1)
    transcript = Transcript(
        video_id="vidCrossChunkRecombine", language="ja",
        segments=[
            _segment(0, start=0.0, text="ギアを入れてアクセルオフの方がニュートラルより燃費が良いです。"),
            _segment(1, start=100.0, text="減速時には一定条件で燃料噴射が止まるためです。"),
        ],
    )
    # chunk A: a hook material only (no reason material in this chunk).
    hook_material = _raw_material(0, 0, material_type="hook", usefulness_score=90)
    # chunk B: weak as a Shorts hook (calm technical explanation, no
    # strong opening) but a real, important reason -- must still be
    # fetched by Stage1, since the material prefilter never judges hook
    # strength.
    reason_material = _raw_material(1, 1, material_type="reason", usefulness_score=55)

    # Both survive the lightweight material prefilter -- confirms the
    # weak-as-hook material is never rejected before ever reaching Stage2,
    # which is exactly the bug this round fixes.
    assert clip_selector._material_is_usable(hook_material, transcript) is True
    assert clip_selector._material_is_usable(reason_material, transcript) is True

    monkeypatch.setattr(clip_selector, "run_stage1", lambda *a, **k: [hook_material, reason_material])

    def _fake_design(materials, t, title):
        # Stage2 combines the hook material's segment with the reason
        # material's segment -- drawn from two different materials --
        # into one semantically-complete finished candidate.
        assert len(materials) == 2
        return [
            RawClipCandidate(
                hook_type="strong_take",
                segments=[
                    RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0),
                    RawUsedSegment(role="answer", start_segment_id=1, end_segment_id=1),
                ],
                hook_text="h", opening_hook_strength=90, title="", description="",
                score=90, reasoning="", caveats="",
            )
        ]

    monkeypatch.setattr(clip_selector, "design_final_candidates", _fake_design)

    result = clip_selector.select_candidates(transcript, "タイトル")

    assert len(result) == 1
    assert [s.start_segment_id for s in result[0].segments] == [0, 1]


def test_design_final_candidates_reassigns_reason_material_segment_to_hook_role(monkeypatch):
    # H: a reason material's segment doesn't need to become the finished
    # candidate's hook, but it CAN be -- Stage2, not the material's own
    # material_type, decides the final role. Confirms the conversion
    # doesn't carry any material-side "role" forward (RawMaterial has
    # none), it only uses whatever role Stage2's own output specifies.
    transcript = _long_transcript(minutes=1)
    materials = {"s1_m000": _raw_material(0, 0, material_type="reason", usefulness_score=40)}
    output = Stage2Output(
        candidates=[
            Stage2CandidateOutput(
                hook_type="strong_take",
                segments=[Stage2SegmentOutput(role="hook", start_segment_id=0, end_segment_id=0)],
                opening_hook_strength=85,
                score=85,
            )
        ]
    )
    monkeypatch.setattr(clip_selector.structured_output, "call", lambda schema_model, **kwargs: output)

    designed = clip_selector.design_final_candidates(materials, transcript, "タイトル")

    assert designed[0].segments[0].role == "hook"


# --- Stage2 final-edit-design: end-to-end through the unchanged local ----
# --- gate (evaluate_local_candidate_with_repair/finalize_candidates) -----


def test_design_finalize_A_recombines_three_materials_into_one_complete_candidate(monkeypatch):
    # A: three separate materials (a strong conclusion, its reason, and a
    # concrete example) -- Stage2 combines their segments into a single
    # final candidate design. Python never judges whether the combination
    # is a *good* edit (that's Stage2's editorial call, exercised only by
    # a real API call) -- it only verifies the result: real segment
    # references, safe junctions, correct duration.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    monkeypatch.setattr(config, "NUM_CANDIDATES", 1)  # only 1 design under test here
    transcript = Transcript(
        video_id="vidRecombine", language="ja",
        segments=[
            _segment(0, start=0.0, text="結論から言うと、ギアを入れてアクセルオフの方が燃費がいいです。"),
            _segment(1, start=5.0, text="その理由は減速時の燃料カットが働くからです。"),
            _segment(2, start=10.0, text="実際に試した人の例でも燃費が改善しています。"),
        ],
    )
    materials = [_raw_material(0, 0), _raw_material(1, 1), _raw_material(2, 2)]

    def _fake_design(m, t, title):
        return [
            RawClipCandidate(
                hook_type="strong_take",
                segments=[
                    RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0),
                    RawUsedSegment(role="answer", start_segment_id=1, end_segment_id=1),
                    RawUsedSegment(role="payoff", start_segment_id=2, end_segment_id=2),
                ],
                hook_text="h", opening_hook_strength=90, title="", description="",
                score=90, reasoning="", caveats="",
            )
        ]

    monkeypatch.setattr(clip_selector, "design_final_candidates", _fake_design)
    monkeypatch.setattr(clip_selector, "run_stage1", lambda *a, **k: materials)

    result = clip_selector._design_finalize_and_cache(materials, transcript, "タイトル")
    assert len(result) == 1
    assert [s.start_segment_id for s in result[0].segments] == [0, 1, 2]


def test_design_finalize_C_accepts_stage2_shortened_selection_from_long_material(monkeypatch):
    # C: a material over 50s on its own -- Stage2 designs a final
    # candidate using only part of it (a natural shorter sub-selection),
    # never requiring any Python-side repair.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 20.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 50.0)
    monkeypatch.setattr(config, "NUM_CANDIDATES", 1)  # only 1 design under test here
    transcript = Transcript(
        video_id="vidStage2Shortens", language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=25.0,
                text="結論から言うと、ギアを入れてアクセルオフの方が燃費がいいです。",
                words=[
                    TranscriptWord(
                        start=0.0, end=25.0, text="結論から言うと、ギアを入れてアクセルオフの方が燃費がいいです。"
                    )
                ],
            ),
            TranscriptSegment(
                id=1, start=25.3, end=85.3,
                text="長い説明が延々と続きます。" * 10,
                words=[TranscriptWord(start=25.3, end=85.3, text="長い説明が延々と続きます。" * 10)],
            ),
        ],
    )
    long_material = _raw_material(1, 1)

    def _fake_design(m, t, title):
        return [
            RawClipCandidate(
                hook_type="strong_take",
                segments=[RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0)],
                hook_text="h", opening_hook_strength=90, title="", description="",
                score=90, reasoning="", caveats="",
            )
        ]

    monkeypatch.setattr(clip_selector, "design_final_candidates", _fake_design)

    result = clip_selector._design_finalize_and_cache([long_material], transcript, "タイトル")
    assert len(result) == 1
    assert result[0].segments[0].end_segment_id == 0


def test_design_finalize_G_rejects_fabricated_segment_id(monkeypatch):
    # G: Stage2 referencing a segment_id that doesn't exist in the
    # transcript must be rejected by the unchanged referential-integrity
    # check (invalid_segment_reference is the very first check evaluate_
    # local_candidate runs) -- never accepted, never crash.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _long_transcript(minutes=1)
    materials = [_raw_material(0, 0)]

    def _fake_design(m, t, title):
        return [_raw_candidate(9999, 9999)]  # fabricated segment_id

    monkeypatch.setattr(clip_selector, "design_final_candidates", _fake_design)

    with pytest.raises(RuntimeError, match="ローカル検証を通過したのは"):
        clip_selector._design_finalize_and_cache(materials, transcript, "タイトル")


def test_design_finalize_I_rejects_out_of_bounds_duration(monkeypatch):
    # I: a Stage2 design outside [DURATION_HARD_MIN_SEC, DURATION_HARD_
    # MAX_SEC] is rejected by the unchanged duration_too_long/_too_short
    # checks -- no new logic needed, evaluate_local_candidate already
    # enforces this for whatever it's given.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 20.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 50.0)
    transcript = _long_transcript(minutes=1)
    materials = [_raw_material(0, 0)]

    def _fake_design(m, t, title):
        return [_raw_candidate(0, 0)]  # a single 2-second segment -- far under 20s

    monkeypatch.setattr(clip_selector, "design_final_candidates", _fake_design)

    with pytest.raises(RuntimeError, match="ローカル検証を通過したのは"):
        clip_selector._design_finalize_and_cache(materials, transcript, "タイトル")


def test_design_finalize_J_rejects_unsafe_junction(monkeypatch):
    # J: a Stage2 design whose segments cut together unsafely (an
    # unfinished clause hard-cut into an unrelated topic) is rejected by
    # the unchanged evaluate_candidate_junctions check.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _junction_transcript()
    materials = [_raw_material(0, 1)]

    def _fake_design(m, t, title):
        return [_junction_candidate(3, 3)]  # segment 1 -> unrelated distant segment 3

    monkeypatch.setattr(clip_selector, "design_final_candidates", _fake_design)

    with pytest.raises(RuntimeError, match="ローカル検証を通過したのは"):
        clip_selector._design_finalize_and_cache(materials, transcript, "タイトル")


# --- select_candidates: no automatic retry (item H) ----------------------


def test_select_candidates_raises_without_calling_stage2_when_no_usable_material(monkeypatch):
    # The material prefilter (_material_is_usable) only checks referential
    # integrity/disfluency/restart -- NOT duration -- so a merely short
    # material is no longer grounds to skip Stage2 (Stage2 is now
    # responsible for assembling duration-valid final candidates from
    # materials of any length). What *does* still short-circuit before
    # Stage2 is a material with actively broken speech.
    transcript = Transcript(
        video_id="vidNoUsableMaterial", language="ja",
        segments=[_segment(0, start=0.0, text="ちょっと表現が難しいんですけども、要するにこうです。")],
    )
    monkeypatch.setattr(clip_selector, "run_stage1", lambda *a, **k: [_raw_material(0, 0)])
    monkeypatch.setattr(
        clip_selector, "design_final_candidates",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Stage2 must not be called")),
    )

    with pytest.raises(RuntimeError, match="参照整合性・発話品質の基本チェック"):
        clip_selector.select_candidates(transcript, "タイトル")


def test_select_candidates_raises_when_stage2_returns_too_few_ids(monkeypatch):
    transcript = _long_transcript(minutes=1)
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    materials = [_raw_material(0, 2), _raw_material(0, 2), _raw_material(0, 2)]
    monkeypatch.setattr(clip_selector, "run_stage1", lambda *a, **k: materials)
    monkeypatch.setattr(
        clip_selector, "design_final_candidates",
        lambda materials, t, title: [_raw_candidate(0, 2) for _ in range(1)],
    )

    with pytest.raises(RuntimeError, match="ローカル検証を通過したのは"):
        clip_selector.select_candidates(transcript, "タイトル")


def test_select_candidates_raises_when_only_two_candidates_pass_semantic_closure(monkeypatch):
    # J: exactly the "only 2 pass semantic closure" shape -- Stage2 must
    # not pad ranked_candidate_ids back up to config.NUM_CANDIDATES, and
    # the existing no-auto-retry RuntimeError must fire exactly as it does
    # for any other id shortfall (referential-integrity or closure alike).
    transcript = _long_transcript(minutes=1)
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    materials = [_raw_material(0, 2), _raw_material(0, 2), _raw_material(0, 2)]
    monkeypatch.setattr(clip_selector, "run_stage1", lambda *a, **k: materials)
    monkeypatch.setattr(
        clip_selector, "design_final_candidates",
        lambda materials, t, title: [_raw_candidate(0, 2) for _ in range(2)],
    )

    with pytest.raises(RuntimeError, match="ローカル検証を通過したのは"):
        clip_selector.select_candidates(transcript, "タイトル")


def test_select_candidates_happy_path(monkeypatch):
    transcript = _long_transcript(minutes=1)
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    materials = [_raw_material(0, 2, usefulness_score=s) for s in (10, 20, 30, 40)]
    monkeypatch.setattr(clip_selector, "run_stage1", lambda *a, **k: materials)
    monkeypatch.setattr(
        clip_selector, "design_final_candidates",
        lambda materials, t, title: [_raw_candidate(0, 2) for _ in range(len(materials))],
    )

    result = clip_selector.select_candidates(transcript, "タイトル")
    assert len(result) == config.NUM_CANDIDATES


def test_select_candidates_accepts_when_stage2_excludes_one_closure_failing_candidate_of_four(monkeypatch):
    # I/K: Stage2's semantic-closure gate expresses "this candidate fails"
    # by omitting its id entirely (see rank_and_finalize.md), not by
    # ranking it lower. When 3 of 4 locally-valid candidates pass, the
    # result must contain exactly config.NUM_CANDIDATES candidates (never
    # padded back up with the excluded one), reached via exactly one
    # Stage2 call -- zero additional Anthropic API calls.
    transcript = _long_transcript(minutes=1)
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    materials = [_raw_material(0, 2, usefulness_score=s) for s in (10, 20, 30, 40)]
    monkeypatch.setattr(clip_selector, "run_stage1", lambda *a, **k: materials)

    call_count = {"n": 0}

    def _fake_design(materials, t, title):
        call_count["n"] += 1
        # Simulate Stage2 omitting one design entirely for failing semantic
        # closure -- the remaining 3 of 4 are returned, never padded back.
        assert len(materials) == 4
        return [_raw_candidate(0, 2) for _ in range(3)]

    monkeypatch.setattr(clip_selector, "design_final_candidates", _fake_design)

    result = clip_selector.select_candidates(transcript, "タイトル")

    assert len(result) == config.NUM_CANDIDATES
    assert call_count["n"] == 1


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


def test_stage1_material_output_field_set_excludes_finished_candidate_properties():
    # Stage1 structurally cannot produce hook_type/opening_hook_strength/
    # score/hook_text/title/description/reasoning/caveats any more -- those
    # all describe a *finished* candidate design, which only Stage2
    # produces (see Stage2CandidateOutput). A material's own fields are
    # material_type/segments/usefulness_score only.
    fields = set(Stage1MaterialOutput.model_fields)
    assert fields == {"material_type", "segments", "usefulness_score"}
    assert "hook_type" not in fields
    assert "opening_hook_strength" not in fields
    assert "score" not in fields
    assert "hook_text" not in fields
    assert "title" not in fields
    assert "description" not in fields
    assert "reasoning" not in fields
    assert "caveats" not in fields


def test_stage1_material_segment_output_has_no_role():
    # A material is single-purpose -- it has no internal structural
    # position (hook/context/answer/payoff) the way a finished candidate's
    # segments do. That's only ever decided by Stage2 (Stage2SegmentOutput
    # still has `role`).
    assert set(Stage1MaterialSegmentOutput.model_fields) == {
        "start_segment_id", "end_segment_id", "start_anchor_text",
    }


def test_stage2_candidate_output_field_set_is_the_only_place_finished_properties_exist():
    # Stage2's candidate/segment schemas are the only place hook_type/
    # opening_hook_strength/score/role/end_anchor_text exist -- Stage1's
    # material schema deliberately has none of these.
    assert set(Stage2CandidateOutput.model_fields) == {
        "hook_type", "segments", "opening_hook_strength", "score",
    }
    assert set(Stage2SegmentOutput.model_fields) == {
        "role", "start_segment_id", "end_segment_id", "start_anchor_text", "end_anchor_text",
    }
    assert set(Stage2Output.model_fields) == {"candidates"}


def _valid_material_segment_kwargs():
    return {"start_segment_id": 0, "end_segment_id": 0}


def _valid_material_kwargs():
    return {
        "material_type": "hook", "segments": [_valid_material_segment_kwargs()],
        "usefulness_score": 80,
    }


def test_stage1_output_accepts_zero_to_six_materials():
    """Stage1's per-chunk cap was widened 3 -> config.STAGE1_MAX_CANDIDATES_
    PER_CHUNK (6): Stage1's job is recall (cast a wide net of materials
    across every material_type), not picking the final best-3 -- that
    narrowing still happens via the material prefilter + Stage2's final
    edit design, not by capping Stage1's search breadth.
    """
    assert config.STAGE1_MAX_CANDIDATES_PER_CHUNK == 6
    assert Stage1Output(materials=[]).materials == []
    for n in range(1, 7):
        out = Stage1Output(materials=[Stage1MaterialOutput(**_valid_material_kwargs()) for _ in range(n)])
        assert len(out.materials) == n
    with pytest.raises(ValidationError):
        Stage1Output(materials=[Stage1MaterialOutput(**_valid_material_kwargs()) for _ in range(7)])


def test_stage1_material_output_segments_length_bounds():
    for n in (1, 2, 3):
        kwargs = _valid_material_kwargs()
        kwargs["segments"] = [_valid_material_segment_kwargs() for _ in range(n)]
        Stage1MaterialOutput(**kwargs)
    for n in (0, 4):
        kwargs = _valid_material_kwargs()
        kwargs["segments"] = [_valid_material_segment_kwargs() for _ in range(n)]
        with pytest.raises(ValidationError):
            Stage1MaterialOutput(**kwargs)


def test_stage1_material_output_usefulness_score_bounds():
    for value in (0, 100):
        kwargs = _valid_material_kwargs()
        kwargs["usefulness_score"] = value
        Stage1MaterialOutput(**kwargs)
    for value in (-1, 101):
        kwargs = _valid_material_kwargs()
        kwargs["usefulness_score"] = value
        with pytest.raises(ValidationError):
            Stage1MaterialOutput(**kwargs)


def test_stage1_material_output_rejects_invalid_material_type():
    kwargs = _valid_material_kwargs()
    kwargs["material_type"] = "not_a_real_material_type"
    with pytest.raises(ValidationError):
        Stage1MaterialOutput(**kwargs)


@pytest.mark.parametrize("material_type", ["hook", "reason", "example", "context", "payoff"])
def test_stage1_material_output_accepts_every_material_type(material_type):
    kwargs = _valid_material_kwargs()
    kwargs["material_type"] = material_type
    Stage1MaterialOutput(**kwargs)


def test_stage1_material_segment_output_rejects_wrongly_typed_segment_id():
    with pytest.raises(ValidationError):
        Stage1MaterialSegmentOutput(start_segment_id=["not", "an", "int"], end_segment_id=0)


def test_stage1_material_output_rejects_unknown_fields():
    # extra="forbid" -> additionalProperties: false in the schema sent to
    # Claude, and the same strictness applies locally.
    kwargs = _valid_material_kwargs()
    kwargs["hook_type"] = "should not be accepted"
    with pytest.raises(ValidationError):
        Stage1MaterialOutput(**kwargs)


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
    output = Stage1Output(materials=[Stage1MaterialOutput(**_valid_material_kwargs())])
    monkeypatch.setattr(
        clip_selector.structured_output, "call",
        lambda schema_model, **kwargs: output,
    )

    segments = [_segment(0, start=0.0, text="強い発言です")]
    result = clip_selector.extract_candidates_for_chunk(segments, "タイトル")

    assert len(result) == 1
    assert isinstance(result[0], RawMaterial)
    assert result[0].material_type == "hook"
    assert result[0].usefulness_score == 80
    # A material has none of the finished-candidate display/scoring
    # properties -- those only ever exist on RawClipCandidate, produced
    # solely by Stage2's conversion (_raw_candidate_from_stage2_output).
    assert not hasattr(result[0], "hook_text")
    assert not hasattr(result[0], "title")


def test_extract_candidates_for_chunk_carries_anchor_text_through(monkeypatch):
    # The material conversion must preserve start_anchor_text verbatim --
    # boundary.py verifies/applies it later, at Stage2-design-conversion
    # and resolve time, exactly as it always has for Stage1 output.
    kwargs = _valid_material_kwargs()
    kwargs["segments"] = [
        {"start_segment_id": 0, "end_segment_id": 0, "start_anchor_text": "86は"}
    ]
    output = Stage1Output(materials=[Stage1MaterialOutput(**kwargs)])
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

    material = result[0]
    assert material.segments[0].start_anchor_text == "86は"


def test_design_final_candidates_converts_stage2_output_to_raw_candidates(monkeypatch):
    # Stage2 now designs final candidates directly (real segment_id
    # references it may freely recombine), not an id ranking -- the
    # conversion mirrors _raw_candidate_from_stage1_output exactly, plus
    # end_anchor_text.
    transcript = _long_transcript(minutes=1)
    materials = {"s1_m000": _raw_material(0, 2)}
    output = Stage2Output(
        candidates=[
            Stage2CandidateOutput(
                hook_type="story",
                segments=[
                    Stage2SegmentOutput(role="hook", start_segment_id=0, end_segment_id=2)
                ],
                opening_hook_strength=85,
                score=85,
            )
        ]
    )
    monkeypatch.setattr(clip_selector.structured_output, "call", lambda schema_model, **kwargs: output)

    designed = clip_selector.design_final_candidates(materials, transcript, "タイトル")

    assert len(designed) == 1
    assert designed[0].hook_type == "story"
    assert designed[0].opening_hook_strength == 85
    assert designed[0].score == 85
    assert designed[0].segments == [
        RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=2)
    ]


def test_design_final_candidates_can_recombine_segments_across_materials(monkeypatch):
    # Item 4: Stage2 may build a final candidate out of segments drawn
    # from *different* materials -- e.g. material A's hook + material B's
    # reason -- as long as every segment_id it references is real. The
    # conversion doesn't care which material a segment "belongs to" (it
    # never did -- segment_ids are transcript-global), so this just
    # confirms a design combining ids 0 (from material A) and 2 (from
    # material B) round-trips correctly.
    transcript = _long_transcript(minutes=1)
    materials = {
        "s1_m000": _raw_material(0, 0, material_type="hook"),  # a strong standalone conclusion
        "s1_m001": _raw_material(2, 2, material_type="reason"),  # a separate reason material
    }
    output = Stage2Output(
        candidates=[
            Stage2CandidateOutput(
                hook_type="strong_take",
                segments=[
                    Stage2SegmentOutput(role="hook", start_segment_id=0, end_segment_id=0),
                    Stage2SegmentOutput(role="answer", start_segment_id=2, end_segment_id=2),
                ],
                opening_hook_strength=90,
                score=90,
            )
        ]
    )
    monkeypatch.setattr(clip_selector.structured_output, "call", lambda schema_model, **kwargs: output)

    designed = clip_selector.design_final_candidates(materials, transcript, "タイトル")

    assert len(designed) == 1
    assert [s.start_segment_id for s in designed[0].segments] == [0, 2]
    assert [s.role for s in designed[0].segments] == ["hook", "answer"]


def test_design_final_candidates_carries_end_anchor_text_through(monkeypatch):
    # Stage2SegmentOutput.end_anchor_text is the one genuinely new field
    # versus Stage1SegmentOutput -- confirm it survives the conversion
    # into RawUsedSegment (boundary.py already knows how to verify/apply
    # it, unchanged since the duration-repair round).
    transcript = _long_transcript(minutes=1)
    materials = {"s1_m000": _raw_material(0, 2)}
    output = Stage2Output(
        candidates=[
            Stage2CandidateOutput(
                hook_type="story",
                segments=[
                    Stage2SegmentOutput(
                        role="hook", start_segment_id=0, end_segment_id=2,
                        start_anchor_text="segment 0", end_anchor_text="segment 2",
                    )
                ],
                opening_hook_strength=85,
                score=85,
            )
        ]
    )
    monkeypatch.setattr(clip_selector.structured_output, "call", lambda schema_model, **kwargs: output)

    designed = clip_selector.design_final_candidates(materials, transcript, "タイトル")

    assert designed[0].segments[0].start_anchor_text == "segment 0"
    assert designed[0].segments[0].end_anchor_text == "segment 2"


def test_design_final_candidates_does_not_send_full_transcript(monkeypatch):
    # F: Stage2 only sees a compact per-material summary -- an
    # unreferenced transcript segment's distinctive text must never
    # appear in what gets sent to the API.
    transcript = _long_transcript(minutes=5)
    transcript.segments[-1].text = "この文言はどの候補にも含まれない特徴的な発言マーカーXYZ123"
    materials = {"s1_m000": _raw_material(0, 2)}

    captured = {}

    def _spy(schema_model, *, stage, system_prompt, user_content, max_tokens):
        captured["user_content"] = user_content
        return Stage2Output(candidates=[])

    monkeypatch.setattr(clip_selector.structured_output, "call", _spy)
    clip_selector.design_final_candidates(materials, transcript, "タイトル")

    assert "マーカーXYZ123" not in captured["user_content"]


def test_design_final_candidates_deduplicates_exact_repeat_designs(monkeypatch):
    # New Python-side safety net (item 6's "duplicate" check): two
    # byte-identical final designs collapse to one, even if Stage2's own
    # prompt-level dedup instruction fails to catch it.
    transcript = _long_transcript(minutes=1)
    materials = {"s1_m000": _raw_material(0, 2)}
    same_segment = Stage2SegmentOutput(role="hook", start_segment_id=0, end_segment_id=2)
    output = Stage2Output(
        candidates=[
            Stage2CandidateOutput(hook_type="story", segments=[same_segment], opening_hook_strength=85, score=85),
            Stage2CandidateOutput(hook_type="story", segments=[same_segment], opening_hook_strength=85, score=85),
        ]
    )
    monkeypatch.setattr(clip_selector.structured_output, "call", lambda schema_model, **kwargs: output)

    designed = clip_selector.design_final_candidates(materials, transcript, "タイトル")
    assert len(designed) == 1


# --- Stage2 semantic closure hard gate (real-machine incident: a --------
# --- candidate whose hook posed "why is X better than Y" ranked high ---
# --- despite its body never explaining why -- see rank_and_finalize.md) -


def test_design_final_candidates_supports_omitting_a_closure_failing_design(monkeypatch):
    # Item 12/K: Stage2Output needs no new field to express semantic-
    # closure failure -- the prompt instructs Claude to simply omit the
    # design (see rank_and_finalize.md's "意味的な完結性" section) rather
    # than including a broken one, and the schema's min_length=0 already
    # supports returning fewer designs than materials given, with no
    # further changes.
    transcript = _long_transcript(minutes=1)
    materials = {
        "s1_m000": _raw_material(0, 2),  # hook + real reason: passes closure
        "s1_m001": _raw_material(0, 2),  # hook only, no reason: Stage2 omits this design
    }
    output = Stage2Output(
        candidates=[
            Stage2CandidateOutput(
                hook_type="story",
                segments=[Stage2SegmentOutput(role="hook", start_segment_id=0, end_segment_id=2)],
                opening_hook_strength=85,
                score=85,
            )
        ]
    )
    monkeypatch.setattr(clip_selector.structured_output, "call", lambda schema_model, **kwargs: output)

    designed = clip_selector.design_final_candidates(materials, transcript, "タイトル")

    assert len(designed) == 1
    assert set(Stage2Output.model_fields) == {"candidates"}


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
    materials = [_raw_material(0, 2) for _ in range(3)]
    monkeypatch.setattr(clip_selector, "run_stage1", lambda *a, **k: materials)

    call_count = {"n": 0}

    def _fake_design(materials, t, title):
        call_count["n"] += 1
        return [_raw_candidate(0, 2) for _ in materials]

    monkeypatch.setattr(clip_selector, "design_final_candidates", _fake_design)
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
    materials = [_raw_material(0, 2, usefulness_score=90) for _ in range(3)]
    cache.save_stage1_chunk(transcript.video_id, 0, materials)

    call_count = {"n": 0}

    def _fake_design(materials, t, title):
        call_count["n"] += 1
        return [_raw_candidate(0, 2, opening_hook_strength=90) for _ in materials]

    monkeypatch.setattr(clip_selector, "design_final_candidates", _fake_design)
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
    # Stage2 either (Anthropic API calls = 0 for this failure).
    transcript = _long_transcript(minutes=1)
    monkeypatch.setattr(
        clip_selector, "design_final_candidates",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Stage2 must not be called")),
    )

    with pytest.raises(RuntimeError, match="Stage1素材キャッシュ"):
        clip_selector.refresh_candidates_only(transcript, "タイトル")


def test_refresh_candidates_only_raises_without_stage2_call_when_no_usable_material(monkeypatch):
    # The material prefilter no longer checks duration (that's now
    # Stage2's + the final local gate's job), so a short cached material
    # is no longer grounds to skip Stage2 on its own -- only actively
    # broken speech is. Zero usable materials must still raise before
    # ever calling Stage2 (Anthropic API calls = 0 for this failure).
    transcript = Transcript(
        video_id="vidRefreshNoUsableMaterial", language="ja",
        segments=[_segment(0, start=0.0, text="ちょっと表現が難しいんですけども、要するにこうです。")],
    )
    materials = [_raw_material(0, 0, usefulness_score=90) for _ in range(3)]
    cache.save_stage1_chunk(transcript.video_id, 0, materials)

    monkeypatch.setattr(
        clip_selector, "design_final_candidates",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Stage2 must not be called")),
    )

    with pytest.raises(RuntimeError, match="参照整合性・発話品質の基本チェック"):
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
    # A stale cached chunk result -- refresh_stage1_and_candidates must
    # never reuse this, only what a fresh Stage1 call returns.
    stale_bad = [_raw_material(0, 0, usefulness_score=10)]
    cache.save_stage1_chunk(transcript.video_id, 0, stale_bad)

    fresh_good = [_raw_material(0, 2, usefulness_score=90) for _ in range(3)]
    call_count = {"n": 0}

    def _fake_extract(chunk_segments, video_title):
        call_count["n"] += 1
        return fresh_good

    monkeypatch.setattr(clip_selector, "extract_candidates_for_chunk", _fake_extract)
    monkeypatch.setattr(
        clip_selector, "design_final_candidates",
        lambda materials, t, title: [_raw_candidate(0, 2, opening_hook_strength=90) for _ in materials],
    )

    result = clip_selector.refresh_stage1_and_candidates(transcript, "タイトル")

    assert len(result) == 3
    assert call_count["n"] == 1  # exactly one chunk for a 1-minute transcript
    # The stale cached material must be gone: Stage1 was fully
    # regenerated, not reused from the existing (now-outdated) cache --
    # and the chunk cache on disk is overwritten with the new result.
    reloaded_chunk = cache.load_stage1_chunk(transcript.video_id, 0)
    assert all(m.usefulness_score == 90 for m in reloaded_chunk)
    # The finalized Stage2 result is saved too.
    reloaded_stage2 = cache.load_stage2(transcript.video_id)
    assert len(reloaded_stage2) == 3


def test_refresh_stage1_and_candidates_keeps_earlier_chunk_success_on_later_failure(monkeypatch):
    monkeypatch.setattr(config, "CHUNK_MINUTES", 10.0)
    monkeypatch.setattr(config, "CHUNK_OVERLAP_MINUTES", 1.0)
    transcript = _long_transcript(minutes=25)
    chunks = clip_selector._build_chunks(clip_selector._usable_segments(transcript))
    assert len(chunks) >= 2  # sanity: this test needs at least 2 chunks

    good = [_raw_material(0, 2, usefulness_score=90)]
    call_count = {"n": 0}

    def _fake_extract(chunk_segments, video_title):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return good
        raise RuntimeError("simulated Stage1 API failure on chunk 2")

    monkeypatch.setattr(clip_selector, "extract_candidates_for_chunk", _fake_extract)
    monkeypatch.setattr(
        clip_selector, "design_final_candidates",
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


def test_refresh_stage1_and_candidates_raises_without_stage2_call_when_no_usable_material(monkeypatch):
    transcript = _long_transcript(minutes=1)
    # Freshly "regenerated" Stage1 materials that are all actively broken
    # speech -- the only thing the material prefilter still screens for --
    # so Stage2 must never be reached (Anthropic API calls = 0 for this
    # failure).
    bad_material = _raw_material(0, 0, usefulness_score=90)
    transcript.segments[0].text = "ちょっと表現が難しいんですけども、要するにこうです。"
    transcript.segments[0].words = [
        TranscriptWord(start=0.0, end=2.0, text="ちょっと表現が難しいんですけども、要するにこうです。")
    ]

    monkeypatch.setattr(clip_selector, "extract_candidates_for_chunk", lambda *a, **k: [bad_material])
    monkeypatch.setattr(
        clip_selector, "design_final_candidates",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Stage2 must not be called")),
    )

    with pytest.raises(RuntimeError, match="参照整合性・発話品質の基本チェック"):
        clip_selector.refresh_stage1_and_candidates(transcript, "タイトル")


def test_refresh_stage1_and_candidates_does_not_overwrite_stage2_cache_when_finalize_fails(monkeypatch):
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _long_transcript(minutes=1)

    # Existing Stage2 cache from a prior successful run -- must survive
    # completely untouched if this refresh's final gate fails.
    old_stage2 = [_raw_candidate(0, 2, opening_hook_strength=90)] * 3
    cache.save_stage2(transcript.video_id, old_stage2)

    fresh_materials = [_raw_material(0, 2, usefulness_score=90) for _ in range(3)]
    monkeypatch.setattr(clip_selector, "extract_candidates_for_chunk", lambda *a, **k: fresh_materials)
    # Stage2 design runs (costs 1 API call) but returns only 2 valid
    # designs -- _design_finalize_and_cache must raise before ever calling
    # cache.save_stage2, leaving the old cache exactly as it was.
    monkeypatch.setattr(
        clip_selector, "design_final_candidates",
        lambda materials, t, title: [_raw_candidate(0, 2, opening_hook_strength=90) for _ in range(2)],
    )

    with pytest.raises(RuntimeError, match="ローカル検証を通過したのは"):
        clip_selector.refresh_stage1_and_candidates(transcript, "タイトル")

    reloaded = cache.load_stage2(transcript.video_id)
    assert len(reloaded) == 3
    assert all(c.segments[0].end_segment_id == 2 for c in reloaded)  # untouched old cache


# --- speech fluency gate: real-machine incident (word-search/self- -------
# --- correction/incomplete-thought hooks must be rejected, never a -------
# --- matter of AI opening_hook_strength self-rating) ---------------------


def test_evaluate_local_candidate_drops_body_disfluency(monkeypatch):
    # A clean hook followed by a context segment where the speaker
    # searches for words and re-explains the same content -- no check
    # before this gate ever inspected body/context/answer/payoff text at
    # all, so this must be the thing that catches it.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = Transcript(
        video_id="vidBodyDisfluency", language="ja",
        segments=[
            _segment(0, start=0.0, text="Nレンジでの走行、高めの空気圧、これ全部損している可能性があります。"),
            _segment(
                1, start=5.0,
                text="空気圧が高くなりすぎるとタイヤが丸い状態になるので、ちょっと表現が難しいんですけども、"
                     "接地面が丸くなるので、実は転がり抵抗が増えるんです。",
            ),
        ],
    )
    raw = RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0),
            RawUsedSegment(role="context", start_segment_id=1, end_segment_id=1),
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=80, reasoning="", caveats="",
    )

    result = clip_selector.evaluate_local_candidate(raw, transcript)
    assert result.accepted is False
    assert result.reason == "speech_disfluency"
    assert result.disfluency_detail == "context: 表現が難しい"


def test_evaluate_local_candidate_drops_the_real_candidate1_body(monkeypatch):
    # Pinned against the exact real-machine incident text -- word-search
    # marker sandwiched between two re-explanations of the same concept
    # ("丸い状態になるので" / "丸くなるので").
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = Transcript(
        video_id="vidCandidate1", language="ja",
        segments=[
            _segment(
                0, start=0.0,
                text="Nレンジでの走行、高めの空気圧、エアコンを切って窓を開ける。"
                     "これ、全部損している可能性があります。",
            ),
            _segment(
                1, start=5.0,
                text="空気圧が高くなりすぎると、タイヤが丸い状態になるので、"
                     "ちょっと表現が難しいんですけども、接地面が丸くなるので、",
            ),
        ],
    )
    raw = RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0),
            RawUsedSegment(role="context", start_segment_id=1, end_segment_id=1),
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=80, reasoning="", caveats="",
    )

    result = clip_selector.evaluate_local_candidate(raw, transcript)
    assert result.accepted is False
    assert result.reason == "speech_disfluency"


def test_evaluate_local_candidate_drops_incomplete_disfluent_hook(monkeypatch):
    # The real "candidate 3" hook -- front-dependent, self-correcting, and
    # trails off unfinished. opening_hook_strength=85 (a high AI
    # self-rating, above config.MIN_OPENING_HOOK_STRENGTH) must never
    # override the local veto.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = Transcript(
        video_id="vidCandidate3", language="ja",
        segments=[
            _segment(0, start=0.0, text="でもこれって、実は逆っていうのか、間違っていて、一般的な車、エンジン車は、"),
            _segment(1, start=5.0, text="アクセルオフのほうが実は効率がいいんです。"),
        ],
    )
    raw = RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0),
            RawUsedSegment(role="answer", start_segment_id=1, end_segment_id=1),
        ],
        hook_text="h", opening_hook_strength=85, title="", description="",
        score=80, reasoning="", caveats="",
    )

    result = clip_selector.evaluate_local_candidate(raw, transcript)
    assert result.accepted is False
    assert result.reason == "speech_disfluency"
    assert result.disfluency_detail is not None
    assert result.disfluency_detail.startswith("hook:")


def test_evaluate_local_candidate_keeps_natural_colloquial_hook(monkeypatch):
    # A previously-good, natural colloquial hook must not be broken by
    # the new gate -- "っていうのが" (nominalizer) is not "っていうのか"
    # (reformulation connective).
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = Transcript(
        video_id="vidNaturalHook", language="ja",
        segments=[_segment(0, start=0.0, text="ギアを入れてアクセルオフっていうのが、実はニュートラルよりも燃費は良いです。")],
    )
    raw = _raw_candidate(0, 0, opening_hook_strength=90)

    result = clip_selector.evaluate_local_candidate(raw, transcript)
    assert result.accepted is True


def test_evaluate_local_candidate_keeps_plain_reformulation_connective(monkeypatch):
    # A bare "というか" used as an ordinary paraphrase connective must
    # never be rejected on its own.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = Transcript(
        video_id="vidPlainConnective", language="ja",
        segments=[_segment(0, start=0.0, text="燃費改善というか、時短のためにやっています。")],
    )
    raw = _raw_candidate(0, 0, opening_hook_strength=90)

    result = clip_selector.evaluate_local_candidate(raw, transcript)
    assert result.accepted is True


def test_evaluate_local_candidate_keeps_mild_filler_in_body(monkeypatch):
    # A single "あの" mid-sentence in a body segment is normal,
    # unremarkable speech -- must not trigger the fluency veto.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = Transcript(
        video_id="vidMildFiller", language="ja",
        segments=[
            _segment(0, start=0.0, text="86はエンジン特性が独特です。"),
            _segment(1, start=5.0, text="そこはですね、あの、少し複雑な話になるんですが、結論だけ言うと問題ありません。"),
        ],
    )
    raw = RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0),
            RawUsedSegment(role="context", start_segment_id=1, end_segment_id=1),
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=80, reasoning="", caveats="",
    )

    result = clip_selector.evaluate_local_candidate(raw, transcript)
    assert result.accepted is True


def test_repair_disfluency_trim_repairs_clean_hook_preamble(monkeypatch):
    # A word-search filler clause immediately followed by an independent,
    # clean complete sentence, with real word timestamps -- repair should
    # move the start point past the filler via start_anchor_text.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = Transcript(
        video_id="repairDisfluencyClean", language="ja",
        segments=[
            _segment_with_words(0, 0.0, "ちょっと表現が難しいんですけども、", "要するに新品タイヤは長持ちします。"),
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
    assert result.repair_method == "disfluency_trim"
    assert result.original_reason == "speech_disfluency"
    resolved = boundary.resolve_candidate(result.candidate, transcript, candidate_id="c1")
    assert resolved.segments[0].text.startswith("要するに")
    assert "表現が難しい" not in resolved.segments[0].text


def test_repair_disfluency_trim_declines_without_word_timestamps(monkeypatch):
    # Same shape as the repair case above, but with no word-timestamp
    # data at all -- must never guess a cut point, so the candidate stays
    # rejected with no repair applied.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = Transcript(
        video_id="repairDisfluencyNoWords", language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=2.0,
                text="ちょっと表現が難しいんですけども、要するに新品タイヤは長持ちします。",
                words=[],
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
    assert result.reason == "speech_disfluency"
    assert result.repair_method is None


def test_repair_disfluency_trim_declines_candidate3_shaped_hook(monkeypatch):
    # The real "candidate 3" incident shape: a disfluent clause followed
    # by a bare continuative clause ("間違っていて、") still grammatically
    # glued to it, not an independent clean clause -- must never be
    # "repaired" into starting playback mid-correction. Rejected outright,
    # not force-rescued (item 6's explicit instruction).
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = Transcript(
        video_id="repairDisfluencyCandidate3", language="ja",
        segments=[
            _segment_with_words(
                0, 0.0,
                "でもこれって、", "実は逆っていうのか、", "間違っていて、", "一般的な車、", "エンジン車は、",
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
    assert result.reason == "speech_disfluency"
    assert result.repair_method is None


def test_repair_disfluency_drops_disfluent_body_segment(monkeypatch):
    # When the hook itself is clean but a body (context/answer/payoff)
    # segment is disfluent, disfluency_trim can't help (it only targets
    # the hook's own preamble) -- dropping the offending optional segment
    # (the existing drop_context_segment/drop_non_context_segment repair)
    # rescues the candidate instead.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = Transcript(
        video_id="repairDisfluencyDropBody", language="ja",
        segments=[
            _segment(0, start=0.0, text="空気圧管理は重要なポイントです。多くの人が見落としています。"),
            _segment(
                1, start=5.0,
                text="空気圧が高くなりすぎるとタイヤが丸い状態になるので、ちょっと表現が難しいんですけども、"
                     "接地面が丸くなるので、実は転がり抵抗が増えるんです。",
            ),
        ],
    )
    candidate = RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0),
            RawUsedSegment(role="context", start_segment_id=1, end_segment_id=1),
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=80, reasoning="", caveats="",
    )

    result = clip_selector.evaluate_local_candidate_with_repair(candidate, transcript)
    assert result.accepted is True
    assert result.repair_method == "drop_context_segment"
    assert result.original_reason == "speech_disfluency"
    assert len(result.candidate.segments) == 1


def test_diagnostic_summary_shows_disfluency_reason_segment_and_marker(monkeypatch):
    # Diagnostic output should name the reason/segment/marker, not dump
    # the full candidate text.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = Transcript(
        video_id="vidDiagDisfluency", language="ja",
        segments=[_segment(0, start=0.0, text="ちょっと表現が難しいんですけども、要するにこうです。")],
    )
    raw = _raw_candidate(0, 0, opening_hook_strength=90)

    evaluations = clip_selector._evaluate_all_local_candidates([raw], transcript)
    summary = clip_selector._format_diagnostic_summary(evaluations)

    assert "reject=speech_disfluency" in summary
    assert "disfluency=hook: 表現が難しい" in summary


def test_finalize_candidates_also_rejects_disfluent_candidates(monkeypatch):
    # The cache-hit / render-defensive enforcement point
    # (finalize_candidates) must independently apply the same fluency
    # veto -- not just the fresh-selection path (_filter_local_quality).
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = Transcript(
        video_id="vidFinalizeDisfluency", language="ja",
        segments=[_segment(0, start=0.0, text="ちょっと表現が難しいんですけども、要するにこうです。")],
    )
    raw = _raw_candidate(0, 0, opening_hook_strength=90)

    with pytest.raises(RuntimeError, match="流暢性"):
        clip_selector.finalize_candidates([raw, raw, raw], transcript)


def test_select_candidates_cache_hit_rejects_disfluent_cached_candidates(monkeypatch):
    # The exact real-world scenario this whole gate exists for: a
    # disfluent candidate already sitting in a stale Stage2 cache (from
    # before this gate existed) must be caught on its very next read,
    # with zero Anthropic API calls -- no re-analysis needed.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = Transcript(
        video_id="vidCacheHitDisfluency", language="ja",
        segments=[_segment(0, start=0.0, text="ちょっと表現が難しいんですけども、要するにこうです。")],
    )
    stale_bad = _raw_candidate(0, 0, opening_hook_strength=90)
    cache.save_stage2(transcript.video_id, [stale_bad] * 3)

    with pytest.raises(RuntimeError, match="流暢性"):
        clip_selector.select_candidates(transcript, "タイトル")


# --- speech restart: real-machine incident (an unfinished clause -------
# --- abandoned and restarted with the same content phrase in a --------
# --- different construction; has_speech_disfluency alone cannot catch --
# --- this -- no word-search marker, no reformulation/reversal ---------
# --- connective) ---------------------------------------------------------


def test_evaluate_local_candidate_drops_speech_restart_candidate1_example(monkeypatch):
    # A/E: the real incident -- "Nレンジで下るというのは、" is abandoned
    # mid-construction and restarted as "Nレンジにすると、". The segment's
    # own full text still ends confidently ("。"), so this must be caught
    # by the new speech_restart gate, not incomplete_final_ending.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = Transcript(
        video_id="vidSpeechRestart", language="ja",
        segments=[
            _segment_with_words(
                0, 0.0,
                "ホンダも取扱説明書の中に、",
                "走行中にNレンジで下るというのは、",
                "Nレンジにすると、",
                "エンジンブレーキが効かなくなって、",
                "思わぬ事故の原因になるので、",
                "急な坂道では注意が必要です。",
            ),
        ],
    )
    raw = _raw_candidate(0, 0, opening_hook_strength=90)

    result = clip_selector.evaluate_local_candidate(raw, transcript)
    assert result.accepted is False
    assert result.reason == "speech_restart"
    assert result.disfluency_detail is not None
    assert "Nレンジ" in result.disfluency_detail
    assert result.disfluency_detail.startswith("hook:")


def test_evaluate_local_candidate_drops_speech_restart_in_body_segment(monkeypatch):
    # G: a restart occurring in a BODY segment (not the hook) must still
    # reject the whole candidate, mirroring how a disfluent body segment
    # already fails the candidate even with a clean hook.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = Transcript(
        video_id="vidSpeechRestartBody", language="ja",
        segments=[
            _segment(0, start=0.0, text="86はスープラをベースに作られています。"),
            _segment_with_words(
                1, 5.0,
                "ホンダも取扱説明書の中に、",
                "走行中にNレンジで下るというのは、",
                "Nレンジにすると、",
                "エンジンブレーキが効かなくなって、",
                "思わぬ事故の原因になるので、",
                "急な坂道では注意が必要です。",
            ),
        ],
    )
    raw = RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0),
            RawUsedSegment(role="context", start_segment_id=1, end_segment_id=1),
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=80, reasoning="", caveats="",
    )

    result = clip_selector.evaluate_local_candidate(raw, transcript)
    assert result.accepted is False
    assert result.reason == "speech_restart"
    assert result.disfluency_detail is not None
    assert result.disfluency_detail.startswith("context:")


def test_evaluate_local_candidate_accepts_hook_payoff_exact_repeat_despite_restart_check(monkeypatch):
    # Regression (item 4/19): the existing, intentionally-allowed hook->
    # context->payoff exact-repeat structure must keep working -- the
    # restart check runs independently per RawUsedSegment, so reusing the
    # same clean segment twice must never be mistaken for a restart.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _junction_transcript()
    raw = RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=2, end_segment_id=2),
            RawUsedSegment(role="context", start_segment_id=0, end_segment_id=0),
            RawUsedSegment(role="payoff", start_segment_id=2, end_segment_id=2),
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )

    result = clip_selector.evaluate_local_candidate(raw, transcript)
    assert result.accepted is True


def test_diagnostic_summary_shows_speech_restart_reason_segment_and_marker(monkeypatch):
    # Item 17: diagnostic output should name the reason/segment/marker,
    # never dump the full candidate text.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = Transcript(
        video_id="vidDiagRestart", language="ja",
        segments=[
            _segment_with_words(
                0, 0.0,
                "ホンダも取扱説明書の中に、",
                "走行中にNレンジで下るというのは、",
                "Nレンジにすると、",
                "エンジンブレーキが効かなくなって、",
                "思わぬ事故の原因になるので、",
                "急な坂道では注意が必要です。",
            ),
        ],
    )
    raw = _raw_candidate(0, 0, opening_hook_strength=90)

    evaluations = clip_selector._evaluate_all_local_candidates([raw], transcript)
    summary = clip_selector._format_diagnostic_summary(evaluations)

    assert "reject=speech_restart" in summary
    assert "disfluency=hook: Nレンジ" in summary
    assert "急な坂道では注意が必要です" not in summary


def test_finalize_candidates_also_rejects_restart_candidates(monkeypatch):
    # The cache-hit / render-defensive enforcement point
    # (finalize_candidates) must independently apply the same restart
    # veto -- not just the fresh-selection path (_filter_local_quality).
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = Transcript(
        video_id="vidFinalizeRestart", language="ja",
        segments=[
            _segment_with_words(
                0, 0.0,
                "ホンダも取扱説明書の中に、",
                "走行中にNレンジで下るというのは、",
                "Nレンジにすると、",
                "エンジンブレーキが効かなくなって、",
                "思わぬ事故の原因になるので、",
                "急な坂道では注意が必要です。",
            ),
        ],
    )
    raw = _raw_candidate(0, 0, opening_hook_strength=90)

    with pytest.raises(RuntimeError, match="流暢性"):
        clip_selector.finalize_candidates([raw, raw, raw], transcript)


def test_select_candidates_cache_hit_rejects_restart_cached_candidates(monkeypatch):
    # The exact real-world scenario this gate exists for: a restart
    # candidate already sitting in a stale Stage2 cache (from before this
    # gate existed) must be caught on its very next read, with zero
    # Anthropic API calls -- no re-analysis needed.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = Transcript(
        video_id="vidCacheHitRestart", language="ja",
        segments=[
            _segment_with_words(
                0, 0.0,
                "ホンダも取扱説明書の中に、",
                "走行中にNレンジで下るというのは、",
                "Nレンジにすると、",
                "エンジンブレーキが効かなくなって、",
                "思わぬ事故の原因になるので、",
                "急な坂道では注意が必要です。",
            ),
        ],
    )
    stale_bad = _raw_candidate(0, 0, opening_hook_strength=90)
    cache.save_stage2(transcript.video_id, [stale_bad] * 3)

    with pytest.raises(RuntimeError, match="流暢性"):
        clip_selector.select_candidates(transcript, "タイトル")
