import pytest
from pydantic import ValidationError

from podcast_clipper import boundary, cache, clip_selector, config
from podcast_clipper.clip_selector import (
    Stage1FallbackSpanOutput,
    Stage1HookSeedOutput,
    Stage1MaterialSegmentOutput,
    Stage1Output,
    Stage1SupportMaterialOutput,
    Stage2AttemptOutput,
    Stage2CandidateOutput,
    Stage2FallbackCandidateOutput,
    Stage2Output,
    Stage2SegmentOutput,
)
from podcast_clipper.models import (
    RawClipCandidate,
    RawFallbackSpan,
    RawHookSeed,
    RawMaterial,
    RawMaterialSegment,
    RawUsedSegment,
    Stage1ChunkResult,
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


def _raw_candidate(start_id, end_id, role="hook", opening_hook_strength=80, score=80, **kwargs):
    return RawClipCandidate(
        hook_type="story",
        segments=[RawUsedSegment(role=role, start_segment_id=start_id, end_segment_id=end_id)],
        hook_text="h", opening_hook_strength=opening_hook_strength, title="", description="",
        score=score, reasoning="", caveats="",
        **kwargs,
    )


def _raw_material(start_id, end_id, material_type="hook", usefulness_score=80):
    return RawMaterial(
        material_type=material_type,
        segments=[RawMaterialSegment(start_segment_id=start_id, end_segment_id=end_id)],
        usefulness_score=usefulness_score,
    )


def _raw_hook_seed(start_id, end_id, signal_type="money_or_number", soft_score=80):
    return RawHookSeed(
        signal_type=signal_type,
        segments=[RawMaterialSegment(start_segment_id=start_id, end_segment_id=end_id)],
        soft_score=soft_score,
    )


def _raw_fallback_span(start_id, end_id, safety_score=60):
    return RawFallbackSpan(
        segments=[RawMaterialSegment(start_segment_id=start_id, end_segment_id=end_id)],
        safety_score=safety_score,
    )


def _valid_stage2_candidate_kwargs():
    return {
        "hook_type": "story",
        "segments": [{"role": "hook", "start_segment_id": 0, "end_segment_id": 0}],
        "opening_hook_strength": 80,
        "score": 80,
        "opening_self_contained": True,
        "hook_claim_resolved": True,
        "semantic_ending_complete": True,
        "ending_rationale_code": "natural_conclusion",
        "recomposed_for_duration": False,
    }


def _valid_stage2_attempt_kwargs(hook_seed_id, status="candidate"):
    if status == "candidate":
        return {
            "hook_seed_id": hook_seed_id,
            "status": "candidate",
            "candidate": Stage2CandidateOutput(**_valid_stage2_candidate_kwargs()),
        }
    return {
        "hook_seed_id": hook_seed_id,
        "status": "rejected",
        "reject_reason_code": "insufficient_context_available",
    }


def _valid_stage2_fallback_candidate_kwargs(fallback_id="fb1"):
    return {
        "fallback_id": fallback_id,
        "candidate": Stage2CandidateOutput(**_valid_stage2_candidate_kwargs()),
    }


def _stage2_output_single_attempt(hook_seed_id="s1_hookseed_000", candidate_overrides=None):
    """Builds a Stage2Output with exactly one attempt (status=candidate)
    for `hook_seed_id`, its candidate built from _valid_stage2_candidate_
    kwargs() merged with `candidate_overrides` -- the common shape most
    single-candidate design_final_candidates mocks need.
    """
    kwargs = {**_valid_stage2_candidate_kwargs(), **(candidate_overrides or {})}
    return Stage2Output(
        attempts=[
            Stage2AttemptOutput(
                hook_seed_id=hook_seed_id, status="candidate",
                candidate=Stage2CandidateOutput(**kwargs),
            )
        ],
        fallback_candidates=[], ranking=[hook_seed_id],
    )


def _stage2_design_result_single_candidate(candidate, hook_seed_id="s1_hookseed_000"):
    """Builds a Stage2DesignResult (the return type of the real
    design_final_candidates) with exactly one accepted attempt wrapping
    `candidate` -- the common shape most _design_finalize_and_cache mocks
    need."""
    return clip_selector.Stage2DesignResult(
        attempts=[
            clip_selector.Stage2AttemptResult(hook_seed_id=hook_seed_id, status="candidate", candidate=candidate)
        ],
        fallback_candidates={}, ranking=[hook_seed_id],
    )


def _stage2_design_result_candidates(candidates, prefix="s1_hookseed_"):
    """Same as _stage2_design_result_single_candidate but for several
    candidates at once, each getting its own hook_seed_id and all ranked
    in the given order."""
    ids = [f"{prefix}{i:03d}" for i in range(len(candidates))]
    return clip_selector.Stage2DesignResult(
        attempts=[
            clip_selector.Stage2AttemptResult(hook_seed_id=hsid, status="candidate", candidate=c)
            for hsid, c in zip(ids, candidates)
        ],
        fallback_candidates={}, ranking=ids,
    )


def _stage2_output_rejected_attempt(hook_seed_id="s1_hookseed_000", reject_reason_code="insufficient_context_available"):
    return Stage2Output(
        attempts=[
            Stage2AttemptOutput(hook_seed_id=hook_seed_id, status="rejected", reject_reason_code=reject_reason_code)
        ],
        fallback_candidates=[], ranking=[],
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


def test_min_opening_hook_strength_no_longer_exists_as_a_hard_gate():
    """Hook-seed-discovery redesign: MIN_OPENING_HOOK_STRENGTH was retired
    entirely (real-machine incident: Claude self-rated a context-
    dependent, unusable opening at 80 and it sailed through, while a
    genuinely usable candidate could be hard-rejected for scoring 78-79 --
    a pure self-reported number was not a reliable pass/fail signal). Its
    role is now served by the explicit semantic field opening_self_
    contained (see test_filter_local_quality_G_low_hook_strength_still_
    passes_when_self_contained / test_filter_local_quality_H_high_hook_
    strength_still_rejected_when_not_self_contained below).
    """
    assert not hasattr(config, "MIN_OPENING_HOOK_STRENGTH")


def test_filter_local_quality_G_low_hook_strength_still_passes_when_self_contained(monkeypatch):
    """A candidate scoring low (78) on the now-soft opening_hook_strength
    self-rating is still ACCEPTED as long as it's self-contained, fluent,
    and semantically complete -- proves the numeric threshold no longer
    gates accept/reject on its own."""
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _long_transcript(minutes=1)
    candidates = [_raw_candidate(0, 2, opening_hook_strength=78, opening_self_contained=True)]

    kept = clip_selector._filter_local_quality(candidates, transcript)
    assert len(kept) == 1


def test_filter_local_quality_H_high_hook_strength_still_rejected_when_not_self_contained(monkeypatch):
    """A candidate scoring high (95) on opening_hook_strength is still
    REJECTED when opening_self_contained is False -- proves the semantic
    field, not the number, is what actually gates."""
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _long_transcript(minutes=1)
    candidates = [_raw_candidate(0, 2, opening_hook_strength=95, opening_self_contained=False)]

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
    """Hook-seed-discovery redesign: explanatory/abstract openings
    ("だと思っています" etc.) are still called out as weak, but are no
    longer schema-enforced as a hard hook_seed exclusion -- they may
    still be extracted (recall priority) as long as it's understood they
    are weak."""
    text = _extract_candidates_prompt_text()
    assert "だと思っています" in text or "と思っています" in text
    assert "弱いと分かった上で出す" in text


def test_rank_and_finalize_prompt_no_longer_hard_gates_on_hook_strength():
    """Hook-seed-discovery redesign: opening_hook_strength is soft/
    diagnostic-only now (feeds Stage2's own ranking, never a Python
    accept/reject decision) -- see config.py's retired
    MIN_OPENING_HOOK_STRENGTH."""
    text = _rank_and_finalize_prompt_text()
    assert "この数値はhard rejectには使われません" in text


# --- prompt content: Stage1 widened to recall-oriented discovery ----------


def test_extract_candidates_prompt_allows_up_to_six_hook_seeds():
    text = _extract_candidates_prompt_text()
    assert "STAGE1_MAX_HOOK_SEEDS_PER_CHUNK" not in text  # never leak Python constant names
    assert config.STAGE1_MAX_HOOK_SEEDS_PER_CHUNK == 6


def test_extract_candidates_prompt_requires_scanning_whole_chunk():
    """Stage1 must not stop after finding candidates early in the chunk --
    it has to read to the end before finalizing its candidate list, so a
    stronger later utterance isn't missed."""
    text = _extract_candidates_prompt_text()
    assert "冒頭から末尾まで全体を読んで" in text


def test_extract_candidates_prompt_forbids_using_up_slots_on_the_first_half():
    text = _extract_candidates_prompt_text()
    assert "前半で見つかったものだけで枠を使い切り" in text
    assert "後半" in text


def test_extract_candidates_prompt_forbids_padding_weak_candidates_to_fill_six():
    text = _extract_candidates_prompt_text()
    assert "件数を埋めるために水増しする必要はありません" in text


def test_extract_candidates_prompt_forbids_near_duplicate_candidates():
    text = _extract_candidates_prompt_text()
    assert "複数枠に並べないでください" in text


def test_extract_candidates_prompt_states_stage1_is_recall_not_final_selection():
    """Documents the Stage1/Stage2 role split: Stage1 casts a wide net of
    single-purpose materials, Stage2 (seeing the pooled materials from
    every chunk) assembles and picks the final best-3."""
    text = _extract_candidates_prompt_text()
    assert "完成したShorts候補を組み立てる係ではありません" in text


def test_extract_candidates_prompt_allows_single_purpose_hook_seeds_and_materials():
    """Hook-seed-discovery redesign: hook_seeds/support_materials/
    fallback_spans are three genuinely independent output groups -- a
    hook_seed never needs an accompanying reason/example in the same
    chunk to be useful."""
    text = _extract_candidates_prompt_text()
    assert "3つの独立したグループに分かれます" in text
    assert "`hook_seeds`" in text and "`support_materials`" in text and "`fallback_spans`" in text


def test_extract_candidates_prompt_does_not_force_duration_target():
    """Stage1/Stage2 redesign: Stage1 must not skip/shape candidates to
    fit 20-50s -- that's now Stage2's responsibility."""
    text = _extract_candidates_prompt_text()
    assert "無理に収めようとしないでください" in text
    assert "尺の最終調整は後段のStage2の責務です" in text


def test_rank_and_finalize_prompt_requires_one_attempt_per_coverage_target():
    """Hook-seed-discovery redesign (the core fix): Stage2 must never
    silently skip a coverage-target hook_seed -- exactly one attempt per
    target, and a strong hook_seed must always be tried at least once
    even if it doesn't end up ranked first."""
    text = _rank_and_finalize_prompt_text()
    assert "ちょうどN件返してください" in text
    assert "無言でhook_seedを無視して終了することは禁止です" in text
    assert "必ず一度は完成candidateとして試作してください" in text


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
    """Stage2 may build a final candidate out of segments drawn from
    *different* support_materials/hook_seeds/fallback_spans, not just a
    single input as-is."""
    text = _rank_and_finalize_prompt_text()
    assert "support_materials/他のhook_seed/fallback_spanのsegmentを組み合わせてよい" in text


def test_rank_and_finalize_prompt_forbids_fabricated_segment_ids():
    text = _rank_and_finalize_prompt_text()
    assert "実在する`segment_id`の組み合わせのみを使うこと" in text
    assert "存在しない`segment_id`を作文しないこと" in text


def test_rank_and_finalize_prompt_documents_duration_target():
    text = _rank_and_finalize_prompt_text()
    assert "20〜50秒" in text
    assert "すぐにreject/50秒地点で機械的に切る、のどちらも禁止です" in text


def test_rank_and_finalize_prompt_documents_ending_design_selection_criteria():
    # Item 3: Stage2 must pick the semantically optimal ending, not the
    # earliest legal one -- the A-E criteria and the worked example that
    # pins down "select 34s, not 23s or 41s" must survive in the prompt.
    text = _rank_and_finalize_prompt_text()
    assert "最短で終われる地点ではなく、最適な自然終了を選ぶこと" in text
    assert "hookで作った期待" in text
    assert "34秒を選ぶこと" in text


def test_rank_and_finalize_prompt_documents_lookahead():
    text = _rank_and_finalize_prompt_text()
    assert "`lookahead`" in text
    assert "参考用の閲覧材料" in text


def test_rank_and_finalize_prompt_documents_ten_step_design_order():
    text = _rank_and_finalize_prompt_text()
    assert "設計手順（この順序で考えること）" in text
    assert "強いhookを選ぶ" in text
    assert "自然に完結していて20〜50秒に収まっていれば" in text


def test_rank_and_finalize_prompt_documents_recomposition_for_duration():
    # Item 5/6/9: 50s-over designs must be recomposed in the same single
    # call, never truncated at exactly 50s, with a clear deletion priority.
    text = _rank_and_finalize_prompt_text()
    assert "すぐにreject/50秒地点で機械的に切る、のどちらも禁止です" in text
    assert "冗長な`context`" in text
    assert "追加のAPI呼び出しを発生させません" in text


def test_rank_and_finalize_prompt_documents_new_output_fields():
    text = _rank_and_finalize_prompt_text()
    assert "semantic_ending_complete" in text
    assert "ending_rationale_code" in text
    assert "recomposed_for_duration" in text


def test_rank_and_finalize_prompt_documents_end_anchor_text():
    """end_anchor_text is the one genuinely new field vs Stage1's own
    schema -- the prompt must explain it symmetrically to start_anchor_
    text, including the same word-boundary/no-fabrication constraints."""
    text = _rank_and_finalize_prompt_text()
    assert "end_anchor_text" in text
    assert "start_anchor_text" in text
    assert "word境界に一致する必要がある" in text


def test_rank_and_finalize_prompt_never_lowers_the_bar_for_fallback():
    """Hook-seed-discovery redesign: fallback candidates must clear the
    exact same bar as primary attempts -- no leniency to avoid a
    0-candidate result, and it's fine for fallback_candidates to stay
    empty rather than pad with junk."""
    text = _rank_and_finalize_prompt_text()
    assert "fallbackだからといって基準を緩めないでください" in text
    assert "そのfallback candidateは0件のままにしてください" in text


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


def test_extract_candidates_prompt_scores_hook_seed_soft_score_strictly():
    """soft_score self-scoring must still be done strictly (never lenient
    self-grading), even though it's explicitly non-gating."""
    text = _extract_candidates_prompt_text()
    assert "自己採点は厳しく行うこと（甘い採点を禁止）" in text


def test_extract_candidates_prompt_scopes_hook_criteria_to_hook_seed_only():
    """The root-cause fix (now expressed via separate output groups
    rather than a material_type scoping sentence): hook-strength/opening
    criteria apply only to hook_seeds; support_materials have their own,
    much lighter quality bar (self-contained + not disfluent + useful),
    independent of hook-opening strength -- otherwise a reason material
    like the fuel-cut example would be rejected by hook-strength rules
    before ever reaching Stage2."""
    text = _extract_candidates_prompt_text()
    assert "## `hook_seed`の基準" in text
    assert "## `support_material`の基準" in text
    assert "穏やかな説明調であること自体、あるいはhookとしては弱いことは、`support_material`では不合格理由にしないでください" in text


def test_extract_candidates_prompt_documents_material_type_enum():
    # "hook" moved entirely to hook_seed/signal_type -- support_material's
    # material_type is now only reason/example/context/payoff.
    text = _extract_candidates_prompt_text()
    for material_type in ("reason", "example", "context", "payoff"):
        assert f"`{material_type}`" in text
    assert "material_type" in text


def test_extract_candidates_prompt_documents_signal_type_enum():
    text = _extract_candidates_prompt_text()
    for signal_type in (
        "money_or_number", "failure_or_loss", "surprising_fact", "strong_claim",
        "comparison", "direct_question", "strong_conclusion", "story_turn", "other",
    ):
        assert f"`{signal_type}`" in text
    assert "signal_type" in text


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


def test_evaluate_local_candidate_A_opening_not_self_contained(monkeypatch):
    # Replaces the retired hook_strength_below_80 hard gate: a low
    # opening_hook_strength self-rating no longer rejects on its own (see
    # test_filter_local_quality_G_low_hook_strength_still_passes_when_
    # self_contained); opening_self_contained=False is what actually
    # gates now.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _long_transcript(minutes=1)
    raw = _raw_candidate(0, 2, opening_hook_strength=79, opening_self_contained=False)

    result = clip_selector.evaluate_local_candidate(raw, transcript)
    assert result.accepted is False
    assert result.reason == "opening_not_self_contained"


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
            opening_self_contained=False,
        ),  # opening_not_self_contained
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
    # (which becomes job.error, already rendered to the user). Simulates
    # Stage2 designing a candidate for the one coverage-target hook_seed
    # that fails the local hard gate (opening_not_self_contained).
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _long_transcript(minutes=1)
    stage1_result = Stage1ChunkResult(
        hook_seeds=[_raw_hook_seed(0, 0)], support_materials=[], fallback_spans=[],
    )
    monkeypatch.setattr(clip_selector, "run_stage1", lambda *a, **k: stage1_result)
    design_result = clip_selector.Stage2DesignResult(
        attempts=[
            clip_selector.Stage2AttemptResult(
                hook_seed_id="s1_hookseed_000", status="candidate",
                candidate=_raw_candidate(0, 0, opening_self_contained=False),
            )
        ],
        fallback_candidates={}, ranking=["s1_hookseed_000"],
    )
    monkeypatch.setattr(
        clip_selector, "design_final_candidates",
        lambda coverage_targets, support_materials, fallback_spans, t, title: design_result,
    )

    with pytest.raises(RuntimeError) as exc_info:
        clip_selector.refresh_stage1_and_candidates(transcript, "タイトル")

    message = str(exc_info.value)
    assert "【診断】" in message
    assert "評価対象候補: 1件" in message
    assert "冒頭が自己完結していない" in message


def test_diagnose_local_filter_K_makes_zero_api_calls(monkeypatch):
    # K: relies on this module's autouse _forbid_real_anthropic_client
    # fixture (poisons anthropic.Anthropic()) plus an explicit guard on
    # run_stage1/extract_candidates_for_chunk -- diagnose_local_filter
    # must never reach either.
    transcript = _long_transcript(minutes=1)
    cache.save_stage1_chunk(
        transcript.video_id, 0,
        Stage1ChunkResult(hook_seeds=[], support_materials=[_raw_material(0, 0, usefulness_score=90)], fallback_spans=[]),
    )

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


def test_candidate_schema_version_still_15():
    # Segment-count relaxation round: Stage2CandidateOutput.segments' max_
    # length went 3->6 (see config.MAX_SEGMENTS_PER_CANDIDATE's docstring)
    # -- a genuine Stage2 schema change, so the version bumped once more
    # (14->15); it must not drift further within this round.
    assert config.CANDIDATE_SCHEMA_VERSION == 15


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
    hook_seed = _raw_hook_seed(0, 1)
    candidate = _raw_candidate(0, 1, opening_hook_strength=90)
    monkeypatch.setattr(
        clip_selector, "run_stage1",
        lambda *a, **k: Stage1ChunkResult(hook_seeds=[hook_seed], support_materials=[], fallback_spans=[]),
    )
    monkeypatch.setattr(
        clip_selector, "design_final_candidates",
        lambda coverage_targets, materials, fallback_spans, t, title: _stage2_design_result_single_candidate(
            candidate, hook_seed_id=coverage_targets[0][0]
        ),
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


def test_select_candidates_returns_partial_when_only_some_cached_candidates_remain_eligible(monkeypatch):
    """F (revised): NUM_CANDIDATES is a target/ceiling, not a required
    minimum -- 2 of 3 cached candidates staying within hard bounds after
    extension must be returned as a 2-candidate result, never discarded
    just because a 3rd didn't also survive (real-machine incident: doing
    so threw away genuinely good candidates as a total failure).
    """
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 5.0)
    transcript = _transcript_with_gap(
        0.3, ["冒頭の発言です。", "それが起きた理由としては、こういうことが考えられるので", "そのあたりも確認する必要があります。"]
    )
    good = _raw_candidate(0, 0, opening_hook_strength=90)  # ends naturally, short -> stays valid
    bad = _raw_candidate(0, 1, opening_hook_strength=90)  # extends past segment 2 -> exceeds hard max
    cache.save_stage2(transcript.video_id, [good, good, bad])

    result = clip_selector.select_candidates(transcript, "タイトル")

    assert len(result) == 2
    reloaded = cache.load_stage2(transcript.video_id)
    assert len(reloaded) == 2


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
    hook_seeds = [_raw_hook_seed(0, 0), _raw_hook_seed(1, 1), _raw_hook_seed(2, 2)]
    # Distinct start_anchor_text per candidate so _select_final_candidates'
    # exact-segment-sequence dedup does not collapse these 3 attempts (all
    # spanning the same segments 0-1) into 1 -- the point of this test is
    # 3 independently-designed candidates all extending identically.
    candidates = [_raw_candidate(0, 1, opening_hook_strength=90) for _ in range(3)]
    for c, anchor in zip(candidates, ["冒頭の発言です", "それが起きた理由", None]):
        c.segments[0].start_anchor_text = anchor

    monkeypatch.setattr(
        clip_selector, "run_stage1",
        lambda *a, **k: Stage1ChunkResult(hook_seeds=hook_seeds, support_materials=[], fallback_spans=[]),
    )
    monkeypatch.setattr(
        clip_selector, "design_final_candidates",
        lambda coverage_targets, materials, fallback_spans, t, title: _stage2_design_result_candidates(
            candidates
        ),
    )
    fresh_result = clip_selector.select_candidates(transcript, "タイトル")

    cache_transcript = Transcript(
        video_id=transcript.video_id + "-cache", language="ja", segments=transcript.segments
    )
    cache.save_stage2(cache_transcript.video_id, [candidates[0]] * 3)
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


def test_stage1_stage2_recombine_hook_seed_with_reason_material(monkeypatch):
    # Item 7 (carried into the hook-seed-discovery redesign) -- the single
    # most important cross-chunk recombination test: chunk A has only a
    # hook_seed ("Xの方がYより燃費が良い"); chunk B has a support_material
    # that would be weak as a Shorts hook but is an important reason
    # explanation ("減速時には一定条件で燃料噴射が止まるためです"). Both must
    # survive Stage1's lightweight prefilter regardless of the reason
    # material's weak-hook shape, and Stage2 must be able to combine them
    # (via one attempt on the hook_seed's coverage target) into one
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
    # chunk A: a hook_seed only (no reason material in this chunk).
    hook_seed = _raw_hook_seed(0, 0, signal_type="comparison", soft_score=90)
    # chunk B: weak as a Shorts hook (calm technical explanation, no
    # strong opening) but a real, important reason -- must still be
    # fetched by Stage1, since the lightweight prefilter never judges hook
    # strength.
    reason_material = _raw_material(1, 1, material_type="reason", usefulness_score=55)

    # Both survive the lightweight prefilter -- confirms the weak-as-hook
    # material is never rejected before ever reaching Stage2, which is
    # exactly the bug this round fixes.
    assert clip_selector._hook_seed_is_usable(hook_seed, transcript) is True
    assert clip_selector._material_is_usable(reason_material, transcript) is True

    stage1_result = Stage1ChunkResult(
        hook_seeds=[hook_seed], support_materials=[reason_material], fallback_spans=[],
    )
    monkeypatch.setattr(clip_selector, "run_stage1", lambda *a, **k: stage1_result)

    def _fake_design(coverage_targets, support_materials, fallback_spans, t, title):
        # Stage2 combines the hook_seed's segment with the reason
        # material's segment -- drawn from two different inputs -- into
        # one semantically-complete finished candidate.
        assert len(coverage_targets) == 1
        assert len(support_materials) == 1
        hook_seed_id = coverage_targets[0][0]
        candidate = RawClipCandidate(
            hook_type="strong_take",
            segments=[
                RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0),
                RawUsedSegment(role="answer", start_segment_id=1, end_segment_id=1),
            ],
            hook_text="h", opening_hook_strength=90, title="", description="",
            score=90, reasoning="", caveats="",
        )
        return clip_selector.Stage2DesignResult(
            attempts=[
                clip_selector.Stage2AttemptResult(
                    hook_seed_id=hook_seed_id, status="candidate", candidate=candidate,
                )
            ],
            fallback_candidates={}, ranking=[hook_seed_id],
        )

    monkeypatch.setattr(clip_selector, "design_final_candidates", _fake_design)

    result = clip_selector.select_candidates(transcript, "タイトル")

    assert len(result) == 1
    assert [s.start_segment_id for s in result[0].segments] == [0, 1]


def test_design_final_candidates_reassigns_reason_material_segment_to_hook_role(monkeypatch):
    # H: a support_material's segment doesn't need to become the finished
    # candidate's hook, but it CAN be -- Stage2, not the material's own
    # material_type, decides the final role. Confirms the conversion
    # doesn't carry any material-side "role" forward (RawMaterial has
    # none), it only uses whatever role Stage2's own output specifies.
    transcript = _long_transcript(minutes=1)
    coverage_targets = [("s1_hookseed_000", _raw_hook_seed(0, 0))]
    support_materials = {"s1_m000": _raw_material(0, 0, material_type="reason", usefulness_score=40)}
    output = _stage2_output_single_attempt(
        "s1_hookseed_000",
        {"segments": [{"role": "hook", "start_segment_id": 0, "end_segment_id": 0}]},
    )
    monkeypatch.setattr(clip_selector.structured_output, "call", lambda schema_model, **kwargs: output)

    design = clip_selector.design_final_candidates(coverage_targets, support_materials, {}, transcript, "タイトル")

    assert design.attempts[0].candidate.segments[0].role == "hook"


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

    candidate = RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0),
            RawUsedSegment(role="answer", start_segment_id=1, end_segment_id=1),
            RawUsedSegment(role="payoff", start_segment_id=2, end_segment_id=2),
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )
    design_result = _stage2_design_result_single_candidate(candidate)
    monkeypatch.setattr(
        clip_selector, "design_final_candidates",
        lambda coverage_targets, support_materials, fallback_spans, t, title: design_result,
    )
    monkeypatch.setattr(
        clip_selector, "run_stage1",
        lambda *a, **k: Stage1ChunkResult(hook_seeds=[_raw_hook_seed(0, 0)], support_materials=materials, fallback_spans=[]),
    )

    result = clip_selector._design_finalize_and_cache([_raw_hook_seed(0, 0)], materials, [], transcript, "タイトル")
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
    candidate = RawClipCandidate(
        hook_type="strong_take",
        segments=[RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0)],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )
    design_result = _stage2_design_result_single_candidate(candidate)
    monkeypatch.setattr(
        clip_selector, "design_final_candidates",
        lambda coverage_targets, support_materials, fallback_spans, t, title: design_result,
    )

    result = clip_selector._design_finalize_and_cache([_raw_hook_seed(0, 0)], [long_material], [], transcript, "タイトル")
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
    hook_seeds = [_raw_hook_seed(0, 0)]
    design_result = _stage2_design_result_single_candidate(_raw_candidate(9999, 9999))  # fabricated segment_id
    monkeypatch.setattr(
        clip_selector, "design_final_candidates",
        lambda coverage_targets, support_materials, fallback_spans, t, title: design_result,
    )

    with pytest.raises(RuntimeError, match="ローカル検証を通過したものが0件"):
        clip_selector._design_finalize_and_cache(hook_seeds, materials, [], transcript, "タイトル")


def test_design_finalize_I_rejects_out_of_bounds_duration(monkeypatch):
    # I: a Stage2 design outside [DURATION_HARD_MIN_SEC, DURATION_HARD_
    # MAX_SEC] is rejected by the unchanged duration_too_long/_too_short
    # checks -- no new logic needed, evaluate_local_candidate already
    # enforces this for whatever it's given.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 20.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 50.0)
    transcript = _long_transcript(minutes=1)
    materials = [_raw_material(0, 0)]
    hook_seeds = [_raw_hook_seed(0, 0)]
    design_result = _stage2_design_result_single_candidate(_raw_candidate(0, 0))  # a single 2-second segment -- far under 20s
    monkeypatch.setattr(
        clip_selector, "design_final_candidates",
        lambda coverage_targets, support_materials, fallback_spans, t, title: design_result,
    )

    with pytest.raises(RuntimeError, match="ローカル検証を通過したものが0件"):
        clip_selector._design_finalize_and_cache(hook_seeds, materials, [], transcript, "タイトル")


def test_design_finalize_J_rejects_unsafe_junction(monkeypatch):
    # J: a Stage2 design whose segments cut together unsafely (an
    # unfinished clause hard-cut into an unrelated topic) is rejected by
    # the unchanged evaluate_candidate_junctions check. Uses
    # _unfixable_bad_junction_transcript (not _junction_transcript): the
    # gap to the chronologically-next segment there is deliberately too
    # large for _extend_internal_junctions to bridge, so the hook's
    # unfinished "...のであれば" ending truly can't be fixed by extension
    # first -- with a close-by next segment, extension alone resolves the
    # incompleteness before the junction check ever runs, which would
    # make this scenario safe (and accepted) instead of unsafe.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _unfixable_bad_junction_transcript()
    materials = [_raw_material(0, 0)]
    hook_seeds = [_raw_hook_seed(0, 0)]
    candidate = RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0),
            RawUsedSegment(role="context", start_segment_id=2, end_segment_id=2),
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )  # segment 0 (unfinished) -> unrelated distant segment 2, skipping segment 1
    design_result = _stage2_design_result_single_candidate(candidate)
    monkeypatch.setattr(
        clip_selector, "design_final_candidates",
        lambda coverage_targets, support_materials, fallback_spans, t, title: design_result,
    )

    with pytest.raises(RuntimeError, match="ローカル検証を通過したものが0件"):
        clip_selector._design_finalize_and_cache(hook_seeds, materials, [], transcript, "タイトル")


# --- stage2 diagnostic: raw Stage2 output survives a failed run ----------
# (real-machine incident: Stage2 designed only 2 candidates, one of which
# was locally rejected for hook_strength_below_80 -- the resulting
# RuntimeError meant stage2_result.json was never written, and until this,
# NOTHING recorded what Stage2 had actually built, making the failure
# undiagnosable without a fresh, API-calling re-analysis)


def test_design_finalize_A_succeeds_with_one_candidate_and_still_saves_diagnostic(monkeypatch):
    # A (revised): Stage2 returns only 2 designs and one is locally
    # rejected -- since NUM_CANDIDATES is now a target/ceiling, not a
    # required minimum, the surviving 1 candidate is a success, not a
    # RuntimeError. The diagnostic snapshot must still record both of
    # Stage2's raw designs and each one's verdict either way.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _long_transcript(minutes=1)
    materials = [_raw_material(0, 2), _raw_material(0, 2)]
    hook_seeds = [_raw_hook_seed(0, 2), _raw_hook_seed(0, 2)]
    two_designs = [
        _raw_candidate(0, 2, opening_hook_strength=90),
        _raw_candidate(0, 2, opening_hook_strength=78, opening_self_contained=False),  # will fail
    ]
    design_result = _stage2_design_result_candidates(two_designs)

    monkeypatch.setattr(
        clip_selector, "design_final_candidates",
        lambda coverage_targets, support_materials, fallback_spans, t, title: design_result,
    )

    result = clip_selector._design_finalize_and_cache(hook_seeds, materials, [], transcript, "タイトル")
    assert len(result) == 1

    # stage2_result.json (the production cache) now holds the 1 successful
    # candidate -- partial success is real success, not withheld.
    saved = cache.load_stage2(transcript.video_id)
    assert saved is not None
    assert len(saved) == 1

    # The diagnostic snapshot records both of Stage2's raw designs and a
    # per-candidate local-validation verdict, regardless of the outcome.
    diagnostic = cache.load_stage2_diagnostic(transcript.video_id)
    assert diagnostic is not None
    assert len(diagnostic["candidates"]) == 2
    assert len(diagnostic["evaluations"]) == 2
    accepted_flags = [e["accepted"] for e in diagnostic["evaluations"]]
    assert accepted_flags == [True, False]
    assert diagnostic["evaluations"][1]["reason"] == "opening_not_self_contained"


def test_design_finalize_B_succeeds_with_two_candidates_and_records_all_three_in_diagnostic(monkeypatch):
    # B (revised): Stage2 returns 3 designs but one fails local validation
    # (fabricated segment_id) -- the surviving 2 are returned as a
    # 2-candidate success (never padded, never rejected merely for being
    # fewer than NUM_CANDIDATES). The diagnostic still shows all 3
    # original designs plus each one's verdict, not just the 2 that
    # succeeded.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _long_transcript(minutes=1)
    materials = [_raw_material(0, 2), _raw_material(0, 2), _raw_material(0, 2)]
    hook_seeds = [_raw_hook_seed(0, 0), _raw_hook_seed(1, 1), _raw_hook_seed(2, 2)]
    # Distinct segment ranges so the exact-segment-sequence dedup in
    # _select_final_candidates does not collapse the 2 valid designs into 1.
    three_designs = [
        _raw_candidate(0, 0, opening_hook_strength=90),
        _raw_candidate(0, 1, opening_hook_strength=90),
        _raw_candidate(9999, 9999, opening_hook_strength=90),  # fabricated segment_id
    ]
    design_result = _stage2_design_result_candidates(three_designs)

    monkeypatch.setattr(
        clip_selector, "design_final_candidates",
        lambda coverage_targets, support_materials, fallback_spans, t, title: design_result,
    )

    result = clip_selector._design_finalize_and_cache(hook_seeds, materials, [], transcript, "タイトル")
    assert len(result) == 2

    saved = cache.load_stage2(transcript.video_id)
    assert saved is not None
    assert len(saved) == 2

    diagnostic = cache.load_stage2_diagnostic(transcript.video_id)
    assert len(diagnostic["candidates"]) == 3
    assert len(diagnostic["evaluations"]) == 3
    reasons = [e["reason"] for e in diagnostic["evaluations"]]
    assert reasons.count("accepted") == 2
    assert "invalid_segment_reference" in reasons


def test_design_finalize_C_success_path_still_writes_diagnostic_separately(monkeypatch):
    # C: on a fully successful run, stage2_result.json holds only the
    # final accepted/finalized candidates (unchanged behavior); the
    # diagnostic snapshot exists alongside it, not instead of it.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _long_transcript(minutes=1)
    materials = [_raw_material(0, 2) for _ in range(3)]
    hook_seeds = [_raw_hook_seed(0, 0), _raw_hook_seed(1, 1), _raw_hook_seed(2, 2)]
    # Distinct segment ranges so the exact-segment-sequence dedup in
    # _select_final_candidates does not collapse the 3 designs into 1.
    three_designs = [
        _raw_candidate(0, 0, opening_hook_strength=90),
        _raw_candidate(0, 1, opening_hook_strength=90),
        _raw_candidate(0, 2, opening_hook_strength=90),
    ]
    design_result = _stage2_design_result_candidates(three_designs)

    monkeypatch.setattr(
        clip_selector, "design_final_candidates",
        lambda coverage_targets, support_materials, fallback_spans, t, title: design_result,
    )

    result = clip_selector._design_finalize_and_cache(hook_seeds, materials, [], transcript, "タイトル")
    assert len(result) == 3

    final_cache = cache.load_stage2(transcript.video_id)
    assert final_cache is not None
    assert len(final_cache) == 3

    diagnostic = cache.load_stage2_diagnostic(transcript.video_id)
    assert diagnostic is not None
    assert len(diagnostic["candidates"]) == 3
    assert all(e["accepted"] for e in diagnostic["evaluations"])


def test_design_finalize_takes_top_num_candidates_when_stage2_overproduces(monkeypatch):
    # The actual fix under test: when Stage2 uses its widened headroom
    # (STAGE2_MAX_DESIGNS=6) and designs more than NUM_CANDIDATES(3) valid
    # candidates, the run must succeed (never fail just because there were
    # "too many" good options) and return exactly NUM_CANDIDATES, taking
    # them in Stage2's own strongest-first order -- never all of them,
    # never a random subset.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _long_transcript(minutes=1)
    materials = [_raw_material(0, 2) for _ in range(5)]
    hook_seeds = [
        _raw_hook_seed(0, 0), _raw_hook_seed(0, 1), _raw_hook_seed(0, 2),
        _raw_hook_seed(1, 1), _raw_hook_seed(1, 2),
    ]
    # Distinct segment ranges (limited to the 3-segment transcript) so the
    # exact-segment-sequence dedup in _select_final_candidates does not
    # collapse these 5 designs down to 1.
    five_designs = [
        _raw_candidate(rng[0], rng[1], opening_hook_strength=90, score=s)
        for rng, s in zip([(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)], (95, 90, 85, 80, 75))
    ]
    design_result = _stage2_design_result_candidates(five_designs)

    monkeypatch.setattr(
        clip_selector, "design_final_candidates",
        lambda coverage_targets, support_materials, fallback_spans, t, title: design_result,
    )

    result = clip_selector._design_finalize_and_cache(hook_seeds, materials, [], transcript, "タイトル")

    assert len(result) == config.NUM_CANDIDATES == 3
    assert [c.score for c in result] == [95, 90, 85]

    diagnostic = cache.load_stage2_diagnostic(transcript.video_id)
    # The diagnostic still records all 5 of Stage2's designs, not just the
    # 3 that were ultimately selected.
    assert len(diagnostic["candidates"]) == 5
    assert all(e["accepted"] for e in diagnostic["evaluations"])


def test_design_finalize_D_takes_top_three_when_five_designed_four_accepted(monkeypatch):
    # D: Stage2 over-produces 5 designs, 1 of which fails local validation
    # -- the remaining 4 accepted still only yield the top NUM_CANDIDATES
    # (3), by score, never all 4.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _long_transcript(minutes=1)
    materials = [_raw_material(0, 2) for _ in range(5)]
    hook_seeds = [
        _raw_hook_seed(0, 0), _raw_hook_seed(0, 1), _raw_hook_seed(0, 2),
        _raw_hook_seed(1, 1), _raw_hook_seed(1, 2),
    ]
    # Distinct segment ranges (limited to the 3-segment transcript) so the
    # exact-segment-sequence dedup in _select_final_candidates does not
    # collapse the 4 valid designs down to fewer than 4.
    five_designs = [
        _raw_candidate(0, 0, opening_hook_strength=90, score=95),
        _raw_candidate(0, 1, opening_hook_strength=90, score=90),
        _raw_candidate(9999, 9999, opening_hook_strength=90, score=88),  # fabricated -> rejected
        _raw_candidate(1, 1, opening_hook_strength=90, score=85),
        _raw_candidate(1, 2, opening_hook_strength=90, score=80),
    ]
    design_result = _stage2_design_result_candidates(five_designs)

    monkeypatch.setattr(
        clip_selector, "design_final_candidates",
        lambda coverage_targets, support_materials, fallback_spans, t, title: design_result,
    )

    result = clip_selector._design_finalize_and_cache(hook_seeds, materials, [], transcript, "タイトル")

    assert len(result) == config.NUM_CANDIDATES == 3
    assert [c.score for c in result] == [95, 90, 85]

    diagnostic = cache.load_stage2_diagnostic(transcript.video_id)
    assert len(diagnostic["candidates"]) == 5
    reasons = [e["reason"] for e in diagnostic["evaluations"]]
    assert reasons.count("accepted") == 4
    assert "invalid_segment_reference" in reasons


def test_design_finalize_F_raises_when_two_designed_zero_accepted(monkeypatch):
    # F: the one remaining failure case -- Stage2 designs 2 candidates,
    # but ZERO survive local validation (both fabricated). Unlike a
    # shortfall below NUM_CANDIDATES (no longer a failure), zero
    # candidates is still a hard failure with no substitute available.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _long_transcript(minutes=1)
    materials = [_raw_material(0, 2), _raw_material(0, 2)]
    hook_seeds = [_raw_hook_seed(0, 2), _raw_hook_seed(0, 2)]
    two_bad_designs = [
        _raw_candidate(9999, 9999, opening_hook_strength=90),
        _raw_candidate(8888, 8888, opening_hook_strength=90),
    ]
    design_result = _stage2_design_result_candidates(two_bad_designs)

    monkeypatch.setattr(
        clip_selector, "design_final_candidates",
        lambda coverage_targets, support_materials, fallback_spans, t, title: design_result,
    )

    with pytest.raises(RuntimeError, match="ローカル検証を通過したものが0件"):
        clip_selector._design_finalize_and_cache(hook_seeds, materials, [], transcript, "タイトル")

    assert cache.load_stage2(transcript.video_id) is None
    diagnostic = cache.load_stage2_diagnostic(transcript.video_id)
    assert len(diagnostic["candidates"]) == 2
    assert all(not e["accepted"] for e in diagnostic["evaluations"])


def test_design_final_candidates_saves_raw_diagnostic_before_local_validation(monkeypatch):
    # design_final_candidates saves a diagnostic snapshot of every
    # attempt/fallback candidate it built, immediately after Stage2's
    # Structured Output is parsed -- before local validation (or the
    # dedup safety net in _select_final_candidates) ever runs -- so a
    # crash anywhere downstream still leaves this record behind.
    # design_final_candidates itself does NOT dedupe (that safety net now
    # lives in _select_final_candidates, applied after local validation).
    transcript = _long_transcript(minutes=1)
    coverage_targets = [("s1_hookseed_000", _raw_hook_seed(0, 2)), ("s1_hookseed_001", _raw_hook_seed(0, 2))]
    same_segment = Stage2SegmentOutput(role="hook", start_segment_id=0, end_segment_id=2)
    output = Stage2Output(
        attempts=[
            Stage2AttemptOutput(
                hook_seed_id="s1_hookseed_000", status="candidate",
                candidate=Stage2CandidateOutput(**{**_valid_stage2_candidate_kwargs(), "segments": [same_segment]}),
            ),
            Stage2AttemptOutput(
                hook_seed_id="s1_hookseed_001", status="candidate",
                candidate=Stage2CandidateOutput(**{**_valid_stage2_candidate_kwargs(), "segments": [same_segment]}),
            ),
        ],
        fallback_candidates=[], ranking=["s1_hookseed_000", "s1_hookseed_001"],
    )
    monkeypatch.setattr(clip_selector.structured_output, "call", lambda schema_model, **kwargs: output)

    design = clip_selector.design_final_candidates(coverage_targets, {}, {}, transcript, "タイトル")
    assert len(design.attempts) == 2  # not deduped at this point

    diagnostic = cache.load_stage2_diagnostic(transcript.video_id)
    assert len(diagnostic["candidates"]) == 2
    assert "evaluations" not in diagnostic  # not yet evaluated at this point


def test_select_final_candidates_dedupes_identical_segment_sequences(monkeypatch):
    # The safety-net dedup that replaced the old standalone
    # _dedupe_by_segment_sequence: two different hook_seed_ids whose
    # designed candidates have byte-identical segment sequences collapse
    # to one in the final selection -- rank order decides which survives.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _long_transcript(minutes=1)
    candidate_a = _raw_candidate(0, 2, score=90)
    candidate_b = _raw_candidate(0, 2, score=90)  # identical segment sequence to candidate_a
    attempt_evals = {
        "s1_hookseed_000": clip_selector.evaluate_local_candidate(candidate_a, transcript),
        "s1_hookseed_001": clip_selector.evaluate_local_candidate(candidate_b, transcript),
    }
    ranking = ["s1_hookseed_000", "s1_hookseed_001"]

    result = clip_selector._select_final_candidates(attempt_evals, {}, ranking)

    assert len(result) == 1
    assert result[0][0] == "s1_hookseed_000"


def test_stage2_diagnostic_is_never_read_by_the_production_selection_path(monkeypatch):
    # D: load_stage2 (the only function the real candidate-selection path
    # ever calls) must never be satisfied by a diagnostic-only file.
    transcript = _long_transcript(minutes=1)
    cache.save_stage2_diagnostic(transcript.video_id, [_raw_candidate(0, 2)])
    assert cache.load_stage2(transcript.video_id) is None


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
    monkeypatch.setattr(
        clip_selector, "run_stage1",
        lambda *a, **k: Stage1ChunkResult(
            hook_seeds=[_raw_hook_seed(0, 0)], support_materials=[], fallback_spans=[]
        ),
    )
    monkeypatch.setattr(
        clip_selector, "design_final_candidates",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Stage2 must not be called")),
    )

    with pytest.raises(RuntimeError, match="参照整合性・発話品質の基本チェック"):
        clip_selector.select_candidates(transcript, "タイトル")


def test_select_candidates_succeeds_with_one_candidate_when_stage2_designs_only_one(monkeypatch):
    # B: Stage2 designing (and locally passing) only 1 candidate is a
    # success returning that 1 candidate -- NUM_CANDIDATES is a target,
    # not a required minimum (real-machine incident: 1-2 good candidates
    # used to be discarded entirely as a failure).
    transcript = _long_transcript(minutes=1)
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    hook_seeds = [_raw_hook_seed(0, 2), _raw_hook_seed(0, 2), _raw_hook_seed(0, 2)]
    monkeypatch.setattr(
        clip_selector, "run_stage1",
        lambda *a, **k: Stage1ChunkResult(hook_seeds=hook_seeds, support_materials=[], fallback_spans=[]),
    )
    monkeypatch.setattr(
        clip_selector, "design_final_candidates",
        lambda coverage_targets, materials, fallback_spans, t, title: _stage2_design_result_single_candidate(
            _raw_candidate(0, 2)
        ),
    )

    result = clip_selector.select_candidates(transcript, "タイトル")
    assert len(result) == 1

    saved = cache.load_stage2(transcript.video_id)
    assert saved is not None
    assert len(saved) == 1


def test_select_candidates_succeeds_with_two_candidates_when_only_two_pass_semantic_closure(monkeypatch):
    # A: exactly the "only 2 pass semantic closure" shape -- this is now a
    # 2-candidate success, never padded up to config.NUM_CANDIDATES and
    # never rejected merely for being fewer.
    transcript = _long_transcript(minutes=1)
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    hook_seeds = [_raw_hook_seed(0, 2), _raw_hook_seed(0, 2), _raw_hook_seed(0, 2)]
    monkeypatch.setattr(
        clip_selector, "run_stage1",
        lambda *a, **k: Stage1ChunkResult(hook_seeds=hook_seeds, support_materials=[], fallback_spans=[]),
    )
    monkeypatch.setattr(
        clip_selector, "design_final_candidates",
        lambda coverage_targets, materials, fallback_spans, t, title: _stage2_design_result_candidates(
            [_raw_candidate(0, 0), _raw_candidate(0, 1)]
        ),
    )

    result = clip_selector.select_candidates(transcript, "タイトル")
    assert len(result) == 2

    saved = cache.load_stage2(transcript.video_id)
    assert saved is not None
    assert len(saved) == 2


def test_select_candidates_happy_path(monkeypatch):
    transcript = _long_transcript(minutes=1)
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    hook_seeds = [_raw_hook_seed(0, 2, soft_score=s) for s in (10, 20, 30, 40)]
    monkeypatch.setattr(
        clip_selector, "run_stage1",
        lambda *a, **k: Stage1ChunkResult(hook_seeds=hook_seeds, support_materials=[], fallback_spans=[]),
    )
    monkeypatch.setattr(
        clip_selector, "design_final_candidates",
        lambda coverage_targets, materials, fallback_spans, t, title: _stage2_design_result_candidates(
            [_raw_candidate(0, 0), _raw_candidate(0, 1), _raw_candidate(0, 2)]
        ),
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
    transcript = _long_transcript(minutes=2)
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    hook_seeds = [_raw_hook_seed(0, 0), _raw_hook_seed(1, 1), _raw_hook_seed(2, 2), _raw_hook_seed(3, 3)]
    monkeypatch.setattr(
        clip_selector, "run_stage1",
        lambda *a, **k: Stage1ChunkResult(hook_seeds=hook_seeds, support_materials=[], fallback_spans=[]),
    )

    call_count = {"n": 0}

    def _fake_design(coverage_targets, materials, fallback_spans, t, title):
        call_count["n"] += 1
        # Simulate Stage2 omitting one attempt entirely for failing semantic
        # closure -- the remaining 3 of 4 are returned, never padded back.
        assert len(coverage_targets) == 4
        return _stage2_design_result_candidates(
            [_raw_candidate(0, 0), _raw_candidate(1, 1), _raw_candidate(2, 2)]
        )

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


def test_stage1_support_material_output_field_set_excludes_finished_candidate_properties():
    # Stage1 structurally cannot produce hook_type/opening_hook_strength/
    # score/hook_text/title/description/reasoning/caveats any more -- those
    # all describe a *finished* candidate design, which only Stage2
    # produces (see Stage2CandidateOutput). A support material's own
    # fields are material_type/segments/usefulness_score only.
    fields = set(Stage1SupportMaterialOutput.model_fields)
    assert fields == {"material_type", "segments", "usefulness_score"}
    assert "hook_type" not in fields
    assert "opening_hook_strength" not in fields
    assert "score" not in fields
    assert "hook_text" not in fields


def test_stage1_support_material_output_excludes_hook_material_type():
    # "hook" was moved to hook_seeds/signal_type entirely -- support
    # materials are only ever reason/example/context/payoff now.
    with pytest.raises(ValidationError):
        Stage1SupportMaterialOutput(
            material_type="hook",
            segments=[Stage1MaterialSegmentOutput(start_segment_id=0, end_segment_id=0)],
            usefulness_score=80,
        )


def test_stage1_hook_seed_output_field_set():
    assert set(Stage1HookSeedOutput.model_fields) == {"signal_type", "segments", "soft_score"}


def test_stage1_fallback_span_output_field_set():
    assert set(Stage1FallbackSpanOutput.model_fields) == {"segments", "safety_score"}


def test_stage1_material_segment_output_has_no_role():
    # None of Stage1's three output groups have an internal structural
    # position (hook/context/answer/payoff) the way a finished candidate's
    # segments do. That's only ever decided by Stage2 (Stage2SegmentOutput
    # still has `role`).
    assert set(Stage1MaterialSegmentOutput.model_fields) == {
        "start_segment_id", "end_segment_id", "start_anchor_text",
    }


def test_stage2_candidate_output_field_set_is_the_only_place_finished_properties_exist():
    # Stage2's candidate/segment schemas are the only place hook_type/
    # opening_hook_strength/score/role/end_anchor_text exist -- Stage1's
    # schemas deliberately have none of these.
    assert set(Stage2CandidateOutput.model_fields) == {
        "hook_type", "segments", "opening_hook_strength", "score",
        "opening_self_contained", "hook_claim_resolved",
        "semantic_ending_complete", "ending_rationale_code", "recomposed_for_duration",
    }
    assert set(Stage2SegmentOutput.model_fields) == {
        "role", "start_segment_id", "end_segment_id", "start_anchor_text", "end_anchor_text",
    }
    assert set(Stage2AttemptOutput.model_fields) == {
        "hook_seed_id", "status", "candidate", "reject_reason_code",
    }
    assert set(Stage2FallbackCandidateOutput.model_fields) == {"fallback_id", "candidate"}
    assert set(Stage2Output.model_fields) == {"attempts", "fallback_candidates", "ranking"}


# --- segment count: 1-3 preferred, up to 6 allowed -------------------------
# (real-machine incident: Stage2 designed several 4-segment candidates --
# needed to keep hook->answer->payoff natural -- and the whole Stage2Output
# failed schema validation outright because Stage2CandidateOutput.segments
# was hard-capped at max_length=3. Segment count itself is never a quality
# signal: 1-3 stays the preferred shape, but a well-connected 4-6 segment
# design must be accepted on its own merits, and 7+ is never allowed. See
# config.py's MAX_SEGMENTS_PER_CANDIDATE/PREFERRED_MAX_SEGMENTS_PER_
# CANDIDATE docstring.)


def _stage2_segment(i):
    return Stage2SegmentOutput(role="context", start_segment_id=i, end_segment_id=i)


def test_stage2_candidate_output_accepts_1_segment():
    # A
    kwargs = {**_valid_stage2_candidate_kwargs(), "segments": [_stage2_segment(0)]}
    out = Stage2CandidateOutput(**kwargs)
    assert len(out.segments) == 1


def test_stage2_candidate_output_accepts_3_segments():
    # B
    kwargs = {**_valid_stage2_candidate_kwargs(), "segments": [_stage2_segment(i) for i in range(3)]}
    out = Stage2CandidateOutput(**kwargs)
    assert len(out.segments) == 3


def test_stage2_candidate_output_accepts_4_segments():
    # C: schema parse succeeds -- this used to be a hard ValidationError
    # (max_length=3) that took down the entire Stage2Output response.
    kwargs = {**_valid_stage2_candidate_kwargs(), "segments": [_stage2_segment(i) for i in range(4)]}
    out = Stage2CandidateOutput(**kwargs)
    assert len(out.segments) == 4


def test_stage2_candidate_output_accepts_5_segments():
    # D
    kwargs = {**_valid_stage2_candidate_kwargs(), "segments": [_stage2_segment(i) for i in range(5)]}
    out = Stage2CandidateOutput(**kwargs)
    assert len(out.segments) == 5


def test_stage2_candidate_output_accepts_6_segments():
    # E
    kwargs = {**_valid_stage2_candidate_kwargs(), "segments": [_stage2_segment(i) for i in range(6)]}
    out = Stage2CandidateOutput(**kwargs)
    assert len(out.segments) == 6


def test_stage2_candidate_output_rejects_7_segments():
    # F: 7+ is a hard schema-level ceiling, never allowed.
    kwargs = {**_valid_stage2_candidate_kwargs(), "segments": [_stage2_segment(i) for i in range(7)]}
    with pytest.raises(ValidationError):
        Stage2CandidateOutput(**kwargs)


def test_evaluate_local_candidate_accepts_4_segments_solely_for_junction_safety(monkeypatch):
    # G: a 4-segment candidate must not be rejected merely for its count --
    # only actual defects (unsafe junctions, disfluency, etc.) reject it.
    # Every segment here ends with terminal punctuation and the transcript
    # has clean natural breaks, so nothing else should trip the gate.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = Transcript(
        video_id="vidSeg4", language="ja",
        segments=[
            _segment(i, start=i * 3.0, text=f"項目{i}の話をします。") for i in range(4)
        ],
    )
    raw = RawClipCandidate(
        hook_type="story",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0),
            RawUsedSegment(role="context", start_segment_id=1, end_segment_id=1),
            RawUsedSegment(role="answer", start_segment_id=2, end_segment_id=2),
            RawUsedSegment(role="payoff", start_segment_id=3, end_segment_id=3),
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )
    evaluation = clip_selector.evaluate_local_candidate(raw, transcript)
    assert evaluation.accepted is True
    assert len(evaluation.candidate.segments) == 4


def test_evaluate_local_candidate_accepts_6_segments_when_junctions_are_safe(monkeypatch):
    # H is covered by the schema test above (7 segments rejected at the
    # Structured Output boundary, before local validation even runs); this
    # test is the local-validation counterpart -- 6 well-connected segments
    # must pass exactly like a 1-3 segment candidate would.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = Transcript(
        video_id="vidSeg6", language="ja",
        segments=[
            _segment(i, start=i * 3.0, text=f"項目{i}の話をします。") for i in range(6)
        ],
    )
    raw = RawClipCandidate(
        hook_type="story",
        segments=[RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0)]
        + [RawUsedSegment(role="context", start_segment_id=i, end_segment_id=i) for i in range(1, 6)],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )
    evaluation = clip_selector.evaluate_local_candidate(raw, transcript)
    assert evaluation.accepted is True
    assert len(evaluation.candidate.segments) == 6


def test_evaluate_local_candidate_rejects_6_segments_with_an_unsafe_junction(monkeypatch):
    # I: 6 segments does not grant a pass on junction safety -- an unsafe
    # cut among them must still reject, exactly as it would with 2 segments.
    # Uses a clean-terminal-punctuation transcript (_overlap_transcript_
    # with_clean_endings) so the ending-completeness check never fires
    # first, isolating the overlap check itself.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _overlap_transcript_with_clean_endings()
    raw = RawClipCandidate(
        hook_type="story",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=2, end_segment_id=2),
            RawUsedSegment(role="context", start_segment_id=0, end_segment_id=0),
            RawUsedSegment(role="answer", start_segment_id=2, end_segment_id=2),
            RawUsedSegment(role="payoff", start_segment_id=0, end_segment_id=0),
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )
    evaluation = clip_selector.evaluate_local_candidate(raw, transcript)
    assert evaluation.accepted is False
    assert evaluation.reason == "overlap"


def test_evaluate_local_candidate_rejects_2_segments_with_semantically_broken_junction():
    # J: fewer segments does not grant a free pass either -- a 2-segment
    # candidate whose first segment demands a continuation the second
    # segment doesn't provide must still reject.
    transcript = Transcript(
        video_id="vidSeg2Bad", language="ja",
        segments=[
            _segment_with_words(0, 0.0, "冷やす", "のであれば", "こうしてください。"),
            _segment_with_words(1, 5.0, "連続周回", "をする場合は", "注意が必要です。"),
        ],
    )
    raw = RawClipCandidate(
        hook_type="story",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0),
            RawUsedSegment(role="answer", start_segment_id=1, end_segment_id=1),
        ],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )
    evaluation = clip_selector.evaluate_local_candidate(raw, transcript)
    assert evaluation.accepted is False


def test_evaluate_local_candidate_4_to_6_segments_still_enforces_duration_bounds(monkeypatch):
    # K: segment count relaxation does not loosen the 20-50s hard duration
    # bounds -- a 4-segment candidate whose total exceeds them still rejects.
    # Each segment is individually 15s long (4 * 15 = 60s > 50s hard max) --
    # _candidate_duration sums each segment's own resolved duration, not
    # just the gap between the first and last, so segments must each be
    # long, not merely far apart.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 20.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 50.0)
    transcript = Transcript(
        video_id="vidSeg4Duration", language="ja",
        segments=[
            TranscriptSegment(
                id=i, start=i * 15.0, end=i * 15.0 + 15.0, text=f"項目{i}の話をします。",
                words=[TranscriptWord(start=i * 15.0, end=i * 15.0 + 15.0, text=f"項目{i}の話をします。")],
            )
            for i in range(4)
        ],
    )
    raw = RawClipCandidate(
        hook_type="story",
        segments=[RawUsedSegment(role=r, start_segment_id=i, end_segment_id=i) for i, r in enumerate(
            ["hook", "context", "answer", "payoff"]
        )],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )
    evaluation = clip_selector.evaluate_local_candidate(raw, transcript)
    assert evaluation.accepted is False
    assert evaluation.reason == "duration_too_long"


def test_stage2_diagnostic_evaluation_flags_high_segment_count(monkeypatch):
    # Diagnostic-only fields (item 10): segment_count/preferred_segment_
    # range_met/high_segment_count must never affect accepted, only report.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = Transcript(
        video_id="vidSegDiag", language="ja",
        segments=[
            _segment(i, start=i * 3.0, text=f"項目{i}の話をします。") for i in range(4)
        ],
    )
    raw = RawClipCandidate(
        hook_type="story",
        segments=[RawUsedSegment(role=r, start_segment_id=i, end_segment_id=i) for i, r in enumerate(
            ["hook", "context", "answer", "payoff"]
        )],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )
    evaluation = clip_selector.evaluate_local_candidate(raw, transcript)
    assert evaluation.accepted is True
    diagnostic = clip_selector._stage2_diagnostic_evaluation(evaluation)
    assert diagnostic["segment_count"] == 4
    assert diagnostic["preferred_segment_range_met"] is False
    assert diagnostic["high_segment_count"] is True


def test_stage2_diagnostic_evaluation_does_not_flag_3_segments_as_high_count(monkeypatch):
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = Transcript(
        video_id="vidSegDiag3", language="ja",
        segments=[
            _segment(i, start=i * 3.0, text=f"項目{i}の話をします。") for i in range(3)
        ],
    )
    raw = RawClipCandidate(
        hook_type="story",
        segments=[RawUsedSegment(role=r, start_segment_id=i, end_segment_id=i) for i, r in enumerate(
            ["hook", "context", "answer"]
        )],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )
    evaluation = clip_selector.evaluate_local_candidate(raw, transcript)
    assert evaluation.accepted is True
    diagnostic = clip_selector._stage2_diagnostic_evaluation(evaluation)
    assert diagnostic["segment_count"] == 3
    assert diagnostic["preferred_segment_range_met"] is True
    assert diagnostic["high_segment_count"] is False


def test_select_final_candidates_treats_fallback_4_to_6_segments_like_primary(monkeypatch):
    # K (fallback side): fallback candidates go through the exact same
    # segment-count rule as primary ones -- 4-6 segments is not restricted
    # to 3 for fallback just because they're the safety net.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = Transcript(
        video_id="vidFallbackSeg", language="ja",
        segments=[
            _segment(i, start=i * 3.0, text=f"項目{i}の話をします。") for i in range(5)
        ],
    )
    raw = RawClipCandidate(
        hook_type="story",
        segments=[RawUsedSegment(role="hook" if i == 0 else "context", start_segment_id=i, end_segment_id=i)
                  for i in range(5)],
        hook_text="h", opening_hook_strength=90, title="", description="",
        score=90, reasoning="", caveats="",
    )
    evaluation = clip_selector.evaluate_local_candidate_with_repair(raw, transcript)
    fallback_evals = {"fb0": evaluation}
    final = clip_selector._select_final_candidates({}, fallback_evals, ["fb0"])
    assert len(final) == 1
    assert final[0][0] == "fb0"
    assert len(final[0][1].segments) == 5


def test_stage2_attempt_output_requires_candidate_when_status_is_candidate():
    with pytest.raises(ValidationError):
        Stage2AttemptOutput(hook_seed_id="s1_hookseed_000", status="candidate")


def test_stage2_attempt_output_forbids_candidate_when_status_is_rejected():
    with pytest.raises(ValidationError):
        Stage2AttemptOutput(
            hook_seed_id="s1_hookseed_000", status="rejected",
            candidate=Stage2CandidateOutput(**_valid_stage2_candidate_kwargs()),
            reject_reason_code="insufficient_context_available",
        )


def test_stage2_attempt_output_requires_reject_reason_code_when_status_is_rejected():
    with pytest.raises(ValidationError):
        Stage2AttemptOutput(hook_seed_id="s1_hookseed_000", status="rejected")


def test_stage2_attempt_output_forbids_reject_reason_code_when_status_is_candidate():
    with pytest.raises(ValidationError):
        Stage2AttemptOutput(
            hook_seed_id="s1_hookseed_000", status="candidate",
            candidate=Stage2CandidateOutput(**_valid_stage2_candidate_kwargs()),
            reject_reason_code="insufficient_context_available",
        )


def test_stage2_attempt_output_accepts_valid_candidate_and_rejected_shapes():
    Stage2AttemptOutput(**_valid_stage2_attempt_kwargs("s1_hookseed_000", status="candidate"))
    Stage2AttemptOutput(**_valid_stage2_attempt_kwargs("s1_hookseed_000", status="rejected"))


def test_stage2_output_accepts_up_to_stage2_max_coverage_targets():
    """coverage-target redesign: Stage2Output.attempts is capped at
    config.STAGE2_MAX_COVERAGE_TARGETS -- matching the number of coverage
    targets Python will ever actually send (see
    _select_hook_seed_coverage_targets), not an independent design ceiling.
    """
    assert config.STAGE2_MAX_COVERAGE_TARGETS == 8
    for n in range(1, config.STAGE2_MAX_COVERAGE_TARGETS + 1):
        out = Stage2Output(
            attempts=[
                Stage2AttemptOutput(**_valid_stage2_attempt_kwargs(f"s1_hookseed_{i:03d}"))
                for i in range(n)
            ],
            fallback_candidates=[], ranking=[],
        )
        assert len(out.attempts) == n
    with pytest.raises(ValidationError):
        Stage2Output(
            attempts=[
                Stage2AttemptOutput(**_valid_stage2_attempt_kwargs(f"s1_hookseed_{i:03d}"))
                for i in range(config.STAGE2_MAX_COVERAGE_TARGETS + 1)
            ],
            fallback_candidates=[], ranking=[],
        )


def test_stage2_output_accepts_up_to_stage2_max_fallback_candidates():
    assert config.STAGE2_MAX_FALLBACK_CANDIDATES == config.NUM_CANDIDATES
    for n in range(1, config.STAGE2_MAX_FALLBACK_CANDIDATES + 1):
        out = Stage2Output(
            attempts=[],
            fallback_candidates=[
                Stage2FallbackCandidateOutput(**_valid_stage2_fallback_candidate_kwargs(f"fb{i}"))
                for i in range(n)
            ],
            ranking=[],
        )
        assert len(out.fallback_candidates) == n
    with pytest.raises(ValidationError):
        Stage2Output(
            attempts=[],
            fallback_candidates=[
                Stage2FallbackCandidateOutput(**_valid_stage2_fallback_candidate_kwargs(f"fb{i}"))
                for i in range(config.STAGE2_MAX_FALLBACK_CANDIDATES + 1)
            ],
            ranking=[],
        )


def test_stage2_output_max_json_size_is_well_under_max_tokens():
    """Guards against the worst-case Stage2Output JSON (STAGE2_MAX_
    COVERAGE_TARGETS attempts + STAGE2_MAX_FALLBACK_CANDIDATES fallback
    candidates, each with a full config.MAX_SEGMENTS_PER_CANDIDATE(6)-
    segment candidate, long enum values, 3-digit segment ids, a max-length
    anchor on both ends of every segment, plus a ranking list) approaching
    STAGE2_MAX_OUTPUT_TOKENS closely enough to risk the same stop_reason=
    "max_tokens" truncation this codebase has hit before.

    Re-audited for the segment-count relaxation round (Stage2CandidateOutput
    .segments' max_length went 3->6, see config.MAX_SEGMENTS_PER_CANDIDATE's
    docstring): worst case grew accordingly since a single candidate can now
    carry twice as many segments.
    """
    roles = ["hook", "context", "context", "context", "answer", "payoff"]
    kana = ["あ", "い", "う", "え", "お", "か"]
    candidate = Stage2CandidateOutput(
        hook_type="surprising_fact",
        segments=[
            Stage2SegmentOutput(
                role=roles[i], start_segment_id=100 + 2 * i, end_segment_id=101 + 2 * i,
                start_anchor_text=kana[i] * 60, end_anchor_text=kana[(i + 1) % 6] * 60,
            )
            for i in range(config.MAX_SEGMENTS_PER_CANDIDATE)
        ],
        opening_hook_strength=95,
        score=92,
        opening_self_contained=True,
        hook_claim_resolved=True,
        semantic_ending_complete=True,
        ending_rationale_code="recomposed_for_duration",
        recomposed_for_duration=True,
    )
    attempts = [
        Stage2AttemptOutput(hook_seed_id=f"s1_hookseed_{i:03d}", status="candidate", candidate=candidate)
        for i in range(config.STAGE2_MAX_COVERAGE_TARGETS)
    ]
    fallback_candidates = [
        Stage2FallbackCandidateOutput(fallback_id=f"fb{i}", candidate=candidate)
        for i in range(config.STAGE2_MAX_FALLBACK_CANDIDATES)
    ]
    ranking = [a.hook_seed_id for a in attempts] + [f.fallback_id for f in fallback_candidates]
    worst_case = Stage2Output(attempts=attempts, fallback_candidates=fallback_candidates, ranking=ranking)
    text = worst_case.model_dump_json()

    char_count = len(text)
    estimated_tokens = char_count / 3.5

    assert estimated_tokens < config.STAGE2_MAX_OUTPUT_TOKENS / 2, (
        f"worst-case Stage2Output JSON is {char_count} chars (~{estimated_tokens:.0f} "
        f"estimated tokens) -- unexpectedly close to STAGE2_MAX_OUTPUT_TOKENS="
        f"{config.STAGE2_MAX_OUTPUT_TOKENS}."
    )


def _valid_material_segment_kwargs():
    return {"start_segment_id": 0, "end_segment_id": 0}


def _valid_support_material_kwargs():
    return {
        "material_type": "reason", "segments": [_valid_material_segment_kwargs()],
        "usefulness_score": 80,
    }


def _valid_hook_seed_kwargs():
    return {
        "signal_type": "money_or_number", "segments": [_valid_material_segment_kwargs()],
        "soft_score": 80,
    }


def _valid_fallback_span_kwargs():
    return {"segments": [_valid_material_segment_kwargs()], "safety_score": 60}


def test_stage1_output_accepts_zero_to_max_hook_seeds_support_materials_fallback_spans():
    """Stage1's per-chunk output is now three independently-capped groups
    (hook_seeds/support_materials/fallback_spans), not one shared list --
    each caps out at its own config constant.
    """
    assert config.STAGE1_MAX_HOOK_SEEDS_PER_CHUNK == 6
    assert config.STAGE1_MAX_SUPPORT_MATERIALS_PER_CHUNK == 6
    assert config.STAGE1_MAX_FALLBACK_SPANS_PER_CHUNK == 2

    empty = Stage1Output(hook_seeds=[], support_materials=[], fallback_spans=[])
    assert empty.hook_seeds == [] and empty.support_materials == [] and empty.fallback_spans == []

    for n in range(1, config.STAGE1_MAX_HOOK_SEEDS_PER_CHUNK + 1):
        out = Stage1Output(
            hook_seeds=[Stage1HookSeedOutput(**_valid_hook_seed_kwargs()) for _ in range(n)],
            support_materials=[], fallback_spans=[],
        )
        assert len(out.hook_seeds) == n
    with pytest.raises(ValidationError):
        Stage1Output(
            hook_seeds=[
                Stage1HookSeedOutput(**_valid_hook_seed_kwargs())
                for _ in range(config.STAGE1_MAX_HOOK_SEEDS_PER_CHUNK + 1)
            ],
            support_materials=[], fallback_spans=[],
        )

    for n in range(1, config.STAGE1_MAX_SUPPORT_MATERIALS_PER_CHUNK + 1):
        out = Stage1Output(
            hook_seeds=[],
            support_materials=[Stage1SupportMaterialOutput(**_valid_support_material_kwargs()) for _ in range(n)],
            fallback_spans=[],
        )
        assert len(out.support_materials) == n
    with pytest.raises(ValidationError):
        Stage1Output(
            hook_seeds=[], fallback_spans=[],
            support_materials=[
                Stage1SupportMaterialOutput(**_valid_support_material_kwargs())
                for _ in range(config.STAGE1_MAX_SUPPORT_MATERIALS_PER_CHUNK + 1)
            ],
        )

    for n in range(1, config.STAGE1_MAX_FALLBACK_SPANS_PER_CHUNK + 1):
        out = Stage1Output(
            hook_seeds=[], support_materials=[],
            fallback_spans=[Stage1FallbackSpanOutput(**_valid_fallback_span_kwargs()) for _ in range(n)],
        )
        assert len(out.fallback_spans) == n
    with pytest.raises(ValidationError):
        Stage1Output(
            hook_seeds=[], support_materials=[],
            fallback_spans=[
                Stage1FallbackSpanOutput(**_valid_fallback_span_kwargs())
                for _ in range(config.STAGE1_MAX_FALLBACK_SPANS_PER_CHUNK + 1)
            ],
        )


def test_stage1_output_max_json_size_is_well_under_max_tokens():
    """Worst-case Stage1Output JSON: max hook_seeds (2 segments each) +
    max support_materials (3 segments each) + max fallback_spans (3
    segments each), long enum values, a max-length anchor on every
    segment -- must stay comfortably under half of STAGE1_MAX_OUTPUT_
    TOKENS to avoid the stop_reason="max_tokens" truncation risk this
    codebase has hit before.
    """
    seg = {"start_segment_id": 123, "end_segment_id": 124, "start_anchor_text": "あ" * 60}
    hook_seeds = [
        Stage1HookSeedOutput(signal_type="surprising_fact", segments=[seg, seg], soft_score=90)
        for _ in range(config.STAGE1_MAX_HOOK_SEEDS_PER_CHUNK)
    ]
    support_materials = [
        Stage1SupportMaterialOutput(material_type="context", segments=[seg, seg, seg], usefulness_score=90)
        for _ in range(config.STAGE1_MAX_SUPPORT_MATERIALS_PER_CHUNK)
    ]
    fallback_spans = [
        Stage1FallbackSpanOutput(segments=[seg, seg, seg], safety_score=90)
        for _ in range(config.STAGE1_MAX_FALLBACK_SPANS_PER_CHUNK)
    ]
    worst_case = Stage1Output(
        hook_seeds=hook_seeds, support_materials=support_materials, fallback_spans=fallback_spans
    )
    text = worst_case.model_dump_json()

    char_count = len(text)
    estimated_tokens = char_count / 3.5

    assert estimated_tokens < config.STAGE1_MAX_OUTPUT_TOKENS / 2, (
        f"worst-case Stage1Output JSON is {char_count} chars (~{estimated_tokens:.0f} "
        f"estimated tokens) -- unexpectedly close to STAGE1_MAX_OUTPUT_TOKENS="
        f"{config.STAGE1_MAX_OUTPUT_TOKENS}."
    )


def test_stage1_hook_seed_output_segments_length_bounds():
    for n in (1, 2):
        kwargs = _valid_hook_seed_kwargs()
        kwargs["segments"] = [_valid_material_segment_kwargs() for _ in range(n)]
        Stage1HookSeedOutput(**kwargs)
    for n in (0, 3):
        kwargs = _valid_hook_seed_kwargs()
        kwargs["segments"] = [_valid_material_segment_kwargs() for _ in range(n)]
        with pytest.raises(ValidationError):
            Stage1HookSeedOutput(**kwargs)


def test_stage1_support_material_output_segments_length_bounds():
    for n in (1, 2, 3):
        kwargs = _valid_support_material_kwargs()
        kwargs["segments"] = [_valid_material_segment_kwargs() for _ in range(n)]
        Stage1SupportMaterialOutput(**kwargs)
    for n in (0, 4):
        kwargs = _valid_support_material_kwargs()
        kwargs["segments"] = [_valid_material_segment_kwargs() for _ in range(n)]
        with pytest.raises(ValidationError):
            Stage1SupportMaterialOutput(**kwargs)


def test_stage1_support_material_output_usefulness_score_bounds():
    for value in (0, 100):
        kwargs = _valid_support_material_kwargs()
        kwargs["usefulness_score"] = value
        Stage1SupportMaterialOutput(**kwargs)
    for value in (-1, 101):
        kwargs = _valid_support_material_kwargs()
        kwargs["usefulness_score"] = value
        with pytest.raises(ValidationError):
            Stage1SupportMaterialOutput(**kwargs)


def test_stage1_support_material_output_rejects_invalid_material_type():
    kwargs = _valid_support_material_kwargs()
    kwargs["material_type"] = "not_a_real_material_type"
    with pytest.raises(ValidationError):
        Stage1SupportMaterialOutput(**kwargs)


@pytest.mark.parametrize("material_type", ["reason", "example", "context", "payoff"])
def test_stage1_support_material_output_accepts_every_material_type(material_type):
    kwargs = _valid_support_material_kwargs()
    kwargs["material_type"] = material_type
    Stage1SupportMaterialOutput(**kwargs)


@pytest.mark.parametrize("signal_type", [
    "money_or_number", "failure_or_loss", "surprising_fact", "strong_claim",
    "comparison", "direct_question", "strong_conclusion", "story_turn", "other",
])
def test_stage1_hook_seed_output_accepts_every_signal_type(signal_type):
    kwargs = _valid_hook_seed_kwargs()
    kwargs["signal_type"] = signal_type
    Stage1HookSeedOutput(**kwargs)


def test_stage1_hook_seed_output_rejects_invalid_signal_type():
    kwargs = _valid_hook_seed_kwargs()
    kwargs["signal_type"] = "not_a_real_signal_type"
    with pytest.raises(ValidationError):
        Stage1HookSeedOutput(**kwargs)


def test_stage1_material_segment_output_rejects_wrongly_typed_segment_id():
    with pytest.raises(ValidationError):
        Stage1MaterialSegmentOutput(start_segment_id=["not", "an", "int"], end_segment_id=0)


def test_stage1_support_material_output_rejects_unknown_fields():
    # extra="forbid" -> additionalProperties: false in the schema sent to
    # Claude, and the same strictness applies locally.
    kwargs = _valid_support_material_kwargs()
    kwargs["hook_type"] = "should not be accepted"
    with pytest.raises(ValidationError):
        Stage1SupportMaterialOutput(**kwargs)


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
    output = Stage1Output(
        hook_seeds=[], support_materials=[Stage1SupportMaterialOutput(**_valid_support_material_kwargs())],
        fallback_spans=[],
    )
    monkeypatch.setattr(
        clip_selector.structured_output, "call",
        lambda schema_model, **kwargs: output,
    )

    segments = [_segment(0, start=0.0, text="強い発言です")]
    result = clip_selector.extract_candidates_for_chunk(segments, "タイトル")

    assert isinstance(result, Stage1ChunkResult)
    assert result.hook_seeds == []
    assert result.fallback_spans == []
    assert len(result.support_materials) == 1
    assert isinstance(result.support_materials[0], RawMaterial)
    assert result.support_materials[0].material_type == "reason"
    assert result.support_materials[0].usefulness_score == 80
    # A material has none of the finished-candidate display/scoring
    # properties -- those only ever exist on RawClipCandidate, produced
    # solely by Stage2's conversion (_raw_candidate_from_stage2_output).
    assert not hasattr(result.support_materials[0], "hook_text")
    assert not hasattr(result.support_materials[0], "title")


def test_extract_candidates_for_chunk_carries_anchor_text_through(monkeypatch):
    # The material conversion must preserve start_anchor_text verbatim --
    # boundary.py verifies/applies it later, at Stage2-design-conversion
    # and resolve time, exactly as it always has for Stage1 output.
    kwargs = _valid_support_material_kwargs()
    kwargs["segments"] = [
        {"start_segment_id": 0, "end_segment_id": 0, "start_anchor_text": "86は"}
    ]
    output = Stage1Output(
        hook_seeds=[], support_materials=[Stage1SupportMaterialOutput(**kwargs)], fallback_spans=[],
    )
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

    material = result.support_materials[0]
    assert material.segments[0].start_anchor_text == "86は"


def test_design_final_candidates_converts_stage2_output_to_raw_candidates(monkeypatch):
    # Stage2 now designs final candidates directly (real segment_id
    # references it may freely recombine) via an attempt-per-coverage-
    # target -- the conversion mirrors _raw_candidate_from_stage1_output
    # exactly, plus end_anchor_text.
    transcript = _long_transcript(minutes=1)
    coverage_targets = [("s1_hookseed_000", _raw_hook_seed(0, 2))]
    materials = {"s1_m000": _raw_material(0, 2)}
    output = Stage2Output(
        attempts=[
            Stage2AttemptOutput(
                hook_seed_id="s1_hookseed_000", status="candidate",
                candidate=Stage2CandidateOutput(
                    **{
                        **_valid_stage2_candidate_kwargs(),
                        "hook_type": "story",
                        "segments": [Stage2SegmentOutput(role="hook", start_segment_id=0, end_segment_id=2)],
                        "opening_hook_strength": 85,
                        "score": 85,
                    }
                ),
            )
        ],
        fallback_candidates=[], ranking=["s1_hookseed_000"],
    )
    monkeypatch.setattr(clip_selector.structured_output, "call", lambda schema_model, **kwargs: output)

    design = clip_selector.design_final_candidates(coverage_targets, materials, {}, transcript, "タイトル")

    assert len(design.attempts) == 1
    designed = design.attempts[0].candidate
    assert designed.hook_type == "story"
    assert designed.opening_hook_strength == 85
    assert designed.score == 85
    assert designed.segments == [
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
    coverage_targets = [("s1_hookseed_000", _raw_hook_seed(0, 0))]  # a strong standalone conclusion
    materials = {
        "s1_m001": _raw_material(2, 2, material_type="reason"),  # a separate reason material
    }
    output = Stage2Output(
        attempts=[
            Stage2AttemptOutput(
                hook_seed_id="s1_hookseed_000", status="candidate",
                candidate=Stage2CandidateOutput(
                    **{
                        **_valid_stage2_candidate_kwargs(),
                        "hook_type": "strong_take",
                        "segments": [
                            Stage2SegmentOutput(role="hook", start_segment_id=0, end_segment_id=0),
                            Stage2SegmentOutput(role="answer", start_segment_id=2, end_segment_id=2),
                        ],
                        "opening_hook_strength": 90,
                        "score": 90,
                    }
                ),
            )
        ],
        fallback_candidates=[], ranking=["s1_hookseed_000"],
    )
    monkeypatch.setattr(clip_selector.structured_output, "call", lambda schema_model, **kwargs: output)

    design = clip_selector.design_final_candidates(coverage_targets, materials, {}, transcript, "タイトル")

    assert len(design.attempts) == 1
    designed = design.attempts[0].candidate
    assert [s.start_segment_id for s in designed.segments] == [0, 2]
    assert [s.role for s in designed.segments] == ["hook", "answer"]


def test_design_final_candidates_carries_end_anchor_text_through(monkeypatch):
    # Stage2SegmentOutput.end_anchor_text is the one genuinely new field
    # versus Stage1SegmentOutput -- confirm it survives the conversion
    # into RawUsedSegment (boundary.py already knows how to verify/apply
    # it, unchanged since the duration-repair round).
    transcript = _long_transcript(minutes=1)
    coverage_targets = [("s1_hookseed_000", _raw_hook_seed(0, 2))]
    output = Stage2Output(
        attempts=[
            Stage2AttemptOutput(
                hook_seed_id="s1_hookseed_000", status="candidate",
                candidate=Stage2CandidateOutput(
                    **{
                        **_valid_stage2_candidate_kwargs(),
                        "hook_type": "story",
                        "segments": [
                            Stage2SegmentOutput(
                                role="hook", start_segment_id=0, end_segment_id=2,
                                start_anchor_text="segment 0", end_anchor_text="segment 2",
                            )
                        ],
                        "opening_hook_strength": 85,
                        "score": 85,
                    }
                ),
            )
        ],
        fallback_candidates=[], ranking=["s1_hookseed_000"],
    )
    monkeypatch.setattr(clip_selector.structured_output, "call", lambda schema_model, **kwargs: output)

    design = clip_selector.design_final_candidates(coverage_targets, {}, {}, transcript, "タイトル")

    designed = design.attempts[0].candidate
    assert designed.segments[0].start_anchor_text == "segment 0"
    assert designed.segments[0].end_anchor_text == "segment 2"


def test_design_final_candidates_does_not_send_full_transcript(monkeypatch):
    # F: Stage2 only sees a compact per-material summary -- an
    # unreferenced transcript segment's distinctive text must never
    # appear in what gets sent to the API.
    transcript = _long_transcript(minutes=5)
    transcript.segments[-1].text = "この文言はどの候補にも含まれない特徴的な発言マーカーXYZ123"
    coverage_targets = [("s1_hookseed_000", _raw_hook_seed(0, 2))]
    materials = {"s1_m000": _raw_material(0, 2)}

    captured = {}

    def _spy(schema_model, *, stage, system_prompt, user_content, max_tokens):
        captured["user_content"] = user_content
        return Stage2Output(attempts=[], fallback_candidates=[], ranking=[])

    monkeypatch.setattr(clip_selector.structured_output, "call", _spy)
    clip_selector.design_final_candidates(coverage_targets, materials, {}, transcript, "タイトル")

    assert "マーカーXYZ123" not in captured["user_content"]


# NOTE: the exact-repeat-design dedup safety net now lives in
# _select_final_candidates (see test_select_final_candidates_dedupes_
# identical_segment_sequences), not in design_final_candidates itself --
# see test_design_final_candidates_saves_raw_diagnostic_before_local_
# validation, which proves design_final_candidates deliberately returns
# every attempt Stage2 built, undeduped, so the diagnostic snapshot always
# reflects Stage2's raw output.


# --- semantic-ending-design round: lookahead exposure + Stage2's own -----
# --- ending-completeness self-report (real-machine incident: a candidate -
# --- ended ~23s mid-explanation despite passing duration/hook/junction ---
# --- checks -- Stage2 had no visibility past a material's own segments) --


def test_material_lookahead_segments_returns_following_real_segments():
    # A: the concrete fix for "Stage2 can't see past its own chosen
    # segments" -- lookahead surfaces real transcript segments after the
    # material's own last segment, as plain reference data.
    transcript = _long_transcript(minutes=5)
    material = _raw_material(0, 0)

    lookahead = clip_selector._material_lookahead_segments(material, transcript)

    assert len(lookahead) == config.STAGE2_LOOKAHEAD_MAX_SEGMENTS
    assert [seg["segment_id"] for seg in lookahead] == [1, 2, 3, 4]
    assert lookahead[0]["text"] == "segment 1"
    assert lookahead[0]["start_sec"] == 20.0


def test_material_lookahead_segments_bounded_by_segment_count(monkeypatch):
    transcript = _long_transcript(minutes=5)
    material = _raw_material(0, 0)
    monkeypatch.setattr(config, "STAGE2_LOOKAHEAD_MAX_SEGMENTS", 2)

    lookahead = clip_selector._material_lookahead_segments(material, transcript)

    assert len(lookahead) == 2


def test_material_lookahead_segments_bounded_by_cumulative_seconds(monkeypatch):
    # Each _long_transcript segment is 2.0s long; capping the seconds
    # budget below one full segment's duration must still not return zero
    # segments outright once at least one has been included -- the cap
    # only stops adding *further* segments once the budget is exceeded.
    transcript = _long_transcript(minutes=5)
    material = _raw_material(0, 0)
    monkeypatch.setattr(config, "STAGE2_LOOKAHEAD_MAX_SEC", 3.0)

    lookahead = clip_selector._material_lookahead_segments(material, transcript)

    assert len(lookahead) == 2  # first segment (2.0s) included, then cumulative >= 3.0s stops it


def test_material_lookahead_segments_always_includes_at_least_one_when_available():
    # Even a single very long following segment (alone exceeding the
    # seconds budget) must not leave Stage2 with an empty, uninformative
    # lookahead just because the very next segment happens to be long.
    transcript = _long_transcript(minutes=5)
    transcript.segments[1] = TranscriptSegment(
        id=1, start=20.0, end=20.0 + config.STAGE2_LOOKAHEAD_MAX_SEC + 10.0, text="very long segment",
        words=[TranscriptWord(start=20.0, end=20.0 + config.STAGE2_LOOKAHEAD_MAX_SEC + 10.0, text="very long segment")],
    )
    material = _raw_material(0, 0)

    lookahead = clip_selector._material_lookahead_segments(material, transcript)

    assert len(lookahead) == 1
    assert lookahead[0]["segment_id"] == 1


def test_material_lookahead_segments_empty_at_end_of_transcript():
    transcript = _long_transcript(minutes=5)
    last_id = transcript.segments[-1].id
    material = _raw_material(last_id, last_id)

    assert clip_selector._material_lookahead_segments(material, transcript) == []


def test_stage2_material_summary_includes_lookahead_key():
    transcript = _long_transcript(minutes=5)
    material = _raw_material(0, 0)

    summary = clip_selector._stage2_material_summary("s1_m000", material, transcript)

    assert "lookahead" in summary
    assert summary["lookahead"][0]["segment_id"] == 1


def test_design_final_candidates_stage2_can_reference_lookahead_segment_as_final(monkeypatch):
    # H (plumbing half): Stage2 is free to reference a lookahead segment_id
    # as part of its own designed final candidate -- referential integrity
    # is judged the same way regardless of whether an id came from a
    # material's own segments or its lookahead, since segment_ids are
    # transcript-global. This doesn't (and can't, without a real API call)
    # test that Stage2 *chooses well* -- only that the plumbing lets it
    # extend into lookahead content when it does choose to.
    transcript = _long_transcript(minutes=5)
    coverage_targets = [("s1_hookseed_000", _raw_hook_seed(0, 0))]
    output = Stage2Output(
        attempts=[
            Stage2AttemptOutput(
                hook_seed_id="s1_hookseed_000", status="candidate",
                candidate=Stage2CandidateOutput(
                    **{
                        **_valid_stage2_candidate_kwargs(),
                        "segments": [
                            Stage2SegmentOutput(role="hook", start_segment_id=0, end_segment_id=0),
                            Stage2SegmentOutput(role="answer", start_segment_id=1, end_segment_id=1),
                        ],
                    }
                ),
            )
        ],
        fallback_candidates=[], ranking=["s1_hookseed_000"],
    )
    monkeypatch.setattr(clip_selector.structured_output, "call", lambda schema_model, **kwargs: output)

    design = clip_selector.design_final_candidates(coverage_targets, {}, {}, transcript, "タイトル")

    assert [s.start_segment_id for s in design.attempts[0].candidate.segments] == [0, 1]


def test_design_final_candidates_saves_materials_lookahead_to_diagnostic(monkeypatch):
    # Item 12: materials_lookahead must land in stage2_diagnostic.json so a
    # future mid-cutoff incident is debuggable -- what lookahead Stage2 had
    # available, not just what it ultimately chose.
    transcript = _long_transcript(minutes=5)
    coverage_targets = [("s1_hookseed_000", _raw_hook_seed(0, 0))]
    output = _stage2_output_single_attempt("s1_hookseed_000")
    monkeypatch.setattr(clip_selector.structured_output, "call", lambda schema_model, **kwargs: output)

    clip_selector.design_final_candidates(coverage_targets, {}, {}, transcript, "タイトル")

    diagnostic = cache.load_stage2_diagnostic(transcript.video_id)
    assert "materials_lookahead" in diagnostic
    assert diagnostic["materials_lookahead"]["s1_hookseed_000"][0]["segment_id"] == 1


def test_design_finalize_and_cache_preserves_materials_lookahead_after_second_save(monkeypatch):
    # The second diagnostic save (inside _design_finalize_and_cache, with
    # per-candidate evaluations attached) must not silently drop the
    # materials_lookahead the first save already wrote -- cache.
    # save_stage2_diagnostic merges it in when not re-passed.
    transcript = _long_transcript(minutes=5)
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    hook_seeds = [_raw_hook_seed(0, 0)]
    output = _stage2_output_single_attempt("s1_hookseed_000")
    monkeypatch.setattr(clip_selector.structured_output, "call", lambda schema_model, **kwargs: output)

    clip_selector._design_finalize_and_cache(hook_seeds, [], [], transcript, "タイトル")

    diagnostic = cache.load_stage2_diagnostic(transcript.video_id)
    assert "materials_lookahead" in diagnostic
    assert "evaluations" in diagnostic
    assert diagnostic["materials_lookahead"]["s1_hookseed_000"][0]["segment_id"] == 1


def test_stage2_diagnostic_evaluation_includes_ending_design_fields(monkeypatch):
    # Item 12/13: the per-candidate diagnostic must expose Stage2's own
    # ending-point decision and rationale, not just accept/reject.
    transcript = _long_transcript(minutes=5)
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    hook_seeds = [_raw_hook_seed(0, 0)]
    output = _stage2_output_single_attempt(
        "s1_hookseed_000",
        candidate_overrides={"ending_rationale_code": "hook_resolved", "recomposed_for_duration": True},
    )
    monkeypatch.setattr(clip_selector.structured_output, "call", lambda schema_model, **kwargs: output)

    clip_selector._design_finalize_and_cache(hook_seeds, [], [], transcript, "タイトル")

    diagnostic = cache.load_stage2_diagnostic(transcript.video_id)
    evaluation = diagnostic["evaluations"][0]
    assert evaluation["ending_rationale_code"] == "hook_resolved"
    assert evaluation["recomposed_for_duration"] is True
    assert evaluation["semantic_ending_complete"] is True
    assert "selected_end_segment_id" in evaluation


def test_raw_clip_candidate_defaults_semantic_ending_fields():
    # Every pre-existing direct RawClipCandidate(...) construction across
    # the test suite (hundreds of call sites) must keep working unchanged
    # -- these three fields are defaulted, never required, on the internal
    # dataclass (only Stage2CandidateOutput, the Claude-facing schema,
    # requires them).
    raw = _raw_candidate(0, 0)
    assert raw.semantic_ending_complete is True
    assert raw.ending_rationale_code == "natural_conclusion"
    assert raw.recomposed_for_duration is False


# --- Stage2 semantic closure hard gate (real-machine incident: a --------
# --- candidate whose hook posed "why is X better than Y" ranked high ---
# --- despite its body never explaining why -- see rank_and_finalize.md) -


def test_design_final_candidates_supports_omitting_a_closure_failing_design(monkeypatch):
    # Item 12/K (revised for the hook-seed-coverage redesign): the old
    # architecture let Stage2 express semantic-closure failure by simply
    # omitting the design from its output (min_length=0 on the candidate
    # list). The whole point of the coverage-target redesign is that this
    # is no longer allowed -- a coverage target that fails semantic closure
    # must come back as an explicit status="rejected" attempt (see
    # Stage2AttemptOutput's model_validator and rank_and_finalize.md's
    # anti-silent-skip instruction), never a silently missing id.
    transcript = _long_transcript(minutes=1)
    coverage_targets = [
        ("s1_hookseed_000", _raw_hook_seed(0, 2)),  # hook + real reason: passes closure
        ("s1_hookseed_001", _raw_hook_seed(0, 2)),  # hook only, no reason: Stage2 rejects this one
    ]
    output = Stage2Output(
        attempts=[
            Stage2AttemptOutput(
                hook_seed_id="s1_hookseed_000", status="candidate",
                candidate=Stage2CandidateOutput(
                    **{
                        **_valid_stage2_candidate_kwargs(),
                        "hook_type": "story",
                        "segments": [Stage2SegmentOutput(role="hook", start_segment_id=0, end_segment_id=2)],
                        "opening_hook_strength": 85,
                        "score": 85,
                    }
                ),
            ),
            Stage2AttemptOutput(
                hook_seed_id="s1_hookseed_001", status="rejected",
                reject_reason_code="semantic_closure_unavailable",
            ),
        ],
        fallback_candidates=[], ranking=["s1_hookseed_000"],
    )
    monkeypatch.setattr(clip_selector.structured_output, "call", lambda schema_model, **kwargs: output)

    design = clip_selector.design_final_candidates(coverage_targets, {}, {}, transcript, "タイトル")

    assert len(design.attempts) == 2
    accepted = [a for a in design.attempts if a.status == "candidate"]
    rejected = [a for a in design.attempts if a.status == "rejected"]
    assert len(accepted) == 1
    assert len(rejected) == 1
    assert rejected[0].reject_reason_code == "semantic_closure_unavailable"


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
    hook_seeds = [_raw_hook_seed(0, 2) for _ in range(3)]
    monkeypatch.setattr(
        clip_selector, "run_stage1",
        lambda *a, **k: Stage1ChunkResult(hook_seeds=hook_seeds, support_materials=[], fallback_spans=[]),
    )

    call_count = {"n": 0}

    def _fake_design(coverage_targets, materials, fallback_spans, t, title):
        call_count["n"] += 1
        return _stage2_design_result_candidates([_raw_candidate(0, 2)])

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
    hook_seeds = [_raw_hook_seed(0, 0), _raw_hook_seed(0, 1), _raw_hook_seed(0, 2)]
    cache.save_stage1_chunk(
        transcript.video_id, 0,
        Stage1ChunkResult(hook_seeds=hook_seeds, support_materials=[], fallback_spans=[]),
    )

    call_count = {"n": 0}

    def _fake_design(coverage_targets, materials, fallback_spans, t, title):
        call_count["n"] += 1
        return _stage2_design_result_candidates(
            [_raw_candidate(0, 0, opening_hook_strength=90), _raw_candidate(0, 1, opening_hook_strength=90),
             _raw_candidate(0, 2, opening_hook_strength=90)]
        )

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
    cache.save_stage1_chunk(
        transcript.video_id, 0,
        Stage1ChunkResult(hook_seeds=[_raw_hook_seed(0, 0)], support_materials=[], fallback_spans=[]),
    )

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
    stale_bad = Stage1ChunkResult(
        hook_seeds=[_raw_hook_seed(0, 0, soft_score=10)], support_materials=[], fallback_spans=[],
    )
    cache.save_stage1_chunk(transcript.video_id, 0, stale_bad)

    fresh_good = Stage1ChunkResult(
        hook_seeds=[_raw_hook_seed(0, 0), _raw_hook_seed(0, 1), _raw_hook_seed(0, 2)],
        support_materials=[], fallback_spans=[],
    )
    call_count = {"n": 0}

    def _fake_extract(chunk_segments, video_title):
        call_count["n"] += 1
        return fresh_good

    monkeypatch.setattr(clip_selector, "extract_candidates_for_chunk", _fake_extract)
    monkeypatch.setattr(
        clip_selector, "design_final_candidates",
        lambda coverage_targets, materials, fallback_spans, t, title: _stage2_design_result_candidates(
            [_raw_candidate(0, 0, opening_hook_strength=90), _raw_candidate(0, 1, opening_hook_strength=90),
             _raw_candidate(0, 2, opening_hook_strength=90)]
        ),
    )

    result = clip_selector.refresh_stage1_and_candidates(transcript, "タイトル")

    assert len(result) == 3
    assert call_count["n"] == 1  # exactly one chunk for a 1-minute transcript
    # The stale cached material must be gone: Stage1 was fully
    # regenerated, not reused from the existing (now-outdated) cache --
    # and the chunk cache on disk is overwritten with the new result.
    reloaded_chunk = cache.load_stage1_chunk(transcript.video_id, 0)
    assert all(s.soft_score == 80 for s in reloaded_chunk.hook_seeds)
    # The finalized Stage2 result is saved too.
    reloaded_stage2 = cache.load_stage2(transcript.video_id)
    assert len(reloaded_stage2) == 3


def test_refresh_stage1_and_candidates_keeps_earlier_chunk_success_on_later_failure(monkeypatch):
    monkeypatch.setattr(config, "CHUNK_MINUTES", 10.0)
    monkeypatch.setattr(config, "CHUNK_OVERLAP_MINUTES", 1.0)
    transcript = _long_transcript(minutes=25)
    chunks = clip_selector._build_chunks(clip_selector._usable_segments(transcript))
    assert len(chunks) >= 2  # sanity: this test needs at least 2 chunks

    good = Stage1ChunkResult(hook_seeds=[_raw_hook_seed(0, 0)], support_materials=[], fallback_spans=[])
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
    bad_result = Stage1ChunkResult(
        hook_seeds=[_raw_hook_seed(0, 0)], support_materials=[], fallback_spans=[],
    )
    transcript.segments[0].text = "ちょっと表現が難しいんですけども、要するにこうです。"
    transcript.segments[0].words = [
        TranscriptWord(start=0.0, end=2.0, text="ちょっと表現が難しいんですけども、要するにこうです。")
    ]

    monkeypatch.setattr(clip_selector, "extract_candidates_for_chunk", lambda *a, **k: bad_result)
    monkeypatch.setattr(
        clip_selector, "design_final_candidates",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Stage2 must not be called")),
    )

    with pytest.raises(RuntimeError, match="参照整合性・発話品質の基本チェック"):
        clip_selector.refresh_stage1_and_candidates(transcript, "タイトル")


def test_refresh_stage1_and_candidates_overwrites_stage2_cache_with_partial_success(monkeypatch):
    # Revised: 2 valid designs is a real success (NUM_CANDIDATES is a
    # target, not a required minimum), so it DOES overwrite the old
    # Stage2 cache with the new, smaller-but-valid result -- a refresh
    # that finds fewer-but-still-good candidates must not keep serving a
    # stale 3-candidate result instead.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _long_transcript(minutes=1)

    old_stage2 = [_raw_candidate(0, 2, opening_hook_strength=90)] * 3
    cache.save_stage2(transcript.video_id, old_stage2)

    fresh_result = Stage1ChunkResult(
        hook_seeds=[_raw_hook_seed(0, 0), _raw_hook_seed(0, 1), _raw_hook_seed(0, 2)],
        support_materials=[], fallback_spans=[],
    )
    monkeypatch.setattr(clip_selector, "extract_candidates_for_chunk", lambda *a, **k: fresh_result)
    monkeypatch.setattr(
        clip_selector, "design_final_candidates",
        lambda coverage_targets, materials, fallback_spans, t, title: _stage2_design_result_candidates(
            [_raw_candidate(0, 0, opening_hook_strength=90), _raw_candidate(0, 1, opening_hook_strength=90)]
        ),
    )

    result = clip_selector.refresh_stage1_and_candidates(transcript, "タイトル")
    assert len(result) == 2

    reloaded = cache.load_stage2(transcript.video_id)
    assert len(reloaded) == 2


def test_refresh_stage1_and_candidates_does_not_overwrite_stage2_cache_when_nothing_accepted(monkeypatch):
    # The still-valid safety net: when truly ZERO designs survive local
    # validation, the old Stage2 cache must survive completely untouched
    # -- this is the only case _design_finalize_and_cache still raises for.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    transcript = _long_transcript(minutes=1)

    old_stage2 = [_raw_candidate(0, 2, opening_hook_strength=90)] * 3
    cache.save_stage2(transcript.video_id, old_stage2)

    fresh_result = Stage1ChunkResult(
        hook_seeds=[_raw_hook_seed(0, 0), _raw_hook_seed(0, 1)], support_materials=[], fallback_spans=[],
    )
    monkeypatch.setattr(clip_selector, "extract_candidates_for_chunk", lambda *a, **k: fresh_result)
    # Every design Stage2 returns is fabricated (invalid_segment_reference)
    # -- zero survive local validation, so this must still raise before
    # ever calling cache.save_stage2, leaving the old cache exactly as it was.
    monkeypatch.setattr(
        clip_selector, "design_final_candidates",
        lambda coverage_targets, materials, fallback_spans, t, title: _stage2_design_result_candidates(
            [_raw_candidate(9999, 9999, opening_hook_strength=90), _raw_candidate(9998, 9998, opening_hook_strength=90)]
        ),
    )

    with pytest.raises(RuntimeError, match="ローカル検証を通過したものが0件"):
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
