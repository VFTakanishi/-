import json

import pytest

from podcast_clipper import cache, config
from podcast_clipper.models import (
    MalformedCandidateError,
    RawClipCandidate,
    RawMaterial,
    RawMaterialSegment,
    RawUsedSegment,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)


def _transcript():
    return Transcript(
        video_id="vidA",
        language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=2.0, text="hello",
                words=[TranscriptWord(start=0.0, end=1.0, text="he"), TranscriptWord(start=1.0, end=2.0, text="llo")],
            )
        ],
    )


def _raw_candidate(hook_type="open_loop", caveats=""):
    return RawClipCandidate(
        hook_type=hook_type,
        segments=[RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0)],
        hook_text="h", opening_hook_strength=80, title="", description="",
        score=70, reasoning="", caveats=caveats,
    )


def _raw_material(material_type="hook", usefulness_score=80):
    return RawMaterial(
        material_type=material_type,
        segments=[RawMaterialSegment(start_segment_id=0, end_segment_id=0)],
        usefulness_score=usefulness_score,
    )


def test_transcript_round_trip():
    original = _transcript()
    cache.save_transcript(original)
    loaded = cache.load_transcript("vidA")

    assert loaded is not None
    assert loaded.video_id == original.video_id
    assert loaded.segments[0].text == "hello"
    assert loaded.segments[0].words[1].text == "llo"


def test_transcript_missing_returns_none():
    assert cache.load_transcript("does-not-exist") is None


# --- Stage1: per-chunk incremental cache ---------------------------------


def test_stage1_chunk_round_trip():
    material = _raw_material(material_type="hook")
    cache.save_stage1_chunk("vidA", 0, [material])
    loaded = cache.load_stage1_chunk("vidA", 0)

    assert loaded is not None
    assert loaded[0].material_type == "hook"


def test_stage1_missing_chunk_returns_none():
    assert cache.load_stage1_chunk("vid-never-cached", 0) is None


def test_stage1_chunk_partial_failure_keeps_earlier_successful_chunks():
    """chunk0 succeeds and is cached, chunk1 succeeds and is cached, chunk2
    fails (never saved) -- chunk0/1's already-paid-for results must survive
    on disk, and the still-missing chunk2 must read back as a plain miss,
    not raise or corrupt the file.
    """
    cache.save_stage1_chunk("vidChunks", 0, [_raw_material(material_type="hook")])
    cache.save_stage1_chunk("vidChunks", 1, [_raw_material(material_type="reason")])
    # chunk 2's API call "failed" -- save_stage1_chunk is simply never called for it.

    assert cache.load_stage1_chunk("vidChunks", 0)[0].material_type == "hook"
    assert cache.load_stage1_chunk("vidChunks", 1)[0].material_type == "reason"
    assert cache.load_stage1_chunk("vidChunks", 2) is None


def test_stage1_chunk_save_does_not_overwrite_other_chunks():
    cache.save_stage1_chunk("vidG", 0, [_raw_material(material_type="hook")])
    cache.save_stage1_chunk("vidG", 1, [_raw_material(material_type="reason")])
    # re-saving chunk 0 (e.g. force_refresh) must not disturb chunk 1
    cache.save_stage1_chunk("vidG", 0, [_raw_material(material_type="example")])

    assert cache.load_stage1_chunk("vidG", 0)[0].material_type == "example"
    assert cache.load_stage1_chunk("vidG", 1)[0].material_type == "reason"


def test_stage1_with_unversioned_legacy_shape_is_treated_as_cache_miss():
    """Even older caches (from before schema_version wrapping existed at
    all) were a bare list, not {"schema_version": ..., "chunks": {...}}.
    """
    cache.stage1_path("vidC").write_text(
        json.dumps([{"chunk_index": 0, "candidates": []}]), encoding="utf-8"
    )
    assert cache.load_stage1_chunk("vidC", 0) is None


# --- Stage2: whole-file cache (unchanged shape; only ranking runs once) --


def test_stage2_round_trip():
    raw = RawClipCandidate(
        hook_type="strong_take",
        segments=[RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0)],
        hook_text="h", opening_hook_strength=80, title="", description="",
        score=90, reasoning="", caveats="注意",
    )
    cache.save_stage2("vidA", [raw, raw, raw])
    loaded = cache.load_stage2("vidA")

    assert loaded is not None
    assert len(loaded) == 3
    assert loaded[0].caveats == "注意"


def test_stage2_round_trip_with_one_candidate():
    # G: NUM_CANDIDATES is a target/ceiling, not a required minimum -- a
    # successful run may legitimately produce just 1 candidate, and the
    # cache must store/load that as-is, never assuming exactly 3.
    raw = _raw_candidate(hook_type="strong_take")
    cache.save_stage2("vidOne", [raw])
    loaded = cache.load_stage2("vidOne")

    assert loaded is not None
    assert len(loaded) == 1


def test_stage2_round_trip_with_two_candidates():
    # H: same for 2 candidates.
    raw = _raw_candidate(hook_type="strong_take")
    cache.save_stage2("vidTwo", [raw, raw])
    loaded = cache.load_stage2("vidTwo")

    assert loaded is not None
    assert len(loaded) == 2


# --- Stage2 diagnostic: raw Stage2 output survives a failed run ---------
# (separate from stage2_result.json, which only ever holds a fully
# successful run's finalized, accepted candidates)


def test_stage2_diagnostic_round_trip():
    raw = _raw_candidate(hook_type="strong_take")
    cache.save_stage2_diagnostic("vidDiag", [raw, raw])
    loaded = cache.load_stage2_diagnostic("vidDiag")

    assert loaded is not None
    assert loaded["schema_version"] == config.CANDIDATE_SCHEMA_VERSION
    assert len(loaded["candidates"]) == 2
    assert "evaluations" not in loaded


def test_stage2_diagnostic_with_evaluations_round_trip():
    raw = _raw_candidate(hook_type="strong_take")
    evaluations = [
        {"accepted": True, "reason": "accepted", "duration_sec": 30.0, "opening_text": "h"},
        {"accepted": False, "reason": "hook_strength_below_80", "duration_sec": 25.0, "opening_text": "w"},
    ]
    cache.save_stage2_diagnostic("vidDiagEval", [raw, raw], evaluations=evaluations)
    loaded = cache.load_stage2_diagnostic("vidDiagEval")

    assert loaded["evaluations"] == evaluations


def test_stage2_diagnostic_missing_returns_none():
    assert cache.load_stage2_diagnostic("vid-never-cached-diagnostic") is None


def test_load_stage2_never_reads_diagnostic_file():
    raw = _raw_candidate(hook_type="strong_take")
    cache.save_stage2_diagnostic("vidDiagOnly", [raw])
    # Only the diagnostic file was written -- the production stage2_result
    # cache must still read back as a plain miss.
    assert cache.load_stage2("vidDiagOnly") is None


def test_stage2_with_stale_schema_version_is_treated_as_cache_miss():
    """A cache written by an older clip_selector.py schema/prompt version
    must never be deserialized against the new RawClipCandidate shape --
    that would raise KeyError deep inside select_candidates. It must be
    treated as a plain cache miss instead, so the caller recomputes fresh.
    """
    stale_payload = {
        "schema_version": config.CANDIDATE_SCHEMA_VERSION - 1,
        "candidates": [
            {
                "hook_type": "story",
                "segments": [{"role": "hook", "start_segment_id": 0, "end_segment_id": 0}],
                "hook_text": "h",
                "cta_end_text": "old field, no longer valid",
                "title": "t", "description": "d", "score": 80,
                "reasoning": "r", "caveats": "",
            }
        ],
    }
    cache.stage2_path("vidB").write_text(
        json.dumps(stale_payload, ensure_ascii=False), encoding="utf-8"
    )
    assert cache.load_stage2("vidB") is None


def test_schema_v3_is_miss_and_current_version_is_hit():
    """The tool_use/messages.parse() -> minimal-schema Structured Outputs
    migration bumped CANDIDATE_SCHEMA_VERSION from 3 to 4 (Stage1's cache
    file shape itself changed: per-chunk dict instead of a chunk list). A
    cache written under version 3 must be treated as a miss on both
    stage1 and stage2, while a cache written under the current schema
    version must hit normally.
    """
    assert config.CANDIDATE_SCHEMA_VERSION == 12

    v3_stage2_payload = {
        "schema_version": 3,
        "candidates": [
            {
                "hook_type": "story",
                "segments": [{"role": "hook", "start_segment_id": 0, "end_segment_id": 0}],
                "hook_text": "h", "opening_hook_strength": 80,
                "title": "t", "description": "d", "score": 80,
                "reasoning": "r", "caveats": "",
            }
        ],
    }
    cache.stage2_path("vidF").write_text(
        json.dumps(v3_stage2_payload, ensure_ascii=False), encoding="utf-8"
    )
    assert cache.load_stage2("vidF") is None

    v3_stage1_payload = {
        "schema_version": 3,
        "chunks": [{"chunk_index": 0, "candidates": []}],
    }
    cache.stage1_path("vidH").write_text(
        json.dumps(v3_stage1_payload, ensure_ascii=False), encoding="utf-8"
    )
    assert cache.load_stage1_chunk("vidH", 0) is None

    raw = _raw_candidate(hook_type="story")
    cache.save_stage2("vidF", [raw, raw, raw])
    loaded = cache.load_stage2("vidF")
    assert loaded is not None
    assert len(loaded) == 3

    cache.save_stage1_chunk("vidH", 0, [_raw_material(material_type="hook")])
    assert cache.load_stage1_chunk("vidH", 0) is not None


def test_schema_v4_is_miss_after_hook_scoring_prompt_bump():
    """The Stage1/Stage2 hook-scoring rubric was strengthened (stricter
    opening_hook_strength bands, MIN_OPENING_HOOK_STRENGTH raised 60->80,
    Stage2 independently re-evaluates the hook) without changing the
    candidate JSON shape, so CANDIDATE_SCHEMA_VERSION was bumped 4->5 purely
    to invalidate scores computed under the old, looser rubric. A cache
    written under version 4 must still be treated as a miss under the
    current (later-bumped) schema version too.
    """
    assert config.CANDIDATE_SCHEMA_VERSION == 12

    v4_stage2_payload = {
        "schema_version": 4,
        "candidates": [
            {
                "hook_type": "story",
                "segments": [{"role": "hook", "start_segment_id": 0, "end_segment_id": 0}],
                "hook_text": "h", "opening_hook_strength": 78,
                "title": "t", "description": "d", "score": 80,
                "reasoning": "r", "caveats": "",
            }
        ],
    }
    cache.stage2_path("vidV4").write_text(
        json.dumps(v4_stage2_payload, ensure_ascii=False), encoding="utf-8"
    )
    assert cache.load_stage2("vidV4") is None

    v4_stage1_payload = {
        "schema_version": 4,
        "chunks": [{"chunk_index": 0, "candidates": []}],
    }
    cache.stage1_path("vidV4b").write_text(
        json.dumps(v4_stage1_payload, ensure_ascii=False), encoding="utf-8"
    )
    assert cache.load_stage1_chunk("vidV4b", 0) is None

    raw = _raw_candidate(hook_type="story")
    cache.save_stage2("vidV4", [raw, raw, raw])
    assert cache.load_stage2("vidV4") is not None


def test_schema_v5_is_miss_after_stage1_recall_widening():
    """Stage1's per-chunk candidate cap was widened 3 -> 6
    (STAGE1_MAX_CANDIDATES_PER_CHUNK) to give Stage1 more search breadth
    under the stricter MIN_OPENING_HOOK_STRENGTH=80 filter, without
    changing the per-candidate JSON shape. CANDIDATE_SCHEMA_VERSION was
    bumped 5->6 purely to invalidate Stage1 caches written when the model
    was still constrained to a 3-candidate ceiling per chunk (they may be
    missing viable candidates the wider search would have found). A cache
    written under version 5 must be treated as a miss.
    """
    assert config.CANDIDATE_SCHEMA_VERSION == 12

    v5_stage1_payload = {
        "schema_version": 5,
        "chunks": [{"chunk_index": 0, "candidates": []}],
    }
    cache.stage1_path("vidV5").write_text(
        json.dumps(v5_stage1_payload, ensure_ascii=False), encoding="utf-8"
    )
    assert cache.load_stage1_chunk("vidV5", 0) is None

    v5_stage2_payload = {
        "schema_version": 5,
        "candidates": [
            {
                "hook_type": "story",
                "segments": [{"role": "hook", "start_segment_id": 0, "end_segment_id": 0}],
                "hook_text": "h", "opening_hook_strength": 85,
                "title": "t", "description": "d", "score": 85,
                "reasoning": "r", "caveats": "",
            }
        ],
    }
    cache.stage2_path("vidV5").write_text(
        json.dumps(v5_stage2_payload, ensure_ascii=False), encoding="utf-8"
    )
    assert cache.load_stage2("vidV5") is None

    raw = _raw_candidate(hook_type="story")
    cache.save_stage1_chunk("vidV5", 0, [_raw_material(material_type="hook")])
    assert cache.load_stage1_chunk("vidV5", 0) is not None
    cache.save_stage2("vidV5", [raw, raw, raw])
    assert cache.load_stage2("vidV5") is not None


def test_schema_v6_is_miss_after_anchor_trim_and_reorder_support():
    """RawUsedSegment/Stage1SegmentOutput gained an optional
    start_anchor_text field (lets a candidate start mid-segment at a real
    word boundary instead of only ever using the segment's literal first
    word), and segment order within a candidate is no longer required to
    be transcript-chronological (a stronger later utterance can be placed
    first as the hook). Both change what a cached candidate *means*
    without changing the outer JSON shape enough to fail plain
    deserialization on its own, so CANDIDATE_SCHEMA_VERSION was bumped
    6->7 to force old (pre-anchor, chronological-only) Stage1/Stage2
    caches to be recomputed. A cache written under version 6 must be
    treated as a miss.
    """
    assert config.CANDIDATE_SCHEMA_VERSION == 12

    v6_stage1_payload = {
        "schema_version": 6,
        "chunks": [{"chunk_index": 0, "candidates": []}],
    }
    cache.stage1_path("vidV6").write_text(
        json.dumps(v6_stage1_payload, ensure_ascii=False), encoding="utf-8"
    )
    assert cache.load_stage1_chunk("vidV6", 0) is None

    v6_stage2_payload = {
        "schema_version": 6,
        "candidates": [
            {
                "hook_type": "story",
                "segments": [{"role": "hook", "start_segment_id": 0, "end_segment_id": 0}],
                "hook_text": "h", "opening_hook_strength": 85,
                "title": "t", "description": "d", "score": 85,
                "reasoning": "r", "caveats": "",
            }
        ],
    }
    cache.stage2_path("vidV6").write_text(
        json.dumps(v6_stage2_payload, ensure_ascii=False), encoding="utf-8"
    )
    assert cache.load_stage2("vidV6") is None

    raw = _raw_candidate(hook_type="story")
    cache.save_stage1_chunk("vidV6", 0, [_raw_material(material_type="hook")])
    assert cache.load_stage1_chunk("vidV6", 0) is not None
    cache.save_stage2("vidV6", [raw, raw, raw])
    assert cache.load_stage2("vidV6") is not None


def test_schema_v7_is_miss_after_junction_safety_support():
    """Real-machine feedback showed a candidate could pass every existing
    check (each segment fine on its own, hook fine, final ending fine)
    yet still cut together into an unnatural mid-clip junction (e.g.
    "車を冷やしますっていうのであれば" hard-cut into an unrelated "連続周回
    をする場合は"), or open on a dangling reference ("これのクラッチ交換の
    際に..."). Junction validation (_validate_candidate_junctions,
    _extend_internal_junctions, the limited hook/payoff exact-repeat
    overlap allowance) closes this without changing the Stage1 output
    shape, so CANDIDATE_SCHEMA_VERSION was bumped 7->8 purely to force
    caches built under the old, junction-unaware rubric to be
    recomputed. A cache written under version 7 must be treated as a
    miss.
    """
    assert config.CANDIDATE_SCHEMA_VERSION == 12

    v7_stage1_payload = {
        "schema_version": 7,
        "chunks": [{"chunk_index": 0, "candidates": []}],
    }
    cache.stage1_path("vidV7").write_text(
        json.dumps(v7_stage1_payload, ensure_ascii=False), encoding="utf-8"
    )
    assert cache.load_stage1_chunk("vidV7", 0) is None

    v7_stage2_payload = {
        "schema_version": 7,
        "candidates": [
            {
                "hook_type": "story",
                "segments": [{"role": "hook", "start_segment_id": 0, "end_segment_id": 0}],
                "hook_text": "h", "opening_hook_strength": 85,
                "title": "t", "description": "d", "score": 85,
                "reasoning": "r", "caveats": "",
            }
        ],
    }
    cache.stage2_path("vidV7").write_text(
        json.dumps(v7_stage2_payload, ensure_ascii=False), encoding="utf-8"
    )
    assert cache.load_stage2("vidV7") is None

    raw = _raw_candidate(hook_type="story")
    cache.save_stage1_chunk("vidV7", 0, [_raw_material(material_type="hook")])
    assert cache.load_stage1_chunk("vidV7", 0) is not None
    cache.save_stage2("vidV7", [raw, raw, raw])
    assert cache.load_stage2("vidV7") is not None


def test_schema_v8_is_miss_after_restart_and_closure_support():
    """Real-machine feedback showed two further gaps: (1) a candidate could
    contain a speech restart -- an abandoned, unfinished clause immediately
    restarted with the same content phrase in a different construction
    (e.g. "Nレンジで下るというのは、Nレンジにすると...") -- that the existing
    marker/connective-based speech_disfluency check could not catch, and
    (2) a candidate could rank highly in Stage2 despite its body never
    resolving the question/claim its hook posed (e.g. a fuel-economy
    comparison hook followed only by a safety aside and a "気がする"-style
    guess). find_speech_restart_marker/_candidate_speech_restart_marker adds
    a local veto for (1), and the rank_and_finalize.md prompt now excludes
    an id from ranked_candidate_ids entirely for (2) -- neither changes the
    Stage1 output shape, so CANDIDATE_SCHEMA_VERSION was bumped 8->9 purely
    to force caches built under the old, restart/closure-unaware rubric to
    be recomputed. A cache written under version 8 must be treated as a
    miss.
    """
    assert config.CANDIDATE_SCHEMA_VERSION == 12

    v8_stage1_payload = {
        "schema_version": 8,
        "chunks": [{"chunk_index": 0, "candidates": []}],
    }
    cache.stage1_path("vidV8").write_text(
        json.dumps(v8_stage1_payload, ensure_ascii=False), encoding="utf-8"
    )
    assert cache.load_stage1_chunk("vidV8", 0) is None

    v8_stage2_payload = {
        "schema_version": 8,
        "candidates": [
            {
                "hook_type": "story",
                "segments": [{"role": "hook", "start_segment_id": 0, "end_segment_id": 0}],
                "hook_text": "h", "opening_hook_strength": 85,
                "title": "t", "description": "d", "score": 85,
                "reasoning": "r", "caveats": "",
            }
        ],
    }
    cache.stage2_path("vidV8").write_text(
        json.dumps(v8_stage2_payload, ensure_ascii=False), encoding="utf-8"
    )
    assert cache.load_stage2("vidV8") is None


def test_schema_v9_is_miss_after_stage2_final_design_support():
    """Real-machine feedback showed that Stage1 alone can't reliably
    assemble a complete Shorts construct (hook + reason + example) within
    one chunk's single API call, and Stage2's old ranking-only role
    (Stage2RankingOutput -- a bare id list) had no way to fix that: it
    could only pick from or exclude Stage1's own already-complete
    candidates, never recombine segments across them. Stage2 was
    redesigned into a final-edit-design role (Stage2Output/
    Stage2CandidateOutput/Stage2SegmentOutput -- the same shape as
    Stage1's own schema, plus end_anchor_text) that can freely recombine
    real segments from any Stage1 material into a new final candidate.
    This is a genuine Structured Outputs schema change on the Stage2 side
    (Claude's output contract itself is different), so CANDIDATE_SCHEMA_
    VERSION was bumped 9->10. A cache written under version 9 must be
    treated as a miss -- it holds candidates Stage2 only ever ranked/
    excluded, never validated against the new final-design local gate.
    """
    assert config.CANDIDATE_SCHEMA_VERSION == 12

    v9_stage1_payload = {
        "schema_version": 9,
        "chunks": [{"chunk_index": 0, "candidates": []}],
    }
    cache.stage1_path("vidV9").write_text(
        json.dumps(v9_stage1_payload, ensure_ascii=False), encoding="utf-8"
    )
    assert cache.load_stage1_chunk("vidV9", 0) is None

    v9_stage2_payload = {
        "schema_version": 9,
        "candidates": [
            {
                "hook_type": "story",
                "segments": [{"role": "hook", "start_segment_id": 0, "end_segment_id": 0}],
                "hook_text": "h", "opening_hook_strength": 85,
                "title": "t", "description": "d", "score": 85,
                "reasoning": "r", "caveats": "",
            }
        ],
    }
    cache.stage2_path("vidV9").write_text(
        json.dumps(v9_stage2_payload, ensure_ascii=False), encoding="utf-8"
    )
    assert cache.load_stage2("vidV9") is None

    raw = _raw_candidate(hook_type="story")
    cache.save_stage1_chunk("vidV9", 0, [_raw_material(material_type="hook")])
    assert cache.load_stage1_chunk("vidV9", 0) is not None
    cache.save_stage2("vidV9", [raw, raw, raw])
    assert cache.load_stage2("vidV9") is not None

    raw = _raw_candidate(hook_type="story")
    cache.save_stage1_chunk("vidV8", 0, [_raw_material(material_type="hook")])
    assert cache.load_stage1_chunk("vidV8", 0) is not None
    cache.save_stage2("vidV8", [raw, raw, raw])
    assert cache.load_stage2("vidV8") is not None


def test_schema_v10_is_miss_after_stage1_material_contract_support():
    """The Stage1/Stage2 material-contract fix: Stage1's Structured Outputs
    contract changed from candidate-shaped (Stage1CandidateOutput --
    hook_type/opening_hook_strength/score required on every item, forcing
    even a non-hook-shaped reason/example material to carry hook-strength
    properties it was never meant to be judged by) to a genuine material
    contract (Stage1MaterialOutput -- material_type/segments/
    usefulness_score). The Stage1 chunk cache's own JSON shape changed
    (`materials` key instead of `candidates`, no `role` per segment), so
    CANDIDATE_SCHEMA_VERSION was bumped 10->11. Stage2's schema/prompt/
    cache shape did not change this round, but the shared single-version
    scheme (established since v3->v4) invalidates both stages together, so
    a v10 cache must be a miss on both Stage1 and Stage2.
    """
    assert config.CANDIDATE_SCHEMA_VERSION == 12

    v10_stage1_payload = {
        "schema_version": 10,
        "chunks": {"0": {"candidates": []}},
    }
    cache.stage1_path("vidV10").write_text(
        json.dumps(v10_stage1_payload, ensure_ascii=False), encoding="utf-8"
    )
    assert cache.load_stage1_chunk("vidV10", 0) is None

    v10_stage2_payload = {
        "schema_version": 10,
        "candidates": [
            {
                "hook_type": "story",
                "segments": [{"role": "hook", "start_segment_id": 0, "end_segment_id": 0}],
                "hook_text": "h", "opening_hook_strength": 85,
                "title": "t", "description": "d", "score": 85,
                "reasoning": "r", "caveats": "",
            }
        ],
    }
    cache.stage2_path("vidV10").write_text(
        json.dumps(v10_stage2_payload, ensure_ascii=False), encoding="utf-8"
    )
    assert cache.load_stage2("vidV10") is None

    material = _raw_material(material_type="hook")
    cache.save_stage1_chunk("vidV10", 0, [material])
    assert cache.load_stage1_chunk("vidV10", 0) is not None

    raw = _raw_candidate(hook_type="story")
    cache.save_stage2("vidV10", [raw, raw, raw])
    assert cache.load_stage2("vidV10") is not None


def test_schema_v11_is_miss_after_stage2_overproduction_support():
    """Real-machine incident: Stage2 designed only 2 final candidates when
    NUM_CANDIDATES=3 were required, despite Stage1 having supplied plenty
    of still-unused material -- both designs even passed local validation,
    so the failure was purely "too few designs attempted," not over-strict
    validation. Stage2Output.candidates' schema ceiling was decoupled from
    NUM_CANDIDATES(3) to the new STAGE2_MAX_DESIGNS(6), and rank_and_
    finalize.md now actively encourages designing as many independently
    valid candidates as the material supports (never lowering the
    semantic-closure/junction/duration bar to hit a count). This is a
    genuine Structured Outputs schema change on the Stage2 side (Claude's
    output contract allows more items now) plus a prompt change, so
    CANDIDATE_SCHEMA_VERSION was bumped 11->12. A cache written under
    version 11 must be treated as a miss on both stage1 and stage2.
    """
    assert config.CANDIDATE_SCHEMA_VERSION == 12

    v11_stage1_payload = {
        "schema_version": 11,
        "chunks": {"0": {"materials": []}},
    }
    cache.stage1_path("vidV11").write_text(
        json.dumps(v11_stage1_payload, ensure_ascii=False), encoding="utf-8"
    )
    assert cache.load_stage1_chunk("vidV11", 0) is None

    v11_stage2_payload = {
        "schema_version": 11,
        "candidates": [
            {
                "hook_type": "story",
                "segments": [{"role": "hook", "start_segment_id": 0, "end_segment_id": 0}],
                "hook_text": "h", "opening_hook_strength": 85,
                "title": "t", "description": "d", "score": 85,
                "reasoning": "r", "caveats": "",
            }
        ],
    }
    cache.stage2_path("vidV11").write_text(
        json.dumps(v11_stage2_payload, ensure_ascii=False), encoding="utf-8"
    )
    assert cache.load_stage2("vidV11") is None

    material = _raw_material(material_type="hook")
    cache.save_stage1_chunk("vidV11", 0, [material])
    assert cache.load_stage1_chunk("vidV11", 0) is not None

    raw = _raw_candidate(hook_type="story")
    cache.save_stage2("vidV11", [raw, raw, raw])
    assert cache.load_stage2("vidV11") is not None


def test_transcript_cache_is_unaffected_by_candidate_schema_versioning():
    """Whisper transcription is unrelated to the Stage1/Stage2 prompt/schema
    -- the transcript cache format itself is never versioned, and stays
    reusable across candidate-schema changes.
    """
    original = _transcript()
    cache.save_transcript(original)
    raw = json.loads(cache.transcript_path("vidA").read_text(encoding="utf-8"))
    assert "schema_version" not in raw
    assert cache.load_transcript("vidA") is not None


# --- diagnostics for malformed cached candidates (retained from earlier
# --- incidents). A cache file that DOES report the current schema_version
# --- but contains a malformed (non-dict) candidate entry -- e.g.
# --- hand-edited or corrupted on disk -- must raise a diagnosable error
# --- naming the cache file/index, not a bare TypeError. No repair/retry
# --- behavior is added -- diagnostics only.


def test_load_stage2_raises_diagnosable_error_for_malformed_cached_candidate():
    payload = {
        "schema_version": config.CANDIDATE_SCHEMA_VERSION,
        "candidates": ["not a candidate dict"],
    }
    cache.stage2_path("vidD").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(MalformedCandidateError) as exc_info:
        cache.load_stage2("vidD")
    message = str(exc_info.value)
    assert "stage2" in message
    assert "vidD" in message
    assert "candidates[0]" in message
    assert "str" in message


def test_load_stage1_chunk_raises_diagnosable_error_for_malformed_cached_segment():
    payload = {
        "schema_version": config.CANDIDATE_SCHEMA_VERSION,
        "chunks": {
            "0": {
                "materials": [
                    {
                        "material_type": "hook",
                        "segments": ["not a segment dict"],
                        "usefulness_score": 80,
                    }
                ],
            }
        },
    }
    cache.stage1_path("vidE").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(MalformedCandidateError) as exc_info:
        cache.load_stage1_chunk("vidE", 0)
    message = str(exc_info.value)
    assert "stage1" in message
    assert "vidE" in message
    assert "materials[0].segments[0]" in message
    assert "str" in message
