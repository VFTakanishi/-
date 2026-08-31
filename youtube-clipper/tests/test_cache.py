import json

import pytest

from podcast_clipper import cache, config
from podcast_clipper.models import (
    MalformedCandidateError,
    RawClipCandidate,
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
    raw = _raw_candidate(hook_type="open_loop")
    cache.save_stage1_chunk("vidA", 0, [raw])
    loaded = cache.load_stage1_chunk("vidA", 0)

    assert loaded is not None
    assert loaded[0].hook_type == "open_loop"


def test_stage1_missing_chunk_returns_none():
    assert cache.load_stage1_chunk("vid-never-cached", 0) is None


def test_stage1_chunk_partial_failure_keeps_earlier_successful_chunks():
    """chunk0 succeeds and is cached, chunk1 succeeds and is cached, chunk2
    fails (never saved) -- chunk0/1's already-paid-for results must survive
    on disk, and the still-missing chunk2 must read back as a plain miss,
    not raise or corrupt the file.
    """
    cache.save_stage1_chunk("vidChunks", 0, [_raw_candidate(hook_type="open_loop")])
    cache.save_stage1_chunk("vidChunks", 1, [_raw_candidate(hook_type="strong_take")])
    # chunk 2's API call "failed" -- save_stage1_chunk is simply never called for it.

    assert cache.load_stage1_chunk("vidChunks", 0)[0].hook_type == "open_loop"
    assert cache.load_stage1_chunk("vidChunks", 1)[0].hook_type == "strong_take"
    assert cache.load_stage1_chunk("vidChunks", 2) is None


def test_stage1_chunk_save_does_not_overwrite_other_chunks():
    cache.save_stage1_chunk("vidG", 0, [_raw_candidate(hook_type="open_loop")])
    cache.save_stage1_chunk("vidG", 1, [_raw_candidate(hook_type="strong_take")])
    # re-saving chunk 0 (e.g. force_refresh) must not disturb chunk 1
    cache.save_stage1_chunk("vidG", 0, [_raw_candidate(hook_type="surprising_fact")])

    assert cache.load_stage1_chunk("vidG", 0)[0].hook_type == "surprising_fact"
    assert cache.load_stage1_chunk("vidG", 1)[0].hook_type == "strong_take"


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
    assert config.CANDIDATE_SCHEMA_VERSION == 7

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

    cache.save_stage1_chunk("vidH", 0, [raw])
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
    assert config.CANDIDATE_SCHEMA_VERSION == 7

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
    assert config.CANDIDATE_SCHEMA_VERSION == 7

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
    cache.save_stage1_chunk("vidV5", 0, [raw])
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
    assert config.CANDIDATE_SCHEMA_VERSION == 7

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
    cache.save_stage1_chunk("vidV6", 0, [raw])
    assert cache.load_stage1_chunk("vidV6", 0) is not None
    cache.save_stage2("vidV6", [raw, raw, raw])
    assert cache.load_stage2("vidV6") is not None


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
                "candidates": [
                    {
                        "hook_type": "story",
                        "segments": ["not a segment dict"],
                        "hook_text": "h", "opening_hook_strength": 80,
                        "title": "t", "description": "d", "score": 80,
                        "reasoning": "r", "caveats": "",
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
    assert "candidates[0].segments[0]" in message
    assert "str" in message
