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


def test_stage1_round_trip():
    raw = RawClipCandidate(
        hook_type="open_loop",
        segments=[RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0)],
        hook_text="h", opening_hook_strength=80, title="t", description="d",
        score=70, reasoning="r", caveats="",
    )
    cache.save_stage1("vidA", [{"chunk_index": 0, "candidates": [raw]}])
    loaded = cache.load_stage1("vidA")

    assert loaded is not None
    assert loaded[0]["chunk_index"] == 0
    assert loaded[0]["candidates"][0].hook_type == "open_loop"


def test_stage2_round_trip():
    raw = RawClipCandidate(
        hook_type="strong_take",
        segments=[RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0)],
        hook_text="h", opening_hook_strength=80, title="t", description="d",
        score=90, reasoning="r", caveats="注意",
    )
    cache.save_stage2("vidA", [raw, raw, raw])
    loaded = cache.load_stage2("vidA")

    assert loaded is not None
    assert len(loaded) == 3
    assert loaded[0].caveats == "注意"


def test_stage2_with_stale_schema_version_is_treated_as_cache_miss():
    """A cache written by an older clip_selector.py schema/prompt version
    (e.g. one that still had cta_end_text instead of opening_hook_strength)
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


def test_stage1_with_unversioned_legacy_shape_is_treated_as_cache_miss():
    """Even older caches (from before schema_version wrapping existed at
    all) were a bare list, not {"schema_version": ..., "chunks": [...]}.
    """
    cache.stage1_path("vidC").write_text(
        json.dumps([{"chunk_index": 0, "candidates": []}]), encoding="utf-8"
    )
    assert cache.load_stage1("vidC") is None


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


# --- diagnostics for the 2026-08-30 "string indices must be integers, not
# --- 'str'" incident (see clip_selector.py's matching tests). A cache file
# --- that DOES report the current schema_version but contains a malformed
# --- (non-dict) candidate entry -- e.g. hand-edited or corrupted on disk --
# --- must raise a diagnosable error naming the cache file/index, not a bare
# --- TypeError. No repair/retry behavior is added -- diagnostics only.


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


def test_load_stage1_raises_diagnosable_error_for_malformed_cached_segment():
    payload = {
        "schema_version": config.CANDIDATE_SCHEMA_VERSION,
        "chunks": [
            {
                "chunk_index": 0,
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
        ],
    }
    cache.stage1_path("vidE").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(MalformedCandidateError) as exc_info:
        cache.load_stage1("vidE")
    message = str(exc_info.value)
    assert "stage1" in message
    assert "vidE" in message
    assert "candidates[0].segments[0]" in message
    assert "str" in message
