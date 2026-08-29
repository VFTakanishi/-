"""Disk cache for the expensive pipeline stages, keyed by YouTube video id.

Absolute condition #12 requires the transcript, Stage1 candidates, and
Stage2 result to be cached so re-running the pipeline (including recovery
from an interrupted job, see jobs.py) avoids redundant YouTube downloads,
Whisper runs, and Claude API calls.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from . import config
from .models import (
    RawClipCandidate,
    RawUsedSegment,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)


def _atomic_write_text(path: Path, content: str) -> None:
    """Write via temp file + os.replace so a concurrent reader never sees a
    truncated/partial cache file (same rationale as jobs._atomic_write_text).
    """
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _cache_dir(video_id: str) -> Path:
    d = config.OUTPUT_DIR / video_id / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def transcript_path(video_id: str) -> Path:
    return _cache_dir(video_id) / "transcript.json"


def stage1_path(video_id: str) -> Path:
    return _cache_dir(video_id) / "stage1_candidates.json"


def stage2_path(video_id: str) -> Path:
    return _cache_dir(video_id) / "stage2_result.json"


# --- Transcript ---------------------------------------------------------


def save_transcript(transcript: Transcript) -> None:
    _atomic_write_text(
        transcript_path(transcript.video_id),
        json.dumps(asdict(transcript), ensure_ascii=False, indent=2),
    )


def load_transcript(video_id: str) -> Transcript | None:
    path = transcript_path(video_id)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    segments = [
        TranscriptSegment(
            id=seg["id"],
            start=seg["start"],
            end=seg["end"],
            text=seg["text"],
            words=[TranscriptWord(**w) for w in seg["words"]],
        )
        for seg in raw["segments"]
    ]
    return Transcript(video_id=raw["video_id"], language=raw["language"], segments=segments)


# --- Stage1 (per-chunk raw candidates) -----------------------------------
# Each chunk's extraction result is stored as
# {"chunk_index": int, "candidates": [RawClipCandidate, ...]}


def save_stage1(video_id: str, chunk_results: list[dict]) -> None:
    payload = {
        "schema_version": config.CANDIDATE_SCHEMA_VERSION,
        "chunks": [
            {
                "chunk_index": chunk["chunk_index"],
                "candidates": [asdict(c) for c in chunk["candidates"]],
            }
            for chunk in chunk_results
        ],
    }
    _atomic_write_text(
        stage1_path(video_id), json.dumps(payload, ensure_ascii=False, indent=2)
    )


def load_stage1(video_id: str) -> list[dict] | None:
    path = stage1_path(video_id)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != config.CANDIDATE_SCHEMA_VERSION:
        # Stale (pre-versioning) or schema-incompatible cache from before a
        # clip_selector.py prompt/schema change: treat as a miss so the
        # caller recomputes Stage1 fresh, rather than trying to deserialize
        # old-shape data.
        return None
    return [
        {
            "chunk_index": chunk["chunk_index"],
            "candidates": [_raw_candidate_from_dict(c) for c in chunk["candidates"]],
        }
        for chunk in raw["chunks"]
    ]


# --- Stage2 (final 3 candidates) -----------------------------------------


def save_stage2(video_id: str, candidates: list[RawClipCandidate]) -> None:
    payload = {
        "schema_version": config.CANDIDATE_SCHEMA_VERSION,
        "candidates": [asdict(c) for c in candidates],
    }
    _atomic_write_text(
        stage2_path(video_id), json.dumps(payload, ensure_ascii=False, indent=2)
    )


def load_stage2(video_id: str) -> list[RawClipCandidate] | None:
    path = stage2_path(video_id)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != config.CANDIDATE_SCHEMA_VERSION:
        return None
    return [_raw_candidate_from_dict(c) for c in raw["candidates"]]


def _raw_candidate_from_dict(d: dict) -> RawClipCandidate:
    return RawClipCandidate(
        hook_type=d["hook_type"],
        segments=[RawUsedSegment(**s) for s in d["segments"]],
        hook_text=d["hook_text"],
        opening_hook_strength=d["opening_hook_strength"],
        title=d["title"],
        description=d["description"],
        score=d["score"],
        reasoning=d["reasoning"],
        caveats=d["caveats"],
    )
