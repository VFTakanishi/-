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
    RawFallbackSpan,
    RawHookSeed,
    RawMaterial,
    RawMaterialSegment,
    RawUsedSegment,
    Stage1ChunkResult,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
    require_dict,
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


def stage2_diagnostic_path(video_id: str) -> Path:
    return _cache_dir(video_id) / "stage2_diagnostic.json"


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


# --- Stage1 (per-chunk discovery output, cached incrementally) ----------
# File shape: {"schema_version": int, "chunks": {"<chunk_index>":
# {"hook_seeds": [...], "support_materials": [...], "fallback_spans": [...]}}}
# (hook-seed-discovery redesign -- previously a single "materials" list).
# Each chunk is written the moment it succeeds (save_stage1_chunk), via a
# read-modify-write of the whole file -- still one atomic write per call,
# so a crash mid-write never corrupts previously-saved chunks, it just
# leaves the prior version in place. This means a run that fails partway
# through a long video keeps every chunk result that already cost an API
# call, and a retry only re-requests the chunks that are still missing.


def _load_stage1_payload(video_id: str) -> dict | None:
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
    return raw


def load_stage1_chunk(video_id: str, chunk_index: int) -> Stage1ChunkResult | None:
    payload = _load_stage1_payload(video_id)
    if payload is None:
        return None
    chunk = payload.get("chunks", {}).get(str(chunk_index))
    if chunk is None:
        return None
    context = f"cache stage1 video_id={video_id} chunk[{chunk_index}]"
    return Stage1ChunkResult(
        hook_seeds=[
            _raw_hook_seed_from_dict(s, context=f"{context}.hook_seeds[{i}]")
            for i, s in enumerate(chunk["hook_seeds"])
        ],
        support_materials=[
            _raw_material_from_dict(m, context=f"{context}.support_materials[{i}]")
            for i, m in enumerate(chunk["support_materials"])
        ],
        fallback_spans=[
            _raw_fallback_span_from_dict(f, context=f"{context}.fallback_spans[{i}]")
            for i, f in enumerate(chunk["fallback_spans"])
        ],
    )


def save_stage1_chunk(video_id: str, chunk_index: int, result: Stage1ChunkResult) -> None:
    payload = _load_stage1_payload(video_id) or {
        "schema_version": config.CANDIDATE_SCHEMA_VERSION,
        "chunks": {},
    }
    payload["chunks"][str(chunk_index)] = {
        "hook_seeds": [asdict(s) for s in result.hook_seeds],
        "support_materials": [asdict(m) for m in result.support_materials],
        "fallback_spans": [asdict(f) for f in result.fallback_spans],
    }
    _atomic_write_text(stage1_path(video_id), json.dumps(payload, ensure_ascii=False, indent=2))


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
    return [
        _raw_candidate_from_dict(c, context=f"cache stage2 video_id={video_id} candidates[{i}]")
        for i, c in enumerate(raw["candidates"])
    ]


# Optional stage2_diagnostic.json fields that are only ever available at
# ONE of the two save points (see save_stage2_diagnostic) -- each is
# merge-preserved from the existing on-disk file when not re-passed, so
# neither save point clobbers what the other one wrote. materials_
# lookahead/selected_hook_seed_ids/attempts/fallback_ids/ranking come from
# the first save (design_final_candidates, right after the Structured
# Output parse); coverage_gap_hook_seed_ids/final_selected_ids come from
# the second (_design_finalize_and_cache, after local validation).
_MERGE_PRESERVED_DIAGNOSTIC_FIELDS = (
    "materials_lookahead",
    "selected_hook_seed_ids",
    "attempts",
    "fallback_ids",
    "ranking",
    "coverage_gap_hook_seed_ids",
    "final_selected_ids",
)


def save_stage2_diagnostic(
    video_id: str,
    candidates: list[RawClipCandidate],
    *,
    evaluations: list[dict] | None = None,
    materials_lookahead: dict | None = None,
    selected_hook_seed_ids: list[str] | None = None,
    attempts: list[dict] | None = None,
    fallback_ids: list[str] | None = None,
    ranking: list[str] | None = None,
    coverage_gap_hook_seed_ids: list[str] | None = None,
    final_selected_ids: list[str] | None = None,
) -> None:
    """Diagnostic-only snapshot of what Stage2 actually designed, written
    at two points: (1) immediately after Stage2's Structured Output is
    successfully parsed, before any local validation, so even a run that
    later selects zero final candidates leaves a record of what Stage2
    returned, and (2) again once local validation (and fallback-tier
    selection) has run, this time including per-candidate evaluations
    (accepted/reason/duration/opening text/opening_self_contained/hook_
    claim_resolved/opening_hook_strength/score) -- see _design_finalize_
    and_cache. Deliberately separate from stage2_path/save_stage2, which
    only ever holds the final, finalized, *accepted* candidates from a
    fully successful run: before this existed, a failed run left stage2_
    result.json completely untouched (correct -- see finalize_candidates'
    docstring) but also left NO record anywhere of what Stage2 had
    actually built, making a real "why did this fail" diagnosis
    impossible after the fact without a fresh, API-calling re-analysis.

    materials_lookahead/selected_hook_seed_ids/attempts/fallback_ids/
    ranking (first save) and coverage_gap_hook_seed_ids/final_selected_ids
    (second save) are each only ever available at ONE of the two save
    points -- see _MERGE_PRESERVED_DIAGNOSTIC_FIELDS. Whichever of these
    isn't passed on a given call is merged in from whatever the existing
    on-disk file (at the current schema version) already has under that
    key, rather than being silently dropped. This lets a future "was this
    hook seed ever tried, and why did it lose" diagnosis answer from one
    file: was it a selected coverage target, what did Stage2 return for
    it, did it pass the local hard gate, and was it in the final
    selection.

    Never read by the production candidate-selection path -- load_stage2
    only ever reads stage2_path, never this file. This is purely a
    developer/diagnostic artifact (see load_stage2_diagnostic).
    """
    payload = {
        "schema_version": config.CANDIDATE_SCHEMA_VERSION,
        "candidates": [asdict(c) for c in candidates],
    }
    if evaluations is not None:
        payload["evaluations"] = evaluations

    provided = {
        "materials_lookahead": materials_lookahead,
        "selected_hook_seed_ids": selected_hook_seed_ids,
        "attempts": attempts,
        "fallback_ids": fallback_ids,
        "ranking": ranking,
        "coverage_gap_hook_seed_ids": coverage_gap_hook_seed_ids,
        "final_selected_ids": final_selected_ids,
    }
    existing = load_stage2_diagnostic(video_id)
    existing_is_current = existing is not None and existing.get("schema_version") == config.CANDIDATE_SCHEMA_VERSION
    for field in _MERGE_PRESERVED_DIAGNOSTIC_FIELDS:
        value = provided[field]
        if value is not None:
            payload[field] = value
        elif existing_is_current and existing.get(field) is not None:
            payload[field] = existing[field]

    _atomic_write_text(
        stage2_diagnostic_path(video_id), json.dumps(payload, ensure_ascii=False, indent=2)
    )


def load_stage2_diagnostic(video_id: str) -> dict | None:
    """Diagnostic-only read-back (tooling/tests) -- returns the raw parsed
    JSON dict as-is (schema_version/candidates/optionally evaluations),
    never deserialized into RawClipCandidate objects, since this is meant
    to be inspected as data, never fed back into the selection pipeline.
    """
    path = stage2_diagnostic_path(video_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _raw_candidate_from_dict(d: dict, *, context: str) -> RawClipCandidate:
    """Diagnostic-only: validates shape before indexing so a malformed
    cached candidate raises a message naming exactly which cache entry and
    what it actually was, instead of a bare "TypeError: string indices
    must be integers, not 'str'".
    """
    require_dict(d, context=context)
    segments = []
    for i, s in enumerate(d["segments"]):
        require_dict(s, context=f"{context}.segments[{i}]")
        segments.append(RawUsedSegment(**s))
    return RawClipCandidate(
        hook_type=d["hook_type"],
        segments=segments,
        hook_text=d["hook_text"],
        opening_hook_strength=d["opening_hook_strength"],
        title=d["title"],
        description=d["description"],
        score=d["score"],
        reasoning=d["reasoning"],
        caveats=d["caveats"],
        # .get()-based with the same defaults as RawClipCandidate itself:
        # a cache entry written before these fields existed on the model
        # still deserializes cleanly (schema_version bumps invalidate
        # genuinely incompatible caches; these are additive and don't need
        # a hard miss).
        semantic_ending_complete=d.get("semantic_ending_complete", True),
        ending_rationale_code=d.get("ending_rationale_code", "natural_conclusion"),
        recomposed_for_duration=d.get("recomposed_for_duration", False),
        opening_self_contained=d.get("opening_self_contained", True),
        hook_claim_resolved=d.get("hook_claim_resolved", True),
    )


def _raw_material_from_dict(d: dict, *, context: str) -> RawMaterial:
    """The Stage1 (RawMaterial) mirror of _raw_candidate_from_dict --
    identical diagnostic-shape-validation approach, but for the material
    schema (material_type/usefulness_score, no hook_type/opening_hook_
    strength/score/hook_text/title/description/reasoning/caveats, no
    per-segment role/end_anchor_text).
    """
    require_dict(d, context=context)
    segments = []
    for i, s in enumerate(d["segments"]):
        require_dict(s, context=f"{context}.segments[{i}]")
        segments.append(RawMaterialSegment(**s))
    return RawMaterial(
        material_type=d["material_type"],
        segments=segments,
        usefulness_score=d["usefulness_score"],
    )


def _raw_hook_seed_from_dict(d: dict, *, context: str) -> RawHookSeed:
    """The RawHookSeed mirror of _raw_material_from_dict (signal_type/
    soft_score instead of material_type/usefulness_score)."""
    require_dict(d, context=context)
    segments = []
    for i, s in enumerate(d["segments"]):
        require_dict(s, context=f"{context}.segments[{i}]")
        segments.append(RawMaterialSegment(**s))
    return RawHookSeed(
        signal_type=d["signal_type"],
        segments=segments,
        soft_score=d["soft_score"],
    )


def _raw_fallback_span_from_dict(d: dict, *, context: str) -> RawFallbackSpan:
    """The RawFallbackSpan mirror of _raw_material_from_dict (safety_score
    instead of material_type/usefulness_score)."""
    require_dict(d, context=context)
    segments = []
    for i, s in enumerate(d["segments"]):
        require_dict(s, context=f"{context}.segments[{i}]")
        segments.append(RawMaterialSegment(**s))
    return RawFallbackSpan(
        segments=segments,
        safety_score=d["safety_score"],
    )
