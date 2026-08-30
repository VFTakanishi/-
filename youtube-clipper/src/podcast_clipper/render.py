"""Renders one selected ClipCandidate to a final mp4.

Plan fix #7 requires this exact order (video-level Content QA must see the
vertical video *before* any text is burned in, otherwise a black/frozen
source frame could be masked by the overlay text sitting on top of it):

  1. extract + concat the candidate's segments from the source video
  2. convert to vertical (blurred background fit)      -> intermediate mp4
  3. [qa.py video Content QA runs on the intermediate]   (called by caller)
  4. burn in the watermark                               -> final mp4
  5. [qa.py technical + audio/speech QA runs on the final] (called by caller)

This module only produces the two video files and a RenderManifest
recording exactly what edit points/text were used; qa.py (called by
web.py/jobs.py between steps) decides pass/fail.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

from . import config, text_overlay, vertical
from .models import ClipCandidate, RenderManifest, UsedSegment


class FfmpegError(RuntimeError):
    pass


def _run_ffmpeg(args: list[str]) -> None:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise FfmpegError(
            "ffmpeg failed (exit "
            f"{result.returncode}): {' '.join(args)}\n--- stderr (tail) ---\n"
            + result.stderr[-4000:]
        )


def _render_dir(video_id: str) -> Path:
    d = config.OUTPUT_DIR / video_id / "renders"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _tmp_dir(video_id: str, candidate_id: str) -> Path:
    d = config.OUTPUT_DIR / video_id / "renders" / f"tmp_{candidate_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _trim_concat_and_verticalize_filter(segments: list[UsedSegment]) -> str:
    clauses = []
    video_labels, audio_labels = [], []
    for i, seg in enumerate(segments):
        v, a = f"v{i}", f"a{i}"
        clauses.append(f"[0:v]trim=start={seg.start}:end={seg.end},setpts=PTS-STARTPTS[{v}]")
        clauses.append(f"[0:a]atrim=start={seg.start}:end={seg.end},asetpts=PTS-STARTPTS[{a}]")
        video_labels.append(v)
        audio_labels.append(a)

    concat_inputs = "".join(f"[{v}][{a}]" for v, a in zip(video_labels, audio_labels))
    clauses.append(f"{concat_inputs}concat=n={len(segments)}:v=1:a=1[vcat][acat]")
    clauses.append(vertical.vertical_filter_chain("vcat", "vout"))
    return ";".join(clauses)


def extract_and_verticalize(
    source_path: Path, candidate: ClipCandidate, out_dir: Path
) -> Path:
    """Step 1-2: extract the candidate's segments, concat, convert to
    vertical. No text is drawn yet — this file is what video Content QA
    (black/freeze detection) must run against.
    """
    filter_complex = _trim_concat_and_verticalize_filter(candidate.segments)
    out_path = out_dir / f"clip_{candidate.id}_novtext.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(source_path),
        "-filter_complex", filter_complex,
        "-map", "[vout]", "-map", "[acat]",
        "-c:v", "libx264", "-c:a", "aac",
        "-movflags", "+faststart",
        str(out_path),
    ]
    _run_ffmpeg(cmd)
    return out_path


def apply_text_overlays(
    intermediate_path: Path,
    candidate: ClipCandidate,
    out_dir: Path,
    tmp_dir: Path,
) -> Path:
    """Step 4: burn in the always-on watermark. No hook-text overlay and no
    end-of-clip CTA text is burned in -- the only in-video text is the
    watermark; the main funnel to the full episode is the YouTube Studio
    "related video" setting, not an in-video subtitle. candidate.hook_text
    remains a data field (real transcript text, used only around
    candidate selection) but is not drawn into the video.
    """
    specs = [
        text_overlay.TextOverlaySpec(
            text=config.WATERMARK_TEXT,
            x_expr="(w-text_w)/2",
            y_expr="h-text_h-140",
            fontsize=56,
            fontcolor="white",
            box=True,
            box_color="black@0.55",
            box_borderw=18,
        ),
    ]

    filter_complex, textfiles = text_overlay.chain_drawtext_filters(
        "0:v", "vout", specs, tmp_dir
    )
    out_path = out_dir / f"clip_{candidate.id}.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(intermediate_path),
        "-filter_complex", filter_complex,
        "-map", "[vout]", "-map", "0:a",
        "-c:v", "libx264", "-c:a", "copy",
        "-movflags", "+faststart",
        str(out_path),
    ]
    try:
        _run_ffmpeg(cmd)
    finally:
        for f in textfiles:
            f.unlink(missing_ok=True)
    return out_path


def render_candidate(
    source_path: Path, candidate: ClipCandidate, video_id: str
) -> RenderManifest:
    out_dir = _render_dir(video_id)
    tmp_dir = _tmp_dir(video_id, candidate.id)

    intermediate_path = extract_and_verticalize(source_path, candidate, out_dir)
    final_path = apply_text_overlays(intermediate_path, candidate, out_dir, tmp_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    manifest = RenderManifest(
        video_id=video_id,
        candidate_id=candidate.id,
        segments=candidate.segments,
        hook_text=candidate.hook_text,
        watermark_text=config.WATERMARK_TEXT,
        total_duration=candidate.total_duration,
        intermediate_video_path=str(intermediate_path),
        final_video_path=str(final_path),
    )
    manifest_path = out_dir / f"clip_{candidate.id}_manifest.json"
    manifest_path.write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
