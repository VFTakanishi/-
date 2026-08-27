"""Download a YouTube video (and its metadata) with yt-dlp.

Resolution is capped at 1080p and h264 is preferred (config.YT_DLP_FORMAT):
the final deliverable is a ~1080x1920 short, so pulling 4K/8K source wastes
bandwidth/disk, and h264 avoids a slow re-encode from AV1/VP9 during the
ffmpeg pipeline stages.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yt_dlp

from . import config

# YouTube increasingly requires a proof-of-origin token for its default
# "web" player client, which datacenter IPs (including CI runners) often
# can't satisfy and get rejected with "Sign in to confirm you're not a
# bot." The "tv" and "android" clients don't have that requirement as of
# this writing, so trying them first is yt-dlp's documented mitigation
# for that error (see https://github.com/yt-dlp/yt-dlp/wiki/FAQ). This
# does not touch the pipeline's own logic -- extraction still returns the
# same info dict shape either way.
_YT_DLP_COMMON_OPTS = {
    "extractor_args": {"youtube": {"player_client": ["tv", "android", "web"]}},
}


@dataclass
class VideoInfo:
    video_id: str
    title: str
    duration: float
    path: Path


def _safe_dirname(title: str) -> str:
    cleaned = re.sub(r"[^\w\-一-龥ぁ-んァ-ヶー ]+", "_", title).strip()
    return cleaned[:80] or "video"


def download_video(url: str, out_root: Path, force_refresh: bool = False) -> VideoInfo:
    """Download the source video for `url` under `out_root/<video_id>/source/`.

    Returns metadata needed by the rest of the pipeline (video id, title,
    duration, and the local file path). A lightweight metadata-only probe
    resolves the video id first; if a source file for that id already
    exists on disk, the actual download is skipped (this is what lets an
    interrupted job, or a render step running after analyze, reuse the
    already-downloaded file instead of re-fetching from YouTube).
    """
    probe_opts = {"quiet": True, "noplaylist": True, **_YT_DLP_COMMON_OPTS}
    with yt_dlp.YoutubeDL(probe_opts) as probe:
        probe_info = probe.extract_info(url, download=False)
    video_id = probe_info["id"]

    out_dir = out_root / video_id / "source"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob(f"{video_id}.*"))

    if existing and not force_refresh:
        path = existing[0]
        info = probe_info
    else:
        ydl_opts = {
            "format": config.YT_DLP_FORMAT,
            "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
            "merge_output_format": "mp4",
            "noplaylist": True,
            "quiet": True,
            **_YT_DLP_COMMON_OPTS,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            path = Path(ydl.prepare_filename(info))
            if not path.exists():
                # merge_output_format may have changed the extension
                path = path.with_suffix(".mp4")

    return VideoInfo(
        video_id=video_id,
        title=info.get("title", video_id),
        duration=float(info.get("duration") or 0.0),
        path=path,
    )
