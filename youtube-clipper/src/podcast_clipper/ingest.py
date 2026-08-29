"""Ingest a locally-uploaded video file (replaces download.py).

The MVP input is a video file the user already owns and has uploaded to
their own YouTube channel -- there is no need to re-fetch it from YouTube,
so this module has no network dependency at all (no yt-dlp, no bot
detection, no cookies, no datacenter-IP problems).

Design points:
- The upload is streamed to disk in fixed-size chunks (`_CHUNK_SIZE`) and
  never buffered whole in memory, so this scales to large video files.
- `video_id` is derived from a SHA-256 hash of the file's *content*,
  computed while streaming (single pass). Re-uploading byte-identical
  content always resolves to the same video_id, which is what lets
  transcribe.py/clip_selector.py's existing video_id-keyed cache
  (cache.py) skip redundant Whisper/Claude work on re-analysis.
- The caller-supplied filename is only ever used for display (`title`)
  and to pick a safe file extension; it is never used to build a path, so
  a malicious filename (e.g. containing "..") cannot escape
  `config.OUTPUT_DIR`.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from . import config

_CHUNK_SIZE = 1024 * 1024  # 1 MiB

# Containers ffmpeg/ffprobe can read directly; anything else falls back to
# .mp4 (the content is sniffed by ffmpeg from the bytes regardless, so an
# unrecognized-but-allowed extension does not make the file unreadable).
_ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}


class IngestError(ValueError):
    """Raised for an empty upload or a file ffprobe cannot read as video."""


@dataclass
class IngestResult:
    video_id: str
    title: str
    duration: float
    path: Path


def _safe_extension(filename: str | None) -> str:
    if not filename:
        return ".mp4"
    ext = Path(filename).suffix.lower()
    return ext if ext in _ALLOWED_EXTENSIONS else ".mp4"


def _probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True, text=True,
    )
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        duration = 0.0
    if result.returncode != 0 or duration <= 0:
        raise IngestError(
            "アップロードされたファイルを動画として読み取れませんでした"
            f"（ffprobe: {result.stderr.strip()[-500:]}）"
        )
    return duration


def ingest_uploaded_file(
    fileobj: BinaryIO, original_filename: str | None, display_title: str | None = None
) -> IngestResult:
    """Streams `fileobj` (any object with a chunked `.read(size)`, e.g. the
    underlying file of a FastAPI/Starlette `UploadFile`, or a plain local
    file opened in "rb" mode) to disk under `config.OUTPUT_DIR`, hashing as
    it goes. Raises `IngestError` for an empty or unreadable file.
    """
    ext = _safe_extension(original_filename)
    tmp_dir = config.OUTPUT_DIR / "_ingest_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"upload_{uuid.uuid4().hex}{ext}"

    hasher = hashlib.sha256()
    total_bytes = 0
    try:
        with open(tmp_path, "wb") as out:
            while True:
                chunk = fileobj.read(_CHUNK_SIZE)
                if not chunk:
                    break
                total_bytes += len(chunk)
                hasher.update(chunk)
                out.write(chunk)

        if total_bytes == 0:
            raise IngestError("アップロードされたファイルが空です")

        # Validate (via ffprobe) before the file ever reaches its permanent
        # location, so an invalid upload never lingers under OUTPUT_DIR.
        duration = _probe_duration(tmp_path)

        video_id = hasher.hexdigest()[:16]
        out_dir = config.OUTPUT_DIR / video_id / "source"
        out_dir.mkdir(parents=True, exist_ok=True)
        final_path = out_dir / f"{video_id}{ext}"

        if final_path.exists():
            # Same content already ingested: reuse it (and, by extension,
            # any transcript/Stage1/Stage2 cache already keyed on video_id).
            tmp_path.unlink(missing_ok=True)
        else:
            shutil.move(str(tmp_path), str(final_path))
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    title = (display_title or original_filename or video_id).strip() or video_id
    return IngestResult(video_id=video_id, title=title, duration=duration, path=final_path)
