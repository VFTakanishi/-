"""Tests for ingest.py -- the local video file upload path that replaced
download.py's YouTube fetch. Covers the minimum coverage explicitly
required for the MVP pivot: successful ingest, correct save location,
path-traversal prevention, empty/invalid input rejection, no-full-RAM-load
streaming, and content-hash-based dedup/cache-reuse via video_id.
"""
from __future__ import annotations

import io
import subprocess

import pytest

from conftest import requires_ffmpeg
from podcast_clipper import config, ingest


def _make_test_video(path, duration=1) -> bytes:
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=10:duration={duration}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
            "-c:v", "libx264", "-c:a", "aac", str(path),
        ],
        check=True, capture_output=True,
    )
    return path.read_bytes()


class _RecordingFile:
    """A minimal chunked-read file-like object that records every size
    requested via .read(size), so tests can assert ingest.py never asks
    for the whole file in one call (which would defeat streaming).
    """

    def __init__(self, data: bytes):
        self._buf = io.BytesIO(data)
        self.max_chunk_requested = 0
        self.read_calls = 0

    def read(self, size=-1):
        assert size is not None and size > 0, "ingest.py must request a bounded chunk size"
        self.max_chunk_requested = max(self.max_chunk_requested, size)
        self.read_calls += 1
        return self._buf.read(size)


@requires_ffmpeg
def test_ingest_uploaded_file_success_and_saved_at_correct_location(tmp_path):
    source = tmp_path / "source.mp4"
    data = _make_test_video(source)

    result = ingest.ingest_uploaded_file(io.BytesIO(data), "my podcast episode.mp4")

    assert result.video_id
    assert result.duration > 0
    assert result.path.exists()
    assert result.path == config.OUTPUT_DIR / result.video_id / "source" / f"{result.video_id}.mp4"


def test_path_traversal_prevention(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest, "_probe_duration", lambda path: 5.0)

    malicious_names = [
        "../../../../etc/passwd.mp4",
        "..\\..\\..\\Windows\\System32\\evil.mp4",
        "../../../secret.mov",
    ]
    for name in malicious_names:
        result = ingest.ingest_uploaded_file(io.BytesIO(b"fake video bytes"), name)
        resolved = result.path.resolve()
        output_root = config.OUTPUT_DIR.resolve()
        assert output_root in resolved.parents or resolved == output_root
        assert ".." not in result.path.parts


def test_empty_file_is_rejected(tmp_path):
    with pytest.raises(ingest.IngestError):
        ingest.ingest_uploaded_file(io.BytesIO(b""), "empty.mp4")

    # no leftover temp files under OUTPUT_DIR after a rejected upload
    tmp_dir = config.OUTPUT_DIR / "_ingest_tmp"
    assert not tmp_dir.exists() or not list(tmp_dir.iterdir())


@requires_ffmpeg
def test_invalid_non_video_file_is_rejected():
    with pytest.raises(ingest.IngestError):
        ingest.ingest_uploaded_file(io.BytesIO(b"this is not a video file at all"), "fake.mp4")

    tmp_dir = config.OUTPUT_DIR / "_ingest_tmp"
    assert not tmp_dir.exists() or not list(tmp_dir.iterdir())


def test_large_file_is_streamed_not_loaded_whole(monkeypatch):
    monkeypatch.setattr(ingest, "_probe_duration", lambda path: 30.0)
    monkeypatch.setattr(ingest, "_CHUNK_SIZE", 64 * 1024)

    data = b"x" * (64 * 1024 * 5 + 123)  # a few chunks plus a partial last chunk
    fake_file = _RecordingFile(data)

    result = ingest.ingest_uploaded_file(fake_file, "big.mp4")

    assert fake_file.max_chunk_requested <= ingest._CHUNK_SIZE
    assert fake_file.read_calls > 1  # streamed, never a single unbounded read
    assert result.path.stat().st_size == len(data)


def test_reuploading_identical_content_reuses_video_id(monkeypatch):
    monkeypatch.setattr(ingest, "_probe_duration", lambda path: 12.0)

    data = b"identical content bytes for dedup test"
    first = ingest.ingest_uploaded_file(io.BytesIO(data), "episode_v1.mp4")
    second = ingest.ingest_uploaded_file(io.BytesIO(data), "episode_v2_renamed.mp4")

    assert first.video_id == second.video_id
    assert first.path == second.path
    assert first.path.exists()


def test_different_content_yields_different_video_id(monkeypatch):
    monkeypatch.setattr(ingest, "_probe_duration", lambda path: 12.0)

    first = ingest.ingest_uploaded_file(io.BytesIO(b"content A"), "a.mp4")
    second = ingest.ingest_uploaded_file(io.BytesIO(b"content B"), "b.mp4")

    assert first.video_id != second.video_id
