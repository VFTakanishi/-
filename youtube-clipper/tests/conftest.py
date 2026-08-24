import shutil
import subprocess

import pytest

from podcast_clipper import config


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


HAS_FFMPEG = _has_ffmpeg()

requires_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not available in this environment")


@pytest.fixture(autouse=True)
def isolated_output_dir(tmp_path, monkeypatch):
    """Every test gets its own throwaway config.OUTPUT_DIR so cache/render/
    job state from one test never leaks into another.
    """
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "output")
    yield tmp_path
