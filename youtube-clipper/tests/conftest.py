import glob
import shutil
import subprocess
from pathlib import Path

import pytest

from podcast_clipper import config


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


HAS_FFMPEG = _has_ffmpeg()

requires_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not available in this environment")


def find_available_japanese_font() -> str | None:
    """Resolves a real, existing Japanese-capable font file to smoke-test
    drawtext against, OS-independently.

    Prefers whatever is actually configured (config.FONT_PATH already
    reflects PODCAST_CLIPPER_FONT_PATH when set -- e.g. by CI on Windows
    after locating/downloading a font). Falls back to a common Linux CJK
    font location for local development convenience when no env var is
    set. Returns None if nothing is found, so callers can skip.
    """
    if Path(config.FONT_PATH).exists():
        return config.FONT_PATH
    linux_fonts = glob.glob("/usr/share/fonts/**/NotoSansCJK*.ttc", recursive=True)
    return linux_fonts[0] if linux_fonts else None


@pytest.fixture(autouse=True)
def isolated_output_dir(tmp_path, monkeypatch):
    """Every test gets its own throwaway config.OUTPUT_DIR so cache/render/
    job state from one test never leaks into another.
    """
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "output")
    yield tmp_path
