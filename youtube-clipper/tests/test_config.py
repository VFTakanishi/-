"""Cloud-deploy config: env-driven OUTPUT_DIR (pre-existing, pinned here as
a regression), cross-platform FONT_PATH default (round 8: Railway/Docker
has no Windows font directory), and TOOL_PASSWORD (round 8: simple
cloud-only password gate, see test_web.py for the actual gate behavior).
"""
from __future__ import annotations

import importlib
import os

from podcast_clipper import config


def test_output_dir_env_override_regression(monkeypatch, tmp_path):
    # C/D: PODCAST_CLIPPER_OUTPUT_DIR already existed before this round --
    # this pins that a cloud Volume path (e.g. /data/output) is honored,
    # and that the default (unset) still falls back to BASE_DIR/output,
    # exactly as local usage has always relied on.
    monkeypatch.setenv("PODCAST_CLIPPER_OUTPUT_DIR", str(tmp_path / "cloud-volume" / "output"))
    reloaded = importlib.reload(config)
    try:
        assert reloaded.OUTPUT_DIR == tmp_path / "cloud-volume" / "output"
    finally:
        monkeypatch.delenv("PODCAST_CLIPPER_OUTPUT_DIR", raising=False)
        importlib.reload(config)  # restore module state for every later test

    assert config.OUTPUT_DIR == config.BASE_DIR / "output"


def test_font_path_prefers_windows_path_when_present():
    # K: local Windows usage must see zero behavior change -- the Windows
    # path is still tried first, unconditionally.
    result = config._default_font_path(_exists=lambda p: p == config._WINDOWS_DEFAULT_FONT_PATH)
    assert result == config._WINDOWS_DEFAULT_FONT_PATH


def test_font_path_falls_back_to_linux_noto_cjk_when_windows_path_absent():
    # K/L: on Linux/Docker (no Windows font dir at all), falls back to the
    # Noto Sans CJK JP path the Dockerfile's `fonts-noto-cjk` apt package
    # installs -- so the CTA watermark ("VF高西で検索！") doesn't hit
    # text_overlay.ensure_font_available()'s "font not found" error, and
    # doesn't render as tofu/mojibake.
    noto_path = config._LINUX_FONT_PATH_CANDIDATES[0]
    result = config._default_font_path(_exists=lambda p: p == noto_path)
    assert result == noto_path


def test_font_path_falls_back_unchanged_when_nothing_found():
    # Neither the Windows path nor any Linux candidate exists -- must
    # return the original Windows path unchanged (never guess/invent a
    # path), so text_overlay.ensure_font_available()'s existing clear
    # error message still fires exactly as before this round.
    result = config._default_font_path(_exists=lambda p: False)
    assert result == config._WINDOWS_DEFAULT_FONT_PATH


def test_font_path_env_override_always_wins(monkeypatch):
    monkeypatch.setenv("PODCAST_CLIPPER_FONT_PATH", "/custom/my-font.otf")
    reloaded = importlib.reload(config)
    try:
        assert reloaded.FONT_PATH == "/custom/my-font.otf"
    finally:
        monkeypatch.delenv("PODCAST_CLIPPER_FONT_PATH", raising=False)
        importlib.reload(config)


def test_tool_password_unset_by_default():
    # E: no TOOL_PASSWORD env var at all -- gate stays fully disabled.
    assert os.environ.get("TOOL_PASSWORD") is None
    assert config.TOOL_PASSWORD is None


def test_tool_password_env_override(monkeypatch):
    monkeypatch.setenv("TOOL_PASSWORD", "sekret123")
    reloaded = importlib.reload(config)
    try:
        assert reloaded.TOOL_PASSWORD == "sekret123"
    finally:
        monkeypatch.delenv("TOOL_PASSWORD", raising=False)
        importlib.reload(config)


def test_tool_password_empty_string_treated_as_unset(monkeypatch):
    monkeypatch.setenv("TOOL_PASSWORD", "")
    reloaded = importlib.reload(config)
    try:
        assert reloaded.TOOL_PASSWORD is None
    finally:
        monkeypatch.delenv("TOOL_PASSWORD", raising=False)
        importlib.reload(config)
