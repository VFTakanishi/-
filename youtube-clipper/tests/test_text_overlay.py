import subprocess

import pytest

from conftest import find_available_japanese_font, requires_ffmpeg
from podcast_clipper import config, text_overlay


def test_escape_path_for_filter_converts_backslashes_and_escapes_colon():
    result = text_overlay._escape_path_for_filter("C:\\Windows\\Fonts\\meiryo.ttc")
    assert result == "C\\:/Windows/Fonts/meiryo.ttc"


def test_ensure_font_available_raises_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "FONT_PATH", str(tmp_path / "does-not-exist.ttf"))
    with pytest.raises(text_overlay.FontNotAvailableError):
        text_overlay.ensure_font_available()


def test_ensure_font_available_passes_when_file_exists(monkeypatch, tmp_path):
    font = tmp_path / "font.ttf"
    font.write_bytes(b"fake")
    monkeypatch.setattr(config, "FONT_PATH", str(font))
    text_overlay.ensure_font_available()  # should not raise


def test_write_textfile_writes_utf8(tmp_path):
    path = text_overlay.write_textfile("本編は関連動画から", tmp_path, name="cta")
    assert path.exists()
    assert path.read_text(encoding="utf-8") == "本編は関連動画から"


def test_chain_drawtext_filters_empty_specs_returns_passthrough(tmp_path):
    filter_str, textfiles = text_overlay.chain_drawtext_filters("0:v", "vout", [], tmp_path)
    assert filter_str == "[0:v]null[vout]"
    assert textfiles == []


def test_chain_drawtext_filters_builds_fontfile_and_textfile_options(monkeypatch, tmp_path):
    font = tmp_path / "font.ttf"
    font.write_bytes(b"fake")
    monkeypatch.setattr(config, "FONT_PATH", str(font))

    spec = text_overlay.TextOverlaySpec(
        text="テスト", x_expr="10", y_expr="20", fontsize=40, enable_expr="between(t,0,2)"
    )
    filter_str, textfiles = text_overlay.chain_drawtext_filters("0:v", "vout", [spec], tmp_path)

    assert len(textfiles) == 1
    assert "fontfile=" in filter_str
    assert "textfile=" in filter_str
    assert "enable='between(t,0,2)'" in filter_str
    assert filter_str.startswith("[0:v]drawtext=")
    assert filter_str.endswith("[vout]")


@requires_ffmpeg
def test_drawtext_actually_renders_with_real_ffmpeg(monkeypatch, tmp_path):
    """Smoke test: font-loading failure must fail the ffmpeg process (plan
    condition: font failure is never a silent success), and a valid font
    must let ffmpeg exit 0 and produce a real output file.
    """
    font_path = find_available_japanese_font()
    if font_path is None:
        pytest.skip(
            "no Japanese font available to smoke-test drawtext "
            "(set PODCAST_CLIPPER_FONT_PATH, or run where a CJK font is installed)"
        )
    monkeypatch.setattr(config, "FONT_PATH", font_path)

    spec = text_overlay.TextOverlaySpec(
        text="日本語テスト表示", x_expr="10", y_expr="10", fontsize=32
    )
    filter_str, textfiles = text_overlay.chain_drawtext_filters("0:v", "vout", [spec], tmp_path)

    out_path = tmp_path / "out.mp4"
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=320x240:d=1",
        "-filter_complex", filter_str, "-map", "[vout]",
        "-frames:v", "1", str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    for f in textfiles:
        f.unlink(missing_ok=True)

    assert result.returncode == 0, result.stderr[-2000:]
    assert out_path.exists() and out_path.stat().st_size > 0


@requires_ffmpeg
def test_drawtext_fails_when_font_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "FONT_PATH", str(tmp_path / "no-such-font.ttf"))
    spec = text_overlay.TextOverlaySpec(text="x", x_expr="0", y_expr="0", fontsize=20)
    with pytest.raises(text_overlay.FontNotAvailableError):
        text_overlay.chain_drawtext_filters("0:v", "vout", [spec], tmp_path)
