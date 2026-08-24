"""Japanese-safe ffmpeg drawtext helpers (Windows-oriented).

Two deliberate choices to avoid mojibake/tofu and brittle escaping on
Windows:

- `fontfile=` is always passed explicitly instead of `font=` (fontconfig
  name lookup), which Windows ffmpeg builds frequently cannot resolve to a
  Japanese-capable face.
- Text is written to a UTF-8 temp file and referenced via `textfile=`
  instead of being inlined into the filter string. AI-generated
  hook/CTA/title text can contain punctuation, quotes, and colons that are
  fragile to hand-escape inline; textfile sidesteps that entirely.

`ensure_font_available()` fails fast (raises) rather than letting a
missing/unreadable font silently degrade rendering into tofu output.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from . import config


class FontNotAvailableError(RuntimeError):
    pass


def ensure_font_available() -> None:
    path = Path(config.FONT_PATH)
    if not path.exists() or not path.is_file():
        raise FontNotAvailableError(
            f"日本語フォントが見つかりません: {config.FONT_PATH}\n"
            "config.FONT_PATH (環境変数 PODCAST_CLIPPER_FONT_PATH) に、"
            "実在する日本語対応の.ttf/.otf(.ttcは非推奨)フォントファイルのパスを設定してください。"
        )


def _escape_path_for_filter(path: str) -> str:
    """Forward-slash + escape the drive-letter colon so the path is safe
    to embed as a `fontfile=`/`textfile=` value inside an ffmpeg filter
    string on Windows (e.g. C:/Windows/Fonts/x.ttf -> C\\:/Windows/Fonts/x.ttf).
    """
    normalized = path.replace("\\", "/")
    return normalized.replace(":", "\\:")


@dataclass
class TextOverlaySpec:
    text: str
    x_expr: str
    y_expr: str
    fontsize: int
    fontcolor: str = "white"
    box: bool = True
    box_color: str = "black@0.5"
    box_borderw: int = 12
    enable_expr: str | None = None


def write_textfile(text: str, tmp_dir: Path, name: str) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / f"text_{name}_{uuid.uuid4().hex[:8]}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def _drawtext_clause(spec: TextOverlaySpec, textfile_path: Path) -> str:
    ensure_font_available()
    fontfile = _escape_path_for_filter(str(Path(config.FONT_PATH).resolve()))
    textfile = _escape_path_for_filter(str(textfile_path.resolve()))
    parts = [
        f"fontfile='{fontfile}'",
        f"textfile='{textfile}'",
        f"fontsize={spec.fontsize}",
        f"fontcolor={spec.fontcolor}",
        f"x={spec.x_expr}",
        f"y={spec.y_expr}",
    ]
    if spec.box:
        parts += ["box=1", f"boxcolor={spec.box_color}", f"boxborderw={spec.box_borderw}"]
    if spec.enable_expr:
        parts.append(f"enable='{spec.enable_expr}'")
    return "drawtext=" + ":".join(parts)


def chain_drawtext_filters(
    input_label: str,
    output_label: str,
    specs: list[TextOverlaySpec],
    tmp_dir: Path,
) -> tuple[str, list[Path]]:
    """Builds an ffmpeg filter_complex fragment applying each spec's
    drawtext in sequence, plus the list of textfile paths it created
    (caller is responsible for cleaning these up after the render).
    """
    if not specs:
        return f"[{input_label}]null[{output_label}]", []

    textfiles: list[Path] = []
    clauses = []
    current = input_label
    for i, spec in enumerate(specs):
        textfile_path = write_textfile(spec.text, tmp_dir, name=f"{output_label}_{i}")
        textfiles.append(textfile_path)
        next_label = output_label if i == len(specs) - 1 else f"{output_label}_step{i}"
        clauses.append(f"[{current}]{_drawtext_clause(spec, textfile_path)}[{next_label}]")
        current = next_label

    return ";".join(clauses), textfiles
