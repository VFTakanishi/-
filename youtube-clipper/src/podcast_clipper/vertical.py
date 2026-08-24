"""Vertical (9:16) conversion: fit the whole source frame + blurred background.

Absolute condition #7: this is the *only* MVP conversion method. There is
no face-detection / auto-crop path — a tight crop risks cutting off a
person, on-screen text, or a screen-share, which matters more than the
slightly less "native vertical" look of letterboxing. Active-speaker
tracking is an explicit v2-or-later idea, not MVP scope.
"""
from __future__ import annotations

from . import config


def vertical_filter_chain(input_label: str, output_label: str) -> str:
    """Returns an ffmpeg filter_complex fragment that turns the video stream
    at `[input_label]` into a `[output_label]` stream of size
    VERTICAL_WIDTH x VERTICAL_HEIGHT: the source scaled to fit entirely
    within the frame, over a blurred/enlarged copy of the same source
    filling the rest of the canvas.

    `input_label` may be an intermediate pad (not a raw input stream), so
    the source is explicitly `split` before being consumed twice.
    """
    w, h, blur = config.VERTICAL_WIDTH, config.VERTICAL_HEIGHT, config.BACKGROUND_BLUR_SIGMA
    src1, src2 = f"{output_label}_src1", f"{output_label}_src2"
    bg, fg = f"{output_label}_bg", f"{output_label}_fg"
    return (
        f"[{input_label}]split=2[{src1}][{src2}];"
        f"[{src1}]scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{h},gblur=sigma={blur}[{bg}];"
        f"[{src2}]scale={w}:{h}:force_original_aspect_ratio=decrease[{fg}];"
        f"[{bg}][{fg}]overlay=(W-w)/2:(H-h)/2[{output_label}]"
    )
