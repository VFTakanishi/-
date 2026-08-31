import subprocess
from pathlib import Path

import pytest

from conftest import find_available_japanese_font, requires_ffmpeg
from podcast_clipper import config, qa, render
from podcast_clipper.models import ClipCandidate, UsedSegment


def _candidate(segments):
    return ClipCandidate(
        id="c1", hook_type="story", segments=segments, hook_text="つかみのテキスト",
        opening_hook_strength=80, title="t", description="d",
        score=80, reasoning="r", caveats="",
    )


def test_trim_concat_and_verticalize_filter_has_one_trim_pair_per_segment():
    segments = [
        UsedSegment(role="hook", start=1.0, end=3.0, text="a"),
        UsedSegment(role="answer", start=10.0, end=13.5, text="b"),
    ]
    filt = render._trim_concat_and_verticalize_filter(segments)

    assert filt.count("[0:v]trim=start=") == 2
    assert filt.count("[0:a]atrim=start=") == 2
    assert "start=1.0:end=3.0" in filt
    assert "start=10.0:end=13.5" in filt
    assert "concat=n=2:v=1:a=1[vcat][acat]" in filt
    # the vertical conversion must be chained onto the concatenated stream
    assert "[vcat]split=2" in filt


def test_trim_concat_and_verticalize_filter_never_reorders_by_timestamp():
    # D: a reordered candidate (hook = chronologically-*later* segment,
    # context = chronologically-earlier segment) must render in exactly
    # the given list order -- render.py must never re-sort segments by
    # their start time back into chronological order.
    segments = [
        UsedSegment(role="hook", start=10.0, end=13.0, text="later, but plays first"),
        UsedSegment(role="context", start=1.0, end=3.0, text="earlier, but plays second"),
    ]
    filt = render._trim_concat_and_verticalize_filter(segments)

    # v0/a0 (the first trim/concat slot) must correspond to the *first*
    # list entry (start=10.0), not the chronologically-earliest one.
    first_trim_clause = filt.split(";")[0]
    assert "start=10.0:end=13.0" in first_trim_clause
    assert filt.index("start=10.0:end=13.0") < filt.index("start=1.0:end=3.0")
    assert "[v0][a0][v1][a1]concat=n=2:v=1:a=1[vcat][acat]" in filt


def test_trim_concat_filter_supports_single_and_triple_segment_candidates():
    one = [UsedSegment(role="hook", start=0, end=1, text="a")]
    three = [
        UsedSegment(role="hook", start=0, end=1, text="a"),
        UsedSegment(role="context", start=5, end=6, text="b"),
        UsedSegment(role="answer", start=10, end=11, text="c"),
    ]
    assert render._trim_concat_and_verticalize_filter(one).count("[0:v]trim=start=") == 1
    assert render._trim_concat_and_verticalize_filter(three).count("[0:v]trim=start=") == 3


def test_apply_text_overlays_only_ever_burns_in_the_watermark(monkeypatch, tmp_path):
    """The hook-text overlay was retired entirely (per user decision: the
    only in-video text should be the always-on watermark), and
    cta_end_text was retired earlier (an AI-authored closing CTA risked
    asserting unverified claims about the full episode). This pins that
    apply_text_overlays burns in exactly one spec -- the watermark -- and
    never a hook or end-of-clip spec, so neither can reappear.
    """
    captured = {}

    def fake_chain(input_label, output_label, specs, tmp_dir):
        captured["specs"] = specs
        return f"[{input_label}]null[{output_label}]", []

    monkeypatch.setattr(render.text_overlay, "chain_drawtext_filters", fake_chain)
    monkeypatch.setattr(render, "_run_ffmpeg", lambda cmd: None)

    candidate = _candidate([UsedSegment(role="hook", start=0.0, end=2.0, text="a")])
    render.apply_text_overlays(Path("intermediate.mp4"), candidate, tmp_path, tmp_path)

    specs = captured["specs"]
    assert len(specs) == 1
    assert specs[0].text == config.WATERMARK_TEXT


def test_watermark_spec_is_prominent_and_always_on(monkeypatch, tmp_path):
    """Pins the more-prominent styling (bigger, fully-opaque, bottom-center
    safe area) and that it's never time-gated (always visible for the
    whole clip, unlike the retired hook overlay).
    """
    captured = {}

    def fake_chain(input_label, output_label, specs, tmp_dir):
        captured["specs"] = specs
        return f"[{input_label}]null[{output_label}]", []

    monkeypatch.setattr(render.text_overlay, "chain_drawtext_filters", fake_chain)
    monkeypatch.setattr(render, "_run_ffmpeg", lambda cmd: None)

    candidate = _candidate([UsedSegment(role="hook", start=0.0, end=2.0, text="a")])
    render.apply_text_overlays(Path("intermediate.mp4"), candidate, tmp_path, tmp_path)

    watermark = captured["specs"][0]
    assert watermark.enable_expr is None  # always on, no time window
    assert watermark.fontsize >= 48  # bigger than the old fontsize=36
    assert watermark.fontcolor == "white"  # fully opaque, not translucent
    assert watermark.box is True
    assert "(w-text_w)/2" == watermark.x_expr  # horizontally centered
    assert "h-text_h" in watermark.y_expr  # bottom safe area

    # Pins the stronger CTA styling values, and that render.py reads them
    # from config rather than hardcoding them (see config.WATERMARK_*).
    assert watermark.fontsize == config.WATERMARK_FONT_SIZE == 80
    assert watermark.box_color == config.WATERMARK_BOX_COLOR == "black@0.82"
    assert watermark.box_borderw == config.WATERMARK_BOX_BORDERW == 26
    assert watermark.y_expr == f"h-text_h-{config.WATERMARK_BOTTOM_MARGIN}"
    assert config.WATERMARK_BOTTOM_MARGIN == 210


def test_watermark_spec_follows_config_overrides(monkeypatch, tmp_path):
    """render.py must read the watermark styling from config at call time
    (not bake in the defaults), so operators can retune it without editing
    render.py -- see config.py's WATERMARK_* constants.
    """
    monkeypatch.setattr(config, "WATERMARK_FONT_SIZE", 99)
    monkeypatch.setattr(config, "WATERMARK_BOX_COLOR", "black@0.5")
    monkeypatch.setattr(config, "WATERMARK_BOX_BORDERW", 5)
    monkeypatch.setattr(config, "WATERMARK_BOTTOM_MARGIN", 1)

    captured = {}

    def fake_chain(input_label, output_label, specs, tmp_dir):
        captured["specs"] = specs
        return f"[{input_label}]null[{output_label}]", []

    monkeypatch.setattr(render.text_overlay, "chain_drawtext_filters", fake_chain)
    monkeypatch.setattr(render, "_run_ffmpeg", lambda cmd: None)

    candidate = _candidate([UsedSegment(role="hook", start=0.0, end=2.0, text="a")])
    render.apply_text_overlays(Path("intermediate.mp4"), candidate, tmp_path, tmp_path)

    watermark = captured["specs"][0]
    assert watermark.fontsize == 99
    assert watermark.box_color == "black@0.5"
    assert watermark.box_borderw == 5
    assert watermark.y_expr == "h-text_h-1"


@requires_ffmpeg
def test_render_candidate_end_to_end_produces_valid_files(monkeypatch, tmp_path):
    font_path = find_available_japanese_font()
    if font_path is None:
        pytest.skip(
            "no Japanese font available to exercise the full render pipeline "
            "(set PODCAST_CLIPPER_FONT_PATH, or run where a CJK font is installed)"
        )
    monkeypatch.setattr(config, "FONT_PATH", font_path)

    # Build a 20s synthetic source video (colour bars + tone) so segment
    # extraction has real content to cut from.
    source_path = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=size=640x360:rate=15:duration=20",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=20",
            "-c:v", "libx264", "-c:a", "aac", str(source_path),
        ],
        check=True, capture_output=True,
    )

    candidate = _candidate(
        [
            UsedSegment(role="hook", start=1.0, end=4.0, text="フック部分"),
            UsedSegment(role="answer", start=10.0, end=13.0, text="答え部分"),
        ]
    )

    manifest = render.render_candidate(source_path, candidate, video_id="testvid")

    assert Path(manifest.intermediate_video_path).exists()
    assert Path(manifest.final_video_path).exists()
    assert manifest.total_duration == 6.0  # (4-1) + (13-10)

    technical_checks = qa.technical_qa(Path(manifest.final_video_path), manifest.total_duration)
    assert all(c.passed for c in technical_checks), [
        (c.name, c.detail) for c in technical_checks if not c.passed
    ]
