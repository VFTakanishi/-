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


def test_trim_concat_filter_supports_single_and_triple_segment_candidates():
    one = [UsedSegment(role="hook", start=0, end=1, text="a")]
    three = [
        UsedSegment(role="hook", start=0, end=1, text="a"),
        UsedSegment(role="context", start=5, end=6, text="b"),
        UsedSegment(role="answer", start=10, end=11, text="c"),
    ]
    assert render._trim_concat_and_verticalize_filter(one).count("[0:v]trim=start=") == 1
    assert render._trim_concat_and_verticalize_filter(three).count("[0:v]trim=start=") == 3


def test_apply_text_overlays_never_builds_a_cta_spec(monkeypatch, tmp_path):
    """cta_end_text was retired entirely (an AI-authored closing CTA risked
    asserting unverified claims about the full episode). This pins that
    apply_text_overlays only ever burns in the watermark and hook_text --
    never a third, end-of-clip spec -- so no such subtitle can reappear.
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
    assert len(specs) == 2
    assert specs[0].text == config.WATERMARK_TEXT
    assert specs[1].text == candidate.hook_text


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
