import json
import subprocess
from pathlib import Path

from podcast_clipper import boundary, config, qa
from podcast_clipper.models import (
    RawClipCandidate,
    RawUsedSegment,
    RenderManifest,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
    UsedSegment,
)


def _fake_completed(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


# --- black/freeze detection parsing (mocked ffmpeg stderr) ---------------


def test_video_content_qa_flags_opening_black_frame(monkeypatch):
    def fake_run(args):
        if "blackdetect" in " ".join(args):
            return _fake_completed(stderr="[blackdetect @ 0x0] black_start:0 black_end:1.2\n")
        return _fake_completed(stderr="")  # no freeze

    monkeypatch.setattr(qa, "_run", fake_run)
    checks = qa.video_content_qa(Path("dummy.mp4"))
    black_check = next(c for c in checks if c.name == "黒画面検出")
    assert black_check.passed is False
    assert black_check.critical is True


def test_video_content_qa_passes_when_no_opening_black_or_freeze(monkeypatch):
    monkeypatch.setattr(qa, "_run", lambda args: _fake_completed(stderr=""))
    checks = qa.video_content_qa(Path("dummy.mp4"))
    assert all(c.passed for c in checks)


def test_video_content_qa_flags_opening_freeze(monkeypatch):
    def fake_run(args):
        if "freezedetect" in " ".join(args):
            return _fake_completed(stderr="[freezedetect @ 0x0] freeze_start: 0.5\n")
        return _fake_completed(stderr="")

    monkeypatch.setattr(qa, "_run", fake_run)
    checks = qa.video_content_qa(Path("dummy.mp4"))
    freeze_check = next(c for c in checks if c.name == "静止画/フリーズ検出")
    assert freeze_check.passed is False
    assert freeze_check.critical is True


def test_video_content_qa_ignores_freeze_well_after_opening(monkeypatch):
    def fake_run(args):
        if "freezedetect" in " ".join(args):
            return _fake_completed(stderr="[freezedetect @ 0x0] freeze_start: 20.0\n")
        return _fake_completed(stderr="")

    monkeypatch.setattr(qa, "_run", fake_run)
    checks = qa.video_content_qa(Path("dummy.mp4"))
    freeze_check = next(c for c in checks if c.name == "静止画/フリーズ検出")
    assert freeze_check.passed is True


# --- audio presence QA (signal only, not "speech" -- plan fix #6) -------


def test_audio_presence_qa_flags_silent_start_as_critical(monkeypatch):
    monkeypatch.setattr(qa, "_mean_volume_db", lambda path, start=None, t=None: -80.0)
    checks = qa.audio_presence_qa(Path("dummy.mp4"))
    start_check = next(c for c in checks if c.name == "無音開始チェック")
    assert start_check.passed is False
    assert start_check.critical is True


def test_audio_presence_qa_low_overall_volume_is_warning_not_critical(monkeypatch):
    calls = {"n": 0}

    def fake_mean_volume(path, start=None, t=None):
        calls["n"] += 1
        # First call = opening window (fine), second = overall (low but not silent)
        return -10.0 if calls["n"] == 1 else -40.0

    monkeypatch.setattr(qa, "_mean_volume_db", fake_mean_volume)
    checks = qa.audio_presence_qa(Path("dummy.mp4"))
    overall_check = next(c for c in checks if c.name == "全体音量チェック")
    assert overall_check.passed is False
    assert overall_check.critical is False  # warning only, never blocks download


# --- technical QA (mocked ffprobe) ---------------------------------------


def test_technical_qa_passes_with_matching_resolution(monkeypatch):
    probe_json = {
        "streams": [
            {"codec_type": "video", "width": config.VERTICAL_WIDTH, "height": config.VERTICAL_HEIGHT, "codec_name": "h264"},
            {"codec_type": "audio", "codec_name": "aac"},
        ],
        "format": {"duration": "30.0"},
    }
    monkeypatch.setattr(qa, "_run", lambda args: _fake_completed(stdout=json.dumps(probe_json)))
    checks = qa.technical_qa(Path("dummy.mp4"), expected_duration=30.0)
    assert all(c.passed for c in checks)


def test_technical_qa_fails_on_wrong_resolution(monkeypatch):
    probe_json = {
        "streams": [
            {"codec_type": "video", "width": 1920, "height": 1080, "codec_name": "h264"},
            {"codec_type": "audio", "codec_name": "aac"},
        ],
        "format": {"duration": "30.0"},
    }
    monkeypatch.setattr(qa, "_run", lambda args: _fake_completed(stdout=json.dumps(probe_json)))
    checks = qa.technical_qa(Path("dummy.mp4"), expected_duration=30.0)
    res_check = next(c for c in checks if c.name == "解像度")
    assert res_check.passed is False
    assert res_check.critical is True


def test_technical_qa_fails_when_ffprobe_cannot_open_file(monkeypatch):
    monkeypatch.setattr(qa, "_run", lambda args: _fake_completed(returncode=1, stderr="no such file"))
    checks = qa.technical_qa(Path("missing.mp4"), expected_duration=30.0)
    assert len(checks) == 1
    assert checks[0].passed is False
    assert checks[0].critical is True


# --- boundary integrity / speech-start self-consistency ------------------


def _transcript():
    return Transcript(
        video_id="vidQ",
        language="ja",
        segments=[
            TranscriptSegment(id=0, start=10.0, end=12.0, text="a", words=[TranscriptWord(10.0, 12.0, "a")]),
            TranscriptSegment(id=1, start=13.0, end=16.0, text="b", words=[TranscriptWord(13.0, 16.0, "b")]),
        ],
    )


def test_boundary_integrity_qa_passes_when_manifest_matches_recomputation():
    transcript = _transcript()
    raw = RawClipCandidate(
        hook_type="story",
        segments=[RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=1)],
        hook_text="h", cta_end_text="c", title="t", description="d", score=1, reasoning="r", caveats="",
    )
    resolved = boundary.resolve_candidate(raw, transcript, candidate_id="c1")
    manifest = RenderManifest(
        video_id="vidQ", candidate_id="c1", segments=resolved.segments,
        hook_text="h", watermark_text="w", cta_end_text="c",
        total_duration=resolved.total_duration,
        intermediate_video_path="mid.mp4", final_video_path="final.mp4",
    )
    check = qa.boundary_integrity_qa(raw, transcript, manifest)
    assert check.passed is True


def test_boundary_integrity_qa_fails_when_manifest_diverges():
    transcript = _transcript()
    raw = RawClipCandidate(
        hook_type="story",
        segments=[RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=1)],
        hook_text="h", cta_end_text="c", title="t", description="d", score=1, reasoning="r", caveats="",
    )
    tampered_segment = UsedSegment(role="hook", start=999.0, end=1000.0, text="wrong")
    manifest = RenderManifest(
        video_id="vidQ", candidate_id="c1", segments=[tampered_segment],
        hook_text="h", watermark_text="w", cta_end_text="c",
        total_duration=1.0,
        intermediate_video_path="mid.mp4", final_video_path="final.mp4",
    )
    check = qa.boundary_integrity_qa(raw, transcript, manifest)
    assert check.passed is False
    assert check.critical is True


def test_speech_start_alignment_qa_uses_first_segment_only():
    transcript = _transcript()
    raw = RawClipCandidate(
        hook_type="story",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0),
            RawUsedSegment(role="answer", start_segment_id=1, end_segment_id=1),
        ],
        hook_text="h", cta_end_text="c", title="t", description="d", score=1, reasoning="r", caveats="",
    )
    resolved = boundary.resolve_candidate(raw, transcript, candidate_id="c1")
    manifest = RenderManifest(
        video_id="vidQ", candidate_id="c1", segments=resolved.segments,
        hook_text="h", watermark_text="w", cta_end_text="c",
        total_duration=resolved.total_duration,
        intermediate_video_path="mid.mp4", final_video_path="final.mp4",
    )
    check = qa.speech_start_alignment_qa(raw, transcript, manifest)
    assert check.passed is True


# --- ordering guarantee (plan fix #7): video Content QA must run on the --
# --- intermediate (pre-text) video, never on the final (post-text) mp4  --


def test_run_full_qa_runs_video_content_qa_on_intermediate_only(monkeypatch):
    transcript = _transcript()
    raw = RawClipCandidate(
        hook_type="story",
        segments=[RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=1)],
        hook_text="h", cta_end_text="c", title="t", description="d", score=1, reasoning="r", caveats="",
    )
    resolved = boundary.resolve_candidate(raw, transcript, candidate_id="c1")
    manifest = RenderManifest(
        video_id="vidQ", candidate_id="c1", segments=resolved.segments,
        hook_text="h", watermark_text="w", cta_end_text="c",
        total_duration=resolved.total_duration,
        intermediate_video_path="intermediate_novtext.mp4", final_video_path="final.mp4",
    )

    seen_video_content_qa_paths = []
    seen_technical_qa_paths = []
    seen_audio_qa_paths = []

    monkeypatch.setattr(qa, "video_content_qa", lambda p: (seen_video_content_qa_paths.append(str(p)), [])[1])
    monkeypatch.setattr(qa, "technical_qa", lambda p, d: (seen_technical_qa_paths.append(str(p)), [])[1])
    monkeypatch.setattr(qa, "audio_presence_qa", lambda p: (seen_audio_qa_paths.append(str(p)), [])[1])
    monkeypatch.setattr(qa, "extract_thumbnails", lambda *a, **k: [])

    qa.run_full_qa(raw, transcript, manifest)

    assert seen_video_content_qa_paths == ["intermediate_novtext.mp4"]
    assert seen_technical_qa_paths == ["final.mp4"]
    assert seen_audio_qa_paths == ["final.mp4"]
    # Critically: the final (post-text) mp4 must never be passed to video_content_qa
    assert "final.mp4" not in seen_video_content_qa_paths
