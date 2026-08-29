"""QA (absolute condition #13), split into the checks plan fixes #2, #6, #7
call for. Each check is independent and separately labeled so the UI can
show exactly what failed rather than one opaque pass/fail.

Ordering matters (plan fix #7): `video_content_qa` must be run by the
caller against the *intermediate* (pre-text) video from render.py, before
text is burned in. Everything else here runs against the *final* mp4.
Nothing in this module re-decides whether Claude's semantic selection was
good — `boundary_integrity_qa`/`speech_start_alignment_qa` only verify
render.py actually used the exact edit points boundary.py computed
(self-consistency, not a fresh audio/semantic judgement).
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from . import boundary, config
from .models import RawClipCandidate, RenderManifest, Transcript


@dataclass
class QACheck:
    name: str
    passed: bool
    critical: bool
    detail: str


@dataclass
class QAReport:
    checks: list[QACheck] = field(default_factory=list)
    thumbnails: list[str] = field(default_factory=list)

    @property
    def download_allowed(self) -> bool:
        return not any(c.critical and not c.passed for c in self.checks)


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True)


# --- Video Content QA (runs on the pre-text intermediate video) ---------


def _blackdetect(video_path: Path) -> list[tuple[float, float]]:
    vf = (
        f"blackdetect=d={config.BLACKDETECT_MIN_DURATION_SEC}:"
        f"pix_th={config.BLACKDETECT_PIXEL_BLACK_TH}:"
        f"pic_th={config.BLACKDETECT_PICTURE_BLACK_RATIO_TH}"
    )
    result = _run(["ffmpeg", "-i", str(video_path), "-vf", vf, "-an", "-f", "null", "-"])
    intervals = []
    for m in re.finditer(
        r"black_start:(?P<start>[\d.]+) black_end:(?P<end>[\d.]+)", result.stderr
    ):
        intervals.append((float(m.group("start")), float(m.group("end"))))
    return intervals


def _freezedetect(
    video_path: Path, start: float | None = None, t: float | None = None
) -> list[float]:
    vf = (
        f"freezedetect=n={config.FREEZEDETECT_NOISE_TOLERANCE_DB}dB:"
        f"d={config.FREEZEDETECT_MIN_FREEZE_DURATION_SEC}"
    )
    cmd = ["ffmpeg"]
    if start is not None:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(video_path)]
    if t is not None:
        cmd += ["-t", str(t)]
    cmd += ["-vf", vf, "-an", "-f", "null", "-"]
    result = _run(cmd)
    starts = [float(m.group(1)) for m in re.finditer(r"freeze_start: ([\d.]+)", result.stderr)]
    return starts


def _source_has_opening_freeze(source_path: Path, segment_start: float) -> bool:
    """Checks whether the *original* source also looks frozen over the same
    opening window the candidate's first segment starts at. Used only to
    tell a genuinely static source (e.g. a slide-heavy podcast) apart from a
    freeze render.py itself introduced.

    The extraction window is intentionally a bit longer than
    CONTENT_QA_OPENING_WINDOW_SEC (it adds FREEZEDETECT_MIN_FREEZE_DURATION_SEC
    of slack) so freezedetect has enough trailing context to confirm a freeze
    that starts near the edge of the window -- mirroring how the
    intermediate-side check scans the whole (unbounded) clip and only
    filters by the opening window afterwards. Any freeze_start reported at
    all within this bounded extraction is treated as "source is also
    static here": since the extraction is already scoped to the opening
    window, this doesn't need to interpret the specific timestamp value
    ffmpeg reports (which depends on how -ss seeking renumbers timestamps).
    """
    window = config.CONTENT_QA_OPENING_WINDOW_SEC + config.FREEZEDETECT_MIN_FREEZE_DURATION_SEC
    starts = _freezedetect(source_path, start=segment_start, t=window)
    return bool(starts)


def video_content_qa(
    intermediate_video_path: Path,
    source_path: Path | None = None,
    source_segment_start: float | None = None,
) -> list[QACheck]:
    checks = []

    black_intervals = _blackdetect(intermediate_video_path)
    opening_black = [
        (s, e) for s, e in black_intervals if s < config.CONTENT_QA_OPENING_WINDOW_SEC
    ]
    checks.append(
        QACheck(
            name="黒画面検出",
            passed=not opening_black,
            critical=True,
            detail=(
                "冒頭付近に黒画面を検出しませんでした"
                if not opening_black
                else f"冒頭付近({opening_black})に黒画面を検出しました"
            ),
        )
    )

    freeze_starts = _freezedetect(intermediate_video_path)
    opening_freeze = [s for s in freeze_starts if s < config.CONTENT_QA_OPENING_WINDOW_SEC]

    source_confirms_freeze = False
    if opening_freeze and source_path is not None and source_segment_start is not None:
        source_confirms_freeze = _source_has_opening_freeze(source_path, source_segment_start)

    if opening_freeze and source_confirms_freeze:
        checks.append(
            QACheck(
                name="静止画/フリーズ検出",
                passed=True,
                critical=True,
                detail=(
                    f"冒頭付近({opening_freeze}秒付近)に静止を検出しましたが、"
                    "元動画の同じ区間も同様に静止しているため、"
                    "コンテンツ由来（スライド等）の静止と判断しPASS扱いとしました"
                ),
            )
        )
    else:
        checks.append(
            QACheck(
                name="静止画/フリーズ検出",
                passed=not opening_freeze,
                critical=True,
                detail=(
                    "冒頭付近に静止画/フリーズを検出しませんでした"
                    if not opening_freeze
                    else f"冒頭付近({opening_freeze}秒付近)に静止画/フリーズを検出しました"
                ),
            )
        )
    return checks


# --- Technical QA (runs on the final mp4) --------------------------------


def technical_qa(final_video_path: Path, expected_duration: float) -> list[QACheck]:
    result = _run(
        [
            "ffprobe", "-v", "error",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(final_video_path),
        ]
    )
    checks = []
    if result.returncode != 0 or not result.stdout.strip():
        checks.append(
            QACheck(
                name="ファイル正常性",
                passed=False,
                critical=True,
                detail=f"ffprobeでファイルを開けませんでした: {result.stderr[-500:]}",
            )
        )
        return checks

    info = json.loads(result.stdout)
    streams = info.get("streams", [])
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    checks.append(
        QACheck(
            name="映像ストリーム",
            passed=bool(video_streams),
            critical=True,
            detail="映像ストリームあり" if video_streams else "映像ストリームが見つかりません",
        )
    )
    checks.append(
        QACheck(
            name="音声ストリーム",
            passed=bool(audio_streams),
            critical=True,
            detail="音声ストリームあり" if audio_streams else "音声ストリームが見つかりません",
        )
    )

    if video_streams:
        w, h = video_streams[0].get("width"), video_streams[0].get("height")
        resolution_ok = (w, h) == (config.VERTICAL_WIDTH, config.VERTICAL_HEIGHT)
        checks.append(
            QACheck(
                name="解像度",
                passed=resolution_ok,
                critical=True,
                detail=f"{w}x{h} (期待値 {config.VERTICAL_WIDTH}x{config.VERTICAL_HEIGHT})",
            )
        )
        codec_name = video_streams[0].get("codec_name", "")
        checks.append(
            QACheck(
                name="コーデック",
                passed=bool(codec_name),
                critical=True,
                detail=f"video codec: {codec_name or '不明'}",
            )
        )

    duration_str = info.get("format", {}).get("duration")
    duration_ok = False
    if duration_str is not None:
        actual = float(duration_str)
        duration_ok = abs(actual - expected_duration) <= 1.0
        detail = f"{actual:.2f}秒 (想定 {expected_duration:.2f}秒)"
    else:
        detail = "durationを取得できませんでした"
    checks.append(QACheck(name="duration整合", passed=duration_ok, critical=True, detail=detail))

    return checks


# --- Audio presence QA (plan fix #6: signal only, not "speech") ----------


def _mean_volume_db(video_path: Path, start: float | None = None, t: float | None = None) -> float | None:
    cmd = ["ffmpeg"]
    if start is not None:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(video_path)]
    if t is not None:
        cmd += ["-t", str(t)]
    cmd += ["-af", "volumedetect", "-vn", "-f", "null", "-"]
    result = _run(cmd)
    m = re.search(r"mean_volume:\s*(-?[\d.]+) dB", result.stderr)
    return float(m.group(1)) if m else None


def audio_presence_qa(final_video_path: Path) -> list[QACheck]:
    checks = []
    start_volume = _mean_volume_db(final_video_path, start=0.0, t=config.CONTENT_QA_OPENING_WINDOW_SEC)
    start_silent = start_volume is None or start_volume < config.SILENCE_MEAN_VOLUME_DB_THRESHOLD
    checks.append(
        QACheck(
            name="無音開始チェック",
            passed=not start_silent,
            critical=True,
            detail=(
                f"開始直後の平均音量 {start_volume:.1f} dB"
                if start_volume is not None
                else "音量を検出できませんでした"
            ),
        )
    )

    overall_volume = _mean_volume_db(final_video_path)
    low_overall = overall_volume is None or overall_volume < config.LOW_VOLUME_MEAN_DB_THRESHOLD
    checks.append(
        QACheck(
            name="全体音量チェック",
            passed=not low_overall,
            critical=False,
            detail=(
                f"全体平均音量 {overall_volume:.1f} dB"
                if overall_volume is not None
                else "音量を検出できませんでした"
            ),
        )
    )
    return checks


# --- Boundary integrity / speech-start alignment (self-consistency) -----
# Both recompute boundary.resolve_candidate deterministically from the
# same (raw candidate, transcript) render.py used, and compare against
# what render.py actually recorded in the manifest. This is not a new
# audio analysis -- it's a check that render.py used boundary.py's output
# correctly (plan fix #1 and #6).


def _recompute_pairs(raw_candidate: RawClipCandidate, transcript: Transcript, manifest: RenderManifest):
    resolved = boundary.resolve_candidate(raw_candidate, transcript, manifest.candidate_id)
    return list(zip(resolved.segments, manifest.segments))


def boundary_integrity_qa(
    raw_candidate: RawClipCandidate, transcript: Transcript, manifest: RenderManifest
) -> QACheck:
    pairs = _recompute_pairs(raw_candidate, transcript, manifest)
    tol = config.SPEECH_ALIGNMENT_TOLERANCE_SEC
    mismatches = [
        i
        for i, (expected, actual) in enumerate(pairs)
        if abs(expected.start - actual.start) > tol or abs(expected.end - actual.end) > tol
    ]
    return QACheck(
        name="編集境界の整合性",
        passed=not mismatches,
        critical=True,
        detail=(
            "すべての区間で境界補正結果と実際の編集点が一致しています"
            if not mismatches
            else f"区間 {mismatches} で境界補正結果と実際の編集点が一致しません"
        ),
    )


def speech_start_alignment_qa(
    raw_candidate: RawClipCandidate, transcript: Transcript, manifest: RenderManifest
) -> QACheck:
    pairs = _recompute_pairs(raw_candidate, transcript, manifest)
    expected0, actual0 = pairs[0]
    tol = config.SPEECH_ALIGNMENT_TOLERANCE_SEC
    diff = abs(expected0.start - actual0.start)
    return QACheck(
        name="発話開始の整合性",
        passed=diff <= tol,
        critical=True,
        detail=(
            f"クリップ開始位置は選定時の発話開始位置と一致しています(差 {diff:.3f}秒)"
            if diff <= tol
            else f"クリップ開始位置が選定時の発話開始位置と{diff:.3f}秒ずれています"
        ),
    )


# --- Thumbnails -----------------------------------------------------------


def extract_thumbnails(final_video_path: Path, duration: float, out_dir: Path, candidate_id: str) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    positions = {
        "start": min(0.5, max(duration - 0.1, 0.0)),
        "middle": duration / 2,
        "end": max(duration - 0.5, 0.0),
    }
    paths = []
    for label, ts in positions.items():
        out_path = out_dir / f"clip_{candidate_id}_thumb_{label}.jpg"
        _run(
            [
                "ffmpeg", "-y",
                "-ss", str(ts),
                "-i", str(final_video_path),
                "-frames:v", "1",
                str(out_path),
            ]
        )
        if out_path.exists():
            paths.append(str(out_path))
    return paths


# --- Orchestration ---------------------------------------------------------


def run_full_qa(
    raw_candidate: RawClipCandidate,
    transcript: Transcript,
    manifest: RenderManifest,
    source_path: Path,
) -> QAReport:
    intermediate_path = Path(manifest.intermediate_video_path)
    final_path = Path(manifest.final_video_path)

    checks: list[QACheck] = []
    checks += video_content_qa(
        intermediate_path,
        source_path=Path(source_path),
        source_segment_start=manifest.segments[0].start,
    )
    checks += technical_qa(final_path, manifest.total_duration)
    checks += audio_presence_qa(final_path)
    checks.append(speech_start_alignment_qa(raw_candidate, transcript, manifest))
    checks.append(boundary_integrity_qa(raw_candidate, transcript, manifest))

    thumbnails = extract_thumbnails(
        final_path, manifest.total_duration, final_path.parent, manifest.candidate_id
    )
    return QAReport(checks=checks, thumbnails=thumbnails)
