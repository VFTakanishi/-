"""End-to-end wiring test for the FastAPI app via TestClient (plan:
"web.pyのAPIをFastAPIのTestClientで疎通確認"). The expensive pipeline calls
(YouTube download, Whisper, Claude, ffmpeg) are mocked so this only
verifies the HTTP wiring / job orchestration, not the pipeline internals
(those are covered by the other test modules).
"""
import time

from fastapi.testclient import TestClient

from podcast_clipper import cache, config, download, qa, transcribe, web
from podcast_clipper.models import (
    RawClipCandidate,
    RawUsedSegment,
    RenderManifest,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)


def _fake_transcript(video_id):
    return Transcript(
        video_id=video_id,
        language="ja",
        segments=[
            TranscriptSegment(id=0, start=0.0, end=2.0, text="質問です",
                               words=[TranscriptWord(0.0, 2.0, "質問です")]),
            TranscriptSegment(id=1, start=3.0, end=6.0, text="答えです",
                               words=[TranscriptWord(3.0, 6.0, "答えです")]),
        ],
    )


def _fake_raw_candidate():
    return RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0),
            RawUsedSegment(role="answer", start_segment_id=1, end_segment_id=1),
        ],
        hook_text="フック", cta_end_text="本編は関連動画から",
        title="タイトル", description="説明", score=88, reasoning="理由", caveats="",
    )


def _wait_for_status(client, url, target_statuses, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(url)
        assert resp.status_code == 200
        body = resp.json()
        if body["status"] in target_statuses:
            return body
        time.sleep(0.05)
    raise TimeoutError(f"job at {url} did not reach {target_statuses} in time")


def test_analyze_then_render_then_download_flow(monkeypatch, tmp_path):
    video_id = "webtestvid"

    monkeypatch.setattr(
        download, "download_video",
        lambda url, out_root, force_refresh=False: download.VideoInfo(
            video_id=video_id, title="テスト番組", duration=6.0, path=tmp_path / "source.mp4"
        ),
    )
    def fake_transcribe(path, vid, force_refresh=False):
        t = _fake_transcript(vid)
        cache.save_transcript(t)
        return t

    def fake_select_candidates(transcript, title, force_refresh=False):
        candidates = [_fake_raw_candidate() for _ in range(3)]
        cache.save_stage2(transcript.video_id, candidates)
        return candidates

    monkeypatch.setattr(transcribe, "transcribe_video", fake_transcribe)
    monkeypatch.setattr(
        "podcast_clipper.web.clip_selector.select_candidates", fake_select_candidates
    )

    client = TestClient(web.app)

    resp = client.post("/api/analyze", json={"url": "https://youtu.be/xxxx"})
    assert resp.status_code == 200
    analyze_job_id = resp.json()["job_id"]

    job = _wait_for_status(client, f"/api/jobs/{analyze_job_id}", {"completed", "failed"})
    assert job["status"] == "completed", job.get("error")
    candidates = job["result"]["candidates"]
    assert len(candidates) == 3
    assert all(1 <= len(c["segments"]) <= 3 for c in candidates)

    # --- render a candidate, with render.render_candidate/qa.run_full_qa mocked ---
    fake_final = tmp_path / "final.mp4"
    fake_final.write_bytes(b"fake mp4 bytes")

    def fake_render_candidate(source_path, candidate, video_id_):
        return RenderManifest(
            video_id=video_id_, candidate_id=candidate.id, segments=candidate.segments,
            hook_text=candidate.hook_text, watermark_text=config.WATERMARK_TEXT,
            cta_end_text=candidate.cta_end_text, total_duration=candidate.total_duration,
            intermediate_video_path=str(tmp_path / "mid.mp4"),
            final_video_path=str(fake_final),
        )

    monkeypatch.setattr("podcast_clipper.web.render.render_candidate", fake_render_candidate)
    monkeypatch.setattr(
        "podcast_clipper.web.qa.run_full_qa",
        lambda raw, transcript, manifest: qa.QAReport(checks=[], thumbnails=[]),
    )

    resp = client.post(f"/api/jobs/{analyze_job_id}/render", json={"candidate_id": "c1"})
    assert resp.status_code == 200
    render_id = resp.json()["render_id"]

    render_job = _wait_for_status(
        client, f"/api/jobs/{analyze_job_id}/render/{render_id}", {"completed", "failed"}
    )
    assert render_job["status"] == "completed", render_job.get("error")
    assert render_job["result"]["download_allowed"] is True
    assert render_job["result"]["related_video_instructions"] == config.RELATED_VIDEO_INSTRUCTIONS

    dl = client.get(f"/api/jobs/{analyze_job_id}/render/{render_id}/download")
    assert dl.status_code == 200
    assert dl.content == b"fake mp4 bytes"
    # plan fix #4: download endpoint returns the mp4 file only (video/mp4
    # binary body), never a JSON payload carrying the related-video guidance.
    assert dl.headers["content-type"] == "video/mp4"


def test_download_blocked_when_qa_has_critical_failure(monkeypatch, tmp_path):
    video_id = "webtestvid2"
    monkeypatch.setattr(
        download, "download_video",
        lambda url, out_root, force_refresh=False: download.VideoInfo(
            video_id=video_id, title="テスト番組2", duration=6.0, path=tmp_path / "source.mp4"
        ),
    )
    def fake_transcribe(path, vid, force_refresh=False):
        t = _fake_transcript(vid)
        cache.save_transcript(t)
        return t

    def fake_select_candidates(transcript, title, force_refresh=False):
        candidates = [_fake_raw_candidate() for _ in range(3)]
        cache.save_stage2(transcript.video_id, candidates)
        return candidates

    monkeypatch.setattr(transcribe, "transcribe_video", fake_transcribe)
    monkeypatch.setattr(
        "podcast_clipper.web.clip_selector.select_candidates", fake_select_candidates
    )

    client = TestClient(web.app)
    resp = client.post("/api/analyze", json={"url": "https://youtu.be/yyyy"})
    analyze_job_id = resp.json()["job_id"]
    _wait_for_status(client, f"/api/jobs/{analyze_job_id}", {"completed", "failed"})

    fake_final = tmp_path / "final2.mp4"
    fake_final.write_bytes(b"should not be downloadable")

    def fake_render_candidate(source_path, candidate, video_id_):
        return RenderManifest(
            video_id=video_id_, candidate_id=candidate.id, segments=candidate.segments,
            hook_text=candidate.hook_text, watermark_text=config.WATERMARK_TEXT,
            cta_end_text=candidate.cta_end_text, total_duration=candidate.total_duration,
            intermediate_video_path=str(tmp_path / "mid2.mp4"),
            final_video_path=str(fake_final),
        )

    critical_fail_check = qa.QACheck(
        name="黒画面検出", passed=False, critical=True, detail="冒頭が黒画面でした"
    )
    monkeypatch.setattr("podcast_clipper.web.render.render_candidate", fake_render_candidate)
    monkeypatch.setattr(
        "podcast_clipper.web.qa.run_full_qa",
        lambda raw, transcript, manifest: qa.QAReport(checks=[critical_fail_check], thumbnails=[]),
    )

    resp = client.post(f"/api/jobs/{analyze_job_id}/render", json={"candidate_id": "c1"})
    render_id = resp.json()["render_id"]
    render_job = _wait_for_status(
        client, f"/api/jobs/{analyze_job_id}/render/{render_id}", {"completed", "failed"}
    )
    assert render_job["result"]["download_allowed"] is False

    dl = client.get(f"/api/jobs/{analyze_job_id}/render/{render_id}/download")
    assert dl.status_code == 403
