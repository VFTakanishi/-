"""End-to-end wiring test for the FastAPI app via TestClient (plan:
"web.pyのAPIをFastAPIのTestClientで疎通確認"). The expensive pipeline calls
(Whisper, Claude, ffmpeg) are mocked so this only verifies the HTTP wiring
/ job orchestration, not the pipeline internals (those are covered by the
other test modules). The multipart upload itself goes through the real
ingest.ingest_uploaded_file (only transcribe/clip_selector/render/qa are
mocked), so this also exercises the real upload -> video_id -> source_path
wiring end to end.
"""
import io
import time

import pytest
from fastapi.testclient import TestClient

from podcast_clipper import boundary, cache, clip_selector, config, ingest, qa, transcribe, web
from podcast_clipper.models import (
    RawClipCandidate,
    RawUsedSegment,
    RenderManifest,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)

_FAKE_VIDEO_BYTES = b"fake mp4 bytes for upload wiring test"


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
        hook_text="フック", opening_hook_strength=85,
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
    monkeypatch.setattr(ingest, "_probe_duration", lambda path: 6.0)
    # _run_render now defensively re-applies clip_selector.finalize_candidates
    # to whatever it reads back from cache.load_stage2 (see web.py), which
    # re-enforces the real hard duration bounds -- the fake fixture
    # candidates below are only a few seconds long, well under the real
    # default DURATION_HARD_MIN_SEC, so bounds are relaxed here exactly as
    # tests/test_clip_selector.py does for the same reason.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)

    captured = {}

    def fake_transcribe(path, vid, force_refresh=False):
        captured["source_path"] = path
        captured["video_id"] = vid
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

    resp = client.post(
        "/api/analyze",
        files={"file": ("テスト番組.mp4", io.BytesIO(_FAKE_VIDEO_BYTES), "video/mp4")},
    )
    assert resp.status_code == 200
    analyze_job_id = resp.json()["job_id"]

    job = _wait_for_status(client, f"/api/jobs/{analyze_job_id}", {"completed", "failed"})
    assert job["status"] == "completed", job.get("error")
    candidates = job["result"]["candidates"]
    assert len(candidates) == 3
    assert all(1 <= len(c["segments"]) <= 3 for c in candidates)
    # regression: total_duration is a ClipCandidate @property, so it must be
    # added explicitly in _serialize_candidate -- dataclasses.asdict() alone
    # drops it, which crashed the frontend's `c.total_duration.toFixed(1)`.
    for c in candidates:
        assert isinstance(c["total_duration"], (int, float))
        assert c["total_duration"] == pytest.approx(
            sum(s["end"] - s["start"] for s in c["segments"])
        )
        # the "冒頭の実音声" UI feature reads segments[0].text directly off
        # this API response (real transcript text, never AI-generated) --
        # confirm the data it needs is actually present.
        assert c["segments"][0]["role"] == "hook"
        assert c["segments"][0]["text"]

    # job.input carries video_id/video_title/source_path, and transcribe was
    # handed the correct local source_path -- no YouTube download involved.
    assert job["input"]["video_id"]
    assert job["input"]["video_title"] == "テスト番組.mp4"
    assert job["input"]["source_path"]
    assert str(captured["source_path"]) == job["input"]["source_path"]
    assert captured["video_id"] == job["input"]["video_id"]

    # --- render a candidate, with render.render_candidate/qa.run_full_qa mocked ---
    fake_final = tmp_path / "final.mp4"
    fake_final.write_bytes(b"fake mp4 bytes")

    def fake_render_candidate(source_path, candidate, video_id_):
        return RenderManifest(
            video_id=video_id_, candidate_id=candidate.id, segments=candidate.segments,
            hook_text=candidate.hook_text, watermark_text=config.WATERMARK_TEXT,
            total_duration=candidate.total_duration,
            intermediate_video_path=str(tmp_path / "mid.mp4"),
            final_video_path=str(fake_final),
        )

    monkeypatch.setattr("podcast_clipper.web.render.render_candidate", fake_render_candidate)
    monkeypatch.setattr(
        "podcast_clipper.web.qa.run_full_qa",
        lambda raw, transcript, manifest, source_path: qa.QAReport(checks=[], thumbnails=[]),
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
    monkeypatch.setattr(ingest, "_probe_duration", lambda path: 6.0)
    # See the comment in test_analyze_then_render_then_download_flow above:
    # _run_render's defensive finalize_candidates re-check needs the hard
    # duration bounds relaxed for these short fixture candidates.
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)

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
    resp = client.post(
        "/api/analyze",
        files={"file": ("テスト番組2.mp4", io.BytesIO(_FAKE_VIDEO_BYTES), "video/mp4")},
    )
    analyze_job_id = resp.json()["job_id"]
    _wait_for_status(client, f"/api/jobs/{analyze_job_id}", {"completed", "failed"})

    fake_final = tmp_path / "final2.mp4"
    fake_final.write_bytes(b"should not be downloadable")

    def fake_render_candidate(source_path, candidate, video_id_):
        return RenderManifest(
            video_id=video_id_, candidate_id=candidate.id, segments=candidate.segments,
            hook_text=candidate.hook_text, watermark_text=config.WATERMARK_TEXT,
            total_duration=candidate.total_duration,
            intermediate_video_path=str(tmp_path / "mid2.mp4"),
            final_video_path=str(fake_final),
        )

    critical_fail_check = qa.QACheck(
        name="黒画面検出", passed=False, critical=True, detail="冒頭が黒画面でした"
    )
    monkeypatch.setattr("podcast_clipper.web.render.render_candidate", fake_render_candidate)
    monkeypatch.setattr(
        "podcast_clipper.web.qa.run_full_qa",
        lambda raw, transcript, manifest, source_path: qa.QAReport(
            checks=[critical_fail_check], thumbnails=[]
        ),
    )

    resp = client.post(f"/api/jobs/{analyze_job_id}/render", json={"candidate_id": "c1"})
    render_id = resp.json()["render_id"]
    render_job = _wait_for_status(
        client, f"/api/jobs/{analyze_job_id}/render/{render_id}", {"completed", "failed"}
    )
    assert render_job["result"]["download_allowed"] is False

    dl = client.get(f"/api/jobs/{analyze_job_id}/render/{render_id}/download")
    assert dl.status_code == 403


def _fake_transcript_extendable(video_id):
    # segment 1 has no terminal punctuation and a short (0.3s) gap into
    # segment 2, which does -- extend_to_natural_ending must pull segment
    # 2 in, exactly the shape of the real "candidate 2" mid-utterance
    # incident this regression test is pinned against.
    return Transcript(
        video_id=video_id,
        language="ja",
        segments=[
            TranscriptSegment(id=0, start=0.0, end=2.0, text="質問です",
                               words=[TranscriptWord(0.0, 2.0, "質問です")]),
            TranscriptSegment(id=1, start=2.3, end=5.3, text="それについてはこう考えられます",
                               words=[TranscriptWord(2.3, 5.3, "それについてはこう考えられます")]),
            TranscriptSegment(id=2, start=5.6, end=7.6, text="というのが結論です。",
                               words=[TranscriptWord(5.6, 7.6, "というのが結論です。")]),
        ],
    )


def _stale_raw_candidate():
    return RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0),
            RawUsedSegment(role="answer", start_segment_id=1, end_segment_id=1),
        ],
        hook_text="フック", opening_hook_strength=85,
        title="タイトル", description="説明", score=88, reasoning="理由", caveats="",
    )


def test_run_render_uses_finalized_candidate_not_stale_cache(monkeypatch, tmp_path):
    """Regression test for the real-machine incident: web._run_render used
    to read cache.load_stage2 directly and render it as-is, bypassing the
    same extend_to_natural_ending/duration correction the analyze UI
    already applied -- so a candidate the UI showed extended to a natural
    ending could render from the stale, shorter, mid-utterance-ending
    cache entry instead. _run_render must now apply
    clip_selector.finalize_candidates before indexing, so render always
    uses the identical corrected candidate the UI displayed.
    """
    monkeypatch.setattr(ingest, "_probe_duration", lambda path: 8.0)
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)

    def fake_transcribe(path, vid, force_refresh=False):
        t = _fake_transcript_extendable(vid)
        cache.save_transcript(t)
        return t

    def fake_select_candidates(transcript, title, force_refresh=False):
        # Simulate the exact bug scenario: the on-disk Stage2 cache holds
        # the stale, un-extended candidate (end_segment_id=1) -- what a
        # pre-fix cache.save_stage2(finalists) call before
        # finalize_candidates would have written -- while the value
        # actually shown to the UI (and returned here, as the real
        # select_candidates now guarantees) is the finalized/extended one.
        stale = _stale_raw_candidate()
        cache.save_stage2(transcript.video_id, [stale] * 3)
        extended = clip_selector.extend_to_natural_ending(stale, transcript)
        assert extended.segments[-1].end_segment_id == 2  # sanity: extension actually happens
        return [extended] * 3

    monkeypatch.setattr(transcribe, "transcribe_video", fake_transcribe)
    monkeypatch.setattr(
        "podcast_clipper.web.clip_selector.select_candidates", fake_select_candidates
    )

    client = TestClient(web.app)
    resp = client.post(
        "/api/analyze",
        files={"file": ("テスト番組3.mp4", io.BytesIO(_FAKE_VIDEO_BYTES), "video/mp4")},
    )
    analyze_job_id = resp.json()["job_id"]
    job = _wait_for_status(client, f"/api/jobs/{analyze_job_id}", {"completed", "failed"})
    assert job["status"] == "completed", job.get("error")

    ui_candidate = job["result"]["candidates"][0]
    ui_end = ui_candidate["segments"][-1]["end"]

    captured_manifest = {}
    fake_final = tmp_path / "final3.mp4"
    fake_final.write_bytes(b"fake mp4 bytes")

    def fake_render_candidate(source_path, candidate, video_id_):
        captured_manifest["candidate"] = candidate
        return RenderManifest(
            video_id=video_id_, candidate_id=candidate.id, segments=candidate.segments,
            hook_text=candidate.hook_text, watermark_text=config.WATERMARK_TEXT,
            total_duration=candidate.total_duration,
            intermediate_video_path=str(tmp_path / "mid3.mp4"),
            final_video_path=str(fake_final),
        )

    monkeypatch.setattr("podcast_clipper.web.render.render_candidate", fake_render_candidate)
    monkeypatch.setattr(
        "podcast_clipper.web.qa.run_full_qa",
        lambda raw, transcript, manifest, source_path: qa.QAReport(checks=[], thumbnails=[]),
    )

    resp = client.post(f"/api/jobs/{analyze_job_id}/render", json={"candidate_id": "c1"})
    render_id = resp.json()["render_id"]
    render_job = _wait_for_status(
        client, f"/api/jobs/{analyze_job_id}/render/{render_id}", {"completed", "failed"}
    )
    assert render_job["status"] == "completed", render_job.get("error")

    rendered_end = captured_manifest["candidate"].segments[-1].end
    # The render must use the same (extended, end_segment_id=2) candidate
    # the UI showed -- never the stale end_segment_id=1 straight from the
    # raw, un-finalized cache entry.
    assert rendered_end == pytest.approx(ui_end)
    assert rendered_end > 6.0  # segment 1 alone would end well before this


def test_refresh_candidates_endpoint_reuses_analyze_job_input_without_rerunning_whisper(monkeypatch):
    """The candidate-only refresh action must work off a *failed* analyze
    job (the exact "insufficient eligible candidates" scenario), reusing
    only job.input (video_id/video_title) -- never job.result, which is
    None on a failed job -- and must never re-run transcribe_video
    (Whisper), only clip_selector.refresh_candidates_only.
    """
    monkeypatch.setattr(ingest, "_probe_duration", lambda path: 6.0)

    def fake_transcribe(path, vid, force_refresh=False):
        t = _fake_transcript(vid)
        cache.save_transcript(t)
        return t

    def fake_select_candidates_failing(transcript, title, force_refresh=False):
        raise RuntimeError("有効な3候補を確保できませんでした（テスト用）")

    monkeypatch.setattr(transcribe, "transcribe_video", fake_transcribe)
    monkeypatch.setattr(
        "podcast_clipper.web.clip_selector.select_candidates", fake_select_candidates_failing
    )

    client = TestClient(web.app)
    resp = client.post(
        "/api/analyze",
        files={"file": ("テスト番組4.mp4", io.BytesIO(_FAKE_VIDEO_BYTES), "video/mp4")},
    )
    analyze_job_id = resp.json()["job_id"]
    job = _wait_for_status(client, f"/api/jobs/{analyze_job_id}", {"completed", "failed"})
    assert job["status"] == "failed"
    assert job["result"] is None

    def fail_if_called(*a, **k):
        raise AssertionError("Whisper must not be re-run for candidate-only refresh")

    monkeypatch.setattr(transcribe, "transcribe_video", fail_if_called)

    def fake_refresh_candidates_only(transcript, title):
        assert transcript.video_id == job["input"]["video_id"]
        assert title == job["input"]["video_title"]
        return [_fake_raw_candidate() for _ in range(3)]

    monkeypatch.setattr(
        "podcast_clipper.web.clip_selector.refresh_candidates_only", fake_refresh_candidates_only
    )

    resp = client.post(f"/api/jobs/{analyze_job_id}/refresh-candidates")
    assert resp.status_code == 200
    refresh_job_id = resp.json()["job_id"]

    refresh_job = _wait_for_status(client, f"/api/jobs/{refresh_job_id}", {"completed", "failed"})
    assert refresh_job["status"] == "completed", refresh_job.get("error")
    assert len(refresh_job["result"]["candidates"]) == 3
    assert refresh_job["result"]["video_id"] == job["input"]["video_id"]


def test_refresh_stage1_endpoint_reuses_job_input_without_rerunning_whisper(monkeypatch):
    """The mid-cost Stage1 re-analysis action must work off a *failed*
    refresh-candidates job (the exact "cached Stage1 candidates no longer
    meet the quality bar" scenario from the real-machine incident),
    reusing only job.input (video_id/video_title) -- never job.result --
    and must never re-run transcribe_video (Whisper), only
    clip_selector.refresh_stage1_and_candidates.
    """
    monkeypatch.setattr(ingest, "_probe_duration", lambda path: 6.0)

    def fake_transcribe(path, vid, force_refresh=False):
        t = _fake_transcript(vid)
        cache.save_transcript(t)
        return t

    def fake_select_candidates_failing(transcript, title, force_refresh=False):
        raise RuntimeError("有効な3候補を確保できませんでした（テスト用）")

    monkeypatch.setattr(transcribe, "transcribe_video", fake_transcribe)
    monkeypatch.setattr(
        "podcast_clipper.web.clip_selector.select_candidates", fake_select_candidates_failing
    )

    client = TestClient(web.app)
    resp = client.post(
        "/api/analyze",
        files={"file": ("テスト番組5.mp4", io.BytesIO(_FAKE_VIDEO_BYTES), "video/mp4")},
    )
    analyze_job_id = resp.json()["job_id"]
    job = _wait_for_status(client, f"/api/jobs/{analyze_job_id}", {"completed", "failed"})
    assert job["status"] == "failed"

    def fake_refresh_candidates_only_failing(transcript, title):
        raise RuntimeError("保存済みStage1候補のうちローカル品質フィルタを通過したのは2件です（テスト用）")

    monkeypatch.setattr(
        "podcast_clipper.web.clip_selector.refresh_candidates_only", fake_refresh_candidates_only_failing
    )

    resp = client.post(f"/api/jobs/{analyze_job_id}/refresh-candidates")
    refresh_job_id = resp.json()["job_id"]
    refresh_job = _wait_for_status(client, f"/api/jobs/{refresh_job_id}", {"completed", "failed"})
    assert refresh_job["status"] == "failed"

    def fail_if_called(*a, **k):
        raise AssertionError("Whisper must not be re-run for the Stage1 re-analysis action")

    monkeypatch.setattr(transcribe, "transcribe_video", fail_if_called)

    def fake_refresh_stage1_and_candidates(transcript, title):
        assert transcript.video_id == refresh_job["input"]["video_id"]
        assert title == refresh_job["input"]["video_title"]
        return [_fake_raw_candidate() for _ in range(3)]

    monkeypatch.setattr(
        "podcast_clipper.web.clip_selector.refresh_stage1_and_candidates",
        fake_refresh_stage1_and_candidates,
    )

    # The Stage1 re-analysis button can be triggered off any prior
    # analyze-style job in this chain -- here off the refresh_candidates
    # job that itself just failed, matching the real UI flow.
    resp = client.post(f"/api/jobs/{refresh_job_id}/refresh-stage1")
    assert resp.status_code == 200
    stage1_job_id = resp.json()["job_id"]

    stage1_job = _wait_for_status(client, f"/api/jobs/{stage1_job_id}", {"completed", "failed"})
    assert stage1_job["status"] == "completed", stage1_job.get("error")
    assert len(stage1_job["result"]["candidates"]) == 3
    assert stage1_job["result"]["video_id"] == refresh_job["input"]["video_id"]


def test_serialize_candidate_includes_total_duration():
    """Low-level pin for the exact bug reported in real-machine E2E: the
    frontend called c.total_duration.toFixed(1), but total_duration is a
    ClipCandidate @property, so plain dataclasses.asdict(c) silently dropped
    it from the JSON, leaving c.total_duration undefined in the browser.
    """
    transcript = _fake_transcript("webtestvid3")
    raw_candidate = _fake_raw_candidate()
    candidate = boundary.resolve_candidate(raw_candidate, transcript, candidate_id="c1")

    data = web._serialize_candidate(candidate)

    assert "total_duration" in data
    assert data["total_duration"] == candidate.total_duration
    assert isinstance(data["total_duration"], (int, float))
