"""FastAPI backend for the browser UI (absolute condition #3).

Flow (absolute condition #2): the user uploads a local video file via
POST /api/analyze (multipart), which is persisted to disk synchronously
via ingest.py, then transcribe -> Stage1/Stage2 selection runs as a
background job; the client polls GET /api/jobs/{id} until it sees
exactly 3 candidates (absolute condition #1); the user picks one and
POST /api/jobs/{id}/render starts rendering+QA for that candidate only;
the client polls GET /api/jobs/{id}/render/{render_id}; GET .../download
returns the mp4 (plan fix #4: nothing else) once QA allows it.
"""
from __future__ import annotations

import hashlib
import hmac
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import boundary, cache, clip_selector, config, ingest, jobs, qa, render, transcribe
from .models import ClipCandidate

config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    jobs.recover_interrupted_jobs()
    yield


app = FastAPI(title="Podcast Clipper", lifespan=_lifespan)


# --- cloud deployment: simple password gate (see config.TOOL_PASSWORD) --
# Deliberately not a real auth system (no OAuth, no user DB, no server-side
# session store) -- a single shared password, gating every route except
# /health (Railway's health checker never has a cookie) and /auth/login
# (must be reachable *before* authenticating). The cookie holds only an
# HMAC-SHA256 digest of the password (never the password itself), so it's
# both unforgeable without knowing TOOL_PASSWORD and safe to store
# stateless-ly with no server-side session table. When config.TOOL_PASSWORD
# is unset (the local/default case), _is_authenticated always returns True
# and the gate is a complete no-op -- existing local usage (and every
# existing test using TestClient(web.app) with no TOOL_PASSWORD set) is
# unaffected.
_AUTH_COOKIE_NAME = "pc_auth"
_AUTH_PAYLOAD = b"podcast-clipper-authenticated"
_AUTH_EXEMPT_PATHS = {"/health", "/auth/login"}

_LOGIN_PAGE_HTML = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<title>Podcast Clipper - ログイン</title>
<style>
body{{font-family:sans-serif;display:flex;align-items:center;justify-content:center;
height:100vh;margin:0;background:#111;color:#eee;}}
form{{background:#1c1c1c;padding:2rem 2.5rem;border-radius:10px;text-align:center;}}
input[type=password]{{padding:.6rem;font-size:1rem;border-radius:4px;border:1px solid #444;
background:#222;color:#eee;}}
button{{margin-left:.5rem;padding:.6rem 1.2rem;font-size:1rem;border-radius:4px;border:none;
background:#3a9a5c;color:#fff;cursor:pointer;}}
.error{{color:#f66;margin-top:.7rem;font-size:.9rem;}}
</style></head>
<body>
<form method="post" action="/auth/login">
<div style="margin-bottom:1rem;">Podcast Clipper</div>
<input type="password" name="password" placeholder="パスワード" autofocus required>
<button type="submit">入る</button>
{error}
</form>
</body></html>"""


def _expected_auth_token() -> str | None:
    if not config.TOOL_PASSWORD:
        return None
    return hmac.new(config.TOOL_PASSWORD.encode("utf-8"), _AUTH_PAYLOAD, hashlib.sha256).hexdigest()


def _is_authenticated(request: Request) -> bool:
    expected = _expected_auth_token()
    if expected is None:
        return True
    cookie = request.cookies.get(_AUTH_COOKIE_NAME)
    return bool(cookie) and hmac.compare_digest(cookie, expected)


def _password_matches(candidate: str) -> bool:
    if not config.TOOL_PASSWORD:
        return False
    return hmac.compare_digest(candidate.encode("utf-8"), config.TOOL_PASSWORD.encode("utf-8"))


@app.middleware("http")
async def _tool_password_gate(request: Request, call_next):
    if request.url.path in _AUTH_EXEMPT_PATHS or _is_authenticated(request):
        return await call_next(request)
    return HTMLResponse(_LOGIN_PAGE_HTML.format(error=""), status_code=401)


@app.post("/auth/login")
def auth_login(password: str = Form(...)) -> Response:
    if not _password_matches(password):
        return HTMLResponse(
            _LOGIN_PAGE_HTML.format(error='<div class="error">パスワードが違います</div>'),
            status_code=401,
        )
    resp = RedirectResponse(url="/", status_code=303)
    resp.set_cookie(
        _AUTH_COOKIE_NAME, _expected_auth_token(),
        httponly=True, samesite="lax", max_age=60 * 60 * 24 * 30,
    )
    return resp


@app.get("/health")
def health() -> dict[str, str]:
    """Railway health check target. Deliberately makes no Anthropic API
    call and touches no cache/job state -- just confirms the process is up.
    """
    return {"status": "ok"}


class RenderRequest(BaseModel):
    candidate_id: str


def _serialize_candidate(c: ClipCandidate) -> dict[str, Any]:
    # total_duration is a @property (derived from segments), so
    # dataclasses.asdict() -- which only walks declared fields -- omits it;
    # it must be added explicitly or the frontend never receives it.
    data = asdict(c)
    data["total_duration"] = c.total_duration
    return data


def _to_media_url(path: str) -> str:
    rel = Path(path).resolve().relative_to(config.OUTPUT_DIR.resolve())
    return f"/media/{rel.as_posix()}"


def _serialize_qa(report: qa.QAReport) -> dict[str, Any]:
    return {
        "checks": [asdict(c) for c in report.checks],
        "thumbnails": [_to_media_url(p) for p in report.thumbnails],
        "download_allowed": report.download_allowed,
    }


def _run_analyze(job: jobs.Job) -> dict[str, Any]:
    video_id = job.input["video_id"]
    video_title = job.input["video_title"]
    source_path = Path(job.input["source_path"])
    force_refresh = job.input.get("force_refresh", False)

    transcript = transcribe.transcribe_video(source_path, video_id, force_refresh=force_refresh)
    raw_candidates = clip_selector.select_candidates(
        transcript, video_title, force_refresh=force_refresh
    )
    resolved = [
        boundary.resolve_candidate(rc, transcript, candidate_id=f"c{i + 1}")
        for i, rc in enumerate(raw_candidates)
    ]

    return {
        "video_id": video_id,
        "video_title": video_title,
        "candidates": [_serialize_candidate(c) for c in resolved],
    }


def _run_render(job: jobs.Job) -> dict[str, Any]:
    analyze_job_id = job.input["analyze_job_id"]
    candidate_id = job.input["candidate_id"]

    analyze_job = jobs.get_job(analyze_job_id)
    if analyze_job is None or analyze_job.status != "completed" or analyze_job.result is None:
        raise RuntimeError("元の解析ジョブが見つからないか、まだ完了していません")

    video_id = analyze_job.result["video_id"]
    source_path = Path(analyze_job.input["source_path"])

    transcript = cache.load_transcript(video_id)
    raw_candidates = cache.load_stage2(video_id)
    if transcript is None or raw_candidates is None:
        raise RuntimeError("文字起こし/候補のキャッシュが見つかりません。解析をやり直してください")

    # Never index into the raw cache read directly: cache.load_stage2 can
    # return a pre-correction result (an older cache file written before
    # this normalization existed, or any other path that saved candidates
    # without going through clip_selector.select_candidates/
    # refresh_candidates_only). Re-applying finalize_candidates here makes
    # this self-healing and guarantees render always uses the identical
    # ending-corrected/duration-validated/opening-trimmed candidate the UI
    # showed -- never a stale, mid-utterance-ending raw one.
    raw_candidates = clip_selector.finalize_candidates(raw_candidates, transcript)

    try:
        idx = int(candidate_id.lstrip("c")) - 1
        raw_candidate = raw_candidates[idx]
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"不正な candidate_id: {candidate_id}") from exc

    resolved_candidate = boundary.resolve_candidate(raw_candidate, transcript, candidate_id)

    manifest = render.render_candidate(source_path, resolved_candidate, video_id)
    qa_report = qa.run_full_qa(raw_candidate, transcript, manifest, source_path)

    return {
        "video_id": video_id,
        "candidate_id": candidate_id,
        "qa": _serialize_qa(qa_report),
        "download_allowed": qa_report.download_allowed,
        "final_video_path": manifest.final_video_path,
        "related_video_instructions": config.RELATED_VIDEO_INSTRUCTIONS,
    }


def _run_refresh_candidates(job: jobs.Job) -> dict[str, Any]:
    """Low-cost re-selection: reuses the Transcript and Stage1 chunk cache
    from a prior analyze run (never re-runs Whisper, never calls the
    Stage1 API) and re-derives Stage2 candidates via
    clip_selector.refresh_candidates_only -- at most one Anthropic API
    call (Stage2 ranking), only if enough valid Stage1 candidates already
    exist in cache. Returns the identical result shape _run_analyze does
    (video_id/video_title/candidates) so the frontend can reuse its
    existing analyze-result rendering path unchanged.
    """
    video_id = job.input["video_id"]
    video_title = job.input["video_title"]

    transcript = cache.load_transcript(video_id)
    if transcript is None:
        raise RuntimeError("文字起こしのキャッシュが見つかりません。最初から解析をやり直してください")

    raw_candidates = clip_selector.refresh_candidates_only(transcript, video_title)
    resolved = [
        boundary.resolve_candidate(rc, transcript, candidate_id=f"c{i + 1}")
        for i, rc in enumerate(raw_candidates)
    ]

    return {
        "video_id": video_id,
        "video_title": video_title,
        "candidates": [_serialize_candidate(c) for c in resolved],
    }


def _run_refresh_stage1(job: jobs.Job) -> dict[str, Any]:
    """Mid-cost re-analysis: reuses the cached Transcript (never calls
    transcribe.transcribe_video / Whisper) but regenerates Stage1 for
    every chunk via clip_selector.refresh_stage1_and_candidates -- for
    when the cached Stage1 candidates themselves no longer clear the
    current local quality filter, which refresh_candidates_only (reusing
    Stage1 cache as-is) can't fix. Same result shape as
    _run_analyze/_run_refresh_candidates.
    """
    video_id = job.input["video_id"]
    video_title = job.input["video_title"]

    transcript = cache.load_transcript(video_id)
    if transcript is None:
        raise RuntimeError("文字起こしのキャッシュが見つかりません。最初から解析をやり直してください")

    raw_candidates = clip_selector.refresh_stage1_and_candidates(transcript, video_title)
    resolved = [
        boundary.resolve_candidate(rc, transcript, candidate_id=f"c{i + 1}")
        for i, rc in enumerate(raw_candidates)
    ]

    return {
        "video_id": video_id,
        "video_title": video_title,
        "candidates": [_serialize_candidate(c) for c in resolved],
    }


@app.post("/api/analyze")
def analyze(file: UploadFile = File(...), force_refresh: bool = Form(False)) -> dict[str, Any]:
    # Persisted to disk synchronously, before the background job is created:
    # UploadFile is request-scoped and must never be handed to a background
    # thread (its underlying file may already be closed by the time a
    # background job would get around to reading it).
    try:
        result = ingest.ingest_uploaded_file(file.file, file.filename)
    except ingest.IngestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job = jobs.create_job(
        "analyze",
        {
            "video_id": result.video_id,
            "video_title": result.title,
            "source_path": str(result.path),
            "force_refresh": force_refresh,
        },
    )
    jobs.run_async(job, _run_analyze, running_status="analyzing")
    return {"job_id": job.id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    resp = asdict(job)
    if job.status == config.JOB_STATE_INTERRUPTED:
        video_id = job.input.get("video_id")
        resp["resumable"] = bool(video_id and cache.load_transcript(video_id) is not None)
    return resp


@app.post("/api/jobs/{job_id}/render")
def start_render(job_id: str, req: RenderRequest) -> dict[str, Any]:
    analyze_job = jobs.get_job(job_id)
    if analyze_job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if analyze_job.status != "completed":
        raise HTTPException(status_code=400, detail="解析がまだ完了していません")

    render_job = jobs.create_job(
        "render", {"analyze_job_id": job_id, "candidate_id": req.candidate_id}
    )
    jobs.run_async(render_job, _run_render, running_status="rendering")
    return {"render_id": render_job.id}


@app.post("/api/jobs/{job_id}/refresh-candidates")
def refresh_candidates(job_id: str) -> dict[str, Any]:
    """Kicks off the low-cost candidate-only re-selection (see
    _run_refresh_candidates) against the same video an earlier analyze
    job already processed -- including one whose result was a
    RuntimeError (e.g. "insufficient eligible candidates") -- since only
    job.input (video_id/video_title), not job.result, is needed here.
    Never issued automatically: the frontend only calls this when the
    user explicitly clicks the "候補だけ再選定" action.
    """
    analyze_job = jobs.get_job(job_id)
    if analyze_job is None:
        raise HTTPException(status_code=404, detail="job not found")

    new_job = jobs.create_job("refresh_candidates", dict(analyze_job.input))
    jobs.run_async(new_job, _run_refresh_candidates, running_status="analyzing")
    return {"job_id": new_job.id}


@app.post("/api/jobs/{job_id}/refresh-stage1")
def refresh_stage1(job_id: str) -> dict[str, Any]:
    """Kicks off the mid-cost Stage1 re-analysis (see _run_refresh_stage1)
    against the same video an earlier analyze/refresh-candidates job
    already processed -- reusing only job.input (video_id/video_title),
    which works even when that job's own result was a RuntimeError (e.g.
    "cached Stage1 candidates no longer meet the quality bar"). Never
    issued automatically: the frontend only calls this when the user
    explicitly clicks the "Stage1からやり直す" action, since it costs one
    Anthropic API call per Stage1 chunk plus at most one Stage2 call.
    """
    analyze_job = jobs.get_job(job_id)
    if analyze_job is None:
        raise HTTPException(status_code=404, detail="job not found")

    new_job = jobs.create_job("refresh_stage1", dict(analyze_job.input))
    jobs.run_async(new_job, _run_refresh_stage1, running_status="analyzing")
    return {"job_id": new_job.id}


@app.get("/api/jobs/{job_id}/render/{render_id}")
def get_render(job_id: str, render_id: str) -> dict[str, Any]:
    render_job = jobs.get_job(render_id)
    if render_job is None or render_job.input.get("analyze_job_id") != job_id:
        raise HTTPException(status_code=404, detail="render job not found")
    resp = asdict(render_job)
    if render_job.status == config.JOB_STATE_INTERRUPTED:
        resp["resumable"] = True  # re-POST /render with the same candidate_id
    return resp


@app.get("/api/jobs/{job_id}/render/{render_id}/download")
def download_render(job_id: str, render_id: str) -> FileResponse:
    render_job = jobs.get_job(render_id)
    if render_job is None or render_job.input.get("analyze_job_id") != job_id:
        raise HTTPException(status_code=404, detail="render job not found")
    if render_job.status != "completed" or render_job.result is None:
        raise HTTPException(status_code=404, detail="レンダリングが完了していません")
    if not render_job.result.get("download_allowed"):
        raise HTTPException(status_code=403, detail="QAが重大不合格のためダウンロードできません")

    path = Path(render_job.result["final_video_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="ファイルが見つかりません")
    return FileResponse(path, media_type="video/mp4", filename=path.name)


# Static/media mounts registered last so they don't shadow the /api routes above.
app.mount("/media", StaticFiles(directory=config.OUTPUT_DIR), name="media")
if _STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
