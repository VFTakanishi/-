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

from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
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

    try:
        idx = int(candidate_id.lstrip("c")) - 1
        raw_candidate = raw_candidates[idx]
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"不正な candidate_id: {candidate_id}") from exc

    resolved_candidate = boundary.resolve_candidate(raw_candidate, transcript, candidate_id)

    manifest = render.render_candidate(source_path, resolved_candidate, video_id)
    qa_report = qa.run_full_qa(raw_candidate, transcript, manifest)

    return {
        "video_id": video_id,
        "candidate_id": candidate_id,
        "qa": _serialize_qa(qa_report),
        "download_allowed": qa_report.download_allowed,
        "final_video_path": manifest.final_video_path,
        "related_video_instructions": config.RELATED_VIDEO_INSTRUCTIONS,
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
