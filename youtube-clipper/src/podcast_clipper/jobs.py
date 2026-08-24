"""Minimal background job runner (plan: ThreadPoolExecutor + JSON state,
deliberately not Celery/Redis/etc — this is a single-user local tool).

Plan fix #5: if the process is killed mid-job, its JSON state file is left
sitting at an in-progress status ("queued"/"analyzing"/"rendering") even
though no worker is actually running it anymore. `recover_interrupted_jobs`
must be called once at web.py startup to sweep those into
config.JOB_STATE_INTERRUPTED, so the UI can tell the user the job didn't
finish rather than showing a spinner forever. Because transcript/Stage1/
Stage2 are cached (cache.py) keyed by video_id, re-analyzing after an
interruption reuses whatever finished before the crash instead of paying
for YouTube download / Whisper / Claude calls again.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import config

_executor = ThreadPoolExecutor(max_workers=2)
_lock = threading.Lock()


def _atomic_write_text(path: Path, content: str) -> None:
    """Writes via a temp file + os.replace so a concurrent reader (e.g. the
    UI polling GET /api/jobs/{id} while a background job is mid-save) never
    observes a truncated/partial file. os.replace is atomic on both POSIX
    and Windows.
    """
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _jobs_dir() -> Path:
    d = config.OUTPUT_DIR / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _job_path(job_id: str) -> Path:
    return _jobs_dir() / f"{job_id}.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    id: str
    type: str  # "analyze" | "render"
    status: str
    created_at: str
    updated_at: str
    input: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None


def save_job(job: Job) -> None:
    job.updated_at = _now()
    with _lock:
        _atomic_write_text(
            _job_path(job.id), json.dumps(asdict(job), ensure_ascii=False, indent=2)
        )


def get_job(job_id: str) -> Job | None:
    path = _job_path(job_id)
    if not path.exists():
        return None
    return Job(**json.loads(path.read_text(encoding="utf-8")))


def list_jobs() -> list[Job]:
    return [Job(**json.loads(p.read_text(encoding="utf-8"))) for p in _jobs_dir().glob("*.json")]


def create_job(job_type: str, input_data: dict[str, Any]) -> Job:
    job = Job(
        id=uuid.uuid4().hex,
        type=job_type,
        status="queued",
        created_at=_now(),
        updated_at=_now(),
        input=input_data,
    )
    save_job(job)
    return job


def run_async(job: Job, fn: Callable[[Job], dict[str, Any]], running_status: str) -> None:
    """Runs `fn(job)` in the background executor. `fn` should return a
    JSON-serializable result dict on success, or raise on failure.
    """
    job.status = running_status
    save_job(job)

    def _worker() -> None:
        try:
            result = fn(job)
            latest = get_job(job.id) or job
            latest.status = "completed"
            latest.result = result
            save_job(latest)
        except Exception as exc:  # noqa: BLE001 - job failures must be captured, not raised in a thread
            latest = get_job(job.id) or job
            latest.status = "failed"
            latest.error = str(exc)
            save_job(latest)

    _executor.submit(_worker)


def recover_interrupted_jobs() -> list[str]:
    """Call once at startup. Returns the ids of jobs marked interrupted."""
    recovered = []
    for job in list_jobs():
        if job.status in config.JOB_STATES_IN_PROGRESS:
            job.status = config.JOB_STATE_INTERRUPTED
            save_job(job)
            recovered.append(job.id)
    return recovered
