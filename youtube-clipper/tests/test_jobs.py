import threading
import time

from podcast_clipper import jobs


def test_create_and_get_job():
    job = jobs.create_job("analyze", {"url": "https://example.com"})
    loaded = jobs.get_job(job.id)
    assert loaded is not None
    assert loaded.status == "queued"
    assert loaded.input["url"] == "https://example.com"


def test_run_async_success_updates_status_and_result():
    job = jobs.create_job("analyze", {})
    done = threading.Event()

    def fn(j):
        return {"ok": True, "job_id": j.id}

    jobs.run_async(job, fn, running_status="analyzing")

    for _ in range(50):
        latest = jobs.get_job(job.id)
        if latest.status == "completed":
            done.set()
            break
        time.sleep(0.05)

    assert done.is_set()
    latest = jobs.get_job(job.id)
    assert latest.result == {"ok": True, "job_id": job.id}
    assert latest.error is None
    assert latest.error_traceback is None


def test_run_async_failure_captures_error_not_raised_in_thread():
    job = jobs.create_job("render", {})

    def fn(_j):
        raise RuntimeError("boom")

    jobs.run_async(job, fn, running_status="rendering")

    for _ in range(50):
        latest = jobs.get_job(job.id)
        if latest.status == "failed":
            break
        time.sleep(0.05)

    latest = jobs.get_job(job.id)
    assert latest.status == "failed"
    assert "boom" in latest.error


def test_run_async_failure_persists_full_traceback():
    """Diagnostic-only (2026-08-30 incident): str(exc) alone was too little
    to find the actual failure (e.g. a bare "TypeError: string indices must
    be integers, not 'str'" names no file/line). error_traceback must carry
    the full traceback so a future failure is diagnosable from the job JSON
    file alone, without reproducing it by hand.
    """
    job = jobs.create_job("analyze", {})

    def fn(_j):
        raise TypeError("string indices must be integers, not 'str'")

    jobs.run_async(job, fn, running_status="analyzing")

    for _ in range(50):
        latest = jobs.get_job(job.id)
        if latest.status == "failed":
            break
        time.sleep(0.05)

    latest = jobs.get_job(job.id)
    assert latest.status == "failed"
    assert latest.error_traceback is not None
    assert "Traceback (most recent call last)" in latest.error_traceback
    assert "in fn" in latest.error_traceback
    assert "TypeError" in latest.error_traceback


def test_recover_interrupted_jobs_only_flags_in_progress_states():
    running_job = jobs.create_job("analyze", {})
    running_job.status = "analyzing"
    jobs.save_job(running_job)

    done_job = jobs.create_job("analyze", {})
    done_job.status = "completed"
    done_job.result = {"x": 1}
    jobs.save_job(done_job)

    recovered = jobs.recover_interrupted_jobs()

    assert running_job.id in recovered
    assert done_job.id not in recovered
    assert jobs.get_job(running_job.id).status == "interrupted"
    assert jobs.get_job(done_job.id).status == "completed"
