"""Tests for launch.py's pure helper functions (round: desktop shortcut /
pythonw.exe support). launch.py lives at the repo root, not inside the
podcast_clipper package (it's operational glue -- see its own docstring),
so it's imported here via an explicit sys.path insert of the repo root.
Every side-effecting step (log redirection, the already-running check,
importing uvicorn/starting the server) lives inside main(), guarded by
`if __name__ == "__main__"`, so `import launch` here never starts a real
server or blocks -- only the pure functions this file exercises run.
"""
from __future__ import annotations

import http.server
import socketserver
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import launch  # noqa: E402  (must follow the sys.path insert above)


class _NotFoundHandler(http.server.BaseHTTPRequestHandler):
    """Simulates an unrelated process that happens to hold the port --
    listens, but has no real /health endpoint."""

    def do_GET(self):  # noqa: N802 (stdlib naming)
        self.send_response(404)
        self.end_headers()

    def log_message(self, *args):  # silence stdlib's default request logging
        pass


class _HealthyHandler(http.server.BaseHTTPRequestHandler):
    """Simulates a real, already-running Podcast Clipper instance."""

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def _http_server():
    """Starts a throwaway local HTTP server on an OS-assigned free port,
    torn down after the test. Yields (server, port).
    """
    started = []

    def _start(handler_cls):
        srv = socketserver.TCPServer(("127.0.0.1", 0), handler_cls)
        port = srv.server_address[1]
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        started.append(srv)
        time.sleep(0.05)  # let the listener actually come up
        return srv, port

    yield _start
    for srv in started:
        srv.shutdown()
        srv.server_close()


def test_server_already_running_false_when_nothing_listening():
    # A: nothing bound to this port at all -- must return quickly (bounded
    # by connect_timeout), never hang waiting for a launch that isn't
    # coming.
    start = time.monotonic()
    result = launch.server_already_running(port=18091, connect_timeout=0.3, health_timeout=0.5)
    elapsed = time.monotonic() - start
    assert result is False
    assert elapsed < 1.0


def test_server_already_running_false_for_unrelated_process_on_port(_http_server):
    # B: something is listening, but it isn't Podcast Clipper (no real
    # /health) -- must not be mistaken for "already running", so the
    # normal uvicorn "address already in use" error can still surface
    # exactly as it did before this feature existed.
    _srv, port = _http_server(_NotFoundHandler)
    assert launch.server_already_running(port=port, connect_timeout=0.3, health_timeout=0.5) is False


def test_server_already_running_true_for_real_health_endpoint(_http_server):
    # C: a genuine 200 from GET /health -- the exact signal a second
    # double-click of the desktop shortcut must detect to avoid starting
    # a duplicate server.
    _srv, port = _http_server(_HealthyHandler)
    assert launch.server_already_running(port=port, connect_timeout=0.3, health_timeout=0.5) is True


def test_open_browser_never_raises_even_if_webbrowser_fails(monkeypatch):
    # A failed browser auto-open must never take the server down (mirrors
    # the pre-existing try/except in the original _open_browser).
    def _boom(url):
        raise RuntimeError("no browser available")

    monkeypatch.setattr(launch.webbrowser, "open", _boom)
    launch.open_browser("http://localhost:8000")  # must not raise


def test_redirect_output_to_log_file_creates_log_and_redirects(monkeypatch, tmp_path):
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(launch, "LOG_DIR", log_dir)
    monkeypatch.setattr(launch, "LOG_PATH", log_dir / "launcher.log")

    launch.redirect_output_to_log_file()
    try:
        print("test message for the log file")
        sys.stdout.flush()
    finally:
        # Restore real stdout/stderr so pytest's own output capture isn't
        # left pointed at a closed test file after this test ends.
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    assert (log_dir / "launcher.log").exists()
    assert "test message for the log file" in (log_dir / "launcher.log").read_text(encoding="utf-8")


def test_redirect_output_to_log_file_never_raises_when_log_dir_uncreatable(monkeypatch):
    # Defensive: if the log directory genuinely can't be created (e.g. a
    # permissions issue), this must not crash the launcher -- worst case
    # is no logging, never a hard failure to start at all. Patches only
    # launch.LOG_DIR itself (a fake object), never pathlib.Path globally
    # -- a global Path.mkdir patch would break pytest's own file handling
    # for the rest of the process.
    class _UncreatableDir:
        def mkdir(self, *args, **kwargs):
            raise OSError("permission denied (simulated)")

    monkeypatch.setattr(launch, "LOG_DIR", _UncreatableDir())
    launch.redirect_output_to_log_file()  # must not raise


def test_main_is_guarded_by_dunder_main(monkeypatch):
    # Regression: importing launch.py (as this test file's own top-level
    # `import launch` already did) must never itself call main() / start
    # a real uvicorn server -- that only happens via `python launch.py`.
    # If this guard were ever removed, collecting this test file would
    # already have hung/crashed before reaching this assertion.
    assert callable(launch.main)
    assert launch.__name__ == "launch"  # not "__main__" under pytest's import
