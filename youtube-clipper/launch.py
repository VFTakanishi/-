"""Windows double-click launcher entry point (invoked by start.bat, and by
the desktop shortcut setup_desktop_shortcut.ps1 creates -- see its
docstring).

Not part of the podcast_clipper package: this is operational glue only.
Responsibilities:
- make `import podcast_clipper` work even if the venv doesn't have it
  pip-installed as editable (falls back to adding src/ to sys.path)
- fail loudly, in Japanese, naming the missing piece, instead of letting a
  bare traceback or a silent crash confuse a non-technical user
- warn (but not block) if ANTHROPIC_API_KEY isn't set, since the
  upload/drag-and-drop screen itself doesn't need it
- open the default browser to the app once the server is about to start
- write everything to logs/launcher.log instead of the console
  (redirect_output_to_log_file): the desktop-shortcut path runs via
  pythonw.exe, which has no console at all, so sys.stdout/sys.stderr may
  be closed or silently discard writes depending on the Python build --
  every print() below would otherwise vanish with no way to diagnose a
  failed startup. Redirecting both streams to a real file *before*
  anything else runs (in particular, before uvicorn.run() below builds
  its own logging config from sys.stderr) means uvicorn's own
  request/error logs land in the same file too, not just this script's
  own messages. start.bat's console path is unaffected in substance --
  the same text still exists, just in the log file instead of only on
  screen.
- detect an already-running instance (server_already_running, via a
  real GET /health -- no external network involved) so double-clicking
  the desktop shortcut a second time while Podcast Clipper is already
  open never tries to bind the same port twice: it just re-opens the
  browser to the existing instance and exits.

Every side-effecting step (log redirection, the already-running check,
importing uvicorn/podcast_clipper.web, actually starting the server) is
gathered under main(), guarded by `if __name__ == "__main__"` -- so this
module can be imported (e.g. by tests) without starting a real server,
while `python launch.py` / `pythonw.exe launch.py` behaves exactly as
before.
"""
from __future__ import annotations

import datetime
import os
import socket
import sys
import threading
import urllib.request
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

HOST = "127.0.0.1"
PORT = 8000
URL = "http://localhost:8000"
LOG_DIR = BASE_DIR / "logs"
LOG_PATH = LOG_DIR / "launcher.log"


def redirect_output_to_log_file() -> None:
    """Points sys.stdout/sys.stderr at logs/launcher.log (append mode,
    line-buffered so a later crash doesn't lose already-written lines).
    Safe to call unconditionally: under a normal console (start.bat) this
    just moves the same text from the window into a file; under
    pythonw.exe (no console at all) it's what makes any output possible
    to inspect after the fact. Never raises -- if the log file can't be
    created (e.g. a permissions issue), silently leaves stdout/stderr as
    whatever they already were, which is no worse than before this
    function existed.
    """
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
        sys.stdout = log_file
        sys.stderr = log_file
    except OSError:
        pass


def server_already_running(
    host: str = HOST, port: int = PORT,
    connect_timeout: float = 0.5, health_timeout: float = 1.0,
) -> bool:
    """True only if something on host:port answers a real GET /health with
    200 -- i.e. this is genuinely an already-running Podcast Clipper
    instance, not just some unrelated process that happens to hold the
    port (which must still be allowed to surface its own "address already
    in use" error from uvicorn.run(), same as before this function
    existed, rather than being silently treated as "already running").
    No external network involved (127.0.0.1 only, and /health itself
    never calls the Anthropic API -- see web.py). The raw TCP connect
    first is a near-instant way to rule out "nothing is listening at all"
    without waiting on an HTTP round trip in the common case (nothing
    running yet).
    """
    try:
        with socket.create_connection((host, port), timeout=connect_timeout):
            pass
    except OSError:
        return False
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=health_timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def open_browser(url: str = URL) -> None:
    try:
        webbrowser.open(url)
    except Exception:
        pass  # a failed browser auto-open must never take the server down


def main() -> None:
    redirect_output_to_log_file()
    print(f"\n{'=' * 60}")
    print(f"起動: {datetime.datetime.now().isoformat(timespec='seconds')}")
    print("=" * 60)

    if server_already_running():
        print("Podcast Clipper は既に起動しています。ブラウザで開いて終了します。")
        open_browser()
        return

    try:
        import uvicorn
        from podcast_clipper.web import app
    except ImportError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        print("=" * 60)
        print("必要なPythonパッケージが不足しています。")
        print(f"  不足しているモジュール: {missing}")
        print()
        print("このフォルダ（youtube-clipper）で以下を実行してから")
        print("もう一度 start.bat をダブルクリックしてください:")
        print()
        print("  pip install -e .")
        print("  pip install -r requirements.txt")
        print("=" * 60)
        sys.exit(1)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("=" * 60)
        print("警告: ANTHROPIC_API_KEY が設定されていません。")
        print()
        print("このフォルダの .env ファイルに以下の形式で設定してください:")
        print("  ANTHROPIC_API_KEY=sk-ant-...")
        print()
        print("このまま起動しますが、設定するまで動画の解析（AI候補選定）は")
        print("失敗します。詳しくは README.md を参照してください。")
        print("=" * 60)
        print()

    threading.Timer(1.5, open_browser).start()

    print(f"Podcast Clipper を起動しています。まもなく {URL} が開きます。")
    print(f"ログは {LOG_PATH} に出力されます。")
    print("終了するには、このプロセスを終了してください（コンソールがある場合は Ctrl+C）。")
    print()

    try:
        uvicorn.run(app, host=HOST, port=PORT)
    except Exception as exc:
        print()
        print("=" * 60)
        print("サーバーの起動中にエラーが発生しました:")
        print(f"  {exc}")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
