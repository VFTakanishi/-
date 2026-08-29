"""Windows double-click launcher entry point (invoked by start.bat).

Not part of the podcast_clipper package: this is operational glue only.
Responsibilities:
- make `import podcast_clipper` work even if the venv doesn't have it
  pip-installed as editable (falls back to adding src/ to sys.path)
- fail loudly, in Japanese, naming the missing piece, instead of letting a
  bare traceback or a silent crash confuse a non-technical user
- warn (but not block) if ANTHROPIC_API_KEY isn't set, since the
  upload/drag-and-drop screen itself doesn't need it
- open the default browser to the app once the server is about to start
"""
from __future__ import annotations

import os
import sys
import threading
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

URL = "http://localhost:8000"

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


def _open_browser() -> None:
    try:
        webbrowser.open(URL)
    except Exception:
        pass  # a failed browser auto-open must never take the server down


threading.Timer(1.5, _open_browser).start()

print(f"Podcast Clipper を起動しています。まもなく {URL} が開きます。")
print("終了するには、このウィンドウで Ctrl+C を押してください。")
print()

try:
    uvicorn.run(app, host="127.0.0.1", port=8000)
except Exception as exc:
    print()
    print("=" * 60)
    print("サーバーの起動中にエラーが発生しました:")
    print(f"  {exc}")
    print("=" * 60)
    sys.exit(1)
