# 引き継ぎメモ（youtube-clipper: ポッドキャスト切り抜きツール）

最終更新: 2026-08-29（MVP設計変更「YouTube URL取得 → ローカル動画ファイルアップロード」を実装・feature branchへpush済みの時点）

## 現在地

- **ブランチ**: `claude/youtube-podcast-clip-tool-1qwqlh`（origin にpush済み）
- **実装状態**: 全モジュール実装済み・`pytest tests/` は58件全てパス（実ffmpeg統合テスト含む）
- **mainへは未マージ**（ユーザー指示により、実際のポッドキャスト1本が最後まで正常に完成するまでマージしない）

## MVP設計変更の経緯（2026-08-29）

以前のMVPは「YouTube URLをyt-dlpで取得 → Whisper文字起こし → …」だったが、以下の理由で**YouTube取得処理をMVPから完全に削除**し、「ローカル動画ファイルをブラウザへアップロード」を入力の起点に変更した。

- YouTube側のbot検知（`Sign in to confirm you're not a bot`）が、GitHub Actions等のデータセンターIPからのyt-dlp取得を恒常的にブロックすることを実測で確認（`player_client=[tv,android,web]`フォールバックでも解消せず）
- 対象動画はユーザー自身の配信であり、すでに手元にある。YouTubeへ再度取りに行く必要がない
- ネットワーク依存自体をなくすことで、bot検知・Cookie・PO Token・player_client・データセンターIP問題を設計上まとめて排除

Windowsローカル検証用に使った一時的なBAT/PowerShell/SABR回避パッチは製品設計に一切持ち込んでいない。

## 新しいエンドツーエンド処理フロー

1. ブラウザUIで動画ファイルをドラッグ&ドロップ、または「ファイルを選択」で指定し「解析開始」
2. `POST /api/analyze`（multipart）→ `ingest.ingest_uploaded_file`が**バックグラウンドジョブ開始前に同期的に**ファイルをチャンクストリーミングでディスクへ保存（内容のSHA-256から`video_id`を決定。同一内容の再アップロードは同じ`video_id`になりキャッシュを再利用）
3. バックグラウンドジョブが`transcribe.transcribe_video(source_path, video_id)`から開始（YouTube取得ステップは存在しない）
4. `clip_selector.select_candidates`（10分チャンク＋1分オーバーラップの2段階AI選定、キャッシュあり）
5. `boundary.resolve_candidate`で意味範囲→実秒への変換（3候補、`c1`/`c2`/`c3`）
6. ブラウザUIに3候補を表示 → ユーザーが1件選択
7. `POST /api/jobs/{id}/render`で選択した1件のみレンダリング（元映像全体＋ぼかし背景の縦型変換 → Content QA(中間映像) → テキスト焼き込み → Technical QA/音声QA(最終mp4)）
8. QA重大不合格でなければ`GET .../download`でmp4のみ取得（関連動画設定手順は別途JSON/UIで案内、自動投稿なし）

CLIは`python -m podcast_clipper.cli analyze-file "C:\path\podcast.mp4"`に変更（内部デバッグ専用、ブラウザUIが正式な入口という条件は維持）。

## 変更ファイル一覧

- 新規: `src/podcast_clipper/ingest.py`、`tests/test_ingest.py`
- 削除: `src/podcast_clipper/download.py`、`.github/workflows/youtube-clipper-e2e.yml`
- 変更: `src/podcast_clipper/{web.py, cli.py, config.py}`、`src/podcast_clipper/static/{index.html, app.js, style.css}`、`tests/test_web.py`、`requirements.txt`（yt-dlp削除・python-multipart追加）、`README.md`、`.github/workflows/youtube-clipper-windows-smoke.yml`（yt-dlp reachabilityステップ削除）

`.env.example`にYouTube固有の環境変数は元々なく、変更不要だった。

## 既知の残存事項

- `main`ブランチには以前のE2E検証準備の一環で`youtube-clipper-e2e.yml`（YouTube URL専用・現在はfeature branchから削除済み）のみをコピーしたコミットが残っている。今回の変更は**feature branchのみ**が対象のため、`main`上のこの古いworkflowファイルは今回未対応（本体コードは元々mainに入っていない）。次回mainへの反映を検討する際に合わせて整理するかどうかはユーザー判断待ち。
- 実際のポッドキャスト動画ファイルでのWhisper文字起こし〜候補選定〜レンダリングの実データE2Eテストは、本ピボット後まだ実施していない（下記「未完了の作業」参照）。

## 未完了の作業

- [ ] 実際の動画ファイルをアップロードしてのWhisper文字起こし〜候補3件表示までの実データ動作確認（faster-whisperモデルの初回ダウンロードにHugging Face到達性が必要。クラウドLinuxサンドボックスでは`huggingface.co`が403で拒否されるため、このセッションでは実行不可。Windows実機またはHugging Faceに到達できる環境で確認する必要がある）
- [ ] 候補選択→レンダリング→Technical QA→Content QA→ダウンロード可否判定の実データでの動作確認
- [ ] 完成mp4の実際の目視確認（尺25〜45秒/発話境界/2カット構成/1080x1920/ぼかし背景/字幕なし/常時ウォーターマーク/末尾CTA重畳/日本語文字化けなし/黒画面・フリーズなし/音声存在/QA不合格時DL不可）
- [ ] 上記で問題が見つかった場合の修正
- [ ] mainへのマージ判断（**現時点ではマージ不可** — 実際に1本最後まで正常完成していないため）

## 次にやること

1. この`HANDOFF.md`を読む
2. `git status`とブランチ（`claude/youtube-podcast-clip-tool-1qwqlh`）を確認
3. Hugging Face Hub（`huggingface.co`）と通常のインターネットアクセスがある環境（Windows実機等）で環境構築（`README.md`のセットアップ手順に従う）
4. `ANTHROPIC_API_KEY`を`.env`に設定（チャットに貼り付けない）
5. ブラウザUIで実際の動画ファイルをアップロードし、一連の流れ（解析→候補3件→選択→レンダリング→QA→ダウンロード）を実行し、「未完了の作業」の各項目を確認
6. 問題があれば修正→`pytest tests/`再実行→再確認
7. 実際の1本が最後まで正常に完成したらmainマージを検討（ユーザーの最終判断を仰ぐ）

## 必要なAPIキー・環境設定

- `ANTHROPIC_API_KEY`（必須。Claude API候補選定に使用）— `.env`に設定。チャットに貼り付けない
- `PODCAST_CLIPPER_FONT_PATH`（Windows実機では既定値`C:/Windows/Fonts/meiryo.ttc`を確認し、問題があれば単体`.ttf/.otf`日本語フォントに変更）
- ffmpeg/ffprobeがPATHに通っていること
- faster-whisperの初回モデル取得のためHugging Face Hub（`huggingface.co`）に到達できること（YouTubeへの到達性は不要になった）

## 14項目の絶対条件（変更しないこと）

以下はユーザーとの複数回の合意事項であり、今後の作業でも**変更・簡略化・追加機能の投入をしないこと**。

1. 候補は3件
2. ユーザーが3件から1件選択した後にのみレンダリング
3. ブラウザUIを主動線とする（CLIは内部デバッグ用途のみ）
4. Shortsは25〜45秒を基本目標
5. 基本2カット、内容に応じ1〜3カット
6. 通常の全文字幕は実装しない
7. 縦型変換は「元映像全体＋ぼかし背景」のみ（顔検出クロップは実装しない）
8. 「VF高西で検索！」常時表示＋末尾3〜5秒のみ追加CTA（別画面のCTAカードは使わない）
9. オープンループを絶対条件にしない（`hook_type`4分類は維持、「答えを絶対に見せない」ルールは禁止）
10. OP・ロゴ・静止画・黒画面・ジングル等を候補開始位置から除外（固定秒数のハードコードは禁止、内容判断＋Content QAで対処）
11. Claudeは意味範囲選択のみ、プログラム（`boundary.py`）は実編集点の微調整のみ（`boundary.py`が意味判断をする設計にはしない）
12. 10分チャンク＋1分オーバーラップの2段階AI選定＋キャッシュ（`cache.py`）を維持
13. Technical QA / Content QA（黒画面・静止画・音声存在・発話整合性・編集境界整合性を個別チェック）を実施し、重大不合格時はダウンロード禁止。映像Content QAはテキスト焼き込み前の中間映像に対して実行（焼き込み後の最終mp4には実行しない）
14. YouTube「関連動画」機能を本編誘導の主要導線とする。自動投稿はMVP対象外。`/download`APIはmp4のみ返却し、関連動画設定手順はレンダリング結果JSON/UIに分離

追加しないもの: 全文字幕、顔検出クロップ、自動リフレーム、人物追跡、クロスフェード、YouTube自動投稿・再取得、Celery/Redis等の不要な外部インフラ、その他今回決めていない新機能。
