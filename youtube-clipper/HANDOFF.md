# 引き継ぎメモ（youtube-clipper: YouTubeポッドキャスト切り抜きツール）

最終更新: 2026-08-24（Linuxサンドボックスでの部分E2E検証を実施した時点）

## 現在地

- **ブランチ**: `claude/youtube-podcast-clip-tool-1qwqlh`（origin にpush済み）
- **実装状態**: 全モジュール実装済み・`pytest tests/` は51件全てパス（実ffmpeg統合テスト含む）
- **mainへは未マージ**（ユーザー指示により、実際のポッドキャスト1本が最後まで正常に完成するまでマージしない）
- 直近のコミット: `Add YouTube podcast Shorts clipper (browser UI + pipeline)`

## 今日完了した内容

1. 実装完了報告後、mainマージ前のE2Eテストを依頼された
2. 本セッションがクラウドLinuxサンドボックス（Windows実機ではない）であることを確認し、ユーザーに「このLinux環境で代行する」承認を得た
3. **重大な環境制約を2件発見**（詳細は下記「発見した環境ブロッカー」）
4. ユーザー承認のもと、YouTube以外の実在する日本語音声（パブリックドメイン）で代替検証する方針に切り替え
5. 実際に以下を検証済み（下記「E2Eテスト結果」参照）:
   - 依存関係（Python/ffmpeg/ffprobe/yt-dlp/faster-whisper/anthropic/FastAPI等）は全てインストール済み・正常動作
   - 実際の`podcast_clipper.web`（無改変）をuvicornで起動し、ブラウザ（Playwright/Chromium）で`http://localhost:8000`のUIが実際に正常表示されることをスクリーンショットで確認（日本語文字化けなし）
   - UIの「解析開始」ボタンを実際にクリックし、実際の`POST /api/analyze`→ジョブ管理（`jobs.py`）→`download.py`の呼び出しが正しく配線されていることを確認（`download.download_video`のみ、テスト用に用意した実音声ファイルを返すようモンキーパッチ。リポジトリのソースコード自体は無変更）
   - `pytest tests/`（51件）が引き続き全てパス

## E2Eテスト結果

### 使用した代替素材
- YouTube実ダウンロードがネットワークポリシー上不可能なため、GitHub Release経由で実在するパブリックドメイン日本語音声を取得
- 出典: `kaiidams/Kokoro-Speech-Dataset`（LibriVox録音・青空文庫テキスト、夏目漱石『こころ』の朗読クリップ34件、本文中の順序通りに結合、実時間 約2分39秒）
- 映像トラックは到達可能な実写素材が存在しないため、ffmpeg合成のテストパターン映像（1280x720）と合成。**音声は実在する公有音声、映像は合成プレースホルダー**という構成
- 素材の保存場所: このセッションのスクラッチパッド（`/tmp/.../scratchpad/e2e/`）。**リポジトリには含めていない**（再現する場合は同じ手順でGitHub Releaseから再取得可能。詳細は本ファイル末尾の「代替素材の再現手順」）

### 到達できたところ
- ブラウザUI表示 ✅（実際にPlaywrightでスクリーンショット確認、日本語表示も正常）
- URL入力→解析開始ボタン→`POST /api/analyze`→ジョブ作成→`download.py`（モック経由）まで ✅
- ジョブの`input`に正しく`video_id`・`video_title`・`source_path`が記録されることを確認 ✅

### ブロックされた地点
`transcribe.py`の`faster-whisper`モデル初期化（`WhisperModel(...)`のコンストラクタ）で失敗。原因は後述の環境ブロッカー。**そこから先（Whisper文字起こし→Claude 2段階候補選定→候補3件表示→ユーザー選択→レンダリング→QA→ダウンロード）は本セッションでは実行できていない。**

## 発見した環境ブロッカー（コードのバグではない）

1. **YouTube等へのアクセスがネットワークポリシーで403拒否**
   `youtube.com` / `archive.org` / `wikimedia.org` が、このサンドボックスの outbound プロキシ（許可リスト方式）で拒否される。`yt-dlp`によるYouTube実ダウンロードは本環境では実行不可能。
2. **Hugging Face Hubへのアクセスも403拒否 → faster-whisperのモデルダウンロードが失敗**
   `faster-whisper`は初回実行時にWhisperモデル重みを`huggingface_hub`経由でダウンロードする実装になっており、`huggingface.co`も同じ理由で403拒否される。そのため**このサンドボックスではWhisper文字起こしそのものが実行不可能**（コード側の問題ではなく、モデルの重みを一度も取得できていないことが原因）。
   - 実際のエラー: `httpx.ProxyError: 403 Forbidden`（`faster_whisper.utils.download_model` → `huggingface_hub.snapshot_download` 内で発生）
   - 到達可能だったドメイン（参考）: `github.com` / `raw.githubusercontent.com` / `objects.githubusercontent.com`（GitHub Releaseアセット）/ `storage.googleapis.com`
3. **`ANTHROPIC_API_KEY`が未設定**
   ユーザー指示により、チャットへのキー貼り付けは求めていない。Claude API呼び出し（候補選定）に到達する前に確認が必要（ブロッカー2により、そもそも今回はそこまで到達しなかった）。

**重要**: 上記3点はいずれも**このクラウドサンドボックス特有のネットワーク制約**であり、Windows実機（通常のインターネットアクセスがある環境）では発生しないと想定される。今回の検証で**リポジトリのコード自体に不具合は見つからなかった**（見つかった範囲では）。

## 未完了の作業

- [ ] Whisper文字起こし〜候補3件表示までの実データでの動作確認
- [ ] 候補選択→レンダリング→Technical QA→Content QA→ダウンロード可否判定の実データでの動作確認
- [ ] 完成mp4の実際の目視確認（尺25〜45秒/発話境界/2カット構成/1080x1920/ぼかし背景/字幕なし/常時ウォーターマーク/末尾CTA重畳/日本語文字化けなし/黒画面・フリーズなし/音声存在/QA不合格時DL不可）
- [ ] 上記で問題が見つかった場合の修正
- [ ] mainへのマージ判断（**現時点ではマージ不可** — 実際に1本最後まで正常完成していないため）

## 明日最初にやること

1. この`HANDOFF.md`を読む
2. `git status`とブランチ（`claude/youtube-podcast-clip-tool-1qwqlh`）を確認
3. **実行環境をWindows実機（または少なくとも通常のインターネットアクセスがある環境）に変更する**（本セッションのクラウドサンドボックスではWhisperモデルのダウンロードすらできないため、E2Eテスト続行は実質的に不可能）
4. Windows実機で `youtube-clipper/README.md` のセットアップ手順に従い環境構築
5. `ANTHROPIC_API_KEY`を`.env`に設定（チャットに貼り付けない）
6. 実際のYouTubeポッドキャストURLで一連の流れ（解析→候補3件→選択→レンダリング→QA→ダウンロード）を実行し、「未完了の作業」の各項目を確認
7. 問題があれば修正→`pytest tests/`再実行→再確認
8. 実際の1本が最後まで正常に完成したらmainマージを検討（ユーザーの最終判断を仰ぐ）

## 必要なAPIキー・環境設定

- `ANTHROPIC_API_KEY`（必須。Claude API候補選定に使用）— `.env`に設定。チャットに貼り付けない
- `PODCAST_CLIPPER_FONT_PATH`（Windows実機では既定値`C:/Windows/Fonts/meiryo.ttc`を確認し、問題があれば単体`.ttf/.otf`日本語フォントに変更）
- ffmpeg/ffprobeがPATHに通っていること
- 通常のインターネットアクセス（YouTube・Hugging Face Hubへ到達可能なこと）— 本クラウドサンドボックスでは両方とも403でブロックされていた

## 14項目の絶対条件（変更しないこと）

以下はユーザーとの複数回の合意事項であり、明日以降の作業でも**変更・簡略化・追加機能の投入をしないこと**。

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

追加しないもの: 全文字幕、顔検出クロップ、自動リフレーム、人物追跡、クロスフェード、YouTube自動投稿、Celery/Redis等の不要な外部インフラ、その他今回決めていない新機能。

## 代替素材の再現手順（参考）

本セッションのスクラッチパッドは次回のセッションには引き継がれないため、同様の代替検証を再度行う場合は以下で再現できる:

```bash
curl -sSL -o sample.zip \
  "https://github.com/kaiidams/Kokoro-Speech-Dataset/releases/download/1.3/kokoro-speech-v1_3-sample-flac.zip"
unzip sample.zip -d extracted
# extracted/wavs/kokoro-by-soseki-natsume-*.flac を番号順（=本文中の位置順）に結合
# → ffmpeg concatで1本の音声にし、lavfi testsrc2等の合成映像と合成してmp4化
```

ただし本命はWindows実機での**実際のYouTube URL**によるテストであり、この代替素材はあくまで昨日ネットワーク制約下で可能だった範囲の代替検証用。
