# Podcast Clipper

自分のYouTubeポッドキャストの本編動画ファイルから、YouTube Shorts向けの切り抜き動画を作るツールです。

主動線はブラウザUIです。手元の動画ファイルをアップロードすると、AIが文字起こしを解析して切り抜き候補を3件提示し、選んだ1件だけをレンダリング・QAした上でダウンロードできます。動画はすでに自分がYouTubeへ公開している（またはこれから公開する）ものを想定しており、YouTubeから動画を取得する処理はこのツールには含まれません。

このツールの設計判断（絶対条件・技術的な理由）は `../../.claude/plans/` に残っているプラン文書、またはリポジトリのやり取りを参照してください。ここでは使い方のみを説明します。

PCで都度 `start.bat` を起動する代わりに、Chromeのブックマークから固定URLで使いたい場合は [CLOUD_DEPLOY.md](./CLOUD_DEPLOY.md)（Railwayへのデプロイ手順）を参照してください。

## セットアップ

### 1. 依存関係

- Python 3.10以上
- [ffmpeg](https://ffmpeg.org/download.html)（`ffmpeg`と`ffprobe`にPATHが通っていること）
- Anthropic APIキー（Claude API）

```bash
cd youtube-clipper
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -e .
pip install -r requirements.txt
```

### 2. 環境変数

`.env.example` を `.env` にコピーして編集してください。

```bash
copy .env.example .env    # Windows
# cp .env.example .env    # macOS/Linux
```

最低限、`ANTHROPIC_API_KEY` の設定が必須です。

### 3. 日本語フォントの設定（Windows環境で重要）

ffmpegでの日本語テキスト描画（ウォーターマーク「VF高西で検索！」・フックテキスト・CTAテキスト）が文字化け・豆腐表示にならないよう、`PODCAST_CLIPPER_FONT_PATH` に**実在する日本語対応フォントファイル**のパスを設定してください。

- 既定値は `C:/Windows/Fonts/meiryo.ttc` ですが、`.ttc`（TrueType Collection）は複数書体を含み、意図しない書体が選ばれることがあります。可能であれば単体の `.ttf`/`.otf` 日本語フォント（例: [Noto Sans JP](https://fonts.google.com/noto/specimen/Noto+Sans+JP)）をダウンロードし、そのパスを設定することを推奨します。
- パスはフォワードスラッシュ表記（`C:/Fonts/...`）で設定してください。
- 指定したフォントファイルが存在しない場合、レンダリングは**失敗として扱われます**（文字化けしたまま成功扱いにはなりません）。

**手動確認が必要な項目**: 生成されたクリップを実際に再生し、ウォーターマーク・フックテキスト・CTAテキストの日本語が正しく表示されている（豆腐化・文字化けしていない）ことを目視で確認してください。この確認は自動化されていません。

## 起動方法（ブラウザUI）

### Windowsで簡単に起動する（推奨）

`youtube-clipper` フォルダ直下の **`start.bat`をダブルクリック**するだけで起動できます。サーバーが立ち上がると自動的に既定のブラウザで `http://localhost:8000` が開きます。

- `.venv` が存在すればそれを使用します（無ければPATH上のPythonを探します）
- Pythonが見つからない場合や、依存パッケージ・`ANTHROPIC_API_KEY`が不足している場合は、ウィンドウを閉じずに日本語でその旨を表示します
- 終了するには、開いたウィンドウで`Ctrl+C`を押すか、ウィンドウを閉じてください

### 手動で起動する場合（Windows以外・上級者向け）

```bash
uvicorn podcast_clipper.web:app --reload --port 8000
```

起動後、ブラウザで `http://localhost:8000` を開いてください。

1. 動画ファイルをドラッグ＆ドロップ、または「ファイルを選択」で指定し「解析開始」を押す（アップロード・文字起こし・AI候補選定が実行されます。数分かかることがあります）
2. 提示された3候補から1件を選び「この候補で作成」を押す（選んだ候補のみレンダリングされます）
3. QA結果を確認する。**重大な不合格がある場合はダウンロードボタンが無効化されます**
4. QA合格後、ダウンロードボタンからmp4を取得する

## アップロード後の設定（重要）

このツールは自動投稿を行いません。生成したmp4をYouTubeへ手動でアップロードした後、**YouTube Studioでそのショートの「関連動画」に元のポッドキャスト本編を設定してください**。本編への誘導はこの機能が主要な導線です（詳細はレンダリング結果画面にも表示されます）。

## 内部デバッグ用CLI

ブラウザUIを経由せずパイプラインを直接叩きたい場合:

```bash
python -m podcast_clipper.cli analyze-file "C:\path\to\podcast.mp4"
python -m podcast_clipper.cli render <video_id> c1
```

## キャッシュと再実行

アップロードされた動画ファイルは内容のSHA-256ハッシュから`video_id`を決定するため、同じ内容のファイルを再アップロードすると同じ`video_id`になります。`output/<video_id>/cache/` に文字起こし・AI候補選定結果がキャッシュされ、同じ動画を再解析する際はこのキャッシュが再利用され、Whisper再実行・Claude API再呼び出しは行われません（`force_refresh`を指定した場合を除く）。

アプリ/PCが処理途中で終了した場合、次回起動時に中断されたジョブは自動的に`interrupted`状態として検出されます。ブラウザUIにその旨が表示されるので、同じファイルを再アップロードして再解析を実行してください（上記のキャッシュにより、完了済みの処理はスキップされます）。

## テスト

```bash
pip install pytest
pytest tests/
```

ffmpegがローカルにない場合、実際のffmpeg実行を伴うテストは自動的にスキップされます。

## このリポジトリでの位置づけ

`youtube-clipper/` は既存の車検整備メモアプリ（リポジトリルートの`src/`等）とは完全に独立したツールです。依存関係・実行方法はいずれも別個であり、既存アプリには一切影響しません。
