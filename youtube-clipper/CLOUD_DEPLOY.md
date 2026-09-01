# クラウド運用（Railway）

PCで `start.bat` を使う従来の方法はそのまま使えます。これはそれとは別に、
Chromeのブックマークから固定URLで使うための追加の手段です。

## ユーザーが実際にやること

1. [Railway](https://railway.app/) で **New Project**
2. **Deploy from GitHub repo** → `VFTakanishi/-` を選択
3. Settings → **Root Directory** を `youtube-clipper` に設定
4. Variables に `ANTHROPIC_API_KEY` を設定（お手持ちのAPIキー）
5. Variables に `TOOL_PASSWORD` を設定（他人に使われないための合言葉。好きな文字列でOK）
6. Variables に `PODCAST_CLIPPER_OUTPUT_DIR=/data/output` を設定
7. **Volume** を作成し、マウント先を `/data` に設定
8. **Deploy**
9. デプロイ完了後、Settings → **Generate Domain** でURLを発行
   （例: `https://xxxxx.up.railway.app`）
10. そのURLをChromeにブックマーク

以降は、Chrome → ブックマーク → パスワード入力 → Podcast Clipper が使えます。
コマンドプロンプト・localhost・Windowsスタートアップ設定は一切不要です。

## 注意

- `PODCAST_CLIPPER_OUTPUT_DIR` は文字起こし/候補選定キャッシュと、Whisperモデルの
  保存先です。Volumeをマウントしないと、再デプロイのたびに消えて解析コストが
  再発生します。
- `TOOL_PASSWORD` を設定しなかった場合、そのURLは誰でもアクセスできてしまいます。
- ローカル利用（`start.bat`）では `TOOL_PASSWORD` は無関係です（未設定として動作）。
