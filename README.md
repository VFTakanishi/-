# YouTube Live Skincare Analyzer MVP

YouTubeライブを視聴しているPCブラウザの映像を、視聴者側の画面共有経由で解析し、顔付近での反復的な「塗布動作」からスキンケア工程数を推定する React + TypeScript の MVP です。

## できること

- `getDisplayMedia()` で YouTube ライブのタブ共有映像を取得
- MediaPipe の Face Landmarker / Hand Landmarker で顔と手を検出
- 手が顔付近に 3 秒以上あり、小刻みに往復しているときに「1工程」としてカウント
- 検出後は 15 秒クールダウン
- 4工程の進捗表示
- 手動の `1つ戻す` `1つ進める` `リセット`
- YouTube コメント文のコピー

## セットアップ

Node.js が必要です。

- 推奨: Node.js `20.19+` もしくは `22.12+`

インストール後、プロジェクト直下で以下を実行してください。

```bash
npm install
```

## 起動方法

```bash
npm run dev
```

ブラウザで表示されたローカル URL を開き、`画面共有開始` を押してください。

## 他の人へ共有する方法

ローカルの `localhost` URL は自分の PC だけで使えます。別の人にも使ってもらう場合は、Vercel などへ公開してください。

- 公開用設定ファイル: [vercel.json](C:/Users/USER/Documents/Codex/2026-06-18/pc-youtube-web-pc-youtube-youtube/vercel.json)
- 手順メモ: [DEPLOY_VERCEL.txt](C:/Users/USER/Documents/Codex/2026-06-18/pc-youtube-web-pc-youtube-youtube/DEPLOY_VERCEL.txt)

## 使い方

1. PCブラウザで YouTube ライブを開く
2. このアプリで `画面共有開始` を押す
3. YouTube ライブのタブ、またはライブ再生中ウィンドウを共有する
4. アプリが共有映像を解析し、塗布動作を工程としてカウントする

## 判定ロジック

- 顔ランドマークから顔領域の外接矩形を作成
- その領域を少し広げた近接範囲を判定対象にする
- 手ランドマーク群の重心、または手の点が顔近接範囲に入れば候補開始
- 候補状態が `3秒` 以上続く
- その間の手の移動が「小刻みで往復的」なら塗布動作とみなす
- 1回の確定で 1工程カウント
- その後 `15秒` は再検出しない

## 閾値の変更

調整用の設定は以下にまとめています。

- [src/config/detectionConfig.ts](C:/Users/USER/Documents/Codex/2026-06-18/pc-youtube-web-pc-youtube-youtube/src/config/detectionConfig.ts)

主な項目:

- `requiredContinuousMs`: 顔付近に手が必要な連続時間
- `cooldownMs`: 1工程検出後の待機時間
- `facePaddingRatio`: 顔近接範囲の広げ幅
- `minPathDistanceRatio`: 候補期間中の最小移動距離
- `maxNetDisplacementRatio`: 始点終点の最大ずれ量
- `minDirectionChanges`: 往復動作とみなす最小方向転換数
- `minAverageStepRatio`, `maxAverageStepRatio`: 小刻み動作の1ステップ幅

## 注意点

- 初回起動時は MediaPipe の WASM とモデル読み込みのため、少し待つことがあります
- YouTube の映像品質、手の写り方、顔の角度で検出精度は変わります
- この MVP は「商品名」「音声」ではなく、顔付近への手の塗布動作だけを使って推定します
- `1つ戻す` `1つ進める` `リセット` の手動修正は、同じブラウザ内に学習データとして保存され、翌日以降の判定調整にも使われます
- ローカル環境に Node.js が未導入だと `npm install` と `npm run dev` は実行できません
