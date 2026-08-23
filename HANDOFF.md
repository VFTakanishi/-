# 引き継ぎメモ（車検整備 音声点検メモツール）

最終更新: 2026-08-23（本日の作業終了時点）

## 現在地

- **ブランチ**: `claude/auto-repair-voice-memo-pjt3oc`
- **最新コミットSHA**: `493a926`（直近正常確認済み。作業終了時点でもこのSHAのまま、以降の変更なし）
- **git working tree**: クリーン（未コミット変更なし）
- **`npm run build`**: 成功（型エラーなし、`tsc -b && vite build` 完走）
- **Vercelプロジェクト名**: `shaken-voice-inspection`（Team: VFTakanishi / scope: vft-akanishi、Preview Deployment運用、production/mainには一切デプロイ・mergeしていない）
- **iPhone Safari実機テスト中**。ここまでの複数ラウンドの実機フィードバックを都度取り込み済み。

## 現在までに完成している重要機能

- 音声点検メモ本体（新規点検開始・お客様名/車種入力・過去点検一覧/再開/削除）
- **複数項目の連続発話解析**（`parseVoiceInspections()`）: 1発話に複数項目が含まれても、登録済み項目エイリアスの出現順にセグメント分割し、後続項目の判定語が前の項目に混入しないよう設計済み
- 判定語の音声エイリアス（OK/NG/交換/推奨/なし/要望/不要 等の短い言い回し含む）
- iPhone Safari対応（`webkitSpeechRecognition`の非continuousセッション＋onend自動再開による疑似連続リスニング）
- **手動マイク停止時の誤エラー対策**（`manualStopRef`でユーザー自身のstop()由来のaborted等を握りつぶし、idle/isListening=falseに正しく戻す。二重start防止ガードも実装済み）
- 音声認識エラー表示（`VoiceErrorBanner`、event.error/event.messageをそのまま画面表示）
- `localStorage`保存（点検記録の永続化、リロード後も復元）
- 手動編集（`EditInspectionItem`、判定・位置・測定値・コメントをタップで修正可能）
- 未登録項目確認フロー（キュー方式、複数件あれば「ほかN件あります」表示）
- CO/HC入力（"CO"/"HC"を分割検出後にマージして数値抽出）
- 項目名向けの軽量fuzzy match（`lib/fuzzyMatch.ts`、編集距離1・同じ文字数の窓のみ・判定語には未適用・候補が複数項目に割れたら不採用）
- Vercel Preview Deployment運用（GitHub連携済み、pushで自動発火）

## 明日最初に対応する未実装要件

今日ユーザーから指定された内容。省略せずそのまま記録。

- タイロッドエンドブーツ「タイロットエンドブーツ」等の認識揺れ対応
- 「タイロッドブーツ」は項目エイリアスから削除
- ロアブーツ：ロワブーツ / ロアーブール / ロワーブーツ等の認識揺れ対応
- エアエレメント表示を「エアーエレメント」に戻す
- リヤブレーキパッドを「リヤブレーキパッド/ライニング」に変更
- brake_liningを統合
- タイヤ残量を「フロントタイヤ残量」「リヤタイヤ残量」に分離
- 最低地上高「最低地条項」等の誤認識対応
- 新規入力項目「（整備）」追加
- ドライブシャフトブーツで フロント/リヤ・右/左・インナー/アウター を正しく保持・表示
- `ItemPosition`に`leftRight`を追加
- 項目名より前にある位置語も解析する
- COとHCを別々に発話しても両方の数値を保持する
- `measurements`を上書きではなくマージする
- 全項目で音声文字起こしの軽微なズレを安全に吸収する
- fuzzy matchを長い項目名では編集距離2程度まで許可する
- 完全一致・明示エイリアスを必ずfuzzyより優先
- 誤った別項目への自動記録を避ける
- 保存済みlocalStorageデータと新しいDEFAULT_CHECKLISTを安全にreconcileする

## 絶対に維持する仕様

- 症状から整備判定を勝手に推測しない
- 判定語が明示されたときだけstatusを変更する
- 複数項目を連続して発話できる
- iPhone Safariで実際に使うことを前提にする
- 誤認識して別項目へ勝手に記録するより、認識しない方を優先する
- CO/HC、未登録項目、手動編集、localStorageを壊さない

## 明日の作業手順

1. `HANDOFF.md`を読む
2. `git status`と現在ブランチを確認
3. 現行コードを確認
4. 上記未実装要件を実装
5. 必須テスト
6. `npm run build`
7. 同じブランチ（`claude/auto-repair-voice-memo-pjt3oc`）へcommit/push
8. Vercel PreviewでiPhone Safari実機テスト
