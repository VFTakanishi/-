# 引き継ぎメモ（車検整備 音声点検メモツール）

最終更新: 2026-08-24（iPhone Safari実機フィードバック第2弾を実装完了した時点）

## 現在地

- **ブランチ**: `claude/auto-repair-voice-memo-pjt3oc`
- **`npm run build`**: 成功（型エラーなし）
- **Vercelプロジェクト名**: `shaken-voice-inspection`（Team: VFTakanishi / scope: vft-akanishi、Preview Deployment運用、production/mainには一切デプロイ・mergeしていない）
- **iPhone Safari実機テスト中**。今回の変更もローカルの単体テスト・Playwrightでのブラウザ検証のみ実施済みで、**実機での確認はまだ**。

## 現在までに完成している重要機能

- 音声点検メモ本体（新規点検開始・お客様名/車種入力・過去点検一覧/再開/削除）
- 複数項目の連続発話解析（`parseVoiceInspections()`）: 登録済み項目エイリアスの出現順にセグメント分割し、後続項目の判定語が前の項目に混入しない
- 項目名の前にある位置語も解析（例:「リヤドライブシャフトブーツ」の「リヤ」）。前後・左右・インナーアウターを組み合わせて保持
- `ItemPosition`に`leftRight`（右/左）を追加。手動編集UIにもトグルを追加
- CO/HCは別々の発話でもmeasurementsをマージして両方保持（上書きしない）
- **電気回り（`electrical`）はfree-form defect list**: 複数の灯火不具合等を「/」区切りでnoteに追記（重複は追加しない）。位置語（フロント/リヤ/リア/右/左等）は構造化せずnoteの文言としてそのまま残す。ストップランプ/ブレーキランプ/テールランプ/ウインカー/ウィンカー/ヘッドライト/ヘッドランプ/スモールランプ/バックランプ/ナンバー灯/ライセンスランプの各灯火名を代表エイリアスとして持ち、「電気回り」を言わなくても認識できる
- **`JudgementStatus`に`maintenance`（表示名「整備」）を追加**。7判定+整備+未確認の計9状態。STATUS_ORDER/STATUS_COLOR_CLASS/SUMMARY_GROUPS/VoiceWordHint/判定語エイリアス（「整備」→maintenance）すべて対応済み。サマリーに【整備】グループとして表示
- 判定語の音声エイリアス（OK/NG/交換/推奨/整備/なし/要望/不要 等）
- iPhone Safari対応（`webkitSpeechRecognition`の非continuousセッション＋onend自動再開による疑似連続リスニング）
- 手動マイク停止時の誤エラー対策（`manualStopRef`）、二重start防止ガード
- 音声認識エラー表示（`VoiceErrorBanner`）
- `localStorage`保存、保存済みデータと新しいDEFAULT_CHECKLISTの安全なreconcile（`reconcileItems`、削除/統合されたidでもデータがあればカスタム項目として保持、空なら破棄）
- 手動編集、未登録項目確認フロー（キュー方式）
- 項目名向けfuzzy match（`lib/fuzzyMatch.ts`、Hamming距離・同じ文字数の窓のみ・短い項目名は距離1まで/8文字以上の長い項目名は距離2まで・複数項目に候補が割れたら不採用）
- Vercel Preview Deployment運用

## 本日のiPhone実機フィードバック → すべて実装済み

- 電気回りで複数の不具合メモを追記できるようにする ✅（位置語はnoteにそのまま残す、重複は追加しない、灯火名エイリアス追加）
- ステアリングラックブーツ（`steering_rack_boot`）を点検項目から削除 ✅（reconcileItemsが既存データを安全に保持）
- 「（整備）」を点検項目から削除し、`JudgementStatus`の`maintenance`として追加 ✅（`maintenance_note`項目は削除。reconcileItemsで旧データは他項目へ移行せず、記録があればlegacy/custom項目として保持のみ）
- localStorage移行 ✅（既存のreconcileItemsの仕組みで対応、追加コードなし）

## 絶対に維持する仕様

- 症状から整備判定を勝手に推測しない
- 判定語が明示されたときだけstatusを変更する
- 複数項目を連続して発話できる
- iPhone Safariで実際に使うことを前提にする
- 誤認識して別項目へ勝手に記録するより、認識しない方を優先する
- CO/HC、未登録項目、手動編集、localStorageを壊さない

## 次回最初に対応すること

1. `HANDOFF.md`を読む
2. `git status`と現在ブランチを確認
3. Vercel PreviewでiPhone Safari実機テスト（本日の変更は実機未検証）
4. 実機で見つかった不具合があれば対応
