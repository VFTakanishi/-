# 引き継ぎメモ（車検整備 音声点検メモツール）

最終更新: 2026-08-24（前回の未実装要件を実装完了した時点）

## 現在地

- **ブランチ**: `claude/auto-repair-voice-memo-pjt3oc`
- **`npm run build`**: 成功（型エラーなし）
- **Vercelプロジェクト名**: `shaken-voice-inspection`（Team: VFTakanishi / scope: vft-akanishi、Preview Deployment運用、production/mainには一切デプロイ・mergeしていない）
- **iPhone Safari実機テスト中**。今回の変更はローカルの単体テスト・Playwrightでのブラウザ検証のみ実施済みで、**実機での確認はまだ**。

## 現在までに完成している重要機能

- 音声点検メモ本体（新規点検開始・お客様名/車種入力・過去点検一覧/再開/削除）
- 複数項目の連続発話解析（`parseVoiceInspections()`）: 登録済み項目エイリアスの出現順にセグメント分割し、後続項目の判定語が前の項目に混入しない
- **項目名の前にある位置語も解析**（例:「リヤドライブシャフトブーツ」の「リヤ」）。前後・左右・インナーアウターを組み合わせて保持
- `ItemPosition`に`leftRight`（右/左）を追加。手動編集UIにもトグルを追加
- CO/HCは**別々の発話でもmeasurementsをマージ**して両方保持（上書きしない）
- 判定語の音声エイリアス（OK/NG/交換/推奨/なし/要望/不要 等）
- iPhone Safari対応（`webkitSpeechRecognition`の非continuousセッション＋onend自動再開による疑似連続リスニング）
- 手動マイク停止時の誤エラー対策（`manualStopRef`）、二重start防止ガード
- 音声認識エラー表示（`VoiceErrorBanner`）
- `localStorage`保存、**保存済みデータと新しいDEFAULT_CHECKLISTの安全なreconcile**（`reconcileItems`、削除/統合されたidでもデータがあればカスタム項目として保持、空なら破棄）
- 手動編集、未登録項目確認フロー（キュー方式）
- 項目名向けfuzzy match（`lib/fuzzyMatch.ts`、**Hamming距離**・同じ文字数の窓のみ・短い項目名は距離1まで/8文字以上の長い項目名は距離2まで・複数項目に候補が割れたら不採用）
- Vercel Preview Deployment運用

## 前回指定された未実装要件 → 本日すべて実装済み

- タイロッドエンドブーツ「タイロットエンドブーツ」等の認識揺れ対応 ✅（「タイロッドブーツ」は削除）
- ロアブーツ：ロワブーツ / ロアーブール / ロワーブーツ等の認識揺れ対応 ✅
- エアエレメント表示を「エアーエレメント」に戻す ✅
- リヤブレーキパッドを「リヤブレーキパッド/ライニング」に変更、brake_liningを統合 ✅
- タイヤ残量を「フロントタイヤ残量」「リヤタイヤ残量」に分離（`tire_front`/`tire_rear`） ✅
- 最低地上高「最低地条項」等の誤認識対応 ✅
- 新規入力項目「（整備）」追加 ✅（**要確認**: 意図が明示されていなかったため、特定部品に紐づかない汎用の整備メモ項目`maintenance_note`（エイリアス「整備」）として実装。意図と異なる場合は修正が必要）
- ドライブシャフトブーツでフロント/リヤ・右/左・インナー/アウターを保持・表示 ✅
- `ItemPosition`にleftRight追加 ✅
- 項目名より前にある位置語も解析 ✅
- CO/HCを別々に発話しても両方保持、measurementsをマージ ✅
- fuzzy matchを長い項目名では距離2まで許可 ✅（**実装中に重大なバグを発見・修正**: Levenshtein距離のまま距離2を許すと「1文字分ずれた窓」が常に距離2で誤一致する問題があり、Hamming距離＋8文字以上のみ距離2、に変更して解消）
- localStorageと新DEFAULT_CHECKLISTの安全なreconcile ✅

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
3. Vercel PreviewでiPhone Safari実機テスト（今回の変更は実機未検証）
4. 「（整備）」項目の意図を確認・必要なら修正
5. 実機で見つかった不具合があれば対応
