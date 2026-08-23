/**
 * 音声入力で認識されやすい短い判定語の一覧を、マイクボタン付近に
 * 常時うっすら表示する。グローブ作業中でも一瞬見て確認できるように
 * するためのもので、タップ操作は不要。
 */
export function VoiceWordHint() {
  return (
    <div className="voice-word-hint">
      OK＝良好 ｜ NG＝不合格 ｜ 交換＝要交換 ｜ おすすめ ｜ なし＝該当なし ｜ 要望＝ご要望 ｜ 不要＝お客様不要
    </div>
  );
}
