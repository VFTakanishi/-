import type { ParsedUnmatched } from "../lib/parseVoiceInspection";

export interface RecognitionResultData {
  transcript: string;
  matchedItems?: Array<{ category: string; lines: string[] }>;
}

interface RecognitionResultProps {
  result: RecognitionResultData | null;
  /** unmatchedが出ても点検作業を止めないための、控えめなインライン表示。自動でモーダルは開かない。 */
  pendingUnmatched: ParsedUnmatched | null;
  pendingUnmatchedCount: number;
  onRequestAdd: () => void;
}

/**
 * 音声認識結果を表示する固定高さの確認欄。
 * unmatched（未登録項目候補）が出ても自動でモーダルは表示しない。代わりに
 * この固定欄の中に小さく「未認識：◯◯」＋「追加」ボタンを表示するだけに留め、
 * ユーザーが明示的に「追加」をタップした場合だけ呼び出し元で確認モーダルを開く。
 * これにより、聞き間違い程度のunmatchedでは点検作業（音声入力・スクロール位置・
 * 全項目一覧の表示位置）が一切妨げられない。
 * 内容量に関わらず外側の高さは常に一定で、はみ出す分は内部スクロールする。
 */
export function RecognitionResult({ result, pendingUnmatched, pendingUnmatchedCount, onRequestAdd }: RecognitionResultProps) {
  if (!result && !pendingUnmatched) {
    return (
      <div className="recognition-result recognition-result--empty">
        マイクをタップして話しかけてください。
      </div>
    );
  }

  return (
    <div className="recognition-result">
      <div className="recognition-result-scroll">
        {result && (
          <div className="recognition-result-transcript">
            <span className="recognition-result-label">認識</span>
            <span>「{result.transcript}」</span>
          </div>
        )}

        {result?.matchedItems && result.matchedItems.length > 0 && (
          <div className="recognition-result-record">
            <span className="recognition-result-label">
              記録{result.matchedItems.length > 1 ? `（${result.matchedItems.length}件）` : ""}
            </span>
            {result.matchedItems.map((matched, index) => (
              <div key={`${matched.category}-${index}`} className="recognition-result-record-item">
                {matched.lines.map((line) => (
                  <div key={line}>{line}</div>
                ))}
              </div>
            ))}
          </div>
        )}

        {pendingUnmatched && (
          <div className="recognition-result-unmatched-hint">
            <span>
              未認識：{pendingUnmatched.customCategoryName || "（項目名なし）"}
              {pendingUnmatchedCount > 1 ? `　ほか${pendingUnmatchedCount - 1}件` : ""}
            </span>
            <button type="button" className="recognition-result-unmatched-add" onClick={onRequestAdd}>
              追加
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
