import { statusLabel } from "../lib/format";
import type { ParsedUnmatched } from "../lib/parseVoiceInspection";

export interface RecognitionResultData {
  transcript: string;
  matchedItems?: Array<{ category: string; lines: string[] }>;
}

interface RecognitionResultProps {
  result: RecognitionResultData | null;
  pendingUnmatched: ParsedUnmatched | null;
  pendingUnmatchedCount: number;
  onConfirmAdd: () => void;
  onDiscard: () => void;
}

export function RecognitionResult({
  result,
  pendingUnmatched,
  pendingUnmatchedCount,
  onConfirmAdd,
  onDiscard,
}: RecognitionResultProps) {
  if (!result && !pendingUnmatched) {
    return (
      <div className="recognition-result recognition-result--empty">
        マイクをタップして話しかけてください。
      </div>
    );
  }

  return (
    <div className="recognition-result">
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
        <div className="recognition-result-unmatched">
          <div className="recognition-result-label">
            未登録項目の可能性{pendingUnmatchedCount > 1 ? `（ほか${pendingUnmatchedCount - 1}件あります）` : ""}
          </div>
          <div className="recognition-result-unmatched-name">{pendingUnmatched.customCategoryName}</div>
          {pendingUnmatched.status && <div>判定: {statusLabel(pendingUnmatched.status)}</div>}
          <div className="recognition-result-actions">
            <button type="button" className="big-button big-button--primary" onClick={onConfirmAdd}>
              未登録項目として追加
            </button>
            <button type="button" className="big-button" onClick={onDiscard}>
              追加しない
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
