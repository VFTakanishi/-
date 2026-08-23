import { statusLabel } from "../lib/format";
import type { JudgementStatus } from "../types";

export interface RecognitionResultData {
  transcript: string;
  matchedCategory?: string;
  lines?: string[];
  unmatched?: {
    customCategoryName: string;
    status?: JudgementStatus;
  };
}

interface RecognitionResultProps {
  result: RecognitionResultData | null;
  onConfirmAdd: () => void;
  onDiscard: () => void;
}

export function RecognitionResult({ result, onConfirmAdd, onDiscard }: RecognitionResultProps) {
  if (!result) {
    return (
      <div className="recognition-result recognition-result--empty">
        マイクをタップして話しかけてください。
      </div>
    );
  }

  return (
    <div className="recognition-result">
      <div className="recognition-result-transcript">
        <span className="recognition-result-label">認識</span>
        <span>「{result.transcript}」</span>
      </div>

      {result.unmatched ? (
        <div className="recognition-result-unmatched">
          <div className="recognition-result-label">未登録項目の可能性</div>
          <div className="recognition-result-unmatched-name">{result.unmatched.customCategoryName}</div>
          {result.unmatched.status && (
            <div>判定: {statusLabel(result.unmatched.status)}</div>
          )}
          <div className="recognition-result-actions">
            <button type="button" className="big-button big-button--primary" onClick={onConfirmAdd}>
              未登録項目として追加
            </button>
            <button type="button" className="big-button" onClick={onDiscard}>
              追加しない
            </button>
          </div>
        </div>
      ) : (
        <div className="recognition-result-record">
          <span className="recognition-result-label">記録</span>
          {(result.lines ?? []).map((line) => (
            <div key={line}>{line}</div>
          ))}
        </div>
      )}
    </div>
  );
}
