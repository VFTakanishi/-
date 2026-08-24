export interface RecognitionResultData {
  transcript: string;
  matchedItems?: Array<{ category: string; lines: string[] }>;
}

interface RecognitionResultProps {
  result: RecognitionResultData | null;
}

/**
 * 通常の音声認識結果だけを表示する固定高さの確認欄。
 * 未登録項目の確認UIはここには含めない（別コンポーネントの
 * オーバーレイとして表示し、全項目一覧の位置に影響させないため）。
 * 内容量に関わらず外側の高さは常に一定で、はみ出す分は内部スクロールする。
 */
export function RecognitionResult({ result }: RecognitionResultProps) {
  if (!result) {
    return (
      <div className="recognition-result recognition-result--empty">
        マイクをタップして話しかけてください。
      </div>
    );
  }

  return (
    <div className="recognition-result">
      <div className="recognition-result-scroll">
        <div className="recognition-result-transcript">
          <span className="recognition-result-label">認識</span>
          <span>「{result.transcript}」</span>
        </div>

        {result.matchedItems && result.matchedItems.length > 0 && (
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
      </div>
    </div>
  );
}
