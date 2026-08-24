import { useState } from "react";
import { InspectionItemRow } from "../components/InspectionItemRow";
import { buildSummaryText } from "../lib/format";
import type { Inspection } from "../types";

interface SummaryScreenProps {
  inspection: Inspection;
  onBack: () => void;
  onBackToStart: () => void;
}

const PROBLEM_STATUSES = new Set(["ng", "replace_strong", "recommend", "maintenance", "customer_request", "customer_declined"]);

export function SummaryScreen({ inspection, onBack, onBackToStart }: SummaryScreenProps) {
  const [copied, setCopied] = useState(false);
  const [showAll, setShowAll] = useState(false);

  const summaryText = buildSummaryText(inspection);
  const problemItems = inspection.items.filter((i) => PROBLEM_STATUSES.has(i.status));

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(summaryText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      window.prompt("コピーできませんでした。以下のテキストを手動でコピーしてください。", summaryText);
    }
  };

  return (
    <div className="screen">
      <div className="screen-header">
        <button type="button" className="link-button" onClick={onBack}>
          ← 点検に戻る
        </button>
      </div>
      <h1 className="screen-title">見積メモ サマリー</h1>
      <div className="empty-note">
        {inspection.customerName || "お客様名未入力"}
        {inspection.vehicleModel ? ` / ${inspection.vehicleModel}` : ""}
      </div>

      <div className="card">
        <pre className="summary-text">{summaryText}</pre>
        <button type="button" className="big-button big-button--primary" onClick={handleCopy}>
          {copied ? "コピーしました" : "クリップボードにコピー"}
        </button>
      </div>

      {problemItems.length === 0 && <div className="empty-note">問題項目はありません。</div>}

      <button type="button" className="link-button" onClick={() => setShowAll((v) => !v)}>
        {showAll ? "全項目一覧を閉じる" : "全項目一覧を見る"}
      </button>

      {showAll && (
        <div className="item-list">
          {inspection.items.map((item) => (
            <InspectionItemRow key={item.id} item={item} onClick={() => {}} />
          ))}
        </div>
      )}

      <button type="button" className="big-button" onClick={onBackToStart}>
        点検一覧に戻る
      </button>
    </div>
  );
}
