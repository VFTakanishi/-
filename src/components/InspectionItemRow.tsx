import { STATUS_COLOR_CLASS, itemSummaryLine, statusLabel } from "../lib/format";
import type { InspectionItem } from "../types";

interface InspectionItemRowProps {
  item: InspectionItem;
  onClick: () => void;
}

export function InspectionItemRow({ item, onClick }: InspectionItemRowProps) {
  const summary = itemSummaryLine(item);
  return (
    <button type="button" className="item-row" onClick={onClick}>
      <div className="item-row-main">
        <span className="item-row-category">
          {item.category}
          {item.isCustom && <span className="item-row-custom-tag">追加項目</span>}
        </span>
        {summary && <span className="item-row-summary">{summary}</span>}
      </div>
      <span className={`status-badge ${STATUS_COLOR_CLASS[item.status]}`}>{statusLabel(item.status)}</span>
    </button>
  );
}
