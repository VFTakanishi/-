import { useState } from "react";
import { STATUS_COLOR_CLASS, STATUS_ORDER, statusLabel } from "../lib/format";
import type { InspectionItem, JudgementStatus } from "../types";

interface EditInspectionItemProps {
  item: InspectionItem;
  onSave: (updated: InspectionItem) => void;
  onClose: () => void;
  onDelete?: () => void;
}

export function EditInspectionItem({ item, onSave, onClose, onDelete }: EditInspectionItemProps) {
  const [status, setStatus] = useState<JudgementStatus>(item.status);
  const [frontRear, setFrontRear] = useState(item.position?.frontRear ?? "");
  const [leftRight, setLeftRight] = useState(item.position?.leftRight ?? "");
  const [innerOuter, setInnerOuter] = useState(item.position?.innerOuter ?? "");
  const [measurementValue, setMeasurementValue] = useState(
    item.measurement?.value !== undefined ? String(item.measurement.value) : ""
  );
  const [note, setNote] = useState(item.note ?? "");

  const handleSave = () => {
    const updated: InspectionItem = {
      ...item,
      status,
      note: note.trim() || undefined,
      position:
        frontRear || leftRight || innerOuter
          ? {
              frontRear: (frontRear || undefined) as "front" | "rear" | undefined,
              leftRight: (leftRight || undefined) as "left" | "right" | undefined,
              innerOuter: (innerOuter || undefined) as "inner" | "outer" | undefined,
            }
          : undefined,
      measurement: measurementValue.trim()
        ? { value: Number(measurementValue), unit: "mm" }
        : undefined,
      updatedAt: new Date().toISOString(),
    };
    onSave(updated);
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-panel" onClick={(e) => e.stopPropagation()}>
        <h2 className="modal-title">{item.category}</h2>

        <div className="modal-section">
          <div className="modal-section-title">判定</div>
          <div className="status-grid">
            {STATUS_ORDER.map((s) => (
              <button
                key={s}
                type="button"
                className={`status-choice ${STATUS_COLOR_CLASS[s]} ${status === s ? "status-choice--active" : ""}`}
                onClick={() => setStatus(s)}
              >
                {statusLabel(s)}
              </button>
            ))}
          </div>
        </div>

        <div className="modal-section">
          <div className="modal-section-title">位置</div>
          <div className="toggle-row">
            {(["", "front", "rear"] as const).map((v) => (
              <button
                key={v || "none"}
                type="button"
                className={`toggle-choice ${frontRear === v ? "toggle-choice--active" : ""}`}
                onClick={() => setFrontRear(v)}
              >
                {v === "" ? "指定なし" : v === "front" ? "フロント" : "リヤ"}
              </button>
            ))}
          </div>
          <div className="toggle-row">
            {(["", "left", "right"] as const).map((v) => (
              <button
                key={v || "none"}
                type="button"
                className={`toggle-choice ${leftRight === v ? "toggle-choice--active" : ""}`}
                onClick={() => setLeftRight(v)}
              >
                {v === "" ? "指定なし" : v === "left" ? "左" : "右"}
              </button>
            ))}
          </div>
          <div className="toggle-row">
            {(["", "inner", "outer"] as const).map((v) => (
              <button
                key={v || "none"}
                type="button"
                className={`toggle-choice ${innerOuter === v ? "toggle-choice--active" : ""}`}
                onClick={() => setInnerOuter(v)}
              >
                {v === "" ? "指定なし" : v === "inner" ? "インナー" : "アウター"}
              </button>
            ))}
          </div>
        </div>

        <div className="modal-section">
          <div className="modal-section-title">測定値（mm）</div>
          <input
            className="text-input"
            type="number"
            inputMode="decimal"
            value={measurementValue}
            onChange={(e) => setMeasurementValue(e.target.value)}
            placeholder="例: 5"
          />
        </div>

        <div className="modal-section">
          <div className="modal-section-title">コメント</div>
          <textarea
            className="text-input textarea"
            value={note}
            onChange={(e) => setNote(e.target.value)}
            placeholder="異常内容や補足など"
          />
        </div>

        <div className="modal-actions">
          <button type="button" className="big-button big-button--primary" onClick={handleSave}>
            保存
          </button>
          <button type="button" className="big-button" onClick={onClose}>
            キャンセル
          </button>
          {onDelete && (
            <button type="button" className="big-button big-button--danger" onClick={onDelete}>
              この項目を削除
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
