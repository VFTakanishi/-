import { useState } from "react";
import { createInspection, deleteInspection, listInspections } from "../lib/storage";
import type { Inspection } from "../types";

interface StartScreenProps {
  onOpenInspection: (id: string) => void;
}

function formatDate(iso: string): string {
  const d = new Date(iso);
  return `${d.getMonth() + 1}/${d.getDate()} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

export function StartScreen({ onOpenInspection }: StartScreenProps) {
  const [customerName, setCustomerName] = useState("");
  const [vehicleModel, setVehicleModel] = useState("");
  const [inspections, setInspections] = useState<Inspection[]>(() => listInspections());

  const handleStart = () => {
    const inspection = createInspection(customerName, vehicleModel);
    onOpenInspection(inspection.id);
  };

  const handleDelete = (id: string) => {
    if (!window.confirm("この点検記録を削除しますか？")) return;
    deleteInspection(id);
    setInspections(listInspections());
  };

  return (
    <div className="screen">
      <h1 className="screen-title">車検点検メモ</h1>

      <div className="card">
        <div className="modal-section-title">新規点検を開始</div>
        <input
          className="text-input"
          placeholder="お客様名（任意）"
          value={customerName}
          onChange={(e) => setCustomerName(e.target.value)}
        />
        <input
          className="text-input"
          placeholder="車種（任意）"
          value={vehicleModel}
          onChange={(e) => setVehicleModel(e.target.value)}
        />
        <button type="button" className="big-button big-button--primary" onClick={handleStart}>
          点検を開始
        </button>
      </div>

      <div className="modal-section-title">過去の点検</div>
      {inspections.length === 0 && <div className="empty-note">まだ点検記録はありません。</div>}
      <div className="past-list">
        {inspections.map((inspection) => (
          <div key={inspection.id} className="past-item">
            <button type="button" className="past-item-main" onClick={() => onOpenInspection(inspection.id)}>
              <div className="past-item-title">
                {inspection.customerName || "お客様名未入力"}
                {inspection.vehicleModel ? ` / ${inspection.vehicleModel}` : ""}
              </div>
              <div className="past-item-date">{formatDate(inspection.updatedAt)}</div>
            </button>
            <button type="button" className="past-item-delete" onClick={() => handleDelete(inspection.id)}>
              削除
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
