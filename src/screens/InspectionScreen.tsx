import { useEffect, useMemo, useState } from "react";
import { EditInspectionItem } from "../components/EditInspectionItem";
import { InspectionItemRow } from "../components/InspectionItemRow";
import { RecognitionResult, type RecognitionResultData } from "../components/RecognitionResult";
import { VoiceButton } from "../components/VoiceButton";
import { VoiceErrorBanner } from "../components/VoiceErrorBanner";
import { VoiceWordHint } from "../components/VoiceWordHint";
import { useVoiceRecognition } from "../hooks/useVoiceRecognition";
import { itemSummaryLine, statusLabel } from "../lib/format";
import { parseVoiceInspection, type ParsedUnmatched } from "../lib/parseVoiceInspection";
import { getInspection, makeItemId, saveInspection } from "../lib/storage";
import type { Inspection, InspectionItem } from "../types";

interface InspectionScreenProps {
  inspectionId: string;
  onOpenSummary: () => void;
  onBackToStart: () => void;
}

function buildRecordLines(item: InspectionItem): string[] {
  const lines: string[] = [item.category];
  const summary = itemSummaryLine(item);
  if (summary) lines.push(summary);
  lines.push(`判定: ${statusLabel(item.status)}`);
  return lines;
}

export function InspectionScreen({ inspectionId, onOpenSummary, onBackToStart }: InspectionScreenProps) {
  const [inspection, setInspection] = useState<Inspection | null>(() => getInspection(inspectionId) ?? null);
  const [editingItemId, setEditingItemId] = useState<string | null>(null);
  const [lastRecognition, setLastRecognition] = useState<RecognitionResultData | null>(null);
  const [pendingUnmatched, setPendingUnmatched] = useState<ParsedUnmatched | null>(null);

  useEffect(() => {
    if (inspection) saveInspection(inspection);
  }, [inspection]);

  const handleVoiceResult = (transcript: string) => {
    const parsed = parseVoiceInspection(transcript);

    if (parsed.matched) {
      setPendingUnmatched(null);
      setInspection((prev) => {
        if (!prev) return prev;
        const now = new Date().toISOString();
        const items = prev.items.map((item) => {
          if (item.id !== parsed.itemId) return item;
          const updated: InspectionItem = {
            ...item,
            status: parsed.status ?? item.status,
            position: parsed.position ?? item.position,
            measurement: parsed.measurement ?? item.measurement,
            measurements: parsed.measurements ?? item.measurements,
            note: parsed.note ?? item.note,
            updatedAt: now,
          };
          setLastRecognition({ transcript, matchedCategory: item.category, lines: buildRecordLines(updated) });
          return updated;
        });
        return { ...prev, items, updatedAt: now };
      });
    } else {
      setPendingUnmatched(parsed);
      setLastRecognition({
        transcript,
        unmatched: { customCategoryName: parsed.customCategoryName, status: parsed.status },
      });
    }
  };

  const { state: voiceState, isListening, lastError, toggle } = useVoiceRecognition({ onResult: handleVoiceResult });

  const handleConfirmAdd = () => {
    if (!pendingUnmatched) return;
    setInspection((prev) => {
      if (!prev) return prev;
      const now = new Date().toISOString();
      const newItem: InspectionItem = {
        id: makeItemId(),
        category: pendingUnmatched.customCategoryName,
        status: pendingUnmatched.status ?? "unset",
        isCustom: true,
        updatedAt: now,
      };
      setLastRecognition({
        transcript: pendingUnmatched.rawText,
        matchedCategory: newItem.category,
        lines: buildRecordLines(newItem),
      });
      return { ...prev, items: [...prev.items, newItem], updatedAt: now };
    });
    setPendingUnmatched(null);
  };

  const handleDiscardUnmatched = () => {
    setPendingUnmatched(null);
    setLastRecognition(null);
  };

  const recentItems = useMemo(() => {
    if (!inspection) return [];
    return [...inspection.items]
      .filter((i) => i.updatedAt)
      .sort((a, b) => (b.updatedAt ?? "").localeCompare(a.updatedAt ?? ""))
      .slice(0, 5);
  }, [inspection]);

  const editingItem = inspection?.items.find((i) => i.id === editingItemId) ?? null;

  const handleSaveEdit = (updated: InspectionItem) => {
    setInspection((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        items: prev.items.map((i) => (i.id === updated.id ? updated : i)),
        updatedAt: new Date().toISOString(),
      };
    });
    setEditingItemId(null);
  };

  const handleDeleteEdit = () => {
    if (!editingItem) return;
    setInspection((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        items: prev.items.filter((i) => i.id !== editingItem.id),
        updatedAt: new Date().toISOString(),
      };
    });
    setEditingItemId(null);
  };

  if (!inspection) {
    return (
      <div className="screen">
        <div className="empty-note">点検記録が見つかりませんでした。</div>
        <button type="button" className="big-button" onClick={onBackToStart}>
          点検一覧に戻る
        </button>
      </div>
    );
  }

  return (
    <div className="screen">
      <div className="screen-header">
        <button type="button" className="link-button" onClick={onBackToStart}>
          ← 点検一覧
        </button>
        <button type="button" className="link-button" onClick={onOpenSummary}>
          サマリーを見る →
        </button>
      </div>
      <div className="empty-note">
        {inspection.customerName || "お客様名未入力"}
        {inspection.vehicleModel ? ` / ${inspection.vehicleModel}` : ""}
      </div>

      <VoiceButton state={voiceState} isListening={isListening} onClick={toggle} />
      <VoiceWordHint />
      <VoiceErrorBanner error={lastError} />

      <RecognitionResult result={lastRecognition} onConfirmAdd={handleConfirmAdd} onDiscard={handleDiscardUnmatched} />

      {recentItems.length > 0 && (
        <>
          <div className="modal-section-title">最近の記録</div>
          <div className="item-list">
            {recentItems.map((item) => (
              <InspectionItemRow key={item.id} item={item} onClick={() => setEditingItemId(item.id)} />
            ))}
          </div>
        </>
      )}

      <div className="modal-section-title">全項目一覧</div>
      <div className="item-list">
        {inspection.items.map((item) => (
          <InspectionItemRow key={item.id} item={item} onClick={() => setEditingItemId(item.id)} />
        ))}
      </div>

      {editingItem && (
        <EditInspectionItem
          item={editingItem}
          onSave={handleSaveEdit}
          onClose={() => setEditingItemId(null)}
          onDelete={editingItem.isCustom ? handleDeleteEdit : undefined}
        />
      )}
    </div>
  );
}
