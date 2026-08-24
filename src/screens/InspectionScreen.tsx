import { useEffect, useState } from "react";
import { EditInspectionItem } from "../components/EditInspectionItem";
import { InspectionItemRow } from "../components/InspectionItemRow";
import { RecognitionResult, type RecognitionResultData } from "../components/RecognitionResult";
import { UnmatchedConfirmModal } from "../components/UnmatchedConfirmModal";
import { VoiceButton } from "../components/VoiceButton";
import { VoiceErrorBanner } from "../components/VoiceErrorBanner";
import { VoiceWordHint } from "../components/VoiceWordHint";
import { useVoiceRecognition } from "../hooks/useVoiceRecognition";
import { itemSummaryLine, statusLabel } from "../lib/format";
import { chooseBestTranscript, type ParsedMatched, type ParsedUnmatched } from "../lib/parseVoiceInspection";
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

// unmatchedは自動でモーダルを出さず、RecognitionResultの固定欄に小さく
// ヒント表示するだけにする。無限に蓄積させる必要はないため、同じ未認識文字列は
// 積み増さず、最新の候補を優先して最大UNMATCHED_QUEUE_LIMIT件までに留める。
const UNMATCHED_QUEUE_LIMIT = 3;

function appendUnmatchedQueue(prev: ParsedUnmatched[], additions: ParsedUnmatched[]): ParsedUnmatched[] {
  const seen = new Set(prev.map((u) => u.customCategoryName));
  const merged = [...prev];
  for (const addition of additions) {
    if (seen.has(addition.customCategoryName)) continue;
    seen.add(addition.customCategoryName);
    merged.push(addition);
  }
  return merged.length > UNMATCHED_QUEUE_LIMIT ? merged.slice(merged.length - UNMATCHED_QUEUE_LIMIT) : merged;
}

const ELECTRICAL_NOTE_SEPARATOR = " / ";

/**
 * 電気回りは1台の車で複数箇所の灯火不具合等が同時に存在しうるため、
 * noteを上書きせず区切り文字で追記する。同一内容の重複追加は避ける。
 */
function appendElectricalNote(existingNote: string | undefined, newDefect: string): string | undefined {
  const trimmed = newDefect.trim();
  if (!trimmed) return existingNote;
  const existingList = existingNote
    ? existingNote.split(ELECTRICAL_NOTE_SEPARATOR).map((s) => s.trim()).filter(Boolean)
    : [];
  if (existingList.includes(trimmed)) return existingNote;
  return [...existingList, trimmed].join(ELECTRICAL_NOTE_SEPARATOR);
}

function applyParsedToItem(item: InspectionItem, parsed: ParsedMatched, now: string): InspectionItem {
  return {
    ...item,
    status: parsed.status ?? item.status,
    position: parsed.position ?? item.position,
    measurement: parsed.measurement ?? item.measurement,
    measurements: parsed.measurements ? { ...item.measurements, ...parsed.measurements } : item.measurements,
    note:
      item.id === "electrical" && parsed.note
        ? appendElectricalNote(item.note, parsed.note)
        : parsed.note ?? item.note,
    updatedAt: now,
  };
}

export function InspectionScreen({ inspectionId, onOpenSummary, onBackToStart }: InspectionScreenProps) {
  const [inspection, setInspection] = useState<Inspection | null>(() => getInspection(inspectionId) ?? null);
  const [editingItemId, setEditingItemId] = useState<string | null>(null);
  const [lastRecognition, setLastRecognition] = useState<RecognitionResultData | null>(null);
  const [pendingUnmatchedQueue, setPendingUnmatchedQueue] = useState<ParsedUnmatched[]>([]);
  const [showUnmatchedModal, setShowUnmatchedModal] = useState(false);
  const pendingUnmatched = pendingUnmatchedQueue[0] ?? null;

  useEffect(() => {
    if (inspection) saveInspection(inspection);
  }, [inspection]);

  const handleVoiceResult = (transcripts: string[]) => {
    const { transcript, results: parsedList } = chooseBestTranscript(transcripts);
    const matchedList = parsedList.filter((p): p is ParsedMatched => p.matched);
    const unmatchedList = parsedList.filter((p): p is ParsedUnmatched => !p.matched);
    const now = new Date().toISOString();

    if (matchedList.length > 0) {
      setInspection((prev) => {
        if (!prev) return prev;
        let items = prev.items;
        for (const parsed of matchedList) {
          items = items.map((item) => (item.id === parsed.itemId ? applyParsedToItem(item, parsed, now) : item));
        }
        return { ...prev, items, updatedAt: now };
      });
    }

    if (unmatchedList.length > 0) {
      setPendingUnmatchedQueue((prev) => appendUnmatchedQueue(prev, unmatchedList));
    }

    // 表示用のサマリーは現在のスナップショットから算出する（保存処理とは独立した副作用のない計算）
    const matchedItems = inspection
      ? matchedList.flatMap((parsed) => {
          const current = inspection.items.find((i) => i.id === parsed.itemId);
          if (!current) return [];
          const updated = applyParsedToItem(current, parsed, now);
          return [{ category: updated.category, lines: buildRecordLines(updated) }];
        })
      : [];

    setLastRecognition({ transcript, matchedItems: matchedItems.length > 0 ? matchedItems : undefined });
  };

  const { state: voiceState, isListening, lastError, toggle } = useVoiceRecognition({ onResult: handleVoiceResult });

  const handleConfirmAdd = () => {
    if (!pendingUnmatched) return;
    const now = new Date().toISOString();
    const newItem: InspectionItem = {
      id: makeItemId(),
      category: pendingUnmatched.customCategoryName,
      status: pendingUnmatched.status ?? "unset",
      isCustom: true,
      updatedAt: now,
    };
    setInspection((prev) => {
      if (!prev) return prev;
      return { ...prev, items: [...prev.items, newItem], updatedAt: now };
    });
    setLastRecognition({
      transcript: pendingUnmatched.rawText,
      matchedItems: [{ category: newItem.category, lines: buildRecordLines(newItem) }],
    });
    setPendingUnmatchedQueue((prev) => prev.slice(1));
    setShowUnmatchedModal(false);
  };

  const handleDiscardUnmatched = () => {
    setPendingUnmatchedQueue((prev) => prev.slice(1));
    setShowUnmatchedModal(false);
  };

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
        {inspection.mileage ? ` / ${inspection.mileage}km` : ""}
      </div>

      <VoiceButton state={voiceState} isListening={isListening} onClick={toggle} />
      <VoiceWordHint />
      <VoiceErrorBanner error={lastError} />

      <RecognitionResult
        result={lastRecognition}
        pendingUnmatched={pendingUnmatched}
        pendingUnmatchedCount={pendingUnmatchedQueue.length}
        onRequestAdd={() => setShowUnmatchedModal(true)}
      />

      {showUnmatchedModal && pendingUnmatched && (
        <UnmatchedConfirmModal
          pendingUnmatched={pendingUnmatched}
          pendingUnmatchedCount={pendingUnmatchedQueue.length}
          onConfirmAdd={handleConfirmAdd}
          onDiscard={handleDiscardUnmatched}
        />
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
