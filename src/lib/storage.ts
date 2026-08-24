import { DEFAULT_CHECKLIST } from "../data/defaultChecklist";
import type { Inspection, InspectionItem } from "../types";

const STORAGE_KEY = "seibi_inspections_v1";

function readAll(): Inspection[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function writeAll(inspections: Inspection[]): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(inspections));
}

export function listInspections(): Inspection[] {
  return readAll().sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
}

/**
 * 保存済みのInspectionItem配列を、現在のDEFAULT_CHECKLISTの形に安全に
 * 追従させる。チェックリストの項目構成（id・表示名）は今後も変わりうるため、
 * 古いデータを読み込んだときに記録済みの内容を失わないようにする。
 *
 * - 現行チェックリストに存在するidはそのまま記録を維持し、表示名だけ最新に更新
 * - 現行チェックリストに存在しないidは、新しいidに置き換わった旧項目
 *   （例: 旧"tire"→新"tire_front"/"tire_rear"、旧"brake_lining"→"brake_pad_rear"に統合）
 *   の可能性があるため、実際に記録（判定・数値・位置・コメント）が入っている場合だけ
 *   カスタム項目として残し、未記録（unsetかつ空）のものは黙って破棄する
 */
export function reconcileItems(items: InspectionItem[]): InspectionItem[] {
  const byId = new Map(items.map((item) => [item.id, item]));
  const reconciled: InspectionItem[] = [];

  for (const def of DEFAULT_CHECKLIST) {
    const existing = byId.get(def.id);
    if (existing) {
      reconciled.push({ ...existing, category: def.category });
      byId.delete(def.id);
    } else {
      reconciled.push({ id: def.id, category: def.category, status: "unset" });
    }
  }

  for (const orphan of byId.values()) {
    const hasData =
      orphan.status !== "unset" ||
      !!orphan.note ||
      !!orphan.measurement ||
      !!orphan.measurements ||
      !!orphan.position;
    if (hasData) {
      reconciled.push({ ...orphan, isCustom: true });
    }
  }

  return reconciled;
}

export function getInspection(id: string): Inspection | undefined {
  const inspection = readAll().find((i) => i.id === id);
  if (!inspection) return undefined;
  return { ...inspection, items: reconcileItems(inspection.items) };
}

export function saveInspection(inspection: Inspection): void {
  const all = readAll();
  const index = all.findIndex((i) => i.id === inspection.id);
  if (index >= 0) {
    all[index] = inspection;
  } else {
    all.push(inspection);
  }
  writeAll(all);
}

export function deleteInspection(id: string): void {
  writeAll(readAll().filter((i) => i.id !== id));
}

function makeId(): string {
  return `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

export function createDefaultItems(): InspectionItem[] {
  return DEFAULT_CHECKLIST.map((def) => ({
    id: def.id,
    category: def.category,
    status: "unset",
  }));
}

/** 「123,456」等の入力でも壊れないよう数字だけを残して保存する（number型にはしない） */
function sanitizeMileage(raw: string): string | undefined {
  const digitsOnly = raw.replace(/[^0-9]/g, "");
  return digitsOnly.length > 0 ? digitsOnly : undefined;
}

export function createInspection(customerName: string, vehicleModel: string, mileage = ""): Inspection {
  const now = new Date().toISOString();
  const inspection: Inspection = {
    id: makeId(),
    customerName: customerName.trim() || undefined,
    vehicleModel: vehicleModel.trim() || undefined,
    mileage: sanitizeMileage(mileage),
    createdAt: now,
    updatedAt: now,
    items: createDefaultItems(),
  };
  saveInspection(inspection);
  return inspection;
}

export function makeItemId(): string {
  return `custom_${makeId()}`;
}
