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

export function getInspection(id: string): Inspection | undefined {
  return readAll().find((i) => i.id === id);
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

export function createInspection(customerName: string, vehicleModel: string): Inspection {
  const now = new Date().toISOString();
  const inspection: Inspection = {
    id: makeId(),
    customerName: customerName.trim() || undefined,
    vehicleModel: vehicleModel.trim() || undefined,
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
