import { STATUS_LABELS } from "../data/voiceAliases";
import type { Inspection, InspectionItem, JudgementStatus } from "../types";

export const STATUS_ORDER: JudgementStatus[] = [
  "ng",
  "replace_strong",
  "recommend",
  "customer_request",
  "customer_declined",
  "na",
  "good",
  "unset",
];

export const STATUS_COLOR_CLASS: Record<JudgementStatus, string> = {
  ng: "status-ng",
  replace_strong: "status-replace",
  recommend: "status-recommend",
  na: "status-na",
  good: "status-good",
  customer_request: "status-request",
  customer_declined: "status-declined",
  unset: "status-unset",
};

export function statusLabel(status: JudgementStatus): string {
  return STATUS_LABELS[status];
}

function positionLabel(item: InspectionItem): string {
  const parts: string[] = [];
  if (item.position?.frontRear === "front") parts.push("フロント");
  if (item.position?.frontRear === "rear") parts.push("リア");
  if (item.position?.innerOuter === "inner") parts.push("インナー");
  if (item.position?.innerOuter === "outer") parts.push("アウター");
  return parts.join("・");
}

export function itemSummaryLine(item: InspectionItem): string {
  const parts: string[] = [];
  const pos = positionLabel(item);
  if (pos) parts.push(pos);
  if (item.measurement?.value !== undefined) {
    parts.push(`${item.measurement.value}${item.measurement.unit ?? ""}`);
  }
  if (item.measurements) {
    for (const [key, value] of Object.entries(item.measurements)) {
      const unit = key === "CO" ? "%" : key === "HC" ? "ppm" : "";
      parts.push(`${key}: ${value}${unit}`);
    }
  }
  if (item.note) parts.push(item.note);
  return parts.join(" / ");
}

export function itemFullLabel(item: InspectionItem): string {
  const summary = itemSummaryLine(item);
  return summary ? `${item.category} ${summary}` : item.category;
}

const SUMMARY_GROUPS: Array<{ status: JudgementStatus; title: string }> = [
  { status: "ng", title: "不合格" },
  { status: "replace_strong", title: "要交換" },
  { status: "recommend", title: "おすすめ" },
  { status: "customer_request", title: "ご要望" },
  { status: "customer_declined", title: "お客様不要" },
];

export function buildSummaryText(inspection: Inspection): string {
  const lines: string[] = [];
  const headerParts: string[] = [];
  if (inspection.customerName) headerParts.push(`お客様名: ${inspection.customerName}`);
  if (inspection.vehicleModel) headerParts.push(`車種: ${inspection.vehicleModel}`);
  if (headerParts.length > 0) {
    lines.push(headerParts.join(" / "));
    lines.push("");
  }

  for (const group of SUMMARY_GROUPS) {
    const items = inspection.items.filter((i) => i.status === group.status);
    if (items.length === 0) continue;
    lines.push(`【${group.title}】`);
    for (const item of items) {
      lines.push(`・${itemFullLabel(item)}`);
    }
    lines.push("");
  }

  if (lines.length === 0 || (headerParts.length > 0 && lines.length === 2)) {
    lines.push("問題項目はありません。");
  }

  return lines.join("\n").trim();
}
