import { FRONT_REAR_ALIASES, INNER_OUTER_ALIASES } from "../data/voiceAliases";
import { matchCategory, removeAlias } from "./matchInspectionItem";
import { matchJudgement, removeJudgementAlias } from "./matchJudgement";
import type { ItemMeasurement, ItemPosition, JudgementStatus } from "../types";

export interface ParsedMatched {
  matched: true;
  itemId: string;
  status?: JudgementStatus;
  position?: ItemPosition;
  measurement?: ItemMeasurement;
  measurements?: Record<string, number>;
  note?: string;
  rawText: string;
}

export interface ParsedUnmatched {
  matched: false;
  customCategoryName: string;
  status?: JudgementStatus;
  rawText: string;
}

export type ParsedVoiceInspection = ParsedMatched | ParsedUnmatched;

const FULLWIDTH_RE = /[０-９Ａ-Ｚａ-ｚ．]/g;

function toHalfWidth(char: string): string {
  const code = char.charCodeAt(0);
  if (code === 0xff0e) return ".";
  return String.fromCharCode(code - 0xfee0);
}

function normalize(text: string): string {
  return text
    .replace(FULLWIDTH_RE, toHalfWidth)
    .replace(/\s+/g, "")
    .trim();
}

function extractPosition(text: string): { position: ItemPosition | undefined; remaining: string } {
  let remaining = text;
  const position: ItemPosition = {};
  let found = false;

  const frontRearSorted = [...FRONT_REAR_ALIASES].sort((a, b) => b.alias.length - a.alias.length);
  for (const entry of frontRearSorted) {
    if (remaining.includes(entry.alias)) {
      position.frontRear = entry.value;
      remaining = removeAlias(remaining, entry.alias);
      found = true;
      break;
    }
  }

  const innerOuterSorted = [...INNER_OUTER_ALIASES].sort((a, b) => b.alias.length - a.alias.length);
  for (const entry of innerOuterSorted) {
    if (remaining.includes(entry.alias)) {
      position.innerOuter = entry.value;
      remaining = removeAlias(remaining, entry.alias);
      found = true;
      break;
    }
  }

  return { position: found ? position : undefined, remaining };
}

function extractMeasurement(text: string): { measurement: ItemMeasurement | undefined; remaining: string } {
  const match = text.match(/(\d+(?:\.\d+)?)(mm|ミリ)/i);
  if (!match) return { measurement: undefined, remaining: text };
  const value = Number(match[1]);
  const remaining = text.replace(match[0], "");
  return { measurement: { value, unit: "mm" }, remaining };
}

function cleanNote(text: string): string | undefined {
  const trimmed = text.replace(/^[、,。\s]+/, "").replace(/[、,。\s]+$/, "").trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

function tryParseCoHc(text: string): ParsedMatched | null {
  const coMatch = text.match(/CO(\d+(?:\.\d+)?)/i);
  const hcMatch = text.match(/HC(\d+(?:\.\d+)?)/i);
  if (!coMatch && !hcMatch) return null;

  const measurements: Record<string, number> = {};
  let remaining = text;
  if (coMatch) {
    measurements.CO = Number(coMatch[1]);
    remaining = remaining.replace(coMatch[0], "");
  }
  if (hcMatch) {
    measurements.HC = Number(hcMatch[1]);
    remaining = remaining.replace(hcMatch[0], "");
  }

  const judgement = matchJudgement(remaining);
  if (judgement) remaining = removeJudgementAlias(remaining, judgement.alias);

  return {
    matched: true,
    itemId: "co_hc",
    status: judgement?.status,
    measurements,
    note: cleanNote(remaining),
    rawText: text,
  };
}

/**
 * 1回の発話（音声認識結果）を解析する。
 * 「整備上の判定は発話に判定語が明示されているときだけ設定する」というルールを守り、
 * 異常語だけの発話では status を設定しない。
 */
export function parseVoiceInspection(rawText: string): ParsedVoiceInspection {
  const text = normalize(rawText);

  const coHc = tryParseCoHc(text);
  if (coHc) return coHc;

  const category = matchCategory(text);

  if (category) {
    let remaining = removeAlias(text, category.alias);

    const { position, remaining: afterPosition } = extractPosition(remaining);
    remaining = afterPosition;

    const { measurement, remaining: afterMeasurement } = extractMeasurement(remaining);
    remaining = afterMeasurement;

    const judgement = matchJudgement(remaining);
    if (judgement) remaining = removeJudgementAlias(remaining, judgement.alias);

    return {
      matched: true,
      itemId: category.itemId,
      status: judgement?.status,
      position,
      measurement,
      note: cleanNote(remaining),
      rawText,
    };
  }

  // 未登録項目候補: 判定語のみ抽出し、位置・数値は推測せず名称に残す
  const judgement = matchJudgement(text);
  const remaining = judgement ? removeJudgementAlias(text, judgement.alias) : text;
  const customCategoryName = cleanNote(remaining) ?? remaining;

  return {
    matched: false,
    customCategoryName,
    status: judgement?.status,
    rawText,
  };
}
