import { FRONT_REAR_ALIASES, INNER_OUTER_ALIASES, LEFT_RIGHT_ALIASES } from "../data/voiceAliases";
import { findCategoryOccurrences, removeAlias, type CategoryToken } from "./matchInspectionItem";
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

const FRONT_REAR_SORTED = [...FRONT_REAR_ALIASES].sort((a, b) => b.alias.length - a.alias.length);
const LEFT_RIGHT_SORTED = [...LEFT_RIGHT_ALIASES].sort((a, b) => b.alias.length - a.alias.length);
const INNER_OUTER_SORTED = [...INNER_OUTER_ALIASES].sort((a, b) => b.alias.length - a.alias.length);

function extractPosition(text: string): { position: ItemPosition | undefined; remaining: string } {
  let remaining = text;
  const position: ItemPosition = {};
  let found = false;

  for (const entry of FRONT_REAR_SORTED) {
    if (remaining.includes(entry.alias)) {
      position.frontRear = entry.value;
      remaining = removeAlias(remaining, entry.alias);
      found = true;
      break;
    }
  }

  for (const entry of LEFT_RIGHT_SORTED) {
    if (remaining.includes(entry.alias)) {
      position.leftRight = entry.value;
      remaining = removeAlias(remaining, entry.alias);
      found = true;
      break;
    }
  }

  for (const entry of INNER_OUTER_SORTED) {
    if (remaining.includes(entry.alias)) {
      position.innerOuter = entry.value;
      remaining = removeAlias(remaining, entry.alias);
      found = true;
      break;
    }
  }

  return { position: found ? position : undefined, remaining };
}

function mergePositions(leading: ItemPosition | undefined, trailing: ItemPosition | undefined): ItemPosition | undefined {
  if (!leading && !trailing) return undefined;
  const merged: ItemPosition = {
    frontRear: trailing?.frontRear ?? leading?.frontRear,
    leftRight: trailing?.leftRight ?? leading?.leftRight,
    innerOuter: trailing?.innerOuter ?? leading?.innerOuter,
  };
  return merged;
}

/**
 * テキストの「末尾」から連続する位置語（前後・左右・インナーアウター）だけを
 * 貪欲に剥がす。項目名の直前に置かれた位置語（例:「リヤドライブシャフトブーツ」の
 * 「リヤ」）を、次の項目のものとして正しく拾うために使う。
 * 位置語以外の文字が混ざっている場合はそこで打ち切り、誤って無関係な文字列を
 * 位置語とみなさないようにする。
 */
function extractTrailingPositionWords(text: string): { position: ItemPosition | undefined; consumedLength: number } {
  let end = text.length;
  const position: ItemPosition = {};
  let consumed = 0;
  let progressed = true;

  while (progressed && end > 0) {
    progressed = false;
    const tail = text.slice(0, end);

    if (!position.frontRear) {
      for (const entry of FRONT_REAR_SORTED) {
        if (tail.endsWith(entry.alias)) {
          position.frontRear = entry.value;
          end -= entry.alias.length;
          consumed += entry.alias.length;
          progressed = true;
          break;
        }
      }
      if (progressed) continue;
    }

    if (!position.leftRight) {
      for (const entry of LEFT_RIGHT_SORTED) {
        if (tail.endsWith(entry.alias)) {
          position.leftRight = entry.value;
          end -= entry.alias.length;
          consumed += entry.alias.length;
          progressed = true;
          break;
        }
      }
      if (progressed) continue;
    }

    if (!position.innerOuter) {
      for (const entry of INNER_OUTER_SORTED) {
        if (tail.endsWith(entry.alias)) {
          position.innerOuter = entry.value;
          end -= entry.alias.length;
          consumed += entry.alias.length;
          progressed = true;
          break;
        }
      }
    }
  }

  const hasPosition = position.frontRear !== undefined || position.leftRight !== undefined || position.innerOuter !== undefined;
  return { position: hasPosition ? position : undefined, consumedLength: consumed };
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
 * 同じ項目が隣接して複数回検出された場合（間に他の項目が挟まらない場合）は
 * 1つのセグメントにまとめる。CO・HCが "CO" と "HC" の2エイリアスに分かれて
 * 検出されるケースや、電気回りで「電気回り」＋「ストップランプ」のように
 * 総称語と具体的な灯火名が同じ発話内で連続するケースに対応するため。
 */
function mergeAdjacentSameItemTokens(tokens: CategoryToken[]): CategoryToken[] {
  const merged: CategoryToken[] = [];
  for (const token of tokens) {
    const last = merged[merged.length - 1];
    if (last?.itemId === token.itemId) {
      continue; // 直前の同項目セグメントに吸収させる（区間は次トークンのstartまで自動的に伸びる）
    }
    merged.push(token);
  }
  return merged;
}

// 電気回りのうち、これらのエイリアスは「灯火名」そのものが不具合内容の一部
// なので、項目名として消費せずnoteにそのまま残す（「電気回り」「電装」は
// 総称ラベルなので通常どおり取り除く）。
const ELECTRICAL_DESCRIPTIVE_ALIASES = new Set([
  "ストップランプ",
  "ブレーキランプ",
  "テールランプ",
  "ウインカー",
  "ウィンカー",
  "ヘッドライト",
  "ヘッドランプ",
  "スモールランプ",
  "バックランプ",
  "ナンバー灯",
  "ライセンスランプ",
]);

function parseUnmatchedSegment(rawText: string, text: string): ParsedUnmatched {
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

/**
 * 1回の発話（音声認識結果）を解析する。
 * 「整備上の判定は発話に判定語が明示されているときだけ設定する」というルールを守り、
 * 異常語だけの発話では status を設定しない。
 *
 * 実機では複数の点検項目を連続して発話するケースが多いため、文中に出現する
 * 登録済み項目エイリアスをすべて検出し、「その項目から次の項目が始まる直前まで」
 * の範囲だけを対象に判定語・数値・コメントを抽出する。これにより後続項目の
 * 判定語が前の項目に誤って適用されることを防ぐ。
 */
export function parseVoiceInspections(rawText: string): ParsedVoiceInspection[] {
  const text = normalize(rawText);
  const tokens = mergeAdjacentSameItemTokens(findCategoryOccurrences(text));

  if (tokens.length === 0) {
    // 登録済み項目が1つも見つからない: 未登録項目候補として扱う
    return [parseUnmatchedSegment(rawText, text)];
  }

  // トークン間の「隙間」を求める。gaps[i] はトークンiの直前（=トークンi-1の直後）、
  // gaps[tokens.length] は最後のトークンの直後。
  const gaps: string[] = [];
  for (let i = 0; i <= tokens.length; i++) {
    const start = i === 0 ? 0 : tokens[i - 1].end;
    const end = i < tokens.length ? tokens[i].start : text.length;
    gaps.push(text.slice(start, end));
  }

  // 各隙間の「末尾」にある位置語（例:「…リヤ」）は、直後のトークン（項目名の前に
  // 置かれた位置語）に属するとみなし、直前のトークンの内容からは取り除く。
  // 電気回りだけは位置語を構造化せず生テキストのままnoteに残すため、
  // 剥がした文字列そのもの（leadingRawTextForToken）も別途保持しておく。
  const leadingPositionForToken: Array<ItemPosition | undefined> = [];
  const leadingRawTextForToken: string[] = [];
  const trimmedGap: string[] = new Array(gaps.length);
  trimmedGap[gaps.length - 1] = gaps[gaps.length - 1]; // 最後の隙間（末尾）は次の項目が無いのでそのまま
  for (let i = 0; i < tokens.length; i++) {
    const { position, consumedLength } = extractTrailingPositionWords(gaps[i]);
    leadingPositionForToken[i] = position;
    leadingRawTextForToken[i] = consumedLength > 0 ? gaps[i].slice(gaps[i].length - consumedLength) : "";
    trimmedGap[i] = gaps[i].slice(0, gaps[i].length - consumedLength);
  }

  const results: ParsedVoiceInspection[] = [];

  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    const nextStart = tokens[i + 1]?.start ?? text.length;
    const ownContent = i + 1 < tokens.length ? trimmedGap[i + 1] : gaps[tokens.length];

    if (token.itemId === "co_hc") {
      const span = text.slice(token.start, nextStart);
      const coHc = tryParseCoHc(span);
      if (coHc) {
        results.push({ ...coHc, rawText });
        continue;
      }
      // CO/HCの数値が取れなければ通常の項目として処理を続ける（fall through）
    }

    if (token.itemId === "electrical") {
      // 電気回りは複数箇所の灯火不具合等を自由記述で記録するfree-form項目。
      // 前後・左右等の位置語は構造化せず、発話どおりの文言としてnoteに残す。
      // 「電気回り」「電装」という総称ラベルは取り除くが、「ストップランプ」等の
      // 具体的な灯火名は不具合内容そのものなのでnoteに残す。
      const keepAliasInNote = ELECTRICAL_DESCRIPTIVE_ALIASES.has(token.alias);
      let trailing = ownContent;
      const judgement = matchJudgement(trailing);
      if (judgement) trailing = removeJudgementAlias(trailing, judgement.alias);

      const noteText = leadingRawTextForToken[i] + (keepAliasInNote ? token.alias : "") + trailing;

      results.push({
        matched: true,
        itemId: "electrical",
        status: judgement?.status,
        note: cleanNote(noteText),
        rawText,
      });
      continue;
    }

    let remaining = ownContent;

    const { position: trailingPosition, remaining: afterPosition } = extractPosition(remaining);
    remaining = afterPosition;

    const { measurement, remaining: afterMeasurement } = extractMeasurement(remaining);
    remaining = afterMeasurement;

    const judgement = matchJudgement(remaining);
    if (judgement) remaining = removeJudgementAlias(remaining, judgement.alias);

    const position = mergePositions(leadingPositionForToken[i], trailingPosition);

    results.push({
      matched: true,
      itemId: token.itemId,
      status: judgement?.status,
      position,
      measurement,
      note: cleanNote(remaining),
      rawText,
    });
  }

  return results;
}

/** 後方互換用: 1発話につき1項目だけを扱いたい場合に最初の結果を返す。 */
export function parseVoiceInspection(rawText: string): ParsedVoiceInspection {
  return parseVoiceInspections(rawText)[0];
}
