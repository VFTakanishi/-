import { FRONT_REAR_ALIASES, INNER_OUTER_ALIASES, LEFT_RIGHT_ALIASES } from "../data/voiceAliases";
import { findCategoryOccurrences, resolveApproximateAlias, type CategoryToken } from "./matchInspectionItem";
import { matchJudgement } from "./matchJudgement";
import { buildAligned, normalize, type AlignedText } from "./normalizeText";
import type { ItemMeasurement, ItemPosition, JudgementStatus } from "../types";

export interface ParsedMatched {
  matched: true;
  itemId: string;
  /**
   * 項目名の一致方法。候補選定用メタ情報。
   * exact: 既知エイリアスに完全一致 / fuzzy: 同文字数Hamming距離での軽微な誤認識吸収 /
   * approx: どちらでも見つからなかったセグメント全体を、判定語・数値・位置語を
   * 除いた上でエイリアス全体とLevenshtein比較して解決したもの（最も信頼度が低い）
   */
  matchType?: "exact" | "fuzzy" | "approx";
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

// 比較対象のテキストは常にnormalize()済みのため、比較する別名側も同様に
// normalize()しておく（別名にひらがなが含まれる場合、正規化後のカタカナ
// テキストと表記が一致せずマッチしなくなるのを防ぐ）。
const FRONT_REAR_SORTED = [...FRONT_REAR_ALIASES]
  .map((e) => ({ ...e, alias: normalize(e.alias) }))
  .sort((a, b) => b.alias.length - a.alias.length);
const LEFT_RIGHT_SORTED = [...LEFT_RIGHT_ALIASES]
  .map((e) => ({ ...e, alias: normalize(e.alias) }))
  .sort((a, b) => b.alias.length - a.alias.length);
const INNER_OUTER_SORTED = [...INNER_OUTER_ALIASES]
  .map((e) => ({ ...e, alias: normalize(e.alias) }))
  .sort((a, b) => b.alias.length - a.alias.length);

/**
 * compare（マッチング用に正規化済み）とdisplay（空白のみ除去した原文）を
 * 常に同じ長さ・同じインデックス対応で保持する組。マッチング自体はcompare側で
 * 行い、実際にnoteとして残す文字列はdisplay側から取り出すことで、
 * 「ぶーつ破れ」のような原文のひらがな表記をカタカナに変換してしまう
 * ことなく、ユーザーへの表示内容を発話どおりに保つ。
 */
function slicePair(pair: AlignedText, start: number, end: number): AlignedText {
  return { compare: pair.compare.slice(start, end), display: pair.display.slice(start, end) };
}

/** compare側で見つかった最初のneedleを、display側の同じ位置からも取り除く。 */
function removeFirstPair(pair: AlignedText, needle: string): AlignedText {
  const idx = pair.compare.indexOf(needle);
  if (idx === -1) return pair;
  return {
    compare: pair.compare.slice(0, idx) + pair.compare.slice(idx + needle.length),
    display: pair.display.slice(0, idx) + pair.display.slice(idx + needle.length),
  };
}

/** 末尾からlen文字を両方から取り除く（インデックスのみに基づくため内容に依存しない）。 */
function trimTailPair(pair: AlignedText, len: number): AlignedText {
  const end = pair.compare.length - len;
  return { compare: pair.compare.slice(0, end), display: pair.display.slice(0, end) };
}

function extractPosition(pair: AlignedText): { position: ItemPosition | undefined; remaining: AlignedText } {
  let remaining = pair;
  const position: ItemPosition = {};
  let found = false;

  for (const entry of FRONT_REAR_SORTED) {
    if (remaining.compare.includes(entry.alias)) {
      position.frontRear = entry.value;
      remaining = removeFirstPair(remaining, entry.alias);
      found = true;
      break;
    }
  }

  for (const entry of LEFT_RIGHT_SORTED) {
    if (remaining.compare.includes(entry.alias)) {
      position.leftRight = entry.value;
      remaining = removeFirstPair(remaining, entry.alias);
      found = true;
      break;
    }
  }

  for (const entry of INNER_OUTER_SORTED) {
    if (remaining.compare.includes(entry.alias)) {
      position.innerOuter = entry.value;
      remaining = removeFirstPair(remaining, entry.alias);
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

/**
 * テキストの「先頭」から連続する位置語だけを貪欲に剥がす。extractTrailingPositionWords
 * の先頭版。項目名として認識されたトークンが1つも無い発話（whole-segment approximate
 * resolverの対象）では、位置語を前のトークンに帰属させる仕組みが使えないため、
 * 「項目名らしい部分」を切り出す前段としてこちらを使う。
 */
function extractLeadingPositionWords(text: string): { position: ItemPosition | undefined; consumedLength: number } {
  let start = 0;
  const position: ItemPosition = {};
  let consumed = 0;
  let progressed = true;

  while (progressed && start < text.length) {
    progressed = false;
    const head = text.slice(start);

    if (!position.frontRear) {
      for (const entry of FRONT_REAR_SORTED) {
        if (head.startsWith(entry.alias)) {
          position.frontRear = entry.value;
          start += entry.alias.length;
          consumed += entry.alias.length;
          progressed = true;
          break;
        }
      }
      if (progressed) continue;
    }

    if (!position.leftRight) {
      for (const entry of LEFT_RIGHT_SORTED) {
        if (head.startsWith(entry.alias)) {
          position.leftRight = entry.value;
          start += entry.alias.length;
          consumed += entry.alias.length;
          progressed = true;
          break;
        }
      }
      if (progressed) continue;
    }

    if (!position.innerOuter) {
      for (const entry of INNER_OUTER_SORTED) {
        if (head.startsWith(entry.alias)) {
          position.innerOuter = entry.value;
          start += entry.alias.length;
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

function extractMeasurement(pair: AlignedText): { measurement: ItemMeasurement | undefined; remaining: AlignedText } {
  const match = pair.compare.match(/(\d+(?:\.\d+)?)(mm|ミリ)/i);
  if (!match) return { measurement: undefined, remaining: pair };
  const value = Number(match[1]);
  const remaining = removeFirstPair(pair, match[0]);
  return { measurement: { value, unit: "mm" }, remaining };
}

function cleanNote(text: string): string | undefined {
  const trimmed = text.replace(/^[、,。\s]+/, "").replace(/[、,。\s]+$/, "").trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

function tryParseCoHc(pair: AlignedText): ParsedMatched | null {
  const coMatch = pair.compare.match(/CO(\d+(?:\.\d+)?)/i);
  const hcMatch = pair.compare.match(/HC(\d+(?:\.\d+)?)/i);
  if (!coMatch && !hcMatch) return null;

  const measurements: Record<string, number> = {};
  let remaining = pair;
  if (coMatch) {
    measurements.CO = Number(coMatch[1]);
    remaining = removeFirstPair(remaining, coMatch[0]);
  }
  if (hcMatch) {
    measurements.HC = Number(hcMatch[1]);
    remaining = removeFirstPair(remaining, hcMatch[0]);
  }

  const judgement = matchJudgement(remaining.compare);
  if (judgement) remaining = removeFirstPair(remaining, judgement.alias);

  return {
    matched: true,
    itemId: "co_hc",
    matchType: "exact",
    status: judgement?.status,
    measurements,
    note: cleanNote(remaining.display),
    rawText: pair.display,
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

/**
 * whole-segment approximate resolver。findCategoryOccurrences（完全一致 +
 * 同文字数Hammingフォールバック）で登録済み項目が1つも見つからなかった発話に
 * 対してだけ呼ぶ最終フォールバック。判定語・数値・位置語を取り除いた
 * 「項目名らしい部分」の文字列全体を、登録済みエイリアス全体とLevenshtein距離で
 * 比較する（文字列中をずらしながら比較することはしない）。「文字が抜ける／
 * 増える」タイプの誤認識（例:「タイロットエンドブース」）は同文字数Hammingでは
 * 拾えないが、こちらなら吸収できる。一意に確信を持てる項目が無ければnullを返し、
 * 呼び出し側は通常どおり未登録項目として扱う。
 */
function tryResolveAsApproximateItem(pair: AlignedText): ParsedMatched | null {
  const { position: leadingPosition, consumedLength: leadConsumed } = extractLeadingPositionWords(pair.compare);
  let working = leadConsumed > 0 ? slicePair(pair, leadConsumed, pair.compare.length) : pair;

  const { position: trailingPosition, remaining: afterPosition } = extractPosition(working);
  working = afterPosition;

  const { measurement, remaining: afterMeasurement } = extractMeasurement(working);
  working = afterMeasurement;

  const judgement = matchJudgement(working.compare);
  if (judgement) working = removeFirstPair(working, judgement.alias);

  const candidate = working.compare;
  if (!candidate) return null;

  const approxMatch = resolveApproximateAlias(candidate);
  if (!approxMatch) return null;

  const position = mergePositions(leadingPosition, trailingPosition);

  return {
    matched: true,
    itemId: approxMatch.itemId,
    matchType: "approx",
    status: judgement?.status,
    position,
    measurement,
    note: undefined,
    rawText: pair.display,
  };
}

function parseUnmatchedSegment(rawText: string, pair: AlignedText): ParsedUnmatched {
  const judgement = matchJudgement(pair.compare);
  const remaining = judgement ? removeFirstPair(pair, judgement.alias) : pair;
  const customCategoryName = cleanNote(remaining.display) ?? remaining.display;

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
  const aligned = buildAligned(rawText);
  const tokens = mergeAdjacentSameItemTokens(findCategoryOccurrences(aligned.compare));

  if (tokens.length === 0) {
    // 登録済み項目が1つも見つからない: whole-segment approximate resolverで
    // 最後の望みを試し、それでも一意に決まらなければ未登録項目候補として扱う
    const approx = tryResolveAsApproximateItem(aligned);
    if (approx) return [{ ...approx, rawText }];
    return [parseUnmatchedSegment(rawText, aligned)];
  }

  // トークン間の「隙間」を求める。gaps[i] はトークンiの直前（=トークンi-1の直後）、
  // gaps[tokens.length] は最後のトークンの直後。compare/displayは常に同じ長さ・
  // 同じインデックス対応なので、compareで見つけたトークン境界をdisplay側の
  // スライスにもそのまま使える。
  const gaps: AlignedText[] = [];
  for (let i = 0; i <= tokens.length; i++) {
    const start = i === 0 ? 0 : tokens[i - 1].end;
    const end = i < tokens.length ? tokens[i].start : aligned.compare.length;
    gaps.push(slicePair(aligned, start, end));
  }

  // 各隙間の「末尾」にある位置語（例:「…リヤ」）は、直後のトークン（項目名の前に
  // 置かれた位置語）に属するとみなし、直前のトークンの内容からは取り除く。
  // 電気回りだけは位置語を構造化せず発話どおりの文言（display）のままnoteに
  // 残すため、剥がした文字列そのもの（leadingRawTextForToken）も別途保持しておく。
  const leadingPositionForToken: Array<ItemPosition | undefined> = [];
  const leadingRawTextForToken: string[] = [];
  const trimmedGap: AlignedText[] = new Array(gaps.length);
  trimmedGap[gaps.length - 1] = gaps[gaps.length - 1]; // 最後の隙間（末尾）は次の項目が無いのでそのまま
  for (let i = 0; i < tokens.length; i++) {
    const { position, consumedLength } = extractTrailingPositionWords(gaps[i].compare);
    leadingPositionForToken[i] = position;
    leadingRawTextForToken[i] = consumedLength > 0 ? gaps[i].display.slice(gaps[i].display.length - consumedLength) : "";
    trimmedGap[i] = trimTailPair(gaps[i], consumedLength);
  }

  const results: ParsedVoiceInspection[] = [];

  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    const nextStart = tokens[i + 1]?.start ?? aligned.compare.length;
    const ownContent = i + 1 < tokens.length ? trimmedGap[i + 1] : gaps[tokens.length];

    if (token.itemId === "co_hc") {
      const span = slicePair(aligned, token.start, nextStart);
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
      const judgement = matchJudgement(trailing.compare);
      if (judgement) trailing = removeFirstPair(trailing, judgement.alias);

      const noteText = leadingRawTextForToken[i] + (keepAliasInNote ? token.alias : "") + trailing.display;

      results.push({
        matched: true,
        itemId: "electrical",
        matchType: token.matchType,
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

    const judgement = matchJudgement(remaining.compare);
    if (judgement) remaining = removeFirstPair(remaining, judgement.alias);

    const position = mergePositions(leadingPositionForToken[i], trailingPosition);

    results.push({
      matched: true,
      itemId: token.itemId,
      matchType: token.matchType,
      status: judgement?.status,
      position,
      measurement,
      note: cleanNote(remaining.display),
      rawText,
    });
  }

  return results;
}

/**
 * Safariが返す複数の認識候補（alternatives）から、
 * 「点検項目として最も自然に成立する候補」を1つだけ選ぶ。
 *
 * 優先順位:
 *  1. 登録済み点検項目へのmatch件数が多い
 *  2. unmatchedが少ない
 *  3. 完全一致/明示aliasによるmatchを優先（fuzzy matchより高評価）
 *  4. 条件が同等ならSafariの候補順位が上のもの（先頭）を優先
 *
 * 複数候補の解析結果を混ぜることはしない: 採用した1つの候補のtranscriptを
 * そのままparseVoiceInspectionsした結果をまとめて返す。
 */
export function chooseBestTranscript(candidates: string[]): { transcript: string; results: ParsedVoiceInspection[] } {
  if (candidates.length === 0) return { transcript: "", results: [] };

  let best: { transcript: string; results: ParsedVoiceInspection[]; score: number } | null = null;

  for (const candidate of candidates) {
    const results = parseVoiceInspections(candidate);
    const score = scoreParsedResults(results);
    if (!best || score > best.score) {
      best = { transcript: candidate, results, score };
    }
  }

  return { transcript: best!.transcript, results: best!.results };
}

function scoreParsedResults(results: ParsedVoiceInspection[]): number {
  let score = 0;
  for (const r of results) {
    if (r.matched) {
      score += 10;
      if (r.matchType === "exact") score += 5; // exact/明示alias一致を優遇（fuzzy/approxより上）
    } else {
      score -= 3; // unmatchedは減点（曖昧なら無理に別項目へ記録しないというルールと整合）
    }
  }
  return score;
}

/** 後方互換用: 1発話につき1項目だけを扱いたい場合に最初の結果を返す。 */
export function parseVoiceInspection(rawText: string): ParsedVoiceInspection {
  return parseVoiceInspections(rawText)[0];
}
