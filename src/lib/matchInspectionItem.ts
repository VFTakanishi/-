import { DEFAULT_CHECKLIST } from "../data/defaultChecklist";
import { hammingDistance, levenshteinDistance } from "./fuzzyMatch";
import { normalize } from "./normalizeText";
import type { ChecklistItemDef } from "../types";

// 項目名のfuzzy matchを許可する条件（安全側に倒す）:
// ・完全一致・既知エイリアスで見つからない場合のフォールバックとしてのみ使う
// ・短すぎる単語には使わない（誤爆しやすいため）
// ・長いエイリアスほど許容編集距離を広げる（短い単語で距離2まで許すと誤爆しやすいため）
// ・該当項目が複数（別項目にまたがって）候補になった場合は採用しない
const FUZZY_MIN_ALIAS_LENGTH = 3;
// 「○○ブーツ」のように語尾が共通する項目名が多く、6〜7文字程度では距離2まで
// 許すと別項目と衝突しやすいため、距離2まで許すのは十分に長い（8文字以上の）
// エイリアスに限定する。
const FUZZY_LONG_ALIAS_LENGTH = 8;

function maxDistanceForAliasLength(aliasLength: number): number {
  return aliasLength >= FUZZY_LONG_ALIAS_LENGTH ? 2 : 1;
}

interface AliasEntry {
  itemId: string;
  alias: string;
}

// 比較対象のテキストは常にnormalize()済みのため、比較する別名側も同様に
// normalize()しておく（別名にひらがなが含まれる場合、正規化後のカタカナ
// テキストと表記が一致せずマッチしなくなるのを防ぐ）。
function toNormalizedEntries(checklist: ChecklistItemDef[]): AliasEntry[] {
  return checklist
    .flatMap((item) => item.aliases.map((alias) => ({ itemId: item.id, alias: normalize(alias) })))
    .sort((a, b) => b.alias.length - a.alias.length);
}

const SORTED_ALIAS_ENTRIES: AliasEntry[] = toNormalizedEntries(DEFAULT_CHECKLIST);

export interface CategoryMatch {
  itemId: string;
  alias: string;
}

/**
 * テキスト中に実際に含まれる項目エイリアスのうち、最長のものを採用する。
 * 一致が無ければ null（未登録項目の候補として扱う）。
 */
export function matchCategory(text: string, checklist: ChecklistItemDef[] = DEFAULT_CHECKLIST): CategoryMatch | null {
  const entries = checklist === DEFAULT_CHECKLIST ? SORTED_ALIAS_ENTRIES : toNormalizedEntries(checklist);

  for (const entry of entries) {
    if (text.includes(entry.alias)) {
      return { itemId: entry.itemId, alias: entry.alias };
    }
  }
  return null;
}

export function removeAlias(text: string, alias: string): string {
  return text.replace(alias, "");
}

export interface CategoryToken {
  itemId: string;
  alias: string;
  start: number;
  end: number;
  /** 完全一致（既知エイリアス）か、軽微な誤認識を吸収したfuzzy matchか */
  matchType: "exact" | "fuzzy";
}

interface FuzzyMatch {
  itemId: string;
  alias: string;
  end: number;
}

/**
 * 位置iで完全一致が見つからなかった場合だけ呼ばれるフォールバック。
 * 各エイリアスについて、その長さ±1の範囲でテキストとの編集距離を調べ、
 * 距離1以内のものだけを候補にする。候補の項目IDが1つに定まらない場合は
 * 誤った項目への記録を避けるため、マッチなし（null）を返す。
 */
function findFuzzyMatchAt(text: string, i: number, entries: AliasEntry[]): FuzzyMatch | null {
  const candidatesByItem = new Map<string, FuzzyMatch>();

  for (const entry of entries) {
    if (entry.alias.length < FUZZY_MIN_ALIAS_LENGTH) continue;

    // 比較窓はエイリアスと同じ長さだけに限定する（1文字分だけ窓をずらす等を
    // 許すと、直後に本物の完全一致が続く箇所の手前が「先頭1文字挿入」という
    // 形で常にfuzzy一致してしまい、隣接語の末尾を誤って飲み込んでしまうため）。
    // 同じ長さ同士の比較であれば、距離1は「1文字の置換」のみを意味し、
    // 本当に近い1文字違いの誤認識だけを安全に拾える。
    const windowLen = entry.alias.length;
    if (i + windowLen > text.length) continue;
    const window = text.slice(i, i + windowLen);
    const distance = hammingDistance(window, entry.alias);
    if (distance > 0 && distance <= maxDistanceForAliasLength(entry.alias.length)) {
      if (!candidatesByItem.has(entry.itemId)) {
        candidatesByItem.set(entry.itemId, { itemId: entry.itemId, alias: entry.alias, end: i + windowLen });
      }
    }
  }

  if (candidatesByItem.size !== 1) return null; // 複数項目にまたがる場合は勝手に決定しない
  return [...candidatesByItem.values()][0];
}

/**
 * テキストを先頭から走査し、登録済み項目エイリアスが出現する箇所を
 * 出現順にすべて検出する。各位置では最長一致するエイリアスを優先する。
 * 完全一致が見つからない位置だけ、軽微な音声誤認識を吸収するための
 * fuzzy match（編集距離1以内）を試す。1回の発話に複数の点検項目が
 * 含まれる場合の分割にも使う。
 */
export function findCategoryOccurrences(
  text: string,
  checklist: ChecklistItemDef[] = DEFAULT_CHECKLIST
): CategoryToken[] {
  const entries = checklist === DEFAULT_CHECKLIST ? SORTED_ALIAS_ENTRIES : toNormalizedEntries(checklist);

  const tokens: CategoryToken[] = [];
  let i = 0;
  while (i < text.length) {
    let matched: AliasEntry | null = null;
    for (const entry of entries) {
      if (text.startsWith(entry.alias, i)) {
        matched = entry;
        break;
      }
    }
    if (matched) {
      tokens.push({ itemId: matched.itemId, alias: matched.alias, start: i, end: i + matched.alias.length, matchType: "exact" });
      i += matched.alias.length;
      continue;
    }

    const fuzzy = findFuzzyMatchAt(text, i, entries);
    if (fuzzy) {
      tokens.push({ itemId: fuzzy.itemId, alias: fuzzy.alias, start: i, end: fuzzy.end, matchType: "fuzzy" });
      i = fuzzy.end;
      continue;
    }

    i += 1;
  }
  return tokens;
}

// whole-segment approximate resolver（フォールバックの最終段）:
// findCategoryOccurrences の完全一致・同文字数Hammingフォールバックで一切
// 見つからなかった発話セグメントに対してだけ使う。文字列中をずらしながら
// Levenshteinで走査することはしない（過去に隣接語を飲み込む誤爆の原因になった
// ため、明確に禁止）。ここでは「判定語・数値・位置語等を取り除いた後の
// 項目名らしい部分の文字列全体」1つと、登録済みエイリアス「全体」を
// 1対1で比較するだけ。「文字が抜ける／増える」タイプのASR誤認識
// （例:「タイロットエンドブース」）は同文字数Hammingでは長さが違うため
// 拾えないが、こちらのLevenshtein比較なら挿入・削除も吸収できる。
const APPROX_MIN_ALIAS_LENGTH = 5; // 4文字以下のエイリアスはapprox対象外（誤爆しやすいため）
const APPROX_MIN_SIMILARITY = 0.75; // 類似率75%未満は不採用

function maxApproxDistanceForAliasLength(aliasLength: number): number {
  if (aliasLength <= 7) return 1; // 5〜7文字: 距離1まで
  if (aliasLength <= 11) return 2; // 8〜11文字: 距離2まで
  return 3; // 12文字以上: 距離3まで
}

export interface ApproxMatch {
  itemId: string;
  alias: string;
  distance: number;
}

/**
 * 「項目名らしい部分」として切り出された文字列candidate全体を、登録済み
 * エイリアス全体と1対1のLevenshtein距離で比較し、最も近い項目を1つだけ返す。
 * 以下をすべて満たす場合のみ採用する（安全側に倒す）:
 *  ・エイリアス長が5文字以上（4文字以下は対象外）
 *  ・エイリアス長に応じた最大編集距離以内（5〜7文字:1 / 8〜11文字:2 / 12文字以上:3）
 *  ・類似率（1 - 距離/長い方の文字数）が75%以上
 *  ・該当項目が複数（別項目）で同距離以下に並ぶ場合は採用しない（一意に決まる場合のみ）
 * 例えば「オイル」のような短い曖昧語だけでは、どのオイル系項目にも
 * 安全マージンを満たさないため採用されない。
 */
export function resolveApproximateAlias(candidate: string, checklist: ChecklistItemDef[] = DEFAULT_CHECKLIST): ApproxMatch | null {
  if (!candidate) return null;
  const entries = checklist === DEFAULT_CHECKLIST ? SORTED_ALIAS_ENTRIES : toNormalizedEntries(checklist);

  const bestByItem = new Map<string, { alias: string; distance: number }>();
  for (const entry of entries) {
    if (entry.alias.length < APPROX_MIN_ALIAS_LENGTH) continue;
    const distance = levenshteinDistance(candidate, entry.alias);
    if (distance === 0) continue; // 完全一致は上位のフォールバックで既に処理済みのはず
    if (distance > maxApproxDistanceForAliasLength(entry.alias.length)) continue;

    const longer = Math.max(candidate.length, entry.alias.length);
    const similarity = longer === 0 ? 0 : 1 - distance / longer;
    if (similarity < APPROX_MIN_SIMILARITY) continue;

    const existing = bestByItem.get(entry.itemId);
    if (!existing || distance < existing.distance) {
      bestByItem.set(entry.itemId, { alias: entry.alias, distance });
    }
  }

  if (bestByItem.size === 0) return null;

  const ranked = [...bestByItem.entries()].sort((a, b) => a[1].distance - b[1].distance);
  const [bestItemId, bestInfo] = ranked[0];
  const second = ranked[1];

  // 2位の項目が同距離以下（＝同程度）なら、どちらか一方に決められないため不採用
  if (second && second[1].distance <= bestInfo.distance) return null;

  return { itemId: bestItemId, alias: bestInfo.alias, distance: bestInfo.distance };
}
