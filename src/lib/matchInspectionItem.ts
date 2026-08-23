import { DEFAULT_CHECKLIST } from "../data/defaultChecklist";
import { levenshteinDistance } from "./fuzzyMatch";
import type { ChecklistItemDef } from "../types";

// 項目名のfuzzy matchを許可する条件（安全側に倒す）:
// ・完全一致・既知エイリアスで見つからない場合のフォールバックとしてのみ使う
// ・短すぎる単語には使わない（誤爆しやすいため）
// ・編集距離1までしか許容しない
// ・該当項目が複数（別項目にまたがって）候補になった場合は採用しない
const FUZZY_MIN_ALIAS_LENGTH = 3;
const FUZZY_MAX_DISTANCE = 1;

interface AliasEntry {
  itemId: string;
  alias: string;
}

const SORTED_ALIAS_ENTRIES: AliasEntry[] = DEFAULT_CHECKLIST.flatMap((item) =>
  item.aliases.map((alias) => ({ itemId: item.id, alias }))
).sort((a, b) => b.alias.length - a.alias.length);

export interface CategoryMatch {
  itemId: string;
  alias: string;
}

/**
 * テキスト中に実際に含まれる項目エイリアスのうち、最長のものを採用する。
 * 一致が無ければ null（未登録項目の候補として扱う）。
 */
export function matchCategory(text: string, checklist: ChecklistItemDef[] = DEFAULT_CHECKLIST): CategoryMatch | null {
  const entries = checklist === DEFAULT_CHECKLIST
    ? SORTED_ALIAS_ENTRIES
    : checklist.flatMap((item) => item.aliases.map((alias) => ({ itemId: item.id, alias })))
        .sort((a, b) => b.alias.length - a.alias.length);

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
    const distance = levenshteinDistance(window, entry.alias);
    if (distance > 0 && distance <= FUZZY_MAX_DISTANCE) {
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
  const entries = checklist === DEFAULT_CHECKLIST
    ? SORTED_ALIAS_ENTRIES
    : checklist.flatMap((item) => item.aliases.map((alias) => ({ itemId: item.id, alias })))
        .sort((a, b) => b.alias.length - a.alias.length);

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
      tokens.push({ itemId: matched.itemId, alias: matched.alias, start: i, end: i + matched.alias.length });
      i += matched.alias.length;
      continue;
    }

    const fuzzy = findFuzzyMatchAt(text, i, entries);
    if (fuzzy) {
      tokens.push({ itemId: fuzzy.itemId, alias: fuzzy.alias, start: i, end: fuzzy.end });
      i = fuzzy.end;
      continue;
    }

    i += 1;
  }
  return tokens;
}
