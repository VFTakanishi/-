import { DEFAULT_CHECKLIST } from "../data/defaultChecklist";
import type { ChecklistItemDef } from "../types";

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

/**
 * テキストを先頭から走査し、登録済み項目エイリアスが出現する箇所を
 * 出現順にすべて検出する。各位置では最長一致するエイリアスを優先する。
 * 1回の発話に複数の点検項目が含まれる場合の分割に使う。
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
    } else {
      i += 1;
    }
  }
  return tokens;
}
