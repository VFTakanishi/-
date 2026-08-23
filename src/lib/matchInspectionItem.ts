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
