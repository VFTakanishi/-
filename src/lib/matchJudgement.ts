import { JUDGEMENT_ALIASES } from "../data/voiceAliases";
import { normalize } from "./normalizeText";
import type { JudgementStatus } from "../types";

// 比較対象のテキストは常にnormalize()済みのため、比較する判定語エイリアス側も
// 同様にnormalize()しておく（「いらない」等ひらがなを含むエイリアスが、
// 正規化後のカタカナテキストと表記が一致せずマッチしなくなるのを防ぐ）。
const SORTED_ALIASES = [...JUDGEMENT_ALIASES]
  .map((e) => ({ ...e, alias: normalize(e.alias) }))
  .sort((a, b) => b.alias.length - a.alias.length);

export interface JudgementMatch {
  status: JudgementStatus;
  alias: string;
}

/**
 * 発話テキストに実際に含まれる判定語だけを検出する。
 * 判定語が見つからない場合は null を返し、呼び出し側は status を
 * "unset" のままにする（推測で判定を補完しない）。
 */
export function matchJudgement(text: string): JudgementMatch | null {
  for (const entry of SORTED_ALIASES) {
    if (text.includes(entry.alias)) {
      return { status: entry.status, alias: entry.alias };
    }
  }
  return null;
}

/** 見つかった判定語エイリアスをテキストから1回だけ取り除く */
export function removeJudgementAlias(text: string, alias: string): string {
  return text.replace(alias, "");
}
