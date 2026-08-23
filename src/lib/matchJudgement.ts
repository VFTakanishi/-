import { JUDGEMENT_ALIASES } from "../data/voiceAliases";
import type { JudgementStatus } from "../types";

const SORTED_ALIASES = [...JUDGEMENT_ALIASES].sort((a, b) => b.alias.length - a.alias.length);

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
