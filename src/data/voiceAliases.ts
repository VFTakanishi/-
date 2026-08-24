import type { JudgementStatus } from "../types";

// 判定語の同義語辞書。誤って部分一致しないよう、マッチングは常に
// 「文中に存在する最長のエイリアス」を優先して探す（matchJudgement.ts参照）。
export const JUDGEMENT_ALIASES: Array<{ status: JudgementStatus; alias: string }> = [
  { status: "ng", alias: "不合格" },
  { status: "ng", alias: "アウト" },
  { status: "ng", alias: "ダメ" },
  { status: "ng", alias: "車検NG" },
  { status: "ng", alias: "NG" },
  { status: "ng", alias: "エヌジー" },

  { status: "replace_strong", alias: "交換推奨" },
  { status: "replace_strong", alias: "交換必要" },
  { status: "replace_strong", alias: "要交換" },
  { status: "replace_strong", alias: "交換" },

  { status: "recommend", alias: "おすすめ" },
  { status: "recommend", alias: "オススメ" },
  { status: "recommend", alias: "お勧め" },
  { status: "recommend", alias: "推奨" },

  { status: "na", alias: "該当なし" },
  { status: "na", alias: "付いていない" },
  { status: "na", alias: "付いてない" },
  { status: "na", alias: "なし" },

  { status: "good", alias: "問題なし" },
  { status: "good", alias: "オーケー" },
  { status: "good", alias: "オッケー" },
  { status: "good", alias: "良好" },
  { status: "good", alias: "OK" },

  { status: "customer_request", alias: "お客様要望" },
  { status: "customer_request", alias: "お客さん要望" },
  { status: "customer_request", alias: "依頼あり" },
  { status: "customer_request", alias: "ご要望" },
  { status: "customer_request", alias: "要望" },

  { status: "customer_declined", alias: "不要とのこと" },
  { status: "customer_declined", alias: "お客様不要" },
  { status: "customer_declined", alias: "お客さん不要" },
  { status: "customer_declined", alias: "いらない" },
  { status: "customer_declined", alias: "やらない" },
  { status: "customer_declined", alias: "不要" },

  { status: "maintenance", alias: "整備" },
];

export const FRONT_REAR_ALIASES: Array<{ value: "front" | "rear"; alias: string }> = [
  { value: "front", alias: "フロント" },
  { value: "rear", alias: "リア" },
  { value: "rear", alias: "リヤ" },
  { value: "front", alias: "前" },
  { value: "rear", alias: "後ろ" },
];

export const INNER_OUTER_ALIASES: Array<{ value: "inner" | "outer"; alias: string }> = [
  { value: "inner", alias: "インナー" },
  { value: "outer", alias: "アウター" },
];

export const LEFT_RIGHT_ALIASES: Array<{ value: "left" | "right"; alias: string }> = [
  { value: "right", alias: "右" },
  { value: "left", alias: "左" },
];

export const STATUS_LABELS: Record<JudgementStatus, string> = {
  ng: "不合格",
  replace_strong: "要交換",
  recommend: "おすすめ",
  na: "該当なし",
  good: "良好",
  customer_request: "ご要望",
  customer_declined: "お客様不要",
  maintenance: "整備",
  unset: "未確認",
};
