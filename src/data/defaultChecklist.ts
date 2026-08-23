import type { ChecklistItemDef } from "../types";

export const DEFAULT_CHECKLIST: ChecklistItemDef[] = [
  // 油脂・エンジン関連
  { id: "engine_oil", category: "エンジンオイル", group: "油脂・エンジン関連", aliases: ["エンジンオイル"] },
  { id: "oil_element", category: "オイルエレメント", group: "油脂・エンジン関連", aliases: ["オイルエレメント", "オイルフィルター"] },
  { id: "air_element", category: "エアエレメント（清掃）", group: "油脂・エンジン関連", aliases: ["エアエレメント", "エアクリーナーエレメント", "エアクリーナー"] },
  { id: "power_steering_oil", category: "パワステオイル", group: "油脂・エンジン関連", aliases: ["パワステオイル", "パワーステアリングオイル"] },
  { id: "coolant", category: "クーラント", group: "油脂・エンジン関連", aliases: ["クーラント", "冷却水", "空欄と", "空欄ト", "クーランド"] },
  { id: "accessory_belt", category: "補機ベルト", group: "油脂・エンジン関連", aliases: ["補機ベルト", "ファンベルト", "エアコンベルト", "補器ベルト", "ホキベルト", "ほきベルト"] },
  { id: "timing_belt", category: "タイミングベルト", group: "油脂・エンジン関連", aliases: ["タイミングベルト"] },
  { id: "spark_plug", category: "スパークプラグ", group: "油脂・エンジン関連", aliases: ["スパークプラグ"] },
  { id: "battery", category: "バッテリー", group: "油脂・エンジン関連", aliases: ["バッテリー"] },
  { id: "brake_fluid", category: "ブレーキフルード", group: "油脂・エンジン関連", aliases: ["ブレーキフルード"] },
  { id: "mission_oil", category: "ミッションオイル", group: "油脂・エンジン関連", aliases: ["ミッションオイル"] },
  { id: "transfer_oil", category: "トランスファーオイル", group: "油脂・エンジン関連", aliases: ["トランスファーオイル"] },
  { id: "diff_oil", category: "デフオイル", group: "油脂・エンジン関連", aliases: ["デフオイル", "デファレンシャルオイル"] },
  { id: "ac_filter", category: "エアコンフィルター", group: "油脂・エンジン関連", aliases: ["エアコンフィルター"] },

  // ブレーキ・足回り
  { id: "driveshaft_boot", category: "ドライブシャフトブーツ", group: "ブレーキ・足回り", aliases: ["ドライブシャフトブーツ", "ドライブシャフト"] },
  { id: "steering_boot", category: "ステアリングブーツ", group: "ブレーキ・足回り", aliases: ["ステアリングブーツ"] },
  { id: "steering_rack_boot", category: "ステアリングラックブーツ", group: "ブレーキ・足回り", aliases: ["ステアリングラックブーツ", "ラックブーツ"] },
  { id: "lower_boot", category: "ロアブーツ", group: "ブレーキ・足回り", aliases: ["ロアブーツ", "ロアアームブーツ"] },
  { id: "tie_rod_end_boot", category: "タイロッドエンドブーツ", group: "ブレーキ・足回り", aliases: ["タイロッドエンドブーツ", "タイロッドブーツ"] },
  { id: "stabilizer_link_boot", category: "スタビリンクブーツ", group: "ブレーキ・足回り", aliases: ["スタビリンクブーツ", "スタビライザーリンクブーツ"] },
  { id: "brake_pad_front", category: "フロントブレーキパッド", group: "ブレーキ・足回り", aliases: ["フロントブレーキパッド", "Fブレーキパッド", "前ブレーキパッド"] },
  { id: "brake_pad_rear", category: "リアブレーキパッド", group: "ブレーキ・足回り", aliases: ["リアブレーキパッド", "Rブレーキパッド", "後ろブレーキパッド"] },
  { id: "brake_lining", category: "ブレーキライニング", group: "ブレーキ・足回り", aliases: ["ブレーキライニング", "ライニング"] },

  // タイヤ・車体
  { id: "tire", category: "タイヤ残量", group: "タイヤ・車体", aliases: ["タイヤ残量", "タイヤ"] },
  { id: "tire_overhang", category: "タイヤはみ出し", group: "タイヤ・車体", aliases: ["タイヤはみ出し", "はみ出し"] },
  { id: "ground_clearance", category: "最低地上高", group: "タイヤ・車体", aliases: ["最低地上高", "車高"] },

  // 保安・車検関連
  { id: "shift_indicator", category: "シフト表示", group: "保安・車検関連", aliases: ["シフト表示"] },
  { id: "electrical", category: "電気回り", group: "保安・車検関連", aliases: ["電気回り", "電装"] },
  { id: "wiper_blade", category: "ワイパーゴム", group: "保安・車検関連", aliases: ["ワイパーゴム", "ワイパー"] },
  { id: "catalyst", category: "触媒", group: "保安・車検関連", aliases: ["触媒", "マフラー触媒"] },
  { id: "seatbelt_warning", category: "シートベルト警告灯", group: "保安・車検関連", aliases: ["シートベルト警告灯", "シートベルトランプ"] },
  { id: "flare", category: "発煙筒", group: "保安・車検関連", aliases: ["発煙筒"] },
  { id: "window_operation", category: "ウインドウ作動", group: "保安・車検関連", aliases: ["ウインドウ作動", "パワーウインドウ"] },
  { id: "co_hc", category: "CO・HC", group: "保安・車検関連", aliases: ["CO", "HC"] },
];
