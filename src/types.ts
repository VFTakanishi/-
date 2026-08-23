export type JudgementStatus =
  | "ng"
  | "replace_strong"
  | "recommend"
  | "na"
  | "good"
  | "customer_request"
  | "customer_declined"
  | "unset";

export interface ItemPosition {
  frontRear?: "front" | "rear";
  innerOuter?: "inner" | "outer";
}

export interface ItemMeasurement {
  value?: number;
  unit?: string;
}

export interface InspectionItem {
  id: string;
  category: string;
  status: JudgementStatus;
  note?: string;
  position?: ItemPosition;
  measurement?: ItemMeasurement;
  measurements?: Record<string, number>;
  isCustom?: boolean;
  updatedAt?: string;
}

export interface Inspection {
  id: string;
  customerName?: string;
  vehicleModel?: string;
  createdAt: string;
  updatedAt: string;
  items: InspectionItem[];
}

export interface ChecklistItemDef {
  id: string;
  category: string;
  aliases: string[];
  group: string;
}
