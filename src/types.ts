export type Point = {
  x: number;
  y: number;
};

export type Box = {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
};

export type StageStatus = "done" | "current" | "todo";

export type DetectionLog = {
  id: number;
  timestamp: string;
  message: string;
};
