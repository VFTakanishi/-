import type { Box, Point } from "../types";

export const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

export const buildBoundingBox = (points: Point[]): Box => {
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);

  return {
    minX: Math.min(...xs),
    minY: Math.min(...ys),
    maxX: Math.max(...xs),
    maxY: Math.max(...ys)
  };
};

export const expandBox = (box: Box, ratio: number): Box => {
  const width = box.maxX - box.minX;
  const height = box.maxY - box.minY;
  const padX = width * ratio;
  const padY = height * ratio;

  return {
    minX: clamp(box.minX - padX, 0, 1),
    minY: clamp(box.minY - padY, 0, 1),
    maxX: clamp(box.maxX + padX, 0, 1),
    maxY: clamp(box.maxY + padY, 0, 1)
  };
};

export const pointInBox = (point: Point, box: Box) =>
  point.x >= box.minX &&
  point.x <= box.maxX &&
  point.y >= box.minY &&
  point.y <= box.maxY;

export const distance = (a: Point, b: Point) =>
  Math.hypot(a.x - b.x, a.y - b.y);

export const averagePoint = (points: Point[]): Point => {
  const total = points.reduce(
    (sum, point) => ({
      x: sum.x + point.x,
      y: sum.y + point.y
    }),
    { x: 0, y: 0 }
  );

  return {
    x: total.x / points.length,
    y: total.y / points.length
  };
};
