import { detectionConfig } from "../config/detectionConfig";
import { distance } from "./geometry";
import type { Point } from "../types";

export type MotionSample = {
  point: Point;
  timestamp: number;
};

export type MotionStats = {
  pathDistance: number;
  netDisplacement: number;
  averageStep: number;
  directionChanges: number;
  isPattingLike: boolean;
};

export const summarizeMotion = (samples: MotionSample[]): MotionStats => {
  if (samples.length < 3) {
    return {
      pathDistance: 0,
      netDisplacement: 0,
      averageStep: 0,
      directionChanges: 0,
      isPattingLike: false
    };
  }

  let pathDistance = 0;
  let directionChanges = 0;
  let previousDirection: Point | null = null;

  for (let index = 1; index < samples.length; index += 1) {
    const prev = samples[index - 1].point;
    const current = samples[index].point;
    pathDistance += distance(prev, current);

    const direction = {
      x: current.x - prev.x,
      y: current.y - prev.y
    };

    if (previousDirection) {
      const dot =
        previousDirection.x * direction.x + previousDirection.y * direction.y;
      if (dot < 0) {
        directionChanges += 1;
      }
    }

    previousDirection = direction;
  }

  const netDisplacement = distance(
    samples[0].point,
    samples[samples.length - 1].point
  );
  const averageStep = pathDistance / (samples.length - 1);

  const isPattingLike =
    pathDistance >= detectionConfig.minPathDistanceRatio &&
    netDisplacement <= detectionConfig.maxNetDisplacementRatio &&
    averageStep >= detectionConfig.minAverageStepRatio &&
    averageStep <= detectionConfig.maxAverageStepRatio &&
    directionChanges >= detectionConfig.minDirectionChanges;

  return {
    pathDistance,
    netDisplacement,
    averageStep,
    directionChanges,
    isPattingLike
  };
};
