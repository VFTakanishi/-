import type { DetectionConfig } from "../config/detectionConfig";

export type LearningAction = "step_back" | "step_forward" | "reset";

export type LearningSnapshot = {
  phase: string;
  totalNearFaceMs: number;
  visitedZoneCount: number;
  pickupLikely: boolean;
  pickupReturnConfirmed: boolean;
};

export type LearningEntry = {
  id: string;
  timestamp: string;
  action: LearningAction;
  stageBefore: number;
  stageAfter: number;
  snapshot: LearningSnapshot;
};

export type LearningAdjustments = {
  completionAwayMs: number;
  minVisitedZones: number;
  minHandPointsInsideFace: number;
  minFingerTipsInsideContactZone: number;
  offscreenPickupMinMs: number;
};

export type LearningProfile = {
  version: number;
  updatedAt: string;
  falsePositiveCount: number;
  falseNegativeCount: number;
  entries: LearningEntry[];
  adjustments: LearningAdjustments;
};

const learningStorageKey = "youtube-live-skincare-learning-profile";
const maxEntries = 240;

const defaultAdjustments = (): LearningAdjustments => ({
  completionAwayMs: 0,
  minVisitedZones: 0,
  minHandPointsInsideFace: 0,
  minFingerTipsInsideContactZone: 0,
  offscreenPickupMinMs: 0
});

export const createDefaultLearningProfile = (): LearningProfile => ({
  version: 1,
  updatedAt: new Date().toISOString(),
  falsePositiveCount: 0,
  falseNegativeCount: 0,
  entries: [],
  adjustments: defaultAdjustments()
});

export const loadLearningProfile = (): LearningProfile => {
  if (typeof window === "undefined") {
    return createDefaultLearningProfile();
  }

  try {
    const raw = window.localStorage.getItem(learningStorageKey);
    if (!raw) {
      return createDefaultLearningProfile();
    }

    const parsed = JSON.parse(raw) as Partial<LearningProfile>;
    return {
      ...createDefaultLearningProfile(),
      ...parsed,
      adjustments: {
        ...defaultAdjustments(),
        ...parsed.adjustments
      },
      entries: Array.isArray(parsed.entries) ? parsed.entries.slice(0, maxEntries) : []
    };
  } catch {
    return createDefaultLearningProfile();
  }
};

export const saveLearningProfile = (profile: LearningProfile) => {
  if (typeof window === "undefined") {
    return;
  }

  window.localStorage.setItem(learningStorageKey, JSON.stringify(profile));
};

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

export const recordLearningFeedback = (
  profile: LearningProfile,
  entry: LearningEntry
): LearningProfile => {
  const next = {
    ...profile,
    updatedAt: new Date().toISOString(),
    entries: [entry, ...profile.entries].slice(0, maxEntries),
    adjustments: {
      ...profile.adjustments
    }
  };

  if (entry.action === "step_back") {
    next.falsePositiveCount += 1;
    next.adjustments.minHandPointsInsideFace = clamp(
      next.adjustments.minHandPointsInsideFace + 1,
      0,
      4
    );
    next.adjustments.minFingerTipsInsideContactZone = clamp(
      next.adjustments.minFingerTipsInsideContactZone + 1,
      0,
      2
    );
    next.adjustments.minVisitedZones = clamp(
      next.adjustments.minVisitedZones + 1,
      0,
      1
    );
    next.adjustments.offscreenPickupMinMs = clamp(
      next.adjustments.offscreenPickupMinMs + 150,
      0,
      1200
    );
    next.adjustments.completionAwayMs = clamp(
      next.adjustments.completionAwayMs + 300,
      0,
      2000
    );
  }

  if (entry.action === "step_forward") {
    next.falseNegativeCount += 1;
    next.adjustments.minHandPointsInsideFace = clamp(
      next.adjustments.minHandPointsInsideFace - 1,
      -2,
      4
    );
    next.adjustments.minFingerTipsInsideContactZone = clamp(
      next.adjustments.minFingerTipsInsideContactZone - 1,
      -1,
      2
    );
    next.adjustments.minVisitedZones = clamp(
      next.adjustments.minVisitedZones - 1,
      -1,
      1
    );
    next.adjustments.offscreenPickupMinMs = clamp(
      next.adjustments.offscreenPickupMinMs - 150,
      -600,
      1200
    );
    next.adjustments.completionAwayMs = clamp(
      next.adjustments.completionAwayMs - 300,
      -1500,
      2000
    );
  }

  return next;
};

export const deriveAdaptiveConfig = (
  base: DetectionConfig,
  profile: LearningProfile
): DetectionConfig => ({
  ...base,
  completionAwayMs: clamp(
    base.completionAwayMs + profile.adjustments.completionAwayMs,
    1500,
    7000
  ),
  minVisitedZones: clamp(
    base.minVisitedZones + profile.adjustments.minVisitedZones,
    1,
    3
  ),
  minHandPointsInsideFace: clamp(
    base.minHandPointsInsideFace + profile.adjustments.minHandPointsInsideFace,
    3,
    10
  ),
  minFingerTipsInsideContactZone: clamp(
    base.minFingerTipsInsideContactZone +
      profile.adjustments.minFingerTipsInsideContactZone,
    1,
    4
  ),
  offscreenPickupMinMs: clamp(
    base.offscreenPickupMinMs + profile.adjustments.offscreenPickupMinMs,
    500,
    2200
  )
});
