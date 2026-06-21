import { useEffect, useMemo, useRef, useState } from "react";
import type {
  FaceLandmarkerResult,
  HandLandmarkerResult
} from "@mediapipe/tasks-vision";
import { detectionConfig } from "./config/detectionConfig";
import {
  averagePoint,
  buildBoundingBox,
  distance,
  expandBox,
  pointInBox
} from "./lib/geometry";
import {
  createDefaultLearningProfile,
  deriveAdaptiveConfig,
  loadLearningProfile,
  recordLearningFeedback,
  saveLearningProfile,
  type LearningEntry,
  type LearningProfile,
  type LearningSnapshot
} from "./lib/learning";
import { createVisionTasks } from "./lib/mediapipe";
import { summarizeMotion, type MotionSample } from "./lib/motion";
import type { Box, DetectionLog, Point, StageStatus } from "./types";

const stageNames = [
  "セラムベールディープリペア",
  "メラノショットP",
  "セラムシールド"
] as const;

const totalStages = stageNames.length;
const faceZones = ["left-cheek", "right-cheek", "forehead", "mouth"] as const;
const fingerTipIndices = [4, 8, 12, 16, 20] as const;

type AnalysisState = "idle" | "preparing" | "sharing" | "cooldown" | "error";
type FaceZone = (typeof faceZones)[number];
type DetectionPhase = "idle" | "candidate" | "applying" | "awaiting-completion";

type PresenceState = {
  phase: DetectionPhase;
  candidateStartedAt: number | null;
  applyingStartedAt: number | null;
  lastNearFaceAt: number | null;
  lastAnyHandAt: number | null;
  totalNearFaceMs: number;
  nearFaceSamples: MotionSample[];
  zoneVisits: FaceZone[];
  completionStartedAt: number | null;
  completionSamples: MotionSample[];
  offscreenStartedAt: number | null;
  pickupLikely: boolean;
  pickupReadyAt: number | null;
  pickupReturnConfirmed: boolean;
};

type NextStageGate = {
  awaitingPickup: boolean;
  offscreenStartedAt: number | null;
  pickupDetectedAt: number | null;
};

type DisplayMediaOptionsWithHints = DisplayMediaStreamOptions & {
  preferCurrentTab?: boolean;
  selfBrowserSurface?: "include" | "exclude";
  surfaceSwitching?: "include" | "exclude";
};

const initialPresenceState = (): PresenceState => ({
  phase: "idle",
  candidateStartedAt: null,
  applyingStartedAt: null,
  lastNearFaceAt: null,
  lastAnyHandAt: null,
  totalNearFaceMs: 0,
  nearFaceSamples: [],
  zoneVisits: [],
  completionStartedAt: null,
  completionSamples: [],
  offscreenStartedAt: null,
  pickupLikely: false,
  pickupReadyAt: null,
  pickupReturnConfirmed: false
});

const initialNextStageGate = (): NextStageGate => ({
  awaitingPickup: false,
  offscreenStartedAt: null,
  pickupDetectedAt: null
});

const formatTimestamp = (time: number) =>
  new Intl.DateTimeFormat("ja-JP", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  }).format(time);

const uniqueZones = (zones: FaceZone[]) => [...new Set(zones)];

const insetBox = (box: Box, ratio: number): Box => {
  const width = box.maxX - box.minX;
  const height = box.maxY - box.minY;
  const insetX = width * ratio;
  const insetY = height * ratio;

  return {
    minX: box.minX + insetX,
    minY: box.minY + insetY,
    maxX: box.maxX - insetX,
    maxY: box.maxY - insetY
  };
};

const getFaceZone = (point: Point, faceBox: Box): FaceZone | null => {
  const width = faceBox.maxX - faceBox.minX;
  const height = faceBox.maxY - faceBox.minY;
  if (width <= 0 || height <= 0) {
    return null;
  }

  const relX = (point.x - faceBox.minX) / width;
  const relY = (point.y - faceBox.minY) / height;

  if (relY < 0.28) {
    return "forehead";
  }
  if (relY > 0.72) {
    return "mouth";
  }
  return relX < 0.5 ? "left-cheek" : "right-cheek";
};

const isPointNearScreenEdge = (point: Point, ratio: number) =>
  point.x <= ratio ||
  point.x >= 1 - ratio ||
  point.y <= ratio ||
  point.y >= 1 - ratio;

function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationRef = useRef<number | null>(null);
  const cooldownUntilRef = useRef<number>(0);
  const presenceRef = useRef<PresenceState>(initialPresenceState());
  const nextStageGateRef = useRef<NextStageGate>(initialNextStageGate());
  const detectedCountRef = useRef(0);
  const logIdRef = useRef(0);

  const [analysisState, setAnalysisState] = useState<AnalysisState>("idle");
  const [currentStage, setCurrentStage] = useState(0);
  const [cooldownRemainingMs, setCooldownRemainingMs] = useState(0);
  const [logs, setLogs] = useState<DetectionLog[]>([]);
  const [errorMessage, setErrorMessage] = useState("");
  const [isCopySuccess, setIsCopySuccess] = useState(false);
  const [isModelReady, setIsModelReady] = useState(false);
  const [learningProfile, setLearningProfile] = useState<LearningProfile>(
    createDefaultLearningProfile()
  );

  useEffect(() => {
    setLearningProfile(loadLearningProfile());
  }, []);

  const adaptiveConfig = useMemo(
    () => deriveAdaptiveConfig(detectionConfig, learningProfile),
    [learningProfile]
  );

  const pushLog = (message: string) => {
    const entry: DetectionLog = {
      id: logIdRef.current++,
      timestamp: formatTimestamp(Date.now()),
      message
    };

    setLogs((currentLogs) =>
      [entry, ...currentLogs].slice(0, detectionConfig.maxLogs)
    );
  };

  const buildLearningSnapshot = (): LearningSnapshot => {
    const presence = presenceRef.current;

    return {
      phase: presence.phase,
      totalNearFaceMs: presence.totalNearFaceMs,
      visitedZoneCount: uniqueZones(presence.zoneVisits).length,
      pickupLikely: presence.pickupLikely,
      pickupReturnConfirmed: presence.pickupReturnConfirmed
    };
  };

  const persistLearningFeedback = (
    action: LearningEntry["action"],
    stageBefore: number,
    stageAfter: number
  ) => {
    const entry: LearningEntry = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      timestamp: new Date().toISOString(),
      action,
      stageBefore,
      stageAfter,
      snapshot: buildLearningSnapshot()
    };

    setLearningProfile((currentProfile) => {
      const nextProfile = recordLearningFeedback(currentProfile, entry);
      saveLearningProfile(nextProfile);
      return nextProfile;
    });
  };

  const resetPresenceState = (message?: string) => {
    if (message) {
      pushLog(message);
    }
    presenceRef.current = initialPresenceState();
  };

  const resetNextStageGate = () => {
    nextStageGateRef.current = initialNextStageGate();
  };

  const stopSharing = () => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }

    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    presenceRef.current = initialPresenceState();
    resetNextStageGate();

    const video = videoRef.current;
    if (video) {
      video.pause();
      video.srcObject = null;
    }
  };

  useEffect(() => stopSharing, []);

  useEffect(() => {
    detectedCountRef.current = currentStage;
  }, [currentStage]);

  const stageStatuses = useMemo<StageStatus[]>(
    () =>
      Array.from({ length: totalStages }, (_, index) => {
        if (index < currentStage) {
          return "done";
        }
        if (index === currentStage && currentStage < totalStages) {
          return "current";
        }
        return "todo";
      }),
    [currentStage]
  );

  const commentText = useMemo(() => {
    const nextStage = Math.min(currentStage + 1, totalStages);
    if (currentStage >= totalStages) {
      return "3工程まで進んだ可能性があります。見落としがあれば教えてください。";
    }
    return `${stageNames[nextStage - 1]} がまだの可能性があります`;
  }, [currentStage]);

  const updateCooldownState = (now: number) => {
    const remaining = Math.max(0, cooldownUntilRef.current - now);
    setCooldownRemainingMs(remaining);

    if (remaining > 0) {
      setAnalysisState("cooldown");
    } else if (analysisState !== "sharing") {
      setAnalysisState("sharing");
    }
  };

  const drawOverlay = (
    faceBox: Box | null,
    handPoints: Point[],
    videoWidth: number,
    videoHeight: number
  ) => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    canvas.width = videoWidth;
    canvas.height = videoHeight;

    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    context.clearRect(0, 0, canvas.width, canvas.height);

    if (faceBox) {
      context.strokeStyle = "#55e6a5";
      context.lineWidth = 3;
      context.strokeRect(
        faceBox.minX * canvas.width,
        faceBox.minY * canvas.height,
        (faceBox.maxX - faceBox.minX) * canvas.width,
        (faceBox.maxY - faceBox.minY) * canvas.height
      );
    }

    handPoints.forEach((point) => {
      context.beginPath();
      context.arc(point.x * canvas.width, point.y * canvas.height, 5, 0, Math.PI * 2);
      context.fillStyle = "#ffd166";
      context.fill();
    });
  };

  const trimSamples = (samples: MotionSample[], now: number) =>
    samples.filter(
      (sample) => now - sample.timestamp <= adaptiveConfig.sampleWindowMs
    );

  const completeStage = (reason: string, now: number) => {
    const nextCount = Math.min(detectedCountRef.current + 1, totalStages);
    detectedCountRef.current = nextCount;
    setCurrentStage(nextCount);
    cooldownUntilRef.current = now + adaptiveConfig.cooldownMs;
    presenceRef.current = initialPresenceState();
    nextStageGateRef.current = {
      awaitingPickup: false,
      offscreenStartedAt: null,
      pickupDetectedAt: null
    };
    setAnalysisState("cooldown");
    pushLog(`${reason} 工程を ${nextCount} / ${totalStages} に更新しました。`);
  };

  const updateNextStageGate = (handDetected: boolean, handNearFace: boolean, now: number) => {
    const gate = nextStageGateRef.current;
    const needsPickup = gate.awaitingPickup && detectedCountRef.current > 0;

    if (!needsPickup) {
      return true;
    }

    if (!handDetected) {
      if (!gate.offscreenStartedAt) {
        gate.offscreenStartedAt = now;
      } else if (
        !gate.pickupDetectedAt &&
        now - gate.offscreenStartedAt >= adaptiveConfig.offscreenPickupMinMs
      ) {
        gate.pickupDetectedAt = now;
        pushLog("次の工程候補として、画面外で新しい商品を取った可能性があります。");
      }
      return false;
    }

    if (gate.pickupDetectedAt) {
      if (handNearFace && now - gate.pickupDetectedAt <= adaptiveConfig.offscreenReturnWindowMs) {
        gate.awaitingPickup = false;
        gate.offscreenStartedAt = null;
        gate.pickupDetectedAt = null;
        pushLog("新しい商品を取って顔へ戻った可能性があります。次の工程の判定を再開します。");
        return true;
      }

      if (now - gate.pickupDetectedAt > adaptiveConfig.offscreenReturnWindowMs) {
        gate.offscreenStartedAt = null;
        gate.pickupDetectedAt = null;
      }
    }

    if (handDetected) {
      gate.offscreenStartedAt = null;
    }

    return !gate.awaitingPickup;
  };

  const updatePresenceState = (
    handNearFace: boolean,
    handDetected: boolean,
    handPoint: Point | null,
    faceZone: FaceZone | null,
    handAtEdge: boolean,
    now: number
  ) => {
    const canStartNextStage = updateNextStageGate(handDetected, handNearFace, now);
    const presence = presenceRef.current;

    if (!canStartNextStage && presence.phase === "idle") {
      return;
    }

    if (handDetected) {
      presence.lastAnyHandAt = now;
    }

    if (handNearFace && handPoint) {
      if (presence.phase === "idle") {
        presence.phase = "candidate";
        presence.candidateStartedAt = now;
        pushLog("顔付近に片手または両手を検出。塗布候補を追跡します。");
      }

      if (presence.phase === "awaiting-completion") {
        presence.phase = "applying";
        presence.completionStartedAt = null;
        pushLog("手が顔付近へ戻りました。同じ工程の塗布継続として扱います。");
      }

      if (
        presence.phase === "candidate" &&
        presence.candidateStartedAt &&
        now - presence.candidateStartedAt >= adaptiveConfig.requiredContinuousMs
      ) {
        presence.phase = "applying";
        presence.applyingStartedAt = presence.candidateStartedAt;
        pushLog("塗布中に入りました。完了タイミングを追跡します。");
      }

      if (presence.lastNearFaceAt) {
        presence.totalNearFaceMs += now - presence.lastNearFaceAt;
      }
      presence.lastNearFaceAt = now;
      presence.nearFaceSamples.push({ point: handPoint, timestamp: now });
      presence.nearFaceSamples = trimSamples(presence.nearFaceSamples, now);
      presence.offscreenStartedAt = null;

      if (faceZone) {
        const lastZone = presence.zoneVisits[presence.zoneVisits.length - 1];
        if (lastZone !== faceZone) {
          presence.zoneVisits.push(faceZone);
          presence.zoneVisits = presence.zoneVisits.slice(-8);
        }
      }

      if (presence.phase === "applying") {
        presence.completionSamples.push({ point: handPoint, timestamp: now });
        presence.completionSamples = trimSamples(presence.completionSamples, now);
      }

      return;
    }

    if (
      presence.lastNearFaceAt &&
      now - presence.lastNearFaceAt <= adaptiveConfig.presenceResetGraceMs
    ) {
      return;
    }

    presence.lastNearFaceAt = null;

    if (
      handDetected &&
      !handNearFace &&
      presence.pickupLikely &&
      !presence.pickupReturnConfirmed &&
      !handAtEdge &&
      presence.pickupReadyAt !== null &&
      now - presence.pickupReadyAt >= 300 &&
      presence.phase === "awaiting-completion"
    ) {
      presence.pickupReturnConfirmed = true;
      pushLog("商品方向への離脱後に手が画面内へ戻りました。次の商品を持った可能性があります。");
    }

    if (
      (!handDetected || handAtEdge) &&
      (presence.phase === "applying" || presence.phase === "awaiting-completion")
    ) {
      if (!presence.offscreenStartedAt) {
        presence.offscreenStartedAt = now;
      } else if (
        !presence.pickupLikely &&
        now - presence.offscreenStartedAt >= adaptiveConfig.offscreenPickupMinMs
      ) {
        presence.pickupLikely = true;
        presence.pickupReadyAt = now;
        presence.pickupReturnConfirmed = false;
        pushLog(
          handAtEdge
            ? "手が画面端へ移動しました。次の商品を取った可能性があります。"
            : "手が画面外へ出ました。次の商品を取った可能性があります。"
        );
      }
    }

    if (presence.phase === "candidate") {
      resetPresenceState("手が顔付近から離れたため、候補追跡をリセットしました。");
      return;
    }

    if (presence.phase === "applying") {
      presence.phase = "awaiting-completion";
      presence.completionStartedAt = now;
      pushLog("手が顔から離れました。工程完了かどうか確認します。");
    }

    if (presence.phase === "awaiting-completion") {
      if (!presence.completionStartedAt) {
        presence.completionStartedAt = now;
      }

      const visitedZoneCount = uniqueZones(presence.zoneVisits).length;
      const longPresenceDetected =
        presence.totalNearFaceMs >= adaptiveConfig.longPresenceMs;
      const motionDetected = summarizeMotion(presence.nearFaceSamples).isPattingLike;
      const completionMotionDetected =
        summarizeMotion(presence.completionSamples).isPattingLike;
      const applyingDetected =
        longPresenceDetected ||
        motionDetected ||
        completionMotionDetected ||
        visitedZoneCount >= adaptiveConfig.minVisitedZones ||
        presence.pickupReturnConfirmed;

      if (presence.pickupReturnConfirmed && applyingDetected) {
        const reason =
          handAtEdge
            ? "塗布後に商品方向へ離脱し、手が戻ってきたため、次の商品へ移ったと判定しました。"
            : "塗布後に商品を取りに行って手が戻ってきたため、次の商品へ移ったと判定しました。";
        completeStage(reason, now);
        return;
      }

      if (now - presence.completionStartedAt >= adaptiveConfig.candidateTimeoutMs) {
        resetPresenceState("完了判定が長引いたため、今回の追跡をリセットしました。");
      }
    }
  };

  const getFaceBox = (result: FaceLandmarkerResult): Box | null => {
    const firstFace = result.faceLandmarks[0];
    if (!firstFace) {
      return null;
    }

    const points = firstFace.map((landmark) => ({
      x: landmark.x,
      y: landmark.y
    }));

    return expandBox(buildBoundingBox(points), adaptiveConfig.facePaddingRatio);
  };

  const getHandPointsNearFace = (
    result: HandLandmarkerResult,
    faceBox: Box | null
  ): Point[] => {
    if (!faceBox) {
      return [];
    }

    const contactBox = insetBox(faceBox, adaptiveConfig.contactZoneInsetRatio);
    const faceCenter = {
      x: (faceBox.minX + faceBox.maxX) / 2,
      y: (faceBox.minY + faceBox.maxY) / 2
    };
    const faceSize = Math.max(
      faceBox.maxX - faceBox.minX,
      faceBox.maxY - faceBox.minY
    );

    return result.landmarks
      .map((hand) =>
        hand.map((landmark) => ({
          x: landmark.x,
          y: landmark.y
        }))
      )
      .map((points) => {
        const centroid = averagePoint(points);
        const insideCount = points.filter((point) => pointInBox(point, faceBox)).length;
        const fingerTipsInsideCount = fingerTipIndices.filter((index) =>
          pointInBox(points[index], contactBox)
        ).length;
        const centroidDistanceRatio =
          faceSize > 0 ? distance(centroid, faceCenter) / faceSize : Number.POSITIVE_INFINITY;

        return {
          centroid,
          insideCount,
          fingerTipsInsideCount,
          centroidDistanceRatio
        };
      })
      .filter(
        (hand) =>
          hand.insideCount >= adaptiveConfig.minHandPointsInsideFace &&
          hand.fingerTipsInsideCount >=
            adaptiveConfig.minFingerTipsInsideContactZone &&
          hand.centroidDistanceRatio <= adaptiveConfig.maxHandCentroidDistanceRatio
      )
      .map((hand) => hand.centroid);
  };

  const hasHandNearScreenEdge = (result: HandLandmarkerResult) =>
    result.landmarks.some((hand) =>
      hand.some((landmark) =>
        isPointNearScreenEdge(
          { x: landmark.x, y: landmark.y },
          adaptiveConfig.screenEdgePickupRatio
        )
      )
    );

  const analyzeFrame = async () => {
    const video = videoRef.current;
    if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      animationRef.current = requestAnimationFrame(() => {
        void analyzeFrame();
      });
      return;
    }

    try {
      const tasks = await createVisionTasks();
      if (!isModelReady) {
        setIsModelReady(true);
      }

      const videoNow = performance.now();
      const wallClockNow = Date.now();
      updateCooldownState(wallClockNow);

      const faceResult = tasks.faceLandmarker.detectForVideo(video, videoNow);
      const handResult = tasks.handLandmarker.detectForVideo(video, videoNow);
      const faceBox = getFaceBox(faceResult);
      const nearFaceHands = getHandPointsNearFace(handResult, faceBox);
      const handAtEdge = hasHandNearScreenEdge(handResult);
      const primaryHandPoint = nearFaceHands[0] ?? null;
      const primaryZone =
        faceBox && primaryHandPoint ? getFaceZone(primaryHandPoint, faceBox) : null;
      const handDetected = handResult.landmarks.length > 0;
      const allHandPoints = handResult.landmarks.flat().map((landmark) => ({
        x: landmark.x,
        y: landmark.y
      }));

      drawOverlay(faceBox, allHandPoints, video.videoWidth, video.videoHeight);
      updatePresenceState(
        nearFaceHands.length > 0,
        handDetected,
        primaryHandPoint,
        primaryZone,
        handAtEdge,
        wallClockNow
      );
      setErrorMessage("");
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "解析中に不明なエラーが発生しました。";
      setAnalysisState("error");
      setErrorMessage(message);
    }

    animationRef.current = requestAnimationFrame(() => {
      void analyzeFrame();
    });
  };

  const startShare = async () => {
    stopSharing();
    setAnalysisState("preparing");
    setErrorMessage("");
    setIsCopySuccess(false);
    pushLog(
      "画面共有の準備を開始しました。YouTubeライブのタブを選択してください。"
    );

    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: {
          frameRate: { ideal: 24, max: 30 },
          width: { ideal: 1920 },
          height: { ideal: 1080 }
        },
        audio: false,
        preferCurrentTab: false,
        selfBrowserSurface: "exclude",
        surfaceSwitching: "include"
      } as DisplayMediaOptionsWithHints);

      streamRef.current = stream;

      const video = videoRef.current;
      if (!video) {
        throw new Error("映像プレビューの準備に失敗しました。");
      }

      video.srcObject = stream;
      await video.play();
      setAnalysisState("sharing");
      pushLog("共有映像の取得を開始しました。リアルタイム解析を実行中です。");

      const [track] = stream.getVideoTracks();
      track?.addEventListener("ended", () => {
        stopSharing();
        setAnalysisState("idle");
        pushLog("画面共有が終了しました。");
      });

      await analyzeFrame();
    } catch (error) {
      stopSharing();
      setAnalysisState("error");
      const message =
        error instanceof Error
          ? error.message
          : "画面共有の開始に失敗しました。";
      setErrorMessage(message);
      pushLog(`画面共有の開始に失敗しました: ${message}`);
    }
  };

  const handleStepBack = () => {
    const previous = currentStage;
    const next = Math.max(0, currentStage - 1);
    setCurrentStage(next);
    detectedCountRef.current = next;
    persistLearningFeedback("step_back", previous, next);
    pushLog(`手動で 1 つ戻しました。現在 ${next} / ${totalStages} です。`);
  };

  const handleStepForward = () => {
    const previous = currentStage;
    const next = Math.min(totalStages, currentStage + 1);
    setCurrentStage(next);
    detectedCountRef.current = next;
    persistLearningFeedback("step_forward", previous, next);
    pushLog(`手動で 1 つ進めました。現在 ${next} / ${totalStages} です。`);
  };

  const handleReset = () => {
    const previous = currentStage;
    setCurrentStage(0);
    detectedCountRef.current = 0;
    cooldownUntilRef.current = 0;
    presenceRef.current = initialPresenceState();
    resetNextStageGate();
    setCooldownRemainingMs(0);
    persistLearningFeedback("reset", previous, 0);
    pushLog("工程カウントをリセットしました。");
  };

  const handleCopyComment = async () => {
    try {
      await navigator.clipboard.writeText(commentText);
      setIsCopySuccess(true);
      pushLog("コメント文をコピーしました。");
      window.setTimeout(() => setIsCopySuccess(false), 1500);
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "コメントのコピーに失敗しました。";
      setErrorMessage(message);
      pushLog(`コメント文のコピーに失敗しました: ${message}`);
    }
  };

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">YouTube Live Skincare Analyzer</p>
          <h1>視聴者側だけで塗布工程を推定するMVP</h1>
          <p className="hero-copy">
            YouTubeライブのタブを共有し、片手または両手の顔付近滞在、
            顔の部位移動、離脱後に戻らない時間を組み合わせて工程完了を推定します。
          </p>
        </div>
        <div className="status-card">
          <span className={`dot dot-${analysisState}`} />
          <div>
            <strong>{isModelReady ? "解析準備OK" : "初回モデル読込あり"}</strong>
            <p>
              {analysisState === "cooldown"
                ? `クールダウン中: ${Math.ceil(cooldownRemainingMs / 1000)}秒`
                : analysisState === "sharing"
                  ? "リアルタイム解析中"
                  : analysisState === "preparing"
                    ? "共有開始待ち"
                    : analysisState === "error"
                      ? "エラー"
                      : "待機中"}
            </p>
          </div>
        </div>
      </header>

      <main className="layout">
        <section className="panel main-panel">
          <div className="panel-head">
            <h2>共有映像</h2>
            <p>YouTubeライブのブラウザタブ共有を選ぶと、その映像を解析します。</p>
          </div>

          <div className="video-stage">
            <video ref={videoRef} className="video-layer" playsInline muted />
            <canvas ref={canvasRef} className="canvas-layer" />
            {analysisState === "idle" && (
              <div className="placeholder">
                <strong>画面共有開始を押してください</strong>
                <p>
                  共有対象は YouTubeライブのタブまたはライブ再生中のウィンドウを想定しています。
                </p>
              </div>
            )}
          </div>

          <div className="button-row">
            <button className="primary" onClick={() => void startShare()}>
              画面共有開始
            </button>
            <button onClick={handleStepBack}>1つ戻す</button>
            <button onClick={handleStepForward}>1つ進める</button>
            <button onClick={handleReset}>リセット</button>
            <button className="accent" onClick={() => void handleCopyComment()}>
              コメント文をコピー
            </button>
          </div>

          {errorMessage && <p className="error-text">{errorMessage}</p>}
        </section>

        <aside className="side-stack">
          <section className="panel progress-panel">
            <div className="panel-head">
              <h2>現在の推定工程</h2>
              <p className="stage-count">
                {currentStage >= totalStages
                  ? `${totalStages}工程まで完了した可能性があります / 完了 ${currentStage} / ${totalStages}`
                  : `${currentStage + 1}工程目を観測中 / 完了 ${currentStage} / ${totalStages}`}
              </p>
            </div>

            <div className="stage-list">
              {stageStatuses.map((status, index) => (
                <div key={stageNames[index]} className={`stage-item stage-${status}`}>
                  <span className="stage-mark">
                    {status === "done" ? "✅" : status === "current" ? "▶" : "□"}
                  </span>
                  <span>{stageNames[index]}</span>
                </div>
              ))}
            </div>

            <div className="comment-box">
              <p className="comment-label">YouTubeコメント用文言</p>
              <p className="comment-text">{commentText}</p>
              {isCopySuccess && <p className="copy-success">コピーしました</p>}
              <p className="comment-label">
                学習データ保存済み: {learningProfile.entries.length}件
              </p>
            </div>
          </section>

          <section className="panel log-panel">
            <div className="panel-head">
              <h2>検出状態ログ</h2>
              <p>塗布候補、塗布中、顔の部位移動、完了待ち、工程確定を記録します。</p>
            </div>

            <div className="log-list">
              {logs.length === 0 ? (
                <p className="empty-text">まだログはありません。</p>
              ) : (
                logs.map((log) => (
                  <div key={log.id} className="log-item">
                    <span>{log.timestamp}</span>
                    <p>{log.message}</p>
                  </div>
                ))
              )}
            </div>
          </section>
        </aside>
      </main>
    </div>
  );
}

export default App;
