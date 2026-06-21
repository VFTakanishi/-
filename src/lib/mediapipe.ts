import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker
} from "@mediapipe/tasks-vision";
import { detectionConfig } from "../config/detectionConfig";

const wasmRoot =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";

const faceModelUrl =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const handModelUrl =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

export type VisionTasks = {
  faceLandmarker: FaceLandmarker;
  handLandmarker: HandLandmarker;
};

let visionTasksPromise: Promise<VisionTasks> | null = null;

export const createVisionTasks = async (): Promise<VisionTasks> => {
  if (visionTasksPromise) {
    return visionTasksPromise;
  }

  visionTasksPromise = (async () => {
    const vision = await FilesetResolver.forVisionTasks(wasmRoot);

    const [faceLandmarker, handLandmarker] = await Promise.all([
      FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: faceModelUrl
        },
        runningMode: "VIDEO",
        numFaces: 1,
        minFaceDetectionConfidence: detectionConfig.faceDetectionConfidence,
        minFacePresenceConfidence: detectionConfig.facePresenceConfidence,
        minTrackingConfidence: detectionConfig.faceTrackingConfidence
      }),
      HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: handModelUrl
        },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: detectionConfig.handDetectionConfidence,
        minHandPresenceConfidence: detectionConfig.handPresenceConfidence
      })
    ]);

    return {
      faceLandmarker,
      handLandmarker
    };
  })();

  return visionTasksPromise;
};
