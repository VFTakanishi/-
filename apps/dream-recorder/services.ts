import { createSqliteDatabase } from '../db/sqliteDatabase';
import { ExpoAudioRecordingService } from '../audio/expoAudioRecordingService';
import { ExpoHapticsService } from '../haptics/expoHapticsService';
import { ExpoFileStorageService } from '../files/expoFileStorageService';
import { SystemClock } from '../core/clock';
import { generateId } from '../ids/uuid';
import { RecordingController } from '../core/RecordingController';
import { OnDeviceTranscriptionProvider } from '../transcription/onDeviceProvider';
import { ServerTranscriptionProvider } from '../transcription/serverProvider';
import type { TranscriptionProvider } from '../transcription/types';
import type { DreamDatabase } from '../db/types';

/**
 * 実機用サービスのシングルトン群。
 * アプリ全体で1つのRecordingController・1つのDB接続を共有する
 * （Reactのstateを正本にせず、これらのサービス経由で常にSQLite/ファイルを読み書きする）。
 */
export const db: DreamDatabase = createSqliteDatabase();
export const files = new ExpoFileStorageService();
export const audio = new ExpoAudioRecordingService();
export const haptics = new ExpoHapticsService();
export const clock = new SystemClock();

export const recordingController = new RecordingController({
  db,
  audio,
  haptics,
  files,
  clock,
  generateId,
});

// 優先順位: 1. iOSオンデバイスSpeech framework 2. ユーザー設定のサーバーAPI
export const transcriptionProviders: TranscriptionProvider[] = [
  new OnDeviceTranscriptionProvider(),
  new ServerTranscriptionProvider(),
];
