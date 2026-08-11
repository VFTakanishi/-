import type { DreamDatabase } from '../db/types';
import type { FileStorageService } from '../files/types';
import type { DreamRecording, OrphanFile } from '../types/models';

export type IntegrityCheckResult = {
  /** ステータスを修正したレコード（transcribing/processingで終了していたものなど） */
  repairedRecordings: DreamRecording[];
  /** 音声ファイルが見つからず error にしたレコード */
  missingFileRecordings: DreamRecording[];
  /** 0バイトだったため error にしたレコード */
  emptyFileRecordings: DreamRecording[];
  /** DBレコードの無い孤立ファイル（削除はしない。復旧候補として提示する） */
  orphanFiles: OrphanFile[];
};

/**
 * アプリ起動のたびに実行する整合性検査。
 * - 孤立ファイルは削除しない（復旧候補として返すだけ）
 * - 欠損レコードは削除しない（error状態にするだけ）
 * - transcribing/processingのまま終了した処理は再試行可能な状態へ戻す
 */
export async function runStartupIntegrityCheck(
  db: DreamDatabase,
  files: FileStorageService,
  nowIso: () => string
): Promise<IntegrityCheckResult> {
  const recordings = await db.recordings.getAll();
  const knownFileUris = new Set(recordings.map((r) => r.fileUri));

  const repairedRecordings: DreamRecording[] = [];
  const missingFileRecordings: DreamRecording[] = [];
  const emptyFileRecordings: DreamRecording[] = [];

  for (const recording of recordings) {
    // 1. 中断された処理を再試行可能な状態へ戻す
    if (recording.status === 'transcribing') {
      await db.recordings.update(recording.id, { status: 'saved' });
      repairedRecordings.push({ ...recording, status: 'saved' });
      continue;
    }
    if (recording.status === 'processing') {
      const fallback = recording.transcript ? 'transcribed' : 'saved';
      await db.recordings.update(recording.id, { status: fallback });
      repairedRecordings.push({ ...recording, status: fallback });
      continue;
    }

    // 2. 音声ファイルの存在・サイズ確認
    const info = await files.getFileInfo(recording.fileUri);
    if (!info.exists) {
      if (recording.status !== 'error') {
        await db.recordings.update(recording.id, {
          status: 'error',
          errorMessage: 'missing_file',
        });
      }
      missingFileRecordings.push({ ...recording, status: 'error', errorMessage: 'missing_file' });
      continue;
    }
    if (info.size < 1) {
      if (recording.status !== 'error') {
        await db.recordings.update(recording.id, {
          status: 'error',
          errorMessage: 'empty_file',
        });
      }
      emptyFileRecordings.push({ ...recording, status: 'error', errorMessage: 'empty_file' });
    }
  }

  // 3. 孤立ファイル検出（削除しない・登録候補として返すだけ）
  const persistedFiles = await files.listPersistedFiles();
  const orphanFiles: OrphanFile[] = persistedFiles
    .filter((f) => !knownFileUris.has(f.uri))
    .map((f) => ({ fileUri: f.uri, fileSize: f.size, detectedAt: nowIso() }));

  return { repairedRecordings, missingFileRecordings, emptyFileRecordings, orphanFiles };
}
