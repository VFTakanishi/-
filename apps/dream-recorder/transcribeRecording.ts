import type { DreamDatabase } from '../db/types';
import type { DreamRecording } from '../types/models';
import type { TranscriptionProvider } from './types';

/**
 * 文字起こしを優先順位付きプロバイダで試みる。
 * どのプロバイダも使えない／失敗した場合でも、元音声（fileUri）は一切変更・削除しない。
 * OpenAI APIのようなサーバーAPIが未設定でも、この関数を呼ばなければ夜間保存の成功条件には影響しない。
 */
export async function transcribeRecording(
  db: DreamDatabase,
  providers: TranscriptionProvider[],
  recording: DreamRecording
): Promise<DreamRecording> {
  await db.recordings.update(recording.id, { status: 'transcribing', errorMessage: undefined });

  for (const provider of providers) {
    const available = await provider.isAvailable().catch(() => false);
    if (!available) continue;

    const outcome = await provider.transcribe(recording.fileUri).catch((error) => ({
      ok: false as const,
      errorMessage: error instanceof Error ? error.message : String(error),
    }));

    if (outcome.ok) {
      await db.recordings.update(recording.id, {
        status: 'transcribed',
        transcript: outcome.transcript,
        errorMessage: undefined,
      });
      const updated = await db.recordings.getById(recording.id);
      return updated ?? { ...recording, status: 'transcribed', transcript: outcome.transcript };
    }
  }

  // 未処理として保持。元音声(fileUri)には一切触れない。
  await db.recordings.update(recording.id, {
    status: 'saved',
    errorMessage: 'transcription_unavailable',
  });
  const updated = await db.recordings.getById(recording.id);
  return updated ?? { ...recording, status: 'saved', errorMessage: 'transcription_unavailable' };
}
