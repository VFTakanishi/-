import type { DreamRecording } from '../types/models';

function formatTimeHHmm(iso: string): string {
  const d = new Date(iso);
  const hh = String(d.getHours()).padStart(2, '0');
  const mm = String(d.getMinutes()).padStart(2, '0');
  return `${hh}:${mm}`;
}

/**
 * 文字起こし済みの録音を時刻順に並べ、GPT貼り付け用の文章を組み立てる。
 * 文字起こしが無い録音は含めない（推測で内容を補わないため）。
 */
export function buildGptPasteText(recordings: DreamRecording[]): string {
  const transcribed = recordings
    .filter((r) => r.transcript && r.transcript.trim().length > 0)
    .slice()
    .sort((a, b) => a.createdAt.localeCompare(b.createdAt));

  const header = [
    '以下は昨夜見た夢の音声記録です。',
    '内容を推測で追加せず、複数の夢を時刻順に整理してください。',
    '夢の意味や心理分析は不要です。',
  ].join('\n');

  if (transcribed.length === 0) {
    return header;
  }

  const body = transcribed
    .map((r, i) => `【録音${i + 1}：${formatTimeHHmm(r.createdAt)}】\n${r.transcript}`)
    .join('\n\n');

  return `${header}\n\n${body}`;
}
