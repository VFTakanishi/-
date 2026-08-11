/**
 * 永続データの型定義。
 * これらの型がアプリの「正本」であり、Reactのstateはここから復元される
 * ビューにすぎない（メモリ上のstateを正本にしない）。
 */

/** 1回分の録音レコード */
export type DreamRecording = {
  id: string;
  sessionId: string;
  createdAt: string; // ISO 8601
  durationMs: number;
  fileUri: string;
  fileSize: number;
  status:
    | 'saved'
    | 'interrupted'
    | 'transcribing'
    | 'transcribed'
    | 'processing'
    | 'completed'
    | 'error';
  transcript?: string;
  errorMessage?: string;
};

/** 一晩分の睡眠セッション */
export type SleepSession = {
  id: string;
  startedAt: string; // ISO 8601
  closedAt?: string; // ISO 8601
  status: 'open' | 'closed' | 'processing' | 'completed';
  summary?: string;
};

/** 起動時整合性検査で見つかった孤立ファイル（DBレコードが存在しない音声ファイル） */
export type OrphanFile = {
  fileUri: string;
  fileSize: number;
  detectedAt: string;
};

/** 録音ステートマシンの状態 */
export type RecorderState =
  | 'idle'
  | 'starting'
  | 'recording'
  | 'stopping'
  | 'saving'
  | 'saved'
  | 'error';
