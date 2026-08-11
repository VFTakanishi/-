import { transcribeRecording } from '../src/transcription/transcribeRecording';
import { createMemoryDatabase } from '../src/db/memoryDatabase';
import type { TranscriptionProvider } from '../src/transcription/types';
import type { DreamRecording } from '../src/types/models';

function makeRecording(overrides: Partial<DreamRecording> = {}): DreamRecording {
  return {
    id: 'rec-1',
    sessionId: 'session-1',
    createdAt: '2026-08-11T01:42:00.000Z',
    durationMs: 4000,
    fileUri: 'file:///persistent/rec-1.m4a',
    fileSize: 500,
    status: 'saved',
    ...overrides,
  };
}

const failingProvider: TranscriptionProvider = {
  id: 'on-device',
  isAvailable: async () => true,
  transcribe: async () => ({ ok: false, errorMessage: 'boom' }),
};

const unavailableProvider: TranscriptionProvider = {
  id: 'server',
  isAvailable: async () => false,
  transcribe: async () => ({ ok: true, transcript: 'never used' }),
};

const succeedingProvider: TranscriptionProvider = {
  id: 'server',
  isAvailable: async () => true,
  transcribe: async () => ({ ok: true, transcript: '夢の中で空を飛んでいた' }),
};

describe('transcribeRecording', () => {
  it('11. すべてのプロバイダが失敗しても元音声(fileUri)は変更されない', async () => {
    const db = createMemoryDatabase();
    const recording = makeRecording();
    await db.recordings.insert(recording);

    const result = await transcribeRecording(db, [failingProvider], recording);

    expect(result.fileUri).toBe(recording.fileUri);
    expect(result.fileSize).toBe(recording.fileSize);
    expect(result.status).toBe('saved');
    expect(result.errorMessage).toBe('transcription_unavailable');
    expect(result.transcript).toBeUndefined();
  });

  it('利用不可なプロバイダはスキップして次の候補を試す', async () => {
    const db = createMemoryDatabase();
    const recording = makeRecording({ id: 'rec-2' });
    await db.recordings.insert(recording);

    const result = await transcribeRecording(db, [unavailableProvider, succeedingProvider], recording);

    expect(result.status).toBe('transcribed');
    expect(result.transcript).toBe('夢の中で空を飛んでいた');
  });

  it('13. 文字起こし原文(transcript)とセッションの要約(summary)は別々に保持される', async () => {
    const db = createMemoryDatabase();
    const recording = makeRecording({ id: 'rec-3' });
    await db.recordings.insert(recording);
    await db.sessions.insert({ id: 'session-1', startedAt: recording.createdAt, status: 'open' });

    await transcribeRecording(db, [succeedingProvider], recording);
    await db.sessions.update('session-1', { summary: '夢の要約テキスト' });

    const storedRecording = await db.recordings.getById('rec-3');
    const storedSession = await db.sessions.getById('session-1');

    expect(storedRecording?.transcript).toBe('夢の中で空を飛んでいた');
    expect(storedSession?.summary).toBe('夢の要約テキスト');
    expect(storedRecording?.transcript).not.toBe(storedSession?.summary);
  });
});
