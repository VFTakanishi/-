import { createMemoryDatabase } from '../src/db/memoryDatabase';
import type { DreamRecording } from '../src/types/models';

function rec(id: string): DreamRecording {
  return {
    id,
    sessionId: 'session-1',
    createdAt: '2026-08-11T01:00:00.000Z',
    durationMs: 1000,
    fileUri: `file:///${id}.m4a`,
    fileSize: 10,
    status: 'saved',
  };
}

describe('memoryDatabase (repository contract)', () => {
  it('同じIDのレコードは上書き挿入できない(録音ごとに必ず新規INSERT)', async () => {
    const db = createMemoryDatabase();
    await db.recordings.insert(rec('dup'));
    await expect(db.recordings.insert(rec('dup'))).rejects.toThrow();
  });

  it('update は既存フィールドを部分的に書き換えるだけで、新しい行を作らない', async () => {
    const db = createMemoryDatabase();
    await db.recordings.insert(rec('a'));
    await db.recordings.update('a', { status: 'transcribed', transcript: 'hello' });
    const all = await db.recordings.getAll();
    expect(all).toHaveLength(1);
    expect(all[0].status).toBe('transcribed');
    expect(all[0].transcript).toBe('hello');
  });

  it('明示的な deleteAll 以外で全件削除されることはない', async () => {
    const db = createMemoryDatabase();
    await db.recordings.insert(rec('a'));
    await db.recordings.insert(rec('b'));
    expect(await db.recordings.getAll()).toHaveLength(2);
    await db.recordings.deleteAll();
    expect(await db.recordings.getAll()).toHaveLength(0);
  });
});
