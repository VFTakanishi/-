import { runStartupIntegrityCheck } from '../src/core/integrityCheck';
import { createMemoryDatabase } from '../src/db/memoryDatabase';
import { FakeFileStorageService } from './fakes';
import type { DreamRecording, SleepSession } from '../src/types/models';

function nowIso() {
  return new Date('2026-08-11T09:00:00.000Z').toISOString();
}

describe('runStartupIntegrityCheck', () => {
  it('15. 孤立ファイルを勝手に削除せず、候補として返す', async () => {
    const db = createMemoryDatabase();
    const files = new FakeFileStorageService();
    files.files.set('file:///persistent/orphan.m4a', 999);

    const result = await runStartupIntegrityCheck(db, files, nowIso);

    expect(result.orphanFiles).toHaveLength(1);
    expect(result.orphanFiles[0].fileUri).toBe('file:///persistent/orphan.m4a');
    // 削除されていないこと
    expect(files.files.has('file:///persistent/orphan.m4a')).toBe(true);
  });

  it('14. 日付が変わっていても録音履歴・セッションは自動削除されない', async () => {
    const db = createMemoryDatabase();
    const files = new FakeFileStorageService();

    const yesterday: SleepSession = {
      id: 'session-1',
      startedAt: '2026-08-10T15:00:00.000Z', // 前日夜
      status: 'open',
    };
    await db.sessions.insert(yesterday);

    const recording: DreamRecording = {
      id: 'rec-1',
      sessionId: 'session-1',
      createdAt: '2026-08-10T16:00:00.000Z',
      durationMs: 5000,
      fileUri: 'file:///persistent/rec-1.m4a',
      fileSize: 100,
      status: 'saved',
    };
    files.files.set(recording.fileUri, 100);
    await db.recordings.insert(recording);

    // 「日付が変わった後」の起動を再現するだけで、日付ロジックによる削除は一切行わない
    await runStartupIntegrityCheck(db, files, nowIso);

    const sessions = await db.sessions.getAll();
    const recordings = await db.recordings.getAll();
    expect(sessions).toHaveLength(1);
    expect(sessions[0].status).toBe('open'); // 日付が変わっても自動クローズしない
    expect(recordings).toHaveLength(1);
  });

  it('transcribing のまま終了した処理は saved に戻して再試行可能にする', async () => {
    const db = createMemoryDatabase();
    const files = new FakeFileStorageService();
    const recording: DreamRecording = {
      id: 'rec-2',
      sessionId: 'session-1',
      createdAt: nowIso(),
      durationMs: 1000,
      fileUri: 'file:///persistent/rec-2.m4a',
      fileSize: 10,
      status: 'transcribing',
    };
    files.files.set(recording.fileUri, 10);
    await db.recordings.insert(recording);

    const result = await runStartupIntegrityCheck(db, files, nowIso);

    expect(result.repairedRecordings).toHaveLength(1);
    const updated = await db.recordings.getById('rec-2');
    expect(updated?.status).toBe('saved');
  });

  it('欠損ファイル・0バイトファイルはエラー状態にするがレコードは消さない', async () => {
    const db = createMemoryDatabase();
    const files = new FakeFileStorageService();

    const missing: DreamRecording = {
      id: 'rec-missing',
      sessionId: 'session-1',
      createdAt: nowIso(),
      durationMs: 1000,
      fileUri: 'file:///persistent/missing.m4a',
      fileSize: 10,
      status: 'saved',
    };
    await db.recordings.insert(missing); // filesには登録しない = 欠損

    const empty: DreamRecording = {
      id: 'rec-empty',
      sessionId: 'session-1',
      createdAt: nowIso(),
      durationMs: 1000,
      fileUri: 'file:///persistent/empty.m4a',
      fileSize: 0,
      status: 'saved',
    };
    files.files.set(empty.fileUri, 0);
    await db.recordings.insert(empty);

    const result = await runStartupIntegrityCheck(db, files, nowIso);

    expect(result.missingFileRecordings).toHaveLength(1);
    expect(result.emptyFileRecordings).toHaveLength(1);

    const all = await db.recordings.getAll();
    expect(all).toHaveLength(2); // 削除されていない
    expect(all.every((r) => r.status === 'error')).toBe(true);
  });
});
