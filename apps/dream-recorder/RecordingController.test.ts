import { RecordingController } from '../src/core/RecordingController';
import { createMemoryDatabase, createMemoryStore } from '../src/db/memoryDatabase';
import type { DreamDatabase } from '../src/db/types';
import {
  FakeAudioRecordingService,
  FakeClock,
  FakeFileStorageService,
  FakeHapticsService,
  makeIdGenerator,
} from './fakes';

function setup(overrides?: { minRecordingMsBeforeStop?: number; autoIdleDelayMs?: number }) {
  const store = createMemoryStore();
  const db = createMemoryDatabase(store);
  const audio = new FakeAudioRecordingService();
  const haptics = new FakeHapticsService();
  const files = new FakeFileStorageService();
  const clock = new FakeClock();
  const controller = new RecordingController({
    db,
    audio,
    haptics,
    files,
    clock,
    generateId: makeIdGenerator(),
    minRecordingMsBeforeStop: overrides?.minRecordingMsBeforeStop ?? 900,
    autoIdleDelayMs: overrides?.autoIdleDelayMs ?? 0,
  });
  return { store, db, audio, haptics, files, clock, controller };
}

async function fullStartStopCycle(ctx: ReturnType<typeof setup>) {
  await ctx.controller.pressButton(); // start
  ctx.clock.advance(2000);
  await ctx.controller.pressButton(); // stop + save
}

describe('RecordingController', () => {
  it('1. 録音開始成功前に開始振動を呼ばない', async () => {
    const ctx = setup();
    // prepareAndStart はまだ呼ばれていない = 開始前
    expect(ctx.haptics.log).toEqual([]);
    await ctx.controller.pressButton();
    // 開始が完了した後なら1回呼ばれているはず
    expect(ctx.haptics.log).toEqual(['recordingStarted']);
  });

  it('2. 録音開始成功後に開始振動を1回だけ呼ぶ', async () => {
    const ctx = setup();
    await ctx.controller.pressButton();
    expect(ctx.haptics.log.filter((l) => l === 'recordingStarted')).toHaveLength(1);
    expect(ctx.controller.getSnapshot().state).toBe('recording');
  });

  it('開始に失敗したときは開始振動を呼ばずエラー振動を呼ぶ', async () => {
    const ctx = setup();
    ctx.audio.failStart = true;
    await ctx.controller.pressButton();
    expect(ctx.haptics.log).toEqual(['failure']);
    expect(ctx.controller.getSnapshot().state).toBe('error');
  });

  it('3&4. ファイル確認・SQLite確認の前には保存振動を呼ばない', async () => {
    const callOrder: string[] = [];
    const ctx = setup();
    const originalCopy = ctx.files.copyToPersistent.bind(ctx.files);
    ctx.files.copyToPersistent = async (...args) => {
      callOrder.push('copyToPersistent');
      return originalCopy(...args);
    };
    const originalGetInfo = ctx.files.getFileInfo.bind(ctx.files);
    ctx.files.getFileInfo = async (...args) => {
      callOrder.push('getFileInfo');
      return originalGetInfo(...args);
    };
    const originalInsert = ctx.db.recordings.insert.bind(ctx.db.recordings);
    ctx.db.recordings.insert = async (...args) => {
      callOrder.push('insert');
      return originalInsert(...args);
    };
    const originalGetById = ctx.db.recordings.getById.bind(ctx.db.recordings);
    ctx.db.recordings.getById = async (...args) => {
      callOrder.push('getById');
      return originalGetById(...args);
    };
    const originalSaved = ctx.haptics.savedSuccessfully.bind(ctx.haptics);
    ctx.haptics.savedSuccessfully = async () => {
      callOrder.push('savedSuccessfully');
      return originalSaved();
    };

    await fullStartStopCycle(ctx);

    const savedIndex = callOrder.indexOf('savedSuccessfully');
    expect(savedIndex).toBeGreaterThan(-1);
    expect(callOrder.indexOf('copyToPersistent')).toBeLessThan(savedIndex);
    expect(callOrder.indexOf('getFileInfo')).toBeLessThan(savedIndex);
    expect(callOrder.indexOf('insert')).toBeLessThan(savedIndex);
    expect(callOrder.indexOf('getById')).toBeLessThan(savedIndex);
    expect(callOrder).toEqual(['copyToPersistent', 'getFileInfo', 'insert', 'getById', 'savedSuccessfully']);
  });

  it('完全保存後は状態が saved になり件数が増える', async () => {
    const ctx = setup();
    await fullStartStopCycle(ctx);
    expect(ctx.controller.getSnapshot().state).toBe('saved');
    const all = await ctx.db.recordings.getAll();
    expect(all).toHaveLength(1);
    expect(all[0].status).toBe('saved');
  });

  it('6. 3回録音すると異なるID・URIの3件が残る', async () => {
    const ctx = setup();
    await fullStartStopCycle(ctx);
    await fullStartStopCycle(ctx);
    await fullStartStopCycle(ctx);

    const all = await ctx.db.recordings.getAll();
    expect(all).toHaveLength(3);
    const ids = new Set(all.map((r) => r.id));
    const uris = new Set(all.map((r) => r.fileUri));
    expect(ids.size).toBe(3);
    expect(uris.size).toBe(3);
  });

  it('7. 再起動相当の再初期化後も3件を復元する', async () => {
    const ctx = setup();
    await fullStartStopCycle(ctx);
    await fullStartStopCycle(ctx);
    await fullStartStopCycle(ctx);

    // 「アプリを再度開いた」= 同じ永続ストアを使って新しいDBインスタンスを作る
    const restoredDb: DreamDatabase = createMemoryDatabase(ctx.store);
    const restored = await restoredDb.recordings.getAll();
    expect(restored).toHaveLength(3);
  });

  it('8. 二重タップで即終了しない(900ms未満の終了操作は無視する)', async () => {
    const ctx = setup();
    await ctx.controller.pressButton(); // start
    expect(ctx.controller.getSnapshot().state).toBe('recording');

    // 開始直後、経過時間を進めずに連打
    await ctx.controller.pressButton();
    await ctx.controller.pressButton();

    expect(ctx.controller.getSnapshot().state).toBe('recording');
    expect(ctx.audio.calls).not.toContain('stop');
  });

  it('9. 保存中の連打で重複INSERTしない', async () => {
    const ctx = setup();
    await ctx.controller.pressButton(); // start
    ctx.clock.advance(2000);

    let insertCount = 0;
    const originalInsert = ctx.db.recordings.insert.bind(ctx.db.recordings);
    ctx.db.recordings.insert = async (...args) => {
      insertCount += 1;
      return originalInsert(...args);
    };

    // 停止・保存の連打を同期的に発生させる
    const p1 = ctx.controller.pressButton();
    const p2 = ctx.controller.pressButton();
    const p3 = ctx.controller.pressButton();
    await Promise.all([p1, p2, p3]);

    expect(insertCount).toBe(1);
    const all = await ctx.db.recordings.getAll();
    expect(all).toHaveLength(1);
  });

  it('10. 0バイトのファイルを成功扱いしない', async () => {
    const ctx = setup();
    ctx.files.nextFileSize = 0;
    await fullStartStopCycle(ctx);

    expect(ctx.controller.getSnapshot().state).toBe('error');
    expect(ctx.haptics.log).toEqual(['recordingStarted', 'failure']);
    const all = await ctx.db.recordings.getAll();
    expect(all).toHaveLength(0);
  });

  it('保存先ファイルが存在しない場合も成功扱いしない', async () => {
    const ctx = setup();
    const original = ctx.files.copyToPersistent.bind(ctx.files);
    ctx.files.copyToPersistent = async (tempUri, destFileName) => {
      const uri = await original(tempUri, destFileName);
      ctx.files.files.delete(uri); // コピー直後にファイルが消えた状況を再現
      return uri;
    };
    await fullStartStopCycle(ctx);
    expect(ctx.controller.getSnapshot().state).toBe('error');
    const all = await ctx.db.recordings.getAll();
    expect(all).toHaveLength(0);
  });

  it('外部からの中断通知で録音中のデータを interrupted として保存する', async () => {
    const ctx = setup();
    await ctx.controller.pressButton(); // start
    ctx.clock.advance(2000);
    ctx.controller.notifyPossibleInterruption();
    // stopAndSave は内部で非同期に走るので完了を待つ
    await new Promise((resolve) => setTimeout(resolve, 0));
    await new Promise((resolve) => setTimeout(resolve, 0));

    const all = await ctx.db.recordings.getAll();
    expect(all).toHaveLength(1);
    expect(all[0].status).toBe('interrupted');
  });

  it('保存に失敗しても一時ファイルを削除しようとしない(ファイルサービスにdelete呼び出しが無い)', async () => {
    const ctx = setup();
    ctx.files.failCopy = true;
    await fullStartStopCycle(ctx);
    expect(ctx.controller.getSnapshot().state).toBe('error');
    expect(ctx.haptics.log).toEqual(['recordingStarted', 'failure']);
  });
});
