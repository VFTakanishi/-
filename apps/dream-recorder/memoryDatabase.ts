import type { DreamRecording, SleepSession } from '../types/models';
import type { DreamDatabase, DreamRecordingsRepo, SleepSessionsRepo } from './types';

/**
 * インメモリ実装。
 *
 * ユニットテストで「再起動相当の再初期化後も復元できる」ことを検証するため、
 * バックする Map を外部から渡せるようにしてある。
 * テストでは同じ Map を新しい MemoryDreamDatabase インスタンスへ渡し直すことで、
 * 「アプリを終了して再度開いたら、同じ永続ストアから読み直す」状況を再現する。
 * （実アプリでは、この永続ストアの役割を SQLite のファイルが担う。）
 */
export type MemoryStore = {
  recordings: Map<string, DreamRecording>;
  sessions: Map<string, SleepSession>;
};

export function createMemoryStore(): MemoryStore {
  return { recordings: new Map(), sessions: new Map() };
}

class MemoryRecordingsRepo implements DreamRecordingsRepo {
  constructor(private store: MemoryStore) {}

  async insert(record: DreamRecording): Promise<void> {
    if (this.store.recordings.has(record.id)) {
      throw new Error(`duplicate recording id: ${record.id}`);
    }
    this.store.recordings.set(record.id, { ...record });
  }

  async getById(id: string): Promise<DreamRecording | null> {
    const found = this.store.recordings.get(id);
    return found ? { ...found } : null;
  }

  async update(id: string, patch: Partial<DreamRecording>): Promise<void> {
    const existing = this.store.recordings.get(id);
    if (!existing) throw new Error(`recording not found: ${id}`);
    this.store.recordings.set(id, { ...existing, ...patch, id: existing.id });
  }

  async getAll(): Promise<DreamRecording[]> {
    return [...this.store.recordings.values()]
      .map((r) => ({ ...r }))
      .sort((a, b) => a.createdAt.localeCompare(b.createdAt));
  }

  async getBySession(sessionId: string): Promise<DreamRecording[]> {
    return (await this.getAll()).filter((r) => r.sessionId === sessionId);
  }

  async deleteAll(): Promise<void> {
    this.store.recordings.clear();
  }
}

class MemorySessionsRepo implements SleepSessionsRepo {
  constructor(private store: MemoryStore) {}

  async insert(session: SleepSession): Promise<void> {
    if (this.store.sessions.has(session.id)) {
      throw new Error(`duplicate session id: ${session.id}`);
    }
    this.store.sessions.set(session.id, { ...session });
  }

  async getById(id: string): Promise<SleepSession | null> {
    const found = this.store.sessions.get(id);
    return found ? { ...found } : null;
  }

  async update(id: string, patch: Partial<SleepSession>): Promise<void> {
    const existing = this.store.sessions.get(id);
    if (!existing) throw new Error(`session not found: ${id}`);
    this.store.sessions.set(id, { ...existing, ...patch, id: existing.id });
  }

  async getOpenSession(): Promise<SleepSession | null> {
    for (const s of this.store.sessions.values()) {
      if (s.status === 'open') return { ...s };
    }
    return null;
  }

  async getAll(): Promise<SleepSession[]> {
    return [...this.store.sessions.values()].sort((a, b) =>
      a.startedAt.localeCompare(b.startedAt)
    );
  }

  async deleteAll(): Promise<void> {
    this.store.sessions.clear();
  }
}

export function createMemoryDatabase(store: MemoryStore = createMemoryStore()): DreamDatabase {
  return {
    recordings: new MemoryRecordingsRepo(store),
    sessions: new MemorySessionsRepo(store),
  };
}
