import * as SQLite from 'expo-sqlite';
import type { DreamRecording, SleepSession } from '../types/models';
import type { DreamDatabase, DreamRecordingsRepo, SleepSessionsRepo } from './types';

const DB_NAME = 'dream_recorder.db';

let dbPromise: Promise<SQLite.SQLiteDatabase> | null = null;

/**
 * SQLiteデータベースを開き、必要なテーブルを作成する。
 * Reactのstateはここから復元される「ビュー」にすぎず、正本は常にこのファイル。
 */
async function openDb(): Promise<SQLite.SQLiteDatabase> {
  if (!dbPromise) {
    dbPromise = (async () => {
      const db = await SQLite.openDatabaseAsync(DB_NAME);
      await db.execAsync('PRAGMA journal_mode = WAL;');
      await db.execAsync(`
        CREATE TABLE IF NOT EXISTS sleep_sessions (
          id TEXT PRIMARY KEY NOT NULL,
          startedAt TEXT NOT NULL,
          closedAt TEXT,
          status TEXT NOT NULL,
          summary TEXT
        );
        CREATE TABLE IF NOT EXISTS dream_recordings (
          id TEXT PRIMARY KEY NOT NULL,
          sessionId TEXT NOT NULL,
          createdAt TEXT NOT NULL,
          durationMs INTEGER NOT NULL,
          fileUri TEXT NOT NULL,
          fileSize INTEGER NOT NULL,
          status TEXT NOT NULL,
          transcript TEXT,
          errorMessage TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_recordings_session ON dream_recordings(sessionId);
        CREATE INDEX IF NOT EXISTS idx_recordings_createdAt ON dream_recordings(createdAt);
      `);
      return db;
    })();
  }
  return dbPromise;
}

function rowToRecording(row: any): DreamRecording {
  return {
    id: row.id,
    sessionId: row.sessionId,
    createdAt: row.createdAt,
    durationMs: row.durationMs,
    fileUri: row.fileUri,
    fileSize: row.fileSize,
    status: row.status,
    transcript: row.transcript ?? undefined,
    errorMessage: row.errorMessage ?? undefined,
  };
}

function rowToSession(row: any): SleepSession {
  return {
    id: row.id,
    startedAt: row.startedAt,
    closedAt: row.closedAt ?? undefined,
    status: row.status,
    summary: row.summary ?? undefined,
  };
}

class SqliteRecordingsRepo implements DreamRecordingsRepo {
  async insert(record: DreamRecording): Promise<void> {
    const db = await openDb();
    const existing = await db.getFirstAsync('SELECT id FROM dream_recordings WHERE id = ?', [
      record.id,
    ]);
    if (existing) {
      throw new Error(`duplicate recording id: ${record.id}`);
    }
    await db.runAsync(
      `INSERT INTO dream_recordings
        (id, sessionId, createdAt, durationMs, fileUri, fileSize, status, transcript, errorMessage)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [
        record.id,
        record.sessionId,
        record.createdAt,
        record.durationMs,
        record.fileUri,
        record.fileSize,
        record.status,
        record.transcript ?? null,
        record.errorMessage ?? null,
      ]
    );
  }

  async getById(id: string): Promise<DreamRecording | null> {
    const db = await openDb();
    const row = await db.getFirstAsync('SELECT * FROM dream_recordings WHERE id = ?', [id]);
    return row ? rowToRecording(row) : null;
  }

  async update(id: string, patch: Partial<DreamRecording>): Promise<void> {
    const db = await openDb();
    const current = await this.getById(id);
    if (!current) throw new Error(`recording not found: ${id}`);
    const merged = { ...current, ...patch, id };
    await db.runAsync(
      `UPDATE dream_recordings SET
        sessionId = ?, createdAt = ?, durationMs = ?, fileUri = ?, fileSize = ?,
        status = ?, transcript = ?, errorMessage = ?
       WHERE id = ?`,
      [
        merged.sessionId,
        merged.createdAt,
        merged.durationMs,
        merged.fileUri,
        merged.fileSize,
        merged.status,
        merged.transcript ?? null,
        merged.errorMessage ?? null,
        id,
      ]
    );
  }

  async getAll(): Promise<DreamRecording[]> {
    const db = await openDb();
    const rows = await db.getAllAsync('SELECT * FROM dream_recordings ORDER BY createdAt ASC');
    return rows.map(rowToRecording);
  }

  async getBySession(sessionId: string): Promise<DreamRecording[]> {
    const db = await openDb();
    const rows = await db.getAllAsync(
      'SELECT * FROM dream_recordings WHERE sessionId = ? ORDER BY createdAt ASC',
      [sessionId]
    );
    return rows.map(rowToRecording);
  }

  async deleteAll(): Promise<void> {
    const db = await openDb();
    await db.runAsync('DELETE FROM dream_recordings');
  }
}

class SqliteSessionsRepo implements SleepSessionsRepo {
  async insert(session: SleepSession): Promise<void> {
    const db = await openDb();
    const existing = await db.getFirstAsync('SELECT id FROM sleep_sessions WHERE id = ?', [
      session.id,
    ]);
    if (existing) {
      throw new Error(`duplicate session id: ${session.id}`);
    }
    await db.runAsync(
      `INSERT INTO sleep_sessions (id, startedAt, closedAt, status, summary) VALUES (?, ?, ?, ?, ?)`,
      [session.id, session.startedAt, session.closedAt ?? null, session.status, session.summary ?? null]
    );
  }

  async getById(id: string): Promise<SleepSession | null> {
    const db = await openDb();
    const row = await db.getFirstAsync('SELECT * FROM sleep_sessions WHERE id = ?', [id]);
    return row ? rowToSession(row) : null;
  }

  async update(id: string, patch: Partial<SleepSession>): Promise<void> {
    const db = await openDb();
    const current = await this.getById(id);
    if (!current) throw new Error(`session not found: ${id}`);
    const merged = { ...current, ...patch, id };
    await db.runAsync(
      `UPDATE sleep_sessions SET startedAt = ?, closedAt = ?, status = ?, summary = ? WHERE id = ?`,
      [merged.startedAt, merged.closedAt ?? null, merged.status, merged.summary ?? null, id]
    );
  }

  async getOpenSession(): Promise<SleepSession | null> {
    const db = await openDb();
    const row = await db.getFirstAsync(
      `SELECT * FROM sleep_sessions WHERE status = 'open' ORDER BY startedAt DESC LIMIT 1`
    );
    return row ? rowToSession(row) : null;
  }

  async getAll(): Promise<SleepSession[]> {
    const db = await openDb();
    const rows = await db.getAllAsync('SELECT * FROM sleep_sessions ORDER BY startedAt ASC');
    return rows.map(rowToSession);
  }

  async deleteAll(): Promise<void> {
    const db = await openDb();
    await db.runAsync('DELETE FROM sleep_sessions');
  }
}

export function createSqliteDatabase(): DreamDatabase {
  return {
    recordings: new SqliteRecordingsRepo(),
    sessions: new SqliteSessionsRepo(),
  };
}
