import type { DreamRecording, SleepSession } from '../types/models';

/**
 * 永続化リポジトリのインターフェース。
 * 実装は expo-sqlite（実機・実アプリ用）と、
 * インメモリ実装（ユニットテスト用、SQLiteの実I/Oは行わないがAPI契約は同一）の2種類を用意する。
 * コア処理（RecordingController・整合性検査）はこのインターフェースにのみ依存し、
 * expo-sqlite を直接importしないため、Jest上でも実機相当のロジック検証ができる。
 */
export interface DreamRecordingsRepo {
  /** 新規レコードをINSERTする。同じidの上書きは禁止（既に存在すればエラー）。 */
  insert(record: DreamRecording): Promise<void>;
  /** idで1件取得する。INSERT直後の再取得確認に使う。 */
  getById(id: string): Promise<DreamRecording | null>;
  /** 部分更新（transcript, status, errorMessageなど）。新規行は作らない。 */
  update(id: string, patch: Partial<DreamRecording>): Promise<void>;
  /** 全件取得（作成日時の昇順）。 */
  getAll(): Promise<DreamRecording[]>;
  /** セッションIDに紐づく録音を作成日時の昇順で取得する。 */
  getBySession(sessionId: string): Promise<DreamRecording[]>;
  /** 明示リセット専用。確認ダイアログを通過した呼び出し元のみが使うこと。 */
  deleteAll(): Promise<void>;
}

export interface SleepSessionsRepo {
  insert(session: SleepSession): Promise<void>;
  getById(id: string): Promise<SleepSession | null>;
  update(id: string, patch: Partial<SleepSession>): Promise<void>;
  /** status === 'open' のセッションを1件返す。無ければnull。 */
  getOpenSession(): Promise<SleepSession | null>;
  getAll(): Promise<SleepSession[]>;
  deleteAll(): Promise<void>;
}

export interface DreamDatabase {
  recordings: DreamRecordingsRepo;
  sessions: SleepSessionsRepo;
}
