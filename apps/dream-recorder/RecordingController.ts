import { buildRecordingFileName } from '../ids/filenames';
import type { AudioRecordingService, InterruptionReason } from '../audio/types';
import type { HapticsService } from '../haptics/types';
import type { FileStorageService } from '../files/types';
import type { DreamDatabase } from '../db/types';
import type { Clock } from './clock';
import type { DreamRecording, RecorderState, SleepSession } from '../types/models';
import { isLockedState, nextRecorderState } from './recorderMachine';

const MIN_RECORDING_MS_BEFORE_STOP = 900;
const AUTO_IDLE_DELAY_MS = 1600;

export type RecordingControllerDeps = {
  db: DreamDatabase;
  audio: AudioRecordingService;
  haptics: HapticsService;
  files: FileStorageService;
  clock: Clock;
  generateId: () => string;
  /** 900ms未満の終了操作を無視するガード。テストで無効化できるように注入可能。 */
  minRecordingMsBeforeStop?: number;
  /** 保存成功後、自動的にidleへ戻すまでの遅延。テストでは0にできる。 */
  autoIdleDelayMs?: number;
};

export type ControllerSnapshot = {
  state: RecorderState;
  errorMessage?: string;
  recordingStartedAtMs?: number;
};

type Listener = (snapshot: ControllerSnapshot) => void;

/**
 * 録音の開始〜保存までを一貫して管理するオーケストレーター。
 *
 * 二重実行防止について:
 * このクラスは React コンポーネントではない、ただのJSクラスであり、
 * `this.state` はReactのstateのようにバッチ処理・遅延されることがない
 * （同期的に即座に書き換わる、ref相当の実体）。
 * starting/stopping/saving に遷移する行は各メソッドの最初の同期処理として実行するため、
 * JavaScriptのランタイムはシングルスレッドかつ「awaitに達するまで割り込まれない」という
 * 実行モデルにより、ボタン連打（同期的な複数回呼び出し）で同じ処理が二重実行されることはない。
 */
export class RecordingController {
  private state: RecorderState = 'idle';
  private errorMessage: string | undefined;
  private recordingStartedAtMs: number | undefined;
  private currentRecordingId: string | undefined;
  private currentSessionId: string | undefined;
  private interruptedDuringRecording = false;
  private listeners = new Set<Listener>();
  private unsubscribeInterruption: (() => void) | null = null;
  private idleTimer: ReturnType<typeof setTimeout> | null = null;

  private readonly minRecordingMsBeforeStop: number;
  private readonly autoIdleDelayMs: number;

  constructor(private readonly deps: RecordingControllerDeps) {
    this.minRecordingMsBeforeStop = deps.minRecordingMsBeforeStop ?? MIN_RECORDING_MS_BEFORE_STOP;
    this.autoIdleDelayMs = deps.autoIdleDelayMs ?? AUTO_IDLE_DELAY_MS;
  }

  getSnapshot(): ControllerSnapshot {
    return {
      state: this.state,
      errorMessage: this.errorMessage,
      recordingStartedAtMs: this.recordingStartedAtMs,
    };
  }

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /** アプリ起動時に一度だけ呼ぶ。中断監視の購読を開始する。 */
  attachInterruptionWatcher(): void {
    if (this.unsubscribeInterruption) return;
    this.unsubscribeInterruption = this.deps.audio.subscribeInterruption((reason) =>
      this.handleExternalInterruption(reason)
    );
  }

  detachInterruptionWatcher(): void {
    this.unsubscribeInterruption?.();
    this.unsubscribeInterruption = null;
  }

  /**
   * 巨大な赤いボタンが押されたときに呼ぶ唯一の入口。
   * 現在の状態に応じて開始・終了のどちらかに振り分ける。
   * locked状態（starting/stopping/saving）では何もしない。
   */
  async pressButton(): Promise<void> {
    if (isLockedState(this.state)) {
      return;
    }
    if (this.state === 'idle' || this.state === 'saved' || this.state === 'error') {
      await this.startRecording();
      return;
    }
    if (this.state === 'recording') {
      const elapsed = this.deps.clock.now() - (this.recordingStartedAtMs ?? 0);
      if (elapsed < this.minRecordingMsBeforeStop) {
        return; // 開始直後の誤タップは無視する
      }
      await this.stopAndSave();
    }
  }

  /**
   * AppStateの復帰時など、実際に録音が続いているか怪しいタイミングで
   * 外部（React層）から中断の可能性を通知するための公開メソッド。
   */
  notifyPossibleInterruption(): void {
    this.handleExternalInterruption('interrupted');
  }

  private handleExternalInterruption(reason: InterruptionReason): void {
    if (reason === 'interrupted' && this.state === 'recording') {
      this.interruptedDuringRecording = true;
      // 可能な限りそこまでの音声を保存する。ユーザー操作を待たずベストエフォートで実行する。
      void this.stopAndSave();
    }
  }

  private setState(state: RecorderState, errorMessage?: string): void {
    this.state = state;
    this.errorMessage = errorMessage;
    this.listeners.forEach((listener) => listener(this.getSnapshot()));
  }

  private async startRecording(): Promise<void> {
    // 同期的に最初の行でロック状態へ遷移させる（二重実行防止の要）。
    const startingState = nextRecorderState(this.state, { type: 'START_REQUESTED' });
    this.setState(startingState);
    if (startingState !== 'starting') return; // 不正な遷移だった場合は何もしない

    try {
      let granted = await this.deps.audio.hasPermission();
      if (!granted) {
        granted = await this.deps.audio.requestPermission();
      }
      if (!granted) {
        this.setState(nextRecorderState(this.state, { type: 'START_FAILED' }), 'permission_denied');
        await this.deps.haptics.failure();
        return;
      }

      await this.deps.audio.prepareAndStart();

      this.recordingStartedAtMs = this.deps.clock.now();
      this.currentRecordingId = this.deps.generateId();
      this.interruptedDuringRecording = false;

      // 録音開始成功が確認できた「後」にだけ振動を鳴らす。
      await this.deps.haptics.recordingStarted();

      this.setState(nextRecorderState(this.state, { type: 'START_SUCCEEDED' }));
    } catch (error) {
      this.setState(
        nextRecorderState(this.state, { type: 'START_FAILED' }),
        error instanceof Error ? error.message : String(error)
      );
      await this.deps.haptics.failure();
    }
  }

  private async stopAndSave(): Promise<void> {
    if (this.state !== 'recording') return;
    const stoppingState = nextRecorderState(this.state, { type: 'STOP_REQUESTED' });
    this.setState(stoppingState);
    if (stoppingState !== 'stopping') return;

    let stopResult;
    try {
      stopResult = await this.deps.audio.stop();
    } catch (error) {
      this.setState('error', error instanceof Error ? error.message : String(error));
      await this.deps.haptics.failure();
      return;
    }

    const savingState = nextRecorderState(this.state, { type: 'STOPPED' });
    this.setState(savingState);
    if (savingState !== 'saving') return;

    const recordingId = this.currentRecordingId ?? this.deps.generateId();
    const wasInterrupted = this.interruptedDuringRecording;

    try {
      const createdAtIso = this.deps.clock.nowIso();
      const fileName = buildRecordingFileName(createdAtIso, recordingId);

      // 1. 一時URIから永続保存領域へコピー
      const persistedUri = await this.deps.files.copyToPersistent(stopResult.uri, fileName);

      // 2. 保存先ファイルの存在確認・サイズ確認（0バイトは失敗扱い）
      const info = await this.deps.files.getFileInfo(persistedUri);
      if (!info.exists) {
        throw new Error('saved_file_missing');
      }
      if (info.size < 1) {
        throw new Error('saved_file_empty');
      }

      // 3. 開いているセッションが無ければ作成
      const sessionId = await this.ensureOpenSession(createdAtIso);
      this.currentSessionId = sessionId;

      // 4. SQLiteへメタデータをINSERT
      const record: DreamRecording = {
        id: recordingId,
        sessionId,
        createdAt: createdAtIso,
        durationMs: stopResult.durationMs,
        fileUri: persistedUri,
        fileSize: info.size,
        status: wasInterrupted ? 'interrupted' : 'saved',
      };
      await this.deps.db.recordings.insert(record);

      // 5. INSERTしたレコードを再取得して確認できて初めて保存成功
      const verified = await this.deps.db.recordings.getById(recordingId);
      if (!verified) {
        throw new Error('insert_verification_failed');
      }

      // ここまで来て初めて保存成功の振動（一定間隔で2回）
      await this.deps.haptics.savedSuccessfully();

      this.setState(nextRecorderState(this.state, { type: 'SAVE_SUCCEEDED' }));
      this.scheduleReturnToIdle();
    } catch (error) {
      // 一時ファイル(stopResult.uri)は削除しない。保存済みデータも削除しない。
      this.setState('error', error instanceof Error ? error.message : String(error));
      await this.deps.haptics.failure();
    } finally {
      this.interruptedDuringRecording = false;
      this.recordingStartedAtMs = undefined;
      this.currentRecordingId = undefined;
    }
  }

  private async ensureOpenSession(nowIso: string): Promise<string> {
    const open = await this.deps.db.sessions.getOpenSession();
    if (open) return open.id;
    const session: SleepSession = {
      id: this.deps.generateId(),
      startedAt: nowIso,
      status: 'open',
    };
    await this.deps.db.sessions.insert(session);
    return session.id;
  }

  private scheduleReturnToIdle(): void {
    if (this.idleTimer) clearTimeout(this.idleTimer);
    this.idleTimer = setTimeout(() => {
      this.idleTimer = null;
      if (this.state === 'saved') {
        this.setState(nextRecorderState(this.state, { type: 'RESET' }));
      }
    }, this.autoIdleDelayMs);
  }

  /** テスト・エラー画面からの明示的な「もう一度」操作用。 */
  resetToIdleFromError(): void {
    if (this.state === 'error') {
      this.setState(nextRecorderState(this.state, { type: 'RESET' }));
    }
  }

  dispose(): void {
    if (this.idleTimer) clearTimeout(this.idleTimer);
    this.detachInterruptionWatcher();
    this.listeners.clear();
  }
}
