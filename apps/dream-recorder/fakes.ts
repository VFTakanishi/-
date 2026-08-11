import type { AudioRecordingService, InterruptionReason, StopResult } from '../src/audio/types';
import type { HapticsService } from '../src/haptics/types';
import type { FileStorageService, FileInfo, PersistedFileEntry } from '../src/files/types';
import type { Clock } from '../src/core/clock';

/** テスト用の疑似録音サービス。実際の録音は行わず、呼び出しの記録だけを行う。 */
export class FakeAudioRecordingService implements AudioRecordingService {
  granted = true;
  failStart = false;
  failStop = false;
  stopUri = 'file:///tmp/fake-recording.m4a';
  stopDurationMs = 5000;
  private interruptionListeners = new Set<(reason: InterruptionReason) => void>();
  stillRecording = true;

  calls: string[] = [];

  async hasPermission(): Promise<boolean> {
    this.calls.push('hasPermission');
    return this.granted;
  }
  async requestPermission(): Promise<boolean> {
    this.calls.push('requestPermission');
    return this.granted;
  }
  async prepareAndStart(): Promise<void> {
    this.calls.push('prepareAndStart');
    if (this.failStart) throw new Error('start_failed');
  }
  async stop(): Promise<StopResult> {
    this.calls.push('stop');
    if (this.failStop) throw new Error('stop_failed');
    return { uri: this.stopUri, durationMs: this.stopDurationMs };
  }
  async checkStillRecording(): Promise<boolean> {
    return this.stillRecording;
  }
  subscribeInterruption(callback: (reason: InterruptionReason) => void): () => void {
    this.interruptionListeners.add(callback);
    return () => this.interruptionListeners.delete(callback);
  }
  triggerInterruption(reason: InterruptionReason = 'interrupted'): void {
    this.interruptionListeners.forEach((cb) => cb(reason));
  }
}

/** テスト用の疑似振動サービス。呼び出し順序・回数を記録する。 */
export class FakeHapticsService implements HapticsService {
  log: string[] = [];
  async recordingStarted(): Promise<void> {
    this.log.push('recordingStarted');
  }
  async savedSuccessfully(): Promise<void> {
    this.log.push('savedSuccessfully');
  }
  async failure(): Promise<void> {
    this.log.push('failure');
  }
}

/** テスト用の疑似ファイルストレージ。メモリ上のMapで「永続ファイル」を模倣する。 */
export class FakeFileStorageService implements FileStorageService {
  files = new Map<string, number>(); // uri -> size
  failCopy = false;
  nextFileSize = 12345;
  copyCalls: string[] = [];

  async ensureDirectoryExists(): Promise<void> {}

  async copyToPersistent(tempUri: string, destFileName: string): Promise<string> {
    this.copyCalls.push(tempUri);
    if (this.failCopy) throw new Error('copy_failed');
    const uri = `file:///persistent/${destFileName}`;
    this.files.set(uri, this.nextFileSize);
    return uri;
  }

  async getFileInfo(uri: string): Promise<FileInfo> {
    if (!this.files.has(uri)) return { exists: false, size: 0 };
    return { exists: true, size: this.files.get(uri)! };
  }

  async listPersistedFiles(): Promise<PersistedFileEntry[]> {
    return [...this.files.entries()].map(([uri, size]) => ({
      uri,
      fileName: uri.split('/').pop() ?? uri,
      size,
    }));
  }
}

/** テスト用の疑似時計。固定値を進められる。 */
export class FakeClock implements Clock {
  private currentMs: number;
  constructor(startMs = 1_700_000_000_000) {
    this.currentMs = startMs;
  }
  now(): number {
    return this.currentMs;
  }
  nowIso(): string {
    return new Date(this.currentMs).toISOString();
  }
  advance(ms: number): void {
    this.currentMs += ms;
  }
}

export function makeIdGenerator(prefix = 'id') {
  let counter = 0;
  return () => `${prefix}-${++counter}`;
}
