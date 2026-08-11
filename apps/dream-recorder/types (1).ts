/**
 * 録音サービスのインターフェース。
 * コア処理（RecordingController）はこのインターフェースにのみ依存し、
 * expo-audio を直接importしない。これによりユニットテストでは
 * フェイク実装（__tests__/fakes.ts）に差し替えて振る舞いを検証できる。
 */
export type StopResult = {
  /** 録音直後の一時ファイルURI（まだ永続領域ではない） */
  uri: string;
  durationMs: number;
};

export type InterruptionReason = 'interrupted' | 'resumed';

export interface AudioRecordingService {
  /** 現在の権限状態を確認する（ユーザーへのダイアログは出さない） */
  hasPermission(): Promise<boolean>;
  /** 権限ダイアログを表示して結果を返す */
  requestPermission(): Promise<boolean>;
  /** 録音準備・開始。成功しなければ例外を投げる。 */
  prepareAndStart(): Promise<void>;
  /** 録音停止。一時URIと長さを返す。 */
  stop(): Promise<StopResult>;
  /**
   * OS側が実際にまだ録音中かを再確認する（バックグラウンド復帰時のチェック用）。
   * 録音中でなければ、中断が起きていた可能性が高い。
   */
  checkStillRecording(): Promise<boolean>;
  /**
   * OSによる録音の中断（電話着信、他アプリの音声割り込みなど）を監視する。
   * 呼び出すと購読解除関数を返す。
   */
  subscribeInterruption(callback: (reason: InterruptionReason) => void): () => void;
}
