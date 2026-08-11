import {
  AudioModule,
  RecordingPresets,
  getRecordingPermissionsAsync,
  requestRecordingPermissionsAsync,
  setAudioModeAsync,
} from 'expo-audio';
import type { AudioRecordingService, InterruptionReason, StopResult } from './types';

/**
 * expo-audio を使った実機用の録音サービス実装。
 *
 * 注意（README にも記載）:
 * expo-audio の JS API には「中断の開始/終了」を直接表す専用イベントが無いため、
 * ここでは `recordingStatusUpdate` イベントの `hasError` / `mediaServicesDidReset` を
 * 中断の代理シグナルとして扱う。加えて呼び出し側（useDreamRecorder）で
 * React Native の AppState 変化も監視し、両方を合わせて「中断の可能性」として扱う。
 * 実機（電話着信・他アプリ切替）での最終確認が必要。
 */
export class ExpoAudioRecordingService implements AudioRecordingService {
  private recorder: InstanceType<typeof AudioModule.AudioRecorder> | null = null;
  private listeners = new Set<(reason: InterruptionReason) => void>();
  private removeStatusListener: (() => void) | null = null;

  async hasPermission(): Promise<boolean> {
    const result = await getRecordingPermissionsAsync();
    return result.granted;
  }

  async requestPermission(): Promise<boolean> {
    const result = await requestRecordingPermissionsAsync();
    return result.granted;
  }

  async prepareAndStart(): Promise<void> {
    await setAudioModeAsync({
      allowsRecording: true,
      playsInSilentMode: true,
      interruptionMode: 'doNotMix',
      shouldPlayInBackground: true,
      allowsBackgroundRecording: true,
      shouldRouteThroughEarpiece: false,
    });

    // AudioModule はネイティブ側の実行時プロパティとして AudioRecorder を公開しており、
    // 静的な名前空間解析では検出できないための既知の誤検知。
    // eslint-disable-next-line import/namespace
    const recorder = new AudioModule.AudioRecorder({
      ...RecordingPresets.HIGH_QUALITY,
      directory: 'cache',
      isMeteringEnabled: false,
    });

    const subscription = recorder.addListener('recordingStatusUpdate', (status) => {
      if (status.hasError || status.mediaServicesDidReset) {
        this.listeners.forEach((cb) => cb('interrupted'));
      }
    });
    this.removeStatusListener = () => subscription.remove();

    await recorder.prepareToRecordAsync();
    recorder.record();

    const status = recorder.getStatus();
    if (!status.isRecording) {
      this.removeStatusListener?.();
      this.removeStatusListener = null;
      throw new Error('recording_did_not_start');
    }

    this.recorder = recorder;
  }

  async stop(): Promise<StopResult> {
    const recorder = this.recorder;
    if (!recorder) {
      throw new Error('no_active_recorder');
    }
    const statusBeforeStop = recorder.getStatus();
    await recorder.stop();
    const uri = recorder.uri;
    this.removeStatusListener?.();
    this.removeStatusListener = null;
    this.recorder = null;

    if (!uri) {
      throw new Error('missing_uri_after_stop');
    }
    return { uri, durationMs: statusBeforeStop.durationMillis };
  }

  async checkStillRecording(): Promise<boolean> {
    if (!this.recorder) return false;
    try {
      return this.recorder.getStatus().isRecording;
    } catch {
      return false;
    }
  }

  subscribeInterruption(callback: (reason: InterruptionReason) => void): () => void {
    this.listeners.add(callback);
    return () => this.listeners.delete(callback);
  }
}
