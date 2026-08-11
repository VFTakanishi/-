import { ExpoSpeechRecognitionModule } from 'expo-speech-recognition';
import type { TranscriptionOutcome, TranscriptionProvider } from './types';

const LOCALE = 'ja-JP';

/**
 * iOSのオンデバイスSpeech framework（SFSpeechRecognizer）をネイティブモジュール経由で利用する。
 *
 * 重要な制約（README にも記載）:
 * - Expo Go では動作しない。Development Build が必須（expo-speech-recognition はネイティブコードを含む）。
 * - `requiresOnDeviceRecognition: true` を指定しているため、常に完全オフラインで動作する。
 *   ただし日本語のオンデバイス音声認識モデルが端末にダウンロード済みである必要がある
 *   （設定アプリ > 一般 > キーボード > 音声入力 など、iOSのバージョンにより導線が異なる）。
 *   モデル未ダウンロードの端末では `language-not-supported` 等のエラーになり、
 *   その場合はサーバーAPI（設定されていれば）にフォールバックする。
 * - 実機での動作は未検証。Development Build を作成して実機で確認すること。
 */
export class OnDeviceTranscriptionProvider implements TranscriptionProvider {
  id = 'on-device' as const;

  async isAvailable(): Promise<boolean> {
    try {
      const permission = await ExpoSpeechRecognitionModule.getPermissionsAsync();
      if (permission.status === 'denied') return false;
      return ExpoSpeechRecognitionModule.isRecognitionAvailable();
    } catch {
      return false;
    }
  }

  async transcribe(fileUri: string): Promise<TranscriptionOutcome> {
    const permission = await ExpoSpeechRecognitionModule.requestPermissionsAsync();
    if (!permission.granted) {
      return { ok: false, errorMessage: 'speech_permission_denied' };
    }

    return new Promise<TranscriptionOutcome>((resolve) => {
      let settled = false;
      const finish = (outcome: TranscriptionOutcome) => {
        if (settled) return;
        settled = true;
        resultSub.remove();
        errorSub.remove();
        endSub.remove();
        resolve(outcome);
      };

      const resultSub = ExpoSpeechRecognitionModule.addListener('result', (event) => {
        if (event.isFinal) {
          const transcript = event.results[0]?.transcript ?? '';
          finish(transcript ? { ok: true, transcript } : { ok: false, errorMessage: 'empty_transcript' });
        }
      });
      const errorSub = ExpoSpeechRecognitionModule.addListener('error', (event) => {
        finish({ ok: false, errorMessage: event.error });
      });
      const endSub = ExpoSpeechRecognitionModule.addListener('end', () => {
        finish({ ok: false, errorMessage: 'no_final_result' });
      });

      try {
        ExpoSpeechRecognitionModule.start({
          lang: LOCALE,
          interimResults: false,
          continuous: false,
          requiresOnDeviceRecognition: true,
          addsPunctuation: true,
          audioSource: { uri: fileUri },
        });
      } catch (error) {
        finish({ ok: false, errorMessage: error instanceof Error ? error.message : String(error) });
      }
    });
  }
}
