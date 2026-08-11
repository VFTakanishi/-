import { File, UploadType } from 'expo-file-system';
import { getTranscriptionSettings } from './settingsStore';
import type { TranscriptionOutcome, TranscriptionProvider } from './types';

/**
 * ユーザーが朝用画面の設定で任意に登録したサーバーAPIへ音声ファイルを送って文字起こしする。
 * - 秘密鍵はアプリへ埋め込まない。すべてSecureStoreに保存されたユーザー入力値。
 * - 未設定ならこのプロバイダは isAvailable() が false を返し、夜間保存の成功条件には含まれない。
 * - サーバーは `{ transcript: string }` を含むJSONを返す想定（multipart/form-data, フィールド名 "audio"）。
 */
export class ServerTranscriptionProvider implements TranscriptionProvider {
  id = 'server' as const;

  async isAvailable(): Promise<boolean> {
    const settings = await getTranscriptionSettings();
    return Boolean(settings.endpointUrl);
  }

  async transcribe(fileUri: string): Promise<TranscriptionOutcome> {
    const settings = await getTranscriptionSettings();
    if (!settings.endpointUrl) {
      return { ok: false, errorMessage: 'server_not_configured' };
    }

    try {
      const file = new File(fileUri);
      const result = await file.upload(settings.endpointUrl, {
        httpMethod: 'POST',
        uploadType: UploadType.MULTIPART,
        fieldName: 'audio',
        mimeType: 'audio/m4a',
        headers: settings.apiKey ? { Authorization: `Bearer ${settings.apiKey}` } : undefined,
      });

      if (result.status < 200 || result.status >= 300) {
        return { ok: false, errorMessage: `server_error_${result.status}` };
      }

      const parsed = JSON.parse(result.body) as { transcript?: unknown };
      if (typeof parsed.transcript !== 'string' || parsed.transcript.length === 0) {
        return { ok: false, errorMessage: 'invalid_server_response' };
      }
      return { ok: true, transcript: parsed.transcript };
    } catch (error) {
      return { ok: false, errorMessage: error instanceof Error ? error.message : String(error) };
    }
  }
}
