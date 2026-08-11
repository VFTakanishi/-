import * as SecureStore from 'expo-secure-store';

const ENDPOINT_KEY = 'dream_recorder_transcription_endpoint';
const API_KEY_KEY = 'dream_recorder_transcription_api_key';

export type TranscriptionSettings = {
  endpointUrl?: string;
  apiKey?: string;
};

/**
 * ユーザーが後から設定できる文字起こしサーバーAPIの接続情報。
 * 秘密鍵をアプリへ埋め込まない：この値は端末のセキュアストレージにのみ保存され、
 * 未設定なら server プロバイダは単に「使えない」ものとして扱われる。
 */
export async function getTranscriptionSettings(): Promise<TranscriptionSettings> {
  const [endpointUrl, apiKey] = await Promise.all([
    SecureStore.getItemAsync(ENDPOINT_KEY),
    SecureStore.getItemAsync(API_KEY_KEY),
  ]);
  return {
    endpointUrl: endpointUrl ?? undefined,
    apiKey: apiKey ?? undefined,
  };
}

export async function saveTranscriptionSettings(settings: TranscriptionSettings): Promise<void> {
  if (settings.endpointUrl) {
    await SecureStore.setItemAsync(ENDPOINT_KEY, settings.endpointUrl);
  } else {
    await SecureStore.deleteItemAsync(ENDPOINT_KEY);
  }
  if (settings.apiKey) {
    await SecureStore.setItemAsync(API_KEY_KEY, settings.apiKey);
  } else {
    await SecureStore.deleteItemAsync(API_KEY_KEY);
  }
}
