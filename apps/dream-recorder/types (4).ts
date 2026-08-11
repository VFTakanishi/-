export type TranscriptionOutcome =
  | { ok: true; transcript: string }
  | { ok: false; errorMessage: string };

export type TranscriptionProviderId = 'on-device' | 'server';

export interface TranscriptionProvider {
  id: TranscriptionProviderId;
  /** 実行前に「使える見込みがあるか」を軽く確認する。使えなければ次の候補へ回す。 */
  isAvailable(): Promise<boolean>;
  transcribe(fileUri: string): Promise<TranscriptionOutcome>;
}
