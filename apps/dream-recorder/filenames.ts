/**
 * 録音ファイル名の生成。
 * - 日時 + UUID を含める
 * - ISO日時に含まれる `:` はファイルシステムで使えない場合があるため `-` に変換する
 */
export function buildRecordingFileName(createdAtIso: string, id: string): string {
  const safeTimestamp = createdAtIso.replace(/:/g, '-');
  return `dream_${safeTimestamp}_${id}.m4a`;
}
