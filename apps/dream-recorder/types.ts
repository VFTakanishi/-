export interface HapticsService {
  /** 録音開始成功時に短い振動を1回。 */
  recordingStarted(): Promise<void>;
  /** 保存完全成功時に一定間隔で短い振動を2回。 */
  savedSuccessfully(): Promise<void>;
  /** 保存成功と明確に異なるエラー振動。 */
  failure(): Promise<void>;
}
