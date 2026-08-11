import * as Haptics from 'expo-haptics';
import type { HapticsService } from './types';

const DOUBLE_TAP_INTERVAL_MS = 180;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * expo-haptics を使った実機用の振動サービス実装。
 * 振動の失敗（デバイス側の一時的な問題など）は上位の保存処理に影響させない。
 * 呼び出し元は catch 不要になるよう、ここで例外を握りつぶしログにするだけにする。
 */
export class ExpoHapticsService implements HapticsService {
  async recordingStarted(): Promise<void> {
    await this.safeCall(() => Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success));
  }

  async savedSuccessfully(): Promise<void> {
    await this.safeCall(() => Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success));
    await sleep(DOUBLE_TAP_INTERVAL_MS);
    await this.safeCall(() => Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success));
  }

  async failure(): Promise<void> {
    await this.safeCall(() => Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error));
  }

  private async safeCall(fn: () => Promise<void>): Promise<void> {
    try {
      await fn();
    } catch (error) {
      // 振動失敗は保存済みデータに影響させない。ログのみ。
      console.warn('[haptics] failed', error);
    }
  }
}
