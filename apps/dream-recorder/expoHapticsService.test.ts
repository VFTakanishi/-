import * as Haptics from 'expo-haptics';
import { ExpoHapticsService } from '../src/haptics/expoHapticsService';

// jest.mock は babel-jest によりファイル先頭へ自動的にホイストされるため、
// import文の後に書いても実行順序上は import より先に評価される。
jest.mock('expo-haptics', () => ({
  notificationAsync: jest.fn(async () => undefined),
  NotificationFeedbackType: { Success: 'success', Warning: 'warning', Error: 'error' },
}));

describe('ExpoHapticsService', () => {
  beforeEach(() => {
    (Haptics.notificationAsync as jest.Mock).mockClear();
  });

  it('5. 保存完全成功後は notificationAsync(Success) を2回呼ぶ', async () => {
    const service = new ExpoHapticsService();
    await service.savedSuccessfully();
    const calls = (Haptics.notificationAsync as jest.Mock).mock.calls;
    expect(calls).toHaveLength(2);
    expect(calls[0][0]).toBe(Haptics.NotificationFeedbackType.Success);
    expect(calls[1][0]).toBe(Haptics.NotificationFeedbackType.Success);
  });

  it('録音開始成功時は1回だけ呼ぶ', async () => {
    const service = new ExpoHapticsService();
    await service.recordingStarted();
    expect((Haptics.notificationAsync as jest.Mock).mock.calls).toHaveLength(1);
  });

  it('失敗時は成功と異なる Error タイプを呼ぶ', async () => {
    const service = new ExpoHapticsService();
    await service.failure();
    const calls = (Haptics.notificationAsync as jest.Mock).mock.calls;
    expect(calls[0][0]).toBe(Haptics.NotificationFeedbackType.Error);
  });

  it('振動が失敗しても例外を投げない(保存済みデータに影響させない)', async () => {
    (Haptics.notificationAsync as jest.Mock).mockRejectedValueOnce(new Error('device busy'));
    const service = new ExpoHapticsService();
    await expect(service.savedSuccessfully()).resolves.toBeUndefined();
  });
});
