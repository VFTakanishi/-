import * as SecureStore from 'expo-secure-store';

const KEY = 'dream_recorder_onboarding_seen';

export async function hasSeenOnboarding(): Promise<boolean> {
  const value = await SecureStore.getItemAsync(KEY);
  return value === '1';
}

export async function markOnboardingSeen(): Promise<void> {
  await SecureStore.setItemAsync(KEY, '1');
}
