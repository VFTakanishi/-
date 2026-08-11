import React, { useEffect, useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { Link } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { BigButton } from './BigButton';
import { theme } from './theme';
import { useDreamStore } from '../state/DreamStoreContext';
import { hasSeenOnboarding, markOnboardingSeen } from '../state/onboarding';

function formatElapsed(ms: number): string {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const mm = String(Math.floor(totalSeconds / 60)).padStart(2, '0');
  const ss = String(totalSeconds % 60).padStart(2, '0');
  return `${mm}:${ss}`;
}

export function NightScreen() {
  const { recorderState, tonightCount, pressButton, resetErrorState } = useDreamStore();
  const [elapsedMs, setElapsedMs] = useState(0);
  const [showOnboarding, setShowOnboarding] = useState(false);

  useEffect(() => {
    hasSeenOnboarding().then((seen) => setShowOnboarding(!seen));
  }, []);

  const isRecording = recorderState.state === 'recording';
  const recordingStartedAtMs = recorderState.recordingStartedAtMs;

  useEffect(() => {
    if (!isRecording || !recordingStartedAtMs) {
      return;
    }
    const id = setInterval(() => setElapsedMs(Date.now() - recordingStartedAtMs), 500);
    return () => clearInterval(id);
  }, [isRecording, recordingStartedAtMs]);

  const handlePress = () => {
    if (showOnboarding) {
      setShowOnboarding(false);
      void markOnboardingSeen();
    }
    if (recorderState.state === 'error') {
      resetErrorState();
      return;
    }
    void pressButton();
  };

  return (
    <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
      <View style={styles.topArea}>
        <Text style={styles.count}>今夜 {tonightCount}件保存済み</Text>
        {showOnboarding && (
          <Text style={styles.onboarding}>
            初回のみ：ボタンを押すとマイクへのアクセス許可を確認します
          </Text>
        )}
        {recorderState.state === 'recording' && <Text style={styles.timer}>{formatElapsed(elapsedMs)}</Text>}
        {recorderState.state === 'error' && (
          <Text style={styles.errorText}>保存できませんでした。もう一度お試しください</Text>
        )}
      </View>

      <BigButton state={recorderState.state} onPress={handlePress} />

      <View style={styles.bottomArea}>
        <Link href="/morning" style={styles.morningLink}>
          朝の確認へ
        </Link>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.background,
  },
  topArea: {
    paddingTop: 12,
    alignItems: 'center',
    gap: 8,
  },
  count: {
    color: theme.faintWhite,
    fontSize: 14,
  },
  onboarding: {
    color: theme.fainterWhite,
    fontSize: 12,
    textAlign: 'center',
    paddingHorizontal: 32,
  },
  timer: {
    color: theme.faintWhite,
    fontSize: 16,
    fontVariant: ['tabular-nums'],
  },
  errorText: {
    color: theme.error,
    fontSize: 13,
    textAlign: 'center',
    paddingHorizontal: 32,
  },
  bottomArea: {
    flex: 1,
    justifyContent: 'flex-end',
    alignItems: 'center',
    paddingBottom: 18,
  },
  morningLink: {
    color: theme.fainterWhite,
    fontSize: 12,
  },
});
