import React from 'react';
import { Pressable, StyleSheet, Text, useWindowDimensions } from 'react-native';
import { theme } from './theme';
import type { RecorderState } from '../types/models';

const LABELS: Record<RecorderState, string> = {
  idle: '● 夢を録音',
  starting: '● 夢を録音',
  recording: '■ 録音を終了',
  stopping: '■ 録音を終了',
  saving: '保存中…',
  saved: '保存しました',
  error: 'もう一度タップ',
};

type Props = {
  state: RecorderState;
  onPress: () => void;
};

/**
 * 夜間画面でいちばん目立つ、唯一の操作要素。
 * 幅は画面の約88%、高さは220〜280px。状態が変わってもサイズと位置は動かさない
 * （文字ラベルだけが変わる）。
 */
export function BigButton({ state, onPress }: Props) {
  const { width, height } = useWindowDimensions();
  const buttonWidth = width * 0.88;
  const buttonHeight = Math.min(280, Math.max(220, height * 0.28));
  const locked = state === 'starting' || state === 'stopping' || state === 'saving';
  const top = height * 0.55 - buttonHeight / 2;

  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={LABELS[state]}
      disabled={locked}
      onPress={onPress}
      style={({ pressed }) => [
        styles.button,
        {
          width: buttonWidth,
          height: buttonHeight,
          left: (width - buttonWidth) / 2,
          top,
          backgroundColor: pressed ? theme.redPressed : theme.red,
          opacity: locked ? 0.85 : 1,
        },
      ]}
    >
      <Text style={styles.label}>{LABELS[state]}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  button: {
    position: 'absolute',
    borderRadius: 40,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#000',
    shadowOpacity: 0.6,
    shadowRadius: 20,
    shadowOffset: { width: 0, height: 8 },
    elevation: 8,
  },
  label: {
    color: theme.white,
    fontSize: 30,
    fontWeight: '700',
    textAlign: 'center',
    paddingHorizontal: 16,
  },
});
