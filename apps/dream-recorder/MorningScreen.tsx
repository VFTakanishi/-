import React, { useMemo, useState } from 'react';
import { Alert, ScrollView, StyleSheet, Text, TextInput, TouchableOpacity, View } from 'react-native';
import { Link } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import * as Clipboard from 'expo-clipboard';
import * as WebBrowser from 'expo-web-browser';
import { useAudioPlayer } from 'expo-audio';
import { theme } from './theme';
import { useDreamStore } from '../state/DreamStoreContext';
import { buildGptPasteText } from '../export/gptExport';
import { saveTranscriptionSettings, getTranscriptionSettings } from '../transcription/settingsStore';
import type { DreamRecording } from '../types/models';

const STATUS_LABEL: Record<DreamRecording['status'], string> = {
  saved: '保存済み（未文字起こし）',
  interrupted: '中断あり（保存済み）',
  transcribing: '文字起こし中…',
  transcribed: '文字起こし済み',
  processing: '処理中…',
  completed: '完了',
  error: 'エラー',
};

function formatTime(iso: string): string {
  const d = new Date(iso);
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}

function formatDuration(ms: number): string {
  const totalSeconds = Math.round(ms / 1000);
  const mm = String(Math.floor(totalSeconds / 60)).padStart(2, '0');
  const ss = String(totalSeconds % 60).padStart(2, '0');
  return `${mm}:${ss}`;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  return `${(kb / 1024).toFixed(2)} MB`;
}

function RecordingRow({
  recording,
  onPlay,
  onRetry,
}: {
  recording: DreamRecording;
  onPlay: (uri: string) => void;
  onRetry: (id: string) => void;
}) {
  const canRetry = recording.status === 'error' || recording.status === 'saved' || recording.status === 'interrupted';
  return (
    <View style={styles.row}>
      <View style={styles.rowHeader}>
        <Text style={styles.rowTime}>{formatTime(recording.createdAt)}</Text>
        <Text style={styles.rowMeta}>
          {formatDuration(recording.durationMs)} ・ {formatSize(recording.fileSize)}
        </Text>
      </View>
      <Text style={styles.rowStatus}>{STATUS_LABEL[recording.status]}</Text>
      {recording.errorMessage && <Text style={styles.rowError}>エラー: {recording.errorMessage}</Text>}
      {recording.transcript ? (
        <Text style={styles.transcript}>{recording.transcript}</Text>
      ) : (
        <Text style={styles.noTranscript}>文字起こしなし</Text>
      )}
      <View style={styles.rowActions}>
        <TouchableOpacity style={styles.smallButton} onPress={() => onPlay(recording.fileUri)}>
          <Text style={styles.smallButtonText}>再生</Text>
        </TouchableOpacity>
        {canRetry && (
          <TouchableOpacity style={styles.smallButton} onPress={() => onRetry(recording.id)}>
            <Text style={styles.smallButtonText}>文字起こしを再試行</Text>
          </TouchableOpacity>
        )}
      </View>
    </View>
  );
}

export function MorningScreen() {
  const {
    sessions,
    recordings,
    openSession,
    integrity,
    retryTranscription,
    closeSession,
    resetAllData,
  } = useDreamStore();
  const player = useAudioPlayer(null);
  const [endpointUrl, setEndpointUrl] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [settingsLoaded, setSettingsLoaded] = useState(false);

  React.useEffect(() => {
    getTranscriptionSettings().then((s) => {
      setEndpointUrl(s.endpointUrl ?? '');
      setApiKey(s.apiKey ?? '');
      setSettingsLoaded(true);
    });
  }, []);

  const currentSession = openSession ?? sessions[sessions.length - 1] ?? null;
  const sessionRecordings = useMemo(
    () =>
      currentSession
        ? recordings
            .filter((r) => r.sessionId === currentSession.id)
            .slice()
            .sort((a, b) => a.createdAt.localeCompare(b.createdAt))
        : [],
    [currentSession, recordings]
  );

  const errorRecordings = useMemo(() => recordings.filter((r) => r.status === 'error'), [recordings]);

  const handlePlay = (uri: string) => {
    player.replace(uri);
    player.play();
  };

  const handleCopyGpt = async () => {
    const text = buildGptPasteText(sessionRecordings);
    await Clipboard.setStringAsync(text);
    Alert.alert('コピーしました', 'GPT貼り付け用の文章をクリップボードにコピーしました。');
  };

  const handleOpenChatGpt = async () => {
    await WebBrowser.openBrowserAsync('https://chat.openai.com/');
  };

  const handleConfirmSession = () => {
    if (!currentSession || currentSession.status !== 'open') return;
    Alert.alert('昨夜分を確定しますか？', 'このセッションを閉じます。録音データは削除されません。', [
      { text: 'キャンセル', style: 'cancel' },
      { text: '確定する', onPress: () => void closeSession(currentSession.id) },
    ]);
  };

  const handleResetAll = () => {
    Alert.alert(
      'すべての録音・履歴を削除しますか？',
      'この操作は取り消せません。保存されているすべての録音とセッション履歴が削除されます。',
      [
        { text: 'キャンセル', style: 'cancel' },
        { text: '削除する', style: 'destructive', onPress: () => void resetAllData() },
      ]
    );
  };

  const handleSaveSettings = async () => {
    await saveTranscriptionSettings({ endpointUrl: endpointUrl.trim(), apiKey: apiKey.trim() });
    Alert.alert('保存しました', '文字起こしサーバーAPIの設定を保存しました。');
  };

  return (
    <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <Link href="/" style={styles.backLink}>
          ← 夜間画面へ戻る
        </Link>
        <Text style={styles.title}>朝の確認</Text>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>睡眠セッション</Text>
          {currentSession ? (
            <>
              <Text style={styles.bodyText}>開始: {formatTime(currentSession.startedAt)}</Text>
              <Text style={styles.bodyText}>状態: {currentSession.status}</Text>
              <Text style={styles.bodyText}>録音件数: {sessionRecordings.length}件</Text>
            </>
          ) : (
            <Text style={styles.bodyText}>まだ録音がありません</Text>
          )}
        </View>

        <View style={styles.actionsRow}>
          <TouchableOpacity style={styles.actionButton} onPress={handleConfirmSession}>
            <Text style={styles.actionButtonText}>昨夜分を確定</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.actionButton} onPress={handleCopyGpt}>
            <Text style={styles.actionButtonText}>GPT貼り付け用文章をコピー</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.actionButton} onPress={handleOpenChatGpt}>
            <Text style={styles.actionButtonText}>ChatGPTを開く</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>録音一覧</Text>
          {sessionRecordings.length === 0 && <Text style={styles.bodyText}>録音がありません</Text>}
          {sessionRecordings.map((r) => (
            <RecordingRow key={r.id} recording={r} onPlay={handlePlay} onRetry={retryTranscription} />
          ))}
        </View>

        {errorRecordings.length > 0 && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>処理エラー（全期間）</Text>
            {errorRecordings.map((r) => (
              <RecordingRow key={r.id} recording={r} onPlay={handlePlay} onRetry={retryTranscription} />
            ))}
          </View>
        )}

        {integrity && integrity.orphanFiles.length > 0 && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>復旧候補ファイル（DB未登録）</Text>
            <Text style={styles.bodyText}>
              録音記録に紐づいていない音声ファイルが{integrity.orphanFiles.length}件見つかりました。自動削除はしていません。
            </Text>
            {integrity.orphanFiles.map((f) => (
              <Text key={f.fileUri} style={styles.orphanText}>
                {f.fileUri.split('/').pop()}（{formatSize(f.fileSize)}）
              </Text>
            ))}
          </View>
        )}

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>文字起こしサーバーAPI設定（任意）</Text>
          <Text style={styles.bodyText}>
            設定しなくても録音・保存は完全に動作します。設定した場合のみ、端末内音声認識が使えないときにサーバーへ送信します。
          </Text>
          {settingsLoaded && (
            <>
              <TextInput
                style={styles.input}
                placeholder="https://example.com/transcribe"
                placeholderTextColor="#666"
                autoCapitalize="none"
                autoCorrect={false}
                value={endpointUrl}
                onChangeText={setEndpointUrl}
              />
              <TextInput
                style={styles.input}
                placeholder="APIキー（任意）"
                placeholderTextColor="#666"
                autoCapitalize="none"
                autoCorrect={false}
                secureTextEntry
                value={apiKey}
                onChangeText={setApiKey}
              />
              <TouchableOpacity style={styles.actionButton} onPress={handleSaveSettings}>
                <Text style={styles.actionButtonText}>設定を保存</Text>
              </TouchableOpacity>
            </>
          )}
        </View>

        <View style={styles.dangerSection}>
          <Text style={styles.sectionTitle}>削除・リセット</Text>
          <Text style={styles.bodyText}>この操作は朝用画面からのみ実行できます。確認ダイアログが必ず表示されます。</Text>
          <TouchableOpacity style={styles.dangerButton} onPress={handleResetAll}>
            <Text style={styles.dangerButtonText}>すべての録音・履歴を削除</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: theme.background },
  scrollContent: { padding: 20, gap: 16 },
  backLink: { color: theme.fainterWhite, fontSize: 13 },
  title: { color: theme.white, fontSize: 24, fontWeight: '700' },
  section: { gap: 6 },
  sectionTitle: { color: theme.white, fontSize: 16, fontWeight: '600', marginBottom: 4 },
  bodyText: { color: theme.faintWhite, fontSize: 13 },
  actionsRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  actionButton: {
    backgroundColor: '#1c1c1c',
    borderRadius: 10,
    paddingVertical: 10,
    paddingHorizontal: 14,
  },
  actionButtonText: { color: theme.white, fontSize: 13 },
  row: {
    backgroundColor: '#0d0d0d',
    borderRadius: 12,
    padding: 12,
    gap: 4,
  },
  rowHeader: { flexDirection: 'row', justifyContent: 'space-between' },
  rowTime: { color: theme.white, fontSize: 15, fontWeight: '600' },
  rowMeta: { color: theme.faintWhite, fontSize: 12 },
  rowStatus: { color: theme.faintWhite, fontSize: 12 },
  rowError: { color: theme.error, fontSize: 12 },
  transcript: { color: '#ddd', fontSize: 13, marginTop: 4 },
  noTranscript: { color: theme.fainterWhite, fontSize: 12, marginTop: 4 },
  rowActions: { flexDirection: 'row', gap: 8, marginTop: 6 },
  smallButton: {
    backgroundColor: '#1c1c1c',
    borderRadius: 8,
    paddingVertical: 6,
    paddingHorizontal: 10,
  },
  smallButtonText: { color: theme.white, fontSize: 12 },
  orphanText: { color: theme.faintWhite, fontSize: 12 },
  input: {
    backgroundColor: '#111',
    color: theme.white,
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 13,
  },
  dangerSection: {
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: '#333',
    paddingTop: 16,
    gap: 8,
  },
  dangerButton: {
    backgroundColor: '#3a0d0d',
    borderRadius: 10,
    paddingVertical: 12,
    alignItems: 'center',
  },
  dangerButtonText: { color: '#ff8080', fontSize: 14, fontWeight: '600' },
});
