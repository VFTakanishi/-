import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { AppState, type AppStateStatus } from 'react-native';
import { activateKeepAwakeAsync, deactivateKeepAwake } from 'expo-keep-awake';
import { db, audio, files, clock, recordingController, transcriptionProviders } from './services';
import { runStartupIntegrityCheck, type IntegrityCheckResult } from '../core/integrityCheck';
import { transcribeRecording } from '../transcription/transcribeRecording';
import type { ControllerSnapshot } from '../core/RecordingController';
import type { DreamRecording, SleepSession } from '../types/models';

const KEEP_AWAKE_TAG = 'dream-recorder-recording';

type DreamStoreValue = {
  ready: boolean;
  recorderState: ControllerSnapshot;
  recordings: DreamRecording[];
  sessions: SleepSession[];
  openSession: SleepSession | null;
  integrity: IntegrityCheckResult | null;
  tonightCount: number;
  pressButton: () => Promise<void>;
  resetErrorState: () => void;
  refresh: () => Promise<void>;
  retryTranscription: (id: string) => Promise<void>;
  closeSession: (sessionId: string, summary?: string) => Promise<void>;
  /** 確認ダイアログを通過した呼び出し元だけが使うこと。朝用画面にのみ置く。 */
  resetAllData: () => Promise<void>;
};

const DreamStoreCtx = createContext<DreamStoreValue | null>(null);

export function DreamStoreProvider({ children }: { children: React.ReactNode }) {
  const [ready, setReady] = useState(false);
  const [recorderState, setRecorderState] = useState<ControllerSnapshot>(recordingController.getSnapshot());
  const [recordings, setRecordings] = useState<DreamRecording[]>([]);
  const [sessions, setSessions] = useState<SleepSession[]>([]);
  const [integrity, setIntegrity] = useState<IntegrityCheckResult | null>(null);
  const keepAwakeActive = useRef(false);

  const refresh = useCallback(async () => {
    const [allRecordings, allSessions] = await Promise.all([db.recordings.getAll(), db.sessions.getAll()]);
    setRecordings(allRecordings);
    setSessions(allSessions);
  }, []);

  // 起動時: SQLiteから復元 + 整合性検査。ここで自動リセットは絶対に行わない。
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const result = await runStartupIntegrityCheck(db, files, () => clock.nowIso());
      if (cancelled) return;
      setIntegrity(result);
      await refresh();
      recordingController.attachInterruptionWatcher();
      if (!cancelled) setReady(true);
    })();
    return () => {
      cancelled = true;
      recordingController.dispose();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // RecordingController の状態変化を購読。保存完了のたびにDBから再読込する。
  useEffect(() => {
    const unsubscribe = recordingController.subscribe((snapshot) => {
      setRecorderState(snapshot);
      if (snapshot.state === 'saved') {
        void refresh();
      }
    });
    return unsubscribe;
  }, [refresh]);

  // 録音中だけ画面をスリープさせない。
  useEffect(() => {
    const shouldKeepAwake = recorderState.state === 'recording' || recorderState.state === 'starting';
    if (shouldKeepAwake && !keepAwakeActive.current) {
      keepAwakeActive.current = true;
      void activateKeepAwakeAsync(KEEP_AWAKE_TAG);
    } else if (!shouldKeepAwake && keepAwakeActive.current) {
      keepAwakeActive.current = false;
      void deactivateKeepAwake(KEEP_AWAKE_TAG);
    }
  }, [recorderState.state]);

  // AppStateの変化を監視。フォアグラウンド復帰時、録音が実は止まっていないか再確認する。
  useEffect(() => {
    const onChange = (nextState: AppStateStatus) => {
      if (nextState === 'active' && recordingController.getSnapshot().state === 'recording') {
        void audio.checkStillRecording().then((stillRecording) => {
          if (!stillRecording) {
            recordingController.notifyPossibleInterruption();
          }
        });
      }
    };
    const sub = AppState.addEventListener('change', onChange);
    return () => sub.remove();
  }, []);

  const pressButton = useCallback(() => recordingController.pressButton(), []);
  const resetErrorState = useCallback(() => recordingController.resetToIdleFromError(), []);

  const retryTranscription = useCallback(
    async (id: string) => {
      const recording = await db.recordings.getById(id);
      if (!recording) return;
      await transcribeRecording(db, transcriptionProviders, recording);
      await refresh();
    },
    [refresh]
  );

  const closeSession = useCallback(
    async (sessionId: string, summary?: string) => {
      await db.sessions.update(sessionId, {
        status: 'closed',
        closedAt: clock.nowIso(),
        summary,
      });
      await refresh();
    },
    [refresh]
  );

  const resetAllData = useCallback(async () => {
    await db.recordings.deleteAll();
    await db.sessions.deleteAll();
    await refresh();
  }, [refresh]);

  const openSession = useMemo(() => sessions.find((s) => s.status === 'open') ?? null, [sessions]);

  const tonightCount = useMemo(() => {
    if (!openSession) return 0;
    return recordings.filter((r) => r.sessionId === openSession.id).length;
  }, [openSession, recordings]);

  const value: DreamStoreValue = {
    ready,
    recorderState,
    recordings,
    sessions,
    openSession,
    integrity,
    tonightCount,
    pressButton,
    resetErrorState,
    refresh,
    retryTranscription,
    closeSession,
    resetAllData,
  };

  return <DreamStoreCtx.Provider value={value}>{children}</DreamStoreCtx.Provider>;
}

export function useDreamStore(): DreamStoreValue {
  const ctx = useContext(DreamStoreCtx);
  if (!ctx) throw new Error('useDreamStore must be used within DreamStoreProvider');
  return ctx;
}
