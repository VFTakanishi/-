import { useCallback, useEffect, useRef, useState } from "react";

export type VoiceRecognitionState = "idle" | "listening" | "processing" | "error" | "unsupported";

export interface VoiceErrorInfo {
  code: string;
  message: string;
}

interface UseVoiceRecognitionOptions {
  /**
   * Safariが返す複数の認識候補（alternatives）をそのまま渡す。
   * 候補は1件以上、Safariの信頼度順（先頭が最有力候補）。
   * 「点検項目として最も自然に成立する候補」を選ぶ判断は呼び出し側（parser）に委ねる。
   */
  onResult: (transcripts: string[]) => void;
  lang?: string;
}

interface UseVoiceRecognitionResult {
  state: VoiceRecognitionState;
  isListening: boolean;
  lastError: VoiceErrorInfo | null;
  start: () => void;
  stop: () => void;
  toggle: () => void;
}

const RESTART_DELAY_MS = 300;

// no-speech 以外は「再試行しても意味がない／再試行を続けると危険」なエラーとして
// 自動再開ループを止める（not-allowed, service-not-allowed, audio-capture, network,
// aborted, その他未知のコードすべてを含む）。
const CONTINUE_LISTENING_ERROR_CODES = new Set(["no-speech"]);

const ERROR_MESSAGES: Record<string, string> = {
  "not-allowed": "マイク/音声認識の許可が拒否されています",
  "service-not-allowed": "音声認識サービスが許可されていません",
  "audio-capture": "マイクを取得できません",
  network: "音声認識サービスへの通信に失敗しました",
  "no-speech": "音声が検出されませんでした",
  aborted: "音声認識が中断されました",
};

export function describeVoiceErrorCode(code: string): string {
  return ERROR_MESSAGES[code] ?? `エラーコード: ${code}`;
}

function getSpeechRecognitionCtor(): (new () => SpeechRecognitionLike) | undefined {
  if (typeof window === "undefined") return undefined;
  return window.SpeechRecognition ?? window.webkitSpeechRecognition;
}

/**
 * iOS Safari の Web Speech API は continuous:true が不安定なため、
 * 短時間の non-continuous セッションを onend のたびに自動再開して
 * 疑似的な連続リスニングを実現する。自動再開に失敗した場合は
 * isListening が false に落ちるので、大きな手動マイクボタン側から
 * 再度 start() を呼べば復帰できる。
 *
 * no-speech 以外のエラー（not-allowed / service-not-allowed / audio-capture /
 * network / aborted / 未知のコード）では、再試行しても回復しないか無意味なため
 * 自動再開ループを止め、isListening を false に戻す。
 *
 * ただし、ユーザー自身がマイクボタンで stop() した結果として Safari が
 * aborted / no-speech 等を返すことがあるため、manualStopRef で
 * 「意図的な停止」と「予期しないエラー」を区別し、意図的な停止では
 * エラー表示をしない。
 */
export function useVoiceRecognition({ onResult, lang = "ja-JP" }: UseVoiceRecognitionOptions): UseVoiceRecognitionResult {
  const [state, setState] = useState<VoiceRecognitionState>(() => (getSpeechRecognitionCtor() ? "idle" : "unsupported"));
  const [isListening, setIsListening] = useState(false);
  const [lastError, setLastError] = useState<VoiceErrorInfo | null>(null);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const wantsListeningRef = useRef(false);
  const endedByFatalErrorRef = useRef(false);
  const manualStopRef = useRef(false);
  const onResultRef = useRef(onResult);
  onResultRef.current = onResult;

  const startSession = useCallback(() => {
    const Ctor = getSpeechRecognitionCtor();
    if (!Ctor) {
      setState("unsupported");
      return;
    }

    // 直前のセッションがまだ動いている場合、同一インスタンスへの二重start等を
    // 避けるため新規セッションは開始しない（onendで確実にnullへ戻る）。
    if (recognitionRef.current) {
      return;
    }

    const recognition = new Ctor();
    recognition.lang = lang;
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.maxAlternatives = 5;

    recognition.onstart = () => setState("listening");

    recognition.onresult = (event) => {
      setState("processing");
      setLastError(null);
      const lastIndex = event.results.length - 1;
      const result = event.results[lastIndex];
      const transcripts: string[] = [];
      if (result) {
        for (let i = 0; i < result.length; i++) {
          const t = result[i]?.transcript?.trim();
          if (t) transcripts.push(t);
        }
      }
      if (transcripts.length > 0) {
        onResultRef.current(transcripts);
      }
    };

    recognition.onerror = (event) => {
      if (manualStopRef.current) {
        // ユーザーによる意図的な停止に伴うエラー（aborted/no-speech等）は無視する
        return;
      }

      const code = event.error || "unknown";
      const message = event.message || "";
      setLastError({ code, message });
      setState("error");

      if (!CONTINUE_LISTENING_ERROR_CODES.has(code)) {
        endedByFatalErrorRef.current = true;
        wantsListeningRef.current = false;
        setIsListening(false);
      }
    };

    recognition.onend = () => {
      recognitionRef.current = null;

      if (manualStopRef.current) {
        manualStopRef.current = false;
        setState("idle");
        return;
      }

      if (wantsListeningRef.current) {
        setTimeout(() => {
          if (wantsListeningRef.current) startSession();
        }, RESTART_DELAY_MS);
      } else if (endedByFatalErrorRef.current) {
        // エラー内容を確認できるよう、idle に戻さず error 表示を維持する
        endedByFatalErrorRef.current = false;
      } else {
        setState("idle");
      }
    };

    recognitionRef.current = recognition;
    try {
      recognition.start();
    } catch (err) {
      recognitionRef.current = null;
      const message = err instanceof Error ? err.message : String(err);
      setLastError({ code: "start-failed", message });
      setState("error");
      wantsListeningRef.current = false;
      setIsListening(false);
    }
  }, [lang]);

  const start = useCallback(() => {
    if (!getSpeechRecognitionCtor()) {
      setState("unsupported");
      return;
    }
    if (recognitionRef.current) {
      // 既にセッションが動作中（二重タップ等）: 何もしない
      return;
    }
    manualStopRef.current = false;
    setLastError(null);
    wantsListeningRef.current = true;
    setIsListening(true);
    startSession();
  }, [startSession]);

  const stop = useCallback(() => {
    manualStopRef.current = true;
    wantsListeningRef.current = false;
    setIsListening(false);
    setLastError(null);
    setState("idle");
    recognitionRef.current?.stop();
  }, []);

  const toggle = useCallback(() => {
    if (isListening) stop();
    else start();
  }, [isListening, start, stop]);

  useEffect(() => {
    return () => {
      wantsListeningRef.current = false;
      recognitionRef.current?.stop();
    };
  }, []);

  return { state, isListening, lastError, start, stop, toggle };
}
