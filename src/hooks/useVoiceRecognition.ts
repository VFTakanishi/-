import { useCallback, useEffect, useRef, useState } from "react";

export type VoiceRecognitionState = "idle" | "listening" | "processing" | "error" | "unsupported";

export interface VoiceErrorInfo {
  code: string;
  message: string;
}

interface UseVoiceRecognitionOptions {
  onResult: (transcript: string) => void;
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
 */
export function useVoiceRecognition({ onResult, lang = "ja-JP" }: UseVoiceRecognitionOptions): UseVoiceRecognitionResult {
  const [state, setState] = useState<VoiceRecognitionState>(() => (getSpeechRecognitionCtor() ? "idle" : "unsupported"));
  const [isListening, setIsListening] = useState(false);
  const [lastError, setLastError] = useState<VoiceErrorInfo | null>(null);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const wantsListeningRef = useRef(false);
  const endedByFatalErrorRef = useRef(false);
  const onResultRef = useRef(onResult);
  onResultRef.current = onResult;

  const startSession = useCallback(() => {
    const Ctor = getSpeechRecognitionCtor();
    if (!Ctor) {
      setState("unsupported");
      return;
    }

    const recognition = new Ctor();
    recognition.lang = lang;
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => setState("listening");

    recognition.onresult = (event) => {
      setState("processing");
      setLastError(null);
      const lastIndex = event.results.length - 1;
      const transcript = event.results[lastIndex]?.[0]?.transcript ?? "";
      if (transcript.trim()) {
        onResultRef.current(transcript.trim());
      }
    };

    recognition.onerror = (event) => {
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
    wantsListeningRef.current = true;
    setIsListening(true);
    startSession();
  }, [startSession]);

  const stop = useCallback(() => {
    wantsListeningRef.current = false;
    setIsListening(false);
    recognitionRef.current?.stop();
    setState("idle");
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
