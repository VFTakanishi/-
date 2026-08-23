import { useCallback, useEffect, useRef, useState } from "react";

export type VoiceRecognitionState = "idle" | "listening" | "processing" | "error" | "unsupported";

interface UseVoiceRecognitionOptions {
  onResult: (transcript: string) => void;
  lang?: string;
}

interface UseVoiceRecognitionResult {
  state: VoiceRecognitionState;
  isListening: boolean;
  start: () => void;
  stop: () => void;
  toggle: () => void;
}

const RESTART_DELAY_MS = 300;

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
 */
export function useVoiceRecognition({ onResult, lang = "ja-JP" }: UseVoiceRecognitionOptions): UseVoiceRecognitionResult {
  const [state, setState] = useState<VoiceRecognitionState>(() => (getSpeechRecognitionCtor() ? "idle" : "unsupported"));
  const [isListening, setIsListening] = useState(false);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const wantsListeningRef = useRef(false);
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
      const lastIndex = event.results.length - 1;
      const transcript = event.results[lastIndex]?.[0]?.transcript ?? "";
      if (transcript.trim()) {
        onResultRef.current(transcript.trim());
      }
    };

    recognition.onerror = () => {
      setState("error");
    };

    recognition.onend = () => {
      recognitionRef.current = null;
      if (wantsListeningRef.current) {
        setTimeout(() => {
          if (wantsListeningRef.current) startSession();
        }, RESTART_DELAY_MS);
      } else {
        setState("idle");
      }
    };

    recognitionRef.current = recognition;
    try {
      recognition.start();
    } catch {
      setState("error");
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

  return { state, isListening, start, stop, toggle };
}
