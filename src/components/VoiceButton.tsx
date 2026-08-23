import type { VoiceRecognitionState } from "../hooks/useVoiceRecognition";

const STATE_LABEL: Record<VoiceRecognitionState, string> = {
  idle: "音声入力停止中",
  listening: "認識中…",
  processing: "解析中…",
  error: "認識失敗（下のエラー内容を確認してください）",
  unsupported: "この端末では音声入力を利用できません",
};

interface VoiceButtonProps {
  state: VoiceRecognitionState;
  isListening: boolean;
  onClick: () => void;
}

export function VoiceButton({ state, isListening, onClick }: VoiceButtonProps) {
  const disabled = state === "unsupported";
  return (
    <div className="voice-button-wrap">
      <button
        type="button"
        className={`voice-button voice-button--${state}`}
        onClick={onClick}
        disabled={disabled}
        aria-pressed={isListening}
      >
        {isListening ? "🎙" : "🎤"}
      </button>
      <div className={`voice-button-label voice-button-label--${state}`}>{STATE_LABEL[state]}</div>
    </div>
  );
}
