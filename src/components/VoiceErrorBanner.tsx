import { describeVoiceErrorCode, type VoiceErrorInfo } from "../hooks/useVoiceRecognition";

interface VoiceErrorBannerProps {
  error: VoiceErrorInfo | null;
}

/**
 * iPhone Safari実機での原因調査用に、直近の音声認識エラーのコードと
 * 内容を大きく表示する。onerror が握りつぶされないよう、
 * event.error / event.message をそのまま確認できるようにしている。
 */
export function VoiceErrorBanner({ error }: VoiceErrorBannerProps) {
  if (!error) return null;

  return (
    <div className="voice-error-banner">
      <div className="voice-error-banner-title">音声認識エラー</div>
      <div className="voice-error-banner-desc">{describeVoiceErrorCode(error.code)}</div>
      <div className="voice-error-banner-code">code: {error.code}</div>
      {error.message && <div className="voice-error-banner-message">{error.message}</div>}
    </div>
  );
}
