const MEASUREMENT_ID = "G-4281T68VRT";
const PRODUCTION_HOSTNAME = "shaken-voice-inspection.vercel.app";

export type AnalyticsEventName =
  | "inspection_start"
  | "microphone_start"
  | "speech_recognition_success"
  | "unrecognized_input"
  | "summary_view"
  | "back_to_top";

declare global {
  interface Window {
    dataLayer?: unknown[];
    gtag?: (...args: unknown[]) => void;
  }
}

/**
 * GA4 is deliberately enabled only on the canonical production hostname.
 * Preview deployments, localhost, and branch deployments never load gtag.js.
 */
export function initializeAnalytics(): void {
  if (typeof window === "undefined" || window.location.hostname !== PRODUCTION_HOSTNAME || window.gtag) {
    return;
  }

  window.dataLayer = window.dataLayer ?? [];
  window.gtag = (...args: unknown[]) => {
    window.dataLayer?.push(args);
  };

  window.gtag("js", new Date());
  window.gtag("config", MEASUREMENT_ID, {
    allow_google_signals: false,
    allow_ad_personalization_signals: false,
    send_page_view: true,
  });

  const script = document.createElement("script");
  script.async = true;
  script.src = `https://www.googletagmanager.com/gtag/js?id=${MEASUREMENT_ID}`;
  document.head.appendChild(script);
}

/**
 * Event payloads are intentionally parameter-free. Never add customer, vehicle,
 * mileage, transcript, memo, inspection ID, or other record data here.
 */
export function trackAnalyticsEvent(eventName: AnalyticsEventName): void {
  if (typeof window === "undefined" || window.location.hostname !== PRODUCTION_HOSTNAME) {
    return;
  }
  window.gtag?.("event", eventName);
}
