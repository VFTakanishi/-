/**
 * 夜間画面はライト/ダークモードに関わらず常にほぼ完全な黒を維持する。
 * ここで固定色として定義し、システムのカラースキームには依存させない。
 */
export const theme = {
  background: '#000000',
  surface: '#0a0a0a',
  red: '#e11d2e',
  redPressed: '#b8121f',
  white: '#ffffff',
  faintWhite: 'rgba(255,255,255,0.35)',
  fainterWhite: 'rgba(255,255,255,0.22)',
  error: '#ff6b6b',
} as const;
