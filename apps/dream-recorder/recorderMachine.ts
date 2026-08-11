import type { RecorderState } from '../types/models';

export type RecorderEvent =
  | { type: 'START_REQUESTED' }
  | { type: 'START_SUCCEEDED' }
  | { type: 'START_FAILED' }
  | { type: 'STOP_REQUESTED' }
  | { type: 'STOPPED' }
  | { type: 'SAVE_SUCCEEDED' }
  | { type: 'SAVE_FAILED' }
  | { type: 'RESET' };

/**
 * 録音ステートマシンの純粋な遷移関数。
 * 不正な遷移（例: idle中にSTOPPED）は状態を変えずそのまま返す。
 * 副作用は一切持たないため、RecordingController から切り離して単体テストできる。
 */
export function nextRecorderState(current: RecorderState, event: RecorderEvent): RecorderState {
  switch (event.type) {
    case 'START_REQUESTED':
      return current === 'idle' || current === 'saved' || current === 'error'
        ? 'starting'
        : current;
    case 'START_SUCCEEDED':
      return current === 'starting' ? 'recording' : current;
    case 'START_FAILED':
      return current === 'starting' ? 'error' : current;
    case 'STOP_REQUESTED':
      return current === 'recording' ? 'stopping' : current;
    case 'STOPPED':
      return current === 'stopping' ? 'saving' : current;
    case 'SAVE_SUCCEEDED':
      return current === 'saving' ? 'saved' : current;
    case 'SAVE_FAILED':
      return current === 'saving' ? 'error' : current;
    case 'RESET':
      return current === 'saved' || current === 'error' ? 'idle' : current;
    default:
      return current;
  }
}

/**
 * starting / stopping / saving の間は操作を受け付けない。
 * これが「ロック中かどうか」の唯一の判定基準になる。
 */
export function isLockedState(state: RecorderState): boolean {
  return state === 'starting' || state === 'stopping' || state === 'saving';
}
