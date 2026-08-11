import { isLockedState, nextRecorderState } from '../src/core/recorderMachine';

describe('recorderMachine', () => {
  it('idle -> starting -> recording の正常遷移', () => {
    let state = nextRecorderState('idle', { type: 'START_REQUESTED' });
    expect(state).toBe('starting');
    state = nextRecorderState(state, { type: 'START_SUCCEEDED' });
    expect(state).toBe('recording');
  });

  it('recording -> stopping -> saving -> saved の正常遷移', () => {
    let state = nextRecorderState('recording', { type: 'STOP_REQUESTED' });
    expect(state).toBe('stopping');
    state = nextRecorderState(state, { type: 'STOPPED' });
    expect(state).toBe('saving');
    state = nextRecorderState(state, { type: 'SAVE_SUCCEEDED' });
    expect(state).toBe('saved');
  });

  it('saved / error からは RESET で idle に戻れる', () => {
    expect(nextRecorderState('saved', { type: 'RESET' })).toBe('idle');
    expect(nextRecorderState('error', { type: 'RESET' })).toBe('idle');
  });

  it('starting/stopping/saving 中に不正なイベントを送っても状態は変わらない', () => {
    expect(nextRecorderState('starting', { type: 'STOP_REQUESTED' })).toBe('starting');
    expect(nextRecorderState('saving', { type: 'START_REQUESTED' })).toBe('saving');
    expect(nextRecorderState('stopping', { type: 'START_REQUESTED' })).toBe('stopping');
  });

  it('isLockedState は starting/stopping/saving のときだけ true', () => {
    expect(isLockedState('starting')).toBe(true);
    expect(isLockedState('stopping')).toBe(true);
    expect(isLockedState('saving')).toBe(true);
    expect(isLockedState('idle')).toBe(false);
    expect(isLockedState('recording')).toBe(false);
    expect(isLockedState('saved')).toBe(false);
    expect(isLockedState('error')).toBe(false);
  });
});
