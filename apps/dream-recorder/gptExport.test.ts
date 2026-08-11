import { buildGptPasteText } from '../src/export/gptExport';
import type { DreamRecording } from '../src/types/models';

function rec(overrides: Partial<DreamRecording>): DreamRecording {
  return {
    id: overrides.id ?? 'id',
    sessionId: 'session-1',
    createdAt: overrides.createdAt ?? '2026-08-11T01:00:00.000Z',
    durationMs: 1000,
    fileUri: 'file:///x.m4a',
    fileSize: 10,
    status: 'transcribed',
    ...overrides,
  };
}

describe('buildGptPasteText', () => {
  it('12. 文字起こし済みの録音を時刻順に並べる', () => {
    const recordings = [
      rec({ id: 'c', createdAt: '2026-08-11T04:18:00.000Z', transcript: '3つ目の夢' }),
      rec({ id: 'a', createdAt: '2026-08-11T01:42:00.000Z', transcript: '最初の夢' }),
      rec({ id: 'b', createdAt: '2026-08-11T02:50:00.000Z', transcript: '2つ目の夢' }),
    ];

    const text = buildGptPasteText(recordings);

    const idxA = text.indexOf('最初の夢');
    const idxB = text.indexOf('2つ目の夢');
    const idxC = text.indexOf('3つ目の夢');
    expect(idxA).toBeGreaterThan(-1);
    expect(idxB).toBeGreaterThan(idxA);
    expect(idxC).toBeGreaterThan(idxB);
    expect(text).toContain('【録音1：');
    expect(text).toContain('【録音2：');
    expect(text).toContain('【録音3：');
    expect(text).toContain('内容を推測で追加せず');
  });

  it('文字起こしの無い録音は含めない', () => {
    const recordings = [
      rec({ id: 'a', transcript: undefined }),
      rec({ id: 'b', createdAt: '2026-08-11T02:00:00.000Z', transcript: '記録された夢' }),
    ];
    const text = buildGptPasteText(recordings);
    expect(text).toContain('記録された夢');
    expect(text).not.toContain('【録音2：');
  });

  it('文字起こしが1件も無ければヘッダーのみを返す', () => {
    const text = buildGptPasteText([]);
    expect(text).toContain('以下は昨夜見た夢の音声記録です。');
    expect(text).not.toContain('【録音');
  });
});
