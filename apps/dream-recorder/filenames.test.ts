import { buildRecordingFileName } from '../src/ids/filenames';

describe('buildRecordingFileName', () => {
  it('ISO日時の「:」をファイル名用に「-」へ変換し、日時とUUIDを含める', () => {
    const name = buildRecordingFileName('2026-08-11T01:42:05.123Z', 'abc-123-uuid');
    expect(name).not.toContain(':');
    expect(name).toContain('2026-08-11T01-42-05.123Z');
    expect(name).toContain('abc-123-uuid');
    expect(name.endsWith('.m4a')).toBe(true);
  });
});
