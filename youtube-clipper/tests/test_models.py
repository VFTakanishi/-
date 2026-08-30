import pytest

from podcast_clipper.models import (
    ClipCandidate,
    RawClipCandidate,
    RawUsedSegment,
    TranscriptSegment,
    TranscriptWord,
    UsedSegment,
    find_opening_trim_point,
)


def _segment(role="hook", start=0.0, end=1.0):
    return UsedSegment(role=role, start=start, end=end, text="hello")


def test_clip_candidate_accepts_1_to_3_segments():
    for n in (1, 2, 3):
        segs = [_segment(start=i, end=i + 1) for i in range(n)]
        c = ClipCandidate(
            id="c1", hook_type="strong_take", segments=segs, hook_text="h",
            opening_hook_strength=80, title="t", description="d", score=50,
            reasoning="r", caveats="",
        )
        assert len(c.segments) == n


def test_clip_candidate_rejects_zero_or_more_than_3_segments():
    with pytest.raises(ValueError):
        ClipCandidate(
            id="c1", hook_type="strong_take", segments=[], hook_text="h",
            opening_hook_strength=80, title="t", description="d", score=50,
            reasoning="r", caveats="",
        )
    with pytest.raises(ValueError):
        ClipCandidate(
            id="c1", hook_type="strong_take",
            segments=[_segment(start=i, end=i + 1) for i in range(4)],
            hook_text="h", opening_hook_strength=80, title="t", description="d",
            score=50, reasoning="r", caveats="",
        )


def test_clip_candidate_rejects_score_out_of_range():
    with pytest.raises(ValueError):
        ClipCandidate(
            id="c1", hook_type="strong_take", segments=[_segment()],
            hook_text="h", opening_hook_strength=80, title="t", description="d",
            score=101, reasoning="r", caveats="",
        )


def test_clip_candidate_rejects_opening_hook_strength_out_of_range():
    with pytest.raises(ValueError):
        ClipCandidate(
            id="c1", hook_type="strong_take", segments=[_segment()],
            hook_text="h", opening_hook_strength=101, title="t", description="d",
            score=50, reasoning="r", caveats="",
        )
    with pytest.raises(ValueError):
        ClipCandidate(
            id="c1", hook_type="strong_take", segments=[_segment()],
            hook_text="h", opening_hook_strength=-1, title="t", description="d",
            score=50, reasoning="r", caveats="",
        )


def test_used_segment_rejects_inverted_range():
    with pytest.raises(ValueError):
        UsedSegment(role="hook", start=5.0, end=1.0, text="x")


def test_total_duration_sums_segments():
    segs = [_segment(start=0, end=2), _segment(start=10, end=13)]
    c = ClipCandidate(
        id="c1", hook_type="story", segments=segs, hook_text="h",
        opening_hook_strength=80, title="t", description="d", score=10,
        reasoning="r", caveats="",
    )
    assert c.total_duration == 5.0


def test_raw_clip_candidate_segment_count_bounds():
    seg = RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0)
    with pytest.raises(ValueError):
        RawClipCandidate(
            hook_type="open_loop", segments=[], hook_text="h", opening_hook_strength=80,
            title="t", description="d", score=1, reasoning="r", caveats="",
        )
    RawClipCandidate(
        hook_type="open_loop", segments=[seg, seg], hook_text="h", opening_hook_strength=80,
        title="t", description="d", score=1, reasoning="r", caveats="",
    )


def test_raw_clip_candidate_rejects_opening_hook_strength_out_of_range():
    seg = RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0)
    with pytest.raises(ValueError):
        RawClipCandidate(
            hook_type="open_loop", segments=[seg], hook_text="h", opening_hook_strength=200,
            title="t", description="d", score=1, reasoning="r", caveats="",
        )


def _words(*texts_and_times: tuple[str, float, float]) -> list[TranscriptWord]:
    return [TranscriptWord(start=s, end=e, text=t) for t, s, e in texts_and_times]


def test_find_opening_trim_point_detects_konoyouni_at_start():
    words = _words(
        ("このように", 0.0, 0.55),
        ("弱点を", 0.56, 0.90),
        ("直すと", 0.91, 1.30),
    )
    seg = TranscriptSegment(id=0, start=0.0, end=1.30, text="このように弱点を直すと", words=words)
    trim_word = find_opening_trim_point(seg)
    assert trim_word is not None
    assert trim_word.text == "弱点を"
    assert trim_word.start == 0.56


def test_find_opening_trim_point_detects_other_known_prefix():
    words = _words(("あの", 0.0, 0.2), ("今日", 0.2, 0.5), ("は", 0.5, 0.6))
    seg = TranscriptSegment(id=0, start=0.0, end=0.6, text="あの今日は", words=words)
    trim_word = find_opening_trim_point(seg)
    assert trim_word is not None
    assert trim_word.text == "今日"


def test_find_opening_trim_point_ignores_mid_sentence_occurrence():
    # "このように" only appears once here, but it's not a prefix-buildable
    # match from word 0 ("私" doesn't accumulate toward any known phrase),
    # so no trim point should ever be found -- mirroring the real
    # mid-sentence case ("私はこのように考えています") where the phrase
    # is never at the very start.
    words = _words(
        ("私", 0.0, 0.2),
        ("は", 0.2, 0.3),
        ("このように", 0.3, 0.8),
        ("考えています", 0.8, 1.5),
    )
    seg = TranscriptSegment(id=0, start=0.0, end=1.5, text="私はこのように考えています", words=words)
    assert find_opening_trim_point(seg) is None


def test_find_opening_trim_point_returns_none_without_word_timestamps():
    seg = TranscriptSegment(id=0, start=0.0, end=1.0, text="このように弱点を直すと", words=[])
    assert find_opening_trim_point(seg) is None


def test_find_opening_trim_point_returns_none_when_prefix_is_last_word():
    # A weak lead-in phrase with nothing after it in the segment -- there
    # is no word to move the start to, so no trim point is returned.
    words = _words(("このように", 0.0, 0.55))
    seg = TranscriptSegment(id=0, start=0.0, end=0.55, text="このように", words=words)
    assert find_opening_trim_point(seg) is None
