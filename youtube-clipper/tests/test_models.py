import pytest

from podcast_clipper.models import (
    ClipCandidate,
    RawClipCandidate,
    RawUsedSegment,
    UsedSegment,
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
