from podcast_clipper import boundary, config
from podcast_clipper.models import (
    RawClipCandidate,
    RawUsedSegment,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)


def _word(start, end, text="w"):
    return TranscriptWord(start=start, end=end, text=text)


def _transcript():
    # 3 segments with clear gaps between them, each with word timestamps.
    return Transcript(
        video_id="vid1",
        language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=10.0, end=12.0, text="segment zero",
                words=[_word(10.0, 10.5), _word(10.6, 12.0)],
            ),
            TranscriptSegment(
                id=1, start=13.0, end=16.0, text="segment one",
                words=[_word(13.0, 14.0), _word(14.2, 16.0)],
            ),
            TranscriptSegment(
                id=2, start=17.0, end=20.0, text="segment two",
                words=[_word(17.0, 18.0), _word(18.5, 20.0)],
            ),
        ],
    )


def test_resolve_segment_applies_padding_within_gap(monkeypatch):
    monkeypatch.setattr(config, "BOUNDARY_PADDING_MS", 200)
    transcript = _transcript()
    raw = RawUsedSegment(role="hook", start_segment_id=1, end_segment_id=1)
    resolved = boundary.resolve_segment(raw, transcript)

    # word_start=13.0, padding=0.2 -> 12.8; prev_end=12.0 -> max(12.8, 12.0) = 12.8
    assert resolved.start == 12.8
    # word_end=16.0, padding=0.2 -> 16.2; next_start=17.0 -> min(16.2, 17.0) = 16.2
    assert resolved.end == 16.2


def test_resolve_segment_does_not_cross_into_neighbour(monkeypatch):
    # Tight gap: padding would overshoot into the previous segment's speech.
    monkeypatch.setattr(config, "BOUNDARY_PADDING_MS", 5000)  # 5s, deliberately huge
    transcript = _transcript()
    raw = RawUsedSegment(role="answer", start_segment_id=1, end_segment_id=1)
    resolved = boundary.resolve_segment(raw, transcript)

    assert resolved.start >= transcript.segments[0].end  # never before prev segment's end
    assert resolved.end <= transcript.segments[2].start  # never past next segment's start
    assert resolved.start < resolved.end


def test_resolve_segment_multi_segment_range_spans_start_to_end():
    transcript = _transcript()
    raw = RawUsedSegment(role="answer", start_segment_id=0, end_segment_id=2)
    resolved = boundary.resolve_segment(raw, transcript)

    assert "segment zero" in resolved.text
    assert "segment two" in resolved.text
    assert resolved.start < transcript.segments[0].words[0].start + 1
    assert resolved.end > transcript.segments[2].words[-1].end - 1


def test_resolve_candidate_resolves_all_segments_in_order():
    transcript = _transcript()
    raw = RawClipCandidate(
        hook_type="story",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0),
            RawUsedSegment(role="answer", start_segment_id=2, end_segment_id=2),
        ],
        hook_text="h", cta_end_text="c", title="t", description="d",
        score=80, reasoning="r", caveats="",
    )
    candidate = boundary.resolve_candidate(raw, transcript, candidate_id="c1")
    assert candidate.id == "c1"
    assert len(candidate.segments) == 2
    assert candidate.segments[0].role == "hook"
    assert candidate.segments[1].role == "answer"
    # boundary.py must not decide "is this mid-sentence" itself -- it should
    # simply resolve whatever range Claude selected without judging it.
    assert candidate.total_duration > 0
