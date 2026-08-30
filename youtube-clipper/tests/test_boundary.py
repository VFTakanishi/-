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
        hook_text="h", opening_hook_strength=80, title="t", description="d",
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


def _transcript_with_weak_opening():
    # segment 0's spoken opening is "このように弱点を直すと" -- a weak
    # lead-in phrase ("このように") followed by the real content
    # ("弱点を直すと"). This is a distinct, narrow, mechanical
    # normalization (see boundary.py's module docstring) -- not the same
    # thing as the ending-completeness "is this mid-sentence" judgement
    # tested above, which boundary.py still never makes.
    return Transcript(
        video_id="vid1",
        language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=1.30, text="このように弱点を直すと",
                words=[
                    _word(0.0, 0.55, "このように"),
                    _word(0.56, 0.90, "弱点を"),
                    _word(0.91, 1.30, "直すと"),
                ],
            ),
            TranscriptSegment(
                id=1, start=2.0, end=3.0, text="segment one",
                words=[_word(2.0, 2.5), _word(2.6, 3.0)],
            ),
        ],
    )


def test_resolve_candidate_trims_weak_opening_lead_in(monkeypatch):
    monkeypatch.setattr(config, "BOUNDARY_PADDING_MS", 200)
    transcript = _transcript_with_weak_opening()
    raw = RawClipCandidate(
        hook_type="story",
        segments=[RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=1)],
        hook_text="h", opening_hook_strength=80, title="t", description="d",
        score=80, reasoning="r", caveats="",
    )
    candidate = boundary.resolve_candidate(raw, transcript, candidate_id="c1")
    resolved = candidate.segments[0]

    # Start moves to the word-start of "弱点を" (0.56s) minus padding, not
    # a guessed/estimated second count.
    assert resolved.start == max(0.56 - 0.2, 0.0)
    assert resolved.text.startswith("弱点を")
    assert "このように" not in resolved.text


def test_resolve_candidate_does_not_trim_mid_sentence_occurrence(monkeypatch):
    # "このように" appears in the segment, but not at the very start
    # (word 0 is "私", never a prefix of any known lead-in phrase), so it
    # must never be mechanically trimmed -- mirrors the explicit example
    # "私はこのように考えています" must be left untouched.
    monkeypatch.setattr(config, "BOUNDARY_PADDING_MS", 200)
    transcript = Transcript(
        video_id="vid1",
        language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=1.5, text="私はこのように考えています",
                words=[
                    _word(0.0, 0.2, "私"),
                    _word(0.2, 0.3, "は"),
                    _word(0.3, 0.8, "このように"),
                    _word(0.8, 1.5, "考えています"),
                ],
            ),
        ],
    )
    raw = RawClipCandidate(
        hook_type="story",
        segments=[RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0)],
        hook_text="h", opening_hook_strength=80, title="t", description="d",
        score=80, reasoning="r", caveats="",
    )
    candidate = boundary.resolve_candidate(raw, transcript, candidate_id="c1")
    resolved = candidate.segments[0]
    assert "このように" in resolved.text
    assert resolved.text.startswith("私")


def test_resolve_candidate_no_trim_without_word_timestamps():
    # No word-timestamp data at all -- never guess a cut point, leave the
    # candidate uncorrected (safe fallback).
    transcript = Transcript(
        video_id="vid1",
        language="ja",
        segments=[
            TranscriptSegment(id=0, start=0.0, end=1.3, text="このように弱点を直すと", words=[]),
        ],
    )
    raw = RawClipCandidate(
        hook_type="story",
        segments=[RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0)],
        hook_text="h", opening_hook_strength=80, title="t", description="d",
        score=80, reasoning="r", caveats="",
    )
    candidate = boundary.resolve_candidate(raw, transcript, candidate_id="c1")
    assert candidate.segments[0].text == "このように弱点を直すと"
