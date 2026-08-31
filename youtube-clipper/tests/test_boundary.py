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


# --- start_anchor_text: mid-segment start at a real word boundary --------


def _transcript_candidate1():
    # Real-machine example 1: "これも私の愛車である86はスープラを..." -- the
    # self-introduction lead-in is weak, "86は..." is the real content.
    return Transcript(
        video_id="vid1",
        language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=3.0,
                text="これも私の愛車である86はスープラをベースに作られています",
                words=[
                    _word(0.0, 0.3, "これも"),
                    _word(0.3, 0.6, "私の"),
                    _word(0.6, 0.9, "愛車である"),
                    _word(0.9, 1.2, "86は"),
                    _word(1.2, 3.0, "スープラをベースに作られています"),
                ],
            ),
        ],
    )


def test_resolve_candidate_trims_via_start_anchor_text(monkeypatch):
    # A: candidate 1 fixture -- anchor="86は" -> resolved opening starts
    # with "86は...", the self-intro lead-in is gone.
    monkeypatch.setattr(config, "BOUNDARY_PADDING_MS", 200)
    transcript = _transcript_candidate1()
    raw = RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0, start_anchor_text="86は")
        ],
        hook_text="h", opening_hook_strength=85, title="t", description="d",
        score=85, reasoning="r", caveats="",
    )
    candidate = boundary.resolve_candidate(raw, transcript, candidate_id="c1")
    resolved = candidate.segments[0]

    assert resolved.text.startswith("86は")
    assert "これも私の愛車である" not in resolved.text
    assert resolved.start == max(0.9 - 0.2, 0.0)


def _transcript_candidate2():
    # Real-machine example 2: "よくある話が、私も乗っているZN6-86であったり、
    # あとはBRZ、あとGR86、メタルクラッチを入れるとミッションが壊れやすくなる
    # っていうのはよく言われてます" -- "よくある話が" is the weak lead-in.
    return Transcript(
        video_id="vid1",
        language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=5.0,
                text="よくある話が私も乗っているZN6-86であったりあとはBRZあとGR86メタルクラッチを入れるとミッションが壊れやすくなるっていうのはよく言われてます",
                words=[
                    _word(0.0, 0.4, "よくある話が"),
                    _word(0.4, 0.8, "私も乗っている"),
                    _word(0.8, 1.2, "ZN6-86であったり"),
                    _word(1.2, 5.0, "あとはBRZあとGR86メタルクラッチを入れるとミッションが壊れやすくなるっていうのはよく言われてます"),
                ],
            ),
        ],
    )


def test_resolve_candidate_trims_self_reference_lead_in_via_anchor(monkeypatch):
    # B: candidate 2 fixture -- anchor drops "よくある話が私も乗っている" and
    # starts at "ZN6-86であったり...".
    monkeypatch.setattr(config, "BOUNDARY_PADDING_MS", 200)
    transcript = _transcript_candidate2()
    raw = RawClipCandidate(
        hook_type="surprising_fact",
        segments=[
            RawUsedSegment(
                role="hook", start_segment_id=0, end_segment_id=0,
                start_anchor_text="ZN6-86であったり",
            )
        ],
        hook_text="h", opening_hook_strength=85, title="t", description="d",
        score=85, reasoning="r", caveats="",
    )
    candidate = boundary.resolve_candidate(raw, transcript, candidate_id="c1")
    resolved = candidate.segments[0]

    assert resolved.text.startswith("ZN6-86であったり")
    assert "よくある話が" not in resolved.text


def test_resolve_candidate_reorders_segments_conclusion_first(monkeypatch):
    # C: candidate 3 fixture -- hook (conclusion, chronologically later) is
    # placed first, context (chronologically earlier) placed second. The
    # resolved/render order must be exactly the given order, not sorted by
    # transcript-chronological start time.
    monkeypatch.setattr(config, "BOUNDARY_PADDING_MS", 200)
    transcript = Transcript(
        video_id="vid1",
        language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=2.0,
                text="真冬のサーキットで2、3周しかアタックをしません",
                words=[_word(0.0, 2.0, "真冬のサーキットで2、3周しかアタックをしません")],
            ),
            TranscriptSegment(
                id=1, start=2.5, end=4.5,
                text="車を冷やしますっていうのであれば",
                words=[_word(2.5, 4.5, "車を冷やしますっていうのであれば")],
            ),
            TranscriptSegment(
                id=2, start=5.0, end=8.0,
                text="冷却効率を上げるために重量を増やすというのはアンチパターンになるかなと思います",
                words=[
                    _word(5.0, 6.0, "冷却効率を"),
                    _word(6.0, 6.5, "上げるために"),
                    _word(6.5, 8.0, "重量を増やすというのはアンチパターンになるかなと思います"),
                ],
            ),
        ],
    )
    raw = RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(
                role="hook", start_segment_id=2, end_segment_id=2,
                start_anchor_text="冷却効率を",
            ),
            RawUsedSegment(role="context", start_segment_id=0, end_segment_id=1),
        ],
        hook_text="h", opening_hook_strength=90, title="t", description="d",
        score=90, reasoning="r", caveats="",
    )
    candidate = boundary.resolve_candidate(raw, transcript, candidate_id="c1")

    assert [s.role for s in candidate.segments] == ["hook", "context"]
    assert candidate.segments[0].text.startswith("冷却効率を")
    assert "真冬のサーキット" in candidate.segments[1].text
    # the hook (segment id 2) chronologically starts *after* the context
    # (segment ids 0-1), confirming this is a genuine reorder, not
    # accidentally still-chronological.
    assert candidate.segments[0].start > candidate.segments[1].start


def test_start_anchor_text_not_found_falls_back_to_segment_start(monkeypatch):
    # F: anchor text that doesn't exist in the transcript must never cause
    # a guessed/approximate cut -- falls back to the segment's own start.
    monkeypatch.setattr(config, "BOUNDARY_PADDING_MS", 200)
    transcript = _transcript_candidate1()
    raw = RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(
                role="hook", start_segment_id=0, end_segment_id=0,
                start_anchor_text="存在しない発話テキスト",
            )
        ],
        hook_text="h", opening_hook_strength=85, title="t", description="d",
        score=85, reasoning="r", caveats="",
    )
    candidate = boundary.resolve_candidate(raw, transcript, candidate_id="c1")
    resolved = candidate.segments[0]

    assert resolved.text.startswith("これも私の愛車である")
    assert resolved.start == max(0.0 - 0.2, 0.0)


def test_start_anchor_text_without_word_timestamps_falls_back_safely():
    # G: no word-timestamp data at all -- never guess a cut point, even
    # with an anchor supplied.
    transcript = Transcript(
        video_id="vid1",
        language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=3.0,
                text="これも私の愛車である86はスープラです", words=[],
            ),
        ],
    )
    raw = RawClipCandidate(
        hook_type="strong_take",
        segments=[
            RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0, start_anchor_text="86は")
        ],
        hook_text="h", opening_hook_strength=85, title="t", description="d",
        score=85, reasoning="r", caveats="",
    )
    candidate = boundary.resolve_candidate(raw, transcript, candidate_id="c1")
    assert candidate.segments[0].text == "これも私の愛車である86はスープラです"


def test_start_anchor_text_collapsing_segment_to_empty_falls_back(monkeypatch):
    # H: a pathologically tight next-segment gap can clamp this segment's
    # resolved end (used.end) to well before the anchor word's own start
    # (e.g. overlapping/misaligned Whisper timestamps) -- trimming to the
    # anchor would then collapse or invert the used range. That trim must
    # be rejected, falling back to the untrimmed segment.
    monkeypatch.setattr(config, "BOUNDARY_PADDING_MS", 0)
    transcript = Transcript(
        video_id="vid1",
        language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=1.0, text="A B",
                words=[_word(0.0, 0.5, "A"), _word(0.5, 1.0, "B")],
            ),
            # Starts *before* segment 0's anchor word ("B") even begins --
            # clamps used.end for segment 0 down to 0.4.
            TranscriptSegment(id=1, start=0.4, end=1.5, text="C", words=[_word(0.4, 1.5, "C")]),
        ],
    )
    raw = RawClipCandidate(
        hook_type="story",
        segments=[RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0, start_anchor_text="B")],
        hook_text="h", opening_hook_strength=85, title="t", description="d",
        score=85, reasoning="r", caveats="",
    )
    candidate = boundary.resolve_candidate(raw, transcript, candidate_id="c1")
    resolved = candidate.segments[0]
    # Falls back to the untrimmed resolved segment (start=0.0, end=0.4),
    # never an inverted/empty range, and never silently drops "B".
    assert resolved.start < resolved.end
    assert resolved.start == 0.0
    assert "B" in resolved.text
