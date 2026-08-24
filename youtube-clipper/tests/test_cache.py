from podcast_clipper import cache
from podcast_clipper.models import (
    RawClipCandidate,
    RawUsedSegment,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)


def _transcript():
    return Transcript(
        video_id="vidA",
        language="ja",
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=2.0, text="hello",
                words=[TranscriptWord(start=0.0, end=1.0, text="he"), TranscriptWord(start=1.0, end=2.0, text="llo")],
            )
        ],
    )


def test_transcript_round_trip():
    original = _transcript()
    cache.save_transcript(original)
    loaded = cache.load_transcript("vidA")

    assert loaded is not None
    assert loaded.video_id == original.video_id
    assert loaded.segments[0].text == "hello"
    assert loaded.segments[0].words[1].text == "llo"


def test_transcript_missing_returns_none():
    assert cache.load_transcript("does-not-exist") is None


def test_stage1_round_trip():
    raw = RawClipCandidate(
        hook_type="open_loop",
        segments=[RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0)],
        hook_text="h", cta_end_text="c", title="t", description="d",
        score=70, reasoning="r", caveats="",
    )
    cache.save_stage1("vidA", [{"chunk_index": 0, "candidates": [raw]}])
    loaded = cache.load_stage1("vidA")

    assert loaded is not None
    assert loaded[0]["chunk_index"] == 0
    assert loaded[0]["candidates"][0].hook_type == "open_loop"


def test_stage2_round_trip():
    raw = RawClipCandidate(
        hook_type="strong_take",
        segments=[RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0)],
        hook_text="h", cta_end_text="c", title="t", description="d",
        score=90, reasoning="r", caveats="注意",
    )
    cache.save_stage2("vidA", [raw, raw, raw])
    loaded = cache.load_stage2("vidA")

    assert loaded is not None
    assert len(loaded) == 3
    assert loaded[0].caveats == "注意"
