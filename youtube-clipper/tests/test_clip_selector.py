from podcast_clipper import clip_selector, config
from podcast_clipper.models import (
    RawClipCandidate,
    RawUsedSegment,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)


def _segment(i, start):
    return TranscriptSegment(
        id=i, start=start, end=start + 2.0, text=f"segment {i}",
        words=[TranscriptWord(start=start, end=start + 2.0, text=f"seg{i}")],
    )


def _long_transcript(minutes=25):
    segments = [_segment(i, start=i * 20.0) for i in range(int(minutes * 60 / 20))]
    return Transcript(video_id="vid1", language="ja", segments=segments)


def test_build_chunks_covers_whole_transcript_with_overlap(monkeypatch):
    monkeypatch.setattr(config, "CHUNK_MINUTES", 10.0)
    monkeypatch.setattr(config, "CHUNK_OVERLAP_MINUTES", 1.0)
    transcript = _long_transcript(minutes=25)

    chunks = clip_selector._build_chunks(transcript.segments)

    assert len(chunks) >= 3  # 25 min / (10 min window, 1 min stride overlap) needs multiple chunks
    # every segment must be covered by at least one chunk
    covered_ids = {s.id for _, segs in chunks for s in segs}
    assert covered_ids == {s.id for s in transcript.segments}


def test_usable_segments_returns_all_when_op_exclusion_unset(monkeypatch):
    monkeypatch.setattr(config, "OP_EXCLUSION_SECONDS", None)
    transcript = _long_transcript(minutes=2)
    assert clip_selector._usable_segments(transcript) == transcript.segments


def test_usable_segments_excludes_only_when_explicitly_configured(monkeypatch):
    monkeypatch.setattr(config, "OP_EXCLUSION_SECONDS", 30.0)
    transcript = _long_transcript(minutes=2)
    usable = clip_selector._usable_segments(transcript)
    assert all(s.start >= 30.0 for s in usable)
    assert len(usable) < len(transcript.segments)


def _raw_candidate(start_id, end_id):
    return RawClipCandidate(
        hook_type="story",
        segments=[RawUsedSegment(role="hook", start_segment_id=start_id, end_segment_id=end_id)],
        hook_text="h", cta_end_text="c", title="t", description="d",
        score=80, reasoning="r", caveats="",
    )


def test_select_candidates_retries_once_on_out_of_range_duration(monkeypatch):
    transcript = _long_transcript(minutes=1)  # 3 short segments

    monkeypatch.setattr(config, "MAX_STAGE2_RETRIES", 1)
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 20.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 50.0)

    monkeypatch.setattr(
        clip_selector, "run_stage1",
        lambda t, title, force_refresh=False: [{"chunk_index": 0, "candidates": [_raw_candidate(0, 0)]}],
    )

    call_count = {"n": 0}

    def fake_rank_and_finalize(all_candidates, transcript, video_title, feedback=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # too short: 2s segment duration, way below 20s hard minimum
            return [_raw_candidate(0, 0), _raw_candidate(0, 0), _raw_candidate(0, 0)]
        # second call (after feedback) returns a longer, in-range candidate
        return [_raw_candidate(0, 2), _raw_candidate(0, 2), _raw_candidate(0, 2)]

    monkeypatch.setattr(clip_selector, "rank_and_finalize", fake_rank_and_finalize)

    result = clip_selector.select_candidates(transcript, "タイトル")

    assert call_count["n"] == 2  # exactly one retry, per config.MAX_STAGE2_RETRIES=1
    assert len(result) == 3


def test_select_candidates_caches_and_skips_recompute(monkeypatch):
    from podcast_clipper import cache

    transcript = _long_transcript(minutes=1)
    cache.save_stage2("vid1", [_raw_candidate(0, 0)] * 3)

    monkeypatch.setattr(
        clip_selector, "run_stage1", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run"))
    )

    result = clip_selector.select_candidates(transcript, "タイトル")
    assert len(result) == 3
