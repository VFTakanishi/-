"""Resolve Claude's semantic segment selection into actual edit points.

Plan fix #1: this module does *not* decide what to include — that is a
semantic judgement Claude already made in clip_selector.py by choosing
which transcript segment IDs belong to the clip. This module only nudges
those chosen boundaries to a natural-sounding *audio* edit point (a word
boundary, with a small safety margin into the surrounding silence) so the
cut doesn't land mid-syllable. It must never widen or narrow the semantic
range Claude selected beyond that small, fixed padding.

This function is deterministic (pure function of the raw candidate + the
transcript), which qa.py relies on: it recomputes the same resolution to
verify render.py actually used these exact edit points (see qa.py's
"edit boundary integrity" check).
"""
from __future__ import annotations

from . import config
from .models import ClipCandidate, RawClipCandidate, RawUsedSegment, Transcript, UsedSegment


def resolve_segment(raw_seg: RawUsedSegment, transcript: Transcript) -> UsedSegment:
    start_idx = transcript.segment_index(raw_seg.start_segment_id)
    end_idx = transcript.segment_index(raw_seg.end_segment_id)
    if end_idx < start_idx:
        raise ValueError(
            f"end_segment_id ({raw_seg.end_segment_id}) precedes "
            f"start_segment_id ({raw_seg.start_segment_id})"
        )

    start_seg = transcript.segments[start_idx]
    end_seg = transcript.segments[end_idx]
    padding = config.BOUNDARY_PADDING_MS / 1000.0

    word_start = start_seg.words[0].start if start_seg.words else start_seg.start
    prev_end = transcript.segments[start_idx - 1].end if start_idx > 0 else 0.0
    start_time = max(word_start - padding, prev_end)

    word_end = end_seg.words[-1].end if end_seg.words else end_seg.end
    next_start = (
        transcript.segments[end_idx + 1].start
        if end_idx + 1 < len(transcript.segments)
        else word_end + padding
    )
    end_time = min(word_end + padding, next_start)

    if start_time >= end_time:
        # Padding collided (e.g. a very short segment sandwiched between
        # neighbours with almost no gap). Fall back to the unpadded word
        # times rather than producing an invalid/inverted range.
        start_time, end_time = word_start, word_end

    text = " ".join(
        transcript.segments[i].text for i in range(start_idx, end_idx + 1)
    )
    return UsedSegment(role=raw_seg.role, start=start_time, end=end_time, text=text)


def resolve_candidate(
    raw: RawClipCandidate, transcript: Transcript, candidate_id: str
) -> ClipCandidate:
    segments = [resolve_segment(rs, transcript) for rs in raw.segments]
    return ClipCandidate(
        id=candidate_id,
        hook_type=raw.hook_type,
        segments=segments,
        hook_text=raw.hook_text,
        opening_hook_strength=raw.opening_hook_strength,
        title=raw.title,
        description=raw.description,
        score=raw.score,
        reasoning=raw.reasoning,
        caveats=raw.caveats,
    )
