"""Resolve Claude's semantic segment selection into actual edit points.

Plan fix #1: this module does *not* decide what to include — that is a
semantic judgement Claude already made in clip_selector.py by choosing
which transcript segment IDs belong to the clip. This module only nudges
those chosen boundaries to a natural-sounding *audio* edit point (a word
boundary, with a small safety margin into the surrounding silence) so the
cut doesn't land mid-syllable. It must never widen or narrow the semantic
range Claude selected beyond that small, fixed padding.

The one further adjustment this module makes -- skipping a known weak
lead-in phrase ("このように", "えー", ...) at the very front of a
candidate's opening (see _apply_opening_trim) -- is the same category of
change as the padding above: a fixed, narrow, non-semantic normalization
against a known closed list (models.WEAK_OPENING_PREFIXES), never a
judgement about which content or topic Claude selected.

This function is deterministic (pure function of the raw candidate + the
transcript), which qa.py relies on: it recomputes the same resolution to
verify render.py actually used these exact edit points (see qa.py's
"edit boundary integrity" check). Because resolve_candidate is the single
place both the opening trim and the ending logic converge, the UI
("冒頭の実音声"), render.py, and qa.py's self-consistency checks all see
the identical corrected result without any of them needing their own copy
of this logic.
"""
from __future__ import annotations

from dataclasses import replace

from . import config, models
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


def _apply_opening_trim(
    used: UsedSegment, raw_used: RawUsedSegment, transcript: Transcript
) -> UsedSegment:
    """Mechanically skips a known weak lead-in phrase at the very start of
    the candidate (see module docstring) -- returns `used` unchanged if
    there's nothing to trim, no word-timestamp data, or trimming would
    collapse the segment to empty.
    """
    first_seg = transcript.segment_by_id(raw_used.start_segment_id)
    trim_word = models.find_opening_trim_point(first_seg)
    if trim_word is None:
        return used

    padding = config.BOUNDARY_PADDING_MS / 1000.0
    new_start = max(trim_word.start - padding, used.start)
    if new_start >= used.end:
        return used

    # used.text begins with first_seg.text verbatim (resolve_segment joins
    # segments in order with no leading text before the first one), so the
    # lead-in phrase's word-length span can be stripped directly off the
    # front of it.
    lead_in_len = sum(len(w.text) for w in first_seg.words if w.start < trim_word.start)
    trimmed_text = used.text[lead_in_len:].lstrip()
    return replace(used, start=new_start, text=trimmed_text or used.text)


def resolve_candidate(
    raw: RawClipCandidate, transcript: Transcript, candidate_id: str
) -> ClipCandidate:
    segments = [resolve_segment(rs, transcript) for rs in raw.segments]
    if segments:
        segments[0] = _apply_opening_trim(segments[0], raw.segments[0], transcript)
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
