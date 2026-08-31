"""Resolve Claude's semantic segment selection into actual edit points.

Plan fix #1: this module does *not* decide what to include — that is a
semantic judgement Claude already made in clip_selector.py by choosing
which transcript segment IDs belong to the clip. This module only nudges
those chosen boundaries to a natural-sounding *audio* edit point (a word
boundary, with a small safety margin into the surrounding silence) so the
cut doesn't land mid-syllable. It must never widen or narrow the semantic
range Claude selected beyond that small, fixed padding.

The one further adjustment this module makes -- moving a segment's start
past a weak lead-in to a real, verified word boundary (see
_apply_start_trim) -- is the same category of change as the padding
above: a narrow, mechanical normalization, never a judgement about which
content or topic Claude selected. Two sources feed it: a fixed closed list
of known lead-in phrases ("このように", "えー", ...,
models.WEAK_OPENING_PREFIXES), or -- when Claude supplied one for a given
segment -- an AI-chosen start_anchor_text, which is only ever trusted
after models.find_anchor_start_word verifies it's an exact, contiguous,
word-boundary-aligned substring of that segment's real transcript text.
Neither ever invents or rewrites words; both only ever move the start
point later within words Claude (or Whisper) already produced. Segment
*order* itself (which segment plays first, second, ...) is entirely
clip_selector.py's/Claude's choice, expressed as the order of
raw.segments -- this module resolves each segment independently and never
reorders the list, so a candidate whose segments are not in chronological
transcript order (a stronger later utterance placed first as the hook) is
resolved, rendered, and QA'd in that same given order throughout.

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


def _apply_start_trim(
    used: UsedSegment, raw_used: RawUsedSegment, transcript: Transcript
) -> UsedSegment:
    """Mechanically moves this segment's start to whichever real word
    models.resolve_segment_start_word decides: an AI-chosen
    start_anchor_text (an exact, word-boundary-aligned match within the
    start_segment_id transcript segment), or -- only when no anchor was
    given at all -- the fixed WEAK_OPENING_PREFIXES lead-in trim (see
    module docstring). Applied to every segment in a candidate, not just
    the first, since a mid-candidate segment reached after a reorder/jump
    can equally start with a weak lead-in or benefit from an anchor.
    Returns `used` unchanged if there's nothing to trim, no word-timestamp
    data, or trimming would collapse the segment to empty -- this never
    guesses an approximate cut point.
    """
    start_seg = transcript.segment_by_id(raw_used.start_segment_id)
    trim_word = models.resolve_segment_start_word(start_seg, raw_used.start_anchor_text)
    if trim_word is None:
        return used

    padding = config.BOUNDARY_PADDING_MS / 1000.0
    new_start = max(trim_word.start - padding, used.start)
    if new_start >= used.end:
        return used

    # used.text begins with start_seg.text verbatim (resolve_segment joins
    # segments in order with no leading text before the first one), so the
    # lead-in span's word-length can be stripped directly off the front of
    # it.
    lead_in_len = sum(len(w.text) for w in start_seg.words if w.start < trim_word.start)
    trimmed_text = used.text[lead_in_len:].lstrip()
    return replace(used, start=new_start, text=trimmed_text or used.text)


def resolve_candidate(
    raw: RawClipCandidate, transcript: Transcript, candidate_id: str
) -> ClipCandidate:
    segments = [resolve_segment(rs, transcript) for rs in raw.segments]
    segments = [
        _apply_start_trim(seg, raw_used, transcript)
        for seg, raw_used in zip(segments, raw.segments)
    ]
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
