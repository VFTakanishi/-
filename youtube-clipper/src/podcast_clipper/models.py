"""Data models shared across the pipeline stages.

Two "layers" of models exist on purpose:

- Raw* models hold what Claude decides: a *semantic* range, expressed as
  references to transcript segment IDs (never raw seconds). This is the
  "which utterances make up this clip" decision (see clip_selector.py).
- The resolved models (UsedSegment / ClipCandidate) hold actual seconds,
  produced by boundary.py from the Raw* models plus the transcript's word
  timestamps. boundary.py only nudges edit points to a natural audio
  boundary within the range Claude already chose; it does not re-decide
  what the range should be.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

HookType = Literal["open_loop", "strong_take", "surprising_fact", "story"]
SegmentRole = Literal["hook", "context", "answer", "payoff"]

_MAX_VALUE_REPR_LEN = 200


class MalformedCandidateError(TypeError):
    """Raised when a candidate/segment item from Claude's tool_use response
    (or from a cached candidate on disk) isn't the expected dict shape.

    Both clip_selector.py (parsing Claude's real response) and cache.py
    (parsing a cached candidate) share this so a plain `d["key"]` on an
    unexpectedly non-dict item never surfaces as a bare, undiagnosable
    "TypeError: string indices must be integers, not 'str'" -- the message
    instead carries exactly which item (stage/context, index) and what it
    actually was (type + a truncated repr).
    """


def describe_value(value: object) -> str:
    """A repr of `value` safe to embed in an error message: truncated so a
    very long/garbled string doesn't blow up the error text itself.
    """
    r = repr(value)
    if len(r) > _MAX_VALUE_REPR_LEN:
        r = r[:_MAX_VALUE_REPR_LEN] + "...(truncated)"
    return r


def require_dict(value: object, *, context: str) -> dict:
    """Validates `value` is a dict before the caller does `value["key"]` on
    it, raising MalformedCandidateError with `context` (e.g. "Stage1
    candidates[2]" or "Stage2 candidates[0].segments[1]") plus the actual
    type/value if not. Returns `value` unchanged when it is a dict.
    """
    if not isinstance(value, dict):
        raise MalformedCandidateError(
            f"{context}: expected a dict, got {type(value).__name__}: {describe_value(value)}"
        )
    return value


@dataclass
class TranscriptWord:
    start: float
    end: float
    text: str


@dataclass
class TranscriptSegment:
    id: int
    start: float
    end: float
    text: str
    words: list[TranscriptWord] = field(default_factory=list)


@dataclass
class Transcript:
    video_id: str
    language: str
    segments: list[TranscriptSegment] = field(default_factory=list)

    def segment_by_id(self, segment_id: int) -> TranscriptSegment:
        for seg in self.segments:
            if seg.id == segment_id:
                return seg
        raise KeyError(f"no transcript segment with id={segment_id}")

    def segment_index(self, segment_id: int) -> int:
        for i, seg in enumerate(self.segments):
            if seg.id == segment_id:
                return i
        raise KeyError(f"no transcript segment with id={segment_id}")


@dataclass
class RawUsedSegment:
    """A semantic range Claude selected, referencing transcript segment IDs."""

    role: SegmentRole
    start_segment_id: int
    end_segment_id: int  # inclusive


@dataclass
class RawClipCandidate:
    """Claude's Stage2 output for one candidate: semantic selection only."""

    hook_type: HookType
    segments: list[RawUsedSegment]
    hook_text: str
    opening_hook_strength: int
    title: str
    description: str
    score: int
    reasoning: str
    caveats: str

    def __post_init__(self) -> None:
        if not (1 <= len(self.segments) <= 3):
            raise ValueError(
                f"segments must contain 1-3 entries (got {len(self.segments)})"
            )
        if not (0 <= self.score <= 100):
            raise ValueError(f"score must be within 0-100 (got {self.score})")
        if not (0 <= self.opening_hook_strength <= 100):
            raise ValueError(
                f"opening_hook_strength must be within 0-100 (got {self.opening_hook_strength})"
            )


@dataclass
class UsedSegment:
    """A resolved (actual-seconds) edit range, after boundary.py correction."""

    role: SegmentRole
    start: float
    end: float
    text: str

    def __post_init__(self) -> None:
        if not (self.start < self.end):
            raise ValueError(f"start must be < end (got {self.start}, {self.end})")

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class ClipCandidate:
    """One proposed short clip with resolved (actual-seconds) edit points.

    `segments` is 1-3 entries; the default shape is 2 (hook + answer), with
    any filler/tangent between them cut out. This is never a single
    contiguous [start, end] span by design (absolute condition #5).
    """

    id: str
    hook_type: HookType
    segments: list[UsedSegment]
    hook_text: str
    opening_hook_strength: int
    title: str
    description: str
    score: int
    reasoning: str
    caveats: str

    def __post_init__(self) -> None:
        if not (1 <= len(self.segments) <= 3):
            raise ValueError(
                f"segments must contain 1-3 entries (got {len(self.segments)})"
            )
        if not (0 <= self.score <= 100):
            raise ValueError(f"score must be within 0-100 (got {self.score})")
        if not (0 <= self.opening_hook_strength <= 100):
            raise ValueError(
                f"opening_hook_strength must be within 0-100 (got {self.opening_hook_strength})"
            )

    @property
    def total_duration(self) -> float:
        return sum(seg.duration for seg in self.segments)


@dataclass
class RenderManifest:
    """Record of exactly what render.py did for one candidate, so qa.py can
    verify the final mp4 matches the boundary-resolved edit points it was
    supposed to use (the "edit boundary integrity" / "speech alignment"
    checks are both self-consistency checks against this manifest, not
    fresh audio analysis).
    """

    video_id: str
    candidate_id: str
    segments: list[UsedSegment]
    hook_text: str
    watermark_text: str
    total_duration: float
    intermediate_video_path: str
    final_video_path: str
