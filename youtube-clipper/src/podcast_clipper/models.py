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


# Known weak lead-in phrases that may appear at the very start of a
# candidate's opening. Used two ways: clip_selector.py's
# _looks_like_weak_opening (reject a candidate whose opening still looks
# weak after trimming) and boundary.py's opening-trim (mechanically skip
# past one of these at the very front of a candidate's first segment,
# never a mid-sentence occurrence). Defined once here so both stay
# consistent.
WEAK_OPENING_PREFIXES = (
    "今回は", "今日は", "ということで", "えー", "えーと", "えっと", "あの", "まあ", "さて", "このように",
)

# Deictic/anaphoric openings: unlike WEAK_OPENING_PREFIXES (filler that can
# simply be skipped past), these words carry a referent the viewer needs
# ("これ"/"それ" point at *something*) -- skipping past them doesn't fix a
# candidate whose opening depends on them, since the thing they refer to
# may not even be in the clip. Used only to *detect* (never mechanically
# trim) a context-dependent opening; clip_selector.py rejects a candidate
# whose resolved opening still starts with one of these after any
# start_anchor_text trim, rather than guessing a substitute.
CONTEXT_DEPENDENT_OPENING_PREFIXES = (
    "これの", "これ", "それ", "この", "その", "こういう", "こういった",
    "なので", "だから", "それで", "ということで", "その場合",
)


def find_opening_trim_point(segment: TranscriptSegment) -> TranscriptWord | None:
    """If segment's word sequence begins with one of WEAK_OPENING_PREFIXES
    (matched only at the very start of the segment -- never a
    mid-sentence occurrence, since matching stops as soon as the
    accumulated text is no longer a prefix of any known phrase), returns
    the first word after that phrase so playback can start there instead.
    Returns None if there's no word-timestamp data at all (never guess a
    cut point without it) or no known prefix matches.
    """
    if not segment.words:
        return None
    accumulated = ""
    for i, word in enumerate(segment.words):
        accumulated += word.text
        if accumulated in WEAK_OPENING_PREFIXES:
            return segment.words[i + 1] if i + 1 < len(segment.words) else None
        if not any(prefix.startswith(accumulated) for prefix in WEAK_OPENING_PREFIXES):
            return None
    return None


# Self-narrative asides ("私も乗っている...") and generic preamble phrases
# ("よくある話が...") -- unlike WEAK_OPENING_PREFIXES (pure filler with no
# informational content), these carry a little more, but the point that
# follows never depends on them, so -- unlike CONTEXT_DEPENDENT_OPENING_
# PREFIXES -- skipping past them is safe. Repair-only (clip_selector.py's
# repair_local_candidate): never applied by boundary.py's automatic
# per-render trim, since deciding "this aside is safe to drop" is closer
# to a semantic judgement than the closed, purely-mechanical
# WEAK_OPENING_PREFIXES list.
SELF_REFERENCE_OPENING_PREFIXES = (
    "これも私の愛車である", "私も乗っている", "私の場合は", "私の車では", "私が思うに", "よくある話が",
)

# Combined catalog find_sequential_removable_prefix_word chains through --
# repair-only, kept separate from WEAK_OPENING_PREFIXES (which stays the
# list boundary.py's automatic trim uses) so widening this list can never
# silently change what render.py/UI trim on every single candidate.
_REPAIR_REMOVABLE_OPENING_PREFIXES = WEAK_OPENING_PREFIXES + SELF_REFERENCE_OPENING_PREFIXES

# Repair-only bound: prevents find_sequential_removable_prefix_word from
# ever looping unboundedly over a pathological word sequence.
_MAX_SEQUENTIAL_REMOVABLE_PREFIX_TRIMS = 5


def find_sequential_removable_prefix_word(segment: TranscriptSegment) -> TranscriptWord | None:
    """Like find_opening_trim_point, but chains multiple known-removable
    phrases (WEAK_OPENING_PREFIXES then SELF_REFERENCE_OPENING_PREFIXES,
    combined) from the very front of the segment, one after another --
    e.g. "よくある話が" then "私も乗っている" then landing on "ZN6-86で
    あったり..." -- up to _MAX_SEQUENTIAL_REMOVABLE_PREFIX_TRIMS trims, so a
    stack of several weak/self-referential lead-ins can be cleared in one
    pass instead of only ever recognizing the first one.

    Repair-only (see SELF_REFERENCE_OPENING_PREFIXES) -- never used by
    boundary.py's automatic per-render trim. Returns None if there's no
    word-timestamp data, no known prefix matches at all, or trimming would
    consume the entire segment (nothing left to start from).
    """
    if not segment.words:
        return None
    words = segment.words
    idx = 0
    for _ in range(_MAX_SEQUENTIAL_REMOVABLE_PREFIX_TRIMS):
        accumulated = ""
        matched_at: int | None = None
        for i in range(idx, len(words)):
            accumulated += words[i].text
            if accumulated in _REPAIR_REMOVABLE_OPENING_PREFIXES:
                matched_at = i + 1
                break
            if not any(p.startswith(accumulated) for p in _REPAIR_REMOVABLE_OPENING_PREFIXES):
                break
        if matched_at is None:
            break
        idx = matched_at
        if idx >= len(words):
            return None  # trimmed the entire segment away -- nothing left
    return words[idx] if idx > 0 else None


def find_anchor_start_word(segment: TranscriptSegment, anchor_text: str) -> TranscriptWord | None:
    """Locates an AI-chosen start_anchor_text (e.g. "86は" within a segment
    whose full text is "これも私の愛車である86はスープラを...") as an exact,
    contiguous substring of the segment's word sequence, aligned to a real
    word boundary, and returns the word it begins on -- so a candidate can
    start mid-segment at a natural phrase/clause boundary instead of only
    ever using the segment's own first word (see find_opening_trim_point,
    which only ever recognizes a small fixed list of lead-in phrases).

    Never fuzzy-matches and never guesses: returns None if anchor_text is
    falsy, the segment has no word-timestamp data, the exact text doesn't
    appear in the segment at all, or the match's start falls in the
    middle of a word (a genuine mid-word start is never allowed, even
    though a mid-*segment*, phrase-boundary start now is). The caller
    (boundary.py) must treat None as "don't trim" and fall back to the
    segment's own start -- never approximate a cut point.
    """
    if not segment.words or not anchor_text:
        return None
    concatenated = ""
    word_start_offsets: list[int] = []
    for word in segment.words:
        word_start_offsets.append(len(concatenated))
        concatenated += word.text
    idx = concatenated.find(anchor_text)
    if idx == -1:
        return None
    try:
        return segment.words[word_start_offsets.index(idx)]
    except ValueError:
        # anchor_text was found, but not starting exactly on a word
        # boundary -- that would be a mid-word start, which is forbidden.
        return None


def resolve_segment_start_word(
    segment: TranscriptSegment, start_anchor_text: str | None
) -> TranscriptWord | None:
    """Decides which real word a segment's playback should actually start
    from. If start_anchor_text is set, trusts it exclusively: an exact,
    word-boundary-aligned match (find_anchor_start_word) is used, and an
    invalid/not-found anchor falls straight back to "no trim" (None) --
    never silently substituting the unrelated fixed-prefix heuristic below
    for a trim Claude explicitly chose not to get. If no anchor was given
    at all, falls back to the fixed WEAK_OPENING_PREFIXES lead-in trim
    (find_opening_trim_point) exactly as before, so candidates that don't
    use anchors keep their old behavior unchanged.
    """
    if start_anchor_text:
        return find_anchor_start_word(segment, start_anchor_text)
    return find_opening_trim_point(segment)


@dataclass
class RawUsedSegment:
    """A semantic range Claude selected, referencing transcript segment IDs.

    start_anchor_text is optional: when set, it's a short substring Claude
    asserts exists verbatim, contiguously, at a real word boundary near
    the start of the start_segment_id transcript segment (e.g. "86は"
    within "これも私の愛車である86はスープラを..."), letting playback begin
    mid-segment at a natural phrase boundary instead of always using the
    segment's literal first word. It is never AI-authored replacement
    text -- boundary.py verifies it against the real transcript
    (models.find_anchor_start_word) and falls back to "no trim" (the
    segment's own start) if it doesn't match exactly.
    """

    role: SegmentRole
    start_segment_id: int
    end_segment_id: int  # inclusive
    start_anchor_text: str | None = None


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
