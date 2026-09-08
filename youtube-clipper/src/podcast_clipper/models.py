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

# Stage1's own output shape (see RawMaterial below): a material is a single-
# purpose raw ingredient, not a structural position within a finished
# candidate (that's SegmentRole, assigned only once Stage2 designs a final
# candidate). "conclusion" folds into "hook" and "evidence" folds into
# "reason" -- five values, not the seven a first pass might suggest, to
# avoid over-splitting a schema that downstream code only ever branches on
# for "is this hook material or not" (see prompts/extract_candidates.md).
MaterialType = Literal["hook", "reason", "example", "context", "payoff"]

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
#
# Real-machine incident (candidate2, round 7): a repair that only trimmed
# filler off the *front* of the hook's own segment still landed on "あと
# GR86もそうだと思うんですけども..." -- grammatically fine Japanese, but
# "あと" ("also"/"additionally") explicitly continues an enumeration whose
# first item ("ZN6-86であったり...") was spoken earlier and is not in the
# clip, so the opening is just as context-dependent as "これの...". These
# enumeration-continuation words are included here (leading-position only,
# same as every other entry -- a mid-sentence "あと" is never matched) so
# a repair landing on one of them is rejected and a deeper repair (e.g.
# clip_selector's multi-segment prepend) is required instead of silently
# accepting a still-dependent opening.
CONTEXT_DEPENDENT_OPENING_PREFIXES = (
    "これの", "これ", "それ", "この", "その", "こういう", "こういった",
    "なので", "だから", "それで", "ということで", "その場合",
    "あと", "それから", "さらに",
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


# --- Speech fluency: word-search / self-correction / hesitation --------
#
# Real-machine incident: a candidate's hook and/or body text can be
# fluent-looking on paper (no filler prefix, no context-dependent
# pronoun) while still being the speaker visibly searching for words or
# correcting themselves mid-thought -- a genuinely separate quality axis
# from every check above, checked against the *whole* resolved candidate
# (every segment, not just the opening) and treated as a hard local veto:
# Claude's own opening_hook_strength self-rating must never override it
# (a real "opening_hook_strength=85" case was written against this).

# Meta-commentary about the act of speaking itself -- there is no natural,
# benign use of "表現が難しい" etc. the way there is for a bare "というか",
# so these are checked via plain substring containment anywhere in the
# text (not just the start) and treated as an unconditional signal.
WORD_SEARCH_MARKERS = (
    "表現が難しい", "説明が難しい", "言い方が難しい",
    "なんていうか", "なんて言うか", "なんていうんですかね",
    "どう言えばいいか", "どう説明すれば",
)

# Self-correction: "というか"/"というより" are extremely common, entirely
# benign connectives on their own, so a bare match on them is never
# enough. They only count as self-correction when a nearby reversal/
# negation word, or an immediately-following hesitation filler, confirms
# the speaker is actually taking back what they just said (e.g. "実は逆
# っていうのか" -- reversal marker "逆" appears *before* the connective;
# "というか、違って" -- reversal marker appears after; "というより、えー"
# -- trails straight into a hesitation filler).
_REFORMULATION_CONNECTIVES = ("というのか", "っていうのか", "というか", "っていうか", "というより")
_REVERSAL_MARKERS = ("じゃなくて", "ではなくて", "ではなく", "そうじゃな", "違って", "逆", "間違って", "訂正")
_ADJACENT_HESITATION_FILLERS = ("えー", "えっと", "あの", "その")
_PROXIMITY_WINDOW = 20  # chars on each side of the connective to search for a reversal marker
_ADJACENT_WINDOW = 6  # chars immediately after the connective a hesitation filler must fall within

# Standalone phrases unambiguous enough to need no nearby-connective check.
_STANDALONE_SELF_CORRECTION_MARKERS = ("そうじゃなくて", "そうじゃない", "じゃなくて、", "ではなくて、")


def has_speech_disfluency(text: str) -> str | None:
    """Returns the matched marker (for diagnostics) if `text` contains a
    word-search marker or a self-correction pattern, else None. This is
    the single detector shared by clip_selector.py's local quality gate
    and find_disfluent_hook_repair_point below, so the reject-check and
    the repair-attempt can never disagree about what counts as disfluent.
    """
    for marker in WORD_SEARCH_MARKERS:
        if marker in text:
            return marker
    for marker in _STANDALONE_SELF_CORRECTION_MARKERS:
        if marker in text:
            return marker
    for connective in _REFORMULATION_CONNECTIVES:
        idx = text.find(connective)
        if idx == -1:
            continue
        window = text[max(0, idx - _PROXIMITY_WINDOW): idx + len(connective) + _PROXIMITY_WINDOW]
        for marker in _REVERSAL_MARKERS:
            if marker in window:
                return f"{connective}+{marker}"
        adjacent = text[idx + len(connective): idx + len(connective) + _ADJACENT_WINDOW]
        for filler in _ADJACENT_HESITATION_FILLERS:
            if filler in adjacent:
                return f"{connective}+{filler}"
    return None


def _split_words_into_clauses(words: list[TranscriptWord]) -> list[tuple[int, int, str]]:
    """Groups a flat word list into comma-delimited clauses using real word
    text only (never estimating boundaries): each entry is
    (start_word_index, end_word_index_exclusive, clause_text_incl_comma).
    `words` may be the word list of a single TranscriptSegment, or a
    concatenation spanning several consecutive ones (see
    find_speech_restart_marker, which needs to compare clauses across a
    candidate's full segment range, not just within one original
    transcript segment).
    """
    clauses: list[tuple[int, int, str]] = []
    start = 0
    accumulated = ""
    for i, word in enumerate(words):
        accumulated += word.text
        if word.text.endswith(("、", "，", ",")) or i == len(words) - 1:
            clauses.append((start, i + 1, accumulated))
            start = i + 1
            accumulated = ""
    return clauses


def _split_into_clauses(segment: TranscriptSegment) -> list[tuple[int, int, str]]:
    """Groups segment.words into comma-delimited clauses -- see
    _split_words_into_clauses (this is a thin single-segment wrapper).
    """
    return _split_words_into_clauses(segment.words)


# A clause immediately following an already-disfluent one that ends in a
# bare continuative form (te-form etc.) is still grammatically glued to
# what was just skipped -- it can't stand alone as a hook opening even
# though it triggers no marker of its own (e.g. "間違っていて、" right
# after "実は逆っていうのか、"). Only applied to extend an already-started
# skip run (never to a clean clause 0 on its own), so this can only make
# the repair *more* conservative -- it never rejects a candidate outright,
# it only ever declines to repair one, leaving the veto check to decide.
_CONTINUATION_CLAUSE_SUFFIXES = ("て、", "で、", "し、", "くて、", "けど、", "ので、")


def find_disfluent_hook_repair_point(
    segment: TranscriptSegment, max_clauses_to_skip: int = 3
) -> TranscriptWord | None:
    """Bounded, deterministic repair for a hook whose disfluency is
    confined to a short preamble: reads the segment's words as
    comma-delimited clauses from the very start, and if the first (up to
    max_clauses_to_skip) clauses are each disfluent (has_speech_disfluency,
    a known _REPAIR_REMOVABLE_OPENING_PREFIXES phrase, or a bare
    continuative clause directly extending an already-disfluent run -- see
    _CONTINUATION_CLAUSE_SUFFIXES) and are followed by a clause that is
    itself clean AND whose remainder (all the way to the end of the
    segment) never re-triggers has_speech_disfluency, returns the first
    word of that clean remainder so playback can start there instead.

    Returns None (never guesses, never force-rescues) if: there's no
    word-timestamp data, the disfluent preamble exceeds
    max_clauses_to_skip, no clean remainder is ever reached within that
    budget, or the disfluency recurs later in the segment -- a candidate
    whose problem isn't confined to a skippable preamble is left for the
    caller to reject outright, not repaired into something it never
    actually said.
    """
    if not segment.words:
        return None
    clauses = _split_into_clauses(segment)
    if len(clauses) < 2:
        return None

    in_disfluent_run = False
    for idx, (start_idx, _end_idx, text) in enumerate(clauses):
        if idx >= max_clauses_to_skip:
            return None
        stripped = text.rstrip("、，,")
        is_disfluent_clause = (
            has_speech_disfluency(text) is not None
            or stripped in _REPAIR_REMOVABLE_OPENING_PREFIXES
            or (in_disfluent_run and stripped.endswith(_CONTINUATION_CLAUSE_SUFFIXES))
        )
        if is_disfluent_clause:
            in_disfluent_run = True
            continue
        if idx == 0:
            return None  # first clause is already clean -- nothing to skip
        remainder_text = "".join(c[2] for c in clauses[idx:])
        if has_speech_disfluency(remainder_text) is not None:
            return None  # disfluency recurs later -- don't force-rescue
        return segment.words[start_idx]
    return None  # every clause within the budget was disfluent -- unrepairable


# --- Speech restart: an unfinished clause abandoned and restarted -------
#
# Real-machine incident: "Nレンジで下るというのは、Nレンジにすると、エンジン
# ブレーキが効かなくなって..." -- has_speech_disfluency cannot catch this
# (no word-search marker, no "というか"-style self-correction connective).
# The actual signature is structural: a clause ends dangling on a bare
# topic/case particle (never completing a predicate) and the *very next*
# clause restates a shared, content-bearing phrase from it -- the speaker
# abandoning an unfinished construction and starting over with the same
# subject, rather than continuing it. Detection is purely text/timing
# based (TranscriptWord carries no confidence score -- faster-whisper's
# per-word probability is never captured -- so this can only ever be a
# deterministic text-structure heuristic, not an audio-confidence one).

# Separate from clip_selector._DANGLING_PARTICLE_ENDINGS (same values, by
# deliberate duplication rather than a shared import: that tuple is
# private to clip_selector's already-tested _looks_like_hook_incomplete_
# thought, and reusing it here isn't worth the risk of coupling two
# independent checks to one constant).
DANGLING_CLAUSE_PARTICLE_ENDINGS = ("は、", "が、", "を、", "に、", "で、", "と、", "も、")

_MIN_SHARED_CONTENT_PHRASE_LEN = 2


def _longest_common_substring(a: str, b: str) -> str:
    if not a or not b:
        return ""
    lengths = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    best_len = 0
    best_end_a = 0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                lengths[i][j] = lengths[i - 1][j - 1] + 1
                if lengths[i][j] > best_len:
                    best_len = lengths[i][j]
                    best_end_a = i
            else:
                lengths[i][j] = 0
    return a[best_end_a - best_len: best_end_a]


# U+3040-U+309F is the full Unicode Hiragana block (including the
# combining marks/small kana at its edges); anything outside it (kanji,
# katakana, alphanumerics, punctuation) counts as "not pure hiragana".
_HIRAGANA_RANGE = ("぀", "ゟ")


def _is_content_bearing(text: str) -> bool:
    """True if `text` contains at least one non-hiragana character (kanji,
    katakana, or alphanumeric). Japanese particles/connectives are almost
    always pure hiragana, while content words (nouns, technical terms,
    model names, etc.) almost always contain kanji/katakana/alphanumerics
    -- this generalizes "is this a real content word" without hardcoding
    any topic-specific vocabulary (car model names, gear names, ...).
    """
    return any(not (_HIRAGANA_RANGE[0] <= ch <= _HIRAGANA_RANGE[1]) for ch in text)


def find_speech_restart_marker(words: list[TranscriptWord]) -> str | None:
    """Detects a clause that ends dangling on a bare topic/case particle
    (never completing a predicate) immediately followed by a clause that
    restates a shared, content-bearing phrase from it -- the speaker
    abandoning an unfinished construction and restarting with the same
    subject in a different construction, rather than continuing it
    naturally (e.g. "Nレンジで下るというのは、Nレンジにすると、...").

    `words` may span multiple consecutive original transcript segments
    (the caller concatenates a candidate's full raw_used range), so a
    restart that faster-whisper's VAD happened to split across two
    segments is still caught, not just one confined to a single segment.

    Returns the shared phrase (for diagnostics) or None. Never guesses
    without word-timestamp data. A pause/gap is never used as a signal by
    itself -- only this text-level restart pattern matters (deliberately
    conservative: prefers a missed restart over a false positive on
    natural emphasis repetition or enumeration, since those have no
    dangling-particle clause or no shared content word respectively).
    """
    if not words:
        return None
    clauses = _split_words_into_clauses(words)
    for i in range(len(clauses) - 1):
        _, _, text_a = clauses[i]
        _, _, text_b = clauses[i + 1]
        # text_a still carries its trailing clause-delimiting comma here
        # (see _split_words_into_clauses) -- checked directly against the
        # comma-inclusive particle endings, not stripped first.
        if not text_a.endswith(DANGLING_CLAUSE_PARTICLE_ENDINGS):
            continue
        if has_speech_disfluency(text_a) is not None:
            continue  # already flagged separately -- avoid double-signaling
        core_a = text_a.rstrip("、，,")
        for particle in DANGLING_CLAUSE_PARTICLE_ENDINGS:
            bare = particle.rstrip("、，,")
            if core_a.endswith(bare):
                core_a = core_a[: -len(bare)]
                break
        core_b = text_b.rstrip("、，,")
        shared = _longest_common_substring(core_a, core_b)
        if len(shared) >= _MIN_SHARED_CONTENT_PHRASE_LEN and _is_content_bearing(shared):
            return shared
    return None


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


# --- End-side trimming (real-machine incident: duration_too_long repair --
# --- only ever dropped a whole non-hook segment, swinging duration by --
# --- far more than needed -- e.g. 51.0s -> 20.7s -- because there was no
# --- symmetric way to shorten a segment's own *end* using real word ------
# --- timestamps, the way start_anchor_text already shortens its start) ---


def find_anchor_end_word(segment: TranscriptSegment, anchor_text: str) -> TranscriptWord | None:
    """The end-side mirror of find_anchor_start_word: locates anchor_text
    as an exact, contiguous substring of the segment's word sequence whose
    *end* lands exactly on a real word boundary, and returns that last
    included word (so a candidate can end mid-segment at a natural clause
    boundary instead of always running through the segment's own last
    word). Never fuzzy-matches and never guesses: returns None if
    anchor_text is falsy, the segment has no word-timestamp data, the
    exact text doesn't appear in the segment at all, or the match's end
    falls in the middle of a word. The caller (boundary.py) must treat
    None as "don't trim" and fall back to the segment's own natural end.
    """
    if not segment.words or not anchor_text:
        return None
    concatenated = ""
    word_end_offsets: list[int] = []
    for word in segment.words:
        concatenated += word.text
        word_end_offsets.append(len(concatenated))
    idx = concatenated.find(anchor_text)
    if idx == -1:
        return None
    end_offset = idx + len(anchor_text)
    try:
        return segment.words[word_end_offsets.index(end_offset)]
    except ValueError:
        # anchor_text was found, but doesn't end exactly on a word
        # boundary -- that would be a mid-word end, which is forbidden.
        return None


def resolve_segment_end_word(
    segment: TranscriptSegment, end_anchor_text: str | None
) -> TranscriptWord | None:
    """The end-side mirror of resolve_segment_start_word. Unlike the start
    side, there is no fixed-vocabulary fallback heuristic here (there is
    no generalizable "known trailing filler phrase" list the way
    WEAK_OPENING_PREFIXES exists for openings) -- if end_anchor_text isn't
    set, this returns None (no trim at all; boundary.py uses the
    segment's own natural end), rather than inventing one.
    """
    if not end_anchor_text:
        return None
    return find_anchor_end_word(segment, end_anchor_text)


# clip_selector._segment_ending_is_confident's judgement, deliberately
# duplicated rather than imported (models.py must never depend on
# clip_selector.py -- same rationale as DANGLING_CLAUSE_PARTICLE_ENDINGS
# above), so find_natural_end_trim_points can judge whether a *sub*-
# segment sentence boundary is a safe place to cut. Unlike clip_selector's
# version (which only ever judges a *whole segment's own real* final
# ending -- one Stage1 already chose as presumably complete), this must
# also positively rule out a sentence dangling on a bare topic/case
# particle (e.g. "それについては、") -- an internal boundary is exactly
# where that shape shows up, so DANGLING_CLAUSE_PARTICLE_ENDINGS (already
# defined above, same module -- no duplication needed here) is checked
# too, not just the continuation-connective suffixes. Also deliberately
# narrower than clip_selector's own marker list: "」"/"』" (closing
# quotation marks) are excluded here because this same tuple doubles as
# _split_words_into_sentences' split points -- a closing quote can appear
# *mid*-sentence when quoting speech ("彼は「はい」と言いました"), so
# treating it as a sentence boundary could produce a cut point that isn't
# actually a complete sentence. clip_selector's version never has this
# problem (it only ever judges a segment's own real, already-chosen final
# word, never uses its marker list to *decide* where an internal boundary
# is), so it can safely include them.
_CLAUSE_TERMINAL_MARKERS = ("。", "！", "？", "!", "?")
_CLAUSE_CONTINUATION_SUFFIXES = (
    "ので", "から", "けど", "けども", "けれど", "けれども", "ですが", "ますが",
    "という", "ということで", "し", "て", "で", "たら", "れば",
)


def _clause_ending_is_confident(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.endswith(_CLAUSE_TERMINAL_MARKERS):
        return True
    if stripped.endswith(DANGLING_CLAUSE_PARTICLE_ENDINGS):
        return False
    return not stripped.endswith(_CLAUSE_CONTINUATION_SUFFIXES)


def _split_words_into_sentences(words: list[TranscriptWord]) -> list[tuple[int, int, str]]:
    """Groups a flat word list into sentence-terminal-punctuation-
    delimited spans (_CLAUSE_TERMINAL_MARKERS: "。", "！", "？", ...) --
    the same accumulate-and-split shape as _split_words_into_clauses, but
    keyed on sentence endings rather than commas. A segment often
    contains several independent, complete sentences run together with no
    comma at all between them (e.g. "最初の文です。二番目の文です。"), which
    _split_words_into_clauses (comma-only) would never separate -- this is
    the natural unit find_natural_end_trim_points needs ("does this
    segment contain a later, independent sentence that can be safely
    dropped from the end"). Kept deliberately separate from
    _split_words_into_clauses (used by find_disfluent_hook_repair_point/
    find_speech_restart_marker for a different purpose -- detecting
    disfluency/restart *within* one sentence) so neither's already-tested
    behavior is put at risk by changing its delimiter set.
    """
    sentences: list[tuple[int, int, str]] = []
    start = 0
    accumulated = ""
    for i, word in enumerate(words):
        accumulated += word.text
        if word.text.endswith(_CLAUSE_TERMINAL_MARKERS) or i == len(words) - 1:
            sentences.append((start, i + 1, accumulated))
            start = i + 1
            accumulated = ""
    return sentences


def find_natural_end_trim_points(segment: TranscriptSegment) -> list[TranscriptWord]:
    """Finds every word in `segment` that a trailing portion of it could
    safely be cut after: every sentence boundary (_split_words_into_
    sentences) other than the segment's own final one, whose accumulated
    text from the segment's start up through that sentence already reads
    as a confidently complete unit (_clause_ending_is_confident --
    terminal punctuation, or at least no dangling particle/grammatical
    continuation marker). Returned ordered from closest to the segment's
    own natural end (the smallest possible trim) to closest to its start
    (the largest), so a caller trying the smallest change first can
    simply iterate in order.

    Never guesses: returns [] if there's no word-timestamp data, or the
    segment has only one sentence (nothing before its own natural end to
    cut after). Never a mid-word or mid-sentence point -- only real
    sentence boundaries whose text is already itself confidently
    complete.
    """
    if not segment.words:
        return []
    sentences = _split_words_into_sentences(segment.words)
    if len(sentences) < 2:
        return []
    points: list[TranscriptWord] = []
    accumulated = ""
    for _start_idx, end_idx, text in sentences[:-1]:  # never the segment's own last sentence -- that's "no trim"
        accumulated += text
        if _clause_ending_is_confident(accumulated):
            points.append(segment.words[end_idx - 1])
    points.reverse()
    return points


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

    end_anchor_text is the symmetric counterpart for the *end* of the
    end_segment_id transcript segment. Two sources set it: clip_
    selector.py's own deterministic, API-0 duration_too_long repair
    (_try_end_trim_repairs -- the real text from the segment's own start
    through a natural, confidently-complete internal clause boundary,
    models.find_natural_end_trim_points), and -- since the Stage2
    final-edit-design redesign -- Stage2 itself (Stage2SegmentOutput has
    an end_anchor_text field Stage1MaterialSegmentOutput does not), when it
    needs to end a segment it designed at an earlier natural point than the
    segment's own real end. Either way it is verified identically at
    resolve time (models.find_anchor_end_word) and falls back to "no
    trim" (the segment's own natural end) if it doesn't match real
    transcript text exactly -- never AI-authored replacement text.
    """

    role: SegmentRole
    start_segment_id: int
    end_segment_id: int  # inclusive
    start_anchor_text: str | None = None
    end_anchor_text: str | None = None


@dataclass
class RawClipCandidate:
    """Claude's Stage2 output for one *finished* candidate design: a
    complete, self-contained Shorts edit -- hook_type/opening_hook_strength/
    score all describe this specific finished design, self-scored by Stage2
    after assembling it (never copied forward from any Stage1 material it
    drew segments from). Stage1 no longer produces this type -- it produces
    RawMaterial (below), a single-purpose raw ingredient that Stage2 freely
    recombines into candidates like this one.
    """

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
class RawMaterialSegment:
    """A segment reference within a RawMaterial. Deliberately has no `role`
    (materials are single-purpose -- they don't have an internal structural
    position like hook/context/answer/payoff, that's decided only when
    Stage2 designs a finished RawClipCandidate) and no end_anchor_text
    (Stage1 never had that capability even for RawUsedSegment; keeping this
    type minimal rather than speculatively adding it)."""

    start_segment_id: int
    end_segment_id: int  # inclusive
    start_anchor_text: str | None = None


@dataclass
class RawMaterial:
    """Stage1's output: one single-purpose raw ingredient (hook, reason,
    example, context, or payoff material), not a finished Shorts candidate.
    Unlike RawClipCandidate, this deliberately has no hook_type/
    opening_hook_strength/score/hook_text/title/description/reasoning/
    caveats -- those all describe a *finished* candidate design, which only
    Stage2 produces (see design_final_candidates in clip_selector.py).
    Forcing a non-hook material (e.g. a calm reason explanation) to carry a
    hook_type/opening_hook_strength would either be meaningless or bias
    Stage1 into judging it by hook-strength standards it was never meant to
    meet -- see prompts/extract_candidates.md's material-type-specific
    quality bar.
    """

    material_type: MaterialType
    segments: list[RawMaterialSegment]
    usefulness_score: int  # 0-100: how useful this material is for its own material_type, not a hook-strength or overall-candidate score

    def __post_init__(self) -> None:
        if not (1 <= len(self.segments) <= 3):
            raise ValueError(
                f"segments must contain 1-3 entries (got {len(self.segments)})"
            )
        if not (0 <= self.usefulness_score <= 100):
            raise ValueError(
                f"usefulness_score must be within 0-100 (got {self.usefulness_score})"
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
