"""Two-stage AI candidate selection (absolute condition #12).

Basic philosophy: Claude's *only* job is deciding which real spoken words
to use ("which segment_ids make a good Shorts clip"). Everything else --
candidate IDs, hook_text, title/description/reasoning/caveats, duration
math, quality filtering, caching, UI/render/QA -- is the program's job.
Claude is never asked to author display text, so there is nothing for it
to invent or get factually wrong, and its structured-output schemas stay
small and cheap.

Stage 1 (per-chunk extraction) proposes candidates as segment_id ranges
only (absolute condition #11) -- never raw seconds, never prose.
boundary.py later turns those IDs into actual edit points. A local quality
filter (duration bounds, spoken-opening strength, and ending completeness
-- see extend_to_natural_ending) runs before Stage 2, so weak candidates
never reach Claude a second time.

Stage 2 (ranking) sees only a compact summary of the Stage1 survivors
(candidate_id/hook_type/opening_hook_strength/score/duration/segment
text) -- never the full transcript, since Stage 1 already narrowed the
search space and Stage 2 only has to compare a handful of candidates
against each other. It returns nothing but an ordered list of candidate
ids; the program takes the top NUM_CANDIDATES.

Neither stage retries automatically on a Structured Outputs failure or on
insufficient candidates -- see StructuredOutputError and
select_candidates. API usage per analyze is therefore predictable: one
call per not-yet-cached Stage1 chunk, plus at most one Stage2 call.

Long transcripts are chunked (config.CHUNK_MINUTES, with
config.CHUNK_OVERLAP_MINUTES of overlap) before Stage 1 rather than sent
whole: Claude's context window is large enough to fit a full 60-90 minute
transcript, but long-context recall degrades for "find every good moment
in this document" style tasks, so chunking trades a bit of latency/cost
for materially better recall across the whole episode. Each chunk's result
is cached the moment it succeeds (cache.save_stage1_chunk), so a failure
partway through a long video never discards chunks that already cost API
calls to produce.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from . import boundary, cache, config, models, structured_output
from .models import RawClipCandidate, RawUsedSegment, Transcript, TranscriptSegment

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


# --- Claude output models (Structured Outputs) --------------------------
# Deliberately minimal: Claude only ever produces segment_id ranges plus
# the small set of scores needed for filtering/ranking. `extra="forbid"`
# makes model_json_schema() emit `additionalProperties: false`, which
# Structured Outputs requires.


class Stage1SegmentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["hook", "context", "answer", "payoff"]
    start_segment_id: int
    end_segment_id: int
    # Optional: a short substring that exists verbatim, contiguously, at a
    # real word boundary near the start of the start_segment_id transcript
    # segment (e.g. "86は" within "これも私の愛車である86はスープラを...").
    # Lets a candidate start mid-segment at a natural phrase boundary
    # instead of always using the segment's literal first word. Never
    # AI-authored replacement text -- boundary.py verifies it against the
    # real transcript (models.find_anchor_start_word) and falls back to no
    # trim if it doesn't match exactly. Length-bounded since it's meant to
    # be a short phrase/clause, not a rewritten sentence.
    start_anchor_text: str | None = Field(default=None, min_length=1, max_length=60)


class Stage1CandidateOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hook_type: Literal["open_loop", "strong_take", "surprising_fact", "story"]
    segments: list[Stage1SegmentOutput] = Field(min_length=1, max_length=3)
    opening_hook_strength: int = Field(ge=0, le=100)
    score: int = Field(ge=0, le=100)


class Stage1Output(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[Stage1CandidateOutput] = Field(
        min_length=0, max_length=config.STAGE1_MAX_CANDIDATES_PER_CHUNK
    )


class Stage2RankingOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ranked_candidate_ids: list[str]


# Weak "warm-up" openings explicitly called out as unacceptable: a mechanical
# safety net that backs up Claude's own opening_hook_strength self-rating by
# catching the most obvious literal cases. This checks the *actual spoken*
# transcript text of the candidate's first (hook) segment (after
# boundary.py's opening trim -- see _opening_text below), never the
# on-screen hook_text overlay -- a strong overlay must never excuse a weak
# spoken opening. The prefix list itself lives in models.py
# (WEAK_OPENING_PREFIXES) so it stays identical to the list
# boundary.py's opening-trim mechanically skips past.
def _looks_like_weak_opening(text: str) -> bool:
    stripped = text.strip()
    return any(stripped.startswith(p) for p in models.WEAK_OPENING_PREFIXES)


# A candidate's spoken opening (or a non-chronological jump's destination
# -- see _validate_candidate_junctions) must be understandable on its own,
# with no prior context. Unlike _looks_like_weak_opening's filler prefixes
# (skippable), a deictic/anaphoric opening ("これの...", "その場合...") has
# a referent the clip may never actually establish, so this is only ever
# used to detect and reject -- never to pick a substitute trim point.
def _looks_context_dependent_opening(text: str) -> bool:
    stripped = text.strip()
    return any(stripped.startswith(p) for p in models.CONTEXT_DEPENDENT_OPENING_PREFIXES)


def _force_first_segment_is_hook(raw: RawClipCandidate) -> None:
    """The first segment of a candidate must be tagged role=hook (this is a
    labeling/consistency requirement, not a semantic judgement -- whatever
    plays first *is* the hook by definition), so this corrects it directly
    rather than asking Claude to regenerate over a mere label mismatch.
    """
    if raw.segments and raw.segments[0].role != "hook":
        raw.segments[0].role = "hook"


def _format_segments(segments: list[TranscriptSegment]) -> str:
    lines = []
    for seg in segments:
        mm, ss = divmod(int(seg.start), 60)
        lines.append(f"[segment_id={seg.id} start={mm:02d}:{ss:02d}] {seg.text}")
    return "\n".join(lines)


def _deterministic_hook_text(
    segment_id: int, start_anchor_text: str | None, segments: list[TranscriptSegment]
) -> str:
    """hook_text is never AI-authored. It's the real transcript text of
    whatever actually plays first -- the candidate's first segment, after
    the same anchor/lead-in trim boundary.py applies when resolving the
    real edit points (models.resolve_segment_start_word), so this display
    text never shows a weak lead-in ("これも私の愛車である...") that the
    rendered clip itself has already trimmed away -- truncated to a safe
    on-screen length by character count only, never rewritten,
    embellished, or invented.
    """
    segment = next(s for s in segments if s.id == segment_id)
    trim_word = models.resolve_segment_start_word(segment, start_anchor_text)
    if trim_word is not None:
        lead_in_len = sum(len(w.text) for w in segment.words if w.start < trim_word.start)
        text = (segment.text[lead_in_len:].lstrip() or segment.text).strip()
    else:
        text = segment.text.strip()
    if len(text) > config.HOOK_TEXT_MAX_CHARS:
        return text[: config.HOOK_TEXT_MAX_CHARS].rstrip() + "…"
    return text


def _raw_candidate_from_stage1_output(
    c: Stage1CandidateOutput, chunk_segments: list[TranscriptSegment]
) -> RawClipCandidate:
    """Converts Claude's minimal Stage1 output into the internal
    RawClipCandidate pipeline dataclass. title/description/reasoning/
    caveats are always empty -- the program never asks Claude to generate
    display copy, so there's nothing to carry over for those fields.
    """
    segments = [
        RawUsedSegment(
            role=s.role,
            start_segment_id=s.start_segment_id,
            end_segment_id=s.end_segment_id,
            start_anchor_text=s.start_anchor_text,
        )
        for s in c.segments
    ]
    return RawClipCandidate(
        hook_type=c.hook_type,
        segments=segments,
        hook_text=_deterministic_hook_text(
            segments[0].start_segment_id, segments[0].start_anchor_text, chunk_segments
        ),
        opening_hook_strength=c.opening_hook_strength,
        title="",
        description="",
        score=c.score,
        reasoning="",
        caveats="",
    )


def _usable_segments(transcript: Transcript) -> list[TranscriptSegment]:
    """Segments Claude is allowed to reference at all.

    Only excludes anything when the user has explicitly set an OP
    exclusion duration (plan fix #3) — there is no hardcoded default.
    """
    if config.OP_EXCLUSION_SECONDS is None:
        return transcript.segments
    return [s for s in transcript.segments if s.start >= config.OP_EXCLUSION_SECONDS]


def _build_chunks(
    segments: list[TranscriptSegment],
) -> list[tuple[int, list[TranscriptSegment]]]:
    if not segments:
        return []
    chunk_len = config.CHUNK_MINUTES * 60
    overlap = config.CHUNK_OVERLAP_MINUTES * 60
    total_end = segments[-1].end

    chunks: list[tuple[int, list[TranscriptSegment]]] = []
    chunk_index = 0
    window_start = 0.0
    while window_start < total_end:
        window_end = window_start + chunk_len
        chunk_segments = [s for s in segments if window_start <= s.start < window_end]
        if chunk_segments:
            chunks.append((chunk_index, chunk_segments))
            chunk_index += 1
        window_start = window_end - overlap
    return chunks


def extract_candidates_for_chunk(
    chunk_segments: list[TranscriptSegment], video_title: str
) -> list[RawClipCandidate]:
    system_prompt = (_PROMPTS_DIR / "extract_candidates.md").read_text(encoding="utf-8")
    user_content = (
        f"# 番組タイトル\n{video_title}\n\n"
        f"# 文字起こし（このチャンクのみ）\n{_format_segments(chunk_segments)}"
    )

    parsed = structured_output.call(
        Stage1Output,
        stage="Stage1",
        system_prompt=system_prompt,
        user_content=user_content,
        max_tokens=config.STAGE1_MAX_OUTPUT_TOKENS,
    )
    return [_raw_candidate_from_stage1_output(c, chunk_segments) for c in parsed.candidates]


def run_stage1(
    transcript: Transcript, video_title: str, force_refresh: bool = False
) -> list[RawClipCandidate]:
    """Runs Stage1 chunk by chunk, caching each chunk's result the moment
    it succeeds (cache.save_stage1_chunk) -- so if a later chunk's API
    call fails, the already-paid-for results from earlier chunks are not
    discarded, and a subsequent run only re-requests the missing chunk(s).
    """
    all_candidates: list[RawClipCandidate] = []
    for chunk_index, chunk_segments in _build_chunks(_usable_segments(transcript)):
        if not force_refresh:
            cached = cache.load_stage1_chunk(transcript.video_id, chunk_index)
            if cached is not None:
                all_candidates.extend(cached)
                continue

        candidates = extract_candidates_for_chunk(chunk_segments, video_title)
        cache.save_stage1_chunk(transcript.video_id, chunk_index, candidates)
        all_candidates.extend(candidates)
    return all_candidates


def _candidate_duration(raw: RawClipCandidate, transcript: Transcript) -> float:
    resolved = boundary.resolve_candidate(raw, transcript, candidate_id="_tmp")
    return resolved.total_duration


def _opening_text(raw: RawClipCandidate, transcript: Transcript) -> str:
    """The candidate's actual opening text *after* boundary.py's start
    trim (see boundary._apply_start_trim) -- never the raw untrimmed
    transcript text. Reading the raw text here would make this reject a
    candidate for a weak lead-in ("このように...") that render/UI have
    already mechanically trimmed away, defeating the trim entirely. Since
    segments may be reordered, `segments[0]` here is whichever segment
    Claude placed first (the actual hook), not necessarily the
    chronologically-first one.
    """
    resolved = boundary.resolve_candidate(raw, transcript, candidate_id="_tmp")
    return resolved.segments[0].text


def _is_allowed_exact_repeat_pair(
    raw: RawClipCandidate, i: int, j: int, exact_group_size: int
) -> bool:
    """A narrow exception to the overlap ban: the *same* source range used
    to punch the hook, then repeated verbatim later as the clip's
    conclusion (answer/payoff) -- e.g. stating the conclusion up front,
    walking through the supporting example, then landing on the exact
    same conclusion again as the payoff. Allowed only when ALL hold:
    - this exact range is used exactly twice in the whole candidate
      (exact_group_size == 2 -- a 3rd use, or a use alongside a merely
      *overlapping*-but-not-identical range, is never allowed here)
    - the first use (i) is the candidate's opening hook (index 0, role
      "hook")
    - the second use (j) is role "answer" or "payoff" (never a second
      "hook" or "context")
    - they are not adjacent (j > i + 1) -- there must be at least one
      other segment (context) between them; immediately repeating the
      hook right after itself is never allowed
    Every other overlap -- partial, 3+ reuse, wrong roles, adjacent -- is
    rejected by the caller.
    """
    if exact_group_size != 2:
        return False
    if i != 0 or raw.segments[i].role != "hook":
        return False
    if raw.segments[j].role not in ("answer", "payoff"):
        return False
    return j > i + 1


def _has_overlapping_segments(raw: RawClipCandidate, transcript: Transcript) -> bool:
    """True if any two of this candidate's segments reference overlapping
    transcript segment_id ranges, UNLESS the pair is the narrow allowed
    exact-repeat pattern (_is_allowed_exact_repeat_pair -- an identical
    range reused verbatim as hook-then-payoff). Segment order is no
    longer required to be chronological (a reordered candidate can place
    a later, stronger utterance first), so this is the safety net against
    the failure modes that freedom opens up: the same real speech playing
    twice in one clip in an unintended way, or extend_to_natural_ending
    accidentally walking one segment into content another segment of the
    same candidate already uses. Never itself decides *which* segment is
    wrong on a disallowed overlap -- callers drop the whole candidate
    rather than guess which side to keep.
    """
    ranges = [
        (transcript.segment_index(rs.start_segment_id), transcript.segment_index(rs.end_segment_id))
        for rs in raw.segments
    ]
    exact_group_sizes: dict[tuple[int, int], int] = {}
    for r in ranges:
        exact_group_sizes[r] = exact_group_sizes.get(r, 0) + 1

    for i in range(len(ranges)):
        for j in range(i + 1, len(ranges)):
            a_start, a_end = ranges[i]
            b_start, b_end = ranges[j]
            if not (a_start <= b_end and b_start <= a_end):
                continue
            if ranges[i] == ranges[j] and _is_allowed_exact_repeat_pair(
                raw, i, j, exact_group_sizes[ranges[i]]
            ):
                continue
            return True
    return False


# The only "confidently complete" text signal -- sentence-final
# punctuation. Everything else about whether a candidate's ending is safe
# is judged structurally (see extend_to_natural_ending): the gap to the
# next transcript segment, not a fixed dictionary of Japanese
# sentence-ending words/particles.
_SENTENCE_END_MARKERS = ("。", "！", "？", "!", "?", "」", "』")


def _ends_with_terminal_punctuation(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and stripped.endswith(_SENTENCE_END_MARKERS)


# Clear grammatical continuation markers ("〜ので", "〜けど", ...): used
# ONLY as a strong *negative* signal that a text is obviously still
# continuing, in combination with the structural (inter-segment gap)
# check below -- never on its own as the sole judge of completeness. Do
# not go back to deciding completeness from this list alone.
_CONFIRMED_CONTINUATION_SUFFIXES = (
    "ので", "から", "けど", "けども", "けれど", "けれども", "ですが", "ますが",
    "という", "ということで", "し", "て", "で", "たら", "れば",
)


def _ends_with_confirmed_continuation(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and stripped.endswith(_CONFIRMED_CONTINUATION_SUFFIXES)


def _extend_raw_segment_to_natural_ending(
    seg: RawUsedSegment, transcript: Transcript, blocked_index_ranges: list[tuple[int, int]]
) -> RawUsedSegment:
    """Core extension loop, shared by extend_to_natural_ending (applied to
    a candidate's last-*played* segment) and _extend_internal_junctions
    (applied to an earlier segment whose next junction is a
    non-chronological jump): if seg doesn't confidently end at a complete
    sentence, pulls in following transcript segments (up to
    config.MAX_END_EXTENSION_SEGMENTS) as long as each is a plausible
    continuation of the same utterance: the gap to the next segment is at
    most config.END_EXTENSION_MAX_GAP_SEC. faster-whisper's VAD only
    splits segments on detected silence, so a short gap is itself a
    continuity signal; a real pause, or no next segment at all, is taken
    as the best available evidence that this *was* an intentional
    stopping point -- but ONLY when the text isn't also grammatically
    signalling that it's still continuing. A pause alone is never treated
    as "the sentence is complete": when the current text ends with a
    confirmed continuation marker (_ends_with_confirmed_continuation,
    e.g. "〜ので"), that is much stronger evidence of an unfinished
    thought than an ambiguous, unpunctuated-but-otherwise-neutral ending,
    so a longer gap (config.END_EXTENSION_CONTINUATION_MAX_GAP_SEC) is
    tolerated before giving up on bridging it -- the marker is used only
    as a negative signal combined with the structural gap check, never as
    the sole judge on its own. Even so, this loop can still stop while
    the text is confirmed-still-continuing (gap too large even under the
    lenient threshold, or budget/next-segment exhausted); callers must
    re-check completeness afterward (has_confident_natural_ending for the
    last segment, _segment_ending_is_confident for an internal one) rather
    than treating "extension stopped" as "now complete".

    Never walks into transcript content another segment of the same
    candidate already uses -- segments may be reordered (not necessarily
    transcript-chronological -- see extract_candidates.md), so blindly
    extending forward could otherwise reuse/duplicate a range another
    segment already covers. blocked_index_ranges (every *other* segment's
    [start_index, end_index], supplied by the caller) stops extension the
    moment the next transcript segment would fall inside one of them,
    exactly as if it had hit the end of the transcript.

    Returns seg unchanged if it already looks complete or can't be safely
    extended further, never None. Never checks duration itself -- callers
    re-check duration bounds afterward. No Claude API call is made either
    way, and this never re-decides *which* segments to use semantically --
    only whether to include a couple more of the segments Claude already
    had available.
    """
    end_id = seg.end_segment_id
    extensions = 0
    while True:
        current = transcript.segment_by_id(end_id)
        if _ends_with_terminal_punctuation(current.text):
            break
        if extensions >= config.MAX_END_EXTENSION_SEGMENTS:
            break
        next_index = transcript.segment_index(end_id) + 1
        if next_index >= len(transcript.segments):
            break
        if any(start <= next_index <= end for start, end in blocked_index_ranges):
            break
        next_segment = transcript.segments[next_index]
        gap = next_segment.start - current.end
        max_gap = (
            config.END_EXTENSION_CONTINUATION_MAX_GAP_SEC
            if _ends_with_confirmed_continuation(current.text)
            else config.END_EXTENSION_MAX_GAP_SEC
        )
        if gap > max_gap:
            break
        end_id = next_segment.id
        extensions += 1

    if end_id == seg.end_segment_id:
        return seg
    return replace(seg, end_segment_id=end_id)


def extend_to_natural_ending(
    raw: RawClipCandidate, transcript: Transcript
) -> RawClipCandidate:
    """Applies _extend_raw_segment_to_natural_ending to the candidate's
    last-*played* segment (raw.segments[-1]), blocked from walking into
    any of this candidate's *other* segments' ranges. Always returns a
    RawClipCandidate -- the input unchanged if nothing was extended.
    Every caller (both _filter_local_quality and finalize_candidates)
    re-checks duration bounds afterward and drops the candidate if
    extension pushed it past DURATION_HARD_MAX_SEC -- the rule is
    identical regardless of whether an alternative candidate happens to
    be available. Used identically by qa.utterance_completeness_qa as a
    safety net, so the primary fix and the QA backstop can never disagree.
    """
    last = raw.segments[-1]
    other_index_ranges = [
        (transcript.segment_index(rs.start_segment_id), transcript.segment_index(rs.end_segment_id))
        for rs in raw.segments[:-1]
    ]
    extended = _extend_raw_segment_to_natural_ending(last, transcript, other_index_ranges)
    if extended is last:
        return raw
    new_segments = list(raw.segments[:-1]) + [extended]
    return replace(raw, segments=new_segments)


def _is_chronological_continuation(
    prev_end_segment_id: int, next_start_segment_id: int, transcript: Transcript
) -> bool:
    """True if next_start_segment_id is literally the transcript segment
    immediately following prev_end_segment_id -- i.e. these two chosen
    segments were adjacent in the original recording with no jump, so
    playing prev's end straight into next's start is just the source
    audio's own natural continuation, never an edited-in transition
    (see extract_candidates.md's "候補内の並び替え").
    """
    return transcript.segment_index(next_start_segment_id) == transcript.segment_index(prev_end_segment_id) + 1


def _extend_internal_junctions(raw: RawClipCandidate, transcript: Transcript) -> RawClipCandidate:
    """For every segment except the last-played one, if the *next*
    segment in this candidate is a non-chronological jump (see
    _is_chronological_continuation) and this segment's current ending
    isn't confidently complete, tries extending it forward through the
    original transcript to a natural ending first -- exactly like
    extend_to_natural_ending does for the last segment -- rather than
    immediately treating the candidate as an unsafe cut. A chronological-
    continuation junction is left untouched regardless of ending shape:
    that's just the source's own natural flow playing on, not a cut, so
    it needs no completeness check here (see _validate_candidate_
    junctions, which only applies that check to non-chronological jumps).

    Never mutates a segment into another segment's range -- each
    extension attempt is blocked by every *other* segment's
    [start_index, end_index] in this same candidate, same as
    extend_to_natural_ending. Returns raw unchanged if nothing needed
    extending.
    """
    segments = list(raw.segments)
    changed = False
    for i in range(len(segments) - 1):
        prev, nxt = segments[i], segments[i + 1]
        if _is_chronological_continuation(prev.end_segment_id, nxt.start_segment_id, transcript):
            continue
        current_text = transcript.segment_by_id(prev.end_segment_id).text
        if _segment_ending_is_confident(current_text):
            continue
        blocked = [
            (transcript.segment_index(rs.start_segment_id), transcript.segment_index(rs.end_segment_id))
            for j, rs in enumerate(segments)
            if j != i
        ]
        extended = _extend_raw_segment_to_natural_ending(prev, transcript, blocked)
        if extended is not prev:
            segments[i] = extended
            changed = True
    if not changed:
        return raw
    return replace(raw, segments=segments)


def _segment_ending_is_confident(text: str) -> bool:
    """True unless text lacks terminal punctuation AND grammatically
    signals it's still continuing (_ends_with_confirmed_continuation) --
    the shared judgement behind has_confident_natural_ending (the
    candidate's very last segment) and _validate_candidate_junctions (any
    internal segment ending a non-chronological jump). A pause (of any
    length) is never, by itself, evidence of completeness for such text.
    """
    if _ends_with_terminal_punctuation(text):
        return True
    return not _ends_with_confirmed_continuation(text)


def has_confident_natural_ending(raw: RawClipCandidate, transcript: Transcript) -> bool:
    """True unless the candidate's current last segment lacks terminal
    punctuation AND grammatically signals it's still continuing
    (_ends_with_confirmed_continuation) -- i.e. extend_to_natural_ending
    ran out of budget, ran off the end of the transcript, or hit a gap
    too large to bridge while the text was still confidently incomplete.
    A pause (of any length) is never, by itself, evidence of completeness
    for such text -- callers (_filter_local_quality, finalize_candidates,
    qa.utterance_completeness_qa) must call this *after*
    extend_to_natural_ending and drop/fail the candidate if it returns
    False, rather than accepting whatever ending extension happened to
    stop at.
    """
    text = transcript.segment_by_id(raw.segments[-1].end_segment_id).text
    return _segment_ending_is_confident(text)


JunctionRejectReason = Literal[
    "hook_context_dependent",
    "jump_prev_incomplete",
    "jump_next_context_dependent",
]


@dataclass
class JunctionEvaluation:
    """Diagnostic-friendly result of evaluate_candidate_junctions: `safe`
    is the production-relevant verdict (see _validate_candidate_junctions,
    is_candidate_junction_safe); `reason`/`junction_index` exist purely so
    a diagnostic report (evaluate_local_candidate, diagnose_local_filter)
    can say *which* rule failed and *where*, without clip_selector.py and
    its own diagnostics ever implementing the judgment twice.
    """

    safe: bool
    reason: JunctionRejectReason | None = None
    # Index i of the failing segments[i] -> segments[i+1] junction; None
    # for a hook_context_dependent failure (that's about segments[0]
    # itself, not a junction between two segments).
    junction_index: int | None = None


def evaluate_candidate_junctions(raw: RawClipCandidate, transcript: Transcript) -> JunctionEvaluation:
    """The candidate reads naturally end-to-end only if the hook's opening
    is understandable with no prior context, and every adjacent A->B
    segment pair forms a safe cut junction. This is a real-machine-
    observed failure mode _has_overlapping_segments/has_confident_natural_
    ending don't catch on their own: a candidate whose *individual*
    segments and *final* ending all look fine can still cut together into
    nonsense mid-clip, e.g. segment A ending "車を冷やしますっていうので
    あれば" (grammatically demanding a following clause) hard-cut into an
    unrelated segment B starting "連続周回をする場合は" (a different
    condition entirely) -- both segments pass every other check, but the
    A->B cut itself is broken Japanese.

    Per junction (segments[i] -> segments[i+1]):
    - Chronological continuation (_is_chronological_continuation: B is
      literally the next transcript segment after A) is always safe
      regardless of A's ending shape -- it's the source recording's own
      unedited flow, not a cut clip_selector introduced. (Note:
      _extend_internal_junctions already tries to turn a would-be-unsafe
      non-chronological jump into a safe chronological one first, by
      extending A to a natural ending within the original transcript --
      this function runs *after* that and judges the result.)
    - A non-chronological jump (a genuine edit) requires BOTH: A's
      ending is confidently complete (_segment_ending_is_confident --
      never cut away from an utterance still grammatically demanding a
      continuation into unrelated content: reason="jump_prev_incomplete"
      if not), AND B's opening doesn't depend on missing prior context
      (_looks_context_dependent_opening on B's resolved text --
      "これの..."/"その場合..." with no antecedent left in the clip:
      reason="jump_next_context_dependent" if it does) -- UNLESS B is an
      exact repeat of the hook itself (same start/end/anchor as
      segments[0]; see _is_exact_hook_repeat), in which case the
      "A confidently complete" half is waived: the viewer is being cut
      back to real, already-clearly-heard content (the hook), not to
      something unrelated, so an abruptly-trailing A does not make that
      cut unsafe the way it would for genuinely different content. B's
      own opening is the hook's, whose independence was already checked
      above, so no further check is needed for that half either. This is
      the narrow counterpart to _has_overlapping_segments' existing
      limited hook/payoff exact-repeat allowance -- both exist to support
      the same deliberate "state the conclusion, give the example, land
      on the exact same conclusion again" structure.

    The hook (segments[0]) is additionally never allowed to open on a
    context-dependent reference, chronological-continuation or not
    (reason="hook_context_dependent"): it's the very first thing played,
    so there is no "prior segment" for it to depend on either way.
    """
    resolved = boundary.resolve_candidate(raw, transcript, candidate_id="_tmp")
    if _looks_context_dependent_opening(resolved.segments[0].text):
        return JunctionEvaluation(safe=False, reason="hook_context_dependent")

    for i in range(len(raw.segments) - 1):
        prev_raw, next_raw = raw.segments[i], raw.segments[i + 1]
        chronological = _is_chronological_continuation(
            prev_raw.end_segment_id, next_raw.start_segment_id, transcript
        )
        if chronological:
            continue
        if i > 0 and _is_exact_hook_repeat(raw, next_raw):
            continue
        prev_text = transcript.segment_by_id(prev_raw.end_segment_id).text
        if not _segment_ending_is_confident(prev_text):
            return JunctionEvaluation(safe=False, reason="jump_prev_incomplete", junction_index=i)
        if _looks_context_dependent_opening(resolved.segments[i + 1].text):
            return JunctionEvaluation(
                safe=False, reason="jump_next_context_dependent", junction_index=i
            )
    return JunctionEvaluation(safe=True)


def _is_exact_hook_repeat(raw: RawClipCandidate, seg: RawUsedSegment) -> bool:
    """True if seg references the exact same source range as this
    candidate's hook (segments[0]) -- same start/end segment_id and same
    start_anchor_text. Used only to recognize the specific, already-
    limited-and-allowed hook/payoff exact-repeat pattern (see
    _has_overlapping_segments and evaluate_candidate_junctions), never as
    a general similarity check.
    """
    hook = raw.segments[0]
    return (
        seg.start_segment_id == hook.start_segment_id
        and seg.end_segment_id == hook.end_segment_id
        and seg.start_anchor_text == hook.start_anchor_text
    )


def _validate_candidate_junctions(raw: RawClipCandidate, transcript: Transcript) -> bool:
    """Thin boolean wrapper around evaluate_candidate_junctions for
    production call sites (_filter_local_quality, finalize_candidates)
    that only ever needed the pass/fail verdict.
    """
    return evaluate_candidate_junctions(raw, transcript).safe


def is_candidate_junction_safe(raw: RawClipCandidate, transcript: Transcript) -> bool:
    """Public wrapper combining _validate_candidate_junctions and
    _has_overlapping_segments for callers outside this module (qa.py's
    junction_safety_qa) that need the identical judgment
    _filter_local_quality/finalize_candidates already apply, without
    reaching into this module's private helpers directly.
    """
    return _validate_candidate_junctions(raw, transcript) and not _has_overlapping_segments(raw, transcript)


LocalRejectReason = Literal[
    "invalid_segment_reference",
    "incomplete_final_ending",
    "overlap",
    "duration_too_short",
    "duration_too_long",
    "hook_strength_below_80",
    "weak_opening_prefix",
    "context_dependent_opening",
    "unsafe_junction",
    "accepted",
]

# Deterministic, API-0 repair strategies repair_local_candidate/
# generate_local_repair_variants can try before giving up on a rejected
# candidate -- see their docstrings for what each one does and which
# reject reason it targets. prepend_previous_{1,2,3} are numbered rather
# than a single "prepend_previous_and_trim" because a real-machine
# incident (round 7) showed pulling in only the immediately-preceding
# segment can land on another context-dependent opening ("あとGR86も...")
# -- the number distinguishes how far back a variant actually reached in
# any diagnostic report.
RepairMethod = Literal[
    "opening_trim",
    "prepend_previous_1",
    "prepend_previous_2",
    "prepend_previous_3",
    "drop_context_segment",
    "drop_non_context_segment",
    "hook_repeat_payoff",
    "replace_incomplete_final_with_hook_payoff",
]


@dataclass
class RepairBuildResult:
    """Internal result of *attempting to construct* one repair variant --
    deliberately separate from evaluating it (see RepairAttemptDiagnostic
    below). `candidate` is the built RawClipCandidate ready for
    evaluate_local_candidate, or None if this method couldn't even be
    constructed for this candidate (e.g. no word timestamps, hook ending
    not confident, already at the 3-segment ceiling with no room for the
    intended replacement); `skip_reason` is set only in that None case, so
    a diagnostic report can say *why* a method was never generated instead
    of just that it wasn't.
    """

    method: RepairMethod
    candidate: RawClipCandidate | None
    skip_reason: str | None = None


@dataclass
class RepairAttemptDiagnostic:
    """Diagnostic-friendly record of one repair method's outcome for a
    candidate, whether or not it ever got as far as evaluate_local_
    candidate. `generated=False` means _build_repair_candidates couldn't
    even construct this variant (see `generation_skip_reason`); the
    `accepted`/`reject_reason`/`junction_reason`/`duration_sec`/
    `opening_text` fields are only meaningful when `generated=True` --
    they are then exactly what evaluate_local_candidate returned for this
    one variant, never a separate judgment.
    """

    method: RepairMethod
    generated: bool
    generation_skip_reason: str | None = None
    accepted: bool = False
    reject_reason: LocalRejectReason | None = None
    junction_reason: JunctionRejectReason | None = None
    duration_sec: float = 0.0
    opening_text: str = ""


@dataclass
class LocalCandidateEvaluation:
    """Diagnostic-friendly result of evaluate_local_candidate. `accepted`/
    `reason` is the exact verdict _filter_local_quality's production
    accept/reject decision is derived from -- evaluate_local_candidate is
    the single implementation both share, so a diagnostic report (see
    diagnose_local_filter) can never disagree with what actually ran.
    `candidate` is the (possibly internally- and finally-extended)
    RawClipCandidate as evaluated up to the point of accept/reject; for an
    "accepted" evaluation this is exactly what _filter_local_quality would
    keep. `opening_text`/`duration_sec` are best-effort and stay at their
    defaults ("" / 0.0) only for reason == "invalid_segment_reference",
    where the candidate's segment_ids can't even be resolved.
    `junction_reason` is only set when reason is "context_dependent_
    opening" or "unsafe_junction" (see evaluate_candidate_junctions). The
    two extension flags are pure debug context, not used for the verdict.
    """

    accepted: bool
    reason: LocalRejectReason
    candidate: RawClipCandidate
    opening_text: str = ""
    duration_sec: float = 0.0
    junction_reason: JunctionRejectReason | None = None
    internal_extension_applied: bool = False
    final_extension_applied: bool = False
    # Set only by evaluate_local_candidate_with_repair when a repair
    # variant was the one actually accepted: repair_method names which
    # strategy worked, original_reason preserves what the *unrepaired*
    # candidate was rejected for, so a diagnostic report can show both
    # ("was: context_dependent_opening, repaired via opening_trim").
    # evaluate_local_candidate itself never sets these.
    repair_method: RepairMethod | None = None
    original_reason: LocalRejectReason | None = None
    # Every repair method evaluate_local_candidate_with_repair actually
    # *generated and evaluated* for this candidate, in order, whether or
    # not any of them succeeded -- so a diagnostic report can show "repair
    # was attempted via X but still rejected" as well as "repair via X
    # succeeded". Empty when no repair was applicable to this candidate's
    # reject reason, or when the candidate was accepted outright (no
    # repair needed). Subset of repair_attempts (generated=True entries).
    attempted_repair_methods: tuple[RepairMethod, ...] = ()
    # Full per-method diagnostic detail -- every method _build_repair_
    # candidates considered for this candidate's reject reason, including
    # ones it couldn't even construct (generated=False,
    # generation_skip_reason set) -- so a diagnostic report can show
    # exactly why e.g. hook_repeat_payoff never got tried, not just that
    # it wasn't. Empty under the same conditions as attempted_repair_
    # methods.
    repair_attempts: tuple[RepairAttemptDiagnostic, ...] = ()


def evaluate_local_candidate(
    candidate: RawClipCandidate, transcript: Transcript
) -> LocalCandidateEvaluation:
    """The single implementation of the local (pre-Stage2) quality gate.
    _filter_local_quality (production accept/reject) and
    diagnose_local_filter (diagnostic report, zero API calls) both call
    this exactly once per candidate rather than re-implementing the
    checks, so production behavior and diagnostics can never disagree.

    Checks run in this exact order (first failure wins), mirroring what
    used to be _filter_local_quality's inline logic:
    1. referential integrity -- every segment_id must exist in transcript
       (a Stage1 chunk only ever sees its own segment_ids, but this stays
       defensive) -> "invalid_segment_reference"
    2. _force_first_segment_is_hook (label fix, never a rejection)
    3. _extend_internal_junctions, then extend_to_natural_ending -- tries
       to turn an unfinished ending (internal or final) into a complete
       one via the original transcript before judging anything below
    4. has_confident_natural_ending on the (possibly extended) last
       segment -> "incomplete_final_ending"
    5. _has_overlapping_segments (beyond the allowed hook/payoff
       exact-repeat exception) -> "overlap"
    6. hard duration bounds, re-checked *after* extension ->
       "duration_too_short" / "duration_too_long"
    7. opening_hook_strength vs config.MIN_OPENING_HOOK_STRENGTH ->
       "hook_strength_below_80" (named for the current default; see
       config.py for the live threshold)
    8. _looks_like_weak_opening on the resolved opening
       (models.WEAK_OPENING_PREFIXES) -> "weak_opening_prefix"
    9. evaluate_candidate_junctions -- the hook itself opening on a
       dangling reference maps to "context_dependent_opening"
       (junction_reason="hook_context_dependent"); an unsafe A->B cut
       elsewhere maps to "unsafe_junction" (junction_reason=
       "jump_prev_incomplete" or "jump_next_context_dependent")
    Anything surviving all of the above is "accepted".
    """
    try:
        for s in candidate.segments:
            transcript.segment_by_id(s.start_segment_id)
            transcript.segment_by_id(s.end_segment_id)
    except KeyError:
        return LocalCandidateEvaluation(
            accepted=False, reason="invalid_segment_reference", candidate=candidate
        )

    _force_first_segment_is_hook(candidate)
    extended = _extend_internal_junctions(candidate, transcript)
    internal_extension_applied = extended is not candidate
    c = extend_to_natural_ending(extended, transcript)
    final_extension_applied = c is not extended

    opening_text = _opening_text(c, transcript)
    duration_sec = _candidate_duration(c, transcript)

    def _rejected(
        reason: LocalRejectReason, junction_reason: JunctionRejectReason | None = None
    ) -> LocalCandidateEvaluation:
        return LocalCandidateEvaluation(
            accepted=False, reason=reason, candidate=c,
            opening_text=opening_text, duration_sec=duration_sec,
            junction_reason=junction_reason,
            internal_extension_applied=internal_extension_applied,
            final_extension_applied=final_extension_applied,
        )

    if not has_confident_natural_ending(c, transcript):
        return _rejected("incomplete_final_ending")

    if _has_overlapping_segments(c, transcript):
        return _rejected("overlap")

    if duration_sec < config.DURATION_HARD_MIN_SEC:
        return _rejected("duration_too_short")
    if duration_sec > config.DURATION_HARD_MAX_SEC:
        return _rejected("duration_too_long")

    if c.opening_hook_strength < config.MIN_OPENING_HOOK_STRENGTH:
        return _rejected("hook_strength_below_80")
    if _looks_like_weak_opening(opening_text):
        return _rejected("weak_opening_prefix")

    junction = evaluate_candidate_junctions(c, transcript)
    if not junction.safe:
        reason: LocalRejectReason = (
            "context_dependent_opening"
            if junction.reason == "hook_context_dependent"
            else "unsafe_junction"
        )
        return _rejected(reason, junction_reason=junction.reason)

    return LocalCandidateEvaluation(
        accepted=True, reason="accepted", candidate=c,
        opening_text=opening_text, duration_sec=duration_sec,
        internal_extension_applied=internal_extension_applied,
        final_extension_applied=final_extension_applied,
    )


# --- deterministic, API-0 repair (real-machine incident: 4/4 Stage1
# candidates rejected -- rather than lowering any threshold, try a small,
# fixed set of structural repairs built only from real transcript text/
# word timestamps/existing segments, then re-run the *exact same*
# evaluate_local_candidate on the result) ----------------------------------


def _try_opening_trim_repair(
    candidate: RawClipCandidate, transcript: Transcript
) -> tuple[RawClipCandidate | None, str | None]:
    """Targets context_dependent_opening: if the hook's own transcript
    segment begins with one or more known-removable phrases
    (models.find_sequential_removable_prefix_word -- weak filler and/or a
    self-narrative aside, chained), sets start_anchor_text to the real
    remaining text from that point on, so boundary.py starts playback
    there. Never guesses a substitute anchor -- returns (None,
    skip_reason) if there's nothing to trim: "no_word_timestamps" (can't
    locate a real word boundary at all), "no_removable_prefix" (the
    opening isn't one of the known filler/self-narrative phrases -- most
    often because the real problem is a dangling reference to something
    named in an *earlier* segment, which _try_prepend_previous_segments_
    repairs targets instead), or "trim_would_empty_segment" (the whole
    segment is just removable lead-in, nothing left to start from).
    Returns (candidate, None) on success.
    """
    hook = candidate.segments[0]
    hook_seg = transcript.segment_by_id(hook.start_segment_id)
    if not hook_seg.words:
        return None, "no_word_timestamps"
    trim_word = models.find_sequential_removable_prefix_word(hook_seg)
    if trim_word is None:
        return None, "no_removable_prefix"
    anchor_text = "".join(w.text for w in hook_seg.words if w.start >= trim_word.start)
    if not anchor_text:
        return None, "trim_would_empty_segment"
    new_hook = replace(hook, start_anchor_text=anchor_text)
    return replace(candidate, segments=[new_hook] + candidate.segments[1:]), None


# Repair-only bounds for _try_prepend_previous_segments_repairs: how many
# whole transcript segments it will ever pull backward from the hook
# (never a distant search), and a real-time safety cap on top of that --
# even within the segment-count bound, a handful of unusually long
# segments could otherwise pull in a lot more audio than intended.
_MAX_PREPEND_LOOKBACK_SEGMENTS = 3
_MAX_PREPEND_LOOKBACK_SEC = 20.0


def _try_prepend_previous_segments_repairs(
    candidate: RawClipCandidate, transcript: Transcript
) -> list[RepairBuildResult]:
    """Targets context_dependent_opening in the case _try_opening_trim_
    repair can't fix: the hook's own segment never names the thing it
    refers to (e.g. "これのクラッチ交換の際に..." -- trimming "これの" alone
    still never says which car), because the real antecedent is spoken in
    an earlier transcript segment (e.g. "...ZN6-86であったり..."). Real-
    machine incident (round 7): pulling in only the single immediately-
    preceding segment can itself land on another dependent opening (e.g.
    "あとGR86もそうだと思うんですけども..." -- an enumeration continuation
    whose first item is still further back), so this builds one variant
    per lookback depth n=1..3 -- prepending exactly n whole, real,
    chronologically-preceding transcript segments -- rather than only
    ever trying one. Each variant independently tries the same sequential
    removable-prefix trim on whichever segment ends up first (that
    segment may itself start with its own filler, e.g. "よくある話が").
    evaluate_local_candidate re-checks context_dependent_opening on every
    variant exactly like any other candidate, so a variant that still
    lands on a dependent opening is simply rejected like any other --
    this function never judges independence itself, only builds.

    Never a distant/unbounded search: n never exceeds _MAX_PREPEND_
    LOOKBACK_SEGMENTS, and a variant whose prepended segment starts more
    than _MAX_PREPEND_LOOKBACK_SEC before the hook's own original start is
    skipped (skip_reason="lookback_exceeds_time_cap") rather than
    generated, even if a segment technically exists there. A variant
    already used by another segment of this same candidate is skipped
    (skip_reason="would_overlap_existing_segment") -- would create an
    overlap. Never fabricates or guesses text; every entry (including
    skipped ones) is reported so a diagnostic report can show why.
    """
    hook = candidate.segments[0]
    results: list[RepairBuildResult] = []
    try:
        hook_idx = transcript.segment_index(hook.start_segment_id)
    except KeyError:
        return results
    original_hook_start = transcript.segments[hook_idx].start
    other_ranges = [
        (transcript.segment_index(rs.start_segment_id), transcript.segment_index(rs.end_segment_id))
        for rs in candidate.segments[1:]
    ]

    for n in range(1, _MAX_PREPEND_LOOKBACK_SEGMENTS + 1):
        method: RepairMethod = f"prepend_previous_{n}"  # type: ignore[assignment]
        start_idx = hook_idx - n
        if start_idx < 0:
            results.append(RepairBuildResult(method, None, "not_enough_preceding_segments"))
            continue
        prepend_seg = transcript.segments[start_idx]
        if original_hook_start - prepend_seg.start > _MAX_PREPEND_LOOKBACK_SEC:
            results.append(RepairBuildResult(method, None, "lookback_exceeds_time_cap"))
            continue
        if any(start <= start_idx <= end for start, end in other_ranges):
            results.append(RepairBuildResult(method, None, "would_overlap_existing_segment"))
            continue

        new_hook = RawUsedSegment(
            role="hook", start_segment_id=prepend_seg.id, end_segment_id=hook.end_segment_id,
        )
        trim_word = models.find_sequential_removable_prefix_word(prepend_seg)
        if trim_word is not None:
            anchor_text = "".join(w.text for w in prepend_seg.words if w.start >= trim_word.start)
            if anchor_text:
                new_hook = replace(new_hook, start_anchor_text=anchor_text)
        variant = replace(candidate, segments=[new_hook] + candidate.segments[1:])
        results.append(RepairBuildResult(method, variant, None))

    return results


def _try_drop_optional_segment_repairs(candidate: RawClipCandidate) -> list[RepairBuildResult]:
    """Targets duration_too_long: for a multi-segment candidate, drops one
    non-hook segment at a time (the hook is never dropped) and lets the
    full evaluator judge whether the shorter, still-structurally-valid
    result is acceptable -- never cuts any segment's own content, only
    removes whole segments Claude already chose. context-role segments
    are tried first (drop_context_segment) since the user's own
    preference is to keep hook+payoff over hook+context when only one can
    fit; non-context segments (answer/payoff) are tried after
    (drop_non_context_segment). A single-segment candidate has nothing
    optional to drop -- both methods are reported as skipped
    (skip_reason="single_segment_candidate") rather than silently absent.
    """
    if len(candidate.segments) < 2:
        return [
            RepairBuildResult("drop_context_segment", None, "single_segment_candidate"),
            RepairBuildResult("drop_non_context_segment", None, "single_segment_candidate"),
        ]
    results: list[RepairBuildResult] = []
    droppable = sorted(
        range(1, len(candidate.segments)),
        key=lambda i: 0 if candidate.segments[i].role == "context" else 1,
    )
    for i in droppable:
        method: RepairMethod = (
            "drop_context_segment" if candidate.segments[i].role == "context" else "drop_non_context_segment"
        )
        new_segments = [s for j, s in enumerate(candidate.segments) if j != i]
        results.append(RepairBuildResult(method, replace(candidate, segments=new_segments), None))
    return results


def _try_hook_repeat_payoff_repairs(
    candidate: RawClipCandidate, transcript: Transcript
) -> list[RepairBuildResult]:
    """Targets incomplete_final_ending: if the hook's own real ending is
    already confidently complete (_segment_ending_is_confident on its
    resolved end-segment text), lands on the same real conclusion again
    (e.g. state the conclusion, walk through the example, restate the
    exact same conclusion) as an *exact* repeat of the hook (identical
    start/end/anchor) with role "payoff" -- rather than leaving the clip
    hanging on a mid-utterance ending. Relies entirely on the existing
    limited hook/payoff exact-repeat exception in _has_overlapping_
    segments/evaluate_candidate_junctions and the full evaluate_local_
    candidate to decide whether this is actually safe (duration,
    junction, overlap shape) -- this function only ever builds the
    candidate; it never pre-judges acceptance.

    Two shapes, chosen by the candidate's current segment count
    (RawClipCandidate allows at most 3 -- a 4th segment would raise):
    - 2 segments (hook + one other): appends the repeat as a 3rd segment
      -- method "hook_repeat_payoff".
    - 3 segments already (hook + two others, real-machine incident round
      7: candidate4's actual shape): there is no room to append, so
      instead *replaces* the current (incomplete) final segment with the
      hook repeat -- method "replace_incomplete_final_with_hook_payoff".
      This still discards nothing Claude chose without judgment: the
      replaced segment is exactly the one whose incompleteness caused the
      rejection in the first place, and the repeat is real, already-
      spoken hook content, not fabricated text.
    Never applied to a single-segment candidate (nothing to repeat
    *after*) -- reported as skipped (skip_reason="single_segment_
    candidate"). Never applied when the hook's own ending isn't
    confidently complete (skip_reason="hook_ending_not_confident") --
    repeating an unfinished hook would just create a second incomplete
    ending.
    """
    if len(candidate.segments) == 1:
        return [RepairBuildResult("hook_repeat_payoff", None, "single_segment_candidate")]

    method: RepairMethod = (
        "hook_repeat_payoff" if len(candidate.segments) == 2
        else "replace_incomplete_final_with_hook_payoff"
    )
    hook = candidate.segments[0]
    hook_end_text = transcript.segment_by_id(hook.end_segment_id).text
    if not _segment_ending_is_confident(hook_end_text):
        return [RepairBuildResult(method, None, "hook_ending_not_confident")]

    payoff = replace(hook, role="payoff")
    if len(candidate.segments) == 2:
        variant = replace(candidate, segments=list(candidate.segments) + [payoff])
    else:
        variant = replace(candidate, segments=list(candidate.segments[:-1]) + [payoff])
    return [RepairBuildResult(method, variant, None)]


# Repair-only bound: caps how many variants a single rejected candidate
# can actually get *evaluated* (built-but-skipped diagnostic entries --
# see RepairBuildResult -- don't count against this), so a pathological
# multi-segment candidate can never balloon into an unbounded local
# re-evaluation loop.
_MAX_LOCAL_REPAIR_VARIANTS = 8


def _build_repair_candidates(
    candidate: RawClipCandidate, transcript: Transcript, reason: LocalRejectReason
) -> list[RepairBuildResult]:
    """Returns every repair method's build attempt (RepairBuildResult) --
    both variants ready for evaluate_local_candidate and ones that
    couldn't be constructed at all -- scoped to the specific reason the
    original candidate was rejected for. A reason with no known repair
    (e.g. hook_strength_below_80, weak_opening_prefix, overlap,
    unsafe_junction on a non-hook junction) yields an empty list, leaving
    the original rejection as the final answer. Every variant is built
    only from real transcript text/word timestamps/existing segments (see
    each _try_*_repair*'s docstring) -- nothing here ever authors new
    speech, reorders words within a sentence, or guesses a timestamp.
    """
    if reason == "context_dependent_opening":
        trimmed, skip = _try_opening_trim_repair(candidate, transcript)
        return [RepairBuildResult("opening_trim", trimmed, skip)] + _try_prepend_previous_segments_repairs(
            candidate, transcript
        )

    if reason == "duration_too_long":
        return _try_drop_optional_segment_repairs(candidate)

    if reason == "incomplete_final_ending":
        return _try_hook_repeat_payoff_repairs(candidate, transcript)

    return []


def generate_local_repair_variants(
    candidate: RawClipCandidate, transcript: Transcript, reason: LocalRejectReason
) -> list[tuple[RepairMethod, RawClipCandidate]]:
    """Thin wrapper over _build_repair_candidates for callers that only
    ever wanted the small, bounded list of (method, repaired candidate)
    pairs actually worth re-evaluating -- drops build attempts that
    couldn't even be constructed (RepairBuildResult.candidate is None)
    and caps the rest at _MAX_LOCAL_REPAIR_VARIANTS.
    """
    built = _build_repair_candidates(candidate, transcript, reason)
    generated = [(r.method, r.candidate) for r in built if r.candidate is not None]
    return generated[:_MAX_LOCAL_REPAIR_VARIANTS]


def evaluate_local_candidate_with_repair(
    candidate: RawClipCandidate, transcript: Transcript
) -> LocalCandidateEvaluation:
    """Repair-before-reject: evaluates `candidate` exactly as evaluate_
    local_candidate would; if and only if that's a rejection, builds every
    applicable repair method's variant (_build_repair_candidates, scoped
    to the specific reject reason) and evaluates each generated one
    through that identical evaluate_local_candidate -- never a separate,
    more lenient judgment. The first variant that comes back accepted
    wins (tagged with repair_method/original_reason so the caller can
    tell it was repaired); if none of them are accepted, returns the
    *original* (unrepaired) rejection, annotated with a full repair_
    attempts breakdown (see RepairAttemptDiagnostic) covering every
    method tried -- including ones that couldn't even be built, and why.
    Production (_filter_local_quality) and diagnostics (diagnose_local_
    filter) both go through this single function via _evaluate_all_local_
    candidates, so they can never disagree about what got repaired or
    why a repair wasn't attempted.

    At most _MAX_LOCAL_REPAIR_VARIANTS built variants are ever actually
    evaluated (evaluate_local_candidate called) for one candidate; any
    beyond that are recorded as skipped (generation_skip_reason=
    "repair_variant_cap_reached") rather than silently dropped.
    """
    original = evaluate_local_candidate(candidate, transcript)
    if original.accepted:
        return original

    built = _build_repair_candidates(candidate, transcript, original.reason)
    attempts: list[RepairAttemptDiagnostic] = []
    evaluated_count = 0

    for r in built:
        if r.candidate is None:
            attempts.append(RepairAttemptDiagnostic(
                method=r.method, generated=False, generation_skip_reason=r.skip_reason,
            ))
            continue
        if evaluated_count >= _MAX_LOCAL_REPAIR_VARIANTS:
            attempts.append(RepairAttemptDiagnostic(
                method=r.method, generated=False, generation_skip_reason="repair_variant_cap_reached",
            ))
            continue

        evaluated_count += 1
        repaired = evaluate_local_candidate(r.candidate, transcript)
        attempts.append(RepairAttemptDiagnostic(
            method=r.method, generated=True, accepted=repaired.accepted,
            reject_reason=None if repaired.accepted else repaired.reason,
            junction_reason=repaired.junction_reason,
            duration_sec=repaired.duration_sec, opening_text=repaired.opening_text,
        ))
        if repaired.accepted:
            repaired.repair_method = r.method
            repaired.original_reason = original.reason
            repaired.attempted_repair_methods = tuple(a.method for a in attempts if a.generated)
            repaired.repair_attempts = tuple(attempts)
            return repaired

    original.attempted_repair_methods = tuple(a.method for a in attempts if a.generated)
    original.repair_attempts = tuple(attempts)
    return original


def _evaluate_all_local_candidates(
    candidates: list[RawClipCandidate], transcript: Transcript
) -> list[LocalCandidateEvaluation]:
    return [evaluate_local_candidate_with_repair(c, transcript) for c in candidates]


def _filter_local_quality(
    candidates: list[RawClipCandidate], transcript: Transcript
) -> list[RawClipCandidate]:
    """Mechanical quality gate that runs before Stage2, so weak candidates
    never cost a second API call. Thin wrapper over
    evaluate_local_candidate (see its docstring for the exact check order
    and reason codes) -- kept as a separate, stable entry point since a
    large existing test suite (and refresh_candidates_only/
    refresh_stage1_and_candidates/select_candidates below) calls it
    directly for just the accepted list, not the full diagnostic detail.
    Candidates that fail any check are dropped silently; no feedback is
    sent back to Claude and no retry happens here.
    """
    return [e.candidate for e in _evaluate_all_local_candidates(candidates, transcript) if e.accepted]


_LOCAL_REJECT_REASON_LABELS: dict[str, str] = {
    "invalid_segment_reference": "存在しないsegment参照",
    "incomplete_final_ending": "終端未完結",
    "overlap": "segment重複",
    "duration_too_short": "尺不足(20秒未満)",
    "duration_too_long": "尺超過(50秒超)",
    "hook_strength_below_80": "hook強度不足",
    "weak_opening_prefix": "弱い導入句",
    "context_dependent_opening": "文脈依存の冒頭",
    "unsafe_junction": "カット接続不自然",
}


def _format_diagnostic_summary(evaluations: list[LocalCandidateEvaluation]) -> str:
    """Renders a compact, scannable breakdown of why each Stage1 candidate
    was accepted/rejected by the local filter, meant to be appended to
    the RuntimeError raised when too few candidates survive it
    (select_candidates / refresh_candidates_only /
    refresh_stage1_and_candidates). That message becomes job.error
    (jobs.py), which the existing frontend already renders as multi-line
    text (static/style.css's `.status` is `white-space: pre-wrap`), so
    this reaches the user's screen with no other UI changes needed. Never
    dumps the full transcript -- only each candidate's already-truncated
    resolved opening text.
    """
    if not evaluations:
        return ""
    accepted = sum(1 for e in evaluations if e.accepted)
    repaired = sum(1 for e in evaluations if e.accepted and e.repair_method is not None)
    counts: dict[str, int] = {}
    for e in evaluations:
        counts[e.reason] = counts.get(e.reason, 0) + 1

    lines = ["", "【診断】", f"Stage1候補: {len(evaluations)}件", f"通過: {accepted}件"]
    if repaired:
        lines.append(f"（うちローカル自動修復: {repaired}件）")
    lines += ["", "不合格内訳:"]
    for reason, label in _LOCAL_REJECT_REASON_LABELS.items():
        n = counts.get(reason, 0)
        if n:
            lines.append(f"- {label}: {n}件")

    lines.append("")
    for i, e in enumerate(evaluations, start=1):
        opening = e.opening_text[:30] + ("…" if len(e.opening_text) > 30 else "")
        if e.accepted and e.repair_method is not None:
            # Repaired and accepted: show what it originally failed for
            # and which repair fixed it (item 11's requested format).
            detail = (
                f"候補{i}: 冒頭「{opening}」 original_reject={e.original_reason} "
                f"repair={e.repair_method} → accepted"
            )
            lines.append(detail)
            continue

        detail = (
            f"候補{i}: 冒頭「{opening}」 "
            f"hook={e.candidate.opening_hook_strength} duration={e.duration_sec:.1f}秒 "
            f"reject={e.reason}"
        )
        if e.junction_reason:
            detail += f" junction={e.junction_reason}"
        lines.append(detail)

        # Full per-method repair breakdown, so a rejection that survived
        # repair shows exactly *why* every attempted method still failed
        # (never just "repair_tried=X/Y → reject") -- and why a method
        # applicable to this reject reason was never even attempted
        # (generated=False, e.g. "hook_ending_not_confident").
        if e.repair_attempts:
            lines.append("  repair_attempts:")
            for a in e.repair_attempts:
                if not a.generated:
                    lines.append(f"  - {a.method}: repair_not_generated reason={a.generation_skip_reason}")
                    continue
                a_detail = f"  - {a.method} → reject={a.reject_reason}"
                if a.junction_reason:
                    a_detail += f" junction={a.junction_reason}"
                a_detail += f" duration={a.duration_sec:.1f}秒"
                lines.append(a_detail)

    return "\n".join(lines)


def diagnose_local_filter(transcript: Transcript) -> list[LocalCandidateEvaluation]:
    """API 0: reuses only the already-cached Stage1 chunk results for this
    transcript (_load_stage1_from_cache_only -- never run_stage1 with
    force_refresh=True, never the Stage1 or Stage2 Anthropic API) and
    returns the per-candidate evaluate_local_candidate verdict for every
    one of them. Lets "why did the local filter leave too few
    candidates" be re-answered against an already-populated cache (e.g.
    after pulling a prompt/threshold change) at zero additional API cost.

    Raises RuntimeError (no evaluations computed) if the Stage1 chunk
    cache is missing or incomplete for this transcript -- this never
    silently falls back to a fresh, API-calling analysis; the caller must
    run a real analysis (or refresh_stage1_and_candidates) first to
    populate the cache.
    """
    stage1_candidates = _load_stage1_from_cache_only(transcript)
    if stage1_candidates is None:
        raise RuntimeError(
            "診断用のStage1候補キャッシュが見つからないか不完全です。"
            "先に解析（またはStage1からの再解析）を一度実行してキャッシュを作成してから、"
            "この診断を実行してください。（この診断自体はAPIを呼び出しません）"
        )
    return _evaluate_all_local_candidates(stage1_candidates, transcript)


def _stage2_summary(candidate_id: str, raw: RawClipCandidate, transcript: Transcript) -> dict:
    """Builds the compact per-candidate summary Stage2 sees -- never the
    full transcript. Stage1 already narrowed the search space to a
    handful of candidates, so Stage2 only needs enough to rank/dedupe
    them: the real text they'd actually use, their duration, and their
    Stage1 scores.
    """
    resolved = boundary.resolve_candidate(raw, transcript, candidate_id=candidate_id)
    segments_text = "\n".join(f"[{s.role}] {s.text}" for s in resolved.segments)
    return {
        "candidate_id": candidate_id,
        "hook_type": raw.hook_type,
        "opening_hook_strength": raw.opening_hook_strength,
        "score": raw.score,
        "duration_sec": round(resolved.total_duration, 1),
        "segments_text": segments_text,
    }


def rank_candidates(
    id_map: dict[str, RawClipCandidate], transcript: Transcript, video_title: str
) -> list[str]:
    """Stage2: ranking only. Claude sees compact per-candidate summaries
    (never the full transcript) and returns nothing but an ordered list of
    candidate ids -- no new title/description/reasoning/caveats/hook_text
    is generated here. Unknown or duplicate ids in the response are
    filtered out (referential-integrity check, not JSON repair); the
    caller decides what to do if too few valid ids remain.
    """
    system_prompt = (_PROMPTS_DIR / "rank_and_finalize.md").read_text(encoding="utf-8")
    summaries = [_stage2_summary(cid, c, transcript) for cid, c in id_map.items()]
    user_content = (
        f"# 番組タイトル\n{video_title}\n\n"
        f"# Stage1候補一覧（重複あり得る）\n{json.dumps(summaries, ensure_ascii=False, indent=2)}"
    )

    parsed = structured_output.call(
        Stage2RankingOutput,
        stage="Stage2",
        system_prompt=system_prompt,
        user_content=user_content,
        max_tokens=config.STAGE2_MAX_OUTPUT_TOKENS,
    )

    seen: set[str] = set()
    ranked: list[str] = []
    for cid in parsed.ranked_candidate_ids:
        if cid in id_map and cid not in seen:
            seen.add(cid)
            ranked.append(cid)
    return ranked


def finalize_candidates(
    candidates: list[RawClipCandidate], transcript: Transcript
) -> list[RawClipCandidate]:
    """Re-applies extend_to_natural_ending, re-checks
    has_confident_natural_ending, AND re-enforces the hard duration bounds
    on a final candidate list, regardless of where it came from -- a fresh
    Stage2 selection or a cache hit. The rule is identical for both: a
    candidate still confidently mid-utterance after extension (e.g. still
    ends in "〜ので" with no viable further segment to bridge to -- a
    pause is never treated as completeness on its own), or one whose
    natural ending pushes it outside
    [DURATION_HARD_MIN_SEC, DURATION_HARD_MAX_SEC], makes it ineligible. It
    is never truncated mid-utterance to fit, and a cache hit does not
    relax this just because there's no alternative candidate to
    substitute -- "no substitute available" is a reason to fail loudly,
    not a reason to accept an out-of-range or incomplete clip. Also
    re-applies _extend_internal_junctions/_has_overlapping_segments/
    _validate_candidate_junctions, the identical junction-safety rules
    _filter_local_quality uses, so a cache hit can never bypass them.

    select_candidates and refresh_candidates_only both call this on every
    return path -- and, critically, *before* cache.save_stage2 rather than
    after -- so the Stage2 cache on disk always holds the finalized
    (extended/duration-validated) result, never the raw pre-correction
    one. web.py's render path (_run_render) also re-applies this
    defensively to whatever it reads back from cache.load_stage2, so a
    cached or in-flight raw candidate can never reach render/QA without
    going through the identical correction the UI already showed.

    If fewer than config.NUM_CANDIDATES remain eligible after this,
    raises immediately -- there is no substitute available without
    re-running Stage1/Stage2 against the Claude API, which this function
    must never do. The caller (re-analyzing an already-cached video)
    needs an explicit signal that a fresh analysis is required, not a
    silently short candidate list. Callers must not call
    cache.save_stage2 until *after* this returns successfully: a raise
    here must never leave a partially-corrected or unresolvable result
    written to disk, so an already-cached-but-now-insufficient candidate
    set is left completely untouched on disk when this raises.
    """
    finalized = []
    for c in candidates:
        extended = _extend_internal_junctions(c, transcript)
        extended = extend_to_natural_ending(extended, transcript)
        if not has_confident_natural_ending(extended, transcript):
            continue
        if _has_overlapping_segments(extended, transcript):
            continue
        if not _validate_candidate_junctions(extended, transcript):
            continue
        dur = _candidate_duration(extended, transcript)
        if config.DURATION_HARD_MIN_SEC <= dur <= config.DURATION_HARD_MAX_SEC:
            finalized.append(extended)

    if len(finalized) < config.NUM_CANDIDATES:
        raise RuntimeError(
            "保存済み候補のうち、自然な発話終端を確保できない候補（例: 「〜ので」等の"
            "継続表現で終わっており、これ以上延長できないもの）、または延長後の尺が"
            f"目標範囲（{config.DURATION_HARD_MIN_SEC:.0f}〜{config.DURATION_HARD_MAX_SEC:.0f}秒）"
            f"を外れる候補があり、有効な{config.NUM_CANDIDATES}候補を確保できませんでした"
            f"（有効{len(finalized)}件）。再解析が必要です。APIへの自動再要求は行いません。"
        )
    return finalized


def select_candidates(
    transcript: Transcript, video_title: str, force_refresh: bool = False
) -> list[RawClipCandidate]:
    """Runs Stage1 (per-chunk cached) -> local quality filter -> Stage2
    (ranking only) and returns exactly config.NUM_CANDIDATES candidates.
    Neither stage retries automatically: if the local filter leaves too
    few candidates, or Stage2 doesn't return enough valid ids, this raises
    immediately rather than requesting more from the API. Every return
    path calls finalize_candidates *before* cache.save_stage2 (never
    after), so the Stage2 cache on disk always holds the finalized
    (ending-corrected, duration-validated) result -- a cache hit never
    bypasses that correction, and render.py's cache.load_stage2 reads
    (see web._run_render) always see the same finalized state the UI
    already showed. A cache hit whose candidates are no longer
    sufficiently valid (finalize_candidates raises) leaves the on-disk
    cache completely untouched -- it is never overwritten with a
    known-bad/insufficient result.
    """
    if not force_refresh:
        cached = cache.load_stage2(transcript.video_id)
        if cached is not None:
            finalized = finalize_candidates(cached, transcript)
            cache.save_stage2(transcript.video_id, finalized)
            return finalized

    stage1_candidates = run_stage1(transcript, video_title, force_refresh=force_refresh)
    evaluations = _evaluate_all_local_candidates(stage1_candidates, transcript)
    filtered = [e.candidate for e in evaluations if e.accepted]
    if len(filtered) < config.NUM_CANDIDATES:
        raise RuntimeError(
            f"ローカル品質フィルタを通過した候補が{len(filtered)}件しかありません"
            f"（{config.NUM_CANDIDATES}件必要）。APIへの自動再要求は行いません。"
            + _format_diagnostic_summary(evaluations)
        )

    return _rank_finalize_and_cache(filtered, transcript, video_title)


def _rank_finalize_and_cache(
    filtered: list[RawClipCandidate], transcript: Transcript, video_title: str
) -> list[RawClipCandidate]:
    """Stage2 ranking (at most once) -> finalize_candidates -> cache.save_stage2
    on success only. Shared by select_candidates, refresh_candidates_only,
    and refresh_stage1_and_candidates so all three apply the identical
    final correction/caching rule instead of each re-implementing it. If
    finalize_candidates raises (too few candidates remain eligible after
    ending-completeness/duration re-validation), the Stage2 cache is never
    touched -- callers only ever see either a full, cached, finalized
    result or an exception, never a partially-written cache.
    """
    id_map = {f"s1_c{i:03d}": c for i, c in enumerate(filtered)}
    ranked_ids = rank_candidates(id_map, transcript, video_title)
    top_ids = ranked_ids[: config.NUM_CANDIDATES]
    if len(top_ids) < config.NUM_CANDIDATES:
        raise RuntimeError(
            f"Stage2ランキングが有効な候補ID{len(top_ids)}件しか返しませんでした"
            f"（{config.NUM_CANDIDATES}件必要）。APIへの自動再要求は行いません。"
        )

    finalists = [id_map[cid] for cid in top_ids]
    finalized = finalize_candidates(finalists, transcript)
    cache.save_stage2(transcript.video_id, finalized)
    return finalized


def _load_stage1_from_cache_only(transcript: Transcript) -> list[RawClipCandidate] | None:
    """Like run_stage1, but never calls the Stage1 API under any
    circumstance -- used by refresh_candidates_only, which must reuse
    only what's already on disk. Returns None the moment any chunk's
    cache entry is missing (including "no chunks at all"), so the caller
    can tell "cache fully covers this transcript" apart from "at least
    one Stage1 API call would be needed to fill a gap" and refuse to make
    that call itself.
    """
    chunks = _build_chunks(_usable_segments(transcript))
    if not chunks:
        return None
    all_candidates: list[RawClipCandidate] = []
    for chunk_index, _ in chunks:
        cached = cache.load_stage1_chunk(transcript.video_id, chunk_index)
        if cached is None:
            return None
        all_candidates.extend(cached)
    return all_candidates


def refresh_candidates_only(
    transcript: Transcript, video_title: str
) -> list[RawClipCandidate]:
    """Low-cost re-selection: reuses the already-cached Transcript (passed
    in by the caller) and Stage1 chunk cache, re-applies the current local
    quality filter (opening trim / natural ending / duration / hook
    strength), and -- only if that leaves enough candidates -- runs
    Stage2 ranking exactly once. Never calls the Stage1 API and never
    re-runs Whisper: this exists specifically so a candidate set that's
    become insufficient after a local-rule change (e.g. the ending-
    completeness fix) can be re-derived without paying for a full
    Stage1+Stage2 re-analysis.

    Ignores any existing Stage2 cache -- a fresh Stage2 ranking always
    runs here -- but that Stage2 call is the *only* Anthropic API request
    this function can ever make, and only after confirming enough valid
    Stage1 candidates exist locally. Both failure paths below (missing
    Stage1 cache, too few candidates after the local filter) raise before
    ever calling rank_candidates, so they're guaranteed to cost 0 API
    calls. A full re-analysis (Stage1 from scratch) is never triggered
    automatically -- the caller must request that separately.
    """
    stage1_candidates = _load_stage1_from_cache_only(transcript)
    if stage1_candidates is None:
        raise RuntimeError(
            "保存済みのStage1候補キャッシュが見つからないか不完全です。"
            "完全な再解析（Stage1からのやり直し）が必要です。"
        )

    evaluations = _evaluate_all_local_candidates(stage1_candidates, transcript)
    filtered = [e.candidate for e in evaluations if e.accepted]
    if len(filtered) < config.NUM_CANDIDATES:
        raise RuntimeError(
            f"保存済みStage1候補のうちローカル品質フィルタを通過したのは{len(filtered)}件です"
            f"（{config.NUM_CANDIDATES}件必要）。完全な再解析が必要です。"
            + _format_diagnostic_summary(evaluations)
        )

    return _rank_finalize_and_cache(filtered, transcript, video_title)


def refresh_stage1_and_candidates(
    transcript: Transcript, video_title: str
) -> list[RawClipCandidate]:
    """Mid-cost re-analysis: reuses the already-cached Transcript (never
    re-runs Whisper) but regenerates Stage1 candidates for every chunk via
    run_stage1(..., force_refresh=True) -- ignoring any existing Stage1
    chunk cache entirely -- because the whole point of this path is that
    the old Stage1 candidates no longer clear the current local quality
    filter (refresh_candidates_only, which only reuses cached Stage1
    results, can't fix that). Each chunk's new result is still saved the
    moment it succeeds (run_stage1 -> cache.save_stage1_chunk), so a later
    chunk's API failure never discards an earlier chunk's freshly-paid-for
    result, and there are zero automatic retries either way
    (structured_output.py's max_retries=0, unchanged).

    After Stage1, the identical local quality filter runs, and if fewer
    than config.NUM_CANDIDATES candidates survive, this raises *before*
    ever calling Stage2 ranking -- a mid-cost re-analysis attempt must
    never silently cascade into more API spend than Stage1 (chunk count)
    + at most one Stage2 call.
    """
    stage1_candidates = run_stage1(transcript, video_title, force_refresh=True)
    evaluations = _evaluate_all_local_candidates(stage1_candidates, transcript)
    filtered = [e.candidate for e in evaluations if e.accepted]
    if len(filtered) < config.NUM_CANDIDATES:
        raise RuntimeError(
            f"Stage1を再解析しましたが、現在の品質基準を満たす候補が{len(filtered)}件しか"
            f"ありませんでした（{config.NUM_CANDIDATES}件必要）。"
            "Stage2ランキングは実行していません。"
            + _format_diagnostic_summary(evaluations)
        )

    return _rank_finalize_and_cache(filtered, transcript, video_title)
