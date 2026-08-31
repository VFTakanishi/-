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
from dataclasses import replace
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


def _validate_candidate_junctions(raw: RawClipCandidate, transcript: Transcript) -> bool:
    """True only if the candidate reads naturally end-to-end: the hook's
    opening is understandable with no prior context, and every adjacent
    A->B segment pair forms a safe cut junction. This is a real-machine-
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
      continuation into unrelated content), AND B's opening doesn't
      depend on missing prior context (_looks_context_dependent_opening
      on B's resolved text -- "これの..."/"その場合..." with no
      antecedent left in the clip).

    The hook (segments[0]) is additionally never allowed to open on a
    context-dependent reference, chronological-continuation or not: it's
    the very first thing played, so there is no "prior segment" for it to
    depend on either way.
    """
    resolved = boundary.resolve_candidate(raw, transcript, candidate_id="_tmp")
    if _looks_context_dependent_opening(resolved.segments[0].text):
        return False

    for i in range(len(raw.segments) - 1):
        prev_raw, next_raw = raw.segments[i], raw.segments[i + 1]
        chronological = _is_chronological_continuation(
            prev_raw.end_segment_id, next_raw.start_segment_id, transcript
        )
        if chronological:
            continue
        prev_text = transcript.segment_by_id(prev_raw.end_segment_id).text
        if not _segment_ending_is_confident(prev_text):
            return False
        if _looks_context_dependent_opening(resolved.segments[i + 1].text):
            return False
    return True


def is_candidate_junction_safe(raw: RawClipCandidate, transcript: Transcript) -> bool:
    """Public wrapper combining _validate_candidate_junctions and
    _has_overlapping_segments for callers outside this module (qa.py's
    junction_safety_qa) that need the identical judgment
    _filter_local_quality/finalize_candidates already apply, without
    reaching into this module's private helpers directly.
    """
    return _validate_candidate_junctions(raw, transcript) and not _has_overlapping_segments(raw, transcript)


def _filter_local_quality(
    candidates: list[RawClipCandidate], transcript: Transcript
) -> list[RawClipCandidate]:
    """Mechanical quality gate that runs before Stage2, so weak candidates
    never cost a second API call: duration range, spoken opening strength,
    and referential integrity (segment_ids must actually exist -- a Stage1
    chunk only ever sees its own segment_ids, but this stays defensive).
    Candidates that fail any check are dropped silently; no feedback is
    sent back to Claude and no retry happens here. Ending completeness
    (see extend_to_natural_ending) is applied here too: a candidate still
    confidently mid-utterance after extension (has_confident_natural_ending
    returns False -- e.g. still ends in "〜ので" with no further segment
    to bridge to) is dropped rather than accepted just because a pause
    followed it, and the hard duration bounds are re-checked *after*
    extension -- since Stage1 produced many candidates, one whose natural
    ending pushes it past DURATION_HARD_MAX_SEC is simply dropped in favor
    of another rather than cut off early to hit the ceiling. Also drops
    any candidate whose segments overlap (_has_overlapping_segments,
    beyond its narrow allowed hook/payoff exact-repeat exception) --
    segments may now be reordered rather than strictly chronological, so
    this is the safety net against reused/duplicated transcript content --
    and any candidate with an unsafe internal cut junction
    (_validate_candidate_junctions), after first trying to fix an
    unfinished internal segment by extending it within the original
    transcript (_extend_internal_junctions), exactly like
    extend_to_natural_ending does for the final segment.
    """
    kept = []
    for c in candidates:
        try:
            for s in c.segments:
                transcript.segment_by_id(s.start_segment_id)
                transcript.segment_by_id(s.end_segment_id)
        except KeyError:
            continue

        _force_first_segment_is_hook(c)
        c = _extend_internal_junctions(c, transcript)
        c = extend_to_natural_ending(c, transcript)
        if not has_confident_natural_ending(c, transcript):
            continue

        if _has_overlapping_segments(c, transcript):
            continue

        dur = _candidate_duration(c, transcript)
        if not (config.DURATION_HARD_MIN_SEC <= dur <= config.DURATION_HARD_MAX_SEC):
            continue

        if c.opening_hook_strength < config.MIN_OPENING_HOOK_STRENGTH:
            continue
        if _looks_like_weak_opening(_opening_text(c, transcript)):
            continue
        if not _validate_candidate_junctions(c, transcript):
            continue

        kept.append(c)
    return kept


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
    filtered = _filter_local_quality(stage1_candidates, transcript)
    if len(filtered) < config.NUM_CANDIDATES:
        raise RuntimeError(
            f"ローカル品質フィルタを通過した候補が{len(filtered)}件しかありません"
            f"（{config.NUM_CANDIDATES}件必要）。APIへの自動再要求は行いません。"
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

    filtered = _filter_local_quality(stage1_candidates, transcript)
    if len(filtered) < config.NUM_CANDIDATES:
        raise RuntimeError(
            f"保存済みStage1候補のうちローカル品質フィルタを通過したのは{len(filtered)}件です"
            f"（{config.NUM_CANDIDATES}件必要）。完全な再解析が必要です。"
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
    filtered = _filter_local_quality(stage1_candidates, transcript)
    if len(filtered) < config.NUM_CANDIDATES:
        raise RuntimeError(
            f"Stage1を再解析しましたが、現在の品質基準を満たす候補が{len(filtered)}件しか"
            f"ありませんでした（{config.NUM_CANDIDATES}件必要）。"
            "Stage2ランキングは実行していません。"
        )

    return _rank_finalize_and_cache(filtered, transcript, video_title)
