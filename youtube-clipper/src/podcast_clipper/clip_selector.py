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
-- see _extend_to_natural_ending) runs before Stage 2, so weak candidates
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

from . import boundary, cache, config, structured_output
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


class Stage1CandidateOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hook_type: Literal["open_loop", "strong_take", "surprising_fact", "story"]
    segments: list[Stage1SegmentOutput] = Field(min_length=1, max_length=3)
    opening_hook_strength: int = Field(ge=0, le=100)
    score: int = Field(ge=0, le=100)


class Stage1Output(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[Stage1CandidateOutput] = Field(min_length=0, max_length=3)


class Stage2RankingOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ranked_candidate_ids: list[str]


# Weak "warm-up" openings explicitly called out as unacceptable: a mechanical
# safety net that backs up Claude's own opening_hook_strength self-rating by
# catching the most obvious literal cases. This checks the *actual spoken*
# transcript text of the candidate's first (hook) segment, never the
# on-screen hook_text overlay -- a strong overlay must never excuse a weak
# spoken opening.
_WEAK_OPENING_PREFIXES = [
    "今回は", "今日は", "ということで", "えー", "えーと", "えっと", "あの", "まあ", "さて",
]


def _looks_like_weak_opening(text: str) -> bool:
    stripped = text.strip()
    return any(stripped.startswith(p) for p in _WEAK_OPENING_PREFIXES)


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


def _deterministic_hook_text(segment_id: int, segments: list[TranscriptSegment]) -> str:
    """hook_text is never AI-authored. It's the real transcript text of
    whatever plays first (the candidate's first segment), truncated to a
    safe on-screen length by character count only -- never rewritten,
    embellished, or invented.
    """
    text = next(s.text for s in segments if s.id == segment_id).strip()
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
        RawUsedSegment(role=s.role, start_segment_id=s.start_segment_id, end_segment_id=s.end_segment_id)
        for s in c.segments
    ]
    return RawClipCandidate(
        hook_type=c.hook_type,
        segments=segments,
        hook_text=_deterministic_hook_text(segments[0].start_segment_id, chunk_segments),
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
    return transcript.segment_by_id(raw.segments[0].start_segment_id).text


# Sentence-final punctuation. A weak signal alone (Whisper's Japanese
# punctuation output isn't guaranteed), used together with the
# continuation-suffix list and the inter-segment gap check below -- never
# as the sole judge of completeness.
_SENTENCE_END_MARKERS = ("。", "！", "？", "!", "?", "」", "』")

# Well-known non-final grammatical particles/conjunctions. Ending on one
# of these (with no sentence-final punctuation) is a strong signal the
# thought continues into the next transcript segment.
_CONTINUATION_SUFFIXES = (
    "ので", "のに", "けど", "けれど", "けれども", "という", "ということで",
    "だから", "ですが", "ますが", "が", "し", "て", "で", "たら", "れば",
)


def is_natural_sentence_ending(text: str) -> bool:
    """True if text plausibly ends at a complete thought: sentence-final
    punctuation, or (absent that) not ending in a well-known non-final
    particle/conjunction. Used both to decide whether
    _extend_to_natural_ending needs to pull in more segments, and by
    qa.utterance_completeness_qa as a post-render safety net.
    """
    stripped = text.strip()
    if not stripped:
        return True
    if stripped.endswith(_SENTENCE_END_MARKERS):
        return True
    return not stripped.endswith(_CONTINUATION_SUFFIXES)


def _extend_to_natural_ending(
    raw: RawClipCandidate, transcript: Transcript
) -> RawClipCandidate | None:
    """If the candidate's last segment looks cut off mid-utterance, pulls
    in following transcript segments (up to
    config.MAX_END_EXTENSION_SEGMENTS) as long as each is a plausible
    continuation of the same utterance -- the gap to the next segment is
    at most config.END_EXTENSION_MAX_GAP_SEC. A real pause (which is how
    faster-whisper's VAD splits segments in the first place) means the
    next segment is a new thought, not a safe continuation.

    Returns a new RawClipCandidate (input untouched) with the last
    segment's end_segment_id extended, the original candidate unchanged
    if it already ends naturally, or None if a natural ending can't be
    reached within the extension budget -- the caller should reject the
    candidate rather than cut it off mid-utterance anyway. This never
    calls the Claude API and never re-decides *which* segments to use
    semantically, only whether to include a couple more of the segments
    Claude already had available.
    """
    last = raw.segments[-1]
    end_id = last.end_segment_id
    extensions = 0
    while True:
        current = transcript.segment_by_id(end_id)
        if is_natural_sentence_ending(current.text):
            break
        if extensions >= config.MAX_END_EXTENSION_SEGMENTS:
            return None
        next_index = transcript.segment_index(end_id) + 1
        if next_index >= len(transcript.segments):
            return None
        next_segment = transcript.segments[next_index]
        if next_segment.start - current.end > config.END_EXTENSION_MAX_GAP_SEC:
            return None
        end_id = next_segment.id
        extensions += 1

    if end_id == last.end_segment_id:
        return raw
    new_segments = list(raw.segments[:-1]) + [
        RawUsedSegment(role=last.role, start_segment_id=last.start_segment_id, end_segment_id=end_id)
    ]
    return replace(raw, segments=new_segments)


def _filter_local_quality(
    candidates: list[RawClipCandidate], transcript: Transcript
) -> list[RawClipCandidate]:
    """Mechanical quality gate that runs before Stage2, so weak candidates
    never cost a second API call: duration range, spoken opening strength,
    and referential integrity (segment_ids must actually exist -- a Stage1
    chunk only ever sees its own segment_ids, but this stays defensive).
    Candidates that fail any check are dropped silently; no feedback is
    sent back to Claude and no retry happens here. Ending completeness
    (see _extend_to_natural_ending) is checked here too: a candidate that
    can't reach a natural ending within its extension budget is dropped
    rather than rendered mid-utterance, and the hard duration bounds are
    re-checked *after* extension -- a natural ending that pushes the clip
    past DURATION_HARD_MAX_SEC drops the candidate rather than cutting it
    off early to hit the ceiling.
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

        extended = _extend_to_natural_ending(c, transcript)
        if extended is None:
            continue
        c = extended

        dur = _candidate_duration(c, transcript)
        if not (config.DURATION_HARD_MIN_SEC <= dur <= config.DURATION_HARD_MAX_SEC):
            continue

        if c.opening_hook_strength < config.MIN_OPENING_HOOK_STRENGTH:
            continue
        if _looks_like_weak_opening(_opening_text(c, transcript)):
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


def select_candidates(
    transcript: Transcript, video_title: str, force_refresh: bool = False
) -> list[RawClipCandidate]:
    """Runs Stage1 (per-chunk cached) -> local quality filter -> Stage2
    (ranking only) and returns exactly config.NUM_CANDIDATES candidates.
    Neither stage retries automatically: if the local filter leaves too
    few candidates, or Stage2 doesn't return enough valid ids, this raises
    immediately rather than requesting more from the API.
    """
    if not force_refresh:
        cached = cache.load_stage2(transcript.video_id)
        if cached is not None:
            return cached

    stage1_candidates = run_stage1(transcript, video_title, force_refresh=force_refresh)
    filtered = _filter_local_quality(stage1_candidates, transcript)
    if len(filtered) < config.NUM_CANDIDATES:
        raise RuntimeError(
            f"ローカル品質フィルタを通過した候補が{len(filtered)}件しかありません"
            f"（{config.NUM_CANDIDATES}件必要）。APIへの自動再要求は行いません。"
        )

    id_map = {f"s1_c{i:03d}": c for i, c in enumerate(filtered)}
    ranked_ids = rank_candidates(id_map, transcript, video_title)
    top_ids = ranked_ids[: config.NUM_CANDIDATES]
    if len(top_ids) < config.NUM_CANDIDATES:
        raise RuntimeError(
            f"Stage2ランキングが有効な候補ID{len(top_ids)}件しか返しませんでした"
            f"（{config.NUM_CANDIDATES}件必要）。APIへの自動再要求は行いません。"
        )

    finalists = [id_map[cid] for cid in top_ids]
    cache.save_stage2(transcript.video_id, finalists)
    return finalists
