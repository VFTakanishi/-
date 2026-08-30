"""Two-stage AI candidate selection (absolute condition #12).

Stage 1 (per-chunk extraction) and Stage 2 (merge/dedupe/final ranking)
both ask Claude to choose *transcript segment IDs* only (absolute
condition #11) — never raw seconds. boundary.py later turns those IDs
into actual edit points.

Long transcripts are chunked (config.CHUNK_MINUTES, with
config.CHUNK_OVERLAP_MINUTES of overlap) before Stage 1 rather than sent
whole: Claude's context window is large enough to fit a full 60-90 minute
transcript, but long-context recall degrades for "find every good moment
in this document" style tasks, so chunking trades a bit of latency/cost
for materially better recall across the whole episode.
"""
from __future__ import annotations

import json
from pathlib import Path

import anthropic

from . import boundary, cache, config
from .models import (
    RawClipCandidate,
    RawUsedSegment,
    Transcript,
    TranscriptSegment,
    require_dict,
)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"

_CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "hook_type": {
            "type": "string",
            "enum": ["open_loop", "strong_take", "surprising_fact", "story"],
        },
        "segments": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "enum": ["hook", "context", "answer", "payoff"],
                    },
                    "start_segment_id": {"type": "integer"},
                    "end_segment_id": {"type": "integer"},
                },
                "required": ["role", "start_segment_id", "end_segment_id"],
            },
        },
        "hook_text": {"type": "string"},
        "opening_hook_strength": {"type": "integer", "minimum": 0, "maximum": 100},
        "title": {"type": "string"},
        "description": {"type": "string"},
        "score": {"type": "integer", "minimum": 0, "maximum": 100},
        "reasoning": {"type": "string"},
        "caveats": {"type": "string"},
    },
    "required": [
        "hook_type",
        "segments",
        "hook_text",
        "opening_hook_strength",
        "title",
        "description",
        "score",
        "reasoning",
        "caveats",
    ],
}

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


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def _format_segments(segments: list[TranscriptSegment]) -> str:
    lines = []
    for seg in segments:
        mm, ss = divmod(int(seg.start), 60)
        lines.append(f"[segment_id={seg.id} start={mm:02d}:{ss:02d}] {seg.text}")
    return "\n".join(lines)


def _raw_candidate_from_tool_input(
    d: dict, *, stage: str, candidate_index: int
) -> RawClipCandidate:
    """Parses one candidate from Claude's real tool_use response.

    Diagnostic-only (2026-08-30 incident): a real-machine analyze job
    failed with a bare "TypeError: string indices must be integers, not
    'str'" and no traceback (jobs.py only persisted str(exc)). Reproducing
    it showed this exact function raises that exact message whenever `d`
    (or one of its `segments` items) isn't a dict -- most plausibly because
    Claude's structured response didn't match the schema for one item. The
    require_dict() calls below turn that into a message naming the stage,
    the candidate/segment index, and the actual type/value, so the next
    real occurrence is diagnosable without manual reproduction. This does
    NOT change what's accepted -- still raises on the same malformed input,
    just with a better error -- and does not add any retry/repair behavior.
    """
    require_dict(d, context=f"{stage} candidates[{candidate_index}]")
    segments = []
    for seg_index, s in enumerate(d["segments"]):
        require_dict(
            s, context=f"{stage} candidates[{candidate_index}].segments[{seg_index}]"
        )
        segments.append(
            RawUsedSegment(
                role=s["role"],
                start_segment_id=s["start_segment_id"],
                end_segment_id=s["end_segment_id"],
            )
        )
    return RawClipCandidate(
        hook_type=d["hook_type"],
        segments=segments,
        hook_text=d["hook_text"],
        opening_hook_strength=d["opening_hook_strength"],
        title=d["title"],
        description=d["description"],
        score=d["score"],
        reasoning=d["reasoning"],
        caveats=d["caveats"],
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

    tool = {
        "name": "submit_chunk_candidates",
        "description": "このチャンクから抽出した切り抜き候補を提出する",
        "input_schema": {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "maxItems": 3,
                    "items": _CANDIDATE_SCHEMA,
                }
            },
            "required": ["candidates"],
        },
    }

    response = _client().messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=4096,
        system=system_prompt,
        tools=[tool],
        tool_choice={"type": "tool", "name": "submit_chunk_candidates"},
        messages=[{"role": "user", "content": user_content}],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_chunk_candidates":
            return [
                _raw_candidate_from_tool_input(c, stage="Stage1", candidate_index=i)
                for i, c in enumerate(block.input["candidates"])
            ]
    return []


def run_stage1(
    transcript: Transcript, video_title: str, force_refresh: bool = False
) -> list[dict]:
    if not force_refresh:
        cached = cache.load_stage1(transcript.video_id)
        if cached is not None:
            return cached

    chunks = _build_chunks(_usable_segments(transcript))
    results = []
    for chunk_index, chunk_segments in chunks:
        candidates = extract_candidates_for_chunk(chunk_segments, video_title)
        results.append({"chunk_index": chunk_index, "candidates": candidates})

    cache.save_stage1(transcript.video_id, results)
    return results


def _candidate_duration(raw: RawClipCandidate, transcript: Transcript) -> float:
    resolved = boundary.resolve_candidate(raw, transcript, candidate_id="_tmp")
    return resolved.total_duration


def rank_and_finalize(
    all_candidates: list[RawClipCandidate],
    transcript: Transcript,
    video_title: str,
    feedback: str | None = None,
) -> list[RawClipCandidate]:
    system_prompt = (_PROMPTS_DIR / "rank_and_finalize.md").read_text(encoding="utf-8")

    candidates_json = json.dumps(
        [
            {
                "hook_type": c.hook_type,
                "segments": [
                    {
                        "role": s.role,
                        "start_segment_id": s.start_segment_id,
                        "end_segment_id": s.end_segment_id,
                    }
                    for s in c.segments
                ],
                "hook_text": c.hook_text,
                "opening_hook_strength": c.opening_hook_strength,
                "title": c.title,
                "description": c.description,
                "score": c.score,
                "reasoning": c.reasoning,
                "caveats": c.caveats,
            }
            for c in all_candidates
        ],
        ensure_ascii=False,
        indent=2,
    )

    user_content = (
        f"# 番組タイトル\n{video_title}\n\n"
        f"# Stage1で抽出された候補一覧（重複あり得る）\n{candidates_json}\n\n"
        f"# 参考: 文字起こし全体（segment_idの意味確認用）\n{_format_segments(transcript.segments)}"
    )
    if feedback:
        user_content += f"\n\n# 修正依頼\n{feedback}"

    tool = {
        "name": "submit_final_candidates",
        "description": "統合・重複排除・スコアリングを行った最終3候補を提出する",
        "input_schema": {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": _CANDIDATE_SCHEMA,
                }
            },
            "required": ["candidates"],
        },
    }

    response = _client().messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=4096,
        system=system_prompt,
        tools=[tool],
        tool_choice={"type": "tool", "name": "submit_final_candidates"},
        messages=[{"role": "user", "content": user_content}],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_final_candidates":
            return [
                _raw_candidate_from_tool_input(c, stage="Stage2", candidate_index=i)
                for i, c in enumerate(block.input["candidates"])
            ]
    raise RuntimeError("Claude did not return submit_final_candidates")


def _opening_text(raw: RawClipCandidate, transcript: Transcript) -> str:
    return transcript.segment_by_id(raw.segments[0].start_segment_id).text


def _find_issues(
    finalists: list[RawClipCandidate], transcript: Transcript
) -> list[str]:
    """Mechanical validation the AI's own judgement can't be fully trusted
    to self-enforce: duration range (unchanged from before) and spoken
    opening strength (new). Both feed the same feedback+retry pass rather
    than separate mechanisms.
    """
    issues = []
    for i, c in enumerate(finalists):
        dur = _candidate_duration(c, transcript)
        if not (config.DURATION_HARD_MIN_SEC <= dur <= config.DURATION_HARD_MAX_SEC):
            issues.append(
                f"候補{i + 1}の合計尺が{dur:.1f}秒で、目標範囲"
                f"({config.DURATION_HARD_MIN_SEC:.0f}〜{config.DURATION_HARD_MAX_SEC:.0f}秒)"
                "から外れています。区間の取り方を見直してください。"
            )

        opening_text = _opening_text(c, transcript)
        if c.opening_hook_strength < config.MIN_OPENING_HOOK_STRENGTH or _looks_like_weak_opening(
            opening_text
        ):
            issues.append(
                f"候補{i + 1}: 冒頭の実際の発言「{opening_text}」が弱い"
                f"（助走・前置き的、または opening_hook_strength={c.opening_hook_strength} が低すぎます）。"
                "最初の1〜3秒で強い主張・意外な事実・明確な疑問・結論先出し・"
                "具体的で続きを聞きたくなる一言・強い違和感/対立/問題提起のいずれかを満たす"
                "実際の発言から始まるsegment_idを選び直してください。"
                "元動画の時系列上の最初から始める必要はありません。"
            )
    return issues


def select_candidates(
    transcript: Transcript, video_title: str, force_refresh: bool = False
) -> list[RawClipCandidate]:
    """Runs Stage1 -> Stage2 (with caching) and returns exactly 3 candidates,
    validating their total duration against config.DURATION_HARD_*_SEC and
    their spoken opening strength against config.MIN_OPENING_HOOK_STRENGTH,
    retrying Stage2 (once) with feedback if any candidate fails either check.
    """
    if not force_refresh:
        cached = cache.load_stage2(transcript.video_id)
        if cached is not None:
            return cached

    stage1_results = run_stage1(transcript, video_title, force_refresh=force_refresh)
    all_candidates = [c for chunk in stage1_results for c in chunk["candidates"]]
    if not all_candidates:
        raise RuntimeError("Stage1 produced no candidates for this video")

    finalists = rank_and_finalize(all_candidates, transcript, video_title)
    for c in finalists:
        _force_first_segment_is_hook(c)

    for attempt in range(config.MAX_STAGE2_RETRIES):
        issues = _find_issues(finalists, transcript)
        if not issues:
            break
        finalists = rank_and_finalize(
            all_candidates, transcript, video_title, feedback="\n".join(issues)
        )
        for c in finalists:
            _force_first_segment_is_hook(c)
    else:
        # Retries exhausted; keep the result but flag any still-failing
        # candidate's caveats so the UI surfaces it to the user, rather than
        # crashing the whole analyze job over a still-imperfect candidate.
        for i, c in enumerate(finalists):
            dur = _candidate_duration(c, transcript)
            notes = []
            if not (config.DURATION_HARD_MIN_SEC <= dur <= config.DURATION_HARD_MAX_SEC):
                notes.append(f"尺が目標範囲外（{dur:.1f}秒）")
            opening_text = _opening_text(c, transcript)
            if c.opening_hook_strength < config.MIN_OPENING_HOOK_STRENGTH or _looks_like_weak_opening(
                opening_text
            ):
                notes.append("冒頭の発言が弱い可能性")
            if notes:
                c.caveats = (c.caveats + " / " + " / ".join(notes)).strip(" /")

    cache.save_stage2(transcript.video_id, finalists)
    return finalists
