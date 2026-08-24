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
from .models import RawClipCandidate, RawUsedSegment, Transcript, TranscriptSegment

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
        "cta_end_text": {"type": "string"},
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
        "cta_end_text",
        "title",
        "description",
        "score",
        "reasoning",
        "caveats",
    ],
}


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def _format_segments(segments: list[TranscriptSegment]) -> str:
    lines = []
    for seg in segments:
        mm, ss = divmod(int(seg.start), 60)
        lines.append(f"[segment_id={seg.id} start={mm:02d}:{ss:02d}] {seg.text}")
    return "\n".join(lines)


def _raw_candidate_from_tool_input(d: dict) -> RawClipCandidate:
    return RawClipCandidate(
        hook_type=d["hook_type"],
        segments=[
            RawUsedSegment(
                role=s["role"],
                start_segment_id=s["start_segment_id"],
                end_segment_id=s["end_segment_id"],
            )
            for s in d["segments"]
        ],
        hook_text=d["hook_text"],
        cta_end_text=d["cta_end_text"],
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
                _raw_candidate_from_tool_input(c) for c in block.input["candidates"]
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
                "cta_end_text": c.cta_end_text,
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
            return [_raw_candidate_from_tool_input(c) for c in block.input["candidates"]]
    raise RuntimeError("Claude did not return submit_final_candidates")


def select_candidates(
    transcript: Transcript, video_title: str, force_refresh: bool = False
) -> list[RawClipCandidate]:
    """Runs Stage1 -> Stage2 (with caching) and returns exactly 3 candidates,
    validating their total duration against config.DURATION_HARD_*_SEC and
    retrying Stage2 (once) with feedback if any candidate is out of range.
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

    for attempt in range(config.MAX_STAGE2_RETRIES):
        durations = [_candidate_duration(c, transcript) for c in finalists]
        out_of_range = [
            (i, dur)
            for i, dur in enumerate(durations)
            if not (config.DURATION_HARD_MIN_SEC <= dur <= config.DURATION_HARD_MAX_SEC)
        ]
        if not out_of_range:
            break
        feedback_lines = [
            f"候補{i + 1}の合計尺が{dur:.1f}秒で、目標範囲"
            f"({config.DURATION_HARD_MIN_SEC:.0f}〜{config.DURATION_HARD_MAX_SEC:.0f}秒)"
            "から外れています。区間の取り方を見直してください。"
            for i, dur in out_of_range
        ]
        finalists = rank_and_finalize(
            all_candidates, transcript, video_title, feedback="\n".join(feedback_lines)
        )
    else:
        # Retries exhausted; keep the result but flag any still out-of-range
        # candidate's caveats so the UI surfaces it to the user.
        for c in finalists:
            dur = _candidate_duration(c, transcript)
            if not (config.DURATION_HARD_MIN_SEC <= dur <= config.DURATION_HARD_MAX_SEC):
                c.caveats = (
                    c.caveats + f" / 尺が目標範囲外（{dur:.1f}秒）"
                ).strip(" /")

    cache.save_stage2(transcript.video_id, finalists)
    return finalists
