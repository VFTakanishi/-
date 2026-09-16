"""Central configuration/constants. Values a user may reasonably want to
tune live here or via environment variables (see .env.example); everything
else is a fixed implementation detail, not a knob.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Output / storage -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = Path(os.environ.get("PODCAST_CLIPPER_OUTPUT_DIR", BASE_DIR / "output"))

# --- Candidate selection (absolute conditions #1, #4, #5, #12) --------
# The target/ceiling for how many final candidates to show the user --
# NOT a required minimum. Real-machine incident: a run that produced 1-2
# genuinely good, locally-accepted candidates used to be thrown away as a
# total failure (RuntimeError, nothing shown to the user) just because a
# 3rd didn't also survive every gate -- aiming for 100 points and shipping
# 0 is worse than shipping the 1-2 solid candidates that do exist.
# _design_finalize_and_cache/finalize_candidates now only raise when
# ZERO candidates survive; 1 or more always succeeds and returns exactly
# that many (never padded up to NUM_CANDIDATES, never rejected merely for
# being fewer). Quality bars themselves (hook strength, duration, semantic
# closure, disfluency/restart, junction safety, ending completeness,
# overlap, segment validity) are never relaxed to reach a higher count.
NUM_CANDIDATES = 3

# Real-machine incident (predates the hook-seed-discovery redesign): Stage2
# designed only 2 final candidates (both of which passed local validation)
# when NUM_CANDIDATES=3 were required, even though Stage1 had supplied
# plenty of still-unused material across chunks -- Stage2 simply stopped
# short instead of exploring further combinations. The hook-seed-discovery
# redesign replaces "Stage2 freely designs up to N candidates from a pooled
# material list" with "Python selects up to STAGE2_MAX_COVERAGE_TARGETS
# distinct hook seeds and Stage2 must return exactly one attempt (a real
# candidate design, or an explicit reject) per target" -- so under-
# production is now structurally impossible (a missing attempt is a
# detectable coverage gap, not a silent shortfall) rather than merely
# discouraged by prompt wording.
#
# STAGE2_MAX_COVERAGE_TARGETS is the user's own "roughly 6-8" -- picked at
# the upper bound so a genuinely distinct 7th/8th strong hook seed is never
# dropped purely by the cap (Python's own overlap-based dedup, not this
# cap, is what keeps duplicate/near-duplicate seeds from ever reaching
# Stage2 -- see _select_hook_seed_coverage_targets).
STAGE2_MAX_COVERAGE_TARGETS = 8

# Fallback candidates exist purely to guarantee a non-empty result when too
# few primary attempts pass the hard gate -- the worst case that could ever
# actually use one is "0 primary accepted, need all NUM_CANDIDATES from
# fallback", so there is no reason to allow more than NUM_CANDIDATES of
# them; capping here also keeps Stage2's fallback design work bounded
# instead of open-ended.
STAGE2_MAX_FALLBACK_CANDIDATES = NUM_CANDIDATES

# Stage1's per-chunk output is now split into three groups (hook_seeds/
# support_materials/fallback_spans -- see the hook-seed-discovery redesign)
# rather than one shared "candidates" list, so a single combined cap no
# longer describes it. Each cap below carries forward the same recall-
# breadth reasoning the old STAGE1_MAX_CANDIDATES_PER_CHUNK(6) comment
# gave: Stage1's role is recall (cast a wide net), never picking/assembling
# the final best-N -- that's Stage2's job, applied to material pooled
# across every chunk, recombining across it freely.
STAGE1_MAX_HOOK_SEEDS_PER_CHUNK = 6
STAGE1_MAX_SUPPORT_MATERIALS_PER_CHUNK = 6
# Fallback spans are a 0-candidate safety net, not a primary source -- kept
# deliberately small (never more than STAGE2_MAX_FALLBACK_CANDIDATES could
# ever use) so the prompt has no room to pad with junk just to fill a slot
# ("無理に埋めるためのfallback_spanを作らないこと" -- see
# prompts/extract_candidates.md).
STAGE1_MAX_FALLBACK_SPANS_PER_CHUNK = 2

TARGET_DURATION_MIN_SEC = 25.0
TARGET_DURATION_MAX_SEC = 45.0
# Hard validation bounds: outside this range triggers one feedback+retry
# pass to Claude before the candidate is accepted as-is.
DURATION_HARD_MIN_SEC = 20.0
DURATION_HARD_MAX_SEC = 50.0

MIN_SEGMENTS_PER_CANDIDATE = 1
MAX_SEGMENTS_PER_CANDIDATE = 3
DEFAULT_SEGMENTS_PER_CANDIDATE = 2

# MIN_OPENING_HOOK_STRENGTH (formerly a hard 80-point gate on Stage2's own
# self-reported opening_hook_strength) was RETIRED in the hook-seed-
# discovery redesign, not merely retuned. Real-machine incident: Stage2
# self-rated a context-dependent, unusable opening ("ちょくちょく壊れちゃい
# ます...") at 80 and it sailed through, while a genuinely usable candidate
# elsewhere could be hard-rejected for scoring 78-79 -- a pure self-reported
# number is not a reliable pass/fail signal (Claude grading its own
# output). opening_hook_strength still exists on Stage2CandidateOutput but
# is now soft/diagnostic-only, feeding Stage2's own relative `ranking`
# rather than any Python accept/reject decision. The hard-gate role a
# strong number was trying (and failing) to serve -- catching a hook that
# LOOKS acceptable but isn't actually self-contained -- is now served by
# the explicit semantic field RawClipCandidate.opening_self_contained
# (Stage2's own bool judgment, informed by the same lookahead/lookback
# material it already sees), hard-gated in evaluate_local_candidate. Do
# not reintroduce a numeric hook-strength floor as a substitute.

# Ending completeness: if a candidate's last segment looks cut off
# mid-utterance, clip_selector extends into following transcript segments
# (see clip_selector.extend_to_natural_ending) rather than ending on an
# incomplete thought. Bounded so a bad heuristic match can't run away.
MAX_END_EXTENSION_SEGMENTS = 3
# A gap this short between transcript segments is treated as the same
# breath/utterance continuing (faster-whisper's VAD only splits segments
# on detected silence, so a short gap is itself a continuity signal)
# rather than a real pause marking a completed thought.
END_EXTENSION_MAX_GAP_SEC = 0.8
# A confirmed non-final grammatical ending (ので/から/けど/という/...) is
# much stronger evidence of continuation than an unpunctuated-but-
# otherwise-ambiguous ending, so it gets a longer (but still bounded)
# allowance before an inter-segment gap is trusted as a real pause -- a
# pause alone must never be enough to call "...と思うので" complete. No
# real transcript data was available to calibrate this exactly, so it's
# set conservatively at ~2x the base gap threshold rather than guessed
# loosely (e.g. not several seconds).
END_EXTENSION_CONTINUATION_MAX_GAP_SEC = 1.5

# Real-machine incident: a candidate ended at ~23s mid-explanation -- the
# duration and hook/junction checks all passed, but the speech content was
# clearly not finished. Root cause: Stage2 never saw any transcript content
# past a material's own chosen segments, so it had no way to judge whether a
# more complete, still-natural ending existed just beyond its current
# choice. STAGE2_LOOKAHEAD_MAX_SEGMENTS/_SEC bound a small, reference-only
# window of real transcript segments immediately following each material's
# last segment, shown to Stage2 so it can check "is there a cleaner, still-
# natural stopping point a bit further on" -- not a mandate to always use
# this much extra footage, and deliberately not the full remaining
# transcript (API input size).
STAGE2_LOOKAHEAD_MAX_SEGMENTS = 4
STAGE2_LOOKAHEAD_MAX_SEC = 20.0

# Bump this whenever clip_selector.py's Claude prompt text or Structured
# Outputs schema changes in a way that makes previously-cached Stage1/Stage2
# JSON stale/incompatible. cache.py stores this alongside the cached data
# and treats a mismatch as a cache miss (falls back to a fresh Stage1/Stage2
# run) rather than trying to deserialize old-shape data. The Whisper
# transcript cache has no dependency on this and is unaffected.
# Kept as a single shared version (not split per-stage) even though this
# round changes both Stage1's and Stage2's schemas: a split would save no
# recomputation this round (both caches must invalidate together
# regardless), and would require auditing every one of test_cache.py's many
# existing "schema vN is a miss" historical tests to determine which stage
# each was really about. Revisit a STAGE1_SCHEMA_VERSION/
# STAGE2_SCHEMA_VERSION split only if a future round changes just one
# stage's prompt/schema in isolation -- Stage1 is the metered-per-chunk
# stage, so an asymmetric change is what would actually make a split pay
# for itself.
CANDIDATE_SCHEMA_VERSION = 14

CHUNK_MINUTES = 10.0
CHUNK_OVERLAP_MINUTES = 1.0

ANTHROPIC_MODEL = os.environ.get("PODCAST_CLIPPER_ANTHROPIC_MODEL", "claude-sonnet-5")

# Ceilings for Stage1/Stage2 Structured Outputs responses. Stage1's output
# is now split into three small groups (hook_seeds/support_materials/
# fallback_spans, up to STAGE1_MAX_HOOK_SEEDS_PER_CHUNK(6)/
# STAGE1_MAX_SUPPORT_MATERIALS_PER_CHUNK(6)/STAGE1_MAX_FALLBACK_SPANS_PER_
# CHUNK(2) respectively -- up to 14 items total per chunk vs the old
# single 6-item list), and hook_seeds deliberately carry no text/reasoning
# field (Python resolves real text from segment_id via the transcript) to
# keep the per-item cost small despite the higher item count. Stage2's
# output is now attempts (up to STAGE2_MAX_COVERAGE_TARGETS(8), each
# carrying a full Stage2CandidateOutput on the candidate path) +
# fallback_candidates (up to STAGE2_MAX_FALLBACK_CANDIDATES(3), same
# per-item shape) + a ranking id list -- materially larger than the old
# single up-to-6-candidates list this replaces. Measured directly via
# test_stage1_output_max_json_size_is_well_under_max_tokens/
# test_stage2_output_max_json_size_is_well_under_max_tokens: worst-case
# Stage1Output JSON came out to ~1578 estimated tokens, Stage2Output to
# ~3227 -- both ceilings below were set so each stays comfortably (not
# just barely) under half its ceiling; adjust both together if a future
# schema change shifts that measurement.
STAGE1_MAX_OUTPUT_TOKENS = 4096
STAGE2_MAX_OUTPUT_TOKENS = 8192

# hook_text is the candidate's real opening transcript text (never
# AI-authored -- see clip_selector.py's _deterministic_hook_text), truncated
# to this many characters for on-screen display.
HOOK_TEXT_MAX_CHARS = 40

# --- Transcription --------------------------------------------------
WHISPER_MODEL_SIZE = os.environ.get("PODCAST_CLIPPER_WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.environ.get("PODCAST_CLIPPER_WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.environ.get("PODCAST_CLIPPER_WHISPER_COMPUTE_TYPE", "int8")
TRANSCRIBE_LANGUAGE = "ja"

# --- OP/intro exclusion (absolute condition #10, plan fix #3) ---------
# Deliberately NOT a hardcoded default like 60s: only excludes anything
# when the user explicitly sets this env var. Otherwise Claude's own
# content judgement + Content QA are the only lines of defense against
# picking OP/logo/jingle/greeting-only openings.
_op_exclusion_env = os.environ.get("PODCAST_CLIPPER_OP_EXCLUSION_SECONDS")
OP_EXCLUSION_SECONDS: float | None = (
    float(_op_exclusion_env) if _op_exclusion_env else None
)

# --- Boundary correction (absolute condition #11) ----------------------
BOUNDARY_PADDING_MS = 200

# --- Vertical video (absolute condition #7) -----------------------------
VERTICAL_WIDTH = 1080
VERTICAL_HEIGHT = 1920
BACKGROUND_BLUR_SIGMA = 20

# --- Text overlays (absolute condition #8) ------------------------------
# The only in-video text is this always-on watermark -- no hook-text
# overlay, no end-of-clip CTA subtitle.
WATERMARK_TEXT = os.environ.get("PODCAST_CLIPPER_WATERMARK_TEXT", "VF高西で検索！")

# Watermark/CTA styling. Real-machine validation found the original values
# (fontsize=56, box_color=black@0.55, box_borderw=18, bottom margin=140) too
# subtle to read as a call-to-action, so these were raised for stronger
# on-screen presence while staying clear of the Shorts UI (like/comment
# buttons) at the bottom of the frame.
WATERMARK_FONT_SIZE = 80
WATERMARK_BOX_COLOR = "black@0.82"
WATERMARK_BOX_BORDERW = 26
WATERMARK_BOTTOM_MARGIN = 210

RELATED_VIDEO_INSTRUCTIONS = (
    "YouTubeへアップロード後、YouTube Studioでこのショートの「関連動画」に"
    "元のポッドキャスト本編を設定してください。"
)

# --- Japanese font handling (Windows-oriented, with a Linux/Docker fallback) -
# A single .ttf/.otf is recommended over a .ttc (see README) because a
# TrueType Collection can resolve to an unexpected face inside drawtext.
# PODCAST_CLIPPER_FONT_PATH always wins when set (unchanged). When it isn't,
# the previous behavior was to hardcode the Windows path even though it
# never exists on Linux (Docker/Railway) -- _default_font_path() now checks
# for that Windows path first (local Windows users see no change at all)
# and only falls back to a common Noto Sans CJK JP install location
# (installed via the Dockerfile's `fonts-noto-cjk` apt package -- see
# tests/conftest.py's find_available_japanese_font, which already looks in
# the same place for local dev) if neither exists, this still returns the
# Windows path unchanged, so text_overlay.ensure_font_available()'s
# existing "not found" error message and behavior are unaffected.
_WINDOWS_DEFAULT_FONT_PATH = "C:/Windows/Fonts/meiryo.ttc"
_LINUX_FONT_PATH_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
)


def _default_font_path(_exists=lambda p: Path(p).exists()) -> str:
    if _exists(_WINDOWS_DEFAULT_FONT_PATH):
        return _WINDOWS_DEFAULT_FONT_PATH
    for candidate in _LINUX_FONT_PATH_CANDIDATES:
        if _exists(candidate):
            return candidate
    return _WINDOWS_DEFAULT_FONT_PATH


FONT_PATH = os.environ.get("PODCAST_CLIPPER_FONT_PATH") or _default_font_path()

# --- QA thresholds (absolute condition #13) ------------------------------
BLACKDETECT_MIN_DURATION_SEC = 0.5
BLACKDETECT_PIXEL_BLACK_TH = 0.10
BLACKDETECT_PICTURE_BLACK_RATIO_TH = 0.98

# Freeze detection is based on decoded-frame identity (via ffmpeg's
# framemd5 muxer), not ffmpeg's freezedetect filter: freezedetect's
# average-changed-pixels heuristic false-positives on low-motion-but-real
# content (e.g. a small moving speaker inset against mostly-static slides).
# A run of byte-identical decoded frames lasting at least this long counts
# as a real freeze; frames are sampled at FREEZE_FRAME_SAMPLE_FPS.
FREEZEDETECT_MIN_FREEZE_DURATION_SEC = 1.5
FREEZE_FRAME_SAMPLE_FPS = 5.0
# Only the first few seconds of the clip are checked for a frozen/static
# opening (a real talking-head clip is expected to move).
CONTENT_QA_OPENING_WINDOW_SEC = 3.0

SILENCE_MEAN_VOLUME_DB_THRESHOLD = -50.0
LOW_VOLUME_MEAN_DB_THRESHOLD = -35.0

# Tolerance for the speech-start-alignment self-consistency check: the
# rendered clip's start must line up with the word timestamp boundary.py
# resolved, within this many seconds (accounts for the boundary padding).
SPEECH_ALIGNMENT_TOLERANCE_SEC = 0.35

# --- Jobs ------------------------------------------------------------
JOB_STATES_IN_PROGRESS = {"queued", "analyzing", "rendering"}
JOB_STATE_INTERRUPTED = "interrupted"

# --- Cloud deployment: simple password gate (see web.py) ----------------
# When this is a public Railway URL rather than localhost, a bare password
# check keeps strangers out without any user database/OAuth/session store
# -- deliberately not a real auth system. Unset (the local/default case)
# means the gate is disabled entirely, so nothing changes for local use.
# Server-side only: never sent to the frontend, never logged (see web.py's
# _tool_password_gate/_password_matches -- only an HMAC digest of this
# value is ever placed in a cookie, never the raw password itself).
TOOL_PASSWORD = os.environ.get("TOOL_PASSWORD") or None
