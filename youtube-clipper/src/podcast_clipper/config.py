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
NUM_CANDIDATES = 3

# Real-machine incident: Stage2 designed only 2 final candidates (both of
# which passed local validation) when NUM_CANDIDATES=3 were required, even
# though Stage1 had supplied plenty of still-unused hook/reason/example
# material across chunks -- Stage2 simply stopped at 2 designs instead of
# exploring further combinations, so the whole run failed despite having
# two perfectly good candidates. Stage2Output.candidates' schema ceiling
# was previously tied directly to NUM_CANDIDATES (max 3), which meant
# Stage2 could never even attempt to over-produce so the local gate (which
# never rejected anything in that incident) could pick the best
# NUM_CANDIDATES of a larger pool. STAGE2_MAX_DESIGNS decouples "how many
# designs Stage2 may propose before validation" from NUM_CANDIDATES ("how
# many finalized candidates must survive validation to succeed") --
# _design_finalize_and_cache still requires the same NUM_CANDIDATES to
# pass local validation to consider the run successful (never relaxed),
# it just now has more candidates to validate against when Stage1 material
# supports it.
STAGE2_MAX_DESIGNS = 6

# Stage1's per-chunk candidate (material) cap. This is *search breadth*,
# not the final candidate count: raising MIN_OPENING_HOOK_STRENGTH to 80
# means Stage1 capping itself at 3 candidates/chunk can leave too few
# survivors for Stage2 to design NUM_CANDIDATES final candidates from,
# especially when only one or two utterances per chunk actually clear that
# bar. Stage1's role is recall (cast a wide net of everything usable --
# strong conclusions, reasons, examples, potential hooks/context, not
# necessarily complete 20-50s constructs), not picking or assembling the
# final best-3 -- that's Stage2's job (final edit design), applied to
# materials pooled across every chunk, recombining across them freely.
# Deliberately left at 6, not raised, for the Stage1/Stage2 responsibility
# redesign: test_stage1_output_max_json_size_is_well_under_max_tokens
# already showed the worst-case Stage1Output JSON grows close to half of
# STAGE1_MAX_OUTPUT_TOKENS at 6; raising this without also raising that
# ceiling would reopen the exact max_tokens-truncation risk that test
# guards against, and this redesign doesn't need a wider per-chunk cap to
# work (recall breadth comes from Stage1 no longer needing to bundle a
# complete hook+reason+example into one candidate, not from more of them).
STAGE1_MAX_CANDIDATES_PER_CHUNK = 6

TARGET_DURATION_MIN_SEC = 25.0
TARGET_DURATION_MAX_SEC = 45.0
# Hard validation bounds: outside this range triggers one feedback+retry
# pass to Claude before the candidate is accepted as-is.
DURATION_HARD_MIN_SEC = 20.0
DURATION_HARD_MAX_SEC = 50.0

MIN_SEGMENTS_PER_CANDIDATE = 1
MAX_SEGMENTS_PER_CANDIDATE = 3
DEFAULT_SEGMENTS_PER_CANDIDATE = 2

# Below this, a candidate's *spoken* opening (the real transcript text of
# its first/hook segment) is treated as too weak and triggers the same
# feedback+retry pass as an out-of-range duration (see
# clip_selector.select_candidates). Real-machine validation showed 60 let
# through openings that are explanatory/abstract rather than an actual
# scroll-stopping hook (see prompts/extract_candidates.md's 70-79 band), so
# this is raised to 80: only openings the prompt would score as "clearly
# makes you want to keep watching" or stronger pass. If fewer than
# NUM_CANDIDATES clear this bar, that's a real quality shortfall -- do not
# lower this threshold to force 3 candidates.
MIN_OPENING_HOOK_STRENGTH = 80

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

# Bump this whenever clip_selector.py's Claude prompt text or Structured
# Outputs schema changes in a way that makes previously-cached Stage1/Stage2
# JSON stale/incompatible. cache.py stores this alongside the cached data
# and treats a mismatch as a cache miss (falls back to a fresh Stage1/Stage2
# run) rather than trying to deserialize old-shape data. The Whisper
# transcript cache has no dependency on this and is unaffected.
CANDIDATE_SCHEMA_VERSION = 12

CHUNK_MINUTES = 10.0
CHUNK_OVERLAP_MINUTES = 1.0

ANTHROPIC_MODEL = os.environ.get("PODCAST_CLIPPER_ANTHROPIC_MODEL", "claude-sonnet-5")

# Ceilings for Stage1/Stage2 Structured Outputs responses. Stage1 now
# generates material_type/segments/usefulness_score per material (a raw
# ingredient, not a finished candidate); Stage2 generates hook_type/
# segments/opening_hook_strength/score per finished candidate design (the
# only place those finished-candidate properties are ever produced) --
# everything else (hook_text/title/description/reasoning/caveats) is
# filled in deterministically by the program, so the schemas are small and
# these ceilings are sized to match, not left at a large shared default.
# Each is a ceiling, not a fixed cost: a response that finishes naturally
# does not consume all of it. STAGE2_MAX_OUTPUT_TOKENS raised 512->1024 for
# the Stage2 redesign (its output now includes a full segments array with
# anchors per candidate instead of a bare id list), then 1024->4096 when
# Stage2Output.candidates' ceiling was decoupled from NUM_CANDIDATES(3) to
# STAGE2_MAX_DESIGNS(6) -- see STAGE2_MAX_DESIGNS's docstring. Not a plain
# doubling: each Stage2 segment carries *two* anchors (start_anchor_text
# AND end_anchor_text, unlike Stage1's material segments which only ever
# have one), so the worst case (6 candidates x 3 max-length-anchor
# segments) came out to ~1307 estimated tokens -- see test_stage2_output_
# max_json_size_is_well_under_max_tokens -- comfortably under half of 4096.
STAGE1_MAX_OUTPUT_TOKENS = 2048
STAGE2_MAX_OUTPUT_TOKENS = 4096

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
