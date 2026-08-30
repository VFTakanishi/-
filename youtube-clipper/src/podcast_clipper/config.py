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
TARGET_DURATION_MIN_SEC = 25.0
TARGET_DURATION_MAX_SEC = 45.0
# Hard validation bounds: outside this range triggers one feedback+retry
# pass to Claude before the candidate is accepted as-is.
DURATION_HARD_MIN_SEC = 20.0
DURATION_HARD_MAX_SEC = 50.0
MAX_STAGE2_RETRIES = 1

MIN_SEGMENTS_PER_CANDIDATE = 1
MAX_SEGMENTS_PER_CANDIDATE = 3
DEFAULT_SEGMENTS_PER_CANDIDATE = 2

# Below this, a candidate's *spoken* opening (the real transcript text of
# its first/hook segment, not the on-screen hook_text overlay) is treated
# as too weak and triggers the same feedback+retry pass as an out-of-range
# duration (see clip_selector.select_candidates).
MIN_OPENING_HOOK_STRENGTH = 60

# Bump this whenever clip_selector.py's Claude prompt text or Structured
# Outputs schema changes in a way that makes previously-cached Stage1/Stage2
# JSON stale/incompatible. cache.py stores this alongside the cached data
# and treats a mismatch as a cache miss (falls back to a fresh Stage1/Stage2
# run) rather than trying to deserialize old-shape data. The Whisper
# transcript cache has no dependency on this and is unaffected.
CANDIDATE_SCHEMA_VERSION = 3

CHUNK_MINUTES = 10.0
CHUNK_OVERLAP_MINUTES = 1.0

ANTHROPIC_MODEL = os.environ.get("PODCAST_CLIPPER_ANTHROPIC_MODEL", "claude-sonnet-5")

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
WATERMARK_TEXT = os.environ.get("PODCAST_CLIPPER_WATERMARK_TEXT", "VF高西で検索！")
HOOK_TEXT_DISPLAY_SEC = 2.0

RELATED_VIDEO_INSTRUCTIONS = (
    "YouTubeへアップロード後、YouTube Studioでこのショートの「関連動画」に"
    "元のポッドキャスト本編を設定してください。"
)

# --- Japanese font handling (Windows-oriented) --------------------------
# A single .ttf/.otf is recommended over a .ttc (see README) because a
# TrueType Collection can resolve to an unexpected face inside drawtext.
FONT_PATH = os.environ.get(
    "PODCAST_CLIPPER_FONT_PATH", "C:/Windows/Fonts/meiryo.ttc"
)

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
