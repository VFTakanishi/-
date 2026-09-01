# Railway (or any container host) deploy image for Podcast Clipper.
#
# Build context is expected to be the youtube-clipper/ directory itself
# (Railway "Root Directory" = youtube-clipper -- see CLOUD_DEPLOY.md),
# never the monorepo root, so every COPY below is relative to this file's
# own directory.
#
# This does not replace local usage: start.bat/launch.py still run the app
# directly on 127.0.0.1:8000 with no Docker involved. This image is only
# the cloud path -- same podcast_clipper package, started via uvicorn
# instead, bound to 0.0.0.0 and Railway's dynamic $PORT.
FROM python:3.11-slim

# ffmpeg: required for ffprobe (ingest.py) and all render/QA steps.
# fonts-noto-cjk: Japanese-capable font for the CTA watermark
# (text_overlay.py/config.py's FONT_PATH auto-detects this install path
# when PODCAST_CLIPPER_FONT_PATH isn't set -- see config.py's
# _default_font_path). fontconfig lets fc-cache register it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ffmpeg \
       fonts-noto-cjk \
       fontconfig \
    && fc-cache -f \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, so a source-only change doesn't invalidate this
# (slow: faster-whisper/ctranslate2) layer.
COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN pip install -r requirements.txt \
    && pip install -e .

COPY prompts ./prompts

# Anthropic API key, TOOL_PASSWORD, and PODCAST_CLIPPER_OUTPUT_DIR are all
# supplied as Railway Environment Variables at deploy time -- never baked
# into this image (see CLOUD_DEPLOY.md). If PODCAST_CLIPPER_OUTPUT_DIR
# points at a mounted Railway Volume (e.g. /data/output), the Transcript/
# Stage1/Stage2 caches and the faster-whisper model download below both
# survive a redeploy instead of re-costing API calls / a multi-GB
# re-download every time.
ENV HF_HOME=/data/huggingface

EXPOSE 8000

# Railway injects PORT at runtime; ${PORT:-8000} keeps `docker run` (no
# PORT set) working too. host=0.0.0.0 is required -- Railway's edge proxy
# connects to the container over the network, not localhost.
CMD ["sh", "-c", "uvicorn podcast_clipper.web:app --host 0.0.0.0 --port ${PORT:-8000}"]
