"""Transcribe the downloaded video with faster-whisper.

Segmentation is VAD-based (`vad_filter=True`): faster-whisper only emits
segments for detected speech, so segment boundaries already tend to fall on
natural pauses and non-speech (silence/music/jingles) is not turned into
bogus segments. This is what boundary.py leans on when nudging edit points
to natural audio boundaries (absolute condition #11) — it is not a
guarantee of *semantic* correctness, only of *audio* naturalness.
"""
from __future__ import annotations

from pathlib import Path

from faster_whisper import WhisperModel

from . import cache, config
from .models import Transcript, TranscriptSegment, TranscriptWord

_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(
            config.WHISPER_MODEL_SIZE,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
    return _model


def transcribe_video(
    video_path: Path, video_id: str, force_refresh: bool = False
) -> Transcript:
    if not force_refresh:
        cached = cache.load_transcript(video_id)
        if cached is not None:
            return cached

    model = _get_model()
    segments_iter, info = model.transcribe(
        str(video_path),
        language=config.TRANSCRIBE_LANGUAGE,
        word_timestamps=True,
        vad_filter=True,
    )

    segments: list[TranscriptSegment] = []
    for i, seg in enumerate(segments_iter):
        words = [
            TranscriptWord(start=w.start, end=w.end, text=w.word)
            for w in (seg.words or [])
        ]
        segments.append(
            TranscriptSegment(
                id=i,
                start=seg.start,
                end=seg.end,
                text=seg.text.strip(),
                words=words,
            )
        )

    transcript = Transcript(
        video_id=video_id, language=info.language or config.TRANSCRIBE_LANGUAGE, segments=segments
    )
    cache.save_transcript(transcript)
    return transcript
