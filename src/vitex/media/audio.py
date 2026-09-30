"""Extract a transcription-friendly audio track from the source video."""

from __future__ import annotations

import logging
from pathlib import Path

from vitex.utils import require_tool, run

log = logging.getLogger("vitex")


def extract_audio(video_path: Path, workdir: Path) -> Path:
    """Write mono 16 kHz PCM WAV next to the work dir and return its path.

    16 kHz mono is what both faster-whisper and Amazon Transcribe expect, and
    it keeps the file small.
    """
    ffmpeg = require_tool("ffmpeg")
    workdir.mkdir(parents=True, exist_ok=True)
    audio_path = workdir / "audio.wav"
    run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(audio_path),
        ],
        desc="ffmpeg audio extraction",
    )
    log.info("Extracted audio: %s", audio_path)
    return audio_path
