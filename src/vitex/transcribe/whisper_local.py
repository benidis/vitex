"""Local speech-to-text via faster-whisper (the default backend).

Runs entirely on the machine — no cloud calls, no per-minute cost — and returns
segment-level timestamps out of the box.
"""

from __future__ import annotations

import logging
from pathlib import Path

from vitex.errors import DependencyError, TranscriptionError
from vitex.models import Segment

log = logging.getLogger("vitex")


class WhisperTranscriber:
    def __init__(self, model_size: str = "small"):
        self.model_size = model_size

    def transcribe(self, audio_path: Path) -> list[Segment]:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - import guard
            raise DependencyError(
                "faster-whisper is not installed. Install the optional extra:\n"
                "  uv pip install -e '.[whisper]'\n"
                "or switch to the AWS backend with --transcriber aws."
            ) from exc

        log.info("Loading faster-whisper model %r (CPU int8)...", self.model_size)
        # int8 on CPU is the portable default; users on GPU can edit these.
        model = WhisperModel(self.model_size, device="cpu", compute_type="int8")

        log.info("Transcribing %s ...", audio_path)

        def _decode(vad: bool) -> list[Segment]:
            # faster-whisper returns a lazy generator; a missing VAD backend can
            # therefore raise during iteration, so we materialize inside here.
            seg_iter, info = model.transcribe(str(audio_path), vad_filter=vad)
            out = [
                Segment(start=float(s.start), end=float(s.end), text=s.text.strip())
                for s in seg_iter
                if s.text.strip()
            ]
            log.info("Detected language: %s (p=%.2f)", info.language, info.language_probability)
            return out

        # VAD trims silence but needs onnxruntime (silero), which is unavailable
        # on some platforms (e.g. old glibc). Fall back to plain decoding.
        try:
            segments = _decode(vad=True)
        except Exception as exc:  # noqa: BLE001 - VAD backend is best-effort
            log.warning("VAD unavailable (%s); transcribing without silence filtering.", exc)
            segments = _decode(vad=False)

        if not segments:
            raise TranscriptionError("Whisper produced no transcript segments.")
        log.info("Transcribed %d segments", len(segments))
        return segments
