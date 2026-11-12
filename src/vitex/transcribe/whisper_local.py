"""Local speech-to-text via faster-whisper (the default backend).

Runs entirely on the machine — no cloud calls, no per-minute cost — and returns
segment-level timestamps out of the box.

faster-whisper's VAD (silence filtering) needs onnxruntime, which has no wheel
for some platforms (e.g. Python 3.12 on glibc < 2.28). Without it, Whisper tends
to hallucinate text over silence and music ("Thank you." loops). In that case
we detect silence with ffmpeg's ``silencedetect`` instead and drop segments that
fall inside it, and collapse runs of repeated segments.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from vitex.errors import DependencyError, TranscriptionError
from vitex.models import Segment
from vitex.utils import require_tool, run

log = logging.getLogger("vitex")

_SILENCE_NOISE_DB = -35  # quieter than this counts as silence
_SILENCE_MIN_SECONDS = 1.5
_SILENT_FRACTION = 0.8  # drop a segment this much covered by silence
_MAX_REPEATS = 2  # keep at most this many consecutive identical segments

_SILENCE_START_RE = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*(-?[\d.]+)")


def detect_silences(audio_path: Path) -> list[tuple[float, float]]:
    """Return (start, end) spans of silence in ``audio_path`` via ffmpeg."""
    ffmpeg = require_tool("ffmpeg")
    out = run(
        [
            ffmpeg,
            "-hide_banner",
            "-nostats",
            "-i",
            str(audio_path),
            "-af",
            f"silencedetect=noise={_SILENCE_NOISE_DB}dB:d={_SILENCE_MIN_SECONDS}",
            "-f",
            "null",
            "-",
        ],
        desc="ffmpeg silence detection",
    )
    return parse_silences(out)


def parse_silences(ffmpeg_output: str) -> list[tuple[float, float]]:
    """Parse ``silencedetect`` log lines into (start, end) spans."""
    spans: list[tuple[float, float]] = []
    start: float | None = None
    for line in ffmpeg_output.splitlines():
        if m := _SILENCE_START_RE.search(line):
            start = max(0.0, float(m.group(1)))
        elif (m := _SILENCE_END_RE.search(line)) and start is not None:
            spans.append((start, float(m.group(1))))
            start = None
    if start is not None:  # silence runs to the end of the file
        spans.append((start, float("inf")))
    return spans


def drop_silent(segments: list[Segment], silences: list[tuple[float, float]]) -> list[Segment]:
    """Drop segments mostly covered by silence (likely hallucinated text)."""
    kept: list[Segment] = []
    for seg in segments:
        length = max(seg.end - seg.start, 1e-6)
        covered = sum(max(0.0, min(seg.end, b) - max(seg.start, a)) for a, b in silences)
        if covered / length < _SILENT_FRACTION:
            kept.append(seg)
    return kept


def collapse_repeats(segments: list[Segment], max_repeats: int = _MAX_REPEATS) -> list[Segment]:
    """Cap runs of consecutive identical segments (Whisper's repetition loops)."""
    kept: list[Segment] = []
    run_len = 0
    for seg in segments:
        norm = seg.text.strip().lower()
        if kept and norm == kept[-1].text.strip().lower():
            run_len += 1
            if run_len >= max_repeats:
                continue
        else:
            run_len = 0
        kept.append(seg)
    return kept


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
            # Without VAD, don't condition on previous text: that is what lets a
            # hallucination over silence repeat itself for minutes.
            seg_iter, info = model.transcribe(
                str(audio_path), vad_filter=vad, condition_on_previous_text=vad
            )
            out = [
                Segment(start=float(s.start), end=float(s.end), text=s.text.strip())
                for s in seg_iter
                if s.text.strip()
            ]
            log.info("Detected language: %s (p=%.2f)", info.language, info.language_probability)
            return out

        try:
            segments = _decode(vad=True)
        except Exception as exc:  # noqa: BLE001 - VAD backend is best-effort
            log.warning("VAD unavailable (%s); using ffmpeg silence detection instead.", exc)
            segments = _decode(vad=False)
            before = len(segments)
            segments = collapse_repeats(drop_silent(segments, detect_silences(audio_path)))
            if before != len(segments):
                log.info("Dropped %d segment(s) over silence or repeated", before - len(segments))

        if not segments:
            raise TranscriptionError("Whisper produced no transcript segments.")
        log.info("Transcribed %d segments", len(segments))
        return segments
