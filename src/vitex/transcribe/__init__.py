"""Transcription backends and a factory that picks one from config."""

from __future__ import annotations

from vitex.config import Config
from vitex.transcribe.aws_transcribe import AWSTranscriber
from vitex.transcribe.base import Transcriber
from vitex.transcribe.whisper_local import WhisperTranscriber

__all__ = ["Transcriber", "WhisperTranscriber", "AWSTranscriber", "build_transcriber"]


def build_transcriber(cfg: Config) -> Transcriber:
    """Instantiate the transcriber selected by ``cfg.transcriber``."""
    if cfg.transcriber == "whisper":
        return WhisperTranscriber(model_size=cfg.whisper_model)
    if cfg.transcriber == "aws":
        assert cfg.transcribe_bucket and cfg.aws_profile  # guaranteed by Config.validate()
        return AWSTranscriber(bucket=cfg.transcribe_bucket, region=cfg.aws_region, profile=cfg.aws_profile)
    raise ValueError(f"Unknown transcriber: {cfg.transcriber!r}")
