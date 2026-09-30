"""Runtime configuration, resolved from (lowest to highest priority): built-in
defaults, then a ``.env``/environment, then explicit CLI overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _env(name: str, default: str | None = None) -> str | None:
    val = os.getenv(name)
    return val if val not in (None, "") else default


@dataclass
class Config:
    # --- source / io --------------------------------------------------------
    source: str = ""  # YouTube URL or local video path
    workdir: Path = Path("work")  # intermediate, resumable artifacts
    outdir: Path = Path("output")  # final .tex / .pdf / figures

    # --- Claude backend -----------------------------------------------------
    provider: str = "bedrock"  # "bedrock" | "anthropic"
    model: str = "us.anthropic.claude-opus-4-8"
    fix_model: str | None = None  # LaTeX-repair model; falls back to `model`
    aws_region: str = "us-east-1"
    # Named ~/.aws profile for the Bedrock client. None uses the ambient AWS
    # credential chain (AWS_PROFILE / env vars / instance role / [default]).
    aws_profile: str | None = None
    anthropic_api_key: str | None = None
    max_output_tokens: int = 8192
    llm_retries: int = 4
    # A single response is capped at max_output_tokens; long documents exceed it
    # and stop with reason "max_tokens". We then continue the assistant turn up to
    # this many extra times so the full document is produced instead of truncated.
    max_continuations: int = 8

    # --- grounding / research ----------------------------------------------
    # When true, the model may call web/arXiv research tools while writing so the
    # tutorial is grounded in real sources and cites them. Disable for a fully
    # offline run. `research_max_iterations` caps tool calls per generation pass.
    ground: bool = True
    research_max_iterations: int = 8

    # --- transcription ------------------------------------------------------
    transcriber: str = "whisper"  # "whisper" | "aws"
    whisper_model: str = "small"
    transcribe_bucket: str | None = None  # required for the aws transcriber

    # --- frame extraction ---------------------------------------------------
    frame_interval: float = 4.0  # seconds between sampled frames (pre-dedup)
    phash_distance: int = 6  # min perceptual-hash Hamming dist to keep a frame
    max_frames: int = 400  # global cap after dedup
    image_max_width: int = 1280  # downscale before sending to the model

    # --- chunking / reconstruction -----------------------------------------
    chunk_minutes: float = 8.0  # transcript window per generated chapter
    max_images_per_chunk: int = 12

    # --- behavior flags -----------------------------------------------------
    build_pdf: bool = True
    keep_frames: bool = True
    force: bool = False  # ignore cached artifacts and recompute every stage
    verbose: bool = False

    def __post_init__(self) -> None:
        self.workdir = Path(self.workdir)
        self.outdir = Path(self.outdir)

    @property
    def effective_fix_model(self) -> str:
        return self.fix_model or self.model

    @property
    def frames_dir(self) -> Path:
        return self.workdir / "frames"

    @classmethod
    def from_env(cls) -> "Config":
        """Build a Config from ``.env``/environment, leaving CLI to override."""
        load_dotenv()
        ground = _env("VITEX_GROUND")
        return cls(
            provider=_env("VITEX_PROVIDER", "bedrock"),
            model=_env("VITEX_MODEL", cls.model),
            fix_model=_env("VITEX_FIX_MODEL"),
            aws_region=_env("AWS_REGION", "us-east-1"),
            aws_profile=_env("VITEX_AWS_PROFILE") or _env("AWS_PROFILE"),
            anthropic_api_key=_env("ANTHROPIC_API_KEY"),
            transcriber=_env("VITEX_TRANSCRIBER", "whisper"),
            whisper_model=_env("VITEX_WHISPER_MODEL", "small"),
            transcribe_bucket=_env("VITEX_TRANSCRIBE_BUCKET"),
            ground=ground.lower() not in ("0", "false", "no") if ground is not None else cls.ground,
        )

    def validate(self) -> None:
        """Fail fast on internally inconsistent configuration."""
        if self.provider not in ("bedrock", "anthropic"):
            raise ValueError(f"Unknown provider: {self.provider!r}")
        if self.provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError("provider=anthropic requires ANTHROPIC_API_KEY (env or .env).")
        if self.transcriber not in ("whisper", "aws"):
            raise ValueError(f"Unknown transcriber: {self.transcriber!r}")
        if self.transcriber == "aws" and not self.transcribe_bucket:
            raise ValueError("transcriber=aws requires VITEX_TRANSCRIBE_BUCKET (an S3 bucket).")
