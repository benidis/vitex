"""Runtime configuration, resolved from (lowest to highest priority): built-in
defaults, then a ``.env``/environment, then explicit CLI overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import overload

from dotenv import load_dotenv


@overload
def _env(name: str, default: str) -> str: ...
@overload
def _env(name: str, default: None = None) -> str | None: ...
def _env(name: str, default: str | None = None) -> str | None:
    val = os.getenv(name)
    return val if val not in (None, "") else default


def _truthy(val: str) -> bool:
    return val.lower() not in ("0", "false", "no", "off")


@dataclass
class Config:
    # --- sources / io -------------------------------------------------------
    # Every source is a SEED: videos (YouTube/yt-dlp URLs or local files) and
    # documents (.tex/.md/.txt/.pdf, e.g. an existing tutorial to improve). All
    # of their content is carried into one fact-checked tutorial.
    sources: list[str] = field(default_factory=list)  # video URLs / paths
    docs: list[Path] = field(default_factory=list)  # document paths
    title: str | None = None  # tutorial title; proposed by the outline if unset
    # Name of the output directory and files (output/<name>/<name>.tex);
    # defaults to the title made filesystem-safe.
    name: str | None = None
    # Author's brief: free-text steering applied to every writing pass (scope,
    # focus, emphasis, audience, special instructions).
    guide: str | None = None
    workdir: Path = Path("work")  # intermediate, resumable artifacts
    outdir: Path = Path("output")  # final .tex / .pdf / figures

    # --- Claude backend -----------------------------------------------------
    provider: str = "bedrock"  # "bedrock" | "anthropic"
    model: str = "us.anthropic.claude-opus-4-8"
    fix_model: str | None = None  # LaTeX-repair model; falls back to `model`
    # Vision model that rates every distinct video frame. Called once per batch
    # of frames, so a small model (e.g. a Haiku profile) keeps it cheap.
    triage_model: str | None = None  # falls back to `model`
    aws_region: str = "us-east-1"
    # Named ~/.aws profile for every AWS call (Bedrock, and S3/Transcribe for the
    # aws transcriber). Read ONLY from VITEX_AWS_PROFILE and required whenever
    # AWS is used: there is deliberately no fallback to AWS_PROFILE, env keys,
    # or [default], so a run never silently uses unintended credentials.
    aws_profile: str | None = None
    anthropic_api_key: str | None = None
    max_output_tokens: int = 64000  # per response (streamed); longer output is continued
    # Reasoning effort for the main model: low | medium | high | xhigh | max.
    # Claude Opus 5.5 defaults to "medium"; tutorials benefit from "high".
    effort: str | None = "high"
    # Attempts per request for transient failures (throttling, 5xx, network).
    # Non-retryable errors (bad request, auth, missing model) fail immediately.
    llm_retries: int = 6
    # Mark the system prompt and the (image-heavy) user turn as cacheable, so the
    # repeated calls of a tool loop or a continuation don't re-bill those tokens.
    prompt_caching: bool = True
    # A single response is capped at max_output_tokens; long documents exceed it
    # and stop with reason "max_tokens". We then continue the assistant turn up to
    # this many extra times so the full document is produced instead of truncated.
    max_continuations: int = 8
    # Generations per pass when a response fails validation (empty, truncated,
    # malformed). After the last one the run stops — it never skips content.
    generation_attempts: int = 3

    # --- grounding / research ----------------------------------------------
    # When true, the model may call web/arXiv research tools while writing so the
    # tutorial is grounded in real sources and cites them. Disable for a fully
    # offline run. `research_max_iterations` caps tool-call rounds per generation
    # pass (each round may run several tools in parallel).
    ground: bool = True
    research_max_iterations: int = 16

    # --- transcription ------------------------------------------------------
    transcriber: str = "whisper"  # "whisper" | "aws"
    whisper_model: str = "small"
    transcribe_bucket: str | None = None  # required for the aws transcriber

    # --- frame extraction + triage ------------------------------------------
    frame_interval: float = 2.0  # seconds between sampled frames
    # A new visual state starts when more than this fraction of pixels changes
    # between consecutive samples. Small enough to catch one new equation line.
    frame_change: float = 0.015
    # Cap on distinct visual states, per minute of video, so long videos keep
    # proportionally many frames for triage (29 min of slides -> ~160 states).
    max_frames_per_minute: float = 20.0
    image_max_width: int = 1280  # downscale before sending to the model
    triage_batch: int = 8  # frames per triage call
    min_frame_value: int = 5  # 0-10 triage score a frame needs to be attached

    # --- seeds / reconstruction ---------------------------------------------
    chunk_minutes: float = 8.0  # transcript window per video seed unit
    max_images_per_chunk: int = 16
    doc_unit_chars: int = 24000  # max characters of a document per seed unit
    # Rounds of review (coverage + step-by-step rigor) and revision per section.
    review_rounds: int = 2
    # Outline attempts to place every content item before the remainder is
    # attached deterministically (content is never dropped).
    outline_attempts: int = 3

    # --- behavior flags -----------------------------------------------------
    build_pdf: bool = True
    force: bool = False  # ignore cached artifacts and recompute every stage
    verbose: bool = False

    def __post_init__(self) -> None:
        self.workdir = Path(self.workdir)
        self.outdir = Path(self.outdir)
        self.docs = [Path(d) for d in self.docs]

    @property
    def uses_aws(self) -> bool:
        return self.provider == "bedrock" or (bool(self.sources) and self.transcriber == "aws")

    @property
    def effective_fix_model(self) -> str:
        return self.fix_model or self.model

    @property
    def effective_triage_model(self) -> str:
        return self.triage_model or self.model

    @property
    def tutorial_dir(self) -> Path:
        """Cache dir for the cross-source passes (outline, sections, bibliography)."""
        return self.workdir / "tutorial"

    @classmethod
    def from_env(cls) -> "Config":
        """Build a Config from ``.env``/environment, leaving CLI to override."""
        load_dotenv()
        ground = _env("VITEX_GROUND")
        caching = _env("VITEX_PROMPT_CACHE")
        return cls(
            provider=_env("VITEX_PROVIDER", "bedrock"),
            model=_env("VITEX_MODEL", cls.model),
            fix_model=_env("VITEX_FIX_MODEL"),
            triage_model=_env("VITEX_TRIAGE_MODEL"),
            effort=_env("VITEX_EFFORT", "high"),
            aws_region=_env("AWS_REGION", "us-east-1"),
            aws_profile=_env("VITEX_AWS_PROFILE"),
            anthropic_api_key=_env("ANTHROPIC_API_KEY"),
            transcriber=_env("VITEX_TRANSCRIBER", "whisper"),
            whisper_model=_env("VITEX_WHISPER_MODEL", "small"),
            transcribe_bucket=_env("VITEX_TRANSCRIBE_BUCKET"),
            ground=_truthy(ground) if ground is not None else cls.ground,
            prompt_caching=_truthy(caching) if caching is not None else cls.prompt_caching,
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
        if self.uses_aws and not self.aws_profile:
            raise ValueError(
                "AWS is used (provider=bedrock or transcriber=aws) but VITEX_AWS_PROFILE is not "
                "set. Set it in .env to the ~/.aws profile to use, e.g. "
                "VITEX_AWS_PROFILE=aws-devtools-sci."
            )
        if self.uses_aws:
            import botocore.session

            available = botocore.session.Session().available_profiles
            if self.aws_profile not in available:
                raise ValueError(
                    f"VITEX_AWS_PROFILE={self.aws_profile!r} is not a profile in ~/.aws/config or "
                    "~/.aws/credentials."
                )
        if self.research_max_iterations < 1:
            raise ValueError("research_max_iterations must be at least 1.")
        if not self.sources and not self.docs:
            raise ValueError("Give at least one video source or --doc document.")
        for doc in self.docs:
            if not Path(doc).is_file():
                raise ValueError(f"Document not found: {doc}")
