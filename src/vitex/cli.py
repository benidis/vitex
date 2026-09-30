"""Command-line interface: ``vitex run <url|path> [options]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vitex.config import Config
from vitex.errors import VitexError
from vitex.pipeline import Pipeline
from vitex.utils import log, setup_logging


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vitex",
        description="Reconstruct a tutorial video into a structured LaTeX document.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the full pipeline on a video URL or file.")
    run.add_argument("source", help="YouTube (or other yt-dlp) URL, or a local video file.")

    run.add_argument(
        "--outdir", type=Path,
        help="Parent directory for per-tutorial output dirs, i.e. <outdir>/<stem>/ (default: output).",
    )
    run.add_argument(
        "--workdir", type=Path, help="Where to cache intermediate artifacts (default: work)."
    )

    run.add_argument("--provider", choices=["bedrock", "anthropic"], help="Claude backend.")
    run.add_argument("--model", help="Model id / Bedrock inference-profile id.")
    run.add_argument("--fix-model", help="Model for the LaTeX repair pass (default: --model).")
    run.add_argument("--aws-region", help="AWS region for Bedrock / Transcribe.")
    run.add_argument(
        "--aws-profile",
        help="Named ~/.aws profile for the Bedrock client "
        "(default: ambient AWS credentials — AWS_PROFILE/env/instance role).",
    )

    run.add_argument("--transcriber", choices=["whisper", "aws"], help="Speech-to-text backend.")
    run.add_argument("--whisper-model", help="faster-whisper size (tiny..large-v3).")

    run.add_argument(
        "--frame-interval", type=float, help="Seconds between sampled frames (pre-dedup)."
    )
    run.add_argument("--max-images", type=int, help="Max frames sent per chapter.")
    run.add_argument(
        "--chunk-minutes", type=float, help="Transcript minutes per generated chapter."
    )

    run.add_argument(
        "--ground",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Ground the tutorial in real articles/papers via web+arXiv research "
        "and cite them (default: enabled). Use --no-ground for a fully offline run.",
    )
    run.add_argument("--no-pdf", action="store_true", help="Emit .tex only; skip PDF compilation.")
    run.add_argument(
        "--force", action="store_true", help="Ignore cached artifacts; recompute all stages."
    )
    run.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return p


def _config_from_args(args: argparse.Namespace) -> Config:
    """Start from env/.env, then apply any explicit CLI overrides."""
    cfg = Config.from_env()
    cfg.source = args.source
    overrides = {
        "outdir": args.outdir,
        "workdir": args.workdir,
        "provider": args.provider,
        "model": args.model,
        "fix_model": args.fix_model,
        "aws_region": args.aws_region,
        "aws_profile": args.aws_profile,
        "transcriber": args.transcriber,
        "whisper_model": args.whisper_model,
        "frame_interval": args.frame_interval,
        "max_images_per_chunk": args.max_images,
        "chunk_minutes": args.chunk_minutes,
    }
    for key, val in overrides.items():
        if val is not None:
            setattr(cfg, key, val)
    if args.no_pdf:
        cfg.build_pdf = False
    if args.ground is not None:  # only override env/.env when a flag was passed
        cfg.ground = args.ground
    cfg.force = args.force
    cfg.verbose = args.verbose
    cfg.__post_init__()  # re-coerce Path fields after overrides
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    setup_logging(verbose=getattr(args, "verbose", False))

    try:
        cfg = _config_from_args(args)
        result = Pipeline(cfg).run()
    except VitexError as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.error("Interrupted. Re-run the same command to resume from cache.")
        return 130

    print("\n" + "=" * 60)
    print(f"LaTeX : {result.tex_path}")
    if result.pdf_path:
        print(f"PDF   : {result.pdf_path}")
    else:
        print("PDF   : skipped (--no-pdf)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
