"""Small shared helpers: logging, shell commands, filenames, JSON I/O."""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

from vitex.errors import DependencyError, VitexError

log = logging.getLogger("vitex")


def setup_logging(verbose: bool = False) -> None:
    """Configure a single line-oriented logger for the CLI.

    ``verbose`` raises only vitex's own logger to DEBUG; chatty third-party
    libraries (httpx, botocore, huggingface, faster-whisper) stay at WARNING so
    debug output remains readable.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    for noisy in ("httpx", "httpcore", "botocore", "boto3", "urllib3", "huggingface_hub", "faster_whisper", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def require_tool(name: str) -> str:
    """Return the path to external CLI ``name`` or raise a helpful error."""
    path = shutil.which(name)
    if not path:
        raise DependencyError(
            f"Required tool '{name}' was not found on PATH. "
            f"Install it (e.g. `brew install {name}` or your OS package manager)."
        )
    return path


def run(cmd: Sequence[str], *, desc: str = "") -> str:
    """Run a command, streaming nothing; raise ``VitexError`` on failure.

    Returns combined stdout+stderr on success so callers can parse it.
    """
    log.debug("exec: %s", " ".join(map(str, cmd)))
    proc = subprocess.run(
        list(map(str, cmd)),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-40:])
        raise VitexError(
            f"{desc or ' '.join(map(str, cmd[:2]))} failed (exit {proc.returncode}).\n{tail}"
        )
    return proc.stdout


def safe_stem(text: str, fallback: str = "tutorial") -> str:
    """Turn arbitrary text (e.g. a video title) into a filesystem-safe stem."""
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text[:100] or fallback


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
