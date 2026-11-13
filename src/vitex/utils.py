"""Small shared helpers: logging, shell commands, filenames, JSON I/O."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar

from vitex.errors import DependencyError, VitexError

log = logging.getLogger("vitex")

T = TypeVar("T")
R = TypeVar("R")


def run_parallel(fn: Callable[[T], R], items: Sequence[T], workers: int) -> list[R]:
    """Apply ``fn`` to ``items`` on up to ``workers`` threads; results keep input order.

    Every task runs to completion even if another one fails, so the work that
    did finish is cached for the next run; the first error is raised afterwards.
    """
    if workers <= 1 or len(items) <= 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
        futures = [pool.submit(fn, item) for item in items]
        errors = [f.exception() for f in futures]
    first = next((e for e in errors if e is not None), None)
    if first is not None:
        failed = sum(e is not None for e in errors)
        if failed > 1:
            log.error("%d of %d parallel tasks failed; raising the first", failed, len(items))
        raise first
    return [f.result() for f in futures]


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
    for noisy in (
        "httpx", "httpx2", "httpcore", "httpcore2", "anthropic", "botocore", "boto3",
        "urllib3", "huggingface_hub", "faster_whisper", "filelock",
    ):
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


_LATEX_CMD_RE = re.compile(r"\\[A-Za-z@]+\*?|[{}$&^_~\\%]")


_LATEX_ENV_RE = re.compile(r"\\(?:begin|end)\{[^}]*\}(?:\[[^\]]*\])?")


def word_count(text: str) -> int:
    """Approximate prose word count of plain text or LaTeX (commands and
    environment names ignored)."""
    return len(_LATEX_CMD_RE.sub(" ", _LATEX_ENV_RE.sub(" ", text)).split())


def safe_stem(text: str, fallback: str = "tutorial") -> str:
    """Turn arbitrary text (e.g. a video title) into a filesystem-safe stem."""
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text[:100] or fallback


def fingerprint(*parts: Any) -> str:
    """Stable hash of everything that determines a cached artifact's content."""
    blob = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _key_path(path: Path) -> Path:
    return path.with_name(path.name + ".key")


def read_cached(path: Path, key: str, force: bool = False) -> str | None:
    """Return the cached text at ``path`` if it was produced from ``key``.

    Each artifact has a ``<name>.key`` sidecar holding the fingerprint of the
    inputs it was built from, so a change in config, prompts, or upstream
    content invalidates exactly the stages that depend on it.
    """
    if force or not path.exists():
        return None
    kp = _key_path(path)
    if not kp.exists() or kp.read_text(encoding="utf-8").strip() != key:
        return None
    return path.read_text(encoding="utf-8")


def write_cached(path: Path, key: str, text: str) -> None:
    """Write ``text`` to ``path`` and record the input fingerprint beside it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    _key_path(path).write_text(key, encoding="utf-8")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
