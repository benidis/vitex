"""Resolve a video source into a local video file plus metadata.

Accepts either a local path (used as-is) or a URL (downloaded with yt-dlp).
Each source gets its own work directory (``<workroot>/<slug>/``) named after
its identity, so several videos can be processed side by side and re-runs with
the same sources reuse each one's cached artifacts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from vitex.errors import DownloadError
from vitex.utils import read_json, require_tool, run, write_json

log = logging.getLogger("vitex")

_VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}

# Identity of the source the cached work dir was built from. Every artifact
# under work/ (video, audio, transcript, frames, drafts, ...) is derived from a
# single source, so if the source changes the whole cache is stale.
_SOURCE_MARKER = "source.json"


def _stale_cache(workdir: Path, identity: str) -> bool:
    """True if work/ was last built from a *different* source than ``identity``."""
    marker = workdir / _SOURCE_MARKER
    if not workdir.exists():
        return False
    if not marker.exists():
        # No marker: either a fresh work dir, or one from before this guard
        # existed. Treat a non-empty dir as stale so we never reuse mystery
        # artifacts of unknown provenance.
        return any(workdir.iterdir())
    try:
        return str(read_json(marker).get("identity")) != identity
    except Exception:
        return True


def _source_dir(workroot: Path, slug: str) -> Path:
    """``workroot/slug``, moving it there from the pre-``sources/`` layout
    (``work/<slug>``) if needed, so nothing is downloaded or transcribed again."""
    workdir = workroot / slug
    legacy = workroot.parent / slug
    if not workdir.exists() and legacy.is_dir():
        workroot.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy), str(workdir))
        log.info("Moved %s -> %s (new work layout)", legacy, workdir)
    return workdir


def _clear_video(workdir: Path) -> None:
    """Remove any downloaded video file so yt-dlp cannot silently reuse it."""
    for p in workdir.glob("video.*"):
        if p.suffix.lower() in _VIDEO_SUFFIXES:
            p.unlink()


def _slug(prefix: str, text: str, unique: str) -> str:
    """Filesystem- and LaTeX-safe source id, e.g. ``yt-Y-o545eYjXM``."""
    clean = re.sub(r"[^A-Za-z0-9-]+", "-", text).strip("-")[:30]
    digest = hashlib.sha1(unique.encode()).hexdigest()[:6]
    return f"{prefix}-{clean}-{digest}" if clean else f"{prefix}-{digest}"


@dataclass
class SourceInfo:
    video_path: Path
    title: str
    description: str
    duration: float  # seconds; 0.0 if unknown
    slug: str  # unique source id; also the figure-name prefix of its frames
    workdir: Path  # this source's own cache directory
    changed: bool = False  # True if the source differs from the cached work dir


def _probe_duration(video_path: Path) -> float:
    """Return duration in seconds via ffprobe (0.0 if it can't be read)."""
    ffprobe = require_tool("ffprobe")
    out = run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        desc="ffprobe duration",
    ).strip()
    try:
        return float(out.splitlines()[-1])
    except (ValueError, IndexError):
        return 0.0


def _resolve_local(path: Path, workroot: Path) -> SourceInfo:
    if not path.exists():
        raise DownloadError(f"Local source not found: {path}")
    identity = f"file:{path.resolve()}"
    slug = _slug("file", path.stem, identity)
    workdir = _source_dir(workroot, slug)
    changed = _stale_cache(workdir, identity)
    workdir.mkdir(parents=True, exist_ok=True)
    write_json(workdir / _SOURCE_MARKER, {"identity": identity, "path": str(path)})
    return SourceInfo(
        video_path=path,
        title=path.stem,
        description="",
        duration=_probe_duration(path),
        slug=slug,
        workdir=workdir,
        changed=changed,
    )


def _resolve_url(url: str, workroot: Path, force: bool) -> SourceInfo:
    ytdlp = require_tool("yt-dlp")
    require_tool("ffmpeg")  # needed by yt-dlp to merge video+audio

    # Fetch metadata first (cheap, no download) so we can identify the source
    # *before* touching the possibly-stale cached video file.
    meta_raw = run(
        [ytdlp, "--dump-single-json", "--no-warnings", "--skip-download", url],
        desc="yt-dlp metadata",
    )
    metadata: dict = {}
    for line in meta_raw.splitlines():
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            metadata = candidate
            break

    video_id = str(metadata.get("id") or "")
    identity = f"youtube:{video_id}" if video_id else f"url:{url}"
    slug = f"yt-{re.sub(r'[^A-Za-z0-9-]+', '-', video_id)}" if video_id else _slug("url", "", identity)
    workdir = _source_dir(workroot, slug)
    workdir.mkdir(parents=True, exist_ok=True)
    changed = _stale_cache(workdir, identity)

    # If the cached video belongs to a different source (or --force), delete it
    # so yt-dlp actually re-downloads instead of skipping an already-present
    # file. Without this, a stale video.mp4 from a previous run is silently
    # reused and every downstream stage processes the wrong video.
    if force or changed:
        _clear_video(workdir)

    write_json(workdir / "metadata.json", metadata)
    write_json(workdir / _SOURCE_MARKER, {"identity": identity, "url": url})

    out_tmpl = str(workdir / "video.%(ext)s")
    run(
        [
            ytdlp,
            "-f",
            "bv*+ba/b",
            "--merge-output-format",
            "mp4",
            "-o",
            out_tmpl,
            "--no-warnings",
            url,
        ],
        desc="yt-dlp download",
    )

    candidates = sorted(p for p in workdir.glob("video.*") if p.suffix.lower() in _VIDEO_SUFFIXES)
    if not candidates:
        raise DownloadError("yt-dlp finished but no video file was produced.")
    video_path = candidates[0]

    duration = float(metadata.get("duration") or 0.0)
    if duration <= 0:
        duration = _probe_duration(video_path)

    return SourceInfo(
        video_path=video_path,
        title=str(metadata.get("title") or video_path.stem),
        description=str(metadata.get("description") or ""),
        duration=duration,
        slug=slug,
        workdir=workdir,
        changed=changed,
    )


def resolve_source(source: str, workroot: Path, force: bool = False) -> SourceInfo:
    """Return a :class:`SourceInfo` for a local path or a downloadable URL.

    The source's artifacts live in ``SourceInfo.workdir`` under ``workroot``.
    ``SourceInfo.changed`` is True when that dir was last built from a different
    source; callers should then invalidate its cached artifacts.
    """
    p = Path(source)
    if p.exists():
        log.info("Using local video: %s", p)
        return _resolve_local(p, workroot)
    if source.startswith(("http://", "https://", "www.")):
        log.info("Resolving video: %s", source)
        return _resolve_url(source, workroot, force)
    raise DownloadError(f"Source {source!r} is neither an existing file nor an http(s) URL.")
