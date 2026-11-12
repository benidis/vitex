"""Sample frames from the video and reduce them to one per distinct visual state.

Tutorial videos are mostly static slides interrupted by changes. We sample on a
fixed interval and split the samples into *runs*: a run continues while
consecutive samples barely differ (codec noise, a moving cursor) and ends when a
meaningful fraction of pixels changes (new slide, new equation line, new build
step). From each run we keep a frame near its END — a slide that builds up step
by step is most complete just before it changes — choosing the sharpest of the
last few so a fade or motion blur isn't picked.

Change is measured per pixel on a small grayscale thumbnail, not with a global
perceptual hash: a hash summarizes overall layout and can rate two slides with
the same template but different equations as duplicates. Judging which states
are worth showing a reader is left to :mod:`vitex.media.triage`.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from vitex.models import Frame, _fmt_timestamp
from vitex.utils import require_tool, run

log = logging.getLogger("vitex")

_THUMB = (160, 90)  # thumbnail used for change detection
_PIXEL_DELTA = 24  # grey-level change below this is treated as noise
_TAIL = 3  # pick the sharpest of this many frames at the end of a run


def _label_for(timestamp: float) -> str:
    """Stable, LaTeX-safe figure label derived from the timestamp."""
    return "fig:t" + _fmt_timestamp(timestamp).replace(":", "")


def _sample(video_path: Path, frames_dir: Path, interval: float) -> list[tuple[int, Path]]:
    """Extract one JPEG every ``interval`` seconds; return (index, path) pairs."""
    ffmpeg = require_tool("ffmpeg")
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in [*frames_dir.glob("raw_*.jpg"), *frames_dir.glob("*frame_*.jpg")]:
        old.unlink()
    fps = 1.0 / interval
    run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"fps={fps:.6f}",
            "-q:v",
            "3",
            str(frames_dir / "raw_%05d.jpg"),
        ],
        desc="ffmpeg frame sampling",
    )
    # ffmpeg numbers output starting at 1; frame N corresponds to t≈(N-1)*interval.
    return sorted((i, p) for i, p in enumerate(sorted(frames_dir.glob("raw_*.jpg"))))


def _thumb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L").resize(_THUMB), dtype=np.int16)


def changed_fraction(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of thumbnail pixels whose grey level changed beyond noise."""
    return float(np.mean(np.abs(a - b) > _PIXEL_DELTA))


def sharpness(path: Path) -> float:
    """Edge-energy score; higher means crisper text and lines."""
    with Image.open(path) as im:
        grey = im.convert("L")
        grey.thumbnail((640, 640))
        return float(np.var(np.asarray(grey.filter(ImageFilter.FIND_EDGES), dtype=np.float32)))


def group_runs(thumbs: list[np.ndarray], threshold: float) -> list[list[int]]:
    """Split consecutive samples into runs of a stable visual state."""
    runs: list[list[int]] = []
    for i, t in enumerate(thumbs):
        if runs and changed_fraction(thumbs[i - 1], t) <= threshold:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs


def _dedup(
    raw: list[tuple[int, Path]], interval: float, change: float, max_per_minute: float, prefix: str
) -> list[Frame]:
    """Keep one representative frame per run; delete the rest."""
    if not raw:
        return []
    runs = group_runs([_thumb(p) for _, p in raw], change)

    keep: list[int] = []
    for r in runs:
        tail = r[-_TAIL:]
        # Latest first, so on equal sharpness the most complete state wins.
        keep.append(max(reversed(tail), key=lambda i: sharpness(raw[i][1])) if len(tail) > 1 else tail[0])

    # The cap scales with the video's length, so it only bites on pathological
    # footage (e.g. constant camera motion), never on long slide lectures.
    max_frames = max(1, math.ceil(max_per_minute * len(raw) * interval / 60))
    if len(keep) > max_frames:
        log.warning("%d visual states exceed the cap of %d; subsampling uniformly", len(keep), max_frames)
        step = len(keep) / max_frames
        keep = [keep[int(i * step)] for i in range(max_frames)]

    kept_set = set(keep)
    for i, (_, path) in enumerate(raw):
        if i not in kept_set:
            path.unlink(missing_ok=True)

    # Re-number and rename to a clean, contiguous, source-unique sequence.
    final: list[Frame] = []
    for new_idx, i in enumerate(keep):
        idx, path = raw[i]
        timestamp = idx * interval
        new_path = path.with_name(f"{prefix}frame_{new_idx:05d}.jpg")
        path.rename(new_path)
        final.append(Frame(index=new_idx, timestamp=timestamp, path=new_path, label=_label_for(timestamp)))
    return final


def extract_frames(
    video_path: Path,
    frames_dir: Path,
    *,
    interval: float,
    change: float,
    max_per_minute: float,
    prefix: str = "",
) -> list[Frame]:
    """Extract one frame per distinct visual state, in timestamp order.

    ``prefix`` makes file names unique across sources (they become figure names).
    """
    raw = _sample(video_path, frames_dir, interval)
    log.info("Sampled %d raw frames (every %.1fs)", len(raw), interval)
    frames = _dedup(raw, interval, change, max_per_minute, prefix)
    log.info("Kept %d distinct visual states", len(frames))
    return frames
