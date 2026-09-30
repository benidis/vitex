"""Sample frames from the video and collapse visually-identical ones.

Tutorial videos are mostly static slides interrupted by occasional changes, so
we sample on a fixed interval and then drop frames that are perceptually near
their predecessor. What survives is roughly "one frame per distinct visual
state" — exactly the images worth spending vision tokens on.
"""

from __future__ import annotations

import logging
from pathlib import Path

import imagehash
from PIL import Image

from vitex.models import Frame, _fmt_timestamp
from vitex.utils import require_tool, run

log = logging.getLogger("vitex")


def _label_for(timestamp: float) -> str:
    """Stable, LaTeX-safe figure label derived from the timestamp."""
    return "fig:t" + _fmt_timestamp(timestamp).replace(":", "")


def _sample(video_path: Path, frames_dir: Path, interval: float) -> list[tuple[int, Path]]:
    """Extract one JPEG every ``interval`` seconds; return (index, path) pairs."""
    ffmpeg = require_tool("ffmpeg")
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("raw_*.jpg"):
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


def _dedup(
    raw: list[tuple[int, Path]], interval: float, phash_distance: int, max_frames: int
) -> list[Frame]:
    """Keep frames that differ perceptually from the previously kept one."""
    kept: list[Frame] = []
    last_hash: imagehash.ImageHash | None = None
    for idx, path in raw:
        with Image.open(path) as im:
            h = imagehash.phash(im)
        if last_hash is not None and (h - last_hash) <= phash_distance:
            path.unlink(missing_ok=True)  # near-duplicate; discard
            continue
        last_hash = h
        timestamp = idx * interval
        kept.append(Frame(index=idx, timestamp=timestamp, path=path, label=_label_for(timestamp)))

    if len(kept) > max_frames:
        # Uniformly subsample to the cap; delete the frames we drop.
        step = len(kept) / max_frames
        keep_idx = {int(i * step) for i in range(max_frames)}
        pruned: list[Frame] = []
        for i, f in enumerate(kept):
            if i in keep_idx:
                pruned.append(f)
            else:
                f.path.unlink(missing_ok=True)
        kept = pruned

    # Re-number and rename to a clean, contiguous sequence for the output figures.
    final: list[Frame] = []
    for new_idx, f in enumerate(kept):
        new_path = f.path.with_name(f"frame_{new_idx:05d}.jpg")
        f.path.rename(new_path)
        final.append(
            Frame(
                index=new_idx,
                timestamp=f.timestamp,
                path=new_path,
                label=_label_for(f.timestamp),
            )
        )
    return final


def extract_frames(
    video_path: Path,
    frames_dir: Path,
    *,
    interval: float,
    phash_distance: int,
    max_frames: int,
) -> list[Frame]:
    """Extract and de-duplicate frames, returning them in timestamp order."""
    raw = _sample(video_path, frames_dir, interval)
    log.info("Sampled %d raw frames (every %.1fs)", len(raw), interval)
    frames = _dedup(raw, interval, phash_distance, max_frames)
    log.info("Kept %d distinct frames after de-duplication", len(frames))
    return frames
