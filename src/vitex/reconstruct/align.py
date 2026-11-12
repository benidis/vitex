"""Attach extracted frames to the chunk whose time window contains them.

This is the payoff of having real timestamps everywhere: each distinct on-screen
visual is handed to the exact chapter that was being narrated when it appeared,
instead of being smeared across the document by a character-count heuristic.
"""

from __future__ import annotations

from vitex.models import Chunk, Frame


def _subsample(frames: list[Frame], limit: int) -> list[Frame]:
    """Evenly thin ``frames`` down to at most ``limit`` (keeps time spread)."""
    if limit <= 0 or len(frames) <= limit:
        return frames
    step = len(frames) / limit
    return [frames[int(i * step)] for i in range(limit)]


def assign_frames(chunks: list[Chunk], frames: list[Frame], max_images_per_chunk: int) -> None:
    """Populate ``chunk.frames`` in place by matching timestamps to windows.

    A frame is placed in the chunk whose ``[start, end]`` interval contains its
    timestamp. A frame outside every window (before the first, after the last,
    or in a pause between two chunks) goes to the nearest chunk, preferring the
    later one on a tie — a slide that appears in a pause usually introduces the
    narration that follows. Nothing is silently dropped.
    """
    if not chunks:
        return
    for c in chunks:
        c.frames = []

    def distance(c: Chunk, t: float) -> float:
        return max(c.start - t, t - c.end, 0.0)

    for f in frames:
        # min() keeps the first minimum, so iterate latest-first to break ties late.
        target = min(reversed(chunks), key=lambda c: distance(c, f.timestamp))
        target.frames.append(f)

    for c in chunks:
        c.frames.sort(key=lambda fr: fr.timestamp)
        c.frames = _subsample(c.frames, max_images_per_chunk)
