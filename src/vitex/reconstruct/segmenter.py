"""Group timestamped transcript segments into chapter-sized time windows.

Chunking by *time* (rather than by character count) is what keeps narration and
on-screen frames aligned: a chunk covers a real interval of the video, so the
frames from that same interval belong to it.
"""

from __future__ import annotations

from vitex.models import Chunk, Segment


def build_chunks(segments: list[Segment], chunk_seconds: float) -> list[Chunk]:
    """Partition ``segments`` into contiguous chunks no longer than the window.

    Segments are never split; a chunk closes once adding the next segment would
    push its span past ``chunk_seconds``.
    """
    if not segments:
        return []
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")

    chunks: list[Chunk] = []
    current: list[Segment] = []
    window_start = segments[0].start

    for seg in segments:
        if current and (seg.end - window_start) > chunk_seconds:
            chunks.append(
                Chunk(
                    index=len(chunks),
                    start=current[0].start,
                    end=current[-1].end,
                    segments=current,
                )
            )
            current = []
            window_start = seg.start
        current.append(seg)

    if current:
        chunks.append(
            Chunk(
                index=len(chunks),
                start=current[0].start,
                end=current[-1].end,
                segments=current,
            )
        )
    return chunks
