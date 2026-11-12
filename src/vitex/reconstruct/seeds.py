"""Turn prepared sources into seed units — the slices that get drafted.

A video becomes one unit per time chunk: its transcript, the highest-rated
frames of that window as attached images, and an on-screen timeline of every
frame's content. A document becomes one unit per part (split at section
boundaries), with the figures that part references.
"""

from __future__ import annotations

from pathlib import Path

from vitex.config import Config
from vitex.media.documents import Document, split_document
from vitex.media.download import SourceInfo
from vitex.media.triage import select_frames, visual_timeline
from vitex.models import Figure, Frame, FrameNote, Segment, SeedUnit, _fmt_timestamp
from vitex.reconstruct.align import assign_frames
from vitex.reconstruct.segmenter import build_chunks

# Formats the model can be shown as images; others (e.g. PDF/EPS figures from a
# LaTeX document) are still staged and may be included by file name.
VIEWABLE = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def viewable(fig: Figure) -> bool:
    return Path(fig.name).suffix.lower() in VIEWABLE


def video_units(
    info: SourceInfo,
    segments: list[Segment],
    frames: list[Frame],
    notes: dict[str, FrameNote],
    cfg: Config,
) -> list[SeedUnit]:
    chunks = build_chunks(segments, cfg.chunk_minutes * 60)
    assign_frames(chunks, frames, max_images_per_chunk=0)  # every frame, no thinning
    units: list[SeedUnit] = []
    for c in chunks:
        chosen = select_frames(c.frames, notes, cfg.max_images_per_chunk, cfg.min_frame_value)
        figures = [
            Figure(name=f.path.name, path=f.path, hint=f"shown at {f.timestamp_text}", note=notes.get(f.path.name))
            for f in chosen
        ]
        units.append(
            SeedUnit(
                id=f"{info.slug}-c{c.index:02d}",
                kind="video",
                source_title=info.title,
                position=f"part {c.index + 1} of {len(chunks)} ({_fmt_timestamp(c.start)}-{_fmt_timestamp(c.end)})",
                text=c.transcript,
                figures=figures,
                visual_notes=visual_timeline(c.frames, notes, {f.name for f in figures}),
            )
        )
    return units


def document_units(doc: Document, cfg: Config) -> list[SeedUnit]:
    parts = split_document(doc, cfg.doc_unit_chars)
    return [
        SeedUnit(
            id=f"{doc.slug}-u{i:02d}",
            kind="document",
            source_title=doc.title,
            position=f"part {i + 1} of {len(parts)}",
            text=part,
            figures=[f for f in doc.figures if f.name in part],
        )
        for i, part in enumerate(parts)
    ]


def figure_lines(figures: list[Figure]) -> str:
    """One line per figure for prompts: exact name, provenance, and content."""
    lines = []
    for f in figures:
        line = f"  - {f.name} | {f.hint}"
        if f.note is not None:
            line += f" | {f.note.kind}, value {f.note.value}/10: {f.note.description.strip()}"
        if not viewable(f):
            line += " | (not attached as an image; keep its role from the source text)"
        lines.append(line)
    return "\n".join(lines)
