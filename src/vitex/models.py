"""Interchange data contracts passed between pipeline stages.

These are deliberately plain dataclasses with ``to_dict``/``from_dict`` helpers
so every stage can serialize its output to JSON in the work directory. That
serialization is what makes the pipeline resumable: an expensive stage (e.g.
transcription) is skipped if its artifact already exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _fmt_timestamp(seconds: float) -> str:
    """Render ``seconds`` as ``HH:MM:SS`` (used in prompts and labels)."""
    s = int(round(seconds))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


@dataclass
class Segment:
    """One timestamped span of transcript text.

    ``start``/``end`` are seconds from the beginning of the video. ``text`` is
    the spoken words in that span.
    """

    start: float
    end: float
    text: str

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "text": self.text}

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        return cls(start=float(d["start"]), end=float(d["end"]), text=str(d["text"]))


@dataclass
class Frame:
    """A single extracted, de-duplicated video frame.

    ``timestamp`` is seconds into the video. ``path`` points at the JPEG on
    disk. ``label`` is a stable LaTeX-safe identifier (e.g. ``fig:t000325``)
    that the model is told to use in ``\\ref`` / ``\\label``.
    """

    index: int
    timestamp: float
    path: Path
    label: str

    @property
    def timestamp_text(self) -> str:
        return _fmt_timestamp(self.timestamp)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "path": str(self.path),
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Frame":
        return cls(
            index=int(d["index"]),
            timestamp=float(d["timestamp"]),
            path=Path(d["path"]),
            label=str(d["label"]),
        )


@dataclass
class Chunk:
    """A contiguous slice of the tutorial handed to the model as one chapter.

    A chunk bundles the transcript ``segments`` that fall inside its time window
    with the ``frames`` captured during that same window, so the model sees the
    words and the on-screen visuals together.
    """

    index: int
    start: float
    end: float
    segments: list[Segment] = field(default_factory=list)
    frames: list[Frame] = field(default_factory=list)

    @property
    def transcript(self) -> str:
        """Transcript with per-line ``[HH:MM:SS]`` prefixes for alignment."""
        return "\n".join(f"[{_fmt_timestamp(s.start)}] {s.text.strip()}" for s in self.segments)


@dataclass
class FrameNote:
    """A vision model's assessment of one distinct frame (see media.triage).

    ``value`` (0-10) is how much a reader would learn from seeing this exact
    image in the tutorial. ``superseded_by`` names a later frame showing the same
    content more completely (e.g. the finished version of a slide being built
    up). ``text`` is the legible on-screen text/equations, so the content of a
    frame reaches the writer even when the image itself is not attached.
    """

    name: str  # figure file name
    kind: str
    legible: bool
    complete: bool
    value: int
    superseded_by: str | None
    description: str
    text: str
    reason: str

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "FrameNote":
        return cls(
            name=str(d["name"]),
            kind=str(d.get("kind", "other")),
            legible=bool(d.get("legible", True)),
            complete=bool(d.get("complete", True)),
            value=int(d.get("value", 0)),
            superseded_by=d.get("superseded_by") or None,
            description=str(d.get("description", "")),
            text=str(d.get("text", "")),
            reason=str(d.get("reason", "")),
        )


@dataclass
class Figure:
    """An image the tutorial may include: a video frame or a document figure.

    ``name`` is the globally unique file name the model writes in
    ``\\includegraphics`` (and the name it is staged under in ``figures/``).
    """

    name: str
    path: Path
    hint: str = ""  # e.g. "shown at 00:12:05" or the original caption
    note: FrameNote | None = None


@dataclass
class SeedUnit:
    """One slice of one source, drafted on its own: a video chunk or a part of
    a document. Seeds are raw material — every piece of their content must reach
    the tutorial, fact-checked and expanded.
    """

    id: str  # globally unique, e.g. "yt-abc123-c02" / "doc-notes-u01"
    kind: str  # "video" | "document"
    source_title: str
    position: str  # human-readable place in the source, e.g. "part 2 of 4"
    text: str  # timestamped transcript, or document text
    figures: list[Figure] = field(default_factory=list)  # attached as images
    visual_notes: str = ""  # on-screen timeline from frame triage (video only)
