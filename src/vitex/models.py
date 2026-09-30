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
class ChapterDraft:
    """LaTeX body fragment produced for a single :class:`Chunk`."""

    index: int
    latex: str

    def to_dict(self) -> dict:
        return {"index": self.index, "latex": self.latex}

    @classmethod
    def from_dict(cls, d: dict) -> "ChapterDraft":
        return cls(index=int(d["index"]), latex=str(d["latex"]))
