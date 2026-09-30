"""Transcriber interface shared by the local and AWS backends."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from vitex.models import Segment


class Transcriber(Protocol):
    """Turns an audio file into timestamped :class:`Segment` objects.

    Timestamps are essential downstream: they are what lets vitex attach the
    right on-screen frame to the right piece of narration.
    """

    def transcribe(self, audio_path: Path) -> list[Segment]: ...
