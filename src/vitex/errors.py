"""Exception types raised across the vitex pipeline."""

from __future__ import annotations


class VitexError(Exception):
    """Base class for all vitex errors."""


class DependencyError(VitexError):
    """A required external tool or optional dependency is missing."""


class DownloadError(VitexError):
    """Failed to download or locate the source video."""


class TranscriptionError(VitexError):
    """Speech-to-text failed or produced no usable segments."""


class LLMError(VitexError):
    """A Claude request failed after exhausting retries."""


class BuildError(VitexError):
    """LaTeX failed to compile even after the auto-repair pass."""
