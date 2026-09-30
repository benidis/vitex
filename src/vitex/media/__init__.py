"""Media acquisition and preprocessing: download, audio, frames."""

from vitex.media.audio import extract_audio
from vitex.media.download import SourceInfo, resolve_source
from vitex.media.frames import extract_frames

__all__ = ["SourceInfo", "resolve_source", "extract_audio", "extract_frames"]
