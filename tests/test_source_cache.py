"""Regression tests for work/ cache-staleness detection in media.download.

Guards the bug where a leftover video.mp4 from a previous run was silently
reused for a different source URL, so the transcript/frames came from the wrong
video while only the title was fresh.
"""

from vitex.media.download import _SOURCE_MARKER, _clear_video, _stale_cache
from vitex.utils import write_json


def test_empty_workdir_not_stale(tmp_path):
    # A brand-new (empty) work dir has nothing to invalidate.
    assert _stale_cache(tmp_path, "youtube:abc123") is False


def test_populated_workdir_without_marker_is_stale(tmp_path):
    # Artifacts of unknown provenance (no marker) must not be trusted.
    (tmp_path / "video.mp4").write_bytes(b"stale")
    assert _stale_cache(tmp_path, "youtube:abc123") is True


def test_matching_identity_not_stale(tmp_path):
    write_json(tmp_path / _SOURCE_MARKER, {"identity": "youtube:abc123"})
    assert _stale_cache(tmp_path, "youtube:abc123") is False


def test_different_identity_is_stale(tmp_path):
    write_json(tmp_path / _SOURCE_MARKER, {"identity": "youtube:OLDvideo"})
    assert _stale_cache(tmp_path, "youtube:NEWvideo") is True


def test_clear_video_removes_only_video_files(tmp_path):
    (tmp_path / "video.mp4").write_bytes(b"v")
    (tmp_path / "video.webm").write_bytes(b"v")
    (tmp_path / "audio.wav").write_bytes(b"a")
    (tmp_path / "transcript.json").write_text("[]")
    _clear_video(tmp_path)
    assert not (tmp_path / "video.mp4").exists()
    assert not (tmp_path / "video.webm").exists()
    # Non-video artifacts are left for the caller's force logic to overwrite.
    assert (tmp_path / "audio.wav").exists()
    assert (tmp_path / "transcript.json").exists()
