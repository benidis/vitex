from vitex.models import Segment
from vitex.reconstruct.segmenter import build_chunks


def _segs(*spans):
    return [Segment(start=a, end=b, text=f"seg {a}-{b}") for a, b in spans]


def test_empty_returns_no_chunks():
    assert build_chunks([], 60) == []


def test_single_chunk_when_within_window():
    segs = _segs((0, 5), (5, 10), (10, 15))
    chunks = build_chunks(segs, chunk_seconds=60)
    assert len(chunks) == 1
    assert chunks[0].start == 0 and chunks[0].end == 15
    assert len(chunks[0].segments) == 3


def test_splits_when_window_exceeded():
    # 4 x 10s segments, window 25s: two segments (20s span) fit, three (30s) don't
    # -> [0-10,10-20] then [20-30,30-40]
    segs = _segs((0, 10), (10, 20), (20, 30), (30, 40))
    chunks = build_chunks(segs, chunk_seconds=25)
    assert len(chunks) == 2
    assert [len(c.segments) for c in chunks] == [2, 2]
    assert chunks[0].index == 0 and chunks[1].index == 1


def test_each_segment_alone_when_window_too_small():
    # window smaller than a single 10s segment span -> one segment per chunk
    segs = _segs((0, 10), (10, 20), (20, 30))
    chunks = build_chunks(segs, chunk_seconds=15)
    assert [len(c.segments) for c in chunks] == [1, 1, 1]


def test_segments_never_split():
    segs = _segs((0, 100))  # one long segment, tiny window
    chunks = build_chunks(segs, chunk_seconds=5)
    assert len(chunks) == 1
    assert chunks[0].segments == segs


def test_transcript_has_timestamp_prefixes():
    chunks = build_chunks(_segs((0, 5), (65, 70)), chunk_seconds=600)
    text = chunks[0].transcript
    assert "[00:00:00]" in text and "[00:01:05]" in text
