from pathlib import Path

from vitex.models import Chunk, Frame, Segment
from vitex.reconstruct.align import assign_frames


def _chunk(idx, start, end):
    return Chunk(index=idx, start=start, end=end, segments=[Segment(start, end, "x")])


def _frame(i, t):
    return Frame(index=i, timestamp=t, path=Path(f"/tmp/frame_{i}.jpg"), label=f"fig:{i}")


def test_frames_land_in_containing_window():
    chunks = [_chunk(0, 0, 30), _chunk(1, 30, 60)]
    frames = [_frame(0, 5), _frame(1, 25), _frame(2, 45)]
    assign_frames(chunks, frames, max_images_per_chunk=10)
    assert [f.index for f in chunks[0].frames] == [0, 1]
    assert [f.index for f in chunks[1].frames] == [2]


def test_edge_frames_are_not_dropped():
    chunks = [_chunk(0, 10, 20)]
    frames = [_frame(0, 0), _frame(1, 999)]  # before first / after last
    assign_frames(chunks, frames, max_images_per_chunk=10)
    assert len(chunks[0].frames) == 2


def test_subsample_respects_cap_and_keeps_order():
    chunks = [_chunk(0, 0, 100)]
    frames = [_frame(i, i) for i in range(20)]
    assign_frames(chunks, frames, max_images_per_chunk=5)
    kept = chunks[0].frames
    assert len(kept) == 5
    assert kept == sorted(kept, key=lambda f: f.timestamp)


def test_reassignment_is_idempotent():
    chunks = [_chunk(0, 0, 60)]
    frames = [_frame(0, 5)]
    assign_frames(chunks, frames, 10)
    assign_frames(chunks, frames, 10)
    assert len(chunks[0].frames) == 1
