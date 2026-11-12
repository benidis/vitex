"""Tests for visual-state frame selection and triage-based picking (no ffmpeg, no network)."""

import numpy as np
from PIL import Image, ImageDraw

from vitex.config import Config
from vitex.media.frames import _dedup, changed_fraction, group_runs
from vitex.media.triage import attachable, select_frames, triage_frames, visual_timeline
from vitex.models import Frame, FrameNote, Segment


def _slide(path, lines, blur=False):
    im = Image.new("RGB", (320, 180), "white")
    d = ImageDraw.Draw(im)
    for i in range(lines):
        d.rectangle([20, 20 + 30 * i, 300, 34 + 30 * i], fill="black")
    if blur:
        from PIL import ImageFilter

        im = im.filter(ImageFilter.GaussianBlur(4))
    im.save(path)
    return path


def test_changed_fraction_and_runs():
    a = np.zeros((90, 160), dtype=np.int16)
    b = a.copy()
    b[:10, :] = 255  # ~11% of pixels change
    assert changed_fraction(a, a) == 0.0
    assert 0.1 < changed_fraction(a, b) < 0.12
    assert group_runs([a, a, b, b, a], threshold=0.015) == [[0, 1], [2, 3], [4]]


def test_dedup_keeps_the_latest_state_of_each_build_step(tmp_path):
    # A slide built up line by line: each step is its own visual state, and the
    # latest sample of each step (its most complete version) is kept.
    raw = [(i, _slide(tmp_path / f"raw_{i:05d}.jpg", n)) for i, n in enumerate([1, 1, 3, 3])]
    frames = _dedup(raw, interval=2.0, change=0.015, max_per_minute=60, prefix="v_")
    assert [f.timestamp for f in frames] == [2.0, 6.0]
    assert [f.path.name for f in frames] == ["v_frame_00000.jpg", "v_frame_00001.jpg"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["v_frame_00000.jpg", "v_frame_00001.jpg"]


def test_dedup_prefers_a_sharp_frame_over_a_blurred_tail(tmp_path):
    # Same state, but the last sample is mid-fade: the sharp one is kept.
    raw = [(0, _slide(tmp_path / "raw_00000.jpg", 3)), (1, _slide(tmp_path / "raw_00001.jpg", 3, blur=True))]
    frames = _dedup(raw, interval=2.0, change=0.9, max_per_minute=60, prefix="")
    assert [f.timestamp for f in frames] == [0.0]


def _note(name, value=8, kind="diagram", legible=True, superseded_by=None):
    return FrameNote(name=name, kind=kind, legible=legible, complete=True, value=value,
                     superseded_by=superseded_by, description=f"desc {name}", text="E=mc^2", reason="r")


def _frame(i, t):
    return Frame(index=i, timestamp=t, path=__import__("pathlib").Path(f"/tmp/f{i}.jpg"), label="")


def test_selection_ranks_by_value_and_respects_supersession():
    frames = [_frame(i, 10 * i) for i in range(5)]
    notes = {
        "f0.jpg": _note("f0.jpg", 9, superseded_by="f1.jpg"),  # build step, finished in f1
        "f1.jpg": _note("f1.jpg", 9),
        "f2.jpg": _note("f2.jpg", 2, kind="talking_head"),
        "f3.jpg": _note("f3.jpg", 7),
        "f4.jpg": _note("f4.jpg", 6, legible=False),
    }
    assert not attachable(notes["f0.jpg"], notes, 5)
    picked = select_frames(frames, notes, limit=2, min_value=5)
    assert [f.path.name for f in picked] == ["f1.jpg", "f3.jpg"]  # time order, top-2 by value
    timeline = visual_timeline(frames, notes, {"f1.jpg"})
    assert "f1.jpg (ATTACHED" in timeline and "f3.jpg (not attached" in timeline
    assert "f2.jpg" not in timeline  # talking heads carry no content


class FakeTriageClient:
    def __init__(self):
        self.calls = []

    def complete_structured(self, **kw):
        self.calls.append(kw)
        names = [img.label for img in kw["images"]]
        # Omit the last frame of each batch to exercise the fallback, and claim
        # each frame is superseded by an EARLIER one (or an unknown one), which
        # must be ignored — otherwise mutual claims would exclude both frames.
        return {"frames": [
            {"name": n, "kind": "diagram", "legible": True, "complete": True, "value": 7,
             "superseded_by": names[i - 1] if i else "nope", "description": "d", "text": "",
             "reason": "r"}
            for i, n in enumerate(names[:-1])
        ]}


def test_triage_batches_caches_and_never_loses_a_frame(tmp_path):
    paths = [_slide(tmp_path / f"v_frame_{i:05d}.jpg", i % 3 + 1) for i in range(5)]
    frames = [Frame(index=i, timestamp=2.0 * i, path=p, label="") for i, p in enumerate(paths)]
    cfg = Config(workdir=tmp_path, triage_batch=3)
    client = FakeTriageClient()
    notes = triage_frames(client, frames, [Segment(0, 10, "talk")], "V", tmp_path, cfg)
    assert len(client.calls) == 2  # batches of 3 + 2
    assert set(notes) == {p.name for p in paths}
    assert notes["v_frame_00002.jpg"].description == "(not assessed)"  # omitted -> fallback
    assert all(n.superseded_by is None for n in notes.values())  # unknown successor dropped

    again = FakeTriageClient()
    assert triage_frames(again, frames, [], "V", tmp_path, cfg).keys() == notes.keys()
    assert again.calls == []  # cached
