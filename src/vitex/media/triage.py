"""Rate every distinct video frame with a vision model, then pick by rating.

Frame extraction keeps one frame per visual state, but cannot tell a mechanism
diagram from a talking head. Here a (cheap) vision model looks at every distinct
frame, in time order and with the surrounding narration, and records what it
shows, whether it is legible and complete, whether a later frame supersedes it,
the text/equations on it, and a 0-10 teaching value with a reason. The notes are
cached in ``triage.json`` per source, so every keep/drop decision is inspectable.

Per chunk, the attached images are then the highest-value eligible frames, and
the notes for ALL frames become an on-screen timeline for the writer — so the
content of a frame reaches the tutorial even when its image is not attached.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from vitex.config import Config
from vitex.errors import LLMError
from vitex.llm.client import ClaudeClient, ImageRef, staged
from vitex.llm.prompts import TRIAGE_SCHEMA, TRIAGE_SYSTEM, triage_user
from vitex.models import Frame, FrameNote, Segment, _fmt_timestamp
from vitex.utils import fingerprint, read_cached, write_cached

log = logging.getLogger("vitex")

# Kinds never worth attaching as an image, whatever their score.
_NEVER_ATTACH = {"title", "talking_head", "transition"}
_NARRATION_WINDOW = 15.0  # seconds of narration shown on each side of a frame
_MAX_SUPERSEDE_GAP = 3  # a frame can only be superseded by one of the next 3 states


def _narration_near(segments: list[Segment], t: float) -> str:
    lo, hi = t - _NARRATION_WINDOW, t + _NARRATION_WINDOW
    return " ".join(s.text.strip() for s in segments if s.end >= lo and s.start <= hi)


def _fallback_note(frame: Frame) -> FrameNote:
    """Used when the model omits a frame: keep it attachable rather than lose it."""
    return FrameNote(
        name=frame.path.name,
        kind="other",
        legible=True,
        complete=True,
        value=5,
        superseded_by=None,
        description="(not assessed)",
        text="",
        reason="The triage model returned no assessment for this frame.",
    )


def triage_frames(
    client: ClaudeClient,
    frames: list[Frame],
    segments: list[Segment],
    video_title: str,
    workdir: Path,
    cfg: Config,
) -> dict[str, FrameNote]:
    """Return a note per frame (keyed by file name), cached in ``triage.json``."""
    cache = workdir / "triage.json"
    key = fingerprint(
        "triage", cfg.effective_triage_model, TRIAGE_SYSTEM, cfg.triage_batch, cfg.image_max_width,
        [(f.path.name, f.timestamp) for f in frames],
    )
    notes: dict[str, FrameNote] = {}
    cached = read_cached(cache, key, cfg.force)
    if cached is not None:
        notes = {d["name"]: FrameNote.from_dict(d) for d in json.loads(cached)}
        log.info("Frame triage: cached (%d frames)", len(notes))
        return notes

    batches = [frames[i : i + cfg.triage_batch] for i in range(0, len(frames), cfg.triage_batch)]
    log.info("Frame triage: %d frames in %d batches (%s)", len(frames), len(batches), cfg.effective_triage_model)
    for n, batch in enumerate(batches, 1):
        names = {f.path.name for f in batch}
        lines = "\n".join(
            f"- {f.path.name} (shown at {f.timestamp_text}); narration: "
            f"\"{_narration_near(segments, f.timestamp)[:600]}\""
            for f in batch
        )
        # Each batch is cached on its own, so an interrupted triage resumes
        # without paying again for the batches already rated.
        batch_cache = workdir / "triage" / f"batch_{n:03d}.json"
        batch_key = fingerprint(key, lines)
        cached_batch = read_cached(batch_cache, batch_key, cfg.force)
        result: dict | None = json.loads(cached_batch) if cached_batch is not None else None
        images = [ImageRef(path=f.path, label=f.path.name, caption_hint=f.timestamp_text) for f in batch]
        for attempt in range(1, cfg.generation_attempts + 1):
            if result is not None:
                break
            try:
                with staged(client, f"triage {n}/{len(batches)}"):
                    result = client.complete_structured(
                        system=TRIAGE_SYSTEM,
                        text=triage_user(video_title, lines),
                        schema=TRIAGE_SCHEMA,
                        description="Record the assessment of every frame in the batch.",
                        images=images,
                        model=cfg.effective_triage_model,
                        max_tokens=8192,
                    )
                break
            except LLMError as exc:
                log.warning("Triage batch %d/%d failed (%s); attempt %d/%d", n, len(batches), exc, attempt, cfg.generation_attempts)
        if result is None:
            raise LLMError(f"Frame triage failed for batch {n} after {cfg.generation_attempts} attempts.")
        if cached_batch is None:
            write_cached(batch_cache, batch_key, json.dumps(result, ensure_ascii=False))

        order = {f.path.name: i for i, f in enumerate(batch)}
        for entry in result.get("frames", []):
            if isinstance(entry, dict) and entry.get("name") in names:
                note = FrameNote.from_dict(entry)
                # Only one of the next few frames of the batch can supersede: a
                # slide being built up spans consecutive visual states. Requiring
                # "later" also makes cycles (which would exclude both) impossible.
                gap = order.get(note.superseded_by or "", -1) - order[note.name]
                if not 0 < gap <= _MAX_SUPERSEDE_GAP:
                    note.superseded_by = None
                notes[note.name] = note
        for f in batch:
            notes.setdefault(f.path.name, _fallback_note(f))

    write_cached(cache, key, json.dumps([notes[f.path.name].to_dict() for f in frames], indent=2, ensure_ascii=False))
    kept = sum(1 for n in notes.values() if attachable(n, notes, cfg.min_frame_value))
    log.info("Frame triage: %d of %d frames rated attachable (value >= %d)", kept, len(notes), cfg.min_frame_value)
    return notes


def attachable(note: FrameNote, notes: dict[str, FrameNote], min_value: int) -> bool:
    """Worth attaching as an image: valuable, legible, and not superseded by a
    later frame that is itself attachable. (Successors are always later frames,
    so the recursion terminates.)"""
    if note.kind in _NEVER_ATTACH or not note.legible or note.value < min_value:
        return False
    successor = notes.get(note.superseded_by) if note.superseded_by else None
    return successor is None or not attachable(successor, notes, min_value)


def select_frames(frames: list[Frame], notes: dict[str, FrameNote], limit: int, min_value: int) -> list[Frame]:
    """The ``limit`` highest-value attachable frames, returned in time order."""
    eligible = [f for f in frames if f.path.name in notes and attachable(notes[f.path.name], notes, min_value)]
    ranked = sorted(eligible, key=lambda f: (-notes[f.path.name].value, f.timestamp))
    return sorted(ranked[:limit] if limit > 0 else ranked, key=lambda f: f.timestamp)


def visual_timeline(frames: list[Frame], notes: dict[str, FrameNote], attached: set[str]) -> str:
    """One line per frame: what was on screen, and whether it is attached."""
    lines = []
    for f in frames:
        n = notes.get(f.path.name)
        if n is None or n.kind in _NEVER_ATTACH:
            continue
        status = "ATTACHED" if f.path.name in attached else "not attached"
        text = f" | on screen: {n.text.strip()}" if n.text.strip() else ""
        lines.append(f"[{_fmt_timestamp(f.timestamp)}] {f.path.name} ({status}, {n.kind}): {n.description.strip()}{text}")
    return "\n".join(lines)
