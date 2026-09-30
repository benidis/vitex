"""End-to-end orchestration of the vitex stages.

Every expensive stage writes its result under ``cfg.workdir`` and is skipped on
a later run if that artifact already exists (unless ``--force``). So a run that
dies during, say, chapter drafting resumes without re-downloading,
re-transcribing, or re-extracting frames.

Stages:
  1. resolve source (download or local)      -> video + metadata
  2. extract audio                            -> work/audio.wav
  3. transcribe                               -> work/transcript.json
  4. extract + de-duplicate frames            -> work/frames.json (+ images)
  5. chunk transcript by time, align frames   -> in memory
  6. draft one LaTeX fragment per chunk        -> work/chapters/*.tex
  7. integrate fragments into one document     -> work/integrated.tex
  8. stage figures, write .tex, build PDF      -> output/<stem>/<stem>.{tex,pdf}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from vitex.config import Config
from vitex.llm.client import ClaudeClient
from vitex.media import extract_audio, resolve_source
from vitex.media.frames import extract_frames
from vitex.models import Frame, Segment
from vitex.output import (
    build_pdf,
    build_preamble,
    cite_keys,
    insert_bibliography,
    referenced_figures,
    stage_figures,
    write_document,
)
from vitex.reconstruct import (
    assign_frames,
    build_bibliography,
    build_chunks,
    draft_chapters,
    integrate,
)
from vitex.transcribe import build_transcriber
from vitex.utils import read_json, require_tool, safe_stem, write_json

log = logging.getLogger("vitex")


@dataclass
class Result:
    tex_path: Path
    pdf_path: Path | None
    stem: str


class Pipeline:
    def __init__(self, cfg: Config):
        cfg.validate()
        self.cfg = cfg

    # -- cached stages ------------------------------------------------------

    def _transcript(self, audio_path: Path) -> list[Segment]:
        cache = self.cfg.workdir / "transcript.json"
        if cache.exists() and not self.cfg.force:
            log.info("Transcript: cached")
            return [Segment.from_dict(d) for d in read_json(cache)]
        segments = build_transcriber(self.cfg).transcribe(audio_path)
        write_json(cache, [s.to_dict() for s in segments])
        return segments

    def _frames(self, video_path: Path) -> list[Frame]:
        cache = self.cfg.workdir / "frames.json"
        if cache.exists() and not self.cfg.force:
            frames = [Frame.from_dict(d) for d in read_json(cache)]
            if all(f.path.exists() for f in frames):
                log.info("Frames: cached (%d)", len(frames))
                return frames
            log.info("Frame cache stale (images missing); re-extracting")
        frames = extract_frames(
            video_path,
            self.cfg.frames_dir,
            interval=self.cfg.frame_interval,
            phash_distance=self.cfg.phash_distance,
            max_frames=self.cfg.max_frames,
        )
        write_json(cache, [f.to_dict() for f in frames])
        return frames

    def _audio(self, video_path: Path) -> Path:
        cache = self.cfg.workdir / "audio.wav"
        if cache.exists() and not self.cfg.force:
            log.info("Audio: cached")
            return cache
        return extract_audio(video_path, self.cfg.workdir)

    # -- driver -------------------------------------------------------------

    def run(self) -> Result:
        cfg = self.cfg
        require_tool("ffmpeg")  # fail fast before doing anything expensive
        cfg.workdir.mkdir(parents=True, exist_ok=True)
        cfg.outdir.mkdir(parents=True, exist_ok=True)

        info = resolve_source(cfg.source, cfg.workdir, cfg.force)
        if info.changed and not cfg.force:
            # work/ holds artifacts from a different source; every cached stage
            # (audio, transcript, frames, drafts, ...) is stale. Recompute all.
            log.warning("Source changed since last run; invalidating cached artifacts in %s", cfg.workdir)
            cfg.force = True
        stem = safe_stem(info.title)
        # Each run gets its own directory under outdir so multiple tutorials can
        # coexist without their .tex/.pdf/figures colliding.
        doc_dir = cfg.outdir / stem
        doc_dir.mkdir(parents=True, exist_ok=True)
        log.info("Title: %s  (%.1f min)", info.title, (info.duration or 0) / 60)

        audio_path = self._audio(info.video_path)
        segments = self._transcript(audio_path)
        frames = self._frames(info.video_path)

        chunks = build_chunks(segments, cfg.chunk_minutes * 60)
        assign_frames(chunks, frames, cfg.max_images_per_chunk)
        log.info("Built %d chapter chunks", len(chunks))

        client = ClaudeClient(cfg)
        drafts = draft_chapters(client, chunks, info.title, cfg)

        preamble = build_preamble(info.title)
        document = integrate(client, drafts, info.title, info.description, preamble, cfg)

        # Ground the citations introduced while drafting into a real bibliography.
        keys = cite_keys(document)
        if keys:
            bibliography = build_bibliography(client, document, info.title, keys, cfg)
            document = insert_bibliography(document, bibliography)
        else:
            log.info("No citations found; skipping bibliography.")

        # Stage only the frames the finished document actually references — the
        # model curates figures, so most candidate frames are intentionally unused.
        candidates = {f.path.name: f for c in chunks for f in c.frames}
        used_frames = [candidates[name] for name in referenced_figures(document) if name in candidates]
        log.info("Document references %d of %d candidate figures", len(used_frames), len(candidates))
        stage_figures(used_frames, doc_dir)

        tex_path = doc_dir / f"{stem}.tex"
        pdf_path: Path | None = None
        if cfg.build_pdf:
            pdf_path = build_pdf(client, document, tex_path, cfg)
            # The repair loop may have rewritten the document (e.g. dropped a
            # figure). Re-stage from the final .tex so figures/ mirrors exactly
            # what the compiled PDF references — no orphaned frames left behind.
            final = tex_path.read_text(encoding="utf-8")
            final_frames = [candidates[n] for n in referenced_figures(final) if n in candidates]
            if len(final_frames) != len(used_frames):
                stage_figures(final_frames, doc_dir)
        else:
            write_document(document, doc_dir, stem)

        return Result(tex_path=tex_path, pdf_path=pdf_path, stem=stem)
