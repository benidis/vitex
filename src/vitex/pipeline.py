"""End-to-end orchestration of the vitex stages.

Inputs are SEEDS: any number of videos (URLs or files) and documents. Every
expensive stage writes its result under ``cfg.workdir`` and is skipped on a
later run if that artifact already exists and was built from the same inputs
(recorded in a ``<artifact>.key`` fingerprint sidecar), unless ``--force``. So a
run that dies midway resumes without redoing finished work, while a changed
setting, prompt, or source regenerates exactly the stages it affects.

Stages:
  per video (work/sources/<slug>/): download, audio, transcript, distinct
    frames, vision triage; per document (work/sources/<slug>/): text + figures
  then, for the tutorial (work/runs/<name>/):
    1. seed units (video chunks, document parts)
    2. survey the topic and its literature            -> survey.json
    3. plan the scope (levels, pages, reasons)        -> plan.md   (--plan-only stops)
    4. draft each unit, scoped by the plan (parallel) -> drafts/<unit>.tex
    5. inventory every item of content                -> drafts/<unit>.items.json
    6. outline following the plan                     -> outline.json
    7. write + review + revise each section           -> sections/, coverage.json
    8. editor reads the whole; fix flagged sections   -> editorial/, editorial.json
    9. bibliography; compile; layout check            -> output/<name>/<name>.{tex,pdf}
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from vitex.config import Config
from vitex.llm.client import ClaudeClient
from vitex.media import extract_audio, resolve_source
from vitex.media.documents import load_document
from vitex.media.download import SourceInfo
from vitex.media.frames import extract_frames
from vitex.media.triage import triage_frames
from vitex.models import Figure, Frame, SeedUnit, Segment
from vitex.output import (
    assemble_document,
    build_pdf,
    build_preamble,
    cite_keys,
    drop_unknown_figures,
    insert_bibliography,
    referenced_figures,
    stage_figures,
    write_document,
)
from vitex.reconstruct import (
    build_bibliography,
    build_inventory,
    build_outline,
    build_survey,
    document_units,
    draft_seeds,
    edit_tutorial,
    fit_lengths,
    section_jobs,
    video_units,
    write_sections,
)
from vitex.reconstruct.outline import WORDS_PER_PAGE
from vitex.reconstruct.plan import resolve_plan
from vitex.transcribe import build_transcriber
from vitex.utils import fingerprint, read_cached, require_tool, safe_stem, write_cached

log = logging.getLogger("vitex")


@dataclass
class Result:
    tex_path: Path | None  # None for --plan-only
    pdf_path: Path | None
    stem: str
    plan_path: Path | None = None


class Pipeline:
    def __init__(self, cfg: Config):
        cfg.validate()
        self.cfg = cfg

    # -- cached per-video stages --------------------------------------------

    def _transcript(self, workdir: Path, audio_path: Path, force: bool) -> list[Segment]:
        cfg = self.cfg
        cache = workdir / "transcript.json"
        key = fingerprint(
            "transcript", cfg.transcriber, cfg.whisper_model if cfg.transcriber == "whisper" else None
        )
        cached = read_cached(cache, key, force)
        if cached is not None:
            log.info("Transcript: cached")
            return [Segment.from_dict(d) for d in json.loads(cached)]
        segments = build_transcriber(cfg).transcribe(audio_path)
        write_cached(cache, key, json.dumps([s.to_dict() for s in segments], indent=2, ensure_ascii=False))
        return segments

    def _frames(self, info: SourceInfo, force: bool) -> list[Frame]:
        cfg = self.cfg
        cache = info.workdir / "frames.json"
        key = fingerprint("frames-v2", cfg.frame_interval, cfg.frame_change, cfg.max_frames_per_minute, info.slug)
        cached = read_cached(cache, key, force)
        if cached is not None:
            frames = [Frame.from_dict(d) for d in json.loads(cached)]
            # Paths are stored as written; after the source dir moved, find the
            # frames by name in its current location.
            for f in frames:
                if not f.path.exists():
                    f.path = info.workdir / "frames" / f.path.name
            if all(f.path.exists() for f in frames):
                log.info("Frames: cached (%d)", len(frames))
                return frames
            log.info("Frame cache stale (images missing); re-extracting")
        frames = extract_frames(
            info.video_path,
            info.workdir / "frames",
            interval=cfg.frame_interval,
            change=cfg.frame_change,
            max_per_minute=cfg.max_frames_per_minute,
            prefix=f"{info.slug}_",
        )
        write_cached(cache, key, json.dumps([f.to_dict() for f in frames], indent=2, ensure_ascii=False))
        return frames

    def _audio(self, workdir: Path, video_path: Path, force: bool) -> Path:
        cache = workdir / "audio.wav"
        if cache.exists() and not force:
            log.info("Audio: cached")
            return cache
        return extract_audio(video_path, workdir)

    def _video(self, source: str, client: ClaudeClient) -> tuple[SourceInfo, list[SeedUnit], list[Figure]]:
        """Prepare one video and return its seed units and attached figures."""
        cfg = self.cfg
        info = resolve_source(source, cfg.sources_dir, cfg.force)
        force = cfg.force
        if info.changed and not force:
            # This dir holds artifacts from a different source; all are stale.
            log.warning("Source changed since last run; invalidating cached artifacts in %s", info.workdir)
            force = True
        log.info("Video: %s  (%.1f min)", info.title, (info.duration or 0) / 60)
        audio_path = self._audio(info.workdir, info.video_path, force)
        segments = self._transcript(info.workdir, audio_path, force)
        frames = self._frames(info, force)
        notes = triage_frames(client, frames, segments, info.title, info.workdir, cfg)
        units = video_units(info, segments, frames, notes, cfg)
        return info, units, [f for u in units for f in u.figures]

    # -- driver -------------------------------------------------------------

    def run(self) -> Result:
        cfg = self.cfg
        if cfg.sources:
            require_tool("ffmpeg")  # fail fast before doing anything expensive
        cfg.workdir.mkdir(parents=True, exist_ok=True)
        cfg.outdir.mkdir(parents=True, exist_ok=True)
        client = ClaudeClient(cfg)
        client.check_access()
        try:
            return self._run(client)
        finally:
            # Also on failure, so the spend of an interrupted run is visible.
            usage = getattr(client, "usage", None)
            if usage is not None and usage.by_model:
                log.info("%s", usage.summary())

    def _run(self, client: ClaudeClient) -> Result:
        cfg = self.cfg

        units: list[SeedUnit] = []
        figures: dict[str, Figure] = {}
        titles: list[str] = []
        for source in cfg.sources:
            info, video_seeds, video_figs = self._video(source, client)
            units += video_seeds
            figures.update((f.name, f) for f in video_figs)
            titles.append(info.title)
        for path in cfg.docs:
            doc = load_document(path, cfg.sources_dir)
            doc_seeds = document_units(doc, cfg)
            units += doc_seeds
            figures.update((f.name, f) for f in doc.figures)
            titles.append(doc.title)
        log.info("Seeds: %d unit(s) from %d source(s); %d candidate figures", len(units), len(titles), len(figures))

        # The run's identity: work/runs/<name>/ and output/<name>/.
        if not cfg.name:
            cfg.name = safe_stem(cfg.title or titles[0])
        cfg.run_dir.mkdir(parents=True, exist_ok=True)
        self._log_to_file(cfg.run_dir / "run.log")
        log.info("Run: %s (work in %s)", cfg.name, cfg.run_dir)

        # Scope first: survey the topic, then decide what the tutorial includes,
        # at what level, and why. --plan-only stops here.
        survey = build_survey(client, units, cfg)
        plan = resolve_plan(client, survey, cfg, [u.id for u in units])
        plan_file = cfg.plan_path or cfg.run_dir / "plan.md"
        if cfg.plan_only:
            log.info("Plan only: review or edit %s, then run again without --plan-only "
                     "(or with --plan FILE, or --plan-feedback \"...\").", plan_file)
            return Result(tex_path=None, pdf_path=None, stem=cfg.name, plan_path=plan_file)

        drafts = draft_seeds(client, units, cfg, plan.scope_text())
        items, summaries = build_inventory(client, units, drafts, cfg)
        log.info("Inventory: %d content items", len(items))
        budget = round(plan.budget_pages * WORDS_PER_PAGE)
        log.info("Length budget: %d words (%.0f pages, from the plan)", budget, plan.budget_pages)
        entries = [{"name": e.name, "level": e.level, "pages": e.pages} for e in plan.entries]
        outline = build_outline(
            client, units, items, summaries, list(figures.values()), cfg, budget,
            plan.to_markdown(), entries, round(plan.frame_pages() * WORDS_PER_PAGE),
        )
        jobs = section_jobs(outline, items, drafts, units, figures, {e.name: e.level for e in plan.entries})
        title = cfg.title or plan.title or outline.title or titles[0]
        stem = cfg.name
        if cfg.sections:
            # Sample: write only the chosen sections, to judge the writing
            # before paying for the whole tutorial. They are cached, so the
            # full run reuses them.
            bad = [n for n in cfg.sections if not 1 <= n <= len(jobs)]
            if bad:
                raise ValueError(f"--sections {bad} out of range: the outline has {len(jobs)} sections")
            jobs = [j for j in jobs if j.idx + 1 in cfg.sections]
            log.info("Sample: writing sections %s of %d", ", ".join(str(j.idx + 1) for j in jobs), len(outline.sections))
            sections = write_sections(client, jobs, figures, cfg, report_name="coverage.sample.json")
            stem = f"{cfg.name}_sample"
            title = f"{title} (sample: sections {', '.join(str(j.idx + 1) for j in jobs)})"
        else:
            sections = write_sections(client, jobs, figures, cfg)
            # Holistic pass: an editor reads the whole tutorial and the sections
            # with cross-section problems are revised.
            sections = edit_tutorial(client, outline, items, sections, jobs, cfg)
        # Final length pass: whatever reviews and edits added is brought back to
        # the plan's targets by removing material, not compressing it.
        sections = fit_lengths(client, cfg, jobs, sections)
        # Each tutorial gets its own directory under outdir so several can
        # coexist without their .tex/.pdf/figures colliding.
        doc_dir = cfg.outdir / cfg.name
        doc_dir.mkdir(parents=True, exist_ok=True)
        document = assemble_document(build_preamble(title), sections)

        # A figure pointing at a file the model was never given can't compile;
        # drop it here rather than spend a repair pass on it.
        document, unknown = drop_unknown_figures(document, set(figures))
        if unknown:
            log.warning("Dropped %d figure(s) with unknown image files: %s", len(unknown), ", ".join(unknown))

        # Ground the citations introduced while writing into a real bibliography.
        keys = cite_keys(document)
        if keys:
            bibliography = build_bibliography(client, document, title, keys, cfg)
            document = insert_bibliography(document, bibliography)
        else:
            log.info("No citations found; skipping bibliography.")

        used = [figures[name] for name in sorted(referenced_figures(document)) if name in figures]
        log.info("Document references %d of %d candidate figures", len(used), len(figures))
        stage_figures(used, doc_dir)

        tex_path = doc_dir / f"{stem}.tex"
        pdf_path: Path | None = None
        if cfg.build_pdf:
            pdf_path = build_pdf(client, document, tex_path, cfg)
        else:
            write_document(document, doc_dir, stem)

        return Result(tex_path=tex_path, pdf_path=pdf_path, stem=stem, plan_path=plan_file)

    @staticmethod
    def _log_to_file(path: Path) -> None:
        """Also write this run's log to ``path`` (appending across runs)."""
        root = logging.getLogger("vitex")
        if any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == path.resolve() for h in root.handlers):
            return
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S"))
        root.addHandler(handler)
