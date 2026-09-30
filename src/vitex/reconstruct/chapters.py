"""Drive the Claude passes that turn chunks into LaTeX.

Passes:
  1. ``draft_chapters`` — one vision+transcript call per chunk that, with research
     tools, writes a grounded LaTeX body fragment (cited with \\cite). Cached.
  2. ``integrate`` — stitches the fragments into one coherent, complete document.
  3. ``build_bibliography`` — resolves the document's \\cite keys into a real,
     grounded thebibliography block.
"""

from __future__ import annotations

import logging
import re

from vitex.config import Config
from vitex.llm import research
from vitex.llm.client import ClaudeClient, ImageRef
from vitex.llm.prompts import (
    BIBLIOGRAPHY_SYSTEM,
    CHAPTER_SYSTEM,
    INTEGRATE_SYSTEM,
    bibliography_user,
    chapter_user,
    integrate_user,
)
from vitex.models import ChapterDraft, Chunk

log = logging.getLogger("vitex")

_FENCE_RE = re.compile(r"^\s*```(?:latex|tex)?\s*\n(.*?)\n```\s*$", re.DOTALL)


def strip_fences(text: str) -> str:
    """Remove a wrapping markdown code fence if the model added one anyway."""
    m = _FENCE_RE.match(text.strip())
    return m.group(1) if m else text.strip()


def _images_for(chunk: Chunk) -> list[ImageRef]:
    return [
        ImageRef(path=f.path, label=f.label, caption_hint=f.timestamp_text) for f in chunk.frames
    ]


def draft_chapters(
    client: ClaudeClient,
    chunks: list[Chunk],
    title: str,
    cfg: Config,
) -> list[ChapterDraft]:
    """Produce (and cache) a LaTeX fragment per chunk."""
    cache_dir = cfg.workdir / "chapters"
    cache_dir.mkdir(parents=True, exist_ok=True)
    drafts: list[ChapterDraft] = []

    for chunk in chunks:
        cache = cache_dir / f"section_{chunk.index:03d}.tex"
        if cache.exists() and not cfg.force:
            log.info("Section %d/%d: cached", chunk.index + 1, len(chunks))
            drafts.append(ChapterDraft(index=chunk.index, latex=cache.read_text("utf-8")))
            continue

        log.info(
            "Section %d/%d: drafting (%d segments, %d frames)%s",
            chunk.index + 1,
            len(chunks),
            len(chunk.segments),
            len(chunk.frames),
            " + grounding" if cfg.ground else "",
        )
        system = CHAPTER_SYSTEM
        user = chapter_user(chunk, title, chunk.index + 1, len(chunks))
        images = _images_for(chunk)
        max_tokens = max(cfg.max_output_tokens, 16384)
        if cfg.ground:
            raw = client.complete_with_tools(
                system=system,
                text=user,
                images=images,
                tools=research.TOOL_SPECS,
                dispatch=research.dispatch,
                max_tokens=max_tokens,
            )
        else:
            raw = client.complete(system=system, text=user, images=images, max_tokens=max_tokens)
        latex = strip_fences(raw)
        cache.write_text(latex, encoding="utf-8")
        drafts.append(ChapterDraft(index=chunk.index, latex=latex))

    return drafts


def integrate(
    client: ClaudeClient,
    drafts: list[ChapterDraft],
    title: str,
    description: str,
    preamble: str,
    cfg: Config,
) -> str:
    """Merge fragments into one full LaTeX document (cached)."""
    cache = cfg.workdir / "integrated.tex"
    if cache.exists() and not cfg.force:
        log.info("Integration: cached")
        return cache.read_text("utf-8")

    log.info("Integrating %d sections into one document...", len(drafts))
    fragments = "\n\n".join(
        f"% ===== SECTION {d.index + 1} =====\n{d.latex}"
        for d in sorted(drafts, key=lambda x: x.index)
    )
    document = strip_fences(
        client.complete(
            system=INTEGRATE_SYSTEM,
            text=integrate_user(title, description, preamble, fragments),
            max_tokens=max(cfg.max_output_tokens, 16384),
        )
    )
    cache.write_text(document, encoding="utf-8")
    return document


def build_bibliography(
    client: ClaudeClient,
    document: str,
    title: str,
    keys: list[str],
    cfg: Config,
) -> str:
    """Resolve the document's \\cite keys into a grounded thebibliography block.

    Uses the research tools (when grounding is enabled) to look up the real title,
    authors, year, and arXiv id / URL for each cited work, so references are
    correct rather than guessed. Cached at ``work/bibliography.tex``.
    """
    cache = cfg.workdir / "bibliography.tex"
    if cache.exists() and not cfg.force:
        log.info("Bibliography: cached")
        return cache.read_text("utf-8")

    log.info("Building bibliography for %d citation(s)%s...", len(keys), " + grounding" if cfg.ground else "")
    system = BIBLIOGRAPHY_SYSTEM
    user = bibliography_user(title, keys, document)
    max_tokens = max(cfg.max_output_tokens, 8192)
    if cfg.ground:
        raw = client.complete_with_tools(
            system=system,
            text=user,
            tools=research.TOOL_SPECS,
            dispatch=research.dispatch,
            max_tokens=max_tokens,
        )
    else:
        raw = client.complete(system=system, text=user, max_tokens=max_tokens)
    bibliography = strip_fences(raw)
    cache.write_text(bibliography, encoding="utf-8")
    return bibliography
