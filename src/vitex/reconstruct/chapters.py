"""Shared generation helpers, the per-seed draft pass, and the bibliography pass.

Every pass validates its output before caching it and retries a bad response
(empty, truncated — detected by a required end-sentinel — or malformed) up to
``cfg.generation_attempts`` times. If it still fails the run stops with an
error rather than skipping content; re-running resumes from the cache. Each
cache entry is keyed on the pass's full inputs (model, prompts, seed content,
figures), so changing any of them regenerates exactly the affected artifacts.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Callable

from vitex.config import Config
from vitex.errors import LLMError
from vitex.llm import research
from vitex.llm.client import ClaudeClient, ImageRef, staged
from vitex.llm.prompts import (
    BIBLIOGRAPHY_SYSTEM,
    DRAFT_END,
    bibliography_user,
    draft_system,
    draft_user,
)
from vitex.models import Figure, SeedUnit
from vitex.output.latex import cite_contexts, complete_bibliography, referenced_figures
from vitex.reconstruct.seeds import figure_lines, viewable
from vitex.utils import fingerprint, read_cached, run_parallel, write_cached

log = logging.getLogger("vitex")

_FENCE_RE = re.compile(r"^\s*```(?:latex|tex)?\s*\n(.*?)\n```\s*$", re.DOTALL)


def strip_fences(text: str) -> str:
    """Remove a wrapping markdown code fence if the model added one anyway."""
    m = _FENCE_RE.match(text.strip())
    return m.group(1) if m else text.strip()


def check_fragment(latex: str, sentinel: str) -> str | None:
    """Problem with a section-like fragment, or None if it is usable."""
    if not latex.strip():
        return "empty response"
    if "\\section" not in latex:
        return "no \\section in the fragment"
    if sentinel not in latex:
        return f"missing end marker {sentinel!r} (output truncated)"
    return None


def without_sentinel(latex: str, sentinel: str) -> str:
    return latex.replace(sentinel, "").rstrip() + "\n"


def _check_document(latex: str) -> str | None:
    for marker in ("\\documentclass", "\\begin{document}", "\\end{document}"):
        if marker not in latex:
            return f"missing {marker} (truncated or malformed document)"
    return None


def generate_valid(
    produce: Callable[[], str], check: Callable[[str], str | None], what: str, attempts: int
) -> str:
    """Call ``produce`` until ``check`` accepts its (fence-stripped) output."""
    problem = ""
    for attempt in range(1, attempts + 1):
        text = strip_fences(produce())
        problem = check(text) or ""
        if not problem:
            return text
        log.warning("%s: unusable output (%s); attempt %d/%d", what, problem, attempt, attempts)
    raise LLMError(f"{what}: model output unusable after {attempts} attempts ({problem}).")


def structured_valid(
    produce: Callable[[], dict], check: Callable[[dict], str | None], what: str, attempts: int
) -> dict:
    """Like :func:`generate_valid` for structured (tool-submitted) answers."""
    problem = ""
    for attempt in range(1, attempts + 1):
        try:
            data = produce()
        except LLMError as exc:
            problem = str(exc)
        else:
            problem = check(data) or ""
            if not problem:
                return data
        log.warning("%s: unusable answer (%s); attempt %d/%d", what, problem, attempt, attempts)
    raise LLMError(f"{what}: no usable answer after {attempts} attempts ({problem}).")


def generate(
    client: ClaudeClient,
    cfg: Config,
    *,
    system: str,
    text: str,
    images: list[ImageRef] | None = None,
    max_iterations: int | None = None,
) -> str:
    """One text generation, with research tools when grounding is enabled."""
    if cfg.ground:
        return client.complete_with_tools(
            system=system,
            text=text,
            images=images,
            tools=research.TOOL_SPECS,
            dispatch=research.dispatch,
            max_tokens=cfg.max_output_tokens,
            max_iterations=max_iterations,
        )
    return client.complete(system=system, text=text, images=images, max_tokens=cfg.max_output_tokens)


def generate_structured(
    client: ClaudeClient,
    cfg: Config,
    *,
    system: str,
    text: str,
    schema: dict,
    description: str,
    research_allowed: bool,
    images: list[ImageRef] | None = None,
) -> dict:
    """One structured generation; research tools only if allowed and grounding."""
    use_tools = research_allowed and cfg.ground
    return client.complete_structured(
        system=system,
        text=text,
        schema=schema,
        description=description,
        images=images,
        max_tokens=cfg.max_output_tokens,
        tools=research.TOOL_SPECS if use_tools else None,
        dispatch=research.dispatch if use_tools else None,
    )


def pass_key(cfg: Config, *parts: object) -> str:
    """Fingerprint of everything that determines one generation pass's output."""
    return fingerprint(cfg.model, cfg.ground, cfg.research_max_iterations, cfg.image_max_width, parts)


def image_refs(figures: list[Figure]) -> list[ImageRef]:
    """Attachable images, each labeled with the exact file name to write."""
    return [ImageRef(path=f.path, label=f.name, caption_hint=f.hint) for f in figures if viewable(f)]


def draft_seeds(client: ClaudeClient, units: list[SeedUnit], cfg: Config) -> dict[str, str]:
    """Produce (and cache) the fact-checked LaTeX draft of every seed unit.

    Units are independent, so they are drafted ``cfg.parallel`` at a time.
    """
    cache_dir = cfg.workdir / "drafts"
    system = draft_system(cfg.ground, cfg.guide)

    def one(n: int) -> str:
        unit = units[n - 1]
        what = f"Draft {n}/{len(units)} [{unit.id}]"
        user = draft_user(
            kind=unit.kind,
            source_title=unit.source_title,
            position=unit.position,
            text=unit.text,
            figure_lines=figure_lines(unit.figures),
            visual_notes=unit.visual_notes,
        )
        cache = cache_dir / f"{unit.id}.tex"
        key = pass_key(cfg, "draft", system, user, [f.name for f in unit.figures])
        latex = read_cached(cache, key, cfg.force)
        if latex is not None:
            log.info("%s: cached", what)
            return latex

        log.info(
            "%s: drafting (%s, %d chars, %d figures)%s",
            what, unit.kind, len(unit.text), len(unit.figures), " + research" if cfg.ground else "",
        )
        images = image_refs(unit.figures)
        with staged(client, f"draft {n}/{len(units)}"):
            latex = generate_valid(
                lambda: generate(client, cfg, system=system, text=user, images=images),
                lambda t: check_fragment(t, DRAFT_END),
                what,
                cfg.generation_attempts,
            )
        latex = without_sentinel(latex, DRAFT_END)
        log.info("%s: done (%d chars, uses %d of %d attached figures)",
                 what, len(latex), len(referenced_figures(latex)), len(images))
        write_cached(cache, key, latex)
        return latex

    log.info("Drafting %d seed units, %d at a time", len(units), cfg.parallel)
    latexes = run_parallel(one, list(range(1, len(units) + 1)), cfg.parallel)
    return {u.id: latex for u, latex in zip(units, latexes)}


_BIB_BATCH = 12  # cite keys resolved per research loop
_BIB_BODY_RE = re.compile(r"\\begin\{thebibliography\}\{[^}]*\}(.*?)\\end\{thebibliography\}", re.DOTALL)


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
    correct rather than guessed. The model sees the text around every citation
    of each key. Keys are resolved in batches of ``_BIB_BATCH``, each with its
    own research budget and cache entry, so the cost grows linearly with the
    number of citations instead of re-sending one ever-growing conversation.
    Any key still left out gets a visible placeholder entry, so no citation
    renders as "[?]". Cached in ``work/tutorial/``.
    """
    contexts = cite_contexts(document, keys)
    batches = [keys[i : i + _BIB_BATCH] for i in range(0, len(keys), _BIB_BATCH)]
    log.info("Building bibliography for %d citation(s) in %d batch(es)%s...",
             len(keys), len(batches), " + research" if cfg.ground else "")

    def check(bib: str) -> str | None:
        return None if _BIB_BODY_RE.search(bib) else "no complete thebibliography environment"

    def one(n: int) -> str:
        what = f"Bibliography {n}/{len(batches)}"
        batch = batches[n - 1]
        user = bibliography_user(title, {k: contexts[k] for k in batch})
        cache = cfg.tutorial_dir / f"bibliography_{n:02d}.tex"
        key = pass_key(cfg, "bibliography", BIBLIOGRAPHY_SYSTEM, user)
        part = read_cached(cache, key, cfg.force)
        if part is None:
            with staged(client, f"bibliography {n}/{len(batches)}"):
                part = generate_valid(
                    lambda: generate(client, cfg, system=BIBLIOGRAPHY_SYSTEM, text=user),
                    check,
                    what,
                    cfg.generation_attempts,
                )
            write_cached(cache, key, part)
        else:
            log.info("%s: cached", what)
        m = _BIB_BODY_RE.search(part)
        return m.group(1).strip() if m else ""

    bodies = run_parallel(one, list(range(1, len(batches) + 1)), cfg.parallel)

    merged = "\\begin{thebibliography}{99}\n" + "\n\n".join(bodies) + "\n\\end{thebibliography}"
    bibliography, missing = complete_bibliography(merged, keys)
    if missing:
        log.warning("Bibliography left %d key(s) unresolved: %s", len(missing), ", ".join(missing))
    return bibliography


def dump(obj: object) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)
