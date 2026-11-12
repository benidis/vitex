"""Write the tutorial section by section from the outline, then review each.

Each section is written with the whole outline in view (what the reader already
knows, what comes next), its step-by-step plan, the full details of every item
assigned to it, the fact-checked drafts those items come from, its figures, the
end of the previous section (for continuity), and the citation keys in use.

Each section then goes through up to ``cfg.review_rounds`` rounds of review and
revision. The review checks coverage of every assigned item, step-by-step
rigor (no jumps, nothing undefined or unmotivated), flow, and figures, plus
deterministic checks in code. A revision that drops a large share of the text
is rejected. The outcome of every review is written to ``coverage.json``.
"""

from __future__ import annotations

import json
import logging

from vitex.config import Config
from vitex.llm.client import ClaudeClient
from vitex.llm.prompts import (
    REVIEW_SCHEMA,
    SECTION_END,
    review_system,
    review_user,
    revise_system,
    revise_user,
    section_system,
    section_user,
)
from vitex.models import Figure, SeedUnit
from vitex.output.latex import cite_keys, referenced_figures
from vitex.reconstruct.chapters import (
    check_fragment,
    dump,
    generate,
    generate_structured,
    generate_valid,
    image_refs,
    pass_key,
    structured_valid,
    without_sentinel,
)
from vitex.reconstruct.outline import Item, Outline, Section
from vitex.reconstruct.seeds import figure_lines
from vitex.utils import read_cached, write_cached

log = logging.getLogger("vitex")

_MATERIAL_CHARS = 160_000  # cap on draft material per section prompt (~40k tokens)
_PREVIOUS_TAIL_CHARS = 3000
_MIN_KEPT_FRACTION = 0.85  # a revision shorter than this share of the original is rejected


def _items_text(items: list[Item]) -> str:
    return "\n".join(f"- {i.id} | {i.kind} | {i.title}\n    {i.detail.strip()}" for i in items)


def _material(items: list[Item], drafts: dict[str, str], units: dict[str, SeedUnit]) -> str:
    """Drafts of the units these items come from, most-relevant first."""
    counts: dict[str, int] = {}
    for i in items:
        counts[i.unit_id] = counts.get(i.unit_id, 0) + 1
    parts: list[str] = []
    total = 0
    for unit_id in sorted(counts, key=lambda u: -counts[u]):
        u = units[unit_id]
        block = f"=== Draft from {u.kind} \"{u.source_title}\", {u.position} ===\n{drafts[unit_id]}"
        if total + len(block) > _MATERIAL_CHARS and parts:
            log.warning("Draft material for a section exceeds %d chars; omitting draft %s (its items' details are still included)", _MATERIAL_CHARS, unit_id)
            continue
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts) or "(no draft material: this section is built from the plan and new material)"


def mechanical_checks(latex: str, section: Section, known_figures: set[str]) -> list[str]:
    """Problems detectable in code, reported to the reviewer and reviser."""
    used = referenced_figures(latex)
    problems = [f"assigned figure {f} is not included" for f in section.figures if f not in used]
    problems += [f"figure file {f} does not exist; use only provided names" for f in sorted(used - known_figures)]
    if "$$" in latex:
        problems.append("uses $$...$$ display math; use equation/align or \\[...\\]")
    return problems


def _review_text(review: dict, checks: list[str]) -> str:
    lines = [f"- MISSING item {m['id']}: {m['what_is_missing']}" for m in review.get("missing_items", [])]
    lines += [f"- ISSUE at {i['location']}: {i['problem']} -> FIX: {i['fix']}" for i in review.get("issues", [])]
    lines += [f"- CHECK: {c}" for c in checks]
    return "\n".join(lines)


def _review_and_revise(
    client: ClaudeClient,
    cfg: Config,
    outline: Outline,
    idx: int,
    section: Section,
    items_text: str,
    request: str,
    latex: str,
    figures: list[Figure],
    known_figures: set[str],
    what: str,
) -> tuple[str, dict]:
    """Return the final section text and a report of its reviews."""
    report: dict = {"section": section.title, "items": section.items, "figures": section.figures, "rounds": []}
    rsystem, vsystem = review_system(cfg.ground, cfg.guide), revise_system(cfg.ground, cfg.guide)
    for rnd in range(1, cfg.review_rounds + 1):
        checks = mechanical_checks(latex, section, known_figures)
        ruser = review_user(
            overview=outline.overview(idx), spec=section.spec(), items=items_text, section=latex,
            checks="\n".join(f"- {c}" for c in checks),
        )
        review = structured_valid(
            lambda: generate_structured(
                client, cfg, system=rsystem, text=ruser, schema=REVIEW_SCHEMA,
                description="Submit the review of the section.", research_allowed=True,
            ),
            lambda d: None if "approved" in d else "review lacks a verdict",
            f"{what} review {rnd}",
            cfg.generation_attempts,
        )
        missing, issues = review.get("missing_items", []), review.get("issues", [])
        report["rounds"].append({"review": review, "checks": checks})
        if review.get("approved") and not missing and not checks:
            log.info("%s: approved in review round %d", what, rnd)
            report["approved"] = True
            return latex, report
        log.info("%s: review %d — %d missing item(s), %d issue(s), %d check(s); revising",
                 what, rnd, len(missing), len(issues), len(checks))
        revised = generate_valid(
            lambda: generate(
                client, cfg, system=vsystem,
                text=revise_user(section_request=request, section=latex, review=_review_text(review, checks)),
                images=image_refs(figures),
            ),
            lambda t: check_fragment(t, SECTION_END),
            f"{what} revision {rnd}",
            cfg.generation_attempts,
        )
        revised = without_sentinel(revised, SECTION_END)
        if len(revised) < _MIN_KEPT_FRACTION * len(latex):
            log.warning("%s: revision %d dropped %.0f%% of the text; keeping the previous version",
                        what, rnd, 100 * (1 - len(revised) / len(latex)))
            report["rounds"][-1]["revision_rejected"] = True
            continue
        latex = revised
    report["final_checks"] = mechanical_checks(latex, section, known_figures)
    if not cfg.review_rounds:
        report["approved"] = None  # review disabled
        return latex, report
    report["approved"] = False
    log.warning("%s: not fully approved after %d review round(s); see coverage.json", what, cfg.review_rounds)
    return latex, report


def write_sections(
    client: ClaudeClient,
    outline: Outline,
    items: list[Item],
    drafts: dict[str, str],
    units: list[SeedUnit],
    figures: dict[str, Figure],
    cfg: Config,
) -> list[str]:
    """Write, review, and cache every section of the outline, in order."""
    by_id = {i.id: i for i in items}
    unit_by_id = {u.id: u for u in units}
    known_figures = set(figures)
    system = section_system(cfg.ground, cfg.guide)
    cache_dir = cfg.tutorial_dir / "sections"
    written: list[str] = []
    reports: list[dict] = []
    keys_in_use: list[str] = []

    for idx, section in enumerate(outline.sections):
        what = f"Section {idx + 1}/{len(outline.sections)} '{section.title}'"
        sec_items = [by_id[i] for i in section.items if i in by_id]
        items_text = _items_text(sec_items)
        sec_figures = [figures[f] for f in section.figures if f in figures]
        request = section_user(
            title=outline.title,
            overview=outline.overview(idx),
            notation=outline.notation_text(),
            spec=section.spec(),
            items=items_text,
            material=_material(sec_items, drafts, unit_by_id),
            figure_lines=figure_lines(sec_figures),
            previous_tail=written[-1][-_PREVIOUS_TAIL_CHARS:] if written else "",
            cite_keys=", ".join(keys_in_use),
        )
        cache = cache_dir / f"section_{idx:03d}.tex"
        report_cache = cache_dir / f"section_{idx:03d}.review.json"
        key = pass_key(
            cfg, "section", system, request, cfg.review_rounds,
            review_system(cfg.ground, cfg.guide), revise_system(cfg.ground, cfg.guide),
        )
        latex = read_cached(cache, key, cfg.force)
        report_text = read_cached(report_cache, key, cfg.force)
        if latex is not None and report_text is not None:
            log.info("%s: cached", what)
            reports.append(json.loads(report_text))
        else:
            # The first version is cached on its own, so a run that dies during
            # review resumes at the review instead of rewriting the section.
            first_cache = cache_dir / f"section_{idx:03d}.first.tex"
            first_key = pass_key(cfg, "section-first", system, request)
            latex = read_cached(first_cache, first_key, cfg.force)
            if latex is None:
                log.info("%s: writing (%d items, %d figures)%s", what, len(sec_items), len(sec_figures),
                         " + research" if cfg.ground else "")
                latex = generate_valid(
                    lambda: generate(client, cfg, system=system, text=request, images=image_refs(sec_figures)),
                    lambda t: check_fragment(t, SECTION_END),
                    what,
                    cfg.generation_attempts,
                )
                latex = without_sentinel(latex, SECTION_END)
                write_cached(first_cache, first_key, latex)
            latex, report = _review_and_revise(
                client, cfg, outline, idx, section, items_text, request, latex, sec_figures, known_figures, what
            )
            write_cached(cache, key, latex)
            write_cached(report_cache, key, dump(report))
            reports.append(report)
        written.append(latex)
        keys_in_use += [k for k in cite_keys(latex) if k not in keys_in_use]

    (cfg.tutorial_dir / "coverage.json").write_text(dump(reports), encoding="utf-8")
    unapproved = [r["section"] for r in reports if r.get("approved") is False]
    if unapproved:
        log.warning("%d section(s) not fully approved by review: %s", len(unapproved), "; ".join(unapproved))
    return written
