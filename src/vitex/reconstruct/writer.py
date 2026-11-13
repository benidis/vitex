"""Write the tutorial section by section from the outline, review each, then
edit the whole.

Holistic view, before writing: every section is written with a map of the whole
tutorial — each section's purpose and the content items it covers — so the
writer knows exactly what the reader already has and what comes later. It also
gets its step-by-step plan, the full details of its items, the fact-checked
drafts they come from, its figures, the plans of the neighbouring sections, and
the citation keys the drafts use.

Per section: up to ``cfg.review_rounds`` rounds of review and revision check
coverage of every assigned item, step-by-step rigor, depth, flow, and figures
(plus checks in code). Sections do not depend on each other's text, so they are
written and reviewed in parallel (``cfg.parallel`` at a time).

Holistic view, after writing: an editor reads the whole assembled tutorial and
reports what only shows across sections — repetition, inconsistent notation or
terms, forward references, transitions that don't fit, uneven depth,
contradictions, storyline gaps. The affected sections are revised in parallel
with the editor's notes and the text of the related sections; then the editor
reads again (``cfg.editorial_rounds``). Revisions that drop a large share of a
section are rejected. Outcomes go to ``coverage.json`` and ``editorial.json``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from vitex.config import Config
from vitex.llm.client import ClaudeClient, staged
from vitex.llm.prompts import (
    EDITORIAL_SCHEMA,
    REVIEW_SCHEMA,
    SECTION_END,
    edit_fix_user,
    editor_system,
    editor_user,
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
from vitex.utils import fingerprint, read_cached, run_parallel, word_count, write_cached

log = logging.getLogger("vitex")

_MATERIAL_CHARS = 320_000  # cap on draft material per section prompt (~80k tokens)
_MIN_KEPT_FRACTION = 0.85  # a review revision shorter than this share is rejected...
_OVER_BUDGET = 1.15  # ...unless the section is this far over its target: then it may be
_FLOOR_OF_TARGET = 0.7  # cut, but not below this share of the target (stops a collapsed reply)
_UNDER_BUDGET = 0.5  # a section below this share of its target is flagged
# Editorial fixes may remove repetition, so they may shrink a section more; the
# removed material exists elsewhere by construction.
_MIN_KEPT_EDITORIAL = 0.6
_RELATED_CHARS = 120_000  # cap on related-section text sent with editorial fixes


@dataclass
class SectionJob:
    """Everything needed to write, review, or revise one section."""

    idx: int
    section: Section
    label: str  # e.g. "section 3/14" (log stage)
    what: str  # e.g. "Section 3/14 'Policy Gradients'" (log messages)
    items_text: str
    map_text: str
    request: str  # the full writing brief
    figures: list[Figure]


def _items_text(items: list[Item]) -> str:
    return "\n".join(f"- {i.id} | {i.kind} | {i.importance} | {i.title}\n    {i.detail.strip()}" for i in items)


_FLOAT_WORDS = 150  # page space of a figure, table, or algorithm box, in words


def effective_words(latex: str) -> int:
    """Words plus the page space of floats, comparable to a word target."""
    floats = sum(latex.count(f"\\begin{{{env}}}") for env in ("figure", "table", "algorithm"))
    return word_count(latex) + _FLOAT_WORDS * floats


def length_note(latex: str, target: int) -> str:
    words = effective_words(latex)
    if target <= 0:
        return f"{words} words (no target)"
    return f"{words} words against a target of about {target} ({100 * words / target:.0f}%)"


def min_kept_words(before_words: int, target: int) -> float:
    """Shortest acceptable revision: revisions may not drop content, except that
    a section well over its target may be cut back toward it."""
    if target > 0 and before_words > _OVER_BUDGET * target:
        return _FLOOR_OF_TARGET * target
    return _MIN_KEPT_FRACTION * before_words


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


def _neighbors(outline: Outline, idx: int) -> str:
    """The plans of the sections just before and after ``idx``."""
    parts = []
    for label, j in (("Previous section", idx - 1), ("Next section", idx + 1)):
        if 0 <= j < len(outline.sections):
            parts.append(f"{label} ({j + 1}):\n{outline.sections[j].spec()}")
        else:
            parts.append(f"{label}: (none — this is the {'first' if j < 0 else 'last'} section)")
    return "\n\n".join(parts)


def section_jobs(
    outline: Outline,
    items: list[Item],
    drafts: dict[str, str],
    units: list[SeedUnit],
    figures: dict[str, Figure],
) -> list[SectionJob]:
    """Build the writing brief of every section."""
    by_id = {i.id: i for i in items}
    unit_by_id = {u.id: u for u in units}
    # One shared key list, so independently written sections cite a work alike.
    draft_keys: list[str] = []
    for d in drafts.values():
        draft_keys += [k for k in cite_keys(d) if k not in draft_keys]
    jobs = []
    n = len(outline.sections)
    for idx, section in enumerate(outline.sections):
        sec_items = [by_id[i] for i in section.items if i in by_id]
        items_text = _items_text(sec_items)
        sec_figures = [figures[f] for f in section.figures if f in figures]
        map_text = outline.map(by_id, mark=idx)
        request = section_user(
            title=outline.title,
            overview=map_text,
            notation=outline.notation_text(),
            spec=section.spec(),
            items=items_text,
            material=_material(sec_items, drafts, unit_by_id),
            figure_lines=figure_lines(sec_figures),
            neighbors=_neighbors(outline, idx),
            cite_keys=", ".join(draft_keys),
        )
        jobs.append(
            SectionJob(
                idx=idx, section=section, label=f"section {idx + 1}/{n}",
                what=f"Section {idx + 1}/{n} '{section.title}'", items_text=items_text,
                map_text=map_text, request=request, figures=sec_figures,
            )
        )
    return jobs


def mechanical_checks(latex: str, section: Section, known_figures: set[str]) -> list[str]:
    """Problems detectable in code, reported to the reviewer and reviser."""
    words, target = effective_words(latex), section.target_words
    problems: list[str] = []
    if target > 0 and words > _OVER_BUDGET * target:
        problems.append(
            f"the section is {words} words, {100 * words / target - 100:.0f}% over its target of {target}; "
            "cut what teaches least (proofs, tangents, repetition, extras that add little)"
        )
    elif target > 0 and words < _UNDER_BUDGET * target:
        problems.append(f"the section is {words} words, far below its target of {target}; check for under-explained ideas")
    used = referenced_figures(latex)
    problems += [f"assigned figure {f} is not included" for f in section.figures if f not in used]
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
    client: ClaudeClient, cfg: Config, job: SectionJob, latex: str, known_figures: set[str]
) -> tuple[str, dict]:
    """Return the final section text and a report of its reviews."""
    section = job.section
    report: dict = {"section": section.title, "items": section.items, "figures": section.figures, "rounds": []}
    rsystem, vsystem = review_system(cfg.ground, cfg.guide), revise_system(cfg.ground, cfg.guide)
    for rnd in range(1, cfg.review_rounds + 1):
        checks = mechanical_checks(latex, section, known_figures)
        ruser = review_user(
            overview=job.map_text, spec=section.spec(), items=job.items_text, section=latex,
            checks="\n".join(f"- {c}" for c in checks), length=length_note(latex, section.target_words),
        )
        with staged(client, f"{job.label} review {rnd}"):
            review = structured_valid(
                lambda: generate_structured(
                    client, cfg, system=rsystem, text=ruser, schema=REVIEW_SCHEMA,
                    description="Submit the review of the section.", research_allowed=True,
                ),
                lambda d: None if "approved" in d else "review lacks a verdict",
                f"{job.what} review {rnd}",
                cfg.generation_attempts,
            )
        missing, issues = review.get("missing_items", []), review.get("issues", [])
        report["rounds"].append({"review": review, "checks": checks})
        if review.get("approved") and not missing and not checks:
            log.info("%s: approved in review round %d", job.what, rnd)
            report["approved"] = True
            return latex, report
        log.info("%s: review %d — %d missing item(s), %d issue(s), %d check(s); revising",
                 job.what, rnd, len(missing), len(issues), len(checks))
        with staged(client, f"{job.label} revise {rnd}"):
            revised = generate_valid(
                lambda: generate(
                    client, cfg, system=vsystem,
                    text=revise_user(section_request=job.request, section=latex, review=_review_text(review, checks)),
                    images=image_refs(job.figures),
                ),
                lambda t: check_fragment(t, SECTION_END),
                f"{job.what} revision {rnd}",
                cfg.generation_attempts,
            )
        revised = without_sentinel(revised, SECTION_END)
        if effective_words(revised) < min_kept_words(effective_words(latex), section.target_words):
            log.warning("%s: revision %d cut %.0f%% of the text, more than allowed; keeping the previous version",
                        job.what, rnd, 100 * (1 - effective_words(revised) / max(1, effective_words(latex))))
            report["rounds"][-1]["revision_rejected"] = True
            continue
        latex = revised
    report["final_checks"] = mechanical_checks(latex, section, known_figures)
    if not cfg.review_rounds:
        report["approved"] = None  # review disabled
        return latex, report
    report["approved"] = False
    log.warning("%s: not fully approved after %d review round(s); see coverage.json", job.what, cfg.review_rounds)
    return latex, report


def _condense(client: ClaudeClient, cfg: Config, job: SectionJob, latex: str) -> str:
    """Bring a first version that is well over its target back toward it."""
    target, words = job.section.target_words, effective_words(latex)
    if target <= 0 or words <= _OVER_BUDGET * target:
        return latex
    log.info("%s: first version is %d words against a target of %d; condensing", job.what, words, target)
    notes = (
        f"- CHECK: the section is {words} words (figures, tables, and algorithms count ~{_FLOAT_WORDS} each) "
        f"against a target of {target}. Condense it to about {target}: cut what teaches least first "
        "(proofs, tangents, repetition, extra examples, over-developed minor points), keep the key "
        "derivations, and keep every assigned item at its level."
    )
    with staged(client, f"{job.label} condense"):
        condensed = generate_valid(
            lambda: generate(
                client, cfg, system=revise_system(cfg.ground, cfg.guide),
                text=revise_user(section_request=job.request, section=latex, review=notes),
                images=image_refs(job.figures),
            ),
            lambda t: check_fragment(t, SECTION_END),
            f"{job.what} condense",
            cfg.generation_attempts,
        )
    condensed = without_sentinel(condensed, SECTION_END)
    if effective_words(condensed) < min_kept_words(words, target):
        log.warning("%s: condensing cut too much; keeping the first version", job.what)
        return latex
    return condensed


def write_sections(
    client: ClaudeClient, jobs: list[SectionJob], figures: dict[str, Figure], cfg: Config
) -> list[str]:
    """Write, review, and cache every section (in parallel)."""
    system = section_system(cfg.ground, cfg.guide)
    cache_dir = cfg.tutorial_dir / "sections"
    known_figures = set(figures)

    def one(job: SectionJob) -> tuple[str, dict]:
        cache = cache_dir / f"section_{job.idx:03d}.tex"
        report_cache = cache_dir / f"section_{job.idx:03d}.review.json"
        key = pass_key(
            cfg, "section", system, job.request, cfg.review_rounds,
            review_system(cfg.ground, cfg.guide), revise_system(cfg.ground, cfg.guide),
        )
        latex = read_cached(cache, key, cfg.force)
        report_text = read_cached(report_cache, key, cfg.force)
        if latex is not None and report_text is not None:
            log.info("%s: cached", job.what)
            return latex, json.loads(report_text)

        # The first version is cached on its own, so a run that dies during
        # review resumes at the review instead of rewriting the section.
        first_cache = cache_dir / f"section_{job.idx:03d}.first.tex"
        first_key = pass_key(cfg, "section-first", system, job.request)
        latex = read_cached(first_cache, first_key, cfg.force)
        if latex is None:
            log.info("%s: writing (%d items, %d figures)%s", job.what, len(job.section.items),
                     len(job.figures), " + research" if cfg.ground else "")
            with staged(client, f"{job.label} write"):
                latex = generate_valid(
                    lambda: generate(client, cfg, system=system, text=job.request, images=image_refs(job.figures)),
                    lambda t: check_fragment(t, SECTION_END),
                    job.what,
                    cfg.generation_attempts,
                )
            latex = without_sentinel(latex, SECTION_END)
            latex = _condense(client, cfg, job, latex)
            write_cached(first_cache, first_key, latex)
        latex, report = _review_and_revise(client, cfg, job, latex, known_figures)
        write_cached(cache, key, latex)
        write_cached(report_cache, key, dump(report))
        log.info("%s: done", job.what)
        return latex, report

    log.info("Writing %d sections, %d at a time", len(jobs), cfg.parallel)
    results = run_parallel(one, jobs, cfg.parallel)
    written = [latex for latex, _ in results]
    reports = [report for _, report in results]

    cfg.tutorial_dir.mkdir(parents=True, exist_ok=True)
    (cfg.tutorial_dir / "coverage.json").write_text(dump(reports), encoding="utf-8")
    unapproved = [r["section"] for r in reports if r.get("approved") is False]
    if unapproved:
        log.warning("%d section(s) not fully approved by review: %s", len(unapproved), "; ".join(unapproved))
    return written


def _assembled(sections: list[str], jobs: list[SectionJob]) -> str:
    return "\n\n".join(
        f"% ===== SECTION {j.idx + 1}: {j.section.title} =====\n{text}" for j, text in zip(jobs, sections)
    )


def _issue_text(issue: dict) -> str:
    related = issue.get("related_sections") or []
    rel = f" (related: sections {', '.join(map(str, related))})" if related else ""
    return f"- [{issue.get('kind', 'other')}]{rel} at \"{issue.get('location', '')}\": {issue['problem']} -> FIX: {issue['fix']}"


def edit_tutorial(
    client: ClaudeClient,
    outline: Outline,
    items: list[Item],
    sections: list[str],
    jobs: list[SectionJob],
    cfg: Config,
) -> list[str]:
    """Editorial passes over the whole tutorial; returns the edited sections."""
    if not cfg.editorial_rounds or not sections:
        return sections
    by_id = {i.id: i for i in items}
    esystem, vsystem = editor_system(cfg.guide), revise_system(cfg.ground, cfg.guide)
    cache_dir = cfg.tutorial_dir / "editorial"
    sections = list(sections)
    report: list[dict] = []

    for rnd in range(1, cfg.editorial_rounds + 1):
        document = _assembled(sections, jobs)
        lengths = "\n".join(f"    section {j.idx + 1}: {length_note(t, j.section.target_words)}" for j, t in zip(jobs, sections))
        map_text = f"{outline.map(by_id)}\nActual lengths:\n{lengths}"
        euser = editor_user(map_text=map_text, notation=outline.notation_text(), document=document)
        cache = cache_dir / f"round_{rnd}.json"
        key = pass_key(cfg, "editor", esystem, euser)
        cached = read_cached(cache, key, cfg.force)
        if cached is not None:
            review = json.loads(cached)
            log.info("Editorial round %d: cached (%d issue(s))", rnd, len(review.get("issues", [])))
        else:
            log.info("Editorial round %d/%d: reading the whole tutorial (%d chars)", rnd, cfg.editorial_rounds, len(document))
            with staged(client, f"editor {rnd}/{cfg.editorial_rounds}"):
                review = structured_valid(
                    lambda: generate_structured(
                        client, cfg, system=esystem, text=euser, schema=EDITORIAL_SCHEMA,
                        description="Submit the editorial review of the whole tutorial.", research_allowed=False,
                    ),
                    lambda d: None if "issues" in d else "editorial review lacks an issue list",
                    f"Editorial round {rnd}",
                    cfg.generation_attempts,
                )
            write_cached(cache, key, dump(review))

        by_section: dict[int, list[dict]] = {}
        for issue in review.get("issues", []):
            n = issue.get("section")
            if isinstance(n, int) and 1 <= n <= len(sections):
                by_section.setdefault(n - 1, []).append(issue)
            else:
                log.warning("Editorial issue names no valid section; ignored: %s", issue.get("problem", "")[:120])
        round_report: dict = {"round": rnd, "assessment": review.get("assessment", ""), "issues": review.get("issues", []),
                              "revised": [], "rejected": []}
        report.append(round_report)
        if not by_section:
            log.info("Editorial round %d: no cross-section issues — the tutorial reads as a whole", rnd)
            break
        log.info("Editorial round %d: %d issue(s) in %d section(s); revising them, %d at a time",
                 rnd, sum(map(len, by_section.values())), len(by_section), cfg.parallel)

        snapshot = list(sections)  # fixes in this round all see the same text

        def fix(idx: int) -> tuple[int, str, bool]:
            job, issues = jobs[idx], by_section[idx]
            related_ids = sorted({r - 1 for i in issues for r in (i.get("related_sections") or [])
                                  if isinstance(r, int) and 1 <= r <= len(sections) and r - 1 != idx})
            related, size = [], 0
            for r in related_ids:
                block = f"% ===== SECTION {r + 1}: {jobs[r].section.title} =====\n{snapshot[r]}"
                if size + len(block) > _RELATED_CHARS:
                    break
                related.append(block)
                size += len(block)
            notes = "\n".join(_issue_text(i) for i in issues)
            text = edit_fix_user(section_request=job.request, section=snapshot[idx], notes=notes,
                                 related="\n\n".join(related))
            fcache = cache_dir / f"round_{rnd}_section_{idx:03d}.tex"
            fkey = fingerprint(key, idx, notes, snapshot[idx])
            revised = read_cached(fcache, fkey, cfg.force)
            if revised is None:
                with staged(client, f"editor {rnd} fix {job.label}"):
                    revised = generate_valid(
                        lambda: generate(client, cfg, system=vsystem, text=text, images=image_refs(job.figures)),
                        lambda t: check_fragment(t, SECTION_END),
                        f"{job.what} editorial fix {rnd}",
                        cfg.generation_attempts,
                    )
                revised = without_sentinel(revised, SECTION_END)
                write_cached(fcache, fkey, revised)
            if len(revised) < _MIN_KEPT_EDITORIAL * len(snapshot[idx]):
                log.warning("%s: editorial fix dropped %.0f%% of the text; keeping the previous version",
                            job.what, 100 * (1 - len(revised) / len(snapshot[idx])))
                return idx, snapshot[idx], False
            return idx, revised, True

        for idx, text, accepted in run_parallel(fix, sorted(by_section), cfg.parallel):
            sections[idx] = text
            (round_report["revised"] if accepted else round_report["rejected"]).append(idx + 1)

    cfg.tutorial_dir.mkdir(parents=True, exist_ok=True)
    (cfg.tutorial_dir / "editorial.json").write_text(dump(report), encoding="utf-8")
    return sections
