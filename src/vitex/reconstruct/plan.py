"""The plan: what the tutorial includes, at what level, and why.

The plan is the scoping decision of a tutorial, made before any writing: the
scope (reader, goal, throughline) and every candidate method/topic with a level
— main line (full treatment), stepping stone (compact but proper), mention (a
few sentences), excluded — a page allocation, and a one-line reason. It is
built from the survey's literature map and the author's brief, under a page
budget that is spent by cutting breadth, never depth.

It is stored as ``plan.md`` in the run dir, in a markdown format meant to be
edited by hand: move a line to another level, change pages, add or delete
entries, rewrite the scope. ``--plan-feedback`` revises it from notes instead;
every revision keeps the previous version as ``plan.vN.md``. ``--plan FILE``
runs with exactly a given plan. Everything downstream follows the plan: drafts
develop what it includes, the outline places its entries, and section lengths
come from its pages.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from vitex.config import Config
from vitex.errors import VitexError
from vitex.llm.client import ClaudeClient, staged
from vitex.llm.prompts import PLAN_LEVELS, PLAN_SCHEMA, plan_revise_user, plan_system, plan_user
from vitex.reconstruct.chapters import generate_structured, structured_valid
from vitex.reconstruct.survey import WORDS_PER_PAGE, Survey
from vitex.utils import fingerprint

log = logging.getLogger("vitex")

FRAME_PAGES = 4.0  # introduction (with an overview of the field) + summary
MENTION_PAGES = 0.15  # a few sentences in context
# Minimum pages for a proper treatment at each level; below it an entry is
# demoted one level (a half-page "stepping stone" is really a mention).
MIN_PAGES = {"main": 1.5, "stepping": 0.75}

_HEADINGS = {
    "main": "Main line — full treatment",
    "stepping": "Stepping stones — compact but proper",
    "mention": "Mentions — a few sentences in context",
    "excluded": "Excluded",
}
_INPUTS_RE = re.compile(r"<!--\s*vitex-plan\s+inputs=(\w+)\s*-->")
_BUDGET_RE = re.compile(r"^budget:\s*([\d.]+)", re.IGNORECASE)
_PAGES_RE = re.compile(r"^\s*([\d.]+)\s*(pages?|pp?\.?)?\s*$", re.IGNORECASE)


class PlanError(VitexError):
    """A plan file could not be read."""


@dataclass
class PlanEntry:
    name: str
    level: str  # main | stepping | mention | excluded
    pages: float
    reason: str


@dataclass
class Plan:
    title: str
    budget_pages: float
    reader: str
    goal: str
    throughline: str
    entries: list[PlanEntry] = field(default_factory=list)
    inputs: str = ""  # fingerprint of what the plan was made from

    def level(self, level: str) -> list[PlanEntry]:
        return [e for e in self.entries if e.level == level]

    @property
    def planned_pages(self) -> float:
        return sum(e.pages for e in self.entries if e.level in ("main", "stepping"))

    def frame_pages(self) -> float:
        """Pages left for the introduction and summary (at least 2)."""
        rest = self.budget_pages - self.planned_pages - MENTION_PAGES * len(self.level("mention"))
        return max(2.0, rest)

    def to_markdown(self) -> str:
        lines = [
            f"# Plan: {self.title}",
            f"<!-- vitex-plan inputs={self.inputs} -->",
            "<!-- Edit freely: move lines between levels, change pages, add/remove entries.",
            "     Lines: '- name | pages | reason' (main/stepping), '- name | reason' (mention/excluded). -->",
            f"Budget: {self.budget_pages:g} pages",
            "",
            "## Scope",
            f"Reader: {self.reader}",
            f"Goal: {self.goal}",
            f"Throughline: {self.throughline}",
        ]
        for level in PLAN_LEVELS:
            lines += ["", f"## {_HEADINGS[level]}"]
            for e in self.level(level):
                if level in ("main", "stepping"):
                    lines.append(f"- {e.name} | {e.pages:g} | {e.reason}")
                else:
                    lines.append(f"- {e.name} | {e.reason}")
        lines += ["", f"<!-- planned: {self.planned_pages:g} pages + ~{self.frame_pages():g} for introduction/summary -->"]
        return "\n".join(lines) + "\n"

    def scope_text(self) -> str:
        """The plan as a compact brief for drafting and outlining prompts."""
        parts = [f"Reader: {self.reader}", f"Goal: {self.goal}", f"Throughline: {self.throughline}"]
        for level in PLAN_LEVELS:
            entries = self.level(level)
            if not entries:
                continue
            parts.append(f"{_HEADINGS[level].upper()}:")
            for e in entries:
                pages = f" ({e.pages:g} pages)" if level in ("main", "stepping") else ""
                parts.append(f"  - {e.name}{pages}: {e.reason}")
        return "\n".join(parts)


def parse_plan(text: str) -> Plan:
    """Read a plan in the markdown format of :meth:`Plan.to_markdown`."""
    title, budget, inputs = "", 0.0, ""
    scope: dict[str, str] = {}
    entries: list[PlanEntry] = []
    level: str | None = None
    problems: list[str] = []
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if m := _INPUTS_RE.search(line):
            inputs = m.group(1)
        if not line or line.startswith("<!--") or line.endswith("-->"):
            continue
        if line.startswith("# "):
            title = re.sub(r"^#\s*plan:\s*", "", line, flags=re.IGNORECASE).strip()
            continue
        if m := _BUDGET_RE.match(line):
            budget = float(m.group(1))
            continue
        if line.startswith("## "):
            head = line[3:].lower()
            level = ("scope" if "scope" in head else "main" if "main" in head else
                     "stepping" if "stepping" in head else "mention" if "mention" in head else
                     "excluded" if "exclud" in head else None)
            if level is None:
                problems.append(f"line {n}: unknown heading {line!r}")
            continue
        if level == "scope":
            key, _, value = line.partition(":")
            if key.strip().lower() in ("reader", "goal", "throughline"):
                scope[key.strip().lower()] = value.strip()
            continue
        if not line.startswith("- ") or level is None:
            problems.append(f"line {n}: not an entry or heading: {line!r}")
            continue
        parts = [p.strip() for p in line[2:].split("|")]
        name = parts[0]
        if level in ("main", "stepping"):
            m = _PAGES_RE.match(parts[1]) if len(parts) >= 2 else None
            if not m:
                problems.append(f"line {n}: expected '- name | pages | reason', got {line!r}")
                continue
            entries.append(PlanEntry(name, level, float(m.group(1)), " | ".join(parts[2:])))
        else:
            reason = parts[-1] if len(parts) >= 2 else ""
            entries.append(PlanEntry(name, level, 0.0, reason))
    if not any(e.level == "main" for e in entries):
        problems.append("the plan has no main-line entries")
    if problems:
        raise PlanError("Cannot read the plan:\n  " + "\n  ".join(problems))
    return Plan(
        title=title, budget_pages=budget or FRAME_PAGES + sum(e.pages for e in entries),
        reader=scope.get("reader", ""), goal=scope.get("goal", ""),
        throughline=scope.get("throughline", ""), entries=entries, inputs=inputs,
    )


def load_plan(path: Path) -> Plan:
    plan = parse_plan(path.read_text(encoding="utf-8"))
    if plan.planned_pages + FRAME_PAGES > plan.budget_pages * 1.05:
        log.warning("Plan %s allocates %.0f pages (+~%.0f for the frame) against a budget of %.0f",
                    path, plan.planned_pages, FRAME_PAGES, plan.budget_pages)
    return plan


def save_plan(plan: Plan, path: Path) -> None:
    """Write ``plan.md``, keeping a different previous version as ``plan.vN.md``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = plan.to_markdown()
    if path.exists() and path.read_text(encoding="utf-8") != text:
        n = 1
        while (old := path.with_name(f"{path.stem}.v{n}{path.suffix}")).exists():
            n += 1
        path.rename(old)
        log.info("Plan: previous version kept as %s", old)
    path.write_text(text, encoding="utf-8")


def plan_diff(old: Plan, new: Plan) -> list[str]:
    """Human-readable changes between two plans."""
    before = {e.name.lower(): e for e in old.entries}
    after = {e.name.lower(): e for e in new.entries}
    lines = []
    for key, e in after.items():
        if key not in before:
            lines.append(f"+ added {e.name} as {e.level}" + (f" ({e.pages:g} pages)" if e.pages else ""))
        else:
            b = before[key]
            if b.level != e.level:
                lines.append(f"~ {e.name}: {b.level} -> {e.level}")
            elif abs(b.pages - e.pages) > 0.01:
                lines.append(f"~ {e.name}: {b.pages:g} -> {e.pages:g} pages")
    lines += [f"- removed {b.name}" for key, b in before.items() if key not in after]
    if old.budget_pages != new.budget_pages:
        lines.append(f"~ budget: {old.budget_pages:g} -> {new.budget_pages:g} pages")
    for name in ("reader", "goal", "throughline"):
        if getattr(old, name) != getattr(new, name):
            lines.append(f"~ {name} changed")
    return lines or ["(no changes)"]


def enforce_minimums(entries: list[PlanEntry]) -> list[str]:
    """Demote entries whose pages cannot hold a proper treatment at their level."""
    changes = []
    for e in entries:
        if e.level == "main" and e.pages < MIN_PAGES["main"]:
            e.level = "stepping"
            changes.append(f"{e.name}: main -> stepping ({e.pages:g} pages is below a full treatment)")
        if e.level == "stepping" and e.pages < MIN_PAGES["stepping"]:
            e.level, e.pages = "mention", 0.0
            changes.append(f"{e.name}: stepping -> mention (below a proper compact treatment)")
    return changes


def _from_answer(data: dict, budget: float, inputs: str) -> Plan:
    entries = [
        PlanEntry(str(e["name"]).strip(), str(e["level"]),
                  float(e.get("pages") or 0) if e["level"] in ("main", "stepping") else 0.0,
                  str(e.get("reason", "")).strip())
        for e in data.get("entries", [])
        if e.get("level") in PLAN_LEVELS and str(e.get("name", "")).strip()
    ]
    for change in enforce_minimums(entries):
        log.info("Plan: %s", change)
    return Plan(title=str(data.get("title", "")).strip(), budget_pages=budget, reader=str(data.get("reader", "")),
                goal=str(data.get("goal", "")), throughline=str(data.get("throughline", "")),
                entries=entries, inputs=inputs)


def _budget(cfg: Config, survey: Survey) -> tuple[float, str]:
    if cfg.pages:
        return float(cfg.pages), f"{cfg.pages} pages (fixed by the author)"
    ceiling = max(20, round(survey.budget() / WORDS_PER_PAGE))
    return float(ceiling), f"up to {ceiling} pages; use what the scope needs"


def _ask(client: ClaudeClient, cfg: Config, user: str, what: str) -> dict:
    with staged(client, what.lower()):
        return structured_valid(
            lambda: generate_structured(
                client, cfg, system=plan_system(cfg.guide), text=user, schema=PLAN_SCHEMA,
                description="Submit the plan of the tutorial.", research_allowed=True,
            ),
            lambda d: None if any(e.get("level") == "main" for e in d.get("entries", [])) else "no main-line entries",
            what,
            cfg.generation_attempts,
        )


def resolve_plan(client: ClaudeClient, survey: Survey, cfg: Config, source_ids: list[str]) -> Plan:
    """The plan for this run: given (``--plan``), saved, revised, or new."""
    if cfg.plan_path is not None:
        given = load_plan(cfg.plan_path)
        log.info("Plan: using %s as given (%d entries, %.0f pages)", cfg.plan_path, len(given.entries), given.budget_pages)
        return given

    budget, budget_text = _budget(cfg, survey)
    inputs = fingerprint("plan", cfg.model, survey.overview, survey.topics, cfg.guide, cfg.title, cfg.pages, source_ids)[:16]
    path = cfg.run_dir / "plan.md"
    landscape = survey.landscape()

    plan: Plan | None = None
    if path.exists() and not cfg.replan:
        plan = load_plan(path)
        if plan.inputs != inputs and not cfg.plan_feedback:
            log.warning("Plan: the sources, brief, title, or page budget changed since %s was made; "
                        "using it anyway (run with --replan for a fresh plan)", path)
        else:
            log.info("Plan: using %s", path)
    if plan is None:
        log.info("Plan: deciding the scope (%s)%s...", budget_text, " + research" if cfg.ground else "")
        plan = _from_answer(_ask(client, cfg, plan_user(cfg.title, budget_text, landscape), "Plan"), budget, inputs)
        save_plan(plan, path)
    if cfg.plan_feedback:
        log.info("Plan: revising with your feedback...")
        user = plan_revise_user(plan.to_markdown(), cfg.plan_feedback, budget_text, landscape)
        revised = _from_answer(_ask(client, cfg, user, "Plan revision"), plan.budget_pages if not cfg.pages else budget, inputs)
        for line in plan_diff(plan, revised):
            log.info("  %s", line)
        plan = revised
        save_plan(plan, path)

    counts = {lvl: len(plan.level(lvl)) for lvl in PLAN_LEVELS}
    log.info("Plan: %d main, %d stepping, %d mentions, %d excluded; %.0f of %.0f pages planned -> %s",
             counts["main"], counts["stepping"], counts["mention"], counts["excluded"],
             plan.planned_pages + plan.frame_pages(), plan.budget_pages, path)
    return plan
