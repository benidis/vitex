"""Inventory every item of content in the drafts, then outline the tutorial.

The inventory is the checklist that guarantees nothing is dropped: each draft is
broken into items (concepts, derivations, results, examples, figures, ...) with
ids. The outline must place every item id in a section, in a pedagogical order
with a purpose and a step-by-step plan per section. It is checked in code; the
model gets ``outline_attempts`` tries to fix unplaced or unknown ids, and any
item still unplaced after that is attached deterministically (to the section
holding the most items from the same seed unit) — content is never dropped.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field

from vitex.config import Config
from vitex.errors import LLMError
from vitex.llm.client import ClaudeClient, staged
from vitex.llm.prompts import (
    INVENTORY_SCHEMA,
    INVENTORY_SYSTEM,
    OUTLINE_SCHEMA,
    inventory_user,
    outline_retry_user,
    outline_system,
    outline_user,
)
from vitex.models import Figure, SeedUnit
from vitex.reconstruct.chapters import dump, generate_structured, pass_key, structured_valid
from vitex.utils import read_cached, run_parallel, write_cached

log = logging.getLogger("vitex")


@dataclass
class Item:
    id: str  # e.g. "I3.07": unit 3, item 7
    unit_id: str
    kind: str
    title: str
    detail: str
    figures: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    importance: str = "core"  # core | supporting | brief

    def line(self, detail_chars: int = 400) -> str:
        detail = " ".join(self.detail.split())
        if len(detail) > detail_chars:
            detail = detail[:detail_chars] + "..."
        return f"{self.id} | {self.kind} | {self.importance} | {self.title} | {detail}"


@dataclass
class Section:
    title: str
    role: str
    purpose: str  # planning note: what it establishes and why it is needed here
    plan: list[str]
    items: list[str]
    figures: list[str]
    new_material: list[str] = field(default_factory=list)
    depth: str = "core"  # core | supporting
    target_words: int = 0
    topics: list[str] = field(default_factory=list)  # literature-map topics covered

    def spec(self) -> str:
        plan = "\n".join(f"  {i}. {step}" for i, step in enumerate(self.plan, 1))
        new = "\n".join(f"  - {m}" for m in self.new_material) or "  (none)"
        return (
            f"Title: {self.title}\nRole: {self.role}\nDepth: {self.depth}\n"
            f"Target length: about {self.target_words} words\nPurpose (planning note): {self.purpose}\n"
            f"Plan:\n{plan}\nBridging/prerequisite material to add:\n{new}"
        )


@dataclass
class Outline:
    title: str
    reader_background: str
    notation: list[dict]
    sections: list[Section]
    storyline: str = ""
    # Items left out at the author's explicit request: [{"id", "reason"}].
    excluded: list[dict] = field(default_factory=list)

    def overview(self, mark: int | None = None) -> str:
        """Section list for prompts; ``mark`` flags the section being worked on."""
        lines = [f"Storyline: {self.storyline}", f"Reader background: {self.reader_background}"]
        for i, s in enumerate(self.sections):
            flag = ">>> " if i == mark else "    "
            lines.append(f"{flag}{i + 1}. {s.title} [{s.role}] — {s.purpose}")
        return "\n".join(lines)

    def map(self, items: dict[str, "Item"], mark: int | None = None) -> str:
        """Every section with its purpose and the content items it covers.

        Gives each writer (and the editor) a concrete view of the whole: what the
        reader already has before a section and what comes after it.
        """
        lines = [f"Storyline: {self.storyline}", f"Reader background: {self.reader_background}"]
        for i, s in enumerate(self.sections):
            flag = ">>> " if i == mark else "    "
            lines.append(f"{flag}{i + 1}. {s.title} [{s.role}, {s.depth}, ~{s.target_words} words] — {s.purpose}")
            covered = [f"{items[x].title} ({items[x].kind})" for x in s.items if x in items]
            if covered:
                lines.append("        covers: " + "; ".join(covered))
            if s.new_material:
                lines.append("        adds: " + "; ".join(s.new_material))
        return "\n".join(lines)

    def notation_text(self) -> str:
        return "\n".join(f"  {n.get('symbol', '')}: {n.get('meaning', '')}" for n in self.notation)

    def to_json(self) -> str:
        return dump(asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> "Outline":
        sections = [
            Section(
                title=str(s.get("title", "")).strip(),
                role=str(s.get("role", "core")),
                # Outlines cached before "purpose" existed carried question/motivation.
                purpose=str(s.get("purpose") or " ".join(
                    str(s.get(k, "")) for k in ("question", "motivation") if s.get(k)
                )),
                plan=[str(p) for p in s.get("plan", [])],
                items=[str(i).strip() for i in s.get("items", [])],
                figures=[str(f).strip() for f in s.get("figures", [])],
                new_material=[str(m) for m in s.get("new_material", [])],
                depth=str(s.get("depth") or "core"),
                target_words=int(s.get("target_words") or 0),
                topics=[str(t).strip() for t in s.get("topics", [])],
            )
            for s in d.get("sections", [])
        ]
        return cls(
            title=str(d.get("title", "")).strip(),
            reader_background=str(d.get("reader_background", "")),
            notation=list(d.get("notation", [])),
            sections=sections,
            storyline=str(d.get("storyline", "")),
            excluded=[e for e in d.get("excluded_items", d.get("excluded", [])) if isinstance(e, dict)],
        )


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------


def build_inventory(
    client: ClaudeClient, units: list[SeedUnit], drafts: dict[str, str], cfg: Config
) -> tuple[list[Item], dict[str, str]]:
    """Return every content item (with global ids) and a summary per unit."""
    items: list[Item] = []
    summaries: dict[str, str] = {}
    cache_dir = cfg.workdir / "drafts"

    def check(data: dict) -> str | None:
        return None if data.get("items") else "no items extracted"

    def one(u_idx: int) -> dict:
        unit = units[u_idx - 1]
        what = f"Inventory {u_idx}/{len(units)} [{unit.id}]"
        user = inventory_user(unit.source_title, unit.position, drafts[unit.id])
        cache = cache_dir / f"{unit.id}.items.json"
        key = pass_key(cfg, "inventory", INVENTORY_SYSTEM, user)
        cached = read_cached(cache, key, cfg.force)
        if cached is not None:
            data = json.loads(cached)
            log.info("%s: cached (%d items)", what, len(data["items"]))
            return data
        with staged(client, f"inventory {u_idx}/{len(units)}"):
            data = structured_valid(
                lambda: generate_structured(
                    client, cfg, system=INVENTORY_SYSTEM, text=user, schema=INVENTORY_SCHEMA,
                    description="Record the complete inventory of the draft.", research_allowed=False,
                ),
                check,
                what,
                cfg.generation_attempts,
            )
        write_cached(cache, key, dump(data))
        log.info("%s: %d items", what, len(data["items"]))
        return data

    inventories = run_parallel(one, list(range(1, len(units) + 1)), cfg.parallel)
    for u_idx, (unit, data) in enumerate(zip(units, inventories), 1):
        summaries[unit.id] = str(data.get("summary", ""))
        for n, raw in enumerate(data["items"], 1):
            items.append(
                Item(
                    id=f"I{u_idx}.{n:02d}",
                    unit_id=unit.id,
                    kind=str(raw.get("kind", "concept")),
                    title=str(raw.get("title", "")),
                    detail=str(raw.get("detail", "")),
                    figures=[str(f) for f in raw.get("figures", [])],
                    depends_on=[str(d) for d in raw.get("depends_on", [])],
                    importance=str(raw.get("importance") or "core"),
                )
            )
    return items, summaries


# ---------------------------------------------------------------------------
# Outline
# ---------------------------------------------------------------------------


def outline_problems(outline: Outline, items: list[Item], figures: set[str]) -> tuple[list[str], list[str]]:
    """Return (unplaced item ids, unknown ids/figure names) for an outline.

    Items the outline excluded (allowed only under an author's brief) count as
    placed.
    """
    known = {i.id for i in items}
    placed = {i for s in outline.sections for i in s.items} | {str(e.get("id")) for e in outline.excluded}
    used_figs = {f for s in outline.sections for f in s.figures}
    missing = [i.id for i in items if i.id not in placed]
    unknown = sorted((placed - known) | (used_figs - figures))
    return missing, unknown


def repair_outline(outline: Outline, items: list[Item], figures: set[str]) -> list[str]:
    """Make the outline complete in code; return a description of each change.

    Unknown ids and figure names are removed; every unplaced item, and every
    figure an item uses but no section includes, is attached to the section
    holding the most items from the same seed unit (or the last core section).
    """
    by_id = {i.id: i for i in items}
    changes: list[str] = []
    for s in outline.sections:
        bad = [i for i in s.items if i not in by_id] + [f for f in s.figures if f not in figures]
        if bad:
            changes.append(f"removed unknown {bad} from '{s.title}'")
        s.items = [i for i in s.items if i in by_id]
        s.figures = [f for f in s.figures if f in figures]

    body = [s for s in outline.sections if s.role not in ("introduction", "summary")] or outline.sections

    def home(item: Item) -> Section:
        def score(s: Section) -> int:
            return sum(1 for i in s.items if by_id[i].unit_id == item.unit_id)

        best = max(body, key=score)
        return best if score(best) > 0 else body[-1]

    placed = {i for s in outline.sections for i in s.items} | {str(e.get("id")) for e in outline.excluded}
    for item in items:
        if item.id not in placed:
            target = home(item)
            target.items.append(item.id)
            target.plan.append(f"Cover in full: {item.title}")
            changes.append(f"attached {item.id} ({item.title}) to '{target.title}'")

    for item in items:
        for fig in item.figures:
            if fig in figures and not any(fig in s.figures for s in outline.sections):
                target = next(s for s in outline.sections if item.id in s.items)
                target.figures.append(fig)
                changes.append(f"attached figure {fig} to '{target.title}'")
    return changes


# Words per literature-map topic by role: the relative weights that decide how
# the budget is split (and, in the survey, how large the default budget is).
ROLE_WORDS = {"core": 2500, "supporting": 900, "brief": 120}
# Fixed shares for the frame of the tutorial.
_FRAME_SHARE = {"introduction": 0.05, "summary": 0.04}
_MIN_SECTION_WORDS = 300


def _norm(name: str) -> str:
    return " ".join(name.lower().split())


def topic_problems(outline: Outline, topics: list[dict]) -> tuple[list[str], list[str]]:
    """Return (core/supporting map topics no section covers, unknown section topics)."""
    known = {_norm(t["topic"]): t for t in topics}
    listed = {_norm(x) for s in outline.sections for x in s.topics}
    uncovered = [t["topic"] for t in topics if t.get("role") in ("core", "supporting") and _norm(t["topic"]) not in listed]
    unknown = sorted({x for s in outline.sections for x in s.topics if _norm(x) not in known})
    return uncovered, unknown


def allocate(outline: Outline, topics: list[dict], budget: int) -> list[str]:
    """Set every section's word target from the roles of the topics it covers.

    The introduction and summary get fixed shares; the rest of the budget is
    split in proportion to each section's topic weight (core 2500 : supporting
    900 : brief 120; a topic listed in several sections is split between them),
    so important topics get proportionally more room by construction. Sections
    that cover no map topic keep the outline's estimate within the remainder.
    Returns a printable allocation table.
    """
    role = {_norm(t["topic"]): t.get("role", "brief") for t in topics}
    listings: dict[str, int] = {}
    for s in outline.sections:
        for x in {_norm(x) for x in s.topics}:
            listings[x] = listings.get(x, 0) + 1
    frame = {i: _FRAME_SHARE[s.role] * budget for i, s in enumerate(outline.sections) if s.role in _FRAME_SHARE}
    weights: dict[int, float] = {}
    for i, s in enumerate(outline.sections):
        if i in frame:
            continue
        w = sum(ROLE_WORDS[role[x]] / listings[x] for x in {_norm(x) for x in s.topics} if x in role)
        weights[i] = w if w > 0 else float(max(s.target_words, _MIN_SECTION_WORDS))
    pool = budget - sum(frame.values())
    total_w = sum(weights.values()) or 1.0
    for i, s in enumerate(outline.sections):
        share = frame[i] if i in frame else pool * weights[i] / total_w
        s.target_words = max(_MIN_SECTION_WORDS, round(share))

    by_depth: dict[str, int] = {}
    for s in outline.sections:
        key = s.role if s.role in _FRAME_SHARE else s.depth
        by_depth[key] = by_depth.get(key, 0) + s.target_words
    total = sum(by_depth.values()) or 1
    lines = [f"Allocation of {budget} words (~{budget // 450} pages):"]
    lines += [f"  {k}: {v} words ({100 * v / total:.0f}%)" for k, v in sorted(by_depth.items(), key=lambda kv: -kv[1])]
    lines += [f"  {i + 1:2}. {s.target_words:6} words  [{s.depth:10}] {s.title}" for i, s in enumerate(outline.sections)]
    return lines


def fit_budget(outline: Outline, items: list[Item], budget: int) -> list[str]:
    """Make the section targets add up to ``budget`` words; return what changed.

    Sections without a target get one in proportion to their items (core items
    weigh more); then all targets are scaled together if their sum is off by
    more than 15%, so the outline's relative allocation is kept.
    """
    by_id = {i.id: i for i in items}
    weight = {"core": 3.0, "supporting": 1.5, "brief": 0.5}
    changes: list[str] = []
    missing = [s for s in outline.sections if s.target_words <= 0]
    if missing:
        set_total = sum(s.target_words for s in outline.sections if s.target_words > 0)
        weights = {id(s): 1.0 + sum(weight.get(by_id[i].importance, 1.0) for i in s.items if i in by_id) for s in missing}
        pool = max(budget - set_total, budget * len(missing) // max(1, len(outline.sections)))
        total_w = sum(weights.values())
        for s in missing:
            s.target_words = max(300, round(pool * weights[id(s)] / total_w))
        changes.append(f"assigned targets to {len(missing)} section(s) without one")
    total = sum(s.target_words for s in outline.sections)
    if total and abs(total - budget) > 0.15 * budget:
        scale = budget / total
        for s in outline.sections:
            s.target_words = max(300, round(s.target_words * scale))
        changes.append(f"scaled section targets by {scale:.2f} ({total} -> {budget} words)")
    return changes


def build_outline(
    client: ClaudeClient,
    units: list[SeedUnit],
    items: list[Item],
    summaries: dict[str, str],
    figures: list[Figure],
    cfg: Config,
    budget_words: int,
    landscape: str = "",
    topics: list[dict] | None = None,
) -> Outline:
    """Design the tutorial's outline, guaranteed to place every item."""
    fig_names = {f.name for f in figures}
    sources = "\n".join(
        f"- [{u.id}] {u.kind} \"{u.source_title}\", {u.position}: {summaries.get(u.id, '')}" for u in units
    )
    fig_text = "\n".join(
        f"  {f.name} | {f.note.value if f.note else '-'} | {f.note.description if f.note else f.hint}"
        for f in figures
    )
    inventory = "\n".join(i.line() for i in items)
    system = outline_system(cfg.ground, cfg.guide)
    user = outline_user(sources, inventory, fig_text, cfg.title, budget_words, landscape or "(no survey)")

    cache = cfg.tutorial_dir / "outline.json"
    key = pass_key(cfg, "outline", system, user, cfg.outline_attempts)
    cached = read_cached(cache, key, cfg.force)
    if cached is not None:
        log.info("Outline: cached")
        return Outline.from_dict(json.loads(cached))

    def check(data: dict) -> str | None:
        return None if data.get("sections") else "outline has no sections"

    log.info("Outlining %d items from %d seed units%s...", len(items), len(units), " + research" if cfg.ground else "")
    text = user
    outline: Outline | None = None
    for attempt in range(1, cfg.outline_attempts + 1):
        with staged(client, f"outline {attempt}/{cfg.outline_attempts}"):
            data = structured_valid(
                lambda: generate_structured(
                    client, cfg, system=system, text=text, schema=OUTLINE_SCHEMA,
                    description="Submit the complete outline of the tutorial.", research_allowed=True,
                ),
                check,
                "Outline",
                cfg.generation_attempts,
            )
        outline = Outline.from_dict(data)
        uncovered, bad_topics = topic_problems(outline, topics) if topics else ([], [])
        if not cfg.guide and outline.excluded:
            # Exclusion is only legitimate when the author's brief asks for it.
            log.warning("Outline excluded %d item(s) without an author's brief; restoring them", len(outline.excluded))
            outline.excluded = []
        missing, unknown = outline_problems(outline, items, fig_names)
        if not missing and not unknown and not uncovered and not bad_topics:
            break
        log.warning(
            "Outline attempt %d/%d: %d unplaced item(s), %d unknown reference(s), %d uncovered topic(s), %d unknown topic(s)",
            attempt, cfg.outline_attempts, len(missing), len(unknown), len(uncovered), len(bad_topics),
        )
        text = user + "\n\n" + outline_retry_user(
            dump(data), ", ".join(missing), ", ".join(unknown), "; ".join(uncovered), "; ".join(bad_topics)
        )
    if outline is None:  # pragma: no cover - loop runs at least once
        raise LLMError("Outline: no outline produced.")

    for change in repair_outline(outline, items, fig_names):
        log.warning("Outline fix: %s", change)
    if topics:
        uncovered, _ = topic_problems(outline, topics)
        for name in uncovered:
            log.warning("Outline: map topic not covered by any section: %s", name)
        for line in allocate(outline, topics, budget_words):
            log.info("%s", line)
    else:
        for change in fit_budget(outline, items, budget_words):
            log.info("Outline budget: %s", change)
    for e in outline.excluded:
        log.warning("Outline: item %s left out per the author's brief: %s", e.get("id"), e.get("reason", ""))
    if cfg.title:
        outline.title = cfg.title
    write_cached(cache, key, outline.to_json())
    log.info("Outline: %d sections, %d words budgeted — %s", len(outline.sections),
             sum(s.target_words for s in outline.sections), " | ".join(s.title for s in outline.sections))
    return outline
