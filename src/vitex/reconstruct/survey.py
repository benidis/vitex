"""Survey the topic (a literature review) and size the tutorial from it.

The seeds say what material is available; they do not say what matters in the
field. The survey maps the topic itself — its important methods and concepts,
each with a role (core / supporting / brief), a reason, and key references,
including important topics the seeds miss — using the research tools. The
outline follows this map to decide what to develop fully, what to keep compact,
and what to add.

The length budget is derived from the same map: a word allowance per distinct
topic by role. It therefore depends on the scope of the topic, not on how much
source material there is — five overlapping documents describe the same topics
and get the same budget as one. ``cfg.pages`` overrides it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from vitex.config import Config
from vitex.llm.client import ClaudeClient, staged
from vitex.llm.prompts import SURVEY_SCHEMA, survey_system, survey_user
from vitex.models import SeedUnit
from vitex.reconstruct.chapters import dump, generate_structured, pass_key, structured_valid
from vitex.reconstruct.outline import Item
from vitex.utils import read_cached, write_cached

log = logging.getLogger("vitex")

# Words per topic by role, plus the introduction and summary. Calibrated so a
# core method gets its full treatment (setup, intuition, derivation, algorithm,
# and — where it matters — cost and an example) in a few pages.
WORDS_PER_TOPIC = {"core": 2500, "supporting": 900, "brief": 120}
_FRAME_WORDS = 3000  # introduction (with the overview of the field) + summary
WORDS_PER_PAGE = 450


@dataclass
class Survey:
    overview: str
    topics: list[dict] = field(default_factory=list)

    def landscape(self) -> str:
        """The map as text for the outline prompt."""
        lines = [f"Overview: {self.overview}", ""]
        for role in ("core", "supporting", "brief"):
            group = [t for t in self.topics if t.get("role") == role]
            if group:
                lines.append(f"{role.upper()} ({len(group)}):")
                for t in group:
                    refs = "; ".join(t.get("key_refs") or [])
                    seeds = "" if t.get("in_seeds", True) else " [NOT IN SEEDS]"
                    lines.append(f"  - {t['topic']}{seeds}: {t.get('why', '')}" + (f" (refs: {refs})" if refs else ""))
        return "\n".join(lines)

    def budget(self) -> int:
        """Words for the whole tutorial: an allowance per distinct topic by role."""
        return _FRAME_WORDS + sum(WORDS_PER_TOPIC.get(t.get("role", "brief"), 0) for t in self.topics)


def build_survey(
    client: ClaudeClient,
    units: list[SeedUnit],
    items: list[Item],
    summaries: dict[str, str],
    cfg: Config,
) -> Survey:
    """Map the topic with research; cached at ``work/tutorial/survey.json``."""
    sources = "\n".join(f"- {u.kind} \"{u.source_title}\", {u.position}: {summaries.get(u.id, '')}" for u in units)
    titles = sorted({i.title for i in items})
    system = survey_system(cfg.ground, cfg.guide)
    user = survey_user(cfg.title, sources, "; ".join(titles))
    cache = cfg.tutorial_dir / "survey.json"
    key = pass_key(cfg, "survey", system, user)
    cached = read_cached(cache, key, cfg.force)
    if cached is not None:
        log.info("Survey: cached")
        data = json.loads(cached)
    else:
        log.info("Survey: mapping the topic and its literature%s...", " + research" if cfg.ground else "")
        with staged(client, "survey"):
            data = structured_valid(
                lambda: generate_structured(
                    client, cfg, system=system, text=user, schema=SURVEY_SCHEMA,
                    description="Submit the literature map of the topic.", research_allowed=True,
                ),
                lambda d: None if d.get("topics") else "survey lists no topics",
                "Survey",
                cfg.generation_attempts,
            )
        write_cached(cache, key, dump(data))
    survey = Survey(overview=str(data.get("overview", "")), topics=list(data.get("topics", [])))
    roles = {r: sum(1 for t in survey.topics if t.get("role") == r) for r in WORDS_PER_TOPIC}
    missing = sum(1 for t in survey.topics if not t.get("in_seeds", True))
    log.info("Survey: %d topics (%d core, %d supporting, %d brief), %d not in the seeds",
             len(survey.topics), roles["core"], roles["supporting"], roles["brief"], missing)
    return survey
