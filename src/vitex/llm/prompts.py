"""Prompt templates for every Claude pass.

The contract they encode, shared through ``TUTORIAL_STANDARD``, ``SEED_RULES``
and the research rules:

* The product is a TUTORIAL — not a summary, survey, or catalogue of methods. A
  reader with general technical background but no knowledge of the topic must be
  able to understand all of it, front to back, with no other material: problems
  are set up before solutions, every derivation is step by step, every symbol is
  defined, nothing appears out of the blue, and each section flows into the next.
* Videos and documents are SEEDS. Everything of substance in every seed must
  reach the tutorial — fact-checked against the literature, corrected where
  wrong, and expanded — so the result is more complete and more correct than any
  single source.

Passes: frame triage -> per-seed draft -> content inventory -> outline ->
section writing -> review/revision -> bibliography -> LaTeX repair.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Shared rules
# ---------------------------------------------------------------------------

TUTORIAL_STANDARD = r"""
THE STANDARD THE TUTORIAL MUST MEET
It is a TUTORIAL on its topic — not a summary, and not a survey of every method
that exists. The reader has general technical maturity (undergraduate linear
algebra, calculus, probability, programming) but is new to this topic. Reading
front to back, with no other material, they must understand everything that is
included.
1. Problem first. Introduce each idea through the problem it solves, set up in
   terms the reader already has, before the solution. Give the intuition, then
   the precise math, and tie the symbols back to the intuition.
2. Explain, don't compress. Write at the pace of a good textbook, in full
   sentences that let the reader follow every step. Explain each term —
   including hardware and systems terms — when it first appears. Derive what
   explains how a method works (its objective, gradient, update rule, key
   identities) step by step; state theorems with their intuition and a
   citation rather than proving them; assume general background math.
3. Depth follows the plan. Main-line entries get the full treatment: the
   problem, the intuition, the derivation, and — where they teach — the
   algorithm as it is run in practice, a cost analysis, and a worked example.
   Stepping stones get a compact but proper explanation: the problem they fix,
   the idea, the key equation, how they are used, and the trade-off. Mentions
   get a few sentences in context.
4. Fit by choosing, never by squeezing. Each section has a target length. If
   the material does not fit, leave out the least important parts entirely —
   an extra example, a side remark, a detail the reader can do without — and
   keep what remains fully explained.
5. Flow. One continuous text, built up from easier to harder; each part follows
   from what came before, and the reader always knows why it matters. Openings,
   transitions, and closings fit the section; there are no templates. One
   notation throughout.
6. Concrete where it helps: worked examples with real numbers where a mechanism
   is not obvious from the equations, and comparison tables to set related
   methods side by side.
""".strip()

SEED_RULES = r"""
THE SOURCES
The videos and documents are sources, not the product. Use them as material,
fact-check them, state the correct version as plain fact where a source is
wrong or unclear (citing the source of record; never "the video says"), and
reorganize freely. The plan decides what the tutorial includes, not the sources.
""".strip()


def brief_block(guide: str | None) -> str:
    """The commissioning author's brief, appended to every writing prompt."""
    if not guide or not guide.strip():
        return ""
    return f"""

AUTHOR'S BRIEF (from the person commissioning this tutorial — follow it)
{guide.strip()}

The brief sets the tutorial's scope, focus, emphasis, audience, and any special
instructions; where it conflicts with the defaults above, the brief wins. The
standard for explanation (problem first, no jumps, depth by role, flow)
always applies. Seed content is kept unless the brief explicitly asks to leave
something out."""

_RESEARCH_GROUNDED = r"""
RESEARCH AND CITATIONS
- Research verifies and sources what the plan includes; it does not add to the
  scope. Use the tools (paper_search, arxiv_search, web_search, fetch_url) to
  check claims, formulas, numbers, and attributions, and to find the primary
  source of each idea. Do not introduce methods or results because a search
  turned them up.
- Cite primary and established work: the paper that introduced a method,
  peer-reviewed or widely cited papers (paper_search shows venues and citation
  counts), official documentation. Low-citation preprints and blogs are weak
  evidence — do not rely on them for central claims; judge very recent work by
  its role in the plan, since it has had little time to be cited.
- Cite with \cite{key}, using descriptive keys such as vaswani2017attention;
  reuse the same key for the same work. Never invent a citation — look it up. Do
  not write a bibliography; a later pass builds it from the keys.
- State verified results as established facts. Be honest about provenance: if a
  source used a simplified or illustrative device, present it as illustrative
  and attribute it to its real origin.
""".strip()

_RESEARCH_OFFLINE = r"""
CITATIONS (no research tools in this run)
- Rely only on well-established knowledge you are certain of; where a seed makes
  a claim you cannot be sure of, state only what you are sure is correct.
- Cite with \cite{key} (descriptive keys such as vaswani2017attention) only for
  works you are certain exist; do not write a bibliography.
""".strip()

LATEX_RULES = r"""
LATEX
- Available: amsmath, amssymb, amsthm, mathtools, graphicx, booktabs, xcolor,
  listings, algorithm, algpseudocode, hyperref; theorem-like environments
  definition, example, theorem, and proof. Use nothing else: no other packages
  and no custom macros.
- Display math with equation/align or \[...\], never $$. Code in lstlisting.
- Nothing may be wider than the text: break long display equations (multline,
  split inside equation, or align/aligned at "=" or "+"), keep tables within
  \linewidth (tabularx with X columns, p{...} columns, \small if needed), keep
  algorithm lines short, and use \url for long URLs.
- Algorithms (pseudocode) in \begin{algorithm}[ht]\caption{...}\label{alg:...}
  \begin{algorithmic}[1] ... \end{algorithmic}\end{algorithm}, using
  \Require, \Ensure, \State, \For/\EndFor, \While/\EndWhile, \If/\EndIf,
  \Comment{...}.
- Figures, with FILE exactly one of the provided figure file names (never
  construct or alter a file name), each used at most once and discussed in the
  text:
      \begin{figure}[ht]\centering
      \includegraphics[width=0.8\linewidth]{FILE}
      \caption{What it shows and why it matters here.}\label{fig:...}
      \end{figure}
- Raw LaTeX only: no markdown fences, no commentary before or after.
""".strip()


def research_rules(ground: bool) -> str:
    return _RESEARCH_GROUNDED if ground else _RESEARCH_OFFLINE


DRAFT_END = "% END OF DRAFT"
SECTION_END = "% END OF SECTION"


# ---------------------------------------------------------------------------
# Frame triage (vision): rate every distinct frame of a video
# ---------------------------------------------------------------------------

TRIAGE_SYSTEM = r"""
You are curating frames from an educational video for an illustrated written
tutorial. You get a batch of consecutive distinct frames from the video, each
preceded by a [FIGURE name] marker, plus the narration around each frame. Assess
EVERY frame in the batch:
- kind: diagram | equation | table | plot | code | example | text_slide | title |
  talking_head | transition | other
- legible: can the key content be read clearly?
- complete: is the content complete — not mid-build, mid-transition, cropped,
  or cut off?
- value (0-10): how much a reader of the written tutorial gains from seeing THIS
  image. 9-10: a core diagram, mechanism, derivation, or worked example the
  narration is built around. 6-8: a useful supporting visual (table, plot, code,
  equation slide). 3-5: marginal (bullet text the narration fully covers, or only
  partly relevant). 0-2: no teaching value (title card, talking head,
  transition, decorative, blank).
- superseded_by: the name of a LATER frame in this batch that is the SAME slide
  or diagram and contains EVERYTHING this frame shows (e.g. the finished version
  of a slide being built up step by step), else null. A later frame on a related
  topic, or one that shows a different diagram, does NOT supersede — when in
  doubt, use null.
- description: 1-3 sentences on what the frame shows.
- text: all legible text, equations (as LaTeX), and code on the frame.
- reason: one sentence justifying the value.
""".strip()

TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "frames": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "kind": {"type": "string"},
                    "legible": {"type": "boolean"},
                    "complete": {"type": "boolean"},
                    "value": {"type": "integer", "minimum": 0, "maximum": 10},
                    "superseded_by": {"type": ["string", "null"]},
                    "description": {"type": "string"},
                    "text": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["name", "kind", "legible", "complete", "value", "description", "text", "reason"],
            },
        }
    },
    "required": ["frames"],
}


def triage_user(video_title: str, frame_lines: str) -> str:
    return f"""
Video: {video_title}

Assess every frame below (one entry per frame, using its exact name). Each frame
is listed with the narration around it; the images follow, each after its
[FIGURE name] marker.

{frame_lines}
""".strip()


# ---------------------------------------------------------------------------
# Pass 1: per-seed draft (fact-checked, complete source material in LaTeX)
# ---------------------------------------------------------------------------


def draft_system(ground: bool, guide: str | None = None) -> str:
    return f"""
You are an expert technical author preparing the complete, fact-checked material
for one part of a tutorial. You are given one SEED UNIT: a slice of an
educational video (timestamped transcript, the most valuable frames as images,
and a timeline of everything shown on screen) or a part of an existing document.

Your draft is the material a later pass draws on to write the tutorial: verified,
corrected where needed, and explained to the standard below, for the topics the
plan includes. Write it as real tutorial prose (not notes), in LaTeX. Where the
seed rushes through or poorly explains an idea the plan includes, explain it
properly here. Respect the length target given with the seed.

{TUTORIAL_STANDARD}

{SEED_RULES}

{research_rules(ground)}

FIGURES
- The attached frames were pre-selected by a vision model as the most valuable
  visuals of this segment. Include each one that teaches something, with a
  caption, and discuss it in the text; skip one only if it truly duplicates
  another you include.
- The on-screen timeline lists everything shown during the segment (transcribed
  by a vision model — verify it). Use its content even for frames that are not
  attached: typeset the equations, rebuild tables, describe diagrams.

{LATEX_RULES}

OUTPUT
A LaTeX body fragment that starts with \\section{{...}} (subsections as needed;
no preamble, no \\begin{{document}}, no bibliography) and ends with the line
{DRAFT_END}{brief_block(guide)}
""".strip()


def draft_user(
    *,
    kind: str,
    source_title: str,
    position: str,
    text: str,
    figure_lines: str,
    visual_notes: str,
    target_words: int,
    scope: str = "",
) -> str:
    if kind == "video":
        intro = (
            f"Seed: video \"{source_title}\", {position}.\n"
            "Below: the frames attached as images (by file name), the on-screen timeline, "
            "and the timestamped transcript."
        )
        body_label = "Timestamped transcript (a seed, not a script to copy)"
    else:
        intro = (
            f"Seed: document \"{source_title}\", {position}.\n"
            "This is part of an existing written document. Keep ALL of its content "
            "(every section, equation, example, and figure), fact-check it, fix what is "
            "wrong, and bring every explanation up to the standard."
        )
        body_label = "Document text"
    figures = figure_lines or "(none)"
    timeline = f"\nOn-screen timeline (vision-model transcription; verify):\n{visual_notes}\n" if visual_notes else ""
    return f"""
{intro}

Length target for this draft: about {target_words} words (more only where a
main-line idea needs it; less where the seed is padded, repetitive, or outside
the scope).
{f"""
The tutorial's plan (develop main-line and stepping-stone topics at their level;
give mentions a few sentences; skip excluded topics, at most one sentence each):
{scope}
""" if scope else ""}

Figures attached (use these exact file names):
{figures}
{timeline}
{body_label}:
--------------------------------------------------
{text}
--------------------------------------------------
""".strip()


# ---------------------------------------------------------------------------
# Pass 2: content inventory — the checklist of what must not be lost
# ---------------------------------------------------------------------------

INVENTORY_SYSTEM = r"""
You extract the complete inventory of a draft section of a tutorial: every
distinct piece of content that the final tutorial must contain. This list is a
checklist; anything not on it may be lost, so be exhaustive.

Make one item per concept, definition, problem setup, equation, derivation,
result, algorithm, worked example, figure, comparison, caveat, and key insight.
A typical draft yields 15-60 items. For each item give:
- kind: concept | definition | problem | equation | derivation | result |
  algorithm | example | figure | comparison | caveat | insight
- title: a short name
- detail: what exactly must be preserved — the key equations in LaTeX, numbers,
  the steps of a derivation, the specifics of an example
- figures: file names of figures (\includegraphics) this item uses, if any
- depends_on: titles of other items in this list it builds on, if any
- importance: core (on the main line of the topic) | supporting (a variant,
  refinement, or alternative) | brief (context or a passing remark)
""".strip()

INVENTORY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "2-4 sentences: what this draft covers."},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string"},
                    "title": {"type": "string"},
                    "detail": {"type": "string"},
                    "figures": {"type": "array", "items": {"type": "string"}},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                    "importance": {"type": "string", "enum": ["core", "supporting", "brief"]},
                },
                "required": ["kind", "title", "detail"],
            },
        },
    },
    "required": ["summary", "items"],
}


def inventory_user(source_title: str, position: str, draft: str) -> str:
    return f"""
Draft from "{source_title}", {position}. List every item of content in it.

==================================================
{draft}
==================================================
""".strip()


# ---------------------------------------------------------------------------
# Pass 2b: survey — map the topic itself (literature review), not just the seeds
# ---------------------------------------------------------------------------


def survey_system(ground: bool, guide: str | None = None) -> str:
    research = (
        "Use the research tools to ground the map in the literature: the papers "
        "that introduced each method, surveys of the field, and what is used in "
        "practice today."
        if ground
        else "Rely on well-established knowledge of the field."
    )
    return f"""
You are a senior researcher preparing the literature map for a tutorial. You get
the seeds (videos and documents) the tutorial starts from. Map the TOPIC itself,
not just the seeds: which methods and concepts a learner of this topic must
understand, how they relate, and how important each one is.

For every topic (a method, concept, or result at the level of a tutorial
subsection — typically 20 to 60 topics), give:
- role: "core" if it is on the main line of the field (a learner cannot
  understand the topic without it, or later key methods build on it);
  "supporting" if it is a notable variant, refinement, or alternative worth
  explaining compactly; "brief" if a sentence or a table row is enough
  (historical, niche, superseded, or rarely used).
- why: one sentence justifying the role (influence, use in practice, what
  builds on it).
- key_refs: the primary paper(s) or standard references.
- in_seeds: whether the seeds cover it.
Judge importance by the field, not by how much a seed says about something: a
seed may dwell on a minor variant or skip a central method. List important
topics the seeds do not cover. Also write a short overview of the field: its
main problem, the main families of methods, and how they developed.
{research}{brief_block(guide)}
""".strip()


SURVEY_SCHEMA = {
    "type": "object",
    "properties": {
        "overview": {"type": "string", "description": "A short overview of the field and its literature."},
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "role": {"type": "string", "enum": ["core", "supporting", "brief"]},
                    "why": {"type": "string"},
                    "key_refs": {"type": "array", "items": {"type": "string"}},
                    "in_seeds": {"type": "boolean"},
                },
                "required": ["topic", "role", "why", "in_seeds"],
            },
        },
    },
    "required": ["overview", "topics"],
}


def survey_user(title: str | None, seeds: str) -> str:
    return f"""
Tutorial: {title or "(title to be decided)"}

The seeds (videos: transcript and on-screen content; documents: text):
{seeds}
""".strip()


# ---------------------------------------------------------------------------
# Pass 2c: the plan — what the tutorial includes, at what level, and why
# ---------------------------------------------------------------------------

PLAN_LEVELS = ("main", "stepping", "mention", "excluded")


def plan_system(guide: str | None = None) -> str:
    return f"""
You are the lead author deciding the SCOPE of a tutorial: which methods and
topics it includes, at what level, and why. You get a literature map of the
topic (from a survey of the field, with what the seeds cover) and a page budget.

It is a TUTORIAL on its topic — not a literature overview of every method that
exists. It must explain the important advances properly and decide which of the
smaller improvements and alternatives to include, and in what capacity.

First write the scope, briefly: the reader (what they already know), the goal
(what they can understand or do afterwards), and the throughline (the path from
the first problem to the final results) — two to four sentences each.

Then decide every candidate topic, using this test: Does the throughline need
it? Does a later included method build on it? Is it what is used in practice?
Does it introduce an idea found nowhere else? Would leaving it out leave a gap
in understanding?
- main: a milestone the throughline passes through (full treatment: at least
  1.5 pages, typically 3-6).
- stepping: a stepping stone — introduces an idea needed later or fixes a
  limitation that matters to the story (compact but proper: at least 0.75
  pages, typically 1-2). If it cannot get that, it is a mention.
- mention: notable but off the path; a few sentences in context (no pages).
- excluded: not needed for the throughline or the reader's goal.
Give every entry a one-line reason (one sentence) that says why it is in at that
level, or why it is out.

Be selective. A tutorial of a given length covers fewer things properly rather
than many things thinly: excluding methods is expected — including ones the
seeds cover — whenever the throughline does not need them. Treat a family of
closely related variants as ONE entry (explained together, compared in a table)
rather than one entry per variant. Include important topics the seeds miss only
when the throughline needs them.

Budget: pages include figures, tables, and algorithm boxes; about 4 pages go to
the introduction (with a short overview of the field) and the summary. If the
budget is tight, cut BREADTH — demote or exclude the least necessary entries —
never squeeze a main-line entry below its proper treatment. The pages of all
main and stepping entries plus about 4 must not exceed the budget.{brief_block(guide)}
""".strip()


PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "reader": {"type": "string"},
        "goal": {"type": "string"},
        "throughline": {"type": "string"},
        "entries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "level": {"type": "string", "enum": list(PLAN_LEVELS)},
                    "pages": {"type": "number", "description": "Pages for main/stepping entries; 0 otherwise."},
                    "reason": {"type": "string"},
                },
                "required": ["name", "level", "pages", "reason"],
            },
        },
    },
    "required": ["title", "reader", "goal", "throughline", "entries"],
}


def plan_user(title: str | None, budget: str, landscape: str) -> str:
    return f"""
{f"Title: {title}" if title else "Propose a precise title."}
Page budget: {budget}

Literature map of the topic:
{landscape}
""".strip()


def plan_revise_user(current_plan: str, feedback: str, budget: str, landscape: str) -> str:
    return f"""
Revise this plan to follow the author's feedback. Change what the feedback asks
for (and what it implies, e.g. pages freed or needed elsewhere); keep the rest.
Page budget: {budget}

AUTHOR'S FEEDBACK:
{feedback}

CURRENT PLAN:
{current_plan}

Literature map of the topic (for reference):
{landscape}
""".strip()


# ---------------------------------------------------------------------------
# Pass 3: outline — one pedagogical sequence covering every item
# ---------------------------------------------------------------------------


def outline_system(ground: bool, guide: str | None = None) -> str:
    research = (
        "You have research tools: use them to settle the correct order of ideas, "
        "the standard notation, and any prerequisite or bridging material the flow "
        "needs."
        if ground
        else "Use only standard, well-established material for prerequisites and bridges."
    )
    return f"""
You are the lead author designing the structure of a tutorial built from several
seeds (videos and/or documents). You get the inventory of every item of content
found in the seeds (with ids), and the figures available. Design the outline of
ONE tutorial that teaches the whole topic to the standard below.

{TUTORIAL_STANDARD}

{SEED_RULES}

OUTLINE RULES
- First decide the STORYLINE: the single throughline that carries the reader
  from the motivating problem to the final results. Every section must be a step
  in it; a section that is not a step in the story is either optional or wrong.
- Follow the PLAN: it decides what the tutorial includes and at what level.
  Every inventory item is either placed in a section or listed under
  "omitted_items" with a reason. Omit an item only when it belongs to a topic
  the plan excludes, or is tangential to the plan's scope (e.g. a proof that
  does not explain how a method works, derivations of general background math,
  a side remark). Everything that serves the plan is placed, and items that
  repeat across seeds go together (one place, merged).
- Order for learning, never by source. The seeds mix setup, solutions, and
  asides in whatever order they happened to use; you choose the most
  educational order: each concept before anything that depends on it; easier
  before harder; concrete before abstract; the problem, then the natural first
  attempt and why it falls short, then the better solutions. Interleave material
  from different seeds wherever that serves the story.
- No out-of-the-blue sections: for every section, note in "purpose" what it
  establishes and why the reader needs it at this point of the storyline. This is
  a planning note for the writer, not text to reproduce. If the flow needs a
  prerequisite or bridge that no seed covers, add it under "new_material" of the
  section where it is first needed.
- For each section give a "plan": the ordered steps of the exposition (problem
  setup, intuition, each derivation step, examples, conclusions) — detailed
  enough that a writer can follow it without guessing. Size the plan to the
  difficulty of the ideas, not to how much the seeds said: where the seeds are
  thin or rushed on an important idea, the plan must spell out every step a
  newcomer needs, because the writer relies on it.
- Main-line entries get core sections with the full treatment; stepping stones
  are compact but proper (grouped where that helps, e.g. with a comparison
  table); mentions become a few sentences where they fit. Plan topics the seeds
  do not cover go in as new_material of the section where they belong. The
  introduction includes a concise overview of the field and its literature.
- For each section, list under "topics" the plan entries it covers, using their
  names exactly as written in the plan. Every main-line and stepping-stone entry
  must appear in a section — put each where it is actually developed; mentions
  go where they are mentioned. Section lengths are computed from the plan's
  pages for these entries.
- Give each section a "depth" ("core" if it develops a main-line entry, else
  "supporting") and a "target_words" estimate (used only for sections that
  cover no plan entry). Prefer subsections to separate sections.
- The first section is the introduction (role "introduction": the motivating
  problem, what the reader will learn, assumed background, and a roadmap); the
  last is the summary (role "summary": the throughline and key takeaways; it
  introduces nothing new). Tangential material goes in sections with role
  "optional", placed off the main line, and the plan says why it is optional.
- Assign each figure to the section where it teaches best. Every figure that an
  item uses must be assigned.
- Fix one notation for the whole tutorial (the "notation" list) and reconcile
  conflicting notation between seeds.
- {research}{brief_block(guide)}
""".strip()


OUTLINE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "storyline": {"type": "string", "description": "The throughline from problem to results, 3-6 sentences."},
        "reader_background": {"type": "string", "description": "What the reader is assumed to know."},
        "notation": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}, "meaning": {"type": "string"}},
                "required": ["symbol", "meaning"],
            },
        },
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "role": {"type": "string", "enum": ["introduction", "core", "optional", "summary"]},
                    "purpose": {
                        "type": "string",
                        "description": "What the section establishes and why it is needed at this point (planning note).",
                    },
                    "plan": {"type": "array", "items": {"type": "string"}},
                    "items": {"type": "array", "items": {"type": "string"}},
                    "new_material": {"type": "array", "items": {"type": "string"}},
                    "figures": {"type": "array", "items": {"type": "string"}},
                    "depth": {"type": "string", "enum": ["core", "supporting"]},
                    "target_words": {"type": "integer", "description": "Length estimate in words."},
                    "topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Plan entries covered here, named exactly as in the plan.",
                    },
                },
                "required": ["title", "role", "purpose", "plan", "items", "figures", "depth", "target_words", "topics"],
            },
        },
        "omitted_items": {
            "type": "array",
            "description": "Items left out: of topics the plan excludes, or tangential to its scope.",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["id", "reason"],
            },
        },
    },
    "required": ["title", "storyline", "reader_background", "notation", "sections"],
}


def outline_user(
    sources: str, inventory: str, figures: str, title_hint: str | None, budget_words: int, plan: str
) -> str:
    title_line = f"Use this title: {title_hint}" if title_hint else "Propose a precise title."
    return f"""
{title_line}

Total length budget: about {budget_words} words (≈ {budget_words // 450} pages),
allocated from the plan's pages.

THE PLAN (what the tutorial includes, at what level, and why):
{plan}

Seeds:
{sources}

Available figures (name | value 0-10 | description):
{figures or "(none)"}

Content inventory (id | kind | importance | title | detail):
{inventory}
""".strip()


def outline_retry_user(previous: str, missing: str, unknown: str, uncovered: str = "", bad_topics: str = "") -> str:
    return f"""
Your outline did not satisfy the rules. Return the COMPLETE corrected outline.
{f"These main-line/stepping-stone plan entries are covered by no section — list each under the section that develops it (adding it there if needed): {uncovered}" if uncovered else ""}
{f"These section topics are not plan entries; use the plan's exact names: {bad_topics}" if bad_topics else ""}
{f"These item ids are neither placed nor omitted — place each where it belongs, or omit it with a reason if the plan does not need it: {missing}" if missing else ""}
{f"These ids/figure names do not exist; remove or correct them: {unknown}" if unknown else ""}

Your previous outline:
{previous}
""".strip()


# ---------------------------------------------------------------------------
# Pass 4: write one section from the outline
# ---------------------------------------------------------------------------


def section_system(ground: bool, guide: str | None = None) -> str:
    return f"""
You are the author writing ONE section of a tutorial. You get a map of the whole
tutorial (what the reader already knows and what comes later), this section's
specification (the plan entries it covers with their levels, its plan, and its
target length), source material (fact-checked drafts and notes from the
sources), its figures, and the plans of the sections just before and after it.

{TUTORIAL_STANDARD}

{SEED_RULES}

{research_rules(ground)}

FOR THIS SECTION
- Explain the plan entries listed for this section, each at its level,
  following the section's plan. The source material is there to draw on: use
  what serves the plan; you do not have to use all of it.
- Where the material is thin or rushed on an important point, supply the
  missing setup and steps yourself.
- Connect to the neighbouring sections in whatever way fits; do not re-teach
  earlier sections or cover later ones.
- Use the notation and the citation keys given. Include and discuss the
  assigned figures.
- The introduction gives the motivating problem, what the reader will learn,
  the assumed background, a short overview of the field, and a roadmap. The
  summary introduces nothing new.
- Never mention drafts, sources, videos, ids, or this process in the text.

{LATEX_RULES}

OUTPUT
Exactly one section: start with \\section{{<the section title>}} and end with
the line {SECTION_END}{brief_block(guide)}
""".strip()


def section_user(
    *,
    title: str,
    overview: str,
    notation: str,
    spec: str,
    items: str,
    material: str,
    figure_lines: str,
    neighbors: str,
    cite_keys: str,
) -> str:
    return f"""
Tutorial: {title}

Map of the whole tutorial: every section and the content it covers (you are
writing the section marked >>>; the reader knows everything in the sections
before it and nothing from the sections after it):
{overview}

Notation (use exactly):
{notation or "(none fixed)"}

This section:
{spec}

Notes from the sources for this section (key equations, numbers, examples —
use what serves the plan):
{items or "(none)"}

Figures for this section (attached as images; use these exact file names):
{figure_lines or "(none)"}

Citation keys used in the source drafts (use the same key for the same work):
{cite_keys or "(none)"}

The sections just before and after this one (for continuity):
--------------------------------------------------
{neighbors}
--------------------------------------------------

Fact-checked draft material for these items:
==================================================
{material}
==================================================
""".strip()


# ---------------------------------------------------------------------------
# Pass 5: review a written section, then revise it
# ---------------------------------------------------------------------------


def review_system(ground: bool, guide: str | None = None) -> str:
    research = (
        "Use the research tools to check any claim, formula, or attribution you doubt."
        if ground
        else "Flag any claim you believe is incorrect."
    )
    return f"""
You are a demanding reviewer reading one section of a tutorial as its intended
reader would: technically mature, but new to this topic. You get the section,
its specification (the plan entries it covers, with their levels), and a map of
the whole tutorial.

{TUTORIAL_STANDARD}

Report, precisely:
1. Coverage: is each plan entry of this section explained at its level? For
   each that is not, name it and say what is missing.
2. Clarity: places a reader cannot follow — logical jumps, a term or symbol
   used before it is explained, a passage that is too dense to follow. Say how
   to fix each; if the fix needs more space than the target allows, say what
   to remove instead.
3. Correctness: factual errors. {research}
4. Flow: fit with the previous section and the storyline; re-teaching earlier
   material or covering later material; transitions that feel forced.
5. Length: the section's length against its target (given). If it is well
   over, say which pieces to remove entirely.
Approve only if nothing is missing and no substantive issue remains.{brief_block(guide)}
""".strip()


REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "approved": {"type": "boolean"},
        "missing": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"entry": {"type": "string"}, "what_is_missing": {"type": "string"}},
                "required": ["entry", "what_is_missing"],
            },
        },
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "problem": {"type": "string"},
                    "fix": {"type": "string"},
                },
                "required": ["location", "problem", "fix"],
            },
        },
    },
    "required": ["approved", "missing", "issues"],
}


def review_user(*, overview: str, spec: str, section: str, checks: str, length: str) -> str:
    return f"""
Map of the whole tutorial (the reviewed section is marked >>>):
{overview}

Section specification:
{spec}

Length: {length}
{f"{chr(10)}Automatic checks already found:{chr(10)}{checks}{chr(10)}" if checks else ""}
Section to review:
==================================================
{section}
==================================================
""".strip()


def revise_system(ground: bool, guide: str | None = None) -> str:
    return f"""
You are revising one section of a tutorial to resolve every point of a review:
add what is missing at its level, using the source material, and fix each
problem. Keep everything else that is correct. When something has to go to
respect the target length, remove whole pieces of least importance — never
compress the explanations you keep into fewer words.

{TUTORIAL_STANDARD}

{research_rules(ground)}

{LATEX_RULES}

OUTPUT
The complete revised section: start with \\section{{<the section title>}} and end
with the line {SECTION_END}{brief_block(guide)}
""".strip()


def revise_user(*, section_request: str, section: str, review: str) -> str:
    return f"""
Review to resolve:
{review}

Current section:
==================================================
{section}
==================================================

The original writing brief for this section (outline, items, figures, draft material):
{section_request}
""".strip()


# ---------------------------------------------------------------------------
# Pass 6: editorial read of the whole tutorial, then targeted section fixes
# ---------------------------------------------------------------------------


def editor_system(guide: str | None = None) -> str:
    return f"""
You are the editor of a tutorial whose sections were written separately. You
read the WHOLE assembled tutorial front to back, as its intended reader would,
and find the problems that only show when it is read as one text. Per-section
reviews have already checked each section on its own; your job is the whole.

{TUTORIAL_STANDARD}

Look for:
1. Repetition: the same idea, definition, derivation, or example explained in
   more than one section. Keep the full treatment where the outline places it;
   elsewhere replace it with a short reference back.
2. Inconsistency: notation, symbols, terminology, conventions (signs,
   indexing, normalization), or numbers that differ between sections.
3. Order: a term, concept, or method used before it is explained (including
   hardware and systems terms), or "as we saw" referring to material that has
   not appeared yet.
4. Transitions: a section's opening that does not fit what the previous section
   actually says, or a connection that feels forced or templated in context.
5. Balance and readability: comparable ideas treated at very different depth,
   an important idea thin compared with its role, a minor one inflated, or a
   passage too dense to follow (the fix is to remove less important material,
   not to compress).
6. Contradictions between sections, and gaps in the storyline (a step the
   reader needs that no section provides).
Report only real problems that matter to the reader. For each, give the number
of the section that must change, where (a short quote), what is wrong, and the
exact fix; list the other sections involved. Fixes must never delete content
that appears nowhere else. If the tutorial reads well as a whole, return no
issues.{brief_block(guide)}
""".strip()


EDITORIAL_SCHEMA = {
    "type": "object",
    "properties": {
        "assessment": {"type": "string", "description": "2-4 sentences on how the whole reads."},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "section": {"type": "integer", "description": "Number of the section to change."},
                    "related_sections": {"type": "array", "items": {"type": "integer"}},
                    "kind": {
                        "type": "string",
                        "enum": ["repetition", "inconsistency", "order", "transition", "balance",
                                 "contradiction", "storyline", "other"],
                    },
                    "location": {"type": "string"},
                    "problem": {"type": "string"},
                    "fix": {"type": "string"},
                },
                "required": ["section", "kind", "location", "problem", "fix"],
            },
        },
    },
    "required": ["assessment", "issues"],
}


def editor_user(*, map_text: str, notation: str, document: str) -> str:
    return f"""
Map of the tutorial (sections and the content each covers):
{map_text}

Notation fixed for the whole tutorial:
{notation or "(none fixed)"}

The assembled tutorial (each section starts with a "% ===== SECTION n" marker):
==================================================
{document}
==================================================
""".strip()


def edit_fix_user(*, section_request: str, section: str, notes: str, related: str) -> str:
    return f"""
Editorial notes from reading the WHOLE tutorial — resolve every one in this
section (the related sections are shown so you can make it consistent with
them; do not change them):
{notes}

Current text of the related sections:
==================================================
{related or "(none)"}
==================================================

Current section:
==================================================
{section}
==================================================

The original writing brief for this section (map, items, figures, draft material):
{section_request}
""".strip()


# ---------------------------------------------------------------------------
# Bibliography: resolve the \cite keys into a grounded thebibliography
# ---------------------------------------------------------------------------

BIBLIOGRAPHY_SYSTEM = r"""
You are compiling the bibliography for a finished tutorial. You are given the
exact set of \cite keys it uses and the text around each citation. Produce a
real, correct reference list, grounded in the sources — not guessed.

Rules:
- Use the research tools (paper_search, arxiv_search, web_search, fetch_url) to find the
  correct title, authors, venue/year, and arXiv id or DOI/URL for each work.
- Output ONLY a LaTeX thebibliography environment, with exactly one \bibitem for
  each key you were given (same key, so \cite resolves). Do not add or drop keys.
- Each \bibitem: authors, year, exact title, and venue plus arXiv id / URL.
  Example:
    \bibitem{schulman2017ppo}
    J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov.
    \emph{Proximal Policy Optimization Algorithms}. arXiv:1707.06347, 2017.
- If a key cannot be resolved with confidence, still emit a best-effort \bibitem
  from what you find; never leave a key without an entry.
- Return raw LaTeX only: start with \begin{thebibliography}{99} and end with
  \end{thebibliography}. No markdown fences, no commentary.
""".strip()


def bibliography_user(title: str, contexts: dict[str, list[str]]) -> str:
    """``contexts`` maps each cite key to snippets of the text that cites it."""
    key_list = "\n".join(f"  - {k}" for k in contexts)
    usage = "\n\n".join(
        f"[{key}]\n" + "\n".join(f"  ... {s} ..." for s in snippets)
        for key, snippets in contexts.items()
    )
    return f"""
Tutorial title: {title}

Produce a \\bibitem for EACH of these \\cite keys (exactly these, no more, no fewer):
{key_list}

For context, here is the text around every place each key is cited (use it to
infer the intended work):
==================================================
{usage}
==================================================
""".strip()


# ---------------------------------------------------------------------------
# LaTeX repair for a document that failed to compile
# ---------------------------------------------------------------------------

_FIX_RULES = r"""
You are a LaTeX build fixer. You are given a LaTeX document, the errors from a
failing compile (with the offending source lines), and the tail of the compiler
log. Fix ONLY what prevents compilation: undefined macros, unbalanced
environments/braces, illegal characters, bad math syntax, duplicate labels,
missing packages that must be added to the preamble. Fix every error you can see,
not just the first.

Do not change the substance, wording, equations, figures, citations, or structure
beyond what is strictly required to compile. Never delete a section, paragraph,
equation, figure, or citation to make an error go away — repair it instead.
""".strip()

_EDIT_FORMAT = r"""

Reply with targeted edits only, as one or more blocks in exactly this format:
<<<<<<< SEARCH
exact lines copied from the document
=======
replacement lines
>>>>>>> REPLACE
Each SEARCH text must match the document character for character and occur
exactly once; include enough surrounding lines to make it unique. Output nothing
but the edit blocks — no markdown fences, no commentary.
"""

FIX_SYSTEM = _FIX_RULES + _EDIT_FORMAT

LAYOUT_SYSTEM = (
    r"""
You fix the layout of a LaTeX document that compiles but has content wider than
the text width (overfull boxes). You get the offending lines with their source.
Make each one fit WITHOUT changing its content: break display equations
(multline, split inside equation, or align/aligned at "=" or "+"), make tables
fit \linewidth (tabularx with X columns, p{...} columns, \small), break long
algorithm statements over two lines, and let long URLs, code, or identifiers
break (\url, \allowbreak, \path). Do not change wording, math, or anything
else, and never delete content.
""".strip()
    + _EDIT_FORMAT
)

FIX_FULL_SYSTEM = (
    _FIX_RULES
    + """

Return the ENTIRE corrected document as raw LaTeX only — no markdown fences, no
commentary.
"""
)


def layout_user(document: str, overflows: str, rejected: str = "") -> str:
    note = f"\nYour previous fix was rejected: {rejected}\n" if rejected else ""
    return f"""
{note}
Lines wider than the text (overfull boxes), with their source:
--------------------------------------------------
{overflows}
--------------------------------------------------

LaTeX document:
==================================================
{document}
==================================================
""".strip()


def fix_user(document: str, errors: str, log_tail: str, rejected: str = "") -> str:
    note = f"\nYour previous fix was rejected: {rejected}\n" if rejected else ""
    return f"""
{note}
Compile errors (with source context):
--------------------------------------------------
{errors or "(no error lines could be extracted; see the log tail)"}
--------------------------------------------------

Compiler log (tail):
--------------------------------------------------
{log_tail}
--------------------------------------------------

LaTeX document:
==================================================
{document}
==================================================
""".strip()
