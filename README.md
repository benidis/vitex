# vitex

Build a self-contained **LaTeX tutorial** (and PDF) from any number of
**videos** and **documents**, using **Claude** (over Amazon Bedrock by default).

The sources are **seeds**, not the product. vitex first decides the tutorial's
**scope** — which methods and topics it covers, at what level, and why — then
writes it from the sources, fact-checked against the literature, corrected
where wrong, and explained properly. Typical uses:

- turn one or more lectures into a written tutorial;
- improve an existing tutorial (`--doc`) with new lectures;
- rewrite a poorly written document into a rigorous one (documents only);
- steer any of these with an author's brief (`--guide`).

## What "tutorial" means here

Every writing pass is held to one standard (`TUTORIAL_STANDARD` in
[prompts.py](src/vitex/llm/prompts.py)): a tutorial on its topic — **not a
summary, and not a literature overview of every method that exists**. A reader
with general technical maturity but no knowledge of the topic must understand
everything that is included, front to back, without other material:

- **Problem first**, then **intuition, then formalism**, with every symbol
  defined at first use and nothing introduced before the reader needs it.
- **Derivations scoped to the method** — the objective, gradient, update rule,
  and key identities a method relies on are derived step by step; general
  background math is assumed or stated in a line; theorems are stated with
  their intuition and cited, not proved (unless the proof shows how the method
  works).
- **Depth by level in the plan** — *main line*: full treatment; *stepping
  stones*: compact but proper (the problem they fix, the idea, the key
  equations and what changes, how they are used, the trade-off); *mentions*: a
  few sentences in context. Whatever is included is treated properly at its
  level; a source's length never inflates a minor point.
- **Extras where they teach** — algorithm boxes for main-line methods, cost
  analyses where cost drives the story, worked examples where a mechanism is
  not obvious, comparison tables for variants.
- **Flow and economy** — one continuous, well-structured text, built up from
  easier to harder; openings, transitions, and closings fit each section, never
  a template; no proofs that add nothing, tangents, repetition, or exhaustive
  lists.
- **Nothing wider than the page** — long equations are split, tables fit the
  text width, and a layout check fixes any overflow after compiling.

## How it works

```
videos ──► audio ──► transcript (faster-whisper | Amazon Transcribe)
   └──► frames: one per distinct visual state ──► vision triage (rate every frame)
documents (.tex/.md/.txt/.pdf) ──► text + figures
                        │
   survey     map the TOPIC with research: methods, importance, key papers,
              what the sources miss
   plan       SCOPE: every method/topic as main line / stepping stone /
              mention / excluded, with pages and a reason  ──► plan.md (editable)
   ─────────── --plan-only stops here ───────────
   draft      each source part → fact-checked material, scoped by the plan
   inventory  each draft → items of content
   outline    one storyline; items placed (or omitted with a reason);
              section lengths from the plan's pages
   write      each section, with a map of the whole tutorial   (parallel)
   review     coverage, rigor, depth, length, flow → revise (×2)
   edit       an editor reads the WHOLE tutorial → fix flagged sections (×2)
   bibliography, compile, layout check
```

**Scope is an explicit decision.** The plan tests every candidate against the
throughline: does it need it, does a later method build on it, is it used in
practice, does it bring an idea found nowhere else, would leaving it out leave
a gap? A page budget is spent by cutting **breadth** (demoting or excluding the
least necessary entries), never by squeezing main-line methods. The plan is
shown before anything is written, and you can change it.

**Lengths follow the plan.** Section targets come from the pages the plan gives
each entry (figures, tables, and algorithm boxes count against them). Writers
treat the target as a cap; a first version well over it is condensed before
review; reviews flag over- and under-length; the editor sees every length.

**Holistic view.** Every writer gets a map of the whole tutorial (each section's
purpose and the content it covers) and the plans of its neighbours. After
writing, an editor reads the whole assembled tutorial for repetition,
inconsistency, order, transitions, balance, contradictions, and storyline gaps,
and the affected sections are revised.

**Nothing is lost silently.** Every item of content is placed in a section or
omitted with a recorded reason (its topic is excluded by the plan, or it is
tangential). Every model output is validated and retried; if it still fails the
run **stops** rather than skipping content, and re-running resumes from the
cache. Revisions and LaTeX repairs that delete content are rejected.

### Frames

Frames are sampled every 2 s and reduced to one per distinct visual state (the
sharpest frame at the end of each run, i.e. the most complete version of a
slide being built up). A vision model (`VITEX_TRIAGE_MODEL`) rates every
distinct frame — kind, legibility, completeness, a 0–10 teaching value with a
reason, the text/equations on it, and whether a later frame supersedes it.
Each source part attaches its best frames and gets a timeline of all frames'
content.

### Research

The survey, plan, drafts, outline, sections, reviews, and bibliography can call
keyless research tools — arXiv search, DuckDuckGo web search, page fetching —
up to `--research-iterations` rounds per pass, to verify claims and find
primary sources, cited with `\cite{...}`. `fetch_url` only reaches public
http(s) hosts.

## Install

Requires Python 3.12+, `ffmpeg`/`ffprobe` for videos, and a TeX install with
`latexmk` (or `pdflatex`) for the PDF.

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[whisper,pdf,dev]"   # Bedrock + local Whisper + PDF documents
# add the `aws` extra for Amazon Transcribe
```

Run `vitex` with the venv active (it also provides `yt-dlp`). The package is
installed in editable mode. If you use `uv sync`, name the extras
(`uv sync --extra whisper --extra pdf --extra dev`) — a plain `uv sync` removes
faster-whisper and pypdf.

## Setup

Copy `.env.example` to `.env`. For Bedrock (the default):

```bash
# REQUIRED: the ~/.aws profile for every AWS call. There is deliberately no
# fallback to AWS_PROFILE, env keys, or [default] — without it the run stops.
VITEX_AWS_PROFILE=my-bedrock-profile
AWS_REGION=us-west-2

VITEX_MODEL=us.anthropic.claude-opus-5-5          # all writing passes
VITEX_TRIAGE_MODEL=us.anthropic.claude-haiku-5-5  # frame rating (cheap; default: VITEX_MODEL)
VITEX_FIX_MODEL=us.anthropic.claude-sonnet-5-5    # LaTeX repair/layout (default: VITEX_MODEL)
# VITEX_EFFORT=high                               # low…max; Opus 5.5's own default is medium
```

`aws sts get-caller-identity --profile <profile>` shows which account a profile
uses; vitex only needs `bedrock:InvokeModel` (plus S3 and Transcribe for
`--transcriber aws`). For the Anthropic API instead: `VITEX_PROVIDER=anthropic`,
`ANTHROPIC_API_KEY=...`, public model ids (e.g. `claude-opus-5-5`); no AWS
profile is needed then (unless you use Amazon Transcribe).

## Usage

Run from the repo root with the venv active. `--name` is the tutorial's
identity: its work dir `work/runs/<name>/` and its output `output/<name>/`.

### End to end

```bash
vitex run "https://www.youtube.com/watch?v=XXXX" --name attention --pages 60

# several videos and documents, steered by a brief
vitex run URL1 URL2 lecture3.mp4 --doc notes.md --doc old_tutorial/main.tex \
  --name efficient_attention --title "Efficient Attention" \
  --guide-file briefs/attention.md --pages 80
```

### In two steps: plan, then write

```bash
# 1. plan only (fast): survey + plan, then stop
vitex run URL --doc old.tex --name modern_rl --guide-file brief.md --pages 70 --plan-only
#    -> work/runs/modern_rl/plan.md

# 2. review/change the plan (see below), then write the tutorial from it
vitex run URL --doc old.tex --name modern_rl --guide-file brief.md --plan work/runs/modern_rl/plan.md
```

`--plan FILE` uses exactly that file — the saved `plan.md`, an edited copy kept
anywhere, or another tutorial's plan adapted for this one. Without `--plan`, a
run reuses the saved `plan.md` of its `--name` (it never silently replaces it;
`--replan` makes a fresh one, and vitex warns if the sources or brief changed
since the plan was made).

### Changing the plan

`plan.md` is plain markdown, written to be edited:

```markdown
# Plan: A Comprehensive Tutorial on Reinforcement Learning ...
Budget: 70 pages

## Scope
Reader: technically mature, new to RL; knows supervised learning.
Goal: understand and implement the RL methods used to train LLMs.
Throughline: MDPs -> value methods -> policy gradients -> PPO -> RLHF, GRPO, DPO.

## Main line — full treatment
- DQN | 4 | first deep value method; replay + target networks reused later
- PPO | 5 | default on-policy optimizer; the optimizer of RLHF; GRPO's template

## Stepping stones — compact but proper
- Double DQN | 1 | max-overestimation; the same fix is TD3's clipped double Q

## Mentions — a few sentences in context
- Noisy Nets | parameter-space exploration; not needed for the throughline

## Excluded
- C51 / QR-DQN | distributional RL is not used by any later method in scope
```

Two ways to change it:

1. **Edit the file**: move a line to another level, change pages, add or delete
   entries, rewrite the scope or throughline, change the budget. Lines are
   `- name | pages | reason` (main line, stepping stones) or `- name | reason`
   (mentions, excluded). The next run reads your version as is; vitex reports
   errors with line numbers and warns if the pages exceed the budget.
2. **Give feedback** and let the planner revise it:
   ```bash
   vitex run ... --name modern_rl --plan-only --plan-feedback "Drop offline RL. Give DPO a full section."
   # or --plan-feedback-file notes.md
   ```
   The revised plan replaces `plan.md` (the previous one is kept as
   `plan.v1.md`, `plan.v2.md`, …) and the changes are printed. Repeat until it
   is right, then run without `--plan-only`.

### Title, name, brief

- **`--title`** is printed in the document; without it the plan proposes one.
- **`--name`** names the run and the output; without it, the title (made
  filesystem-safe), else the first source's title.
- **`--guide` / `--guide-file`** is an **author's brief** added to the plan and
  every writing pass: scope, focus, emphasis, audience, special instructions.
  Example briefs live in [briefs/](briefs/).
- **`--pages`** sets the budget; without it the plan uses what the scope needs,
  up to a ceiling sized from the survey.

## Files

```
work/
  sources/<source>/     per video/document: download, audio, transcript, frames,
                        triage.json — shared by every tutorial using the source
  runs/<name>/          one tutorial:
    plan.md (+ plan.vN.md)   the scope decision (editable)
    survey.json              the literature map: topics, importance, key refs
    drafts/                  fact-checked material and inventories per source part
    outline.json             storyline, sections, item placement, omitted items + reasons
    sections/, editorial/    written sections, review and editorial caches
    coverage.json            per section: items, reviews, approval
    editorial.json           whole-tutorial editorial rounds and fixes
    run.log                  the full log of every run
output/<name>/<name>.tex, .pdf, figures/
```

Every artifact has a `<name>.key` sidecar fingerprinting its inputs, so a rerun
resumes where it left off and a changed setting, prompt, or plan regenerates
exactly what it affects. `--force` recomputes everything.

## Configuration

Precedence: built-in defaults → `.env`/environment → CLI flags.

| Flag | Env | Meaning |
|---|---|---|
| `sources…` | — | video URLs / files (any number, or none with `--doc`) |
| `--doc PATH` | — | document seed (.tex/.md/.txt/.pdf), repeatable |
| `--name` | — | the tutorial's identity: `work/runs/<name>/`, `output/<name>/` |
| `--title` | — | title printed in the document (default: from the plan) |
| `--guide` / `--guide-file` | — | author's brief for the plan and every writing pass |
| `--pages` | — | page budget (default: what the scope needs, up to a survey-sized ceiling) |
| `--plan-only` | — | stop after writing the plan |
| `--plan FILE` | — | write the tutorial from exactly this plan |
| `--plan-feedback` / `--plan-feedback-file` | — | revise the current plan with notes |
| `--replan` | — | make a fresh plan instead of reusing the saved one |
| `--provider` | `VITEX_PROVIDER` | `bedrock` (default) or `anthropic` |
| — | `VITEX_AWS_PROFILE` | `~/.aws` profile for all AWS calls; **required** for Bedrock / Transcribe |
| `--aws-region` | `AWS_REGION` | region for Bedrock / Transcribe |
| `--model` | `VITEX_MODEL` | main model (default `us.anthropic.claude-opus-5-5`) |
| — | `VITEX_EFFORT` | reasoning effort of the main model, `low`…`max` (default `high`) |
| `--triage-model` | `VITEX_TRIAGE_MODEL` | vision model for frame rating (default: `--model`) |
| `--fix-model` | `VITEX_FIX_MODEL` | model for LaTeX repair and layout fixes (default: `--model`) |
| `--ground` / `--no-ground` | `VITEX_GROUND` | research tools + citations, on by default |
| `--research-iterations` | — | research rounds per pass (default 16) |
| `--review-rounds` | — | review+revise rounds per section (default 2; 0 = off) |
| `--editorial-rounds` | — | whole-tutorial editing rounds (default 2; 0 = off) |
| `--parallel` | `VITEX_PARALLEL` | independent Claude requests at once (default 6) |
| `--prompt-cache` / `--no-prompt-cache` | `VITEX_PROMPT_CACHE` | prompt caching, on by default |
| `--transcriber` | `VITEX_TRANSCRIBER` | `whisper` (default) or `aws` |
| `--whisper-model` | `VITEX_WHISPER_MODEL` | `tiny`…`large-v3` |
| — | `VITEX_TRANSCRIBE_BUCKET` | S3 bucket, required for `--transcriber aws` |
| `--frame-interval` | — | seconds between sampled frames (default 2) |
| `--max-images` | — | frames attached per video part (default 16) |
| `--min-frame-value` | — | triage score needed to attach a frame (default 5) |
| `--chunk-minutes` | — | minutes of video per part (default 8) |
| `--no-pdf` / `--force` | — | skip compilation / ignore cache |

Every Claude request is logged with its stage, model, duration, tokens, and a
running cost estimate; a per-model usage summary ends every run.

## How vitex talks to Claude

Requests are streamed (up to 128K output tokens; longer output is continued by
quoting what was written so far in a new user turn). The conversation is
append-only and structured answers use an unforced `submit` tool call checked
against its schema, because current models (e.g. Claude Opus 5.5) reject
assistant prefill, forced `tool_choice`, and edited history. The system prompt
and the first (image-heavy) turn are prompt-cached. Only transient errors are
retried; independent requests run in parallel.

## Package layout

```
src/vitex/
  cli.py, config.py, pipeline.py   entry point, settings, stage orchestration
  models.py                        Segment / Frame / FrameNote / Figure / SeedUnit
  media/        download, audio, frames, triage (vision), documents
  transcribe/   whisper_local, aws_transcribe
  llm/          Claude client, prompts, research tools
  reconstruct/  seeds, survey, plan, chapters (drafts + bibliography),
                outline (inventory + outline), writer (sections, review, editor)
  output/       latex (preamble, assembly, figures), build (compile, repair, layout)
briefs/         example author's briefs (--guide-file)
tests/          unit tests, plus end-to-end runs with a scripted model
```

Run the tests with `pytest`; type-check with `mypy src`.

## Limitations

- **Long inputs:** no hard limit; cost and time grow with the sources and the
  page budget. Local Whisper runs at about 8× real time on CPU (`small`);
  Amazon Transcribe accepts at most 4 hours of audio per file.
- **Models:** Bedrock model ids differ by account and region. Claude Fable
  models require 30-day data retention and are rejected on accounts whose
  retention setting doesn't meet it.
- Handwriting and low-contrast slides may not read reliably; triage and the
  writers ground ambiguous content against the literature.
- Web search is best-effort (keyless DuckDuckGo) and may be rate-limited; arXiv
  is the reliable backbone. The `fetch_url` host check resolves DNS before the
  request, so DNS rebinding is not covered.
- PDF documents contribute text only (their figures are not extracted).
- faster-whisper's VAD needs onnxruntime, which has no wheel for Python 3.12 on
  glibc < 2.28 (e.g. Amazon Linux 2); vitex then uses ffmpeg silence detection.
