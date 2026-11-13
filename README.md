# vitex

Build a complete, self-contained **LaTeX tutorial** (and PDF) from any number of
**videos** and **documents**, using **Claude** (over Amazon Bedrock by default).

The sources are **seeds**, not the product: everything of substance in every
video and document is carried into one tutorial — fact-checked against the
literature, corrected where wrong, and expanded — so the result is more
complete, more detailed, and more correct than any single source. Typical uses:

- turn one or more lectures into a written tutorial;
- improve an existing tutorial (`--doc`) with new lectures;
- rewrite a poorly written document into a rigorous one (documents only);
- steer any of these with an author's brief (`--guide`).

## What "tutorial" means here

Every writing pass is held to one standard (`TUTORIAL_STANDARD` in
[prompts.py](src/vitex/llm/prompts.py)): a tutorial, **not a summary, survey, or
catalogue of methods**. A reader with general technical maturity (undergraduate
linear algebra, calculus, probability, programming) but no knowledge of the
topic must understand all of it, front to back, without other material:

- **Problem first** — every idea is introduced through the problem it solves,
  set up fully (setting, givens, goal, assumptions, why the obvious fails).
- **Intuition, then formalism** — then every symbol tied back to the intuition.
- **No logical jumps** — every derivation step by step, each step named.
- **No assumed topic knowledge** — every term and symbol defined at first use.
- **Nothing out of the blue** — a method appears only once the reader needs it.
- **Flow** — one continuous, well-structured text: each part follows
  naturally from what came before. How a section opens, connects, and closes
  is decided by what fits it — its purpose and its neighbours; a motivating
  question, signpost, or recap is used where it genuinely helps, never as a
  template. One notation throughout.
- **Concrete** — worked examples with numbers; algorithms walked on small
  inputs, written as numbered pseudocode (`algorithm`/`algpseudocode`).
- **Depth follows difficulty, not the source** — if a video rushes through an
  idea or a document gives it one line, the tutorial still gives it the full
  treatment (setup, intuition, every step, an example). Reviews flag ideas that
  are explained more briefly than they need.
- **Build up** — easier before harder, concrete before abstract; the problem,
  the natural first attempt and why it fails, then the better solution. The
  outline fixes one storyline and orders material from all sources along it,
  regardless of the order the sources used.

## How it works

```
videos ──► audio ──► transcript (faster-whisper | Amazon Transcribe)
   └──► frames: one per distinct visual state ──► vision triage (rate every frame)
documents (.tex/.md/.txt/.pdf) ──► text + figures
                        │
        seed units (8-minute video chunks, document parts)
                        │
   draft        each unit → complete, fact-checked LaTeX (research tools)
   inventory    each draft → checklist of every content item (ids)
   outline      one storyline; every item placed (checked in code)
   write        each section, with a map of what every section covers (parallel)
   review       per section: coverage + rigor + depth + flow → revise (×2)
   edit         an editor reads the WHOLE tutorial: repetition, consistency,
                order, transitions, balance → fix flagged sections (×2)
   bibliography \cite keys → grounded thebibliography (batches of 12)
   compile      latexmk + edit-based repair that may not delete content
```

**A holistic view at both ends.** The outline is designed in one pass over the
inventory of all sources. Every section writer gets a map of the whole
tutorial — each section's purpose and the content it covers — so it knows
exactly what the reader already has and what comes later, plus the plans of its
neighbouring sections. After writing, an editor reads the whole assembled
tutorial and reports what only shows across sections (repetition, inconsistent
notation or terms, forward references, transitions that don't fit, uneven
depth, contradictions, storyline gaps); the affected sections are revised with
the editor's notes and the text of the related sections, and the editor reads
again (`--editorial-rounds`, default 2). Findings go to
`work/tutorial/editorial.json`.

Independent requests run in parallel (`--parallel`, default 6): triage
batches, drafts, inventories, sections (each written and reviewed on its own),
editorial fixes, and bibliography batches.

**Nothing is dropped.** The inventory is a checklist: the outline must place
every item, and anything it misses after retries is attached in code. Each
section is reviewed for coverage of its items and revised; revisions that cut
text, and LaTeX repairs that delete figures, citations, or sections, are
rejected. Every model output is validated (empty or truncated responses are
detected by required end markers) and retried; if it still fails the run
**stops** rather than skipping content, and re-running resumes from the cache.

### Frames: how images are chosen

1. Sample a frame every 2 s.
2. Split into **runs of a stable visual state** by per-pixel change between
   consecutive samples (a new slide, a new equation line, a new build step all
   start a new run). Keep the sharpest of the **last** frames of each run — a
   slide being built up is most complete just before it changes.
3. **Triage**: a vision model (`VITEX_TRIAGE_MODEL`) looks at every distinct
   frame, in order, with the narration around it, and records its kind,
   legibility, completeness, a 0–10 teaching value with a reason, the
   text/equations on it, and whether one of the next frames is the same slide
   more complete. Notes are saved in `work/<source>/triage.json`.
4. Each seed unit attaches its highest-value eligible frames (`--max-images`,
   `--min-frame-value`) and gets a timeline of **all** frames' content, so what
   was on screen reaches the tutorial even when the image isn't attached.

### Research

During drafting, outlining, writing, reviewing, and the bibliography, Claude
can call keyless tools — arXiv search, DuckDuckGo web search, and page fetching
— up to `--research-iterations` rounds per pass (default 16), to verify claims,
formulas, numbers, and attributions and to find primary sources, cited with
`\cite{...}`. `fetch_url` only reaches public http(s) hosts.

## Install

Requires Python 3.12+, `ffmpeg`/`ffprobe` for videos, and a TeX install with
`latexmk` (or `pdflatex`) for the PDF.

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[whisper,pdf,dev]"   # Bedrock + local Whisper + PDF documents
# add the `aws` extra for Amazon Transcribe
```

The package is installed in editable mode, so code changes take effect without
reinstalling. If you use `uv sync`, name the extras
(`uv sync --extra whisper --extra pdf --extra dev`) — a plain `uv sync` removes
faster-whisper and pypdf.

## Setup

Copy `.env.example` to `.env`. For Bedrock (the default) two settings matter:

```bash
# REQUIRED: the ~/.aws profile used for every AWS call. There is deliberately
# no fallback to AWS_PROFILE, env keys, or [default] — without it the run stops.
VITEX_AWS_PROFILE=my-bedrock-profile
AWS_REGION=us-west-2

# Models (Bedrock inference-profile ids your account can invoke)
VITEX_MODEL=us.anthropic.claude-opus-5-5          # all writing passes
VITEX_TRIAGE_MODEL=us.anthropic.claude-haiku-5-5  # frame rating (cheap; default: VITEX_MODEL)
VITEX_FIX_MODEL=us.anthropic.claude-sonnet-5-5    # LaTeX repair (default: VITEX_MODEL)
# VITEX_EFFORT=high                               # low…max; Opus 5.5's own default is medium
```

Check which account a profile resolves to with
`aws sts get-caller-identity --profile <profile>`; vitex only needs
`bedrock:InvokeModel` (plus S3 and Transcribe for `--transcriber aws`), so a
least-privilege profile is preferable to an admin one.

To use the Anthropic API instead: `VITEX_PROVIDER=anthropic`,
`ANTHROPIC_API_KEY=...`, and public model ids (e.g. `claude-opus-5-5`); no AWS
profile is needed then (unless you use Amazon Transcribe).

## Usage

Run from the repo root with the venv active.

```bash
# one video
vitex run "https://www.youtube.com/watch?v=XXXX"

# several videos merged into one tutorial
vitex run "https://youtu.be/AAAA" "https://youtu.be/BBBB" lecture3.mp4 --title "Efficient Attention"

# improve an existing tutorial with a video, steered by a brief in a file
vitex run "https://www.youtube.com/watch?v=mg-iU-WxiNs" \
  --doc output/rl_tutorial/rl_tutorial.tex \
  --title "A Comprehensive Tutorial on Reinforcement Learning: From Dynamic Programming to Language-Model Reasoning" \
  --name modern_rl \
  --guide-file briefs/modern_rl.md

# documents only: rewrite an existing tutorial that isn't well written
vitex run --doc draft.tex --doc notes.md --guide "Rewrite this into a rigorous, much more detailed tutorial."
```

- **`--title`** is the title printed in the document; without it, the outline
  pass proposes one.
- **`--name`** names the output: `output/<name>/<name>.{tex,pdf}` plus
  `figures/`. Without it, the title made filesystem-safe (punctuation removed,
  spaces → `_`).
- **`--guide` / `--guide-file`** is an **author's brief** added to every writing
  pass (drafts, outline, sections, reviews, revisions): scope, focus, emphasis,
  audience, which parts to keep and which to rework, special instructions. It
  overrides the defaults where they conflict; the explanation standard always
  applies. Seed content is kept unless the brief explicitly asks to leave
  something out — such items are listed in the outline and the log. Example
  briefs live in [briefs/](briefs/).
- **`--doc`** accepts `.tex` (with `\input`/`\include` resolved and figures
  carried over), `.md` (images carried over), `.txt`, and `.pdf` (text only).
  vitex always writes a whole new tutorial: a brief can ask to keep parts close
  to the original, but they are regenerated, not copied.

### Results and what to inspect

| Path | Contents |
|---|---|
| `output/<name>/<name>.tex`, `.pdf`, `figures/` | the tutorial |
| `work/tutorial/coverage.json` | per section: assigned items and figures, every review round, whether it was approved |
| `work/tutorial/editorial.json` | each whole-tutorial editorial round: assessment, cross-section issues, which sections were revised |
| `work/tutorial/outline.json` | the storyline, sections, plans, and item placement (and any items the brief excluded) |
| `work/drafts/<unit>.tex`, `<unit>.items.json` | the fact-checked draft and content inventory of each seed unit |
| `work/<source>/triage.json` | the rating and reason for every distinct video frame |
| `work/<source>/transcript.json`, `frames/` | transcript and distinct frames of each video |

### Resuming

Every expensive stage caches under `work/`. Each artifact has a `<name>.key`
sidecar fingerprinting its inputs (settings, model, prompts, brief, upstream
content), so re-running the same command picks up where it left off, and
changing a setting regenerates exactly what it affects. `--force` recomputes
everything.

## Configuration

Precedence: built-in defaults → `.env`/environment → CLI flags.

| Flag | Env | Meaning |
|---|---|---|
| `sources…` | — | video URLs / files (any number, or none with `--doc`) |
| `--doc PATH` | — | document seed (.tex/.md/.txt/.pdf), repeatable |
| `--title` | — | title printed in the document (default: proposed by the outline) |
| `--name` | — | output dir/file name (default: the title, filesystem-safe) |
| `--guide` / `--guide-file` | — | author's brief steering every writing pass |
| `--provider` | `VITEX_PROVIDER` | `bedrock` (default) or `anthropic` |
| — | `VITEX_AWS_PROFILE` | `~/.aws` profile for all AWS calls; **required** for Bedrock / Amazon Transcribe |
| `--aws-region` | `AWS_REGION` | region for Bedrock / Transcribe |
| `--model` | `VITEX_MODEL` | main model (default `us.anthropic.claude-opus-5-5`) |
| — | `VITEX_EFFORT` | reasoning effort of the main model, `low`…`max` (default `high`) |
| `--triage-model` | `VITEX_TRIAGE_MODEL` | vision model for frame rating (default: `--model`) |
| `--fix-model` | `VITEX_FIX_MODEL` | model for the LaTeX repair pass (default: `--model`) |
| `--ground` / `--no-ground` | `VITEX_GROUND` | research tools + citations, on by default |
| `--research-iterations` | — | research rounds per pass (default 16) |
| `--review-rounds` | — | review+revise rounds per section (default 2; 0 = off) |
| `--editorial-rounds` | — | whole-tutorial editing rounds (default 2; 0 = off) |
| `--parallel` | `VITEX_PARALLEL` | independent Claude requests run at once (default 6; 1 = sequential) |
| `--prompt-cache` / `--no-prompt-cache` | `VITEX_PROMPT_CACHE` | prompt caching, on by default |
| `--transcriber` | `VITEX_TRANSCRIBER` | `whisper` (default) or `aws` |
| `--whisper-model` | `VITEX_WHISPER_MODEL` | `tiny`…`large-v3` |
| — | `VITEX_TRANSCRIBE_BUCKET` | S3 bucket, required for `--transcriber aws` |
| `--frame-interval` | — | seconds between sampled frames (default 2) |
| `--max-images` | — | frames attached per video seed unit (default 16) |
| `--min-frame-value` | — | triage score needed to attach a frame (default 5) |
| `--chunk-minutes` | — | minutes of video per seed unit (default 8) |
| `--no-pdf` / `--force` | — | skip compilation / ignore cache |

**Cost** scales with input length and thoroughness: per seed unit a draft and an
inventory; per section a write plus up to 2 reviews and revisions, each with
research. Lower `--review-rounds`, `--research-iterations`, or `VITEX_EFFORT`
for cheaper runs; a Haiku triage model keeps frame rating cheap.

## How vitex talks to Claude

Requests are streamed (up to 128K output tokens; longer output is continued by
quoting what was written so far in a new user turn). The conversation is
append-only and structured answers use an unforced `submit` tool call, because
current models (e.g. Claude Opus 5.5) reject assistant prefill, forced
`tool_choice`, and edited history. The system prompt and the first (image-heavy)
turn are prompt-cached. Only transient errors (throttling, 5xx, network) are
retried; a rejected request fails immediately with the API's message.

## Package layout

```
src/vitex/
  cli.py, config.py, pipeline.py   entry point, settings, stage orchestration
  models.py                        Segment / Frame / FrameNote / Figure / SeedUnit
  media/        download, audio, frames (visual states), triage (vision), documents
  transcribe/   whisper_local, aws_transcribe
  llm/          Claude client (text, tools, structured), prompts, research tools
  reconstruct/  seeds, chapters (draft + bibliography), outline (inventory +
                outline), writer (sections + review), segmenter, align
  output/       latex (preamble, assembly, figures), build (compile + repair)
briefs/         example author's briefs (--guide-file)
tests/          unit tests, plus an end-to-end run with a scripted model
```

Run the tests with `pytest`; type-check with `mypy src`.

## Limitations

- **Long inputs:** no hard limit. Videos are drafted in 8-minute units, frames
  are capped at 20 distinct states per minute, and citations are resolved in
  batches; the outline sees the inventory of all sources in one request, which
  fits the 1M-token context for many hours of material. Cost and time grow
  linearly. Local Whisper runs at about 8× real time on CPU (`small` model);
  Amazon Transcribe accepts at most 4 hours of audio per file.
- **Models:** Bedrock model ids differ by account and region. Claude Fable
  models require 30-day data retention and are rejected on accounts whose
  retention setting doesn't meet it ("data retention mode 'default' is not
  available for this model").
- Handwriting and low-contrast slides may not read reliably; triage and the
  writers ground ambiguous content against the literature.
- Web search is best-effort (keyless DuckDuckGo) and may be rate-limited; arXiv
  is the reliable backbone. The `fetch_url` host check resolves DNS before the
  request, so DNS rebinding is not covered.
- PDF documents contribute text only (their figures are not extracted).
- faster-whisper's VAD needs onnxruntime, which has no wheel for Python 3.12 on
  glibc < 2.28 (e.g. Amazon Linux 2); vitex then uses ffmpeg silence detection.
