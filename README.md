# vitex

Build a complete, self-contained **LaTeX tutorial** from any number of
**videos** and **documents**, using **Claude** (over Amazon Bedrock by default).

The sources are **seeds**, not the product: everything of substance in every
video and document is carried into one tutorial — fact-checked against the
literature, corrected where wrong, and expanded — so the result is more
complete, more detailed, and more correct than any single source. Point it at an
existing tutorial (`--doc`) plus a few lectures and you get an improved,
merged version.

## What "tutorial" means here

This is the standard every writing pass is held to (see `TUTORIAL_STANDARD` in
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
- **Flow** — each section opens from the previous one and hands an open
  question to the next; one notation throughout.
- **Concrete** — worked examples with numbers; algorithms walked on small inputs.
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
        seed units (video time chunks, document parts)
                        │
   draft   each unit → complete, fact-checked LaTeX (research tools)
   inventory  each draft → checklist of every content item (ids)
   outline    one pedagogical order; every item placed (checked in code)
   write      section by section, from the outline + drafts (research tools)
   review     coverage + step-by-step rigor + flow → revise (×2)
   bibliography  \cite keys → grounded thebibliography
   compile    latexmk + edit-based repair that may not delete content
```

**Nothing is dropped.** The inventory is a checklist: the outline must place
every item, and anything it misses after retries is attached in code. Each
section is reviewed for coverage of its items and revised; revisions that cut
text, and LaTeX repairs that delete figures/citations/sections, are rejected.
Every model output is validated (empty or truncated responses are detected by
required end markers) and retried; if it still fails the run **stops** rather
than skipping content, and re-running resumes from the cache. Review outcomes
per section are written to `work/tutorial/coverage.json`.

### Frames: how images are chosen

1. Sample a frame every 2 s.
2. Split into **runs of a stable visual state** by per-pixel change between
   consecutive samples (a new slide, a new equation line, a new build step all
   start a new run). Keep the sharpest of the **last** frames of each run — a
   slide being built up is most complete just before it changes.
3. **Triage**: a vision model (set `VITEX_TRIAGE_MODEL` to a Haiku profile to
   keep it cheap) looks at every distinct frame, in order, with the narration
   around it, and records its kind, legibility, completeness, a 0–10 teaching
   value with a reason, the text/equations on it, and whether a later frame of
   the same slide supersedes it. Notes are saved in `work/<source>/triage.json`.
4. Each seed unit attaches its highest-value eligible frames (`--max-images`,
   `--min-frame-value`), and gets a timeline of **all** frames' content — so
   what was on screen reaches the tutorial even when the image isn't attached.

### Research

During drafting, outlining, writing, reviewing, and the bibliography, Claude
can call keyless tools — arXiv search, DuckDuckGo web search, and page fetching
— up to `--research-iterations` rounds per pass (default 16), to verify claims,
formulas, numbers and attributions and to find primary sources, cited with
`\cite{...}`. `fetch_url` only reaches public http(s) hosts.

## Install

Requires Python 3.12+, `ffmpeg`/`ffprobe` for videos, and a TeX install
(`latexmk` or `pdflatex`) for the PDF.

```bash
uv pip install -e ".[whisper,dev]"          # Bedrock + local Whisper
uv pip install -e ".[whisper,aws,pdf,dev]"  # + Amazon Transcribe, PDF documents
```

Copy `.env.example` to `.env` and adjust (model ids, region, transcriber).

## Usage

```bash
# one video
vitex run "https://www.youtube.com/watch?v=XXXX"

# several videos merged into one tutorial
vitex run "https://youtu.be/AAAA" "https://youtu.be/BBBB" lecture3.mp4 --title "Efficient Attention"

# improve an existing tutorial with new lectures (documents are seeds too)
vitex run "https://youtu.be/AAAA" --doc my_tutorial/main.tex --doc notes.md --doc paper.pdf

# documents only: rewrite an existing tutorial that isn't well written
vitex run --doc draft.tex --guide "Rewrite this into a rigorous, much more detailed tutorial."

# steer the whole tutorial with an author's brief (inline or from a file)
vitex run URL1 URL2 --doc notes.md \
  --guide "A tutorial on all the modern RL methods for LLMs. Focus on policy-gradient \
derivations and GRPO; assume the reader knows supervised learning."
vitex run URL1 --guide-file brief.txt --title "Modern RL: A Tutorial" --name modern_rl
```

`--title` is the title printed in the document; without it, the outline pass
proposes one. The output goes to `output/<name>/<name>.{tex,pdf}` plus
`figures/`, where `<name>` is `--name` if given, else the title made
filesystem-safe (punctuation removed, spaces → `_`).

`--guide` / `--guide-file` is an **author's brief** added to every writing pass
(drafts, outline, sections, reviews, revisions): scope, focus, emphasis,
audience, special instructions. It overrides the defaults where they conflict;
the explanation standard always applies. Seed content is kept unless the brief
explicitly asks to leave something out — such items are listed in the outline
and the log.

### Resuming

Every expensive stage caches under `work/` (per video in `work/<source-slug>/`,
drafts in `work/drafts/`, the cross-source passes in `work/tutorial/`). Each
artifact has a `<name>.key` sidecar fingerprinting its inputs (settings, model,
prompts, upstream content), so re-running the same command picks up where it
left off, and changing a setting regenerates exactly what it affects. `--force`
recomputes everything.

## Configuration

Precedence: built-in defaults → `.env`/environment → CLI flags.

| Flag | Env | Meaning |
|---|---|---|
| `sources…` | — | video URLs / files (any number) |
| `--doc PATH` | — | document seed (.tex/.md/.txt/.pdf), repeatable |
| `--title` | — | title printed in the document (default: proposed by the outline) |
| `--name` | — | output dir/file name (default: the title, filesystem-safe) |
| `--guide` / `--guide-file` | — | author's brief steering every writing pass |
| `--provider` | `VITEX_PROVIDER` | `bedrock` (default) or `anthropic` |
| `--model` | `VITEX_MODEL` | model id / Bedrock inference-profile id |
| `--triage-model` | `VITEX_TRIAGE_MODEL` | vision model for frame rating (default: `--model`) |
| — | `VITEX_EFFORT` | reasoning effort of the main model: `low`…`max` (default `high`; Opus 5.5's own default is `medium`) |
| `--fix-model` | `VITEX_FIX_MODEL` | model for the LaTeX repair pass |
| `--aws-region` | `AWS_REGION` | region for Bedrock / Transcribe |
| — | `VITEX_AWS_PROFILE` | `~/.aws` profile for all AWS calls; **required** for Bedrock / Amazon Transcribe (no fallback to other credentials) |
| `--ground` / `--no-ground` | `VITEX_GROUND` | research tools + citations, on by default |
| `--research-iterations` | — | research rounds per pass (default 16) |
| `--review-rounds` | — | review+revise rounds per section (default 2; 0 = off) |
| `--prompt-cache` / `--no-prompt-cache` | `VITEX_PROMPT_CACHE` | prompt caching, on by default |
| `--transcriber` | `VITEX_TRANSCRIBER` | `whisper` (default) or `aws` |
| `--whisper-model` | `VITEX_WHISPER_MODEL` | `tiny`…`large-v3` |
| `--frame-interval` | — | seconds between sampled frames (default 2) |
| `--max-images` | — | frames attached per video seed unit (default 16) |
| `--min-frame-value` | — | triage score needed to attach a frame (default 5) |
| `--chunk-minutes` | — | minutes of video per seed unit (default 8) |
| `--no-pdf` / `--force` | — | skip compilation / ignore cache |

Cost scales with thoroughness: per seed unit a draft + inventory; per section a
write + up to 2 reviews and revisions, each with research. Lower
`--review-rounds` / `--research-iterations` for cheaper runs.

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
tests/          unit tests, plus an end-to-end run with a scripted model
```

## Limitations

- Handwriting and low-contrast slides may not read reliably; triage and the
  writers ground ambiguous content against the literature.
- Web search is best-effort (keyless DuckDuckGo) and may be rate-limited; arXiv
  is the reliable backbone. The `fetch_url` host check resolves DNS before the
  request, so DNS rebinding is not covered.
- PDF documents contribute text only (their figures are not extracted).
- Long inputs: there is no hard length limit. Videos are drafted in 8-minute
  units; frames are capped at 20 distinct states per minute; requests stream
  (up to 128K output tokens, longer output is continued); citations are
  resolved in batches of 12. The outline sees the inventory of all sources in
  one request, which fits the 1M-token context for many hours of material.
  Cost and time grow linearly with input length. Amazon Transcribe accepts at
  most 4 hours of audio per file; local Whisper has no limit (~8x real time on
  CPU with the `small` model).
- faster-whisper's VAD needs onnxruntime, which has no wheel for Python 3.12 on
  glibc < 2.28 (e.g. Amazon Linux 2); vitex then uses ffmpeg silence detection.
- Bedrock model ids differ by account/region; set `VITEX_MODEL` (and
  `VITEX_TRIAGE_MODEL`) to profiles you have access to.
