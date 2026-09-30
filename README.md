# vitex

Reconstruct a tutorial-style video into a structured, self-contained **LaTeX**
technical document — not a summary, but a faithful, step-by-step rebuild of the
instruction, using **Claude** (over Amazon Bedrock by default) to read both the
narration *and* what is shown on screen.

vitex transcribes the audio with timestamps, samples and de-duplicates the
frames, aligns each distinct on-screen visual to the narration it accompanies,
and asks Claude to rebuild the material section by section: definitions,
intuition, equations, and derivations preserved step by step, every symbol
defined on first use, on-screen math and diagrams recovered from the frames.

## How it works

```
source (URL or file)
  │  yt-dlp / local
  ▼
video ──► audio (ffmpeg, 16 kHz mono) ──► transcript with timestamps
  │                                          (faster-whisper | Amazon Transcribe)
  └──► frames (ffmpeg sample + perceptual-hash de-dup)
                     │
        time-based chunking + frame↔narration alignment
                     │
        per-chunk draft  (Claude: transcript + frames + research tools ──► LaTeX fragment)
                     │
        integration pass (Claude: fragments ──► one coherent .tex)
                     │
        bibliography pass (Claude + research tools: \cite keys ──► thebibliography)
                     │
        compile (latexmk) with a Claude-driven repair loop ──► PDF
```

The single most important design choice is that **everything carries real
timestamps**. Because transcription is segment-timestamped and frames are
timestamped, each distinct slide/diagram is handed to the exact section that was
being narrated when it appeared — instead of being smeared across the document
by a character-count heuristic.

## Authoring philosophy

vitex writes an **authoritative, self-contained tutorial that uses the video as a
seed** — not a transcript. Concretely:

- **Intuition first.** Each concept is built up in the order a learner needs it:
  the problem it solves, then the intuition or a concrete analogy, then the math
  with every symbol tied back to the picture — never a bare run of equations.
- **Clarity is woven into the prose.** When the source is thin, the model adds the
  missing motivation, definitions, and derivation steps directly in the text, the
  way a textbook would — not as flagged "editorial" asides.
- **Grounded while writing.** During generation the model uses keyless research
  tools (the arXiv API, a DuckDuckGo web search, and page fetching) to confirm
  formulas and attributions and to find primary sources. This is part of the
  generation loop, not a postfix.
- **Real references.** Claims are cited inline with `\cite{...}`, and a dedicated
  pass resolves every key into a correct `thebibliography` entry (title, authors,
  year, arXiv id / URL). Grounding replaces the old "needs verification" notes:
  where the video was ambiguous or a frame was cut off, the model states the
  correct version from the source of record.
- **Figure-rich, filtered only for quality.** The model is shown candidate frames
  and includes every one that genuinely teaches — architecture and mechanism
  diagrams, worked examples, comparison tables — erring toward more, each with a
  caption and discussed in the prose. It excludes only low-value or illegible
  frames (title/talking-head/duplicate/cut-off); valuable-but-poor frames are
  reconstructed in text (e.g. the full equation typeset) instead of embedded.
  Only the frames the final document references are staged.

Grounding is on by default; run with `--no-ground` (or `VITEX_GROUND=0`) for a
fully offline run that writes from the model's own knowledge.

## Install

Requires Python 3.12+, plus the external tools `ffmpeg` (and `ffprobe`) and, for
URL inputs, `yt-dlp` (bundled as a Python dependency).

```bash
# ffmpeg is a system binary — install it once:
#   macOS:  brew install ffmpeg
#   Debian: sudo apt-get install ffmpeg

# default install (Bedrock + local Whisper transcription)
uv pip install -e ".[whisper,dev]"

# to use Amazon Transcribe instead of local Whisper:
uv pip install -e ".[whisper,aws,dev]"
```

Copy `.env.example` to `.env` and adjust (model id, region, transcriber).

## Usage

```bash
# Bedrock + local Whisper (uses your ambient AWS credentials)
vitex run "https://www.youtube.com/watch?v=XXXX"

# a local file, .tex only (skip PDF), more frames per chapter
vitex run lecture.mp4 --no-pdf --max-images 16

# use the Anthropic API instead of Bedrock
vitex run "https://youtu.be/XXXX" --provider anthropic --model claude-sonnet-4-20250514

# use Amazon Transcribe (needs VITEX_TRANSCRIBE_BUCKET set)
vitex run "https://youtu.be/XXXX" --transcriber aws
```

Outputs land in `output/`: `<title>.tex`, `<title>.pdf`, and `figures/`.

### Resuming

Every expensive stage caches under `work/`. Re-running the same command picks up
where it left off (download, transcript, frames, and per-section drafts are all
reused). Pass `--force` to recompute everything.

## Configuration

Precedence: built-in defaults → `.env`/environment → CLI flags. Key options:

| Flag | Env | Meaning |
|---|---|---|
| `--provider` | `VITEX_PROVIDER` | `bedrock` (default) or `anthropic` |
| `--model` | `VITEX_MODEL` | model id / Bedrock inference-profile id |
| `--fix-model` | `VITEX_FIX_MODEL` | model for the LaTeX repair pass |
| `--aws-region` | `AWS_REGION` | region for Bedrock / Transcribe |
| `--aws-profile` | `VITEX_AWS_PROFILE` | named `~/.aws` profile for Bedrock (default: ambient credentials) |
| `--ground` / `--no-ground` | `VITEX_GROUND` | literature grounding + citations, on by default |
| `--transcriber` | `VITEX_TRANSCRIBER` | `whisper` (default) or `aws` |
| `--whisper-model` | `VITEX_WHISPER_MODEL` | `tiny`…`large-v3` |
| `--frame-interval` | — | seconds between sampled frames (pre-dedup) |
| `--chunk-minutes` | — | transcript minutes per generated chapter |
| `--max-images` | — | frames sent per chapter |
| `--no-pdf` / `--force` | — | skip compilation / ignore cache |

## Package layout

```
src/vitex/
  cli.py            argparse entry point
  config.py         Config dataclass (env + CLI)
  pipeline.py       stage orchestration + caching
  models.py         Segment / Frame / Chunk / ChapterDraft
  media/            download, audio, frames
  transcribe/       whisper_local, aws_transcribe (+ factory)
  llm/              Claude client (Bedrock/Anthropic) + prompts + research tools
  reconstruct/      segmenter, align, chapters (draft + integrate + bibliography)
  output/           latex (preamble/assembly), build (compile + repair)
tests/              unit tests for the pure logic
```

## Limitations

- Handwriting and low-contrast slides may not read reliably; the model grounds
  ambiguous content against the literature, but cross-check unusual claims.
- Web search is best-effort (keyless DuckDuckGo endpoint) and may be rate-limited;
  arXiv is the reliable backbone for citations.
- Frame de-duplication is perceptual; heavy camera motion inflates the frame
  count (raise `--frame-interval` or lower `max_frames`).
- Bedrock model ids differ by account/region; set `VITEX_MODEL` to one you have
  access to (cross-region inference profiles are prefixed, e.g. `us.`).
```
