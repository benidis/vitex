"""Compile the LaTeX document to PDF, with a Claude-driven repair loop.

First compile attempt uses the model's output verbatim. If it fails, the errors
(with the offending source lines) and the tail of the compiler log go to Claude
(see ``FIX_SYSTEM``), which replies with targeted SEARCH/REPLACE edits. Edits keep
the repair cheap and leave the rest of the document byte-for-byte unchanged; only
if they don't apply cleanly do we fall back to asking for the whole corrected
document. A repair that removes figures, citations, sections, or more than a
sliver of text is rejected — the build must be fixed, not the content cut. We
retry a bounded number of times.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

from vitex.config import Config
from vitex.errors import BuildError
from vitex.llm.client import ClaudeClient, staged
from vitex.llm.prompts import FIX_FULL_SYSTEM, FIX_SYSTEM, fix_user
from vitex.output.latex import cite_keys, referenced_figures

log = logging.getLogger("vitex")

_MAX_REPAIRS = 5
_MAX_ERRORS = 5  # error excerpts sent per repair round
_MAX_SHRINK = 0.03  # a repair may not remove more than this share of the text

_EDIT_RE = re.compile(r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)>>>>>>> REPLACE", re.DOTALL)
_LINE_RE = re.compile(r"^l\.(\d+)")


def _compile_once(tex_path: Path) -> tuple[bool, str]:
    """Run one PDF build; return (success, combined log).

    No ``-halt-on-error``: in nonstopmode TeX keeps going after an error, so one
    failed build reports every error and a single repair round can fix them all.
    """
    outdir = tex_path.parent
    if shutil.which("latexmk"):
        cmds = [
            [
                "latexmk",
                "-pdf",
                "-interaction=nonstopmode",
                f"-output-directory={outdir}",
                str(tex_path),
            ]
        ]
    elif shutil.which("pdflatex"):
        # Two passes so the table of contents and cross-references resolve.
        cmd = ["pdflatex", "-interaction=nonstopmode", f"-output-directory={outdir}", str(tex_path)]
        cmds = [cmd, cmd]
    else:
        raise BuildError("Neither latexmk nor pdflatex is on PATH; cannot build a PDF.")

    output = ""
    for cmd in cmds:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace"
        )
        output += proc.stdout
        if proc.returncode != 0:
            return False, output
    return tex_path.with_suffix(".pdf").exists(), output


def error_excerpts(tex_log: str, document: str) -> str:
    """Pull each ``! ...`` error from a TeX log with the source lines it points at."""
    lines = tex_log.splitlines()
    src = document.splitlines()
    out: list[str] = []
    for i, line in enumerate(lines):
        if not line.startswith("!"):
            continue
        block = [line]
        lineno = None
        for follow in lines[i + 1 : i + 8]:
            block.append(follow)
            if m := _LINE_RE.match(follow):
                lineno = int(m.group(1))
                break
        if lineno is not None and src:
            lo, hi = max(1, lineno - 3), min(len(src), lineno + 2)
            block.append(f"Source lines {lo}-{hi}:")
            block += [f"{n:>6}| {src[n - 1]}" for n in range(lo, hi + 1)]
        out.append("\n".join(block))
        if len(out) == _MAX_ERRORS:
            break
    return "\n\n".join(out)


def apply_edits(document: str, reply: str) -> str | None:
    """Apply SEARCH/REPLACE blocks from ``reply``; None if any fails to apply.

    Every SEARCH text must occur exactly once (in the document as edited so
    far), so an edit can never land in the wrong place.
    """
    edits = _EDIT_RE.findall(reply)
    if not edits:
        return None
    for old, new in edits:
        if not old or document.count(old) != 1:
            return None
        document = document.replace(old, new.removesuffix("\n"), 1)
    return document


def content_losses(before: str, after: str) -> list[str]:
    """What a repair removed that it must not: figures, citations, sections, text."""
    losses = []
    if lost := referenced_figures(before) - referenced_figures(after):
        losses.append(f"removed figure(s) {', '.join(sorted(lost))}")
    if lost_cites := set(cite_keys(before)) - set(cite_keys(after)):
        losses.append(f"removed citation(s) {', '.join(sorted(lost_cites))}")
    if after.count("\\section") < before.count("\\section"):
        losses.append("removed a section")
    if len(after) < (1 - _MAX_SHRINK) * len(before):
        losses.append(f"shortened the document by {100 * (1 - len(after) / len(before)):.0f}%")
    return losses


def _repair(client: ClaudeClient, document: str, user: str, cfg: Config) -> str | None:
    """One repair round: targeted edits, else a full rewrite. None if both fail."""
    from vitex.reconstruct.chapters import _check_document, strip_fences

    reply = client.complete(
        system=FIX_SYSTEM,
        text=user,
        model=cfg.effective_fix_model,
        max_tokens=cfg.max_output_tokens,
    )
    patched = apply_edits(document, strip_fences(reply))
    if patched is not None:
        return patched

    log.warning("Repair edits did not apply cleanly; requesting the full corrected document.")
    rewritten = strip_fences(
        client.complete(
            system=FIX_FULL_SYSTEM,
            text=user,
            model=cfg.effective_fix_model,
            max_tokens=max(cfg.max_output_tokens, 16384),
        )
    )
    if problem := _check_document(rewritten):
        log.warning("Full-document repair unusable (%s); discarding it.", problem)
        return None
    return rewritten


def build_pdf(
    client: ClaudeClient,
    document: str,
    tex_path: Path,
    cfg: Config,
) -> Path:
    """Write ``document`` to ``tex_path``, compile, and repair on failure.

    Returns the PDF path on success. On unrecoverable failure raises
    ``BuildError`` but leaves the latest ``.tex`` on disk for manual fixing.
    """
    tex_path.write_text(document, encoding="utf-8")
    tex_log = tex_path.with_suffix(".log")

    ok, logtxt = _compile_once(tex_path)
    if ok:
        log.info("PDF built: %s", tex_path.with_suffix(".pdf"))
        return tex_path.with_suffix(".pdf")

    rejected = ""
    for attempt in range(1, _MAX_REPAIRS + 1):
        log.warning("LaTeX build failed; repair attempt %d/%d", attempt, _MAX_REPAIRS)
        full_log = tex_log.read_text(encoding="utf-8", errors="replace") if tex_log.exists() else logtxt
        errors = error_excerpts(full_log, document)
        log_tail = "\n".join(logtxt.splitlines()[-60:])
        with staged(client, f"latex repair {attempt}/{_MAX_REPAIRS}"):
            patched = _repair(client, document, fix_user(document, errors, log_tail, rejected), cfg)
        if patched is None:
            continue
        # A repair must fix the build, never delete content to silence an error.
        if losses := content_losses(document, patched):
            rejected = "; ".join(losses) + ". Repair the problem without deleting content."
            log.warning("Rejected repair %d: %s", attempt, rejected)
            continue
        rejected = ""
        document = patched
        tex_path.write_text(document, encoding="utf-8")
        ok, logtxt = _compile_once(tex_path)
        if ok:
            log.info("PDF built after repair: %s", tex_path.with_suffix(".pdf"))
            return tex_path.with_suffix(".pdf")

    raise BuildError(
        f"LaTeX still fails after {_MAX_REPAIRS} repair attempts. "
        f"The document is saved at {tex_path}; inspect the build log to fix it by hand."
    )
