"""LaTeX preamble, figure staging, and document writing.

vitex owns the preamble (so the document reliably compiles) and lets the model
fill everything from ``\\begin{document}`` onward. It also provides small helpers
to read back what the generated document references — the figures it actually
uses and the citation keys it needs — so we stage only those figures and build a
matching bibliography.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

from vitex.models import Figure

log = logging.getLogger("vitex")

# Deliberately limited to packages present in a standard TeX Live install.
_PREAMBLE_TEMPLATE = r"""
\documentclass[11pt]{article}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath, amssymb, amsthm, mathtools}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{listings}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{tabularx}
\usepackage[hidelinks]{hyperref}
\usepackage{xurl}

% Keep content inside the text width: let TeX stretch spacing before a line
% overflows, and let long derivations break across pages.
\setlength{\emergencystretch}{3em}
\allowdisplaybreaks

\graphicspath{{figures/}}

\theoremstyle{definition}
\newtheorem{definition}{Definition}
\newtheorem{example}{Example}
\theoremstyle{plain}
\newtheorem{theorem}{Theorem}

\lstset{basicstyle=\ttfamily\small, breaklines=true, columns=fullflexible}

\title{__VITEX_TITLE__}
\date{}
""".strip()

_INCLUDEGRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
_CITE_RE = re.compile(r"\\cite[tp]?\*?(?:\[[^\]]*\])?\{([^}]+)\}")
_BIBITEM_RE = re.compile(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}")
_END_BIB_RE = re.compile(r"\\end\{thebibliography\}")
_END_DOC_RE = re.compile(r"\\end\{document\}")
_FIGURE_ENV_RE = re.compile(r"\\begin\{(figure\*?)\}.*?\\end\{\1\}\s*", re.DOTALL)


def referenced_figures(document: str) -> set[str]:
    """Return the set of image file names the document actually \\includegraphics."""
    names = set()
    for arg in _INCLUDEGRAPHICS_RE.findall(document):
        names.add(Path(arg.strip()).name)
    return names


def drop_unknown_figures(document: str, allowed: set[str]) -> tuple[str, list[str]]:
    """Remove figures whose image is not one of the ``allowed`` file names.

    The model occasionally invents or misspells a frame file name; that is a
    guaranteed compile failure, and repairing it costs a full LLM pass that may
    damage other figures. A whole ``figure`` environment is dropped when any
    image in it is unknown; a stray ``\\includegraphics`` outside a figure is
    dropped on its own. Returns the cleaned document and the removed names.
    """
    removed: list[str] = []

    def unknown(fragment: str) -> list[str]:
        return [
            a.strip() for a in _INCLUDEGRAPHICS_RE.findall(fragment)
            if Path(a.strip()).name not in allowed
        ]

    def drop_env(m: re.Match) -> str:
        bad = unknown(m.group(0))
        if not bad:
            return m.group(0)
        removed.extend(bad)
        return ""

    def drop_cmd(m: re.Match) -> str:
        if Path(m.group(1).strip()).name in allowed:
            return m.group(0)
        removed.append(m.group(1).strip())
        return ""

    document = _FIGURE_ENV_RE.sub(drop_env, document)
    document = _INCLUDEGRAPHICS_RE.sub(drop_cmd, document)
    return document, removed


def cite_contexts(document: str, keys: list[str], width: int = 300, per_key: int = 3) -> dict[str, list[str]]:
    """Return up to ``per_key`` snippets of surrounding text for each cite key.

    The bibliography pass needs to know which work each key means; showing the
    text around every citation (instead of a truncated prefix of the document)
    gives it that for every key, however late in the document it appears.
    """
    contexts: dict[str, list[str]] = {k: [] for k in keys}
    for m in _CITE_RE.finditer(document):
        snippet = " ".join(document[max(0, m.start() - width) : m.end() + width].split())
        for key in m.group(1).split(","):
            bucket = contexts.get(key.strip())
            if bucket is not None and len(bucket) < per_key:
                bucket.append(snippet)
    return contexts


def bibitem_keys(bibliography: str) -> set[str]:
    """Return the set of keys that have a ``\\bibitem`` entry."""
    return {k.strip() for k in _BIBITEM_RE.findall(bibliography)}


def complete_bibliography(bibliography: str, keys: list[str]) -> tuple[str, list[str]]:
    """Ensure every cite key has a ``\\bibitem``; return the result and the gaps.

    A key with no entry renders as "[?]" in the PDF. Missing keys get an honest
    placeholder entry (not an invented reference) so the gap is visible.
    """
    missing = [k for k in keys if k not in bibitem_keys(bibliography)]
    if not missing:
        return bibliography, []
    filler = "\n".join(
        f"\\bibitem{{{k}}} Reference details could not be resolved for \\texttt{{\\detokenize{{{k}}}}}."
        for k in missing
    )
    m = None
    for m in _END_BIB_RE.finditer(bibliography):
        pass  # keep the last match
    if m is None:
        body = bibliography.strip()
        return f"\\begin{{thebibliography}}{{99}}\n{body}\n{filler}\n\\end{{thebibliography}}", missing
    return bibliography[: m.start()] + filler + "\n" + bibliography[m.start() :], missing


def cite_keys(document: str) -> list[str]:
    """Return the ordered, de-duplicated list of \\cite keys used in the document."""
    keys: list[str] = []
    for group in _CITE_RE.findall(document):
        for key in group.split(","):
            key = key.strip()
            if key and key not in keys:
                keys.append(key)
    return keys


def insert_bibliography(document: str, bibliography: str) -> str:
    """Insert a thebibliography block just before the final \\end{document}."""
    bibliography = bibliography.strip()
    if not bibliography:
        return document
    m = None
    for m in _END_DOC_RE.finditer(document):
        pass  # keep the last match
    if m is None:
        return document.rstrip() + "\n\n" + bibliography + "\n\\end{document}\n"
    return document[: m.start()] + bibliography + "\n\n" + document[m.start() :]


def _tex_escape(text: str) -> str:
    """Escape the characters that would break a LaTeX title."""
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(c, c) for c in text)


def build_preamble(title: str) -> str:
    """Return the preamble with the (escaped) document title filled in.

    A literal sentinel replace is used (not ``%``-formatting) because the
    preamble is full of LaTeX ``%`` comment characters.
    """
    return _PREAMBLE_TEMPLATE.replace("__VITEX_TITLE__", _tex_escape(title))


def stage_figures(figures: list[Figure], outdir: Path) -> None:
    """Copy the figures the document references into ``outdir/figures``.

    Each figure is staged under its unique name, so every ``\\includegraphics``
    the model wrote resolves.

    The directory is wiped first so a re-run never leaves behind figures the
    current document no longer references.
    """
    figures_dir = outdir / "figures"
    if figures_dir.exists():
        shutil.rmtree(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    for f in figures:
        if f.path.exists():
            shutil.copy2(f.path, figures_dir / f.name)
    log.info("Staged %d figures into %s", len(figures), figures_dir)


def assemble_document(preamble: str, sections: list[str]) -> str:
    """Join the preamble and the written sections into one complete document."""
    body = "\n\n".join(s.strip() for s in sections)
    return (
        f"{preamble}\n\n\\begin{{document}}\n\\maketitle\n\\tableofcontents\n\\newpage\n\n"
        f"{body}\n\n\\end{{document}}\n"
    )


def write_document(document: str, outdir: Path, stem: str) -> Path:
    """Write the final ``.tex`` and return its path."""
    outdir.mkdir(parents=True, exist_ok=True)
    tex_path = outdir / f"{stem}.tex"
    tex_path.write_text(document, encoding="utf-8")
    log.info("Wrote LaTeX: %s", tex_path)
    return tex_path
