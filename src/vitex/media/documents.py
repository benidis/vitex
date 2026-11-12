"""Load documents (e.g. an existing tutorial to improve) as seeds.

Supported: LaTeX (``\\input``/``\\include`` are inlined and figures are carried
over), Markdown (images carried over), plain text, and PDF (text only; needs the
``pdf`` extra or ``pdftotext``). Each figure is copied into the work dir under a
globally unique name, and the document text is rewritten to use that name, so a
document's figures can appear in the final tutorial like video frames.

Documents are split into seed units at section boundaries, each at most
``doc_unit_chars`` characters, so every part gets a full drafting pass.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from vitex.errors import DependencyError
from vitex.models import Figure
from vitex.utils import run

log = logging.getLogger("vitex")

_TEXT_SUFFIXES = {".tex", ".md", ".markdown", ".txt", ".rst"}
_IMAGE_EXTS = ["", ".pdf", ".png", ".jpg", ".jpeg", ".eps"]

_INPUT_RE = re.compile(r"\\(?:input|include)\{([^}]+)\}")
_GRAPHICSPATH_RE = re.compile(r"\\graphicspath\{((?:\{[^}]*\})+)\}")
_INCLUDEGRAPHICS_RE = re.compile(r"(\\includegraphics(?:\[[^\]]*\])?\{)([^}]+)(\})")
_MD_IMAGE_RE = re.compile(r"(!\[[^\]]*\]\()([^)\s]+)(\))")
_TEX_TITLE_RE = re.compile(r"\\title\{(.+?)\}", re.DOTALL)
_TEX_BREAK_RE = re.compile(r"^(?=\\(?:part|chapter|section)\*?\{)", re.MULTILINE)
_MD_BREAK_RE = re.compile(r"^(?=#{1,2} )", re.MULTILINE)
_PARAGRAPH_RE = re.compile(r"(?<=\n\n)")  # split after blank lines, keeping them


@dataclass
class Document:
    slug: str  # unique id, also the figure-name prefix
    title: str
    path: Path
    text: str
    figures: list[Figure] = field(default_factory=list)


def _slug_for(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9]+", "-", path.stem).strip("-")[:30] or "doc"
    digest = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:6]
    return f"doc-{stem}-{digest}"


def _inline_tex(path: Path, root: Path, depth: int = 0) -> str:
    """Return ``path``'s text with ``\\input``/``\\include`` files spliced in."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if depth > 10:
        return text

    def splice(m: re.Match) -> str:
        target = root / m.group(1).strip()
        if not target.suffix:
            target = target.with_suffix(".tex")
        return _inline_tex(target, root, depth + 1) if target.is_file() else m.group(0)

    return _INPUT_RE.sub(splice, text)


def _resolve_image(arg: str, dirs: list[Path]) -> Path | None:
    for d in dirs:
        for ext in _IMAGE_EXTS:
            candidate = d / (arg.strip() + ext)
            if candidate.is_file():
                return candidate
    return None


def _carry_figures(text: str, pattern: re.Pattern, dirs: list[Path], slug: str, fig_dir: Path) -> tuple[str, list[Figure]]:
    """Copy every referenced image into ``fig_dir`` under a unique name and
    rewrite the references to it. Unresolvable images are left untouched."""
    figures: dict[str, Figure] = {}

    def rewrite(m: re.Match) -> str:
        src = _resolve_image(m.group(2), dirs)
        if src is None:
            log.warning("Document figure not found, left as-is: %s", m.group(2))
            return m.group(0)
        name = f"{slug}_{re.sub(r'[^A-Za-z0-9_-]+', '_', src.stem)}{src.suffix.lower()}"
        if name not in figures:
            fig_dir.mkdir(parents=True, exist_ok=True)
            dest = fig_dir / name
            shutil.copy2(src, dest)
            figures[name] = Figure(name=name, path=dest, hint="figure from the document")
        return m.group(1) + name + m.group(3)

    return pattern.sub(rewrite, text), list(figures.values())


def _pdf_text(path: Path) -> str:
    if shutil.which("pdftotext"):
        return run(["pdftotext", "-layout", str(path), "-"], desc="pdftotext")
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise DependencyError(
            "Reading PDF documents needs pdftotext (poppler) or pypdf:\n  uv pip install -e '.[pdf]'"
        ) from exc
    reader = PdfReader(str(path))
    return "\n\n".join(f"[page {i}]\n{p.extract_text() or ''}" for i, p in enumerate(reader.pages, 1))


def load_document(path: Path, workroot: Path) -> Document:
    """Read ``path`` into a :class:`Document`, carrying over its figures."""
    path = Path(path)
    slug = _slug_for(path)
    fig_dir = workroot / slug / "figures"
    suffix = path.suffix.lower()
    title = path.stem
    figures: list[Figure] = []

    if suffix == ".pdf":
        text = _pdf_text(path)
    elif suffix == ".tex":
        root = path.parent
        text = _inline_tex(path, root)
        dirs = [root] + [root / d for d in re.findall(r"\{([^}]*)\}", "".join(_GRAPHICSPATH_RE.findall(text)))]
        text, figures = _carry_figures(text, _INCLUDEGRAPHICS_RE, dirs, slug, fig_dir)
        if m := _TEX_TITLE_RE.search(text):
            title = " ".join(m.group(1).replace("\\\\", " ").split())
    elif suffix in (".md", ".markdown"):
        text = path.read_text(encoding="utf-8", errors="replace")
        text, figures = _carry_figures(text, _MD_IMAGE_RE, [path.parent], slug, fig_dir)
        if m := re.search(r"^# (.+)$", text, re.MULTILINE):
            title = m.group(1).strip()
    elif suffix in _TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8", errors="replace")
    else:
        raise DependencyError(f"Unsupported document type {suffix!r} ({path}); use .tex, .md, .txt, or .pdf.")

    if not text.strip():
        raise DependencyError(f"Document {path} has no readable text.")
    log.info("Document: %s (%d chars, %d figures)", path, len(text), len(figures))
    return Document(slug=slug, title=title, path=path, text=text, figures=figures)


def _pack(pieces: list[str], max_chars: int) -> list[str]:
    """Greedily join consecutive pieces into units of at most ``max_chars``."""
    units: list[str] = []
    for piece in pieces:
        if units and len(units[-1]) + len(piece) <= max_chars:
            units[-1] += piece
        else:
            units.append(piece)
    return units


def _split_long(piece: str, max_chars: int) -> list[str]:
    if len(piece) <= max_chars:
        return [piece]
    paragraphs = _PARAGRAPH_RE.split(piece)
    hard = [p[i : i + max_chars] for p in paragraphs for i in range(0, len(p), max_chars)]
    return _pack(hard, max_chars)


def split_document(doc: Document, max_chars: int) -> list[str]:
    """Split the text into units at section boundaries; none exceeds ``max_chars``."""
    suffix = doc.path.suffix.lower()
    if suffix == ".tex":
        pieces = _TEX_BREAK_RE.split(doc.text)
    elif suffix in (".md", ".markdown"):
        pieces = _MD_BREAK_RE.split(doc.text)
    else:
        pieces = _PARAGRAPH_RE.split(doc.text)
    pieces = [s for p in pieces if p.strip() for s in _split_long(p, max_chars)]
    return _pack(pieces, max_chars)
