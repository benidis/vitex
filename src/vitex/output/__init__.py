"""LaTeX assembly and PDF building."""

from vitex.output.build import build_pdf
from vitex.output.latex import (
    assemble_document,
    build_preamble,
    cite_keys,
    drop_unknown_figures,
    insert_bibliography,
    referenced_figures,
    stage_figures,
    write_document,
)

__all__ = [
    "assemble_document",
    "build_preamble",
    "stage_figures",
    "write_document",
    "build_pdf",
    "cite_keys",
    "drop_unknown_figures",
    "insert_bibliography",
    "referenced_figures",
]
