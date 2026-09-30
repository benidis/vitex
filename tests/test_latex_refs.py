"""Tests for reading references back out of a generated document."""

from vitex.models import Frame
from vitex.output.latex import (
    cite_keys,
    insert_bibliography,
    referenced_figures,
    stage_figures,
)


def test_referenced_figures_extracts_basenames():
    doc = r"""
    \includegraphics[width=0.8\linewidth]{frame_00010.jpg}
    text \includegraphics{figures/frame_00020.jpg}
    \includegraphics[width=3cm]{frame_00010.jpg}
    """
    assert referenced_figures(doc) == {"frame_00010.jpg", "frame_00020.jpg"}


def test_cite_keys_ordered_deduped_and_multi():
    doc = r"see \citep{schulman2017ppo} and \cite{schulman2016gae,shao2024deepseekmath} " \
          r"then \citet{schulman2017ppo} again"
    assert cite_keys(doc) == ["schulman2017ppo", "schulman2016gae", "shao2024deepseekmath"]


def test_insert_bibliography_before_end_document():
    doc = "\\begin{document}\nBody.\n\\end{document}\n"
    bib = "\\begin{thebibliography}{99}\n\\bibitem{a} A.\n\\end{thebibliography}"
    out = insert_bibliography(doc, bib)
    assert out.index("thebibliography") < out.index("\\end{document}")
    assert "Body." in out and out.count("\\end{document}") == 1


def test_insert_bibliography_noop_when_empty():
    doc = "\\begin{document}\nBody.\n\\end{document}\n"
    assert insert_bibliography(doc, "   ") == doc


def test_stage_figures_wipes_stale_frames(tmp_path):
    # A prior run left frames in output/figures; staging a curated subset must
    # not leave the old ones behind (that resurrects the "dump everything" bug).
    figures = tmp_path / "figures"
    figures.mkdir()
    (figures / "stale_00099.jpg").write_bytes(b"old")

    src = tmp_path / "src.jpg"
    src.write_bytes(b"new")
    frame = Frame(index=0, timestamp=0.0, path=src, label="")

    stage_figures([frame], tmp_path)

    assert {p.name for p in figures.iterdir()} == {"src.jpg"}
