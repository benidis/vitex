"""Tests for loading documents as seeds: LaTeX inputs, figures, splitting."""

from PIL import Image

from vitex.config import Config
from vitex.media.documents import load_document, split_document
from vitex.reconstruct.seeds import document_units


def _tex_project(tmp_path):
    root = tmp_path / "tut"
    (root / "sections").mkdir(parents=True)
    (root / "figures").mkdir()
    Image.new("RGB", (10, 10), "red").save(root / "figures" / "diagram.png")
    (root / "main.tex").write_text(
        "\\documentclass{article}\n\\graphicspath{{figures/}}\n\\title{My RL Tutorial}\n"
        "\\begin{document}\n\\input{sections/one}\n\\include{sections/two}\n\\end{document}\n"
    )
    (root / "sections" / "one.tex").write_text(
        "\\section{One}\nIntro text.\n\\includegraphics[width=3cm]{diagram}\n"
    )
    (root / "sections" / "two.tex").write_text("\\section{Two}\n" + "Body paragraph.\n\n" * 50)
    return root / "main.tex"


def test_tex_inputs_are_inlined_and_figures_carried_over(tmp_path):
    doc = load_document(_tex_project(tmp_path), tmp_path / "work")
    assert doc.title == "My RL Tutorial"
    assert "\\section{One}" in doc.text and "\\section{Two}" in doc.text
    assert len(doc.figures) == 1
    fig = doc.figures[0]
    assert fig.name.startswith(doc.slug) and fig.name.endswith(".png") and fig.path.exists()
    assert f"\\includegraphics[width=3cm]{{{fig.name}}}" in doc.text


def test_split_at_sections_and_units_keep_their_figures(tmp_path):
    doc = load_document(_tex_project(tmp_path), tmp_path / "work")
    parts = split_document(doc, max_chars=400)
    assert all(len(p) <= 400 for p in parts)
    assert "".join(parts) == doc.text  # nothing lost in splitting
    units = document_units(doc, Config(doc_unit_chars=400))
    with_fig = [u for u in units if u.figures]
    assert len(with_fig) == 1 and "\\section{One}" in with_fig[0].text


def test_markdown_images_and_title(tmp_path):
    Image.new("RGB", (10, 10), "blue").save(tmp_path / "pic.jpg")
    md = tmp_path / "notes.md"
    md.write_text("# Attention Notes\n\n![a diagram](pic.jpg)\n\n## Part 2\ntext\n")
    doc = load_document(md, tmp_path / "work")
    assert doc.title == "Attention Notes"
    assert doc.figures and f"]({doc.figures[0].name})" in doc.text
