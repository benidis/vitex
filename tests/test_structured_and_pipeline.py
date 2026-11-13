"""Tests for structured (tool-submitted) answers and an end-to-end pipeline run
over a document seed with a scripted model (no network, no TeX)."""

import re
from types import SimpleNamespace

import pytest

from vitex import pipeline as pipeline_mod
from vitex.config import Config
from vitex.errors import LLMError
from vitex.llm.client import ClaudeClient
from vitex.llm.prompts import EDITORIAL_SCHEMA, INVENTORY_SCHEMA, OUTLINE_SCHEMA, REVIEW_SCHEMA
from vitex.output.build import content_losses
from vitex.pipeline import Pipeline


def _bare(**cfg):
    c = ClaudeClient.__new__(ClaudeClient)
    c.cfg = Config(**cfg)
    c.model = "m"
    return c


def _tool(name, inp, id_="t1"):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=inp)


def test_structured_submit_is_requested_not_forced():
    client = _bare()
    seen = []
    script = [
        SimpleNamespace(content=[SimpleNamespace(type="text", text="Here is my answer...")], stop_reason="end_turn"),
        SimpleNamespace(content=[_tool("submit", {"x": 1})], stop_reason="tool_use"),
    ]

    def fake(**kw):
        seen.append(kw)
        return script.pop(0)

    client._create = fake
    assert client.complete_structured(system="s", text="u", schema={"type": "object"}, description="d") == {"x": 1}
    assert all("tool_choice" not in kw for kw in seen)  # forced tool_choice is rejected by current models
    assert "submit" in seen[1]["messages"][-1]["content"][-1]["text"]  # nudged after a prose answer


def test_structured_research_then_submit():
    client = _bare(research_max_iterations=3)
    script = [
        SimpleNamespace(content=[_tool("arxiv_search", {"query": "q"})], stop_reason="tool_use"),
        SimpleNamespace(content=[SimpleNamespace(type="text", text="thinking")], stop_reason="end_turn"),
        SimpleNamespace(content=[_tool("submit", {"ok": True}, "t2")], stop_reason="tool_use"),
    ]
    client._create = lambda **kw: script.pop(0)
    out = client.complete_structured(
        system="s", text="u", schema={}, description="d",
        tools=[{"name": "arxiv_search"}], dispatch=lambda n, i: "res",
    )
    assert out == {"ok": True}


def test_structured_truncated_submission_raises():
    client = _bare()
    client._create = lambda **kw: SimpleNamespace(content=[_tool("submit", {"partial": 1})], stop_reason="max_tokens")
    with pytest.raises(LLMError):
        client.complete_structured(system="s", text="u", schema={}, description="d")


def test_content_losses_flags_deletions():
    before = "\\section{A} x \\cite{k} \\includegraphics{f.jpg} " + "y" * 1000
    assert content_losses(before, before + " fixed") == []
    after = "\\section{A} x " + "y" * 900
    losses = " ".join(content_losses(before, after))
    assert "figure" in losses and "citation" in losses and "shortened" in losses


class ScriptedModel:
    """Answers every pass of the pipeline based on its prompt."""

    def __init__(self, cfg):
        self.calls = []

    def complete(self, *, system, text, **kw):
        self.calls.append(("text", system[:40]))
        if "END OF DRAFT" in system:
            fig = re.search(r"- (\S+\.png) \|", text)
            figure = f"\\includegraphics{{{fig.group(1)}}}" if fig else ""
            return f"\\section{{Draft}}\nVerified content \\cite{{vaswani2017attention}}. {figure}\n% END OF DRAFT"
        if "END OF SECTION" in system:
            title = re.search(r"^Title: (.+)$", text, re.MULTILINE).group(1)
            figs = re.findall(r"- (\S+\.png) \|", text.split("Citation keys")[0])
            body = " ".join(f"\\includegraphics{{{f}}}" for f in figs)
            return f"\\section{{{title}}}\nStep-by-step text \\cite{{vaswani2017attention}}. {body}\n% END OF SECTION"
        if "bibliography" in system:
            return "\\begin{thebibliography}{99}\n\\bibitem{vaswani2017attention} A. Vaswani et al. 2017.\n\\end{thebibliography}"
        raise AssertionError(f"unexpected text pass: {system[:80]}")

    complete_with_tools = complete

    def complete_structured(self, *, system, text, schema, **kw):
        self.calls.append(("json", system[:40]))
        if schema is INVENTORY_SCHEMA:
            return {"summary": "s", "items": [{"kind": "concept", "title": "Attention", "detail": "QK^T"}]}
        if schema is OUTLINE_SCHEMA:
            ids = re.findall(r"^(I\d+\.\d+) \|", text, re.MULTILINE)
            figs = re.findall(r"^\s+(\S+\.png) \|", text, re.MULTILINE)
            sec = lambda t, r, i, f=(): {"title": t, "role": r, "purpose": "why here",  # noqa: E731
                                         "plan": ["p"], "items": list(i), "figures": list(f)}
            return {"title": "Attention From Scratch", "storyline": "s", "reader_background": "b", "notation": [],
                    "sections": [sec("Introduction", "introduction", []), sec("Core", "core", ids, figs),
                                 sec("Summary", "summary", [])]}
        if schema is REVIEW_SCHEMA:
            return {"approved": True, "missing_items": [], "issues": []}
        if schema is EDITORIAL_SCHEMA:
            assert "% ===== SECTION 2: Core" in text  # the editor sees the whole document
            return {"assessment": "reads well", "issues": []}
        raise AssertionError("unexpected structured pass")


def test_end_to_end_document_seed(tmp_path, monkeypatch):
    from PIL import Image

    (tmp_path / "doc").mkdir()
    Image.new("RGB", (8, 8), "red").save(tmp_path / "doc" / "fig.png")
    (tmp_path / "doc" / "old.tex").write_text(
        "\\title{Old}\\begin{document}\\section{Attention}\nText.\n\\includegraphics{fig}\n\\end{document}"
    )
    monkeypatch.setattr(pipeline_mod, "ClaudeClient", ScriptedModel)
    cfg = Config(docs=[tmp_path / "doc" / "old.tex"], workdir=tmp_path / "work", outdir=tmp_path / "out",
                 build_pdf=False, ground=False, provider="anthropic", anthropic_api_key="test")
    result = Pipeline(cfg).run()

    tex = result.tex_path.read_text()
    assert result.stem == "Attention_From_Scratch"
    assert tex.count("\\section{") == 3 and "\\begin{thebibliography}" in tex
    assert tex.rstrip().endswith("\\end{document}") and "END OF" not in tex
    staged = list((result.tex_path.parent / "figures").iterdir())
    assert len(staged) == 1 and staged[0].name in tex  # the document's figure survived end to end
    assert (tmp_path / "work" / "tutorial" / "coverage.json").exists()
    assert (tmp_path / "work" / "tutorial" / "editorial.json").exists()


def test_conform_decodes_stringified_nesting_and_rejects_mismatches():
    from vitex.llm.client import conform
    from vitex.llm.prompts import TRIAGE_SCHEMA

    frame = {"name": "f.jpg", "kind": "diagram", "legible": "true", "complete": True, "value": "7",
             "superseded_by": None, "description": "d", "text": "", "reason": "r"}
    import json as _json

    # The exact failure seen live: the array arrives as a JSON string.
    out = conform({"frames": _json.dumps([frame])}, TRIAGE_SCHEMA)
    assert out["frames"][0]["value"] == 7 and out["frames"][0]["legible"] is True
    with pytest.raises(ValueError, match="missing required"):
        conform({"frames": [{"name": "f.jpg"}]}, TRIAGE_SCHEMA)
    with pytest.raises(ValueError, match="non-JSON"):
        conform({"frames": "not json"}, TRIAGE_SCHEMA)


def test_structured_answer_not_matching_schema_raises_for_retry():
    client = _bare()
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    client._create = lambda **kw: SimpleNamespace(content=[_tool("submit", {"n": "many"})], stop_reason="tool_use")
    with pytest.raises(LLMError, match="does not match its schema"):
        client.complete_structured(system="s", text="u", schema=schema, description="d")
