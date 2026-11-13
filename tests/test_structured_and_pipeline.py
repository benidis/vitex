"""Tests for structured (tool-submitted) answers and an end-to-end pipeline run
over a document seed with a scripted model (no network, no TeX)."""

import re
from types import SimpleNamespace

import pytest

from vitex import pipeline as pipeline_mod
from vitex.config import Config
from vitex.errors import LLMError
from vitex.llm.client import ClaudeClient
from vitex.llm.prompts import (
    EDITORIAL_SCHEMA,
    INVENTORY_SCHEMA,
    OUTLINE_SCHEMA,
    PLAN_SCHEMA,
    REVIEW_SCHEMA,
    SURVEY_SCHEMA,
)
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

    seen_outline: list[str] = []

    def __init__(self, cfg):
        self.calls = []

    def check_access(self):
        self.calls.append(("access", ""))

    def complete(self, *, system, text, **kw):
        self.calls.append(("text", system[:40]))
        if "END OF DRAFT" in system:
            assert "STEPPING STONES" in text or "MAIN LINE" in text  # drafts are scoped by the plan
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
        if schema is SURVEY_SCHEMA:
            assert "Text." in text  # the survey reads the seeds themselves
            return {"overview": "Attention maps queries to keys.", "topics": [
                {"topic": "Scaled dot-product attention", "role": "core", "why": "everything builds on it", "in_seeds": True},
                {"topic": "Multi-query attention", "role": "supporting", "why": "cuts the KV cache", "in_seeds": False},
            ]}
        if schema is PLAN_SCHEMA:
            entries = [
                {"name": "Scaled dot-product attention", "level": "main", "pages": 4, "reason": "the core mechanism"},
                {"name": "Multi-query attention", "level": "stepping", "pages": 1, "reason": "cuts the KV cache"},
                {"name": "Linear attention", "level": "excluded", "pages": 0, "reason": "not needed"},
            ]
            if "AUTHOR'S FEEDBACK" in text:  # revision: follow the feedback
                entries[1]["pages"] = 2
            return {"title": "Attention From Scratch", "reader": "ML engineers", "goal": "understand attention",
                    "throughline": "dot-product -> MQA", "entries": entries}
        if schema is INVENTORY_SCHEMA:
            return {"summary": "s", "items": [{"kind": "concept", "title": "Attention", "detail": "QK^T"}]}
        if schema is OUTLINE_SCHEMA:
            ScriptedModel.seen_outline.append(text)
            ids = re.findall(r"^(I\d+\.\d+) \|", text, re.MULTILINE)
            figs = re.findall(r"^\s+(\S+\.png) \|", text, re.MULTILINE)
            topics = [n for n in ("Scaled dot-product attention", "Multi-query attention") if f"- {n} |" in text]
            sec = lambda t, r, i, f=(), tp=(): {"title": t, "role": r, "purpose": "why here",  # noqa: E731
                                                 "plan": ["p"], "items": list(i), "figures": list(f), "topics": list(tp)}
            return {"title": "Attention From Scratch", "storyline": "s", "reader_background": "b", "notation": [],
                    "sections": [sec("Introduction", "introduction", []), sec("Core", "core", ids, figs, topics),
                                 sec("Summary", "summary", [])]}
        if schema is REVIEW_SCHEMA:
            return {"approved": True, "missing_items": [], "issues": []}
        if schema is EDITORIAL_SCHEMA:
            assert "% ===== SECTION 2: Core" in text  # the editor sees the whole document
            return {"assessment": "reads well", "issues": []}
        raise AssertionError("unexpected structured pass")


def _doc_cfg(tmp_path, **kw):
    from PIL import Image

    (tmp_path / "doc").mkdir(exist_ok=True)
    Image.new("RGB", (8, 8), "red").save(tmp_path / "doc" / "fig.png")
    (tmp_path / "doc" / "old.tex").write_text(
        "\\title{Old}\\begin{document}\\section{Attention}\nText.\n\\includegraphics{fig}\n\\end{document}"
    )
    return Config(docs=[tmp_path / "doc" / "old.tex"], workdir=tmp_path / "work", outdir=tmp_path / "out",
                  build_pdf=False, ground=False, provider="anthropic", anthropic_api_key="test", **kw)


def test_end_to_end_document_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_mod, "ClaudeClient", ScriptedModel)
    result = Pipeline(_doc_cfg(tmp_path, name="attn")).run()

    run = tmp_path / "work" / "runs" / "attn"
    tex = result.tex_path.read_text()
    assert result.stem == "attn" and result.tex_path == tmp_path / "out" / "attn" / "attn.tex"
    assert tex.count("\\section{") == 3 and "\\begin{thebibliography}" in tex
    assert "\\title{Attention From Scratch}" in tex  # the plan's title
    assert tex.rstrip().endswith("\\end{document}") and "END OF" not in tex
    staged = list((result.tex_path.parent / "figures").iterdir())
    assert len(staged) == 1 and staged[0].name in tex  # the document's figure survived end to end
    for name in ("plan.md", "survey.json", "outline.json", "coverage.json", "editorial.json", "run.log"):
        assert (run / name).exists(), name
    assert (tmp_path / "work" / "sources").is_dir()  # per-source files live apart from runs


def test_two_step_run_with_an_edited_plan_and_feedback(tmp_path, monkeypatch):
    from vitex.reconstruct.plan import load_plan

    monkeypatch.setattr(pipeline_mod, "ClaudeClient", ScriptedModel)
    ScriptedModel.seen_outline = []
    # Step 1: plan only.
    result = Pipeline(_doc_cfg(tmp_path, name="attn", plan_only=True)).run()
    plan_md = tmp_path / "work" / "runs" / "attn" / "plan.md"
    assert result.tex_path is None and result.plan_path == plan_md
    assert ScriptedModel.seen_outline == []  # nothing written yet
    plan = load_plan(plan_md)
    assert [e.level for e in plan.entries] == ["main", "stepping", "excluded"]

    # Feedback revises the plan and keeps the previous version.
    Pipeline(_doc_cfg(tmp_path, name="attn", plan_only=True, plan_feedback="More on MQA")).run()
    assert (plan_md.parent / "plan.v1.md").exists()
    assert load_plan(plan_md).entries[1].pages == 2

    # The author edits the plan by hand: MQA becomes excluded, saved elsewhere.
    edited = plan_md.read_text().replace("- Multi-query attention | 2 | cuts the KV cache",
                                         "")
    edited = edited.replace("## Excluded", "## Excluded\n- Multi-query attention | out of scope after all")
    mine = tmp_path / "my_plan.md"
    mine.write_text(edited)

    # Step 2: write the tutorial from exactly that plan.
    result = Pipeline(_doc_cfg(tmp_path, name="attn", plan_path=mine)).run()
    assert result.tex_path is not None and result.tex_path.exists()
    outline_prompt = ScriptedModel.seen_outline[-1]
    assert "- Multi-query attention | out of scope after all" in outline_prompt


def test_plan_markdown_round_trips_and_reports_errors():
    import pytest as _pytest

    from vitex.reconstruct.plan import Plan, PlanEntry, PlanError, parse_plan

    plan = Plan(title="T", budget_pages=70, reader="r", goal="g", throughline="t", entries=[
        PlanEntry("DQN", "main", 4, "first deep value method"),
        PlanEntry("Double DQN", "stepping", 1.5, "overestimation fix reused by TD3"),
        PlanEntry("Noisy Nets", "mention", 0, "off the path"),
        PlanEntry("C51", "excluded", 0, "not used later"),
    ], inputs="abc")
    back = parse_plan(plan.to_markdown())
    assert back == plan
    with _pytest.raises(PlanError, match="line"):
        parse_plan("# Plan: T\nBudget: 10 pages\n## Main line\n- DQN | four | reason\n")
    with _pytest.raises(PlanError, match="no main-line"):
        parse_plan("# Plan: T\n## Mentions\n- X | y\n")


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


def test_budget_depends_on_distinct_topics_not_on_source_volume():
    from vitex.reconstruct.survey import Survey

    topics = [{"topic": "PPO", "role": "core"}, {"topic": "A2C", "role": "supporting"},
              {"topic": "REINFORCE history", "role": "brief"}]
    # Same topics, whether they came from one document or five overlapping ones.
    assert Survey("o", topics).budget() == 3000 + 2500 + 900 + 120
    landscape = Survey("Overview text", topics).landscape()
    assert "CORE (1):" in landscape and "SUPPORTING (1):" in landscape and "Overview text" in landscape


def test_plan_entries_below_their_level_minimum_are_demoted():
    from vitex.reconstruct.plan import PlanEntry, enforce_minimums

    entries = [PlanEntry("PPO", "main", 4, "r"), PlanEntry("REINFORCE", "main", 1, "r"),
               PlanEntry("Dueling", "stepping", 0.5, "r"), PlanEntry("Double DQN", "stepping", 1, "r")]
    changes = enforce_minimums(entries)
    assert [(e.name, e.level) for e in entries] == [
        ("PPO", "main"), ("REINFORCE", "stepping"), ("Dueling", "mention"), ("Double DQN", "stepping")]
    assert len(changes) == 2


def test_sample_run_writes_only_the_chosen_sections(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_mod, "ClaudeClient", ScriptedModel)
    result = Pipeline(_doc_cfg(tmp_path, name="attn", sections=[2])).run()
    tex = result.tex_path.read_text()
    assert result.tex_path == tmp_path / "out" / "attn" / "attn_sample.tex"
    assert tex.count("\\section{") == 1 and "\\section{Core}" in tex and "(sample: sections 2)" in tex
    assert (tmp_path / "work" / "runs" / "attn" / "coverage.sample.json").exists()
    assert not (tmp_path / "work" / "runs" / "attn" / "editorial.json").exists()  # no editor on a sample
