"""Tests for the outline's no-drop guarantee and the section review loop."""

from vitex.config import Config
from vitex.models import SeedUnit
from vitex.reconstruct.outline import Item, Outline, Section, build_outline, outline_problems, repair_outline
from vitex.reconstruct.writer import edit_tutorial, section_jobs, write_sections

SECTION_END = "% END OF SECTION"


def _items():
    return [
        Item(id="I1.01", unit_id="u1", kind="concept", title="Attention", detail="softmax(QK^T)V", figures=["f1.jpg"]),
        Item(id="I1.02", unit_id="u1", kind="derivation", title="Scaling", detail="divide by sqrt(d)"),
        Item(id="I2.01", unit_id="u2", kind="result", title="KV cache", detail="memory O(nd)"),
    ]


def _sec(title, role, items, figures=()):
    return Section(title=title, role=role, purpose="why it is needed here", plan=["p"], items=list(items),
                   figures=list(figures))


def test_repair_outline_places_every_item_and_figure():
    outline = Outline(title="T", reader_background="b", notation=[], sections=[
        _sec("Intro", "introduction", []),
        _sec("Attention", "core", ["I1.02", "BOGUS"]),
        _sec("Caching", "core", ["I2.01"], ["nope.jpg"]),
        _sec("Summary", "summary", []),
    ])
    missing, unknown = outline_problems(outline, _items(), {"f1.jpg"})
    assert missing == ["I1.01"] and unknown == ["BOGUS", "nope.jpg"]

    changes = repair_outline(outline, _items(), {"f1.jpg"})
    assert outline_problems(outline, _items(), {"f1.jpg"}) == ([], [])
    attention = outline.sections[1]
    assert "I1.01" in attention.items  # same unit as I1.02 -> same section
    assert "f1.jpg" in attention.figures  # figure follows its item
    assert any("attached I1.01" in c for c in changes)


class OutlineClient:
    """First outline forgets an item; the retry is complete."""

    def __init__(self):
        self.calls = []

    def complete_structured(self, **kw):
        self.calls.append(kw["text"])
        items = ["I1.01", "I1.02"] if len(self.calls) == 1 else ["I1.01", "I1.02", "I2.01"]
        return {"title": "T", "reader_background": "b", "notation": [], "sections": [
            {"title": "Body", "role": "core", "question": "q", "motivation": "m", "plan": ["p"],
             "items": items, "figures": [], "leads_to": "l"}]}


def test_build_outline_retries_with_the_missing_ids(tmp_path):
    units = [SeedUnit(id=u, kind="video", source_title="V", position="p", text="t") for u in ("u1", "u2")]
    client = OutlineClient()
    outline = build_outline(client, units, _items(), {}, [], Config(name="t", workdir=tmp_path, ground=False), 5000)
    assert outline.sections[0].items == ["I1.01", "I1.02", "I2.01"]
    assert len(client.calls) == 2 and "I2.01" in client.calls[1]


class WriterClient:
    """Writes a section, rejects it once in review, then approves the revision."""

    def __init__(self, shrink=False):
        self.reviews = 0
        self.shrink = shrink

    def complete(self, **kw):
        if "Review to resolve" in kw["text"]:
            body = "short" if self.shrink else "long revised body with the scaling derivation " * 20
            return f"\\section{{Body}}\n{body}\n{SECTION_END}"
        return f"\\section{{Body}}\n{'original body ' * 20}\n{SECTION_END}"

    complete_with_tools = complete

    def complete_structured(self, **kw):
        self.reviews += 1
        if self.reviews == 1:
            return {"approved": False, "missing_items": [{"id": "I1.02", "what_is_missing": "steps"}], "issues": []}
        return {"approved": True, "missing_items": [], "issues": []}


def _write(tmp_path, client, rounds=2):
    outline = Outline(title="T", reader_background="b", notation=[],
                      sections=[_sec("Body", "core", ["I1.01", "I1.02"])])
    units = [SeedUnit(id="u1", kind="video", source_title="V", position="p", text="t")]
    cfg = Config(name="t", workdir=tmp_path, ground=False, review_rounds=rounds)
    jobs = section_jobs(outline, _items(), {"u1": "draft"}, units, {})
    return write_sections(client, jobs, {}, cfg)


def test_review_loop_revises_until_approved(tmp_path):
    client = WriterClient()
    [section] = _write(tmp_path, client)
    assert "revised" in section and client.reviews == 2
    assert SECTION_END not in section
    assert (tmp_path / "runs" / "t" / "coverage.json").exists()


def test_revision_that_drops_text_is_rejected(tmp_path):
    [section] = _write(tmp_path, WriterClient(shrink=True))
    assert "original body" in section and "short" not in section


class OmittingClient:
    """Outline that places I1.* and omits I2.01 with a reason."""

    def complete_structured(self, **kw):
        return {"title": "T", "storyline": "s", "reader_background": "b", "notation": [], "sections": [
            {"title": "Body", "role": "core", "purpose": "p", "plan": ["p"], "items": ["I1.01", "I1.02"],
             "figures": [], "topics": []}],
            "omitted_items": [{"id": "I2.01", "reason": "belongs to a topic the plan excludes"}]}


def test_items_can_be_omitted_with_a_reason(tmp_path):
    units = [SeedUnit(id=u, kind="video", source_title="V", position="p", text="t") for u in ("u1", "u2")]
    outline = build_outline(OmittingClient(), units, _items(), {}, [], Config(name="t", workdir=tmp_path, ground=False), 5000)
    assert [e["id"] for e in outline.omitted] == ["I2.01"]
    assert all("I2.01" not in s.items for s in outline.sections)  # omitted, not force-attached


def test_run_parallel_keeps_order_and_finishes_all_before_raising():
    import threading
    import time

    import pytest

    from vitex.utils import run_parallel

    done = []
    lock = threading.Lock()

    def work(i):
        time.sleep(0.01 * (5 - i))  # finish out of order
        if i == 1:
            raise ValueError("task 1 failed")
        with lock:
            done.append(i)
        return i * 10

    assert run_parallel(lambda i: i * 10, [3, 1, 2], workers=3) == [30, 10, 20]
    with pytest.raises(ValueError, match="task 1"):
        run_parallel(work, [0, 1, 2, 3, 4], workers=5)
    assert sorted(done) == [0, 2, 3, 4]  # the others still completed (and would be cached)


def test_stage_labels_are_per_thread():
    import threading

    from vitex.config import Config
    from vitex.llm.client import ClaudeClient

    client = ClaudeClient.__new__(ClaudeClient)
    client.cfg = Config()
    seen = {}

    def worker(name):
        with client.stage(name):
            threading.Event().wait(0.02)
            seen[name] = client._stage

    threads = [threading.Thread(target=worker, args=(f"s{i}",)) for i in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert seen == {f"s{i}": f"s{i}" for i in range(4)}


class ParallelWriterClient:
    def __init__(self):
        self.requests = []

    def complete(self, *, text, **kw):
        self.requests.append(text)
        title = next(line[7:] for line in text.splitlines() if line.startswith("Title: "))
        return f"\\section{{{title}}}\n{'body ' * 30}\n{SECTION_END}"

    complete_with_tools = complete

    def complete_structured(self, **kw):
        return {"approved": True, "missing_items": [], "issues": []}


def test_sections_are_written_in_parallel_with_neighbor_plans(tmp_path):
    outline = Outline(title="T", reader_background="b", notation=[], storyline="s", sections=[
        _sec("Intro", "introduction", []), _sec("Body", "core", ["I1.01", "I1.02"]), _sec("Summary", "summary", []),
    ])
    units = [SeedUnit(id="u1", kind="video", source_title="V", position="p", text="t")]
    cfg = Config(name="t", workdir=tmp_path, ground=False, review_rounds=1, parallel=3)
    client = ParallelWriterClient()
    jobs = section_jobs(outline, _items(), {"u1": "draft \\cite{vaswani2017attention}"}, units, {})
    written = write_sections(client, jobs, {}, cfg)

    assert [w.splitlines()[0] for w in written] == ["\\section{Intro}", "\\section{Body}", "\\section{Summary}"]
    body = next(r for r in client.requests if "This section:\nTitle: Body\n" in r)
    assert "Previous section (1):\nTitle: Intro" in body and "Next section (3):\nTitle: Summary" in body
    assert "vaswani2017attention" in body  # shared citation keys from the drafts
    # The map shows what every section covers, not just titles.
    assert "covers: Attention (concept); Scaling (derivation)" in body


def test_flow_rules_do_not_mandate_a_closing_question():
    from vitex.llm.prompts import TUTORIAL_STANDARD, review_system, section_system

    for text in (TUTORIAL_STANDARD, section_system(True), review_system(True)):
        assert "leads_to" not in text and "what problem remains" not in text
    assert "there are no templates" in TUTORIAL_STANDARD
    assert "transitions that feel forced" in review_system(True)


class EditorClient:
    """Editor flags a repetition in section 2 (related: 1); round 2 is clean."""

    def __init__(self, shrink=False):
        self.reads, self.fix_requests, self.shrink = 0, [], shrink

    def complete_structured(self, **kw):
        self.reads += 1
        if self.reads == 1:
            return {"assessment": "mostly good", "issues": [
                {"section": 2, "related_sections": [1], "kind": "repetition", "location": "Attention is...",
                 "problem": "re-explains attention", "fix": "refer back to section 1"},
                {"section": 99, "kind": "other", "location": "x", "problem": "bogus", "fix": "none"},
            ]}
        return {"assessment": "reads well", "issues": []}

    def complete(self, *, text, **kw):
        self.fix_requests.append(text)
        body = "x" if self.shrink else "Body, now referring back to Section 1. " * 10
        return f"\\section{{Body}}\n{body}\n{SECTION_END}"

    complete_with_tools = complete


def _edit(tmp_path, client):
    outline = Outline(title="T", reader_background="b", notation=[], storyline="s",
                      sections=[_sec("Intro", "introduction", ["I1.01"]), _sec("Body", "core", ["I1.02"])])
    units = [SeedUnit(id="u1", kind="video", source_title="V", position="p", text="t")]
    jobs = section_jobs(outline, _items(), {"u1": "draft"}, units, {})
    cfg = Config(name="t", workdir=tmp_path, ground=False, editorial_rounds=2)
    sections = ["\\section{Intro}\nAttention defined here.", "\\section{Body}\n" + "Attention is explained again. " * 10]
    return edit_tutorial(client, outline, _items(), sections, jobs, cfg), sections


def test_editor_reads_the_whole_and_fixes_the_flagged_sections(tmp_path):
    import json as _json

    client = EditorClient()
    edited, original = _edit(tmp_path, client)
    assert edited[0] == original[0]  # untouched
    assert "referring back to Section 1" in edited[1]
    assert client.reads == 2  # re-read after fixes; clean -> stop
    assert len(client.fix_requests) == 1
    fix = client.fix_requests[0]
    assert "re-explains attention" in fix and "SECTION 1: Intro" in fix  # related section text included
    report = _json.loads((tmp_path / "runs" / "t" / "editorial.json").read_text())
    assert report[0]["revised"] == [2] and report[1]["issues"] == []


def test_editorial_fix_that_guts_a_section_is_rejected(tmp_path):
    edited, original = _edit(tmp_path, EditorClient(shrink=True))
    assert edited[1] == original[1]


def test_fit_budget_assigns_missing_targets_and_scales_to_the_budget():
    from vitex.reconstruct.outline import fit_budget

    outline = Outline(title="T", reader_background="b", notation=[], sections=[
        _sec("A", "core", ["I1.01", "I1.02"]), _sec("B", "core", ["I2.01"]),
    ])
    outline.sections[0].target_words = 20000  # wildly over a 6000-word budget
    changes = fit_budget(outline, _items(), 6000)
    assert outline.sections[1].target_words > 0
    total = sum(s.target_words for s in outline.sections)
    assert abs(total - 6000) <= 0.15 * 6000 and any("scaled" in c for c in changes)


def test_length_checks_and_shrink_guard():
    from vitex.reconstruct.writer import mechanical_checks, min_kept_words

    sec = _sec("A", "core", [])
    sec.target_words = 1000
    long_text = "\\section{A}\n" + "word " * 2000
    assert any("over its target" in c for c in mechanical_checks(long_text, sec, set()))
    assert any("far below" in c for c in mechanical_checks("\\section{A} short", sec, set()))
    # Normal revisions may not cut more than 15%; an over-long section may be cut toward its target.
    assert min_kept_words(1000, 1000) == 850
    assert min_kept_words(2000, 1000) == 700  # over budget: may be cut down to 70% of the target


def test_allocation_follows_plan_pages_and_coverage_is_enforced():
    from vitex.reconstruct.outline import allocate, topic_problems

    entries = [{"name": "PPO", "level": "main", "pages": 5}, {"name": "TRPO", "level": "main", "pages": 3},
               {"name": "A2C", "level": "stepping", "pages": 1}, {"name": "Noisy Nets", "level": "mention", "pages": 0},
               {"name": "C51", "level": "excluded", "pages": 0}]
    outline = Outline(title="T", reader_background="b", notation=[], sections=[
        _sec("Intro", "introduction", []), _sec("PPO", "core", []), _sec("Variants", "supporting", []),
        _sec("Summary", "summary", []),
    ])
    outline.sections[1].topics = ["ppo", "TRPO"]  # names match case-insensitively
    outline.sections[2].topics = ["A2C", "Noisy Nets", "Made Up"]
    assert topic_problems(outline, entries) == ([], ["Made Up"])
    outline.sections[1].topics = ["PPO"]
    assert topic_problems(outline, entries)[0] == ["TRPO"]  # uncovered main entry -> outline retried
    outline.sections[1].topics = ["PPO", "TRPO"]

    lines = allocate(outline, entries, 10000, frame_words=1000)
    intro, main, variants, summary = (s.target_words for s in outline.sections)
    assert (intro, summary) == (550, 450)  # the frame is split 55/45
    assert abs(main / variants - 8 / 1.15) < 0.2  # 8 plan pages vs 1 page + a mention
    assert abs(sum((intro, main, variants, summary)) - 10000) <= 2
    assert any("Allocation of 10000 words" in line for line in lines)


def test_over_long_first_version_is_condensed_before_review(tmp_path):
    from vitex.reconstruct.writer import effective_words

    class CondenseClient:
        def __init__(self):
            self.condense_requests = 0

        def complete(self, *, text, **kw):
            if "by REMOVING whole pieces" in text:
                self.condense_requests += 1
                return f"\\section{{Body}}\n{'tight ' * 450}\n{SECTION_END}"
            return f"\\section{{Body}}\n{'wordy ' * 1500}\n{SECTION_END}"

        complete_with_tools = complete

        def complete_structured(self, **kw):
            return {"approved": True, "missing_items": [], "issues": []}

    outline = Outline(title="T", reader_background="b", notation=[], sections=[_sec("Body", "core", ["I1.01"])])
    outline.sections[0].target_words = 500
    units = [SeedUnit(id="u1", kind="video", source_title="V", position="p", text="t")]
    jobs = section_jobs(outline, _items(), {"u1": "draft"}, units, {})
    client = CondenseClient()
    [section] = write_sections(client, jobs, {}, Config(name="t", workdir=tmp_path, ground=False, review_rounds=1))
    assert client.condense_requests == 1 and "tight" in section and "wordy" not in section
    assert effective_words("\\begin{figure}x\\end{figure} a b") == 150 + 3


def test_review_coverage_is_by_plan_entry_and_notes_are_material():
    from vitex.llm.prompts import REVIEW_SCHEMA, section_system

    assert "missing" in REVIEW_SCHEMA["properties"] and "missing_items" not in REVIEW_SCHEMA["properties"]
    outline = Outline(title="T", reader_background="b", notation=[], sections=[_sec("PPO", "core", ["I1.01"])])
    outline.sections[0].topics = ["PPO"]
    units = [SeedUnit(id="u1", kind="video", source_title="V", position="p", text="t")]
    [job] = section_jobs(outline, _items(), {"u1": "draft"}, units, {}, {"PPO": "main"})
    assert "Plan entries covered here: PPO (main)" in job.spec
    assert "I1.01" not in job.request  # item ids are routing, not requirements shown to the writer
    assert "use what serves the plan" in job.request
    assert "you do not have to use all of it" in section_system(True)


def test_cli_parses_section_lists():
    from vitex.cli import _section_list

    assert _section_list("2,5") == [2, 5] and _section_list("1,3-5") == [1, 3, 4, 5]
