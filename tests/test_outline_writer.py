"""Tests for the outline's no-drop guarantee and the section review loop."""

from vitex.config import Config
from vitex.models import SeedUnit
from vitex.reconstruct.outline import Item, Outline, Section, build_outline, outline_problems, repair_outline
from vitex.reconstruct.writer import write_sections

SECTION_END = "% END OF SECTION"


def _items():
    return [
        Item(id="I1.01", unit_id="u1", kind="concept", title="Attention", detail="softmax(QK^T)V", figures=["f1.jpg"]),
        Item(id="I1.02", unit_id="u1", kind="derivation", title="Scaling", detail="divide by sqrt(d)"),
        Item(id="I2.01", unit_id="u2", kind="result", title="KV cache", detail="memory O(nd)"),
    ]


def _sec(title, role, items, figures=()):
    return Section(title=title, role=role, question="q", motivation="m", plan=["p"], items=list(items),
                   figures=list(figures), leads_to="l")


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
    outline = build_outline(client, units, _items(), {}, [], Config(workdir=tmp_path, ground=False))
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
    cfg = Config(workdir=tmp_path, ground=False, review_rounds=rounds)
    return write_sections(client, outline, _items(), {"u1": "draft"}, units, {}, cfg)


def test_review_loop_revises_until_approved(tmp_path):
    client = WriterClient()
    [section] = _write(tmp_path, client)
    assert "revised" in section and client.reviews == 2
    assert SECTION_END not in section
    assert (tmp_path / "tutorial" / "coverage.json").exists()


def test_revision_that_drops_text_is_rejected(tmp_path):
    [section] = _write(tmp_path, WriterClient(shrink=True))
    assert "original body" in section and "short" not in section


class ExcludingClient:
    """Outline that leaves I2.01 out, citing the author's brief."""

    def complete_structured(self, **kw):
        return {"title": "T", "storyline": "s", "reader_background": "b", "notation": [], "sections": [
            {"title": "Body", "role": "core", "question": "q", "motivation": "m", "plan": ["p"],
             "items": ["I1.01", "I1.02"], "figures": [], "leads_to": "l"}],
            "excluded_items": [{"id": "I2.01", "reason": "brief excludes caching"}]}


def _outline_with(tmp_path, guide):
    units = [SeedUnit(id=u, kind="video", source_title="V", position="p", text="t") for u in ("u1", "u2")]
    cfg = Config(workdir=tmp_path, ground=False, guide=guide)
    return build_outline(ExcludingClient(), units, _items(), {}, [], cfg)


def test_exclusion_is_honored_only_with_an_authors_brief(tmp_path):
    with_brief = _outline_with(tmp_path / "a", "Skip KV caching entirely.")
    assert [e["id"] for e in with_brief.excluded] == ["I2.01"]
    assert "I2.01" not in with_brief.sections[0].items
    assert with_brief.storyline == "s" and "Storyline: s" in with_brief.overview()

    without = _outline_with(tmp_path / "b", None)
    assert without.excluded == []
    assert "I2.01" in without.sections[0].items  # restored and attached in code
