"""Tests for draft validation and input-keyed caching in the drafting passes."""

import pytest

from vitex.config import Config
from vitex.errors import LLMError
from vitex.models import SeedUnit
from vitex.reconstruct.chapters import build_bibliography, draft_seeds

END = "% END OF DRAFT"


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def complete(self, **kw):
        self.calls += 1
        return self.replies.pop(0)

    complete_with_tools = complete


def _unit():
    return SeedUnit(id="u1", kind="video", source_title="V", position="part 1 of 1", text="[00:00:00] hello")


def _cfg(tmp_path, **kw):
    return Config(workdir=tmp_path, ground=False, **kw)


def test_empty_draft_is_retried_and_sentinel_stripped(tmp_path):
    client = FakeClient(["", f"\\section{{A}}\nbody\n{END}"])
    drafts = draft_seeds(client, [_unit()], _cfg(tmp_path))
    assert drafts["u1"] == "\\section{A}\nbody\n"
    assert client.calls == 2


def test_truncated_draft_without_sentinel_is_rejected(tmp_path):
    client = FakeClient(["\\section{A}\ncut mid-sente", "\\section{A}\nstill cut", "\\section{A}\nagain"])
    with pytest.raises(LLMError, match="truncated"):
        draft_seeds(client, [_unit()], _cfg(tmp_path))
    assert not (tmp_path / "drafts" / "u1.tex").exists()


def test_cached_draft_reused_only_for_identical_inputs(tmp_path):
    draft_seeds(FakeClient([f"\\section{{A}}\n{END}"]), [_unit()], _cfg(tmp_path))

    reused = FakeClient([])
    assert draft_seeds(reused, [_unit()], _cfg(tmp_path))["u1"].startswith("\\section{A}")
    assert reused.calls == 0

    # A different model must not reuse the old draft.
    other = FakeClient([f"\\section{{B}}\n{END}"])
    assert draft_seeds(other, [_unit()], _cfg(tmp_path, model="x"))["u1"].startswith("\\section{B}")


def test_bibliography_fills_missing_keys(tmp_path):
    bib = "\\begin{thebibliography}{99}\n\\bibitem{a} A.\n\\end{thebibliography}"
    out = build_bibliography(FakeClient([bib]), "\\cite{a,b}", "T", ["a", "b"], _cfg(tmp_path))
    assert "\\bibitem{a} A." in out
    assert "\\bibitem{b}" in out
    assert out.rstrip().endswith("\\end{thebibliography}")


def test_bibliography_is_resolved_in_batches_and_merged(tmp_path):
    keys = [f"k{i}" for i in range(30)]

    class BibClient:
        calls = 0

        def complete(self, *, text, **kw):
            BibClient.calls += 1
            asked = [k for k in keys if f"  - {k}\n" in text + "\n"]
            items = "\n".join(f"\\bibitem{{{k}}} Ref {k}." for k in asked)
            return f"\\begin{{thebibliography}}{{99}}\n{items}\n\\end{{thebibliography}}"

        complete_with_tools = complete

    doc = " ".join(f"\\cite{{{k}}}" for k in keys)
    out = build_bibliography(BibClient(), doc, "T", keys, _cfg(tmp_path))
    assert BibClient.calls == 3  # 30 keys in batches of 12
    assert out.count("\\bibitem{") == 30 and out.count("\\begin{thebibliography}") == 1
    assert "could not be resolved" not in out
