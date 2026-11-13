"""Tests for the research tools' pure parsing helpers (no network)."""

from vitex.llm import research


def test_strip_html_unescapes_and_removes_tags():
    assert research._strip_html("<b>PPO</b> &amp; TRPO") == "PPO & TRPO"


def test_unwrap_ddg_redirect():
    wrapped = "//duckduckgo.com/l/?uddg=https%3A%2F%2Farxiv.org%2Fabs%2F1707.06347&rut=x"
    assert research._unwrap_ddg(wrapped) == "https://arxiv.org/abs/1707.06347"


def test_unwrap_ddg_passthrough_and_scheme():
    assert research._unwrap_ddg("https://example.com/x") == "https://example.com/x"
    assert research._unwrap_ddg("//example.com/x") == "https://example.com/x"


def test_dispatch_unknown_tool_is_non_fatal():
    assert "Unknown tool" in research.dispatch("nope", {})


def test_dispatch_bad_arguments_is_non_fatal():
    # arxiv_search requires `query`; a missing arg must return a string, not raise.
    out = research.dispatch("arxiv_search", {})
    assert isinstance(out, str) and "Bad arguments" in out


def test_tool_specs_cover_dispatch_table():
    spec_names = {t["name"] for t in research.TOOL_SPECS}
    assert spec_names == set(research._DISPATCH)


def test_paper_search_parses_openalex_with_quality_signals(monkeypatch):
    import json

    payload = {"results": [{
        "display_name": "ZeRO: Memory optimizations Toward Training Trillion Parameter Models",
        "publication_year": 2020, "cited_by_count": 1433, "doi": "https://doi.org/10.1109/sc41405.2020.00024",
        "primary_location": {"source": {"display_name": "SC20"}},
        "locations": [{"landing_page_url": "https://arxiv.org/abs/1910.02054v3"}],
        "authorships": [{"author": {"display_name": f"A{i}"}} for i in range(12)],
        "abstract_inverted_index": {"Large": [0], "models": [1]},
    }]}
    monkeypatch.setattr(research, "_get", lambda url, **kw: json.dumps(payload))
    [hit] = json.loads(research.paper_search("zero"))
    assert (hit["venue"], hit["citations"], hit["arxiv_id"]) == ("SC20", 1433, "1910.02054")
    assert hit["abstract"] == "Large models" and len(hit["authors"]) == 9  # 8 + "... (12 authors)"
    assert "paper_search" in research._DISPATCH and research.TOOL_SPECS[0]["name"] == "paper_search"
