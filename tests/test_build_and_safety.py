"""Tests for LaTeX repair edits, figure/citation helpers, URL vetting, and
silence-based transcript filtering. No network, no TeX install required."""

import pytest

from vitex.llm import research
from vitex.models import Segment
from vitex.output.build import apply_edits, error_excerpts
from vitex.output.latex import cite_contexts, complete_bibliography, drop_unknown_figures
from vitex.transcribe.whisper_local import collapse_repeats, drop_silent, parse_silences

# --- repair edits -----------------------------------------------------------


def test_apply_edits_replaces_unique_matches():
    doc = "a\n\\foo{x}\nb\n"
    reply = "<<<<<<< SEARCH\n\\foo{x}\n=======\n\\textbf{x}\n>>>>>>> REPLACE"
    assert apply_edits(doc, reply) == "a\n\\textbf{x}\nb\n"


def test_apply_edits_supports_deletion():
    doc = "keep\n\\bad\nkeep2\n"
    reply = "<<<<<<< SEARCH\n\\bad\n\n=======\n>>>>>>> REPLACE"
    assert apply_edits(doc, reply) == "keep\nkeep2\n"


def test_apply_edits_rejects_ambiguous_or_missing_search():
    assert apply_edits("x\nx\n", "<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE") is None
    assert apply_edits("abc", "<<<<<<< SEARCH\nzzz\n=======\ny\n>>>>>>> REPLACE") is None
    assert apply_edits("abc", "no edit blocks here") is None


def test_error_excerpts_include_source_lines():
    tex_log = "junk\n! Undefined control sequence.\n<recently read> \\foo\nl.3 \\foo\nmore"
    doc = "line1\nline2\n\\foo\nline4"
    out = error_excerpts(tex_log, doc)
    assert "! Undefined control sequence." in out
    assert "     3| \\foo" in out


# --- figures and citations --------------------------------------------------


def test_drop_unknown_figures_removes_whole_environment():
    doc = (
        "A\n\\begin{figure}[ht]\\centering\n\\includegraphics{frame_00001.jpg}\n"
        "\\caption{ok}\\end{figure}\n"
        "\\begin{figure}[ht]\\includegraphics[width=1cm]{frame_99999.jpg}\\end{figure}\nB"
    )
    out, removed = drop_unknown_figures(doc, {"frame_00001.jpg"})
    assert "frame_00001.jpg" in out
    assert "frame_99999" not in out
    assert removed == ["frame_99999.jpg"]
    assert out.endswith("B")


def test_cite_contexts_cover_late_citations():
    doc = "x" * 50000 + " PPO clipping \\cite{schulman2017ppo} objective"
    ctx = cite_contexts(doc, ["schulman2017ppo"], width=30)
    assert "PPO clipping" in ctx["schulman2017ppo"][0]


def test_complete_bibliography_is_noop_when_all_keys_present():
    bib = "\\begin{thebibliography}{99}\n\\bibitem[X]{a} A.\n\\end{thebibliography}"
    assert complete_bibliography(bib, ["a"]) == (bib, [])


# --- URL vetting --------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "http://127.0.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/",
        "http://[::1]/",
        "http://[::ffff:169.254.169.254]/",
        "http://localhost:8080/",
    ],
)
def test_check_public_url_rejects_unsafe_targets(url):
    with pytest.raises(research.UnsafeURLError):
        research.check_public_url(url)


def test_check_public_url_accepts_public_ip():
    research.check_public_url("https://8.8.8.8/")


def test_fetch_url_reports_refusal_instead_of_fetching():
    out = research.fetch_url("file:///etc/passwd")
    assert out.startswith("fetch_url failed") and "http(s)" in out


# --- silence filtering ----------------------------------------------------------


def test_parse_silences_pairs_start_and_end():
    log = (
        "[silencedetect @ 0x1] silence_start: 1.5\n"
        "[silencedetect @ 0x1] silence_end: 4.0 | silence_duration: 2.5\n"
        "[silencedetect @ 0x1] silence_start: 10\n"
    )
    assert parse_silences(log) == [(1.5, 4.0), (10.0, float("inf"))]


def test_drop_silent_and_collapse_repeats():
    segs = [Segment(0, 1, "real"), Segment(2, 3.5, "Thank you."), Segment(5, 6, "talk")]
    assert [s.text for s in drop_silent(segs, [(1.5, 4.0)])] == ["real", "talk"]

    loop = [Segment(i, i + 1, "Thank you.") for i in range(5)] + [Segment(9, 10, "next")]
    assert [s.text for s in collapse_repeats(loop)] == ["Thank you.", "Thank you.", "next"]
