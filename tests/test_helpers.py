from vitex.config import Config
from vitex.llm.prompts import chapter_user
from vitex.models import Chunk, Segment, _fmt_timestamp
from vitex.output.latex import _tex_escape, build_preamble
from vitex.reconstruct.chapters import strip_fences
from vitex.utils import safe_stem


def test_fmt_timestamp():
    assert _fmt_timestamp(0) == "00:00:00"
    assert _fmt_timestamp(65) == "00:01:05"
    assert _fmt_timestamp(3661) == "01:01:01"


def test_safe_stem():
    assert safe_stem("Hello, World! (2024)") == "Hello_World_2024"
    assert safe_stem("") == "tutorial"
    assert safe_stem("///") == "tutorial"


def test_strip_fences_removes_latex_fence():
    body = "\\section{X}\ntext"
    assert strip_fences(f"```latex\n{body}\n```") == body
    assert strip_fences(body) == body


def test_tex_escape_and_preamble_title():
    assert _tex_escape("a_b & c%") == r"a\_b \& c\%"
    pre = build_preamble("Cost & Value_Model")
    assert r"\title{Cost \& Value\_Model}" in pre
    assert r"\newtheorem{definition}" in pre and r"\graphicspath{{figures/}}" in pre


def test_chapter_user_lists_frames_or_none():
    seg = Segment(0, 5, "hello")
    empty = Chunk(index=0, start=0, end=5, segments=[seg], frames=[])
    assert "No frames are available" in chapter_user(empty, "T", 1, 1)


def test_config_validate_rejects_bad_provider():
    cfg = Config(provider="nope")
    try:
        cfg.validate()
    except ValueError as e:
        assert "provider" in str(e)
    else:
        raise AssertionError("expected ValueError")
