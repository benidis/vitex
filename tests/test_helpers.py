from vitex.config import Config
from vitex.llm.prompts import draft_system, draft_user
from vitex.models import _fmt_timestamp
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


def test_draft_prompts_carry_the_tutorial_standard_and_seed_rules():
    system = draft_system(ground=True)
    assert "not a summary" in system and "SEEDS" in system and "arxiv_search" in system
    assert "arxiv_search" not in draft_system(ground=False)
    user = draft_user(kind="document", source_title="Notes", position="part 1 of 2",
                      text="BODY", figure_lines="", visual_notes="", target_words=1200)
    assert "Keep ALL of its content" in user and "BODY" in user and "(none)" in user
    assert "about 1200 words" in user


def test_config_validate_rejects_bad_provider():
    cfg = Config(provider="nope")
    try:
        cfg.validate()
    except ValueError as e:
        assert "provider" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_authors_brief_reaches_every_writing_prompt():
    from vitex.llm.prompts import outline_system, review_system, revise_system, section_system

    brief = "Focus on policy-gradient derivations."
    from vitex.llm.prompts import plan_system

    for build in (draft_system, outline_system, section_system, review_system, revise_system):
        assert brief in build(True, brief)
        assert "AUTHOR'S BRIEF" not in build(True, None)
    assert brief in plan_system(brief)
    text = section_system(True)
    flat = " ".join(text.split())
    assert "Depth follows the idea's LEVEL" in flat and "do not prove them unless" in flat
    assert "Length is never a concern" not in text


def test_cli_title_name_and_guide_file(tmp_path, monkeypatch):
    from vitex.cli import _build_parser, _config_from_args

    monkeypatch.setattr("vitex.config.load_dotenv", lambda: None)  # ignore the real .env
    monkeypatch.setenv("VITEX_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")

    brief = tmp_path / "brief.txt"
    brief.write_text("Modern RL only.")
    doc = tmp_path / "old.tex"
    doc.write_text("x")
    args = _build_parser().parse_args(
        ["run", "--doc", str(doc), "--title", "Modern RL: A Tutorial", "--name", "modern_rl",
         "--guide-file", str(brief)]
    )
    cfg = _config_from_args(args)
    assert cfg.sources == [] and cfg.docs == [doc]
    assert (cfg.title, cfg.name, cfg.guide) == ("Modern RL: A Tutorial", "modern_rl", "Modern RL only.")
    cfg.validate()  # documents only is a valid run


def test_aws_profile_is_required_and_never_inferred(monkeypatch):
    import pytest

    monkeypatch.setattr("vitex.config.load_dotenv", lambda: None)  # ignore the real .env
    monkeypatch.setenv("AWS_PROFILE", "aws-devtools-sci")  # must NOT be picked up
    monkeypatch.delenv("VITEX_AWS_PROFILE", raising=False)
    monkeypatch.setenv("VITEX_PROVIDER", "bedrock")
    cfg = Config.from_env()
    cfg.sources = ["video.mp4"]
    assert cfg.aws_profile is None
    with pytest.raises(ValueError, match="VITEX_AWS_PROFILE is not set"):
        cfg.validate()

    cfg.aws_profile = "no-such-profile-xyz"
    with pytest.raises(ValueError, match="not a profile"):
        cfg.validate()

    # The Anthropic API needs no AWS profile (with the local whisper transcriber).
    api = Config(provider="anthropic", anthropic_api_key="k", sources=["video.mp4"])
    api.validate()
    api.transcriber, api.transcribe_bucket = "aws", "bucket"
    with pytest.raises(ValueError, match="VITEX_AWS_PROFILE"):
        api.validate()
