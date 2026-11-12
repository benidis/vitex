"""Tests for ClaudeClient.complete()'s continuation loop.

A single Messages response is capped at max_tokens; long LaTeX documents exceed
one response and must be continued rather than silently truncated. These tests
use a scripted fake in place of the real SDK client so no network call happens.
"""

from types import SimpleNamespace

from vitex.config import Config
from vitex.llm.client import ClaudeClient


def _make_client(scripted, capture=None):
    """Build a ClaudeClient whose _create replays ``scripted`` (text, stop_reason)."""
    client = ClaudeClient.__new__(ClaudeClient)
    client.cfg = Config(max_continuations=8)
    client.model = "test-model"
    state = {"n": 0}

    def fake_create(*, model, max_tokens, system, messages, **kw):
        if capture is not None and len(messages) > 1:
            capture.append(messages[-1]["content"][0]["text"])
        text, stop = scripted[state["n"]]
        state["n"] += 1
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)], stop_reason=stop
        )

    client._create = fake_create
    return client


def test_single_response_returned_verbatim():
    client = _make_client([("\\section{A}\nDone.", "end_turn")])
    assert client.complete(system="s", text="u") == "\\section{A}\nDone."


def test_continuation_concatenates_complete_lines():
    # Each truncated response ends on a line boundary, so nothing is dropped and
    # the pieces concatenate exactly. (Mid-line cuts are covered by the seam test,
    # where the model re-emits the dropped partial line.)
    scripted = [
        ("\\section{A}\nline1\nline2\n", "max_tokens"),
        ("line3\nline4\n", "max_tokens"),
        ("\\end{document}\n", "end_turn"),
    ]
    out = _make_client(scripted).complete(system="s", text="u")
    assert out == "\\section{A}\nline1\nline2\nline3\nline4\n\\end{document}\n"


def test_seam_never_splits_a_command_mid_token():
    # Response 1 is cut inside "\linewidth" (right after "\line"). A naive prefill
    # would let the model resume as if "\line" were complete, yielding the invalid
    # "\line width". Retreating to the last complete line must avoid that.
    scripted = [
        ("\\begin{figure}\n\\centering\n\\includegraphics[width=0.8\\line", "max_tokens"),
        ("\\includegraphics[width=0.8\\linewidth]{f.jpg}\n\\end{figure}\n", "end_turn"),
    ]
    captured: list[str] = []
    out = _make_client(scripted, capture=captured).complete(system="s", text="u")

    assert "\\line width" not in out  # the corruption we are guarding against
    assert "\\linewidth]{f.jpg}" in out
    assert out.count("\\includegraphics") == 1  # partial line dropped, not duplicated
    # The continuation is a new user turn (no assistant prefill) quoting the text
    # up to the last complete line only.
    assert len(captured) == 1
    assert "<<<PARTIAL\n\\begin{figure}\n\\centering\nPARTIAL>>>" in captured[0]


def test_tool_loop_executes_tools_then_returns_final_text():
    tool_use = SimpleNamespace(
        type="tool_use", id="tu_1", name="arxiv_search", input={"query": "ppo"}
    )
    scripted = [
        SimpleNamespace(content=[tool_use], stop_reason="tool_use"),
        SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Grounded answer \\cite{x}.")],
            stop_reason="end_turn",
        ),
    ]
    client = ClaudeClient.__new__(ClaudeClient)
    client.cfg = Config(research_max_iterations=5)
    client.model = "m"
    seen = {"calls": [], "tools_arg": []}

    def fake_create(*, model, max_tokens, system, messages, tools=None, **kw):
        seen["tools_arg"].append(tools)
        return scripted[len(seen["calls"])] if len(seen["calls"]) < len(scripted) else scripted[-1]

    def fake_create_wrapped(**kw):
        r = fake_create(**kw)
        seen["calls"].append(kw["messages"])
        return r

    client._create = fake_create_wrapped
    dispatched = []

    def dispatch(name, inp):
        dispatched.append((name, inp))
        return "arxiv result text"

    out = client.complete_with_tools(
        system="s", text="u", tools=[{"name": "arxiv_search"}], dispatch=dispatch
    )
    assert out == "Grounded answer \\cite{x}."
    assert dispatched == [("arxiv_search", {"query": "ppo"})]
    # The follow-up turn must carry a tool_result for the tool call.
    followup = seen["calls"][1]
    assert followup[-1]["role"] == "user"
    assert followup[-1]["content"][0]["type"] == "tool_result"
    assert followup[-1]["content"][0]["tool_use_id"] == "tu_1"


def test_stops_at_max_continuations_without_infinite_loop():
    cfg = Config(max_continuations=2)
    # Always claims more is coming; the loop must still terminate.
    client = ClaudeClient.__new__(ClaudeClient)
    client.cfg = cfg
    client.model = "m"
    calls = {"n": 0}

    def always_truncated(*, model, max_tokens, system, messages, **kw):
        calls["n"] += 1
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=f"line{calls['n']}\n")],
            stop_reason="max_tokens",
        )

    client._create = always_truncated
    out = client.complete(system="s", text="u")
    # 1 initial + max_continuations attempts, then it gives up.
    assert calls["n"] == cfg.max_continuations + 1
    assert "line1" in out
