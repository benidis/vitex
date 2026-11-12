"""Tests for the client's tool-loop budget handling, retry policy, and caching.

All network calls are replaced by scripted fakes.
"""

from types import SimpleNamespace

import anthropic
import pytest

from vitex.config import Config
from vitex.errors import LLMError
from vitex.llm import client as client_mod
from vitex.llm.client import ClaudeClient, _with_cache_breakpoints

try:  # the SDK ships its HTTP layer under one of these names
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx


def _bare_client(**cfg) -> ClaudeClient:
    client = ClaudeClient.__new__(ClaudeClient)
    client.cfg = Config(**cfg)
    client.model = "m"
    return client


def _text(text, stop="end_turn"):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason=stop)


def _tool_call(i):
    tu = SimpleNamespace(type="tool_use", id=f"tu_{i}", name="arxiv_search", input={"query": f"q{i}"})
    return SimpleNamespace(content=[tu], stop_reason="tool_use")


def _status_error(cls, status):
    resp = httpx.Response(status, request=httpx.Request("POST", "https://example.invalid"))
    return cls("boom", response=resp, body=None)


def test_budget_exhaustion_instructs_model_and_returns_final_text():
    client = _bare_client(research_max_iterations=2)
    calls = []

    def fake_create(*, model, max_tokens, system, messages, tools=None, tool_choice=None):
        calls.append({"messages": list(messages), "tool_choice": tool_choice, "system": system})
        return _text("\\section{Final}") if tool_choice == {"type": "none"} else _tool_call(len(calls))

    client._create = fake_create
    out = client.complete_with_tools(
        system="s", text="u", tools=[{"name": "arxiv_search"}], dispatch=lambda n, i: "r"
    )
    assert out == "\\section{Final}"
    assert len(calls) == 3  # two tool rounds, then one tool-free answer
    assert calls[-1]["tool_choice"] == {"type": "none"}  # tools stay declared, answer in text
    last_user = calls[-1]["messages"][-1]["content"]
    assert last_user[-1]["type"] == "text" and "budget exhausted" in last_user[-1]["text"]
    assert "at most 2 rounds" in calls[0]["system"]


def test_tool_loop_continues_a_truncated_final_answer():
    client = _bare_client(research_max_iterations=3, max_continuations=3)
    scripted = [_tool_call(1), _text("line1\npart", "max_tokens"), _text("line2\n")]

    def fake_create(*, model, max_tokens, system, messages, **kw):
        return scripted.pop(0)

    client._create = fake_create
    out = client.complete_with_tools(
        system="s", text="u", tools=[{"name": "t"}], dispatch=lambda n, i: "r"
    )
    assert out == "line1\nline2\n"


def test_cache_breakpoint_is_fixed_on_the_first_turn_without_mutating():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "c"}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "r"}]},
        {"role": "assistant", "content": "prefill"},
    ]
    marked = _with_cache_breakpoints(messages)
    assert "cache_control" in marked[0]["content"][-1]
    assert "cache_control" not in marked[0]["content"][0]
    # Later turns are untouched, so history never changes between requests.
    assert marked[1:] == messages[1:]
    assert all("cache_control" not in b for m in messages[:3] for b in m["content"])


class _Stream:
    def __init__(self, create, kw):
        self.create, self.kw = create, kw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.create(**self.kw)


def _client_with_sdk(create, **cfg):
    """Fake SDK whose streaming call runs ``create`` (raising or returning)."""
    client = _bare_client(**cfg)
    client._client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kw: _Stream(create, kw)))
    return client


def test_retries_transient_errors_then_succeeds(monkeypatch):
    monkeypatch.setattr(client_mod.time, "sleep", lambda s: None)
    attempts = []

    def create(**kw):
        attempts.append(kw)
        if len(attempts) < 3:
            raise _status_error(anthropic.RateLimitError, 429)
        return _text("ok")

    client = _client_with_sdk(create, llm_retries=5)
    resp = client._create(model="m", max_tokens=10, system="s", messages=[{"role": "user", "content": "u"}])
    assert resp.content[0].text == "ok"
    assert len(attempts) == 3
    # Prompt caching wraps the system prompt in a cacheable block.
    assert attempts[0]["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_non_retryable_error_fails_immediately(monkeypatch):
    monkeypatch.setattr(client_mod.time, "sleep", lambda s: None)
    attempts = []

    def create(**kw):
        attempts.append(kw)
        raise _status_error(anthropic.BadRequestError, 400)

    client = _client_with_sdk(create, llm_retries=5)
    with pytest.raises(LLMError, match="not retryable"):
        client._create(model="m", max_tokens=10, system="s", messages=[{"role": "user", "content": "u"}])
    assert len(attempts) == 1


def test_prompt_caching_can_be_disabled():
    seen = {}

    def create(**kw):
        seen.update(kw)
        return _text("ok")

    client = _client_with_sdk(create, prompt_caching=False, effort=None)
    client._create(model="m", max_tokens=10, system="s", messages=[{"role": "user", "content": [{"type": "text", "text": "u"}]}])
    assert seen["system"] == "s"
    assert "cache_control" not in seen["messages"][0]["content"][0]


def test_effort_is_sent_for_the_main_model_only():
    seen = []
    client = _client_with_sdk(lambda **kw: seen.append(kw) or _text("ok"), effort="high", model="main")
    msgs = [{"role": "user", "content": "u"}]
    client._create(model="main", max_tokens=10, system="s", messages=msgs)
    client._create(model="haiku", max_tokens=10, system="s", messages=msgs)
    assert seen[0]["output_config"] == {"effort": "high"} and "output_config" not in seen[1]
