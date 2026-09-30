"""A thin Claude client that works over Amazon Bedrock or the Anthropic API.

Both backends speak the same Messages API (text + base64 image blocks), so the
only difference is which SDK client we construct. Images are downscaled and
re-encoded as JPEG before sending, which keeps token cost and payload size in
check without hurting slide legibility.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image

from vitex.config import Config
from vitex.errors import LLMError

log = logging.getLogger("vitex")


def _block_to_dict(block) -> dict:
    """Convert an SDK response content block back into a request-shaped dict."""
    kind = getattr(block, "type", None)
    if kind == "text":
        return {"type": "text", "text": block.text}
    if kind == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    # Fall back to whatever the SDK exposes; unknown blocks are passed through.
    return block.model_dump() if hasattr(block, "model_dump") else {"type": kind}


@dataclass
class ImageRef:
    """An image to embed in a request, with the label the model should cite."""

    path: Path
    label: str
    caption_hint: str = ""


class ClaudeClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = cfg.model
        self._client = self._build_client(cfg)

    @staticmethod
    def _build_client(cfg: Config):
        if cfg.provider == "bedrock":
            from anthropic import AnthropicBedrock

            return AnthropicBedrock(aws_region=cfg.aws_region, aws_profile=cfg.aws_profile)
        from anthropic import Anthropic

        return Anthropic(api_key=cfg.anthropic_api_key)

    # -- request building ---------------------------------------------------

    def _encode_image(self, path: Path) -> dict:
        with Image.open(path) as im:
            im = im.convert("RGB")
            if im.width > self.cfg.image_max_width:
                ratio = self.cfg.image_max_width / im.width
                im = im.resize((self.cfg.image_max_width, int(im.height * ratio)))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=80)
        data = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": data},
        }

    def _build_blocks(self, text: str, images: list[ImageRef]) -> list[dict]:
        blocks: list[dict] = [{"type": "text", "text": text}]
        for img in images:
            # Precede each image with a text marker so the model can correlate
            # the picture it is about to see with the label it must reference.
            marker = f"[FIGURE {img.label}]"
            if img.caption_hint:
                marker += f" (around {img.caption_hint})"
            blocks.append({"type": "text", "text": marker})
            blocks.append(self._encode_image(img.path))
        return blocks

    # -- public API ---------------------------------------------------------

    def complete(
        self,
        *,
        system: str,
        text: str,
        images: list[ImageRef] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send one user turn and return the full text response.

        A single Messages response is capped at ``max_tokens`` and stops with
        reason ``"max_tokens"`` if the model wanted to write more. A long LaTeX
        document routinely exceeds one response, so when that happens we prefill
        the partial text back as an assistant turn and let the model continue,
        repeating until it finishes (or we hit ``max_continuations``). Without
        this, the document is silently truncated mid-command and never compiles.
        """
        blocks = self._build_blocks(text, images or [])
        model = model or self.model
        max_tokens = max_tokens or self.cfg.max_output_tokens

        messages: list[dict] = [{"role": "user", "content": blocks}]
        full = ""
        for cont in range(self.cfg.max_continuations + 1):
            resp = self._create(model=model, max_tokens=max_tokens, system=system, messages=messages)
            piece = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            if getattr(resp, "stop_reason", None) != "max_tokens":
                return full + piece
            if cont == self.cfg.max_continuations:
                log.warning(
                    "Output still truncated after %d continuations; returning partial text.",
                    self.cfg.max_continuations,
                )
                return full + piece
            # The cut can land mid-token (e.g. "\\linewidth" -> "\\line"), and the
            # model would resume treating the stump as a finished command. So we
            # keep only through the last complete line and let the model rewrite
            # the partial final line. The API also rejects a prefill ending in
            # whitespace, so we strip that for the request (the newline it removes
            # is re-added by the model continuing onto a fresh line).
            candidate = full + piece
            cut = candidate.rfind("\n")
            full = candidate[: cut + 1] if cut != -1 else candidate
            prefill = full.rstrip()
            if not prefill:  # no usable boundary yet; avoid an infinite loop
                return candidate
            log.info("Response hit max_tokens; continuing (%d)...", cont + 1)
            messages = [
                {"role": "user", "content": blocks},
                {"role": "assistant", "content": prefill},
            ]
        return full

    def complete_with_tools(
        self,
        *,
        system: str,
        text: str,
        tools: list[dict],
        dispatch: Callable[[str, dict], str],
        images: list[ImageRef] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        max_iterations: int | None = None,
    ) -> str:
        """Run an agentic tool-use loop and return the model's final text.

        The model may call the given ``tools`` (executed by ``dispatch``) to
        research and ground its answer; we feed results back until it stops
        requesting tools, then return the text it produced. This is how grounding
        happens during generation rather than as a postfix. Works identically on
        Bedrock and the Anthropic API, since both speak the same tool-use schema.
        """
        blocks = self._build_blocks(text, images or [])
        model = model or self.model
        max_tokens = max_tokens or self.cfg.max_output_tokens
        max_iterations = max_iterations or self.cfg.research_max_iterations

        messages: list[dict] = [{"role": "user", "content": blocks}]
        for _ in range(max_iterations):
            resp = self._create(
                model=model, max_tokens=max_tokens, system=system, messages=messages, tools=tools
            )
            text_out = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            if getattr(resp, "stop_reason", None) != "tool_use":
                return text_out
            messages.append({"role": "assistant", "content": [_block_to_dict(b) for b in resp.content]})
            results = []
            for b in resp.content:
                if getattr(b, "type", None) == "tool_use":
                    tool_input = dict(b.input)
                    out = dispatch(b.name, tool_input)
                    log.info("  research: %s(%s) -> %d chars", b.name, tool_input.get("query") or tool_input.get("url") or "", len(out))
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
            messages.append({"role": "user", "content": results})

        # Iterations exhausted while still calling tools: force a tool-free answer.
        log.info("Research budget exhausted (%d iterations); requesting final text.", max_iterations)
        resp = self._create(model=model, max_tokens=max_tokens, system=system, messages=messages)
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

    def _create(
        self, *, model: str, max_tokens: int, system: str, messages: list[dict], tools: list[dict] | None = None
    ):
        """One Messages call with retry/backoff, normalizing errors to LLMError."""
        last_exc: Exception | None = None
        kwargs: dict = {}
        if tools:
            kwargs["tools"] = tools
        for attempt in range(1, self.cfg.llm_retries + 1):
            try:
                return self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=messages,
                    **kwargs,
                )
            except Exception as exc:  # noqa: BLE001 - normalized into LLMError below
                last_exc = exc
                wait = min(2**attempt, 30)
                log.warning(
                    "Claude request failed (attempt %d/%d): %s — retrying in %ds",
                    attempt,
                    self.cfg.llm_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
        raise LLMError(f"Claude request failed after {self.cfg.llm_retries} attempts: {last_exc}")
