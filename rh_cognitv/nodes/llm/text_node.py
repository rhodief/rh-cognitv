"""LLMTextNode — single-shot (non-streaming) LLM completion (Phase 2).

This is the reference node implementation. It accepts a prompt as either a
bare ``str`` (auto-wrapped as a single user message) or a ``list[Message]``
(DD-10), delegates the actual provider call to an injected
:class:`~rh_cognitv.nodes.llm_adapters.base.TextAdapter` (DD-03), and returns a
canonical :class:`~rh_cognitv.nodes.llm.types.TextResult`.

Optional ``on_request`` / ``on_response`` callbacks provide observability hooks
(SG-01); each may be a sync or async callable and receives a Pydantic model.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from rh_cognitv.nodes.base import BaseNode
from rh_cognitv.nodes.llm.types import (
    LLMConfig,
    LLMRequest,
    Message,
    TextResult,
    normalize_prompt,
)
from rh_cognitv.nodes.llm_adapters.base import TextAdapter

OnRequest = Callable[[LLMRequest], Awaitable[None] | None]
OnResponse = Callable[[TextResult], Awaitable[None] | None]


async def _maybe_await(value: Any) -> None:
    """Await ``value`` if it is awaitable; otherwise do nothing."""
    if inspect.isawaitable(value):
        await value


class LLMTextNode(BaseNode[TextResult]):
    """Single-shot text completion node."""

    def __init__(
        self,
        adapter: TextAdapter,
        *,
        on_request: OnRequest | None = None,
        on_response: OnResponse | None = None,
    ) -> None:
        self.adapter = adapter
        self.on_request = on_request
        self.on_response = on_response

    async def run(
        self,
        prompt: str | list[Message],
        config: LLMConfig,
    ) -> TextResult:
        """Run a single-shot completion and return a canonical ``TextResult``."""
        messages = normalize_prompt(prompt)
        request = LLMRequest(messages=messages, config=config)

        if self.on_request is not None:
            await _maybe_await(self.on_request(request))

        result = await self.adapter.generate_text(messages, config)

        if self.on_response is not None:
            await _maybe_await(self.on_response(result))

        return result
