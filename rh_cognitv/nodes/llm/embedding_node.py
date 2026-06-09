"""LLMEmbeddingNode — batch text→embedding (Phase 5).

Accepts a ``list[str]`` (a bare ``str`` is accepted as a one-item batch),
delegates to an injected
:class:`~rh_cognitv.nodes.llm_adapters.base.EmbeddingAdapter` (DD-11), and
returns a canonical :class:`~rh_cognitv.nodes.llm.types.EmbeddingResult` whose
``embeddings`` is a ``list[list[float]]`` aligned with the input order.

Optional ``on_request`` / ``on_response`` callbacks provide observability hooks
(SG-01); each may be a sync or async callable and receives a Pydantic model.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from rh_cognitv.nodes.base import BaseNode
from rh_cognitv.nodes.llm.types import (
    EmbeddingRequest,
    EmbeddingResult,
    LLMConfig,
)
from rh_cognitv.nodes.llm_adapters.base import EmbeddingAdapter

OnRequest = Callable[[EmbeddingRequest], Awaitable[None] | None]
OnResponse = Callable[[EmbeddingResult], Awaitable[None] | None]


async def _maybe_await(value: Any) -> None:
    """Await ``value`` if it is awaitable; otherwise do nothing."""
    if inspect.isawaitable(value):
        await value


class LLMEmbeddingNode(BaseNode[EmbeddingResult]):
    """Batch text→embedding node."""

    def __init__(
        self,
        adapter: EmbeddingAdapter,
        *,
        on_request: OnRequest | None = None,
        on_response: OnResponse | None = None,
    ) -> None:
        self.adapter = adapter
        self.on_request = on_request
        self.on_response = on_response

    async def run(
        self,
        texts: str | list[str],
        config: LLMConfig,
    ) -> EmbeddingResult:
        """Embed ``texts`` and return a canonical ``EmbeddingResult``."""
        batch = [texts] if isinstance(texts, str) else list(texts)
        if not batch:
            raise ValueError("LLMEmbeddingNode requires at least one text")

        request = EmbeddingRequest(texts=batch, config=config)

        if self.on_request is not None:
            await _maybe_await(self.on_request(request))

        result = await self.adapter.embed(batch, config)

        if self.on_response is not None:
            await _maybe_await(self.on_response(result))

        return result
