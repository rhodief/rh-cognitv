"""Abstract per-capability adapter interfaces (DD-01, DD-02, DD-11).

There is one ABC per node capability. A concrete adapter implements all the
interfaces its provider supports via multiple inheritance, e.g.::

    class OpenAIAdapter(TextAdapter, StreamAdapter, StructuredAdapter,
                        EmbeddingAdapter):
        ...

Each node type-hints its ``adapter`` parameter with the specific ABC it needs,
so passing an adapter that lacks the required capability is caught by the type
checker before runtime. Embedding-only providers implement only
:class:`EmbeddingAdapter` without stubbing chat methods (DD-11).

All adapter methods are ``async`` (DD-12). Each adapter is responsible for
mapping provider-specific exceptions onto the canonical
:class:`~rh_cognitv.nodes.llm.errors.LLMError` taxonomy (DD-09).
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator

from rh_cognitv.nodes.llm.types import (
    EmbeddingResult,
    LLMConfig,
    Message,
    StreamDelta,
    StructuredResult,
    TextResult,
    ToolDefinition,
)


class TextAdapter(abc.ABC):
    """Single-shot (non-streaming) chat completion capability."""

    @abc.abstractmethod
    async def generate_text(
        self,
        messages: list[Message],
        config: LLMConfig,
    ) -> TextResult:
        """Produce a single completion for ``messages``."""
        raise NotImplementedError


class StreamAdapter(abc.ABC):
    """Streaming chat completion capability.

    Implementations yield raw :class:`StreamDelta` chunks; the node is
    responsible for batching, event emission, and consolidation.
    """

    @abc.abstractmethod
    def stream_text(
        self,
        messages: list[Message],
        config: LLMConfig,
    ) -> AsyncIterator[StreamDelta]:
        """Yield incremental :class:`StreamDelta` chunks for ``messages``."""
        raise NotImplementedError


class StructuredAdapter(abc.ABC):
    """Tool-calling / function-calling capability."""

    @abc.abstractmethod
    async def generate_structured(
        self,
        messages: list[Message],
        config: LLMConfig,
        tools: list[ToolDefinition],
        tool_choice: str | None = None,
    ) -> StructuredResult:
        """Invoke the model with ``tools`` and return structured tool calls."""
        raise NotImplementedError


class EmbeddingAdapter(abc.ABC):
    """Batch text→embedding capability (separate ABC, DD-11)."""

    @abc.abstractmethod
    async def embed(
        self,
        texts: list[str],
        config: LLMConfig,
    ) -> EmbeddingResult:
        """Embed a batch of ``texts`` into float vectors."""
        raise NotImplementedError
