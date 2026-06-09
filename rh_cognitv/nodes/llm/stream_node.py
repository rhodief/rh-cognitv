"""LLMStreamNode — streaming LLM completion (Phase 3).

``run()`` is an :class:`~collections.abc.AsyncGenerator` that yields canonical
:data:`~rh_cognitv.nodes.llm.events.StreamEvent` instances (DD-04): one
``StreamStarted``, zero or more ``StreamTextDelta`` (optionally batched per
DD-05), and a final ``StreamCompleted`` carrying the consolidated text, object,
and metadata. If the underlying stream fails, a ``StreamErrorEvent`` is emitted
and the canonical error is re-raised.

Every emitted event is also delivered to the optional ``on_event`` callback for
secondary consumers (logging, monitoring, a future EventBus).

Because an async generator cannot ``return`` a value, consumers obtain the
consolidated :class:`~rh_cognitv.nodes.llm.types.StreamResult` either from the
final ``StreamCompleted`` event or via the convenience :meth:`collect` method.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

from rh_cognitv.nodes.base import BaseNode
from rh_cognitv.nodes.llm.errors import LLMError, LLMErrorFamily
from rh_cognitv.nodes.llm.events import (
    StreamCompleted,
    StreamErrorEvent,
    StreamEvent,
    StreamStarted,
    StreamTextDelta,
)
from rh_cognitv.nodes.llm.types import (
    LLMConfig,
    LLMResultMeta,
    Message,
    StreamResult,
    TokenUsage,
    normalize_prompt,
)
from rh_cognitv.nodes.llm_adapters.base import StreamAdapter

OnEvent = Callable[[StreamEvent], Awaitable[None] | None]


async def _maybe_await(value: Any) -> None:
    """Await ``value`` if it is awaitable; otherwise do nothing."""
    if inspect.isawaitable(value):
        await value


def _merge_object(
    base: dict[str, Any] | None, fragment: dict[str, Any]
) -> dict[str, Any]:
    """Shallow-merge a streamed ``object_fragment`` into the accumulator."""
    if base is None:
        return dict(fragment)
    base.update(fragment)
    return base


class LLMStreamNode(BaseNode[AsyncGenerator[StreamEvent, None]]):
    """Streaming text/object completion node."""

    def __init__(
        self,
        adapter: StreamAdapter,
        *,
        on_event: OnEvent | None = None,
    ) -> None:
        self.adapter = adapter
        self.on_event = on_event

    async def run(
        self,
        prompt: str | list[Message],
        config: LLMConfig,
        *,
        batch_size: int = 1,
        on_event: OnEvent | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream a completion, yielding canonical :data:`StreamEvent` objects."""
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        sink = on_event if on_event is not None else self.on_event

        async def dispatch(event: StreamEvent) -> None:
            if sink is not None:
                await _maybe_await(sink(event))

        messages = normalize_prompt(prompt)
        provider = getattr(self.adapter, "provider", "unknown")

        started = StreamStarted(model=config.model, provider=provider)
        await dispatch(started)
        yield started

        text_parts: list[str] = []
        accumulated_object: dict[str, Any] | None = None
        final_usage = TokenUsage()
        final_model = config.model

        batch_text: list[str] = []
        batch_object: dict[str, Any] | None = None
        batch_count = 0
        emit_index = 0

        start = time.perf_counter()
        try:
            async for delta in self.adapter.stream_text(messages, config):
                if delta.usage is not None:
                    final_usage = delta.usage
                if delta.model is not None:
                    final_model = delta.model

                has_content = delta.text is not None or delta.object_fragment is not None
                if not has_content:
                    continue

                if delta.text is not None:
                    text_parts.append(delta.text)
                    batch_text.append(delta.text)
                if delta.object_fragment is not None:
                    accumulated_object = _merge_object(
                        accumulated_object, delta.object_fragment
                    )
                    batch_object = _merge_object(batch_object, delta.object_fragment)

                batch_count += 1
                if batch_count >= batch_size:
                    event = StreamTextDelta(
                        text="".join(batch_text) if batch_text else None,
                        object_fragment=batch_object,
                        index=emit_index,
                    )
                    emit_index += 1
                    batch_text = []
                    batch_object = None
                    batch_count = 0
                    await dispatch(event)
                    yield event
        except Exception as exc:
            canonical = (
                exc
                if isinstance(exc, LLMError)
                else LLMError(str(exc) or type(exc).__name__)
            )
            error_event = StreamErrorEvent(
                family=canonical.family.value
                if isinstance(canonical.family, LLMErrorFamily)
                else str(canonical.family),
                code=canonical.code,
                message=canonical.message,
                retryable=canonical.retryable,
            )
            await dispatch(error_event)
            yield error_event
            raise canonical from exc

        if batch_count > 0:
            event = StreamTextDelta(
                text="".join(batch_text) if batch_text else None,
                object_fragment=batch_object,
                index=emit_index,
            )
            await dispatch(event)
            yield event

        duration_ms = (time.perf_counter() - start) * 1000
        meta = LLMResultMeta(
            model=final_model,
            provider=provider,
            tokens_used=final_usage,
            duration_ms=duration_ms,
        )
        completed = StreamCompleted(
            text="".join(text_parts),
            object=accumulated_object,
            meta=meta,
        )
        await dispatch(completed)
        yield completed

    def __call__(
        self,
        prompt: str | list[Message],
        config: LLMConfig,
        *,
        batch_size: int = 1,
        on_event: OnEvent | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Return the stream generator (DD-08), so ``node(...)`` is iterable."""
        return self.run(prompt, config, batch_size=batch_size, on_event=on_event)

    async def collect(
        self,
        prompt: str | list[Message],
        config: LLMConfig,
        *,
        batch_size: int = 1,
        on_event: OnEvent | None = None,
    ) -> StreamResult:
        """Drive the stream to completion and return the consolidated result."""
        result: StreamResult | None = None
        async for event in self.run(
            prompt, config, batch_size=batch_size, on_event=on_event
        ):
            if isinstance(event, StreamCompleted):
                result = StreamResult(
                    text=event.text, object=event.object, meta=event.meta
                )
        if result is None:  # pragma: no cover - run always emits StreamCompleted
            raise RuntimeError("stream completed without a StreamCompleted event")
        return result
