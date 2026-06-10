"""LLMStreamNode — streaming LLM completion (Phase 3).

``run()`` is an :class:`~collections.abc.AsyncGenerator` that yields canonical
:data:`~rh_cognitv.nodes.llm.events.StreamEvent` instances (DD-04): one
``StreamStarted``, zero or more ``StreamTextDelta`` (optionally batched per
DD-05), and a final ``StreamCompleted`` carrying the consolidated text, object,
tool calls, and metadata. If the underlying stream fails, a ``StreamErrorEvent``
is emitted and the canonical error is re-raised.

Tool calling (DD-07) works the same way as :class:`LLMStructuredNode`: pass a
list of :class:`~rh_cognitv.nodes.llm.types.ToolDefinition` wrappers and an
optional ``tool_choice``. Streamed tool-call fragments are reconstructed into a
consolidated ``tool_calls`` list, exposed on the final ``StreamCompleted`` event
and on the :class:`~rh_cognitv.nodes.llm.types.StreamResult` returned by
:meth:`collect`. With ``validate_tool_args=True`` (default) each call's
arguments are validated against its tool schema (SG-05).

Every emitted event is also delivered to the optional ``on_event`` callback for
secondary consumers (logging, monitoring, a future EventBus).

Because an async generator cannot ``return`` a value, consumers obtain the
consolidated :class:`~rh_cognitv.nodes.llm.types.StreamResult` either from the
final ``StreamCompleted`` event or via the convenience :meth:`collect` method.
"""

from __future__ import annotations

import inspect
import json
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

from rh_cognitv.nodes.base import BaseNode
from rh_cognitv.nodes.llm.errors import LLMError, LLMErrorFamily, ToolValidationError
from rh_cognitv.nodes.llm.events import (
    StreamCompleted,
    StreamErrorEvent,
    StreamEvent,
    StreamStarted,
    StreamTextDelta,
)
from rh_cognitv.nodes.llm.structured_node import validate_tool_calls
from rh_cognitv.nodes.llm.types import (
    LLMConfig,
    LLMResultMeta,
    Message,
    StreamResult,
    StreamToolCallDelta,
    ToolCallResult,
    ToolDefinition,
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


def _to_error_event(exc: Exception) -> tuple[LLMError, StreamErrorEvent]:
    """Coerce ``exc`` to a canonical error and its matching ``StreamErrorEvent``."""
    canonical = (
        exc if isinstance(exc, LLMError) else LLMError(str(exc) or type(exc).__name__)
    )
    event = StreamErrorEvent(
        family=canonical.family.value
        if isinstance(canonical.family, LLMErrorFamily)
        else str(canonical.family),
        code=canonical.code,
        message=canonical.message,
        retryable=canonical.retryable,
    )
    return canonical, event


class _ToolCallAccumulator:
    """Reconstructs consolidated tool calls from streamed fragments (DD-07).

    Fragments are keyed by ``index``: ``tool_name`` / ``call_id`` are captured
    from whichever fragment supplies them and ``arguments_delta`` fragments are
    concatenated into the final JSON-encoded argument string.
    """

    def __init__(self) -> None:
        self._slots: dict[int, dict[str, Any]] = {}
        self._order: list[int] = []

    @property
    def has_calls(self) -> bool:
        return bool(self._order)

    def add(self, fragments: list[StreamToolCallDelta]) -> None:
        for frag in fragments:
            slot = self._slots.get(frag.index)
            if slot is None:
                slot = {"call_id": None, "tool_name": None, "args": []}
                self._slots[frag.index] = slot
                self._order.append(frag.index)
            if frag.call_id is not None:
                slot["call_id"] = frag.call_id
            if frag.tool_name is not None:
                slot["tool_name"] = frag.tool_name
            if frag.arguments_delta is not None:
                slot["args"].append(frag.arguments_delta)

    def build(self) -> list[ToolCallResult]:
        calls: list[ToolCallResult] = []
        for index in self._order:
            slot = self._slots[index]
            name = slot["tool_name"] or ""
            raw = "".join(slot["args"]).strip() or "{}"
            try:
                arguments = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ToolValidationError(
                    f"Streamed tool {name!r} produced non-JSON arguments: {raw!r}",
                    tool_name=name,
                ) from exc
            calls.append(
                ToolCallResult(
                    tool_name=name, arguments=arguments, call_id=slot["call_id"]
                )
            )
        return calls


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
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | None = "auto",
        validate_tool_args: bool = True,
        batch_size: int = 1,
        on_event: OnEvent | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream a completion, yielding canonical :data:`StreamEvent` objects.

        Pass ``tools`` (and optionally ``tool_choice``) to stream tool calls the
        same way as :class:`LLMStructuredNode` (DD-07). The consolidated calls
        are exposed on the final ``StreamCompleted`` event and validated against
        their schemas when ``validate_tool_args`` is ``True`` (SG-05).
        """
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
        tool_accumulator = _ToolCallAccumulator()
        final_usage = TokenUsage()
        final_model = config.model

        batch_text: list[str] = []
        batch_object: dict[str, Any] | None = None
        batch_tool_deltas: list[StreamToolCallDelta] = []
        batch_count = 0
        emit_index = 0

        start = time.perf_counter()
        try:
            async for delta in self.adapter.stream_text(
                messages, config, tools, tool_choice
            ):
                if delta.usage is not None:
                    final_usage = delta.usage
                if delta.model is not None:
                    final_model = delta.model

                has_content = (
                    delta.text is not None
                    or delta.object_fragment is not None
                    or delta.tool_call_deltas is not None
                )
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
                if delta.tool_call_deltas is not None:
                    tool_accumulator.add(delta.tool_call_deltas)
                    batch_tool_deltas.extend(delta.tool_call_deltas)

                batch_count += 1
                if batch_count >= batch_size:
                    event = StreamTextDelta(
                        text="".join(batch_text) if batch_text else None,
                        object_fragment=batch_object,
                        tool_call_deltas=batch_tool_deltas or None,
                        index=emit_index,
                    )
                    emit_index += 1
                    batch_text = []
                    batch_object = None
                    batch_tool_deltas = []
                    batch_count = 0
                    await dispatch(event)
                    yield event
        except Exception as exc:
            canonical, error_event = _to_error_event(exc)
            await dispatch(error_event)
            yield error_event
            raise canonical from exc

        if batch_count > 0:
            event = StreamTextDelta(
                text="".join(batch_text) if batch_text else None,
                object_fragment=batch_object,
                tool_call_deltas=batch_tool_deltas or None,
                index=emit_index,
            )
            await dispatch(event)
            yield event

        try:
            tool_calls = tool_accumulator.build()
            if tool_calls and validate_tool_args and tools:
                validate_tool_calls(tool_calls, tools)
        except Exception as exc:
            canonical, error_event = _to_error_event(exc)
            await dispatch(error_event)
            yield error_event
            raise canonical from exc

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
            tool_calls=tool_calls,
            meta=meta,
        )
        await dispatch(completed)
        yield completed

    def __call__(
        self,
        prompt: str | list[Message],
        config: LLMConfig,
        *,
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | None = "auto",
        validate_tool_args: bool = True,
        batch_size: int = 1,
        on_event: OnEvent | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Return the stream generator (DD-08), so ``node(...)`` is iterable."""
        return self.run(
            prompt,
            config,
            tools=tools,
            tool_choice=tool_choice,
            validate_tool_args=validate_tool_args,
            batch_size=batch_size,
            on_event=on_event,
        )

    async def collect(
        self,
        prompt: str | list[Message],
        config: LLMConfig,
        *,
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | None = "auto",
        validate_tool_args: bool = True,
        batch_size: int = 1,
        on_event: OnEvent | None = None,
    ) -> StreamResult:
        """Drive the stream to completion and return the consolidated result."""
        result: StreamResult | None = None
        async for event in self.run(
            prompt,
            config,
            tools=tools,
            tool_choice=tool_choice,
            validate_tool_args=validate_tool_args,
            batch_size=batch_size,
            on_event=on_event,
        ):
            if isinstance(event, StreamCompleted):
                result = StreamResult(
                    text=event.text,
                    object=event.object,
                    tool_calls=event.tool_calls,
                    meta=event.meta,
                )
        if result is None:  # pragma: no cover - run always emits StreamCompleted
            raise RuntimeError("stream completed without a StreamCompleted event")
        return result
