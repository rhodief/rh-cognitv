"""Unit tests for LLMStreamNode (Phase 3)."""

from __future__ import annotations

import pytest

from rh_cognitv.nodes.llm.errors import LLMError, RateLimitError
from rh_cognitv.nodes.llm.events import (
    StreamCompleted,
    StreamErrorEvent,
    StreamStarted,
    StreamTextDelta,
)
from rh_cognitv.nodes.llm.stream_node import LLMStreamNode
from rh_cognitv.nodes.llm.types import (
    LLMConfig,
    Message,
    StreamDelta,
    StreamResult,
    TokenUsage,
)
from rh_cognitv.nodes.llm_adapters.base import StreamAdapter


class FakeStreamAdapter(StreamAdapter):
    """Yields a scripted list of StreamDelta chunks (optionally raising)."""

    provider = "fake"

    def __init__(self, deltas: list[StreamDelta], raise_at: int | None = None, exc=None):
        self.deltas = deltas
        self.raise_at = raise_at
        self.exc = exc or RateLimitError("boom")
        self.calls: list[tuple[list[Message], LLMConfig]] = []

    def stream_text(self, messages, config):
        self.calls.append((messages, config))

        async def _gen():
            for i, delta in enumerate(self.deltas):
                if self.raise_at is not None and i == self.raise_at:
                    raise self.exc
                yield delta

        return _gen()


def text_deltas(*chunks: str) -> list[StreamDelta]:
    return [StreamDelta(text=c) for c in chunks]


@pytest.fixture
def config() -> LLMConfig:
    return LLMConfig(model="gpt-test")


@pytest.mark.asyncio
class TestBasics:
    async def test_emits_started_deltas_completed(self, config):
        adapter = FakeStreamAdapter(text_deltas("Hello", " world"))
        node = LLMStreamNode(adapter)
        events = [e async for e in node.run("hi", config)]

        assert isinstance(events[0], StreamStarted)
        assert events[0].provider == "fake"
        assert isinstance(events[-1], StreamCompleted)
        deltas = [e for e in events if isinstance(e, StreamTextDelta)]
        assert [d.text for d in deltas] == ["Hello", " world"]

    async def test_completed_consolidates_text(self, config):
        adapter = FakeStreamAdapter(text_deltas("Hello", " ", "world"))
        node = LLMStreamNode(adapter)
        events = [e async for e in node.run("hi", config)]
        completed = events[-1]
        assert isinstance(completed, StreamCompleted)
        assert completed.text == "Hello world"

    async def test_prompt_normalized(self, config):
        adapter = FakeStreamAdapter(text_deltas("x"))
        node = LLMStreamNode(adapter)
        _ = [e async for e in node.run("hello", config)]
        messages, _cfg = adapter.calls[0]
        assert messages == [Message(role="user", content="hello")]

    async def test_call_returns_generator(self, config):
        adapter = FakeStreamAdapter(text_deltas("a", "b"))
        node = LLMStreamNode(adapter)
        events = [e async for e in node("hi", config)]
        assert isinstance(events[-1], StreamCompleted)

    async def test_delta_indices_are_sequential(self, config):
        adapter = FakeStreamAdapter(text_deltas("a", "b", "c"))
        node = LLMStreamNode(adapter)
        events = [e async for e in node.run("hi", config)]
        deltas = [e for e in events if isinstance(e, StreamTextDelta)]
        assert [d.index for d in deltas] == [0, 1, 2]


@pytest.mark.asyncio
class TestBatching:
    async def test_batch_size_1_emits_each(self, config):
        adapter = FakeStreamAdapter(text_deltas("a", "b", "c"))
        node = LLMStreamNode(adapter)
        events = [e async for e in node.run("hi", config, batch_size=1)]
        deltas = [e for e in events if isinstance(e, StreamTextDelta)]
        assert [d.text for d in deltas] == ["a", "b", "c"]

    async def test_batch_size_3_concatenates(self, config):
        chunks = ["Hello", " world!", " I'm", " very", " happy", " to", " help", " you"]
        adapter = FakeStreamAdapter(text_deltas(*chunks))
        node = LLMStreamNode(adapter)
        events = [e async for e in node.run("hi", config, batch_size=3)]
        deltas = [e for e in events if isinstance(e, StreamTextDelta)]
        assert [d.text for d in deltas] == [
            "Hello world! I'm",
            " very happy to",
            " help you",
        ]
        assert (events[-1]).text == "".join(chunks)

    async def test_batch_larger_than_total_flushes_once(self, config):
        adapter = FakeStreamAdapter(text_deltas("a", "b"))
        node = LLMStreamNode(adapter)
        events = [e async for e in node.run("hi", config, batch_size=10)]
        deltas = [e for e in events if isinstance(e, StreamTextDelta)]
        assert len(deltas) == 1
        assert deltas[0].text == "ab"

    async def test_invalid_batch_size_raises(self, config):
        adapter = FakeStreamAdapter(text_deltas("a"))
        node = LLMStreamNode(adapter)
        with pytest.raises(ValueError):
            _ = [e async for e in node.run("hi", config, batch_size=0)]


@pytest.mark.asyncio
class TestMetadata:
    async def test_usage_and_model_from_final_chunk(self, config):
        deltas = [
            StreamDelta(text="hi", model="gpt-test"),
            StreamDelta(usage=TokenUsage(prompt_tokens=3, completion_tokens=4, total_tokens=7), model="gpt-test"),
        ]
        adapter = FakeStreamAdapter(deltas)
        node = LLMStreamNode(adapter)
        events = [e async for e in node.run("hi", config)]
        completed = events[-1]
        assert isinstance(completed, StreamCompleted)
        assert completed.meta.tokens_used.total_tokens == 7
        assert completed.meta.model == "gpt-test"
        assert completed.meta.provider == "fake"
        assert completed.meta.duration_ms >= 0

    async def test_metadata_only_chunk_not_emitted_as_delta(self, config):
        deltas = [
            StreamDelta(text="hi"),
            StreamDelta(usage=TokenUsage(total_tokens=5)),
        ]
        adapter = FakeStreamAdapter(deltas)
        node = LLMStreamNode(adapter)
        events = [e async for e in node.run("hi", config)]
        delta_events = [e for e in events if isinstance(e, StreamTextDelta)]
        assert len(delta_events) == 1


@pytest.mark.asyncio
class TestObjectStreaming:
    async def test_object_fragments_merged(self, config):
        deltas = [
            StreamDelta(object_fragment={"a": 1}),
            StreamDelta(object_fragment={"b": 2}),
        ]
        adapter = FakeStreamAdapter(deltas)
        node = LLMStreamNode(adapter)
        events = [e async for e in node.run("hi", config)]
        completed = events[-1]
        assert isinstance(completed, StreamCompleted)
        assert completed.object == {"a": 1, "b": 2}


@pytest.mark.asyncio
class TestErrors:
    async def test_error_event_then_raise(self, config):
        adapter = FakeStreamAdapter(
            text_deltas("a", "b"), raise_at=1, exc=RateLimitError("rate")
        )
        node = LLMStreamNode(adapter)
        collected = []
        with pytest.raises(RateLimitError):
            async for event in node.run("hi", config):
                collected.append(event)
        assert isinstance(collected[-1], StreamErrorEvent)
        assert collected[-1].family == "rate_limit"
        assert collected[-1].retryable is True

    async def test_non_llm_error_wrapped(self, config):
        adapter = FakeStreamAdapter(
            text_deltas("a"), raise_at=0, exc=ValueError("oops")
        )
        node = LLMStreamNode(adapter)
        with pytest.raises(LLMError):
            async for _ in node.run("hi", config):
                pass


@pytest.mark.asyncio
class TestCallback:
    async def test_on_event_receives_all_events(self, config):
        adapter = FakeStreamAdapter(text_deltas("a", "b"))
        seen = []
        node = LLMStreamNode(adapter, on_event=lambda e: seen.append(e))
        events = [e async for e in node.run("hi", config)]
        assert seen == events

    async def test_async_on_event(self, config):
        adapter = FakeStreamAdapter(text_deltas("a"))
        seen = []

        async def on_event(e):
            seen.append(e)

        node = LLMStreamNode(adapter)
        events = [e async for e in node.run("hi", config, on_event=on_event)]
        assert seen == events

    async def test_per_call_on_event_overrides_constructor(self, config):
        adapter = FakeStreamAdapter(text_deltas("a"))
        ctor_seen, call_seen = [], []
        node = LLMStreamNode(adapter, on_event=lambda e: ctor_seen.append(e))
        _ = [e async for e in node.run("hi", config, on_event=lambda e: call_seen.append(e))]
        assert ctor_seen == []
        assert len(call_seen) > 0


@pytest.mark.asyncio
class TestCollect:
    async def test_collect_returns_stream_result(self, config):
        adapter = FakeStreamAdapter(text_deltas("Hello", " world"))
        node = LLMStreamNode(adapter)
        result = await node.collect("hi", config)
        assert isinstance(result, StreamResult)
        assert result.text == "Hello world"
