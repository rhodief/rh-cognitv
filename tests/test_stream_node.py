"""Unit tests for LLMStreamNode (Phase 3)."""

from __future__ import annotations

import pytest

from rh_cognitv.nodes.llm.errors import LLMError, RateLimitError, ToolValidationError
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
    StreamToolCallDelta,
    TokenUsage,
    ToolDefinition,
)
from rh_cognitv.nodes.llm_adapters.base import StreamAdapter

from pydantic import BaseModel


class FakeStreamAdapter(StreamAdapter):
    """Yields a scripted list of StreamDelta chunks (optionally raising)."""

    provider = "fake"

    def __init__(self, deltas: list[StreamDelta], raise_at: int | None = None, exc=None):
        self.deltas = deltas
        self.raise_at = raise_at
        self.exc = exc or RateLimitError("boom")
        self.calls: list[tuple[list[Message], LLMConfig]] = []
        self.last_tools = None
        self.last_tool_choice = None

    def stream_text(self, messages, config, tools=None, tool_choice=None):
        self.calls.append((messages, config))
        self.last_tools = tools
        self.last_tool_choice = tool_choice

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


class _WeatherArgs(BaseModel):
    city: str


_WEATHER_TOOL = ToolDefinition(
    name="get_weather",
    description="Get weather",
    parameters_model=_WeatherArgs,
)


def _tool_delta(index, name=None, args=None, call_id=None):
    return StreamDelta(
        tool_call_deltas=[
            StreamToolCallDelta(
                index=index, tool_name=name, arguments_delta=args, call_id=call_id
            )
        ]
    )


@pytest.mark.asyncio
class TestStreamingToolCalls:
    async def test_consolidates_fragmented_tool_call(self, config):
        # OpenAI-style: name/id first, then argument fragments.
        deltas = [
            _tool_delta(0, name="get_weather", call_id="c1"),
            _tool_delta(0, args='{"city": '),
            _tool_delta(0, args='"Paris"}'),
        ]
        node = LLMStreamNode(FakeStreamAdapter(deltas))
        events = [e async for e in node.run("weather?", config, tools=[_WEATHER_TOOL])]

        completed = events[-1]
        assert isinstance(completed, StreamCompleted)
        assert len(completed.tool_calls) == 1
        call = completed.tool_calls[0]
        assert call.tool_name == "get_weather"
        assert call.arguments == {"city": "Paris"}
        assert call.call_id == "c1"
        assert isinstance(call.parsed_arguments, _WeatherArgs)
        assert call.parsed_arguments.city == "Paris"

    async def test_multiple_tool_calls_by_index(self, config):
        deltas = [
            _tool_delta(0, name="get_weather", args='{"city": "Paris"}', call_id="c1"),
            _tool_delta(1, name="get_weather", args='{"city": "Rome"}', call_id="c2"),
        ]
        node = LLMStreamNode(FakeStreamAdapter(deltas))
        result = await node.collect("x", config, tools=[_WEATHER_TOOL])
        assert [c.arguments["city"] for c in result.tool_calls] == ["Paris", "Rome"]

    async def test_collect_exposes_tool_calls(self, config):
        deltas = [_tool_delta(0, name="get_weather", args='{"city": "Paris"}')]
        node = LLMStreamNode(FakeStreamAdapter(deltas))
        result = await node.collect("x", config, tools=[_WEATHER_TOOL])
        assert isinstance(result, StreamResult)
        assert result.tool_calls[0].parsed_arguments.city == "Paris"

    async def test_tool_call_deltas_emitted_on_events(self, config):
        deltas = [_tool_delta(0, name="get_weather", args='{"city": "Paris"}')]
        node = LLMStreamNode(FakeStreamAdapter(deltas))
        events = [e async for e in node.run("x", config, tools=[_WEATHER_TOOL])]
        text_deltas_with_tools = [
            e
            for e in events
            if isinstance(e, StreamTextDelta) and e.tool_call_deltas
        ]
        assert len(text_deltas_with_tools) == 1

    async def test_validation_failure_raises(self, config):
        deltas = [_tool_delta(0, name="get_weather", args='{"wrong": "field"}')]
        node = LLMStreamNode(FakeStreamAdapter(deltas))
        with pytest.raises(ToolValidationError):
            async for _ in node.run("x", config, tools=[_WEATHER_TOOL]):
                pass

    async def test_unknown_tool_raises(self, config):
        deltas = [_tool_delta(0, name="unknown_tool", args="{}")]
        node = LLMStreamNode(FakeStreamAdapter(deltas))
        with pytest.raises(ToolValidationError):
            async for _ in node.run("x", config, tools=[_WEATHER_TOOL]):
                pass

    async def test_validate_opt_out_keeps_raw_dicts(self, config):
        deltas = [_tool_delta(0, name="get_weather", args='{"wrong": "field"}')]
        node = LLMStreamNode(FakeStreamAdapter(deltas))
        result = await node.collect(
            "x", config, tools=[_WEATHER_TOOL], validate_tool_args=False
        )
        assert result.tool_calls[0].arguments == {"wrong": "field"}
        assert result.tool_calls[0].parsed_arguments is None

    async def test_bad_json_emits_error_and_raises(self, config):
        deltas = [_tool_delta(0, name="get_weather", args="{not json")]
        node = LLMStreamNode(FakeStreamAdapter(deltas))
        collected = []
        with pytest.raises(ToolValidationError):
            async for event in node.run("x", config, tools=[_WEATHER_TOOL]):
                collected.append(event)
        assert isinstance(collected[-1], StreamErrorEvent)

    async def test_no_tools_leaves_tool_calls_empty(self, config):
        node = LLMStreamNode(FakeStreamAdapter(text_deltas("hi")))
        result = await node.collect("x", config)
        assert result.tool_calls == []

    async def test_adapter_receives_tools_and_choice(self, config):
        adapter = FakeStreamAdapter(text_deltas("hi"))
        node = LLMStreamNode(adapter)
        _ = [
            e
            async for e in node.run(
                "x", config, tools=[_WEATHER_TOOL], tool_choice="get_weather"
            )
        ]
        assert adapter.last_tools == [_WEATHER_TOOL]
        assert adapter.last_tool_choice == "get_weather"
