"""Unit tests for OpenAIAdapter.stream_text (Phase 3). No network."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rh_cognitv.nodes.llm.errors import RateLimitError
from rh_cognitv.nodes.llm.types import LLMConfig, Message, ToolDefinition
from rh_cognitv.nodes.llm_adapters.base import StreamAdapter
from rh_cognitv.nodes.llm_adapters.openai_adapter import OpenAIAdapter

from pydantic import BaseModel


class _WeatherArgs(BaseModel):
    city: str

def make_chunk(content=None, model="gpt-4o-mini", usage=None):
    delta = SimpleNamespace(content=content)
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice], model=model, usage=usage)


def make_tool_chunk(index=0, call_id=None, name=None, arguments=None, model="gpt-4o-mini"):
    function = SimpleNamespace(name=name, arguments=arguments)
    raw_call = SimpleNamespace(index=index, id=call_id, function=function)
    delta = SimpleNamespace(content=None, tool_calls=[raw_call])
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice], model=model, usage=None)


def usage_chunk(prompt=5, completion=7, total=12, model="gpt-4o-mini"):
    usage = SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=total
    )
    return SimpleNamespace(choices=[], model=model, usage=usage)


class FakeStream:
    def __init__(self, chunks, exc=None):
        self._chunks = chunks
        self._exc = exc

    def __aiter__(self):
        async def _gen():
            for c in self._chunks:
                yield c
            if self._exc is not None:
                raise self._exc

        return _gen()


class FakeCompletions:
    def __init__(self, stream):
        self._stream = stream
        self.last_kwargs = None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._stream


class FakeClient:
    def __init__(self, stream):
        self.completions = FakeCompletions(stream)
        self.chat = SimpleNamespace(completions=self.completions)


@pytest.fixture
def config() -> LLMConfig:
    return LLMConfig(model="gpt-4o-mini")


@pytest.mark.asyncio
class TestStreamText:
    async def test_is_stream_adapter(self):
        adapter = OpenAIAdapter(client=FakeClient(FakeStream([])))
        assert isinstance(adapter, StreamAdapter)

    async def test_yields_text_deltas(self, config):
        chunks = [make_chunk("Hello"), make_chunk(" world")]
        adapter = OpenAIAdapter(client=FakeClient(FakeStream(chunks)))
        out = [d async for d in adapter.stream_text([Message(role="user", content="x")], config)]
        texts = [d.text for d in out if d.text is not None]
        assert texts == ["Hello", " world"]

    async def test_sets_stream_flags_in_payload(self, config):
        client = FakeClient(FakeStream([make_chunk("hi")]))
        adapter = OpenAIAdapter(client=client)
        _ = [d async for d in adapter.stream_text([Message(role="user", content="x")], config)]
        kwargs = client.completions.last_kwargs
        assert kwargs["stream"] is True
        assert kwargs["stream_options"] == {"include_usage": True}

    async def test_final_usage_chunk_carries_tokens(self, config):
        chunks = [make_chunk("hi"), usage_chunk()]
        adapter = OpenAIAdapter(client=FakeClient(FakeStream(chunks)))
        out = [d async for d in adapter.stream_text([Message(role="user", content="x")], config)]
        usage_deltas = [d for d in out if d.usage is not None]
        assert len(usage_deltas) == 1
        assert usage_deltas[0].usage.total_tokens == 12

    async def test_error_is_mapped(self, config):
        exc = type("RateLimitError", (Exception,), {})()
        exc.status_code = 429
        stream = FakeStream([make_chunk("hi")], exc=exc)
        adapter = OpenAIAdapter(client=FakeClient(stream))
        with pytest.raises(RateLimitError):
            _ = [d async for d in adapter.stream_text([Message(role="user", content="x")], config)]


_WEATHER_TOOL = ToolDefinition(
    name="get_weather", description="Get weather", parameters_model=_WeatherArgs
)


@pytest.mark.asyncio
class TestStreamToolCalls:
    async def test_yields_tool_call_deltas(self, config):
        chunks = [
            make_tool_chunk(index=0, call_id="c1", name="get_weather"),
            make_tool_chunk(index=0, arguments='{"city": '),
            make_tool_chunk(index=0, arguments='"Paris"}'),
        ]
        adapter = OpenAIAdapter(client=FakeClient(FakeStream(chunks)))
        out = [
            d
            async for d in adapter.stream_text(
                [Message(role="user", content="x")], config, tools=[_WEATHER_TOOL]
            )
        ]
        frags = [f for d in out if d.tool_call_deltas for f in d.tool_call_deltas]
        assert frags[0].tool_name == "get_weather"
        assert frags[0].call_id == "c1"
        assert "".join(f.arguments_delta or "" for f in frags) == '{"city": "Paris"}'

    async def test_tools_and_choice_in_payload(self, config):
        client = FakeClient(FakeStream([make_chunk("hi")]))
        adapter = OpenAIAdapter(client=client)
        _ = [
            d
            async for d in adapter.stream_text(
                [Message(role="user", content="x")],
                config,
                tools=[_WEATHER_TOOL],
                tool_choice="get_weather",
            )
        ]
        kwargs = client.completions.last_kwargs
        assert kwargs["tools"][0]["function"]["name"] == "get_weather"
        assert kwargs["tool_choice"] == {
            "type": "function",
            "function": {"name": "get_weather"},
        }

    async def test_no_tools_omits_payload_keys(self, config):
        client = FakeClient(FakeStream([make_chunk("hi")]))
        adapter = OpenAIAdapter(client=client)
        _ = [d async for d in adapter.stream_text([Message(role="user", content="x")], config)]
        assert "tools" not in client.completions.last_kwargs
        assert "tool_choice" not in client.completions.last_kwargs

