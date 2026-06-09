"""Unit tests for OpenAIAdapter.generate_structured (Phase 4). No network."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from rh_cognitv.nodes.llm.errors import RateLimitError, ToolValidationError
from rh_cognitv.nodes.llm.types import LLMConfig, Message, ToolDefinition
from rh_cognitv.nodes.llm_adapters.base import StructuredAdapter
from rh_cognitv.nodes.llm_adapters.openai_adapter import (
    OpenAIAdapter,
    _map_tool_choice,
    _to_openai_tool,
)


class WeatherArgs(BaseModel):
    city: str


WEATHER_TOOL = ToolDefinition(
    name="get_weather",
    description="Get weather",
    parameters_model=WeatherArgs,
)


def make_tool_call(name, arguments, call_id="call_1"):
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(id=call_id, function=function)


def make_response(tool_calls, model="gpt-4o-mini"):
    message = SimpleNamespace(tool_calls=tool_calls)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(prompt_tokens=5, completion_tokens=7, total_tokens=12)
    return SimpleNamespace(choices=[choice], model=model, usage=usage)


class FakeCompletions:
    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc
        self.last_kwargs = None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        if self._exc is not None:
            raise self._exc
        return self._response


class FakeClient:
    def __init__(self, response=None, exc=None):
        self.completions = FakeCompletions(response=response, exc=exc)
        self.chat = SimpleNamespace(completions=self.completions)


@pytest.fixture
def config() -> LLMConfig:
    return LLMConfig(model="gpt-4o-mini")


class TestHelpers:
    def test_to_openai_tool_schema(self):
        schema = _to_openai_tool(WEATHER_TOOL)
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_weather"
        assert schema["function"]["description"] == "Get weather"
        assert "city" in schema["function"]["parameters"]["properties"]

    def test_map_tool_choice_none_is_required(self):
        assert _map_tool_choice(None) == "required"

    def test_map_tool_choice_keywords_passthrough(self):
        assert _map_tool_choice("auto") == "auto"
        assert _map_tool_choice("required") == "required"
        assert _map_tool_choice("none") == "none"

    def test_map_tool_choice_specific_name(self):
        assert _map_tool_choice("get_weather") == {
            "type": "function",
            "function": {"name": "get_weather"},
        }


@pytest.mark.asyncio
class TestGenerateStructured:
    async def test_is_structured_adapter(self):
        adapter = OpenAIAdapter(client=FakeClient(make_response([])))
        assert isinstance(adapter, StructuredAdapter)

    async def test_single_tool_call(self, config):
        response = make_response(
            [make_tool_call("get_weather", json.dumps({"city": "Paris"}))]
        )
        adapter = OpenAIAdapter(client=FakeClient(response))
        result = await adapter.generate_structured(
            [Message(role="user", content="x")], config, [WEATHER_TOOL]
        )
        assert len(result.tool_calls) == 1
        call = result.tool_calls[0]
        assert call.tool_name == "get_weather"
        assert call.arguments == {"city": "Paris"}
        assert call.call_id == "call_1"
        assert call.parsed_arguments is None  # adapter leaves validation to node
        assert result.meta.provider == "openai"
        assert result.meta.tokens_used.total_tokens == 12

    async def test_multiple_tool_calls(self, config):
        response = make_response(
            [
                make_tool_call("get_weather", json.dumps({"city": "Paris"}), "c1"),
                make_tool_call("get_weather", json.dumps({"city": "Rome"}), "c2"),
            ]
        )
        adapter = OpenAIAdapter(client=FakeClient(response))
        result = await adapter.generate_structured(
            [Message(role="user", content="x")], config, [WEATHER_TOOL]
        )
        assert [c.call_id for c in result.tool_calls] == ["c1", "c2"]

    async def test_no_tool_calls_returns_empty_list(self, config):
        adapter = OpenAIAdapter(client=FakeClient(make_response(None)))
        result = await adapter.generate_structured(
            [Message(role="user", content="x")], config, [WEATHER_TOOL]
        )
        assert result.tool_calls == []

    async def test_payload_includes_tools_and_choice(self, config):
        client = FakeClient(make_response([]))
        adapter = OpenAIAdapter(client=client)
        await adapter.generate_structured(
            [Message(role="user", content="x")],
            config,
            [WEATHER_TOOL],
            tool_choice="get_weather",
        )
        kwargs = client.completions.last_kwargs
        assert kwargs["tools"][0]["function"]["name"] == "get_weather"
        assert kwargs["tool_choice"] == {
            "type": "function",
            "function": {"name": "get_weather"},
        }

    async def test_malformed_json_raises_tool_validation_error(self, config):
        response = make_response([make_tool_call("get_weather", "{not json")])
        adapter = OpenAIAdapter(client=FakeClient(response))
        with pytest.raises(ToolValidationError) as info:
            await adapter.generate_structured(
                [Message(role="user", content="x")], config, [WEATHER_TOOL]
            )
        assert info.value.tool_name == "get_weather"

    async def test_empty_arguments_default_to_empty_dict(self, config):
        response = make_response([make_tool_call("get_weather", "")])
        adapter = OpenAIAdapter(client=FakeClient(response))
        result = await adapter.generate_structured(
            [Message(role="user", content="x")], config, [WEATHER_TOOL]
        )
        assert result.tool_calls[0].arguments == {}

    async def test_provider_error_is_mapped(self, config):
        exc = Exception("rate")
        exc.status_code = 429
        adapter = OpenAIAdapter(client=FakeClient(exc=exc))
        with pytest.raises(RateLimitError):
            await adapter.generate_structured(
                [Message(role="user", content="x")], config, [WEATHER_TOOL]
            )
