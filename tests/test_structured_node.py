"""Unit tests for LLMStructuredNode (Phase 4)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from rh_cognitv.nodes.base import BaseNode
from rh_cognitv.nodes.llm.errors import ToolValidationError
from rh_cognitv.nodes.llm.structured_node import LLMStructuredNode
from rh_cognitv.nodes.llm.types import (
    LLMConfig,
    LLMRequest,
    LLMResultMeta,
    Message,
    StructuredResult,
    TokenUsage,
    ToolCallResult,
    ToolDefinition,
)
from rh_cognitv.nodes.llm_adapters.base import StructuredAdapter


class WeatherArgs(BaseModel):
    city: str
    units: str = "celsius"


class SearchArgs(BaseModel):
    query: str


WEATHER_TOOL = ToolDefinition(
    name="get_weather",
    description="Get the weather for a city",
    parameters_model=WeatherArgs,
)
SEARCH_TOOL = ToolDefinition(
    name="search",
    description="Search the web",
    parameters_model=SearchArgs,
)


def make_meta() -> LLMResultMeta:
    return LLMResultMeta(
        model="gpt-test",
        provider="fake",
        tokens_used=TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        duration_ms=1.0,
    )


class RecordingAdapter(StructuredAdapter):
    """A fake StructuredAdapter that records the call and returns canned calls."""

    provider = "fake"

    def __init__(self, tool_calls: list[ToolCallResult]):
        self.result = StructuredResult(tool_calls=tool_calls, meta=make_meta())
        self.calls: list[tuple] = []

    async def generate_structured(self, messages, config, tools, tool_choice=None):
        self.calls.append((messages, config, tools, tool_choice))
        return self.result


@pytest.fixture
def config() -> LLMConfig:
    return LLMConfig(model="gpt-test")


@pytest.mark.asyncio
class TestRun:
    async def test_returns_structured_result(self, config):
        adapter = RecordingAdapter(
            [ToolCallResult(tool_name="get_weather", arguments={"city": "Paris"})]
        )
        node = LLMStructuredNode(adapter)
        result = await node.run("weather?", config, [WEATHER_TOOL])
        assert isinstance(result, StructuredResult)
        assert result.tool_calls[0].tool_name == "get_weather"

    async def test_prompt_normalized_and_args_forwarded(self, config):
        adapter = RecordingAdapter(
            [ToolCallResult(tool_name="get_weather", arguments={"city": "Paris"})]
        )
        node = LLMStructuredNode(adapter)
        await node.run("weather?", config, [WEATHER_TOOL], tool_choice="get_weather")
        messages, used_config, tools, tool_choice = adapter.calls[0]
        assert messages == [Message(role="user", content="weather?")]
        assert used_config is config
        assert tools == [WEATHER_TOOL]
        assert tool_choice == "get_weather"

    async def test_is_base_node_and_callable(self, config):
        adapter = RecordingAdapter(
            [ToolCallResult(tool_name="get_weather", arguments={"city": "Paris"})]
        )
        node = LLMStructuredNode(adapter)
        assert isinstance(node, BaseNode)
        result = await node("weather?", config, [WEATHER_TOOL])
        assert result.tool_calls[0].tool_name == "get_weather"

    async def test_empty_tools_raises(self, config):
        adapter = RecordingAdapter([])
        node = LLMStructuredNode(adapter)
        with pytest.raises(ValueError):
            await node.run("hi", config, [])

    async def test_multiple_tool_calls_returned_as_list(self, config):
        adapter = RecordingAdapter(
            [
                ToolCallResult(tool_name="get_weather", arguments={"city": "Paris"}),
                ToolCallResult(tool_name="search", arguments={"query": "news"}),
            ]
        )
        node = LLMStructuredNode(adapter)
        result = await node.run("do both", config, [WEATHER_TOOL, SEARCH_TOOL])
        assert [c.tool_name for c in result.tool_calls] == ["get_weather", "search"]


@pytest.mark.asyncio
class TestValidation:
    async def test_valid_args_populate_parsed(self, config):
        adapter = RecordingAdapter(
            [ToolCallResult(tool_name="get_weather", arguments={"city": "Paris"})]
        )
        node = LLMStructuredNode(adapter)
        result = await node.run("weather?", config, [WEATHER_TOOL])
        parsed = result.tool_calls[0].parsed_arguments
        assert isinstance(parsed, WeatherArgs)
        assert parsed.city == "Paris"
        assert parsed.units == "celsius"

    async def test_invalid_args_raise_tool_validation_error(self, config):
        adapter = RecordingAdapter(
            [ToolCallResult(tool_name="get_weather", arguments={"wrong": "field"})]
        )
        node = LLMStructuredNode(adapter)
        with pytest.raises(ToolValidationError) as info:
            await node.run("weather?", config, [WEATHER_TOOL])
        assert info.value.tool_name == "get_weather"
        assert info.value.retryable is True

    async def test_unknown_tool_raises(self, config):
        adapter = RecordingAdapter(
            [ToolCallResult(tool_name="mystery", arguments={})]
        )
        node = LLMStructuredNode(adapter)
        with pytest.raises(ToolValidationError) as info:
            await node.run("?", config, [WEATHER_TOOL])
        assert info.value.tool_name == "mystery"

    async def test_validate_false_keeps_raw_dicts(self, config):
        adapter = RecordingAdapter(
            [ToolCallResult(tool_name="get_weather", arguments={"wrong": "field"})]
        )
        node = LLMStructuredNode(adapter)
        result = await node.run(
            "weather?", config, [WEATHER_TOOL], validate_tool_args=False
        )
        assert result.tool_calls[0].parsed_arguments is None
        assert result.tool_calls[0].arguments == {"wrong": "field"}


@pytest.mark.asyncio
class TestCallbacks:
    async def test_on_request_includes_tools(self, config):
        adapter = RecordingAdapter(
            [ToolCallResult(tool_name="get_weather", arguments={"city": "Paris"})]
        )
        seen: list[LLMRequest] = []
        node = LLMStructuredNode(adapter, on_request=lambda r: seen.append(r))
        await node.run("weather?", config, [WEATHER_TOOL], tool_choice="get_weather")
        assert len(seen) == 1
        assert seen[0].tools == [WEATHER_TOOL]
        assert seen[0].tool_choice == "get_weather"

    async def test_on_response_receives_result(self, config):
        adapter = RecordingAdapter(
            [ToolCallResult(tool_name="get_weather", arguments={"city": "Paris"})]
        )
        seen: list[StructuredResult] = []
        node = LLMStructuredNode(adapter, on_response=lambda r: seen.append(r))
        await node.run("weather?", config, [WEATHER_TOOL])
        assert seen == [adapter.result]

    async def test_on_response_not_called_when_validation_fails(self, config):
        adapter = RecordingAdapter(
            [ToolCallResult(tool_name="get_weather", arguments={"bad": 1})]
        )
        seen: list[StructuredResult] = []
        node = LLMStructuredNode(adapter, on_response=lambda r: seen.append(r))
        with pytest.raises(ToolValidationError):
            await node.run("weather?", config, [WEATHER_TOOL])
        assert seen == []
