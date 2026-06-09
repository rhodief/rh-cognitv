"""Opt-in integration test for the OpenAI text path.

Skipped automatically when ``OPENAI_API_KEY`` is not set. Run explicitly with::

    pytest -m integration

These tests make real network calls and may incur cost.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from rh_cognitv.nodes.llm.stream_node import LLMStreamNode
from rh_cognitv.nodes.llm.structured_node import LLMStructuredNode
from rh_cognitv.nodes.llm.text_node import LLMTextNode
from rh_cognitv.nodes.llm.events import StreamCompleted, StreamStarted
from rh_cognitv.nodes.llm.types import LLMConfig, TextResult, ToolDefinition
from rh_cognitv.nodes.llm_adapters.openai_adapter import OpenAIAdapter

from .conftest import requires_openai

pytestmark = [pytest.mark.integration, requires_openai]


@pytest.mark.asyncio
async def test_openai_text_node_real_call():
    node = LLMTextNode(OpenAIAdapter())
    config = LLMConfig(model="gpt-4o-mini", temperature=0.0, max_tokens=16)
    result = await node.run(
        "Reply with exactly the word: pong", config
    )

    assert isinstance(result, TextResult)
    assert result.text.strip() != ""
    assert result.meta.provider == "openai"
    assert result.meta.tokens_used.total_tokens > 0
    assert result.meta.duration_ms > 0


@pytest.mark.asyncio
async def test_openai_stream_node_real_call():
    node = LLMStreamNode(OpenAIAdapter())
    config = LLMConfig(model="gpt-4o-mini", temperature=0.0, max_tokens=32)

    events = [
        e async for e in node.run("Count from 1 to 5, space separated.", config)
    ]

    assert isinstance(events[0], StreamStarted)
    completed = events[-1]
    assert isinstance(completed, StreamCompleted)
    assert completed.text.strip() != ""
    assert completed.meta.provider == "openai"
    assert completed.meta.tokens_used.total_tokens > 0


class _WeatherArgs(BaseModel):
    city: str


@pytest.mark.asyncio
async def test_openai_structured_node_real_call():
    tool = ToolDefinition(
        name="get_weather",
        description="Get the current weather for a city",
        parameters_model=_WeatherArgs,
    )
    node = LLMStructuredNode(OpenAIAdapter())
    config = LLMConfig(model="gpt-4o-mini", temperature=0.0)

    result = await node.run(
        "What's the weather in Paris?", config, [tool], tool_choice="get_weather"
    )

    assert len(result.tool_calls) >= 1
    call = result.tool_calls[0]
    assert call.tool_name == "get_weather"
    assert isinstance(call.parsed_arguments, _WeatherArgs)
    assert call.parsed_arguments.city.strip() != ""
    assert result.meta.provider == "openai"
