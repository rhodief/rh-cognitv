"""Opt-in integration tests for the Gemini adapter.

Skipped automatically when ``GEMINI_API_KEY`` is not set. Run explicitly with::

    pytest -m integration

These tests make real network calls and may incur cost. They also verify the
spec exit criterion that the *same* node classes work with a different adapter
(GeminiAdapter) swapped in for OpenAIAdapter.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from rh_cognitv.nodes.llm.embedding_node import LLMEmbeddingNode
from rh_cognitv.nodes.llm.events import StreamCompleted, StreamStarted
from rh_cognitv.nodes.llm.stream_node import LLMStreamNode
from rh_cognitv.nodes.llm.structured_node import LLMStructuredNode
from rh_cognitv.nodes.llm.text_node import LLMTextNode
from rh_cognitv.nodes.llm.types import LLMConfig, TextResult, ToolDefinition
from rh_cognitv.nodes.llm_adapters.gemini_adapter import GeminiAdapter

from .conftest import requires_gemini

pytestmark = [pytest.mark.integration, requires_gemini]

_CHAT_MODEL = "gemini-2.5-flash"
_EMBED_MODEL = "gemini-embedding-001"


@pytest.mark.asyncio
async def test_gemini_text_node_real_call():
    node = LLMTextNode(GeminiAdapter())
    config = LLMConfig(model=_CHAT_MODEL, temperature=0.0, max_tokens=512)
    result = await node.run("Reply with exactly the word: pong", config)

    assert isinstance(result, TextResult)
    assert result.text.strip() != ""
    assert result.meta.provider == "gemini"
    assert result.meta.tokens_used.total_tokens > 0
    assert result.meta.duration_ms > 0


@pytest.mark.asyncio
async def test_gemini_stream_node_real_call():
    node = LLMStreamNode(GeminiAdapter())
    config = LLMConfig(model=_CHAT_MODEL, temperature=0.0, max_tokens=512)

    events = [e async for e in node.run("Count from 1 to 5, space separated.", config)]

    assert isinstance(events[0], StreamStarted)
    completed = events[-1]
    assert isinstance(completed, StreamCompleted)
    assert completed.text.strip() != ""
    assert completed.meta.provider == "gemini"


class _WeatherArgs(BaseModel):
    city: str


@pytest.mark.asyncio
async def test_gemini_structured_node_real_call():
    tool = ToolDefinition(
        name="get_weather",
        description="Get the current weather for a city",
        parameters_model=_WeatherArgs,
    )
    node = LLMStructuredNode(GeminiAdapter())
    config = LLMConfig(model=_CHAT_MODEL, temperature=0.0)

    result = await node.run(
        "What's the weather in Paris?", config, [tool], tool_choice="get_weather"
    )

    assert len(result.tool_calls) >= 1
    call = result.tool_calls[0]
    assert call.tool_name == "get_weather"
    assert isinstance(call.parsed_arguments, _WeatherArgs)
    assert call.parsed_arguments.city.strip() != ""
    assert result.meta.provider == "gemini"


@pytest.mark.asyncio
async def test_gemini_embedding_node_real_call():
    node = LLMEmbeddingNode(GeminiAdapter())
    config = LLMConfig(model=_EMBED_MODEL)

    result = await node.run(["hello world", "goodbye world"], config)

    assert len(result.embeddings) == 2
    assert all(len(vec) > 0 for vec in result.embeddings)
    assert all(isinstance(x, float) for x in result.embeddings[0])
    assert result.meta.provider == "gemini"
