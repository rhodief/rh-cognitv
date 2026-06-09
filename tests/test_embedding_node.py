"""Unit tests for LLMEmbeddingNode (Phase 5)."""

from __future__ import annotations

import pytest

from rh_cognitv.nodes.base import BaseNode
from rh_cognitv.nodes.llm.embedding_node import LLMEmbeddingNode
from rh_cognitv.nodes.llm.types import (
    EmbeddingRequest,
    EmbeddingResult,
    LLMConfig,
    LLMResultMeta,
    TokenUsage,
)
from rh_cognitv.nodes.llm_adapters.base import EmbeddingAdapter


def make_result(embeddings) -> EmbeddingResult:
    return EmbeddingResult(
        embeddings=embeddings,
        meta=LLMResultMeta(
            model="text-embedding-test",
            provider="fake",
            tokens_used=TokenUsage(prompt_tokens=3, completion_tokens=0, total_tokens=3),
            duration_ms=1.0,
        ),
    )


class RecordingAdapter(EmbeddingAdapter):
    """A fake EmbeddingAdapter recording the call and returning canned vectors."""

    provider = "fake"

    def __init__(self, embeddings):
        self.result = make_result(embeddings)
        self.calls: list[tuple[list[str], LLMConfig]] = []

    async def embed(self, texts, config):
        self.calls.append((texts, config))
        return self.result


@pytest.fixture
def config() -> LLMConfig:
    return LLMConfig(model="text-embedding-test")


@pytest.mark.asyncio
class TestRun:
    async def test_returns_embedding_result(self, config):
        adapter = RecordingAdapter([[0.1, 0.2], [0.3, 0.4]])
        node = LLMEmbeddingNode(adapter)
        result = await node.run(["a", "b"], config)
        assert isinstance(result, EmbeddingResult)
        assert result.embeddings == [[0.1, 0.2], [0.3, 0.4]]

    async def test_single_string_wrapped_as_batch(self, config):
        adapter = RecordingAdapter([[0.1, 0.2]])
        node = LLMEmbeddingNode(adapter)
        await node.run("hello", config)
        texts, used_config = adapter.calls[0]
        assert texts == ["hello"]
        assert used_config is config

    async def test_list_passed_through(self, config):
        adapter = RecordingAdapter([[0.1], [0.2], [0.3]])
        node = LLMEmbeddingNode(adapter)
        await node.run(["x", "y", "z"], config)
        texts, _ = adapter.calls[0]
        assert texts == ["x", "y", "z"]

    async def test_empty_list_raises(self, config):
        adapter = RecordingAdapter([])
        node = LLMEmbeddingNode(adapter)
        with pytest.raises(ValueError):
            await node.run([], config)

    async def test_is_base_node_and_callable(self, config):
        adapter = RecordingAdapter([[0.1]])
        node = LLMEmbeddingNode(adapter)
        assert isinstance(node, BaseNode)
        result = await node("hi", config)
        assert result is adapter.result

    async def test_token_usage_tracked(self, config):
        adapter = RecordingAdapter([[0.1, 0.2]])
        node = LLMEmbeddingNode(adapter)
        result = await node.run("hi", config)
        assert result.meta.tokens_used.total_tokens == 3


@pytest.mark.asyncio
class TestCallbacks:
    async def test_on_request_receives_embedding_request(self, config):
        adapter = RecordingAdapter([[0.1]])
        seen: list[EmbeddingRequest] = []
        node = LLMEmbeddingNode(adapter, on_request=lambda r: seen.append(r))
        await node.run(["a"], config)
        assert len(seen) == 1
        assert isinstance(seen[0], EmbeddingRequest)
        assert seen[0].texts == ["a"]
        assert seen[0].config is config

    async def test_on_response_receives_result(self, config):
        adapter = RecordingAdapter([[0.1]])
        seen: list[EmbeddingResult] = []
        node = LLMEmbeddingNode(adapter, on_response=lambda r: seen.append(r))
        await node.run("a", config)
        assert seen == [adapter.result]

    async def test_async_callbacks_awaited_in_order(self, config):
        order: list[str] = []

        async def on_request(_req):
            order.append("request")

        async def on_response(_res):
            order.append("response")

        node = LLMEmbeddingNode(
            RecordingAdapter([[0.1]]), on_request=on_request, on_response=on_response
        )
        await node.run("a", config)
        assert order == ["request", "response"]
