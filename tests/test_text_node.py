"""Unit tests for LLMTextNode (Phase 2)."""

from __future__ import annotations

import pytest

from rh_cognitv.nodes.base import BaseNode
from rh_cognitv.nodes.llm.text_node import LLMTextNode
from rh_cognitv.nodes.llm.types import (
    LLMConfig,
    LLMRequest,
    LLMResultMeta,
    Message,
    TextResult,
    TokenUsage,
)
from rh_cognitv.nodes.llm_adapters.base import TextAdapter


def make_result(text: str = "hello") -> TextResult:
    return TextResult(
        text=text,
        meta=LLMResultMeta(
            model="gpt-test",
            provider="fake",
            tokens_used=TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
            duration_ms=1.0,
        ),
    )


class RecordingAdapter(TextAdapter):
    """A fake TextAdapter that records the call and returns a canned result."""

    def __init__(self, result: TextResult | None = None) -> None:
        self.result = result or make_result()
        self.calls: list[tuple[list[Message], LLMConfig]] = []

    async def generate_text(self, messages, config):
        self.calls.append((messages, config))
        return self.result


@pytest.fixture
def config() -> LLMConfig:
    return LLMConfig(model="gpt-test")


@pytest.mark.asyncio
class TestRun:
    async def test_returns_adapter_result(self, config):
        adapter = RecordingAdapter()
        node = LLMTextNode(adapter)
        result = await node.run("hi", config)
        assert result is adapter.result

    async def test_string_prompt_normalized_to_user_message(self, config):
        adapter = RecordingAdapter()
        node = LLMTextNode(adapter)
        await node.run("hello there", config)
        messages, used_config = adapter.calls[0]
        assert messages == [Message(role="user", content="hello there")]
        assert used_config is config

    async def test_message_list_passed_through(self, config):
        adapter = RecordingAdapter()
        node = LLMTextNode(adapter)
        prompt = [
            Message(role="system", content="be nice"),
            Message(role="user", content="hi"),
        ]
        await node.run(prompt, config)
        messages, _ = adapter.calls[0]
        assert messages == prompt

    async def test_is_base_node_and_callable(self, config):
        adapter = RecordingAdapter()
        node = LLMTextNode(adapter)
        assert isinstance(node, BaseNode)
        result = await node("hi", config)
        assert result is adapter.result


@pytest.mark.asyncio
class TestCallbacks:
    async def test_on_request_receives_llm_request(self, config):
        seen: list[LLMRequest] = []
        node = LLMTextNode(RecordingAdapter(), on_request=lambda r: seen.append(r))
        await node.run("hi", config)
        assert len(seen) == 1
        assert isinstance(seen[0], LLMRequest)
        assert seen[0].messages == [Message(role="user", content="hi")]
        assert seen[0].config is config

    async def test_on_response_receives_result(self, config):
        adapter = RecordingAdapter()
        seen: list[TextResult] = []
        node = LLMTextNode(adapter, on_response=lambda r: seen.append(r))
        await node.run("hi", config)
        assert seen == [adapter.result]

    async def test_async_callbacks_awaited(self, config):
        order: list[str] = []

        async def on_request(_req):
            order.append("request")

        async def on_response(_res):
            order.append("response")

        node = LLMTextNode(
            RecordingAdapter(), on_request=on_request, on_response=on_response
        )
        await node.run("hi", config)
        assert order == ["request", "response"]

    async def test_on_request_runs_before_adapter(self, config):
        order: list[str] = []

        class OrderAdapter(TextAdapter):
            async def generate_text(self, messages, config):
                order.append("adapter")
                return make_result()

        node = LLMTextNode(
            OrderAdapter(), on_request=lambda _r: order.append("request")
        )
        await node.run("hi", config)
        assert order == ["request", "adapter"]
