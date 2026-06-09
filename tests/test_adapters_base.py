"""Unit tests for the abstract adapter interfaces and BaseNode (DD-01/02/08)."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from rh_cognitv.nodes.base import BaseNode
from rh_cognitv.nodes.llm.types import (
    EmbeddingResult,
    LLMResultMeta,
    StreamDelta,
    StructuredResult,
    TextResult,
    TokenUsage,
)
from rh_cognitv.nodes.llm_adapters.base import (
    EmbeddingAdapter,
    StreamAdapter,
    StructuredAdapter,
    TextAdapter,
)


def make_meta(provider: str = "fake") -> LLMResultMeta:
    return LLMResultMeta(
        model="m",
        provider=provider,
        tokens_used=TokenUsage(),
        duration_ms=0.0,
    )


class TestAbstractContracts:
    @pytest.mark.parametrize(
        "cls", [TextAdapter, StreamAdapter, StructuredAdapter, EmbeddingAdapter]
    )
    def test_cannot_instantiate_abstract(self, cls):
        with pytest.raises(TypeError):
            cls()  # type: ignore[abstract]

    def test_missing_method_still_abstract(self):
        class Incomplete(TextAdapter):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]


class TestMultipleInheritance:
    def test_one_class_implements_all_capabilities(self):
        class FakeAdapter(TextAdapter, StreamAdapter, StructuredAdapter, EmbeddingAdapter):
            async def generate_text(self, messages, config):
                return TextResult(text="ok", meta=make_meta())

            def stream_text(self, messages, config) -> AsyncIterator[StreamDelta]:
                async def _gen():
                    yield StreamDelta(text="ok")

                return _gen()

            async def generate_structured(self, messages, config, tools, tool_choice=None):
                return StructuredResult(tool_calls=[], meta=make_meta())

            async def embed(self, texts, config):
                return EmbeddingResult(embeddings=[[0.0]], meta=make_meta())

        adapter = FakeAdapter()
        assert isinstance(adapter, TextAdapter)
        assert isinstance(adapter, StreamAdapter)
        assert isinstance(adapter, StructuredAdapter)
        assert isinstance(adapter, EmbeddingAdapter)

    def test_embedding_only_adapter(self):
        class EmbedOnly(EmbeddingAdapter):
            async def embed(self, texts, config):
                return EmbeddingResult(embeddings=[[1.0]], meta=make_meta())

        adapter = EmbedOnly()
        assert isinstance(adapter, EmbeddingAdapter)
        assert not isinstance(adapter, TextAdapter)


@pytest.mark.asyncio
class TestBaseNode:
    async def test_call_delegates_to_run(self):
        class EchoNode(BaseNode[str]):
            async def run(self, value: str) -> str:
                return value.upper()

        node = EchoNode()
        assert await node.run("hi") == "HI"
        assert await node("hi") == "HI"

    async def test_call_forwards_args_and_kwargs(self):
        class AddNode(BaseNode[int]):
            async def run(self, a: int, b: int = 0) -> int:
                return a + b

        node = AddNode()
        assert await node(2, b=3) == 5


class TestBaseNodeAbstract:
    def test_abstract_run_blocks_instantiation(self):
        class Bad(BaseNode):  # type: ignore[type-arg]
            pass

        with pytest.raises(TypeError):
            Bad()
