"""Unit tests for OpenAIAdapter.embed (Phase 5). No network."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rh_cognitv.nodes.llm.errors import AuthenticationError
from rh_cognitv.nodes.llm.types import LLMConfig
from rh_cognitv.nodes.llm_adapters.base import EmbeddingAdapter
from rh_cognitv.nodes.llm_adapters.openai_adapter import OpenAIAdapter


def make_response(vectors, model="text-embedding-3-small", indices=None):
    indices = indices if indices is not None else list(range(len(vectors)))
    data = [
        SimpleNamespace(embedding=vec, index=idx)
        for vec, idx in zip(vectors, indices)
    ]
    usage = SimpleNamespace(prompt_tokens=8, total_tokens=8)
    return SimpleNamespace(data=data, model=model, usage=usage)


class FakeEmbeddings:
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
        self.embeddings = FakeEmbeddings(response=response, exc=exc)


@pytest.fixture
def config() -> LLMConfig:
    return LLMConfig(model="text-embedding-3-small")


@pytest.mark.asyncio
class TestEmbed:
    async def test_is_embedding_adapter(self):
        adapter = OpenAIAdapter(client=FakeClient(make_response([[0.1]])))
        assert isinstance(adapter, EmbeddingAdapter)

    async def test_single_text(self, config):
        adapter = OpenAIAdapter(client=FakeClient(make_response([[0.1, 0.2]])))
        result = await adapter.embed(["hi"], config)
        assert result.embeddings == [[0.1, 0.2]]
        assert result.meta.provider == "openai"
        assert result.meta.model == "text-embedding-3-small"
        assert result.meta.tokens_used.total_tokens == 8

    async def test_batch_texts(self, config):
        adapter = OpenAIAdapter(
            client=FakeClient(make_response([[0.1], [0.2], [0.3]]))
        )
        result = await adapter.embed(["a", "b", "c"], config)
        assert result.embeddings == [[0.1], [0.2], [0.3]]

    async def test_results_sorted_by_index(self, config):
        # Provider returns out-of-order; adapter must realign by index.
        adapter = OpenAIAdapter(
            client=FakeClient(
                make_response([[0.3], [0.1], [0.2]], indices=[2, 0, 1])
            )
        )
        result = await adapter.embed(["a", "b", "c"], config)
        assert result.embeddings == [[0.1], [0.2], [0.3]]

    async def test_payload_uses_input_and_model(self, config):
        client = FakeClient(make_response([[0.1]]))
        adapter = OpenAIAdapter(client=client)
        await adapter.embed(["hi"], config)
        kwargs = client.embeddings.last_kwargs
        assert kwargs["model"] == "text-embedding-3-small"
        assert kwargs["input"] == ["hi"]

    async def test_extra_config_passed_through(self):
        client = FakeClient(make_response([[0.1]]))
        adapter = OpenAIAdapter(client=client)
        config = LLMConfig(model="text-embedding-3-small", extra={"dimensions": 256})
        await adapter.embed(["hi"], config)
        assert client.embeddings.last_kwargs["dimensions"] == 256

    async def test_error_is_mapped(self, config):
        exc = Exception("nope")
        exc.status_code = 401
        adapter = OpenAIAdapter(client=FakeClient(exc=exc))
        with pytest.raises(AuthenticationError):
            await adapter.embed(["hi"], config)
