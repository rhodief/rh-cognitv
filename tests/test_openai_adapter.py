"""Unit tests for the OpenAI adapter (Phase 2).

These tests never hit the network. The adapter's provider call is exercised
through a fake async client injected via ``OpenAIAdapter(client=...)``; the
error-mapping logic is tested directly against simple stand-in exceptions.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rh_cognitv.nodes.llm.errors import (
    AuthenticationError,
    ContextLengthError,
    InvalidRequestError,
    LLMError,
    ProviderError,
    RateLimitError,
    TimeoutError,
)
from rh_cognitv.nodes.llm.types import LLMConfig, Message
from rh_cognitv.nodes.llm_adapters.openai_adapter import (
    OpenAIAdapter,
    map_openai_exception,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
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


class FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeClient:
    def __init__(self, response=None, exc=None):
        self.completions = FakeCompletions(response=response, exc=exc)
        self.chat = FakeChat(self.completions)


def make_response(content="hello", model="gpt-4o-mini", usage=True):
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    usage_obj = (
        SimpleNamespace(prompt_tokens=5, completion_tokens=7, total_tokens=12)
        if usage
        else None
    )
    return SimpleNamespace(choices=[choice], model=model, usage=usage_obj)


def make_status_exc(status, message="boom", code=None):
    exc = Exception(message)
    exc.status_code = status
    if code is not None:
        exc.code = code
    return exc


@pytest.fixture
def config() -> LLMConfig:
    return LLMConfig(model="gpt-4o-mini")


# --------------------------------------------------------------------------- #
# generate_text
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
class TestGenerateText:
    async def test_returns_text_result(self, config):
        client = FakeClient(response=make_response("hi there"))
        adapter = OpenAIAdapter(client=client)
        result = await adapter.generate_text([Message(role="user", content="x")], config)
        assert result.text == "hi there"
        assert result.meta.provider == "openai"
        assert result.meta.model == "gpt-4o-mini"
        assert result.meta.tokens_used.total_tokens == 12
        assert result.meta.duration_ms >= 0

    async def test_builds_payload_from_config(self):
        client = FakeClient(response=make_response())
        adapter = OpenAIAdapter(client=client)
        config = LLMConfig(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=50,
            top_p=0.8,
            stop=["END"],
            extra={"seed": 1},
        )
        await adapter.generate_text([Message(role="user", content="hi")], config)
        kwargs = client.completions.last_kwargs
        assert kwargs["model"] == "gpt-4o-mini"
        assert kwargs["temperature"] == 0.3
        assert kwargs["max_tokens"] == 50
        assert kwargs["top_p"] == 0.8
        assert kwargs["stop"] == ["END"]
        assert kwargs["seed"] == 1
        assert kwargs["messages"] == [{"role": "user", "content": "hi"}]

    async def test_omits_unset_optional_params(self, config):
        client = FakeClient(response=make_response())
        adapter = OpenAIAdapter(client=client)
        await adapter.generate_text([Message(role="user", content="hi")], config)
        kwargs = client.completions.last_kwargs
        assert "max_tokens" not in kwargs
        assert "top_p" not in kwargs
        assert "stop" not in kwargs

    async def test_tool_message_includes_name(self, config):
        client = FakeClient(response=make_response())
        adapter = OpenAIAdapter(client=client)
        await adapter.generate_text(
            [Message(role="tool", content="r", name="search")], config
        )
        msg = client.completions.last_kwargs["messages"][0]
        assert msg == {"role": "tool", "content": "r", "name": "search"}

    async def test_none_content_becomes_empty_string(self, config):
        client = FakeClient(response=make_response(content=None))
        adapter = OpenAIAdapter(client=client)
        result = await adapter.generate_text([Message(role="user", content="x")], config)
        assert result.text == ""

    async def test_missing_usage_defaults_to_zero(self, config):
        client = FakeClient(response=make_response(usage=False))
        adapter = OpenAIAdapter(client=client)
        result = await adapter.generate_text([Message(role="user", content="x")], config)
        assert result.meta.tokens_used.total_tokens == 0

    async def test_provider_exception_is_mapped(self, config):
        client = FakeClient(exc=make_status_exc(429, "slow down"))
        adapter = OpenAIAdapter(client=client)
        with pytest.raises(RateLimitError) as info:
            await adapter.generate_text([Message(role="user", content="x")], config)
        assert info.value.__cause__ is not None


# --------------------------------------------------------------------------- #
# error mapping
# --------------------------------------------------------------------------- #
class TestErrorMapping:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (429, RateLimitError),
            (401, AuthenticationError),
            (403, AuthenticationError),
            (408, TimeoutError),
            (400, InvalidRequestError),
            (404, InvalidRequestError),
            (422, InvalidRequestError),
            (500, ProviderError),
            (503, ProviderError),
        ],
    )
    def test_status_codes(self, status, expected):
        err = map_openai_exception(make_status_exc(status))
        assert isinstance(err, expected)
        assert err.provider == "openai"
        assert err.status_code == status

    def test_context_length_by_code(self):
        err = map_openai_exception(
            make_status_exc(400, "too long", code="context_length_exceeded")
        )
        assert isinstance(err, ContextLengthError)

    def test_context_length_by_message(self):
        err = map_openai_exception(
            make_status_exc(400, "This model's maximum context length is 8192 tokens")
        )
        assert isinstance(err, ContextLengthError)

    def test_timeout_by_class_name(self):
        class APITimeoutError(Exception):
            pass

        err = map_openai_exception(APITimeoutError("timed out"))
        assert isinstance(err, TimeoutError)

    def test_connection_error_by_class_name(self):
        class APIConnectionError(Exception):
            pass

        err = map_openai_exception(APIConnectionError("no route"))
        assert isinstance(err, ProviderError)
        assert err.retryable is True

    def test_unknown_status_falls_back_to_llmerror(self):
        err = map_openai_exception(make_status_exc(418))
        assert type(err) is LLMError

    def test_code_extracted_from_body_dict(self):
        exc = make_status_exc(400, "bad")
        exc.body = {"error": {"code": "context_length_exceeded"}}
        err = map_openai_exception(exc)
        assert isinstance(err, ContextLengthError)


class TestConstruction:
    def test_injected_client_used_directly(self):
        client = FakeClient(response=make_response())
        adapter = OpenAIAdapter(client=client)
        assert adapter._client is client
