"""Unit tests for the Gemini adapter (Phase 6).

These tests never hit the network. The adapter's provider calls are exercised
through a fake async client injected via ``GeminiAdapter(client=...)``; the
error-mapping logic is tested directly against simple stand-in exceptions.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from rh_cognitv.nodes.llm.errors import (
    AuthenticationError,
    ContextLengthError,
    InvalidRequestError,
    LLMError,
    ProviderError,
    RateLimitError,
    TimeoutError,
)
from rh_cognitv.nodes.llm.types import LLMConfig, Message, ToolDefinition
from rh_cognitv.nodes.llm_adapters.base import (
    EmbeddingAdapter,
    StreamAdapter,
    StructuredAdapter,
    TextAdapter,
)
from rh_cognitv.nodes.llm_adapters.gemini_adapter import (
    GeminiAdapter,
    _build_config,
    _map_tool_config,
    _system_instruction,
    _to_contents,
    _to_gemini_tool,
    map_gemini_exception,
    _response_text,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeModels:
    def __init__(self, response=None, stream=None, exc=None):
        self._response = response
        self._stream = stream
        self._exc = exc
        self.last_kwargs = None

    async def generate_content(self, **kwargs):
        self.last_kwargs = kwargs
        if self._exc is not None:
            raise self._exc
        return self._response

    async def generate_content_stream(self, **kwargs):
        self.last_kwargs = kwargs
        if self._exc is not None:
            raise self._exc
        return self._stream

    async def embed_content(self, **kwargs):
        self.last_kwargs = kwargs
        if self._exc is not None:
            raise self._exc
        return self._response


class FakeClient:
    def __init__(self, response=None, stream=None, exc=None):
        self.models = FakeModels(response=response, stream=stream, exc=exc)
        self.aio = SimpleNamespace(models=self.models)


class FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        async def _gen():
            for c in self._chunks:
                yield c

        return _gen()


def make_usage(prompt=5, completion=7, total=12):
    return SimpleNamespace(
        prompt_token_count=prompt,
        candidates_token_count=completion,
        total_token_count=total,
    )


def make_text_response(text="hello", model="gemini-2.0-flash", usage=True):
    return SimpleNamespace(
        text=text,
        model_version=model,
        usage_metadata=make_usage() if usage else None,
    )


def make_chunk(text=None, model="gemini-2.0-flash", usage=None):
    return SimpleNamespace(text=text, model_version=model, usage_metadata=usage)


def make_tool_chunk(calls, model="gemini-2.0-flash", usage=None):
    return SimpleNamespace(
        text=None,
        function_calls=calls,
        model_version=model,
        usage_metadata=usage,
    )


def make_function_call(name, args, call_id="call_1"):
    return SimpleNamespace(name=name, args=args, id=call_id)


def make_structured_response(calls, model="gemini-2.0-flash"):
    return SimpleNamespace(
        function_calls=calls,
        model_version=model,
        usage_metadata=make_usage(),
    )


def make_embed_response(vectors):
    return SimpleNamespace(
        embeddings=[SimpleNamespace(values=v) for v in vectors],
        metadata=None,
    )


def make_status_exc(status, message="boom"):
    exc = Exception(message)
    exc.code = status
    exc.message = message
    return exc


@pytest.fixture
def config() -> LLMConfig:
    return LLMConfig(model="gemini-2.0-flash")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class TestHelpers:
    def test_to_contents_maps_roles(self):
        contents = _to_contents(
            [
                Message(role="user", content="hi"),
                Message(role="assistant", content="yo"),
            ]
        )
        assert contents == [
            {"role": "user", "parts": [{"text": "hi"}]},
            {"role": "model", "parts": [{"text": "yo"}]},
        ]

    def test_to_contents_strips_system(self):
        contents = _to_contents(
            [
                Message(role="system", content="be brief"),
                Message(role="user", content="hi"),
            ]
        )
        assert contents == [{"role": "user", "parts": [{"text": "hi"}]}]

    def test_system_instruction_joins(self):
        assert (
            _system_instruction(
                [
                    Message(role="system", content="a"),
                    Message(role="system", content="b"),
                    Message(role="user", content="x"),
                ]
            )
            == "a\n\nb"
        )

    def test_system_instruction_none_when_absent(self):
        assert _system_instruction([Message(role="user", content="x")]) is None

    def test_response_text_with_candidates(self):
        # Case 1: normal text part
        part1 = SimpleNamespace(text="hello", thought=False)
        content = SimpleNamespace(parts=[part1])
        candidate = SimpleNamespace(content=content)
        resp = SimpleNamespace(candidates=[candidate])
        assert _response_text(resp) == "hello"

        # Case 2: text + thought + function call parts
        part_text = SimpleNamespace(text="actual text", thought=False)
        part_thought = SimpleNamespace(text="internal thought", thought=True)
        part_func = SimpleNamespace(text=None, function_call=SimpleNamespace(name="foo", args={}))
        content = SimpleNamespace(parts=[part_text, part_thought, part_func])
        candidate = SimpleNamespace(content=content)
        resp = SimpleNamespace(candidates=[candidate])
        # It should only extract the non-thought text part
        assert _response_text(resp) == "actual text"

        # Case 3: only function call part
        content = SimpleNamespace(parts=[part_func])
        candidate = SimpleNamespace(content=content)
        resp = SimpleNamespace(candidates=[candidate])
        assert _response_text(resp) == ""


    def test_build_config_from_llm_config(self):
        cfg = LLMConfig(
            model="gemini-2.0-flash",
            temperature=0.3,
            max_tokens=50,
            top_p=0.8,
            stop=["END"],
            extra={"seed": 1},
        )
        out = _build_config([Message(role="system", content="sys")], cfg)
        assert out["temperature"] == 0.3
        assert out["max_output_tokens"] == 50
        assert out["top_p"] == 0.8
        assert out["stop_sequences"] == ["END"]
        assert out["system_instruction"] == "sys"
        assert out["seed"] == 1

    def test_build_config_omits_unset(self, config):
        out = _build_config([Message(role="user", content="x")], config)
        assert "max_output_tokens" not in out
        assert "top_p" not in out
        assert "stop_sequences" not in out
        assert "system_instruction" not in out

    def test_build_config_stop_list(self):
        cfg = LLMConfig(model="gemini-2.0-flash", stop=["A", "B"])
        out = _build_config([Message(role="user", content="x")], cfg)
        assert out["stop_sequences"] == ["A", "B"]

    def test_to_gemini_tool_schema(self):
        class WeatherArgs(BaseModel):
            city: str

        tool = ToolDefinition(
            name="get_weather", description="Get weather", parameters_model=WeatherArgs
        )
        wrapped = _to_gemini_tool([tool])
        decl = wrapped["function_declarations"][0]
        assert decl["name"] == "get_weather"
        assert decl["description"] == "Get weather"
        assert "city" in decl["parameters_json_schema"]["properties"]

    def test_map_tool_config_none_is_any(self):
        assert _map_tool_config(None) == {"function_calling_config": {"mode": "ANY"}}
        assert _map_tool_config("required") == {
            "function_calling_config": {"mode": "ANY"}
        }

    def test_map_tool_config_keywords(self):
        assert _map_tool_config("auto") == {"function_calling_config": {"mode": "AUTO"}}
        assert _map_tool_config("none") == {"function_calling_config": {"mode": "NONE"}}

    def test_map_tool_config_specific_name(self):
        assert _map_tool_config("get_weather") == {
            "function_calling_config": {
                "mode": "ANY",
                "allowed_function_names": ["get_weather"],
            }
        }


# --------------------------------------------------------------------------- #
# generate_text
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
class TestGenerateText:
    async def test_is_text_adapter(self):
        adapter = GeminiAdapter(client=FakeClient(make_text_response()))
        assert isinstance(adapter, TextAdapter)

    async def test_returns_text_result(self, config):
        adapter = GeminiAdapter(client=FakeClient(make_text_response("hi there")))
        result = await adapter.generate_text(
            [Message(role="user", content="x")], config
        )
        assert result.text == "hi there"
        assert result.meta.provider == "gemini"
        assert result.meta.model == "gemini-2.0-flash"
        assert result.meta.tokens_used.total_tokens == 12
        assert result.meta.tokens_used.prompt_tokens == 5
        assert result.meta.tokens_used.completion_tokens == 7
        assert result.meta.duration_ms >= 0

    async def test_passes_contents_and_config(self, config):
        client = FakeClient(make_text_response())
        adapter = GeminiAdapter(client=client)
        await adapter.generate_text(
            [
                Message(role="system", content="be brief"),
                Message(role="user", content="hi"),
            ],
            config,
        )
        kwargs = client.models.last_kwargs
        assert kwargs["model"] == "gemini-2.0-flash"
        assert kwargs["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
        assert kwargs["config"]["system_instruction"] == "be brief"

    async def test_missing_usage_defaults_to_zero(self, config):
        adapter = GeminiAdapter(client=FakeClient(make_text_response(usage=False)))
        result = await adapter.generate_text(
            [Message(role="user", content="x")], config
        )
        assert result.meta.tokens_used.total_tokens == 0

    async def test_text_property_raising_yields_empty(self, config):
        class Raising:
            model_version = "gemini-2.0-flash"
            usage_metadata = None

            @property
            def text(self):
                raise ValueError("no text part")

        adapter = GeminiAdapter(client=FakeClient(Raising()))
        result = await adapter.generate_text(
            [Message(role="user", content="x")], config
        )
        assert result.text == ""

    async def test_provider_exception_is_mapped(self, config):
        adapter = GeminiAdapter(client=FakeClient(exc=make_status_exc(429, "slow down")))
        with pytest.raises(RateLimitError) as info:
            await adapter.generate_text([Message(role="user", content="x")], config)
        assert info.value.__cause__ is not None


# --------------------------------------------------------------------------- #
# stream_text
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
class TestStreamText:
    async def test_is_stream_adapter(self):
        adapter = GeminiAdapter(client=FakeClient(stream=FakeStream([])))
        assert isinstance(adapter, StreamAdapter)

    async def test_yields_text_deltas(self, config):
        chunks = [make_chunk("Hello"), make_chunk(" world")]
        adapter = GeminiAdapter(client=FakeClient(stream=FakeStream(chunks)))
        out = [
            d
            async for d in adapter.stream_text(
                [Message(role="user", content="x")], config
            )
        ]
        texts = [d.text for d in out if d.text is not None]
        assert texts == ["Hello", " world"]

    async def test_final_usage_chunk_carries_tokens(self, config):
        chunks = [make_chunk("hi"), make_chunk(text=None, usage=make_usage())]
        adapter = GeminiAdapter(client=FakeClient(stream=FakeStream(chunks)))
        out = [
            d
            async for d in adapter.stream_text(
                [Message(role="user", content="x")], config
            )
        ]
        usage_deltas = [d for d in out if d.usage is not None]
        assert len(usage_deltas) == 1
        assert usage_deltas[0].usage.total_tokens == 12

    async def test_empty_chunks_skipped(self, config):
        chunks = [make_chunk(text=None, model=None, usage=None), make_chunk("hi")]
        adapter = GeminiAdapter(client=FakeClient(stream=FakeStream(chunks)))
        out = [
            d
            async for d in adapter.stream_text(
                [Message(role="user", content="x")], config
            )
        ]
        # the all-None chunk is dropped; "hi" still carries model_version
        assert all(
            d.text is not None or d.usage is not None or d.model is not None
            for d in out
        )
        assert [d.text for d in out if d.text] == ["hi"]

    async def test_stream_exception_is_mapped(self, config):
        adapter = GeminiAdapter(client=FakeClient(exc=make_status_exc(401, "bad key")))
        with pytest.raises(AuthenticationError):
            _ = [
                d
                async for d in adapter.stream_text(
                    [Message(role="user", content="x")], config
                )
            ]

    async def test_stream_yields_tool_call_deltas(self, config):
        chunks = [
            make_tool_chunk([make_function_call("get_weather", {"city": "Paris"})]),
        ]
        adapter = GeminiAdapter(client=FakeClient(stream=FakeStream(chunks)))
        out = [
            d
            async for d in adapter.stream_text(
                [Message(role="user", content="x")], config, tools=[WEATHER_TOOL]
            )
        ]
        frags = [f for d in out if d.tool_call_deltas for f in d.tool_call_deltas]
        assert len(frags) == 1
        assert frags[0].tool_name == "get_weather"
        assert frags[0].call_id == "call_1"
        assert json.loads(frags[0].arguments_delta) == {"city": "Paris"}

    async def test_stream_multiple_tool_calls_get_distinct_indices(self, config):
        chunks = [
            make_tool_chunk([make_function_call("get_weather", {"city": "Paris"})]),
            make_tool_chunk([make_function_call("get_weather", {"city": "Rome"})]),
        ]
        adapter = GeminiAdapter(client=FakeClient(stream=FakeStream(chunks)))
        out = [
            d
            async for d in adapter.stream_text(
                [Message(role="user", content="x")], config, tools=[WEATHER_TOOL]
            )
        ]
        frags = [f for d in out if d.tool_call_deltas for f in d.tool_call_deltas]
        assert [f.index for f in frags] == [0, 1]

    async def test_stream_tools_in_gen_config(self, config):
        client = FakeClient(stream=FakeStream([make_chunk("hi")]))
        adapter = GeminiAdapter(client=client)
        _ = [
            d
            async for d in adapter.stream_text(
                [Message(role="user", content="x")],
                config,
                tools=[WEATHER_TOOL],
                tool_choice="get_weather",
            )
        ]
        gen_config = client.models.last_kwargs["config"]
        assert "tools" in gen_config
        assert gen_config["tool_config"] == {
            "function_calling_config": {
                "mode": "ANY",
                "allowed_function_names": ["get_weather"],
            }
        }



# --------------------------------------------------------------------------- #
# generate_structured
# --------------------------------------------------------------------------- #
class WeatherArgs(BaseModel):
    city: str


WEATHER_TOOL = ToolDefinition(
    name="get_weather", description="Get weather", parameters_model=WeatherArgs
)


@pytest.mark.asyncio
class TestGenerateStructured:
    async def test_is_structured_adapter(self):
        adapter = GeminiAdapter(client=FakeClient(make_structured_response([])))
        assert isinstance(adapter, StructuredAdapter)

    async def test_single_tool_call(self, config):
        response = make_structured_response(
            [make_function_call("get_weather", {"city": "Paris"})]
        )
        adapter = GeminiAdapter(client=FakeClient(response))
        result = await adapter.generate_structured(
            [Message(role="user", content="x")], config, [WEATHER_TOOL]
        )
        assert len(result.tool_calls) == 1
        call = result.tool_calls[0]
        assert call.tool_name == "get_weather"
        assert call.arguments == {"city": "Paris"}
        assert call.call_id == "call_1"
        assert call.parsed_arguments is None
        assert result.meta.provider == "gemini"
        assert result.meta.tokens_used.total_tokens == 12

    async def test_multiple_tool_calls(self, config):
        response = make_structured_response(
            [
                make_function_call("get_weather", {"city": "Paris"}, "c1"),
                make_function_call("get_weather", {"city": "Rome"}, "c2"),
            ]
        )
        adapter = GeminiAdapter(client=FakeClient(response))
        result = await adapter.generate_structured(
            [Message(role="user", content="x")], config, [WEATHER_TOOL]
        )
        assert [c.arguments["city"] for c in result.tool_calls] == ["Paris", "Rome"]

    async def test_no_tool_calls_returns_empty(self, config):
        adapter = GeminiAdapter(client=FakeClient(make_structured_response(None)))
        result = await adapter.generate_structured(
            [Message(role="user", content="x")], config, [WEATHER_TOOL]
        )
        assert result.tool_calls == []

    async def test_passes_tools_and_tool_config(self, config):
        client = FakeClient(make_structured_response([]))
        adapter = GeminiAdapter(client=client)
        await adapter.generate_structured(
            [Message(role="user", content="x")], config, [WEATHER_TOOL]
        )
        cfg = client.models.last_kwargs["config"]
        assert cfg["tools"][0]["function_declarations"][0]["name"] == "get_weather"
        assert cfg["tool_config"] == {"function_calling_config": {"mode": "ANY"}}

    async def test_tool_choice_specific(self, config):
        client = FakeClient(make_structured_response([]))
        adapter = GeminiAdapter(client=client)
        await adapter.generate_structured(
            [Message(role="user", content="x")],
            config,
            [WEATHER_TOOL],
            tool_choice="get_weather",
        )
        cfg = client.models.last_kwargs["config"]
        assert cfg["tool_config"]["function_calling_config"]["allowed_function_names"] == [
            "get_weather"
        ]

    async def test_structured_exception_is_mapped(self, config):
        adapter = GeminiAdapter(client=FakeClient(exc=make_status_exc(500, "oops")))
        with pytest.raises(ProviderError):
            await adapter.generate_structured(
                [Message(role="user", content="x")], config, [WEATHER_TOOL]
            )


# --------------------------------------------------------------------------- #
# embed
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
class TestEmbed:
    async def test_is_embedding_adapter(self):
        adapter = GeminiAdapter(client=FakeClient(make_embed_response([[0.1]])))
        assert isinstance(adapter, EmbeddingAdapter)

    async def test_returns_embeddings(self):
        config = LLMConfig(model="text-embedding-004")
        response = make_embed_response([[0.1, 0.2], [0.3, 0.4]])
        adapter = GeminiAdapter(client=FakeClient(response))
        result = await adapter.embed(["a", "b"], config)
        assert result.embeddings == [[0.1, 0.2], [0.3, 0.4]]
        assert result.meta.provider == "gemini"
        assert result.meta.model == "text-embedding-004"

    async def test_passes_contents(self):
        config = LLMConfig(model="text-embedding-004")
        client = FakeClient(make_embed_response([[0.1]]))
        adapter = GeminiAdapter(client=client)
        await adapter.embed(["hello"], config)
        kwargs = client.models.last_kwargs
        assert kwargs["model"] == "text-embedding-004"
        assert kwargs["contents"] == ["hello"]

    async def test_embed_exception_is_mapped(self):
        config = LLMConfig(model="text-embedding-004")
        adapter = GeminiAdapter(client=FakeClient(exc=make_status_exc(429, "slow")))
        with pytest.raises(RateLimitError):
            await adapter.embed(["a"], config)


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
        err = map_gemini_exception(make_status_exc(status))
        assert isinstance(err, expected)
        assert err.provider == "gemini"
        assert err.status_code == status

    def test_context_length_by_message(self):
        err = map_gemini_exception(
            make_status_exc(400, "The input token count exceeds the maximum number of tokens")
        )
        assert isinstance(err, ContextLengthError)

    def test_timeout_by_class_name(self):
        class DeadlineExceeded(Exception):
            pass

        err = map_gemini_exception(DeadlineExceeded("timed out"))
        assert isinstance(err, TimeoutError)

    def test_connection_error_by_class_name(self):
        class ConnectionError(Exception):
            pass

        err = map_gemini_exception(ConnectionError("no route"))
        assert isinstance(err, ProviderError)
        assert err.retryable is True

    def test_unknown_status_falls_back_to_llmerror(self):
        err = map_gemini_exception(make_status_exc(418))
        assert type(err) is LLMError

    def test_message_attr_preferred(self):
        err = map_gemini_exception(make_status_exc(429, "rate limited friend"))
        assert "rate limited friend" in str(err)


class TestConstruction:
    def test_injected_client_used_directly(self):
        client = FakeClient(make_text_response())
        adapter = GeminiAdapter(client=client)
        assert adapter._client is client
        assert adapter.provider == "gemini"
