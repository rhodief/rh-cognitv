"""Unit tests for the canonical LLM I/O models (spec §4)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from rh_cognitv.nodes.llm.types import (
    EmbeddingResult,
    LLMConfig,
    LLMResultMeta,
    Message,
    StreamDelta,
    StreamResult,
    StructuredResult,
    TextResult,
    TokenUsage,
    ToolCallResult,
    ToolDefinition,
    normalize_prompt,
)


def make_meta(**overrides) -> LLMResultMeta:
    base = dict(
        model="gpt-test",
        provider="openai",
        tokens_used=TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        duration_ms=12.5,
    )
    base.update(overrides)
    return LLMResultMeta(**base)


class TestMessage:
    def test_minimal(self):
        msg = Message(role="user", content="hi")
        assert msg.role == "user"
        assert msg.content == "hi"
        assert msg.name is None

    def test_tool_message_with_name(self):
        msg = Message(role="tool", content="result", name="search")
        assert msg.name == "search"

    def test_invalid_role_rejected(self):
        with pytest.raises(ValidationError):
            Message(role="robot", content="hi")  # type: ignore[arg-type]

    def test_roundtrip_serialization(self):
        msg = Message(role="system", content="be nice")
        assert Message.model_validate(msg.model_dump()) == msg


class TestLLMConfig:
    def test_defaults(self):
        cfg = LLMConfig(model="gpt-test")
        assert cfg.temperature == 1.0
        assert cfg.max_tokens is None
        assert cfg.top_p is None
        assert cfg.stop is None
        assert cfg.extra == {}

    def test_extra_is_independent_per_instance(self):
        a = LLMConfig(model="m")
        b = LLMConfig(model="m")
        a.extra["k"] = "v"
        assert b.extra == {}

    def test_full(self):
        cfg = LLMConfig(
            model="m",
            temperature=0.2,
            max_tokens=100,
            top_p=0.9,
            stop=["\n"],
            extra={"seed": 42},
        )
        assert cfg.stop == ["\n"]
        assert cfg.extra["seed"] == 42


class TestTokenUsage:
    def test_defaults_zero(self):
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0


class TestLLMResultMeta:
    def test_roundtrip(self):
        meta = make_meta()
        assert LLMResultMeta.model_validate(meta.model_dump()) == meta

    def test_raw_response_allows_arbitrary(self):
        sentinel = object()
        meta = make_meta(raw_response=sentinel)
        assert meta.raw_response is sentinel


class TestResultModels:
    def test_text_result(self):
        res = TextResult(text="hello", meta=make_meta())
        assert res.text == "hello"
        assert res.meta.provider == "openai"

    def test_stream_delta_text_only(self):
        delta = StreamDelta(text="tok")
        assert delta.text == "tok"
        assert delta.object_fragment is None

    def test_stream_result(self):
        res = StreamResult(text="full", object={"a": 1}, meta=make_meta())
        assert res.object == {"a": 1}

    def test_embedding_result(self):
        res = EmbeddingResult(embeddings=[[0.1, 0.2], [0.3, 0.4]], meta=make_meta())
        assert len(res.embeddings) == 2
        assert res.embeddings[0] == [0.1, 0.2]


class _Args(BaseModel):
    city: str


class TestStructured:
    def test_tool_definition_holds_model_type(self):
        td = ToolDefinition(
            name="get_weather",
            description="Get weather",
            parameters_model=_Args,
        )
        assert td.parameters_model is _Args

    def test_tool_call_result_with_parsed(self):
        parsed = _Args(city="Paris")
        tc = ToolCallResult(
            tool_name="get_weather",
            arguments={"city": "Paris"},
            parsed_arguments=parsed,
            call_id="call_1",
        )
        assert tc.parsed_arguments is parsed
        assert tc.arguments == {"city": "Paris"}

    def test_tool_call_result_parsed_optional(self):
        tc = ToolCallResult(tool_name="t", arguments={"city": "X"})
        assert tc.parsed_arguments is None
        assert tc.call_id is None

    def test_structured_result_always_list(self):
        res = StructuredResult(
            tool_calls=[ToolCallResult(tool_name="t", arguments={})],
            meta=make_meta(),
        )
        assert isinstance(res.tool_calls, list)
        assert res.tool_calls[0].tool_name == "t"


class TestNormalizePrompt:
    def test_str_wraps_as_user_message(self):
        msgs = normalize_prompt("hello")
        assert msgs == [Message(role="user", content="hello")]

    def test_list_passes_through(self):
        original = [
            Message(role="system", content="sys"),
            Message(role="user", content="hi"),
        ]
        msgs = normalize_prompt(original)
        assert msgs == original

    def test_list_is_copied(self):
        original = [Message(role="user", content="hi")]
        msgs = normalize_prompt(original)
        msgs.append(Message(role="user", content="more"))
        assert len(original) == 1
