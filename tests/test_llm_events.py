"""Unit tests for canonical stream event models (DD-04)."""

from __future__ import annotations

from pydantic import BaseModel, TypeAdapter

from rh_cognitv.nodes.llm.events import (
    StreamCompleted,
    StreamErrorEvent,
    StreamEvent,
    StreamStarted,
    StreamTextDelta,
)
from rh_cognitv.nodes.llm.types import LLMResultMeta, TokenUsage

_event_adapter: TypeAdapter[StreamEvent] = TypeAdapter(StreamEvent)


def make_meta() -> LLMResultMeta:
    return LLMResultMeta(
        model="gpt-test",
        provider="openai",
        tokens_used=TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        duration_ms=5.0,
    )


class TestEventDefaults:
    def test_started_type(self):
        ev = StreamStarted(model="m", provider="openai")
        assert ev.type == "stream_started"

    def test_delta_defaults(self):
        ev = StreamTextDelta(text="hi")
        assert ev.type == "stream_delta"
        assert ev.index == 0
        assert ev.object_fragment is None

    def test_completed(self):
        ev = StreamCompleted(text="done", object={"a": 1}, meta=make_meta())
        assert ev.type == "stream_completed"
        assert ev.object == {"a": 1}

    def test_error_event(self):
        ev = StreamErrorEvent(
            family="rate_limit", code="rate_limit", message="slow", retryable=True
        )
        assert ev.type == "stream_error"
        assert ev.retryable is True


class TestDiscriminatedUnion:
    def test_parses_each_variant_by_tag(self):
        cases = [
            ({"type": "stream_started", "model": "m", "provider": "p"}, StreamStarted),
            ({"type": "stream_delta", "text": "x"}, StreamTextDelta),
            (
                {
                    "type": "stream_completed",
                    "text": "t",
                    "meta": make_meta().model_dump(),
                },
                StreamCompleted,
            ),
            (
                {
                    "type": "stream_error",
                    "family": "timeout",
                    "code": "timeout",
                    "message": "m",
                    "retryable": True,
                },
                StreamErrorEvent,
            ),
        ]
        for payload, expected_cls in cases:
            parsed = _event_adapter.validate_python(payload)
            assert isinstance(parsed, expected_cls)

    def test_serialization_includes_tag(self):
        ev = StreamTextDelta(text="hi", index=3)
        dumped = ev.model_dump()
        assert dumped["type"] == "stream_delta"
        assert dumped["index"] == 3
