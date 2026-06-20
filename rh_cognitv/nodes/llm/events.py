"""Canonical stream event models (DD-04).

``LLMStreamNode.run()`` yields these events as an ``AsyncGenerator``. They are
also delivered to the optional ``on_event`` callback for secondary consumers
(logging, monitoring, a future EventBus).

The events form a discriminated union keyed on the ``type`` field, so consumers
can ``match`` on it or rely on Pydantic's tagged-union parsing.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field

from rh_cognitv.nodes.llm.types import (
    LLMResultMeta,
    StreamToolCallDelta,
    ToolCallResult,
)


class StreamStarted(BaseModel):
    """Emitted once, before any deltas, when the stream begins."""

    type: Literal["stream_started"] = "stream_started"
    model: str
    provider: str


class StreamTextDelta(BaseModel):
    """Emitted for each (possibly batched) chunk of streamed content."""

    type: Literal["stream_delta"] = "stream_delta"
    text: str | None = None
    object_fragment: dict[str, Any] | None = None
    tool_call_deltas: list[StreamToolCallDelta] | None = None
    index: int = 0  # sequence number of this emitted (post-batch) event


class StreamThinkingDelta(BaseModel):
    """Emitted for each chunk of model thinking/reasoning content.

    Only produced by models that support extended thinking (e.g. Claude,
    Gemini thinking mode). Distinct from :class:`StreamTextDelta` which
    carries regular text output.
    """

    type: Literal["stream_thinking_delta"] = "stream_thinking_delta"
    text: str
    index: int = 0


class StreamCompleted(BaseModel):
    """Emitted once, after the final delta, with consolidated metadata."""

    type: Literal["stream_completed"] = "stream_completed"
    text: str
    thinking: str | None = None
    object: dict[str, Any] | None = None
    tool_calls: list[ToolCallResult] = Field(default_factory=list)
    meta: LLMResultMeta


class StreamErrorEvent(BaseModel):
    """Emitted if the stream fails mid-flight."""

    type: Literal["stream_error"] = "stream_error"
    family: str
    code: str
    message: str
    retryable: bool


StreamEvent = Annotated[
    Union[StreamStarted, StreamTextDelta, StreamThinkingDelta, StreamCompleted, StreamErrorEvent],
    Field(discriminator="type"),
]
"""Discriminated union of all canonical stream events."""
