"""Canonical Pydantic I/O models shared across all LLM nodes (spec §4).

These models form the provider-agnostic contract: inputs (``Message``,
``LLMConfig``, ``ToolDefinition``), outputs (``TextResult``, ``StreamResult``,
``StructuredResult``, ``EmbeddingResult``), and the metadata carried by every
result (``LLMResultMeta``, ``TokenUsage``).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool"]


# --------------------------------------------------------------------------- #
# Shared
# --------------------------------------------------------------------------- #
class Message(BaseModel):
    """A single chat message (DD-10)."""

    role: Role
    content: str
    name: str | None = None  # for tool messages


class LLMConfig(BaseModel):
    """Provider-agnostic generation configuration."""

    model: str
    temperature: float = 1.0
    max_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    extra: dict[str, Any] = Field(default_factory=dict)  # provider pass-through


class LLMRequest(BaseModel):
    """Canonical request payload delivered to ``on_request`` callbacks (SG-01).

    ``tools`` / ``tool_choice`` are populated only for tool-calling requests
    (``LLMStructuredNode``); they are ``None`` for plain text/stream requests.
    """

    messages: list[Message]
    config: LLMConfig
    tools: list["ToolDefinition"] | None = None
    tool_choice: str | None = None


class TokenUsage(BaseModel):
    """Token accounting for a single LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResultMeta(BaseModel):
    """Metadata attached to every canonical result object."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str
    provider: str
    tokens_used: TokenUsage
    duration_ms: float
    raw_response: Any | None = None  # optional, for debugging


# --------------------------------------------------------------------------- #
# Stream
# --------------------------------------------------------------------------- #
class StreamDelta(BaseModel):
    """An incremental chunk produced during streaming.

    ``text`` / ``object_fragment`` carry incremental content. ``usage`` and
    ``model`` are optional trailing metadata that providers attach to the final
    chunk (e.g. OpenAI with ``stream_options={"include_usage": True}``); the
    node uses them to build the consolidated :class:`StreamResult` meta.
    """

    text: str | None = None
    object_fragment: dict[str, Any] | None = None
    usage: TokenUsage | None = None
    model: str | None = None


class StreamResult(BaseModel):
    """Consolidated result produced after a stream completes."""

    text: str
    object: dict[str, Any] | None = None
    meta: LLMResultMeta


# --------------------------------------------------------------------------- #
# Text
# --------------------------------------------------------------------------- #
class TextResult(BaseModel):
    """Result of a single-shot (non-streaming) completion."""

    text: str
    meta: LLMResultMeta


# --------------------------------------------------------------------------- #
# Structured (tool calling)
# --------------------------------------------------------------------------- #
class ToolDefinition(BaseModel):
    """Explicit wrapper describing a callable tool (DD-06)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    parameters_model: type[BaseModel]  # Pydantic model for the args schema


class ToolCallResult(BaseModel):
    """A single tool call returned by the model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool_name: str
    arguments: dict[str, Any]  # raw arguments
    parsed_arguments: BaseModel | None = None  # validated instance (SG-05)
    call_id: str | None = None


class StructuredResult(BaseModel):
    """Result of a tool-calling invocation. ``tool_calls`` is always a list (SG-04)."""

    tool_calls: list[ToolCallResult]
    meta: LLMResultMeta


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
class EmbeddingRequest(BaseModel):
    """Canonical embedding request delivered to ``on_request`` callbacks (SG-01)."""

    texts: list[str]
    config: LLMConfig


class EmbeddingResult(BaseModel):
    """Result of a batch text→embedding call."""

    embeddings: list[list[float]]
    meta: LLMResultMeta


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def normalize_prompt(prompt: str | list[Message]) -> list[Message]:
    """Normalize a prompt into a list of ``Message`` instances (DD-10).

    A bare ``str`` is wrapped as a single user message; a list is returned
    as-is (after a defensive copy).
    """
    if isinstance(prompt, str):
        return [Message(role="user", content=prompt)]
    return list(prompt)


# ``LLMRequest`` forward-references ``ToolDefinition`` (defined above); resolve it.
LLMRequest.model_rebuild()
