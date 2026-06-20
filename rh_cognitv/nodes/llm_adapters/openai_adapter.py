"""OpenAI provider adapter.

Translates the canonical node contract to the OpenAI Python SDK and maps
OpenAI exceptions onto the canonical :mod:`rh_cognitv.nodes.llm.errors`
taxonomy (DD-09).

The ``openai`` SDK is an *optional* dependency (SG-03): it is imported lazily
inside the constructor, and a clear install hint is raised if it is missing. A
pre-built ``client`` may be injected (e.g. for testing) to bypass the import.

Per DD-02, this single ``OpenAIAdapter`` class is the place where every
OpenAI-backed capability lives. Phase 2 implements :class:`TextAdapter`; later
phases extend it with the stream, structured, and embedding interfaces.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from rh_cognitv.nodes.llm.errors import (
    AuthenticationError,
    ContextLengthError,
    InvalidRequestError,
    LLMError,
    LLMErrorFamily,
    ProviderError,
    RateLimitError,
    TimeoutError,
    ToolValidationError,
    map_http_status_to_error_family,
)
from rh_cognitv.nodes.llm.types import (
    EmbeddingResult,
    LLMConfig,
    LLMResultMeta,
    Message,
    StreamDelta,
    StreamToolCallDelta,
    StructuredResult,
    TextResult,
    TokenUsage,
    ToolCallResult,
    ToolDefinition,
)
from rh_cognitv.nodes.llm_adapters.base import (
    EmbeddingAdapter,
    StreamAdapter,
    StructuredAdapter,
    TextAdapter,
)

PROVIDER = "openai"

_CONTEXT_LENGTH_MARKERS = ("context_length_exceeded", "maximum context length", "context length")


def _extract_error_code(exc: Exception) -> str | None:
    """Best-effort extraction of a provider error code from an exception."""
    code = getattr(exc, "code", None)
    if isinstance(code, str):
        return code
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and isinstance(error.get("code"), str):
            return error["code"]
        if isinstance(body.get("code"), str):
            return body["code"]
    return None


def _is_context_length(message: str, code: str | None) -> bool:
    if code == "context_length_exceeded":
        return True
    lowered = message.lower()
    return any(marker in lowered for marker in _CONTEXT_LENGTH_MARKERS)


def map_openai_exception(exc: Exception) -> LLMError:
    """Map an OpenAI SDK exception onto a canonical :class:`LLMError` (DD-09).

    Uses duck typing (``status_code`` attribute and class name) rather than
    ``isinstance`` checks, so it works without importing the SDK and is robust
    across SDK versions.
    """
    message = str(exc) or type(exc).__name__
    code = _extract_error_code(exc)
    status = getattr(exc, "status_code", None)

    if not isinstance(status, int):
        name = type(exc).__name__
        if "Timeout" in name:
            return TimeoutError(message, provider=PROVIDER)
        if "Connection" in name:
            return ProviderError(message, provider=PROVIDER, retryable=True)
        return LLMError(message, provider=PROVIDER)

    family = map_http_status_to_error_family(status)
    if family is LLMErrorFamily.RATE_LIMIT:
        return RateLimitError(message, provider=PROVIDER, status_code=status)
    if family is LLMErrorFamily.AUTHENTICATION:
        return AuthenticationError(message, provider=PROVIDER, status_code=status)
    if family is LLMErrorFamily.TIMEOUT:
        return TimeoutError(message, provider=PROVIDER, status_code=status)
    if family is LLMErrorFamily.INVALID_REQUEST:
        if _is_context_length(message, code):
            return ContextLengthError(message, provider=PROVIDER, status_code=status)
        return InvalidRequestError(message, provider=PROVIDER, status_code=status)
    if family is LLMErrorFamily.PROVIDER:
        return ProviderError(message, provider=PROVIDER, status_code=status)
    return LLMError(message, provider=PROVIDER, status_code=status)


def _to_openai_message(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.name is not None:
        payload["name"] = message.name
    return payload


def _build_payload(messages: list[Message], config: LLMConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": [_to_openai_message(m) for m in messages],
        "temperature": config.temperature,
    }
    if config.max_tokens is not None:
        payload["max_tokens"] = config.max_tokens
    if config.top_p is not None:
        payload["top_p"] = config.top_p
    if config.stop is not None:
        payload["stop"] = config.stop
    payload.update(config.extra)
    return payload


def _token_usage(usage: Any) -> TokenUsage:
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        total_tokens=getattr(usage, "total_tokens", 0) or 0,
    )


def _to_openai_tool(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters_model.model_json_schema(),
        },
    }


def _map_tool_choice(tool_choice: str | None) -> str | dict[str, Any]:
    """Map canonical ``tool_choice`` to OpenAI's native format (DD-07).

    ``None`` → ``"required"`` (the node exists to call a tool; the model picks
    which). The keywords ``"auto"`` / ``"required"`` / ``"none"`` pass through;
    any other string forces that specific tool by name.
    """
    if tool_choice is None or tool_choice == "required":
        return "required"
    if tool_choice in ("auto", "none"):
        return tool_choice
    return {"type": "function", "function": {"name": tool_choice}}


def _stream_tool_call_deltas(delta: Any) -> list[StreamToolCallDelta] | None:
    """Extract streamed tool-call fragments from an OpenAI delta.

    OpenAI streams tool calls incrementally: the ``id`` / ``name`` arrive on the
    first fragment for a given ``index`` and the ``arguments`` arrive as partial
    JSON strings on subsequent fragments.
    """
    raw_calls = getattr(delta, "tool_calls", None)
    if not raw_calls:
        return None
    fragments: list[StreamToolCallDelta] = []
    for raw in raw_calls:
        function = getattr(raw, "function", None)
        fragments.append(
            StreamToolCallDelta(
                index=getattr(raw, "index", 0) or 0,
                call_id=getattr(raw, "id", None),
                tool_name=getattr(function, "name", None) if function else None,
                arguments_delta=getattr(function, "arguments", None)
                if function
                else None,
            )
        )
    return fragments


class OpenAIAdapter(TextAdapter, StreamAdapter, StructuredAdapter, EmbeddingAdapter):
    """OpenAI-backed adapter.

    Implements :class:`TextAdapter` (Phase 2), :class:`StreamAdapter`
    (Phase 3), :class:`StructuredAdapter` (Phase 4), and
    :class:`EmbeddingAdapter` (Phase 5).
    """

    provider = PROVIDER

    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: Any | None = None,
        **client_kwargs: Any,
    ) -> None:
        if client is not None:
            self._client = client
            return
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - exercised when SDK absent
            raise ImportError(
                "Install rh_cognitv[openai] to use the OpenAIAdapter"
            ) from exc
        self._client = AsyncOpenAI(api_key=api_key, **client_kwargs)

    async def generate_text(
        self,
        messages: list[Message],
        config: LLMConfig,
    ) -> TextResult:
        payload = _build_payload(messages, config)

        start = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(**payload)
        except Exception as exc:
            raise map_openai_exception(exc) from exc
        duration_ms = (time.perf_counter() - start) * 1000

        choice = response.choices[0]
        text = choice.message.content or ""
        meta = LLMResultMeta(
            model=getattr(response, "model", config.model),
            provider=PROVIDER,
            tokens_used=_token_usage(getattr(response, "usage", None)),
            duration_ms=duration_ms,
            raw_response=response,
        )
        return TextResult(text=text, meta=meta)

    async def stream_text(
        self,
        messages: list[Message],
        config: LLMConfig,
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | None = None,
    ) -> AsyncIterator[StreamDelta]:
        payload = _build_payload(messages, config)
        payload["stream"] = True
        payload.setdefault("stream_options", {"include_usage": True})
        if tools:
            payload["tools"] = [_to_openai_tool(t) for t in tools]
            payload["tool_choice"] = _map_tool_choice(tool_choice)

        try:
            stream = await self._client.chat.completions.create(**payload)
            async for chunk in stream:
                text: str | None = None
                thinking: str | None = None
                tool_call_deltas: list[StreamToolCallDelta] | None = None
                choices = getattr(chunk, "choices", None) or []
                if choices:
                    delta = getattr(choices[0], "delta", None)
                    if delta is not None:
                        text = getattr(delta, "content", None)
                        # OpenAI-compatible APIs (including Anthropic via
                        # OpenAI compat layer) may expose model thinking /
                        # reasoning via a `reasoning_content` field.
                        thinking = getattr(delta, "reasoning_content", None)
                        tool_call_deltas = _stream_tool_call_deltas(delta)

                raw_usage = getattr(chunk, "usage", None)
                usage = _token_usage(raw_usage) if raw_usage is not None else None
                model = getattr(chunk, "model", None)

                if (
                    text is None
                    and thinking is None
                    and tool_call_deltas is None
                    and usage is None
                    and model is None
                ):
                    continue
                yield StreamDelta(
                    text=text,
                    thinking=thinking,
                    tool_call_deltas=tool_call_deltas,
                    usage=usage,
                    model=model,
                )
        except Exception as exc:
            raise map_openai_exception(exc) from exc

    async def generate_structured(
        self,
        messages: list[Message],
        config: LLMConfig,
        tools: list[ToolDefinition],
        tool_choice: str | None = None,
    ) -> StructuredResult:
        payload = _build_payload(messages, config)
        payload["tools"] = [_to_openai_tool(t) for t in tools]
        payload["tool_choice"] = _map_tool_choice(tool_choice)

        start = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(**payload)
        except Exception as exc:
            raise map_openai_exception(exc) from exc
        duration_ms = (time.perf_counter() - start) * 1000

        message = response.choices[0].message
        raw_calls = getattr(message, "tool_calls", None) or []
        tool_calls: list[ToolCallResult] = []
        for raw in raw_calls:
            function = raw.function
            name = function.name
            raw_args = function.arguments or "{}"
            try:
                arguments = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                raise ToolValidationError(
                    f"Tool {name!r} returned non-JSON arguments: {raw_args!r}",
                    tool_name=name,
                    provider=PROVIDER,
                ) from exc
            tool_calls.append(
                ToolCallResult(
                    tool_name=name,
                    arguments=arguments,
                    call_id=getattr(raw, "id", None),
                )
            )

        meta = LLMResultMeta(
            model=getattr(response, "model", config.model),
            provider=PROVIDER,
            tokens_used=_token_usage(getattr(response, "usage", None)),
            duration_ms=duration_ms,
            raw_response=response,
        )
        return StructuredResult(tool_calls=tool_calls, meta=meta)

    async def embed(
        self,
        texts: list[str],
        config: LLMConfig,
    ) -> EmbeddingResult:
        payload: dict[str, Any] = {"model": config.model, "input": texts}
        payload.update(config.extra)

        start = time.perf_counter()
        try:
            response = await self._client.embeddings.create(**payload)
        except Exception as exc:
            raise map_openai_exception(exc) from exc
        duration_ms = (time.perf_counter() - start) * 1000

        items = sorted(response.data, key=lambda d: getattr(d, "index", 0))
        embeddings = [list(item.embedding) for item in items]
        meta = LLMResultMeta(
            model=getattr(response, "model", config.model),
            provider=PROVIDER,
            tokens_used=_token_usage(getattr(response, "usage", None)),
            duration_ms=duration_ms,
            raw_response=response,
        )
        return EmbeddingResult(embeddings=embeddings, meta=meta)

