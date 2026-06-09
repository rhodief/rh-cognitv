"""Google Gemini provider adapter.

Translates the canonical node contract to the ``google-genai`` SDK and maps
Gemini exceptions onto the canonical :mod:`rh_cognitv.nodes.llm.errors`
taxonomy (DD-09).

The ``google-genai`` SDK is an *optional* dependency (SG-03): it is imported
lazily inside the constructor, and a clear install hint is raised if it is
missing. A pre-built ``client`` may be injected (e.g. for testing) to bypass the
import.

Per DD-02, this single ``GeminiAdapter`` class is the place where every
Gemini-backed capability lives. Gemini supports text, streaming, structured
(function-calling), and embeddings, so it implements all four ABCs.

All requests are built as plain dicts (``ContentDict`` /
``GenerateContentConfigDict``), which the SDK accepts natively. This keeps the
adapter constructible and unit-testable with an injected fake client without
importing the SDK types.
"""

from __future__ import annotations

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
    map_http_status_to_error_family,
)
from rh_cognitv.nodes.llm.types import (
    EmbeddingResult,
    LLMConfig,
    LLMResultMeta,
    Message,
    StreamDelta,
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

PROVIDER = "gemini"

# Gemini does not use a stable machine code for "input too long"; detect it from
# the human-readable message instead.
_CONTEXT_LENGTH_MARKERS = (
    "exceeds the maximum number of tokens",
    "input token count",
    "context length",
    "too many tokens",
    "request payload size exceeds",
)

# Gemini uses "model" rather than "assistant" for the model's turn; system
# prompts are hoisted out of the message list into ``system_instruction``.
_ROLE_MAP = {
    "user": "user",
    "assistant": "model",
    "model": "model",
    "tool": "user",
}


def _is_context_length(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _CONTEXT_LENGTH_MARKERS)


def map_gemini_exception(exc: Exception) -> LLMError:
    """Map a ``google-genai`` SDK exception onto a canonical :class:`LLMError`.

    Gemini's :class:`google.genai.errors.APIError` carries the HTTP status on a
    ``code`` attribute and the message on ``message``. We duck-type both so the
    mapping works without importing the SDK and is robust across versions.
    """
    message = getattr(exc, "message", None) or str(exc) or type(exc).__name__
    status = getattr(exc, "code", None)

    if not isinstance(status, int):
        name = type(exc).__name__
        if "Timeout" in name or "Deadline" in name:
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
        if _is_context_length(message):
            return ContextLengthError(message, provider=PROVIDER, status_code=status)
        return InvalidRequestError(message, provider=PROVIDER, status_code=status)
    if family is LLMErrorFamily.PROVIDER:
        return ProviderError(message, provider=PROVIDER, status_code=status)
    return LLMError(message, provider=PROVIDER, status_code=status)


def _to_contents(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert canonical messages to Gemini ``contents`` (system stripped out)."""
    contents: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            continue
        role = _ROLE_MAP.get(message.role, "user")
        contents.append({"role": role, "parts": [{"text": message.content}]})
    return contents


def _system_instruction(messages: list[Message]) -> str | None:
    systems = [m.content for m in messages if m.role == "system"]
    if not systems:
        return None
    return "\n\n".join(systems)


def _build_config(messages: list[Message], config: LLMConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {"temperature": config.temperature}
    system = _system_instruction(messages)
    if system is not None:
        payload["system_instruction"] = system
    if config.max_tokens is not None:
        payload["max_output_tokens"] = config.max_tokens
    if config.top_p is not None:
        payload["top_p"] = config.top_p
    if config.stop is not None:
        payload["stop_sequences"] = list(config.stop)
    payload.update(config.extra)
    return payload


def _token_usage(usage: Any) -> TokenUsage:
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
        completion_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        total_tokens=getattr(usage, "total_token_count", 0) or 0,
    )


def _to_gemini_tool(tools: list[ToolDefinition]) -> dict[str, Any]:
    """Wrap canonical tools into a single Gemini ``Tool`` declaration set."""
    return {
        "function_declarations": [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters_json_schema": tool.parameters_model.model_json_schema(),
            }
            for tool in tools
        ]
    }


def _map_tool_config(tool_choice: str | None) -> dict[str, Any]:
    """Map canonical ``tool_choice`` to Gemini's ``function_calling_config`` (DD-07).

    ``None`` → ``ANY`` (the node exists to call a tool; the model picks which).
    ``"auto"`` → ``AUTO``; ``"none"`` → ``NONE``; ``"required"`` → ``ANY``; any
    other string forces that specific function by name.
    """
    if tool_choice is None or tool_choice == "required":
        return {"function_calling_config": {"mode": "ANY"}}
    if tool_choice == "auto":
        return {"function_calling_config": {"mode": "AUTO"}}
    if tool_choice == "none":
        return {"function_calling_config": {"mode": "NONE"}}
    return {
        "function_calling_config": {
            "mode": "ANY",
            "allowed_function_names": [tool_choice],
        }
    }


def _response_text(response: Any) -> str:
    """Safely read ``response.text`` (the property raises when no text parts)."""
    try:
        return response.text or ""
    except Exception:
        return ""


class GeminiAdapter(TextAdapter, StreamAdapter, StructuredAdapter, EmbeddingAdapter):
    """Google Gemini-backed adapter.

    Implements :class:`TextAdapter`, :class:`StreamAdapter`,
    :class:`StructuredAdapter`, and :class:`EmbeddingAdapter`.
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
            from google import genai
        except ImportError as exc:  # pragma: no cover - exercised when SDK absent
            raise ImportError(
                "Install rh_cognitv[gemini] to use the GeminiAdapter"
            ) from exc
        if api_key is not None:
            client_kwargs.setdefault("api_key", api_key)
        self._client = genai.Client(**client_kwargs)

    async def generate_text(
        self,
        messages: list[Message],
        config: LLMConfig,
    ) -> TextResult:
        contents = _to_contents(messages)
        gen_config = _build_config(messages, config)

        start = time.perf_counter()
        try:
            response = await self._client.aio.models.generate_content(
                model=config.model,
                contents=contents,
                config=gen_config,
            )
        except Exception as exc:
            raise map_gemini_exception(exc) from exc
        duration_ms = (time.perf_counter() - start) * 1000

        meta = LLMResultMeta(
            model=getattr(response, "model_version", None) or config.model,
            provider=PROVIDER,
            tokens_used=_token_usage(getattr(response, "usage_metadata", None)),
            duration_ms=duration_ms,
            raw_response=response,
        )
        return TextResult(text=_response_text(response), meta=meta)

    async def stream_text(
        self,
        messages: list[Message],
        config: LLMConfig,
    ) -> AsyncIterator[StreamDelta]:
        contents = _to_contents(messages)
        gen_config = _build_config(messages, config)

        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=config.model,
                contents=contents,
                config=gen_config,
            )
            async for chunk in stream:
                text = _response_text(chunk) or None

                raw_usage = getattr(chunk, "usage_metadata", None)
                usage = _token_usage(raw_usage) if raw_usage is not None else None
                model = getattr(chunk, "model_version", None)

                if text is None and usage is None and model is None:
                    continue
                yield StreamDelta(text=text, usage=usage, model=model)
        except Exception as exc:
            raise map_gemini_exception(exc) from exc

    async def generate_structured(
        self,
        messages: list[Message],
        config: LLMConfig,
        tools: list[ToolDefinition],
        tool_choice: str | None = None,
    ) -> StructuredResult:
        contents = _to_contents(messages)
        gen_config = _build_config(messages, config)
        gen_config["tools"] = [_to_gemini_tool(tools)]
        gen_config["tool_config"] = _map_tool_config(tool_choice)

        start = time.perf_counter()
        try:
            response = await self._client.aio.models.generate_content(
                model=config.model,
                contents=contents,
                config=gen_config,
            )
        except Exception as exc:
            raise map_gemini_exception(exc) from exc
        duration_ms = (time.perf_counter() - start) * 1000

        raw_calls = getattr(response, "function_calls", None) or []
        tool_calls: list[ToolCallResult] = []
        for raw in raw_calls:
            tool_calls.append(
                ToolCallResult(
                    tool_name=raw.name,
                    arguments=dict(raw.args or {}),
                    call_id=getattr(raw, "id", None),
                )
            )

        meta = LLMResultMeta(
            model=getattr(response, "model_version", None) or config.model,
            provider=PROVIDER,
            tokens_used=_token_usage(getattr(response, "usage_metadata", None)),
            duration_ms=duration_ms,
            raw_response=response,
        )
        return StructuredResult(tool_calls=tool_calls, meta=meta)

    async def embed(
        self,
        texts: list[str],
        config: LLMConfig,
    ) -> EmbeddingResult:
        embed_config: dict[str, Any] = dict(config.extra)

        start = time.perf_counter()
        try:
            response = await self._client.aio.models.embed_content(
                model=config.model,
                contents=texts,
                config=embed_config or None,
            )
        except Exception as exc:
            raise map_gemini_exception(exc) from exc
        duration_ms = (time.perf_counter() - start) * 1000

        embeddings = [list(item.values) for item in (response.embeddings or [])]
        meta = LLMResultMeta(
            model=config.model,
            provider=PROVIDER,
            tokens_used=TokenUsage(),
            duration_ms=duration_ms,
            raw_response=response,
        )
        return EmbeddingResult(embeddings=embeddings, meta=meta)
