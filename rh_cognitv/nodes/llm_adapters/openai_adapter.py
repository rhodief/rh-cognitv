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
    LLMConfig,
    LLMResultMeta,
    Message,
    StreamDelta,
    TextResult,
    TokenUsage,
)
from rh_cognitv.nodes.llm_adapters.base import StreamAdapter, TextAdapter

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


class OpenAIAdapter(TextAdapter, StreamAdapter):
    """OpenAI-backed adapter.

    Implements :class:`TextAdapter` (Phase 2) and :class:`StreamAdapter`
    (Phase 3).
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
    ) -> AsyncIterator[StreamDelta]:
        payload = _build_payload(messages, config)
        payload["stream"] = True
        payload.setdefault("stream_options", {"include_usage": True})

        try:
            stream = await self._client.chat.completions.create(**payload)
            async for chunk in stream:
                text: str | None = None
                choices = getattr(chunk, "choices", None) or []
                if choices:
                    delta = getattr(choices[0], "delta", None)
                    text = getattr(delta, "content", None) if delta is not None else None

                raw_usage = getattr(chunk, "usage", None)
                usage = _token_usage(raw_usage) if raw_usage is not None else None
                model = getattr(chunk, "model", None)

                if text is None and usage is None and model is None:
                    continue
                yield StreamDelta(text=text, usage=usage, model=model)
        except Exception as exc:
            raise map_openai_exception(exc) from exc
