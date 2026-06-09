"""Canonical LLM error taxonomy (DD-09, SG-05).

Each adapter maps provider-specific exceptions onto these canonical errors so
that calling code — and a future retry/runtime engine — can react to a stable,
structured contract instead of provider SDK internals.

Every error carries:

- ``family`` — a coarse-grained classification (:class:`LLMErrorFamily`).
- ``code`` — a stable, machine-readable string.
- ``message`` — a human-readable description.
- ``retryable`` — whether a runtime may retry the operation.
- ``provider`` — the originating provider, when known.
- ``status_code`` — the HTTP status code, when applicable.
- ``__cause__`` — the original exception, preserved via ``raise ... from``.
"""

from __future__ import annotations

from enum import Enum


class LLMErrorFamily(str, Enum):
    """Coarse-grained classification of LLM errors."""

    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    INVALID_REQUEST = "invalid_request"
    CONTEXT_LENGTH = "context_length"
    TIMEOUT = "timeout"
    PROVIDER = "provider"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


class LLMError(Exception):
    """Base class for all canonical LLM errors.

    Subclasses set sensible class-level defaults for ``family``, ``code`` and
    ``retryable``; any of these may be overridden per-instance.
    """

    family: LLMErrorFamily = LLMErrorFamily.UNKNOWN
    code: str = "llm_error"
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        family: LLMErrorFamily | None = None,
        code: str | None = None,
        retryable: bool | None = None,
        provider: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if family is not None:
            self.family = family
        if code is not None:
            self.code = code
        if retryable is not None:
            self.retryable = retryable
        self.provider = provider
        self.status_code = status_code

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"{type(self).__name__}(family={self.family.value!r}, "
            f"code={self.code!r}, retryable={self.retryable}, "
            f"provider={self.provider!r}, status_code={self.status_code!r}, "
            f"message={self.message!r})"
        )


class RateLimitError(LLMError):
    """The provider rejected the request due to rate limiting (HTTP 429)."""

    family = LLMErrorFamily.RATE_LIMIT
    code = "rate_limit"
    retryable = True


class AuthenticationError(LLMError):
    """Authentication or authorization failed (HTTP 401/403)."""

    family = LLMErrorFamily.AUTHENTICATION
    code = "authentication"
    retryable = False


class InvalidRequestError(LLMError):
    """The request was malformed or otherwise rejected (HTTP 400/404/422)."""

    family = LLMErrorFamily.INVALID_REQUEST
    code = "invalid_request"
    retryable = False


class ContextLengthError(InvalidRequestError):
    """The prompt exceeded the model's maximum context length."""

    family = LLMErrorFamily.CONTEXT_LENGTH
    code = "context_length_exceeded"
    retryable = False


class TimeoutError(LLMError):
    """The request timed out (HTTP 408 or a client-side timeout)."""

    family = LLMErrorFamily.TIMEOUT
    code = "timeout"
    retryable = True


class ProviderError(LLMError):
    """The provider returned a server-side error (HTTP 5xx)."""

    family = LLMErrorFamily.PROVIDER
    code = "provider_error"
    retryable = True


class ToolValidationError(LLMError):
    """The model's tool-call arguments failed schema validation (SG-05).

    Retryable by default: a future runtime can re-prompt the model with the
    validation error message.
    """

    family = LLMErrorFamily.VALIDATION
    code = "tool_validation"
    retryable = True

    def __init__(
        self,
        message: str,
        *,
        tool_name: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(message, **kwargs)  # type: ignore[arg-type]
        self.tool_name = tool_name


def map_http_status_to_error_family(status_code: int) -> LLMErrorFamily:
    """Map an HTTP status code to a canonical :class:`LLMErrorFamily`.

    Shared by adapters to reduce duplication in their error-mapping logic
    (DD-09).
    """
    if status_code == 429:
        return LLMErrorFamily.RATE_LIMIT
    if status_code in (401, 403):
        return LLMErrorFamily.AUTHENTICATION
    if status_code == 408:
        return LLMErrorFamily.TIMEOUT
    if status_code in (400, 404, 422):
        return LLMErrorFamily.INVALID_REQUEST
    if 500 <= status_code < 600:
        return LLMErrorFamily.PROVIDER
    return LLMErrorFamily.UNKNOWN
