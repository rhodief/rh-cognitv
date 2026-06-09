"""Unit tests for the canonical LLM error taxonomy (DD-09, SG-05)."""

from __future__ import annotations

import pytest

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


class TestErrorDefaults:
    @pytest.mark.parametrize(
        ("cls", "family", "code", "retryable"),
        [
            (RateLimitError, LLMErrorFamily.RATE_LIMIT, "rate_limit", True),
            (AuthenticationError, LLMErrorFamily.AUTHENTICATION, "authentication", False),
            (InvalidRequestError, LLMErrorFamily.INVALID_REQUEST, "invalid_request", False),
            (ContextLengthError, LLMErrorFamily.CONTEXT_LENGTH, "context_length_exceeded", False),
            (TimeoutError, LLMErrorFamily.TIMEOUT, "timeout", True),
            (ProviderError, LLMErrorFamily.PROVIDER, "provider_error", True),
            (ToolValidationError, LLMErrorFamily.VALIDATION, "tool_validation", True),
        ],
    )
    def test_class_defaults(self, cls, family, code, retryable):
        err = cls("boom")
        assert err.family is family
        assert err.code == code
        assert err.retryable is retryable
        assert err.message == "boom"

    def test_all_subclass_llmerror(self):
        for cls in (
            RateLimitError,
            AuthenticationError,
            InvalidRequestError,
            ContextLengthError,
            TimeoutError,
            ProviderError,
            ToolValidationError,
        ):
            assert issubclass(cls, LLMError)

    def test_context_length_is_invalid_request(self):
        assert issubclass(ContextLengthError, InvalidRequestError)


class TestErrorOverrides:
    def test_per_instance_overrides(self):
        err = LLMError(
            "custom",
            family=LLMErrorFamily.PROVIDER,
            code="weird",
            retryable=True,
            provider="openai",
            status_code=503,
        )
        assert err.family is LLMErrorFamily.PROVIDER
        assert err.code == "weird"
        assert err.retryable is True
        assert err.provider == "openai"
        assert err.status_code == 503

    def test_base_defaults_unknown(self):
        err = LLMError("x")
        assert err.family is LLMErrorFamily.UNKNOWN
        assert err.code == "llm_error"
        assert err.retryable is False
        assert err.provider is None
        assert err.status_code is None

    def test_is_raisable_and_caught_as_exception(self):
        with pytest.raises(LLMError):
            raise RateLimitError("slow down")

    def test_cause_preserved(self):
        original = ValueError("orig")
        try:
            try:
                raise original
            except ValueError as exc:
                raise ProviderError("wrapped") from exc
        except ProviderError as err:
            assert err.__cause__ is original


class TestToolValidationError:
    def test_tool_name_captured(self):
        err = ToolValidationError("bad args", tool_name="get_weather")
        assert err.tool_name == "get_weather"
        assert err.retryable is True

    def test_tool_name_optional(self):
        err = ToolValidationError("bad args")
        assert err.tool_name is None


class TestStatusMapping:
    @pytest.mark.parametrize(
        ("status", "family"),
        [
            (429, LLMErrorFamily.RATE_LIMIT),
            (401, LLMErrorFamily.AUTHENTICATION),
            (403, LLMErrorFamily.AUTHENTICATION),
            (408, LLMErrorFamily.TIMEOUT),
            (400, LLMErrorFamily.INVALID_REQUEST),
            (404, LLMErrorFamily.INVALID_REQUEST),
            (422, LLMErrorFamily.INVALID_REQUEST),
            (500, LLMErrorFamily.PROVIDER),
            (503, LLMErrorFamily.PROVIDER),
            (418, LLMErrorFamily.UNKNOWN),
            (200, LLMErrorFamily.UNKNOWN),
        ],
    )
    def test_map_http_status(self, status, family):
        assert map_http_status_to_error_family(status) is family
