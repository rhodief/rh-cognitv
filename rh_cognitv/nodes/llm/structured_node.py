"""LLMStructuredNode — tool-calling / function-calling completion (Phase 4).

Accepts a list of :class:`~rh_cognitv.nodes.llm.types.ToolDefinition` wrappers
(DD-06), delegates to an injected
:class:`~rh_cognitv.nodes.llm_adapters.base.StructuredAdapter`, and returns a
canonical :class:`~rh_cognitv.nodes.llm.types.StructuredResult` whose
``tool_calls`` is always a list (SG-04).

``tool_choice`` (DD-07): ``None`` lets the model decide which tool to call (a
tool call is still expected for this node); a tool *name* forces that specific
tool. The adapter maps these to the provider's native format.

Auto-validation (SG-05): when ``validate_tool_args=True`` (default), each tool
call's raw ``arguments`` are validated against the matching
``ToolDefinition.parameters_model``; the validated instance is stored in
``parsed_arguments``. On failure a retryable
:class:`~rh_cognitv.nodes.llm.errors.ToolValidationError` is raised. Pass
``validate_tool_args=False`` to keep raw dicts and skip validation.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import ValidationError

from rh_cognitv.nodes.base import BaseNode
from rh_cognitv.nodes.llm.errors import ToolValidationError
from rh_cognitv.nodes.llm.types import (
    LLMConfig,
    LLMRequest,
    Message,
    StructuredResult,
    ToolCallResult,
    ToolDefinition,
    normalize_prompt,
)
from rh_cognitv.nodes.llm_adapters.base import StructuredAdapter

OnRequest = Callable[[LLMRequest], Awaitable[None] | None]
OnResponse = Callable[[StructuredResult], Awaitable[None] | None]


async def _maybe_await(value: Any) -> None:
    """Await ``value`` if it is awaitable; otherwise do nothing."""
    if inspect.isawaitable(value):
        await value


def validate_tool_calls(
    tool_calls: list[ToolCallResult], tools: list[ToolDefinition]
) -> None:
    """Validate each tool call's arguments and populate ``parsed_arguments`` (SG-05).

    Shared by :class:`LLMStructuredNode` and ``LLMStreamNode``. Raises a
    retryable :class:`~rh_cognitv.nodes.llm.errors.ToolValidationError` when a
    call names an unknown tool or its arguments fail schema validation.
    """
    tool_map = {tool.name: tool for tool in tools}
    for call in tool_calls:
        tool = tool_map.get(call.tool_name)
        if tool is None:
            raise ToolValidationError(
                f"Model called unknown tool {call.tool_name!r}; "
                f"expected one of {sorted(tool_map)}",
                tool_name=call.tool_name,
            )
        try:
            call.parsed_arguments = tool.parameters_model.model_validate(
                call.arguments
            )
        except ValidationError as exc:
            raise ToolValidationError(
                f"Arguments for tool {call.tool_name!r} failed validation: {exc}",
                tool_name=call.tool_name,
            ) from exc


class LLMStructuredNode(BaseNode[StructuredResult]):
    """Tool-calling completion node."""

    def __init__(
        self,
        adapter: StructuredAdapter,
        *,
        on_request: OnRequest | None = None,
        on_response: OnResponse | None = None,
    ) -> None:
        self.adapter = adapter
        self.on_request = on_request
        self.on_response = on_response

    async def run(
        self,
        prompt: str | list[Message],
        config: LLMConfig,
        tools: list[ToolDefinition],
        *,
        tool_choice: str | None = None,
        validate_tool_args: bool = True,
    ) -> StructuredResult:
        """Invoke the model with ``tools`` and return a ``StructuredResult``."""
        if not tools:
            raise ValueError("LLMStructuredNode requires at least one tool")

        messages = normalize_prompt(prompt)
        request = LLMRequest(
            messages=messages,
            config=config,
            tools=tools,
            tool_choice=tool_choice,
        )

        if self.on_request is not None:
            await _maybe_await(self.on_request(request))

        result = await self.adapter.generate_structured(
            messages, config, tools, tool_choice
        )

        if validate_tool_args:
            self._validate(result, tools)

        if self.on_response is not None:
            await _maybe_await(self.on_response(result))

        return result

    @staticmethod
    def _validate(result: StructuredResult, tools: list[ToolDefinition]) -> None:
        """Validate each tool call's arguments and populate ``parsed_arguments``."""
        validate_tool_calls(result.tool_calls, tools)
