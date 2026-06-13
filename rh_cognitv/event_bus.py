"""EventBus & Observability Infrastructure.

Provides a decoupled async event bus supporting publish/subscribe patterns.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable
from typing import Any, Literal
from pydantic import BaseModel, Field


class AgentEvent(BaseModel):
    """Base model for all agent-related events."""

    type: str
    agent_id: str
    timestamp: float = Field(default_factory=time.time)


class AgentStepStarted(AgentEvent):
    """Emitted when a new reasoning step starts."""

    type: Literal["agent_step_started"] = "agent_step_started"
    step_index: int


class AgentThoughtDelta(AgentEvent):
    """Emitted as reasoning text is streamed from the model."""

    type: Literal["agent_thought_delta"] = "agent_thought_delta"
    text: str


class AgentFactExtracted(AgentEvent):
    """Emitted when a new Fact is distilled and added to context."""

    type: Literal["agent_fact_extracted"] = "agent_fact_extracted"
    fact_id: str
    content: str


class AgentDecisionMade(AgentEvent):
    """Emitted when a Decision is parsed and added to context."""

    type: Literal["agent_decision_made"] = "agent_decision_made"
    decision_id: str
    content: str


class AgentTodoUpdated(AgentEvent):
    """Emitted when the TODO list is updated."""

    type: Literal["agent_todo_updated"] = "agent_todo_updated"
    goal: str
    steps: list[dict[str, Any]]


class AgentToolCallStarted(AgentEvent):
    """Emitted when a capability/tool starts execution."""

    type: Literal["agent_tool_call_started"] = "agent_tool_call_started"
    tool_name: str
    arguments: dict[str, Any]
    call_id: str | None = None


class AgentToolCallFinished(AgentEvent):
    """Emitted when a capability/tool finishes execution."""

    type: Literal["agent_tool_call_finished"] = "agent_tool_call_finished"
    tool_name: str
    arguments: dict[str, Any]
    output: str
    error: str | None = None
    call_id: str | None = None


class AgentStepCompleted(AgentEvent):
    """Emitted when a reasoning step completes.

    ``tokens_prompt``, ``tokens_completion``, ``tokens_total`` and
    ``duration_ms`` reflect the LLM call that drove this step.
    They are ``None`` when no LLM call was made (e.g. a no-op step).
    """

    type: Literal["agent_step_completed"] = "agent_step_completed"
    step_index: int
    status: str
    # LLM telemetry for this step
    tokens_prompt: int | None = None
    tokens_completion: int | None = None
    tokens_total: int | None = None
    duration_ms: float | None = None


class EventBus:
    """An asynchronous pub/sub event bus for decoupled framework observability."""

    def __init__(self) -> None:
        self._subscribers: dict[Any, set[Callable[[Any], Any]]] = {}

    def subscribe(self, event_type: Any, handler: Callable[[Any], Any]) -> None:
        """Register a handler for a specific event type (class, string name, or '*' for wildcard)."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = set()
        self._subscribers[event_type].add(handler)

    def unsubscribe(self, event_type: Any, handler: Callable[[Any], Any]) -> None:
        """Unsubscribe a handler from an event type."""
        if event_type in self._subscribers:
            self._subscribers[event_type].discard(handler)
            if not self._subscribers[event_type]:
                del self._subscribers[event_type]

    async def publish(self, event: Any) -> None:
        """Publish an event to all subscribed handlers.

        Handlers are executed concurrently. Coroutine handlers are scheduled
        in the event loop using `asyncio.create_task` to prevent blocking the
        publisher.
        """
        handlers: set[Callable[[Any], Any]] = set()

        # Match exact type
        event_type = type(event)
        if event_type in self._subscribers:
            handlers.update(self._subscribers[event_type])

        # Match wildcard
        if "*" in self._subscribers:
            handlers.update(self._subscribers["*"])

        if not handlers:
            return

        for handler in handlers:
            try:
                if inspect.iscoroutinefunction(handler):
                    asyncio.create_task(handler(event))
                else:
                    handler(event)
            except Exception:
                # Telemetry handlers should not crash the main execution flow
                pass
