"""Unit tests for EventBus (Phase 1)."""

from __future__ import annotations

import asyncio
import pytest

from rh_cognitv.event_bus import EventBus, AgentStepStarted, AgentTextDelta, AgentThoughtDelta


@pytest.mark.asyncio
async def test_event_bus_subscribe_and_publish() -> None:
    event_bus = EventBus()
    received_events: list[AgentStepStarted] = []
    event_received = asyncio.Event()

    async def async_handler(event: AgentStepStarted) -> None:
        received_events.append(event)
        event_received.set()

    event_bus.subscribe(AgentStepStarted, async_handler)

    test_event = AgentStepStarted(agent_id="test-agent", step_index=1)
    await event_bus.publish(test_event)

    # Since publishing uses asyncio.create_task for coroutine handlers,
    # wait for the handler to complete via the event.
    await asyncio.wait_for(event_received.wait(), timeout=1.0)

    assert len(received_events) == 1
    assert received_events[0].agent_id == "test-agent"
    assert received_events[0].step_index == 1


@pytest.mark.asyncio
async def test_event_bus_unsubscribe() -> None:
    event_bus = EventBus()
    received_count = 0

    def sync_handler(event: AgentStepStarted) -> None:
        nonlocal received_count
        received_count += 1

    event_bus.subscribe(AgentStepStarted, sync_handler)
    await event_bus.publish(AgentStepStarted(agent_id="test", step_index=1))
    assert received_count == 1

    event_bus.unsubscribe(AgentStepStarted, sync_handler)
    await event_bus.publish(AgentStepStarted(agent_id="test", step_index=2))
    assert received_count == 1  # Should not increase since unsubscribed


@pytest.mark.asyncio
async def test_event_bus_wildcard_subscription() -> None:
    event_bus = EventBus()
    received_events: list[Any] = []
    step_received = asyncio.Event()
    text_received = asyncio.Event()
    thought_received = asyncio.Event()

    async def wildcard_handler(event: Any) -> None:
        received_events.append(event)
        if isinstance(event, AgentStepStarted):
            step_received.set()
        elif isinstance(event, AgentTextDelta):
            text_received.set()
        elif isinstance(event, AgentThoughtDelta):
            thought_received.set()

    event_bus.subscribe("*", wildcard_handler)

    step_event = AgentStepStarted(agent_id="test", step_index=1)
    text_event = AgentTextDelta(agent_id="test", text="regular output")
    thought_event = AgentThoughtDelta(agent_id="test", text="thinking")

    await event_bus.publish(step_event)
    await event_bus.publish(text_event)
    await event_bus.publish(thought_event)

    await asyncio.wait_for(step_received.wait(), timeout=1.0)
    await asyncio.wait_for(text_received.wait(), timeout=1.0)
    await asyncio.wait_for(thought_received.wait(), timeout=1.0)

    assert len(received_events) == 3
    assert any(isinstance(e, AgentStepStarted) for e in received_events)
    assert any(isinstance(e, AgentTextDelta) for e in received_events)
    assert any(isinstance(e, AgentThoughtDelta) for e in received_events)


@pytest.mark.asyncio
async def test_event_bus_handler_error_isolation() -> None:
    event_bus = EventBus()
    success_received = asyncio.Event()

    async def failing_handler(event: AgentStepStarted) -> None:
        raise ValueError("Intentional crash")

    async def success_handler(event: AgentStepStarted) -> None:
        success_received.set()

    # The failing handler should not block the success handler or crash publish
    event_bus.subscribe(AgentStepStarted, failing_handler)
    event_bus.subscribe(AgentStepStarted, success_handler)

    test_event = AgentStepStarted(agent_id="test", step_index=1)
    await event_bus.publish(test_event)

    await asyncio.wait_for(success_received.wait(), timeout=1.0)
    # If we got here without throwing an exception in publish(), isolation is working
