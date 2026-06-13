"""Unit tests for AgentOrchestrator (Phase 5)."""

from __future__ import annotations

import asyncio
import pytest
from typing import Any, AsyncGenerator

from rh_cognitv import LLMConfig
from rh_cognitv.agents.context import ActiveContext, TodoState
from rh_cognitv.agents.orchestrator import AgentOrchestrator
from rh_cognitv.event_bus import EventBus
from rh_cognitv.nodes.function_node import FunctionNode
from rh_cognitv.nodes.llm.stream_node import LLMStreamNode
from rh_cognitv.nodes.llm.types import StreamDelta, StreamToolCallDelta, TokenUsage
from rh_cognitv.nodes.llm_adapters.base import StreamAdapter


class MockStreamAdapter(StreamAdapter):
    """A mock StreamAdapter that yields pre-programmed list of StreamDeltas."""

    def __init__(self, steps_deltas: list[list[StreamDelta]]) -> None:
        self.provider = "mock-provider"
        self.steps_deltas = steps_deltas
        self.current_step = 0

    async def stream_text(
        self,
        messages: Any,
        config: Any,
        tools: Any = None,
        tool_choice: Any = None,
    ) -> AsyncGenerator[StreamDelta, None]:
        if self.current_step < len(self.steps_deltas):
            deltas = self.steps_deltas[self.current_step]
            self.current_step += 1
            for d in deltas:
                yield d
        else:
            # Yield nothing if out of steps
            yield StreamDelta()


def read_file(path: str) -> str:
    """Mock action tool to read a file."""
    return f"content of {path}: port=8080"


@pytest.mark.asyncio
async def test_agent_orchestrator_loop() -> None:
    # Pre-program the LLM output deltas for each step of the loop
    # Step 1: LLM says "I need to plan" and calls todo.create(description="Read config")
    step1_deltas = [
        StreamDelta(text="Thinking: I need to plan."),
        StreamDelta(
            tool_call_deltas=[
                StreamToolCallDelta(
                    index=0,
                    call_id="call-1",
                    tool_name="todo__create",
                    arguments_delta='{"description": "Read config"}',
                )
            ]
        ),
    ]

    # Step 2: LLM says "Reading the file..." and calls read_file(path="config.json")
    step2_deltas = [
        StreamDelta(text="Thinking: Reading the file..."),
        StreamDelta(
            tool_call_deltas=[
                StreamToolCallDelta(
                    index=0,
                    call_id="call-2",
                    tool_name="read_file",
                    arguments_delta='{"path": "config.json"}',
                )
            ]
        ),
    ]

    # Step 3: LLM says "Port found" and calls context.extract_facts and todo.update
    step3_deltas = [
        StreamDelta(text="Thinking: Port found."),
        # First tool call: context.extract_facts(facts=["Port is 8080"])
        StreamDelta(
            tool_call_deltas=[
                StreamToolCallDelta(
                    index=0,
                    call_id="call-3",
                    tool_name="context__extract_facts",
                    arguments_delta='{"facts": ["Port is 8080"]}',
                )
            ]
        ),
        # Second tool call: todo.update(item_id="...", status="done")
        # We'll rely on the handler parsing the argument. Since todo.create returns the created step,
        # in a mock we can update a placeholder. But to test parameter matching, let's pass a dummy id.
        StreamDelta(
            tool_call_deltas=[
                StreamToolCallDelta(
                    index=1,
                    call_id="call-4",
                    tool_name="todo__update",
                    arguments_delta='{"item_id": "dummy-id", "status": "done"}',
                )
            ]
        ),
    ]

    # Step 4: Concludes without tool calls
    step4_deltas = [
        StreamDelta(text="Thinking: Finished task."),
    ]

    mock_adapter = MockStreamAdapter([step1_deltas, step2_deltas, step3_deltas, step4_deltas])
    llm_node = LLMStreamNode(mock_adapter)
    
    # We wrap the user action tool
    action_node = FunctionNode(read_file, name="read_file")

    event_bus = EventBus()
    published_events = []

    async def log_event(event: Any) -> None:
        published_events.append(event)

    event_bus.subscribe("*", log_event)

    orchestrator = AgentOrchestrator(
        llm_node=llm_node,
        action_tools=[action_node],
        event_bus=event_bus,
        agent_id="test-agent-0",
    )

    config = LLMConfig(model="mock-model")
    
    # Run loop
    final_context = await orchestrator.run_task(
        task="Find config port",
        config=config,
        max_steps=5,
        max_active_observations=2,
    )

    # Assert final state is complete
    assert final_context.task == "Find config port"
    
    # We should have 1 task step in todo (from Step 1)
    assert len(final_context.todo.steps) == 1
    assert final_context.todo.steps[0].description == "Read config"
    # Step 3 updated the step's status (since our mock todo.update updated all steps to status="done")
    # Wait, the default tool todo_update updates steps matching the item_id. 
    # In Step 3, we passed 'dummy-id', so it won't match our actual created step's auto-generated ID!
    # That is expected since we passed "dummy-id" in the mock deltas. Let's make sure it reported task not found
    assert any("Error: TODO task with ID dummy-id not found" in obs.content for obs in final_context.recent_observations)

    # Let's verify facts
    assert len(final_context.recent_facts) == 1
    assert final_context.recent_facts[0].content == "Port is 8080"

    # Let's verify observations (with K=2 hygiene limit enforced at start of step)
    # Total observations added:
    # Step 1: todo.create -> 1 obs
    # Step 2: read_file -> 1 obs
    # Step 3: context.extract_facts -> 1 obs
    # Step 3: todo.update -> 1 obs
    # Since K=2 is applied at the beginning of each step, at Step 4 it has pruned to 2 observations.
    assert len(final_context.recent_observations) <= 2

    # Let's verify event bus has received step, thought, tool, and completion events
    await asyncio.sleep(0.05)
    assert len(published_events) > 0
    assert any(e.type == "agent_step_started" for e in published_events)
    assert any(e.type == "agent_thought_delta" for e in published_events)
    assert any(e.type == "agent_tool_call_started" for e in published_events)
    assert any(e.type == "agent_tool_call_finished" for e in published_events)
    assert any(e.type == "agent_fact_extracted" for e in published_events)
    assert any(e.type == "agent_step_completed" for e in published_events)
