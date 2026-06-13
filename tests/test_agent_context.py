"""Unit tests for Agent Context models (Phase 3)."""

from __future__ import annotations

import time
import pytest

from rh_cognitv.agents.context import (
    ActiveContext,
    Observation,
    Fact,
    Decision,
    TodoItem,
    TodoState,
    RetrievalEntry,
)


def test_agent_context_instantiation() -> None:
    todo = TodoState(goal="Validate user email", steps=[])
    context = ActiveContext(task="Implement registration validation", todo=todo)

    assert context.task == "Implement registration validation"
    assert context.plan is None
    assert len(context.recent_observations) == 0
    assert context.max_active_observations == 3


def test_context_hygiene_pruning() -> None:
    todo = TodoState(goal="Task Goal", steps=[])
    context = ActiveContext(
        task="Test Task", todo=todo, max_active_observations=2
    )

    # Add 4 observations with sequential timestamps
    now = time.time()
    obs1 = Observation(content="Observation 1", source="tool1", timestamp=now - 40)
    obs2 = Observation(content="Observation 2", source="tool2", timestamp=now - 30)
    obs3 = Observation(content="Observation 3", source="tool1", timestamp=now - 20)
    obs4 = Observation(content="Observation 4", source="tool3", timestamp=now - 10)

    context.recent_observations = [obs1, obs2, obs3, obs4]
    assert len(context.recent_observations) == 4

    context.apply_hygiene()

    # Should prune to the last K=2 observations (obs3 and obs4)
    assert len(context.recent_observations) == 2
    assert context.recent_observations[0].content == "Observation 3"
    assert context.recent_observations[1].content == "Observation 4"


def test_format_context_empty() -> None:
    todo = TodoState(goal="Goal", steps=[])
    context = ActiveContext(task="Simple Task", todo=todo)

    formatted = context.format_context()
    assert "# ACTIVE TASK\nSimple Task" in formatted
    assert "Goal: Goal" in formatted
    assert "Steps:\n  (No steps defined)" in formatted
    assert "Pending Decisions:\n  (None)" in formatted
    assert "Recent Facts:\n  (None)" in formatted
    assert "Recent Observations:\n  (None)" in formatted


def test_format_context_populated() -> None:
    step1 = TodoItem(description="Step 1 description", status="done")
    step2 = TodoItem(description="Step 2 description", status="in_progress")
    todo = TodoState(goal="Main Goal", steps=[step1, step2])

    obs = Observation(content="Found file config.json", source="file.read")
    fact = Fact(content="Port is 8080", extracted_from=[obs.id])
    decision = Decision(content="Use port 8080 for dev", reasoning="Port 80 is blocked")
    retrieval = RetrievalEntry(
        topic="check port config", outcome="Port is 8080", source="config.json", status="resolved"
    )

    context = ActiveContext(
        task="Setup server port",
        plan="1. Read config\n2. Set port",
        todo=todo,
        subgoal="Read config file",
        pending_decisions=[decision],
        recent_facts=[fact],
        recent_observations=[obs],
        retrieval_ledger=[retrieval],
        auto_memory=["Environment is Linux"],
        notebook_entries=["Database is sqlite"],
        artifacts=["server_run.sh"],
    )

    formatted = context.format_context()

    assert "# ACTIVE TASK\nSetup server port" in formatted
    assert "Current Plan: 1. Read config\n2. Set port" in formatted
    assert "[done] Step 1 description" in formatted
    assert "[in_progress] Step 2 description" in formatted
    assert "Current Subgoal: Read config file" in formatted
    assert f"- [{decision.id}] Use port 8080 for dev (reasoning: Port 80 is blocked)" in formatted
    assert f"- [{fact.id}] Port is 8080" in formatted
    assert f"- [{obs.id}] Found file config.json (source: file.read)" in formatted
    assert "Topic: check port config" in formatted
    assert "Outcome: Port is 8080" in formatted
    assert "Source: config.json" in formatted
    assert "- Environment is Linux" in formatted
    assert "- Database is sqlite" in formatted
    assert "- server_run.sh" in formatted


@pytest.mark.asyncio
async def test_get_default_context_tools() -> None:
    from rh_cognitv.agents.context import get_default_context_tools

    todo = TodoState(goal="Test goal", steps=[])
    context = ActiveContext(task="Test task", todo=todo)

    tools = get_default_context_tools(context)
    tools_map = {t.name: t for t in tools}

    assert "todo__create" in tools_map
    assert "todo__update" in tools_map
    assert "notebook__append" in tools_map
    assert "context__extract_facts" in tools_map
    assert "context__make_decision" in tools_map

    # Test todo__create
    res = await tools_map["todo__create"].run(description="First step")
    assert "Created TODO task" in res.output
    assert len(context.todo.steps) == 1
    step_id = context.todo.steps[0].id
    assert context.todo.steps[0].description == "First step"
    assert context.todo.steps[0].status == "pending"

    # Test todo.update
    res = await tools_map["todo__update"].run(item_id=step_id, status="done")
    assert "Updated TODO task" in res.output
    assert context.todo.steps[0].status == "done"

    # Test notebook.append
    res = await tools_map["notebook__append"].run(content="Some knowledge")
    assert "Appended knowledge" in res.output
    assert "Some knowledge" in context.notebook_entries

    # Test context.extract_facts
    res = await tools_map["context__extract_facts"].run(facts=["Fact A", "Fact B"])
    assert "Extracted facts" in res.output
    assert len(context.recent_facts) == 2
    assert context.recent_facts[0].content == "Fact A"
    assert context.recent_facts[1].content == "Fact B"

    # Test context.make_decision
    res = await tools_map["context__make_decision"].run(content="Use PostgreSQL", reasoning="Scale")
    assert "Recorded decision" in res.output
    assert len(context.pending_decisions) == 1
    assert context.pending_decisions[0].content == "Use PostgreSQL"
    assert context.pending_decisions[0].reasoning == "Scale"

