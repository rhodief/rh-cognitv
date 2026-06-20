"""Agent Context models representing the Agent Operating System state (Phase 3).

Follows Pydantic-first structure to represent observations, facts, decisions,
todos, retrieval ledger, auto memory, notebook, and artifacts.
"""

from __future__ import annotations

import time
from typing import Any, Literal
import ulid
from pydantic import BaseModel, Field


class Observation(BaseModel):
    """Raw, temporary information captured during execution (e.g. tool results)."""

    id: str = Field(default_factory=lambda: str(ulid.new()))
    content: str
    source: str | None = None
    timestamp: float = Field(default_factory=time.time)


class Fact(BaseModel):
    """A distilled, factual statement extracted from observations."""

    id: str = Field(default_factory=lambda: str(ulid.new()))
    content: str
    extracted_from: list[str] = Field(default_factory=list)  # Observation IDs
    timestamp: float = Field(default_factory=time.time)


class Decision(BaseModel):
    """A commitment, choice, or design direction made by the agent."""

    id: str = Field(default_factory=lambda: str(ulid.new()))
    content: str
    reasoning: str | None = None
    timestamp: float = Field(default_factory=time.time)


class TodoItem(BaseModel):
    """A single tracking item in the TODO checklist."""

    id: str = Field(default_factory=lambda: str(ulid.new()))
    description: str
    status: Literal["pending", "in_progress", "done"] = "pending"


class TodoState(BaseModel):
    """The goal and list of sequential steps to resolve it."""

    goal: str
    steps: list[TodoItem] = Field(default_factory=list)


class RetrievalEntry(BaseModel):
    """An entry in the retrieval ledger detailing search/investigation steps."""

    topic: str
    outcome: str
    source: str
    status: Literal["pending", "resolved", "failed"] = "resolved"


class ActiveContext(BaseModel):
    """The complete in-memory cognitive working context for the agent reasoning loop."""

    task: str
    plan: str | None = None
    todo: TodoState
    subgoal: str | None = None
    pending_decisions: list[Decision] = Field(default_factory=list)
    recent_facts: list[Fact] = Field(default_factory=list)
    recent_observations: list[Observation] = Field(default_factory=list)
    retrieval_ledger: list[RetrievalEntry] = Field(default_factory=list)
    auto_memory: list[str] = Field(default_factory=list)
    notebook_entries: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)

    # Configurable observation prune count (K parameter)
    max_active_observations: int = Field(default=3, ge=1)

    def apply_hygiene(self) -> None:
        """Enforces structural context hygiene budget rules.

        Keeps only the most-recent K observations in the active context.
        """
        if len(self.recent_observations) > self.max_active_observations:
            # Sort by timestamp (oldest first) and keep only the last K
            sorted_obs = sorted(self.recent_observations, key=lambda o: o.timestamp)
            self.recent_observations = sorted_obs[-self.max_active_observations :]

    def format_context(self) -> str:
        """Renders the context into a formatted system prompt matching 02_agent_context.md."""
        todo_steps = ""
        if self.todo.steps:
            for step in self.todo.steps:
                todo_steps += f"  [{step.status}] {step.description} (id: {step.id})\n"
        else:
            todo_steps = "  (No steps defined)\n"

        decisions_str = ""
        if self.pending_decisions:
            for d in self.pending_decisions:
                reason_suffix = f" (reasoning: {d.reasoning})" if d.reasoning else ""
                decisions_str += f"  - [{d.id}] {d.content}{reason_suffix}\n"
        else:
            decisions_str = "  (None)\n"

        facts_str = ""
        if self.recent_facts:
            for f in self.recent_facts:
                facts_str += f"  - [{f.id}] {f.content}\n"
        else:
            facts_str = "  (None)\n"

        obs_str = ""
        if self.recent_observations:
            for o in self.recent_observations:
                source_suffix = f" (source: {o.source})" if o.source else ""
                obs_str += f"  - [{o.id}] {o.content}{source_suffix}\n"
        else:
            obs_str = "  (None)\n"

        ledger_str = ""
        if self.retrieval_ledger:
            for entry in self.retrieval_ledger:
                ledger_str += (
                    f"Topic: {entry.topic}\n"
                    f"Outcome: {entry.outcome}\n"
                    f"Source: {entry.source}\n"
                    f"Status: {entry.status}\n\n"
                )
        else:
            ledger_str = "(Empty)\n"

        auto_mem_str = (
            "\n".join(f"- {m}" for m in self.auto_memory) if self.auto_memory else "(None)"
        )
        notebook_str = (
            "\n".join(f"- {n}" for n in self.notebook_entries)
            if self.notebook_entries
            else "(None)"
        )
        artifacts_str = (
            "\n".join(f"- {a}" for a in self.artifacts) if self.artifacts else "(None)"
        )

        return (
            f"# ACTIVE TASK\n{self.task}\n\n"
            f"# ACTIVE CONTEXT\n"
            f"- Current Task: {self.task}\n"
            f"- Current Plan: {self.plan or '(No plan created yet)'}\n"
            f"- Current TODO State:\n"
            f"  Goal: {self.todo.goal or self.task}\n"
            f"  Steps:\n{todo_steps}"
            f"- Current Subgoal: {self.subgoal or '(None)'}\n"
            f"- Pending Decisions:\n{decisions_str}"
            f"- Recent Facts:\n{facts_str}"
            f"- Recent Observations:\n{obs_str}\n"
            f"# RETRIEVAL LEDGER\n{ledger_str}\n"
            f"# AUTO MEMORY\n{auto_mem_str}\n\n"
            f"# NOTEBOOK\n{notebook_str}\n\n"
            f"# AVAILABLE ARTIFACTS\n{artifacts_str}"
        )


def get_default_context_tools(context: ActiveContext) -> list[Any]:
    """Helper function to build a list of FunctionNodes bound to mutate the given ActiveContext."""
    from rh_cognitv.nodes.function_node import FunctionNode

    def todo_create(description: str) -> str:
        """Create a new task in the TODO checklist."""
        item = TodoItem(description=description, status="pending")
        context.todo.steps.append(item)
        return f"Created TODO task {item.id}: {description}"

    def todo_update(item_id: str, status: Literal["pending", "in_progress", "done"]) -> str:
        """Update the status of an existing TODO task by its ID."""
        for step in context.todo.steps:
            if step.id == item_id:
                step.status = status
                return f"Updated TODO task {item_id} to status: {status}"
        return f"Error: TODO task with ID {item_id} not found."

    def notebook_append(content: str) -> str:
        """Append new working knowledge to the project Notebook."""
        context.notebook_entries.append(content)
        return "Appended knowledge entry to Notebook."

    def context_extract_facts(facts: list[str]) -> str:
        """Distill new facts from recent observations and append them to context."""
        added_ids = []
        for fact_text in facts:
            f = Fact(content=fact_text)
            context.recent_facts.append(f)
            added_ids.append(f.id)
        return f"Extracted facts: {added_ids}"

    def context_make_decision(content: str, reasoning: str | None = None) -> str:
        """Record a key design decision or course of action."""
        d = Decision(content=content, reasoning=reasoning)
        context.pending_decisions.append(d)
        return f"Recorded decision: {d.id}"

    return [
        FunctionNode(todo_create,           name="todo.create"),
        FunctionNode(todo_update,           name="todo.update"),
        FunctionNode(notebook_append,       name="notebook.append"),
        FunctionNode(context_extract_facts, name="context.extract_facts"),
        FunctionNode(context_make_decision, name="context.make_decision"),
    ]

