"""Agent Reasoning Loop — full orchestrator example.

Demonstrates the AgentOrchestrator driving a multi-step reasoning loop:
  - Plans work via todo.create
  - Executes action tools (read_file, list_dir)
  - Distills facts via context.extract_facts
  - Logs decisions via context.make_decision
  - Updates todo state via todo.update
  - Observes all of the above in real-time via the EventBus

Usage::

    python examples/agent_reasoning.py
    RH_PROVIDER=gemini python examples/agent_reasoning.py
"""

from __future__ import annotations

import asyncio
import os
import textwrap

from rh_cognitv import (
    LLMConfig,
    LLMStreamNode,
    AgentOrchestrator,
    EventBus,
    FunctionNode,
    AgentStepStarted,
    AgentTextDelta,
    AgentThoughtDelta,
    AgentFactExtracted,
    AgentDecisionMade,
    AgentTodoUpdated,
    AgentToolCallStarted,
    AgentToolCallFinished,
    AgentStepCompleted,
    AgentPersona
)

from _common import make_adapter, chat_model

# ── Action tools available to the agent ──────────────────────────────────────


def read_file(path: str) -> str:
    """Read the contents of a local file and return it as a string."""
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: file '{path}' does not exist."
    except Exception as exc:
        return f"Error reading file: {exc}"


def list_dir(path: str) -> str:
    """List the files and directories at the given path."""
    try:
        entries = os.listdir(path)
        return "\n".join(sorted(entries)) if entries else "(empty directory)"
    except FileNotFoundError:
        return f"Error: path '{path}' does not exist."
    except Exception as exc:
        return f"Error listing directory: {exc}"


# ── EventBus subscriber — a simple formatted trace printer ──────────────────


def make_trace_subscriber(bus: EventBus) -> None:
    """Register pretty-printing subscribers on the bus for every agent event type."""
    CYAN    = "\033[96m"
    YELLOW  = "\033[93m"
    GREEN   = "\033[92m"
    MAGENTA = "\033[95m"
    RED     = "\033[91m"
    BLUE    = "\033[94m"
    RESET   = "\033[0m"
    DIM     = "\033[2m"

    def on_step_started(e: AgentStepStarted) -> None:
        print(f"\n{CYAN}{'─' * 60}{RESET}")
        print(f"{CYAN}▶  STEP {e.step_index}{RESET}")

    async def on_text(e: AgentTextDelta) -> None:
        print(f"{DIM}{e.text}{RESET}", end="", flush=True)

    async def on_thought(e: AgentThoughtDelta) -> None:
        print(f"{MAGENTA}💭 {e.text}{RESET}", end="", flush=True)

    def on_tool_start(e: AgentToolCallStarted) -> None:
        args = ", ".join(f"{k}={v!r}" for k, v in e.arguments.items())
        kind_tag = f" [{e.tool_kind}]" if e.tool_kind else ""
        print(f"\n{YELLOW}⚙  TOOL CALL{kind_tag} → {e.tool_name}({args}){RESET}")

    def on_tool_finish(e: AgentToolCallFinished) -> None:
        if e.error:
            print(f"{RED}   ✗ ERROR: {e.error}{RESET}")
        else:
            snippet = textwrap.shorten(str(e.output), width=120, placeholder="…")
            print(f"{GREEN}   ✓ OUTPUT: {snippet}{RESET}")

    def on_fact(e: AgentFactExtracted) -> None:
        print(f"{BLUE}💡 FACT [{e.fact_id[:8]}]: {e.content}{RESET}")

    def on_decision(e: AgentDecisionMade) -> None:
        print(f"{MAGENTA}🎯 DECISION [{e.decision_id[:8]}]: {e.content}{RESET}")

    def on_todo(e: AgentTodoUpdated) -> None:
        steps_summary = [f"{s['status']} {s['description']}" for s in e.steps]
        print(f"{CYAN}📋 TODO: {' | '.join(steps_summary)}{RESET}")

    def on_step_done(e: AgentStepCompleted) -> None:
        tokens_part = ""
        if e.tokens_total is not None:
            tokens_part = (
                f"  │  🪙 {e.tokens_prompt}↑ {e.tokens_completion}↓ {e.tokens_total}Σ tok"
            )
        duration_part = f"  │  ⏱ {e.duration_ms:.0f} ms" if e.duration_ms is not None else ""
        print(f"{GREEN}✅ STEP {e.step_index} [{e.status}]{tokens_part}{duration_part}{RESET}")

    bus.subscribe(AgentStepStarted,    on_step_started)
    bus.subscribe(AgentTextDelta,      on_text)
    bus.subscribe(AgentThoughtDelta,   on_thought)
    bus.subscribe(AgentToolCallStarted,  on_tool_start)
    bus.subscribe(AgentToolCallFinished, on_tool_finish)
    bus.subscribe(AgentFactExtracted,  on_fact)
    bus.subscribe(AgentDecisionMade,   on_decision)
    bus.subscribe(AgentTodoUpdated,    on_todo)
    bus.subscribe(AgentStepCompleted,  on_step_done)


# ── Main ─────────────────────────────────────────────────────────────────────


async def main() -> None:
    # Always resolve paths relative to the project root, not the examples dir
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(project_root)

    adapter   = make_adapter()
    llm_node  = LLMStreamNode(adapter)
    config    = LLMConfig(model=chat_model(), temperature=0.0, max_tokens=1024)

    # Register real action tools the agent may call
    action_tools = [
        FunctionNode(read_file, name="read_file"),
        FunctionNode(list_dir,  name="list_dir"),
    ]

    # Set up decoupled telemetry via the EventBus
    bus = EventBus()
    make_trace_subscriber(bus)

    orchestrator = AgentOrchestrator(
        llm_node=llm_node,
        action_tools=action_tools,
        event_bus=bus,
        agent_id="explorer-agent",
        persona=AgentPersona(
            name="AI Researcher",
            role="Your name is BeeCaBoo, you're a funny agent. Respond with good mood and using emojis."
        )
    )

    
    #task = (
    #    "Explore the /app project and produce a short knowledge summary of what it is. "
    #    "Inspect the top-level directory and read the README to understand the project, "
    #    "then capture the key facts, decide which module appears most important."        
    #)
    
    #task = "Who are you and how do you work?"
    task = "Resuma o teor do arquivo README.md, destaque suas seções e brevemente o que contém em cada seção. Escreva um texto final formatado usando markdown"

    print(f"\n🚀  TASK: {task}\n")

    context = await orchestrator.run_task(
        task=task,
        config=config,
        max_steps=15,
        max_active_observations=5,
        auto_memory=[
            "Project root is /app",
            "The framework is called rh_cognitv",
        ],
    )

    # Wait for all async bus tasks to flush
    await asyncio.sleep(0.1)

    print(f"\n{'═' * 60}")
    print("📦  FINAL CONTEXT SUMMARY")
    print(f"{'═' * 60}")
    print(f"Facts distilled  : {len(context.recent_facts)}")
    print(f"Decisions logged : {len(context.pending_decisions)}")
    print(f"TODO steps       : {len(context.todo.steps)}")
    for step in context.todo.steps:
        print(f"  [{step.status:11s}] {step.description}")
    if context.notebook_entries:
        print(f"Notebook entries : {len(context.notebook_entries)}")
        for entry in context.notebook_entries:
            print(f"  • {textwrap.shorten(entry, width=90, placeholder='…')}")


if __name__ == "__main__":
    asyncio.run(main())
