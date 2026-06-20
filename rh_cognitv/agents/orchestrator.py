"""AgentOrchestrator — drives multi-step agent loops (Phase 5).

Integrates the streaming LLM node, the context capability tools, the EventBus,
and custom actions to build a complete reasoning agent loop.
"""

from __future__ import annotations

import inspect
from typing import Any, Literal
from pydantic import BaseModel, Field, create_model

from rh_cognitv.agents.context import (
    ActiveContext,
    TodoState,
    Observation,
    get_default_context_tools,
)
from rh_cognitv.event_bus import (
    EventBus,
    AgentStepStarted,
    AgentTextDelta,
    AgentThoughtDelta,
    AgentFactExtracted,
    AgentDecisionMade,
    AgentTodoUpdated,
    AgentToolCallStarted,
    AgentToolCallFinished,
    AgentStepCompleted,
)
from rh_cognitv.nodes.function_node import FunctionNode
from rh_cognitv.nodes.llm.stream_node import LLMStreamNode
from rh_cognitv.nodes.llm.types import LLMConfig, Message, ToolDefinition, ToolCallResult


def function_node_to_tool_definition(node: FunctionNode) -> ToolDefinition:
    """Helper to dynamically construct a ToolDefinition from a FunctionNode."""
    sig = inspect.signature(node.fn)
    fields: dict[str, Any] = {}
    for param_name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        annotation = param.annotation if param.annotation is not inspect.Parameter.empty else Any
        default = param.default if param.default is not inspect.Parameter.empty else ...
        fields[param_name] = (annotation, default)

    model_name = f"{node.name.replace('.', '_').replace('__', '_')}_args"
    params_model = create_model(model_name, **fields)
    return ToolDefinition(
        name=node.name,
        description=node.description,
        parameters_model=params_model,
    )


class AgentPersona(BaseModel):
    """Identity and general role/capabilities of the reasoning agent."""
    name: str = Field(default="an autonomous reasoning agent", description="The name or style of the agent persona")
    role: str = Field(
        default="You are given a single high-level task and must drive it to completion yourself — deciding *what* to do, in *what order*, and *when you are done*, without being told the individual tool calls. You are also a context steward: you keep your own working memory compact and high-signal.",
        description="The general role, capabilities, and system guidelines for the agent persona"
    )


class AgentOrchestrator:
    """Orchestrates multi-step agent reasoning loops using low-level nodes."""

    def __init__(
        self,
        llm_node: LLMStreamNode,
        action_tools: list[FunctionNode],
        *,
        event_bus: EventBus | None = None,
        agent_id: str = "agent-0",
        persona: AgentPersona | None = None,
    ) -> None:
        self.llm_node = llm_node
        self.action_tools = action_tools
        self.event_bus = event_bus
        self.agent_id = agent_id
        self.persona = persona or AgentPersona()

    async def _publish(self, event: Any) -> None:
        if self.event_bus is not None:
            await self.event_bus.publish(event)

    async def run_task(
        self,
        task: str,
        config: LLMConfig,
        *,
        max_steps: int = 10,
        max_active_observations: int = 3,
        auto_memory: list[str] | None = None,
    ) -> ActiveContext:
        """Runs the reasoning loop until the task is complete or max_steps is reached."""
        import json as _json

        context = ActiveContext(
            task=task,
            todo=TodoState(goal=task, steps=[]),
            auto_memory=auto_memory or [],
            max_active_observations=max_active_observations,
        )

        # Determine task complexity before starting the loop.
        classification_prompt = (
            "You are a task complexity classifier. Analyze the following task and determine if it is a "
            "complex multi-step task requiring planning, tool calls, and sequential execution, or if it is "
            "a simple conversational question/query/message that can be answered directly in a single turn without "
            "needing a TODO checklist or action tools.\n\n"
            "Respond with exactly one word: 'COMPLEX' or 'SIMPLE'.\n\n"
            f"Task: {task}\n"
            "Response:"
        )

        async def silent_sink(e):
            pass

        classification_result = await self.llm_node.collect(
            classification_prompt,
            config,
            on_event=silent_sink,
        )
        decision = (classification_result.text or "").strip().upper()
        is_complex = "SIMPLE" not in decision

        if not is_complex:
            # Register history in notebook entries
            context.notebook_entries.append(f"User asked: {task}")
            
            # Run a single direct answer step
            step = 1
            await self._publish(AgentStepStarted(agent_id=self.agent_id, step_index=step))
            
            direct_prompt = (
                f"You are {self.persona.name}.\n"
                f"Role/Guidelines: {self.persona.role}\n\n"
                f"The user has asked: {task}\n\n"
                "Please provide a direct answer to the user's query."
            )
            
            stream = self.llm_node.run(
                prompt=direct_prompt,
                config=config,
                tools=[],
                tool_choice="none",
            )
            
            accumulated_text = ""
            async for event in stream:
                if event.type == "stream_delta" and event.text:
                    accumulated_text += event.text
                    await self._publish(AgentTextDelta(agent_id=self.agent_id, text=event.text))
                elif event.type == "stream_thinking_delta" and event.text:
                    await self._publish(AgentThoughtDelta(agent_id=self.agent_id, text=event.text))
            
            _step_meta = self.llm_node.result.meta if self.llm_node.result else None
            step_tokens_prompt     = _step_meta.tokens_used.prompt_tokens     if _step_meta else None
            step_tokens_completion = _step_meta.tokens_used.completion_tokens if _step_meta else None
            step_tokens_total      = _step_meta.tokens_used.total_tokens      if _step_meta else None
            step_duration_ms       = _step_meta.duration_ms                   if _step_meta else None
            
            await self._publish(
                AgentStepCompleted(
                    agent_id=self.agent_id,
                    step_index=step,
                    status="completed",
                    tokens_prompt=step_tokens_prompt,
                    tokens_completion=step_tokens_completion,
                    tokens_total=step_tokens_total,
                    duration_ms=step_duration_ms,
                )
            )
            return context

        # Persistent multi-turn history shared across all steps.
        # Initialised on step 1 then grown in-place; system prompt is refreshed
        # at the start of each step so the agent always sees current context state.
        messages: list[Message] = []

        for step in range(1, max_steps + 1):
            await self._publish(AgentStepStarted(agent_id=self.agent_id, step_index=step))

            # Build tools list for this step.
            # External (user-provided) tools are prefixed with "external." in
            # the tool definitions sent to the LLM so they can never collide
            # with internal context tool names (e.g. "todo.create").  The
            # prefix is stripped when emitting bus events so consumers always
            # see the original tool name + tool_kind.
            EXTERNAL_PREFIX = "external."

            ctx_tools = get_default_context_tools(context)
            internal_tool_names = {t.name for t in ctx_tools}

            # Validate no duplicate names among user-provided action tools.
            seen_external: set[str] = set()
            for t in self.action_tools:
                if t.name in seen_external:
                    raise ValueError(
                        f"Duplicate external tool name: '{t.name}'. "
                        f"Each action tool must have a unique name."
                    )
                seen_external.add(t.name)

            # Build handler maps: internal tools keep original names,
            # external tools get the prefix.
            internal_tools_map: dict[str, FunctionNode] = {t.name: t for t in ctx_tools}
            external_tools_map: dict[str, FunctionNode] = {
                f"{EXTERNAL_PREFIX}{t.name}": t for t in self.action_tools
            }

            # Merged map for handler lookup (no collisions possible thanks to prefix)
            tools_map = {**internal_tools_map, **external_tools_map}

            # Tool definitions for the LLM: internal keep original names,
            # external get the prefix.
            internal_defs = [function_node_to_tool_definition(t) for t in ctx_tools]
            external_defs = []
            for t in self.action_tools:
                td = function_node_to_tool_definition(t)
                td = td.model_copy(update={"name": f"{EXTERNAL_PREFIX}{t.name}"})
                external_defs.append(td)
            tool_definitions = external_defs + internal_defs

            # Enforce hygiene on context
            context.apply_hygiene()

            # Compile a fresh system prompt with the latest context state
            prompt_content = context.format_context()
            system_prompt = (
                "## Role\n"
                f"You are {self.persona.name}. {self.persona.role}\n\n"
                "## Tools available to you\n"
                "You have two kinds of tools.\n\n"
                "**Internal context tools** (always available) — use these to manage your own cognition:\n"
                "- `todo.create(description)` — add a step to your plan. Decompose the task into steps up front.\n"
                "- `todo.update(item_id, status)` — move a step to `in_progress` or `done`. Use the exact "
                "`item_id` shown in the TODO state below. Keep statuses accurate and mark a step `done` the "
                "moment it is verified.\n"
                "- `context.extract_facts(facts)` — distil durable facts from raw tool output. Call this right "
                "after a retrieval so the conclusion survives even after the raw observation is pruned.\n"
                "- `context.make_decision(content, reasoning)` — record a choice or conclusion you commit to.\n"
                "- `notebook.append(content)` — store reference knowledge you will need in later steps.\n\n"
                "**Action tools** (task-specific, namespaced under `external.`) — use these to actually do "
                "the work: read files, list directories, search, call APIs, etc. Call them by their full "
                "`external.<name>` identifier as listed in your tool schema.\n\n"
                "## How to operate each step\n"
                "1. **Think first** — write one or two short plain-text sentences explaining what you "
                "observe and what you intend to do next. This text is streamed live to the user. (Note: If "
                "the step requires producing a summary, report, or answer, write it directly here.)\n"
                "2. **Then act** — call the tool(s) you need. Independent calls may be batched in one step. "
                "Always batch planning updates (e.g., `todo.update` to `in_progress` or `done`) with "
                "your actions or text outputs in the same step. Do not dedicate separate steps just to "
                "update a TODO's status.\n\n"
                "Run this loop autonomously:\n"
                "1. **Plan & Start** — at the start, break the task into TODO steps with `todo.create`, and "
                "immediately mark the first step as `in_progress` in the same turn.\n"
                "2. **Retrieve / act** — call the relevant action tools to gather information or make changes.\n"
                "3. **Distil** — immediately convert raw tool output into facts via `context.extract_facts` "
                "(and `context.make_decision` when you commit to a choice). Do not re-read raw observations; "
                "rely on the facts you extracted.\n"
                "4. **Record** — use `notebook.append` for reference material future steps will need.\n"
                "5. **Update the plan** — mark finished steps `done` with `todo.update` in the same step you "
                "finish them and start the next one.\n\n"
                "## Context hygiene\n"
                "Your active context is a scarce resource and raw observations are pruned automatically.\n"
                "- Prefer Facts, Decisions, and TODO State over raw observations.\n"
                "- After reading something, extract the conclusion as a fact and move on — never depend on "
                "raw tool output staying in context.\n"
                "- Avoid duplicate notebook entries, facts, or decisions; update your plan instead of repeating work.\n"
                "- **CRITICAL**: Do not repeat information (such as previously generated summaries or content) "
                "across multiple turns. If you already wrote a summary or answer in a previous turn, do not "
                "repeat it in subsequent steps or in the final closing summary; instead, write a short completion "
                "message referencing it.\n\n"
                "## Finishing\n"
                "When every TODO item is `done` and the objective is met, write a concise closing summary in "
                "plain text and call **no** tools. Do not stop while TODO items are still `pending` or `in_progress`."
                f"\n\n{prompt_content}"
            )
            
            if step == 1:
                # First step: initialise the conversation
                messages = [
                    Message(role="system", content=system_prompt),
                    Message(role="user", content=f"Task: {task}\n\nPlease begin."),
                ]
            else:
                # Subsequent steps: refresh the system prompt in-place (context evolved),
                # then add a brief user nudge to keep the conversation going
                messages[0] = Message(role="system", content=system_prompt)
                
                # Check for unfinished TODO items to nudge specifically if agent called no tools
                # but left tasks unfinished.
                unfinished_tasks = [s.description for s in context.todo.steps if s.status != "done"]
                if unfinished_tasks:
                    nudge_content = (
                        "Please continue with the next step. Note that the following TODO tasks "
                        f"are still not marked 'done': {unfinished_tasks}. If you have completed them, "
                        "you must update their status to 'done' using todo.update. Do not repeat "
                        "previously outputted summaries or facts."
                    )
                else:
                    nudge_content = "Please continue with the next step. Do not repeat previously outputted summaries or facts."
                
                messages.append(
                    Message(role="user", content=nudge_content)
                )

            # Stream response
            stream = self.llm_node.run(
                prompt=messages,
                config=config,
                tools=tool_definitions,
                tool_choice="auto",
            )

            accumulated_text = ""
            tool_calls_to_run: list[ToolCallResult] = []

            async for event in stream:
                if event.type == "stream_delta" and event.text:
                    accumulated_text += event.text
                    await self._publish(AgentTextDelta(agent_id=self.agent_id, text=event.text))
                elif event.type == "stream_thinking_delta" and event.text:
                    await self._publish(AgentThoughtDelta(agent_id=self.agent_id, text=event.text))
                elif event.type == "stream_completed":
                    tool_calls_to_run = event.tool_calls

            # Capture per-step LLM telemetry from the accumulated StreamResult
            _step_meta            = self.llm_node.result.meta if self.llm_node.result else None
            step_tokens_prompt     = _step_meta.tokens_used.prompt_tokens     if _step_meta else None
            step_tokens_completion = _step_meta.tokens_used.completion_tokens if _step_meta else None
            step_tokens_total      = _step_meta.tokens_used.total_tokens      if _step_meta else None
            step_duration_ms       = _step_meta.duration_ms                   if _step_meta else None

            # Append the assistant's reply (text only) to the shared history
            messages.append(
                Message(role="assistant", content=accumulated_text or "(no response)")
            )

            # Execute tool calls if any
            if not tool_calls_to_run:
                # Only stop if the task is truly done (all todos marked done),
                # or this is the final allowed step.
                all_done = len(context.todo.steps) > 0 and all(
                    s.status == "done" for s in context.todo.steps
                )
                if all_done or step == max_steps:
                    await self._publish(
                        AgentStepCompleted(
                            agent_id=self.agent_id,
                            step_index=step,
                            status="completed",
                            tokens_prompt=step_tokens_prompt,
                            tokens_completion=step_tokens_completion,
                            tokens_total=step_tokens_total,
                            duration_ms=step_duration_ms,
                        )
                    )
                    break

            # Collect all tool results for this step and append as a single
            # user-role observation block (avoids the provider's strict
            # tool_calls ↔ tool-role pairing constraint)
            tool_results: list[str] = []

            for call in tool_calls_to_run:
                # The LLM returns the prefixed name; resolve the original name
                # and tool_kind by checking the prefix.
                raw_tool_name = call.tool_name
                if raw_tool_name.startswith(EXTERNAL_PREFIX):
                    original_name = raw_tool_name[len(EXTERNAL_PREFIX):]
                    tool_kind = "external"
                else:
                    original_name = raw_tool_name
                    tool_kind = "internal"

                arguments = call.arguments
                call_id = call.call_id

                await self._publish(
                    AgentToolCallStarted(
                        agent_id=self.agent_id,
                        tool_name=original_name,
                        arguments=arguments,
                        call_id=call_id,
                        tool_kind=tool_kind,
                    )
                )

                handler = tools_map.get(raw_tool_name)
                if handler is not None:
                    res = await handler.run(**arguments)
                    output = str(res.output) if res.output is not None else ""
                    error = res.error
                else:
                    output = ""
                    error = f"Capability tool '{original_name}' not found."

                await self._publish(
                    AgentToolCallFinished(
                        agent_id=self.agent_id,
                        tool_name=original_name,
                        arguments=arguments,
                        output=output,
                        error=error,
                        call_id=call_id,
                        tool_kind=tool_kind,
                    )
                )

                # Accumulate for the batched observation message
                result_line = f"[{original_name}] → {output if not error else f'ERROR: {error}'}"
                tool_results.append(result_line)

                # Store tool result as observation in context
                obs_content = output if not error else f"Error executing tool: {error}"
                context.recent_observations.append(
                    Observation(content=obs_content, source=original_name)
                )

                # Publish rich events for context mutations
                if original_name == "context.extract_facts" and not error:
                    for fact in context.recent_facts[-len(arguments.get("facts", [])) :]:
                        await self._publish(
                            AgentFactExtracted(
                                agent_id=self.agent_id, fact_id=fact.id, content=fact.content
                            )
                        )
                elif original_name == "context.make_decision" and not error:
                    dec = context.pending_decisions[-1]
                    await self._publish(
                        AgentDecisionMade(
                            agent_id=self.agent_id, decision_id=dec.id, content=dec.content
                        )
                    )
                elif original_name in ("todo.create", "todo.update") and not error:
                    await self._publish(
                        AgentTodoUpdated(
                            agent_id=self.agent_id,
                            goal=context.todo.goal,
                            steps=[s.model_dump() for s in context.todo.steps],
                        )
                    )

            # Append all tool results from this step as a single user-role message
            # so the model sees a clean observations block for the next step
            if tool_results:
                obs_block = "Tool results from this step:\n" + "\n".join(tool_results)
                messages.append(Message(role="user", content=obs_block))

            await self._publish(
                AgentStepCompleted(
                    agent_id=self.agent_id,
                    step_index=step,
                    status="success",
                    tokens_prompt=step_tokens_prompt,
                    tokens_completion=step_tokens_completion,
                    tokens_total=step_tokens_total,
                    duration_ms=step_duration_ms,
                )
            )

        return context
