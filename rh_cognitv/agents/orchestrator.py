"""AgentOrchestrator — drives multi-step agent loops (Phase 5).

Integrates the streaming LLM node, the context capability tools, the EventBus,
and custom actions to build a complete reasoning agent loop.
"""

from __future__ import annotations

import inspect
from typing import Any, Literal
from pydantic import create_model

from rh_cognitv.agents.context import (
    ActiveContext,
    TodoState,
    Observation,
    get_default_context_tools,
)
from rh_cognitv.event_bus import (
    EventBus,
    AgentStepStarted,
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

    model_name = f"{node.name.replace('__', '_').replace('.', '_')}_args"
    params_model = create_model(model_name, **fields)
    return ToolDefinition(
        name=node.name,
        description=node.description,
        parameters_model=params_model,
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
    ) -> None:
        self.llm_node = llm_node
        self.action_tools = action_tools
        self.event_bus = event_bus
        self.agent_id = agent_id

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

        # Persistent multi-turn history shared across all steps.
        # Initialised on step 1 then grown in-place; system prompt is refreshed
        # at the start of each step so the agent always sees current context state.
        messages: list[Message] = []

        for step in range(1, max_steps + 1):
            await self._publish(AgentStepStarted(agent_id=self.agent_id, step_index=step))

            # Build tools list for this step
            ctx_tools = get_default_context_tools(context)
            all_tools = list(self.action_tools) + list(ctx_tools)
            tools_map = {t.name: t for t in all_tools}
            tool_definitions = [function_node_to_tool_definition(t) for t in all_tools]

            # Enforce hygiene on context
            context.apply_hygiene()

            # Compile a fresh system prompt with the latest context state
            prompt_content = context.format_context()
            system_prompt = (
                f"{prompt_content}\n\n"
                "## Role\n"
                "You are an autonomous reasoning agent. "
                "You work step by step, thinking out loud **in plain text** before you invoke any tool.\n\n"
                "## How to behave each step\n"
                "1. **Think first** — write one or more short sentences in plain text explaining what you "
                "observe, what you intend to do next, and why. This text is visible to the user as a live "
                "thought stream.\n"
                "2. **Then act** — call the appropriate tool(s). You may call multiple tools in a single "
                "step when they are independent.\n"
                "3. **Use context tools** when you need to plan, distil knowledge, or record decisions:\n"
                "   - `todo__create` / `todo__update` — manage the step-by-step plan.\n"
                "   - `context__extract_facts` — distil stable facts from observations.\n"
                "   - `context__make_decision` — record an architectural or strategic choice.\n"
                "   - `notebook__append` — store reference material for later steps.\n"
                "4. **Finish cleanly** — when the task is fully done and all TODO items are marked "
                "'done', write a concise closing summary in plain text and call **no** tools.\n\n"
                "Always prefer writing at least one sentence of reasoning text before each batch of tool "
                "calls so the user understands your intent."
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
                messages.append(
                    Message(role="user", content="Please continue with the next step.")
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
                tool_name = call.tool_name
                arguments = call.arguments
                call_id = call.call_id

                await self._publish(
                    AgentToolCallStarted(
                        agent_id=self.agent_id,
                        tool_name=tool_name,
                        arguments=arguments,
                        call_id=call_id,
                    )
                )

                handler = tools_map.get(tool_name)
                if handler is not None:
                    res = await handler.run(**arguments)
                    output = str(res.output) if res.output is not None else ""
                    error = res.error
                else:
                    output = ""
                    error = f"Capability tool '{tool_name}' not found."

                await self._publish(
                    AgentToolCallFinished(
                        agent_id=self.agent_id,
                        tool_name=tool_name,
                        arguments=arguments,
                        output=output,
                        error=error,
                        call_id=call_id,
                    )
                )

                # Accumulate for the batched observation message
                result_line = f"[{tool_name}] → {output if not error else f'ERROR: {error}'}"
                tool_results.append(result_line)

                # Store tool result as observation in context
                obs_content = output if not error else f"Error executing tool: {error}"
                context.recent_observations.append(
                    Observation(content=obs_content, source=tool_name)
                )

                # Publish rich events for context mutations
                if tool_name == "context__extract_facts" and not error:
                    for fact in context.recent_facts[-len(arguments.get("facts", [])) :]:
                        await self._publish(
                            AgentFactExtracted(
                                agent_id=self.agent_id, fact_id=fact.id, content=fact.content
                            )
                        )
                elif tool_name == "context__make_decision" and not error:
                    dec = context.pending_decisions[-1]
                    await self._publish(
                        AgentDecisionMade(
                            agent_id=self.agent_id, decision_id=dec.id, content=dec.content
                        )
                    )
                elif tool_name in ("todo__create", "todo__update") and not error:
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
