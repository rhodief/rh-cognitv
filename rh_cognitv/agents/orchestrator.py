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

    model_name = f"{node.name.replace('.', '_')}_args"
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
        context = ActiveContext(
            task=task,
            todo=TodoState(goal=task, steps=[]),
            auto_memory=auto_memory or [],
            max_active_observations=max_active_observations,
        )

        for step in range(1, max_steps + 1):
            await self._publish(AgentStepStarted(agent_id=self.agent_id, step_index=step))

            # Build tools list for this step
            ctx_tools = get_default_context_tools(context)
            all_tools = list(self.action_tools) + list(ctx_tools)
            tools_map = {t.name: t for t in all_tools}
            tool_definitions = [function_node_to_tool_definition(t) for t in all_tools]

            # Enforce hygiene on context
            context.apply_hygiene()

            # Compile prompt
            prompt_content = context.format_context()

            system_prompt = (
                f"{prompt_content}\n\n"
                "You are an autonomous agent operating according to the system specification above. "
                "Review the ACTIVE TASK, ACTIVE CONTEXT, and the TODO list.\n"
                "Execute the next step using your capabilities (tools).\n"
                "If you need to make plans, update tasks, extract facts, or log decisions, use the context capabilities.\n"
                "Always call at least one tool per step if the task is not yet fully completed. "
                "If the task is fully complete and all TODO items are marked as 'done', do not call any tools and summarize your findings."
            )

            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content="Please proceed with the next step."),
            ]

            # Stream response
            stream = self.llm_node.run(
                prompt=messages,
                config=config,
                tools=tool_definitions,
                tool_choice="auto",
            )

            tool_calls_to_run: list[ToolCallResult] = []

            async for event in stream:
                # Dispatch stream thought delta events
                if event.type == "stream_delta" and event.text:
                    await self._publish(AgentThoughtDelta(agent_id=self.agent_id, text=event.text))
                elif event.type == "stream_completed":
                    tool_calls_to_run = event.tool_calls

            # Execute tool calls if any
            if not tool_calls_to_run:
                # No tool calls: stop loop if todo is complete (or if the agent decided not to take action)
                all_done = len(context.todo.steps) > 0 and all(
                    s.status == "done" for s in context.todo.steps
                )
                if all_done or step > 1:
                    await self._publish(
                        AgentStepCompleted(
                            agent_id=self.agent_id, step_index=step, status="completed"
                        )
                    )
                    break

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
                    # Run tool
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

                # Store tool result as observation in context
                obs_content = output if not error else f"Error executing tool: {error}"
                context.recent_observations.append(
                    Observation(content=obs_content, source=tool_name)
                )

                # Publish updates if context was mutated via default tools
                if tool_name == "context.extract_facts" and not error:
                    # Publish facts extracted
                    for fact in context.recent_facts[-len(arguments.get("facts", [])) :]:
                        await self._publish(
                            AgentFactExtracted(
                                agent_id=self.agent_id, fact_id=fact.id, content=fact.content
                            )
                        )
                elif tool_name == "context.make_decision" and not error:
                    dec = context.pending_decisions[-1]
                    await self._publish(
                        AgentDecisionMade(
                            agent_id=self.agent_id, decision_id=dec.id, content=dec.content
                        )
                    )
                elif tool_name in ("todo.create", "todo.update") and not error:
                    await self._publish(
                        AgentTodoUpdated(
                            agent_id=self.agent_id,
                            goal=context.todo.goal,
                            steps=[s.model_dump() for s in context.todo.steps],
                        )
                    )

            await self._publish(
                AgentStepCompleted(agent_id=self.agent_id, step_index=step, status="success")
            )

        return context
