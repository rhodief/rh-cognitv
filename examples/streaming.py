"""Streaming completion with ``LLMStreamNode``.

Demonstrates the async-generator interface, the ``on_event`` callback, and the
``collect()`` convenience method.

Usage::

    python examples/streaming.py
    RH_PROVIDER=gemini python examples/streaming.py
"""

from __future__ import annotations

import asyncio

from rh_cognitv import (
    LLMConfig,
    LLMStreamNode,
    StreamCompleted,
    StreamStarted,
    StreamTextDelta,
    Message,
)
from rh_cognitv.nodes.llm.types import ToolDefinition

from pydantic import BaseModel

from _common import chat_model, make_adapter


class GetWeatherArgs(BaseModel):
    city: str
    unit: str = "celsius"


GET_WEATHER = ToolDefinition(
    name="get_weather",
    description="Return the current weather for a city.",
    parameters_model=GetWeatherArgs,
)


async def main() -> None:
    node = LLMStreamNode(make_adapter())
    config = LLMConfig(model=chat_model(), temperature=0.0, max_tokens=512)

    # LLMStreamNode can deliver text tokens AND tool calls in the same stream.
    # The model streams a friendly intro (text) while simultaneously invoking
    # get_weather (tool call) — both arrive token-by-token via StreamTextDelta.
    print("=== Streaming: text + tool calls in one stream ===\n")

    system_prompt = (
        "You are a friendly weather assistant. "
        "Greet the user warmly in one sentence, then call get_weather "
        "to look up the requested city. Always do both."
    )
    user_prompt = "What's the weather like in San Francisco?"

    text_buf: list[str] = []

    async for event in node.run(
        prompt=[
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ],
        config=config,
        tools=[GET_WEATHER],
        tool_choice="auto",
    ):
        if isinstance(event, StreamStarted):
            print(f"[provider: {event.provider}  model: {event.model}]\n")
        elif isinstance(event, StreamTextDelta):
            if event.text:
                print(event.text, end="", flush=True)
                text_buf.append(event.text)
            if event.tool_call_deltas:
                for delta in event.tool_call_deltas:
                    if delta.tool_name:
                        print(f"\n\n[tool call → {delta.tool_name}] ", end="", flush=True)
                    if delta.arguments_delta:
                        print(delta.arguments_delta, end="", flush=True)
        elif isinstance(event, StreamCompleted):
            print(f"\n\n[{event.meta.tokens_used.total_tokens} tokens]")

            # Consolidated results live on the completed event (and on
            # StreamResult when using collect()).  Both text and tool_calls
            # are available after the stream finishes.
            if event.tool_calls:
                for call in event.tool_calls:
                    args: GetWeatherArgs = call.parsed_arguments  # type: ignore[assignment]
                    print(f"\nresult.tool_calls[0]  → {call.tool_name}(city={args.city!r}, unit={args.unit!r})")
            if text_buf:
                print(f"result.text           → {''.join(text_buf)!r}")

    # collect() is identical — StreamResult carries both .text and .tool_calls.
    print("\n=== collect() — same data, one await ===\n")
    result = await node.collect(
        [
            Message(role="system", content=system_prompt),
            Message(role="user", content="How about Rome?"),
        ],
        config,
        tools=[GET_WEATHER],
        tool_choice="auto",
    )
    print(f"text       : {result.text!r}")
    for call in result.tool_calls:
        print(f"tool_calls : {call.tool_name}({call.arguments})")


if __name__ == "__main__":
    asyncio.run(main())
