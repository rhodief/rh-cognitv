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
import time

THROTTLE_TIME = 0.05

class GetWeatherArgs(BaseModel):
    city: str
    unit: str = "celsius"


GET_WEATHER = ToolDefinition(
    name="get_weather",
    description="Return the current weather for a city.",
    parameters_model=GetWeatherArgs,
)


async def main() -> None:
    # accumulate=True (default): after the async-for loop, node.result holds
    # the consolidated StreamResult regardless of the provider.
    node = LLMStreamNode(make_adapter())
    config = LLMConfig(model=chat_model(), temperature=0.0, max_tokens=512)

    # LLMStreamNode delivers text tokens AND tool calls in the same stream.
    # The model can stream a friendly intro (text) while simultaneously
    # invoking get_weather (tool call) — both arrive via StreamTextDelta.
    print("=== Streaming: text + tool calls in one stream ===\n")

    system_prompt = (
        "You are a friendly weather assistant. "
        "Greet the user warmly in one sentence, then call get_weather "
        "to look up the requested city. Always do both."
    )

    async for event in node.run(
        prompt=[
            Message(role="system", content=system_prompt),
            Message(role="user", content="What's the weather like in San Francisco?"),
        ],
        config=config,
        tools=[GET_WEATHER],
        tool_choice="auto",
    ):
        if isinstance(event, StreamStarted):
            print(f"[provider: {event.provider}  model: {event.model}]\n")
            time.sleep(THROTTLE_TIME)
        elif isinstance(event, StreamTextDelta):
            if event.text:
                print(event.text, end="", flush=True)
                time.sleep(THROTTLE_TIME)
            if event.tool_call_deltas:
                for delta in event.tool_call_deltas:
                    if delta.tool_name:
                        print(f"\n\n[tool call → {delta.tool_name}] ", end="", flush=True)
                        time.sleep(THROTTLE_TIME)
                    if delta.arguments_delta:
                        print(delta.arguments_delta, end="", flush=True)
                        time.sleep(THROTTLE_TIME)
        elif isinstance(event, StreamCompleted):
            print(f"\n\n[{event.meta.tokens_used.total_tokens} tokens]\n")
            time.sleep(THROTTLE_TIME)

    # node.result is populated the moment the loop finishes — no second call,
    # no re-running the stream, works identically across all providers.
    assert node.result is not None
    print(f"node.result.text       : {node.result.text!r}")
    for call in node.result.tool_calls:
        args: GetWeatherArgs = call.parsed_arguments  # type: ignore[assignment]
        print(f"node.result.tool_calls : {call.tool_name}(city={args.city!r}, unit={args.unit!r})")

    # collect() is still available as a convenience when you don't need the
    # event-by-event loop at all.  It also updates node.result.
    print("\n=== collect() — skips the loop, same node.result ===\n")
    result = await node.collect(
        [
            Message(role="system", content=system_prompt),
            Message(role="user", content="How about Rome?"),
        ],
        config,
        tools=[GET_WEATHER],
        tool_choice="auto",
    )
    # result is the return value; node.result is the same object.
    assert result is node.result
    print(f"text       : {result.text!r}")
    for call in result.tool_calls:
        print(f"tool_calls : {call.tool_name}({call.arguments})")


if __name__ == "__main__":
    asyncio.run(main())
