"""Tool calling with ``LLMStructuredNode``.

Defines a Pydantic argument model, wraps it in a ``ToolDefinition``, and lets
the model call it. Returned arguments are auto-validated into a Pydantic
instance (SG-05).

Usage::

    python examples/structured_tools.py
    RH_PROVIDER=gemini python examples/structured_tools.py
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from rh_cognitv import LLMConfig, LLMStructuredNode, ToolDefinition

from _common import chat_model, make_adapter


class GetWeather(BaseModel):
    """Arguments for the get_weather tool."""

    city: str = Field(description="The city to get the weather for")
    unit: str = Field(default="celsius", description="celsius or fahrenheit")


async def main() -> None:
    tool = ToolDefinition(
        name="get_weather",
        description="Get the current weather for a city",
        parameters_model=GetWeather,
    )
    node = LLMStructuredNode(make_adapter())
    config = LLMConfig(model=chat_model(), temperature=0.0)

    result = await node.run(
        "What's the weather like in Paris?",
        config,
        [tool],
        tool_choice="get_weather",
    )

    for call in result.tool_calls:
        print(f"tool={call.tool_name} raw_args={call.arguments}")
        # parsed_arguments is a validated GetWeather instance.
        assert isinstance(call.parsed_arguments, GetWeather)
        print(f"  parsed: city={call.parsed_arguments.city!r} "
              f"unit={call.parsed_arguments.unit!r}")

    print(f"\nprovider={result.meta.provider} model={result.meta.model}")


if __name__ == "__main__":
    asyncio.run(main())
