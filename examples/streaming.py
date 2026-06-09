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
)

from _common import chat_model, make_adapter


async def main() -> None:
    node = LLMStreamNode(make_adapter())
    config = LLMConfig(model=chat_model(), temperature=0.0, max_tokens=512)

    print("Streaming (live tokens):")
    async for event in node.run("Count from 1 to 5, space separated.", config):
        if isinstance(event, StreamStarted):
            print(f"  [started: {event.provider}/{event.model}]")
        elif isinstance(event, StreamTextDelta):
            print(event.text, end="", flush=True)
        elif isinstance(event, StreamCompleted):
            print(f"\n  [completed: {event.meta.tokens_used.total_tokens} tokens]")

    # collect() drives the stream to completion and returns the consolidated
    # StreamResult in one call.
    result = await node.collect("Name three primary colors.", config, batch_size=3)
    print("\nCollected result (batch_size=3):")
    print(" ", result.text.strip())


if __name__ == "__main__":
    asyncio.run(main())
