"""Single-shot text completion with ``LLMTextNode``.

Usage::

    python examples/text_completion.py
    RH_PROVIDER=gemini python examples/text_completion.py
"""

from __future__ import annotations

import asyncio

from rh_cognitv import LLMConfig, LLMTextNode, Message

from _common import chat_model, make_adapter


async def main() -> None:
    node = LLMTextNode(make_adapter())
    config = LLMConfig(model=chat_model(), temperature=0.0, max_tokens=512)

    # A bare string is auto-wrapped as a single user message (DD-10).
    result = await node.run("In one sentence, what is an embedding?", config)
    print("Prompt (string shorthand):")
    print(" ", result.text.strip())

    # Full control: multi-turn with a system message.
    messages = [
        Message(role="system", content="You answer with a single word."),
        Message(role="user", content="What is the capital of France?"),
    ]
    result = await node.run(messages, config)
    print("\nPrompt (message list):")
    print(" ", result.text.strip())

    meta = result.meta
    print(
        f"\nprovider={meta.provider} model={meta.model} "
        f"tokens={meta.tokens_used.total_tokens} duration_ms={meta.duration_ms:.0f}"
    )


if __name__ == "__main__":
    asyncio.run(main())
