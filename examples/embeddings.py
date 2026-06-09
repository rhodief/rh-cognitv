"""Batch embeddings with ``LLMEmbeddingNode``.

Usage::

    python examples/embeddings.py
    RH_PROVIDER=gemini python examples/embeddings.py
"""

from __future__ import annotations

import asyncio

from rh_cognitv import LLMConfig, LLMEmbeddingNode

from _common import embed_model, make_adapter


async def main() -> None:
    node = LLMEmbeddingNode(make_adapter())
    config = LLMConfig(model=embed_model())

    texts = ["the cat sat on the mat", "a feline rested on the rug", "stock prices fell"]
    result = await node.run(texts, config)

    print(f"embedded {len(result.embeddings)} texts with {result.meta.model}")
    for text, vector in zip(texts, result.embeddings):
        print(f"  dims={len(vector)}  first3={vector[:3]}  text={text!r}")

    print(f"\nprovider={result.meta.provider} "
          f"tokens={result.meta.tokens_used.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
