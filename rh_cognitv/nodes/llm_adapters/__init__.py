"""Provider-adapter package.

Adapters translate between the canonical, provider-agnostic node contract and
each provider's native SDK. Concrete adapters (OpenAI, Anthropic, Gemini) are
added in later phases; Phase 1 defines only the abstract per-capability
interfaces (:mod:`base`).
"""

from rh_cognitv.nodes.llm_adapters.base import (
    EmbeddingAdapter,
    StreamAdapter,
    StructuredAdapter,
    TextAdapter,
)

__all__ = [
    "TextAdapter",
    "StreamAdapter",
    "StructuredAdapter",
    "EmbeddingAdapter",
]
