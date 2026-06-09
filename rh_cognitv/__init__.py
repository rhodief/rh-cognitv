"""rh_cognitv — Cognitive Skill-driven Orchestration Framework.

This top-level package re-exports the stable public API: the four LLM
execution nodes, their canonical Pydantic I/O models, the structured error
taxonomy, and the stream event models.

Provider adapters are **not** re-exported here because their SDKs are optional
dependencies (SG-02 / SG-03). Import the adapter you need directly, e.g.::

    from rh_cognitv.nodes.llm_adapters.openai_adapter import OpenAIAdapter
    from rh_cognitv.nodes.llm_adapters.gemini_adapter import GeminiAdapter

Quick start::

    import asyncio
    from rh_cognitv import LLMTextNode, LLMConfig
    from rh_cognitv.nodes.llm_adapters.openai_adapter import OpenAIAdapter

    async def main() -> None:
        node = LLMTextNode(OpenAIAdapter())
        result = await node.run("Say hello", LLMConfig(model="gpt-4o-mini"))
        print(result.text)

    asyncio.run(main())
"""

from rh_cognitv.nodes import (
    BaseNode,
    LLMEmbeddingNode,
    LLMStreamNode,
    LLMStructuredNode,
    LLMTextNode,
)
from rh_cognitv.nodes.llm.errors import (
    AuthenticationError,
    ContextLengthError,
    InvalidRequestError,
    LLMError,
    LLMErrorFamily,
    ProviderError,
    RateLimitError,
    TimeoutError,
    ToolValidationError,
    map_http_status_to_error_family,
)
from rh_cognitv.nodes.llm.events import (
    StreamCompleted,
    StreamErrorEvent,
    StreamEvent,
    StreamStarted,
    StreamTextDelta,
)
from rh_cognitv.nodes.llm.types import (
    EmbeddingRequest,
    EmbeddingResult,
    LLMConfig,
    LLMRequest,
    LLMResultMeta,
    Message,
    StreamDelta,
    StreamResult,
    StructuredResult,
    TextResult,
    TokenUsage,
    ToolCallResult,
    ToolDefinition,
    normalize_prompt,
)

__version__ = "0.0.0b0"

__all__ = [
    "__version__",
    # nodes
    "BaseNode",
    "LLMTextNode",
    "LLMStreamNode",
    "LLMStructuredNode",
    "LLMEmbeddingNode",
    # types
    "Message",
    "LLMConfig",
    "LLMRequest",
    "TokenUsage",
    "LLMResultMeta",
    "StreamDelta",
    "StreamResult",
    "TextResult",
    "ToolDefinition",
    "ToolCallResult",
    "StructuredResult",
    "EmbeddingRequest",
    "EmbeddingResult",
    "normalize_prompt",
    # errors
    "LLMError",
    "LLMErrorFamily",
    "RateLimitError",
    "AuthenticationError",
    "ContextLengthError",
    "InvalidRequestError",
    "TimeoutError",
    "ProviderError",
    "ToolValidationError",
    "map_http_status_to_error_family",
    # events
    "StreamEvent",
    "StreamStarted",
    "StreamTextDelta",
    "StreamCompleted",
    "StreamErrorEvent",
]
