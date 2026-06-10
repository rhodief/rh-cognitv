"""LLM execution nodes and their shared infrastructure.

Phase 1 exposes the foundational pieces: canonical I/O models
(:mod:`types`), the error taxonomy (:mod:`errors`), and stream event models
(:mod:`events`). Concrete nodes are added in later phases.
"""

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
from rh_cognitv.nodes.llm.embedding_node import LLMEmbeddingNode
from rh_cognitv.nodes.llm.stream_node import LLMStreamNode
from rh_cognitv.nodes.llm.structured_node import LLMStructuredNode
from rh_cognitv.nodes.llm.text_node import LLMTextNode
from rh_cognitv.nodes.llm.types import (
    EmbeddingRequest,
    EmbeddingResult,
    LLMConfig,
    LLMRequest,
    LLMResultMeta,
    Message,
    StreamDelta,
    StreamResult,
    StreamToolCallDelta,
    StructuredResult,
    TextResult,
    TokenUsage,
    ToolCallResult,
    ToolDefinition,
    normalize_prompt,
)

__all__ = [
    # types
    "Message",
    "LLMConfig",
    "LLMRequest",
    "TokenUsage",
    "LLMResultMeta",
    "StreamDelta",
    "StreamResult",
    "StreamToolCallDelta",
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
    # nodes
    "LLMTextNode",
    "LLMStreamNode",
    "LLMStructuredNode",
    "LLMEmbeddingNode",
]
