# Execution Nodes — Development Spec

> Derived from [00_execution_nodes_scketch.md](./00_execution_nodes_scketch.md)
> Decisions finalized: 2026-06-09

---

## 1. Overview

We are building the **first functional layer** of `rh_cognitv` — a brand-new framework starting from scratch. The package structure exists (`pyproject.toml`, dependencies) but there is **no code yet**. These execution nodes are the **foundation** of the entire system, providing first-class support for:

| Node | Purpose |
|------|---------|
| **LLMStreamNode** | Stream tokens/objects from any LLM provider via async generators, emitting canonical events for each chunk (or batched chunks). Produces a consolidated `StreamResult` at the end. |
| **LLMTextNode** | Single-shot (non-streaming) LLM completion. Designed for workflow steps that don't need real-time token delivery. Returns a canonical `TextResult`. |
| **LLMStructuredNode** | Tool-calling / function-calling LLM invocations. Accepts `ToolDefinition` wrappers around Pydantic models, returns structured `ToolCallResult`(s) with function name, parsed arguments, etc. |
| **LLMEmbeddingNode** | Batch text→embedding. Takes a list of strings, returns a list of float vectors plus usage metadata. |

All four nodes share these cross-cutting concerns:

- **Provider abstraction via DI** — a pluggable adapter interface (ABC) so consumers switch between OpenAI, Anthropic, Gemini (and future providers) without changing calling code.
- **Pydantic-first I/O** — inputs, outputs, configs, and events are Pydantic `BaseModel` instances, not raw dicts.
- **Canonical result objects** — every node returns a result carrying `output`, `tokens_used`, `duration_ms`, `model`, `provider`, and error information when applicable.
- **Canonical error types** — LLM-specific errors follow a structured taxonomy (`family`, `code`, `retryable`) designed so that a future retry/runtime engine can consume them directly.
- **Async-only** — all node operations are `async`. No sync wrappers.

These nodes are the **foundation for the agentic ecosystem** to be built afterward (agents, function nodes, flow nodes like ForEach/MapReduce, etc.).

---

## 2. Architecture

```
rh_cognitv/
└── nodes/
    ├── __init__.py
    ├── base.py                    # BaseNode ABC
    ├── llm/
    │   ├── __init__.py
    │   ├── types.py               # Shared Pydantic models (LLMConfig, TokenUsage, Message, etc.)
    │   ├── errors.py              # LLM error taxonomy
    │   ├── events.py              # Canonical LLM events
    │   ├── stream_node.py         # LLMStreamNode
    │   ├── text_node.py           # LLMTextNode
    │   ├── structured_node.py     # LLMStructuredNode
    │   └── embedding_node.py      # LLMEmbeddingNode
    └── llm_adapters/
        ├── __init__.py
        ├── base.py                # Abstract adapter ABCs (per-capability)
        ├── openai_adapter.py      # OpenAI SDK adapter
        ├── anthropic_adapter.py   # Anthropic SDK adapter
        └── gemini_adapter.py      # Google Gemini SDK adapter
```

---

## 3. Design Decisions (Consolidated)

### DD-01: Adapter Interface — ABC

Adapter contracts use `abc.ABC` with abstract methods. This gives explicit contracts, immediate errors at instantiation if a method is missing, and IDE auto-generation of stubs. We can relax to `Protocol` later if we need a public plugin ecosystem.

---

### DD-02: Separate Adapter Interfaces Per Capability

Five separate ABCs:

| ABC | Used by |
|-----|---------|
| `StreamAdapter` | `LLMStreamNode` |
| `TextAdapter` | `LLMTextNode` |
| `StructuredAdapter` | `LLMStructuredNode` |
| `EmbeddingAdapter` | `LLMEmbeddingNode` |

A concrete adapter class (e.g., `OpenAIAdapter`) implements **all interfaces the provider supports** via multiple inheritance. This means:

- Each node type-hints its adapter parameter with the specific ABC it needs.
- If you try to pass an adapter that doesn't implement the required interface, **the type checker / linter catches it** before runtime.
- Embedding-only providers (e.g., Cohere, Voyage AI) only implement `EmbeddingAdapter` without stubbing chat methods.

---

### DD-03: Constructor Injection

Adapters are injected at node construction time:

```python
adapter = OpenAIAdapter(api_key="...")
node = LLMStreamNode(adapter=adapter)
```

Nodes are cheap to instantiate. If you need multiple providers, create multiple node instances.

---

### DD-04: AsyncGenerator + Optional Event Side-Emission

`LLMStreamNode.run()` is an **`AsyncGenerator`** that yields canonical stream events. This is the primary consumer interface.

For secondary consumers (logging, monitoring, future EventBus), the node accepts an optional `on_event: Callable` callback. When we build an EventBus later, it plugs in naturally as a callback consumer.

---

### DD-05: Count-Based Batch Chunking

Streaming uses a `batch_size: int = 1` parameter. When `batch_size=N`, N incoming chunks from the provider are concatenated into a single emitted event before yielding. Default is 1 (emit every chunk as-is).

---

### DD-06: ToolDefinition Wrapper

Tools are passed as explicit `ToolDefinition` wrappers:

```python
class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters_model: type[BaseModel]
```

This gives full control over name, description, and schema. In documentation and examples, we recommend using `MyTool.__name__` or similar for consistency.

---

### DD-07: Tool Choice Semantics (Split by Node)

The `tool_choice` semantics differ per node:

**`LLMStructuredNode`:**
- `tool_choice: str | None = None`
- `None` (default) → the model decides which tool to call ("auto" behavior). `"required"` is always implied for this node since it doesn't make sense to use StructuredNode without tools.
- A specific tool name as `str` → forces that tool. Documentation recommends using `my_tool_definition.name` to avoid typos.

**`LLMStreamNode`** (when tools are supported in future iterations):
- `tool_choice: Literal["none", "auto", "required"] | str = "auto"`
- `"none"` → don't use tools (stream text only)
- `"auto"` → model decides
- `"required"` → model must call a tool
- A specific `str` → forces a specific tool

The adapter layer maps these values to each provider's native format.

---

### DD-08: Standalone + `__call__` Protocol

Nodes work independently right now with `async run()`. They also implement `async __call__()` that delegates to `run()`, so a future runtime can treat them as generic async callables.

```python
# Direct use
result = await node.run(messages=..., config=...)

# Future runtime use (node is a callable)
await runtime.run(Execution(fn=node, kwargs={...}))
```

---

### DD-09: Error Mapping Inside Each Adapter

Each adapter catches provider-specific exceptions (e.g., `openai.RateLimitError`, `anthropic.APIStatusError`) and raises canonical `LLMError` subclasses with structured fields: `family`, `code`, `message`, `retryable`, `traceback`.

Shared utility functions (e.g., `map_http_status_to_error_family(status_code)`) reduce duplication across adapters.

---

### DD-10: String Shorthand + Message Models

Prompts accept `str | list[Message]`:

```python
# Simple — auto-wrapped as a single user message
await node.run("What is the capital of France?", config=...)

# Full control — multi-turn with system prompt
await node.run([
    Message(role="system", content="You are a helpful assistant."),
    Message(role="user", content="What is the capital of France?"),
], config=...)
```

Internally, `str` is normalized to `[Message(role="user", content=input)]`.

---

### DD-11: Separate EmbeddingAdapter

`EmbeddingAdapter` is a completely separate ABC from the chat adapters. A concrete class like `OpenAIAdapter` can implement both `TextAdapter` and `EmbeddingAdapter` via multiple inheritance, but embedding-only providers are not forced to stub chat methods.

---

### DD-12: Async-Only

All node operations are `async`. No sync wrappers. Users in sync contexts use `asyncio.run()` at the top level.

---

### SG-01: Observability via Callback Hooks

Nodes accept optional `on_request: Callable` and `on_response: Callable` callbacks for observability. These receive the full request/response as Pydantic models. When we later build an EventBus, it can be wired in as a callback consumer.

---

### SG-02: Manual Adapter Instantiation

For v1, users manually import and instantiate the adapter they want:

```python
from rh_cognitv.nodes.llm_adapters.openai_adapter import OpenAIAdapter
adapter = OpenAIAdapter(api_key="...")
```

No factory, no registry. Explicit and simple.

---

### SG-03: Lazy Import with Clear Error

Provider SDKs are optional dependencies. Each adapter module uses lazy imports:

```python
try:
    import openai
except ImportError:
    raise ImportError("Install rh_cognitv[openai] to use the OpenAI adapter")
```

Update `pyproject.toml` to add `anthropic` and `google-genai` optional dependency groups.

---

### SG-04: Multiple Tool Calls — Always Return a List

`StructuredResult.tool_calls` is always `list[ToolCallResult]`, even if only one tool was called. This is future-proof for parallel function calling. Single-tool users just do `result.tool_calls[0]`.

---

### SG-05: Auto-Validation of Tool Call Arguments (with Opt-Out)

By default (`validate_tool_args=True`), the node validates the LLM's returned arguments against the `ToolDefinition.parameters_model` schema using Pydantic. On success, `ToolCallResult.parsed_arguments` contains a validated Pydantic model instance.

On validation failure, a canonical `ToolValidationError` is raised. This error is designed to be **retriable** by a future runtime (re-prompt the LLM with the validation error message).

Users can disable with `validate_tool_args=False` to get raw dicts.

---

## 4. Core Models

```python
# --- Shared ---
class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None           # for tool messages

class LLMConfig(BaseModel):
    model: str
    temperature: float = 1.0
    max_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    extra: dict[str, Any] = {}        # provider-specific pass-through

class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class LLMResultMeta(BaseModel):
    model: str
    provider: str
    tokens_used: TokenUsage
    duration_ms: float
    raw_response: Any | None = None   # optional, for debugging

# --- Stream ---
class StreamDelta(BaseModel):
    text: str | None = None
    object_fragment: dict[str, Any] | None = None

class StreamResult(BaseModel):
    text: str
    object: dict[str, Any] | None = None
    meta: LLMResultMeta

# --- Text ---
class TextResult(BaseModel):
    text: str
    meta: LLMResultMeta

# --- Structured ---
class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters_model: type[BaseModel]       # the Pydantic model for args schema

class ToolCallResult(BaseModel):
    tool_name: str
    arguments: dict[str, Any]               # raw arguments
    parsed_arguments: BaseModel | None = None  # validated Pydantic instance (when validate_tool_args=True)
    call_id: str | None = None

class StructuredResult(BaseModel):
    tool_calls: list[ToolCallResult]
    meta: LLMResultMeta

# --- Embedding ---
class EmbeddingResult(BaseModel):
    embeddings: list[list[float]]
    meta: LLMResultMeta
```

---

## 5. Implementation Phases

### Phase 1 — Foundation (Types, Errors, Adapter ABCs)

**Goal:** Build the shared infrastructure that all nodes and adapters depend on.

**Deliverables:**
- `nodes/__init__.py` and `nodes/base.py` — `BaseNode` ABC with `async run()` and `async __call__()`
- `nodes/llm/types.py` — All shared Pydantic models: `Message`, `LLMConfig`, `TokenUsage`, `LLMResultMeta`, `ToolDefinition`, `ToolCallResult`, all result types
- `nodes/llm/errors.py` — Canonical error taxonomy: `LLMError` base, subclasses for `RateLimitError`, `AuthenticationError`, `ContextLengthError`, `ToolValidationError`, etc. Each with `family`, `code`, `retryable`
- `nodes/llm/events.py` — Stream event models: `StreamStarted`, `StreamDelta`, `StreamCompleted`, `StreamError`
- `nodes/llm_adapters/base.py` — ABC definitions: `StreamAdapter`, `TextAdapter`, `StructuredAdapter`, `EmbeddingAdapter`
- Update `pyproject.toml` — Add `anthropic` and `google-genai` optional deps

**Tests:** Unit tests for all Pydantic models (serialization, validation, edge cases). Unit tests for error taxonomy.

**Exit criteria:** All types importable. Models serialize/deserialize correctly. Error hierarchy is clean.

---

### Phase 2 — LLMTextNode + OpenAI TextAdapter

**Goal:** Build the simplest node end-to-end as the reference implementation.

**Deliverables:**
- `nodes/llm/text_node.py` — `LLMTextNode` with `run()`, `__call__()`, `on_request` / `on_response` callbacks, string shorthand + `list[Message]` input
- `nodes/llm_adapters/openai_adapter.py` — `OpenAITextAdapter` implementing `TextAdapter` ABC. Lazy import of `openai` SDK. Error mapping inside adapter.

**Tests:**
- Unit tests with mocked OpenAI SDK responses
- Integration test (optional, requires API key) calling a real model

**Exit criteria:** `LLMTextNode` returns a valid `TextResult` with tokens, duration, model. Error mapping works for common OpenAI errors (rate limit, auth, context length).

---

### Phase 3 — LLMStreamNode + OpenAI StreamAdapter

**Goal:** Add streaming with async generator, batch chunking, and event side-emission.

**Deliverables:**
- `nodes/llm/stream_node.py` — `LLMStreamNode` yielding `StreamEvent` instances via `AsyncGenerator`. Count-based `batch_size` parameter. Optional `on_event` callback. Returns consolidated `StreamResult` after iteration.
- Extend `openai_adapter.py` — Add `OpenAIStreamAdapter` implementing `StreamAdapter` ABC.

**Tests:**
- Unit tests with mocked streaming responses (simulate chunk sequence)
- Test batch_size=1, batch_size=3, batch_size > total chunks
- Test on_event callback invocation
- Test error mid-stream

**Exit criteria:** Streaming works end-to-end. Batching consolidates correctly. Events are emitted. Final `StreamResult` matches accumulated content.

---

### Phase 4 — LLMStructuredNode + OpenAI StructuredAdapter

**Goal:** Add tool-calling with ToolDefinition, auto-validation, and multiple tool calls.

**Deliverables:**
- `nodes/llm/structured_node.py` — `LLMStructuredNode` accepting `list[ToolDefinition]`. `tool_choice: str | None`. Auto-validation with `validate_tool_args` flag. Returns `StructuredResult` with `list[ToolCallResult]`.
- Extend `openai_adapter.py` — Add `OpenAIStructuredAdapter` implementing `StructuredAdapter` ABC. Converts `ToolDefinition` to OpenAI function schema. Maps `tool_choice` to OpenAI format.

**Tests:**
- Unit tests with mocked tool-call responses (single and multiple tool calls)
- Test auto-validation success and failure (`ToolValidationError`)
- Test `tool_choice=None` vs specific tool name
- Test `validate_tool_args=False` returns raw dicts

**Exit criteria:** Tool calling works. Validation catches bad arguments. Multiple tool calls returned as list.

---

### Phase 5 — LLMEmbeddingNode + OpenAI EmbeddingAdapter

**Goal:** Add embedding support.

**Deliverables:**
- `nodes/llm/embedding_node.py` — `LLMEmbeddingNode` accepting `list[str]`. Returns `EmbeddingResult`.
- Extend `openai_adapter.py` — Add `OpenAIEmbeddingAdapter` implementing `EmbeddingAdapter` ABC.

**Tests:**
- Unit tests with mocked embedding responses
- Test single and batch text inputs
- Test token usage tracking

**Exit criteria:** Embeddings returned as `list[list[float]]` with correct metadata.

---

### Phase 6 — Gemini Adapters

**Goal:** Prove the adapter abstraction works across providers.

> Previuos version had anthopic adapter, I've defer it for while, put a note on future.md

**Deliverables:**
- `nodes/llm_adapters/gemini_adapter.py` — Implements `TextAdapter`, `StreamAdapter`, `StructuredAdapter`, `EmbeddingAdapter`. Error mapping for Gemini exceptions.

**Tests:**
- Unit tests with mocked SDK responses per provider
- Verify that the same node works with different adapters swapped in
- Integration tests (optional, requires API keys)

**Exit criteria:** All four nodes work with all three providers (where supported). Error mapping is consistent.

---

### Phase 7 — Polish, Documentation, Examples

**Goal:** Make the package usable and documented.

**Deliverables:**
- `__init__.py` public exports (clean API surface)
- README update with installation, quick start, and usage examples
- Example scripts in `examples/` directory
- Docstrings on all public classes and methods

**Exit criteria:** A new user can install, configure an adapter, and run all four node types from the README alone.

---

## 6. Deferred Items

See [future.md](./future.md) for deferred design decisions and future enhancement plans.
