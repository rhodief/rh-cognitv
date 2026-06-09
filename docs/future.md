# Future Enhancements — Deferred Design

> Items explicitly deferred from the [execution nodes spec](./01_execution_nodes_spec.md).
> These are not in scope for v1 but are tracked here for future phases.

---

## 1. EventBus & Observability Infrastructure

**What:** A framework-wide async event bus (pub/sub) for decoupled event-driven observability.

**Current state:** Nodes use simple `on_event`, `on_request`, `on_response` callbacks. This works for v1 but limits us to one consumer per hook.

**How it enhances the package:**
- **Multiple subscribers** — loggers, cost trackers, compliance auditors, and dashboards all subscribe independently to the same event stream without modifying node code.
- **Event persistence** — events can be forwarded to external brokers (Kafka, Redis Streams, etc.) for audit trails and replay.
- **Runtime integration** — when combined with a future runtime, the EventBus becomes the observability backbone for the entire framework: execution lifecycle events, LLM events, flow events, all unified.

**Design considerations when building:**
- The `LLMStreamNode` already has `on_event: Callable | None`. When EventBus is built, it should be compatible as a callback consumer (e.g., `on_event=event_bus.emit`).
- The SG-01 observability callbacks (`on_request`, `on_response`) should migrate to EventBus events: `llm_request_sent`, `llm_response_received`.
- Consider queue-based architecture with subscriber isolation (one slow subscriber shouldn't block others).

---

## 2. Runtime / Execution Engine

**What:** A generic execution engine that runs arbitrary async callables with retries, hooks, timeouts, and parallel execution.

**Current state:** Nodes implement `async __call__()` (DD-08) specifically so a future runtime can consume them.

**How it enhances the package:**
- **Retry policies** — automatic retry on transient LLM errors (rate limits, timeouts) with exponential backoff, without the user writing retry loops.
- **Lifecycle hooks** — intercept execution at before_start, before_attempt, after_finish, etc. for cross-cutting concerns (auth checks, metadata enrichment, caching).
- **Parallel execution** — run multiple LLM calls concurrently with max worker limits and aggregate results.
- **Error classification** — the canonical error taxonomy from DD-09 (`family`, `code`, `retryable`) feeds directly into the retry engine's decision logic.

**Design considerations when building:**
- The `ToolValidationError` (SG-05) should be treated as retryable by default — the runtime can re-prompt the LLM with the validation error message.
- Streaming nodes (`LLMStreamNode`) return async generators, not plain return values. The runtime must handle this (e.g., via a `run_to_completion()` variant or a streaming-aware execution path).

---

## 3. FunctionNodes

**What:** Execution nodes that wrap arbitrary Python functions (sync or async) with the same canonical interface (result objects, error taxonomy, `__call__` protocol).

**How it enhances the package:**
- Enables building **hybrid pipelines** that mix LLM calls with regular Python logic (data transformation, validation, API calls) under the same execution model.
- Functions get the same retry, hooks, and observability as LLM nodes — no special-casing.
- Natural building block for agent tool execution — when an LLM calls a tool, the tool's implementation runs as a FunctionNode.

---

## 4. FlowNodes (ForEach, MapReduce, Conditional, DAG)

**What:** Higher-order nodes that orchestrate other nodes in control-flow patterns.

**How it enhances the package:**
- **ForEach** — iterate over a collection, running a node per item (with configurable concurrency).
- **MapReduce** — fan-out to multiple nodes, aggregate results.
- **Conditional** — branch execution based on a predicate or LLM output.
- **DAG** — define directed acyclic graphs of node dependencies, execute with topological ordering and parallelism.
- These transform the package from "LLM call wrappers" into a **full orchestration framework**.

---

## 5. Agent Loops

**What:** Multi-step reasoning loops where the LLM observes, thinks, acts (calls tools), and observes again until a goal is met.

**How it enhances the package:**
- Core pattern for agentic AI: ReAct loops, tool-use chains, plan-and-execute.
- Built on top of `LLMStructuredNode` (for tool calls) + `FunctionNode` (for tool execution) + `FlowNodes` (for the loop structure).
- The `ToolDefinition` wrapper (DD-06) and auto-validation (SG-05) become critical here — the agent needs validated tool arguments to execute reliably.

---

## 6. Memory / Context Management

**What:** Conversation history management, context window tracking, and automatic truncation/summarization.

**How it enhances the package:**
- LLMs have finite context windows. As conversations grow, automatic management prevents context overflow errors.
- Strategies: sliding window, summarization of old messages, retrieval-augmented generation (RAG) with embeddings.
- The `LLMEmbeddingNode` provides the foundation for RAG-based memory retrieval.
- The `Message` model (DD-10) already supports multi-turn conversations — memory management extends this with persistence and retrieval.

---

## 7. Multi-Modal Inputs

**What:** Support for images, audio, video, and documents in message content.

**How it enhances the package:**
- Modern LLMs (GPT-4o, Claude 3.5, Gemini) support multi-modal inputs.
- The `Message` model would need a richer `content` field (e.g., `list[TextContent | ImageContent | AudioContent]` instead of plain `str`).
- The adapter layer would need to map multi-modal content to each provider's format.

**Design considerations when building:**
- The current `Message(role=..., content=str)` model would need a breaking change or a union type for content. Plan this migration carefully.

---

## 8. Deferred Design Decisions

### Time-Based Batch Flushing (from DD-05)

Count-based batching (v1) doesn't account for timing. If chunks arrive slowly, the user waits for N chunks before seeing anything. A `batch_timeout_ms` parameter that forces a flush after T ms (even if N chunks haven't arrived) would improve real-time UX. Add this when real-world streaming usage reveals the need.

### Hybrid Tool Definition Format (from DD-06)

v1 uses explicit `ToolDefinition` wrappers. A future enhancement could accept raw Pydantic model classes directly and derive `name` from the class name, `description` from the docstring. This is a DX improvement for simple cases, additive and non-breaking.

### Adapter Registry / Factory (from SG-02)

v1 uses manual adapter instantiation. When the provider ecosystem grows (5+ adapters, third-party contributions), a registry pattern (`@register_adapter("openai")`) or factory function (`create_adapter("openai", api_key=...)`) would reduce boilerplate and enable config-driven adapter selection.

### EventBus Observability (from SG-01)

v1 uses callback hooks. When the EventBus is built (item #1 above), migrate observability from callbacks to structured event emission (`llm_request_sent`, `llm_response_received`). This enables multiple subscribers and integrates with the framework-wide observability infrastructure.

---

## 9. Anthropic Adapter (Deferred)

**What:** A `ClaudeAdapter` / `AnthropicAdapter` implementing the canonical adapter contract (`TextAdapter`, `StreamAdapter`, `StructuredAdapter`) for Anthropic Claude models.

**Current state:** A previous iteration included an Anthropic adapter. It has been deferred for now to keep Phase 6 focused on the Gemini adapter. The `anthropic` optional dependency group still exists in `pyproject.toml` so the adapter can be added back without packaging changes.

**How it enhances the package:**
- Adds a third first-party provider, strengthening the multi-provider story (OpenAI + Gemini + Anthropic) and proving the adapter abstraction across three distinct SDK shapes.
- Claude's tool-use API maps cleanly onto `LLMStructuredNode` via the `ToolDefinition` wrapper (DD-06) and tool-argument validation (SG-05).

**Design considerations when building:**
- Mirror the structure of [openai_adapter.py](../rh_cognitv/nodes/llm_adapters/openai_adapter.py) and [gemini_adapter.py](../rh_cognitv/nodes/llm_adapters/gemini_adapter.py): `provider = "anthropic"`, constructor injection of the client (DD-03), lazy SDK import (SG-03), and a `map_anthropic_exception()` mapping HTTP status to the canonical `LLMError` taxonomy (DD-09).
- Anthropic has **no native embeddings API**, so the adapter should implement only the text / stream / structured capabilities (not `EmbeddingAdapter`) — exactly the per-capability ABC composition that DD-02 was designed for.
- System prompts are a top-level `system` parameter (not a message role), similar to Gemini's `system_instruction`.
