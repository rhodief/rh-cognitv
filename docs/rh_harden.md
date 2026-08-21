# rh_cognitv Hardening Specification — v0.1.0

> **Purpose**: Development spec for the next phase of `rh_cognitv`. Each section follows the pattern:
> **Issue** → **Why it matters** → **Decision** → **POI specification**.
> Tensions, ambiguities, and contradictions in the current implementation are called out explicitly.

---

## Notation

- **DD-H##** — Design Decision (Hardening phase)
- **POI-##** — Point of Improvement (implementation unit)
- **T-##** — Tension / Contradiction identified in the current codebase
- References to the original codebase specs use the existing DD-## notation (e.g., DD-04 for stream events)

---

## Current State Summary

```
rh_cognitv v0.0.0b0
├── nodes/           # 4 LLM nodes + FunctionNode — solid
├── llm_adapters/    # OpenAI + Gemini — solid
├── agents/
│   ├── context.py   # ActiveContext data model — well-designed, hollow in places
│   └── orchestrator.py  # Reasoning loop — functional, fragile
└── event_bus.py     # Pub/sub — working, incomplete coverage
```

---

## Identified Tensions & Contradictions

Before diving into design decisions, these are the structural inconsistencies that the hardening phase must resolve. Each is referenced by its T-## identifier within the relevant DD-H section.

### T-01 — Asymmetric compaction philosophy

**Observations** are aggressively pruned (sliding window, K parameter) via [apply_hygiene()](file:///app/rh_cognitv/agents/context.py#L84-L92), but the **message history** (`messages: list[Message]` in [orchestrator.py:180](file:///app/rh_cognitv/agents/orchestrator.py#L180)) grows unbounded. These are both context-window resources competing for the same token budget, yet one is managed and the other is not. The system prompt even tells the agent:

> *"Your active context is a scarce resource and raw observations are pruned automatically."*

…but says nothing about message history being a scarce resource too.

### T-02 — Retryable errors without a retry mechanism

[ToolValidationError](file:///app/rh_cognitv/nodes/llm/errors.py#L125-L134) is `retryable = True` by design:

> *"Retryable by default: a future runtime can re-prompt the model with the validation error message."*

Similarly, [RateLimitError](file:///app/rh_cognitv/nodes/llm/errors.py#L77-L82), [TimeoutError](file:///app/rh_cognitv/nodes/llm/errors.py#L109-L114), and [ProviderError](file:///app/rh_cognitv/nodes/llm/errors.py#L117-L122) are all `retryable = True`. But no component in the framework actually consults this flag. The error taxonomy was built for a retry engine that doesn't exist yet — meaning the `retryable` field is currently dead metadata.

### T-03 — Hollow data model fields

[ActiveContext](file:///app/rh_cognitv/agents/context.py#L66-L82) declares three fields that exist as data structures but have **no tools to populate them** and **no orchestrator logic that reads them**:

| Field | Type | Populated by | Read by |
|---|---|---|---|
| `artifacts` | `list[str]` | Nothing | `format_context()` renders it, but it's always empty |
| `retrieval_ledger` | `list[RetrievalEntry]` | Nothing | `format_context()` renders it, but it's always `(Empty)` |
| `plan` | `str \| None` | Nothing | `format_context()` renders it as `(No plan created yet)` |

These fields create a **promise to the LLM** (they appear in the system prompt) that the framework cannot fulfill. The LLM sees sections like `# RETRIEVAL LEDGER\n(Empty)` and `# AVAILABLE ARTIFACTS\n(None)` and may attempt to interact with them, but no tools exist.

### T-04 — FunctionNode error swallowing vs. LLM node error propagation

[FunctionNode.run()](file:///app/rh_cognitv/nodes/function_node.py#L55-L67) catches **all** exceptions and returns them as a `FunctionResult(error=str(exc))` — the caller never sees the exception. In contrast, LLM nodes (e.g., [LLMStreamNode](file:///app/rh_cognitv/nodes/llm/stream_node.py#L276-L280)) re-raise canonical `LLMError` exceptions. This means:

- A failing external tool **silently succeeds** (returns a result with an error string)
- A failing LLM call **loudly crashes** (raises an exception)

The orchestrator [handles FunctionNode errors](file:///app/rh_cognitv/agents/orchestrator.py#L396-L399) by checking `res.error`, but it treats them as observations, not as failures that might warrant retry or escalation.

### T-05 — Classification call is invisible

The [complexity classifier](file:///app/rh_cognitv/agents/orchestrator.py#L107-L126) makes a full LLM call (`llm_node.collect()`) with a custom prompt, but:
- No `AgentStepStarted`/`AgentStepCompleted` events are emitted
- The classification result is not published to the EventBus
- Token usage from this call is untracked (not in any `StepCompleted` event)
- The `silent_sink` callback explicitly discards all stream events

This creates a gap in cost tracking and observability.

### T-06 — Simple-task path returns ambiguous state

The [simple-task branch](file:///app/rh_cognitv/agents/orchestrator.py#L128-L175) returns an `ActiveContext` with an empty `todo.steps` list. The [complex-task termination check](file:///app/rh_cognitv/agents/orchestrator.py#L346-L349) also considers `len(context.todo.steps) > 0 and all(s.status == "done")` as "all done." But if the complex path ends at `max_steps` with zero TODO steps created (the LLM never called `todo.create`), it looks identical to a simple-task completion. The caller has no way to distinguish:

- Simple task, answered directly ✅
- Complex task, completed all TODOs ✅  
- Complex task, LLM never created TODOs (stuck/confused) ❌
- Complex task, hit max_steps with work remaining ❌

### T-07 — EventBus fire-and-forget async handlers

[EventBus.publish()](file:///app/rh_cognitv/event_bus.py#L136-L165) uses `asyncio.create_task()` for coroutine handlers and silently catches all exceptions. This is intentional (telemetry shouldn't crash the main loop), but it means:

- Subscriber errors are **completely invisible** — no logging, no event, no callback
- If a subscriber raises, the task's exception is never awaited, causing `asyncio` runtime warnings
- There's no backpressure — a slow subscriber with high event volume creates unbounded task accumulation

### T-08 — Auto-memory is write-once, read-always

[auto_memory](file:///app/rh_cognitv/agents/context.py#L77) is injected by the caller at `run_task()` time and rendered in every system prompt, but the agent has **no tool to modify it**. If the agent discovers a correction to an auto-memory entry during execution, it cannot update it. This makes auto-memory a static assertion that may become stale mid-task.

---

## DD-H01 — EventBus Context Transparency

### Issue

The EventBus provides good **output-side** observability (what the agent said, what tools it called, what facts it distilled) but has a critical blind spot on the **input-side**: subscribers cannot see what the LLM actually received as its context window. This makes it impossible to debug prompt engineering issues, audit context quality, or build external dashboards showing the LLM's "world view" at each step.

### Why it matters

- **Debugging**: When an agent makes a wrong decision, the first question is "what did the LLM see?" Currently, answering that requires reading the orchestrator source code and mentally reconstructing the prompt.
- **Cost accounting**: The classification call (T-05) consumes tokens that are invisible to any subscriber.
- **Compliance**: Regulated environments may require logging every LLM input for audit.

### Decision

Introduce four new event types that complete the observability picture. Events are **informational snapshots**, not control signals — subscribers observe but never modify the data.

### POI-01 — `AgentContextSnapshot` event

> Resolves: T-05 (classification visibility)

Emitted **before every LLM call** in the orchestrator, including the classification call.

```python
class AgentContextSnapshot(AgentEvent):
    """Full input-side snapshot of what the LLM will receive."""
    type: Literal["agent_context_snapshot"] = "agent_context_snapshot"
    step_index: int                          # 0 for classification, 1+ for reasoning steps
    step_kind: Literal["classification", "simple_answer", "reasoning"]
    messages: list[dict[str, str]]           # Serialized message history
    tool_names: list[str]                    # Tool definitions available
    context_tokens_estimate: int | None = None  # See POI-03
```

**Where to emit**: [orchestrator.py:120](file:///app/rh_cognitv/agents/orchestrator.py#L120) (before classification call), [orchestrator.py:143](file:///app/rh_cognitv/agents/orchestrator.py#L143) (before simple-answer call), [orchestrator.py:311](file:///app/rh_cognitv/agents/orchestrator.py#L311) (before each reasoning step call).

**Design note on `messages` serialization**: Use `[{"role": m.role, "content": m.content} for m in messages]` — intentionally not the raw `Message` Pydantic model, so subscribers don't depend on internal types. Content may be truncated for very large payloads (configurable `max_snapshot_content_chars`).

---

### POI-02 — `AgentContextPruned` event

> Resolves: T-01 (asymmetric compaction — observability half)

Emitted when [apply_hygiene()](file:///app/rh_cognitv/agents/context.py#L84-L92) drops observations.

```python
class AgentContextPruned(AgentEvent):
    """Emitted when hygiene rules prune observations from working memory."""
    type: Literal["agent_context_pruned"] = "agent_context_pruned"
    pruned_ids: list[str]           # Observation IDs removed
    pruned_count: int
    remaining_count: int
    budget_k: int                   # The max_active_observations value
```

**Implementation**: Modify `apply_hygiene()` to return the pruned IDs, then emit from the orchestrator after calling it at [orchestrator.py:227](file:///app/rh_cognitv/agents/orchestrator.py#L227).

---

### POI-03 — Context token estimation

> Supports: POI-01 (`context_tokens_estimate` field), POI-08 (message compaction trigger)

Add a `estimate_tokens(text: str) -> int` utility using a simple heuristic (chars/4 or a tiktoken-based counter if available).

```python
# rh_cognitv/agents/token_utils.py

def estimate_tokens(text: str, method: str = "chars") -> int:
    """Estimate token count for a string.
    
    Methods:
      - "chars": len(text) // 4 (fast, ~80% accurate)
      - "tiktoken": Use tiktoken cl100k_base (accurate, requires optional dep)
    """
```

**Decision**: Default to the chars/4 heuristic. `tiktoken` is an optional dependency, not a requirement. The estimate doesn't need to be precise — it's for budget tracking and compaction triggers, not billing.

---

### POI-04 — `AgentNotebookUpdated` event

Currently, `notebook.append` is the only internal tool with **no dedicated bus event**. Add:

```python
class AgentNotebookUpdated(AgentEvent):
    type: Literal["agent_notebook_updated"] = "agent_notebook_updated"
    entry: str
    total_entries: int
```

**Emit from**: [orchestrator.py:438](file:///app/rh_cognitv/agents/orchestrator.py#L438), add an `elif` branch for `notebook.append`.

---

### POI-05 — `AgentClassificationResult` event

> Resolves: T-05 directly

```python
class AgentClassificationResult(AgentEvent):
    type: Literal["agent_classification_result"] = "agent_classification_result"
    classification: Literal["simple", "complex"]
    raw_response: str
    tokens_total: int | None = None
```

**Emit from**: After [orchestrator.py:125-126](file:///app/rh_cognitv/agents/orchestrator.py#L125-L126), before branching.

---

## DD-H02 — Context Compaction Pipeline

### Issue

The `messages: list[Message]` array in the orchestrator loop grows by at least 2 messages per step (assistant reply + user nudge/tool-results). For a 15-step task, this can reach 30+ messages totaling thousands of tokens. Combined with the system prompt (which is refreshed with current context state each step), this will eventually exceed the model's context window, causing a `ContextLengthError`.

### Why it matters

> [!CAUTION]
> This is the **single most likely production failure mode**. A long-running agent task will silently accumulate context until the LLM call fails with a context-length error. The error is not retryable (the context is too large, not transiently failing), so the entire task is lost.

### Tensions addressed

- **T-01**: Observations are compacted but messages are not. This spec resolves the asymmetry by introducing message compaction.
- **T-03 (partial)**: The `plan` field on `ActiveContext` is never populated — the compaction system could optionally use it to store a condensed task plan.

### Decision

Implement a **two-stage compaction pipeline** triggered by token-budget thresholds:

```
Stage 1: Message Summarization (LLM-driven)
  - When estimated message tokens > threshold, summarize older messages
  - Replace N oldest non-system messages with a single summary message
  
Stage 2: Observation Window (already exists)
  - apply_hygiene() keeps only K most-recent observations
```

### POI-06 — Message compaction engine

```python
# rh_cognitv/agents/compaction.py

class CompactionConfig(BaseModel):
    """Configuration for the context compaction pipeline."""
    enabled: bool = True
    token_budget: int = 8000          # Trigger compaction when messages exceed this
    preserve_recent: int = 4          # Always keep the last N messages verbatim
    summary_model: str | None = None  # Model to use for summarization (defaults to task model)
    summary_max_tokens: int = 300     # Max tokens for the summary

class CompactionResult(BaseModel):
    """Result of a compaction pass."""
    original_message_count: int
    compacted_message_count: int
    summarized_range: tuple[int, int]   # (start_index, end_index) of messages summarized
    summary_text: str
    estimated_tokens_saved: int
```

**Integration point in orchestrator**: At the start of each step, after refreshing the system prompt and before calling the LLM:

```python
# orchestrator.py, inside the step loop, after line 308
if compaction_config.enabled:
    messages, compaction_result = await self._maybe_compact(messages, config, compaction_config)
    if compaction_result:
        await self._publish(AgentContextCompacted(...))
```

**New event**:

```python
class AgentContextCompacted(AgentEvent):
    type: Literal["agent_context_compacted"] = "agent_context_compacted"
    messages_before: int
    messages_after: int
    tokens_saved_estimate: int
    summary_preview: str    # First 200 chars of the summary
```

> [!IMPORTANT]
> **Design tension — who does the summarization?**
>
> **Option A**: Use the same LLM as the task (reuse `self.llm_node`). Pro: no extra configuration. Con: the summarization call itself consumes tokens and time within the step budget, and it's invisible unless we emit events for it.
>
> **Option B**: Use a dedicated cheaper/faster model (configurable via `summary_model`). Pro: cost-efficient, doesn't pollute the task model's context. Con: requires an extra LLM call configuration.
>
> **Decision**: Default to Option A (same model) for simplicity, but accept `summary_model` override. The summarization call MUST emit an `AgentContextSnapshot` event with `step_kind="compaction"` so it's observable.

---

### POI-07 — Context budget tracking

Add a `context_budget` parameter to `run_task()` and track cumulative token usage:

```python
class ContextBudget(BaseModel):
    """Token budget tracking across the full task lifecycle."""
    max_total_tokens: int | None = None       # Hard cap on total tokens consumed
    max_context_tokens: int | None = None     # Max tokens in any single LLM call
    cumulative_prompt_tokens: int = 0
    cumulative_completion_tokens: int = 0
    
    @property
    def cumulative_total(self) -> int:
        return self.cumulative_prompt_tokens + self.cumulative_completion_tokens
    
    def check_budget(self) -> bool:
        """Returns True if within budget, False if exceeded."""
        if self.max_total_tokens and self.cumulative_total >= self.max_total_tokens:
            return False
        return True
```

**New event**:

```python
class AgentBudgetExceeded(AgentEvent):
    type: Literal["agent_budget_exceeded"] = "agent_budget_exceeded"
    budget_kind: Literal["total_tokens", "context_tokens"]
    limit: int
    actual: int
```

---

## DD-H03 — State Persistence & Crash Recovery

### Issue

All agent state is held in local variables within `run_task()`. If the process crashes, is killed, or encounters an unrecoverable error, all progress — facts, decisions, TODO state, notebook entries, and the conversation history — is permanently lost.

### Why it matters

- Agent tasks can run for minutes (15 steps with LLM calls + tool execution).
- In production, processes may be pre-empted (Kubernetes pod eviction, OOM kills, deployment rotations).
- Users expect to resume interrupted work, especially for expensive long-running tasks.

### Tensions addressed

- **T-03 (partial)**: Hollow fields (`artifacts`, `retrieval_ledger`, `plan`) become more meaningful once state is persistable — they become resumable bookmarks.
- **T-06**: With checkpointing, the distinction between "simple completed" and "complex stuck" becomes explicit in the checkpoint metadata.

### Decision

Implement an **opt-in checkpoint protocol** using the existing Pydantic serialization. The checkpoint is a complete snapshot that enables resuming from any completed step.

> [!IMPORTANT]
> **Key design decision**: Checkpoints are **passive** (serialized state) not **active** (event-sourced replay). We serialize the full state, not a log of events. This is simpler, doesn't require event ordering guarantees, and leverages the existing Pydantic models directly.

### POI-08 — Checkpoint model & serialization

```python
# rh_cognitv/agents/checkpoint.py

class AgentCheckpoint(BaseModel):
    """Complete snapshot of an agent's execution state at a step boundary."""
    version: str = "1"                # Schema version for forward compat
    task: str
    agent_id: str
    step_index: int                   # Step that just completed
    status: Literal["in_progress", "completed", "failed", "max_steps_reached"]
    
    # Serialized state
    context: ActiveContext             # Full cognitive state
    messages: list[Message]            # Conversation history
    
    # Metadata
    total_tokens_used: int = 0
    total_duration_ms: float = 0.0
    created_at: float = Field(default_factory=time.time)
    
    # Classification result (so we don't re-classify on resume)
    is_complex: bool = True
```

### POI-09 — Checkpoint store interface

```python
class CheckpointStore(abc.ABC):
    """Abstract interface for persisting agent checkpoints."""
    
    @abc.abstractmethod
    async def save(self, checkpoint: AgentCheckpoint) -> None: ...
    
    @abc.abstractmethod
    async def load(self, agent_id: str, task: str) -> AgentCheckpoint | None: ...
    
    @abc.abstractmethod
    async def delete(self, agent_id: str, task: str) -> None: ...


class FileCheckpointStore(CheckpointStore):
    """Persists checkpoints as JSON files in a directory."""
    
    def __init__(self, directory: Path) -> None:
        self.directory = directory


class InMemoryCheckpointStore(CheckpointStore):
    """In-memory store for testing."""
    ...
```

**Decision — storage backend**: Ship with `FileCheckpointStore` (JSON files) and `InMemoryCheckpointStore` (testing). The ABC allows users to implement Redis, SQLite, S3, etc. without framework changes.

### POI-10 — Resume protocol in `run_task()`

Extend `run_task()` signature:

```python
async def run_task(
    self,
    task: str,
    config: LLMConfig,
    *,
    max_steps: int = 10,
    max_active_observations: int = 3,
    auto_memory: list[str] | None = None,
    # New parameters:
    checkpoint_store: CheckpointStore | None = None,     # Enable checkpointing
    resume_from: AgentCheckpoint | None = None,          # Resume from a checkpoint
    compaction: CompactionConfig | None = None,          # See DD-H02
    context_budget: ContextBudget | None = None,         # See DD-H02/POI-07
) -> ActiveContext:
```

**Resume semantics**:
1. If `resume_from` is provided, skip classification, restore `context` and `messages`, and continue from `step_index + 1`.
2. If `checkpoint_store` is provided, save a checkpoint after each completed step.
3. On completion or failure, save a final checkpoint with terminal status.

> [!WARNING]
> **Ambiguity — tool idempotency**: If step N called `external.read_file("config.json")` and the checkpoint was saved after step N, resuming from step N+1 is safe (the tool result is already in observations). But if the crash happened **mid-step** (after some tool calls but before all), we face a partial-execution problem. 
>
> **Decision**: Checkpoints are only saved at **step boundaries** (after all tool calls in a step complete). A crash mid-step means that step is replayed from scratch. This is safe for read-only tools but may cause side effects for write tools.
>
> **Mitigation for write tools**: Document that action tools used with checkpointing should be idempotent. Provide a `call_id` in the checkpoint so advanced users can implement deduplication.

---

## DD-H04 — Retry & Resilience Engine

### Issue

The error taxonomy ([errors.py](file:///app/rh_cognitv/nodes/llm/errors.py)) meticulously classifies errors with `retryable` flags, but no component in the framework acts on them. A single `RateLimitError` from the LLM provider crashes the entire agent task.

### Why it matters

- LLM APIs have **high transient failure rates** (rate limits, timeouts, server errors).
- The `retryable` flag on errors is a promise to users that the framework will eventually handle retries — currently broken.
- `ToolValidationError` being retryable implies the LLM should be re-prompted with the validation error — this is a critical agentic pattern for self-correction.

### Tensions addressed

- **T-02**: Directly resolves the "retryable without retry" contradiction.
- **T-04**: Clarifies the FunctionNode error-swallowing behavior in the context of retry decisions.

### Decision

Implement retry at **two levels**:

1. **LLM call retry** (transient errors: rate limit, timeout, provider error) — exponential backoff, configurable max attempts.
2. **Tool validation retry** (re-prompt the LLM with the validation error) — the orchestrator re-injects the error as context and asks the LLM to correct its tool call.

### POI-11 — LLM call retry wrapper

```python
# rh_cognitv/agents/retry.py

class RetryConfig(BaseModel):
    """Configuration for LLM call retry behavior."""
    max_attempts: int = 3
    initial_delay_ms: int = 1000
    max_delay_ms: int = 30000
    backoff_factor: float = 2.0
    retryable_families: set[str] = Field(default_factory=lambda: {
        "rate_limit", "timeout", "provider"
    })

class RetryOutcome(BaseModel):
    attempts: int
    last_error: str | None = None
    total_retry_delay_ms: float = 0.0
```

**New event**:

```python
class AgentRetryAttempt(AgentEvent):
    type: Literal["agent_retry_attempt"] = "agent_retry_attempt"
    step_index: int
    attempt: int
    max_attempts: int
    error_family: str
    delay_ms: float
    error_message: str
```

**Integration**: Wrap the `self.llm_node.run()` call in the orchestrator's step loop with a retry-aware wrapper. The wrapper checks `exc.retryable` and `exc.family` against the `RetryConfig`.

### POI-12 — Tool validation self-correction

When a `ToolValidationError` occurs during tool-call execution, instead of crashing:

1. Append the validation error as a user-role message: `"Tool call error: {validation_error}. Please correct your tool call arguments."`
2. Re-run the LLM step (counting against `max_attempts`).
3. Emit an `AgentRetryAttempt` event with `error_family="validation"`.

**Where**: In the orchestrator's tool execution block, after [orchestrator.py:293-300](file:///app/rh_cognitv/agents/orchestrator.py#L292-L300) where `tool_accumulator.build()` and `validate_tool_calls()` are called.

> [!NOTE]
> **Tension with FunctionNode error handling (T-04)**: FunctionNode swallows exceptions and returns `FunctionResult(error=...)`. The orchestrator already handles this gracefully — it adds the error as an observation. This is actually the **correct** behavior for tool-execution errors (the agent should see the error and decide how to proceed). The retry mechanism described here is specifically for **tool-argument validation** errors (the LLM produced malformed arguments), not for tool-execution errors. These are distinct failure modes.

---

## DD-H05 — Artifact & Retrieval System

### Issue

`ActiveContext` declares `artifacts: list[str]` and `retrieval_ledger: list[RetrievalEntry]`, and `format_context()` renders them into the system prompt. But no tools exist to populate them, making them phantom sections that mislead the LLM.

### Why it matters

- The LLM sees `# AVAILABLE ARTIFACTS\n(None)` in its context and may try to interact with an artifact system that doesn't exist.
- The retrieval ledger was designed for tracking investigation steps, but its absence forces the agent to duplicate this tracking in notebook entries or facts.
- Without artifacts, the agent has no way to produce persistent, structured outputs (e.g., a report, a generated file).

### Tensions addressed

- **T-03**: Directly resolves the hollow data model fields.
- **T-08 (partial)**: An artifact system could provide a way for the agent to persist knowledge that outlives auto-memory.

### Decision

Implement **three new internal tools** and wire them to the existing data model fields. Artifacts are **in-memory strings** in v0.1.0 (not files on disk). Persistence comes from the checkpoint system (DD-H03).

### POI-13 — Artifact tools

```python
# Added to get_default_context_tools() in context.py

def artifact_create(name: str, content: str) -> str:
    """Create or overwrite a named artifact (e.g., a report, summary, or generated content)."""
    # Store as "name::content" in context.artifacts for simplicity,
    # or use a dict-based artifact store (see design note below)
    ...

def artifact_read(name: str) -> str:
    """Read the content of a previously created artifact by name."""
    ...

def artifact_list() -> str:
    """List all available artifact names."""
    ...
```

> [!IMPORTANT]
> **Design tension — artifact storage structure**:
>
> The current `artifacts: list[str]` field stores artifact names as plain strings. To store content, we need either:
>
> **Option A**: Change to `artifacts: dict[str, str]` (name → content). Clean, but it's a breaking change to the Pydantic model (affects serialization, format_context, tests).
>
> **Option B**: Add a parallel `artifact_contents: dict[str, str]` field, keep `artifacts` as the name list for backward compat.
>
> **Decision**: Go with Option A. The package is `v0.0.0b0` — breaking changes are expected. Update `format_context()` to render artifact names and a content preview.

**Updated model field**:

```python
class ActiveContext(BaseModel):
    # ...existing fields...
    artifacts: dict[str, str] = Field(default_factory=dict)  # name → content
```

**New event**:

```python
class AgentArtifactCreated(AgentEvent):
    type: Literal["agent_artifact_created"] = "agent_artifact_created"
    artifact_name: str
    content_length: int
```

---

### POI-14 — Retrieval ledger tool

```python
def retrieval_log(topic: str, outcome: str, source: str, 
                  status: Literal["pending", "resolved", "failed"] = "resolved") -> str:
    """Log a retrieval/investigation step in the retrieval ledger."""
    entry = RetrievalEntry(topic=topic, outcome=outcome, source=source, status=status)
    context.retrieval_ledger.append(entry)
    return f"Logged retrieval: {topic} [{status}]"
```

**System prompt update**: Add `retrieval.log(topic, outcome, source, status)` to the internal tools documentation in the orchestrator system prompt.

---

### POI-15 — Plan tool

> Resolves the unused `plan` field in ActiveContext

```python
def plan_set(plan: str) -> str:
    """Set or update the high-level execution plan for the current task."""
    context.plan = plan
    return "Plan updated."
```

**Design note**: The `plan` field is distinct from `todo.steps`. The plan is a free-form text description of the approach; the TODO is a structured checklist. The system prompt should instruct the agent to set a plan before creating TODO steps.

---

## DD-H06 — Tool Ecosystem Completion

### Issue

Beyond the missing tools for hollow fields (DD-H05), there are functional gaps in the internal tool set that limit the agent's self-management capabilities.

### POI-16 — Notebook search

Currently, `notebook.append` is the only notebook tool. For tasks with many notebook entries, the agent has no way to search or retrieve relevant entries.

```python
def notebook_search(query: str) -> str:
    """Search notebook entries for entries containing the query string (case-insensitive)."""
    matches = [entry for entry in context.notebook_entries if query.lower() in entry.lower()]
    if not matches:
        return f"No notebook entries matching '{query}'."
    return "\n---\n".join(matches)
```

**Why not embedding-based search?**: For v0.1.0, substring search is sufficient. Embedding-based retrieval (using `LLMEmbeddingNode`) is a v0.2.0 enhancement — it requires maintaining an embedding index, which adds significant complexity.

---

### POI-17 — Auto-memory mutation tool

> Resolves: T-08

```python
def auto_memory_add(entry: str) -> str:
    """Add a new persistent fact to auto-memory."""
    context.auto_memory.append(entry)
    return f"Added to auto-memory: {entry}"

def auto_memory_remove(entry: str) -> str:
    """Remove an entry from auto-memory (exact match)."""
    if entry in context.auto_memory:
        context.auto_memory.remove(entry)
        return f"Removed from auto-memory: {entry}"
    return f"Entry not found in auto-memory: {entry}"
```

**Design note**: Auto-memory is still injected by the caller at `run_task()` time, but the agent can now refine it during execution.

---

### POI-18 — Fact management tools

Currently, facts can only be added. The agent may extract a fact that later proves incorrect. Add:

```python
def context_retract_fact(fact_id: str) -> str:
    """Retract a previously extracted fact by its ID."""
    for i, f in enumerate(context.recent_facts):
        if f.id == fact_id:
            context.recent_facts.pop(i)
            return f"Retracted fact: {fact_id}"
    return f"Fact {fact_id} not found."
```

---

## DD-H07 — Error & Signal Consistency

### Issue

The current codebase has inconsistent error handling, signal propagation, and completion semantics across components.

### Tensions addressed

- **T-04**: FunctionNode error swallowing vs. LLM error propagation
- **T-06**: Ambiguous completion state
- **T-07**: EventBus silent error swallowing

### POI-19 — Task completion status model

> Resolves: T-06

Add an explicit completion status to `run_task()` return value:

```python
class TaskCompletionStatus(str, Enum):
    COMPLETED = "completed"              # All TODOs done
    SIMPLE_ANSWERED = "simple_answered"  # Simple task, direct answer
    MAX_STEPS_REACHED = "max_steps_reached"  # Hit step limit
    BUDGET_EXCEEDED = "budget_exceeded"  # Hit token budget
    ERROR = "error"                      # Unrecoverable error

class TaskResult(BaseModel):
    context: ActiveContext
    status: TaskCompletionStatus
    total_steps: int
    total_tokens: int
    total_duration_ms: float
    error: str | None = None             # Set when status is ERROR
```

**Change `run_task()` return type** from `ActiveContext` to `TaskResult`. This is a breaking change, but necessary for clarity.

> [!WARNING]
> **Breaking change**: All existing callers of `run_task()` will need to update from `context = await orchestrator.run_task(...)` to `result = await orchestrator.run_task(...)` and access `result.context`.

---

### POI-20 — EventBus error reporting

> Resolves: T-07

Add optional error handling to the EventBus without changing the non-crashing guarantee:

```python
class EventBus:
    def __init__(self, on_handler_error: Callable[[Exception, Any], None] | None = None) -> None:
        self._subscribers: dict[Any, set[Callable]] = {}
        self._on_handler_error = on_handler_error

    async def publish(self, event: Any) -> None:
        # ...existing logic...
        for handler in handlers:
            try:
                if inspect.iscoroutinefunction(handler):
                    task = asyncio.create_task(handler(event))
                    task.add_done_callback(self._handle_task_exception)
                else:
                    handler(event)
            except Exception as exc:
                if self._on_handler_error:
                    self._on_handler_error(exc, event)
    
    def _handle_task_exception(self, task: asyncio.Task) -> None:
        if task.exception() and self._on_handler_error:
            self._on_handler_error(task.exception(), None)
```

**Rationale**: The default behavior (silent swallow) remains unchanged. Users who pass `on_handler_error` get visibility into subscriber failures. The `add_done_callback` approach avoids the `asyncio` runtime warning about unhandled task exceptions.

---

### POI-21 — FunctionNode error severity levels

> Addresses: T-04

Rather than changing FunctionNode's error-swallowing behavior (which is correct for the orchestrator's needs), add metadata so the orchestrator can make smarter decisions:

```python
class FunctionResult(BaseModel):
    output: Any
    duration_ms: float
    error: str | None = None
    error_retryable: bool = False      # NEW: Is this error worth retrying?
    error_severity: Literal["info", "warning", "error", "fatal"] = "error"  # NEW
```

The orchestrator can then decide whether to treat a tool error as a minor observation or a step-level failure.

---

## DD-H08 — Orchestrator Control Flow Improvements

### Issue

The orchestrator's control flow has several hard-coded behaviors that limit flexibility and observability.

### POI-22 — Configurable classification

> Addresses: T-05

The SIMPLE/COMPLEX classification is currently a hard-coded LLM call with an invisible prompt. Make it configurable:

```python
class ClassificationConfig(BaseModel):
    enabled: bool = True                     # Set False to always treat as complex
    prompt_template: str | None = None       # Custom classification prompt
    default: Literal["simple", "complex"] = "complex"  # Default when disabled
```

**When `enabled=False`**: Skip the classification call entirely, saving one LLM round-trip. This is useful when the caller already knows the task complexity.

---

### POI-23 — Step-level hooks

Allow callers to inject logic at step boundaries without subscribing to the EventBus:

```python
class StepHook(Protocol):
    async def before_step(self, step_index: int, context: ActiveContext, messages: list[Message]) -> None: ...
    async def after_step(self, step_index: int, context: ActiveContext, messages: list[Message]) -> None: ...
```

**Use cases**:
- Custom checkpointing logic
- Dynamic tool injection (add/remove tools based on context)
- External approval gates (pause and wait for human approval before continuing)

---

### POI-24 — Max tool calls per step guard

Currently, there is no limit on how many tool calls the LLM can make in a single step. A misbehaving model could call dozens of tools in one step, consuming excessive resources.

```python
max_tool_calls_per_step: int = 20  # New parameter in run_task()
```

If the LLM returns more tool calls than the limit, execute only the first N and inject a warning message for the next step.

---

## Implementation Priority Matrix

| Priority | POI | Area | Effort | Resolves Tension |
|---|---|---|---|---|
| **P0 — Critical** | | | | |
| | POI-06 | Message compaction | M | T-01 |
| | POI-11 | LLM call retry | M | T-02 |
| | POI-19 | Task completion status | S | T-06 |
| **P1 — High** | | | | |
| | POI-01 | Context snapshot event | S | T-05 |
| | POI-05 | Classification event | S | T-05 |
| | POI-08 | Checkpoint model | M | — |
| | POI-09 | Checkpoint store | M | — |
| | POI-10 | Resume protocol | M | — |
| | POI-12 | Tool validation retry | S | T-02 |
| **P2 — Medium** | | | | |
| | POI-02 | Context pruned event | S | T-01 |
| | POI-03 | Token estimation | S | — |
| | POI-07 | Context budget tracking | M | — |
| | POI-13 | Artifact tools | M | T-03 |
| | POI-14 | Retrieval ledger tool | S | T-03 |
| | POI-15 | Plan tool | S | T-03 |
| | POI-20 | EventBus error reporting | S | T-07 |
| **P3 — Low** | | | | |
| | POI-04 | Notebook updated event | S | — |
| | POI-16 | Notebook search | S | — |
| | POI-17 | Auto-memory mutation | S | T-08 |
| | POI-18 | Fact retraction | S | — |
| | POI-21 | FunctionNode error severity | S | T-04 |
| | POI-22 | Configurable classification | S | T-05 |
| | POI-23 | Step-level hooks | M | — |
| | POI-24 | Max tool calls guard | S | — |

> **Effort scale**: S = hours, M = 1-3 days, L = 1-2 weeks

---

## Breaking Changes Summary

The following POIs introduce breaking changes. All are justified by the `v0.0.0b0` pre-release status:

| POI | Change | Migration |
|---|---|---|
| POI-13 | `ActiveContext.artifacts: list[str]` → `dict[str, str]` | Update `format_context()`, update tests |
| POI-19 | `run_task()` returns `TaskResult` instead of `ActiveContext` | Callers access `result.context` |
| POI-10 | `run_task()` signature gains new keyword arguments | Non-breaking (all optional) |

---

## Verification Strategy

### Unit tests

- Each new event type: serialization round-trip, EventBus dispatch
- Compaction engine: token threshold triggering, message preservation, summary injection
- Checkpoint: serialize → deserialize round-trip, resume from checkpoint
- Retry: mock transient errors, verify backoff timing, verify max attempts
- New tools: artifact CRUD, retrieval logging, fact retraction, notebook search
- TaskResult: verify correct status for each termination path

### Integration tests (markers: `@pytest.mark.integration`)

- End-to-end agent task with compaction triggered (requires real LLM)
- Checkpoint save/resume with FileCheckpointStore
- Retry on simulated rate limit (mock adapter that fails N times then succeeds)

### Regression

- All existing tests must pass unchanged (except where breaking changes require test updates)
- The `agent_reasoning.py` example must continue to work with the new return type
