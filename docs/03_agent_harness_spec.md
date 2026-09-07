# Agent Harness — Development Spec

> Derived from [02_agent_context](./02_agent_context) (the Agent Operating System prompt)
> Builds on [01_execution_nodes_spec.md](./01_execution_nodes_spec.md) (execution nodes, shipped)
> Deferred items tracked in [§8](#8-deferred-to-later-versions) and [future.md](./future.md)
> Status: **Consolidated — all decisions resolved; ready to build from Phase 0 (see [§9](#9-development-readiness))**

---

## 1. Overview

The execution-node layer (`LLMTextNode`, `LLMStreamNode`, `LLMStructuredNode`, `LLMEmbeddingNode` + the adapter ABCs) gives us **single, stateless LLM calls with canonical results and a canonical error taxonomy**. It does not give us an *agent*.

This spec defines the **Agent Harness**: a mini-harness that turns those nodes into a reusable, embeddable agent loop that can be plugged into arbitrary services.

The guiding constraint is **embeddability**. This is not an application. It is a library that a FastAPI worker, a Celery task, a Lambda, an MCP server, or a CLI can import and drive. Everything that touches the outside world — LLM providers, databases, blob storage, queues, vector stores, clocks — enters through an adapter port. The harness core has **zero infrastructure dependencies**.

| Capability | Component | Purpose |
|-----------|-----------|---------|
| **Agent loop** | `AgentLoop` | Deterministic think → act → observe cycle; the only place that decides "what happens next". |
| **Session state** | `SessionState` + `SessionStore` | Full serializable snapshot of a run. Enables suspend/resume, distribution, crash recovery, and time-travel debugging. |
| **Memory** | `MemoryStore` + `MemoryPolicy` | Typed memory records (observation / fact / decision / state / note) with retention, promotion, and retrieval policies. |
| **Context assembly** | `ContextAssembler` | Deterministic, budgeted rendering of session + memory into the prompt. The single point of truth for "what the model sees". |
| **Tools** | `ToolRegistry` + `ToolProvider` | Unified namespace over cognitive (internal) tools, builtin tools, MCP servers, and user code. |
| **Task management** | `TodoList` + `Plan` | Model-managed checklist for hard tasks; skipped for trivial ones. |
| **Workflows** | `Workflow` / `Pipeline` primitives | Deterministic fan-out (`ForEach`, `Map`, `Reduce`) so bulk work never becomes 300 TODO items. |
| **Subagents** | `SubagentTool` | Spawn isolated child loops for parallel exploration; only the summary returns to the parent context. |
| **Artifacts** | `ArtifactStore` | Durable, addressable outputs passed between steps by reference, not by value. |
| **Interruption** | `InterruptRequest` / `Resumption` | Suspend for human or peer-agent input, then resume from the exact same state. |
| **Observability** | `EventBus` + `AgentEvent` | Structured event stream for streaming UIs, cost tracking, auditing, and replay. |

Cross-cutting principles, inherited and extended from the node layer:

- **Dependency inversion everywhere** — the core depends on `Protocol`/ABC ports (`SessionStore`, `MemoryStore`, `ArtifactStore`, `ToolProvider`, `Clock`, `IdGenerator`). Postgres, S3, Redis, and MCP are *adapters* living outside the core.
- **Pydantic-first I/O** — every state, event, tool argument, and result is a `BaseModel`. Everything the harness owns is serializable.
- **Stateless core, externalized state** — `AgentLoop` holds no run state between steps. The state lives in `SessionState`, which is persisted after every step.
- **Async-only** — consistent with DD-12.
- **Canonical errors** — the harness extends the existing `LLMError` taxonomy (`family`, `code`, `retryable`) rather than inventing a second one.
- **The context window is the scarcest resource** — the harness is fundamentally a *context management engine*. Every design decision below is subordinate to that.

---

## 2. SWOT

### Strengths

- **A real foundation already exists.** Canonical results, a `family`/`code`/`retryable` error taxonomy, discriminated-union stream events, `ToolDefinition` with Pydantic schemas, and tool-argument auto-validation (SG-05) are exactly the primitives an agent loop needs. Most harnesses bolt these on late and badly.
- **Provider neutrality is already proven.** Two adapters (OpenAI, Gemini) implement the same four ABCs. The agent loop inherits multi-provider support for free.
- **Prior art in-repo.** `build/lib/rh_cognitv/` contains an earlier `EventBus`, `ActiveContext`, `AgentOrchestrator`, and `FunctionNode` iteration. We know concretely what worked (event taxonomy, typed context records, tool-name prefixing) and what didn't (in-memory-only context, no persistence, orchestrator doing too much).
- **A written cognitive model.** [02_agent_context](./02_agent_context) already defines the Observation → Fact → Decision → State lifecycle and a context budget policy. Most projects never write this down; we can encode it as types.
- **Clean scope boundary.** The harness is a library, not a product. It does not need a UI, a scheduler, or a control plane to be valuable.

### Weaknesses

- **No persistence layer exists at all.** Everything to date is in-process. Sessions, memory, and artifacts are entirely greenfield, and getting the storage seams wrong is expensive to undo.
- **No `FunctionNode` in the live tree.** Tool execution has no canonical node type yet; it must be re-landed in Phase 0 before the harness can be built.
- **No green test baseline.** Provider integration tests currently fail rather than skip when the SDKs are absent (see [§9](#9-development-readiness)).
- **Single-maintainer surface area.** This spec describes a system roughly 5× the size of the node layer. Phasing is not optional.
- **The context assembler is the hardest component** and the least specifiable up front. Truncation, summarization, and relevance are empirical; they need real traces to tune.
- **No evaluation harness.** Without traces and regression tests over real tasks, "improvements" to the loop are unfalsifiable.

### Opportunities

- **The harness layer is where the market is.** Node-level LLM wrappers are commodity; durable, resumable, distributable agent loops are not. This is the differentiating layer.
- **Context-as-a-first-class-concern is a genuine differentiator.** Typed memory with explicit promotion policies (observation → fact → decision) is more principled than the flat scratchpads most harnesses use.
- **MCP is standardizing tools.** Implementing `ToolProvider` once and adding an MCP adapter gives access to a growing third-party tool ecosystem at near-zero marginal cost.
- **Deterministic workflows are underserved.** Most harnesses force everything through the model's attention. Offering `ForEach` over 300 documents as a *deterministic* pipeline with subagent workers is a concrete, defensible advantage.
- **Sessions become a product surface.** A serializable `SessionState` unlocks replay, audit, debugging UIs, fork-and-retry, and cost attribution — features that are otherwise very hard to retrofit.

### Threats

- **Scope explosion.** Memory + workflows + MCP + HITL + distribution is four products. Attempting them concurrently guarantees none ship.
- **Premature abstraction.** Designing `MemoryPolicy` or `ContextAssembler` before real traces exist risks abstractions that fit no actual workload. This is the strongest argument for the maturity gating in §7.
- **Provider drift.** Prompt caching, server-side conversation state, and native agent loops (OpenAI Responses/Assistants, Anthropic tool-use loops) are moving targets. Over-fitting the loop to one provider's shape is a real risk.
- **State migration.** Once persisted `SessionState` exists in someone's Postgres, schema changes become migrations. Versioning must be designed in from day one, not added later.
- **Cost and latency opacity.** Agent loops multiply token spend. Without per-step accounting wired in from Phase 1, cost regressions become invisible.

---

## 3. Architecture

```
rh_cognitv/
├── nodes/                              # EXISTING (spec 01) — unchanged
│   ├── base.py                         # BaseNode
│   ├── function_node.py                # (re-land) FunctionNode
│   ├── llm/                            # LLMText/Stream/Structured/Embedding nodes
│   └── llm_adapters/                   # OpenAI, Gemini
│
├── harness/                            # NEW — pure core, zero infra deps
│   ├── __init__.py
│   ├── loop.py                         # AgentLoop, AgentConfig, StepOutcome
│   ├── state.py                        # SessionState, Step, RunStatus, Plan, TodoList
│   ├── events.py                       # AgentEvent discriminated union
│   ├── errors.py                       # HarnessError taxonomy (extends LLMError)
│   ├── policy.py                       # StopPolicy, BudgetPolicy, RetryPolicy
│   ├── interrupt.py                    # InterruptRequest, Resumption
│   │
│   ├── memory/
│   │   ├── models.py                   # MemoryRecord, MemoryKind, MemoryQuery, MemoryScope
│   │   ├── policy.py                   # MemoryPolicy (retention, promotion, recall)
│   │   └── service.py                  # MemoryService (policy over the MemoryStore port)
│   │
│   ├── context/
│   │   ├── assembler.py                # ContextAssembler
│   │   ├── budget.py                   # TokenBudget, Tokenizer port
│   │   └── sections.py                 # ContextSection renderers
│   │
│   ├── tools/
│   │   ├── registry.py                 # ToolRegistry (namespacing, resolution)
│   │   ├── spec.py                     # HarnessTool, ToolResult, ToolScope
│   │   ├── cognitive.py                # todo.*, memory.*, notes.*, artifact.*, subagent.*
│   │   └── builtin.py                  # small, dependency-free builtins
│   │
│   ├── workflow/
│   │   ├── models.py                   # WorkflowSpec, StepSpec, IOSchema
│   │   ├── engine.py                   # WorkflowEngine (ForEach / Map / Reduce / Sequence)
│   │   └── subagent.py                 # SubagentRunner
│   │
│   └── ports/                          # ALL dependency-inversion boundaries
│       ├── session_store.py            # SessionStore
│       ├── memory_store.py             # MemoryStore
│       ├── artifact_store.py           # ArtifactStore
│       ├── blob_store.py               # BlobStore
│       ├── tool_provider.py            # ToolProvider
│       ├── vector_index.py             # VectorIndex        (declared, DF-06)
│       ├── event_sink.py               # EventSink          (SG-06)
│       ├── redactor.py                 # Redactor           (SG-14)
│       ├── lock.py                     # LockProvider
│       ├── tokenizer.py                # Tokenizer
│       └── clock.py                    # Clock, IdGenerator
│
└── harness_adapters/                   # NEW — concrete infra, optional deps
    ├── memory_backend.py               # in-process dict/list (default, always available)
    ├── filesystem.py                   # local-disk session/artifact/blob store
    ├── sqlalchemy_backend.py           # Postgres / SQLite  [harness-sql]
    ├── redis_backend.py                # session store + locks  [harness-redis]
    ├── s3_backend.py                   # blob/artifact store    [harness-s3]
    └── mcp_provider.py                 # MCP ToolProvider       [harness-mcp]
```

**Dependency rule (enforced, one direction only):**

```
harness_adapters ──► harness/ports ◄── harness (core) ──► nodes
```

`harness/` imports from `harness/ports/` and `nodes/`. It **never** imports from `harness_adapters/`. Adapters are constructed by the caller and injected — the same constructor-injection pattern as DD-03.

### Control flow of one step

```
AgentLoop.step(session)
  1. BudgetPolicy.check(session)               -> may raise BudgetExceeded
  2. MemoryService.recall(session)             -> relevant MemoryRecords
  3. ContextAssembler.assemble(session, recall)-> list[Message] within TokenBudget
  4. ToolRegistry.resolve(session)             -> list[ToolDefinition]
  5. reasoning call, per resolved ReasoningMode -> ReasoningResult (events -> EventBus)
        stream     -> LLMStreamNode
        structured -> LLMStructuredNode
  6. dispatch tool calls
        cognitive -> mutate session in-process
        external  -> ToolProvider.invoke (concurrent when independent)
        interrupt -> return StepOutcome(status=awaiting_input)
        workflow  -> WorkflowEngine.run (may spawn subagents)
  7. record observations; MemoryService.apply_policies(session)
  8. SessionStore.save(session)                <- durability checkpoint
  9. StopPolicy.evaluate(session)              -> continue | done | awaiting_input | failed
```

Step 8 is the crux of the whole design: **the session is durable after every step**, so any step boundary is a valid resume point, on any machine.

---

## 4. Design Decisions (Consolidated)

Each decision below states the issue, why it matters, and the decision taken. Alternatives that were considered and rejected are not reproduced; where a decision defers part of its scope, the deferred part is listed in [§8 Deferred to Later Versions](#8-deferred-to-later-versions). Where an answer follows directly from the node-layer spec (async-only, Pydantic-first, constructor injection) it is inherited rather than re-litigated.

---

### DD-13: Ownership of Agent State — Externalized Snapshot

**Issue.** The prior `AgentOrchestrator` held `ActiveContext` in memory and returned it at the end of `run_task()`. That is incompatible with a stateless, distributable harness. Where does run state live, and who owns its lifecycle?

**Why it matters.** This is the single most load-bearing decision in the spec. It determines whether the harness can survive a process restart, whether a run can be handed between workers, whether human-in-the-loop is possible at all (an interrupt may last hours), whether runs can be replayed or forked, and whether horizontal scaling is achievable. Every other component's signature depends on the answer. Retrofitting externalized state onto an object-oriented loop is effectively a rewrite.

**Decision.** A `SessionState` Pydantic model is the **sole source of truth**. `AgentLoop` is a stateless service — `async def step(state) -> StepOutcome` — and state is persisted via `SessionStore` after every step. Distribution, HITL, replay, and crash recovery all fall out of this.

The cost is serialization discipline: every piece of run state must be a serializable model, so no open handles, live generators, or provider client objects may be held across a step boundary.

`SessionState` carries `schema_version: int` from day one (DD-25), and the `EventBus` stream is persistable separately (SG-06) so we keep the audit trail without paying for an event-fold layer.

The convenience API (`await agent.run(task)`) is a thin driver that loops `step()` against an in-memory store — trivial use stays one line, and the durable path is the same code.

---

### DD-14: Memory Model — Typed Records with a Kind Discriminator

**Issue.** [02_agent_context](./02_agent_context) defines four information forms — Observation, Fact, Decision, State — plus a lifecycle for promoting between them. Are those enforced types in the data model, or merely prompt-level guidance?

**Why it matters.** Retention, recall, and context-budget policy all need to discriminate between kinds of information. "Keep the last 3 observations but never drop a decision" is expressible only if kind is structured data. If memory is a free-form scratchpad, every policy degrades to "truncate the oldest", which is precisely the failure mode that makes long agent runs lose their own conclusions.

**Decision.** A single `MemoryRecord` with a `kind` discriminator (`observation | fact | decision | state | note | artifact_ref`), plus scope, salience, provenance, TTL, and timestamps — mirroring the discriminated-union pattern already used for stream events.

**Kind is advisory to the model but authoritative to the policy engine.** `memory.write` defaults to `kind="note"`, and promotion (note → fact → decision) happens either through an explicit cognitive tool or a periodic consolidation step. This avoids forcing the model to classify at write time — the failure mode that produces empty or mis-classified records — while still yielding structured memory the policy engine can reason over.

`derived_from` provenance is recorded from the start: nearly free to capture, impossible to reconstruct later, and the only thing that makes "why does the agent believe this?" answerable.

---

### DD-15: Context Assembly — Rendered State Header + Bounded Raw Window

**Issue.** Two ways to build the prompt each step: append to a growing `list[Message]` (the chat-transcript model), or re-render the prompt deterministically from `SessionState` + recalled memory (the state-projection model). The prior `AgentOrchestrator` did both.

**Why it matters.** This governs long-run behaviour, cost, and debuggability. A growing transcript inevitably overflows the window and forces destructive truncation. Pure state projection is bounded and reproducible but loses everything the model never wrote to memory.

**Decision.** A hybrid with an explicit boundary: a **rendered state header** (task, plan, TODO, decisions, facts, recalled memory, retrieval ledger) followed by a **bounded window of recent raw turns**. When the window overflows, evicted turns are compacted into memory records — the eviction path always routes through `MemoryService`, never a silent `del`.

This directly encodes the context-budget priority from [02_agent_context](./02_agent_context): **State > Decisions > Facts > Observations**. Observations are evicted first, facts compacted second, state and decisions preserved longest.

`ContextAssembler` is a port with a default implementation, so a caller can supply a domain-specific assembler without forking the loop. It emits a `ContextAssembled` event carrying the per-section token breakdown (SG-09), which is the only realistic way to debug context problems, and orders sections most-stable-first for prompt caching (SG-10).

---

### DD-16: Tool Namespace — Dot-Namespaced Registry with Origin Prefix and Scopes

**Issue.** Four distinct tool sources must coexist: cognitive tools (`todo.create`, `memory.write` — internal to the agent's own working process), builtins, user-registered functions, and dynamically discovered MCP tools. We must define the naming and identity scheme.

**Why it matters.** MCP servers are discovered at runtime and their names are outside our control; collisions are a matter of when, not if. Beyond collisions, tool *identity* drives permissions (which tools may a subagent use?), audit (what actually ran?), cost attribution, and approval gating (which tools need a human to confirm?). A flat namespace forecloses all of these. Additionally, exposing every available tool on every call inflates the prompt and measurably degrades tool-selection accuracy once the count grows past roughly two dozen.

**Decision.** A namespaced `ToolRegistry` storing `HarnessTool` records carrying `origin`, `namespace`, `scope` (`cognitive | builtin | user | mcp | subagent`), `provider_id`, `requires_approval`, and a `ToolDefinition`. A `ToolFilter` selects the subset exposed for a given step or subagent — the mechanism that keeps prompts small as the tool ecosystem grows.

**Naming scheme — dot notation with a reserved origin prefix:**

```
<origin>.<namespace>.<name>          origin ∈ {internal, external}

internal.memory.write                cognitive + builtin tools
internal.todo.update
external.github.create_issue         MCP / user-registered tools
external.docs.classify
```

- `internal.` is **reserved**. `ToolRegistry.register()` raises `ToolNamespaceError` if a caller attempts to register under it, so a package user can never shadow a cognitive tool.
- The fully-qualified dotted name is the tool's identity everywhere: registry keys, `ToolResult.tool_name`, events, audit records, and `ToolFilter` patterns (`external.github.*`).

**Provider name mangling.** Some providers reject `.` in function names (OpenAI constrains names to `^[A-Za-z0-9_-]{1,64}$`). Translation is an **adapter concern, not a core concern**: the adapter encodes `.` → `__` on the way out and decodes `__` → `.` on every returned tool call, so the harness only ever sees dotted names. To keep the mapping bijective, the registry rejects `__` inside a raw segment (`ToolNamespaceError`). Adapters that accept dots pass names through unchanged.

This requires a change to the shipped `OpenAIAdapter` plus round-trip tests (encode → provider → decode) — see [§9 Development Readiness](#9-development-readiness).

**Providers vs. cognitive tools.** `ToolProvider` is the port (`list_tools()`, `invoke(name, args)`), with `McpToolProvider`, `FunctionToolProvider`, and `SubagentToolProvider` as adapters. Cognitive tools are *not* a provider — they mutate `SessionState` in-process and are dispatched by the loop directly, since they must not cross a serialization boundary.

---

### DD-17: Bulk Work — Two-Tier TODOs and Deterministic Workflows

**Issue.** The 300-document classification scenario. Should the agent enumerate work as TODO items and iterate under model control, or emit a declarative workflow spec that a deterministic engine executes?

**Why it matters.** This is the difference between a harness that works on demos and one that works on real workloads. Model-driven iteration over 300 items is quadratic in context (each item's result pollutes the window), non-resumable mid-way, impossible to parallelize safely, and priced accordingly. It also fails unpredictably: the model loses track around item 40 and starts hallucinating progress.

**Decision.** Two tiers. The model uses `todo.*` for the handful of **high-level reasoning steps**, and calls `workflow.run` with a declarative `WorkflowSpec` for anything **homogeneous and repetitive**. The engine executes deterministically — spawning subagents as per-item workers, writing each result to the `ArtifactStore`, and returning a **summary plus artifact references** to the parent context, never the 300 payloads.

The v1 engine supports `Sequence`, `ForEach` (with `max_concurrency`), `Map`, and `Reduce`. Each `StepSpec` declares `input_schema` and `output_schema` as Pydantic models, so per-item workers use `LLMStructuredNode` and produce validated, typed output rather than prose that must be re-parsed.

Conditionals, dynamically terminating loops, and arbitrary DAGs are **out of scope for v1** (§8) — they turn the spec into a programming language. `WorkflowSpec` is versioned so the vocabulary can grow additively.

---

### DD-18: Human-in-the-Loop — Suspend/Resume Interrupt as the Primitive

**Issue.** The agent must be able to ask a human or a peer agent for clarification. Mechanism?

**Why it matters.** A blocking callback (`await ask_human(question)`) is trivial in a CLI and impossible in a request/response service — you cannot hold an HTTP handler or a worker slot open for a two-hour approval, and it cannot survive a deploy. The same machinery is required for tool-approval gating and for agent-to-agent delegation, so getting it right pays off three times.

**Decision.** Suspend/resume is the **sole primitive**. A tool call (`human.ask`, `agent.ask`, or an approval-gated tool) returns `StepOutcome(status="awaiting_input", interrupt=InterruptRequest(...))`. The loop persists and returns. The caller later invokes `resume(session_id, Resumption(...))`, which injects the response as an observation and continues. The caller owns the transport (webhook, queue, UI) — correctly, since that is a service concern, not a library concern.

A thin `InteractiveResponder` adapter turns the interrupt into a blocking call for CLI ergonomics.

Because interruption is unified with approval gating, `requires_approval` on a `HarnessTool` (DD-16) produces exactly the same suspend/resume flow — destructive tools can require explicit confirmation with no extra machinery.

`InterruptRequest` carries `kind` (`clarification | approval | delegation`), a `response_schema` so replies are validated on resume, and an optional `timeout_at` for the caller's scheduler to act on.

---

### DD-19: Storage Ports — Four Segregated Ports

**Issue.** Sessions, memory records, artifacts, and raw blobs all need persistence. One general `Store` port, or separate segregated ports?

**Why it matters.** The access patterns are genuinely different, and conflating them forces every adapter to be bad at something. Sessions are read-modify-write of a single mid-sized document, needing optimistic concurrency. Memory is append-heavy with query-by-kind/scope and eventually semantic search. Artifacts are write-once, read-many, addressed by ID, and potentially large. Blobs are opaque bytes wanting streaming and presigned URLs. A single interface satisfying all four is either lowest-common-denominator or a god-interface that every adapter partially stubs.

**Decision.** Four segregated ports — `SessionStore`, `MemoryStore`, `ArtifactStore`, `BlobStore` — each minimal and independently implementable, each with an in-memory default so the harness runs out of the box with zero configuration. Realistic deployments mix backends (Postgres for sessions and memory, S3 for blobs), which segregation makes natural.

Two specifics are fixed now because they are painful to add later:

- `SessionStore.save()` takes an expected `version` and raises `SessionConflictError` on mismatch — optimistic concurrency is mandatory once two workers can touch one session.
- `ArtifactStore` stores *metadata plus a `BlobRef`*, delegating actual bytes to `BlobStore`. Artifacts can then live in Postgres while their payloads live in S3, without either port knowing about the other.

`VectorIndex` is declared as a port in v1 with no required implementation (§8).

---

### DD-20: Subagents — Isolated Session with Briefing and Structured Return

**Issue.** When the agent spawns a subagent for parallel exploration, does the child share the parent's context and memory, or get an isolated session returning only a summary?

**Why it matters.** This is the entire point of subagents. If a child shares and pollutes the parent's context, spawning five explorers multiplies context consumption by five and the parent drowns in raw findings — strictly worse than doing the work inline. The value of a subagent is precisely **context isolation**. But full isolation means the child lacks task context and may explore uselessly, so the briefing mechanism matters as much as the isolation.

**Decision.** An isolated session with explicit briefing and structured return. The child gets a fresh `SessionState` seeded with a brief (task, the facts/decisions the parent selects or that `MemoryService` recalls, and a restricted tool subset per DD-16). It runs its own loop with its own budget and returns a `SubagentResult` — summary, optional schema-validated structured output, and artifact references — which enters the parent's context as a **single observation**.

Children may write to a **shared `ArtifactStore`** (so bulk output never transits the parent's context) but get their **own memory scope** by default, with promotion of selected findings to the parent scope on return. That combination is what makes the 300-document workflow tractable: 300 children write 300 artifacts, and the parent sees one summary plus a manifest reference.

Recursion depth is capped by `AgentConfig.max_subagent_depth` (default 2), and child budgets are drawn from the parent's remaining budget, so a runaway tree cannot spend unboundedly.

---

### DD-21: Loop Termination — Composable Stop Policies

**Issue.** How does the loop know it is done? The prior orchestrator stopped when all TODO items were `done` or `max_steps` was hit.

**Why it matters.** Termination is where agent loops fail expensively. Relying on the model to self-declare completion produces both premature stops and non-termination (looping on a failing tool, burning budget). Relying only on `max_steps` produces truncated work with no signal about *why* it stopped. Callers need to distinguish "finished", "hit the wall", "stuck", and "needs input" in order to retry, escalate, or surface to a user.

**Decision.** An ordered, composable `StopPolicy` chain; each policy returns `continue | stop(reason)`. Ships with `MaxStepsPolicy`, `BudgetPolicy` (tokens / cost / wall-clock), `NoProgressPolicy` (N consecutive steps with no state mutation), `FinishToolPolicy`, and `TodoCompletePolicy`. Callers append their own domain policies (e.g. "stop when the classification artifact exists").

`StepOutcome.status` is a discriminated `completed | max_steps | budget_exceeded | no_progress | awaiting_input | failed | cancelled`, and `stop_reason` names the policy that fired.

`NoProgressPolicy` is in the default chain specifically because unproductive looping is the most common and most expensive real-world failure — and it is only detectable because DD-13 gives us a diffable state snapshot per step.

---

### DD-22: Long-Running Execution — `step()` as the Primitive, Drivers on Top

**Issue.** A run may be sub-second or may span hours across interrupts and bulk workflows. What is the execution contract?

**Why it matters.** These are different operational shapes: short runs want low latency and simple ergonomics; long runs need to survive deploys, avoid holding worker slots, report progress, and be cancellable. If the harness only supports the blocking shape, long runs are impossible; if it only supports the continuation shape, simple use becomes needlessly painful. Because DD-13 already externalizes state, supporting both is cheap — but the contract (who owns the scheduler?) must be explicit, or callers will build incompatible drivers.

**Decision.** `AgentLoop.step()` is the sole primitive. Drivers sit on top:

- **`LocalDriver`** — loops `step()` in-process. Short runs, CLI, tests. **Ships with the harness.**
- **`QueueDriver`** — enqueues a continuation message after each step. Long runs, distributed workers. **An adapter, not a core component** (§8), because queue semantics are a deployment concern the library should not opinionate on.

The published contract is: *load state → step → save state → re-enqueue unless terminal*, with a reference implementation in the docs.

A `LockProvider` port prevents two workers stepping the same session concurrently — no-op in-memory default, Redis/Postgres advisory-lock adapters later. Cancellation is cooperative: a `cancel_requested` flag on `SessionState` is checked at each step boundary, which is the only safe cancellation point given tools may be mid-flight.

---

### DD-23: Reasoning Node Selection — An Explicit `ReasoningMode`

**Issue.** Each step needs one LLM call that may emit both prose and tool calls. `LLMStreamNode` supports tools and reconstructs streamed tool calls; `LLMStructuredNode` returns tool calls non-streaming. Which drives the loop?

**Why it matters.** Streaming is required for any interactive surface, and `LLMStreamNode` already handles the hard part — reconstructing fragmented tool-call arguments across chunks. But streaming buys nothing when nobody is watching: it adds reconstruction complexity, weakens some providers' strict structured-output guarantees, and costs latency-to-first-tool-call. Both shapes are genuinely needed, so the selection must be an explicit, inspectable setting rather than a hard-coded choice.

**Decision.** Introduce `ReasoningMode` on `AgentConfig`:

```python
class ReasoningMode(str, Enum):
    AUTO = "auto"              # default — resolved per step (see below)
    STREAM = "stream"          # LLMStreamNode  — UI / interactive surfaces
    STRUCTURED = "structured"  # LLMStructuredNode — headless / workflow
```

Resolution rules, in order:

| Context | Resolved mode | Rationale |
|---|---|---|
| `AgentConfig.reasoning_mode` set explicitly | that mode | Caller knows their surface. |
| `AUTO` **and** the `EventBus` has ≥1 subscriber for text-delta events | `STREAM` | Someone is plugged into a UI; deltas are worth producing. |
| `AUTO` and no delta subscriber | `STRUCTURED` | Headless run — take the stronger schema guarantees and the simpler path. |
| Inside `WorkflowEngine` (per-item worker) | `STRUCTURED` — **always, non-overridable** | Output must validate against `output_schema`; nobody is watching a stream. |
| Subagent | inherits the parent's resolved mode unless the brief overrides it | Children are usually headless even when the parent streams. |

Both paths funnel into a single internal `ReasoningResult` (`text`, `tool_calls`, `usage`), so the dispatch logic downstream of the call is mode-agnostic and only one code path handles tool dispatch. Both share tool-argument validation (SG-05), so `ToolValidationError` remains the single retryable failure mode for malformed tool arguments in either mode.

A caller wiring the harness into a UI therefore does exactly one thing — subscribe to the event stream — and streaming turns itself on.

---

### DD-24: Cognitive Tools — Real Tool Calls with Conditional Exposure

**Issue.** Should the agent's self-management operations (write memory, update TODO, extract facts, record decisions) be tool calls, or structured markers parsed out of its prose?

**Why it matters.** Parsed markers are cheaper but fragile — format drift breaks silently and produces an agent that appears to work while quietly losing all its state updates. Tool calls are validated, observable, and auditable, but consume schema tokens on every call. There is also a real risk of the model over-using cognitive tools and spending its budget on bookkeeping instead of the task.

**Decision.** Cognitive operations are real tool calls, with a **minimal always-on core** and the rest **conditionally exposed**:

- **Always on:** `internal.memory.write`, `internal.todo.update`, `internal.finish`.
- **Conditional:** `internal.todo.create`, `internal.workflow.run`, `internal.subagent.spawn`, `internal.human.ask`, `internal.artifact.*` — exposed only once a plan exists or the task is classified as complex.

This preserves validation and auditability while keeping the prompt small for trivial tasks — the "don't create TODOs for trivial questions" requirement expressed as a mechanism rather than a plea in the prompt. Conditional exposure is implemented via `ToolFilter` (DD-16), so no special-casing leaks into the loop.

The complexity classification the prior orchestrator performed as a separate LLM call is folded into the first step: the model is given `todo.create` and simply doesn't call it for trivial tasks. That removes a full round-trip from every run.

---

### DD-25: State Schema Evolution — Versioned Now, Migrations Later

**Issue.** Once `SessionState` and `MemoryRecord` are persisted in a caller's database, changing their shape breaks stored data. Do we plan for this now or react later?

**Why it matters.** This is cheap now and very expensive later. The moment a user has in-flight sessions in Postgres, any field rename becomes a migration problem *in their infrastructure*, not ours. Long-running sessions (DD-22) mean a session may be written by version N and read by N+1 after a deploy — so cross-version reads are the normal case for any run spanning a release. Without a version field there is no way to even detect the mismatch; the failure is a confusing Pydantic validation error at resume time.

**Decision.** `schema_version: int` on `SessionState`, `MemoryRecord`, `WorkflowSpec`, and `ArtifactMeta` from Phase 1, with `SchemaVersionError` raised on mismatch — converting silent corruption into an actionable message. A documented hook is left where the migration registry will attach.

The **migration registry itself is deferred** (§8): the version field costs nothing and is impossible to add retroactively to already-persisted data; the migration machinery costs a lot and is easy to add once there is a schema worth migrating.

---

## 5. Adopted Suggestions

Non-blocking recommendations, all **accepted** and scheduled. Each is additive and can be adopted independently. Suggestions that were accepted *in principle but deferred* appear in [§8](#8-deferred-to-later-versions) instead.

| ID | Suggestion | Adopted in |
|---|---|---|
| SG-06 | Persist the event stream as the audit trail | Phase 2 (port) / Phase 7 (backend) |
| SG-07 | Per-step cost and token accounting | Phase 1 (fields) / Phase 2 (enforcement) |
| SG-08 | Tool-result truncation with artifact spillover | Phase 3 |
| SG-09 | `ContextAssembled` debug event | Phase 3 |
| SG-10 | Prompt-caching-aware section ordering | Phase 3 |
| SG-11 | Idempotency keys on tool invocations | Phase 4 |
| SG-13 | Structured concurrency for parallel tool calls | Phase 4 |
| SG-14 | Redaction hooks before persistence and emission | Phase 1 (seam) / Phase 7 (impl) |

SG-12 (trace-based evaluation harness) was accepted but deferred — see [DF-10](#8-deferred-to-later-versions).

### SG-06: Persist the Event Stream as the Audit Trail

`SessionState` snapshots give current state; the `AgentEvent` stream gives *how we got here*. Persisting events (via an `EventSink` port with a no-op default) gives audit, cost attribution, replay for debugging, and training data for evaluation — without paying for full event sourcing (DD-13). Adopted from Phase 2, since events are already being emitted and the incremental cost is one port.

### SG-07: Per-Step Cost and Token Accounting as a First-Class Field

`LLMResultMeta.tokens_used` already exists. Aggregate it onto `Step` and `SessionState` (`total_tokens`, `estimated_cost`, `wall_clock_ms`), and let `BudgetPolicy` enforce ceilings. Agent loops multiply spend, and cost regressions are invisible unless measured from day one. A `PricingTable` (model → cost per 1M tokens) supplied by the caller keeps pricing data out of the library, where it would immediately go stale.

### SG-08: Tool Result Truncation with Artifact Spillover

Tool outputs are the single largest uncontrolled context consumer — one `read_file` on a large document can blow the window. `ToolResult` should carry a size limit: outputs exceeding it are written to the `ArtifactStore` and replaced in-context with a preview plus an artifact reference the agent can query. This is the same mechanism as DD-20's subagent summarization, applied one level down, and it is the highest-leverage single context optimization available.

### SG-09: A `ContextAssembled` Debug Event with Section Token Breakdown

Emit an event each step carrying the per-section token counts and what was evicted. Context problems are otherwise nearly undebuggable — you see a bad answer with no visibility into what the model actually saw. Cheap to add, disproportionately valuable.

### SG-10: Prompt Caching Awareness in the Assembler

Anthropic and OpenAI both offer substantial discounts for stable prompt prefixes. `ContextAssembler` should order sections **most-stable-first** (persona, tool definitions, task, plan) and **most-volatile-last** (recent turns, recalled memory), and expose an optional `cache_breakpoint` marker adapters can translate to provider-specific cache controls. This is a pure ordering constraint with meaningful cost impact, and it is much easier to honour from the start than to retrofit.

### SG-11: Idempotency Keys on Tool Invocations

In a distributed setup (DD-22), a worker can crash after invoking a tool but before saving state, causing re-execution on resume. For side-effecting tools this is a correctness bug, not a performance one. Give each tool invocation a deterministic `idempotency_key` (session + step + call index) and let `ToolProvider` implementations honour it. Also enables safe at-least-once queue delivery.

### SG-13: Structured Concurrency for Parallel Tool Calls

When a step returns multiple independent tool calls, execute them under `asyncio.TaskGroup` with a concurrency cap, preserving result ordering. Substantial latency win for exploration-heavy steps at low implementation cost. Requires tools to declare `is_side_effecting` so ordering can be preserved where it actually matters.

### SG-14: Redaction Hooks Before Persistence and Emission

Sessions and events will contain user data and possibly credentials returned by tools. A `Redactor` port applied before `SessionStore.save()` and before event emission makes compliance an integration point rather than a fork. Cheap to add as a seam now; invasive to add once persistence is spread across the codebase.

---

## 6. Core Models

```python
# ── harness/state.py ─────────────────────────────────────────────────────────

class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_INPUT = "awaiting_input"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TodoItem(BaseModel):
    id: str
    description: str
    status: Literal["pending", "in_progress", "done", "blocked"] = "pending"
    note: str | None = None

class TodoList(BaseModel):
    goal: str | None = None
    items: list[TodoItem] = Field(default_factory=list)

class Plan(BaseModel):
    summary: str
    revision: int = 0
    updated_at: float

class StepUsage(BaseModel):                       # SG-07
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    estimated_cost: float | None = None
    duration_ms: float = 0.0

class Step(BaseModel):
    index: int
    started_at: float
    finished_at: float | None = None
    text: str = ""
    tool_calls: list[ToolCallResult] = Field(default_factory=list)
    tool_results: list["ToolResult"] = Field(default_factory=list)
    usage: StepUsage = Field(default_factory=StepUsage)
    error: str | None = None

class SessionState(BaseModel):
    schema_version: int = 1                       # DD-25
    session_id: str
    parent_session_id: str | None = None          # DD-20 subagent lineage
    depth: int = 0
    version: int = 0                              # DD-19 optimistic concurrency
    status: RunStatus = RunStatus.PENDING
    task: str
    persona: "AgentPersona | None" = None
    plan: Plan | None = None
    todo: TodoList = Field(default_factory=TodoList)
    steps: list[Step] = Field(default_factory=list)
    window: list[Message] = Field(default_factory=list)   # DD-15 bounded raw turns
    interrupt: "InterruptRequest | None" = None
    stop_reason: str | None = None
    cancel_requested: bool = False
    total_usage: StepUsage = Field(default_factory=StepUsage)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float
    updated_at: float


# ── harness/memory/models.py ─────────────────────────────────────────────────

class MemoryKind(str, Enum):                      # DD-14
    OBSERVATION = "observation"
    FACT = "fact"
    DECISION = "decision"
    STATE = "state"
    NOTE = "note"
    ARTIFACT_REF = "artifact_ref"

class MemoryScope(BaseModel):
    session_id: str | None = None                 # None => cross-session (long-term)
    namespace: str = "default"

class MemoryRecord(BaseModel):
    schema_version: int = 1
    id: str
    kind: MemoryKind = MemoryKind.NOTE
    scope: MemoryScope
    content: str
    summary: str | None = None
    salience: float = 0.5                         # 0..1, drives eviction order
    derived_from: list[str] = Field(default_factory=list)   # provenance
    tags: list[str] = Field(default_factory=list)
    embedding_ref: str | None = None              # set when a VectorIndex is wired
    expires_at: float | None = None
    created_at: float
    updated_at: float

class MemoryQuery(BaseModel):
    scope: MemoryScope | None = None
    kinds: list[MemoryKind] | None = None
    tags: list[str] | None = None
    text: str | None = None                       # semantic when VectorIndex present
    limit: int = 20

class MemoryPolicy(BaseModel):
    max_active_observations: int = 5              # from 02_agent_context
    observation_ttl_s: float | None = 3600.0
    promote_observations: bool = True             # observation -> fact on consolidation
    protected_kinds: list[MemoryKind] = Field(
        default_factory=lambda: [MemoryKind.DECISION, MemoryKind.STATE]
    )
    recall_limit: int = 10


# ── harness/tools/spec.py ────────────────────────────────────────────────────

class ToolScope(str, Enum):                       # DD-16
    COGNITIVE = "cognitive"
    BUILTIN = "builtin"
    USER = "user"
    MCP = "mcp"
    SUBAGENT = "subagent"

class ToolOrigin(str, Enum):                      # DD-16 — "internal." is reserved
    INTERNAL = "internal"
    EXTERNAL = "external"

class HarnessTool(BaseModel):
    origin: ToolOrigin
    namespace: str                                # no "." and no "__" inside a segment
    definition: ToolDefinition                    # reuses spec 01 model
    scope: ToolScope
    provider_id: str | None = None
    requires_approval: bool = False               # DD-18 approval gating
    is_side_effecting: bool = True                # SG-13 ordering
    max_result_chars: int = 8_000                 # SG-08 spillover threshold

    @property
    def qualified_name(self) -> str:
        return f"{self.origin.value}.{self.namespace}.{self.definition.name}"

# Adapter-level, not core: providers that reject "." in function names.
def encode_tool_name(qualified: str) -> str:      # "external.github.create_issue"
    return qualified.replace(".", "__")           # -> "external__github__create_issue"

def decode_tool_name(wire: str) -> str:
    return wire.replace("__", ".")

class ToolResult(BaseModel):
    tool_name: str                                # always the dotted qualified name
    call_id: str | None = None
    ok: bool = True
    output: str = ""
    artifact_ref: str | None = None               # SG-08 when truncated
    truncated: bool = False
    error: str | None = None
    duration_ms: float = 0.0
    idempotency_key: str | None = None            # SG-11


# ── harness/workflow/models.py ───────────────────────────────────────────────

class StepSpec(BaseModel):                        # DD-17
    id: str
    kind: Literal["agent", "llm", "function"] = "llm"
    prompt: str | None = None
    tool_filter: list[str] | None = None
    input_schema: str | None = None               # registered Pydantic model name
    output_schema: str | None = None
    max_steps: int = 8

class ForEachSpec(BaseModel):
    id: str
    over: str                                     # artifact ref or memory query id
    step: StepSpec
    max_concurrency: int = 4
    continue_on_error: bool = True

class WorkflowSpec(BaseModel):
    schema_version: int = 1
    id: str
    description: str
    nodes: list[StepSpec | ForEachSpec]
    output_artifact: str | None = None

class WorkflowResult(BaseModel):
    workflow_id: str
    summary: str
    artifact_refs: list[str] = Field(default_factory=list)
    succeeded: int = 0
    failed: int = 0
    errors: list[str] = Field(default_factory=list)


# ── harness/interrupt.py ─────────────────────────────────────────────────────

class InterruptRequest(BaseModel):                # DD-18
    id: str
    kind: Literal["clarification", "approval", "delegation"]
    question: str
    context: str | None = None
    options: list[str] | None = None
    response_schema: str | None = None
    target: str | None = None                     # e.g. "human" | "agent:reviewer"
    tool_call: ToolCallResult | None = None       # set for approval gating
    timeout_at: float | None = None
    created_at: float

class Resumption(BaseModel):
    interrupt_id: str
    response: str | None = None
    structured_response: dict[str, Any] | None = None
    approved: bool | None = None
    responder: str | None = None


# ── harness/loop.py ──────────────────────────────────────────────────────────

class AgentPersona(BaseModel):
    name: str = "agent"
    role: str = "You are an autonomous assistant."
    instructions: str | None = None

class BudgetPolicy(BaseModel):                    # SG-07
    max_tokens: int | None = None
    max_cost: float | None = None
    max_wall_clock_s: float | None = None

class ReasoningMode(str, Enum):                   # DD-23
    AUTO = "auto"
    STREAM = "stream"
    STRUCTURED = "structured"

class ReasoningResult(BaseModel):                 # DD-23 — unified output of either mode
    text: str = ""
    tool_calls: list[ToolCallResult] = Field(default_factory=list)
    usage: "StepUsage" = Field(default_factory=StepUsage)

class AgentConfig(BaseModel):
    llm: LLMConfig                                # reuses spec 01 model
    persona: AgentPersona = Field(default_factory=AgentPersona)
    reasoning_mode: ReasoningMode = ReasoningMode.AUTO   # DD-23
    max_steps: int = 20
    max_subagent_depth: int = 2
    max_parallel_tools: int = 4
    context_token_budget: int = 100_000
    window_size: int = 12                         # DD-15 raw turns retained
    budget: BudgetPolicy = Field(default_factory=BudgetPolicy)
    memory_policy: MemoryPolicy = Field(default_factory=MemoryPolicy)

class StepOutcome(BaseModel):                     # DD-21
    status: Literal[
        "continue", "completed", "max_steps", "budget_exceeded",
        "no_progress", "awaiting_input", "failed", "cancelled",
    ]
    stop_reason: str | None = None
    step: Step | None = None
    interrupt: InterruptRequest | None = None
    error: str | None = None


# ── harness/ports/ ───────────────────────────────────────────────────────────

class SessionStore(abc.ABC):                      # DD-19
    @abc.abstractmethod
    async def load(self, session_id: str) -> SessionState | None: ...

    @abc.abstractmethod
    async def save(self, state: SessionState, *, expected_version: int | None = None) -> SessionState:
        """Persist state. Raise SessionConflictError on version mismatch."""

    @abc.abstractmethod
    async def delete(self, session_id: str) -> None: ...

class MemoryStore(abc.ABC):
    @abc.abstractmethod
    async def write(self, records: list[MemoryRecord]) -> None: ...

    @abc.abstractmethod
    async def query(self, query: MemoryQuery) -> list[MemoryRecord]: ...

    @abc.abstractmethod
    async def delete(self, ids: list[str]) -> None: ...

class ArtifactStore(abc.ABC):
    @abc.abstractmethod
    async def put(self, meta: "ArtifactMeta", data: bytes | AsyncIterator[bytes]) -> str: ...

    @abc.abstractmethod
    async def get_meta(self, ref: str) -> "ArtifactMeta": ...

    @abc.abstractmethod
    async def open(self, ref: str) -> AsyncIterator[bytes]: ...

    @abc.abstractmethod
    async def list(self, *, session_id: str | None = None, tags: list[str] | None = None) -> list["ArtifactMeta"]: ...

class ToolProvider(abc.ABC):                      # DD-16
    provider_id: str

    @abc.abstractmethod
    async def list_tools(self) -> list[HarnessTool]: ...

    @abc.abstractmethod
    async def invoke(self, name: str, arguments: dict[str, Any], *, idempotency_key: str | None = None) -> ToolResult: ...

class Tokenizer(abc.ABC):
    @abc.abstractmethod
    def count(self, text: str, *, model: str) -> int: ...

class LockProvider(abc.ABC):                      # DD-22
    @abc.abstractmethod
    async def acquire(self, key: str, *, ttl_s: float) -> "Lock": ...

class Clock(abc.ABC):
    @abc.abstractmethod
    def now(self) -> float: ...

class IdGenerator(abc.ABC):
    @abc.abstractmethod
    def new_id(self) -> str: ...                  # ULID by default (already a dependency)
```

### Harness error taxonomy

Extends the existing `LLMError` hierarchy so a single `except LLMError` still catches everything, and `retryable` remains meaningful to any retry engine.

| Error | Family | Retryable | Raised when |
|-------|--------|-----------|-------------|
| `HarnessError` | `harness` | `False` | Base class for all harness errors. |
| `SessionConflictError` | `harness` | `True` | `SessionStore.save()` version mismatch (DD-19). |
| `SchemaVersionError` | `harness` | `False` | Persisted `schema_version` unsupported (DD-25). |
| `BudgetExceededError` | `harness` | `False` | Token / cost / wall-clock ceiling hit (SG-07). |
| `MaxStepsExceededError` | `harness` | `False` | Step ceiling hit (DD-21). |
| `NoProgressError` | `harness` | `False` | `NoProgressPolicy` fired (DD-21). |
| `ToolNotFoundError` | `harness` | `False` | Model called an unregistered tool. |
| `ToolNamespaceError` | `harness` | `False` | Registration under the reserved `internal.` prefix, or a segment containing `.` / `__` (DD-16). |
| `ToolExecutionError` | `harness` | varies | Provider-side tool failure; `retryable` set by the provider. |
| `InterruptTimeoutError` | `harness` | `False` | `InterruptRequest.timeout_at` elapsed (DD-18). |
| `WorkflowError` | `harness` | `False` | Workflow validation or fatal execution failure (DD-17). |

`ToolValidationError` (SG-05) is deliberately reused unchanged: malformed tool arguments are already modelled as retryable, and the harness's response — re-prompt with the validation message — is exactly the intended behaviour.

---

## 7. Implementation Phases

Phases 0–5 are **buildable now** with the information on hand. Phases 6–8 are **maturity-gated**: each names the concrete signal required before starting, because designing them without real traces risks abstractions that fit no real workload. Deferred work that belongs to no phase is listed in [§8](#8-deferred-to-later-versions).

### Complexity scale and model assignment

Each deliverable is rated so implementation can be routed to the cheapest model that can do it safely.

| Level | Character of the work | Assign to |
|---|---|---|
| **C4 — Architectural** | Defines a contract everything else depends on; the failure mode is an irreversible design error, not a bug. Ports, state model, loop state machine, budget/eviction ordering, workflow resumability. | **Opus 5** |
| **C3 — High** | Concurrency, state machines, provider quirks, non-obvious algorithms. Correct-looking code can be subtly wrong. | **Opus 5**, or **Sonnet 5** with an Opus 5 design note and review |
| **C2 — Moderate** | Implementation against a contract that is already fully written down. Judgement needed, but the shape is fixed. | **Sonnet 5** |
| **C1 — Mechanical** | Boilerplate, in-memory adapters, tests derived from a written table, examples, docstrings, code motion. | **GPT-Luna** |

**Routing rules**

1. **Contract first.** For any C4 item, Opus 5 writes the ABCs, Pydantic models, docstrings, and the test table *before* any C2/C1 work starts against it. Smaller models fill in; they never define the seam.
2. **Never split a C4 item.** Architectural items go to one model in one pass; a split contract is how inconsistencies enter.
3. **Review gate.** C1/C2 output is accepted only against the phase's stated exit criteria, not by inspection.
4. **Escalate on a second failed attempt.** If a C2 item fails its tests twice, it was mis-rated — re-route up one level.
5. **Tests inherit the rating of what they test**, except pure fixture/boilerplate tests, which are always C1.

Phase-level rating is the maximum of its items.

**A phase is not a pool of tasks models pick from — it is a sequence of stages with a hard handoff.** Within a phase:

- **Stage 1 (contract).** One model, one pass, produces the interfaces/models everything else in the phase imports: ABCs, Pydantic models, signatures, docstrings, the test table. Nothing in Stage 2 starts before Stage 1 is merged and reviewed.
- **Stage 2 (implementation against the contract).** Items here import only what Stage 1 produced, not each other. They are mutually independent by construction — this is the *only* point where different models can genuinely work in parallel, because none of them can break another's inputs.
- **Stage 3 (mechanical).** Tests, fixtures, and adapters that exercise a specific Stage 2 item. Each Stage 3 item depends on the one Stage 2 item it tests, not on the whole phase — start it as soon as that item lands.

So within a phase you get real parallelism only inside Stage 2, and only across items that don't reference each other's output. Every phase table below is ordered by stage and states what each row depends on.

### Phase 0 — Pre-flight (unblocks Phase 1)

**Goal:** Clear the concrete blockers identified in [§9](#9-development-readiness) so Phase 1 starts on a green tree.

| Deliverable | Complexity | Model |
|---|---|---|
| Re-land `nodes/function_node.py` + `FunctionResult` from `build/lib` into the live tree, with tests | C1 | GPT-Luna |
| Remove the stale `build/` tree from the repo and add it to `.gitignore` (it is a second, divergent source of truth today) | C1 | GPT-Luna |
| Remove the empty `rh_cognitv/agents/` package — the harness lives in `rh_cognitv/harness/` | C1 | GPT-Luna |
| Gate provider integration tests on **SDK availability** as well as API key (they currently fail, not skip, when `openai` is absent) | C1 | GPT-Luna |
| Add `ruff` + `mypy` to `requirements_dev.txt` — a `ruff` config exists with no linter installed | C1 | GPT-Luna |
| Decide and document where the [02_agent_context](./02_agent_context) prompt ships (package resource vs. default `AgentPersona` text) | C2 | Sonnet 5 |

**Exit criteria:** `pytest -q` is fully green with no provider SDKs installed; `FunctionNode` is importable from `rh_cognitv.nodes`; `ruff check` and `mypy` run clean on the live tree.

**Phase rating: C2 — Sonnet 5 leads, GPT-Luna executes.**

### Phase 1 — Ports, State, Events, Errors

**Goal:** Establish every dependency-inversion boundary and the serializable state model before any logic depends on them.

**Deliverables:**

| Stage | Deliverable | Complexity | Model |
|---|---|---|---|
| 1 | `harness/ports/*` — `SessionStore` (optimistic concurrency), `MemoryStore`, `ArtifactStore`, `BlobStore`, `ToolProvider`, `Tokenizer`, `LockProvider`, `Clock`, `IdGenerator`, `VectorIndex` (declared, unimplemented) | **C4** | Opus 5 |
| 1 | `harness/state.py` — `SessionState`, `Step`, `TodoList`, `Plan`, `RunStatus`, `StepUsage`, all with `schema_version` (DD-25) and cost fields (SG-07) | **C4** | Opus 5 |
| 2 | `harness/errors.py` — the taxonomy above, extending `LLMError` | C2 | Sonnet 5 |
| 2 | `harness/events.py` — `AgentEvent` discriminated union, ported and extended from the `build/lib` prototype | C2 | Sonnet 5 |
| 2 | `Redactor` seam declared (no-op default) before any persistence lands (SG-14) | C2 | Sonnet 5 |
| 2 | Default `Tokenizer` — heuristic estimator, **no new dependency**; `tiktoken`-backed variant is an optional adapter | C2 | Sonnet 5 |
| 3 | `harness_adapters/memory_backend.py` — in-memory implementations of every port | C1 | GPT-Luna |
| 3 | Serialization round-trip and ABC-conformance tests | C1 | GPT-Luna |

**Dependencies:** both Stage 1 items must land together (they reference each other: `Step`/`SessionState` fields are typed against the port signatures) — treat them as one Opus 5 pass, not two. All four Stage 2 items only need Stage 1's port/state signatures and are independent of each other; they can go to Sonnet 5 in parallel or in one batch. Stage 3 needs every Stage 1 port signature (it implements all of them) and at least one Stage 2 model (`SessionState`/`AgentEvent`) to write round-trip tests against — so it is the last stage to start, not a parallel one.

**Tests:** Round-trip serialization of every persisted model; `schema_version` mismatch raises `SchemaVersionError`; optimistic-concurrency conflict on `SessionStore.save()`; in-memory adapters satisfy their ABCs.

**Exit criteria:** A `SessionState` survives `model_dump_json()` → `model_validate_json()` losslessly. No module under `harness/` imports from `harness_adapters/`.

**Phase rating: C4 — Opus 5 leads.** This phase fixes every seam in the system; it is the single worst place to economize on model capability.

---

### Phase 2 — Minimal Loop (Reason → Tool → Observe)

**Goal:** The smallest thing that is genuinely an agent: multi-step, tool-calling, persisted after every step.

**Deliverables:**

| Stage | Deliverable | Complexity | Model |
|---|---|---|---|
| 1 | `harness/loop.py` — `AgentLoop.step()` state machine, `StepOutcome`, `AgentConfig` (DD-13, DD-22) | **C4** | Opus 5 |
| 1 | `harness/tools/registry.py` + `spec.py` — dotted identity, reserved `internal.` prefix, collision rules, `ToolFilter` (DD-16) | C3 | Opus 5 |
| 2 | `ReasoningMode` resolution + unified `ReasoningResult` over stream/structured paths (DD-23) | C3 | Opus 5 |
| 2 | `harness/policy.py` — `MaxStepsPolicy`, `FinishToolPolicy`, `NoProgressPolicy` (DD-21) | C2 | Sonnet 5 |
| 2 | `harness/tools/cognitive.py` — always-on core: `internal.memory.write`, `internal.todo.update`, `internal.finish` (DD-24) | C2 | Sonnet 5 |
| 2 | `harness/context/assembler.py` — first `ContextAssembler`: state header + bounded raw window (DD-15) | C3 | Sonnet 5 + Opus 5 review |
| 2 | `EventSink` port and per-step usage aggregation (SG-06, SG-07) | C2 | Sonnet 5 |
| 2 | `OpenAIAdapter` tool-name codec (`.` ⇄ `__`) with round-trip tests (DD-16) — a change to shipped code | C2 | Sonnet 5 |
| 3 | `LocalDriver` and the `run()` convenience wrapper | C1 | GPT-Luna |
| 3 | `FunctionToolProvider` wrapping `FunctionNode` instances | C1 | GPT-Luna |
| 3 | Fake adapter scripting multi-step tool-call sequences + tests | C1 | GPT-Luna |

**Dependencies:** the two Stage 1 items are the phase's real contract (the loop's state machine and the tool registry it dispatches against) and must land as one Opus 5 pass before anything else starts. The six Stage 2 items each consume that contract but not each other's output — `ReasoningMode` doesn't need `policy.py`, the codec doesn't need `cognitive.py` — so they run in parallel across Sonnet 5 (and Opus 5 for the two C3 items) once Stage 1 is reviewed. Stage 3 needs a working loop *and* at least the cognitive-tool core and `FunctionToolProvider`'s target (`FunctionNode`, already landed in Phase 0) — start it once the specific Stage 2 item it exercises is done, not after the whole phase.

**Tests:** Fake adapter scripting multi-step tool-call sequences; state persisted and reloaded between steps; each stop policy fires correctly; `NoProgressPolicy` halts a deliberately stuck loop; tool-name collisions resolve deterministically; registering under `internal.` raises `ToolNamespaceError`.

**Exit criteria:** An agent completes a 3-tool task, is killed mid-run, and resumes from the persisted session on a fresh `AgentLoop` instance with identical results.

**Phase rating: C4 — Opus 5 leads.** The step state machine and tool identity are the two things everything after this phase builds on.

---

### Phase 3 — Memory & Context Discipline

**Goal:** Typed memory with enforced policies, and a context assembler that never silently drops information.

**Deliverables:**

| Stage | Deliverable | Complexity | Model |
|---|---|---|---|
| 1 | `harness/memory/models.py` — `MemoryRecord`, `MemoryKind`, `MemoryScope`, `MemoryQuery` (DD-14) | C2 | Sonnet 5 |
| 2 | `harness/context/budget.py` — `TokenBudget` with the State > Decisions > Facts > Observations eviction order, and compaction-on-eviction into `MemoryService` (DD-15) | **C4** | Opus 5 |
| 2 | `harness/memory/policy.py` + `service.py` — retention, promotion (observation → fact), recall | C3 | Opus 5 |
| 3 | Tool-result truncation with artifact spillover (SG-08) | C2 | Sonnet 5 |
| 3 | Cognitive tools: `internal.memory.read`, `internal.memory.promote`, `internal.notes.append` | C1 | GPT-Luna |
| 3 | `ContextAssembled` event with per-section token breakdown (SG-09); stable-prefix section ordering (SG-10) | C1 | GPT-Luna |

**Dependencies:** `MemoryRecord`/`MemoryKind`/`MemoryScope` (Stage 1) is the type every later item imports, so it goes first even though it's only C2 — it just happens to be simple *and* blocking. `budget.py` and `memory/policy.py` (Stage 2) both consume Stage 1's types but not each other's code directly, though they must agree on eviction semantics — treat them as one Opus 5 pass rather than two independent ones, since a mismatch here is exactly the "plausible-but-wrong" failure mode called out below. Stage 3 items each depend on a specific Stage 2 piece (spillover needs `budget.py`'s truncation hook; the debug event needs the assembler from Phase 2 plus `budget.py`) and are independent of each other.

**Tests:** Observation eviction respects `max_active_observations` while decisions survive; budget overflow evicts in priority order; evicted content is recoverable from memory; a large tool result spills to an artifact and leaves a working reference.

**Exit criteria:** A 30-step run stays within `context_token_budget` and can still cite a decision made at step 2.

**Phase rating: C4 — Opus 5 leads.** Eviction ordering is the least mechanically verifiable component in the system; a plausible-but-wrong implementation degrades answers silently.

---

### Phase 4 — Artifacts, Subagents, and Parallelism

**Goal:** Bounded parallel exploration with context isolation.

**Deliverables:**

| Stage | Deliverable | Complexity | Model |
|---|---|---|---|
| 1 | `harness/workflow/subagent.py` — `SubagentRunner`, `SubagentResult`, context isolation, depth capping, budget inheritance (DD-20) | C3 | Opus 5 |
| 2 | Parallel tool execution under `asyncio.TaskGroup` with `max_parallel_tools`, result ordering, and cancellation semantics (SG-13) | C3 | Opus 5 |
| 2 | Idempotency keys on tool invocations (SG-11) | C2 | Sonnet 5 |
| 2 | `harness_adapters/filesystem.py` — local session / artifact / blob store | C1 | GPT-Luna |
| 3 | `subagent.spawn` tool with an explicit brief and a restricted tool subset | C2 | Sonnet 5 |
| 3 | Cognitive tools: `internal.artifact.write`, `internal.artifact.read`, `internal.artifact.list` | C1 | GPT-Luna |

**Dependencies:** `SubagentRunner` (Stage 1) fixes the isolation contract the spawn tool wraps, so it goes first. Parallel tool dispatch, idempotency keys, and the filesystem store (Stage 2) don't depend on `SubagentRunner` or on each other — they can run alongside Stage 1, not strictly after it, so this phase has two independent tracks rather than a single line. Stage 3 needs its specific Stage 1/2 counterpart: `subagent.spawn` needs `SubagentRunner`; the artifact tools need the filesystem store.

**Tests:** Subagent context is isolated (parent window unchanged apart from the single summary observation); depth cap enforced; child budget deducted from parent; parallel tool results preserve call order; repeated invocation with the same idempotency key executes once.

**Exit criteria:** A parent agent spawns 5 concurrent explorers and its own context grows by 5 summaries, not 5 transcripts.

**Phase rating: C3 — Opus 5 leads the concurrency and isolation items; GPT-Luna does the filesystem adapter and artifact tools.**

---

### Phase 5 — Deterministic Workflows

**Goal:** The 300-document scenario, executed deterministically and resumably.

**Deliverables:**

| Stage | Deliverable | Complexity | Model |
|---|---|---|---|
| 1 | `harness/workflow/models.py` — `WorkflowSpec`, `StepSpec`, `ForEachSpec`, `WorkflowResult` (DD-17) | C2 | Sonnet 5 |
| 2 | `harness/workflow/engine.py` — `Sequence`, `ForEach` (with `max_concurrency`), `Map`, `Reduce`, plus per-item checkpointing into `SessionState` so a mid-workflow crash resumes at the next unprocessed item | **C4** | Opus 5 |
| 3 | Per-item workers pinned to `LLMStructuredNode` against `output_schema` (DD-23) | C2 | Sonnet 5 |
| 3 | `internal.workflow.run` cognitive tool, exposed conditionally (DD-24) | C1 | GPT-Luna |
| 3 | Per-item results written to `ArtifactStore`; manifest artifact returned to the parent | C1 | GPT-Luna |

**Dependencies:** `WorkflowSpec` (Stage 1) is data-only and simple, but the engine (Stage 2) is defined in terms of it, so it must land first. The engine's checkpoint contract is the phase's real risk, and every Stage 3 item plugs into a specific engine hook (the worker call, the tool entry point, the artifact-write path) — none of them can be written, let alone tested, before Stage 2 exists, so this phase is the most strictly sequential one in the plan.

**Tests:** `ForEach` over 100 fake items respects `max_concurrency`; per-item failures are isolated under `continue_on_error`; a crash at item 50 resumes at item 50, not item 0; outputs validate against `output_schema`.

**Exit criteria:** 300 documents classified in a single agent run, with the parent context growing by one summary and one manifest reference.

**Phase rating: C4 — Opus 5 leads.** Resumable partial fan-out is the hard part; everything else in the phase is straightforward once the engine's checkpoint contract exists.

---

### Phase 6 — Human / Agent in the Loop *(gated)*

**Gate:** A concrete integration exists that owns a transport (webhook, queue, or UI). Interrupt ergonomics are unknowable without a real consumer, and building the resume API against an imagined one guarantees rework.

**Deliverables:**

| Stage | Deliverable | Complexity | Model |
|---|---|---|---|
| 1 | `harness/interrupt.py` — `InterruptRequest`, `Resumption` (DD-18) | C2 | Sonnet 5 |
| 2 | `AgentLoop.resume(session_id, resumption)` and the suspend path through `StepOutcome` | C3 | Opus 5 |
| 3 | Approval gating via `HarnessTool.requires_approval`, reusing the same suspend/resume path | C2 | Sonnet 5 |
| 3 | `internal.human.ask` and `internal.agent.ask` tools | C1 | GPT-Luna |
| 3 | `InteractiveResponder` adapter for synchronous CLI use | C1 | GPT-Luna |
| 3 | Timeout handling driven by the caller's scheduler (documented contract, no scheduler shipped) | C1 | GPT-Luna |

**Dependencies:** the interrupt/resumption models (Stage 1) are simple but must exist before `resume()` (Stage 2) can be typed. Everything in Stage 3 wraps `resume()` for a specific caller (approval, human, CLI, scheduler) and can be split across models in parallel once Stage 2 lands.

**Exit criteria:** An agent suspends on a clarification, the process exits entirely, and a new process resumes it from persisted state with the answer injected.

**Phase rating: C3 — Opus 5 for the resume path, Sonnet 5/GPT-Luna for the rest.**

---

### Phase 7 — Production Persistence Adapters *(gated)*

**Gate:** At least one real deployment with a chosen infrastructure stack. Writing a Postgres schema before knowing the query patterns — and the actual size distribution of sessions and memory — produces a schema that will be rewritten.

**Deliverables:**

| Stage | Deliverable | Complexity | Model |
|---|---|---|---|
| 1 | `QueueDriver` reference implementation and the documented continuation contract (DD-22) | C3 | Opus 5 |
| 2 | `harness_adapters/sqlalchemy_backend.py` — session + memory + artifact-metadata stores `[harness-sql]` | C3 | Opus 5 |
| 2 | `harness_adapters/redis_backend.py` — session store and `LockProvider` `[harness-redis]` | C2 | Sonnet 5 |
| 2 | `harness_adapters/s3_backend.py` — `BlobStore` `[harness-s3]` | C1 | GPT-Luna |
| 2 | `Redactor` implementation applied before persistence and emission (SG-14) | C2 | Sonnet 5 |
| 2 | Persisted event trail behind the existing `EventSink` port (SG-06) | C1 | GPT-Luna |
| 3 | Migration registry activated if a schema change has landed (DD-25) | C3 | Opus 5 |

**Dependencies:** the continuation contract (Stage 1) is what every backend adapter must honour for correct at-least-once resume semantics, so it goes first. The five Stage 2 adapters/seams each implement one existing port against one real backend and don't depend on each other — genuinely parallel across all three models. The migration registry (Stage 3) only activates once a real schema change exists, which by definition can't happen before the Stage 2 backends are in use.

**Exit criteria:** A single session is stepped by two different worker processes without state loss or lock contention.

**Phase rating: C3 — Opus 5 for the concurrency-sensitive stores and the driver contract.**

---

### Phase 8 — MCP, Semantic Recall, and Evaluation *(gated)*

**Gate:** Distinct signals per item — MCP requires a target server worth integrating; semantic recall requires memory volumes where keyword and kind filtering demonstrably fail; evaluation requires a corpus of real traces.

**Deliverables:**

| Stage | Deliverable | Complexity | Model |
|---|---|---|---|
| 1 | `VectorIndex` implementation and embedding-backed `MemoryQuery.text` recall, using the existing `LLMEmbeddingNode` | C3 | Opus 5 |
| 1 | Trace recording adapter and replay-based regression suite (SG-12) | C3 | Opus 5 |
| 1 | `harness_adapters/mcp_provider.py` — MCP `ToolProvider` with dynamic discovery `[harness-mcp]` | C2 | Sonnet 5 |
| 2 | `PricingTable` (caller-supplied) and cost reporting (SG-07) | C1 | GPT-Luna |

**Dependencies:** the three Stage 1 items are unrelated ports/adapters (`VectorIndex`, trace recorder, MCP provider) with no dependency on each other — this phase's three items are effectively three independent, gated mini-projects that happen to share a maturity phase, not a pipeline. `PricingTable` (Stage 2) only needs the `StepUsage`/`BudgetPolicy` fields already in place since Phase 1, so it can actually start earlier than this phase implies if a caller needs it sooner.

**Exit criteria:** MCP tools are callable without any core change; semantic recall measurably beats keyword recall on a real corpus; a prompt change can be evaluated against recorded traces before merge.

**Phase rating: C3 — Opus 5 for recall and evaluation; MCP is a straightforward port implementation.**

---

## 8. Deferred to Later Versions

Everything below was **considered and accepted in principle, but deliberately not built in v1**. Each item names the seam that exists today so the work stays purely additive, and the concrete trigger that should start it.

The organising principle: **build every seam now, defer every implementation that needs empirical data.** Ports, versioning, and accounting fields are cheap now and prohibitively expensive to retrofit. Policies, schemas, and heuristics are the opposite — cheap later, and near-certainly wrong if guessed at now.

| ID | Deferred item | Source | Seam that exists now | Trigger to build |
|---|---|---|---|---|
| DF-01 | Event-sourced state (fold/projection) and `jsonpatch` delta encoding of sessions | DD-13 | Per-step snapshots + persisted event stream; `jsonpatch` is already a dependency | Session snapshots become large enough that per-step full writes dominate storage or latency |
| DF-02 | Migration registry (`@migration(from=N, to=N+1)`) | DD-25 | `schema_version` on every persisted model + `SchemaVersionError` | The first breaking schema change with sessions already in someone's database |
| DF-03 | `QueueDriver` (Celery / SQS / Temporal) | DD-22 | `step()` primitive + documented continuation contract + `LockProvider` port | A deployment that cannot hold a worker slot for a whole run |
| DF-04 | Production persistence adapters (Postgres, Redis, S3) | DD-19, Phase 7 | Four segregated store ports with in-memory + filesystem defaults | One real deployment with a chosen infrastructure stack |
| DF-05 | Human/agent-in-the-loop transports | DD-18, Phase 6 | `awaiting_input` status and persisted-state model already support suspension | An integration that owns a transport (webhook, queue, UI) |
| DF-06 | Semantic recall (`VectorIndex` implementation, embedding-backed `MemoryQuery.text`) | DD-19, Phase 8 | `VectorIndex` port declared; `MemoryRecord.embedding_ref` pre-placed; `LLMEmbeddingNode` already ships | Memory volumes where kind/tag/keyword filtering demonstrably fails |
| DF-07 | MCP tool provider | DD-16, Phase 8 | `ToolProvider` port + dotted namespacing designed for runtime-discovered names | A target MCP server worth integrating |
| DF-08 | Workflow conditionals, dynamically terminating loops, arbitrary DAGs | DD-17 | `WorkflowSpec.schema_version` so the vocabulary grows additively | A real workload that `Sequence`/`ForEach`/`Map`/`Reduce` cannot express |
| DF-09 | Learned or adaptive context compaction | DD-15 | `ContextAssembler` is a swappable port; eviction already routes through `MemoryService` | Recorded traces to evaluate a compaction strategy against (needs DF-10) |
| DF-10 | Trace-based evaluation harness | SG-12 | `EventBus` + `EventSink` produce the raw material from Phase 2 | A corpus of real runs; design the recording adapter early so traces accumulate first |
| DF-11 | `PricingTable` and cost dashboards | SG-07 | `StepUsage.estimated_cost` and `BudgetPolicy` are structural from Phase 1 | A caller who needs cost attribution; pricing data goes stale and belongs to them |
| DF-12 | Multi-modal inputs | [future.md](./future.md) #7 | `ArtifactStore` / `BlobStore` are the intended carriers | Requires the `Message.content` change noted in future.md |
| DF-13 | Anthropic adapter | [future.md](./future.md) #9 | Adapter ABCs; the harness is provider-agnostic by construction | Independent of this spec |

---

## 9. Development Readiness

Assessed against the live tree, not the spec.

### What is ready

| Signal | State |
|---|---|
| Node layer (spec 01) | Complete — `LLMTextNode`, `LLMStreamNode`, `LLMStructuredNode`, `LLMEmbeddingNode` in `rh_cognitv/nodes/llm/` |
| Provider adapters | Two shipped (`OpenAIAdapter`, `GeminiAdapter`) against the same ABCs — provider neutrality is proven, not theoretical |
| Canonical contracts | `Message`, `LLMConfig`, `ToolDefinition`, `ToolCallResult`, `TokenUsage`, `LLMResultMeta` all exist and are re-exported |
| Error taxonomy | `LLMError` with `family` / `code` / `retryable`; the harness taxonomy extends it rather than competing with it |
| Test suite | 227 unit tests passing, fake-adapter pattern already established — the harness can be tested the same way |
| Dependencies | `pydantic`, `jsonpatch`, `ulid-py`, `jsonschema` already declared; **Phases 0–5 need no new runtime dependency** |
| Prior art | `build/lib/rh_cognitv/` holds a working `EventBus` (165 LOC), `ActiveContext` (214), `AgentOrchestrator` (465), `FunctionNode` (67) to port from |
| Written cognitive model | [02_agent_context](./02_agent_context) (677 lines) defines the memory lifecycle and context budget the harness encodes as types |

### Blockers — all cleared by Phase 0

| # | Blocker | Impact |
|---|---|---|
| 1 | `FunctionNode` exists only in `build/`, not in the live tree | `FunctionToolProvider` (Phase 2) has no substrate |
| 2 | `build/` is committed and diverges from `rh_cognitv/` | Two sources of truth; a model asked to "port from build/lib" may resurrect stale code |
| 3 | `rh_cognitv/agents/` exists but is empty (only `__pycache__`) | Ambiguous target package; the spec places the harness in `rh_cognitv/harness/` |
| 4 | 8 integration tests **fail** (not skip) — they gate on `OPENAI_API_KEY` but not on the `openai` SDK being installed | No green baseline, so "did my change break anything?" is unanswerable |
| 5 | No linter or type-checker installed despite a `[tool.ruff]` config | The review gate in §7 has nothing to enforce against |
| 6 | No `Tokenizer` default and no tokenizer dependency | `TokenBudget` (Phase 3) needs one; resolved by shipping a heuristic default and making `tiktoken` an optional adapter |

### Open items that are decisions, not blockers

- **Packaging extras.** `[harness-sql]`, `[harness-redis]`, `[harness-s3]`, `[harness-mcp]` are named in §3 but not yet in `pyproject.toml`. Only needed from Phase 7; add them when the adapters land.
- **`py.typed` coverage.** `[tool.setuptools.package-data]` lists each subpackage explicitly, so every new harness subpackage must be added there or it ships untyped.
- **CI.** No workflow exists. Without one, the §7 review gate is manual.

### Verdict

**Ready to start Phase 0 immediately, and Phase 1 as soon as Phase 0's exit criteria are met.**

Phases 0–5 are fully specified, require no new runtime dependency, and depend on nothing external. The only true prerequisite is a green test baseline. Phases 6–8 remain gated on the signals stated in §7 and should not be started early.

**Recommended first three actions:**

1. Run Phase 0 (GPT-Luna, with the `AgentPersona` packaging decision to Sonnet 5).
2. Opus 5 writes Phase 1's ports and `state.py` in a single pass — no splitting, no delegation of the contract.
3. Only then parallelize: Sonnet 5 on errors/events, GPT-Luna on in-memory adapters and round-trip tests.

---

## 10. Relationship to future.md

Items from [future.md](./future.md) that this spec supersedes or advances:

| future.md item | Status |
|---|---|
| #1 EventBus & Observability | **Advanced** — `EventBus` lands in Phase 1; persisted `EventSink` in Phase 7 (SG-06). |
| #2 Runtime / Execution Engine | **Partially superseded** — the harness provides step-level orchestration, budgets, and stop policies. A generic retry runtime remains separate and complementary; `retryable` on `LLMError` is the shared contract. |
| #3 FunctionNodes | **Advanced** — re-landed in Phase 0 as the substrate for `FunctionToolProvider`. |
| #4 FlowNodes | **Partially superseded** — `ForEach` / `Map` / `Reduce` / `Sequence` land in Phase 5 as `WorkflowSpec`. Full DAG support remains deferred (DF-08). |
| #5 Agent Loops | **Superseded** — this spec. |
| #6 Memory / Context Management | **Superseded** — DD-14, DD-15, Phase 3. |
| #7 Multi-Modal Inputs | **Still deferred** — DF-12. |
| #9 Anthropic Adapter | **Still deferred** — DF-13; orthogonal to this spec, since the harness is provider-agnostic by construction. |

New deferrals introduced by this spec are catalogued in [§8](#8-deferred-to-later-versions) as DF-01 … DF-11.
