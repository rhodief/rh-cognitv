# Agent Harness — Development Spec

> Derived from [02_agent_context](./02_agent_context) (the Agent Operating System prompt)
> Builds on [01_execution_nodes_spec.md](./01_execution_nodes_spec.md) (execution nodes, shipped)
> Deferred items tracked in [future.md](./future.md)
> Status: **Draft — pending decisions marked in §4**

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
- **No `FunctionNode` in the live tree.** Tool execution has no canonical node type yet; the harness cannot be built without first re-landing it.
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
│       ├── vector_index.py             # VectorIndex
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
  5. LLMStreamNode / LLMStructuredNode         -> text + tool_calls (events -> EventBus)
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

## 4. Design Decisions

Each decision below states the issue, why it is non-obvious, the options considered, a recommendation, and a slot for your response. Trivial decisions are omitted; where an answer follows directly from the existing spec (async-only, Pydantic-first, constructor injection) it is inherited rather than re-litigated.

---

### DD-13: Ownership of Agent State — Loop Object vs. Externalized Snapshot

**Issue.** The prior `AgentOrchestrator` held `ActiveContext` in memory and returned it at the end of `run_task()`. That is incompatible with the stated requirement for a stateless, distributable harness. We must decide where run state lives and who owns its lifecycle.

**Why it matters.** This is the single most load-bearing decision in the spec. It determines whether the harness can survive a process restart, whether a run can be handed between workers, whether human-in-the-loop is possible at all (an interrupt may last hours), whether runs can be replayed or forked, and whether horizontal scaling is achievable. Every other component's signature depends on the answer. Retrofitting externalized state onto an object-oriented loop is effectively a rewrite.

**Options.**

1. *Loop owns state (prior art).* `AgentLoop` holds mutable state; `run()` blocks to completion. Simplest and fastest to write, best ergonomics for short synchronous runs. Cannot suspend, cannot distribute, cannot recover from crash. Rejected — it fails the primary requirement.
2. *Externalized snapshot.* A `SessionState` Pydantic model is the sole source of truth. `AgentLoop` is a stateless service: `async def step(state) -> StepOutcome`, with state persisted via `SessionStore` after each step. Distribution, HITL, replay, and recovery all fall out naturally. Costs serialization discipline — every piece of state must be a serializable model, which forbids holding open handles or live generators across step boundaries.
3. *Event-sourced log.* Persist an append-only event log; rebuild state by folding events. Gives perfect audit and time-travel. Adds a fold/projection layer, snapshotting for performance, and event-schema migration — substantial complexity for benefits we can approximate with per-step snapshots plus the event stream.

**Recommendation: Option 2**, with a deliberate nod to Option 3. `SessionState` carries `schema_version: int` from day one, and the `EventBus` stream is persistable separately so we retain the audit trail without paying for a fold layer. Option 3 remains available later as an optimization for large sessions (snapshot + delta), and `jsonpatch` is already a dependency, which makes delta-encoding cheap when we want it.

The high-level convenience API (`await agent.run(task)`) is preserved as a thin driver that loops `step()` against an in-memory store — so trivial use is still one line, while the durable path is the same code.

**Your comment:**
>
>

---

### DD-14: Memory Model — Typed Records vs. Free-Form Scratchpad

**Issue.** [02_agent_context](./02_agent_context) defines four information forms — Observation, Fact, Decision, State — plus a lifecycle for promoting between them. We must decide whether those are enforced types in the data model or merely prompt-level guidance.

**Why it matters.** Retention, recall, and context-budget policy all need to discriminate between kinds of information. "Keep the last 3 observations but never drop a decision" is expressible only if kind is structured data. If memory is a free-form scratchpad, every policy degrades to "truncate the oldest", which is precisely the failure mode that makes long agent runs lose their own conclusions. Conversely, over-typing risks the model refusing to fit its output into our taxonomy, producing empty or mis-classified records.

**Options.**

1. *Free-form notes.* A single `notes: list[str]`. Trivial to implement; the model writes whatever it wants. No policy leverage, no structured recall, no dedup.
2. *Typed `MemoryRecord` with a `kind` discriminator.* One table/collection, records tagged `observation | fact | decision | state | note | artifact_ref`, each with scope, salience, provenance (`derived_from: list[str]`), TTL, and timestamps. Policies operate per-kind. Retrieval can filter by kind. Requires the model to classify when writing — mitigated by defaulting to `note` and letting a promotion step reclassify.
3. *Separate stores per kind.* Maximum type safety, but multiplies the port surface (four stores instead of one) and makes cross-kind queries and promotion awkward.

**Recommendation: Option 2.** A single `MemoryRecord` with a `kind` discriminator, mirroring the discriminated-union pattern already used for stream events. Crucially, **kind is advisory to the model but authoritative to the policy engine**: `memory.write` defaults `kind="note"`, and promotion (note → fact → decision) happens either via an explicit cognitive tool or a periodic consolidation step. This avoids forcing classification at write time while still yielding structured memory.

`derived_from` provenance is included from the start; it is nearly free to record and impossible to reconstruct later, and it is what makes "why does the agent believe this?" answerable.

**Your comment:**
>
>

---

### DD-15: Context Assembly — Implicit Message History vs. Deterministic Re-Rendering

**Issue.** Two ways to build the prompt each step: append to a growing `list[Message]` (the chat-transcript model), or discard the transcript and deterministically re-render the prompt from `SessionState` + recalled memory each step (the state-projection model). The prior `AgentOrchestrator` did both, growing a message history *and* re-rendering the system prompt.

**Why it matters.** This governs long-run behaviour, cost, and debuggability. A growing transcript inevitably overflows the context window and forces destructive truncation; it also makes prompt caching effective (stable prefix) but makes the prompt a function of history rather than of state. Deterministic re-rendering keeps the prompt bounded and reproducible — the same state always produces the same prompt, which makes bugs reproducible and enables replay — but it breaks naive prefix caching and risks losing conversational nuance that was never distilled into memory. The hybrid is what most mature harnesses converge on, but the split point is the actual design question.

**Options.**

1. *Pure transcript.* Standard chat loop. Best model fidelity, best caching, unbounded growth, non-reproducible prompts.
2. *Pure state projection.* Prompt is a pure function of `SessionState` + recall. Bounded and reproducible; loses everything not explicitly written to memory, which puts enormous pressure on the model's discipline in calling `memory.write`.
3. *Hybrid with an explicit boundary.* A rendered state header (task, plan, TODO, decisions, facts, recalled memory, retrieval ledger) followed by a **bounded window** of recent raw turns. When the window overflows, the evicted turns are compacted into memory records rather than deleted.

**Recommendation: Option 3**, with the boundary owned by `ContextAssembler` and the eviction path always routing through `MemoryService` — never a silent `del`. This directly encodes the context-budget priority from [02_agent_context](./02_agent_context): State > Decisions > Facts > Observations. Observations are evicted first, facts compacted second, state and decisions preserved longest.

`ContextAssembler` is a port with a default implementation, so a caller can supply a domain-specific assembler without forking the loop. It emits a `ContextAssembled` event carrying the per-section token breakdown, which is the only realistic way to debug context problems.

**Your comment:**
>
>

---

### DD-16: Tool Namespace — Flat vs. Namespaced Registry with Scopes

**Issue.** Four distinct tool sources must coexist: cognitive tools (`todo.create`, `memory.write` — internal to the agent's own working process), builtins, user-registered functions, and dynamically discovered MCP tools. The prior orchestrator prefixed user tools with `external.` to avoid collisions. We must define the general scheme.

**Why it matters.** MCP servers are discovered at runtime and their names are outside our control; collisions are a matter of when, not if. Beyond collisions, tool *identity* drives permissions (which tools may a subagent use?), audit (what actually ran?), cost attribution, and approval gating (which tools need a human to confirm?). A flat namespace forecloses all of these. Additionally, exposing every available tool on every call inflates the prompt and measurably degrades tool-selection accuracy once the count grows past roughly two dozen.

**Options.**

1. *Flat names, last-write-wins.* Simplest. Silent shadowing, no provenance, no scoping. Rejected.
2. *Prefixed names, flat storage.* Prior art (`external.foo`). Solves collisions only, and prefixing corrupts the name the model sees, which subtly degrades selection quality.
3. *Namespaced registry with scopes and per-run filtering.* `ToolRegistry` stores `HarnessTool` records carrying `namespace`, `name`, `scope` (`cognitive | builtin | user | mcp`), `provider`, `requires_approval`, and a `ToolDefinition`. The fully-qualified `namespace:name` is the internal identity; the exposed name is qualified only on actual collision. A `ToolFilter` selects the subset exposed for a given step or subagent.

**Recommendation: Option 3.** The registry is the natural place for approval gating (DD-18) and subagent capability restriction, and per-run filtering is the mechanism that keeps prompts small as the tool ecosystem grows.

Concretely: `ToolProvider` is the port (`list_tools()`, `invoke(name, args)`), with `McpToolProvider`, `FunctionToolProvider`, and `SubagentToolProvider` as adapters. Cognitive tools are *not* a provider — they mutate `SessionState` in-process and are dispatched by the loop directly, since they must not cross a serialization boundary.

**Your comment:**
>
>

---

### DD-17: Bulk Work — Model-Driven TODOs vs. Deterministic Workflows

**Issue.** Your 300-document classification scenario. Should the agent enumerate work as TODO items and iterate under model control, or emit a declarative workflow spec that a deterministic engine executes?

**Why it matters.** This is the difference between a harness that works on demos and one that works on real workloads. Model-driven iteration over 300 items is quadratic in context (each item's result pollutes the window), non-resumable mid-way, impossible to parallelize safely, and priced accordingly. It also fails unpredictably: the model loses track around item 40 and starts hallucinating progress. Deterministic execution gives bounded per-item context, natural parallelism with a concurrency cap, per-item retry, and exact resumability — at the cost of requiring the model to correctly *specify* the workflow up front.

**Options.**

1. *TODO-only.* One TODO per item. Rejected for the reasons above.
2. *Workflow-only.* All iteration must go through the engine. Rigid; genuinely exploratory tasks don't have a knowable shape in advance.
3. *Two-tier: TODOs for reasoning, workflows for bulk.* The model uses `todo.*` for the handful of high-level reasoning steps, and calls `workflow.run` with a declarative `WorkflowSpec` for anything homogeneous and repetitive. The engine executes deterministically, spawning subagents as per-item workers, writing each result to the `ArtifactStore`, and returning a **summary plus artifact references** — never the 300 payloads — to the parent context.

**Recommendation: Option 3.** The v1 engine supports `Sequence`, `ForEach` (with `max_concurrency`), `Map`, and `Reduce` — enough for the document-classification case and most realistic bulk work. Each `StepSpec` declares `input_schema` and `output_schema` as Pydantic models, so per-item workers use `LLMStructuredNode` and produce validated, typed output rather than prose that must be re-parsed.

Deliberately excluded from v1: conditionals, loops with dynamic termination, and arbitrary DAGs. Those turn the spec into a programming language, and we should see real workloads before designing that surface. `WorkflowSpec` is versioned so the vocabulary can grow additively.

**Your comment:**
>
>

---

### DD-18: Human-in-the-Loop — Blocking Callback vs. Suspend/Resume Interrupt

**Issue.** The agent must be able to ask a human or a peer agent for clarification. Mechanism?

**Why it matters.** A blocking callback (`await ask_human(question)`) is trivial in a CLI and impossible in a request/response service — you cannot hold an HTTP handler or a worker slot open for a two-hour approval. It also cannot survive a deploy. The interrupt model is strictly more general (a CLI can trivially implement a synchronous responder on top of it) but forces the harness to represent "waiting" as a first-class, persisted state. The same machinery is required for tool-approval gating and for agent-to-agent delegation, so getting it right pays off three times.

**Options.**

1. *Blocking callback.* `HumanProvider.ask()` awaited inline. Trivial, incompatible with distribution. Rejected.
2. *Suspend/resume interrupt.* A tool call (`human.ask`, `agent.ask`, or an approval-gated tool) returns `StepOutcome(status="awaiting_input", interrupt=InterruptRequest(...))`. The loop persists and returns. The caller later invokes `resume(session_id, Resumption(...))`, which injects the response as an observation and continues. Requires the caller to own the transport (webhook, queue, UI) — which is correct, since that is a service concern, not a library concern.
3. *Both, with the callback as sugar.* Interrupt is the primitive; an optional `InteractiveResponder` adapter turns it into a blocking call for CLI use.

**Recommendation: Option 3** — Option 2 as the sole primitive, with a thin `InteractiveResponder` for ergonomics. Because interruption is unified with tool-approval gating, `requires_approval` on a `HarnessTool` (DD-16) produces exactly the same suspend/resume flow, which is a meaningful safety feature: destructive tools can require explicit confirmation with no extra machinery.

`InterruptRequest` carries `kind` (`clarification | approval | delegation`), a `response_schema` (a Pydantic model) so replies are validated on resume, and an optional `timeout_at` for the caller's scheduler to act on.

**Your comment:**
>
>

---

### DD-19: Storage Ports — One Store vs. Four Segregated Ports

**Issue.** Sessions, memory records, artifacts, and raw blobs all need persistence. One general `Store` port, or separate segregated ports?

**Why it matters.** The access patterns are genuinely different, and conflating them forces every adapter to be bad at something. Sessions are read-modify-write of a single mid-sized document, needing optimistic concurrency. Memory is append-heavy with query-by-kind/scope and eventually semantic search. Artifacts are write-once, read-many, addressed by ID, and potentially large. Blobs are opaque bytes wanting streaming and presigned URLs. A single interface satisfying all four is either lowest-common-denominator (losing concurrency control and query capability) or a god-interface that every adapter partially stubs — violating interface segregation and making a Redis or S3 adapter awkward or impossible.

**Options.**

1. *One `Store` port.* Uniform, minimal surface. Forces the worst trade-off on every adapter. Rejected.
2. *Four segregated ports* — `SessionStore`, `MemoryStore`, `ArtifactStore`, `BlobStore` — each minimal and each independently implementable. Realistic deployments mix backends (Postgres for sessions and memory, S3 for blobs) which segregation makes natural. Costs four ABCs and four in-memory defaults.
3. *Two ports* (`StateStore` + `BlobStore`). A middle ground, but it merges session concurrency semantics with memory query semantics, which is exactly the conflation worth avoiding.

**Recommendation: Option 2.** Four ports, each with an in-memory default so the harness runs out of the box with zero configuration.

Two specifics worth fixing now because they are painful to add later: `SessionStore.save()` takes an expected `version` and raises `SessionConflictError` on mismatch (optimistic concurrency — mandatory once two workers can touch one session); and `ArtifactStore` stores *metadata plus a `BlobRef`*, delegating actual bytes to `BlobStore`. That separation lets artifacts live in Postgres while their payloads live in S3, without either port knowing about the other.

`VectorIndex` is defined as a port in v1 but has no required implementation — semantic recall is gated on maturity (§7).

**Your comment:**
>
>

---

### DD-20: Subagents — Shared Context vs. Isolated Session with Summary Return

**Issue.** When the agent spawns a subagent for parallel exploration, does the child share the parent's context and memory, or get an isolated session returning only a summary?

**Why it matters.** This is the entire point of subagents. If a child shares and pollutes the parent's context, spawning five explorers multiplies context consumption by five and the parent drowns in raw findings — strictly worse than doing the work inline. The value of a subagent is precisely **context isolation**: the child burns its own window on exploration and returns a distilled result. Getting this wrong makes the feature actively harmful. But full isolation means the child lacks task context and may explore uselessly, so the briefing mechanism matters as much as the isolation.

**Options.**

1. *Shared context.* Child appends to the parent's session. Defeats the purpose. Rejected.
2. *Full isolation.* Child gets only its prompt string. Cheap and safe, but under-briefed children waste tokens rediscovering context the parent already had.
3. *Isolated session with explicit briefing and structured return.* Child gets a fresh `SessionState` seeded with an explicit brief (task, relevant facts/decisions the parent selects or that `MemoryService` recalls, and a restricted tool subset per DD-16). It runs its own loop with its own budget. It returns a `SubagentResult` — a summary, optional schema-validated structured output, and artifact references — which enters the parent's context as a **single observation**.

**Recommendation: Option 3.** Children may write to a **shared `ArtifactStore`** (so bulk output never transits the parent's context) but get their **own memory scope** by default, with promotion of selected findings to the parent scope on return. That combination is what makes the 300-document workflow tractable: 300 children write 300 artifacts, and the parent sees one summary plus a manifest reference.

Recursion depth is capped by `AgentConfig.max_subagent_depth` (default 2), and child budgets are drawn from the parent's remaining budget, so a runaway tree cannot spend unboundedly.

**Your comment:**
>
>

---

### DD-21: Loop Termination — Model Self-Declaration vs. Composable Stop Policies

**Issue.** How does the loop know it is done? The prior orchestrator stopped when all TODO items were `done` or `max_steps` was hit.

**Why it matters.** Termination is where agent loops fail expensively and embarrassingly. Relying on the model to self-declare completion produces both premature stops (declaring victory on partial work) and non-termination (looping on a tool that keeps failing, burning budget). Relying only on `max_steps` produces truncated work with no signal about *why* it stopped. Callers need to distinguish "finished", "hit the wall", "stuck", and "needs input" in order to react correctly — retry, escalate, or surface to a user. Hard-coding one rule also prevents callers from imposing domain constraints such as "stop when the classification artifact exists".

**Options.**

1. *Model self-declaration only* via a `finish` tool. Natural and clear intent, but no protection against non-termination.
2. *Fixed built-in rules* (all TODOs done, or `max_steps`). Predictable but inflexible and gives poor stop-reason granularity.
3. *Composable `StopPolicy` chain.* An ordered list of policies, each returning `continue | stop(reason)`. Ships with `MaxStepsPolicy`, `BudgetPolicy` (tokens/cost/wall-clock), `NoProgressPolicy` (N consecutive steps with no state mutation — the standard defence against loops), `FinishToolPolicy`, and `TodoCompletePolicy`. Callers append their own.

**Recommendation: Option 3.** `StepOutcome.status` is a discriminated `completed | max_steps | budget_exceeded | no_progress | awaiting_input | failed | cancelled`, and `stop_reason` names the policy that fired. `NoProgressPolicy` is included in the default chain specifically because unproductive looping is the most common and most expensive real-world failure, and it is only detectable because DD-13 gives us a diffable state snapshot per step.

**Your comment:**
>
>

---

### DD-22: Long-Running Execution — In-Process Await vs. Externalized Continuation

**Issue.** "Manage short and long running executions." A run may be sub-second or may span hours across interrupts and bulk workflows. What is the execution contract?

**Why it matters.** These are different operational shapes: short runs want low latency and simple ergonomics; long runs need to survive deploys, avoid holding worker slots, report progress, and be cancellable. If the harness only supports the blocking shape, long runs are impossible; if it only supports the continuation shape, simple use becomes needlessly painful and adoption suffers. Because DD-13 already externalizes state, supporting both is cheap — but the *contract* (who owns the scheduler?) must be explicit, or callers will build incompatible drivers.

**Options.**

1. *Blocking `run()` only.* Simple; fails long runs.
2. *Continuation only.* Caller drives every step. Maximally flexible, poor ergonomics for the common case.
3. *`step()` as the primitive, with drivers on top.* `AgentLoop.step()` is the sole primitive. `LocalDriver` loops it in-process (short runs, CLI, tests). `QueueDriver` enqueues a continuation message after each step (long runs, distributed workers). Both are thin and use identical harness code — the only difference is who calls `step()`.

**Recommendation: Option 3.** The harness ships `LocalDriver` only; `QueueDriver` is an adapter (Celery, SQS, Temporal, whatever the caller uses) because queue semantics are a deployment concern the library should not opinionate on. We publish the contract — *load state, step, save state, re-enqueue unless terminal* — plus a reference implementation in the docs.

A `LockProvider` port prevents two workers stepping the same session concurrently; the in-memory default is a no-op, and Redis/Postgres advisory-lock adapters are provided for real deployments. Cancellation is cooperative: a `cancel_requested` flag on `SessionState` is checked at each step boundary, which is the only safe cancellation point given tools may be mid-flight.

**Your comment:**
>
>

---

### DD-23: Reasoning Node Selection — Stream vs. Structured for the Loop's Core Call

**Issue.** Each step needs one LLM call that may emit both prose and tool calls. `LLMStreamNode` supports tools and reconstructs streamed tool calls; `LLMStructuredNode` returns tool calls non-streaming. Which drives the loop?

**Why it matters.** This determines whether the harness can support real-time UI. Streaming is required for any interactive surface, and the existing `LLMStreamNode` already handles the hard part — reconstructing fragmented tool-call arguments across chunks. But streaming adds latency-to-first-tool-call complexity and makes some providers' structured-output guarantees weaker. Choosing wrong means either no streaming UI (a serious product gap) or a loop that cannot exploit strict structured outputs where they matter.

**Options.**

1. *Structured only.* Simplest, strongest output guarantees, no streaming. Rejected — the UI gap is disqualifying.
2. *Stream only.* Uniform and streaming-capable; slightly weaker guarantees for strict-schema cases.
3. *Stream for reasoning, Structured for workflow workers.* The main loop uses `LLMStreamNode` (so events reach the UI and tool calls are reconstructed centrally). Deterministic workflow steps — where output must validate against `output_schema` and nobody is watching a stream — use `LLMStructuredNode`.

**Recommendation: Option 3.** It uses each node where its guarantees actually matter and requires no new node types. Both paths share tool-argument validation (SG-05), so `ToolValidationError` remains the single retryable failure mode for malformed tool arguments in either path.

**Your comment:**
>
>

---

### DD-24: Cognitive Tools — Prompt Instructions vs. Real Tool Calls

**Issue.** Should the agent's self-management operations (write memory, update TODO, extract facts, record decisions) be tool calls, or should the model be asked to emit structured markers in its prose that we parse?

**Why it matters.** Parsed markers are cheaper (no extra round-trip, no tool-schema tokens) but fragile — format drift breaks silently and produces an agent that appears to work while quietly losing all its state updates. Tool calls are validated, observable, and auditable (each produces a `ToolCallStarted`/`Finished` event), but consume schema tokens on every call and add round-trips. There is also a real risk of the model over-using cognitive tools and spending its whole budget on bookkeeping instead of the task.

**Options.**

1. *Prompt markers, parsed from prose.* Cheapest, most fragile. Rejected — silent state loss is the worst failure mode.
2. *Full cognitive tool suite, always exposed.* Validated and observable; costs prompt tokens and invites bookkeeping loops.
3. *Tool calls, with a minimal always-on core and the rest conditionally exposed.* `memory.write`, `todo.update`, and `finish` are always available. `todo.create`, `workflow.run`, `subagent.spawn`, `human.ask`, and `artifact.*` are exposed conditionally — only after the task is classified as complex, or once a plan exists.

**Recommendation: Option 3.** It preserves validation and auditability while keeping the prompt small for trivial tasks, which is exactly the "don't create TODOs for trivial questions" requirement from the brief expressed as a mechanism rather than a plea in the prompt. Conditional exposure is implemented via `ToolFilter` (DD-16), so no special-casing leaks into the loop.

The complexity classification the prior orchestrator did as a separate LLM call is instead folded into the first step: the model is given `todo.create` and simply doesn't call it for trivial tasks. This removes a full round-trip from every single run.

**Your comment:**
>
>

---

### DD-25: State Schema Evolution — Implicit vs. Versioned with Migrations

**Issue.** Once `SessionState` and `MemoryRecord` are persisted in a caller's database, changing their shape breaks stored data. Do we plan for this now or react later?

**Why it matters.** This is cheap now and very expensive later. The moment a user has in-flight sessions in Postgres, any field rename becomes a migration problem *in their infrastructure*, not ours. Long-running sessions (DD-22) mean a session may be written by version N and read by N+1 after a deploy — so cross-version reads are not an edge case, they are the normal case for any run spanning a release. Without a version field there is no way to even detect the mismatch; the failure is a confusing Pydantic validation error at resume time.

**Options.**

1. *No versioning.* Free now, unbounded pain later. Rejected.
2. *Version field only.* `schema_version: int` on persisted models; on mismatch, fail loudly with a clear error. Nearly free, converts a silent corruption into an actionable message. Does not actually migrate anything.
3. *Version plus a migration registry.* `@migration(from=1, to=2)` functions applied on load. Full forward compatibility, but real machinery to build and test before we have any schemas worth migrating.

**Recommendation: Option 2 now, Option 3 when it is needed.** Add `schema_version` to `SessionState`, `MemoryRecord`, `WorkflowSpec`, and `ArtifactMeta` from Phase 1, with a `SchemaVersionError` raised on mismatch. Leave a documented hook where the migration registry will attach. The version field costs nothing and is impossible to add retroactively to already-persisted data; the migration machinery costs a lot and is easy to add later.

**Your comment:**
>
>

---

## 5. Suggestions

Non-blocking recommendations. Each is additive and can be adopted independently.

### SG-06: Persist the Event Stream as the Audit Trail

`SessionState` snapshots give current state; the `AgentEvent` stream gives *how we got here*. Persisting events (via an `EventSink` port with a no-op default) gives audit, cost attribution, replay for debugging, and training data for evaluation — without paying for full event sourcing (DD-13). Recommended from Phase 2, since events are already being emitted and the incremental cost is one port.

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

### SG-12: A Trace-Based Evaluation Harness

Once real runs exist, capture traces as fixtures and replay them against the loop with a recording adapter. This converts prompt and policy changes from unfalsifiable judgement calls into regression tests. Gated on maturity — it needs real traces first — but worth designing the recording adapter early so traces are being captured before we need them.

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

class HarnessTool(BaseModel):
    namespace: str
    definition: ToolDefinition                    # reuses spec 01 model
    scope: ToolScope
    provider_id: str | None = None
    requires_approval: bool = False               # DD-18 approval gating
    is_side_effecting: bool = True                # SG-13 ordering
    max_result_chars: int = 8_000                 # SG-08 spillover threshold

    @property
    def qualified_name(self) -> str:
        return f"{self.namespace}:{self.definition.name}"

class ToolResult(BaseModel):
    tool_name: str
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

class AgentConfig(BaseModel):
    llm: LLMConfig                                # reuses spec 01 model
    persona: AgentPersona = Field(default_factory=AgentPersona)
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
| `ToolExecutionError` | `harness` | varies | Provider-side tool failure; `retryable` set by the provider. |
| `InterruptTimeoutError` | `harness` | `False` | `InterruptRequest.timeout_at` elapsed (DD-18). |
| `WorkflowError` | `harness` | `False` | Workflow validation or fatal execution failure (DD-17). |

`ToolValidationError` (SG-05) is deliberately reused unchanged: malformed tool arguments are already modelled as retryable, and the harness's response — re-prompt with the validation message — is exactly the intended behaviour.

---

## 7. Implementation Phases

Phases 1–5 are **buildable now** with the information on hand. Phases 6–8 are **maturity-gated**: each names the concrete signal required before starting, because designing them without real traces risks abstractions that fit no real workload.

### Phase 1 — Ports, State, Events, Errors

**Goal:** Establish every dependency-inversion boundary and the serializable state model before any logic depends on them.

**Deliverables:**
- `harness/ports/*` — `SessionStore`, `MemoryStore`, `ArtifactStore`, `BlobStore`, `ToolProvider`, `Tokenizer`, `LockProvider`, `Clock`, `IdGenerator`, `VectorIndex` (declared, unimplemented)
- `harness/state.py` — `SessionState`, `Step`, `TodoList`, `Plan`, `RunStatus`, `StepUsage`, all with `schema_version` (DD-25)
- `harness/events.py` — `AgentEvent` discriminated union, ported and extended from the `build/lib` prototype
- `harness/errors.py` — the taxonomy above, extending `LLMError`
- `harness_adapters/memory_backend.py` — in-memory implementations of all ports so the harness runs with zero configuration
- Re-land `nodes/function_node.py` (`FunctionNode`, `FunctionResult`) from `build/lib` into the live tree

**Tests:** Round-trip serialization of every persisted model; `schema_version` mismatch raises `SchemaVersionError`; optimistic-concurrency conflict on `SessionStore.save()`; in-memory adapters satisfy their ABCs.

**Exit criteria:** A `SessionState` survives `model_dump_json()` → `model_validate_json()` losslessly. No module under `harness/` imports from `harness_adapters/`.

---

### Phase 2 — Minimal Loop (Reason → Tool → Observe)

**Goal:** The smallest thing that is genuinely an agent: multi-step, tool-calling, persisted after every step.

**Deliverables:**
- `harness/loop.py` — `AgentLoop.step()` (DD-13, DD-22), `AgentConfig`, `StepOutcome`, `LocalDriver`, and the convenience `run()` wrapper
- `harness/tools/registry.py` + `spec.py` — `ToolRegistry`, `HarnessTool`, `ToolResult`, `ToolFilter` (DD-16)
- `harness/tools/cognitive.py` — the always-on core: `memory.write`, `todo.update`, `finish` (DD-24)
- `harness/policy.py` — `MaxStepsPolicy`, `FinishToolPolicy`, `NoProgressPolicy` (DD-21)
- `harness/context/assembler.py` — first `ContextAssembler`: state header + bounded raw window (DD-15)
- Reasoning call via `LLMStreamNode` with tools (DD-23); events published to `EventBus`
- `FunctionToolProvider` wrapping `FunctionNode` instances

**Tests:** Fake adapter scripting multi-step tool-call sequences; state persisted and reloaded between steps; each stop policy fires correctly; `NoProgressPolicy` halts a deliberately stuck loop; tool-name collisions resolve deterministically.

**Exit criteria:** An agent completes a 3-tool task, is killed mid-run, and resumes from the persisted session on a fresh `AgentLoop` instance with identical results.

---

### Phase 3 — Memory & Context Discipline

**Goal:** Typed memory with enforced policies, and a context assembler that never silently drops information.

**Deliverables:**
- `harness/memory/models.py` — `MemoryRecord`, `MemoryKind`, `MemoryScope`, `MemoryQuery` (DD-14)
- `harness/memory/policy.py` + `service.py` — retention, promotion (observation → fact), and recall
- Cognitive tools: `memory.read`, `memory.promote`, `notes.append`
- `harness/context/budget.py` — `TokenBudget` with the State > Decisions > Facts > Observations priority from [02_agent_context](./02_agent_context)
- Eviction routes through `MemoryService` — never a silent delete (DD-15)
- `ContextAssembled` event with per-section token breakdown (SG-09); stable-prefix section ordering (SG-10)
- Tool-result truncation with artifact spillover (SG-08)

**Tests:** Observation eviction respects `max_active_observations` while decisions survive; budget overflow evicts in priority order; evicted content is recoverable from memory; a large tool result spills to an artifact and leaves a working reference.

**Exit criteria:** A 30-step run stays within `context_token_budget` and can still cite a decision made at step 2.

---

### Phase 4 — Artifacts, Subagents, and Parallelism

**Goal:** Bounded parallel exploration with context isolation.

**Deliverables:**
- `harness/ports/artifact_store.py` implementation + `harness_adapters/filesystem.py` (local session/artifact/blob store)
- Cognitive tools: `artifact.write`, `artifact.read`, `artifact.list`
- `harness/workflow/subagent.py` — `SubagentRunner`, `SubagentResult`, depth capping, budget inheritance (DD-20)
- `subagent.spawn` tool with an explicit brief and a restricted tool subset
- Parallel tool execution under `TaskGroup` with `max_parallel_tools` (SG-13)
- Idempotency keys on tool invocations (SG-11)

**Tests:** Subagent context is isolated (parent window unchanged apart from the single summary observation); depth cap enforced; child budget deducted from parent; parallel tool results preserve call order; repeated invocation with the same idempotency key executes once.

**Exit criteria:** A parent agent spawns 5 concurrent explorers and its own context grows by 5 summaries, not 5 transcripts.

---

### Phase 5 — Deterministic Workflows

**Goal:** The 300-document scenario, executed deterministically and resumably.

**Deliverables:**
- `harness/workflow/models.py` — `WorkflowSpec`, `StepSpec`, `ForEachSpec`, `WorkflowResult` (DD-17)
- `harness/workflow/engine.py` — `Sequence`, `ForEach` (with `max_concurrency`), `Map`, `Reduce`
- Per-item workers use `LLMStructuredNode` against `output_schema` (DD-23)
- Per-item results written to `ArtifactStore`; a manifest artifact returned to the parent
- `workflow.run` cognitive tool, exposed conditionally (DD-24)
- Workflow progress checkpointed into `SessionState` so a mid-workflow crash resumes at the next unprocessed item

**Tests:** `ForEach` over 100 fake items respects `max_concurrency`; per-item failures are isolated under `continue_on_error`; a crash at item 50 resumes at item 50, not item 0; outputs validate against `output_schema`.

**Exit criteria:** 300 documents classified in a single agent run, with the parent context growing by one summary and one manifest reference.

---

### Phase 6 — Human / Agent in the Loop *(gated)*

**Gate:** A concrete integration exists that owns a transport (webhook, queue, or UI). Interrupt ergonomics are unknowable without a real consumer, and building the resume API against an imagined one guarantees rework.

**Deliverables:**
- `harness/interrupt.py` — `InterruptRequest`, `Resumption` (DD-18)
- `AgentLoop.resume(session_id, resumption)`
- `human.ask` and `agent.ask` tools
- Approval gating via `HarnessTool.requires_approval`, reusing the same suspend/resume path
- `InteractiveResponder` adapter for synchronous CLI use
- Timeout handling driven by the caller's scheduler

**Exit criteria:** An agent suspends on a clarification, the process exits entirely, and a new process resumes it from persisted state with the answer injected.

---

### Phase 7 — Production Persistence Adapters *(gated)*

**Gate:** At least one real deployment with a chosen infrastructure stack. Writing a Postgres schema before knowing the query patterns — and the actual size distribution of sessions and memory — produces a schema that will be rewritten.

**Deliverables:**
- `harness_adapters/sqlalchemy_backend.py` — session + memory + artifact-metadata stores `[harness-sql]`
- `harness_adapters/redis_backend.py` — session store and `LockProvider` `[harness-redis]`
- `harness_adapters/s3_backend.py` — `BlobStore` `[harness-s3]`
- `QueueDriver` reference implementation and the documented continuation contract (DD-22)
- `EventSink` port and a persisted event trail (SG-06)
- `Redactor` port applied before persistence and emission (SG-14)
- Migration registry hook activated if a schema change has landed (DD-25)

**Exit criteria:** A single session is stepped by two different worker processes without state loss or lock contention.

---

### Phase 8 — MCP, Semantic Recall, and Evaluation *(gated)*

**Gate:** Distinct signals per item — MCP requires a target server worth integrating; semantic recall requires memory volumes where keyword and kind filtering demonstrably fail; evaluation requires a corpus of real traces.

**Deliverables:**
- `harness_adapters/mcp_provider.py` — MCP `ToolProvider` with dynamic discovery `[harness-mcp]`
- `VectorIndex` implementation and embedding-backed `MemoryQuery.text` recall, using the existing `LLMEmbeddingNode`
- Trace recording adapter and replay-based regression suite (SG-12)
- `PricingTable` and cost dashboards (SG-07)

**Exit criteria:** MCP tools are callable without any core change; semantic recall measurably beats keyword recall on a real corpus; a prompt change can be evaluated against recorded traces before merge.

---

## 8. What to Build Now vs. What to Wait For

| Now (Phases 1–5) | Wait (Phases 6–8) | Why wait |
|---|---|---|
| Ports, `SessionState`, `schema_version` | Migration registry (DD-25) | No schemas to migrate yet; the version field is the part that must exist now. |
| Loop, stop policies, cognitive tool core | Interrupt / resume (DD-18) | Transport ownership is the unknown; the state model already supports it. |
| Typed memory + retention/promotion | Semantic recall via `VectorIndex` | Relevance tuning needs real volume; the port and `embedding_ref` field are pre-placed. |
| Bounded context window + eviction to memory | Learned or adaptive compaction | Requires traces to evaluate against (SG-12). |
| In-memory + filesystem adapters | Postgres / Redis / S3 adapters | Schema follows real query patterns, not speculation. |
| `FunctionToolProvider`, `ToolRegistry`, scopes | MCP provider | The port makes MCP purely additive whenever a target appears. |
| `ForEach` / `Map` / `Reduce` / `Sequence` | Conditionals, dynamic loops, arbitrary DAGs | Avoids accidentally designing a programming language before a workload demands one. |
| Per-step token and cost accounting | `PricingTable` + dashboards | Accounting must be structural from day one; pricing data goes stale and belongs to the caller. |

The organising principle: **build every seam now, defer every implementation that needs empirical data.** Ports, versioning, and accounting fields are cheap now and prohibitively expensive to retrofit. Policies, schemas, and heuristics are the opposite — cheap later, and near-certainly wrong if guessed at now.

---

## 9. Deferred Items

Items from [future.md](./future.md) that this spec supersedes or advances:

| future.md item | Status |
|---|---|
| #1 EventBus & Observability | **Advanced** — `EventBus` lands in Phase 1; persisted `EventSink` in Phase 7 (SG-06). |
| #2 Runtime / Execution Engine | **Partially superseded** — the harness provides step-level orchestration, budgets, and stop policies. A generic retry runtime remains separate and complementary; `retryable` on `LLMError` is the shared contract. |
| #3 FunctionNodes | **Advanced** — re-landed in Phase 1 as the substrate for `FunctionToolProvider`. |
| #4 FlowNodes | **Partially superseded** — `ForEach` / `Map` / `Reduce` / `Sequence` land in Phase 5 as `WorkflowSpec`. Full DAG support remains deferred (DD-17). |
| #5 Agent Loops | **Superseded** — this spec. |
| #6 Memory / Context Management | **Superseded** — DD-14, DD-15, Phase 3. |
| #7 Multi-Modal Inputs | **Still deferred** — requires the `Message.content` change noted in future.md. `ArtifactStore` and `BlobStore` are the intended carriers when it lands. |
| #9 Anthropic Adapter | **Still deferred** — orthogonal to this spec; the harness is provider-agnostic by construction. |

New deferrals introduced here: workflow conditionals and DAGs (DD-17), event-sourced state (DD-13), migration registry (DD-25), and the learned-compaction context strategies implied by DD-15.
