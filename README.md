# RH COGNITV

**Cognitive Skill-driven Orchestration Framework**

`rh_cognitv` provides a set of async-first **execution nodes** for working with
Large Language Models behind a clean, provider-agnostic interface. Swap between
OpenAI and Google Gemini (more providers to come) without changing your calling
code.

| Node | Purpose |
|------|---------|
| `LLMTextNode` | Single-shot (non-streaming) completion → `TextResult` |
| `LLMStreamNode` | Token/object streaming via async generator → `StreamResult` |
| `LLMStructuredNode` | Tool / function calling with Pydantic schemas → `StructuredResult` |
| `LLMEmbeddingNode` | Batch text → embedding vectors → `EmbeddingResult` |

All nodes share: **Pydantic-first I/O**, **canonical result metadata**
(`tokens_used`, `duration_ms`, `model`, `provider`), a **structured error
taxonomy** (`family`, `code`, `retryable`), dependency-injected **provider
adapters**, and an **async-only** API.

---

## Installation

Requires Python 3.12+.

```bash
# Core package
pip install rh_cognitv

# With a provider SDK (optional dependency groups)
pip install "rh_cognitv[openai]"
pip install "rh_cognitv[gemini]"
pip install "rh_cognitv[all]"
```

Set the matching API key in your environment:

```bash
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=...
```

---

## Quick start

```python
import asyncio

from rh_cognitv import LLMTextNode, LLMConfig
from rh_cognitv.nodes.llm_adapters.openai_adapter import OpenAIAdapter


async def main() -> None:
    node = LLMTextNode(OpenAIAdapter())          # adapter injected at construction
    config = LLMConfig(model="gpt-4o-mini", temperature=0.0)
    result = await node.run("What is the capital of France?", config)
    print(result.text)
    print(result.meta.tokens_used.total_tokens, "tokens")


asyncio.run(main())
```

Switching providers is a one-line change — the node API is identical:

```python
from rh_cognitv.nodes.llm_adapters.gemini_adapter import GeminiAdapter

node = LLMTextNode(GeminiAdapter())
config = LLMConfig(model="gemini-2.5-flash")
```

---

## Usage

### Streaming

`LLMStreamNode.run()` is an async generator of canonical stream events. Use
`collect()` for the consolidated result in one call.

```python
from rh_cognitv import LLMStreamNode, LLMConfig, StreamTextDelta

node = LLMStreamNode(OpenAIAdapter())
config = LLMConfig(model="gpt-4o-mini")

async for event in node.run("Count to five.", config):
    if isinstance(event, StreamTextDelta):
        print(event.text, end="", flush=True)

# Or collect the full StreamResult:
result = await node.collect("Count to five.", config, batch_size=3)
print(result.text)
```

### Tool calling

Wrap a Pydantic model in a `ToolDefinition`. Returned arguments are
auto-validated into a model instance (`parsed_arguments`).

```python
from pydantic import BaseModel
from rh_cognitv import LLMStructuredNode, LLMConfig, ToolDefinition


class GetWeather(BaseModel):
    city: str


tool = ToolDefinition(
    name="get_weather",
    description="Get the current weather for a city",
    parameters_model=GetWeather,
)
node = LLMStructuredNode(OpenAIAdapter())
result = await node.run(
    "What's the weather in Paris?",
    LLMConfig(model="gpt-4o-mini"),
    [tool],
    tool_choice="get_weather",
)
call = result.tool_calls[0]
print(call.tool_name, call.parsed_arguments)  # validated GetWeather instance
```

### Embeddings

```python
from rh_cognitv import LLMEmbeddingNode, LLMConfig

node = LLMEmbeddingNode(OpenAIAdapter())
result = await node.run(["hello world", "goodbye world"],
                        LLMConfig(model="text-embedding-3-small"))
print(len(result.embeddings), "vectors")
```

---

## Examples

Runnable scripts live in [`examples/`](./examples). They select a provider via
the `RH_PROVIDER` environment variable (`openai` by default, or `gemini`):

```bash
python examples/text_completion.py
RH_PROVIDER=gemini python examples/streaming.py
python examples/structured_tools.py
python examples/embeddings.py
```

---

## Error handling

Adapters map provider exceptions onto a canonical taxonomy so calling code is
provider-independent and a future retry engine can consume them directly:

```python
from rh_cognitv import LLMError, RateLimitError

try:
    await node.run("...", config)
except RateLimitError as exc:
    print(exc.family, exc.code, exc.retryable)   # structured fields
except LLMError as exc:
    print("other LLM error:", exc)
```

Error families: `RATE_LIMIT`, `AUTHENTICATION`, `INVALID_REQUEST`,
`CONTEXT_LENGTH`, `TIMEOUT`, `PROVIDER`, `VALIDATION`, `UNKNOWN`.

---

## License

MIT
