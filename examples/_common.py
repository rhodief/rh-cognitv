"""Shared helpers for the example scripts.

Each example picks a provider adapter based on the ``RH_PROVIDER`` environment
variable (``openai`` by default, or ``gemini``). The matching API key must be
available in the environment (``OPENAI_API_KEY`` / ``GEMINI_API_KEY``); a local
``.env`` file is loaded automatically if ``python-dotenv`` is installed.

Run an example with, e.g.::

    RH_PROVIDER=openai python examples/text_completion.py
    RH_PROVIDER=gemini python examples/streaming.py
"""

from __future__ import annotations

import os

try:  # optional convenience for local development
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is only a dev convenience
    pass

# Sensible default models per provider and capability.
_CHAT_MODELS = {"openai": "gpt-4o-mini", "gemini": "gemini-2.5-flash"}
_EMBED_MODELS = {"openai": "text-embedding-3-small", "gemini": "gemini-embedding-001"}


def provider_name() -> str:
    """Return the selected provider name (``openai`` or ``gemini``)."""
    name = os.getenv("RH_PROVIDER", "openai").lower()
    if name not in _CHAT_MODELS:
        raise SystemExit(
            f"Unknown RH_PROVIDER={name!r}; expected one of {sorted(_CHAT_MODELS)}"
        )
    return name


def make_adapter():
    """Construct the adapter for the selected provider."""
    name = provider_name()
    if name == "openai":
        from rh_cognitv.nodes.llm_adapters.openai_adapter import OpenAIAdapter

        return OpenAIAdapter()
    from rh_cognitv.nodes.llm_adapters.gemini_adapter import GeminiAdapter

    return GeminiAdapter()


def chat_model() -> str:
    """Default chat model for the selected provider."""
    return _CHAT_MODELS[provider_name()]


def embed_model() -> str:
    """Default embedding model for the selected provider."""
    return _EMBED_MODELS[provider_name()]
