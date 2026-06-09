"""Shared pytest configuration.

Loads the project ``.env`` (if present) so opt-in integration tests can read
real provider API keys. Unit tests never touch the network — they use fakes.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:  # pragma: no cover - python-dotenv is a dev dependency
    pass


def _has_key(name: str) -> bool:
    return bool(os.getenv(name))


requires_openai = pytest.mark.skipif(
    not _has_key("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; skipping OpenAI integration test",
)

requires_gemini = pytest.mark.skipif(
    not _has_key("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set; skipping Gemini integration test",
)
