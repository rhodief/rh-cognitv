"""FunctionNode — wrap arbitrary Python callables as execution nodes (DD-08, DD-12).

Enables running regular Python logic under the same execution model as LLM nodes,
with uniform results, error handling, and validation.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar
from pydantic import BaseModel, validate_call

from rh_cognitv.nodes.base import BaseNode

P = ParamSpec("P")
R = TypeVar("R")


class FunctionResult(BaseModel):
    """Eventual response carrying the output and metadata of a FunctionNode execution."""

    output: Any
    duration_ms: float
    error: str | None = None


class FunctionNode(BaseNode[FunctionResult]):
    """Wraps a Python callable (sync or async) as a first-class execution node.

    Uses Pydantic's `validate_call` to automatically enforce type-hint validation
    on execution arguments when `validate_args` is True.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        validate_args: bool = True,
    ) -> None:
        self.fn = fn
        self.name = name or fn.__name__
        self.description = description or fn.__doc__ or f"Function node wrapping {self.name}"
        self.validate_args = validate_args

        if validate_args:
            # Pydantic's validate_call decorator enforces annotations at runtime
            self._callable = validate_call(fn)
        else:
            self._callable = fn

    async def run(self, *args: Any, **kwargs: Any) -> FunctionResult:
        """Execute the wrapped function and return a canonical FunctionResult."""
        start = time.perf_counter()
        try:
            if inspect.iscoroutinefunction(self.fn):
                output = await self._callable(*args, **kwargs)
            else:
                output = self._callable(*args, **kwargs)
            duration_ms = (time.perf_counter() - start) * 1000
            return FunctionResult(output=output, duration_ms=duration_ms)
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            return FunctionResult(output=None, duration_ms=duration_ms, error=str(exc))
