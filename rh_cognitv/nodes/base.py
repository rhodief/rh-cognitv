"""BaseNode — the abstract base for all execution nodes (DD-08, DD-12).

Every node is an async-only unit of work that can be used in two ways:

1. **Directly** — ``result = await node.run(...)``.
2. **As a generic async callable** — ``await node(...)``, which a future
   runtime engine can treat uniformly via ``Execution(fn=node, kwargs={...})``.

``__call__`` delegates to ``run`` by default. Streaming nodes override
``__call__``/``run`` to return an ``AsyncGenerator`` instead of a coroutine.
"""

from __future__ import annotations

import abc
from typing import Any, Generic, TypeVar

ResultT = TypeVar("ResultT")


class BaseNode(abc.ABC, Generic[ResultT]):
    """Abstract base class for all execution nodes.

    Subclasses implement :meth:`run`. The default :meth:`__call__` simply
    delegates to :meth:`run` so nodes satisfy a generic async-callable
    protocol (DD-08).
    """

    @abc.abstractmethod
    async def run(self, *args: Any, **kwargs: Any) -> ResultT:
        """Execute the node and return its canonical result."""
        raise NotImplementedError

    async def __call__(self, *args: Any, **kwargs: Any) -> ResultT:
        """Delegate to :meth:`run` so the node is a generic async callable."""
        return await self.run(*args, **kwargs)
