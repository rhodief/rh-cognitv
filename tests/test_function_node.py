"""Unit tests for FunctionNode (Phase 2)."""

from __future__ import annotations

import asyncio
import pytest
from pydantic import ValidationError

from rh_cognitv.nodes.function_node import FunctionNode, FunctionResult


def sync_add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


async def async_multiply(x: float, y: float) -> float:
    """Multiply two floats asynchronously."""
    await asyncio.sleep(0.01)
    return x * y


def sync_divide(a: float, b: float) -> float:
    return a / b


@pytest.mark.asyncio
async def test_function_node_sync_execution() -> None:
    node = FunctionNode(sync_add, name="adder")
    assert node.name == "adder"
    assert node.description == "Add two integers."

    result = await node.run(3, 4)
    assert isinstance(result, FunctionResult)
    assert result.output == 7
    assert result.duration_ms > 0
    assert result.error is None


@pytest.mark.asyncio
async def test_function_node_async_execution() -> None:
    node = FunctionNode(async_multiply)
    assert node.name == "async_multiply"

    result = await node.run(2.5, 4.0)
    assert isinstance(result, FunctionResult)
    assert result.output == 10.0
    assert result.duration_ms >= 10.0  # sleep was 10ms
    assert result.error is None


@pytest.mark.asyncio
async def test_function_node_validation_success() -> None:
    node = FunctionNode(sync_add, validate_args=True)
    result = await node.run(a=5, b=10)
    assert result.output == 15
    assert result.error is None


@pytest.mark.asyncio
async def test_function_node_validation_failure() -> None:
    node = FunctionNode(sync_add, validate_args=True)
    result = await node.run(a="not an int", b=10)
    assert result.output is None
    assert result.error is not None
    assert "Input should be a valid integer" in result.error


@pytest.mark.asyncio
async def test_function_node_validation_disabled() -> None:
    # When validate_args is False, type check is bypassed by Python runtime
    node = FunctionNode(sync_add, validate_args=False)
    # This might fail with a TypeError inside the function body, which is caught and reported
    result = await node.run(a="not an int", b=10)
    assert result.output is None
    assert result.error is not None
    assert "can only concatenate str" in result.error


@pytest.mark.asyncio
async def test_function_node_runtime_error_catching() -> None:
    node = FunctionNode(sync_divide)
    result = await node.run(10.0, 0.0)
    assert result.output is None
    assert result.error is not None
    assert "division by zero" in result.error


@pytest.mark.asyncio
async def test_function_node_call_protocol() -> None:
    node = FunctionNode(sync_add)
    result = await node(2, 2)  # Test __call__ wrapper from BaseNode
    assert result.output == 4
