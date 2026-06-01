"""Minimal interfaces for LLM policy adapters."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class PolicyAdapter(Protocol):
    """A callable policy used by agents, planners, and compressors."""

    def __call__(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Run one LLM call over OpenAI-style messages."""
        ...

    def set_tools(self, tools: list[dict[str, Any]]) -> None:
        """Bind OpenAI function-calling tool schemas for the next calls."""
        ...
