"""Tool registry owned by an Agent instance.

The registry is reusable and stateless with respect to a single run. Runtime
tool call messages still belong to ToolCallingLoop, not this object.
"""
from __future__ import annotations

from typing import Any


class ToolRegistry:
    """Name-based lookup and schema export for agent tools."""

    def __init__(self, tools: list | None = None) -> None:
        self.tools = tools or []
        self.tool_map: dict[str, Any] = {t.name: t for t in self.tools}

    def schemas(self) -> list[dict]:
        """Return OpenAI-compatible tool schemas for tools that expose them."""
        schemas = []
        for tool in self.tools:
            if hasattr(tool, "get_openai_tool_schema"):
                schemas.append(tool.get_openai_tool_schema())
        return schemas

    async def execute(self, tool_name: str, args: dict) -> dict:
        """Execute a registered tool and normalize failures into dict errors."""
        tool = self.tool_map.get(tool_name)
        if tool is None:
            return {"error": f"Tool '{tool_name}' not found"}
        try:
            return await tool.execute(**args)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
