"""ResearcherAgent executor.

The pooled agent owns reusable components: policy binding, prompt builder,
tool registry, and loop config. Per-run mutable state is created inside
ToolCallingLoop on every run(...) call.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any

from .base_agent import BaseAgent
from .prompt_builder import ResearchPromptBuilder
from .tool_calling_loop import ToolCallingLoop, ToolLoopConfig
from .tool_registry import ToolRegistry
from ..orchestrator.schemas import AgentResult, AgentStatus, SubTask
from ..utils.tracing import trace_agent


__all__ = ["ResearcherAgent"]


class ResearcherAgent(BaseAgent):
    """Research / analysis / verification agent.

    Reusable state:
      - policy: one LLM provider binding for this pooled executor
      - prompt_builder: task/system prompt construction
      - tool_registry: tool lookup and schema export
      - loop_config: static loop limits

    Per-run state:
      - messages, trajectory, token counters, tool results
      - owned by a fresh ToolCallingLoop instance
    """

    def __init__(
        self,
        name: str,
        policy,
        tools: list | None = None,
        max_turns: int = 10,
        pool_type_key: str | None = None,
    ) -> None:
        super().__init__(name, policy, tools, pool_type_key=pool_type_key)
        self.max_turns = max_turns
        self.prompt_builder = ResearchPromptBuilder()
        self.tool_registry = ToolRegistry(tools)
        self.loop_config = ToolLoopConfig(max_turns=max_turns)
        # Compatibility alias for code that still reads researcher.tool_map.
        self.tool_map: dict[str, Any] = self.tool_registry.tool_map

    @trace_agent(name="researcher.run", tags=["agent", "researcher"])
    async def run(self, task: SubTask, context: dict) -> AgentResult:
        """Execute one SubTask with a fresh loop runtime."""
        task_prompt = self.prompt_builder.task_prompt(task, context)

        if self.prompt_builder.is_non_searchable(task, context):
            return await self._run_direct_analysis(task, task_prompt)

        loop = ToolCallingLoop(
            policy=self.policy,
            tool_registry=self.tool_registry,
            config=self.loop_config,
        )
        result = await loop.run(
            task=task,
            system_prompt=self.prompt_builder.system_prompt(),
            user_prompt=task_prompt,
        )
        return self.finalize_result(result)

    async def _run_direct_analysis(self, task: SubTask, task_prompt: str) -> AgentResult:
        """Handle private/subjective tasks without forcing a web search."""
        messages = [
            {"role": "system", "content": self.prompt_builder.direct_analysis_system_prompt()},
            {"role": "user", "content": task_prompt},
        ]
        try:
            response = await asyncio.to_thread(self.policy, messages)
            content = response.get("content", "") or ""
            return self.finalize_result(AgentResult(
                task_id=task.task_id,
                status=AgentStatus.SUCCESS,
                output=content,
                trajectory=[{"role": "assistant", "content": content}],
                token_usage=len(content) // 3,
                confidence=self._extract_confidence(content),
            ))
        except Exception as e:
            return self.finalize_result(AgentResult(
                task_id=task.task_id,
                status=AgentStatus.FAILED,
                output=f"Direct analysis failed: {e}",
                trajectory=[{"error": str(e)}],
                token_usage=0,
                confidence=0.0,
            ))

    def _extract_confidence(self, content: str) -> float:
        """Extract a confidence score from direct-analysis output."""
        patterns = [
            r"[Cc]onfidence[:\s]+(0\.\d+|1\.0|1)",
            r"置信度[:\s]+(0\.\d+|1\.0|1)",
        ]
        for pat in patterns:
            m = re.search(pat, content)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    continue
        return 0.6
