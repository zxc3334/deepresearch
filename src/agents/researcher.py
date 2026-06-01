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
        loop_config: ToolLoopConfig | None = None,
        external_prefetch_config: dict[str, Any] | None = None,
        trace_recorder=None,
        progress_callback=None,
    ) -> None:
        super().__init__(name, policy, tools, pool_type_key=pool_type_key)
        self.max_turns = max_turns
        self.prompt_builder = ResearchPromptBuilder()
        self.tool_registry = ToolRegistry(tools)
        self.loop_config = loop_config or ToolLoopConfig(max_turns=max_turns)
        self.external_prefetch_config = external_prefetch_config or {}
        self.trace_recorder = trace_recorder
        self.progress_callback = progress_callback
        # Compatibility alias for code that still reads researcher.tool_map.
        self.tool_map: dict[str, Any] = self.tool_registry.tool_map

    @trace_agent(name="researcher.run", tags=["agent", "researcher"])
    async def run(self, task: SubTask, context: dict) -> AgentResult:
        """Execute one SubTask with a fresh loop runtime."""
        task_prompt = self.prompt_builder.task_prompt(task, context)

        if self.prompt_builder.is_non_searchable(task, context):
            return await self._run_direct_analysis(task, task_prompt)

        prefetch_steps = await self._external_prefetch(task)
        if prefetch_steps:
            task_prompt = self._inject_prefetch_context(task_prompt, prefetch_steps)

        loop = ToolCallingLoop(
            policy=self.policy,
            tool_registry=self.tool_registry,
            config=self.loop_config,
            trace_recorder=self.trace_recorder,
            progress_callback=self.progress_callback,
        )
        result = await loop.run(
            task=task,
            system_prompt=self.prompt_builder.system_prompt(self.tool_registry.tools),
            user_prompt=task_prompt,
        )
        if prefetch_steps:
            result.trajectory = prefetch_steps + result.trajectory
        return self.finalize_result(result)

    async def _external_prefetch(self, task: SubTask) -> list[dict[str, Any]]:
        """Optionally prefetch external evidence before the LLM chooses tools."""
        cfg = self.external_prefetch_config
        if not cfg or not cfg.get("enabled", False):
            return []

        query = self._prefetch_query(task)
        timeout = float(cfg.get("timeout_seconds", 15))
        steps: list[dict[str, Any]] = []
        requests = [
            ("web_search", {"query": query, "top_n": int(cfg.get("web_search_top_n", 3))}),
            ("paper_search", {"query": query, "max_results": int(cfg.get("paper_search_top_n", 3))}),
        ]

        if self.trace_recorder:
            self.trace_recorder.record(
                "external_prefetch_start",
                task_id=task.task_id,
                query=query,
                tools=[name for name, _ in requests if name in self.tool_registry.tool_map],
            )

        for tool_name, args in requests:
            if tool_name not in self.tool_registry.tool_map:
                continue
            try:
                result = await asyncio.wait_for(
                    self.tool_registry.execute(tool_name, args),
                    timeout=timeout,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                steps.append({
                    "turn": -1,
                    "role": "tool",
                    "name": tool_name,
                    "tool_call_id": f"prefetch_{tool_name}",
                    "error": error,
                    "prefetch": True,
                })
                if self.trace_recorder:
                    self.trace_recorder.record(
                        "external_prefetch_error",
                        task_id=task.task_id,
                        tool=tool_name,
                        error=error,
                    )
                continue

            if isinstance(result, dict) and result.get("error"):
                error = str(result["error"])
                steps.append({
                    "turn": -1,
                    "role": "tool",
                    "name": tool_name,
                    "tool_call_id": f"prefetch_{tool_name}",
                    "error": error,
                    "prefetch": True,
                })
                if self.trace_recorder:
                    self.trace_recorder.record(
                        "external_prefetch_error",
                        task_id=task.task_id,
                        tool=tool_name,
                        error=error,
                    )
                continue

            normalized = self._limit_prefetch_result(result)
            steps.append({
                "turn": -1,
                "role": "tool",
                "name": tool_name,
                "tool_call_id": f"prefetch_{tool_name}",
                "result": normalized,
                "prefetch": True,
            })
            if self.trace_recorder:
                self.trace_recorder.record(
                    "external_prefetch_result",
                    task_id=task.task_id,
                    tool=tool_name,
                    total=normalized.get("total") if isinstance(normalized, dict) else None,
                    source=normalized.get("source", "") if isinstance(normalized, dict) else "",
                    items=self._prefetch_trace_items(normalized),
                )

        return steps

    def _prefetch_query(self, task: SubTask) -> str:
        hints = " ".join(task.search_hints or [])
        return " ".join(part for part in [task.description, hints] if part).strip()

    def _limit_prefetch_result(self, result: Any) -> Any:
        if not isinstance(result, dict):
            return result
        limited = dict(result)
        max_chars = int(self.external_prefetch_config.get("max_chars_per_item", 500))

        if isinstance(limited.get("results"), list):
            limited["results"] = [
                self._limit_prefetch_item(item, max_chars)
                for item in limited["results"]
                if isinstance(item, dict)
            ]
            limited["total"] = len(limited["results"])

        if isinstance(limited.get("papers"), list):
            limited["papers"] = [
                self._limit_prefetch_paper(paper, max_chars)
                for paper in limited["papers"]
                if isinstance(paper, dict)
            ]
            limited["total"] = len(limited["papers"])

        return limited

    def _limit_prefetch_item(self, item: dict[str, Any], max_chars: int) -> dict[str, Any]:
        limited = dict(item)
        limited["snippet"] = str(limited.get("snippet", "") or "")[:max_chars].rstrip()
        return limited

    def _limit_prefetch_paper(self, paper: dict[str, Any], max_chars: int) -> dict[str, Any]:
        limited = dict(paper)
        limited["summary"] = str(limited.get("summary", "") or "")[:max_chars].rstrip()
        return limited

    def _inject_prefetch_context(self, task_prompt: str, steps: list[dict[str, Any]]) -> str:
        blocks = []
        for step in steps:
            if step.get("error"):
                blocks.append(f"- {step.get('name')}: ERROR {step['error']}")
                continue
            result = step.get("result")
            if not isinstance(result, dict):
                continue
            blocks.extend(self._format_prefetch_result(step.get("name", ""), result))
        if not blocks:
            return task_prompt
        return (
            f"{task_prompt}\n\n"
            "## External Prefetch Evidence\n"
            "Use these pre-fetched external sources as initial evidence. You may still call tools to verify or refine them.\n"
            + "\n".join(blocks[:10])
        )

    def _format_prefetch_result(self, tool_name: str, result: dict[str, Any]) -> list[str]:
        lines: list[str] = []
        for item in result.get("results", []) or []:
            if not isinstance(item, dict):
                continue
            title = item.get("title", "")
            url = item.get("url", "")
            tier = item.get("_source_tier", "")
            score = item.get("_quality_score", "")
            snippet = item.get("snippet", "")
            lines.append(f"- {tool_name}: {title} | {url} | source_tier={tier} | quality={score} | {snippet}")
        for paper in result.get("papers", []) or []:
            if not isinstance(paper, dict):
                continue
            title = paper.get("title", "")
            url = paper.get("url", "") or paper.get("pdf_url", "")
            summary = paper.get("summary", "")
            lines.append(f"- {tool_name}: {title} | {url} | source_tier=academic | {summary}")
        return lines

    def _prefetch_trace_items(self, result: Any) -> list[dict[str, Any]]:
        if not isinstance(result, dict):
            return []
        items: list[dict[str, Any]] = []
        for item in result.get("results", []) or []:
            if isinstance(item, dict):
                items.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "source_tier": item.get("_source_tier", ""),
                    "quality_score": item.get("_quality_score"),
                })
        for paper in result.get("papers", []) or []:
            if isinstance(paper, dict):
                items.append({
                    "title": paper.get("title", ""),
                    "url": paper.get("url", "") or paper.get("pdf_url", ""),
                    "source_tier": "academic",
                    "citation_count": paper.get("citation_count"),
                })
        return items[:10]

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
