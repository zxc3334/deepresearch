"""Per-run tool-calling loop runtime.

This object is intentionally created fresh for every Agent.run(...) call. It
owns mutable runtime state such as messages, trajectory, and token counters, so
AgentPool can reuse Agent executors without leaking context across tasks.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

from ..orchestrator.schemas import AgentResult, AgentStatus, SubTask
from .tool_registry import ToolRegistry


@dataclass(frozen=True)
class ToolLoopConfig:
    """Configuration for one tool-calling loop execution."""

    max_turns: int = 10
    max_tool_calls_before_summary: int = 2


class ToolCallingLoop:
    """Run a single task through LLM tool calls and tool execution."""

    def __init__(
        self,
        policy,
        tool_registry: ToolRegistry,
        config: ToolLoopConfig | None = None,
    ) -> None:
        self.policy = policy
        self.tool_registry = tool_registry
        self.config = config or ToolLoopConfig()
        self.messages: list[dict] = []
        self.trajectory: list[dict] = []
        self.total_tokens: int = 0

    async def run(self, task: SubTask, system_prompt: str, user_prompt: str) -> AgentResult:
        """Execute the tool-calling loop for one task."""
        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if hasattr(self.policy, "set_tools") and self.tool_registry.tools:
            self.policy.set_tools(self.tool_registry.schemas())

        fallback_tool = self._fallback_tool(task)

        for turn in range(self.config.max_turns):
            self._maybe_force_tool_use(turn, fallback_tool)

            try:
                response = await asyncio.to_thread(self.policy, self.messages)
            except RuntimeError as e:
                self.trajectory.append({"turn": turn, "error": str(e)})
                return AgentResult(
                    task_id=task.task_id,
                    status=AgentStatus.FAILED,
                    output=str(e),
                    trajectory=self.trajectory,
                    token_usage=self.total_tokens,
                    confidence=0.0,
                )
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                self.trajectory.append({"turn": turn, "error": error_msg})
                return AgentResult(
                    task_id=task.task_id,
                    status=AgentStatus.FAILED,
                    output=f"Policy call failed: {error_msg}",
                    trajectory=self.trajectory,
                    token_usage=self.total_tokens,
                    confidence=0.0,
                )

            content = response.get("content", "") or ""
            tool_calls = response.get("tool_calls", []) or []

            self.trajectory.append({
                "turn": turn,
                "role": "assistant",
                "content": content,
                "tool_calls": [dict(tc) for tc in tool_calls],
            })

            self.total_tokens += len(json.dumps(self.messages, ensure_ascii=False)) // 3

            if not tool_calls:
                if self._is_tool_failure_explanation(content):
                    return AgentResult(
                        task_id=task.task_id,
                        status=AgentStatus.FAILED,
                        output=content,
                        trajectory=self.trajectory,
                        token_usage=self.total_tokens,
                        confidence=0.0,
                    )
                return AgentResult(
                    task_id=task.task_id,
                    status=AgentStatus.SUCCESS,
                    output=content,
                    trajectory=self.trajectory,
                    token_usage=self.total_tokens,
                    confidence=self._extract_confidence(content),
                )

            tool_results = await self._execute_tool_calls(task, turn, tool_calls)
            if isinstance(tool_results, AgentResult):
                return tool_results

            force_summary = self._should_force_summary(tool_results)
            self._append_assistant_and_tool_messages(response, content, tool_calls, tool_results, force_summary)

        return AgentResult(
            task_id=task.task_id,
            status=AgentStatus.TIMEOUT,
            output="Reached max_turns without final answer.",
            trajectory=self.trajectory,
            token_usage=self.total_tokens,
            confidence=0.0,
        )

    async def _execute_tool_calls(self, task: SubTask, turn: int, tool_calls: list) -> list[dict] | AgentResult:
        tool_results = []
        for tc in tool_calls:
            func = tc.get("function", {})
            tool_name = func.get("name", "")
            try:
                args = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}

            result = await self.tool_registry.execute(tool_name, args)

            if isinstance(result, dict) and result.get("error"):
                error_msg = result["error"]
                self.trajectory.append({
                    "turn": turn,
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "name": tool_name,
                    "error": error_msg,
                })
                return AgentResult(
                    task_id=task.task_id,
                    status=AgentStatus.FAILED,
                    output=f"Tool '{tool_name}' failed: {error_msg}",
                    trajectory=self.trajectory,
                    token_usage=self.total_tokens,
                    confidence=0.0,
                )

            tool_result = {
                "tool_call_id": tc.get("id", ""),
                "name": tool_name,
                "result": result,
            }
            tool_results.append(tool_result)
            self.trajectory.append({
                "turn": turn,
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "name": tool_name,
                "result": result,
            })

        return tool_results

    def _append_assistant_and_tool_messages(
        self,
        response: dict,
        content: str,
        tool_calls: list,
        tool_results: list[dict],
        force_summary: bool,
    ) -> None:
        assistant_msg = {
            "role": "assistant",
            "content": content,
        }
        if tool_calls:
            assistant_msg["tool_calls"] = [dict(tc) for tc in tool_calls]
        if response.get("reasoning_content"):
            assistant_msg["reasoning_content"] = response["reasoning_content"]
        self.messages.append(assistant_msg)

        for tr in tool_results:
            msg_content = json.dumps(tr["result"], ensure_ascii=False, default=str)
            if force_summary:
                msg_content += "\n\n[SYSTEM NOTICE] You have already searched enough. Write your final summary NOW. Do NOT call any more tools."
            self.messages.append({
                "role": "tool",
                "tool_call_id": tr["tool_call_id"],
                "content": msg_content,
            })

    def _maybe_force_tool_use(self, turn: int, fallback_tool: str) -> None:
        if turn > 0 and self.messages and self.messages[-1].get("role") == "assistant":
            last_tool_calls = self.messages[-1].get("tool_calls", [])
            if not last_tool_calls:
                self.messages.append({
                    "role": "user",
                    "content": (
                        f"You did not use any tools. "
                        f"You MUST call the '{fallback_tool}' tool now to search for information. "
                        f"Do not write a summary without searching first."
                    ),
                })

    def _should_force_summary(self, tool_results: list[dict]) -> bool:
        all_empty = True
        for tr in tool_results:
            if tr["name"] == "web_search":
                res = tr["result"]
                if isinstance(res, dict) and res.get("results"):
                    for r in res["results"]:
                        if r.get("snippet", "").strip():
                            all_empty = False
                            break

        search_count = sum(
            1 for t in self.trajectory
            if t.get("role") == "tool" and t.get("name") == "web_search"
        )
        if search_count >= self.config.max_tool_calls_before_summary:
            return True
        if all_empty and tool_results:
            return True
        return False

    def _fallback_tool(self, task: SubTask) -> str:
        desc_lower = (task.description or "").lower()
        academic_keywords = ["论文", "paper", "publication", "学术", "arxiv", "neurips", "icml", "iclr", "scholar", "citation", "文献"]
        return "arxiv_reader" if any(kw in desc_lower for kw in academic_keywords) else "web_search"

    def _is_tool_failure_explanation(self, content: str) -> bool:
        if not content:
            return False
        c = content.lower()
        failure_keywords = [
            "无法通过", "无法执行", "无法使用", "无法获取", "无法访问",
            "额度已用尽", "配额已用完", "额度已用完", "搜索配额",
            "cannot search", "unable to search", "quota exceeded",
            "api key", "额度不足", "余额不足", "余额为", "余额：0",
            "网络错误", "连接失败", "无法连接到",
        ]
        return any(kw in c for kw in failure_keywords)

    def _extract_confidence(self, content: str) -> float:
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
