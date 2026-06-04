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
from .tool_context import ContextBudgetManager, ToolResultCompactPolicy, ToolResultNormalizer
from .tool_registry import ToolRegistry


@dataclass(frozen=True)
class ToolLoopConfig:
    """Configuration for one tool-calling loop execution."""

    max_turns: int = 10
    max_tool_calls_before_summary: int = 2
    context_budget_tokens: int = 12000
    compact_threshold_ratio: float = 0.70
    compact_tool_result_chars: int = 4000
    chars_per_token: float = 3.5
    head_tail_ratio: float = 0.70
    auto_fetch_official_docs: bool = True
    auto_fetch_official_max_urls: int = 1
    auto_fetch_official_timeout_seconds: float = 12.0
    auto_fetch_official_max_chars: int = 6000
    auto_fetch_official_max_snippets: int = 3


class ToolCallingLoop:
    """Run a single task through LLM tool calls and tool execution."""

    def __init__(
        self,
        policy,
        tool_registry: ToolRegistry,
        config: ToolLoopConfig | None = None,
        trace_recorder=None,
        progress_callback=None,
    ) -> None:
        self.policy = policy
        self.tool_registry = tool_registry
        self.config = config or ToolLoopConfig()
        self.trace_recorder = trace_recorder
        self.progress_callback = progress_callback
        self.messages: list[dict] = []
        self.trajectory: list[dict] = []
        self.total_tokens: int = 0
        self.tool_result_normalizer = ToolResultNormalizer()
        self.context_budget_manager = ContextBudgetManager(
            budget_tokens=self.config.context_budget_tokens,
            compact_trigger_ratio=self.config.compact_threshold_ratio,
            chars_per_token=self.config.chars_per_token,
        )
        self.tool_compact_policy = ToolResultCompactPolicy(
            max_chars=self.config.compact_tool_result_chars,
            head_ratio=self.config.head_tail_ratio,
            budget_manager=self.context_budget_manager,
        )

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
            usage = response.get("usage", {}) or {}

            self.trajectory.append({
                "turn": turn,
                "role": "assistant",
                "content": content,
                "tool_calls": [dict(tc) for tc in tool_calls],
                "usage": usage,
            })
            if self.trace_recorder:
                self.trace_recorder.record(
                    "llm_call",
                    task_id=task.task_id,
                    turn=turn,
                    role="researcher",
                    usage=usage,
                    tool_call_count=len(tool_calls),
                    output_chars=len(content),
                )

            self.total_tokens += usage.get("total_tokens", 0) or len(json.dumps(self.messages, ensure_ascii=False)) // 3

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

            await self._publish_progress(
                "tool_call_start",
                task_id=task.task_id,
                turn=turn,
                tool_name=tool_name,
                args_summary=self._summarize_args(args),
            )
            result = await self.tool_registry.execute(tool_name, args)
            if self.trace_recorder:
                self.trace_recorder.record(
                    "tool_call",
                    task_id=task.task_id,
                    turn=turn,
                    tool=tool_name,
                    args=args,
                )

            if isinstance(result, dict) and result.get("error"):
                error_msg = result["error"]
                if self.trace_recorder:
                    self.trace_recorder.record(
                        "tool_result",
                        task_id=task.task_id,
                        turn=turn,
                        tool=tool_name,
                        status="error",
                        error=error_msg,
                    )
                self.trajectory.append({
                    "turn": turn,
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "name": tool_name,
                    "error": error_msg,
                })
                await self._publish_progress(
                    "tool_call_error",
                    task_id=task.task_id,
                    turn=turn,
                    tool_name=tool_name,
                    error=error_msg,
                )
                return AgentResult(
                    task_id=task.task_id,
                    status=AgentStatus.FAILED,
                    output=f"Tool '{tool_name}' failed: {error_msg}",
                    trajectory=self.trajectory,
                    token_usage=self.total_tokens,
                    confidence=0.0,
                )

            extra_trajectory = await self._maybe_auto_fetch_official_docs(
                task=task,
                turn=turn,
                source_tool_name=tool_name,
                source_args=args,
                source_result=result,
            )

            tool_result = {
                "tool_call_id": tc.get("id", ""),
                "name": tool_name,
                "result": result,
            }
            self._log_tool_result(task, turn, tool_name, args, result)
            self._trace_tool_result(task, turn, tool_name, result)
            await self._publish_progress(
                "tool_call_result",
                task_id=task.task_id,
                turn=turn,
                tool_name=tool_name,
                result_summary=self._summarize_tool_result(result),
            )
            tool_results.append(tool_result)
            self.trajectory.append({
                "turn": turn,
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "name": tool_name,
                "result": result,
            })
            self.trajectory.extend(extra_trajectory)

        return tool_results

    async def _maybe_auto_fetch_official_docs(
        self,
        task: SubTask,
        turn: int,
        source_tool_name: str,
        source_args: dict[str, Any],
        source_result: Any,
    ) -> list[dict[str, Any]]:
        """Upgrade official search hits into page-grounded official evidence.

        The fetched pages are merged into the official search result so the next
        LLM turn can use the snippets without needing another tool choice. They
        are also recorded as synthetic trajectory/trace tool events so
        EvidenceStore can classify them as official_doc_fetcher evidence.
        """
        if not self.config.auto_fetch_official_docs:
            return []
        if source_tool_name != "official_source_search":
            return []
        if "official_doc_fetcher" not in self.tool_registry.tool_map:
            return []
        if not isinstance(source_result, dict):
            return []

        urls = self._official_search_urls(source_result)
        max_urls = max(0, int(self.config.auto_fetch_official_max_urls))
        if not urls or max_urls <= 0:
            return []

        query = str(source_args.get("query") or source_result.get("query") or task.description or "")
        selected_urls = urls[:max_urls]
        fetched_documents: list[dict[str, Any]] = []
        extra_trajectory: list[dict[str, Any]] = []

        if self.trace_recorder:
            self.trace_recorder.record(
                "official_auto_fetch_start",
                task_id=task.task_id,
                turn=turn,
                urls=selected_urls,
                query=query,
            )

        for index, url in enumerate(selected_urls):
            fetch_args = {
                "url": url,
                "query": query,
                "max_chars": int(self.config.auto_fetch_official_max_chars),
                "max_snippets": int(self.config.auto_fetch_official_max_snippets),
            }
            if self.trace_recorder:
                self.trace_recorder.record(
                    "tool_call",
                    task_id=task.task_id,
                    turn=turn,
                    tool="official_doc_fetcher",
                    args={**fetch_args, "auto_fetch": True},
                )
            try:
                fetched = await asyncio.wait_for(
                    self.tool_registry.execute("official_doc_fetcher", fetch_args),
                    timeout=float(self.config.auto_fetch_official_timeout_seconds),
                )
            except Exception as exc:
                fetched = {"error": f"{type(exc).__name__}: {exc}", "url": url}

            if isinstance(fetched, dict) and fetched.get("error"):
                error_msg = str(fetched.get("error", ""))
                if self.trace_recorder:
                    self.trace_recorder.record(
                        "tool_result",
                        task_id=task.task_id,
                        turn=turn,
                        tool="official_doc_fetcher",
                        status="error",
                        error=error_msg,
                        auto_fetch=True,
                        url=url,
                    )
                extra_trajectory.append({
                    "turn": turn,
                    "role": "tool",
                    "tool_call_id": f"auto_official_doc_fetcher_{index}",
                    "name": "official_doc_fetcher",
                    "error": error_msg,
                    "auto_fetch": True,
                    "source_tool": source_tool_name,
                    "url": url,
                })
                continue

            if not isinstance(fetched, dict):
                continue
            fetched["auto_fetch"] = True
            fetched["source_tool"] = source_tool_name
            fetched_documents.append(fetched)
            self._trace_tool_result(task, turn, "official_doc_fetcher", fetched)
            extra_trajectory.append({
                "turn": turn,
                "role": "tool",
                "tool_call_id": f"auto_official_doc_fetcher_{index}",
                "name": "official_doc_fetcher",
                "result": fetched,
                "auto_fetch": True,
                "source_tool": source_tool_name,
            })

        if fetched_documents:
            source_result["auto_fetched_documents"] = fetched_documents
            source_result["auto_fetched_total"] = len(fetched_documents)
        if self.trace_recorder:
            self.trace_recorder.record(
                "official_auto_fetch_end",
                task_id=task.task_id,
                turn=turn,
                requested=len(selected_urls),
                fetched=len(fetched_documents),
            )
        return extra_trajectory

    def _official_search_urls(self, result: dict[str, Any]) -> list[str]:
        urls: list[str] = []
        for item in result.get("results", []) or []:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url", "") or "").strip()
            if not url or url in urls:
                continue
            urls.append(url)
        return urls

    async def _publish_progress(self, event_type: str, **payload: Any) -> None:
        callback = self.progress_callback
        if callback is None:
            return
        event = {"event_type": event_type, **payload}
        try:
            result = callback(event)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            return

    def _summarize_args(self, args: dict[str, Any]) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for key in ("query", "paper_id", "url", "top_n", "max_results"):
            if key in args:
                value = args[key]
                summary[key] = str(value)[:300] if isinstance(value, str) else value
        return summary

    def _summarize_tool_result(self, result: Any) -> dict[str, Any]:
        if not isinstance(result, dict):
            return {"result_type": type(result).__name__}
        items: list[dict[str, Any]] = []
        for item in result.get("results", []) or []:
            if isinstance(item, dict):
                items.append({
                    "title": str(item.get("title", ""))[:160],
                    "url": item.get("url", ""),
                    "source_tier": item.get("_source_tier", ""),
                    "quality_score": item.get("_quality_score"),
                })
        for paper in result.get("papers", []) or []:
            if isinstance(paper, dict):
                items.append({
                    "title": str(paper.get("title", ""))[:160],
                    "url": paper.get("url", "") or paper.get("pdf_url", ""),
                    "source_tier": "academic",
                    "citation_count": paper.get("citation_count"),
                })
        return {
            "source": result.get("source", ""),
            "total": result.get("total"),
            "items": items[:5],
        }

    def _trace_tool_result(self, task: SubTask, turn: int, tool_name: str, result) -> None:
        if not self.trace_recorder:
            return
        if not isinstance(result, dict):
            self.trace_recorder.record(
                "tool_result",
                task_id=task.task_id,
                turn=turn,
                tool=tool_name,
                status="success",
                result_type=type(result).__name__,
            )
            return

        urls = self._extract_urls(result)
        self.trace_recorder.record(
            "tool_result",
            task_id=task.task_id,
            turn=turn,
            tool=tool_name,
            status="success",
            source=result.get("source", ""),
            total=result.get("total"),
            url_count=len(urls),
            urls=urls[:10],
        )

    def _log_tool_result(self, task: SubTask, turn: int, tool_name: str, args: dict, result) -> None:
        """Print compact tool observability for logs and demo debugging."""
        if not isinstance(result, dict):
            print(f"[ToolCall] task={task.task_id} turn={turn} tool={tool_name} args={args} result_type={type(result).__name__}")
            return

        urls = self._extract_urls(result)
        if result.get("error"):
            print(f"[ToolCall] task={task.task_id} turn={turn} tool={tool_name} args={args} error={result['error']}")
            return
        url_preview = ", ".join(urls[:3]) if urls else "no-url"
        print(
            f"[ToolCall] task={task.task_id} turn={turn} tool={tool_name} "
            f"total={result.get('total', 'n/a')} source={result.get('source', '')} urls={url_preview}"
        )

    def _extract_urls(self, result: dict[str, Any]) -> list[str]:
        urls: list[str] = []
        for item in result.get("results", []) or []:
            if isinstance(item, dict) and item.get("url"):
                urls.append(str(item["url"]))
        for paper in result.get("papers", []) or []:
            if not isinstance(paper, dict):
                continue
            url = paper.get("url") or paper.get("pdf_url")
            if url:
                urls.append(str(url))
        return urls

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
            msg_content = self.tool_result_normalizer.dumps(tr["result"])
            msg_content = self._maybe_compact_tool_content(msg_content, tr["name"])
            if force_summary:
                msg_content += "\n\n[SYSTEM NOTICE] You have already searched enough. Write your final summary NOW. Do NOT call any more tools."
            self.messages.append({
                "role": "tool",
                "tool_call_id": tr["tool_call_id"],
                "content": msg_content,
            })

    def _maybe_compact_tool_content(self, content: str, tool_name: str) -> str:
        """Compact tool result only when projected context exceeds the threshold.

        Error text is never compacted; preserving failure details is more important
        than saving context.
        """
        current_chars = self._messages_chars(self.messages)
        decision = self.tool_compact_policy.compact(content, current_context_chars=current_chars)
        if not decision.compacted:
            return decision.content

        print(
            f"[compact] tool={tool_name} chars={decision.before_chars}->{decision.after_chars} "
            f"projected={current_chars + decision.before_chars}/{decision.threshold_chars}"
        )
        if self.trace_recorder:
            self.trace_recorder.record(
                "compact",
                scope="tool_result",
                tool=tool_name,
                before_chars=decision.before_chars,
                after_chars=decision.after_chars,
                threshold_chars=decision.threshold_chars,
                strategy=decision.strategy,
                reason=decision.reason,
            )
        return decision.content

    def _messages_chars(self, messages: list[dict]) -> int:
        total = 0
        for message in messages:
            if not isinstance(message, dict):
                continue
            total += len(str(message.get("content", "")))
            if message.get("tool_calls"):
                total += len(json.dumps(message.get("tool_calls"), ensure_ascii=False, default=str))
        return total

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
            "error: connection error", "error: request timed out",
            "connection error", "request timed out",
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
