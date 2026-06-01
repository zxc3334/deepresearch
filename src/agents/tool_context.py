"""Tool result normalization and context-budget compaction."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


ERROR_MARKERS = (
    '"error"',
    "error:",
    "failed",
    "traceback",
    "exception",
    "connection error",
    "request timed out",
    "api key",
    "unsupported",
    "not supported",
    "rejected",
    "无法",
    "不支持",
)


@dataclass(frozen=True)
class ContextBudgetManager:
    """Estimate prompt context size and decide whether compaction is needed."""

    budget_tokens: int = 12000
    compact_trigger_ratio: float = 0.70
    chars_per_token: float = 3.5

    @property
    def threshold_chars(self) -> int:
        return int(self.budget_tokens * self.chars_per_token * self.compact_trigger_ratio)

    def should_compact(self, current_chars: int, incoming_chars: int) -> bool:
        return current_chars + incoming_chars > self.threshold_chars


@dataclass(frozen=True)
class CompactDecision:
    """Result of compacting a tool result for prompt insertion."""

    content: str
    compacted: bool = False
    before_chars: int = 0
    after_chars: int = 0
    strategy: str = "none"
    reason: str = ""
    threshold_chars: int = 0


class ToolResultNormalizer:
    """Convert arbitrary tool payloads into prompt-safe structured objects."""

    STRUCTURED_KEYS = {
        "title",
        "url",
        "pdf_url",
        "source",
        "source_type",
        "_source_tier",
        "_quality_score",
        "evidence_level",
        "citation_count",
    }

    TEXT_KEYS = ("snippet", "summary", "abstract", "content", "text")

    def normalize(self, payload: Any) -> Any:
        if isinstance(payload, dict):
            return self._normalize_dict(payload)
        if isinstance(payload, list):
            return [self.normalize(item) for item in payload]
        return payload

    def _normalize_dict(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload)
        for list_key in ("results", "papers"):
            if isinstance(normalized.get(list_key), list):
                items = [
                    self._normalize_item(item)
                    for item in normalized[list_key]
                    if isinstance(item, dict)
                ]
                normalized[list_key] = self._sort_items_by_source_quality(items, academic_default=list_key == "papers")
        return normalized

    def _normalize_item(self, item: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in item.items():
            if key in self.STRUCTURED_KEYS:
                normalized[key] = value
            elif key in self.TEXT_KEYS:
                normalized[key] = str(value or "")
            elif key in ("authors", "published", "id"):
                normalized[key] = value
        return normalized

    def _sort_items_by_source_quality(
        self,
        items: list[dict[str, Any]],
        academic_default: bool = False,
    ) -> list[dict[str, Any]]:
        tier_rank = {
            "official": 5,
            "academic": 4,
            "authoritative": 3,
            "general": 2,
            "unverified": 1,
            "": 0,
        }

        def sort_key(index_item: tuple[int, dict[str, Any]]) -> tuple[int, float, int]:
            index, item = index_item
            tier = str(item.get("_source_tier") or item.get("source_tier") or "").lower()
            if academic_default and not tier:
                tier = "academic"
            quality = item.get("_quality_score")
            try:
                quality_score = float(quality)
            except (TypeError, ValueError):
                quality_score = 0.0
            return (tier_rank.get(tier, 0), quality_score, -index)

        return [
            item
            for _, item in sorted(
                enumerate(items),
                key=sort_key,
                reverse=True,
            )
        ]

    def dumps(self, payload: Any) -> str:
        return json.dumps(self.normalize(payload), ensure_ascii=False, default=str)


class ToolResultCompactPolicy:
    """Compact tool results only when they threaten the context budget."""

    def __init__(
        self,
        max_chars: int = 4000,
        head_ratio: float = 0.70,
        budget_manager: ContextBudgetManager | None = None,
    ) -> None:
        self.max_chars = max_chars
        self.head_ratio = head_ratio
        self.budget_manager = budget_manager or ContextBudgetManager()

    def compact(self, content: str, current_context_chars: int = 0) -> CompactDecision:
        before_chars = len(content)
        if self._looks_like_error_content(content):
            return CompactDecision(
                content=content,
                before_chars=before_chars,
                after_chars=before_chars,
                reason="error_or_constraint_preserved",
                threshold_chars=self.budget_manager.threshold_chars,
            )

        if not self.budget_manager.should_compact(current_context_chars, before_chars):
            return CompactDecision(
                content=content,
                before_chars=before_chars,
                after_chars=before_chars,
                reason="within_budget",
                threshold_chars=self.budget_manager.threshold_chars,
            )

        if before_chars <= self.max_chars:
            return CompactDecision(
                content=content,
                before_chars=before_chars,
                after_chars=before_chars,
                reason="small_result",
                threshold_chars=self.budget_manager.threshold_chars,
            )

        compacted = self._head_tail_compact(content)
        return CompactDecision(
            content=compacted,
            compacted=True,
            before_chars=before_chars,
            after_chars=len(compacted),
            strategy=f"head_tail_{int(self.head_ratio * 100)}_{100 - int(self.head_ratio * 100)}",
            reason="projected_context_exceeds_budget",
            threshold_chars=self.budget_manager.threshold_chars,
        )

    def _head_tail_compact(self, text: str) -> str:
        head_chars = max(1, int(self.max_chars * self.head_ratio))
        tail_chars = max(1, self.max_chars - head_chars)
        omitted = max(0, len(text) - head_chars - tail_chars)
        return (
            f"{text[:head_chars].rstrip()}\n"
            f"[compact] omitted {omitted} chars from middle of tool result; "
            f"preserved head/tail {int(self.head_ratio * 100)}/{100 - int(self.head_ratio * 100)}.\n"
            f"{text[-tail_chars:].lstrip()}"
        )

    def _looks_like_error_content(self, content: str) -> bool:
        text = (content or "").lower()
        return any(marker in text for marker in ERROR_MARKERS)
