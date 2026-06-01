"""Progress and user-interaction primitives for orchestrator runs."""
from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable


ProgressCallback = Callable[[dict[str, Any]], Any]
ContextModifier = Callable[[dict[str, Any], list["UserInstruction"]], dict[str, Any] | None]


@dataclass(frozen=True)
class UserInstruction:
    """User direction adjustment captured during a run."""

    text: str
    task_id: str | None = None
    created_at: float = field(default_factory=time.time)
    source: str = "user"


class ProgressBus:
    """Small callback-backed event bus for progress/SSE adapters."""

    def __init__(self, callback: ProgressCallback | None = None) -> None:
        self.callback = callback
        self.events: list[dict[str, Any]] = []
        self.errors: list[str] = []

    async def publish(self, event_type: str, **payload: Any) -> None:
        event = {
            "event_type": event_type,
            "timestamp": time.time(),
            **payload,
        }
        self.events.append(event)
        if self.callback is None:
            return
        try:
            result = self.callback(event)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")


class InteractiveBus:
    """In-memory user instruction buffer.

    Current behavior is intentionally non-interrupting: instructions are applied
    to the next task context, not to a currently running tool call.
    """

    def __init__(self) -> None:
        self._pending: list[UserInstruction] = []

    def add_instruction(self, text: str, task_id: str | None = None, source: str = "user") -> UserInstruction:
        instruction = UserInstruction(text=text, task_id=task_id, source=source)
        self._pending.append(instruction)
        return instruction

    def drain_instructions(self, task_id: str | None = None) -> list[UserInstruction]:
        if task_id is None:
            drained = list(self._pending)
            self._pending.clear()
            return drained

        matched: list[UserInstruction] = []
        remaining: list[UserInstruction] = []
        for instruction in self._pending:
            if instruction.task_id in (None, task_id):
                matched.append(instruction)
            else:
                remaining.append(instruction)
        self._pending = remaining
        return matched

    def has_pending(self) -> bool:
        return bool(self._pending)


def default_context_modifier(context: dict[str, Any], instructions: list[UserInstruction]) -> dict[str, Any]:
    """Append user instructions to the next task context."""
    if not instructions:
        return context
    updated = dict(context)
    existing = str(updated.get("user_instructions", "") or "").strip()
    new_text = "\n".join(f"- {instruction.text}" for instruction in instructions)
    updated["user_instructions"] = f"{existing}\n{new_text}".strip() if existing else new_text
    updated["user_instruction_count"] = len(instructions)
    return updated
