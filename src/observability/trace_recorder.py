"""Local JSONL tracing for reproducible agent observability."""
from __future__ import annotations

import json
import os
import time
from typing import Any


class TraceRecorder:
    """Append-only JSONL event recorder.

    The recorder is intentionally small and local-first. It stores
    project-specific evidence and compact events in a reproducible file under
    outputs/.
    """

    def __init__(self, path: str, run_id: str | None = None) -> None:
        self.path = path
        self.run_id = run_id or str(int(time.time() * 1000))
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "a", encoding="utf-8"):
            pass

    def record(self, event: str, **payload: Any) -> None:
        row = {
            "ts": time.time(),
            "run_id": self.run_id,
            "event": event,
            **self._safe_payload(payload),
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def _safe_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            key: self._sanitize(value)
            for key, value in payload.items()
            if not self._looks_sensitive(str(key))
        }

    def _sanitize(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self._sanitize(v) for k, v in value.items() if not self._looks_sensitive(str(k))}
        if isinstance(value, list):
            return [self._sanitize(item) for item in value[:50]]
        if isinstance(value, tuple):
            return [self._sanitize(item) for item in value[:50]]
        if isinstance(value, str):
            return value[:4000]
        return value

    def _looks_sensitive(self, key: str) -> bool:
        lowered = key.lower()
        if any(marker in lowered for marker in ("api_key", "apikey", "authorization", "secret", "password")):
            return True
        return lowered in {"token", "access_token", "refresh_token", "id_token", "bearer_token"}
