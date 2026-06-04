"""No-op tracing decorators kept for backward-compatible call sites.

The project uses the local JSONL TraceRecorder for observability. These helpers
intentionally do not integrate with external tracing services.
"""
from __future__ import annotations

from contextlib import contextmanager
import functools
import inspect
from typing import Any, Callable


def is_tracing_enabled() -> bool:
    """External tracing is disabled; local JSONL tracing is handled elsewhere."""
    return False


def maybe_wrap_openai_client(client: Any) -> Any:
    """Return the raw OpenAI-compatible client without external wrapping."""
    return client


def traceable(
    run_type: str = "chain",
    name: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Callable:
    """Return a no-op decorator.

    Parameters are accepted to keep existing decorators stable.
    """

    def decorator(func: Callable) -> Callable:
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await func(*args, **kwargs)

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return sync_wrapper

    return decorator


@contextmanager
def trace_block(
    name: str,
    run_type: str = "chain",
    inputs: dict[str, Any] | None = None,
    tags: list[str] | None = None,
):
    """No-op context manager with an add_output-compatible object."""

    class _DummyRun:
        def add_output(self, outputs: dict[str, Any]) -> None:
            return None

    yield _DummyRun()


def trace_agent(name: str | None = None, tags: list[str] | None = None):
    return traceable(run_type="chain", name=name, tags=tags)


def trace_tool(name: str | None = None, tags: list[str] | None = None):
    return traceable(run_type="tool", name=name, tags=tags)


def trace_chain(name: str | None = None, tags: list[str] | None = None):
    return traceable(run_type="chain", name=name, tags=tags)


def trace_retriever(name: str | None = None, tags: list[str] | None = None):
    return traceable(run_type="retriever", name=name, tags=tags)
