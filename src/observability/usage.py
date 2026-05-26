"""Provider-agnostic LLM usage normalization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StepUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens

    @property
    def cache_hit_rate(self) -> float:
        denominator = self.input_tokens + self.cache_read_tokens
        if denominator <= 0:
            return 0.0
        return round(self.cache_read_tokens / denominator, 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "total_tokens": self.total_tokens,
            "cache_hit_rate": self.cache_hit_rate,
        }


def normalize_usage(usage: Any) -> dict[str, Any]:
    """Normalize usage from OpenAI-compatible, Anthropic, or AI SDK-like shapes."""
    if not usage:
        return StepUsage().to_dict()

    input_tokens = _first_int(
        usage,
        "inputTokens",
        "input_tokens",
        "prompt_tokens",
    )
    output_tokens = _first_int(
        usage,
        "outputTokens",
        "output_tokens",
        "completion_tokens",
    )
    cache_read, cache_read_source = _first_int_with_source(
        usage,
        "cachedInputTokens",
        "cached_input_tokens",
        "cache_read_input_tokens",
        ("providerMetadata", "openai", "cachedTokens"),
        ("provider_metadata", "openai", "cached_tokens"),
        ("prompt_tokens_details", "cached_tokens"),
    )
    cache_write = _first_int(
        usage,
        "cacheCreationInputTokens",
        "cache_creation_input_tokens",
        ("providerMetadata", "anthropic", "cacheCreationInputTokens"),
        ("provider_metadata", "anthropic", "cache_creation_input_tokens"),
    )

    # OpenAI and AI SDK-style input totals include cached tokens. Anthropic
    # native fields report cache read/write separately, so do not subtract
    # `cache_read_input_tokens` from native `input_tokens`.
    subtract_cache_read = cache_read_source in {
        "cachedInputTokens",
        "cached_input_tokens",
        ("providerMetadata", "openai", "cachedTokens"),
        ("provider_metadata", "openai", "cached_tokens"),
        ("prompt_tokens_details", "cached_tokens"),
    }
    if subtract_cache_read and cache_read and input_tokens >= cache_read:
        input_tokens -= cache_read

    return StepUsage(
        input_tokens=max(0, input_tokens),
        output_tokens=max(0, output_tokens),
        cache_read_tokens=max(0, cache_read),
        cache_write_tokens=max(0, cache_write),
    ).to_dict()


def _first_int(obj: Any, *paths: str | tuple[str, ...]) -> int:
    value, _source = _first_int_with_source(obj, *paths)
    return value


def _first_int_with_source(obj: Any, *paths: str | tuple[str, ...]) -> tuple[int, str | tuple[str, ...] | None]:
    for path in paths:
        value = _get_path(obj, path)
        if value is None:
            continue
        try:
            return int(value), path
        except (TypeError, ValueError):
            continue
    return 0, None


def _get_path(obj: Any, path: str | tuple[str, ...]) -> Any:
    keys = (path,) if isinstance(path, str) else path
    current = obj
    for key in keys:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
    return current
