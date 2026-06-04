"""Domain adapter contracts.

Adapters describe how a research domain changes prompts, tools, evidence
rules, and output shape. They deliberately return plain dict profiles so the
existing runner and prompt builder can stay decoupled from adapter internals.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class DomainAdapterMetadata:
    """Small public descriptor for a domain adapter."""

    name: str
    display_name: str
    description: str = ""
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdapterMatchResult:
    """Adapter match score for auto domain selection."""

    name: str
    score: float
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class DomainAdapter:
    """Base interface for pluggable domain behavior."""

    metadata: DomainAdapterMetadata

    @property
    def name(self) -> str:
        return self.metadata.name

    def match(self, query: str) -> AdapterMatchResult:
        """Return how strongly this adapter matches the user query."""
        text = query.lower()
        keywords = [item.lower() for item in self.metadata.keywords if item]
        if not text or not keywords:
            return AdapterMatchResult(self.name, 0.0)

        hits = [keyword for keyword in keywords if keyword in text]
        score = min(1.0, len(hits) / max(3, len(keywords) * 0.25))
        reason = ", ".join(hits[:5])
        return AdapterMatchResult(self.name, score, reason=reason, metadata={"hits": hits})

    def parse_constraints(self, query: str) -> dict[str, Any]:
        """Extract domain-specific constraints from a query.

        Built-in adapters keep this lightweight for now. Future declarative
        adapters can delegate this step to a policy-backed generator.
        """
        return {}

    def build_profile(self, query: str = "", config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build a plain domain profile dict consumed by the core pipeline."""
        raise NotImplementedError

    def prompt_sections(self, constraints: dict[str, Any] | None = None) -> list[str]:
        return list(self.build_profile().get("prompt_sections", []) or [])

    def evidence_rules(self, constraints: dict[str, Any] | None = None) -> list[str]:
        return list(self.build_profile().get("evidence_checklist", []) or [])

    def output_sections(self, constraints: dict[str, Any] | None = None) -> list[str]:
        return list(self.build_profile().get("output_sections", []) or [])


class StaticProfileAdapter(DomainAdapter):
    """Adapter backed by static profile data."""

    def __init__(
        self,
        metadata: DomainAdapterMetadata,
        profile: dict[str, Any],
    ) -> None:
        self.metadata = metadata
        self._profile = dict(profile)

    def build_profile(self, query: str = "", config: dict[str, Any] | None = None) -> dict[str, Any]:
        profile = deepcopy(self._profile)
        profile.setdefault("adapter", self.name)
        profile.setdefault("display_name", self.metadata.display_name)
        profile.setdefault("description", self.metadata.description)
        return profile


def dedupe_preserving_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique
