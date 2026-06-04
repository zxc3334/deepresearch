"""Declarative YAML-backed domain adapters."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from src.domain_adapters.base import DomainAdapterMetadata, StaticProfileAdapter


ALLOWED_DECLARATIVE_KEYS = {
    "name",
    "display_name",
    "description",
    "extends",
    "keywords",
    "exposed_tools",
    "recommended_tools",
    "prompt_sections",
    "preferred_official_domains",
    "evidence_checklist",
    "output_sections",
}

IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


class DeclarativeAdapterError(ValueError):
    """Raised when a declarative adapter file is invalid."""


class DeclarativeDomainAdapter(StaticProfileAdapter):
    """Domain adapter loaded from a safe declarative YAML file."""

    @classmethod
    def from_file(cls, path: str | Path) -> "DeclarativeDomainAdapter":
        source_path = Path(path)
        raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise DeclarativeAdapterError(f"Adapter file must contain a mapping: {source_path}")

        unknown = set(raw) - ALLOWED_DECLARATIVE_KEYS
        if unknown:
            raise DeclarativeAdapterError(
                f"Adapter file contains unsupported fields {sorted(unknown)}: {source_path}"
            )

        name = _required_string(raw, "name", source_path)
        if not IDENTIFIER_RE.match(name):
            raise DeclarativeAdapterError(
                f"Adapter name must match {IDENTIFIER_RE.pattern}: {source_path}"
            )

        display_name = _optional_string(raw, "display_name", name)
        description = _optional_string(raw, "description", "")
        keywords = tuple(_string_list(raw, "keywords", source_path))

        profile: dict[str, Any] = {
            "extends": _optional_string(raw, "extends", "general"),
            "exposed_tools": _string_list(raw, "exposed_tools", source_path),
            "recommended_tools": _string_list(raw, "recommended_tools", source_path),
            "prompt_sections": _string_list(raw, "prompt_sections", source_path),
            "preferred_official_domains": _string_list(raw, "preferred_official_domains", source_path),
            "evidence_checklist": _string_list(raw, "evidence_checklist", source_path),
            "output_sections": _string_list(raw, "output_sections", source_path),
            "source_path": str(source_path),
        }
        return cls(
            metadata=DomainAdapterMetadata(
                name=name,
                display_name=display_name,
                description=description,
                keywords=keywords,
            ),
            profile=profile,
        )


def load_declarative_adapters(directory: str | Path) -> list[DeclarativeDomainAdapter]:
    adapter_dir = Path(directory)
    if not adapter_dir.exists():
        return []
    if not adapter_dir.is_dir():
        raise DeclarativeAdapterError(f"Adapter path is not a directory: {adapter_dir}")

    adapters: list[DeclarativeDomainAdapter] = []
    for path in sorted(adapter_dir.glob("*.yaml")) + sorted(adapter_dir.glob("*.yml")):
        adapters.append(DeclarativeDomainAdapter.from_file(path))
    return adapters


def _required_string(raw: dict[str, Any], key: str, source_path: Path) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DeclarativeAdapterError(f"Adapter field '{key}' must be a non-empty string: {source_path}")
    return value.strip()


def _optional_string(raw: dict[str, Any], key: str, default: str) -> str:
    value = raw.get(key, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise DeclarativeAdapterError(f"Adapter field '{key}' must be a string")
    return value.strip()


def _string_list(raw: dict[str, Any], key: str, source_path: Path) -> list[str]:
    value = raw.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise DeclarativeAdapterError(f"Adapter field '{key}' must be a list: {source_path}")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise DeclarativeAdapterError(f"Adapter field '{key}' must contain only non-empty strings: {source_path}")
        result.append(item.strip())
    return result
