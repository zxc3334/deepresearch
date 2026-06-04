"""Domain profile resolution for tool exposure and prompt specialization."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.domain_adapters import AdapterRegistry


def resolve_domain_profile(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve the active domain profile from config.

    The profile is intentionally plain data so runner, prompt builder, and
    future APIs can consume the same contract without coupling to each other.
    """
    adapter_cfg = config.get("domain_adapter", {})
    user_adapters_dir = adapter_cfg.get("user_adapters_dir")
    AdapterRegistry.load_user_adapters(user_adapters_dir)

    active = _resolve_adapter_name(config, adapter_cfg)
    profiles = AdapterRegistry.default_profiles()
    resolved = _collect_profile(str(active), profiles, seen=set())
    resolved["name"] = str(active)
    return resolved


def _resolve_adapter_name(
    config: dict[str, Any],
    adapter_cfg: dict[str, Any],
) -> str:
    mode = adapter_cfg.get("mode", "general")
    query = str(config.get("_query") or config.get("query") or "")
    return AdapterRegistry.resolve(query=query, mode=str(mode)).name


def _collect_profile(profile_name: str, profiles: dict[str, dict[str, Any]], seen: set[str]) -> dict[str, Any]:
    if profile_name in seen:
        raise ValueError(f"Cyclic domain profile inheritance detected: {profile_name}")
    seen.add(profile_name)
    profile = profiles.get(profile_name)
    if profile is None:
        raise ValueError(f"Unknown domain profile: {profile_name}")

    parent_name = profile.get("extends")
    if parent_name:
        parent = _collect_profile(str(parent_name), profiles, seen)
        merged = _merge_profile(parent, profile)
    else:
        merged = _merge_profile({}, profile)

    for key in (
        "exposed_tools",
        "recommended_tools",
        "prompt_sections",
        "preferred_official_domains",
        "evidence_checklist",
        "output_sections",
    ):
        merged[key] = _dedupe([str(item) for item in merged.get(key, []) or []])
    return merged


def _merge_profile(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in {"include", "exposed_tools"}:
            merged["exposed_tools"] = list(merged.get("exposed_tools", []) or []) + list(value or [])
        elif key in {
            "recommended_tools",
            "prompt_sections",
            "preferred_official_domains",
            "evidence_checklist",
            "output_sections",
        }:
            merged[key] = list(merged.get(key, []) or []) + list(value or [])
        elif key == "exclude":
            excluded = {str(item) for item in value or []}
            merged["exposed_tools"] = [item for item in merged.get("exposed_tools", []) or [] if str(item) not in excluded]
        else:
            merged[key] = deepcopy(value)
    return merged


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique
