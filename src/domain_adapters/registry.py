"""Registry for built-in and future user-defined domain adapters."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Iterable

from src.domain_adapters.base import AdapterMatchResult, DomainAdapter
from src.domain_adapters.builtin import GeneralDomainAdapter, GeoRemoteSensingAdapter
from src.domain_adapters.declarative import load_declarative_adapters


class AdapterRegistry:
    """In-process registry for domain adapters.

    The registry is intentionally small: adapters are extension points, while
    the resolved profile remains plain data consumed by the existing pipeline.
    """

    _adapters: dict[str, DomainAdapter] = {}
    _builtins_loaded = False
    _builtin_adapter_names = {"general", "geo_remote_sensing"}
    _user_adapter_names: set[str] = set()

    @classmethod
    def ensure_builtins(cls) -> None:
        if cls._builtins_loaded:
            return
        cls.register(GeneralDomainAdapter())
        cls.register(GeoRemoteSensingAdapter())
        cls._builtins_loaded = True

    @classmethod
    def register(cls, adapter: DomainAdapter, *, replace: bool = False) -> None:
        if adapter.name in cls._adapters and not replace:
            raise ValueError(f"Domain adapter already registered: {adapter.name}")
        cls._adapters[adapter.name] = adapter

    @classmethod
    def get(cls, name: str) -> DomainAdapter:
        cls.ensure_builtins()
        try:
            return cls._adapters[name]
        except KeyError as exc:
            raise ValueError(f"Unknown domain adapter: {name}") from exc

    @classmethod
    def list_adapters(cls) -> list[DomainAdapter]:
        cls.ensure_builtins()
        return list(cls._adapters.values())

    @classmethod
    def load_user_adapters(cls, directory: str | Path | None) -> list[DomainAdapter]:
        cls.ensure_builtins()
        if not directory:
            return []

        adapter_dir = Path(directory)
        adapters = load_declarative_adapters(adapter_dir)
        for adapter in adapters:
            if adapter.name in cls._builtin_adapter_names:
                raise ValueError(f"User adapter cannot override built-in adapter: {adapter.name}")
            cls.register(adapter, replace=adapter.name in cls._user_adapter_names)
            cls._user_adapter_names.add(adapter.name)
        return adapters

    @classmethod
    def default_profiles(cls) -> dict[str, dict]:
        cls.ensure_builtins()
        return {name: deepcopy(adapter.build_profile()) for name, adapter in cls._adapters.items()}

    @classmethod
    def match(cls, query: str, candidates: Iterable[str] | None = None) -> list[AdapterMatchResult]:
        cls.ensure_builtins()
        names = list(candidates or cls._adapters)
        results = [cls.get(name).match(query) for name in names]
        return sorted(results, key=lambda item: item.score, reverse=True)

    @classmethod
    def resolve(cls, query: str = "", mode: str = "general") -> DomainAdapter:
        cls.ensure_builtins()
        if not mode or mode == "general":
            return cls.get("general")
        if mode != "auto":
            return cls.get(mode)

        matches = cls.match(query)
        best = matches[0] if matches else AdapterMatchResult("general", 0.0)
        if best.name != "general" and best.score >= 0.35:
            return cls.get(best.name)
        return cls.get("general")
