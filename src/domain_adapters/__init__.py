"""Pluggable domain adapter framework."""
from __future__ import annotations

from src.domain_adapters.base import AdapterMatchResult, DomainAdapter, DomainAdapterMetadata, StaticProfileAdapter
from src.domain_adapters.declarative import DeclarativeAdapterError, DeclarativeDomainAdapter
from src.domain_adapters.registry import AdapterRegistry

__all__ = [
    "AdapterMatchResult",
    "AdapterRegistry",
    "DeclarativeAdapterError",
    "DeclarativeDomainAdapter",
    "DomainAdapter",
    "DomainAdapterMetadata",
    "StaticProfileAdapter",
]
