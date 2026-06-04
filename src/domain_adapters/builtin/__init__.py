"""Built-in domain adapters."""
from __future__ import annotations

from src.domain_adapters.builtin.general import GeneralDomainAdapter
from src.domain_adapters.builtin.geo_remote_sensing import GeoRemoteSensingAdapter

__all__ = ["GeneralDomainAdapter", "GeoRemoteSensingAdapter"]
