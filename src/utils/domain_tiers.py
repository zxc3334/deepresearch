"""Shared domain-to-source-tier classification.

This module is the single source of truth for source provenance quality.
Search ranking, evidence classification, and trace/report rendering should use
the same rules so "NASA" and "CSDN" do not accidentally receive equal weight.
"""
from __future__ import annotations

from urllib.parse import urlparse

from ..orchestrator.schemas import SourceTier

__all__ = [
    "DOMAIN_TIER_TABLE",
    "TIER_BASE_SCORE",
    "authority_score",
    "best_tier",
    "classify_url",
    "extract_hostname",
]


DOMAIN_TIER_TABLE: dict[str, SourceTier] = {
    # Official / government / space agency sources.
    "nasa.gov": SourceTier.OFFICIAL,
    "earthdata.nasa.gov": SourceTier.OFFICIAL,
    "modis.gsfc.nasa.gov": SourceTier.OFFICIAL,
    "jpl.nasa.gov": SourceTier.OFFICIAL,
    "usgs.gov": SourceTier.OFFICIAL,
    "lpdaac.usgs.gov": SourceTier.OFFICIAL,
    "esa.int": SourceTier.OFFICIAL,
    "sentinel.esa.int": SourceTier.OFFICIAL,
    "copernicus.eu": SourceTier.OFFICIAL,
    "sentinels.copernicus.eu": SourceTier.OFFICIAL,
    "sentiwiki.copernicus.eu": SourceTier.OFFICIAL,
    "documentation.dataspace.copernicus.eu": SourceTier.OFFICIAL,
    "developers.google.com": SourceTier.OFFICIAL,
    "earthengine.google.com": SourceTier.OFFICIAL,
    "planetarycomputer.microsoft.com": SourceTier.OFFICIAL,
    "microsoft.com": SourceTier.OFFICIAL,
    "noaa.gov": SourceTier.OFFICIAL,
    "ecmwf.int": SourceTier.OFFICIAL,
    "eumetsat.int": SourceTier.OFFICIAL,
    "geoservice.dlr.de": SourceTier.OFFICIAL,

    # Academic sources and publishers.
    "openalex.org": SourceTier.ACADEMIC,
    "arxiv.org": SourceTier.ACADEMIC,
    "doi.org": SourceTier.ACADEMIC,
    "semanticscholar.org": SourceTier.ACADEMIC,
    "scholar.google.com": SourceTier.ACADEMIC,
    "ieee.org": SourceTier.ACADEMIC,
    "ieee.com": SourceTier.ACADEMIC,
    "springer.com": SourceTier.ACADEMIC,
    "link.springer.com": SourceTier.ACADEMIC,
    "sciencedirect.com": SourceTier.ACADEMIC,
    "nature.com": SourceTier.ACADEMIC,
    "wiley.com": SourceTier.ACADEMIC,
    "tandfonline.com": SourceTier.ACADEMIC,
    "mdpi.com": SourceTier.ACADEMIC,
    "pubmed.ncbi.nlm.nih.gov": SourceTier.ACADEMIC,
    "researchgate.net": SourceTier.ACADEMIC,
    "academia.edu": SourceTier.ACADEMIC,

    # Authoritative references.
    "wikipedia.org": SourceTier.AUTHORITATIVE,
    "en.wikipedia.org": SourceTier.AUTHORITATIVE,
    "stackoverflow.com": SourceTier.AUTHORITATIVE,
    "github.com": SourceTier.AUTHORITATIVE,
    "docs.python.org": SourceTier.AUTHORITATIVE,
    "learn.microsoft.com": SourceTier.AUTHORITATIVE,
    "cloud.google.com": SourceTier.AUTHORITATIVE,
    "aws.amazon.com": SourceTier.AUTHORITATIVE,

    # General / user-generated sources.
    "csdn.net": SourceTier.GENERAL,
    "blog.csdn.net": SourceTier.GENERAL,
    "zhihu.com": SourceTier.GENERAL,
    "jianshu.com": SourceTier.GENERAL,
    "baidu.com": SourceTier.GENERAL,
    "baike.baidu.com": SourceTier.GENERAL,
    "juejin.cn": SourceTier.GENERAL,
    "segmentfault.com": SourceTier.GENERAL,
    "cnblogs.com": SourceTier.GENERAL,
    "bilibili.com": SourceTier.GENERAL,
    "medium.com": SourceTier.GENERAL,
    "dev.to": SourceTier.GENERAL,
    "reddit.com": SourceTier.GENERAL,
    "quora.com": SourceTier.GENERAL,
}

TIER_BASE_SCORE: dict[SourceTier, float] = {
    SourceTier.OFFICIAL: 95.0,
    SourceTier.ACADEMIC: 75.0,
    SourceTier.AUTHORITATIVE: 55.0,
    SourceTier.GENERAL: 30.0,
    SourceTier.UNVERIFIED: 10.0,
}

_TLD_TIER_RULES: list[tuple[str, SourceTier]] = [
    (".gov.cn", SourceTier.OFFICIAL),
    (".edu.cn", SourceTier.ACADEMIC),
    (".ac.cn", SourceTier.ACADEMIC),
    (".gov", SourceTier.OFFICIAL),
    (".mil", SourceTier.OFFICIAL),
    (".edu", SourceTier.ACADEMIC),
    (".org", SourceTier.AUTHORITATIVE),
]


def classify_url(url: str) -> SourceTier:
    """Classify a URL by source quality, not by claim truth."""
    hostname = extract_hostname(url)
    if not hostname:
        return SourceTier.UNVERIFIED

    if hostname in DOMAIN_TIER_TABLE:
        return DOMAIN_TIER_TABLE[hostname]

    for domain, tier in DOMAIN_TIER_TABLE.items():
        if hostname == domain or hostname.endswith("." + domain):
            return tier

    for suffix, tier in _TLD_TIER_RULES:
        if hostname.endswith(suffix):
            return tier

    return SourceTier.GENERAL


def authority_score(url: str) -> float:
    """Return a stable 0-100 authority score for search result ranking."""
    tier = classify_url(url)
    score = TIER_BASE_SCORE[tier]
    hostname = extract_hostname(url)
    boosts = {
        "nasa.gov": 5.0,
        "usgs.gov": 5.0,
        "esa.int": 5.0,
        "nature.com": 3.0,
        "ieee.org": 3.0,
    }
    for domain, boost in boosts.items():
        if hostname == domain or hostname.endswith("." + domain):
            return min(100.0, score + boost)
    return score


def best_tier(urls: list[str]) -> SourceTier:
    """Return the highest-quality tier among URLs."""
    rank = {
        SourceTier.OFFICIAL: 4,
        SourceTier.ACADEMIC: 3,
        SourceTier.AUTHORITATIVE: 2,
        SourceTier.GENERAL: 1,
        SourceTier.UNVERIFIED: 0,
    }
    best = SourceTier.UNVERIFIED
    for url in urls:
        tier = classify_url(url)
        if rank[tier] > rank[best]:
            best = tier
    return best


def extract_hostname(url: str) -> str:
    """Extract normalized hostname from a URL."""
    if not url:
        return ""
    try:
        hostname = urlparse(url).hostname or ""
    except Exception:
        return ""
    return hostname.lower().lstrip("www.")
