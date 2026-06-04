"""General-purpose DeepResearch adapter."""
from __future__ import annotations

from src.domain_adapters.base import DomainAdapterMetadata, StaticProfileAdapter


GENERAL_PROFILE = {
    "extends": None,
    "exposed_tools": [
        "wiki_search",
        "web_search",
        "official_source_search",
        "official_doc_fetcher",
        "paper_search",
        "browser",
        "calculator",
    ],
    "recommended_tools": [
        "wiki_search",
        "web_search",
        "paper_search",
        "official_source_search",
        "official_doc_fetcher",
        "browser",
        "calculator",
    ],
    "prompt_sections": [],
    "preferred_official_domains": [],
    "evidence_checklist": [
        "Prefer source-backed claims over unsupported model knowledge.",
        "Use official sources for product specifications and academic sources for methods.",
    ],
}


class GeneralDomainAdapter(StaticProfileAdapter):
    def __init__(self) -> None:
        super().__init__(
            metadata=DomainAdapterMetadata(
                name="general",
                display_name="General DeepResearch",
                description="General-purpose deep research without domain-specific assumptions.",
            ),
            profile=GENERAL_PROFILE,
        )

    def match(self, query: str):
        result = super().match(query)
        return type(result)(self.name, max(result.score, 0.05), reason=result.reason, metadata=result.metadata)
