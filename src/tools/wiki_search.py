"""Search the local Markdown wiki knowledge base."""
from __future__ import annotations

from typing import Any

from ..wiki import WikiStore


class WikiSearchTool:
    """Retrieve previously gated wiki pages for the current user."""

    name = "wiki_search"
    description = (
        "Search the local project wiki for previously saved, evidence-gated research notes. "
        "Use this to reuse prior findings before starting external search. "
        "Input: {'query': str, 'top_k': int(optional)}."
    )

    def __init__(self, wiki_store: WikiStore) -> None:
        self.wiki_store = wiki_store

    def has_pages(self) -> bool:
        return self.wiki_store.has_pages()

    def get_openai_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Wiki search query."},
                        "top_k": {"type": "integer", "description": "Maximum wiki pages to return.", "default": 5},
                    },
                    "required": ["query"],
                },
            },
        }

    async def execute(self, query: str, top_k: int = 5) -> dict[str, Any]:
        pages = self.wiki_store.search(query=query, top_k=top_k)
        results = [
            {
                "title": page.title,
                "path": page.path,
                "snippet": self._snippet(page.content),
                "score": round(page.score, 3),
                "metadata": page.metadata,
                "source_type": "wiki",
                "evidence_level": page.metadata.get("evidence_level", ""),
                "source_tier": page.metadata.get("source_tier", ""),
            }
            for page in pages
        ]
        return {
            "query": query,
            "results": results,
            "total": len(results),
            "source": "wiki_search",
            "source_type": "wiki",
            "evidence_level": self._best_evidence_level(results),
        }

    def _snippet(self, content: str) -> str:
        normalized = " ".join(str(content or "").split())
        return normalized[:500]

    def _best_evidence_level(self, results: list[dict[str, Any]]) -> str:
        priority = {
            "verified": 4,
            "evidence_backed": 3,
            "speculative": 1,
            "rejected": 0,
        }
        best = "speculative"
        for result in results:
            level = str(result.get("evidence_level") or "speculative")
            if priority.get(level, 1) > priority[best]:
                best = level
        return best
