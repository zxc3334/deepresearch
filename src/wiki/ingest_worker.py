"""LLM-backed structured wiki ingest.

Raw reports are saved synchronously by the orchestrator. This worker compiles
one raw report into entity pages using the same core workflow as the local
``wiki-ingest`` skill:

- one knowledge entity per page
- create or update, never overwrite existing pages blindly
- maintain Obsidian-style cross references
- update index/log through ``WikiStore``
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any

from .store import WikiStore
from ..utils.output_language import normalize_output_language, output_language_instruction


@dataclass(frozen=True)
class WikiEntity:
    title: str
    category: str
    definition: str
    key_points: list[str] = field(default_factory=list)
    details: str = ""
    context: str = ""
    related_pages: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    confidence: float = 0.5


class WikiIngestWorker:
    """Compile a raw report into structured wiki pages with an LLM extractor."""

    ALLOWED_CATEGORIES = (
        "concepts",
        "products",
        "patterns",
        "comparisons",
        "sensors",
        "methods",
        "analyses",
        "projects",
        "notes",
    )

    def __init__(
        self,
        store: WikiStore,
        policy: Any,
        max_entities: int = 8,
        max_source_chars: int = 18000,
        max_repair_chars: int = 12000,
        output_language: str = "zh-CN",
    ) -> None:
        if policy is None:
            raise ValueError("WikiIngestWorker requires an LLM policy.")
        self.store = store
        self.policy = policy
        self.max_entities = max_entities
        self.max_source_chars = max_source_chars
        self.max_repair_chars = max_repair_chars
        self.output_language = normalize_output_language(output_language)

    def ingest_raw_report(
        self,
        raw_path: str,
        title: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata = dict(metadata or {})
        entities = self.extract_entities(title=title, content=content, metadata=metadata)
        created_or_updated: list[str] = []

        for entity in entities[: self.max_entities]:
            page = self._render_entity_page(entity, raw_path=raw_path, source_metadata=metadata)
            path = self.store.save_page(
                category=entity.category,
                title=entity.title,
                content=page,
                metadata={
                    "source_raw_path": raw_path,
                    "source_query": metadata.get("query", title),
                    "evidence_level": metadata.get("evidence_level", ""),
                    "source_tier": metadata.get("source_tier", ""),
                    "ingest_extractor": "llm",
                    "ingest_confidence": entity.confidence,
                },
                status="draft",
            )
            created_or_updated.append(path)

        self.store.append_log(
            source_title=title,
            source_path=raw_path,
            new_or_updated_pages=created_or_updated,
            details={
                "extractor": "llm",
                "entity_count": len(created_or_updated),
                "query": metadata.get("query", title),
            },
        )
        return {
            "raw_path": raw_path,
            "entity_count": len(created_or_updated),
            "pages": created_or_updated,
            "extractor": "llm",
        }

    def extract_entities(
        self,
        title: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[WikiEntity]:
        prompt = self._build_extraction_prompt(
            title=title,
            content=self._trim_source(content),
            metadata=metadata or {},
        )
        response = self._call_policy_without_tools([
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": prompt},
        ])
        raw_content = response.get("content", "") if isinstance(response, dict) else str(response)
        payload = self._parse_or_repair_payload(raw_content)
        raw_entities = payload.get("entities", [])
        if not isinstance(raw_entities, list):
            raise ValueError("wiki ingest LLM returned invalid JSON: entities must be a list")
        entities = [self._entity_from_mapping(item) for item in raw_entities if isinstance(item, dict)]
        return self._deduplicate_entities([entity for entity in entities if entity.title and entity.definition])

    def _call_policy_without_tools(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        old_tools = getattr(self.policy, "tools", None)
        try:
            if hasattr(self.policy, "tools"):
                self.policy.tools = None
            response = self.policy(messages)
        finally:
            if hasattr(self.policy, "tools"):
                self.policy.tools = old_tools
        return response if isinstance(response, dict) else {"content": str(response)}

    def _system_prompt(self) -> str:
        language_instruction = output_language_instruction(self.output_language)
        return (
            "You are a knowledge-base compiler. Convert research text into a structured Markdown wiki. "
            "Follow these rules strictly: one wiki page equals one reusable knowledge entity; do not create "
            "pages for trivial mentions; update-oriented content is preferred over raw summarization; do not "
            "invent facts that are not supported by the source text; use concise definitions; use category "
            "names exactly from the allowed list; return JSON only. "
            f"{language_instruction}"
        )

    def _build_extraction_prompt(self, title: str, content: str, metadata: dict[str, Any]) -> str:
        existing_index = self.store.get_index()
        return f"""
Compile the following raw report into structured wiki entities.

Allowed categories:
{", ".join(self.ALLOWED_CATEGORIES)}

Existing wiki index:
{existing_index}

Source metadata:
{json.dumps(metadata, ensure_ascii=False, sort_keys=True)}

Source title:
{title}

Source content:
{content}

Return JSON only with this exact shape:
{{
  "entities": [
    {{
      "title": "Entity name",
      "category": "concepts|products|patterns|comparisons|sensors|methods|analyses|projects|notes",
      "definition": "One-line definition grounded in the source",
      "key_points": ["3-5 short source-grounded points"],
      "details": "Expanded explanation grounded in the source",
      "context": "Where this entity appears and what role it plays in the source",
      "related_pages": ["[[category/page-name]]"],
      "sources": ["Specific source sentence, URL, citation, or raw report reference"],
      "confidence": 0.0
    }}
  ]
}}

Constraints:
- Extract at most {self.max_entities} entities.
- {output_language_instruction(self.output_language)}
- Keep only reusable domain knowledge. Do not create pages for one-off report sections or generic headings.
- Prefer 3-6 high-value entities over many noisy pages.
- Use concise entity titles in the configured output language, e.g. "辐射传输方程法", "尺度错配", "Landsat 8/9 OLI/TIRS".
- Prefer entities that will be referenced by future research.
- If an entity already exists in the index, use the same title/category where possible.
- Related pages must use [[category/page-name]] format.
- Sources must point to concrete evidence in the source text; if no URL exists, cite the raw report.
- Do not output mojibake or garbled text. If the source contains garbled text, ignore that garbled fragment.
""".strip()

    def _entity_from_mapping(self, item: dict[str, Any]) -> WikiEntity:
        category = str(item.get("category") or "notes").strip()
        if category not in self.ALLOWED_CATEGORIES:
            category = "notes"
        confidence = self._float_between_0_and_1(item.get("confidence", 0.5))
        return WikiEntity(
            title=self._clean_title(item.get("title", "")),
            category=category,
            definition=self._clean_text(item.get("definition", "")),
            key_points=self._string_list(item.get("key_points", []), limit=6),
            details=self._clean_text(item.get("details", "")),
            context=self._clean_text(item.get("context", "")),
            related_pages=self._normalize_related_pages(item.get("related_pages", [])),
            sources=self._string_list(item.get("sources", []), limit=8),
            confidence=confidence,
        )

    def _render_entity_page(
        self,
        entity: WikiEntity,
        raw_path: str,
        source_metadata: dict[str, Any],
    ) -> str:
        raw_link = f"[[raws/{Path(raw_path).stem}]]"
        related = self._dedupe_strings([*entity.related_pages, raw_link])
        sources = self._dedupe_strings([
            *entity.sources,
            f"Raw report: {raw_path}",
            f"Query: {source_metadata.get('query', '')}",
        ])
        points = "\n".join(f"- {point}" for point in entity.key_points) or "- No key points extracted."
        related_lines = "\n".join(f"- {page}" for page in related) or "- None"
        source_lines = "\n".join(f"- {source}" for source in sources if source.strip()) or f"- Raw report: {raw_path}"

        return (
            f"> {entity.definition}\n\n"
            f"## Key Points\n\n{points}\n\n"
            f"## Details\n\n{entity.details or entity.definition}\n\n"
            f"## Context\n\n{entity.context or source_metadata.get('query', '')}\n\n"
            f"## Related Pages\n\n{related_lines}\n\n"
            f"## Sources\n\n{source_lines}"
        )

    def _parse_json_payload(self, text: str) -> dict[str, Any]:
        stripped = str(text or "").strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
        if fenced:
            stripped = fenced.group(1)
        if not stripped.startswith("{"):
            start = stripped.find("{")
            end = stripped.rfind("}")
            if start >= 0 and end > start:
                stripped = stripped[start:end + 1]
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"wiki ingest LLM returned non-JSON content: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("wiki ingest LLM returned JSON that is not an object")
        return payload

    def _parse_or_repair_payload(self, raw_content: str) -> dict[str, Any]:
        try:
            return self._parse_json_payload(raw_content)
        except ValueError as first_error:
            repaired = self._repair_json_payload(raw_content, error=str(first_error))
            try:
                return self._parse_json_payload(repaired)
            except ValueError as repair_error:
                raise ValueError(
                    "wiki ingest LLM returned non-JSON content after repair. "
                    f"first_error={first_error}; repair_error={repair_error}"
                ) from repair_error

    def _repair_json_payload(self, raw_content: str, error: str) -> str:
        clipped = str(raw_content or "")[: self.max_repair_chars]
        response = self._call_policy_without_tools([
            {
                "role": "system",
                "content": (
                    "You repair malformed JSON. Return valid JSON only. "
                    "Do not add markdown fences, explanations, or facts not present in the input. "
                    f"{output_language_instruction(self.output_language)}"
                ),
            },
            {
                "role": "user",
                "content": f"""
The following wiki-ingest JSON was invalid.

Parser error:
{error}

Malformed content:
{clipped}

Repair it into this exact JSON object shape:
{{
  "entities": [
    {{
      "title": "Entity name",
      "category": "concepts|products|patterns|comparisons|sensors|methods|analyses|projects|notes",
      "definition": "One-line definition",
      "key_points": ["short point"],
      "details": "details",
      "context": "context",
      "related_pages": ["[[category/page-name]]"],
      "sources": ["source reference"],
      "confidence": 0.5
    }}
  ]
}}

Return JSON only.
""".strip(),
            },
        ])
        return response.get("content", "") if isinstance(response, dict) else str(response)

    def _deduplicate_entities(self, entities: list[WikiEntity]) -> list[WikiEntity]:
        seen: set[tuple[str, str]] = set()
        unique: list[WikiEntity] = []
        for entity in entities:
            key = (entity.category, entity.title.lower())
            if key in seen:
                continue
            seen.add(key)
            unique.append(entity)
        return unique

    def _normalize_related_pages(self, value: Any) -> list[str]:
        pages = self._string_list(value, limit=12)
        normalized: list[str] = []
        for page in pages:
            clean = page.strip()
            if not clean:
                continue
            if clean.startswith("[[") and clean.endswith("]]"):
                normalized.append(clean)
                continue
            normalized.append(f"[[notes/{self._slug(clean)}]]")
        return self._dedupe_strings(normalized)

    def _trim_source(self, content: str) -> str:
        text = str(content or "")
        if len(text) <= self.max_source_chars:
            return text
        head_chars = int(self.max_source_chars * 0.7)
        tail_chars = self.max_source_chars - head_chars
        return (
            text[:head_chars]
            + "\n\n[WIKI_INGEST_SOURCE_TRUNCATED]\n\n"
            + text[-tail_chars:]
        )

    def _string_list(self, value: Any, limit: int) -> list[str]:
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, list):
            items = value
        else:
            items = []
        return self._dedupe_strings([self._clean_text(item) for item in items if self._clean_text(item)])[:limit]

    def _dedupe_strings(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(value)
        return result

    def _clean_title(self, value: Any) -> str:
        title = re.sub(r"\s+", " ", str(value or "")).strip()[:100]
        return title.replace("/", "-").replace("\\", "-")

    def _clean_text(self, value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    def _float_between_0_and_1(self, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, number))

    def _slug(self, text: str) -> str:
        return re.sub(r"[^\w\u4e00-\u9fff-]+", "-", text.lower(), flags=re.UNICODE).strip("-")[:64] or "page"
