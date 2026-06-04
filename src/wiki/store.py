"""File-backed Markdown wiki store.

The wiki is deliberately simple in Phase 8: it stores inspectable Markdown
drafts and performs lightweight lexical retrieval. It is not a replacement for
the evidence gate; only gated reports should be written here.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import re
from typing import Any


@dataclass(frozen=True)
class WikiPage:
    path: str
    title: str
    content: str
    metadata: dict[str, Any]
    score: float = 0.0


class WikiStore:
    """User-scoped Markdown wiki with lightweight retrieval."""

    CATEGORIES = (
        "raws",
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

    def __init__(self, root_dir: str = "data/wiki", user_id: str = "default") -> None:
        self.root_dir = Path(root_dir)
        self.user_id = self._safe_segment(user_id or "default")
        self.user_dir = self.root_dir / "users" / self.user_id
        self._ensure_layout()

    def _ensure_layout(self) -> None:
        self.user_dir.mkdir(parents=True, exist_ok=True)
        for category in self.CATEGORIES:
            (self.user_dir / category).mkdir(parents=True, exist_ok=True)
        index = self.user_dir / "index.md"
        if not index.exists():
            index.write_text("# Wiki Index\n\nNo pages yet.\n", encoding="utf-8")
        log = self.user_dir / "log.md"
        if not log.exists():
            log.write_text("# Wiki Ingest Log\n\n", encoding="utf-8")

    def save_raw(
        self,
        title: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        *,
        status: str = "draft",
    ) -> str:
        """Save a raw research report as a draft Markdown wiki page."""
        metadata = dict(metadata or {})
        metadata["status"] = status
        metadata.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
        metadata.setdefault("category", "raws")
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{self._slug(title)}.md"
        path = self.user_dir / "raws" / filename
        page = self._render_page(title=title, content=content, metadata=metadata)
        path.write_text(page, encoding="utf-8")
        self._update_index()
        return str(path)

    def find_similar_raw(
        self,
        title: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        *,
        threshold: float = 0.72,
    ) -> WikiPage | None:
        """Find an existing raw report that is likely the same ingest target."""
        metadata = dict(metadata or {})
        query = str(metadata.get("query") or title or "")
        query_terms = self._terms(query)
        content_terms = self._terms(str(content or "")[:4000])
        best: WikiPage | None = None
        best_score = 0.0
        for path in sorted((self.user_dir / "raws").glob("*.md")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            page_metadata, body = self._parse_page(text)
            page_title = str(page_metadata.get("title") or self._title_from_body(body) or path.stem)
            page_query = str(page_metadata.get("query") or page_metadata.get("source_query") or page_title)
            if self._normalize_for_match(page_query) == self._normalize_for_match(query):
                score = 1.0
            else:
                page_query_terms = self._terms(page_query)
                page_content_terms = self._terms(body[:4000])
                query_score = self._jaccard(query_terms, page_query_terms)
                content_score = self._jaccard(content_terms, page_content_terms)
                score = max(query_score, 0.65 * query_score + 0.35 * content_score)
            if score > best_score:
                best_score = score
                best = WikiPage(
                    path=str(path),
                    title=page_title,
                    content=body,
                    metadata=page_metadata,
                    score=score,
                )
        return best if best is not None and best_score >= threshold else None

    def save_page(
        self,
        category: str,
        title: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        *,
        status: str = "draft",
    ) -> str:
        """Save or update one structured wiki page in a known category."""
        category = category if category in self.CATEGORIES else "notes"
        metadata = dict(metadata or {})
        metadata["status"] = status
        metadata["category"] = category
        metadata.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
        filename = f"{self._slug(title).lower()}.md"
        path = self.user_dir / category / filename
        if path.exists():
            existing_metadata, existing_body = self._parse_page(path.read_text(encoding="utf-8", errors="ignore"))
            metadata = {**existing_metadata, **metadata}
            content = self._merge_page_content(existing_body, content)
        page = self._render_page(title=title, content=content, metadata=metadata)
        path.write_text(page, encoding="utf-8")
        self._update_index()
        return str(path)

    def get_index(self) -> str:
        return (self.user_dir / "index.md").read_text(encoding="utf-8")

    def append_log(
        self,
        source_title: str,
        source_path: str,
        new_or_updated_pages: list[str],
        details: dict[str, Any] | None = None,
    ) -> None:
        """Append an ingest operation entry following the wiki-ingest workflow."""
        details = dict(details or {})
        rel_pages = []
        for page in new_or_updated_pages:
            try:
                rel_pages.append(Path(page).relative_to(self.user_dir).as_posix())
            except ValueError:
                rel_pages.append(str(page))
        lines = [
            f"## {datetime.now().date().isoformat()}: Ingest {source_title}",
            "",
            f"**Source**: {source_path}",
            f"**New or updated pages**: {', '.join(rel_pages) if rel_pages else 'none'}",
            f"**Details**: {json.dumps(details, ensure_ascii=False, sort_keys=True)}",
            "",
        ]
        with (self.user_dir / "log.md").open("a", encoding="utf-8") as f:
            f.write("\n".join(lines))
        self._update_index()

    def has_pages(self) -> bool:
        return bool(self._page_paths())

    def get_context(self, query: str, max_chars: int = 2000, top_k: int = 5) -> str:
        pages = self.search(query=query, top_k=top_k)
        if not pages:
            return ""
        parts: list[str] = []
        used = 0
        for page in pages:
            excerpt = self._excerpt(page.content, query, max_chars=max(200, max_chars // max(1, top_k)))
            block = (
                f"- [{page.title}] {page.path}\n"
                f"  status={page.metadata.get('status', '')}; "
                f"evidence_level={page.metadata.get('evidence_level', '')}; "
                f"source_tier={page.metadata.get('source_tier', '')}; score={page.score:.2f}\n"
                f"  {excerpt}"
            )
            if used + len(block) > max_chars:
                break
            parts.append(block)
            used += len(block)
        return "## Wiki Context\n" + "\n".join(parts) if parts else ""

    def search(self, query: str, top_k: int = 5) -> list[WikiPage]:
        terms = self._terms(query)
        if not terms:
            return []
        scored: list[WikiPage] = []
        for path in self._page_paths():
            content = path.read_text(encoding="utf-8", errors="ignore")
            metadata, body = self._parse_page(content)
            title = str(metadata.get("title") or self._title_from_body(body) or path.stem)
            haystack = f"{title}\n{body}".lower()
            hits = sum(1 for term in terms if term in haystack)
            if hits <= 0:
                continue
            score = hits / max(1, len(terms))
            scored.append(WikiPage(
                path=str(path),
                title=title,
                content=body,
                metadata=metadata,
                score=score,
            ))
        scored.sort(key=lambda page: page.score, reverse=True)
        return scored[:top_k]

    def _page_paths(self) -> list[Path]:
        paths: list[Path] = []
        for category in self.CATEGORIES:
            paths.extend((self.user_dir / category).glob("*.md"))
        return sorted(paths)

    def _update_index(self) -> None:
        lines = ["# Wiki Index", "", "- [Operation log](log.md)", ""]
        for category in self.CATEGORIES:
            pages = sorted((self.user_dir / category).glob("*.md"))
            lines.append(f"## {category}")
            if not pages:
                lines.append("")
                continue
            for page in pages:
                metadata, body = self._parse_page(page.read_text(encoding="utf-8", errors="ignore"))
                title = metadata.get("title") or self._title_from_body(body) or page.stem
                rel = page.relative_to(self.user_dir).as_posix()
                lines.append(f"- [{title}]({rel})")
            lines.append("")
        (self.user_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")

    def _render_page(self, title: str, content: str, metadata: dict[str, Any]) -> str:
        full_metadata = {"title": title, **metadata}
        return (
            "---\n"
            f"{json.dumps(full_metadata, ensure_ascii=False, sort_keys=True)}\n"
            "---\n\n"
            f"# {title}\n\n"
            f"{content.strip()}\n"
        )

    def _parse_page(self, text: str) -> tuple[dict[str, Any], str]:
        if text.startswith("---\n"):
            end = text.find("\n---", 4)
            if end != -1:
                raw = text[4:end].strip()
                try:
                    metadata = json.loads(raw)
                except json.JSONDecodeError:
                    metadata = {}
                body = text[end + len("\n---"):].strip()
                return metadata, body
        return {}, text

    def _excerpt(self, content: str, query: str, max_chars: int) -> str:
        normalized = re.sub(r"\s+", " ", content).strip()
        if len(normalized) <= max_chars:
            return normalized
        terms = self._terms(query)
        lower = normalized.lower()
        first_hit = min((lower.find(term) for term in terms if lower.find(term) >= 0), default=0)
        start = max(0, first_hit - max_chars // 3)
        return normalized[start:start + max_chars].strip()

    def _terms(self, text: str) -> set[str]:
        return {
            term.lower()
            for term in re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z0-9][a-zA-Z0-9_\-]{2,}", text)
        }

    def _jaccard(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / max(1, len(left | right))

    def _normalize_for_match(self, text: str) -> str:
        return re.sub(r"\s+", "", str(text or "").lower())

    def _title_from_body(self, body: str) -> str:
        for line in body.splitlines():
            if line.startswith("#"):
                return line.lstrip("#").strip()
        return ""

    def _merge_page_content(self, existing: str, new_content: str) -> str:
        existing = existing.strip()
        new_content = new_content.strip()
        if not existing:
            return new_content
        if new_content in existing:
            return existing
        return f"{existing}\n\n## Update {datetime.now().isoformat(timespec='seconds')}\n\n{new_content}"

    def _slug(self, text: str) -> str:
        slug = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", text, flags=re.UNICODE).strip("_")
        return slug[:48] or "page"

    def _safe_segment(self, text: str) -> str:
        return re.sub(r"[^\w.-]+", "_", text, flags=re.UNICODE)[:80] or "default"
