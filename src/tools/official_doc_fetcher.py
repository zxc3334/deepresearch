"""Official documentation fetcher for page-grounded official evidence.

This tool turns an official URL into page-grounded evidence snippets.  It is
more constrained than the generic browser tool:

- only HTTP(S) URLs are accepted;
- official documentation domains are allowlisted;
- the output is structured for EvidenceStore and trace visualization.
"""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

import aiohttp

from ..utils.text_cleanup import normalize_extracted_text, normalize_snippet


class OfficialDocFetcherTool:
    """Fetch an official page and extract query-relevant evidence snippets."""

    name = "official_doc_fetcher"
    description = (
        "Fetch and read an official documentation URL, then extract page-grounded evidence snippets. "
        "Use this after official_source_search when you need page-grounded evidence instead of only a search result. "
        "Input: {'url': str, 'query': str(optional), 'max_chars': int(optional), 'max_snippets': int(optional)}."
    )

    default_allowed_domains = [
        "sentinel.esa.int",
        "sentinels.copernicus.eu",
        "sentiwiki.copernicus.eu",
        "documentation.dataspace.copernicus.eu",
        "esa.int",
        "usgs.gov",
        "nasa.gov",
        "modis.gsfc.nasa.gov",
        "lpdaac.usgs.gov",
        "copernicus.eu",
        "developers.google.com",
        "planetarycomputer.microsoft.com",
        "docs.python.org",
        "developer.mozilla.org",
        "docs.github.com",
        "learn.microsoft.com",
        "cloud.google.com",
        "docs.aws.amazon.com",
        "platform.openai.com",
    ]

    def __init__(
        self,
        allowed_domains: list[str] | None = None,
        timeout_seconds: int = 20,
        user_agent: str | None = None,
    ) -> None:
        self.allowed_domains = allowed_domains or self.default_allowed_domains
        self.timeout_seconds = timeout_seconds
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

    def get_openai_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Official documentation URL to fetch.",
                        },
                        "query": {
                            "type": "string",
                            "description": "Claim or keywords to match against the page.",
                        },
                        "max_chars": {
                            "type": "integer",
                            "description": "Maximum cleaned page characters to keep internally.",
                            "default": 12000,
                        },
                        "max_snippets": {
                            "type": "integer",
                            "description": "Maximum evidence snippets to return.",
                            "default": 5,
                        },
                    },
                    "required": ["url"],
                },
            },
        }

    async def execute(
        self,
        url: str,
        query: str = "",
        max_chars: int = 12000,
        max_snippets: int = 5,
    ) -> dict[str, Any]:
        if not self._is_http_url(url):
            return self._error(url, "URL must start with http:// or https://.")
        if not self._is_allowed_official_url(url):
            return self._error(
                url,
                "URL is outside the official documentation allowlist.",
                source_type="rejected_url",
            )

        try:
            html, final_url = await self._fetch(url)
        except Exception as exc:
            return self._error(url, f"{type(exc).__name__}: {exc}")
        if not self._is_allowed_official_url(final_url):
            return self._error(
                final_url,
                "Final redirected URL is outside the official documentation allowlist.",
                source_type="rejected_url",
            )

        title, text = self._extract_text(html)
        text = self._clean_text(text)
        truncated = False
        if len(text) > max_chars:
            text = text[:max_chars]
            truncated = True

        snippets = self._extract_snippets(text, query=query, max_snippets=max_snippets)
        match_count = sum(1 for snippet in snippets if snippet.get("match_score", 0) > 0)
        claim_support = self._verify_claim_support(query=query, text=text, snippets=snippets)
        result = {
            "title": title or self._title_from_url(final_url),
            "url": final_url,
            "snippet": normalize_snippet(snippets[0]["text"] if snippets else text[:600], max_chars=600),
            "snippets": snippets,
            "content_chars": len(text),
            "truncated": truncated,
            "official_domain": urlparse(final_url).netloc.lower(),
            "source_type": "official_doc",
            "claim_support": claim_support,
        }
        return {
            "query": query,
            "url": url,
            "final_url": final_url,
            "results": [result],
            "total": 1,
            "match_count": match_count,
            "claim_support": claim_support,
            "source": "official_doc_fetcher",
            "source_type": "official_doc",
            "evidence_level": "evidence_backed" if claim_support["level"] == "supported" else "speculative",
        }

    async def _fetch(self, url: str) -> tuple[str, str]:
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.get(url, allow_redirects=True) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if "text/html" not in content_type and "xml" not in content_type:
                    raise ValueError(f"Unsupported content-type: {content_type}")
                charset = response.charset or "utf-8"
                return await response.text(encoding=charset, errors="replace"), str(response.url)

    def _extract_text(self, html: str) -> tuple[str, str]:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return "", self._fallback_extract(html)

        soup = BeautifulSoup(html, "html.parser")
        for tag_name in ("script", "style", "nav", "header", "footer", "aside", "noscript", "iframe", "svg"):
            for tag in soup.find_all(tag_name):
                tag.decompose()

        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        main = soup.find("main") or soup.find("article") or soup.find("body") or soup
        return title, main.get_text("\n", strip=True)

    def _extract_snippets(self, text: str, query: str, max_snippets: int) -> list[dict[str, Any]]:
        keywords = self._keywords(query)
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if len(p.strip()) >= 40]
        if not keywords:
            return [{"text": p[:800], "match_score": 0, "position": i} for i, p in enumerate(paragraphs[:max_snippets])]
        scored: list[tuple[int, int, str]] = []
        for index, paragraph in enumerate(paragraphs):
            lower = paragraph.lower()
            score = sum(1 for keyword in keywords if keyword in lower)
            if score:
                scored.append((score, index, paragraph))

        if not scored:
            return [{"text": p[:800], "match_score": 0, "position": i} for i, p in enumerate(paragraphs[:max_snippets])]

        scored.sort(key=lambda item: (-item[0], item[1]))
        snippets = []
        for score, index, paragraph in scored[:max_snippets]:
            snippets.append({
                "text": normalize_snippet(paragraph, max_chars=900),
                "match_score": score,
                "position": index,
            })
        return snippets

    def _keywords(self, query: str) -> list[str]:
        raw = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-_/]{2,}|[\u4e00-\u9fff]{2,}", query.lower())
        stopwords = {
            "the", "and", "for", "with", "from", "that", "this", "what", "which", "official",
            "documentation", "source", "verify", "data", "dataset",
        }
        keywords = []
        for token in raw:
            if token in stopwords or token in keywords:
                continue
            keywords.append(token)
        return keywords[:12]

    def _verify_claim_support(
        self,
        query: str,
        text: str,
        snippets: list[dict[str, Any]],
    ) -> dict[str, Any]:
        keywords = self._keywords(query)
        required_groups = self._required_relevance_groups(query)
        haystack = " ".join([
            text[:20000],
            " ".join(str(snippet.get("text", "")) for snippet in snippets if isinstance(snippet, dict)),
        ]).lower()
        matched_keywords = [keyword for keyword in keywords if keyword in haystack]
        missing_keywords = [keyword for keyword in keywords if keyword not in haystack]
        group_matches = [
            {
                "required_any": list(group),
                "matched": [term for term in group if term in haystack],
            }
            for group in required_groups
        ]
        required_pass = all(item["matched"] for item in group_matches)
        coverage = len(matched_keywords) / max(1, len(keywords))

        if required_pass and (coverage >= 0.5 or not keywords):
            level = "supported"
            reason = "Official page contains the required entity terms and enough query keywords."
        elif matched_keywords:
            level = "weak_support"
            reason = "Official page matches some query terms but does not fully support the claim."
        else:
            level = "unsupported"
            reason = "Official page did not match the requested claim."

        return {
            "level": level,
            "reason": reason,
            "coverage": round(coverage, 3),
            "matched_keywords": matched_keywords,
            "missing_keywords": missing_keywords[:8],
            "required_groups": group_matches,
        }

    def _required_relevance_groups(self, query: str) -> list[tuple[str, ...]]:
        lower = str(query or "").lower()
        groups: list[tuple[str, ...]] = []
        if "sentinel-2" in lower or "sentinel 2" in lower or re.search(r"\bmsi\b", lower):
            groups.append(("sentinel-2", "sentinel 2", "/s2", "s2-", "msi"))
        if "landsat" in lower:
            groups.append(("landsat", "tirs", "oli"))
        if "modis" in lower or "mod11" in lower:
            groups.append(("modis", "mod11"))
        if "ecostress" in lower:
            groups.append(("ecostress",))
        if "earth engine" in lower or re.search(r"\bgee\b", lower):
            groups.append(("earth engine", "google earth engine"))
        if "python" in lower:
            groups.append(("python",))
        if "github" in lower:
            groups.append(("github",))
        if "openai" in lower:
            groups.append(("openai",))
        return groups

    def _is_allowed_official_url(self, url: str) -> bool:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        for domain in self.allowed_domains:
            domain_host = urlparse(domain if "://" in domain else f"https://{domain}").netloc.lower()
            if domain_host.startswith("www."):
                domain_host = domain_host[4:]
            if host == domain_host or host.endswith("." + domain_host):
                return True
        return False

    def _is_http_url(self, url: str) -> bool:
        return url.startswith(("http://", "https://"))

    def _error(self, url: str, error: str, source_type: str = "official_doc") -> dict[str, Any]:
        return {
            "query": "",
            "url": url,
            "results": [],
            "total": 0,
            "match_count": 0,
            "source": "official_doc_fetcher",
            "source_type": source_type,
            "evidence_level": "rejected" if source_type == "rejected_url" else "speculative",
            "error": error,
        }

    def _fallback_extract(self, html: str) -> str:
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
        return re.sub(r"<[^>]+>", "\n", html)

    def _clean_text(self, text: str) -> str:
        lines = [line.strip() for line in normalize_extracted_text(text).splitlines()]
        lines = [line for line in lines if len(line) > 3]
        return normalize_extracted_text("\n".join(lines))

    def _title_from_url(self, url: str) -> str:
        parsed = urlparse(url)
        return parsed.path.rstrip("/").split("/")[-1] or parsed.netloc
