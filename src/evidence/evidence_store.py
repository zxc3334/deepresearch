"""Evidence-aware classification for agent results.

This is the M3 bridge layer. It does not replace SharedMemoryStore; it turns
raw AgentResult outputs and tool trajectories into explicit EvidenceItem
records that can be stored in memory metadata, used by the summarizer, and fed
back to replanning.
"""
from __future__ import annotations

from collections import Counter
import re
from typing import Any

from ..orchestrator.schemas import AgentResult, AgentStatus, EvidenceItem, EvidenceLevel, SourceTier, SubTask, TaskType
from ..utils.domain_tiers import best_tier, classify_url, extract_hostname


class EvidenceStore:
    """Classify agent outputs into evidence-aware claims."""

    RELEVANCE_STOPWORDS = {
        "and", "the", "for", "with", "from", "that", "this", "into", "using",
        "use", "uses", "study", "research", "method", "methods", "analysis",
        "remote", "sensing", "surface", "urban", "environment", "effect",
        "effects", "impact", "impacts", "data", "dataset", "datasets",
    }

    def annotate_result(self, result: AgentResult, task: SubTask | None = None) -> AgentResult:
        """Attach evidence items to an AgentResult and return it."""
        if result.evidence_items:
            return result
        result.evidence_items = self.build_evidence_items(result, task=task)
        return result

    def build_evidence_items(self, result: AgentResult, task: SubTask | None = None) -> list[EvidenceItem]:
        """Build evidence items for one result.

        Structured evidence tools may emit multiple claim-level checks. Preserve
        those checks as separate evidence items so one invalid claim does not
        make every valid correction from the same task look rejected.
        """
        if result.status in (AgentStatus.FAILED, AgentStatus.TIMEOUT):
            return [self._fallback_evidence_item(result, task=task, sources=[])]

        structured_items = self._structured_evidence_items(result, task=task)
        if structured_items:
            return structured_items

        sources = self.extract_sources(result)
        level, rationale = self.classify(result, sources=sources, task=task)
        source = sources[0].get("url") or sources[0].get("title", "") if sources else ""

        return [self._fallback_evidence_item(result, task=task, sources=sources, level=level, rationale=rationale, source=source)]

    def _fallback_evidence_item(
        self,
        result: AgentResult,
        task: SubTask | None = None,
        sources: list[dict[str, Any]] | None = None,
        level: EvidenceLevel | None = None,
        rationale: str | None = None,
        source: str = "",
    ) -> EvidenceItem:
        sources = sources or []
        if level is None or rationale is None:
            level, rationale = self.classify(result, sources=sources, task=task)
        source_tier = self._best_source_tier(sources)
        return EvidenceItem(
            claim=self._claim_from_output(result.output),
            level=level,
            source_tier=source_tier,
            source=source,
            rationale=rationale,
            task_id=result.task_id,
            confidence=result.confidence,
            source_count=self._source_count(sources),
            metadata={
                "sources": sources,
                "source_count": len(sources),
                "source_tier": source_tier.value,
                "task_type": task.task_type.value if task else "",
                "status": result.status.value,
            },
        )

    def classify(
        self,
        result: AgentResult,
        sources: list[dict[str, Any]] | None = None,
        task: SubTask | None = None,
    ) -> tuple[EvidenceLevel, str]:
        """Assign an evidence level using execution status, source quality, and task type."""
        sources = sources or []
        output_text = str(result.output or "").lower()

        if result.status in (AgentStatus.FAILED, AgentStatus.TIMEOUT):
            return EvidenceLevel.REJECTED, f"Task ended with status={result.status.value}."

        tool_level = self._level_from_structured_tool(result, sources=sources)
        if tool_level is not None:
            return tool_level, "Structured tool evidence level was capped by source provenance."

        if self._looks_rejected(output_text):
            return EvidenceLevel.REJECTED, "Output explicitly states that the claim is unsupported or the tool failed."

        non_mock_sources = [src for src in sources if not self._is_mock_source(src)]
        has_mock_sources = any(self._is_mock_source(src) for src in sources)
        is_validation_task = bool(task and task.task_type in (TaskType.VERIFY, TaskType.GEO_VALIDATION))

        if non_mock_sources:
            source_tier = self._best_source_tier(non_mock_sources)
            consensus = self._check_consensus(result)

            if is_validation_task and result.confidence >= 0.7 and source_tier in (SourceTier.OFFICIAL, SourceTier.ACADEMIC):
                return EvidenceLevel.VERIFIED, (
                    f"Validation task succeeded with {source_tier.value}-tier source evidence."
                )

            if consensus >= 2 and source_tier in (SourceTier.OFFICIAL, SourceTier.ACADEMIC, SourceTier.AUTHORITATIVE):
                return EvidenceLevel.VERIFIED, f"Cross-source consensus from {consensus} independent source domains."

            if source_tier in (SourceTier.OFFICIAL, SourceTier.ACADEMIC):
                return EvidenceLevel.EVIDENCE_BACKED, f"Result is supported by {source_tier.value}-tier source evidence."

            if source_tier == SourceTier.AUTHORITATIVE:
                return EvidenceLevel.EVIDENCE_BACKED, "Result is supported by an authoritative source; treat with caution."

            return EvidenceLevel.SPECULATIVE, (
                f"External source quality is {source_tier.value}; stronger official or academic verification is needed."
            )

        if has_mock_sources:
            return EvidenceLevel.SPECULATIVE, "Result used mock sources, so it is useful for workflow testing but not real evidence."

        if is_validation_task and result.confidence >= 0.7:
            return EvidenceLevel.SPECULATIVE, "Validation-style task succeeded, but no source evidence was available."

        return EvidenceLevel.SPECULATIVE, "No external evidence source was found in the tool trajectory."

    def summarize(self, results: list[AgentResult]) -> dict[str, Any]:
        """Group evidence counts and claims by evidence level."""
        counts: Counter[str] = Counter()
        claims_by_level: dict[str, list[dict[str, Any]]] = {
            level.value: [] for level in EvidenceLevel
        }
        for result in results:
            for item in result.evidence_items:
                counts[item.level.value] += 1
                claims_by_level[item.level.value].append(item.to_dict())
        return {
            "counts": dict(counts),
            "claims_by_level": claims_by_level,
        }

    def build_replan_feedback(self, results: list[AgentResult]) -> str:
        """Create compact feedback for replanning."""
        rejected = []
        speculative = []
        for result in results:
            for item in result.evidence_items:
                if item.level == EvidenceLevel.REJECTED:
                    rejected.append(f"- {item.task_id}: {item.rationale} Claim: {item.claim[:160]}")
                elif item.level == EvidenceLevel.SPECULATIVE:
                    speculative.append(f"- {item.task_id}: {item.rationale} Claim: {item.claim[:160]}")

        parts = []
        if rejected:
            parts.append("Rejected / unsupported results:\n" + "\n".join(rejected[:5]))
        if speculative:
            parts.append("Speculative results needing verification:\n" + "\n".join(speculative[:5]))
        return "\n\n".join(parts)

    def extract_sources(self, result: AgentResult) -> list[dict[str, Any]]:
        """Extract source-like objects from tool trajectory."""
        sources: list[dict[str, Any]] = []
        for step in result.trajectory:
            if step.get("role") != "tool":
                continue
            tool_name = step.get("name", "")
            payload = step.get("result")
            if not isinstance(payload, dict):
                continue

            if isinstance(payload.get("results"), list):
                for item in payload["results"]:
                    if isinstance(item, dict):
                        official_sources = item.get("official_sources")
                        if isinstance(official_sources, list):
                            for official_source in official_sources:
                                if isinstance(official_source, dict):
                                    self._append_source(sources, {
                                        "tool": tool_name,
                                        "url": official_source.get("url", ""),
                                        "title": official_source.get("title", ""),
                                        "snippet": item.get("dataset", "") or item.get("method", ""),
                                    })
                        self._append_source(sources, {
                            "tool": tool_name,
                            "url": item.get("url", ""),
                            "title": item.get("title", ""),
                            "snippet": item.get("snippet", ""),
                            "_quality_score": item.get("_quality_score"),
                            "_source_tier": item.get("_source_tier"),
                        })

            if isinstance(payload.get("papers"), list):
                for paper in payload["papers"]:
                    if isinstance(paper, dict):
                        self._append_source(sources, {
                            "tool": tool_name,
                            "url": paper.get("pdf_url", "") or paper.get("url", ""),
                            "title": paper.get("title", ""),
                            "snippet": str(paper.get("summary", ""))[:300],
                        })

        seen = set()
        unique = []
        for source in sources:
            key = (source.get("url", ""), source.get("title", ""))
            if key in seen:
                continue
            seen.add(key)
            unique.append(source)
        return unique

    def _append_source(self, sources: list[dict[str, Any]], source: dict[str, Any]) -> None:
        """Append only non-empty sources so classification cannot be inflated by blanks."""
        if source.get("url") or source.get("title") or source.get("snippet"):
            sources.append(source)

    def _claim_from_output(self, output: Any) -> str:
        text = str(output or "").strip()
        if len(text) <= 500:
            return text
        return text[:500]

    def _looks_rejected(self, output_text: str) -> bool:
        markers = [
            "tool failed",
            "failed:",
            "无法验证",
            "无法获取",
            "不支持",
            "not supported",
            "insufficient evidence",
        ]
        return any(marker in output_text for marker in markers)

    def _is_mock_source(self, source: dict[str, Any]) -> bool:
        url = str(source.get("url", "")).lower()
        title = str(source.get("title", "")).lower()
        snippet = str(source.get("snippet", "")).lower()
        return "example.com/mock" in url or "mock result" in title or "mock search result" in snippet

    def _is_external_url(self, source: str) -> bool:
        normalized = str(source or "").lower()
        return normalized.startswith("http://") or normalized.startswith("https://")

    def classify_source_tier(self, url: str) -> SourceTier:
        """Classify a URL into a source quality tier."""
        return classify_url(url)

    def _best_source_tier(self, sources: list[dict[str, Any]]) -> SourceTier:
        """Return the highest-quality source tier from source objects."""
        return best_tier([str(source.get("url", "")) for source in sources])

    def _source_count(self, sources: list[dict[str, Any]]) -> int:
        """Count distinct non-empty source URLs/domains/titles."""
        keys: set[str] = set()
        for source in sources:
            url = str(source.get("url", "") or "")
            if url:
                host = extract_hostname(url)
                keys.add(host or url)
                continue
            title = str(source.get("title", "") or "")
            if title:
                keys.add(title)
        return len(keys)

    def _check_consensus(self, result: AgentResult) -> int:
        """Count distinct non-mock source domains across a result trajectory."""
        domains: set[str] = set()
        for source in self.extract_sources(result):
            if self._is_mock_source(source):
                continue
            domain = extract_hostname(str(source.get("url", "") or ""))
            if domain:
                parts = domain.split(".")
                domains.add(".".join(parts[-2:]) if len(parts) >= 2 else domain)
        return len(domains)

    def _structured_source_type(
        self,
        tool_name: str,
        payload: dict[str, Any],
        source: str = "",
        official_sources: list[Any] | None = None,
    ) -> str:
        explicit = str(payload.get("source_type") or "")
        if explicit:
            if official_sources and self._is_external_url(source):
                return f"{explicit}_with_official_url"
            return explicit
        if tool_name == "official_source_search":
            return "official_search"
        if tool_name == "official_doc_fetcher" or payload.get("source_type") == "official_doc":
            return "official_doc"
        if tool_name == "paper_search" or payload.get("source_type") == "academic_paper":
            return "academic_paper"
        return "structured_tool"

    def _cap_structured_level(
        self,
        level: EvidenceLevel,
        tool_name: str,
        payload: dict[str, Any],
        source: str = "",
    ) -> EvidenceLevel:
        """Keep structured tool evidence conservative unless backed by strong sources."""
        if level == EvidenceLevel.REJECTED:
            return EvidenceLevel.REJECTED

        if tool_name == "official_source_search" and level == EvidenceLevel.VERIFIED:
            return EvidenceLevel.EVIDENCE_BACKED

        if tool_name == "official_doc_fetcher" or payload.get("source_type") == "official_doc":
            if level == EvidenceLevel.VERIFIED:
                return EvidenceLevel.EVIDENCE_BACKED
            return level

        if tool_name == "paper_search" or payload.get("source_type") == "academic_paper":
            if level == EvidenceLevel.VERIFIED:
                return EvidenceLevel.EVIDENCE_BACKED
            return level

        if tool_name == "wiki_search" or payload.get("source_type") == "wiki":
            if level == EvidenceLevel.VERIFIED:
                return EvidenceLevel.EVIDENCE_BACKED
            return level

        return level

    def _structured_evidence_items(self, result: AgentResult, task: SubTask | None = None) -> list[EvidenceItem]:
        """Convert structured tool payloads into claim-level evidence items."""
        items: list[EvidenceItem] = []
        task_type = task.task_type.value if task else ""

        for step in result.trajectory:
            if step.get("role") != "tool":
                continue
            tool_name = step.get("name", "")
            payload = step.get("result")
            if not isinstance(payload, dict):
                continue

            if (tool_name == "wiki_search" or payload.get("source_type") == "wiki") and isinstance(payload.get("results"), list):
                raw_level = self._normalize_level(payload.get("evidence_level"))
                for record in payload["results"]:
                    if not isinstance(record, dict):
                        continue
                    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
                    page_level = self._normalize_level(metadata.get("evidence_level") or record.get("evidence_level") or raw_level.value)
                    level = self._cap_structured_level(page_level, tool_name, payload, source=str(record.get("path", "")))
                    source_tier = self._source_tier_from_value(metadata.get("source_tier") or record.get("source_tier"))
                    items.append(EvidenceItem(
                        claim=self._claim_from_output(
                            str(record.get("snippet", "") or record.get("title", "") or "Wiki search result.")
                        ),
                        level=level,
                        source_tier=source_tier,
                        source=str(record.get("path", "") or ""),
                        rationale="Retrieved from local wiki draft; important claims should still be checked against original sources.",
                        task_id=result.task_id,
                        confidence=result.confidence,
                        source_count=0,
                        metadata={
                            "tool": tool_name,
                            "task_type": task_type,
                            "status": result.status.value,
                            "source_type": "wiki",
                            "title": record.get("title", ""),
                            "score": record.get("score", 0),
                            "wiki_metadata": metadata,
                        },
                    ))
                continue

            if isinstance(payload.get("checks"), list):
                source = self._source_for_structured_payload(tool_name, payload)
                source_type = self._structured_source_type(tool_name, payload, source=source)
                for check in payload["checks"]:
                    if not isinstance(check, dict):
                        continue
                    raw_level = self._normalize_level(check.get("level"))
                    level = self._cap_structured_level(raw_level, tool_name, payload, source=source)
                    rationale = str(check.get("reason", "") or "Structured GIS/remote-sensing validation check.")
                    items.append(EvidenceItem(
                        claim=str(check.get("claim", "") or "Structured validation check"),
                        level=level,
                        source_tier=self.classify_source_tier(source),
                        source=source,
                        rationale=rationale,
                        task_id=result.task_id,
                        confidence=result.confidence,
                        source_count=1 if self._is_external_url(source) else 0,
                        metadata={
                            "tool": tool_name,
                            "task_type": task_type,
                            "status": result.status.value,
                            "fix": check.get("fix", ""),
                            "source_type": source_type,
                            "raw_evidence_level": raw_level.value,
                            "requires_external_verification": bool(payload.get("requires_external_verification", False)),
                        },
                    ))
                continue

            if (tool_name == "official_doc_fetcher" or payload.get("source_type") == "official_doc") and isinstance(payload.get("results"), list):
                raw_level = self._normalize_level(payload.get("evidence_level"))
                for record in payload["results"]:
                    if not isinstance(record, dict):
                        continue
                    snippets = record.get("snippets") if isinstance(record.get("snippets"), list) else []
                    claim_support = payload.get("claim_support")
                    if not isinstance(claim_support, dict):
                        claim_support = record.get("claim_support") if isinstance(record.get("claim_support"), dict) else {}
                    support_level = str(claim_support.get("level", "") or "")
                    has_query_match = bool(payload.get("match_count", 0)) or any(
                        isinstance(snippet, dict) and snippet.get("match_score", 0) > 0
                        for snippet in snippets
                    )
                    source = record.get("url", "")
                    base_level = self._cap_structured_level(raw_level, tool_name, payload, source=source)
                    if support_level == "unsupported":
                        level = EvidenceLevel.SPECULATIVE
                    elif task and task.task_type in (TaskType.VERIFY, TaskType.GEO_VALIDATION) and support_level == "supported" and result.confidence >= 0.7:
                        level = EvidenceLevel.VERIFIED
                    else:
                        level = base_level if has_query_match and support_level in ("supported", "weak_support", "") else EvidenceLevel.SPECULATIVE
                    if snippets:
                        claim = str(snippets[0].get("text", "") or record.get("snippet", "") or record.get("title", ""))
                    else:
                        claim = str(record.get("snippet", "") or record.get("title", "") or "Official document fetched without query match.")
                    items.append(EvidenceItem(
                        claim=self._claim_from_output(claim),
                        level=level,
                        source_tier=self.classify_source_tier(source),
                        source=source,
                        rationale=(
                            str(claim_support.get("reason"))
                            if claim_support else
                            (
                                "Fetched official documentation page and extracted query-matched snippets."
                                if snippets else
                                "Fetched official documentation page, but no query-matched snippet was found."
                            )
                        ),
                        task_id=result.task_id,
                        confidence=result.confidence,
                        source_count=1 if self._is_external_url(source) else 0,
                        metadata={
                            "tool": tool_name,
                            "task_type": task_type,
                            "status": result.status.value,
                            "source_type": "official_doc",
                            "title": record.get("title", ""),
                            "official_domain": record.get("official_domain", ""),
                            "content_chars": record.get("content_chars", 0),
                            "match_count": payload.get("match_count", 0),
                            "has_query_match": has_query_match,
                            "claim_support": claim_support,
                            "snippets": snippets[:3],
                        },
                    ))
                continue

            if (tool_name == "paper_search" or payload.get("source_type") == "academic_paper") and isinstance(payload.get("papers"), list):
                raw_level = self._normalize_level(payload.get("evidence_level"))
                seen_paper_sources: set[str] = set()
                seen_paper_titles: set[str] = set()
                for paper in payload["papers"]:
                    if not isinstance(paper, dict):
                        continue
                    source = paper.get("url", "") or paper.get("pdf_url", "")
                    title = str(paper.get("title", "") or "Academic paper")
                    source_key = str(source).lower()
                    title_key = self._normalize_title_key(title)
                    if (source_key and source_key in seen_paper_sources) or (title_key and title_key in seen_paper_titles):
                        continue
                    if source_key:
                        seen_paper_sources.add(source_key)
                    if title_key:
                        seen_paper_titles.add(title_key)
                    level = self._cap_structured_level(raw_level, tool_name, payload, source=source)
                    summary = str(paper.get("summary", "") or "")
                    relevance = self._paper_relevance_gate(paper, payload, task)
                    if level in (EvidenceLevel.EVIDENCE_BACKED, EvidenceLevel.VERIFIED) and not relevance["passed"]:
                        level = EvidenceLevel.SPECULATIVE
                    citation_count = paper.get("citation_count")
                    citation_note = f" citation_count={citation_count}." if citation_count is not None else ""
                    gate_note = f" Relevance gate: {relevance['quality_gate_reason']}."
                    items.append(EvidenceItem(
                        claim=self._claim_from_output(f"{title}. {summary}"),
                        level=level if source else EvidenceLevel.SPECULATIVE,
                        source_tier=SourceTier.ACADEMIC if source else SourceTier.UNVERIFIED,
                        source=source,
                        rationale=(
                            f"Academic literature search result from {paper.get('source', payload.get('backend', 'unknown'))}."
                            f"{citation_note}{gate_note}"
                        ),
                        task_id=result.task_id,
                        confidence=result.confidence,
                        source_count=1 if source else 0,
                        metadata={
                            "tool": tool_name,
                            "task_type": task_type,
                            "status": result.status.value,
                            "source_type": "academic_paper",
                            "paper_id": paper.get("id", ""),
                            "title": title,
                            "authors": paper.get("authors", []),
                            "published": paper.get("published", ""),
                            "citation_count": citation_count,
                            "backend": payload.get("backend", ""),
                            "evidence_relevance_score": relevance["score"],
                            "matched_terms": relevance["matched_terms"],
                            "quality_gate_reason": relevance["quality_gate_reason"],
                        },
                    ))
                continue

        return items

    def _source_for_structured_payload(self, tool_name: str, payload: dict[str, Any]) -> str:
        results = payload.get("results")
        if isinstance(results, list):
            for item in results:
                if isinstance(item, dict):
                    return item.get("url", "") or item.get("title", "") or tool_name
        return tool_name

    def _normalize_level(self, value: Any) -> EvidenceLevel:
        normalized = str(value or "").lower()
        if normalized == "verified":
            return EvidenceLevel.VERIFIED
        if normalized == "evidence_backed":
            return EvidenceLevel.EVIDENCE_BACKED
        if normalized == "rejected":
            return EvidenceLevel.REJECTED
        return EvidenceLevel.SPECULATIVE

    def _source_tier_from_value(self, value: Any) -> SourceTier:
        normalized = str(value or "").lower()
        for tier in SourceTier:
            if normalized == tier.value:
                return tier
        return SourceTier.UNVERIFIED

    def _paper_relevance_gate(
        self,
        paper: dict[str, Any],
        payload: dict[str, Any],
        task: SubTask | None = None,
    ) -> dict[str, Any]:
        context = " ".join([
            str(payload.get("query", "") or ""),
            task.description if task else "",
            " ".join(task.search_hints) if task else "",
        ])
        query_terms = self._evidence_terms(context)
        if not query_terms:
            return {
                "passed": True,
                "score": 1.0,
                "matched_terms": [],
                "quality_gate_reason": "no_specific_query_terms_available",
            }

        paper_text = " ".join([
            str(paper.get("title", "") or ""),
            str(paper.get("summary", "") or ""),
            str(paper.get("published", "") or ""),
        ]).lower()
        matched_terms = sorted(term for term in query_terms if term in paper_text)
        denominator = max(1, min(len(query_terms), 8))
        score = min(1.0, len(matched_terms) / denominator)

        location_terms = self._location_terms(context)
        requires_location = self._requires_location_match(context, task)
        missing_location = bool(
            requires_location
            and location_terms
            and not any(term in paper_text for term in location_terms)
        )
        if missing_location:
            score = min(score, 0.35)

        passed = score >= 0.45
        reason = "matched_specific_query_terms" if passed else "weak_query_or_location_match"
        if missing_location:
            reason = "missing_location_match"

        return {
            "passed": passed,
            "score": round(score, 3),
            "matched_terms": matched_terms,
            "quality_gate_reason": reason,
        }

    def _evidence_terms(self, text: str) -> set[str]:
        terms = {
            term
            for term in re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}|\d{4}", text.lower())
            if term not in self.RELEVANCE_STOPWORDS
        }
        return terms

    def _location_terms(self, text: str) -> set[str]:
        known_locations = {
            "wuhan", "beijing", "shanghai", "guangzhou", "shenzhen", "nanjing",
            "nanchang", "chengdu", "chongqing", "hangzhou", "dhaka", "delhi",
            "kayseri", "kharkiv",
        }
        lower = text.lower()
        return {location for location in known_locations if location in lower}

    def _requires_location_match(self, context: str, task: SubTask | None = None) -> bool:
        lower = context.lower()
        method_markers = {
            "method", "methods", "algorithm", "formula", "retrieval", "radiative", "single-channel",
            "split-window", "moran", "regression", "validation", "workflow", "方法", "算法",
            "公式", "反演", "验证", "流程", "空间自相关", "回归",
        }
        if any(marker in lower for marker in method_markers):
            return False
        if task and task.task_type in (TaskType.METHOD_DESIGN, TaskType.GEO_VALIDATION):
            return False
        if task and task.task_type in (TaskType.LITERATURE, TaskType.ANALYZE):
            return True
        empirical_markers = {"case study", "impact", "effect", "expansion", "影响", "扩张", "案例", "实证"}
        return any(marker in lower for marker in empirical_markers)

    def _normalize_title_key(self, title: str) -> str:
        return " ".join(re.findall(r"[a-zA-Z0-9]+", title.lower()))

    def _level_from_structured_tool(self, result: AgentResult, sources: list[dict[str, Any]] | None = None) -> EvidenceLevel | None:
        """Read explicit evidence levels from structured tools when available."""
        sources = sources or []
        priority = {
            EvidenceLevel.REJECTED: 4,
            EvidenceLevel.VERIFIED: 3,
            EvidenceLevel.EVIDENCE_BACKED: 2,
            EvidenceLevel.SPECULATIVE: 1,
        }
        best: EvidenceLevel | None = None
        for step in result.trajectory:
            if step.get("role") != "tool":
                continue
            payload = step.get("result")
            if not isinstance(payload, dict):
                continue
            tool_name = step.get("name", "")
            source = self._source_for_structured_payload(tool_name, payload)
            levels = []
            if payload.get("evidence_level"):
                levels.append(str(payload["evidence_level"]))
            for check in payload.get("checks", []) or []:
                if isinstance(check, dict) and check.get("level"):
                    levels.append(str(check["level"]))

            for raw_level in levels:
                candidate = self._cap_structured_level(
                    self._normalize_level(raw_level),
                    tool_name,
                    payload,
                    source=source,
                )
                if best is None or priority[candidate] > priority[best]:
                    best = candidate
        return best
