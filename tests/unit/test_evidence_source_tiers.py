import unittest

from src.evidence.evidence_store import EvidenceStore
from src.orchestrator.schemas import AgentResult, AgentStatus, SourceTier, SubTask, TaskType


def _result_with_web_source(url: str) -> AgentResult:
    return AgentResult(
        task_id="task_web",
        status=AgentStatus.SUCCESS,
        output="Landsat thermal band evidence.",
        confidence=0.8,
        trajectory=[
            {
                "role": "tool",
                "name": "web_search",
                "result": {
                    "results": [
                        {
                            "title": "source",
                            "url": url,
                            "snippet": "Landsat thermal infrared band.",
                        }
                    ]
                },
            }
        ],
    )


class EvidenceSourceTierTests(unittest.TestCase):
    def test_general_web_source_is_speculative(self):
        store = EvidenceStore()
        result = _result_with_web_source("https://blog.csdn.net/example")
        item = store.build_evidence_items(result)[0]

        self.assertEqual(item.level.value, "speculative")
        self.assertEqual(item.source_tier, SourceTier.GENERAL)
        self.assertEqual(item.source_count, 1)

    def test_official_web_source_is_evidence_backed(self):
        store = EvidenceStore()
        result = _result_with_web_source("https://nasa.gov/landsat")
        item = store.build_evidence_items(result)[0]

        self.assertEqual(item.level.value, "evidence_backed")
        self.assertEqual(item.source_tier, SourceTier.OFFICIAL)
        self.assertEqual(item.source_count, 1)

    def test_paper_search_is_academic_evidence_backed(self):
        store = EvidenceStore()
        result = AgentResult(
            task_id="task_paper",
            status=AgentStatus.SUCCESS,
            output="Paper evidence.",
            confidence=0.7,
            trajectory=[
                {
                    "role": "tool",
                    "name": "paper_search",
                    "result": {
                        "source_type": "academic_paper",
                        "evidence_level": "verified",
                        "papers": [
                            {
                                "title": "A Landsat LST method",
                                "summary": "Uses thermal infrared bands.",
                                "url": "https://openalex.org/W123",
                                "source": "openalex",
                            }
                        ],
                    },
                }
            ],
        )
        item = store.build_evidence_items(result)[0]

        self.assertEqual(item.level.value, "evidence_backed")
        self.assertEqual(item.source_tier, SourceTier.ACADEMIC)
        self.assertEqual(item.source_count, 1)

    def test_wiki_search_creates_wiki_evidence_item(self):
        store = EvidenceStore()
        result = AgentResult(
            task_id="task_wiki",
            status=AgentStatus.SUCCESS,
            output="Local wiki evidence.",
            confidence=0.7,
            trajectory=[
                {
                    "role": "tool",
                    "name": "wiki_search",
                    "result": {
                        "source_type": "wiki",
                        "evidence_level": "evidence_backed",
                        "results": [
                            {
                                "title": "Landsat LST",
                                "path": "data/wiki/users/u1/raws/landsat.md",
                                "snippet": "USGS supports Landsat surface temperature products.",
                                "metadata": {
                                    "evidence_level": "evidence_backed",
                                    "source_tier": "official",
                                },
                            }
                        ],
                    },
                }
            ],
        )

        item = store.build_evidence_items(result)[0]

        self.assertEqual(item.level.value, "evidence_backed")
        self.assertEqual(item.source_tier, SourceTier.OFFICIAL)
        self.assertEqual(item.source_count, 0)
        self.assertEqual(item.metadata["source_type"], "wiki")

    def test_failed_task_overrides_structured_sources(self):
        store = EvidenceStore()
        result = AgentResult(
            task_id="task_failed",
            status=AgentStatus.FAILED,
            output="tool failed: timeout",
            confidence=0.0,
            trajectory=[
                {
                    "role": "tool",
                    "name": "official_doc_fetcher",
                    "result": {
                        "source_type": "official_doc",
                        "evidence_level": "verified",
                        "results": [{"url": "https://nasa.gov/landsat", "title": "NASA"}],
                    },
                }
            ],
        )
        item = store.build_evidence_items(result)[0]

        self.assertEqual(item.level.value, "rejected")
        self.assertEqual(item.source_tier, SourceTier.UNVERIFIED)

    def test_official_doc_unsupported_claim_stays_speculative(self):
        store = EvidenceStore()
        result = AgentResult(
            task_id="task_doc",
            status=AgentStatus.SUCCESS,
            output="Official page was fetched.",
            confidence=0.8,
            trajectory=[
                {
                    "role": "tool",
                    "name": "official_doc_fetcher",
                    "result": {
                        "source_type": "official_doc",
                        "evidence_level": "evidence_backed",
                        "match_count": 0,
                        "claim_support": {
                            "level": "unsupported",
                            "reason": "Official page did not match the requested claim.",
                        },
                        "results": [
                            {
                                "url": "https://docs.python.org/3/library/asyncio-task.html",
                                "title": "asyncio tasks",
                                "snippet": "Python asyncio task documentation.",
                                "claim_support": {
                                    "level": "unsupported",
                                    "reason": "Official page did not match the requested claim.",
                                },
                            }
                        ],
                    },
                }
            ],
        )

        item = store.build_evidence_items(result)[0]

        self.assertEqual(item.level.value, "speculative")
        self.assertEqual(item.source_tier, SourceTier.OFFICIAL)
        self.assertEqual(item.metadata["claim_support"]["level"], "unsupported")

    def test_relevant_paper_keeps_evidence_backed_with_gate_metadata(self):
        store = EvidenceStore()
        task = SubTask(
            task_id="task_paper",
            task_type=TaskType.LITERATURE,
            description="Find Wuhan urban expansion LST Landsat impervious surface evidence from 2018 2024.",
        )
        result = AgentResult(
            task_id="task_paper",
            status=AgentStatus.SUCCESS,
            output="Relevant paper evidence.",
            confidence=0.7,
            trajectory=[
                {
                    "role": "tool",
                    "name": "paper_search",
                    "result": {
                        "query": "Wuhan urban expansion LST Landsat impervious surface 2018 2024",
                        "source_type": "academic_paper",
                        "evidence_level": "evidence_backed",
                        "papers": [
                            {
                                "title": "Wuhan urban expansion and Landsat LST response from 2018 to 2024",
                                "summary": "The study analyzes impervious surface expansion and land surface temperature.",
                                "url": "https://openalex.org/W1",
                                "source": "openalex",
                            }
                        ],
                    },
                }
            ],
        )

        item = store.build_evidence_items(result, task=task)[0]

        self.assertEqual(item.level.value, "evidence_backed")
        self.assertGreaterEqual(item.metadata["evidence_relevance_score"], 0.45)
        self.assertIn("wuhan", item.metadata["matched_terms"])
        self.assertEqual(item.metadata["quality_gate_reason"], "matched_specific_query_terms")

    def test_generic_paper_is_downgraded_when_location_specific_task_is_missing_location_match(self):
        store = EvidenceStore()
        task = SubTask(
            task_id="task_paper",
            task_type=TaskType.LITERATURE,
            description="Find Wuhan urban expansion LST Landsat impervious surface evidence from 2018 2024.",
        )
        result = AgentResult(
            task_id="task_paper",
            status=AgentStatus.SUCCESS,
            output="Generic paper evidence.",
            confidence=0.7,
            trajectory=[
                {
                    "role": "tool",
                    "name": "paper_search",
                    "result": {
                        "query": "Wuhan urban expansion LST Landsat impervious surface 2018 2024",
                        "source_type": "academic_paper",
                        "evidence_level": "evidence_backed",
                        "papers": [
                            {
                                "title": "Research overview on urban heat islands driven by computational intelligence",
                                "summary": "A broad review of urban heat island literature across many cities.",
                                "url": "https://openalex.org/W2",
                                "source": "openalex",
                            }
                        ],
                    },
                }
            ],
        )

        item = store.build_evidence_items(result, task=task)[0]

        self.assertEqual(item.level.value, "speculative")
        self.assertEqual(item.metadata["quality_gate_reason"], "missing_location_match")

    def test_method_paper_does_not_require_location_match(self):
        store = EvidenceStore()
        task = SubTask(
            task_id="task_method",
            task_type=TaskType.METHOD_DESIGN,
            description="Design Wuhan LST retrieval method using radiative transfer equation and single-channel algorithm.",
        )
        result = AgentResult(
            task_id="task_method",
            status=AgentStatus.SUCCESS,
            output="Method paper evidence.",
            confidence=0.7,
            trajectory=[
                {
                    "role": "tool",
                    "name": "paper_search",
                    "result": {
                        "query": "Wuhan LST retrieval radiative transfer single-channel algorithm",
                        "source_type": "academic_paper",
                        "evidence_level": "evidence_backed",
                        "papers": [
                            {
                                "title": "Land surface temperature retrieval using single-channel algorithm",
                                "summary": "The method uses thermal infrared radiative transfer for LST retrieval.",
                                "url": "https://openalex.org/W3",
                                "source": "openalex",
                            }
                        ],
                    },
                }
            ],
        )

        item = store.build_evidence_items(result, task=task)[0]

        self.assertEqual(item.level.value, "evidence_backed")
        self.assertEqual(item.metadata["quality_gate_reason"], "matched_specific_query_terms")

    def test_duplicate_papers_are_deduplicated_by_url_or_title(self):
        store = EvidenceStore()
        task = SubTask(
            task_id="task_paper",
            task_type=TaskType.LITERATURE,
            description="Find Wuhan urban expansion LST Landsat impervious surface evidence.",
        )
        result = AgentResult(
            task_id="task_paper",
            status=AgentStatus.SUCCESS,
            output="Duplicate paper evidence.",
            confidence=0.7,
            trajectory=[
                {
                    "role": "tool",
                    "name": "paper_search",
                    "result": {
                        "query": "Wuhan urban expansion LST Landsat impervious surface",
                        "source_type": "academic_paper",
                        "evidence_level": "evidence_backed",
                        "papers": [
                            {
                                "title": "Wuhan urban expansion and Landsat LST response",
                                "summary": "Wuhan Landsat LST impervious surface.",
                                "url": "https://openalex.org/W1",
                                "source": "openalex",
                            },
                            {
                                "title": "Wuhan urban expansion and Landsat LST response",
                                "summary": "Duplicate title.",
                                "url": "https://doi.org/10.1234/duplicate",
                                "source": "openalex",
                            },
                            {
                                "title": "Different title",
                                "summary": "Duplicate URL.",
                                "url": "https://openalex.org/W1",
                                "source": "openalex",
                            },
                        ],
                    },
                }
            ],
        )

        items = store.build_evidence_items(result, task=task)

        self.assertEqual(len(items), 1)


if __name__ == "__main__":
    unittest.main()
