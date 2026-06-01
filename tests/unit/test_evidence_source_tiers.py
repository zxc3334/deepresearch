import unittest

from src.evidence.evidence_store import EvidenceStore
from src.orchestrator.schemas import AgentResult, AgentStatus, SourceTier


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


if __name__ == "__main__":
    unittest.main()
