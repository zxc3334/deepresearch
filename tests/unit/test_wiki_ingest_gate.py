import tempfile
import unittest
from pathlib import Path

from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.schemas import (
    AgentResult,
    AgentStatus,
    EvidenceItem,
    EvidenceLevel,
    ResearchReport,
    SourceTier,
)
from src.wiki import WikiStore


class WikiIngestGateTests(unittest.TestCase):
    def _orchestrator(self, root_dir: str) -> Orchestrator:
        return Orchestrator(
            planner=None,
            agent_pool=None,
            wiki_store=WikiStore(root_dir=root_dir, user_id="u1"),
        )

    def _orchestrator_with_config(self, root_dir: str, wiki_config: dict) -> Orchestrator:
        return Orchestrator(
            planner=None,
            agent_pool=None,
            wiki_store=WikiStore(root_dir=root_dir, user_id="u1"),
            wiki_config=wiki_config,
        )

    def test_saves_draft_when_report_passes_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            orchestrator = self._orchestrator(tmp)
            orchestrator._results = [
                AgentResult(
                    task_id="t1",
                    status=AgentStatus.SUCCESS,
                    evidence_items=[
                        EvidenceItem(
                            claim="Official source supports the claim.",
                            level=EvidenceLevel.EVIDENCE_BACKED,
                            source_tier=SourceTier.OFFICIAL,
                            source="https://www.usgs.gov/example",
                        )
                    ],
                )
            ]
            report = ResearchReport(
                query="Landsat LST",
                content="USGS supports the Landsat LST claim.",
                confidence=0.8,
                sources=[{"url": "https://www.usgs.gov/example", "title": "USGS"}],
                evidence_summary={"counts": {"evidence_backed": 1}},
            )

            orchestrator._maybe_save_wiki_report(report)

            pages = list((Path(tmp) / "users" / "u1" / "raws").glob("*.md"))
            self.assertEqual(len(pages), 1)
            content = pages[0].read_text(encoding="utf-8")
            self.assertIn('"status": "draft"', content)
            self.assertIn('"source_tier": "official"', content)

    def test_skips_speculative_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            orchestrator = self._orchestrator(tmp)
            orchestrator._results = [
                AgentResult(task_id="t1", status=AgentStatus.SUCCESS, evidence_items=[])
            ]
            report = ResearchReport(
                query="Weak report",
                content="Unsupported claim.",
                confidence=0.8,
                evidence_summary={"counts": {"speculative": 1}},
            )

            allowed, reason, metadata = orchestrator._wiki_ingest_gate(report)

            self.assertFalse(allowed)
            self.assertEqual(reason, "no_evidence_backed_or_verified_evidence")
            self.assertEqual(metadata["evidence_level"], "speculative")

    def test_uses_configured_min_confidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            orchestrator = self._orchestrator_with_config(
                tmp,
                {"ingest": {"min_confidence": 0.9}},
            )
            orchestrator._results = [
                AgentResult(
                    task_id="t1",
                    status=AgentStatus.SUCCESS,
                    evidence_items=[
                        EvidenceItem(
                            claim="Official source supports the claim.",
                            level=EvidenceLevel.EVIDENCE_BACKED,
                            source_tier=SourceTier.OFFICIAL,
                        )
                    ],
                )
            ]
            report = ResearchReport(
                query="Threshold report",
                content="Supported but below configured confidence.",
                confidence=0.8,
                evidence_summary={"counts": {"evidence_backed": 1}},
            )

            allowed, reason, _ = orchestrator._wiki_ingest_gate(report)

            self.assertFalse(allowed)
            self.assertEqual(reason, "confidence_below_threshold")

    def test_skips_mock_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            orchestrator = self._orchestrator(tmp)
            orchestrator._results = [
                AgentResult(
                    task_id="t1",
                    status=AgentStatus.SUCCESS,
                    evidence_items=[
                        EvidenceItem(
                            claim="Mock source supports the claim.",
                            level=EvidenceLevel.EVIDENCE_BACKED,
                            source_tier=SourceTier.GENERAL,
                        )
                    ],
                )
            ]
            report = ResearchReport(
                query="Mock report",
                content="Mock-supported claim.",
                confidence=0.8,
                sources=[{"url": "https://example.com/mock", "title": "Mock Web Search Result"}],
                evidence_summary={"counts": {"evidence_backed": 1}},
            )

            allowed, reason, _ = orchestrator._wiki_ingest_gate(report)

            self.assertFalse(allowed)
            self.assertEqual(reason, "mock_source_present")

    def test_reuses_duplicate_raw_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            orchestrator = self._orchestrator_with_config(
                tmp,
                {
                    "ingest": {
                        "raw_dedup": {"enabled": True, "threshold": 0.72, "reingest_duplicate": False},
                        "structured": {"enabled": True},
                    }
                },
            )
            orchestrator._results = [
                AgentResult(
                    task_id="t1",
                    status=AgentStatus.SUCCESS,
                    evidence_items=[
                        EvidenceItem(
                            claim="Official source supports the claim.",
                            level=EvidenceLevel.EVIDENCE_BACKED,
                            source_tier=SourceTier.OFFICIAL,
                            source="https://www.usgs.gov/example",
                        )
                    ],
                )
            ]
            report = ResearchReport(
                query="如何研究武汉城市扩张对地表热环境的影响",
                content="Landsat LST NDVI NDBI 城市扩张 热环境 方法流程",
                confidence=0.8,
                sources=[{"url": "https://www.usgs.gov/example", "title": "USGS"}],
                evidence_summary={"counts": {"evidence_backed": 1}},
            )

            orchestrator._maybe_save_wiki_report(report)
            orchestrator._maybe_save_wiki_report(report)

            pages = list((Path(tmp) / "users" / "u1" / "raws").glob("*.md"))
            self.assertEqual(len(pages), 1)


if __name__ == "__main__":
    unittest.main()
