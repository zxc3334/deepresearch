import unittest

from src.agents.summarizer import SummarizerAgent
from src.orchestrator.schemas import (
    AgentResult,
    AgentStatus,
    EvidenceItem,
    EvidenceLevel,
    SourceTier,
)


class SummarizerConfidenceTests(unittest.TestCase):
    def setUp(self):
        self.agent = SummarizerAgent(name="summarizer", policy=lambda messages: {"content": ""})

    def test_extracts_english_confidence(self):
        self.assertEqual(
            self.agent._extract_report_confidence("Overall Confidence: 0.83"),
            0.83,
        )

    def test_extracts_chinese_confidence(self):
        self.assertEqual(
            self.agent._extract_report_confidence("总体置信度：0.76"),
            0.76,
        )

    def test_extracts_percentage_confidence(self):
        self.assertEqual(
            self.agent._extract_report_confidence("综合可信度：82%"),
            0.82,
        )

    def test_parse_report_uses_evidence_fallback_when_confidence_missing(self):
        results = [
            AgentResult(
                task_id="task_1",
                status=AgentStatus.SUCCESS,
                output="supported claim",
                confidence=0.6,
                evidence_items=[
                    EvidenceItem(
                        claim="USGS Landsat Collection 2 has surface temperature products.",
                        level=EvidenceLevel.EVIDENCE_BACKED,
                        source_tier=SourceTier.OFFICIAL,
                        confidence=0.8,
                    ),
                    EvidenceItem(
                        claim="Sentinel-2 lacks a thermal infrared band.",
                        level=EvidenceLevel.EVIDENCE_BACKED,
                        source_tier=SourceTier.OFFICIAL,
                        confidence=0.8,
                    ),
                ],
            )
        ]

        report = self.agent._parse_report("query", "No explicit confidence line.", results)

        self.assertGreater(report.confidence, 0.5)

    def test_parse_report_applies_success_rate_calibration(self):
        results = [
            AgentResult(task_id="task_1", status=AgentStatus.SUCCESS, confidence=0.6),
            AgentResult(task_id="task_2", status=AgentStatus.FAILED, confidence=0.0),
        ]

        report = self.agent._parse_report("query", "Overall Confidence: 0.8", results)

        self.assertEqual(report.confidence, 0.57)

    def test_system_prompt_uses_configured_output_language(self):
        zh_agent = SummarizerAgent(name="summarizer", policy=lambda messages: {"content": ""}, output_language="zh-CN")
        en_agent = SummarizerAgent(name="summarizer", policy=lambda messages: {"content": ""}, output_language="en-US")

        self.assertIn("Output language: Simplified Chinese", zh_agent._system_prompt("general"))
        self.assertIn("Output language: English", en_agent._system_prompt("general"))


if __name__ == "__main__":
    unittest.main()
