import json
import unittest

from src.agents.tool_calling_loop import ToolCallingLoop, ToolLoopConfig
from src.agents.tool_context import ContextBudgetManager, ToolResultCompactPolicy, ToolResultNormalizer
from src.agents.tool_registry import ToolRegistry


class FakeTraceRecorder:
    def __init__(self) -> None:
        self.events = []

    def record(self, event_type: str, **payload):
        self.events.append({"event_type": event_type, **payload})


class ToolContextCompactTests(unittest.TestCase):
    def test_within_budget_does_not_compact(self):
        policy = ToolResultCompactPolicy(
            max_chars=100,
            budget_manager=ContextBudgetManager(budget_tokens=1000, compact_trigger_ratio=0.7, chars_per_token=4),
        )

        decision = policy.compact("short content", current_context_chars=10)

        self.assertFalse(decision.compacted)
        self.assertEqual(decision.content, "short content")
        self.assertEqual(decision.reason, "within_budget")

    def test_over_budget_uses_head_tail_and_marks_compact(self):
        policy = ToolResultCompactPolicy(
            max_chars=100,
            head_ratio=0.7,
            budget_manager=ContextBudgetManager(budget_tokens=10, compact_trigger_ratio=0.5, chars_per_token=2),
        )
        content = "A" * 150 + "B" * 150

        decision = policy.compact(content, current_context_chars=100)

        self.assertTrue(decision.compacted)
        self.assertIn("[compact]", decision.content)
        self.assertIn("preserved head/tail 70/30", decision.content)
        self.assertLess(decision.after_chars, decision.before_chars)
        self.assertTrue(decision.content.startswith("A"))
        self.assertTrue(decision.content.endswith("B" * 30))

    def test_error_and_rejected_content_is_not_compacted(self):
        policy = ToolResultCompactPolicy(
            max_chars=50,
            budget_manager=ContextBudgetManager(budget_tokens=1, compact_trigger_ratio=0.5, chars_per_token=1),
        )
        content = '{"error": "API key failed and this unsupported result should stay intact"}' + ("x" * 500)

        decision = policy.compact(content, current_context_chars=1000)

        self.assertFalse(decision.compacted)
        self.assertEqual(decision.content, content)
        self.assertEqual(decision.reason, "error_or_constraint_preserved")

    def test_normalizer_preserves_structured_fields_and_text_fields(self):
        normalizer = ToolResultNormalizer()
        payload = {
            "results": [
                {
                    "title": "USGS Landsat LST",
                    "url": "https://www.usgs.gov/example",
                    "snippet": "official snippet",
                    "_source_tier": "official",
                    "_quality_score": 99,
                    "irrelevant_blob": "drop me",
                }
            ]
        }

        normalized = normalizer.normalize(payload)
        item = normalized["results"][0]

        self.assertEqual(item["title"], "USGS Landsat LST")
        self.assertEqual(item["url"], "https://www.usgs.gov/example")
        self.assertEqual(item["snippet"], "official snippet")
        self.assertEqual(item["_source_tier"], "official")
        self.assertEqual(item["_quality_score"], 99)
        self.assertNotIn("irrelevant_blob", item)

    def test_normalizer_sorts_official_and_academic_before_general(self):
        normalizer = ToolResultNormalizer()
        payload = {
            "results": [
                {
                    "title": "General blog",
                    "url": "https://blog.csdn.net/example",
                    "snippet": "general",
                    "_source_tier": "general",
                    "_quality_score": 30,
                },
                {
                    "title": "Academic paper",
                    "url": "https://openalex.org/W123",
                    "snippet": "academic",
                    "_source_tier": "academic",
                    "_quality_score": 75,
                },
                {
                    "title": "Official doc",
                    "url": "https://www.nasa.gov/example",
                    "snippet": "official",
                    "_source_tier": "official",
                    "_quality_score": 99,
                },
            ]
        }

        normalized = normalizer.normalize(payload)

        self.assertEqual([item["title"] for item in normalized["results"]], [
            "Official doc",
            "Academic paper",
            "General blog",
        ])

    def test_loop_records_compact_trace_event(self):
        trace = FakeTraceRecorder()
        loop = ToolCallingLoop(
            policy=lambda messages: {},
            tool_registry=ToolRegistry([]),
            config=ToolLoopConfig(
                context_budget_tokens=1,
                compact_threshold_ratio=0.5,
                compact_tool_result_chars=120,
                chars_per_token=1,
                head_tail_ratio=0.7,
            ),
            trace_recorder=trace,
        )
        loop.messages = [{"role": "system", "content": "existing context"}]
        payload = {
            "results": [
                {
                    "title": "Official result",
                    "url": "https://nasa.gov/example",
                    "snippet": "x" * 1000,
                    "_source_tier": "official",
                    "_quality_score": 100,
                }
            ]
        }

        content = loop.tool_result_normalizer.dumps(payload)
        compacted = loop._maybe_compact_tool_content(content, "web_search")
        event = next(event for event in trace.events if event["event_type"] == "compact")

        self.assertIn("[compact]", compacted)
        self.assertEqual(event["scope"], "tool_result")
        self.assertEqual(event["tool"], "web_search")
        self.assertEqual(event["strategy"], "head_tail_70_30")


if __name__ == "__main__":
    unittest.main()
