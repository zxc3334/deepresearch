import asyncio
import unittest

from src.agents.researcher import ResearcherAgent
from src.evidence.evidence_store import EvidenceStore
from src.orchestrator.schemas import AgentStatus, SubTask, TaskType


class CapturePolicy:
    def __init__(self) -> None:
        self.messages = []
        self.tools = []

    def set_tools(self, tools):
        self.tools = tools

    def __call__(self, messages):
        self.messages = list(messages)
        return {
            "role": "assistant",
            "content": "Summary with Confidence: 0.8",
            "tool_calls": [],
            "usage": {"total_tokens": 10},
        }


class FakeTraceRecorder:
    def __init__(self) -> None:
        self.events = []

    def record(self, event_type: str, **payload):
        self.events.append({"event_type": event_type, **payload})


class FakeWebSearchTool:
    name = "web_search"

    def get_openai_tool_schema(self):
        return {"type": "function", "function": {"name": self.name, "parameters": {"type": "object"}}}

    async def execute(self, query: str, top_n: int = 5):
        return {
            "query": query,
            "results": [
                {
                    "title": "NASA Landsat LST",
                    "url": "https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-temperature",
                    "snippet": "Official source " + ("x" * 1000),
                    "_source_tier": "official",
                    "_quality_score": 99.0,
                }
            ][:top_n],
            "total": 1,
            "source": "fake_web",
        }


class FakePaperSearchTool:
    name = "paper_search"

    def get_openai_tool_schema(self):
        return {"type": "function", "function": {"name": self.name, "parameters": {"type": "object"}}}

    async def execute(self, query: str, max_results: int = 5):
        return {
            "query": query,
            "papers": [
                {
                    "title": "Urban heat island remote sensing review",
                    "url": "https://openalex.org/W123",
                    "summary": "Academic source " + ("y" * 1000),
                    "citation_count": 42,
                }
            ][:max_results],
            "total": 1,
            "source": "paper_search:fake",
            "source_type": "academic_paper",
        }


class SlowWebSearchTool(FakeWebSearchTool):
    async def execute(self, query: str, top_n: int = 5):
        await asyncio.sleep(0.05)
        return await super().execute(query, top_n=top_n)


class ExternalPrefetchTests(unittest.IsolatedAsyncioTestCase):
    def _task(self) -> SubTask:
        return SubTask(
            task_id="t1",
            task_type=TaskType.DATA_DISCOVERY,
            description="Find Landsat LST evidence for urban heat island research",
            expected_type="factual",
        )

    async def test_prefetch_disabled_does_not_call_tools_or_inject_prompt(self):
        policy = CapturePolicy()
        trace = FakeTraceRecorder()
        agent = ResearcherAgent(
            name="researcher",
            policy=policy,
            tools=[FakeWebSearchTool(), FakePaperSearchTool()],
            external_prefetch_config={"enabled": False},
            trace_recorder=trace,
        )

        result = await agent.run(self._task(), {})

        self.assertEqual(result.status, AgentStatus.SUCCESS)
        self.assertNotIn("External Prefetch Evidence", policy.messages[1]["content"])
        self.assertFalse([event for event in trace.events if event["event_type"].startswith("external_prefetch")])

    async def test_prefetch_injects_sources_into_prompt_and_trajectory(self):
        policy = CapturePolicy()
        trace = FakeTraceRecorder()
        agent = ResearcherAgent(
            name="researcher",
            policy=policy,
            tools=[FakeWebSearchTool(), FakePaperSearchTool()],
            external_prefetch_config={
                "enabled": True,
                "web_search_top_n": 1,
                "paper_search_top_n": 1,
                "timeout_seconds": 1,
                "max_chars_per_item": 80,
            },
            trace_recorder=trace,
        )

        result = await agent.run(self._task(), {})
        prompt = policy.messages[1]["content"]

        self.assertEqual(result.status, AgentStatus.SUCCESS)
        self.assertIn("External Prefetch Evidence", prompt)
        self.assertIn("source_tier=official", prompt)
        self.assertIn("source_tier=academic", prompt)
        self.assertLessEqual(len(result.trajectory[0]["result"]["results"][0]["snippet"]), 80)

        sources = EvidenceStore().extract_sources(result)
        urls = {source["url"] for source in sources}
        self.assertIn("https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-temperature", urls)
        self.assertIn("https://openalex.org/W123", urls)

        event_types = [event["event_type"] for event in trace.events]
        self.assertIn("external_prefetch_start", event_types)
        self.assertIn("external_prefetch_result", event_types)

    async def test_prefetch_timeout_does_not_fail_task(self):
        policy = CapturePolicy()
        trace = FakeTraceRecorder()
        agent = ResearcherAgent(
            name="researcher",
            policy=policy,
            tools=[SlowWebSearchTool(), FakePaperSearchTool()],
            external_prefetch_config={
                "enabled": True,
                "web_search_top_n": 1,
                "paper_search_top_n": 1,
                "timeout_seconds": 0.001,
                "max_chars_per_item": 80,
            },
            trace_recorder=trace,
        )

        result = await agent.run(self._task(), {})

        self.assertEqual(result.status, AgentStatus.SUCCESS)
        self.assertTrue(any(step.get("error") for step in result.trajectory if step.get("prefetch")))
        self.assertIn("external_prefetch_error", [event["event_type"] for event in trace.events])


if __name__ == "__main__":
    unittest.main()
