import unittest

from src.agents.tool_calling_loop import ToolCallingLoop, ToolLoopConfig
from src.agents.tool_registry import ToolRegistry
from src.orchestrator.schemas import SubTask, TaskType


class FakeTraceRecorder:
    def __init__(self) -> None:
        self.events = []

    def record(self, event: str, **payload) -> None:
        self.events.append({"event": event, **payload})


class FakePolicy:
    def __init__(self) -> None:
        self.calls = 0
        self.tools = []

    def set_tools(self, tools):
        self.tools = tools

    def __call__(self, messages):
        self.calls += 1
        if self.calls == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_official_search",
                        "type": "function",
                        "function": {
                            "name": "official_source_search",
                            "arguments": '{"query": "Python asyncio cancellation", "top_n": 1}',
                        },
                    }
                ],
                "usage": {"total_tokens": 10},
            }
        return {
            "content": "Final answer. Confidence: 0.8",
            "tool_calls": [],
            "usage": {"total_tokens": 10},
        }


class FakeOfficialSourceSearchTool:
    name = "official_source_search"

    def get_openai_tool_schema(self):
        return {"type": "function", "function": {"name": self.name, "parameters": {"type": "object"}}}

    async def execute(self, query: str, top_n: int = 1):
        return {
            "query": query,
            "source": "official_source_search:test",
            "total": 1,
            "results": [
                {
                    "title": "Python asyncio docs",
                    "url": "https://docs.python.org/3/library/asyncio-task.html",
                    "snippet": "Task cancellation documentation.",
                    "source_type": "official",
                }
            ],
            "evidence_level": "evidence_backed",
        }


class FakeOfficialDocFetcherTool:
    name = "official_doc_fetcher"

    def get_openai_tool_schema(self):
        return {"type": "function", "function": {"name": self.name, "parameters": {"type": "object"}}}

    async def execute(self, url: str, query: str = "", max_chars: int = 6000, max_snippets: int = 3):
        return {
            "query": query,
            "url": url,
            "final_url": url,
            "source": "official_doc_fetcher",
            "source_type": "official_doc",
            "evidence_level": "evidence_backed",
            "match_count": 1,
            "results": [
                {
                    "title": "Python asyncio docs",
                    "url": url,
                    "snippet": "Cancellation propagates through awaited tasks.",
                    "snippets": [
                        {
                            "text": "Cancellation propagates through awaited tasks.",
                            "match_score": 2,
                            "position": 1,
                        }
                    ],
                    "claim_support": {"level": "supported", "reason": "Matched query terms."},
                    "source_type": "official_doc",
                }
            ],
            "claim_support": {"level": "supported", "reason": "Matched query terms."},
        }


class ToolCallingLoopTraceTests(unittest.TestCase):
    def test_trace_tool_result_counts_paper_urls(self):
        recorder = FakeTraceRecorder()
        loop = ToolCallingLoop(
            policy=lambda messages: {},
            tool_registry=ToolRegistry([]),
            trace_recorder=recorder,
        )
        task = SubTask(task_id="t1", task_type=TaskType.LITERATURE, description="Find papers")

        loop._trace_tool_result(
            task,
            turn=1,
            tool_name="paper_search",
            result={
                "source": "paper_search:openalex",
                "total": 2,
                "papers": [
                    {"title": "Paper A", "url": "https://openalex.org/W1"},
                    {"title": "Paper B", "pdf_url": "https://example.org/paper.pdf"},
                ],
            },
        )

        event = recorder.events[0]
        self.assertEqual(event["event"], "tool_result")
        self.assertEqual(event["url_count"], 2)
        self.assertEqual(event["urls"], ["https://openalex.org/W1", "https://example.org/paper.pdf"])


class ToolCallingLoopOfficialAutoFetchTests(unittest.IsolatedAsyncioTestCase):
    async def test_official_search_auto_fetches_official_doc(self):
        recorder = FakeTraceRecorder()
        loop = ToolCallingLoop(
            policy=FakePolicy(),
            tool_registry=ToolRegistry([
                FakeOfficialSourceSearchTool(),
                FakeOfficialDocFetcherTool(),
            ]),
            config=ToolLoopConfig(
                max_turns=2,
                auto_fetch_official_docs=True,
                auto_fetch_official_max_urls=1,
            ),
            trace_recorder=recorder,
        )
        task = SubTask(task_id="t1", task_type=TaskType.VERIFY, description="Verify Python asyncio cancellation")

        result = await loop.run(task, system_prompt="system", user_prompt="user")

        self.assertEqual(result.status.value, "success")
        official_search_steps = [
            step for step in result.trajectory
            if step.get("role") == "tool" and step.get("name") == "official_source_search"
        ]
        official_doc_steps = [
            step for step in result.trajectory
            if step.get("role") == "tool" and step.get("name") == "official_doc_fetcher"
        ]
        self.assertEqual(len(official_search_steps), 1)
        self.assertEqual(len(official_doc_steps), 1)
        self.assertTrue(official_doc_steps[0].get("auto_fetch"))
        self.assertEqual(
            official_search_steps[0]["result"]["auto_fetched_documents"][0]["source"],
            "official_doc_fetcher",
        )

        official_doc_trace = [
            event for event in recorder.events
            if event.get("event") == "tool_result" and event.get("tool") == "official_doc_fetcher"
        ]
        self.assertEqual(len(official_doc_trace), 1)


if __name__ == "__main__":
    unittest.main()
