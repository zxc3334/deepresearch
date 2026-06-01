import unittest

from src.agents.tool_calling_loop import ToolCallingLoop, ToolLoopConfig
from src.agents.tool_registry import ToolRegistry
from src.orchestrator.interaction import InteractiveBus
from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.schemas import AgentResult, AgentStatus, RunConfig, SubTask, TaskType


class FakeTraceRecorder:
    def __init__(self) -> None:
        self.events = []

    def record(self, event_type: str, **payload):
        self.events.append({"event_type": event_type, **payload})


class FakeDAG:
    def __init__(self, task_ids):
        self.task_ids = task_ids

    def __len__(self):
        return len(self.task_ids)

    def __iter__(self):
        return iter(self.task_ids)

    def get_parallel_groups(self):
        return [self.task_ids]

    def get_dependencies(self, node_id):
        return []


class FakePlanner:
    domain = "geo_remote_sensing"
    _last_raw_json = {}

    def __init__(self):
        self.task = SubTask(
            task_id="t1",
            task_type=TaskType.DATA_DISCOVERY,
            description="Find Landsat LST sources",
        )

    def generate_plan(self, query, memory_ctx):
        return FakeDAG(["t1"])

    def get_task_map_from_dag(self, dag, raw_json):
        return {"t1": self.task}


class FakeAgent:
    async def run(self, task, context):
        return AgentResult(
            task_id=task.task_id,
            status=AgentStatus.SUCCESS,
            output=f"context={context.get('user_instructions', '')}",
            confidence=0.8,
        )


class FakeAgentPool:
    def __init__(self):
        self.progress_callback = None

    async def get_agent(self, task_type):
        return FakeAgent()

    async def release_agent(self, agent):
        return None


class FakeTool:
    name = "web_search"

    async def execute(self, query: str, top_n: int = 5):
        return {
            "query": query,
            "results": [
                {
                    "title": "USGS Landsat",
                    "url": "https://www.usgs.gov/example",
                    "snippet": "official",
                    "_source_tier": "official",
                    "_quality_score": 99,
                }
            ],
            "total": 1,
            "source": "fake",
        }


class InteractionProgressTests(unittest.IsolatedAsyncioTestCase):
    async def test_orchestrator_progress_callback_receives_planning_and_task_events(self):
        events = []
        orchestrator = Orchestrator(
            planner=FakePlanner(),
            agent_pool=FakeAgentPool(),
            progress_callback=events.append,
        )
        orchestrator._query = "query"
        orchestrator._config = RunConfig(max_concurrent=1)

        await orchestrator._do_planning()
        await orchestrator._do_dispatching()

        event_types = [event["event_type"] for event in events]
        self.assertIn("planning_start", event_types)
        self.assertIn("planning_end", event_types)
        self.assertIn("task_planned", event_types)
        self.assertIn("task_start", event_types)
        self.assertIn("task_end", event_types)

    async def test_tool_loop_progress_callback_receives_tool_events(self):
        events = []
        tool_call = {
            "id": "call_1",
            "function": {
                "name": "web_search",
                "arguments": '{"query": "Landsat LST", "top_n": 1}',
            },
        }
        loop = ToolCallingLoop(
            policy=lambda messages: {},
            tool_registry=ToolRegistry([FakeTool()]),
            config=ToolLoopConfig(),
            progress_callback=events.append,
        )
        task = SubTask(task_id="t1", task_type=TaskType.SEARCH, description="Search")

        results = await loop._execute_tool_calls(task, 0, [tool_call])

        self.assertIsInstance(results, list)
        event_types = [event["event_type"] for event in events]
        self.assertEqual(event_types, ["tool_call_start", "tool_call_result"])
        self.assertEqual(events[0]["args_summary"]["query"], "Landsat LST")
        self.assertEqual(events[1]["result_summary"]["items"][0]["source_tier"], "official")

    def test_user_instruction_enters_next_task_context(self):
        trace = FakeTraceRecorder()
        bus = InteractiveBus()
        bus.add_instruction("方向偏了，聚焦 Landsat LST")
        orchestrator = Orchestrator(
            planner=FakePlanner(),
            agent_pool=FakeAgentPool(),
            interactive_bus=bus,
            trace_recorder=trace,
        )
        orchestrator._query = "query"
        task = SubTask(task_id="t1", task_type=TaskType.SEARCH, description="Search")

        context = orchestrator._build_task_context(task)

        self.assertIn("Landsat LST", context["user_instructions"])
        event_types = [event["event_type"] for event in trace.events]
        self.assertIn("user_instruction_received", event_types)
        self.assertIn("user_instruction_applied", event_types)
        self.assertIn("context_modified", event_types)

    def test_no_user_instruction_leaves_context_unchanged(self):
        bus = InteractiveBus()
        orchestrator = Orchestrator(
            planner=FakePlanner(),
            agent_pool=FakeAgentPool(),
            interactive_bus=bus,
        )
        orchestrator._query = "query"
        task = SubTask(task_id="t1", task_type=TaskType.SEARCH, description="Search")

        context = orchestrator._build_task_context(task)

        self.assertNotIn("user_instructions", context)

    def test_context_modifier_error_does_not_break_context_building(self):
        trace = FakeTraceRecorder()
        bus = InteractiveBus()
        bus.add_instruction("bad modifier should not break")

        def bad_modifier(context, instructions):
            raise RuntimeError("boom")

        orchestrator = Orchestrator(
            planner=FakePlanner(),
            agent_pool=FakeAgentPool(),
            interactive_bus=bus,
            context_modifier=bad_modifier,
            trace_recorder=trace,
        )
        orchestrator._query = "query"
        task = SubTask(task_id="t1", task_type=TaskType.SEARCH, description="Search")

        context = orchestrator._build_task_context(task)

        self.assertNotIn("user_instructions", context)
        self.assertIn("context_modifier_error", [event["event_type"] for event in trace.events])

    async def test_progress_callback_error_does_not_break_publish(self):
        def bad_callback(event):
            raise RuntimeError("sse disconnected")

        orchestrator = Orchestrator(
            planner=FakePlanner(),
            agent_pool=FakeAgentPool(),
            progress_callback=bad_callback,
        )

        await orchestrator._publish_progress("test_event")

        self.assertEqual(len(orchestrator.progress_bus.errors), 1)


if __name__ == "__main__":
    unittest.main()
