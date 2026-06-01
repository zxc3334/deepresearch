import unittest

from src.agents.prompt_builder import ResearchPromptBuilder
from src.orchestrator.schemas import SubTask, TaskType


class FakeTool:
    def __init__(self, name: str, description: str = "Fake tool") -> None:
        self.name = name
        self.description = description


class PromptBuilderTests(unittest.TestCase):
    def _task(self) -> SubTask:
        return SubTask(
            task_id="t1",
            task_type=TaskType.DATA_DISCOVERY,
            description="Find Landsat LST official dataset evidence",
            expected_type="factual",
            context_keys=["previous_result"],
            search_hints=["Landsat", "LST"],
        )

    def test_system_prompt_without_tools_omits_tool_section(self):
        builder = ResearchPromptBuilder()

        prompt = builder.system_prompt([])
        debug = builder.prompt_debug()

        self.assertNotIn("## AVAILABLE TOOLS", prompt)
        self.assertNotIn("tool_guide", debug["system_sections"])
        self.assertEqual(debug["system_sections"], ["identity", "system_rules", "output_format"])

    def test_system_prompt_lists_only_registered_tools(self):
        builder = ResearchPromptBuilder()

        prompt = builder.system_prompt([FakeTool("web_search"), FakeTool("paper_search")])

        self.assertIn("- web_search:", prompt)
        self.assertIn("- paper_search:", prompt)
        self.assertNotIn("- calculator:", prompt)
        self.assertNotIn("- geo_plan_validator:", prompt)

    def test_task_prompt_dynamic_sections_only_when_context_exists(self):
        builder = ResearchPromptBuilder()
        task = self._task()

        prompt = builder.task_prompt(task, {
            "previous_result": "Use prior AOI constraints.",
            "wiki_context": "Known Landsat LST note.",
            "user_instructions": "Focus on official sources.",
            "external_prefetch": "- web_search: USGS page | https://www.usgs.gov | source_tier=official",
        })
        debug = builder.prompt_debug()

        self.assertIn("## Memory Context", prompt)
        self.assertIn("## Wiki Context", prompt)
        self.assertIn("## User Instructions", prompt)
        self.assertIn("## External Prefetch Evidence", prompt)
        self.assertEqual(
            debug["task_sections"],
            [
                "task_context",
                "tool_recommendations",
                "memory_context",
                "wiki_context",
                "user_instructions",
                "external_prefetch",
                "output_format",
            ],
        )

    def test_task_prompt_omits_absent_dynamic_sections(self):
        builder = ResearchPromptBuilder()

        prompt = builder.task_prompt(self._task(), {})
        debug = builder.prompt_debug()

        self.assertNotIn("## Memory Context", prompt)
        self.assertNotIn("## Wiki Context", prompt)
        self.assertNotIn("## User Instructions", prompt)
        self.assertNotIn("## External Prefetch Evidence", prompt)
        self.assertEqual(
            debug["task_sections"],
            ["task_context", "tool_recommendations", "output_format"],
        )


if __name__ == "__main__":
    unittest.main()
