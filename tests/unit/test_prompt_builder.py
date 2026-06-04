import unittest

from src.agents.prompt_builder import ResearchPromptBuilder
from src.core.domain_profiles import resolve_domain_profile
from src.orchestrator.schemas import SubTask, TaskType


class FakeTool:
    def __init__(self, name: str, description: str = "Fake tool") -> None:
        self.name = name
        self.description = description


class FakeWikiTool(FakeTool):
    def __init__(self, has_pages: bool) -> None:
        super().__init__("wiki_search")
        self._has_pages = has_pages

    def has_pages(self) -> bool:
        return self._has_pages


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
        self.assertEqual(debug["system_sections"], ["identity", "system_rules", "output_language", "evidence_checklist", "output_format"])
        self.assertIn("Output language: Simplified Chinese", prompt)

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
                "output_language",
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
            ["task_context", "tool_recommendations", "output_language", "output_format"],
        )

    def test_output_language_can_be_configured_to_english(self):
        profile = resolve_domain_profile({"domain_adapter": {"mode": "general"}})
        profile["output_language"] = "en-US"
        builder = ResearchPromptBuilder(domain_profile=profile)

        prompt = builder.system_prompt([FakeTool("web_search")])
        task_prompt = builder.task_prompt(self._task(), {})

        self.assertIn("Output language: English", prompt)
        self.assertIn("Output language: English", task_prompt)

    def test_gis_task_recommendations_prefer_external_evidence_tools(self):
        builder = ResearchPromptBuilder(domain_profile=resolve_domain_profile({"domain_adapter": {"mode": "geo_remote_sensing"}}))
        task = SubTask(
            task_id="geo",
            task_type=TaskType.DATA_DISCOVERY,
            description="Validate Landsat Sentinel-2 MODIS LST dataset bands, resolution, and limitations.",
        )

        recommendations = builder.recommend_tools(task)

        self.assertIn("official_source_search", recommendations)
        self.assertIn("official_doc_fetcher", recommendations)
        self.assertIn("paper_search", recommendations)
        self.assertNotIn("dataset_registry", recommendations)
        self.assertNotIn("method_registry", recommendations)
        self.assertNotIn("geo_plan_validator", recommendations)

    def test_general_profile_does_not_render_gis_domain_rules(self):
        builder = ResearchPromptBuilder(domain_profile=resolve_domain_profile({"domain_adapter": {"mode": "general"}}))

        prompt = builder.system_prompt([FakeTool("web_search"), FakeTool("paper_search")])

        self.assertNotIn("## DOMAIN PROFILE: geo_remote_sensing", prompt)
        self.assertNotIn("Sentinel-2 can directly retrieve LST", prompt)

    def test_geo_profile_renders_gis_domain_rules_and_preferred_domains(self):
        builder = ResearchPromptBuilder(domain_profile=resolve_domain_profile({"domain_adapter": {"mode": "geo_remote_sensing"}}))

        prompt = builder.system_prompt([FakeTool("official_source_search"), FakeTool("paper_search")])

        self.assertIn("## DOMAIN PROFILE: geo_remote_sensing", prompt)
        self.assertIn("Sentinel-2 can directly retrieve LST", prompt)
        self.assertIn("developers.google.com/earth-engine", prompt)
        self.assertIn("GIS/RS Constraint Extraction", prompt)
        self.assertIn("Dataset candidate table", prompt)
        self.assertIn("Data-method fit matrix", prompt)
        self.assertIn("GIS/RS risk checklist", prompt)

    def test_wiki_context_title_is_not_duplicated(self):
        builder = ResearchPromptBuilder()
        prompt = builder.task_prompt(self._task(), {"wiki_context": "## Wiki Context\n- Existing note"})

        self.assertEqual(prompt.count("## Wiki Context"), 1)

    def test_empty_wiki_is_not_recommended_as_first_tool(self):
        builder = ResearchPromptBuilder(domain_profile=resolve_domain_profile({"domain_adapter": {"mode": "geo_remote_sensing"}}))
        recommendations = builder.recommend_tools(
            self._task(),
            available_tools=[
                FakeWikiTool(has_pages=False),
                FakeTool("official_source_search"),
                FakeTool("official_doc_fetcher"),
                FakeTool("paper_search"),
                FakeTool("web_search"),
            ],
        )

        self.assertNotEqual(recommendations[0], "wiki_search")
        self.assertNotIn("wiki_search", recommendations)

    def test_non_empty_wiki_can_be_used_after_primary_external_tool(self):
        builder = ResearchPromptBuilder(domain_profile=resolve_domain_profile({"domain_adapter": {"mode": "geo_remote_sensing"}}))
        task = SubTask(
            task_id="method",
            task_type=TaskType.METHOD_DESIGN,
            description="Design LST retrieval workflow and validation method.",
        )
        recommendations = builder.recommend_tools(
            task,
            available_tools=[
                FakeWikiTool(has_pages=True),
                FakeTool("paper_search"),
                FakeTool("official_source_search"),
                FakeTool("web_search"),
            ],
        )

        self.assertIn("wiki_search", recommendations)
        self.assertNotEqual(recommendations[0], "wiki_search")

    def test_gis_data_fact_task_prefers_official_source(self):
        builder = ResearchPromptBuilder(domain_profile=resolve_domain_profile({"domain_adapter": {"mode": "geo_remote_sensing"}}))
        task = SubTask(
            task_id="data",
            task_type=TaskType.DATA_DISCOVERY,
            description="Identify Landsat 8 TIRS bands, spatial resolution, revisit cycle, and product constraints.",
        )
        recommendations = builder.recommend_tools(
            task,
            available_tools=[
                FakeWikiTool(has_pages=True),
                FakeTool("official_source_search"),
                FakeTool("official_doc_fetcher"),
                FakeTool("paper_search"),
                FakeTool("web_search"),
            ],
        )

        self.assertEqual(recommendations[0], "official_source_search")


if __name__ == "__main__":
    unittest.main()
