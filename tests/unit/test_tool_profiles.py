import unittest
import tempfile

from src.core.runner import _create_tools_factory, load_config
from src.wiki import WikiStore


class ToolProfileTests(unittest.TestCase):
    def tool_names(self, config_path: str) -> set[str]:
        return {tool.name for tool in _create_tools_factory(load_config(config_path))}

    def test_geo_mvp_default_profile_excludes_local_registry_tools(self):
        names = self.tool_names("configs/geo_mvp.yaml")

        self.assertIn("web_search", names)
        self.assertIn("paper_search", names)
        self.assertIn("code_sandbox", names)
        self.assertNotIn("arxiv_reader", names)
        self.assertNotIn("file_reader", names)
        self.assertNotIn("notepad", names)
        self.assertNotIn("dataset_registry", names)
        self.assertNotIn("method_registry", names)
        self.assertNotIn("geo_plan_validator", names)

    def test_geo_real_search_default_profile_excludes_local_registry_tools(self):
        names = self.tool_names("configs/geo_real_search.yaml")

        self.assertIn("web_search", names)
        self.assertIn("official_doc_fetcher", names)
        self.assertIn("paper_search", names)
        self.assertIn("code_sandbox", names)
        self.assertNotIn("arxiv_reader", names)
        self.assertNotIn("file_reader", names)
        self.assertNotIn("notepad", names)
        self.assertNotIn("dataset_registry", names)
        self.assertNotIn("method_registry", names)
        self.assertNotIn("geo_plan_validator", names)

    def test_default_profile_excludes_local_registry_tools(self):
        names = self.tool_names("configs/default.yaml")

        self.assertNotIn("wiki_search", names)
        self.assertIn("web_search", names)
        self.assertIn("paper_search", names)
        self.assertIn("official_source_search", names)
        self.assertIn("official_doc_fetcher", names)
        self.assertIn("browser", names)
        self.assertIn("calculator", names)
        self.assertNotIn("code_sandbox", names)
        self.assertNotIn("arxiv_reader", names)
        self.assertNotIn("file_reader", names)
        self.assertNotIn("notepad", names)
        self.assertNotIn("dataset_registry", names)
        self.assertNotIn("method_registry", names)
        self.assertNotIn("geo_plan_validator", names)

    def test_wiki_search_registered_only_when_store_is_injected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config("configs/geo_real_search.yaml")
            config["_wiki_store"] = WikiStore(root_dir=tmp, user_id="u1")
            names = {tool.name for tool in _create_tools_factory(config)}

        self.assertIn("wiki_search", names)


if __name__ == "__main__":
    unittest.main()
