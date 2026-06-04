import unittest

from src.core.domain_profiles import resolve_domain_profile


class DomainProfileTests(unittest.TestCase):
    def test_general_profile_exposes_core_research_tools_only(self):
        profile = resolve_domain_profile({"domain_adapter": {"mode": "general"}})

        self.assertEqual(profile["name"], "general")
        self.assertIn("wiki_search", profile["exposed_tools"])
        self.assertIn("web_search", profile["exposed_tools"])
        self.assertIn("paper_search", profile["exposed_tools"])
        self.assertIn("official_source_search", profile["exposed_tools"])
        self.assertIn("official_doc_fetcher", profile["exposed_tools"])
        self.assertIn("browser", profile["exposed_tools"])
        self.assertIn("calculator", profile["exposed_tools"])
        self.assertNotIn("code_sandbox", profile["exposed_tools"])
        self.assertNotIn("file_reader", profile["exposed_tools"])
        self.assertNotIn("notepad", profile["exposed_tools"])
        self.assertNotIn("arxiv_reader", profile["exposed_tools"])

    def test_geo_profile_extends_general_with_code_and_domain_rules(self):
        profile = resolve_domain_profile({"domain_adapter": {"mode": "geo_remote_sensing"}})

        self.assertEqual(profile["name"], "geo_remote_sensing")
        self.assertIn("wiki_search", profile["exposed_tools"])
        self.assertIn("web_search", profile["exposed_tools"])
        self.assertIn("paper_search", profile["exposed_tools"])
        self.assertIn("code_sandbox", profile["exposed_tools"])
        self.assertIn("gis_rs_data_rules", profile["prompt_sections"])
        self.assertIn("gis_rs_constraint_extraction", profile["prompt_sections"])
        self.assertIn("gis_rs_method_validation", profile["prompt_sections"])
        self.assertIn("gis_rs_risk_checklist", profile["prompt_sections"])
        self.assertIn("usgs.gov", profile["preferred_official_domains"])
        self.assertIn("developers.google.com/earth-engine", profile["preferred_official_domains"])
        self.assertTrue(any("Dataset candidate table" in item for item in profile["output_sections"]))
        self.assertTrue(any("Data-method fit matrix" in item for item in profile["output_sections"]))

    def test_domain_adapter_auto_mode_selects_matching_profile(self):
        profile = resolve_domain_profile(
            {
                "domain_adapter": {"mode": "auto"},
                "query": "How to use Landsat data to study LST and urban heat island change?",
            }
        )

        self.assertEqual(profile["name"], "geo_remote_sensing")
        self.assertIn("code_sandbox", profile["exposed_tools"])

    def test_domain_adapter_defaults_to_general(self):
        profile = resolve_domain_profile({})

        self.assertEqual(profile["name"], "general")
        self.assertNotIn("code_sandbox", profile["exposed_tools"])


if __name__ == "__main__":
    unittest.main()
