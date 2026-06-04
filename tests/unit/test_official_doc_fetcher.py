import unittest

from src.tools.official_doc_fetcher import OfficialDocFetcherTool


class OfficialDocFetcherTests(unittest.TestCase):
    def setUp(self):
        self.tool = OfficialDocFetcherTool()

    def test_claim_support_supported_when_required_entity_and_keywords_match(self):
        text = (
            "Python asyncio documentation explains Task cancellation. "
            "Tasks can be cancelled and CancelledError is raised at the next opportunity."
        )

        support = self.tool._verify_claim_support(
            query="Python asyncio task cancellation official docs",
            text=text,
            snippets=[{"text": text, "match_score": 3}],
        )

        self.assertEqual(support["level"], "supported")
        self.assertIn("python", support["matched_keywords"])
        self.assertIn("asyncio", support["matched_keywords"])

    def test_claim_support_weak_when_page_is_official_but_missing_target_entity(self):
        text = (
            "Copernicus Sentinel-3 provides land surface temperature products. "
            "This page discusses thermal observations and LST imagery."
        )

        support = self.tool._verify_claim_support(
            query="Sentinel-2 MSI bands thermal infrared land surface temperature",
            text=text,
            snippets=[{"text": text, "match_score": 2}],
        )

        self.assertEqual(support["level"], "weak_support")
        self.assertTrue(any(group["required_any"] for group in support["required_groups"]))
        self.assertFalse(all(group["matched"] for group in support["required_groups"]))

    def test_claim_support_unsupported_when_no_query_terms_match(self):
        support = self.tool._verify_claim_support(
            query="OpenAI Responses API official documentation",
            text="This page is about unrelated weather sensor calibration.",
            snippets=[],
        )

        self.assertEqual(support["level"], "unsupported")


if __name__ == "__main__":
    unittest.main()
