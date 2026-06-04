import unittest

from src.tools.browser import BrowserTool
from src.tools.paper_search import PaperSearchTool
from src.tools.web_search import BaseWebSearchTool
from src.utils.text_cleanup import normalize_extracted_text, normalize_snippet


class DummySearchTool(BaseWebSearchTool):
    async def execute(self, query: str, top_n: int = 5) -> dict:
        return {"query": query, "results": [], "total": 0}


class TextCleanupTests(unittest.TestCase):
    def test_normalizes_soft_line_breaks_from_extracted_pages(self):
        raw = "T.F.\nZhao\nand\nK.F.\nFong.\nCharacterization\nof\ndifferent\nheat\nmitigation\nstrategies\nin\nlands"

        cleaned = normalize_extracted_text(raw)

        self.assertEqual(
            cleaned,
            "T.F. Zhao and K.F. Fong. Characterization of different heat mitigation strategies in lands",
        )

    def test_snippet_normalization_returns_one_line(self):
        cleaned = normalize_snippet("Using\nAI:\nanalyze\nmembership\ncancellation\nreasons")

        self.assertEqual(cleaned, "Using AI: analyze membership cancellation reasons")

    def test_web_search_snippet_is_normalized_before_ranking(self):
        tool = DummySearchTool()
        ranked = tool._rank_results(
            [
                {
                    "title": "Python asyncio",
                    "url": "https://docs.python.org/3/library/asyncio-task.html",
                    "snippet": "Task.cancel()\nraises\nCancelledError\nat\nthe\nnext\nopportunity.",
                }
            ],
            "asyncio cancel",
            top_n=1,
        )

        self.assertEqual(
            ranked[0]["snippet"],
            "Task.cancel() raises CancelledError at the next opportunity.",
        )

    def test_browser_clean_text_merges_extraction_noise(self):
        cleaned = BrowserTool._clean_text("Characterization\nof\ndifferent\nheat\nmitigation\nstrategies\nin\nlands")

        self.assertEqual(cleaned, "Characterization of different heat mitigation strategies in lands")

    def test_paper_search_normalizes_summary(self):
        tool = PaperSearchTool()
        paper = tool._normalize_paper(
            {
                "title": "Urban\nheat\nisland",
                "summary": "Landsat\nand\nMODIS\nare\nused\nfor\nLST.",
            }
        )

        self.assertEqual(paper["title"], "Urban heat island")
        self.assertEqual(paper["summary"], "Landsat and MODIS are used for LST.")


if __name__ == "__main__":
    unittest.main()
