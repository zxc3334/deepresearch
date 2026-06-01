import unittest

from src.tools.web_search import BaseWebSearchTool, OfficialSourceSearchTool


class DummySearchTool(BaseWebSearchTool):
    async def execute(self, query: str, top_n: int = 5) -> dict:
        return {"query": query, "results": [], "total": 0}


class FakeOfficialBackend(DummySearchTool):
    def __init__(self, results_by_query: dict[str, list[dict]]) -> None:
        self.results_by_query = results_by_query

    async def execute(self, query: str, top_n: int = 5) -> dict:
        results = self.results_by_query.get(query, [])[:top_n]
        return {"query": query, "results": results, "total": len(results)}

    def _deduplicate_results(self, results: list[dict]) -> list[dict]:
        seen: set[str] = set()
        unique: list[dict] = []
        for item in results:
            url = item.get("url", "")
            if url in seen:
                continue
            seen.add(url)
            unique.append(item)
        return unique


class WebSearchRankingTests(unittest.IsolatedAsyncioTestCase):
    def test_mixed_sources_rank_official_and_academic_before_general(self):
        tool = DummySearchTool()
        results = [
            {
                "title": "Blog guide to Landsat LST",
                "url": "https://blog.csdn.net/example/article/details/1",
                "snippet": "Landsat land surface temperature tutorial",
            },
            {
                "title": "Landsat Collection 2 Surface Temperature",
                "url": "https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-temperature",
                "snippet": "Official Landsat surface temperature product description",
            },
            {
                "title": "OpenAlex work about Landsat LST",
                "url": "https://openalex.org/W123",
                "snippet": "Academic metadata for Landsat LST research",
            },
            {
                "title": "Wikipedia remote sensing",
                "url": "https://en.wikipedia.org/wiki/Remote_sensing",
                "snippet": "Remote sensing overview",
            },
        ]

        ranked = tool._rank_results(results, "Landsat LST", top_n=4)

        self.assertEqual(ranked[0]["_source_tier"], "official")
        self.assertIn("usgs.gov", ranked[0]["url"])
        self.assertLess(
            ranked.index(next(item for item in ranked if item["_source_tier"] == "general")),
            len(ranked),
        )
        self.assertEqual(ranked[-1]["_source_tier"], "general")

    def test_long_snippet_is_truncated_and_quality_fields_are_written(self):
        tool = DummySearchTool()
        ranked = tool._rank_results(
            [
                {
                    "title": "NASA MODIS",
                    "url": "https://modis.gsfc.nasa.gov/data/dataprod/mod11.php",
                    "snippet": "x" * 800,
                }
            ],
            "MODIS LST",
            top_n=1,
        )

        self.assertEqual(len(ranked[0]["snippet"]), tool.MAX_SNIPPET_CHARS)
        self.assertEqual(ranked[0]["_source_tier"], "official")
        self.assertGreaterEqual(ranked[0]["_quality_score"], 95.0)

    async def test_official_source_search_keeps_domain_filter_and_ranks_results(self):
        backend = FakeOfficialBackend({
            "Landsat LST site:usgs.gov": [
                {
                    "title": "Wrong domain result",
                    "url": "https://blog.csdn.net/wrong",
                    "snippet": "Should be filtered",
                },
                {
                    "title": "USGS Landsat Surface Temperature",
                    "url": "https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-temperature",
                    "snippet": "Official Landsat LST source",
                },
            ],
            "Landsat LST site:nasa.gov": [
                {
                    "title": "NASA Landsat Science",
                    "url": "https://landsat.gsfc.nasa.gov/article/example",
                    "snippet": "NASA Landsat science source",
                }
            ],
        })
        tool = OfficialSourceSearchTool(search_tool=backend, max_domain_queries=2)

        result = await tool.execute("Landsat LST", top_n=5, domains=["usgs.gov", "nasa.gov"])

        urls = [item["url"] for item in result["results"]]
        self.assertNotIn("https://blog.csdn.net/wrong", urls)
        self.assertTrue(all(item["_source_tier"] == "official" for item in result["results"]))
        self.assertTrue(all(item["_quality_score"] >= 95.0 for item in result["results"]))


if __name__ == "__main__":
    unittest.main()
