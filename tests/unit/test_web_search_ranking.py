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

    def test_official_source_search_routes_landsat_query_to_usgs_and_nasa(self):
        tool = OfficialSourceSearchTool(search_tool=DummySearchTool(), max_domain_queries=4)

        domains = tool._select_domains("Landsat 8 TIRS surface temperature product")

        self.assertIn("usgs.gov", domains)
        self.assertIn("landsat.gsfc.nasa.gov", domains)
        self.assertIn("nasa.gov", domains)

    def test_official_source_search_routes_sentinel_query_to_esa_and_copernicus(self):
        tool = OfficialSourceSearchTool(search_tool=DummySearchTool(), max_domain_queries=4)

        domains = tool._select_domains("Sentinel-2 MSI bands NDVI NDBI")

        self.assertIn("sentinels.copernicus.eu", domains)
        self.assertIn("sentiwiki.copernicus.eu", domains)
        self.assertIn("esa.int", domains)

    def test_official_source_search_route_domains_beat_preferred_domains(self):
        tool = OfficialSourceSearchTool(
            search_tool=DummySearchTool(),
            max_domain_queries=4,
            preferred_domains=["usgs.gov", "landsat.gsfc.nasa.gov", "nasa.gov", "lpdaac.usgs.gov"],
        )

        domains = tool._select_domains("Sentinel-2 MSI bands thermal infrared surface temperature")

        self.assertEqual(domains[0], "sentinels.copernicus.eu")
        self.assertIn("esa.int", domains)
        self.assertNotEqual(domains[:4], ["usgs.gov", "landsat.gsfc.nasa.gov", "nasa.gov", "lpdaac.usgs.gov"])

    def test_official_source_search_routes_gee_query_to_google_docs(self):
        tool = OfficialSourceSearchTool(search_tool=DummySearchTool(), max_domain_queries=2)

        domains = tool._select_domains("Google Earth Engine Landsat LST workflow")

        self.assertIn("developers.google.com/earth-engine", domains)

    def test_official_source_search_routes_general_python_query_to_python_docs(self):
        tool = OfficialSourceSearchTool(search_tool=DummySearchTool(), max_domain_queries=3)

        domains = tool._select_domains("Python asyncio task cancellation official docs")

        self.assertEqual(domains[0], "docs.python.org")

    def test_official_source_search_routes_openai_query_to_openai_docs(self):
        tool = OfficialSourceSearchTool(search_tool=DummySearchTool(), max_domain_queries=3)

        domains = tool._select_domains("OpenAI Responses API official documentation")

        self.assertIn("platform.openai.com", domains)

    async def test_official_source_search_uses_query_variants_and_broad_fallback(self):
        backend = FakeOfficialBackend({
            "Sentinel-2 MSI bands thermal infrared site:sentinels.copernicus.eu": [],
            "Sentinel-2 MSI bands thermal infrared site:sentiwiki.copernicus.eu": [],
            "Sentinel-2 MSI spectral bands thermal infrared official documentation site:sentinels.copernicus.eu": [],
            "Sentinel-2 MSI spectral bands thermal infrared official documentation site:sentiwiki.copernicus.eu": [],
            "Sentinel-2 MSI bands thermal infrared": [
                {
                    "title": "Blog result",
                    "url": "https://example.com/sentinel-2-msi",
                    "snippet": "Should be filtered out.",
                },
                {
                    "title": "Sentinel-2 MSI Instrument",
                    "url": "https://sentiwiki.copernicus.eu/web/s2-mission",
                    "snippet": "Official Sentinel-2 MSI mission documentation.",
                },
            ],
        })
        tool = OfficialSourceSearchTool(
            search_tool=backend,
            max_domain_queries=2,
            max_query_variants=2,
        )

        result = await tool.execute("Sentinel-2 MSI bands thermal infrared", top_n=3)

        urls = [item["url"] for item in result["results"]]
        self.assertIn("https://sentiwiki.copernicus.eu/web/s2-mission", urls)
        self.assertNotIn("https://example.com/sentinel-2-msi", urls)
        self.assertIn("Sentinel-2 MSI spectral bands thermal infrared official documentation", result["query_variants"])
        self.assertIn("Sentinel-2 MSI bands thermal infrared", result["attempted_queries"])

    async def test_official_source_search_rejects_wrong_sensor_even_on_official_domain(self):
        backend = FakeOfficialBackend({
            "Sentinel-2 MSI bands thermal infrared site:sentinels.copernicus.eu": [],
            "Sentinel-2 MSI bands thermal infrared site:sentiwiki.copernicus.eu": [],
            "Sentinel-2 MSI bands thermal infrared": [
                {
                    "title": "Land-surface temperature from Copernicus Sentinel-3",
                    "url": "https://www.esa.int/ESA_Multimedia/Images/2019/07/Land-surface_temperature_from_Copernicus_Sentinel-3",
                    "snippet": "Official ESA page about Sentinel-3 land surface temperature.",
                },
                {
                    "title": "Sentinel-2 MSI mission",
                    "url": "https://sentiwiki.copernicus.eu/web/s2-mission",
                    "snippet": "Official Sentinel-2 MSI mission documentation.",
                },
            ],
        })
        tool = OfficialSourceSearchTool(
            search_tool=backend,
            max_domain_queries=2,
            max_query_variants=1,
        )

        result = await tool.execute("Sentinel-2 MSI bands thermal infrared", top_n=3)

        urls = [item["url"] for item in result["results"]]
        self.assertNotIn(
            "https://www.esa.int/ESA_Multimedia/Images/2019/07/Land-surface_temperature_from_Copernicus_Sentinel-3",
            urls,
        )
        self.assertIn("https://sentiwiki.copernicus.eu/web/s2-mission", urls)

    async def test_official_source_search_respects_total_attempt_budget(self):
        tool = OfficialSourceSearchTool(
            search_tool=FakeOfficialBackend({}),
            max_domain_queries=4,
            max_query_variants=2,
            max_attempts=3,
        )

        result = await tool.execute("Sentinel-2 MSI bands thermal infrared", top_n=3)

        self.assertLessEqual(len(result["attempted_queries"]), 3)


if __name__ == "__main__":
    unittest.main()
