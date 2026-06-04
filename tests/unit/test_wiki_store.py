import asyncio
import tempfile
import unittest
from pathlib import Path

from src.tools.wiki_search import WikiSearchTool
from src.wiki import WikiIngestWorker, WikiStore


class FakeWikiPolicy:
    def __init__(self) -> None:
        self.tools = ["should be disabled"]
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        return {
            "content": """
{
  "entities": [
    {
      "title": "Landsat Surface Temperature",
      "category": "methods",
      "definition": "Landsat surface temperature is an LST evidence source grounded in thermal infrared observations.",
      "key_points": ["Uses Landsat TIRS evidence", "Supports urban heat analysis"],
      "details": "The report uses Landsat TIRS LST with NDVI and NDBI for urban thermal analysis.",
      "context": "Appears as the main method in the Wuhan LST study.",
      "related_pages": ["[[methods/ndvi]]"],
      "sources": ["Raw report states: Landsat TIRS LST uses NDVI and NDBI."],
      "confidence": 0.82
    },
    {
      "title": "Sentinel-2 Thermal Limitation",
      "category": "concepts",
      "definition": "Sentinel-2 MSI cannot directly retrieve LST because it has no thermal infrared band.",
      "key_points": ["Sentinel-2 has no thermal infrared band"],
      "details": "The source notes that Sentinel-2 MSI has no thermal infrared band.",
      "context": "Used as a constraint for sensor selection.",
      "related_pages": ["[[methods/landsat-surface-temperature]]"],
      "sources": ["Raw report states: Sentinel-2 MSI has no thermal infrared band."],
      "confidence": 0.9
    }
  ]
}
""",
            "usage": {"total_tokens": 100},
        }


class RepairingWikiPolicy:
    def __init__(self) -> None:
        self.tools = ["should be restored"]
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        if len(self.calls) == 1:
            return {"content": '{"entities": [{"title": "Broken", "category": "methods", "definition": "unterminated'}
        return {
            "content": """
{
  "entities": [
    {
      "title": "Repaired LST Workflow",
      "category": "methods",
      "definition": "A repaired LST workflow entity extracted from malformed JSON.",
      "key_points": ["Repair preserved a method entity"],
      "details": "The repaired payload is valid JSON.",
      "context": "Used to test JSON repair.",
      "related_pages": [],
      "sources": ["Raw report"],
      "confidence": 0.7
    }
  ]
}
"""
        }


class WikiStoreTests(unittest.TestCase):
    def test_initializes_user_scoped_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = WikiStore(root_dir=tmp, user_id="user/a")

            user_dir = Path(tmp) / "users" / "user_a"
            self.assertTrue((user_dir / "index.md").exists())
            for category in WikiStore.CATEGORIES:
                self.assertTrue((user_dir / category).is_dir())
            self.assertEqual(store.user_id, "user_a")

    def test_save_raw_updates_index_and_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = WikiStore(root_dir=tmp, user_id="u1")
            path = store.save_raw(
                title="Landsat LST Evidence",
                content="USGS Landsat Collection 2 includes surface temperature products.",
                metadata={"evidence_level": "evidence_backed", "source_tier": "official"},
            )

            self.assertTrue(Path(path).exists())
            self.assertIn("Landsat LST Evidence", store.get_index())
            context = store.get_context("Landsat surface temperature", max_chars=1000)
            self.assertIn("## Wiki Context", context)
            self.assertIn("source_tier=official", context)

    def test_save_raw_does_not_overwrite_same_title_in_fast_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = WikiStore(root_dir=tmp, user_id="u1")

            first = store.save_raw("Same title", "first", {})
            second = store.save_raw("Same title", "second", {})

            self.assertNotEqual(first, second)
            self.assertTrue(Path(first).exists())
            self.assertTrue(Path(second).exists())

    def test_find_similar_raw_detects_same_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = WikiStore(root_dir=tmp, user_id="u1")
            original = store.save_raw(
                "武汉城市扩张热环境研究",
                "Landsat LST NDVI NDBI 城市扩张 热环境 方法流程",
                {"query": "如何研究武汉城市扩张对地表热环境的影响"},
            )

            duplicate = store.find_similar_raw(
                title="如何研究武汉城市扩张对地表热环境的影响",
                content="Landsat LST NDVI NDBI 城市扩张 热环境 验证方案",
                metadata={"query": "如何研究武汉城市扩张对地表热环境的影响"},
            )

            self.assertIsNotNone(duplicate)
            self.assertEqual(Path(duplicate.path), Path(original))
            self.assertGreaterEqual(duplicate.score, 0.72)

    def test_search_returns_ranked_pages(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = WikiStore(root_dir=tmp, user_id="u1")
            store.save_raw("Landsat note", "Landsat thermal infrared surface temperature", {})
            store.save_raw("Policy note", "Urban planning policy", {})

            pages = store.search("Landsat temperature", top_k=1)

            self.assertEqual(len(pages), 1)
            self.assertIn("Landsat", pages[0].title)

    def test_save_page_writes_structured_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = WikiStore(root_dir=tmp, user_id="u1")

            path = store.save_page(
                category="sensors",
                title="Sentinel-2",
                content="Sentinel-2 MSI has no thermal infrared band.",
                metadata={"evidence_level": "evidence_backed"},
            )

            self.assertTrue(Path(path).exists())
            self.assertIn("Sentinel-2", store.get_index())
            self.assertIn("sensors/sentinel-2.md", store.get_index())

    def test_ingest_worker_extracts_structured_pages_from_raw_report_with_llm(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = WikiStore(root_dir=tmp, user_id="u1")
            raw_path = store.save_raw(
                "Wuhan LST Study",
                "## Method\nLandsat TIRS LST uses NDVI and NDBI. Sentinel-2 MSI has no thermal infrared band.",
                {"evidence_level": "evidence_backed", "source_tier": "official"},
            )
            policy = FakeWikiPolicy()
            worker = WikiIngestWorker(store, policy=policy, max_entities=4)

            result = worker.ingest_raw_report(
                raw_path=raw_path,
                title="Wuhan LST Study",
                content="## Method\nLandsat TIRS LST uses NDVI and NDBI. Sentinel-2 MSI has no thermal infrared band.",
                metadata={"query": "Wuhan LST Study", "evidence_level": "evidence_backed", "source_tier": "official"},
            )

            self.assertEqual(result["extractor"], "llm")
            self.assertEqual(result["entity_count"], 2)
            self.assertEqual(policy.tools, ["should be disabled"])
            self.assertIn("Simplified Chinese", policy.calls[0][0]["content"])
            self.assertTrue(list((Path(tmp) / "users" / "u1" / "methods").glob("*.md")))
            self.assertIn("Landsat", store.get_context("Landsat TIRS", max_chars=1200))
            first_page = Path(result["pages"][0]).read_text(encoding="utf-8")
            self.assertIn(f"[[raws/{Path(raw_path).stem}]]", first_page)
            self.assertIn("Raw report states", first_page)
            log = (Path(tmp) / "users" / "u1" / "log.md").read_text(encoding="utf-8")
            self.assertIn("Ingest Wuhan LST Study", log)

    def test_ingest_worker_requires_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = WikiStore(root_dir=tmp, user_id="u1")
            with self.assertRaisesRegex(ValueError, "requires an LLM policy"):
                WikiIngestWorker(store, policy=None)

    def test_ingest_worker_repairs_malformed_json_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = WikiStore(root_dir=tmp, user_id="u1")
            raw_path = store.save_raw("Repair demo", "LST workflow report", {})
            policy = RepairingWikiPolicy()
            worker = WikiIngestWorker(store, policy=policy, max_entities=2)

            result = worker.ingest_raw_report(
                raw_path=raw_path,
                title="Repair demo",
                content="LST workflow report",
                metadata={"query": "Repair demo"},
            )

            self.assertEqual(result["entity_count"], 1)
            self.assertEqual(len(policy.calls), 2)
            self.assertEqual(policy.tools, ["should be restored"])
            self.assertIn("Repaired LST Workflow", store.get_index())


class WikiSearchToolTests(unittest.TestCase):
    def test_tool_returns_wiki_evidence_metadata(self):
        async def run_case() -> dict:
            with tempfile.TemporaryDirectory() as tmp:
                store = WikiStore(root_dir=tmp, user_id="u1")
                store.save_raw(
                    "Sentinel limitation",
                    "Sentinel-2 has no thermal infrared band for direct LST retrieval.",
                    {"evidence_level": "evidence_backed", "source_tier": "official"},
                )
                tool = WikiSearchTool(store)
                return await tool.execute(query="Sentinel thermal LST", top_k=3)

        result = asyncio.run(run_case())

        self.assertEqual(result["source_type"], "wiki")
        self.assertEqual(result["evidence_level"], "evidence_backed")
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["results"][0]["source_tier"], "official")

    def test_tool_schema_is_function_calling_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            tool = WikiSearchTool(WikiStore(root_dir=tmp, user_id="u1"))

            schema = tool.get_openai_tool_schema()

            self.assertEqual(schema["type"], "function")
            self.assertEqual(schema["function"]["name"], "wiki_search")
            self.assertIn("query", schema["function"]["parameters"]["required"])


if __name__ == "__main__":
    unittest.main()
