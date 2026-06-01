import unittest

from src.orchestrator.schemas import SourceTier
from src.utils.domain_tiers import authority_score, best_tier, classify_url


class SourceTierTests(unittest.TestCase):
    def test_classify_known_source_tiers(self):
        self.assertEqual(classify_url("https://nasa.gov/mission"), SourceTier.OFFICIAL)
        self.assertEqual(classify_url("https://lpdaac.usgs.gov/products/mod11a2v061/"), SourceTier.OFFICIAL)
        self.assertEqual(classify_url("https://openalex.org/W123"), SourceTier.ACADEMIC)
        self.assertEqual(classify_url("https://arxiv.org/abs/1706.03762"), SourceTier.ACADEMIC)
        self.assertEqual(classify_url("https://wikipedia.org/wiki/Landsat"), SourceTier.AUTHORITATIVE)
        self.assertEqual(classify_url("https://blog.csdn.net/example"), SourceTier.GENERAL)

    def test_best_tier_and_authority_score(self):
        urls = [
            "https://blog.csdn.net/example",
            "https://openalex.org/W123",
            "https://nasa.gov/landsat",
        ]
        self.assertEqual(best_tier(urls), SourceTier.OFFICIAL)
        self.assertGreater(
            authority_score("https://nasa.gov/landsat"),
            authority_score("https://blog.csdn.net/example"),
        )


if __name__ == "__main__":
    unittest.main()
