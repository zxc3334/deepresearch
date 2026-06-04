import unittest

from scripts.run_geo_integration_demo import apply_adapter_override
from src.core.domain_profiles import resolve_domain_profile


class RunGeoIntegrationDemoTests(unittest.TestCase):
    def test_apply_adapter_override_sets_explicit_adapter(self):
        config = apply_adapter_override(
            {"domain_adapter": {"mode": "general"}},
            adapter="geo_remote_sensing",
            query="plain query",
        )

        self.assertEqual(config["domain_adapter"]["mode"], "geo_remote_sensing")
        self.assertEqual(config["query"], "plain query")

    def test_apply_adapter_override_preserves_configured_adapter_when_argument_absent(self):
        config = apply_adapter_override(
            {"domain_adapter": {"mode": "geo_remote_sensing"}},
            adapter="",
            query="plain query",
        )

        self.assertEqual(config["domain_adapter"]["mode"], "geo_remote_sensing")

    def test_auto_adapter_uses_runtime_query(self):
        config = apply_adapter_override(
            {"domain_adapter": {"mode": "general"}},
            adapter="auto",
            query="Use Landsat and Sentinel data for urban heat island LST analysis.",
        )

        profile = resolve_domain_profile(config)

        self.assertEqual(profile["name"], "geo_remote_sensing")


if __name__ == "__main__":
    unittest.main()
