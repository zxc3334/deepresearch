import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from src.core.domain_profiles import resolve_domain_profile
from src.domain_adapters import AdapterRegistry, DeclarativeAdapterError, DeclarativeDomainAdapter


class DomainAdapterTests(unittest.TestCase):
    def test_builtin_adapters_are_registered(self):
        names = {adapter.name for adapter in AdapterRegistry.list_adapters()}

        self.assertIn("general", names)
        self.assertIn("geo_remote_sensing", names)

    def test_explicit_adapter_resolution(self):
        adapter = AdapterRegistry.resolve(mode="geo_remote_sensing")

        self.assertEqual(adapter.name, "geo_remote_sensing")
        self.assertIn("code_sandbox", adapter.build_profile()["exposed_tools"])

    def test_auto_resolution_selects_geo_for_remote_sensing_query(self):
        adapter = AdapterRegistry.resolve(
            mode="auto",
            query="How to use Landsat and Sentinel data to study LST and urban heat island change?",
        )

        self.assertEqual(adapter.name, "geo_remote_sensing")

    def test_auto_resolution_selects_geo_for_chinese_remote_sensing_query(self):
        adapter = AdapterRegistry.resolve(
            mode="auto",
            query="如何用遥感影像和地表温度分析城市热岛变化？",
        )

        self.assertEqual(adapter.name, "geo_remote_sensing")

    def test_auto_resolution_falls_back_to_general(self):
        adapter = AdapterRegistry.resolve(
            mode="auto",
            query="What is the official behavior of Python asyncio task cancellation?",
        )

        self.assertEqual(adapter.name, "general")

    def test_profile_returns_are_isolated_from_adapter_state(self):
        adapter = AdapterRegistry.resolve(mode="geo_remote_sensing")
        profile = adapter.build_profile()
        profile["exposed_tools"].append("mutated_tool")

        fresh_profile = adapter.build_profile()
        self.assertNotIn("mutated_tool", fresh_profile["exposed_tools"])

    def test_unknown_adapter_is_rejected(self):
        with self.assertRaises(ValueError):
            AdapterRegistry.resolve(mode="unknown_domain")

    def test_declarative_adapter_loads_from_yaml(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "climate_policy.yaml"
            path.write_text(
                "\n".join(
                    [
                        "name: climate_policy",
                        "display_name: Climate Policy",
                        "description: Policy and standards research.",
                        "keywords:",
                        "  - climate policy",
                        "  - IPCC",
                        "exposed_tools:",
                        "  - official_source_search",
                        "recommended_tools:",
                        "  - official_source_search",
                        "prompt_sections:",
                        "  - policy_scope_rules",
                        "preferred_official_domains:",
                        "  - ipcc.ch",
                        "evidence_checklist:",
                        "  - Prefer official policy documents and standards bodies.",
                        "output_sections:",
                        "  - Policy source table",
                    ]
                ),
                encoding="utf-8",
            )

            adapter = DeclarativeDomainAdapter.from_file(path)

        self.assertEqual(adapter.name, "climate_policy")
        profile = adapter.build_profile()
        self.assertEqual(profile["extends"], "general")
        self.assertIn("ipcc.ch", profile["preferred_official_domains"])
        self.assertIn("Policy source table", profile["output_sections"])

    def test_declarative_adapter_rejects_unsupported_fields(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "unsafe.yaml"
            path.write_text("name: unsafe\npython: import os\n", encoding="utf-8")

            with self.assertRaises(DeclarativeAdapterError):
                DeclarativeDomainAdapter.from_file(path)

    def test_domain_profile_can_resolve_user_adapter_from_configured_directory(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "climate_policy.yaml"
            path.write_text(
                "\n".join(
                    [
                        "name: climate_policy",
                        "display_name: Climate Policy",
                        "keywords:",
                        "  - IPCC",
                        "exposed_tools:",
                        "  - official_source_search",
                        "recommended_tools:",
                        "  - official_source_search",
                        "preferred_official_domains:",
                        "  - ipcc.ch",
                    ]
                ),
                encoding="utf-8",
            )
            profile = resolve_domain_profile(
                {
                    "domain_adapter": {
                        "mode": "climate_policy",
                        "user_adapters_dir": tmpdir,
                    }
                }
            )

        self.assertEqual(profile["name"], "climate_policy")
        self.assertIn("official_source_search", profile["exposed_tools"])
        self.assertIn("ipcc.ch", profile["preferred_official_domains"])

    def test_user_adapter_reload_replaces_previous_user_adapter(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "climate_policy.yaml"
            path.write_text(
                "\n".join(
                    [
                        "name: climate_policy",
                        "display_name: Climate Policy",
                        "preferred_official_domains:",
                        "  - ipcc.ch",
                    ]
                ),
                encoding="utf-8",
            )
            resolve_domain_profile({"domain_adapter": {"mode": "climate_policy", "user_adapters_dir": tmpdir}})

            path.write_text(
                "\n".join(
                    [
                        "name: climate_policy",
                        "display_name: Climate Policy",
                        "preferred_official_domains:",
                        "  - unfccc.int",
                    ]
                ),
                encoding="utf-8",
            )
            profile = resolve_domain_profile(
                {"domain_adapter": {"mode": "climate_policy", "user_adapters_dir": tmpdir}}
            )

        self.assertIn("unfccc.int", profile["preferred_official_domains"])
        self.assertNotIn("ipcc.ch", profile["preferred_official_domains"])

    def test_user_adapter_cannot_override_builtin_adapter(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "general.yaml"
            path.write_text("name: general\ndisplay_name: Fake General\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                resolve_domain_profile({"domain_adapter": {"mode": "general", "user_adapters_dir": tmpdir}})


if __name__ == "__main__":
    unittest.main()
