import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.create_domain_adapter import (
    build_adapter_manually,
    build_adapter_with_llm,
    collect_manual_fields,
    save_adapter_yaml,
)


class FakePolicy:
    def __init__(self, content: str):
        self.content = content

    def __call__(self, messages):
        return {"role": "assistant", "content": self.content}


class CreateDomainAdapterTests(unittest.TestCase):
    def test_build_adapter_with_llm_normalizes_json(self):
        payload = build_adapter_with_llm(
            policy=FakePolicy(
                """
                {
                  "keywords": ["policy", "IPCC"],
                  "recommended_tools": ["official_source_search", "paper_search"],
                  "preferred_official_domains": ["ipcc.ch"],
                  "evidence_checklist": ["Prefer official sources."],
                  "output_sections": ["Policy table"]
                }
                """
            ),
            name="climate_policy",
            display_name="Climate Policy",
            description="Climate policy research.",
        )

        self.assertEqual(payload["name"], "climate_policy")
        self.assertEqual(payload["extends"], "general")
        self.assertIn("official_source_search", payload["exposed_tools"])
        self.assertIn("ipcc.ch", payload["preferred_official_domains"])

    def test_build_adapter_manually_uses_recommended_tools_as_exposed_tools(self):
        payload = build_adapter_manually(
            name="legal_research",
            display_name="Legal Research",
            description="Legal and regulatory research.",
            keywords=["law"],
            preferred_domains=["gov"],
            recommended_tools=["official_source_search"],
            evidence_rules=["Prefer statutes."],
            output_sections=["Legal basis"],
        )

        self.assertEqual(payload["exposed_tools"], ["official_source_search"])
        self.assertEqual(payload["recommended_tools"], ["official_source_search"])

    def test_save_adapter_yaml_validates_declarative_schema(self):
        payload = build_adapter_manually(
            name="medical_research",
            display_name="Medical Research",
            description="Medical literature research.",
            keywords=["medicine"],
            preferred_domains=["nih.gov"],
            recommended_tools=["paper_search"],
            evidence_rules=["Prefer peer-reviewed evidence."],
            output_sections=["Evidence table"],
        )

        with TemporaryDirectory() as tmpdir:
            path = save_adapter_yaml(payload, Path(tmpdir))

            self.assertTrue(path.exists())
            self.assertEqual(path.name, "medical_research.yaml")

    def test_save_adapter_yaml_refuses_overwrite_by_default(self):
        payload = build_adapter_manually(
            name="finance_research",
            display_name="Finance Research",
            description="Finance research.",
            keywords=[],
            preferred_domains=[],
            recommended_tools=["web_search"],
            evidence_rules=[],
            output_sections=[],
        )

        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_adapter_yaml(payload, output_dir)

            with self.assertRaises(FileExistsError):
                save_adapter_yaml(payload, output_dir)

    def test_collect_manual_fields_supports_non_interactive_mode(self):
        fields = collect_manual_fields(
            Namespace(
                non_interactive=True,
                keywords="climate, policy",
                preferred_domains="ipcc.ch, unfccc.int",
                recommended_tools="official_source_search, paper_search",
                evidence_rules="Prefer official documents.",
                output_sections="Policy table",
            )
        )

        self.assertEqual(fields["keywords"], ["climate", "policy"])
        self.assertEqual(fields["preferred_domains"], ["ipcc.ch", "unfccc.int"])
        self.assertIn("official_source_search", fields["recommended_tools"])


if __name__ == "__main__":
    unittest.main()
