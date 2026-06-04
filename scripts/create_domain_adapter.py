#!/usr/bin/env python3
"""Create a user declarative domain adapter.

Level 1 uses the configured LLM to draft a safe YAML adapter from a short
description. If LLM generation is unavailable or invalid, Level 2 collects
fields directly from the user.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.runner import load_config  # noqa: E402
from src.domain_adapters.declarative import DeclarativeDomainAdapter  # noqa: E402
from src.models.model_factory import LLMModelFactory  # noqa: E402


DEFAULT_TOOLS = [
    "wiki_search",
    "web_search",
    "official_source_search",
    "official_doc_fetcher",
    "paper_search",
    "browser",
    "calculator",
    "code_sandbox",
]

ADAPTER_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a user Domain Adapter YAML.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Project config used for LLM policy routing.")
    parser.add_argument("--output-dir", default="", help="Adapter directory. Defaults to domain_adapter.user_adapters_dir.")
    parser.add_argument("--name", default="", help="Adapter id, e.g. climate_policy.")
    parser.add_argument("--display-name", default="", help="Human-readable adapter name.")
    parser.add_argument("--description", default="", help="One-sentence domain description.")
    parser.add_argument("--keywords", default="", help="Comma-separated trigger keywords for manual mode.")
    parser.add_argument("--preferred-domains", default="", help="Comma-separated trusted domains for manual mode.")
    parser.add_argument("--recommended-tools", default="", help="Comma-separated tools for manual mode.")
    parser.add_argument("--evidence-rules", default="", help="Comma-separated evidence rules for manual mode.")
    parser.add_argument("--output-sections", default="", help="Comma-separated output sections for manual mode.")
    parser.add_argument("--manual", action="store_true", help="Skip LLM drafting and use manual field mode.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing adapter file.")
    parser.add_argument("--non-interactive", action="store_true", help="Fail instead of prompting for missing fields.")
    return parser.parse_args()


def build_adapter_with_llm(
    *,
    policy: Any,
    name: str,
    display_name: str,
    description: str,
) -> dict[str, Any]:
    """Ask an LLM to draft a declarative adapter dict."""
    messages = [
        {
            "role": "system",
            "content": (
                "You generate safe declarative Domain Adapter JSON. "
                "Return JSON only. No markdown. No comments. "
                "Allowed keys: name, display_name, description, extends, keywords, "
                "exposed_tools, recommended_tools, prompt_sections, preferred_official_domains, "
                "evidence_checklist, output_sections. "
                "All list fields must contain strings only. "
                "Do not invent API keys. Do not include executable code."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Adapter id: {name}\n"
                f"Display name: {display_name}\n"
                f"Domain description: {description}\n\n"
                f"Available tools: {', '.join(DEFAULT_TOOLS)}\n"
                "Generate a practical first draft. Use extends='general'. "
                "Choose 4-10 keywords, 3-6 recommended tools, 2-5 trusted official domains if known, "
                "2-5 evidence checklist rules, and 2-5 output sections."
            ),
        },
    ]
    response = policy(messages)
    content = str(response.get("content", "") if isinstance(response, dict) else getattr(response, "content", ""))
    payload = _parse_json_object(content)
    payload["name"] = name
    payload.setdefault("display_name", display_name)
    payload.setdefault("description", description)
    payload.setdefault("extends", "general")
    return normalize_adapter_payload(payload)


def build_adapter_manually(
    *,
    name: str,
    display_name: str,
    description: str,
    keywords: list[str],
    preferred_domains: list[str],
    recommended_tools: list[str],
    evidence_rules: list[str],
    output_sections: list[str],
) -> dict[str, Any]:
    return normalize_adapter_payload(
        {
            "name": name,
            "display_name": display_name,
            "description": description,
            "extends": "general",
            "keywords": keywords,
            "exposed_tools": recommended_tools,
            "recommended_tools": recommended_tools,
            "prompt_sections": [],
            "preferred_official_domains": preferred_domains,
            "evidence_checklist": evidence_rules,
            "output_sections": output_sections,
        }
    )


def normalize_adapter_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["name"] = _normalize_adapter_id(str(normalized.get("name", "")))
    normalized["display_name"] = str(normalized.get("display_name") or normalized["name"]).strip()
    normalized["description"] = str(normalized.get("description") or "").strip()
    normalized["extends"] = str(normalized.get("extends") or "general").strip()

    for key in (
        "keywords",
        "exposed_tools",
        "recommended_tools",
        "prompt_sections",
        "preferred_official_domains",
        "evidence_checklist",
        "output_sections",
    ):
        normalized[key] = _clean_string_list(normalized.get(key, []))

    if not normalized["exposed_tools"]:
        normalized["exposed_tools"] = list(normalized["recommended_tools"])
    if not normalized["recommended_tools"]:
        normalized["recommended_tools"] = list(normalized["exposed_tools"])
    return normalized


def save_adapter_yaml(payload: dict[str, Any], output_dir: Path, *, overwrite: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{payload['name']}.yaml"
    if path.exists() and not overwrite:
        raise FileExistsError(f"Adapter already exists: {path}")

    text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    path.write_text(text, encoding="utf-8")
    DeclarativeDomainAdapter.from_file(path)
    return path


def create_policy_from_config(config_path: str):
    config = load_config(config_path)
    factory = LLMModelFactory(config.get("model", {}))
    return factory.create_policy("domain_adapter_creator")


def prompt_missing_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.non_interactive:
        missing = [
            name
            for name, value in {
                "--name": args.name,
                "--display-name": args.display_name,
                "--description": args.description,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(f"Missing required non-interactive arguments: {', '.join(missing)}")

    if not args.name:
        args.name = input("Adapter id, e.g. climate_policy: ").strip()
    if not args.display_name:
        args.display_name = input("Display name: ").strip()
    if not args.description:
        args.description = input("One-sentence domain description: ").strip()
    return args


def collect_manual_fields(args: argparse.Namespace) -> dict[str, list[str]]:
    if args.non_interactive:
        recommended_tools = _split_csv(args.recommended_tools) or ["web_search", "official_source_search"]
        return {
            "keywords": _split_csv(args.keywords),
            "preferred_domains": _split_csv(args.preferred_domains),
            "recommended_tools": recommended_tools,
            "evidence_rules": _split_csv(args.evidence_rules),
            "output_sections": _split_csv(args.output_sections),
        }

    print("\nManual Level 2 fields. Use comma-separated values. Leave blank if not needed.")
    return {
        "keywords": _split_csv(input("Keywords: ")),
        "preferred_domains": _split_csv(input("Preferred official domains: ")),
        "recommended_tools": _split_csv(input(f"Recommended tools ({', '.join(DEFAULT_TOOLS)}): ")),
        "evidence_rules": _split_csv(input("Evidence rules: ")),
        "output_sections": _split_csv(input("Output sections: ")),
    }


def main() -> None:
    args = prompt_missing_args(parse_args())
    name = _normalize_adapter_id(args.name)
    output_dir = Path(args.output_dir or _default_adapter_dir(args.config))

    payload: dict[str, Any] | None = None
    if not args.manual:
        try:
            policy = create_policy_from_config(args.config)
            payload = build_adapter_with_llm(
                policy=policy,
                name=name,
                display_name=args.display_name,
                description=args.description,
            )
            print("[adapter] LLM draft generated.")
        except Exception as exc:
            print(f"[adapter] LLM draft failed: {exc}")
            if args.non_interactive:
                raise

    if payload is None:
        fields = collect_manual_fields(args)
        payload = build_adapter_manually(
            name=name,
            display_name=args.display_name,
            description=args.description,
            keywords=fields["keywords"],
            preferred_domains=fields["preferred_domains"],
            recommended_tools=fields["recommended_tools"],
            evidence_rules=fields["evidence_rules"],
            output_sections=fields["output_sections"],
        )

    path = save_adapter_yaml(payload, output_dir, overwrite=args.overwrite)
    print(f"[adapter] Saved: {path}")
    print(f"[adapter] Use with: --adapter {payload['name']}")


def _default_adapter_dir(config_path: str) -> str:
    config = load_config(config_path)
    adapter_cfg = config.get("domain_adapter", {}) or {}
    return str(adapter_cfg.get("user_adapters_dir") or "data/user_adapters")


def _parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("LLM output must be a JSON object.")
    return payload


def _normalize_adapter_id(value: str) -> str:
    name = value.strip().lower().replace("-", "_").replace(" ", "_")
    name = re.sub(r"[^a-z0-9_]", "", name)
    if not ADAPTER_ID_RE.match(name):
        raise ValueError(f"Invalid adapter id: {value!r}. Use lowercase letters, numbers, and underscores.")
    return name


def _clean_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = _split_csv(value)
    if not isinstance(value, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
