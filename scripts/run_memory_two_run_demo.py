#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a focused two-run demo for scoped long-term memory.

This script validates memory behavior without relying on a full LLM research
run. It writes seed memories with explicit scopes, rebuilds stores as if they
were separate runs/sessions/users, and verifies retrieval boundaries.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.core.runner import load_config
from src.memory.long_term import MemoryEntry
from src.memory.memory_store import SharedMemoryStore


DEFAULT_QUERY = "Landsat LST 武汉 城市热岛 Sentinel-2 thermal limitation"


class DemoEmbedder:
    """Small deterministic embedder for fast, stable memory demo checks."""

    dim = 8

    def encode(self, text: str) -> list[float]:
        lowered = text.lower()
        features = [
            "landsat",
            "lst",
            "sentinel",
            "thermal",
            "武汉",
            "城市",
            "failure",
            "cloud",
        ]
        return [1.0 if feature in lowered else 0.0 for feature in features]


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_entry(entry_id: str, claim: str, *, scope: str, evidence_level: str) -> MemoryEntry:
    return MemoryEntry(
        entry_id=entry_id,
        claim=claim,
        source=f"demo:{entry_id}",
        confidence=0.86,
        agent_id=entry_id,
        timestamp=time.time(),
        evidence_type="secondary" if evidence_level == "evidence_backed" else "inference",
        embedding=[],
        topic="武汉城市热岛 Landsat LST",
        metadata={
            "scope": scope,
            "evidence_level": evidence_level,
            "source_tier": "academic" if evidence_level == "evidence_backed" else "general",
            "task_id": entry_id,
        },
    )


def create_store(
    db_path: Path,
    *,
    user_id: str,
    session_id: str,
    run_id: str,
    include_global: bool = True,
) -> SharedMemoryStore:
    return SharedMemoryStore(
        db_path=str(db_path),
        embedder=DemoEmbedder(),
        user_id=user_id,
        session_id=session_id,
        run_id=run_id,
        include_global=include_global,
    )


def summarize_results(results) -> list[dict[str, Any]]:
    return [
        {
            "entry_id": entry.entry_id,
            "similarity": round(score, 4),
            "scope": entry.metadata.get("scope", ""),
            "user_id": entry.metadata.get("user_id", ""),
            "session_id": entry.metadata.get("session_id", ""),
            "run_id": entry.metadata.get("run_id", ""),
            "evidence_level": entry.metadata.get("evidence_level", ""),
            "claim": entry.claim,
        }
        for entry, score in results
    ]


def ids(results) -> set[str]:
    return {entry.entry_id for entry, _ in results}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a focused two-run scoped memory demo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/geo_mvp.yaml", help="Config used only to resolve memory defaults.")
    parser.add_argument("--output-dir", default="", help="Output directory. Defaults to outputs/memory_two_run_demo_<timestamp>.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Query used for memory retrieval.")
    parser.add_argument("--user-id", default="demo-user", help="Primary user id.")
    parser.add_argument("--other-user-id", default="other-user", help="Isolation control user id.")
    parser.add_argument("--session-id", default="", help="Primary session id.")
    parser.add_argument("--other-session-id", default="", help="Isolation control session id.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stamp = timestamp()
    output_dir = Path(args.output_dir or f"outputs/memory_two_run_demo_{stamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    db_path = output_dir / "memory_demo.db"
    session_id = args.session_id or f"session-a-{stamp}"
    other_session_id = args.other_session_id or f"session-b-{stamp}"

    # Run 1: write memories into three scopes.
    run1 = create_store(
        db_path,
        user_id=args.user_id,
        session_id=session_id,
        run_id=f"run-1-{stamp}",
        include_global=True,
    )
    run1.put(make_entry(
        "run1-session-landsat",
        "Session memory: Wuhan Landsat LST workflow should aggregate land cover to thermal effective scale.",
        scope="session",
        evidence_level="evidence_backed",
    ))
    run1.put(make_entry(
        "run1-user-sentinel-limit",
        "User memory: Sentinel-2 has no thermal infrared band and cannot directly retrieve LST.",
        scope="user",
        evidence_level="rejected",
    ))
    run1.put(make_entry(
        "run1-global-cloud-risk",
        "Global memory: Cloud contamination is a key risk for summer Landsat LST studies.",
        scope="global",
        evidence_level="evidence_backed",
    ))

    # Run 2: same user + same session should see session, user, and global memories.
    same_session = create_store(
        db_path,
        user_id=args.user_id,
        session_id=session_id,
        run_id=f"run-2-same-session-{stamp}",
        include_global=True,
    )
    same_session_results = same_session.query_by_similarity(args.query, top_k=10, min_sim=0.1)
    same_session_context = same_session.get_context_for_query(args.query, max_tokens=1000)

    # Same user + different session should see user and global, but not session memory.
    different_session = create_store(
        db_path,
        user_id=args.user_id,
        session_id=other_session_id,
        run_id=f"run-2-different-session-{stamp}",
        include_global=True,
    )
    different_session_results = different_session.query_by_similarity(args.query, top_k=10, min_sim=0.1)

    # Different user should see only global memory.
    different_user = create_store(
        db_path,
        user_id=args.other_user_id,
        session_id=other_session_id,
        run_id=f"run-2-different-user-{stamp}",
        include_global=True,
    )
    different_user_results = different_user.query_by_similarity(args.query, top_k=10, min_sim=0.1)

    # Same user + different session with global disabled should only see user memory.
    no_global = create_store(
        db_path,
        user_id=args.user_id,
        session_id=other_session_id,
        run_id=f"run-2-no-global-{stamp}",
        include_global=False,
    )
    no_global_results = no_global.query_by_similarity(args.query, top_k=10, min_sim=0.1)

    checks = {
        "same_session_reads_session_user_global": {
            "passed": {
                "run1-session-landsat",
                "run1-user-sentinel-limit",
                "run1-global-cloud-risk",
            }.issubset(ids(same_session_results)),
        },
        "different_session_excludes_session_memory": {
            "passed": "run1-session-landsat" not in ids(different_session_results)
            and "run1-user-sentinel-limit" in ids(different_session_results)
            and "run1-global-cloud-risk" in ids(different_session_results),
        },
        "different_user_excludes_user_and_session_memory": {
            "passed": ids(different_user_results) == {"run1-global-cloud-risk"},
        },
        "include_global_false_excludes_global_memory": {
            "passed": "run1-global-cloud-risk" not in ids(no_global_results)
            and "run1-user-sentinel-limit" in ids(no_global_results),
        },
    }
    all_passed = all(item["passed"] for item in checks.values())

    summary = {
        "passed": all_passed,
        "config": args.config,
        "config_memory_db_path": config.get("memory", {}).get("db_path", ""),
        "demo_db_path": str(db_path),
        "query": args.query,
        "user_id": args.user_id,
        "session_id": session_id,
        "other_user_id": args.other_user_id,
        "other_session_id": other_session_id,
        "checks": checks,
        "same_session": {
            "results": summarize_results(same_session_results),
            "context_preview": same_session_context[:1600],
        },
        "different_session": {
            "results": summarize_results(different_session_results),
        },
        "different_user": {
            "results": summarize_results(different_user_results),
        },
        "no_global": {
            "results": summarize_results(no_global_results),
        },
    }

    summary_path = output_dir / "memory_two_run_summary.json"
    write_json(summary_path, summary)

    print("=== Memory Two-Run Demo ===")
    print(f"Passed: {all_passed}")
    print(f"Summary: {summary_path}")
    print(f"DB:      {db_path}")
    print("")
    for name, item in checks.items():
        print(f"- {name}: {'PASS' if item['passed'] else 'FAIL'}")
    print("")
    print("Same session retrieved:")
    for item in summary["same_session"]["results"]:
        print(f"  - {item['entry_id']} [{item['scope']}] sim={item['similarity']}")

    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
