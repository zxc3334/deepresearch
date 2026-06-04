#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run one end-to-end DeepResearch integration demo."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime
import json
import logging
from pathlib import Path
import time
from typing import Any

from scripts.render_trace import render_trace_report
from src.core.runner import initialize_modules, load_config, run_research, save_report, setup_logging


DEFAULT_GENERAL_QUERY = (
    "Python asyncio 中 task cancellation 的官方语义是什么？"
    "请结合官方文档和实践风险说明。"
)

DEFAULT_GEO_QUERY = (
    "如何研究 2018-2024 年武汉城市扩张对地表热环境的影响？"
    "请给出数据选择、方法流程、验证方案和潜在风险。"
)

PRESET_DEFAULTS = {
    "general": {
        "query": DEFAULT_GENERAL_QUERY,
        "config": "configs/default.yaml",
        "adapter": "general",
        "output_prefix": "general_demo",
        "user_id": "demo-general",
        "session_prefix": "general-demo",
        "run_prefix": "general-demo",
    },
    "geo": {
        "query": DEFAULT_GEO_QUERY,
        "config": "configs/geo_real_search.yaml",
        "adapter": "geo_remote_sensing",
        "output_prefix": "geo_demo",
        "user_id": "demo-geo",
        "session_prefix": "geo-demo",
        "run_prefix": "geo-demo",
    },
}


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_progress_callback(progress_path: Path):
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    def progress_callback(event: dict[str, Any]) -> None:
        safe_event = {key: _json_safe(value) for key, value in event.items()}
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(safe_event, ensure_ascii=False) + "\n")

        event_type = str(event.get("event", ""))
        if event_type in {
            "planning_start",
            "planning_end",
            "task_start",
            "task_end",
            "tool_call_start",
            "tool_call_result",
            "tool_call_error",
        }:
            task = event.get("task_id", "")
            tool = event.get("tool_name", "")
            status = event.get("status", "")
            print(f"[progress] {event_type} task={task} tool={tool} status={status}")

    return progress_callback


def inspect_memory(modules: dict[str, Any], query: str, session_id: str) -> dict[str, Any]:
    store = modules.get("memory_store")
    if store is None:
        return {"enabled": False}

    session_entries = store.lt.get_entries_by_session(session_id)
    context = store.get_context_for_query(query, max_tokens=1000)
    scoped_counts: dict[str, int] = {}
    evidence_counts: dict[str, int] = {}
    for entry in session_entries:
        scope = str(entry.metadata.get("scope", "unknown"))
        scoped_counts[scope] = scoped_counts.get(scope, 0) + 1
        evidence_level = str(entry.metadata.get("evidence_level", "") or "unknown")
        evidence_counts[evidence_level] = evidence_counts.get(evidence_level, 0) + 1

    return {
        "enabled": True,
        "user_id": store.user_id,
        "session_id": store.session_id,
        "run_id": store.run_id,
        "include_global": store.include_global,
        "session_entry_count": len(session_entries),
        "scope_counts": scoped_counts,
        "evidence_level_counts": evidence_counts,
        "retrieved_context_chars": len(context),
        "retrieved_context_preview": context[:1200],
    }


def inspect_trace(trace_path: Path) -> dict[str, Any]:
    if not trace_path.exists():
        return {"available": False}

    events: list[dict[str, Any]] = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))

    tool_calls = [event for event in events if event.get("event") == "tool_call"]
    prefetch_results = [event for event in events if event.get("event") == "external_prefetch_result"]
    evidence_items = [event for event in events if event.get("event") == "evidence_item"]
    return {
        "available": True,
        "event_count": len(events),
        "event_counts": dict(Counter(str(event.get("event", "")) for event in events)),
        "tool_call_counts": dict(Counter(str(event.get("tool", "")) for event in tool_calls)),
        "external_prefetch_tool_counts": dict(Counter(str(event.get("tool", "")) for event in prefetch_results)),
        "evidence_level_counts": dict(Counter(str(event.get("level", "")) for event in evidence_items)),
        "evidence_source_tier_counts": dict(Counter(str(event.get("source_tier", "")) for event in evidence_items)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a DeepResearch end-to-end integration demo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", default="geo", choices=["general", "geo"], help="Demo preset.")
    parser.add_argument("--query", default="", help="Research question. Defaults to the selected preset query.")
    parser.add_argument("--config", default="", help="YAML config path. Defaults to the selected preset config.")
    parser.add_argument(
        "--adapter",
        default="",
        help="Domain adapter: auto, general, geo_remote_sensing, or a user adapter id.",
    )
    parser.add_argument(
        "--user-adapters-dir",
        default="",
        help="Directory containing user adapter YAML files.",
    )
    parser.add_argument("--output-dir", default="", help="Output directory. Defaults to outputs/<preset>_<timestamp>.")
    parser.add_argument("--user-id", default="", help="Memory user scope. Defaults to the selected preset user.")
    parser.add_argument("--session-id", default="", help="Memory session scope. Defaults to <preset>-demo-<timestamp>.")
    parser.add_argument("--run-id", default="", help="Run id for trace and memory metadata. Defaults to <preset>-demo-<timestamp>.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def apply_adapter_override(
    config: dict[str, Any],
    *,
    adapter: str = "",
    user_adapters_dir: str = "",
    query: str = "",
) -> dict[str, Any]:
    adapter_cfg = dict(config.get("domain_adapter", {}) or {})
    if adapter:
        adapter_cfg["mode"] = adapter
    adapter_cfg.setdefault("mode", "general")
    if user_adapters_dir:
        adapter_cfg["user_adapters_dir"] = user_adapters_dir
    adapter_cfg.setdefault("user_adapters_dir", "data/user_adapters")
    config["domain_adapter"] = adapter_cfg
    if query:
        config["query"] = query
    return config


def main() -> None:
    args = parse_args()
    stamp = _timestamp()
    preset = PRESET_DEFAULTS[args.preset]
    query = args.query or preset["query"]
    config_path = args.config or preset["config"]
    adapter = args.adapter or preset["adapter"]
    user_id = args.user_id or preset["user_id"]
    output_dir = Path(args.output_dir or f"outputs/{preset['output_prefix']}_{stamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    session_id = args.session_id or f"{preset['session_prefix']}-{stamp}"
    run_id = args.run_id or f"{preset['run_prefix']}-{stamp}"
    trace_path = output_dir / "trace.jsonl"
    progress_path = output_dir / "progress_events.jsonl"
    summary_path = output_dir / "integration_summary.json"

    setup_logging(args.log_level)
    logger = logging.getLogger("integration_demo")

    logger.info("Loading config: %s", config_path)
    config = load_config(config_path)
    config = apply_adapter_override(
        config,
        adapter=adapter,
        user_adapters_dir=args.user_adapters_dir,
        query=query,
    )
    config["_trace_path"] = str(trace_path)
    config["_progress_callback"] = build_progress_callback(progress_path)
    config["_user_id"] = user_id
    config["_session_id"] = session_id
    config["_run_id"] = run_id

    started = time.time()
    modules = initialize_modules(config, user_id=user_id, session_id=session_id, run_id=run_id)
    report_text = asyncio.run(run_research(query, config, modules))
    elapsed = time.time() - started

    report_path = Path(save_report(report_text, query, str(output_dir)))
    trace_report_path = ""
    try:
        trace_report_path = render_trace_report(trace_path)
    except Exception as exc:
        logger.warning("Trace HTML render failed: %s", exc)

    memory_summary = inspect_memory(modules, query, session_id)
    trace_summary = inspect_trace(trace_path)
    summary = {
        "preset": args.preset,
        "adapter": config.get("domain_adapter", {}).get("mode", ""),
        "query": query,
        "config": config_path,
        "output_dir": str(output_dir),
        "report_path": str(report_path),
        "trace_path": str(trace_path),
        "trace_report_path": trace_report_path,
        "progress_path": str(progress_path),
        "user_id": user_id,
        "session_id": session_id,
        "run_id": run_id,
        "elapsed_seconds": round(elapsed, 3),
        "memory": memory_summary,
        "trace_summary": trace_summary,
    }
    _write_json(summary_path, summary)

    print("\n=== DeepResearch Integration Demo ===")
    print(f"Adapter:        {summary['adapter']}")
    print(f"Report:         {report_path}")
    print(f"Trace JSONL:    {trace_path}")
    print(f"Trace HTML:     {trace_report_path or 'not generated'}")
    print(f"Progress JSONL: {progress_path}")
    print(f"Summary JSON:   {summary_path}")
    print(f"Memory entries in session: {memory_summary.get('session_entry_count', 0)}")
    print(f"Tool calls: {trace_summary.get('tool_call_counts', {})}")
    print(f"External prefetch tools: {trace_summary.get('external_prefetch_tool_counts', {})}")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
