#!/usr/bin/env python3
"""Run the frozen v0.2 read-only probes over 48 matched-development trajectories."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[3]
CASE_ROOT = ROOT / "cases/science/mendelian_genetics_known_plant_aer"
DEFAULT_CONFIG = CASE_ROOT / "construction/matched-probe-study.v0.2-development.json"
DEFAULT_SCHEMA = CASE_ROOT / "construction/matched-probe-output.schema.v0.2-development.json"
DEFAULT_SOURCE = (
    ROOT
    / "artifacts/aer_pea_case/gate-e-matched-development-v0.4.2-development-0001"
)
DEFAULT_OUTPUT = ROOT / "artifacts/aer_pea_case/matched-probes-v0.2-development-0001"
sys.path.insert(0, str(ROOT / "src"))

from aer_bench.codex_runner import CodexRunConfig, CodexRunner  # noqa: E402
from aer_bench.pea_metric_evaluation import (  # noqa: E402
    read_json,
    sha256_path,
    write_json,
)

JOBS = ("P1", "P2", "P3", "P4")


def schema_for(schema: dict[str, Any], probe: str) -> dict[str, Any]:
    definition = schema["$defs"][probe]
    return {
        "$schema": schema["$schema"],
        "$id": f"https://aer-bench.local/schemas/matched-probe-{probe}.v0.2.json",
        **definition,
    }


def context_block(run_dir: Path, terminal: bool, *, compact_terminal: bool = False) -> str:
    parts = [(run_dir / "prompt.txt").read_text(encoding="utf-8")]
    if terminal:
        parts.append("\n\nSaved public episode record follows.\n")
        relatives = (
            ("codex/transcript.jsonl", "codex/final.json")
            if compact_terminal
            else (
                "public_environment_trajectory.jsonl",
                "codex/transcript.jsonl",
                "codex/final.json",
            )
        )
        for relative in relatives:
            path = run_dir / relative
            if path.is_file():
                parts.append(f"\n--- {relative} ---\n{path.read_text(encoding='utf-8')}\n")
    return "".join(parts)


def build_prompt(
    run_dir: Path,
    config: dict[str, Any],
    probe: str,
    *,
    compact_terminal: bool = False,
) -> str:
    terminal = probe in {"P3", "P4"}
    return (
        context_block(run_dir, terminal, compact_terminal=compact_terminal)
        + "\n\nThis is an independent read-only evaluation probe. It does not continue "
        "or alter the saved episode. Answer only from the context above and return the "
        "required JSON object.\n\n"
        + config["probe_prompts"][probe]
    )


def _run_probe(
    *,
    source_run: dict[str, Any],
    probe: str,
    config: dict[str, Any],
    schema: dict[str, Any],
    output_root: Path,
    repair: bool = False,
) -> dict[str, Any]:
    source_dir = Path(source_run["source_run_dir"])
    episode_key = source_run["episode_key"]
    probe_dir = output_root / "runs" / source_run["condition"] / episode_key / probe
    artifact_dir = probe_dir / "repair-input-too-large-01" if repair else probe_dir
    manifest_path = artifact_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        return {
            "episode_key": episode_key,
            "condition": source_run["condition"],
            "probe": probe,
            "status": manifest["status"],
            "preserved": True,
        }
    prompt = build_prompt(source_dir, config, probe, compact_terminal=repair)
    with tempfile.TemporaryDirectory(prefix="aer-matched-probe-v02-", dir="/tmp") as tmp:
        workspace = Path(tmp)
        schema_path = workspace / "output.schema.json"
        write_json(schema_path, schema_for(schema, probe))
        result = CodexRunner().run(
            CodexRunConfig(
                workspace=workspace,
                artifact_dir=artifact_dir,
                prompt=prompt,
                output_schema=schema_path,
                model=config["model"],
                reasoning_effort=config["reasoning_effort"],
                timeout_seconds=config["timeout_seconds"],
                sandbox="read-only",
                ephemeral=True,
                shell_tool_enabled=False,
            )
        )
    return {
        "episode_key": episode_key,
        "condition": source_run["condition"],
        "probe": probe,
        "status": result.status,
        "errors": result.errors,
        "preserved": False,
        "repair": repair,
    }


def _effective_probe_dir(
    output_root: Path, source: dict[str, Any], probe: str
) -> tuple[Path, str] | None:
    primary = output_root / "runs" / source["condition"] / source["episode_key"] / probe
    repair = primary / "repair-input-too-large-01"
    for path, label in ((primary, "primary"), (repair, "repair-input-too-large-01")):
        manifest_path = path / "manifest.json"
        if manifest_path.is_file() and read_json(manifest_path).get("status") == "completed":
            return path, label
    return None


def repair_input_too_large(
    *, config_path: Path, schema_path: Path, source_root: Path, output_root: Path, workers: int
) -> dict[str, Any]:
    config = read_json(config_path)
    schema = read_json(schema_path)
    aggregate = read_json(source_root / "matched_development_aggregate.json")
    sources = {source["episode_key"]: source for source in aggregate["runs"]}
    initial_path = output_root / "probe_run_summary.json"
    initial = read_json(initial_path)
    initial_copy = output_root / "probe_run_summary.initial.json"
    if not initial_copy.is_file():
        write_json(initial_copy, initial)
    failed = [
        item
        for item in initial["outcomes"]
        if item["status"] != "completed" and item["probe"] in {"P3", "P4"}
    ]
    if not failed:
        raise ValueError("no failed terminal probes are eligible for input-size repair")
    for item in failed:
        primary = (
            output_root
            / "runs"
            / item["condition"]
            / item["episode_key"]
            / item["probe"]
        )
        stderr = (primary / "stderr.log").read_text(encoding="utf-8")
        if "input_too_large" not in stderr:
            raise ValueError(
                f"repair refuses non-size failure: {item['episode_key']} {item['probe']}"
            )

    repaired: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _run_probe,
                source_run=sources[item["episode_key"]],
                probe=item["probe"],
                config=config,
                schema=schema,
                output_root=output_root,
                repair=True,
            ): item
            for item in failed
        }
        for future in as_completed(futures):
            outcome = future.result()
            repaired.append(outcome)
            print(
                f"REPAIR {outcome['status']} {outcome['episode_key']} {outcome['probe']}",
                flush=True,
            )

    effective: list[dict[str, Any]] = []
    for source in aggregate["runs"]:
        for probe in JOBS:
            resolved = _effective_probe_dir(output_root, source, probe)
            effective.append(
                {
                    "episode_key": source["episode_key"],
                    "condition": source["condition"],
                    "probe": probe,
                    "status": "completed" if resolved else "failed",
                    "effective_artifact": resolved[1] if resolved else None,
                }
            )
    effective.sort(key=lambda item: (item["condition"], item["episode_key"], item["probe"]))
    summary = {
        "schema_version": "aer.pea.matched-probe-run-summary.v1",
        "study_version": config["study_version"],
        "status": (
            "completed"
            if all(item["status"] == "completed" for item in effective)
            else "incomplete"
        ),
        "registered_probe_count": 192,
        "completed_probe_count": sum(item["status"] == "completed" for item in effective),
        "repair_policy": {
            "reason": "initial terminal context exceeded the Codex 1048576-character input limit",
            "change": (
                "omit duplicated public_environment_trajectory.jsonl; retain prompt, "
                "Codex transcript, and final submission"
            ),
            "failed_primary_artifacts_preserved": True,
            "repair_attempt_count": len(repaired),
        },
        "outcomes": effective,
    }
    write_json(initial_path, summary)
    write_json(output_root / "probe_input_size_repair_summary.json", {"repairs": repaired})
    return summary


def run(
    *, config_path: Path, schema_path: Path, source_root: Path, output_root: Path, workers: int
) -> dict[str, Any]:
    config = read_json(config_path)
    schema = read_json(schema_path)
    aggregate_path = source_root / "matched_development_aggregate.json"
    aggregate = read_json(aggregate_path)
    source_runs = aggregate.get("runs", [])
    if aggregate.get("status") != "all_registered_episodes_finalized" or len(source_runs) != 48:
        raise ValueError("source matched-development matrix must contain 48 finalized runs")
    conditions = set(config["conditions"])
    if {run["condition"] for run in source_runs} != conditions:
        raise ValueError("source conditions do not match the frozen v0.2 config")
    output_root.mkdir(parents=True, exist_ok=True)
    inventory = {
        "schema_version": "aer.pea.matched-probe-source-inventory.v1",
        "study_version": config["study_version"],
        "source_aggregate": {
            "path": str(aggregate_path.resolve()),
            "sha256": sha256_path(aggregate_path),
        },
        "config": {"path": str(config_path.resolve()), "sha256": sha256_path(config_path)},
        "schema": {"path": str(schema_path.resolve()), "sha256": sha256_path(schema_path)},
        "runs": [
            {
                "episode_key": item["episode_key"],
                "condition": item["condition"],
                "world": item["world"],
                "repetition": item["repetition"],
                "source_run_dir": item["source_run_dir"],
                "prompt_sha256": sha256_path(Path(item["source_run_dir"]) / "prompt.txt"),
                "transcript_sha256": item["transcript_sha256"],
            }
            for item in source_runs
        ],
    }
    inventory_path = output_root / "source_inventory.json"
    if inventory_path.is_file() and read_json(inventory_path) != inventory:
        raise RuntimeError("existing v0.2 source inventory differs from frozen inputs")
    write_json(inventory_path, inventory)

    outcomes: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _run_probe,
                source_run=source,
                probe=probe,
                config=config,
                schema=schema,
                output_root=output_root,
            ): (source["episode_key"], probe)
            for source in source_runs
            for probe in JOBS
        }
        for future in as_completed(futures):
            outcome = future.result()
            outcomes.append(outcome)
            print(
                f"PROBE {outcome['status']} {outcome['episode_key']} {outcome['probe']}",
                flush=True,
            )
    outcomes.sort(key=lambda item: (item["condition"], item["episode_key"], item["probe"]))
    summary = {
        "schema_version": "aer.pea.matched-probe-run-summary.v1",
        "study_version": config["study_version"],
        "status": (
            "completed"
            if all(item["status"] == "completed" for item in outcomes)
            else "incomplete"
        ),
        "registered_probe_count": 192,
        "completed_probe_count": sum(item["status"] == "completed" for item in outcomes),
        "outcomes": outcomes,
    }
    write_json(output_root / "probe_run_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-input-too-large", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    function = repair_input_too_large if args.repair_input_too_large else run
    summary = function(
        config_path=args.config.resolve(),
        schema_path=args.schema.resolve(),
        source_root=args.source.resolve(),
        output_root=args.output.resolve(),
        workers=args.workers,
    )
    print(json.dumps({key: summary[key] for key in ("status", "completed_probe_count")}, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
