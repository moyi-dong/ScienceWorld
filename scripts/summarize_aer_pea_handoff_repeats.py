#!/usr/bin/env python3
"""Validate and aggregate repeated pea handoff runs without model judging."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[3]
sys.path.insert(0, str(ROOT / "src"))

from aer_bench.pea_handoff_v06 import (  # noqa: E402
    aggregate_v063,
    read_json,
    sha256_path,
    write_json,
)

TASK_IDS = (
    "M1-P",
    "M1-N",
    "M2-P",
    "M2-N",
    "M3-P",
    "M3-N",
    "M4-P",
    "M4-N",
    "N1-P",
    "N1-N",
    "N2-P",
    "N2-N",
    "N3-P",
    "N3-N",
    "C1-P",
    "C1-N",
    "C2-P",
    "C2-N",
    "C3-P",
    "C3-N",
)
METRICS = ("Detection", "Triage", "Discovery-Existence", "Discovery-Exact")


def _required_artifacts(run_root: Path, task_id: str) -> dict[str, Path]:
    return {
        "run_manifest": run_root / f"runs/{task_id}/run_manifest.json",
        "handoff_context": run_root / f"runs/{task_id}/handoff_context.txt",
        "main_prompt": run_root / f"runs/{task_id}/main_prompt.txt",
        "replay_semantic_payload": run_root
        / f"runs/{task_id}/run_replay_semantic_payload.restricted.json",
        "hidden_summary": run_root / f"runs/{task_id}/hidden_summary.restricted.json",
        "public_environment_trajectory": run_root
        / f"runs/{task_id}/environment/public_environment_trajectory.jsonl",
        "operator_action_windows": run_root
        / f"runs/{task_id}/environment/operator_action_windows.jsonl",
        "run_json_rpc": run_root / f"runs/{task_id}/app-server-run/app_server_rpc.jsonl",
        "evaluation_result": run_root / f"evaluation/{task_id}/evaluation_result.json",
        "evaluation_json_rpc": run_root
        / f"evaluation/{task_id}/app-server-evaluation/app_server_rpc.jsonl",
    }


def _artifact_record(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required artifact is missing: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_path(path),
    }


def _score_value(summary: dict[str, Any], metric: str) -> float:
    if metric == "Discovery-Exact":
        return float(summary[metric]["mean"])
    return float(summary[metric]["f1"])


def load_run(run_root: Path, run_number: int) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_evaluation = read_json(run_root / "evaluation_summary.json")
    if raw_evaluation.get("task_count") != len(TASK_IDS):
        raise ValueError(f"{run_root.name}: expected 20 evaluated Tasks")
    raw_rows = raw_evaluation.get("tasks", [])
    if [row.get("task_id") for row in raw_rows] != list(TASK_IDS):
        raise ValueError(f"{run_root.name}: Task order or membership changed")

    run_summary = read_json(run_root / "run_summary.json")
    report = read_json(run_root / "study_report.json")
    model = run_summary.get("model")
    effort = run_summary.get("reasoning_effort")
    fast_mode = run_summary.get("fast_mode")
    if (model, effort, fast_mode) != ("gpt-5.6-sol", "high", False):
        raise ValueError(f"{run_root.name}: model profile changed")

    inventory_tasks = []
    completed_probe_count = 0
    for task_id in TASK_IDS:
        artifacts = {
            name: _artifact_record(path, run_root)
            for name, path in _required_artifacts(run_root, task_id).items()
        }
        evaluation_result = read_json(
            run_root / f"evaluation/{task_id}/evaluation_result.json"
        )
        probe_status = {
            probe: evaluation_result.get(probe, {}).get("status") for probe in ("P1", "P2", "P4")
        }
        completed_probe_count += sum(status == "completed" for status in probe_status.values())
        if any(evaluation_result.get(probe, {}).get("output") is None for probe in probe_status):
            raise ValueError(f"{run_root.name}/{task_id}: missing structured probe output")
        session_files = sorted((run_root / f"runs/{task_id}/codex-home/sessions").rglob("*.jsonl"))
        if not session_files:
            raise ValueError(f"{run_root.name}/{task_id}: native session JSONL is missing")
        native_sessions = [_artifact_record(path, run_root) for path in session_files]
        inventory_tasks.append(
            {
                "task_id": task_id,
                "probe_status": probe_status,
                "native_session_jsonl_count": len(session_files),
                "native_sessions": native_sessions,
                "artifacts": artifacts,
            }
        )

    scored = aggregate_v063(
        raw_rows,
        provenance={
            "run_number": run_number,
            "run_id": run_root.name,
            "source_evaluation": "evaluation_summary.json",
            "source_evaluation_sha256": sha256_path(run_root / "evaluation_summary.json"),
            "model": model,
            "reasoning_effort": effort,
            "fast_mode": fast_mode,
        },
    )
    if run_number > 1:
        write_json(run_root / "evaluation_summary.v0.6.3-development.json", scored)
    diagnostic_by_task = {item["task_id"]: item for item in report.get("diagnostics", [])}
    g0_completed = sum(bool(diagnostic_by_task[task_id]["G0_completed"]) for task_id in TASK_IDS)
    row = {
        "run_number": run_number,
        "run_id": run_root.name,
        "path": str(run_root),
        "model": model,
        "reasoning_effort": effort,
        "fast_mode": fast_mode,
        "G0_completed": g0_completed,
        "model_calls": report.get("call_accounting", {}).get(
            "run_and_evaluation_native_turn_calls"
        ),
        "metrics": {metric: _score_value(scored, metric) for metric in METRICS},
        "confusion_matrices": {
            metric: scored[metric]["confusion_counts"]
            for metric in METRICS
            if metric != "Discovery-Exact"
        },
    }
    inventory = {
        "run_number": run_number,
        "run_id": run_root.name,
        "task_count": len(inventory_tasks),
        "probe_count": len(TASK_IDS) * 3,
        "completed_probe_count": completed_probe_count,
        "run_evaluate_decoupled": report.get("run_evaluate_decoupling"),
        "tasks": inventory_tasks,
    }
    return row, inventory


def _distribution(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "population_sd": statistics.pstdev(values),
    }


def _write_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run_number",
                "run_id",
                "model",
                "reasoning_effort",
                "fast_mode",
                "Detection_F1",
                "Triage_F1",
                "Discovery_Existence_F1",
                "Discovery_Exact",
                "G0_completed",
            ]
        )
        for run in runs:
            writer.writerow(
                [
                    run["run_number"],
                    run["run_id"],
                    run["model"],
                    run["reasoning_effort"],
                    run["fast_mode"],
                    run["metrics"]["Detection"],
                    run["metrics"]["Triage"],
                    run["metrics"]["Discovery-Existence"],
                    run["metrics"]["Discovery-Exact"],
                    run["G0_completed"],
                ]
            )
    temporary.replace(path)


def summarize(run_roots: list[Path], output_root: Path) -> dict[str, Any]:
    if len(run_roots) != 5:
        raise ValueError("the registered repeat study requires exactly five groups")
    loaded = [load_run(path.resolve(), index) for index, path in enumerate(run_roots, 1)]
    runs = [item[0] for item in loaded]
    inventories = [item[1] for item in loaded]
    summary = {
        "schema_version": "aer.pea.five-run-summary.v0.6.3-development",
        "status": "complete_development_only",
        "official_leaderboard_result": False,
        "llm_judge_used_for_score": False,
        "model_profiles": sorted(
            {
                (run["model"], run["reasoning_effort"], run["fast_mode"])
                for run in runs
            }
        ),
        "run_count": len(runs),
        "task_count_per_run": len(TASK_IDS),
        "Experiment": "not_run",
        "composite_score": None,
        "runs": runs,
        "metric_distributions": {
            metric: _distribution([run["metrics"][metric] for run in runs])
            for metric in METRICS
        },
        "G0_completed_distribution": _distribution(
            [float(run["G0_completed"]) for run in runs]
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "five_run_summary.json", summary)
    write_json(
        output_root / "artifact_inventory.json",
        {
            "schema_version": "aer.pea.five-run-artifact-inventory.v0.6.3-development",
            "status": "complete",
            "runs": inventories,
        },
    )
    _write_csv(output_root / "five_run_table.csv", runs)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summarize(args.run, args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
