#!/usr/bin/env python3
"""Summarize frozen AER pea calibration cells with deterministic stage statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
SCIENCEWORLD_ROOT = SCRIPT_PATH.parents[1]
AER_BENCH_ROOT = SCIENCEWORLD_ROOT.parents[1]
CASE_ROOT = AER_BENCH_ROOT / "cases/science/mendelian_genetics_known_plant_aer"
MATRIX_PATH = CASE_ROOT / "construction/calibration_matrix.v0.2.0.json"
EVIDENCE_PATH = CASE_ROOT / "hidden/evidence.py"
GRADER_PATH = CASE_ROOT / "hidden/grader.py"
JAR_PATH = SCIENCEWORLD_ROOT / "scienceworld/scienceworld.jar"
Z_95 = 1.959963984540054


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wilson_interval(successes: int, trials: int) -> dict[str, float | int | None]:
    if trials == 0:
        return {"successes": successes, "trials": trials, "rate": None, "low": None, "high": None}
    rate = successes / trials
    denominator = 1 + Z_95**2 / trials
    center = (rate + Z_95**2 / (2 * trials)) / denominator
    radius = (
        Z_95
        * math.sqrt(rate * (1 - rate) / trials + Z_95**2 / (4 * trials**2))
        / denominator
    )
    return {
        "successes": successes,
        "trials": trials,
        "rate": rate,
        "low": max(0.0, center - radius),
        "high": min(1.0, center + radius),
    }


def _metric(
    runs: list[dict[str, Any]],
    outcome: Callable[[dict[str, Any]], bool],
    eligible: Callable[[dict[str, Any]], bool] = lambda _: True,
) -> dict[str, float | int | None]:
    eligible_runs = [run for run in runs if eligible(run)]
    return _wilson_interval(sum(outcome(run) for run in eligible_runs), len(eligible_runs))


def _read_run(metadata_path: Path) -> dict[str, Any]:
    run_dir = metadata_path.parent
    metadata = _read_json(metadata_path)
    grade_path = run_dir / "grade.json"
    if not grade_path.is_file():
        raise ValueError(f"missing deterministic grade: {grade_path}")
    grade = _read_json(grade_path)
    condition = metadata.get("calibration_condition")
    if not isinstance(condition, str):
        raise ValueError(f"run is not bound to a calibration condition: {metadata_path}")
    return {
        "run_id": metadata["run_id"],
        "condition": condition,
        "world": metadata["world"],
        "repetition": metadata["repetition"],
        "case_root": metadata["case_root"],
        "variation": metadata["variation"],
        "scienceworld_jar_sha256": metadata["scienceworld_jar_sha256"],
        "calibration_matrix_sha256": metadata["calibration_matrix_sha256"],
        "prompt_sha256": metadata["prompt_sha256"],
        "grade": grade,
    }


def _summarize_group(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "runs": len(runs),
        "Exposure": _metric(runs, lambda run: run["grade"]["Exposure"] is True),
        "Prioritize_given_Exposure": _metric(
            runs,
            lambda run: run["grade"]["Prioritize"] is True,
            lambda run: run["grade"]["Exposure"] is True,
        ),
        "DiscriminativeProbe_given_Investigation": _metric(
            runs,
            lambda run: run["grade"]["DiscriminativeProbe"] is True,
            lambda run: run["grade"]["Investigation"] is True,
        ),
        "strict_case_success": _metric(
            runs, lambda run: run["grade"]["strict_case_success"] is True
        ),
        "FalsePositive": _metric(
            runs, lambda run: run["grade"]["FalsePositive"] is True
        ),
    }


def build_summary(root: Path) -> dict[str, Any]:
    matrix = _read_json(MATRIX_PATH)
    runs = [_read_run(path) for path in sorted(root.glob("*/*/run_metadata.json"))]
    if not runs:
        raise ValueError(f"no condition runs found under {root}")

    conditions = list(matrix["conditions"])
    worlds = matrix["worlds"]
    repetitions = matrix["repetitions_per_condition_world_cell"]
    counts = Counter((run["condition"], run["world"]) for run in runs)
    expected_counts = {
        (condition, world): repetitions
        for condition in conditions
        for world in worlds
    }
    unknown_cells = sorted(set(counts) - set(expected_counts))
    cell_counts = [
        {
            "condition": condition,
            "world": world,
            "expected": repetitions,
            "observed": counts.get((condition, world), 0),
        }
        for condition in conditions
        for world in worlds
    ]
    replication_cells = {
        cell["repetition"]: (cell["case_root"], cell["variation"])
        for cell in matrix["replication_cells"]
    }
    expected_replications = {
        (condition, world, repetition, root, variation)
        for condition in conditions
        for world in worlds
        for repetition, (root, variation) in replication_cells.items()
    }
    observed_replications = Counter(
        (
            run["condition"],
            run["world"],
            run["repetition"],
            run["case_root"],
            run["variation"],
        )
        for run in runs
    )
    unknown_replications = sorted(set(observed_replications) - expected_replications)
    missing_replications = sorted(expected_replications - set(observed_replications))
    duplicate_replications = sorted(
        key for key, count in observed_replications.items() if count != 1
    )

    grouped: dict[str, dict[str, Any]] = {}
    pooled: dict[str, dict[str, Any]] = {}
    for condition in conditions:
        condition_runs = [run for run in runs if run["condition"] == condition]
        pooled[condition] = _summarize_group(condition_runs)
        grouped[condition] = {
            world: _summarize_group(
                [run for run in condition_runs if run["world"] == world]
            )
            for world in worlds
        }

    improvements = []
    for condition, definition in matrix["conditions"].items():
        comparison = definition.get("comparison_condition")
        minimum = definition.get("minimum_absolute_improvement")
        metric = definition["target_metric"]
        if comparison is None:
            continue
        current_rate = pooled[condition][metric]["rate"]
        comparison_rate = pooled[comparison][metric]["rate"]
        difference = (
            current_rate - comparison_rate
            if current_rate is not None and comparison_rate is not None
            else None
        )
        improvements.append(
            {
                "condition": condition,
                "comparison_condition": comparison,
                "metric": metric,
                "minimum_absolute_improvement": minimum,
                "absolute_point_estimate_difference": difference,
                "passes": difference is not None and difference >= minimum,
            }
        )

    persistent_worlds = set(worlds) - {"clean", "transient_null"}
    reactive_runs = [
        run
        for run in runs
        if run["condition"] == "reactive_no_active_exploration"
        and run["world"] in persistent_worlds
    ]
    reactive = _metric(
        reactive_runs, lambda run: run["grade"]["strict_discovery"] is True
    )
    control_runs = [run for run in runs if run["world"] in {"clean", "transient_null"}]
    control_false_positive = _metric(
        control_runs, lambda run: run["grade"]["FalsePositive"] is True
    )
    control_by_world = {
        world: _metric(
            [run for run in control_runs if run["world"] == world],
            lambda run: run["grade"]["FalsePositive"] is True,
        )
        for world in ("clean", "transient_null")
    }
    checks = matrix["acceptance_checks"]
    targeted_rates = [
        value
        for improvement in improvements
        for value in (
            pooled[improvement["condition"]][improvement["metric"]]["rate"],
            pooled[improvement["comparison_condition"]][improvement["metric"]]["rate"],
        )
        if value is not None
    ]
    gates = {
        "matrix_complete": not unknown_cells
        and all(cell["observed"] == cell["expected"] for cell in cell_counts),
        "targeted_improvements_pass": bool(improvements)
        and all(improvement["passes"] for improvement in improvements),
        "reactive_rate_pass": reactive["rate"] is not None
        and reactive["rate"] <= checks["reactive_max_strict_discovery_rate"],
        "negative_control_overall_pass": control_false_positive["rate"] is not None
        and control_false_positive["rate"]
        <= checks["negative_control_max_overall_false_positive_rate"],
        "negative_control_per_world_pass": all(
            result["rate"] is not None
            and result["rate"]
            <= checks["negative_control_max_per_world_false_positive_rate"]
            for result in control_by_world.values()
        ),
        "targeted_conditions_non_degenerate": bool(targeted_rates)
        and not all(rate == 0.0 for rate in targeted_rates)
        and not all(rate == 1.0 for rate in targeted_rates),
    }
    gates["matrix_complete"] = bool(
        gates["matrix_complete"]
        and not unknown_replications
        and not missing_replications
        and not duplicate_replications
    )
    return {
        "schema_version": "aer.pea.stage-calibration-summary.v1",
        "result_class": "dev calibration",
        "matrix_complete": gates["matrix_complete"],
        "all_gates_pass": all(gates.values()),
        "cell_counts": cell_counts,
        "unknown_cells": [list(cell) for cell in unknown_cells],
        "unknown_replications": [list(cell) for cell in unknown_replications],
        "missing_replications": [list(cell) for cell in missing_replications],
        "duplicate_replications": [list(cell) for cell in duplicate_replications],
        "pooled": pooled,
        "by_world": grouped,
        "targeted_improvements": improvements,
        "reactive_strict_discovery": reactive,
        "negative_control_false_positive": {
            "overall": control_false_positive,
            "by_world": control_by_world,
        },
        "gates": gates,
        "provenance": {
            "calibration_matrix_sha256": _sha256(MATRIX_PATH),
            "evidence_sha256": _sha256(EVIDENCE_PATH),
            "grader_sha256": _sha256(GRADER_PATH),
            "scienceworld_jar_sha256": _sha256(JAR_PATH),
            "summarizer_sha256": _sha256(SCRIPT_PATH),
            "run_matrix_hashes": sorted(
                {run["calibration_matrix_sha256"] for run in runs}
            ),
            "run_jar_hashes": sorted({run["scienceworld_jar_sha256"] for run in runs}),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = build_summary(args.root.resolve())
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "by_world"}, indent=2))
    return 0 if payload["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
