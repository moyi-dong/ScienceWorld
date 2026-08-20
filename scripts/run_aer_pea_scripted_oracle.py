#!/usr/bin/env python3
"""Run the world-blind scripted oracle through the same public request surface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
SCIENCEWORLD_ROOT = SCRIPT_PATH.parents[1]
AER_BENCH_ROOT = SCIENCEWORLD_ROOT.parents[1]
CASE_ROOT = AER_BENCH_ROOT / "cases/science/mendelian_genetics_known_plant_aer"
sys.path.insert(0, str(SCIENCEWORLD_ROOT))

from scripts.run_aer_pea_calibration import (  # noqa: E402
    SUPPORTED_WORLDS,
    EpisodeService,
    _load_case_module,
    _safe_write_json,
    _sha256,
    _validate_split,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def run_one(
    output_root: Path,
    world: str,
    variation: int,
    case_root: int,
    *,
    result_class: str = "dev calibration",
) -> dict[str, Any]:
    run_id = f"{world}-variation-{variation:02d}-root-{case_root:04d}"
    run_dir = output_root / run_id
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True)
    trajectory_path = run_dir / "public_environment_trajectory.jsonl"
    operator_path = run_dir / "operator_action_windows.jsonl"
    service = EpisodeService(world, variation, case_root, trajectory_path, operator_path, 1_000)
    oracle_module = _load_case_module("aer_pea_oracle", CASE_ROOT / "hidden/oracle.py")
    evidence_module = _load_case_module("aer_pea_evidence", CASE_ROOT / "hidden/evidence.py")
    grader_module = _load_case_module("aer_pea_grader", CASE_ROOT / "hidden/grader.py")
    try:
        submission = oracle_module.AdaptiveOracle(service.handle).run()
        hidden_summary = service.env.get_aer_pea_case_summary()
        hidden_events = service.env.get_aer_pea_case_events()
        hidden_reproduction = service.env.get_aer_pea_case_reproduction_events()
    finally:
        service.close()

    _safe_write_json(run_dir / "submission.json", submission)
    _safe_write_json(run_dir / "hidden_summary.json", hidden_summary)
    _safe_write_json(run_dir / "hidden_events.json", hidden_events)
    _safe_write_json(run_dir / "hidden_reproduction_events.json", hidden_reproduction)
    grading_events = evidence_module.build_events(
        _read_jsonl(trajectory_path), _read_jsonl(operator_path), world=world
    )
    grading_path = run_dir / "grading_events.jsonl"
    grading_path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in grading_events),
        encoding="utf-8",
    )
    grade = grader_module.grade(
        submission,
        grading_events,
        expected_world=world,
        expected_height_trait="dominant" if variation < 15 else "recessive",
    )
    _safe_write_json(run_dir / "grade.json", grade)
    _safe_write_json(
        run_dir / "run_metadata.json",
        {
            "schema_version": "aer.pea.scripted-oracle-run.v1",
            "run_id": run_id,
            "result_class": result_class,
            "world": world,
            "variation": variation,
            "case_root": case_root,
            "policy": "world_blind_adaptive_oracle",
            "scienceworld_jar_sha256": _sha256(
                SCIENCEWORLD_ROOT / "scienceworld/scienceworld.jar"
            ),
            "oracle_sha256": _sha256(CASE_ROOT / "hidden/oracle.py"),
            "evidence_sha256": _sha256(CASE_ROOT / "hidden/evidence.py"),
            "grader_sha256": _sha256(CASE_ROOT / "hidden/grader.py"),
            "strict_case_success": grade["strict_case_success"],
        },
    )
    return {"run_id": run_id, "grade": grade, "submission": submission}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world", action="append", choices=SUPPORTED_WORLDS)
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--case-root", type=int, default=101)
    args = parser.parse_args()
    try:
        _validate_split("development", args.variation, args.case_root, None)
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for world in args.world or SUPPORTED_WORLDS:
        print(f"START {world}", flush=True)
        result = run_one(args.output, world, args.variation, args.case_root)
        results.append(result)
        print(
            f"DONE {world} strict_case_success={result['grade']['strict_case_success']}",
            flush=True,
        )
    _safe_write_json(args.output / "summary.json", results)
    return 0 if all(result["grade"]["strict_case_success"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
