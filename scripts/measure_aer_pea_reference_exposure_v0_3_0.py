#!/usr/bin/env python3
"""Run the frozen 0.3.0 held-out G0 and Exposure matrix."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from measure_aer_pea_reference_exposure import (
    AER_BENCH_ROOT,
    CASE_ROOT,
    SCIENCEWORLD_ROOT,
    TASK,
    WORLDS,
    _gold_actions,
    _load_evidence_module,
    _sha256,
)
from scienceworld import ScienceWorldEnv

SCRIPT_PATH = Path(__file__).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--held-out-freeze-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    freeze = json.loads(args.held_out_freeze_manifest.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen_for_held_out":
        parser.error("held-out freeze manifest is not frozen_for_held_out")
    if freeze.get("acceptance_spec_version") != "0.3.0":
        parser.error("held-out freeze manifest is not for acceptance spec 0.3.0")
    if freeze["acceptance_matrix"]["sha256"] != _sha256(args.matrix):
        parser.error("acceptance matrix does not match the frozen SHA-256")
    expected_runner = freeze["implementation_provenance"]["reference_runner_sha256"]
    if expected_runner != _sha256(SCRIPT_PATH):
        parser.error("reference runner does not match the frozen SHA-256")

    registered = matrix["held_out"]
    variations = registered["g0_variations"]
    roots = [registered["g0_root"]]
    evidence = _load_evidence_module()
    env = ScienceWorldEnv("", serverPath=None, envStepLimit=10_000)
    runs = []
    try:
        for variation in variations:
            actions = _gold_actions(env, variation)
            for case_root in roots:
                for world in WORLDS:
                    env.configure_aer_pea_case(world, case_root)
                    env.load(TASK, variation, "easy", generateGoldPath=False)
                    final = None
                    observations = []
                    for action in actions:
                        final = env.step(action)
                        observations.append(final[0])
                        if final[2]:
                            break
                    summary = env.get_aer_pea_case_summary()
                    public_text = "\n".join(observations)
                    public_fertility_opportunity = all(
                        marker in public_text
                        for marker in (
                            "a white flower [flower ",
                            "a purple flower [flower ",
                            "was pollinated",
                            "formed a fruit",
                        )
                    )
                    exposure_sufficient = evidence._exposure_sufficient(world, summary)
                    if world == "fertility_difference":
                        exposure_sufficient = (
                            exposure_sufficient and public_fertility_opportunity
                        )
                    runs.append(
                        {
                            "variation": variation,
                            "case_root": case_root,
                            "world": world,
                            "g0_completed": bool(
                                final and final[2] and final[3]["score"] == 100
                            ),
                            "exposure_sufficient": exposure_sufficient,
                            "public_fertility_opportunity": (
                                public_fertility_opportunity
                                if world == "fertility_difference"
                                else None
                            ),
                            "summary": summary,
                        }
                    )
                    print(
                        variation,
                        case_root,
                        world,
                        runs[-1]["g0_completed"],
                        runs[-1]["exposure_sufficient"],
                        flush=True,
                    )
    finally:
        env.close()

    counts = Counter(run["world"] for run in runs)
    exposed = Counter(run["world"] for run in runs if run["exposure_sufficient"])
    payload = {
        "schema_version": "aer.pea.reference-exposure.v1",
        "result_class": "held-out acceptance",
        "acceptance_spec_version": "0.3.0",
        "split": "held_out",
        "policy": "scienceworld_generated_g0_gold_actions",
        "variations": variations,
        "case_roots": roots,
        "worlds": list(WORLDS),
        "all_g0_completed": all(run["g0_completed"] for run in runs),
        "exposure_rates": {
            world: exposed[world] / counts[world] for world in sorted(counts)
        },
        "provenance": {
            "scienceworld_jar_sha256": _sha256(
                SCIENCEWORLD_ROOT / "scienceworld/scienceworld.jar"
            ),
            "script_sha256": _sha256(SCRIPT_PATH),
            "matrix_sha256": _sha256(args.matrix),
            "case_root": str(CASE_ROOT),
            "aer_bench_root": str(AER_BENCH_ROOT),
        },
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if payload["all_g0_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
