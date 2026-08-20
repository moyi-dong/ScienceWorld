#!/usr/bin/env python3
"""Measure case-provided exposure under the deterministic ScienceWorld G0 gold policy."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any

from run_aer_pea_calibration import _validate_split
from scienceworld import ScienceWorldEnv

TASK = "mendelian-genetics-known-plant-aer"
WORLDS = (
    "white_preference",
    "position_attraction",
    "plant_attractiveness",
    "fertility_difference",
    "transient_null",
    "clean",
)
SCRIPT_PATH = Path(__file__).resolve()
SCIENCEWORLD_ROOT = SCRIPT_PATH.parents[1]
AER_BENCH_ROOT = SCIENCEWORLD_ROOT.parents[1]
CASE_ROOT = AER_BENCH_ROOT / "cases/science/mendelian_genetics_known_plant_aer"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_evidence_module() -> Any:
    path = CASE_ROOT / "hidden/evidence.py"
    spec = importlib.util.spec_from_file_location("aer_pea_evidence", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gold_actions(env: ScienceWorldEnv, variation: int) -> list[str]:
    env.configure_aer_pea_case("clean", 0)
    env.load(TASK, variation, "easy", generateGoldPath=True)
    return list(env.server.getGoldActionSequence())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variation", action="append", type=int)
    parser.add_argument("--case-root", action="append", type=int)
    parser.add_argument("--world", action="append", choices=WORLDS)
    parser.add_argument(
        "--split", choices=("development", "held_out"), default="development"
    )
    parser.add_argument("--held-out-freeze-manifest", type=Path)
    args = parser.parse_args()

    split_path = CASE_ROOT / "construction/split.v0.1.0.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))[args.split]
    variations = args.variation or split["variations"]
    roots = args.case_root or split["roots"]
    worlds = args.world or WORLDS
    if not set(variations).issubset(split["variations"]):
        parser.error(f"reference exposure may use only registered {args.split} variations")
    if not set(roots).issubset(split["roots"]):
        parser.error(f"reference exposure may use only registered {args.split} roots")
    for variation in variations:
        for case_root in roots:
            try:
                _validate_split(
                    args.split,
                    variation,
                    case_root,
                    args.held_out_freeze_manifest,
                )
            except ValueError as error:
                parser.error(str(error))

    evidence = _load_evidence_module()
    env = ScienceWorldEnv("", serverPath=None, envStepLimit=10_000)
    runs = []
    try:
        for variation in variations:
            actions = _gold_actions(env, variation)
            for case_root in roots:
                for world in worlds:
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
                            "g0_completed": bool(final and final[2] and final[3]["score"] == 100),
                            "exposure_sufficient": exposure_sufficient,
                            "public_fertility_opportunity": public_fertility_opportunity
                            if world == "fertility_difference"
                            else None,
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
        "result_class": (
            "held-out acceptance" if args.split == "held_out" else "dev calibration"
        ),
        "split": args.split,
        "policy": "scienceworld_generated_g0_gold_actions",
        "variations": variations,
        "case_roots": roots,
        "worlds": list(worlds),
        "all_g0_completed": all(run["g0_completed"] for run in runs),
        "exposure_rates": {
            world: exposed[world] / counts[world] for world in sorted(counts)
        },
        "provenance": {
            "scienceworld_jar_sha256": _sha256(
                SCIENCEWORLD_ROOT / "scienceworld/scienceworld.jar"
            ),
            "script_sha256": _sha256(SCRIPT_PATH),
            "split_sha256": _sha256(split_path),
        },
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "runs"}, indent=2))
    return 0 if payload["all_g0_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
