#!/usr/bin/env python3
"""Build and pilot the formulation-aligned 20-Task pea development matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

import run_aer_pea_calibration as frozen
from jsonschema import Draft202012Validator
from scienceworld import ScienceWorldEnv

SCRIPT_PATH = Path(__file__).resolve()
SCIENCEWORLD_ROOT = SCRIPT_PATH.parents[1]
ROOT = SCRIPT_PATH.parents[3]
sys.path.insert(0, str(SCIENCEWORLD_ROOT))
sys.path.insert(0, str(ROOT / "src"))

from run_aer_pea_v1_pilot import V1EpisodeService  # noqa: E402

from aer_bench import trace as aer_trace  # noqa: E402
from aer_bench.codex_runner import CodexRunConfig, CodexRunner  # noqa: E402
from aer_bench.formulation_evaluation import (  # noqa: E402
    aggregate,
    score_task,
    validate_contracts,
)

CASE_ROOT = (
    ROOT
    / "cases/science/mendelian_genetics_known_plant_aer/revisions/v1_development"
)
CONSTRUCTION = CASE_ROOT / "construction"
PUBLIC = CASE_ROOT / "public"
FIXTURES = CASE_ROOT / "hidden/fixtures"
MATRIX_PATH = CONSTRUCTION / "hidden-configuration-matrix.v0.5-development.json"
TASKS_PATH = CONSTRUCTION / "formulation-task-matrix.v0.5-development.json"
EVALUATION_PATH = CONSTRUCTION / "evaluation-contract.v0.5-development.json"
DIFFERENTIAL_PATH = FIXTURES / "differential-test-manifest.v0.5-development.json"
ALIGNMENT_PATH = CONSTRUCTION / "formulation-alignment.v0.5-development.json"
PROBE_SCHEMA_PATH = PUBLIC / "evaluation-output.schema.v0.5-development.json"
P4_V051_SCHEMA_PATH = PUBLIC / "evaluation-output.schema.v0.5.1-development.json"
MECHANISM_GOLD_V051_PATH = CASE_ROOT / "hidden/mechanism-gold.v0.5.1-development.json"
PROTOCOL_CHANGE_V051_PATH = CONSTRUCTION / "protocol-change.evaluation-v0.5.1-development.json"
MAIN_SCHEMA_PATH = PUBLIC / "submission.schema.json"
CLIENT_PATH = PUBLIC / "lab.py"
FORMULATION_PATH = Path(
    "/Users/yrmac/Documents/Obsidian Vault/Research&Engineer/AER-Bench/Formulation.md"
)
MODEL = "gpt-5.6-sol"
REASONING_EFFORT = "high"
FAST_MODE = False
TASK_NAME = "mendelian-genetics-known-plant-aer"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_contracts() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    matrix = _read_json(MATRIX_PATH)
    tasks = _read_json(TASKS_PATH)
    evaluation = _read_json(EVALUATION_PATH)
    differential = _read_json(DIFFERENTIAL_PATH)
    configurations, task_index, tests = validate_contracts(matrix, tasks, differential)
    alignment = _read_json(ALIGNMENT_PATH)
    if not FORMULATION_PATH.is_file():
        raise FileNotFoundError(FORMULATION_PATH)
    if _sha256(FORMULATION_PATH) != alignment["source_document"]["sha256"]:
        raise ValueError("Formulation.md changed after the v0.5 development contract was written")
    if alignment["source_document"]["read_only"] is not True:
        raise ValueError("Formulation.md must remain read-only")
    if evaluation.get("scoring", {}).get("composite_score") is not None:
        raise ValueError("the formulation-aligned evaluation cannot define a composite score")
    return matrix, tasks, evaluation, differential, configurations, task_index, tests


class ConstructionService(V1EpisodeService):
    """Label top-level construction calls separately from Solver calls."""

    actor = "s0_construction"

    def _record_v1_response(
        self, request: dict[str, Any], response: dict[str, Any]
    ) -> dict[str, Any]:
        self._append({"source": self.actor, "request": request, "response": response})
        return response


def _new_service(
    configuration: dict[str, Any], directory: Path, *, actor: str = "s0_construction"
) -> ConstructionService:
    service = ConstructionService(
        configuration["world"],
        0,
        configuration["case_root"],
        directory / "public_environment_trajectory.jsonl",
        directory / "operator_action_windows.jsonl",
        2_000,
        configuration["noise_levels"],
    )
    service.actor = actor
    return service


def _request(
    service: ConstructionService,
    records: list[dict[str, Any]],
    command: str,
    spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"command": command}
    if spec is not None:
        payload["spec"] = spec
    response = service.handle(payload)
    if response.get("ok") is not True:
        raise RuntimeError(f"construction request failed: {payload}: {response}")
    records.append({"request": payload, "response": response})
    return response


def _cultivate_parents(
    service: ConstructionService, records: list[dict[str, Any]]
) -> tuple[str, str]:
    status = _request(service, records, "pots")
    seeds = [seed["seed_id"] for seed in status["seeds"] if seed["source"] == "initial seed stock"]
    if len(seeds) != 2:
        raise RuntimeError("the pea Case must start with exactly two public seed stocks")
    _request(
        service,
        records,
        "cultivate",
        {
            "assignments": [
                {"seed_id": seeds[0], "pot": "flower pot 1"},
                {"seed_id": seeds[1], "pot": "flower pot 2"},
            ],
            "target_stage": "flowering",
            "max_ticks": 100,
            "maintain_water": True,
        },
    )
    return "flower pot 1", "flower pot 2"


def _cross(
    service: ConstructionService,
    records: list[dict[str, Any]],
    crosses: list[dict[str, Any]],
) -> list[str]:
    result = _request(service, records, "controlled-cross", {"crosses": crosses})
    attempt_ids = result["attempt_ids"]
    _request(
        service,
        records,
        "wait-until",
        {
            "attempt_ids": attempt_ids,
            "condition": "resolved",
            "max_ticks": 30,
            "maintain_water": True,
        },
    )
    return attempt_ids


def _wait_for_flowers(
    service: ConstructionService,
    records: list[dict[str, Any]],
    pots: list[str],
) -> None:
    _request(
        service,
        records,
        "wait-until",
        {
            "targets": pots,
            "condition": "flowering",
            "scope": "all",
            "max_ticks": 20,
            "maintain_water": True,
        },
    )


def _pod_seeds(service: ConstructionService, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    status = _request(service, records, "pots")
    return [seed for seed in status["seeds"] if seed["pod_id"] is not None]


def _grow_seeds(
    service: ConstructionService,
    records: list[dict[str, Any]],
    seed_ids: list[str],
    pot_numbers: list[int],
) -> None:
    if len(seed_ids) != len(pot_numbers):
        raise ValueError("seed and pot counts differ")
    _request(
        service,
        records,
        "cultivate",
        {
            "assignments": [
                {"seed_id": seed_ids[index], "pot": f"flower pot {pot_numbers[index]}"}
                for index in range(len(seed_ids))
            ],
            "target_stage": "flowering",
            "max_ticks": 150,
            "maintain_water": True,
        },
    )


def _gold_actions(configuration: dict[str, Any]) -> list[str]:
    env = ScienceWorldEnv("", serverPath=None, envStepLimit=2_000)
    try:
        env.configure_aer_pea_case(
            configuration["world"],
            configuration["case_root"],
            noise_levels=configuration["noise_levels"],
        )
        env.load(TASK_NAME, 0, "easy", generateGoldPath=True)
        return list(env.server.getGoldActionSequence())
    finally:
        env.close()


def _build_gold_prefix(
    service: ConstructionService,
    records: list[dict[str, Any]],
    configuration: dict[str, Any],
    builder: dict[str, Any],
) -> None:
    actions = _gold_actions(configuration)
    boundary = builder["boundary_action_index"]
    if boundary >= len(actions):
        raise ValueError("gold prefix boundary exceeds the action sequence")
    for action in actions[: boundary + 1]:
        response = service._step(action, source="s0_construction")
        records.append(
            {"request": {"command": "act", "action": action}, "response": response}
        )


def _build_fertility(
    service: ConstructionService,
    records: list[dict[str, Any]],
    positive: bool,
) -> None:
    tall, short = _cultivate_parents(service, records)
    if not positive:
        _request(service, records, "pots")
        return
    if positive:
        reciprocal = [
            {"recipient_pot": tall, "pollen_pot": short, "emasculated": True, "bagged": True},
            {"recipient_pot": short, "pollen_pot": tall, "emasculated": True, "bagged": True},
        ]
        _request(service, records, "pots")
        _cross(service, records, reciprocal)
        _wait_for_flowers(service, records, [tall, short])
        _request(service, records, "pots")
        _cross(service, records, reciprocal)
        seeds = _pod_seeds(service, records)
        first_pod = min(seed["pod_id"] for seed in seeds)
        f1_ids = [seed["seed_id"] for seed in seeds if seed["pod_id"] == first_pod]
    _grow_seeds(service, records, f1_ids[:4], [3, 4, 5, 6])
    _cross(
        service,
        records,
        [
            {
                "recipient_pot": f"flower pot {pot}",
                "pollen_pot": f"flower pot {pot}",
                "emasculated": True,
                "bagged": True,
            }
            for pot in (3, 4, 5)
        ],
    )
    f1_set = set(f1_ids)
    f2_ids = [
        seed["seed_id"]
        for seed in _pod_seeds(service, records)
        if seed["seed_id"] not in f1_set and seed["pod_id"] is not None
    ][-12:]
    _grow_seeds(service, records, f2_ids, list(range(7, 19)))
    _request(service, records, "pots")


def _build_fruit_noise(
    service: ConstructionService,
    records: list[dict[str, Any]],
    positive: bool,
) -> None:
    tall, short = _cultivate_parents(service, records)
    if positive:
        _request(service, records, "pots")
        _cross(
            service,
            records,
            [
                {"recipient_pot": tall, "pollen_pot": short, "emasculated": True, "bagged": True},
                {"recipient_pot": short, "pollen_pot": tall, "emasculated": True, "bagged": True},
            ],
        )
        _request(service, records, "pots")


def _build_parentage(
    service: ConstructionService,
    records: list[dict[str, Any]],
    positive: bool,
) -> None:
    tall, short = _cultivate_parents(service, records)
    _request(service, records, "pots")
    _cross(
        service,
        records,
        [
            {
                "recipient_pot": short,
                "pollen_pot": tall,
                "emasculated": not positive,
                "bagged": not positive,
            }
        ],
    )
    seed_ids = [seed["seed_id"] for seed in _pod_seeds(service, records)]
    _grow_seeds(service, records, seed_ids[:4], [3, 4, 5, 6])
    _request(service, records, "pots")


def _build_soil(
    service: ConstructionService,
    records: list[dict[str, Any]],
    positive: bool,
) -> None:
    tall, short = _cultivate_parents(service, records)
    for index in range(3):
        _cross(
            service,
            records,
            [{"recipient_pot": short, "pollen_pot": tall, "emasculated": True, "bagged": True}],
        )
        if index < 2:
            _wait_for_flowers(service, records, [short])
    seed_ids = [seed["seed_id"] for seed in _pod_seeds(service, records)][:12]
    targets = [f"flower pot {pot}" for pot in range(3, 15)]
    _request(
        service,
        records,
        "operate",
        {
            "operation": "sow",
            "assignments": [
                {"seed_id": seed_ids[index], "pot": targets[index]}
                for index in range(len(seed_ids))
            ],
        },
    )
    _request(service, records, "operate", {"operation": "water", "targets": targets})
    if positive:
        pending = targets
        while pending:
            result = _request(
                service,
                records,
                "wait-until",
                {
                    "targets": pending,
                    "condition": "flowering",
                    "scope": "any",
                    "max_ticks": 100,
                    "maintain_water": True,
                },
            )
            ready = set(result["ready"])
            if not ready:
                raise RuntimeError("soil block made no progress")
            pending = [target for target in pending if target not in ready]
    _request(service, records, "pots")


def _build_recipe(
    service: ConstructionService,
    task: dict[str, Any],
    configuration: dict[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    builder = task["builder"]
    positive = task["surface"] == "surface_positive"
    kind = builder["kind"]
    if kind == "gold_visit_prefix":
        _build_gold_prefix(service, records, configuration, builder)
    elif kind == "fertility_multigeneration":
        _build_fertility(service, records, positive)
    elif kind == "fruit_set_noise":
        _build_fruit_noise(service, records, positive)
    elif kind == "parentage_contamination":
        _build_parentage(service, records, positive)
    elif kind == "soil_lot_block":
        _build_soil(service, records, positive)
    else:
        raise ValueError(f"unknown handoff builder: {kind}")
    return records


def _replay_recipe(
    service: ConstructionService, recipe: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    replayed: list[dict[str, Any]] = []
    for item in recipe:
        request = item["request"]
        if request["command"] == "act":
            response = service._step(request["action"], source="s0_construction")
        else:
            response = service.handle(request)
        if response.get("ok") is not True:
            raise RuntimeError(f"handoff replay failed: {request}: {response}")
        replayed.append({"request": request, "response": response})
    return replayed


def _flower_phenotypes(status: dict[str, Any]) -> list[tuple[str, str, str]]:
    values: list[tuple[str, str, str]] = []
    for pot in status["pots"]:
        for plant in pot["plants"]:
            for flower in plant.get("active_flowers", []):
                values.append((pot["name"], plant["height"], flower["perceived_color"]))
    return values


def _gate_evidence(
    task: dict[str, Any],
    configuration: dict[str, Any],
    records: list[dict[str, Any]],
    public_status: dict[str, Any],
    hidden_summary: dict[str, Any],
) -> dict[str, Any]:
    builder = task["builder"]
    gate = builder["gate"]
    kind = builder["kind"]
    passed = False
    evidence: dict[str, Any] = {}
    if kind == "gold_visit_prefix":
        total = hidden_summary["comparable_visits"]
        if configuration["world"] == "white_preference":
            target = hidden_summary["white_visits"]
        elif configuration["world"] == "position_attraction":
            target = hidden_summary["preferred_position_visits"]
        elif configuration["world"] == "plant_attractiveness":
            target = hidden_summary["preferred_plant_visits"]
        else:
            target = max(hidden_summary["white_visits"], hidden_summary["purple_visits"])
        fraction = target / total if total else 0.0
        evidence = {"comparable_visits": total, "target_count": target, "target_fraction": fraction}
        if "comparable_visit_count" in gate:
            passed = total == gate["comparable_visit_count"]
        elif task["surface"] == "surface_positive":
            passed = total >= gate["minimum_comparable_visits"] and fraction >= gate.get(
                "minimum_target_fraction", gate.get("minimum_color_fraction")
            )
        else:
            passed = total >= gate["minimum_comparable_visits"] and fraction <= gate.get(
                "maximum_target_fraction", gate.get("maximum_color_fraction")
            )
    elif kind == "fertility_multigeneration":
        white = hidden_summary["white_fruit_lags"]
        purple = hidden_summary["purple_fruit_lags"]
        phenotypes = _flower_phenotypes(public_status)
        by_height: dict[str, set[str]] = {}
        for _, height, color in phenotypes:
            by_height.setdefault(height, set()).add(color)
        same_height_colors = any(
            {"white", "purple"}.issubset(colors) for colors in by_height.values()
        )
        f2_count = sum(
            7 <= int(pot["name"].split()[-1]) <= 18 and pot["plant_count"] == 1
            for pot in public_status["pots"]
        )
        pair_count = min(len(white), len(purple))
        evidence = {
            "white_fruit_lags": white,
            "purple_fruit_lags": purple,
            "comparable_lag_pair_count": pair_count,
            "f2_plant_count": f2_count,
            "same_height_both_colors": same_height_colors,
        }
        if task["surface"] == "surface_positive":
            passed = (
                len(white) >= gate["minimum_white_lag_count"]
                and len(purple) >= gate["minimum_purple_lag_count"]
                and max(white) < min(purple)
                and f2_count >= gate["minimum_f2_plants"]
                and same_height_colors
            )
        else:
            passed = (
                pair_count <= gate["maximum_comparable_lag_pairs"]
                and f2_count == gate["f2_plant_count"]
            )
    elif kind == "fruit_set_noise":
        attempts = hidden_summary["cross_attempts"]
        aborted = sum(attempt["status"] == "aborted" for attempt in attempts)
        evidence = {"valid_attempt_count": len(attempts), "aborted_attempt_count": aborted}
        passed = (
            len(attempts) >= gate.get("minimum_valid_attempts", 0)
            and aborted >= gate.get("minimum_aborted_attempts", 0)
            if task["surface"] == "surface_positive"
            else len(attempts) == gate["cross_attempt_count"]
        )
    elif kind == "parentage_contamination":
        attempts = hidden_summary["cross_attempts"]
        contaminated = sum(attempt["contamination_occurred"] is True for attempt in attempts)
        offspring = [
            plant["height"]
            for pot in public_status["pots"]
            if pot["name"] in {f"flower pot {index}" for index in range(3, 7)}
            for plant in pot["plants"]
        ]
        mismatch = bool(offspring) and all(height == "short" for height in offspring)
        consistent = bool(offspring) and all(height == "tall" for height in offspring)
        evidence = {
            "contaminated_attempt_count": contaminated,
            "offspring_heights": offspring,
            "offspring_inconsistent_with_intended_cross": mismatch,
        }
        passed = (
            contaminated >= gate.get("minimum_contaminated_attempts", 0) and mismatch
            if task["surface"] == "surface_positive"
            else contaminated == gate["contaminated_attempt_count"] and consistent
        )
    elif kind == "soil_lot_block":
        planted = [
            plant
            for pot in public_status["pots"]
            if 3 <= int(pot["name"].split()[-1]) <= 14
            for plant in pot["plants"]
        ]
        lots = {plant["soil_lot_id"] for plant in planted}
        cumulative = 0
        completion_times: list[int] = []
        comparison_count = 0
        for record in records:
            if record["request"]["command"] != "wait-until":
                continue
            spec = record["request"].get("spec", {})
            if spec.get("condition") != "flowering" or spec.get("scope") != "any":
                continue
            cumulative += record["response"]["elapsed_ticks"]
            completion_times.extend([cumulative] * len(record["response"]["ready"]))
            comparison_count += 1
        tick_range = max(completion_times) - min(completion_times) if completion_times else 0
        evidence = {
            "sibling_count": len(planted),
            "visible_lots": sorted(lots),
            "completion_times": completion_times,
            "completion_tick_range": tick_range,
            "completed_growth_comparison_count": comparison_count,
        }
        passed = (
            len(planted) >= gate["minimum_sibling_count"]
            and len(lots) >= gate["minimum_visible_lot_count"]
            and tick_range >= gate["minimum_completion_tick_range"]
            if task["surface"] == "surface_positive"
            else len(planted) >= gate["minimum_sibling_count"]
            and comparison_count == gate["completed_growth_comparison_count"]
        )
    return {"passed": bool(passed), "evidence": evidence}


def _semantic_payload(
    service: ConstructionService, records: list[dict[str, Any]], task: dict[str, Any]
) -> dict[str, Any]:
    return {
        "task_id": task["id"],
        "task_description": service.env.taskdescription(),
        "initial_observation": service.initial["observation"],
        "prefix_records": records,
        "public_status": service.env.get_aer_pea_case_public_status(),
        "hidden_summary": service.env.get_aer_pea_case_summary(),
    }


def build_handoff(
    output_root: Path,
    task: dict[str, Any],
    configuration: dict[str, Any],
) -> dict[str, Any]:
    task_root = output_root / "handoffs" / task["id"]
    manifest_path = task_root / "handoff_manifest.json"
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        if manifest.get("status") != "verified_deterministic_candidate":
            raise RuntimeError(f"existing handoff is not verified: {manifest_path}")
        return manifest
    task_root.mkdir(parents=True, exist_ok=False)
    payloads: list[dict[str, Any]] = []
    recipe: list[dict[str, Any]] | None = None
    gate_result: dict[str, Any] | None = None
    for restore in (1, 2):
        restore_root = task_root / f"restore-{restore:02d}"
        restore_root.mkdir()
        service = _new_service(configuration, restore_root)
        try:
            if recipe is None:
                records = _build_recipe(service, task, configuration)
                recipe = [{"request": item["request"]} for item in records]
            else:
                records = _replay_recipe(service, recipe)
            payload = _semantic_payload(service, records, task)
            gate = _gate_evidence(
                task,
                configuration,
                records,
                payload["public_status"],
                payload["hidden_summary"],
            )
            if not gate["passed"]:
                raise RuntimeError(f"handoff gate failed for {task['id']}: {gate}")
            gate_result = gate
            _write_json(restore_root / "semantic_payload.json", payload)
            payloads.append(payload)
        finally:
            service.close()
    signatures = {_sha256_json(payload) for payload in payloads}
    if len(signatures) != 1:
        raise RuntimeError(f"handoff replay is nondeterministic: {task['id']}")
    public_packet = {
        "task_id": task["id"],
        "task_description": payloads[0]["task_description"],
        "initial_observation": payloads[0]["initial_observation"],
        "prefix_records": payloads[0]["prefix_records"],
        "public_status": payloads[0]["public_status"],
    }
    _write_json(task_root / "solver_visible_handoff.json", public_packet)
    _write_json(task_root / "replay_recipe.json", recipe)
    manifest = {
        "schema_version": "aer.pea.formulation-handoff.v0.5-development",
        "status": "verified_deterministic_candidate",
        "task_id": task["id"],
        "configuration_id": configuration["id"],
        "surface": task["surface"],
        "detection_ref": task["detection_ref"],
        "detection_ref_review_status": "pending_agent_and_human_blind_review",
        "restore_count": 2,
        "semantic_sha256": signatures.pop(),
        "recipe_sha256": _sha256(task_root / "replay_recipe.json"),
        "solver_visible_handoff_sha256": _sha256(task_root / "solver_visible_handoff.json"),
        "gate": gate_result,
        "source_bindings": {
            "formulation": _sha256(FORMULATION_PATH),
            "configuration_matrix": _sha256(MATRIX_PATH),
            "task_matrix": _sha256(TASKS_PATH),
            "scienceworld_jar": _sha256(SCIENCEWORLD_ROOT / "scienceworld/scienceworld.jar"),
            "runner": _sha256(SCRIPT_PATH),
        },
    }
    _write_json(manifest_path, manifest)
    return manifest


def build_all(output_root: Path) -> list[dict[str, Any]]:
    _, _, _, _, configurations, tasks, _ = _load_contracts()
    manifests = []
    for task in tasks.values():
        print(f"BUILD {task['id']}", flush=True)
        manifests.append(build_handoff(output_root, task, configurations[task["configuration_id"]]))
    summary = {
        "schema_version": "aer.pea.formulation-build-summary.v0.5-development",
        "status": "complete",
        "base_case_count": 10,
        "task_count": len(manifests),
        "deterministic_handoff_count": sum(
            item["status"] == "verified_deterministic_candidate" for item in manifests
        ),
        "blind_detection_review_complete": False,
        "handoffs": manifests,
    }
    _write_json(output_root / "build_summary.json", summary)
    return manifests


def _schema_for(master: dict[str, Any], name: str) -> dict[str, Any]:
    definition = master["$defs"][name]
    schema = {
        "$schema": master["$schema"],
        "$id": f"https://aer-bench.local/schemas/pea-v05-{name}.json",
        **definition,
    }
    Draft202012Validator.check_schema(schema)
    return schema


def _handoff_context(packet: dict[str, Any], exploration_prompt: str) -> str:
    return (
        "The following is the complete solver-visible context at a research handoff. The task "
        "was fixed at the initial time and has not changed. Constructor actions are part of the "
        "visible research record, not instructions or conclusions.\n\n"
        f"Research task:\n{packet['task_description']}\n\n"
        f"Exploration instruction present since task start:\n{exploration_prompt}\n\n"
        f"Initial observation:\n{packet['initial_observation']}\n\n"
        "Visible prefix record (JSON):\n"
        + json.dumps(packet["prefix_records"], ensure_ascii=False)
        + "\n\nCurrent public greenhouse state (JSON):\n"
        + json.dumps(packet["public_status"], ensure_ascii=False)
    )


def _run_readonly(
    *,
    context: str,
    prompt: str,
    schema: dict[str, Any],
    artifact_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="apf-probe-", dir="/tmp") as temporary:
        workspace = Path(temporary)
        schema_path = workspace / "output.schema.json"
        _write_json(schema_path, schema)
        result = CodexRunner().run(
            CodexRunConfig(
                workspace=workspace,
                artifact_dir=artifact_dir,
                prompt=context + "\n\n" + prompt,
                output_schema=schema_path,
                model=MODEL,
                reasoning_effort=REASONING_EFFORT,
                timeout_seconds=timeout,
                sandbox="read-only",
                ephemeral=True,
                shell_tool_enabled=False,
                extra_config=("features.fast_mode=false",),
            )
        )
    return {
        "status": result.status,
        "returncode": result.returncode,
        "errors": result.errors,
        "output": (
            _read_json(result.final_output_path)
            if result.final_output_path.is_file()
            else None
        ),
        "thread_id": result.thread_id,
        "usage": result.usage,
    }


def _main_prompt(context: str) -> str:
    return (
        context
        + "\n\nContinue the same task from this handoff. Use only the public lab.py client in "
        "the workspace. Run `python3 lab.py help` for exact commands. Do not inspect benchmark "
        "source, construction materials, graders, hidden tests, or oracle data. Return only the "
        "JSON required by submission.schema.json when finished."
    )


def _run_main(
    *,
    task_root: Path,
    configuration: dict[str, Any],
    recipe: list[dict[str, Any]],
    context: str,
    timeout: int,
) -> dict[str, Any]:
    run_root = task_root / "main"
    with tempfile.TemporaryDirectory(prefix="apf-main-", dir="/tmp") as temporary:
        workspace = Path(temporary)
        shutil.copy2(CLIENT_PATH, workspace / "lab.py")
        shutil.copy2(MAIN_SCHEMA_PATH, workspace / "submission.schema.json")
        service = _new_service(configuration, run_root, actor="s0_construction")
        try:
            _replay_recipe(service, recipe)
            service.actor = "solver"
            socket_path = workspace / "scienceworld.sock"
            server = frozen._UnixServer(str(socket_path), frozen._Handler)
            server.episode = service  # type: ignore[attr-defined]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                result = CodexRunner().run(
                    CodexRunConfig(
                        workspace=workspace,
                        artifact_dir=run_root / "codex",
                        prompt=_main_prompt(context),
                        output_schema=workspace / "submission.schema.json",
                        model=MODEL,
                        reasoning_effort=REASONING_EFFORT,
                        timeout_seconds=timeout,
                        sandbox="workspace-write",
                        ephemeral=True,
                        shell_tool_enabled=True,
                        extra_config=("features.fast_mode=false",),
                        unix_socket_allowlist=(socket_path,),
                        deny_read_paths=(ROOT, SCIENCEWORLD_ROOT, task_root.parent.parent),
                    )
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
            if result.events_path.is_file():
                aer_trace.write_normalized(
                    run_root / "codex/normalized_events.jsonl",
                    aer_trace.normalize_events(aer_trace.read_jsonl(result.events_path)),
                )
            hidden_summary = service.env.get_aer_pea_case_summary()
            _write_json(run_root / "hidden_summary.json", hidden_summary)
            return {
                "status": result.status,
                "returncode": result.returncode,
                "errors": result.errors,
                "output": (
                    _read_json(result.final_output_path)
                    if result.final_output_path.is_file()
                    else None
                ),
                "thread_id": result.thread_id,
                "usage": result.usage,
                "environment_completed": service.completed,
                "public_trajectory_path": str(service.trajectory_path),
            }
        finally:
            service.close()


def _terminal_context(handoff_context: str, task_root: Path, main: dict[str, Any]) -> str:
    parts = [handoff_context, "\n\nSolver continuation record:\n"]
    trajectory = Path(main["public_trajectory_path"])
    if trajectory.is_file():
        rows = [
            json.loads(line)
            for line in trajectory.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        visible = [row for row in rows if row.get("source") == "solver"]
        parts.append(json.dumps(visible, ensure_ascii=False))
    normalized = task_root / "main/codex/normalized_events.jsonl"
    if normalized.is_file():
        parts.append(
            "\n\nSolver-visible model event record:\n"
            + normalized.read_text(encoding="utf-8")
        )
    parts.append(
        "\n\nFinal structured submission:\n"
        + json.dumps(main.get("output"), ensure_ascii=False)
    )
    return "".join(parts)


def _trim_text(value: Any, limit: int = 2_000) -> Any:
    if not isinstance(value, str) or len(value) <= limit:
        return value
    half = limit // 2
    return value[:half] + "\n...[deterministically elided]...\n" + value[-half:]


def _compact_status(status: Any) -> dict[str, Any] | None:
    if not isinstance(status, dict):
        return None
    occupied = []
    for pot in status.get("pots", []):
        if not isinstance(pot, dict) or not pot.get("plant_count"):
            continue
        occupied.append(
            {
                "name": pot.get("name"),
                "plant_stages": pot.get("plant_stages"),
                "plants": pot.get("plants"),
                "flower_count": pot.get("flower_count"),
                "pending_fruit_count": pot.get("pending_fruit_count"),
                "formed_pod_count": pot.get("formed_pod_count"),
                "formed_seed_count": pot.get("formed_seed_count"),
            }
        )
    seeds = status.get("seeds", [])
    pod_counts: dict[str, int] = {}
    for seed in seeds if isinstance(seeds, list) else []:
        pod = str(seed.get("pod_id"))
        pod_counts[pod] = pod_counts.get(pod, 0) + 1
    return {
        "episode_tick": status.get("episode_tick"),
        "occupied_pots": occupied,
        "seed_count": len(seeds) if isinstance(seeds, list) else None,
        "seed_count_by_pod": pod_counts,
        "cross_attempts": status.get("cross_attempts"),
        "reproduction_history_tail": status.get("reproduction_history", [])[-40:],
    }


def _compact_response(response: Any) -> Any:
    if not isinstance(response, dict):
        return response
    keep = {
        key: response.get(key)
        for key in (
            "ok",
            "kind",
            "event_id",
            "action",
            "reward",
            "score",
            "completed",
            "condition",
            "scope",
            "targets",
            "attempt_ids",
            "ready",
            "pending",
            "elapsed_ticks",
            "timed_out",
            "requested_visit_count",
            "observed_visit_count",
            "out_of_scope_visit_count",
            "water_mutations",
            "visits_by_flower",
            "visits_by_perceived_color",
            "record",
            "experiment_id",
            "logical_actions",
            "primitive_mutations",
        )
        if key in response
    }
    if "observation" in response:
        keep["observation"] = _trim_text(response["observation"])
    status = response if response.get("kind") == "pots" else response.get("status")
    compact_status = _compact_status(status)
    if compact_status is not None:
        keep["compact_status"] = compact_status
    result = response.get("result")
    if isinstance(result, dict):
        keep["result"] = {
            key: result.get(key)
            for key in ("operation", "results", "ok", "error")
            if key in result
        }
    return keep


def _terminal_context_compact(
    handoff_context: str, task_root: Path, main: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    trajectory_path = Path(main["public_trajectory_path"])
    rows = [
        json.loads(line)
        for line in trajectory_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    public_events = [
        {
            "index": row.get("index"),
            "request": row.get("request"),
            "response": _compact_response(row.get("response")),
        }
        for row in rows
        if row.get("source") == "solver"
    ]
    normalized_path = task_root / "main/codex/normalized_events.jsonl"
    model_messages: list[dict[str, Any]] = []
    if normalized_path.is_file():
        for line in normalized_path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("kind") == "agent_message":
                model_messages.append(
                    {
                        "index": event.get("index"),
                        "text": event.get("payload", {}).get("text"),
                    }
                )
    packet = {
        "schema_version": "aer.pea.terminal-context-compaction.v0.5.1-development",
        "public_solver_events": public_events,
        "public_agent_messages": model_messages,
        "final_submission": main.get("output"),
        "source_bindings": {
            "public_trajectory_sha256": _sha256(trajectory_path),
            "normalized_model_events_sha256": (
                _sha256(normalized_path) if normalized_path.is_file() else None
            ),
        },
    }
    context = (
        handoff_context
        + "\n\nDeterministically compacted terminal research record. Repeated large state "
        "snapshots are represented by compact structured status; all Solver requests and all "
        "public Agent messages are retained.\n"
        + json.dumps(packet, ensure_ascii=False, separators=(",", ":"))
    )
    return context, packet


def run_pilot_task(output_root: Path, task_id: str, timeout: int) -> dict[str, Any]:
    _, _, evaluation, differential, configurations, tasks, tests = _load_contracts()
    task = tasks[task_id]
    configuration = configurations[task["configuration_id"]]
    build_handoff(output_root, task, configuration)
    handoff_root = output_root / "handoffs" / task_id
    packet = _read_json(handoff_root / "solver_visible_handoff.json")
    recipe = _read_json(handoff_root / "replay_recipe.json")
    context = _handoff_context(packet, evaluation["prompts"]["exploration_normal"])
    context_hash = hashlib.sha256(context.encode()).hexdigest()
    master_schema = _read_json(PROBE_SCHEMA_PATH)
    pilot_root = output_root / "pilot" / task_id
    result_path = pilot_root / "result.json"
    if result_path.is_file():
        preserved = _read_json(result_path)
        if preserved.get("handoff_context_sha256") != context_hash:
            raise RuntimeError(f"preserved pilot context changed for {task_id}")
        return preserved
    pilot_root.mkdir(parents=True, exist_ok=False)
    (pilot_root / "handoff_context.txt").write_text(context, encoding="utf-8")

    p1 = _run_readonly(
        context=context,
        prompt=evaluation["prompts"]["P1"],
        schema=_schema_for(master_schema, "P1"),
        artifact_dir=pilot_root / "P1",
        timeout=timeout,
    )
    p2 = _run_readonly(
        context=context,
        prompt=evaluation["prompts"]["P2"],
        schema=_schema_for(master_schema, "P2"),
        artifact_dir=pilot_root / "P2",
        timeout=timeout,
    )
    menu = "\n".join(
        f"- {item['id']}: {item['public_description']}" for item in differential["tests"]
    )
    differential_choice = _run_readonly(
        context=context,
        prompt=(
            "This is the separate differential Test Case for Experiment. Choose the one "
            "registered experiment that would best distinguish the live candidate explanations. "
            "The evaluator will execute its preregistered outcome comparison; this answer does "
            "not alter the main trajectory.\n\nAvailable experiments:\n"
            + menu
        ),
        schema=_schema_for(master_schema, "differential_test"),
        artifact_dir=pilot_root / "differential_test",
        timeout=timeout,
    )
    main = _run_main(
        task_root=pilot_root,
        configuration=configuration,
        recipe=recipe,
        context=context,
        timeout=timeout,
    )
    terminal_context = _terminal_context(context, pilot_root, main)
    p4 = _run_readonly(
        context=terminal_context,
        prompt=evaluation["prompts"]["P4"],
        schema=_schema_for(master_schema, "P4"),
        artifact_dir=pilot_root / "P4",
        timeout=timeout,
    )
    row = score_task(
        task_id=task_id,
        p1=p1["output"],
        p2=p2["output"],
        p4=p4["output"],
        differential_choice=differential_choice["output"],
        configurations=configurations,
        tasks=tasks,
        differential_tests=tests,
        candidate_worlds=differential["candidate_world_ids"],
    )
    result = {
        "schema_version": "aer.pea.formulation-pilot-task.v0.5-development",
        "task_id": task_id,
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "fast_mode": FAST_MODE,
        "handoff_context_sha256": context_hash,
        "P1_context_sha256": context_hash,
        "P2_context_sha256": context_hash,
        "P1": p1,
        "P2": p2,
        "differential_test": differential_choice,
        "main": main,
        "P4": p4,
        "score": row,
        "probe_isolation": {
            "P1_received_P2": False,
            "P2_received_P1": False,
            "probe_outputs_reentered_main": False,
            "main_trajectory_used_for_Experiment_score": False,
        },
    }
    _write_json(result_path, result)
    return result


def run_pilot(output_root: Path, task_ids: list[str], timeout: int) -> dict[str, Any]:
    rows = []
    results = []
    for task_id in task_ids:
        print(f"PILOT {task_id}", flush=True)
        result = run_pilot_task(output_root, task_id, timeout)
        results.append(result)
        rows.append(result["score"])
        _write_json(output_root / "pilot/pilot_partial.json", results)
    summary = aggregate(rows)
    summary["study"] = {
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "fast_mode": FAST_MODE,
        "task_ids": task_ids,
        "formal_20_task_estimate": False,
    }
    _write_json(output_root / "pilot/pilot_summary.json", summary)
    return summary


def recover_p4_v051(output_root: Path, task_ids: list[str], timeout: int) -> dict[str, Any]:
    _, _, evaluation, differential, configurations, tasks, tests = _load_contracts()
    protocol_change = _read_json(PROTOCOL_CHANGE_V051_PATH)
    if protocol_change.get("status") != "approved_for_development_recovery":
        raise ValueError("v0.5.1 P4 recovery is not approved")
    mechanism_gold = _read_json(MECHANISM_GOLD_V051_PATH)
    if mechanism_gold.get("status") != "frozen_for_development_recovery_before_v0.5.1_P4_calls":
        raise ValueError("v0.5.1 MechanismGold is not frozen")
    schema = _read_json(P4_V051_SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    rows = []
    recovery_results = []
    for task_id in task_ids:
        pilot_root = output_root / "pilot" / task_id
        prior = _read_json(pilot_root / "result.json")
        handoff_context = (pilot_root / "handoff_context.txt").read_text(encoding="utf-8")
        terminal_context, compact_packet = _terminal_context_compact(
            handoff_context, pilot_root, prior["main"]
        )
        compact_path = pilot_root / "terminal_context.v0.5.1.json"
        _write_json(compact_path, compact_packet)
        artifact_dir = pilot_root / "P4-v0.5.1"
        manifest_path = artifact_dir / "manifest.json"
        if manifest_path.is_file():
            p4 = {
                "status": _read_json(manifest_path)["status"],
                "output": (
                    _read_json(artifact_dir / "final.json")
                    if (artifact_dir / "final.json").is_file()
                    else None
                ),
                "errors": _read_json(manifest_path).get("errors", []),
            }
        else:
            p4 = _run_readonly(
                context=terminal_context,
                prompt=evaluation["prompts"]["P4"],
                schema=schema,
                artifact_dir=artifact_dir,
                timeout=timeout,
            )
        row = score_task(
            task_id=task_id,
            p1=prior["P1"]["output"],
            p2=prior["P2"]["output"],
            p4=p4["output"],
            differential_choice=prior["differential_test"]["output"],
            configurations=configurations,
            tasks=tasks,
            differential_tests=tests,
            candidate_worlds=differential["candidate_world_ids"],
            mechanism_gold_override=mechanism_gold["gold"],
        )
        recovery = {
            "schema_version": "aer.pea.formulation-pilot-task.v0.5.1-development",
            "task_id": task_id,
            "reused_v0_5": {
                "P1": True,
                "P2": True,
                "differential_test": True,
                "main": True,
            },
            "P4": p4,
            "terminal_context_compaction_sha256": _sha256(compact_path),
            "terminal_context_character_count": len(terminal_context),
            "score": row,
        }
        _write_json(pilot_root / "evaluation.v0.5.1.json", recovery)
        rows.append(row)
        recovery_results.append(recovery)
    summary = aggregate(rows)
    summary["schema_version"] = "aer.pea.formulation-evaluation-result.v0.5.1-development"
    summary["protocol_change"] = {
        "path": str(PROTOCOL_CHANGE_V051_PATH),
        "sha256": _sha256(PROTOCOL_CHANGE_V051_PATH),
    }
    summary["mechanism_gold"] = {
        "path": str(MECHANISM_GOLD_V051_PATH),
        "sha256": _sha256(MECHANISM_GOLD_V051_PATH),
    }
    summary["task_count"] = len(recovery_results)
    summary["study"] = {
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "fast_mode": FAST_MODE,
        "P4_calls_only": True,
        "task_ids": task_ids,
    }
    _write_json(output_root / "pilot/pilot_summary.v0.5.1-development.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "pilot", "recover-p4", "full"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", action="append")
    parser.add_argument("--timeout", type=int, default=1_800)
    args = parser.parse_args()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _, _, _, _, _, tasks, _ = _load_contracts()
    selected = args.task or ["M1-P", "M1-N", "M4-P", "N1-P", "N2-P", "C1-N"]
    unknown = sorted(set(selected) - set(tasks))
    if unknown:
        parser.error(f"unknown tasks: {', '.join(unknown)}")
    if args.command in {"build", "full"}:
        build_all(output_root)
    if args.command in {"pilot", "full"}:
        run_pilot(output_root, selected, args.timeout)
    if args.command == "recover-p4":
        recover_p4_v051(output_root, selected, args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
