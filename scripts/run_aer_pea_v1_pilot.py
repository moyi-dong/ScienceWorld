#!/usr/bin/env python3
"""Run the development-only V1 pea batch/noise pilot with live Codex."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from scripts.run_aer_pea_calibration import (
    AER_BENCH_ROOT,
    AER_PEA_CASE_ROOT,
    SCIENCEWORLD_ROOT,
    CodexRunConfig,
    CodexRunner,
    EpisodeService,
    _Handler,
    _safe_write_json,
    _sha256,
    _UnixServer,
    normalize_events,
    read_jsonl,
    write_normalized,
)

MODEL = "gpt-5.6-sol"
REASONING_EFFORT = "high"
SUPPORTED_WORLDS = (
    "white_preference",
    "position_attraction",
    "plant_attractiveness",
    "fertility_difference",
    "cross_direction_delay",
    "transient_null",
    "clean",
)
NOISE_LEVELS = ("none", "weak", "medium", "strong")
V1_ROOT = AER_PEA_CASE_ROOT / "revisions" / "v1_development"
V1_CLIENT = V1_ROOT / "public" / "lab.py"
V1_SCHEMA = V1_ROOT / "public" / "submission.schema.json"
MATRIX_MANIFEST = V1_ROOT / "construction" / "hidden-configuration-matrix.v0.4.1-development.json"
MAX_BATCH_TARGETS = 20
NO_ADDED_NOISE = {
    "soil_nutrient_lot": "none",
    "fruit_set_success": "none",
    "cross_parentage_contamination": "none",
}
WORLD_CONCLUSIONS = {
    "white_preference": "perceived_flower_color",
    "position_attraction": "flower_pot_position",
    "plant_attractiveness": "plant_identity",
    "fertility_difference": "post_pollination_fruit_set_speed",
    "cross_direction_delay": "cross_direction_fruit_set_delay",
    "transient_null": "finite_sample_fluctuation",
    "clean": "uniform_no_anomaly",
}
PUBLIC_HELP = {
    "batch_target_rule": "Every changed pot must be named explicitly; 1-20 unique targets.",
    "post_task_exploration": (
        "Completing the commissioned score does not close v0.4 batch cultivation, controlled "
        "crosses, resolved waits, pots, or notebook records. Finish anomaly validation before "
        "the final submission."
    ),
    "commands": {
        "operate_water": {
            "operation": "water",
            "targets": ["flower pot 1", "flower pot 3"],
        },
        "operate_sow": {
            "operation": "sow",
            "assignments": [
                {"seed_id": "seed-0", "pot": "flower pot 1"},
                {"seed_id": "seed-1", "pot": "flower pot 3"},
            ],
        },
        "cultivate": {
            "assignments": [{"seed_id": "seed-0", "pot": "flower pot 1"}],
            "target_stage": "flowering",
            "max_ticks": 100,
            "maintain_water": True,
        },
        "controlled-cross": {
            "crosses": [
                {
                    "recipient_pot": "flower pot 1",
                    "pollen_pot": "flower pot 3",
                    "emasculated": True,
                    "bagged": True,
                }
            ]
        },
        "wait-until": {
            "attempt_ids": ["cross-0"],
            "condition": "resolved",
            "scope": "all",
            "max_ticks": 100,
            "maintain_water": True,
        },
        "observe-visits": {
            "targets": ["flower pot 1", "flower pot 3"],
            "min_visits": 12,
            "max_ticks": 100,
            "maintain_water": True,
        },
    },
    "wait_conditions": [
        "seedling", "adult", "reproducing", "flowering", "fruit", "resolved"
    ],
    "notebook": {
        "notice": {"kind": "notice", "surface": "undetermined"},
        "prioritize": {"kind": "prioritize", "surface": "undetermined"},
        "experiment_preregister": {
            "kind": "experiment_preregister",
            "experiment_id": "EXP-followup-1",
            "phase": "investigation",
            "probe_kind": "expanded_sampling",
            "hypothesis": "undetermined",
            "prediction": "state the result that would distinguish the live alternatives",
        },
        "experiment_end": {
            "kind": "experiment_end",
            "experiment_id": "EXP-followup-1",
        },
        "surfaces": ["visit_imbalance", "fruit_set_timing", "none", "undetermined"],
        "probe_kinds": [
            "expanded_sampling",
            "position_swap",
            "perceived_color_swap",
            "reciprocal_manual_pollination",
            "fresh_flowering_period",
        ],
        "hypotheses": [
            "perceived_flower_color",
            "flower_pot_position",
            "plant_identity",
            "post_pollination_fruit_set_speed",
            "cross_direction_fruit_set_delay",
            "fruit_set_stochastic_failure",
            "cross_parentage_contamination",
            "soil_nutrient_lot",
            "finite_sample_fluctuation",
            "uniform_no_anomaly",
            "undetermined",
        ],
    },
}


def load_hidden_configuration_matrix(path: Path = MATRIX_MANIFEST) -> dict[str, dict[str, Any]]:
    """Load and validate the development-only 4/3/3 configuration matrix."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    configurations = payload.get("configurations")
    if not isinstance(configurations, list):
        raise ValueError("hidden configuration matrix must contain a configurations list")
    by_id: dict[str, dict[str, Any]] = {}
    counts = {"mechanism": 0, "single_noise": 0, "clean_control": 0}
    noise_keys = set(NO_ADDED_NOISE)
    for item in configurations:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError("every hidden configuration must have a string id")
        config_id = item["id"]
        if config_id in by_id:
            raise ValueError(f"duplicate hidden configuration id: {config_id}")
        group = item.get("group")
        if group not in counts:
            raise ValueError(f"unknown hidden configuration group for {config_id}: {group}")
        world = item.get("world")
        if world not in SUPPORTED_WORLDS:
            raise ValueError(f"unsupported world for {config_id}: {world}")
        root = item.get("development_root")
        if isinstance(root, bool) or not isinstance(root, int) or root < 0:
            raise ValueError(f"invalid development root for {config_id}")
        noise = item.get("noise_levels")
        if not isinstance(noise, dict) or set(noise) != noise_keys:
            raise ValueError(f"{config_id} must configure exactly the three v0.4 noise axes")
        if any(level not in NOISE_LEVELS for level in noise.values()):
            raise ValueError(f"{config_id} contains an invalid noise level")
        active_noise = [key for key, level in noise.items() if level != "none"]
        if group == "mechanism":
            if active_noise:
                raise ValueError(f"mechanism configuration {config_id} must have no added noise")
            if world in {"clean", "transient_null"}:
                raise ValueError(f"mechanism configuration {config_id} uses a null world")
        elif group == "single_noise":
            if world != "clean" or len(active_noise) != 1:
                raise ValueError(
                    f"single-noise configuration {config_id} must use clean and one noise axis"
                )
        elif world != "clean" or active_noise:
            raise ValueError(f"clean control {config_id} must use clean with no added noise")
        h_star = item.get("h_star")
        if not isinstance(h_star, dict) or not isinstance(h_star.get("conclusion_id"), str):
            raise ValueError(f"{config_id} must define h_star.conclusion_id")
        counts[group] += 1
        by_id[config_id] = item
    if counts != {"mechanism": 4, "single_noise": 3, "clean_control": 3}:
        raise ValueError(f"hidden configuration matrix must be 4/3/3, got {counts}")
    return by_id


class V1EpisodeService(EpisodeService):
    """Development service with explicit-target mutations and completion-aware waits."""

    note_mechanisms = EpisodeService.note_mechanisms | {
        "cross_direction_fruit_set_delay",
        "fruit_set_stochastic_failure",
        "cross_parentage_contamination",
        "soil_nutrient_lot",
    }

    def __init__(
        self,
        world: str,
        variation: int,
        case_root: int,
        trajectory_path: Path,
        operator_window_path: Path,
        step_limit: int,
        noise_levels: dict[str, str],
    ) -> None:
        super().__init__(
            world,
            variation,
            case_root,
            trajectory_path,
            operator_window_path,
            step_limit,
            noise_levels=noise_levels,
        )

    def _status(self, *, compact: bool = False) -> dict[str, Any]:
        status = self.env.get_aer_pea_case_public_status()
        if compact:
            status.pop("reproduction_history", None)
        return status

    def _validate_targets(self, raw_targets: Any) -> list[str]:
        if not isinstance(raw_targets, list) or not raw_targets:
            raise ValueError("targets must be a non-empty list")
        if len(raw_targets) > MAX_BATCH_TARGETS:
            raise ValueError(f"targets may contain at most {MAX_BATCH_TARGETS} pots")
        if any(not isinstance(target, str) or not target.strip() for target in raw_targets):
            raise ValueError("every target must be a non-empty string")
        targets = [target.strip() for target in raw_targets]
        if len(targets) != len(set(targets)):
            raise ValueError("targets must be unique")
        known = {pot["name"] for pot in self._status()["pots"]}
        unknown = [target for target in targets if target not in known]
        if unknown:
            raise ValueError(f"unknown flower pots: {', '.join(unknown)}")
        return targets

    @staticmethod
    def _pot_map(status: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {pot["name"]: pot for pot in status["pots"]}

    @staticmethod
    def _ready(pot: dict[str, Any], condition: str) -> bool:
        if condition == "fruit":
            return pot["formed_seed_count"] > 0
        if condition == "flowering":
            return pot["flower_count"] > 0
        ranks = {"seed": 0, "seedling": 1, "adult": 2, "reproducing": 3}
        if condition not in ranks:
            raise ValueError("condition must be seedling, adult, reproducing, flowering, or fruit")
        stages = pot["plant_stages"]
        return bool(stages) and all(ranks.get(stage, -1) >= ranks[condition] for stage in stages)

    def _water(self, targets: list[str]) -> dict[str, Any]:
        return self.env.batch_water_aer_pea_case(targets)

    def _sow(self, assignments: Any) -> tuple[list[str], dict[str, Any]]:
        if not isinstance(assignments, list) or not assignments:
            raise ValueError("assignments must be a non-empty list")
        if any(
            not isinstance(item, dict) or set(item) != {"seed_id", "pot"} for item in assignments
        ):
            raise ValueError("each assignment must contain exactly seed_id and pot")
        seed_ids = [item["seed_id"] for item in assignments]
        if any(not isinstance(seed_id, str) or not seed_id for seed_id in seed_ids):
            raise ValueError("every seed_id must be a non-empty string")
        if len(seed_ids) != len(set(seed_ids)):
            raise ValueError("seed IDs must be unique")
        targets = self._validate_targets([item["pot"] for item in assignments])
        result = self.env.batch_sow_aer_pea_case(seed_ids, targets)
        return targets, result

    def _wait_until(
        self,
        targets: list[str],
        condition: str,
        scope: str,
        max_ticks: int,
        maintain_water: bool,
        attempt_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        if scope not in {"all", "any"}:
            raise ValueError("scope must be all or any")
        if (
            isinstance(max_ticks, bool)
            or not isinstance(max_ticks, int)
            or not 1 <= max_ticks <= 500
        ):
            raise ValueError("max_ticks must be an integer from 1 to 500")
        if not isinstance(maintain_water, bool):
            raise ValueError("maintain_water must be true or false")

        water_mutations = 0
        for elapsed in range(max_ticks + 1):
            status = self._status(compact=True)
            pots = self._pot_map(status)
            if condition == "resolved":
                attempts = {item["attempt_id"]: item for item in status["cross_attempts"]}
                selected = attempt_ids or []
                unknown = [attempt_id for attempt_id in selected if attempt_id not in attempts]
                if unknown:
                    raise ValueError(f"unknown cross attempts: {', '.join(unknown)}")
                ready = [
                    attempt_id
                    for attempt_id in selected
                    if attempts[attempt_id]["status"] in {"pod_set", "aborted", "rejected"}
                ]
                pending = [attempt_id for attempt_id in selected if attempt_id not in ready]
                expected_count = len(selected)
            else:
                ready = [target for target in targets if self._ready(pots[target], condition)]
                pending = [target for target in targets if target not in ready]
                expected_count = len(targets)
            complete = len(ready) == expected_count if scope == "all" else bool(ready)
            if complete or elapsed == max_ticks:
                return {
                    "ok": True,
                    "kind": "wait-until",
                    "condition": condition,
                    "scope": scope,
                    "targets": targets,
                    "attempt_ids": attempt_ids or [],
                    "ready": ready,
                    "pending": pending,
                    "elapsed_ticks": elapsed,
                    "timed_out": not complete,
                    "water_mutations": water_mutations,
                    "status": status,
                }
            if maintain_water:
                dry = [
                    target
                    for target in targets
                    if pots[target]["plant_count"] > 0 and not pots[target]["has_water"]
                ]
                if dry:
                    water_result = self._water(dry)
                    if not water_result.get("ok"):
                        raise RuntimeError(water_result.get("error", "batch watering failed"))
                    water_mutations += len(dry)
            self._step("wait1", source="v1_wait_until")
        raise AssertionError("unreachable")

    def _observe_visits(
        self,
        targets: list[str],
        min_visits: int,
        max_ticks: int,
        maintain_water: bool,
    ) -> dict[str, Any]:
        if (
            isinstance(min_visits, bool)
            or not isinstance(min_visits, int)
            or not 1 <= min_visits <= 200
        ):
            raise ValueError("min_visits must be an integer from 1 to 200")
        if (
            isinstance(max_ticks, bool)
            or not isinstance(max_ticks, int)
            or not 1 <= max_ticks <= 500
        ):
            raise ValueError("max_ticks must be an integer from 1 to 500")
        if not isinstance(maintain_water, bool):
            raise ValueError("maintain_water must be true or false")

        start_index = len(self.env.get_aer_pea_case_events())
        target_set = set(targets)
        water_mutations = 0
        selected: list[dict[str, Any]] = []
        fresh: list[dict[str, Any]] = []
        status = self._status(compact=True)
        for elapsed in range(max_ticks + 1):
            fresh = self.env.get_aer_pea_case_events()[start_index:]
            selected = [event for event in fresh if event["flower_pot"] in target_set]
            if len(selected) >= min_visits or elapsed == max_ticks:
                status = self._status(compact=True)
                break
            status = self._status(compact=True)
            pots = self._pot_map(status)
            if maintain_water:
                dry = [
                    target
                    for target in targets
                    if pots[target]["plant_count"] > 0 and not pots[target]["has_water"]
                ]
                if dry:
                    water_result = self._water(dry)
                    if not water_result.get("ok"):
                        raise RuntimeError(water_result.get("error", "batch watering failed"))
                    water_mutations += len(dry)
            self._step("wait1", source="v1_observe_visits")
        else:
            raise AssertionError("unreachable")

        by_flower: dict[int, dict[str, Any]] = {}
        by_color: dict[str, int] = {}
        for event in selected:
            flower_id = event["flower_id"]
            aggregate = by_flower.setdefault(
                flower_id,
                {
                    "flower_id": flower_id,
                    "plant_id": event["plant_id"],
                    "flower_pot": event["flower_pot"],
                    "perceived_color": event["perceived_color"],
                    "plant_height": event["plant_height"],
                    "visit_count": 0,
                    "first_tick": event["tick"],
                    "last_tick": event["tick"],
                },
            )
            aggregate["visit_count"] += 1
            aggregate["last_tick"] = event["tick"]
            color = event["perceived_color"]
            by_color[color] = by_color.get(color, 0) + 1

        return {
            "ok": True,
            "kind": "observe-visits",
            "targets": targets,
            "requested_visit_count": min_visits,
            "observed_visit_count": len(selected),
            "out_of_scope_visit_count": len(fresh) - len(selected),
            "elapsed_ticks": elapsed,
            "timed_out": len(selected) < min_visits,
            "water_mutations": water_mutations,
            "visits_by_flower": [by_flower[key] for key in sorted(by_flower)],
            "visits_by_perceived_color": dict(sorted(by_color.items())),
            "status": status,
        }

    def _controlled_cross(self, raw_crosses: Any) -> tuple[list[str], dict[str, Any]]:
        if not isinstance(raw_crosses, list) or not raw_crosses:
            raise ValueError("crosses must be a non-empty list")
        if len(raw_crosses) > MAX_BATCH_TARGETS:
            raise ValueError(f"crosses may contain at most {MAX_BATCH_TARGETS} items")
        required = {"recipient_pot", "pollen_pot", "emasculated", "bagged"}
        if any(not isinstance(item, dict) or set(item) != required for item in raw_crosses):
            raise ValueError(
                "each cross must contain exactly recipient_pot, pollen_pot, emasculated, bagged"
            )
        recipients = self._validate_targets([item["recipient_pot"] for item in raw_crosses])
        self._validate_targets(list(dict.fromkeys(item["pollen_pot"] for item in raw_crosses)))
        if any(
            not isinstance(item["emasculated"], bool) or not isinstance(item["bagged"], bool)
            for item in raw_crosses
        ):
            raise ValueError("emasculated and bagged must be booleans")
        result = self.env.controlled_cross_aer_pea_case(raw_crosses)
        return recipients, result

    def _record_v1_response(
        self, request: dict[str, Any], response: dict[str, Any]
    ) -> dict[str, Any]:
        self._append({"source": "solver", "request": request, "response": response})
        return response

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        command = request.get("command")
        if command == "help":
            response = {"ok": True, "kind": "help", **PUBLIC_HELP}
            return self._record_v1_response(request, response)
        if command not in {
            "pots",
            "operate",
            "cultivate",
            "controlled-cross",
            "wait-until",
            "observe-visits",
        }:
            return super().handle(request)

        with self._lock:
            try:
                if command == "pots":
                    response = {"ok": True, "kind": "pots", **self._status()}
                else:
                    spec = request.get("spec")
                    if not isinstance(spec, dict):
                        raise ValueError("spec must be an object")
                    if command == "controlled-cross":
                        if set(spec) != {"crosses"}:
                            raise ValueError("controlled-cross spec must contain exactly crosses")
                        targets, mutation = self._controlled_cross(spec["crosses"])
                        if not mutation.get("ok"):
                            raise ValueError(mutation.get("error", "controlled cross failed"))
                        response = {
                            "ok": True,
                            "kind": "controlled-cross",
                            "targets": targets,
                            "attempt_ids": [item["attempt_id"] for item in mutation["results"]],
                            "logical_actions": 1,
                            "primitive_mutations": len(targets),
                            "result": mutation,
                            "status": self._status(compact=True),
                        }
                    elif command == "operate":
                        operation = spec.get("operation")
                        if operation == "water":
                            if set(spec) != {"operation", "targets"}:
                                raise ValueError("water spec fields must be operation and targets")
                            targets = self._validate_targets(spec["targets"])
                            mutation = self._water(targets)
                        elif operation == "sow":
                            if set(spec) != {"operation", "assignments"}:
                                raise ValueError(
                                    "sow spec fields must be operation and assignments"
                                )
                            targets, mutation = self._sow(spec["assignments"])
                        else:
                            raise ValueError("operation must be water or sow")
                        if not mutation.get("ok"):
                            raise ValueError(mutation.get("error", "batch mutation failed"))
                        response = {
                            "ok": True,
                            "kind": "operate",
                            "operation": operation,
                            "targets": targets,
                            "logical_actions": 1,
                            "primitive_mutations": len(targets),
                            "result": mutation,
                            "status": self._status(compact=True),
                        }
                    elif command == "wait-until":
                        required = {"condition", "max_ticks"}
                        optional = {"targets", "attempt_ids", "scope", "maintain_water"}
                        if not required.issubset(spec) or not set(spec).issubset(
                            required | optional
                        ):
                            raise ValueError(
                                "wait-until requires condition and max_ticks plus targets or "
                                "attempt_ids; scope and maintain_water are optional"
                            )
                        condition = str(spec["condition"])
                        if condition == "resolved":
                            raw_attempt_ids = spec.get("attempt_ids")
                            if not isinstance(raw_attempt_ids, list) or not raw_attempt_ids:
                                raise ValueError(
                                    "resolved waits require a non-empty attempt_ids list"
                                )
                            if len(raw_attempt_ids) != len(set(raw_attempt_ids)) or any(
                                not isinstance(item, str) or not item for item in raw_attempt_ids
                            ):
                                raise ValueError("attempt_ids must be unique non-empty strings")
                            attempt_ids = raw_attempt_ids
                            attempts = {
                                item["attempt_id"]: item
                                for item in self._status(compact=True)["cross_attempts"]
                            }
                            targets = (
                                list(
                                    dict.fromkeys(
                                        attempts[item]["recipient_pot"]
                                        for item in attempt_ids
                                    )
                                )
                                if all(item in attempts for item in attempt_ids)
                                else []
                            )
                        else:
                            attempt_ids = None
                            targets = self._validate_targets(spec.get("targets"))
                        response = self._wait_until(
                            targets,
                            condition,
                            str(spec.get("scope", "all")),
                            spec["max_ticks"],
                            spec.get("maintain_water", True),
                            attempt_ids,
                        )
                    elif command == "observe-visits":
                        required = {"targets", "min_visits", "max_ticks"}
                        optional = {"maintain_water"}
                        if not required.issubset(spec) or not set(spec).issubset(
                            required | optional
                        ):
                            raise ValueError(
                                "observe-visits requires targets, min_visits, and max_ticks; "
                                "maintain_water is optional"
                            )
                        targets = self._validate_targets(spec["targets"])
                        response = self._observe_visits(
                            targets,
                            spec["min_visits"],
                            spec["max_ticks"],
                            spec.get("maintain_water", True),
                        )
                    else:
                        required = {"assignments", "target_stage", "max_ticks"}
                        optional = {"maintain_water"}
                        if not required.issubset(spec) or not set(spec).issubset(
                            required | optional
                        ):
                            raise ValueError(
                                "cultivate requires assignments, target_stage, and max_ticks"
                            )
                        targets, sow_result = self._sow(spec["assignments"])
                        if not sow_result.get("ok"):
                            raise ValueError(sow_result.get("error", "batch sowing failed"))
                        water_result = self._water(targets)
                        if not water_result.get("ok"):
                            raise ValueError(water_result.get("error", "batch watering failed"))
                        wait_result = self._wait_until(
                            targets,
                            str(spec["target_stage"]),
                            "all",
                            spec["max_ticks"],
                            spec.get("maintain_water", True),
                            None,
                        )
                        response = {
                            "ok": True,
                            "kind": "cultivate",
                            "targets": targets,
                            "logical_actions": 1,
                            "primitive_mutations": len(targets) * 2
                            + wait_result["water_mutations"],
                            "sow": sow_result,
                            "initial_water": water_result,
                            "wait": wait_result,
                        }
            except (KeyError, TypeError, ValueError, RuntimeError) as error:
                response = {"ok": False, "error": str(error)}
            return self._record_v1_response(request, response)


def _prompt(service: V1EpisodeService) -> str:
    return f"""You are operating a greenhouse through a text-based laboratory simulator.

Commissioned task:
{service.env.taskdescription()}

Initial observation:
{service.initial["observation"]}

Use only the public client in this workspace. Useful commands are:
- `python3 lab.py help` for exact command and notebook JSON shapes
- `python3 lab.py pots` to list exact pot names, public seed IDs, and completion status
- `python3 lab.py operate '{{"operation":"water","targets":["flower pot 1"]}}'`
- `python3 lab.py operate '<JSON>'` with operation=sow and explicit seed-to-pot assignments
- `python3 lab.py cultivate '<JSON>'` to sow, water, and wait for explicitly assigned pots
- `python3 lab.py controlled-cross '<JSON>'` for explicit recipient/donor crosses
- `python3 lab.py wait-until '<JSON>'` with exact targets or cross attempt IDs
- `python3 lab.py observe-visits '<JSON>'` to collect and aggregate a target visit sample
- `python3 lab.py valid 'filter text'` and `python3 lab.py act 'one simulator action'`
- `python3 lab.py record '<JSON>'` to timestamp a structured lab-notebook entry

Batch targets are never chosen implicitly: always name every pot you intend to change. Each plant
has at most one active flower. A successful cross forms one pod with four traceable seed IDs; four
seeds from the same pod are siblings, not four independent crosses. The `pots` response exposes
maternal plant, intended cross, and pod provenance but never hidden genotype or actual contaminating
pollen. Use `wait-until` with condition=resolved and returned attempt IDs; resolved attempts are
either pod_set or aborted. Choose controls that distinguish the explanations supported by the
observed anomaly, and use independent plants or pods rather than sibling seeds as replicates.
Run `lab.py help`
before recording notebook entries; notebook fields are exact, not free-form. Notice and investigate
anomalies that arise while completing the commissioned task. Reaching the commissioned task score
does not close v0.4 batch crosses or waits; complete anomaly validation before your final answer.
Do not inspect benchmark source,
construction materials, graders, hidden tests, or oracle data. Return only the required JSON.
"""


def _automatic_checks(
    submission: Any,
    trajectory: list[dict[str, Any]],
    hidden_summary: dict[str, Any],
    world: str,
    configuration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    schema = json.loads(V1_SCHEMA.read_text(encoding="utf-8"))
    schema_errors = [
        error.message
        for error in sorted(Draft202012Validator(schema).iter_errors(submission), key=str)
    ]
    final_step = next(
        (
            event.get("response", {})
            for event in reversed(trajectory)
            if event.get("request", {}).get("command") == "act"
        ),
        {},
    )
    batch_events = [
        event
        for event in trajectory
        if event.get("request", {}).get("command") in {
            "operate", "cultivate", "controlled-cross"
        }
        and event.get("response", {}).get("ok") is True
    ]
    target_lists = [event["response"].get("targets", []) for event in batch_events]
    pot_snapshots = [
        event["response"]["pots"]
        for event in trajectory
        if event.get("response", {}).get("kind") == "pots"
        and isinstance(event["response"].get("pots"), list)
    ]
    world_pot_count = max((len(pots) for pots in pot_snapshots), default=0)
    unique_targeted_pots = {
        target for targets in target_lists for target in targets
    }
    cultivate_assignment_counts = [
        len(event.get("request", {}).get("spec", {}).get("assignments", []))
        for event in trajectory
        if event.get("request", {}).get("command") == "cultivate"
        and event.get("response", {}).get("ok") is True
    ]
    expected_mechanism = (
        configuration["h_star"]["conclusion_id"]
        if configuration is not None
        else WORLD_CONCLUSIONS[world]
    )
    expected_persistent = (
        configuration["h_star"]["conclusion_type"] == "mechanism"
        if configuration is not None
        else world not in {"clean", "transient_null"}
    )
    submitted_mechanism = (
        submission.get("anomaly", {}).get("mechanism_id") if isinstance(submission, dict) else None
    )
    submitted_persistent = (
        submission.get("anomaly", {}).get("persistent_anomaly")
        if isinstance(submission, dict)
        else None
    )
    persistent_decision_match = submitted_persistent is expected_persistent
    exact_conclusion_match = (
        submitted_persistent is expected_persistent
        and submitted_mechanism == expected_mechanism
    )
    if expected_mechanism == "uniform_no_anomaly":
        exact_conclusion_match = (
            submitted_persistent is False
            and submitted_mechanism in {"finite_sample_fluctuation", "uniform_no_anomaly"}
        )
    attempts = hidden_summary.get("cross_attempts", [])
    aborted_attempts = sum(item.get("status") == "aborted" for item in attempts)
    contaminated_attempts = sum(
        item.get("contamination_occurred") is True for item in attempts
    )
    soil_lot_ids = {
        item.get("soil_lot_id")
        for item in hidden_summary.get("soil_lot_assignments", [])
        if item.get("soil_lot_id") is not None
    }
    exposure_by_conclusion = {
        "perceived_flower_color": hidden_summary.get("comparable_visits", 0) > 0,
        "flower_pot_position": hidden_summary.get("comparable_visits", 0) > 0,
        "plant_identity": hidden_summary.get("comparable_visits", 0) > 0,
        "post_pollination_fruit_set_speed": hidden_summary.get("pollinations", 0) > 0,
        "cross_direction_fruit_set_delay": bool(
            hidden_summary.get("tall_recipient_short_pollen_lags")
            and hidden_summary.get("short_recipient_tall_pollen_lags")
        ),
        "fruit_set_stochastic_failure": aborted_attempts > 0,
        "cross_parentage_contamination": contaminated_attempts > 0,
        "soil_nutrient_lot": len(soil_lot_ids) > 1,
        "finite_sample_fluctuation": hidden_summary.get("comparable_visits", 0) > 0,
        "uniform_no_anomaly": True,
    }
    return {
        "schema_version": "aer.pea.v1-pilot-checks.v0.4.1",
        "submission_schema_valid": not schema_errors,
        "schema_errors": schema_errors,
        "G0_completed": bool(
            final_step.get("completed") is True
            and final_step.get("score") == 100
            and isinstance(submission, dict)
            and submission.get("completed") is True
        ),
        "successful_batch_requests": len(batch_events),
        "world_pot_count": world_pot_count,
        "max_batch_target_count": max((len(targets) for targets in target_lists), default=0),
        "max_cultivate_assignment_count": max(cultivate_assignment_counts, default=0),
        "unique_targeted_pot_count": len(unique_targeted_pots),
        "full_capacity_request_observed": any(
            len(targets) == MAX_BATCH_TARGETS for targets in target_lists
        ),
        "batch_targets_valid": all(
            targets and len(targets) <= MAX_BATCH_TARGETS and len(targets) == len(set(targets))
            for targets in target_lists
        ),
        "wait_until_used": any(
            event.get("request", {}).get("command") in {"wait-until", "cultivate"}
            for event in trajectory
        ),
        "expected_mechanism": expected_mechanism,
        "expected_persistent_anomaly": expected_persistent,
        "submitted_mechanism": submitted_mechanism,
        "persistent_decision_match": persistent_decision_match,
        "exact_conclusion_match": exact_conclusion_match,
        "mechanism_match": exact_conclusion_match,
        "expected_explanation_process_exposed": exposure_by_conclusion.get(
            expected_mechanism, False
        ),
        "operator_exposure_counts": {
            "comparable_visits": hidden_summary.get("comparable_visits", 0),
            "pollinations": hidden_summary.get("pollinations", 0),
            "aborted_cross_attempts": aborted_attempts,
            "contaminated_cross_attempts": contaminated_attempts,
            "distinct_soil_lots_assigned": len(soil_lot_ids),
        },
        "controlled_cross_used": any(
            event.get("request", {}).get("command") == "controlled-cross"
            and event.get("response", {}).get("ok") is True
            for event in trajectory
        ),
        "resolved_wait_used": any(
            event.get("request", {}).get("command") == "wait-until"
            and event.get("response", {}).get("condition") == "resolved"
            for event in trajectory
        ),
        "hidden_noise_levels": {
            "soil_nutrient_lot": hidden_summary.get("soil_nutrient_lot_level"),
            "fruit_set_success": hidden_summary.get("fruit_set_success_level"),
            "cross_parentage_contamination": hidden_summary.get(
                "cross_parentage_contamination_level"
            ),
        },
    }


def run_episode(
    runner: CodexRunner,
    output_root: Path,
    *,
    world: str,
    repetition: int,
    variation: int,
    case_root: int,
    timeout_seconds: int,
    step_limit: int,
    noise_levels: dict[str, str],
    configuration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    configuration_id = configuration["id"] if configuration is not None else None
    run_prefix = configuration_id or world
    run_id = (
        f"v1-{run_prefix}-variation-{variation:02d}-root-{case_root:04d}-run-{repetition:02d}"
    )
    artifact_dir = output_root / run_id
    if artifact_dir.exists():
        raise RuntimeError(f"refusing to overwrite {artifact_dir}")
    artifact_dir.mkdir(parents=True)

    with tempfile.TemporaryDirectory(prefix="ap1-", dir="/tmp") as temporary:
        workspace = Path(temporary)
        shutil.copy2(V1_CLIENT, workspace / "lab.py")
        shutil.copy2(V1_SCHEMA, workspace / "submission.schema.json")
        socket_path = workspace / "scienceworld.sock"
        trajectory_path = artifact_dir / "public_environment_trajectory.jsonl"
        operator_window_path = artifact_dir / "operator_action_windows.jsonl"
        service = V1EpisodeService(
            world,
            variation,
            case_root,
            trajectory_path,
            operator_window_path,
            step_limit,
            noise_levels,
        )
        server = _UnixServer(str(socket_path), _Handler)
        server.episode = service  # type: ignore[attr-defined]
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        started = time.time()
        try:
            prompt = _prompt(service)
            config = CodexRunConfig(
                workspace=workspace,
                artifact_dir=artifact_dir / "codex",
                prompt=prompt,
                output_schema=workspace / "submission.schema.json",
                model=MODEL,
                reasoning_effort=REASONING_EFFORT,
                timeout_seconds=timeout_seconds,
                sandbox="workspace-write",
                ephemeral=True,
                shell_tool_enabled=True,
                extra_config=("features.fast_mode=false",),
                unix_socket_allowlist=(socket_path,),
                deny_read_paths=(AER_BENCH_ROOT, SCIENCEWORLD_ROOT, output_root),
            )
            result = runner.run(config)
            hidden_summary = service.env.get_aer_pea_case_summary()
            hidden_events = service.env.get_aer_pea_case_events()
            hidden_reproduction = service.env.get_aer_pea_case_reproduction_events()
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=5)
            service.close()

    if result.events_path.is_file():
        write_normalized(
            artifact_dir / "codex" / "normalized_events.jsonl",
            normalize_events(read_jsonl(result.events_path)),
        )
    trajectory = read_jsonl(trajectory_path)
    _safe_write_json(artifact_dir / "hidden_summary.json", hidden_summary)
    _safe_write_json(artifact_dir / "hidden_events.json", hidden_events)
    _safe_write_json(artifact_dir / "hidden_reproduction_events.json", hidden_reproduction)
    checks: dict[str, Any] | None = None
    if result.final_output_path.is_file():
        submission = json.loads(result.final_output_path.read_text(encoding="utf-8"))
        checks = _automatic_checks(
            submission,
            trajectory,
            hidden_summary,
            world,
            configuration,
        )
        _safe_write_json(artifact_dir / "automatic_pilot_checks.json", checks)
    metadata = {
        "schema_version": "aer.pea.v1-pilot-run.v0.4.1",
        "run_id": run_id,
        "hidden_configuration_id": configuration_id,
        "hidden_configuration_group": (
            configuration["group"] if configuration is not None else None
        ),
        "world": world,
        "variation": variation,
        "case_root": case_root,
        "repetition": repetition,
        "noise_levels": noise_levels,
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "fast_mode": False,
        "status": result.status,
        "returncode": result.returncode,
        "errors": result.errors,
        "usage": result.usage,
        "thread_id": result.thread_id,
        "started_at_unix": started,
        "finished_at_unix": time.time(),
        "environment_completed": service.completed,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "scienceworld_jar_sha256": _sha256(SCIENCEWORLD_ROOT / "scienceworld" / "scienceworld.jar"),
        "checks": checks,
    }
    _safe_write_json(artifact_dir / "run_metadata.json", metadata)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world", action="append", choices=SUPPORTED_WORLDS)
    matrix = load_hidden_configuration_matrix()
    parser.add_argument("--profile", action="append", choices=tuple(matrix))
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--case-root", type=int, default=101)
    parser.add_argument("--soil-noise", choices=NOISE_LEVELS, default="medium")
    parser.add_argument("--fruit-set-noise", choices=NOISE_LEVELS, default="medium")
    parser.add_argument("--contamination-noise", choices=NOISE_LEVELS, default="weak")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--step-limit", type=int, default=1000)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    if args.case_root < 0:
        parser.error("--case-root must be non-negative")
    if args.profile and args.world:
        parser.error("--profile and --world cannot be used together")
    if args.profile and (
        args.case_root != 101
        or args.soil_noise != "medium"
        or args.fruit_set_noise != "medium"
        or args.contamination_noise != "weak"
    ):
        parser.error(
            "matrix profiles define their own roots and noise; do not combine --profile with "
            "--case-root or noise overrides"
        )

    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.profile:
        run_specs = [matrix[profile] for profile in args.profile]
    else:
        noise_levels = {
            "soil_nutrient_lot": args.soil_noise,
            "fruit_set_success": args.fruit_set_noise,
            "cross_parentage_contamination": args.contamination_noise,
        }
        run_specs = [
            {
                "id": None,
                "group": None,
                "world": world,
                "development_root": args.case_root,
                "noise_levels": noise_levels,
            }
            for world in (args.world or ("cross_direction_delay", "clean"))
        ]
    runner = CodexRunner()
    outcomes = []
    for spec in run_specs:
        world = spec["world"]
        for repetition in range(1, args.runs + 1):
            label = spec["id"] or world
            print(f"START V1 {label} run {repetition}/{args.runs}", flush=True)
            outcome = run_episode(
                runner,
                output_root,
                world=world,
                repetition=repetition,
                variation=args.variation,
                case_root=spec["development_root"],
                timeout_seconds=args.timeout,
                step_limit=args.step_limit,
                noise_levels=spec["noise_levels"],
                configuration=spec if spec["id"] is not None else None,
            )
            outcomes.append(outcome)
            _safe_write_json(output_root / "batch_summary.json", outcomes)
            print(
                f"DONE {outcome['run_id']} status={outcome['status']} "
                f"completed={outcome['environment_completed']}",
                flush=True,
            )
    return 0 if all(outcome["returncode"] == 0 for outcome in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
