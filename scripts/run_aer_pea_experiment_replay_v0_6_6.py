#!/usr/bin/env python3
"""Run the frozen v0.6.6 post-hoc Experiment replay pilot.

The runner reconstructs the archived Sol R1 ``M2-P`` state immediately before the
registered solver request, branches only the future AER mechanism/random streams, and
replays that complete public request under eight candidate profiles and a frozen seed
manifest.  A worker owns one long-lived ScienceWorld JVM and resets it between cells.

This is a development-only pilot.  It makes no model calls and cannot produce an official
leaderboard result while the registered multiplicity policy remains uncalibrated.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import multiprocessing
import os
import re
import sys
import tempfile
import time
import traceback
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
SCIENCEWORLD_ROOT = SCRIPT_PATH.parents[1]
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]
sys.path.insert(0, str(SCRIPT_PATH.parent))
sys.path.insert(0, str(SCIENCEWORLD_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
for external_site_packages in sorted(
    (SCIENCEWORLD_ROOT / ".venv/lib").glob("python*/site-packages")
):
    # The repository Python is 3.12, while the legacy ScienceWorld venv supplies the
    # pure-Python Py4J dependency.  Append (do not prepend) it so binary 3.9 wheels in
    # that environment cannot shadow the repository's 3.12 dependencies.
    sys.path.append(str(external_site_packages))

import run_aer_pea_formulation_v0_5 as v05  # noqa: E402

from aer_bench.pea_experiment_replay_v066 import (  # noqa: E402
    CANDIDATE_WORLD_IDS,
    OPERATOR_PRESERVES,
    OPERATOR_RESETS,
    PILOT_MEASURE_BINDINGS,
    PREFERENCE_WEIGHT,
    ExperimentReplayProfile,
    ExperimentReplayValidationError,
    aggregate_pair,
    aggregate_task,
    read_jsonl,
    score_experiment_pair,
    validate_contract,
)

CASE_CONSTRUCTION = (
    REPOSITORY_ROOT
    / "cases/science/mendelian_genetics_known_plant_aer/revisions/v1_development/construction"
)
DEFAULT_CONTRACT = CASE_CONSTRUCTION / "experiment-replay-contract.v0.6.6-development.json"
CURRENT_JAR = SCIENCEWORLD_ROOT / "scienceworld/scienceworld.jar"
SCORING_HELPER_PATH = REPOSITORY_ROOT / "src/aer_bench/pea_experiment_replay_v066.py"
REPLAY_IMPLEMENTATION_PATH = SCRIPT_PATH.parent / "run_aer_pea_formulation_v0_5.py"

FROZEN_CONTRACT_SHA256 = (
    "0c6ae3cb9172fba1875ac4bad0a5bbd36ed6ef49e3404b88947b5a06ca99ac4b"
)
DIRECT_IMPLEMENTATION_PATHS = {
    "scoring_helper": SCORING_HELPER_PATH,
    "construction_service_and_replay_recipe": REPLAY_IMPLEMENTATION_PATH,
}

PLAN_SCHEMA = "aer.pea.experiment-replay-plan.v0.6.6-development"
SAMPLE_SCHEMA = "aer.pea.experiment-replay-sample.v0.6.6-development"
SUMMARY_SCHEMA = "aer.pea.experiment-replay-pilot-summary.v0.6.6-development"
ERROR_SCHEMA = "aer.pea.experiment-replay-cell-error.v0.6.6-development"
CELL_INVENTORY_SCHEMA = "aer.pea.experiment-replay-cell-inventory.v0.6.6-development"
RUNTIME_TELEMETRY_SCHEMA = "aer.pea.experiment-replay-runtime.v0.6.6-development"
RUNTIME_ATTEMPT_SCHEMA = (
    "aer.pea.experiment-replay-runtime-attempt.v0.6.6-development"
)
PLAN_NAME = "replay_plan.json"
SAMPLES_NAME = "samples.jsonl"
SUMMARY_NAME = "summary.json"
CELL_INVENTORY_NAME = "cell_inventory.json"
RUNTIME_TELEMETRY_NAME = "runtime_telemetry.json"
NO_NOISE = {
    "soil_nutrient_lot": "none",
    "fruit_set_success": "none",
    "cross_parentage_contamination": "none",
}
NOISE_KEYS = frozenset(NO_NOISE)
NOISE_LEVELS = frozenset({"none", "weak", "medium", "strong"})
STATE_AFFECTING_COMMANDS = frozenset(
    {
        "act",
        "batch",
        "record",
        "operate",
        "cultivate",
        "controlled-cross",
        "wait-until",
        "observe-visits",
        "submit",
    }
)

RUNTIME_ATTEMPT_FIELDS = frozenset(
    {
        "schema_version",
        "attempt_number",
        "legacy_migrated",
        "status",
        "started_at_utc",
        "finished_at_utc",
        "elapsed_seconds",
        "requested_worker_count",
        "pool_worker_limit",
        "actual_worker_count",
        "scheduled_cell_count",
        "resumed_cell_count",
        "successful_executed_cell_count",
        "failed_cell_count",
        "worker_assignments",
        "error_type",
        "error",
    }
)
LEGACY_RUNTIME_FIELDS = frozenset(
    {
        "schema_version",
        "run_fingerprint",
        "status",
        "started_at_utc",
        "finished_at_utc",
        "elapsed_seconds",
        "requested_worker_count",
        "pool_worker_limit",
        "actual_worker_count",
        "scheduled_cell_count",
        "resumed_cell_count",
        "successful_executed_cell_count",
        "failed_cell_count",
        "worker_assignments",
        "error_type",
        "error",
    }
)
RUNTIME_TELEMETRY_FIELDS = LEGACY_RUNTIME_FIELDS | frozenset(
    {
        "attempt_count",
        "execution_attempt_count",
        "no_op_attempt_count",
        "complete_attempt_count",
        "failed_attempt_count",
        "running_attempt_count",
        "cumulative_elapsed_seconds",
        "cumulative_scheduled_cell_count",
        "cumulative_successful_executed_cell_count",
        "cumulative_failed_cell_count",
        "cumulative_actual_worker_count",
        "cumulative_worker_assignments",
        "first_started_at_utc",
        "last_finished_at_utc",
        "attempts",
    }
)
INTERRUPTED_ATTEMPT_ERROR_TYPE = "InterruptedReplayAttempt"
INTERRUPTED_ATTEMPT_ERROR = (
    "previous replay process ended without a terminal telemetry checkpoint; "
    "marked failed when --resume recovered the run"
)


class ReplayRunnerError(RuntimeError):
    """Raised when source reconstruction or replay execution is not trustworthy."""


def _configure_java_runtime() -> Path:
    """Select the repository's supported local JDK before a worker launches Py4J."""

    candidates: list[Path] = []
    configured = os.environ.get("JAVA_HOME")
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            Path("/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"),
            Path("/usr/local/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"),
        ]
    )
    for java_home in candidates:
        java = java_home / "bin/java"
        if java.is_file() and os.access(java, os.X_OK):
            os.environ["JAVA_HOME"] = str(java_home)
            current_path = os.environ.get("PATH", "")
            if str(java.parent) not in current_path.split(os.pathsep):
                os.environ["PATH"] = str(java.parent) + os.pathsep + current_path
            return java
    raise ReplayRunnerError("a Java 17 runtime is required to launch ScienceWorld")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayRunnerError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReplayRunnerError(f"{path} must contain a JSON object")
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_json_exclusive(path: Path, value: Any) -> None:
    """Atomically materialize new JSON evidence without replacing an existing cell."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ReplayRunnerError(
                f"refusing to overwrite existing replay cell: {path}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _resolve_contract_path(raw_path: Any) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ReplayRunnerError("a bound artifact path must be a non-empty string")
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _validate_bound_artifact(label: str, value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ReplayRunnerError(f"{label} must bind exactly path and sha256")
    path = _resolve_contract_path(value["path"])
    expected = value["sha256"]
    if not path.is_file():
        raise ReplayRunnerError(f"bound artifact is missing: {path}")
    actual = _sha256_path(path)
    if actual != expected:
        raise ReplayRunnerError(f"{label} hash changed: {actual} != {expected}")
    return {"path": str(path), "sha256": actual}


def _validate_frozen_contract_path(path: Path) -> str:
    """Treat the frozen contract bytes, rather than its mutable contents, as trust root."""

    path = path.expanduser().resolve()
    if not path.is_file():
        raise ReplayRunnerError(f"frozen experiment replay contract is missing: {path}")
    actual = _sha256_path(path)
    if actual != FROZEN_CONTRACT_SHA256:
        raise ReplayRunnerError(
            "frozen experiment replay contract hash changed: "
            f"{actual} != {FROZEN_CONTRACT_SHA256}"
        )
    return actual


def _current_direct_implementation_bindings() -> dict[str, dict[str, str]]:
    """Bind the two directly imported repository implementations used by this runner."""

    bindings: dict[str, dict[str, str]] = {}
    for label, raw_path in DIRECT_IMPLEMENTATION_PATHS.items():
        path = raw_path.resolve()
        if not path.is_file():
            raise ReplayRunnerError(f"direct implementation is missing ({label}): {path}")
        bindings[label] = {"path": str(path), "sha256": _sha256_path(path)}
    return bindings


def _validate_direct_implementation_bindings(value: Any) -> dict[str, dict[str, str]]:
    """Fail closed if a saved plan's direct Python implementation changed on disk."""

    bindings = _require_mapping(value, "direct_implementation_bindings")
    if set(bindings) != set(DIRECT_IMPLEMENTATION_PATHS):
        raise ReplayRunnerError("direct implementation binding names changed")
    validated: dict[str, dict[str, str]] = {}
    for label, registered_path in DIRECT_IMPLEMENTATION_PATHS.items():
        binding = _require_mapping(
            bindings.get(label), f"direct_implementation_bindings.{label}"
        )
        _require_exact_fields(
            binding,
            frozenset({"path", "sha256"}),
            f"direct_implementation_bindings.{label}",
        )
        expected_path = registered_path.resolve()
        actual_path = Path(str(binding.get("path"))).expanduser().resolve()
        if actual_path != expected_path:
            raise ReplayRunnerError(f"direct implementation path changed ({label})")
        expected_hash = _require_sha256(
            binding.get("sha256"),
            f"direct_implementation_bindings.{label}.sha256",
        )
        if not actual_path.is_file():
            raise ReplayRunnerError(
                f"direct implementation is missing ({label}): {actual_path}"
            )
        actual_hash = _sha256_path(actual_path)
        if actual_hash != expected_hash:
            raise ReplayRunnerError(f"direct implementation hash changed ({label})")
        validated[label] = {"path": str(actual_path), "sha256": actual_hash}
    return validated


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReplayRunnerError(f"{label} must be an object")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReplayRunnerError(f"{label} must be an integer >= {minimum}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReplayRunnerError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ReplayRunnerError(f"{label} fields changed: missing={missing}, extra={extra}")


def _require_finite_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        raise ReplayRunnerError(f"{label} must be a finite number")
    return float(value)


def _parse_candidate_profiles(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    replay = _require_mapping(contract.get("replay"), "replay")
    preference_weight = _require_finite_number(
        replay.get("preference_weight"), "replay.preference_weight"
    )
    if preference_weight != PREFERENCE_WEIGHT:
        raise ReplayRunnerError("replay preference_weight must remain 9.0")
    raw = contract.get("candidate_worlds")
    if not isinstance(raw, list):
        raise ReplayRunnerError("candidate_worlds must be a list")
    profiles: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        candidate = _require_mapping(item, f"candidate_worlds[{index}]")
        identifier = candidate.get("id")
        world = candidate.get("world")
        noise = candidate.get("noise_levels")
        if identifier != CANDIDATE_WORLD_IDS[index]:
            raise ReplayRunnerError("candidate World order changed")
        if not isinstance(world, str) or not world:
            raise ReplayRunnerError(f"candidate World {identifier} has no simulator world")
        if not isinstance(noise, Mapping) or set(noise) != NOISE_KEYS:
            raise ReplayRunnerError(f"candidate World {identifier} has invalid noise axes")
        if any(level not in NOISE_LEVELS for level in noise.values()):
            raise ReplayRunnerError(f"candidate World {identifier} has an invalid noise level")
        profiles.append(
            {
                "id": identifier,
                "world": world,
                "preference_weight": preference_weight,
                "noise_levels": dict(noise),
            }
        )
    return profiles


def _parse_measure(pilot: Mapping[str, Any], target_request: Mapping[str, Any]) -> dict[str, Any]:
    measure = _require_mapping(pilot.get("measure"), "pilot_scope.measure")
    if set(measure) != set(PILOT_MEASURE_BINDINGS):
        raise ReplayRunnerError("the pilot Measure fields changed")
    for key, expected in PILOT_MEASURE_BINDINGS.items():
        actual = measure.get(key)
        if key == "hidden_fields_forbidden":
            if not isinstance(actual, list) or tuple(actual) != expected:
                raise ReplayRunnerError("the Measure hidden-field denylist changed")
        elif actual != expected:
            raise ReplayRunnerError(f"the pilot Measure {key} changed")
    forbidden = measure["hidden_fields_forbidden"]

    numerator_names = re.findall(r"flower pot \d+", str(measure.get("numerator", "")))
    denominator_names = re.findall(r"flower pot \d+", str(measure.get("denominator", "")))
    if len(numerator_names) != 1 or len(set(denominator_names)) != 2:
        raise ReplayRunnerError("the registered visit-share pots cannot be parsed")
    spec = _require_mapping(target_request.get("spec"), "registered request spec")
    targets = spec.get("targets")
    if not isinstance(targets, list) or set(targets) != set(denominator_names):
        raise ReplayRunnerError("the registered denominator differs from request targets")
    if numerator_names[0] not in denominator_names:
        raise ReplayRunnerError("the registered numerator is outside the denominator")
    return {
        "id": measure.get("id"),
        "input": measure.get("input"),
        "output_field": "registered_visit_share",
        "grouping_field": "flower_pot",
        "numerator_target": numerator_names[0],
        "denominator_targets": list(dict.fromkeys(denominator_names)),
        "missing_encoded_value": 0.5,
        "hidden_fields_forbidden": list(forbidden),
    }


def _without_volatile_timestamps(value: Any) -> Any:
    """Drop only wall-clock timestamps while preserving complete semantic payloads."""

    if isinstance(value, Mapping):
        return {
            str(key): _without_volatile_timestamps(item)
            for key, item in value.items()
            if key != "timestamp_unix"
        }
    if isinstance(value, (list, tuple)):
        return [_without_volatile_timestamps(item) for item in value]
    return value


def _stable_response_projection(request: Any, response: Any) -> tuple[Any, list[str]]:
    """Project only the known nondeterministic action-enumeration surface."""

    normalized = _without_volatile_timestamps(response)
    command = request.get("command") if isinstance(request, Mapping) else None
    excluded: list[str] = []
    if command in {"actions", "valid"} and isinstance(normalized, Mapping):
        normalized = dict(normalized)
        if "actions" in normalized:
            normalized.pop("actions")
            excluded.append("actions")
    return normalized, excluded


def _normalized_public_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        response, excluded = _stable_response_projection(
            row.get("request"), row.get("response")
        )
        normalized.append(
            {
                "index": row.get("index"),
                "source": row.get("source"),
                "request": _without_volatile_timestamps(row.get("request")),
                "response": response,
                "excluded_response_fields": excluded,
            }
        )
    return normalized


def _source_action_id(pilot: Mapping[str, Any]) -> str:
    task_id = str(pilot["task_id"])
    experiment_id = str(pilot["experiment_id"])
    request_index = int(pilot["solver_public_request_index"])
    return f"{task_id}:{experiment_id}:REQ-{request_index:05d}"


def _build_plan_skeleton(
    contract: Mapping[str, Any],
    profile: ExperimentReplayProfile,
    *,
    contract_path: Path,
    seed_count: int,
) -> dict[str, Any]:
    contract_sha256 = _validate_frozen_contract_path(contract_path)
    pilot = _require_mapping(contract.get("pilot_scope"), "pilot_scope")
    source_artifacts_raw = _require_mapping(contract.get("source_artifacts"), "source_artifacts")
    expected_artifacts = {
        "public_environment_trajectory",
        "operator_action_windows",
        "run_manifest",
        "handoff_replay_recipe",
    }
    if set(source_artifacts_raw) != expected_artifacts:
        raise ReplayRunnerError("source_artifacts fields changed")
    source_artifacts = {
        name: _validate_bound_artifact(f"source_artifacts.{name}", source_artifacts_raw[name])
        for name in sorted(expected_artifacts)
    }

    source_study = _require_mapping(contract.get("source_study"), "source_study")
    source_study_bindings = {
        name: _validate_bound_artifact(f"source_study.{name}", source_study[name])
        for name in ("summary", "artifact_inventory")
    }
    operator = _require_mapping(contract.get("operator_implementation"), "operator_implementation")
    if operator.get("branch_api") != "configure_aer_pea_replay_branch":
        raise ReplayRunnerError("operator replay branch API changed")
    preserves = operator.get("preserves")
    if not isinstance(preserves, list) or tuple(preserves) != OPERATOR_PRESERVES:
        raise ReplayRunnerError("operator replay preserved-state contract changed")
    resets = operator.get("resets")
    if not isinstance(resets, list) or tuple(resets) != OPERATOR_RESETS:
        raise ReplayRunnerError("operator replay reset contract changed")
    if operator.get("double_restore_required") is not True:
        raise ReplayRunnerError("the frozen pilot requires a double restore")
    expected_jar = operator.get("scienceworld_jar_sha256")
    actual_jar = _sha256_path(CURRENT_JAR)
    if actual_jar != expected_jar:
        raise ReplayRunnerError(f"ScienceWorld jar hash changed: {actual_jar} != {expected_jar}")

    public_rows = read_jsonl(Path(source_artifacts["public_environment_trajectory"]["path"]))
    operator_rows = read_jsonl(Path(source_artifacts["operator_action_windows"]["path"]))
    source_manifest = _read_json(Path(source_artifacts["run_manifest"]["path"]))
    if source_manifest.get("task_id") != pilot.get("task_id"):
        raise ReplayRunnerError("source run manifest Task identity changed")
    if source_manifest.get("model") != pilot.get("source_model"):
        raise ReplayRunnerError("source run manifest model identity changed")
    request_index = _require_int(
        pilot.get("solver_public_request_index"), "solver_public_request_index"
    )
    if request_index >= len(public_rows):
        raise ReplayRunnerError("registered solver request index is outside the trajectory")
    target_row = public_rows[request_index]
    if target_row.get("index") != request_index or target_row.get("source") != "solver":
        raise ReplayRunnerError("registered request does not select a solver top-level row")
    target_request = _require_mapping(target_row.get("request"), "registered solver request")
    if target_request.get("command") != pilot.get("solver_public_request_command"):
        raise ReplayRunnerError("registered solver request command changed")
    if target_row.get("response", {}).get("ok") is not True:
        raise ReplayRunnerError("registered source request was not successful")

    primitive_start = _require_int(
        pilot.get("primitive_public_index_start"), "primitive_public_index_start"
    )
    primitive_end = _require_int(
        pilot.get("primitive_public_index_end_inclusive"),
        "primitive_public_index_end_inclusive",
    )
    expected_action_ids = pilot.get("primitive_action_ids")
    if not isinstance(expected_action_ids, list) or any(
        not isinstance(item, str) or not item for item in expected_action_ids
    ):
        raise ReplayRunnerError("primitive_action_ids must be a list of strings")
    selected_operator_rows = [
        row
        for row in operator_rows
        if primitive_start <= int(row.get("public_trajectory_index", -1)) <= primitive_end
    ]
    action_ids = [row.get("action_id") for row in selected_operator_rows]
    if action_ids != expected_action_ids:
        raise ReplayRunnerError("archived primitive expansion differs from the contract")
    if [row.get("public_trajectory_index") for row in selected_operator_rows] != list(
        range(primitive_start, primitive_end + 1)
    ):
        raise ReplayRunnerError("archived primitive public indices are not contiguous")
    if any(
        row.get("active_experiment_id") != pilot.get("experiment_id")
        for row in selected_operator_rows
    ):
        raise ReplayRunnerError("archived primitive expansion escaped its experiment window")
    if primitive_end + 1 != request_index:
        raise ReplayRunnerError("registered macro completion does not follow its expansion")

    prefix_solver_rows = [
        row for row in public_rows[:request_index] if row.get("source") == "solver"
    ]
    if not prefix_solver_rows:
        raise ReplayRunnerError("registered source trajectory has no solver prefix")
    if int(prefix_solver_rows[-1]["index"]) + 1 != primitive_start:
        raise ReplayRunnerError(
            "registered primitive expansion has an ambiguous invocation boundary"
        )
    solver_prefix: list[dict[str, Any]] = []
    for row in prefix_solver_rows:
        request = dict(_require_mapping(row.get("request"), "solver prefix request"))
        stable_response, excluded_fields = _stable_response_projection(
            request, row.get("response")
        )
        solver_prefix.append(
            {
                "public_index": int(row["index"]),
                "request": request,
                "response_sha256": _sha256_json(
                    _without_volatile_timestamps(row.get("response"))
                ),
                "stable_response_sha256": _sha256_json(stable_response),
                "excluded_response_fields": excluded_fields,
                "state_affecting": request.get("command") in STATE_AFFECTING_COMMANDS,
            }
        )

    if (
        isinstance(seed_count, bool)
        or not isinstance(seed_count, int)
        or not 1 <= seed_count <= len(profile.seed_manifest)
    ):
        raise ReplayRunnerError(
            f"--seeds must be from 1 through {len(profile.seed_manifest)}"
        )
    seed_manifest = tuple(profile.seed_manifest[:seed_count])
    contract_seed_manifest_complete = seed_count == len(profile.seed_manifest)
    profiles = _parse_candidate_profiles(contract)
    true_world_id = pilot.get("true_world_id")
    by_id = {item["id"]: item for item in profiles}
    if true_world_id not in by_id:
        raise ReplayRunnerError("pilot true World is not a candidate profile")
    if (
        by_id[str(true_world_id)]["world"] != pilot.get("source_world")
        or pilot.get("configuration_id") != "M2-position-attraction"
    ):
        raise ReplayRunnerError("pilot source configuration changed")

    measure = _parse_measure(pilot, target_request)
    source_prefix = public_rows[:primitive_start]
    plan = {
        "schema_version": PLAN_SCHEMA,
        "status": "prepared_double_restore_pending",
        "development_only": True,
        "official_leaderboard_result": False,
        "contract": {
            "path": str(contract_path.resolve()),
            "sha256": contract_sha256,
            "validated_profile": _jsonable(asdict(profile)),
        },
        "direct_implementation_bindings": _current_direct_implementation_bindings(),
        "source_study": source_study_bindings,
        "source_artifacts": source_artifacts,
        "operator_implementation": {
            "scienceworld_jar": {"path": str(CURRENT_JAR.resolve()), "sha256": actual_jar},
            "branch_api": "configure_aer_pea_replay_branch",
            "preserves": list(OPERATOR_PRESERVES),
            "resets": list(OPERATOR_RESETS),
            "double_restore_required": True,
        },
        "source": {
            "source_model": pilot.get("source_model"),
            "source_run_root": str(_resolve_contract_path(pilot.get("source_run_root"))),
            "source_run_id": Path(str(pilot.get("source_run_root"))).name,
            "task_id": pilot.get("task_id"),
            "configuration_id": pilot.get("configuration_id"),
            "true_world_id": true_world_id,
            "source_world": pilot.get("source_world"),
            "source_case_root": _require_int(pilot.get("source_case_root"), "source_case_root"),
            "variation": _require_int(pilot.get("variation"), "variation"),
            "experiment_id": pilot.get("experiment_id"),
            "source_action_id": _source_action_id(pilot),
            "solver_public_request_index": request_index,
            "pre_action_public_index": primitive_start,
            "target_request": dict(target_request),
            "target_request_sha256": _sha256_json(target_request),
            "solver_prefix": solver_prefix,
            "archived_prefix_sha256": _sha256_json(_normalized_public_rows(source_prefix)),
            "archived_primitive_expansion": {
                "public_index_start": primitive_start,
                "public_index_end_inclusive": primitive_end,
                "action_ids": action_ids,
                "action_count": len(action_ids),
                "sha256": _sha256_json(
                    [
                        {
                            "action_id": row["action_id"],
                            "action": row["action"],
                            "public_trajectory_index": row["public_trajectory_index"],
                        }
                        for row in selected_operator_rows
                    ]
                ),
            },
        },
        "source_configuration": {
            "world": pilot.get("source_world"),
            "case_root": int(pilot["source_case_root"]),
            "variation": int(pilot["variation"]),
            "noise_levels": dict(by_id[str(true_world_id)]["noise_levels"]),
        },
        "candidate_profiles": profiles,
        "seed_manifest": list(seed_manifest),
        "seed_count_per_world": len(seed_manifest),
        "expected_sample_count": len(profiles) * len(seed_manifest),
        "run_kind": (
            "contract_seed_manifest"
            if contract_seed_manifest_complete
            else "noncontract_smoke"
        ),
        "contract_seed_manifest_complete": contract_seed_manifest_complete,
        "noncontract_smoke": not contract_seed_manifest_complete,
        "measure": measure,
        "comparison": dict(_require_mapping(contract.get("comparison"), "comparison")),
        "runner": {"path": str(SCRIPT_PATH), "sha256": _sha256_path(SCRIPT_PATH)},
    }
    return plan


class _ReusableRuntime:
    """One ScienceWorld JVM that is reset and reconstructed for many replay cells."""

    def __init__(self, plan: Mapping[str, Any]) -> None:
        _configure_java_runtime()
        self.plan = plan
        self._temporary = tempfile.TemporaryDirectory(prefix="aer-v066-replay-worker-")
        directory = Path(self._temporary.name)
        configuration = plan["source_configuration"]
        self.service = v05.ConstructionService(
            str(configuration["world"]),
            int(configuration["variation"]),
            int(configuration["case_root"]),
            directory / "public_environment_trajectory.jsonl",
            directory / "operator_action_windows.jsonl",
            2_000,
            dict(configuration["noise_levels"]),
        )
        self.service.actor = "s0_construction"

    def close(self) -> None:
        try:
            self.service.close()
        finally:
            self._temporary.cleanup()

    def reset(self) -> None:
        service = self.service
        for path in (service.trajectory_path, service.operator_window_path):
            path.unlink(missing_ok=True)
        configuration = self.plan["source_configuration"]
        service.env.configure_aer_pea_case(
            str(configuration["world"]),
            int(configuration["case_root"]),
            noise_levels=dict(configuration["noise_levels"]),
        )
        service.env.load(
            v05.TASK_NAME,
            int(configuration["variation"]),
            "easy",
            generateGoldPath=False,
        )
        service._index = 0
        service._note_index = 0
        service._experiment_ids = set()
        service._active_experiment_id = None
        service.completed = False
        service.pre_exposure_observations = []
        service.actor = "s0_construction"
        service.initial = service._step("look around", source="initial")


def _state_fingerprint(service: v05.ConstructionService) -> str:
    payload = {
        "public_status": service.env.get_aer_pea_case_public_status(),
        "look": service.env.look(),
        "inventory": service.env.inventory(),
        "aer_events": service.env.get_aer_pea_case_events(),
        "reproduction_events": service.env.get_aer_pea_case_reproduction_events(),
        "aer_summary": service.env.get_aer_pea_case_summary(),
        "service_state": {
            "public_index": service._index,
            "note_index": service._note_index,
            "experiment_ids": sorted(service._experiment_ids),
            "active_experiment_id": service._active_experiment_id,
            "completed": service.completed,
        },
    }
    return _sha256_json(payload)


def _reconstruct_pre_action(
    runtime: _ReusableRuntime,
    *,
    verify_archived_prefix: bool,
) -> str:
    plan = runtime.plan
    service = runtime.service
    runtime.reset()
    recipe_path = Path(plan["source_artifacts"]["handoff_replay_recipe"]["path"])
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if not isinstance(recipe, list):
        raise ReplayRunnerError("handoff replay recipe must be a list")
    v05._replay_recipe(service, recipe)
    service.actor = "solver"
    for item in plan["source"]["solver_prefix"]:
        response = service.handle(dict(item["request"]))
        if response.get("ok") is not True:
            raise ReplayRunnerError(
                f"solver prefix replay failed at {item['public_index']}: {response}"
            )
        actual_index = service._index - 1
        if actual_index != item["public_index"]:
            raise ReplayRunnerError(
                f"solver prefix expansion drift at {item['public_index']}: got {actual_index}"
            )
        if verify_archived_prefix:
            stable_response, excluded_fields = _stable_response_projection(
                item["request"], response
            )
            if excluded_fields != item["excluded_response_fields"]:
                raise ReplayRunnerError(
                    f"solver response projection drift at public index {item['public_index']}"
                )
            if _sha256_json(stable_response) != item["stable_response_sha256"]:
                raise ReplayRunnerError(
                    f"solver-visible response drift at public index {item['public_index']}"
                )
    expected_pre_index = int(plan["source"]["pre_action_public_index"])
    if service._index != expected_pre_index:
        raise ReplayRunnerError(
            f"reconstructed pre-action index is {service._index}, expected {expected_pre_index}"
        )
    if verify_archived_prefix:
        replayed_rows = read_jsonl(service.trajectory_path)
        replayed_hash = _sha256_json(_normalized_public_rows(replayed_rows))
        if replayed_hash != plan["source"]["archived_prefix_sha256"]:
            raise ReplayRunnerError("reconstructed public prefix differs from the archive")
    return _state_fingerprint(service)


def _finalize_plan(plan: dict[str, Any]) -> dict[str, Any]:
    runtime = _ReusableRuntime(plan)
    try:
        first = _reconstruct_pre_action(runtime, verify_archived_prefix=True)
        second = _reconstruct_pre_action(runtime, verify_archived_prefix=True)
    finally:
        runtime.close()
    if first != second:
        raise ReplayRunnerError(f"double restore mismatch: {first} != {second}")
    plan["status"] = "prepared_double_restore_verified"
    plan["restore"] = {
        "count": 2,
        "fingerprints": [first, second],
        "prestate_fingerprint": first,
        "semantic_match": "exact_sha256",
        "archived_prefix_match": True,
    }
    plan["run_fingerprint"] = _plan_fingerprint(plan)
    return plan


def _plan_fingerprint(plan: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in plan.items() if key != "run_fingerprint"}
    return _sha256_json(payload)


def _validate_saved_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute a saved plan's identity and every mutable file binding."""

    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ReplayRunnerError("saved replay plan schema changed")
    if plan.get("status") != "prepared_double_restore_verified":
        raise ReplayRunnerError("saved replay plan has not passed the double restore")
    reported = plan.get("run_fingerprint")
    actual = _plan_fingerprint(plan)
    if not isinstance(reported, str) or reported != actual:
        raise ReplayRunnerError("saved replay plan payload does not match its run_fingerprint")

    contract_binding = _require_mapping(plan.get("contract"), "saved plan contract")
    contract_path = Path(str(contract_binding.get("path"))).resolve()
    contract_sha256 = _validate_frozen_contract_path(contract_path)
    if contract_binding.get("sha256") != contract_sha256:
        raise ReplayRunnerError("saved replay plan contract binding changed")
    _validate_direct_implementation_bindings(
        plan.get("direct_implementation_bindings")
    )
    contract = _read_json(contract_path)
    try:
        profile = validate_contract(contract, repository_root=REPOSITORY_ROOT)
    except ExperimentReplayValidationError as error:
        raise ReplayRunnerError(str(error)) from error

    seeds = plan.get("seed_manifest")
    if (
        not isinstance(seeds, list)
        or not seeds
        or tuple(seeds) != tuple(profile.seed_manifest[: len(seeds)])
    ):
        raise ReplayRunnerError("saved replay plan seed manifest is not a frozen prefix")
    fresh = _build_plan_skeleton(
        contract,
        profile,
        contract_path=contract_path,
        seed_count=len(seeds),
    )
    immutable_sections = (
        "contract",
        "direct_implementation_bindings",
        "source_study",
        "source_artifacts",
        "operator_implementation",
        "source",
        "source_configuration",
        "candidate_profiles",
        "seed_manifest",
        "seed_count_per_world",
        "expected_sample_count",
        "run_kind",
        "contract_seed_manifest_complete",
        "noncontract_smoke",
        "measure",
        "comparison",
        "runner",
    )
    for section in immutable_sections:
        if _jsonable(plan.get(section)) != _jsonable(fresh.get(section)):
            raise ReplayRunnerError(f"saved replay plan {section} binding changed")
    restore = _require_mapping(plan.get("restore"), "saved plan restore")
    fingerprints = restore.get("fingerprints")
    if (
        restore.get("count") != 2
        or not isinstance(fingerprints, list)
        or len(fingerprints) != 2
        or fingerprints[0] != fingerprints[1]
        or restore.get("prestate_fingerprint") != fingerprints[0]
        or restore.get("semantic_match") != "exact_sha256"
        or restore.get("archived_prefix_match") is not True
    ):
        raise ReplayRunnerError("saved replay plan double-restore evidence is invalid")
    verified = _finalize_plan(fresh)
    if _canonical_bytes(plan) != _canonical_bytes(verified):
        raise ReplayRunnerError(
            "saved replay plan differs from a freshly reconstructed double restore"
        )
    return verified


def load_validated_plan(output_root: Path) -> dict[str, Any]:
    return _validate_saved_plan(_read_json(output_root / PLAN_NAME))


def prepare_plan(
    contract_path: Path,
    output_root: Path,
    *,
    seed_count: int,
    resume: bool,
) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    plan_path = output_root / PLAN_NAME
    if plan_path.exists():
        existing = load_validated_plan(output_root)
        if Path(existing["contract"]["path"]).resolve() != contract_path:
            raise ReplayRunnerError("existing replay plan uses a different contract path")
        if existing["seed_count_per_world"] != seed_count:
            raise ReplayRunnerError("existing replay plan uses a different seed count")
        return existing

    _validate_frozen_contract_path(contract_path)
    contract = _read_json(contract_path)
    try:
        profile = validate_contract(contract, repository_root=REPOSITORY_ROOT)
    except ExperimentReplayValidationError as error:
        raise ReplayRunnerError(str(error)) from error
    skeleton = _build_plan_skeleton(
        contract, profile, contract_path=contract_path, seed_count=seed_count
    )
    plan = _finalize_plan(skeleton)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(plan_path, plan)
    return plan


def _measure_visit_share(
    response: Mapping[str, Any], measure: Mapping[str, Any]
) -> tuple[float | None, dict[str, int], str | None]:
    raw = response.get("visits_by_flower")
    if not isinstance(raw, list):
        raise ReplayRunnerError("observe-visits response lacks solver-visible visits_by_flower")
    denominator_targets = set(measure["denominator_targets"])
    numerator_target = measure["numerator_target"]
    counts = {target: 0 for target in measure["denominator_targets"]}
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ReplayRunnerError(f"visits_by_flower[{index}] is not an object")
        flower_pot = item.get("flower_pot")
        visit_count = item.get("visit_count")
        if (
            not isinstance(flower_pot, str)
            or isinstance(visit_count, bool)
            or not isinstance(visit_count, int)
            or visit_count < 0
        ):
            raise ReplayRunnerError("solver-visible visit aggregate is malformed")
        if flower_pot not in denominator_targets:
            raise ReplayRunnerError(
                "solver-visible visit aggregate contains an unregistered flower pot"
            )
        counts[flower_pot] += visit_count
    denominator = sum(counts.values())
    observed = response.get("observed_visit_count")
    if isinstance(observed, bool) or not isinstance(observed, int) or observed < 0:
        raise ReplayRunnerError("observe-visits observed_visit_count is malformed")
    if denominator != observed:
        raise ReplayRunnerError(
            "solver-visible visit aggregates disagree with observed_visit_count"
        )
    if denominator == 0:
        return None, counts, "no_registered_visits"
    return counts[numerator_target] / denominator, counts, None


def _operator_rows(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.is_file() else []


def _sample_content_sha256(row: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in row.items() if key != "content_sha256"}
    return _sha256_json(payload)


def _run_cell(runtime: _ReusableRuntime, world_id: str, seed: int) -> dict[str, Any]:
    plan = runtime.plan
    profile = next(item for item in plan["candidate_profiles"] if item["id"] == world_id)
    prestate_fingerprint = _reconstruct_pre_action(
        runtime, verify_archived_prefix=False
    )
    expected_fingerprint = plan["restore"]["prestate_fingerprint"]
    if prestate_fingerprint != expected_fingerprint:
        raise ReplayRunnerError(
            f"prestate fingerprint drift for {world_id}/{seed}: "
            f"{prestate_fingerprint} != {expected_fingerprint}"
        )
    service = runtime.service
    before_public_index = service._index
    before_operator_count = len(_operator_rows(service.operator_window_path))
    branch_receipt = service.env.configure_aer_pea_replay_branch(
        str(profile["world"]),
        seed,
        preference_weight=float(profile["preference_weight"]),
        noise_levels=dict(profile["noise_levels"]),
    )
    response = service.handle(dict(plan["source"]["target_request"]))
    if response.get("ok") is not True:
        raise ReplayRunnerError(f"registered solver request failed: {response}")
    value, visit_counts, missing_reason = _measure_visit_share(response, plan["measure"])
    all_operator_rows = _operator_rows(service.operator_window_path)
    new_operator_rows = all_operator_rows[before_operator_count:]
    primitives = [
        {
            "action_id": row["action_id"],
            "action": row["action"],
            "relative_public_index": int(row["public_trajectory_index"]) - before_public_index,
        }
        for row in new_operator_rows
    ]
    encoded_value = plan["measure"]["missing_encoded_value"] if value is None else value
    row = {
        "schema_version": SAMPLE_SCHEMA,
        "run_fingerprint": plan["run_fingerprint"],
        "run_id": plan["source"]["source_run_id"],
        "task_id": plan["source"]["task_id"],
        "experiment_id": plan["source"]["experiment_id"],
        "source_action_id": plan["source"]["source_action_id"],
        "action_request_index": plan["source"]["solver_public_request_index"],
        "action_request_sha256": plan["source"]["target_request_sha256"],
        "world_id": world_id,
        "world": profile["world"],
        "preference_weight": profile["preference_weight"],
        "noise_levels": profile["noise_levels"],
        "seed": seed,
        "status": "missing" if value is None else "ok",
        "value": value,
        "registered_visit_share": value,
        "encoded_registered_visit_share": encoded_value,
        "missing_reason": missing_reason,
        "registered_visit_counts": visit_counts,
        "prestate_fingerprint": prestate_fingerprint,
        "branch_receipt": branch_receipt,
        "primitive_action_count": len(primitives),
        "primitive_actions": primitives,
        "primitive_actions_sha256": _sha256_json(primitives),
        "solver_response": {
            "kind": response.get("kind"),
            "requested_visit_count": response.get("requested_visit_count"),
            "observed_visit_count": response.get("observed_visit_count"),
            "out_of_scope_visit_count": response.get("out_of_scope_visit_count"),
            "elapsed_ticks": response.get("elapsed_ticks"),
            "timed_out": response.get("timed_out"),
        },
        "measure_id": plan["measure"]["id"],
        "measure_input": plan["measure"]["input"],
        "source_artifacts": plan["source_artifacts"],
        "contract_seed_manifest_complete": plan["contract_seed_manifest_complete"],
        "noncontract_smoke": plan["noncontract_smoke"],
        "development_only": True,
        "official_leaderboard_result": False,
    }
    row["content_sha256"] = _sample_content_sha256(row)
    return row


_WORKER_RUNTIME: _ReusableRuntime | None = None


def _worker_init(plan: dict[str, Any]) -> None:
    global _WORKER_RUNTIME
    _WORKER_RUNTIME = _ReusableRuntime(plan)
    atexit.register(_WORKER_RUNTIME.close)


def _worker_cell(job: tuple[str, int]) -> dict[str, Any]:
    if _WORKER_RUNTIME is None:
        raise ReplayRunnerError("worker runtime was not initialized")
    return {"sample": _run_cell(_WORKER_RUNTIME, *job), "worker_pid": os.getpid()}


def _cell_path(output_root: Path, world_id: str, seed: int) -> Path:
    return output_root / "cells" / world_id / f"seed-{seed:04d}.json"


def _error_path(output_root: Path, world_id: str, seed: int) -> Path:
    return output_root / "errors" / world_id / f"seed-{seed:04d}.json"


SAMPLE_FIELDS = frozenset(
    {
        "schema_version",
        "run_fingerprint",
        "run_id",
        "task_id",
        "experiment_id",
        "source_action_id",
        "action_request_index",
        "action_request_sha256",
        "world_id",
        "world",
        "preference_weight",
        "noise_levels",
        "seed",
        "status",
        "value",
        "registered_visit_share",
        "encoded_registered_visit_share",
        "missing_reason",
        "registered_visit_counts",
        "prestate_fingerprint",
        "branch_receipt",
        "primitive_action_count",
        "primitive_actions",
        "primitive_actions_sha256",
        "solver_response",
        "measure_id",
        "measure_input",
        "source_artifacts",
        "contract_seed_manifest_complete",
        "noncontract_smoke",
        "development_only",
        "official_leaderboard_result",
        "content_sha256",
    }
)
PRIMITIVE_FIELDS = frozenset({"action_id", "action", "relative_public_index"})
SOLVER_RESPONSE_FIELDS = frozenset(
    {
        "kind",
        "requested_visit_count",
        "observed_visit_count",
        "out_of_scope_visit_count",
        "elapsed_ticks",
        "timed_out",
    }
)
INVENTORY_FIELDS = frozenset(
    {
        "schema_version",
        "run_fingerprint",
        "expected_cell_count",
        "completed_cell_count",
        "cells",
        "content_sha256",
    }
)
INVENTORY_ENTRY_FIELDS = frozenset(
    {
        "world_id",
        "seed",
        "relative_path",
        "file_sha256",
        "size_bytes",
        "sample_content_sha256",
    }
)


def _profile_for_world(plan: Mapping[str, Any], world_id: str) -> Mapping[str, Any]:
    matches = [
        profile for profile in plan["candidate_profiles"] if profile.get("id") == world_id
    ]
    if len(matches) != 1:
        raise ReplayRunnerError(f"replay plan has no unique profile for {world_id}")
    return matches[0]


def _expected_branch_receipt(profile: Mapping[str, Any], seed: int) -> str:
    level_ids = {"none": 0, "weak": 1, "medium": 2, "strong": 3}
    noise = profile["noise_levels"]
    return (
        f"{profile['world']}:replay:{seed}:{float(profile['preference_weight'])}:v0.4:"
        f"{level_ids[noise['soil_nutrient_lot']]}:"
        f"{level_ids[noise['fruit_set_success']]}:"
        f"{level_ids[noise['cross_parentage_contamination']]}"
    )


def validate_sample_row(
    row: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    world_id: str,
    seed: int,
) -> dict[str, Any]:
    """Validate the complete deterministic meaning of one materialized replay cell."""

    if not isinstance(row, Mapping):
        raise ReplayRunnerError("replay cell must be a JSON object")
    _require_exact_fields(row, SAMPLE_FIELDS, "replay cell")
    if row.get("schema_version") != SAMPLE_SCHEMA:
        raise ReplayRunnerError("replay cell schema changed")
    reported_hash = _require_sha256(row.get("content_sha256"), "cell content_sha256")
    if reported_hash != _sample_content_sha256(row):
        raise ReplayRunnerError("replay cell content_sha256 does not match its payload")

    source = plan["source"]
    expected_identity = {
        "run_fingerprint": plan["run_fingerprint"],
        "run_id": source["source_run_id"],
        "task_id": source["task_id"],
        "experiment_id": source["experiment_id"],
        "source_action_id": source["source_action_id"],
        "action_request_index": source["solver_public_request_index"],
        "action_request_sha256": source["target_request_sha256"],
        "world_id": world_id,
        "seed": seed,
    }
    for field, expected in expected_identity.items():
        if row.get(field) != expected:
            raise ReplayRunnerError(f"replay cell {field} binding changed")
    _require_sha256(row.get("run_fingerprint"), "cell run_fingerprint")
    _require_sha256(row.get("action_request_sha256"), "cell action_request_sha256")

    profile = _profile_for_world(plan, world_id)
    if row.get("world") != profile["world"]:
        raise ReplayRunnerError("replay cell simulator World changed")
    preference_weight = _require_finite_number(
        row.get("preference_weight"), "cell preference_weight"
    )
    if preference_weight != float(profile["preference_weight"]):
        raise ReplayRunnerError("replay cell preference_weight changed")
    if _jsonable(row.get("noise_levels")) != _jsonable(profile["noise_levels"]):
        raise ReplayRunnerError("replay cell noise profile changed")
    if row.get("branch_receipt") != _expected_branch_receipt(profile, seed):
        raise ReplayRunnerError("replay cell branch receipt changed")

    if row.get("prestate_fingerprint") != plan["restore"]["prestate_fingerprint"]:
        raise ReplayRunnerError("replay cell prestate fingerprint changed")
    _require_sha256(row.get("prestate_fingerprint"), "cell prestate_fingerprint")
    if row.get("measure_id") != plan["measure"]["id"]:
        raise ReplayRunnerError("replay cell Measure id changed")
    if row.get("measure_input") != plan["measure"]["input"]:
        raise ReplayRunnerError("replay cell Measure input changed")
    if _jsonable(row.get("source_artifacts")) != _jsonable(plan["source_artifacts"]):
        raise ReplayRunnerError("replay cell source artifact bindings changed")
    expected_flags = {
        "contract_seed_manifest_complete": plan["contract_seed_manifest_complete"],
        "noncontract_smoke": plan["noncontract_smoke"],
        "development_only": True,
        "official_leaderboard_result": False,
    }
    for field, expected in expected_flags.items():
        if row.get(field) is not expected:
            raise ReplayRunnerError(f"replay cell {field} flag changed")

    counts = _require_mapping(row.get("registered_visit_counts"), "visit counts")
    expected_targets = list(plan["measure"]["denominator_targets"])
    if set(counts) != set(expected_targets):
        raise ReplayRunnerError("replay cell visit-count targets changed")
    normalized_counts = {
        target: _require_int(counts.get(target), f"visit count for {target}")
        for target in expected_targets
    }
    denominator = sum(normalized_counts.values())
    numerator = normalized_counts[plan["measure"]["numerator_target"]]
    expected_value = None if denominator == 0 else numerator / denominator
    expected_status = "missing" if expected_value is None else "ok"
    if row.get("status") != expected_status:
        raise ReplayRunnerError("replay cell status disagrees with registered visit counts")
    if row.get("value") != expected_value or row.get("registered_visit_share") != expected_value:
        raise ReplayRunnerError("replay cell registered visit share is inconsistent")
    if expected_value is not None:
        value = _require_finite_number(row.get("value"), "registered visit share")
        if not 0.0 <= value <= 1.0:
            raise ReplayRunnerError("registered visit share must be from zero to one")
    expected_encoded = (
        plan["measure"]["missing_encoded_value"]
        if expected_value is None
        else expected_value
    )
    if row.get("encoded_registered_visit_share") != expected_encoded:
        raise ReplayRunnerError("replay cell encoded registered visit share is inconsistent")
    expected_missing_reason = "no_registered_visits" if expected_value is None else None
    if row.get("missing_reason") != expected_missing_reason:
        raise ReplayRunnerError("replay cell missing reason is inconsistent")

    primitive_count = _require_int(
        row.get("primitive_action_count"), "primitive_action_count"
    )
    primitives = row.get("primitive_actions")
    if not isinstance(primitives, list) or len(primitives) != primitive_count:
        raise ReplayRunnerError("replay cell primitive action count is inconsistent")
    pre_action_index = int(source["pre_action_public_index"])
    for relative_index, raw_primitive in enumerate(primitives):
        primitive = _require_mapping(
            raw_primitive, f"primitive_actions[{relative_index}]"
        )
        _require_exact_fields(
            primitive, PRIMITIVE_FIELDS, f"primitive_actions[{relative_index}]"
        )
        if primitive.get("relative_public_index") != relative_index:
            raise ReplayRunnerError("replay cell primitive public indices are not contiguous")
        if primitive.get("action_id") != f"ACT-{pre_action_index + relative_index:05d}":
            raise ReplayRunnerError("replay cell primitive action ids are not contiguous")
        if not isinstance(primitive.get("action"), str) or not primitive["action"]:
            raise ReplayRunnerError("replay cell primitive action is invalid")
    primitive_hash = _require_sha256(
        row.get("primitive_actions_sha256"), "primitive_actions_sha256"
    )
    if primitive_hash != _sha256_json(primitives):
        raise ReplayRunnerError("replay cell primitive action hash is inconsistent")

    solver = _require_mapping(row.get("solver_response"), "solver_response")
    _require_exact_fields(solver, SOLVER_RESPONSE_FIELDS, "solver_response")
    if solver.get("kind") != "observe-visits":
        raise ReplayRunnerError("replay cell solver response kind changed")
    request_spec = _require_mapping(source["target_request"].get("spec"), "request spec")
    requested = _require_int(
        solver.get("requested_visit_count"), "requested_visit_count", minimum=1
    )
    observed = _require_int(solver.get("observed_visit_count"), "observed_visit_count")
    _require_int(solver.get("out_of_scope_visit_count"), "out_of_scope_visit_count")
    elapsed = _require_int(solver.get("elapsed_ticks"), "elapsed_ticks")
    timed_out = solver.get("timed_out")
    if not isinstance(timed_out, bool):
        raise ReplayRunnerError("solver_response.timed_out must be boolean")
    if requested != request_spec.get("min_visits"):
        raise ReplayRunnerError("replay cell requested visit count changed")
    if observed != denominator:
        raise ReplayRunnerError("solver observed visit count disagrees with Measure counts")
    if elapsed != primitive_count:
        raise ReplayRunnerError("solver elapsed ticks disagree with primitive expansion")
    max_ticks = _require_int(request_spec.get("max_ticks"), "request max_ticks", minimum=1)
    if elapsed > max_ticks or timed_out is not (observed < requested):
        raise ReplayRunnerError("solver timeout/count relationship is inconsistent")
    if timed_out and elapsed != max_ticks:
        raise ReplayRunnerError("timed-out solver response did not exhaust max_ticks")
    return dict(row)


def _expected_cell_layout(
    output_root: Path, plan: Mapping[str, Any]
) -> list[tuple[tuple[str, int], Path]]:
    return [
        ((str(profile["id"]), int(seed)), _cell_path(output_root, profile["id"], seed))
        for profile in plan["candidate_profiles"]
        for seed in plan["seed_manifest"]
    ]


def _inventory_content_sha256(inventory: Mapping[str, Any]) -> str:
    return _sha256_json(
        {key: value for key, value in inventory.items() if key != "content_sha256"}
    )


def _cell_inventory_entry(
    output_root: Path,
    path: Path,
    row: Mapping[str, Any],
    *,
    world_id: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "world_id": world_id,
        "seed": seed,
        "relative_path": path.relative_to(output_root).as_posix(),
        "file_sha256": _sha256_path(path),
        "size_bytes": path.stat().st_size,
        "sample_content_sha256": row["content_sha256"],
    }


def _write_cell_inventory(
    output_root: Path,
    plan: Mapping[str, Any],
    entries: Mapping[tuple[str, int], Mapping[str, Any]],
) -> None:
    ordered = [
        dict(entries[key])
        for key, _ in _expected_cell_layout(output_root, plan)
        if key in entries
    ]
    inventory = {
        "schema_version": CELL_INVENTORY_SCHEMA,
        "run_fingerprint": plan["run_fingerprint"],
        "expected_cell_count": plan["expected_sample_count"],
        "completed_cell_count": len(ordered),
        "cells": ordered,
    }
    inventory["content_sha256"] = _inventory_content_sha256(inventory)
    _write_json_atomic(output_root / CELL_INVENTORY_NAME, inventory)


def _load_cell_inventory(
    output_root: Path,
    plan: Mapping[str, Any],
    expected_paths: Mapping[tuple[str, int], Path],
) -> dict[tuple[str, int], dict[str, Any]] | None:
    path = output_root / CELL_INVENTORY_NAME
    if path.is_symlink():
        raise ReplayRunnerError("cell inventory must not be a symbolic link")
    if not path.exists():
        return None
    inventory = _read_json(path)
    _require_exact_fields(inventory, INVENTORY_FIELDS, "cell inventory")
    if inventory.get("schema_version") != CELL_INVENTORY_SCHEMA:
        raise ReplayRunnerError("cell inventory schema changed")
    if inventory.get("run_fingerprint") != plan["run_fingerprint"]:
        raise ReplayRunnerError("cell inventory run fingerprint changed")
    reported_hash = _require_sha256(
        inventory.get("content_sha256"), "cell inventory content_sha256"
    )
    if reported_hash != _inventory_content_sha256(inventory):
        raise ReplayRunnerError("cell inventory content hash does not match its payload")
    if inventory.get("expected_cell_count") != plan["expected_sample_count"]:
        raise ReplayRunnerError("cell inventory expected count changed")
    raw_entries = inventory.get("cells")
    if not isinstance(raw_entries, list):
        raise ReplayRunnerError("cell inventory cells must be a list")
    if inventory.get("completed_cell_count") != len(raw_entries):
        raise ReplayRunnerError("cell inventory completed count is inconsistent")

    entries: dict[tuple[str, int], dict[str, Any]] = {}
    for index, raw_entry in enumerate(raw_entries):
        entry = _require_mapping(raw_entry, f"cell inventory entry {index}")
        _require_exact_fields(entry, INVENTORY_ENTRY_FIELDS, f"cell inventory entry {index}")
        world_id = entry.get("world_id")
        seed = entry.get("seed")
        if not isinstance(world_id, str):
            raise ReplayRunnerError("cell inventory world_id must be a string")
        seed = _require_int(seed, "cell inventory seed")
        key = (world_id, seed)
        if key not in expected_paths or key in entries:
            raise ReplayRunnerError("cell inventory contains an unexpected or duplicate cell")
        expected_relative = expected_paths[key].relative_to(output_root).as_posix()
        if entry.get("relative_path") != expected_relative:
            raise ReplayRunnerError("cell inventory relative path changed")
        _require_sha256(entry.get("file_sha256"), "cell inventory file_sha256")
        _require_sha256(
            entry.get("sample_content_sha256"),
            "cell inventory sample_content_sha256",
        )
        _require_int(entry.get("size_bytes"), "cell inventory size_bytes", minimum=1)
        entries[key] = dict(entry)
    return entries


def _reconcile_cell_inventory(
    output_root: Path, plan: Mapping[str, Any]
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    layout = _expected_cell_layout(output_root, plan)
    expected_paths = dict(layout)
    expected_path_set = set(expected_paths.values())
    cells_root = output_root / "cells"
    if cells_root.exists():
        for candidate in cells_root.rglob("*"):
            if candidate.is_symlink():
                raise ReplayRunnerError(f"symbolic link is forbidden in replay cells: {candidate}")
            if candidate.is_file() and candidate not in expected_path_set:
                raise ReplayRunnerError(f"unexpected replay cell artifact: {candidate}")

    loaded_entries = _load_cell_inventory(output_root, plan, expected_paths)
    entries = {} if loaded_entries is None else dict(loaded_entries)
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    repaired = loaded_entries is None
    for key, path in layout:
        world_id, seed = key
        if path.is_symlink():
            raise ReplayRunnerError(f"replay cell must not be a symbolic link: {path}")
        if not path.exists():
            if key in entries:
                raise ReplayRunnerError(f"inventoried replay cell is missing: {path}")
            continue
        if not path.is_file():
            raise ReplayRunnerError(f"replay cell is not a regular file: {path}")
        try:
            row = validate_sample_row(
                _read_json(path), plan, world_id=world_id, seed=seed
            )
        except ReplayRunnerError as error:
            raise ReplayRunnerError(
                f"existing replay cell is invalid and will not be overwritten: {path}: {error}"
            ) from error
        error_path = _error_path(output_root, world_id, seed)
        if error_path.exists() or error_path.is_symlink():
            raise ReplayRunnerError(
                f"replay cell conflicts with a retained error artifact: {error_path}"
            )
        actual_entry = _cell_inventory_entry(
            output_root, path, row, world_id=world_id, seed=seed
        )
        if key in entries and entries[key] != actual_entry:
            raise ReplayRunnerError(f"replay cell differs from its inventory entry: {path}")
        if key not in entries:
            entries[key] = actual_entry
            repaired = True
        rows[key] = row
    if repaired:
        _write_cell_inventory(output_root, plan, entries)
    return entries, rows


def _collect_cells(
    output_root: Path, plan: Mapping[str, Any], *, require_complete: bool
) -> list[dict[str, Any]]:
    _, by_key = _reconcile_cell_inventory(output_root, plan)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for key, _ in _expected_cell_layout(output_root, plan):
        if key not in by_key:
            missing.append(f"{key[0]}/{key[1]}")
        else:
            rows.append(by_key[key])
    if require_complete and missing:
        preview = ", ".join(missing[:10])
        raise ReplayRunnerError(f"replay grid is incomplete ({len(missing)} missing): {preview}")
    _write_jsonl_atomic(output_root / SAMPLES_NAME, rows)
    return rows


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _validate_runtime_attempt(
    value: Mapping[str, Any], *, attempt_number: int, expected_cell_count: int
) -> dict[str, Any]:
    """Validate one immutable runtime-attempt record before retaining it."""

    _require_exact_fields(value, RUNTIME_ATTEMPT_FIELDS, "runtime telemetry attempt")
    if value.get("schema_version") != RUNTIME_ATTEMPT_SCHEMA:
        raise ReplayRunnerError("runtime telemetry attempt schema changed")
    if value.get("attempt_number") != attempt_number:
        raise ReplayRunnerError("runtime telemetry attempt numbers are not contiguous")
    if not isinstance(value.get("legacy_migrated"), bool):
        raise ReplayRunnerError("runtime telemetry legacy_migrated must be boolean")
    status = value.get("status")
    if status not in {"running", "complete", "failed"}:
        raise ReplayRunnerError("runtime telemetry attempt status is invalid")
    if not isinstance(value.get("started_at_utc"), str) or not value["started_at_utc"]:
        raise ReplayRunnerError("runtime telemetry attempt start time is invalid")
    finished = value.get("finished_at_utc")
    if finished is not None and (not isinstance(finished, str) or not finished):
        raise ReplayRunnerError("runtime telemetry attempt finish time is invalid")
    elapsed = value.get("elapsed_seconds")
    if elapsed is not None and (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, int | float)
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0
    ):
        raise ReplayRunnerError("runtime telemetry attempt elapsed time is invalid")
    if status == "running" and finished is not None:
        raise ReplayRunnerError("running telemetry attempt cannot have a finish time")
    if status != "running" and (finished is None or elapsed is None):
        raise ReplayRunnerError("finished telemetry attempt lacks finish time or elapsed time")

    requested_workers = _require_int(
        value.get("requested_worker_count"), "runtime requested_worker_count", minimum=1
    )
    pool_workers = _require_int(
        value.get("pool_worker_limit"), "runtime pool_worker_limit"
    )
    actual_workers = _require_int(
        value.get("actual_worker_count"), "runtime actual_worker_count"
    )
    scheduled = _require_int(
        value.get("scheduled_cell_count"), "runtime scheduled_cell_count"
    )
    resumed = _require_int(value.get("resumed_cell_count"), "runtime resumed_cell_count")
    successful = _require_int(
        value.get("successful_executed_cell_count"),
        "runtime successful_executed_cell_count",
    )
    failed = _require_int(value.get("failed_cell_count"), "runtime failed_cell_count")
    if pool_workers > requested_workers:
        raise ReplayRunnerError("runtime pool worker limit exceeds requested workers")
    if scheduled + resumed != expected_cell_count:
        raise ReplayRunnerError("runtime scheduled/resumed cells do not cover the replay grid")
    if successful > scheduled or failed > scheduled:
        raise ReplayRunnerError("runtime successful/failed counts exceed scheduled cells")

    raw_assignments = value.get("worker_assignments")
    if not isinstance(raw_assignments, list):
        raise ReplayRunnerError("runtime worker_assignments must be a list")
    assignments: list[dict[str, int]] = []
    seen_pids: set[int] = set()
    for index, raw_assignment in enumerate(raw_assignments):
        assignment = _require_mapping(
            raw_assignment, f"runtime worker_assignments[{index}]"
        )
        _require_exact_fields(
            assignment,
            frozenset({"worker_pid", "cell_count"}),
            f"runtime worker_assignments[{index}]",
        )
        pid = _require_int(
            assignment.get("worker_pid"),
            f"runtime worker_assignments[{index}].worker_pid",
            minimum=1,
        )
        count = _require_int(
            assignment.get("cell_count"),
            f"runtime worker_assignments[{index}].cell_count",
            minimum=1,
        )
        if pid in seen_pids:
            raise ReplayRunnerError("runtime worker assignments contain a duplicate PID")
        seen_pids.add(pid)
        assignments.append({"worker_pid": pid, "cell_count": count})
    if actual_workers != len(assignments):
        raise ReplayRunnerError("runtime actual worker count disagrees with assignments")

    error_type = value.get("error_type")
    error = value.get("error")
    if (error_type is None) != (error is None):
        raise ReplayRunnerError("runtime error type and message must appear together")
    if error_type is not None and (
        not isinstance(error_type, str) or not isinstance(error, str)
    ):
        raise ReplayRunnerError("runtime error type and message must be strings")
    if status == "failed" and error_type is None:
        raise ReplayRunnerError("failed runtime attempt lacks error evidence")
    if status != "failed" and error_type is not None:
        raise ReplayRunnerError("non-failed runtime attempt contains error evidence")
    return dict(value)


def _runtime_telemetry_document(
    plan: Mapping[str, Any], attempts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Build the compatibility view plus append-preserved cumulative evidence."""

    if not attempts:
        raise ReplayRunnerError("runtime telemetry must contain at least one attempt")
    validated = [
        _validate_runtime_attempt(
            attempt,
            attempt_number=index,
            expected_cell_count=int(plan["expected_sample_count"]),
        )
        for index, attempt in enumerate(attempts, 1)
    ]
    latest = validated[-1]
    cumulative_assignments: dict[int, int] = {}
    for attempt in validated:
        for assignment in attempt["worker_assignments"]:
            pid = int(assignment["worker_pid"])
            cumulative_assignments[pid] = (
                cumulative_assignments.get(pid, 0) + int(assignment["cell_count"])
            )
    finished_times = [
        str(attempt["finished_at_utc"])
        for attempt in validated
        if attempt["finished_at_utc"] is not None
    ]
    compatibility_fields = LEGACY_RUNTIME_FIELDS - {"schema_version", "run_fingerprint"}
    document = {
        "schema_version": RUNTIME_TELEMETRY_SCHEMA,
        "run_fingerprint": plan["run_fingerprint"],
        **{field: latest[field] for field in compatibility_fields},
        "attempt_count": len(validated),
        "execution_attempt_count": sum(
            int(attempt["scheduled_cell_count"] > 0) for attempt in validated
        ),
        "no_op_attempt_count": sum(
            int(attempt["scheduled_cell_count"] == 0) for attempt in validated
        ),
        "complete_attempt_count": sum(
            int(attempt["status"] == "complete") for attempt in validated
        ),
        "failed_attempt_count": sum(
            int(attempt["status"] == "failed") for attempt in validated
        ),
        "running_attempt_count": sum(
            int(attempt["status"] == "running") for attempt in validated
        ),
        "cumulative_elapsed_seconds": sum(
            float(attempt["elapsed_seconds"])
            for attempt in validated
            if attempt["elapsed_seconds"] is not None
        ),
        "cumulative_scheduled_cell_count": sum(
            int(attempt["scheduled_cell_count"]) for attempt in validated
        ),
        "cumulative_successful_executed_cell_count": sum(
            int(attempt["successful_executed_cell_count"])
            for attempt in validated
        ),
        "cumulative_failed_cell_count": sum(
            int(attempt["failed_cell_count"]) for attempt in validated
        ),
        "cumulative_actual_worker_count": len(cumulative_assignments),
        "cumulative_worker_assignments": [
            {"worker_pid": pid, "cell_count": count}
            for pid, count in sorted(cumulative_assignments.items())
        ],
        "first_started_at_utc": validated[0]["started_at_utc"],
        "last_finished_at_utc": finished_times[-1] if finished_times else None,
        "attempts": validated,
    }
    return document


def _load_runtime_attempts(
    output_root: Path,
    plan: Mapping[str, Any],
    *,
    recover_interrupted: bool,
) -> list[dict[str, Any]]:
    """Load append-preserved attempts, migrating the pre-history telemetry shape once."""

    path = output_root / RUNTIME_TELEMETRY_NAME
    if path.is_symlink():
        raise ReplayRunnerError("runtime telemetry must not be a symbolic link")
    if not path.exists():
        return []
    value = _read_json(path)
    if value.get("schema_version") != RUNTIME_TELEMETRY_SCHEMA:
        raise ReplayRunnerError("runtime telemetry schema changed")
    if value.get("run_fingerprint") != plan["run_fingerprint"]:
        raise ReplayRunnerError("runtime telemetry run fingerprint changed")
    if "attempts" not in value:
        _require_exact_fields(value, LEGACY_RUNTIME_FIELDS, "legacy runtime telemetry")
        attempt = {
            "schema_version": RUNTIME_ATTEMPT_SCHEMA,
            "attempt_number": 1,
            "legacy_migrated": True,
            **{
                field: value[field]
                for field in LEGACY_RUNTIME_FIELDS
                if field not in {"schema_version", "run_fingerprint"}
            },
        }
        attempts = [
            _validate_runtime_attempt(
                attempt,
                attempt_number=1,
                expected_cell_count=int(plan["expected_sample_count"]),
            )
        ]
    else:
        _require_exact_fields(value, RUNTIME_TELEMETRY_FIELDS, "runtime telemetry")
        raw_attempts = value.get("attempts")
        if not isinstance(raw_attempts, list) or not raw_attempts:
            raise ReplayRunnerError("runtime telemetry attempts must be a non-empty list")
        attempts = [
            _validate_runtime_attempt(
                _require_mapping(raw_attempt, f"runtime telemetry attempt {index}"),
                attempt_number=index,
                expected_cell_count=int(plan["expected_sample_count"]),
            )
            for index, raw_attempt in enumerate(raw_attempts, 1)
        ]
        expected = _runtime_telemetry_document(plan, attempts)
        if value != expected:
            raise ReplayRunnerError(
                "runtime telemetry cumulative or compatibility view changed"
            )

    running_indices = [
        index for index, attempt in enumerate(attempts) if attempt["status"] == "running"
    ]
    if running_indices:
        if running_indices != [len(attempts) - 1]:
            raise ReplayRunnerError(
                "runtime telemetry contains a non-latest running attempt"
            )
        if not recover_interrupted:
            raise ReplayRunnerError(
                "the previous replay attempt lacks a terminal checkpoint; use --resume"
            )
        interrupted = dict(attempts[-1])
        interrupted.update(
            {
                "status": "failed",
                "finished_at_utc": _utc_now(),
                "elapsed_seconds": (
                    0.0
                    if interrupted["elapsed_seconds"] is None
                    else interrupted["elapsed_seconds"]
                ),
                "error_type": INTERRUPTED_ATTEMPT_ERROR_TYPE,
                "error": INTERRUPTED_ATTEMPT_ERROR,
            }
        )
        attempts[-1] = _validate_runtime_attempt(
            interrupted,
            attempt_number=len(attempts),
            expected_cell_count=int(plan["expected_sample_count"]),
        )
        _write_json_atomic(path, _runtime_telemetry_document(plan, attempts))
    return attempts


def run_replay_grid(
    output_root: Path,
    plan: Mapping[str, Any],
    *,
    workers: int,
    resume: bool,
) -> list[dict[str, Any]]:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ReplayRunnerError("--workers must be a positive integer")
    if not isinstance(resume, bool):
        raise ReplayRunnerError("resume must be boolean")

    entries, existing_rows = _reconcile_cell_inventory(output_root, plan)
    if existing_rows and not resume:
        first_key = next(iter(existing_rows))
        raise ReplayRunnerError(
            f"cell already exists; use --resume: "
            f"{_cell_path(output_root, first_key[0], first_key[1])}"
        )
    jobs = [
        key
        for key, _ in _expected_cell_layout(output_root, plan)
        if key not in existing_rows
    ]
    worker_count = min(workers, len(jobs)) if jobs else 0
    completed = len(existing_rows)
    successful_executions = 0
    failures: list[tuple[str, int, BaseException]] = []
    assignments: dict[int, int] = {}
    started_monotonic = time.monotonic()
    previous_attempts = _load_runtime_attempts(
        output_root, plan, recover_interrupted=resume
    )
    attempt_number = len(previous_attempts) + 1
    started_at_utc = _utc_now()

    def record_telemetry(status: str, error: BaseException | None = None) -> None:
        finished = status != "running"
        attempt = {
            "schema_version": RUNTIME_ATTEMPT_SCHEMA,
            "attempt_number": attempt_number,
            "legacy_migrated": False,
            "status": status,
            "started_at_utc": started_at_utc,
            "finished_at_utc": _utc_now() if finished else None,
            "elapsed_seconds": time.monotonic() - started_monotonic,
            "requested_worker_count": workers,
            "pool_worker_limit": worker_count,
            "actual_worker_count": len(assignments),
            "scheduled_cell_count": len(jobs),
            "resumed_cell_count": len(existing_rows),
            "successful_executed_cell_count": successful_executions,
            "failed_cell_count": len(failures),
            "worker_assignments": [
                {"worker_pid": pid, "cell_count": count}
                for pid, count in sorted(assignments.items())
            ],
            "error_type": type(error).__name__ if error is not None else None,
            "error": str(error) if error is not None else None,
        }
        document = _runtime_telemetry_document(plan, [*previous_attempts, attempt])
        _write_json_atomic(output_root / RUNTIME_TELEMETRY_NAME, document)

    record_telemetry("running")

    try:
        if jobs:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=context,
                initializer=_worker_init,
                initargs=(dict(plan),),
            ) as executor:
                futures = {executor.submit(_worker_cell, job): job for job in jobs}
                for future in as_completed(futures):
                    world_id, seed = futures[future]
                    cell_path = _cell_path(output_root, world_id, seed)
                    try:
                        result = future.result()
                        if not isinstance(result, Mapping) or set(result) != {
                            "sample",
                            "worker_pid",
                        }:
                            raise ReplayRunnerError("worker returned a malformed result envelope")
                        worker_pid = _require_int(
                            result.get("worker_pid"), "worker_pid", minimum=1
                        )
                        assignments[worker_pid] = assignments.get(worker_pid, 0) + 1
                        sample = _require_mapping(result.get("sample"), "worker sample")
                        row = validate_sample_row(
                            sample, plan, world_id=world_id, seed=seed
                        )
                        _write_json_exclusive(cell_path, row)
                        entries[(world_id, seed)] = _cell_inventory_entry(
                            output_root,
                            cell_path,
                            row,
                            world_id=world_id,
                            seed=seed,
                        )
                        _write_cell_inventory(output_root, plan, entries)
                        _error_path(output_root, world_id, seed).unlink(missing_ok=True)
                    except BaseException as error:  # retain every failed cell diagnostic
                        failures.append((world_id, seed, error))
                        if not cell_path.exists():
                            _write_json_atomic(
                                _error_path(output_root, world_id, seed),
                                {
                                    "schema_version": ERROR_SCHEMA,
                                    "run_fingerprint": plan["run_fingerprint"],
                                    "world_id": world_id,
                                    "seed": seed,
                                    "error_type": type(error).__name__,
                                    "error": str(error),
                                    "traceback": "".join(
                                        traceback.format_exception(
                                            type(error), error, error.__traceback__
                                        )
                                    ),
                                },
                            )
                        record_telemetry("running")
                        continue
                    successful_executions += 1
                    completed += 1
                    record_telemetry("running")
                    if completed == plan["expected_sample_count"] or completed % 10 == 0:
                        print(
                            f"completed {completed}/{plan['expected_sample_count']} replay cells",
                            flush=True,
                        )

        rows = _collect_cells(output_root, plan, require_complete=False)
        if failures:
            preview = ", ".join(
                f"{world}/{seed}: {error}" for world, seed, error in failures[:5]
            )
            raise ReplayRunnerError(
                f"{len(failures)} replay cells failed; rerun with --resume. {preview}"
            )
        if len(rows) != plan["expected_sample_count"]:
            raise ReplayRunnerError("replay grid ended without all expected samples")
    except BaseException as error:
        record_telemetry("failed", error)
        raise
    record_telemetry("complete")
    return rows


def _profile_for_selected_seeds(plan: Mapping[str, Any]) -> ExperimentReplayProfile:
    contract = _read_json(Path(plan["contract"]["path"]))
    try:
        profile = validate_contract(contract, repository_root=REPOSITORY_ROOT)
    except ExperimentReplayValidationError as error:
        raise ReplayRunnerError(str(error)) from error
    return replace(profile, seed_manifest=tuple(plan["seed_manifest"]))


def summarize_replay_grid(output_root: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    rows = _collect_cells(output_root, plan, require_complete=True)
    profile = _profile_for_selected_seeds(plan)
    by_world: dict[str, list[dict[str, Any]]] = {world_id: [] for world_id in CANDIDATE_WORLD_IDS}
    for row in rows:
        by_world[str(row["world_id"])].append(row)
    for values in by_world.values():
        values.sort(key=lambda item: int(item["seed"]))

    source = plan["source"]
    true_world_id = str(source["true_world_id"])
    pair_rows: list[dict[str, Any]] = []
    pair_aggregates: list[dict[str, Any]] = []
    for alternative_world_id in CANDIDATE_WORLD_IDS:
        if alternative_world_id == true_world_id:
            continue
        pair = score_experiment_pair(
            run_id=str(source["source_run_id"]),
            task_id=str(source["task_id"]),
            experiment_id=str(source["experiment_id"]),
            true_world_id=true_world_id,
            alternative_world_id=alternative_world_id,
            true_samples=by_world[true_world_id],
            alternative_samples=by_world[alternative_world_id],
            profile=profile,
        )
        pair_rows.append(pair)
        pair_aggregates.append(
            aggregate_pair(
                run_id=str(source["source_run_id"]),
                task_id=str(source["task_id"]),
                true_world_id=true_world_id,
                alternative_world_id=alternative_world_id,
                experiment_results=[pair],
                profile=profile,
            )
        )
    task = aggregate_task(
        run_id=str(source["source_run_id"]),
        task_id=str(source["task_id"]),
        true_world_id=true_world_id,
        pair_results=pair_aggregates,
        profile=profile,
    )
    n = len(plan["seed_manifest"])
    contract_seed_manifest_complete = n == 100
    if (
        plan.get("contract_seed_manifest_complete") is not contract_seed_manifest_complete
        or plan.get("noncontract_smoke") is contract_seed_manifest_complete
    ):
        raise ReplayRunnerError("replay plan seed-manifest classification is inconsistent")
    critical = profile.critical_constant * math.sqrt(2 / n)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "status": (
            "complete_development_pilot"
            if contract_seed_manifest_complete
            else "complete_noncontract_smoke"
        ),
        "run_fingerprint": plan["run_fingerprint"],
        "run_kind": plan["run_kind"],
        "contract_seed_manifest_complete": contract_seed_manifest_complete,
        "noncontract_smoke": not contract_seed_manifest_complete,
        "source_run_id": source["source_run_id"],
        "task_id": source["task_id"],
        "experiment_id": source["experiment_id"],
        "source_action_id": source["source_action_id"],
        "action_request_index": source["solver_public_request_index"],
        "true_world_id": true_world_id,
        "candidate_world_ids": list(CANDIDATE_WORLD_IDS),
        "alternative_count": 7,
        "sample_count": len(rows),
        "n_per_world": n,
        "seed_manifest": list(plan["seed_manifest"]),
        "measure": plan["measure"],
        "comparison": {
            "statistic": "two_sample_ks",
            "critical_constant": profile.critical_constant,
            "critical_value_for_equal_n": critical,
            "operator": "strict_gt",
            "missing_policy": profile.missing_policy,
            "multiplicity_policy": profile.multiplicity_policy,
        },
        "encoded_missing_value_count": sum(row["status"] == "missing" for row in rows),
        "pairs": pair_rows,
        "pair_aggregates": pair_aggregates,
        "task_aggregate": task,
        "scoring_model_calls": 0,
        "llm_judge_used_for_score": False,
        "development_only": True,
        "official_leaderboard_result": False,
        "formal_experiment_score": None,
        "formal_four_model_experiment_score": None,
    }
    _write_json_atomic(output_root / SUMMARY_NAME, summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "run", "summarize", "full"), nargs="?", default="full"
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--workers", type=int, default=max(1, min(8, os.cpu_count() or 1))
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=100,
        help="Use the first N entries of the frozen seed manifest (1-100).",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root = args.output_root.resolve()
    if args.command == "summarize":
        plan = load_validated_plan(output_root)
        summary = summarize_replay_grid(output_root, plan)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0

    plan = prepare_plan(
        args.contract,
        output_root,
        seed_count=args.seeds,
        resume=args.resume,
    )
    if args.command == "prepare":
        print(json.dumps(plan, ensure_ascii=False, sort_keys=True))
        return 0
    run_replay_grid(output_root, plan, workers=args.workers, resume=args.resume)
    if args.command == "run":
        return 0
    summary = summarize_replay_grid(output_root, plan)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReplayRunnerError, ExperimentReplayValidationError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
