#!/usr/bin/env python3
"""Run one frozen shard of the 0.4.1 Gate E development confirmation."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import run_aer_pea_calibration as frozen
import run_aer_pea_gate_e_study as episode_runner

SCRIPT_PATH = Path(__file__).resolve()
EXPECTED_WORLDS = (
    "white_preference",
    "position_attraction",
    "plant_attractiveness",
    "fertility_difference",
    "transient_null",
    "clean",
)
EXPECTED_CONDITION = "l4_explicit_elimination"
SPLIT_PATH = (
    frozen.AER_PEA_CASE_ROOT / "construction" / "split.v0.3.0.json"
)
EXPECTED_CELLS = (
    {"repetition": 1, "case_root": 211, "variation": 5},
    {"repetition": 2, "case_root": 401, "variation": 23},
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_confirmation_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "aer.pea.gate-e-confirmation-study.v1":
        raise ValueError("unsupported Gate E confirmation config schema")
    if config.get("status") != "frozen_for_development_confirmation":
        raise ValueError("Gate E confirmation config is not frozen")
    if config.get("held_out_execution_allowed") is not False:
        raise ValueError("Gate E confirmation must forbid held-out execution")
    if config.get("historical_results_may_be_rewritten") is not False:
        raise ValueError("Gate E confirmation must preserve historical results")
    if config.get("selected_condition") != EXPECTED_CONDITION:
        raise ValueError("Gate E confirmation condition changed")
    if tuple(config.get("conditions", ())) != (EXPECTED_CONDITION,):
        raise ValueError("Gate E confirmation must contain only frozen L4")

    confirmation = config.get("confirmation")
    if not isinstance(confirmation, dict):
        raise ValueError("Gate E confirmation schedule is missing")
    if tuple(confirmation.get("worlds", ())) != EXPECTED_WORLDS:
        raise ValueError("Gate E confirmation worlds changed")
    cells = confirmation.get("replication_cells")
    if tuple(cells or ()) != EXPECTED_CELLS:
        raise ValueError("Gate E confirmation replication cells changed")
    expected_count = len(EXPECTED_WORLDS) * len(cells)
    if confirmation.get("registered_episode_count") != expected_count:
        raise ValueError("Gate E confirmation episode count changed")
    if config.get("registered_confirmation_episode_count") != expected_count:
        raise ValueError("top-level confirmation episode count changed")
    if confirmation.get("review_begins_only_after_all_registered_episodes_finish") is not True:
        raise ValueError("Gate E review must wait for all confirmation episodes")
    if config.get("previous_valid_live_episode_count", 0) + expected_count != config.get(
        "cumulative_valid_live_episode_count_if_complete"
    ):
        raise ValueError("Gate E cumulative live count is inconsistent")
    if config["cumulative_valid_live_episode_count_if_complete"] > config.get(
        "authorized_live_episode_cap", 0
    ):
        raise ValueError("Gate E confirmation exceeds the live authorization")

    shards = confirmation.get("shards")
    if not isinstance(shards, dict) or not shards:
        raise ValueError("Gate E confirmation shards are missing")
    registered_keys = {
        (world, cell["repetition"])
        for world in EXPECTED_WORLDS
        for cell in EXPECTED_CELLS
    }
    flattened = [
        (entry.get("world"), entry.get("repetition"))
        for entries in shards.values()
        for entry in entries
    ]
    if len(flattened) != len(set(flattened)) or set(flattened) != registered_keys:
        raise ValueError("Gate E confirmation shards must partition all 12 cells")
    if any(len(entries) != 4 for entries in shards.values()):
        raise ValueError("each Gate E confirmation shard must contain four cells")

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    if split.get("spec_version") != "0.3.0" or split.get("status") != "frozen":
        raise ValueError("Gate E confirmation requires frozen split 0.3.0")
    for cell in EXPECTED_CELLS:
        if (
            cell["case_root"] not in split["development"]["roots"]
            or cell["variation"] not in split["development"]["variations"]
        ):
            raise ValueError("Gate E confirmation cell is not development-only")
        if (
            cell["case_root"] in split["held_out"]["roots"]
            or cell["variation"] in split["held_out"]["variations"]
        ):
            raise ValueError("Gate E confirmation cell touches held-out")

    source = config["selected_prompt_source"]
    source_path = (frozen.AER_BENCH_ROOT / source["path"]).resolve()
    if not source_path.is_file() or _sha256(source_path) != source["sha256"]:
        raise ValueError("selected pilot prompt source hash mismatch")
    pilot = json.loads(source_path.read_text(encoding="utf-8"))
    if config["common_interface_instruction"] != pilot["common_interface_instruction"]:
        raise ValueError("confirmation common interface differs from frozen pilot")
    if config["conditions"][EXPECTED_CONDITION] != pilot["conditions"][
        EXPECTED_CONDITION
    ]:
        raise ValueError("confirmation L4 prompt differs from frozen pilot")

    basis = config["selection_basis"]
    result_path = (frozen.AER_BENCH_ROOT / basis["pilot_result_path"]).resolve()
    if not result_path.is_file() or _sha256(result_path) != basis["pilot_result_sha256"]:
        raise ValueError("pilot selection result hash mismatch")
    return config


def _source_paths(
    config: dict[str, Any],
    config_path: Path,
    rubric_path: Path,
) -> dict[str, Path]:
    case_root = frozen.AER_PEA_CASE_ROOT
    return {
        "confirmation_config": config_path,
        "review_rubric": rubric_path,
        "confirmation_runner": SCRIPT_PATH,
        "episode_runner": Path(episode_runner.__file__).resolve(),
        "frozen_v0_2_0_runner": Path(frozen.__file__).resolve(),
        "selected_pilot_prompt_source": frozen.AER_BENCH_ROOT
        / config["selected_prompt_source"]["path"],
        "pilot_selection_result": frozen.AER_BENCH_ROOT
        / config["selection_basis"]["pilot_result_path"],
        "public_lab_client": case_root / "public/lab.py",
        "public_submission_schema": case_root / "public/submission.schema.json",
        "hidden_evidence_builder": case_root / "hidden/evidence.py",
        "hidden_deterministic_grader": case_root / "hidden/grader.py",
        "development_split": SPLIT_PATH,
        "scienceworld_jar": frozen.SCIENCEWORLD_ROOT
        / "scienceworld/scienceworld.jar",
    }


def _central_freeze_manifest(
    config: dict[str, Any], config_path: Path, rubric_path: Path
) -> dict[str, Any]:
    source_paths = _source_paths(config, config_path, rubric_path)
    missing = [name for name, path in source_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing confirmation freeze inputs: {missing}")
    return {
        "schema_version": "aer.pea.gate-e-confirmation-freeze.v1",
        "status": "frozen_for_development_confirmation_execution",
        "study_version": config["study_version"],
        "registered_episode_count": config["registered_confirmation_episode_count"],
        "cumulative_valid_live_episode_count_if_complete": config[
            "cumulative_valid_live_episode_count_if_complete"
        ],
        "selected_condition": config["selected_condition"],
        "held_out_execution_allowed": False,
        "model": config["model"],
        "reasoning_effort": config["reasoning_effort"],
        "formal_episode_action_budget": config["formal_episode_action_budget"],
        "replication_cells": config["confirmation"]["replication_cells"],
        "worlds": config["confirmation"]["worlds"],
        "shards": config["confirmation"]["shards"],
        "source_sha256": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path.resolve())}
            for name, path in source_paths.items()
        },
    }


def _write_once_or_validate(path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        for _ in range(100):
            try:
                if path.read_bytes() == encoded:
                    return
            except OSError:
                pass
            time.sleep(0.05)
        raise RuntimeError(
            f"existing central freeze differs or is incomplete: {path}"
        ) from None
    try:
        offset = 0
        while offset < len(encoded):
            offset += os.write(descriptor, encoded[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _shard_freeze_manifest(
    config: dict[str, Any], shard: str, central_manifest_path: Path
) -> dict[str, Any]:
    return {
        "schema_version": "aer.pea.gate-e-confirmation-shard-freeze.v1",
        "status": "frozen_for_development_confirmation_execution",
        "study_version": config["study_version"],
        "shard": shard,
        "registered_cells": config["confirmation"]["shards"][shard],
        "shard_registered_episode_count": len(
            config["confirmation"]["shards"][shard]
        ),
        "central_freeze_manifest": {
            "path": str(central_manifest_path),
            "sha256": _sha256(central_manifest_path),
        },
        "held_out_execution_allowed": False,
    }


def _run_id(world: str, cell: dict[str, int]) -> str:
    return (
        f"{world}-variation-{cell['variation']:02d}-"
        f"root-{cell['case_root']:04d}-run-{cell['repetition']:02d}"
    )


def _verify_finalized(
    state: dict[str, Any], output_root: Path, condition: str
) -> dict[str, Any]:
    if state.get("phase") != "finalized":
        raise RuntimeError("only finalized confirmation episodes may be resumed")
    outcome = state.get("outcome")
    if not isinstance(outcome, dict):
        raise RuntimeError("finalized confirmation state is missing its outcome")
    artifact_dir = output_root / condition / state["run_id"]
    metadata_path = artifact_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for relative, expected in metadata["files_sha256"].items():
        if expected is None:
            continue
        path = artifact_dir / relative
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"finalized confirmation artifact hash mismatch: {path}")
    if _sha256(metadata_path) != state.get("metadata_sha256"):
        raise RuntimeError("finalized confirmation metadata hash mismatch")
    return outcome


class _JournaledRunner:
    def __init__(
        self,
        delegate: frozen.CodexRunner,
        state_path: Path,
        state: dict[str, Any],
    ) -> None:
        self.delegate = delegate
        self.state_path = state_path
        self.state = state

    def run(self, config: frozen.CodexRunConfig) -> Any:
        attempt = self.state["attempts"][-1]
        attempt["solver_started"] = True
        attempt["solver_started_at_unix"] = time.time()
        self.state["phase"] = "solver_started"
        frozen._safe_write_json(self.state_path, self.state)
        try:
            result = self.delegate.run(config)
        except Exception as error:
            attempt["error_type"] = type(error).__name__
            attempt["error"] = str(error)
            self.state["phase"] = "solver_failed"
            frozen._safe_write_json(self.state_path, self.state)
            raise
        attempt["solver_finished_at_unix"] = time.time()
        attempt["thread_id"] = result.thread_id
        attempt["usage"] = result.usage
        self.state["phase"] = "solver_finished"
        frozen._safe_write_json(self.state_path, self.state)
        return result


def _is_allowed_pre_solver_failure(error: Exception) -> bool:
    return isinstance(error, (FileNotFoundError, ConnectionError, OSError)) or (
        isinstance(error, ValueError)
        and "invalid literal for int()" in str(error)
    )


def _execute_registered_episode(
    delegate: frozen.CodexRunner,
    output_root: Path,
    config: dict[str, Any],
    config_path: Path,
    rubric_path: Path,
    central_manifest_path: Path,
    shard: str,
    world: str,
    cell: dict[str, int],
    timeout_seconds: int,
    *,
    resume: bool,
) -> dict[str, Any]:
    condition = config["selected_condition"]
    run_id = _run_id(world, cell)
    state_path = output_root / "_episode_states" / f"{run_id}.json"
    artifact_dir = output_root / condition / run_id
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if resume and state.get("phase") == "finalized":
            return _verify_finalized(state, output_root, condition)
        if not resume:
            raise RuntimeError(f"confirmation state already exists: {state_path}")
        if state.get("phase") not in {"registered", "pre_solver_failed"}:
            raise RuntimeError(
                f"refusing to retry post-solver confirmation state: {state['phase']}"
            )
    else:
        state = {
            "schema_version": "aer.pea.gate-e-confirmation-episode-state.v1",
            "run_id": run_id,
            "shard": shard,
            "world": world,
            "cell": cell,
            "phase": "registered",
            "attempts": [],
        }
        frozen._safe_write_json(state_path, state)

    retry_limit = config["stopping_rules"][
        "pre_solver_machine_verified_infrastructure_retry_limit"
    ]
    while True:
        attempt_number = len(state["attempts"]) + 1
        if attempt_number > retry_limit + 1:
            raise RuntimeError(f"pre-solver retry limit exhausted for {run_id}")
        attempt = {
            "attempt": attempt_number,
            "solver_started": False,
            "preflight_passed_at_unix": time.time(),
        }
        state["attempts"].append(attempt)
        state["phase"] = "preflight_passed"
        frozen._safe_write_json(state_path, state)
        runner = _JournaledRunner(delegate, state_path, state)
        try:
            outcome = episode_runner.run_episode(
                runner,
                output_root,
                config,
                config_path,
                rubric_path,
                condition,
                world,
                cell,
                timeout_seconds=timeout_seconds,
            )
        except Exception as error:
            attempt["error_type"] = type(error).__name__
            attempt["error"] = str(error)
            if attempt["solver_started"] or not _is_allowed_pre_solver_failure(error):
                state["phase"] = (
                    "post_solver_failed"
                    if attempt["solver_started"]
                    else "pre_solver_nonretryable_failed"
                )
                frozen._safe_write_json(state_path, state)
                raise
            state["phase"] = "pre_solver_failed"
            attempt["retry_allowed"] = attempt_number <= retry_limit
            if artifact_dir.exists():
                preserved = (
                    output_root
                    / "failed_attempts"
                    / run_id
                    / f"attempt-{attempt_number:02d}"
                )
                preserved.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(artifact_dir), str(preserved))
                attempt["preserved_artifact_dir"] = str(preserved)
            frozen._safe_write_json(state_path, state)
            if attempt_number > retry_limit:
                raise
            continue

        metadata_path = Path(outcome["artifact_dir"]) / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["confirmation_runner_sha256"] = _sha256(SCRIPT_PATH)
        metadata["confirmation_config_sha256"] = _sha256(config_path)
        metadata["confirmation_central_freeze_sha256"] = _sha256(
            central_manifest_path
        )
        metadata["confirmation_split_v0_3_0_sha256"] = _sha256(SPLIT_PATH)
        metadata["confirmation_shard"] = shard
        metadata["confirmation_attempt"] = attempt_number
        frozen._safe_write_json(metadata_path, metadata)
        outcome["metadata_sha256"] = _sha256(metadata_path)
        attempt["finalized_at_unix"] = time.time()
        state["phase"] = "finalized"
        state["outcome"] = outcome
        state["metadata_sha256"] = outcome["metadata_sha256"]
        frozen._safe_write_json(state_path, state)
        return outcome


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-parent", required=True, type=Path)
    parser.add_argument("--study-config", required=True, type=Path)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config_path = args.study_config.resolve()
    config = _load_confirmation_config(config_path)
    shards = config["confirmation"]["shards"]
    if args.shard not in shards:
        parser.error(f"unknown confirmation shard: {args.shard}")
    rubric_path = (frozen.AER_BENCH_ROOT / config["review"]["rubric_path"]).resolve()
    if not rubric_path.is_file():
        parser.error(f"missing review rubric: {rubric_path}")

    output_parent = args.output_parent.resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    central_manifest_path = output_parent / "confirmation_freeze_manifest.json"
    _write_once_or_validate(
        central_manifest_path,
        _central_freeze_manifest(config, config_path, rubric_path),
    )
    output_root = output_parent / "shards" / args.shard
    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        parser.error(f"refusing to reuse non-empty shard output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_parent / "locks" / f"{args.shard}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error(f"confirmation shard is already running: {args.shard}")
    shard_manifest_path = output_root / "confirmation_shard_freeze_manifest.json"
    shard_manifest = _shard_freeze_manifest(
        config, args.shard, central_manifest_path
    )
    if shard_manifest_path.exists():
        if not args.resume:
            parser.error(f"confirmation shard manifest already exists: {shard_manifest_path}")
        if json.loads(shard_manifest_path.read_text(encoding="utf-8")) != shard_manifest:
            parser.error("existing shard manifest differs from frozen confirmation")
    else:
        frozen._safe_write_json(shard_manifest_path, shard_manifest)

    cell_by_repetition = {
        cell["repetition"]: cell
        for cell in config["confirmation"]["replication_cells"]
    }
    condition = config["selected_condition"]
    runner = frozen.CodexRunner()
    outcomes: list[dict[str, Any]] = []
    try:
        for entry in shards[args.shard]:
            world = entry["world"]
            cell = cell_by_repetition[entry["repetition"]]
            print(
                f"START shard={args.shard} condition={condition} world={world} "
                f"root={cell['case_root']} variation={cell['variation']}",
                flush=True,
            )
            outcome = _execute_registered_episode(
                runner,
                output_root,
                config,
                config_path,
                rubric_path,
                central_manifest_path,
                args.shard,
                world,
                cell,
                args.timeout,
                resume=args.resume,
            )
            outcomes.append(outcome)
            frozen._safe_write_json(output_root / "batch_summary.json", outcomes)
            print(
                f"DONE shard={args.shard} {outcome['run_id']} "
                f"status={outcome['status']} "
                f"environment_completed={outcome['environment_completed']}",
                flush=True,
            )
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
    return 0 if all(outcome["succeeded"] for outcome in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
