#!/usr/bin/env python3
"""Run one frozen shard of the 0.4.2 matched Gate E development matrix."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import run_aer_pea_calibration as frozen
import run_aer_pea_gate_e_study as episode_runner

SCRIPT_PATH = Path(__file__).resolve()
EXPECTED_CONDITIONS = (
    "l0_interface_only",
    "l1_generic_exploration",
    "l2_observation_statistics",
    "l4_explicit_elimination",
)
EXPECTED_WORLDS = (
    "white_preference",
    "position_attraction",
    "plant_attractiveness",
    "fertility_difference",
    "transient_null",
    "clean",
)
EXPECTED_CELLS = (
    {"repetition": 1, "case_root": 211, "variation": 13},
    {"repetition": 2, "case_root": 401, "variation": 25},
)
EXPECTED_LATIN_ORDERS = (
    (
        "l0_interface_only",
        "l1_generic_exploration",
        "l4_explicit_elimination",
        "l2_observation_statistics",
    ),
    (
        "l1_generic_exploration",
        "l2_observation_statistics",
        "l0_interface_only",
        "l4_explicit_elimination",
    ),
    (
        "l2_observation_statistics",
        "l4_explicit_elimination",
        "l1_generic_exploration",
        "l0_interface_only",
    ),
    (
        "l4_explicit_elimination",
        "l0_interface_only",
        "l2_observation_statistics",
        "l1_generic_exploration",
    ),
)
EXPECTED_SOURCE_BINDINGS = {
    "prompt_source": (
        "cases/science/mendelian_genetics_known_plant_aer/construction/"
        "gate-e-prompt-study.v0.4.0-development.json",
        "b1c48c1d6bbbd32b9c0821a30ceae3af971858056fc1d008e42eb4e1358f475b",
    ),
    "pilot_result": (
        "cases/science/mendelian_genetics_known_plant_aer/construction/"
        "gate-e-pilot-result.v0.4.0-development.json",
        "d04da8bdff5512617c3c3fa97027d8817a6b0d5b8acae9930488a0c956c2963b",
    ),
    "confirmation_study": (
        "cases/science/mendelian_genetics_known_plant_aer/construction/"
        "gate-e-confirmation-study.v0.4.1-development.json",
        "15fd18e839e4b1c19ab839935386a85aeade26e4e0e2415b8db71b9ca662756a",
    ),
    "confirmation_result": (
        "cases/science/mendelian_genetics_known_plant_aer/construction/"
        "gate-e-confirmation-result.v0.4.1-development.json",
        "f6b68646e881f8f3709cd6c0c2d5363b5e0da1214e246f165a372dd94f4a898a",
    ),
    "development_split": (
        "cases/science/mendelian_genetics_known_plant_aer/construction/"
        "split.v0.3.0.json",
        "7d6ec7dd9c7dfbbaff1f51a695b7ab7df51efe27dd56defc76b43ba7209c8de8",
    ),
    "review_rubric": (
        "cases/science/mendelian_genetics_known_plant_aer/construction/"
        "gate-e-review-rubric.v0.4.0-development.json",
        "32ff7839aa5628d1760423e2aa8ef72b5a1bac495b66fcd4175e747aaa1a2715",
    ),
}


def _expected_shards() -> dict[str, dict[str, Any]]:
    shards: dict[str, dict[str, Any]] = {}
    index = 0
    for world in EXPECTED_WORLDS:
        for cell in EXPECTED_CELLS:
            shards[f"shard-{index + 1:02d}"] = {
                "world": world,
                "repetition": cell["repetition"],
                "condition_order": list(EXPECTED_LATIN_ORDERS[index % 4]),
            }
            index += 1
    return shards


EXPECTED_SHARDS = _expected_shards()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bound_path(config: dict[str, Any], name: str) -> Path:
    binding = config["source_bindings"][name]
    return (frozen.AER_BENCH_ROOT / binding["path"]).resolve()


def _validate_latin_balance(shards: dict[str, dict[str, Any]]) -> None:
    orders = [tuple(shard["condition_order"]) for shard in shards.values()]
    if Counter(orders) != Counter({order: 3 for order in EXPECTED_LATIN_ORDERS}):
        raise ValueError("matched development condition orders are not frozen")
    for position in range(len(EXPECTED_CONDITIONS)):
        if Counter(order[position] for order in orders) != Counter(
            {condition: 3 for condition in EXPECTED_CONDITIONS}
        ):
            raise ValueError("matched development condition positions are unbalanced")
    transitions = Counter(
        (order[index], order[index + 1])
        for order in orders
        for index in range(len(order) - 1)
    )
    expected_transitions = Counter(
        {
            (left, right): 3
            for left in EXPECTED_CONDITIONS
            for right in EXPECTED_CONDITIONS
            if left != right
        }
    )
    if transitions != expected_transitions:
        raise ValueError("matched development condition carryover is unbalanced")


def _load_study_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != (
        "aer.pea.gate-e-matched-development-study.v1"
    ):
        raise ValueError("unsupported Gate E matched development config schema")
    if config.get("status") != "frozen_for_matched_development":
        raise ValueError("Gate E matched development config is not frozen")
    if config.get("held_out_execution_allowed") is not False:
        raise ValueError("matched development must forbid held-out execution")
    if config.get("historical_results_may_be_rewritten") is not False:
        raise ValueError("matched development must preserve historical results")
    if config.get("official_leaderboard_result") is not False:
        raise ValueError("matched development cannot be an official result")
    if (
        config.get("model") != "gpt-5.6-sol"
        or config.get("reasoning_effort") != "high"
        or config.get("formal_episode_action_budget") != 1000
    ):
        raise ValueError("matched development model or action budget changed")
    if tuple(config.get("conditions", ())) != EXPECTED_CONDITIONS:
        raise ValueError("matched development conditions or order changed")
    if config.get("newness_scope") != (
        "previously_unused_live_root_variation_cells"
    ):
        raise ValueError("matched development newness scope changed")
    if config.get("roots_not_globally_unseen") is not True:
        raise ValueError("matched development must disclose reused roots")

    matrix = config.get("matched_development")
    if not isinstance(matrix, dict):
        raise ValueError("matched development matrix is missing")
    if tuple(matrix.get("worlds", ())) != EXPECTED_WORLDS:
        raise ValueError("matched development worlds changed")
    if tuple(matrix.get("replication_cells", ())) != EXPECTED_CELLS:
        raise ValueError("matched development replication cells changed")
    if matrix.get("shard_parallelism") != 12:
        raise ValueError("matched development must register twelve shards")
    if matrix.get("one_world_repetition_cell_per_shard") is not True:
        raise ValueError("matched development shard isolation changed")
    if matrix.get("conditions_run_sequentially_within_shard") is not True:
        raise ValueError("matched development within-shard execution changed")
    if matrix.get("review_begins_only_after_all_registered_episodes_finish") is not True:
        raise ValueError("matched development review boundary changed")
    expected_count = (
        len(EXPECTED_CONDITIONS) * len(EXPECTED_WORLDS) * len(EXPECTED_CELLS)
    )
    if matrix.get("registered_episode_count") != expected_count or config.get(
        "registered_matched_development_episode_count"
    ) != expected_count:
        raise ValueError("matched development episode count changed")
    previous = config.get("previous_valid_live_episode_count")
    cumulative = config.get("cumulative_valid_live_episode_count_if_complete")
    if previous != 26 or cumulative != previous + expected_count:
        raise ValueError("matched development cumulative live count changed")
    if cumulative > config.get("authorized_live_episode_cap", 0):
        raise ValueError("matched development exceeds the live authorization")
    shards = matrix.get("shards")
    if shards != EXPECTED_SHARDS:
        raise ValueError("matched development shard schedule changed")
    _validate_latin_balance(shards)

    bindings = config.get("source_bindings")
    if not isinstance(bindings, dict):
        raise ValueError("matched development source bindings are missing")
    for name, (expected_relative, expected_hash) in EXPECTED_SOURCE_BINDINGS.items():
        binding = bindings.get(name)
        if binding != {"path": expected_relative, "sha256": expected_hash}:
            raise ValueError(f"matched development {name} binding changed")
        source_path = (frozen.AER_BENCH_ROOT / expected_relative).resolve()
        if not source_path.is_file() or _sha256(source_path) != expected_hash:
            raise ValueError(f"matched development {name} hash mismatch")

    prompt_source = json.loads(
        _bound_path(config, "prompt_source").read_text(encoding="utf-8")
    )
    if config.get("common_interface_instruction") != prompt_source.get(
        "common_interface_instruction"
    ):
        raise ValueError("common notebook interface differs from frozen v0.4.0")
    selected_prompts = {
        condition: prompt_source["conditions"][condition]
        for condition in EXPECTED_CONDITIONS
    }
    if config["conditions"] != selected_prompts:
        raise ValueError("condition prompt differs from frozen v0.4.0")

    split = json.loads(
        _bound_path(config, "development_split").read_text(encoding="utf-8")
    )
    if split.get("spec_version") != "0.3.0" or split.get("status") != "frozen":
        raise ValueError("matched development requires frozen split 0.3.0")
    for cell in EXPECTED_CELLS:
        if (
            cell["case_root"] not in split["development"]["roots"]
            or cell["variation"] not in split["development"]["variations"]
        ):
            raise ValueError("matched development cell is not development-only")
        if (
            cell["case_root"] in split["held_out"]["roots"]
            or cell["variation"] in split["held_out"]["variations"]
        ):
            raise ValueError("matched development cell touches held-out")

    confirmation = json.loads(
        _bound_path(config, "confirmation_study").read_text(encoding="utf-8")
    )
    prior_cells = {
        (cell["case_root"], cell["variation"])
        for cell in prompt_source["pilot"]["replication_cells"]
    }
    prior_cells.update(
        (cell["case_root"], cell["variation"])
        for cell in confirmation["confirmation"]["replication_cells"]
    )
    new_cells = {
        (cell["case_root"], cell["variation"]) for cell in EXPECTED_CELLS
    }
    if not new_cells.isdisjoint(prior_cells):
        raise ValueError("matched development root-variation cells were used live")
    if not {cell["case_root"] for cell in EXPECTED_CELLS}.issubset(
        {root for root, _ in prior_cells}
    ):
        raise ValueError("matched development reused-root disclosure is inconsistent")

    result = json.loads(
        _bound_path(config, "confirmation_result").read_text(encoding="utf-8")
    )
    if result.get("cumulative_valid_live_episode_count") != previous:
        raise ValueError("matched development prior live count is not v0.4.1")
    review = config.get("review", {})
    if review.get("rubric_path") != bindings["review_rubric"]["path"]:
        raise ValueError("matched development review rubric path changed")
    if review.get("llm_scores_are_development_diagnostics_only") is not True:
        raise ValueError("matched development LLM review boundary changed")
    if review.get("future_leaderboard_llm_judge_allowed") is not False:
        raise ValueError("matched development cannot authorize an LLM judge")
    stopping = config.get("stopping_rules", {})
    if stopping.get("completed_solver_episode_may_be_retried") is not False:
        raise ValueError("matched development cannot retry a solver episode")
    if stopping.get(
        "pre_solver_machine_verified_infrastructure_retry_limit"
    ) != 1:
        raise ValueError("matched development pre-solver retry limit changed")
    return config


def _source_paths(
    config: dict[str, Any], config_path: Path
) -> dict[str, Path]:
    case_root = frozen.AER_PEA_CASE_ROOT
    paths = {
        "matched_development_config": config_path,
        "matched_development_runner": SCRIPT_PATH,
        "episode_runner": Path(episode_runner.__file__).resolve(),
        "frozen_v0_2_0_runner": Path(frozen.__file__).resolve(),
        "public_lab_client": case_root / "public/lab.py",
        "public_submission_schema": case_root / "public/submission.schema.json",
        "hidden_evidence_builder": case_root / "hidden/evidence.py",
        "hidden_deterministic_grader": case_root / "hidden/grader.py",
        "selection_ledger": frozen.AER_BENCH_ROOT
        / "artifacts/aer_pea_case/selection_ledger.jsonl",
        "scienceworld_jar": frozen.SCIENCEWORLD_ROOT
        / "scienceworld/scienceworld.jar",
    }
    for name in EXPECTED_SOURCE_BINDINGS:
        paths[name] = _bound_path(config, name)
    return paths


def _central_freeze_manifest(
    config: dict[str, Any], config_path: Path
) -> dict[str, Any]:
    source_paths = _source_paths(config, config_path)
    missing = [name for name, path in source_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing matched development inputs: {missing}")
    matrix = config["matched_development"]
    return {
        "schema_version": "aer.pea.gate-e-matched-development-freeze.v1",
        "status": "frozen_for_matched_development_execution",
        "study_version": config["study_version"],
        "registered_episode_count": config[
            "registered_matched_development_episode_count"
        ],
        "previous_valid_live_episode_count": config[
            "previous_valid_live_episode_count"
        ],
        "cumulative_valid_live_episode_count_if_complete": config[
            "cumulative_valid_live_episode_count_if_complete"
        ],
        "authorized_live_episode_cap": config["authorized_live_episode_cap"],
        "held_out_execution_allowed": False,
        "historical_results_may_be_rewritten": False,
        "official_leaderboard_result": False,
        "model": config["model"],
        "reasoning_effort": config["reasoning_effort"],
        "formal_episode_action_budget": config[
            "formal_episode_action_budget"
        ],
        "newness_scope": config["newness_scope"],
        "roots_not_globally_unseen": config["roots_not_globally_unseen"],
        "conditions": list(EXPECTED_CONDITIONS),
        "worlds": matrix["worlds"],
        "replication_cells": matrix["replication_cells"],
        "shards": matrix["shards"],
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


def _run_id(world: str, cell: dict[str, int]) -> str:
    return (
        f"{world}-variation-{cell['variation']:02d}-"
        f"root-{cell['case_root']:04d}-run-{cell['repetition']:02d}"
    )


def _episode_key(condition: str, world: str, cell: dict[str, int]) -> str:
    return f"{condition}--{_run_id(world, cell)}"


def _verify_finalized(
    state: dict[str, Any], output_root: Path, config_path: Path,
    central_manifest_path: Path
) -> dict[str, Any]:
    if state.get("phase") != "finalized":
        raise RuntimeError("only finalized matched episodes may be resumed")
    outcome = state.get("outcome")
    if not isinstance(outcome, dict):
        raise RuntimeError("finalized matched state is missing its outcome")
    artifact_dir = output_root / state["condition"] / state["run_id"]
    metadata_path = artifact_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("matched_development_episode_key") != state.get("episode_key"):
        raise RuntimeError("finalized matched episode identity mismatch")
    bindings = metadata.get("matched_development_bindings", {})
    if bindings.get("config", {}).get("sha256") != _sha256(config_path):
        raise RuntimeError("finalized matched config hash mismatch")
    if bindings.get("central_freeze", {}).get("sha256") != _sha256(
        central_manifest_path
    ):
        raise RuntimeError("finalized matched central freeze hash mismatch")
    for relative, expected in metadata["files_sha256"].items():
        if expected is None:
            continue
        artifact = artifact_dir / relative
        if not artifact.is_file() or _sha256(artifact) != expected:
            raise RuntimeError(f"finalized matched artifact hash mismatch: {artifact}")
    if _sha256(metadata_path) != state.get("metadata_sha256"):
        raise RuntimeError("finalized matched metadata hash mismatch")
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


def _materialize_transcript(artifact_dir: Path) -> Path:
    transcript_path = artifact_dir / "codex/transcript.jsonl"
    if transcript_path.exists():
        raise RuntimeError(f"refusing to overwrite transcript: {transcript_path}")
    candidates = (
        artifact_dir / "codex/normalized_events.jsonl",
        artifact_dir / "codex/events.jsonl",
    )
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        raise FileNotFoundError("solver transcript source is missing")
    shutil.copy2(source, transcript_path)
    return transcript_path


def _finalize_metadata(
    outcome: dict[str, Any],
    config: dict[str, Any],
    config_path: Path,
    central_manifest_path: Path,
    shard: str,
    condition: str,
    condition_position: int,
    episode_key: str,
    attempt_number: int,
) -> str:
    artifact_dir = Path(outcome["artifact_dir"])
    transcript_path = _materialize_transcript(artifact_dir)
    required = (
        artifact_dir / "prompt.txt",
        artifact_dir / "public_environment_trajectory.jsonl",
        transcript_path,
        artifact_dir / "codex/final.json",
        artifact_dir / "grade.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"matched episode required artifacts missing: {missing}")
    metadata_path = artifact_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    split_path = _bound_path(config, "development_split")
    prompt_source_path = _bound_path(config, "prompt_source")
    metadata["matched_development_episode_key"] = episode_key
    metadata["matched_development_shard"] = shard
    metadata["matched_development_condition_position"] = condition_position
    metadata["matched_development_attempt"] = attempt_number
    metadata["matched_development_bindings"] = {
        "runner": {"path": str(SCRIPT_PATH), "sha256": _sha256(SCRIPT_PATH)},
        "config": {
            "path": str(config_path),
            "sha256": _sha256(config_path),
        },
        "central_freeze": {
            "path": str(central_manifest_path),
            "sha256": _sha256(central_manifest_path),
        },
        "development_split": {
            "path": str(split_path),
            "sha256": _sha256(split_path),
        },
        "prompt_source": {
            "path": str(prompt_source_path),
            "sha256": _sha256(prompt_source_path),
        },
    }
    metadata["files_sha256"]["codex/transcript.jsonl"] = _sha256(
        transcript_path
    )
    frozen._safe_write_json(metadata_path, metadata)
    return _sha256(metadata_path)


def _execute_registered_episode(
    delegate: frozen.CodexRunner,
    output_root: Path,
    config: dict[str, Any],
    config_path: Path,
    rubric_path: Path,
    central_manifest_path: Path,
    shard: str,
    condition: str,
    condition_position: int,
    world: str,
    cell: dict[str, int],
    timeout_seconds: int,
    *,
    resume: bool,
) -> dict[str, Any]:
    run_id = _run_id(world, cell)
    episode_key = _episode_key(condition, world, cell)
    state_path = output_root / "_episode_states" / f"{episode_key}.json"
    artifact_dir = output_root / condition / run_id
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if resume and state.get("phase") == "finalized":
            return _verify_finalized(
                state, output_root, config_path, central_manifest_path
            )
        if not resume:
            raise RuntimeError(f"matched episode state already exists: {state_path}")
        if state.get("phase") not in {"registered", "pre_solver_failed"}:
            raise RuntimeError(
                f"refusing to retry post-solver matched state: {state['phase']}"
            )
    else:
        state = {
            "schema_version": "aer.pea.gate-e-matched-development-episode-state.v1",
            "episode_key": episode_key,
            "run_id": run_id,
            "shard": shard,
            "condition": condition,
            "condition_position": condition_position,
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
            raise RuntimeError(f"pre-solver retry limit exhausted for {episode_key}")
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
            metadata_sha256 = _finalize_metadata(
                outcome,
                config,
                config_path,
                central_manifest_path,
                shard,
                condition,
                condition_position,
                episode_key,
                attempt_number,
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
                    / condition
                    / run_id
                    / f"attempt-{attempt_number:02d}"
                )
                preserved.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(artifact_dir), str(preserved))
                attempt["preserved_artifact_dir"] = str(preserved)
            frozen._safe_write_json(state_path, state)
            if attempt_number > retry_limit:
                state["phase"] = "pre_solver_retry_exhausted"
                frozen._safe_write_json(state_path, state)
                raise
            continue

        outcome["metadata_sha256"] = metadata_sha256
        attempt["finalized_at_unix"] = time.time()
        state["phase"] = "finalized"
        state["outcome"] = outcome
        state["metadata_sha256"] = metadata_sha256
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
    config = _load_study_config(config_path)
    shards = config["matched_development"]["shards"]
    if args.shard not in shards:
        parser.error(f"unknown matched development shard: {args.shard}")
    rubric_path = _bound_path(config, "review_rubric")

    output_parent = args.output_parent.resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    central_manifest_path = (
        output_parent / "matched_development_freeze_manifest.json"
    )
    _write_once_or_validate(
        central_manifest_path, _central_freeze_manifest(config, config_path)
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
        parser.error(f"matched development shard is already running: {args.shard}")

    registration = {
        "schema_version": "aer.pea.gate-e-matched-development-shard.v1",
        "study_version": config["study_version"],
        "shard": args.shard,
        **shards[args.shard],
        "central_freeze": {
            "path": str(central_manifest_path),
            "sha256": _sha256(central_manifest_path),
        },
        "held_out_execution_allowed": False,
    }
    registration_path = output_root / "shard_registration.json"
    if registration_path.exists():
        if not args.resume:
            parser.error(f"shard registration already exists: {registration_path}")
        if json.loads(registration_path.read_text(encoding="utf-8")) != registration:
            parser.error("existing shard registration differs from frozen matrix")
    else:
        frozen._safe_write_json(registration_path, registration)

    cell_by_repetition = {
        cell["repetition"]: cell
        for cell in config["matched_development"]["replication_cells"]
    }
    shard_registration = shards[args.shard]
    world = shard_registration["world"]
    cell = cell_by_repetition[shard_registration["repetition"]]
    runner = frozen.CodexRunner()
    outcomes: list[dict[str, Any]] = []
    try:
        for position, condition in enumerate(
            shard_registration["condition_order"], start=1
        ):
            print(
                f"START shard={args.shard} position={position} "
                f"condition={condition} world={world} "
                f"root={cell['case_root']} variation={cell['variation']}",
                flush=True,
            )
            try:
                outcome = _execute_registered_episode(
                    runner,
                    output_root,
                    config,
                    config_path,
                    rubric_path,
                    central_manifest_path,
                    args.shard,
                    condition,
                    position,
                    world,
                    cell,
                    args.timeout,
                    resume=args.resume,
                )
            except Exception as error:
                episode_key = _episode_key(condition, world, cell)
                state_path = (
                    output_root / "_episode_states" / f"{episode_key}.json"
                )
                phase = "state_missing"
                if state_path.is_file():
                    phase = json.loads(
                        state_path.read_text(encoding="utf-8")
                    ).get("phase", "unknown")
                outcome = {
                    "run_id": _run_id(world, cell),
                    "condition": condition,
                    "world": world,
                    "status": "runner_exception",
                    "succeeded": False,
                    "environment_completed": False,
                    "artifact_dir": str(
                        output_root / condition / _run_id(world, cell)
                    ),
                    "episode_key": episode_key,
                    "episode_state_path": str(state_path),
                    "episode_state_phase": phase,
                    "errors": [
                        {"type": type(error).__name__, "message": str(error)}
                    ],
                }
                print(
                    f"FAILED shard={args.shard} condition={condition} "
                    f"{outcome['run_id']} phase={phase} "
                    f"error={type(error).__name__}: {error}",
                    flush=True,
                )
            outcomes.append(outcome)
            frozen._safe_write_json(output_root / "batch_summary.json", outcomes)
            if outcome["status"] != "runner_exception":
                print(
                    f"DONE shard={args.shard} condition={condition} "
                    f"{outcome['run_id']} status={outcome['status']} "
                    f"environment_completed={outcome['environment_completed']}",
                    flush=True,
                )
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
    return 0 if all(outcome["succeeded"] for outcome in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
