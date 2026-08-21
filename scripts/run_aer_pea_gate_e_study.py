#!/usr/bin/env python3
"""Run the isolated 0.4.0-development Gate E prompt study.

This is intentionally separate from the hash-bound 0.2.0 calibration runner.
"""

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

import run_aer_pea_calibration as frozen

SCRIPT_PATH = Path(__file__).resolve()
EXPECTED_CONDITION_ORDER = (
    "l0_interface_only",
    "l1_generic_exploration",
    "l2_observation_statistics",
    "l3_competing_hypotheses",
    "l4_explicit_elimination",
    "l5_unrelated_few_shot",
    "l6_analogous_few_shot",
)
EXPECTED_PILOT_WORLDS = ("white_preference", "transient_null")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_study_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "aer.pea.gate-e-prompt-study.v1":
        raise ValueError("unsupported Gate E study config schema")
    if config.get("status") != "frozen_for_development_pilot":
        raise ValueError("Gate E study config is not frozen_for_development_pilot")
    if config.get("held_out_execution_allowed") is not False:
        raise ValueError("Gate E development study must forbid held-out execution")
    conditions = config.get("conditions")
    if not isinstance(conditions, dict) or not conditions:
        raise ValueError("Gate E study config requires conditions")
    if tuple(conditions) != EXPECTED_CONDITION_ORDER:
        raise ValueError("Gate E study conditions or their order changed")
    pilot = config.get("pilot")
    if not isinstance(pilot, dict) or not pilot.get("replication_cells"):
        raise ValueError("Gate E study config requires a pilot schedule")
    if tuple(pilot.get("condition_order", ())) != EXPECTED_CONDITION_ORDER:
        raise ValueError("Gate E pilot condition order changed")
    if tuple(pilot.get("worlds", ())) != EXPECTED_PILOT_WORLDS:
        raise ValueError("Gate E pilot worlds changed")
    if pilot.get("review_begins_only_after_all_registered_episodes_finish") is not True:
        raise ValueError("Gate E review must wait for the complete registered pilot")
    expected = (
        len(conditions) * len(pilot["worlds"]) * len(pilot["replication_cells"])
    )
    if pilot.get("registered_episode_count") != expected:
        raise ValueError("registered pilot episode count does not match the matrix")
    if expected > config.get("authorized_live_episode_cap", 0):
        raise ValueError("registered pilot exceeds the authorized live episode cap")
    return config


def _build_prompt(
    service: frozen.EpisodeService,
    study_config: dict[str, Any],
    condition: str,
) -> str:
    prompt = frozen._prompt(service, "baseline")
    common = study_config["common_interface_instruction"]
    prompt += f"\nPublic notebook protocol shared by every study condition:\n{common}\n"
    addition = study_config["conditions"][condition]["prompt_addition"]
    if addition:
        prompt += f"\nDevelopment-study instruction:\n{addition}\n"
    return prompt


def _hash_if_file(path: Path) -> str | None:
    return _sha256(path) if path.is_file() else None


def _save_exact_prompt(artifact_dir: Path, prompt: str) -> Path:
    prompt_path = artifact_dir / "prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    if prompt_path.read_text(encoding="utf-8") != prompt:
        raise RuntimeError("saved prompt does not match the exact submitted prompt")
    return prompt_path


def _freeze_manifest(
    study_config: dict[str, Any],
    study_config_path: Path,
    rubric_path: Path,
) -> dict[str, Any]:
    case_root = frozen.AER_PEA_CASE_ROOT
    source_paths = {
        "study_config": study_config_path,
        "review_rubric": rubric_path,
        "study_runner": SCRIPT_PATH,
        "frozen_v0_2_0_runner": Path(frozen.__file__).resolve(),
        "public_lab_client": case_root / "public/lab.py",
        "public_submission_schema": case_root / "public/submission.schema.json",
        "hidden_evidence_builder": case_root / "hidden/evidence.py",
        "hidden_deterministic_grader": case_root / "hidden/grader.py",
        "development_split": case_root / "construction/split.v0.1.0.json",
        "scienceworld_jar": frozen.SCIENCEWORLD_ROOT
        / "scienceworld/scienceworld.jar",
    }
    missing = [name for name, path in source_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing freeze inputs: {missing}")
    return {
        "schema_version": "aer.pea.gate-e-study-freeze.v1",
        "status": "frozen_for_development_pilot_execution",
        "study_version": study_config["study_version"],
        "held_out_execution_allowed": False,
        "historical_results_may_be_rewritten": False,
        "authorized_live_episode_cap": study_config["authorized_live_episode_cap"],
        "registered_episode_count": study_config["pilot"][
            "registered_episode_count"
        ],
        "model": study_config["model"],
        "reasoning_effort": study_config["reasoning_effort"],
        "formal_episode_action_budget": study_config[
            "formal_episode_action_budget"
        ],
        "pilot_schedule": {
            "condition_order": study_config["pilot"]["condition_order"],
            "worlds": study_config["pilot"]["worlds"],
            "replication_cells": study_config["pilot"]["replication_cells"],
        },
        "source_sha256": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in source_paths.items()
        },
    }


def run_episode(
    runner: frozen.CodexRunner,
    output_root: Path,
    study_config: dict[str, Any],
    study_config_path: Path,
    rubric_path: Path,
    condition: str,
    world: str,
    cell: dict[str, int],
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    repetition = cell["repetition"]
    variation = cell["variation"]
    case_root = cell["case_root"]
    run_id = (
        f"{world}-variation-{variation:02d}-root-{case_root:04d}-run-{repetition:02d}"
    )
    artifact_dir = output_root / condition / run_id
    if artifact_dir.exists():
        raise RuntimeError(f"refusing to overwrite {artifact_dir}")
    artifact_dir.mkdir(parents=True)

    with tempfile.TemporaryDirectory(prefix="ape4-", dir="/tmp") as temporary:
        workspace = Path(temporary)
        client_path = workspace / "lab.py"
        schema_path = workspace / "submission.schema.json"
        protocol_path = workspace / "STUDY_PROTOCOL.md"
        shutil.copy2(frozen.AER_PEA_CASE_ROOT / "public" / "lab.py", client_path)
        shutil.copy2(
            frozen.AER_PEA_CASE_ROOT / "public" / "submission.schema.json",
            schema_path,
        )
        protocol_path.write_text(
            study_config["common_interface_instruction"] + "\n", encoding="utf-8"
        )
        socket_path = workspace / "scienceworld.sock"
        trajectory_path = artifact_dir / "public_environment_trajectory.jsonl"
        operator_window_path = artifact_dir / "operator_action_windows.jsonl"
        service = frozen.EpisodeService(
            world,
            variation,
            case_root,
            trajectory_path,
            operator_window_path,
            study_config["formal_episode_action_budget"],
            matched_pre_exposure=bool(
                study_config["matched_pre_exposure"]["required"]
            ),
        )
        server = frozen._UnixServer(str(socket_path), frozen._Handler)
        server.episode = service  # type: ignore[attr-defined]
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        started = time.time()
        try:
            prompt = _build_prompt(service, study_config, condition)
            _save_exact_prompt(artifact_dir, prompt)
            config = frozen.CodexRunConfig(
                workspace=workspace,
                artifact_dir=artifact_dir / "codex",
                prompt=prompt,
                output_schema=schema_path,
                model=study_config["model"],
                reasoning_effort=study_config["reasoning_effort"],
                timeout_seconds=timeout_seconds,
                sandbox="workspace-write",
                ephemeral=True,
                shell_tool_enabled=True,
                extra_config=("features.fast_mode=false",),
                unix_socket_allowlist=(socket_path,),
                deny_read_paths=(
                    frozen.AER_BENCH_ROOT,
                    frozen.SCIENCEWORLD_ROOT,
                    output_root,
                ),
            )
            result = runner.run(config)
            hidden_summary = service.env.get_aer_pea_case_summary()
            hidden_events = service.env.get_aer_pea_case_events()
            hidden_reproduction_events = (
                service.env.get_aer_pea_case_reproduction_events()
            )
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=5)
            service.close()

    normalized_path = artifact_dir / "codex" / "normalized_events.jsonl"
    if result.events_path.is_file():
        frozen.write_normalized(
            normalized_path,
            frozen.normalize_events(frozen.read_jsonl(result.events_path)),
        )
    frozen._safe_write_json(artifact_dir / "hidden_summary.json", hidden_summary)
    frozen._safe_write_json(artifact_dir / "hidden_events.json", hidden_events)
    frozen._safe_write_json(
        artifact_dir / "hidden_reproduction_events.json",
        hidden_reproduction_events,
    )
    evidence_module = frozen._load_case_module(
        "aer_pea_gate_e_evidence",
        frozen.AER_PEA_CASE_ROOT / "hidden" / "evidence.py",
    )
    grader_module = frozen._load_case_module(
        "aer_pea_gate_e_grader",
        frozen.AER_PEA_CASE_ROOT / "hidden" / "grader.py",
    )
    grading_events = evidence_module.build_events(
        frozen.read_jsonl(trajectory_path),
        frozen.read_jsonl(operator_window_path),
        world=world,
    )
    grading_events_path = artifact_dir / "grading_events.jsonl"
    with grading_events_path.open("w", encoding="utf-8") as handle:
        for event in grading_events:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    grade_path = artifact_dir / "grade.json"
    if result.final_output_path.is_file():
        submission = json.loads(result.final_output_path.read_text(encoding="utf-8"))
        grade = grader_module.grade(
            submission,
            grading_events,
            expected_world=world,
            expected_height_trait="dominant" if variation < 15 else "recessive",
        )
        frozen._safe_write_json(grade_path, grade)

    metadata_path = artifact_dir / "run_metadata.json"
    frozen._safe_write_json(
        metadata_path,
        {
            "schema_version": "aer.pea.gate-e-study-run.v1",
            "run_id": run_id,
            "study_version": study_config["study_version"],
            "condition": condition,
            "assistance_coordinates": study_config["conditions"][condition][
                "assistance_coordinates"
            ],
            "world": world,
            "variation": variation,
            "case_root": case_root,
            "repetition": repetition,
            "model": study_config["model"],
            "reasoning_effort": study_config["reasoning_effort"],
            "step_limit": study_config["formal_episode_action_budget"],
            "matched_pre_exposure": True,
            "matched_pre_exposure_observation_count": len(
                service.pre_exposure_observations
            ),
            "prompt_sha256": _sha256(artifact_dir / "prompt.txt"),
            "study_config_sha256": _sha256(study_config_path),
            "review_rubric_sha256": _sha256(rubric_path),
            "runner_sha256": _sha256(SCRIPT_PATH),
            "frozen_v0_2_0_runner_sha256": _sha256(Path(frozen.__file__).resolve()),
            "lab_client_sha256": _sha256(frozen.AER_PEA_CASE_ROOT / "public/lab.py"),
            "submission_schema_sha256": _sha256(
                frozen.AER_PEA_CASE_ROOT / "public/submission.schema.json"
            ),
            "evidence_builder_sha256": _sha256(
                frozen.AER_PEA_CASE_ROOT / "hidden/evidence.py"
            ),
            "deterministic_grader_sha256": _sha256(
                frozen.AER_PEA_CASE_ROOT / "hidden/grader.py"
            ),
            "scienceworld_jar_sha256": _sha256(
                frozen.SCIENCEWORLD_ROOT / "scienceworld/scienceworld.jar"
            ),
            "codex_version": result.codex_version,
            "thread_id": result.thread_id,
            "status": result.status,
            "returncode": result.returncode,
            "errors": result.errors,
            "usage": result.usage,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "environment_completed": service.completed,
            "files_sha256": {
                "prompt.txt": _sha256(artifact_dir / "prompt.txt"),
                "public_environment_trajectory.jsonl": _sha256(trajectory_path),
                "operator_action_windows.jsonl": _sha256(operator_window_path),
                "hidden_events.json": _sha256(artifact_dir / "hidden_events.json"),
                "hidden_reproduction_events.json": _sha256(
                    artifact_dir / "hidden_reproduction_events.json"
                ),
                "hidden_summary.json": _sha256(artifact_dir / "hidden_summary.json"),
                "grading_events.jsonl": _sha256(grading_events_path),
                "grade.json": _hash_if_file(grade_path),
                "codex/events.jsonl": _hash_if_file(result.events_path),
                "codex/normalized_events.jsonl": _hash_if_file(normalized_path),
                "codex/final.json": _hash_if_file(result.final_output_path),
            },
        },
    )
    return {
        "run_id": run_id,
        "condition": condition,
        "world": world,
        "status": result.status,
        "succeeded": result.succeeded,
        "environment_completed": service.completed,
        "artifact_dir": str(artifact_dir),
        "metadata_sha256": _sha256(metadata_path),
        "errors": result.errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study-config", type=Path, required=True)
    parser.add_argument("--condition", action="append")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    study_config_path = args.study_config.resolve()
    study_config = _load_study_config(study_config_path)
    rubric_path = (
        frozen.AER_BENCH_ROOT / study_config["review"]["rubric_path"]
    ).resolve()
    if not rubric_path.is_file():
        parser.error(f"missing review rubric: {rubric_path}")
    conditions = args.condition or study_config["pilot"]["condition_order"]
    if len(conditions) != len(set(conditions)):
        parser.error("duplicate study condition")
    unknown = sorted(set(conditions) - set(study_config["conditions"]))
    if unknown:
        parser.error(f"unknown study conditions: {unknown}")
    for cell in study_config["pilot"]["replication_cells"]:
        try:
            frozen._validate_split(
                "development",
                cell["variation"],
                cell["case_root"],
                None,
            )
        except ValueError as error:
            parser.error(str(error))

    output_root = args.output.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        parser.error(f"refusing to reuse non-empty output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    frozen._safe_write_json(
        output_root / "study_freeze_manifest.json",
        _freeze_manifest(study_config, study_config_path, rubric_path),
    )
    runner = frozen.CodexRunner()
    outcomes: list[dict[str, Any]] = []
    for condition in conditions:
        for world in study_config["pilot"]["worlds"]:
            for cell in study_config["pilot"]["replication_cells"]:
                print(
                    f"START condition={condition} world={world} "
                    f"root={cell['case_root']} variation={cell['variation']}",
                    flush=True,
                )
                outcome = run_episode(
                    runner,
                    output_root,
                    study_config,
                    study_config_path,
                    rubric_path,
                    condition,
                    world,
                    cell,
                    timeout_seconds=args.timeout,
                )
                outcomes.append(outcome)
                frozen._safe_write_json(output_root / "batch_summary.json", outcomes)
                print(
                    f"DONE {condition}/{outcome['run_id']} "
                    f"status={outcome['status']} "
                    f"environment_completed={outcome['environment_completed']}",
                    flush=True,
                )
    return 0 if all(outcome["succeeded"] for outcome in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
