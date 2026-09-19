#!/usr/bin/env python3
"""Run the frozen 18-episode Gate E prompt-separation development mini-study.

This runner is deliberately independent of the frozen v0.4.0--v0.4.2 runners.  It gives all
three conditions the same neutral G0 submission schema and base laboratory interface, then adds
only the condition-specific prompt text frozen in the v0.4.3 development configuration.  It
never invokes the deterministic anomaly grader and it never retries a solver-started episode.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import inspect
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import run_aer_pea_calibration as frozen

SCRIPT_PATH = Path(__file__).resolve()
CONFIG_SCHEMA = "aer.pea.gate-e-prompt-separation-mini-study.v1"
STUDY_VERSION = "0.4.3-prompt-separation-mini-development"
EXPECTED_STATUS = "frozen_for_development_execution"
EXPECTED_CONDITIONS = (
    "l0_g0_only",
    "l1_generic_salience",
    "l4_explicit_investigation",
)
EXPECTED_WORLDS = (
    "white_preference",
    "position_attraction",
    "plant_attractiveness",
    "fertility_difference",
    "transient_null",
    "clean",
)
EXPECTED_CELL = {"repetition": 1, "case_root": 307, "variation": 21}
EXPECTED_EPISODE_COUNT = 18
EXPECTED_PARALLELISM = 12
EXPECTED_MODEL = "gpt-5.6-sol"
EXPECTED_REASONING_EFFORT = "high"
EXPECTED_ACTION_BUDGET = 1000
PROMPT_SUFFIX = (
    "\nComplete the commissioned Mendelian-height task autonomously and stop when it "
    "is complete.\nReturn only one JSON object satisfying the provided neutral "
    "mini-study schema.\n"
)
EXPECTED_BINDING_PATHS = {
    "neutral_submission_schema": (
        "cases/science/mendelian_genetics_known_plant_aer/construction/"
        "gate-e-prompt-separation-mini-submission.schema.json"
    ),
    "development_split": (
        "cases/science/mendelian_genetics_known_plant_aer/construction/split.v0.3.0.json"
    ),
    "public_lab_client": (
        "cases/science/mendelian_genetics_known_plant_aer/public/lab.py"
    ),
    "scienceworld_jar": "external/ScienceWorld/scienceworld/scienceworld.jar",
    "selection_ledger_at_freeze": "artifacts/aer_pea_case/selection_ledger.jsonl",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _bound_path(config: dict[str, Any], name: str) -> Path:
    binding = config["source_bindings"][name]
    path = (frozen.AER_BENCH_ROOT / binding["path"]).resolve()
    try:
        path.relative_to(frozen.AER_BENCH_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"mini-study binding escapes the repository: {name}") from error
    return path


def _validate_neutral_schema(path: Path) -> None:
    schema = json.loads(path.read_text(encoding="utf-8"))
    required = schema.get("required")
    properties = schema.get("properties")
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise ValueError("neutral mini-study submission schema must be a closed object")
    expected_fields = ["height_trait", "completed", "evidence", "notes"]
    if required != expected_fields:
        raise ValueError("neutral mini-study schema required fields changed")
    if not isinstance(properties, dict) or set(properties) != set(expected_fields):
        raise ValueError("neutral mini-study schema must contain only G0 and neutral notes fields")
    if properties["notes"] != {"type": "string"}:
        raise ValueError("neutral mini-study notes field must remain free text")


def _task_key(condition: str, world: str) -> tuple[str, str, int]:
    return condition, world, EXPECTED_CELL["repetition"]


def _validate_shards(matrix: dict[str, Any]) -> None:
    shards = matrix.get("shards")
    if not isinstance(shards, dict):
        raise ValueError("mini-study shards are missing")
    expected = {
        _task_key(condition, world)
        for condition in EXPECTED_CONDITIONS
        for world in EXPECTED_WORLDS
    }
    actual: set[tuple[str, str, int]] = set()
    for shard_id, shard in shards.items():
        if not isinstance(shard_id, str) or not isinstance(shard, dict):
            raise ValueError("mini-study shard registration is invalid")
        if "condition" in shard:
            key = (
                shard.get("condition"),
                shard.get("world"),
                shard.get("repetition", 1),
            )
            if key in actual:
                raise ValueError(f"duplicate mini-study shard cell: {key}")
            actual.add(key)
            continue
        order = shard.get("condition_order")
        if (
            not isinstance(order, list)
            or set(order) != set(EXPECTED_CONDITIONS)
            or len(order) != len(EXPECTED_CONDITIONS)
        ):
            raise ValueError(f"invalid mini-study condition order: {shard_id}")
        world = shard.get("world")
        repetition = shard.get("repetition", 1)
        for condition in order:
            key = condition, world, repetition
            if key in actual:
                raise ValueError(f"duplicate mini-study shard cell: {key}")
            actual.add(key)
    if actual != expected:
        raise ValueError("mini-study shards do not cover exactly 18 registered episodes")


def _load_study_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported prompt-separation mini-study config schema")
    if config.get("study_version") != STUDY_VERSION:
        raise ValueError("prompt-separation mini-study version changed")
    if config.get("status") != EXPECTED_STATUS:
        raise ValueError("prompt-separation mini-study config is not frozen")
    if config.get("held_out_execution_allowed") is not False:
        raise ValueError("prompt-separation mini-study must forbid held-out execution")
    if config.get("historical_results_may_be_rewritten") is not False:
        raise ValueError("prompt-separation mini-study must preserve historical results")
    if config.get("official_leaderboard_result") is not False:
        raise ValueError("prompt-separation mini-study cannot be a leaderboard result")
    if (
        config.get("model") != EXPECTED_MODEL
        or config.get("reasoning_effort") != EXPECTED_REASONING_EFFORT
        or config.get("formal_episode_action_budget") != EXPECTED_ACTION_BUDGET
    ):
        raise ValueError("mini-study model, reasoning effort, or action budget changed")
    if (
        config.get("authorized_live_episode_cap") != 200
        or config.get("previous_valid_live_episode_count") != 74
        or config.get("registered_prompt_separation_mini_episode_count")
        != EXPECTED_EPISODE_COUNT
        or config.get("cumulative_valid_live_episode_count_if_complete") != 92
    ):
        raise ValueError("mini-study live-episode registration or authorization changed")

    conditions = config.get("conditions")
    if not isinstance(conditions, dict) or tuple(conditions) != EXPECTED_CONDITIONS:
        raise ValueError("mini-study conditions or their frozen order changed")
    for condition, entry in conditions.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("prompt_addition"), str):
            raise ValueError(f"mini-study prompt is invalid: {condition}")
    if conditions["l0_g0_only"]["prompt_addition"]:
        raise ValueError("L0 must not receive a prompt addition")
    if not conditions["l1_generic_salience"]["prompt_addition"].strip():
        raise ValueError("L1 generic salience instruction is missing")
    if not conditions["l4_explicit_investigation"]["prompt_addition"].strip():
        raise ValueError("L4 explicit investigation instruction is missing")

    matrix = config.get("prompt_separation_mini")
    if not isinstance(matrix, dict):
        raise ValueError("prompt-separation mini-study matrix is missing")
    if tuple(matrix.get("worlds", ())) != EXPECTED_WORLDS:
        raise ValueError("mini-study worlds changed")
    cells = matrix.get("replication_cells")
    if cells != [EXPECTED_CELL]:
        raise ValueError("mini-study must use only development root 307 / variation 21")
    if matrix.get("registered_episode_count") != EXPECTED_EPISODE_COUNT:
        raise ValueError("mini-study must register exactly 18 episodes")
    parallelism = matrix.get("parallelism", matrix.get("shard_parallelism"))
    if parallelism != EXPECTED_PARALLELISM:
        raise ValueError("mini-study must freeze twelve-way parallelism")
    if matrix.get("one_episode_per_shard") is not True:
        raise ValueError("mini-study must isolate one episode per shard")
    if matrix.get("review_begins_only_after_all_registered_episodes_finish") is not True:
        raise ValueError("mini-study review must wait for all 18 episodes")
    _validate_shards(matrix)

    matched = config.get("matched_pre_exposure", {})
    if (
        matched.get("required") is not True
        or matched.get("identical_for_all_three_conditions_within_each_world") is not True
        or matched.get("solver_receives_prefix_actions") is not False
    ):
        raise ValueError("mini-study matched pre-exposure policy changed")
    stopping = config.get("stopping_rules", {})
    if (
        stopping.get("completed_solver_episode_may_be_retried") is not False
        or stopping.get("solver_started_episode_may_be_retried") is not False
        or stopping.get("failed_episode_may_be_repaired_or_replaced") is not False
        or stopping.get("all_registered_episodes_complete_before_review") is not True
    ):
        raise ValueError("mini-study stopping policy changed")

    bindings = config.get("source_bindings")
    if not isinstance(bindings, dict):
        raise ValueError("mini-study source bindings are missing")
    for name, expected_relative in EXPECTED_BINDING_PATHS.items():
        binding = bindings.get(name)
        if not isinstance(binding, dict):
            raise ValueError(f"mini-study source binding is missing: {name}")
        if binding.get("path") != expected_relative:
            raise ValueError(f"mini-study source binding path changed: {name}")
        source = _bound_path(config, name)
        if not source.is_file() or binding.get("sha256") != _sha256(source):
            raise ValueError(f"mini-study source binding hash mismatch: {name}")

    split = json.loads(
        _bound_path(config, "development_split").read_text(encoding="utf-8")
    )
    if split.get("spec_version") != "0.3.0" or split.get("status") != "frozen":
        raise ValueError("mini-study requires frozen development split v0.3.0")
    development = split.get("development", {})
    held_out = split.get("held_out", {})
    if (
        EXPECTED_CELL["case_root"] not in development.get("roots", ())
        or EXPECTED_CELL["variation"] not in development.get("variations", ())
    ):
        raise ValueError("mini-study cell is not registered development data")
    if (
        EXPECTED_CELL["case_root"] in held_out.get("roots", ())
        or EXPECTED_CELL["variation"] in held_out.get("variations", ())
    ):
        raise ValueError("mini-study cell intersects held-out")
    _validate_neutral_schema(_bound_path(config, "neutral_submission_schema"))
    return config


def _central_freeze_manifest(
    config: dict[str, Any],
    config_path: Path,
    audited_prompt_sha256: dict[tuple[str, str], str],
) -> dict[str, Any]:
    source_paths = {
        "study_config": config_path,
        "mini_runner": SCRIPT_PATH,
        "codex_runner": Path(inspect.getfile(frozen.CodexRunner)).resolve(),
        **{
            name: _bound_path(config, name)
            for name in EXPECTED_BINDING_PATHS
        },
    }
    matrix = config["prompt_separation_mini"]
    return {
        "schema_version": "aer.pea.gate-e-prompt-separation-mini-freeze.v1",
        "status": "frozen_for_development_execution",
        "study_version": STUDY_VERSION,
        "registered_episode_count": EXPECTED_EPISODE_COUNT,
        "parallelism": EXPECTED_PARALLELISM,
        "held_out_execution_allowed": False,
        "historical_results_may_be_rewritten": False,
        "official_leaderboard_result": False,
        "deterministic_anomaly_grader_used": False,
        "model": EXPECTED_MODEL,
        "reasoning_effort": EXPECTED_REASONING_EFFORT,
        "formal_episode_action_budget": EXPECTED_ACTION_BUDGET,
        "conditions": list(EXPECTED_CONDITIONS),
        "worlds": list(EXPECTED_WORLDS),
        "replication_cells": matrix["replication_cells"],
        "shards": matrix["shards"],
        "audited_prompt_sha256": {
            _episode_key(condition, world): value
            for (condition, world), value in sorted(audited_prompt_sha256.items())
        },
        "source_sha256": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path.resolve())}
            for name, path in source_paths.items()
        },
    }


def _build_prompt(
    service: frozen.EpisodeService,
    config: dict[str, Any],
    condition: str,
) -> str:
    prompt = f"""You are operating a greenhouse through a text-based laboratory simulator.

Commissioned task:
{service.env.taskdescription()}

Initial observation:
{service.initial["observation"]}

Use only the public client in this workspace. Common commands are:
- `python3 lab.py state`
- `python3 lab.py actions`
- `python3 lab.py objects`
- `python3 lab.py valid 'filter text'`
- `python3 lab.py act 'one simulator action'`
- `python3 lab.py batch` with one simulator action per stdin line
"""
    if service.pre_exposure_observations:
        transcript = "\n\n".join(service.pre_exposure_observations)
        prompt += (
            "\nMatched public pre-exposure observations (the same deterministic handoff "
            "is reused across study conditions; prefix actions are not provided):\n"
            f"{transcript}\n"
        )
    addition = config["conditions"][condition]["prompt_addition"]
    if addition:
        prompt += f"\nDevelopment-study instruction:\n{addition}\n"
    prompt += PROMPT_SUFFIX
    return prompt


def _require_absent(text: str, forbidden: list[str], scope: str) -> None:
    folded = text.casefold()
    leaks = [token for token in forbidden if token.casefold() in folded]
    if leaks:
        raise ValueError(f"prompt-contamination audit failed for {scope}: {leaks}")


def _audit_prompt_contamination(
    config: dict[str, Any],
) -> dict[tuple[str, str], str]:
    audit = config.get("prompt_contamination_audit", {})
    if (
        audit.get("required_before_solver_start") is not True
        or audit.get("failure_policy")
        != "abort before any Solver process starts; do not consume or replace an episode"
    ):
        raise ValueError("mini-study prompt-contamination audit policy changed")
    assertions = {
        item.get("id"): item
        for item in audit.get("assertions", [])
        if isinstance(item, dict)
    }
    if set(assertions) != {f"PCA-{index:03d}" for index in range(1, 8)}:
        raise ValueError("mini-study prompt-contamination assertions changed")

    _require_absent(
        config["conditions"]["l1_generic_salience"]["prompt_addition"],
        assertions["PCA-003"]["require_absent"],
        "L1 prompt addition",
    )
    _require_absent(
        config["conditions"]["l4_explicit_investigation"]["prompt_addition"],
        assertions["PCA-004"]["require_absent"],
        "L4 prompt addition",
    )
    schema_text = _bound_path(config, "neutral_submission_schema").read_text(
        encoding="utf-8"
    )
    _require_absent(
        schema_text,
        assertions["PCA-005"]["require_absent"],
        "neutral submission schema",
    )

    prompt_sha256: dict[tuple[str, str], str] = {}
    with tempfile.TemporaryDirectory(prefix="ape43-preflight-", dir="/tmp") as temporary:
        audit_root = Path(temporary)
        for world in EXPECTED_WORLDS:
            service = frozen.EpisodeService(
                world,
                EXPECTED_CELL["variation"],
                EXPECTED_CELL["case_root"],
                audit_root / f"{world}-trajectory.jsonl",
                audit_root / f"{world}-operator-windows.jsonl",
                EXPECTED_ACTION_BUDGET,
                matched_pre_exposure=True,
            )
            try:
                prompts = {
                    condition: _build_prompt(service, config, condition)
                    for condition in EXPECTED_CONDITIONS
                }
            finally:
                service.close()

            l0_prompt = prompts["l0_g0_only"]
            _require_absent(
                l0_prompt,
                assertions["PCA-001"]["require_absent"],
                f"shared base / {world}",
            )
            _require_absent(
                l0_prompt,
                assertions["PCA-002"]["require_absent"],
                f"L0 full prompt / {world}",
            )
            shared_prefix = l0_prompt[: -len(PROMPT_SUFFIX)]
            for condition in EXPECTED_CONDITIONS:
                addition = config["conditions"][condition]["prompt_addition"]
                block = f"\nDevelopment-study instruction:\n{addition}\n" if addition else ""
                if prompts[condition] != shared_prefix + block + PROMPT_SUFFIX:
                    raise ValueError(
                        f"condition prompt differs outside its addition: {condition}/{world}"
                    )
                prompt_sha256[(condition, world)] = hashlib.sha256(
                    prompts[condition].encode("utf-8")
                ).hexdigest()
    return prompt_sha256


def _run_id(world: str) -> str:
    return (
        f"{world}-variation-{EXPECTED_CELL['variation']:02d}-"
        f"root-{EXPECTED_CELL['case_root']:04d}-run-01"
    )


def _episode_key(condition: str, world: str) -> str:
    return f"{condition}--{_run_id(world)}"


def _hash_if_file(path: Path) -> str | None:
    return _sha256(path) if path.is_file() else None


def _materialize_transcript(artifact_dir: Path) -> Path:
    transcript_path = artifact_dir / "codex/transcript.jsonl"
    if transcript_path.exists():
        raise RuntimeError(f"refusing to overwrite transcript: {transcript_path}")
    normalized = artifact_dir / "codex/normalized_events.jsonl"
    source = normalized if normalized.is_file() else artifact_dir / "codex/events.jsonl"
    if not source.is_file():
        raise FileNotFoundError("solver transcript source is missing")
    shutil.copy2(source, transcript_path)
    return transcript_path


def _run_episode(
    output_root: Path,
    config: dict[str, Any],
    config_path: Path,
    central_freeze_path: Path,
    condition: str,
    world: str,
    timeout_seconds: int,
    expected_prompt_sha256: str,
) -> dict[str, Any]:
    run_id = _run_id(world)
    episode_key = _episode_key(condition, world)
    artifact_dir = output_root / "episodes" / condition / run_id
    state_path = output_root / "_episode_states" / f"{episode_key}.json"
    state = {
        "schema_version": "aer.pea.gate-e-prompt-separation-mini-episode-state.v1",
        "study_version": STUDY_VERSION,
        "episode_key": episode_key,
        "condition": condition,
        "world": world,
        "cell": EXPECTED_CELL,
        "run_id": run_id,
        "phase": "registered",
        "attempt": 1,
        "solver_started": False,
    }
    if state_path.exists() or artifact_dir.exists():
        raise RuntimeError(f"refusing to reuse mini-study episode: {episode_key}")
    _safe_write_json(state_path, state)
    artifact_dir.mkdir(parents=True)

    service: frozen.EpisodeService | None = None
    server: frozen._UnixServer | None = None
    server_thread: threading.Thread | None = None
    result: Any = None
    started = time.time()
    hidden_summary: Any = None
    hidden_events: Any = None
    hidden_reproduction_events: Any = None
    try:
        with tempfile.TemporaryDirectory(prefix="ape43-", dir="/tmp") as temporary:
            workspace = Path(temporary)
            lab_path = workspace / "lab.py"
            schema_path = workspace / "submission.schema.json"
            shutil.copy2(_bound_path(config, "public_lab_client"), lab_path)
            shutil.copy2(_bound_path(config, "neutral_submission_schema"), schema_path)
            socket_path = workspace / "scienceworld.sock"
            trajectory_path = artifact_dir / "public_environment_trajectory.jsonl"
            operator_window_path = artifact_dir / "operator_action_windows.jsonl"
            service = frozen.EpisodeService(
                world,
                EXPECTED_CELL["variation"],
                EXPECTED_CELL["case_root"],
                trajectory_path,
                operator_window_path,
                EXPECTED_ACTION_BUDGET,
                matched_pre_exposure=True,
            )
            server = frozen._UnixServer(str(socket_path), frozen._Handler)
            server.episode = service  # type: ignore[attr-defined]
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            visible_names = {path.name for path in workspace.iterdir()}
            if visible_names != {"lab.py", "submission.schema.json", "scienceworld.sock"}:
                raise RuntimeError(
                    f"unexpected solver-visible workspace entries: {sorted(visible_names)}"
                )
            prompt = _build_prompt(service, config, condition)
            if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != expected_prompt_sha256:
                raise RuntimeError("assembled live prompt differs from the audited prompt")
            prompt_path = artifact_dir / "prompt.txt"
            prompt_path.write_text(prompt, encoding="utf-8")
            if prompt_path.read_text(encoding="utf-8") != prompt:
                raise RuntimeError("saved prompt differs from the submitted prompt")
            run_config = frozen.CodexRunConfig(
                workspace=workspace,
                artifact_dir=artifact_dir / "codex",
                prompt=prompt,
                output_schema=schema_path,
                model=EXPECTED_MODEL,
                reasoning_effort=EXPECTED_REASONING_EFFORT,
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
            state["phase"] = "solver_started"
            state["solver_started"] = True
            state["solver_started_at_unix"] = time.time()
            _safe_write_json(state_path, state)
            result = frozen.CodexRunner().run(run_config)
            state["phase"] = "solver_finished"
            state["solver_finished_at_unix"] = time.time()
            state["thread_id"] = result.thread_id
            state["usage"] = result.usage
            _safe_write_json(state_path, state)
            hidden_summary = service.env.get_aer_pea_case_summary()
            hidden_events = service.env.get_aer_pea_case_events()
            hidden_reproduction_events = service.env.get_aer_pea_case_reproduction_events()
    except Exception as error:
        state["phase"] = (
            "post_solver_failed" if state["solver_started"] else "pre_solver_failed"
        )
        state["error_type"] = type(error).__name__
        state["error"] = str(error)
        state["finished_at_unix"] = time.time()
        _safe_write_json(state_path, state)
        return {
            "episode_key": episode_key,
            "run_id": run_id,
            "condition": condition,
            "world": world,
            "status": "runner_exception",
            "succeeded": False,
            "environment_completed": False,
            "artifact_dir": str(artifact_dir),
            "episode_state_path": str(state_path),
            "episode_state_phase": state["phase"],
            "errors": [{"type": type(error).__name__, "message": str(error)}],
        }
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if server_thread is not None:
            server_thread.join(timeout=5)
        if service is not None:
            service.close()

    if result is None or service is None:
        raise RuntimeError("mini-study runner reached an impossible empty result")
    trajectory_path = artifact_dir / "public_environment_trajectory.jsonl"
    operator_window_path = artifact_dir / "operator_action_windows.jsonl"
    normalized_path = artifact_dir / "codex/normalized_events.jsonl"
    if result.events_path.is_file():
        frozen.write_normalized(
            normalized_path,
            frozen.normalize_events(frozen.read_jsonl(result.events_path)),
        )
    transcript_path = _materialize_transcript(artifact_dir)
    frozen._safe_write_json(artifact_dir / "hidden_summary.json", hidden_summary)
    frozen._safe_write_json(artifact_dir / "hidden_events.json", hidden_events)
    frozen._safe_write_json(
        artifact_dir / "hidden_reproduction_events.json",
        hidden_reproduction_events,
    )
    metadata_path = artifact_dir / "run_metadata.json"
    files_sha256 = {
        "prompt.txt": _sha256(artifact_dir / "prompt.txt"),
        "public_environment_trajectory.jsonl": _sha256(trajectory_path),
        "operator_action_windows.jsonl": _sha256(operator_window_path),
        "hidden_summary.json": _sha256(artifact_dir / "hidden_summary.json"),
        "hidden_events.json": _sha256(artifact_dir / "hidden_events.json"),
        "hidden_reproduction_events.json": _sha256(
            artifact_dir / "hidden_reproduction_events.json"
        ),
        "codex/events.jsonl": _hash_if_file(result.events_path),
        "codex/normalized_events.jsonl": _hash_if_file(normalized_path),
        "codex/transcript.jsonl": _sha256(transcript_path),
        "codex/final.json": _hash_if_file(result.final_output_path),
        "codex/stderr.log": _hash_if_file(result.stderr_path),
        "codex/manifest.json": _hash_if_file(result.manifest_path),
    }
    metadata = {
        "schema_version": "aer.pea.gate-e-prompt-separation-mini-run.v1",
        "study_version": STUDY_VERSION,
        "episode_key": episode_key,
        "run_id": run_id,
        "condition": condition,
        "assistance_coordinates": config["conditions"][condition].get(
            "assistance_coordinates", []
        ),
        "world": world,
        "repetition": EXPECTED_CELL["repetition"],
        "case_root": EXPECTED_CELL["case_root"],
        "variation": EXPECTED_CELL["variation"],
        "model": EXPECTED_MODEL,
        "reasoning_effort": EXPECTED_REASONING_EFFORT,
        "step_limit": EXPECTED_ACTION_BUDGET,
        "matched_pre_exposure": True,
        "matched_pre_exposure_observation_count": len(
            service.pre_exposure_observations
        ),
        "deterministic_anomaly_grader_used": False,
        "study_config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "central_freeze": {
            "path": str(central_freeze_path),
            "sha256": _sha256(central_freeze_path),
        },
        "mini_runner_sha256": _sha256(SCRIPT_PATH),
        "neutral_submission_schema_sha256": _sha256(
            _bound_path(config, "neutral_submission_schema")
        ),
        "lab_client_sha256": _sha256(_bound_path(config, "public_lab_client")),
        "scienceworld_jar_sha256": _sha256(_bound_path(config, "scienceworld_jar")),
        "codex_version": result.codex_version,
        "thread_id": result.thread_id,
        "status": result.status,
        "returncode": result.returncode,
        "errors": result.errors,
        "usage": result.usage,
        "started_at_unix": started,
        "finished_at_unix": time.time(),
        "environment_completed": service.completed,
        "files_sha256": files_sha256,
    }
    _safe_write_json(metadata_path, metadata)
    state["phase"] = "finalized"
    state["finished_at_unix"] = time.time()
    state["metadata_sha256"] = _sha256(metadata_path)
    state["outcome"] = {
        "status": result.status,
        "succeeded": result.succeeded,
        "environment_completed": service.completed,
    }
    _safe_write_json(state_path, state)
    return {
        "episode_key": episode_key,
        "run_id": run_id,
        "condition": condition,
        "world": world,
        "status": result.status,
        "succeeded": result.succeeded,
        "environment_completed": service.completed,
        "artifact_dir": str(artifact_dir),
        "metadata_sha256": _sha256(metadata_path),
        "episode_state_path": str(state_path),
        "episode_state_phase": "finalized",
        "errors": result.errors,
    }


def _write_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        offset = 0
        while offset < len(encoded):
            offset += os.write(descriptor, encoded[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--study-config", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=EXPECTED_PARALLELISM)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if not 1 <= args.workers <= EXPECTED_PARALLELISM:
        parser.error(f"--workers must be between 1 and {EXPECTED_PARALLELISM}")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    config_path = args.study_config.resolve()
    config = _load_study_config(config_path)
    audited_prompt_sha256 = _audit_prompt_contamination(config)
    output_root = args.output.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        parser.error(f"refusing to reuse non-empty mini-study output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    central_freeze_path = output_root / "prompt_separation_mini_freeze_manifest.json"
    _write_once(
        central_freeze_path,
        _central_freeze_manifest(config, config_path, audited_prompt_sha256),
    )

    tasks = [
        (shard["condition"], shard["world"])
        for shard in config["prompt_separation_mini"]["shards"].values()
    ]
    summary_lock = threading.Lock()
    outcomes: dict[str, dict[str, Any]] = {}

    def run_task(condition: str, world: str) -> dict[str, Any]:
        print(f"START condition={condition} world={world}", flush=True)
        try:
            outcome = _run_episode(
                output_root,
                config,
                config_path,
                central_freeze_path,
                condition,
                world,
                args.timeout,
                audited_prompt_sha256[(condition, world)],
            )
        except Exception as error:
            episode_key = _episode_key(condition, world)
            state_path = output_root / "_episode_states" / f"{episode_key}.json"
            state: dict[str, Any] = {}
            if state_path.is_file():
                state = json.loads(state_path.read_text(encoding="utf-8"))
            failure_phase = (
                "post_solver_finalization_failed"
                if state.get("solver_started") is True
                else "pre_solver_failed"
            )
            state.update(
                {
                    "schema_version": (
                        "aer.pea.gate-e-prompt-separation-mini-episode-state.v1"
                    ),
                    "study_version": STUDY_VERSION,
                    "episode_key": episode_key,
                    "condition": condition,
                    "world": world,
                    "cell": EXPECTED_CELL,
                    "run_id": _run_id(world),
                    "phase": failure_phase,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "finished_at_unix": time.time(),
                }
            )
            _safe_write_json(state_path, state)
            outcome = {
                "episode_key": episode_key,
                "run_id": _run_id(world),
                "condition": condition,
                "world": world,
                "status": "runner_exception",
                "succeeded": False,
                "environment_completed": False,
                "artifact_dir": str(
                    output_root / "episodes" / condition / _run_id(world)
                ),
                "episode_state_path": str(state_path),
                "episode_state_phase": state["phase"],
                "errors": [
                    {"type": type(error).__name__, "message": str(error)}
                ],
            }
        with summary_lock:
            outcomes[outcome["episode_key"]] = outcome
            _safe_write_json(
                output_root / "batch_summary.json",
                [outcomes[key] for key in sorted(outcomes)],
            )
        print(
            f"DONE condition={condition} world={world} status={outcome['status']} "
            f"phase={outcome['episode_state_phase']}",
            flush=True,
        )
        return outcome

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_task, condition, world) for condition, world in tasks]
        for future in concurrent.futures.as_completed(futures):
            # _run_episode converts episode-local exceptions into immutable failed outcomes.
            future.result()

    final_outcomes = [outcomes[key] for key in sorted(outcomes)]
    if len(final_outcomes) != EXPECTED_EPISODE_COUNT:
        raise RuntimeError("mini-study did not produce all 18 registered outcomes")
    return 0 if all(outcome["succeeded"] for outcome in final_outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
