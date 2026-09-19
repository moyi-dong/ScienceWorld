#!/usr/bin/env python3
"""Prepare and run the immutable pre-solver recovery for the v0.4.3 mini-study.

The original attempt remains immutable.  ``prepare`` registers only its seventeen
``pre_solver_failed`` episodes, freezes one canonical handoff per world, and binds every
attempt-1 artifact by SHA-256.  ``run`` rejects the already-finalized episode, replays the
canonical action sequence, checks semantic equality before starting Codex, and records attempt 2
in a separate artifact tree.  A solver-started recovery episode is never retried.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import run_aer_pea_calibration as frozen
import run_aer_pea_gate_e_prompt_separation_mini as mini

SCRIPT_PATH = Path(__file__).resolve()
RECOVERY_SCHEMA = "aer.pea.gate-e-prompt-separation-mini-recovery.v1"
RECOVERY_STATE_SCHEMA = "aer.pea.gate-e-prompt-separation-mini-recovery-state.v1"
RECOVERY_RUN_SCHEMA = "aer.pea.gate-e-prompt-separation-mini-recovery-run.v1"
COMPOSITION_SCHEMA = "aer.pea.gate-e-prompt-separation-mini-composition.v1"
MINI_STATE_SCHEMA = "aer.pea.gate-e-prompt-separation-mini-episode-state.v1"
MINI_RUN_SCHEMA = "aer.pea.gate-e-prompt-separation-mini-run.v1"
PRESERVED_EPISODE = (
    "l1_generic_salience--plant_attractiveness-variation-21-root-0307-run-01"
)
DEFAULT_CANDIDATE_LIMIT = 256


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_write_json(path: Path, value: Any) -> None:
    mini._safe_write_json(path, value)


def _write_once(path: Path, value: Any) -> None:
    encoded = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        offset = 0
        while offset < len(encoded):
            offset += os.write(descriptor, encoded[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_sha_sidecar(path: Path) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(f"{_sha256(path)}  {path.name}\n", encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _regular_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _relative_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in _regular_files(root)
    }


def _normalize_flower_order(text: str) -> str:
    """Ignore only the order of flower items within one plant description."""

    pattern = re.compile(r"(On the pea plant you see: )(.+?)(\. , soil\))")

    def normalize(match: re.Match[str]) -> str:
        items = match.group(2).split(", ")
        indices = [
            index
            for index, item in enumerate(items)
            if "flower" in item.casefold()
        ]
        ordered = sorted(items[index] for index in indices)
        for offset, index in enumerate(indices):
            items[index] = ordered[offset]
        return match.group(1) + ", ".join(items) + match.group(3)

    return pattern.sub(normalize, text)


def _semantic_payload(
    task_description: str,
    actions: list[str],
    initial_observation: str,
    pre_exposure_observations: list[str],
) -> dict[str, Any]:
    return {
        "task_description": task_description,
        "actions": actions,
        "initial_observation": _normalize_flower_order(initial_observation),
        "pre_exposure_observations": [
            _normalize_flower_order(value) for value in pre_exposure_observations
        ],
        "action_count": len(actions),
        "pre_exposure_observation_count": len(pre_exposure_observations),
    }


def _semantic_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _trajectory_handoff(path: Path) -> tuple[list[str], list[str], str]:
    events = frozen.read_jsonl(path)
    matched = [event for event in events if event.get("source") == "matched_pre_exposure"]
    actions = [str(event["request"]["action"]) for event in matched]
    observations = [
        str(event.get("response", {}).get("observation", ""))
        for event in matched
        if "Greenhouse activity since your last action:"
        in str(event.get("response", {}).get("observation", ""))
    ]
    initial_events = [event for event in events if event.get("source") == "initial"]
    if len(initial_events) != 1:
        raise ValueError(f"canonical trajectory must contain one initial event: {path}")
    initial = str(initial_events[0]["response"]["observation"])
    if not actions or not observations or not initial:
        raise ValueError(f"canonical trajectory handoff is incomplete: {path}")
    return actions, observations, initial


def _common_prompt_from_condition(
    prompt: str, config: dict[str, Any], condition: str
) -> str:
    addition = config["conditions"][condition]["prompt_addition"]
    block = f"\nDevelopment-study instruction:\n{addition}\n" if addition else ""
    if block:
        if prompt.count(block) != 1:
            raise ValueError("condition addition is not unique in canonical prompt")
        prompt = prompt.replace(block, "", 1)
    if not prompt.endswith(mini.PROMPT_SUFFIX):
        raise ValueError("canonical common prompt has an unexpected suffix")
    return prompt


def _condition_prompt(
    common_prompt: str, config: dict[str, Any], condition: str
) -> str:
    if not common_prompt.endswith(mini.PROMPT_SUFFIX):
        raise ValueError("frozen common prompt suffix mismatch")
    prefix = common_prompt[: -len(mini.PROMPT_SUFFIX)]
    addition = config["conditions"][condition]["prompt_addition"]
    block = f"\nDevelopment-study instruction:\n{addition}\n" if addition else ""
    return prefix + block + mini.PROMPT_SUFFIX


def _extract_task_description(prompt: str) -> str:
    marker = "Commissioned task:\n"
    end = "\n\nInitial observation:"
    if marker not in prompt or end not in prompt:
        raise ValueError("canonical prompt is missing its task section")
    return prompt.split(marker, 1)[1].split(end, 1)[0]


def _original_inventory(
    original_root: Path, config_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    freeze_path = original_root / "prompt_separation_mini_freeze_manifest.json"
    batch_path = original_root / "batch_summary.json"
    if not freeze_path.is_file() or not batch_path.is_file():
        raise FileNotFoundError("original mini-study freeze or batch summary is missing")
    freeze = _load_json(freeze_path)
    if freeze.get("study_version") != mini.STUDY_VERSION:
        raise ValueError("original mini-study version mismatch")
    if freeze.get("registered_episode_count") != mini.EXPECTED_EPISODE_COUNT:
        raise ValueError("original mini-study registration count mismatch")
    config_sha = _sha256(config_path)
    config_bindings = [
        value
        for name, value in freeze.get("source_sha256", {}).items()
        if name == "study_config" and isinstance(value, dict)
    ]
    if not config_bindings or config_bindings[0].get("sha256") != config_sha:
        raise ValueError("original mini-study config binding mismatch")
    for name, binding in freeze.get("source_sha256", {}).items():
        path = Path(binding["path"])
        if not path.is_file() or _sha256(path) != binding["sha256"]:
            raise ValueError(f"original freeze source hash mismatch: {name}")

    state_paths = sorted((original_root / "_episode_states").glob("*.json"))
    if len(state_paths) != mini.EXPECTED_EPISODE_COUNT:
        raise ValueError("original mini-study must contain exactly 18 states")
    states = [_load_json(path) for path in state_paths]
    keys = [state.get("episode_key") for state in states]
    if len(set(keys)) != mini.EXPECTED_EPISODE_COUNT:
        raise ValueError("original mini-study state keys are not unique")
    finalized = [state for state in states if state.get("phase") == "finalized"]
    if len(finalized) != 1 or finalized[0].get("episode_key") != PRESERVED_EPISODE:
        raise ValueError("recovery must preserve exactly the registered finalized episode")
    if finalized[0].get("solver_started") is not True:
        raise ValueError("preserved finalized episode lacks its solver-start record")
    retryable = [state for state in states if state.get("episode_key") != PRESERVED_EPISODE]
    for state in retryable:
        if (
            state.get("phase") != "pre_solver_failed"
            or state.get("solver_started") is not False
            or state.get("attempt") != 1
            or state.get("error") != "assembled live prompt differs from the audited prompt"
        ):
            raise ValueError(f"original state is not retryable: {state.get('episode_key')}")
    return freeze, states


def _canonical_package(
    world: str,
    condition: str,
    prompt: str,
    trajectory_path: Path,
    source: str,
) -> dict[str, Any]:
    actions, observations, initial = _trajectory_handoff(trajectory_path)
    common_prompt = _common_prompt_from_condition(prompt, _ACTIVE_CONFIG, condition)
    task = _extract_task_description(common_prompt)
    semantic = _semantic_payload(task, actions, initial, observations)
    return {
        "schema_version": "aer.pea.gate-e-canonical-handoff.v1",
        "study_version": mini.STUDY_VERSION,
        "world": world,
        "case_root": mini.EXPECTED_CELL["case_root"],
        "variation": mini.EXPECTED_CELL["variation"],
        "source": source,
        "source_condition": condition,
        "actions": actions,
        "task_description": task,
        "initial_observation": initial,
        "pre_exposure_observations": observations,
        "common_prompt": common_prompt,
        "common_prompt_sha256": _sha256_text(common_prompt),
        "semantic_payload": semantic,
        "semantic_sha256": _semantic_sha256(semantic),
        "normalization": "sort_only_flower_items_within_each_pea_plant_description",
    }


_ACTIVE_CONFIG: dict[str, Any]


def prepare(
    original_root: Path,
    output_root: Path,
    config_path: Path,
    candidate_limit: int,
) -> dict[str, Any]:
    global _ACTIVE_CONFIG
    config = mini._load_study_config(config_path)
    _ACTIVE_CONFIG = config
    original_freeze, states = _original_inventory(original_root, config_path)
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing to reuse non-empty recovery output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    preserved_state = next(
        state for state in states if state["episode_key"] == PRESERVED_EPISODE
    )
    preserved_dir = (
        original_root
        / "episodes"
        / preserved_state["condition"]
        / preserved_state["run_id"]
    )
    plant_trajectory = preserved_dir / "public_environment_trajectory.jsonl"
    plant_prompt = (preserved_dir / "prompt.txt").read_text(encoding="utf-8")
    packages: dict[str, dict[str, Any]] = {
        "plant_attractiveness": _canonical_package(
            "plant_attractiveness",
            "l1_generic_salience",
            plant_prompt,
            plant_trajectory,
            "preserved_finalized_attempt_1",
        )
    }

    with tempfile.TemporaryDirectory(prefix="ape43-recovery-prepare-", dir="/tmp") as temp:
        temporary = Path(temp)
        for world in mini.EXPECTED_WORLDS:
            if world == "plant_attractiveness":
                continue
            trajectory = temporary / world / "trajectory.jsonl"
            operator = temporary / world / "operator.jsonl"
            service = frozen.EpisodeService(
                world,
                mini.EXPECTED_CELL["variation"],
                mini.EXPECTED_CELL["case_root"],
                trajectory,
                operator,
                mini.EXPECTED_ACTION_BUDGET,
                matched_pre_exposure=True,
            )
            try:
                prompt = mini._build_prompt(service, config, "l0_g0_only")
            finally:
                service.close()
            packages[world] = _canonical_package(
                world,
                "l0_g0_only",
                prompt,
                trajectory,
                "fresh_pre_solver_canonical_generation",
            )

    handoff_hashes: dict[str, str] = {}
    for world in mini.EXPECTED_WORLDS:
        path = output_root / "canonical_handoffs" / f"{world}.json"
        _write_once(path, packages[world])
        _write_sha_sidecar(path)
        handoff_hashes[world] = _sha256(path)

    retryable = [state for state in states if state["episode_key"] != PRESERVED_EPISODE]
    original_files = _relative_hashes(original_root)
    registration = {
        "schema_version": RECOVERY_SCHEMA,
        "status": "prepared_for_registered_pre_solver_attempt_2",
        "study_version": mini.STUDY_VERSION,
        "attempt": 2,
        "registered_episode_count": len(retryable),
        "workers": mini.EXPECTED_PARALLELISM,
        "candidate_limit_per_world": candidate_limit,
        "solver_started_episode_retry_allowed": False,
        "held_out_execution_allowed": False,
        "deterministic_anomaly_grader_used": False,
        "preserved_finalized_episode": PRESERVED_EPISODE,
        "preserved_finalized_state_sha256": _sha256(
            original_root / "_episode_states" / f"{PRESERVED_EPISODE}.json"
        ),
        "registered_episode_keys": sorted(state["episode_key"] for state in retryable),
        "prior_state_sha256": {
            state["episode_key"]: _sha256(
                original_root / "_episode_states" / f"{state['episode_key']}.json"
            )
            for state in states
        },
        "original_root": str(original_root),
        "original_freeze_sha256": _sha256(
            original_root / "prompt_separation_mini_freeze_manifest.json"
        ),
        "original_batch_summary_sha256": _sha256(original_root / "batch_summary.json"),
        "original_file_sha256": original_files,
        "study_config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "neutral_schema": {
            "path": str(mini._bound_path(config, "neutral_submission_schema")),
            "sha256": _sha256(mini._bound_path(config, "neutral_submission_schema")),
        },
        "original_runner_sha256": original_freeze["source_sha256"]["mini_runner"][
            "sha256"
        ],
        "recovery_runner": {"path": str(SCRIPT_PATH), "sha256": _sha256(SCRIPT_PATH)},
        "canonical_handoff_sha256": handoff_hashes,
        "canonical_policy": {
            "one_action_sequence_and_common_prompt_per_world": True,
            "condition_prompt_only_differs_by_frozen_addition": True,
            "semantic_replay_required_before_solver": True,
            "allowed_semantic_normalization": (
                "same-plant flower item order only; pot identity, plant identity, actions, "
                "event count and event timing remain exact"
            ),
        },
    }
    registration_path = output_root / "recovery_registration.json"
    _write_once(registration_path, registration)
    _write_sha_sidecar(registration_path)
    return registration


class CanonicalReplayService(frozen.EpisodeService):
    """One JVM that can reset and replay an operator-frozen handoff."""

    def __init__(
        self,
        world: str,
        actions: list[str],
        trajectory_path: Path,
        operator_window_path: Path,
    ) -> None:
        self.world = world
        self.actions = actions
        self.env = frozen.ScienceWorldEnv(
            "", serverPath=None, envStepLimit=mini.EXPECTED_ACTION_BUDGET
        )
        self.env.configure_aer_pea_case(world, mini.EXPECTED_CELL["case_root"])
        self.reset_replay(trajectory_path, operator_window_path)

    def reset_replay(
        self, trajectory_path: Path, operator_window_path: Path
    ) -> None:
        self.env.load(
            frozen.TASK,
            mini.EXPECTED_CELL["variation"],
            "easy",
            generateGoldPath=False,
        )
        self.trajectory_path = trajectory_path
        self.operator_window_path = operator_window_path
        if trajectory_path.exists() or operator_window_path.exists():
            raise RuntimeError("canonical replay refuses to overwrite evidence")
        self._lock = threading.Lock()
        self._index = 0
        self._note_index = 0
        self._experiment_ids = set()
        self._active_experiment_id = None
        self.completed = False
        self.pre_exposure_observations = []
        for action in self.actions:
            response = self._step(action, source="matched_pre_exposure")
            observation = str(response.get("observation", ""))
            if "Greenhouse activity since your last action:" in observation:
                self.pre_exposure_observations.append(observation)
        self.initial = self._step("look around", source="initial")


def _service_semantic(
    service: CanonicalReplayService, package: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    payload = _semantic_payload(
        service.env.taskdescription(),
        service.actions,
        str(service.initial["observation"]),
        service.pre_exposure_observations,
    )
    return payload, _semantic_sha256(payload)


def _validate_prepared(
    original_root: Path, output_root: Path, config_path: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    registration_path = output_root / "recovery_registration.json"
    registration = _load_json(registration_path)
    if registration.get("schema_version") != RECOVERY_SCHEMA:
        raise ValueError("unsupported recovery registration")
    if registration.get("status") != "prepared_for_registered_pre_solver_attempt_2":
        raise ValueError("recovery registration is not prepared")
    if registration.get("recovery_runner", {}).get("sha256") != _sha256(SCRIPT_PATH):
        raise ValueError("recovery runner changed after prepare")
    if registration.get("study_config", {}).get("sha256") != _sha256(config_path):
        raise ValueError("recovery study config changed after prepare")
    current_original = _relative_hashes(original_root)
    if current_original != registration.get("original_file_sha256"):
        raise ValueError("original attempt-1 tree changed after recovery prepare")
    _, states = _original_inventory(original_root, config_path)
    retryable = [state for state in states if state["episode_key"] != PRESERVED_EPISODE]
    if sorted(state["episode_key"] for state in retryable) != registration.get(
        "registered_episode_keys"
    ):
        raise ValueError("recovery registration keys changed")
    packages: dict[str, dict[str, Any]] = {}
    for world, expected in registration["canonical_handoff_sha256"].items():
        path = output_root / "canonical_handoffs" / f"{world}.json"
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"canonical handoff hash mismatch: {world}")
        packages[world] = _load_json(path)
        if packages[world].get("semantic_sha256") != _semantic_sha256(
            packages[world]["semantic_payload"]
        ):
            raise ValueError(f"canonical semantic signature mismatch: {world}")
    return registration, packages, {state["episode_key"]: state for state in retryable}


def _hash_if_file(path: Path) -> str | None:
    return _sha256(path) if path.is_file() else None


def _require_finalized_source(
    state_path: Path,
    artifact_dir: Path,
    expected_state_schema: str,
    expected_run_schema: str,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    state = _load_json(state_path)
    if state.get("schema_version") != expected_state_schema:
        raise ValueError(f"unexpected source state schema: {state_path}")
    if state.get("phase") != "finalized" or state.get("solver_started") is not True:
        raise ValueError(f"source state is not finalized: {state_path}")
    metadata_path = artifact_dir / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"source metadata is missing: {metadata_path}")
    if state.get("metadata_sha256") != _sha256(metadata_path):
        raise ValueError(f"source state metadata hash mismatch: {state_path}")
    metadata = _load_json(metadata_path)
    if metadata.get("schema_version") != expected_run_schema:
        raise ValueError(f"unexpected source metadata schema: {metadata_path}")
    if metadata.get("status") not in {"completed", "finalized"}:
        raise ValueError(f"source run is incomplete: {metadata_path}")
    if metadata.get("environment_completed") is not True:
        raise ValueError(f"source environment did not complete: {metadata_path}")
    return state, metadata, metadata_path


def _move_replay_evidence(
    service: CanonicalReplayService, artifact_dir: Path
) -> None:
    trajectory = artifact_dir / "public_environment_trajectory.jsonl"
    operator = artifact_dir / "operator_action_windows.jsonl"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    shutil.move(str(service.trajectory_path), str(trajectory))
    shutil.move(str(service.operator_window_path), str(operator))
    service.trajectory_path = trajectory
    service.operator_window_path = operator


def _run_solver_on_service(
    service: CanonicalReplayService,
    package: dict[str, Any],
    condition: str,
    world: str,
    output_root: Path,
    config: dict[str, Any],
    config_path: Path,
    registration_path: Path,
    state_path: Path,
    state: dict[str, Any],
    artifact_dir: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    prompt = _condition_prompt(package["common_prompt"], config, condition)
    expected_common = _common_prompt_from_condition(prompt, config, condition)
    if expected_common != package["common_prompt"]:
        raise RuntimeError("condition prompt differs outside its frozen addition")
    _move_replay_evidence(service, artifact_dir)
    prompt_path = artifact_dir / "prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="ape43-recovery-live-", dir="/tmp") as temp:
        workspace = Path(temp)
        schema_path = workspace / "submission.schema.json"
        shutil.copy2(mini._bound_path(config, "public_lab_client"), workspace / "lab.py")
        shutil.copy2(mini._bound_path(config, "neutral_submission_schema"), schema_path)
        socket_path = workspace / "scienceworld.sock"
        server = frozen._UnixServer(str(socket_path), frozen._Handler)
        server.episode = service  # type: ignore[attr-defined]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            visible = {path.name for path in workspace.iterdir()}
            if visible != {"lab.py", "submission.schema.json", "scienceworld.sock"}:
                raise RuntimeError(f"unexpected recovery workspace: {sorted(visible)}")
            run_config = frozen.CodexRunConfig(
                workspace=workspace,
                artifact_dir=artifact_dir / "codex",
                prompt=prompt,
                output_schema=schema_path,
                model=mini.EXPECTED_MODEL,
                reasoning_effort=mini.EXPECTED_REASONING_EFFORT,
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
            hidden_summary = service.env.get_aer_pea_case_summary()
            hidden_events = service.env.get_aer_pea_case_events()
            hidden_reproduction = service.env.get_aer_pea_case_reproduction_events()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    normalized_path = artifact_dir / "codex/normalized_events.jsonl"
    if result.events_path.is_file():
        frozen.write_normalized(
            normalized_path,
            frozen.normalize_events(frozen.read_jsonl(result.events_path)),
        )
    transcript = mini._materialize_transcript(artifact_dir)
    frozen._safe_write_json(artifact_dir / "hidden_summary.json", hidden_summary)
    frozen._safe_write_json(artifact_dir / "hidden_events.json", hidden_events)
    frozen._safe_write_json(
        artifact_dir / "hidden_reproduction_events.json", hidden_reproduction
    )
    files = {
        "prompt.txt": _sha256(prompt_path),
        "public_environment_trajectory.jsonl": _sha256(service.trajectory_path),
        "operator_action_windows.jsonl": _sha256(service.operator_window_path),
        "hidden_summary.json": _sha256(artifact_dir / "hidden_summary.json"),
        "hidden_events.json": _sha256(artifact_dir / "hidden_events.json"),
        "hidden_reproduction_events.json": _sha256(
            artifact_dir / "hidden_reproduction_events.json"
        ),
        "codex/events.jsonl": _hash_if_file(result.events_path),
        "codex/normalized_events.jsonl": _hash_if_file(normalized_path),
        "codex/transcript.jsonl": _sha256(transcript),
        "codex/final.json": _hash_if_file(result.final_output_path),
        "codex/stderr.log": _hash_if_file(result.stderr_path),
        "codex/manifest.json": _hash_if_file(result.manifest_path),
    }
    metadata = {
        "schema_version": RECOVERY_RUN_SCHEMA,
        "study_version": mini.STUDY_VERSION,
        "episode_key": state["episode_key"],
        "attempt": 2,
        "run_id": state["run_id"],
        "condition": condition,
        "world": world,
        "repetition": mini.EXPECTED_CELL["repetition"],
        "case_root": mini.EXPECTED_CELL["case_root"],
        "variation": mini.EXPECTED_CELL["variation"],
        "model": mini.EXPECTED_MODEL,
        "reasoning_effort": mini.EXPECTED_REASONING_EFFORT,
        "step_limit": mini.EXPECTED_ACTION_BUDGET,
        "canonical_handoff_sha256": _sha256(
            output_root / "canonical_handoffs" / f"{world}.json"
        ),
        "semantic_sha256": package["semantic_sha256"],
        "prior_attempt_state_sha256": state["prior_attempt_state_sha256"],
        "study_config_sha256": _sha256(config_path),
        "recovery_registration_sha256": _sha256(registration_path),
        "recovery_runner_sha256": _sha256(SCRIPT_PATH),
        "deterministic_anomaly_grader_used": False,
        "codex_version": result.codex_version,
        "thread_id": result.thread_id,
        "status": result.status,
        "returncode": result.returncode,
        "errors": result.errors,
        "usage": result.usage,
        "environment_completed": service.completed,
        "files_sha256": files,
    }
    metadata_path = artifact_dir / "run_metadata.json"
    _safe_write_json(metadata_path, metadata)
    state["phase"] = "finalized"
    state["metadata_sha256"] = _sha256(metadata_path)
    state["outcome"] = {
        "status": result.status,
        "succeeded": result.succeeded,
        "environment_completed": service.completed,
    }
    state["finished_at_unix"] = time.time()
    _safe_write_json(state_path, state)
    return {
        "episode_key": state["episode_key"],
        "condition": condition,
        "world": world,
        "status": result.status,
        "succeeded": result.succeeded,
        "environment_completed": service.completed,
        "artifact_dir": str(artifact_dir),
        "state_path": str(state_path),
        "state_phase": "finalized",
        "metadata_sha256": _sha256(metadata_path),
        "errors": result.errors,
    }


def run_recovery(
    original_root: Path,
    output_root: Path,
    config_path: Path,
    workers: int,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    registration, packages, prior_states = _validate_prepared(
        original_root, output_root, config_path
    )
    config = mini._load_study_config(config_path)
    registration_path = output_root / "recovery_registration.json"
    if (output_root / "states").exists() or (output_root / "attempt-0002").exists():
        raise RuntimeError("recovery run has already been started")
    (output_root / "states").mkdir()
    (output_root / "attempt-0002").mkdir()

    by_world: dict[str, list[str]] = {world: [] for world in mini.EXPECTED_WORLDS}
    shard_order = config["prompt_separation_mini"]["shards"].values()
    for shard in shard_order:
        key = mini._episode_key(shard["condition"], shard["world"])
        if key in prior_states:
            by_world[shard["world"]].append(shard["condition"])
    if sum(map(len, by_world.values())) != 17:
        raise RuntimeError("recovery world groups do not cover exactly 17 episodes")

    summary_lock = threading.Lock()
    outcomes: dict[str, dict[str, Any]] = {}

    def write_outcome(outcome: dict[str, Any]) -> None:
        with summary_lock:
            outcomes[outcome["episode_key"]] = outcome
            _safe_write_json(
                output_root / "recovery_batch_summary.json",
                [outcomes[key] for key in sorted(outcomes)],
            )

    def world_group(world: str, conditions: list[str]) -> list[dict[str, Any]]:
        package = packages[world]
        service: CanonicalReplayService | None = None
        candidate_records: list[dict[str, Any]] = []
        try:
            for candidate in range(1, registration["candidate_limit_per_world"] + 1):
                candidate_dir = (
                    output_root
                    / "pre_solver_candidates"
                    / world
                    / f"candidate-{candidate:03d}"
                )
                candidate_dir.mkdir(parents=True, exist_ok=False)
                current = CanonicalReplayService(
                    world,
                    package["actions"],
                    candidate_dir / "public_environment_trajectory.jsonl",
                    candidate_dir / "operator_action_windows.jsonl",
                )
                payload, semantic_sha = _service_semantic(current, package)
                record = {
                    "candidate": candidate,
                    "semantic_sha256": semantic_sha,
                    "matched": semantic_sha == package["semantic_sha256"],
                    "trajectory_sha256": _sha256(current.trajectory_path),
                    "operator_windows_sha256": _sha256(current.operator_window_path),
                }
                candidate_records.append(record)
                if record["matched"]:
                    service = current
                    break
                current.close()
            _safe_write_json(
                output_root / "pre_solver_candidates" / world / "candidate_summary.json",
                candidate_records,
            )
            if service is None:
                raise RuntimeError(f"canonical replay candidate limit exhausted: {world}")

            group_outcomes: list[dict[str, Any]] = []
            for index, condition in enumerate(conditions):
                episode_key = mini._episode_key(condition, world)
                prior = prior_states[episode_key]
                state_path = output_root / "states" / f"{episode_key}.json"
                state = {
                    "schema_version": RECOVERY_STATE_SCHEMA,
                    "study_version": mini.STUDY_VERSION,
                    "episode_key": episode_key,
                    "condition": condition,
                    "world": world,
                    "run_id": prior["run_id"],
                    "attempt": 2,
                    "phase": "canonical_replay_verified",
                    "solver_started": False,
                    "prior_attempt_phase": "pre_solver_failed",
                    "prior_attempt_solver_started": False,
                    "prior_attempt_state_sha256": registration["prior_state_sha256"][episode_key],
                    "canonical_handoff_sha256": registration["canonical_handoff_sha256"][world],
                    "candidate_count": len(candidate_records),
                }
                _write_once(state_path, state)
                if index > 0:
                    next_dir = output_root / "replay_staging" / world / condition
                    next_dir.mkdir(parents=True, exist_ok=False)
                    service.reset_replay(
                        next_dir / "public_environment_trajectory.jsonl",
                        next_dir / "operator_action_windows.jsonl",
                    )
                    _, semantic_sha = _service_semantic(service, package)
                    if semantic_sha != package["semantic_sha256"]:
                        state["phase"] = "pre_solver_semantic_replay_failed"
                        state["error"] = "same-JVM canonical replay semantic mismatch"
                        _safe_write_json(state_path, state)
                        outcome = {
                            "episode_key": episode_key,
                            "condition": condition,
                            "world": world,
                            "status": "pre_solver_failed",
                            "succeeded": False,
                            "environment_completed": False,
                            "artifact_dir": None,
                            "state_path": str(state_path),
                            "state_phase": state["phase"],
                            "errors": [{"type": "RuntimeError", "message": state["error"]}],
                        }
                        group_outcomes.append(outcome)
                        write_outcome(outcome)
                        continue
                artifact = output_root / "attempt-0002" / condition / prior["run_id"]
                try:
                    outcome = _run_solver_on_service(
                        service,
                        package,
                        condition,
                        world,
                        output_root,
                        config,
                        config_path,
                        registration_path,
                        state_path,
                        state,
                        artifact,
                        timeout_seconds,
                    )
                except Exception as error:
                    state["phase"] = (
                        "post_solver_failed"
                        if state.get("solver_started") is True
                        else "pre_solver_failed"
                    )
                    state["error_type"] = type(error).__name__
                    state["error"] = str(error)
                    _safe_write_json(state_path, state)
                    outcome = {
                        "episode_key": episode_key,
                        "condition": condition,
                        "world": world,
                        "status": "runner_exception",
                        "succeeded": False,
                        "environment_completed": False,
                        "artifact_dir": str(artifact),
                        "state_path": str(state_path),
                        "state_phase": state["phase"],
                        "errors": [{"type": type(error).__name__, "message": str(error)}],
                    }
                group_outcomes.append(outcome)
                write_outcome(outcome)
                if state.get("solver_started") is True and state.get("phase") != "finalized":
                    # No later condition may reuse uncertain post-solver infrastructure.
                    break
            return group_outcomes
        finally:
            if service is not None:
                service.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(world_group, world, conditions)
            for world, conditions in by_world.items()
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    values = [outcomes[key] for key in sorted(outcomes)]
    if len(values) == 17 and all(
        value["succeeded"] and value["environment_completed"] for value in values
    ):
        preserved_state = _load_json(
            original_root / "_episode_states" / f"{PRESERVED_EPISODE}.json"
        )
        preserved_artifact = (
            original_root
            / "episodes"
            / preserved_state["condition"]
            / preserved_state["run_id"]
        )
        composition = {
            "schema_version": COMPOSITION_SCHEMA,
            "study_version": mini.STUDY_VERSION,
            "status": "recovery_sources_ready_for_compose",
            "episode_count": 18,
            "preserved_attempt_1": {
                "episode_key": PRESERVED_EPISODE,
                "root": str(original_root),
                "artifact_dir": str(preserved_artifact),
                "metadata_path": str(preserved_artifact / "run_metadata.json"),
                "metadata_sha256": _sha256(preserved_artifact / "run_metadata.json"),
                "state_path": str(
                    original_root / "_episode_states" / f"{PRESERVED_EPISODE}.json"
                ),
                "state_sha256": registration["preserved_finalized_state_sha256"],
            },
            "recovery_attempt_2_root": str(output_root / "attempt-0002"),
            "recovery_batch_summary_sha256": _sha256(
                output_root / "recovery_batch_summary.json"
            ),
            "recovery_metadata_sha256": {
                value["episode_key"]: value["metadata_sha256"] for value in values
            },
            "recovery_episodes": [
                {
                    "episode_key": value["episode_key"],
                    "artifact_dir": value["artifact_dir"],
                    "metadata_path": str(Path(value["artifact_dir"]) / "run_metadata.json"),
                    "metadata_sha256": value["metadata_sha256"],
                    "state_path": value["state_path"],
                    "state_sha256": _sha256(Path(value["state_path"])),
                }
                for value in values
            ],
        }
        _safe_write_json(output_root / "composition_manifest.json", composition)
    return values


def compose_recovery(
    original_root: Path,
    output_root: Path,
    config_path: Path,
    composed_root: Path,
) -> dict[str, Any]:
    """Create one builder-compatible, immutable 18-episode evidence root."""

    registration, _, prior_states = _validate_prepared(
        original_root, output_root, config_path
    )
    source_manifest_path = output_root / "composition_manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError("successful recovery composition manifest is missing")
    source_manifest = _load_json(source_manifest_path)
    if source_manifest.get("schema_version") != COMPOSITION_SCHEMA:
        raise ValueError("unsupported recovery composition manifest")
    if source_manifest.get("status") != "recovery_sources_ready_for_compose":
        raise ValueError("recovery sources are not ready to compose")
    if source_manifest.get("episode_count") != mini.EXPECTED_EPISODE_COUNT:
        raise ValueError("recovery composition manifest must register 18 episodes")
    if composed_root.exists():
        raise FileExistsError(f"refusing to overwrite composed root: {composed_root}")

    prepared: list[dict[str, Any]] = []
    original_state_path = (
        original_root / "_episode_states" / f"{PRESERVED_EPISODE}.json"
    )
    original_state = _load_json(original_state_path)
    original_artifact = (
        original_root
        / "episodes"
        / original_state["condition"]
        / original_state["run_id"]
    )
    state, metadata, metadata_path = _require_finalized_source(
        original_state_path,
        original_artifact,
        MINI_STATE_SCHEMA,
        MINI_RUN_SCHEMA,
    )
    prepared.append(
        {
            "source_kind": "preserved_attempt_1",
            "source_state_path": original_state_path,
            "source_artifact_dir": original_artifact,
            "source_metadata_path": metadata_path,
            "state": state,
            "metadata": metadata,
        }
    )

    for episode_key in registration["registered_episode_keys"]:
        prior = prior_states[episode_key]
        state_path = output_root / "states" / f"{episode_key}.json"
        artifact_dir = (
            output_root / "attempt-0002" / prior["condition"] / prior["run_id"]
        )
        state, metadata, metadata_path = _require_finalized_source(
            state_path,
            artifact_dir,
            RECOVERY_STATE_SCHEMA,
            RECOVERY_RUN_SCHEMA,
        )
        if state.get("episode_key") != episode_key:
            raise ValueError(f"recovery state episode key mismatch: {state_path}")
        prepared.append(
            {
                "source_kind": "registered_pre_solver_attempt_2",
                "source_state_path": state_path,
                "source_artifact_dir": artifact_dir,
                "source_metadata_path": metadata_path,
                "state": state,
                "metadata": metadata,
            }
        )

    if len(prepared) != mini.EXPECTED_EPISODE_COUNT:
        raise RuntimeError("compose did not validate exactly 18 finalized sources")
    episode_keys = [item["state"]["episode_key"] for item in prepared]
    if len(set(episode_keys)) != mini.EXPECTED_EPISODE_COUNT:
        raise RuntimeError("compose source episode keys are not unique")

    composed_root.mkdir(parents=True)
    (composed_root / "_episode_states").mkdir()
    records: list[dict[str, Any]] = []
    for item in prepared:
        source_state_path = item["source_state_path"]
        source_artifact = item["source_artifact_dir"]
        source_metadata_path = item["source_metadata_path"]
        state = item["state"]
        metadata = item["metadata"]
        destination_artifact = (
            composed_root / "episodes" / state["condition"] / state["run_id"]
        )
        shutil.copytree(source_artifact, destination_artifact)
        destination_metadata_path = destination_artifact / "run_metadata.json"
        if item["source_kind"] == "registered_pre_solver_attempt_2":
            metadata = dict(metadata)
            metadata["source_schema_version"] = metadata["schema_version"]
            metadata["schema_version"] = MINI_RUN_SCHEMA
            metadata["composition_source"] = {
                "kind": item["source_kind"],
                "artifact_dir": str(source_artifact),
                "metadata_path": str(source_metadata_path),
                "metadata_sha256": _sha256(source_metadata_path),
                "state_path": str(source_state_path),
                "state_sha256": _sha256(source_state_path),
                "recovery_registration_sha256": _sha256(
                    output_root / "recovery_registration.json"
                ),
            }
            _safe_write_json(destination_metadata_path, metadata)

        composed_state = dict(state)
        composed_state["source_schema_version"] = composed_state["schema_version"]
        composed_state["schema_version"] = MINI_STATE_SCHEMA
        composed_state["metadata_sha256"] = _sha256(destination_metadata_path)
        composed_state["composition_source"] = {
            "kind": item["source_kind"],
            "state_path": str(source_state_path),
            "state_sha256": _sha256(source_state_path),
            "metadata_path": str(source_metadata_path),
            "metadata_sha256": _sha256(source_metadata_path),
        }
        destination_state_path = (
            composed_root / "_episode_states" / f"{state['episode_key']}.json"
        )
        _write_once(destination_state_path, composed_state)
        records.append(
            {
                "episode_key": state["episode_key"],
                "source_kind": item["source_kind"],
                "source_artifact_dir": str(source_artifact),
                "source_artifact_files_sha256": _relative_hashes(source_artifact),
                "source_metadata_path": str(source_metadata_path),
                "source_metadata_sha256": _sha256(source_metadata_path),
                "source_state_path": str(source_state_path),
                "source_state_sha256": _sha256(source_state_path),
                "composed_artifact_dir": str(destination_artifact),
                "composed_artifact_files_sha256": _relative_hashes(
                    destination_artifact
                ),
                "composed_metadata_sha256": _sha256(destination_metadata_path),
                "composed_state_path": str(destination_state_path),
                "composed_state_sha256": _sha256(destination_state_path),
            }
        )

    manifest = {
        "schema_version": COMPOSITION_SCHEMA,
        "study_version": mini.STUDY_VERSION,
        "status": "composed_builder_compatible_evidence_root",
        "episode_count": len(records),
        "source_recovery_manifest": {
            "path": str(source_manifest_path),
            "sha256": _sha256(source_manifest_path),
        },
        "recovery_registration": {
            "path": str(output_root / "recovery_registration.json"),
            "sha256": _sha256(output_root / "recovery_registration.json"),
        },
        "study_config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "episodes": sorted(records, key=lambda value: value["episode_key"]),
    }
    manifest_path = composed_root / "composition_manifest.json"
    _write_once(manifest_path, manifest)
    _write_sha_sidecar(manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    run_parser = subparsers.add_parser("run")
    compose_parser = subparsers.add_parser("compose")
    for command_parser in (prepare_parser, run_parser, compose_parser):
        command_parser.add_argument("--original", required=True, type=Path)
        command_parser.add_argument("--output", required=True, type=Path)
        command_parser.add_argument("--study-config", required=True, type=Path)
    prepare_parser.add_argument(
        "--candidate-limit", type=int, default=DEFAULT_CANDIDATE_LIMIT
    )
    run_parser.add_argument("--workers", type=int, default=mini.EXPECTED_PARALLELISM)
    run_parser.add_argument("--timeout", type=int, default=1200)
    compose_parser.add_argument("--composed-output", required=True, type=Path)
    args = parser.parse_args()

    original = args.original.resolve()
    output = args.output.resolve()
    config = args.study_config.resolve()
    if args.command == "prepare":
        if not 1 <= args.candidate_limit <= 1000:
            parser.error("--candidate-limit must be between 1 and 1000")
        registration = prepare(
            original, output, config, args.candidate_limit
        )
        print(json.dumps(registration, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.command == "compose":
        manifest = compose_recovery(
            original,
            output,
            config,
            args.composed_output.resolve(),
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
        return 0
    if not 1 <= args.workers <= mini.EXPECTED_PARALLELISM:
        parser.error(f"--workers must be between 1 and {mini.EXPECTED_PARALLELISM}")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    outcomes = run_recovery(
        original, output, config, args.workers, args.timeout
    )
    return 0 if len(outcomes) == 17 and all(value["succeeded"] for value in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
