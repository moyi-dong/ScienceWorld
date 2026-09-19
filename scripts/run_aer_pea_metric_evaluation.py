#!/usr/bin/env python3
"""Build and execute the frozen 12-run pea metric-evaluation development study.

Live calls are made only by the explicit ``run``/``full`` commands.  ``prepare`` performs the
deterministic source inventory and two-restore S0 gate without invoking Codex.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import run_aer_pea_calibration as frozen
from scienceworld import ScienceWorldEnv

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
SCIENCEWORLD_ROOT = SCRIPT_DIR.parent
AER_BENCH_ROOT = SCIENCEWORLD_ROOT.parents[1]
CASE_ROOT = AER_BENCH_ROOT / "cases/science/mendelian_genetics_known_plant_aer"
DEFAULT_CONFIG = CASE_ROOT / "construction/metric-evaluation-study.v0.1-development.json"
DEFAULT_SCHEMA = CASE_ROOT / "construction/metric-evaluation-output.schema.json"
DEFAULT_SOURCE = (
    AER_BENCH_ROOT
    / "artifacts/aer_pea_case/gate-e-matched-development-v0.4.2-development-0001"
)
DEFAULT_OUTPUT = AER_BENCH_ROOT / "artifacts/aer_pea_case/metric-evaluation-v0.1-development-0001"
MATCHED_CONFIG = CASE_ROOT / "construction/gate-e-matched-development-study.v0.4.2-development.json"
sys.path.insert(0, str(AER_BENCH_ROOT / "src"))

from aer_bench import trace as aer_trace  # noqa: E402
from aer_bench.codex_runner import CodexRunConfig, CodexRunner  # noqa: E402
from aer_bench.pea_metric_evaluation import (  # noqa: E402
    STUDY_VERSION,
    WORLDS,
    build_evidence_bundle,
    build_s0_recipe,
    extract_experiment_plans,
    inventory_source_runs,
    normalize_run_trajectory,
    read_json,
    read_jsonl,
    schema_for_output,
    semantic_restore_payload,
    sha256_json,
    sha256_path,
    validate_event_references,
    validate_study_config,
    write_json,
    write_jsonl,
)


def _hash_if_file(path: Path) -> str | None:
    return sha256_path(path) if path.is_file() else None


class S0ReplayService(frozen.EpisodeService):
    """Reset one registered world and replay an operator-owned S0 action recipe."""

    def __init__(
        self,
        recipe: dict[str, Any],
        trajectory_path: Path,
        operator_path: Path,
        step_limit: int,
    ) -> None:
        self.recipe = recipe
        self.env = ScienceWorldEnv("", serverPath=None, envStepLimit=step_limit)
        self.env.configure_aer_pea_case(recipe["world"], recipe["case_root"])
        self.env.load(frozen.TASK, recipe["variation"], "easy", generateGoldPath=False)
        self.trajectory_path = trajectory_path
        self.operator_window_path = operator_path
        if trajectory_path.exists() or operator_path.exists():
            raise RuntimeError("S0 replay refuses to overwrite evidence")
        self._lock = threading.Lock()
        self._index = 0
        self._note_index = 0
        self._experiment_ids = set()
        self._active_experiment_id = None
        self.completed = False
        self.pre_exposure_observations: list[str] = []
        for action in recipe["recovery"]["actions"]:
            response = self._step(action, source="s0_construction")
            observation = str(response.get("observation", ""))
            if "Greenhouse activity since your last action:" in observation:
                self.pre_exposure_observations.append(observation)
        self.initial = self._step("look around", source="initial")


def _restore_once(recipe: dict[str, Any], root: Path, restore_number: int) -> dict[str, Any]:
    restore_dir = root / f"restore-{restore_number:02d}"
    restore_dir.mkdir(parents=True, exist_ok=False)
    service = S0ReplayService(
        recipe,
        restore_dir / "public_environment_trajectory.jsonl",
        restore_dir / "operator_action_windows.jsonl",
        1000,
    )
    try:
        summary = service.env.get_aer_pea_case_summary()
        verification = service._step("look around", source="s0_verification")
        payload = semantic_restore_payload(
            task=service.env.taskdescription(),
            initial=service.initial,
            summary=summary,
            verification=verification,
        )
        write_json(restore_dir / "semantic_payload.json", payload)
        return {
            "restore": restore_number,
            "semantic_sha256": sha256_json(payload),
            "semantic_payload_path": str((restore_dir / "semantic_payload.json").resolve()),
            "public_trajectory_sha256": sha256_path(
                restore_dir / "public_environment_trajectory.jsonl"
            ),
            "operator_window_sha256": sha256_path(
                restore_dir / "operator_action_windows.jsonl"
            ),
        }
    finally:
        service.close()


def prepare_s0(config_path: Path, source_root: Path, output_root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    validate_study_config(config)
    output_root.mkdir(parents=True, exist_ok=True)
    inventory_path = output_root / "source_inventory.json"
    inventory = inventory_source_runs(source_root)
    inventory_payload = {
        "schema_version": "aer.pea.metric-evaluation-source-inventory.v1",
        "study_version": STUDY_VERSION,
        "source_root": str(source_root.resolve()),
        "runs": inventory,
    }
    if inventory_path.exists():
        if read_json(inventory_path) != inventory_payload:
            raise RuntimeError("existing source inventory differs from current frozen inputs")
    else:
        write_json(inventory_path, inventory_payload)

    manifests: list[dict[str, Any]] = []
    for source in inventory:
        recipe = build_s0_recipe(source)
        s0_dir = output_root / "s0" / recipe["s0_id"]
        manifest_path = s0_dir / "s0_manifest.json"
        if manifest_path.is_file():
            manifest = read_json(manifest_path)
            if manifest.get("verification", {}).get("status") != "verified_deterministic":
                raise RuntimeError(f"existing S0 is not verified: {manifest_path}")
            manifests.append(manifest)
            continue
        s0_dir.mkdir(parents=True, exist_ok=False)
        restores = [_restore_once(recipe, s0_dir, index) for index in (1, 2)]
        signatures = {item["semantic_sha256"] for item in restores}
        if len(signatures) != 1:
            raise RuntimeError(f"S0 restore is nondeterministic: {recipe['s0_id']}")
        recipe["verification"] = {
            "status": "verified_deterministic",
            "restore_count": 2,
            "verification_action": "look around",
            "semantic_sha256": restores[0]["semantic_sha256"],
            "restores": restores,
        }
        recipe["bindings"] = {
            "study_config": {
                "path": str(config_path.resolve()),
                "sha256": sha256_path(config_path),
            },
            "runner": {"path": str(SCRIPT_PATH), "sha256": sha256_path(SCRIPT_PATH)},
            "implementation_helpers": {
                "path": str(
                    (
                        AER_BENCH_ROOT
                        / "src/aer_bench/pea_metric_evaluation.py"
                    ).resolve()
                ),
                "sha256": sha256_path(
                    AER_BENCH_ROOT / "src/aer_bench/pea_metric_evaluation.py"
                ),
            },
            "scienceworld_jar": {
                "path": str((SCIENCEWORLD_ROOT / "scienceworld/scienceworld.jar").resolve()),
                "sha256": sha256_path(SCIENCEWORLD_ROOT / "scienceworld/scienceworld.jar"),
            },
        }
        write_json(manifest_path, recipe)
        manifests.append(recipe)
        print(
            f"S0 VERIFIED {recipe['s0_id']} actions={recipe['boundary']['action_count']} "
            f"comparable={recipe['boundary']['after_comparable_visits']}",
            flush=True,
        )
    freeze = {
        "schema_version": "aer.pea.metric-evaluation-s0-freeze.v1",
        "study_version": STUDY_VERSION,
        "status": "verified_and_frozen_for_twelve_live_runs",
        "registered_s0_count": 12,
        "inventory_sha256": sha256_path(inventory_path),
        "s0": [
            {
                "s0_id": item["s0_id"],
                "manifest_path": str(
                    (output_root / "s0" / item["s0_id"] / "s0_manifest.json").resolve()
                ),
                "manifest_sha256": sha256_path(
                    output_root / "s0" / item["s0_id"] / "s0_manifest.json"
                ),
            }
            for item in manifests
        ],
    }
    freeze_path = output_root / "s0_freeze_manifest.json"
    if freeze_path.exists() and read_json(freeze_path) != freeze:
        raise RuntimeError("existing S0 freeze differs from verified manifests")
    write_json(freeze_path, freeze)
    return freeze


def _build_main_prompt(service: S0ReplayService) -> str:
    prompt = frozen._prompt(service, "baseline")
    matched = read_json(MATCHED_CONFIG)
    prompt += (
        "\nPublic notebook protocol shared by every study condition:\n"
        f"{matched['common_interface_instruction']}\n"
    )
    return prompt


def _run_main_episode(
    config: dict[str, Any], output_root: Path, s0_manifest_path: Path
) -> dict[str, Any]:
    s0 = read_json(s0_manifest_path)
    run_id = f"{s0['world']}--rep-{s0['repetition']:02d}"
    run_dir = output_root / "runs" / run_id
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.is_file():
        metadata = read_json(metadata_path)
        if metadata.get("status") in {"completed", "failed", "timed_out"}:
            print(f"RUN PRESERVED {run_id} status={metadata['status']}", flush=True)
            return metadata
        raise RuntimeError(f"existing main run is not terminal: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="apme-main-", dir="/tmp") as temporary:
        workspace = Path(temporary)
        shutil.copy2(CASE_ROOT / "public/lab.py", workspace / "lab.py")
        schema_path = workspace / "submission.schema.json"
        shutil.copy2(CASE_ROOT / "public/submission.schema.json", schema_path)
        service = S0ReplayService(
            s0,
            run_dir / "public_environment_trajectory.jsonl",
            run_dir / "operator_action_windows.jsonl",
            config["formal_episode_action_budget"],
        )
        socket_path = workspace / "scienceworld.sock"
        server = frozen._UnixServer(str(socket_path), frozen._Handler)
        server.episode = service  # type: ignore[attr-defined]
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        prompt = _build_main_prompt(service)
        (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        started = time.time()
        try:
            result = CodexRunner().run(
                CodexRunConfig(
                    workspace=workspace,
                    artifact_dir=run_dir / "codex",
                    prompt=prompt,
                    output_schema=schema_path,
                    model=config["model"],
                    reasoning_effort=config["reasoning_effort"],
                    timeout_seconds=config["timeout_seconds"],
                    sandbox="workspace-write",
                    ephemeral=True,
                    shell_tool_enabled=True,
                    extra_config=("features.fast_mode=false",),
                    unix_socket_allowlist=(socket_path,),
                    deny_read_paths=(AER_BENCH_ROOT, SCIENCEWORLD_ROOT, output_root),
                )
            )
            hidden_summary = service.env.get_aer_pea_case_summary()
            hidden_events = service.env.get_aer_pea_case_events()
            hidden_reproduction = service.env.get_aer_pea_case_reproduction_events()
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=5)
            service.close()
        normalized_codex = run_dir / "codex/normalized_events.jsonl"
        if result.events_path.is_file():
            aer_trace.write_normalized(
                normalized_codex,
                aer_trace.normalize_events(aer_trace.read_jsonl(result.events_path)),
            )
        write_json(run_dir / "hidden_summary.json", hidden_summary)
        write_json(run_dir / "hidden_events.json", hidden_events)
        write_json(run_dir / "hidden_reproduction_events.json", hidden_reproduction)
        evidence_module = frozen._load_case_module(
            "aer_pea_metric_evidence", CASE_ROOT / "hidden/evidence.py"
        )
        grader_module = frozen._load_case_module(
            "aer_pea_metric_grader", CASE_ROOT / "hidden/grader.py"
        )
        grading_events = evidence_module.build_events(
            read_jsonl(run_dir / "public_environment_trajectory.jsonl"),
            read_jsonl(run_dir / "operator_action_windows.jsonl"),
            world=s0["world"],
        )
        write_jsonl(run_dir / "grading_events.jsonl", grading_events)
        grade_path = run_dir / "grade.json"
        if result.final_output_path.is_file():
            submission = read_json(result.final_output_path)
            grade = grader_module.grade(
                submission,
                grading_events,
                expected_world=s0["world"],
                expected_height_trait="dominant" if s0["variation"] < 15 else "recessive",
            )
            write_json(grade_path, grade)
        metadata = {
            "schema_version": "aer.pea.metric-evaluation-main-run.v1",
            "study_version": STUDY_VERSION,
            "run_id": run_id,
            "s0_id": s0["s0_id"],
            "s0_manifest_sha256": sha256_path(s0_manifest_path),
            "world": s0["world"],
            "repetition": s0["repetition"],
            "case_root": s0["case_root"],
            "variation": s0["variation"],
            "condition": "l0_interface_only",
            "model": config["model"],
            "reasoning_effort": config["reasoning_effort"],
            "status": result.status,
            "returncode": result.returncode,
            "errors": result.errors,
            "environment_completed": service.completed,
            "thread_id": result.thread_id,
            "usage": result.usage,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "files_sha256": {
                relative: _hash_if_file(run_dir / relative)
                for relative in (
                    "prompt.txt",
                    "public_environment_trajectory.jsonl",
                    "operator_action_windows.jsonl",
                    "hidden_summary.json",
                    "hidden_events.json",
                    "hidden_reproduction_events.json",
                    "grading_events.jsonl",
                    "grade.json",
                    "codex/events.jsonl",
                    "codex/normalized_events.jsonl",
                    "codex/final.json",
                )
            },
        }
        write_json(metadata_path, metadata)
        print(
            f"RUN DONE {run_id} status={result.status} completed={service.completed}",
            flush=True,
        )
        return metadata


def run_main_episodes(config_path: Path, output_root: Path) -> list[dict[str, Any]]:
    config = read_json(config_path)
    validate_study_config(config)
    freeze_path = output_root / "s0_freeze_manifest.json"
    if not freeze_path.is_file():
        raise FileNotFoundError("S0 freeze is required before live runs")
    freeze = read_json(freeze_path)
    if freeze.get("status") != "verified_and_frozen_for_twelve_live_runs":
        raise ValueError("S0 freeze is invalid")
    outcomes: list[dict[str, Any]] = []
    for item in freeze["s0"]:
        path = Path(item["manifest_path"])
        if sha256_path(path) != item["manifest_sha256"]:
            raise ValueError(f"S0 manifest changed after freeze: {path}")
        print(f"RUN START {item['s0_id']}", flush=True)
        outcomes.append(_run_main_episode(config, output_root, path))
        write_json(output_root / "main_run_summary.json", outcomes)
    return outcomes


def _context_block(run_dir: Path, terminal: bool) -> str:
    prompt = (run_dir / "prompt.txt").read_text(encoding="utf-8")
    if not terminal:
        return prompt
    parts = [prompt, "\nSaved public S0-to-F record follows.\n"]
    for relative in (
        "public_environment_trajectory.jsonl",
        "codex/normalized_events.jsonl",
        "codex/final.json",
    ):
        path = run_dir / relative
        if path.is_file():
            parts.append(f"\n--- {relative} ---\n{path.read_text(encoding='utf-8')}\n")
    return "".join(parts)


def _run_readonly_codex(
    *,
    prompt: str,
    schema: dict[str, Any],
    artifact_dir: Path,
    model: str,
    reasoning_effort: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    if (artifact_dir / "manifest.json").is_file():
        manifest = read_json(artifact_dir / "manifest.json")
        return {"status": manifest["status"], "errors": manifest.get("errors", [])}
    with tempfile.TemporaryDirectory(prefix="apme-read-", dir="/tmp") as temporary:
        workspace = Path(temporary)
        schema_path = workspace / "output.schema.json"
        write_json(schema_path, schema)
        result = CodexRunner().run(
            CodexRunConfig(
                workspace=workspace,
                artifact_dir=artifact_dir,
                prompt=prompt,
                output_schema=schema_path,
                model=model,
                reasoning_effort=reasoning_effort,
                timeout_seconds=timeout_seconds,
                sandbox="read-only",
                ephemeral=True,
                shell_tool_enabled=False,
            )
        )
    return {"status": result.status, "errors": result.errors}


def _probe_or_extract_job(
    run_dir: Path,
    config: dict[str, Any],
    master_schema: dict[str, Any],
    job: str,
) -> tuple[str, dict[str, Any]]:
    if job in {"P1", "P2", "P3", "P4"}:
        terminal = job in {"P3", "P4"}
        context = _context_block(run_dir, terminal)
        prompt = (
            context
            + "\n\nThis is an independent evaluation probe. It does not continue or alter "
            "the saved "
            "episode. Answer only from the context above and return the required JSON object.\n\n"
            + config["probe_prompts"][job]
        )
        schema = schema_for_output(master_schema, job)
        artifact_dir = run_dir / "evaluation/probes" / job
    else:
        normalized_path = run_dir / "evaluation/normalized_trajectory.jsonl"
        prompt = (
            "You are extracting evidence from a saved agent trajectory for research analysis. "
            "Do not judge correctness and do not assign a score. Cite only exact event IDs and "
            "text present in the record. Return the required JSON object.\n\n"
            f"Task: {config['trajectory_tasks'][job]}\n\n"
            "Trajectory:\n"
            + normalized_path.read_text(encoding="utf-8")
        )
        schema = schema_for_output(master_schema, "trajectory_evidence")
        artifact_dir = run_dir / "evaluation/trajectory_extraction" / job
    outcome = _run_readonly_codex(
        prompt=prompt,
        schema=schema,
        artifact_dir=artifact_dir,
        model=config["model"],
        reasoning_effort=config["reasoning_effort"],
        timeout_seconds=config["timeout_seconds"],
    )
    return job, outcome


def _summary_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    numeric = (
        "comparable_visits",
        "total_visits",
        "white_visits",
        "purple_visits",
        "pollinations",
        "fruit_sets",
        "white_fruit_sets",
        "purple_fruit_sets",
    )
    return {key: after.get(key, 0) - before.get(key, 0) for key in numeric}


def _replay_experiment_once(
    s0: dict[str, Any], plan: dict[str, Any], world: str, repetition: int
) -> dict[str, Any]:
    env = ScienceWorldEnv("", serverPath=None, envStepLimit=1000)
    env.configure_aer_pea_case(world, s0["case_root"])
    env.load(frozen.TASK, s0["variation"], "easy", generateGoldPath=False)
    try:
        for action in s0["recovery"]["actions"]:
            env.step(action, include_valid=False)
        env.step("look around", include_valid=False)
        for index, action in enumerate(plan["setup_actions"]):
            observation, _, completed, _ = env.step(action, include_valid=False)
            if completed and index + 1 < len(plan["setup_actions"]):
                return {
                    "status": "incompatible_setup",
                    "first_failure_index": index,
                    "observation": observation,
                }
        before = env.get_aer_pea_case_summary()
        actions: list[dict[str, Any]] = []
        for index, action in enumerate(plan["window_actions"]):
            observation, reward, completed, info = env.step(action, include_valid=False)
            actions.append(
                {
                    "index": index,
                    "action": action,
                    "observation": observation,
                    "reward": reward,
                    "completed": bool(completed),
                    "score": info.get("score"),
                }
            )
            if completed and index + 1 < len(plan["window_actions"]):
                return {
                    "status": "incompatible_window",
                    "first_failure_index": index,
                    "actions": actions,
                }
        after = env.get_aer_pea_case_summary()
        outcome = {
            "status": "completed",
            "repetition": repetition,
            "actions": actions,
            "summary_delta": _summary_delta(before, after),
            "new_visit_count": len(env.get_aer_pea_case_events())
            - int(before.get("total_visits", 0)),
            "final_public_state": {
                "look": env.look(),
                "inventory": env.inventory(),
            },
        }
        outcome["outcome_sha256"] = sha256_json(
            {key: value for key, value in outcome.items() if key != "repetition"}
        )
        return outcome
    finally:
        env.close()


def _run_differential(
    s0: dict[str, Any], plans: list[dict[str, Any]]
) -> dict[str, Any]:
    experiments: list[dict[str, Any]] = []
    for plan in plans:
        worlds: dict[str, Any] = {}
        for world in WORLDS:
            repetitions = [
                _replay_experiment_once(s0, plan, world, repetition)
                for repetition in (1, 2)
            ]
            hashes = {item.get("outcome_sha256") for item in repetitions}
            stable = len(hashes) == 1 and None not in hashes
            worlds[world] = {"stable": stable, "repetitions": repetitions}
        experiments.append(
            {
                "experiment_id": plan["experiment_id"],
                "worlds": worlds,
                "cross_world_outcome_groups": {
                    digest: sorted(
                        world
                        for world, value in worlds.items()
                        if value["stable"]
                        and value["repetitions"][0].get("outcome_sha256") == digest
                    )
                    for digest in sorted(
                        {
                            value["repetitions"][0].get("outcome_sha256")
                            for value in worlds.values()
                            if value["stable"]
                        }
                    )
                },
            }
        )
    return {
        "schema_version": "aer.pea.metric-evaluation-differential-replay.v1",
        "study_version": STUDY_VERSION,
        "s0_id": s0["s0_id"],
        "experiment_count": len(experiments),
        "difference_threshold": None,
        "experiments": experiments,
    }


def analyze_runs(
    config_path: Path, schema_path: Path, output_root: Path, workers: int
) -> list[dict[str, Any]]:
    config = read_json(config_path)
    validate_study_config(config)
    master_schema = read_json(schema_path)
    freeze = read_json(output_root / "s0_freeze_manifest.json")
    s0_by_id = {item["s0_id"]: item for item in freeze["s0"]}
    bundles: list[dict[str, Any]] = []
    for s0_id, frozen_s0 in s0_by_id.items():
        run_dir = output_root / "runs" / s0_id
        metadata_path = run_dir / "run_metadata.json"
        if not metadata_path.is_file():
            print(f"ANALYSIS SKIP {s0_id}: missing main run", flush=True)
            continue
        metadata = read_json(metadata_path)
        if metadata.get("status") != "completed":
            print(f"ANALYSIS SKIP {s0_id}: main status={metadata.get('status')}", flush=True)
            continue
        evaluation = run_dir / "evaluation"
        normalized_path = evaluation / "normalized_trajectory.jsonl"
        normalized = normalize_run_trajectory(run_dir, normalized_path)
        jobs = ("P1", "P2", "P3", "P4", "AR_IT", "EV_MI")
        print(f"EVALUATION START {s0_id} jobs={len(jobs)}", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _probe_or_extract_job, run_dir, config, master_schema, job
                ): job
                for job in jobs
            }
            for future in as_completed(futures):
                job, result = future.result()
                print(f"EVALUATION DONE {s0_id} {job} status={result['status']}", flush=True)
        extraction_paths = {
            job: evaluation / "trajectory_extraction" / job / "final.json"
            for job in ("AR_IT", "EV_MI")
        }
        for path in extraction_paths.values():
            if path.is_file():
                validate_event_references(read_json(path), normalized)
        s0_path = Path(frozen_s0["manifest_path"])
        s0 = read_json(s0_path)
        plans = extract_experiment_plans(run_dir, s0)
        plans_path = evaluation / "experiment_plans.json"
        write_json(
            plans_path,
            {
                "schema_version": "aer.pea.metric-evaluation-experiment-plan-set.v1",
                "study_version": STUDY_VERSION,
                "run_id": s0_id,
                "plans": plans,
            },
        )
        differential_path = evaluation / "differential_replay.json"
        write_json(differential_path, _run_differential(s0, plans))
        probe_paths = {
            job: evaluation / "probes" / job / "final.json"
            for job in ("P1", "P2", "P3", "P4")
        }
        bundle = build_evidence_bundle(
            run_id=s0_id,
            s0_manifest_path=s0_path,
            run_dir=run_dir,
            normalized_path=normalized_path,
            probe_paths=probe_paths,
            extraction_paths=extraction_paths,
            experiment_plan_path=plans_path,
            differential_path=differential_path,
        )
        bundle_path = evaluation / "evidence_bundle.json"
        write_json(bundle_path, bundle)
        bundles.append(bundle)
        print(
            f"BUNDLE DONE {s0_id} complete={bundle['complete']} experiments={len(plans)}",
            flush=True,
        )
    summary = {
        "schema_version": "aer.pea.metric-evaluation-bundle-set.v1",
        "study_version": STUDY_VERSION,
        "registered_bundle_count": 12,
        "generated_bundle_count": len(bundles),
        "complete_bundle_count": sum(item["complete"] for item in bundles),
        "official_leaderboard_result": False,
        "formal_metric_scores": None,
        "bundles": bundles,
    }
    write_json(output_root / "evidence_bundle_summary.json", summary)
    return bundles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "analyze", "full"))
    parser.add_argument("--study-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--analysis-workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.analysis_workers <= 8:
        parser.error("--analysis-workers must be between 1 and 8")
    config_path = args.study_config.resolve()
    schema_path = args.output_schema.resolve()
    source_root = args.source.resolve()
    output_root = args.output.resolve()
    if args.command in {"prepare", "full"}:
        prepare_s0(config_path, source_root, output_root)
    if args.command in {"run", "full"}:
        run_main_episodes(config_path, output_root)
    if args.command in {"analyze", "full"}:
        analyze_runs(config_path, schema_path, output_root, args.analysis_workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
