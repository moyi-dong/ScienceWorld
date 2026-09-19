#!/usr/bin/env python3
"""Run isolated live Codex calibration episodes for the AER pea case."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import socketserver
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from scienceworld import ScienceWorldEnv

TASK = "mendelian-genetics-known-plant-aer"
SUPPORTED_WORLDS = (
    "white_preference",
    "position_attraction",
    "plant_attractiveness",
    "fertility_difference",
    "transient_null",
    "clean",
)
MODEL = "gpt-5.6-sol"
REASONING_EFFORT = "high"
PROMPT_VARIANTS = ("baseline", "anomaly_cue")
ANOMALY_CUE = "Pay attention to anomalies during the task and explore the underlying mechanisms."
SCRIPT_DIR = Path(__file__).resolve().parent
SCIENCEWORLD_ROOT = SCRIPT_DIR.parent
AER_BENCH_ROOT = SCIENCEWORLD_ROOT.parents[1]
AER_PEA_CASE_ROOT = (
    AER_BENCH_ROOT / "cases" / "science" / "mendelian_genetics_known_plant_aer"
)
CALIBRATION_MATRIX_PATH = (
    AER_PEA_CASE_ROOT / "construction" / "calibration_matrix.v0.2.0.json"
)
CALIBRATION_MATRIX = json.loads(CALIBRATION_MATRIX_PATH.read_text(encoding="utf-8"))
CALIBRATION_CONDITIONS = tuple(CALIBRATION_MATRIX["conditions"])
sys.path.insert(0, str(AER_BENCH_ROOT / "src"))

from aer_bench.codex_runner import CodexRunConfig, CodexRunner  # noqa: E402
from aer_bench.trace import normalize_events, read_jsonl, write_normalized  # noqa: E402


def _pot_sort_key(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", name)
    return (int(match.group(1)), name) if match else (10**9, name)


def _greenhouse_snapshot(status: dict[str, Any]) -> dict[str, str]:
    """Build a compact, solver-safe map from pot names to visible plant/flower state."""
    snapshot: dict[str, str] = {}
    for pot in sorted(status.get("pots", []), key=lambda value: _pot_sort_key(value["name"])):
        flowers: list[tuple[str, str]] = []
        plants = pot.get("plants", [])
        for plant in plants:
            plant_id = str(plant.get("plant_id", "?"))
            for flower in plant.get("active_flowers", []):
                flowers.append((str(flower.get("perceived_color", "unknown")), plant_id))
        if flowers:
            colors = "/".join(color for color, _ in flowers)
            plant_ids = ",".join(dict.fromkeys(plant_id for _, plant_id in flowers))
            snapshot[pot["name"]] = f"{colors} flower (plant {plant_ids})"
        elif plants:
            plant_ids = ",".join(str(plant.get("plant_id", "?")) for plant in plants)
            stages = "/".join(str(plant.get("stage", "unknown")) for plant in plants)
            snapshot[pot["name"]] = f"no active flower (plant {plant_ids}, stage {stages})"
        else:
            snapshot[pot["name"]] = "empty"
    return snapshot


def _greenhouse_note(
    status: dict[str, Any], previous: dict[str, str] | None, action: str
) -> tuple[str | None, dict[str, str]]:
    current = _greenhouse_snapshot(status)
    normalized = " ".join(action.lower().split())
    force_full = previous is None or normalized == "look around" or "greenhouse" in normalized
    tick = status.get("episode_tick", "?")
    if force_full:
        values = "; ".join(f"{name}={value}" for name, value in current.items())
        return f"Greenhouse flower map at tick {tick}: {values}", current
    changed = [name for name in current if current.get(name) != previous.get(name)]
    if not changed:
        return None, current
    values = "; ".join(f"{name}={current[name]}" for name in changed)
    return f"Greenhouse flower update at tick {tick}: {values}", current


def _safe_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_case_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load case module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_split(
    split_name: str,
    variation: int,
    case_root: int,
    held_out_freeze_manifest: Path | None,
) -> None:
    split_path = AER_PEA_CASE_ROOT / "construction" / "split.v0.1.0.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))[split_name]
    if variation not in split["variations"] or case_root not in split["roots"]:
        raise ValueError(
            f"variation {variation} and root {case_root} are not registered for {split_name}"
        )
    if split_name == "held_out":
        if held_out_freeze_manifest is None or not held_out_freeze_manifest.is_file():
            raise ValueError("held-out runs require --held-out-freeze-manifest")
        freeze = json.loads(held_out_freeze_manifest.read_text(encoding="utf-8"))
        if freeze.get("status") != "frozen_for_held_out":
            raise ValueError("held-out freeze manifest is not frozen_for_held_out")
        if freeze.get("acceptance_spec_version") != "0.2.0":
            raise ValueError("held-out freeze manifest uses the wrong acceptance spec")


class EpisodeService:
    """Own one hidden world while exposing only the native public interaction surface."""


    def __init__(
        self,
        world: str,
        variation: int,
        case_root: int,
        trajectory_path: Path,
        operator_window_path: Path,
        step_limit: int,
        matched_pre_exposure: bool = False,
        noise_levels: dict[str, str] | None = None,
        preference_weight: float | None = None,
    ) -> None:
        self.env = ScienceWorldEnv("", serverPath=None, envStepLimit=step_limit)
        preference_options = (
            {} if preference_weight is None else {"preference_weight": preference_weight}
        )
        self.env.configure_aer_pea_case(
            world, case_root, noise_levels=noise_levels, **preference_options
        )
        self.env.load(TASK, variation, "easy", generateGoldPath=matched_pre_exposure)
        self.trajectory_path = trajectory_path
        self.operator_window_path = operator_window_path
        self._lock = threading.Lock()
        self._index = 0
        self._note_index = 0
        self._experiment_ids: set[str] = set()
        self._active_experiment_id: str | None = None
        self.completed = False
        self._last_greenhouse_state: dict[str, str] | None = None
        self.pre_exposure_observations: list[str] = []
        if matched_pre_exposure:
            gold_actions = list(self.env.server.getGoldActionSequence())
            open_hive_index = next(
                index
                for index, action in enumerate(gold_actions)
                if action.startswith("open bee hive")
            )
            boundary = next(
                index
                for index, action in enumerate(
                    gold_actions[open_hive_index + 1 :], open_hive_index + 1
                )
                if action == "look at seed jar"
            )
            self.env.load(TASK, variation, "easy", generateGoldPath=False)
            for action in gold_actions[: boundary + 1]:
                response = self._step(action, source="matched_pre_exposure")
                observation = str(response.get("observation", ""))
                if "Greenhouse activity since your last action:" in observation:
                    self.pre_exposure_observations.append(observation)
        self.initial = self._step("look around", source="initial")

    def _append(self, event: dict[str, Any]) -> dict[str, Any]:
        event = {"index": self._index, "timestamp_unix": time.time(), **event}
        self._index += 1
        self.trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trajectory_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        return event

    def _step(self, action: str, source: str = "solver") -> dict[str, Any]:
        action_id = f"ACT-{self._index:05d}"
        before_events = self.env.get_aer_pea_case_events()
        before_reproduction_events = self.env.get_aer_pea_case_reproduction_events()
        before_summary = self.env.get_aer_pea_case_summary()
        observation, reward, completed, info = self.env.step(action, include_valid=False)
        after_events = self.env.get_aer_pea_case_events()
        after_reproduction_events = self.env.get_aer_pea_case_reproduction_events()
        after_summary = self.env.get_aer_pea_case_summary()
        greenhouse_status = self.env.get_aer_pea_case_public_status()
        greenhouse_note, self._last_greenhouse_state = _greenhouse_note(
            greenhouse_status, self._last_greenhouse_state, action
        )
        if greenhouse_note:
            observation = f"{observation}\n\n{greenhouse_note}"
        self.completed = bool(completed)
        response = {
            "ok": True,
            "kind": "step",
            "event_id": action_id,
            "action": action,
            "observation": observation,
            "reward": reward,
            "score": info["score"],
            "completed": self.completed,
            "moves": info["moves"],
            "look": info["look"],
            "inventory": info["inv"],
        }
        self._append(
            {
                "source": source,
                "request": {"command": "act", "action": action},
                "response": response,
            }
        )
        self.operator_window_path.parent.mkdir(parents=True, exist_ok=True)
        with self.operator_window_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "action_id": action_id,
                        "action": action,
                        "active_experiment_id": self._active_experiment_id,
                        "public_trajectory_index": self._index - 1,
                        "before_summary": before_summary,
                        "after_summary": after_summary,
                        "new_visits": after_events[len(before_events) :],
                        "new_reproduction_events": after_reproduction_events[
                            len(before_reproduction_events) :
                        ],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
        return response

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            command = request.get("command")
            if command == "task":
                response = {"ok": True, "kind": "task", "task": self.env.taskdescription()}
            elif command == "state":
                response = {
                    "ok": True,
                    "kind": "state",
                    "task": self.env.taskdescription(),
                    "look": self.env.look(),
                    "inventory": self.env.inventory(),
                    "completed": self.completed,
                }
            elif command == "actions":
                response = {
                    "ok": True,
                    "kind": "actions",
                    "actions": self.env.get_possible_actions(),
                }
            elif command == "objects":
                response = {
                    "ok": True,
                    "kind": "objects",
                    "objects": self.env.get_possible_objects(),
                }
            elif command == "valid":
                needle = str(request.get("filter", "")).lower()
                actions = self.env.get_valid_action_object_combinations()
                if needle:
                    actions = [action for action in actions if needle in action.lower()]
                response = {
                    "ok": True,
                    "kind": "valid",
                    "filter": needle,
                    "actions": actions[:500],
                    "truncated": len(actions) > 500,
                }
            elif command == "act":
                action = str(request.get("action", "")).strip()
                if not action:
                    response = {"ok": False, "error": "action must be non-empty"}
                elif self.completed:
                    response = {"ok": False, "error": "episode is already complete"}
                else:
                    return self._step(action)
            elif command == "batch":
                actions = request.get("actions")
                if not isinstance(actions, list) or not actions:
                    response = {"ok": False, "error": "actions must be a non-empty list"}
                else:
                    results = []
                    for raw_action in actions[:200]:
                        if self.completed:
                            break
                        results.append(self._step(str(raw_action).strip()))
                    return {
                        "ok": True,
                        "kind": "batch",
                        "results": results,
                        "completed": self.completed,
                    }
            else:
                response = {"ok": False, "error": "unsupported public command"}

            self._append({"source": "solver", "request": request, "response": response})
            return response

    def close(self) -> None:
        self.env.close()


class _UnixServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            request = json.loads(self.rfile.readline().decode("utf-8"))
            response = self.server.episode.handle(request)  # type: ignore[attr-defined]
        except Exception as error:
            response = {"ok": False, "error": f"request failed: {type(error).__name__}: {error}"}
        self.wfile.write((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))


def _prompt(
    service: EpisodeService, prompt_variant: str, world: str | None = None
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
- `python3 lab.py record '<JSON>'` to timestamp a structured lab-notebook entry

Complete the commissioned task autonomously. Base your conclusion on observations from this
episode and use the remaining simulator budget carefully. Do not inspect benchmark source,
construction materials, graders, hidden tests, or oracle data. When finished, return only the
JSON object required by submission.schema.json.
"""
    pre_exposure_observations = getattr(service, "pre_exposure_observations", [])
    if pre_exposure_observations:
        transcript = "\n\n".join(pre_exposure_observations)
        prompt += (
            "\nMatched public pre-exposure observations (the same deterministic handoff "
            "is reused across calibration conditions; prefix actions are not provided):\n"
            f"{transcript}\n"
        )
    if prompt_variant == "anomaly_cue":
        prompt += f"\n{ANOMALY_CUE}\n"
    elif prompt_variant in CALIBRATION_CONDITIONS:
        addition = CALIBRATION_MATRIX["conditions"][prompt_variant]["prompt_addition"]
        if addition:
            if "{target_surface_statement}" in addition:
                if world is None:
                    raise ValueError("target-surface calibration prompts require a world")
                addition = addition.format(
                    target_surface_statement=CALIBRATION_MATRIX[
                        "target_surface_by_world"
                    ][world]
                )
            prompt += f"\n{addition}\n"
    return prompt


def run_episode(
    runner: CodexRunner,
    output_root: Path,
    world: str,
    repetition: int,
    variation: int,
    case_root: int,
    timeout_seconds: int,
    step_limit: int,
    prompt_variant: str,
) -> dict[str, Any]:
    run_id = (
        f"{world}-variation-{variation:02d}-root-{case_root:04d}-run-{repetition:02d}"
    )
    artifact_dir = output_root / run_id
    if artifact_dir.exists():
        raise RuntimeError(f"refusing to overwrite {artifact_dir}")
    artifact_dir.mkdir(parents=True)

    # macOS limits AF_UNIX addresses to roughly 104 bytes, so keep the per-run
    # workspace and socket names deliberately short.
    with tempfile.TemporaryDirectory(prefix="ap-", dir="/tmp") as temporary:
        workspace = Path(temporary)
        client_path = workspace / "lab.py"
        schema_path = workspace / "submission.schema.json"
        shutil.copy2(AER_PEA_CASE_ROOT / "public" / "lab.py", client_path)
        shutil.copy2(AER_PEA_CASE_ROOT / "public" / "submission.schema.json", schema_path)
        socket_path = workspace / "scienceworld.sock"
        trajectory_path = artifact_dir / "public_environment_trajectory.jsonl"
        operator_window_path = artifact_dir / "operator_action_windows.jsonl"

        matched_pre_exposure = bool(
            prompt_variant in CALIBRATION_CONDITIONS
            and CALIBRATION_MATRIX["matched_pre_exposure"]["required"]
        )
        service = EpisodeService(
            world,
            variation,
            case_root,
            trajectory_path,
            operator_window_path,
            step_limit,
            matched_pre_exposure=matched_pre_exposure,
        )
        server = _UnixServer(str(socket_path), _Handler)
        server.episode = service  # type: ignore[attr-defined]
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        started = time.time()
        try:
            prompt = _prompt(service, prompt_variant, world)
            config = CodexRunConfig(
                workspace=workspace,
                artifact_dir=artifact_dir / "codex",
                prompt=prompt,
                output_schema=schema_path,
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
            hidden_reproduction_events = service.env.get_aer_pea_case_reproduction_events()
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=5)
            service.close()

        normalized_path = artifact_dir / "codex" / "normalized_events.jsonl"
        if result.events_path.is_file():
            write_normalized(normalized_path, normalize_events(read_jsonl(result.events_path)))
        _safe_write_json(artifact_dir / "hidden_summary.json", hidden_summary)
        _safe_write_json(artifact_dir / "hidden_events.json", hidden_events)
        _safe_write_json(
            artifact_dir / "hidden_reproduction_events.json", hidden_reproduction_events
        )
        evidence_module = _load_case_module(
            "aer_pea_evidence", AER_PEA_CASE_ROOT / "hidden" / "evidence.py"
        )
        grader_module = _load_case_module(
            "aer_pea_grader", AER_PEA_CASE_ROOT / "hidden" / "grader.py"
        )
        grading_events = evidence_module.build_events(
            read_jsonl(trajectory_path),
            read_jsonl(operator_window_path),
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
            _safe_write_json(grade_path, grade)
        _safe_write_json(
            artifact_dir / "run_metadata.json",
            {
                "run_id": run_id,
                "task": TASK,
                "world": world,
                "variation": variation,
                "case_root": case_root,
                "repetition": repetition,
                "model": MODEL,
                "reasoning_effort": REASONING_EFFORT,
                "fast_mode": False,
                "prompt_variant": prompt_variant,
                "calibration_condition": (
                    prompt_variant if prompt_variant in CALIBRATION_CONDITIONS else None
                ),
                "matched_pre_exposure": matched_pre_exposure,
                "matched_pre_exposure_observation_count": len(
                    service.pre_exposure_observations
                ),
                "anomaly_cue": ANOMALY_CUE if prompt_variant == "anomaly_cue" else None,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "calibration_matrix_sha256": _sha256(CALIBRATION_MATRIX_PATH),
                "scienceworld_jar_sha256": _sha256(
                    SCIENCEWORLD_ROOT / "scienceworld" / "scienceworld.jar"
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
                    "public_environment_trajectory.jsonl": _sha256(trajectory_path),
                    "operator_action_windows.jsonl": _sha256(operator_window_path),
                    "hidden_events.json": _sha256(artifact_dir / "hidden_events.json"),
                    "hidden_reproduction_events.json": _sha256(
                        artifact_dir / "hidden_reproduction_events.json"
                    ),
                    "hidden_summary.json": _sha256(artifact_dir / "hidden_summary.json"),
                    "grading_events.jsonl": _sha256(grading_events_path),
                    "grade.json": _sha256(grade_path) if grade_path.is_file() else None,
                    "codex/events.jsonl": _sha256(result.events_path),
                    "codex/final.json": _sha256(result.final_output_path)
                    if result.final_output_path.is_file()
                    else None,
                },
            },
        )
        return {
            "run_id": run_id,
            "status": result.status,
            "succeeded": result.succeeded,
            "environment_completed": service.completed,
            "hidden_summary": hidden_summary,
            "artifact_dir": str(artifact_dir),
            "errors": result.errors,
            "thread_id": result.thread_id,
            "usage": result.usage,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world", action="append", choices=SUPPORTED_WORLDS)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--variation", type=int)
    parser.add_argument("--case-root", type=int)
    parser.add_argument("--split", choices=("development", "held_out"), default="development")
    parser.add_argument("--held-out-freeze-manifest", type=Path)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--step-limit", type=int, default=1000)
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt-variant", choices=PROMPT_VARIANTS)
    prompt_group.add_argument("--condition", choices=CALIBRATION_CONDITIONS)
    args = parser.parse_args()
    if args.runs is not None and args.runs < 1:
        parser.error("--runs must be positive")
    if args.case_root is not None and args.case_root < 0:
        parser.error("--case-root must be non-negative")

    if args.condition:
        if args.split != "development":
            parser.error("the proposed live calibration matrix is development-only")
        if any(value is not None for value in (args.runs, args.variation, args.case_root)):
            parser.error(
                "formal --condition runs use the frozen replication_cells; "
                "do not pass --runs, --variation, or --case-root"
            )
        schedule = CALIBRATION_MATRIX["replication_cells"]
    else:
        runs = args.runs or 3
        variation = 0 if args.variation is None else args.variation
        case_root = 101 if args.case_root is None else args.case_root
        schedule = [
            {
                "repetition": repetition,
                "variation": variation,
                "case_root": case_root,
            }
            for repetition in range(1, runs + 1)
        ]

    try:
        for cell in schedule:
            _validate_split(
                args.split,
                cell["variation"],
                cell["case_root"],
                args.held_out_freeze_manifest,
            )
    except ValueError as error:
        parser.error(str(error))

    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    worlds = tuple(args.world or SUPPORTED_WORLDS)
    prompt_variant = args.condition or args.prompt_variant or "baseline"
    episode_output_root = output_root / args.condition if args.condition else output_root
    episode_output_root.mkdir(parents=True, exist_ok=True)
    runner = CodexRunner()
    outcomes = []
    for world in worlds:
        for cell in schedule:
            repetition = cell["repetition"]
            print(
                f"START {world} replication {repetition}/{len(schedule)} "
                f"variation={cell['variation']} root={cell['case_root']}",
                flush=True,
            )
            outcome = run_episode(
                runner,
                episode_output_root,
                world,
                repetition,
                cell["variation"],
                cell["case_root"],
                args.timeout,
                args.step_limit,
                prompt_variant,
            )
            outcomes.append(outcome)
            print(
                f"DONE {outcome['run_id']} status={outcome['status']} "
                f"environment_completed={outcome['environment_completed']}",
                flush=True,
            )
            _safe_write_json(episode_output_root / "batch_summary.json", outcomes)
    return 0 if all(outcome["succeeded"] for outcome in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
