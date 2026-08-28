#!/usr/bin/env python3
"""Construct, run, and evaluate the v0.6 pea native-handoff study.

The four commands intentionally have disjoint responsibilities:

* ``construct`` lets an isolated Codex builder create paired handoff states, then performs
  deterministic reset/replay and independent blind surface review;
* ``run`` replays only a frozen recipe and preserves a native Codex checkpoint/fork tree;
* ``evaluate`` resumes the preserved sibling forks without starting ScienceWorld;
* ``report`` deterministically aggregates the three registered metrics.

This is a development study. Human blind review remains required before any packet can be
promoted, Experiment is deliberately not run, and no composite score is computed.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
SCIENCEWORLD_ROOT = SCRIPT_PATH.parents[1]
ROOT = SCRIPT_PATH.parents[3]
sys.path.insert(0, str(SCIENCEWORLD_ROOT))
sys.path.insert(0, str(ROOT / "src"))

import run_aer_pea_calibration as frozen  # noqa: E402
import run_aer_pea_formulation_v0_5 as v05  # noqa: E402
from jsonschema import Draft202012Validator  # noqa: E402

from aer_bench.codex_app_server import (  # noqa: E402
    CodexAppServer,
    TurnResult,
    session_inventory,
)
from aer_bench.codex_runner import CodexRunConfig, CodexRunner  # noqa: E402
from aer_bench.pea_handoff_v06 import (  # noqa: E402
    aggregate,
    read_json,
    score_task,
    sha256_json,
    sha256_path,
    validate_contracts,
    write_json,
)

CASE_ROOT = ROOT / "cases/science/mendelian_genetics_known_plant_aer/revisions/v1_development"
CONSTRUCTION = CASE_ROOT / "construction"
PUBLIC = CASE_ROOT / "public"
HIDDEN = CASE_ROOT / "hidden"
MATRIX_PATH = CONSTRUCTION / "hidden-configuration-matrix.v0.5-development.json"
TASKS_PATH = CONSTRUCTION / "formulation-task-matrix.v0.6-development.json"
CONSTRUCTION_PATH = CONSTRUCTION / "handoff-construction.v0.6-development.json"
EVALUATION_PATH = CONSTRUCTION / "evaluation-contract.v0.6-development.json"
BUILDER_SCHEMA_PATH = CONSTRUCTION / "handoff-builder-output.schema.v0.6-development.json"
BUILDER_SCHEDULE_PATH = CONSTRUCTION / "handoff-builder-schedule.v0.6-development.json"
REVIEW_RUBRIC_PATH = CONSTRUCTION / "handoff-review-rubric.v0.6-development.json"
REVIEW_SCHEMA_PATH = PUBLIC / "handoff-review.schema.v0.6-development.json"
PROBE_SCHEMA_PATH = PUBLIC / "evaluation-output.schema.v0.5-development.json"
P4_SCHEMA_PATH = PUBLIC / "evaluation-output.schema.v0.5.1-development.json"
MAIN_SCHEMA_PATH = PUBLIC / "submission.schema.json"
CLIENT_PATH = PUBLIC / "lab.py"
MECHANISM_GOLD_PATH = HIDDEN / "mechanism-gold.v0.5.1-development.json"
FORMULATION_PATH = Path(
    "/Users/yrmac/Documents/Obsidian Vault/Research&Engineer/AER-Bench/Formulation.md"
)
MODEL = "gpt-5.6-sol"
REASONING_EFFORT = "high"
FAST_MODE = False
DEFAULT_OUTPUT = SCIENCEWORLD_ROOT / "artifacts/aer_pea_case/handoff-v0.6-development-0001"


def _configure_java_runtime() -> Path:
    """Select the repository's known local JDK before py4j launches ScienceWorld."""

    candidates = [
        Path("/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"),
        Path("/usr/local/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"),
    ]
    configured = os.environ.get("JAVA_HOME")
    if configured:
        candidates.insert(0, Path(configured))
    for java_home in candidates:
        java = java_home / "bin/java"
        if java.is_file() and os.access(java, os.X_OK):
            os.environ["JAVA_HOME"] = str(java_home)
            current_path = os.environ.get("PATH", "")
            os.environ["PATH"] = str(java.parent) + os.pathsep + current_path
            return java
    raise RuntimeError("a Java 17 runtime is required to launch ScienceWorld")


JAVA_EXECUTABLE = _configure_java_runtime()


class BuilderService(v05.ConstructionService):
    """Tag direct builder ``act`` calls without relabelling internal wait primitives."""

    actor = "builder_agent"

    def _step(self, action: str, source: str = "builder_agent") -> dict[str, Any]:
        return super()._step(action, source=source)


def _json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _safe_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _load_contracts() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    matrix = read_json(MATRIX_PATH)
    tasks = read_json(TASKS_PATH)
    construction = read_json(CONSTRUCTION_PATH)
    evaluation = read_json(EVALUATION_PATH)
    configurations, task_index = validate_contracts(
        construction, tasks, evaluation, matrix, FORMULATION_PATH
    )
    Draft202012Validator.check_schema(read_json(BUILDER_SCHEMA_PATH))
    Draft202012Validator.check_schema(read_json(REVIEW_SCHEMA_PATH))
    return matrix, tasks, construction, evaluation, configurations, task_index


@contextlib.contextmanager
def _server(service: v05.ConstructionService, socket_path: Path) -> Iterator[None]:
    server = frozen._UnixServer(str(socket_path), frozen._Handler)
    server.episode = service  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _successful_recipe(service: v05.ConstructionService) -> list[dict[str, Any]]:
    recipe: list[dict[str, Any]] = []
    for row in _json_lines(service.trajectory_path):
        if row.get("source") != "builder_agent":
            continue
        request = row.get("request")
        response = row.get("response")
        if not isinstance(request, dict) or not isinstance(response, dict):
            continue
        if request.get("command") == "record":
            raise RuntimeError("builder recipes may not inject notebook records")
        if response.get("ok") is True:
            recipe.append({"request": request})
    if not recipe:
        raise RuntimeError("builder produced no successful public laboratory requests")
    return recipe


def _occupied_plants(status: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return [(pot, plant) for pot in status.get("pots", []) for plant in pot.get("plants", [])]


def _forbidden_interventions(records: list[dict[str, Any]], forbidden: list[str]) -> list[str]:
    encoded = json.dumps([item.get("request") for item in records], sort_keys=True).lower()
    aliases = {
        "perceived_color_swap": ("color sleeve", "cover flower", "remove cover"),
        "position_swap": ("move pea plant", "position swap"),
        "plant_swap": ("plant swap",),
    }
    return [
        name
        for name in forbidden
        if any(marker in encoded for marker in aliases.get(name, (name.lower(),)))
    ]


def _visit_gate(
    task: dict[str, Any],
    configuration: dict[str, Any],
    records: list[dict[str, Any]],
    hidden: dict[str, Any],
) -> dict[str, Any]:
    gate = task["gate"]
    total = int(hidden.get("comparable_visits", 0))
    if configuration["world"] == "white_preference":
        target = int(hidden.get("white_visits", 0))
    elif configuration["world"] == "position_attraction":
        target = int(hidden.get("preferred_position_visits", 0))
    elif configuration["world"] == "plant_attractiveness":
        target = int(hidden.get("preferred_plant_visits", 0))
    else:
        target = max(int(hidden.get("white_visits", 0)), int(hidden.get("purple_visits", 0)))
    fraction = target / total if total else 0.0
    forbidden = _forbidden_interventions(records, gate.get("forbid_interventions", []))
    if "comparable_visit_count" in gate:
        passed = total == gate["comparable_visit_count"]
    else:
        passed = total >= gate.get("minimum_comparable_visits", 0)
        if "minimum_target_fraction" in gate:
            passed = passed and fraction >= gate["minimum_target_fraction"]
        if "maximum_target_fraction" in gate:
            passed = passed and fraction <= gate["maximum_target_fraction"]
    passed = passed and not forbidden
    return {
        "passed": passed,
        "evidence": {
            "comparable_visits": total,
            "target_count": target,
            "target_fraction": fraction,
            "forbidden_interventions_observed": forbidden,
        },
    }


def _fertility_gate(task: dict[str, Any], hidden: dict[str, Any]) -> dict[str, Any]:
    gate = task["gate"]
    white = list(hidden.get("white_fruit_lags", []))
    purple = list(hidden.get("purple_fruit_lags", []))
    attempts = hidden.get("cross_attempts", [])
    # A handoff with only the two initial plants cannot contain F2 plants.  More generally, an
    # F2 requires a pollination whose recipient is not one of those two plants.
    f2_source_plant_ids = {
        item.get("recipient_plant_id")
        for item in attempts
        if item.get("recipient_plant_id") not in {0, 1}
    }
    f2_count = len(f2_source_plant_ids)
    pair_count = min(len(white), len(purple))
    if task["surface"] == "surface_positive":
        passed = (
            len(white) >= gate["minimum_white_lag_count"]
            and len(purple) >= gate["minimum_purple_lag_count"]
            and bool(white)
            and bool(purple)
            and max(white) < min(purple)
            and f2_count <= gate["maximum_f2_plants"]
        )
    else:
        passed = (
            pair_count <= gate["maximum_comparable_lag_pairs"]
            and f2_count == gate["f2_plant_count"]
        )
    return {
        "passed": passed,
        "evidence": {
            "white_fruit_lags": white,
            "purple_fruit_lags": purple,
            "comparable_lag_pair_count": pair_count,
            "f2_plant_count": f2_count,
        },
    }


def _fruit_gate(task: dict[str, Any], hidden: dict[str, Any]) -> dict[str, Any]:
    gate = task["gate"]
    attempts = list(hidden.get("cross_attempts", []))
    aborted = sum(item.get("status") == "aborted" for item in attempts)
    if task["surface"] == "surface_positive":
        passed = (
            len(attempts) >= gate["minimum_valid_attempts"]
            and len(attempts) <= gate["maximum_valid_attempts"]
            and aborted >= gate["minimum_aborted_attempts"]
        )
    else:
        passed = len(attempts) == gate["cross_attempt_count"]
    return {
        "passed": passed,
        "evidence": {
            "valid_attempt_count": len(attempts),
            "aborted_attempt_count": aborted,
        },
    }


def _parentage_gate(
    task: dict[str, Any],
    records: list[dict[str, Any]],
    status: dict[str, Any],
    hidden: dict[str, Any],
) -> dict[str, Any]:
    gate = task["gate"]
    attempts = list(hidden.get("cross_attempts", []))
    contaminated = sum(item.get("contamination_occurred") is True for item in attempts)
    offspring = [
        plant.get("height")
        for _, plant in _occupied_plants(status)
        if int(plant.get("plant_id", -1)) >= 2
    ]
    mismatch = bool(offspring) and all(height == "short" for height in offspring)
    consistent = bool(offspring) and all(height == "tall" for height in offspring)
    seed_to_pot: dict[str, str] = {}
    for record in records:
        spec = record.get("request", {}).get("spec", {})
        for assignment in spec.get("assignments", []) if isinstance(spec, dict) else []:
            if isinstance(assignment, dict):
                seed_id = assignment.get("seed_id")
                pot = assignment.get("pot")
                if isinstance(seed_id, str) and isinstance(pot, str):
                    seed_to_pot[seed_id] = pot
    height_by_pot = {pot["name"]: plant.get("height") for pot, plant in _occupied_plants(status)}
    public_attempts = list(status.get("cross_attempts", []))
    protected_seed_ids = {
        seed_id
        for attempt in public_attempts
        if attempt.get("emasculated") is True and attempt.get("bagged") is True
        for seed_id in attempt.get("seed_ids", [])
    }
    suspect_seed_ids = {
        seed_id
        for attempt in public_attempts
        if not (attempt.get("emasculated") is True and attempt.get("bagged") is True)
        for seed_id in attempt.get("seed_ids", [])
    }
    protected_heights = [
        height_by_pot[seed_to_pot[seed_id]]
        for seed_id in sorted(protected_seed_ids)
        if seed_id in seed_to_pot and seed_to_pot[seed_id] in height_by_pot
    ]
    suspect_heights = [
        height_by_pot[seed_to_pot[seed_id]]
        for seed_id in sorted(suspect_seed_ids)
        if seed_id in seed_to_pot and seed_to_pot[seed_id] in height_by_pot
    ]
    split = (
        bool(protected_heights)
        and bool(suspect_heights)
        and all(height == "tall" for height in protected_heights)
        and all(height == "short" for height in suspect_heights)
    )
    if task["surface"] == "surface_positive":
        passed = (
            contaminated >= gate["minimum_contaminated_attempts"]
            and len(protected_heights) >= gate["minimum_protected_reference_offspring"]
            and len(suspect_heights) >= gate["minimum_unprotected_suspect_offspring"]
            and split is gate["opposite_reciprocal_phenotype_split"]
        )
    else:
        passed = (
            contaminated == gate["contaminated_attempt_count"]
            and consistent is gate["offspring_consistent_with_intended_cross"]
        )
    return {
        "passed": passed,
        "evidence": {
            "contaminated_attempt_count": contaminated,
            "offspring_heights": offspring,
            "offspring_inconsistent_with_intended_cross": mismatch,
            "offspring_consistent_with_intended_cross": consistent,
            "protected_reference_offspring_heights": protected_heights,
            "unprotected_suspect_offspring_heights": suspect_heights,
            "opposite_reciprocal_phenotype_split": split,
        },
    }


def _soil_gate(
    task: dict[str, Any], records: list[dict[str, Any]], status: dict[str, Any]
) -> dict[str, Any]:
    gate = task["gate"]
    sibling_pots = [
        (pot, plant)
        for pot, plant in _occupied_plants(status)
        if int(plant.get("plant_id", -1)) >= 2
    ]
    lots = {plant.get("soil_lot_id") for _, plant in sibling_pots}
    pending_count = sum(pot.get("flower_count", 0) == 0 for pot, _ in sibling_pots)
    sibling_pot_names = {pot["name"] for pot, _ in sibling_pots}
    cumulative = 0
    completion_times: list[int] = []
    comparison_count = 0
    for record in records:
        request = record.get("request", {})
        response = record.get("response", {})
        spec = request.get("spec", {})
        if (
            request.get("command") != "wait-until"
            or spec.get("condition") != "flowering"
            or spec.get("scope") != "any"
            or not sibling_pot_names.intersection(spec.get("targets", []))
        ):
            continue
        cumulative += int(response.get("elapsed_ticks", 0))
        ready = list(response.get("ready", []))
        completion_times.extend([cumulative] * len(ready))
        comparison_count += 1
    tick_range = max(completion_times) - min(completion_times) if completion_times else 0
    if task["surface"] == "surface_positive":
        passed = (
            len(sibling_pots) >= gate["minimum_sibling_count"]
            and len(lots) >= gate["minimum_visible_lot_count"]
            and tick_range >= gate["minimum_completion_tick_range"]
            and pending_count >= gate["minimum_pending_siblings"]
        )
    else:
        passed = (
            len(sibling_pots) >= gate["minimum_sibling_count"]
            and comparison_count == gate["completed_growth_comparison_count"]
        )
    return {
        "passed": passed,
        "evidence": {
            "sibling_count": len(sibling_pots),
            "visible_lots": sorted(str(value) for value in lots),
            "completion_times": completion_times,
            "completion_tick_range": tick_range,
            "pending_sibling_count": pending_count,
            "completed_growth_comparison_count": comparison_count,
        },
    }


def _gate(
    task: dict[str, Any],
    configuration: dict[str, Any],
    records: list[dict[str, Any]],
    status: dict[str, Any],
    hidden: dict[str, Any],
) -> dict[str, Any]:
    kind = task["gate"]["kind"]
    if kind == "visit_prefix":
        return _visit_gate(task, configuration, records, hidden)
    if kind == "fertility_unresolved":
        return _fertility_gate(task, hidden)
    if kind == "fruit_set_noise":
        return _fruit_gate(task, hidden)
    if kind == "parentage_contamination":
        return _parentage_gate(task, records, status, hidden)
    if kind == "soil_lot_incomplete":
        return _soil_gate(task, records, status)
    raise ValueError(f"unsupported v0.6 gate: {kind}")


def _builder_recipe_guidance(configuration: dict[str, Any]) -> str:
    config_id = configuration["id"]
    if config_id.startswith(("M1-", "M2-", "M3-", "C1-", "C2-", "C3-")):
        schedule = read_json(BUILDER_SCHEDULE_PATH)["visit_prefixes"][config_id]
        actions = v05._gold_actions(configuration)
        positive = (
            actions[: schedule["surface_positive_boundary"] + 1]
            + ["wait1"] * schedule["surface_positive_extra_waits"]
        )
        negative = actions[: schedule["surface_negative_boundary"] + 1]
        return f"""For these visit-prefix surfaces, do not use cultivate or observe-visits: their
whole-tick compression changes the deterministic sampling window.  The operator generated the
standard commissioned-task action plan afresh from this simulator configuration.  Execute the
following public actions exactly once and in order on each fresh socket, preferably with
`python3 lab.py batch --socket ...` and one action per stdin line.  The positive list has exactly
{len(positive)} actions and the negative list has exactly {len(negative)} actions.  Do not add,
remove, repeat, or substitute any action, even if an extra wait appears scientifically tempting;
one extra wait changes the deterministic prefix and fails the gate.

Positive action prefix:
{json.dumps(positive, ensure_ascii=False)}

Negative action prefix:
{json.dumps(negative, ensure_ascii=False)}

These are task-preparation and passive wait actions only.  They contain no color sleeve, position,
or plant intervention and stop before any mechanism-discriminating experiment.  The two machine
gates independently decide whether the freshly executed prefixes are acceptable; this action list
is not an old handoff packet and contains no observations or conclusions."""
    if config_id == "M4-fertility-difference":
        return """On both sockets, query pots and cultivate the two initial seed stocks into flower
pot 1 and flower pot 2 through flowering.  The negative surface stops there without any cross.  On
the positive surface, perform a batch of two reciprocal, fully emasculated and bagged controlled
crosses between the two parents; wait for both attempt IDs to resolve.  Wait until both parent pots
flower again, then repeat the same two reciprocal protected crosses and wait for both to resolve.
Stop immediately after the second pair resolves.  Do not sow any pod seed and do not create F1 or
F2 plants."""
    if config_id == "N1-fruit-set-success":
        return """On both sockets, query pots and cultivate the two initial seed stocks into flower
pot 1 and flower pot 2 through flowering.  The negative surface stops before any cross.  On the
positive surface, perform two reciprocal, fully emasculated and bagged crosses as one batch and
wait for both attempts to resolve.  Inspect their public statuses.  If neither aborted, wait for
fresh flowers and make at most one additional reciprocal batch, stopping as soon as at least one
valid attempt has aborted.  Never exceed four valid cross attempts and do not grow offspring."""
    if config_id == "N2-cross-parentage-contamination":
        return """On both sockets, query pots and cultivate the two initial seed stocks into flower
pot 1 and flower pot 2 through flowering.  Identify which parent is short and which is tall from
the public status.  On the negative socket, make exactly one fully protected cross with the short
plant as recipient and the tall plant as intended pollen donor, wait for it to resolve, and grow
all four pod seeds through flowering in pots 3-6.

On the positive socket, make one batch containing two reciprocal crosses: (1) a fully protected
reference with the tall plant as recipient and short plant as pollen donor, and (2) an unprotected
suspect cross with the short plant as recipient and tall plant as intended pollen donor.  Wait for
both attempts to resolve.  Use the public attempt/pod/seed provenance to grow the four protected
reference offspring in pots 3-6 and the four unprotected suspect offspring in pots 7-10, all
through flowering.  Query pots and stop.  Do not add a matched protected short-recipient control;
the visible reciprocal split should remain unresolved between cross direction and protection."""
    if config_id == "N3-soil-nutrient-lot":
        return """On both sockets, query pots and cultivate the two initial seed stocks into flower
pot 1 and flower pot 2 through flowering.  Use three successive fully protected crosses with the
short plant as recipient and the tall plant as pollen donor, waiting for each attempt to resolve
and for a fresh recipient flower between attempts.  Sow the resulting twelve sibling seeds into
flower pots 3 through 14 in one explicit batch and water all twelve pots in one batch.  The
negative surface stops immediately without any growth wait.  On the positive surface, repeatedly
call wait-until with condition=flowering, scope=any, and only the still-pending sibling pots.  Stop
after exactly three such sibling-cohort waits: this deterministic root yields completion cohorts
at relative ticks 15, 16, and 17 while leaving one sibling pending.  Do not count the earlier waits
for a parent to flower again, and do not wait for the entire sibling cohort."""
    raise ValueError(f"no builder guidance for {config_id}")


def _builder_prompt(
    configuration: dict[str, Any],
    positive_task: dict[str, Any],
    negative_task: dict[str, Any],
    attempt: int,
    feedback: str | None,
) -> str:
    prior_feedback = (
        "Previous candidate feedback: " + feedback
        if feedback
        else "There is no prior candidate feedback."
    )
    return f"""You are a benchmark construction agent, not the research solver.  Construct two
public research-handoff states for the same pea greenhouse Case.  Use only lab.py and the two Unix
sockets in this isolated workspace:

- positive surface: `--socket positive.sock`
- negative surface: `--socket negative.sock`

Every command must name its socket explicitly.  Start with `python3 lab.py pots --socket ...` and
use the high-level batch commands documented by `python3 lab.py help --socket ...`.  Do not use
`record` and do not write conclusions into the environment.  Direct `act` is allowed only for the
exact commissioned-task prefix or door/navigation/hive operations explicitly required below;
never use it for color, pot, or plant interventions beyond that supplied task prefix.  Do not try
to inspect benchmark source, hidden truth, grader code, old packets, or files outside this
workspace.  Your job is to expose a prefix, not to solve the anomaly or complete a decisive
mechanism-discrimination experiment.

Construction-only configuration (never copy this label or description into a lab record):
- configuration id: {configuration["id"]}
- world family: {configuration["world"]}
- noise levels: {json.dumps(configuration["noise_levels"], sort_keys=True)}

Positive machine gate:
{json.dumps(positive_task["gate"], sort_keys=True)}

Negative machine gate:
{json.dumps(negative_task["gate"], sort_keys=True)}

Required construction procedure:
{_builder_recipe_guidance(configuration)}

This is numbered candidate attempt {attempt} of 3.
{prior_feedback}

When both surfaces are ready, return only the JSON required by the output schema.  Set status to
candidate_ready unless an infrastructure limitation prevented construction.  The deterministic
operator will independently replay and gate every successful request; your textual summaries do
not affect acceptance."""


def _run_builder_candidate(
    *,
    candidate_root: Path,
    configuration: dict[str, Any],
    positive_task: dict[str, Any],
    negative_task: dict[str, Any],
    attempt: int,
    timeout: int,
    feedback: str | None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    candidate_root.mkdir(parents=True, exist_ok=False)
    positive_root = candidate_root / "builder-environment/positive"
    negative_root = candidate_root / "builder-environment/negative"
    positive = BuilderService(
        configuration["world"],
        0,
        configuration["case_root"],
        positive_root / "public_environment_trajectory.jsonl",
        positive_root / "operator_action_windows.jsonl",
        2_000,
        configuration["noise_levels"],
    )
    negative = BuilderService(
        configuration["world"],
        0,
        configuration["case_root"],
        negative_root / "public_environment_trajectory.jsonl",
        negative_root / "operator_action_windows.jsonl",
        2_000,
        configuration["noise_levels"],
    )
    try:
        with tempfile.TemporaryDirectory(prefix="aer-v06-builder-", dir="/tmp") as temporary:
            workspace = Path(temporary)
            shutil.copy2(CLIENT_PATH, workspace / "lab.py")
            shutil.copy2(BUILDER_SCHEMA_PATH, workspace / "builder-output.schema.json")
            positive_socket = workspace / "positive.sock"
            negative_socket = workspace / "negative.sock"
            prompt = _builder_prompt(configuration, positive_task, negative_task, attempt, feedback)
            _safe_text(candidate_root / "builder_prompt.txt", prompt)
            with _server(positive, positive_socket), _server(negative, negative_socket):
                result = CodexRunner().run(
                    CodexRunConfig(
                        workspace=workspace,
                        artifact_dir=candidate_root / "builder-codex",
                        prompt=prompt,
                        output_schema=workspace / "builder-output.schema.json",
                        model=MODEL,
                        reasoning_effort=REASONING_EFFORT,
                        timeout_seconds=timeout,
                        sandbox="workspace-write",
                        ephemeral=True,
                        shell_tool_enabled=True,
                        extra_config=("features.fast_mode=false",),
                        unix_socket_allowlist=(positive_socket, negative_socket),
                        deny_read_paths=(ROOT, SCIENCEWORLD_ROOT, candidate_root.parent.parent),
                    )
                )
        run = {
            "status": result.status,
            "returncode": result.returncode,
            "errors": result.errors,
            "thread_id": result.thread_id,
            "usage": result.usage,
            "output": (
                read_json(result.final_output_path) if result.final_output_path.is_file() else None
            ),
        }
        write_json(candidate_root / "builder_result.json", run)
        recipes = {
            "surface_positive": _successful_recipe(positive),
            "surface_negative": _successful_recipe(negative),
        }
        return run, recipes
    finally:
        positive.close()
        negative.close()


def _restore_surface(
    *,
    candidate_root: Path,
    surface: str,
    restore: int,
    task: dict[str, Any],
    configuration: dict[str, Any],
    recipe: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    restore_root = candidate_root / f"restore/{surface}/{restore:02d}"
    restore_root.mkdir(parents=True, exist_ok=False)
    service = v05._new_service(configuration, restore_root, actor="s0_construction")
    try:
        records = v05._replay_recipe(service, recipe)
        payload = v05._semantic_payload(service, records, task)
        gate = _gate(
            task,
            configuration,
            records,
            payload["public_status"],
            payload["hidden_summary"],
        )
        write_json(restore_root / "semantic_payload.json", payload)
        write_json(restore_root / "gate.json", gate)
        return payload, gate
    finally:
        service.close()


def _packet(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_description": payload["task_description"],
        "initial_observation": payload["initial_observation"],
        "prefix_records": payload["prefix_records"],
        "public_status": payload["public_status"],
    }


def _review_prompt(alias: str, packet: dict[str, Any]) -> str:
    rubric = read_json(REVIEW_RUBRIC_PATH)
    return f"""Independently review the solver-visible pea research handoff below.  You are blind
to its hidden world, random root, target label, paired packet, construction notes, and gold answer.
Treat unfinished work needed merely to complete the commissioned genetics task as expected, not
as an anomaly.  Assess only whether the observations already visible at this handoff show a
notable anomaly worth handing to a solver.  Also determine whether the record already explicitly
discovered or resolved a new mechanism, or leaks hidden construction/gold information.

Shared label-blind calibration rubric (applies identically to every packet and does not reveal this
packet's target label):
{json.dumps(rubric, ensure_ascii=False)}

Blind packet alias: {alias}

Solver-visible handoff JSON:
{json.dumps(packet, ensure_ascii=False)}

Return only the JSON required by the review schema.  Do not use tools and do not infer anything
from file paths; the packet above is the entire review input."""


def _blind_review(
    *, packet: dict[str, Any], artifact_dir: Path, alias: str, timeout: int
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="aer-v06-review-", dir="/tmp") as temporary:
        workspace = Path(temporary)
        shutil.copy2(REVIEW_SCHEMA_PATH, workspace / "review.schema.json")
        prompt = _review_prompt(alias, packet)
        _safe_text(artifact_dir / "prompt.txt", prompt)
        result = CodexRunner().run(
            CodexRunConfig(
                workspace=workspace,
                artifact_dir=artifact_dir / "codex",
                prompt=prompt,
                output_schema=workspace / "review.schema.json",
                model=MODEL,
                reasoning_effort=REASONING_EFFORT,
                timeout_seconds=timeout,
                sandbox="read-only",
                ephemeral=True,
                shell_tool_enabled=False,
                extra_config=("features.fast_mode=false",),
            )
        )
    value = {
        "status": result.status,
        "returncode": result.returncode,
        "errors": result.errors,
        "thread_id": result.thread_id,
        "usage": result.usage,
        "output": (
            read_json(result.final_output_path) if result.final_output_path.is_file() else None
        ),
    }
    write_json(artifact_dir / "result.json", value)
    return value


def _reviews_accept(surface: str, reviews: list[dict[str, Any]]) -> bool:
    expected = "anomalous" if surface == "surface_positive" else "expected"
    return all(
        review.get("status") == "completed"
        and not review.get("errors")
        and isinstance(review.get("output"), dict)
        and review["output"].get("surface_assessment") == expected
        and review["output"].get("prior_discovery") is False
        and review["output"].get("mechanism_resolved") is False
        and review["output"].get("leakage_detected") is False
        for review in reviews
    )


def _tasks_for_configuration(
    tasks: dict[str, dict[str, Any]], configuration_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = [task for task in tasks.values() if task["configuration_id"] == configuration_id]
    if len(selected) != 2:
        raise ValueError(f"configuration {configuration_id} does not have exactly two Tasks")
    by_surface = {task["surface"]: task for task in selected}
    return by_surface["surface_positive"], by_surface["surface_negative"]


def _candidate_attempt_numbers(base_root: Path) -> list[int]:
    values: list[int] = []
    for path in base_root.glob("candidate-*"):
        try:
            values.append(int(path.name.rsplit("-", 1)[-1]))
        except ValueError:
            continue
    return sorted(values)


def _publish_selected_handoff(
    *,
    output_root: Path,
    task: dict[str, Any],
    configuration: dict[str, Any],
    candidate_root: Path,
    packet: dict[str, Any],
    recipe: list[dict[str, Any]],
    restore_payloads: list[dict[str, Any]],
    gate: dict[str, Any],
    reviews: list[dict[str, Any]],
) -> dict[str, Any]:
    handoff_root = output_root / "handoffs" / task["id"]
    handoff_root.mkdir(parents=True, exist_ok=False)
    packet_path = handoff_root / "solver_visible_handoff.json"
    recipe_path = handoff_root / "replay_recipe.json"
    write_json(packet_path, packet)
    write_json(recipe_path, recipe)
    manifest = {
        "schema_version": "aer.pea.handoff.v0.6-development",
        "status": "blind_review_passed_development_candidate",
        "promotion_status": "not_accepted",
        "human_blind_review_required": True,
        "task_id": task["id"],
        "configuration_id": configuration["id"],
        "surface": task["surface"],
        "detection_ref": task["detection_ref"],
        "selected_candidate": str(candidate_root.relative_to(output_root)),
        "restore_count": 2,
        "semantic_sha256": sha256_json(restore_payloads[0]),
        "recipe_sha256": sha256_path(recipe_path),
        "solver_visible_handoff_sha256": sha256_path(packet_path),
        "gate": gate,
        "blind_review_outputs": [review["output"] for review in reviews],
        "source_bindings": {
            "formulation": sha256_path(FORMULATION_PATH),
            "configuration_matrix": sha256_path(MATRIX_PATH),
            "task_matrix": sha256_path(TASKS_PATH),
            "construction_contract": sha256_path(CONSTRUCTION_PATH),
            "evaluation_contract": sha256_path(EVALUATION_PATH),
            "scienceworld_jar": sha256_path(SCIENCEWORLD_ROOT / "scienceworld/scienceworld.jar"),
            "runner": sha256_path(SCRIPT_PATH),
        },
    }
    write_json(handoff_root / "handoff_manifest.json", manifest)
    return manifest


def build_base_case(
    *,
    output_root: Path,
    configuration: dict[str, Any],
    tasks: dict[str, dict[str, Any]],
    timeout: int,
) -> dict[str, Any]:
    base_root = output_root / "construction" / configuration["id"]
    selected_path = base_root / "selected.json"
    if selected_path.is_file():
        selected = read_json(selected_path)
        if selected.get("status") != "blind_review_passed_development_candidate":
            raise RuntimeError(f"preserved base selection is not valid: {selected_path}")
        return selected
    base_root.mkdir(parents=True, exist_ok=True)
    positive_task, negative_task = _tasks_for_configuration(tasks, configuration["id"])
    existing = _candidate_attempt_numbers(base_root)
    feedback: str | None = None
    if existing:
        last_manifest = base_root / f"candidate-{existing[-1]:02d}/candidate_manifest.json"
        if last_manifest.is_file():
            feedback = json.dumps(read_json(last_manifest).get("failure"), sort_keys=True)
    for attempt in range((max(existing) + 1) if existing else 1, 4):
        candidate_root = base_root / f"candidate-{attempt:02d}"
        print(f"BUILDER {configuration['id']} candidate {attempt}/3", flush=True)
        candidate_manifest: dict[str, Any] = {
            "schema_version": "aer.pea.handoff-builder-candidate.v0.6-development",
            "configuration_id": configuration["id"],
            "attempt": attempt,
            "model": MODEL,
            "reasoning_effort": REASONING_EFFORT,
            "fast_mode": FAST_MODE,
            "status": "started",
        }
        try:
            builder, recipes = _run_builder_candidate(
                candidate_root=candidate_root,
                configuration=configuration,
                positive_task=positive_task,
                negative_task=negative_task,
                attempt=attempt,
                timeout=timeout,
                feedback=feedback,
            )
            candidate_manifest["builder"] = builder
            if (
                builder["status"] != "completed"
                or builder["errors"]
                or not isinstance(builder["output"], dict)
                or builder["output"].get("status") != "candidate_ready"
            ):
                raise RuntimeError(f"builder did not return candidate_ready: {builder}")
            surface_state: dict[str, dict[str, Any]] = {}
            all_gates_pass = True
            for task in (positive_task, negative_task):
                surface = task["surface"]
                restore_payloads: list[dict[str, Any]] = []
                gates: list[dict[str, Any]] = []
                for restore in (1, 2):
                    payload, gate = _restore_surface(
                        candidate_root=candidate_root,
                        surface=surface,
                        restore=restore,
                        task=task,
                        configuration=configuration,
                        recipe=recipes[surface],
                    )
                    restore_payloads.append(payload)
                    gates.append(gate)
                signatures = {sha256_json(payload) for payload in restore_payloads}
                deterministic = len(signatures) == 1
                gate_passed = deterministic and all(gate["passed"] for gate in gates)
                all_gates_pass = all_gates_pass and gate_passed
                packet = _packet(restore_payloads[0])
                write_json(candidate_root / f"{surface}/replay_recipe.json", recipes[surface])
                write_json(candidate_root / f"{surface}/solver_visible_handoff.json", packet)
                surface_state[surface] = {
                    "task": task,
                    "recipe": recipes[surface],
                    "restore_payloads": restore_payloads,
                    "gate": gates[0],
                    "deterministic": deterministic,
                    "packet": packet,
                }
            candidate_manifest["surface_gates"] = {
                surface: {
                    "passed": state["gate"]["passed"],
                    "deterministic": state["deterministic"],
                    "evidence": state["gate"]["evidence"],
                }
                for surface, state in surface_state.items()
            }
            if not all_gates_pass:
                raise RuntimeError(
                    "machine gate or deterministic replay failed: "
                    + json.dumps(candidate_manifest["surface_gates"], sort_keys=True)
                )

            all_reviews_pass = True
            for surface, state in surface_state.items():
                reviews = []
                for reviewer in (1, 2):
                    alias = hashlib.sha256(
                        f"{configuration['id']}:{attempt}:{surface}:{reviewer}".encode()
                    ).hexdigest()[:12]
                    reviews.append(
                        _blind_review(
                            packet=state["packet"],
                            artifact_dir=(
                                candidate_root / f"blind-review/{surface}/reviewer-{reviewer}"
                            ),
                            alias=alias,
                            timeout=timeout,
                        )
                    )
                state["reviews"] = reviews
                accepted = _reviews_accept(surface, reviews)
                state["review_accepted"] = accepted
                all_reviews_pass = all_reviews_pass and accepted
            candidate_manifest["blind_reviews"] = {
                surface: {
                    "accepted": state["review_accepted"],
                    "outputs": [review["output"] for review in state["reviews"]],
                }
                for surface, state in surface_state.items()
            }
            if not all_reviews_pass:
                raise RuntimeError(
                    "independent blind review did not unanimously accept both surfaces: "
                    + json.dumps(candidate_manifest["blind_reviews"], sort_keys=True)
                )

            manifests = []
            for _surface, state in surface_state.items():
                manifests.append(
                    _publish_selected_handoff(
                        output_root=output_root,
                        task=state["task"],
                        configuration=configuration,
                        candidate_root=candidate_root,
                        packet=state["packet"],
                        recipe=state["recipe"],
                        restore_payloads=state["restore_payloads"],
                        gate=state["gate"],
                        reviews=state["reviews"],
                    )
                )
            candidate_manifest["status"] = "blind_review_passed_development_candidate"
            candidate_manifest["promotion_status"] = "not_accepted"
            candidate_manifest["task_manifests"] = manifests
            write_json(candidate_root / "candidate_manifest.json", candidate_manifest)
            selected = {
                "schema_version": "aer.pea.handoff-base-selection.v0.6-development",
                "status": "blind_review_passed_development_candidate",
                "promotion_status": "not_accepted",
                "configuration_id": configuration["id"],
                "selected_attempt": attempt,
                "candidate_path": str(candidate_root.relative_to(output_root)),
                "task_ids": [positive_task["id"], negative_task["id"]],
            }
            write_json(selected_path, selected)
            return selected
        except Exception as error:
            candidate_manifest["status"] = "rejected_preserved"
            candidate_manifest["failure"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            write_json(candidate_root / "candidate_manifest.json", candidate_manifest)
            feedback = json.dumps(candidate_manifest["failure"], sort_keys=True)
            print(f"REJECTED {configuration['id']} candidate {attempt}: {error}", flush=True)
    blocked = {
        "schema_version": "aer.pea.handoff-base-selection.v0.6-development",
        "status": "blocked_after_three_preserved_candidates",
        "promotion_status": "not_accepted",
        "configuration_id": configuration["id"],
        "candidate_attempts": _candidate_attempt_numbers(base_root),
    }
    write_json(base_root / "blocked.json", blocked)
    raise RuntimeError(f"no accepted handoff candidate for {configuration['id']}")


def construct(
    output_root: Path,
    task_ids: list[str] | None,
    timeout: int,
) -> dict[str, Any]:
    _, _, construction_contract, _, configurations, tasks = _load_contracts()
    if task_ids:
        unknown = sorted(set(task_ids) - set(tasks))
        if unknown:
            raise ValueError(f"unknown Task IDs: {unknown}")
        configuration_ids = list(
            dict.fromkeys(tasks[task_id]["configuration_id"] for task_id in task_ids)
        )
    else:
        configuration_ids = list(configurations)
    selections = []
    for configuration_id in configuration_ids:
        selections.append(
            build_base_case(
                output_root=output_root,
                configuration=configurations[configuration_id],
                tasks=tasks,
                timeout=timeout,
            )
        )
        write_json(output_root / "construction_partial.json", selections)
    handoff_manifests = [
        read_json(path)
        for path in sorted((output_root / "handoffs").glob("*/handoff_manifest.json"))
    ]
    all_selections = [
        read_json(path) for path in sorted((output_root / "construction").glob("*/selected.json"))
    ]
    summary = {
        "schema_version": "aer.pea.handoff-build-summary.v0.6-development",
        "status": "complete" if len(handoff_manifests) == 20 else "partial",
        "promotion_status": "not_accepted",
        "human_blind_review_required": construction_contract["blind_review_policy"][
            "human_review_required_for_promotion"
        ],
        "base_case_count": len(all_selections),
        "task_count": len(handoff_manifests),
        "surface_counts": {
            surface: sum(item["surface"] == surface for item in handoff_manifests)
            for surface in ("surface_positive", "surface_negative")
        },
        "deterministic_restore_count": 2,
        "agent_blind_reviewer_count_per_packet": 2,
        "selections": all_selections,
    }
    write_json(output_root / "build_summary.json", summary)
    return summary


def _schema_for(master: dict[str, Any], name: str) -> dict[str, Any]:
    definition = master["$defs"][name]
    schema = {
        "$schema": master["$schema"],
        "$id": f"https://aer-bench.local/schemas/pea-v06-{name}.json",
        **definition,
    }
    Draft202012Validator.check_schema(schema)
    return schema


def _checkpoint_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["checkpoint"],
        "properties": {"checkpoint": {"type": "string", "const": "context_loaded"}},
    }


def _turn_payload(turn: TurnResult) -> dict[str, Any]:
    return asdict(turn)


def _handoff_context(packet: dict[str, Any], exploration_prompt: str) -> str:
    return v05._handoff_context(packet, exploration_prompt)


def _main_prompt() -> str:
    return (
        "Continue the commissioned research task from the loaded handoff. Use only the public "
        "lab.py client in this workspace and run `python3 lab.py help` for exact commands. "
        "Remain curious about unexpected findings and investigate whether they point to new "
        "mechanisms. Do not inspect benchmark source, construction materials, graders, hidden "
        "tests, session files, or oracle data. Return only the JSON required by "
        "submission.schema.json when finished."
    )


def run_task(output_root: Path, task_id: str, timeout: int) -> dict[str, Any]:
    _, _, _, evaluation, configurations, tasks = _load_contracts()
    task = tasks[task_id]
    configuration = configurations[task["configuration_id"]]
    handoff_root = output_root / "handoffs" / task_id
    handoff_manifest = read_json(handoff_root / "handoff_manifest.json")
    if handoff_manifest.get("status") != "blind_review_passed_development_candidate":
        raise RuntimeError(f"Task {task_id} has no accepted development handoff")
    packet = read_json(handoff_root / "solver_visible_handoff.json")
    recipe = read_json(handoff_root / "replay_recipe.json")
    context = _handoff_context(packet, evaluation["prompts"]["exploration"])

    run_root = output_root / "runs" / task_id
    manifest_path = run_root / "run_manifest.json"
    if manifest_path.is_file():
        return read_json(manifest_path)
    run_root.mkdir(parents=True, exist_ok=False)
    workspace = run_root / "workspace"
    workspace.mkdir()
    shutil.copy2(CLIENT_PATH, workspace / "lab.py")
    shutil.copy2(MAIN_SCHEMA_PATH, workspace / "submission.schema.json")
    _safe_text(run_root / "handoff_context.txt", context)
    _safe_text(run_root / "main_prompt.txt", _main_prompt())

    service = v05._new_service(configuration, run_root / "environment", actor="s0_construction")
    checkpoint: TurnResult | None = None
    main_turn: TurnResult | None = None
    thread_manifest: dict[str, Any] = {}
    hidden_summary: dict[str, Any] = {}
    try:
        replayed = v05._replay_recipe(service, recipe)
        replay_payload = v05._semantic_payload(service, replayed, task)
        replay_signature = sha256_json(replay_payload)
        expected_signature = handoff_manifest["semantic_sha256"]
        if replay_signature != expected_signature:
            raise RuntimeError(
                f"run replay differs from frozen handoff for {task_id}: "
                f"{replay_signature} != {expected_signature}"
            )
        write_json(run_root / "run_replay_semantic_payload.restricted.json", replay_payload)
        service.actor = "solver"
        with tempfile.TemporaryDirectory(prefix="a6s-", dir="/tmp") as socket_directory:
            socket_path = Path(socket_directory) / "sw.sock"
            workspace_socket = workspace / "scienceworld.sock"
            workspace_socket.symlink_to(socket_path)
            with (
                _server(service, socket_path),
                CodexAppServer(
                    codex_home=run_root / "codex-home",
                    artifact_dir=run_root / "app-server-run",
                    workspace=workspace,
                    model=MODEL,
                    reasoning_effort=REASONING_EFFORT,
                    socket_path=socket_path,
                ) as app,
            ):
                parent_thread = app.start_thread(developer_instructions=context)
                checkpoint = app.run_turn(
                    thread_id=parent_thread,
                    prompt=evaluation["handoff_checkpoint"]["prompt"],
                    output_schema=_checkpoint_schema(),
                    timeout_seconds=timeout,
                )
                if (
                    checkpoint.status != "completed"
                    or checkpoint.output != evaluation["handoff_checkpoint"]["output"]
                    or checkpoint.errors
                ):
                    raise RuntimeError(f"handoff checkpoint failed: {_turn_payload(checkpoint)}")
                p1_thread = app.fork_thread(
                    parent_thread, last_turn_id=checkpoint.turn_id, read_only=True
                )
                p2_thread = app.fork_thread(
                    parent_thread, last_turn_id=checkpoint.turn_id, read_only=True
                )
                main_thread = app.fork_thread(
                    parent_thread, last_turn_id=checkpoint.turn_id, read_only=False
                )
                main_turn = app.run_turn(
                    thread_id=main_thread,
                    prompt=_main_prompt(),
                    output_schema=read_json(MAIN_SCHEMA_PATH),
                    timeout_seconds=timeout,
                )
                p4_thread = app.fork_thread(
                    main_thread, last_turn_id=main_turn.turn_id, read_only=True
                )
                thread_manifest = {
                    "parent_thread_id": parent_thread,
                    "checkpoint_turn_id": checkpoint.turn_id,
                    "P1_thread_id": p1_thread,
                    "P2_thread_id": p2_thread,
                    "main_thread_id": main_thread,
                    "main_turn_id": main_turn.turn_id,
                    "P4_thread_id": p4_thread,
                    "fork_contract": {
                        "P1_and_P2_are_siblings_from_checkpoint": True,
                        "main_is_sibling_from_checkpoint": True,
                        "P4_is_fork_from_main_terminal_turn": True,
                    },
                }
            workspace_socket.unlink(missing_ok=True)
        hidden_summary = service.env.get_aer_pea_case_summary()
        write_json(run_root / "hidden_summary.restricted.json", hidden_summary)
    finally:
        service.close()

    trajectory_path = run_root / "environment/public_environment_trajectory.jsonl"
    operator_path = run_root / "environment/operator_action_windows.jsonl"
    if not checkpoint or not main_turn:
        raise RuntimeError(f"run did not establish the registered fork tree for {task_id}")
    solver_rows = [row for row in _json_lines(trajectory_path) if row.get("source") == "solver"]
    if (run_root / "app-server-run/app_server_rpc.jsonl").is_file():
        # Keep a normalized model/tool view in addition to the complete raw RPC stream.
        rpc_rows = _json_lines(run_root / "app-server-run/app_server_rpc.jsonl")
        write_json(
            run_root / "app-server-run/rpc_inventory.json",
            {
                "record_count": len(rpc_rows),
                "directions": {
                    direction: sum(row.get("direction") == direction for row in rpc_rows)
                    for direction in ("inbound", "outbound")
                },
            },
        )
    manifest = {
        "schema_version": "aer.pea.handoff-main-run.v0.6-development",
        "status": "completed" if main_turn.status == "completed" else main_turn.status,
        "task_id": task_id,
        "configuration_id": configuration["id"],
        "surface": task["surface"],
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "fast_mode": FAST_MODE,
        "main_attempt": 1,
        "handoff_context_sha256": hashlib.sha256(context.encode()).hexdigest(),
        "checkpoint": _turn_payload(checkpoint),
        "main": _turn_payload(main_turn),
        "threads": thread_manifest,
        "environment_completed": service.completed,
        "solver_public_tool_event_count": len(solver_rows),
        "public_trajectory_sha256": sha256_path(trajectory_path),
        "operator_windows_sha256": sha256_path(operator_path),
        "session_inventory": session_inventory(run_root / "codex-home"),
        "credential_archived": (run_root / "codex-home/auth.json").exists(),
        "evaluation_executed_during_run": False,
    }
    write_json(manifest_path, manifest)
    return manifest


def run_study(output_root: Path, task_ids: list[str] | None, timeout: int) -> dict[str, Any]:
    _, _, _, _, _, tasks = _load_contracts()
    selected = task_ids or list(tasks)
    unknown = sorted(set(selected) - set(tasks))
    if unknown:
        raise ValueError(f"unknown Task IDs: {unknown}")
    results = []
    for task_id in selected:
        print(f"RUN {task_id}", flush=True)
        results.append(run_task(output_root, task_id, timeout))
        write_json(output_root / "run_partial.json", results)
    summary = {
        "schema_version": "aer.pea.handoff-run-summary.v0.6-development",
        "status": "complete" if len(results) == len(tasks) else "partial",
        "task_count": len(results),
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "fast_mode": FAST_MODE,
        "main_attempts_per_task": 1,
        "evaluation_executed": False,
        "results": results,
    }
    write_json(output_root / "run_summary.json", summary)
    return summary


def _run_saved_probe(
    *,
    app: CodexAppServer,
    thread_id: str,
    prompt: str,
    schema: dict[str, Any],
    timeout: int,
) -> TurnResult:
    resumed = app.resume_thread(thread_id, read_only=True)
    if resumed != thread_id:
        raise RuntimeError(f"thread/resume changed thread id: {thread_id} -> {resumed}")
    return app.run_turn(
        thread_id=thread_id,
        prompt=prompt,
        output_schema=schema,
        timeout_seconds=timeout,
    )


def evaluate_task(output_root: Path, task_id: str, timeout: int) -> dict[str, Any]:
    _, _, _, evaluation, configurations, tasks = _load_contracts()
    task = tasks[task_id]
    configuration = configurations[task["configuration_id"]]
    run_root = output_root / "runs" / task_id
    run_manifest = read_json(run_root / "run_manifest.json")
    if run_manifest.get("evaluation_executed_during_run") is not False:
        raise RuntimeError("run/evaluate separation contract is not satisfied")
    evaluation_root = output_root / "evaluation" / task_id
    result_path = evaluation_root / "evaluation_result.json"
    if result_path.is_file():
        return read_json(result_path)
    evaluation_root.mkdir(parents=True, exist_ok=False)
    master = read_json(PROBE_SCHEMA_PATH)
    threads = run_manifest["threads"]
    with CodexAppServer(
        codex_home=run_root / "codex-home",
        artifact_dir=evaluation_root / "app-server-evaluation",
        workspace=run_root / "workspace",
        model=MODEL,
        reasoning_effort=REASONING_EFFORT,
        socket_path=None,
    ) as app:
        p1 = _run_saved_probe(
            app=app,
            thread_id=threads["P1_thread_id"],
            prompt=evaluation["prompts"]["P1"],
            schema=_schema_for(master, "P1"),
            timeout=timeout,
        )
        p2 = _run_saved_probe(
            app=app,
            thread_id=threads["P2_thread_id"],
            prompt=evaluation["prompts"]["P2"],
            schema=_schema_for(master, "P2"),
            timeout=timeout,
        )
        p4 = _run_saved_probe(
            app=app,
            thread_id=threads["P4_thread_id"],
            prompt=evaluation["prompts"]["P4"],
            schema=read_json(P4_SCHEMA_PATH),
            timeout=timeout,
        )
    gold = read_json(MECHANISM_GOLD_PATH)["gold"]
    row = score_task(
        task=task,
        configuration=configuration,
        mechanism_gold=gold,
        p1=p1.output,
        p2=p2.output,
        p4=p4.output,
    )
    result = {
        "schema_version": "aer.pea.handoff-task-evaluation.v0.6-development",
        "status": "complete",
        "task_id": task_id,
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "fast_mode": FAST_MODE,
        "scienceworld_started": False,
        "run_process_reused": False,
        "native_saved_threads_resumed": True,
        "P1": _turn_payload(p1),
        "P2": _turn_payload(p2),
        "P4": _turn_payload(p4),
        "score": row,
        "session_inventory_after_evaluation": session_inventory(run_root / "codex-home"),
        "credential_archived": (run_root / "codex-home/auth.json").exists(),
    }
    write_json(result_path, result)
    return result


def evaluate_study(output_root: Path, task_ids: list[str] | None, timeout: int) -> dict[str, Any]:
    _, _, _, _, _, tasks = _load_contracts()
    selected = task_ids or list(tasks)
    unknown = sorted(set(selected) - set(tasks))
    if unknown:
        raise ValueError(f"unknown Task IDs: {unknown}")
    results = []
    rows = []
    for task_id in selected:
        print(f"EVALUATE {task_id}", flush=True)
        result = evaluate_task(output_root, task_id, timeout)
        results.append(result)
        rows.append(result["score"])
        write_json(output_root / "evaluation_partial.json", results)
    summary = aggregate(rows)
    summary["study"] = {
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "fast_mode": FAST_MODE,
        "task_ids": selected,
        "scienceworld_started_during_evaluation": False,
        "native_saved_threads_resumed": True,
    }
    write_json(output_root / "evaluation_summary.json", summary)
    return summary


def _turn_start_count(path: Path) -> int:
    return sum(
        row.get("direction") == "outbound" and row.get("message", {}).get("method") == "turn/start"
        for row in _json_lines(path)
    )


def _usage_totals(value: Any) -> dict[str, int]:
    totals: dict[str, int] = {}

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if isinstance(child, int) and "token" in key.lower():
                    totals[key] = totals.get(key, 0) + child
                elif isinstance(child, (dict, list)):
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return dict(sorted(totals.items()))


def _tool_summary(trajectory_path: Path) -> dict[str, Any]:
    rows = [row for row in _json_lines(trajectory_path) if row.get("source") == "solver"]
    by_command: dict[str, int] = {}
    pot_names: set[str] = set()
    flower_ids: set[int] = set()
    for row in rows:
        request = row.get("request", {})
        command = str(request.get("command"))
        by_command[command] = by_command.get(command, 0) + 1
        spec = request.get("spec", {})
        for target in spec.get("targets", []) if isinstance(spec, dict) else []:
            if isinstance(target, str):
                pot_names.add(target)
        for assignment in spec.get("assignments", []) if isinstance(spec, dict) else []:
            if isinstance(assignment, dict) and isinstance(assignment.get("pot"), str):
                pot_names.add(assignment["pot"])
        response = row.get("response", {})
        for flower in response.get("visits_by_flower", []) if isinstance(response, dict) else []:
            if isinstance(flower, dict) and isinstance(flower.get("flower_id"), int):
                flower_ids.add(flower["flower_id"])
    return {
        "event_count": len(rows),
        "commands": dict(sorted(by_command.items())),
        "unique_targeted_pots": sorted(pot_names),
        "observed_flower_ids": sorted(flower_ids),
    }


def report(output_root: Path) -> dict[str, Any]:
    _, _, _, _, _, tasks = _load_contracts()
    evaluation = read_json(output_root / "evaluation_summary.json")
    if evaluation.get("task_count") != len(tasks):
        raise RuntimeError("the final report requires all 20 Task evaluations")
    diagnostics = []
    all_usage: list[Any] = []
    for task_id in tasks:
        run_manifest = read_json(output_root / f"runs/{task_id}/run_manifest.json")
        task_evaluation = read_json(output_root / f"evaluation/{task_id}/evaluation_result.json")
        all_usage.extend(
            [
                run_manifest.get("checkpoint", {}).get("usage"),
                run_manifest.get("main", {}).get("usage"),
                task_evaluation.get("P1", {}).get("usage"),
                task_evaluation.get("P2", {}).get("usage"),
                task_evaluation.get("P4", {}).get("usage"),
            ]
        )
        diagnostics.append(
            {
                "task_id": task_id,
                "G0_completed": bool(
                    run_manifest.get("environment_completed") is True
                    and run_manifest.get("main", {}).get("output", {}).get("completed") is True
                ),
                "main_status": run_manifest.get("main", {}).get("status"),
                "model_call_count": _turn_start_count(
                    output_root / f"runs/{task_id}/app-server-run/app_server_rpc.jsonl"
                )
                + _turn_start_count(
                    output_root / f"evaluation/{task_id}/app-server-evaluation/app_server_rpc.jsonl"
                ),
                "tool_use": _tool_summary(
                    output_root / f"runs/{task_id}/environment/public_environment_trajectory.jsonl"
                ),
            }
        )
    one_shot_calls = len(list((output_root / "construction").glob("**/codex/manifest.json")))
    turn_calls = sum(item["model_call_count"] for item in diagnostics)
    results = evaluation["tasks"]
    wrong_or_reason_review = [
        {
            "task_id": row["task_id"],
            "Detection": row["Detection"],
            "Triage": row["Triage"],
            "Discovery": row["Discovery"],
        }
        for row in results
        if not (row["Detection"]["score"] and row["Triage"]["score"] and row["Discovery"]["score"])
    ]
    payload = {
        "schema_version": "aer.pea.handoff-study-report.v0.6-development",
        "status": "complete_development_only",
        "promotion_status": "not_accepted",
        "official_leaderboard_result": False,
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "fast_mode": FAST_MODE,
        "metrics": {
            key: evaluation[key] for key in ("Detection", "Triage", "Experiment", "Discovery")
        },
        "composite_score": None,
        "call_accounting": {
            "construction_and_blind_review_one_shot_calls": one_shot_calls,
            "run_and_evaluation_native_turn_calls": turn_calls,
            "total_model_calls": one_shot_calls + turn_calls,
            "registered_calls_per_task_after_handoff_selection": 5,
            "usage_field_totals_diagnostic_only": _usage_totals(all_usage),
        },
        "run_evaluate_decoupling": {
            "separate_process_phases": True,
            "scienceworld_started_during_evaluation": False,
            "native_saved_threads_resumed": True,
            "full_rpc_and_session_artifacts_preserved": True,
        },
        "diagnostics": diagnostics,
        "error_and_reason_review_queue": wrong_or_reason_review,
        "Experiment": "not_run",
    }
    write_json(output_root / "study_report.json", payload)

    detection = payload["metrics"]["Detection"]
    triage = payload["metrics"]["Triage"]
    discovery = payload["metrics"]["Discovery"]
    discovery_groups = json.dumps(
        discovery["macro_by_configuration_group"], ensure_ascii=False, sort_keys=True
    )
    lines = [
        "# 豌豆 Case v0.6-development：20 个原生 handoff 实验报告",
        "",
        "> 本报告是开发结果，不是 leaderboard 结果；仍需人工盲审后才能晋级。",
        "",
        "## 结论摘要",
        "",
        (
            f"- Detection：均分 {detection['mean']:.3f}，灵敏度 "
            f"{detection['sensitivity']:.3f}，特异度 {detection['specificity']:.3f}，"
            f"平衡准确率 {detection['balanced_accuracy']:.3f}。"
        ),
        (
            f"- Triage：均分 {triage['mean']:.3f}，灵敏度 "
            f"{triage['sensitivity']:.3f}，特异度 {triage['specificity']:.3f}，"
            f"平衡准确率 {triage['balanced_accuracy']:.3f}。"
        ),
        f"- Discovery：均分 {discovery['mean']:.3f}；分组结果 {discovery_groups}。",
        "- Experiment：本轮按冻结方案不运行；composite score 为空。",
        f"- 20 个 Task 中完成 G0 的数量：{sum(item['G0_completed'] for item in diagnostics)}/20。",
        (
            "- handoff 选定后每个 Task 固定 5 次模型调用（checkpoint、main、P1、P2、P4）；"
            "总调用账目见 `study_report.json`。"
        ),
        "",
        "## 运行与评测解耦证据",
        "",
        (
            "主运行阶段只重放冻结 recipe、执行 checkpoint/main 并保存原生 sibling forks；"
            "评测阶段在新的 app-server 进程中恢复 P1、P2、P4 线程，没有启动 "
            "ScienceWorld。完整 JSON-RPC、session JSONL、公开环境轨迹和 operator "
            "windows 均保留。"
        ),
        "",
        "## 逐 Task 结果",
        "",
        "| Task | Detection | Triage | Discovery | G0 | 模型调用 | 工具事件 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    by_task = {row["task_id"]: row for row in results}
    for diagnostic in diagnostics:
        row = by_task[diagnostic["task_id"]]
        lines.append(
            f"| {row['task_id']} | {row['Detection']['score']} | "
            f"{row['Triage']['score']} | {row['Discovery']['score']} | "
            f"{int(diagnostic['G0_completed'])} | {diagnostic['model_call_count']} | "
            f"{diagnostic['tool_use']['event_count']} |"
        )
    lines.extend(
        [
            "",
            "## 需要人工复核的错例",
            "",
            (
                "所有三项均正确，没有自动加入错例队列。"
                if not wrong_or_reason_review
                else "以下 Task 至少有一项精确匹配失败："
            ),
        ]
    )
    for item in wrong_or_reason_review:
        lines.append(
            f"- {item['task_id']}：Detection={item['Detection']['score']}，"
            f"Triage={item['Triage']['score']}，Discovery={item['Discovery']['score']}。"
        )
    lines.append("")
    _safe_text(output_root / "study_report.md", "\n".join(lines))
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("construct", "run", "evaluate", "report", "full"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--task", action="append")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.command == "report" and args.task:
        parser.error("report always aggregates the complete 20-Task study")
    return args


def main() -> int:
    args = _parse_args()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.command in {"construct", "full"}:
        construct(output_root, args.task, args.timeout)
    if args.command in {"run", "full"}:
        run_study(output_root, args.task, args.timeout)
    if args.command in {"evaluate", "full"}:
        evaluate_study(output_root, args.task, args.timeout)
    if args.command in {"report", "full"}:
        report(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
