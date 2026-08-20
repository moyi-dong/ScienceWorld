#!/usr/bin/env python3
"""Build a deterministic operator summary for AER pea prompt-control runs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_EXPECTED_WORLDS = {
    "position_attraction",
    "plant_attractiveness",
    "fertility_difference",
    "transient_null",
}
EXPECTED_PROMPT_VARIANTS = {"baseline", "anomaly_cue"}
MECHANISM_BY_WORLD = {
    "white_preference": "perceived_flower_color",
    "position_attraction": "flower_pot_position",
    "plant_attractiveness": "plant_identity",
    "fertility_difference": "post_pollination_fruit_set_speed",
    # Retained only so the summarizer can read the immutable retired 7-world archive.
    "observation_artifact": "public_log_omission",
    "transient_null": "selected_uniform_null_root",
    "clean": "uniform_flower_choice",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mechanism_operated(world: str, summary: dict[str, Any]) -> bool:
    if world == "position_attraction":
        return summary["preferred_flower_pot"] is not None
    if world == "plant_attractiveness":
        return summary["preferred_plant_id"] is not None
    if world == "fertility_difference":
        return summary["fertility_accelerated_ticks"] > 0
    if world == "observation_artifact":
        return summary.get(
            "public_omitted_visits",
            summary["total_visits"] - summary.get("public_recorded_visits", 0),
        ) > 0
    if world == "transient_null":
        return summary["transient_root_seed"] is not None
    return summary["total_visits"] > 0


def _exposure_sufficient(world: str, summary: dict[str, Any]) -> bool:
    if world in {"position_attraction", "plant_attractiveness"}:
        return summary["comparable_visits"] >= 8
    if world == "fertility_difference":
        return (
            summary["fertility_accelerated_ticks"] > 0
            and summary["white_recipient_pollinations"] > 0
            and summary["purple_recipient_pollinations"] > 0
            and summary["white_fruit_sets"] > 0
            and summary["purple_fruit_sets"] > 0
        )
    if world == "observation_artifact":
        logged_total = summary.get(
            "public_logged_comparable_visits", summary.get("public_recorded_visits", 0)
        )
        logged_white = summary.get(
            "public_logged_white_visits", summary.get("public_white_visits", 0)
        )
        public_white_fraction = _ratio(
            logged_white,
            logged_total,
        )
        return (
            summary["comparable_visits"] >= 8
            and logged_total >= 4
            and public_white_fraction is not None
            and public_white_fraction >= 0.75
            and summary.get(
                "public_omitted_visits",
                summary["total_visits"] - summary.get("public_recorded_visits", 0),
            )
            >= 2
        )
    if world == "transient_null":
        return (
            summary["initial_comparable_window_size"] >= 6
            and summary["initial_white_visits"] == summary["initial_comparable_window_size"]
        )
    return summary["comparable_visits"] >= 8


def _agent_messages(events: list[dict[str, Any]]) -> str:
    return "\n".join(
        str(event.get("payload", {}).get("text", ""))
        for event in events
        if event.get("kind") == "agent_message"
    )


def _case_term_flags(messages: str) -> dict[str, bool]:
    lowered = messages.lower()
    patterns = {
        "mentions_anomaly": r"\banomal",
        "mentions_bee": r"\bbees?\b",
        "mentions_flower_visit": r"\bvisit(?:s|ed|ing)?\b.*\bflower|\bflower.*\bvisit",
        "mentions_preference": r"\bprefer(?:s|red|ence|ential)?\b|\bbias(?:ed)?\b",
        "mentions_position": r"\bposition\b|\blocation\b",
        "mentions_log_artifact": r"\blog\b|\brecord(?:ed|ing)?\b.*\bartifact",
        "mentions_fertility": r"\bfertility\b|\bfruit set\b|\breproductive rate\b",
        "mentions_transient": r"\btransient\b|\brandom streak\b|\bfinite sample\b",
    }
    return {name: re.search(pattern, lowered) is not None for name, pattern in patterns.items()}


def _summarize_run(run_dir: Path) -> dict[str, Any]:
    metadata = _read_json(run_dir / "run_metadata.json")
    hidden = _read_json(run_dir / "hidden_summary.json")
    final = _read_json(run_dir / "codex" / "final.json")
    public_events = _read_jsonl(run_dir / "public_environment_trajectory.jsonl")
    normalized_events = _read_jsonl(run_dir / "codex" / "normalized_events.jsonl")
    actions = [
        str(event.get("request", {}).get("action"))
        for event in public_events
        if event.get("request", {}).get("action") is not None
    ]
    messages = _agent_messages(normalized_events)
    flags = _case_term_flags(messages)
    last_score = next(
        (
            event.get("response", {}).get("score")
            for event in reversed(public_events)
            if "score" in event.get("response", {})
        ),
        None,
    )

    comparable = hidden["comparable_visits"]
    preferred_visits = None
    if hidden["world"] == "position_attraction":
        preferred_visits = hidden["preferred_position_visits"]
    elif hidden["world"] == "plant_attractiveness":
        preferred_visits = hidden["preferred_plant_visits"]

    prompt_variant = metadata.get("prompt_variant")
    if prompt_variant is None:
        prompt_variant = run_dir.parent.name.replace("-", "_")

    return {
        "run_id": metadata["run_id"],
        "prompt_variant": prompt_variant,
        "world": metadata["world"],
        "repetition": metadata["repetition"],
        "case_root": metadata.get("case_root", 0),
        "model": metadata["model"],
        "reasoning_effort": metadata["reasoning_effort"],
        "fast_mode": metadata["fast_mode"],
        "status": metadata["status"],
        "environment_completed": metadata["environment_completed"],
        "score": last_score,
        "height_trait": final.get("height_trait"),
        "final_completed": final.get("completed"),
        "final_evidence": final.get("evidence"),
        "thread_id": metadata["thread_id"],
        "prompt_sha256": metadata.get("prompt_sha256"),
        "scienceworld_jar_sha256": metadata.get("scienceworld_jar_sha256"),
        "mechanism": hidden.get("mechanism", MECHANISM_BY_WORLD[metadata["world"]]),
        "mechanism_operated": _mechanism_operated(metadata["world"], hidden),
        "exposure_sufficient": _exposure_sufficient(metadata["world"], hidden),
        "total_visits": hidden["total_visits"],
        "comparable_visits": comparable,
        "white_visits": hidden["white_visits"],
        "purple_visits": hidden["purple_visits"],
        "white_fraction": _ratio(hidden["white_visits"], comparable),
        "preferred_visits": preferred_visits,
        "preferred_fraction": _ratio(preferred_visits, comparable)
        if preferred_visits is not None
        else None,
        "public_recorded_visits": hidden.get(
            "public_logged_comparable_visits", hidden["total_visits"]
        ),
        "public_white_visits": hidden.get(
            "public_logged_white_visits", hidden["white_visits"]
        ),
        "public_purple_visits": hidden.get(
            "public_logged_purple_visits", hidden["purple_visits"]
        ),
        "public_white_fraction": _ratio(
            hidden.get("public_logged_white_visits", hidden["white_visits"]),
            hidden.get("public_logged_comparable_visits", hidden["total_visits"]),
        ),
        "pollinations": hidden.get("pollinations", 0),
        "white_recipient_pollinations": hidden.get("white_recipient_pollinations", 0),
        "purple_recipient_pollinations": hidden.get("purple_recipient_pollinations", 0),
        "fruit_sets": hidden.get("fruit_sets", 0),
        "white_fruit_sets": hidden.get("white_fruit_sets", 0),
        "purple_fruit_sets": hidden.get("purple_fruit_sets", 0),
        "fertility_accelerated_ticks": hidden.get("fertility_accelerated_ticks", 0),
        "initial_comparable_window_size": hidden.get("initial_comparable_window_size", 0),
        "initial_white_visits": hidden.get("initial_white_visits", 0),
        "initial_purple_visits": hidden.get("initial_purple_visits", 0),
        "actions": len(actions),
        "look_actions": sum(action.startswith("look") for action in actions),
        "wait_actions": sum(action.startswith("wait") for action in actions),
        "manual_bee_moves": sum(
            action.startswith("move bee") or action.startswith("move adult bee")
            for action in actions
        ),
        **flags,
    }


def _aggregate(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in runs:
        key = (run["prompt_variant"], run["world"])
        groups.setdefault(key, []).append(run)

    return [
        {
            "prompt_variant": prompt_variant,
            "world": world,
            "runs": len(group),
            "g0_completed": sum(
                run["environment_completed"] and run["score"] == 100 for run in group
            ),
            "mechanism_operated": sum(run["mechanism_operated"] for run in group),
            "exposure_sufficient": sum(run["exposure_sufficient"] for run in group),
            "mentions_anomaly": sum(run["mentions_anomaly"] for run in group),
            "mentions_preference": sum(run["mentions_preference"] for run in group),
            "mentions_position": sum(run["mentions_position"] for run in group),
            "mentions_log_artifact": sum(run["mentions_log_artifact"] for run in group),
            "mentions_fertility": sum(run["mentions_fertility"] for run in group),
            "mentions_transient": sum(run["mentions_transient"] for run in group),
        }
        for (prompt_variant, world), group in sorted(groups.items())
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--world", action="append")
    parser.add_argument("--runs-per-cell", type=int, default=3)
    parser.add_argument(
        "--legacy-jar-sha256",
        help="JAR hash for runs created before run_metadata recorded the binary hash",
    )
    parser.add_argument(
        "--legacy-baseline-prompt-sha256",
        help="baseline prompt hash for legacy metadata that predates prompt recording",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if args.runs_per_cell < 1:
        parser.error("--runs-per-cell must be positive")
    run_dirs = sorted(path.parent for path in root.glob("*/*/run_metadata.json"))
    if not run_dirs:
        parser.error(f"no completed runs found under {root}")

    runs = [_summarize_run(run_dir) for run_dir in run_dirs]
    expected_worlds = set(args.world or DEFAULT_EXPECTED_WORLDS)
    if args.world is None and any(
        run["world"] == "observation_artifact" for run in runs
    ):
        # Old five-world archives remain self-describing and reproducible even
        # though new calibration runs no longer generate this retired world.
        expected_worlds.add("observation_artifact")
    legacy_jar_runs = 0
    legacy_baseline_prompt_runs = 0
    for run in runs:
        if run["scienceworld_jar_sha256"] is None and args.legacy_jar_sha256:
            run["scienceworld_jar_sha256"] = args.legacy_jar_sha256
            legacy_jar_runs += 1
        if (
            run["prompt_sha256"] is None
            and run["prompt_variant"] == "baseline"
            and args.legacy_baseline_prompt_sha256
        ):
            run["prompt_sha256"] = args.legacy_baseline_prompt_sha256
            legacy_baseline_prompt_runs += 1
    matrix_counts = Counter((run["prompt_variant"], run["world"]) for run in runs)
    expected_cells = {
        (prompt_variant, world): args.runs_per_cell
        for prompt_variant in EXPECTED_PROMPT_VARIANTS
        for world in expected_worlds
    }
    payload = {
        "root": str(root),
        "completed_runs": len(runs),
        "expected_runs": len(expected_worlds)
        * len(EXPECTED_PROMPT_VARIANTS)
        * args.runs_per_cell,
        "matrix_complete": dict(matrix_counts) == expected_cells,
        "all_g0_completed": all(
            run["environment_completed"] and run["score"] == 100 for run in runs
        ),
        "configuration_valid": all(
            run["model"] == "gpt-5.6-sol"
            and run["reasoning_effort"] == "high"
            and run["fast_mode"] is False
            for run in runs
        ),
        "prompt_hashes": {
            variant: sorted(
                {run["prompt_sha256"] for run in runs if run["prompt_variant"] == variant}
            )
            for variant in sorted(EXPECTED_PROMPT_VARIANTS)
        },
        "scienceworld_jar_hashes_by_world": {
            world: sorted(
                {
                    run["scienceworld_jar_sha256"]
                    for run in runs
                    if run["world"] == world
                    and run["scienceworld_jar_sha256"] is not None
                }
            )
            for world in sorted(expected_worlds)
        },
        "legacy_jar_hash_applied_to_runs": legacy_jar_runs,
        "runs_missing_jar_hash": sum(
            run["scienceworld_jar_sha256"] is None for run in runs
        ),
        "legacy_baseline_prompt_hash_applied_to_runs": legacy_baseline_prompt_runs,
        "aggregate": _aggregate(runs),
        "runs": runs,
    }
    output = (args.output or root / "prompt_control_summary.json").resolve()
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "runs"}, indent=2))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
