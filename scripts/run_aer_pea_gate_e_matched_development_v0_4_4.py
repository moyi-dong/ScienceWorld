#!/usr/bin/env python3
"""Run the versioned 2:1 preference-strength matched development rerun."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import run_aer_pea_calibration as frozen
import run_aer_pea_gate_e_matched_development as legacy

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[3]
CASE_ROOT = ROOT / "cases/science/mendelian_genetics_known_plant_aer"
LEGACY_CONFIG = CASE_ROOT / "construction/gate-e-matched-development-study.v0.4.2-development.json"
CHANGE_REQUEST = CASE_ROOT / "construction/protocol-change-preference-strength.v0.4.4-development.json"
LEGACY_RUNTIME = ROOT / "artifacts/aer_pea_case/runtime-archives/scienceworld-legacy-9to1-before-v0.4.4.jar"
PREFERENCE_WEIGHT = 2.0
EXPECTED_STUDY_VERSION = "0.4.4-matched-development-2to1"

_legacy_load_study_config = legacy._load_study_config
_legacy_central_freeze_manifest = legacy._central_freeze_manifest
_legacy_finalize_metadata = legacy._finalize_metadata
_legacy_configure_aer_pea_case = frozen.ScienceWorldEnv.configure_aer_pea_case


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_study_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    legacy_config = _legacy_load_study_config(LEGACY_CONFIG)
    if config.get("schema_version") != legacy_config["schema_version"]:
        raise ValueError("2:1 matched rerun schema changed")
    if config.get("study_version") != EXPECTED_STUDY_VERSION:
        raise ValueError("2:1 matched rerun study version changed")
    if config.get("status") != "frozen_for_matched_development_2to1_rerun":
        raise ValueError("2:1 matched rerun is not frozen")
    if config.get("rerun_of_study_version") != legacy_config["study_version"]:
        raise ValueError("2:1 matched rerun legacy binding changed")
    if config.get("previous_valid_live_episode_count") != 74:
        raise ValueError("2:1 matched rerun prior live count changed")
    if config.get("registered_matched_development_episode_count") != 48:
        raise ValueError("2:1 matched rerun episode count changed")
    if config.get("cumulative_valid_live_episode_count_if_complete") != 122:
        raise ValueError("2:1 matched rerun cumulative live count changed")
    if config.get("authorized_live_episode_cap", 0) < 122:
        raise ValueError("2:1 matched rerun exceeds live authorization")
    if config.get("newness_scope") != "same_development_cells_new_preference_protocol":
        raise ValueError("2:1 matched rerun newness disclosure changed")

    for key in (
        "held_out_execution_allowed",
        "historical_results_may_be_rewritten",
        "official_leaderboard_result",
        "model",
        "reasoning_effort",
        "formal_episode_action_budget",
        "source_bindings",
        "matched_pre_exposure",
        "common_interface_instruction",
        "conditions",
        "matched_development",
        "review",
        "stopping_rules",
    ):
        if config.get(key) != legacy_config.get(key):
            raise ValueError(f"2:1 matched rerun changed frozen field: {key}")

    protocol = config.get("environment_protocol")
    if protocol != {
        "version": "0.4.4-development",
        "preference_weight": PREFERENCE_WEIGHT,
        "legacy_default_preference_weight": 9.0,
        "applies_to_worlds": [
            "white_preference",
            "position_attraction",
            "plant_attractiveness",
        ],
        "change_request": (
            "cases/science/mendelian_genetics_known_plant_aer/construction/"
            "protocol-change-preference-strength.v0.4.4-development.json"
        ),
        "historical_environment_behavior_changed": False,
    }:
        raise ValueError("2:1 matched rerun environment protocol changed")
    if not CHANGE_REQUEST.is_file() or not LEGACY_RUNTIME.is_file():
        raise FileNotFoundError("2:1 protocol change request or legacy runtime is missing")
    change = json.loads(CHANGE_REQUEST.read_text(encoding="utf-8"))
    if change.get("status") != "approved_for_development_rerun":
        raise ValueError("2:1 preference change is not approved")
    if change["new_protocol"].get("preference_weight") != PREFERENCE_WEIGHT:
        raise ValueError("2:1 preference change request differs from the study")
    if _sha256(LEGACY_RUNTIME) != change["legacy_protocol"]["scienceworld_jar_sha256"]:
        raise ValueError("archived 9:1 ScienceWorld runtime hash changed")
    return config


def _configure_two_to_one(
    self: Any, world_name: str = "white_preference", case_root: int = 0
) -> str:
    return _legacy_configure_aer_pea_case(
        self,
        world_name,
        case_root,
        preference_weight=PREFERENCE_WEIGHT,
    )


def _central_freeze_manifest(
    config: dict[str, Any], config_path: Path
) -> dict[str, Any]:
    payload = _legacy_central_freeze_manifest(config, config_path)
    payload["environment_protocol"] = config["environment_protocol"]
    payload["rerun_of_study_version"] = config["rerun_of_study_version"]
    payload["source_sha256"]["preference_change_request"] = {
        "path": str(CHANGE_REQUEST),
        "sha256": _sha256(CHANGE_REQUEST),
    }
    payload["source_sha256"]["legacy_9to1_runtime_archive"] = {
        "path": str(LEGACY_RUNTIME),
        "sha256": _sha256(LEGACY_RUNTIME),
    }
    return payload


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
    _legacy_finalize_metadata(
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
    metadata_path = Path(outcome["artifact_dir"]) / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["environment_protocol"] = config["environment_protocol"]
    metadata["matched_development_bindings"]["versioned_runner"] = {
        "path": str(SCRIPT_PATH),
        "sha256": _sha256(SCRIPT_PATH),
    }
    frozen._safe_write_json(metadata_path, metadata)
    return _sha256(metadata_path)


def main() -> int:
    legacy.SCRIPT_PATH = SCRIPT_PATH
    legacy._load_study_config = _load_study_config
    legacy._central_freeze_manifest = _central_freeze_manifest
    legacy._finalize_metadata = _finalize_metadata
    frozen.ScienceWorldEnv.configure_aer_pea_case = _configure_two_to_one
    return legacy.main()


if __name__ == "__main__":
    sys.exit(main())
