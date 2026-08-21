import copy
import json
import sys
from pathlib import Path

import pytest

SCIENCEWORLD_ROOT = Path(__file__).resolve().parents[1]
AER_BENCH_ROOT = SCIENCEWORLD_ROOT.parents[1]
sys.path.insert(0, str(SCIENCEWORLD_ROOT / "scripts"))

from run_aer_pea_gate_e_confirmation import (  # noqa: E402
    EXPECTED_CONDITION,
    EXPECTED_WORLDS,
    _central_freeze_manifest,
    _load_confirmation_config,
    _shard_freeze_manifest,
    _write_once_or_validate,
)

CONFIG_PATH = (
    AER_BENCH_ROOT
    / "cases/science/mendelian_genetics_known_plant_aer/construction"
    / "gate-e-confirmation-study.v0.4.1-development.json"
)
RUBRIC_PATH = (
    AER_BENCH_ROOT
    / "cases/science/mendelian_genetics_known_plant_aer/construction"
    / "gate-e-review-rubric.v0.4.0-development.json"
)


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_confirmation_freezes_l4_all_worlds_and_two_unused_cells() -> None:
    config = _load_confirmation_config(CONFIG_PATH)

    assert config["selected_condition"] == EXPECTED_CONDITION
    assert tuple(config["confirmation"]["worlds"]) == EXPECTED_WORLDS
    assert config["confirmation"]["replication_cells"] == [
        {"repetition": 1, "case_root": 211, "variation": 5},
        {"repetition": 2, "case_root": 401, "variation": 23},
    ]
    assert config["registered_confirmation_episode_count"] == 12
    assert config["cumulative_valid_live_episode_count_if_complete"] == 26
    assert config["held_out_execution_allowed"] is False


def test_confirmation_shards_partition_all_six_worlds() -> None:
    config = _load_confirmation_config(CONFIG_PATH)
    flattened = [
        (entry["world"], entry["repetition"])
        for entries in config["confirmation"]["shards"].values()
        for entry in entries
    ]

    assert len(flattened) == 12
    assert set(flattened) == {
        (world, repetition)
        for world in EXPECTED_WORLDS
        for repetition in (1, 2)
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(held_out_execution_allowed=True), "forbid held-out"),
        (
            lambda value: value.update(selected_condition="l3_competing_hypotheses"),
            "condition changed",
        ),
        (
            lambda value: value["confirmation"].update(
                worlds=["white_preference"]
            ),
            "worlds changed",
        ),
        (
            lambda value: value["confirmation"].update(
                registered_episode_count=11
            ),
            "episode count changed",
        ),
    ],
)
def test_confirmation_rejects_boundary_mutations(
    tmp_path: Path, mutation, message: str
) -> None:
    config = copy.deepcopy(_config())
    mutation(config)
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _load_confirmation_config(path)


def test_confirmation_freeze_manifest_binds_selection_and_execution() -> None:
    config = _load_confirmation_config(CONFIG_PATH)
    manifest = _central_freeze_manifest(config, CONFIG_PATH, RUBRIC_PATH)

    assert manifest["registered_episode_count"] == 12
    assert manifest["held_out_execution_allowed"] is False
    assert {
        "confirmation_config",
        "review_rubric",
        "confirmation_runner",
        "episode_runner",
        "frozen_v0_2_0_runner",
        "selected_pilot_prompt_source",
        "pilot_selection_result",
        "public_lab_client",
        "public_submission_schema",
        "hidden_evidence_builder",
        "hidden_deterministic_grader",
        "development_split",
        "scienceworld_jar",
    } == set(manifest["source_sha256"])


def test_confirmation_parallel_freeze_is_write_once_and_shard_specific(
    tmp_path: Path,
) -> None:
    config = _load_confirmation_config(CONFIG_PATH)
    central = tmp_path / "confirmation_freeze_manifest.json"
    payload = _central_freeze_manifest(config, CONFIG_PATH, RUBRIC_PATH)

    _write_once_or_validate(central, payload)
    _write_once_or_validate(central, payload)
    shard = _shard_freeze_manifest(config, "shard-a", central)

    assert shard["shard"] == "shard-a"
    assert shard["shard_registered_episode_count"] == 4
    assert shard["central_freeze_manifest"]["sha256"]

    changed = copy.deepcopy(payload)
    changed["selected_condition"] = "changed"
    with pytest.raises(RuntimeError, match="differs"):
        _write_once_or_validate(central, changed)
