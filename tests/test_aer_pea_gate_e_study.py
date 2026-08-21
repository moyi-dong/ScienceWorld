import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCIENCEWORLD_ROOT = Path(__file__).resolve().parents[1]
AER_BENCH_ROOT = SCIENCEWORLD_ROOT.parents[1]
sys.path.insert(0, str(SCIENCEWORLD_ROOT / "scripts"))

from run_aer_pea_gate_e_study import (  # noqa: E402
    EXPECTED_CONDITION_ORDER,
    EXPECTED_PILOT_WORLDS,
    _build_prompt,
    _freeze_manifest,
    _load_study_config,
    _save_exact_prompt,
)

STUDY_PATH = (
    AER_BENCH_ROOT
    / "cases/science/mendelian_genetics_known_plant_aer/construction"
    / "gate-e-prompt-study.v0.4.0-development.json"
)
RUBRIC_PATH = (
    AER_BENCH_ROOT
    / "cases/science/mendelian_genetics_known_plant_aer/construction"
    / "gate-e-review-rubric.v0.4.0-development.json"
)


def _study() -> dict:
    return json.loads(STUDY_PATH.read_text(encoding="utf-8"))


def _service() -> SimpleNamespace:
    return SimpleNamespace(
        env=SimpleNamespace(taskdescription=lambda: "Determine the height trait."),
        initial={"observation": "You are in the greenhouse."},
        pre_exposure_observations=["Matched public observation."],
    )


def test_registered_gate_e_pilot_is_exactly_fourteen_development_episodes() -> None:
    study = _load_study_config(STUDY_PATH)

    assert tuple(study["conditions"]) == EXPECTED_CONDITION_ORDER
    assert tuple(study["pilot"]["worlds"]) == EXPECTED_PILOT_WORLDS
    assert study["pilot"]["replication_cells"] == [
        {"repetition": 1, "case_root": 101, "variation": 0}
    ]
    assert study["pilot"]["registered_episode_count"] == 14
    assert study["held_out_execution_allowed"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(held_out_execution_allowed=True), "forbid held-out"),
        (
            lambda value: value["pilot"].update(worlds=["white_preference"]),
            "pilot worlds changed",
        ),
        (
            lambda value: value["pilot"].update(registered_episode_count=13),
            "episode count",
        ),
        (
            lambda value: value["pilot"].update(
                review_begins_only_after_all_registered_episodes_finish=False
            ),
            "review must wait",
        ),
    ],
)
def test_study_config_rejects_boundary_mutations(
    tmp_path: Path, mutation, message: str
) -> None:
    study = copy.deepcopy(_study())
    mutation(study)
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(study), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _load_study_config(path)


def test_every_condition_receives_identical_common_interface_text() -> None:
    study = _study()
    prompts = {
        name: _build_prompt(_service(), study, name) for name in study["conditions"]
    }
    common = study["common_interface_instruction"]

    assert all(prompt.count(common) == 1 for prompt in prompts.values())
    assert prompts["l0_interface_only"].endswith(common + "\n")
    assert study["conditions"]["l4_explicit_elimination"]["prompt_addition"] in prompts[
        "l4_explicit_elimination"
    ]
    assert study["conditions"]["l4_explicit_elimination"]["prompt_addition"] not in prompts[
        "l0_interface_only"
    ]


def test_exact_submitted_prompt_is_saved_byte_for_byte(tmp_path: Path) -> None:
    prompt = "line one\nline two with unicode 豌豆\n"
    path = _save_exact_prompt(tmp_path, prompt)

    assert path.read_bytes() == prompt.encode("utf-8")


def test_freeze_manifest_binds_every_scoring_and_execution_input() -> None:
    study = _load_study_config(STUDY_PATH)
    manifest = _freeze_manifest(study, STUDY_PATH, RUBRIC_PATH)

    assert manifest["held_out_execution_allowed"] is False
    assert manifest["registered_episode_count"] == 14
    assert set(manifest["source_sha256"]) == {
        "study_config",
        "review_rubric",
        "study_runner",
        "frozen_v0_2_0_runner",
        "public_lab_client",
        "public_submission_schema",
        "hidden_evidence_builder",
        "hidden_deterministic_grader",
        "development_split",
        "scienceworld_jar",
    }
    assert all(
        len(binding["sha256"]) == 64
        for binding in manifest["source_sha256"].values()
    )
