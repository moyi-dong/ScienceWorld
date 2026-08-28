import json
import sys
from pathlib import Path

import pytest
from scienceworld import ScienceWorldEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_aer_pea_v1_pilot import (  # noqa: E402
    MATRIX_MANIFEST,
    V1EpisodeService,
    _automatic_checks,
    load_hidden_configuration_matrix,
)

TASK = "mendelian-genetics-known-plant-aer"
NO_NOISE = {
    "soil_nutrient_lot": "none",
    "fruit_set_success": "none",
    "cross_parentage_contamination": "none",
}
PILOT_NOISE = {
    "soil_nutrient_lot": "medium",
    "fruit_set_success": "medium",
    "cross_parentage_contamination": "weak",
}


def test_hidden_configuration_matrix_is_isolated_4_3_3():
    matrix = load_hidden_configuration_matrix()
    assert MATRIX_MANIFEST.is_file()
    assert len(matrix) == 10
    groups = {
        group: [item for item in matrix.values() if item["group"] == group]
        for group in ("mechanism", "single_noise", "clean_control")
    }
    assert {group: len(items) for group, items in groups.items()} == {
        "mechanism": 4,
        "single_noise": 3,
        "clean_control": 3,
    }
    assert all(item["noise_levels"] == NO_NOISE for item in groups["mechanism"])
    assert all(
        item["world"] == "clean"
        and sum(level != "none" for level in item["noise_levels"].values()) == 1
        for item in groups["single_noise"]
    )
    assert all(
        item["world"] == "clean" and item["noise_levels"] == NO_NOISE
        for item in groups["clean_control"]
    )
    assert {item["development_root"] for item in groups["clean_control"]} == {101, 211, 401}


@pytest.mark.parametrize(
    ("group", "world", "noise", "error"),
    [
        (
            "mechanism",
            "white_preference",
            {**NO_NOISE, "fruit_set_success": "weak"},
            "must have no added noise",
        ),
        ("single_noise", "clean", NO_NOISE, "must use clean and one noise axis"),
        (
            "clean_control",
            "clean",
            {**NO_NOISE, "soil_nutrient_lot": "medium"},
            "must use clean with no added noise",
        ),
    ],
)
def test_hidden_configuration_matrix_rejects_group_contamination(
    tmp_path, group, world, noise, error
):
    original = json.loads(MATRIX_MANIFEST.read_text(encoding="utf-8"))
    target = next(item for item in original["configurations"] if item["group"] == group)
    target["world"] = world
    target["noise_levels"] = noise
    broken = tmp_path / "broken-matrix.json"
    broken.write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        load_hidden_configuration_matrix(broken)


def test_pilot_checks_separate_null_decision_from_exact_noise_explanation():
    configuration = load_hidden_configuration_matrix()[
        "N2-cross-parentage-contamination-only"
    ]
    submission = {
        "height_trait": "dominant",
        "completed": True,
        "evidence": "protected crosses showed no stable anomaly",
        "anomaly": {
            "surface": "none",
            "persistent_anomaly": False,
            "mechanism_id": "uniform_no_anomaly",
            "notice_record_id": None,
            "evidence_experiment_ids": [],
            "validation_experiment_id": None,
        },
    }
    trajectory = [
        {
            "source": "solver",
            "request": {"command": "act"},
            "response": {"completed": True, "score": 100},
        }
    ]
    hidden_summary = {
        "cross_attempts": [],
        "soil_lot_assignments": [],
        "comparable_visits": 0,
        "pollinations": 0,
    }
    checks = _automatic_checks(
        submission,
        trajectory,
        hidden_summary,
        "clean",
        configuration,
    )
    assert checks["persistent_decision_match"] is True
    assert checks["exact_conclusion_match"] is False
    assert checks["expected_explanation_process_exposed"] is False


def _service(tmp_path: Path, root: int = 101) -> V1EpisodeService:
    run = tmp_path / f"root-{root}"
    run.mkdir(parents=True)
    return V1EpisodeService(
        "clean",
        0,
        root,
        run / "trajectory.jsonl",
        run / "windows.jsonl",
        1_000,
        PILOT_NOISE,
    )


def _cultivate_two(service: V1EpisodeService) -> dict:
    status = service.handle({"command": "pots"})
    targets = [status["pots"][0]["name"], status["pots"][2]["name"]]
    assignments = [
        {"seed_id": status["seeds"][index]["seed_id"], "pot": pot}
        for index, pot in enumerate(targets)
    ]
    return service.handle(
        {
            "command": "cultivate",
            "spec": {
                "assignments": assignments,
                "target_stage": "flowering",
                "max_ticks": 100,
                "maintain_water": True,
            },
        }
    )


def test_noise_configuration_is_explicit_and_validated():
    env = ScienceWorldEnv("", serverPath=None, envStepLimit=100)
    try:
        with pytest.raises(ValueError, match="must contain"):
            env.configure_aer_pea_case("clean", 101, noise_levels={"soil_nutrient_lot": "weak"})
        with pytest.raises(ValueError, match="none, weak, medium, or strong"):
            env.configure_aer_pea_case(
                "clean",
                101,
                noise_levels={**NO_NOISE, "fruit_set_success": "extreme"},
            )
        configured = env.configure_aer_pea_case(
            "cross_direction_delay", 101, noise_levels=PILOT_NOISE
        )
        assert configured.endswith(":v0.4:2:2:1")
    finally:
        env.close()


def test_every_hidden_matrix_profile_reaches_the_simulator_with_exact_noise_levels():
    matrix = load_hidden_configuration_matrix()
    encoded = {"none": 0, "weak": 1, "medium": 2, "strong": 3}
    env = ScienceWorldEnv("", serverPath=None, envStepLimit=100)
    try:
        for item in matrix.values():
            env.configure_aer_pea_case(
                item["world"],
                item["development_root"],
                noise_levels=item["noise_levels"],
            )
            env.load(TASK, 0, "easy")
            summary = env.get_aer_pea_case_summary()
            assert summary["soil_nutrient_lot_level"] == encoded[
                item["noise_levels"]["soil_nutrient_lot"]
            ]
            assert summary["fruit_set_success_level"] == encoded[
                item["noise_levels"]["fruit_set_success"]
            ]
            assert summary["cross_parentage_contamination_level"] == encoded[
                item["noise_levels"]["cross_parentage_contamination"]
            ]
    finally:
        env.close()


def test_v04_world_has_twenty_pots_without_changing_legacy_protocol(tmp_path):
    service = _service(tmp_path / "v04")
    try:
        status = service.handle({"command": "pots"})
        names = [pot["name"] for pot in status["pots"]]
        assert len(names) == 20
        assert set(names) == {f"flower pot {index}" for index in range(1, 21)}

        watered = service.handle(
            {
                "command": "operate",
                "spec": {"operation": "water", "targets": names},
            }
        )
        assert watered["ok"] is True
        assert watered["targets"] == names
        assert watered["logical_actions"] == 1
        assert watered["primitive_mutations"] == 20
        assert all(pot["has_water"] for pot in watered["status"]["pots"])
    finally:
        service.close()

    env = ScienceWorldEnv("", serverPath=None, envStepLimit=100)
    try:
        env.configure_aer_pea_case("clean", 101)
        env.load(TASK, 0, "easy")
        assert len(env.get_aer_pea_case_public_status()["pots"]) == 6
    finally:
        env.close()


def test_batch_targets_are_exact_and_unlisted_pots_do_not_change(tmp_path):
    service = _service(tmp_path)
    try:
        help_response = service.handle({"command": "help"})
        assert help_response["commands"]["cultivate"]["target_stage"] == "flowering"
        assert help_response["commands"]["controlled-cross"]["crosses"][0]["bagged"] is True
        preregister = help_response["notebook"]["experiment_preregister"]
        assert set(preregister) == {
            "kind",
            "experiment_id",
            "phase",
            "probe_kind",
            "hypothesis",
            "prediction",
        }
        before = service.handle({"command": "pots"})
        target = before["pots"][0]["name"]
        untouched = before["pots"][1]["name"]
        result = service.handle(
            {"command": "operate", "spec": {"operation": "water", "targets": [target]}}
        )
        assert result["ok"] is True
        assert result["targets"] == [target]
        after = {pot["name"]: pot for pot in result["status"]["pots"]}
        assert after[target]["has_water"] is True
        assert after[untouched]["has_water"] is False

        duplicate = service.handle(
            {
                "command": "operate",
                "spec": {"operation": "water", "targets": [target, target]},
            }
        )
        assert duplicate == {"ok": False, "error": "targets must be unique"}
        unknown = service.handle(
            {
                "command": "operate",
                "spec": {"operation": "water", "targets": ["flower pot 404"]},
            }
        )
        assert unknown["ok"] is False
        assert "unknown flower pots" in unknown["error"]
    finally:
        service.close()


def test_cultivate_waits_until_every_selected_pot_is_flowering(tmp_path):
    service = _service(tmp_path)
    try:
        result = _cultivate_two(service)
        assert result["ok"] is True
        assert result["wait"]["timed_out"] is False
        assert result["wait"]["pending"] == []
        assert set(result["wait"]["ready"]) == set(result["targets"])
        pots = {pot["name"]: pot for pot in result["wait"]["status"]["pots"]}
        assert all(pots[target]["flower_count"] >= 1 for target in result["targets"])

        trajectory = [
            json.loads(line)
            for line in service.trajectory_path.read_text(encoding="utf-8").splitlines()
        ]
        assert sum(
            event.get("request", {}).get("command") == "cultivate" for event in trajectory
        ) == 1
        assert sum(event.get("source") == "v1_wait_until" for event in trajectory) > 1
    finally:
        service.close()


def test_soil_lots_replay_at_one_root_and_can_vary_across_roots(tmp_path):
    signatures = []
    runs = (
        ("first", 101),
        ("replay", 101),
        ("other-1", 211),
        ("other-2", 307),
        ("other-3", 401),
    )
    for run_name, root in runs:
        service = _service(tmp_path / run_name, root)
        try:
            result = _cultivate_two(service)
            operator_lots = {
                item["plant_id"]: item
                for item in service.env.get_aer_pea_case_summary()["soil_lot_assignments"]
            }
            selected = tuple(
                sorted(
                    (
                        pot["name"],
                        pot["plants"][0]["soil_lot_id"],
                        operator_lots[pot["plants"][0]["plant_id"]][
                            "soil_stage_multiplier"
                        ],
                    )
                    for pot in result["wait"]["status"]["pots"]
                    if pot["name"] in result["targets"]
                )
            )
            signatures.append((result["wait"]["elapsed_ticks"], selected))
        finally:
            service.close()
    assert signatures[0] == signatures[1]
    assert any(signature != signatures[0] for signature in signatures[2:])


def test_public_status_exposes_soil_lot_but_not_hidden_multiplier(tmp_path):
    service = _service(tmp_path)
    try:
        result = _cultivate_two(service)
        planted = [
            plant
            for pot in result["wait"]["status"]["pots"]
            for plant in pot["plants"]
        ]
        assert planted
        assert all("soil_lot_id" in plant for plant in planted)
        assert all("soil_stage_multiplier" not in plant for plant in planted)
        hidden_lots = service.env.get_aer_pea_case_summary()["soil_lot_assignments"]
        assert hidden_lots
        assert all("soil_stage_multiplier" in item for item in hidden_lots)
    finally:
        service.close()


def test_public_status_exposes_visible_active_flower_color_for_batch_selection(tmp_path):
    service = _service(tmp_path)
    try:
        result = _cultivate_two(service)
        planted = [
            plant
            for pot in result["wait"]["status"]["pots"]
            for plant in pot["plants"]
        ]
        assert planted
        assert all(plant["active_flowers"] for plant in planted)
        assert {
            flower["perceived_color"]
            for plant in planted
            for flower in plant["active_flowers"]
        } == {"purple", "white"}
        assert all(
            set(flower) == {"flower_id", "perceived_color"}
            for plant in planted
            for flower in plant["active_flowers"]
        )
        assert all("native_color" not in flower for plant in planted for flower in plant["active_flowers"])
    finally:
        service.close()


def test_controlled_cross_resolves_to_one_pod_with_four_traceable_sibling_seeds(tmp_path):
    run = tmp_path / "controlled"
    run.mkdir()
    service = V1EpisodeService(
        "clean", 0, 101, run / "trajectory.jsonl", run / "windows.jsonl", 1_000, NO_NOISE
    )
    try:
        cultivated = _cultivate_two(service)
        recipient, donor = cultivated["targets"]
        # v0.4 exploration remains available after the commissioned score is reached.
        service.completed = True
        crossed = service.handle(
            {
                "command": "controlled-cross",
                "spec": {
                    "crosses": [
                        {
                            "recipient_pot": recipient,
                            "pollen_pot": donor,
                            "emasculated": True,
                            "bagged": True,
                        },
                        {
                            "recipient_pot": donor,
                            "pollen_pot": recipient,
                            "emasculated": True,
                            "bagged": True,
                        }
                    ]
                },
            }
        )
        assert crossed["ok"] is True
        assert crossed["attempt_ids"] == ["cross-0", "cross-1"]
        waited = service.handle(
            {
                "command": "wait-until",
                "spec": {
                    "attempt_ids": crossed["attempt_ids"],
                    "condition": "resolved",
                    "max_ticks": 30,
                    "maintain_water": True,
                },
            }
        )
        assert waited["timed_out"] is False
        assert waited["ready"] == ["cross-0", "cross-1"]
        status = service.handle({"command": "pots"})
        attempts = [
            next(item for item in status["cross_attempts"] if item["attempt_id"] == attempt_id)
            for attempt_id in crossed["attempt_ids"]
        ]
        assert {attempt["pod_id"] for attempt in attempts} == {0, 1}
        for attempt in attempts:
            assert "actual_pollen_plant_id" not in attempt
            assert "contamination_occurred" not in attempt
            assert attempt["status"] == "pod_set"
            assert len(attempt["seed_ids"]) == 4
            siblings = [
                seed for seed in status["seeds"] if seed["seed_id"] in attempt["seed_ids"]
            ]
            assert len(siblings) == 4
            assert {seed["pod_id"] for seed in siblings} == {attempt["pod_id"]}
            assert {seed["maternal_plant_id"] for seed in siblings} == {
                attempt["recipient_plant_id"]
            }
            assert {seed["intended_pollen_plant_id"] for seed in siblings} == {
                attempt["intended_pollen_plant_id"]
            }
        selected_statuses = [
            pot for pot in status["pots"] if pot["name"] in {recipient, donor}
        ]
        assert all(pot["flower_count"] <= 1 for pot in selected_statuses)
    finally:
        service.close()


def test_cross_direction_world_survives_noise_free_gold_and_has_reciprocal_lag():
    env = ScienceWorldEnv("", serverPath=None, envStepLimit=10_000)
    try:
        env.configure_aer_pea_case(
            "cross_direction_delay", 101, noise_levels=NO_NOISE
        )
        env.load(TASK, 0, "easy", generateGoldPath=True)
        actions = list(env.server.getGoldActionSequence())
        env.load(TASK, 0, "easy", generateGoldPath=False)
        final = None
        for action in actions:
            final = env.step(action)
            if final[2]:
                break
        assert final is not None
        assert final[2] is True
        assert final[3]["score"] == 100
        summary = env.get_aer_pea_case_summary()
        slow = summary["tall_recipient_short_pollen_lags"]
        fast = summary["short_recipient_tall_pollen_lags"]
        assert slow and fast
        assert min(slow) > max(fast)
        public_status = env.get_aer_pea_case_public_status()
        history = public_status["reproduction_history"]
        assert history
        assert any(event["pollination_to_fruit_ticks"] for event in history)
        assert all("pollen_source_height" not in event for event in history)
    finally:
        env.close()
