import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from scienceworld import ScienceWorldEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_aer_pea_calibration import (  # noqa: E402
    ANOMALY_CUE,
    EpisodeService,
    _prompt,
    _validate_split,
)

TASK = "mendelian-genetics-known-plant-aer"
SUPPORTED_WORLDS = {
    "white_preference",
    "position_attraction",
    "plant_attractiveness",
    "fertility_difference",
    "transient_null",
    "clean",
}
EXPECTED_MECHANISMS = {
    "white_preference": "perceived_flower_color",
    "position_attraction": "flower_pot_position",
    "plant_attractiveness": "plant_identity",
    "fertility_difference": "post_pollination_fruit_set_speed",
    "transient_null": "selected_uniform_null_root",
    "clean": "uniform_flower_choice",
}
DEV_ROOTS = (101, 211, 307, 401)


def _run_actions(env, actions):
    outputs = []
    for action in actions:
        outputs.append((action, env.step(action)))
    return outputs


def _gold_and_parent_actions(env):
    env.configure_aer_pea_case("white_preference")
    env.load(TASK, 0, "easy", generateGoldPath=True)
    gold_actions = list(env.server.getGoldActionSequence())
    open_hive_idx = next(
        idx for idx, action in enumerate(gold_actions) if action.startswith("open bee hive")
    )
    parent_end_idx = next(
        idx
        for idx, action in enumerate(gold_actions[open_hive_idx + 1 :], open_hive_idx + 1)
        if action == "look at seed jar"
    )
    return gold_actions, gold_actions[: parent_end_idx + 1], open_hive_idx


def _comparable_fraction(summary, color):
    return summary[f"{color}_visits"] / summary["comparable_visits"]


def _is_comparable(event):
    return "white" in event["candidate_colors"] and "purple" in event["candidate_colors"]


def test_anomaly_prompt_control_adds_exactly_one_sentence():
    service = SimpleNamespace(
        env=SimpleNamespace(taskdescription=lambda: "Determine the height trait."),
        initial={"observation": "You are in the greenhouse."},
    )
    baseline = _prompt(service, "baseline")
    anomaly_cue = _prompt(service, "anomaly_cue")

    assert anomaly_cue == f"{baseline}\n{ANOMALY_CUE}\n"


def test_stage_control_prompts_supply_only_the_registered_bypass() -> None:
    service = SimpleNamespace(
        env=SimpleNamespace(taskdescription=lambda: "Determine the height trait."),
        initial={"observation": "You are in the greenhouse."},
        pre_exposure_observations=["Public matched exposure."],
    )
    notice = _prompt(service, "notice_bypass", "white_preference")
    assert "persistent flower-visit imbalance" in notice
    assert "not being told its mechanism or how to test it" in notice
    assert "perceived flower color" not in notice

    design = _prompt(service, "experiment_design", "white_preference")
    assert "Competing explanations include" in design
    assert "not being given an experiment plan" in design
    assert "sleeve" not in design
    assert "swap" not in design
    assert "additional sampling" not in design


def test_held_out_split_requires_a_matching_freeze_manifest(tmp_path):
    with pytest.raises(ValueError, match="require --held-out-freeze-manifest"):
        _validate_split("held_out", 2, 509, None)
    with pytest.raises(ValueError, match="not registered"):
        _validate_split("development", 2, 101, None)

    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "status": "frozen_for_held_out",
                "acceptance_spec_version": "0.2.0",
            }
        ),
        encoding="utf-8",
    )
    _validate_split("held_out", 2, 509, freeze)


def test_public_notebook_records_are_structured_and_timestamped(tmp_path):
    service = EpisodeService.__new__(EpisodeService)
    service._index = 0
    service._note_index = 0
    service._experiment_ids = set()
    service._active_experiment_id = None
    service.trajectory_path = tmp_path / "trajectory.jsonl"

    notice = service._record({"kind": "notice", "surface": "visit_imbalance"})
    assert notice["record_id"] == "NOTE-0001"
    assert service._record({"kind": "notice", "surface": "free text"})["ok"] is False
    registration = service._record(
        {
            "kind": "experiment_preregister",
            "experiment_id": "EXP-swap",
            "phase": "investigation",
            "probe_kind": "perceived_color_swap",
            "hypothesis": "perceived_flower_color",
            "prediction": "the preference follows displayed white",
        }
    )
    assert registration["ok"] is True
    assert service._active_experiment_id == "EXP-swap"
    assert service._record(registration["record"])["ok"] is False
    assert service._record(
        {"kind": "experiment_end", "experiment_id": "EXP-other"}
    )["ok"] is False
    ended = service._record(
        {"kind": "experiment_end", "experiment_id": "EXP-swap"}
    )
    assert ended["ok"] is True
    assert service._active_experiment_id is None
    records = [
        json.loads(line)
        for line in service.trajectory_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["index"] for record in records] == [0, 1, 2]


def test_public_submission_schema_matches_the_case_package():
    external_schema = json.loads(
        (Path(__file__).resolve().parents[1] / "scripts/aer_pea_submission.schema.json").read_text(
            encoding="utf-8"
        )
    )
    case_schema = json.loads(
        (
            Path(__file__).resolve().parents[3]
            / "cases/science/mendelian_genetics_known_plant_aer/public/submission.schema.json"
        ).read_text(encoding="utf-8")
    )
    assert external_schema == case_schema


def test_matched_pre_exposure_handoff_contains_only_public_observations(tmp_path):
    service = EpisodeService(
        "clean",
        0,
        101,
        tmp_path / "trajectory.jsonl",
        tmp_path / "windows.jsonl",
        10_000,
        matched_pre_exposure=True,
    )
    try:
        assert service.pre_exposure_observations
        assert service.env.get_aer_pea_case_summary()["comparable_visits"] >= 24
        prompt = _prompt(service, "original_end_to_end", "clean")
        assert "Matched public pre-exposure observations" in prompt
        assert "prefix actions are not provided" in prompt
        assert "Greenhouse activity since your last action:" in prompt

        trajectory = [
            json.loads(line)
            for line in service.trajectory_path.read_text(encoding="utf-8").splitlines()
        ]
        prefix = [event for event in trajectory if event["source"] == "matched_pre_exposure"]
        assert prefix
        assert all(event["request"]["command"] == "act" for event in prefix)
    finally:
        service.close()


def test_operator_case_roots_are_validated_and_replay_independently():
    env = ScienceWorldEnv("", serverPath=None, envStepLimit=10_000)
    try:
        with pytest.raises(ValueError, match="non-negative integer"):
            env.configure_aer_pea_case("clean", -1)
        with pytest.raises(ValueError, match="non-negative integer"):
            env.configure_aer_pea_case("clean", True)

        _, parent_actions, _ = _gold_and_parent_actions(env)
        root_events = {}
        for case_root in DEV_ROOTS[:2]:
            env.configure_aer_pea_case("clean", case_root)
            env.load(TASK, 0, "easy", generateGoldPath=False)
            _run_actions(env, parent_actions)
            first_summary = env.get_aer_pea_case_summary()
            first_events = env.get_aer_pea_case_events()
            assert first_summary["case_root"] == case_root

            env.load(TASK, 0, "easy", generateGoldPath=False)
            _run_actions(env, parent_actions)
            assert env.get_aer_pea_case_summary() == first_summary
            assert env.get_aer_pea_case_events() == first_events
            root_events[case_root] = first_events

        assert root_events[DEV_ROOTS[0]] != root_events[DEV_ROOTS[1]]
    finally:
        env.close()


def test_aer_task_surface_and_all_hidden_world_mechanisms_are_deterministic():
    env = ScienceWorldEnv("", serverPath=None, envStepLimit=10_000)
    try:
        assert TASK in env.get_task_names()
        assert env.get_max_variations(TASK) == 30
        assert set(env.get_aer_pea_case_worlds()) == SUPPORTED_WORLDS

        _, parent_actions, open_hive_idx = _gold_and_parent_actions(env)
        description = env.get_task_description().lower()
        assert "tall plant height" in description
        assert "flower" not in description
        assert "bee" not in description

        snapshots = _run_actions(env, parent_actions)
        first_flowering_look = next(
            result[0]
            for action, result in snapshots[open_hive_idx + 1 :]
            if action == "look around"
        )
        assert first_flowering_look.count("purple flower") >= 3
        assert first_flowering_look.count("white flower") >= 3
        assert any(
            "Greenhouse activity since your last action:" in result[0]
            for _, result in snapshots
        )

        results = {}
        reproduction_results = {}
        public_activity = {}
        for world in sorted(SUPPORTED_WORLDS):
            env.configure_aer_pea_case(world)
            env.load(TASK, 0, "easy", generateGoldPath=False)
            world_outputs = _run_actions(env, parent_actions)
            summary = env.get_aer_pea_case_summary()
            events = env.get_aer_pea_case_events()
            reproduction_events = env.get_aer_pea_case_reproduction_events()

            assert summary["world"] == world
            assert summary["mechanism"] == EXPECTED_MECHANISMS[world]
            assert summary["total_visits"] > 0
            assert summary["comparable_visits"] >= 30
            assert len(events) == summary["total_visits"]
            assert all("selection_weight" in event for event in events)
            results[world] = (summary, events)
            reproduction_results[world] = reproduction_events
            public_activity[world] = "\n".join(result[0] for _, result in world_outputs)

            env.load(TASK, 0, "easy", generateGoldPath=False)
            _run_actions(env, parent_actions)
            assert env.get_aer_pea_case_summary() == summary
            assert env.get_aer_pea_case_events() == events

        preferred, preferred_events = results["white_preference"]
        assert preferred["white_preference_weight"] == 9.0
        assert _comparable_fraction(preferred, "white") >= 0.80
        assert all(
            event["selection_weight"]
            == pytest.approx(9.0 / event["candidate_colors"].count("white"))
            for event in preferred_events
            if _is_comparable(event) and event["perceived_color"] == "white"
        )

        position, position_events = results["position_attraction"]
        assert position["preferred_flower_pot"] is not None
        assert position["preferred_position_visits"] / position["comparable_visits"] >= 0.80
        assert all(
            3.0 <= event["selection_weight"] <= 9.0
            for event in position_events
            if _is_comparable(event)
            and event["flower_pot"] == position["preferred_flower_pot"]
        )

        plant, plant_events = results["plant_attractiveness"]
        assert plant["preferred_plant_id"] is not None
        assert plant["preferred_plant_visits"] / plant["comparable_visits"] >= 0.80
        assert all(
            3.0 <= event["selection_weight"] <= 9.0
            for event in plant_events
            if _is_comparable(event) and event["plant_id"] == plant["preferred_plant_id"]
        )

        fertility, fertility_events = results["fertility_difference"]
        assert 0.35 <= _comparable_fraction(fertility, "white") <= 0.65
        assert fertility["fertility_accelerated_ticks"] > 0
        assert fertility["pollinations"] > 0
        assert fertility["fruit_sets"] > 0
        assert all(event["selection_weight"] == 1.0 for event in fertility_events)
        fertility_reproduction = reproduction_results["fertility_difference"]
        white_lags = [
            event["pollination_to_fruit_ticks"]
            for event in fertility_reproduction
            if event["event_type"] == "fruit_set" and event["native_color"] == "white"
        ]
        purple_lags = [
            event["pollination_to_fruit_ticks"]
            for event in fertility_reproduction
            if event["event_type"] == "fruit_set" and event["native_color"] == "purple"
        ]
        assert white_lags and purple_lags
        assert max(white_lags) < min(purple_lags)
        assert "was pollinated" in public_activity["fertility_difference"]
        assert "formed a fruit" in public_activity["fertility_difference"]

        transient, transient_events = results["transient_null"]
        assert transient["transient_root_seed"] is not None
        assert transient["initial_comparable_window_size"] == 6
        assert transient["initial_white_visits"] == 6
        assert transient["initial_purple_visits"] == 0
        assert 0.35 <= _comparable_fraction(transient, "white") <= 0.65
        assert all(event["selection_weight"] == 1.0 for event in transient_events)

        clean, clean_events = results["clean"]
        assert 0.35 <= _comparable_fraction(clean, "white") <= 0.65
        assert clean["clean_root_seed"] is not None
        assert 2 <= clean["initial_white_visits"] <= 4
        assert all(event["selection_weight"] == 1.0 for event in clean_events)
    finally:
        env.close()


def test_aer_all_hidden_worlds_preserve_g0_gold_completion():
    env = ScienceWorldEnv("", serverPath=None, envStepLimit=10_000)
    try:
        gold_actions, _, _ = _gold_and_parent_actions(env)

        for world in sorted(SUPPORTED_WORLDS):
            env.configure_aer_pea_case(world)
            env.load(TASK, 0, "easy", generateGoldPath=False)
            final_result = None
            for action in gold_actions:
                final_result = env.step(action)
                if final_result[2]:
                    break

            assert final_result is not None, world
            assert final_result[2] is True, world
            assert final_result[3]["score"] == 100, world
    finally:
        env.close()


def test_position_and_plant_worlds_follow_different_targets_after_swap():
    env = ScienceWorldEnv("", serverPath=None, envStepLimit=10_000)
    try:
        gold_actions, _, _ = _gold_and_parent_actions(env)
        open_hive_idx = next(
            idx
            for idx, action in enumerate(gold_actions)
            if action.startswith("open bee hive")
        )
        pre_release_actions = gold_actions[:open_hive_idx]

        for world in ("position_attraction", "plant_attractiveness"):
            env.configure_aer_pea_case(world)
            env.load(TASK, 0, "easy", generateGoldPath=False)
            _run_actions(env, pre_release_actions)
            look = env.look()
            tall_pot = re.search(
                r"a (flower pot \d+) \(containing a pea plant[^\n]*tall height", look
            ).group(1)
            short_pot = re.search(
                r"a (flower pot \d+) \(containing a pea plant[^\n]*short height", look
            ).group(1)
            empty_pot = next(
                pot
                for pot in re.findall(r"a (flower pot \d+) \(containing soil", look)
                if pot not in {tall_pot, short_pot}
            )

            env.step("open bee hive")
            for _ in range(20):
                env.step("wait1")
                summary = env.get_aer_pea_case_summary()
                target_ready = (
                    summary["preferred_flower_pot"] is not None
                    if world == "position_attraction"
                    else summary["preferred_plant_id"] is not None
                )
                if target_ready and summary["total_visits"]:
                    break

            cutoff = len(env.get_aer_pea_case_events())
            swap_actions = (
                f"move living thing in {short_pot} to {empty_pot}",
                f"move living thing in {tall_pot} to {short_pot}",
                f"move living thing in {empty_pot} to {tall_pot}",
            )
            for action in swap_actions:
                observation = env.step(action)[0]
                assert observation.startswith("You move"), (world, action, observation)
            for _ in range(12):
                env.step("wait")

            summary = env.get_aer_pea_case_summary()
            post_swap = env.get_aer_pea_case_events()[cutoff:]
            comparable = [event for event in post_swap if _is_comparable(event)]
            assert len(comparable) >= 8
            if world == "position_attraction":
                target_fraction = sum(
                    event["flower_pot"] == summary["preferred_flower_pot"]
                    for event in comparable
                ) / len(comparable)
                purple_fraction = sum(
                    event["perceived_color"] == "purple" for event in comparable
                ) / len(comparable)
                assert target_fraction >= 0.75
                assert purple_fraction >= 0.75
            else:
                target_fraction = sum(
                    event["plant_id"] == summary["preferred_plant_id"]
                    for event in comparable
                ) / len(comparable)
                white_fraction = sum(
                    event["perceived_color"] == "white" for event in comparable
                ) / len(comparable)
                assert target_fraction >= 0.75
                assert white_fraction >= 0.75
    finally:
        env.close()


def test_color_sleeves_separate_perceived_color_from_plant_identity():
    env = ScienceWorldEnv("", serverPath=None, envStepLimit=10_000)
    try:
        gold_actions, _, _ = _gold_and_parent_actions(env)
        open_hive_idx = next(
            idx
            for idx, action in enumerate(gold_actions)
            if action.startswith("open bee hive")
        )
        pre_release_actions = gold_actions[:open_hive_idx]

        for world in ("white_preference", "plant_attractiveness"):
            env.configure_aer_pea_case(world)
            env.load(TASK, 0, "easy", generateGoldPath=False)
            _run_actions(env, pre_release_actions)
            look = env.look()
            tall_pot = re.search(
                r"a (flower pot \d+) \(containing a pea plant[^\n]*tall height", look
            ).group(1)
            short_pot = re.search(
                r"a (flower pot \d+) \(containing a pea plant[^\n]*short height", look
            ).group(1)

            sleeve_actions = (
                f"move purple flower color sleeve to living thing in {short_pot}",
                f"move white flower color sleeve to living thing in {tall_pot}",
            )
            for action in sleeve_actions:
                observation = env.step(action)[0]
                assert observation.startswith("You move"), (world, action, observation)

            stacked = env.step(
                f"move white flower color sleeve to living thing in {short_pot}"
            )[0]
            assert "already has a flower color sleeve" in stacked

            removed = env.step("pick up purple flower color sleeve")[0]
            assert removed.startswith("You move")
            restored = env.step(
                f"move purple flower color sleeve to living thing in {short_pot}"
            )[0]
            assert restored.startswith("You move")
            native_description = env.look()
            assert "white flower" in native_description
            assert "purple flower" in native_description

            env.step("open bee hive")
            for _ in range(20):
                env.step("wait1")

            summary = env.get_aer_pea_case_summary()
            events = [event for event in env.get_aer_pea_case_events() if _is_comparable(event)]
            assert len(events) >= 8
            if world == "white_preference":
                white_fraction = sum(
                    event["perceived_color"] == "white" for event in events
                ) / len(events)
                tall_fraction = sum(
                    event["plant_height"] == "tall" for event in events
                ) / len(events)
                assert white_fraction >= 0.75
                assert tall_fraction >= 0.75
            else:
                target_fraction = sum(
                    event["plant_id"] == summary["preferred_plant_id"] for event in events
                ) / len(events)
                purple_fraction = sum(
                    event["perceived_color"] == "purple" for event in events
                ) / len(events)
                assert target_fraction >= 0.75
                assert purple_fraction >= 0.75
    finally:
        env.close()


def test_transient_null_streak_is_independent_of_agent_created_object_order():
    env = ScienceWorldEnv("", serverPath=None, envStepLimit=10_000)
    try:
        env.configure_aer_pea_case("transient_null")
        action_sequences = [
            [
                "look around",
                "go greenhouse",
                "look at pea seed in the seed stage in seed jar",
                "look at round green pea seed in the seed stage on cup",
                "look at ceramic cup",
                "look in seed jar",
                "move pea seed to flower pot 1",
                "0",
                "move pea seed to flower pot 2",
                "wait",
                "wait",
                "wait",
                "wait",
                "wait",
                "move pea seed to flower pot 2",
                "1",
                "wait",
                "wait",
                "wait",
                "open bee hive",
                "wait",
                "wait",
                "move pea seed on short pea plant to flower pot 3",
                "wait",
                "wait",
                "move living thing in reproducing short height pea plant to flower pot 3",
                "move living thing in reproducing tall height pea plant to flower pot 6",
                "wait",
                "wait",
                "wait",
            ],
            [
                "look around",
                "go greenhouse",
                "look at seed jar",
                "look at flower pot 1",
                "look at flower pot 2",
                "look at flower pot 3",
                "look at flower pot 6",
                "look at flower pot 7",
                "look at flower pot 9",
                "look at bee hive",
                "move organism in cup containing pea seed to flower pot 1",
                "move seed pea seed in seed jar containing pea seed to flower pot 2",
                "wait",
                "wait",
                "wait",
                "wait",
                "wait",
                "move round green pea seed in the seed stage in seed jar to flower pot 1",
                "move seed round green pea seed in seed jar containing pea seed to flower pot 2",
                "move pea seed to flower pot 1",
                "0",
                "move pea seed to flower pot 2",
                "wait",
                "1",
                "wait",
                "move pea seed to flower pot 2",
                "1",
                "wait",
                "wait",
                "open bee hive",
                "wait",
                "wait",
                "wait",
            ],
        ]

        for actions in action_sequences:
            env.load(TASK, 0, "easy", generateGoldPath=False)
            _run_actions(env, actions)
            summary = env.get_aer_pea_case_summary()

            assert summary["comparable_visits"] >= 6
            assert summary["initial_comparable_window_size"] == 6
            assert summary["initial_white_visits"] == 6
            assert summary["initial_purple_visits"] == 0
    finally:
        env.close()
