"""Independent integration check for the v0.6.6 operator replay branch.

This test intentionally does not inspect the branch's hidden target fields.  Operator-only
event APIs are used only to prove that pre-branch history is preserved; the registered
measure is evaluated through an access-audited view of the solver response.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCIENCEWORLD_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SCIENCEWORLD_ROOT / "scripts"))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import run_aer_pea_experiment_replay_v0_6_6 as replay  # noqa: E402


def _public_and_history_snapshot(service: Any) -> dict[str, Any]:
    """Capture physical/public state and history, excluding branch configuration."""

    public_rows = replay._normalized_public_rows(replay.read_jsonl(service.trajectory_path))
    operator_rows = replay._without_volatile_timestamps(
        replay.read_jsonl(service.operator_window_path)
    )
    return {
        "solver_visible": {
            "pots": service.env.get_aer_pea_case_public_status(),
            "look": service.env.look(),
            "inventory": service.env.inventory(),
        },
        "operator_history": {
            "flower_visits": service.env.get_aer_pea_case_events(),
            "reproduction": service.env.get_aer_pea_case_reproduction_events(),
        },
        "public_history": public_rows,
        "operator_action_history": operator_rows,
        "cursors": {
            "public_index": service._index,
            "note_index": service._note_index,
            "experiment_ids": sorted(service._experiment_ids),
            "active_experiment_id": service._active_experiment_id,
            "completed": service.completed,
            "public_row_count": len(public_rows),
            "operator_row_count": len(operator_rows),
        },
    }


class _AccessAuditedMapping(Mapping[str, Any]):
    """Mapping view that records every field read, recursively through list items."""

    def __init__(
        self,
        value: Mapping[str, Any],
        accesses: list[tuple[str | int, ...]],
        path: tuple[str | int, ...] = (),
    ) -> None:
        self._value = value
        self._accesses = accesses
        self._path = path

    def __getitem__(self, key: str) -> Any:
        path = (*self._path, key)
        self._accesses.append(path)
        return _audit_value(self._value[key], self._accesses, path)

    def __iter__(self):
        return iter(self._value)

    def __len__(self) -> int:
        return len(self._value)


def _audit_value(
    value: Any,
    accesses: list[tuple[str | int, ...]],
    path: tuple[str | int, ...],
) -> Any:
    if isinstance(value, Mapping):
        return _AccessAuditedMapping(value, accesses, path)
    if isinstance(value, list):
        return [_audit_value(item, accesses, (*path, index)) for index, item in enumerate(value)]
    return value


def _measure_from_solver_response(
    response: Mapping[str, Any], measure: Mapping[str, Any]
) -> tuple[float, dict[str, int], tuple[tuple[str | int, ...], ...]]:
    accesses: list[tuple[str | int, ...]] = []
    audited = _AccessAuditedMapping(response, accesses)
    value, counts, missing_reason = replay._measure_visit_share(audited, measure)

    accessed_fields = {path[-1] for path in accesses if isinstance(path[-1], str)}
    assert accessed_fields == {
        "visits_by_flower",
        "flower_pot",
        "visit_count",
        "observed_visit_count",
    }
    assert accessed_fields.isdisjoint(measure["hidden_fields_forbidden"])
    assert missing_reason is None
    assert value is not None
    return value, counts, tuple(accesses)


def _profile(plan: Mapping[str, Any], world_id: str) -> Mapping[str, Any]:
    return next(item for item in plan["candidate_profiles"] if item["id"] == world_id)


def _branch_and_observe(runtime: Any, world_id: str, seed: int) -> dict[str, Any]:
    fingerprint = replay._reconstruct_pre_action(runtime, verify_archived_prefix=True)
    assert fingerprint == runtime.plan["restore"]["prestate_fingerprint"]

    service = runtime.service
    before = _public_and_history_snapshot(service)
    profile = _profile(runtime.plan, world_id)
    service.env.configure_aer_pea_replay_branch(
        profile["world"],
        seed,
        preference_weight=profile["preference_weight"],
        noise_levels=dict(profile["noise_levels"]),
    )
    after = _public_and_history_snapshot(service)
    assert after == before

    response = service.handle(dict(runtime.plan["source"]["target_request"]))
    assert response["ok"] is True
    share, counts, accesses = _measure_from_solver_response(response, runtime.plan["measure"])
    return {
        "response": response,
        "share": share,
        "counts": counts,
        "measure_accesses": accesses,
    }


def _weighted_mode(visits: list[Mapping[str, Any]], field: str) -> str | int:
    counts: Counter[str | int] = Counter()
    for visit in visits:
        counts[visit[field]] += int(visit["visit_count"])
    assert counts
    return counts.most_common(1)[0][0]


def test_archived_swap_branch_is_state_preserving_matched_and_causal() -> None:
    contract = replay._read_json(replay.DEFAULT_CONTRACT)
    profile = replay.validate_contract(contract, repository_root=replay.REPOSITORY_ROOT)
    plan = replay._build_plan_skeleton(
        contract,
        profile,
        contract_path=replay.DEFAULT_CONTRACT,
        seed_count=1,
    )

    runtime = replay._ReusableRuntime(plan)
    try:
        first_fingerprint = replay._reconstruct_pre_action(runtime, verify_archived_prefix=True)
        first_prestate = _public_and_history_snapshot(runtime.service)
        second_fingerprint = replay._reconstruct_pre_action(runtime, verify_archived_prefix=True)
        second_prestate = _public_and_history_snapshot(runtime.service)
        assert first_fingerprint == second_fingerprint
        assert first_prestate == second_prestate
        plan["restore"] = {
            "count": 2,
            "fingerprints": [first_fingerprint, second_fingerprint],
            "prestate_fingerprint": first_fingerprint,
            "semantic_match": "exact_sha256",
            "archived_prefix_match": True,
        }
        assert (
            second_prestate["cursors"]["public_index"] == plan["source"]["pre_action_public_index"]
        )

        pots = {pot["name"]: pot for pot in second_prestate["solver_visible"]["pots"]["pots"]}
        assert pots["flower pot 7"]["plants"][0]["height"] == "short"
        assert pots["flower pot 15"]["plants"][0]["height"] == "tall"

        comparable_history = [
            event
            for event in second_prestate["operator_history"]["flower_visits"]
            if {"white", "purple"}.issubset(event["candidate_colors"])
        ]
        assert comparable_history
        first_comparable = comparable_history[0]
        # These are behavioral event coordinates, not hidden target/weight fields.
        anchor_pot = first_comparable["flower_pot"]
        anchor_plant = first_comparable["plant_id"]
        assert anchor_pot == "flower pot 15"
        assert anchor_plant == pots["flower pot 7"]["plants"][0]["plant_id"]

        seed = plan["seed_manifest"][0]
        position_first = _branch_and_observe(runtime, "M2", seed)
        position_second = _branch_and_observe(runtime, "M2", seed)
        plant = _branch_and_observe(runtime, "M3", seed)

        # A full restore plus the same registered seed/World is exactly repeatable.
        assert json.dumps(
            position_first["response"], sort_keys=True, separators=(",", ":")
        ) == json.dumps(position_second["response"], sort_keys=True, separators=(",", ":"))
        assert position_first["share"] == position_second["share"]
        assert position_first["counts"] == position_second["counts"]

        # The plants were swapped before the branch: position follows the old pot, while
        # plant attractiveness follows the old plant now located in the opposite pot.
        assert position_first["share"] > 0.50
        assert plant["share"] < 0.50
        assert position_first["share"] > plant["share"]
        position_visits = position_first["response"]["visits_by_flower"]
        plant_visits = plant["response"]["visits_by_flower"]
        assert _weighted_mode(position_visits, "flower_pot") == anchor_pot
        assert _weighted_mode(position_visits, "plant_id") != anchor_plant
        assert _weighted_mode(plant_visits, "plant_id") == anchor_plant
        assert _weighted_mode(plant_visits, "flower_pot") != anchor_pot
    finally:
        runtime.close()
