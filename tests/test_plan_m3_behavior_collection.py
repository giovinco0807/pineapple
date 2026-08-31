import copy
import json
from pathlib import Path

import pytest

from ai.tutor.behavior_calibration_contract import canonical_json, canonical_sha256
from ai.tutor.behavior_temperature_calibration import (
    ROLE_KEYS,
    build_temperature_gate_config,
)
from ai.tutor.plan_m3_behavior_collection import (
    DEFAULT_CHALLENGE_SEED_NAMESPACE,
    DEFAULT_NATURAL_SEED_NAMESPACE,
    PLAN_SCHEMA,
    build_behavior_collection_plan,
    build_smoke_size_observation,
    verify_behavior_collection_plan,
    verify_size_observation,
)


@pytest.fixture(scope="module")
def size_observation():
    return build_smoke_size_observation(Path("."))


def _small_gate(*, test_count=2):
    return build_temperature_gate_config(
        gate_id="m3_collection_plan_test_gate_v2",
        min_fit_decisions_per_role=7,
        min_dev_decisions_per_role=3,
        min_test_decisions_per_role=test_count,
        min_challenge_decisions_per_role_joker=2,
        min_roots_per_split_role=1,
        min_challenge_roots_per_role_joker=2,
        bootstrap_replicates=20,
    )


def _small_plan(size_observation):
    gate = _small_gate()
    return gate, build_behavior_collection_plan(
        gate_config=gate,
        size_observation=size_observation,
        natural_seed_namespace="m3-collection-plan-tests-natural-v1",
        challenge_seed_namespace="m3-collection-plan-tests-challenge-v1",
        shard_size=5,
        plan_id="m3_collection_plan_tests",
    )


def test_plan_scans_exact_minimum_prefixes_and_rederives(size_observation):
    gate, plan = _small_plan(size_observation)
    assert plan["schema"] == PLAN_SCHEMA
    assert plan["large_collection_executed"] is False
    assert plan["strategic_strength_evaluated"] is False
    assert plan["strategic_strength_claimed"] is False
    assert plan["promotion_eligible"] is False

    natural = plan["natural"]
    required = natural["required_decisions_per_role_by_split"]
    achieved = natural["achieved_root_and_decision_counts_by_split"]
    assert all(achieved[split] >= required[split] for split in required)
    assert natural["minimum_prefix_proof"]["previous_prefix_satisfied"] is False
    assert natural["minimum_prefix_proof"]["unmet_splits_before_final_root"]
    for split, split_count in achieved.items():
        assert natural["achieved_decision_counts_by_split_role"][split] == {
            role: split_count for role in ROLE_KEYS
        }

    challenge = plan["joker_challenge"]
    assert challenge["cycle_length_roots"] == 12
    assert len({entry["cell"] for entry in challenge["cycle"]}) == 12
    assert challenge["range"]["root_count"] == 24
    assert challenge["targeted_decision_count"] == 24
    assert challenge["total_logged_decision_count"] == 96
    assert set(challenge["achieved_targeted_root_and_decision_counts_by_cell"].values()) == {2}
    assert challenge["minimum_prefix_proof"]["previous_prefix_satisfied"] is False
    assert len(challenge["minimum_prefix_proof"]["unmet_cells_before_final_root"]) == 1

    assert verify_behavior_collection_plan(
        plan, gate_config=gate, workspace_root=Path(".")
    ) == plan


def test_verifier_rejects_rehashed_derived_field_tamper(size_observation):
    gate, plan = _small_plan(size_observation)
    tampered = copy.deepcopy(plan)
    tampered["natural"]["range"]["root_count"] += 1
    unsigned = dict(tampered)
    unsigned.pop("plan_sha256")
    tampered["plan_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="exact rederivation"):
        verify_behavior_collection_plan(tampered, gate_config=gate)


def test_verifier_rejects_temperature_gate_config_drift(size_observation):
    gate, plan = _small_plan(size_observation)
    drifted_gate = _small_gate(test_count=3)
    assert drifted_gate["gate_config_sha256"] != gate["gate_config_sha256"]
    with pytest.raises(ValueError, match="gate config drift"):
        verify_behavior_collection_plan(plan, gate_config=drifted_gate)


def test_size_estimate_is_bound_to_saved_smoke_bytes_and_rejects_drift(
    size_observation,
):
    assert verify_size_observation(size_observation, workspace_root=Path(".")) == size_observation
    for population, directory in (
        ("natural", "natural"),
        ("joker_challenge", "challenge"),
    ):
        files = size_observation["populations"][population]["files"]
        assert files["decisions"]["observed_bytes"] == (
            Path("ai/reports/m3_behavior_calibration_smoke_20260713")
            / directory
            / "decisions.jsonl"
        ).stat().st_size

    tampered = copy.deepcopy(size_observation)
    tampered["populations"]["natural"]["files"]["decisions"][
        "observed_bytes"
    ] += 1
    tampered["populations"]["natural"]["observed_payload_bytes"] += 1
    unsigned = dict(tampered)
    unsigned.pop("observation_sha256")
    tampered["observation_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="size source drift"):
        verify_size_observation(tampered, workspace_root=Path("."))


def test_saved_production_plan_is_canonical_self_bound_and_nonpromoting():
    path = Path("ai/reports/m3_behavior_collection_plan_20260713/plan.json")
    if not path.is_file():
        pytest.skip("saved production behavior collection plan is not present")
    raw = path.read_text(encoding="utf-8")
    assert raw.endswith("\n") and raw.count("\n") == 1
    plan = json.loads(raw)
    assert canonical_json(plan) == raw[:-1]
    gate = build_temperature_gate_config()
    assert verify_behavior_collection_plan(
        plan, gate_config=gate, workspace_root=Path(".")
    ) == plan
    assert plan["natural"]["collection_config"]["seed_namespace"] == (
        DEFAULT_NATURAL_SEED_NAMESPACE
    )
    assert plan["joker_challenge"]["collection_config"]["seed_namespace"] == (
        DEFAULT_CHALLENGE_SEED_NAMESPACE
    )
    assert plan["natural"]["required_decisions_per_role_by_split"] == {
        "fit": 50_000,
        "dev": 10_000,
        "test": 20_000,
    }
    assert plan["joker_challenge"][
        "required_targeted_roots_and_decisions_per_cell"
    ] == 2_000
    assert plan["joker_challenge"]["range"]["root_count"] == 24_000
    assert plan["claims"]["proves_collection_was_run"] is False
    assert plan["claims"]["proves_m3_strength"] is False
    assert plan["claims"]["promotes_any_policy_or_behavior_model"] is False
