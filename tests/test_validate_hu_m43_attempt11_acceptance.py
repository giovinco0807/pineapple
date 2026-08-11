from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import ofc_regular.validate_hu_m43_attempt11_acceptance as acceptance


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "hu_joint_policy_m43_attempt11_population.json"


def test_attempt11_population_plan_is_fresh_fixed_and_opt_in() -> None:
    plan = acceptance.load_and_validate_attempt11_population_plan(PLAN)
    assert plan["seed"] == 190_108_071_901
    assert plan["seed_stride"] == 1_000_003
    assert plan["paired_seeds_per_opponent"] == 1000
    assert plan["shards"] == 20
    assert plan["paired_seeds_per_shard"] == 50
    assert plan["_freshness_counts"] == {
        "teacher_overlap_count": 0,
        "prior_population_overlap_count": 0,
    }
    assert plan["_freshness_registry"]["planned_overlap_count"] == 0
    assert plan["activation_guards"]["current_profile_changed"] is False
    assert plan["activation_guards"]["runtime_policy_activated"] is False
    assert plan["post_acceptance_activation"]["automatic_activation_allowed"] is False


def test_attempt11_population_plan_rejects_seed_or_registry_drift() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    changed = copy.deepcopy(plan)
    changed["seed"] += 1
    with pytest.raises(ValueError, match="schedule changed"):
        acceptance.load_and_validate_attempt11_population_plan(changed)
    changed = copy.deepcopy(plan)
    changed["freshness"]["excluded_schedule_registry_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hashed seed registry snapshot"):
        acceptance.load_and_validate_attempt11_population_plan(changed)


def test_attempt11_training_diagnostics_bind_variable_groups_and_rows() -> None:
    groups = [1 + index % 13 for index in range(200)]
    diagnostics = {
        "states": 200,
        "rows": sum(groups),
        "group_sizes": groups,
        "candidate_count_min": 0,
        "candidate_count_max": 12,
        "candidate_count_histogram": {
            str(candidate_count): sum(
                group - 1 == candidate_count for group in groups
            )
            for candidate_count in range(13)
        },
    }
    assert acceptance._valid_variable_training_diagnostics(diagnostics)
    changed = copy.deepcopy(diagnostics)
    changed["rows"] += 1
    assert not acceptance._valid_variable_training_diagnostics(changed)
    changed = copy.deepcopy(diagnostics)
    changed["group_sizes"][0] = 14
    assert not acceptance._valid_variable_training_diagnostics(changed)
    changed = copy.deepcopy(diagnostics)
    changed["candidate_count_histogram"]["0"] -= 1
    assert not acceptance._valid_variable_training_diagnostics(changed)
