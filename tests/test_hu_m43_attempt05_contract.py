from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from ofc_regular import hu_m43_attempt05_contract as contract


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "hu_joint_policy_m43_attempt05.json"
STATUS = ROOT / "configs" / "hu_joint_policy_m43_attempt05_status.json"


def test_attempt05_plan_freezes_top4_e4_e8_and_independent_e128_labels() -> None:
    plan = contract.load_and_validate_attempt05_plan(PLAN)
    assert plan["teacher_search"]["candidate_top_k"] == 4
    assert plan["teacher_search"]["search_evaluation_samples_candidates"] == [4, 8]
    assert plan["teacher_search"]["final_teacher_label_evaluation_samples"] == 128
    assert plan["teacher_search"][
        "candidate_selection_and_final_evaluation_rng_disjoint"
    ] is True


def test_attempt05_development_is_1000_balanced_roots_in_100_small_shards() -> None:
    plan = contract.load_and_validate_attempt05_plan(PLAN)
    roles = plan["development_roles"]
    assert {role: spec["roots"] for role, spec in roles.items()} == {
        "pilot_train": 200,
        "pilot_audit": 50,
        "expand_train": 600,
        "final_audit": 150,
    }
    assert sum(spec["roots"] for spec in roles.values()) == 1000
    assert sum(spec["shards"] for spec in roles.values()) == 100
    for spec in roles.values():
        assert spec["shards"] * 10 == spec["roots"]
        assert spec["roots_per_profile"] * 5 == spec["roots"]


def test_attempt05_inherits_exact_attempt04_acceptance_and_population_schedules() -> None:
    plan = contract.load_and_validate_attempt05_plan(PLAN)
    roles = plan["inherited_attempt04_acceptance_roles"]
    assert {
        role: (
            spec["roots"],
            spec["seed_start"],
            spec["candidate_seed_start"],
            spec["evaluation_seed_start"],
            spec["child_policy_seed_start"],
        )
        for role, spec in roles.items()
    } == {
        "precal_holdout": (300, 13106071901, 20106071901, 21106071901, 22106071901),
        "calibration_safety_fit": (100, 13506071901, 20506071901, 21506071901, 22506071901),
        "calibration_threshold_lock": (100, 13606071901, 20606071901, 21606071901, 22606071901),
        "locked_holdout": (200, 13806071901, 20806071901, 21806071901, 22806071901),
    }
    population = plan["population_acceptance"]
    assert population["seed_start"] == 14106071901
    assert population["seed_stride"] == 1000003
    assert population["paired_seeds_per_opponent"] == 1000


def test_attempt05_freezes_threshold_grid_and_dev900_as_nonfresh() -> None:
    plan = contract.load_and_validate_attempt05_plan(PLAN)
    assert tuple(plan["threshold_contract"]["grid"]) == (
        contract.M43_ATTEMPT05_THRESHOLD_GRID
    )
    assert plan["prior_attempt04_development"]["eligible_count"] == 0
    assert plan["prior_attempt04_development"]["classification"] == (
        "development_only_already_consumed_not_a_fresh_gate"
    )
    assert plan["prior_attempt04_development"][
        "acceptance_or_generalization_claim_allowed"
    ] is False


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda value: value["teacher_search"].__setitem__("candidate_top_k", 8),
            "topK4/e4-e8/e128",
        ),
        (
            lambda value: value["development_roles"]["pilot_audit"].__setitem__(
                "seed_start", 17306071902
            ),
            "pilot_audit changed: seed_start",
        ),
        (
            lambda value: value["freshness"]["excluded_rng_schedules"].pop(),
            "excluded_rng_schedules changed",
        ),
        (
            lambda value: value["threshold_contract"]["grid"].append(0.995),
            "threshold grid",
        ),
        (
            lambda value: value["activation_guards"].__setitem__(
                "runtime_policy_activated", True
            ),
            "activation guard",
        ),
    ],
)
def test_attempt05_plan_rejects_mutated_frozen_boundaries(mutator, message: str) -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    mutator(plan)
    with pytest.raises(ValueError, match=message):
        contract.validate_attempt05_plan(plan)


def test_attempt05_status_closes_architecture_no_go_without_opening_fresh_roles() -> None:
    status = contract.load_and_validate_attempt05_status(STATUS)
    assert status["status"] == "complete_no_go_architecture_one_shot"
    architecture = status["architecture_one_shot"]
    assert architecture["selected_family"] is None
    assert architecture["winner_is_runtime_frozen"] is False
    assert all(
        family["raw_gate_pass"] is False
        for family in architecture["families"].values()
    )
    assert status["development_roles"] == {
        "pilot_train": "not_started_closed_by_architecture_no_go",
        "pilot_audit": "not_started_unopened",
        "expand_train": "not_started_unopened",
        "final_audit": "not_started_unopened",
    }
    assert all(
        value == "unopened"
        for value in status["inherited_attempt04_acceptance_roles"].values()
    )
    assert status["population_acceptance"] == "unopened"
    assert status["spot_execution"]["scripts_overhauled"] is False
    assert status["spot_execution"]["run_started"] is False
    assert all(value is False for value in status["guards"].values())


def test_attempt05_status_binds_one_shot_trust_audit_and_timing_boundary() -> None:
    status = contract.load_and_validate_attempt05_status(STATUS)
    architecture = status["architecture_one_shot"]
    architecture_path = ROOT / architecture["report_path"]
    architecture_bytes = architecture_path.read_bytes()
    architecture_report = json.loads(architecture_bytes)
    assert hashlib.sha256(architecture_bytes).hexdigest() == architecture["report_sha256"]
    assert architecture_report["selected_family"] is None
    assert architecture_report["status"] == architecture["report_status"]

    trust = status["postrun_trust_audit"]
    trust_bytes = (ROOT / trust["path"]).read_bytes()
    trust_report = json.loads(trust_bytes)
    assert hashlib.sha256(trust_bytes).hexdigest() == trust["sha256"]
    assert trust_report["status"] == trust["status"]
    assert trust_report["one_shot_report_sha256"] == architecture["report_sha256"]
    assert trust_report["one_shot_wall_seconds"] == trust["one_shot_wall_seconds"]
    assert trust_report["one_shot_selected_family"] is None
    assert trust_report["new_audit_authorized"] is False
    assert trust_report["spot_started"] is False

    timing = status["search_feasibility_boundary"]
    assert timing["exact_all_legal_actions_wall_seconds"] == pytest.approx(400.711)
    assert timing["mc1_wall_seconds"] == pytest.approx(2.509)
    assert timing["exact_all_scaled_generation_authorized"] is False
    assert timing["mc1_quality_or_acceptance_claim_allowed"] is False
    assert status["attempt06_boundary"] == {
        "top4_diagnostic_status": "complete_design_evidence_only",
        "candidate_set_diagnostic_status": (
            "complete_metric_corrected_design_evidence_only"
        ),
        "attempt06_plan": {
            "path": "configs/hu_joint_policy_m43_attempt06.json",
            "schema": "hu_m43_attempt06_corrected_pre_fresh_plan_v3",
            "sha256": (
                "4844fb970780c04ff093eb43b1672e403f006515c47b287e6abdbea17867f5b8"
            ),
        },
        "plan_frozen": True,
        "selected_search_design": {
            "candidate_generator": "lambda_rank_artifact_design_only",
            "learned_nonbaseline_top_k": 8,
            "candidate_selection_samples": 8,
            "independent_evaluation_samples": 128,
        },
        "runtime_model_selected": False,
        "fresh_generation_authorized": False,
        "spot_authorized": False,
        "runtime_authorized": False,
    }
    assert (ROOT / status["postmortem"]).is_file()


def test_attempt05_contract_binds_unchanged_ai_profiles_registry() -> None:
    contract.load_and_validate_attempt05_plan(PLAN)
    actual = hashlib.sha256(
        (ROOT / "src" / "ofc_regular" / "ai_profiles.py").read_bytes()
    ).hexdigest()
    assert actual == contract.AI_PROFILES_SHA256


def test_attempt05_status_rejects_claimed_execution() -> None:
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    mutated = copy.deepcopy(status)
    mutated["spot_execution"]["run_started"] = True
    with pytest.raises(ValueError, match="Spot execution"):
        contract.validate_attempt05_status(mutated)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda value: value["architecture_one_shot"].__setitem__(
                "selected_family", "lambda_rank"
            ),
            "architecture No-Go",
        ),
        (
            lambda value: value["attempt06_boundary"].__setitem__(
                "fresh_generation_authorized", True
            ),
            "handoff",
        ),
        (
            lambda value: value["guards"].__setitem__(
                "current_profile_changed", True
            ),
            "activation guard",
        ),
    ],
)
def test_attempt05_status_rejects_no_go_or_attempt06_boundary_drift(
    mutator, message: str
) -> None:
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    mutator(status)
    with pytest.raises(ValueError, match=message):
        contract.validate_attempt05_status(status)
