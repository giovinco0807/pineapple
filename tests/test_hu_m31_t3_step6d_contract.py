from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from ofc_regular.hu_m31_t3_step6d_contract import (
    EXPECTED_STEP6D_CONTRACT_BYTE_SHA256,
    EXPECTED_STEP6D_CONTRACT_CANONICAL_SHA256,
    HISTORICAL_CONFIG_SEED_MAX,
    PLANNED_SEED_COUNT,
    PLANNED_SEED_MAX,
    PLANNED_SEED_MIN,
    PLANNED_SEED_SET_SHA256,
    SEED_SCHEDULES,
    SEED_STRIDE,
    STEP6D_RUN_ID,
    canonical_sha256,
    planned_seed_values,
    schedule_by_name,
    seed_schedule_payload,
    validate_seed_schedule,
)
from ofc_regular.validate_hu_m31_t3_step6d_contract import (
    validate_contract_file,
    validate_contract_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO_ROOT / "configs/hu_joint_policy_m31_t3_step6d_contract.json"


def _contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _set_path(payload: Any, path: tuple[str | int, ...], value: Any) -> None:
    target = payload
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value


def test_step6d_contract_hashes_are_frozen() -> None:
    raw = CONTRACT_PATH.read_bytes()
    payload = json.loads(raw)
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_STEP6D_CONTRACT_BYTE_SHA256
    assert canonical_sha256(payload) == EXPECTED_STEP6D_CONTRACT_CANONICAL_SHA256
    assert STEP6D_RUN_ID == "hu-m31-step6d-performance-repair-attempt01-v1"


def test_all_seed_splits_and_namespaces_are_exact_and_disjoint() -> None:
    payload = _contract()["seed_contract"]
    values = planned_seed_values()
    audit = validate_seed_schedule()
    assert payload["seed_stride"] == SEED_STRIDE == 1_000_003
    assert payload["schedules"] == seed_schedule_payload()
    assert len(SEED_SCHEDULES) == 11
    assert len(values) == len(set(values)) == PLANNED_SEED_COUNT == 67_500
    assert min(values) == PLANNED_SEED_MIN == 480_108_071_901
    assert max(values) == PLANNED_SEED_MAX == 680_607_073_398
    assert min(values) > HISTORICAL_CONFIG_SEED_MAX
    assert audit["planned_seed_set_sha256"] == PLANNED_SEED_SET_SHA256
    assert audit["schedule_seed_counts"] == {
        "performance_development": 600,
        "performance_lock": 600,
        "quality_pilot": 300,
        "train": 36_000,
        "safety_fit": 6_000,
        "threshold_lock": 6_000,
        "diagnostic_teacher_holdout": 6_000,
        "development_population": 1_500,
        "locked_population": 6_000,
        "abr_development": 1_500,
        "locked_abr": 3_000,
    }


def test_seed_role_eligibility_and_locking_are_frozen() -> None:
    assert {
        schedule.name for schedule in SEED_SCHEDULES if schedule.training_eligible
    } == {"train", "safety_fit"}
    assert {
        schedule.name
        for schedule in SEED_SCHEDULES
        if schedule.locked_before_content_read
    } == {
        "performance_lock",
        "quality_pilot",
        "threshold_lock",
        "diagnostic_teacher_holdout",
        "locked_population",
        "locked_abr",
    }
    assert schedule_by_name("performance_development").namespace_kind == "teacher"
    assert schedule_by_name("locked_population").namespace_kind == "population"
    with pytest.raises(KeyError, match="unknown"):
        schedule_by_name("step6c")


def test_substep_order_and_critical_gates_are_frozen() -> None:
    substeps = _contract()["substeps"]
    assert list(substeps) == [
        "contract_freeze",
        "performance_development",
        "performance_lock",
        "quality_pilot",
        "artifact_rebuild",
        "model_and_threshold_freeze",
        "development_population",
        "locked_promotion",
    ]
    assert substeps["performance_development"]["search_budget"] == {
        "candidate_samples": 8,
        "evaluation_samples": 32,
        "downstream_t3_samples": 4,
        "downstream_t4_samples": 0,
    }
    assert substeps["performance_lock"]["go_gates"] == {
        "missing_or_censored_roots_max": 0,
        "portable_semantic_parity_fraction": 1.0,
        "first_p95_hard_seconds_max": 180.0,
        "first_p95_quality_launch_guard_seconds_max": 150.0,
        "first_p99_and_max_seconds_max": 240.0,
        "second_p95_hard_seconds_max": 6.0,
        "second_p95_quality_launch_guard_seconds_max": 5.0,
        "peak_rss_hard_bytes_max": 1_073_741_824,
        "peak_rss_quality_launch_guard_bytes_max": 858_993_459,
    }
    assert substeps["quality_pilot"]["confirmation_selected_regret_gates"] == {
        "mean_max": 0.75,
        "p95_max": 3.0,
        "p99_max": 6.0,
        "max_max": 15.0,
    }
    assert substeps["artifact_rebuild"]["paired_hands"] == {
        "train": 6_000,
        "safety_fit": 1_000,
        "threshold_lock": 1_000,
        "diagnostic_teacher_holdout": 1_000,
    }
    assert (
        substeps["model_and_threshold_freeze"]["teacher_ev_lcb_is_runtime_gate"]
        is False
    )
    assert substeps["locked_promotion"]["current_profile_changed"] is False


def test_reuse_matrix_allows_only_predeclared_evidence_roles() -> None:
    reuse = _contract()["reuse_matrix"]
    flags = (
        "training_eligible",
        "threshold_eligible",
        "quality_evidence",
        "promotion_evidence",
    )
    true_by_flag = {
        flag: {name for name, row in reuse.items() if row[flag]} for flag in flags
    }
    assert true_by_flag == {
        "training_eligible": {"train", "safety_fit"},
        "threshold_eligible": {"threshold_lock"},
        "quality_evidence": {"quality_pilot"},
        "promotion_evidence": {"locked_population", "locked_abr"},
    }
    assert all(reuse[name]["training_eligible"] is False for name in (
        "step6a_step6b_rows",
        "step6c_rows",
        "performance_development",
        "performance_lock",
        "quality_pilot",
    ))


def test_live_contract_step6c_and_current_policy_anchors_validate() -> None:
    report = validate_contract_file(CONTRACT_PATH, repo_root=REPO_ROOT)
    assert report["status"] == "pass"
    assert report["contract_canonical_sha256"] == (
        EXPECTED_STEP6D_CONTRACT_CANONICAL_SHA256
    )
    assert report["checks"]["step6c_complete_no_go_anchored"] is True
    assert report["checks"]["policy_registry_and_current_resolution_anchored"] is True
    assert report["anchors"]["step6c_quality"]["byte_sha256"] == (
        "66ddae312ee9561a0101ba94deb5e5e867d407dda78af00b03da10fc96739731"
    )
    assert report["anchors"]["policy_registry"]["byte_sha256"] == (
        "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
    )
    for flag in (
        "performance_execution_authorized",
        "quality_pilot_authorized",
        "artifact_fanout_authorized",
        "training_authorized",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
        "full_replacement_enabled",
        "m31_complete",
    ):
        assert report[flag] is False


def test_step6c_no_go_is_recorded_exactly() -> None:
    outcome = _contract()["step6c_fixed_outcome"]
    quality = json.loads(
        (
            REPO_ROOT
            / "outputs/hu_joint_policy/m31_t3_step6c/quality100_validation_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert outcome["status"] == "complete_no_go"
    assert outcome["observed_first_p95_seconds"] == 326.2574110039998
    assert outcome["reopened"] is False
    assert outcome["rows_training_eligible"] is False
    assert quality["all_gates_passed"] is False
    assert {name for name, passed in quality["gates"].items() if not passed} == {
        "all_shard_summaries_pass",
        "first_primary_p95_within_180_seconds",
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("step6c_fixed_outcome", "reseed_allowed"), True),
        (("immutable_boundaries", "approximate_t4_allowed"), True),
        (("seed_contract", "schedules", 0, "namespace_bases", "hand"), 1),
        (
            (
                "substeps",
                "performance_lock",
                "go_gates",
                "first_p95_hard_seconds_max",
            ),
            181.0,
        ),
        (
            (
                "substeps",
                "quality_pilot",
                "confirmation_selected_regret_gates",
                "p95_max",
            ),
            4.0,
        ),
        (("substeps", "artifact_rebuild", "confirmation_fraction_each_split"), 0.0),
        (
            (
                "substeps",
                "model_and_threshold_freeze",
                "model_gates",
                "teacher_regret_p95_max",
            ),
            4.0,
        ),
        (
            (
                "substeps",
                "locked_promotion",
                "promotion_gates",
                "paired_delta_ev_per_hand_ci95_low_min_exclusive",
            ),
            -0.01,
        ),
        (
            (
                "substeps",
                "locked_promotion",
                "exploitability_proxy_gates",
                "worst_response_ev_per_hand_min",
            ),
            -1.0,
        ),
        (("reuse_matrix", "step6c_rows", "training_eligible"), True),
        (("activation_guards", "current_profile_changed"), True),
    ],
)
def test_every_gate_family_fails_closed_without_outer_contract_hash(
    path: tuple[str | int, ...], value: Any
) -> None:
    payload = copy.deepcopy(_contract())
    _set_path(payload, path, value)
    with pytest.raises(ValueError, match="changed|enabled"):
        validate_contract_payload(
            payload,
            repo_root=REPO_ROOT,
            verify_anchors=False,
            verify_contract_hash=False,
        )


def test_whole_contract_hash_rejects_even_non_gate_mutation() -> None:
    payload = copy.deepcopy(_contract())
    payload["next_step"] = "run everything"
    with pytest.raises(ValueError, match="canonical contract"):
        validate_contract_payload(payload, repo_root=REPO_ROOT, verify_anchors=False)


def test_byte_different_contract_file_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "contract.json"
    target.write_bytes(CONTRACT_PATH.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="byte SHA-256"):
        validate_contract_file(target, repo_root=REPO_ROOT, verify_anchors=False)


def test_missing_live_anchor_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="anchor file/hash mismatch"):
        validate_contract_payload(_contract(), repo_root=tmp_path)
