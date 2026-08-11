from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLOSEOUT = ROOT / "configs" / "hu_joint_policy_m43_attempt06_closeout.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_attempt06_closeout_matches_immutable_audit_and_baseline() -> None:
    closeout = json.loads(CLOSEOUT.read_text(encoding="utf-8"))
    evidence = closeout["immutable_evidence"]
    audit_path = ROOT / evidence["search_quality_audit_path"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    assert closeout["status"] == "complete_no_go_search_quality"
    assert closeout["decision"] == audit["decision_scope"]
    assert audit["decision"] == "no_go"
    assert evidence["search_quality_audit_sha256"] == _sha256(audit_path)
    assert evidence["plan_sha256"] == audit["source"]["plan_sha256"]
    assert evidence["audit_consumption_marker_sha256"] == audit["source"][
        "consumption_marker_sha256"
    ]
    assert evidence["merged_teacher_sha256"] == audit["source"][
        "merged_teacher_sha256"
    ]
    assert evidence["merge_receipt_sha256"] == audit["source"][
        "merge_receipt_sha256"
    ]

    observed = {row["name"]: row["observed"] for row in audit["gates"]}
    assert closeout["failed_quality_gates"] == {
        "mean_delta_per_state": observed["mean_delta_per_state"],
        "mean_delta_per_fire": observed["mean_delta_per_fire"],
        "false_positive_rate_per_fire": observed[
            "false_positive_rate_per_fire"
        ],
        "metric_b_p95_loss": observed["metric_b_p95_loss"],
        "metric_b_p99_loss": observed["metric_b_p99_loss"],
        "metric_b_max_loss": observed["metric_b_max_loss"],
    }
    assert closeout["passed_integrity_gates"] == {
        "action_mapping_violation_count": 0,
        "rng_domain_violation_count": 0,
        "hidden_information_violation_count": 0,
        "nonfire_counterfactual_cancellation_verified": True,
    }
    assert closeout["baseline_hash_audit"]["policy_registry_sha256"] == _sha256(
        ROOT / "src" / "ofc_regular" / "ai_profiles.py"
    )


def test_attempt06_closeout_stays_fail_closed() -> None:
    closeout = json.loads(CLOSEOUT.read_text(encoding="utf-8"))
    boundary = closeout["science_boundary"]

    assert boundary["attempt06_threshold_reselected"] is False
    assert boundary["attempt06_same_seed_retry_allowed"] is False
    assert boundary["attempt06_fit_performed"] is False
    assert boundary["attempt06_model_changed"] is False
    assert boundary["attempt06_spot_expansion_authorized"] is False
    assert boundary["teacher_values_are_realized_match_ev"] is False
    assert boundary["teacher_lcb_used_as_runtime_gate"] is False
    assert boundary["current_profile_changed"] is False
    assert boundary["runtime_policy_activated"] is False
    assert boundary["full_replacement_enabled"] is False
    assert closeout["next"]["new_disjoint_audit_required"] is True
