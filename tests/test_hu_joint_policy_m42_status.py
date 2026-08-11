from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _status() -> dict[str, object]:
    return _read_json(ROOT / "configs" / "hu_joint_policy_m42_status.json")


def _artifact(status: dict[str, object], name: str) -> Path:
    artifacts = status["artifacts"]
    assert isinstance(artifacts, dict)
    relative = artifacts[name]
    assert isinstance(relative, str)
    return ROOT / relative


def test_m42_is_complete_no_go_and_current_registry_is_frozen() -> None:
    status = _status()
    assert status["status"] == "complete_no_go_policy_not_promoted"
    assert status["current_profile_changed"] is False
    assert status["runtime_policy_activated"] is False
    assert status["policy_promoted"] is False
    assert status["scale_authorized"] is False
    assert status["m5_authorized"] is False
    assert status["safety_calibration"]["safety_enabled"] is False
    assert status["acceptance"]["status"] == "complete_no_go"
    assert status["acceptance"]["promotion_decision"] == "no_go"
    assert "do_not_start_m5" in status["no_go_actions"]
    assert "do_not_scale_the_failed_ranker" in status["no_go_actions"]

    expected = status["baseline_hash_audit"]["policy_registry_expected_sha256"]
    actual = hashlib.sha256(
        (ROOT / "src" / "ofc_regular" / "ai_profiles.py").read_bytes()
    ).hexdigest()
    assert actual == expected
    assert status["baseline_hash_audit"]["current_mapping_changed"] is False


def test_m42_gate40_model_population_and_acceptance_evidence_match() -> None:
    status = _status()
    receipt = _read_json(_artifact(status, "teacher_receipt"))
    audit = _read_json(_artifact(status, "data_audit"))
    manifest = _read_json(_artifact(status, "training_manifest"))
    population = _read_json(_artifact(status, "population_smoke"))
    acceptance = _read_json(_artifact(status, "acceptance_status"))

    assert receipt["status"] == "verified_and_audited"
    assert receipt["verified_shards"] == 8
    assert receipt["verified_roots"] == 40
    assert receipt["audit_sha256"] == status["teacher_gate40"]["audit_sha256"]
    assert sum(
        row["roots"] for row in status["teacher_gate40"]["root_population"]
    ) == 40

    assert audit["status"] == "pass"
    assert audit["paired_delta_mode"] == "required_v2"
    assert audit["gates"]["current_profile_resolved"] is False
    assert audit["gates"]["holdout_threshold_search_allowed"] is False
    assert audit["gates"]["candidate_evaluation_rng_disjoint"] is True

    assert manifest["cross_fit"]["status"] == "pass"
    assert manifest["cross_fit"]["schema"] == (
        "hu_m4_identity_group_nested_cross_fit_v2"
    )
    assert manifest["locked_holdout_used_for_threshold_or_training"] is False
    assert manifest["calibration"]["status"] == "no_go"
    assert manifest["calibration"]["locked_holdout_used"] is False
    assert manifest["locked_holdout"]["safety_enabled"] is False
    model_sha256 = hashlib.sha256(_artifact(status, "model").read_bytes()).hexdigest()
    assert model_sha256 == status["training"]["model_sha256"]

    assert population["seed"] == 2126071901
    assert population["seed_stride"] == 1000003
    assert population["population"]["all_seats"]["overrides"] == 0
    assert population["invalid_counterfactuals"] == 0
    assert population["nonfire_cancellation_mismatches"] == 0
    assert population["nonfire_nonzero_deltas"] == 0
    assert population["nonfire_cancellation_unknown"] == 0

    assert acceptance["status"] == "complete_no_go"
    assert acceptance["gates_passed"] == 19
    assert acceptance["gates_total"] == 25
    fresh_seed_gate = next(
        gate
        for gate in acceptance["gates"]
        if gate["name"] == "fresh_evaluation_seed_disjoint"
    )
    assert fresh_seed_gate["passed"] is True
    assert fresh_seed_gate["observed"]["overlap_count"] == 0
    assert status["acceptance"]["added_m42_gate"] == fresh_seed_gate["name"]
    assert status["acceptance"]["teacher_metrics_used_for_promotion"] is False
    assert status["acceptance"]["top1_accuracy_used_for_promotion"] is False


def test_m42_cost_is_unverified_and_m43_requires_fresh_locked_evidence() -> None:
    status = _status()
    cost = status["cloud_compute_and_cost"]
    assert cost["billing_cost_usd"] is None
    assert cost["billing_cost_verified"] is False
    assert cost["exact_cost_status"] == "unverified"
    assert cost["successful_teacher_run"]["spot_worker_vms"] == 8
    assert cost["model_run"]["spot_worker_vms"] == 1
    assert cost["total_observed_spot_vm_or_worker_status_records"] == 25
    assert cost["active_vms_after_closure"] == 0

    redesign = status["m43_redesign"]
    assert redesign["baseline_action_score_exact_zero"] is True
    assert redesign["threshold_selection"] == "threshold_lock_only"
    assert redesign["reuse_m42_locked_holdout_for_m43_selection"] is False
    assert redesign["acceptance_gates_must_not_be_lowered"] is True
    assert status["next_milestone"].startswith("M4_3")

    audit_doc = (
        ROOT / "docs" / "hu_joint_policy_m42_completion_audit.md"
    ).read_text(encoding="utf-8")
    assert "19 of 25 gates passed" in audit_doc
    assert "`null` / unverified" in audit_doc
    assert "baseline_paired_delta_risk_ensemble" in audit_doc
