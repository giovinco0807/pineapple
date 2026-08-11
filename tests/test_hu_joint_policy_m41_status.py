from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _status() -> dict[str, object]:
    return json.loads(
        (ROOT / "configs" / "hu_joint_policy_m41_status.json").read_text(
            encoding="utf-8"
        )
    )


def test_m41_status_is_fail_closed_and_current_registry_is_unchanged() -> None:
    status = _status()
    assert status["status"] == "complete_no_go_policy_not_promoted"
    assert status["current_profile_changed"] is False
    assert status["runtime_policy_activated"] is False
    assert status["policy_promoted"] is False
    assert status["spot_vm_started"] is False
    assert status["safety_calibration"]["safety_enabled"] is False
    assert status["acceptance"]["status"] == "complete_no_go"
    assert status["acceptance"]["teacher_metrics_used_for_promotion"] is False
    assert status["acceptance"]["top1_accuracy_used_for_promotion"] is False
    expected = status["baseline_hash_audit"]["policy_registry_expected_sha256"]
    actual = hashlib.sha256(
        (ROOT / "src" / "ofc_regular" / "ai_profiles.py").read_bytes()
    ).hexdigest()
    assert actual == expected


def test_m41_no_go_blocks_scale_and_preserves_fixed_chain() -> None:
    status = _status()
    assert status["fixed_legacy_baseline_chain"][:4] == [
        "stage19_p0",
        "stage18_p1",
        "stage9f_p2",
        "stage7_m5_r10",
    ]
    assert status["fixed_continuation"]["rollback_baseline"] == "stage18_p1"
    assert status["fixed_continuation"]["t2_profile"] == "stage9f_p2"
    assert "do_not_start_spot_scale_from_this_candidate" in status["no_go_actions"]
    assert "do_not_start_m5" in status["no_go_actions"]
    assert status["next_milestone"].startswith("M4_2")


def test_m41_pilot_and_acceptance_evidence_are_frozen() -> None:
    status = _status()
    assert status["pilot"]["roots"] == 100
    assert status["pilot"]["data_audit_status"] == "pass"
    assert sum(row["roots"] for row in status["root_population"]["profiles"]) == 100
    assert status["safety_calibration"]["partition_status"] == "pass"
    assert status["safety_calibration"]["fit_lock_seed_overlap"] == 0
    assert status["fresh_population_smoke"]["invalid_counterfactuals"] == 0
    assert status["fresh_population_smoke"]["nonfire_cancellation_mismatches"] == 0
    assert status["acceptance"]["gates_passed"] == 18
    assert status["acceptance"]["gates_total"] == 24
