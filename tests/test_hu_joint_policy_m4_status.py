from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_m4_status_is_fail_closed_and_current_registry_is_unchanged() -> None:
    status = json.loads(
        (ROOT / "configs" / "hu_joint_policy_m4_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert status["status"] == "complete_no_go_policy_not_promoted"
    assert status["current_profile_changed"] is False
    assert status["runtime_policy_activated"] is False
    assert status["policy_promoted"] is False
    assert status["spot_vm_started"] is False
    assert status["training"]["safety_enabled"] is False
    assert status["acceptance"]["status"] == "complete_no_go"
    assert status["acceptance"]["teacher_metrics_used_for_promotion"] is False
    assert status["acceptance"]["top1_accuracy_used_for_promotion"] is False
    expected = status["baseline_hash_audit"]["policy_registry_expected_m1_sha256"]
    actual = hashlib.sha256(
        (ROOT / "src" / "ofc_regular" / "ai_profiles.py").read_bytes()
    ).hexdigest()
    assert actual == expected


def test_m4_no_go_blocks_m5_and_preserves_fixed_baselines() -> None:
    status = json.loads(
        (ROOT / "configs" / "hu_joint_policy_m4_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert status["fixed_legacy_baseline_chain"][:4] == [
        "stage19_p0",
        "stage18_p1",
        "stage9f_p2",
        "stage7_m5_r10",
    ]
    assert "do_not_start_m5" in status["no_go_actions"]
    assert status["next_milestone"].startswith("M4_1")
