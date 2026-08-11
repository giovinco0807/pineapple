from __future__ import annotations

from ofc_regular.verify_hu_turn1_c4_candidate import build_checks


def _row() -> dict[str, object]:
    return {
        "config_id": "cfg",
        "decision_count": 60000,
        "valid_decision_count": 59350,
        "valid_override_count": 650,
        "invalid_override_count": 0,
        "realized_per_fire_delta_mean": 1.0,
        "realized_per_fire_ci95_low": 0.1,
        "estimated_ev_per_decision_ci95_low": 0.001,
        "non_fired_counterfactual_nonzero_count": 0,
        "non_fired_final_mismatch_count": 0,
        "non_fired_final_match_unknown_count": 0,
        "p95_loss": 20.0,
        "p99_loss": 30.0,
        "max_loss": 40.0,
    }


def test_c4_acceptance_passes_only_complete_positive_safe_result() -> None:
    checks = build_checks(
        _row(),
        {"avg_score_per_hand_for_a": 0.01},
        expected_config="cfg",
        expected_decisions=60000,
        min_fires=300,
        max_p95_loss=25.0,
        max_p99_loss=40.0,
        max_loss=50.0,
    )

    assert all(check["passed"] for check in checks)


def test_c4_acceptance_rejects_nonpositive_ci_and_dirty_cancellation() -> None:
    row = _row()
    row["realized_per_fire_ci95_low"] = -0.1
    row["non_fired_final_mismatch_count"] = 1
    checks = build_checks(
        row,
        {"avg_score_per_hand_for_a": 0.01},
        expected_config="cfg",
        expected_decisions=60000,
        min_fires=300,
        max_p95_loss=25.0,
        max_p99_loss=40.0,
        max_loss=50.0,
    )

    failures = {check["check"] for check in checks if not check["passed"]}
    assert failures == {"per_fire_ci95_low_positive", "non_fired_final_mismatch_zero"}
