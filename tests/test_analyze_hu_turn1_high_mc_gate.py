from ofc_regular.analyze_hu_turn1_high_mc_gate import analyze, audit_row


def _action(index, ev, se, row):
    return {
        "action_index": index,
        "placements": [[f"{index}c", row]],
        "discards": ["2d"],
        "ev": ev,
        "se": se,
        "rollout_count": 512,
    }


def _row(candidate_ev=3.0, baseline_ev=1.0):
    candidate = _action(7, candidate_ev, 0.3, "top")
    baseline = _action(2, baseline_ev, 0.4, "middle")
    return {
        "hand_seed": 101,
        "seat": "first",
        "actions": [baseline, candidate],
        "runtime_candidate_action_index": 7,
        "runtime_baseline_action_index": 2,
        "runtime_candidate_action": candidate,
        "runtime_baseline_action": baseline,
        "future_samples": 512,
        "realized_delta": -4.0,
        "confirm_delta": 2.0,
        "confirm_delta_se": 1.0,
        "stage_a_delta": 3.0,
        "stage_a_delta_se": 2.0,
        "runtime_predicted_margin": 0.2,
        "runtime_candidate_score": 1.5,
        "runtime_baseline_score": 1.0,
    }


def test_high_mc_audit_maps_actions_by_index_and_computes_lcb():
    audited = audit_row(_row(), row_index=0)

    assert audited["action_mapping_ok"] is True
    assert audited["high_mc_delta"] == 2.0
    assert audited["high_mc_delta_se_independent"] == 0.5
    assert audited["high_mc_lcb196"] == 1.02
    assert audited["confirm_z"] == 2.0


def test_high_mc_analysis_reports_mapping_and_threshold_metrics():
    audited, thresholds, summary = analyze([_row(), _row(candidate_ev=0.0, baseline_ev=1.0)])

    assert len(audited) == 2
    assert summary["mapped_rows"] == 2
    assert summary["mapping_failures"] == 0
    assert summary["future_sample_counts"] == [512]
    all_rows = [row for row in thresholds if row["threshold"] == "all"]
    assert len(all_rows) == 1
    assert all_rows[0]["fires"] == 2
    assert all_rows[0]["false_positive_count"] == 1
