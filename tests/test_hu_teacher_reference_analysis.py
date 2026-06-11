from ofc_regular.analyze_hu_teacher_references import analyze_samples


def test_analyze_hu_teacher_references_uses_reference_action_deltas():
    sample = {
        "score_gap": 0.5,
        "selection": {"predicted_margin_vs_baseline": 8.0},
        "reference_actions": {
            "baseline": {"score": 1.0},
            "selection_hu": {"score": 2.0},
            "delta_best_vs_baseline": 2.0,
            "delta_best_vs_selection_hu": 1.0,
            "delta_selection_hu_vs_baseline": 1.0,
            "selection_hu_regret": 1.0,
            "baseline_regret": 2.0,
        },
    }

    summary = analyze_samples([sample], positive_deltas=[0.25, 1.5])

    assert summary["samples_read"] == 1
    assert summary["samples_with_references"] == 1
    assert summary["delta_best_vs_selection_hu"]["mean"] == 1.0
    assert summary["positive_delta_counts"][0]["samples"] == 1.0
    assert summary["positive_delta_counts"][1]["samples"] == 0.0
    assert summary["stage3_selection_teacher_net"]["positive"] == 1.0
