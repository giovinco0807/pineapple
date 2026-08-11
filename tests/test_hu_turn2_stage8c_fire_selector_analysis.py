from ofc_regular.analyze_hu_turn2_stage8c_fire_selector import (
    best_rows,
    threshold_metric_rows,
    topk_metric_rows,
)


def _row(prob, delta, label=1, split="test", seat="first", rank=1):
    return {
        "risk_probability": str(prob),
        "realized_delta": str(delta),
        "realized_delta_observed": "1",
        "label": str(label),
        "split": split,
        "seat": seat,
        "candidate_ev_rank": str(rank),
        "recommended_training_use": "topk_confirm_replay_positive" if label else "topk_confirm_replay_negative",
    }


def test_fire_selector_topk_treats_high_probability_as_fire_candidate():
    rows = [
        _row(0.9, 3.0, label=1),
        _row(0.8, -2.0, label=0),
        _row(0.1, 5.0, label=1),
    ]

    metrics = topk_metric_rows(rows, topk_values=[1, 2], group_fields=["seat"])
    test_all = [
        row
        for row in metrics
        if row["split"] == "test" and row["group_field"] == "all" and row["group_value"] == "all"
    ]

    assert test_all[0]["topk"] == 1
    assert test_all[0]["selected_realized_delta_sum"] == 3.0
    assert test_all[0]["precision"] == 1.0
    assert test_all[1]["topk"] == 2
    assert test_all[1]["selected_realized_delta_sum"] == 1.0
    assert test_all[1]["selected_negative_rows"] == 1


def test_fire_selector_threshold_and_best_rows_rank_by_delta_per_row():
    rows = [
        _row(0.95, 2.0, label=1, seat="first"),
        _row(0.90, -1.0, label=0, seat="first"),
        _row(0.95, 4.0, label=1, seat="second"),
        _row(0.10, -5.0, label=0, seat="second"),
    ]

    metrics = threshold_metric_rows(rows, thresholds=[0.9], group_fields=["seat"])
    best = best_rows(metrics, min_selected=1)

    assert best[0]["group_field"] == "seat"
    assert best[0]["group_value"] == "second"
    assert best[0]["selected_realized_delta_sum"] == 4.0
    assert best[0]["estimated_realized_delta_per_row"] == 2.0
