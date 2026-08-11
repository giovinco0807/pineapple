from ofc_regular.analyze_hu_turn2_stage8c_fire_veto_combo import join_predictions, metric_row


def _row(key: str, *, probability: float, label: int, split: str = "test", delta: float = 0.0):
    return {
        "state_signature": f"state-{key}",
        "action_signature": f"action-{key}",
        "baseline_action_signature": f"baseline-{key}",
        "split": split,
        "risk_probability": str(probability),
        "label": str(label),
        "local_replay_delta": str(delta),
    }


def test_join_predictions_matches_state_action_and_baseline_signatures():
    fire_rows = [_row("a", probability=0.8, label=1), _row("b", probability=0.7, label=0)]
    veto_rows = [_row("a", probability=0.2, label=0), _row("c", probability=0.9, label=1)]

    joined, summary = join_predictions(fire_rows, veto_rows)

    assert summary["fire_rows"] == 2
    assert summary["veto_rows"] == 2
    assert summary["joined_rows"] == 1
    assert joined[0]["fire_probability"] == 0.8
    assert joined[0]["veto_probability"] == 0.2


def test_metric_row_keeps_fire_rows_below_veto_threshold():
    rows = [
        {
            "fire_probability": 0.9,
            "veto_probability": 0.1,
            "fire_label": 1,
            "local_replay_delta": 4.0,
        },
        {
            "fire_probability": 0.9,
            "veto_probability": 0.8,
            "fire_label": 0,
            "local_replay_delta": -3.0,
        },
        {
            "fire_probability": 0.4,
            "veto_probability": 0.1,
            "fire_label": 1,
            "local_replay_delta": 2.0,
        },
    ]

    metrics = metric_row(rows, fire_threshold=0.8, veto_threshold=0.5)

    assert metrics["fire_selected"] == 2
    assert metrics["kept_after_veto"] == 1
    assert metrics["kept_positive"] == 1
    assert metrics["kept_negative"] == 0
    assert metrics["kept_precision"] == 1.0
    assert metrics["kept_delta_mean"] == 4.0
    assert metrics["vetoed_negative"] == 1
    assert metrics["vetoed_positive"] == 0
