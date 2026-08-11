from ofc_regular.analyze_hu_turn2_stage8c_unknown_replay import analyze, adoption_decision


def _summary(row_index: int, delta: float, *, seat: str = "first", label: str = "gray"):
    return {
        "row_index": str(row_index),
        "source_log": "run/runtime_decisions.jsonl",
        "hand_seed": str(1000 + row_index),
        "seat": seat,
        "status": "ok",
        "action_mapping_status": "ok",
        "future_samples": "128",
        "delta_for_label": str(delta),
        "delta_standard_error_for_label": "0.5",
        "safe_lcb196_label": label,
        "hard_negative_label": "1" if delta < 0 else "0",
        "old_confirm_delta": "1.0",
        "old_predicted_delta": "0.5",
        "old_gate_probability": "0.1",
    }


def _source(row_index: int, *, source: str = "below_confirm_delta", split: str = "test"):
    return {
        "candidate_source": source,
        "no_override_reason": source,
        "risk_prediction_split": split,
        "risk_probability": 0.8 + row_index * 0.01,
        "recommended_training_use": "topk_confirm_rejected",
        "replay_ready": True,
    }


def test_unknown_replay_analysis_joins_source_rows_and_breaks_down_groups():
    enriched, breakdown, losses = analyze(
        replay_summary_rows=[
            _summary(0, 2.0, seat="first", label="positive"),
            _summary(1, -1.0, seat="second", label="negative"),
            _summary(2, 3.0, seat="second", label="positive"),
        ],
        source_rows=[
            _source(0, source="below_confirm_delta"),
            _source(1, source="below_confirm_se"),
            _source(2, source="topk_empty"),
        ],
        top_loss_count=2,
    )

    overall = breakdown[0]
    assert len(enriched) == 3
    assert overall["rows"] == 3
    assert overall["ok_rows"] == 3
    assert overall["mean_delta"] == 4.0 / 3.0
    assert overall["negative_rows"] == 1
    assert losses[0]["delta_for_label"] == -1.0
    assert {row["group_value"] for row in breakdown if row["group_field"] == "candidate_source"} == {
        "below_confirm_delta",
        "below_confirm_se",
        "topk_empty",
    }


def test_adoption_decision_rejects_mapping_failures_ci_and_negative_rate():
    assert adoption_decision({"mapping_bad_rows": 1}) == "No-Go: action mapping failures"
    assert adoption_decision({"mapping_bad_rows": 0, "delta_ci95_low": -0.1}) == (
        "No-Go: replay CI low is not positive"
    )
    assert adoption_decision({"mapping_bad_rows": 0, "delta_ci95_low": 0.1, "negative_rate": 0.25}) == (
        "No-Go: negative replay rate is high"
    )
    assert adoption_decision({"mapping_bad_rows": 0, "delta_ci95_low": 0.1, "negative_rate": 0.0}) == (
        "Continue: replay signal positive, still not production"
    )
