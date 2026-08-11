from ofc_regular.analyze_hu_turn2_stage8c_fire_veto_replay import analyze, decision


def _summary(row_index: int, delta: float, *, seat: str = "first", label: str = "gray"):
    return {
        "row_index": str(row_index),
        "source_log": "run/runtime_decisions.jsonl",
        "hand_seed": str(1000 + row_index),
        "seat": seat,
        "status": "ok",
        "action_mapping_status": "ok",
        "future_samples": "512",
        "delta_for_label": str(delta),
        "delta_standard_error_for_label": "0.5",
        "safe_lcb196_label": label,
        "hard_negative_label": "1" if delta < 0 else "0",
        "old_predicted_delta": "0.25",
        "old_gate_probability": "0.75",
    }


def _source(
    *,
    source: str = "below_confirm_delta",
    fire: float = 0.86,
    veto: float = 0.35,
    reasons: str = "fire_boundary,kept_after_fire_veto",
):
    return {
        "candidate_source": source,
        "no_override_reason": source,
        "recommended_training_use": "topk_confirm_rejected",
        "replay_target_reasons": reasons,
        "fire_probability": fire,
        "veto_probability": veto,
        "fire_prediction_label": 0,
        "veto_prediction_label": -1,
        "veto_prediction_missing": False,
        "replay_ready": True,
    }


def test_fire_veto_replay_analysis_expands_reason_members_and_probability_bins():
    _enriched, breakdown, reasons, losses = analyze(
        replay_summary_rows=[
            _summary(0, 2.0, seat="first", label="positive"),
            _summary(1, -1.0, seat="second", label="negative"),
            _summary(2, 0.5, seat="second", label="gray"),
        ],
        source_rows=[
            _source(source="below_confirm_delta", fire=0.86, veto=0.35, reasons="fire_boundary,kept_after_fire_veto"),
            _source(source="topk_empty", fire=0.79, veto=0.61, reasons="fire_boundary,veto_boundary"),
            _source(source="below_confirm_se", fire=0.93, veto=0.25, reasons="just_kept_by_veto"),
        ],
        top_loss_count=2,
    )

    overall = breakdown[0]
    assert overall["rows"] == 3
    assert overall["negative_rows"] == 1
    assert overall["mean_delta"] == 1.5 / 3.0
    assert {row["group_value"] for row in reasons} == {
        "fire_boundary",
        "just_kept_by_veto",
        "kept_after_fire_veto",
        "veto_boundary",
    }
    assert any(
        row["group_field"] == "fire_probability_bin" and row["group_value"] == "[0.85,0.90)"
        for row in breakdown
    )
    assert losses[0]["candidate_source"] == "topk_empty"
    assert losses[0]["delta_for_label"] == -1.0


def test_fire_veto_replay_decision_requires_positive_ci_and_low_negative_rate():
    assert decision({"mapping_bad_rows": 1}) == "No-Go: action mapping failures"
    assert decision({"mapping_bad_rows": 0, "delta_ci95_low": -0.1}) == (
        "No-Go: replay CI low is not positive"
    )
    assert decision({"mapping_bad_rows": 0, "delta_ci95_low": 0.1, "negative_rate": 0.2}) == (
        "No-Go: negative replay rate is high"
    )
    assert decision({"mapping_bad_rows": 0, "delta_ci95_low": 0.1, "negative_rate": 0.0}) == (
        "Continue: positive replay signal, still not production"
    )
