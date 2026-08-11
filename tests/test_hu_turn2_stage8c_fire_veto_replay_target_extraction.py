from ofc_regular.extract_hu_turn2_stage8c_fire_veto_replay_targets import (
    extract_fire_veto_targets,
    has_observed_delta,
)


def _action(card="As", row="top"):
    return {"placements": [[card, row], ["Kh", "middle"]], "discards": ["2c"]}


def _distillation_row(**overrides):
    row = {
        "schema": "hu_turn2_stage8c_topk_confirm_distillation_v1",
        "source_log": "outputs/run/runtime_decisions.jsonl",
        "config_id": "cfg",
        "hand_seed": "2026061501",
        "seat": "first",
        "seat_swap": "ab",
        "state_signature": "state-a",
        "action_signature": "action-a",
        "baseline_action_signature": "baseline-a",
        "candidate_index": 3,
        "baseline_index": 1,
        "recommended_training_use": "topk_confirm_rejected",
        "candidate_source": "below_confirm_delta",
        "no_override_reason": "below_confirm_delta",
        "hero_board": {"top": ["Qh"], "middle": ["2c"], "bottom": ["3d", "4h", "5s", "6c"]},
        "opponent_board": {"top": ["Ah"], "middle": ["7c"], "bottom": ["8d", "9h", "Ts"]},
        "dead_cards": ["Jc", "2d"],
        "cards_to_place": ["As", "Kh", "2c"],
        "baseline_action": _action("Qs", "bottom"),
        "candidate_action": _action(),
        "realized_delta_observed": False,
        "realized_delta": 0.0,
        "local_replay_status": "",
        "local_replay_label": "",
    }
    row.update(overrides)
    return row


def _prediction(row, *, probability="0.85", split="test", label="0", **overrides):
    prediction = {
        "source_log": row["source_log"].replace("\\", "/"),
        "config_id": row["config_id"],
        "hand_seed": row["hand_seed"],
        "seat": row["seat"],
        "state_signature": row["state_signature"],
        "action_signature": row["action_signature"],
        "baseline_action_signature": row["baseline_action_signature"],
        "candidate_index": str(row["candidate_index"]),
        "baseline_index": str(row["baseline_index"]),
        "split": split,
        "risk_probability": probability,
        "label": label,
        "recommended_training_use": row["recommended_training_use"],
        "realized_delta_observed": "0",
    }
    prediction.update(overrides)
    return prediction


def test_has_observed_delta_includes_local_replay_labels():
    assert has_observed_delta(_distillation_row(realized_delta_observed=True))
    assert has_observed_delta(_distillation_row(local_replay_status="ok"))
    assert has_observed_delta(_distillation_row(local_replay_label="negative"))
    assert not has_observed_delta(_distillation_row())


def test_extract_fire_veto_targets_keeps_unknown_rows_after_veto():
    row = _distillation_row()

    targets, threshold_rows, breakdown, summary = extract_fire_veto_targets(
        [row],
        [_prediction(row, probability="0.90")],
        [_prediction(row, probability="0.30")],
        fire_thresholds=[0.8],
        veto_thresholds=[0.4],
        fire_boundary_width=0.05,
        veto_boundary_width=0.05,
    )

    assert len(targets) == 1
    assert targets[0]["schema"] == "hu_turn2_stage8c_fire_veto_replay_target_v1"
    assert targets[0]["replay_ready"] is True
    assert targets[0]["veto_prediction_missing"] is False
    assert "kept_after_fire_veto" in targets[0]["replay_target_reasons"]
    assert summary["targets"] == 1
    assert summary["targets_with_veto_prediction"] == 1
    assert any(row["reason"] == "kept_after_fire_veto" for row in threshold_rows)
    assert breakdown[0]["group_field"] == "overall"


def test_extract_fire_veto_targets_skips_observed_rows():
    observed = _distillation_row(realized_delta_observed=True)

    targets, _threshold_rows, _breakdown, summary = extract_fire_veto_targets(
        [observed],
        [_prediction(observed, probability="0.95", realized_delta_observed="1")],
        [_prediction(observed, probability="0.20")],
        fire_thresholds=[0.8],
        veto_thresholds=[0.4],
        fire_boundary_width=0.05,
        veto_boundary_width=0.05,
    )

    assert targets == []
    assert summary["observed_delta_skipped"] == 1


def test_extract_fire_veto_targets_marks_missing_veto_predictions():
    row = _distillation_row()

    targets, threshold_rows, _breakdown, summary = extract_fire_veto_targets(
        [row],
        [_prediction(row, probability="0.91")],
        [],
        fire_thresholds=[0.9],
        veto_thresholds=[0.4],
        fire_boundary_width=0.05,
        veto_boundary_width=0.05,
    )

    assert len(targets) == 1
    assert targets[0]["veto_prediction_missing"] is True
    assert targets[0]["veto_probability"] == ""
    assert "fire_selected_missing_veto_prediction" in targets[0]["replay_target_reasons"]
    assert summary["targets_missing_veto_prediction"] == 1
    assert any(row["reason"] == "fire_selected_missing_veto_prediction" for row in threshold_rows)


def test_extract_fire_veto_targets_reports_replay_blockers():
    row = _distillation_row(dead_cards=[])

    targets, _threshold_rows, _breakdown, summary = extract_fire_veto_targets(
        [row],
        [_prediction(row, probability="0.90")],
        [_prediction(row, probability="0.30")],
        fire_thresholds=[0.8],
        veto_thresholds=[0.4],
        fire_boundary_width=0.05,
        veto_boundary_width=0.05,
    )

    assert targets[0]["replay_ready"] is False
    assert targets[0]["replay_blocker"] == "dead_cards"
    assert summary["replay_ready_targets"] == 0


def test_extract_fire_veto_targets_can_filter_bad_candidate_sources_and_high_veto_probability():
    topk_empty = _distillation_row(
        hand_seed="2026061502",
        state_signature="state-empty",
        action_signature="action-empty",
        candidate_source="topk_empty",
    )
    high_veto = _distillation_row(
        hand_seed="2026061503",
        state_signature="state-veto",
        action_signature="action-veto",
        candidate_source="below_confirm_delta",
    )
    kept = _distillation_row(
        hand_seed="2026061504",
        state_signature="state-kept",
        action_signature="action-kept",
        candidate_source="below_confirm_se",
    )

    targets, _threshold_rows, _breakdown, summary = extract_fire_veto_targets(
        [topk_empty, high_veto, kept],
        [
            _prediction(topk_empty, probability="0.91"),
            _prediction(high_veto, probability="0.91"),
            _prediction(kept, probability="0.91"),
        ],
        [
            _prediction(topk_empty, probability="0.20"),
            _prediction(high_veto, probability="0.75"),
            _prediction(kept, probability="0.20"),
        ],
        fire_thresholds=[0.9],
        veto_thresholds=[0.4],
        fire_boundary_width=0.05,
        veto_boundary_width=0.05,
        excluded_candidate_sources={"topk_empty"},
        max_veto_probability=0.6,
    )

    assert [target["candidate_source"] for target in targets] == ["below_confirm_se"]
    assert summary["filtered_candidate_source"] == 1
    assert summary["filtered_veto_probability"] == 1
