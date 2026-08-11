from ofc_regular.extract_hu_turn2_stage8c_topk_replay_targets import (
    extract_targets,
    filter_distillation_rows,
    load_distillation_index,
    row_key,
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
    }
    row.update(overrides)
    return row


def _prediction(row, **overrides):
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
        "split": "test",
        "risk_probability": "0.85",
        "label": "0",
        "risk_target_group": row["recommended_training_use"],
        "realized_delta_observed": "0",
    }
    prediction.update(overrides)
    return prediction


def test_row_key_normalizes_paths_for_prediction_join():
    row = _distillation_row(source_log="outputs\\run\\runtime_decisions.jsonl")
    prediction = _prediction(row, source_log="outputs/run/runtime_decisions.jsonl")

    index = load_distillation_index([row])

    assert row_key(prediction) in index


def test_extract_targets_outputs_only_selected_unknown_rows():
    unknown = _distillation_row(hand_seed="1", state_signature="state-unknown", action_signature="action-unknown")
    observed = _distillation_row(
        hand_seed="2",
        state_signature="state-observed",
        action_signature="action-observed",
        realized_delta_observed=True,
        realized_delta=4.0,
    )
    low_probability = _distillation_row(
        hand_seed="3",
        state_signature="state-low",
        action_signature="action-low",
    )

    targets, threshold_rows, breakdown = extract_targets(
        [unknown, observed, low_probability],
        [
            _prediction(unknown, risk_probability="0.90", realized_delta_observed="0"),
            _prediction(observed, risk_probability="0.95", realized_delta_observed="1"),
            _prediction(low_probability, risk_probability="0.70", realized_delta_observed="0"),
        ],
        thresholds=[0.8],
        splits={"test"},
    )

    assert [row["hand_seed"] for row in targets] == ["1"]
    assert targets[0]["schema"] == "hu_turn2_stage8c_topk_confirm_unknown_replay_target_v1"
    assert targets[0]["replay_ready"] is True
    assert targets[0]["realized_delta_observed"] is False
    assert targets[0]["risk_selected_thresholds"] == [0.8]
    assert threshold_rows[0]["selected_predictions"] == 2
    assert threshold_rows[0]["selected_observed_delta"] == 1
    assert threshold_rows[0]["selected_unknown_delta"] == 1
    assert threshold_rows[0]["matched_replay_ready"] == 1
    assert breakdown[0]["group_field"] == "overall"
    assert breakdown[0]["rows"] == 1


def test_extract_targets_reports_replay_blockers_without_dropping_rows():
    missing_dead = _distillation_row(dead_cards=[])

    targets, threshold_rows, _breakdown = extract_targets(
        [missing_dead],
        [_prediction(missing_dead, risk_probability="0.90", realized_delta_observed="0")],
        thresholds=[0.8],
        splits=None,
    )

    assert len(targets) == 1
    assert targets[0]["replay_ready"] is False
    assert targets[0]["replay_blocker"] == "dead_cards"
    assert threshold_rows[0]["matched_unknown_delta"] == 1
    assert threshold_rows[0]["matched_replay_ready"] == 0


def test_extract_targets_counts_missing_distillation_matches():
    row = _distillation_row()
    missing_prediction = _prediction(row, state_signature="missing-state", risk_probability="0.90")

    targets, threshold_rows, _breakdown = extract_targets(
        [row],
        [missing_prediction],
        thresholds=[0.8],
        splits=None,
    )

    assert targets == []
    assert threshold_rows[0]["selected_unknown_delta"] == 1
    assert threshold_rows[0]["matched_unknown_delta"] == 0
    assert threshold_rows[0]["missing_distillation_row"] == 1


def test_filter_distillation_rows_excludes_recommended_use_groups():
    keep = _distillation_row(hand_seed="keep", recommended_training_use="topk_confirm_rejected")
    drop = _distillation_row(hand_seed="drop", recommended_training_use="topk_confirm_topk_empty")

    kept, removed = filter_distillation_rows(
        [keep, drop],
        excluded_recommended_uses=["topk_confirm_topk_empty"],
    )

    assert [row["hand_seed"] for row in kept] == ["keep"]
    assert removed == {"topk_confirm_topk_empty": 1}


def test_extract_targets_skips_excluded_prediction_uses_without_missing_source():
    keep = _distillation_row(hand_seed="keep", recommended_training_use="topk_confirm_rejected")
    drop = _distillation_row(hand_seed="drop", recommended_training_use="topk_confirm_topk_empty")

    targets, threshold_rows, _breakdown = extract_targets(
        [keep],
        [
            _prediction(keep, risk_probability="0.90", realized_delta_observed="0"),
            _prediction(
                drop,
                risk_probability="0.95",
                realized_delta_observed="0",
                recommended_training_use="topk_confirm_topk_empty",
            ),
        ],
        thresholds=[0.8],
        splits=None,
        excluded_recommended_uses={"topk_confirm_topk_empty"},
    )

    assert [row["hand_seed"] for row in targets] == ["keep"]
    assert threshold_rows[0]["selected_predictions"] == 1
    assert threshold_rows[0]["missing_distillation_row"] == 0
    assert threshold_rows[0]["excluded_predictions"] == 1
