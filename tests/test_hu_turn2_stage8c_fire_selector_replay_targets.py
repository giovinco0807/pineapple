from ofc_regular.extract_hu_turn2_stage8c_fire_selector_replay_targets import (
    extract_targets,
    selected_prediction_rows,
)
from ofc_regular.extract_hu_turn2_stage8c_topk_replay_targets import row_key


def _action(card="As", row="top"):
    return {"placements": [[card, row], ["Kh", "middle"]], "discards": ["2c"]}


def _source(**overrides):
    row = {
        "source_log": "outputs/run/runtime_decisions.jsonl",
        "config_id": "cfg",
        "hand_seed": "2026061701",
        "seat": "second",
        "seat_swap": "ab",
        "state_signature": "state-a",
        "action_signature": "action-a",
        "baseline_action_signature": "baseline-a",
        "candidate_index": 2,
        "baseline_index": 1,
        "recommended_training_use": "topk_confirm_rejected",
        "hero_board": {"top": ["Qh"], "middle": ["2c"], "bottom": ["3d", "4h", "5s", "6c"]},
        "opponent_board": {"top": ["Ah"], "middle": ["7c"], "bottom": ["8d", "9h", "Ts"]},
        "dead_cards": ["Jc", "2d"],
        "cards_to_place": ["As", "Kh", "2c"],
        "baseline_action": _action("Qs", "bottom"),
        "candidate_action": _action(),
        "realized_delta_observed": False,
    }
    row.update(overrides)
    return row


def _prediction(source, **overrides):
    row = {
        "source_log": source["source_log"],
        "config_id": source["config_id"],
        "hand_seed": str(source["hand_seed"]),
        "seat": source["seat"],
        "seat_swap": source["seat_swap"],
        "state_signature": source["state_signature"],
        "action_signature": source["action_signature"],
        "baseline_action_signature": source["baseline_action_signature"],
        "candidate_index": str(source["candidate_index"]),
        "baseline_index": str(source["baseline_index"]),
        "recommended_training_use": source["recommended_training_use"],
        "split": "test",
        "risk_probability": "0.95",
        "label": "0",
        "risk_target_group": "topk_confirm_rejected",
        "realized_delta_observed": "0",
        "predicted_delta": "1.0",
        "gate_probability": "0.7",
        "candidate_ev_rank": "2",
    }
    row.update(overrides)
    return row


def test_selected_prediction_rows_filters_split_seat_and_threshold():
    second = _source(seat="second", hand_seed="1", state_signature="s1", action_signature="a1")
    first = _source(seat="first", hand_seed="2", state_signature="s2", action_signature="a2")
    rows = [
        _prediction(second, risk_probability="0.95"),
        _prediction(first, risk_probability="0.99"),
        _prediction(second, split="train", risk_probability="0.99"),
        _prediction(second, risk_probability="0.20"),
    ]

    selected, manifest = selected_prediction_rows(rows, splits={"test"}, seats={"second"}, threshold=0.9)

    assert len(selected) == 1
    assert selected[0]["hand_seed"] == "1"
    assert manifest["selection_counts"]["skipped_seat_filter"] == 1
    assert manifest["selection_counts"]["skipped_split_filter"] == 1
    assert manifest["selection_counts"]["skipped_threshold"] == 1


def test_extract_targets_joins_predictions_and_skips_observed_rows():
    unknown = _source(hand_seed="1", state_signature="s1", action_signature="a1")
    observed = _source(
        hand_seed="2",
        state_signature="s2",
        action_signature="a2",
        realized_delta_observed=True,
        realized_delta=5.0,
    )
    index = {row_key(row): row for row in [unknown, observed]}

    targets, audit, manifest = extract_targets(
        index,
        [
            _prediction(unknown, risk_probability="0.95"),
            _prediction(observed, risk_probability="0.99", realized_delta_observed="1"),
        ],
        include_observed=False,
        max_targets=10,
    )

    assert len(targets) == 1
    assert targets[0]["schema"] == "hu_turn2_stage8c_fire_selector_replay_target_v1"
    assert targets[0]["ranker"] == "fire_selector_probability"
    assert targets[0]["fire_probability"] == 0.95
    assert targets[0]["fire_selector_probability"] == 0.95
    assert targets[0]["replay_ready"] is True
    assert targets[0]["observed_performance_claim"] == "No"
    assert audit[0]["status"] == "target"
    assert manifest["counts"]["targets"] == 1
    assert manifest["counts"]["skipped_observed_delta"] == 1
