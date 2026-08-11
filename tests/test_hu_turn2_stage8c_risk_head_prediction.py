import numpy as np

from ofc_regular.predict_hu_turn2_stage8c_risk_head import materialize_prediction_rows, metadata_for_row


def _action(card="As", row="top"):
    return {"placements": [[card, row], ["Kh", "middle"]], "discards": ["2c"]}


def _row(**overrides):
    row = {
        "source_log": "outputs/run/runtime_decisions.jsonl",
        "config_id": "cfg",
        "hand_seed": "2026072001000001",
        "seat": "first",
        "seat_swap": "ab",
        "state_signature": "state-a",
        "action_signature": "action-a",
        "baseline_action_signature": "baseline-a",
        "candidate_index": 3,
        "baseline_index": 1,
        "recommended_training_use": "topk_confirm_replay_negative",
        "hero_board": {"top": ["Qh"], "middle": ["2c"], "bottom": ["3d", "4h", "5s", "6c"]},
        "opponent_board": {"top": ["Ah"], "middle": ["7c"], "bottom": ["8d", "9h", "Ts"]},
        "dead_cards": ["Jc", "2d"],
        "cards_to_place": ["As", "Kh", "2c"],
        "baseline_action": _action("Qs", "bottom"),
        "candidate_action": _action(),
        "realized_delta_observed": True,
        "local_replay_status": "ok",
        "local_replay_action_mapping_status": "ok",
        "local_replay_label": "negative",
        "local_replay_bucket": "local_negative",
        "local_replay_delta": -1.0,
        "local_replay_delta_se": 0.25,
        "confirm_delta": 0.5,
        "confirm_delta_se": 0.25,
        "predicted_delta": 1.5,
        "gate_probability": 0.8,
        "candidate_ev_rank": 1,
    }
    row.update(overrides)
    return row


def test_metadata_for_prediction_rows_preserves_join_keys_and_target_label():
    row = _row()

    metadata = metadata_for_row(row, row_index=7, target_mode="local_ev_negative")

    assert metadata["row_index"] == 7
    assert metadata["state_signature"] == "state-a"
    assert metadata["action_signature"] == "action-a"
    assert metadata["baseline_action_signature"] == "baseline-a"
    assert metadata["label"] == 1
    assert metadata["risk_target_group"] == "local_ev_negative"
    assert metadata["split_group_source_seed"] == "2026072001"


def test_materialize_prediction_rows_scores_rows_even_when_target_is_not_trainable():
    trainable = _row()
    gray = _row(
        hand_seed="2026072001000002",
        state_signature="state-b",
        action_signature="action-b",
        recommended_training_use="topk_confirm_replay_gray",
        local_replay_label="gray",
        local_replay_bucket="local_positive_gray",
    )

    features, metadata, skipped, counters = materialize_prediction_rows(
        [trainable, gray],
        feature_mode="preconfirm_meta_only",
        target_mode="local_ev_negative",
    )

    assert features.shape == (2, 12)
    assert features.dtype == np.float32
    assert [row["label"] for row in metadata] == [1, ""]
    assert skipped == []
    assert counters == {}


def test_materialize_prediction_rows_reports_feature_errors_when_requested():
    bad = _row(hero_board=None)

    features, metadata, skipped, counters = materialize_prediction_rows(
        [bad],
        feature_mode="hu_only",
        target_mode="local_ev_negative",
        include_feature_errors=True,
    )

    assert features.shape == (0, 0)
    assert metadata == []
    assert len(skipped) == 1
    assert skipped[0]["skip_reason"].startswith("feature_error:")
    assert sum(counters.values()) == 1
