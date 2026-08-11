import pytest
import numpy as np

from ofc_regular.action_space import generate_turn_actions
from ofc_regular.hu_turn3_model import HU_FEATURE_DIM
from ofc_regular.state import Board
from ofc_regular.train_hu_turn2_stage8c_risk_head import (
    CONFIRM_TAIL_FEATURE_NAMES,
    DEFAULT_THRESHOLDS,
    FEATURE_MODE_HU_DELTA_ONLY,
    FEATURE_MODE_HU_DELTA_PLUS_RUNTIME_TAIL_META,
    FEATURE_MODE_HU_PLUS_RUNTIME_META,
    FEATURE_MODE_HU_PLUS_RUNTIME_TAIL_META,
    FEATURE_MODE_LOOKAHEAD_PROXY_ONLY,
    FEATURE_MODE_LOOKAHEAD_PROXY_PLUS_PRECONFIRM_META,
    FEATURE_MODE_LOOKAHEAD_PROXY_PLUS_RUNTIME_TAIL_META,
    FEATURE_MODE_OPPORTUNITY_LOOKAHEAD_PROXY_PLUS_PRECONFIRM_META,
    FEATURE_MODE_OPPORTUNITY_PROXY_ONLY,
    FEATURE_MODE_OPPORTUNITY_LOOKAHEAD_PROXY_PLUS_RUNTIME_TAIL_META,
    FEATURE_MODE_OPPORTUNITY_PROXY_PLUS_PRECONFIRM_META,
    FEATURE_MODE_OPPORTUNITY_PROXY_PLUS_RUNTIME_TAIL_META,
    FEATURE_MODE_PRECONFIRM_META_ONLY,
    FEATURE_MODE_RUNTIME_META_ONLY,
    FEATURE_MODE_RUNTIME_TAIL_META_ONLY,
    MAX_ROWS_MODE_FIRST,
    MAX_ROWS_MODE_STRATIFIED,
    OPPORTUNITY_PROXY_FEATURE_NAMES,
    LOOKAHEAD_PROXY_FEATURE_NAMES,
    PRECONFIRM_META_FEATURE_NAMES,
    POS_WEIGHT_MODE_NONE,
    RUNTIME_META_FEATURE_NAMES,
    RUNTIME_TAIL_META_FEATURE_NAMES,
    SPLIT_MODE_SOURCE_SEED,
    SPLIT_NAME_TO_ID,
    TARGET_MODE_LOCAL_EV_NEGATIVE,
    TARGET_MODE_REALIZED_WHOLE_GAME_LOSS,
    TARGET_MODE_TOPK_CONFIRM_FIRE,
    confirm_tail_feature_vector,
    exclude_recommended_use_rows,
    feature_column_names,
    fixed_group_split,
    group_stratified_split,
    hu_delta_feature_vector,
    limit_rows_for_training,
    lookahead_proxy_feature_vector,
    materialize_training_rows,
    opportunity_proxy_feature_vector,
    preconfirm_meta_feature_vector,
    risk_feature_vector_for_row,
    risk_row_to_sample,
    sample_weights_for_metadata,
    resolved_thresholds,
    runtime_meta_feature_vector,
    runtime_tail_meta_feature_vector,
    split_group_key,
    stratified_split,
    threshold_metrics,
    topk_selection_metrics,
    train_model,
)


def _action_dict(index: int = 0):
    board = Board.from_rows(
        top=["Qh"],
        middle=["2c", "3d", "4h", "5s"],
        bottom=["7c", "8d", "9h", "Ts"],
    )
    action = generate_turn_actions(board, ["As", "Kh", "3c"])[index]
    return {
        "placements": [[card, row] for card, row in action.placements],
        "discards": list(action.discards),
    }


def _row(**overrides):
    row = {
        "schema": "hu_turn2_stage8b_counterfactual_loss_target_v1",
        "source_log": "run/runtime_decisions.jsonl",
        "config_id": "cfg",
        "hand_seed": 1,
        "seat": "first",
        "seat_swap": "ab",
        "realized_delta": -4.0,
        "candidate_index": 3,
        "baseline_index": 1,
        "predicted_delta": 1.5,
        "gate_probability": 0.9,
        "candidate_ev_rank": 2,
        "confirm_delta": 2.0,
        "confirm_delta_se": 0.5,
        "confirm_paired_delta_summary": {
            "count": 128,
            "mean": 2.0,
            "paired_delta_standard_error": 0.5,
            "std": 5.0,
            "min": -10.0,
            "p01": -8.0,
            "p05": -4.0,
            "p25": 0.0,
            "p50": 2.0,
            "p75": 4.0,
            "p95": 9.0,
            "p99": 12.0,
            "max": 14.0,
        },
        "local_replay_bucket": "local_positive_lcb",
        "recommended_training_use": "whole_game_risk_only",
        "use_for_local_ev_hard_negative": 0,
        "use_for_whole_game_risk_head": 1,
        "requires_local_replay": 0,
        "hero_board": {
            "top": ["Qh"],
            "middle": ["2c", "3d", "4h", "5s"],
            "bottom": ["7c", "8d", "9h", "Ts"],
        },
        "opponent_board": {
            "top": ["Ah"],
            "middle": ["Jc", "Jd", "6h", "6s"],
            "bottom": ["4c", "5d", "6c", "7s"],
        },
        "dead_cards": ["9c", "9d"],
        "cards_to_place": ["As", "Kh", "3c"],
        "baseline_action": _action_dict(1),
        "candidate_action": _action_dict(0),
        "local_replay_status": "ok",
        "local_replay_future_samples": 512,
        "local_replay_delta": 2.5,
        "local_replay_delta_se": 0.5,
        "local_replay_lcb196": 1.52,
        "local_replay_label": "positive",
        "local_replay_action_mapping_status": "ok",
    }
    row.update(overrides)
    return row


def test_risk_row_to_sample_encodes_candidate_action():
    sample = risk_row_to_sample(_row())

    assert sample["phase"] == "hu_turn2_stage8c_topk_risk"
    assert sample["seat"] == "first"
    assert sample["actions"][0]["placements"]


def test_risk_row_to_sample_can_encode_baseline_action():
    row = _row()
    candidate = risk_row_to_sample(row)
    baseline = risk_row_to_sample(row, action_key="baseline_action")

    assert candidate["actions"][0] == row["candidate_action"]
    assert baseline["actions"][0] == row["baseline_action"]
    assert candidate["actions"][0] != baseline["actions"][0]


def test_materialize_training_rows_keeps_whole_game_risk_separate_from_local_ev():
    rows = [
        _row(hand_seed=1, recommended_training_use="whole_game_risk_only", realized_delta=-4.0),
        _row(
            hand_seed=2,
            recommended_training_use="whole_game_non_loss_control",
            realized_delta=3.0,
            use_for_whole_game_risk_head=1,
        ),
        _row(
            hand_seed=3,
            recommended_training_use="local_ev_hard_negative",
            use_for_whole_game_risk_head=0,
            use_for_local_ev_hard_negative=1,
            local_replay_bucket="local_negative",
        ),
    ]

    features, labels, metadata = materialize_training_rows(rows, feature_mode=FEATURE_MODE_HU_PLUS_RUNTIME_META)

    assert features.shape[0] == 2
    assert labels.tolist() == [1.0, 0.0]
    assert [row["recommended_training_use"] for row in metadata] == [
        "whole_game_risk_only",
        "whole_game_non_loss_control",
    ]


def test_materialize_training_rows_uses_non_loss_controls_without_local_replay():
    rows = [
        _row(hand_seed=1, recommended_training_use="whole_game_risk_only", realized_delta=-4.0),
        _row(
            hand_seed=2,
            recommended_training_use="whole_game_non_loss_control",
            realized_delta=3.0,
            use_for_whole_game_risk_head=1,
            local_replay_bucket="missing",
            local_replay_status="",
            local_replay_action_mapping_status="",
            local_replay_delta="",
            local_replay_delta_se="",
            local_replay_lcb196="",
            local_replay_label="",
        ),
        _row(
            hand_seed=3,
            recommended_training_use="whole_game_risk_only",
            realized_delta=-3.0,
            local_replay_bucket="missing",
            local_replay_status="",
            local_replay_action_mapping_status="",
            local_replay_delta="",
            local_replay_delta_se="",
            local_replay_lcb196="",
            local_replay_label="",
        ),
    ]

    features, labels, metadata = materialize_training_rows(
        rows,
        feature_mode=FEATURE_MODE_RUNTIME_META_ONLY,
        target_mode="whole_game_risk",
    )

    assert features.shape[0] == 2
    assert labels.tolist() == [1.0, 0.0]
    assert [row["hand_seed"] for row in metadata] == [1, 2]
    assert [row["recommended_training_use"] for row in metadata] == [
        "whole_game_risk_only",
        "whole_game_non_loss_control",
    ]


def test_materialize_training_rows_supports_realized_whole_game_loss_without_local_replay():
    loss = _row(
        hand_seed=1,
        recommended_training_use="requires_local_replay",
        realized_delta=-4.0,
        local_replay_status="",
        local_replay_action_mapping_status="",
        local_replay_delta="",
        local_replay_delta_se="",
    )
    non_loss = _row(
        hand_seed=2,
        recommended_training_use="whole_game_non_loss_control",
        realized_delta=3.0,
        local_replay_status="",
        local_replay_action_mapping_status="",
        local_replay_delta="",
        local_replay_delta_se="",
    )

    features, labels, metadata = materialize_training_rows(
        [loss, non_loss],
        feature_mode=FEATURE_MODE_RUNTIME_META_ONLY,
        target_mode=TARGET_MODE_REALIZED_WHOLE_GAME_LOSS,
    )

    assert features.shape == (2, len(RUNTIME_META_FEATURE_NAMES))
    assert labels.tolist() == [1.0, 0.0]
    assert [row["risk_target_group"] for row in metadata] == [
        "realized_whole_game_loss",
        "realized_whole_game_non_loss",
    ]
    assert [row["recommended_training_use"] for row in metadata] == [
        "requires_local_replay",
        "whole_game_non_loss_control",
    ]


def test_materialize_training_rows_supports_topk_confirm_fire_without_local_replay():
    positive = _row(
        recommended_training_use="topk_confirm_realized_positive",
        local_replay_status="",
        local_replay_action_mapping_status="",
        candidate_source="fired",
        candidate_index=3,
        baseline_index=1,
        state_signature="state-a",
        action_signature="action-a",
        baseline_action_signature="baseline-a",
    )
    negative = _row(
        hand_seed=2,
        recommended_training_use="topk_confirm_rejected",
        local_replay_status="",
        local_replay_action_mapping_status="",
    )

    features, labels, metadata = materialize_training_rows(
        [positive, negative],
        feature_mode=FEATURE_MODE_RUNTIME_META_ONLY,
        target_mode=TARGET_MODE_TOPK_CONFIRM_FIRE,
    )

    assert features.shape == (2, len(RUNTIME_META_FEATURE_NAMES))
    assert labels.tolist() == [1.0, 0.0]
    assert [row["risk_target_group"] for row in metadata] == [
        "topk_confirm_realized_positive",
        "topk_confirm_rejected",
    ]
    assert metadata[0]["candidate_source"] == "fired"
    assert metadata[0]["candidate_index"] == 3
    assert metadata[0]["baseline_index"] == 1
    assert metadata[0]["state_signature"] == "state-a"
    assert metadata[0]["action_signature"] == "action-a"


def test_materialize_training_rows_can_append_runtime_meta_features():
    row = _row(
        predicted_delta=1.25,
        gate_probability=0.75,
        confirm_delta=2.0,
        confirm_delta_se=0.5,
        candidate_ev_rank=3,
        seat="second",
    )

    features, labels, _metadata = materialize_training_rows([row], feature_mode=FEATURE_MODE_HU_PLUS_RUNTIME_META)
    meta = runtime_meta_feature_vector(row)

    assert features.shape == (1, HU_FEATURE_DIM + len(RUNTIME_META_FEATURE_NAMES))
    assert labels.tolist() == [1.0]
    assert features[0, -len(RUNTIME_META_FEATURE_NAMES) :].tolist() == pytest.approx(meta.tolist())


def test_materialize_training_rows_can_use_runtime_meta_only():
    row = _row(
        predicted_delta=1.25,
        gate_probability=0.75,
        confirm_delta=2.0,
        confirm_delta_se=0.5,
        candidate_ev_rank=3,
        seat="second",
    )

    features, labels, _metadata = materialize_training_rows([row], feature_mode=FEATURE_MODE_RUNTIME_META_ONLY)
    meta = runtime_meta_feature_vector(row)

    assert features.shape == (1, len(RUNTIME_META_FEATURE_NAMES))
    assert labels.tolist() == [1.0]
    assert features[0].tolist() == pytest.approx(meta.tolist())


def test_preconfirm_meta_excludes_confirm_mc_fields():
    row = _row(
        predicted_delta=1.25,
        gate_probability=0.75,
        confirm_delta=99.0,
        confirm_delta_se=12.0,
        candidate_ev_rank=3,
        model_score=0.4,
        topk_score=0.8,
        top_k=5,
        seat="second",
    )
    changed_confirm = dict(row, confirm_delta=-99.0, confirm_delta_se=1.0)

    features = preconfirm_meta_feature_vector(row)
    changed = preconfirm_meta_feature_vector(changed_confirm)

    assert features.shape == (len(PRECONFIRM_META_FEATURE_NAMES),)
    assert features.tolist() == pytest.approx(changed.tolist())
    assert features[-1] == pytest.approx(1.0)


def test_materialize_training_rows_can_use_opportunity_plus_preconfirm_meta():
    row = _row(
        recommended_training_use="topk_confirm_realized_positive",
        local_replay_status="",
        local_replay_action_mapping_status="",
        model_score=0.4,
        topk_score=0.8,
        top_k=5,
    )

    features, labels, _metadata = materialize_training_rows(
        [row],
        feature_mode=FEATURE_MODE_OPPORTUNITY_PROXY_PLUS_PRECONFIRM_META,
        target_mode=TARGET_MODE_TOPK_CONFIRM_FIRE,
    )

    assert features.shape == (1, len(OPPORTUNITY_PROXY_FEATURE_NAMES) + len(PRECONFIRM_META_FEATURE_NAMES))
    assert labels.tolist() == [1.0]


def test_limit_rows_for_training_supports_first_and_stratified_modes():
    rows = []
    for index in range(12):
        rows.append(
            _row(
                hand_seed=index,
                recommended_training_use=(
                    "topk_confirm_realized_positive" if index >= 8 else "topk_confirm_rejected"
                ),
                seat="first" if index % 2 == 0 else "second",
                candidate_source="fired" if index >= 8 else "below_confirm_delta",
            )
        )

    first = limit_rows_for_training(
        rows,
        max_rows=4,
        mode=MAX_ROWS_MODE_FIRST,
        seed=7,
        target_mode=TARGET_MODE_TOPK_CONFIRM_FIRE,
    )
    stratified = limit_rows_for_training(
        rows,
        max_rows=4,
        mode=MAX_ROWS_MODE_STRATIFIED,
        seed=7,
        target_mode=TARGET_MODE_TOPK_CONFIRM_FIRE,
    )

    assert [row["hand_seed"] for row in first] == [0, 1, 2, 3]
    assert len(stratified) == 4
    assert any(row["recommended_training_use"] == "topk_confirm_realized_positive" for row in stratified)
    assert any(row["recommended_training_use"] == "topk_confirm_rejected" for row in stratified)
    assert {row["seat"] for row in stratified} == {"first", "second"}


def test_sample_weights_for_topk_confirm_fire_can_emphasize_replay_negatives():
    rows = [
        _row(recommended_training_use="topk_confirm_replay_negative"),
        _row(recommended_training_use="topk_confirm_realized_loss"),
        _row(recommended_training_use="topk_confirm_replay_positive"),
        _row(recommended_training_use="topk_confirm_rejected"),
    ]
    _features, _labels, metadata = materialize_training_rows(
        rows,
        feature_mode=FEATURE_MODE_RUNTIME_META_ONLY,
        target_mode=TARGET_MODE_TOPK_CONFIRM_FIRE,
    )

    weights = sample_weights_for_metadata(
        metadata,
        target_mode=TARGET_MODE_TOPK_CONFIRM_FIRE,
        topk_replay_negative_weight=8.0,
        topk_realized_loss_weight=2.0,
        topk_replay_positive_weight=3.0,
    )

    by_group = {row["risk_target_group"]: weights[index] for index, row in enumerate(metadata)}
    assert by_group["topk_confirm_replay_negative"] == pytest.approx(8.0)
    assert by_group["topk_confirm_realized_loss"] == pytest.approx(2.0)
    assert by_group["topk_confirm_replay_positive"] == pytest.approx(3.0)
    assert by_group["topk_confirm_rejected"] == pytest.approx(1.0)


def test_exclude_recommended_use_rows_removes_requested_controls_only():
    rows = [
        _row(recommended_training_use="topk_confirm_realized_positive"),
        _row(recommended_training_use="topk_confirm_topk_empty"),
        _row(recommended_training_use="topk_confirm_topk_empty"),
        _row(recommended_training_use="topk_confirm_rejected"),
    ]

    kept, removed = exclude_recommended_use_rows(rows, ["topk_confirm_topk_empty"])

    assert [row["recommended_training_use"] for row in kept] == [
        "topk_confirm_realized_positive",
        "topk_confirm_rejected",
    ]
    assert removed == {"topk_confirm_topk_empty": 2}


def test_topk_selection_metrics_reports_precision_and_observed_delta():
    labels = np.asarray([1, 0, 1, 0], dtype=np.float32)
    probabilities = np.asarray([0.9, 0.8, 0.7, 0.1], dtype=np.float32)
    realized = np.asarray([5.0, -2.0, 3.0, 0.0], dtype=np.float32)
    observed = np.asarray([True, True, False, True])

    rows = topk_selection_metrics(
        labels,
        probabilities,
        split_name="test",
        topk_counts=(1, 2, 3),
        realized_deltas=realized,
        realized_observed=observed,
    )

    assert rows[0]["topk"] == 1
    assert rows[0]["precision"] == pytest.approx(1.0)
    assert rows[0]["selected_observed_realized_delta_mean"] == pytest.approx(5.0)
    assert rows[1]["topk"] == 2
    assert rows[1]["precision"] == pytest.approx(0.5)
    assert rows[1]["selected_realized_delta_mean"] == pytest.approx(1.5)
    assert rows[2]["selected_unknown_realized_delta_count"] == 1


def test_downstream_trajectory_audit_fields_do_not_enter_runtime_features():
    row = _row()
    row_with_audit_fields = dict(row)
    row_with_audit_fields.update(
        {
            "downstream_trajectory_complete": 1,
            "downstream_trajectory_present_fields": 16,
            "downstream_trajectory_total_fields": 16,
        }
    )
    audit_field_fragments = ("downstream_trajectory", "trajectory_complete", "present_fields")

    for feature_mode in (
        FEATURE_MODE_RUNTIME_META_ONLY,
        FEATURE_MODE_RUNTIME_TAIL_META_ONLY,
        FEATURE_MODE_HU_PLUS_RUNTIME_META,
        FEATURE_MODE_HU_PLUS_RUNTIME_TAIL_META,
        FEATURE_MODE_HU_DELTA_PLUS_RUNTIME_TAIL_META,
        FEATURE_MODE_OPPORTUNITY_PROXY_PLUS_RUNTIME_TAIL_META,
        FEATURE_MODE_LOOKAHEAD_PROXY_PLUS_PRECONFIRM_META,
        FEATURE_MODE_LOOKAHEAD_PROXY_PLUS_RUNTIME_TAIL_META,
        FEATURE_MODE_OPPORTUNITY_LOOKAHEAD_PROXY_PLUS_PRECONFIRM_META,
        FEATURE_MODE_OPPORTUNITY_LOOKAHEAD_PROXY_PLUS_RUNTIME_TAIL_META,
    ):
        names = feature_column_names(feature_mode)
        assert not any(fragment in name for name in names for fragment in audit_field_fragments)
        assert risk_feature_vector_for_row(row_with_audit_fields, feature_mode).tolist() == pytest.approx(
            risk_feature_vector_for_row(row, feature_mode).tolist()
        )


def test_materialize_training_rows_can_use_runtime_tail_meta_only():
    row = _row(
        predicted_delta=1.25,
        gate_probability=0.75,
        confirm_delta=2.0,
        confirm_delta_se=0.5,
        candidate_ev_rank=3,
        seat="second",
    )

    features, labels, _metadata = materialize_training_rows([row], feature_mode=FEATURE_MODE_RUNTIME_TAIL_META_ONLY)
    meta = runtime_tail_meta_feature_vector(row)
    tail = confirm_tail_feature_vector(row)

    assert features.shape == (1, len(RUNTIME_TAIL_META_FEATURE_NAMES))
    assert labels.tolist() == [1.0]
    assert len(tail) == len(CONFIRM_TAIL_FEATURE_NAMES)
    assert features[0].tolist() == pytest.approx(meta.tolist())


def test_runtime_tail_meta_accepts_json_encoded_confirm_summary():
    row = _row(
        confirm_paired_delta_summary=(
            '{"count":64,"mean":1.5,"paired_delta_standard_error":0.25,'
            '"std":2.0,"min":-3.0,"p01":-2.5,"p05":-1.0,"p25":0.5,'
            '"p50":1.5,"p75":2.5,"p95":4.0,"p99":5.0,"max":6.0}'
        )
    )

    tail = confirm_tail_feature_vector(row)

    assert tail[0] == pytest.approx(4.174387, rel=1e-5)
    assert tail[1] == pytest.approx(1.5)
    assert tail[2] == pytest.approx(0.25)
    assert tail[20] == pytest.approx(3.0)


def test_runtime_tail_meta_uses_component_delta_summaries():
    row = _row(
        confirm_paired_delta_summary={
            "count": 64,
            "mean": 1.5,
            "paired_delta_standard_error": 0.25,
            "std": 2.0,
            "min": -3.0,
            "p01": -2.5,
            "p05": -1.0,
            "p25": 0.5,
            "p50": 1.5,
            "p75": 2.5,
            "p95": 4.0,
            "p99": 5.0,
            "max": 6.0,
            "lt0_rate": 0.125,
            "le_neg6_rate": 0.0625,
            "le_neg12_rate": 0.03125,
            "le_neg20_rate": 0.015625,
            "component_delta_summaries": {
                "fl_delta": {
                    "mean": -2.0,
                    "min": -20.0,
                    "p05": -12.0,
                    "p25": -5.0,
                    "p50": 0.0,
                    "p75": 0.0,
                    "p95": 10.0,
                    "max": 10.0,
                    "lt0_rate": 0.25,
                    "le_neg6_rate": 0.125,
                    "le_neg12_rate": 0.0625,
                    "le_neg20_rate": 0.03125,
                },
                "foul_delta": {
                    "mean": -0.5,
                    "min": -12.0,
                    "p05": -6.0,
                    "p25": 0.0,
                    "p50": 0.0,
                    "p75": 0.0,
                    "p95": 6.0,
                    "max": 6.0,
                    "lt0_rate": 0.1,
                    "le_neg6_rate": 0.05,
                    "le_neg12_rate": 0.01,
                    "le_neg20_rate": 0.0,
                },
            },
        }
    )

    names = list(CONFIRM_TAIL_FEATURE_NAMES)
    values = dict(zip(names, confirm_tail_feature_vector(row).tolist()))

    assert values["confirm_tail_lt0_rate"] == pytest.approx(0.125)
    assert values["confirm_tail_le_neg20_rate"] == pytest.approx(0.015625)
    assert values["confirm_component_fl_delta_mean"] == pytest.approx(-2.0)
    assert values["confirm_component_fl_delta_p05_loss"] == pytest.approx(12.0)
    assert values["confirm_component_fl_delta_le_neg20_rate"] == pytest.approx(0.03125)
    assert values["confirm_component_foul_delta_min_loss"] == pytest.approx(12.0)
    assert values["confirm_component_foul_delta_le_neg12_rate"] == pytest.approx(0.01)


def test_materialize_training_rows_can_append_runtime_tail_meta_features():
    row = _row()

    features, labels, _metadata = materialize_training_rows([row], feature_mode=FEATURE_MODE_HU_PLUS_RUNTIME_TAIL_META)

    assert features.shape == (1, HU_FEATURE_DIM + len(RUNTIME_TAIL_META_FEATURE_NAMES))
    assert labels.tolist() == [1.0]
    assert features[0, -len(RUNTIME_TAIL_META_FEATURE_NAMES) :].tolist() == pytest.approx(
        runtime_tail_meta_feature_vector(row).tolist()
    )


def test_materialize_training_rows_can_use_hu_delta_features():
    row = _row()

    features, labels, _metadata = materialize_training_rows([row], feature_mode=FEATURE_MODE_HU_DELTA_ONLY)
    swapped = dict(row)
    swapped["candidate_action"], swapped["baseline_action"] = row["baseline_action"], row["candidate_action"]

    assert features.shape == (1, HU_FEATURE_DIM)
    assert labels.tolist() == [1.0]
    assert features[0].tolist() == pytest.approx(hu_delta_feature_vector(row).tolist())
    assert hu_delta_feature_vector(swapped).tolist() == pytest.approx((-features[0]).tolist())


def test_materialize_training_rows_can_use_hu_delta_plus_runtime_tail_meta():
    row = _row()

    features, labels, _metadata = materialize_training_rows(
        [row],
        feature_mode=FEATURE_MODE_HU_DELTA_PLUS_RUNTIME_TAIL_META,
    )

    assert features.shape == (1, HU_FEATURE_DIM + len(RUNTIME_TAIL_META_FEATURE_NAMES))
    assert labels.tolist() == [1.0]
    assert features[0, :HU_FEATURE_DIM].tolist() == pytest.approx(hu_delta_feature_vector(row).tolist())
    assert features[0, HU_FEATURE_DIM:].tolist() == pytest.approx(runtime_tail_meta_feature_vector(row).tolist())


def test_risk_feature_vector_for_row_matches_materialized_training_features():
    row = _row()

    for feature_mode in (
        FEATURE_MODE_RUNTIME_META_ONLY,
        FEATURE_MODE_RUNTIME_TAIL_META_ONLY,
        FEATURE_MODE_OPPORTUNITY_PROXY_PLUS_RUNTIME_TAIL_META,
        FEATURE_MODE_LOOKAHEAD_PROXY_PLUS_RUNTIME_TAIL_META,
        FEATURE_MODE_OPPORTUNITY_LOOKAHEAD_PROXY_PLUS_RUNTIME_TAIL_META,
    ):
        features, _labels, _metadata = materialize_training_rows([row], feature_mode=feature_mode)
        runtime_feature = risk_feature_vector_for_row(row, feature_mode)
        assert runtime_feature.shape == features[0].shape
        assert runtime_feature.tolist() == pytest.approx(features[0].tolist())


def test_opportunity_proxy_features_compare_candidate_and_baseline_after_boards():
    row = _row(
        hero_board={"top": ["Qh"], "middle": ["2c", "3d", "4h"], "bottom": ["7c", "8d", "9h", "Ts"]},
        cards_to_place=["Qs", "Kh", "3c"],
        candidate_action={"placements": [["Qs", "top"], ["3c", "middle"]], "discards": ["Kh"]},
        baseline_action={"placements": [["Kh", "top"], ["3c", "middle"]], "discards": ["Qs"]},
        dead_cards=["9c", "9d"],
    )

    vector = opportunity_proxy_feature_vector(row)
    names = list(OPPORTUNITY_PROXY_FEATURE_NAMES)
    values = dict(zip(names, vector.tolist()))

    assert vector.shape == (len(OPPORTUNITY_PROXY_FEATURE_NAMES),)
    assert values["opportunity_candidate_top_pair_qqplus_now"] == 1.0
    assert values["opportunity_baseline_top_pair_qqplus_now"] == 0.0
    assert values["opportunity_delta_top_fl_entry_now"] == 0.0
    assert values["opportunity_delta_top_pair_qqplus_now"] == 1.0


def test_materialize_training_rows_can_use_opportunity_proxy_modes():
    row = _row()

    features, labels, _metadata = materialize_training_rows([row], feature_mode=FEATURE_MODE_OPPORTUNITY_PROXY_ONLY)
    with_tail, labels_tail, _metadata_tail = materialize_training_rows(
        [row],
        feature_mode=FEATURE_MODE_OPPORTUNITY_PROXY_PLUS_RUNTIME_TAIL_META,
    )

    assert features.shape == (1, len(OPPORTUNITY_PROXY_FEATURE_NAMES))
    assert labels.tolist() == [1.0]
    assert features[0].tolist() == pytest.approx(opportunity_proxy_feature_vector(row).tolist())
    assert with_tail.shape == (1, len(OPPORTUNITY_PROXY_FEATURE_NAMES) + len(RUNTIME_TAIL_META_FEATURE_NAMES))
    assert labels_tail.tolist() == [1.0]
    assert with_tail[0, : len(OPPORTUNITY_PROXY_FEATURE_NAMES)].tolist() == pytest.approx(features[0].tolist())
    assert with_tail[0, len(OPPORTUNITY_PROXY_FEATURE_NAMES) :].tolist() == pytest.approx(
        runtime_tail_meta_feature_vector(row).tolist()
    )


def test_lookahead_proxy_features_are_finite_and_materializable():
    row = _row(
        hero_board={"top": ["Qh"], "middle": ["2c", "3d"], "bottom": ["7c", "8d", "9h", "Ts"]},
        cards_to_place=["As", "Kh", "3c"],
        candidate_action={"placements": [["As", "top"], ["Kh", "middle"]], "discards": ["3c"]},
        baseline_action={"placements": [["Kh", "top"], ["3c", "middle"]], "discards": ["As"]},
    )

    vector = lookahead_proxy_feature_vector(row)
    features, labels, _metadata = materialize_training_rows([row], feature_mode=FEATURE_MODE_LOOKAHEAD_PROXY_ONLY)
    with_tail, labels_tail, _metadata_tail = materialize_training_rows(
        [row],
        feature_mode=FEATURE_MODE_LOOKAHEAD_PROXY_PLUS_RUNTIME_TAIL_META,
    )
    with_preconfirm, labels_preconfirm, _metadata_preconfirm = materialize_training_rows(
        [row],
        feature_mode=FEATURE_MODE_LOOKAHEAD_PROXY_PLUS_PRECONFIRM_META,
    )
    combined, labels_combined, _metadata_combined = materialize_training_rows(
        [row],
        feature_mode=FEATURE_MODE_OPPORTUNITY_LOOKAHEAD_PROXY_PLUS_RUNTIME_TAIL_META,
    )
    combined_preconfirm, labels_combined_preconfirm, _metadata_combined_preconfirm = materialize_training_rows(
        [row],
        feature_mode=FEATURE_MODE_OPPORTUNITY_LOOKAHEAD_PROXY_PLUS_PRECONFIRM_META,
    )

    assert vector.shape == (len(LOOKAHEAD_PROXY_FEATURE_NAMES),)
    assert all(abs(value) < 1e6 for value in vector.tolist())
    assert vector[0] > 0.0
    assert features.shape == (1, len(LOOKAHEAD_PROXY_FEATURE_NAMES))
    assert labels.tolist() == [1.0]
    assert features[0].tolist() == pytest.approx(vector.tolist())
    assert with_tail.shape == (1, len(LOOKAHEAD_PROXY_FEATURE_NAMES) + len(RUNTIME_TAIL_META_FEATURE_NAMES))
    assert labels_tail.tolist() == [1.0]
    assert with_preconfirm.shape == (1, len(LOOKAHEAD_PROXY_FEATURE_NAMES) + len(PRECONFIRM_META_FEATURE_NAMES))
    assert labels_preconfirm.tolist() == [1.0]
    assert combined.shape == (
        1,
        len(OPPORTUNITY_PROXY_FEATURE_NAMES) + len(LOOKAHEAD_PROXY_FEATURE_NAMES) + len(RUNTIME_TAIL_META_FEATURE_NAMES),
    )
    assert labels_combined.tolist() == [1.0]
    assert combined_preconfirm.shape == (
        1,
        len(OPPORTUNITY_PROXY_FEATURE_NAMES) + len(LOOKAHEAD_PROXY_FEATURE_NAMES) + len(PRECONFIRM_META_FEATURE_NAMES),
    )
    assert labels_combined_preconfirm.tolist() == [1.0]


def test_lookahead_proxy_features_are_stable_for_card_order_changes():
    row = _row(
        hero_board={"top": ["Qh"], "middle": ["2c", "3d"], "bottom": ["7c", "8d", "9h", "Ts"]},
        opponent_board={
            "top": ["Ah"],
            "middle": ["Jc", "Jd", "6h", "6s"],
            "bottom": ["4c", "5d", "6c", "7s"],
        },
        dead_cards=["9c", "9d", "2h"],
        cards_to_place=["As", "Kh", "3c"],
        candidate_action={"placements": [["As", "top"], ["Kh", "middle"]], "discards": ["3c"]},
        baseline_action={"placements": [["Kh", "top"], ["3c", "middle"]], "discards": ["As"]},
    )
    permuted = _row(
        hero_board={"top": ["Qh"], "middle": ["3d", "2c"], "bottom": ["Ts", "9h", "8d", "7c"]},
        opponent_board={
            "top": ["Ah"],
            "middle": ["6s", "6h", "Jd", "Jc"],
            "bottom": ["7s", "6c", "5d", "4c"],
        },
        dead_cards=["2h", "9d", "9c"],
        cards_to_place=["3c", "Kh", "As"],
        candidate_action={"placements": [["Kh", "middle"], ["As", "top"]], "discards": ["3c"]},
        baseline_action={"placements": [["3c", "middle"], ["Kh", "top"]], "discards": ["As"]},
    )

    assert lookahead_proxy_feature_vector(permuted).tolist() == pytest.approx(
        lookahead_proxy_feature_vector(row).tolist()
    )


def test_lookahead_proxy_features_zero_when_after_board_is_not_t3_sized():
    vector = lookahead_proxy_feature_vector(_row())

    assert vector.shape == (len(LOOKAHEAD_PROXY_FEATURE_NAMES),)
    assert vector.tolist() == pytest.approx([0.0] * len(LOOKAHEAD_PROXY_FEATURE_NAMES))


def test_materialize_training_rows_can_target_local_ev_negative():
    rows = [
        _row(hand_seed=1, local_replay_label="negative", local_replay_bucket="local_negative"),
        _row(hand_seed=2, local_replay_label="positive", local_replay_bucket="local_positive_lcb"),
        _row(hand_seed=3, local_replay_label="gray", local_replay_bucket="local_positive_gray"),
    ]

    features, labels, metadata = materialize_training_rows(
        rows,
        feature_mode=FEATURE_MODE_RUNTIME_META_ONLY,
        target_mode=TARGET_MODE_LOCAL_EV_NEGATIVE,
    )

    assert features.shape == (2, len(RUNTIME_META_FEATURE_NAMES))
    assert labels.tolist() == [1.0, 0.0]
    assert [row["risk_target_group"] for row in metadata] == [
        "local_ev_negative",
        "local_ev_positive_lcb_control",
    ]


def test_resolved_thresholds_do_not_mix_defaults_with_explicit_values():
    assert resolved_thresholds(None) == list(DEFAULT_THRESHOLDS)
    assert resolved_thresholds([0.6, 0.8]) == [0.6, 0.8]


def test_stratified_split_and_train_model_smoke():
    pytest.importorskip("torch")
    rows = []
    for index in range(12):
        use = "whole_game_risk_only" if index % 2 == 0 else "whole_game_non_loss_control"
        rows.append(
            _row(
                hand_seed=index,
                seat="first" if index % 4 < 2 else "second",
                recommended_training_use=use,
                realized_delta=-3.0 if use == "whole_game_risk_only" else 2.0,
                use_for_whole_game_risk_head=1,
            )
        )
    features, labels, metadata = materialize_training_rows(rows, feature_mode=FEATURE_MODE_HU_PLUS_RUNTIME_META)
    split = stratified_split(metadata, labels, seed=7, train_fraction=0.5, val_fraction=0.25)

    assert set(split.tolist()) == {0, 1, 2}

    payload, probabilities, history = train_model(
        features,
        labels,
        split,
        hidden_layer_sizes=(8,),
        dropout=0.0,
        learning_rate=1e-3,
        weight_decay=0.0,
        batch_size=4,
        epochs=1,
        patience=1,
        seed=7,
        device_choice="cpu",
        feature_mode=FEATURE_MODE_HU_PLUS_RUNTIME_META,
        target_mode="whole_game_risk",
        pos_weight_mode=POS_WEIGHT_MODE_NONE,
        sample_weights=np.linspace(1.0, 2.0, num=labels.shape[0], dtype=np.float32),
    )

    assert payload["model_kind"] == "hu_turn2_stage8c_whole_game_risk_head_mlp"
    assert payload["feature_mode"] == FEATURE_MODE_HU_PLUS_RUNTIME_META
    assert payload["feature_dim"] == HU_FEATURE_DIM + len(RUNTIME_META_FEATURE_NAMES)
    assert payload["pos_weight_mode"] == POS_WEIGHT_MODE_NONE
    assert payload["pos_weight"] == 1.0
    assert probabilities.shape == labels.shape
    assert history


def test_group_stratified_split_keeps_source_seed_in_one_split():
    rows = []
    for seed_index in range(6):
        for duplicate in range(2):
            use = "whole_game_risk_only" if duplicate == 0 else "whole_game_non_loss_control"
            rows.append(
                _row(
                    hand_seed=f"20260699{seed_index:02d}{duplicate:04d}",
                    source_log=f"outputs/run/shard_seed20260699{seed_index:02d}/runtime_decisions.jsonl",
                    seat="first" if seed_index % 2 == 0 else "second",
                    recommended_training_use=use,
                    realized_delta=-3.0 if use == "whole_game_risk_only" else 2.0,
                    use_for_whole_game_risk_head=1,
                )
            )
    features, labels, metadata = materialize_training_rows(rows)
    split = group_stratified_split(
        metadata,
        labels,
        seed=11,
        train_fraction=0.5,
        val_fraction=0.25,
        split_mode=SPLIT_MODE_SOURCE_SEED,
    )

    assert set(split.tolist()) == {0, 1, 2}
    group_to_splits = {}
    for index, row in enumerate(metadata):
        group_to_splits.setdefault(row["split_group_source_seed"], set()).add(int(split[index]))

    assert len(group_to_splits) == 6
    assert all(len(values) == 1 for values in group_to_splits.values())
    assert split_group_key(rows[0], SPLIT_MODE_SOURCE_SEED).startswith("20260699")


def test_fixed_group_split_uses_named_source_seed_groups():
    rows = []
    for seed_index in range(4):
        for duplicate in range(2):
            rows.append(
                _row(
                    hand_seed=f"2026072{seed_index:03d}{duplicate:04d}",
                    source_log=f"outputs/run/shard_seed2026072{seed_index:03d}/runtime_decisions.jsonl",
                    recommended_training_use=(
                        "topk_confirm_realized_positive" if duplicate == 0 else "topk_confirm_rejected"
                    ),
                )
            )
    _features, labels, metadata = materialize_training_rows(
        rows,
        feature_mode=FEATURE_MODE_RUNTIME_META_ONLY,
        target_mode=TARGET_MODE_TOPK_CONFIRM_FIRE,
    )

    split = fixed_group_split(
        metadata,
        split_mode=SPLIT_MODE_SOURCE_SEED,
        val_groups={"2026072001"},
        test_groups={"2026072002"},
    )

    groups_by_split = {}
    for index, row in enumerate(metadata):
        groups_by_split.setdefault(int(split[index]), set()).add(row["split_group_source_seed"])
    assert groups_by_split[SPLIT_NAME_TO_ID["val"]] == {"2026072001"}
    assert groups_by_split[SPLIT_NAME_TO_ID["test"]] == {"2026072002"}
    assert labels.shape[0] == 8


def test_threshold_metrics_reports_realized_delta_value():
    metrics = threshold_metrics(
        labels=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        probabilities=np.asarray([0.9, 0.8, 0.1], dtype=np.float32),
        split_name="test",
        thresholds=[0.5],
        realized_deltas=np.asarray([6.0, -2.0, 0.0], dtype=np.float32),
        realized_observed=np.asarray([True, True, False], dtype=bool),
    )

    row = metrics[0]
    assert row["selected_rows"] == 2
    assert row["selected_realized_delta_sum"] == pytest.approx(4.0)
    assert row["selected_realized_delta_mean"] == pytest.approx(2.0)
    assert row["estimated_realized_delta_per_row"] == pytest.approx(4.0 / 3.0)
    assert row["selected_realized_loss_count"] == 1
    assert row["selected_realized_max_loss"] == pytest.approx(2.0)
    assert row["selected_observed_realized_delta_count"] == 2
    assert row["selected_unknown_realized_delta_count"] == 0
    assert row["selected_observed_realized_delta_mean"] == pytest.approx(2.0)


def test_threshold_metrics_reports_unknown_realized_delta_count():
    metrics = threshold_metrics(
        labels=np.asarray([1.0, 0.0], dtype=np.float32),
        probabilities=np.asarray([0.9, 0.8], dtype=np.float32),
        split_name="test",
        thresholds=[0.5],
        realized_deltas=np.asarray([6.0, 0.0], dtype=np.float32),
        realized_observed=np.asarray([True, False], dtype=bool),
    )

    row = metrics[0]
    assert row["selected_rows"] == 2
    assert row["selected_observed_realized_delta_count"] == 1
    assert row["selected_unknown_realized_delta_count"] == 1
    assert row["selected_observed_realized_delta_mean"] == pytest.approx(6.0)
