import pytest

from ofc_regular.train_hu_turn2_stage8c_delta_head import (
    TARGET_MODE_OBSERVED_DELTA,
    TARGET_MODE_POLICY_DELTA,
    delta_target_for_row,
    materialize_delta_rows,
    regression_metrics,
    topk_delta_metrics,
)
from ofc_regular.train_hu_turn2_stage8c_risk_head import FEATURE_MODE_PRECONFIRM_META_ONLY


def _row(**overrides):
    row = {
        "recommended_training_use": "topk_confirm_realized_positive",
        "realized_delta": 4.0,
        "realized_delta_observed": True,
        "realized_delta_valid": True,
        "predicted_delta": 1.25,
        "gate_probability": 0.75,
        "candidate_ev_rank": 2,
        "model_score": 3.0,
        "topk_score": 0.5,
        "top_k": 5,
        "seat": "first",
    }
    row.update(overrides)
    return row


def test_policy_delta_treats_rejected_as_zero_policy_outcome():
    row = _row(
        recommended_training_use="topk_confirm_rejected",
        realized_delta=0.0,
        realized_delta_observed=False,
        realized_delta_valid=True,
    )

    assert delta_target_for_row(row, TARGET_MODE_POLICY_DELTA) == (0.0, "topk_confirm_rejected")
    assert delta_target_for_row(row, TARGET_MODE_OBSERVED_DELTA) is None


def test_observed_delta_keeps_replay_rows_and_skips_unobserved_rejected():
    rows = [
        _row(recommended_training_use="topk_confirm_replay_positive", realized_delta=1.5, local_replay_label="positive"),
        _row(recommended_training_use="topk_confirm_replay_negative", realized_delta=-0.5, local_replay_label="negative"),
        _row(recommended_training_use="topk_confirm_rejected", realized_delta=0.0, realized_delta_observed=False),
    ]

    features, targets, metadata, skipped = materialize_delta_rows(
        rows,
        feature_mode=FEATURE_MODE_PRECONFIRM_META_ONLY,
        target_mode=TARGET_MODE_OBSERVED_DELTA,
        target_clip=40.0,
    )

    assert features.shape[0] == 2
    assert targets.tolist() == pytest.approx([1.5, -0.5])
    assert [row["delta_target_group"] for row in metadata] == [
        "topk_confirm_replay_positive",
        "topk_confirm_replay_negative",
    ]
    assert skipped["target_missing"] == 1


def test_delta_metadata_preserves_action_identity_for_joining_predictions():
    rows = [
        _row(
            state_signature="state-a",
            action_signature="action-a",
            baseline_action_signature="baseline-a",
            candidate_index="",
            candidate_action_index="",
            rerank_best_index=7,
            baseline_index="",
            baseline_action_index=3,
        )
    ]

    _features, _targets, metadata, _skipped = materialize_delta_rows(
        rows,
        feature_mode=FEATURE_MODE_PRECONFIRM_META_ONLY,
        target_mode=TARGET_MODE_POLICY_DELTA,
        target_clip=40.0,
    )

    assert metadata[0]["state_signature"] == "state-a"
    assert metadata[0]["action_signature"] == "action-a"
    assert metadata[0]["baseline_action_signature"] == "baseline-a"
    assert metadata[0]["candidate_index"] == 7
    assert metadata[0]["baseline_index"] == 3


def test_policy_delta_can_clip_large_realized_values():
    rows = [
        _row(realized_delta=100.0),
        _row(recommended_training_use="topk_confirm_realized_loss", realized_delta=-100.0),
    ]

    _features, targets, _metadata, _skipped = materialize_delta_rows(
        rows,
        feature_mode=FEATURE_MODE_PRECONFIRM_META_ONLY,
        target_mode=TARGET_MODE_POLICY_DELTA,
        target_clip=10.0,
    )

    assert targets.tolist() == pytest.approx([10.0, -10.0])


def test_regression_and_topk_metrics_use_delta_not_binary_label():
    import numpy as np

    targets = np.asarray([-5.0, 1.0, 8.0, 2.0], dtype=np.float32)
    predictions = np.asarray([0.1, 0.2, 0.9, 0.8], dtype=np.float32)

    metrics = regression_metrics(targets, predictions, split_name="test")
    topk = topk_delta_metrics(targets, predictions, split_name="test", topk_counts=[2])[0]

    assert metrics["rows"] == 4
    assert metrics["positive_rate"] == 0.75
    assert topk["selected_rows"] == 2
    assert topk["selected_target_delta_sum"] == pytest.approx(10.0)
    assert topk["selected_positive_rate"] == 1.0
