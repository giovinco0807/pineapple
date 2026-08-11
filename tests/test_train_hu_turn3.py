import numpy as np
import pytest

from ofc_regular.train_hu_turn3 import (
    build_action_source_weights,
    parse_source_weights,
    sample_source,
    source_counts,
    weighted_action_summary,
)
from ofc_regular import hu_turn3_model


def test_parse_source_weights_accepts_repeated_and_comma_separated_entries():
    weights = parse_source_weights(
        [
            "self_play_rollout=1.0,mined_state_rollout=0.25",
            "override_counterfactual=2",
        ]
    )

    assert weights == {
        "self_play_rollout": 1.0,
        "mined_state_rollout": 0.25,
        "override_counterfactual": 2.0,
    }


def test_parse_source_weights_rejects_invalid_entries():
    with pytest.raises(ValueError, match="SOURCE=WEIGHT"):
        parse_source_weights(["mined_state_rollout"])
    with pytest.raises(ValueError, match="non-negative"):
        parse_source_weights(["mined_state_rollout=-1"])


def test_build_action_source_weights_repeats_sample_weight_per_action():
    samples = [
        {"source": "self_play_rollout", "actions": [{}, {}]},
        {"source": "mined_state_rollout", "actions": [{}, {}, {}]},
        {"actions": [{}]},
    ]

    weights = build_action_source_weights(samples, {"mined_state_rollout": 0.25})

    assert weights is not None
    np.testing.assert_allclose(weights, [1.0, 1.0, 0.25, 0.25, 0.25, 1.0])
    assert weighted_action_summary(weights)["train_action_weight_sum"] == pytest.approx(3.75)


def test_build_action_source_weights_rejects_all_zero_actions():
    samples = [{"source": "mined_state_rollout", "actions": [{}, {}]}]

    with pytest.raises(ValueError, match="zeroed every training action"):
        build_action_source_weights(samples, {"mined_state_rollout": 0.0})


def test_source_counts_uses_unknown_for_missing_source():
    samples = [{"source": "a"}, {"source": "a"}, {}]

    assert source_counts(samples) == {"a": 2, "unknown": 1}


def test_source_helpers_fall_back_to_label_source_then_profile():
    samples = [
        {"label_source": "stage9f_p2_refinement", "actions": [{}, {}]},
        {"profile": "stage9f_fast_t2_t1_teacher", "actions": [{}]},
    ]

    assert sample_source(samples[0]) == "stage9f_p2_refinement"
    assert sample_source(samples[1]) == "stage9f_fast_t2_t1_teacher"
    assert source_counts(samples) == {
        "stage9f_fast_t2_t1_teacher": 1,
        "stage9f_p2_refinement": 1,
    }

    weights = build_action_source_weights(samples, {"stage9f_p2_refinement": 5.0})

    assert weights is not None
    np.testing.assert_allclose(weights, [5.0, 5.0, 1.0])


def test_evaluate_model_top3_includes_stable_top1_on_prediction_ties(monkeypatch):
    samples = [
        {"sample_id": "a", "actions": [{}, {}, {}, {}]},
        {"sample_id": "b", "actions": [{}, {}, {}, {}]},
    ]
    targets_by_id = {
        "a": np.asarray([10.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "b": np.asarray([0.0, 10.0, 0.0, 0.0], dtype=np.float64),
    }

    def fake_sample_to_matrix(sample):
        targets = targets_by_id[sample["sample_id"]]
        return np.zeros((len(targets), 1), dtype=np.float32), targets

    class TiedModel:
        def predict_matrix(self, features):
            return np.zeros(features.shape[0], dtype=np.float64)

    monkeypatch.setattr(hu_turn3_model, "sample_to_matrix", fake_sample_to_matrix)

    metrics = hu_turn3_model.evaluate_model(TiedModel(), samples)

    assert metrics["top1_accuracy"] == pytest.approx(0.5)
    assert metrics["top3_accuracy"] == pytest.approx(1.0)
