import numpy as np
import pytest

from ofc_regular.train_hu_turn3 import (
    build_action_source_weights,
    parse_source_weights,
    source_counts,
    weighted_action_summary,
)


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
