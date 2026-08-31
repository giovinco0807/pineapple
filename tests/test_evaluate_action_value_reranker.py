import numpy as np
import pytest

from ai.training.evaluate_action_value_reranker import (
    build_group_bounds,
    filter_group_bounds_by_position,
    summarize_groups,
)


def test_filter_group_bounds_by_position_keeps_whole_groups():
    group_ids = np.asarray([0, 0, 1, 1, 1, 2], dtype=np.int64)
    positions = np.asarray([0, 0, 1, 1, 1, 0], dtype=np.int8)
    bounds = build_group_bounds(group_ids)

    assert filter_group_bounds_by_position(bounds, positions, "bb") == [
        (0, 2),
        (5, 6),
    ]
    assert filter_group_bounds_by_position(bounds, positions, "btn") == [(2, 5)]
    assert filter_group_bounds_by_position(bounds, positions, "all") == bounds


def test_filter_group_bounds_rejects_mixed_position_group():
    with pytest.raises(ValueError, match="mixed positions"):
        filter_group_bounds_by_position([(0, 2)], np.asarray([0, 1]), "bb")


def test_sample_metrics_only_use_selected_group_bounds():
    scores = np.asarray([1.0, 2.0, 100.0, 200.0], dtype=np.float32)
    predicted = np.asarray([1.0, 4.0, -100.0, -200.0], dtype=np.float32)
    zeros = np.zeros(4, dtype=np.float32)

    metrics, _ = summarize_groups(
        scores,
        zeros,
        zeros,
        predicted,
        zeros,
        zeros,
        [(0, 2)],
        [1],
        0.999,
    )

    assert metrics["samples"] == 2
    assert metrics["score_mae"] == 1.0
