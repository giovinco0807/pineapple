import numpy as np
import pytest

from ai.tutor.benchmark_t2_t3_value_model import add_source_comparison, adjusted_t3_priority_score


def test_adjusted_t3_priority_score_defaults_to_raw_score():
    scores = np.array([10.0, 9.0], dtype=np.float32)
    bust = np.array([0.0, 0.5], dtype=np.float32)
    fl = np.array([0.0, 0.8], dtype=np.float32)

    adjusted = adjusted_t3_priority_score(scores, bust, fl, bust_weight=0.0, fl_any_weight=0.0)

    assert adjusted.tolist() == pytest.approx([10.0, 9.0])


def test_adjusted_t3_priority_score_applies_bust_and_fl_weights():
    scores = np.array([10.0, 9.0], dtype=np.float32)
    bust = np.array([0.0, 0.5], dtype=np.float32)
    fl = np.array([0.0, 0.8], dtype=np.float32)

    adjusted = adjusted_t3_priority_score(scores, bust, fl, bust_weight=2.0, fl_any_weight=2.0)

    assert adjusted.tolist() == pytest.approx([10.0, 9.6])


def test_add_source_comparison_reports_source_teacher_rank_and_regret():
    row = {
        "candidates": [
            {"action_key": "model-best", "source_score": 1.0},
            {"action_key": "source-best", "source_score": 3.0},
            {"action_key": "other", "source_score": 2.0},
        ]
    }

    add_source_comparison(row)

    comparison = row["source_comparison"]
    assert comparison["source_top1_model_rank"] == 2
    assert comparison["model_top1_source_regret"] == pytest.approx(2.0)
    assert not comparison["topk_recall"]["1"]
    assert comparison["topk_recall"]["3"]
