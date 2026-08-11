from __future__ import annotations

import pytest

from ofc_regular.analyze_hu_turn1_safe_threshold_sweep import summarize_thresholds


def test_nested_safe_threshold_sweep_filters_actual_fires() -> None:
    rows = [
        {"safe_selector_score": 0.35, "realized_candidate_seat_delta": -2.0},
        {"safe_selector_score": 0.65, "realized_candidate_seat_delta": 3.0},
        {"safe_selector_score": 0.80, "realized_candidate_seat_delta": 1.0},
    ]

    summaries = summarize_thresholds(rows, thresholds=[0.3, 0.6], parent_decisions=100)

    assert summaries[0]["fires"] == 3
    assert summaries[0]["realized_per_fire_delta_mean"] == pytest.approx(2.0 / 3.0)
    assert summaries[1]["fires"] == 2
    assert summaries[1]["realized_per_fire_delta_mean"] == pytest.approx(2.0)
    assert summaries[1]["estimated_ev_per_decision"] == pytest.approx(0.04)
    assert summaries[1]["max_loss"] == 0.0


def test_nested_safe_threshold_sweep_requires_positive_parent_count() -> None:
    with pytest.raises(ValueError, match="parent_decisions"):
        summarize_thresholds([], thresholds=[0.3], parent_decisions=0)
