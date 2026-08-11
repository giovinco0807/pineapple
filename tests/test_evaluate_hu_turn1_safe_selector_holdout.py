from __future__ import annotations

import pytest

from ofc_regular.evaluate_hu_turn1_safe_selector_holdout import summarize_holdout


def test_summarize_holdout_uses_only_locked_threshold_fires() -> None:
    summary = summarize_holdout(
        [0.2, 0.3, 0.7, 0.9],
        [-5.0, 0.5, 2.0, -1.0],
        threshold=0.3,
        accept_delta=1.0,
        parent_runtime_fire_rate=0.025,
    )

    assert summary["rows"] == 4
    assert summary["fires"] == 3
    assert summary["selector_fire_rate_within_runtime_fires"] == pytest.approx(0.75)
    assert summary["mean_high_mc_delta"] == pytest.approx(0.5)
    assert summary["accept_positive_count"] == 1
    assert summary["gray_positive_count"] == 1
    assert summary["false_positive_count"] == 1
    assert summary["max_loss"] == pytest.approx(1.0)
    assert summary["estimated_runtime_fire_rate"] == pytest.approx(0.01875)
    assert summary["estimated_ev_per_decision"] == pytest.approx(0.009375)


def test_summarize_holdout_rejects_mismatched_or_nonfinite_inputs() -> None:
    with pytest.raises(ValueError, match="same shape"):
        summarize_holdout([0.5], [1.0, 2.0], threshold=0.3)
    with pytest.raises(ValueError, match="finite"):
        summarize_holdout([float("nan")], [1.0], threshold=0.3)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        summarize_holdout([0.5], [1.0], threshold=0.3, parent_runtime_fire_rate=1.1)
