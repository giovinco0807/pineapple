from __future__ import annotations

from ofc_regular.select_hu_turn0_replay_refinement import select_refinement_targets


def _row(index: int, *, label: str, margin: float, source: str, seat: str = "first") -> dict:
    return {
        "target_id": f"{index:064x}",
        "seat": seat,
        "safe_override_label": label,
        "predicted_margin": margin,
        "source_config_id": source,
        "candidate_delta_lcb196": float(index),
        "future_samples": 32,
    }


def test_refinement_selector_unions_positive_strict_negative_and_top_gray() -> None:
    rows = [
        _row(1, label="positive", margin=0.5, source="broad"),
        _row(2, label="gray", margin=3.0, source="p0_strict_f30"),
        _row(3, label="negative", margin=2.5, source="broad"),
        _row(4, label="gray", margin=1.0, source="broad"),
        _row(5, label="gray", margin=1.0, source="broad"),
        _row(6, label="negative", margin=1.0, source="broad"),
        _row(7, label="positive", margin=3.0, source="p0_strict_f30", seat="second"),
    ]

    selected, summary = select_refinement_targets(rows, gray_top_n=1)

    assert {row["target_id"] for row in selected} == {
        f"{index:064x}" for index in (1, 2, 3, 5)
    }
    assert summary["selected_rows"] == 4
    assert summary["duplicate_target_count"] == 0
    assert summary["reason_counts"]["all_mc32_positive"] == 1
    assert summary["reason_counts"]["all_strict_source"] == 1
    assert summary["reason_counts"]["high_margin_hard_negative"] == 1
    assert summary["reason_counts"]["top_gray_by_lcb196"] == 1
