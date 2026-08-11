import csv
import json
from pathlib import Path

from ofc_regular.analyze_hu_turn1_topk_confirm import (
    CONFIRM_DELTA_PERFORMANCE_CLAIM_ALLOWED,
    summarize,
    write_csv,
    write_markdown,
)


def _row(
    *,
    fired: bool,
    valid: bool,
    delta: float = 0.0,
    reason: str = "",
    final_index: int = 1,
    baseline_index: int = 1,
) -> dict:
    row = {
        "config_id": "k2_mc1",
        "override_fired": fired,
        "realized_delta_valid": valid,
        "confirm_delta": 2.0 if fired else None,
        "confirm_delta_se": 0.5 if fired else None,
        "runtime_latency_ms": 10.0,
        "mc_rerank_latency_ms": 3.0,
        "confirm_mc_latency_ms": 4.0 if fired else 0.0,
        "no_override_reason": reason,
        "final_action_index": final_index,
        "baseline_action_index": baseline_index,
    }
    if valid:
        row["realized_candidate_seat_delta"] = delta
    return row


def test_summarize_uses_realized_fired_delta_not_confirm_delta():
    rows = [
        _row(fired=True, valid=True, delta=-1.0, reason="override_fired"),
        _row(fired=True, valid=True, delta=3.0, reason="override_fired"),
        _row(fired=True, valid=False, reason="override_fired"),
        _row(fired=False, valid=True, delta=0.0, reason="mc_best_is_baseline"),
    ]

    summary = summarize(rows)[0]

    assert summary["decision_count"] == 4
    assert summary["valid_decision_count"] == 3
    assert summary["override_count"] == 3
    assert summary["valid_override_count"] == 2
    assert summary["invalid_override_count"] == 1
    assert summary["realized_per_fire_delta_mean"] == 1.0
    assert summary["confirm_delta_mean_on_valid_fired"] == 2.0
    assert summary["confirm_delta_performance_claim_allowed"] is False
    assert CONFIRM_DELTA_PERFORMANCE_CLAIM_ALLOWED is False
    assert summary["non_fired_nonzero_count"] == 0
    assert summary["non_fired_counterfactual_nonzero_count"] == 0
    assert summary["non_fired_final_mismatch_count"] == 0
    assert summary["loss_count"] == 1


def test_non_fired_counterfactual_delta_is_not_final_mismatch():
    rows = [
        _row(
            fired=False,
            valid=True,
            delta=-8.0,
            reason="below_confirm_se",
            final_index=25,
            baseline_index=25,
        ),
        _row(
            fired=False,
            valid=True,
            delta=0.0,
            reason="mc_best_is_baseline",
            final_index=8,
            baseline_index=3,
        ),
    ]

    summary = summarize(rows)[0]

    assert summary["non_fired_nonzero_count"] == 1
    assert summary["non_fired_counterfactual_nonzero_count"] == 1
    assert summary["non_fired_final_matches_baseline_count"] == 1
    assert summary["non_fired_final_mismatch_count"] == 1


def test_write_outputs_summary_artifacts(tmp_path: Path):
    rows = summarize([
        _row(fired=True, valid=True, delta=2.0, reason="override_fired"),
        _row(fired=False, valid=True, delta=0.0, reason="mc_best_is_baseline"),
    ])
    csv_path = tmp_path / "summary.csv"
    md_path = tmp_path / "summary.md"

    write_csv(csv_path, rows)
    write_markdown(md_path, rows, input_paths=[tmp_path / "decisions.jsonl"])

    parsed = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    assert parsed[0]["config_id"] == "k2_mc1"
    assert parsed[0]["confirm_delta_performance_claim_allowed"] == "False"
    text = md_path.read_text(encoding="utf-8")
    assert "confirm delta performance claim allowed" in text
    assert "realized_seat_swap_counterfactual" in text
