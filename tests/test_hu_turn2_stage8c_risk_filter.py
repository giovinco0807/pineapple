import csv
import json

from ofc_regular.analyze_hu_turn2_stage8c_risk_filter import (
    RiskGuard,
    analyze,
    guard_passes,
    main,
    metric_row,
)


def _row(**overrides):
    row = {
        "config_id": "cfg",
        "override_fired": True,
        "realized_delta_valid": True,
        "realized_candidate_seat_delta": 3.0,
        "predicted_delta": 1.0,
        "rerank_delta": 4.0,
        "rerank_delta_se": 1.0,
        "candidate_ev_rank": 2,
        "gate_probability": 0.9,
    }
    row.update(overrides)
    return row


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_guard_passes_uses_runtime_available_thresholds():
    row = _row(predicted_delta=1.0, rerank_delta=4.0, rerank_delta_se=2.0, candidate_ev_rank=3)

    assert guard_passes(row, RiskGuard(min_predicted_delta=1.0, min_confirm_z=2.0, max_candidate_rank=3))
    assert not guard_passes(row, RiskGuard(min_predicted_delta=1.1))
    assert not guard_passes(row, RiskGuard(min_confirm_z=2.1))
    assert not guard_passes(row, RiskGuard(max_candidate_rank=2))
    assert not guard_passes(row, RiskGuard(max_confirm_se=1.5))


def test_guard_prefers_confirm_delta_over_stage_a_rerank_delta():
    row = _row(rerank_delta=9.0, rerank_delta_se=1.0, confirm_delta=1.0, confirm_delta_se=1.0)

    assert guard_passes(row, RiskGuard(min_confirm_delta=1.0, min_confirm_z=1.0))
    assert not guard_passes(row, RiskGuard(min_confirm_delta=2.0))
    assert not guard_passes(row, RiskGuard(min_confirm_z=2.0))


def test_metric_row_scores_realized_deltas_not_confirm_deltas():
    rows = [
        _row(realized_candidate_seat_delta=5.0, rerank_delta=99.0),
        _row(realized_candidate_seat_delta=-2.0, rerank_delta=99.0, predicted_delta=0.0),
        _row(
            override_fired=False,
            realized_candidate_seat_delta=0.0,
            no_override_reason="topk_empty",
        ),
    ]
    fired = rows[:2]

    metrics = metric_row("cfg", rows, fired, RiskGuard(min_predicted_delta=0.5))

    assert metrics["decision_count"] == 3
    assert metrics["input_fires"] == 2
    assert metrics["kept_fires"] == 1
    assert metrics["blocked_loss_count"] == 1
    assert metrics["blocked_gain_count"] == 0
    assert metrics["per_fire_delta_mean"] == 5.0
    assert metrics["ev_per_hand_after_guard"] == 5.0 / 3.0


def test_analyze_keeps_base_guard_and_filters_by_min_kept():
    rows = [
        _row(realized_candidate_seat_delta=2.0, predicted_delta=2.0),
        _row(realized_candidate_seat_delta=4.0, predicted_delta=2.0),
        _row(realized_candidate_seat_delta=-3.0, predicted_delta=0.0),
    ]

    grid_rows, top_rows = analyze(rows, min_kept=2, top_n=5)

    base_rows = [row for row in grid_rows if row["guard_id"] == "no_extra_guard"]
    assert len(base_rows) == 1
    assert base_rows[0]["kept_fires"] == 3
    assert all(row["kept_fires"] >= 2 for row in top_rows)


def test_cli_writes_grid_top_summary_and_manifest(tmp_path, monkeypatch):
    decision_log = tmp_path / "runtime_decisions.jsonl"
    output_dir = tmp_path / "risk"
    _write_jsonl(
        decision_log,
        [
            _row(realized_candidate_seat_delta=3.0, predicted_delta=2.0),
            _row(realized_candidate_seat_delta=-1.0, predicted_delta=0.0),
            _row(override_fired=False, realized_candidate_seat_delta=0.0),
        ],
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--decision-log",
            str(decision_log),
            "--output-dir",
            str(output_dir),
            "--min-kept",
            "1",
            "--top-n",
            "3",
        ],
    )

    main()

    assert (output_dir / "risk_filter_summary.md").exists()
    assert (output_dir / "risk_filter_manifest.json").exists()
    assert _read_csv(output_dir / "risk_filter_grid.csv")
    assert len(_read_csv(output_dir / "risk_filter_top.csv")) <= 3
