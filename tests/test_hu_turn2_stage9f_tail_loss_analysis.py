import json

from ofc_regular.analyze_hu_turn2_stage9f_tail_loss import (
    audit_rows,
    fired_decisions,
    guard_sweep_rows,
    loss_components,
    primary_loss_label,
    summarize,
)


def _decision(**overrides):
    row = {
        "config_id": "cfg",
        "override_fired": True,
        "realized_delta_valid": True,
        "realized_candidate_seat_delta": -12.0,
        "dead_cards": ["2c"],
        "visible_dead_cards": ["2c"],
        "hero_private_discards": ["2c"],
        "opponent_private_discards": ["3d"],
        "fl_delta_vs_baseline": -10.0,
        "foul_delta_vs_baseline": 0.0,
        "royalty_delta_vs_baseline": -2.0,
        "line_score_delta_vs_baseline": 0.0,
        "scoop_delta_vs_baseline": 0.0,
        "confirm_delta": 4.0,
        "confirm_delta_se": 3.0,
        "candidate_ev_rank": 2,
    }
    row.update(overrides)
    return row


def test_tail_loss_components_prioritize_lost_fl():
    row = _decision()

    assert "lost_fl_value" in loss_components(row)
    assert "royalty_regression" in loss_components(row)
    assert "high_confirm_se" in loss_components(row)
    assert primary_loss_label(row) == "lost_fl_value"


def test_tail_loss_audit_reads_fired_negative_rows(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [
        _decision(seed=1, hand_id=10),
        _decision(seed=2, hand_id=20, realized_candidate_seat_delta=5.0),
        _decision(seed=3, hand_id=30, override_fired=False),
    ]
    with (run_dir / "runtime_decisions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    losses = audit_rows([run_dir], {"cfg"})
    summary, breakdown = summarize(losses, [run_dir])

    assert len(losses) == 1
    assert losses[0]["loss"] == 12.0
    assert losses[0]["replay_ready"] is True
    assert summary[0]["loss_count"] == 1
    labels = {(row["breakdown_type"], row["label"]) for row in breakdown}
    assert ("primary", "lost_fl_value") in labels


def test_tail_loss_audit_reads_profile_topk_decision_logs(tmp_path):
    run_dir = tmp_path / "profile_run"
    run_dir.mkdir()
    rows = [
            _decision(
                runtime_profile="stage9f_cse1p5_firstseat",
                config_id="",
                seed=1,
                hand_id=10,
                seat="first",
            ),
    ]
    with (run_dir / "topk_decisions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    losses = audit_rows([run_dir], {"stage9f_cse1p5_firstseat"})
    summary, _breakdown = summarize(losses, [run_dir])

    assert len(losses) == 1
    assert losses[0]["config_id"] == "stage9f_cse1p5_firstseat"
    assert losses[0]["source_log"].endswith("topk_decisions.jsonl")
    assert summary[0]["config_id"] == "stage9f_cse1p5_firstseat"

    fired, decision_counts, first_counts = fired_decisions(
        [run_dir], {"stage9f_cse1p5_firstseat"}
    )
    assert fired[0]["config_id"] == "stage9f_cse1p5_firstseat"
    assert decision_counts["stage9f_cse1p5_firstseat"] == 1
    assert first_counts["stage9f_cse1p5_firstseat"] == 1


def test_guard_sweep_reports_all_and_first_seat_ev(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [
        _decision(seed=1, hand_id=10, seat="first", candidate_ev_rank=1, realized_candidate_seat_delta=4.0),
        _decision(seed=2, hand_id=20, seat="first", candidate_ev_rank=2, realized_candidate_seat_delta=-2.0),
        _decision(seed=3, hand_id=30, seat="second", override_fired=False),
    ]
    with (run_dir / "runtime_decisions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    fired, decision_counts, first_counts = fired_decisions([run_dir], {"cfg"})
    sweep = guard_sweep_rows(fired, decision_counts, first_counts)
    all_row = next(row for row in sweep if row["guard_id"] == "all")
    rank1_row = next(row for row in sweep if row["guard_id"] == "rank<=1")

    assert all_row["fired_count"] == 2
    assert all_row["estimated_ev_per_all_decision"] == (2 / 3) * 1.0
    assert all_row["estimated_ev_per_first_decision"] == (2 / 2) * 1.0
    assert rank1_row["fired_count"] == 1
    assert rank1_row["per_fire_delta_mean"] == 4.0
