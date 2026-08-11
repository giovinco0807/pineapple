import csv

import pytest

from ofc_regular.compare_hu_turn2_stage8c_prediction_rankers import (
    evaluate_rankers,
    joined_rows,
    ranker_role,
    recommendation_rows,
    write_summary,
)


def _write_csv(path, rows):
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _prediction_row(**overrides):
    row = {
        "source_log": "log-a",
        "hand_seed": "1001",
        "candidate_index": "7",
        "baseline_index": "3",
        "recommended_training_use": "topk_confirm_realized_positive",
        "split": "test",
        "seat": "first",
        "realized_delta": "5.0",
        "realized_loss": "0.0",
        "risk_probability": "0.90",
        "predicted_delta": "1.0",
        "confirm_delta": "2.0",
        "confirm_delta_se": "0.2",
        "candidate_ev_rank": "1",
    }
    row.update(overrides)
    return row


def test_joined_rows_uses_action_identity_and_excludes_topk_empty(tmp_path):
    fire_path = tmp_path / "fire.csv"
    delta_path = tmp_path / "delta.csv"
    rows = [
        _prediction_row(hand_seed="1001", candidate_index="7", baseline_index="3", realized_delta="5.0", split=""),
        _prediction_row(hand_seed="1002", candidate_index="8", baseline_index="3", realized_delta="-2.0", split=""),
        _prediction_row(
            hand_seed="1003",
            candidate_index="9",
            baseline_index="3",
            recommended_training_use="topk_confirm_topk_empty",
            realized_delta="10.0",
        ),
    ]
    _write_csv(fire_path, rows)
    _write_csv(
        delta_path,
        [
            {
                **{key: row[key] for key in ("source_log", "hand_seed", "candidate_index", "baseline_index", "recommended_training_use")},
                "split": "test",
                "delta_prediction": value,
            }
            for row, value in zip(rows, ["0.1", "4.0", "99.0"])
        ],
    )

    joined, manifest = joined_rows(
        fire_path,
        [f"policy={delta_path}"],
        exclude_recommended_use={"topk_confirm_topk_empty"},
    )

    assert len(joined) == 2
    assert manifest["fire_rows_after_filter"] == 2
    assert manifest["joined_delta_counts"] == {"policy": 2}
    assert {row["hand_seed"] for row in joined} == {"1001", "1002"}
    assert {row["split"] for row in joined} == {"test"}
    assert joined[0]["policy_delta_prediction"] == pytest.approx(0.1)


def test_evaluate_rankers_reports_realized_delta_topk_not_gate_mean():
    rows = [
        {
            "split": "test",
            "realized_delta": 5.0,
            "risk_probability": 0.90,
            "predicted_delta": 1.0,
            "confirm_delta": 1.0,
            "policy_delta_prediction": 0.1,
        },
        {
            "split": "test",
            "realized_delta": -2.0,
            "risk_probability": 0.80,
            "predicted_delta": 4.0,
            "confirm_delta": 2.0,
            "policy_delta_prediction": 4.0,
        },
    ]

    metrics, top_rows = evaluate_rankers(rows, [1, 2])
    by_ranker_top1 = {
        row["ranker"]: row
        for row in metrics
        if row["split"] == "test" and row["topk"] == 1
    }

    assert by_ranker_top1["fire_probability"]["selected_delta_mean"] == pytest.approx(5.0)
    assert by_ranker_top1["policy"]["selected_delta_mean"] == pytest.approx(-2.0)
    assert by_ranker_top1["confirm_delta"]["ranker_role"] == "replay_triage_only"
    assert by_ranker_top1["confirm_delta"]["runtime_eligible"] is False
    assert by_ranker_top1["fire_probability"]["ranker_role"] == "deployable_ranker_input"
    assert by_ranker_top1["fire_probability"]["runtime_eligible"] is True
    assert any(row["ranker"] == "fire_probability" for row in top_rows)


def test_ranker_role_separates_runtime_inputs_from_replay_and_diagnostics():
    assert ranker_role("fire_probability")["runtime_eligible"] is True
    assert ranker_role("predicted_delta")["runtime_eligible"] is True

    confirm = ranker_role("confirm_delta")
    assert confirm["ranker_role"] == "replay_triage_only"
    assert confirm["runtime_eligible"] is False

    combo = ranker_role("fire_probability+predicted_delta")
    assert combo["ranker_role"] == "diagnostic_combo"
    assert combo["runtime_eligible"] is False

    observed = ranker_role("observed_delta")
    assert observed["ranker_role"] == "diagnostic_only"
    assert observed["runtime_eligible"] is False


def test_recommendation_rows_do_not_promote_confirm_delta_as_runtime_candidate():
    metrics = [
        {
            "split": "test",
            "ranker": "confirm_delta",
            "topk": 5,
            "eligible_rows": 100,
            "selected_delta_mean": 10.0,
            "selected_delta_sum": 50.0,
            "selected_positive_rate": 1.0,
            "selected_negative_count": 0,
            "selected_max_loss": 0.0,
            **ranker_role("confirm_delta"),
        },
        {
            "split": "test",
            "ranker": "fire_probability",
            "topk": 5,
            "eligible_rows": 100,
            "selected_delta_mean": 1.0,
            "selected_delta_sum": 5.0,
            "selected_positive_rate": 0.6,
            "selected_negative_count": 1,
            "selected_max_loss": 2.0,
            **ranker_role("fire_probability"),
        },
    ]

    rows = {row["ranker"]: row for row in recommendation_rows(metrics)}

    assert rows["confirm_delta"]["recommendation"] == "replay_target_selection_only"
    assert rows["confirm_delta"]["runtime_eligible"] is False
    assert rows["fire_probability"]["recommendation"] == "candidate_generator_replay_triage_go"
    assert rows["fire_probability"]["runtime_eligible"] is True


def test_summary_keeps_ranker_comparison_out_of_runtime_approval(tmp_path):
    output = tmp_path / "summary.md"

    write_summary(
        output,
        [
            {
                "split": "test",
                "ranker": "fire_probability",
                "eligible_rows": 2,
                "topk": 3,
                "selected_delta_mean": 1.5,
                "selected_delta_sum": 3.0,
                "selected_positive_rate": 0.5,
                "selected_max_loss": 2.0,
                **ranker_role("fire_probability"),
            }
        ],
        {"fire_rows_after_filter": 2, "joined_delta_counts": {"policy": 2}},
    )

    text = output.read_text(encoding="utf-8")
    assert "replay-triage analysis only" in text
    assert "runtime eligible" in text
    assert "candidate_generator_replay_triage_go" in text
    assert "runtime gate: `No-Go`" in text
    assert "50k teacher: `No-Go`" in text
