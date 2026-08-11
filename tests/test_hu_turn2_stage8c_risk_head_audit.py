import csv

from ofc_regular.analyze_hu_turn2_stage8c_risk_head import (
    audit_decision,
    fresh_heldout_plan_rows,
    group_breakdown_rows,
    infer_target_mode,
    parse_thresholds,
    raw_score_baseline_rows,
    raw_score_runtime_candidate_rows,
    raw_score_threshold_rows,
    raw_score_value,
    run_audit,
    runtime_group_candidate_rows,
    runtime_group_candidate_decision,
    runtime_gate_expression,
    source_family,
    source_run,
    source_seed,
    split_metric_rows,
    threshold_group_metric_rows,
    threshold_metric_rows,
    top_error_rows,
)


def _prediction(**overrides):
    row = {
        "row_index": "0",
        "source_log": "outputs/gcp_runs/regular-hu-t2-stage8c-risk-expand-20260614/results/shard/runtime_decisions.jsonl",
        "config_id": "cfg",
        "hand_seed": "1",
        "seat": "first",
        "seat_swap": "ab",
        "recommended_training_use": "whole_game_non_loss_control",
        "label": "0",
        "realized_delta": "0.0",
        "realized_loss": "0.0",
        "local_replay_bucket": "local_positive_lcb",
        "local_replay_label": "positive",
        "local_replay_delta": "2.0",
        "confirm_delta": "1.0",
        "predicted_delta": "0.5",
        "gate_probability": "0.1",
        "candidate_ev_rank": "2",
        "split": "test",
        "risk_probability": "0.2",
    }
    row.update(overrides)
    return row


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_parse_thresholds_sorts_and_dedupes():
    assert parse_thresholds("0.5, 0.4,0.5") == [0.4, 0.5]


def test_threshold_metrics_count_precision_recall_and_loss_capture():
    rows = [
        _prediction(label="1", recommended_training_use="whole_game_risk_only", risk_probability="0.9", realized_delta="-6"),
        _prediction(label="0", risk_probability="0.8", realized_delta="4"),
        _prediction(label="1", recommended_training_use="whole_game_risk_only", risk_probability="0.3", realized_delta="-2"),
        _prediction(label="0", risk_probability="0.1", realized_delta="0"),
    ]

    metrics = [row for row in threshold_metric_rows(rows, [0.5]) if row["split"] == "all"][0]

    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["selected_realized_delta_sum"] == -2.0
    assert metrics["selected_realized_loss_sum"] == 6.0
    assert metrics["selected_loss_capture_rate"] == 0.75
    assert metrics["selected_veto_prevented_loss_sum"] == 6.0
    assert metrics["selected_veto_forfeited_gain_sum"] == 4.0
    assert metrics["selected_veto_net_gain_sum"] == 2.0
    assert metrics["selected_veto_net_gain_mean"] == 1.0
    assert metrics["selected_veto_net_gain_per_row"] == 0.5


def test_raw_score_baselines_evaluate_runtime_fields_in_both_directions():
    rows = [
        _prediction(label="1", recommended_training_use="whole_game_risk_only", predicted_delta="-2.0", confirm_delta="1.0", confirm_delta_se="0.5"),
        _prediction(label="0", predicted_delta="2.0", confirm_delta="0.5", confirm_delta_se="0.5"),
    ]

    assert raw_score_value(rows[0], "negative_predicted_delta") == 2.0
    assert raw_score_value(rows[0], "confirm_delta_z") == 2.0

    metrics = {
        (row["split"], row["score_name"]): row
        for row in raw_score_baseline_rows(rows, score_names=("predicted_delta", "negative_predicted_delta"))
    }

    assert metrics[("all", "predicted_delta")]["roc_auc"] == 0.0
    assert metrics[("all", "negative_predicted_delta")]["roc_auc"] == 1.0
    confirm_metrics = raw_score_baseline_rows(rows, score_names=("confirm_delta",))[0]
    assert confirm_metrics["score_metric_role"] == "runtime_field_diagnostic_only"
    assert confirm_metrics["primary_metric_source"] == "realized_veto_utility_on_holdout_labels"
    assert confirm_metrics["performance_claim_allowed_from_score_mean"] is False


def test_raw_score_threshold_rows_use_train_quantiles_and_veto_utility():
    rows = [
        _prediction(split="train", label="0", confirm_delta="1.0", realized_delta="2"),
        _prediction(split="train", label="1", recommended_training_use="whole_game_risk_only", confirm_delta="3.0", realized_delta="-6"),
        _prediction(split="val", label="1", recommended_training_use="whole_game_risk_only", confirm_delta="3.5", realized_delta="-5"),
        _prediction(split="val", label="0", confirm_delta="0.5", realized_delta="4"),
        _prediction(split="test", label="1", recommended_training_use="whole_game_risk_only", confirm_delta="3.2", realized_delta="-7"),
        _prediction(split="test", label="0", confirm_delta="0.2", realized_delta="1"),
    ]

    sweep = raw_score_threshold_rows(rows, score_names=("confirm_delta",), quantiles=(0.5,))
    by_split = {row["split"]: row for row in sweep if row["score_name"] == "confirm_delta"}

    assert by_split["all"]["threshold_source"] == "train_quantile"
    assert by_split["all"]["score_metric_role"] == "runtime_field_diagnostic_only"
    assert by_split["all"]["primary_metric_source"] == "realized_veto_utility_on_holdout_labels"
    assert by_split["all"]["performance_claim_allowed_from_score_mean"] is False
    assert by_split["all"]["threshold"] == 2.0
    assert by_split["val"]["selected"] == 1
    assert by_split["val"]["tp"] == 1
    assert by_split["val"]["selected_veto_net_gain_sum"] == 5.0
    assert by_split["test"]["selected"] == 1
    assert by_split["test"]["selected_veto_net_gain_sum"] == 7.0

    candidates = raw_score_runtime_candidate_rows(
        sweep,
        min_selected_per_split=1,
        target_selected_per_split=10,
        prediction_rows=rows,
    )

    assert candidates[0]["score_name"] == "confirm_delta"
    assert candidates[0]["score_metric_role"] == "runtime_field_diagnostic_only"
    assert candidates[0]["primary_metric_source"] == "realized_veto_utility_on_holdout_labels"
    assert candidates[0]["performance_claim_allowed_from_score_mean"] is False
    assert candidates[0]["stable_runtime_candidate"] == 1
    assert candidates[0]["val_selected_source_seed_count"] == 1
    assert candidates[0]["test_selected_veto_gain_mean"] == 7.0


def test_group_breakdown_adds_source_family_and_rank_bucket():
    rows = [
        _prediction(source_log="outputs/gcp_runs/regular-hu-t2-stage8c-risk-fill-20260614/results/x.jsonl", candidate_ev_rank="1"),
        _prediction(source_log="outputs/gcp_runs/regular-hu-t2-stage8c-risk-second-20260614/results/x.jsonl", candidate_ev_rank="7"),
        _prediction(
            source_log=(
                "outputs/gcp_runs/regular-hu-t2-stage8c-position-specific-scd1-riskplus2-20260614/"
                "results/shard_001_seed2026070902/runtime_decisions.jsonl"
            ),
            candidate_ev_rank="2",
        ),
    ]

    groups = group_breakdown_rows(rows)
    source_values = {row["group_value"] for row in groups if row["group_field"] == "source_family"}
    source_run_values = {row["group_value"] for row in groups if row["group_field"] == "source_run"}
    source_seed_values = {row["group_value"] for row in groups if row["group_field"] == "source_seed"}
    rank_values = {row["group_value"] for row in groups if row["group_field"] == "candidate_rank_bucket"}

    assert {"risk_fill", "risk_second", "scd1_riskplus2"} <= source_values
    assert "regular-hu-t2-stage8c-position-specific-scd1-riskplus2-20260614" in source_run_values
    assert "2026070902" in source_seed_values
    assert {"rank_1", "rank_2_3", "rank_6_plus"} <= rank_values


def test_threshold_group_metrics_include_veto_utility_by_seat():
    rows = [
        _prediction(seat="first", label="1", recommended_training_use="whole_game_risk_only", risk_probability="0.9", realized_delta="-6"),
        _prediction(seat="first", label="0", risk_probability="0.8", realized_delta="4"),
        _prediction(seat="second", label="1", recommended_training_use="whole_game_risk_only", risk_probability="0.2", realized_delta="-2"),
        _prediction(seat="second", label="0", risk_probability="0.1", realized_delta="3"),
    ]

    metrics = {
        (row["split"], row["group_field"], row["group_value"]): row
        for row in threshold_group_metric_rows(rows, [0.5])
        if row["group_field"] == "seat"
    }

    first = metrics[("all", "seat", "first")]
    second = metrics[("all", "seat", "second")]
    assert first["runtime_group_available"] == 1
    assert first["selected"] == 2
    assert first["selected_veto_net_gain_sum"] == 2.0
    assert second["selected"] == 0
    assert second["selected_veto_net_gain_sum"] == 0.0


def test_runtime_group_candidates_require_both_holdout_splits_and_min_count():
    grouped_rows = [
        {
            "split": "val",
            "group_field": "candidate_rank_bucket",
            "group_value": "rank_4_5",
            "threshold": 0.35,
            "runtime_group_available": 1,
            "selected": 6,
            "precision": 0.33,
            "recall": 0.50,
            "selected_veto_net_gain_per_row": 0.53,
            "rows": 38,
        },
        {
            "split": "test",
            "group_field": "candidate_rank_bucket",
            "group_value": "rank_4_5",
            "threshold": 0.35,
            "runtime_group_available": 1,
            "selected": 3,
            "precision": 0.33,
            "recall": 0.17,
            "selected_veto_net_gain_per_row": 0.16,
            "rows": 32,
        },
        {
            "split": "val",
            "group_field": "seat",
            "group_value": "first",
            "threshold": 0.40,
            "runtime_group_available": 1,
            "selected": 12,
            "precision": 0.50,
            "recall": 0.40,
            "selected_veto_net_gain_per_row": 0.20,
            "rows": 100,
        },
        {
            "split": "test",
            "group_field": "seat",
            "group_value": "first",
            "threshold": 0.40,
            "runtime_group_available": 1,
            "selected": 11,
            "precision": 0.45,
            "recall": 0.35,
            "selected_veto_net_gain_per_row": 0.10,
            "rows": 100,
        },
        {
            "split": "val",
            "group_field": "source_seed",
            "group_value": "2026071509",
            "threshold": 0.35,
            "runtime_group_available": 0,
            "selected": 20,
            "precision": 0.90,
            "recall": 0.70,
            "selected_veto_net_gain_per_row": 1.0,
            "rows": 50,
        },
        {
            "split": "test",
            "group_field": "source_seed",
            "group_value": "2026071509",
            "threshold": 0.35,
            "runtime_group_available": 0,
            "selected": 20,
            "precision": 0.90,
            "recall": 0.70,
            "selected_veto_net_gain_per_row": 1.0,
            "rows": 50,
        },
    ]

    prediction_rows = [
        _prediction(
            split="val",
            candidate_ev_rank="4",
            risk_probability="0.7",
            realized_delta="-6",
            source_log="outputs/gcp_runs/run/results/shard_seed2026071501/runtime_decisions.jsonl",
        ),
        _prediction(
            split="val",
            candidate_ev_rank="5",
            risk_probability="0.6",
            realized_delta="4",
            source_log="outputs/gcp_runs/run/results/shard_seed2026071502/runtime_decisions.jsonl",
        ),
        _prediction(
            split="test",
            candidate_ev_rank="4",
            risk_probability="0.6",
            realized_delta="-5",
            source_log="outputs/gcp_runs/run/results/shard_seed2026071503/runtime_decisions.jsonl",
        ),
    ]

    candidates = runtime_group_candidate_rows(
        grouped_rows,
        min_selected_per_split=10,
        target_selected_per_split=50,
        prediction_rows=prediction_rows,
    )
    by_key = {(row["group_field"], row["group_value"], row["threshold"]): row for row in candidates}

    stable = by_key[("seat", "first", 0.40)]
    near_miss = by_key[("candidate_rank_bucket", "rank_4_5", 0.35)]
    assert stable["stable_runtime_candidate"] == 1
    assert stable["near_miss_positive_but_underpowered"] == 0
    assert near_miss["stable_runtime_candidate"] == 0
    assert near_miss["near_miss_positive_but_underpowered"] == 1
    assert near_miss["additional_selected_needed_for_min"] == 7
    assert near_miss["additional_selected_needed_for_target"] == 47
    assert near_miss["estimated_rows_per_split_for_target_selected"] == 534
    assert near_miss["val_selected_source_seed_count"] == 2
    assert near_miss["test_selected_source_seed_count"] == 1
    assert near_miss["val_selected_veto_gain_mean"] == 1.0
    assert near_miss["val_selected_veto_gain_se"] > 0.0
    assert near_miss["val_selected_veto_gain_lcb95"] < 1.0
    assert near_miss["test_selected_veto_gain_mean"] == 5.0
    assert near_miss["test_selected_veto_gain_se"] == 0.0
    assert near_miss["test_selected_veto_gain_lcb95"] == 5.0
    assert ("source_seed", "2026071509", 0.35) not in by_key

    decision = runtime_group_candidate_decision(candidates)
    assert decision["stable_runtime_candidate_count"] == 1
    assert decision["near_miss_positive_but_underpowered_count"] == 1
    assert decision["runtime_group_decision"] == "Candidate-Ready"

    plan_rows = fresh_heldout_plan_rows(candidates)
    near_miss_plan = [
        row
        for row in plan_rows
        if row["group_field"] == "candidate_rank_bucket" and row["group_value"] == "rank_4_5"
    ][0]
    assert near_miss_plan["plan_status"] == "near_miss_needs_fresh_heldout"
    assert near_miss_plan["runtime_gate_expression"] == "risk_probability >= 0.35 and 4 <= candidate_ev_rank <= 5"
    assert near_miss_plan["estimated_rows_per_split"] == 534
    assert near_miss_plan["production_p2_fixed"] == "No-Go"
    assert near_miss_plan["required_evaluation_metric"] == "fresh heldout realized veto utility, not training/cache LCB"


def test_runtime_gate_expression_uses_only_runtime_available_fields():
    assert (
        runtime_gate_expression({"group_field": "candidate_rank_bucket", "group_value": "rank_1", "threshold": 0.5})
        == "risk_probability >= 0.50 and candidate_ev_rank == 1"
    )
    assert (
        runtime_gate_expression({"group_field": "seat", "group_value": "second", "threshold": 0.4})
        == "risk_probability >= 0.40 and seat == 'second'"
    )


def test_source_helpers_extract_current_stage8c_run_family_and_seed():
    row = _prediction(
        source_log=(
            "outputs/gcp_runs/regular-hu-t2-stage8c-position-specific-scd1-500fires-20260614/"
            "results/shard_000_seed2026070301/runtime_decisions.jsonl"
        )
    )

    assert source_run(row) == "regular-hu-t2-stage8c-position-specific-scd1-500fires-20260614"
    assert source_family(row) == "scd1_500fires"
    assert source_seed(row) == "2026070301"


def test_top_error_rows_sort_false_positive_and_false_negative():
    rows = [
        _prediction(row_index="fp_low", label="0", risk_probability="0.6"),
        _prediction(row_index="fp_high", label="0", risk_probability="0.9"),
        _prediction(row_index="fn_low", label="1", recommended_training_use="whole_game_risk_only", risk_probability="0.1"),
        _prediction(row_index="fn_high", label="1", recommended_training_use="whole_game_risk_only", risk_probability="0.4"),
    ]

    false_positives, false_negatives = top_error_rows(rows, limit=1)

    assert false_positives[0]["row_index"] == "fp_high"
    assert false_negatives[0]["row_index"] == "fn_low"


def test_audit_decision_blocks_weak_holdout_signal():
    rows = [
        {"split": "train", "rows": 10, "positives": 5, "average_precision": 0.9, "roc_auc": 0.95},
        {"split": "val", "rows": 10, "positives": 5, "average_precision": 0.2, "roc_auc": 0.55},
        {"split": "test", "rows": 10, "positives": 5, "average_precision": 0.2, "roc_auc": 0.56},
    ]

    decision = audit_decision(rows)

    assert decision["runtime_integration_ready"] is False
    assert "val_auc_lt_0p65" in decision["blockers"]
    assert "test_auc_lt_0p65" in decision["blockers"]
    assert decision["runtime_integration_approval"] == "No-Go"
    assert decision["runtime_integration_ready_scope"] == "model_quality_screen_only"


def test_infer_target_mode_distinguishes_local_ev_from_whole_game_risk():
    local_rows = [
        _prediction(risk_target_group="local_ev_negative", label="1"),
        _prediction(risk_target_group="local_ev_positive_lcb_control", label="0"),
    ]
    whole_game_rows = [
        _prediction(risk_target_group="whole_game_risk_only", label="1"),
        _prediction(risk_target_group="whole_game_non_loss_control", label="0"),
    ]

    assert infer_target_mode(local_rows) == "local_ev_negative"
    assert infer_target_mode(whole_game_rows) == "whole_game_risk"


def test_run_audit_writes_artifacts(tmp_path):
    predictions = tmp_path / "predictions.csv"
    rows = [
        _prediction(row_index=str(index), split="train" if index < 4 else "test", label=str(index % 2), risk_probability=str(0.1 * index))
        for index in range(8)
    ]
    with predictions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    output_dir = tmp_path / "audit"
    manifest = run_audit(rows, thresholds=[0.5], deciles=2, top_errors=2, output_dir=output_dir, predictions_path=predictions)

    assert manifest["rows"] == 8
    assert manifest["runtime_integration_approval"] == "No-Go"
    assert manifest["raw_score_primary_metric_source"] == "realized_veto_utility_on_holdout_labels"
    assert manifest["raw_confirm_score_role"] == "runtime_field_diagnostic_only"
    assert manifest["raw_confirm_score_performance_claim_allowed"] is False
    assert manifest["decision"]["runtime_integration_ready_scope"] == "model_quality_screen_only"
    assert "runtime_group_candidate_decision" in manifest
    assert (output_dir / "risk_head_audit_summary.md").exists()
    summary = (output_dir / "risk_head_audit_summary.md").read_text(encoding="utf-8")
    assert "model-quality screen ready" in summary
    assert "Confirm-delta score means are `runtime_field_diagnostic_only`" in summary
    assert "runtime integration approval: `No-Go`" in summary
    assert "target mode:" in summary
    assert "Raw Runtime Score Veto Utility" in summary
    assert "Raw Runtime Score Candidate Check" in summary
    assert "Veto Utility Sweep" in summary
    assert "Runtime-Available Group Candidate Check" in summary
    assert "Top Diagnostic-Only Group Veto Utility" in summary
    assert _read_csv(output_dir / "risk_head_split_metrics.csv")
    assert _read_csv(output_dir / "risk_head_threshold_sweep.csv")
    assert _read_csv(output_dir / "risk_head_threshold_group_sweep.csv")
    assert (output_dir / "risk_head_runtime_group_candidates.csv").exists()
    assert (output_dir / "risk_head_fresh_heldout_plan.csv").exists()
    assert (output_dir / "risk_head_fresh_heldout_plan.md").exists()
    plan = (output_dir / "risk_head_fresh_heldout_plan.md").read_text(encoding="utf-8")
    assert "production / P2 fixed: `No-Go`" in plan
    assert _read_csv(output_dir / "risk_head_raw_score_baselines.csv")
    assert _read_csv(output_dir / "risk_head_raw_score_threshold_sweep.csv")
    assert (output_dir / "risk_head_raw_score_runtime_candidates.csv").exists()
