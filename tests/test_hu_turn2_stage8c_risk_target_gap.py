import csv
import json

from ofc_regular.analyze_hu_turn2_stage8c_risk_target_gap import (
    DOWNSTREAM_FIELDS,
    diagnostic_flags,
    field_coverage_by_group_rows,
    field_coverage_rows,
    main,
    matrix_rows,
    metric_rows,
    normalized_row,
    trajectory_component_rows,
    trajectory_feature_auc_rows,
)


def _target(**overrides):
    row = {
        "schema": "hu_turn2_stage8b_counterfactual_loss_target_v1",
        "source_log": "outputs/gcp_runs/regular-hu-t2-stage8c-risk-expand/runtime_decisions.jsonl",
        "config_id": "cfg",
        "hand_seed": 1,
        "hand_id": 1,
        "seat": "first",
        "seat_swap": "ab",
        "realized_delta": -8.0,
        "candidate_index": 3,
        "baseline_index": 1,
        "predicted_delta": 1.5,
        "gate_probability": 0.2,
        "candidate_ev_rank": 2,
        "confirm_delta": 3.0,
        "confirm_delta_se": 0.75,
        "local_replay_bucket": "local_positive_lcb",
        "local_replay_label": "positive",
        "local_replay_delta": 2.5,
        "local_replay_delta_se": 0.5,
        "local_replay_lcb196": 1.52,
        "recommended_training_use": "whole_game_risk_only",
        "hero_board": {"top": ["Qh"], "middle": [], "bottom": []},
        "opponent_board": {"top": [], "middle": [], "bottom": []},
        "dead_cards": ["2c"],
        "cards_to_place": ["Ah", "Ks", "3d"],
        "baseline_action": {"placements": [], "discards": []},
        "candidate_action": {"placements": [], "discards": []},
        "local_replay_status": "ok",
        "local_replay_action_mapping_status": "ok",
        "terminal_score_vs_baseline": -8.0,
        "royalty_delta_vs_baseline": -3.0,
        "fl_delta_vs_baseline": -5.0,
        "line_score_delta_vs_baseline": 0.0,
        "scoop_delta_vs_baseline": 0.0,
        "foul_delta_vs_baseline": 0.0,
        "hero_royalty_vs_baseline": -3.0,
        "opponent_royalty_vs_baseline": 0.0,
        "hero_fl_value_vs_baseline": -5.0,
        "opponent_fl_value_vs_baseline": 0.0,
        "candidate_hero_foul": False,
        "baseline_hero_foul": False,
        "candidate_hero_fl_entry": False,
        "baseline_hero_fl_entry": True,
        "downstream_trajectory_complete": 1,
        "downstream_trajectory_present_fields": 16,
        "downstream_trajectory_total_fields": 16,
    }
    row.update(overrides)
    return row


def _complete_downstream_fields():
    return {
        "post_t2_candidate_board": {"top": ["Qh", "Ah"], "middle": ["Ks"], "bottom": []},
        "post_t2_baseline_board": {"top": ["Qh", "Ks"], "middle": ["Ah"], "bottom": []},
        "t3_decision_summary": {"candidate": {"override_fired": False}, "baseline": {"override_fired": False}},
        "final_board_hero": {
            "top": ["Qh", "Ah", "2c"],
            "middle": ["Ks", "3c", "4c", "5c", "6c"],
            "bottom": [],
        },
        "final_board_opponent": {
            "top": ["2d", "3d", "4d"],
            "middle": ["5d", "6d", "7d", "8d", "9d"],
            "bottom": [],
        },
        "hero_foul": False,
        "opponent_foul": False,
        "hero_fl_entry": False,
        "hero_fl_stay": False,
        "opponent_fl_entry": False,
        "hero_royalty": 0.0,
        "opponent_royalty": 0.0,
        "line_score_delta": 0.0,
        "scoop_delta": 0.0,
        "downstream_override_fired": False,
        "paired_future_delta_summary": {"count": 512, "mean": -8.0},
    }


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_gap_metrics_show_whole_game_risk_can_be_locally_positive():
    rows = [
        _target(hand_seed=1, recommended_training_use="whole_game_risk_only", local_replay_label="positive"),
        _target(
            hand_seed=2,
            recommended_training_use="whole_game_non_loss_control",
            local_replay_label="negative",
            local_replay_bucket="local_negative",
            local_replay_delta=-0.5,
            realized_delta=0.0,
        ),
        _target(
            hand_seed=3,
            recommended_training_use="local_ev_hard_negative",
            local_replay_label="negative",
            local_replay_bucket="local_negative",
            local_replay_delta=-1.0,
            realized_delta=-9.0,
        ),
    ]
    normalized = [normalized_row(row) for row in rows]

    metrics = metric_rows(normalized, rows, [{"source_path": "x", "rows": len(rows)}])
    values = {row["metric"]: row["value"] for row in metrics}
    flags = diagnostic_flags(metrics)
    matrix = matrix_rows(normalized)

    assert values["risk_only_local_positive"] == 1
    assert values["risk_only_local_positive_share"] == 1.0
    assert values["non_loss_controls_local_negative"] == 1
    assert "whole_game_risk_is_mostly_locally_positive" in flags
    assert "local_negative_can_still_be_whole_game_non_loss" in flags
    assert any(
        row["local_replay_label"] == "positive"
        and row["recommended_training_use"] == "whole_game_risk_only"
        and row["rows"] == 1
        for row in matrix
    )
    assert matrix[0]["primary_metric_source"] == "realized_whole_game_and_local_replay_delta"
    assert matrix[0]["confirm_delta_metric_role"] == "runtime_feature_diagnostic_only"
    assert matrix[0]["confirm_delta_performance_claim_allowed"] is False


def test_downstream_field_coverage_detects_missing_trajectory_fields():
    rows = [
        _target(),
        _target(
            hand_seed=2,
            post_t2_candidate_board={"top": ["Qh"]},
            hero_foul=False,
            paired_future_delta_summary={"count": 4, "mean": -1.0},
        ),
    ]

    coverage = field_coverage_rows(rows)
    by_field = {row["field"]: row for row in coverage}

    assert by_field["post_t2_candidate_board"]["present_rows"] == 1
    assert by_field["hero_foul"]["present_rows"] == 1
    assert by_field["paired_future_delta_summary"]["present_rows"] == 1
    assert by_field["final_board_hero"]["missing_rows"] == 2


def test_grouped_coverage_uses_collection_source_family():
    downstream_payload = {field: {"present": True} for field in DOWNSTREAM_FIELDS}
    rows = [
        _target(
            collection_source_path="outputs/evals/hu_turn2_stage8c_30seed_plus_local_risk_veto_rank2_loss_targets/topk_counterfactual_loss_targets_merged.jsonl",
            **downstream_payload,
        ),
        _target(
            hand_seed=2,
            collection_source_path="outputs/evals/hu_turn2_stage8c_c4_allfired_mc512_replay_gcp_received/loss_targets_from_pack/topk_counterfactual_loss_targets.jsonl",
            target_source_path="outputs/evals/hu_turn2_stage8c_c4_rank45_risk035_hard_negatives/topk_all_fired_deduped.jsonl",
        ),
    ]

    grouped = field_coverage_by_group_rows(rows)
    complete_by_source = {
        row["group_value"]: row
        for row in grouped
        if row["group_field"] == "source_family" and row["field"] == "all_downstream_fields"
    }

    assert complete_by_source["stage8c_30seed_local_risk_veto"]["present_rows"] == 1
    assert complete_by_source["stage8c_30seed_local_risk_veto"]["missing_rows"] == 0
    assert complete_by_source["c4_allfired_mc512_replay"]["present_rows"] == 0
    assert complete_by_source["c4_allfired_mc512_replay"]["missing_rows"] == 1


def test_trajectory_component_and_auc_rows_expose_downstream_deltas():
    rows = [
        _target(
            hand_seed=1,
            recommended_training_use="whole_game_risk_only",
            terminal_score_vs_baseline=-8.0,
            fl_delta_vs_baseline=-5.0,
            royalty_delta_vs_baseline=-3.0,
            candidate_hero_fl_entry=False,
            baseline_hero_fl_entry=True,
        ),
        _target(
            hand_seed=2,
            recommended_training_use="whole_game_non_loss_control",
            realized_delta=4.0,
            terminal_score_vs_baseline=4.0,
            fl_delta_vs_baseline=2.0,
            royalty_delta_vs_baseline=2.0,
            candidate_hero_fl_entry=True,
            baseline_hero_fl_entry=False,
        ),
        _target(
            hand_seed=3,
            recommended_training_use="whole_game_non_loss_control",
            realized_delta=2.0,
            terminal_score_vs_baseline=2.0,
            fl_delta_vs_baseline=0.0,
            royalty_delta_vs_baseline=2.0,
            candidate_hero_fl_entry=False,
            baseline_hero_fl_entry=False,
        ),
    ]
    normalized = [normalized_row(row) for row in rows]

    components = {row["field"]: row for row in trajectory_component_rows(normalized)}
    auc = {row["field"]: row for row in trajectory_feature_auc_rows(normalized)}

    assert components["fl_delta_vs_baseline"]["risk_mean"] == -5.0
    assert components["fl_delta_vs_baseline"]["control_mean"] == 1.0
    assert components["hero_fl_entry_changed"]["risk_positive_rows"] == 1
    assert auc["terminal_score_vs_baseline"]["best_direction"] == "low"
    assert auc["terminal_score_vs_baseline"]["best_auc"] == 1.0
    assert auc["confirm_delta"]["feature_metric_role"] == "runtime_feature_diagnostic_only"


def test_cli_writes_gap_artifacts(tmp_path, monkeypatch):
    collection_dir = tmp_path / "collection"
    _write_jsonl(
        collection_dir / "topk_counterfactual_loss_targets_merged.jsonl",
        [
            _target(hand_seed=1, **_complete_downstream_fields()),
            _target(
                hand_seed=2,
                recommended_training_use="whole_game_non_loss_control",
                local_replay_label="negative",
                local_replay_bucket="local_negative",
                local_replay_delta=-0.5,
                realized_delta=0.0,
                **_complete_downstream_fields(),
            ),
        ],
    )
    output_dir = tmp_path / "gap"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--collection-dir",
            str(collection_dir),
            "--output-dir",
            str(output_dir),
            "--top-n",
            "1",
        ],
    )

    main()

    assert (output_dir / "risk_target_gap_summary.md").exists()
    assert (output_dir / "local_vs_whole_game_matrix.csv").exists()
    assert (output_dir / "downstream_field_coverage.csv").exists()
    assert (output_dir / "downstream_field_coverage_by_group.csv").exists()
    assert (output_dir / "trajectory_component_breakdown.csv").exists()
    assert (output_dir / "trajectory_feature_auc.csv").exists()
    assert (output_dir / "risk_only_top_losses.jsonl").exists()
    assert (output_dir / "non_loss_high_local_ev_controls.jsonl").exists()
    metrics = _read_csv(output_dir / "risk_target_gap_metrics.csv")
    values = {row["metric"]: row["value"] for row in metrics}
    assert values["whole_game_risk_only"] == "1"
    assert values["whole_game_non_loss_control"] == "1"
    manifest = json.loads((output_dir / "risk_target_gap_manifest.json").read_text(encoding="utf-8"))
    assert manifest["runtime_risk_integration"] is False
    assert manifest["downstream_trajectory_complete_rows"] == 2
    assert manifest["downstream_trajectory_complete_share"] == 1.0
    assert manifest["downstream_fields_partial_missing"] == 0
    assert manifest["partial_downstream_trajectory_coverage"] is False
    assert manifest["primary_metric_source"] == "realized_whole_game_and_local_replay_delta"
    assert manifest["confirm_delta_metric_role"] == "runtime_feature_diagnostic_only"
    assert manifest["confirm_delta_performance_claim_allowed"] is False
    summary = (output_dir / "risk_target_gap_summary.md").read_text(encoding="utf-8")
    assert "`confirm_delta_mean` and `confirm_delta_z_mean` are `runtime_feature_diagnostic_only`" in summary


def test_cli_accepts_unmerged_loss_target_collection_dir(tmp_path, monkeypatch):
    collection_dir = tmp_path / "collection"
    _write_jsonl(
        collection_dir / "topk_counterfactual_loss_targets.jsonl",
        [_target(hand_seed=1)],
    )
    output_dir = tmp_path / "gap"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--collection-dir",
            str(collection_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    main()

    manifest = json.loads((output_dir / "risk_target_gap_manifest.json").read_text(encoding="utf-8"))
    assert manifest["rows"] == 1
    assert manifest["input_paths"] == [str(collection_dir / "topk_counterfactual_loss_targets.jsonl")]
