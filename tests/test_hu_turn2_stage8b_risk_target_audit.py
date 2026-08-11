import csv
import json

from ofc_regular.analyze_hu_turn2_stage8b_risk_targets import (
    DOWNSTREAM_TRAJECTORY_FIELDS,
    downstream_complete,
    group_breakdown,
    main,
    missing_replay_fields,
    normalized_row,
    readiness,
    replay_ready,
)


def _target(**overrides):
    row = {
        "schema": "hu_turn2_stage8b_counterfactual_loss_target_v1",
        "source_log": "run/runtime_decisions.jsonl",
        "config_id": "cfg",
        "hand_seed": 1,
        "seat": "first",
        "seat_swap": "ab",
        "realized_delta": -6.0,
        "candidate_index": 3,
        "baseline_index": 1,
        "predicted_delta": 1.5,
        "gate_probability": 0.001,
        "candidate_ev_rank": 2,
        "confirm_delta": 3.0,
        "confirm_delta_se": 0.8,
        "local_replay_bucket": "local_positive_lcb",
        "recommended_training_use": "whole_game_risk_only",
        "use_for_local_ev_hard_negative": 0,
        "use_for_whole_game_risk_head": 1,
        "requires_local_replay": 0,
        "hero_board": {"top": ["Qh"], "middle": [], "bottom": []},
        "opponent_board": {"top": [], "middle": [], "bottom": []},
        "dead_cards": ["2c"],
        "cards_to_place": ["Ah", "Ks", "3d"],
        "baseline_action": {"placements": [], "discards": []},
        "candidate_action": {"placements": [], "discards": []},
        "local_replay_status": "ok",
        "local_replay_future_samples": 512,
        "local_replay_delta": 2.5,
        "local_replay_delta_se": 0.5,
        "local_replay_lcb196": 1.52,
        "local_replay_label": "positive",
        "local_replay_action_mapping_status": "ok",
        "downstream_trajectory_complete": 1,
        "downstream_trajectory_present_fields": 16,
        "downstream_trajectory_total_fields": 16,
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


def test_replay_ready_requires_replay_fields_and_ok_mapping():
    row = _target()

    assert missing_replay_fields(row) == []
    assert replay_ready(row)

    row.pop("dead_cards")
    assert missing_replay_fields(row) == ["dead_cards"]
    assert not replay_ready(row)


def test_readiness_keeps_whole_game_risk_separate_from_local_ev_labels():
    rows = [
        _target(hand_seed=1, recommended_training_use="whole_game_risk_only", use_for_whole_game_risk_head=1),
        _target(
            hand_seed=2,
            recommended_training_use="local_ev_hard_negative",
            use_for_whole_game_risk_head=0,
            use_for_local_ev_hard_negative=1,
            local_replay_bucket="local_negative",
            local_replay_delta=-0.2,
            local_replay_lcb196=-1.0,
        ),
    ]

    ready = readiness(rows, min_risk_head_rows=5, min_local_ev_hard_negatives=2)
    normalized = [normalized_row(row) for row in rows]

    assert ready["risk_head_training_ready"] is False
    assert "risk_only_rows_lt_min" in ready["blockers"]
    assert "missing_non_loss_control_rows" in ready["blockers"]
    assert "single_seat_only" in ready["blockers"]
    assert "local_ev_hard_negatives_lt_min" in ready["local_ev_hard_negative_blockers"]
    assert normalized[0]["recommended_training_use"] == "whole_game_risk_only"
    assert normalized[0]["use_for_local_ev_hard_negative"] == 0
    assert normalized[1]["recommended_training_use"] == "local_ev_hard_negative"


def test_readiness_accepts_explicit_non_loss_controls():
    rows = [
        _target(hand_seed=1, seat="first", recommended_training_use="whole_game_risk_only"),
        _target(
            hand_seed=2,
            seat="second",
            realized_delta=3.0,
            recommended_training_use="whole_game_non_loss_control",
            whole_game_risk_label=0,
            use_for_whole_game_risk_head=1,
        ),
        _target(
            hand_seed=3,
            seat="first",
            recommended_training_use="local_ev_hard_negative",
            use_for_whole_game_risk_head=0,
            use_for_local_ev_hard_negative=1,
            local_replay_bucket="local_negative",
            local_replay_delta=-0.2,
            local_replay_lcb196=-1.0,
        ),
    ]

    ready = readiness(rows, min_risk_head_rows=1, min_local_ev_hard_negatives=1)

    assert ready["risk_head_training_ready"] is True
    assert ready["local_ev_hard_negative_training_ready"] is True
    assert ready["trajectory_component_analysis_ready"] is True
    assert ready["non_loss_control_rows"] == 1
    assert ready["blockers"] == []
    assert ready["local_ev_hard_negative_blockers"] == []


def test_readiness_allows_risk_head_without_local_ev_hard_negative_minimum():
    rows = [
        _target(hand_seed=1, seat="first", recommended_training_use="whole_game_risk_only"),
        _target(hand_seed=2, seat="second", recommended_training_use="whole_game_risk_only"),
        _target(
            hand_seed=3,
            seat="first",
            realized_delta=3.0,
            recommended_training_use="whole_game_non_loss_control",
            whole_game_risk_label=0,
            use_for_whole_game_risk_head=1,
        ),
    ]

    ready = readiness(rows, min_risk_head_rows=2, min_local_ev_hard_negatives=5)

    assert ready["risk_head_training_ready"] is True
    assert ready["local_ev_hard_negative_training_ready"] is False
    assert ready["trajectory_component_analysis_ready"] is True
    assert ready["blockers"] == []
    assert ready["risk_head_blockers"] == []
    assert ready["local_ev_hard_negative_blockers"] == ["local_ev_hard_negatives_lt_min"]


def test_readiness_does_not_block_local_ev_on_unrelated_missing_replay_rows():
    rows = [
        _target(
            hand_seed=1,
            recommended_training_use="local_ev_hard_negative",
            use_for_whole_game_risk_head=0,
            use_for_local_ev_hard_negative=1,
            local_replay_bucket="local_negative",
            local_replay_delta=-0.2,
            local_replay_lcb196=-1.0,
            local_replay_label="negative",
        ),
        _target(
            hand_seed=2,
            recommended_training_use="whole_game_risk_only",
            use_for_whole_game_risk_head=1,
            local_replay_bucket="local_positive_lcb",
            local_replay_delta=1.2,
            local_replay_lcb196=0.8,
            local_replay_label="positive",
        ),
        _target(
            hand_seed=3,
            recommended_training_use="requires_local_replay",
            requires_local_replay=1,
            local_replay_status="",
            local_replay_delta="",
            local_replay_delta_se="",
            local_replay_lcb196="",
            local_replay_label="",
            local_replay_action_mapping_status="",
        ),
    ]

    ready = readiness(rows, min_risk_head_rows=5, min_local_ev_hard_negatives=1)

    assert ready["missing_replay_rows"] == 1
    assert ready["local_ev_trainable_rows"] == 2
    assert ready["local_ev_trainable_missing_replay_rows"] == 0
    assert ready["local_ev_positive_lcb_controls"] == 1
    assert ready["local_ev_hard_negative_training_ready"] is True
    assert ready["local_ev_hard_negative_blockers"] == []
    assert ready["risk_head_training_ready"] is False


def test_group_breakdown_reports_local_and_risk_splits():
    rows = [
        _target(hand_seed=1, recommended_training_use="whole_game_risk_only"),
        _target(
            hand_seed=3,
            realized_delta=3.0,
            recommended_training_use="whole_game_non_loss_control",
            use_for_whole_game_risk_head=1,
        ),
        _target(
            hand_seed=2,
            recommended_training_use="local_ev_hard_negative",
            use_for_whole_game_risk_head=0,
            use_for_local_ev_hard_negative=1,
            local_replay_bucket="local_negative",
            local_replay_delta=-0.5,
        ),
    ]

    breakdown = group_breakdown(rows)
    overall = next(row for row in breakdown if row["group_field"] == "overall")

    assert overall["rows"] == 3
    assert overall["downstream_trajectory_complete_rows"] == 3
    assert overall["whole_game_risk_only"] == 1
    assert overall["whole_game_non_loss_control"] == 1
    assert overall["local_ev_hard_negative"] == 1
    assert overall["realized_delta_metric_source"] == "realized_whole_game_delta"
    assert overall["confirm_delta_metric_role"] == "gate_diagnostic_only"
    assert overall["confirm_delta_performance_claim_allowed"] is False


def test_cli_writes_audit_artifacts(tmp_path, monkeypatch):
    collection_dir = tmp_path / "collection"
    _write_jsonl(
        collection_dir / "topk_counterfactual_loss_targets_merged.jsonl",
        [
            _target(hand_seed=1, recommended_training_use="whole_game_risk_only"),
            _target(
                hand_seed=2,
                recommended_training_use="local_ev_hard_negative",
                use_for_whole_game_risk_head=0,
                use_for_local_ev_hard_negative=1,
                local_replay_bucket="local_negative",
                local_replay_delta=-0.5,
            ),
        ],
    )
    output_dir = tmp_path / "audit"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--collection-dir",
            str(collection_dir),
            "--output-dir",
            str(output_dir),
            "--min-risk-head-rows",
            "5",
            "--min-local-ev-hard-negatives",
            "2",
        ],
    )

    main()

    assert (output_dir / "risk_target_summary.md").exists()
    assert (output_dir / "risk_target_readiness.json").exists()
    assert _read_csv(output_dir / "risk_target_breakdown.csv")
    ready = json.loads((output_dir / "risk_target_readiness.json").read_text(encoding="utf-8"))
    assert ready["risk_head_training_ready"] is False
    assert ready["local_ev_hard_negative_training_ready"] is False
    assert ready["trajectory_component_analysis_ready"] is True
    manifest = json.loads((output_dir / "risk_target_audit_manifest.json").read_text(encoding="utf-8"))
    assert manifest["trajectory_component_analysis_rows"] == 2
    assert manifest["trajectory_component_complete_rows"] == 2
    assert manifest["trajectory_component_incomplete_rows"] == 0
    assert manifest["realized_delta_metric_source"] == "realized_whole_game_delta"
    assert manifest["confirm_delta_metric_role"] == "gate_diagnostic_only"
    assert manifest["confirm_delta_performance_claim_allowed"] is False
    summary = _read_csv(output_dir / "risk_target_summary.csv")
    by_metric = {row["metric"]: row["value"] for row in summary}
    assert by_metric["confirm_delta_metric_role"] == "gate_diagnostic_only"
    assert by_metric["confirm_delta_performance_claim_allowed"] == "False"


def test_readiness_separates_training_ready_from_trajectory_component_ready():
    rows = [
        _target(
            hand_seed=1,
            seat="first",
            recommended_training_use="whole_game_risk_only",
            downstream_trajectory_complete=0,
            downstream_trajectory_present_fields=2,
        ),
        _target(
            hand_seed=2,
            seat="second",
            realized_delta=3.0,
            recommended_training_use="whole_game_non_loss_control",
            whole_game_risk_label=0,
            use_for_whole_game_risk_head=1,
            downstream_trajectory_complete=0,
            downstream_trajectory_present_fields=2,
        ),
    ]

    ready = readiness(rows, min_risk_head_rows=1, min_local_ev_hard_negatives=5)

    assert ready["risk_head_training_ready"] is True
    assert ready["trajectory_component_analysis_ready"] is False
    assert ready["trajectory_component_analysis_rows"] == 2
    assert ready["trajectory_component_complete_rows"] == 0
    assert ready["trajectory_component_incomplete_rows"] == 2


def test_audit_downstream_complete_falls_back_to_concrete_trajectory_fields():
    row = _target()
    row.pop("downstream_trajectory_complete")
    row.pop("downstream_trajectory_present_fields")
    row.pop("downstream_trajectory_total_fields")
    for field in DOWNSTREAM_TRAJECTORY_FIELDS:
        row[field] = {"present": True}

    assert downstream_complete(row) == 1
    assert normalized_row(row)["downstream_trajectory_complete"] == 1

    row.pop("final_board_hero")
    assert downstream_complete(row) == 0
