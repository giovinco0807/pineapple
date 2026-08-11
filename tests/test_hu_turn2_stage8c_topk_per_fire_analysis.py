import csv
import json
from types import SimpleNamespace

import pytest

from ofc_regular.analyze_hu_turn2_stage8c_topk_per_fire import (
    aggregate,
    aggregate_manifest,
    aggregate_position_breakdown,
    aggregate_risk_veto_candidate_metrics,
    discover_input_dirs,
    main,
)


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_aggregate_uses_realized_per_fire_not_confirm_delta(tmp_path):
    eval_dir = tmp_path / "eval"
    _write_csv(
        eval_dir / "seed_breakdown.csv",
        [
            {
                "config_id": "cfg",
                "paired_seeds": "2",
                "ev_per_hand": "1.0",
            }
        ],
    )
    _write_jsonl(
        eval_dir / "runtime_decisions.jsonl",
        [
            {
                "config_id": "cfg",
                "override_fired": True,
                "realized_delta_valid": True,
                "realized_candidate_seat_delta": -2.0,
                "rerank_delta": 99.0,
                "confirm_delta": 1.0,
                "predicted_delta": 1.0,
                "dead_cards": ["2c", "3c"],
                "visible_dead_cards": ["Ah", "2c"],
                "hero_private_discards": ["2c"],
                "opponent_private_discards": ["3c"],
            },
            {
                "config_id": "cfg",
                "override_fired": True,
                "realized_delta_valid": True,
                "realized_candidate_seat_delta": 4.0,
                "rerank_delta": 99.0,
                "confirm_delta": 3.0,
                "predicted_delta": 1.0,
            },
            {
                "config_id": "cfg",
                "override_fired": False,
                "realized_delta_valid": True,
                "realized_candidate_seat_delta": 0.0,
                "no_override_reason": "topk_empty",
            },
        ],
    )
    _write_csv(
        eval_dir / "cancellation_audit.csv",
        [
            {
                "config_id": "cfg",
                "valid_realized_delta_count": "3",
                "non_fired_count": "1",
                "non_fired_nonzero_count": "0",
                "non_fired_delta_sum": "0",
                "non_fired_delta_max_abs": "0",
                "fired_count": "2",
                "fired_delta_sum": "2",
            }
        ],
    )

    summary_rows, cancellation_rows, _seed_rows = aggregate([eval_dir])

    assert len(summary_rows) == 1
    assert summary_rows[0]["per_fire_delta_mean"] == 1.0
    assert summary_rows[0]["confirm_delta_mean_on_fired"] == 2.0
    assert summary_rows[0]["confirm_mean_minus_realized_per_fire"] == 1.0
    assert summary_rows[0]["primary_metric_source"] == "realized_fired_whole_game_delta"
    assert summary_rows[0]["per_fire_performance_column"] == "per_fire_delta_mean"
    assert summary_rows[0]["hand_ev_performance_column"] == "estimated_ev_per_hand"
    assert summary_rows[0]["confirm_delta_metric_role"] == "gate_diagnostic_only"
    assert summary_rows[0]["confirm_delta_performance_claim_allowed"] is False
    assert summary_rows[0]["realized_fire_count_sufficient"] is False
    assert summary_rows[0]["realized_loss_count"] == 1
    assert summary_rows[0]["fired_replay_ready_count"] == 1
    assert summary_rows[0]["fired_replay_ready_rate"] == 0.5
    assert json.loads(summary_rows[0]["fired_replay_missing_field_counts"]) == {
        "dead_cards": 1,
        "hero_private_discards": 1,
        "opponent_private_discards": 1,
        "visible_dead_cards": 1,
    }
    assert cancellation_rows[0]["non_fired_nonzero_count"] == 0
    manifest = aggregate_manifest([eval_dir], summary_rows, cancellation_rows)
    assert manifest["primary_metric_valid"] is True
    assert manifest["cancellation_clean"] is True
    assert manifest["total_realized_fires"] == 2
    assert manifest["best_config_id"] == "cfg"
    assert manifest["primary_metric_source"] == "realized_fired_whole_game_delta"
    assert manifest["per_fire_performance_column"] == "per_fire_delta_mean"
    assert manifest["hand_ev_performance_column"] == "estimated_ev_per_hand"
    assert manifest["confirm_delta_metric_role"] == "gate_diagnostic_only"
    assert manifest["confirm_delta_performance_claim_allowed"] is False
    assert manifest["non_fired_cancellation_required_for_primary_metric"] is True
    assert "cancellation_audit must be present" in manifest["non_fired_cancellation_requirement"]
    assert manifest["best_realized_fire_count_sufficient"] is False
    assert manifest["best_positive_per_fire_ci"] is False
    assert manifest["evidence_decision"] == "No-Go"


def test_aggregate_reports_negative_predicted_overrides(tmp_path):
    eval_dir = tmp_path / "eval"
    _write_csv(eval_dir / "seed_breakdown.csv", [{"config_id": "cfg", "paired_seeds": "1", "ev_per_hand": "0"}])
    _write_jsonl(
        eval_dir / "runtime_decisions.jsonl",
        [
            {
                "config_id": "cfg",
                "override_fired": True,
                "realized_delta_valid": True,
                "realized_candidate_seat_delta": 3.0,
                "predicted_delta": -0.1,
            }
        ],
    )
    _write_csv(
        eval_dir / "cancellation_audit.csv",
        [
            {
                "config_id": "cfg",
                "valid_realized_delta_count": "1",
                "non_fired_count": "0",
                "non_fired_nonzero_count": "0",
                "non_fired_delta_sum": "0",
                "non_fired_delta_max_abs": "0",
                "fired_count": "1",
                "fired_delta_sum": "3",
            }
        ],
    )

    summary_rows, _cancellation_rows, _seed_rows = aggregate([eval_dir])

    assert summary_rows[0]["negative_predicted_delta_override_count"] == 1
    assert summary_rows[0]["fired_replay_ready_count"] == 0


def test_aggregate_reports_replay_ready_runtime_decisions(tmp_path):
    eval_dir = tmp_path / "eval"
    _write_csv(eval_dir / "seed_breakdown.csv", [{"config_id": "cfg", "paired_seeds": "1", "ev_per_hand": "0"}])
    _write_jsonl(
        eval_dir / "runtime_decisions.jsonl",
        [
            {
                "config_id": "cfg",
                "override_fired": False,
                "realized_delta_valid": True,
                "realized_candidate_seat_delta": 0.0,
                "dead_cards": ["2c", "3c"],
                "visible_dead_cards": ["Ah", "2c"],
                "hero_private_discards": ["2c"],
                "opponent_private_discards": ["3c"],
            },
            {
                "config_id": "cfg",
                "override_fired": True,
                "realized_delta_valid": True,
                "realized_candidate_seat_delta": 3.0,
                "predicted_delta": 0.2,
                "dead_cards": ["2c", "3c"],
                "visible_dead_cards": ["Ah", "2c"],
                "hero_private_discards": ["2c"],
                "opponent_private_discards": ["3c"],
            },
        ],
    )
    _write_csv(
        eval_dir / "cancellation_audit.csv",
        [
            {
                "config_id": "cfg",
                "valid_realized_delta_count": "2",
                "non_fired_count": "1",
                "non_fired_nonzero_count": "0",
                "non_fired_delta_sum": "0",
                "non_fired_delta_max_abs": "0",
                "fired_count": "1",
                "fired_delta_sum": "3",
            }
        ],
    )

    summary_rows, _cancellation_rows, _seed_rows = aggregate([eval_dir])

    assert summary_rows[0]["decision_replay_ready_count"] == 2
    assert summary_rows[0]["decision_replay_ready_rate"] == 1.0
    assert summary_rows[0]["fired_replay_ready_count"] == 1
    assert summary_rows[0]["fired_replay_ready_rate"] == 1.0
    assert json.loads(summary_rows[0]["replay_missing_field_counts"]) == {}


def test_aggregate_risk_veto_metrics_use_would_veto_candidate_delta(tmp_path):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    _write_jsonl(
        eval_dir / "runtime_decisions.jsonl",
        [
            {
                "config_id": "cfg",
                "local_ev_risk_would_veto": True,
                "local_ev_risk_audit_only": True,
                "local_ev_risk_vetoed": False,
                "realized_candidate_seat_delta": 4.0,
            },
            {
                "config_id": "cfg",
                "local_ev_risk_would_veto": True,
                "local_ev_risk_audit_only": True,
                "local_ev_risk_vetoed": False,
                "realized_candidate_seat_delta": -2.0,
            },
            {
                "config_id": "cfg",
                "local_ev_risk_would_veto": False,
                "realized_candidate_seat_delta": -100.0,
            },
        ],
    )

    [row] = aggregate_risk_veto_candidate_metrics([eval_dir])

    assert row["decision_count"] == 3
    assert row["would_veto_count"] == 2
    assert row["audit_only_would_veto_count"] == 2
    assert row["actual_veto_count"] == 0
    assert row["realized_would_veto_count"] == 2
    assert row["realized_candidate_delta_mean"] == 1.0
    assert row["veto_utility_per_veto"] == -1.0
    assert row["estimated_veto_utility_per_hand"] == pytest.approx(-2 / 3)


def test_aggregate_position_breakdown_reports_seat_specific_realized_delta(tmp_path):
    eval_dir = tmp_path / "eval"
    _write_csv(
        eval_dir / "position_breakdown.csv",
        [
            {"config_id": "cfg", "seat": "first", "hands": "10", "ev_per_hand": "0.5"},
            {"config_id": "cfg", "seat": "second", "hands": "10", "ev_per_hand": "-0.25"},
        ],
    )
    _write_jsonl(
        eval_dir / "runtime_decisions.jsonl",
        [
            {
                "config_id": "cfg",
                "seat": "first",
                "override_fired": True,
                "realized_delta_valid": True,
                "realized_candidate_seat_delta": 4.0,
            },
            {
                "config_id": "cfg",
                "seat": "first",
                "override_fired": False,
                "realized_delta_valid": True,
                "realized_candidate_seat_delta": 0.0,
                "no_override_reason": "topk_empty",
            },
            {
                "config_id": "cfg",
                "seat": "second",
                "override_fired": True,
                "realized_delta_valid": True,
                "realized_candidate_seat_delta": -2.0,
            },
        ],
    )

    rows = aggregate_position_breakdown([eval_dir])
    by_seat = {row["seat"]: row for row in rows}

    assert by_seat["first"]["whole_game_ev_per_hand"] == 0.5
    assert by_seat["first"]["decision_count"] == 2
    assert by_seat["first"]["realized_override_count"] == 1
    assert by_seat["first"]["per_fire_delta_mean"] == 4.0
    assert by_seat["first"]["estimated_ev_per_hand"] == 2.0
    assert by_seat["second"]["whole_game_ev_per_hand"] == -0.25
    assert by_seat["second"]["per_fire_delta_mean"] == -2.0


def test_aggregate_manifest_blocks_broken_non_fired_cancellation(tmp_path):
    eval_dir = tmp_path / "eval"
    _write_csv(eval_dir / "seed_breakdown.csv", [{"config_id": "cfg", "paired_seeds": "1", "ev_per_hand": "0"}])
    _write_jsonl(
        eval_dir / "runtime_decisions.jsonl",
        [
            {
                "config_id": "cfg",
                "override_fired": True,
                "realized_delta_valid": True,
                "realized_candidate_seat_delta": 3.0,
                "predicted_delta": 0.2,
            },
            {
                "config_id": "cfg",
                "override_fired": False,
                "realized_delta_valid": True,
                "realized_candidate_seat_delta": 2.0,
            },
        ],
    )
    _write_csv(
        eval_dir / "position_breakdown.csv",
        [{"config_id": "cfg", "seat": "first", "hands": "1", "ev_per_hand": "0"}],
    )
    _write_csv(
        eval_dir / "cancellation_audit.csv",
        [
            {
                "config_id": "cfg",
                "valid_realized_delta_count": "2",
                "non_fired_count": "1",
                "non_fired_nonzero_count": "1",
                "non_fired_delta_sum": "2",
                "non_fired_delta_max_abs": "2",
                "fired_count": "1",
                "fired_delta_sum": "3",
            }
        ],
    )

    summary_rows, cancellation_rows, _seed_rows = aggregate([eval_dir])
    manifest = aggregate_manifest([eval_dir], summary_rows, cancellation_rows)

    assert manifest["primary_metric_valid"] is False
    assert manifest["cancellation_clean"] is False
    assert manifest["evaluation_decision"] == "No-Go"
    assert manifest["evidence_decision"] == "No-Go"


def test_aggregate_manifest_blocks_missing_cancellation_audit_for_config(tmp_path):
    summary_rows = [
        {
            "config_id": "cfg",
            "paired_seeds": 1000,
            "decision_count": 2000,
            "realized_override_count": 50,
            "estimated_ev_per_hand": 0.05,
            "per_fire_delta_mean": 2.0,
            "per_fire_delta_ci95_low": 0.5,
        }
    ]

    manifest = aggregate_manifest([tmp_path], summary_rows, [])

    assert manifest["missing_cancellation_audit_configs"] == ["cfg"]
    assert manifest["cancellation_audit_present"] is False
    assert manifest["cancellation_clean"] is False
    assert manifest["primary_metric_valid"] is False
    assert manifest["evaluation_decision"] == "No-Go"
    assert manifest["evidence_decision"] == "No-Go"


def test_aggregate_manifest_passes_evidence_only_with_enough_positive_realized_fires(tmp_path):
    summary_rows = [
        {
            "config_id": "cfg",
            "paired_seeds": 1000,
            "decision_count": 2000,
            "realized_override_count": 50,
            "estimated_ev_per_hand": 0.05,
            "per_fire_delta_mean": 2.0,
            "per_fire_delta_ci95_low": 0.5,
        }
    ]
    cancellation_rows = [
        {
            "config_id": "cfg",
            "valid_realized_delta_count": 1000,
            "non_fired_count": 950,
            "non_fired_nonzero_count": 0,
            "non_fired_delta_sum": 0.0,
            "non_fired_delta_max_abs": 0.0,
        }
    ]

    manifest = aggregate_manifest([tmp_path], summary_rows, cancellation_rows)

    assert manifest["primary_metric_valid"] is True
    assert manifest["non_fired_cancellation_required_for_primary_metric"] is True
    assert manifest["best_realized_fire_count_sufficient"] is True
    assert manifest["best_positive_per_fire_ci"] is True
    assert manifest["evidence_decision"] == "Pass"


def test_cli_writes_manifest_with_primary_metric_validity(tmp_path, monkeypatch):
    eval_dir = tmp_path / "eval"
    _write_csv(eval_dir / "seed_breakdown.csv", [{"config_id": "cfg", "paired_seeds": "1", "ev_per_hand": "0"}])
    _write_jsonl(
        eval_dir / "runtime_decisions.jsonl",
        [
            {
                "config_id": "cfg",
                "override_fired": True,
                "realized_delta_valid": True,
                "realized_candidate_seat_delta": 3.0,
                "predicted_delta": 0.2,
                "seat": "first",
                "local_ev_risk_would_veto": True,
                "local_ev_risk_audit_only": True,
            }
        ],
    )
    _write_csv(
        eval_dir / "cancellation_audit.csv",
        [
            {
                "config_id": "cfg",
                "valid_realized_delta_count": "1",
                "non_fired_count": "0",
                "non_fired_nonzero_count": "0",
                "non_fired_delta_sum": "0",
                "non_fired_delta_max_abs": "0",
                "fired_count": "1",
                "fired_delta_sum": "3",
            }
        ],
    )
    output_dir = tmp_path / "aggregate"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--input-dir",
            str(eval_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    main()

    manifest = json.loads((output_dir / "aggregate_manifest.json").read_text(encoding="utf-8"))
    assert manifest["primary_metric_valid"] is True
    assert manifest["cancellation_clean"] is True
    assert manifest["evidence_decision"] == "No-Go"
    assert manifest["position_breakdown_present"] is True
    assert manifest["risk_veto_candidate_metric_present"] is True
    assert manifest["risk_veto_adoption_decision"] == "No-Go"
    assert manifest["production_p2_fixed"] == "No-Go"
    risk_rows = list(csv.DictReader((output_dir / "aggregate_risk_veto_candidate_metrics.csv").open()))
    assert risk_rows[0]["veto_utility_per_veto"] == "-3.0"
    position_rows = list(csv.DictReader((output_dir / "aggregate_position_breakdown.csv").open()))
    assert position_rows[0]["seat"] == "first"
    summary = (output_dir / "aggregate_summary.md").read_text(encoding="utf-8")
    assert "primary metric valid: `True`" in summary
    assert "primary metric source: `realized_fired_whole_game_delta`" in summary
    assert "per-fire performance column: `per_fire_delta_mean`" in summary
    assert "hand EV performance column: `estimated_ev_per_hand`" in summary
    assert "confirm delta role: `gate_diagnostic_only`" in summary
    assert "confirm delta performance claim allowed: `False`" in summary
    assert "non-fired cancellation required: `True`" in summary
    assert "evidence decision: `No-Go`" in summary
    assert "position breakdown present: `True`" in summary
    assert "risk veto adoption: `No-Go`" in summary


def test_discover_input_dirs_rejects_missing_explicit_directory(tmp_path):
    with pytest.raises(SystemExit, match="--input-dir does not exist"):
        discover_input_dirs(SimpleNamespace(input_dir=[tmp_path / "missing"], input_glob=[]))


def test_discover_input_dirs_rejects_unmatched_glob(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit, match="--input-glob matched no directories"):
        discover_input_dirs(SimpleNamespace(input_dir=[], input_glob=["missing-*"]))
