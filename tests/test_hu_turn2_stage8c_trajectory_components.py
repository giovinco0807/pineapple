import csv
import json

from ofc_regular.analyze_hu_turn2_stage8c_trajectory_components import (
    DOWNSTREAM_TRAJECTORY_FIELDS,
    component_coverage_rows,
    downstream_complete,
    main,
    metric_rows,
    normalize_row,
    primary_loss_component,
)


def _row(**overrides):
    row = {
        "schema": "hu_turn2_stage8b_counterfactual_loss_target_v1",
        "source_log": "outputs/run/runtime_decisions.jsonl",
        "config_id": "cfg",
        "hand_seed": 1,
        "hand_id": 1,
        "seat": "first",
        "seat_swap": "ab",
        "candidate_index": 3,
        "baseline_index": 1,
        "recommended_training_use": "whole_game_risk_only",
        "realized_delta": -8.0,
        "terminal_score_vs_baseline": -8.0,
        "foul_delta_vs_baseline": 0.0,
        "line_score_delta_vs_baseline": -1.0,
        "scoop_delta_vs_baseline": -3.0,
        "royalty_delta_vs_baseline": -4.0,
        "fl_delta_vs_baseline": 0.0,
        "hero_royalty_vs_baseline": -4.0,
        "opponent_royalty_vs_baseline": 0.0,
        "hero_fl_value_vs_baseline": 0.0,
        "opponent_fl_value_vs_baseline": 0.0,
        "hero_board": {"top": ["Qh"], "middle": [], "bottom": []},
        "opponent_board": {"top": [], "middle": [], "bottom": []},
        "cards_to_place": ["Ah", "Ks", "3d"],
        "post_t2_candidate_board": {"top": ["Qh", "Ah"], "middle": [], "bottom": []},
        "post_t2_baseline_board": {"top": ["Qh"], "middle": ["Ah"], "bottom": []},
        "candidate_action": {"placements": [], "discards": []},
        "baseline_action": {"placements": [], "discards": []},
    }
    for field in DOWNSTREAM_TRAJECTORY_FIELDS:
        row.setdefault(field, {"present": True})
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


def test_primary_loss_component_uses_largest_negative_terminal_component():
    assert primary_loss_component(_row()) == "royalty"
    assert primary_loss_component(_row(royalty_delta_vs_baseline=0.0, fl_delta_vs_baseline=-10.0)) == "fl"
    assert primary_loss_component(_row(realized_delta=3.0, terminal_score_vs_baseline=3.0)) == "non_loss"
    assert (
        primary_loss_component(
            _row(
                foul_delta_vs_baseline=0.0,
                line_score_delta_vs_baseline=0.0,
                scoop_delta_vs_baseline=0.0,
                royalty_delta_vs_baseline=0.0,
                fl_delta_vs_baseline=0.0,
            )
        )
        == "unexplained"
    )


def test_metric_rows_report_component_counts_and_coverage():
    rows = [
        _row(hand_seed=1),
        _row(hand_seed=2, realized_delta=0.0, terminal_score_vs_baseline=0.0),
        _row(hand_seed=3, royalty_delta_vs_baseline=0.0, fl_delta_vs_baseline=-10.227),
    ]
    normalized = [normalize_row(row) for row in rows]
    metrics = metric_rows(normalized, rows, [{"source_path": "x", "rows": len(rows)}])
    values = {row["metric"]: row["value"] for row in metrics}
    coverage = component_coverage_rows(rows)

    assert values["loss_rows"] == 2
    assert values["primary_loss_component.royalty"] == 1
    assert values["primary_loss_component.fl"] == 1
    assert values["primary_loss_component.non_loss"] == 1
    assert all(row["present_rows"] == 3 for row in coverage)


def test_downstream_complete_falls_back_to_concrete_fields():
    row = _row()

    assert downstream_complete(row)

    row.pop("final_board_hero")
    assert not downstream_complete(row)


def test_cli_writes_component_artifacts(tmp_path, monkeypatch):
    collection_dir = tmp_path / "collection"
    _write_jsonl(
        collection_dir / "topk_counterfactual_loss_targets.jsonl",
        [
            _row(hand_seed=1),
            _row(hand_seed=2, realized_delta=0.0, terminal_score_vs_baseline=0.0),
            _row(
                hand_seed=3,
                realized_delta=-99.0,
                terminal_score_vs_baseline=-99.0,
                downstream_trajectory_complete=0,
                final_board_hero="",
            ),
        ],
    )
    output_dir = tmp_path / "components"
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

    assert (output_dir / "trajectory_component_summary.md").exists()
    assert (output_dir / "trajectory_component_breakdown.csv").exists()
    assert (output_dir / "trajectory_component_top_losses.jsonl").exists()
    metrics = _read_csv(output_dir / "trajectory_component_metrics.csv")
    values = {row["metric"]: row["value"] for row in metrics}
    assert values["deduped_rows"] == "3"
    assert values["analysis_rows"] == "2"
    assert values["trajectory_incomplete_rows_excluded"] == "1"
    assert values["loss_rows"] == "1"
    top_loss = json.loads((output_dir / "trajectory_component_top_losses.jsonl").read_text(encoding="utf-8"))
    assert top_loss["hero_board"] == {"top": ["Qh"], "middle": [], "bottom": []}
    assert top_loss["post_t2_candidate_board"] == {"top": ["Qh", "Ah"], "middle": [], "bottom": []}
    assert top_loss["candidate_action"] == {"placements": [], "discards": []}
    manifest = json.loads((output_dir / "trajectory_component_manifest.json").read_text(encoding="utf-8"))
    assert manifest["rows"] == 3
    assert manifest["analysis_rows"] == 2
    assert manifest["trajectory_incomplete_rows_excluded"] == 1
    assert manifest["runtime_risk_integration"] is False
