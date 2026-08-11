import csv
import json

from ofc_regular.analyze_hu_turn2_stage8c_source_heldout import analyze_runs, parse_run_specs, write_outputs


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_run(path, *, source_seed, selected_deltas, unselected_count=4):
    path.mkdir(parents=True, exist_ok=True)
    (path / "risk_head_training_manifest.json").write_text(
        json.dumps(
            {
                "fixed_test_groups": [str(source_seed)],
                "split_mode": "source_seed",
                "feature_mode": "opportunity_proxy_plus_preconfirm_meta",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(
        path / "risk_head_metrics.csv",
        [
            {
                "split": "test",
                "rows": len(selected_deltas) + unselected_count,
                "positives": sum(1 for delta in selected_deltas if delta > 0),
                "average_precision": 0.5,
                "roc_auc": 0.7,
                "brier": 0.2,
            }
        ],
    )
    rows = []
    for index, delta in enumerate(selected_deltas):
        rows.append(
            {
                "row_index": index,
                "split": "test",
                "split_group_source_seed": source_seed,
                "source_log": f"source_{source_seed}.jsonl",
                "label": int(delta > 0),
                "risk_probability": 0.75,
                "realized_delta": delta,
                "realized_delta_observed": 1,
            }
        )
    for offset in range(unselected_count):
        rows.append(
            {
                "row_index": len(rows),
                "split": "test",
                "split_group_source_seed": source_seed,
                "source_log": f"source_{source_seed}.jsonl",
                "label": 0,
                "risk_probability": 0.1,
                "realized_delta": 0.0,
                "realized_delta_observed": 0,
            }
        )
    _write_csv(path / "risk_head_predictions.csv", rows)


def test_parse_run_specs_requires_label_path(tmp_path):
    run = tmp_path / "run"
    _write_run(run, source_seed=1, selected_deltas=[1, 2])

    parsed = parse_run_specs([f"r1={run}"])

    assert parsed == [("r1", run)]


def test_source_heldout_analysis_keeps_negative_source_no_go(tmp_path):
    positive = tmp_path / "positive"
    negative = tmp_path / "negative"
    neutral = tmp_path / "neutral"
    _write_run(positive, source_seed=1804, selected_deltas=[2.0, 1.0, 3.0])
    _write_run(negative, source_seed=1805, selected_deltas=[-2.0, 1.0, -1.0])
    _write_run(neutral, source_seed=1806, selected_deltas=[1.0, 1.0, 1.0])

    analysis = analyze_runs(
        [("positive", positive), ("negative", negative), ("neutral", neutral)],
        thresholds=[0.5],
        min_selected_per_source=3,
        min_sources_with_selection=3,
    )

    stability = analysis["stability_rows"][0]
    assert analysis["decision"] == "No-Go"
    assert analysis["reason"] == "no_threshold_positive_across_sources"
    assert stability["negative_source_count"] == 1
    assert stability["decision"] == "No-Go"
    assert stability["no_go_reason"] == "negative_source_delta"


def test_source_heldout_analysis_allows_stable_positive_candidate(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    third = tmp_path / "third"
    _write_run(first, source_seed=1804, selected_deltas=[2.0, 1.0, 3.0])
    _write_run(second, source_seed=1805, selected_deltas=[1.0, 1.0, 1.0])
    _write_run(third, source_seed=1806, selected_deltas=[4.0, -1.0, 1.0])

    analysis = analyze_runs(
        [("first", first), ("second", second), ("third", third)],
        thresholds=[0.5],
        min_selected_per_source=3,
        min_sources_with_selection=3,
    )

    stability = analysis["stability_rows"][0]
    assert analysis["decision"] == "Needs runtime validation"
    assert analysis["reason"] == "stable_threshold:0.5"
    assert stability["stable_candidate"] == 1
    assert stability["eligible_min_estimated_delta_per_test_row"] > 0


def test_source_heldout_analysis_writes_decision_artifacts(tmp_path):
    run = tmp_path / "run"
    out = tmp_path / "out"
    _write_run(run, source_seed=1804, selected_deltas=[2.0, 1.0, 3.0])
    analysis = analyze_runs(
        [("run", run)],
        thresholds=[0.5],
        min_selected_per_source=3,
        min_sources_with_selection=1,
    )

    write_outputs(out, analysis)

    decision = json.loads((out / "source_heldout_decision.json").read_text(encoding="utf-8"))
    assert decision["selector_promotion"] == "Needs runtime validation"
    assert (out / "source_heldout_summary.md").exists()
    assert (out / "source_heldout_threshold_by_source.csv").exists()
