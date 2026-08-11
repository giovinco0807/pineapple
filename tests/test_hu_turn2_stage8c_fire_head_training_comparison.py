import csv
import json

from ofc_regular.compare_hu_turn2_stage8c_fire_head_training import compare_runs, parse_run_specs


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_run(path, *, auc, ap, top3_precision, threshold_precision, excluded_empty=0):
    path.mkdir(parents=True, exist_ok=True)
    (path / "risk_head_training_manifest.json").write_text(
        json.dumps(
            {
                "trainable_rows": 100,
                "positive_rows": 10,
                "negative_rows": 90,
                "feature_mode": "opportunity_proxy_plus_preconfirm_meta",
                "split_mode": "source_seed",
                "pos_weight_mode": "auto",
                "pos_weight": 9.0,
                "topk_realized_loss_weight": 1.0,
                "excluded_recommended_use_counts": {"topk_confirm_topk_empty": excluded_empty},
                "target_group_counts": {
                "topk_confirm_realized_positive": 10,
                "topk_confirm_realized_loss": 20,
                "topk_confirm_rejected": 70,
                "topk_confirm_replay_positive": 3,
                "topk_confirm_replay_negative": 4,
                "topk_confirm_replay_gray": 5,
            },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(
        path / "risk_head_metrics.csv",
        [
            {"split": "val", "rows": 50, "positives": 5, "average_precision": ap, "roc_auc": auc, "brier": 0.1},
            {"split": "test", "rows": 50, "positives": 5, "average_precision": ap, "roc_auc": auc, "brier": 0.1},
            {"split": "all", "rows": 100, "positives": 10, "average_precision": ap, "roc_auc": auc, "brier": 0.1},
        ],
    )
    _write_csv(
        path / "risk_head_topk_metrics.csv",
        [
            {
                "split": "test",
                "topk": 3,
                "selected_rows": 3,
                "precision": top3_precision,
                "recall": 0.1,
                "selected_observed_realized_delta_mean": 4.0,
            },
            {
                "split": "test",
                "topk": 5,
                "selected_rows": 5,
                "precision": 0.4,
                "recall": 0.2,
                "selected_observed_realized_delta_mean": 3.0,
            },
            {
                "split": "test",
                "topk": 10,
                "selected_rows": 10,
                "precision": 0.3,
                "recall": 0.3,
                "selected_observed_realized_delta_mean": 2.0,
            },
        ],
    )
    _write_csv(
        path / "risk_head_threshold_metrics.csv",
        [
            {
                "split": "test",
                "threshold": 0.5,
                "selected_rows": 20,
                "precision": 0.2,
                "recall": 0.4,
                "selected_observed_realized_delta_mean": 1.0,
            },
            {
                "split": "test",
                "threshold": 0.8,
                "selected_rows": 10,
                "precision": threshold_precision,
                "recall": 0.2,
                "selected_observed_realized_delta_mean": 1.0,
            },
            {
                "split": "test",
                "threshold": 0.9,
                "selected_rows": 1,
                "precision": 1.0,
                "recall": 0.1,
                "selected_observed_realized_delta_mean": 6.0,
            },
        ],
    )


def test_parse_run_specs_requires_unique_labels(tmp_path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()

    runs = parse_run_specs([f"a={first}", f"b={second}"])

    assert [label for label, _path in runs] == ["a", "b"]


def test_compare_runs_prefers_better_triage_score_and_keeps_runtime_no_go(tmp_path):
    weaker = tmp_path / "weaker"
    stronger = tmp_path / "stronger"
    _write_run(weaker, auc=0.6, ap=0.2, top3_precision=0.33, threshold_precision=0.3)
    _write_run(stronger, auc=0.7, ap=0.25, top3_precision=0.67, threshold_precision=0.35, excluded_empty=4000)

    comparison = compare_runs([("weaker", weaker), ("stronger", stronger)], preferred_topk=3)
    rows = comparison["summary_rows"]

    assert comparison["preferred_run"] == "stronger"
    assert comparison["runtime_gate_decision"] == "No-Go"
    assert rows[0]["excluded_topk_empty_rows"] == 4000
    assert rows[0]["replay_positive_rows"] == 3
    assert rows[0]["replay_negative_rows"] == 4
    assert rows[0]["replay_gray_rows"] == 5
    assert rows[0]["test_top3_precision"] == 0.67
    assert rows[0]["runtime_gate_decision"] == "No-Go"
