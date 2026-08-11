import numpy as np

from ofc_regular.train_hu_turn3_stage9_tail_veto import (
    FEATURE_COLUMNS,
    feature_matrix,
    labels,
    metrics,
    source_split,
)


def _row(source: str, label: str, delta: float, margin: float) -> dict:
    row = {
        "source_name": source,
        "tail_label": label,
        "delta_vs_baseline": str(delta),
        "predicted_margin_vs_baseline": str(margin),
        "reference_margin": "0.5",
        "model_score": "10.0",
        "best_score": "12.0",
        "legal_action_count": "20",
        "hu_top_full": "false",
        "hu_middle_full": "true",
        "hu_bottom_full": "false",
        "baseline_top_full": "true",
        "baseline_middle_full": "false",
        "baseline_bottom_full": "false",
    }
    for col in FEATURE_COLUMNS:
        row.setdefault(col, "0")
    return row


def test_feature_matrix_and_labels_use_runtime_columns_only():
    rows = [
        _row("a", "safe_positive", 1.0, 3.0),
        _row("b", "hard_negative", -1.0, 1.0),
    ]

    X = feature_matrix(rows)
    y = labels(rows)

    assert X.shape == (2, len(FEATURE_COLUMNS))
    assert X.dtype == np.float32
    assert y.tolist() == [0, 1]


def test_source_split_prefers_source_with_hard_negatives():
    rows = [
        _row("safe_source", "safe_positive", 1.0, 3.0),
        _row("hard_source", "hard_negative", -1.0, 1.0),
        _row("hard_source", "gray", 0.0, 1.5),
    ]

    train_idx, test_idx, holdout = source_split(rows)

    assert holdout == "hard_source"
    assert test_idx == [1, 2]
    assert train_idx == [0]


def test_metrics_handles_basic_classifier_scores():
    y = np.asarray([1, 0, 1, 0])
    scores = np.asarray([0.9, 0.8, 0.7, 0.1])

    result = metrics(y, scores, threshold=0.75)

    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["tn"] == 1
    assert result["average_precision"] > 0.0
    assert result["roc_auc"] > 0.0
