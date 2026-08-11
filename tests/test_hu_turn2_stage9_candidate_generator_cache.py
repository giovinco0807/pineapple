import numpy as np

from ofc_regular.prepare_hu_turn2_stage9_candidate_generator_cache import (
    build_candidate_rows,
    metric_rows,
    parse_topk_values,
)


def _cache():
    return {
        "metadata": {"state_count": 2, "action_count": 7},
        "offsets": np.asarray([0, 4, 7], dtype=np.int64),
        "target_ev": np.asarray([0.0, 2.0, 1.0, 0.5, 1.0, 0.7, 1.4], dtype=np.float32),
        "target_delta_baseline": np.asarray([0.0, 2.0, 1.0, 0.5, 0.0, -0.3, 0.4], dtype=np.float32),
        "baseline_action_index": np.asarray([0, 0], dtype=np.int64),
        "best_action_index": np.asarray([1, 2], dtype=np.int64),
        "split": np.asarray([0, 2], dtype=np.int64),
        "state_metadata": [
            {
                "state_hash": "s0",
                "state_id": "s0",
                "seat": "first",
                "bucket_group": "natural",
                "source_bucket": "natural",
                "run_bucket": "natural",
                "pilot_gate_label": "positive",
            },
            {
                "state_hash": "s1",
                "state_id": "s1",
                "seat": "second",
                "bucket_group": "random_off_policy",
                "source_bucket": "random_off_policy",
                "run_bucket": "random_off_policy",
                "pilot_gate_label": "gray",
            },
        ],
    }


def test_parse_topk_values_sorts_and_dedupes():
    assert parse_topk_values("5,1,3,3") == [1, 3, 5]


def test_build_candidate_rows_marks_hard_miss_when_oracle_best_outside_topk():
    predictions = np.zeros((7, 5), dtype=np.float32)
    predictions[:4, 0] = [5.0, 3.0, 4.0, 1.0]  # state 0 misses oracle action 1 in top1
    predictions[4:, 0] = [2.0, 1.0, 3.0]

    built = build_candidate_rows(
        _cache(),
        predictions,
        topk_values=[1, 3],
        hard_miss_topk=1,
        min_hard_miss_regret=0.25,
        missed_positive_weight=6.0,
    )

    hard_misses = built["hard_miss_rows"]
    assert len(hard_misses) == 1
    assert hard_misses[0]["state_index"] == 0
    assert hard_misses[0]["oracle_best_index"] == 1
    assert hard_misses[0]["model_top1_index"] == 0
    assert hard_misses[0]["hard_miss_regret"] == 2.0
    missed = next(row for row in built["training_rows"] if row["label"] == "missed_oracle_positive")
    assert missed["weight"] == 6.0


def test_build_candidate_rows_marks_model_topk_hard_negative():
    predictions = np.zeros((7, 5), dtype=np.float32)
    predictions[:4, 0] = [4.0, 3.0, 2.0, 1.0]
    predictions[4:, 0] = [2.0, 5.0, 1.0]  # state 1 top1 is action 1, below baseline by 0.3

    built = build_candidate_rows(
        _cache(),
        predictions,
        topk_values=[1, 3],
        hard_negative_topk=1,
        min_hard_negative_loss=0.25,
        hard_negative_weight=1.25,
    )

    hard_negatives = built["hard_negative_rows"]
    assert len(hard_negatives) == 1
    assert hard_negatives[0]["state_index"] == 1
    assert hard_negatives[0]["hard_negative_index"] == 1
    assert hard_negatives[0]["hard_negative_delta_vs_baseline"] == -0.30000001192092896
    hard_negative = next(row for row in built["training_rows"] if row["label"] == "model_topk_hard_negative")
    assert hard_negative["weight"] == 1.25


def test_metric_rows_reports_topk_recall_and_regret():
    predictions = np.zeros((7, 5), dtype=np.float32)
    predictions[:4, 0] = [5.0, 3.0, 4.0, 1.0]
    predictions[4:, 0] = [2.0, 1.0, 3.0]

    built = build_candidate_rows(_cache(), predictions, topk_values=[1, 3], hard_miss_topk=1)
    rows = metric_rows(built["state_rows"], [1, 3])
    overall = next(row for row in rows if row["group_type"] == "all")

    assert overall["states"] == 2
    assert overall["top1_oracle_recall"] == 0.5
    assert overall["top3_oracle_recall"] == 1.0
    assert overall["top1_avg_regret"] > 0.0
