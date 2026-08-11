import numpy as np

from ofc_regular.prepare_hu_turn2_stage9_high_mc_training_rows import (
    build_high_mc_training_rows,
    candidate_index,
)


def test_high_mc_training_rows_label_positive_and_negative():
    offsets = np.asarray([0, 4, 10], dtype=np.int64)
    candidates = {
        "pos": {
            "candidate_id": "pos",
            "state_index": "0",
            "candidate_action_local_index": "2",
            "split": "test",
            "reason": "stage9_hard_miss_oracle_best",
        },
        "neg": {
            "candidate_id": "neg",
            "state_index": "1",
            "candidate_action_local_index": "3",
            "split": "val",
            "reason": "stage9_model_topk_hard_negative",
        },
    }
    results = [
        {
            "candidate_id": "pos",
            "replay_status": "success",
            "gain_mean": "2.0",
            "gain_stderr": "0.5",
            "lower_bound_90": "1.18",
            "lower_bound_95": "1.02",
            "candidate_ev": "4.0",
            "mc_n": "512",
        },
        {
            "candidate_id": "neg",
            "replay_status": "success",
            "gain_mean": "-1.5",
            "gain_stderr": "0.5",
            "lower_bound_90": "-2.32",
            "lower_bound_95": "-2.48",
            "candidate_ev": "-3.0",
            "mc_n": "512",
        },
    ]

    rows, audit, summary = build_high_mc_training_rows(
        offsets=offsets,
        candidates=candidates,
        results=results,
        positive_lcb="95",
        positive_weight=6.0,
        negative_weight=5.0,
        gray_weight=0.0,
        max_weight=8.0,
        include_gray=False,
    )

    assert summary["output_rows"] == 2
    assert summary["label_counts"] == {
        "high_mc_hard_negative": 1,
        "high_mc_lcb95_positive": 1,
    }
    assert rows[0]["action_row_index"] == 2
    assert rows[0]["label"] == "high_mc_lcb95_positive"
    assert rows[0]["target"] == 1
    assert rows[1]["action_row_index"] == 7
    assert rows[1]["label"] == "high_mc_hard_negative"
    assert rows[1]["target"] == 0
    assert len(audit) == 2


def test_high_mc_training_rows_skips_gray_by_default():
    offsets = np.asarray([0, 2], dtype=np.int64)
    candidates = {"gray": {"candidate_id": "gray", "state_index": "0", "candidate_action_local_index": "1"}}
    results = [
        {
            "candidate_id": "gray",
            "replay_status": "success",
            "gain_mean": "0.4",
            "gain_stderr": "0.5",
            "lower_bound_90": "-0.42",
            "lower_bound_95": "-0.58",
        }
    ]

    rows, _audit, summary = build_high_mc_training_rows(
        offsets=offsets,
        candidates=candidates,
        results=results,
        positive_lcb="90",
        positive_weight=6.0,
        negative_weight=6.0,
        gray_weight=1.0,
        max_weight=8.0,
        include_gray=False,
    )

    assert rows == []
    assert summary["skipped"] == {"gray": 1}


def test_high_mc_training_rows_rejects_invalid_action_index():
    offsets = np.asarray([0, 2], dtype=np.int64)
    candidates = {"bad": {"candidate_id": "bad", "state_index": "0", "candidate_action_local_index": "9"}}
    results = [{"candidate_id": "bad", "replay_status": "success", "gain_mean": "-1"}]

    rows, _audit, summary = build_high_mc_training_rows(
        offsets=offsets,
        candidates=candidates,
        results=results,
        positive_lcb="90",
        positive_weight=6.0,
        negative_weight=6.0,
        gray_weight=0.0,
        max_weight=8.0,
        include_gray=False,
    )

    assert rows == []
    assert summary["skipped"] == {"invalid_action_index": 1}


def test_candidate_index_accepts_replay_fallback_id():
    rows = [
        {
            "candidate_id": "original",
            "state_index": "7",
            "candidate_action_local_index": "3",
            "baseline_action_local_index": "1",
        }
    ]

    index = candidate_index(rows)

    assert index["original"] is rows[0]
    assert index["state7_cand3_base1_row0"] is rows[0]
