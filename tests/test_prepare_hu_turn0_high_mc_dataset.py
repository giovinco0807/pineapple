import json
from itertools import combinations, islice

import pytest

from ofc_regular.prepare_hu_turn0_high_mc_dataset import (
    prepare_hu_turn0_high_mc_dataset,
)
from ofc_regular.teacher import DEFAULT_FL_EV


def _sample(index: int, seat: str, *, future_samples: int = 32) -> dict:
    deck = [f"{rank}{suit}" for rank in "23456789TJQKA" for suit in "cdhs"]
    dealt = list(next(islice(combinations(deck, 5), index, index + 1)))
    digest = f"digest-{index}-{seat}"
    actions = []
    for action_index in range(3):
        delta = float(action_index)
        actions.append(
            {
                "action_index": action_index,
                "original_index": action_index,
                "score": delta,
                "ev": delta,
                "se": 0.25,
                "delta_vs_baseline": delta,
                "delta_se_vs_baseline": 0.2,
                "delta_z_vs_baseline": delta / 0.2,
                "rollout_count": future_samples,
                "common_random_future_digest": digest,
                "placements": [],
                "discards": [],
                "next_board": {"top": [], "middle": [], "bottom": []},
            }
        )
    return {
        "schema": "hu_turn0_stage1_teacher_v1",
        "phase": "hu_turn0_0card",
        "seat": seat,
        "player": 0 if seat == "first" else 1,
        "board": {"top": [], "middle": [], "bottom": []},
        "opponent_board": {"top": [], "middle": [], "bottom": []},
        "dead_cards": [],
        "visible_dead_cards": [],
        "dealt": dealt,
        "future_samples": future_samples,
        "total_legal_actions": 232,
        "evaluated_action_count": 3,
        "actions": actions,
        "baseline_action_index": 0,
        "best_action_index": 2,
        "delta_best_vs_baseline": 2.0,
        "delta_best_vs_baseline_se": 0.2,
        "common_random_future_digest": digest,
        "common_random_futures_verified": True,
        "replay_ready": True,
        "t1_continuation": "stage18_p1",
        "t2_continuation": "stage9f_p2",
        "t3_continuation": "stage7_m5_r10",
        # The preparer rejects rows labelled under any other FL EV, so this
        # tracks the reader rather than freezing a number.
        "fl_ev": {"14": DEFAULT_FL_EV[14]},
        "visibility_model": "hidden_discard",
        "discard_visibility": "own_private_only",
        "candidate_selector": {"mode": "candidate_model_topk", "topk": 60},
    }


def _write(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_prepare_high_mc_dataset_validates_and_stratifies(tmp_path):
    first_rows = [_sample(index, "first") for index in range(10)]
    second_rows = [_sample(100 + index, "second") for index in range(10)]
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    _write(first_path, first_rows)
    _write(second_path, second_rows)

    summary = prepare_hu_turn0_high_mc_dataset(
        inputs=[first_path, second_path],
        output_dir=tmp_path / "prepared",
        holdout_fraction=0.2,
        seed=42,
        min_future_samples=32,
        expected_evaluated_actions=3,
    )

    assert summary["unique_rows"] == 20
    assert summary["train_rows"] == 16
    assert summary["holdout_rows"] == 4
    assert summary["holdout_seat_counts"] == {"first": 2, "second": 2}
    assert summary["split_overlap"] == 0
    output_rows = [
        json.loads(line)
        for line in (tmp_path / "prepared" / "hu_turn0_high_mc_merged.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert all(row["baseline_action_row_index"] == 0 for row in output_rows)
    assert all(row["best_action_row_index"] == 2 for row in output_rows)
    assert summary["best_delta_lcb196_positive_rate"] == 1.0


def test_prepare_high_mc_dataset_prefers_higher_mc_duplicate(tmp_path):
    low = _sample(1, "first", future_samples=32)
    high = _sample(1, "first", future_samples=64)
    low_path = tmp_path / "low.jsonl"
    high_path = tmp_path / "high.jsonl"
    _write(low_path, [low])
    _write(high_path, [high])

    summary = prepare_hu_turn0_high_mc_dataset(
        inputs=[low_path, high_path],
        output_dir=tmp_path / "prepared",
        holdout_fraction=0.0,
        min_future_samples=32,
        expected_evaluated_actions=3,
    )

    assert summary["unique_rows"] == 1
    assert summary["duplicate_states_dropped"] == 1
    merged = json.loads(
        (tmp_path / "prepared" / "hu_turn0_high_mc_merged.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert merged["future_samples"] == 64
    assert merged["source"] == "hu_turn0_terminal_rollout_mc64"


def test_prepare_high_mc_dataset_rejects_broken_common_future_digest(tmp_path):
    row = _sample(1, "first")
    row["actions"][1]["common_random_future_digest"] = "different"
    input_path = tmp_path / "broken.jsonl"
    _write(input_path, [row])

    with pytest.raises(ValueError, match="future digests differ"):
        prepare_hu_turn0_high_mc_dataset(
            inputs=[input_path],
            output_dir=tmp_path / "prepared",
            holdout_fraction=0.0,
            min_future_samples=32,
            expected_evaluated_actions=3,
        )
