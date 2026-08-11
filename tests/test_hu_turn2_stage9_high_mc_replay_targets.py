from ofc_regular.prepare_hu_turn2_stage9_high_mc_replay_targets import (
    hard_miss_target,
    hard_negative_target,
    near_boundary_target,
    select_stage9_replay_targets,
)


def test_hard_miss_target_replays_oracle_best_vs_baseline():
    row = {
        "state_index": 7,
        "sample_id": "s7",
        "split": "test",
        "seat": "first",
        "baseline_index": 2,
        "oracle_best_index": 5,
        "oracle_best_delta_vs_baseline": 1.5,
        "hard_miss_regret": 0.75,
    }

    target = hard_miss_target(row)

    assert target["reason"] == "stage9_hard_miss_oracle_best"
    assert target["baseline_action_local_index"] == 2
    assert target["candidate_action_local_index"] == 5
    assert target["teacher_best_action_local_index"] == 5
    assert target["teacher_gain_mean_current_mc512"] == 1.5


def test_hard_negative_target_replays_model_candidate_vs_baseline():
    row = {
        "state_index": 8,
        "split": "val",
        "seat": "second",
        "baseline_index": 1,
        "oracle_best_index": 4,
        "hard_negative_index": 3,
        "hard_negative_delta_vs_baseline": -0.5,
        "hard_negative_loss_vs_baseline": 0.5,
    }

    target = hard_negative_target(row)

    assert target["reason"] == "stage9_model_topk_hard_negative"
    assert target["baseline_action_local_index"] == 1
    assert target["candidate_action_local_index"] == 3
    assert target["teacher_best_action_local_index"] == 4
    assert target["teacher_gain_mean_current_mc512"] == -0.5


def test_near_boundary_target_replays_oracle_check():
    row = {
        "state_index": "9",
        "split": "test",
        "baseline_index": "0",
        "oracle_best_index": "2",
        "oracle_best_delta_vs_baseline": "0.4",
        "top5_oracle_regret": "0.1",
    }

    target = near_boundary_target(row)

    assert target["reason"] == "stage9_top5_near_boundary_oracle_check"
    assert target["candidate_action_local_index"] == 2
    assert target["teacher_best_action_local_index"] == 2


def test_select_stage9_replay_targets_filters_splits_and_limits():
    hard_misses = [
        {"state_index": 1, "split": "train", "baseline_index": 0, "oracle_best_index": 1, "hard_miss_regret": 2.0},
        {"state_index": 2, "split": "test", "baseline_index": 0, "oracle_best_index": 1, "hard_miss_regret": 1.0},
    ]
    hard_negatives = [
        {
            "state_index": 3,
            "split": "test",
            "baseline_index": 0,
            "oracle_best_index": 1,
            "hard_negative_index": 2,
            "hard_negative_loss_vs_baseline": 0.5,
        }
    ]
    state_rows = [
        {
            "state_index": 4,
            "split": "test",
            "baseline_index": 0,
            "oracle_best_index": 2,
            "top5_oracle_regret": 0.2,
        },
        {
            "state_index": 5,
            "split": "test",
            "baseline_index": 0,
            "oracle_best_index": 2,
            "top5_oracle_regret": 0.8,
        },
    ]

    targets, audit = select_stage9_replay_targets(
        hard_miss_rows=hard_misses,
        hard_negative_rows=hard_negatives,
        state_rows=state_rows,
        splits={"test"},
        hard_miss_limit=5,
        hard_negative_limit=5,
        near_boundary_limit=5,
        near_boundary_max_regret=0.25,
    )

    assert [target["state_index"] for target in targets] == [2, 3, 4]
    assert {row["group"] for row in audit if row["group_type"] == "reason"} == {
        "stage9_hard_miss_oracle_best",
        "stage9_model_topk_hard_negative",
        "stage9_top5_near_boundary_oracle_check",
    }
