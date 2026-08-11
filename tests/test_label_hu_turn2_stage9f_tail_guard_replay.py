from pathlib import Path

from ofc_regular.label_hu_turn2_stage9f_tail_guard_replay import (
    build_label_row,
    build_labels,
    dedupe_label_rows,
    label_for_row,
    replay_event_key,
)


def replay_row(**overrides):
    row = {
        "target_id": "t1",
        "target_group": "tail_loss",
        "config_id": "stage9f_cse2_csemax2_firstseat",
        "seed": "1",
        "hand_id": "1",
        "seat": "first",
        "mc_n": "512",
        "baseline_action_index": "3",
        "candidate_action_index": "8",
        "gain_mean": "1.0",
        "gain_stderr": "0.2",
        "gain_lower90": "0.67",
        "gain_lower95": "0.6",
        "input_realized_delta": "-10",
        "input_tail_loss_label": "1",
        "input_severe_tail_loss_label": "1",
        "input_safe_positive_label": "0",
        "replay_status": "success",
    }
    row.update(overrides)
    return row


def test_label_for_row_marks_hard_negative_by_mean():
    assert (
        label_for_row(
            replay_row(gain_mean="-0.3", gain_lower95="-1.0"),
            hard_negative_mean_threshold=-0.25,
            safe_positive_lcb95_threshold=0.0,
            safe_positive_mean_threshold=0.5,
        )
        == "hard_negative"
    )


def test_label_for_row_requires_lcb_for_safe_positive():
    assert (
        label_for_row(
            replay_row(gain_mean="1.0", gain_lower95="-0.1"),
            hard_negative_mean_threshold=-0.25,
            safe_positive_lcb95_threshold=0.0,
            safe_positive_mean_threshold=0.5,
        )
        == "gray"
    )
    assert (
        label_for_row(
            replay_row(gain_mean="1.0", gain_lower95="0.1"),
            hard_negative_mean_threshold=-0.25,
            safe_positive_lcb95_threshold=0.0,
            safe_positive_mean_threshold=0.5,
        )
        == "safe_positive"
    )


def test_build_label_row_sets_training_use_and_weight():
    row = build_label_row(
        replay_row(gain_mean="-1.2", gain_lower95="-2.0"),
        hard_negative_mean_threshold=-0.25,
        safe_positive_lcb95_threshold=0.0,
        safe_positive_mean_threshold=0.5,
        hard_negative_weight=5.0,
        safe_positive_weight=1.0,
        gray_weight=0.25,
        source_path=Path("replay.csv"),
    )

    assert row["high_mc_tail_guard_label"] == "hard_negative"
    assert row["high_mc_hard_negative_label"] == 1
    assert row["recommended_training_use"] == "tail_guard_hard_negative"
    assert row["high_mc_training_weight"] == 5.0


def test_build_labels_skips_failed_replay_rows():
    labels = build_labels(
        [
            (replay_row(target_id="ok"), Path("replay.csv")),
            (replay_row(target_id="fail", replay_status="failed"), Path("replay.csv")),
        ],
        hard_negative_mean_threshold=-0.25,
        safe_positive_lcb95_threshold=0.0,
        safe_positive_mean_threshold=0.5,
        hard_negative_weight=5.0,
        safe_positive_weight=1.0,
        gray_weight=0.25,
    )

    assert [row["target_id"] for row in labels] == ["ok"]


def test_replay_event_key_uses_action_pair():
    assert replay_event_key(replay_row()) == "stage9f_cse2_csemax2_firstseat:seed1:hand1:base3:cand8"


def test_dedupe_label_rows_merges_bucket_duplicates_and_or_input_flags():
    labels = build_labels(
        [
            (
                replay_row(
                    target_id="tail",
                    target_group="tail_loss",
                    input_tail_loss_label="1",
                    input_safe_positive_label="0",
                ),
                Path("replay.csv"),
            ),
            (
                replay_row(
                    target_id="boundary",
                    target_group="confirm_z_boundary",
                    input_tail_loss_label="0",
                    input_safe_positive_label="1",
                ),
                Path("replay.csv"),
            ),
        ],
        hard_negative_mean_threshold=-0.25,
        safe_positive_lcb95_threshold=0.0,
        safe_positive_mean_threshold=0.5,
        hard_negative_weight=5.0,
        safe_positive_weight=1.0,
        gray_weight=0.25,
    )

    deduped = dedupe_label_rows(labels)

    assert len(deduped) == 1
    assert deduped[0]["duplicate_source_rows"] == 2
    assert deduped[0]["target_group"] == "confirm_z_boundary+tail_loss"
    assert deduped[0]["input_tail_loss_label"] == 1
    assert deduped[0]["input_safe_positive_label"] == 1
