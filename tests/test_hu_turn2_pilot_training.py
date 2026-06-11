import json

import numpy as np

from ofc_regular.analyze_hu_turn2_pilot_calibration import (
    auc_pr,
    auc_roc,
    binary_metrics,
    rankdata,
    threshold_metrics,
)
from ofc_regular.replay_hu_turn2_event_high_mc import (
    actions_by_original_index,
    build_replay_events,
)
from ofc_regular.analyze_hu_turn2_gate_c1d_threshold_repair import (
    threshold_passes_strategy as c1d_threshold_passes_strategy,
)
from ofc_regular.analyze_hu_turn2_gate_c1e_teacher_ev_cache import (
    c1e_split_for_row,
    gate_status as c1e_gate_status,
    missing_replay_fields,
)
from ofc_regular.analyze_hu_turn2_gate_c1f_expanded_calibration import (
    c2_decision as c1f_c2_decision,
    candidate_status as c1f_candidate_status,
)
from ofc_regular.analyze_hu_turn2_stage8_c2_small import (
    c2_status as c2_small_status,
    proxy_passes as c2_proxy_passes,
)
from ofc_regular.analyze_hu_turn2_gate_c1_followup import (
    classify_false_positive,
    margin_bucket_label,
    threshold_passes_strategy,
)
from ofc_regular.build_hu_turn2_pilot_feature_cache import (
    action_position,
    pilot_gate_label,
    resolve_input_files,
    split_states,
    state_metadata,
)
from ofc_regular.train_hu_turn2_pilot_model import (
    write_markdown as write_training_markdown,
)


def test_action_position_preserves_original_index_zero():
    actions = [
        {"original_index": 3, "placements": [["Ah", "top"]], "discards": ["2c"]},
        {"original_index": 0, "placements": [["Ah", "middle"]], "discards": ["2c"]},
    ]

    assert action_position(actions, {"original_index": 0}) == 1


def test_pilot_gate_label_uses_stricter_positive_rule():
    assert pilot_gate_label({"delta_best_vs_baseline": 0.40, "SE_delta_best_vs_baseline": 0.10}) == "positive"
    assert pilot_gate_label({"delta_best_vs_baseline": 0.40, "SE_delta_best_vs_baseline": 0.30}) == "gray"
    assert pilot_gate_label({"delta_best_vs_baseline": 0.05, "SE_delta_best_vs_baseline": 0.01}) == "negative"


def test_state_metadata_adds_predicted_bucket_alias():
    meta = state_metadata(
        {"sample_id": 1, "seat": "first", "delta_best_vs_baseline": 0.0, "best_margin": 0.5},
        run_bucket="predicted_high_regret_from_pool",
        state_index=3,
    )

    assert meta["run_bucket"] == "predicted_high_regret_from_pool"
    assert meta["bucket_group"] == "predicted_high_regret"
    assert meta["predicted_bucket"] == "predicted_high_regret"


def test_training_summary_title_uses_broad_state_count(tmp_path):
    path = tmp_path / "training_summary.md"
    summary = {
        "model_output": "models/hu_turn2_stage8.pt",
        "cache_dir": "cache",
        "device": "cpu",
        "epochs_ran": 1,
        "best_epoch": 1,
        "recommended_next_step": "validation only",
        "split_counts": {"train": 14000, "val": 3000, "test": 3000},
        "eval": {
            "val": {
                "ev_mae": 1.0,
                "avg_regret": 0.1,
                "top3_recall": 0.5,
                "delta_vs_baseline_mae": 1.0,
                "pairwise_ranking_accuracy": 0.5,
                "gate_accuracy_pos_neg": 0.5,
            },
            "test": {
                "ev_mae": 1.0,
                "avg_regret": 0.1,
                "top3_recall": 0.5,
                "delta_vs_baseline_mae": 1.0,
                "pairwise_ranking_accuracy": 0.5,
                "gate_accuracy_pos_neg": 0.5,
            },
        },
    }
    write_training_markdown(path, summary, [])
    text = path.read_text(encoding="utf-8")

    assert "HU T2 Stage8 Broad 20k MC512 Training" in text
    assert "Pilot 2,000" not in text
    assert "not comparable to the T3 Stage7" in text


def test_split_states_is_state_level_and_stratified():
    metadata = []
    for index in range(60):
        metadata.append(
            {
                "state_index": index,
                "bucket_group": "natural" if index % 2 == 0 else "predicted_high_regret",
                "seat": "first" if index % 3 == 0 else "second",
                "pilot_gate_label": ("positive", "gray", "negative")[index % 3],
            }
        )

    split = split_states(metadata, train_fraction=0.70, val_fraction=0.15, seed=7)

    assert split.shape == (60,)
    assert set(np.unique(split)).issubset({0, 1, 2})
    assert np.sum(split == 0) > np.sum(split == 1)
    assert np.sum(split == 0) > np.sum(split == 2)


def test_resolve_input_files_prefers_complete_bucket_sidecars(tmp_path):
    merged = tmp_path / "merged.jsonl"
    merged.write_text("{}\n{}\n{}\n{}\n{}\n", encoding="utf-8")
    for name in [
        "natural.jsonl",
        "predicted_high_regret_from_pool.jsonl",
        "predicted_low_margin_from_pool.jsonl",
        "predicted_teacher_disagreement_from_pool.jsonl",
        "random_off_policy.jsonl",
    ]:
        (tmp_path / name).write_text(json.dumps({"bucket": name}) + "\n", encoding="utf-8")

    resolved = resolve_input_files(merged, tmp_path)

    assert [bucket for _path, bucket in resolved] == [
        "natural",
        "predicted_high_regret_from_pool",
        "predicted_low_margin_from_pool",
        "predicted_teacher_disagreement_from_pool",
        "random_off_policy",
    ]


def test_binary_auc_metrics_rank_perfect_scores():
    labels = [0, 1, 0, 1]
    scores = [0.1, 0.9, 0.2, 0.8]

    metrics = binary_metrics(labels, scores, threshold=0.5)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert auc_roc(labels, scores) == 1.0
    assert auc_pr(labels, scores) == 1.0


def test_rankdata_averages_ties():
    ranks = rankdata([10.0, 20.0, 20.0, 30.0])

    assert np.allclose(ranks, [1.0, 2.5, 2.5, 4.0])


def test_threshold_metrics_counts_false_positive_and_loss():
    rows = [
        {
            "split": "test",
            "candidate_is_baseline": 0,
            "predicted_delta_vs_baseline": 0.8,
            "reference_margin_raw": 0.3,
            "gate_probability": 0.9,
            "actual_delta_candidate_vs_baseline": 1.2,
            "seat": "first",
            "bucket_group": "natural",
        },
        {
            "split": "test",
            "candidate_is_baseline": 0,
            "predicted_delta_vs_baseline": 0.9,
            "reference_margin_raw": 0.4,
            "gate_probability": 0.9,
            "actual_delta_candidate_vs_baseline": -0.5,
            "seat": "second",
            "bucket_group": "natural",
        },
        {
            "split": "test",
            "candidate_is_baseline": 1,
            "predicted_delta_vs_baseline": 2.0,
            "reference_margin_raw": 1.0,
            "gate_probability": 1.0,
            "actual_delta_candidate_vs_baseline": 0.0,
            "seat": "first",
            "bucket_group": "natural",
        },
    ]

    result = threshold_metrics(
        rows,
        split_name="test",
        min_delta=0.5,
        min_reference=0.25,
        gate_threshold=0.5,
        threshold_kind="unit",
    )

    assert result["override_count"] == 2
    assert result["false_positive_count"] == 1
    assert result["avg_false_positive_cost"] == 0.5


def test_gate_c1_followup_margin_bucket_labels():
    assert margin_bucket_label(1.9, (2.0, 2.5, 3.0), "pred") == "pred_lt_2"
    assert margin_bucket_label(2.2, (2.0, 2.5, 3.0), "pred") == "pred_2_to_2.5"
    assert margin_bucket_label(3.5, (2.0, 2.5, 3.0), "pred") == "pred_ge_3"


def test_gate_c1_followup_false_positive_classification_prefers_noise_when_within_se():
    row = {
        "actual_delta_candidate_vs_baseline": -0.2,
        "gain_stderr_proxy": 0.2,
        "predicted_delta_vs_baseline": 3.0,
        "reference_margin_raw": 0.7,
        "teacher_best_margin": 1.0,
        "run_bucket": "random_off_policy",
        "bucket_group": "random_off_policy",
        "seat": "second",
    }

    assert classify_false_positive(row) == "reference_or_teacher_noise"


def test_gate_c1_followup_threshold_strategy_blocks_baseline_and_random_source():
    strategy = {
        "min_margin": 2.5,
        "reference_min_margin": 0.0,
        "gate_threshold": 0.7,
        "blocked_source_groups": ["random_off_policy"],
    }
    row = {
        "candidate_is_baseline": 0,
        "predicted_delta_vs_baseline": 3.0,
        "reference_margin_raw": 0.1,
        "gate_probability": 0.9,
        "run_bucket": "random_off_policy",
        "bucket_group": "random_off_policy",
    }

    assert not threshold_passes_strategy(row, strategy)

    row["run_bucket"] = "natural"
    row["bucket_group"] = "natural"
    assert threshold_passes_strategy(row, strategy)

    row["candidate_is_baseline"] = 1
    assert not threshold_passes_strategy(row, strategy)


def test_high_mc_replay_uses_original_indices_from_local_actions():
    candidates = [
        {
            "state_index": "7",
            "sample_id": "42",
            "source_group": "random_off_policy",
            "seat": "second",
            "baseline_action_local_index": "1",
            "candidate_action_local_index": "0",
            "teacher_best_action_local_index": "2",
        }
    ]
    sample_rows = {
        7: {
            "sample_id": 42,
            "seat": "second",
            "board": {"top": ["Ah"], "middle": [], "bottom": []},
            "opponent_board": {"top": ["Kh"], "middle": [], "bottom": []},
            "dead_cards": [],
            "dealt": ["2c", "3d", "4h"],
            "future_rollout_seed": 99,
            "rollout_count": 512,
            "actions": [
                {"original_index": 20, "score": 1.0},
                {"original_index": 8, "score": 0.5},
                {"original_index": 3, "score": 1.2},
            ],
        }
    }

    events = build_replay_events(candidates, sample_rows)

    assert len(events) == 1
    event = events[0]
    assert event.replay_ready
    assert event.candidate_original_index == 20
    assert event.baseline_original_index == 8
    assert event.teacher_original_index == 3


def test_high_mc_replay_action_lookup_by_original_index():
    actions = [{"original_index": 5, "score": 1.0}, {"original_index": 2, "score": -1.0}]

    lookup = actions_by_original_index(actions)

    assert lookup[5]["score"] == 1.0
    assert lookup[2]["score"] == -1.0


def test_gate_c1d_source_position_filter_blocks_random_second_only():
    strategy = {
        "min_margin": 2.5,
        "reference_min_margin": 0.0,
        "gate_threshold": 0.7,
        "requires_current_lcb_1p96_positive": True,
        "blocked_source_position": [("random_off_policy", "second")],
    }
    row = {
        "candidate_is_baseline": 0,
        "predicted_delta_vs_baseline": 3.0,
        "reference_margin_raw": 0.0,
        "gate_probability": 0.9,
        "gain_lcb_1p96": 0.2,
        "run_bucket": "random_off_policy",
        "bucket_group": "random_off_policy",
        "seat": "second",
    }

    assert not c1d_threshold_passes_strategy(row, strategy)

    row["seat"] = "first"
    assert c1d_threshold_passes_strategy(row, strategy)


def test_gate_c1d_position_specific_second_threshold_is_stricter():
    strategy = {
        "min_margin": 2.5,
        "reference_min_margin": 0.0,
        "gate_threshold": 0.7,
        "requires_current_lcb_1p96_positive": True,
        "seat_thresholds": {
            "second": {"min_margin": 3.0, "reference_min_margin": 0.0, "gate_threshold": 0.9}
        },
    }
    row = {
        "candidate_is_baseline": 0,
        "predicted_delta_vs_baseline": 2.6,
        "reference_margin_raw": 0.0,
        "gate_probability": 0.8,
        "gain_lcb_1p96": 0.2,
        "run_bucket": "natural",
        "bucket_group": "natural",
        "seat": "first",
    }

    assert c1d_threshold_passes_strategy(row, strategy)

    row["seat"] = "second"
    assert not c1d_threshold_passes_strategy(row, strategy)

    row["predicted_delta_vs_baseline"] = 3.1
    row["gate_probability"] = 0.95
    assert c1d_threshold_passes_strategy(row, strategy)


def test_gate_c1e_split_keeps_policy_distribution_unbiased():
    assert c1e_split_for_row({"run_bucket": "natural", "bucket_group": "natural"}) == "unbiased"
    assert (
        c1e_split_for_row(
            {
                "run_bucket": "predicted_low_margin_from_pool",
                "bucket_group": "predicted_low_margin",
            }
        )
        == "enriched"
    )


def test_gate_c1e_replay_readiness_requires_original_indices():
    sample = {
        "sample_id": 1,
        "state_id": "s1",
        "seat": "first",
        "source": "natural",
        "source_bucket": "natural",
        "board": {"top": ["Ah"], "middle": [], "bottom": []},
        "opponent_board": {"top": ["Kh"], "middle": [], "bottom": []},
        "dead_cards": ["2c"],
        "dealt": ["3d", "4h", "5s"],
        "actions": [{"original_index": 7, "placements": [["3d", "top"]], "discards": ["4h"]}],
        "baseline_action": {"original_index": 7},
        "reference_action": {"original_index": 7},
        "best_action": 7,
        "future_rollout_seed": 99,
        "rollout_count": 512,
    }

    assert missing_replay_fields(sample) == []

    sample["actions"] = [{"placements": [["3d", "top"]], "discards": ["4h"]}]
    assert "actions[0].original_index" in missing_replay_fields(sample)


def test_gate_c1e_split_targets_are_warnings_not_hard_blockers():
    status, blockers, warnings = c1e_gate_status(
        total_rows=20_000,
        unbiased_rows=4_000,
        enriched_rows=16_000,
        replay_ready_rows=20_000,
        missing_replay_rows=0,
        missing_source_position_rows=0,
        unrecoverable_action_rows=0,
        target_min=20_000,
        target_max=30_000,
        target_unbiased=20_000,
        target_enriched=10_000,
    )

    assert status == "hard_pass"
    assert blockers == []
    assert "unbiased_rows_lt_20k" in warnings


def test_gate_c1e_replay_or_action_failures_remain_hard_blockers():
    status, blockers, warnings = c1e_gate_status(
        total_rows=20_000,
        unbiased_rows=20_000,
        enriched_rows=10_000,
        replay_ready_rows=19_999,
        missing_replay_rows=1,
        missing_source_position_rows=1,
        unrecoverable_action_rows=1,
        target_min=20_000,
        target_max=30_000,
        target_unbiased=20_000,
        target_enriched=10_000,
    )

    assert status == "blocked"
    assert "replay_ready_not_100pct" in blockers
    assert "missing_source_or_position" in blockers
    assert "unrecoverable_baseline_candidate_or_teacher_action" in blockers
    assert warnings == []


def test_gate_c1f_unbiased_zero_is_conditional_not_c2_go():
    row = {
        "fires": 33,
        "false_positive_rate": 0.0,
        "avg_gain": 1.5,
        "p95_loss": 0.0,
        "p99_loss": 0.0,
        "first_fires": 16,
        "second_fires": 17,
        "unbiased_fires": 0,
        "dominant_source_share": 0.6,
    }

    status, blockers = c1f_candidate_status(row)
    assert status == "conditional_source_biased"
    assert "unbiased_fires_zero" in blockers

    c2_status, c2_blockers = c1f_c2_decision(
        [
            {
                "seat_swap_status": status,
                "unbiased_fires": 0,
                "blockers": blockers,
            }
        ]
    )
    assert c2_status == "No-Go"
    assert "unbiased_fires_zero_for_all_viable_candidates" in c2_blockers


def test_gate_c1f_negative_control_like_candidate_is_no_go():
    status, blockers = c1f_candidate_status(
        {
            "fires": 225,
            "false_positive_rate": 0.711,
            "avg_gain": -2.0,
            "p95_loss": 6.0,
            "p99_loss": 9.0,
            "first_fires": 100,
            "second_fires": 125,
            "unbiased_fires": 60,
            "dominant_source_share": 0.73,
        }
    )

    assert status == "no_go"
    assert "false_positive_rate_gt_10pct" in blockers
    assert "avg_gain_not_positive" in blockers


def test_c2_proxy_predicted_delta_gate_uses_runtime_available_fields_only():
    row = {
        "candidate_is_baseline": 0,
        "predicted_delta_vs_baseline": 2.6,
        "gate_probability": 0.8,
        "reference_margin_raw": 0.0,
    }
    config = {
        "proxy_family": "predicted_delta_gate",
        "min_margin": 2.5,
        "gate_threshold": 0.75,
        "reference_min_margin": 0.0,
    }

    assert c2_proxy_passes(row, config)
    row["gate_probability"] = 0.7
    assert not c2_proxy_passes(row, config)


def test_c2_requires_seat_swap_when_requested():
    status, blockers = c2_small_status(
        selected=[
            {
                "false_positive_rate": 0.0,
                "unbiased_fires": 10,
            }
        ],
        seat_swap_rows=[],
        run_seat_swap_enabled=True,
    )

    assert status == "No-Go"
    assert "seat_swap_not_completed" in blockers
