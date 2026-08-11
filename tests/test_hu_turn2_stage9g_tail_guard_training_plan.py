import json
from pathlib import Path


def _load_config():
    return json.loads(
        Path("configs/hu_turn2_stage9g_tail_guard_training_plan.json").read_text(
            encoding="utf-8"
        )
    )


def test_stage9g_tail_guard_plan_is_not_production_or_large_training_ready():
    config = _load_config()

    assert config["production_default"] is False
    assert config["production_p2_fixed"] is False
    assert config["full_replacement_enabled"] is False
    assert config["teacher_50k"] == "No-Go"
    assert config["t1_training"] == "No-Go"
    assert config["t3_continuation"]["profile"] == "stage7_m5_r10"
    assert config["hard_blocks_for_large_training"]["hard_negative_count_too_small"] == 21
    assert (
        config["hard_blocks_for_large_training"][
            "minimum_dedup_hard_negatives_before_large_training"
        ]
        == 50
    )


def test_stage9g_tail_guard_plan_uses_dedup_labels_for_training():
    config = _load_config()

    assert config["data"]["raw_replay_rows"] == 114
    assert config["data"]["dedup_label_rows"] == 89
    assert config["data"]["duplicate_source_rows"] == 25
    assert config["data"]["training_rows_excluding_gray"] == 62
    assert config["data"]["training_hard_negative"] == 5
    assert config["artifacts"]["dedup_labels"].endswith("stage9f_tail_guard_labels_dedup.csv")
    assert config["artifacts"]["training_rows"].endswith("stage9g_tail_guard_training_rows.jsonl")


def test_stage9g_hu_delta_smoke_is_offline_ranking_only():
    config = _load_config()
    hu_delta = config["smoke_results"]["hu_delta_plus_preconfirm_meta"]

    assert config["candidate_for_next_offline_ranking"]["feature_mode"] == (
        "hu_delta_plus_preconfirm_meta"
    )
    assert hu_delta["all_roc_auc"] > 0.9
    assert hu_delta["all_average_precision"] > 0.5
    assert hu_delta["all_top1_precision"] == 1.0
    assert hu_delta["threshold_0p7_selected_rows"] == 0
    assert hu_delta["runtime_gate_decision"] == "No-Go"
    assert config["next_step"]["name"] == "stage9g_more_high_mc_labels"
    assert config["next_step"]["decision"] == "Go"


def test_stage9g_expanded_mixed_replay_increases_but_does_not_clear_label_gate():
    config = _load_config()
    data = config["expanded_stage9f_mixed_data"]
    hu_delta = config["expanded_smoke_results"]["hu_delta_plus_preconfirm_meta"]

    assert data["target_rows"] == 317
    assert data["target_replay_ready"] == 317
    assert data["gcp_replay_rows"] == 317
    assert data["gcp_replay_failures"] == 0
    assert data["training_hard_negative"] == 21
    assert data["training_safe_control"] == 195
    assert data["training_hard_negative"] < (
        config["hard_blocks_for_large_training"][
            "minimum_dedup_hard_negatives_before_large_training"
        ]
    )
    assert hu_delta["all_roc_auc"] > 0.9
    assert hu_delta["test_roc_auc"] > 0.8
    assert hu_delta["threshold_0p6_precision"] >= 0.9
    assert hu_delta["threshold_0p7_selected_rows"] == 0
    assert hu_delta["runtime_gate_decision"] == "No-Go"
