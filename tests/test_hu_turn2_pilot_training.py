import json
from types import SimpleNamespace

import numpy as np
import pytest

from ofc_regular.analyze_hu_turn2_pilot_calibration import (
    auc_pr,
    auc_roc,
    binary_metrics,
    rankdata,
    threshold_metrics,
)
from ofc_regular.replay_hu_turn2_event_high_mc import (
    actions_by_original_index,
    build_batched_config,
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
from ofc_regular.analyze_hu_turn2_stage8_c3_larger_seat_swap import (
    c3_row_status,
)
from ofc_regular.audit_hu_turn2_stage8_high_mc import (
    load_selected_states_jsonl,
    load_teacher_samples_jsonl,
)
from ofc_regular.analyze_hu_turn2_gate_c1_followup import (
    classify_false_positive,
    margin_bucket_label,
    threshold_passes_strategy,
)
from ofc_regular.prepare_hu_turn2_stage8_high_mc_replay_pack import (
    selected_slice,
)
from ofc_regular.build_hu_turn2_pilot_feature_cache import (
    FEATURE_CACHE_SCHEMA,
    FEATURE_VALUE_SCHEMA,
    REGULAR_RULES_DIGEST,
    action_position,
    pilot_gate_label,
    resolve_input_files,
    sample_fl_ev_14,
    scoring_objective_metadata,
    split_states,
    state_metadata,
    t3_continuation_metadata,
)
from ofc_regular.train_hu_turn2_pilot_model import (
    CURRENT_FL_EV_14,
    apply_candidate_target_overrides,
    build_loss_weight_metadata,
    candidate_pairwise_loss,
    checkpoint_stats,
    early_stop_metric_improved,
    load_candidate_generator_training_adjustments,
    load_candidate_pairwise_training_labels,
    load_initial_checkpoint_payload,
    load_cache,
    load_candidate_generator_training_weights,
    scoring_metadata_status,
    split_eval,
    teacher_mc_label_from_distribution,
    t3_margin_metadata,
    training_t3_continuation_metadata,
    write_markdown as write_training_markdown,
)


def test_action_position_preserves_original_index_zero():
    actions = [
        {"original_index": 3, "placements": [["Ah", "top"]], "discards": ["2c"]},
        {"original_index": 0, "placements": [["Ah", "middle"]], "discards": ["2c"]},
    ]

    assert action_position(actions, {"original_index": 0}) == 1

    with pytest.raises(ValueError, match="original_index disagrees"):
        action_position(
            actions,
            {
                "original_index": 0,
                "placements": [["Ah", "top"]],
                "discards": ["2c"],
            },
        )


def test_pilot_gate_label_uses_stricter_positive_rule():
    assert pilot_gate_label({"delta_best_vs_baseline": 0.40, "SE_delta_best_vs_baseline": 0.10}) == "positive"
    assert pilot_gate_label({"delta_best_vs_baseline": 0.40, "SE_delta_best_vs_baseline": 0.30}) == "gray"
    assert pilot_gate_label({"delta_best_vs_baseline": 0.05, "SE_delta_best_vs_baseline": 0.01}) == "negative"


def test_state_metadata_adds_predicted_bucket_alias():
    meta = state_metadata(
        {
            "sample_id": 1,
            "seat": "first",
            "turn": "T2",
            "to_act_order": "first",
            "board": {
                "top": ["Qh"],
                "middle": ["Kh", "Kd", "6c"],
                "bottom": ["9c", "9d", "9s"],
            },
            "opponent_board": {
                "top": ["2h"],
                "middle": ["3h", "4h", "5h"],
                "bottom": ["7h", "8h", "Th"],
            },
            "dealt": ["Qs", "Ah", "7d"],
            "hero_private_discards": ["2c"],
            "visible_dead_cards": [
                "2h", "3h", "4h", "5h", "7h", "8h", "Th", "2c"
            ],
            "delta_best_vs_baseline": 0.0,
            "best_margin": 0.5,
            "t3_continuation": "stage3_reference_default",
            "continuation_policy_T3": "Stage3_HU_reference_default",
        },
        run_bucket="predicted_high_regret_from_pool",
        state_index=3,
    )

    assert meta["run_bucket"] == "predicted_high_regret_from_pool"
    assert meta["bucket_group"] == "predicted_high_regret"
    assert meta["predicted_bucket"] == "predicted_high_regret"
    assert meta["t3_continuation"] == "stage3_reference_default"
    assert meta["continuation_policy_T3"] == "Stage3_HU_reference_default"


def test_feature_cache_loader_rejects_v1_and_manifest_mismatch(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps({"schema": "hu_turn2_pilot_feature_cache_v1"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema mismatch"):
        load_cache(tmp_path)

    metadata = {
        "schema": FEATURE_CACHE_SCHEMA,
        "observation_schema": "regular_ofc_actor_observation_v1",
        "policy_feature_sample_schema": "regular_ofc_policy_feature_sample_v1",
        "action_key_schema": "regular_ofc_action_key_v1",
        "feature_value_schema": FEATURE_VALUE_SCHEMA,
        "rules_digest": REGULAR_RULES_DIGEST,
        "feature_dim": 1076,
        "feature_dtype": "float32",
        "state_count": 1,
        "action_count": 1,
        "input_files": [],
        "observation_fingerprint_digest": "0" * 64,
        "legal_action_mapping_digest": "1" * 64,
        "cache_manifest_digest": "2" * 64,
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest digest mismatch"):
        load_cache(tmp_path)


def test_feature_cache_and_training_preserve_t3_continuation_metadata():
    rows = [
        {"t3_continuation": "stage3_reference_default", "continuation_policy_T3": "Stage3_HU_reference_default"},
        {"t3_continuation": "stage3_reference_default", "continuation_policy_T3": "Stage3_HU_reference_default"},
    ]
    metadata = t3_continuation_metadata(rows)

    assert metadata["t3_continuation"] == "stage3_reference_default"
    assert metadata["continuation_policy_T3"] == "Stage3_HU_reference_default"
    assert metadata["t3_continuation_counts"] == {"stage3_reference_default": 2}
    assert t3_margin_metadata(metadata) == (0.0, 0.0)
    assert training_t3_continuation_metadata({"metadata": {"t3_continuation_metadata": metadata}}) == metadata


def test_training_t3_metadata_falls_back_to_state_metadata_and_stage7_margins():
    cache = {
        "metadata": {},
        "state_metadata": [
            {"t3_continuation": "stage7_m5_r10", "continuation_policy_T3": "Stage7_candidate_A_m5_r10"}
        ],
    }
    metadata = training_t3_continuation_metadata(cache)

    assert metadata["t3_continuation"] == "stage7_m5_r10"
    assert metadata["continuation_policy_T3"] == "Stage7_candidate_A_m5_r10"
    assert t3_margin_metadata(metadata) == (5.0, 10.0)


def test_candidate_generator_training_weights_boost_action_and_state(tmp_path):
    path = tmp_path / "candidate_rows.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"state_index": 1, "action_row_index": 4, "label": "missed_oracle_positive", "weight": 4.0}),
                json.dumps({"state_index": 1, "action_row_index": 5, "label": "model_topk_hard_negative", "weight": 3.0}),
                json.dumps({"state_index": 99, "action_row_index": 999, "label": "bad", "weight": 2.0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cache = {"metadata": {"action_count": 8, "state_count": 3}}

    action_weights, state_weights, metadata = load_candidate_generator_training_weights(cache, [path])

    assert action_weights is not None
    assert state_weights is not None
    assert metadata is not None
    assert action_weights[4] == 4.0
    assert action_weights[5] == 3.0
    assert state_weights[1] == 4.0
    assert metadata["applied_rows"] == 2
    assert metadata["invalid_rows"] == 1
    assert metadata["label_counts"]["missed_oracle_positive"] == 1


def test_candidate_generator_training_adjustments_override_high_mc_targets(tmp_path):
    path = tmp_path / "candidate_rows.jsonl"
    path.write_text(
        json.dumps(
            {
                "state_index": 1,
                "action_row_index": 4,
                "label": "high_mc_hard_negative",
                "weight": 6.0,
                "target_ev": -12.5,
                "delta_vs_baseline": -3.25,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cache = {"metadata": {"action_count": 8, "state_count": 3}}

    action_weights, state_weights, overrides, metadata = load_candidate_generator_training_adjustments(cache, [path])

    assert action_weights is not None
    assert state_weights is not None
    assert overrides is not None
    assert action_weights[4] == 6.0
    assert state_weights[1] == 4.0
    assert overrides[4, 0] == -12.5
    assert overrides[4, 1] == -3.25
    assert np.isnan(overrides[4, 2])
    assert metadata["target_override_rows"] == 1
    assert metadata["target_override_cells"] == 2


def test_candidate_generator_training_adjustments_does_not_override_low_mc_targets(tmp_path):
    path = tmp_path / "candidate_rows.jsonl"
    path.write_text(
        json.dumps(
            {
                "state_index": 1,
                "action_row_index": 4,
                "label": "missed_oracle_positive",
                "weight": 2.0,
                "target_ev": 99.0,
                "delta_vs_baseline": 9.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cache = {"metadata": {"action_count": 8, "state_count": 3}}

    action_weights, state_weights, overrides, metadata = load_candidate_generator_training_adjustments(cache, [path])

    assert action_weights is not None
    assert state_weights is not None
    assert overrides is None
    assert action_weights[4] == 2.0
    assert state_weights[1] == 2.0
    assert metadata["target_override_rows"] == 0
    assert metadata["target_override_cells"] == 0


def test_candidate_pairwise_training_labels_use_only_high_mc_rows(tmp_path):
    path = tmp_path / "candidate_rows.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "state_index": 0,
                        "action_row_index": 2,
                        "label": "high_mc_lcb90_positive",
                        "weight": 8.0,
                    }
                ),
                json.dumps(
                    {
                        "state_index": 0,
                        "action_row_index": 1,
                        "label": "high_mc_hard_negative",
                        "weight": 6.0,
                    }
                ),
                json.dumps(
                    {
                        "state_index": 0,
                        "action_row_index": 0,
                        "label": "missed_oracle_positive",
                        "weight": 9.0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cache = {"metadata": {"action_count": 4, "state_count": 2}}

    positive, negative, metadata = load_candidate_pairwise_training_labels(cache, [path])

    assert positive is not None
    assert negative is not None
    assert positive.tolist() == [0.0, 0.0, 8.0, 0.0]
    assert negative.tolist() == [0.0, 6.0, 0.0, 0.0]
    assert metadata["positive_rows"] == 1
    assert metadata["negative_rows"] == 1


def test_candidate_pairwise_loss_pushes_positive_above_topk_and_negative_below_baseline():
    torch = pytest.importorskip("torch")
    groups = [(0, 0, 4)]
    action_indices = np.asarray([0, 1, 2, 3], dtype=np.int64)
    positive = np.asarray([0.0, 0.0, 8.0, 0.0], dtype=np.float32)
    negative = np.asarray([0.0, 6.0, 0.0, 0.0], dtype=np.float32)
    baseline = np.asarray([0], dtype=np.int64)

    bad_pred = torch.tensor([0.5, 0.8, 0.2, 0.7], dtype=torch.float32)
    good_pred = torch.tensor([0.5, -0.2, 1.2, 0.7], dtype=torch.float32)
    bad_loss = candidate_pairwise_loss(
        torch,
        bad_pred,
        groups,
        action_indices,
        positive,
        negative,
        baseline,
        margin=0.1,
        topk=2,
    )
    good_loss = candidate_pairwise_loss(
        torch,
        good_pred,
        groups,
        action_indices,
        positive,
        negative,
        baseline,
        margin=0.1,
        topk=2,
    )

    assert float(bad_loss) > float(good_loss)
    assert float(good_loss) == 0.0


def test_apply_candidate_target_overrides_only_replaces_finite_cells():
    targets = np.zeros((3, 4), dtype=np.float32)
    overrides = np.full((3, 4), np.nan, dtype=np.float32)
    overrides[1, 0] = 7.5
    overrides[1, 1] = -2.0

    adjusted = apply_candidate_target_overrides(targets, overrides)

    assert adjusted[1, 0] == 7.5
    assert adjusted[1, 1] == -2.0
    assert adjusted[1, 2] == 0.0
    assert targets[1, 0] == 0.0


def test_checkpoint_stats_validates_expected_shapes():
    payload = {
        "feature_mean": np.zeros(3, dtype=np.float32),
        "feature_scale": np.ones(3, dtype=np.float32),
        "target_mean": np.zeros(4, dtype=np.float32),
        "target_scale": np.ones(4, dtype=np.float32),
    }

    stats = checkpoint_stats(payload, feature_dim=3)

    assert stats["feature_mean"].shape == (3,)
    assert stats["target_scale"].shape == (4,)


def test_load_initial_checkpoint_rejects_architecture_mismatch(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "model.pt"
    torch.save(
        {
            "model_kind": "hu_turn2_pilot_multihead_mlp",
            "feature_dim": 3,
            "hidden_layer_sizes": [8, 4],
            "state_dict": {},
            "feature_mean": np.zeros(3, dtype=np.float32),
            "feature_scale": np.ones(3, dtype=np.float32),
            "target_mean": np.zeros(4, dtype=np.float32),
            "target_scale": np.ones(4, dtype=np.float32),
        },
        path,
    )

    with pytest.raises(ValueError, match="hidden_layer_sizes"):
        load_initial_checkpoint_payload(torch, path, feature_dim=3, hidden_layers=(8,))


def test_split_eval_reports_candidate_generator_topk_metrics():
    cache = {
        "offsets": np.asarray([0, 6], dtype=np.int64),
        "baseline_action_index": np.asarray([0], dtype=np.int64),
        "gate_label_id": np.asarray([1], dtype=np.int64),
    }
    targets = np.zeros((6, 4), dtype=np.float32)
    targets[:, 0] = np.asarray([0.0, 5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    predictions = np.zeros((6, 5), dtype=np.float32)
    predictions[:, 0] = np.asarray([10.0, 6.0, 9.0, 8.0, 7.0, 5.0], dtype=np.float32)

    row = split_eval(cache, predictions, targets, "test", np.asarray([0], dtype=np.int64))

    assert row["top1_accuracy"] == 0.0
    assert row["top3_recall"] == 0.0
    assert row["top5_recall"] == 1.0
    assert row["top5_avg_regret"] == 0.0
    assert row["top10_recall"] == 1.0


def test_early_stop_metric_direction_matches_metric_type():
    assert early_stop_metric_improved("val_top5_recall", 0.8, 0.7)
    assert not early_stop_metric_improved("val_top5_recall", 0.6, 0.7)
    assert early_stop_metric_improved("val_top5_avg_regret", 0.2, 0.3)
    assert not early_stop_metric_improved("val_top5_avg_regret", 0.4, 0.3)


def test_event_high_mc_batched_config_uses_stage3_default_and_stage7_opt_in():
    base = {
        "mc_samples": 2048,
        "stage3_feature_encoder_mode": "rust_direct",
        "disable_stage3_feature_fast_path": False,
        "batched_continuation_batch_size": 8192,
        "disable_continuation_cache": False,
        "hu_turn3_stage7_reference_model": "stage3-reference.pt",
        "hu_turn3_stage7_model": "stage7.pt",
    }

    stage3 = build_batched_config(SimpleNamespace(**base, t3_continuation="stage3_reference_default"))
    stage7 = build_batched_config(SimpleNamespace(**base, t3_continuation="stage7_m5_r10"))

    assert stage3.stage7_enabled is False
    assert stage3.hu_turn3_min_margin == 0.0
    assert stage3.hu_turn3_reference_min_margin == 0.0
    assert stage7.stage7_enabled is True
    assert stage7.hu_turn3_min_margin == 5.0
    assert stage7.hu_turn3_reference_min_margin == 10.0


def test_feature_cache_infers_fl_ev_from_final_turn_profile_metadata():
    sample = {
        "profiling": {
            "_final_turn_slow_states": [
                {
                    "metadata": {
                        "canonical_key": [
                            "final_turn_exact_v1",
                            "regular_ofc_v1",
                            "regular_ofc_v1",
                            [],
                            [],
                            [],
                            [],
                            "final_turn",
                            "hero",
                            "first",
                            "first",
                            False,
                            [[14, 10.227020614683454]],
                        ]
                    }
                }
            ]
        }
    }

    assert sample_fl_ev_14(sample) == 10.227020614683454
    metadata = scoring_objective_metadata([(sample, "natural")])

    assert abs(metadata["fl_ev_14"] - 10.227020614683454) < 1e-9
    assert metadata["fl_ev_14_status"] == "unique"


def test_feature_cache_prefers_top_level_fl_ev_metadata():
    sample = {
        "fl_ev_14": 10.227020614683454,
        "fl_ev": {"14": 10.227020614683454},
    }

    metadata = scoring_objective_metadata([(sample, "natural")])

    assert metadata["fl_ev_14"] == 10.227020614683454
    assert metadata["fl_ev_14_values"] == {"10.227020614683454": 1}


def test_training_scoring_status_rejects_missing_and_old_fl_ev():
    missing = scoring_metadata_status({})
    legacy_chain = scoring_metadata_status(
        {"scoring_objective": {"fl_ev_14": 12.196164}}
    )
    # The June direct estimate, superseded on 2026-08-03: caches labelled under
    # it are now as unusable for training as the older chain estimate.
    superseded_june = scoring_metadata_status(
        {"scoring_objective": {"fl_ev_14": 10.227020614683454}}
    )
    current = scoring_metadata_status(
        {"scoring_objective": {"fl_ev_14": CURRENT_FL_EV_14}}
    )

    assert missing["training_allowed"] is False
    for stale in (legacy_chain, superseded_june):
        assert stale["status"] == "mismatch"
        assert stale["training_allowed"] is False
    assert current["status"] == "match"
    assert current["training_allowed"] is True


def test_training_summary_title_uses_broad_state_count(tmp_path):
    path = tmp_path / "training_summary.md"
    summary = {
        "model_output": "models/hu_turn2_stage8.pt",
        "cache_dir": "cache",
        "device": "cpu",
        "epochs_ran": 1,
        "best_epoch": 1,
        "recommended_next_step": "validation only",
        "teacher_rollout_count_distribution": {"512": 20000},
        "teacher_mc_label": "MC512",
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
    assert "teacher rollout counts" in text


def test_training_summary_title_uses_mixed_teacher_mc_label(tmp_path):
    path = tmp_path / "training_summary.md"
    summary = {
        "model_output": "models/hu_turn2_stage8.pt",
        "cache_dir": "cache",
        "device": "cpu",
        "epochs_ran": 1,
        "best_epoch": 1,
        "recommended_next_step": "validation only",
        "teacher_rollout_count_distribution": {"16": 5000, "512": 18},
        "teacher_mc_label": teacher_mc_label_from_distribution({"16": 5000, "512": 18}),
        "split_counts": {"train": 3499, "val": 759, "test": 760},
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

    assert "HU T2 Stage8 Broad 5018 mixed MC16/MC512 Training" in text
    assert "5018 MC512 Training" not in text


def test_auxiliary_source_bucket_zeroes_regression_and_ranking_weights():
    metadata = [
        {"source_bucket": "natural"},
        {"source_bucket": "topk_hard_negative_replay"},
        {"source_bucket": "teacher_disagreement"},
    ]

    weights, summary = build_loss_weight_metadata(
        metadata,
        auxiliary_source_buckets=["topk_hard_negative_replay"],
        auxiliary_regression_weight=0.0,
        auxiliary_ranking_weight=0.0,
        auxiliary_listwise_weight=0.0,
        auxiliary_gate_weight=1.0,
    )

    assert weights["regression"].tolist() == [1.0, 0.0, 1.0]
    assert weights["ranking"].tolist() == [1.0, 0.0, 1.0]
    assert weights["listwise"].tolist() == [1.0, 0.0, 1.0]
    assert weights["gate"].tolist() == [1.0, 1.0, 1.0]
    assert summary["auxiliary_state_count"] == 1
    assert summary["auxiliary_source_counts"] == {"topk_hard_negative_replay": 1}


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

    resolved = resolve_input_files([merged], tmp_path)

    assert [bucket for _path, bucket in resolved] == [
        "natural",
        "predicted_high_regret_from_pool",
        "predicted_low_margin_from_pool",
        "predicted_teacher_disagreement_from_pool",
        "random_off_policy",
    ]


def test_resolve_input_files_combines_repeated_inputs(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text("{}\n", encoding="utf-8")
    second.write_text("{}\n", encoding="utf-8")

    resolved = resolve_input_files([first, second], None)

    assert resolved == [(first, "first"), (second, "second")]


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
            "hero_private_discards": ["5s"],
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
        "visible_dead_cards": ["Kh", "2c"],
        "hero_private_discards": ["2c"],
        "opponent_private_discards": ["3c"],
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


def test_c3_row_status_requires_positive_larger_seat_swap():
    row = {
        "paired_seeds": 3000,
        "aggregate_ev_per_hand": 0.01,
        "ci95_low_seed_means": -0.01,
        "avg_gain_on_override": 1.0,
        "false_positive_override_rate": 0.03,
        "p95_loss": 0.0,
        "override_count": 10,
    }

    assert c3_row_status(row) == ("conditional_go", "")

    row["aggregate_ev_per_hand"] = -0.01
    status, blockers = c3_row_status(row)
    assert status == "no_go"
    assert "seat_swap_ev_not_positive" in blockers


def test_c4_replay_pack_slice_and_loaders(tmp_path):
    selected_path = tmp_path / "selected.jsonl"
    selected_path.write_text(
        "\n".join(
            [
                json.dumps({"state_index": 10, "selection_order": 0}),
                json.dumps({"state_index": 20, "selection_order": 1}),
                json.dumps({"state_index": 30, "selection_order": 2}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    sample_path = tmp_path / "samples.jsonl"
    sample_path.write_text(
        "\n".join(
            [
                json.dumps({"state_index": 20, "sample": {"board": "b20"}}),
                json.dumps({"state_index": 30, "sample": {"board": "b30"}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    selected = load_selected_states_jsonl(selected_path)
    assert selected_slice(selected, offset=1, count=2) == selected[1:3]

    samples = load_teacher_samples_jsonl(sample_path, {20, 30})
    assert samples == {20: {"board": "b20"}, 30: {"board": "b30"}}
