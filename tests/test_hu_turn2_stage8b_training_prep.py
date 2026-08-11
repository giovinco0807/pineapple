import csv
from types import SimpleNamespace

import numpy as np
import pytest

from ofc_regular.prepare_hu_turn2_stage8b_training import (
    LABEL_TO_ID,
    attach_stage8b_labels,
    selected_high_mc_states,
    topk_hard_negative_rows,
    whole_game_risk_target_rows,
)
from ofc_regular.analyze_hu_turn2_stage8b_training import (
    binary_counts,
    metric_from_counts,
    runtime_fires,
)
from ofc_regular.teacher import DEFAULT_FL_EV
from ofc_regular.evaluate_hu_turn2_stage8b_topk_mc_rerank import (
    DEFAULT_TOPK_RERANK_CONFIGS,
    _terminal_breakdown,
    aggregate_topk_seed_rows,
    cancellation_audit,
    cancellation_is_clean,
    cancellation_is_clean_for_config,
    conditional_override_metrics,
    has_safety_blocker,
    has_safety_blocker_for_config,
    parse_topk_configs,
    predicted_delta_safety_audit,
    realized_override_count,
    stage_b_delta,
    topk_conditional_go,
    write_go_nogo,
    write_summary,
)
from ofc_regular.train_hu_turn2_pilot_model import apply_stage8b_gate_labels, gate_loss
from ofc_regular.train_hu_turn2_pilot_model import validate_cache_scoring_for_training


def _row(**overrides):
    base = {
        "state_index": 0,
        "split": "test",
        "candidate_is_baseline": 0,
        "actual_delta_candidate_vs_baseline": 1.0,
        "candidate_loss": 0.0,
        "gain_stderr_proxy": 0.1,
        "gain_lcb_1p96": 0.804,
        "gain_lcb_1p64": 0.836,
        "predicted_delta_vs_baseline": 2.6,
        "gate_probability": 0.95,
        "reference_margin_raw": 0.0,
        "candidate_action_local_index": 2,
        "candidate_action_original_index": 8,
        "baseline_action_local_index": 1,
        "baseline_action_original_index": 5,
    }
    base.update(overrides)
    return base


def test_terminal_breakdown_reports_line_scoop_royalty_and_fl_components():
    hero = {
        "top": ["Qh", "Qs", "2d"],
        "middle": ["Kc", "Kd", "2c", "3c", "4h"],
        "bottom": ["Ah", "Ad", "Ac", "7s", "7d"],
    }
    opponent = {
        "top": ["9h", "9s", "3d"],
        "middle": ["Th", "Ts", "4d", "5s", "6c"],
        "bottom": ["Jh", "Js", "Jd", "8c", "8s"],
    }

    breakdown = _terminal_breakdown(hero, opponent)

    assert breakdown["hero_foul"] is False
    assert breakdown["opponent_foul"] is False
    assert breakdown["line_results"] == {"top": 1, "middle": 1, "bottom": 1}
    assert breakdown["line_score_delta"] == 3
    assert breakdown["scoop_delta"] == 3
    assert breakdown["royalty_delta"] == 3
    assert breakdown["hero_fl_entry"] is True
    assert breakdown["hero_fl_card_count"] == 14
    assert breakdown["opponent_fl_entry"] is False
    # 3 line + 3 scoop + 3 royalty, plus the hero's Fantasyland entry: the FL
    # term is named rather than folded into a literal so a re-measured constant
    # shows up here as an FL-EV change and not as an unexplained total.
    assert breakdown["terminal_score"] == pytest.approx(
        3 + 3 + 3 + DEFAULT_FL_EV[14]
    )


def test_stage8b_label_builder_marks_safe_positive_and_hard_negative():
    rows = [
        _row(state_index=1),
        _row(
            state_index=2,
            actual_delta_candidate_vs_baseline=-0.5,
            candidate_loss=0.5,
            gain_lcb_1p96=-0.7,
            gain_lcb_1p64=-0.66,
            predicted_delta_vs_baseline=3.0,
            gate_probability=0.96,
        ),
    ]

    attach_stage8b_labels(rows)

    assert rows[0]["safe_lcb196_label"] == "positive"
    assert rows[0]["safe_lcb196_gate_label_id"] == LABEL_TO_ID["positive"]
    assert rows[0]["hard_negative_label"] == 0
    assert rows[1]["safe_lcb196_label"] == "negative"
    assert rows[1]["safe_lcb196_gate_label_id"] == LABEL_TO_ID["negative"]
    assert rows[1]["hard_negative_label"] == 1
    assert rows[1]["stage8b_gate_weight"] > rows[0]["stage8b_gate_weight"]


def test_stage8b_label_builder_uses_high_mc_override():
    rows = [
        _row(
            state_index=7,
            actual_delta_candidate_vs_baseline=1.0,
            gain_lcb_1p96=0.8,
            predicted_delta_vs_baseline=3.0,
            gate_probability=0.96,
        )
    ]

    attach_stage8b_labels(
        rows,
        {
            7: {
                "high_mc_delta_candidate_vs_baseline": -0.2,
                "high_mc_lower95_candidate_vs_baseline": -0.6,
                "diagnosis": "false_positive_gate",
            }
        },
    )

    assert rows[0]["safe_lcb196_label"] == "negative"
    assert rows[0]["hard_negative_label"] == 1
    assert rows[0]["hard_negative_source"] == "high_mc_runtime_negative"
    assert rows[0]["high_mc_label_source"] == "mc4096"


def test_stage8b_selected_high_mc_states_dedupes_by_state_index():
    rows = [
        _row(state_index=10, actual_delta_candidate_vs_baseline=-0.4, candidate_loss=0.4, gain_lcb_1p96=-0.6),
        _row(state_index=10, actual_delta_candidate_vs_baseline=-0.5, candidate_loss=0.5, gain_lcb_1p96=-0.7),
        _row(state_index=11, actual_delta_candidate_vs_baseline=1.0, gain_lcb_1p96=0.5, predicted_delta_vs_baseline=1.0),
    ]
    attach_stage8b_labels(rows)

    selected = selected_high_mc_states(rows, limit=10)

    assert len({row["state_index"] for row in selected}) == len(selected)
    assert {row["state_index"] for row in selected} == {10, 11}


def test_topk_hard_negative_rows_mark_replay_required(tmp_path):
    path = tmp_path / "topk_hard_negatives.jsonl"
    path.write_text(
        '{"schema":"hu_turn2_stage8b_topk_hard_negative_v1","hand_seed":1,"replay_ready":true}\n',
        encoding="utf-8",
    )

    rows = topk_hard_negative_rows(path)

    assert len(rows) == 1
    assert rows[0]["selection_order"] == 0
    assert rows[0]["replay_origin_group"] == "topk_mc_false_positive"
    assert rows[0]["replay_source"] == "stage8b_topk_hard_negative_pack"
    assert rows[0]["requires_replay_before_training"] is True
    assert rows[0]["replay_source_path"] == str(path)


def test_topk_hard_negative_rows_accept_multiple_inputs(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text('{"hand_seed":1}\n', encoding="utf-8")
    second.write_text('{"hand_seed":2}\n{"hand_seed":3}\n', encoding="utf-8")

    rows = topk_hard_negative_rows([first, tmp_path / "missing.jsonl", second])

    assert [row["hand_seed"] for row in rows] == [1, 2, 3]
    assert [row["selection_order"] for row in rows] == [0, 1, 2]
    assert rows[0]["replay_source_path"] == str(first)
    assert rows[1]["replay_source_path"] == str(second)


def test_whole_game_risk_target_rows_require_separate_head(tmp_path):
    path = tmp_path / "counterfactual_loss_targets.jsonl"
    path.write_text(
        '{"schema":"hu_turn2_stage8b_counterfactual_loss_target_v1",'
        '"recommended_training_use":"whole_game_risk_only",'
        '"use_for_whole_game_risk_head":1,'
        '"use_for_local_ev_hard_negative":0,'
        '"realized_delta":-6.0}\n',
        encoding="utf-8",
    )

    rows = whole_game_risk_target_rows(path)

    assert len(rows) == 1
    assert rows[0]["replay_origin_group"] == "stage8b_topk_whole_game_loss"
    assert rows[0]["replay_source"] == "stage8b_topk_counterfactual_loss_targets"
    assert rows[0]["requires_separate_whole_game_risk_head"] is True
    assert rows[0]["do_not_use_as_local_ev_hard_negative"] is True


def test_apply_stage8b_gate_labels_overrides_cache_labels_and_weights(tmp_path):
    labels_path = tmp_path / "labels.csv"
    with labels_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["state_index", "safe_lcb196_gate_label_id", "stage8b_gate_weight"],
        )
        writer.writeheader()
        writer.writerow({"state_index": 0, "safe_lcb196_gate_label_id": 2, "stage8b_gate_weight": 1.5})
        writer.writerow({"state_index": 1, "safe_lcb196_gate_label_id": 0, "stage8b_gate_weight": 6.0})
    cache = {
        "state_metadata": [{}, {}],
        "gate_label_id": np.asarray([1, 1], dtype=np.int8),
    }

    apply_stage8b_gate_labels(
        cache,
        labels_path,
        label_column="safe_lcb196_gate_label_id",
        weight_column="stage8b_gate_weight",
    )

    assert cache["gate_label_id"].tolist() == [2, 0]
    assert cache["gate_label_weight"].tolist() == [1.5, 6.0]


def test_apply_stage8b_gate_labels_rejects_whole_game_risk_only_rows(tmp_path):
    labels_path = tmp_path / "risk_only_labels.csv"
    with labels_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "state_index",
                "safe_lcb196_gate_label_id",
                "stage8b_gate_weight",
                "recommended_training_use",
                "requires_separate_whole_game_risk_head",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "state_index": 0,
                "safe_lcb196_gate_label_id": 0,
                "stage8b_gate_weight": 6.0,
                "recommended_training_use": "whole_game_risk_only",
                "requires_separate_whole_game_risk_head": 1,
            }
        )
    cache = {
        "state_metadata": [{}],
        "gate_label_id": np.asarray([1], dtype=np.int8),
    }

    with pytest.raises(ValueError, match="whole-game risk-only"):
        apply_stage8b_gate_labels(
            cache,
            labels_path,
            label_column="safe_lcb196_gate_label_id",
            weight_column="stage8b_gate_weight",
        )


def test_validate_cache_scoring_for_training_rejects_missing_metadata_by_default():
    cache = {"metadata": {}}

    with pytest.raises(ValueError, match="scoring metadata"):
        validate_cache_scoring_for_training(cache)

    allowed = validate_cache_scoring_for_training(cache, allow_missing=True)
    assert allowed["training_allowed"] is True
    assert allowed["override"] == "allow_missing_scoring_metadata"


def test_gate_loss_accepts_state_weights():
    import torch

    logits = torch.tensor([0.0, 0.0, 0.0, 0.0])
    groups = [(0, 0, 2), (1, 2, 4)]
    labels = np.asarray([2, 0], dtype=np.int8)
    weights = np.asarray([1.0, 4.0], dtype=np.float32)

    weighted = gate_loss(torch, logits, groups, labels, negative_weight=1.0, gate_weights=weights)
    unweighted = gate_loss(torch, logits, groups, labels, negative_weight=1.0)

    assert weighted.item() == unweighted.item()


def test_stage8b_training_audit_binary_metrics_exclude_gray():
    rows = [
        {"stage8b_safe_lcb196_gate_label_id": 2, "safe_override_probability": 0.95},
        {"stage8b_safe_lcb196_gate_label_id": 2, "safe_override_probability": 0.20},
        {"stage8b_safe_lcb196_gate_label_id": 0, "safe_override_probability": 0.91},
        {"stage8b_safe_lcb196_gate_label_id": 0, "safe_override_probability": 0.10},
        {"stage8b_safe_lcb196_gate_label_id": 1, "safe_override_probability": 0.99},
    ]

    counts = binary_counts(rows, 0.90)
    metrics = metric_from_counts(counts)

    assert counts == {"tp": 1, "fp": 1, "tn": 1, "fn": 1, "pos": 2, "neg": 2, "gray": 1}
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5


def test_stage8b_runtime_fires_uses_safe_probability_and_rank_guard():
    row = {
        "candidate_is_baseline": 0,
        "predicted_delta_vs_baseline": 2.6,
        "safe_override_probability": 0.91,
        "stage8b_model_candidate_ev_rank": 2,
    }

    assert runtime_fires(row, min_delta=2.5, safe_threshold=0.90, rank_max=2)
    assert not runtime_fires(row, min_delta=2.75, safe_threshold=0.90, rank_max=2)
    assert not runtime_fires(row, min_delta=2.5, safe_threshold=0.95, rank_max=2)
    assert not runtime_fires(row, min_delta=2.5, safe_threshold=0.90, rank_max=1)


def test_topk_mc_rerank_config_parser_supports_runtime_variants():
    config = parse_topk_configs("k5/mc128/d0.5/se1.5/confirm256/cse2/cd1/scd1.5/seat=first/rank3/g0.9/pd0/bygate_delta")[0]

    assert config.top_k == 5
    assert config.mc_samples == 128
    assert config.min_delta == 0.5
    assert config.se_multiplier == 1.5
    assert config.confirm_mc_samples == 256
    assert config.confirm_se_multiplier == 2.0
    assert config.min_confirm_delta == 1.0
    assert config.second_min_confirm_delta == 1.5
    assert config.allowed_seats == ("first",)
    assert config.candidate_ev_rank_max == 3
    assert config.min_gate_probability == 0.9
    assert config.min_predicted_delta == 0.0
    assert config.topk_score == "gate_delta"


def test_topk_mc_rerank_default_configs_require_nonnegative_model_delta():
    configs = parse_topk_configs(DEFAULT_TOPK_RERANK_CONFIGS)

    assert configs
    assert all(config.min_predicted_delta == 0.0 for config in configs)


def test_topk_mc_rerank_aggregate_seed_rows_uses_topk_schema():
    rows = aggregate_topk_seed_rows(
        [
            {
                "config_id": "k3_mc64_d0.25_se0",
                "top_k": 3,
                "mc_samples": 64,
                "min_rerank_delta": 0.25,
                "se_multiplier": 0.0,
                "allowed_seats": "",
                "candidate_ev_rank_max": "",
                "min_gate_probability": "",
                "min_predicted_delta": "",
                "topk_score": "delta",
                "paired_seeds": 10,
                "max_paired_seeds": 50,
                "target_realized_overrides": 2,
                "target_realized_overrides_reached": True,
                "stop_reason": "target_realized_overrides_reached",
                "hands": 20,
                "ev_per_hand": 0.2,
                "paired_seed_wins": 6,
                "paired_seed_losses": 3,
                "paired_seed_ties": 1,
                "decision_count": 20,
                "override_count": 2,
                "realized_override_count": 2,
                "avg_rerank_delta_on_override": 1.0,
                "avg_confirm_delta_on_override": 3.0,
                "avg_realized_delta_on_override": -2.0,
            },
            {
                "config_id": "k3_mc64_d0.25_se0",
                "top_k": 3,
                "mc_samples": 64,
                "min_rerank_delta": 0.25,
                "se_multiplier": 0.0,
                "allowed_seats": "",
                "candidate_ev_rank_max": "",
                "min_gate_probability": "",
                "min_predicted_delta": "",
                "topk_score": "delta",
                "paired_seeds": 10,
                "max_paired_seeds": 50,
                "target_realized_overrides": 2,
                "target_realized_overrides_reached": False,
                "stop_reason": "max_games_reached",
                "hands": 20,
                "ev_per_hand": -0.1,
                "paired_seed_wins": 3,
                "paired_seed_losses": 6,
                "paired_seed_ties": 1,
                "decision_count": 20,
                "override_count": 1,
                "realized_override_count": 1,
                "avg_rerank_delta_on_override": 0.5,
                "avg_realized_delta_on_override": 4.0,
            },
        ]
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["paired_seeds"] == 20
    assert row["max_paired_seeds"] == 100
    assert row["target_realized_overrides"] == 2
    assert row["target_realized_overrides_reached_count"] == 1
    assert "target_realized_overrides_reached" in row["stop_reason_counts"]
    assert "max_games_reached" in row["stop_reason_counts"]
    assert row["hands"] == 40
    assert row["aggregate_ev_per_hand"] == 0.05
    assert row["override_count"] == 3
    assert row["realized_override_count"] == 3
    assert row["runtime_override_rate"] == 3 / 40
    assert row["avg_gain_on_override"] == (-2.0 + -2.0 + 4.0) / 3
    assert row["avg_confirm_delta_on_override"] == (3.0 + 3.0 + 0.5) / 3
    assert row["performance_metric_source"] == "realized_fired_whole_game_delta"
    assert row["per_fire_performance_column"] == "avg_gain_on_override"
    assert row["hand_ev_performance_column"] == "aggregate_ev_per_hand"
    assert row["rerank_delta_metric_role"] == "gate_diagnostic_only"
    assert row["confirm_delta_metric_role"] == "gate_diagnostic_only"
    assert row["confirm_delta_performance_claim_allowed"] is False


def test_stage_b_delta_prefers_confirm_delta_with_rerank_fallback():
    assert stage_b_delta({"confirm_delta": 2.5, "rerank_delta": 99.0}) == 2.5
    assert stage_b_delta({"confirm_delta": "", "rerank_delta": 1.25}) == 1.25


def test_realized_override_count_only_counts_fired_rows_with_unbiased_delta():
    assert (
        realized_override_count(
            [
                {
                    "override_fired": True,
                    "realized_delta_valid": True,
                    "realized_candidate_seat_delta": 0.0,
                },
                {
                    "override_fired": True,
                    "realized_delta_valid": True,
                    "realized_candidate_seat_delta": "",
                },
                {
                    "override_fired": True,
                    "realized_delta_valid": False,
                    "realized_candidate_seat_delta": 4.0,
                },
                {
                    "override_fired": False,
                    "realized_delta_valid": True,
                    "realized_candidate_seat_delta": 7.0,
                },
            ]
        )
        == 1
    )


def test_conditional_override_metrics_use_realized_fired_delta_distribution():
    rows = conditional_override_metrics(
        [
            {
                "config_id": "a",
                "override_fired": True,
                "rerank_delta": 10.0,
                "confirm_delta": 1.0,
                "realized_delta_valid": True,
                "realized_candidate_seat_delta": -2.0,
            },
            {
                "config_id": "a",
                "override_fired": True,
                "rerank_delta": 12.0,
                "confirm_delta": 3.0,
                "realized_delta_valid": True,
                "realized_candidate_seat_delta": 4.0,
            },
            {"config_id": "a", "override_fired": False, "rerank_delta": ""},
            {"config_id": "a", "override_fired": False, "rerank_delta": ""},
        ]
    )

    assert rows[0]["config_id"] == "a"
    assert rows[0]["override_count"] == 2
    assert rows[0]["realized_override_count"] == 2
    assert rows[0]["override_rate"] == 0.5
    assert rows[0]["per_override_delta_mean"] == 1.0
    assert rows[0]["confirm_delta_mean_on_fired"] == 2.0
    assert rows[0]["estimated_ev_per_hand"] == 0.5
    assert rows[0]["performance_metric_source"] == "realized_fired_whole_game_delta"
    assert rows[0]["per_fire_performance_column"] == "per_override_delta_mean"
    assert rows[0]["hand_ev_performance_column"] == "estimated_ev_per_hand"
    assert rows[0]["confirm_delta_metric_role"] == "gate_diagnostic_only"
    assert rows[0]["confirm_delta_performance_claim_allowed"] is False


def test_cancellation_audit_reports_non_fired_realized_delta_leakage():
    rows = cancellation_audit(
        [
            {
                "config_id": "a",
                "override_fired": False,
                "realized_delta_valid": True,
                "realized_candidate_seat_delta": 0.0,
            },
            {
                "config_id": "a",
                "override_fired": False,
                "realized_delta_valid": True,
                "realized_candidate_seat_delta": 3.0,
            },
            {
                "config_id": "a",
                "override_fired": True,
                "realized_delta_valid": True,
                "realized_candidate_seat_delta": -2.0,
            },
        ]
    )

    assert rows[0]["non_fired_count"] == 2
    assert rows[0]["non_fired_nonzero_count"] == 1
    assert rows[0]["non_fired_delta_sum"] == 3.0
    assert rows[0]["fired_delta_mean"] == -2.0
    assert not cancellation_is_clean(rows)


def test_cancellation_is_clean_requires_rows_and_zero_non_fired_delta():
    assert not cancellation_is_clean([])
    assert cancellation_is_clean(
        [
            {
                "config_id": "a",
                "non_fired_nonzero_count": 0,
                "non_fired_delta_sum": 0.0,
                "non_fired_delta_max_abs": 0.0,
            }
        ]
    )
    assert not cancellation_is_clean(
        [
            {
                "config_id": "a",
                "non_fired_nonzero_count": 0,
                "non_fired_delta_sum": 0.0,
                "non_fired_delta_max_abs": 0.1,
            }
        ]
    )


def test_topk_summary_blocks_conditional_go_when_cancellation_dirty(tmp_path):
    path = tmp_path / "summary.md"
    args = SimpleNamespace(
        t3_continuation="stage3_reference_default",
        hu_turn2_stage8b_model="model.pt",
        seeds="1",
        games_per_seed=10,
        target_realized_overrides_per_seed=0,
        target_risk_vetoes_per_seed=0,
        seed_stride=1000,
        stage8c_risk_model_resolved=None,
        stage8c_risk_threshold_resolved=None,
        stage8c_risk_model=None,
        local_ev_risk_model=None,
        stage8c_risk_rank_min=None,
        stage8c_risk_rank_max=None,
        stage8c_risk_audit_only=False,
    )
    rows = [
        {
            "config_id": "cfg",
            "paired_seeds": 10,
            "aggregate_ev_per_hand": 0.1,
            "ci95_low_seed_means": 0.0,
            "ci95_high_seed_means": 0.2,
            "override_count": 5,
            "runtime_override_rate": 0.1,
            "avg_gain_on_override": 1.0,
        }
    ]
    cancellation_rows = [
        {
            "config_id": "cfg",
            "non_fired_nonzero_count": 1,
            "non_fired_delta_sum": 2.0,
            "non_fired_delta_max_abs": 2.0,
        }
    ]

    write_summary(path, rows, [], [], cancellation_rows, [], args)

    text = path.read_text(encoding="utf-8")
    assert "TopK+MC rerank validation: `No-Go`" in text
    assert "realized per-fire required for Conditional-Go: `True`" in text
    assert "confirm/rerank diagnostics used for Conditional-Go: `False`" in text
    assert "non-fired cancellation clean for best config: `no`" in text
    assert "non-fired cancellation clean in all configs: `no`" in text


def test_topk_conditional_go_centralizes_ev_safety_and_cancellation_gates():
    best = {
        "config_id": "a",
        "aggregate_ev_per_hand": 0.1,
        "ci95_low_seed_means": 0.0,
    }
    conditional_good = [
        {
            "config_id": "a",
            "realized_override_count": 3,
            "per_override_delta_mean": 1.0,
            "estimated_ev_per_hand": 0.1,
        }
    ]
    conditional_negative = [
        {
            "config_id": "a",
            "realized_override_count": 3,
            "per_override_delta_mean": -1.0,
            "estimated_ev_per_hand": -0.1,
        }
    ]
    conditional_zero_realized = [
        {
            "config_id": "a",
            "realized_override_count": 0,
            "per_override_delta_mean": 3.0,
            "estimated_ev_per_hand": 0.3,
        }
    ]
    clean_cancel = [
        {
            "config_id": "a",
            "non_fired_nonzero_count": 0,
            "non_fired_delta_sum": 0.0,
            "non_fired_delta_max_abs": 0.0,
        }
    ]
    dirty_cancel = [
        {
            "config_id": "a",
            "non_fired_nonzero_count": 1,
            "non_fired_delta_sum": 1.0,
            "non_fired_delta_max_abs": 1.0,
        }
    ]
    unrelated_dirty_cancel = [
        {
            "config_id": "b",
            "non_fired_nonzero_count": 1,
            "non_fired_delta_sum": 1.0,
            "non_fired_delta_max_abs": 1.0,
        },
        {
            "config_id": "a",
            "non_fired_nonzero_count": 0,
            "non_fired_delta_sum": 0.0,
            "non_fired_delta_max_abs": 0.0,
        },
    ]
    safety = [{"config_id": "a", "safety_blocker": 1}]
    unrelated_safety = [{"config_id": "b", "safety_blocker": 1}]

    assert topk_conditional_go(best, conditional_rows=conditional_good, cancellation_rows=clean_cancel, safety_rows=[])
    assert topk_conditional_go(
        best,
        conditional_rows=conditional_good,
        cancellation_rows=unrelated_dirty_cancel,
        safety_rows=[],
    )
    assert topk_conditional_go(
        best,
        conditional_rows=conditional_good,
        cancellation_rows=clean_cancel,
        safety_rows=unrelated_safety,
    )
    assert not topk_conditional_go(best, conditional_rows=[], cancellation_rows=clean_cancel, safety_rows=[])
    assert not topk_conditional_go(
        best,
        conditional_rows=conditional_negative,
        cancellation_rows=clean_cancel,
        safety_rows=[],
    )
    assert not topk_conditional_go(
        best,
        conditional_rows=conditional_zero_realized,
        cancellation_rows=clean_cancel,
        safety_rows=[],
    )
    assert not topk_conditional_go(best, conditional_rows=conditional_good, cancellation_rows=dirty_cancel, safety_rows=[])
    assert not cancellation_is_clean(unrelated_dirty_cancel)
    assert cancellation_is_clean_for_config(best, unrelated_dirty_cancel)
    assert not cancellation_is_clean_for_config(best, dirty_cancel)
    assert not topk_conditional_go(best, conditional_rows=conditional_good, cancellation_rows=clean_cancel, safety_rows=safety)
    assert has_safety_blocker(unrelated_safety)
    assert not has_safety_blocker_for_config(best, unrelated_safety)
    assert has_safety_blocker_for_config(best, safety)
    assert not topk_conditional_go(
        {"config_id": "a", "aggregate_ev_per_hand": 0.1, "ci95_low_seed_means": -0.10},
        conditional_rows=conditional_good,
        cancellation_rows=clean_cancel,
        safety_rows=[],
    )


def test_topk_summary_blocks_conditional_go_when_realized_per_fire_negative(tmp_path):
    path = tmp_path / "summary.md"
    args = SimpleNamespace(
        t3_continuation="stage3_reference_default",
        hu_turn2_stage8b_model="model.pt",
        seeds="1",
        games_per_seed=10,
        target_realized_overrides_per_seed=0,
        target_risk_vetoes_per_seed=0,
        seed_stride=1000,
        stage8c_risk_model_resolved=None,
        stage8c_risk_threshold_resolved=None,
        stage8c_risk_model=None,
        local_ev_risk_model=None,
        stage8c_risk_rank_min=None,
        stage8c_risk_rank_max=None,
        stage8c_risk_audit_only=False,
    )
    rows = [
        {
            "config_id": "cfg",
            "paired_seeds": 10,
            "aggregate_ev_per_hand": 0.1,
            "ci95_low_seed_means": 0.0,
            "ci95_high_seed_means": 0.2,
            "override_count": 5,
            "runtime_override_rate": 0.1,
            "avg_gain_on_override": 1.0,
        }
    ]
    conditional_rows = [
        {
            "config_id": "cfg",
            "override_count": 5,
            "realized_override_count": 5,
            "override_rate": 0.1,
            "per_override_delta_mean": -1.0,
            "confirm_delta_mean_on_fired": 3.0,
            "estimated_ev_per_hand": -0.1,
        }
    ]
    cancellation_rows = [
        {
            "config_id": "cfg",
            "non_fired_nonzero_count": 0,
            "non_fired_delta_sum": 0.0,
            "non_fired_delta_max_abs": 0.0,
        }
    ]

    write_summary(path, rows, conditional_rows, [], cancellation_rows, [], args)

    text = path.read_text(encoding="utf-8")
    assert "TopK+MC rerank validation: `No-Go`" in text
    assert "conditional realized per-fire positive: `no`" in text
    assert "confirm delta mean" in text


def test_topk_go_nogo_records_realized_per_fire_guardrails(tmp_path):
    negative_path = tmp_path / "go_nogo_negative.md"
    positive_path = tmp_path / "go_nogo_positive.md"
    dirty_cancellation_path = tmp_path / "go_nogo_dirty_cancellation.md"
    unrelated_dirty_cancellation_path = tmp_path / "go_nogo_unrelated_dirty_cancellation.md"
    safety_blocker_path = tmp_path / "go_nogo_safety_blocker.md"
    unrelated_safety_blocker_path = tmp_path / "go_nogo_unrelated_safety_blocker.md"
    args = SimpleNamespace(
        t3_continuation="stage3_reference_default",
        stage8c_risk_rank_min=None,
        stage8c_risk_rank_max=None,
    )
    rows = [
        {
            "config_id": "cfg",
            "aggregate_ev_per_hand": 0.1,
            "ci95_low_seed_means": 0.0,
        }
    ]
    conditional_rows = [
        {
            "config_id": "cfg",
            "realized_override_count": 5,
            "per_override_delta_mean": -1.0,
            "estimated_ev_per_hand": -0.1,
        }
    ]
    cancellation_rows = [
        {
            "config_id": "cfg",
            "non_fired_nonzero_count": 0,
            "non_fired_delta_sum": 0.0,
            "non_fired_delta_max_abs": 0.0,
        }
    ]

    write_go_nogo(
        negative_path,
        rows,
        conditional_rows,
        cancellation_rows,
        [],
        args,
        stage8c_risk_model=None,
        stage8c_risk_threshold=0.0,
        elapsed_seconds=1.23,
    )

    text = negative_path.read_text(encoding="utf-8")
    assert "- decision: `No-Go`" in text
    assert "- performance_metric_source: `realized_fired_whole_game_delta`" in text
    assert "- conditional_realized_per_fire_positive: `no`" in text
    assert "- conditional_realized_overrides_for_best_config: `5`" in text
    assert "- conditional_realized_per_fire_required_for_conditional_go: `True`" in text
    assert "- confirm_rerank_diagnostics_used_for_conditional_go: `False`" in text
    assert "- confirm_delta_performance_claim_allowed: `False`" in text
    assert "- cancellation_clean_for_best_config: `True`" in text
    assert "- cancellation_clean_all_configs: `True`" in text
    assert "- cancellation_clean: `True`" in text

    positive_conditional_rows = [
        {
            "config_id": "cfg",
            "realized_override_count": 5,
            "per_override_delta_mean": 1.0,
            "estimated_ev_per_hand": 0.1,
        }
    ]
    write_go_nogo(
        positive_path,
        rows,
        positive_conditional_rows,
        cancellation_rows,
        [],
        args,
        stage8c_risk_model=None,
        stage8c_risk_threshold=0.0,
        elapsed_seconds=1.23,
    )

    positive_text = positive_path.read_text(encoding="utf-8")
    assert "- decision: `Conditional-Go`" in positive_text
    assert "- conditional_realized_per_fire_positive: `yes`" in positive_text
    assert "- conditional_realized_overrides_for_best_config: `5`" in positive_text

    dirty_cancellation_rows = [
        {
            "config_id": "cfg",
            "non_fired_nonzero_count": 1,
            "non_fired_delta_sum": 2.0,
            "non_fired_delta_max_abs": 2.0,
        }
    ]
    write_go_nogo(
        dirty_cancellation_path,
        rows,
        positive_conditional_rows,
        dirty_cancellation_rows,
        [],
        args,
        stage8c_risk_model=None,
        stage8c_risk_threshold=0.0,
        elapsed_seconds=1.23,
    )
    dirty_cancellation_text = dirty_cancellation_path.read_text(encoding="utf-8")
    assert "- decision: `No-Go`" in dirty_cancellation_text
    assert "- conditional_realized_per_fire_positive: `yes`" in dirty_cancellation_text
    assert "- cancellation_clean_for_best_config: `False`" in dirty_cancellation_text
    assert "- cancellation_clean_all_configs: `False`" in dirty_cancellation_text
    assert "- cancellation_clean: `False`" in dirty_cancellation_text

    unrelated_dirty_cancellation_rows = [
        {
            "config_id": "other_cfg",
            "non_fired_nonzero_count": 1,
            "non_fired_delta_sum": 2.0,
            "non_fired_delta_max_abs": 2.0,
        },
        {
            "config_id": "cfg",
            "non_fired_nonzero_count": 0,
            "non_fired_delta_sum": 0.0,
            "non_fired_delta_max_abs": 0.0,
        },
    ]
    write_go_nogo(
        unrelated_dirty_cancellation_path,
        rows,
        positive_conditional_rows,
        unrelated_dirty_cancellation_rows,
        [],
        args,
        stage8c_risk_model=None,
        stage8c_risk_threshold=0.0,
        elapsed_seconds=1.23,
    )
    unrelated_dirty_cancellation_text = unrelated_dirty_cancellation_path.read_text(encoding="utf-8")
    assert "- decision: `Conditional-Go`" in unrelated_dirty_cancellation_text
    assert "- cancellation_clean_for_best_config: `True`" in unrelated_dirty_cancellation_text
    assert "- cancellation_clean_all_configs: `False`" in unrelated_dirty_cancellation_text
    assert "- cancellation_clean: `True`" in unrelated_dirty_cancellation_text

    write_go_nogo(
        safety_blocker_path,
        rows,
        positive_conditional_rows,
        cancellation_rows,
        [{"config_id": "cfg", "safety_blocker": 1}],
        args,
        stage8c_risk_model=None,
        stage8c_risk_threshold=0.0,
        elapsed_seconds=1.23,
    )
    safety_blocker_text = safety_blocker_path.read_text(encoding="utf-8")
    assert "- decision: `No-Go`" in safety_blocker_text
    assert "- conditional_realized_per_fire_positive: `yes`" in safety_blocker_text
    assert "- known_negative_predicted_delta_safety_blocker_for_best_config: `yes`" in safety_blocker_text
    assert "- known_negative_predicted_delta_safety_blocker_any_config: `yes`" in safety_blocker_text

    write_go_nogo(
        unrelated_safety_blocker_path,
        rows,
        positive_conditional_rows,
        cancellation_rows,
        [{"config_id": "other_cfg", "safety_blocker": 1}],
        args,
        stage8c_risk_model=None,
        stage8c_risk_threshold=0.0,
        elapsed_seconds=1.23,
    )
    unrelated_safety_text = unrelated_safety_blocker_path.read_text(encoding="utf-8")
    assert "- decision: `Conditional-Go`" in unrelated_safety_text
    assert "- known_negative_predicted_delta_safety_blocker_for_best_config: `no`" in unrelated_safety_text
    assert "- known_negative_predicted_delta_safety_blocker_any_config: `yes`" in unrelated_safety_text


def test_predicted_delta_safety_audit_blocks_negative_model_delta_losses():
    rows = predicted_delta_safety_audit(
        [
            {
                "config_id": "a",
                "override_fired": True,
                "predicted_delta": -1.0,
                "realized_delta_valid": True,
                "realized_candidate_seat_delta": -6.0,
            },
            {
                "config_id": "a",
                "override_fired": True,
                "predicted_delta": 2.0,
                "realized_delta_valid": True,
                "realized_candidate_seat_delta": 3.0,
            },
            {
                "config_id": "a",
                "override_fired": False,
                "predicted_delta": -3.0,
                "realized_delta_valid": True,
                "realized_candidate_seat_delta": 0.0,
            },
        ]
    )

    assert rows[0]["override_count"] == 2
    assert rows[0]["negative_predicted_delta_override_count"] == 1
    assert rows[0]["negative_predicted_delta_realized_loss_count"] == 1
    assert rows[0]["negative_predicted_delta_max_loss"] == 6.0
    assert rows[0]["safety_blocker"] == 1
    assert has_safety_blocker(rows)


def test_predicted_delta_safety_audit_allows_negative_model_delta_without_realized_loss():
    rows = predicted_delta_safety_audit(
        [
            {
                "config_id": "a",
                "override_fired": True,
                "predicted_delta": -1.0,
                "realized_delta_valid": True,
                "realized_candidate_seat_delta": 2.0,
            },
        ]
    )

    assert rows[0]["negative_predicted_delta_override_count"] == 1
    assert rows[0]["negative_predicted_delta_realized_loss_count"] == 0
    assert rows[0]["safety_blocker"] == 0
    assert not has_safety_blocker(rows)
