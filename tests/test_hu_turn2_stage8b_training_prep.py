import csv

import numpy as np

from ofc_regular.prepare_hu_turn2_stage8b_training import (
    LABEL_TO_ID,
    attach_stage8b_labels,
    selected_high_mc_states,
)
from ofc_regular.analyze_hu_turn2_stage8b_training import (
    binary_counts,
    metric_from_counts,
    runtime_fires,
)
from ofc_regular.evaluate_hu_turn2_stage8b_topk_mc_rerank import (
    aggregate_topk_seed_rows,
    parse_topk_configs,
)
from ofc_regular.train_hu_turn2_pilot_model import apply_stage8b_gate_labels, gate_loss


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
    config = parse_topk_configs("k5/mc128/d0.5/se1.5/seat=first/rank3/g0.9/bygate_delta")[0]

    assert config.top_k == 5
    assert config.mc_samples == 128
    assert config.min_delta == 0.5
    assert config.se_multiplier == 1.5
    assert config.allowed_seats == ("first",)
    assert config.candidate_ev_rank_max == 3
    assert config.min_gate_probability == 0.9
    assert config.topk_score == "gate_delta"


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
                "topk_score": "delta",
                "paired_seeds": 10,
                "hands": 20,
                "ev_per_hand": 0.2,
                "paired_seed_wins": 6,
                "paired_seed_losses": 3,
                "paired_seed_ties": 1,
                "decision_count": 20,
                "override_count": 2,
                "avg_rerank_delta_on_override": 1.0,
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
                "topk_score": "delta",
                "paired_seeds": 10,
                "hands": 20,
                "ev_per_hand": -0.1,
                "paired_seed_wins": 3,
                "paired_seed_losses": 6,
                "paired_seed_ties": 1,
                "decision_count": 20,
                "override_count": 1,
                "avg_rerank_delta_on_override": 0.5,
            },
        ]
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["paired_seeds"] == 20
    assert row["hands"] == 40
    assert row["aggregate_ev_per_hand"] == 0.05
    assert row["override_count"] == 3
    assert row["runtime_override_rate"] == 3 / 40
    assert row["avg_gain_on_override"] == (1.0 + 1.0 + 0.5) / 3
