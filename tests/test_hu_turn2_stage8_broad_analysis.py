import json

from ofc_regular.analyze_hu_turn2_stage8_broad_teacher import (
    action_is_legal,
    correctness,
    signal_row,
)
from ofc_regular.hu_turn2_teacher_data import _attach_broad_bucket_metadata


def _record():
    action = {
        "original_index": 0,
        "placements": [["Ah", "top"], ["Kd", "middle"]],
        "discards": ["2c"],
        "score": 1.0,
        "ev_standard_error": 0.1,
        "rollout_count": 512,
        "common_random_future_digest": "digest",
    }
    return {
        "phase": "hu_turn2_7card",
        "source_bucket": "natural",
        "_run_bucket": "natural",
        "seat": "first",
        "actions": [action],
        "legal_actions": [action],
        "best_action": 0,
        "second_best_action": {"original_index": 0, "action": action},
        "baseline_action": {"original_index": 0, "action": action},
        "reference_action": {"original_index": 0, "action": action},
        "fallback_action": {"original_index": 0, "action": action},
        "common_random_future_digest": "digest",
        "rollout_count": 512,
        "delta_best_vs_baseline": 0.4,
        "best_margin": 0.2,
        "SE_delta_best_vs_baseline": 0.1,
        "teacher_label": "positive",
        "teacher_distribution_metrics": {"baseline_disagreement": True},
        "profiling": {
            "stage3_feature_mode": "rust_direct",
            "stage3_fallback_recomputed_count": 0,
            "seconds_total": 1.0,
        },
    }


def test_stage8_broad_correctness_accepts_valid_record():
    result = correctness([_record()], expected_total=1, expected_per_bucket=1, future_samples=512)

    assert result["ok"] is True
    assert result["common_random_future_digest_mismatch"] == 0
    assert result["action_legal_failures"] == 0


def test_action_is_legal_handles_wrapper_original_index_zero():
    record = _record()

    assert action_is_legal(record, record["baseline_action"])


def test_signal_row_counts_labels_and_se_buckets():
    row = signal_row("all", [_record()])

    assert row["positive"] == 1
    assert row["delta_ge_2se_rate"] == 1.0


def test_broad_metadata_copies_pool_prediction_fields():
    sample = _record()
    pool_record = {
        "state_hash": "abc",
        "prefilter_version": "cheap_no_rollout_v1",
        "predicted_bucket": "predicted_high_regret",
        "accept_reason": "reason",
        "source_bucket_requested": "high_regret",
        "source_bucket_actual": "high_regret",
        "predicted_delta_vs_baseline": 1.25,
        "predicted_delta_vs_reference": 0.75,
        "predicted_margin": 0.5,
        "reference_margin": 0.07,
        "baseline_margin": 0.12,
    }

    _attach_broad_bucket_metadata(
        sample,
        source_bucket_requested="high_regret",
        source_bucket_actual="high_regret",
        pool_record=pool_record,
    )

    assert sample["actual_low_margin"] is True
    assert sample["predicted_bucket"] == "predicted_high_regret"
    assert sample["predicted_delta"] == 1.25
    assert sample["predicted_reference_margin"] == 0.07
    json.dumps(sample)
