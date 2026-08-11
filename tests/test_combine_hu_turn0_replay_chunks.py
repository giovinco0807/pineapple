import math

import pytest

from ofc_regular.combine_hu_turn0_replay_chunks import combine_chunks


def _row(seed, mean):
    action = {"placements": [["As", "top"]], "discards": []}
    baseline = {"placements": [["As", "bottom"]], "discards": []}
    return {
        "target_id": "target",
        "future_samples": 2,
        "replay_seed": seed,
        "future_seed": seed + 1,
        "common_random_future_digest": f"digest-{seed}",
        "common_random_futures_verified": True,
        "action_mapping_verified": True,
        "baseline_action_index": 1,
        "candidate_action_index": 2,
        "baseline_action": baseline,
        "candidate_action": action,
        "baseline_ev": 0.0,
        "candidate_ev": mean,
        "candidate_delta_vs_baseline": mean,
        "candidate_delta_se_vs_baseline": 1.0,
        "replay_seconds": 5.0,
    }


def test_combine_chunks_pools_within_and_between_chunk_variance():
    combined = combine_chunks([_row(10, 1.0), _row(20, 3.0)])

    assert combined["future_samples"] == 4
    assert combined["candidate_delta_vs_baseline"] == 2.0
    assert combined["candidate_delta_se_vs_baseline"] == pytest.approx(
        math.sqrt((8.0 / 3.0) / 4.0)
    )
    assert combined["candidate_ev"] == 2.0
    assert combined["chunk_count"] == 2
    assert combined["pooled_from_independent_chunks"] is True
    assert combined["safe_override_label"] == "positive"


def test_combine_chunks_rejects_action_mapping_drift():
    first = _row(10, 1.0)
    second = _row(20, 1.0)
    second["candidate_action_index"] = 3

    with pytest.raises(ValueError, match="candidate action index changed"):
        combine_chunks([first, second])
