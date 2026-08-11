from __future__ import annotations

from copy import deepcopy

import pytest

import ofc_regular.hu_m31_t3_step6d_fresh_quality_gate_v1 as subject


def _merge(regrets: list[float]) -> dict:
    return {
        "confirmation_quality": {
            "count": 10,
            "mean": sum(regrets) / len(regrets),
            "p95": subject._nearest_rank(regrets, 0.95),
            "p99": subject._nearest_rank(regrets, 0.99),
            "max": max(regrets),
            "selected_regrets": regrets,
        },
        "integrity": {
            "primary_paired_hands": 50,
            "primary_roots": 100,
            "confirmation_paired_hands": 5,
            "confirmation_roots": 10,
            "candidate_evaluation_overlap": 0,
            "candidate_confirmation_overlap": 0,
            "evaluation_confirmation_overlap": 0,
            "hidden_information_field_count": 0,
            "unknown_field_count": 0,
            "action_key_drift_count": 0,
            "missing_or_extra_result_count": 0,
        },
        "teacher_values_are_realized_match_ev": False,
        "training_eligible": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
    }


def _patch_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subject,
        "validate_fresh_quality_merge_value",
        lambda value, *, replay_sources: (
            deepcopy(dict(value))
            if replay_sources is True
            else (_ for _ in ()).throw(PermissionError())
        ),
    )


def test_frozen_regret_thresholds_and_nearest_rank():
    assert subject.REGRET_THRESHOLDS == {
        "mean_max": 0.75,
        "p95_max": 3.0,
        "p99_max": 6.0,
        "max_max": 15.0,
    }
    assert subject._regret_metrics([0.0] * 9 + [3.0]) == {
        "count": 10,
        "mean": 0.3,
        "p95": 3.0,
        "p99": 3.0,
        "max": 3.0,
        "percentile_method": subject.PERCENTILE_METHOD,
        "selected_regrets": [0.0] * 9 + [3.0],
    }


def test_quality_gate_pass_opens_only_25_paired_data_pilot(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_replay(monkeypatch)
    gate = subject.build_fresh_quality_gate(
        merge=_merge([0.5] * 10), replay_sources=True
    )
    assert gate["status"] == "pass"
    assert gate["all_gates_passed"] is True
    assert gate["quality_pilot_passed"] is True
    assert gate["data_pilot_25_paired_authorized"] is True
    assert gate["full_9000_paired_fanout_authorized"] is False
    assert gate["training_eligible"] is False
    assert gate["promotion_evidence"] is False
    assert gate["current_profile_changed"] is False
    assert gate["teacher_values_are_realized_match_ev"] is False


def test_quality_gate_regret_failure_is_no_go_without_same_seed_reselection(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_replay(monkeypatch)
    merge = _merge([0.0] * 9 + [3.0001])
    gate = subject.build_fresh_quality_gate(merge=merge, replay_sources=True)
    assert gate["status"] == "no_go"
    assert gate["gates"]["confirmation_regret_p95_at_most_3"] is False
    assert gate["data_pilot_25_paired_authorized"] is False
    assert gate["decision"] == (
        "fresh_quality_no_go_no_same_seed_threshold_reselection"
    )


def test_integrity_nonzero_fails_gate_even_with_zero_regret(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_replay(monkeypatch)
    merge = _merge([0.0] * 10)
    merge["integrity"]["action_key_drift_count"] = 1
    gate = subject.build_fresh_quality_gate(merge=merge, replay_sources=True)
    assert gate["status"] == "no_go"
    assert gate["gates"]["hidden_unknown_actionkey_rng_missing_zero"] is False


def test_gate_rejects_unknown_field_and_result_contract_fails_before_trust(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_replay(monkeypatch)
    gate = subject.build_fresh_quality_gate(
        merge=_merge([0.25] * 10), replay_sources=True
    )
    gate["unknown"] = True
    with pytest.raises(ValueError, match="fields changed"):
        subject.validate_fresh_quality_gate_value(gate, replay_sources=True)

    with pytest.raises(ValueError, match="fields changed"):
        subject.validate_job_result(
            {"unknown": True},
            job={},
            plan={},
            seal={},
            root_lookup={},
        )
