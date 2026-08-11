from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from ofc_regular.evaluate_hu_m4_population import (
    M4_OPPONENT_PROFILES,
    evaluate_hu_m4_population,
    gameplay_digest,
    parse_args,
    requires_bound_population_runtime,
)
from ofc_regular.hu_m43_joint_model_v5 import HU_M43_V5_MODEL_SCHEMA
from ofc_regular.hu_m43_joint_model_v6 import HU_M43_V6_MODEL_SCHEMA
from ofc_regular.hu_m4_t1_policy import HU_M4_T1_DECISION_SCHEMA


@dataclass
class _FakePolicy:
    kind: str
    seat: str
    policy_seed: int
    decision_log: list[dict[str, Any]] | None
    opponent: str = ""
    omit_log: bool = False
    corrupt_nonfire: bool = False


def _factory(
    kind: str,
    *,
    opponent: str = "",
    omit_log: bool = False,
    corrupt_nonfire: bool = False,
):
    def make(*, policy_seed: int, seat: str, decision_log):
        return _FakePolicy(
            kind=kind,
            seat=seat,
            policy_seed=policy_seed,
            decision_log=decision_log,
            opponent=opponent,
            omit_log=omit_log,
            corrupt_nonfire=corrupt_nonfire,
        )

    return make


def _hero_policy(policy_p0: _FakePolicy, policy_p1: _FakePolicy) -> _FakePolicy:
    return policy_p0 if policy_p0.kind in {"candidate", "baseline"} else policy_p1


def _fake_trace_hand(*, seed, profile_p0, profile_p1, policy_p0, policy_p1):
    hero = _hero_policy(policy_p0, policy_p1)
    # M4 may fire only in the second seat.  The two seeds deliberately contain
    # one positive and one negative realized override.
    fired = hero.kind == "candidate" and hero.seat == "second"
    delta = 2.0 if seed == 10 else -1.0
    hero_score = 3.0 + (delta if fired else 0.0)
    action = "override" if fired else "baseline"
    if hero.kind == "candidate" and hero.corrupt_nonfire and not fired:
        action = "corrupt"
    if hero.kind == "candidate" and not hero.omit_log:
        assert hero.decision_log is not None
        hero.decision_log.append(
            {
                "schema": HU_M4_T1_DECISION_SCHEMA,
                "street": "T1",
                "seat": hero.seat,
                "override_fired": fired,
                "nonfire_reason": None if fired else "first_seat_delegated",
                "predicted_delta": 0.75,
                "safety_probability": 0.9,
            }
        )
    score_p0 = hero_score if hero.seat == "first" else -hero_score
    return {
        "seed": seed,
        "profiles": {"p0": profile_p0, "p1": profile_p1},
        "score_p0": score_p0,
        "turns": [
            {
                "turn": "T1",
                "player": 0 if hero.seat == "first" else 1,
                "profile": profile_p0 if hero.seat == "first" else profile_p1,
                "dealt": ["Ah", "Ks", "2c"],
                "action_key": action,
                "placements": [["Ah", "top"], ["Ks", "middle"]],
                "discards": ["2c"],
                "board": {"top": [action], "middle": [], "bottom": []},
            }
        ],
        "final": {"p0": {"action": action}, "p1": {"action": action}},
        "board_scores": {"p0": {"busted": False}, "p1": {"busted": False}},
    }


def test_paired_population_measures_seats_fires_false_positives_and_tails():
    result = evaluate_hu_m4_population(
        candidate_policy_factory=_factory("candidate"),
        baseline_policy_factory=_factory("baseline"),
        opponent_policy_factories={"opponent_a": _factory("opponent", opponent="a")},
        paired_seeds=2,
        seed=10,
        seed_stride=1,
        trace_fn=_fake_trace_hand,
        include_records=True,
    )

    first = result["population"]["by_seat"]["first"]
    second = result["population"]["by_seat"]["second"]
    paired = result["by_opponent"]["opponent_a"]["seat_swap"]
    assert first["delta_ev_per_hand"]["mean"] == 0.0
    assert first["overrides"] == 0
    assert first["nonfire_cancellation_valid"] == 2
    assert second["delta_ev_per_hand"]["mean"] == pytest.approx(0.5)
    assert second["overrides"] == 2
    assert second["realized_gain_per_override"]["mean"] == pytest.approx(0.5)
    assert second["false_positive_overrides"] == 1
    assert second["false_positive_override_rate"] == 0.5
    assert second["override_loss_tail"]["max"] == 1.0
    assert paired["delta_ev_per_hand"]["mean"] == pytest.approx(0.25)
    assert result["population"]["paired_seat_swap"]["delta_ev_per_hand"][
        "n"
    ] == 2
    assert result["invalid_counterfactuals"] == 0
    assert result["nonfire_cancellation_mismatches"] == 0
    assert result["trace_hands"] == 8


def test_same_seed_and_policy_seeds_are_used_for_candidate_and_baseline():
    calls: list[tuple[str, int, str]] = []

    def recording_factory(kind):
        def make(*, policy_seed, seat, decision_log):
            calls.append((kind, policy_seed, seat))
            return _FakePolicy(kind, seat, policy_seed, decision_log)

        return make

    result = evaluate_hu_m4_population(
        candidate_policy_factory=recording_factory("candidate"),
        baseline_policy_factory=recording_factory("baseline"),
        opponent_policy_factories={"op": recording_factory("opponent")},
        paired_seeds=1,
        seed=17,
        seed_stride=13,
        trace_fn=_fake_trace_hand,
        include_records=True,
    )

    candidate_calls = [(seed, seat) for kind, seed, seat in calls if kind == "candidate"]
    baseline_calls = [(seed, seat) for kind, seed, seat in calls if kind == "baseline"]
    assert candidate_calls == baseline_calls == [(68, "first"), (69, "second")]
    for row in result["records"]:
        assert row["seed"] == 17
        assert row["counterfactual_basis"].startswith("same_seed_physical_seat")


def test_nonfire_digest_ignores_profile_metadata_but_detects_gameplay_change():
    common = {
        "seed": 1,
        "score_p0": 0.0,
        "final": {"p0": {}, "p1": {}},
        "board_scores": {},
        "turns": [{"turn": "T1", "profile": "candidate", "action_key": "x"}],
    }
    renamed = {
        **common,
        "profiles": {"p0": "baseline", "p1": "op"},
        "turns": [{"turn": "T1", "profile": "baseline", "action_key": "x"}],
    }
    changed = {
        **renamed,
        "turns": [{"turn": "T1", "profile": "baseline", "action_key": "y"}],
    }
    assert gameplay_digest(common) == gameplay_digest(renamed)
    assert gameplay_digest(common) != gameplay_digest(changed)

    result = evaluate_hu_m4_population(
        candidate_policy_factory=_factory("candidate", corrupt_nonfire=True),
        baseline_policy_factory=_factory("baseline"),
        opponent_policy_factories={"op": _factory("opponent")},
        paired_seeds=1,
        seed=10,
        trace_fn=_fake_trace_hand,
    )
    assert result["nonfire_cancellation_mismatches"] == 1
    assert result["population"]["by_seat"]["first"][
        "nonfire_cancellation_mismatches"
    ] == 1


def test_missing_m4_decision_log_is_an_invalid_counterfactual():
    result = evaluate_hu_m4_population(
        candidate_policy_factory=_factory("candidate", omit_log=True),
        baseline_policy_factory=_factory("baseline"),
        opponent_policy_factories={"op": _factory("opponent")},
        paired_seeds=1,
        seed=10,
        trace_fn=_fake_trace_hand,
    )
    assert result["invalid_counterfactuals"] == 2
    assert result["population"]["all_seats"]["nonfire_cancellation_unknown"] == 2
    assert result["promotion_eligible_counterfactual_contract"] is False
    assert result["population"]["paired_seat_swap"]["delta_ev_per_hand"]["n"] == 0


def test_population_reports_worst_opponent_and_rejects_current():
    result = evaluate_hu_m4_population(
        candidate_policy_factory=_factory("candidate"),
        baseline_policy_factory=_factory("baseline"),
        opponent_policy_factories={
            "aggressive": _factory("opponent", opponent="aggressive"),
            "conservative": _factory("opponent", opponent="conservative"),
        },
        paired_seeds=1,
        seed=11,
        trace_fn=_fake_trace_hand,
    )
    assert result["worst_case_opponent"]["opponent"] in {
        "aggressive",
        "conservative",
    }
    with pytest.raises(ValueError, match="current"):
        evaluate_hu_m4_population(
            candidate_policy_factory=_factory("candidate"),
            baseline_policy_factory=_factory("baseline"),
            opponent_policy_factories={"current": _factory("opponent")},
            paired_seeds=1,
            seed=1,
            trace_fn=_fake_trace_hand,
        )


def test_cli_requires_explicit_m4_artifacts_and_has_only_fixed_population():
    args = parse_args(
        [
            "--model",
            "joint.pkl",
        ]
    )
    assert args.opponents == list(M4_OPPONENT_PROFILES)
    assert "current" not in args.opponents
    assert args.seed_stride == 1009

    with pytest.raises(SystemExit):
        parse_args(["--candidate-model", "candidate.pkl"])

    legacy = parse_args(
        [
            "--diagnostic-legacy",
            "--candidate-model",
            "candidate.pkl",
            "--safety-model",
            "safety.pkl",
            "--safety-threshold",
            "0.8",
        ]
    )
    assert legacy.diagnostic_legacy is True

    with pytest.raises(SystemExit):
        parse_args(["--model", "joint.pkl", "--threshold-lock", "threshold.json"])


def test_population_promotion_requires_binding_for_v5_and_v6():
    class _Model:
        def __init__(self, schema):
            self.schema = schema

    assert requires_bound_population_runtime(_Model(HU_M43_V5_MODEL_SCHEMA))
    assert requires_bound_population_runtime(_Model(HU_M43_V6_MODEL_SCHEMA))
    assert not requires_bound_population_runtime(_Model("legacy_v4"))


def test_population_ci_clusters_same_seed_across_opponents():
    result = evaluate_hu_m4_population(
        candidate_policy_factory=_factory("candidate"),
        baseline_policy_factory=_factory("baseline"),
        opponent_policy_factories={
            f"opponent_{index}": _factory("opponent") for index in range(4)
        },
        paired_seeds=2,
        seed=10,
        trace_fn=_fake_trace_hand,
    )
    population = result["population"]["paired_seat_swap"]["delta_ev_per_hand"]
    assert population["n"] == 2
    second = result["population"]["by_seat"]["second"]
    assert second["delta_ev_per_hand"]["n"] == 2
    assert second["delta_ev_per_hand"]["ci_independence_unit"] == (
        "hand_seed_cluster"
    )
    assert second["realized_gain_per_override"]["n"] == 8
    assert second["realized_gain_per_override"]["clusters"] == 2
    assert second["realized_gain_per_override"]["ci_independence_unit"] == (
        "hand_seed_cluster_ratio_influence"
    )
    assert all(
        summary["seat_swap"]["delta_ev_per_hand"]["n"] == 2
        for summary in result["by_opponent"].values()
    )


def test_zero_gain_fired_override_is_a_false_positive():
    def zero_gain_trace(**kwargs):
        hand = _fake_trace_hand(**kwargs)
        hero = _hero_policy(kwargs["policy_p0"], kwargs["policy_p1"])
        if hero.seat == "second":
            hand["score_p0"] = -3.0
        return hand

    result = evaluate_hu_m4_population(
        candidate_policy_factory=_factory("candidate"),
        baseline_policy_factory=_factory("baseline"),
        opponent_policy_factories={"op": _factory("opponent")},
        paired_seeds=1,
        seed=10,
        trace_fn=zero_gain_trace,
    )
    second = result["population"]["by_seat"]["second"]
    assert second["overrides"] == 1
    assert second["false_positive_overrides"] == 1
    assert second["false_positive_override_rate"] == 1.0
    assert second["false_positive_definition"] == (
        "realized_override_delta_le_zero"
    )
