from types import SimpleNamespace

from ofc_regular.evaluate_hu_turn0_counterfactual import (
    _hero_score,
    evaluate_counterfactual,
    summarize_events,
)


def test_hero_score_uses_trace_hand_zero_sum_contract():
    trace = {"score_p0": 3.5}

    assert _hero_score(trace, 0) == 3.5
    assert _hero_score(trace, 1) == -3.5


def test_summarize_events_uses_realized_fired_delta_and_checks_nonfires():
    events = [
        {"seed": 1, "seat": "first", "override_fired": True, "realized_delta": 2.0},
        {"seed": 1, "seat": "second", "override_fired": True, "realized_delta": -1.0},
        {
            "seed": 2,
            "seat": "first",
            "override_fired": False,
            "realized_delta": 0.0,
            "no_override_reason": "below_hu_turn0_margin",
        },
        {
            "seed": 2,
            "seat": "second",
            "override_fired": False,
            "realized_delta": 0.5,
            "no_override_reason": "same_as_baseline",
        },
    ]

    summary = summarize_events(events)

    assert summary["events"] == 4
    assert summary["paired_seeds"] == 2
    assert summary["fires"] == 2
    assert summary["avg_delta_per_fire"] == 0.5
    assert summary["realized_fired_contribution_per_hand"] == 0.25
    assert summary["false_positive_rate"] == 0.5
    assert summary["max_fire_loss"] == 1.0
    assert summary["non_fired_nonzero_count"] == 1
    assert summary["by_seat"]["first"]["fires"] == 1
    assert summary["by_seat"]["second"]["fires"] == 1


def test_counterfactual_harness_reuses_world_seeds_and_cancels_nonfire(monkeypatch):
    class FakePolicy(SimpleNamespace):
        pass

    def factory(seed, seat, candidate):
        fired = bool(candidate and seat == "first")
        decision = {
            "override_fired": fired,
            "no_override_reason": "" if fired else "same_as_baseline",
            "hu_turn0_predicted_margin": 2.0 if fired else 0.0,
            "fallback_action_index": 0,
            "candidate_action_index": 1 if fired else 0,
            "final_action_index": 1 if fired else 0,
        }
        return FakePolicy(
            seed=seed,
            seat=seat,
            candidate=candidate,
            fired=fired,
            hu_turn0_decision_log=[decision] if candidate else None,
        )

    seen = []

    def fake_trace_hand(*, seed, profile_p0, profile_p1, policy_p0, policy_p1):
        seen.append((seed, policy_p0.seed, policy_p1.seed, policy_p0.candidate, policy_p1.candidate))
        score_p0 = 2.0 if policy_p0.candidate and policy_p0.fired else 0.0
        return {"score_p0": score_p0}

    monkeypatch.setattr(
        "ofc_regular.evaluate_hu_turn0_counterfactual.trace_hand",
        fake_trace_hand,
    )

    summary, events = evaluate_counterfactual(
        games=3,
        seed=100,
        seed_stride=7,
        policy_factory=factory,
        profile="fixed",
    )

    assert summary["paired_seeds"] == 3
    assert summary["events"] == 6
    assert summary["fires"] == 3
    assert summary["avg_delta_per_hand"] == 1.0
    assert summary["avg_delta_per_fire"] == 2.0
    assert summary["non_fired_nonzero_count"] == 0
    assert [event["seed"] for event in events[::2]] == [100, 107, 114]
    # Each candidate world is followed by a baseline world with identical policy seeds.
    for offset in range(0, len(seen), 2):
        assert seen[offset][:3] == seen[offset + 1][:3]
