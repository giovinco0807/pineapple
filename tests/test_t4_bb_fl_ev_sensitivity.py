import pytest

import ai.tutor.exact_late as exact_late_module
import ai.tutor.t4_bb_fl_ev_sensitivity as sensitivity
from ai.mcts.rollout_evaluator import RolloutEvaluator


def test_fl_live_fixture_builds_valid_infosets_with_physical_pool_26():
    pool = sensitivity.physically_live_pool()
    assert len(pool) == 26
    assert "X1" in pool and "X2" in pool
    for draw in sensitivity.HAND_PICKED_DRAWS:
        information = sensitivity.build_infoset(draw)
        information.canonical_json()
        assert information.phase == "t4_first"
        assert information.actor == "bb"
        assert all(card in pool for card in draw)


def test_sample_draws_is_deterministic_and_unique():
    first = sensitivity.sample_draws(seed=123, random_count=10)
    second = sensitivity.sample_draws(seed=123, random_count=10)
    assert first == second
    assert len(first) == len(sensitivity.HAND_PICKED_DRAWS) + 10
    keys = {tuple(sorted(draw)) for draw in first}
    assert len(keys) == len(first)
    other_seed = sensitivity.sample_draws(seed=124, random_count=10)
    assert other_seed[: len(sensitivity.HAND_PICKED_DRAWS)] == list(
        sensitivity.HAND_PICKED_DRAWS
    )


def test_compute_action_evs_scales_fl_ev_and_restores_it(monkeypatch):
    observed_fl_ev_sums = []

    def stub_distribution(final_board, opponent_board, exclude=None):
        observed_fl_ev_sums.append(sum(RolloutEvaluator.FL_EV.values()))
        return {"score": float(sum(RolloutEvaluator.FL_EV.values()))}

    original_fl_ev = RolloutEvaluator.FL_EV
    base_sum = sum(original_fl_ev.values())
    with monkeypatch.context() as patch:
        patch.setattr(
            exact_late_module,
            "exact_t4_opponent_response_distribution",
            stub_distribution,
        )
        evs = sensitivity.compute_action_evs(("Ts", "9d", "5s"), 1.2)

    assert RolloutEvaluator.FL_EV is original_fl_ev
    assert observed_fl_ev_sums
    for value in observed_fl_ev_sums:
        assert value == pytest.approx(base_sum * 1.2)
    for value in evs.values():
        assert value == pytest.approx(base_sum * 1.2)
    assert len(evs) == 6
