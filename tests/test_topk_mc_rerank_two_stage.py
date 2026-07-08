"""Two-stage (selection + independent confirmation) TopK MC rerank tests."""

from __future__ import annotations

import numpy as np
import pytest

import ofc_regular.evaluate_hu_turn2_stage8b_topk_mc_rerank as rerank_module
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.evaluate_hu_turn2_stage8b_topk_mc_rerank import (
    HuTurn2Stage8bTopKMcRerankPolicy,
    TopKMcRerankConfig,
    parse_topk_configs,
)
from ofc_regular.state import Board


def test_parse_confirm_mc_samples_tokens():
    default, explicit, disabled = parse_topk_configs(
        "k3/mc64/d0.5/se0,k3/mc64/d0.5/se0/cmc128,k3/mc64/d0.5/se0/cmc0"
    )
    assert default.confirm_mc_samples == -1
    assert default.resolved_confirm_mc_samples == 64
    assert "cmc" not in default.config_id
    assert explicit.resolved_confirm_mc_samples == 128
    assert "cmc128" in explicit.config_id
    assert disabled.resolved_confirm_mc_samples == 0
    assert "cmc0" in disabled.config_id


def _hero_board() -> Board:
    return Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c"],
        bottom=["9c", "9d", "9s"],
    )


def _opponent_board() -> Board:
    return Board.from_rows(
        top=["2h"],
        middle=["3h", "4h", "5h"],
        bottom=["8c", "8d", "8s"],
    )


class _FakeStage8bModel:
    pass


class _StubbedPolicy(HuTurn2Stage8bTopKMcRerankPolicy):
    """Overrides the MC stage with scripted results.

    Selection MC ranks a non-baseline candidate far above the baseline.
    Confirmation MC (identified by evaluating exactly two actions)
    returns the scripted ``confirm_delta``.
    """

    def __init__(self, *, confirm_delta: float, **kwargs):
        super().__init__(**kwargs)
        self._scripted_confirm_delta = confirm_delta
        self.rerank_calls: list[dict] = []

    def _rerank_sample(self, *, board, dealt, opponent_board, dead_cards, action_indices, seed, mc_samples=None):
        self.rerank_calls.append(
            {"action_indices": list(action_indices), "seed": seed, "mc_samples": mc_samples}
        )
        baseline_index = 0
        candidate_index = max(action_indices)
        is_confirm = len(action_indices) == 2 and len(self.rerank_calls) > 1
        delta = self._scripted_confirm_delta if is_confirm else 10.0
        actions = [
            {
                "original_index": candidate_index,
                "score": delta,
                "ev_standard_error": 0.1,
            },
            {
                "original_index": baseline_index,
                "score": 0.0,
                "ev_standard_error": 0.1,
            },
        ]
        return {"actions": actions, "common_random_future_digest": f"digest-{len(self.rerank_calls)}"}


def _make_policy(config: TopKMcRerankConfig, *, confirm_delta: float, monkeypatch) -> _StubbedPolicy:
    policy = _StubbedPolicy(
        confirm_delta=confirm_delta,
        hu_turn2_stage8b_model=_FakeStage8bModel(),
        topk_rerank_config=config,
        topk_decision_log=[],
        turn2_model=object(),
        seat="first",
        seed=7,
    )

    def fake_safe_predictions(model, sample, action_count):
        if isinstance(model, _FakeStage8bModel):
            predictions = np.zeros((action_count, 5), dtype=np.float64)
            predictions[:, 0] = np.arange(action_count, dtype=np.float64)
            predictions[:, 1] = np.arange(action_count, dtype=np.float64)
            predictions[:, 4] = 10.0
            return predictions, None
        return np.zeros(action_count, dtype=np.float64), None

    monkeypatch.setattr(rerank_module, "_safe_predictions", fake_safe_predictions)
    return policy


@pytest.mark.parametrize(
    ("confirm_delta", "expect_override", "expect_reason"),
    [
        (5.0, True, ""),
        (0.1, False, "below_confirm_delta"),
    ],
)
def test_two_stage_confirmation_gates_override(monkeypatch, confirm_delta, expect_override, expect_reason):
    config = parse_topk_configs("k3/mc8/d0.5/se0/seat=first")[0]
    policy = _make_policy(config, confirm_delta=confirm_delta, monkeypatch=monkeypatch)
    board = _hero_board()
    dealt = ("As", "Kc", "2d")
    actions = generate_turn_actions(board, dealt)
    assert len(actions) > 3

    chosen = policy._choose_hu_turn2_topk_mc_action(
        board,
        dealt,
        dead_cards=(),
        opponent_board=_opponent_board(),
        hand_id=1,
        game_id=1,
        decision_seed=99,
        street="T2",
    )
    record = policy.topk_decision_log[-1]
    assert record["override_fired"] is expect_override
    assert record["no_override_reason"] == expect_reason
    assert record["confirm_delta"] == pytest.approx(confirm_delta)
    # Two MC stages ran: selection then confirmation.
    assert len(policy.rerank_calls) == 2
    select_call, confirm_call = policy.rerank_calls
    # Confirmation evaluates exactly candidate + baseline on an
    # independent random stream.
    assert len(confirm_call["action_indices"]) == 2
    assert confirm_call["seed"] != select_call["seed"]
    assert confirm_call["mc_samples"] == config.resolved_confirm_mc_samples
    if expect_override:
        assert chosen is not None
    else:
        assert chosen == actions[record["final_action_index"]]
        assert record["final_action_index"] == record["baseline_action_index"]


def test_single_stage_legacy_mode_skips_confirmation(monkeypatch):
    config = parse_topk_configs("k3/mc8/d0.5/se0/cmc0/seat=first")[0]
    policy = _make_policy(config, confirm_delta=0.0, monkeypatch=monkeypatch)
    board = _hero_board()
    dealt = ("As", "Kc", "2d")

    policy._choose_hu_turn2_topk_mc_action(
        board,
        dealt,
        dead_cards=(),
        opponent_board=_opponent_board(),
        hand_id=1,
        game_id=1,
        decision_seed=99,
        street="T2",
    )
    record = policy.topk_decision_log[-1]
    assert record["override_fired"] is True
    assert record["confirm_delta"] is None
    assert len(policy.rerank_calls) == 1
