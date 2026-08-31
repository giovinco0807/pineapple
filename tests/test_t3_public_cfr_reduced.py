from fractions import Fraction

import pytest

from ai.tutor.t3_hu_public_cfr import (
    grouped_information_set_backup,
    reduced_bluff_signaling_game,
    solve_public_signaling_game_cfr,
)


def test_reduced_public_signaling_cfr_learns_mixed_strategy_and_reduces_exploitability():
    game = reduced_bluff_signaling_game()
    result = solve_public_signaling_game_cfr(
        game,
        iterations=20_000,
        checkpoints=(10, 100, 1_000, 5_000, 20_000),
    )

    strong_bet = result.sender_average_strategy["strong"]["bet"]
    weak_bet = result.sender_average_strategy["weak"]["bet"]
    call_bet = result.receiver_average_strategy["bet"]["call"]

    assert strong_bet == pytest.approx(1.0, abs=0.01)
    assert weak_bet == pytest.approx(1 / 3, abs=0.03)
    assert call_bet == pytest.approx(2 / 3, abs=0.03)
    assert 0.05 < weak_bet < 0.95
    assert 0.05 < call_bet < 0.95
    assert result.metrics.value == pytest.approx(1 / 3, abs=0.02)
    assert result.metrics.exploitability < 0.02
    assert result.exploitability_trace[-1][1] < result.exploitability_trace[0][1]

    # A one-shot grouped backup at the receiver information set is pure.  CFR
    # mixes because the sender's public action changes the type range.
    grouped_pure = grouped_information_set_backup(
        {
            "strong": {"call": 2, "fold": 1},
            "weak": {"call": -2, "fold": 1},
        },
        weights={"strong": Fraction(1, 2), "weak": Fraction(1, 2)},
        sense="min",
    )
    assert grouped_pure.selected_action == "call"
    assert result.receiver_average_strategy["bet"][grouped_pure.selected_action] < 0.95
    assert result.metadata == {
        "method": "reduced_public_signaling_cfr_plus",
        "strategy_fusion": False,
        "equilibrium_approx": True,
        "hu_exact": False,
        "runtime_integrated": False,
        "rust_leaf_integrated": False,
        "position_contract_version": "bb_first_v1",
    }
