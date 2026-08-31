from fractions import Fraction

from ai.tutor.t3_hu_public_cfr import (
    bayes_posterior,
    epsilon_smoothed_likelihood,
    grouped_information_set_backup,
)


def test_bb_t4_groups_hidden_btn_discards_before_max():
    result = grouped_information_set_backup(
        {
            "hidden_btn_discards_a": {"A": 4, "B": 0},
            "hidden_btn_discards_b": {"A": -2, "B": 3},
        },
        weights={
            "hidden_btn_discards_a": Fraction(1, 2),
            "hidden_btn_discards_b": Fraction(1, 2),
        },
        sense="max",
    )

    assert result.expected_action_values == {"A": Fraction(1), "B": Fraction(3, 2)}
    assert result.selected_action == "B"
    assert result.selected_value == Fraction(3, 2)
    assert result.per_world_actions == {
        "hidden_btn_discards_a": "A",
        "hidden_btn_discards_b": "B",
    }
    assert result.pimc_value == Fraction(7, 2)
    assert result.strategy_fusion_advantage == Fraction(2)
    assert result.metadata == {
        "method": "grouped_information_set_backup",
        "strategy_fusion": False,
        "equilibrium_approx": False,
        "hu_exact": False,
        "position_contract_version": "bb_first_v1",
    }


def test_btn_t3_groups_hidden_bb_discards_before_min():
    result = grouped_information_set_backup(
        {
            "hidden_bb_discards_a": {"X": -4, "Y": 0},
            "hidden_bb_discards_b": {"X": 3, "Y": -2},
        },
        sense="min",
    )

    assert result.expected_action_values == {"X": Fraction(-1, 2), "Y": Fraction(-1)}
    assert result.selected_action == "Y"
    assert result.selected_value == Fraction(-1)
    assert result.per_world_actions == {
        "hidden_bb_discards_a": "X",
        "hidden_bb_discards_b": "Y",
    }
    assert result.pimc_value == Fraction(-3)
    assert result.strategy_fusion_advantage == Fraction(2)


def test_public_action_likelihood_updates_hidden_discard_range_exactly():
    posterior = bayes_posterior(
        {"discard_a": Fraction(1, 2), "discard_b": Fraction(1, 2)},
        {"discard_a": Fraction(3, 4), "discard_b": Fraction(1, 4)},
        epsilon=Fraction(0),
        action_count=2,
    )

    assert posterior == {
        "discard_a": Fraction(3, 4),
        "discard_b": Fraction(1, 4),
    }


def test_epsilon_smoothing_keeps_every_behavior_hypothesis_reachable():
    assert epsilon_smoothed_likelihood(
        1,
        epsilon=Fraction(1, 10),
        action_count=2,
    ) == Fraction(19, 20)
    assert epsilon_smoothed_likelihood(
        0,
        epsilon=Fraction(1, 10),
        action_count=2,
    ) == Fraction(1, 20)
    posterior = bayes_posterior(
        {"likely": 1, "off_path": 1},
        {"likely": 1, "off_path": 0},
        epsilon=Fraction(1, 10),
        action_count=2,
    )
    assert posterior == {"likely": Fraction(19, 20), "off_path": Fraction(1, 20)}


def test_epsilon_smoothing_never_revives_a_physically_impossible_world():
    posterior = bayes_posterior(
        {"legal_off_path": 1, "missing_observed_card": 1},
        {"legal_off_path": 0, "missing_observed_card": 0},
        epsilon=Fraction(1, 10),
        action_count=2,
        physically_possible={"legal_off_path": True, "missing_observed_card": False},
    )

    assert posterior == {
        "legal_off_path": Fraction(1),
        "missing_observed_card": Fraction(0),
    }
