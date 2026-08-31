import pytest

from ai.engine.game_engine import evaluate_hand, hand_category
from ai.tutor.fl14_allocation_features import (
    ALLOCATION_RANK_SIZE,
    allocation_rank_block,
    partial_tiebreaks,
)
from ai.tutor.encode_fl14_teacher import FEATURE_SIZE_V1, FEATURE_SIZE_V2
from ai.tutor.fl14_encoder_parity import block_ranges
from ai.tutor.t4_first_features import _spread


def test_partial_tiebreaks_exposes_incomplete_pair_rank_and_kicker():
    assert partial_tiebreaks(["Ad", "Ah"], 3) == pytest.approx((1.0, 0.0))
    assert partial_tiebreaks(["3h", "3s", "8d", "Qh"], 5) == pytest.approx(
        (3 / 14, 12 / 14)
    )


def test_partial_tiebreaks_orders_two_pair_and_joker_group():
    assert partial_tiebreaks(["6d", "6h", "Qd", "Qs"], 5) == pytest.approx(
        (12 / 14, 6 / 14)
    )
    assert partial_tiebreaks(["Ac", "X1"], 3) == pytest.approx((1.0, 0.0))
    assert partial_tiebreaks(["Kc", "X1", "X2"], 5) == pytest.approx(
        (13 / 14, 0.0)
    )


@pytest.mark.parametrize(
    "cards",
    [
        ["As", "2d", "3c", "4h", "5s"],
        ["9s", "Ts", "Js", "Qs", "Ks"],
        ["Ac", "Ad", "X1"],
    ],
)
def test_complete_rows_reuse_exact_evaluator_tiebreaks(cards):
    value = evaluate_hand(cards, len(cards))
    expected = _spread(value, hand_category(value))[-2:]
    assert partial_tiebreaks(cards, len(cards)) == pytest.approx(expected)


def test_allocation_block_exposes_same_category_kicker_change():
    low_middle = allocation_rank_block(
        [["As"], ["3h", "3s", "7s", "8d"], ["Jc", "Kc", "X1"]]
    )
    high_middle = allocation_rank_block(
        [["As"], ["3h", "3s", "8d", "Qh"], ["Jc", "Kc", "X1"]]
    )
    assert len(low_middle) == len(high_middle) == ALLOCATION_RANK_SIZE
    assert low_middle[:2] == high_middle[:2]
    assert low_middle[2] == high_middle[2] == pytest.approx(3 / 14)
    assert low_middle[3] == pytest.approx(8 / 14)
    assert high_middle[3] == pytest.approx(12 / 14)


def test_empty_board_has_finite_zero_block():
    assert allocation_rank_block([[], [], []]) == [0.0] * ALLOCATION_RANK_SIZE


def test_fl14_v2_keeps_the_legacy_prefix():
    assert (FEATURE_SIZE_V1, FEATURE_SIZE_V2) == (104, 110)
    ranges = dict((name, (start, stop)) for name, start, stop in block_ranges(110))
    assert ranges["allocation_rank"] == (104, 110)


@pytest.mark.parametrize("unsupported", [109, 113, 116, 123])
def test_non_fl14_widths_are_not_accepted_as_fl14_encoders(unsupported):
    with pytest.raises(ValueError, match="unsupported FL14 feature width"):
        block_ranges(unsupported)
