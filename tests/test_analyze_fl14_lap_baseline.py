import math

import pytest

from ai.tutor.analyze_fl14_lap_baseline import summarise


def _finished(ident, *, value, busted, royalty, unfouled, width=0):
    return {
        "id": ident,
        "value": value,
        "busted": busted,
        "royalty": royalty,
        "unfouled": unfouled,
        "entry_width": width,
    }


def _position(ident, shape, cards):
    rows = []
    offset = 0
    for count in shape:
        rows.append(cards[offset : offset + count])
        offset += count
    return {"id": ident, "rows": rows, "dead": [], "draw": []}


def test_summarise_attributes_realised_forced_fouls_by_t3_shape():
    finished = [
        _finished("a", value=-8, busted=True, royalty=0, unfouled=0),
        _finished("b", value=12, busted=False, royalty=4, unfouled=1, width=15),
        _finished("c", value=7, busted=False, royalty=2, unfouled=3),
    ]
    positions = [
        _position("a", (2, 5, 4), ["As", "Ah", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "Ts"]),
        _position("b", (2, 5, 4), ["Ks", "Kh", "2h", "3h", "4h", "5h", "6h", "7h", "8h", "9h", "Th"]),
        _position("c", (3, 4, 4), ["Qs", "Qh", "Qd", "2d", "3d", "4d", "5d", "6d", "7d", "8d", "9d"]),
    ]

    result = summarise(finished, positions)

    assert result["hands"] == 3
    assert result["forced_fouls"] == 1
    assert result["chosen_fouls"] == 0
    assert result["foul_rate"]["mean"] == pytest.approx(1 / 3)
    assert result["royalty_foul_zero"]["mean"] == pytest.approx(2.0)
    assert result["fl_entry_ev"]["mean"] == pytest.approx(10.7 / 3)
    assert result["t3_board_shapes"]["2-5-4"]["foul_rate"] == pytest.approx(0.5)
    assert result["t3_board_shapes"]["2-5-4"]["foul_contribution"] == 1.0
    assert result["legal_t4_actions_on_realised_draw"] == {"0": 1, "1": 1, "3": 1}
    assert not math.isnan(result["score"]["stderr"])


def test_summarise_refuses_misaligned_position_ids():
    finished = [_finished("a", value=0, busted=False, royalty=0, unfouled=1)]
    positions = [_position("b", (3, 4, 4), [str(index) for index in range(11)])]

    with pytest.raises(ValueError, match="finished/position ids differ"):
        summarise(finished, positions)
