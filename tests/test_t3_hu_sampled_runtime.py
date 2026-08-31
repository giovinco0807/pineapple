from pathlib import Path

import pytest

from ai.engine.encoding import ALL_CARDS, Board
from ai.tutor.exact_late import default_rust_t3_exact_solver_path
from ai.tutor.t3_hu_sampled import evaluate_t3_hu_sampled


pytestmark = pytest.mark.skipif(
    not Path(default_rust_t3_exact_solver_path()).exists(),
    reason="Rust late exact solver is not built",
)


BTN9 = Board(
    top=["5c", "6d", "8c"],
    middle=["9c", "9d", "Jh", "Qs", "Kc"],
    bottom=["As"],
)
BB9 = Board(
    top=["2c", "3d", "4h"],
    middle=["7c", "7d", "8h", "9s", "Tc"],
    bottom=["Qc"],
)
BB11 = Board(
    top=["2c", "3d", "4h"],
    middle=["7c", "7d", "8h", "9s", "Tc"],
    bottom=["Qc", "Qd", "Kh"],
)


def _board_dict(board):
    return {"top": list(board.top), "middle": list(board.middle), "bottom": list(board.bottom)}


def _public_exclude(*, boards, dealt, known_self, pimc_cards):
    keep = set(dealt) | set(known_self) | set(pimc_cards)
    for board in boards:
        keep.update(board.all_cards())
    return [card for card in ALL_CARDS if card not in keep]


def test_t3_sampled_rejects_ambiguous_legacy_exclude_identity():
    dealt = ["2s", "Jd", "Ts"]
    known_self = ["6c", "6s"]
    pimc_cards = ["X1", "X2", "Ac", "Kd", "Ks", "Jc", "Td", "4s", "7h"]
    payload = {
        "turn": 3,
        "position": "btn",
        "first_actor": "bb",
        "board": _board_dict(BTN9),
        "opponent_board": _board_dict(BB11),
        "dealt": dealt,
        "known_discards_self": known_self,
        "public_exclude": _public_exclude(
            boards=[BTN9, BB11],
            dealt=dealt,
            known_self=known_self,
            pimc_cards=pimc_cards,
        ),
        "exclude": ["4s"],
    }

    with pytest.raises(ValueError, match="opponent private discards"):
        evaluate_t3_hu_sampled(payload)


def test_btn_t3_sampled_runtime_is_reproducible_and_uses_exact_t4_leaves():
    dealt = ["2s", "Jd", "Ts"]
    known_self = ["6c", "6s"]
    pimc_cards = ["X1", "X2", "Ac", "Kd", "Ks", "Jc", "Td", "4s", "7h"]
    payload = {
        "turn": 3,
        "actor": "btn",
        "first_actor": "bb",
        "board": _board_dict(BTN9),
        "opponent_board": _board_dict(BB11),
        "dealt": dealt,
        "known_discards_self": known_self,
        "public_exclude": _public_exclude(
            boards=[BTN9, BB11],
            dealt=dealt,
            known_self=known_self,
            pimc_cards=pimc_cards,
        ),
    }

    first = evaluate_t3_hu_sampled(
        payload,
        seed=17,
        outer_samples=2,
        include_scenario_values=True,
    )
    second = evaluate_t3_hu_sampled(
        payload,
        seed=17,
        outer_samples=2,
        include_scenario_values=True,
    )

    assert first["method"] == "hu_sampled_pimc_exact_t4"
    assert first["information_model"] == "pimc_determinization_v1"
    assert first["hu_exact"] is False
    assert first["inner_t4_exact"] is True
    assert first["position"] == "btn"
    assert first["position_contract_version"] == "bb_first_v1"
    assert first["legal_actions"] == first["evaluated_actions"] == 3
    assert first["leaf_positions"] == 6
    assert all(len(candidate["scenario_values"]) == 2 for candidate in first["candidates"])
    assert first["chosen_action"] == second["chosen_action"]
    assert [candidate["metrics"]["score"] for candidate in first["candidates"]] == [
        candidate["metrics"]["score"] for candidate in second["candidates"]
    ]


def test_bb_t3_sampled_runtime_averages_future_before_btn_response_selection():
    dealt = ["Ad", "Ts", "Qh"]
    known_self = ["6c", "6s"]
    pimc_cards = ["X1", "X2", "Ac", "Kd", "Ks", "Jc", "Td", "5s", "Qd", "4s", "Th"]
    payload = {
        "turn": 3,
        "actor": "bb",
        "first_actor": "bb",
        "board": _board_dict(BB9),
        "opponent_board": _board_dict(BTN9),
        "dealt": dealt,
        "known_discards_self": known_self,
        "public_exclude": _public_exclude(
            boards=[BB9, BTN9],
            dealt=dealt,
            known_self=known_self,
            pimc_cards=pimc_cards,
        ),
    }

    result = evaluate_t3_hu_sampled(
        payload,
        seed=23,
        outer_samples=2,
        response_inner_samples=2,
        include_scenario_values=True,
    )

    assert result["position"] == "bb"
    assert result["legal_actions"] == result["evaluated_actions"] == 3
    assert result["outer_samples"] == 2
    assert result["response_inner_samples"] == 2
    assert result["response_selection"] == "mean_before_min"
    assert result["leaf_positions"] == 3 * 2 * 3 * 2
    assert all(len(candidate["scenario_values"]) == 2 for candidate in result["candidates"])
    assert result["paired_gap"]["samples"] == 2
