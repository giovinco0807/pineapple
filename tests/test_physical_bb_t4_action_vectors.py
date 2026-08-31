from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from ai.engine.encoding import ALL_CARDS, Board
from ai.engine.turn_order import POSITION_CONTRACT_VERSION
from ai.tutor.exact_late import (
    PHYSICAL_T4_KNOWN_DISCARDS_RANGE_MODEL,
    PHYSICAL_T4_REMAINING_CARDS_RANGE_MODEL,
    action_key,
    default_rust_t3_exact_solver_path,
    evaluate_late_position,
    evaluate_physical_bb_t4_action_vector_rust,
    evaluate_physical_bb_t4_action_vectors_rust_batch,
    physical_bb_t4_state_commitment,
)


BB_BOARD = Board(
    top=["Qh", "Qs"],
    middle=["Kh", "Ks", "9d", "8c", "7h"],
    bottom=["Ah", "Ad", "Ac", "5s"],
)
BTN_BOARD = Board(
    top=["2c", "3c"],
    middle=["6h", "6d", "7s", "8s", "Tc"],
    bottom=["Jh", "Jd", "Qc", "Qd"],
)
KNOWN_BB_DISCARDS = ["3d", "6c", "Td"]


def _board_payload(board: Board) -> dict[str, list[str]]:
    return {
        "top": list(board.top),
        "middle": list(board.middle),
        "bottom": list(board.bottom),
    }


def _base_payload(*, particle_id: str, dealt: list[str]) -> dict:
    return {
        "turn": 4,
        "actor": "bb",
        "is_btn": False,
        "first_actor": "bb",
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "particle_id": particle_id,
        "board_self": _board_payload(BB_BOARD),
        "board_opponent": _board_payload(BTN_BOARD),
        "dealt_cards": list(dealt),
    }


def _remaining_payload(*, joker: bool) -> dict:
    dealt = ["X1", "4d", "9s"] if joker else ["4d", "9s", "4c"]
    remaining = ["X2", "4c", "Kc"] if joker else ["2s", "Kc", "5d"]
    payload = _base_payload(
        particle_id="x1-x2-particle" if joker else "natural-particle",
        dealt=dealt,
    )
    payload["remaining_cards"] = remaining
    return payload


def _known_discard_payload(*, particle_id: str, opponent_discards: list[str]) -> dict:
    payload = _base_payload(particle_id=particle_id, dealt=["4d", "9s", "4c"])
    payload["known_discards_self"] = list(KNOWN_BB_DISCARDS)
    payload["known_discards_opponent"] = list(opponent_discards)
    return payload


def _python_reference(payload: dict) -> dict[str, dict]:
    live = (
        set(BB_BOARD.all_cards())
        | set(BTN_BOARD.all_cards())
        | set(payload["dealt_cards"])
        | set(payload["remaining_cards"])
    )
    exclude = [card for card in ALL_CARDS if card not in live]
    result = evaluate_late_position(
        BB_BOARD,
        list(payload["dealt_cards"]),
        4,
        opponent_board=BTN_BOARD,
        exclude=exclude,
        top_n=100,
        prefer_rust=False,
    )
    return {
        action_key(candidate["action"]): candidate["metrics"]
        for candidate in result["candidates"]
    }


@pytest.mark.parametrize("joker", [False, True], ids=["natural", "x1_x2"])
def test_physical_remaining_range_returns_all_candidates_with_python_rust_parity(joker: bool):
    solver = default_rust_t3_exact_solver_path()
    if not Path(solver).exists():
        pytest.skip(f"Rust late exact solver is not built: {solver}")

    payload = _remaining_payload(joker=joker)
    result = evaluate_physical_bb_t4_action_vector_rust(payload, timeout_s=10.0)
    reference = _python_reference(payload)

    assert result["particle_id"] == payload["particle_id"]
    assert result["physical_state_commitment"] == physical_bb_t4_state_commitment(payload)
    assert result["position_contract_version"] == POSITION_CONTRACT_VERSION == "bb_first_v1"
    assert result["actor"] == result["position"] == result["first_actor"] == "bb"
    assert result["is_btn"] is False
    assert result["candidate_scope"] == "all_legal"
    assert result["selection_performed"] is False
    assert result["particle_aggregation_performed"] is False
    assert result["exact_scope"] == "t4_conditioned_physical_range_all_draws_best_response"
    assert result["range_model"] == PHYSICAL_T4_REMAINING_CARDS_RANGE_MODEL
    assert result["conditioned_range_exact"] is True
    assert result["terminal_response_exact"] is True
    assert result["hu_exact"] is False
    assert result["equilibrium_exact"] is False
    assert result["private_physical_input_fields_omitted"] is True
    assert result["particle_id_contract"] == "caller_supplied_opaque_diagnostic_only"
    assert result["public_policy_safe"] is False
    assert result["requires_infoset_aggregation"] is True
    assert result["remaining_card_count"] == 3
    assert result["legal_actions"] == result["evaluated_actions"] == 6
    assert result["candidate_count"] == len(result["action_keys"]) == 6
    assert result["action_keys"] == sorted(result["action_keys"])
    assert set(result["action_keys"]) == set(reference)
    assert list(result["metrics_by_action_key"]) == result["action_keys"]
    assert list(result["actions_by_action_key"]) == result["action_keys"]
    assert "best" not in result
    assert "chosen_action" not in result
    assert "remaining_cards" not in result
    assert "known_discards_opponent" not in result
    assert "exclude" not in result

    for index, key in enumerate(result["action_keys"]):
        metrics = result["metrics_by_action_key"][key]
        assert result["metrics_vector"][index] == metrics
        assert action_key(result["actions_by_action_key"][key]) == key
        assert metrics["source"] == "exact_hu_response"
        assert metrics["opponent_response"] is True
        assert metrics["start_turn"] == 4
        assert metrics["remaining_deck_size"] == 3
        assert metrics["enumerated_draws"] == 1
        for field in ("score", "raw_score", "royalty", "bust_rate", "fl_rate"):
            assert metrics[field] == pytest.approx(reference[key][field])


def test_physical_hidden_discard_particles_change_the_complete_action_vector():
    solver = default_rust_t3_exact_solver_path()
    if not Path(solver).exists():
        pytest.skip(f"Rust late exact solver is not built: {solver}")

    jokers_dead = _known_discard_payload(
        particle_id="jokers-dead",
        opponent_discards=["X1", "X2", "2d"],
    )
    jokers_live = _known_discard_payload(
        particle_id="jokers-live",
        opponent_discards=["5c", "5d", "2d"],
    )
    results = evaluate_physical_bb_t4_action_vectors_rust_batch(
        [jokers_dead, jokers_live],
        timeout_s=20.0,
    )

    assert [result["particle_id"] for result in results] == ["jokers-dead", "jokers-live"]
    assert all(
        result["range_model"] == PHYSICAL_T4_KNOWN_DISCARDS_RANGE_MODEL
        for result in results
    )
    assert all(result["remaining_card_count"] == 23 for result in results)
    assert results[0]["action_keys"] == results[1]["action_keys"]
    assert any(
        results[0]["metrics_by_action_key"][key]["score"]
        != pytest.approx(results[1]["metrics_by_action_key"][key]["score"])
        for key in results[0]["action_keys"]
    )
    for result in results:
        assert "known_discards_self" not in result
        assert "known_discards_opponent" not in result


def test_physical_t4_batch_rejects_duplicate_particle_ids_before_rust():
    payload = _remaining_payload(joker=False)
    with pytest.raises(ValueError, match="particle_id values must be unique"):
        evaluate_physical_bb_t4_action_vectors_rust_batch(
            [payload, deepcopy(payload)],
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row.update(actor="btn", is_btn=True), "canonical actor='bb'"),
        (lambda row: row.update(turn=4.0), "integer turn=4"),
        (lambda row: row.pop("particle_id"), "non-empty particle_id"),
        (lambda row: row.update(exclude=[]), "unsupported fields"),
        (lambda row: row.update(public_exclude=[]), "unsupported fields"),
        (lambda row: row.update(candidate_actions=[]), "unsupported fields"),
        (lambda row: row.update(opponent_private_draw=["2h"]), "unsupported fields"),
        (
            lambda row: row.update(
                known_discards_self=list(KNOWN_BB_DISCARDS),
                known_discards_opponent=["X1", "X2", "2d"],
            ),
            "not both",
        ),
        (
            lambda row: (row.pop("remaining_cards"), row.update(known_discards_self=[])),
            "requires both known_discards_self",
        ),
        (
            lambda row: row.update(remaining_cards=["2s", "Kc"]),
            "at least 3 BTN final-draw cards",
        ),
        (
            lambda row: row.update(remaining_cards=["Qh", "Kc", "5d"]),
            "duplicate physical card",
        ),
        (
            lambda row: row.update(remaining_cards=["JK", "Kc", "5d"]),
            "Jokers must be identified as X1 or X2",
        ),
        (
            lambda row: row["board_self"].update(mid=row["board_self"].pop("middle")),
            "exactly top/middle/bottom rows",
        ),
    ],
)
def test_physical_t4_api_rejects_ambiguous_malformed_or_leaky_inputs(mutate, message: str):
    payload = deepcopy(_remaining_payload(joker=False))
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        evaluate_physical_bb_t4_action_vector_rust(payload)
