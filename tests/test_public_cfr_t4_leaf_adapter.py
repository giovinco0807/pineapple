from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from ai.engine.action_space import get_turn_actions
from ai.engine.encoding import ALL_CARDS, Board
from ai.engine.turn_order import POSITION_CONTRACT_VERSION
from ai.tutor import exact_late
from ai.tutor.exact_late import (
    action_key,
    action_to_dict,
    default_rust_t3_exact_solver_path,
    evaluate_late_position,
    evaluate_public_cfr_bb_t4_leaf_rust,
    evaluate_public_cfr_bb_t4_leaves_rust_batch,
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


def _player_view(*, joker: bool) -> dict:
    dealt = ["X1", "4d", "9s"] if joker else ["4d", "9s", "4c"]
    future = (
        ["X2", "4c", "Kc", "5d", "2s", "7c"]
        if joker
        else ["2s", "Kc", "5d", "7c", "8d", "9h"]
    )
    retained = (
        set(BB_BOARD.all_cards())
        | set(BTN_BOARD.all_cards())
        | set(dealt)
        | set(KNOWN_BB_DISCARDS)
        | set(future)
    )
    return {
        "turn": 4,
        "actor": "bb",
        "is_btn": False,
        "first_actor": "bb",
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "board_self": _board_payload(BB_BOARD),
        "board_opponent": _board_payload(BTN_BOARD),
        "dealt_cards": dealt,
        "known_discards_self": list(KNOWN_BB_DISCARDS),
        "public_exclude": [card for card in ALL_CARDS if card not in retained],
    }


def _reference_by_action(payload: dict, *, prefer_rust: bool) -> dict[str, dict]:
    result = evaluate_late_position(
        BB_BOARD,
        list(payload["dealt_cards"]),
        4,
        opponent_board=BTN_BOARD,
        exclude=[*payload["known_discards_self"], *payload["public_exclude"]],
        top_n=50,
        prefer_rust=prefer_rust,
        rust_timeout_s=10.0,
        fallback_on_rust_error=False,
    )
    assert result["candidate_count"] == result["legal_actions"] == 6
    assert len(result["candidates"]) == 6
    return {
        action_key(candidate["action"]): candidate["metrics"]
        for candidate in result["candidates"]
    }


@pytest.mark.parametrize("joker", [False, True], ids=["natural", "x1_x2"])
def test_public_cfr_t4_leaf_returns_all_candidates_with_python_and_rust_parity(joker: bool):
    solver = default_rust_t3_exact_solver_path()
    if not Path(solver).exists():
        pytest.skip(f"Rust late exact solver is not built: {solver}")

    payload = _player_view(joker=joker)
    leaf = evaluate_public_cfr_bb_t4_leaf_rust(
        payload,
        timeout_s=10.0,
        allow_synthetic_public_exclude=True,
    )
    rust_reference = _reference_by_action(payload, prefer_rust=True)
    python_reference = _reference_by_action(payload, prefer_rust=False)

    assert leaf["position_contract_version"] == POSITION_CONTRACT_VERSION == "bb_first_v1"
    assert leaf["actor"] == leaf["position"] == "bb"
    assert leaf["is_btn"] is False
    assert leaf["candidate_scope"] == "all_legal"
    assert leaf["selection_performed"] is False
    assert leaf["exact_scope"] == "t4_uniform_hidden_discard_marginal_all_draws_best_response"
    assert leaf["range_model"] == "uniform_hidden_discards_no_history_v1"
    assert leaf["uniform_range_exact"] is True
    assert leaf["terminal_response_exact"] is True
    assert leaf["metrics_source"] == "exact_hu_response"
    assert leaf["hu_exact"] is False
    assert leaf["synthetic_public_exclude_used"] is True
    assert leaf["legal_actions"] == leaf["evaluated_actions"] == leaf["candidate_count"] == 6
    assert len(leaf["action_keys"]) == len(leaf["metrics_vector"]) == 6
    assert leaf["action_keys"] == sorted(leaf["action_keys"])
    assert set(leaf["action_keys"]) == set(rust_reference) == set(python_reference)
    assert list(leaf["metrics_by_action_key"]) == leaf["action_keys"]
    assert list(leaf["actions_by_action_key"]) == leaf["action_keys"]
    assert "best" not in leaf
    assert "chosen_action" not in leaf

    for index, key in enumerate(leaf["action_keys"]):
        metrics = leaf["metrics_by_action_key"][key]
        assert leaf["metrics_vector"][index] == metrics
        assert action_key(leaf["actions_by_action_key"][key]) == key
        assert metrics["source"] == "exact_hu_response"
        assert metrics["opponent_response"] is True
        assert metrics["start_turn"] == 4
        for field in ("score", "raw_score", "royalty", "bust_rate", "fl_rate"):
            assert metrics[field] == pytest.approx(rust_reference[key][field])
            assert metrics[field] == pytest.approx(python_reference[key][field])


def test_public_cfr_t4_leaf_batch_preserves_input_order_and_full_coverage():
    solver = default_rust_t3_exact_solver_path()
    if not Path(solver).exists():
        pytest.skip(f"Rust late exact solver is not built: {solver}")

    natural = _player_view(joker=False)
    joker = _player_view(joker=True)
    results = evaluate_public_cfr_bb_t4_leaves_rust_batch(
        [natural, joker],
        timeout_s=10.0,
        allow_synthetic_public_exclude=True,
    )

    assert len(results) == 2
    assert results[0]["dealt"] == natural["dealt_cards"]
    assert results[1]["dealt"] == joker["dealt_cards"]
    assert all(result["candidate_count"] == 6 for result in results)
    assert all(len(result["metrics_vector"]) == 6 for result in results)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row.update(actor="btn", is_btn=True), "only accepts BB"),
        (
            lambda row: (row.pop("actor"), row.pop("is_btn")),
            "requires an explicit actor/position",
        ),
        (
            lambda row: row.pop("position_contract_version"),
            "requires explicit position contract",
        ),
        (
            lambda row: row.pop("first_actor"),
            "requires explicit first_actor",
        ),
        (
            lambda row: row["board_opponent"]["top"].append("2h"),
            "requires hero/opponent board counts 11/11",
        ),
        (
            lambda row: row.update(opponent_private_discards=[]),
            "private or ambiguous discard field",
        ),
        (
            lambda row: row.update(
                public_action_history=[{"turn": 3, "actor": "btn", "discard": "2h"}]
            ),
            "private or ambiguous discard field",
        ),
        (lambda row: row.update(exclude=[]), "ambiguous exclude is forbidden"),
        (lambda row: row.update(candidate_actions=[]), "candidate filtering field"),
        (
            lambda row: row["known_discards_self"].pop(),
            "requires exactly 3 prior BB discards",
        ),
    ],
)
def test_public_cfr_t4_leaf_rejects_noncanonical_or_private_inputs(mutate, message: str):
    payload = deepcopy(_player_view(joker=False))
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        evaluate_public_cfr_bb_t4_leaf_rust(
            payload,
            allow_synthetic_public_exclude=True,
        )


def test_public_cfr_t4_leaf_rejects_synthetic_public_exclude_by_default():
    with pytest.raises(ValueError, match="public_exclude is disabled"):
        evaluate_public_cfr_bb_t4_leaf_rust(_player_view(joker=False))


def test_public_cfr_t4_leaf_requires_hidden_discard_and_final_draw_capacity():
    payload = _player_view(joker=False)
    payload["public_exclude"].append("2s")

    with pytest.raises(ValueError, match="at least 6 possible unseen cards"):
        evaluate_public_cfr_bb_t4_leaf_rust(
            payload,
            allow_synthetic_public_exclude=True,
        )


def test_public_cfr_t4_leaf_asserts_full_rust_candidate_coverage(monkeypatch):
    payload = _player_view(joker=False)
    actions = get_turn_actions(payload["dealt_cards"], BB_BOARD)
    assert len(actions) == 6

    def incomplete_batch(_payloads, **_kwargs):
        candidates = [
            {
                "action": action_to_dict(action),
                "metrics": {
                    "source": "exact_hu_response",
                    "opponent_response": True,
                    "start_turn": 4,
                    "score": 0.0,
                },
            }
            for action in actions[:-1]
        ]
        return [
            {
                "legal_actions": 6,
                "evaluated_actions": 6,
                "candidates": candidates,
                "elapsed_ms": 0.0,
            }
        ]

    monkeypatch.setattr(exact_late, "evaluate_late_positions_rust_batch", incomplete_batch)
    with pytest.raises(RuntimeError, match="action coverage mismatch"):
        evaluate_public_cfr_bb_t4_leaf_rust(
            payload,
            allow_synthetic_public_exclude=True,
        )


def test_public_cfr_t4_leaf_rejects_a_malformed_cross_language_action(monkeypatch):
    payload = _player_view(joker=False)
    actions = get_turn_actions(payload["dealt_cards"], BB_BOARD)

    def malformed_batch(_payloads, **_kwargs):
        candidates = []
        for index, action in enumerate(actions):
            raw = action_to_dict(action)
            if index == 0:
                raw["placements"].append(["not-a-card", "sideways"])
            candidates.append(
                {
                    "action": raw,
                    "metrics": {
                        "source": "exact_hu_response",
                        "opponent_response": True,
                        "start_turn": 4,
                        "score": 0.0,
                    },
                }
            )
        return [
            {
                "legal_actions": len(actions),
                "evaluated_actions": len(actions),
                "candidates": candidates,
                "elapsed_ms": 0.0,
            }
        ]

    monkeypatch.setattr(exact_late, "evaluate_late_positions_rust_batch", malformed_batch)
    with pytest.raises(RuntimeError, match="invalid action"):
        evaluate_public_cfr_bb_t4_leaf_rust(
            payload,
            allow_synthetic_public_exclude=True,
        )
