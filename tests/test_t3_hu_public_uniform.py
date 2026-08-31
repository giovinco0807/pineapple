from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from ai.engine.action_space import get_turn_actions
from ai.engine.encoding import ALL_CARDS, Board
from ai.tutor.exact_late import default_rust_t3_exact_solver_path
from ai.tutor import t3_hu_public_uniform as public_uniform
from ai.tutor.t3_hu_public_uniform import evaluate_btn_t3_public_uniform


pytestmark = pytest.mark.skipif(
    not Path(default_rust_t3_exact_solver_path()).exists(),
    reason="Rust late exact solver is not built",
)


BTN9 = Board(
    top=["5c", "6d", "8c"],
    middle=["9c", "9d", "Jh", "Qs", "Kc"],
    bottom=["As"],
)
BB11 = Board(
    top=["2c", "3d", "4h"],
    middle=["7c", "7d", "8h", "9s", "Tc"],
    bottom=["Qc", "Qd", "Kh"],
)
HIDDEN_A = ["6c", "6s"]
HIDDEN_B = ["3s", "4s"]
BB_PRIVATE = ["5s", "7h", "Th"]
EXTRA_PUBLIC_RANGE = ["Jc", "Td", "4d", "5d"]


def _board_payload(board: Board) -> dict[str, list[str]]:
    return {
        "top": list(board.top),
        "middle": list(board.middle),
        "bottom": list(board.bottom),
    }


def _fixture(*, joker: bool, hidden: list[str]) -> tuple[dict, list[dict]]:
    dealt = ["2s", "X1", "Ts"] if joker else ["2s", "Jd", "Ts"]
    bb_draw = ["X2", "Kd", "Ks"] if joker else ["Ac", "Kd", "Ks"]
    retained = (
        set(BTN9.all_cards())
        | set(BB11.all_cards())
        | set(dealt)
        | set(HIDDEN_A)
        | set(HIDDEN_B)
        | set(BB_PRIVATE)
        | set(bb_draw)
        | set(EXTRA_PUBLIC_RANGE)
    )
    payload = {
        "turn": 3,
        "actor": "btn",
        "is_btn": True,
        "first_actor": "bb",
        "position_contract_version": "bb_first_v1",
        "board": _board_payload(BTN9),
        "opponent_board": _board_payload(BB11),
        "dealt": dealt,
        "known_discards_self": list(hidden),
        "public_exclude": [card for card in ALL_CARDS if card not in retained],
    }
    scenarios = [
        {
            "bb_private_discards": list(BB_PRIVATE),
            "bb_t4_draw": bb_draw,
        }
    ]
    return payload, scenarios


def _candidate_map(result: dict) -> dict[str, dict]:
    return {candidate["action_key"]: candidate for candidate in result["candidates"]}


@pytest.mark.parametrize("joker", [False, True], ids=["natural", "x1_x2"])
def test_btn_public_uniform_policy_is_hidden_discard_invariant_but_physical_value_can_change(
    joker: bool,
    monkeypatch,
):
    payload_a, scenarios = _fixture(joker=joker, hidden=HIDDEN_A)
    payload_b, _ = _fixture(joker=joker, hidden=HIDDEN_B)
    original_policy_batch = public_uniform.evaluate_public_cfr_bb_t4_leaves_rust_batch
    captured: list[list[dict]] = []

    def capture_policy(payloads, **kwargs):
        captured.append(deepcopy(list(payloads)))
        return original_policy_batch(payloads, **kwargs)

    monkeypatch.setattr(
        public_uniform,
        "evaluate_public_cfr_bb_t4_leaves_rust_batch",
        capture_policy,
    )
    result_a = evaluate_btn_t3_public_uniform(
        payload_a,
        scenario_overrides=scenarios,
        include_scenario_values=True,
        rust_timeout_s=30.0,
        allow_synthetic_public_exclude=True,
    )
    result_b = evaluate_btn_t3_public_uniform(
        payload_b,
        scenario_overrides=scenarios,
        include_scenario_values=True,
        rust_timeout_s=30.0,
        allow_synthetic_public_exclude=True,
    )

    assert result_a["strategy_fusion"] is False
    assert result_a["range_model"] == "uniform_hidden_discards_no_history_v1"
    assert result_a["equilibrium_approx"] is False
    assert result_a["hu_exact"] is False
    assert result_a["inner_t4_exact"] is True
    assert result_a["position"] == "btn"
    assert result_a["legal_actions"] == result_a["evaluated_actions"] == 3
    assert result_a["policy_leaf_positions"] == result_a["physical_leaf_positions"] == 3
    assert result_a["chance_tape_shared_across_root_actions"] is True

    by_action_a = _candidate_map(result_a)
    by_action_b = _candidate_map(result_b)
    assert set(by_action_a) == set(by_action_b)
    assert {
        key: candidate["scenario_policy_action_keys"] for key, candidate in by_action_a.items()
    } == {
        key: candidate["scenario_policy_action_keys"] for key, candidate in by_action_b.items()
    }
    assert any(
        by_action_a[key]["scenario_values"] != by_action_b[key]["scenario_values"]
        for key in by_action_a
    )

    assert len(captured) == 2
    root_actions = get_turn_actions(payload_a["dealt"], BTN9)
    assert len(root_actions) == len(captured[0]) == 3
    for root_action, policy_payload in zip(root_actions, captured[0]):
        serialized = json.dumps(policy_payload, sort_keys=True)
        assert policy_payload["known_discards_self"] == BB_PRIVATE
        assert policy_payload["public_exclude"] == payload_a["public_exclude"]
        assert "exclude" not in policy_payload
        assert all(card not in serialized for card in HIDDEN_A)
        assert root_action.discard not in serialized

    # Changing only BTN's hidden discards must not alter any BB policy payload.
    assert captured[0] == captured[1]


def test_btn_public_uniform_default_tape_is_reproducible_shared_and_covers_all_actions(monkeypatch):
    payload, _ = _fixture(joker=False, hidden=HIDDEN_A)
    original_policy_batch = public_uniform.evaluate_public_cfr_bb_t4_leaves_rust_batch
    captured: list[list[dict]] = []

    def capture_policy(payloads, **kwargs):
        captured.append(deepcopy(list(payloads)))
        return original_policy_batch(payloads, **kwargs)

    monkeypatch.setattr(
        public_uniform,
        "evaluate_public_cfr_bb_t4_leaves_rust_batch",
        capture_policy,
    )
    first = evaluate_btn_t3_public_uniform(
        payload,
        seed=17,
        outer_samples=2,
        include_scenario_values=True,
        rust_timeout_s=30.0,
        allow_synthetic_public_exclude=True,
    )
    second = evaluate_btn_t3_public_uniform(
        payload,
        seed=17,
        outer_samples=2,
        include_scenario_values=True,
        rust_timeout_s=30.0,
        allow_synthetic_public_exclude=True,
    )

    assert first["chance_source"] == "deterministic_priority"
    assert first["outer_samples"] == 2
    assert first["legal_actions"] == first["evaluated_actions"] == 3
    assert first["chosen_action"] == second["chosen_action"]
    assert [candidate["metrics"]["score"] for candidate in first["candidates"]] == [
        candidate["metrics"]["score"] for candidate in second["candidates"]
    ]
    assert [candidate["scenario_policy_action_keys"] for candidate in first["candidates"]] == [
        candidate["scenario_policy_action_keys"] for candidate in second["candidates"]
    ]
    assert len(captured) == 2
    assert captured[0] == captured[1]

    # Records are root-major.  Each scenario's BB private tape is identical
    # across all root actions.
    first_batch = captured[0]
    for scenario_index in range(2):
        scenario_records = [first_batch[root_index * 2 + scenario_index] for root_index in range(3)]
        assert len({tuple(row["known_discards_self"]) for row in scenario_records}) == 1
        assert len({tuple(row["dealt_cards"]) for row in scenario_records}) == 1


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row.update(actor="bb", is_btn=False), "only accepts actor='btn'"),
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
            lambda row: row.update(opponent_private_discards=["2h", "3h"]),
            "opponent-private or ambiguous discard field",
        ),
        (lambda row: row.update(exclude=[]), "forbids ambiguous field 'exclude'"),
        (
            lambda row: row["opponent_board"]["bottom"].append("2h"),
            "requires hero/opponent board counts 9/11",
        ),
    ],
)
def test_btn_public_uniform_rejects_bb_root_ambiguous_and_private_inputs(mutate, message: str):
    payload, scenarios = _fixture(joker=False, hidden=HIDDEN_A)
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        evaluate_btn_t3_public_uniform(
            payload,
            scenario_overrides=scenarios,
            allow_synthetic_public_exclude=True,
        )


def test_btn_public_uniform_requires_positive_outer_samples():
    payload, _ = _fixture(joker=False, hidden=HIDDEN_A)
    with pytest.raises(ValueError, match="outer_samples must be a positive integer"):
        evaluate_btn_t3_public_uniform(payload, outer_samples=0)


def test_btn_public_uniform_rejects_synthetic_public_exclude_by_default():
    payload, scenarios = _fixture(joker=False, hidden=HIDDEN_A)
    with pytest.raises(ValueError, match="public_exclude is disabled"):
        evaluate_btn_t3_public_uniform(payload, scenario_overrides=scenarios)
