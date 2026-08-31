from copy import deepcopy
from fractions import Fraction
from pathlib import Path

import pytest

from ai.engine.action_space import get_turn_actions
from ai.engine.encoding import Board
from ai.tutor.exact_late import (
    PHYSICAL_BB_T4_ACTION_VECTOR_SCHEMA,
    action_key,
    default_rust_t3_exact_solver_path,
    evaluate_physical_bb_t4_action_vector_rust,
    physical_bb_t4_state_commitment,
)
from ai.tutor.t3_hu_particle_cfr import (
    particle_action_vector_from_physical_t4_result,
    solve_shared_infoset_particle_cfr_plus,
)
from ai.tutor.t3_hu_public_cfr import InfoSetKey, JointParticle, PrivateRecall
from ai.tutor.t3_hu_public_tree import PublicTreeDecisionState
from ai.tutor.t3_hu_public_tree_cfr import (
    public_tree_t4_decision_from_physical_result,
    solve_recursive_public_tree_cfr_plus,
)


BB_BOARD = (
    ("4h", "2c", "3d"),
    ("9s", "7c", "8h", "7d", "Tc"),
    ("Qc", "Qd", "Kh"),
)
BTN_BOARD = (
    ("8c", "5c", "6d"),
    ("Kc", "9c", "Jh", "9d", "Qs"),
    ("As", "2h", "3h"),
)
PUBLIC_HISTORY = (
    (0, "bb", (("4h", "top"), ("2c", "top"), ("3d", "top"), ("7c", "middle"), ("Qc", "bottom"))),
    (0, "btn", (("8c", "top"), ("5c", "top"), ("6d", "top"), ("9c", "middle"), ("As", "bottom"))),
    (1, "bb", (("7d", "middle"), ("8h", "middle"))),
    (1, "btn", (("9d", "middle"), ("Jh", "middle"))),
    (2, "bb", (("9s", "middle"), ("Tc", "middle"))),
    (2, "btn", (("Qs", "middle"), ("Kc", "middle"))),
    (3, "bb", (("Qd", "bottom"), ("Kh", "bottom"))),
    (3, "btn", (("2h", "bottom"), ("3h", "bottom"))),
)
CURRENT_DRAW = ("Ad", "Kd", "Jd")


def _recall(*, actor: str, t3_discard: str) -> PrivateRecall:
    placements = {
        "bb": {
            1: ("7d", "8h"),
            2: ("9s", "Tc"),
            3: ("Qd", "Kh"),
        },
        "btn": {
            1: ("9d", "Jh"),
            2: ("Qs", "Kc"),
            3: ("2h", "3h"),
        },
    }[actor]
    prior_discards = ("6c", "6s") if actor == "bb" else ("4s", "7h")
    discards = (*prior_discards, t3_discard)
    return PrivateRecall(
        dealt_by_turn=tuple(
            (turn, (*placements[turn], discards[turn - 1]))
            for turn in (1, 2, 3)
        ),
        discards_by_turn=tuple((turn, discards[turn - 1]) for turn in (1, 2, 3)),
    )


def _particle(*, opponent_t3_discard: str, remaining: tuple[str, ...], weight: Fraction):
    return JointParticle(
        bb_recall=_recall(actor="bb", t3_discard="2s"),
        btn_recall=_recall(actor="btn", t3_discard=opponent_t3_discard),
        undealt_cards=remaining,
        weight=weight,
    )


def _key(particle: JointParticle) -> InfoSetKey:
    return InfoSetKey.for_particle(
        particle,
        contract_version="bb_first_v1",
        actor="bb",
        turn=4,
        phase="t4_first",
        board_bb=BB_BOARD,
        board_btn=BTN_BOARD,
        public_action_history=PUBLIC_HISTORY,
        current_draw=CURRENT_DRAW,
    )


def _physical_result(
    particle: JointParticle,
    key: InfoSetKey,
    *,
    particle_id: str,
    scores: dict[str, float],
) -> dict:
    commitment = physical_bb_t4_state_commitment(
        {
            "turn": 4,
            "actor": "bb",
            "is_btn": False,
            "first_actor": "bb",
            "position_contract_version": "bb_first_v1",
            "particle_id": particle_id,
            "board_self": {
                row: list(cards)
                for row, cards in zip(("top", "middle", "bottom"), key.board_bb)
            },
            "board_opponent": {
                row: list(cards)
                for row, cards in zip(("top", "middle", "bottom"), key.board_btn)
            },
            "dealt_cards": list(key.current_draw),
            "remaining_cards": list(particle.undealt_cards),
        }
    )
    action_ids = sorted(scores)
    return {
        "schema": PHYSICAL_BB_T4_ACTION_VECTOR_SCHEMA,
        "turn": 4,
        "actor": "bb",
        "is_btn": False,
        "position_contract_version": "bb_first_v1",
        "particle_id": particle_id,
        "source": "rust_exact_physical_t4_action_vector",
        "physical_state_commitment": commitment,
        "selection_performed": False,
        "particle_aggregation_performed": False,
        "requires_infoset_aggregation": True,
        "public_policy_safe": False,
        "legal_actions": len(action_ids),
        "evaluated_actions": len(action_ids),
        "candidate_count": len(action_ids),
        "action_keys": action_ids,
        "metrics_by_action_key": {
            action: {
                "source": "exact_hu_response",
                "opponent_response": True,
                "score": scores[action],
            }
            for action in action_ids
        },
    }


def _physical_payload(
    particle: JointParticle,
    key: InfoSetKey,
    *,
    particle_id: str,
) -> dict:
    return {
        "turn": 4,
        "actor": "bb",
        "is_btn": False,
        "first_actor": "bb",
        "position_contract_version": "bb_first_v1",
        "particle_id": particle_id,
        "board_self": {
            row: list(cards)
            for row, cards in zip(("top", "middle", "bottom"), key.board_bb)
        },
        "board_opponent": {
            row: list(cards)
            for row, cards in zip(("top", "middle", "bottom"), key.board_btn)
        },
        "dealt_cards": list(key.current_draw),
        "remaining_cards": list(particle.undealt_cards),
    }


def _action_ids() -> list[str]:
    board = Board(top=list(BB_BOARD[0]), middle=list(BB_BOARD[1]), bottom=list(BB_BOARD[2]))
    return sorted(action_key(action) for action in get_turn_actions(CURRENT_DRAW, board))


def test_conditioned_t4_results_bind_to_particles_before_shared_infoset_selection():
    particle_a = _particle(
        opponent_t3_discard="4d",
        remaining=("X1", "Ac", "9h"),
        weight=Fraction(1, 4),
    )
    particle_b = _particle(
        opponent_t3_discard="5d",
        remaining=("X2", "Ac", "Th"),
        weight=Fraction(3, 4),
    )
    key_a = _key(particle_a)
    key_b = _key(particle_b)
    assert key_a == key_b

    actions = _action_ids()
    assert len(actions) == 3
    scores_a = {action: -100.0 for action in actions}
    scores_b = {action: -100.0 for action in actions}
    scores_a[actions[0]], scores_a[actions[1]] = 10.0, 0.0
    scores_b[actions[0]], scores_b[actions[1]] = 0.0, 4.0
    vector_a = particle_action_vector_from_physical_t4_result(
        _physical_result(particle_a, key_a, particle_id="world-a", scores=scores_a),
        particle=particle_a,
        infoset_key=key_a,
    )
    vector_b = particle_action_vector_from_physical_t4_result(
        _physical_result(particle_b, key_b, particle_id="world-b", scores=scores_b),
        particle=particle_b,
        infoset_key=key_b,
    )

    result = solve_shared_infoset_particle_cfr_plus((vector_a, vector_b), iterations=2)
    node = result.nodes[key_a]
    assert len(result.nodes) == 1
    assert node.conditional_action_utility[actions[0]] == Fraction(5, 2)
    assert node.conditional_action_utility[actions[1]] == Fraction(3, 1)
    assert node.current_strategy[actions[1]] > node.current_strategy[actions[0]]
    assert node.current_strategy[actions[2]] == 0
    assert result.metadata["strategy_fusion"] is False


@pytest.mark.parametrize("tamper", ["commitment", "selection", "nonfinite"])
def test_physical_t4_adapter_rejects_misbinding_preselection_and_nonfinite_scores(tamper: str):
    particle = _particle(
        opponent_t3_discard="4d",
        remaining=("X1", "Ac", "9h"),
        weight=Fraction(1, 1),
    )
    key = _key(particle)
    actions = _action_ids()
    result = _physical_result(
        particle,
        key,
        particle_id="world-a",
        scores={action: float(index) for index, action in enumerate(actions)},
    )
    if tamper == "commitment":
        result["physical_state_commitment"] = "0" * 64
        message = "commitment does not match"
    elif tamper == "selection":
        result["best"] = {"action_key": actions[0]}
        message = "preselected action"
    else:
        result["metrics_by_action_key"][actions[0]]["score"] = float("inf")
        message = "must be finite"

    with pytest.raises((TypeError, ValueError), match=message):
        particle_action_vector_from_physical_t4_result(
            result,
            particle=particle,
            infoset_key=key,
        )


def test_conditioned_t4_vector_compiles_all_actions_into_recursive_tree_leaves():
    particle = _particle(
        opponent_t3_discard="4d",
        remaining=("X1", "Ac", "9h"),
        weight=Fraction(1, 1),
    )
    key = _key(particle)
    state = PublicTreeDecisionState(infoset_key=key, particle=particle)
    actions = _action_ids()
    result = _physical_result(
        particle,
        key,
        particle_id="world-a",
        scores={action: float(index) for index, action in enumerate(actions)},
    )

    node = public_tree_t4_decision_from_physical_result(result, state)
    assert node.action_ids == tuple(actions)
    assert all(
        child.utility_source == "rust_exact_physical_t4_action_vector"
        for _action, child in node.actions
    )
    solved = solve_recursive_public_tree_cfr_plus(node, iterations=2)
    assert solved.metadata["rust_leaf_integrated"] is True
    assert solved.metadata["terminal_utility_sources"] == [
        "rust_exact_physical_t4_action_vector"
    ]


def test_real_rust_conditioned_vector_reaches_recursive_tree_without_preselection():
    solver = default_rust_t3_exact_solver_path()
    if not Path(solver).exists():
        pytest.skip(f"Rust late exact solver is not built: {solver}")
    particle = _particle(
        opponent_t3_discard="4d",
        remaining=("X1", "Ac", "9h"),
        weight=Fraction(1, 1),
    )
    key = _key(particle)
    state = PublicTreeDecisionState(infoset_key=key, particle=particle)
    result = evaluate_physical_bb_t4_action_vector_rust(
        _physical_payload(particle, key, particle_id="rust-world-a"),
        timeout_s=10.0,
    )

    node = public_tree_t4_decision_from_physical_result(result, state)
    assert node.action_ids == tuple(result["action_keys"])
    assert "best" not in result and "chosen_action" not in result
    for action_id, terminal in node.actions:
        assert terminal.utility_bb == Fraction(
            str(result["metrics_by_action_key"][action_id]["score"])
        )
