from fractions import Fraction

import pytest

from ai.engine.action_space import Action, get_turn_actions
from ai.engine.encoding import Board
from ai.tutor.exact_late import action_key, terminal_metrics
from ai.tutor.t3_hu_public_cfr import JointParticle, PrivateRecall
from ai.tutor.t3_hu_public_tree import (
    PublicTreeDecisionState,
    SuppliedChanceDraw,
    apply_public_tree_action,
    apply_public_tree_terminal_action,
    resolve_supplied_chance,
    transition_public_tree_action,
)


BB_BOARD = (
    ("4h", "2c", "3d"),
    ("9s", "7c", "8h", "7d", "Tc"),
    ("Qc",),
)
BTN_BOARD = (
    ("8c", "5c", "6d"),
    ("Kc", "9c", "Jh", "9d", "Qs"),
    ("As",),
)
PUBLIC_HISTORY = (
    (0, "bb", (("4h", "top"), ("2c", "top"), ("3d", "top"), ("7c", "middle"), ("Qc", "bottom"))),
    (0, "btn", (("8c", "top"), ("5c", "top"), ("6d", "top"), ("9c", "middle"), ("As", "bottom"))),
    (1, "bb", (("7d", "middle"), ("8h", "middle"))),
    (1, "btn", (("9d", "middle"), ("Jh", "middle"))),
    (2, "bb", (("9s", "middle"), ("Tc", "middle"))),
    (2, "btn", (("Qs", "middle"), ("Kc", "middle"))),
)


def _recall(discard_t1: str, discard_t2: str, *, actor: str) -> PrivateRecall:
    placed = {
        "bb": {1: ("7d", "8h"), 2: ("9s", "Tc")},
        "btn": {1: ("9d", "Jh"), 2: ("Qs", "Kc")},
    }[actor]
    return PrivateRecall(
        dealt_by_turn=(
            (1, (*placed[1], discard_t1)),
            (2, (*placed[2], discard_t2)),
        ),
        discards_by_turn=((1, discard_t1), (2, discard_t2)),
    )


def _initial_state(*, include_jokers: bool = False) -> PublicTreeDecisionState:
    remaining = [
        "2h", "3h", "5h", "6h", "Th", "Qh",
        "Ah", "Kh", "Jd", "2d", "3c", "4d",
    ]
    if include_jokers:
        remaining.extend(("X1", "X2"))
    particle = JointParticle(
        bb_recall=_recall("6c", "6s", actor="bb"),
        btn_recall=_recall("4s", "7h", actor="btn"),
        undealt_cards=tuple(remaining),
        weight=Fraction(2, 5),
    )
    return PublicTreeDecisionState.from_particle(
        particle,
        phase="t3_first",
        board_bb=BB_BOARD,
        board_btn=BTN_BOARD,
        public_action_history=PUBLIC_HISTORY,
        current_draw=("Ad", "Kd", "Qd"),
    )


def _board(rows) -> Board:
    return Board(top=list(rows[0]), middle=list(rows[1]), bottom=list(rows[2]))


def _t4_second_state(
    *,
    bb_t4_draw=("Ah", "Kh", "Jd"),
    bb_t4_discard="Jd",
) -> PublicTreeDecisionState:
    initial = _initial_state()
    btn_t3 = transition_public_tree_action(
        initial,
        Action(
            placements=[("Ad", "bottom"), ("Kd", "bottom")],
            discard="Qd",
        ),
        (SuppliedChanceDraw(("2h", "3h", "5h"), 1),),
    )[0].state
    bb_t4 = transition_public_tree_action(
        btn_t3,
        Action(
            placements=[("2h", "bottom"), ("3h", "bottom")],
            discard="5h",
        ),
        (SuppliedChanceDraw(bb_t4_draw, 1),),
    )[0].state
    pending_btn_t4 = apply_public_tree_action(
        bb_t4,
        Action(
            placements=[("Ah", "bottom"), ("Kh", "bottom")],
            discard=bb_t4_discard,
        ),
    )
    return resolve_supplied_chance(
        pending_btn_t4,
        (SuppliedChanceDraw(("2d", "3c", "4d"), 1),),
    )[0].state


def test_natural_t3_first_to_t3_second_to_t4_first_is_physically_exact():
    initial = _initial_state()
    bb_action = Action(
        placements=[("Kd", "bottom"), ("Ad", "bottom")],
        discard="Qd",
    )
    after_bb = apply_public_tree_action(initial, bb_action)

    assert after_bb.completed_phase == "t3_first"
    assert after_bb.next_phase == "t3_second"
    assert after_bb.board_bb[2] == ("Ad", "Kd", "Qc")
    assert after_bb.board_btn == initial.infoset_key.board_btn
    assert after_bb.public_action_history[-1] == (
        3,
        "bb",
        (("Ad", "bottom"), ("Kd", "bottom")),
    )
    assert after_bb.particle.bb_recall.dealt_by_turn[-1] == (
        3,
        ("Ad", "Kd", "Qd"),
    )
    assert after_bb.particle.bb_recall.discards_by_turn[-1] == (3, "Qd")
    assert after_bb.remaining_cards == initial.remaining_cards
    assert after_bb.applied_action_key == action_key(bb_action)

    # Reverse both outcome and card input order.  Resolution is canonical and
    # deterministic, while exact conditional probabilities remain attached.
    btn_outcomes = (
        SuppliedChanceDraw(("Qh", "Th", "6h"), Fraction(2, 3)),
        SuppliedChanceDraw(("5h", "3h", "2h"), Fraction(1, 3)),
    )
    btn_branches = resolve_supplied_chance(after_bb, btn_outcomes)
    assert [branch.state.infoset_key.current_draw for branch in btn_branches] == [
        ("2h", "3h", "5h"),
        ("6h", "Qh", "Th"),
    ]
    first_btn = btn_branches[0]
    assert first_btn.probability == Fraction(1, 3)
    assert first_btn.state.infoset_key.phase == "t3_second"
    assert first_btn.state.infoset_key.actor == "btn"
    assert set(first_btn.state.remaining_cards) == set(initial.remaining_cards) - {
        "2h", "3h", "5h"
    }
    assert first_btn.state.particle.weight == Fraction(2, 15)

    btn_action = Action(
        placements=[("3h", "bottom"), ("2h", "bottom")],
        discard="5h",
    )
    after_btn = apply_public_tree_action(first_btn.state, btn_action)
    assert after_btn.next_phase == "t4_first"
    assert after_btn.board_btn[2] == ("2h", "3h", "As")
    assert after_btn.public_action_history[-1] == (
        3,
        "btn",
        (("2h", "bottom"), ("3h", "bottom")),
    )
    assert after_btn.particle.btn_recall.dealt_by_turn[-1] == (
        3,
        ("2h", "3h", "5h"),
    )
    assert after_btn.particle.btn_recall.discards_by_turn[-1] == (3, "5h")

    t4_branches = transition_public_tree_action(
        first_btn.state,
        btn_action,
        (
            SuppliedChanceDraw(("Kh", "Ah", "Jd"), Fraction(1, 2)),
            SuppliedChanceDraw(("4d", "3c", "2d"), Fraction(1, 2)),
        ),
    )
    assert [branch.state.infoset_key.current_draw for branch in t4_branches] == [
        ("2d", "3c", "4d"),
        ("Ah", "Jd", "Kh"),
    ]
    t4 = t4_branches[1].state
    assert t4.infoset_key.phase == "t4_first"
    assert t4.infoset_key.actor == "bb"
    assert t4.infoset_key.turn == 4
    assert t4.particle.weight == Fraction(1, 15)
    assert set(t4.remaining_cards) == (
        set(initial.remaining_cards)
        - {"2h", "3h", "5h"}
        - {"Ah", "Jd", "Kh"}
    )
    assert t4.infoset_key.own_recall.discards_by_turn[-1] == (3, "Qd")


def test_natural_t4_first_to_t4_second_to_terminal_is_physically_exact():
    initial = _initial_state()
    btn_t3 = transition_public_tree_action(
        initial,
        Action(
            placements=[("Ad", "bottom"), ("Kd", "bottom")],
            discard="Qd",
        ),
        (SuppliedChanceDraw(("2h", "3h", "5h"), 1),),
    )[0].state
    bb_t4 = transition_public_tree_action(
        btn_t3,
        Action(
            placements=[("2h", "bottom"), ("3h", "bottom")],
            discard="5h",
        ),
        (SuppliedChanceDraw(("Ah", "Kh", "Jd"), 1),),
    )[0].state

    after_bb_t4 = apply_public_tree_action(
        bb_t4,
        Action(
            placements=[("Kh", "bottom"), ("Ah", "bottom")],
            discard="Jd",
        ),
    )
    assert after_bb_t4.completed_phase == "t4_first"
    assert after_bb_t4.next_phase == "t4_second"
    assert sum(map(len, after_bb_t4.board_bb)) == 13
    assert sum(map(len, after_bb_t4.board_btn)) == 11
    assert after_bb_t4.board_bb[2] == ("Ad", "Ah", "Kd", "Kh", "Qc")
    assert after_bb_t4.public_action_history[-1] == (
        4,
        "bb",
        (("Ah", "bottom"), ("Kh", "bottom")),
    )
    assert after_bb_t4.particle.bb_recall.dealt_by_turn[-1] == (
        4,
        ("Ah", "Jd", "Kh"),
    )
    assert after_bb_t4.particle.bb_recall.discards_by_turn[-1] == (4, "Jd")

    btn_t4_branches = resolve_supplied_chance(
        after_bb_t4,
        (
            SuppliedChanceDraw(("6h", "Qh", "Th"), Fraction(1, 2)),
            SuppliedChanceDraw(("4d", "3c", "2d"), Fraction(1, 2)),
        ),
    )
    assert [branch.state.infoset_key.current_draw for branch in btn_t4_branches] == [
        ("2d", "3c", "4d"),
        ("6h", "Qh", "Th"),
    ]
    btn_t4 = btn_t4_branches[0].state
    assert btn_t4.infoset_key.phase == "t4_second"
    assert btn_t4.infoset_key.actor == "btn"
    assert btn_t4.infoset_key.turn == 4
    assert btn_t4.particle.weight == Fraction(1, 5)
    assert btn_t4.infoset_key.own_recall.discards_by_turn[-1] == (3, "5h")
    assert btn_t4.particle.bb_recall.discards_by_turn[-1] == (4, "Jd")
    assert set(btn_t4.remaining_cards) == {"6h", "Qh", "Th"}

    final_action = Action(
        placements=[("3c", "bottom"), ("2d", "bottom")],
        discard="4d",
    )
    terminal = apply_public_tree_terminal_action(btn_t4, final_action)
    assert terminal.completed_phase == "t4_second"
    assert tuple(map(len, terminal.board_bb)) == (3, 5, 5)
    assert tuple(map(len, terminal.board_btn)) == (3, 5, 5)
    assert terminal.board_btn[2] == ("2d", "2h", "3c", "3h", "As")
    assert terminal.public_action_history[-1] == (
        4,
        "btn",
        (("2d", "bottom"), ("3c", "bottom")),
    )
    assert terminal.particle.btn_recall.dealt_by_turn[-1] == (
        4,
        ("2d", "3c", "4d"),
    )
    assert terminal.particle.btn_recall.discards_by_turn[-1] == (4, "4d")
    assert terminal.remaining_cards == btn_t4.remaining_cards
    assert terminal.particle.weight == Fraction(1, 5)
    assert terminal.applied_action_key == action_key(final_action)


def test_x1_and_x2_remain_distinct_across_consecutive_chance_transitions():
    initial = _initial_state(include_jokers=True)
    after_bb = apply_public_tree_action(
        initial,
        Action(
            placements=[("Ad", "bottom"), ("Kd", "bottom")],
            discard="Qd",
        ),
    )
    branches = resolve_supplied_chance(
        after_bb,
        (
            SuppliedChanceDraw(("3h", "2h", "X2"), Fraction(1, 2)),
            SuppliedChanceDraw(("3h", "2h", "X1"), Fraction(1, 2)),
        ),
    )
    assert [branch.state.infoset_key.current_draw for branch in branches] == [
        ("2h", "3h", "X1"),
        ("2h", "3h", "X2"),
    ]

    x1_branch = branches[0]
    assert "X1" not in x1_branch.state.remaining_cards
    assert "X2" in x1_branch.state.remaining_cards
    t4_branch = transition_public_tree_action(
        x1_branch.state,
        Action(
            placements=[("2h", "bottom"), ("X1", "bottom")],
            discard="3h",
        ),
        (SuppliedChanceDraw(("Ah", "Kh", "X2"), 1),),
    )[0]
    assert "X1" in t4_branch.state.infoset_key.board_btn[2]
    assert "X2" in t4_branch.state.infoset_key.current_draw
    assert "X1" != "X2"
    assert "X1" not in t4_branch.state.remaining_cards
    assert "X2" not in t4_branch.state.remaining_cards

    btn_t4 = transition_public_tree_action(
        t4_branch.state,
        Action(
            placements=[("Ah", "bottom"), ("X2", "bottom")],
            discard="Kh",
        ),
        (SuppliedChanceDraw(("2d", "3c", "4d"), 1),),
    )[0].state
    assert "X1" in btn_t4.infoset_key.board_btn[2]
    assert "X2" in btn_t4.infoset_key.board_bb[2]
    assert "X1" not in btn_t4.infoset_key.current_draw
    assert "X2" not in btn_t4.infoset_key.current_draw
    assert "X1" not in btn_t4.remaining_cards
    assert "X2" not in btn_t4.remaining_cards

    terminal = apply_public_tree_terminal_action(
        btn_t4,
        Action(
            placements=[("2d", "bottom"), ("3c", "bottom")],
            discard="4d",
        ),
    )
    assert "X1" in terminal.board_btn[2]
    assert "X2" in terminal.board_bb[2]
    assert "X1" != "X2"


def test_t4_second_terminal_values_do_not_depend_on_hidden_bb_t4_discard():
    hidden_jd = _t4_second_state(
        bb_t4_draw=("Ah", "Kh", "Jd"),
        bb_t4_discard="Jd",
    )
    hidden_qh = _t4_second_state(
        bb_t4_draw=("Ah", "Kh", "Qh"),
        bb_t4_discard="Qh",
    )

    # BTN observes the same public boards/history and the same private draw.
    # The physical particles differ only in BB's hidden T4 discard/remainder,
    # neither of which may enter BTN's information-set identity.
    assert hidden_jd.infoset_key == hidden_qh.infoset_key
    assert hidden_jd.infoset_key.digest() == hidden_qh.infoset_key.digest()
    assert hidden_jd.infoset_key.canonical_json() == hidden_qh.infoset_key.canonical_json()
    assert hidden_jd.particle.bb_recall.discards_by_turn[-1] == (4, "Jd")
    assert hidden_qh.particle.bb_recall.discards_by_turn[-1] == (4, "Qh")
    assert hidden_jd.remaining_cards != hidden_qh.remaining_cards

    def terminal_values(state: PublicTreeDecisionState) -> dict[str, float]:
        key = state.infoset_key
        actions = get_turn_actions(list(key.current_draw), _board(key.board_btn))
        values: dict[str, float] = {}
        for candidate in actions:
            terminal = apply_public_tree_terminal_action(state, candidate)
            values[action_key(candidate)] = terminal_metrics(
                _board(terminal.board_btn),
                _board(terminal.board_bb),
            )["score"]
        return values

    values_jd = terminal_values(hidden_jd)
    values_qh = terminal_values(hidden_qh)
    assert values_jd == values_qh
    assert max(values_jd, key=lambda key: (values_jd[key], key)) == max(
        values_qh,
        key=lambda key: (values_qh[key], key),
    )


def test_chance_contract_rejects_non_unit_mass_duplicates_and_unavailable_cards():
    initial = _initial_state()
    pending = apply_public_tree_action(
        initial,
        Action(
            placements=[("Ad", "bottom"), ("Kd", "bottom")],
            discard="Qd",
        ),
    )
    with pytest.raises(ValueError, match="sum exactly to 1"):
        resolve_supplied_chance(
            pending,
            (SuppliedChanceDraw(("2h", "3h", "5h"), Fraction(1, 2)),),
        )
    with pytest.raises(ValueError, match="must be unique"):
        resolve_supplied_chance(
            pending,
            (
                SuppliedChanceDraw(("2h", "3h", "5h"), Fraction(1, 2)),
                SuppliedChanceDraw(("5h", "2h", "3h"), Fraction(1, 2)),
            ),
        )
    with pytest.raises(ValueError, match="outside the physical remainder"):
        resolve_supplied_chance(
            pending,
            (SuppliedChanceDraw(("2h", "3h", "Ad"), 1),),
        )
    with pytest.raises(ValueError, match="duplicate physical cards"):
        SuppliedChanceDraw(("2h", "2h", "3h"), 1)


def test_action_must_use_exact_draw_and_respect_row_capacity():
    initial = _initial_state()
    with pytest.raises(ValueError, match="exact current draw"):
        apply_public_tree_action(
            initial,
            Action(
                placements=[("Ad", "bottom"), ("Kd", "bottom")],
                discard="2h",
            ),
        )
    with pytest.raises(ValueError, match="not legal for the current board capacity"):
        apply_public_tree_action(
            initial,
            Action(
                placements=[("Ad", "top"), ("Kd", "bottom")],
                discard="Qd",
            ),
        )
