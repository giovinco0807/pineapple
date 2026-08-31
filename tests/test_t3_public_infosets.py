import json
from fractions import Fraction

import pytest

from ai.tutor.t3_hu_public_cfr import (
    FORBIDDEN_INFOSET_FIELDS,
    InfoSetKey,
    JointParticle,
    PrivateRecall,
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
BB_T3_PLACEMENTS = (("Qd", "bottom"), ("Kh", "bottom"))
BB_BOARD_11 = (BB_BOARD[0], BB_BOARD[1], ("Qc", "Qd", "Kh"))
PUBLIC_HISTORY_AFTER_BB_T3 = PUBLIC_HISTORY + ((3, "bb", BB_T3_PLACEMENTS),)
BTN_T3_PLACEMENTS = (("2h", "bottom"), ("3h", "bottom"))
BTN_BOARD_11 = (BTN_BOARD[0], BTN_BOARD[1], ("As", "2h", "3h"))
PUBLIC_HISTORY_AFTER_T3 = PUBLIC_HISTORY_AFTER_BB_T3 + (
    (3, "btn", BTN_T3_PLACEMENTS),
)
BB_T4_PLACEMENTS = (("Ad", "bottom"), ("Kd", "bottom"))
BB_BOARD_13 = (BB_BOARD[0], BB_BOARD[1], ("Qc", "Qd", "Kh", "Ad", "Kd"))
PUBLIC_HISTORY_AFTER_BB_T4 = PUBLIC_HISTORY_AFTER_T3 + (
    (4, "bb", BB_T4_PLACEMENTS),
)


def recall(
    t1_discard: str,
    t2_discard: str,
    *,
    actor: str = "bb",
    t3_discard: str | None = None,
    t4_discard: str | None = None,
) -> PrivateRecall:
    placed_by_turn = {
        "bb": {
            1: ("7d", "8h"),
            2: ("9s", "Tc"),
            3: ("Qd", "Kh"),
            4: ("Ad", "Kd"),
        },
        "btn": {
            1: ("9d", "Jh"),
            2: ("Qs", "Kc"),
            3: ("2h", "3h"),
        },
    }
    if actor not in placed_by_turn:
        raise ValueError(f"unsupported recall actor: {actor!r}")
    if t4_discard is not None and (actor != "bb" or t3_discard is None):
        raise ValueError("T4 fixture recall requires BB recall through T3")
    placed = placed_by_turn[actor]
    dealt_by_turn = [
        (1, (*placed[1], t1_discard)),
        (2, (*placed[2], t2_discard)),
    ]
    discards_by_turn = [(1, t1_discard), (2, t2_discard)]
    if t3_discard is not None:
        dealt_by_turn.append((3, (*placed[3], t3_discard)))
        discards_by_turn.append((3, t3_discard))
    if t4_discard is not None:
        dealt_by_turn.append((4, (*placed[4], t4_discard)))
        discards_by_turn.append((4, t4_discard))
    return PrivateRecall(
        dealt_by_turn=tuple(dealt_by_turn),
        discards_by_turn=tuple(discards_by_turn),
    )


def make_key(
    particle: JointParticle,
    *,
    actor="bb",
    current_draw=None,
    history=PUBLIC_HISTORY,
):
    phase = "t3_first" if actor == "bb" else "t3_second"
    if current_draw is None:
        current_draw = ("Ad", "Kd", "Qd") if actor == "bb" else ("Ad", "Kd", "Jd")
    return InfoSetKey.for_particle(
        particle,
        contract_version="bb_first_v1",
        actor=actor,
        turn=3,
        phase=phase,
        board_bb=BB_BOARD if actor == "bb" else BB_BOARD_11,
        board_btn=BTN_BOARD,
        public_action_history=history if actor == "bb" else PUBLIC_HISTORY_AFTER_BB_T3,
        current_draw=current_draw,
    )


def test_infoset_ignores_opponent_private_cards_physical_world_and_weight():
    own = recall("6c", "6s")
    world_a = JointParticle(
        bb_recall=own,
        btn_recall=recall("4s", "7h", actor="btn"),
        undealt_cards=("X1", "Ac", "2d"),
        weight=Fraction(1, 4),
    )
    world_b = JointParticle(
        bb_recall=own,
        btn_recall=recall("4s", "X2", actor="btn"),
        undealt_cards=("Th", "Jc", "Td"),
        weight=Fraction(3, 4),
    )

    key_a = make_key(world_a)
    key_b = make_key(world_b)

    assert key_a == key_b
    assert key_a.digest() == key_b.digest()
    serialized = key_a.canonical_json()
    payload = json.loads(serialized)
    assert set(payload) == {
        "contract_version",
        "actor",
        "turn",
        "phase",
        "board_bb",
        "board_btn",
        "public_action_history",
        "own_recall",
        "current_draw",
        "fantasy_state",
    }
    lowered = serialized.lower()
    assert all(f'"{field}"' not in lowered for field in FORBIDDEN_INFOSET_FIELDS)


def test_btn_infoset_uses_btn_recall_and_ignores_bb_private_recall():
    own_btn = recall("4s", "7h", actor="btn")
    world_a = JointParticle(
        bb_recall=recall("6c", "6s", t3_discard="2h"),
        btn_recall=own_btn,
        undealt_cards=("X1", "Ac", "2d"),
    )
    world_b = JointParticle(
        bb_recall=recall("2s", "3s", t3_discard="4d"),
        btn_recall=own_btn,
        undealt_cards=("X2", "Jc", "Td"),
    )

    key_a = make_key(world_a, actor="btn")
    key_b = make_key(world_b, actor="btn")

    assert key_a == key_b
    assert key_a.own_recall == own_btn


def test_infoset_separates_own_recall_draw_and_public_history():
    base = JointParticle(
        bb_recall=recall("6c", "6s"),
        btn_recall=recall("4s", "7h", actor="btn"),
        undealt_cards=("X1", "Ac", "2d"),
    )
    changed_own = JointParticle(
        bb_recall=recall("6c", "5s"),
        btn_recall=base.btn_recall,
        undealt_cards=base.undealt_cards,
    )
    base_key = make_key(base)

    assert make_key(changed_own) != base_key
    assert make_key(base, current_draw=("Ad", "Kd", "Jd")) != base_key
    changed_history = list(PUBLIC_HISTORY)
    changed_history[2] = (
        1,
        "bb",
        (("7d", "bottom"), ("8h", "middle")),
    )
    changed_key = InfoSetKey.for_particle(
        base,
        contract_version="bb_first_v1",
        actor="bb",
        turn=3,
        phase="t3_first",
        board_bb=(BB_BOARD[0], ("9s", "7c", "8h", "Tc"), ("Qc", "7d")),
        board_btn=BTN_BOARD,
        public_action_history=tuple(changed_history),
        current_draw=("Ad", "Kd", "Qd"),
    )
    assert changed_key != base_key


def test_infoset_canonicalization_is_order_invariant_but_x1_x2_are_distinct():
    particle = JointParticle(
        bb_recall=recall("6c", "6s"),
        btn_recall=recall("4s", "7h", actor="btn"),
        undealt_cards=("Ac", "2d", "3c"),
    )
    reordered_history = tuple(
        (turn, actor, tuple(reversed(placements)))
        for turn, actor, placements in PUBLIC_HISTORY
    )
    canonical = make_key(particle, current_draw=("Ad", "X1", "Qd"))
    reordered = InfoSetKey.for_particle(
        particle,
        contract_version="bb_first_v1",
        actor="bb",
        turn=3,
        phase="t3_first",
        board_bb=tuple(tuple(reversed(row)) for row in BB_BOARD),
        board_btn=tuple(tuple(reversed(row)) for row in BTN_BOARD),
        public_action_history=reordered_history,
        current_draw=("Qd", "Ad", "X1"),
    )
    other_joker = make_key(particle, current_draw=("Ad", "X2", "Qd"))

    assert reordered == canonical
    assert reordered.digest() == canonical.digest()
    assert other_joker != canonical
    assert other_joker.digest() != canonical.digest()


def test_infoset_rejects_a_legacy_position_contract():
    particle = JointParticle(
        bb_recall=recall("6c", "6s"),
        btn_recall=recall("4s", "7h", actor="btn"),
        undealt_cards=("X1", "Ac", "2d"),
    )
    with pytest.raises(ValueError, match="unsupported position contract"):
        InfoSetKey.for_particle(
            particle,
            contract_version="btn_first_legacy",
            actor="bb",
            turn=3,
            phase="t3_first",
            board_bb=BB_BOARD,
            board_btn=BTN_BOARD,
            public_action_history=PUBLIC_HISTORY,
            current_draw=("Ad", "Kd", "Qd"),
        )


def test_infoset_rejects_future_board_history_and_private_recall():
    particle = JointParticle(
        bb_recall=recall("6c", "6s"),
        btn_recall=recall("4s", "7h", actor="btn"),
        undealt_cards=("X1", "Ac", "2d"),
    )
    common = {
        "particle": particle,
        "contract_version": "bb_first_v1",
        "actor": "bb",
        "turn": 3,
        "phase": "t3_first",
        "board_btn": BTN_BOARD,
        "current_draw": ("Ad", "Kd", "Qd"),
    }
    with pytest.raises(ValueError, match="board counts 9/9"):
        InfoSetKey.for_particle(
            board_bb=BB_BOARD_11,
            public_action_history=PUBLIC_HISTORY,
            **common,
        )
    with pytest.raises(ValueError, match="history through T2 btn"):
        InfoSetKey.for_particle(
            board_bb=BB_BOARD,
            public_action_history=PUBLIC_HISTORY_AFTER_BB_T3,
            **common,
        )

    future_recall = PrivateRecall(
        dealt_by_turn=(
            (1, ("2s", "Jd", "6c")),
            (2, ("Ts", "Qh", "6s")),
            (3, ("3s", "4d", "5d")),
        ),
        discards_by_turn=((1, "6c"), (2, "6s"), (3, "5d")),
    )
    future_particle = JointParticle(
        bb_recall=future_recall,
        btn_recall=particle.btn_recall,
        undealt_cards=particle.undealt_cards,
    )
    with pytest.raises(ValueError, match="private recall must contain"):
        InfoSetKey.for_particle(
            future_particle,
            contract_version="bb_first_v1",
            actor="bb",
            turn=3,
            phase="t3_first",
            board_bb=BB_BOARD,
            board_btn=BTN_BOARD,
            public_action_history=PUBLIC_HISTORY,
            current_draw=("Ad", "Kd", "Qd"),
        )


def test_infoset_rejects_recall_that_does_not_match_own_public_placements():
    mismatched = PrivateRecall(
        dealt_by_turn=(
            (1, ("7d", "2s", "6c")),
            (2, ("9s", "Tc", "6s")),
        ),
        discards_by_turn=((1, "6c"), (2, "6s")),
    )
    particle = JointParticle(
        bb_recall=mismatched,
        btn_recall=recall("4s", "7h", actor="btn"),
        undealt_cards=("X1", "Ac", "2d"),
    )

    with pytest.raises(ValueError, match="2 public placements plus own discard"):
        make_key(particle)


def test_infoset_rejects_current_draw_overlapping_public_or_remembered_cards():
    particle = JointParticle(
        bb_recall=recall("6c", "6s"),
        btn_recall=recall("4s", "7h", actor="btn"),
        undealt_cards=("X1", "Ac", "2d"),
    )

    with pytest.raises(ValueError, match="current private draw overlaps"):
        make_key(particle, current_draw=("4h", "Kd", "Qd"))


def test_infoset_rejects_own_discard_overlapping_opponent_public_card():
    particle = JointParticle(
        bb_recall=recall("8c", "6s"),
        btn_recall=recall("4s", "7h", actor="btn"),
        undealt_cards=("X1", "Ac", "2d"),
    )

    with pytest.raises(ValueError, match="discard overlaps a public card"):
        make_key(particle)


@pytest.mark.parametrize("overlap_zone", ["opponent_recall", "undealt"])
def test_joint_particle_rejects_cross_zone_physical_card_overlap(overlap_zone: str):
    bb_recall = recall("6c", "6s")
    btn_recall = (
        recall("6c", "7h", actor="btn")
        if overlap_zone == "opponent_recall"
        else recall("4s", "7h", actor="btn")
    )
    undealt = ("6c", "Ac", "Kd") if overlap_zone == "undealt" else ("X1", "Ac", "Kd")

    with pytest.raises(ValueError, match="particle physical card '6c'"):
        JointParticle(
            bb_recall=bb_recall,
            btn_recall=btn_recall,
            undealt_cards=undealt,
        )


def test_infoset_rejects_invalid_physical_card_name():
    particle = JointParticle(
        bb_recall=recall("6c", "6s"),
        btn_recall=recall("4s", "7h", actor="btn"),
        undealt_cards=("X1", "Ac", "2d"),
    )

    with pytest.raises(ValueError, match="invalid physical cards"):
        make_key(particle, current_draw=("Ad", "Qd", "ZZ"))


def test_private_recall_rejects_discard_without_matching_draw():
    with pytest.raises(ValueError, match="invalid physical card"):
        PrivateRecall(discards_by_turn=((1, "ZZ"),))

    with pytest.raises(ValueError, match="no matching remembered private draw"):
        PrivateRecall(discards_by_turn=((1, "2h"),))


def test_particle_observation_rejects_opponent_discard_on_public_card():
    particle = JointParticle(
        bb_recall=recall("6c", "6s"),
        btn_recall=recall("4h", "7h", actor="btn"),
        undealt_cards=("X1", "Ac", "2d"),
    )

    with pytest.raises(ValueError, match="physical btn discard overlaps public cards"):
        make_key(particle)


def test_particle_observation_rejects_opponent_recall_not_matching_history():
    bad_btn_recall = PrivateRecall(
        dealt_by_turn=(
            (1, ("2s", "9d", "4s")),
            (2, ("Qs", "Kc", "7h")),
        ),
        discards_by_turn=((1, "4s"), (2, "7h")),
    )
    particle = JointParticle(
        bb_recall=recall("6c", "6s"),
        btn_recall=bad_btn_recall,
        undealt_cards=("X1", "Ac", "2d"),
    )

    with pytest.raises(ValueError, match="physical btn recall T1 must equal"):
        make_key(particle)


def test_particle_observation_rejects_current_draw_on_opponent_hidden_card():
    particle = JointParticle(
        bb_recall=recall("6c", "6s"),
        btn_recall=recall("4s", "7h", actor="btn"),
        undealt_cards=("X1", "Ac", "2d"),
    )

    with pytest.raises(ValueError, match="current private draw overlaps a physical particle"):
        make_key(particle, current_draw=("4s", "Kd", "Qd"))


def test_particle_observation_rejects_current_draw_still_marked_undealt():
    particle = JointParticle(
        bb_recall=recall("6c", "6s"),
        btn_recall=recall("4s", "7h", actor="btn"),
        undealt_cards=("X1", "Ac", "Kd"),
    )

    with pytest.raises(ValueError, match="undealt cards overlap public or current"):
        make_key(particle)


def test_btn_particle_requires_bb_t3_recall_after_public_bb_action():
    particle = JointParticle(
        bb_recall=recall("6c", "6s"),
        btn_recall=recall("4s", "7h", actor="btn"),
        undealt_cards=("X1", "Ac", "2d"),
    )

    with pytest.raises(ValueError, match=r"physical bb recall must contain.*\(1, 2, 3\)"):
        make_key(particle, actor="btn")


def test_t4_first_particle_requires_both_players_t3_recall():
    valid = JointParticle(
        bb_recall=recall("6c", "6s", t3_discard="2s"),
        btn_recall=recall("4s", "7h", actor="btn", t3_discard="4d"),
        undealt_cards=("X1", "Ac", "5d"),
    )
    key = InfoSetKey.for_particle(
        valid,
        contract_version="bb_first_v1",
        actor="bb",
        turn=4,
        phase="t4_first",
        board_bb=BB_BOARD_11,
        board_btn=BTN_BOARD_11,
        public_action_history=PUBLIC_HISTORY_AFTER_T3,
        current_draw=("Ad", "Kd", "Jd"),
    )
    assert key.phase == "t4_first"

    missing_btn_t3 = JointParticle(
        bb_recall=valid.bb_recall,
        btn_recall=recall("4s", "7h", actor="btn"),
        undealt_cards=valid.undealt_cards,
    )
    with pytest.raises(ValueError, match=r"physical btn recall must contain.*\(1, 2, 3\)"):
        InfoSetKey.for_particle(
            missing_btn_t3,
            contract_version="bb_first_v1",
            actor="bb",
            turn=4,
            phase="t4_first",
            board_bb=BB_BOARD_11,
            board_btn=BTN_BOARD_11,
            public_action_history=PUBLIC_HISTORY_AFTER_T3,
            current_draw=("Ad", "Kd", "Jd"),
        )


def test_t4_second_particle_requires_bb_t4_recall():
    valid = JointParticle(
        bb_recall=recall(
            "6c",
            "6s",
            t3_discard="2s",
            t4_discard="Jd",
        ),
        btn_recall=recall("4s", "7h", actor="btn", t3_discard="4d"),
        undealt_cards=("X2", "Ac", "9h"),
    )
    key = InfoSetKey.for_particle(
        valid,
        contract_version="bb_first_v1",
        actor="btn",
        turn=4,
        phase="t4_second",
        board_bb=BB_BOARD_13,
        board_btn=BTN_BOARD_11,
        public_action_history=PUBLIC_HISTORY_AFTER_BB_T4,
        current_draw=("X1", "5d", "8d"),
    )
    assert key.phase == "t4_second"

    missing_bb_t4 = JointParticle(
        bb_recall=recall("6c", "6s", t3_discard="2s"),
        btn_recall=valid.btn_recall,
        undealt_cards=valid.undealt_cards,
    )
    with pytest.raises(ValueError, match=r"physical bb recall must contain.*\(1, 2, 3, 4\)"):
        InfoSetKey.for_particle(
            missing_bb_t4,
            contract_version="bb_first_v1",
            actor="btn",
            turn=4,
            phase="t4_second",
            board_bb=BB_BOARD_13,
            board_btn=BTN_BOARD_11,
            public_action_history=PUBLIC_HISTORY_AFTER_BB_T4,
            current_draw=("X1", "5d", "8d"),
        )
