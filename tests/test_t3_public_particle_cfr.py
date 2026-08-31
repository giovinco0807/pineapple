from fractions import Fraction

from ai.tutor.t3_hu_particle_cfr import (
    ParticleActionUtilityVector,
    solve_shared_infoset_particle_cfr_plus,
)
from ai.tutor.t3_hu_public_cfr import InfoSetKey, JointParticle, PrivateRecall


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


def _recall(
    discard_t1: str,
    discard_t2: str,
    *,
    actor: str,
    discard_t3: str | None = None,
) -> PrivateRecall:
    public = {
        "bb": {1: ("7d", "8h"), 2: ("9s", "Tc")},
        "btn": {1: ("9d", "Jh"), 2: ("Qs", "Kc")},
    }[actor]
    if discard_t3 is not None and actor != "bb":
        raise ValueError("only BB has completed T3 in these fixtures")
    dealt_by_turn = [
        (1, (*public[1], discard_t1)),
        (2, (*public[2], discard_t2)),
    ]
    discards_by_turn = [(1, discard_t1), (2, discard_t2)]
    if discard_t3 is not None:
        dealt_by_turn.append((3, ("Qd", "Kh", discard_t3)))
        discards_by_turn.append((3, discard_t3))
    return PrivateRecall(
        dealt_by_turn=tuple(dealt_by_turn),
        discards_by_turn=tuple(discards_by_turn),
    )


def _particles(*, actor: str) -> tuple[JointParticle, JointParticle]:
    if actor == "bb":
        own = _recall("6c", "6s", actor="bb")
        return (
            JointParticle(
                bb_recall=own,
                btn_recall=_recall("4s", "7h", actor="btn"),
                undealt_cards=("X1", "Ac", "2d"),
                weight=Fraction(1, 4),
            ),
            JointParticle(
                bb_recall=own,
                btn_recall=_recall("5s", "X2", actor="btn"),
                undealt_cards=("Th", "Jc", "Td"),
                weight=Fraction(3, 4),
            ),
        )
    own = _recall("4s", "7h", actor="btn")
    return (
        JointParticle(
            bb_recall=_recall("6c", "6s", actor="bb", discard_t3="2h"),
            btn_recall=own,
            undealt_cards=("X1", "Ac", "2d"),
            weight=Fraction(1, 2),
        ),
        JointParticle(
            bb_recall=_recall("2s", "3s", actor="bb", discard_t3="4d"),
            btn_recall=own,
            undealt_cards=("X2", "Jc", "Td"),
            weight=Fraction(1, 2),
        ),
    )


def _key(particle: JointParticle, *, actor: str) -> InfoSetKey:
    return InfoSetKey.for_particle(
        particle,
        contract_version="bb_first_v1",
        actor=actor,
        turn=3,
        phase="t3_first" if actor == "bb" else "t3_second",
        board_bb=BB_BOARD if actor == "bb" else BB_BOARD_11,
        board_btn=BTN_BOARD,
        public_action_history=(
            PUBLIC_HISTORY if actor == "bb" else PUBLIC_HISTORY_AFTER_BB_T3
        ),
        current_draw=("Ad", "Kd", "Qd") if actor == "bb" else ("Ad", "Kd", "Jd"),
    )


def test_particles_with_the_same_observation_share_one_policy_node():
    particle_a, particle_b = _particles(actor="bb")
    key_a = _key(particle_a, actor="bb")
    key_b = _key(particle_b, actor="bb")
    assert key_a == key_b

    vectors = (
        ParticleActionUtilityVector(
            "hidden-world-a",
            particle_a,
            key_a,
            {"place-left": 10, "place-right": 0},
        ),
        ParticleActionUtilityVector(
            "hidden-world-b",
            particle_b,
            key_b,
            {"place-left": 0, "place-right": 4},
        ),
    )
    result = solve_shared_infoset_particle_cfr_plus(vectors, iterations=4)

    assert tuple(result.nodes) == (key_a,)
    assert result.metadata["physical_particle_count"] == 2
    assert result.metadata["policy_node_count"] == 1
    assert result.metadata["strategy_fusion"] is False
    assert result.metadata["chance_aggregation"] == (
        "weighted_action_vector_before_regret_update"
    )
    assert result.metadata["position_contract_version"] == "bb_first_v1"
    assert result.strategy_for_particle(vectors[0], average=False) == result.strategy_for_particle(
        vectors[1], average=False
    )
    assert result.strategy_for_particle(vectors[0], average=False) == {
        "place-left": Fraction(0, 1),
        "place-right": Fraction(1, 1),
    }


def test_chance_weighted_vectors_are_aggregated_before_the_regret_update():
    particle_a, particle_b = _particles(actor="bb")
    key = _key(particle_a, actor="bb")
    result = solve_shared_infoset_particle_cfr_plus(
        (
            ParticleActionUtilityVector(
                "world-quarter",
                particle_a,
                key,
                {"place-left": 10, "place-right": 0},
            ),
            ParticleActionUtilityVector(
                "world-three-quarters",
                particle_b,
                _key(particle_b, actor="bb"),
                {"place-left": 0, "place-right": 4},
            ),
        ),
        iterations=1,
    )
    node = result.nodes[key]

    # First aggregate the complete action vectors:
    # left = 1/4*10 + 3/4*0 = 5/2; right = 1/4*0 + 3/4*4 = 3.
    assert node.chance_mass == 1
    assert node.weighted_action_utility == {
        "place-left": Fraction(5, 2),
        "place-right": Fraction(3, 1),
    }
    # Initial shared strategy is uniform, so its weighted value is 11/4.
    assert node.first_regret_delta == {
        "place-left": Fraction(-1, 4),
        "place-right": Fraction(1, 4),
    }
    assert node.current_strategy == {
        "place-left": Fraction(0, 1),
        "place-right": Fraction(1, 1),
    }


def test_neither_maximizer_nor_minimizer_can_fuse_particle_specific_actions():
    # BB would illegally choose left in world A and right in world B for an
    # E[max] value of 11/2.  Its one legal shared action is right, value 3.
    bb_a, bb_b = _particles(actor="bb")
    bb_vectors = (
        ParticleActionUtilityVector(
            "bb-world-a", bb_a, _key(bb_a, actor="bb"), {"left": 10, "right": 0}
        ),
        ParticleActionUtilityVector(
            "bb-world-b", bb_b, _key(bb_b, actor="bb"), {"left": 0, "right": 4}
        ),
    )
    bb_result = solve_shared_infoset_particle_cfr_plus(bb_vectors, iterations=2)
    bb_key = bb_vectors[0].infoset_key
    assert bb_result.nodes[bb_key].conditional_action_utility["right"] == 3
    assert bb_result.strategy_for_particle(bb_vectors[0], average=False) == {
        "left": Fraction(0, 1),
        "right": Fraction(1, 1),
    }
    assert bb_result.strategy_for_particle(bb_vectors[0], average=False) == (
        bb_result.strategy_for_particle(bb_vectors[1], average=False)
    )
    assert Fraction(3, 1) != Fraction(11, 2)

    # BTN minimizes BB utility.  Per-world E[min] would be -7, but one shared
    # action can only achieve min(-5, -2) = -5.
    btn_a, btn_b = _particles(actor="btn")
    btn_vectors = (
        ParticleActionUtilityVector(
            "btn-world-a", btn_a, _key(btn_a, actor="btn"), {"left": -10, "right": 0}
        ),
        ParticleActionUtilityVector(
            "btn-world-b", btn_b, _key(btn_b, actor="btn"), {"left": 0, "right": -4}
        ),
    )
    btn_result = solve_shared_infoset_particle_cfr_plus(btn_vectors, iterations=2)
    btn_key = btn_vectors[0].infoset_key
    assert btn_result.nodes[btn_key].conditional_action_utility == {
        "left": Fraction(-5, 1),
        "right": Fraction(-2, 1),
    }
    assert btn_result.strategy_for_particle(btn_vectors[0], average=False) == {
        "left": Fraction(1, 1),
        "right": Fraction(0, 1),
    }
    assert btn_result.strategy_for_particle(btn_vectors[0], average=False) == (
        btn_result.strategy_for_particle(btn_vectors[1], average=False)
    )
    assert Fraction(-5, 1) != Fraction(-7, 1)
