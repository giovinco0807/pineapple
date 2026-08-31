from fractions import Fraction

import pytest

from ai.engine.action_space import Action
from ai.tutor.t3_hu_public_cfr import JointParticle, PrivateRecall
from ai.tutor.t3_hu_public_tree import (
    PublicTreeDecisionState,
    SuppliedChanceDraw,
    apply_public_tree_action,
    resolve_supplied_chance,
)
from ai.tutor.t3_hu_public_tree_cfr import (
    PublicTreeChanceBranch,
    PublicTreeChanceNode,
    PublicTreeDecisionNode,
    PublicTreeTerminalNode,
    exact_public_tree_best_response,
    public_tree_strategy_fusion_diagnostic,
    solve_recursive_public_tree_cfr_plus,
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


def _sender_state(
    type_discard: str,
    *,
    particle_weight: Fraction = Fraction(1, 1),
) -> PublicTreeDecisionState:
    particle = JointParticle(
        bb_recall=_recall("6c", type_discard, actor="bb"),
        btn_recall=_recall("4s", "7h", actor="btn"),
        undealt_cards=("2h", "3h", "5h", "Ah", "Jd", "Kh"),
        weight=particle_weight,
    )
    return PublicTreeDecisionState.from_particle(
        particle,
        phase="t3_first",
        board_bb=BB_BOARD,
        board_btn=BTN_BOARD,
        public_action_history=PUBLIC_HISTORY,
        current_draw=("Ad", "Kd", "Qd"),
    )


SIGNAL_ACTIONS = {
    "bet": Action(
        placements=[("Ad", "bottom"), ("Kd", "bottom")],
        discard="Qd",
    ),
    "check": Action(
        placements=[("Ad", "bottom"), ("Qd", "bottom")],
        discard="Kd",
    ),
}


def _receiver_state(
    sender_state: PublicTreeDecisionState,
    signal: str,
) -> PublicTreeDecisionState:
    pending = apply_public_tree_action(sender_state, SIGNAL_ACTIONS[signal])
    return resolve_supplied_chance(
        pending,
        (SuppliedChanceDraw(("2h", "3h", "5h"), 1),),
    )[0].state


def _bluff_tree(
    *,
    reverse_order: bool = False,
    particle_weights=None,
):
    """Known 1/3-bluff, 2/3-call game on strict OFC information keys."""
    particle_weights = particle_weights or {
        "strong": Fraction(1, 1),
        "weak": Fraction(1, 1),
    }
    states = {
        "strong": _sender_state(
            "6s", particle_weight=particle_weights["strong"]
        ),
        "weak": _sender_state(
            "5s", particle_weight=particle_weights["weak"]
        ),
    }
    payoff = {
        "strong": {
            "bet": {"call": 2, "fold": 1},
            "check": {"call": 1, "fold": 1},
        },
        "weak": {
            "bet": {"call": -2, "fold": 1},
            "check": {"call": -1, "fold": -1},
        },
    }
    receiver_states = {
        (private_type, signal): _receiver_state(states[private_type], signal)
        for private_type in ("strong", "weak")
        for signal in ("bet", "check")
    }
    # The receiver sees the public signal but not the sender's earlier hidden
    # discard/type.  Therefore both physical worlds share exactly one key per
    # signal even though their JointParticle objects differ.
    assert (
        receiver_states[("strong", "bet")].infoset_key
        == receiver_states[("weak", "bet")].infoset_key
    )
    assert (
        receiver_states[("strong", "check")].infoset_key
        == receiver_states[("weak", "check")].infoset_key
    )

    type_order = ("weak", "strong") if reverse_order else ("strong", "weak")
    signal_order = ("check", "bet") if reverse_order else ("bet", "check")
    response_order = ("fold", "call") if reverse_order else ("call", "fold")
    sender_nodes = {}
    for private_type in type_order:
        sender_actions = []
        for signal in signal_order:
            receiver_actions = [
                (
                    response,
                    PublicTreeTerminalNode(
                        payoff[private_type][signal][response],
                        f"{private_type}/{signal}/{response}",
                    ),
                )
                for response in response_order
            ]
            sender_actions.append(
                (
                    signal,
                    PublicTreeDecisionNode(
                        receiver_states[(private_type, signal)],
                        receiver_actions,
                    ),
                )
            )
        sender_nodes[private_type] = PublicTreeDecisionNode(
            states[private_type],
            sender_actions,
        )
    chance_branches = [
        PublicTreeChanceBranch(private_type, Fraction(1, 2), sender_nodes[private_type])
        for private_type in type_order
    ]
    return PublicTreeChanceNode(chance_branches), states, receiver_states


def test_recursive_public_tree_cfr_converges_to_reduced_bluff_equilibrium():
    root, sender_states, receiver_states = _bluff_tree()
    result = solve_recursive_public_tree_cfr_plus(
        root,
        iterations=20_000,
        checkpoints=(10, 100, 1_000, 5_000, 20_000),
    )

    strong = result.average_strategy[sender_states["strong"].infoset_key]
    weak = result.average_strategy[sender_states["weak"].infoset_key]
    receiver_bet = result.average_strategy[
        receiver_states[("strong", "bet")].infoset_key
    ]
    assert strong["bet"] == pytest.approx(1.0, abs=0.01)
    assert weak["bet"] == pytest.approx(1 / 3, abs=0.03)
    assert receiver_bet["call"] == pytest.approx(2 / 3, abs=0.03)
    assert result.metrics.value_bb == pytest.approx(1 / 3, abs=0.02)
    assert result.metrics.btn_best_response <= result.metrics.value_bb
    assert result.metrics.value_bb <= result.metrics.bb_best_response
    assert result.metrics.exploitability < 0.02
    assert result.exploitability_trace[-1][1] < result.exploitability_trace[0][1]
    assert all(type(key).__name__ == "InfoSetKey" for key in result.average_strategy)
    assert all("world_id" not in key.canonical_json() for key in result.average_strategy)
    assert result.metadata == {
        "method": "reduced_recursive_public_tree_cfr_plus",
        "tree_scope": "finite_explicit_reduced_non_full_card",
        "recursive": True,
        "synchronous_updates": True,
        "infoset_aware_exact_best_response": True,
        "best_response_exact_scope": "exhaustive_pure_infoset_policy_enumeration",
        "numeric_utility_arithmetic": "float64",
        "chance_path_arithmetic": "exact_fraction_until_regret_multiply",
        "chance_source": "explicit_tree_branches_only",
        "joint_particle_weight_used": False,
        "terminal_utility_sources": ["explicit_reduced_terminal"],
        "strategy_fusion": False,
        "equilibrium_approx": True,
        "full_card": False,
        "hu_exact": False,
        "runtime_integrated": False,
        "rust_leaf_integrated": False,
        "position_contract_version": "bb_first_v1",
    }


def test_tree_and_input_order_do_not_change_recursive_cfr_result():
    canonical, _senders, _receivers = _bluff_tree(reverse_order=False)
    reversed_tree, _senders_reversed, _receivers_reversed = _bluff_tree(
        reverse_order=True
    )
    canonical_result = solve_recursive_public_tree_cfr_plus(
        canonical,
        iterations=5_000,
    )
    reversed_result = solve_recursive_public_tree_cfr_plus(
        reversed_tree,
        iterations=5_000,
    )

    by_digest = lambda result: {
        key.digest(): strategy for key, strategy in result.average_strategy.items()
    }
    assert by_digest(canonical_result) == by_digest(reversed_result)
    assert canonical_result.metrics == reversed_result.metrics
    assert canonical_result.exploitability_trace == reversed_result.exploitability_trace


def test_explicit_chance_is_the_only_probability_source_not_particle_weight():
    unit_weight_tree, _senders, _receivers = _bluff_tree()
    skewed_particle_tree, _skewed_senders, _skewed_receivers = _bluff_tree(
        particle_weights={
            "strong": Fraction(1, 100),
            "weak": Fraction(99, 100),
        }
    )
    unit = solve_recursive_public_tree_cfr_plus(unit_weight_tree, iterations=500)
    skewed = solve_recursive_public_tree_cfr_plus(
        skewed_particle_tree,
        iterations=500,
    )

    by_digest = lambda result: {
        key.digest(): strategy for key, strategy in result.average_strategy.items()
    }
    assert by_digest(unit) == by_digest(skewed)
    assert unit.metrics == skewed.metrics
    assert unit.metadata["chance_source"] == "explicit_tree_branches_only"
    assert unit.metadata["joint_particle_weight_used"] is False


def test_shared_infoset_rejects_different_action_sets():
    _root, _senders, receivers = _bluff_tree()
    state_a = receivers[("strong", "bet")]
    state_b = receivers[("weak", "bet")]
    assert state_a.infoset_key == state_b.infoset_key
    root = PublicTreeChanceNode(
        (
            PublicTreeChanceBranch(
                "a",
                Fraction(1, 2),
                PublicTreeDecisionNode(
                    state_a,
                    {"call": PublicTreeTerminalNode(0), "fold": PublicTreeTerminalNode(1)},
                ),
            ),
            PublicTreeChanceBranch(
                "b",
                Fraction(1, 2),
                PublicTreeDecisionNode(
                    state_b,
                    {"call": PublicTreeTerminalNode(2), "raise": PublicTreeTerminalNode(3)},
                ),
            ),
        )
    )

    with pytest.raises(ValueError, match="shared InfoSetKey action-set mismatch"):
        solve_recursive_public_tree_cfr_plus(root, iterations=1)


def _strategy_fusion_tree():
    _root, _senders, receivers = _bluff_tree()
    state_a = receivers[("strong", "bet")]
    state_b = receivers[("weak", "bet")]
    assert state_a.infoset_key == state_b.infoset_key
    return PublicTreeChanceNode(
        (
            PublicTreeChanceBranch(
                "physical-a",
                Fraction(1, 2),
                PublicTreeDecisionNode(
                    state_a,
                    {"left": PublicTreeTerminalNode(0), "right": PublicTreeTerminalNode(2)},
                ),
            ),
            PublicTreeChanceBranch(
                "physical-b",
                Fraction(1, 2),
                PublicTreeDecisionNode(
                    state_b,
                    {"left": PublicTreeTerminalNode(2), "right": PublicTreeTerminalNode(0)},
                ),
            ),
        )
    )


def test_exact_best_response_is_infoset_aware_and_exposes_strategy_fusion_gap():
    root = _strategy_fusion_tree()
    response = exact_public_tree_best_response(
        root,
        actor="btn",
        opponent_strategy={},
    )
    diagnostic = public_tree_strategy_fusion_diagnostic(
        root,
        actor="btn",
        opponent_strategy={},
    )

    # One shared action has expected BB value 1 in either direction.  The
    # illegal PIMC/world-aware response chooses left in physical-a and right in
    # physical-b, manufacturing a value of zero for the minimizing BTN.
    assert response.value_bb == pytest.approx(1.0)
    assert len(response.policy) == 1
    assert next(iter(response.policy.values())) == "left"  # stable tie break
    assert response.exact_scope == "exhaustive_pure_infoset_policy_enumeration"
    assert response.numeric_arithmetic == "float64"
    assert diagnostic.infoset_aware_value_bb == pytest.approx(1.0)
    assert diagnostic.illegal_per_history_value_bb == pytest.approx(0.0)
    assert diagnostic.strategy_fusion_advantage == pytest.approx(1.0)


def test_synchronous_regret_update_aggregates_shared_worlds_before_rm_plus():
    root = _strategy_fusion_tree()
    result = solve_recursive_public_tree_cfr_plus(root, iterations=1)
    key = next(iter(result.current_strategy))

    # The two worlds have equal-and-opposite counterfactual regrets.  A
    # synchronous shared-infoset update sums them to zero before CFR+ clips.
    assert result.cumulative_regret_plus[key] == {"left": 0.0, "right": 0.0}
    assert result.current_strategy[key] == {"left": 0.5, "right": 0.5}
    assert result.metrics.nash_conv == pytest.approx(0.0)


def _nested_bb_tree():
    root_state = _sender_state("6s")
    receiver_state = _receiver_state(root_state, "bet")
    after_receiver = apply_public_tree_action(
        receiver_state,
        Action(
            placements=[("2h", "bottom"), ("3h", "bottom")],
            discard="5h",
        ),
    )
    t4_state = resolve_supplied_chance(
        after_receiver,
        (SuppliedChanceDraw(("Ah", "Jd", "Kh"), 1),),
    )[0].state
    deeper = PublicTreeDecisionNode(
        t4_state,
        {
            "lose": PublicTreeTerminalNode(-1),
            "win": PublicTreeTerminalNode(1),
        },
    )
    root = PublicTreeDecisionNode(
        root_state,
        {
            "enter": deeper,
            "exit": PublicTreeTerminalNode(-2),
        },
    )
    return root, root_state.infoset_key, t4_state.infoset_key


def test_deep_same_actor_regret_excludes_own_reach_but_average_uses_it():
    root, root_key, deep_key = _nested_bb_tree()
    first = solve_recursive_public_tree_cfr_plus(root, iterations=1)

    # Iteration 1 reaches the deeper BB node with BB reach 1/2.  Regret uses
    # only chance and opponent reach, so win's +1 delta is not halved.
    assert first.cumulative_regret_plus[root_key]["enter"] == pytest.approx(1.0)
    assert first.cumulative_regret_plus[deep_key]["win"] == pytest.approx(1.0)

    second = solve_recursive_public_tree_cfr_plus(root, iterations=2)
    # Average strategy *does* use own reach.  Iteration 1 contributes
    # weight 1 * reach .5 * uniform; iteration 2 contributes
    # weight 2 * reach 1 * pure-win, yielding 2.25 / 2.5 = .9.
    assert second.average_strategy[deep_key]["win"] == pytest.approx(0.9)
    assert second.average_strategy[deep_key]["lose"] == pytest.approx(0.1)
