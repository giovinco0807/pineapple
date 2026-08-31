import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from ai.engine.action_space import Action
from ai.tutor.exact_late import default_rust_t3_exact_solver_path
from ai.tutor.t3_hu_reduced_fixtures import compile_canonical_reduced_fixture
from ai.tutor.t3_hu_public_cfr import JointParticle, PrivateRecall
from ai.tutor.t3_hu_public_mccfr import (
    ExplicitPublicTreeAdapter,
    PublicMccfrCheckpointError,
    compare_external_sampling_with_cfr_plus,
    estimate_external_sampling_regret_deltas,
    solve_external_sampling_public_mccfr,
    uniform_public_tree_strategy,
)
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
    (
        0,
        "bb",
        (
            ("4h", "top"),
            ("2c", "top"),
            ("3d", "top"),
            ("7c", "middle"),
            ("Qc", "bottom"),
        ),
    ),
    (
        0,
        "btn",
        (
            ("8c", "top"),
            ("5c", "top"),
            ("6d", "top"),
            ("9c", "middle"),
            ("As", "bottom"),
        ),
    ),
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
    private_type_discard: str,
    *,
    particle_weight: Fraction = Fraction(1, 1),
) -> PublicTreeDecisionState:
    particle = JointParticle(
        bb_recall=_recall("6c", private_type_discard, actor="bb"),
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


def _asymmetric_hidden_tree(
    *,
    reverse_order: bool = False,
    particle_weights: tuple[Fraction, Fraction] = (
        Fraction(1, 1),
        Fraction(1, 1),
    ),
    world_a_right_utility: int = 4,
):
    hidden_a = _receiver_state(
        _sender_state("6s", particle_weight=particle_weights[0]),
        "bet",
    )
    hidden_b = _receiver_state(
        _sender_state("5s", particle_weight=particle_weights[1]),
        "bet",
    )
    assert hidden_a.infoset_key == hidden_b.infoset_key
    actions_a = [
        ("left", PublicTreeTerminalNode(0, "a/left")),
        (
            "right",
            PublicTreeTerminalNode(world_a_right_utility, "a/right"),
        ),
    ]
    actions_b = [
        ("left", PublicTreeTerminalNode(2, "b/left")),
        ("right", PublicTreeTerminalNode(0, "b/right")),
    ]
    branches = [
        PublicTreeChanceBranch(
            "world-a",
            Fraction(1, 4),
            PublicTreeDecisionNode(
                hidden_a,
                reversed(actions_a) if reverse_order else actions_a,
            ),
        ),
        PublicTreeChanceBranch(
            "world-b",
            Fraction(3, 4),
            PublicTreeDecisionNode(
                hidden_b,
                reversed(actions_b) if reverse_order else actions_b,
            ),
        ),
    ]
    if reverse_order:
        branches.reverse()
    return PublicTreeChanceNode(branches), hidden_a.infoset_key


def _canonical_sha256(value) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _bluff_tree(*, particle_weights=None):
    particle_weights = particle_weights or {
        "strong": Fraction(1, 1),
        "weak": Fraction(1, 1),
    }
    sender_states = {
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
        (private_type, signal): _receiver_state(
            sender_states[private_type], signal
        )
        for private_type in ("strong", "weak")
        for signal in ("bet", "check")
    }
    assert (
        receiver_states[("strong", "bet")].infoset_key
        == receiver_states[("weak", "bet")].infoset_key
    )
    sender_nodes = {}
    for private_type in ("strong", "weak"):
        sender_nodes[private_type] = PublicTreeDecisionNode(
            sender_states[private_type],
            {
                signal: PublicTreeDecisionNode(
                    receiver_states[(private_type, signal)],
                    {
                        response: PublicTreeTerminalNode(
                            payoff[private_type][signal][response],
                            f"{private_type}/{signal}/{response}",
                        )
                        for response in ("call", "fold")
                    },
                )
                for signal in ("bet", "check")
            },
        )
    root = PublicTreeChanceNode(
        (
            PublicTreeChanceBranch("strong", Fraction(1, 2), sender_nodes["strong"]),
            PublicTreeChanceBranch("weak", Fraction(1, 2), sender_nodes["weak"]),
        )
    )
    return root, sender_states, receiver_states


def test_fixed_profile_external_sampling_regret_is_unbiased_and_samples_root_once():
    root, key = _asymmetric_hidden_tree()
    uniform = uniform_public_tree_strategy(root)
    estimate = estimate_external_sampling_regret_deltas(
        root,
        strategy=uniform,
        traverser="btn",
        samples=20_000,
        seed=20260713,
    )
    exact_first = solve_recursive_public_tree_cfr_plus(root, iterations=1)

    # world-a contributes (+2,-2) with probability 1/4 and world-b
    # contributes (-1,+1) with probability 3/4 for minimizing BTN.
    assert estimate.mean_regret_delta[key]["left"] == pytest.approx(
        -0.25, abs=0.03
    )
    assert estimate.mean_regret_delta[key]["right"] == pytest.approx(
        0.25, abs=0.03
    )
    assert estimate.mean_sampled_value_bb == pytest.approx(1.25, abs=0.03)
    assert exact_first.cumulative_regret_plus[key] == {
        "left": pytest.approx(0.0),
        "right": pytest.approx(0.25),
    }
    stats = estimate.sampling_stats
    assert stats["root_posterior_samples"] == 20_000
    assert stats["chance_action_samples"] == 20_000
    assert sum(stats["root_outcome_counts"].values()) == 20_000
    assert estimate.metadata["chance_probability_multiplied_after_sampling"] is False
    assert estimate.metadata["joint_particle_weight_used"] is False
    assert estimate.metadata["root_is_chance"] is True
    assert estimate.metadata["root_posterior_sampled_once_per_traversal"] is True


def test_seed_input_order_and_hidden_particle_weights_cannot_change_policy_identity():
    canonical, key = _asymmetric_hidden_tree()
    reversed_tree, reversed_key = _asymmetric_hidden_tree(reverse_order=True)
    skewed, skewed_key = _asymmetric_hidden_tree(
        particle_weights=(Fraction(1, 100), Fraction(99, 100))
    )
    canonical_result = solve_external_sampling_public_mccfr(
        canonical,
        iterations=500,
        seed=17,
    )
    replay_result = solve_external_sampling_public_mccfr(
        canonical,
        iterations=500,
        seed=17,
    )
    reversed_result = solve_external_sampling_public_mccfr(
        reversed_tree,
        iterations=500,
        seed=17,
    )
    skewed_result = solve_external_sampling_public_mccfr(
        skewed,
        iterations=500,
        seed=17,
    )

    assert key == reversed_key == skewed_key
    assert canonical_result == replay_result
    assert canonical_result.average_strategy == reversed_result.average_strategy
    assert canonical_result.current_strategy == reversed_result.current_strategy
    assert canonical_result.cumulative_regret_plus == (
        reversed_result.cumulative_regret_plus
    )
    assert canonical_result.sampling_stats == reversed_result.sampling_stats
    assert canonical_result.average_strategy == skewed_result.average_strategy
    assert canonical_result.current_strategy == skewed_result.current_strategy
    assert canonical_result.cumulative_regret_plus == skewed_result.cumulative_regret_plus
    assert canonical_result.metrics == skewed_result.metrics
    assert canonical_result.metadata["joint_particle_weight_used"] is False


def test_same_opponent_infoset_samples_one_action_across_traverser_branches():
    bb_state = _sender_state("6s")
    shared_btn_state = _receiver_state(bb_state, "bet")
    shared_key = shared_btn_state.infoset_key
    root = PublicTreeDecisionNode(
        bb_state,
        {
            "left": PublicTreeDecisionNode(
                shared_btn_state,
                {
                    "x": PublicTreeTerminalNode(1),
                    "y": PublicTreeTerminalNode(0),
                },
            ),
            "right": PublicTreeDecisionNode(
                shared_btn_state,
                {
                    "x": PublicTreeTerminalNode(0),
                    "y": PublicTreeTerminalNode(1),
                },
            ),
        },
    )
    estimate = estimate_external_sampling_regret_deltas(
        root,
        strategy=uniform_public_tree_strategy(root),
        traverser="bb",
        samples=1,
        seed=9,
    )

    # BB enumerates both root actions, so it visits two physical BTN nodes.
    # They share one InfoSetKey and therefore one sampled pure action.
    assert estimate.sampling_stats["opponent_action_samples"] == 1
    assert estimate.sampling_stats["opponent_cache_hits"] == 1
    assert list(
        key
        for key in uniform_public_tree_strategy(root)
        if key.actor == "btn"
    ) == [shared_key]
    assert estimate.metadata["opponent_sample_cached_per_infoset"] is True
    assert estimate.metadata["strategy_fusion"] is False
    assert estimate.metadata["root_is_chance"] is False
    assert estimate.metadata["root_posterior_sampled_once_per_traversal"] is False


def test_shared_hidden_worlds_train_one_legal_infoset_policy_not_per_world_pimc():
    root, key = _asymmetric_hidden_tree()
    compiled = ExplicitPublicTreeAdapter(root).compile()
    result = solve_external_sampling_public_mccfr(
        root,
        iterations=2_000,
        seed=31,
        checkpoints=(1, 100, 2_000),
    )

    assert len(compiled.stable_infosets) == 1
    assert list(result.average_strategy) == [key]
    # Legal BTN response minimizes E[utility]: left=1.5, right=1.0.
    # A forbidden per-world response would manufacture utility zero.
    assert result.average_strategy[key]["right"] > 0.90
    assert result.metrics is not None
    assert result.metrics.value_bb == pytest.approx(1.0, abs=0.04)
    assert result.metrics.exploitability < 0.02
    assert result.metadata["strategy_fusion"] is False
    assert result.sampling_stats["traversals_by_actor"] == {
        "bb": 2_000,
        "btn": 2_000,
    }
    assert result.sampling_stats["root_posterior_samples"] == 4_000


def test_reduced_mccfr_has_direct_cfr_plus_oracle_comparison_api():
    root, sender_states, receiver_states = _bluff_tree()
    comparison = compare_external_sampling_with_cfr_plus(
        root,
        sampled_iterations=20_000,
        oracle_iterations=20_000,
        seed=73,
    )

    strong = comparison.sampled.average_strategy[
        sender_states["strong"].infoset_key
    ]
    weak = comparison.sampled.average_strategy[
        sender_states["weak"].infoset_key
    ]
    receiver_bet = comparison.sampled.average_strategy[
        receiver_states[("strong", "bet")].infoset_key
    ]
    assert strong["bet"] == pytest.approx(1.0, abs=0.02)
    assert weak["bet"] == pytest.approx(1 / 3, abs=0.06)
    assert receiver_bet["call"] == pytest.approx(2 / 3, abs=0.06)
    assert comparison.sampled.metrics is not None
    assert comparison.sampled.metrics.exploitability < 0.05
    assert comparison.oracle.metrics.exploitability < 0.02
    assert abs(comparison.value_bb_gap) < 0.05
    assert comparison.max_strategy_total_variation < 0.08
    assert comparison.metadata["full_card_claim"] is False


def test_canonical_real_reduced_fixture_uses_the_same_oracle_comparison_api():
    rust_solver = default_rust_t3_exact_solver_path()
    if not Path(rust_solver).exists():
        pytest.skip(f"Rust late exact solver is not built: {rust_solver}")
    fixture = compile_canonical_reduced_fixture(
        "btn",
        0,
        rust_solver_path=rust_solver,
        rust_timeout_s=30.0,
    )
    comparison = compare_external_sampling_with_cfr_plus(
        fixture.root,
        sampled_iterations=500,
        oracle_iterations=200,
        seed=41,
        max_pure_profiles=100_000,
    )

    assert comparison.sampled.metrics is not None
    assert comparison.sampled.metrics.exploitability < 1e-4
    assert comparison.oracle.metrics.exploitability < 1e-4
    assert abs(comparison.value_bb_gap) < 1e-3
    assert comparison.sampled.metadata["rust_leaf_integrated"] is True
    assert comparison.sampled.metadata["joint_particle_weight_used"] is False


def test_checkpoint_resume_n_plus_m_is_bitwise_identical_to_one_shot(
    tmp_path,
):
    root, key = _asymmetric_hidden_tree()
    checkpoint_n = tmp_path / "after-n.json"
    one_shot_final = tmp_path / "one-shot-final.json"

    first = solve_external_sampling_public_mccfr(
        root,
        iterations=37,
        seed=101,
        checkpoint_path=checkpoint_n,
    )
    envelope_n = json.loads(checkpoint_n.read_text(encoding="utf-8"))
    assert envelope_n["checkpoint_sha256"] == _canonical_sha256(
        envelope_n["payload"]
    )
    assert envelope_n["payload"]["completed_iterations"] == 37
    assert envelope_n["payload"]["sampling_stats"]["traversals"] == 74
    table = envelope_n["payload"]["tables"][0]
    assert table["infoset_canonical_json"] == key.canonical_json()
    assert table["stable_action_ids"] == ["left", "right"]
    assert all(value.startswith("0x") for value in table["regret_plus_hex"])
    assert first.metadata["checkpoint_iteration"] == 37
    assert first.metadata["checkpoint_sha256"] == envelope_n["checkpoint_sha256"]
    assert not list(tmp_path.glob(".after-n.json.*.tmp"))

    resumed = solve_external_sampling_public_mccfr(
        root,
        iterations=63,
        seed=101,
        resume_from=checkpoint_n,
        checkpoint_path=checkpoint_n,
    )
    one_shot = solve_external_sampling_public_mccfr(
        root,
        iterations=100,
        seed=101,
        checkpoint_path=one_shot_final,
    )

    assert resumed == one_shot
    assert resumed.iterations == 100
    assert resumed.traversals == 200
    assert checkpoint_n.read_bytes() == one_shot_final.read_bytes()
    assert not list(tmp_path.glob(".after-n.json.*.tmp"))
    assert resumed.metadata["checkpoint_iteration"] == 100
    assert resumed.metadata["checkpoint_sha256"] == one_shot.metadata[
        "checkpoint_sha256"
    ]


def test_checkpoint_corruption_and_forged_action_table_fail_closed(tmp_path):
    root, _key = _asymmetric_hidden_tree()
    checkpoint = tmp_path / "valid.json"
    solve_external_sampling_public_mccfr(
        root,
        iterations=11,
        seed=29,
        checkpoint_path=checkpoint,
    )
    envelope = json.loads(checkpoint.read_text(encoding="utf-8"))

    truncated = tmp_path / "truncated.json"
    truncated.write_text("{", encoding="utf-8")
    with pytest.raises(PublicMccfrCheckpointError, match="valid UTF-8 JSON"):
        solve_external_sampling_public_mccfr(
            root,
            iterations=1,
            seed=29,
            resume_from=truncated,
        )

    hash_mismatch = tmp_path / "hash-mismatch.json"
    mismatched = json.loads(json.dumps(envelope))
    mismatched["payload"]["completed_iterations"] = 12
    hash_mismatch.write_text(json.dumps(mismatched), encoding="utf-8")
    with pytest.raises(PublicMccfrCheckpointError, match="content SHA-256 mismatch"):
        solve_external_sampling_public_mccfr(
            root,
            iterations=1,
            seed=29,
            resume_from=hash_mismatch,
        )

    forged_actions = tmp_path / "forged-actions.json"
    forged = json.loads(json.dumps(envelope))
    forged["payload"]["tables"][0]["stable_action_ids"][0] = "forged"
    forged["checkpoint_sha256"] = _canonical_sha256(forged["payload"])
    forged_actions.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(PublicMccfrCheckpointError, match="stable action IDs"):
        solve_external_sampling_public_mccfr(
            root,
            iterations=1,
            seed=29,
            resume_from=forged_actions,
        )


def test_checkpoint_rejects_different_tree_seed_and_training_config(tmp_path):
    root, _key = _asymmetric_hidden_tree()
    checkpoint = tmp_path / "valid.json"
    solve_external_sampling_public_mccfr(
        root,
        iterations=13,
        seed=47,
        linear_averaging=True,
        checkpoint_path=checkpoint,
    )

    different_tree, _different_key = _asymmetric_hidden_tree(
        world_a_right_utility=5
    )
    with pytest.raises(PublicMccfrCheckpointError, match="tree manifest"):
        solve_external_sampling_public_mccfr(
            different_tree,
            iterations=1,
            seed=47,
            resume_from=checkpoint,
        )
    with pytest.raises(PublicMccfrCheckpointError, match="seed"):
        solve_external_sampling_public_mccfr(
            root,
            iterations=1,
            seed=48,
            resume_from=checkpoint,
        )
    with pytest.raises(PublicMccfrCheckpointError, match="linear_averaging"):
        solve_external_sampling_public_mccfr(
            root,
            iterations=1,
            seed=47,
            linear_averaging=False,
            resume_from=checkpoint,
        )
