import copy
import hashlib
import itertools
import json
import math
import random
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

import pytest

import ai.tutor.t3_hu_full_card_range as range_module
import ai.tutor.t3_hu_multi_root_mccfr as multi_root_module
from ai.engine.action_space import Action
from ai.engine.encoding import ALL_CARDS
from ai.engine.turn_order import POSITION_CONTRACT_VERSION
from ai.tutor.t3_hu_full_card_mccfr import FullCardGenerativeAdapter
from ai.tutor.t3_hu_full_card_range import (
    UniformLegalBehaviorModel,
    build_history_weighted_full_card_range,
)
from ai.tutor.t3_hu_multi_root_mccfr import (
    CanonicalReducedPublicTreeAdapter,
    MULTI_ROOT_CHECKPOINT_FORMAT,
    MultiRootChanceEntry,
    MultiRootMccfrCheckpointError,
    diagnose_independent_root_strategy_fusion,
    solve_multi_root_external_sampling_mccfr,
    verify_multi_root_checkpoint_against_result,
)
from ai.tutor.t3_hu_public_cfr import InfoSetKey, JointParticle, PrivateRecall
from ai.tutor.t3_hu_public_mccfr import solve_external_sampling_public_mccfr
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
CURRENT_DRAW = ("Ad", "Kd", "Qd")


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


def _sender_state(private_type_discard: str) -> PublicTreeDecisionState:
    particle = JointParticle(
        bb_recall=_recall("6c", private_type_discard, actor="bb"),
        btn_recall=_recall("4s", "7h", actor="btn"),
        undealt_cards=("2h", "3h", "5h", "Ah", "Jd", "Kh"),
        weight=1,
    )
    return PublicTreeDecisionState.from_particle(
        particle,
        phase="t3_first",
        board_bb=BB_BOARD,
        board_btn=BTN_BOARD,
        public_action_history=PUBLIC_HISTORY,
        current_draw=CURRENT_DRAW,
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


def _bluff_tree():
    sender_states = {
        "strong": _sender_state("6s"),
        "weak": _sender_state("5s"),
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
            PublicTreeChanceBranch(
                "strong", Fraction(1, 2), sender_nodes["strong"]
            ),
            PublicTreeChanceBranch(
                "weak", Fraction(1, 2), sender_nodes["weak"]
            ),
        )
    )
    return root, sender_nodes, sender_states, receiver_states


@dataclass(frozen=True)
class _ExplicitRootSample:
    state: PublicTreeDecisionNode


@dataclass(frozen=True)
class _ExplicitTransition:
    state: PublicTreeDecisionNode
    next_phase: str


class _ExplicitBranchAdapter:
    """Unit-probability generative view of one exact reduced chance branch."""

    def __init__(self, root: PublicTreeDecisionNode, commitment: str) -> None:
        self.root = root
        self.observation = root.infoset_key
        self._commitment = commitment
        self._traversals = 0
        self._future_draws = 0

    @property
    def root_distribution(self):
        return ((self._commitment, Fraction(1, 1)),)

    @property
    def checkpoint_binding_manifest(self):
        infoset_actions = {}

        def encode(node):
            if isinstance(node, PublicTreeTerminalNode):
                return {
                    "kind": "terminal",
                    "utility_bb_exact": (
                        f"{node.utility_bb.numerator}/{node.utility_bb.denominator}"
                    ),
                    "terminal_id": node.terminal_id,
                    "utility_source": node.utility_source,
                }
            if isinstance(node, PublicTreeChanceNode):
                return {
                    "kind": "chance",
                    "branches": [
                        {
                            "outcome_id": branch.outcome_id,
                            "probability_exact": (
                                f"{branch.probability.numerator}/"
                                f"{branch.probability.denominator}"
                            ),
                            "child": encode(branch.child),
                        }
                        for branch in node.branches
                    ],
                }
            if not isinstance(node, PublicTreeDecisionNode):
                raise TypeError("unsupported explicit checkpoint node")
            key = node.infoset_key
            prior = infoset_actions.get(key)
            if prior is not None and prior != node.action_ids:
                raise ValueError("shared explicit infoset action mismatch")
            infoset_actions[key] = node.action_ids
            return {
                "kind": "decision",
                "infoset_sha256": key.digest(),
                "actions": [
                    {"action_id": action_id, "child": encode(child)}
                    for action_id, child in node.actions
                ],
            }

        tree = encode(self.root)
        rows = [
            {
                "infoset_canonical_json": key.canonical_json(),
                "infoset_sha256": key.digest(),
                "actor": key.actor,
                "stable_action_ids": list(infoset_actions[key]),
            }
            for key in sorted(
                infoset_actions,
                key=lambda value: (value.digest(), value.canonical_json()),
            )
        ]
        tree_json = json.dumps(
            tree,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return {
            "schema": "test_explicit_public_tree_checkpoint_binding_v1",
            "implementation_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "tree_sha256": hashlib.sha256(tree_json.encode("utf-8")).hexdigest(),
            "infoset_actions": rows,
        }

    def sample_root_for_traversal(self, _rng):
        self._traversals += 1
        return _ExplicitRootSample(self.root)

    @staticmethod
    def information_key(state):
        return state.infoset_key

    @staticmethod
    def legal_actions(state):
        return tuple((action_id, child) for action_id, child in state.actions)

    @staticmethod
    def apply_action_id(state, stable_action_id):
        return dict(state.actions)[stable_action_id]

    @staticmethod
    def is_terminal_result(value):
        return isinstance(value, PublicTreeTerminalNode)

    def sample_next_draw(self, pending, _rng):
        if not isinstance(pending, PublicTreeDecisionNode):
            raise TypeError("reduced direct transition requires a decision node")
        self._future_draws += 1
        return _ExplicitTransition(pending, pending.infoset_key.phase)

    @staticmethod
    def terminal_utility_bb(terminal):
        return float(terminal.utility_bb)

    def sampling_audit(self):
        return {
            "traversals": self._traversals,
            "root_posterior_samples": self._traversals,
            "future_draw_samples": self._future_draws,
        }


def _multi_root_entries(sender_nodes):
    return (
        MultiRootChanceEntry(
            "private-type-strong",
            CanonicalReducedPublicTreeAdapter(
                sender_nodes["strong"], "strong-physical"
            ),
            Fraction(1, 2),
        ),
        MultiRootChanceEntry(
            "private-type-weak",
            CanonicalReducedPublicTreeAdapter(
                sender_nodes["weak"], "weak-physical"
            ),
            Fraction(1, 2),
        ),
    )


def _generic_multi_root_entries(sender_nodes):
    return (
        MultiRootChanceEntry(
            "private-type-strong",
            _ExplicitBranchAdapter(sender_nodes["strong"], "strong-physical"),
            Fraction(1, 2),
        ),
        MultiRootChanceEntry(
            "private-type-weak",
            _ExplicitBranchAdapter(sender_nodes["weak"], "weak-physical"),
            Fraction(1, 2),
        ),
    )


def _max_profile_tv(first, second):
    # Missing an oracle infoset is a coverage failure, not a zero-error row.
    # Compare only after proving complete profile-key and action-set parity.
    assert set(first) == set(second)
    shared = set(first)
    assert shared
    for key in shared:
        assert set(first[key]) == set(second[key])
    return max(
        0.5
        * math.fsum(
            abs(first[key][action_id] - second[key][action_id])
            for action_id in first[key]
        )
        for key in shared
    )


def test_shared_multi_root_solver_converges_toward_existing_exact_cfr_reference():
    root, sender_nodes, _sender_states, receiver_states = _bluff_tree()
    oracle = solve_recursive_public_tree_cfr_plus(root, iterations=5_000)

    low_errors = []
    high_errors = []
    for seed in (11, 37, 73):
        low = solve_multi_root_external_sampling_mccfr(
            _multi_root_entries(sender_nodes),
            iterations=100,
            seed=seed,
            max_infosets=20,
        )
        high = solve_multi_root_external_sampling_mccfr(
            _multi_root_entries(sender_nodes),
            iterations=3_000,
            seed=seed,
            max_infosets=20,
        )
        low_errors.append(
            _max_profile_tv(low.average_strategy, oracle.average_strategy)
        )
        high_errors.append(
            _max_profile_tv(high.average_strategy, oracle.average_strategy)
        )

    assert math.fsum(high_errors) / len(high_errors) < (
        math.fsum(low_errors) / len(low_errors)
    )
    assert max(high_errors) < 0.08

    shared = solve_multi_root_external_sampling_mccfr(
        _multi_root_entries(sender_nodes),
        iterations=3_000,
        seed=73,
        max_infosets=20,
    )
    shared_bet_key = receiver_states[("strong", "bet")].infoset_key
    assert shared.infoset_root_support_count[shared_bet_key] == 2
    assert shared_bet_key in shared.shared_across_roots_infosets
    assert shared.average_strategy[shared_bet_key]["call"] == pytest.approx(
        2 / 3, abs=0.06
    )
    assert shared.sampling_stats["super_root_samples"] == 6_000
    assert shared.sampling_stats["conditional_root_posterior_samples"] == 6_000
    assert shared.sampling_stats["distinct_root_adapters_sampled"] == 2
    assert shared.metadata["policy_table_shared_across_all_roots"] is True
    assert shared.metadata["strategy_fusion"] is False
    assert shared.metadata["table_key_contains_root_id"] is False
    assert shared.metadata["exact_exploitability_computed"] is False
    assert shared.metadata["promotion_eligible"] is False
    assert shared.metadata["global_unseen_state_policy_claim"] is False
    assert shared.metadata["root_id_relabeling_is_generalization_evidence"] is False
    assert shared.metadata["root_disjoint_holdout_profile_reuse_valid"] is False
    assert shared.metadata["checkpoint_artifact_purpose"] == [
        "algorithm_validation",
        "online_root_solve_resume",
    ]
    assert "private-type-strong" not in shared.average_strategy_json
    assert "private-type-weak" not in shared.average_strategy_json


def test_independent_per_root_solutions_expose_strategy_fusion_error():
    root, sender_nodes, _sender_states, receiver_states = _bluff_tree()
    independent_strong = solve_external_sampling_public_mccfr(
        sender_nodes["strong"], iterations=5_000, seed=17
    )
    independent_weak = solve_external_sampling_public_mccfr(
        sender_nodes["weak"], iterations=5_000, seed=19
    )
    shared = solve_multi_root_external_sampling_mccfr(
        _multi_root_entries(sender_nodes),
        iterations=5_000,
        seed=23,
        max_infosets=20,
    )
    diagnostic = diagnose_independent_root_strategy_fusion(
        (
            independent_strong.average_strategy,
            independent_weak.average_strategy,
        ),
        shared.average_strategy,
        prior_masses=(Fraction(1, 2), Fraction(1, 2)),
    )

    shared_bet_key = receiver_states[("strong", "bet")].infoset_key
    strong_bet = independent_strong.average_strategy[shared_bet_key]
    weak_bet = independent_weak.average_strategy[shared_bet_key]
    assert strong_bet["fold"] > 0.95
    assert weak_bet["call"] > 0.95
    assert diagnostic.pairwise_total_variation[shared_bet_key] > 0.90
    assert diagnostic.strategy_fusion_error_detected is True
    assert diagnostic.metadata["pairwise_difference_means_hidden_root_conditioning"] is True
    assert diagnostic.metadata["prior_mixture_is_exploitability"] is False
    assert diagnostic.metadata["exact_exploitability_computed"] is False
    # The legal shared receiver cannot select the strong-root fold policy and
    # weak-root call policy after observing the same public bet infoset.
    assert shared.average_strategy[shared_bet_key]["call"] == pytest.approx(
        2 / 3, abs=0.06
    )
    assert solve_recursive_public_tree_cfr_plus(
        root, iterations=5_000
    ).average_strategy[shared_bet_key]["call"] == pytest.approx(2 / 3, abs=0.02)


def test_opponent_action_is_cached_once_per_shared_infoset_and_traversal():
    roots = []
    for private_type, discard in (("strong", "6s"), ("weak", "5s")):
        sender = _sender_state(discard)
        receiver = _receiver_state(sender, "bet")
        left_receiver = PublicTreeDecisionNode(
            receiver,
            {
                "x": PublicTreeTerminalNode(1),
                "y": PublicTreeTerminalNode(0),
            },
        )
        right_receiver = PublicTreeDecisionNode(
            receiver,
            {
                "x": PublicTreeTerminalNode(0),
                "y": PublicTreeTerminalNode(1),
            },
        )
        root = PublicTreeDecisionNode(
            sender,
            {"left": left_receiver, "right": right_receiver},
        )
        roots.append(
            MultiRootChanceEntry(
                private_type,
                _ExplicitBranchAdapter(root, f"{private_type}-world"),
                Fraction(1, 2),
            )
        )

    result = solve_multi_root_external_sampling_mccfr(
        tuple(roots),
        iterations=1,
        seed=9,
        max_infosets=20,
        linear_averaging=False,
    )

    # The BB traversal expands both sender actions and reaches two physical
    # BTN nodes with one InfoSetKey.  External sampling must reuse one pure BTN
    # action rather than sampling independently in each physical branch.
    assert result.sampling_stats["opponent_action_cache_hits"] == 1
    assert result.sampling_stats["opponent_action_samples"] == 2
    assert result.metadata["opponent_sample_cached_per_infoset"] is True


class _FixedBits:
    def __init__(self, value: int) -> None:
        self.value = value

    def getrandbits(self, k: int) -> int:
        assert 0 <= self.value < (1 << k)
        return self.value


class _ForcedSharedDrawAdapter(FullCardGenerativeAdapter):
    def __init__(self, observation, root_range, forced_draw):
        super().__init__(observation, root_range)
        self._forced_draw = tuple(sorted(forced_draw))

    def sample_next_draw(self, pending, rng):
        if pending.next_phase == "t3_second":
            combinations = tuple(
                itertools.combinations(sorted(pending.remaining_cards), 3)
            )
            rank = combinations.index(self._forced_draw)
            return super().sample_next_draw(pending, _FixedBits(rank))
        return super().sample_next_draw(pending, rng)


def _full_observation(private_t2_discard: str) -> InfoSetKey:
    witness = JointParticle(
        bb_recall=_recall("6c", private_t2_discard, actor="bb"),
        btn_recall=_recall("X1", "X2", actor="btn"),
        undealt_cards=("2h", "3h", "5h"),
        weight=1,
    )
    return InfoSetKey.for_particle(
        witness,
        contract_version=POSITION_CONTRACT_VERSION,
        actor="bb",
        turn=3,
        phase="t3_first",
        board_bb=BB_BOARD,
        board_btn=BTN_BOARD,
        public_action_history=PUBLIC_HISTORY,
        current_draw=CURRENT_DRAW,
    )


def _single_world_range(monkeypatch, observation):
    def select_assignment(pool, count, *, max_particles, seed, observation_digest):
        assert count == 2
        assert {"X1", "X2"}.issubset(pool)
        return (("X1", "X2"),), math.perm(len(pool), count), False

    monkeypatch.setattr(range_module, "_select_assignments", select_assignment)
    return build_history_weighted_full_card_range(
        observation,
        UniformLegalBehaviorModel(),
        epsilon=0,
        max_particles=1,
        seed=31,
    )


def _install_light_terminal(monkeypatch, *adapters):
    card_value = {card: index + 1 for index, card in enumerate(ALL_CARDS)}

    def light_terminal(terminal):
        def board_value(rows):
            return sum(
                (row_index + 1) * card_value[card]
                for row_index, row in enumerate(rows)
                for card in row
            )

        return float(board_value(terminal.board_bb) - board_value(terminal.board_btn))

    for adapter in adapters:
        monkeypatch.setattr(adapter, "terminal_utility_bb", light_terminal)


def test_full_card_private_roots_merge_one_reached_btn_infoset(monkeypatch):
    observation_a = _full_observation("6s")
    observation_b = _full_observation("5s")
    assert observation_a != observation_b
    range_a = _single_world_range(monkeypatch, observation_a)
    range_b = _single_world_range(monkeypatch, observation_b)
    forced_draw = ("2h", "3h", "5h")
    adapter_a = _ForcedSharedDrawAdapter(observation_a, range_a, forced_draw)
    adapter_b = _ForcedSharedDrawAdapter(observation_b, range_b, forced_draw)
    _install_light_terminal(monkeypatch, adapter_a, adapter_b)

    # The two roots have different BB private recall.  After the same public
    # placement and BTN draw, BTN is nevertheless at exactly one InfoSetKey.
    state_a = adapter_a.sample_root_for_traversal(random.Random(1)).state
    state_b = adapter_b.sample_root_for_traversal(random.Random(1)).state
    action_id = adapter_a.legal_actions(state_a)[0][0]
    assert action_id == adapter_b.legal_actions(state_b)[0][0]
    btn_a = adapter_a.sample_next_draw(
        adapter_a.apply_action_id(state_a, action_id), random.Random(2)
    ).state.infoset_key
    btn_b = adapter_b.sample_next_draw(
        adapter_b.apply_action_id(state_b, action_id), random.Random(2)
    ).state.infoset_key
    assert btn_a == btn_b
    assert btn_a.actor == "btn"

    result = solve_multi_root_external_sampling_mccfr(
        (
            MultiRootChanceEntry(
                "bb-private-root-a", adapter_a, Fraction(1, 2)
            ),
            MultiRootChanceEntry(
                "bb-private-root-b", adapter_b, Fraction(1, 2)
            ),
        ),
        iterations=8,
        seed=20260713,
        max_infosets=100_000,
        linear_averaging=False,
    )

    assert result.infoset_root_support_count[btn_a] == 2
    assert btn_a in result.shared_across_roots_infosets
    assert list(key for key in result.average_strategy if key == btn_a) == [btn_a]
    assert all(isinstance(key, InfoSetKey) for key in result.average_strategy)
    assert result.metadata["compatible_full_card_adapters"] is True
    assert result.metadata["table_key_contains_root_id"] is False
    assert result.metadata["table_key_contains_private_type_id"] is False
    assert result.metadata["table_key_contains_particle_commitment"] is False
    assert result.metadata["full_card_policy_promoted"] is False
    assert result.metadata["exact_exploitability_computed"] is False
    assert "bb-private-root-a" not in result.average_strategy_json
    assert "bb-private-root-b" not in result.average_strategy_json


def test_exact_prior_and_adapter_compatibility_fail_closed():
    _root, sender_nodes, _sender_states, _receiver_states = _bluff_tree()
    strong_adapter = _ExplicitBranchAdapter(sender_nodes["strong"], "strong")
    weak_adapter = _ExplicitBranchAdapter(sender_nodes["weak"], "weak")
    with pytest.raises(TypeError, match="fractions.Fraction"):
        MultiRootChanceEntry("strong", strong_adapter, 0.5)
    with pytest.raises(ValueError, match="sum exactly to one"):
        solve_multi_root_external_sampling_mccfr(
            (
                MultiRootChanceEntry(
                    "strong", strong_adapter, Fraction(1, 3)
                ),
                MultiRootChanceEntry("weak", weak_adapter, Fraction(1, 3)),
            ),
            iterations=1,
            seed=1,
            max_infosets=20,
        )
    with pytest.raises(ValueError, match="distinct adapter"):
        solve_multi_root_external_sampling_mccfr(
            (
                MultiRootChanceEntry("first", strong_adapter, Fraction(1, 2)),
                MultiRootChanceEntry("second", strong_adapter, Fraction(1, 2)),
            ),
            iterations=1,
            seed=1,
            max_infosets=20,
        )


def _canonical_checkpoint_bytes(envelope):
    return (
        json.dumps(
            envelope,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _write_checkpoint_envelope(path, envelope):
    path.write_bytes(_canonical_checkpoint_bytes(envelope))


def _rehash_checkpoint(envelope):
    payload_bytes = json.dumps(
        envelope["payload"],
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    envelope["checkpoint_sha256"] = hashlib.sha256(payload_bytes).hexdigest()
    return envelope


def _rehash_runtime_state_and_checkpoint(envelope):
    payload = envelope["payload"]
    runtime_state = {
        "completed_iterations": payload["completed_iterations"],
        "tables": payload["tables"],
        "rng_state": payload["rng_state"],
        "sampling_stats": payload["sampling_stats"],
    }
    payload["runtime_state_binding_sha256"] = hashlib.sha256(
        json.dumps(
            runtime_state,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return _rehash_checkpoint(envelope)


@pytest.mark.parametrize("linear_averaging", (False, True))
def test_checkpoint_resume_is_byte_identical_to_one_shot(
    tmp_path,
    linear_averaging,
):
    _root, sender_nodes, _sender_states, _receiver_states = _bluff_tree()
    one_shot_path = tmp_path / f"one-shot-{linear_averaging}.json"
    split_path = tmp_path / f"split-{linear_averaging}.json"
    first_iterations = 29
    second_iterations = 51
    total_iterations = first_iterations + second_iterations

    one_shot = solve_multi_root_external_sampling_mccfr(
        _multi_root_entries(sender_nodes),
        iterations=total_iterations,
        seed=20260713,
        max_infosets=20,
        linear_averaging=linear_averaging,
        checkpoint_path=one_shot_path,
    )
    first = solve_multi_root_external_sampling_mccfr(
        _multi_root_entries(sender_nodes),
        iterations=first_iterations,
        seed=20260713,
        max_infosets=20,
        linear_averaging=linear_averaging,
        checkpoint_path=split_path,
    )
    assert first.iterations == first_iterations
    first_envelope = json.loads(split_path.read_text(encoding="utf-8"))
    first_runtime_graph_sha256 = first_envelope["payload"]["source_binding"][
        "live_runtime_semantic_binding"
    ]["binding_sha256"]
    # Input order is intentionally reversed. Canonical root-ID ordering must
    # make it irrelevant to RNG replay, tables, manifests, and final bytes.
    resumed = solve_multi_root_external_sampling_mccfr(
        tuple(reversed(_multi_root_entries(sender_nodes))),
        iterations=second_iterations,
        seed=20260713,
        max_infosets=20,
        linear_averaging=linear_averaging,
        resume_from=split_path,
        checkpoint_path=split_path,
        expected_checkpoint_sha256=first.metadata["checkpoint_sha256"],
    )

    assert resumed.iterations == one_shot.iterations == total_iterations
    assert resumed.traversals == one_shot.traversals == 2 * total_iterations
    assert resumed.average_strategy_json == one_shot.average_strategy_json
    assert resumed.current_strategy_json == one_shot.current_strategy_json
    assert resumed.average_strategy_sha256 == one_shot.average_strategy_sha256
    assert resumed.current_strategy_sha256 == one_shot.current_strategy_sha256
    assert resumed.cumulative_regret_plus == one_shot.cumulative_regret_plus
    assert resumed.infoset_root_support_count == one_shot.infoset_root_support_count
    assert resumed.shared_across_roots_infosets == one_shot.shared_across_roots_infosets
    assert resumed.sampling_stats == one_shot.sampling_stats
    assert resumed.metadata == one_shot.metadata
    assert resumed.metadata["checkpoint_iteration_boundary_only"] is True
    assert resumed.metadata["checkpoint_resume_requires_external_sha256"] is True
    assert resumed.metadata["checkpoint_stats_promotion_evidence"] is False
    assert (
        resumed.metadata["checkpoint_non_derived_stats_trusted_for_promotion"]
        is False
    )
    assert resumed.metadata["checkpoint_sha256"] is not None
    assert split_path.read_bytes() == one_shot_path.read_bytes()
    envelope = json.loads(split_path.read_text(encoding="utf-8"))
    assert envelope["format"] == MULTI_ROOT_CHECKPOINT_FORMAT
    assert split_path.read_bytes() == _canonical_checkpoint_bytes(envelope)
    assert envelope["payload"]["source_binding"][
        "live_runtime_semantic_binding"
    ]["binding_sha256"] == first_runtime_graph_sha256
    serialized_strategy = resumed.average_strategy_json
    for forbidden in (
        "private-type-strong",
        "private-type-weak",
        "strong-physical",
        "weak-physical",
        '"particle_commitment"',
    ):
        assert forbidden not in serialized_strategy


def test_verified_checkpoint_snapshot_is_result_exact_and_exposes_root_visits(
    tmp_path,
):
    _root, sender_nodes, _sender_states, _receiver_states = _bluff_tree()
    entries = _multi_root_entries(sender_nodes)
    checkpoint = tmp_path / "verified-snapshot.json"
    result = solve_multi_root_external_sampling_mccfr(
        entries,
        iterations=17,
        seed=20260714,
        max_infosets=20,
        linear_averaging=True,
        checkpoint_path=checkpoint,
    )
    snapshot = verify_multi_root_checkpoint_against_result(
        entries,
        result,
        checkpoint_path=checkpoint,
        expected_checkpoint_sha256=result.metadata["checkpoint_sha256"],
    )

    assert snapshot.checkpoint_sha256 == result.metadata["checkpoint_sha256"]
    assert snapshot.completed_iterations == result.iterations
    assert snapshot.average_strategy == result.average_strategy
    assert snapshot.current_strategy == result.current_strategy
    assert snapshot.cumulative_regret_plus == result.cumulative_regret_plus
    for entry in entries:
        observation = entry.adapter.observation
        assert snapshot.infoset_root_support_opaque_ids[observation] == (
            entry.root_id_sha256,
        )
        assert snapshot.infoset_root_visit_counts_by_opaque_id[observation] == {
            entry.root_id_sha256: result.sampling_stats[
                "root_samples_by_opaque_id"
            ][entry.root_id_sha256]
        }

    with pytest.raises(TypeError):
        snapshot.infoset_root_visit_counts_by_opaque_id[
            entries[0].adapter.observation
        ][entries[0].root_id_sha256] = 0
    with pytest.raises(TypeError):
        snapshot.sampling_stats["root_samples_by_opaque_id"][
            entries[0].root_id_sha256
        ] = 0
    with pytest.raises(
        MultiRootMccfrCheckpointError,
        match="average strategy SHA256 mismatch",
    ):
        verify_multi_root_checkpoint_against_result(
            entries,
            replace(result, average_strategy_sha256="0" * 64),
            checkpoint_path=checkpoint,
            expected_checkpoint_sha256=result.metadata["checkpoint_sha256"],
        )


def test_checkpoint_reader_rejects_duplicate_nan_noncanonical_path_and_tamper(
    tmp_path,
):
    _root, sender_nodes, _sender_states, _receiver_states = _bluff_tree()
    checkpoint = tmp_path / "base.json"
    created = solve_multi_root_external_sampling_mccfr(
        _multi_root_entries(sender_nodes),
        iterations=12,
        seed=41,
        max_infosets=20,
        checkpoint_path=checkpoint,
    )
    expected_sha256 = created.metadata["checkpoint_sha256"]
    assert isinstance(expected_sha256, str)
    raw = checkpoint.read_text(encoding="utf-8")

    with pytest.raises(
        MultiRootMccfrCheckpointError,
        match="external expected_checkpoint_sha256",
    ):
        solve_multi_root_external_sampling_mccfr(
            _multi_root_entries(sender_nodes),
            iterations=1,
            seed=41,
            max_infosets=20,
            resume_from=checkpoint,
        )

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        raw.replace(
            '"sampling_stats":{',
            '"sampling_stats":{},"sampling_stats":{',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(MultiRootMccfrCheckpointError, match="duplicate key"):
        solve_multi_root_external_sampling_mccfr(
            _multi_root_entries(sender_nodes),
            iterations=1,
            seed=41,
            max_infosets=20,
            resume_from=duplicate,
            expected_checkpoint_sha256=expected_sha256,
        )

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text(
        raw.replace('"completed_iterations":12', '"completed_iterations":NaN'),
        encoding="utf-8",
    )
    with pytest.raises(MultiRootMccfrCheckpointError, match="non-finite"):
        solve_multi_root_external_sampling_mccfr(
            _multi_root_entries(sender_nodes),
            iterations=1,
            seed=41,
            max_infosets=20,
            resume_from=nonfinite,
            expected_checkpoint_sha256=expected_sha256,
        )

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(" " + raw, encoding="utf-8")
    with pytest.raises(MultiRootMccfrCheckpointError, match="canonically serialized"):
        solve_multi_root_external_sampling_mccfr(
            _multi_root_entries(sender_nodes),
            iterations=1,
            seed=41,
            max_infosets=20,
            resume_from=noncanonical,
            expected_checkpoint_sha256=expected_sha256,
        )

    tampered = tmp_path / "tampered.json"
    tampered.write_text(
        raw.replace('"seed":41', '"seed":42', 1),
        encoding="utf-8",
    )
    with pytest.raises(MultiRootMccfrCheckpointError, match="SHA-256 mismatch"):
        solve_multi_root_external_sampling_mccfr(
            _multi_root_entries(sender_nodes),
            iterations=1,
            seed=41,
            max_infosets=20,
            resume_from=tampered,
            expected_checkpoint_sha256=expected_sha256,
        )

    traversal_path = tmp_path / "child" / ".." / "escaped.json"
    with pytest.raises(MultiRootMccfrCheckpointError, match="parent traversal"):
        solve_multi_root_external_sampling_mccfr(
            _multi_root_entries(sender_nodes),
            iterations=1,
            seed=41,
            max_infosets=20,
            checkpoint_path=traversal_path,
        )


def test_resume_rejects_live_solver_global_drift_with_trusted_checkpoint_sha(
    monkeypatch,
    tmp_path,
):
    _root, sender_nodes, _sender_states, _receiver_states = _bluff_tree()
    checkpoint = tmp_path / "solver-runtime.json"
    created = solve_multi_root_external_sampling_mccfr(
        _multi_root_entries(sender_nodes),
        iterations=12,
        seed=20260713,
        max_infosets=20,
        checkpoint_path=checkpoint,
    )
    trusted_sha256 = created.metadata["checkpoint_sha256"]
    assert isinstance(trusted_sha256, str)

    monkeypatch.setattr(
        multi_root_module,
        "sample_exact_fraction_index",
        lambda _weights, _rng: 0,
    )
    with pytest.raises(
        MultiRootMccfrCheckpointError,
        match="live runtime globals.*sample_exact_fraction_index",
    ):
        solve_multi_root_external_sampling_mccfr(
            _multi_root_entries(sender_nodes),
            iterations=1,
            seed=20260713,
            max_infosets=20,
            resume_from=checkpoint,
            expected_checkpoint_sha256=trusted_sha256,
        )


def test_resume_preserves_support_when_a_root_first_appears_after_checkpoint(
    tmp_path,
):
    _root, sender_nodes, _sender_states, _receiver_states = _bluff_tree()
    one_shot_path = tmp_path / "support-one-shot.json"
    split_path = tmp_path / "support-split.json"
    one_shot = solve_multi_root_external_sampling_mccfr(
        _multi_root_entries(sender_nodes),
        iterations=3,
        seed=4,
        max_infosets=20,
        linear_averaging=False,
        checkpoint_path=one_shot_path,
    )
    first = solve_multi_root_external_sampling_mccfr(
        _multi_root_entries(sender_nodes),
        iterations=1,
        seed=4,
        max_infosets=20,
        linear_averaging=False,
        checkpoint_path=split_path,
    )
    assert first.sampling_stats["distinct_root_adapters_sampled"] == 1
    resumed = solve_multi_root_external_sampling_mccfr(
        _multi_root_entries(sender_nodes),
        iterations=2,
        seed=4,
        max_infosets=20,
        linear_averaging=False,
        resume_from=split_path,
        checkpoint_path=split_path,
        expected_checkpoint_sha256=first.metadata["checkpoint_sha256"],
    )
    assert resumed.sampling_stats["distinct_root_adapters_sampled"] == 2
    assert resumed.infoset_root_support_count == one_shot.infoset_root_support_count
    assert resumed.shared_across_roots_infosets == one_shot.shared_across_roots_infosets
    assert split_path.read_bytes() == one_shot_path.read_bytes()


def test_rehashed_checkpoint_config_prior_action_support_rng_and_stats_drift_fail(
    tmp_path,
):
    _root, sender_nodes, _sender_states, _receiver_states = _bluff_tree()
    checkpoint = tmp_path / "base.json"
    created = solve_multi_root_external_sampling_mccfr(
        _multi_root_entries(sender_nodes),
        iterations=10,
        seed=43,
        max_infosets=20,
        checkpoint_path=checkpoint,
    )
    original_checkpoint_sha256 = created.metadata["checkpoint_sha256"]
    assert isinstance(original_checkpoint_sha256, str)
    base = json.loads(checkpoint.read_text(encoding="utf-8"))

    cases = {}
    config = copy.deepcopy(base)
    config["payload"]["solver_config"]["max_infosets"] = 21
    cases["config"] = _rehash_checkpoint(config)

    prior = copy.deepcopy(base)
    prior["payload"]["prior_manifest"]["roots"][0][
        "prior_mass_exact"
    ] = "1/3"
    prior["payload"]["prior_manifest_sha256"] = hashlib.sha256(
        json.dumps(
            prior["payload"]["prior_manifest"],
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    cases["prior"] = _rehash_checkpoint(prior)

    action = copy.deepcopy(base)
    row = action["payload"]["tables"][0]
    row["stable_action_ids"] = sorted(
        ["tampered-action", *row["stable_action_ids"][1:]]
    )
    cases["action"] = _rehash_checkpoint(action)

    support = copy.deepcopy(base)
    support["payload"]["tables"][0]["root_support_opaque_ids"] = ["0" * 64]
    cases["support"] = _rehash_checkpoint(support)

    rng = copy.deepcopy(base)
    rng["payload"]["rng_state"]["algorithm"] = "wrong_rng"
    cases["rng"] = _rehash_checkpoint(rng)

    stats = copy.deepcopy(base)
    stats["payload"]["sampling_stats"]["super_root_samples"] += 1
    cases["stats"] = _rehash_checkpoint(stats)

    invalid_float = copy.deepcopy(base)
    invalid_float["payload"]["tables"][0]["regret_plus_hex"][0] = "nan"
    cases["invalid-float"] = _rehash_checkpoint(invalid_float)

    source = copy.deepcopy(base)
    source["payload"]["source_binding"]["rules_contract"][
        "physical_joker_ids"
    ] = ["X1"]
    source["payload"]["source_binding_sha256"] = hashlib.sha256(
        json.dumps(
            source["payload"]["source_binding"],
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    cases["source-rules"] = _rehash_checkpoint(source)

    for label, envelope in cases.items():
        target = tmp_path / f"{label}.json"
        _write_checkpoint_envelope(target, envelope)
        with pytest.raises(MultiRootMccfrCheckpointError):
            solve_multi_root_external_sampling_mccfr(
                _multi_root_entries(sender_nodes),
                iterations=1,
                seed=43,
                max_infosets=20,
                resume_from=target,
                # Supply the mutated envelope digest here so this test reaches
                # the semantic compatibility checks beneath the external
                # production trust anchor.
                expected_checkpoint_sha256=envelope["checkpoint_sha256"],
            )

        with pytest.raises(
            MultiRootMccfrCheckpointError,
            match="external expected_checkpoint_sha256",
        ):
            solve_multi_root_external_sampling_mccfr(
                _multi_root_entries(sender_nodes),
                iterations=1,
                seed=43,
                max_infosets=20,
                resume_from=target,
                expected_checkpoint_sha256=original_checkpoint_sha256,
            )


def test_rehashed_support_and_stats_edits_fail_internal_and_external_bindings(
    tmp_path,
):
    _root, sender_nodes, _sender_states, _receiver_states = _bluff_tree()
    checkpoint = tmp_path / "base-accounting.json"
    created = solve_multi_root_external_sampling_mccfr(
        _multi_root_entries(sender_nodes),
        iterations=40,
        seed=20260713,
        max_infosets=20,
        checkpoint_path=checkpoint,
    )
    trusted_sha256 = created.metadata["checkpoint_sha256"]
    assert isinstance(trusted_sha256, str)
    base = json.loads(checkpoint.read_text(encoding="utf-8"))

    support = copy.deepcopy(base)
    shared_row = next(
        row
        for row in support["payload"]["tables"]
        if len(row["root_support_opaque_ids"]) > 1
    )
    shared_row["root_support_opaque_ids"].pop()
    support = _rehash_runtime_state_and_checkpoint(support)

    stats = copy.deepcopy(base)
    stats["payload"]["sampling_stats"]["decision_visits"] += 123_456
    stats = _rehash_runtime_state_and_checkpoint(stats)

    for label, envelope, semantic_error in (
        ("support", support, "support visit counts"),
        ("stats", stats, "statistics are inconsistent"),
    ):
        target = tmp_path / f"accounting-{label}.json"
        _write_checkpoint_envelope(target, envelope)

        # Even when the caller elects to trust the attacker's new digest, the
        # redundant visit accounting catches these simple semantic edits.
        with pytest.raises(MultiRootMccfrCheckpointError, match=semantic_error):
            solve_multi_root_external_sampling_mccfr(
                _multi_root_entries(sender_nodes),
                iterations=1,
                seed=20260713,
                max_infosets=20,
                resume_from=target,
                expected_checkpoint_sha256=envelope["checkpoint_sha256"],
            )

        # Production resume is anchored to the digest recorded outside the
        # checkpoint, so any coordinated rehash still fails closed.
        with pytest.raises(
            MultiRootMccfrCheckpointError,
            match="external expected_checkpoint_sha256",
        ):
            solve_multi_root_external_sampling_mccfr(
                _multi_root_entries(sender_nodes),
                iterations=1,
                seed=20260713,
                max_infosets=20,
                resume_from=target,
                expected_checkpoint_sha256=trusted_sha256,
            )


def test_resume_rejects_live_prior_root_and_canonical_adapter_drift(
    monkeypatch,
    tmp_path,
):
    _root, sender_nodes, _sender_states, _receiver_states = _bluff_tree()
    checkpoint = tmp_path / "base.json"
    created = solve_multi_root_external_sampling_mccfr(
        _multi_root_entries(sender_nodes),
        iterations=10,
        seed=47,
        max_infosets=20,
        checkpoint_path=checkpoint,
    )
    expected_checkpoint_sha256 = created.metadata["checkpoint_sha256"]
    assert isinstance(expected_checkpoint_sha256, str)

    changed_prior = (
        MultiRootChanceEntry(
            "private-type-strong",
            CanonicalReducedPublicTreeAdapter(
                sender_nodes["strong"], "strong-physical"
            ),
            Fraction(1, 3),
        ),
        MultiRootChanceEntry(
            "private-type-weak",
            CanonicalReducedPublicTreeAdapter(
                sender_nodes["weak"], "weak-physical"
            ),
            Fraction(2, 3),
        ),
    )
    with pytest.raises(MultiRootMccfrCheckpointError, match="prior"):
        solve_multi_root_external_sampling_mccfr(
            changed_prior,
            iterations=1,
            seed=47,
            max_infosets=20,
            resume_from=checkpoint,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
        )

    with pytest.raises(MultiRootMccfrCheckpointError, match="seed"):
        solve_multi_root_external_sampling_mccfr(
            _multi_root_entries(sender_nodes),
            iterations=1,
            seed=48,
            max_infosets=20,
            resume_from=checkpoint,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
        )
    with pytest.raises(MultiRootMccfrCheckpointError, match="linear_averaging"):
        solve_multi_root_external_sampling_mccfr(
            _multi_root_entries(sender_nodes),
            iterations=1,
            seed=47,
            max_infosets=20,
            linear_averaging=False,
            resume_from=checkpoint,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
        )

    changed_root_id = (
        MultiRootChanceEntry(
            "private-type-strong-renamed",
            CanonicalReducedPublicTreeAdapter(
                sender_nodes["strong"], "strong-physical"
            ),
            Fraction(1, 2),
        ),
        MultiRootChanceEntry(
            "private-type-weak",
            CanonicalReducedPublicTreeAdapter(
                sender_nodes["weak"], "weak-physical"
            ),
            Fraction(1, 2),
        ),
    )
    with pytest.raises(MultiRootMccfrCheckpointError, match="prior"):
        solve_multi_root_external_sampling_mccfr(
            changed_root_id,
            iterations=1,
            seed=47,
            max_infosets=20,
            resume_from=checkpoint,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
        )

    instance_shadowed = _multi_root_entries(sender_nodes)
    instance_shadowed[0].adapter.terminal_utility_bb = lambda _terminal: 999.0
    with pytest.raises(
        MultiRootMccfrCheckpointError,
        match="canonical reduced.*terminal_utility_bb",
    ):
        solve_multi_root_external_sampling_mccfr(
            instance_shadowed,
            iterations=1,
            seed=47,
            max_infosets=20,
            resume_from=checkpoint,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
        )

    with monkeypatch.context() as patch:
        patch.setattr(
            CanonicalReducedPublicTreeAdapter,
            "terminal_utility_bb",
            staticmethod(lambda _terminal: 999.0),
        )
        with pytest.raises(MultiRootMccfrCheckpointError, match="canonical reduced"):
            solve_multi_root_external_sampling_mccfr(
                _multi_root_entries(sender_nodes),
                iterations=1,
                seed=47,
                max_infosets=20,
                resume_from=checkpoint,
                expected_checkpoint_sha256=expected_checkpoint_sha256,
            )

    for method_name, replacement in (
        ("sample_root_for_traversal", lambda _self, _rng: None),
        ("apply_action_id", staticmethod(lambda _state, _action_id: None)),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(
                CanonicalReducedPublicTreeAdapter,
                method_name,
                replacement,
            )
            with pytest.raises(
                MultiRootMccfrCheckpointError,
                match="canonical reduced",
            ):
                solve_multi_root_external_sampling_mccfr(
                    _multi_root_entries(sender_nodes),
                    iterations=1,
                    seed=47,
                    max_infosets=20,
                    resume_from=checkpoint,
                    expected_checkpoint_sha256=expected_checkpoint_sha256,
                )

    with monkeypatch.context() as patch:
        patch.setattr(
            multi_root_module,
            "_CanonicalReducedTransition",
            _ExplicitTransition,
        )
        with pytest.raises(
            MultiRootMccfrCheckpointError,
            match="canonical reduced",
        ):
            solve_multi_root_external_sampling_mccfr(
                _multi_root_entries(sender_nodes),
                iterations=1,
                seed=47,
                max_infosets=20,
                resume_from=checkpoint,
                expected_checkpoint_sha256=expected_checkpoint_sha256,
            )

    original_action_ids = PublicTreeDecisionNode.action_ids.fget
    assert original_action_ids is not None
    with monkeypatch.context() as patch:
        patch.setattr(
            PublicTreeDecisionNode,
            "action_ids",
            property(lambda self: original_action_ids(self)),
        )
        with pytest.raises(MultiRootMccfrCheckpointError, match="adapter"):
            solve_multi_root_external_sampling_mccfr(
                _multi_root_entries(sender_nodes),
                iterations=1,
                seed=47,
                max_infosets=20,
                resume_from=checkpoint,
                expected_checkpoint_sha256=expected_checkpoint_sha256,
            )


def test_arbitrary_generic_adapter_checkpoint_is_disabled_even_after_global_drift(
    monkeypatch,
    tmp_path,
):
    _root, sender_nodes, _sender_states, _receiver_states = _bluff_tree()
    generic_entries = _generic_multi_root_entries(sender_nodes)
    # Arbitrary adapters remain valid for disposable non-checkpoint solves.
    result = solve_multi_root_external_sampling_mccfr(
        generic_entries,
        iterations=1,
        seed=47,
        max_infosets=20,
    )
    assert result.iterations == 1

    @dataclass(frozen=True)
    class _DriftedTransition:
        state: PublicTreeDecisionNode
        next_phase: str = "attacker-controlled-phase"

    monkeypatch.setitem(
        _ExplicitBranchAdapter.sample_next_draw.__globals__,
        "_ExplicitTransition",
        _DriftedTransition,
    )
    with pytest.raises(
        MultiRootMccfrCheckpointError,
        match="disabled for arbitrary generic adapters",
    ):
        solve_multi_root_external_sampling_mccfr(
            _generic_multi_root_entries(sender_nodes),
            iterations=1,
            seed=47,
            max_infosets=20,
            checkpoint_path=tmp_path / "generic.json",
        )


def test_atomic_checkpoint_failure_preserves_previous_file_and_cleans_temp(
    monkeypatch,
    tmp_path,
):
    _root, sender_nodes, _sender_states, _receiver_states = _bluff_tree()
    checkpoint = tmp_path / "atomic.json"
    solve_multi_root_external_sampling_mccfr(
        _multi_root_entries(sender_nodes),
        iterations=5,
        seed=53,
        max_infosets=20,
        checkpoint_path=checkpoint,
    )
    before = checkpoint.read_bytes()

    def fail_replace(_source, _target):
        raise OSError("injected atomic replace failure")

    monkeypatch.setattr(multi_root_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected atomic replace failure"):
        solve_multi_root_external_sampling_mccfr(
            _multi_root_entries(sender_nodes),
            iterations=6,
            seed=53,
            max_infosets=20,
            checkpoint_path=checkpoint,
        )
    assert checkpoint.read_bytes() == before
    assert list(tmp_path.glob(f".{checkpoint.name}.*.tmp")) == []


def test_real_full_card_adapter_checkpoint_binding_covers_ranges_and_sources(
    monkeypatch,
):
    observation_a = _full_observation("6s")
    observation_b = _full_observation("5s")
    range_a = _single_world_range(monkeypatch, observation_a)
    range_b = _single_world_range(monkeypatch, observation_b)
    adapter_a = FullCardGenerativeAdapter(observation_a, range_a)
    adapter_b = FullCardGenerativeAdapter(observation_b, range_b)
    entries = (
        MultiRootChanceEntry("full-a", adapter_a, Fraction(1, 2)),
        MultiRootChanceEntry("full-b", adapter_b, Fraction(1, 2)),
    )

    bindings = multi_root_module._checkpoint_adapter_bindings(
        multi_root_module._validated_entries(entries)
    )
    assert all(
        binding["adapter_kind"] == "full_card_generative_t3_t4_v1"
        for binding in bindings
    )
    assert all(
        len(
            binding["live_runtime_semantic_binding"]["binding_sha256"]
        )
        == 64
        for binding in bindings
    )
    assert bindings[0]["range_content_sha256"] in {
        range_a.range_content_sha256,
        range_b.range_content_sha256,
    }
    assert {binding["range_build_sha256"] for binding in bindings} == {
        range_a.range_build_sha256,
        range_b.range_build_sha256,
    }
    source = multi_root_module._multi_root_source_binding()
    required_sources = {
        "ai/tutor/t3_hu_multi_root_mccfr.py",
        "ai/tutor/t3_hu_full_card_mccfr.py",
        "ai/tutor/t3_hu_full_card_range.py",
        "ai/engine/action_space.py",
        "ai/engine/encoding.py",
        "ai/engine/game_engine.py",
        "ai/engine/scoring.py",
        "ai/engine/turn_order.py",
        "ai/config/fl_ev.json",
    }
    assert required_sources.issubset(source["source_sha256"])
    assert source["rules_contract"]["physical_joker_ids"] == ["X1", "X2"]
    assert source["rules_contract"]["position_contract_version"] == (
        POSITION_CONTRACT_VERSION
    )

    # The sampled-support content may be identical, but a different range
    # build seed is still a distinct production input and must alter binding.
    range_b_drift = build_history_weighted_full_card_range(
        observation_b,
        UniformLegalBehaviorModel(),
        epsilon=0,
        max_particles=1,
        seed=32,
    )
    drift_bindings = multi_root_module._checkpoint_adapter_bindings(
        multi_root_module._validated_entries(
            (
                MultiRootChanceEntry(
                    "full-a",
                    FullCardGenerativeAdapter(observation_a, range_a),
                    Fraction(1, 2),
                ),
                MultiRootChanceEntry(
                    "full-b",
                    FullCardGenerativeAdapter(observation_b, range_b_drift),
                    Fraction(1, 2),
                ),
            )
        )
    )
    assert range_b_drift.range_content_sha256 == range_b.range_content_sha256
    assert range_b_drift.range_build_sha256 != range_b.range_build_sha256
    assert drift_bindings != bindings

    # A subclass with changed transition semantics cannot inherit the canonical
    # FullCard binding, and arbitrary generic checkpointing is disabled.
    forced = _ForcedSharedDrawAdapter(
        observation_a, range_a, ("2h", "3h", "5h")
    )
    with pytest.raises(
        MultiRootMccfrCheckpointError,
        match="disabled for arbitrary generic adapters",
    ):
        multi_root_module._checkpoint_adapter_bindings(
            multi_root_module._validated_entries(
                (
                    MultiRootChanceEntry("forced", forced, Fraction(1, 2)),
                    MultiRootChanceEntry("full", adapter_b, Fraction(1, 2)),
                )
            )
        )

    with monkeypatch.context() as patch:
        patch.setattr(
            FullCardGenerativeAdapter,
            "terminal_utility_bb",
            staticmethod(lambda _terminal: 0.0),
        )
        with pytest.raises(
            MultiRootMccfrCheckpointError, match="overridden methods"
        ):
            multi_root_module._checkpoint_adapter_bindings(
                multi_root_module._validated_entries(entries)
            )

    with monkeypatch.context() as patch:
        patch.setattr(
            multi_root_module._full_card_module,
            "get_turn_actions",
            lambda _cards, _board: [],
        )
        with pytest.raises(
            MultiRootMccfrCheckpointError, match="overridden methods"
        ):
            multi_root_module._checkpoint_adapter_bindings(
                multi_root_module._validated_entries(entries)
            )

    with monkeypatch.context() as patch:
        patch.setattr(
            multi_root_module._exact_late_module,
            "_score_against_complete_opponent",
            lambda _board, _opponent, _include_fl_ev: 999.0,
        )
        with pytest.raises(
            MultiRootMccfrCheckpointError, match="overridden methods"
        ):
            multi_root_module._checkpoint_adapter_bindings(
                multi_root_module._validated_entries(entries)
            )

    # The utility wrapper delegates through this helper.  Binding only the
    # wrapper would leave the actual score calculation replaceable.
    with monkeypatch.context() as patch:
        patch.setattr(
            FullCardGenerativeAdapter,
            "terminal_metrics_bb",
            staticmethod(lambda _terminal: {"score": 0}),
        )
        with pytest.raises(
            MultiRootMccfrCheckpointError, match="overridden methods"
        ):
            multi_root_module._checkpoint_adapter_bindings(
                multi_root_module._validated_entries(entries)
            )

    with monkeypatch.context() as patch:
        patch.setattr(
            multi_root_module._full_card_module,
            "terminal_metrics",
            lambda _bb, _btn: {"score": 0},
        )
        with pytest.raises(
            MultiRootMccfrCheckpointError, match="overridden methods"
        ):
            multi_root_module._checkpoint_adapter_bindings(
                multi_root_module._validated_entries(entries)
            )
