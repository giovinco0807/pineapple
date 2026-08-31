import hashlib
import itertools
import json
import math
import random
from collections import Counter
from fractions import Fraction

import pytest

import ai.tutor.t3_hu_full_card_range as range_module
import ai.tutor.t3_hu_full_card_mccfr as full_mccfr_module
from ai.engine.action_space import get_turn_actions
from ai.engine.encoding import ALL_CARDS, Board
from ai.engine.turn_order import POSITION_CONTRACT_VERSION
from ai.tutor.exact_late import action_key, terminal_metrics
from ai.tutor.t3_hu_full_card_mccfr import (
    DRAW_SAMPLING_CONTRACT,
    ROOT_SAMPLING_CONTRACT,
    FullCardGenerativeAdapter,
    sample_exact_fraction_index,
    sample_uniform_three_cards,
    serialize_strategy_profile,
    solve_full_card_external_sampling_mccfr,
)
from ai.tutor.t3_hu_full_card_range import (
    BehaviorDistribution,
    UniformLegalBehaviorModel,
    build_history_weighted_full_card_range,
    verify_full_card_range,
)
from ai.tutor.t3_hu_public_cfr import InfoSetKey, JointParticle, PrivateRecall
from ai.tutor.t3_hu_public_tree import (
    PendingChanceState,
    PublicTreeDecisionState,
    PublicTreeTerminalState,
    apply_public_tree_action,
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


def _canonical_sha256(value) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _recall(actor: str, discard_t1: str, discard_t2: str) -> PrivateRecall:
    placements = {
        "bb": (("7d", "8h"), ("9s", "Tc")),
        "btn": (("9d", "Jh"), ("Qs", "Kc")),
    }[actor]
    return PrivateRecall(
        dealt_by_turn=(
            (1, (*placements[0], discard_t1)),
            (2, (*placements[1], discard_t2)),
        ),
        discards_by_turn=((1, discard_t1), (2, discard_t2)),
    )


def _observation() -> InfoSetKey:
    # The small remainder only supplies a compatible physical witness for the
    # information key.  The range builder reconstructs complete 54-card worlds.
    witness = JointParticle(
        bb_recall=_recall("bb", "6c", "6s"),
        btn_recall=_recall("btn", "4s", "7h"),
        undealt_cards=("X1", "X2", "Ac"),
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


def _board(rows) -> Board:
    return Board(top=list(rows[0]), middle=list(rows[1]), bottom=list(rows[2]))


class _FixedBits:
    def __init__(self, *values: int) -> None:
        self.values = list(values)

    def getrandbits(self, k: int) -> int:
        if not self.values:
            raise AssertionError("fixed RNG was asked for an unexpected value")
        value = self.values.pop(0)
        assert 0 <= value < (1 << k)
        return value


class _FirstTurnJokerWeightedBehavior:
    model_id = "test_first_turn_joker_weighted_v1"

    @property
    def model_manifest(self):
        return {
            "schema": "ofc_frozen_behavior_model/v1",
            "model_id": self.model_id,
            "model_type": "test_exact_distribution",
            "promotion_eligible": False,
            "position_contract_version": POSITION_CONTRACT_VERSION,
        }

    @property
    def model_sha256(self) -> str:
        return _canonical_sha256(self.model_manifest)

    def action_distribution(self, information):
        public_placements = {
            (1, "btn"): (("9d", "middle"), ("Jh", "middle")),
            (2, "btn"): (("Qs", "middle"), ("Kc", "middle")),
        }[(information.turn, information.actor)]
        public_cards = {card for card, _row in public_placements}
        discard = next(card for card in information.current_draw if card not in public_cards)
        target = None
        for stable_id in information.legal_action_ids:
            payload = json.loads(stable_id)
            placements = {
                (card, row) for card, row in payload["placements"]
            }
            if payload["discard"] == discard and placements == set(public_placements):
                target = stable_id
                break
        if target is None:  # pragma: no cover - fixture construction defense
            raise AssertionError("observed public action was not legal")

        if information.turn == 1:
            target_probability = (
                Fraction(3, 4) if discard == "X1" else Fraction(1, 4)
            )
        else:
            target_probability = Fraction(1, 2)
        others = [
            stable_id
            for stable_id in information.legal_action_ids
            if stable_id != target
        ]
        remainder = (1 - target_probability) / len(others)
        probabilities = {
            stable_id: (
                target_probability if stable_id == target else remainder
            )
            for stable_id in information.legal_action_ids
        }
        return BehaviorDistribution(
            information_digest=information.digest(),
            probabilities=probabilities,
            source="model",
            used_fallback=False,
        )


def _two_world_weighted_range(monkeypatch):
    observation = _observation()

    def select_assignments(
        pool, count, *, max_particles, seed, observation_digest
    ):
        assert count == 2
        assert {"X1", "X2"}.issubset(pool)
        return (("X1", "X2"), ("X2", "X1")), math.perm(len(pool), count), False

    monkeypatch.setattr(range_module, "_select_assignments", select_assignments)
    result = build_history_weighted_full_card_range(
        observation,
        _FirstTurnJokerWeightedBehavior(),
        epsilon=0,
        max_particles=2,
        seed=17,
    )
    assert verify_full_card_range(observation, result)["verified"] is True
    assert {particle.weight for particle in result.particles} == {
        Fraction(1, 4),
        Fraction(3, 4),
    }
    return observation, result


def _one_world_natural_range(monkeypatch):
    observation = _observation()

    def select_assignment(
        pool, count, *, max_particles, seed, observation_digest
    ):
        assignment = tuple(
            card for card in pool if card not in ("X1", "X2")
        )[:count]
        assert len(assignment) == count == 2
        return (assignment,), math.perm(len(pool), count), False

    monkeypatch.setattr(range_module, "_select_assignments", select_assignment)
    result = build_history_weighted_full_card_range(
        observation,
        UniformLegalBehaviorModel(),
        epsilon=0,
        max_particles=1,
        seed=23,
    )
    assert result.particles[0].weight == 1
    assert {"X1", "X2"}.issubset(result.particles[0].undealt_cards)
    assert verify_full_card_range(observation, result)["verified"] is True
    return observation, result


def _unit_state(observation: InfoSetKey, particle: JointParticle):
    return PublicTreeDecisionState(
        infoset_key=observation,
        particle=JointParticle(
            bb_recall=particle.bb_recall,
            btn_recall=particle.btn_recall,
            undealt_cards=particle.undealt_cards,
            weight=1,
        ),
    )


def _install_light_terminal(monkeypatch, *adapters):
    """Use a cheap deterministic public-board value in iterative solver tests."""

    card_value = {card: index + 1 for index, card in enumerate(ALL_CARDS)}
    seen_weights = []

    def light_terminal(terminal):
        seen_weights.append(terminal.particle.weight)

        def board_value(rows):
            return sum(
                (row_index + 1) * card_value[card]
                for row_index, row in enumerate(rows)
                for card in row
            )

        return float(board_value(terminal.board_bb) - board_value(terminal.board_btn))

    for adapter in adapters:
        monkeypatch.setattr(adapter, "terminal_utility_bb", light_terminal)
    return seen_weights


def _write_checkpoint_json(path, envelope):
    path.write_text(
        json.dumps(
            envelope,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def _rehash_checkpoint_envelope(envelope):
    manifest = envelope["payload"]["adapter_manifest"]
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    envelope["payload"]["adapter_manifest_sha256"] = _canonical_sha256(
        manifest
    )
    envelope["checkpoint_sha256"] = _canonical_sha256(envelope["payload"])
    return envelope


def test_exact_fraction_and_unordered_combination_samplers_are_integer_exact():
    weights = (Fraction(1, 4), Fraction(3, 4))
    assert sample_exact_fraction_index(weights, _FixedBits(0)) == 0
    assert sample_exact_fraction_index(weights, _FixedBits(1)) == 1
    with pytest.raises(TypeError, match="fractions.Fraction"):
        sample_exact_fraction_index((0.25, 0.75), _FixedBits(0))

    cards = ("2c", "3c", "4c", "X1", "X2")
    expected = tuple(itertools.combinations(sorted(cards), 3))
    sampled = tuple(
        sample_uniform_three_cards(cards, _FixedBits(rank))
        for rank in range(len(expected))
    )
    assert tuple(item.cards for item in sampled) == expected
    assert all(item.combination_count == 10 for item in sampled)
    assert all(item.probability == Fraction(1, 10) for item in sampled)
    assert tuple(item.sampled_rank for item in sampled) == tuple(range(10))


def test_root_posterior_is_stable_sampled_once_and_then_forced_to_unit_weight(
    monkeypatch,
):
    observation, root_range = _two_world_weighted_range(monkeypatch)
    first = FullCardGenerativeAdapter(observation, root_range)
    replay = FullCardGenerativeAdapter(observation, root_range)

    commitments = tuple(commitment for commitment, _weight in first.root_distribution)
    assert commitments == tuple(sorted(commitments))
    assert {weight for _commitment, weight in first.root_distribution} == {
        Fraction(1, 4),
        Fraction(3, 4),
    }

    first_rng = random.Random(20260713)
    replay_rng = random.Random(20260713)
    first_samples = tuple(
        first.sample_root_for_traversal(first_rng) for _ in range(64)
    )
    replay_samples = tuple(
        replay.sample_root_for_traversal(replay_rng) for _ in range(64)
    )
    assert tuple(sample.particle_commitment for sample in first_samples) == tuple(
        sample.particle_commitment for sample in replay_samples
    )
    assert tuple(sample.posterior_probability for sample in first_samples) == tuple(
        sample.posterior_probability for sample in replay_samples
    )
    assert all(sample.state.particle.weight == 1 for sample in first_samples)
    assert all(sample.state.infoset_key == observation for sample in first_samples)
    assert all(
        sample.posterior_probability_applied_to_state is False
        for sample in first_samples
    )

    audit = first.sampling_audit()
    assert audit["traversals"] == audit["root_posterior_samples"] == 64
    assert audit["root_outcome_count_total"] == 64
    assert audit["root_distinct_outcomes_sampled"] == 2
    assert audit["root_sampling_contract"] == ROOT_SAMPLING_CONTRACT
    assert audit["posterior_probability_multiplied_after_sampling"] is False
    assert audit["chance_probability_multiplied_after_sampling"] is False
    assert audit["joint_particle_weight_after_sampling"] == "1/1"
    assert first.metadata["joint_particle_weight_used_after_sampling"] is False

    # Passing an unsampled posterior particle to the decision boundary fails
    # before an action or chance probability can be counted a second time.
    weighted_state = PublicTreeDecisionState(
        infoset_key=observation,
        particle=root_range.particles[0],
    )
    with pytest.raises(ValueError, match="requires unit particle weight"):
        first.legal_actions(weighted_state)
    direct_action = get_turn_actions(
        list(observation.current_draw), _board(observation.board_bb)
    )[0]
    weighted_pending = apply_public_tree_action(weighted_state, direct_action)
    with pytest.raises(ValueError, match="requires unit particle weight"):
        first.sample_next_draw(weighted_pending, random.Random(1))


def test_hidden_opponent_discards_never_change_policy_key_or_action_ids(monkeypatch):
    observation, root_range = _two_world_weighted_range(monkeypatch)
    adapter = FullCardGenerativeAdapter(observation, root_range)
    first_particle, second_particle = root_range.particles

    assert first_particle.btn_recall != second_particle.btn_recall
    # The same two Joker atoms are hidden on opposite private turns, so the
    # remainder is intentionally identical and only opponent recall differs.
    assert first_particle.undealt_cards == second_particle.undealt_cards
    first_state = _unit_state(observation, first_particle)
    second_state = _unit_state(observation, second_particle)
    assert adapter.information_key(first_state) == adapter.information_key(second_state)
    assert first_state.infoset_key.digest() == second_state.infoset_key.digest()
    serialized = first_state.infoset_key.canonical_json()
    assert "X1" not in serialized and "X2" not in serialized

    first_actions = adapter.legal_actions(first_state)
    second_actions = adapter.legal_actions(second_state)
    first_ids = tuple(stable_id for stable_id, _action in first_actions)
    second_ids = tuple(stable_id for stable_id, _action in second_actions)
    assert first_ids == second_ids == tuple(sorted(first_ids))
    assert len(first_ids) == len(set(first_ids))
    assert all(action_key(action) == stable_id for stable_id, action in first_actions)
    assert adapter.metadata["policy_identity_contract"].startswith("infoset_key")


def test_forced_x1_x2_draw_stays_physically_distinct_in_state_and_action_ids(
    monkeypatch,
):
    observation, root_range = _one_world_natural_range(monkeypatch)
    adapter = FullCardGenerativeAdapter(observation, root_range)
    root = adapter.sample_root_for_traversal(random.Random(5))
    first_action_id = adapter.legal_actions(root.state)[0][0]
    pending = adapter.apply_action_id(root.state, first_action_id)
    assert isinstance(pending, PendingChanceState)
    assert {"X1", "X2"}.issubset(pending.remaining_cards)

    natural = next(
        card for card in pending.remaining_cards if card not in ("X1", "X2")
    )
    target = tuple(sorted((natural, "X1", "X2")))
    combinations = tuple(itertools.combinations(sorted(pending.remaining_cards), 3))
    rank = combinations.index(target)
    transition = adapter.sample_next_draw(pending, _FixedBits(rank))
    assert transition.draw.cards == target
    assert transition.draw.probability == Fraction(1, math.comb(29, 3))
    assert {"X1", "X2"}.issubset(transition.state.infoset_key.current_draw)
    assert "X1" not in transition.state.remaining_cards
    assert "X2" not in transition.state.remaining_cards
    assert transition.state.particle.weight == 1
    assert transition.chance_probability_applied_to_state is False

    action_ids = tuple(
        stable_id for stable_id, _action in adapter.legal_actions(transition.state)
    )
    discards = {json.loads(stable_id)["discard"] for stable_id in action_ids}
    assert {"X1", "X2"}.issubset(discards)
    assert any('"X1"' in stable_id for stable_id in action_ids)
    assert any('"X2"' in stable_id for stable_id in action_ids)
    assert len(action_ids) == len(set(action_ids))


def _run_complete_sampled_path(adapter, seed: int):
    rng = random.Random(seed)
    root = adapter.sample_root_for_traversal(rng)
    state = root.state
    remaining_counts = [len(state.remaining_cards)]
    draws = []
    phases = [state.infoset_key.phase]
    while True:
        stable_action_id = adapter.legal_actions(state)[0][0]
        result = adapter.apply_action_id(state, stable_action_id)
        if isinstance(result, PublicTreeTerminalState):
            return root, tuple(phases), tuple(remaining_counts), tuple(draws), result
        assert isinstance(result, PendingChanceState)
        transition = adapter.sample_next_draw(result, rng)
        draws.append(transition)
        state = transition.state
        phases.append(state.infoset_key.phase)
        remaining_counts.append(len(state.remaining_cards))


def test_complete_physical_path_is_29_26_23_20_and_terminal_score_is_canonical(
    monkeypatch,
):
    observation, root_range = _one_world_natural_range(monkeypatch)
    adapter = FullCardGenerativeAdapter(observation, root_range)
    root, phases, counts, draws, terminal = _run_complete_sampled_path(
        adapter, 20260713
    )

    assert phases == ("t3_first", "t3_second", "t4_first", "t4_second")
    assert counts == (29, 26, 23, 20)
    assert tuple(draw.next_phase for draw in draws) == (
        "t3_second",
        "t4_first",
        "t4_second",
    )
    for draw in draws:
        assert draw.before_remaining_count - draw.after_remaining_count == 3
        assert draw.draw.combination_count == math.comb(
            draw.before_remaining_count, 3
        )
        assert draw.draw.probability == Fraction(
            1, draw.draw.combination_count
        )
        assert draw.state.particle.weight == 1
        assert draw.chance_probability_applied_to_state is False
    assert root.state.particle.weight == terminal.particle.weight == 1
    assert len(terminal.remaining_cards) == 20

    public_cards = [
        card
        for _turn, _actor, placements in terminal.public_action_history
        for card, _row in placements
    ]
    private_discards = [
        card
        for recall in (terminal.particle.bb_recall, terminal.particle.btn_recall)
        for _turn, card in recall.discards_by_turn
    ]
    assert Counter((*public_cards, *private_discards, *terminal.remaining_cards)) == Counter(
        ALL_CARDS
    )
    assert (
        public_cards + private_discards + list(terminal.remaining_cards)
    ).count("X1") == 1
    assert (
        public_cards + private_discards + list(terminal.remaining_cards)
    ).count("X2") == 1

    expected = terminal_metrics(
        _board(terminal.board_bb),
        _board(terminal.board_btn),
    )
    actual = adapter.terminal_metrics_bb(terminal)
    assert dict(actual) == expected
    assert adapter.terminal_utility_bb(terminal) == expected["score"]
    audit = adapter.sampling_audit()
    assert audit["future_draw_samples"] == 3
    assert audit["future_draw_samples_by_next_phase"] == {
        "t3_second": 1,
        "t4_first": 1,
        "t4_second": 1,
    }
    assert audit["draw_sampling_contract"] == DRAW_SAMPLING_CONTRACT


def test_seed_replays_root_and_every_physical_draw(monkeypatch):
    observation, root_range = _one_world_natural_range(monkeypatch)
    first = FullCardGenerativeAdapter(observation, root_range)
    replay = FullCardGenerativeAdapter(observation, root_range)
    other = FullCardGenerativeAdapter(observation, root_range)

    first_run = _run_complete_sampled_path(first, 991)
    replay_run = _run_complete_sampled_path(replay, 991)
    other_run = _run_complete_sampled_path(other, 992)

    def trace(run):
        root, phases, counts, draws, terminal = run
        return (
            root.particle_commitment,
            phases,
            counts,
            tuple(draw.draw.cards for draw in draws),
            terminal.public_action_history,
            terminal.particle.bb_recall,
            terminal.particle.btn_recall,
        )

    assert trace(first_run) == trace(replay_run)
    assert trace(first_run) != trace(other_run)
    assert first.sampling_audit()["root_posterior_samples"] == 1
    assert replay.sampling_audit()["root_posterior_samples"] == 1


def test_opponent_action_cache_is_one_choice_per_infoset_not_per_hidden_world(
    monkeypatch,
):
    observation, root_range = _two_world_weighted_range(monkeypatch)
    adapter = FullCardGenerativeAdapter(observation, root_range)
    first_state = _unit_state(observation, root_range.particles[0])
    second_state = _unit_state(observation, root_range.particles[1])
    assert first_state.infoset_key == second_state.infoset_key
    action_ids = tuple(
        stable_id for stable_id, _action in adapter.legal_actions(first_state)
    )
    sigma = {action_id: 1.0 / len(action_ids) for action_id in action_ids}
    cache = {}
    rng = random.Random(77)

    first_choice, first_hit = full_mccfr_module._sample_cached_opponent_action(
        first_state.infoset_key,
        action_ids,
        sigma,
        rng,
        cache,
    )
    second_choice, second_hit = full_mccfr_module._sample_cached_opponent_action(
        second_state.infoset_key,
        action_ids,
        sigma,
        rng,
        cache,
    )
    assert first_hit is False
    assert second_hit is True
    assert first_choice == second_choice
    assert tuple(cache) == (observation,)
    assert all(isinstance(key, InfoSetKey) for key in cache)


def test_dynamic_table_starts_uniform_and_rejects_shared_key_action_drift(
    monkeypatch,
):
    observation, root_range = _one_world_natural_range(monkeypatch)
    adapter = FullCardGenerativeAdapter(observation, root_range)
    state = adapter.sample_root_for_traversal(random.Random(1)).state
    action_ids = tuple(
        stable_id for stable_id, _action in adapter.legal_actions(state)
    )
    tables = full_mccfr_module._DynamicMccfrTables(max_infosets=10)
    assert tables.ensure(observation, action_ids) is True
    assert tables.ensure(observation, action_ids) is False
    assert tables.current_strategy(observation) == {
        action_id: 1.0 / len(action_ids) for action_id in action_ids
    }
    with pytest.raises(ValueError, match="action-set mismatch"):
        tables.ensure(observation, action_ids[:-1])
    with pytest.raises(ValueError, match="lexically stable"):
        tables.ensure(observation, tuple(reversed(action_ids)))


def test_dynamic_full_card_mccfr_smoke_is_physical_and_has_no_exact_claim(
    monkeypatch,
):
    observation, root_range = _one_world_natural_range(monkeypatch)
    adapter = FullCardGenerativeAdapter(observation, root_range)
    terminal_weights = _install_light_terminal(monkeypatch, adapter)
    iterations = 2
    result = solve_full_card_external_sampling_mccfr(
        adapter,
        iterations=iterations,
        seed=314159,
        max_infosets=10_000,
    )

    assert result.iterations == iterations
    assert result.traversals == 2 * iterations
    assert result.encountered_infosets > 0
    assert result.sampling_stats["root_posterior_samples"] == 2 * iterations
    assert result.sampling_stats["traversals_by_actor"] == {
        "bb": iterations,
        "btn": iterations,
    }
    assert result.sampling_stats["future_draw_samples"] > 0
    assert result.sampling_stats["traverser_actions_expanded"] > 0
    assert result.sampling_stats["opponent_action_samples"] > 0
    assert terminal_weights and set(terminal_weights) == {Fraction(1, 1)}

    assert tuple(result.average_strategy) == tuple(
        sorted(
            result.average_strategy,
            key=lambda key: (key.digest(), key.canonical_json()),
        )
    )
    for key, strategy in result.average_strategy.items():
        assert isinstance(key, InfoSetKey)
        assert tuple(strategy) == tuple(sorted(strategy))
        assert math.fsum(strategy.values()) == pytest.approx(1.0)
        assert set(strategy) == set(result.current_strategy[key])
        assert set(strategy) == set(result.cumulative_regret_plus[key])
        assert all(value >= 0.0 for value in result.cumulative_regret_plus[key].values())

    root_keys = [key for key in result.average_strategy if key.phase == "t3_first"]
    assert root_keys == [observation]
    assert serialize_strategy_profile(result.average_strategy) == (
        result.average_strategy_json
    )
    assert serialize_strategy_profile(result.current_strategy) == (
        result.current_strategy_json
    )
    assert hashlib.sha256(result.average_strategy_json.encode()).hexdigest() == (
        result.average_strategy_sha256
    )
    for forbidden in (
        '"particle_commitment"',
        '"undealt_cards"',
        '"remaining_cards"',
        '"particle_weight"',
        '"world_id"',
    ):
        assert forbidden not in result.average_strategy_json

    assert result.metadata["full_card"] is True
    assert result.metadata["full_card_policy_promoted"] is False
    assert result.metadata["hu_exact"] is False
    assert result.metadata["runtime_integrated"] is False
    assert result.metadata["exact_exploitability_computed"] is False
    assert result.metadata["opponent_sample_cached_per_infoset"] is True
    assert result.metadata["future_chance_sampled_per_physical_state"] is True
    assert result.metadata["posterior_probability_multiplied_after_sampling"] is False
    assert result.metadata["chance_probability_multiplied_after_sampling"] is False
    assert result.metadata["joint_particle_weight_used_after_sampling"] is False
    assert result.metadata["checkpoint_iteration"] == iterations
    assert len(result.metadata["checkpoint_sha256"]) == 64
    assert result.metadata["canonical_fl_ev_config_sha256"] == _canonical_sha256(
        result.metadata["canonical_fl_ev_config"]
    )
    assert {
        "ai/tutor/exact_late.py",
        "ai/engine/scoring.py",
        "ai/mcts/rollout_evaluator.py",
        "ai/config/fl_ev.json",
    }.issubset(result.metadata["source_sha256"])
    assert all(
        len(source_hash) == 64
        for source_hash in result.metadata["source_sha256"].values()
    )


def test_dynamic_mccfr_same_seed_replays_complete_tables_and_serialization(
    monkeypatch,
):
    observation, root_range = _one_world_natural_range(monkeypatch)
    first_adapter = FullCardGenerativeAdapter(observation, root_range)
    replay_adapter = FullCardGenerativeAdapter(observation, root_range)
    _install_light_terminal(monkeypatch, first_adapter, replay_adapter)
    first = solve_full_card_external_sampling_mccfr(
        first_adapter,
        iterations=1,
        seed=2026,
        max_infosets=10_000,
    )
    replay = solve_full_card_external_sampling_mccfr(
        replay_adapter,
        iterations=1,
        seed=2026,
        max_infosets=10_000,
    )

    assert first.average_strategy_json == replay.average_strategy_json
    assert first.current_strategy_json == replay.current_strategy_json
    assert first.average_strategy_sha256 == replay.average_strategy_sha256
    assert first.current_strategy_sha256 == replay.current_strategy_sha256
    assert first.sampling_stats == replay.sampling_stats
    assert {
        key.digest(): dict(values)
        for key, values in first.cumulative_regret_plus.items()
    } == {
        key.digest(): dict(values)
        for key, values in replay.cumulative_regret_plus.items()
    }


def test_dynamic_mccfr_two_hidden_discard_worlds_share_one_root_policy_key(
    monkeypatch,
):
    observation, root_range = _two_world_weighted_range(monkeypatch)
    adapter = FullCardGenerativeAdapter(observation, root_range)
    _install_light_terminal(monkeypatch, adapter)
    result = solve_full_card_external_sampling_mccfr(
        adapter,
        iterations=1,
        seed=9917,
        max_infosets=10_000,
        linear_averaging=False,
    )
    root_keys = [key for key in result.average_strategy if key.phase == "t3_first"]
    assert root_keys == [observation]
    assert result.metadata["table_key_contains_particle_commitment"] is False
    assert result.metadata["table_key_contains_remaining_cards"] is False
    assert result.metadata["table_key_contains_particle_weight"] is False
    assert result.metadata["artifact_contains_raw_particle_world"] is False
    assert result.sampling_stats["root_posterior_samples"] == 2


def test_dynamic_mccfr_max_infosets_is_fail_closed(monkeypatch):
    observation, root_range = _one_world_natural_range(monkeypatch)
    adapter = FullCardGenerativeAdapter(observation, root_range)
    _install_light_terminal(monkeypatch, adapter)
    with pytest.raises(RuntimeError, match="max_infosets exceeded"):
        solve_full_card_external_sampling_mccfr(
            adapter,
            iterations=1,
            seed=7,
            max_infosets=1,
        )


def test_dynamic_checkpoint_resume_is_byte_exact_one_shot_equivalent(
    monkeypatch, tmp_path
):
    observation, root_range = _one_world_natural_range(monkeypatch)
    one_shot_adapter = FullCardGenerativeAdapter(observation, root_range)
    first_adapter = FullCardGenerativeAdapter(observation, root_range)
    resumed_adapter = FullCardGenerativeAdapter(observation, root_range)
    _install_light_terminal(
        monkeypatch, one_shot_adapter, first_adapter, resumed_adapter
    )
    one_shot_path = tmp_path / "one-shot.json"
    split_path = tmp_path / "split.json"

    one_shot = solve_full_card_external_sampling_mccfr(
        one_shot_adapter,
        iterations=2,
        seed=86420,
        max_infosets=10_000,
        checkpoint_path=one_shot_path,
    )
    first = solve_full_card_external_sampling_mccfr(
        first_adapter,
        iterations=1,
        seed=86420,
        max_infosets=10_000,
        checkpoint_path=split_path,
    )
    resumed = solve_full_card_external_sampling_mccfr(
        resumed_adapter,
        iterations=1,
        seed=86420,
        max_infosets=10_000,
        resume_from=split_path,
        checkpoint_path=split_path,
    )

    assert first.iterations == 1
    assert resumed == one_shot
    assert split_path.read_bytes() == one_shot_path.read_bytes()
    assert resumed.metadata["checkpoint_iteration"] == 2
    assert resumed.metadata["checkpoint_sha256"] == one_shot.metadata[
        "checkpoint_sha256"
    ]
    envelope = json.loads(split_path.read_text(encoding="utf-8"))
    assert envelope["checkpoint_sha256"] == resumed.metadata["checkpoint_sha256"]
    assert _canonical_sha256(envelope["payload"]) == envelope["checkpoint_sha256"]
    assert envelope["payload"]["completed_iterations"] == 2
    assert envelope["payload"]["sampling_stats"] == dict(
        resumed.sampling_stats
    )


def test_dynamic_checkpoint_rejects_corruption_and_content_hash_mismatch(
    monkeypatch, tmp_path
):
    observation, root_range = _one_world_natural_range(monkeypatch)
    writer = FullCardGenerativeAdapter(observation, root_range)
    _install_light_terminal(monkeypatch, writer)
    checkpoint = tmp_path / "state.json"
    solve_full_card_external_sampling_mccfr(
        writer,
        iterations=1,
        seed=91,
        max_infosets=10_000,
        checkpoint_path=checkpoint,
    )
    valid_bytes = checkpoint.read_bytes()

    checkpoint.write_bytes(valid_bytes[: len(valid_bytes) // 2])
    with pytest.raises(
        full_mccfr_module.FullCardMccfrCheckpointError,
        match="not valid UTF-8 JSON",
    ):
        solve_full_card_external_sampling_mccfr(
            FullCardGenerativeAdapter(observation, root_range),
            iterations=1,
            seed=91,
            max_infosets=10_000,
            resume_from=checkpoint,
        )

    checkpoint.write_bytes(valid_bytes)
    envelope = json.loads(checkpoint.read_text(encoding="utf-8"))
    envelope["payload"]["sampling_stats"]["terminal_visits"] += 1
    _write_checkpoint_json(checkpoint, envelope)
    with pytest.raises(
        full_mccfr_module.FullCardMccfrCheckpointError,
        match="content SHA-256 mismatch",
    ):
        solve_full_card_external_sampling_mccfr(
            FullCardGenerativeAdapter(observation, root_range),
            iterations=1,
            seed=91,
            max_infosets=10_000,
            resume_from=checkpoint,
        )


def test_dynamic_checkpoint_rejects_solver_setting_mismatch(monkeypatch, tmp_path):
    observation, root_range = _one_world_natural_range(monkeypatch)
    writer = FullCardGenerativeAdapter(observation, root_range)
    _install_light_terminal(monkeypatch, writer)
    checkpoint = tmp_path / "state.json"
    solve_full_card_external_sampling_mccfr(
        writer,
        iterations=1,
        seed=1776,
        max_infosets=10_000,
        linear_averaging=True,
        checkpoint_path=checkpoint,
    )

    with pytest.raises(
        full_mccfr_module.FullCardMccfrCheckpointError, match="seed"
    ):
        solve_full_card_external_sampling_mccfr(
            FullCardGenerativeAdapter(observation, root_range),
            iterations=1,
            seed=1777,
            max_infosets=10_000,
            resume_from=checkpoint,
        )
    with pytest.raises(
        full_mccfr_module.FullCardMccfrCheckpointError,
        match="linear_averaging",
    ):
        solve_full_card_external_sampling_mccfr(
            FullCardGenerativeAdapter(observation, root_range),
            iterations=1,
            seed=1776,
            max_infosets=10_000,
            linear_averaging=False,
            resume_from=checkpoint,
        )
    with pytest.raises(
        full_mccfr_module.FullCardMccfrCheckpointError,
        match="solver configuration",
    ):
        solve_full_card_external_sampling_mccfr(
            FullCardGenerativeAdapter(observation, root_range),
            iterations=1,
            seed=1776,
            max_infosets=10_001,
            resume_from=checkpoint,
        )


def test_dynamic_checkpoint_rejects_current_range_mismatch(monkeypatch, tmp_path):
    observation, root_range = _one_world_natural_range(monkeypatch)
    writer = FullCardGenerativeAdapter(observation, root_range)
    _install_light_terminal(monkeypatch, writer)
    checkpoint = tmp_path / "state.json"
    solve_full_card_external_sampling_mccfr(
        writer,
        iterations=1,
        seed=808,
        max_infosets=10_000,
        checkpoint_path=checkpoint,
    )

    weighted_observation, weighted_range = _two_world_weighted_range(monkeypatch)
    mismatched = FullCardGenerativeAdapter(weighted_observation, weighted_range)
    _install_light_terminal(monkeypatch, mismatched)
    with pytest.raises(
        full_mccfr_module.FullCardMccfrCheckpointError,
        match="range, behavior, tree, or scoring binding",
    ):
        solve_full_card_external_sampling_mccfr(
            mismatched,
            iterations=1,
            seed=808,
            max_infosets=10_000,
            resume_from=checkpoint,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda envelope: envelope["payload"]["adapter_manifest"][
                "range_behavior_binding"
            ]["behavior_model_manifest"].__setitem__(
                "promotion_eligible", True
            ),
            "range, behavior, tree, or scoring binding",
        ),
        (
            lambda envelope: envelope["payload"]["adapter_manifest"].__setitem__(
                "remaining_phase_sequence", ["t3_first", "t4_first"]
            ),
            "range, behavior, tree, or scoring binding",
        ),
        (
            lambda envelope: envelope["payload"]["tables"][0][
                "stable_action_ids"
            ].__setitem__(
                0,
                envelope["payload"]["tables"][0]["stable_action_ids"][0]
                + "-tampered",
            ),
            "stable action IDs do not match the current tree",
        ),
    ),
    ids=("behavior-manifest", "tree-contract", "legal-actions"),
)
def test_dynamic_checkpoint_rejects_rehashed_behavior_and_tree_tampering(
    monkeypatch, tmp_path, mutation, message
):
    observation, root_range = _one_world_natural_range(monkeypatch)
    writer = FullCardGenerativeAdapter(observation, root_range)
    _install_light_terminal(monkeypatch, writer)
    original = tmp_path / "original.json"
    solve_full_card_external_sampling_mccfr(
        writer,
        iterations=1,
        seed=5150,
        max_infosets=10_000,
        checkpoint_path=original,
    )
    envelope = json.loads(original.read_text(encoding="utf-8"))
    mutation(envelope)
    _rehash_checkpoint_envelope(envelope)
    tampered = tmp_path / "tampered.json"
    _write_checkpoint_json(tampered, envelope)

    with pytest.raises(
        full_mccfr_module.FullCardMccfrCheckpointError, match=message
    ):
        solve_full_card_external_sampling_mccfr(
            FullCardGenerativeAdapter(observation, root_range),
            iterations=1,
            seed=5150,
            max_infosets=10_000,
            resume_from=tampered,
        )
