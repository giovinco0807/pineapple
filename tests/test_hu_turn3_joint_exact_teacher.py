import csv
import hashlib
import json
from collections import Counter

import pytest

import ofc_regular.hu_turn3_joint_exact_teacher as t3_teacher
from ofc_regular.action_key import ACTION_KEY_SCHEMA, action_key
from ofc_regular.action_space import Action, generate_turn_actions
from ofc_regular.cards import ALL_CARDS
from ofc_regular.evaluator import score_board
from ofc_regular.hu_belief import sample_hidden_card_particles
from ofc_regular.hu_infoset import (
    ActorObservation,
    InformationSetError,
    WorldState,
)
from ofc_regular.hu_turn3_joint_exact_teacher import (
    T3_EXPLICIT_SUPPORT_SCHEMA,
    T3_SEQUENTIAL_TEACHER_SCHEMA,
    T3ExplicitWorld,
    JointExactConfig,
    Turn3TeacherError,
    collect_state_records,
    evaluate_state_record,
    evaluate_t3_exact_explicit_support_actions,
    evaluate_t3_joint_batch,
    evaluate_t3_joint_exact_actions,
    write_summary_csv,
)
from ofc_regular.state import Board
from ofc_regular.teacher import terminal_score


def _constrained_board(cards):
    """Fill top/middle first so each T3/T4 decision has only three actions."""

    cards = tuple(cards)
    return Board.from_rows(
        top=cards[:3],
        middle=cards[3:8],
        bottom=cards[8:],
    )


def _observation(to_act_order: str) -> ActorObservation:
    opponent_count = 9 if to_act_order == "first" else 11
    cursor = 0
    hero_cards = ALL_CARDS[cursor : cursor + 9]
    cursor += 9
    opponent_cards = ALL_CARDS[cursor : cursor + opponent_count]
    cursor += opponent_count
    dealt = ALL_CARDS[cursor : cursor + 3]
    cursor += 3
    hero_discards = ALL_CARDS[cursor : cursor + 2]
    return ActorObservation(
        hero_board=_constrained_board(hero_cards),
        opponent_public_board=_constrained_board(opponent_cards),
        dealt_cards=dealt,
        hero_private_discards=hero_discards,
        seat=to_act_order,  # type: ignore[arg-type]
        street="T3",
        to_act_order=to_act_order,  # type: ignore[arg-type]
    )


def _config(
    run_id: str,
    *,
    candidate_samples: int = 1,
    evaluation_samples: int = 1,
    candidate_seed: int | None = None,
    evaluation_seed: int | None = None,
):
    return JointExactConfig(
        candidate_samples=candidate_samples,
        evaluation_samples=evaluation_samples,
        downstream_t3_samples=1,
        downstream_t4_samples=1,
        seed=2026071303,
        candidate_seed=candidate_seed,
        evaluation_seed=evaluation_seed,
        run_id=run_id,
    )


def _canonical_first_action(observation: ActorObservation):
    actions = generate_turn_actions(
        observation.hero_board,
        observation.dealt_cards,
    )
    return min(actions, key=lambda action: action_key(action).sort_key())


def _reduced_search_context(observation: ActorObservation, run_id: str):
    return t3_teacher._SearchContext(
        config=_config(run_id),
        fl_ev=dict(observation.scoring.fl_ev),
        t4_action_cache={},
        t3_second_action_cache={},
        child_observation_fingerprints=set(),
    )


def _assert_single_terminal_row(row, *, score, hero_final, downstream_keys):
    hero_board_score = score_board(
        hero_final.top,
        hero_final.middle,
        hero_final.bottom,
    )
    assert row["score"] == pytest.approx(score)
    assert row["joint_ev"] == pytest.approx(score)
    assert row["future_count"] == 1
    assert row["score_min"] == pytest.approx(score)
    assert row["score_max"] == pytest.approx(score)
    assert row["bust_count"] == int(hero_board_score.busted)
    assert row["fl_entry_count"] == int(hero_board_score.fl_entry.qualifies)
    assert row["royalty_mean"] == hero_board_score.total_royalty
    assert row["downstream_action_counts"] == dict(Counter(downstream_keys))


def _explicit_worlds(observation: ActorObservation):
    particles = sample_hidden_card_particles(
        observation,
        base_seed=2026071331,
        run_id=f"explicit-worlds-{observation.to_act_order}",
        sample_count=2,
    ).particles
    future_count = 9 if observation.to_act_order == "first" else 6
    weights = (0.25, 0.75)
    return tuple(
        T3ExplicitWorld(
            opponent_private_discards=particle.opponent_private_discards,
            future_cards=particle.unseen_deck[:future_count],
            weight=weights[index],
            world_id=f"{observation.to_act_order}-world-{index}",
        )
        for index, particle in enumerate(particles)
    )


def _manual_explicit_t3_second_score(
    observation: ActorObservation,
    root_action: Action,
    world: T3ExplicitWorld,
):
    after_root = observation.hero_board.place(root_action.placements)
    opponent_t4_observation = ActorObservation(
        hero_board=observation.opponent_public_board,
        opponent_public_board=after_root,
        dealt_cards=world.draw(3, offset=0),
        hero_private_discards=world.opponent_private_discards,
        seat="first",
        street="T4",
        to_act_order="first",
        scoring=observation.scoring,
    )
    opponent_t4_action = _canonical_first_action(opponent_t4_observation)
    opponent_final = observation.opponent_public_board.place(
        opponent_t4_action.placements
    )
    hero_t4_observation = ActorObservation(
        hero_board=after_root,
        opponent_public_board=opponent_final,
        dealt_cards=world.draw(3, offset=3),
        hero_private_discards=(
            *observation.hero_private_discards,
            *root_action.discards,
        ),
        seat="second",
        street="T4",
        to_act_order="second",
        scoring=observation.scoring,
    )
    hero_t4_action = _canonical_first_action(hero_t4_observation)
    hero_final = after_root.place(hero_t4_action.placements)
    score, _ = terminal_score(
        hero_final,
        opponent_final,
        fl_ev=dict(observation.scoring.fl_ev),
    )
    return float(score)


def _manual_explicit_t3_first_score(
    observation: ActorObservation,
    root_action: Action,
    world: T3ExplicitWorld,
):
    after_root = observation.hero_board.place(root_action.placements)
    opponent_t3_observation = ActorObservation(
        hero_board=observation.opponent_public_board,
        opponent_public_board=after_root,
        dealt_cards=world.draw(3, offset=0),
        hero_private_discards=world.opponent_private_discards,
        seat="second",
        street="T3",
        to_act_order="second",
        scoring=observation.scoring,
    )
    opponent_t3_action = _canonical_first_action(opponent_t3_observation)
    opponent_after_t3 = observation.opponent_public_board.place(
        opponent_t3_action.placements
    )
    hero_t4_observation = ActorObservation(
        hero_board=after_root,
        opponent_public_board=opponent_after_t3,
        dealt_cards=world.draw(3, offset=3),
        hero_private_discards=(
            *observation.hero_private_discards,
            *root_action.discards,
        ),
        seat="first",
        street="T4",
        to_act_order="first",
        scoring=observation.scoring,
    )
    hero_t4_action = _canonical_first_action(hero_t4_observation)
    hero_final = after_root.place(hero_t4_action.placements)
    opponent_t4_observation = ActorObservation(
        hero_board=opponent_after_t3,
        opponent_public_board=hero_final,
        dealt_cards=world.draw(3, offset=6),
        hero_private_discards=(
            *world.opponent_private_discards,
            *opponent_t3_action.discards,
        ),
        seat="second",
        street="T4",
        to_act_order="second",
        scoring=observation.scoring,
    )
    opponent_t4_action = _canonical_first_action(opponent_t4_observation)
    opponent_final = opponent_after_t3.place(opponent_t4_action.placements)
    score, _ = terminal_score(
        hero_final,
        opponent_final,
        fl_ev=dict(observation.scoring.fl_ev),
    )
    return float(score)


def _explicit_support_digest(worlds):
    payload = [
        {
            "world_id": world.world_id,
            "weight": world.weight,
            "opponent_private_discards": list(world.opponent_private_discards),
            "future_cards": list(world.future_cards),
        }
        for world in worlds
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@pytest.mark.parametrize("to_act_order", ("first", "second"))
def test_t3_v2_accepts_only_live_first_and_second_geometry(to_act_order):
    observation = _observation(to_act_order)

    sample = evaluate_t3_joint_exact_actions(
        observation=observation,
        config=_config(f"geometry-{to_act_order}"),
    )

    assert sample["schema"] == T3_SEQUENTIAL_TEACHER_SCHEMA
    assert sample["phase"] == "hu_turn3_9card"
    assert sample["street"] == "T3"
    assert sample["seat"] == to_act_order
    assert sample["to_act_order"] == to_act_order
    assert sample["observation_fingerprint"] == observation.fingerprint()
    assert sample["policy_observation"] == observation.to_dict()
    assert sample["action_key_schema"] == ACTION_KEY_SCHEMA
    assert sample["legal_action_count"] == 3
    assert len(sample["actions"]) == 3
    assert sample["future_count"] == 1
    assert sample["candidate_belief"]["sample_count"] == 1
    assert sample["evaluation_belief"]["sample_count"] == 1
    assert sample["sample_independence"] == "disjoint_particle_rng_keys"
    assert sample["teacher_notes"]["primary_metric"] == (
        "locked_action_independent_evaluation_ev"
    )

    selected = [row for row in sample["actions"] if row["selected_by_candidate_plan"]]
    assert len(selected) == 1
    assert selected[0]["action_key"] == sample["selected_action_key"]
    assert selected[0]["original_index"] == sample["selected_action_original_index"]
    assert selected[0]["score"] == sample["selected_action_evaluation_score"]
    for action in sample["actions"]:
        assert action["selection_future_count"] == 1
        assert action["evaluation_future_count"] == 1
        assert action["future_count"] == 1
        assert action["action_key"].startswith("rak1:")
        assert 0.0 <= action["bust_rate"] <= 1.0
        assert 0.0 <= action["fl_entry_rate"] <= 1.0
        assert sum(action["top_category_counts"].values()) == 1
        assert sum(action["middle_category_counts"].values()) == 1
        assert sum(action["bottom_category_counts"].values()) == 1


def test_t3_v2_is_deterministic_for_identical_observation_and_config():
    observation = _observation("first")
    config = _config("deterministic-repeat")

    first = evaluate_t3_joint_exact_actions(
        observation=observation,
        config=config,
    )
    repeated = evaluate_t3_joint_exact_actions(
        observation=observation,
        config=config,
    )

    assert repeated == first


def test_candidate_and_evaluation_particle_rng_keys_are_disjoint():
    observation = _observation("second")
    candidate = sample_hidden_card_particles(
        observation,
        base_seed=811,
        run_id="t3-v2-explicit-candidate",
        sample_count=2,
    )
    evaluation = sample_hidden_card_particles(
        observation,
        base_seed=812,
        run_id="t3-v2-explicit-evaluation",
        sample_count=2,
    )

    sample = evaluate_t3_joint_exact_actions(
        observation=observation,
        config=_config(
            "explicit-disjoint-batches",
            candidate_samples=2,
            evaluation_samples=2,
        ),
        candidate_belief_batch=candidate,
        evaluation_belief_batch=evaluation,
    )

    candidate_keys = {particle.rng_key_digest for particle in candidate.particles}
    evaluation_keys = {particle.rng_key_digest for particle in evaluation.particles}
    assert candidate_keys.isdisjoint(evaluation_keys)
    assert sample["candidate_belief"]["particle_digests"] == list(
        candidate.particle_digests
    )
    assert sample["evaluation_belief"]["particle_digests"] == list(
        evaluation.particle_digests
    )
    assert all(row["selection_future_count"] == 2 for row in sample["actions"])
    assert all(row["evaluation_future_count"] == 2 for row in sample["actions"])

    with pytest.raises(Turn3TeacherError, match="RNG keys overlap"):
        evaluate_t3_joint_exact_actions(
            observation=observation,
            config=_config("overlapping-batches"),
            candidate_belief_batch=candidate,
            evaluation_belief_batch=candidate,
        )


@pytest.mark.parametrize("to_act_order", ("first", "second"))
def test_candidate_and_evaluation_seeds_are_independent_at_t3(to_act_order):
    observation = _observation(to_act_order)
    common = {
        "candidate_samples": 2,
        "evaluation_samples": 2,
    }
    baseline = evaluate_t3_joint_exact_actions(
        observation=observation,
        config=_config(
            f"t3-independent-seeds-{to_act_order}",
            **common,
            candidate_seed=2026071321,
            evaluation_seed=2026071322,
        ),
    )
    changed_evaluation = evaluate_t3_joint_exact_actions(
        observation=observation,
        config=_config(
            f"t3-independent-seeds-{to_act_order}",
            **common,
            candidate_seed=2026071321,
            evaluation_seed=2026071323,
        ),
    )
    changed_candidate = evaluate_t3_joint_exact_actions(
        observation=observation,
        config=_config(
            f"t3-independent-seeds-{to_act_order}",
            **common,
            candidate_seed=2026071324,
            evaluation_seed=2026071322,
        ),
    )

    def candidate_projection(result):
        return {
            "belief": result["candidate_belief"],
            "selected_action_key": result["selected_action_key"],
            "selected_action_original_index": result[
                "selected_action_original_index"
            ],
            "selection_score_gap": result["selection_score_gap"],
            "actions": {
                row["action_key"]: (
                    row["selection_score"],
                    row["selection_future_count"],
                    row["selected_by_candidate_plan"],
                )
                for row in result["actions"]
            },
        }

    def evaluation_projection(result):
        return {
            "belief": result["evaluation_belief"],
            "sample_best": result["evaluation_sample_best_score"],
            "actions": {
                row["action_key"]: (
                    row["score"],
                    row["evaluation_future_count"],
                    row["evaluation_regret_vs_sample_best"],
                )
                for row in result["actions"]
            },
        }

    assert candidate_projection(changed_evaluation) == candidate_projection(baseline)
    assert changed_evaluation["evaluation_belief"] != baseline[
        "evaluation_belief"
    ]
    assert evaluation_projection(changed_candidate) == evaluation_projection(baseline)
    assert changed_candidate["candidate_belief"] != baseline["candidate_belief"]


def test_scalar_and_reference_batch_results_are_identical_for_both_seats():
    observations = [_observation("first"), _observation("second")]
    config = _config("scalar-batch-parity")

    scalar = [
        evaluate_t3_joint_exact_actions(observation=observation, config=config)
        for observation in observations
    ]
    batched = evaluate_t3_joint_batch(observations, config=config)

    assert batched == scalar


def test_reduced_tree_t3_second_matches_independent_event_order_oracle(monkeypatch):
    observation = _observation("second")
    root_actions = generate_turn_actions(
        observation.hero_board,
        observation.dealt_cards,
    )
    batch = sample_hidden_card_particles(
        observation,
        base_seed=2026071314,
        run_id="reduced-tree-t3-second-root",
        sample_count=1,
    )
    particle = batch.particles[0]
    observed_events = []

    def lock_t4(child_observation, _context):
        observed_events.append(("lock_t4", child_observation))
        return _canonical_first_action(child_observation)

    def unexpected_t3_lock(_child_observation, _context):
        raise AssertionError("T3-second root must not create another T3 decision")

    monkeypatch.setattr(t3_teacher, "_locked_t4_action", lock_t4)
    monkeypatch.setattr(
        t3_teacher,
        "_locked_t3_second_action",
        unexpected_t3_lock,
    )
    rows = t3_teacher._score_t3_actions(
        observation,
        root_actions,
        (particle,),
        _reduced_search_context(observation, "reduced-tree-t3-second"),
    )

    expected_events = []
    expected_by_root_key = {}
    for root_action in root_actions:
        after_root = observation.hero_board.place(root_action.placements)
        opponent_t4_observation = ActorObservation(
            hero_board=observation.opponent_public_board,
            opponent_public_board=after_root,
            dealt_cards=particle.draw(3, offset=0),
            hero_private_discards=particle.opponent_private_discards,
            seat="first",
            street="T4",
            to_act_order="first",
            scoring=observation.scoring,
        )
        expected_events.append(("lock_t4", opponent_t4_observation))
        opponent_t4_action = _canonical_first_action(opponent_t4_observation)
        opponent_final = observation.opponent_public_board.place(
            opponent_t4_action.placements
        )
        hero_t4_observation = ActorObservation(
            hero_board=after_root,
            opponent_public_board=opponent_final,
            dealt_cards=particle.draw(3, offset=3),
            hero_private_discards=(
                *observation.hero_private_discards,
                *root_action.discards,
            ),
            seat="second",
            street="T4",
            to_act_order="second",
            scoring=observation.scoring,
        )
        expected_events.append(("lock_t4", hero_t4_observation))
        hero_t4_action = _canonical_first_action(hero_t4_observation)
        hero_final = after_root.place(hero_t4_action.placements)
        score, _ = terminal_score(
            hero_final,
            opponent_final,
            fl_ev=dict(observation.scoring.fl_ev),
        )
        expected_by_root_key[action_key(root_action).to_token()] = (
            float(score),
            hero_final,
            (
                action_key(opponent_t4_action).to_token(),
                action_key(hero_t4_action).to_token(),
            ),
        )

    assert observed_events == expected_events
    assert [
        (
            child.street,
            child.seat,
            child.to_act_order,
            child.hero_board.card_count(),
            child.opponent_public_board.card_count(),
        )
        for _event, child in observed_events
    ] == [
        ("T4", "first", "first", 11, 11),
        ("T4", "second", "second", 11, 13),
    ] * len(root_actions)
    actual_by_root_key = {
        action_key(root_action).to_token(): row
        for root_action, row in zip(root_actions, rows)
    }
    assert actual_by_root_key.keys() == expected_by_root_key.keys()
    for root_key, (score, hero_final, downstream_keys) in expected_by_root_key.items():
        _assert_single_terminal_row(
            actual_by_root_key[root_key],
            score=score,
            hero_final=hero_final,
            downstream_keys=downstream_keys,
        )


def test_reduced_tree_t3_first_matches_independent_event_order_oracle(monkeypatch):
    observation = _observation("first")
    root_actions = generate_turn_actions(
        observation.hero_board,
        observation.dealt_cards,
    )
    batch = sample_hidden_card_particles(
        observation,
        base_seed=2026071315,
        run_id="reduced-tree-t3-first-root",
        sample_count=1,
    )
    particle = batch.particles[0]
    observed_events = []

    def lock_t3_second(child_observation, _context):
        observed_events.append(("lock_t3_second", child_observation))
        return _canonical_first_action(child_observation)

    def lock_t4(child_observation, _context):
        observed_events.append(("lock_t4", child_observation))
        return _canonical_first_action(child_observation)

    monkeypatch.setattr(
        t3_teacher,
        "_locked_t3_second_action",
        lock_t3_second,
    )
    monkeypatch.setattr(t3_teacher, "_locked_t4_action", lock_t4)
    rows = t3_teacher._score_t3_actions(
        observation,
        root_actions,
        (particle,),
        _reduced_search_context(observation, "reduced-tree-t3-first"),
    )

    expected_events = []
    expected_by_root_key = {}
    for root_action in root_actions:
        after_root = observation.hero_board.place(root_action.placements)
        opponent_t3_observation = ActorObservation(
            hero_board=observation.opponent_public_board,
            opponent_public_board=after_root,
            dealt_cards=particle.draw(3, offset=0),
            hero_private_discards=particle.opponent_private_discards,
            seat="second",
            street="T3",
            to_act_order="second",
            scoring=observation.scoring,
        )
        expected_events.append(("lock_t3_second", opponent_t3_observation))
        opponent_t3_action = _canonical_first_action(opponent_t3_observation)
        opponent_after_t3 = observation.opponent_public_board.place(
            opponent_t3_action.placements
        )
        hero_t4_observation = ActorObservation(
            hero_board=after_root,
            opponent_public_board=opponent_after_t3,
            dealt_cards=particle.draw(3, offset=3),
            hero_private_discards=(
                *observation.hero_private_discards,
                *root_action.discards,
            ),
            seat="first",
            street="T4",
            to_act_order="first",
            scoring=observation.scoring,
        )
        expected_events.append(("lock_t4", hero_t4_observation))
        hero_t4_action = _canonical_first_action(hero_t4_observation)
        hero_final = after_root.place(hero_t4_action.placements)
        opponent_t4_observation = ActorObservation(
            hero_board=opponent_after_t3,
            opponent_public_board=hero_final,
            dealt_cards=particle.draw(3, offset=6),
            hero_private_discards=(
                *particle.opponent_private_discards,
                *opponent_t3_action.discards,
            ),
            seat="second",
            street="T4",
            to_act_order="second",
            scoring=observation.scoring,
        )
        expected_events.append(("lock_t4", opponent_t4_observation))
        opponent_t4_action = _canonical_first_action(opponent_t4_observation)
        opponent_final = opponent_after_t3.place(opponent_t4_action.placements)
        score, _ = terminal_score(
            hero_final,
            opponent_final,
            fl_ev=dict(observation.scoring.fl_ev),
        )
        expected_by_root_key[action_key(root_action).to_token()] = (
            float(score),
            hero_final,
            (
                action_key(opponent_t3_action).to_token(),
                action_key(hero_t4_action).to_token(),
                action_key(opponent_t4_action).to_token(),
            ),
        )

    assert observed_events == expected_events
    assert [
        (
            event,
            child.street,
            child.seat,
            child.to_act_order,
            child.hero_board.card_count(),
            child.opponent_public_board.card_count(),
        )
        for event, child in observed_events
    ] == [
        ("lock_t3_second", "T3", "second", "second", 9, 11),
        ("lock_t4", "T4", "first", "first", 11, 11),
        ("lock_t4", "T4", "second", "second", 11, 13),
    ] * len(root_actions)
    actual_by_root_key = {
        action_key(root_action).to_token(): row
        for root_action, row in zip(root_actions, rows)
    }
    assert actual_by_root_key.keys() == expected_by_root_key.keys()
    for root_key, (score, hero_final, downstream_keys) in expected_by_root_key.items():
        _assert_single_terminal_row(
            actual_by_root_key[root_key],
            score=score,
            hero_final=hero_final,
            downstream_keys=downstream_keys,
        )


def test_real_t3_child_strategy_does_not_fuse_different_outer_deck_tails():
    observation = _observation("first")
    root_action = _canonical_first_action(observation)
    particle = sample_hidden_card_particles(
        observation,
        base_seed=2026071341,
        run_id="strategy-fusion-outer-worlds",
        sample_count=1,
    ).particles[0]
    common_t3_deal = particle.unseen_deck[:3]
    first_tail = particle.unseen_deck[3:9]
    second_tail = particle.unseen_deck[9:15]
    worlds = (
        T3ExplicitWorld(
            opponent_private_discards=particle.opponent_private_discards,
            future_cards=(*common_t3_deal, *first_tail),
            weight=0.5,
            world_id="outer-tail-a",
        ),
        T3ExplicitWorld(
            opponent_private_discards=particle.opponent_private_discards,
            future_cards=(*common_t3_deal, *second_tail),
            weight=0.5,
            world_id="outer-tail-b",
        ),
    )
    assert worlds[0].future_cards[:3] == worlds[1].future_cards[:3]
    assert worlds[0].future_cards[3:] != worlds[1].future_cards[3:]

    after_root = observation.hero_board.place(root_action.placements)

    def child_observation(world):
        return ActorObservation(
            hero_board=observation.opponent_public_board,
            opponent_public_board=after_root,
            dealt_cards=world.draw(3, offset=0),
            hero_private_discards=world.opponent_private_discards,
            seat="second",
            street="T3",
            to_act_order="second",
            scoring=observation.scoring,
        )

    first_child = child_observation(worlds[0])
    second_child = child_observation(worlds[1])
    assert first_child == second_child
    assert first_child.fingerprint() == second_child.fingerprint()

    first_context = _reduced_search_context(
        observation,
        "real-strategy-fusion-fresh-context",
    )
    second_context = _reduced_search_context(
        observation,
        "real-strategy-fusion-fresh-context",
    )
    assert first_context is not second_context
    assert not first_context.t3_second_action_cache
    assert not second_context.t3_second_action_cache

    first_terminal = t3_teacher._rollout_t3_first(
        observation,
        root_action,
        worlds[0],
        first_context,
    )
    second_terminal = t3_teacher._rollout_t3_first(
        observation,
        root_action,
        worlds[1],
        second_context,
    )

    fingerprint = first_child.fingerprint()
    first_action = first_context.t3_second_action_cache[fingerprint]
    second_action = second_context.t3_second_action_cache[fingerprint]
    assert action_key(first_action) == action_key(second_action)
    assert first_terminal["downstream_action_keys"][0] == action_key(
        first_action
    ).to_token()
    assert second_terminal["downstream_action_keys"][0] == action_key(
        second_action
    ).to_token()


@pytest.mark.parametrize("to_act_order", ("first", "second"))
def test_exact_explicit_support_matches_manual_weighted_oracle_for_all_actions(
    to_act_order,
):
    observation = _observation(to_act_order)
    worlds = _explicit_worlds(observation)
    root_actions = generate_turn_actions(
        observation.hero_board,
        observation.dealt_cards,
    )
    manual_score = (
        _manual_explicit_t3_first_score
        if to_act_order == "first"
        else _manual_explicit_t3_second_score
    )
    expected_by_key = {
        action_key(root_action).to_token(): sum(
            world.weight * manual_score(observation, root_action, world)
            for world in worlds
        )
        for root_action in root_actions
    }

    result = evaluate_t3_exact_explicit_support_actions(
        observation=observation,
        worlds=worlds,
        t4_selector=_canonical_first_action,
        t3_second_selector=_canonical_first_action,
        continuation_policy_id="unit_test_canonical_first_v1",
    )

    actual_by_key = {row["action_key"]: row["score"] for row in result["actions"]}
    expected_selected_key = min(
        expected_by_key,
        key=lambda token: (
            -expected_by_key[token],
            next(
                action_key(action).sort_key()
                for action in root_actions
                if action_key(action).to_token() == token
            ),
        ),
    )
    assert actual_by_key == pytest.approx(expected_by_key, rel=0.0, abs=1e-12)
    assert result["schema"] == T3_EXPLICIT_SUPPORT_SCHEMA
    assert result["mode"] == "exact_over_declared_finite_support"
    assert result["full_52_card_tree_claimed"] is False
    assert result["support_count"] == 2
    assert result["support_weight_sum"] == pytest.approx(1.0)
    assert result["support_digest"] == _explicit_support_digest(worlds)
    assert result["selected_action_key"] == expected_selected_key
    assert result["best_score"] == pytest.approx(expected_by_key[expected_selected_key])
    assert result["legal_action_count"] == len(root_actions) == 3
    assert all(row["future_count"] == len(worlds) for row in result["actions"])
    assert result["teacher_notes"]["strategy_fusion_guard"] == (
        "selectors_receive_only_ActorObservation"
    )


def test_exact_explicit_support_rejects_invalid_weights_overlap_and_selector():
    observation = _observation("second")
    worlds = _explicit_worlds(observation)
    invalid_weights = tuple(
        T3ExplicitWorld(
            opponent_private_discards=world.opponent_private_discards,
            future_cards=world.future_cards,
            weight=0.4,
            world_id=world.world_id,
        )
        for world in worlds
    )
    with pytest.raises(Turn3TeacherError, match="weights must sum to one"):
        evaluate_t3_exact_explicit_support_actions(
            observation=observation,
            worlds=invalid_weights,
            t4_selector=_canonical_first_action,
            continuation_policy_id="invalid-weight-support",
        )

    first_world = worlds[0]
    overlap_worlds = (
        T3ExplicitWorld(
            opponent_private_discards=first_world.opponent_private_discards,
            future_cards=(
                observation.hero_board.all_cards()[0],
                *first_world.future_cards[1:],
            ),
            weight=first_world.weight,
            world_id=first_world.world_id,
        ),
        worlds[1],
    )
    with pytest.raises(Turn3TeacherError, match="overlaps actor-visible cards"):
        evaluate_t3_exact_explicit_support_actions(
            observation=observation,
            worlds=overlap_worlds,
            t4_selector=_canonical_first_action,
            continuation_policy_id="visible-overlap-support",
        )

    def illegal_selector(child_observation):
        return Action(
            placements=((child_observation.dealt_cards[0], "top"),),
            discards=(),
        )

    with pytest.raises(Turn3TeacherError, match="selector returned an illegal action"):
        evaluate_t3_exact_explicit_support_actions(
            observation=observation,
            worlds=worlds,
            t4_selector=illegal_selector,
            continuation_policy_id="illegal-selector-support",
        )


def test_teacher_rejects_raw_mapping_and_world_state_at_api_boundary():
    observation = _observation("first")
    unknown = [
        card
        for card in ALL_CARDS
        if card not in set(observation.known_unavailable_cards())
    ]
    world = WorldState(
        boards=(observation.hero_board, observation.opponent_public_board),
        private_discards=(
            observation.hero_private_discards,
            tuple(unknown[: observation.opponent_discard_count]),
        ),
        street="T3",
        next_player=0,
    )

    with pytest.raises(TypeError, match="requires ActorObservation"):
        evaluate_t3_joint_exact_actions(  # type: ignore[arg-type]
            observation={"policy_observation": observation.to_dict()},
            config=_config("unsafe-mapping"),
        )
    with pytest.raises(TypeError, match="requires ActorObservation"):
        evaluate_t3_joint_exact_actions(  # type: ignore[arg-type]
            observation=world,
            config=_config("unsafe-world"),
        )


def test_state_record_requires_versioned_policy_observation():
    observation = _observation("second")
    config = _config("state-record")
    legacy = {
        "board": {"top": [], "middle": [], "bottom": []},
        "opponent_board": {"top": [], "middle": [], "bottom": []},
        "dealt": list(observation.dealt_cards),
        "dead_cards": ["As"],
    }
    with pytest.raises(InformationSetError, match="requires versioned policy_observation"):
        evaluate_state_record(legacy, sample_id=0, config=config)

    state = {
        "state_id": "safe-t3-second",
        "source": "unit_test_fresh_state",
        "hand_seed": 2026071311,
        "hand_index": 4,
        "visibility_model": "actor_observation_v1",
        "discard_visibility": "own_private_only",
        "policy_observation": observation.to_dict(),
        "selection": {"baseline_index": 0, "hu_index": 1},
    }
    sample = evaluate_state_record(state, sample_id=7, config=config)

    assert sample["sample_id"] == 7
    assert sample["seat"] == "second"
    assert sample["source"] == "unit_test_fresh_state"
    assert sample["source_state"]["state_id"] == "safe-t3-second"
    assert sample["source_state"]["hand_seed"] == 2026071311
    assert sample["selection"] == state["selection"]


def test_collect_state_records_accepts_multiple_inputs_and_global_limit(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text('{"state_id":1}\n{"state_id":2}\n', encoding="utf-8")
    second.write_text('{"state_id":3}\n', encoding="utf-8")

    records = collect_state_records([first, second], max_states=2)

    assert [record["state_id"] for record in records] == [1, 2]
    assert records[0]["source_input_path"] == str(first)


def test_write_summary_csv_outputs_v2_action_rows(tmp_path):
    sample = evaluate_t3_joint_exact_actions(
        observation=_observation("first"),
        config=_config("summary-csv"),
    )
    sample["sample_id"] = 3
    out = tmp_path / "summary.csv"

    write_summary_csv(out, [sample])

    with out.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == sample["legal_action_count"] == 3
    assert {row["seat"] for row in rows} == {"first"}
    assert all(row["action_key"].startswith("rak1:") for row in rows)
    assert all(row["selection_score"] for row in rows)
    assert all(row["score"] for row in rows)
    assert json.loads(rows[0]["placements"])
    assert json.loads(rows[0]["discards"])
