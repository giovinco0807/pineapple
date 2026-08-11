import random
from itertools import permutations

import numpy as np

from ofc_regular.action_key import action_key
from ofc_regular.action_space import generate_actions
from ofc_regular.cards import create_deck
from ofc_regular.counter_rng import COUNTER_RNG_SCHEMA
from ofc_regular.hu_belief import sample_hidden_card_particles
from ofc_regular.hu_infoset import WorldState
import ofc_regular.hu_turn1_teacher_pilot as pilot
from ofc_regular.hu_turn1_teacher_pilot import (
    _duplicate_profile_stats,
    build_turn1_pilot_samples,
    evaluate_turn1_action_subset,
    evaluate_turn1_actions,
)
from ofc_regular.policy import RegularAiPolicy
from ofc_regular.state import Board


def _belief_kwargs(board, opponent, dealt, *, sample_count, seed):
    world = WorldState(
        boards=(board, opponent),
        private_discards=((), ()),
        street="T1",
        next_player=0,
    )
    observation = world.observe(0, dealt)
    return {
        "observation": observation,
        "belief_batch": sample_hidden_card_particles(
            observation,
            base_seed=seed,
            run_id=f"t1_test_belief|samples={sample_count}",
            sample_count=sample_count,
        ),
    }


def test_evaluate_turn1_actions_smoke_with_random_continuation():
    deck = create_deck(shuffle=False)
    board = Board.from_rows(top=deck[:1], middle=deck[1:3], bottom=deck[3:5])
    opponent_board = Board.from_rows(top=deck[5:6], middle=deck[6:8], bottom=deck[8:10])
    dealt = tuple(deck[10:13])
    remaining = tuple(deck[13:])
    policies = [
        RegularAiPolicy(seed=11, seat="first"),
        RegularAiPolicy(seed=12, seat="second"),
    ]
    profile = {}

    rows, truncated = evaluate_turn1_actions(
        board=board,
        opponent_board=opponent_board,
        dealt=dealt,
        remaining_cards=remaining,
        private_discards=([], []),
        hero_player=0,
        policies=policies,
        hand_seed=2026062301,
        sample_id=0,
        future_samples=1,
        max_actions=2,
        rng=random.Random(2026062301),
        profile=profile,
    )

    assert rows
    assert truncated is True
    assert rows[0]["rollout_count"] == 1
    assert "score" in rows[0]
    assert "se" in rows[0]
    assert rows[0]["placements"]
    assert rows[0]["discards"]
    assert rows[0]["action_eval_seconds"] >= 0.0
    assert profile["rollout_count"] == 2
    assert profile["choose_action_count"] > 0
    assert sum(profile["_t2_state_key_counts"].values()) == profile["choose_action_T2_count"]
    assert sum(profile["_t2_decision_key_counts"].values()) == profile["choose_action_T2_count"]
    assert profile["terminal_score_count"] == 2


def test_duplicate_profile_stats_reports_cache_upper_bound():
    stats = _duplicate_profile_stats({"a": 3, "b": 1})

    assert stats["raw"] == 4
    assert stats["unique"] == 2
    assert stats["repeated"] == 2
    assert stats["duplicate_rate"] == 0.5
    assert stats["max_occurrence"] == 3


def test_t1_candidate_ties_select_same_action_keys_for_all_dealt_permutations():
    class TieModel:
        def predict_sample(self, sample):
            return np.zeros(len(sample["actions"]), dtype=np.float64)

    deck = create_deck(shuffle=False)
    board = Board.from_rows(top=deck[:1], middle=deck[1:3], bottom=deck[3:5])
    opponent = Board.from_rows(top=deck[5:6], middle=deck[6:8], bottom=deck[8:10])
    dealt = tuple(deck[10:13])
    expected_by_mode = {}
    for mode in pilot.CANDIDATE_UNION_MODES:
        selections = []
        for permuted in permutations(dealt):
            actions = generate_actions(board, permuted)
            pairs, selector = pilot._select_candidate_pairs(
                board=board,
                opponent_board=opponent,
                dealt=permuted,
                actions=actions,
                visible_dead_cards=opponent.all_cards(),
                seat="first",
                candidate_model=TieModel(),
                candidate_topk=4,
                candidate_union_mode=mode,
            )
            tokens = tuple(action_key(action).to_token() for _, action in pairs)
            assert selector is not None
            assert tuple(selector["selected_action_keys"]) == tokens
            selections.append(tokens)
        assert all(tokens == selections[0] for tokens in selections)
        expected_by_mode[mode] = selections[0]

    assert len(set(expected_by_mode.values())) == 1


def test_evaluate_turn1_action_subset_reports_paired_delta():
    deck = create_deck(shuffle=False)
    board = Board.from_rows(top=deck[:1], middle=deck[1:3], bottom=deck[3:5])
    opponent_board = Board.from_rows(top=deck[5:6], middle=deck[6:8], bottom=deck[8:10])
    dealt = tuple(deck[10:13])
    remaining = tuple(deck[13:])
    policies = [
        RegularAiPolicy(seed=21, seat="first"),
        RegularAiPolicy(seed=22, seat="second"),
    ]

    result = evaluate_turn1_action_subset(
        board=board,
        opponent_board=opponent_board,
        dealt=dealt,
        hero_player=0,
        policies=policies,
        hand_seed=2026071101,
        sample_id=0,
        future_samples=2,
        action_indices=[0, 1],
        paired_delta_candidate_index=1,
        paired_delta_baseline_index=0,
        rng=random.Random(2026071101),
        **_belief_kwargs(
            board, opponent_board, dealt, sample_count=2, seed=2026071101
        ),
    )

    assert result["evaluated_action_count"] == 2
    assert {row["original_index"] for row in result["actions"]} == {0, 1}
    assert result["paired_delta_candidate_index"] == 1
    assert result["paired_delta_baseline_index"] == 0
    assert result["paired_delta_count"] == 2
    assert "paired_delta_mean" in result
    assert result["paired_delta_standard_error"] >= 0.0
    assert result["rng_schema"] == COUNTER_RNG_SCHEMA
    assert result["policy_seed_common_across_actions"] is True
    assert len(result["legal_action_set_digest"]) == 64
    assert len(result["legal_action_order_digest"]) == 64
    assert all(row["canonical_action_key"].startswith("rak1:") for row in result["actions"])


def test_t1_second_belief_values_ignore_realized_opponent_discard_identity():
    deck = create_deck(shuffle=False)
    hero = Board.from_rows(top=deck[:1], middle=deck[1:3], bottom=deck[3:5])
    opponent = Board.from_rows(
        top=deck[5:7], middle=deck[7:9], bottom=deck[9:12]
    )
    dealt = tuple(deck[12:15])

    def evaluate(opponent_discard: str):
        world = WorldState(
            boards=(opponent, hero),
            private_discards=((opponent_discard,), ()),
            street="T1",
            next_player=1,
        )
        observation = world.observe(1, dealt)
        belief = sample_hidden_card_particles(
            observation,
            base_seed=2026071202,
            run_id="t1_second_hidden_identity_test",
            sample_count=2,
        )
        result = evaluate_turn1_action_subset(
            board=hero,
            opponent_board=opponent,
            dealt=dealt,
            hero_player=1,
            policies=[
                RegularAiPolicy(seed=21, seat="first"),
                RegularAiPolicy(seed=22, seat="second"),
            ],
            hand_seed=2026071202,
            sample_id=0,
            future_samples=2,
            action_indices=[0, 1],
            rng=random.Random(1),
            observation=observation,
            belief_batch=belief,
        )
        return observation, result

    first_observation, first = evaluate(deck[15])
    second_observation, second = evaluate(deck[16])
    assert first_observation == second_observation
    assert first["belief_batch_digest"] == second["belief_batch_digest"]
    assert [(row["canonical_action_key"], row["score"]) for row in first["actions"]] == [
        (row["canonical_action_key"], row["score"]) for row in second["actions"]
    ]


def test_turn1_teacher_policy_seeds_are_common_across_candidate_actions():
    class RecordingPolicy:
        def __init__(self, seat):
            self.seat = seat
            self.seeds = []

        def choose_action(self, board, dealt_cards, *, decision_seed=None, **_kwargs):
            self.seeds.append(decision_seed)
            return generate_actions(board, dealt_cards)[0]

    deck = create_deck(shuffle=False)
    board = Board.from_rows(top=deck[:1], middle=deck[1:3], bottom=deck[3:5])
    opponent_board = Board.from_rows(top=deck[5:6], middle=deck[6:8], bottom=deck[8:10])
    dealt = tuple(deck[10:13])
    remaining = tuple(deck[13:])
    policies = [RecordingPolicy("first"), RecordingPolicy("second")]

    result = evaluate_turn1_action_subset(
        board=board,
        opponent_board=opponent_board,
        dealt=dealt,
        hero_player=0,
        policies=policies,
        hand_seed=2026071201,
        sample_id=9,
        future_samples=1,
        action_indices=[0, 1],
        rng=random.Random(2026071201),
        **_belief_kwargs(
            board, opponent_board, dealt, sample_count=1, seed=2026071201
        ),
    )

    assert result["rng_schema"] == COUNTER_RNG_SCHEMA
    for policy in policies:
        half = len(policy.seeds) // 2
        assert half > 0
        assert policy.seeds[:half] == policy.seeds[half:]


def test_turn1_candidate_iteration_order_preserves_per_action_values():
    deck = create_deck(shuffle=False)
    board = Board.from_rows(top=deck[:1], middle=deck[1:3], bottom=deck[3:5])
    opponent = Board.from_rows(top=deck[5:6], middle=deck[6:8], bottom=deck[8:10])
    dealt = tuple(deck[10:13])
    world = WorldState(
        boards=(board, opponent),
        private_discards=((), ()),
        street="T1",
        next_player=0,
    )
    observation = world.observe(0, dealt)
    belief = sample_hidden_card_particles(
        observation,
        base_seed=2026071203,
        run_id="turn1_candidate_order",
        sample_count=2,
    )

    def evaluate(indices):
        return evaluate_turn1_action_subset(
            board=board,
            opponent_board=opponent,
            dealt=dealt,
            hero_player=0,
            policies=[
                RegularAiPolicy(seed=101, seat="first"),
                RegularAiPolicy(seed=102, seat="second"),
            ],
            hand_seed=2026071203,
            sample_id=1,
            future_samples=2,
            action_indices=indices,
            rng=random.Random(99),
            observation=observation,
            belief_batch=belief,
        )

    forward = evaluate([0, 1, 2])
    reverse = evaluate([2, 1, 0])
    forward_values = {
        row["canonical_action_key"]: row["score"] for row in forward["actions"]
    }
    reverse_values = {
        row["canonical_action_key"]: row["score"] for row in reverse["actions"]
    }
    assert reverse_values == forward_values


def test_evaluate_turn1_action_subset_can_use_candidate_model_topk():
    class ReverseIndexModel:
        def predict_sample(self, sample):
            return [float(index) for index, _action in enumerate(sample["actions"])]

    deck = create_deck(shuffle=False)
    board = Board.from_rows(top=deck[:1], middle=deck[1:3], bottom=deck[3:5])
    opponent_board = Board.from_rows(top=deck[5:6], middle=deck[6:8], bottom=deck[8:10])
    dealt = tuple(deck[10:13])
    remaining = tuple(deck[13:])
    policies = [
        RegularAiPolicy(seed=31, seat="first"),
        RegularAiPolicy(seed=32, seat="second"),
    ]

    result = evaluate_turn1_action_subset(
        board=board,
        opponent_board=opponent_board,
        dealt=dealt,
        hero_player=0,
        policies=policies,
        hand_seed=2026071102,
        sample_id=0,
        future_samples=1,
        candidate_model=ReverseIndexModel(),
        candidate_topk=3,
        rng=random.Random(2026071102),
        **_belief_kwargs(
            board, opponent_board, dealt, sample_count=1, seed=2026071102
        ),
    )

    selected = result["candidate_selector"]["selected_indices"]
    assert result["evaluated_action_count"] == 3
    assert result["actions_truncated"] is True
    assert selected == sorted(selected, reverse=True)
    assert {row["original_index"] for row in result["actions"]} == set(selected)


def test_evaluate_turn1_action_subset_can_use_candidate_model_union_topk():
    class FirstIndexModel:
        def predict_sample(self, sample):
            return [float(-index) for index, _action in enumerate(sample["actions"])]

    class ReverseIndexModel:
        def predict_sample(self, sample):
            return [float(index) for index, _action in enumerate(sample["actions"])]

    deck = create_deck(shuffle=False)
    board = Board.from_rows(top=deck[:1], middle=deck[1:3], bottom=deck[3:5])
    opponent_board = Board.from_rows(top=deck[5:6], middle=deck[6:8], bottom=deck[8:10])
    dealt = tuple(deck[10:13])
    remaining = tuple(deck[13:])
    policies = [
        RegularAiPolicy(seed=41, seat="first"),
        RegularAiPolicy(seed=42, seat="second"),
    ]

    result = evaluate_turn1_action_subset(
        board=board,
        opponent_board=opponent_board,
        dealt=dealt,
        hero_player=0,
        policies=policies,
        hand_seed=2026071103,
        sample_id=0,
        future_samples=1,
        candidate_models=[FirstIndexModel(), ReverseIndexModel()],
        candidate_topk=2,
        candidate_union_cap=3,
        rng=random.Random(2026071103),
        **_belief_kwargs(
            board, opponent_board, dealt, sample_count=1, seed=2026071103
        ),
    )

    selected = result["candidate_selector"]["selected_indices"]
    total_legal = result["total_legal_actions"]
    assert result["candidate_selector"]["mode"] == "candidate_model_union_topk"
    assert result["candidate_selector"]["model_count"] == 2
    assert result["evaluated_action_count"] == 3
    actions = generate_actions(board, dealt)
    rank_zero = sorted(
        (0, total_legal - 1), key=lambda index: action_key(actions[index]).sort_key()
    )
    rank_one = sorted(
        (1, total_legal - 2), key=lambda index: action_key(actions[index]).sort_key()
    )
    assert selected == [*rank_zero, rank_one[0]]
    assert {row["original_index"] for row in result["actions"]} == set(selected)


def test_evaluate_turn1_action_subset_can_use_candidate_union_mode():
    class LowIndexModel:
        def predict_sample(self, sample):
            return [
                100.0 if index == 0 else 90.0 if index == 1 else -1000.0
                for index, _action in enumerate(sample["actions"])
            ]

    class ThirdIndexModel:
        def predict_sample(self, sample):
            return [
                1000.0 if index == 2 else 1.0 if index == 3 else -1000.0
                for index, _action in enumerate(sample["actions"])
            ]

    deck = create_deck(shuffle=False)
    board = Board.from_rows(top=deck[:1], middle=deck[1:3], bottom=deck[3:5])
    opponent_board = Board.from_rows(top=deck[5:6], middle=deck[6:8], bottom=deck[8:10])
    dealt = tuple(deck[10:13])
    remaining = tuple(deck[13:])
    policies = [
        RegularAiPolicy(seed=43, seat="first"),
        RegularAiPolicy(seed=44, seat="second"),
    ]

    result = evaluate_turn1_action_subset(
        board=board,
        opponent_board=opponent_board,
        dealt=dealt,
        hero_player=0,
        policies=policies,
        hand_seed=2026071105,
        sample_id=0,
        future_samples=1,
        candidate_models=[LowIndexModel(), ThirdIndexModel()],
        candidate_topk=2,
        candidate_union_cap=2,
        candidate_union_mode="max_score",
        rng=random.Random(2026071105),
        **_belief_kwargs(
            board, opponent_board, dealt, sample_count=1, seed=2026071105
        ),
    )

    selected = result["candidate_selector"]["selected_indices"]
    assert result["candidate_selector"]["union_mode"] == "max_score"
    assert result["evaluated_action_count"] == 2
    assert selected == [2, 0]
    assert {row["original_index"] for row in result["actions"]} == set(selected)


def test_build_turn1_pilot_samples_fast_skip_preserves_target_state(monkeypatch):
    class FirstLegalPolicy:
        def __init__(self, *, seed, seat):
            self.seed = seed
            self.seat = seat

        def choose_action(self, board, dealt, **_kwargs):
            return generate_actions(board, dealt)[0]

    monkeypatch.setattr(pilot, "required_profiles", lambda *_args: set())
    monkeypatch.setattr(pilot, "load_model_bundle", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        pilot,
        "build_policy",
        lambda _profile, _bundle, *, seed, seat, opening_lookahead_samples: FirstLegalPolicy(
            seed=seed,
            seat=seat,
        ),
    )

    common = {
        "seed": 2026071104,
        "profile": "current",
        "opponent_profile": "current",
        "future_samples": 1,
        "max_actions": 1,
        "opening_lookahead_samples": 1,
        "source_bucket": "test_fast_skip",
    }
    full_rows, _full_summary = build_turn1_pilot_samples(samples=2, **common)
    fast_rows, fast_summary = build_turn1_pilot_samples(
        samples=1,
        skip_records=1,
        fast_skip_records=True,
        **common,
    )

    assert fast_summary["fast_skip_records"] is True
    assert fast_summary["generated_samples"] == 2
    target = full_rows[1]
    skipped = fast_rows[0]
    state_keys = [
        "sample_id",
        "hand_seed",
        "player",
        "seat",
        "board",
        "opponent_board",
        "dealt",
        "dead_cards",
        "visible_dead_cards",
        "hero_private_discards",
        "true_opponent_private_discards",
        "replay_truth",
        "total_legal_actions",
    ]
    for key in state_keys:
        assert skipped[key] == target[key]


def test_build_turn1_pilot_samples_can_collect_first_seat_only(monkeypatch):
    class FirstLegalPolicy:
        def __init__(self, *, seed, seat):
            self.seed = seed
            self.seat = seat

        def choose_action(self, board, dealt, **_kwargs):
            return generate_actions(board, dealt)[0]

    monkeypatch.setattr(pilot, "required_profiles", lambda *_args: set())
    monkeypatch.setattr(pilot, "load_model_bundle", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        pilot,
        "build_policy",
        lambda _profile, _bundle, *, seed, seat, opening_lookahead_samples: FirstLegalPolicy(
            seed=seed,
            seat=seat,
        ),
    )

    rows, summary = build_turn1_pilot_samples(
        samples=3,
        seed=2026071601,
        profile="current",
        opponent_profile="current",
        future_samples=1,
        max_actions=1,
        opening_lookahead_samples=1,
        source_bucket="test_first_only",
        allowed_seats=("first",),
    )

    assert len(rows) == 3
    assert {row["seat"] for row in rows} == {"first"}
    assert summary["allowed_seats"] == ["first"]
    assert summary["attempts"] == 3
    assert all(row["dead_cards"] == row["visible_dead_cards"] for row in rows)
    assert all("true_dead_cards" in row for row in rows)
    assert all("opponent_private_discards" not in row for row in rows)
    assert all(row["replay_ready"] is True for row in rows)
