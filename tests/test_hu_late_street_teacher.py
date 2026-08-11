from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace

import pytest

from ofc_regular.action_key import action_key
from ofc_regular.action_space import Action, generate_turn_actions
from ofc_regular.hu_infoset import ActorObservation, ScoringContext
from ofc_regular.hu_late_street_teacher import (
    T4SearchConfig,
    build_t4_future_plan,
    evaluate_t4_sequential_actions,
    evaluate_t4_sequential_batch,
    score_realized_t4_first_action,
)
from ofc_regular.state import Board
from ofc_regular.teacher import terminal_score


# The FL EV frozen into the cross-language golden vectors below.  It is the
# superseded June constant on purpose: those vectors pin an RNG contract that
# must not move when the economics are re-measured.
_RUST_FIXTURE_FL_EV_14 = 10.227020614683454

_FIVE_DEAL_SUPPORT = (
    ("3d", "Kh", "2h"),
    ("2s", "3h", "9c"),
    ("6d", "Jc", "4c"),
    ("Js", "8c", "Ad"),
    ("3c", "Ts", "Kc"),
)


def _first_observation() -> ActorObservation:
    return ActorObservation(
        hero_board=Board.from_rows(
            top=("4d", "5c"),
            middle=("Jh", "Td", "6h", "8h", "As"),
            bottom=("2c", "Jd", "Kd", "Th"),
        ),
        opponent_public_board=Board.from_rows(
            top=("Tc", "Qc"),
            middle=("7d", "3s", "6c", "8s", "9h"),
            bottom=("9d", "7c", "Qs", "7s"),
        ),
        dealt_cards=("9s", "Ah", "4s"),
        hero_private_discards=("Ac", "2d", "5d"),
        seat="first",
        street="T4",
        to_act_order="first",
    )


def _small_full_exact_observation() -> ActorObservation:
    """Counterexample with only three actions per player for a fast C(24, 3) test."""

    return ActorObservation(
        hero_board=Board.from_rows(
            top=("Kh",),
            middle=("7d", "Jh", "Ac", "6s", "3c"),
            bottom=("Qs", "Qc", "Ks", "As", "Js"),
        ),
        opponent_public_board=Board.from_rows(
            top=("Ad",),
            middle=("3s", "8c", "4c", "7c", "6d"),
            bottom=("Td", "3d", "9d", "8d", "6h"),
        ),
        dealt_cards=("4h", "2c", "Ah"),
        hero_private_discards=("Tc", "7h", "Th"),
        seat="first",
        street="T4",
        to_act_order="first",
    )


def _second_observation() -> ActorObservation:
    return ActorObservation(
        hero_board=Board.from_rows(
            top=("Qh", "Kc"),
            middle=("Ah", "Ac", "4s", "5s"),
            bottom=("Td", "7h", "7s", "7c", "Th"),
        ),
        opponent_public_board=Board.from_rows(
            top=("2h", "3h", "4h"),
            middle=("5c", "2s", "6h", "4c", "8c"),
            bottom=("7d", "Jd", "9d", "Qd", "Kd"),
        ),
        dealt_cards=("2c", "3c", "4d"),
        hero_private_discards=("6c", "8h", "9c"),
        seat="second",
        street="T4",
        to_act_order="second",
    )


def _non_clairvoyance_observation() -> ActorObservation:
    return ActorObservation(
        hero_board=Board.from_rows(
            top=("Kc", "Jd"),
            middle=("2s", "Td", "5d", "As", "4c"),
            bottom=("Ks", "5s", "6s", "9h"),
        ),
        opponent_public_board=Board.from_rows(
            top=("2d", "Qc"),
            middle=("Ac", "3s", "Ah", "Qh", "4h"),
            bottom=("8c", "3c", "8s", "7s"),
        ),
        dealt_cards=("5c", "Kd", "Js"),
        hero_private_discards=("4s", "9d", "2c"),
        seat="first",
        street="T4",
        to_act_order="first",
    )


def _independent_first_values(
    observation: ActorObservation,
    deals: Iterable[Sequence[str]],
) -> tuple[list[Action], list[float]]:
    """Small reference tree that does not call any late-street helper."""

    actions = generate_turn_actions(
        observation.hero_board, observation.dealt_cards
    )
    support = tuple(tuple(deal) for deal in deals)
    values: list[float] = []
    for action in actions:
        hero_final = observation.hero_board.place(action.placements)
        per_deal: list[float] = []
        for opponent_deal in support:
            response_values = [
                terminal_score(
                    hero_final,
                    observation.opponent_public_board.place(response.placements),
                )[0]
                for response in generate_turn_actions(
                    observation.opponent_public_board, opponent_deal
                )
            ]
            per_deal.append(min(response_values))
        values.append(sum(per_deal) / len(per_deal))
    return actions, values


def _canonical_best_key(actions: Sequence[Action], values: Sequence[float]) -> str:
    index = min(
        range(len(actions)),
        key=lambda item: (-float(values[item]), action_key(actions[item]).sort_key()),
    )
    return action_key(actions[index]).to_token()


def _action_for_key(
    observation: ActorObservation, action_token: str
) -> Action:
    for action in generate_turn_actions(
        observation.hero_board, observation.dealt_cards
    ):
        if action_key(action).to_token() == action_token:
            return action
    raise AssertionError("result selected an action outside the legal set")


def test_first_seat_explicit_support_matches_independent_sequential_oracle():
    observation = _first_observation()
    actions, expected_values = _independent_first_values(
        observation, _FIVE_DEAL_SUPPORT
    )

    result = evaluate_t4_sequential_actions(
        observation,
        candidate_deals=_FIVE_DEAL_SUPPORT,
        evaluation_deals=_FIVE_DEAL_SUPPORT,
    )

    expected_by_key = {
        action_key(action).to_token(): value
        for action, value in zip(actions, expected_values)
    }
    actual_by_key = {row["action_key"]: row["score"] for row in result["actions"]}
    assert actual_by_key == pytest.approx(expected_by_key, rel=0.0, abs=1e-12)
    assert result["selected_action_key"] == _canonical_best_key(
        actions, expected_values
    )
    assert result["selected_action_key"] == action_key(
        Action(
            placements=(("9s", "top"), ("Ah", "bottom")),
            discards=("4s",),
        )
    ).to_token()
    assert result["best_score"] == pytest.approx(6.0)

    standalone_values = [
        terminal_score(observation.hero_board.place(action.placements))[0]
        for action in actions
    ]
    assert _canonical_best_key(actions, standalone_values) != result[
        "selected_action_key"
    ]


def test_first_seat_full_exact_uniform_marginal_selects_counterexample_action():
    observation = _small_full_exact_observation()
    expected = Action(
        placements=(("4h", "top"), ("2c", "top")),
        discards=("Ah",),
    )

    result = evaluate_t4_sequential_actions(observation)

    assert result["candidate_plan"]["mode"] == "exact_uniform_marginal"
    assert result["candidate_plan"]["future_count"] == 2024
    assert result["candidate_plan"] == result["evaluation_plan"]
    assert result["sample_independence"] == "not_applicable_exact_enumeration"
    assert result["selected_action_key"] == action_key(expected).to_token()
    assert result["best_score"] == pytest.approx(6.0)
    assert result["selection_score_gap"] == pytest.approx(6.0)

    actions = generate_turn_actions(
        observation.hero_board, observation.dealt_cards
    )
    standalone_values = [
        terminal_score(observation.hero_board.place(action.placements))[0]
        for action in actions
    ]
    assert _canonical_best_key(actions, standalone_values) != result[
        "selected_action_key"
    ]


def test_second_seat_matches_independent_terminal_brute_force():
    observation = _second_observation()
    actions = generate_turn_actions(
        observation.hero_board, observation.dealt_cards
    )
    expected_values = [
        terminal_score(
            observation.hero_board.place(action.placements),
            observation.opponent_public_board,
        )[0]
        for action in actions
    ]

    result = evaluate_t4_sequential_actions(observation)

    actual_by_key = {row["action_key"]: row["score"] for row in result["actions"]}
    expected_by_key = {
        action_key(action).to_token(): value
        for action, value in zip(actions, expected_values)
    }
    assert actual_by_key == pytest.approx(expected_by_key, rel=0.0, abs=1e-12)
    assert result["selected_action_key"] == _canonical_best_key(
        actions, expected_values
    )
    assert result["candidate_plan"] is None
    assert result["evaluation_plan"] is None
    assert result["sample_independence"] == "not_applicable_no_future_chance"


def test_candidate_and_evaluation_rng_streams_are_disjoint_and_deterministic():
    observation = _small_full_exact_observation()
    config = T4SearchConfig(
        candidate_samples=5,
        evaluation_samples=7,
        seed=20260713,
        run_id="late-street-rng-separation",
    )

    first = evaluate_t4_sequential_actions(observation, config=config)
    second = evaluate_t4_sequential_actions(observation, config=config)

    assert first == second
    assert first["sample_independence"] == "disjoint_counter_rng_domains"
    candidate = first["candidate_plan"]
    evaluation = first["evaluation_plan"]
    assert candidate["future_count"] == 5
    assert evaluation["future_count"] == 7
    assert candidate["future_digest"] != evaluation["future_digest"]
    assert set(candidate["rng_key_digests"]).isdisjoint(
        evaluation["rng_key_digests"]
    )


def test_t4_counter_sampling_has_a_portable_golden_vector():
    # The FL EV is part of the observation fingerprint, which keys the counter
    # stream, so this vector is pinned under the constant the matching Rust
    # fixture hardcodes (rust/hu_m3_engine/tests/candidate01_semantics.rs) --
    # not under whatever the production default happens to be today.  What is
    # being pinned here is the Fisher-Yates contract, not the economics.
    plan = build_t4_future_plan(
        replace(
            _small_full_exact_observation(),
            scoring=ScoringContext(fl_ev=((14, _RUST_FIXTURE_FL_EV_14),)),
        ),
        sample_count=3,
        seed=123,
        run_id="portable-golden",
        stream="candidate_selection",
    )

    assert plan.deals == (
        ("4s", "9s", "Ts"),
        ("6c", "4s", "9s"),
        ("8h", "Qd", "9s"),
    )
    assert plan.digest() == (
        "a17af1807de8625f8ed220f1962e3b65a312cde009d484da219a42b0d8194be6"
    )


def test_candidate_and_evaluation_seeds_are_independent_at_t4():
    observation = _small_full_exact_observation()
    common = {
        "candidate_samples": 4,
        "evaluation_samples": 4,
        "seed": 2026071300,
        "run_id": "late-street-independent-seeds",
    }

    baseline = evaluate_t4_sequential_actions(
        observation,
        config=T4SearchConfig(
            **common,
            candidate_seed=2026071301,
            evaluation_seed=2026071302,
        ),
    )
    changed_evaluation = evaluate_t4_sequential_actions(
        observation,
        config=T4SearchConfig(
            **common,
            candidate_seed=2026071301,
            evaluation_seed=2026071303,
        ),
    )
    changed_candidate = evaluate_t4_sequential_actions(
        observation,
        config=T4SearchConfig(
            **common,
            candidate_seed=2026071304,
            evaluation_seed=2026071302,
        ),
    )

    def candidate_projection(result):
        return {
            "plan": result["candidate_plan"],
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
            "plan": result["evaluation_plan"],
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
    assert changed_evaluation["evaluation_plan"] != baseline["evaluation_plan"]
    assert evaluation_projection(changed_candidate) == evaluation_projection(baseline)
    assert changed_candidate["candidate_plan"] != baseline["candidate_plan"]


def test_scalar_and_reference_batch_results_are_identical_for_both_seats():
    observations = [_small_full_exact_observation(), _second_observation()]
    config = T4SearchConfig(
        candidate_samples=3,
        evaluation_samples=4,
        seed=91,
        run_id="late-street-scalar-batch",
    )

    scalar = [
        evaluate_t4_sequential_actions(observation, config=config)
        for observation in observations
    ]
    batched = evaluate_t4_sequential_batch(observations, config=config)

    assert batched == scalar


def test_locked_first_seat_action_does_not_condition_on_realized_opponent_deal():
    observation = _non_clairvoyance_observation()
    support = (
        ("2h", "7c", "Th"),
        ("Qs", "Kh", "6d"),
        ("Qd", "Tc", "9c"),
    )
    result = evaluate_t4_sequential_actions(
        observation,
        candidate_deals=support,
        evaluation_deals=support,
    )
    locked = _action_for_key(observation, result["selected_action_key"])

    realized = [
        score_realized_t4_first_action(observation, locked, deal)
        for deal in support
    ]

    assert {row["locked_action_key"] for row in realized} == {
        result["selected_action_key"]
    }
    clairvoyant_keys = []
    actions = generate_turn_actions(
        observation.hero_board, observation.dealt_cards
    )
    for deal in support:
        _actions, values = _independent_first_values(observation, (deal,))
        assert [action_key(action) for action in _actions] == [
            action_key(action) for action in actions
        ]
        clairvoyant_keys.append(_canonical_best_key(actions, values))
    assert len(set(clairvoyant_keys)) > 1
    assert clairvoyant_keys[0] != result["selected_action_key"]


def test_teacher_rejects_raw_non_actor_observation_state():
    with pytest.raises(TypeError, match="requires ActorObservation"):
        evaluate_t4_sequential_actions(  # type: ignore[arg-type]
            {"board": {}, "opponent_board": {}, "dead_cards": ["As"]}
        )
