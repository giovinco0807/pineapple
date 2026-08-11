"""Information-set-safe sequential late-street teacher for HU regular OFC.

The first player on T4 must place before the second player's three cards are
known.  Under the uniform exchangeable hidden-card prior, those three cards
are marginally uniform over the 24 cards outside the first player's
observation.  Consequently the correct first-seat value is::

    max_a E[ min_b u(hero + a, opponent + b) ]

The expectation is outside the hero action and the exact opponent response is
inside each deal.  The second-seat T4 decision has no future chance node and is
the usual exhaustive terminal maximization.

This module is a Python correctness reference.  It is deliberately rooted in
``ActorObservation`` and has no API for raw dead cards, replay truth, or a
realized deck tail.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable, Sequence

from .action_key import (
    ACTION_KEY_SCHEMA,
    action_key,
    canonical_descending_indices,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from .action_space import Action, generate_turn_actions
from .cards import ALL_CARDS, validate_cards
from .counter_rng import COUNTER_RNG_SCHEMA, CounterRngKey
from .evaluator import score_board
from .hu_infoset import ActorObservation
from .state import Board
from .teacher import DEFAULT_FL_EV, terminal_score


T4_SEQUENTIAL_TEACHER_SCHEMA = "hu_t4_sequential_belief_v1"
T4_FUTURE_PLAN_SCHEMA = "hu_t4_future_plan_v1"
T4_SOLVER_ID = "t4_exchangeable_expectimax_exact_response_v1"
_CARD_INDEX = {card: index for index, card in enumerate(ALL_CARDS)}
_RNG_DOMAIN = 1 << 63
_ATTEMPT_BITS = 32


class LateStreetTeacherError(ValueError):
    """Raised when a late-street request violates its information set."""


@dataclass(frozen=True)
class T4SearchConfig:
    """Chance plan for first-seat T4.

    A sample count of zero means exhaustive marginal enumeration.  When both
    plans are exhaustive there is no sampling-selection bias, so separate
    plans are unnecessary.  Positive counts use disjoint counter-RNG domains.
    """

    candidate_samples: int = 0
    evaluation_samples: int = 0
    seed: int = 42
    candidate_seed: int | None = None
    evaluation_seed: int | None = None
    run_id: str = "hu-m2-t4"

    def __post_init__(self) -> None:
        for name in ("candidate_samples", "evaluation_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        for name in ("candidate_seed", "evaluation_seed"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise TypeError(f"{name} must be an integer or None")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("run_id must not be empty")


@dataclass(frozen=True)
class T4FuturePlan:
    deals: tuple[tuple[str, str, str], ...]
    mode: str
    stream: str
    root_fingerprint: str
    rng_key_digests: tuple[str, ...] = ()
    sample_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        deals = tuple(tuple(deal) for deal in self.deals)
        if not deals:
            raise ValueError("a T4 future plan requires at least one deal")
        for deal in deals:
            if len(deal) != 3:
                raise ValueError("every T4 future deal must contain three cards")
            validate_cards(deal)
        if len(self.rng_key_digests) != len(self.sample_indices):
            raise ValueError("RNG digest/sample-index length mismatch")
        if self.mode == "counter_mc" and len(self.rng_key_digests) != len(deals):
            raise ValueError("counter-MC plans require one RNG key per deal")
        object.__setattr__(self, "deals", deals)

    def digest(self) -> str:
        payload = {
            "schema": T4_FUTURE_PLAN_SCHEMA,
            "mode": self.mode,
            "stream": self.stream,
            "root_fingerprint": self.root_fingerprint,
            "deals": [list(deal) for deal in self.deals],
            "rng_key_digests": list(self.rng_key_digests),
            "sample_indices": list(self.sample_indices),
        }
        return _digest(payload)

    def to_dict(self, *, include_deals: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": T4_FUTURE_PLAN_SCHEMA,
            "mode": self.mode,
            "stream": self.stream,
            "root_fingerprint": self.root_fingerprint,
            "future_count": len(self.deals),
            "future_digest": self.digest(),
            "rng_key_digests": list(self.rng_key_digests),
            "sample_indices": list(self.sample_indices),
        }
        if include_deals:
            payload["deals"] = [list(deal) for deal in self.deals]
        return payload


def build_t4_future_plan(
    observation: ActorObservation,
    *,
    sample_count: int,
    seed: int,
    run_id: str,
    stream: str,
    explicit_deals: Iterable[Sequence[str]] | None = None,
) -> T4FuturePlan:
    """Build an exhaustive, counter-MC, or explicit T4 chance support."""

    _require_t4_first(observation)
    unknown = _unknown_cards(observation)
    unknown_set = set(unknown)
    if len(unknown) != 24:
        raise LateStreetTeacherError(
            f"T4 first uniform belief requires 24 unknown cards, got {len(unknown)}"
        )
    if explicit_deals is not None:
        deals = tuple(_canonical_deal(deal) for deal in explicit_deals)
        if not deals:
            raise ValueError("explicit_deals must not be empty")
        for deal in deals:
            if not set(deal).issubset(unknown_set):
                raise LateStreetTeacherError(
                    "explicit future deal contains an actor-visible card"
                )
        return T4FuturePlan(
            deals=deals,
            mode="explicit_support",
            stream=stream,
            root_fingerprint=observation.fingerprint(),
        )
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise TypeError("sample_count must be an integer")
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    if sample_count == 0:
        return T4FuturePlan(
            deals=tuple(combinations(unknown, 3)),
            mode="exact_uniform_marginal",
            stream=stream,
            root_fingerprint=observation.fingerprint(),
        )

    deals: list[tuple[str, str, str]] = []
    key_digests: list[str] = []
    indices: list[int] = []
    fingerprint = observation.fingerprint()
    for sample_index in range(sample_count):
        key = CounterRngKey(
            base_seed=seed,
            run_id=f"{run_id}:{stream}",
            phase="t4_common_future",
            sample_index=sample_index,
            actor="chance",
            street="T4",
            stream=stream,
            root_fingerprint=fingerprint,
        )
        deal = _counter_sample_deal(
            unknown,
            seed=seed,
            run_id=f"{run_id}:{stream}",
            stream=stream,
            sample_index=sample_index,
            root_fingerprint=fingerprint,
        )
        deals.append(deal)  # sampling with replacement across worlds is unbiased
        key_digests.append(_digest(key.payload()))
        indices.append(sample_index)
    return T4FuturePlan(
        deals=tuple(deals),
        mode="counter_mc",
        stream=stream,
        root_fingerprint=fingerprint,
        rng_key_digests=tuple(key_digests),
        sample_indices=tuple(indices),
    )


def evaluate_t4_sequential_actions(
    observation: ActorObservation,
    *,
    config: T4SearchConfig | None = None,
    candidate_deals: Iterable[Sequence[str]] | None = None,
    evaluation_deals: Iterable[Sequence[str]] | None = None,
    fl_ev: dict[int, float] | None = None,
) -> dict[str, Any]:
    """Evaluate every legal T4 action without exposing future private cards.

    For first seat, the action is locked from the candidate plan and its
    reported score is then calculated only on the evaluation plan.  For an
    exhaustive plan selection/evaluation separation is not applicable because
    the expectation is exact.  For second seat every action is terminal-exact.
    """

    if not isinstance(observation, ActorObservation):
        raise TypeError(
            "T4 sequential teacher requires ActorObservation; raw state/replay truth is forbidden"
        )
    if observation.street != "T4":
        raise LateStreetTeacherError("T4 sequential teacher requires a T4 observation")
    config = config or T4SearchConfig()
    fl_ev = fl_ev or DEFAULT_FL_EV
    actions = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    if not actions:
        raise LateStreetTeacherError("T4 observation has no legal actions")

    if observation.to_act_order == "second":
        return _evaluate_t4_second(observation, actions, fl_ev)
    _require_t4_first(observation)
    candidate_plan = build_t4_future_plan(
        observation,
        sample_count=config.candidate_samples,
        seed=(config.seed if config.candidate_seed is None else config.candidate_seed),
        run_id=config.run_id,
        stream="candidate_selection",
        explicit_deals=candidate_deals,
    )
    if (
        evaluation_deals is None
        and candidate_deals is None
        and config.candidate_samples == 0
        and config.evaluation_samples == 0
    ):
        evaluation_plan = candidate_plan
        independence = "not_applicable_exact_enumeration"
    else:
        evaluation_plan = build_t4_future_plan(
            observation,
            sample_count=config.evaluation_samples,
            seed=(
                config.seed
                if config.evaluation_seed is None
                else config.evaluation_seed
            ),
            run_id=config.run_id,
            stream="locked_evaluation",
            explicit_deals=evaluation_deals,
        )
        if (
            candidate_plan.mode == "explicit_support"
            or evaluation_plan.mode == "explicit_support"
        ):
            independence = "explicit_support_caller_responsibility"
        elif candidate_plan.mode != evaluation_plan.mode:
            independence = "exact_and_counter_mc_no_shared_rng"
        else:
            independence = "disjoint_counter_rng_domains"
    _assert_plan_separation(candidate_plan, evaluation_plan, independence)

    candidate_rows = _score_t4_first_actions(
        observation, actions, candidate_plan.deals, fl_ev
    )
    evaluation_rows = _score_t4_first_actions(
        observation, actions, evaluation_plan.deals, fl_ev
    )
    candidate_values = [float(row["score"]) for row in candidate_rows]
    evaluation_values = [float(row["score"]) for row in evaluation_rows]
    ranked = canonical_descending_indices(candidate_values, actions)
    selected_index = ranked[0]
    candidate_second = candidate_values[ranked[1]] if len(ranked) > 1 else candidate_values[selected_index]
    evaluation_best = max(evaluation_values)

    rows: list[dict[str, Any]] = []
    for sorted_index, original_index in enumerate(ranked):
        action = actions[original_index]
        row = _action_payload(action, original_index)
        row.update(evaluation_rows[original_index])
        row.update(
            {
                "sorted_index": sorted_index,
                "selection_score": candidate_values[original_index],
                "selection_future_count": len(candidate_plan.deals),
                "evaluation_future_count": len(evaluation_plan.deals),
                "evaluation_regret_vs_sample_best": evaluation_best
                - evaluation_values[original_index],
                "selected_by_candidate_plan": original_index == selected_index,
            }
        )
        rows.append(row)

    selected_eval = evaluation_values[selected_index]
    return {
        "schema": T4_SEQUENTIAL_TEACHER_SCHEMA,
        "solver_id": T4_SOLVER_ID,
        "street": "T4",
        "seat": observation.seat,
        "to_act_order": observation.to_act_order,
        "observation_fingerprint": observation.fingerprint(),
        "policy_observation": observation.to_dict(),
        "action_key_schema": ACTION_KEY_SCHEMA,
        "counter_rng_schema": COUNTER_RNG_SCHEMA,
        "legal_action_count": len(actions),
        "legal_action_set_digest": legal_action_set_digest(actions),
        "legal_action_order_digest": ordered_action_mapping_digest(actions),
        "selected_action_original_index": selected_index,
        "selected_action_key": action_key(actions[selected_index]).to_token(),
        "selected_action_evaluation_score": selected_eval,
        "best_score": selected_eval,
        "selection_score_gap": candidate_values[selected_index] - candidate_second,
        "evaluation_sample_best_score": evaluation_best,
        "evaluation_sample_regret_of_locked_selection": evaluation_best - selected_eval,
        "candidate_plan": candidate_plan.to_dict(include_deals=False),
        "evaluation_plan": evaluation_plan.to_dict(include_deals=False),
        "search_config": {
            "candidate_samples": config.candidate_samples,
            "evaluation_samples": config.evaluation_samples,
            "candidate_seed": (
                config.seed if config.candidate_seed is None else config.candidate_seed
            ),
            "evaluation_seed": (
                config.seed
                if config.evaluation_seed is None
                else config.evaluation_seed
            ),
            "run_id": config.run_id,
        },
        "sample_independence": independence,
        "actions": rows,
        "fl_ev": {str(key): float(value) for key, value in fl_ev.items()},
        "teacher_notes": {
            "value_equation": "max_a E_deal[min_response hero_terminal_score]",
            "reported_selection": "locked_on_candidate_plan_evaluated_only_on_evaluation_plan",
            "belief": "uniform_exchangeable_hidden_cards_v1",
            "mc_sampling_algorithm": "counter_partial_fisher_yates_v1",
            "opponent_action_conditioning": "uniform_restart_abstraction_not_policy_reweighted",
            "scope": "exact_sequential_T4_under_stated_belief_not_full_game_optimality",
        },
    }


def evaluate_t4_sequential_batch(
    observations: Sequence[ActorObservation],
    *,
    config: T4SearchConfig | None = None,
    fl_ev: dict[int, float] | None = None,
) -> list[dict[str, Any]]:
    """Reference batch API; intentionally identical to scalar evaluation."""

    return [
        evaluate_t4_sequential_actions(
            observation,
            config=config,
            fl_ev=fl_ev,
        )
        for observation in observations
    ]


def select_t4_action(
    observation: ActorObservation,
    *,
    config: T4SearchConfig | None = None,
    fl_ev: dict[int, float] | None = None,
) -> Action:
    """Return the locked information-set action from a T4 teacher result."""

    result = evaluate_t4_sequential_actions(observation, config=config, fl_ev=fl_ev)
    selected_key = result["selected_action_key"]
    actions = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    for action in actions:
        if action_key(action).to_token() == selected_key:
            return action
    raise RuntimeError("selected T4 ActionKey is not legal at its observation")


def score_realized_t4_first_action(
    observation: ActorObservation,
    action: Action,
    opponent_dealt_cards: Iterable[str],
    *,
    fl_ev: dict[int, float] | None = None,
) -> dict[str, Any]:
    """Realize chance after a locked first-seat action and solve response.

    This helper intentionally receives the realized opponent deal only after
    the first action has already been selected from ``observation``.
    """

    _require_t4_first(observation)
    fl_ev = fl_ev or DEFAULT_FL_EV
    legal = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    key = action_key(action)
    if all(action_key(candidate) != key for candidate in legal):
        raise LateStreetTeacherError("locked T4 action is not legal at observation")
    deal = _canonical_deal(opponent_dealt_cards)
    if not set(deal).issubset(set(_unknown_cards(observation))):
        raise LateStreetTeacherError("realized opponent deal contains an observed card")
    hero_final = observation.hero_board.place(action.placements)
    response = _best_opponent_response(
        hero_final, observation.opponent_public_board, deal, fl_ev
    )
    return {
        "locked_action_key": key.to_token(),
        "opponent_deal": list(deal),
        **response,
    }


def _evaluate_t4_second(
    observation: ActorObservation,
    actions: Sequence[Action],
    fl_ev: dict[int, float],
) -> dict[str, Any]:
    if observation.opponent_public_board.card_count() != 13:
        raise LateStreetTeacherError("T4 second requires a complete opponent board")
    values: list[float] = []
    stats: list[dict[str, Any]] = []
    for action in actions:
        board = observation.hero_board.place(action.placements)
        value, board_score = terminal_score(board, observation.opponent_public_board, fl_ev)
        values.append(float(value))
        stats.append(_terminal_stats(board_score, 1))
    ranked = canonical_descending_indices(values, actions)
    selected = ranked[0]
    second = values[ranked[1]] if len(ranked) > 1 else values[selected]
    rows: list[dict[str, Any]] = []
    for sorted_index, original_index in enumerate(ranked):
        row = _action_payload(actions[original_index], original_index)
        row.update(stats[original_index])
        row.update(
            {
                "sorted_index": sorted_index,
                "score": values[original_index],
                "joint_ev": values[original_index],
                "selection_score": values[original_index],
                "selected_by_candidate_plan": original_index == selected,
                "evaluation_regret_vs_sample_best": values[selected] - values[original_index],
            }
        )
        rows.append(row)
    return {
        "schema": T4_SEQUENTIAL_TEACHER_SCHEMA,
        "solver_id": "t4_second_terminal_exhaustive_v1",
        "street": "T4",
        "seat": observation.seat,
        "to_act_order": observation.to_act_order,
        "observation_fingerprint": observation.fingerprint(),
        "policy_observation": observation.to_dict(),
        "action_key_schema": ACTION_KEY_SCHEMA,
        "counter_rng_schema": COUNTER_RNG_SCHEMA,
        "legal_action_count": len(actions),
        "legal_action_set_digest": legal_action_set_digest(actions),
        "legal_action_order_digest": ordered_action_mapping_digest(actions),
        "selected_action_original_index": selected,
        "selected_action_key": action_key(actions[selected]).to_token(),
        "selected_action_evaluation_score": values[selected],
        "best_score": values[selected],
        "selection_score_gap": values[selected] - second,
        "evaluation_sample_best_score": values[selected],
        "evaluation_sample_regret_of_locked_selection": 0.0,
        "candidate_plan": None,
        "evaluation_plan": None,
        "search_config": None,
        "sample_independence": "not_applicable_no_future_chance",
        "actions": rows,
        "fl_ev": {str(key): float(value) for key, value in fl_ev.items()},
        "teacher_notes": {
            "value_equation": "max_a hero_terminal_score",
            "scope": "exhaustive_second_seat_T4",
        },
    }


def _score_t4_first_actions(
    observation: ActorObservation,
    actions: Sequence[Action],
    deals: Sequence[Sequence[str]],
    fl_ev: dict[int, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for action in actions:
        hero_final = observation.hero_board.place(action.placements)
        hero_board_score = score_board(
            hero_final.top, hero_final.middle, hero_final.bottom
        )
        score_sum = 0.0
        response_counts: dict[str, int] = {}
        for deal in deals:
            response = _best_opponent_response(
                hero_final, observation.opponent_public_board, deal, fl_ev
            )
            score_sum += float(response["hero_score"])
            response_key = str(response["opponent_action_key"])
            response_counts[response_key] = response_counts.get(response_key, 0) + 1
        mean = score_sum / len(deals)
        row = _terminal_stats(hero_board_score, len(deals))
        row.update(
            {
                "score": mean,
                "joint_ev": mean,
                "future_count": len(deals),
                "opponent_exact_response_count": len(deals),
                "opponent_response_action_counts": response_counts,
            }
        )
        rows.append(row)
    return rows


def _best_opponent_response(
    hero_final: Board,
    opponent_board: Board,
    opponent_deal: Sequence[str],
    fl_ev: dict[int, float],
) -> dict[str, Any]:
    responses = generate_turn_actions(opponent_board, opponent_deal)
    if not responses:
        raise LateStreetTeacherError("opponent T4 deal has no legal response")
    best: tuple[float, tuple[int, int, int, int], Action, Board] | None = None
    for response in responses:
        opponent_final = opponent_board.place(response.placements)
        hero_score, _ = terminal_score(hero_final, opponent_final, fl_ev)
        candidate = (
            float(hero_score),
            action_key(response).sort_key(),
            response,
            opponent_final,
        )
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    assert best is not None
    return {
        "hero_score": best[0],
        "opponent_action_key": action_key(best[2]).to_token(),
        "opponent_action": _action_payload(best[2], -1),
        "opponent_final_board": _board_payload(best[3]),
    }


def _terminal_stats(board_score: Any, future_count: int) -> dict[str, Any]:
    busted = int(bool(board_score.busted))
    fl_entry = int(bool(board_score.fl_entry.qualifies))
    return {
        "non_bust_future_count": (1 - busted) * future_count,
        "bust_count": busted * future_count,
        "bust_rate": float(busted),
        "fl_entry_count": fl_entry * future_count,
        "fl_entry_rate": float(fl_entry),
        "royalty_mean": float(board_score.total_royalty),
        "top_royalty_mean": float(board_score.top_royalty),
        "middle_royalty_mean": float(board_score.middle_royalty),
        "bottom_royalty_mean": float(board_score.bottom_royalty),
    }


def _action_payload(action: Action, original_index: int) -> dict[str, Any]:
    return {
        "original_index": original_index,
        "action_key": action_key(action).to_token(),
        "placements": [list(item) for item in action.placements],
        "discards": list(action.discards),
    }


def _board_payload(board: Board) -> dict[str, list[str]]:
    return {
        "top": list(board.top),
        "middle": list(board.middle),
        "bottom": list(board.bottom),
    }


def _unknown_cards(observation: ActorObservation) -> tuple[str, ...]:
    known = set(observation.known_unavailable_cards())
    return tuple(card for card in ALL_CARDS if card not in known)


def _canonical_deal(cards: Iterable[str]) -> tuple[str, str, str]:
    values = tuple(cards)
    if len(values) != 3:
        raise ValueError("future deal must contain exactly three cards")
    validate_cards(values)
    return tuple(sorted(values, key=_CARD_INDEX.__getitem__))  # type: ignore[return-value]


def _counter_sample_deal(
    cards: Sequence[str],
    *,
    seed: int,
    run_id: str,
    stream: str,
    sample_index: int,
    root_fingerprint: str,
) -> tuple[str, str, str]:
    """Portable partial Fisher-Yates sample driven only by counter keys."""

    available = list(cards)
    selected: list[str] = []
    for draw_index in range(3):
        offset = _counter_randbelow(
            len(available),
            seed=seed,
            run_id=run_id,
            stream=stream,
            sample_index=sample_index,
            root_fingerprint=root_fingerprint,
            draw_index=draw_index,
        )
        selected.append(available.pop(offset))
    return tuple(sorted(selected, key=_CARD_INDEX.__getitem__))  # type: ignore[return-value]


def _counter_randbelow(
    upper_bound: int,
    *,
    seed: int,
    run_id: str,
    stream: str,
    sample_index: int,
    root_fingerprint: str,
    draw_index: int,
) -> int:
    limit = _RNG_DOMAIN - (_RNG_DOMAIN % upper_bound)
    attempt = 0
    while True:
        value = CounterRngKey(
            base_seed=seed,
            run_id=run_id,
            phase="t4_common_future",
            sample_index=sample_index,
            actor="chance",
            street="T4",
            stream=stream,
            counter=(draw_index << _ATTEMPT_BITS) | attempt,
            root_fingerprint=root_fingerprint,
        ).seed()
        if value < limit:
            return value % upper_bound
        attempt += 1


def _require_t4_first(observation: ActorObservation) -> None:
    if not isinstance(observation, ActorObservation):
        raise TypeError("T4 search requires ActorObservation")
    if observation.street != "T4" or observation.to_act_order != "first":
        raise LateStreetTeacherError("first-seat T4 search requires T4/first observation")
    if observation.hero_board.card_count() != 11 or observation.opponent_public_board.card_count() != 11:
        raise LateStreetTeacherError("T4 first requires 11-card hero and opponent boards")


def _assert_plan_separation(
    candidate: T4FuturePlan,
    evaluation: T4FuturePlan,
    independence: str,
) -> None:
    if independence != "disjoint_counter_rng_domains":
        return
    overlap = set(candidate.rng_key_digests) & set(evaluation.rng_key_digests)
    if overlap:
        raise LateStreetTeacherError(
            "candidate-selection and evaluation RNG coordinates overlap"
        )


def _digest(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()
