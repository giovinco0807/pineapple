"""HU Turn1 pilot teacher using the accepted Turn2 P2 continuation."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from .action_space import Action, generate_turn_actions
from .action_key import (
    ACTION_KEY_SCHEMA,
    action_key,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from .ai_profiles import ModelPaths, build_policy, load_model_bundle, required_profiles
from .cards import create_deck
from .counter_rng import COUNTER_RNG_SCHEMA, policy_decision_seed
from .decision_trace import attach_replay_truth, capture_decision_log_positions
from .evaluate_matchups import PROFILE_CHOICES, board_to_json
from .hu_belief import HiddenCardParticleBatch, sample_hidden_card_particles
from .hu_infoset import ActorObservation, ReplayTruth, WorldState
from .hu_turn3_model import hu_policy_sample, load_hu_action_value_model
from .play_ai import _choose_from_observation, _visible_dead_cards_for
from .policy import action_to_json
from .state import Board
from .teacher import DEFAULT_FL_EV, terminal_score

CandidateUnionMode = Literal[
    "min_rank",
    "rank_sum",
    "reciprocal_rank_sum",
    "mean_score",
    "max_score",
    "mean_z_score",
    "max_z_score",
]

CANDIDATE_UNION_MODES: tuple[str, ...] = (
    "min_rank",
    "rank_sum",
    "reciprocal_rank_sum",
    "mean_score",
    "max_score",
    "mean_z_score",
    "max_z_score",
)


def _choose_policy_action(
    policy: Any,
    *,
    player: int,
    boards: Sequence[Board],
    private_discards: Sequence[Sequence[str]],
    dealt: Sequence[str],
    street: str,
    hand_id: str | int | None,
    game_id: str | int | None,
    decision_seed: int | None,
    attach_truth: bool,
) -> Action:
    world = WorldState(
        boards=(boards[0], boards[1]),
        private_discards=(
            tuple(private_discards[0]),
            tuple(private_discards[1]),
        ),
        street=street,  # type: ignore[arg-type]
        next_player=player,
    )
    observation = world.observe(player, dealt)
    positions = capture_decision_log_positions(policy) if attach_truth else ()
    action = _choose_from_observation(
        policy,
        observation,
        hand_id=hand_id,
        game_id=game_id,
        decision_seed=decision_seed,
    )
    if attach_truth:
        attach_replay_truth(
            positions,
            ReplayTruth.from_world(
                world,
                actor=player,
                observation=observation,
            ),
        )
    return action


def _seat_name(player: int) -> str:
    return "first" if player == 0 else "second"


def _action_key(action: Action) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    return (tuple(action.placements), tuple(action.discards))


def _row_key(cards: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(cards))


def _board_key(board: Board) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    return (_row_key(board.top), _row_key(board.middle), _row_key(board.bottom))


def _candidate_model_entries(
    candidate_model: Any | None,
    candidate_models: Sequence[Any] | None,
) -> list[tuple[str, Any]]:
    entries: list[tuple[str, Any]] = []
    if candidate_model is not None:
        entries.append(("candidate_model", candidate_model))
    for index, model in enumerate(candidate_models or ()):
        entries.append((f"candidate_model_{index}", model))
    return entries


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _z_scores(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    if variance <= 0.0:
        return [0.0 for _value in values]
    std = math.sqrt(variance)
    return [(value - mean) / std for value in values]


def _t2_state_profile_key(
    *,
    player: int,
    board: Board,
    opponent_board: Board,
    dealt: Sequence[str],
    visible_dead_cards: Sequence[str],
    private_discards: Sequence[Sequence[str]],
) -> str:
    payload = (
        "T2",
        _seat_name(player),
        _board_key(board),
        _board_key(opponent_board),
        _row_key(dealt),
        _row_key(visible_dead_cards),
        _row_key(private_discards[player]),
        _row_key(private_discards[1 - player]),
    )
    return json.dumps(payload, separators=(",", ":"))


def _record_duplicate_key(profile: dict[str, Any] | None, prefix: str, key: str) -> None:
    if profile is None:
        return
    counts_key = f"_{prefix}_key_counts"
    counts = profile.setdefault(counts_key, {})
    counts[key] = int(counts.get(key, 0)) + 1


def _duplicate_profile_stats(counts: dict[str, int]) -> dict[str, Any]:
    raw = sum(int(value) for value in counts.values())
    unique = len(counts)
    repeated = raw - unique
    return {
        "raw": raw,
        "unique": unique,
        "repeated": repeated,
        "duplicate_rate": (repeated / raw) if raw else 0.0,
        "max_occurrence": max(counts.values()) if counts else 0,
    }


def _extract_duplicate_profile_stats(profile: dict[str, Any]) -> dict[str, Any]:
    state_counts = profile.pop("_t2_state_key_counts", {})
    decision_counts = profile.pop("_t2_decision_key_counts", {})
    return {
        "t2_state_only": _duplicate_profile_stats(state_counts),
        "t2_state_plus_decision_seed": _duplicate_profile_stats(decision_counts),
    }


def _profile_add(profile: dict[str, Any] | None, key: str, value: float) -> None:
    if profile is None:
        return
    profile[key] = float(profile.get(key, 0.0)) + float(value)


def _profile_count(profile: dict[str, Any] | None, key: str, value: int = 1) -> None:
    if profile is None:
        return
    profile[key] = int(profile.get(key, 0)) + int(value)


def _select_candidate_pairs(
    *,
    board: Board,
    opponent_board: Board,
    dealt: Sequence[str],
    actions: Sequence[Action],
    visible_dead_cards: Sequence[str],
    seat: str,
    candidate_model: Any | None = None,
    candidate_models: Sequence[Any] | None = None,
    candidate_topk: int = 0,
    candidate_union_cap: int = 0,
    candidate_union_mode: CandidateUnionMode = "min_rank",
) -> tuple[list[tuple[int, Action]], dict[str, Any] | None]:
    entries = _candidate_model_entries(candidate_model, candidate_models)
    if not entries or candidate_topk <= 0:
        return list(enumerate(actions)), None
    if candidate_union_mode not in CANDIDATE_UNION_MODES:
        raise ValueError(f"unknown candidate union mode: {candidate_union_mode}")
    action_sort_keys = [action_key(action).sort_key() for action in actions]

    sample = hu_policy_sample(
        board,
        dealt,
        actions,
        opponent_board=opponent_board,
        dead_cards=visible_dead_cards,
        seat=seat,
        to_act_order=seat,
    )
    union_ranks: dict[int, int] = {}
    rank_by_model: list[dict[int, int]] = []
    score_by_model: list[dict[int, float]] = []
    z_score_by_model: list[dict[int, float]] = []
    per_model: list[dict[str, Any]] = []
    for model_index, (name, model) in enumerate(entries):
        predictions = model.predict_sample(sample)
        if len(predictions) != len(actions):
            raise ValueError(
                f"candidate model {name} prediction count mismatch: "
                f"expected {len(actions)}, got {len(predictions)}"
            )
        order = sorted(
            range(len(actions)),
            key=lambda index: (-float(predictions[index]), action_sort_keys[index]),
        )
        rank_by_model.append({action_index: rank for rank, action_index in enumerate(order)})
        score_by_model.append(
            {action_index: float(predictions[action_index]) for action_index in range(len(actions))}
        )
        z_values = _z_scores([float(value) for value in predictions])
        z_score_by_model.append(
            {action_index: float(z_values[action_index]) for action_index in range(len(actions))}
        )
        selected_indices = order[: min(candidate_topk, len(actions))]
        for rank, action_index in enumerate(selected_indices):
            previous = union_ranks.get(action_index)
            if previous is None or rank < previous:
                union_ranks[action_index] = rank
        per_model.append(
            {
                "model_index": model_index,
                "name": name,
                "topk": int(candidate_topk),
                "selected_indices": [int(index) for index in selected_indices],
                "selected_action_keys": [
                    action_key(actions[index]).to_token() for index in selected_indices
                ],
                "selected_scores": [float(predictions[index]) for index in selected_indices],
                "score_min": float(min(predictions)) if len(predictions) else 0.0,
                "score_max": float(max(predictions)) if len(predictions) else 0.0,
            }
        )

    if candidate_union_mode == "min_rank":
        sort_key = lambda index: (union_ranks[index], action_sort_keys[index])
    elif candidate_union_mode == "rank_sum":
        penalty_rank = len(actions)
        sort_key = lambda index: (
            sum(rank_map.get(index, penalty_rank) for rank_map in rank_by_model),
            union_ranks[index],
            action_sort_keys[index],
        )
    elif candidate_union_mode == "reciprocal_rank_sum":
        sort_key = lambda index: (
            -sum(1.0 / (rank_map[index] + 1.0) for rank_map in rank_by_model if index in rank_map),
            union_ranks[index],
            action_sort_keys[index],
        )
    elif candidate_union_mode == "mean_score":
        sort_key = lambda index: (
            -_mean([score_map[index] for score_map in score_by_model]),
            union_ranks[index],
            action_sort_keys[index],
        )
    elif candidate_union_mode == "max_score":
        sort_key = lambda index: (
            -max(score_map[index] for score_map in score_by_model),
            union_ranks[index],
            action_sort_keys[index],
        )
    elif candidate_union_mode == "mean_z_score":
        sort_key = lambda index: (
            -_mean([score_map[index] for score_map in z_score_by_model]),
            union_ranks[index],
            action_sort_keys[index],
        )
    elif candidate_union_mode == "max_z_score":
        sort_key = lambda index: (
            -max(score_map[index] for score_map in z_score_by_model),
            union_ranks[index],
            action_sort_keys[index],
        )
    else:
        raise ValueError(f"unknown candidate union mode: {candidate_union_mode}")

    ordered_indices = sorted(union_ranks, key=sort_key)
    if candidate_union_cap > 0:
        ordered_indices = ordered_indices[: min(candidate_union_cap, len(ordered_indices))]
    selector = {
        "mode": "candidate_model_topk" if len(entries) == 1 else "candidate_model_union_topk",
        "topk": int(candidate_topk),
        "union_cap": int(candidate_union_cap),
        "union_mode": candidate_union_mode,
        "model_count": len(entries),
        "selected_indices": [int(index) for index in ordered_indices],
        "selected_action_keys": [
            action_key(actions[index]).to_token() for index in ordered_indices
        ],
        "per_model": per_model,
    }
    if len(entries) == 1 and per_model:
        selector.update(
            {
                "selected_scores": per_model[0]["selected_scores"],
                "score_min": per_model[0]["score_min"],
                "score_max": per_model[0]["score_max"],
            }
        )
    return [(index, actions[index]) for index in ordered_indices], selector


def _rollout_after_turn1_action(
    *,
    hero_player: int,
    action: Action,
    boards: Sequence[Board],
    private_discards: Sequence[Sequence[str]],
    policies: Sequence[Any],
    future_cards: Sequence[str],
    hand_seed: int,
    sample_id: int,
    action_index: int,
    future_index: int,
    profile: dict[str, Any] | None = None,
) -> float:
    rollout_boards = [boards[0], boards[1]]
    rollout_discards = [list(private_discards[0]), list(private_discards[1])]
    rollout_boards[hero_player] = rollout_boards[hero_player].place(action.placements)
    rollout_discards[hero_player].extend(action.discards)

    cursor = 0
    if hero_player == 0:
        turn_order: list[tuple[int, int]] = [(1, 1)]
        turn_order.extend((round_index, player) for round_index in range(2, 5) for player in (0, 1))
    else:
        turn_order = [(round_index, player) for round_index in range(2, 5) for player in (0, 1)]

    for round_index, player in turn_order:
        dealt = tuple(future_cards[cursor : cursor + 3])
        cursor += 3
        visible_dead_cards = tuple(
            _visible_dead_cards_for(player, rollout_boards, rollout_discards)
        )
        decision_seed = policy_decision_seed(
            base_seed=hand_seed,
            run_id="hu_turn1_teacher_pilot",
            root_fingerprint=f"sample={sample_id}|hero={hero_player}",
            future_index=future_index,
            actor=player,
            street=f"T{round_index}",
            decision_ordinal=round_index * 2 + player,
        )
        if round_index == 2:
            state_key = _t2_state_profile_key(
                player=player,
                board=rollout_boards[player],
                opponent_board=rollout_boards[1 - player],
                dealt=dealt,
                visible_dead_cards=visible_dead_cards,
                private_discards=rollout_discards,
            )
            _record_duplicate_key(profile, "t2_state", state_key)
            _record_duplicate_key(profile, "t2_decision", f"{state_key}|seed={decision_seed}")
        choose_started = time.perf_counter()
        chosen = _choose_policy_action(
            policies[player],
            player=player,
            boards=rollout_boards,
            private_discards=rollout_discards,
            dealt=dealt,
            street=f"T{round_index}",
            hand_id=hand_seed,
            game_id=hand_seed,
            decision_seed=decision_seed,
            attach_truth=False,
        )
        choose_seconds = time.perf_counter() - choose_started
        _profile_add(profile, "choose_action_seconds", choose_seconds)
        _profile_add(profile, f"choose_action_T{round_index}_seconds", choose_seconds)
        _profile_count(profile, "choose_action_count")
        _profile_count(profile, f"choose_action_T{round_index}_count")
        rollout_boards[player] = rollout_boards[player].place(chosen.placements)
        rollout_discards[player].extend(chosen.discards)

    terminal_started = time.perf_counter()
    hero_score, _ = terminal_score(
        rollout_boards[hero_player],
        rollout_boards[1 - hero_player],
        fl_ev=DEFAULT_FL_EV,
    )
    _profile_add(profile, "terminal_score_seconds", time.perf_counter() - terminal_started)
    _profile_count(profile, "terminal_score_count")
    return float(hero_score)


def evaluate_turn1_actions(
    *,
    board: Board,
    opponent_board: Board,
    dealt: Sequence[str],
    remaining_cards: Sequence[str],
    private_discards: Sequence[Sequence[str]],
    hero_player: int,
    policies: Sequence[Any],
    hand_seed: int,
    sample_id: int,
    future_samples: int,
    max_actions: int = 0,
    candidate_model: Any | None = None,
    candidate_models: Sequence[Any] | None = None,
    candidate_topk: int = 0,
    candidate_union_cap: int = 0,
    candidate_union_mode: CandidateUnionMode = "min_rank",
    rng: random.Random,
    profile: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Evaluate T1 from an actor observation and exchangeable hidden-card belief.

    ``remaining_cards`` and the opponent side of ``private_discards`` are
    retained only for source compatibility; neither can condition labels.
    """
    _ = remaining_cards
    seat = _seat_name(hero_player)
    observation = ActorObservation(
        hero_board=board,
        opponent_public_board=opponent_board,
        dealt_cards=tuple(dealt),
        hero_private_discards=tuple(private_discards[hero_player]),
        seat=seat,  # type: ignore[arg-type]
        street="T1",
        to_act_order=seat,  # type: ignore[arg-type]
    )
    belief_batch = sample_hidden_card_particles(
        observation,
        base_seed=hand_seed,
        run_id=f"hu_turn1_legacy_adapter|sample={sample_id}|player={hero_player}",
        sample_count=future_samples,
    )
    result = evaluate_turn1_action_subset(
        board=board,
        opponent_board=opponent_board,
        dealt=dealt,
        hero_player=hero_player,
        policies=policies,
        hand_seed=hand_seed,
        sample_id=sample_id,
        future_samples=future_samples,
        action_indices=None,
        max_actions=max_actions,
        candidate_model=candidate_model,
        candidate_models=candidate_models,
        candidate_topk=candidate_topk,
        candidate_union_cap=candidate_union_cap,
        candidate_union_mode=candidate_union_mode,
        rng=rng,
        profile=profile,
        observation=observation,
        belief_batch=belief_batch,
    )
    return list(result["actions"]), bool(result["actions_truncated"])


def evaluate_turn1_action_subset(
    *,
    board: Board,
    opponent_board: Board,
    dealt: Sequence[str],
    hero_player: int,
    policies: Sequence[Any],
    hand_seed: int,
    sample_id: int,
    future_samples: int,
    action_indices: Sequence[int] | None = None,
    max_actions: int = 0,
    paired_delta_candidate_index: int | None = None,
    paired_delta_baseline_index: int | None = None,
    candidate_model: Any | None = None,
    candidate_models: Sequence[Any] | None = None,
    candidate_topk: int = 0,
    candidate_union_cap: int = 0,
    candidate_union_mode: CandidateUnionMode = "min_rank",
    rng: random.Random,
    profile: dict[str, Any] | None = None,
    observation: ActorObservation | None = None,
    belief_batch: HiddenCardParticleBatch | None = None,
) -> dict[str, Any]:
    legal_started = time.perf_counter()
    all_actions = generate_turn_actions(board, dealt)
    total_legal_actions = len(all_actions)
    _profile_add(profile, "turn1_legal_action_generation_seconds", time.perf_counter() - legal_started)
    _profile_count(profile, "turn1_legal_action_generation_count")
    boards = [Board.from_rows(), Board.from_rows()]
    boards[hero_player] = board
    boards[1 - hero_player] = opponent_board
    if observation is None or belief_batch is None:
        raise ValueError(
            "T1 teacher requires ActorObservation and HiddenCardParticleBatch"
        )
    expected_seat = _seat_name(hero_player)
    if (
        observation.hero_board != board
        or observation.opponent_public_board != opponent_board
        or observation.dealt_cards != tuple(dealt)
        or observation.seat != expected_seat
        or observation.street != "T1"
    ):
        raise ValueError("belief root disagrees with the Turn1 decision state")
    belief_batch.validate_against(observation)
    visible_dead_cards = observation.legacy_dead_cards()
    candidate_selector: dict[str, Any] | None = None
    if action_indices is None and _candidate_model_entries(candidate_model, candidate_models) and candidate_topk > 0:
        candidate_started = time.perf_counter()
        selected_pairs, candidate_selector = _select_candidate_pairs(
            board=board,
            opponent_board=opponent_board,
            dealt=dealt,
            actions=all_actions,
            visible_dead_cards=visible_dead_cards,
            seat=_seat_name(hero_player),
            candidate_model=candidate_model,
            candidate_models=candidate_models,
            candidate_topk=candidate_topk,
            candidate_union_cap=candidate_union_cap,
            candidate_union_mode=candidate_union_mode,
        )
        _profile_add(
            profile,
            "turn1_candidate_selection_seconds",
            time.perf_counter() - candidate_started,
        )
        _profile_count(profile, "turn1_candidate_selection_count")
    elif action_indices is None:
        selected_pairs = list(enumerate(all_actions))
        if max_actions > 0:
            selected_pairs = selected_pairs[:max_actions]
    else:
        seen: set[int] = set()
        selected_pairs = []
        for raw_index in action_indices:
            index = int(raw_index)
            if index in seen or index < 0 or index >= total_legal_actions:
                continue
            seen.add(index)
            selected_pairs.append((index, all_actions[index]))
    cards_needed = 21 if hero_player == 0 else 18
    future_started = time.perf_counter()
    if len(belief_batch.particles) != future_samples:
        raise ValueError("belief particle count must equal future_samples")
    rollout_roots: list[tuple[tuple[str, ...], Sequence[Sequence[str]]]] = []
    for particle in belief_batch.particles:
        particle_discards: list[list[str]] = [[], []]
        particle_discards[hero_player] = list(observation.hero_private_discards)
        particle_discards[1 - hero_player] = list(
            particle.opponent_private_discards
        )
        rollout_roots.append((particle.draw(cards_needed), particle_discards))
    _profile_add(profile, "future_sampling_seconds", time.perf_counter() - future_started)
    _profile_count(profile, "future_sampling_count")
    evaluated: list[dict[str, Any]] = []
    scores_by_index: dict[int, list[float]] = {}
    for original_index, action in selected_pairs:
        action_started = time.perf_counter()
        scores = []
        for future_index, (future_cards, rollout_discards) in enumerate(rollout_roots):
            rollout_started = time.perf_counter()
            scores.append(
                _rollout_after_turn1_action(
                    hero_player=hero_player,
                    action=action,
                    boards=boards,
                    private_discards=rollout_discards,
                    policies=policies,
                    future_cards=future_cards,
                    hand_seed=hand_seed,
                    sample_id=sample_id,
                    action_index=original_index,
                    future_index=future_index,
                    profile=profile,
                )
            )
            _profile_add(profile, "rollout_seconds", time.perf_counter() - rollout_started)
            _profile_count(profile, "rollout_count")
        scores_by_index[original_index] = scores
        mean = sum(scores) / len(scores)
        variance = sum((score - mean) ** 2 for score in scores) / max(len(scores) - 1, 1)
        action_seconds = time.perf_counter() - action_started
        _profile_add(profile, "turn1_action_eval_seconds", action_seconds)
        _profile_count(profile, "turn1_action_eval_count")
        payload = action_to_json(board, action)
        payload.update(
            {
                "action_index": original_index,
                "original_index": original_index,
                "canonical_action_key": action_key(action).to_token(),
                "score": mean,
                "ev": mean,
                "se": math.sqrt(variance / len(scores)),
                "rollout_count": len(scores),
                "action_eval_seconds": action_seconds,
            }
        )
        evaluated.append(payload)

    evaluated.sort(
        key=lambda item: (-float(item["score"]), str(item["canonical_action_key"]))
    )
    result: dict[str, Any] = {
        "actions": evaluated,
        "actions_truncated": (
            (max_actions > 0 and total_legal_actions > len(selected_pairs))
            or (action_indices is not None and len(selected_pairs) < total_legal_actions)
            or (candidate_selector is not None and len(selected_pairs) < total_legal_actions)
        ),
        "total_legal_actions": total_legal_actions,
        "evaluated_action_count": len(selected_pairs),
        "action_key_schema": ACTION_KEY_SCHEMA,
        "legal_action_set_digest": legal_action_set_digest(all_actions),
        "legal_action_order_digest": ordered_action_mapping_digest(all_actions),
        "rng_schema": COUNTER_RNG_SCHEMA,
        "policy_seed_common_across_actions": True,
    }
    result.update(
        {
            "belief_conditioned": True,
            "observation_fingerprint": observation.fingerprint(),
            "belief_batch_digest": belief_batch.digest(),
            "belief_schema": belief_batch.to_dict()["belief_schema"],
            "belief_prior": belief_batch.prior,
        }
    )
    if candidate_selector is not None:
        result["candidate_selector"] = candidate_selector
    if (
        paired_delta_candidate_index is not None
        and paired_delta_baseline_index is not None
        and paired_delta_candidate_index in scores_by_index
        and paired_delta_baseline_index in scores_by_index
    ):
        candidate_scores = scores_by_index[paired_delta_candidate_index]
        baseline_scores = scores_by_index[paired_delta_baseline_index]
        deltas = [
            float(candidate) - float(baseline)
            for candidate, baseline in zip(candidate_scores, baseline_scores)
        ]
        if deltas:
            mean_delta = sum(deltas) / len(deltas)
            variance = sum((delta - mean_delta) ** 2 for delta in deltas) / max(len(deltas) - 1, 1)
            result.update(
                {
                    "paired_delta_candidate_index": int(paired_delta_candidate_index),
                    "paired_delta_baseline_index": int(paired_delta_baseline_index),
                    "paired_delta_mean": mean_delta,
                    "paired_delta_standard_error": math.sqrt(variance / len(deltas)),
                    "paired_delta_count": len(deltas),
                    "paired_delta_summary": {
                        "candidate_index": int(paired_delta_candidate_index),
                        "baseline_index": int(paired_delta_baseline_index),
                        "mean": mean_delta,
                        "standard_error": math.sqrt(variance / len(deltas)),
                        "count": len(deltas),
                    },
                }
            )
    return result


def build_turn1_pilot_samples(
    *,
    samples: int,
    skip_records: int = 0,
    seed: int,
    profile: str,
    opponent_profile: str,
    future_samples: int,
    max_actions: int,
    candidate_model_path: Path | None = None,
    candidate_model_paths: Sequence[Path] | None = None,
    candidate_topk: int = 0,
    candidate_union_cap: int = 0,
    candidate_union_mode: CandidateUnionMode = "min_rank",
    opening_lookahead_samples: int,
    source_bucket: str = "natural_mc32",
    collect_topk_log: bool = False,
    fast_skip_records: bool = False,
    allowed_seats: Sequence[str] = ("first", "second"),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    allowed_seat_set = frozenset(str(seat) for seat in allowed_seats)
    if not allowed_seat_set or not allowed_seat_set <= {"first", "second"}:
        raise ValueError("allowed_seats must contain first and/or second")
    profile_stats: dict[str, Any] = {}
    model_started = time.perf_counter()
    profiles = required_profiles(profile, opponent_profile)
    bundle = load_model_bundle(ModelPaths(), profiles)
    profile_stats["model_load_seconds"] = time.perf_counter() - model_started
    policy_started = time.perf_counter()
    policies = [
        build_policy(
            profile,
            bundle,
            seed=seed,
            seat="first",
            opening_lookahead_samples=opening_lookahead_samples,
        ),
        build_policy(
            opponent_profile,
            bundle,
            seed=seed + 1,
            seat="second",
            opening_lookahead_samples=opening_lookahead_samples,
        ),
    ]
    profile_stats["policy_build_seconds"] = time.perf_counter() - policy_started
    candidate_model = None
    candidate_models: list[Any] = []
    if candidate_model_path is not None:
        candidate_started = time.perf_counter()
        candidate_model = load_hu_action_value_model(candidate_model_path)
        profile_stats["candidate_model_load_seconds"] = time.perf_counter() - candidate_started
        profile_stats["candidate_model_path"] = str(candidate_model_path)
        profile_stats["candidate_topk"] = int(candidate_topk)
        profile_stats["candidate_union_mode"] = candidate_union_mode
    if candidate_model_paths:
        candidate_started = time.perf_counter()
        candidate_models = [load_hu_action_value_model(path) for path in candidate_model_paths]
        profile_stats["candidate_models_load_seconds"] = time.perf_counter() - candidate_started
        profile_stats["candidate_model_paths"] = [str(path) for path in candidate_model_paths]
        profile_stats["candidate_topk"] = int(candidate_topk)
        profile_stats["candidate_union_cap"] = int(candidate_union_cap)
        profile_stats["candidate_union_mode"] = candidate_union_mode
    topk_decision_log: list[dict[str, Any]] = []
    if collect_topk_log:
        for policy in policies:
            if hasattr(policy, "topk_decision_log"):
                policy.topk_decision_log = topk_decision_log
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    target_rows = samples + skip_records
    generated_rows = 0
    attempts = 0
    started = time.perf_counter()

    while generated_rows < target_rows:
        attempts += 1
        hand_seed = seed + attempts - 1
        hand_rng = random.Random(hand_seed)
        deck = create_deck(shuffle=True, rng=hand_rng)
        cursor = 0
        boards = [Board.from_rows(), Board.from_rows()]
        private_discards: list[list[str]] = [[], []]

        for player in (0, 1):
            dealt0 = tuple(deck[cursor : cursor + 5])
            cursor += 5
            opening_started = time.perf_counter()
            opening_action = _choose_policy_action(
                policies[player],
                player=player,
                boards=boards,
                private_discards=private_discards,
                dealt=dealt0,
                street="T0",
                hand_id=hand_seed,
                game_id=hand_seed,
                decision_seed=hand_seed,
                attach_truth=True,
            )
            _profile_add(profile_stats, "opening_choose_seconds", time.perf_counter() - opening_started)
            _profile_count(profile_stats, "opening_choose_count")
            boards[player] = boards[player].place(opening_action.placements)
            private_discards[player].extend(opening_action.discards)

        for player in (0, 1):
            if generated_rows >= target_rows:
                break
            seat = _seat_name(player)
            should_collect = seat in allowed_seat_set
            dealt1 = tuple(deck[cursor : cursor + 3])
            cursor += 3
            should_evaluate = should_collect and ((not fast_skip_records) or generated_rows >= skip_records)
            if should_evaluate:
                world = WorldState(
                    boards=(boards[0], boards[1]),
                    private_discards=(
                        tuple(private_discards[0]),
                        tuple(private_discards[1]),
                    ),
                    street="T1",
                    next_player=player,
                )
                observation = world.observe(player, dealt1)
                replay_truth = ReplayTruth.from_world(
                    world, actor=player, observation=observation
                )
                belief_batch = sample_hidden_card_particles(
                    observation,
                    base_seed=hand_seed,
                    run_id=(
                        f"hu_turn1_teacher_pilot|sample={generated_rows}|player={player}"
                    ),
                    sample_count=future_samples,
                )
                result = evaluate_turn1_action_subset(
                    board=boards[player],
                    opponent_board=boards[1 - player],
                    dealt=dealt1,
                    hero_player=player,
                    policies=policies,
                    hand_seed=hand_seed,
                    sample_id=generated_rows,
                    future_samples=future_samples,
                    max_actions=max_actions,
                    candidate_model=candidate_model,
                    candidate_models=candidate_models,
                    candidate_topk=candidate_topk,
                    candidate_union_cap=candidate_union_cap,
                    candidate_union_mode=candidate_union_mode,
                    rng=rng,
                    profile=profile_stats,
                    observation=observation,
                    belief_batch=belief_batch,
                )
                action_rows = list(result["actions"])
                truncated = bool(result["actions_truncated"])
                if not action_rows:
                    continue
                best = action_rows[0]
                second = action_rows[1] if len(action_rows) > 1 else None
                score_gap = best["score"] - (second["score"] if second else best["score"])
                rows.append(
                    {
                        "sample_id": generated_rows,
                        "rule_set": "regular",
                        "schema": "hu_turn1_stage1_pilot_v1",
                        "phase": "hu_turn1_5card",
                        "source_bucket": source_bucket,
                        "source_bucket_group": "natural",
                        "profile": profile,
                        "opponent_profile": opponent_profile,
                        "t2_continuation_profile": profile,
                        "t3_continuation": "stage7_m5_r10",
                        "hand_seed": hand_seed,
                        "player": player,
                        "seat": seat,
                        "board": board_to_json(boards[player]),
                        "opponent_board": board_to_json(boards[1 - player]),
                        "dealt": list(dealt1),
                        "policy_observation": observation.to_dict(),
                        "dead_cards": list(_visible_dead_cards_for(player, boards, private_discards)),
                        "visible_dead_cards": list(_visible_dead_cards_for(player, boards, private_discards)),
                        "hero_private_discards": list(private_discards[player]),
                        "artifact_visibility_schema": "actor_observation_plus_replay_truth_v1",
                        **replay_truth.to_legacy_record_fields(),
                        "action_count": len(action_rows),
                        "total_legal_actions": result["total_legal_actions"],
                        "evaluated_action_count": result["evaluated_action_count"],
                        "action_key_schema": result["action_key_schema"],
                        "legal_action_set_digest": result["legal_action_set_digest"],
                        "legal_action_order_digest": result["legal_action_order_digest"],
                        "rng_schema": result["rng_schema"],
                        "policy_seed_common_across_actions": result[
                            "policy_seed_common_across_actions"
                        ],
                        "belief_conditioned": result["belief_conditioned"],
                        "observation_fingerprint": result["observation_fingerprint"],
                        "belief_batch_digest": result["belief_batch_digest"],
                        "belief_schema": result["belief_schema"],
                        "belief_prior": result["belief_prior"],
                        "candidate_model": str(candidate_model_path) if candidate_model_path else None,
                        "candidate_models": [str(path) for path in candidate_model_paths or ()],
                        "candidate_topk": int(candidate_topk),
                        "candidate_union_cap": int(candidate_union_cap),
                        "candidate_union_mode": candidate_union_mode,
                        "candidate_selector": result.get("candidate_selector"),
                        "actions_truncated": truncated,
                        "future_samples": future_samples,
                        "best_action": 0,
                        "score_gap": score_gap,
                        "actions": action_rows,
                    }
                )
            elif should_collect:
                _profile_count(profile_stats, "fast_skip_records")

            actual_t1_started = time.perf_counter()
            actual_action = _choose_policy_action(
                policies[player],
                player=player,
                boards=boards,
                private_discards=private_discards,
                dealt=dealt1,
                street="T1",
                hand_id=hand_seed,
                game_id=hand_seed,
                decision_seed=hand_seed,
                attach_truth=True,
            )
            _profile_add(profile_stats, "actual_t1_choose_seconds", time.perf_counter() - actual_t1_started)
            _profile_count(profile_stats, "actual_t1_choose_count")
            boards[player] = boards[player].place(actual_action.placements)
            private_discards[player].extend(actual_action.discards)
            if should_collect:
                generated_rows += 1

    elapsed = time.perf_counter() - started
    output_rows = rows if fast_skip_records else rows[skip_records:]
    duplicate_profile_stats = _extract_duplicate_profile_stats(profile_stats)
    topk_reason_counts = Counter(
        str(record.get("no_override_reason", "")) for record in topk_decision_log
    )
    summary = {
        "schema": "hu_turn1_stage1_pilot_summary_v1",
        "samples": len(output_rows),
        "generated_samples": int(generated_rows),
        "skip_records": int(skip_records),
        "fast_skip_records": bool(fast_skip_records),
        "allowed_seats": sorted(allowed_seat_set),
        "attempts": attempts,
        "profile": profile,
        "opponent_profile": opponent_profile,
        "source_bucket": source_bucket,
        "source_bucket_group": "natural",
        "future_samples": future_samples,
        "max_actions": max_actions,
        "candidate_model": str(candidate_model_path) if candidate_model_path else None,
        "candidate_models": [str(path) for path in candidate_model_paths or ()],
        "candidate_topk": int(candidate_topk),
        "candidate_union_cap": int(candidate_union_cap),
        "candidate_union_mode": candidate_union_mode,
        "truncated_samples": sum(1 for row in output_rows if row["actions_truncated"]),
        "mean_action_count": sum(row["action_count"] for row in output_rows) / len(output_rows),
        "elapsed_seconds": elapsed,
        "seconds_per_sample": elapsed / len(output_rows),
        "profile_stats": profile_stats,
        "duplicate_profile_stats": duplicate_profile_stats,
        "topk_decision_log_collected": collect_topk_log,
        "topk_decisions": len(topk_decision_log),
        "topk_overrides": sum(1 for record in topk_decision_log if record.get("override_fired")),
        "topk_no_override_reason_counts": dict(sorted(topk_reason_counts.items())),
    }
    return output_rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--skip-records", type=int, default=0)
    parser.add_argument("--fast-skip-records", action="store_true")
    parser.add_argument("--future-samples", type=int, default=4)
    parser.add_argument("--max-actions", type=int, default=0)
    parser.add_argument("--candidate-model", type=Path)
    parser.add_argument("--candidate-models", type=Path, nargs="+")
    parser.add_argument("--candidate-topk", type=int, default=0)
    parser.add_argument("--candidate-union-cap", type=int, default=0)
    parser.add_argument("--candidate-union-mode", choices=CANDIDATE_UNION_MODES, default="min_rank")
    parser.add_argument("--seed", type=int, default=2026062301)
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default="stage9f_p2")
    parser.add_argument("--opponent-profile", choices=PROFILE_CHOICES, default="stage9f_p2")
    parser.add_argument("--opening-lookahead-samples", type=int, default=1)
    parser.add_argument("--source-bucket", default="natural_mc32")
    parser.add_argument("--collect-topk-log", action="store_true")
    parser.add_argument("--seats", nargs="+", choices=("first", "second"), default=("first", "second"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    if args.skip_records < 0:
        raise SystemExit("--skip-records must be non-negative")
    if args.future_samples <= 0:
        raise SystemExit("--future-samples must be positive")
    if args.max_actions < 0:
        raise SystemExit("--max-actions must be non-negative")
    if args.candidate_topk < 0:
        raise SystemExit("--candidate-topk must be non-negative")
    if args.candidate_union_cap < 0:
        raise SystemExit("--candidate-union-cap must be non-negative")
    if args.candidate_model is not None and args.candidate_models:
        raise SystemExit("--candidate-model and --candidate-models are mutually exclusive")
    if args.candidate_model is not None and args.candidate_topk <= 0:
        raise SystemExit("--candidate-topk must be positive when --candidate-model is set")
    if args.candidate_models and args.candidate_topk <= 0:
        raise SystemExit("--candidate-topk must be positive when --candidate-models is set")
    rows, summary = build_turn1_pilot_samples(
        samples=args.samples,
        skip_records=args.skip_records,
        seed=args.seed,
        profile=args.profile,
        opponent_profile=args.opponent_profile,
        future_samples=args.future_samples,
        max_actions=args.max_actions,
        candidate_model_path=args.candidate_model,
        candidate_model_paths=args.candidate_models,
        candidate_topk=args.candidate_topk,
        candidate_union_cap=args.candidate_union_cap,
        candidate_union_mode=args.candidate_union_mode,
        opening_lookahead_samples=args.opening_lookahead_samples,
        source_bucket=args.source_bucket,
        collect_topk_log=args.collect_topk_log,
        allowed_seats=args.seats,
        fast_skip_records=args.fast_skip_records,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
