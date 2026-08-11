"""Generate HU T0 action-value teacher rows with a fixed downstream policy.

Every legal five-card opening placement is available to the evaluator.  All
actions in a state share the same sampled future card sequences, and nested
continuation decisions receive action-independent decision seeds so paired
action deltas retain common-random-number coupling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from .action_key import (
    ACTION_KEY_SCHEMA,
    action_key,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from .action_space import Action, generate_actions, generate_turn_actions
from .ai_profiles import ModelPaths, build_policy, load_model_bundle, required_profiles
from .cards import ALL_CARDS, create_deck, validate_cards
from .counter_rng import (
    COUNTER_RNG_SCHEMA,
    common_future_seed,
    policy_decision_seed,
)
from .evaluate_matchups import PROFILE_CHOICES, board_to_json
from .hu_turn1_teacher_pilot import (
    CANDIDATE_UNION_MODES,
    CandidateUnionMode,
    _candidate_model_entries,
    _profile_add,
    _profile_count,
    _seat_name,
    _select_candidate_pairs,
)
from .hu_infoset import ActorObservation
from .hu_turn0_candidate import load_turn0_candidate_model
from .hu_turn3_model import hu_policy_sample
from .play_ai import _choose_from_observation, _visible_dead_cards_for
from .policy import action_to_json
from .state import Board
from .teacher import DEFAULT_FL_EV, terminal_score


def _action_key(action: Action) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    return tuple(action.placements), tuple(action.discards)


def remaining_for_turn0_state(
    board: Board,
    dealt_cards: Iterable[str],
    opponent_board: Board,
    dead_cards: Iterable[str] = (),
) -> tuple[str, ...]:
    dealt = tuple(dealt_cards)
    dead = tuple(dead_cards)
    used_cards = (*board.all_cards(), *dealt, *opponent_board.all_cards(), *dead)
    validate_cards(used_cards)
    used = set(used_cards)
    return tuple(card for card in ALL_CARDS if card not in used)


def _future_sequences(
    remaining_cards: Sequence[str],
    *,
    cards_needed: int,
    future_samples: int,
    base_seed: int,
    run_id: str,
    root_fingerprint: str,
) -> list[tuple[str, ...]]:
    if future_samples <= 0:
        raise ValueError("future_samples must be positive")
    if cards_needed > len(remaining_cards):
        raise ValueError("not enough remaining cards for T0 rollout")
    remaining = tuple(remaining_cards)
    return [
        tuple(
            random.Random(
                common_future_seed(
                    base_seed=base_seed,
                    run_id=run_id,
                    root_fingerprint=root_fingerprint,
                    sample_index=sample_index,
                    street="T0",
                )
            ).sample(remaining, cards_needed)
        )
        for sample_index in range(future_samples)
    ]


def _future_digest(futures: Sequence[Sequence[str]]) -> str:
    payload = json.dumps(futures, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _decision_seed(
    *,
    hand_seed: int,
    sample_id: int,
    future_index: int,
    street: int,
    player: int,
) -> int:
    # Deliberately excludes the candidate action index.  For a fixed sampled
    # future, every T0 action therefore sees the same continuation RNG stream.
    return policy_decision_seed(
        base_seed=hand_seed,
        run_id=f"hu_turn0_teacher|sample={sample_id}",
        root_fingerprint=f"hand={hand_seed}|sample={sample_id}",
        future_index=future_index,
        actor=player,
        street=f"T{street}",
        decision_ordinal=street * 2 + player,
    )


def _choose_policy_action(
    policy: Any,
    *,
    board: Board,
    dealt: Sequence[str],
    dead_cards: Sequence[str],
    opponent_board: Board,
    hand_seed: int,
    decision_seed: int,
    street: str,
) -> Action:
    opponent_public = set(opponent_board.all_cards())
    observation = ActorObservation(
        hero_board=board,
        opponent_public_board=opponent_board,
        dealt_cards=tuple(dealt),
        hero_private_discards=tuple(
            card for card in dead_cards if card not in opponent_public
        ),
        seat=getattr(
            policy,
            "seat",
            "second" if opponent_board.card_count() > board.card_count() else "first",
        ),
        street=street,  # type: ignore[arg-type]
        to_act_order=(
            "second" if opponent_board.card_count() > board.card_count() else "first"
        ),
    )
    return _choose_from_observation(
        policy,
        observation,
        hand_id=hand_seed,
        game_id=hand_seed,
        decision_seed=decision_seed,
    )


def _rollout_after_turn0_action(
    *,
    hero_player: int,
    action: Action,
    opponent_board: Board,
    policies: Sequence[Any],
    future_cards: Sequence[str],
    hand_seed: int,
    sample_id: int,
    future_index: int,
    profile: dict[str, Any] | None = None,
) -> float:
    boards = [Board.from_rows(), Board.from_rows()]
    private_discards: list[list[str]] = [[], []]
    boards[hero_player] = boards[hero_player].place(action.placements)
    boards[1 - hero_player] = opponent_board
    cursor = 0

    if hero_player == 0:
        opponent_dealt = tuple(future_cards[cursor : cursor + 5])
        cursor += 5
        started = time.perf_counter()
        opponent_action = _choose_policy_action(
            policies[1],
            board=boards[1],
            dealt=opponent_dealt,
            dead_cards=tuple(_visible_dead_cards_for(1, boards, private_discards)),
            opponent_board=boards[0],
            hand_seed=hand_seed,
            decision_seed=_decision_seed(
                hand_seed=hand_seed,
                sample_id=sample_id,
                future_index=future_index,
                street=0,
                player=1,
            ),
            street="T0",
        )
        _profile_add(profile, "opponent_opening_seconds", time.perf_counter() - started)
        _profile_count(profile, "opponent_opening_count")
        boards[1] = boards[1].place(opponent_action.placements)
        private_discards[1].extend(opponent_action.discards)

    for round_index in range(1, 5):
        for player in (0, 1):
            dealt = tuple(future_cards[cursor : cursor + 3])
            cursor += 3
            started = time.perf_counter()
            chosen = _choose_policy_action(
                policies[player],
                board=boards[player],
                dealt=dealt,
                dead_cards=tuple(
                    _visible_dead_cards_for(player, boards, private_discards)
                ),
                opponent_board=boards[1 - player],
                hand_seed=hand_seed,
                decision_seed=_decision_seed(
                    hand_seed=hand_seed,
                    sample_id=sample_id,
                    future_index=future_index,
                    street=round_index,
                    player=player,
                ),
                street=f"T{round_index}",
            )
            elapsed = time.perf_counter() - started
            _profile_add(profile, "continuation_decision_seconds", elapsed)
            _profile_add(profile, f"continuation_T{round_index}_seconds", elapsed)
            _profile_count(profile, "continuation_decision_count")
            _profile_count(profile, f"continuation_T{round_index}_count")
            boards[player] = boards[player].place(chosen.placements)
            private_discards[player].extend(chosen.discards)

    if cursor != len(future_cards):
        raise RuntimeError(
            f"T0 rollout consumed {cursor} future cards, expected {len(future_cards)}"
        )
    started = time.perf_counter()
    hero_score, _ = terminal_score(
        boards[hero_player],
        boards[1 - hero_player],
        fl_ev=DEFAULT_FL_EV,
    )
    _profile_add(profile, "terminal_score_seconds", time.perf_counter() - started)
    _profile_count(profile, "terminal_score_count")
    return float(hero_score)


def _baseline_action_index(
    *,
    board: Board,
    opponent_board: Board,
    dealt: Sequence[str],
    visible_dead_cards: Sequence[str],
    hero_player: int,
    policy: Any,
    actions: Sequence[Action],
    hand_seed: int,
    sample_id: int,
) -> int:
    baseline_action = _choose_policy_action(
        policy,
        board=board,
        dealt=dealt,
        dead_cards=visible_dead_cards,
        opponent_board=opponent_board,
        hand_seed=hand_seed,
        decision_seed=_decision_seed(
            hand_seed=hand_seed,
            sample_id=sample_id,
            future_index=0,
            street=0,
            player=hero_player,
        ),
        street="T0",
    )
    index_by_key = {_action_key(action): index for index, action in enumerate(actions)}
    try:
        return index_by_key[_action_key(baseline_action)]
    except KeyError as exc:
        raise RuntimeError("baseline T0 policy returned an action outside the legal set") from exc


def evaluate_turn0_action_subset(
    *,
    board: Board,
    opponent_board: Board,
    dealt: Sequence[str],
    remaining_cards: Sequence[str],
    hero_player: int,
    policies: Sequence[Any],
    hand_seed: int,
    sample_id: int,
    future_samples: int,
    future_seed: int,
    action_indices: Sequence[int] | None = None,
    max_actions: int = 0,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if board.card_count() != 0:
        raise ValueError("T0 teacher requires an empty hero board")
    if opponent_board.card_count() not in {0, 5}:
        raise ValueError("T0 opponent board must contain zero or five cards")
    if hero_player == 0 and opponent_board.card_count() != 0:
        raise ValueError("first-seat T0 must not see an opponent opening board")
    if hero_player == 1 and opponent_board.card_count() != 5:
        raise ValueError("second-seat T0 requires the first-seat opening board")
    if max_actions < 0:
        raise ValueError("max_actions must be non-negative")

    started = time.perf_counter()
    actions = generate_actions(board, dealt)
    _profile_add(profile, "legal_action_generation_seconds", time.perf_counter() - started)
    _profile_count(profile, "legal_action_generation_count")
    if not actions:
        raise ValueError("no legal T0 actions")

    state_boards = [Board.from_rows(), Board.from_rows()]
    state_boards[hero_player] = board
    state_boards[1 - hero_player] = opponent_board
    state_discards: list[list[str]] = [[], []]
    visible_dead_cards = tuple(
        _visible_dead_cards_for(hero_player, state_boards, state_discards)
    )
    seat = _seat_name(hero_player)
    observation = ActorObservation(
        hero_board=board,
        opponent_public_board=opponent_board,
        dealt_cards=tuple(dealt),
        hero_private_discards=(),
        seat=seat,  # type: ignore[arg-type]
        street="T0",
        to_act_order=seat,  # type: ignore[arg-type]
    )
    baseline_index = _baseline_action_index(
        board=board,
        opponent_board=opponent_board,
        dealt=dealt,
        visible_dead_cards=visible_dead_cards,
        hero_player=hero_player,
        policy=policies[hero_player],
        actions=actions,
        hand_seed=hand_seed,
        sample_id=sample_id,
    )

    if action_indices is None:
        selected_indices = list(range(len(actions)))
        if max_actions > 0:
            selected_indices = selected_indices[:max_actions]
            if baseline_index not in selected_indices:
                selected_indices.append(baseline_index)
    else:
        selected_indices = []
        seen: set[int] = set()
        for raw_index in action_indices:
            index = int(raw_index)
            if 0 <= index < len(actions) and index not in seen:
                selected_indices.append(index)
                seen.add(index)
        if baseline_index not in seen:
            selected_indices.append(baseline_index)

    cards_needed = 29 if hero_player == 0 else 24
    started = time.perf_counter()
    futures = _future_sequences(
        remaining_cards,
        cards_needed=cards_needed,
        future_samples=future_samples,
        base_seed=future_seed,
        run_id=f"hu_turn0_teacher_futures|sample={sample_id}|player={hero_player}",
        root_fingerprint=observation.fingerprint(),
    )
    digest = _future_digest(futures)
    _profile_add(profile, "future_sampling_seconds", time.perf_counter() - started)
    _profile_count(profile, "future_sampling_count")

    evaluated: list[dict[str, Any]] = []
    rollout_scores_by_index: dict[int, list[float]] = {}
    for original_index in selected_indices:
        action_started = time.perf_counter()
        scores = [
            _rollout_after_turn0_action(
                hero_player=hero_player,
                action=actions[original_index],
                opponent_board=opponent_board,
                policies=policies,
                future_cards=future,
                hand_seed=hand_seed,
                sample_id=sample_id,
                future_index=future_index,
                profile=profile,
            )
            for future_index, future in enumerate(futures)
        ]
        mean = sum(scores) / len(scores)
        variance = sum((score - mean) ** 2 for score in scores) / max(len(scores) - 1, 1)
        payload = action_to_json(board, actions[original_index])
        payload.update(
            {
                "action_index": original_index,
                "original_index": original_index,
                "canonical_action_key": action_key(
                    actions[original_index]
                ).to_token(),
                "score": mean,
                "ev": mean,
                "se": math.sqrt(variance / len(scores)),
                "rollout_count": len(scores),
                "common_random_future_digest": digest,
                "action_eval_seconds": time.perf_counter() - action_started,
            }
        )
        evaluated.append(payload)
        rollout_scores_by_index[original_index] = scores
        _profile_count(profile, "evaluated_action_count")
        _profile_count(profile, "raw_rollout_count", len(scores))
        _profile_add(profile, "action_eval_seconds", payload["action_eval_seconds"])

    evaluated.sort(
        key=lambda item: (
            -float(item["score"]),
            str(item["canonical_action_key"]),
        )
    )
    by_index = {int(item["original_index"]): item for item in evaluated}
    baseline = by_index[baseline_index]
    baseline_scores = rollout_scores_by_index[baseline_index]
    for item in evaluated:
        original_index = int(item["original_index"])
        paired_deltas = [
            score - baseline_score
            for score, baseline_score in zip(
                rollout_scores_by_index[original_index], baseline_scores, strict=True
            )
        ]
        delta_mean = sum(paired_deltas) / len(paired_deltas)
        delta_variance = sum(
            (delta - delta_mean) ** 2 for delta in paired_deltas
        ) / max(len(paired_deltas) - 1, 1)
        delta_se = math.sqrt(delta_variance / len(paired_deltas))
        item["delta_vs_baseline"] = delta_mean
        item["delta_se_vs_baseline"] = delta_se
        item["delta_z_vs_baseline"] = delta_mean / max(delta_se, 1e-9)
    best = evaluated[0]
    return {
        "actions": evaluated,
        "total_legal_actions": len(actions),
        "evaluated_action_count": len(evaluated),
        "actions_truncated": len(evaluated) < len(actions),
        "action_key_schema": ACTION_KEY_SCHEMA,
        "rng_schema": COUNTER_RNG_SCHEMA,
        "legal_action_set_digest": legal_action_set_digest(actions),
        "legal_action_order_digest": ordered_action_mapping_digest(actions),
        "baseline_action_index": baseline_index,
        "baseline_action_key": action_key(actions[baseline_index]).to_token(),
        "baseline_ev": float(baseline["ev"]),
        "best_action_index": int(best["original_index"]),
        "best_action_key": str(best["canonical_action_key"]),
        "best_ev": float(best["ev"]),
        "delta_best_vs_baseline": float(best["ev"]) - float(baseline["ev"]),
        "delta_best_vs_baseline_se": float(best["delta_se_vs_baseline"]),
        "common_random_future_digest": digest,
        "common_random_futures_verified": all(
            item["common_random_future_digest"] == digest for item in evaluated
        ),
        "future_seed": int(future_seed),
        "future_cards_per_rollout": cards_needed,
    }


def build_turn0_teacher_sample(
    *,
    board: Board,
    opponent_board: Board,
    dealt: Sequence[str],
    remaining_cards: Sequence[str],
    hero_player: int,
    policies: Sequence[Any],
    hand_seed: int,
    sample_id: int,
    future_samples: int,
    future_seed: int,
    profile_name: str,
    opponent_profile: str,
    source_bucket: str,
    action_indices: Sequence[int] | None = None,
    max_actions: int = 0,
    candidate_model: Any | None = None,
    candidate_models: Sequence[Any] | None = None,
    candidate_topk: int = 0,
    candidate_union_cap: int = 0,
    candidate_union_mode: CandidateUnionMode = "min_rank",
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    actions = generate_actions(board, dealt)
    seat = _seat_name(hero_player)
    visible_dead_cards = tuple(opponent_board.all_cards())
    candidate_selector: dict[str, Any] | None = None
    if (
        action_indices is None
        and _candidate_model_entries(candidate_model, candidate_models)
        and candidate_topk > 0
    ):
        selected_pairs, candidate_selector = _select_candidate_pairs(
            board=board,
            opponent_board=opponent_board,
            dealt=dealt,
            actions=actions,
            visible_dead_cards=visible_dead_cards,
            seat=seat,
            candidate_model=candidate_model,
            candidate_models=candidate_models,
            candidate_topk=candidate_topk,
            candidate_union_cap=candidate_union_cap,
            candidate_union_mode=candidate_union_mode,
        )
        action_indices = [index for index, _action in selected_pairs]

    result = evaluate_turn0_action_subset(
        board=board,
        opponent_board=opponent_board,
        dealt=dealt,
        remaining_cards=remaining_cards,
        hero_player=hero_player,
        policies=policies,
        hand_seed=hand_seed,
        sample_id=sample_id,
        future_samples=future_samples,
        future_seed=future_seed,
        action_indices=action_indices,
        max_actions=max_actions,
        profile=profile,
    )
    sample = hu_policy_sample(
        board,
        dealt,
        actions,
        opponent_board=opponent_board,
        dead_cards=visible_dead_cards,
        seat=seat,
        to_act_order=seat,
    )
    best = result["actions"][0]
    second = result["actions"][1] if len(result["actions"]) > 1 else best
    sample.update(
        {
            "schema": "hu_turn0_stage1_teacher_v1",
            "phase": "hu_turn0_0card",
            "sample_id": sample_id,
            "source": f"hu_turn0_terminal_rollout_mc{future_samples}",
            "source_bucket": source_bucket,
            "source_bucket_group": "natural",
            "profile": profile_name,
            "opponent_profile": opponent_profile,
            "t1_continuation": "stage18_p1",
            "t2_continuation": "stage9f_p2",
            "t3_continuation": "stage7_m5_r10",
            "fl_ev": dict(DEFAULT_FL_EV),
            "hand_seed": hand_seed,
            "player": hero_player,
            "seat": seat,
            "best_action": 0,
            "score_gap": float(best["ev"]) - float(second["ev"]),
            "actions": result["actions"],
            "action_count": result["evaluated_action_count"],
            "total_legal_actions": result["total_legal_actions"],
            "evaluated_action_count": result["evaluated_action_count"],
            "actions_truncated": result["actions_truncated"],
            "action_key_schema": result["action_key_schema"],
            "legal_action_set_digest": result["legal_action_set_digest"],
            "legal_action_order_digest": result["legal_action_order_digest"],
            "baseline_action_index": result["baseline_action_index"],
            "baseline_action_key": result["baseline_action_key"],
            "baseline_ev": result["baseline_ev"],
            "best_action_index": result["best_action_index"],
            "best_action_key": result["best_action_key"],
            "best_ev": result["best_ev"],
            "delta_best_vs_baseline": result["delta_best_vs_baseline"],
            "delta_best_vs_baseline_se": result["delta_best_vs_baseline_se"],
            "candidate_selector": candidate_selector,
            "future_samples": future_samples,
            "future_seed": result["future_seed"],
            "future_cards_per_rollout": result["future_cards_per_rollout"],
            "common_random_future_digest": result["common_random_future_digest"],
            "common_random_futures_verified": result[
                "common_random_futures_verified"
            ],
            "visible_dead_cards": list(visible_dead_cards),
            "true_dead_cards": [],
            "hero_private_discards": [],
            "visibility_model": "actor_observation_v1",
            "discard_visibility": "own_private_only",
            "replay_ready": True,
            "remaining_card_count": len(remaining_cards),
        }
    )
    return sample


def build_turn0_pilot_samples(
    *,
    samples: int,
    seed: int,
    profile_name: str,
    opponent_profile: str,
    future_samples: int,
    max_actions: int,
    opening_lookahead_samples: int,
    source_bucket: str = "natural_terminal_mc1",
    allowed_seats: Sequence[str] = ("first", "second"),
    skip_records: int = 0,
    fast_skip_records: bool = False,
    candidate_model_path: Path | None = None,
    candidate_model_paths: Sequence[Path] | None = None,
    candidate_topk: int = 0,
    candidate_union_cap: int = 0,
    candidate_union_mode: CandidateUnionMode = "min_rank",
    collect_topk_log: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    if skip_records < 0:
        raise ValueError("skip_records must be non-negative")
    allowed = frozenset(str(seat) for seat in allowed_seats)
    if not allowed or not allowed <= {"first", "second"}:
        raise ValueError("allowed_seats must contain first and/or second")

    stats: dict[str, Any] = {}
    started = time.perf_counter()
    bundle = load_model_bundle(
        ModelPaths(), required_profiles(profile_name, opponent_profile)
    )
    stats["model_load_seconds"] = time.perf_counter() - started
    policies = [
        build_policy(
            profile_name,
            bundle,
            seed=seed * 2,
            seat="first",
            opening_lookahead_samples=opening_lookahead_samples,
        ),
        build_policy(
            opponent_profile,
            bundle,
            seed=seed * 2 + 1,
            seat="second",
            opening_lookahead_samples=opening_lookahead_samples,
        ),
    ]
    candidate_model = None
    candidate_models: list[Any] = []
    if candidate_model_path is not None:
        candidate_started = time.perf_counter()
        candidate_model = load_turn0_candidate_model(candidate_model_path)
        stats["candidate_model_load_seconds"] = time.perf_counter() - candidate_started
    if candidate_model_paths:
        candidate_started = time.perf_counter()
        candidate_models = [
            load_turn0_candidate_model(path) for path in candidate_model_paths
        ]
        stats["candidate_models_load_seconds"] = time.perf_counter() - candidate_started

    target_rows = samples + skip_records
    generated = 0
    rows: list[dict[str, Any]] = []
    attempts = 0
    run_started = time.perf_counter()
    while generated < target_rows:
        hand_seed = seed + attempts
        attempts += 1
        deck = create_deck(shuffle=True, rng=random.Random(hand_seed))
        boards = [Board.from_rows(), Board.from_rows()]
        private_discards: list[list[str]] = [[], []]
        dealt0 = tuple(deck[:5])

        if "first" in allowed and generated < target_rows:
            should_evaluate = (not fast_skip_records) or generated >= skip_records
            if should_evaluate:
                remaining = remaining_for_turn0_state(
                    boards[0], dealt0, boards[1]
                )
                future_seed = seed + generated * 1_000_003 + 101
                rows.append(
                    build_turn0_teacher_sample(
                        board=boards[0],
                        opponent_board=boards[1],
                        dealt=dealt0,
                        remaining_cards=remaining,
                        hero_player=0,
                        policies=policies,
                        hand_seed=hand_seed,
                        sample_id=generated,
                        future_samples=future_samples,
                        future_seed=future_seed,
                        profile_name=profile_name,
                        opponent_profile=opponent_profile,
                        source_bucket=source_bucket,
                        max_actions=max_actions,
                        candidate_model=candidate_model,
                        candidate_models=candidate_models,
                        candidate_topk=candidate_topk,
                        candidate_union_cap=candidate_union_cap,
                        candidate_union_mode=candidate_union_mode,
                        profile=stats,
                    )
                )
            generated += 1

        first_action = _choose_policy_action(
            policies[0],
            board=boards[0],
            dealt=dealt0,
            dead_cards=(),
            opponent_board=boards[1],
            hand_seed=hand_seed,
            decision_seed=_decision_seed(
                hand_seed=hand_seed,
                sample_id=generated,
                future_index=0,
                street=0,
                player=0,
            ),
            street="T0",
        )
        boards[0] = boards[0].place(first_action.placements)
        private_discards[0].extend(first_action.discards)

        if "second" in allowed and generated < target_rows:
            dealt1 = tuple(deck[5:10])
            should_evaluate = (not fast_skip_records) or generated >= skip_records
            if should_evaluate:
                remaining = remaining_for_turn0_state(
                    boards[1], dealt1, boards[0]
                )
                future_seed = seed + generated * 1_000_003 + 211
                rows.append(
                    build_turn0_teacher_sample(
                        board=boards[1],
                        opponent_board=boards[0],
                        dealt=dealt1,
                        remaining_cards=remaining,
                        hero_player=1,
                        policies=policies,
                        hand_seed=hand_seed,
                        sample_id=generated,
                        future_samples=future_samples,
                        future_seed=future_seed,
                        profile_name=profile_name,
                        opponent_profile=opponent_profile,
                        source_bucket=source_bucket,
                        max_actions=max_actions,
                        candidate_model=candidate_model,
                        candidate_models=candidate_models,
                        candidate_topk=candidate_topk,
                        candidate_union_cap=candidate_union_cap,
                        candidate_union_mode=candidate_union_mode,
                        profile=stats,
                    )
                )
            generated += 1

    output_rows = rows if fast_skip_records else rows[skip_records:]
    elapsed = time.perf_counter() - run_started
    legal_counts = [int(row["total_legal_actions"]) for row in output_rows]
    evaluated_counts = [int(row["evaluated_action_count"]) for row in output_rows]
    summary = {
        "schema": "hu_turn0_stage1_pilot_summary_v1",
        "samples": len(output_rows),
        "generated_samples": generated,
        "skip_records": skip_records,
        "fast_skip_records": fast_skip_records,
        "allowed_seats": sorted(allowed),
        "attempts": attempts,
        "profile": profile_name,
        "opponent_profile": opponent_profile,
        "t1_continuation": "stage18_p1",
        "t2_continuation": "stage9f_p2",
        "t3_continuation": "stage7_m5_r10",
        "future_samples": future_samples,
        "max_actions": max_actions,
        "source_bucket": source_bucket,
        "candidate_model": str(candidate_model_path) if candidate_model_path else None,
        "candidate_models": [str(path) for path in candidate_model_paths or ()],
        "candidate_topk": int(candidate_topk),
        "candidate_union_cap": int(candidate_union_cap),
        "candidate_union_mode": candidate_union_mode,
        "topk_decision_log_collected": bool(collect_topk_log),
        "missing": samples - len(output_rows),
        "invalid_states": 0,
        "invalid_actions": 0,
        "mean_legal_actions": sum(legal_counts) / len(legal_counts),
        "min_legal_actions": min(legal_counts),
        "max_legal_actions": max(legal_counts),
        "mean_evaluated_actions": sum(evaluated_counts) / len(evaluated_counts),
        "all_common_random_futures_verified": all(
            row["common_random_futures_verified"] for row in output_rows
        ),
        "all_replay_ready": all(row["replay_ready"] for row in output_rows),
        "elapsed_seconds": elapsed,
        "seconds_per_sample": elapsed / len(output_rows),
        "profile_stats": stats,
    }
    return output_rows, summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--skip-records", type=int, default=0)
    parser.add_argument("--fast-skip-records", action="store_true")
    parser.add_argument("--future-samples", type=int, default=1)
    parser.add_argument("--max-actions", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026102001)
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default="stage18_p1")
    parser.add_argument(
        "--opponent-profile", choices=PROFILE_CHOICES, default="stage18_p1"
    )
    parser.add_argument("--opening-lookahead-samples", type=int, default=1)
    parser.add_argument("--source-bucket", default="natural_terminal_mc1")
    parser.add_argument("--candidate-model", type=Path)
    parser.add_argument("--candidate-models", type=Path, nargs="+")
    parser.add_argument("--candidate-topk", type=int, default=0)
    parser.add_argument("--candidate-union-cap", type=int, default=0)
    parser.add_argument(
        "--candidate-union-mode",
        choices=CANDIDATE_UNION_MODES,
        default="min_rank",
    )
    parser.add_argument("--collect-topk-log", action="store_true")
    parser.add_argument(
        "--seats",
        nargs="+",
        choices=("first", "second"),
        default=("first", "second"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.candidate_topk < 0:
        raise SystemExit("--candidate-topk must be non-negative")
    if args.candidate_union_cap < 0:
        raise SystemExit("--candidate-union-cap must be non-negative")
    if args.candidate_model is not None and args.candidate_models:
        raise SystemExit("--candidate-model and --candidate-models are mutually exclusive")
    if args.candidate_model is not None and args.candidate_topk <= 0:
        raise SystemExit("--candidate-topk must be positive with --candidate-model")
    if args.candidate_models and args.candidate_topk <= 0:
        raise SystemExit("--candidate-topk must be positive with --candidate-models")
    rows, summary = build_turn0_pilot_samples(
        samples=args.samples,
        skip_records=args.skip_records,
        seed=args.seed,
        profile_name=args.profile,
        opponent_profile=args.opponent_profile,
        future_samples=args.future_samples,
        max_actions=args.max_actions,
        opening_lookahead_samples=args.opening_lookahead_samples,
        source_bucket=args.source_bucket,
        allowed_seats=args.seats,
        fast_skip_records=args.fast_skip_records,
        candidate_model_path=args.candidate_model,
        candidate_model_paths=args.candidate_models,
        candidate_topk=args.candidate_topk,
        candidate_union_cap=args.candidate_union_cap,
        candidate_union_mode=args.candidate_union_mode,
        collect_topk_log=args.collect_topk_log,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
