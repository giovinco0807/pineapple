"""Collect HU-aware Turn3 teacher samples from self-play states."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable

from .action_space import generate_turn_actions
from .ai_profiles import (
    DEFAULT_OPENING_MODEL,
    DEFAULT_TURN1_MODEL,
    DEFAULT_TURN2_MODEL,
    DEFAULT_TURN3_MODEL,
    ModelPaths,
    build_policy,
    load_model_bundle,
)
from .cards import ALL_CARDS, create_deck, validate_cards
from .hu_turn3_model import hu_policy_sample, load_hu_action_value_model
from .policy import RegularAiPolicy, action_to_json, policy_sample
from .state import Board
from .teacher import DEFAULT_FL_EV, terminal_score
from .visibility import HuDiscardTracker


def remaining_for_hu_teacher(
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


def to_act_order_for(board: Board, opponent_board: Board) -> str:
    return "second" if opponent_board.card_count() > board.card_count() else "first"


def evaluate_hu_self_play_turn3_actions(
    *,
    board: Board,
    dealt_cards: Iterable[str],
    opponent_board: Board,
    dead_cards: Iterable[str] = (),
    hero_seat: str,
    hero_policy: RegularAiPolicy,
    opponent_policy: RegularAiPolicy,
    self_regret_model: object | None = None,
    self_regret_penalty_weight: float = 0.0,
    self_regret_free: float = 0.0,
    future_samples: int,
    rng: random.Random,
) -> list[dict]:
    dealt = tuple(dealt_cards)
    actions = generate_turn_actions(board, dealt)
    dead = tuple(dead_cards)
    remaining = remaining_for_hu_teacher(board, dealt, opponent_board, dead)
    to_act_order = to_act_order_for(board, opponent_board)
    visible_dead = (*opponent_board.all_cards(), *dead)
    action_payloads = hu_policy_sample(
        board,
        dealt,
        actions,
        opponent_board=opponent_board,
        dead_cards=visible_dead,
        seat=hero_seat,
        to_act_order=to_act_order,
    )["actions"]
    self_predictions = None
    best_self_prediction = 0.0
    if self_regret_penalty_weight > 0.0 and self_regret_model is not None:
        self_predictions = self_regret_model.predict_sample(policy_sample(board, dealt, actions))
        best_self_prediction = float(max(self_predictions))

    evaluated: list[dict] = []
    rollout_count = _rollout_count(remaining, future_samples)
    future_rollouts = []
    for _ in range(rollout_count):
        shuffled = list(remaining)
        rng.shuffle(shuffled)
        future_rollouts.append(shuffled)
    precomputed_opponent_t3 = _precompute_opponent_self_board_t3_rollouts(
        opponent_board=opponent_board,
        opponent_policy=opponent_policy,
        future_rollouts=future_rollouts,
    )
    for action_index, (action, payload) in enumerate(zip(actions, action_payloads)):
        hero_after_action = board.place(action.placements)
        score_sum = 0.0
        completed = 0
        for rollout_index, shuffled in enumerate(future_rollouts):
            if precomputed_opponent_t3 is None:
                score = _rollout_after_hero_t3_action(
                    hero_board=hero_after_action,
                    opponent_board=opponent_board,
                    dead_cards=(*dead, *action.discards),
                    hero_seat=hero_seat,
                    future_cards=shuffled,
                    hero_policy=hero_policy,
                    opponent_policy=opponent_policy,
                )
            else:
                future_tail, resolved_opponent, opponent_discards = precomputed_opponent_t3[rollout_index]
                score = _rollout_after_resolved_t3_action(
                    hero_board=hero_after_action,
                    opponent_board=resolved_opponent,
                    dead_cards=(*dead, *action.discards, *opponent_discards),
                    hero_seat=hero_seat,
                    future_cards=future_tail,
                    hero_policy=hero_policy,
                    opponent_policy=opponent_policy,
                )
            if score is None:
                continue
            score_sum += score
            completed += 1
        if completed:
            raw_score = score_sum / completed
            self_regret = 0.0
            penalty = 0.0
            if self_predictions is not None:
                self_score = float(self_predictions[action_index])
                self_regret = max(0.0, best_self_prediction - self_score)
                penalty = self_regret_penalty_weight * max(0.0, self_regret - self_regret_free)
                payload["self_model_score"] = self_score
                payload["self_regret"] = self_regret
                payload["self_regret_penalty"] = penalty
            payload["raw_score"] = raw_score
            payload["score"] = raw_score - penalty
            payload["future_count"] = completed
            evaluated.append(payload)

    evaluated.sort(key=lambda item: item["score"], reverse=True)
    return evaluated


def build_hu_self_play_turn3_sample(
    *,
    sample_id: int,
    board: Board,
    dealt_cards: Iterable[str],
    opponent_board: Board,
    dead_cards: Iterable[str] = (),
    hero_seat: str,
    hero_policy: RegularAiPolicy,
    opponent_policy: RegularAiPolicy,
    self_regret_model: object | None = None,
    self_regret_penalty_weight: float = 0.0,
    self_regret_free: float = 0.0,
    future_samples: int,
    rng: random.Random,
) -> dict | None:
    ranked = evaluate_hu_self_play_turn3_actions(
        board=board,
        dealt_cards=dealt_cards,
        opponent_board=opponent_board,
        dead_cards=dead_cards,
        hero_seat=hero_seat,
        hero_policy=hero_policy,
        opponent_policy=opponent_policy,
        self_regret_model=self_regret_model,
        self_regret_penalty_weight=self_regret_penalty_weight,
        self_regret_free=self_regret_free,
        future_samples=future_samples,
        rng=rng,
    )
    if not ranked:
        return None
    score_gap = ranked[0]["score"] - ranked[1]["score"] if len(ranked) > 1 else 0.0
    sample = hu_policy_sample(
        board,
        dealt_cards,
        generate_turn_actions(board, dealt_cards),
        opponent_board=opponent_board,
        dead_cards=(*opponent_board.all_cards(), *tuple(dead_cards)),
        seat=hero_seat,
        to_act_order=to_act_order_for(board, opponent_board),
    )
    sample.update(
        {
            "sample_id": sample_id,
            "score_gap": score_gap,
            "actions": ranked,
            "source": "self_play_rollout",
            "self_regret_penalty": {
                "penalty_weight": self_regret_penalty_weight,
                "free_regret": self_regret_free,
            },
        }
    )
    return sample


def annotate_hu_turn3_reference_actions(sample: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    """Attach teacher EV deltas for selected/reference actions.

    ``selection`` indices are in the original legal-action order, while teacher
    samples are sorted by rollout score.  Resolve by action identity so downstream
    gate training can use delta-vs-reference labels without depending on order.
    """
    board = Board.from_rows(**sample["board"])
    legal_actions = generate_turn_actions(board, sample["dealt"])
    legal_payloads = [action_to_json(board, action) for action in legal_actions]
    sorted_index_by_key = {
        _json_action_key(action): index
        for index, action in enumerate(sample.get("actions", ()))
    }
    if not sample.get("actions"):
        return sample

    def resolve(label: str, original_index: int | None) -> dict[str, Any] | None:
        if original_index is None or original_index < 0 or original_index >= len(legal_payloads):
            return None
        key = _json_action_key(legal_payloads[original_index])
        sorted_index = sorted_index_by_key.get(key)
        if sorted_index is None:
            return None
        action = sample["actions"][sorted_index]
        return {
            "label": label,
            "original_index": int(original_index),
            "sorted_index": int(sorted_index),
            "score": float(action.get("score", 0.0)),
            "raw_score": float(action.get("raw_score", action.get("score", 0.0))),
            "self_regret": float(action.get("self_regret", 0.0)),
            "self_regret_penalty": float(action.get("self_regret_penalty", 0.0)),
            "action": {
                "placements": action.get("placements", []),
                "discards": action.get("discards", []),
                "next_board": action.get("next_board", {}),
            },
        }

    best = {
        "label": "teacher_best",
        "sorted_index": 0,
        "score": float(sample["actions"][0].get("score", 0.0)),
        "raw_score": float(sample["actions"][0].get("raw_score", sample["actions"][0].get("score", 0.0))),
        "action": {
            "placements": sample["actions"][0].get("placements", []),
            "discards": sample["actions"][0].get("discards", []),
            "next_board": sample["actions"][0].get("next_board", {}),
        },
    }
    second = None
    if len(sample["actions"]) > 1:
        second = {
            "label": "teacher_second",
            "sorted_index": 1,
            "score": float(sample["actions"][1].get("score", 0.0)),
            "raw_score": float(
                sample["actions"][1].get("raw_score", sample["actions"][1].get("score", 0.0))
            ),
        }

    baseline = resolve("baseline", _optional_int(selection.get("baseline_index")))
    selection_hu = resolve("selection_hu", _optional_int(selection.get("hu_index")))
    compare_hu = resolve("compare_hu", _optional_int(selection.get("compare_hu_index")))
    references: dict[str, Any] = {
        "teacher_best": best,
        "teacher_second": second,
        "baseline": baseline,
        "selection_hu": selection_hu,
        "compare_hu": compare_hu,
    }
    best_score = best["score"]
    if baseline is not None:
        references["delta_best_vs_baseline"] = best_score - float(baseline["score"])
        references["baseline_regret"] = best_score - float(baseline["score"])
    if selection_hu is not None:
        references["delta_best_vs_selection_hu"] = best_score - float(selection_hu["score"])
        references["selection_hu_regret"] = best_score - float(selection_hu["score"])
    if baseline is not None and selection_hu is not None:
        references["delta_selection_hu_vs_baseline"] = float(selection_hu["score"]) - float(
            baseline["score"]
        )
    if compare_hu is not None and selection_hu is not None:
        references["delta_selection_hu_vs_compare_hu"] = float(selection_hu["score"]) - float(
            compare_hu["score"]
        )
    sample["reference_actions"] = references
    return sample


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _json_action_key(action: dict[str, Any]) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    placements = tuple((str(card), str(row)) for card, row in action.get("placements", ()))
    discards = tuple(str(card) for card in action.get("discards", ()))
    return placements, discards


def hu_turn3_selection_metadata(
    *,
    board: Board,
    dealt_cards: Iterable[str],
    opponent_board: Board,
    dead_cards: Iterable[str] = (),
    hero_seat: str,
    baseline_turn3_model: object,
    selection_hu_model: object,
    compare_hu_model: object | None = None,
) -> dict | None:
    dealt = tuple(dealt_cards)
    actions = generate_turn_actions(board, dealt)
    if not actions:
        return None
    dead = tuple(dead_cards)
    hu_sample = hu_policy_sample(
        board,
        dealt,
        actions,
        opponent_board=opponent_board,
        dead_cards=(*opponent_board.all_cards(), *dead),
        seat=hero_seat,
        to_act_order=to_act_order_for(board, opponent_board),
    )
    hu_predictions = selection_hu_model.predict_sample(hu_sample)
    hu_index = int(max(range(len(hu_predictions)), key=lambda index: float(hu_predictions[index])))
    baseline_sample = policy_sample(board, dealt, actions)
    baseline_index = int(baseline_turn3_model.choose_action_index(baseline_sample))
    metadata = {
        "hu_index": hu_index,
        "baseline_index": baseline_index,
        "disagreement": hu_index != baseline_index,
        "predicted_margin_vs_baseline": float(hu_predictions[hu_index] - hu_predictions[baseline_index]),
        "predicted_hu_best_score": float(hu_predictions[hu_index]),
        "predicted_baseline_score": float(hu_predictions[baseline_index]),
    }
    if compare_hu_model is not None:
        compare_predictions = compare_hu_model.predict_sample(hu_sample)
        compare_index = int(
            max(range(len(compare_predictions)), key=lambda index: float(compare_predictions[index]))
        )
        metadata.update(
            {
                "compare_hu_index": compare_index,
                "compare_hu_disagreement": hu_index != compare_index,
                "compare_predicted_margin_vs_baseline": float(
                    compare_predictions[compare_index] - compare_predictions[baseline_index]
                ),
                "compare_predicted_hu_best_score": float(compare_predictions[compare_index]),
                "compare_predicted_selection_score": float(compare_predictions[hu_index]),
                "predicted_margin_vs_compare_action": float(
                    hu_predictions[hu_index] - hu_predictions[compare_index]
                ),
            }
        )
    return metadata


def passes_hu_turn3_selection(
    selection: dict,
    *,
    min_margin: float | None = None,
    max_margin: float | None = None,
    require_disagreement: bool = False,
    require_compare_disagreement: bool = False,
) -> bool:
    if require_disagreement and not selection["disagreement"]:
        return False
    if require_compare_disagreement and not selection.get("compare_hu_disagreement", False):
        return False
    margin = float(selection["predicted_margin_vs_baseline"])
    if min_margin is not None and margin < min_margin:
        return False
    if max_margin is not None and margin > max_margin:
        return False
    return True


def collect_hu_self_play_turn3_dataset(
    *,
    output: Path,
    samples: int,
    seed: int,
    policy_bundle: object,
    future_samples: int,
    opening_lookahead_samples: int,
    max_hands: int,
    min_score_gap: float,
    self_regret_penalty_weight: float,
    self_regret_free: float,
    selection_hu_model: object | None = None,
    selection_compare_hu_model: object | None = None,
    selection_min_margin: float | None = None,
    selection_max_margin: float | None = None,
    selection_require_disagreement: bool = False,
    selection_require_compare_disagreement: bool = False,
) -> dict:
    if samples <= 0:
        raise ValueError("samples must be positive")
    rng = random.Random(seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    collected = 0
    hands = 0
    attempts = 0
    skipped_gap = 0
    skipped_selection = 0
    with output.open("w", encoding="utf-8") as handle:
        while collected < samples and hands < max_hands:
            hand_seed = seed + hands
            hands += 1
            deck = create_deck(shuffle=True, rng=random.Random(hand_seed))
            cursor = 0
            boards = [Board.from_rows(), Board.from_rows()]
            discards = HuDiscardTracker()
            policies = [
                build_policy(
                    "current",
                    policy_bundle,
                    seed=hand_seed * 2,
                    opening_lookahead_samples=opening_lookahead_samples,
                ),
                build_policy(
                    "current",
                    policy_bundle,
                    seed=hand_seed * 2 + 1,
                    opening_lookahead_samples=opening_lookahead_samples,
                ),
            ]
            policies[0].seat = "first"
            policies[1].seat = "second"

            for player in (0, 1):
                dealt = tuple(deck[cursor : cursor + 5])
                cursor += 5
                action = policies[player].choose_action(
                    boards[player],
                    dealt,
                    dead_cards=(*boards[1 - player].all_cards(), *discards.own_discards(player)),
                    opponent_board=boards[1 - player],
                )
                boards[player] = boards[player].place(action.placements)
                discards.record(player, action.discards)

            for _round in range(1, 5):
                for player in (0, 1):
                    dealt = tuple(deck[cursor : cursor + 3])
                    cursor += 3
                    if boards[player].card_count() == 9 and collected < samples:
                        attempts += 1
                        selection = None
                        if selection_hu_model is not None:
                            selection = hu_turn3_selection_metadata(
                                board=boards[player],
                                dealt_cards=dealt,
                                opponent_board=boards[1 - player],
                                dead_cards=discards.own_discards(player),
                                hero_seat="first" if player == 0 else "second",
                                baseline_turn3_model=policy_bundle.turn3,
                                selection_hu_model=selection_hu_model,
                                compare_hu_model=selection_compare_hu_model,
                            )
                            if selection is None or not passes_hu_turn3_selection(
                                selection,
                                min_margin=selection_min_margin,
                                max_margin=selection_max_margin,
                                require_disagreement=selection_require_disagreement,
                                require_compare_disagreement=selection_require_compare_disagreement,
                            ):
                                skipped_selection += 1
                                action = policies[player].choose_action(
                                    boards[player],
                                    dealt,
                                    dead_cards=(*boards[1 - player].all_cards(), *discards.own_discards(player)),
                                    opponent_board=boards[1 - player],
                                )
                                boards[player] = boards[player].place(action.placements)
                                discards.record(player, action.discards)
                                continue
                        sample = build_hu_self_play_turn3_sample(
                            sample_id=collected,
                            board=boards[player],
                            dealt_cards=dealt,
                            opponent_board=boards[1 - player],
                            dead_cards=discards.own_discards(player),
                            hero_seat="first" if player == 0 else "second",
                            hero_policy=policies[player],
                            opponent_policy=policies[1 - player],
                            self_regret_model=policy_bundle.turn3,
                            self_regret_penalty_weight=self_regret_penalty_weight,
                            self_regret_free=self_regret_free,
                            future_samples=future_samples,
                            rng=rng,
                        )
                        if sample is not None:
                            if selection is not None:
                                sample["selection"] = selection
                                annotate_hu_turn3_reference_actions(sample, selection)
                            if sample["score_gap"] >= min_score_gap:
                                handle.write(json.dumps(sample, separators=(",", ":")) + "\n")
                                collected += 1
                            else:
                                skipped_gap += 1
                    action = policies[player].choose_action(
                        boards[player],
                        dealt,
                        dead_cards=(*boards[1 - player].all_cards(), *discards.own_discards(player)),
                        opponent_board=boards[1 - player],
                    )
                    boards[player] = boards[player].place(action.placements)
                    discards.record(player, action.discards)
    if collected < samples:
        raise RuntimeError(f"collected {collected}/{samples} samples after {hands} hands")
    return {
        "output": str(output),
        "samples": collected,
        "hands": hands,
        "attempts": attempts,
        "skipped_gap": skipped_gap,
        "seed": seed,
        "future_samples": future_samples,
        "min_score_gap": min_score_gap,
        "self_regret_penalty_weight": self_regret_penalty_weight,
        "self_regret_free": self_regret_free,
        "selection": {
            "enabled": selection_hu_model is not None,
            "min_margin": selection_min_margin,
            "max_margin": selection_max_margin,
            "require_disagreement": selection_require_disagreement,
            "require_compare_disagreement": selection_require_compare_disagreement,
            "compare_enabled": selection_compare_hu_model is not None,
            "skipped": skipped_selection,
        },
    }


def _rollout_count(remaining: tuple[str, ...], future_samples: int) -> int:
    if future_samples < 0:
        raise ValueError("future_samples must be non-negative")
    if future_samples == 0:
        # Exact enumeration of full multi-turn trajectories is intentionally
        # not attempted here; use a large sampled count for this rollout teacher.
        return max(1, len(remaining))
    return future_samples


def _precompute_opponent_self_board_t3_rollouts(
    *,
    opponent_board: Board,
    opponent_policy: RegularAiPolicy,
    future_rollouts: list[list[str]],
) -> list[tuple[list[str], Board, tuple[str, ...]]] | None:
    if opponent_board.card_count() != 9:
        return None
    if opponent_policy.hu_turn3_model is not None or opponent_policy.turn3_model is None:
        return None

    resolved: list[tuple[list[str], Board, tuple[str, ...]]] = []
    for shuffled in future_rollouts:
        if len(shuffled) < 3:
            return None
        dealt = tuple(shuffled[:3])
        actions = generate_turn_actions(opponent_board, dealt)
        if not actions:
            return None
        sample = policy_sample(opponent_board, dealt, actions)
        action_index = opponent_policy.turn3_model.choose_action_index(sample)
        action = actions[action_index]
        resolved.append((list(shuffled[3:]), opponent_board.place(action.placements), action.discards))
    return resolved


def _rollout_after_hero_t3_action(
    *,
    hero_board: Board,
    opponent_board: Board,
    dead_cards: Iterable[str] = (),
    hero_seat: str,
    future_cards: list[str],
    hero_policy: RegularAiPolicy,
    opponent_policy: RegularAiPolicy,
) -> float | None:
    cursor = 0
    hero = hero_board
    opponent = opponent_board
    dead = list(dead_cards)

    def draw3() -> tuple[str, str, str] | None:
        nonlocal cursor
        if cursor + 3 > len(future_cards):
            return None
        dealt = tuple(future_cards[cursor : cursor + 3])
        cursor += 3
        return dealt  # type: ignore[return-value]

    if opponent.card_count() == 9:
        dealt = draw3()
        if dealt is None:
            return None
        action = opponent_policy.choose_action(
            opponent,
            dealt,
            dead_cards=(*hero.all_cards(), *dead),
            opponent_board=hero,
        )
        opponent = opponent.place(action.placements)
        dead.extend(action.discards)

    return _rollout_after_resolved_t3_action(
        hero_board=hero,
        opponent_board=opponent,
        dead_cards=dead,
        hero_seat=hero_seat,
        future_cards=future_cards[cursor:],
        hero_policy=hero_policy,
        opponent_policy=opponent_policy,
    )


def _rollout_after_resolved_t3_action(
    *,
    hero_board: Board,
    opponent_board: Board,
    dead_cards: Iterable[str] = (),
    hero_seat: str,
    future_cards: list[str],
    hero_policy: RegularAiPolicy,
    opponent_policy: RegularAiPolicy,
) -> float | None:
    cursor = 0
    hero = hero_board
    opponent = opponent_board
    dead = list(dead_cards)

    def draw3() -> tuple[str, str, str] | None:
        nonlocal cursor
        if cursor + 3 > len(future_cards):
            return None
        dealt = tuple(future_cards[cursor : cursor + 3])
        cursor += 3
        return dealt  # type: ignore[return-value]

    ordered_policies = (
        (("hero", hero_policy), ("opponent", opponent_policy))
        if hero_seat == "first"
        else (("opponent", opponent_policy), ("hero", hero_policy))
    )
    while hero.card_count() < 13 or opponent.card_count() < 13:
        for actor, policy in ordered_policies:
            if actor == "hero":
                if hero.card_count() >= 13:
                    continue
                dealt = draw3()
                if dealt is None:
                    return None
                action = policy.choose_action(
                    hero,
                    dealt,
                    dead_cards=(*opponent.all_cards(), *dead),
                    opponent_board=opponent,
                )
                hero = hero.place(action.placements)
                dead.extend(action.discards)
            else:
                if opponent.card_count() >= 13:
                    continue
                dealt = draw3()
                if dealt is None:
                    return None
                action = policy.choose_action(
                    opponent,
                    dealt,
                    dead_cards=(*hero.all_cards(), *dead),
                    opponent_board=hero,
                )
                opponent = opponent.place(action.placements)
                dead.extend(action.discards)

    score, _board_score = terminal_score(hero, opponent, fl_ev=DEFAULT_FL_EV)
    return score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--future-samples", type=int, default=16)
    parser.add_argument("--min-score-gap", type=float, default=0.0)
    parser.add_argument("--self-regret-penalty-weight", type=float, default=0.0)
    parser.add_argument("--self-regret-free", type=float, default=0.0)
    parser.add_argument("--selection-hu-turn3-model", type=Path)
    parser.add_argument("--selection-compare-hu-turn3-model", type=Path)
    parser.add_argument("--selection-min-margin", type=float)
    parser.add_argument("--selection-max-margin", type=float)
    parser.add_argument("--selection-require-disagreement", action="store_true")
    parser.add_argument("--selection-require-compare-disagreement", action="store_true")
    parser.add_argument("--max-hands", type=int, default=1000000)
    parser.add_argument("--opening-lookahead-samples", type=int, default=64)
    parser.add_argument("--opening-model", type=Path, default=DEFAULT_OPENING_MODEL)
    parser.add_argument("--turn1-model", type=Path, default=DEFAULT_TURN1_MODEL)
    parser.add_argument("--turn2-model", type=Path, default=DEFAULT_TURN2_MODEL)
    parser.add_argument("--turn3-model", type=Path, default=DEFAULT_TURN3_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_regret_penalty_weight < 0:
        raise SystemExit("--self-regret-penalty-weight must be non-negative")
    if args.self_regret_free < 0:
        raise SystemExit("--self-regret-free must be non-negative")
    if (
        args.selection_hu_turn3_model is None
        and (
            args.selection_min_margin is not None
            or args.selection_max_margin is not None
            or args.selection_require_disagreement
            or args.selection_compare_hu_turn3_model is not None
            or args.selection_require_compare_disagreement
        )
    ):
        raise SystemExit("--selection-hu-turn3-model is required when selection filters are set")
    if args.selection_require_compare_disagreement and args.selection_compare_hu_turn3_model is None:
        raise SystemExit("--selection-compare-hu-turn3-model is required for compare disagreement")
    if (
        args.selection_min_margin is not None
        and args.selection_max_margin is not None
        and args.selection_min_margin > args.selection_max_margin
    ):
        raise SystemExit("--selection-min-margin must be <= --selection-max-margin")
    paths = ModelPaths(
        opening=args.opening_model,
        turn1=args.turn1_model,
        turn2=args.turn2_model,
        turn3=args.turn3_model,
    )
    policy_bundle = load_model_bundle(paths, {"current"})
    selection_hu_model = (
        load_hu_action_value_model(args.selection_hu_turn3_model)
        if args.selection_hu_turn3_model is not None
        else None
    )
    selection_compare_hu_model = (
        load_hu_action_value_model(args.selection_compare_hu_turn3_model)
        if args.selection_compare_hu_turn3_model is not None
        else None
    )
    summary = collect_hu_self_play_turn3_dataset(
        output=args.output,
        samples=args.samples,
        seed=args.seed,
        policy_bundle=policy_bundle,
        future_samples=args.future_samples,
        opening_lookahead_samples=args.opening_lookahead_samples,
        max_hands=args.max_hands,
        min_score_gap=args.min_score_gap,
        self_regret_penalty_weight=args.self_regret_penalty_weight,
        self_regret_free=args.self_regret_free,
        selection_hu_model=selection_hu_model,
        selection_compare_hu_model=selection_compare_hu_model,
        selection_min_margin=args.selection_min_margin,
        selection_max_margin=args.selection_max_margin,
        selection_require_disagreement=args.selection_require_disagreement,
        selection_require_compare_disagreement=args.selection_require_compare_disagreement,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
