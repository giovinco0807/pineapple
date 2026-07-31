"""Role-aware sampled T3 HU evaluator with exact Rust T4 leaves.

This is a bootstrap PIMC evaluator, not a game-theoretic exact solver.  It
samples hidden-dead-card determinizations and lets the exact canonical T4
kernel solve each leaf.  All root candidates share the same sampled chance
tapes, and opponent T3 responses are selected only after averaging their
future T4 samples.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from ai.engine.action_space import Action, get_turn_actions
from ai.engine.encoding import ALL_CARDS, Board
from ai.engine.turn_order import (
    POSITION_CONTRACT_VERSION,
    normalize_position,
    validate_decision_board_counts,
)
from ai.tutor.exact_late import (
    CardNormalizer,
    action_key,
    action_to_dict,
    apply_action,
    board_card_count,
    board_to_dict,
    evaluate_late_positions_rust_batch,
    normalize_board,
)


INFORMATION_MODEL = "pimc_determinization_v1"
METHOD = "hu_sampled_pimc_exact_t4"


@dataclass(frozen=True)
class _LeafTask:
    root_index: int
    outer_index: int
    response_key: str | None
    inner_index: int


def _canonical_cards(cards: Iterable[str], normalizer: CardNormalizer) -> list[str]:
    return normalizer.cards(cards or [])


def _normalize_input(
    payload: dict[str, Any],
) -> tuple[Board, Board, list[str], list[str], list[str], str]:
    normalizer = CardNormalizer()
    board = normalize_board(payload.get("board") or {}, normalizer)
    opponent = normalize_board(payload.get("opponent_board") or {}, normalizer)
    dealt = _canonical_cards(payload.get("dealt") or [], normalizer)
    known_self = _canonical_cards(
        payload.get("known_discards_self") or payload.get("known_discards") or [],
        normalizer,
    )
    public_exclude = _canonical_cards(payload.get("public_exclude") or [], normalizer)

    # Legacy rows put public board cards and known self discards under
    # ``exclude``.  An additional identity is ambiguous: it may be an
    # opponent's face-down discard.  Synthetic removals must be declared
    # explicitly as ``public_exclude``.
    legacy_exclude = _canonical_cards(payload.get("exclude") or [], normalizer)
    already_known = {
        *board.all_cards(),
        *opponent.all_cards(),
        *dealt,
        *known_self,
        *public_exclude,
    }
    ambiguous_exclude = [card for card in legacy_exclude if card not in already_known]
    if ambiguous_exclude:
        raise ValueError(
            "ambiguous exclude cards may reveal opponent private discards; "
            "use public_exclude for synthetic public removals: "
            f"{ambiguous_exclude}"
        )
    public_exclude = list(dict.fromkeys(public_exclude))
    known_self = list(dict.fromkeys(known_self))

    raw_position = payload.get("actor") or payload.get("position") or payload.get("player_position")
    is_btn = payload["is_btn"] if "is_btn" in payload else None
    position = normalize_position(raw_position, is_btn=is_btn)
    first_actor = normalize_position(payload.get("first_actor") or "bb")
    if first_actor != "bb":
        raise ValueError("T3 HU sampled v1 requires first_actor='bb'")
    validate_decision_board_counts(3, position, board_card_count(board), board_card_count(opponent))
    if len(dealt) != 3:
        raise ValueError(f"T3 HU sampled evaluation requires exactly 3 dealt cards, got {len(dealt)}")
    return board, opponent, dealt, known_self, public_exclude, position


def _decision_id(
    board: Board,
    opponent: Board,
    dealt: Sequence[str],
    known_self: Sequence[str],
    public_exclude: Sequence[str],
    position: str,
) -> str:
    canonical = {
        "turn": 3,
        "position": position,
        "board": {key: sorted(value) for key, value in board_to_dict(board).items()},
        "opponent_board": {key: sorted(value) for key, value in board_to_dict(opponent).items()},
        "dealt": sorted(dealt),
        "known_discards_self": sorted(known_self),
        "public_exclude": sorted(public_exclude),
    }
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _priority(
    *,
    seed: int,
    decision_id: str,
    outer_index: int,
    inner_index: int,
    node: str,
    card: str,
) -> bytes:
    raw = f"{int(seed)}|{decision_id}|{outer_index}|{inner_index}|{node}|{card}"
    return hashlib.sha256(raw.encode("utf-8")).digest()


def _sample_cards(
    cards: Sequence[str],
    count: int,
    *,
    seed: int,
    decision_id: str,
    outer_index: int,
    inner_index: int = 0,
    node: str,
) -> tuple[str, ...]:
    if len(cards) < count:
        raise ValueError(f"cannot sample {count} cards from {len(cards)} live cards at {node}")
    ordered = sorted(
        cards,
        key=lambda card: _priority(
            seed=seed,
            decision_id=decision_id,
            outer_index=outer_index,
            inner_index=inner_index,
            node=node,
            card=card,
        ),
    )
    return tuple(ordered[:count])


def _without(cards: Sequence[str], removed: Iterable[str]) -> tuple[str, ...]:
    removed_set = set(removed)
    return tuple(card for card in cards if card not in removed_set)


def _summary_stats(values: Sequence[float]) -> dict[str, Any]:
    n = len(values)
    if n == 0:
        raise ValueError("cannot summarize zero samples")
    mean = sum(values) / n
    if n > 1:
        variance = sum((value - mean) ** 2 for value in values) / (n - 1)
        standard_error = math.sqrt(variance / n)
    else:
        variance = 0.0
        standard_error = 0.0
    half_width = 1.96 * standard_error
    return {
        "score": float(mean),
        "ev": float(mean),
        "samples": n,
        "variance": float(variance),
        "standard_error": float(standard_error),
        "ci95": [float(mean - half_width), float(mean + half_width)],
    }


def _best_leaf_score(result: dict[str, Any]) -> tuple[float, int]:
    best = result.get("best") or {}
    metrics = best.get("metrics") or {}
    if "score" not in metrics:
        raise RuntimeError("Rust T4 leaf returned no best score")
    return float(metrics["score"]), int(metrics.get("samples") or 0)


def evaluate_t3_hu_sampled(
    payload: dict[str, Any],
    *,
    seed: int = 20260712,
    outer_samples: int | None = None,
    response_inner_samples: int | None = None,
    rust_solver_path: str | Path | None = None,
    rust_timeout_s: float = 300.0,
    include_scenario_values: bool = False,
) -> dict[str, Any]:
    """Evaluate every legal T3 root action under a deterministic PIMC tape."""
    started = time.perf_counter()
    board, opponent, dealt, known_self, public_exclude, position = _normalize_input(payload)
    if outer_samples is None:
        outer_samples = 8 if position == "btn" else 2
    if response_inner_samples is None:
        response_inner_samples = 1 if position == "bb" else 0
    outer_n = max(1, int(outer_samples))
    inner_n = max(1, int(response_inner_samples)) if position == "bb" else 0

    root_actions = get_turn_actions(dealt, board)
    if not root_actions:
        raise ValueError("T3 HU sampled position has no legal root actions")
    root_keys = [action_key(action) for action in root_actions]
    decision_id = _decision_id(board, opponent, dealt, known_self, public_exclude, position)
    public_known = {
        *board.all_cards(),
        *opponent.all_cards(),
        *dealt,
        *known_self,
        *public_exclude,
    }
    unseen = tuple(card for card in ALL_CARDS if card not in public_known)

    records: list[dict[str, Any]] = []
    tasks: list[_LeafTask] = []

    if position == "btn":
        if len(unseen) < 6:
            raise ValueError("BTN T3 PIMC requires at least 6 unseen cards")
        scenarios: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
        for outer in range(outer_n):
            hidden_and_draw = _sample_cards(
                unseen,
                6,
                seed=seed,
                decision_id=decision_id,
                outer_index=outer,
                node="btn_root_bb_private_and_t4",
            )
            scenarios.append((hidden_and_draw[:3], hidden_and_draw[3:]))
        for root_index, root_action in enumerate(root_actions):
            btn11 = apply_action(board, root_action)
            for outer, (bb_private_discards, bb_t4_draw) in enumerate(scenarios):
                physical_dead = list(
                    dict.fromkeys(
                        [
                            *known_self,
                            root_action.discard,
                            *bb_private_discards,
                            *public_exclude,
                        ]
                    )
                )
                records.append(
                    {
                        "turn": 4,
                        "board": board_to_dict(opponent),
                        "opponent_board": board_to_dict(btn11),
                        "dealt": list(bb_t4_draw),
                        "exclude": physical_dead,
                    }
                )
                tasks.append(_LeafTask(root_index, outer, None, 0))
    else:
        if len(unseen) < 8:
            raise ValueError("BB T3 PIMC requires at least 8 unseen cards")
        outer_scenarios: list[
            tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]
        ] = []
        for outer in range(outer_n):
            hidden_and_draw = _sample_cards(
                unseen,
                5,
                seed=seed,
                decision_id=decision_id,
                outer_index=outer,
                node="bb_root_btn_private_and_t3",
            )
            btn_private_discards = hidden_and_draw[:2]
            btn_t3_draw = hidden_and_draw[2:]
            after_btn = _without(unseen, hidden_and_draw)
            outer_scenarios.append((btn_private_discards, btn_t3_draw, after_btn))

        for root_index, root_action in enumerate(root_actions):
            bb11 = apply_action(board, root_action)
            for outer, (btn_private_discards, btn_t3_draw, after_btn) in enumerate(outer_scenarios):
                btn_actions = get_turn_actions(list(btn_t3_draw), opponent)
                for response in btn_actions:
                    response_key = action_key(response)
                    btn11 = apply_action(opponent, response)
                    for inner in range(inner_n):
                        bb_t4_draw = _sample_cards(
                            after_btn,
                            3,
                            seed=seed,
                            decision_id=decision_id,
                            outer_index=outer,
                            inner_index=inner,
                            node="bb_root_bb_t4",
                        )
                        physical_dead = list(
                            dict.fromkeys(
                                [
                                    *known_self,
                                    root_action.discard,
                                    *btn_private_discards,
                                    response.discard,
                                    *public_exclude,
                                ]
                            )
                        )
                        records.append(
                            {
                                "turn": 4,
                                "board": board_to_dict(bb11),
                                "opponent_board": board_to_dict(btn11),
                                "dealt": list(bb_t4_draw),
                                "exclude": physical_dead,
                            }
                        )
                        tasks.append(_LeafTask(root_index, outer, response_key, inner))

    leaf_started = time.perf_counter()
    leaf_results = evaluate_late_positions_rust_batch(
        records,
        top_n=1,
        rust_solver_path=rust_solver_path,
        timeout_s=rust_timeout_s,
        position_parallel=True,
    )
    leaf_wall_ms = (time.perf_counter() - leaf_started) * 1000.0
    response_draws_enumerated = 0
    scenario_values: dict[int, list[float]] = {index: [] for index in range(len(root_actions))}

    if position == "btn":
        for task, result in zip(tasks, leaf_results):
            bb_score, draws = _best_leaf_score(result)
            response_draws_enumerated += draws
            scenario_values[task.root_index].append(-bb_score)
    else:
        response_values: dict[tuple[int, int, str], list[float]] = defaultdict(list)
        for task, result in zip(tasks, leaf_results):
            bb_score, draws = _best_leaf_score(result)
            response_draws_enumerated += draws
            if task.response_key is None:
                raise RuntimeError("BB T3 leaf task is missing a BTN response key")
            response_values[(task.root_index, task.outer_index, task.response_key)].append(bb_score)
        for root_index in range(len(root_actions)):
            for outer in range(outer_n):
                means = [
                    sum(values) / len(values)
                    for (candidate_index, outer_index, _response), values in response_values.items()
                    if candidate_index == root_index and outer_index == outer
                ]
                if not means:
                    raise RuntimeError(f"BB T3 root {root_index} outer {outer} has no BTN responses")
                scenario_values[root_index].append(min(means))

    candidates: list[dict[str, Any]] = []
    for index, action in enumerate(root_actions):
        metrics = {
            **_summary_stats(scenario_values[index]),
            "source": METHOD,
            "hu_exact": False,
            "inner_t4_exact": True,
            "outer_samples": outer_n,
            "response_inner_samples": inner_n,
            "seed": int(seed),
            "information_model": INFORMATION_MODEL,
        }
        candidate = {
            "action": action_to_dict(action),
            "action_key": root_keys[index],
            "board": board_to_dict(apply_action(board, action)),
            "metrics": metrics,
        }
        if include_scenario_values:
            candidate["scenario_values"] = list(scenario_values[index])
        candidates.append(candidate)
    candidates.sort(key=lambda candidate: (-float(candidate["metrics"]["score"]), candidate["action_key"]))
    best = candidates[0]
    runner_up = candidates[1] if len(candidates) > 1 else None
    paired_gap = None
    if runner_up is not None:
        best_index = root_keys.index(best["action_key"])
        runner_index = root_keys.index(runner_up["action_key"])
        gap_values = [
            best_value - runner_value
            for best_value, runner_value in zip(
                scenario_values[best_index],
                scenario_values[runner_index],
            )
        ]
        paired_gap = _summary_stats(gap_values)
        paired_gap["decision_confident"] = bool(len(gap_values) > 1 and paired_gap["ci95"][0] > 0.0)

    solver_elapsed_sum = sum(float(result.get("elapsed_ms", 0.0) or 0.0) for result in leaf_results)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "schema": "ofc_hu_gate/v1",
        "turn": 3,
        "position": position,
        "actor": position,
        "first_actor": "bb",
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "method": METHOD,
        "candidate_scope": "all_legal",
        "information_model": INFORMATION_MODEL,
        "hu_exact": False,
        "inner_t4_exact": True,
        "decision_id": decision_id,
        "seed": int(seed),
        "outer_samples": outer_n,
        "response_inner_samples": inner_n,
        "legal_actions": len(root_actions),
        "evaluated_actions": len(candidates),
        "leaf_positions": len(records),
        "response_draws_enumerated": response_draws_enumerated,
        "chosen_action": best["action"],
        "best": best,
        "runner_up": runner_up,
        "paired_gap": paired_gap,
        "candidates": candidates,
        "leaf_solver_elapsed_ms_sum": solver_elapsed_sum,
        "leaf_batch_wall_ms": leaf_wall_ms,
        "elapsed_ms": elapsed_ms,
        "sampling_replacement": "deterministic_priority_with_replacement_across_samples",
        "response_selection": "mean_before_min",
    }
