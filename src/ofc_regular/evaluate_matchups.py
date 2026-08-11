"""Seat-swapped matchup evaluation and hand tracing for regular OFC AI."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .action_key import ACTION_KEY_SCHEMA, action_key, action_key_from_payload
from .ai_profiles import (
    DEFAULT_HU_TURN0_STAGE19_P0_MODEL,
    DEFAULT_HU_TURN0_STAGE19_P0_SAFE_SELECTOR_MODEL,
    DEFAULT_HU_TURN1_STAGE1_MODEL,
    DEFAULT_HU_TURN1_STAGE2_6MODEL_POOL,
    DEFAULT_HU_TURN1_STAGE18_P1_MODEL,
    DEFAULT_HU_TURN1_STAGE18_P1_SAFE_SELECTOR_MODEL,
    DEFAULT_HU_TURN2_STAGE8B_MODEL,
    DEFAULT_OLD_OPENING_MODEL,
    DEFAULT_OPENING_MODEL,
    DEFAULT_TURN1_MODEL,
    DEFAULT_TURN2_MODEL,
    DEFAULT_TURN3_MODEL,
    DEFAULT_HU_TURN1_SAFE_SELECTOR_MODEL,
    DEFAULT_HU_TURN3_CANDIDATE_MODEL,
    DEFAULT_HU_TURN3_STAGE7_MODEL,
    DEFAULT_HU_TURN3_STAGE7_REFERENCE_MODEL,
    DEFAULT_HU_TURN3_STAGE9D_GATE_MODEL,
    DEFAULT_HU_TURN3_STAGE9D_MODEL,
    DEFAULT_HU_TURN3_STAGE9D_SUPPORT_MODEL,
    ModelPaths,
    build_policy,
    load_model_bundle,
    required_profiles,
)
from .cards import create_deck
from .decision_trace import attach_replay_truth, capture_decision_log_positions
from .evaluator import BoardScore, score_board
from .hu_infoset import ReplayTruth, WorldState
from .hu_m3_t4_runtime import (
    HuM3T4ExactPolicy,
    HuM3T4ExactSolver,
    HuM3T4RuntimeConfig,
)
from .play_ai import (
    _choose_from_observation,
    _hand_decision_seed,
    _prediction_thread_context,
)
from .rules import check_fl_entry
from .state import Board
from .teacher import DEFAULT_FL_EV, terminal_score

PROFILE_CHOICES = (
    "current",
    "stage3_baseline",
    "stage7_off",
    "stage7_m5_r10",
    "stage7_m4_r10",
    "stage7_m3_r10_experiment",
    "stage7_m3_r12_experiment",
    "stage9f_cse1p5_firstseat",
    "stage9f_cse1p5_csemax2p5_firstseat",
    "stage9f_cse2_csemax2p5_firstseat",
    "stage9f_cse2_csemax2_firstseat",
    "stage9f_cse2_csemax2_bothseat",
    "stage9f_cse2_csemax2_rank1_firstseat",
    "stage9f_p2",
    "stage9f_p2_hu_t1_stage1",
    "stage9f_p2_hu_t1_topk_confirm",
    "stage18_p1",
    "stage19_p0",
    "stage9f_fast_t2_t1_teacher",
    "stage9d_p07_relaxed_both",
    "old_opening",
    "random_exact_final",
    "late_t2t3",
    "hu_t3_candidate",
)


def board_to_json(board: Board) -> dict[str, list[str]]:
    return {
        "top": list(board.top),
        "middle": list(board.middle),
        "bottom": list(board.bottom),
    }


def board_score_to_json(score: BoardScore) -> dict[str, Any]:
    return {
        "busted": score.busted,
        "top_royalty": score.top_royalty,
        "middle_royalty": score.middle_royalty,
        "bottom_royalty": score.bottom_royalty,
        "total_royalty": score.total_royalty,
        "fl_entry": {
            "qualifies": score.fl_entry.qualifies,
            "card_count": score.fl_entry.card_count,
            "entry_type": score.fl_entry.entry_type,
        },
    }


def classify_board(board: Board) -> str:
    """Classify a completed board for FL-aware hand review."""
    score = score_board(board.top, board.middle, board.bottom)
    top_fl = check_fl_entry(board.top)
    if score.busted:
        return "FL狙いバースト" if top_fl.qualifies else "通常バースト"
    return "FL成功" if score.fl_entry.qualifies else "通常完成"


def trace_hand(
    *,
    seed: int,
    profile_p0: str,
    profile_p1: str,
    policy_p0: Any,
    policy_p1: Any,
) -> dict[str, Any]:
    rng = random.Random(seed)
    deck = create_deck(shuffle=True, rng=rng)
    cursor = 0
    boards = [Board.from_rows(), Board.from_rows()]
    policies = [policy_p0, policy_p1]
    profiles = [profile_p0, profile_p1]
    turns: list[dict[str, Any]] = []
    private_discards: list[list[str]] = [[], []]

    def choose_traced(player: int, dealt: Iterable[str], street: str):
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
        positions = capture_decision_log_positions(policies[player])
        action = _choose_from_observation(
            policies[player],
            observation,
            hand_id=seed,
            game_id=seed,
            decision_seed=_hand_decision_seed(
                base_seed=seed, observation=observation
            ),
        )
        attach_replay_truth(
            positions,
            ReplayTruth.from_world(
                world,
                actor=player,
                observation=observation,
            ),
        )
        return action

    for player in (0, 1):
        dealt = deck[cursor : cursor + 5]
        cursor += 5
        action = choose_traced(player, dealt, "T0")
        boards[player] = boards[player].place(action.placements)
        private_discards[player].extend(action.discards)
        turns.append(_turn_record("T0", player, profiles[player], dealt, action, boards[player]))

    for round_index in range(1, 5):
        for player in (0, 1):
            dealt = deck[cursor : cursor + 3]
            cursor += 3
            action = choose_traced(player, dealt, f"T{round_index}")
            boards[player] = boards[player].place(action.placements)
            private_discards[player].extend(action.discards)
            turns.append(
                _turn_record(
                    f"T{round_index}",
                    player,
                    profiles[player],
                    dealt,
                    action,
                    boards[player],
                )
            )

    score_p0, board_score_p0 = terminal_score(boards[0], boards[1], fl_ev=DEFAULT_FL_EV)
    _reverse_score, board_score_p1 = terminal_score(boards[1], boards[0], fl_ev=DEFAULT_FL_EV)
    return {
        "seed": seed,
        "profiles": {"p0": profile_p0, "p1": profile_p1},
        "score_p0": score_p0,
        "final": {"p0": board_to_json(boards[0]), "p1": board_to_json(boards[1])},
        "board_scores": {
            "p0": board_score_to_json(board_score_p0),
            "p1": board_score_to_json(board_score_p1),
        },
        "classifications": {
            "p0": classify_board(boards[0]),
            "p1": classify_board(boards[1]),
        },
        "turns": turns,
    }


def _turn_record(turn: str, player: int, profile: str, dealt: Iterable[str], action: Any, board: Board) -> dict[str, Any]:
    return {
        "turn": turn,
        "player": player,
        "profile": profile,
        "dealt": list(dealt),
        "placements": [list(placement) for placement in action.placements],
        "discards": list(action.discards),
        "action_key_schema": ACTION_KEY_SCHEMA,
        "action_key": action_key(action).to_token(),
        "board": board_to_json(board),
    }


def summarize_scores(scores: list[float]) -> dict[str, float]:
    if not scores:
        return {
            "avg_score_per_hand_for_a": 0.0,
            "std_error": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
        }
    average = sum(scores) / len(scores)
    variance = sum((score - average) ** 2 for score in scores) / max(len(scores) - 1, 1)
    stderr = math.sqrt(variance / len(scores))
    return {
        "avg_score_per_hand_for_a": average,
        "std_error": stderr,
        "ci95_low": average - 1.96 * stderr,
        "ci95_high": average + 1.96 * stderr,
    }


def evaluate_matchup(
    *,
    profile_a: str,
    profile_b: str,
    games: int,
    seed: int,
    bundle: Any,
    opening_lookahead_samples: int,
    seed_stride: int = 1,
    trace_output: Path | None = None,
    trace_limit: int = 0,
    topk_decision_output: Path | None = None,
    hu_turn0_decision_output: Path | None = None,
    hu_turn1_decision_output: Path | None = None,
    hu_t4_decision_output: Path | None = None,
    t4_solver_a: HuM3T4ExactSolver | None = None,
    t4_solver_b: HuM3T4ExactSolver | None = None,
    hu_turn1_min_margin: float | None = None,
    hu_turn1_topk_config: str | None = None,
    progress_every: int = 0,
) -> dict[str, Any]:
    if games <= 0:
        raise ValueError("games must be positive")
    if seed_stride <= 0:
        raise ValueError("seed_stride must be positive")
    started_at = time.time()
    paired_scores: list[float] = []
    wins = losses = ties = 0
    profile_classes: dict[str, Counter[str]] = defaultdict(Counter)
    trace_count = 0
    topk_decisions: list[dict[str, Any]] = []
    topk_decisions_written = 0
    hu_turn0_decisions: list[dict[str, Any]] = []
    hu_turn0_decisions_written = 0
    hu_turn1_decisions: list[dict[str, Any]] = []
    hu_turn1_decisions_written = 0
    hu_t4_decisions: list[dict[str, Any]] = []
    hu_t4_decisions_written = 0

    trace_context = (
        trace_output.open("w", encoding="utf-8") if trace_output is not None else contextlib.nullcontext(None)
    )
    topk_context = (
        topk_decision_output.open("w", encoding="utf-8")
        if topk_decision_output is not None
        else contextlib.nullcontext(None)
    )
    hu_turn0_context = (
        hu_turn0_decision_output.open("w", encoding="utf-8")
        if hu_turn0_decision_output is not None
        else contextlib.nullcontext(None)
    )
    hu_turn1_context = (
        hu_turn1_decision_output.open("w", encoding="utf-8")
        if hu_turn1_decision_output is not None
        else contextlib.nullcontext(None)
    )
    hu_t4_context = (
        hu_t4_decision_output.open("w", encoding="utf-8")
        if hu_t4_decision_output is not None
        else contextlib.nullcontext(None)
    )

    def make_policy(
        profile: str,
        *,
        policy_seed: int,
        seat: str,
        t4_solver: HuM3T4ExactSolver | None,
        capture_logs: bool = True,
    ) -> Any:
        build_kwargs: dict[str, Any] = {
            "seed": policy_seed,
            "seat": seat,
            "opening_lookahead_samples": opening_lookahead_samples,
        }
        if hu_turn1_min_margin is not None:
            build_kwargs["hu_turn1_min_margin"] = hu_turn1_min_margin
        if hu_turn1_topk_config is not None:
            build_kwargs["hu_turn1_topk_config"] = hu_turn1_topk_config
        policy = build_policy(profile, bundle, **build_kwargs)
        if capture_logs and hasattr(policy, "topk_decision_log"):
            policy.topk_decision_log = topk_decisions
        if capture_logs and hasattr(policy, "hu_turn0_decision_log"):
            policy.hu_turn0_decision_log = hu_turn0_decisions
        if capture_logs and hasattr(policy, "hu_turn1_decision_log"):
            policy.hu_turn1_decision_log = hu_turn1_decisions
        if t4_solver is not None:
            policy = HuM3T4ExactPolicy(
                policy,
                t4_solver,
                decision_log=hu_t4_decisions if capture_logs else None,
            )
        return policy

    def flush_decisions(handle: Any, records: list[dict[str, Any]], cursor: int) -> int:
        if handle is None:
            return cursor
        for record in records[cursor:]:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        if cursor < len(records):
            handle.flush()
        return len(records)

    with (
        trace_context as trace_handle,
        topk_context as topk_handle,
        hu_turn0_context as hu_turn0_handle,
        hu_turn1_context as hu_turn1_handle,
        hu_t4_context as hu_t4_handle,
    ):
        for index in range(games):
            hand_seed = seed + index * seed_stride
            before_ab = len(topk_decisions)
            before_ab_t0 = len(hu_turn0_decisions)
            before_ab_t1 = len(hu_turn1_decisions)
            before_ab_t4 = len(hu_t4_decisions)
            first_policy_seed = hand_seed * 4
            second_policy_seed = hand_seed * 4 + 1
            hand_ab = trace_hand(
                seed=hand_seed,
                profile_p0=profile_a,
                profile_p1=profile_b,
                policy_p0=make_policy(
                    profile_a,
                    policy_seed=first_policy_seed,
                    seat="first",
                    t4_solver=t4_solver_a,
                ),
                policy_p1=make_policy(
                    profile_b,
                    policy_seed=second_policy_seed,
                    seat="second",
                    t4_solver=t4_solver_b,
                ),
            )
            ab_rows = topk_decisions[before_ab:]
            ab_t0_rows = hu_turn0_decisions[before_ab_t0:]
            ab_t1_rows = hu_turn1_decisions[before_ab_t1:]
            ab_t4_rows = hu_t4_decisions[before_ab_t4:]
            shadow_ab = trace_hand(
                seed=hand_seed,
                profile_p0=profile_a,
                profile_p1=profile_b,
                policy_p0=make_policy(
                    profile_a,
                    policy_seed=first_policy_seed,
                    seat="first",
                    t4_solver=t4_solver_a,
                    capture_logs=False,
                ),
                policy_p1=make_policy(
                    profile_b,
                    policy_seed=second_policy_seed,
                    seat="second",
                    t4_solver=t4_solver_b,
                    capture_logs=False,
                ),
            )

            before_ba = len(topk_decisions)
            before_ba_t0 = len(hu_turn0_decisions)
            before_ba_t1 = len(hu_turn1_decisions)
            before_ba_t4 = len(hu_t4_decisions)
            hand_ba = trace_hand(
                seed=hand_seed,
                profile_p0=profile_b,
                profile_p1=profile_a,
                policy_p0=make_policy(
                    profile_b,
                    policy_seed=first_policy_seed,
                    seat="first",
                    t4_solver=t4_solver_b,
                ),
                policy_p1=make_policy(
                    profile_a,
                    policy_seed=second_policy_seed,
                    seat="second",
                    t4_solver=t4_solver_a,
                ),
            )
            ba_rows = topk_decisions[before_ba:]
            ba_t0_rows = hu_turn0_decisions[before_ba_t0:]
            ba_t1_rows = hu_turn1_decisions[before_ba_t1:]
            ba_t4_rows = hu_t4_decisions[before_ba_t4:]
            shadow_ba = trace_hand(
                seed=hand_seed,
                profile_p0=profile_b,
                profile_p1=profile_a,
                policy_p0=make_policy(
                    profile_b,
                    policy_seed=first_policy_seed,
                    seat="first",
                    t4_solver=t4_solver_b,
                    capture_logs=False,
                ),
                policy_p1=make_policy(
                    profile_a,
                    policy_seed=second_policy_seed,
                    seat="second",
                    t4_solver=t4_solver_a,
                    capture_logs=False,
                ),
            )
            _augment_profile_topk_counterfactuals(
                ab_rows=ab_rows,
                ba_rows=ba_rows,
                hand_ab=hand_ab,
                hand_ba=hand_ba,
                shadow_ab=shadow_ab,
                shadow_ba=shadow_ba,
                paired_index=index,
                hand_seed=hand_seed,
            )
            _augment_profile_topk_counterfactuals(
                ab_rows=ab_t0_rows,
                ba_rows=ba_t0_rows,
                hand_ab=hand_ab,
                hand_ba=hand_ba,
                shadow_ab=shadow_ab,
                shadow_ba=shadow_ba,
                paired_index=index,
                hand_seed=hand_seed,
            )
            _augment_profile_topk_counterfactuals(
                ab_rows=ab_t1_rows,
                ba_rows=ba_t1_rows,
                hand_ab=hand_ab,
                hand_ba=hand_ba,
                shadow_ab=shadow_ab,
                shadow_ba=shadow_ba,
                paired_index=index,
                hand_seed=hand_seed,
            )
            _augment_t4_runtime_decisions(
                rows=ab_t4_rows,
                hand=hand_ab,
                paired_index=index,
                hand_seed=hand_seed,
                seat_swap="ab",
                first_role="a",
                second_role="b",
            )
            _augment_t4_runtime_decisions(
                rows=ba_t4_rows,
                hand=hand_ba,
                paired_index=index,
                hand_seed=hand_seed,
                seat_swap="ba",
                first_role="b",
                second_role="a",
            )
            topk_decisions_written = flush_decisions(
                topk_handle,
                topk_decisions,
                topk_decisions_written,
            )
            hu_turn0_decisions_written = flush_decisions(
                hu_turn0_handle,
                hu_turn0_decisions,
                hu_turn0_decisions_written,
            )
            hu_turn1_decisions_written = flush_decisions(
                hu_turn1_handle,
                hu_turn1_decisions,
                hu_turn1_decisions_written,
            )
            hu_t4_decisions_written = flush_decisions(
                hu_t4_handle,
                hu_t4_decisions,
                hu_t4_decisions_written,
            )
            paired_score = (float(hand_ab["score_p0"]) - float(hand_ba["score_p0"])) / 2.0
            paired_scores.append(paired_score)
            if paired_score > 0:
                wins += 1
            elif paired_score < 0:
                losses += 1
            else:
                ties += 1

            _add_classification_counts(profile_classes, hand_ab)
            _add_classification_counts(profile_classes, hand_ba)
            if trace_handle is not None and (trace_limit <= 0 or trace_count < trace_limit):
                trace_handle.write(json.dumps(hand_ab, ensure_ascii=False, separators=(",", ":")) + "\n")
                trace_count += 1
            if trace_handle is not None and (trace_limit <= 0 or trace_count < trace_limit):
                trace_handle.write(json.dumps(hand_ba, ensure_ascii=False, separators=(",", ":")) + "\n")
                trace_count += 1

            if progress_every > 0 and (index + 1) % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "paired_seeds": index + 1,
                            "hands": (index + 1) * 2,
                            **summarize_scores(paired_scores),
                            "elapsed_seconds": time.time() - started_at,
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )

    return {
        "profile_a": profile_a,
        "profile_b": profile_b,
        "paired_seeds": games,
        "hands": games * 2,
        "seed": seed,
        "seed_stride": seed_stride,
        "policy_seed_pairing": "seat_stable",
        **summarize_scores(paired_scores),
        "paired_seed_wins": wins,
        "paired_seed_losses": losses,
        "paired_seed_ties": ties,
        "classification_counts": {
            profile: dict(counter) for profile, counter in sorted(profile_classes.items())
        },
        "classification_rates": {
            profile: _classification_rates(counter)
            for profile, counter in sorted(profile_classes.items())
        },
        "trace_output": str(trace_output) if trace_output else None,
        "trace_hands_written": trace_count,
        "topk_decision_output": str(topk_decision_output) if topk_decision_output else None,
        "topk_decisions_written": topk_decisions_written,
        **_summarize_topk_realized_deltas(topk_decisions),
        "hu_turn0_decision_output": (
            str(hu_turn0_decision_output) if hu_turn0_decision_output else None
        ),
        "hu_turn0_decisions_written": hu_turn0_decisions_written,
        **_summarize_hu_turn0_realized_deltas(hu_turn0_decisions),
        "hu_turn1_decision_output": (
            str(hu_turn1_decision_output) if hu_turn1_decision_output else None
        ),
        "hu_turn1_decisions_written": hu_turn1_decisions_written,
        **_summarize_hu_turn1_realized_deltas(hu_turn1_decisions),
        "t4_mode_a": "m30_exact" if t4_solver_a is not None else "legacy",
        "t4_mode_b": "m30_exact" if t4_solver_b is not None else "legacy",
        "hu_t4_decision_output": (
            str(hu_t4_decision_output) if hu_t4_decision_output else None
        ),
        "hu_t4_decisions_written": hu_t4_decisions_written,
        "hu_t4_runtime": _t4_runtime_summary(t4_solver_a, t4_solver_b, hu_t4_decisions),
        "hu_t4_realized_pair": _summarize_t4_realized_pairs(
            hu_t4_decisions,
            pure_t4_counterfactual=(
                profile_a == profile_b
                and t4_solver_a is not None
                and t4_solver_b is None
            ),
        ),
        "elapsed_seconds": time.time() - started_at,
    }


def _augment_t4_runtime_decisions(
    *,
    rows: list[dict[str, Any]],
    hand: dict[str, Any],
    paired_index: int,
    hand_seed: int,
    seat_swap: str,
    first_role: str,
    second_role: str,
) -> None:
    trajectory_digest = _hand_trajectory_digest(hand)
    for row in rows:
        seat = str(row.get("seat", ""))
        if seat not in {"first", "second"}:
            raise ValueError("T4 runtime decision has invalid seat")
        seat_score = float(hand["score_p0"])
        if seat == "second":
            seat_score = -seat_score
        row.update(
            {
                "paired_index": paired_index,
                "hand_seed": hand_seed,
                "seat_swap": seat_swap,
                "policy_role": first_role if seat == "first" else second_role,
                "realized_seat_score": seat_score,
                "hand_trajectory_digest": trajectory_digest,
            }
        )


def _t4_runtime_summary(
    solver_a: HuM3T4ExactSolver | None,
    solver_b: HuM3T4ExactSolver | None,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    enabled = [solver for solver in (solver_a, solver_b) if solver is not None]

    def latency_summary(selected: list[dict[str, Any]]) -> dict[str, Any]:
        latencies = sorted(float(row["total_latency_ms"]) for row in selected)

        def percentile(fraction: float) -> float | None:
            if not latencies:
                return None
            index = max(
                0,
                min(len(latencies) - 1, math.ceil(len(latencies) * fraction) - 1),
            )
            return latencies[index]

        return {
            "count": len(latencies),
            "p50_ms": percentile(0.50),
            "p95_ms": percentile(0.95),
            "p99_ms": percentile(0.99),
            "max_ms": max(latencies) if latencies else None,
        }

    all_latency = latency_summary(rows)
    by_seat = {
        seat: latency_summary([row for row in rows if row.get("seat") == seat])
        for seat in ("first", "second")
    }

    return {
        "enabled": bool(enabled),
        "strict_no_fallback": bool(enabled),
        "library_sha256": (
            sorted({solver.library_sha256 for solver in enabled}) if enabled else []
        ),
        "engine_versions": (
            sorted({solver.engine_version for solver in enabled}) if enabled else []
        ),
        "decision_count": len(rows),
        "first_decision_count": sum(row.get("seat") == "first" for row in rows),
        "second_decision_count": sum(row.get("seat") == "second" for row in rows),
        "latency_p50_ms": all_latency["p50_ms"],
        "latency_p95_ms": all_latency["p95_ms"],
        "latency_p99_ms": all_latency["p99_ms"],
        "latency_max_ms": all_latency["max_ms"],
        "latency_by_seat": by_seat,
    }


def _summarize_t4_realized_pairs(
    rows: list[dict[str, Any]], *, pure_t4_counterfactual: bool
) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("policy_role") != "a":
            continue
        paired_index = row.get("paired_index")
        if isinstance(paired_index, int) and not isinstance(paired_index, bool):
            grouped[paired_index].append(row)

    paired_scores: list[float] = []
    override_gains: list[float] = []
    nonfire_scores: list[float] = []
    nonfire_trajectory_mismatches = 0
    second_seat_overrides = 0
    incomplete = 0
    invalid_pairs = 0
    for paired_index in sorted(grouped):
        pair = grouped[paired_index]
        if len(pair) != 2:
            if len(pair) < 2:
                incomplete += 1
            else:
                invalid_pairs += 1
            continue
        by_seat = {str(row.get("seat")): row for row in pair}
        if set(by_seat) != {"first", "second"}:
            incomplete += 1
            continue
        if any(not isinstance(row.get("override_fired"), bool) for row in pair):
            invalid_pairs += 1
            continue
        counterfactual_gain = (
            float(by_seat["first"]["realized_seat_score"])
            + float(by_seat["second"]["realized_seat_score"])
        )
        paired_score = counterfactual_gain / 2.0
        paired_scores.append(paired_score)
        fired_rows = [
            row for row in by_seat.values() if row.get("override_fired") is True
        ]
        second_seat_overrides += int(
            by_seat["second"].get("override_fired") is True
        )
        if fired_rows:
            per_decision_gain = counterfactual_gain / len(fired_rows)
            override_gains.extend(per_decision_gain for _ in fired_rows)
        elif all(row.get("override_fired") is False for row in by_seat.values()):
            nonfire_scores.append(paired_score)
            digests = {
                row.get("hand_trajectory_digest") for row in by_seat.values()
            }
            if len(digests) != 1 or None in digests:
                nonfire_trajectory_mismatches += 1

    counterfactual_available = (
        pure_t4_counterfactual
        and bool(paired_scores)
        and incomplete == 0
        and invalid_pairs == 0
    )
    reported_override_gains = override_gains if counterfactual_available else []
    losses = sorted(max(0.0, -gain) for gain in reported_override_gains)
    false_positives = sum(gain < 0.0 for gain in reported_override_gains)
    summary = summarize_scores(paired_scores)
    cancellation_evaluable = counterfactual_available and bool(nonfire_scores)
    nonfire_nonzero = sum(score != 0.0 for score in nonfire_scores)
    return {
        "available": counterfactual_available,
        "pure_t4_counterfactual": pure_t4_counterfactual,
        "paired_count": len(paired_scores),
        "incomplete_pair_count": incomplete,
        "invalid_pair_count": invalid_pairs,
        **summary,
        "override_count": (
            len(reported_override_gains) if counterfactual_available else None
        ),
        "second_seat_override_count": (
            second_seat_overrides if counterfactual_available else None
        ),
        "realized_gain_per_override": (
            sum(reported_override_gains) / len(reported_override_gains)
            if reported_override_gains
            else (0.0 if counterfactual_available else None)
        ),
        "false_positive_override_count": (
            false_positives if counterfactual_available else None
        ),
        "false_positive_override_rate": (
            false_positives / len(reported_override_gains)
            if reported_override_gains
            else (0.0 if counterfactual_available else None)
        ),
        "override_p95_tail_loss": (
            _quantile_nearest_rank(losses, 0.95)
            if counterfactual_available
            else None
        ),
        "override_p99_tail_loss": (
            _quantile_nearest_rank(losses, 0.99)
            if counterfactual_available
            else None
        ),
        "override_max_tail_loss": (
            max(losses) if losses else (0.0 if counterfactual_available else None)
        ),
        "nonfire_count": len(nonfire_scores) if counterfactual_available else None,
        "nonfire_cancellation_evaluable": cancellation_evaluable,
        "nonfire_exact_cancellation": (
            cancellation_evaluable
            and nonfire_nonzero == 0
            and nonfire_trajectory_mismatches == 0
        ),
        "nonfire_nonzero_count": (
            nonfire_nonzero if counterfactual_available else None
        ),
        "nonfire_trajectory_mismatch_count": (
            nonfire_trajectory_mismatches if counterfactual_available else None
        ),
    }


def _quantile_nearest_rank(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    index = max(0, min(len(values) - 1, math.ceil(len(values) * fraction) - 1))
    return values[index]


def _hand_trajectory_digest(hand: dict[str, Any]) -> str:
    payload = {
        "seed": hand.get("seed"),
        "profiles": hand.get("profiles"),
        "turns": hand.get("turns"),
        "final": hand.get("final"),
        "score_p0": hand.get("score_p0"),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _augment_profile_topk_counterfactuals(
    *,
    ab_rows: list[dict[str, Any]],
    ba_rows: list[dict[str, Any]],
    hand_ab: dict[str, Any],
    hand_ba: dict[str, Any],
    shadow_ab: dict[str, Any],
    shadow_ba: dict[str, Any],
    paired_index: int,
    hand_seed: int,
) -> None:
    """Verify non-fire cancellation on a repeated identical world trajectory.

    Seat-swap hands are independent evaluation samples and are never used as a
    baseline counterfactual.  A non-fired decision is valid only when its final
    semantic action equals its baseline action *and* a fresh same-seed replay
    reproduces every action, board, discard, and terminal result.
    """

    def attach(
        rows: list[dict[str, Any]],
        *,
        hand: dict[str, Any],
        shadow: dict[str, Any],
        seat_swap: str,
    ) -> None:
        actual_digest = _hand_trajectory_digest(hand)
        shadow_digest = _hand_trajectory_digest(shadow)
        replay_identical = actual_digest == shadow_digest
        for row in rows:
            seat = str(row.get("seat", "first"))
            seat_score = float(hand["score_p0"])
            if seat == "second":
                seat_score = -seat_score
            fired = bool(row.get("override_fired"))
            final_key, final_key_valid = _row_action_key(row, "final")
            baseline_key, baseline_key_valid = _row_action_key(row, "baseline")
            if baseline_key is None and baseline_key_valid:
                baseline_key, baseline_key_valid = _row_action_key(row, "fallback")
            action_identical = (
                final_key_valid
                and baseline_key_valid
                and final_key is not None
                and final_key == baseline_key
            )
            cancellation_valid = (not fired) and action_identical and replay_identical
            row.update(
                {
                    "paired_index": paired_index,
                    "hand_seed": hand_seed,
                    "seat_swap": seat_swap,
                    "candidate_seat_score": seat_score,
                    "baseline_seat_score": seat_score if cancellation_valid else None,
                    "realized_delta_valid": cancellation_valid,
                    "realized_delta_basis": (
                        "same_snapshot_nonfire_trajectory_identity_v1"
                        if not fired
                        else "same_snapshot_baseline_shadow_required"
                    ),
                    "actual_trajectory_digest": actual_digest,
                    "shadow_trajectory_digest": shadow_digest,
                    "counterfactual_events_identical": replay_identical,
                    "nonfire_action_key_identical": action_identical,
                    "nonfire_cancellation_valid": cancellation_valid,
                }
            )
            if cancellation_valid:
                row["realized_candidate_seat_delta"] = 0.0
            else:
                row.pop("realized_candidate_seat_delta", None)

    attach(ab_rows, hand=hand_ab, shadow=shadow_ab, seat_swap="ab")
    attach(ba_rows, hand=hand_ba, shadow=shadow_ba, seat_swap="ba")


def _summarize_topk_realized_deltas(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _summarize_realized_deltas(rows, "topk")


def _summarize_hu_turn1_realized_deltas(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _summarize_realized_deltas(rows, "hu_turn1")


def _summarize_hu_turn0_realized_deltas(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _summarize_realized_deltas(rows, "hu_turn0")


def _summarize_realized_deltas(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    valid = [
        row
        for row in rows
        if row.get("realized_delta_valid")
        and row.get("realized_candidate_seat_delta") not in (None, "")
    ]
    fired = [row for row in valid if bool(row.get("override_fired"))]
    non_fired = [row for row in valid if not row.get("override_fired")]
    fired_deltas = [float(row.get("realized_candidate_seat_delta", 0.0) or 0.0) for row in fired]
    non_fired_deltas = [
        float(row.get("realized_candidate_seat_delta", 0.0) or 0.0) for row in non_fired
    ]
    nonzero = [delta for delta in non_fired_deltas if abs(delta) > 1e-9]
    final_match_counts = Counter(_final_action_match_status(row) for row in non_fired)
    return {
        f"{prefix}_realized_delta_count": len(valid),
        f"{prefix}_realized_override_count": len(fired),
        f"{prefix}_realized_override_delta_mean": (
            sum(fired_deltas) / len(fired_deltas) if fired_deltas else 0.0
        ),
        f"{prefix}_non_fired_nonzero_count": len(nonzero),
        f"{prefix}_non_fired_counterfactual_nonzero_count": len(nonzero),
        f"{prefix}_non_fired_final_matches_baseline_count": final_match_counts["match"],
        f"{prefix}_non_fired_final_mismatch_count": final_match_counts["mismatch"],
        f"{prefix}_non_fired_final_match_unknown_count": final_match_counts["unknown"],
        f"{prefix}_non_fired_delta_sum": sum(non_fired_deltas),
        f"{prefix}_non_fired_delta_max_abs": max(
            (abs(delta) for delta in non_fired_deltas),
            default=0.0,
        ),
    }


def _final_action_match_status(row: dict[str, Any]) -> str:
    final_key, final_key_valid = _row_action_key(row, "final")
    baseline_key, baseline_key_valid = _row_action_key(row, "baseline")
    if baseline_key is None and baseline_key_valid:
        baseline_key, baseline_key_valid = _row_action_key(row, "fallback")
    if not final_key_valid or not baseline_key_valid:
        return "mismatch"
    if final_key is not None or baseline_key is not None:
        if final_key is None or baseline_key is None:
            return "unknown"
        return "match" if final_key == baseline_key else "mismatch"

    # Positional indices are a legacy fallback only.  They are meaningful only
    # when the producer also persisted the legal-action order digest that bound
    # those indices to a specific enumeration.
    order_digest = row.get("legal_action_order_digest")
    if not isinstance(order_digest, str) or not order_digest:
        return "unknown"
    final_index = row.get("final_action_index")
    baseline_index = row.get("baseline_action_index", row.get("fallback_action_index"))
    if final_index is not None and baseline_index is not None:
        return "match" if final_index == baseline_index else "mismatch"
    return "unknown"


def _row_action_key(
    row: dict[str, Any], prefix: str
) -> tuple[str | None, bool]:
    """Return a checked semantic key and whether all supplied fields agree."""

    explicit = row.get(f"{prefix}_action_key")
    payload = row.get(f"{prefix}_action")
    if explicit is not None and not isinstance(explicit, str):
        return None, False
    if isinstance(payload, dict):
        try:
            derived = action_key_from_payload(payload).to_token()
        except (TypeError, ValueError):
            return None, False
        if isinstance(explicit, str) and explicit != derived:
            return None, False
        return derived, True
    if payload is not None:
        return None, False
    return (explicit if isinstance(explicit, str) else None), True


def _add_classification_counts(profile_classes: dict[str, Counter[str]], hand: dict[str, Any]) -> None:
    for player in ("p0", "p1"):
        profile = hand["profiles"][player]
        classification = hand["classifications"][player]
        profile_classes[profile][classification] += 1


def _classification_rates(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {key: value / total for key, value in sorted(counter.items())}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-a", choices=PROFILE_CHOICES, default="current")
    parser.add_argument("--profile-b", choices=PROFILE_CHOICES, default="old_opening")
    parser.add_argument("--games", type=int, default=100, help="Paired seat-swap seeds; total hands are 2x games.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-stride", type=int, default=1)
    parser.add_argument("--opening-model", type=Path, default=DEFAULT_OPENING_MODEL)
    parser.add_argument("--old-opening-model", type=Path, default=DEFAULT_OLD_OPENING_MODEL)
    parser.add_argument("--turn1-model", type=Path, default=DEFAULT_TURN1_MODEL)
    parser.add_argument("--turn2-model", type=Path, default=DEFAULT_TURN2_MODEL)
    parser.add_argument("--turn3-model", type=Path, default=DEFAULT_TURN3_MODEL)
    parser.add_argument("--hu-turn1-stage1-model", type=Path, default=DEFAULT_HU_TURN1_STAGE1_MODEL)
    parser.add_argument(
        "--hu-turn1-stage1-models",
        type=Path,
        nargs="+",
        help=(
            "Optional HU Turn1 candidate model pool for stage9f_p2_hu_t1_topk_confirm. "
            "When set, runtime candidate generation uses this pool instead of the "
            "single --hu-turn1-stage1-model path."
        ),
    )
    parser.add_argument(
        "--hu-turn1-safe-selector-model",
        type=Path,
        default=DEFAULT_HU_TURN1_SAFE_SELECTOR_MODEL,
    )
    parser.add_argument(
        "--hu-turn1-stage18-p1-model",
        type=Path,
        default=DEFAULT_HU_TURN1_STAGE18_P1_MODEL,
        help="Candidate model for the locked stage18_p1 profile.",
    )
    parser.add_argument(
        "--hu-turn1-stage18-p1-safe-selector-model",
        type=Path,
        default=DEFAULT_HU_TURN1_STAGE18_P1_SAFE_SELECTOR_MODEL,
        help="Safe-selector model for the locked stage18_p1 profile.",
    )
    parser.add_argument(
        "--hu-turn0-stage19-p0-model",
        type=Path,
        default=DEFAULT_HU_TURN0_STAGE19_P0_MODEL,
        help="Candidate model for the locked stage19_p0 profile.",
    )
    parser.add_argument(
        "--hu-turn0-stage19-p0-safe-selector-model",
        type=Path,
        default=DEFAULT_HU_TURN0_STAGE19_P0_SAFE_SELECTOR_MODEL,
        help="Safe-selector model for the locked stage19_p0 profile.",
    )
    parser.add_argument(
        "--hu-turn1-min-margin",
        type=float,
        help="Override HU Turn1 selective margin for profiles that support it.",
    )
    parser.add_argument(
        "--hu-turn1-topk-config",
        help="Override HU Turn1 TopK confirm config for stage9f_p2_hu_t1_topk_confirm.",
    )
    parser.add_argument("--hu-turn2-stage8b-model", type=Path, default=DEFAULT_HU_TURN2_STAGE8B_MODEL)
    parser.add_argument("--hu-turn3-candidate-model", type=Path, default=DEFAULT_HU_TURN3_CANDIDATE_MODEL)
    parser.add_argument("--hu-turn3-stage7-model", type=Path, default=DEFAULT_HU_TURN3_STAGE7_MODEL)
    parser.add_argument(
        "--hu-turn3-stage7-reference-model",
        type=Path,
        default=DEFAULT_HU_TURN3_STAGE7_REFERENCE_MODEL,
    )
    parser.add_argument("--hu-turn3-stage9d-model", type=Path, default=DEFAULT_HU_TURN3_STAGE9D_MODEL)
    parser.add_argument(
        "--hu-turn3-stage9d-support-model",
        type=Path,
        default=DEFAULT_HU_TURN3_STAGE9D_SUPPORT_MODEL,
    )
    parser.add_argument(
        "--hu-turn3-stage9d-gate-model",
        type=Path,
        default=DEFAULT_HU_TURN3_STAGE9D_GATE_MODEL,
    )
    parser.add_argument("--opening-lookahead-samples", type=int, default=64)
    parser.add_argument("--prediction-threads", type=int, default=1)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--trace-limit", type=int, default=0, help="Maximum traced hands; <=0 writes all hands.")
    parser.add_argument("--topk-decision-output", type=Path)
    parser.add_argument("--hu-turn0-decision-output", type=Path)
    parser.add_argument("--hu-turn1-decision-output", type=Path)
    parser.add_argument(
        "--t4-mode-a",
        choices=("legacy", "m30_exact"),
        default="legacy",
        help="Explicit T4 runtime for profile A; does not change the named profile or current.",
    )
    parser.add_argument(
        "--t4-mode-b",
        choices=("legacy", "m30_exact"),
        default="legacy",
        help="Explicit T4 runtime for profile B; does not change the named profile or current.",
    )
    parser.add_argument(
        "--hu-t4-native-library",
        type=Path,
        help="Prebuilt release M3 native library required by m30_exact.",
    )
    parser.add_argument(
        "--hu-t4-native-sha256",
        help="Pinned SHA-256 required by the CLI when m30_exact is selected.",
    )
    parser.add_argument("--hu-t4-decision-output", type=Path)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.prediction_threads > 0:
        os.environ.setdefault("OMP_NUM_THREADS", str(args.prediction_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.prediction_threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.prediction_threads))
    paths = ModelPaths(
        opening=args.opening_model,
        old_opening=args.old_opening_model,
        turn1=args.turn1_model,
        turn2=args.turn2_model,
        turn3=args.turn3_model,
        hu_turn1_stage1=args.hu_turn1_stage1_model,
        hu_turn1_stage1_models=tuple(args.hu_turn1_stage1_models or ()),
        hu_turn1_safe_selector=args.hu_turn1_safe_selector_model,
        hu_turn1_stage18_p1=args.hu_turn1_stage18_p1_model,
        hu_turn1_stage18_p1_safe_selector=args.hu_turn1_stage18_p1_safe_selector_model,
        hu_turn0_stage19_p0=args.hu_turn0_stage19_p0_model,
        hu_turn0_stage19_p0_safe_selector=args.hu_turn0_stage19_p0_safe_selector_model,
        hu_turn2_stage8b=args.hu_turn2_stage8b_model,
        hu_turn3_candidate=args.hu_turn3_candidate_model,
        hu_turn3_stage7=args.hu_turn3_stage7_model,
        hu_turn3_stage7_reference=args.hu_turn3_stage7_reference_model,
        hu_turn3_stage9d=args.hu_turn3_stage9d_model,
        hu_turn3_stage9d_support=args.hu_turn3_stage9d_support_model,
        hu_turn3_stage9d_gate=args.hu_turn3_stage9d_gate_model,
    )
    bundle = load_model_bundle(paths, required_profiles(args.profile_a, args.profile_b))
    if args.trace_output:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    if args.topk_decision_output:
        args.topk_decision_output.parent.mkdir(parents=True, exist_ok=True)
    if args.hu_turn0_decision_output:
        args.hu_turn0_decision_output.parent.mkdir(parents=True, exist_ok=True)
    if args.hu_turn1_decision_output:
        args.hu_turn1_decision_output.parent.mkdir(parents=True, exist_ok=True)
    if args.hu_t4_decision_output:
        args.hu_t4_decision_output.parent.mkdir(parents=True, exist_ok=True)
    t4_solver: HuM3T4ExactSolver | None = None
    if "m30_exact" in {args.t4_mode_a, args.t4_mode_b}:
        if args.hu_t4_native_library is None:
            raise SystemExit("m30_exact requires --hu-t4-native-library")
        if args.hu_t4_native_sha256 is None:
            raise SystemExit("m30_exact requires --hu-t4-native-sha256")
        t4_solver = HuM3T4ExactSolver(
            HuM3T4RuntimeConfig(
                library_path=args.hu_t4_native_library,
                expected_library_sha256=args.hu_t4_native_sha256,
            )
        )
    with _prediction_thread_context(args.prediction_threads):
        summary = evaluate_matchup(
            profile_a=args.profile_a,
            profile_b=args.profile_b,
            games=args.games,
            seed=args.seed,
            seed_stride=args.seed_stride,
            bundle=bundle,
            opening_lookahead_samples=args.opening_lookahead_samples,
            trace_output=args.trace_output,
            trace_limit=args.trace_limit,
            topk_decision_output=args.topk_decision_output,
            hu_turn0_decision_output=args.hu_turn0_decision_output,
            hu_turn1_decision_output=args.hu_turn1_decision_output,
            hu_t4_decision_output=args.hu_t4_decision_output,
            t4_solver_a=(t4_solver if args.t4_mode_a == "m30_exact" else None),
            t4_solver_b=(t4_solver if args.t4_mode_b == "m30_exact" else None),
            hu_turn1_min_margin=args.hu_turn1_min_margin,
            hu_turn1_topk_config=args.hu_turn1_topk_config,
            progress_every=args.progress_every,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
