"""Seat-swapped evaluation between two explicit model sets."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from .ai_profiles import (
    DEFAULT_OPENING_MODEL,
    DEFAULT_TURN1_MODEL,
    DEFAULT_TURN2_MODEL,
    DEFAULT_TURN3_MODEL,
)
from .evaluate_matchups import trace_hand
from .hu_turn3_gate_model import load_hu_turn3_gate_model
from .hu_turn3_model import load_hu_action_value_model
from .play_ai import _prediction_thread_context
from .policy import RegularAiPolicy
from .turn3_model import load_action_value_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name-a", default="candidate")
    parser.add_argument("--name-b", default="baseline")
    parser.add_argument("--opening-a", type=Path, default=DEFAULT_OPENING_MODEL)
    parser.add_argument("--turn1-a", type=Path, default=DEFAULT_TURN1_MODEL)
    parser.add_argument("--turn2-a", type=Path, default=DEFAULT_TURN2_MODEL)
    parser.add_argument("--turn3-a", type=Path, default=DEFAULT_TURN3_MODEL)
    parser.add_argument("--hu-turn3-a", type=Path)
    parser.add_argument("--hu-turn3-reference-a", type=Path)
    parser.add_argument("--hu-turn3-support-a", type=Path)
    parser.add_argument("--hu-turn3-gate-a", type=Path)
    parser.add_argument("--hu-turn3-min-margin-a", type=float, default=0.0)
    parser.add_argument("--hu-turn3-reference-min-margin-a", type=float, default=0.0)
    parser.add_argument("--hu-turn3-min-support-margin-a", type=float, default=0.0)
    parser.add_argument("--hu-turn3-min-gate-probability-a", type=float, default=0.0)
    parser.add_argument("--hu-turn3-max-self-regret-a", type=float)
    parser.add_argument("--disable-hu-turn3-stage7-a", action="store_true")
    parser.add_argument("--opening-b", type=Path, default=DEFAULT_OPENING_MODEL)
    parser.add_argument("--turn1-b", type=Path, default=DEFAULT_TURN1_MODEL)
    parser.add_argument("--turn2-b", type=Path, default=DEFAULT_TURN2_MODEL)
    parser.add_argument("--turn3-b", type=Path, default=DEFAULT_TURN3_MODEL)
    parser.add_argument("--hu-turn3-b", type=Path)
    parser.add_argument("--hu-turn3-reference-b", type=Path)
    parser.add_argument("--hu-turn3-support-b", type=Path)
    parser.add_argument("--hu-turn3-gate-b", type=Path)
    parser.add_argument("--hu-turn3-min-margin-b", type=float, default=0.0)
    parser.add_argument("--hu-turn3-reference-min-margin-b", type=float, default=0.0)
    parser.add_argument("--hu-turn3-min-support-margin-b", type=float, default=0.0)
    parser.add_argument("--hu-turn3-min-gate-probability-b", type=float, default=0.0)
    parser.add_argument("--hu-turn3-max-self-regret-b", type=float)
    parser.add_argument("--disable-hu-turn3-stage7-b", action="store_true")
    parser.add_argument("--opening-lookahead-samples", type=int, default=64)
    parser.add_argument("--prediction-threads", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--trace-limit", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_policy_parts(args: argparse.Namespace, suffix: str) -> dict[str, Any]:
    hu_path = getattr(args, f"hu_turn3_{suffix}")
    hu_reference_path = getattr(args, f"hu_turn3_reference_{suffix}")
    hu_support_path = getattr(args, f"hu_turn3_support_{suffix}")
    hu_gate_path = getattr(args, f"hu_turn3_gate_{suffix}")
    hu_model = _safe_load_hu_action_value_model(hu_path) if hu_path else None
    hu_reference_model = (
        _safe_load_hu_action_value_model(hu_reference_path)
        if hu_reference_path
        else None
    )
    hu_min_margin = getattr(args, f"hu_turn3_min_margin_{suffix}")
    hu_reference_min_margin = getattr(args, f"hu_turn3_reference_min_margin_{suffix}")
    if hu_path and hu_model is None and hu_reference_model is not None:
        hu_model = hu_reference_model
        hu_reference_model = None
        hu_min_margin = hu_reference_min_margin
        hu_reference_min_margin = 0.0
    return {
        "opening_model": load_action_value_model(getattr(args, f"opening_{suffix}")),
        "turn1_model": load_action_value_model(getattr(args, f"turn1_{suffix}")),
        "turn2_model": load_action_value_model(getattr(args, f"turn2_{suffix}")),
        "turn3_model": load_action_value_model(getattr(args, f"turn3_{suffix}")),
        "hu_turn3_model": hu_model,
        "hu_turn3_reference_model": hu_reference_model,
        "hu_turn3_support_model": _safe_load_hu_action_value_model(hu_support_path)
        if hu_support_path
        else None,
        "hu_turn3_gate_model": _safe_load_hu_turn3_gate_model(hu_gate_path) if hu_gate_path else None,
        "hu_turn3_min_margin": hu_min_margin,
        "hu_turn3_reference_min_margin": hu_reference_min_margin,
        "hu_turn3_min_support_margin": getattr(args, f"hu_turn3_min_support_margin_{suffix}"),
        "hu_turn3_min_gate_probability": getattr(
            args,
            f"hu_turn3_min_gate_probability_{suffix}",
        ),
        "hu_turn3_max_self_regret": getattr(args, f"hu_turn3_max_self_regret_{suffix}"),
        "hu_turn3_stage7_enabled": not getattr(args, f"disable_hu_turn3_stage7_{suffix}"),
        "opening_lookahead_samples": args.opening_lookahead_samples,
    }


def make_policy(parts: dict[str, Any], seed: int, seat: str) -> RegularAiPolicy:
    return RegularAiPolicy(seed=seed, seat=seat, **parts)


def _safe_load_hu_action_value_model(path: Path) -> Any | None:
    try:
        return load_hu_action_value_model(path)
    except Exception:
        return None


def _safe_load_hu_turn3_gate_model(path: Path) -> Any | None:
    try:
        return load_hu_turn3_gate_model(path)
    except Exception:
        return None


def summarize(scores: list[float]) -> dict[str, float]:
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


def evaluate_matchup(args: argparse.Namespace) -> dict[str, Any]:
    if args.games <= 0:
        raise SystemExit("--games must be positive")
    if args.prediction_threads > 0:
        os.environ.setdefault("OMP_NUM_THREADS", str(args.prediction_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.prediction_threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.prediction_threads))

    parts_a = load_policy_parts(args, "a")
    parts_b = load_policy_parts(args, "b")
    paired_scores: list[float] = []
    wins = losses = ties = 0
    trace_count = 0
    started_at = time.time()

    if args.trace_output:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    trace_context = (
        args.trace_output.open("w", encoding="utf-8")
        if args.trace_output is not None
        else contextlib.nullcontext(None)
    )
    with _prediction_thread_context(args.prediction_threads), trace_context as trace_handle:
        for index in range(args.games):
            hand_seed = args.seed + index
            hand_ab = trace_hand(
                seed=hand_seed,
                profile_p0=args.name_a,
                profile_p1=args.name_b,
                policy_p0=make_policy(parts_a, hand_seed * 4, "first"),
                policy_p1=make_policy(parts_b, hand_seed * 4 + 1, "second"),
            )
            hand_ba = trace_hand(
                seed=hand_seed,
                profile_p0=args.name_b,
                profile_p1=args.name_a,
                policy_p0=make_policy(parts_b, hand_seed * 4 + 2, "first"),
                policy_p1=make_policy(parts_a, hand_seed * 4 + 3, "second"),
            )
            paired_score = (float(hand_ab["score_p0"]) - float(hand_ba["score_p0"])) / 2.0
            paired_scores.append(paired_score)
            if paired_score > 0:
                wins += 1
            elif paired_score < 0:
                losses += 1
            else:
                ties += 1
            if trace_handle is not None and (args.trace_limit <= 0 or trace_count < args.trace_limit):
                hand_ab["paired_score_for_a"] = paired_score
                trace_handle.write(json.dumps(hand_ab, ensure_ascii=False, separators=(",", ":")) + "\n")
                trace_count += 1
            if trace_handle is not None and (args.trace_limit <= 0 or trace_count < args.trace_limit):
                hand_ba["paired_score_for_a"] = paired_score
                trace_handle.write(json.dumps(hand_ba, ensure_ascii=False, separators=(",", ":")) + "\n")
                trace_count += 1
            if args.progress_every > 0 and (index + 1) % args.progress_every == 0:
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "paired_seeds": index + 1,
                            "hands": (index + 1) * 2,
                            **summarize(paired_scores),
                            "elapsed_seconds": time.time() - started_at,
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )

    summary = {
        "name_a": args.name_a,
        "name_b": args.name_b,
        "paired_seeds": args.games,
        "hands": args.games * 2,
        "seed": args.seed,
        **summarize(paired_scores),
        "paired_seed_wins": wins,
        "paired_seed_losses": losses,
        "paired_seed_ties": ties,
        "elapsed_seconds": time.time() - started_at,
        "trace_output": str(args.trace_output) if args.trace_output else None,
        "trace_hands_written": trace_count,
        "models_a": {
            "opening": str(args.opening_a),
            "turn1": str(args.turn1_a),
            "turn2": str(args.turn2_a),
            "turn3": str(args.turn3_a),
            "hu_turn3": str(args.hu_turn3_a) if args.hu_turn3_a else None,
            "hu_turn3_reference": str(args.hu_turn3_reference_a)
            if args.hu_turn3_reference_a
            else None,
            "hu_turn3_support": str(args.hu_turn3_support_a)
            if args.hu_turn3_support_a
            else None,
            "hu_turn3_gate": str(args.hu_turn3_gate_a) if args.hu_turn3_gate_a else None,
            "hu_turn3_min_margin": args.hu_turn3_min_margin_a,
            "hu_turn3_reference_min_margin": args.hu_turn3_reference_min_margin_a,
            "hu_turn3_min_support_margin": args.hu_turn3_min_support_margin_a,
            "hu_turn3_min_gate_probability": args.hu_turn3_min_gate_probability_a,
            "hu_turn3_max_self_regret": args.hu_turn3_max_self_regret_a,
            "hu_turn3_stage7_enabled": not args.disable_hu_turn3_stage7_a,
        },
        "models_b": {
            "opening": str(args.opening_b),
            "turn1": str(args.turn1_b),
            "turn2": str(args.turn2_b),
            "turn3": str(args.turn3_b),
            "hu_turn3": str(args.hu_turn3_b) if args.hu_turn3_b else None,
            "hu_turn3_reference": str(args.hu_turn3_reference_b)
            if args.hu_turn3_reference_b
            else None,
            "hu_turn3_support": str(args.hu_turn3_support_b)
            if args.hu_turn3_support_b
            else None,
            "hu_turn3_gate": str(args.hu_turn3_gate_b) if args.hu_turn3_gate_b else None,
            "hu_turn3_min_margin": args.hu_turn3_min_margin_b,
            "hu_turn3_reference_min_margin": args.hu_turn3_reference_min_margin_b,
            "hu_turn3_min_support_margin": args.hu_turn3_min_support_margin_b,
            "hu_turn3_min_gate_probability": args.hu_turn3_min_gate_probability_b,
            "hu_turn3_max_self_regret": args.hu_turn3_max_self_regret_b,
            "hu_turn3_stage7_enabled": not args.disable_hu_turn3_stage7_b,
        },
    }
    return summary


def main() -> None:
    args = parse_args()
    summary = evaluate_matchup(args)
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
