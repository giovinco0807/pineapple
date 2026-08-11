"""Evaluate HU T0 selective override with same-seed whole-game counterfactuals."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable, Sequence

from .ai_profiles import ModelPaths, build_policy, load_model_bundle, required_profiles
from .evaluate_matchups import trace_hand
from .hu_turn0_candidate import load_turn0_candidate_model
from .hu_turn0_safe_selector import load_hu_turn0_safe_selector_model
from .policy import RegularAiPolicy


def _mean_ci95(values: Sequence[float]) -> tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    center = mean(values)
    if len(values) < 2:
        return center, 0.0, center, center
    stderr = stdev(values) / math.sqrt(len(values))
    return center, stderr, center - 1.96 * stderr, center + 1.96 * stderr


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _hero_score(trace: dict[str, Any], hero_player: int) -> float:
    """Return the zero-sum terminal score from the requested player's view."""
    score_p0 = float(trace["score_p0"])
    if hero_player == 0:
        return score_p0
    if hero_player == 1:
        return -score_p0
    raise ValueError(f"hero_player must be 0 or 1, got {hero_player}")


def summarize_events(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    deltas = [float(event["realized_delta"]) for event in events]
    fired = [event for event in events if bool(event["override_fired"])]
    fired_deltas = [float(event["realized_delta"]) for event in fired]
    seed_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("seed") is not None:
            seed_groups[int(event["seed"])].append(event)
    paired_seed_deltas = [
        mean(float(event["realized_delta"]) for event in seed_rows)
        for _seed, seed_rows in sorted(seed_groups.items())
    ]
    aggregate_samples = paired_seed_deltas if len(seed_groups) > 0 else deltas
    avg_delta, stderr, ci_low, ci_high = _mean_ci95(aggregate_samples)
    per_fire, per_fire_stderr, per_fire_low, per_fire_high = _mean_ci95(fired_deltas)
    fired_losses = [max(0.0, -delta) for delta in fired_deltas]
    non_fired_nonzero = sum(
        abs(float(event["realized_delta"])) > 1e-12
        for event in events
        if not bool(event["override_fired"])
    )
    by_seat: dict[str, Any] = {}
    for seat in ("first", "second"):
        seat_rows = [event for event in events if event.get("seat") == seat]
        seat_deltas = [float(event["realized_delta"]) for event in seat_rows]
        seat_fired = [event for event in seat_rows if bool(event["override_fired"])]
        seat_fired_deltas = [float(event["realized_delta"]) for event in seat_fired]
        seat_avg, seat_se, seat_low, seat_high = _mean_ci95(seat_deltas)
        seat_per_fire, _seat_pf_se, seat_pf_low, seat_pf_high = _mean_ci95(
            seat_fired_deltas
        )
        by_seat[seat] = {
            "events": len(seat_rows),
            "avg_delta_per_hand": seat_avg,
            "std_error": seat_se,
            "ci95_low": seat_low,
            "ci95_high": seat_high,
            "fires": len(seat_fired),
            "fire_rate": len(seat_fired) / len(seat_rows) if seat_rows else 0.0,
            "avg_delta_per_fire": seat_per_fire,
            "per_fire_ci95_low": seat_pf_low,
            "per_fire_ci95_high": seat_pf_high,
        }
    return {
        "events": len(events),
        "paired_seeds": len(seed_groups),
        "avg_delta_per_hand": avg_delta,
        "std_error": stderr,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "fires": len(fired),
        "fire_rate": len(fired) / len(events) if events else 0.0,
        "avg_delta_per_fire": per_fire,
        "per_fire_std_error": per_fire_stderr,
        "per_fire_ci95_low": per_fire_low,
        "per_fire_ci95_high": per_fire_high,
        "realized_fired_contribution_per_hand": (
            sum(fired_deltas) / len(events) if events else 0.0
        ),
        "positive_fire_count": sum(delta > 0.0 for delta in fired_deltas),
        "negative_fire_count": sum(delta < 0.0 for delta in fired_deltas),
        "zero_fire_count": sum(delta == 0.0 for delta in fired_deltas),
        "false_positive_rate": (
            sum(delta <= 0.0 for delta in fired_deltas) / len(fired_deltas)
            if fired_deltas
            else 0.0
        ),
        "p90_fire_loss": _quantile(fired_losses, 0.90),
        "p95_fire_loss": _quantile(fired_losses, 0.95),
        "p99_fire_loss": _quantile(fired_losses, 0.99),
        "max_fire_loss": max(fired_losses, default=0.0),
        "paired_seed_delta_p05": _quantile(paired_seed_deltas, 0.05),
        "paired_seed_delta_p50": _quantile(paired_seed_deltas, 0.50),
        "paired_seed_delta_p95": _quantile(paired_seed_deltas, 0.95),
        "non_fired_nonzero_count": non_fired_nonzero,
        "no_override_reason_counts": dict(
            sorted(
                Counter(
                    str(event.get("no_override_reason") or "")
                    for event in events
                    if not bool(event["override_fired"])
                ).items()
            )
        ),
        "by_seat": by_seat,
    }


def _make_policy_factory(
    *,
    profile: str,
    bundle: Any,
    candidate_model: Any,
    candidate_topk: int,
    margin_by_seat: dict[str, float],
    allowed_seats: tuple[str, ...] | None,
    safe_selector_model: Any | None = None,
    safe_selector_threshold_by_seat: dict[str, float] | None = None,
    opening_lookahead_samples: int,
) -> Callable[[int, str, bool], RegularAiPolicy]:
    def factory(seed: int, seat: str, candidate: bool) -> RegularAiPolicy:
        policy = build_policy(
            profile,
            bundle,
            seed=seed,
            seat=seat,
            opening_lookahead_samples=opening_lookahead_samples,
        )
        if candidate:
            policy.hu_turn0_model = candidate_model
            policy.hu_turn0_candidate_topk = int(candidate_topk)
            policy.hu_turn0_min_margin_by_seat = dict(margin_by_seat)
            policy.hu_turn0_allowed_seats = allowed_seats
            policy.hu_turn0_safe_selector_model = safe_selector_model
            policy.hu_turn0_safe_selector_enabled = safe_selector_model is not None
            policy.hu_turn0_safe_selector_threshold_by_seat = (
                dict(safe_selector_threshold_by_seat)
                if safe_selector_threshold_by_seat is not None
                else None
            )
            policy.hu_turn0_decision_log = []
        return policy

    return factory


def evaluate_counterfactual(
    *,
    games: int,
    seed: int,
    seed_stride: int,
    policy_factory: Callable[[int, str, bool], RegularAiPolicy],
    profile: str,
    config_id: str | None = None,
    progress_every: int = 0,
    events_output: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if games <= 0:
        raise ValueError("games must be positive")
    if seed_stride <= 0:
        raise ValueError("seed_stride must be positive")
    events: list[dict[str, Any]] = []
    started_at = time.time()
    for index in range(games):
        hand_seed = seed + index * seed_stride
        for seat, hero_player in (("first", 0), ("second", 1)):
            hero_policy_seed = hand_seed * 8 + (0 if seat == "first" else 2)
            opponent_policy_seed = hand_seed * 8 + (1 if seat == "first" else 3)
            if seat == "first":
                candidate_hero = policy_factory(hero_policy_seed, "first", True)
                candidate_opponent = policy_factory(opponent_policy_seed, "second", False)
                baseline_hero = policy_factory(hero_policy_seed, "first", False)
                baseline_opponent = policy_factory(opponent_policy_seed, "second", False)
                candidate_trace = trace_hand(
                    seed=hand_seed,
                    profile_p0=f"{profile}_t0_candidate",
                    profile_p1=profile,
                    policy_p0=candidate_hero,
                    policy_p1=candidate_opponent,
                )
                baseline_trace = trace_hand(
                    seed=hand_seed,
                    profile_p0=profile,
                    profile_p1=profile,
                    policy_p0=baseline_hero,
                    policy_p1=baseline_opponent,
                )
            else:
                candidate_opponent = policy_factory(opponent_policy_seed, "first", False)
                candidate_hero = policy_factory(hero_policy_seed, "second", True)
                baseline_opponent = policy_factory(opponent_policy_seed, "first", False)
                baseline_hero = policy_factory(hero_policy_seed, "second", False)
                candidate_trace = trace_hand(
                    seed=hand_seed,
                    profile_p0=profile,
                    profile_p1=f"{profile}_t0_candidate",
                    policy_p0=candidate_opponent,
                    policy_p1=candidate_hero,
                )
                baseline_trace = trace_hand(
                    seed=hand_seed,
                    profile_p0=profile,
                    profile_p1=profile,
                    policy_p0=baseline_opponent,
                    policy_p1=baseline_hero,
                )

            decision_log = candidate_hero.hu_turn0_decision_log or []
            if len(decision_log) != 1:
                raise RuntimeError(
                    f"expected one HU T0 decision for seed={hand_seed} seat={seat}, got {len(decision_log)}"
                )
            decision = decision_log[0]
            candidate_score = _hero_score(candidate_trace, hero_player)
            baseline_score = _hero_score(baseline_trace, hero_player)
            events.append(
                {
                    "event_id": f"{hand_seed}:{seat}",
                    "config_id": config_id,
                    "seed": hand_seed,
                    "seat": seat,
                    "candidate_score": candidate_score,
                    "baseline_score": baseline_score,
                    "realized_delta": candidate_score - baseline_score,
                    "override_fired": bool(decision["override_fired"]),
                    "no_override_reason": decision.get("no_override_reason"),
                    "hu_turn0_predicted_margin": decision.get("hu_turn0_predicted_margin"),
                    "hu_turn0_safe_selector_score": decision.get(
                        "hu_turn0_safe_selector_score"
                    ),
                    "hu_turn0_safe_selector_threshold": decision.get(
                        "hu_turn0_safe_selector_threshold"
                    ),
                    "fallback_action_index": decision.get("fallback_action_index"),
                    "candidate_action_index": decision.get("candidate_action_index"),
                    "final_action_index": decision.get("final_action_index"),
                    "decision": decision,
                }
            )
        if progress_every > 0 and (index + 1) % progress_every == 0:
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "paired_seeds": index + 1,
                        "elapsed_seconds": time.time() - started_at,
                        **summarize_events(events),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                flush=True,
            )

    if events_output is not None:
        events_output.parent.mkdir(parents=True, exist_ok=True)
        with events_output.open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "schema": "hu_turn0_counterfactual_eval_v1",
        "config_id": config_id,
        "profile": profile,
        "requested_paired_seeds": games,
        "simulated_world_hands": games * 4,
        "seed": seed,
        "seed_stride": seed_stride,
        "elapsed_seconds": time.time() - started_at,
        **summarize_events(events),
    }
    return summary, events


def _parse_allowed_seats(raw: str) -> tuple[str, ...] | None:
    seats = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not seats:
        return None
    invalid = [seat for seat in seats if seat not in {"first", "second"}]
    if invalid:
        raise ValueError(f"invalid allowed seats: {invalid}")
    return seats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--config-id")
    parser.add_argument("--candidate-topk", type=int, default=60)
    parser.add_argument("--min-margin-first", type=float, required=True)
    parser.add_argument("--min-margin-second", type=float, required=True)
    parser.add_argument("--allowed-seats", default="first,second")
    parser.add_argument("--safe-selector-model", type=Path)
    parser.add_argument("--safe-selector-threshold-first", type=float, default=0.0)
    parser.add_argument("--safe-selector-threshold-second", type=float, default=0.0)
    parser.add_argument("--profile", default="stage18_p1")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026106001)
    parser.add_argument("--seed-stride", type=int, default=1000003)
    parser.add_argument("--opening-lookahead-samples", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--events-output", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.candidate_topk <= 0:
        raise SystemExit("--candidate-topk must be positive")
    try:
        candidate_model = load_turn0_candidate_model(args.candidate_model)
        allowed_seats = _parse_allowed_seats(args.allowed_seats)
        safe_selector_model = (
            load_hu_turn0_safe_selector_model(args.safe_selector_model)
            if args.safe_selector_model is not None
            else None
        )
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    profiles = required_profiles(args.profile, args.profile)
    bundle = load_model_bundle(ModelPaths(), profiles)
    policy_factory = _make_policy_factory(
        profile=args.profile,
        bundle=bundle,
        candidate_model=candidate_model,
        candidate_topk=args.candidate_topk,
        margin_by_seat={
            "first": float(args.min_margin_first),
            "second": float(args.min_margin_second),
        },
        allowed_seats=allowed_seats,
        safe_selector_model=safe_selector_model,
        safe_selector_threshold_by_seat={
            "first": float(args.safe_selector_threshold_first),
            "second": float(args.safe_selector_threshold_second),
        },
        opening_lookahead_samples=args.opening_lookahead_samples,
    )
    summary, _events = evaluate_counterfactual(
        games=args.games,
        seed=args.seed,
        seed_stride=args.seed_stride,
        policy_factory=policy_factory,
        profile=args.profile,
        config_id=args.config_id,
        progress_every=args.progress_every,
        events_output=args.events_output,
    )
    summary.update(
        {
            "candidate_model": str(args.candidate_model),
            "candidate_topk": args.candidate_topk,
            "min_margin_by_seat": {
                "first": args.min_margin_first,
                "second": args.min_margin_second,
            },
            "allowed_seats": list(allowed_seats or ()),
            "safe_selector_model": (
                str(args.safe_selector_model) if args.safe_selector_model is not None else None
            ),
            "safe_selector_threshold_by_seat": {
                "first": args.safe_selector_threshold_first,
                "second": args.safe_selector_threshold_second,
            },
            "events_output": str(args.events_output) if args.events_output else None,
            "primary_metric": "realized_fired_whole_game_counterfactual_delta",
        }
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
