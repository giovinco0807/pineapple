"""Fresh paired population evaluation for the opt-in M4 T1 policy.

This evaluator compares the candidate and an explicit frozen baseline profile
on the same shuffled hand, physical seat, opponent profile, and policy seeds.
It therefore measures a real played-hand counterfactual.  Teacher estimates
and legacy validation artifacts are deliberately outside this module.

The core API accepts policy factories so correctness can be tested without
loading model artifacts.  The CLI is intentionally narrower: every opponent
is named explicitly, the M4 action-value/safety artifacts and threshold are
required, and no ``current`` profile is consulted or changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .ai_profiles import (
    DEFAULT_HU_TURN1_STAGE18_P1_MODEL,
    DEFAULT_HU_TURN1_STAGE18_P1_SAFE_SELECTOR_MODEL,
    ModelPaths,
    build_policy,
    load_model_bundle,
    required_profiles,
)
from .evaluate_matchups import trace_hand
from .hu_m43_joint_model_loader import load_hu_m43_joint_action_model
from .hu_m43_joint_model_v5 import HU_M43_V5_MODEL_SCHEMA
from .hu_m43_joint_model_v6 import HU_M43_V6_MODEL_SCHEMA
from .hu_m43_attempt08_distilled_model import (
    HU_M43_ATTEMPT08_DISTILLED_MODEL_SCHEMA,
    is_bound_attempt08_distilled_model,
)
from .hu_m43_attempt08_distilled_runtime import (
    ATTEMPT08_DISTILLED_RUNTIME_SEMANTIC_CLOSURE_SCHEMA,
    validate_distilled_runtime_dependencies as validate_attempt08_runtime_dependencies,
    validate_frozen_execution_modules as validate_attempt08_frozen_modules,
)
from .hu_m43_attempt10_distilled_model import (
    HU_M43_ATTEMPT10_DISTILLED_MODEL_SCHEMA,
    is_bound_attempt10_distilled_model,
)
from .hu_m43_attempt10_distilled_runtime import (
    ATTEMPT10_DISTILLED_RUNTIME_SEMANTIC_CLOSURE_SCHEMA,
    validate_distilled_runtime_dependencies as validate_attempt10_runtime_dependencies,
    validate_frozen_execution_modules as validate_attempt10_frozen_modules,
)
from .hu_m43_attempt11_distilled_model import (
    HU_M43_ATTEMPT11_DISTILLED_MODEL_SCHEMA,
    is_bound_attempt11_distilled_model,
)
from .hu_m43_attempt11_distilled_runtime import (
    ATTEMPT11_DISTILLED_RUNTIME_SEMANTIC_CLOSURE_SCHEMA,
    validate_distilled_runtime_dependencies as validate_attempt11_runtime_dependencies,
    validate_frozen_execution_modules as validate_attempt11_frozen_modules,
)
from .hu_m43_attempt12_distilled_model import (
    HU_M43_ATTEMPT12_DISTILLED_MODEL_SCHEMA,
    is_bound_attempt12_distilled_model,
)
from .hu_m43_attempt12_distilled_runtime import (
    ATTEMPT12_DISTILLED_RUNTIME_SEMANTIC_CLOSURE_SCHEMA,
    validate_distilled_runtime_dependencies as validate_attempt12_runtime_dependencies,
    validate_frozen_execution_modules as validate_attempt12_frozen_modules,
)
from .hu_m43_attempt13_distilled_model import (
    HU_M43_ATTEMPT13_DISTILLED_MODEL_SCHEMA,
    is_bound_attempt13_distilled_model,
)
from .hu_m43_attempt13_distilled_runtime import (
    ATTEMPT13_BOUND_EXECUTION_MODULES,
    ATTEMPT13_DISTILLED_RUNTIME_SEMANTIC_CLOSURE_SCHEMA,
    validate_distilled_runtime_dependencies as validate_attempt13_runtime_dependencies,
    validate_frozen_execution_modules as validate_attempt13_frozen_modules,
)
from .hu_m4_t1_policy import (
    HU_M4_T1_DECISION_SCHEMA,
    HuM4T1SelectiveOverridePolicy,
)


M4_POPULATION_EVALUATION_SCHEMA = "hu_m4_t1_population_evaluation_v1"
M4_HAND_RECORD_SCHEMA = "hu_m4_t1_population_hand_v1"
M4_OPPONENT_PROFILES = (
    "stage19_p0",
    "stage9f_p2",
    "stage7_m5_r10",
    "random_exact_final",
)

PolicyFactory = Callable[..., Any]
TraceFunction = Callable[..., dict[str, Any]]

_BOUND_POPULATION_MODEL_SCHEMAS = frozenset(
    {
        HU_M43_V5_MODEL_SCHEMA,
        HU_M43_V6_MODEL_SCHEMA,
        HU_M43_ATTEMPT08_DISTILLED_MODEL_SCHEMA,
        HU_M43_ATTEMPT10_DISTILLED_MODEL_SCHEMA,
        HU_M43_ATTEMPT11_DISTILLED_MODEL_SCHEMA,
        HU_M43_ATTEMPT12_DISTILLED_MODEL_SCHEMA,
        HU_M43_ATTEMPT13_DISTILLED_MODEL_SCHEMA,
    }
)


def requires_bound_population_runtime(model: object) -> bool:
    """Return whether promotion evaluation must use its complete bindings."""

    return getattr(model, "schema", None) in _BOUND_POPULATION_MODEL_SCHEMAS


def _mean_ci95(values: Sequence[float]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    if not numbers:
        return {
            "n": 0,
            "mean": None,
            "std_error": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    mean = sum(numbers) / len(numbers)
    if len(numbers) == 1:
        std_error = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in numbers) / (
            len(numbers) - 1
        )
        std_error = math.sqrt(variance / len(numbers))
    return {
        "n": len(numbers),
        "mean": mean,
        "std_error": std_error,
        "ci95_low": mean - 1.96 * std_error,
        "ci95_high": mean + 1.96 * std_error,
    }


def _cluster_mean_ci95(
    records: Sequence[Mapping[str, Any]], key: str
) -> dict[str, Any]:
    """Estimate an EV CI with the hand seed as the independence unit."""

    by_seed: dict[int, list[float]] = {}
    for row in records:
        by_seed.setdefault(int(row["seed"]), []).append(float(row[key]))
    result = _mean_ci95(
        [sum(values) / len(values) for _seed, values in sorted(by_seed.items())]
    )
    result["ci_independence_unit"] = "hand_seed_cluster"
    return result


def _cluster_ratio_ci95(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Cluster-robust ratio CI for total realized gain / total overrides.

    Opponent outcomes sharing a physical hand seed are correlated.  The ratio
    influence function therefore aggregates gain and fire count inside each
    seed before estimating its standard error.  Seeds with no fire remain in
    the variance calculation rather than silently disappearing.
    """

    by_seed: dict[int, tuple[float, int]] = {}
    for row in records:
        hand_seed = int(row["seed"])
        gain, fires = by_seed.get(hand_seed, (0.0, 0))
        if row.get("override_fired") is True:
            gain += float(row["delta"])
            fires += 1
        by_seed[hand_seed] = (gain, fires)
    clusters = len(by_seed)
    total_gain = sum(value[0] for value in by_seed.values())
    total_fires = sum(value[1] for value in by_seed.values())
    if clusters == 0 or total_fires == 0:
        return {
            "n": total_fires,
            "clusters": clusters,
            "mean": None,
            "std_error": None,
            "ci95_low": None,
            "ci95_high": None,
            "ci_independence_unit": "hand_seed_cluster_ratio_influence",
        }
    mean = total_gain / total_fires
    mean_fires_per_cluster = total_fires / clusters
    influences = [
        (gain - mean * fires) / mean_fires_per_cluster
        for gain, fires in by_seed.values()
    ]
    if clusters == 1:
        std_error = 0.0
    else:
        influence_mean = sum(influences) / clusters
        variance = sum(
            (value - influence_mean) ** 2 for value in influences
        ) / (clusters - 1)
        std_error = math.sqrt(variance / clusters)
    return {
        "n": total_fires,
        "clusters": clusters,
        "mean": mean,
        "std_error": std_error,
        "ci95_low": mean - 1.96 * std_error,
        "ci95_high": mean + 1.96 * std_error,
        "ci_independence_unit": "hand_seed_cluster_ratio_influence",
    }


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _loss_tail(deltas: Sequence[float]) -> dict[str, Any]:
    """Summarize loss magnitude, with winning overrides represented by zero."""

    losses = [max(0.0, -float(delta)) for delta in deltas]
    return {
        "n": len(losses),
        "loss_count": sum(loss > 0.0 for loss in losses),
        "p95": _percentile(losses, 0.95),
        "p99": _percentile(losses, 0.99),
        "max": max(losses) if losses else None,
    }


def gameplay_digest(hand: Mapping[str, Any]) -> str:
    """Hash the complete gameplay trajectory while excluding profile labels.

    ``trace_hand`` stores profile metadata both at the root and on each turn.
    Those labels differ by construction between candidate and baseline traces,
    but do not describe gameplay.  All deals, placements, discards, boards,
    terminal scores, and board-score details remain in the digest.
    """

    turns = []
    for raw_turn in hand.get("turns", ()):
        turn = dict(raw_turn)
        turn.pop("profile", None)
        turns.append(turn)
    payload = {
        "seed": hand.get("seed"),
        "turns": turns,
        "final": hand.get("final"),
        "score_p0": hand.get("score_p0"),
        "board_scores": hand.get("board_scores"),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _candidate_decision_status(
    decision_log: Sequence[Mapping[str, Any]], *, expected_seat: str
) -> dict[str, Any]:
    rows = [
        row
        for row in decision_log
        if row.get("schema") == HU_M4_T1_DECISION_SCHEMA
        and row.get("street") == "T1"
    ]
    if len(rows) != 1:
        return {
            "valid": False,
            "override_fired": None,
            "reason": "missing_t1_decision" if not rows else "multiple_t1_decisions",
            "t1_decision_count": len(rows),
        }
    row = rows[0]
    if row.get("seat") != expected_seat:
        return {
            "valid": False,
            "override_fired": None,
            "reason": "seat_mismatch",
            "t1_decision_count": 1,
        }
    fired = row.get("override_fired")
    if not isinstance(fired, bool):
        return {
            "valid": False,
            "override_fired": None,
            "reason": "invalid_override_fired",
            "t1_decision_count": 1,
        }
    return {
        "valid": True,
        "override_fired": fired,
        "reason": None,
        "t1_decision_count": 1,
        "nonfire_reason": row.get("nonfire_reason"),
        "safety_probability": row.get("safety_probability"),
        "predicted_delta": row.get("predicted_delta"),
        "runtime_binding_verified": row.get("runtime_binding_verified"),
    }


def _make_policy(
    factory: PolicyFactory,
    *,
    policy_seed: int,
    seat: str,
    decision_log: list[dict[str, Any]] | None,
) -> Any:
    return factory(
        policy_seed=policy_seed,
        seat=seat,
        decision_log=decision_log,
    )


def _trace_candidate_and_baseline(
    *,
    trace_fn: TraceFunction,
    hand_seed: int,
    seat: str,
    opponent_name: str,
    candidate_factory: PolicyFactory,
    baseline_factory: PolicyFactory,
    opponent_factory: PolicyFactory,
) -> dict[str, Any]:
    if seat not in {"first", "second"}:
        raise ValueError(f"invalid seat: {seat!r}")
    first_policy_seed = hand_seed * 4
    second_policy_seed = hand_seed * 4 + 1
    hero_policy_seed = first_policy_seed if seat == "first" else second_policy_seed
    opponent_policy_seed = second_policy_seed if seat == "first" else first_policy_seed

    candidate_log: list[dict[str, Any]] = []
    candidate = _make_policy(
        candidate_factory,
        policy_seed=hero_policy_seed,
        seat=seat,
        decision_log=candidate_log,
    )
    candidate_opponent = _make_policy(
        opponent_factory,
        policy_seed=opponent_policy_seed,
        seat="second" if seat == "first" else "first",
        decision_log=None,
    )
    baseline = _make_policy(
        baseline_factory,
        policy_seed=hero_policy_seed,
        seat=seat,
        decision_log=None,
    )
    baseline_opponent = _make_policy(
        opponent_factory,
        policy_seed=opponent_policy_seed,
        seat="second" if seat == "first" else "first",
        decision_log=None,
    )

    if seat == "first":
        candidate_hand = trace_fn(
            seed=hand_seed,
            profile_p0="m4_candidate",
            profile_p1=opponent_name,
            policy_p0=candidate,
            policy_p1=candidate_opponent,
        )
        baseline_hand = trace_fn(
            seed=hand_seed,
            profile_p0="stage18_p1_baseline",
            profile_p1=opponent_name,
            policy_p0=baseline,
            policy_p1=baseline_opponent,
        )
        candidate_score = float(candidate_hand["score_p0"])
        baseline_score = float(baseline_hand["score_p0"])
    else:
        candidate_hand = trace_fn(
            seed=hand_seed,
            profile_p0=opponent_name,
            profile_p1="m4_candidate",
            policy_p0=candidate_opponent,
            policy_p1=candidate,
        )
        baseline_hand = trace_fn(
            seed=hand_seed,
            profile_p0=opponent_name,
            profile_p1="stage18_p1_baseline",
            policy_p0=baseline_opponent,
            policy_p1=baseline,
        )
        candidate_score = -float(candidate_hand["score_p0"])
        baseline_score = -float(baseline_hand["score_p0"])

    if not math.isfinite(candidate_score) or not math.isfinite(baseline_score):
        raise ValueError("trace returned a non-finite terminal score")
    candidate_digest = gameplay_digest(candidate_hand)
    baseline_digest = gameplay_digest(baseline_hand)
    decision = _candidate_decision_status(candidate_log, expected_seat=seat)
    delta = candidate_score - baseline_score
    fired = decision["override_fired"]
    cancellation: bool | None = None
    if decision["valid"] and fired is False:
        cancellation = candidate_digest == baseline_digest and delta == 0.0

    return {
        "schema": M4_HAND_RECORD_SCHEMA,
        "opponent": opponent_name,
        "seed": hand_seed,
        "seat": seat,
        "hero_policy_seed": hero_policy_seed,
        "opponent_policy_seed": opponent_policy_seed,
        "candidate_score": candidate_score,
        "baseline_score": baseline_score,
        "delta": delta,
        "override_log_valid": decision["valid"],
        "override_log_reason": decision["reason"],
        "override_fired": fired,
        "t1_decision_count": decision["t1_decision_count"],
        "nonfire_reason": decision.get("nonfire_reason"),
        "safety_probability": decision.get("safety_probability"),
        "predicted_delta": decision.get("predicted_delta"),
        "runtime_binding_verified": decision.get("runtime_binding_verified"),
        "candidate_gameplay_digest": candidate_digest,
        "baseline_gameplay_digest": baseline_digest,
        "nonfire_cancellation_valid": cancellation,
        "counterfactual_basis": "same_seed_physical_seat_opponent_policy_seeds_v1",
    }


def _seat_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid_records = [row for row in records if _promotion_record_valid(row)]
    deltas = [float(row["delta"]) for row in valid_records]
    fired = [row for row in valid_records if row.get("override_fired") is True]
    fired_deltas = [float(row["delta"]) for row in fired]
    nonfired = [row for row in records if row.get("override_fired") is False]
    invalid = [row for row in records if not row.get("override_log_valid")]
    valid_cancellations = sum(
        row.get("nonfire_cancellation_valid") is True for row in nonfired
    )
    mismatches = sum(
        row.get("nonfire_cancellation_valid") is False for row in nonfired
    )
    nonzero_nonfires = sum(float(row["delta"]) != 0.0 for row in nonfired)
    # Keep the realized definition identical to v4 calibration: an override is
    # a false positive unless it produces strictly positive terminal match EV.
    false_positives = sum(float(row["delta"]) <= 0.0 for row in fired)
    return {
        "hands": len(records),
        "valid_hands": len(valid_records),
        "candidate_ev_per_hand": _cluster_mean_ci95(
            valid_records, "candidate_score"
        ),
        "baseline_ev_per_hand": _cluster_mean_ci95(
            valid_records, "baseline_score"
        ),
        "delta_ev_per_hand": _cluster_mean_ci95(valid_records, "delta"),
        "overrides": len(fired),
        "override_rate": len(fired) / len(records) if records else 0.0,
        "realized_gain_per_override": _cluster_ratio_ci95(valid_records),
        "false_positive_overrides": false_positives,
        "false_positive_definition": "realized_override_delta_le_zero",
        "false_positive_override_rate": (
            false_positives / len(fired) if fired else None
        ),
        "override_loss_tail": _loss_tail(fired_deltas),
        "nonfires": len(nonfired),
        "nonfire_cancellation_valid": valid_cancellations,
        "nonfire_cancellation_mismatches": mismatches,
        "nonfire_nonzero_deltas": nonzero_nonfires,
        "nonfire_cancellation_unknown": len(invalid),
        "invalid_counterfactuals": len(invalid),
    }


def _summarize_opponent(
    opponent: str, records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    first = [row for row in records if row["seat"] == "first"]
    second = [row for row in records if row["seat"] == "second"]
    paired_candidate, paired_baseline, paired_delta = _paired_values(
        [row for row in records if _promotion_record_valid(row)],
        skip_incomplete=True,
    )
    return {
        "opponent": opponent,
        "paired_seeds": len(paired_delta),
        "hands_per_policy": len(records),
        "seat_swap": {
            "candidate_ev_per_hand": _mean_ci95(paired_candidate),
            "baseline_ev_per_hand": _mean_ci95(paired_baseline),
            "delta_ev_per_hand": _mean_ci95(paired_delta),
            "delta_loss_tail": _loss_tail(paired_delta),
        },
        "by_seat": {
            "first": _seat_summary(first),
            "second": _seat_summary(second),
        },
        "all_seats": _seat_summary(records),
    }


def _paired_values(
    records: Sequence[Mapping[str, Any]],
    *,
    skip_incomplete: bool = False,
) -> tuple[list[float], list[float], list[float]]:
    by_seed: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
    for row in records:
        pair_key = (str(row["opponent"]), int(row["seed"]))
        by_seed.setdefault(pair_key, {})[str(row["seat"])] = row
    paired_candidate: list[float] = []
    paired_baseline: list[float] = []
    paired_delta: list[float] = []
    for (opponent, hand_seed), seat_rows in sorted(by_seed.items()):
        if set(seat_rows) != {"first", "second"}:
            if skip_incomplete:
                continue
            raise ValueError(f"incomplete seat swap for {opponent} seed={hand_seed}")
        candidate = (
            float(seat_rows["first"]["candidate_score"])
            + float(seat_rows["second"]["candidate_score"])
        ) / 2.0
        baseline = (
            float(seat_rows["first"]["baseline_score"])
            + float(seat_rows["second"]["baseline_score"])
        ) / 2.0
        paired_candidate.append(candidate)
        paired_baseline.append(baseline)
        paired_delta.append(candidate - baseline)
    return paired_candidate, paired_baseline, paired_delta


def _promotion_record_valid(row: Mapping[str, Any]) -> bool:
    if row.get("override_log_valid") is not True:
        return False
    if row.get("override_fired") is True:
        return True
    return (
        row.get("override_fired") is False
        and row.get("nonfire_cancellation_valid") is True
        and float(row.get("delta", float("nan"))) == 0.0
    )


def _clustered_population_pairs(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[float], list[float], list[float]]:
    """Average opponents within each seed before estimating population CI."""

    paired_by_seed: dict[int, list[tuple[float, float]]] = {}
    by_pair: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
    for row in records:
        by_pair.setdefault(
            (str(row["opponent"]), int(row["seed"])), {}
        )[str(row["seat"])] = row
    for (_opponent, hand_seed), seat_rows in by_pair.items():
        if set(seat_rows) != {"first", "second"}:
            continue
        candidate = (
            float(seat_rows["first"]["candidate_score"])
            + float(seat_rows["second"]["candidate_score"])
        ) / 2.0
        baseline = (
            float(seat_rows["first"]["baseline_score"])
            + float(seat_rows["second"]["baseline_score"])
        ) / 2.0
        paired_by_seed.setdefault(hand_seed, []).append((candidate, baseline))
    candidates: list[float] = []
    baselines: list[float] = []
    deltas: list[float] = []
    for hand_seed in sorted(paired_by_seed):
        rows = paired_by_seed[hand_seed]
        candidate = sum(row[0] for row in rows) / len(rows)
        baseline = sum(row[1] for row in rows) / len(rows)
        candidates.append(candidate)
        baselines.append(baseline)
        deltas.append(candidate - baseline)
    return candidates, baselines, deltas


def _validate_population_record_grid(
    records: Sequence[Mapping[str, Any]],
    *,
    opponents: Sequence[str],
    paired_seeds: int,
    seed: int,
    seed_stride: int,
) -> None:
    if paired_seeds <= 0 or seed <= 0 or seed_stride <= 0:
        raise ValueError("population seed schedule must be positive")
    if not opponents or len(set(opponents)) != len(opponents):
        raise ValueError("population opponents must be non-empty and unique")
    if "current" in opponents:
        raise ValueError("the M4 population record grid must not reference current")
    expected_seeds = {
        seed + index * seed_stride for index in range(paired_seeds)
    }
    expected = {
        (opponent, hand_seed, seat)
        for opponent in opponents
        for hand_seed in expected_seeds
        for seat in ("first", "second")
    }
    observed: set[tuple[str, int, str]] = set()
    for row_index, row in enumerate(records):
        if row.get("schema") != M4_HAND_RECORD_SCHEMA:
            raise ValueError(f"population record {row_index} schema mismatch")
        opponent = row.get("opponent")
        hand_seed = row.get("seed")
        seat = row.get("seat")
        if (
            not isinstance(opponent, str)
            or isinstance(hand_seed, bool)
            or not isinstance(hand_seed, int)
            or not isinstance(seat, str)
        ):
            raise ValueError(f"population record {row_index} identity is invalid")
        identity = (opponent, hand_seed, seat)
        if identity not in expected:
            raise ValueError(f"population record {row_index} is outside frozen grid")
        if identity in observed:
            raise ValueError(f"duplicate population record identity: {identity}")
        candidate_score = _finite_record_number(
            row.get("candidate_score"), row_index=row_index, field="candidate_score"
        )
        baseline_score = _finite_record_number(
            row.get("baseline_score"), row_index=row_index, field="baseline_score"
        )
        delta = _finite_record_number(
            row.get("delta"), row_index=row_index, field="delta"
        )
        if not math.isclose(
            delta,
            candidate_score - baseline_score,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"population record {row_index} delta does not match terminal scores"
            )
        if row.get("counterfactual_basis") != (
            "same_seed_physical_seat_opponent_policy_seeds_v1"
        ):
            raise ValueError(
                f"population record {row_index} counterfactual basis mismatch"
            )
        expected_hero_seed = hand_seed * 4 + (1 if seat == "second" else 0)
        expected_opponent_seed = hand_seed * 4 + (0 if seat == "second" else 1)
        if (
            row.get("hero_policy_seed") != expected_hero_seed
            or row.get("opponent_policy_seed") != expected_opponent_seed
        ):
            raise ValueError(
                f"population record {row_index} policy seed binding mismatch"
            )
        candidate_digest = _record_digest(
            row.get("candidate_gameplay_digest"),
            row_index=row_index,
            field="candidate_gameplay_digest",
        )
        baseline_digest = _record_digest(
            row.get("baseline_gameplay_digest"),
            row_index=row_index,
            field="baseline_gameplay_digest",
        )
        log_valid = row.get("override_log_valid")
        fired = row.get("override_fired")
        if not isinstance(log_valid, bool):
            raise ValueError(
                f"population record {row_index} override_log_valid is not boolean"
            )
        if log_valid and not isinstance(fired, bool):
            raise ValueError(
                f"population record {row_index} valid log has no boolean fire state"
            )
        stored_cancellation = row.get("nonfire_cancellation_valid")
        if fired is False:
            recomputed_cancellation = (
                candidate_digest == baseline_digest and delta == 0.0
            )
            if stored_cancellation is not recomputed_cancellation:
                raise ValueError(
                    f"population record {row_index} nonfire cancellation claim "
                    "does not match trajectory digests and terminal delta"
                )
        elif stored_cancellation is not None:
            raise ValueError(
                f"population record {row_index} nonfire cancellation must be null"
            )
        observed.add(identity)
    missing = expected - observed
    if missing:
        raise ValueError(
            f"population record grid is incomplete: missing={len(missing)}"
        )


def _finite_record_number(value: Any, *, row_index: int, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"population record {row_index} {field} is invalid")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"population record {row_index} {field} is invalid"
        ) from exc
    if not math.isfinite(number):
        raise ValueError(f"population record {row_index} {field} is not finite")
    return number


def _record_digest(value: Any, *, row_index: int, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value.lower())
    ):
        raise ValueError(f"population record {row_index} {field} is invalid")
    return value.lower()


def summarize_hu_m4_population_records(
    records: Sequence[Mapping[str, Any]],
    *,
    opponents: Sequence[str],
    paired_seeds: int,
    seed: int,
    seed_stride: int,
    records_output: Path | str | None = None,
    elapsed_seconds: float | None = None,
    include_records: bool = False,
    baseline_profile: str = "stage18_p1",
) -> dict[str, Any]:
    """Build the promotion summary from a complete frozen record grid.

    Spot workers may evaluate disjoint seed shards, but final confidence
    intervals must be computed once over the merged seed clusters.  This
    function is therefore shared by the monolithic evaluator and the strict
    shard merger rather than averaging per-shard confidence intervals.
    """

    ordered_opponents = tuple(str(name) for name in opponents)
    if baseline_profile not in {"stage18_p1", "stage19_p0"}:
        raise ValueError("population baseline profile is not explicitly supported")
    _validate_population_record_grid(
        records,
        opponents=ordered_opponents,
        paired_seeds=paired_seeds,
        seed=seed,
        seed_stride=seed_stride,
    )
    opponent_summaries = {
        opponent: _summarize_opponent(
            opponent, [row for row in records if row["opponent"] == opponent]
        )
        for opponent in ordered_opponents
    }
    all_first = [row for row in records if row["seat"] == "first"]
    all_second = [row for row in records if row["seat"] == "second"]
    valid_records = [row for row in records if _promotion_record_valid(row)]
    paired_candidates, paired_baselines, paired_deltas = _clustered_population_pairs(
        valid_records
    )
    worst_name = min(
        opponent_summaries,
        key=lambda name: (
            float(opponent_summaries[name]["seat_swap"]["delta_ev_per_hand"]["mean"])
            if opponent_summaries[name]["seat_swap"]["delta_ev_per_hand"]["mean"]
            is not None
            else math.inf
        ),
    )
    invalid = sum(not row["override_log_valid"] for row in records)
    nonfire_mismatches = sum(
        row["nonfire_cancellation_valid"] is False for row in records
    )
    nonfire_nonzero = sum(
        row.get("override_fired") is False and float(row["delta"]) != 0.0
        for row in records
    )
    nonfire_unknown = sum(
        row.get("override_fired") is False
        and row.get("nonfire_cancellation_valid") is None
        for row in records
    )
    promotion_eligible = (
        invalid == 0
        and nonfire_mismatches == 0
        and nonfire_nonzero == 0
        and nonfire_unknown == 0
    )
    result: dict[str, Any] = {
        "schema": M4_POPULATION_EVALUATION_SCHEMA,
        "evaluation_basis": (
            f"fresh_same_seed_candidate_vs_{baseline_profile}_counterfactual_v1"
        ),
        "paired_seat_swap": True,
        "seed": seed,
        "seed_stride": seed_stride,
        "paired_seeds_per_opponent": paired_seeds,
        "opponents": list(ordered_opponents),
        "candidate_hands": len(records),
        "baseline_hands": len(records),
        "trace_hands": len(records) * 2,
        "by_opponent": opponent_summaries,
        "population": {
            "by_seat": {
                "first": _seat_summary(all_first),
                "second": _seat_summary(all_second),
            },
            "all_seats": _seat_summary(records),
            "paired_seat_swap": {
                "candidate_ev_per_hand": _mean_ci95(paired_candidates),
                "baseline_ev_per_hand": _mean_ci95(paired_baselines),
                "delta_ev_per_hand": _mean_ci95(paired_deltas),
                "delta_loss_tail": _loss_tail(paired_deltas),
                "opponent_weighting": "equal_by_seed_count",
                "ci_independence_unit": "hand_seed_cluster_after_opponent_average",
            },
        },
        "worst_case_opponent": {
            "opponent": worst_name,
            "delta_ev_per_hand": opponent_summaries[worst_name]["seat_swap"][
                "delta_ev_per_hand"
            ],
            "delta_loss_tail": opponent_summaries[worst_name]["seat_swap"][
                "delta_loss_tail"
            ],
        },
        "invalid_counterfactuals": invalid,
        "nonfire_cancellation_mismatches": nonfire_mismatches,
        "nonfire_nonzero_deltas": nonfire_nonzero,
        "nonfire_cancellation_unknown": nonfire_unknown,
        "promotion_eligible_counterfactual_contract": promotion_eligible,
        "primary_metrics_exclude_invalid_records": True,
        "records_output": str(records_output) if records_output else None,
        "elapsed_seconds": elapsed_seconds,
    }
    if include_records:
        result["records"] = [dict(row) for row in records]
    return result


def evaluate_hu_m4_population(
    *,
    candidate_policy_factory: PolicyFactory,
    baseline_policy_factory: PolicyFactory,
    opponent_policy_factories: Mapping[str, PolicyFactory],
    paired_seeds: int,
    seed: int,
    seed_stride: int = 1,
    trace_fn: TraceFunction = trace_hand,
    include_records: bool = False,
    records_output: Path | None = None,
    progress_every: int = 0,
    baseline_profile: str = "stage18_p1",
) -> dict[str, Any]:
    """Run fresh candidate-vs-baseline counterfactuals over a population."""

    if paired_seeds <= 0:
        raise ValueError("paired_seeds must be positive")
    if seed_stride <= 0:
        raise ValueError("seed_stride must be positive")
    if not opponent_policy_factories:
        raise ValueError("at least one explicit opponent factory is required")
    if any(name == "current" for name in opponent_policy_factories):
        raise ValueError("the M4 evaluator must not reference the current profile")

    started_at = time.time()
    records: list[dict[str, Any]] = []
    output_handle = None
    if records_output is not None:
        records_output.parent.mkdir(parents=True, exist_ok=True)
        output_handle = records_output.open("w", encoding="utf-8")
    try:
        completed_pairs = 0
        for opponent_name, opponent_factory in opponent_policy_factories.items():
            if not opponent_name:
                raise ValueError("opponent names must be non-empty")
            for index in range(paired_seeds):
                hand_seed = seed + index * seed_stride
                for seat in ("first", "second"):
                    row = _trace_candidate_and_baseline(
                        trace_fn=trace_fn,
                        hand_seed=hand_seed,
                        seat=seat,
                        opponent_name=opponent_name,
                        candidate_factory=candidate_policy_factory,
                        baseline_factory=baseline_policy_factory,
                        opponent_factory=opponent_factory,
                    )
                    records.append(row)
                    if output_handle is not None:
                        output_handle.write(
                            json.dumps(
                                row,
                                ensure_ascii=True,
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                completed_pairs += 1
                if progress_every > 0 and completed_pairs % progress_every == 0:
                    print(
                        json.dumps(
                            {
                                "event": "m4_population_progress",
                                "completed_opponent_seed_pairs": completed_pairs,
                                "candidate_hands": len(records),
                                "elapsed_seconds": time.time() - started_at,
                            },
                            separators=(",", ":"),
                        ),
                        flush=True,
                    )
    finally:
        if output_handle is not None:
            output_handle.close()
    return summarize_hu_m4_population_records(
        records,
        opponents=tuple(opponent_policy_factories),
        paired_seeds=paired_seeds,
        seed=seed,
        seed_stride=seed_stride,
        records_output=records_output,
        elapsed_seconds=time.time() - started_at,
        include_records=include_records,
        baseline_profile=baseline_profile,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fresh paired population evaluation for M4 T1 second-seat override"
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Single calibrated HuM4JointActionModel for promotion evaluation.",
    )
    parser.add_argument(
        "--baseline-profile",
        choices=("stage18_p1", "stage19_p0"),
        default="stage18_p1",
        help="Explicit whole-chain counterfactual baseline; never resolves current.",
    )
    parser.add_argument("--expected-model-sha256")
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--runtime-source-manifest", type=Path)
    parser.add_argument("--runtime-source-root", type=Path)
    parser.add_argument("--runtime-dependency-root", type=Path)
    parser.add_argument("--seed-registry-sha256")
    parser.add_argument("--threshold-lock", type=Path)
    parser.add_argument("--diagnostic-legacy", action="store_true")
    parser.add_argument("--candidate-model", type=Path)
    parser.add_argument("--safety-model", type=Path)
    parser.add_argument("--safety-threshold", type=float)
    parser.add_argument(
        "--stage18-candidate-model",
        type=Path,
        default=DEFAULT_HU_TURN1_STAGE18_P1_MODEL,
    )
    parser.add_argument(
        "--stage18-safe-selector-model",
        type=Path,
        default=DEFAULT_HU_TURN1_STAGE18_P1_SAFE_SELECTOR_MODEL,
    )
    parser.add_argument(
        "--opponents",
        choices=M4_OPPONENT_PROFILES,
        nargs="+",
        default=list(M4_OPPONENT_PROFILES),
    )
    parser.add_argument("--paired-seeds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026071301)
    parser.add_argument("--seed-stride", type=int, default=1009)
    parser.add_argument("--opening-lookahead-samples", type=int, default=8)
    parser.add_argument("--records-output", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args(argv)
    if args.diagnostic_legacy:
        if args.model is not None:
            parser.error("--model cannot be combined with --diagnostic-legacy")
        if (
            args.candidate_model is None
            or args.safety_model is None
            or args.safety_threshold is None
        ):
            parser.error(
                "diagnostic legacy mode requires --candidate-model, --safety-model, "
                "and --safety-threshold"
            )
        if not 0.0 <= args.safety_threshold <= 1.0:
            parser.error("--safety-threshold must be in [0, 1]")
    elif args.model is None:
        parser.error("promotion evaluation requires the single --model artifact")
    elif any(
        value is not None
        for value in (args.candidate_model, args.safety_model, args.safety_threshold)
    ):
        parser.error("legacy model/threshold options require --diagnostic-legacy")
    core_runtime_binding = (
        args.expected_model_sha256,
        args.freeze_manifest,
        args.training_manifest,
    )
    if sum(value is not None for value in core_runtime_binding) not in {0, 3}:
        parser.error(
            "--expected-model-sha256, --freeze-manifest, and --training-manifest "
            "must be provided together"
        )
    if args.threshold_lock is not None and args.expected_model_sha256 is None:
        parser.error("--threshold-lock requires the three core runtime bindings")
    source_runtime_binding = (
        args.runtime_source_manifest,
        args.runtime_source_root,
        args.runtime_dependency_root,
    )
    if sum(value is not None for value in source_runtime_binding) not in {0, 3}:
        parser.error(
            "--runtime-source-manifest, --runtime-source-root, and "
            "--runtime-dependency-root must be provided together"
        )
    if args.runtime_source_manifest is not None and args.expected_model_sha256 is None:
        parser.error("runtime source bindings require the three core runtime bindings")
    if args.seed_registry_sha256 is not None and (
        len(args.seed_registry_sha256) != 64
        or any(character not in "0123456789abcdef" for character in args.seed_registry_sha256)
    ):
        parser.error("--seed-registry-sha256 must be a lowercase SHA-256 digest")
    if args.paired_seeds <= 0:
        parser.error("--paired-seeds must be positive")
    if args.seed_stride <= 0:
        parser.error("--seed-stride must be positive")
    if len(set(args.opponents)) != len(args.opponents):
        parser.error("--opponents must not contain duplicates")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_runtime_dependencies = validate_attempt08_runtime_dependencies
    if args.runtime_source_manifest is not None:
        source_contract = json.loads(
            args.runtime_source_manifest.read_text(encoding="utf-8-sig")
        )
        semantic_schema = source_contract.get("semantic_closure", {}).get("schema")
        if semantic_schema == ATTEMPT08_DISTILLED_RUNTIME_SEMANTIC_CLOSURE_SCHEMA:
            validate_frozen_modules = validate_attempt08_frozen_modules
            module_names = (
                "ofc_regular.hu_m43_attempt08_distilled_runtime",
                "ofc_regular.hu_m43_attempt08_distilled_model",
                "ofc_regular.evaluate_hu_m4_population",
                "ofc_regular.hu_m43_joint_model_loader",
                "ofc_regular.hu_m4_t1_policy",
                "ofc_regular.ai_profiles",
                "ofc_regular.evaluate_matchups",
            )
        elif semantic_schema == ATTEMPT10_DISTILLED_RUNTIME_SEMANTIC_CLOSURE_SCHEMA:
            validate_frozen_modules = validate_attempt10_frozen_modules
            validate_runtime_dependencies = validate_attempt10_runtime_dependencies
            module_names = (
                "ofc_regular.hu_m43_attempt10_distilled_runtime",
                "ofc_regular.hu_m43_attempt10_distilled_model",
                "ofc_regular.evaluate_hu_m4_population",
                "ofc_regular.hu_m43_joint_model_loader",
                "ofc_regular.hu_m4_t1_policy",
                "ofc_regular.validate_hu_m43_attempt10_acceptance",
                "ofc_regular.ai_profiles",
                "ofc_regular.evaluate_matchups",
            )
        elif semantic_schema == ATTEMPT11_DISTILLED_RUNTIME_SEMANTIC_CLOSURE_SCHEMA:
            validate_frozen_modules = validate_attempt11_frozen_modules
            validate_runtime_dependencies = validate_attempt11_runtime_dependencies
            module_names = (
                "ofc_regular.hu_m43_attempt11_distilled_runtime",
                "ofc_regular.hu_m43_attempt11_distilled_model",
                "ofc_regular.evaluate_hu_m4_population",
                "ofc_regular.hu_m43_joint_model_loader",
                "ofc_regular.hu_m4_t1_policy",
                "ofc_regular.validate_hu_m43_attempt11_acceptance",
                "ofc_regular.ai_profiles",
                "ofc_regular.evaluate_matchups",
            )
        elif semantic_schema == ATTEMPT12_DISTILLED_RUNTIME_SEMANTIC_CLOSURE_SCHEMA:
            validate_frozen_modules = validate_attempt12_frozen_modules
            validate_runtime_dependencies = validate_attempt12_runtime_dependencies
            module_names = (
                "ofc_regular.hu_m43_attempt12_distilled_runtime",
                "ofc_regular.hu_m43_attempt12_distilled_model",
                "ofc_regular.evaluate_hu_m4_population",
                "ofc_regular.hu_m43_joint_model_loader",
                "ofc_regular.hu_m4_t1_policy",
                "ofc_regular.validate_hu_m43_attempt12_acceptance",
                "ofc_regular.ai_profiles",
                "ofc_regular.evaluate_matchups",
            )
        elif semantic_schema == ATTEMPT13_DISTILLED_RUNTIME_SEMANTIC_CLOSURE_SCHEMA:
            validate_frozen_modules = validate_attempt13_frozen_modules
            validate_runtime_dependencies = validate_attempt13_runtime_dependencies
            module_names = tuple(ATTEMPT13_BOUND_EXECUTION_MODULES) + (
                "ofc_regular.ai_profiles",
                "ofc_regular.evaluate_matchups",
            )
        else:
            raise ValueError("unsupported distilled runtime semantic closure")
        validate_frozen_modules(
            extracted_root=args.runtime_source_root,
            manifest=args.runtime_source_manifest,
            module_names=module_names,
        )
    model_paths = ModelPaths(
        hu_turn1_stage18_p1=args.stage18_candidate_model,
        hu_turn1_stage18_p1_safe_selector=args.stage18_safe_selector_model,
    )
    needed: set[str] = set()
    for opponent in args.opponents:
        needed.update(required_profiles(args.baseline_profile, opponent))
    bundle = load_model_bundle(model_paths, needed)
    joint_model = None
    if not args.diagnostic_legacy:
        joint_model = load_hu_m43_joint_action_model(
            args.model,
            expected_sha256=args.expected_model_sha256,
            freeze_manifest=args.freeze_manifest,
            training_manifest_path=args.training_manifest,
            threshold_lock_path=args.threshold_lock,
            runtime_source_manifest_path=args.runtime_source_manifest,
            runtime_source_root=args.runtime_source_root,
            runtime_dependency_root=args.runtime_dependency_root,
        )
        if requires_bound_population_runtime(joint_model) and (
            args.expected_model_sha256 is None
        ):
            raise ValueError(
                "M4.3 population evaluation requires the expected model SHA, "
                "runtime freeze, and final training manifest (plus the frozen "
                "runtime source closure for distilled policies)"
            )
        if (
            getattr(joint_model, "schema", None) in {
                HU_M43_ATTEMPT08_DISTILLED_MODEL_SCHEMA,
                HU_M43_ATTEMPT10_DISTILLED_MODEL_SCHEMA,
                HU_M43_ATTEMPT11_DISTILLED_MODEL_SCHEMA,
                HU_M43_ATTEMPT12_DISTILLED_MODEL_SCHEMA,
                HU_M43_ATTEMPT13_DISTILLED_MODEL_SCHEMA,
            }
            and args.seed_registry_sha256 is None
        ):
            raise ValueError(
                "distilled population evaluation requires the hashed seed registry"
            )
    if joint_model is not None:
        candidate_model_path = args.model
        safety_model_path = args.model
        safety_threshold = float(joint_model.safety_threshold)
    else:
        candidate_model_path = args.candidate_model
        safety_model_path = args.safety_model
        safety_threshold = float(args.safety_threshold)

    runtime_binding_verified = bool(
        joint_model is not None
        and args.expected_model_sha256 is not None
        and (
            (
                getattr(joint_model, "schema", None)
                == HU_M43_ATTEMPT08_DISTILLED_MODEL_SCHEMA
                and is_bound_attempt08_distilled_model(joint_model)
            )
            or (
                getattr(joint_model, "schema", None)
                == HU_M43_ATTEMPT10_DISTILLED_MODEL_SCHEMA
                and is_bound_attempt10_distilled_model(joint_model)
            )
            or (
                getattr(joint_model, "schema", None)
                == HU_M43_ATTEMPT11_DISTILLED_MODEL_SCHEMA
                and is_bound_attempt11_distilled_model(joint_model)
            )
            or (
                getattr(joint_model, "schema", None)
                == HU_M43_ATTEMPT12_DISTILLED_MODEL_SCHEMA
                and is_bound_attempt12_distilled_model(joint_model)
            )
            or (
                getattr(joint_model, "schema", None)
                == HU_M43_ATTEMPT13_DISTILLED_MODEL_SCHEMA
                and is_bound_attempt13_distilled_model(joint_model)
            )
            or getattr(joint_model, "schema", None)
            not in {
                HU_M43_ATTEMPT08_DISTILLED_MODEL_SCHEMA,
                HU_M43_ATTEMPT10_DISTILLED_MODEL_SCHEMA,
                HU_M43_ATTEMPT11_DISTILLED_MODEL_SCHEMA,
                HU_M43_ATTEMPT12_DISTILLED_MODEL_SCHEMA,
                HU_M43_ATTEMPT13_DISTILLED_MODEL_SCHEMA,
            }
        )
    )

    def baseline_factory(
        *, policy_seed: int, seat: str, decision_log: list[dict[str, Any]] | None
    ) -> Any:
        del decision_log
        return build_policy(
            args.baseline_profile,
            bundle,
            seed=policy_seed,
            seat=seat,
            opening_lookahead_samples=args.opening_lookahead_samples,
        )

    def candidate_factory(
        *, policy_seed: int, seat: str, decision_log: list[dict[str, Any]] | None
    ) -> Any:
        baseline = baseline_factory(
            policy_seed=policy_seed,
            seat=seat,
            decision_log=None,
        )
        if joint_model is not None:
            return HuM4T1SelectiveOverridePolicy(
                baseline,
                action_value_model=joint_model,
                safety_model=joint_model,
                safety_probability_threshold=safety_threshold,
                decision_log=decision_log,
                runtime_binding_verified=runtime_binding_verified,
            )
        return HuM4T1SelectiveOverridePolicy.from_model_paths(
            baseline,
            action_value_model_path=candidate_model_path,
            safety_model_path=safety_model_path,
            safety_probability_threshold=safety_threshold,
            decision_log=decision_log,
        )

    opponent_factories: dict[str, PolicyFactory] = {}
    for opponent_name in args.opponents:
        def opponent_factory(
            *,
            policy_seed: int,
            seat: str,
            decision_log: list[dict[str, Any]] | None,
            _profile: str = opponent_name,
        ) -> Any:
            del decision_log
            return build_policy(
                _profile,
                bundle,
                seed=policy_seed,
                seat=seat,
                opening_lookahead_samples=args.opening_lookahead_samples,
            )

        opponent_factories[opponent_name] = opponent_factory

    result = evaluate_hu_m4_population(
        candidate_policy_factory=candidate_factory,
        baseline_policy_factory=baseline_factory,
        opponent_policy_factories=opponent_factories,
        paired_seeds=args.paired_seeds,
        seed=args.seed,
        seed_stride=args.seed_stride,
        records_output=args.records_output,
        progress_every=args.progress_every,
        baseline_profile=args.baseline_profile,
    )
    runtime_source_contract = None
    runtime_dependencies = None
    if args.runtime_source_manifest is not None:
        runtime_source_contract = json.loads(
            args.runtime_source_manifest.read_text(encoding="utf-8-sig")
        )
        runtime_dependencies = validate_runtime_dependencies(
            args.runtime_dependency_root
        )
    result["runtime_config"] = {
        "baseline_profile": args.baseline_profile,
        "candidate_model": str(candidate_model_path),
        "candidate_model_sha256": _sha256_file(candidate_model_path),
        "safety_model": str(safety_model_path),
        "safety_model_sha256": _sha256_file(safety_model_path),
        "safety_threshold": safety_threshold,
        "stage18_candidate_model": str(args.stage18_candidate_model),
        "stage18_safe_selector_model": str(args.stage18_safe_selector_model),
        "opponents": list(args.opponents),
        "current_profile_used": False,
        "promotion_artifact_contract": not args.diagnostic_legacy,
        "diagnostic_legacy": bool(args.diagnostic_legacy),
        "action_score_mode": (
            joint_model.action_score_mode if joint_model is not None else "diagnostic_legacy"
        ),
        "model_id": joint_model.model_id if joint_model is not None else None,
        "model_schema": getattr(joint_model, "schema", None),
        "artifact_schema": getattr(joint_model, "artifact_schema", None),
        "feature_schema": getattr(joint_model, "feature_schema", None),
        "head_schema": getattr(joint_model, "head_schema", None),
        "runtime_binding_verified": runtime_binding_verified,
        "freeze_manifest_sha256": (
            _sha256_file(args.freeze_manifest)
            if args.freeze_manifest is not None
            else None
        ),
        "training_manifest_sha256": (
            _sha256_file(args.training_manifest)
            if args.training_manifest is not None
            else None
        ),
        "runtime_source_manifest_sha256": (
            _sha256_file(args.runtime_source_manifest)
            if args.runtime_source_manifest is not None
            else None
        ),
        "runtime_source_root": (
            str(args.runtime_source_root) if args.runtime_source_root is not None else None
        ),
        "runtime_dependency_root": (
            str(args.runtime_dependency_root)
            if args.runtime_dependency_root is not None
            else None
        ),
        "source_model_manifest_sha256": (
            runtime_dependencies["source_model_manifest_sha256"]
            if runtime_dependencies is not None
            else None
        ),
        "source_native_manifest_sha256": (
            runtime_dependencies["source_native_manifest_sha256"]
            if runtime_dependencies is not None
            else None
        ),
        "runtime_dependency_closure_sha256": (
            runtime_dependencies["sha256"]
            if runtime_dependencies is not None
            else None
        ),
        "runtime_source_closure_sha256": (
            runtime_source_contract["file_set"]["sha256"]
            if runtime_source_contract is not None
            else None
        ),
        "runtime_semantic_closure_sha256": (
            runtime_source_contract["semantic_closure"]["sha256"]
            if runtime_source_contract is not None
            else None
        ),
        "runtime_requirements_sha256": (
            runtime_source_contract["semantic_closure"]["external_runtime"][
                "requirements_sha256"
            ]
            if runtime_source_contract is not None
            else None
        ),
        "runtime_fingerprint_sha256": (
            runtime_source_contract["semantic_closure"]["external_runtime"][
                "runtime_fingerprint_sha256"
            ]
            if runtime_source_contract is not None
            else None
        ),
        "seed_registry_sha256": args.seed_registry_sha256,
        "threshold_lock_sha256": (
            _sha256_file(args.threshold_lock)
            if args.threshold_lock is not None
            else None
        ),
        "safety_enabled": (
            bool(joint_model.safety_enabled) if joint_model is not None else True
        ),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
