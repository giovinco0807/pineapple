"""Seat-swap validation for HU Turn2 Stage8 selective overrides."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .ai_profiles import (
    DEFAULT_OPENING_MODEL,
    DEFAULT_TURN1_MODEL,
    DEFAULT_TURN2_MODEL,
    DEFAULT_TURN3_MODEL,
)
from .evaluate_matchups import trace_hand
from .hu_turn2_stage8_runtime import (
    HuTurn2Stage8RuntimeConfig,
    HuTurn2Stage8SelectiveOverridePolicy,
    load_hu_turn2_stage8_model,
)
from .hu_turn3_model import load_hu_action_value_model
from .play_ai import _prediction_thread_context
from .policy import RegularAiPolicy
from .turn3_model import load_action_value_model


DEFAULT_T2_STAGE8_MODEL = Path("models/hu_turn2_stage8_broad_20k_mc512_reference_override_cached_rank_wide.pt")
DEFAULT_CALIBRATION_VALUES = Path("outputs/hu_turn2_stage8_20k_mc512_calibration/state_calibration_values.csv")
DEFAULT_STAGE7_MODEL = Path("models/hu_turn3_stage7_reference_override_cached_rank_wide.pt")
DEFAULT_STAGE3_REFERENCE = Path("models/hu_turn3_stage3_mc32_500k_plus_m8_12_f128_100k_w2_cached_rank_wide.pt")
DEFAULT_OUTPUT_DIR = Path("outputs/evals/hu_turn2_stage8_20k_seat_swap")

DEFAULT_GRID_MARGINS = (2.00, 2.50, 2.75, 2.91, 3.10, 3.25, 3.50)
DEFAULT_GRID_REFERENCES = (0.00, 0.05, 0.10, 0.20, 0.40)
DEFAULT_GRID_GATES = (0.85, 0.90, 0.925, 0.95)
SHORTLIST_CONFIGS = (
    "2.75/0.05/0.90",
    "2.91/0.10/0.90",
    "3.10/0.10/0.90",
    "3.25/0.10/0.925",
    "3.50/0.20/0.95",
)


@dataclass
class ModelParts:
    opening: Any
    turn1: Any
    turn2_baseline: Any
    turn3: Any
    hu_turn3_stage7: Any
    hu_turn3_reference: Any
    hu_turn2_stage8: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games-per-seed", type=int, default=300)
    parser.add_argument("--seeds", default="2026061701,2026061702,2026061703")
    parser.add_argument(
        "--seed-stride",
        type=int,
        default=1_000_000,
        help="Spacing used to derive non-overlapping hand seeds from close base seeds.",
    )
    parser.add_argument(
        "--configs",
        default="shortlist",
        help="'full', 'shortlist', or comma-separated m/r/g triples.",
    )
    parser.add_argument("--opening-model", type=Path, default=DEFAULT_OPENING_MODEL)
    parser.add_argument("--turn1-model", type=Path, default=DEFAULT_TURN1_MODEL)
    parser.add_argument("--turn2-baseline-model", type=Path, default=DEFAULT_TURN2_MODEL)
    parser.add_argument("--turn3-model", type=Path, default=DEFAULT_TURN3_MODEL)
    parser.add_argument("--hu-turn3-stage7-model", type=Path, default=DEFAULT_STAGE7_MODEL)
    parser.add_argument("--hu-turn3-reference-model", type=Path, default=DEFAULT_STAGE3_REFERENCE)
    parser.add_argument("--hu-turn2-stage8-model", type=Path, default=DEFAULT_T2_STAGE8_MODEL)
    parser.add_argument("--calibration-values", type=Path, default=DEFAULT_CALIBRATION_VALUES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--prediction-threads", type=int, default=1)
    parser.add_argument("--opening-lookahead-samples", type=int, default=64)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--write-decision-log", action="store_true")
    return parser.parse_args()


def parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def parse_configs(value: str) -> list[HuTurn2Stage8RuntimeConfig]:
    if value == "full":
        return [
            HuTurn2Stage8RuntimeConfig(margin, reference, gate)
            for margin in DEFAULT_GRID_MARGINS
            for reference in DEFAULT_GRID_REFERENCES
            for gate in DEFAULT_GRID_GATES
        ]
    if value == "shortlist":
        parts = SHORTLIST_CONFIGS
    else:
        parts = tuple(part.strip() for part in value.split(",") if part.strip())
    configs = []
    for item in parts:
        fields = item.replace("_", "/").split("/")
        if len(fields) < 3:
            raise ValueError(f"config must be m/r/g with optional /seat=first|second and /score=N: {item}")
        min_model_score: float | None = None
        allowed_seats: tuple[str, ...] = ()
        for option in fields[3:]:
            key, separator, raw_value = option.partition("=")
            key = key.strip().lower()
            raw_value = raw_value.strip().lower() if separator else ""
            if key in {"score", "minscore", "min_model_score", "s"} and separator:
                min_model_score = float(raw_value)
            elif key in {"seat", "seats", "allowed_seats"} and separator:
                if raw_value in {"all", "any", "*"}:
                    allowed_seats = ()
                else:
                    seats = tuple(seat.strip() for seat in raw_value.replace("|", "+").split("+") if seat.strip())
                    invalid = [seat for seat in seats if seat not in {"first", "second"}]
                    if invalid:
                        raise ValueError(f"invalid seat filter in config {item}: {invalid}")
                    allowed_seats = seats
            else:
                raise ValueError(f"unknown config option in {item}: {option}")
        configs.append(
            HuTurn2Stage8RuntimeConfig(
                min_margin=float(fields[0]),
                reference_min_margin=float(fields[1]),
                gate_threshold=float(fields[2]),
                min_model_score=min_model_score,
                allowed_seats=allowed_seats,
            )
        )
    if not configs:
        raise ValueError("at least one config is required")
    return configs


def load_parts(args: argparse.Namespace) -> ModelParts:
    return ModelParts(
        opening=load_action_value_model(args.opening_model),
        turn1=load_action_value_model(args.turn1_model),
        turn2_baseline=load_action_value_model(args.turn2_baseline_model),
        turn3=load_action_value_model(args.turn3_model),
        hu_turn3_stage7=load_hu_action_value_model(args.hu_turn3_stage7_model),
        hu_turn3_reference=load_hu_action_value_model(args.hu_turn3_reference_model),
        hu_turn2_stage8=load_hu_turn2_stage8_model(args.hu_turn2_stage8_model, device=args.device),
    )


def make_baseline_policy(parts: ModelParts, *, seed: int, seat: str, opening_lookahead_samples: int) -> RegularAiPolicy:
    return RegularAiPolicy(
        opening_model=parts.opening,
        turn1_model=parts.turn1,
        turn2_model=parts.turn2_baseline,
        turn3_model=parts.turn3,
        hu_turn3_model=parts.hu_turn3_stage7,
        hu_turn3_reference_model=parts.hu_turn3_reference,
        hu_turn3_min_margin=5.0,
        hu_turn3_reference_min_margin=10.0,
        seat=seat,
        seed=seed,
        opening_lookahead_samples=opening_lookahead_samples,
    )


def make_candidate_policy(
    parts: ModelParts,
    *,
    config: HuTurn2Stage8RuntimeConfig,
    decisions: list[dict[str, Any]],
    context: dict[str, Any],
    seed: int,
    seat: str,
    opening_lookahead_samples: int,
) -> HuTurn2Stage8SelectiveOverridePolicy:
    return HuTurn2Stage8SelectiveOverridePolicy(
        opening_model=parts.opening,
        turn1_model=parts.turn1,
        turn2_model=parts.turn2_baseline,
        turn3_model=parts.turn3,
        hu_turn3_model=parts.hu_turn3_stage7,
        hu_turn3_reference_model=parts.hu_turn3_reference,
        hu_turn3_min_margin=5.0,
        hu_turn3_reference_min_margin=10.0,
        hu_turn2_stage8_model=parts.hu_turn2_stage8,
        hu_turn2_stage8_config=config,
        hu_turn2_decision_log=decisions,
        hu_turn2_context=context,
        seat=seat,
        seed=seed,
        opening_lookahead_samples=opening_lookahead_samples,
    )


def summarize_values(values: list[float], prefix: str = "") -> dict[str, float]:
    if not values:
        return {
            f"{prefix}mean": 0.0,
            f"{prefix}std_error": 0.0,
            f"{prefix}ci95_low": 0.0,
            f"{prefix}ci95_high": 0.0,
        }
    mean = float(np.mean(values))
    stderr = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return {
        f"{prefix}mean": mean,
        f"{prefix}std_error": stderr,
        f"{prefix}ci95_low": mean - 1.96 * stderr,
        f"{prefix}ci95_high": mean + 1.96 * stderr,
    }


def evaluate_config_seed(
    *,
    config: HuTurn2Stage8RuntimeConfig,
    seed: int,
    seed_stride: int,
    games: int,
    parts: ModelParts,
    opening_lookahead_samples: int,
    progress_every: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    paired_scores: list[float] = []
    position_scores: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    wins = losses = ties = 0
    started_at = time.time()
    for index in range(games):
        hand_seed = seed * seed_stride + index
        before = len(decisions)
        hand_ab = trace_hand(
            seed=hand_seed,
            profile_p0="hu_t2_stage8",
            profile_p1="baseline",
            policy_p0=make_candidate_policy(
                parts,
                config=config,
                decisions=decisions,
                context={"config_id": config.config_id, "paired_index": index, "seat_swap": "ab"},
                seed=hand_seed * 4,
                seat="first",
                opening_lookahead_samples=opening_lookahead_samples,
            ),
            policy_p1=make_baseline_policy(
                parts,
                seed=hand_seed * 4 + 1,
                seat="second",
                opening_lookahead_samples=opening_lookahead_samples,
            ),
        )
        for row in decisions[before:]:
            row["candidate_seat_score"] = float(hand_ab["score_p0"])
            row["paired_index"] = index
            row["hand_seed"] = hand_seed
        position_scores.append({"config_id": config.config_id, "seed": seed, "seat": "first", "score": float(hand_ab["score_p0"])})

        before = len(decisions)
        hand_ba = trace_hand(
            seed=hand_seed,
            profile_p0="baseline",
            profile_p1="hu_t2_stage8",
            policy_p0=make_baseline_policy(
                parts,
                seed=hand_seed * 4 + 2,
                seat="first",
                opening_lookahead_samples=opening_lookahead_samples,
            ),
            policy_p1=make_candidate_policy(
                parts,
                config=config,
                decisions=decisions,
                context={"config_id": config.config_id, "paired_index": index, "seat_swap": "ba"},
                seed=hand_seed * 4 + 3,
                seat="second",
                opening_lookahead_samples=opening_lookahead_samples,
            ),
        )
        candidate_second_score = -float(hand_ba["score_p0"])
        for row in decisions[before:]:
            row["candidate_seat_score"] = candidate_second_score
            row["paired_index"] = index
            row["hand_seed"] = hand_seed
        position_scores.append({"config_id": config.config_id, "seed": seed, "seat": "second", "score": candidate_second_score})

        paired_score = (float(hand_ab["score_p0"]) - float(hand_ba["score_p0"])) / 2.0
        paired_scores.append(paired_score)
        if paired_score > 0:
            wins += 1
        elif paired_score < 0:
            losses += 1
        else:
            ties += 1
        if progress_every > 0 and (index + 1) % progress_every == 0:
            print(
                json.dumps(
                    {
                        "event": "t2_seat_swap_progress",
                        "config_id": config.config_id,
                        "seed": seed,
                        "paired_seeds": index + 1,
                        **summarize_values(paired_scores, "ev_per_hand_"),
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
    no_override = Counter(str(row.get("no_override_reason", "")) for row in decisions if not row.get("override_fired"))
    summary = {
        "config_id": config.config_id,
        "hu_turn2_min_margin": config.min_margin,
        "hu_turn2_reference_min_margin": config.reference_min_margin,
        "hu_turn2_gate_threshold": config.gate_threshold,
        "hu_turn2_min_model_score": "" if config.min_model_score is None else config.min_model_score,
        "hu_turn2_allowed_seats": "+".join(config.allowed_seats),
        "seed": seed,
        "paired_seeds": games,
        "hands": games * 2,
        "ev_per_hand": float(np.mean(paired_scores)) if paired_scores else 0.0,
        **summarize_values(paired_scores, "paired_score_"),
        "paired_seed_wins": wins,
        "paired_seed_losses": losses,
        "paired_seed_ties": ties,
        "decision_count": len(decisions),
        "override_count": sum(1 for row in decisions if row.get("override_fired")),
        "override_rate": sum(1 for row in decisions if row.get("override_fired")) / max(len(decisions), 1),
        "no_override_reason_counts": json.dumps(dict(no_override), sort_keys=True),
        "elapsed_seconds": time.time() - started_at,
    }
    return summary, decisions, position_scores


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), q / 100.0))


def teacher_fired_rows(rows: list[dict[str, str]], config: HuTurn2Stage8RuntimeConfig, *, split: str = "test") -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("split") == split
        and int(safe_float(row.get("candidate_is_baseline"), 0.0)) == 0
        and (not config.allowed_seats or row.get("seat") in config.allowed_seats)
        and safe_float(row.get("predicted_delta_vs_baseline")) >= config.min_margin
        and safe_float(row.get("reference_margin_raw")) >= config.reference_min_margin
        and safe_float(row.get("gate_probability")) >= config.gate_threshold
        and (
            config.min_model_score is None
            or safe_float(row.get("predicted_candidate_EV", row.get("model_score", "")), -math.inf) >= config.min_model_score
        )
    ]


def teacher_summary(rows: list[dict[str, str]], config: HuTurn2Stage8RuntimeConfig) -> dict[str, Any]:
    test_rows = [row for row in rows if row.get("split") == "test"]
    fired = teacher_fired_rows(rows, config)
    gains = [safe_float(row.get("actual_delta_candidate_vs_baseline")) for row in fired]
    losses = [max(0.0, -gain) for gain in gains]
    false_positive = [row for row, gain in zip(fired, gains) if gain < 0.0]
    first = [row for row in fired if row.get("seat") == "first"]
    second = [row for row in fired if row.get("seat") == "second"]
    score_guard_column_available = config.min_model_score is None or any(
        row.get("predicted_candidate_EV", row.get("model_score", "")) not in (None, "") for row in test_rows
    )
    return {
        "teacher_evaluated_states": len(test_rows),
        "teacher_score_guard_column_available": int(score_guard_column_available),
        "teacher_override_count": len(fired),
        "teacher_override_rate": len(fired) / max(len(test_rows), 1),
        "avg_gain_on_override": float(np.mean(gains)) if gains else 0.0,
        "median_gain_on_override": float(np.median(gains)) if gains else 0.0,
        "false_positive_override_rate": len(false_positive) / max(len(fired), 1),
        "false_positive_count": len(false_positive),
        "avg_false_positive_cost": float(np.mean([loss for loss in losses if loss > 0.0])) if false_positive else 0.0,
        "p90_loss": percentile(losses, 90),
        "p95_loss": percentile(losses, 95),
        "p99_loss": percentile(losses, 99),
        "max_loss": max(losses) if losses else 0.0,
        "count_positive_override": sum(1 for gain in gains if gain > 0.0),
        "count_negative_override": sum(1 for gain in gains if gain < 0.0),
        "first_teacher_override_rate": len(first) / max(sum(1 for row in test_rows if row.get("seat") == "first"), 1),
        "second_teacher_override_rate": len(second) / max(sum(1 for row in test_rows if row.get("seat") == "second"), 1),
        "first_teacher_avg_gain": float(np.mean([safe_float(row.get("actual_delta_candidate_vs_baseline")) for row in first])) if first else 0.0,
        "second_teacher_avg_gain": float(np.mean([safe_float(row.get("actual_delta_candidate_vs_baseline")) for row in second])) if second else 0.0,
    }


def teacher_bucket_breakdown(rows: list[dict[str, str]], configs: list[HuTurn2Stage8RuntimeConfig]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groupers = {
        "bucket_group": lambda row: row.get("bucket_group", "unknown"),
        "actual_high_regret": lambda row: str(row.get("actual_high_regret", "")),
        "actual_low_margin": lambda row: str(row.get("actual_low_margin", "")),
        "actual_teacher_disagreement": lambda row: str(row.get("actual_teacher_disagreement", "")),
    }
    for config in configs:
        fired = teacher_fired_rows(rows, config)
        for group_name, getter in groupers.items():
            grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in fired:
                grouped[getter(row)].append(row)
            for key, group in sorted(grouped.items()):
                gains = [safe_float(row.get("actual_delta_candidate_vs_baseline")) for row in group]
                losses = [max(0.0, -gain) for gain in gains]
                output.append(
                    {
                        "config_id": config.config_id,
                        "bucket_type": group_name,
                        "bucket": key,
                        "override_count": len(group),
                        "avg_gain_on_override": float(np.mean(gains)) if gains else 0.0,
                        "false_positive_count": sum(1 for gain in gains if gain < 0.0),
                        "p95_loss": percentile(losses, 95),
                    }
                )
    return output


def teacher_failure_rows(rows: list[dict[str, str]], configs: list[HuTurn2Stage8RuntimeConfig], limit: int = 30) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for config in configs:
        for row in teacher_fired_rows(rows, config):
            gain = safe_float(row.get("actual_delta_candidate_vs_baseline"))
            if gain >= 0.0:
                continue
            failures.append(
                {
                    "config_id": config.config_id,
                    "state_index": int(safe_float(row.get("state_index"), -1)),
                    "seat": row.get("seat"),
                    "bucket_group": row.get("bucket_group"),
                    "predicted_delta": safe_float(row.get("predicted_delta_vs_baseline")),
                    "gate_probability": safe_float(row.get("gate_probability")),
                    "reference_margin_raw": safe_float(row.get("reference_margin_raw")),
                    "actual_delta_candidate_vs_baseline": gain,
                    "loss": -gain,
                    "teacher_best_margin": safe_float(row.get("teacher_best_margin")),
                    "candidate_loss": safe_float(row.get("candidate_loss")),
                    "failure_label": classify_failure(row),
                }
            )
    failures.sort(key=lambda item: float(item["loss"]), reverse=True)
    return failures[:limit]


def classify_failure(row: dict[str, str]) -> str:
    gain = safe_float(row.get("actual_delta_candidate_vs_baseline"))
    if abs(gain) <= max(0.10, safe_float(row.get("SE_delta")) * 2.0):
        return "low_margin_noise"
    if safe_float(row.get("predicted_delta_vs_baseline")) < 3.0:
        return "false_positive_low_delta"
    if safe_float(row.get("gate_probability")) >= 0.90:
        return "gate_overconfident"
    if safe_float(row.get("reference_margin_raw")) < 0.10:
        return "reference_margin_misleading"
    if row.get("actual_low_margin") == "True":
        return "low_margin_noise"
    return "other"


def aggregate_seed_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[str(row["config_id"])].append(row)
    output = []
    for config_id, rows in grouped.items():
        total_games = sum(int(row["paired_seeds"]) for row in rows)
        mean = sum(float(row["ev_per_hand"]) * int(row["paired_seeds"]) for row in rows) / max(total_games, 1)
        seed_values = [float(row["ev_per_hand"]) for row in rows]
        stderr = float(np.std(seed_values, ddof=1) / math.sqrt(len(seed_values))) if len(seed_values) > 1 else float(rows[0].get("paired_score_std_error", 0.0))
        decisions = sum(int(row["decision_count"]) for row in rows)
        overrides = sum(int(row["override_count"]) for row in rows)
        output.append(
            {
                "config_id": config_id,
                "hu_turn2_min_margin": rows[0]["hu_turn2_min_margin"],
                "hu_turn2_reference_min_margin": rows[0]["hu_turn2_reference_min_margin"],
                "hu_turn2_gate_threshold": rows[0]["hu_turn2_gate_threshold"],
                "hu_turn2_min_model_score": rows[0].get("hu_turn2_min_model_score", ""),
                "hu_turn2_allowed_seats": rows[0].get("hu_turn2_allowed_seats", ""),
                "paired_seeds": total_games,
                "hands": sum(int(row["hands"]) for row in rows),
                "aggregate_ev_per_hand": mean,
                "std_error_seed_means": stderr,
                "ci95_low_seed_means": mean - 1.96 * stderr,
                "ci95_high_seed_means": mean + 1.96 * stderr,
                "seed_count": len(rows),
                "decision_count": decisions,
                "override_count": overrides,
                "runtime_override_rate": overrides / max(decisions, 1),
                "paired_seed_wins": sum(int(row["paired_seed_wins"]) for row in rows),
                "paired_seed_losses": sum(int(row["paired_seed_losses"]) for row in rows),
                "paired_seed_ties": sum(int(row["paired_seed_ties"]) for row in rows),
            }
        )
    return output


def runtime_position_breakdown(position_scores: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in position_scores:
        grouped_scores[(str(row["config_id"]), str(row["seat"]))].append(float(row["score"]))
    grouped_decisions: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in decisions:
        grouped_decisions[(str(row["config_id"]), str(row.get("seat", "unknown")))].append(row)
    output = []
    for key, scores in sorted(grouped_scores.items()):
        group_decisions = grouped_decisions.get(key, [])
        output.append(
            {
                "config_id": key[0],
                "seat": key[1],
                "hands": len(scores),
                "ev_per_hand": float(np.mean(scores)) if scores else 0.0,
                "override_count": sum(1 for row in group_decisions if row.get("override_fired")),
                "decision_count": len(group_decisions),
                "override_rate": sum(1 for row in group_decisions if row.get("override_fired")) / max(len(group_decisions), 1),
            }
        )
    return output


def runtime_decision_buckets(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for config_id, group in _group_by(decisions, "config_id").items():
        for field in ("no_override_reason", "override_fired"):
            for label, rows in _group_by(group, field).items():
                output.append(
                    {
                        "config_id": config_id,
                        "bucket_type": field,
                        "bucket": label,
                        "decision_count": len(rows),
                        "override_count": sum(1 for row in rows if row.get("override_fired")),
                    }
                )
    return output


def _group_by(rows: Iterable[dict[str, Any]], field: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field, "unknown"))].append(row)
    return grouped


def choose_recommended(rows: list[dict[str, Any]]) -> dict[str, Any]:
    viable = [
        row
        for row in rows
        if float(row.get("aggregate_ev_per_hand", 0.0)) > 0.0
        and float(row.get("avg_gain_on_override", 0.0)) > 0.0
        and float(row.get("false_positive_override_rate", 1.0)) <= 0.10
        and float(row.get("p95_loss", 999.0)) <= 5.0
    ]
    if not viable:
        viable = rows[:]
    return max(
        viable,
        key=lambda row: (
            float(row.get("ci95_low_seed_means", -999.0)),
            -float(row.get("false_positive_override_rate", 999.0)),
            -float(row.get("p95_loss", 999.0)),
            float(row.get("aggregate_ev_per_hand", -999.0)),
        ),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_summary(path: Path, grid_rows: list[dict[str, Any]], recommended: dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "# HU Turn2 Stage8 20k Seat-Swap Validation",
        "",
        "## Scope",
        "",
        "- Candidate: T2 Stage8 broad 20k selective override.",
        "- Baseline/default: current HU T2 policy.",
        "- T3 continuation fixed: Stage7_candidate_A m5_r10.",
        "- This is validation only. No 50k teacher, T1, or production deployment was started.",
        "",
        "## Results",
        "",
        "| config | EV/hand | CI low | CI high | runtime override | teacher override | teacher avg gain | false positive | p95 loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(grid_rows, key=lambda item: float(item.get("aggregate_ev_per_hand", 0.0)), reverse=True):
        lines.append(
            "| {config_id} | {ev:.4f} | {low:.4f} | {high:.4f} | {ror:.4f} | {tor:.4f} | {gain:.4f} | {fp:.4f} | {p95:.4f} |".format(
                config_id=row["config_id"],
                ev=float(row.get("aggregate_ev_per_hand", 0.0)),
                low=float(row.get("ci95_low_seed_means", 0.0)),
                high=float(row.get("ci95_high_seed_means", 0.0)),
                ror=float(row.get("runtime_override_rate", 0.0)),
                tor=float(row.get("teacher_override_rate", 0.0)),
                gain=float(row.get("avg_gain_on_override", 0.0)),
                fp=float(row.get("false_positive_override_rate", 0.0)),
                p95=float(row.get("p95_loss", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Recommended Runtime",
            "",
            f"- config: `{recommended['config_id']}`",
            f"- `hu_turn2_min_margin = {recommended['hu_turn2_min_margin']}`",
            f"- `hu_turn2_reference_min_margin = {recommended['hu_turn2_reference_min_margin']}`",
            f"- `hu_turn2_gate_threshold = {recommended['hu_turn2_gate_threshold']}`",
            "",
            "## Decision",
            "",
        ]
    )
    if float(recommended.get("aggregate_ev_per_hand", 0.0)) > 0.0 and float(recommended.get("ci95_low_seed_means", -999.0)) > -0.05:
        lines.append("Positive validation trend. Keep as a T2 selective-override candidate and expand validation before production.")
    else:
        lines.append("Not enough production evidence yet. Do not move to production or T1 from this result.")
    lines.extend(
        [
            "",
            "## Inputs",
            "",
            f"- games_per_seed: `{args.games_per_seed}`",
            f"- seeds: `{args.seeds}`",
            f"- seed_stride: `{args.seed_stride}`",
            f"- model: `{args.hu_turn2_stage8_model}`",
            f"- calibration_values: `{args.calibration_values}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.games_per_seed <= 0:
        raise SystemExit("--games-per-seed must be positive")
    if args.prediction_threads > 0:
        os.environ.setdefault("OMP_NUM_THREADS", str(args.prediction_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.prediction_threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.prediction_threads))

    started_at = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configs = parse_configs(args.configs)
    seeds = parse_seeds(args.seeds)
    parts = load_parts(args)
    calibration_rows = read_csv_rows(args.calibration_values)

    seed_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    position_scores: list[dict[str, Any]] = []
    with _prediction_thread_context(args.prediction_threads):
        for config in configs:
            for seed in seeds:
                summary, decisions, scores = evaluate_config_seed(
                    config=config,
                    seed=seed,
                    seed_stride=args.seed_stride,
                    games=args.games_per_seed,
                    parts=parts,
                    opening_lookahead_samples=args.opening_lookahead_samples,
                    progress_every=args.progress_every,
                )
                seed_rows.append(summary)
                decision_rows.extend(decisions)
                position_scores.extend(scores)

    aggregate_rows = aggregate_seed_rows(seed_rows)
    teacher_by_config = {config.config_id: teacher_summary(calibration_rows, config) for config in configs}
    grid_rows: list[dict[str, Any]] = []
    for row in aggregate_rows:
        combined = dict(row)
        combined.update(teacher_by_config.get(str(row["config_id"]), {}))
        grid_rows.append(combined)
    grid_rows.sort(key=lambda item: float(item.get("aggregate_ev_per_hand", 0.0)), reverse=True)
    recommended = choose_recommended(grid_rows)
    recommended_payload = {
        "schema": "hu_turn2_stage8_20k_recommended_runtime_v1",
        "candidate_model": str(args.hu_turn2_stage8_model),
        "t3_continuation_policy": "Stage7_candidate_A_m5_r10",
        "full_replacement": False,
        **recommended,
    }

    write_csv(args.output_dir / "hu_turn2_stage8_20k_threshold_grid_results.csv", grid_rows)
    write_csv(args.output_dir / "hu_turn2_stage8_20k_seed_breakdown.csv", seed_rows)
    write_csv(args.output_dir / "hu_turn2_stage8_20k_position_breakdown.csv", runtime_position_breakdown(position_scores, decision_rows))
    write_csv(args.output_dir / "hu_turn2_stage8_20k_bucket_breakdown.csv", teacher_bucket_breakdown(calibration_rows, configs) + runtime_decision_buckets(decision_rows))
    write_jsonl(args.output_dir / "hu_turn2_stage8_20k_override_failures_top30.jsonl", teacher_failure_rows(calibration_rows, configs))
    if args.write_decision_log:
        write_jsonl(args.output_dir / "hu_turn2_stage8_20k_runtime_decisions.jsonl", decision_rows)
    (args.output_dir / "hu_turn2_stage8_20k_recommended_runtime.json").write_text(
        json.dumps(recommended_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_summary(args.output_dir / "hu_turn2_stage8_20k_seat_swap_summary.md", grid_rows, recommended, args)
    go = bool(
        float(recommended.get("aggregate_ev_per_hand", 0.0)) > 0.0
        and float(recommended.get("avg_gain_on_override", 0.0)) > 0.0
        and float(recommended.get("false_positive_override_rate", 1.0)) <= 0.10
    )
    (args.output_dir / "hu_turn2_stage8_20k_go_nogo.md").write_text(
        "\n".join(
            [
                "# HU T2 Stage8 20k Go / No-Go",
                "",
                f"- seat_swap_positive: `{float(recommended.get('aggregate_ev_per_hand', 0.0)) > 0.0}`",
                f"- teacher_avg_gain_positive: `{float(recommended.get('avg_gain_on_override', 0.0)) > 0.0}`",
                f"- false_positive_acceptable: `{float(recommended.get('false_positive_override_rate', 1.0)) <= 0.10}`",
                f"- next_stage_candidate_go: `{go}`",
                "- production: `No-Go`",
                "- 50k_teacher: `defer until expanded validation / selected refinement decision`",
                "- T1: `defer`",
                "",
                f"Elapsed seconds: `{time.time() - started_at:.2f}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"recommended": recommended_payload, "elapsed_seconds": time.time() - started_at}, indent=2))


if __name__ == "__main__":
    main()
