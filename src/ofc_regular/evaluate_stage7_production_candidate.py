"""Production-style evaluation for Stage7 HU Turn3 selective overrides."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np

from .action_space import Action, generate_actions, generate_turn_actions
from .ai_profiles import (
    DEFAULT_OPENING_MODEL,
    DEFAULT_TURN1_MODEL,
    DEFAULT_TURN2_MODEL,
    DEFAULT_TURN3_MODEL,
)
from .cards import create_deck
from .evaluate_matchups import board_score_to_json, board_to_json, classify_board
from .hu_turn3_model import hu_policy_sample, load_hu_action_value_model
from .play_ai import _prediction_thread_context
from .policy import (
    RegularAiPolicy,
    action_to_json,
    choose_opening_action_by_turn1_lookahead,
    policy_sample,
)
from .rules import check_fl_entry
from .state import Board
from .teacher import DEFAULT_FL_EV, evaluate_turn_actions, terminal_score
from .turn3_model import load_action_value_model
from .visibility import HuDiscardTracker


DEFAULT_STAGE7_MODEL = Path("models/hu_turn3_stage7_reference_override_cached_rank_wide.pt")
DEFAULT_STAGE3_MODEL = Path(
    "models/hu_turn3_stage3_mc32_500k_plus_m8_12_f128_100k_w2_cached_rank_wide.pt"
)
DEFAULT_TEACHER = Path(
    "outputs/gcp_runs/"
    "regular-hu-t3-stage7-stage3-override-mc2048-m4-25-50k-2h-20260608-001/"
    "hu_turn3_selfplay_merged.jsonl"
)

STAGE3_PRODUCTION_MARGIN = 10.0
CORE_CONFIGS = (
    (3.0, 8.0),
    (3.0, 10.0),
    (3.0, 12.0),
    (3.0, 15.0),
    (4.0, 8.0),
    (4.0, 10.0),
    (4.0, 12.0),
    (4.0, 15.0),
    (5.0, 8.0),
    (5.0, 10.0),
    (5.0, 12.0),
    (5.0, 15.0),
    (6.0, 8.0),
    (6.0, 10.0),
    (6.0, 12.0),
    (6.0, 15.0),
    (4.0, 20.0),
    (5.0, 20.0),
)


@dataclass(frozen=True)
class RuntimeConfig:
    hu_margin: float
    reference_margin: float

    @property
    def config_id(self) -> str:
        return f"m{self.hu_margin:g}_r{self.reference_margin:g}"


@dataclass
class ModelParts:
    opening: Any
    turn1: Any
    turn2: Any
    turn3: Any
    stage3: Any
    stage7: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--stage7-model", type=Path, default=DEFAULT_STAGE7_MODEL)
    parser.add_argument("--stage3-model", type=Path, default=DEFAULT_STAGE3_MODEL)
    parser.add_argument("--opening-model", type=Path, default=DEFAULT_OPENING_MODEL)
    parser.add_argument("--turn1-model", type=Path, default=DEFAULT_TURN1_MODEL)
    parser.add_argument("--turn2-model", type=Path, default=DEFAULT_TURN2_MODEL)
    parser.add_argument("--turn3-model", type=Path, default=DEFAULT_TURN3_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evals"))
    parser.add_argument("--games-per-seed", type=int, default=200)
    parser.add_argument(
        "--seeds",
        default="2026060904,2026063501,2026068201",
        help="Comma-separated base seeds. Each base seed evaluates games-per-seed paired seeds.",
    )
    parser.add_argument(
        "--configs",
        default="",
        help="Comma-separated configs like 4/10,4/12. Empty runs the default grid.",
    )
    parser.add_argument("--teacher-max-samples", type=int)
    parser.add_argument("--teacher-batch-samples", type=int, default=512)
    parser.add_argument("--prediction-threads", type=int, default=1)
    parser.add_argument("--opening-lookahead-samples", type=int, default=64)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--use-existing-matchup-json", action="append", default=[])
    return parser.parse_args()


def parse_seeds(value: str) -> list[int]:
    seeds = []
    for part in value.split(","):
        part = part.strip()
        if part:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def parse_configs(value: str) -> list[RuntimeConfig]:
    if not value.strip():
        return [RuntimeConfig(margin, reference) for margin, reference in CORE_CONFIGS]
    configs: list[RuntimeConfig] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "/" not in part:
            raise ValueError(f"config must be margin/reference: {part}")
        margin, reference = part.split("/", 1)
        configs.append(RuntimeConfig(float(margin), float(reference)))
    if not configs:
        raise ValueError("at least one config is required")
    return configs


def load_parts(args: argparse.Namespace) -> ModelParts:
    return ModelParts(
        opening=load_action_value_model(args.opening_model),
        turn1=load_action_value_model(args.turn1_model),
        turn2=load_action_value_model(args.turn2_model),
        turn3=load_action_value_model(args.turn3_model),
        stage3=load_hu_action_value_model(args.stage3_model),
        stage7=load_hu_action_value_model(args.stage7_model),
    )


class Stage7TracingPolicy(RegularAiPolicy):
    def __init__(
        self,
        *,
        config: RuntimeConfig,
        parts: ModelParts,
        decision_records: list[dict[str, Any]],
        context: dict[str, Any],
        seed: int,
        seat: str,
        opening_lookahead_samples: int,
    ) -> None:
        super().__init__(
            opening_model=parts.opening,
            turn1_model=parts.turn1,
            turn2_model=parts.turn2,
            turn3_model=parts.turn3,
            hu_turn3_model=parts.stage7,
            hu_turn3_reference_model=parts.stage3,
            hu_turn3_min_margin=config.hu_margin,
            hu_turn3_reference_min_margin=config.reference_margin,
            seat=seat,
            seed=seed,
            opening_lookahead_samples=opening_lookahead_samples,
        )
        self.config = config
        self.parts = parts
        self.decision_records = decision_records
        self.context = context

    def choose_action(
        self,
        board: Board,
        dealt_cards: Iterable[str],
        *,
        dead_cards: Iterable[str] = (),
        opponent_board: Board | None = None,
    ) -> Action:
        dealt = tuple(dealt_cards)
        dead = tuple(dead_cards)
        if board.card_count() == 9 and opponent_board is not None:
            actions = generate_turn_actions(board, dealt)
            if actions:
                to_act_order = "second" if opponent_board.card_count() > board.card_count() else "first"
                sample = hu_policy_sample(
                    board,
                    dealt,
                    actions,
                    opponent_board=opponent_board,
                    dead_cards=dead,
                    seat=self.seat,
                    to_act_order=to_act_order,
                )
                fallback_sample = policy_sample(board, dealt, actions)
                fallback_index = int(self.parts.turn3.choose_action_index(fallback_sample))
                stage3_predictions = self.parts.stage3.predict_sample(sample)
                stage7_predictions = self.parts.stage7.predict_sample(sample)
                stage3_best_index = int(np.argmax(stage3_predictions))
                stage7_best_index = int(np.argmax(stage7_predictions))
                stage3_margin = float(stage3_predictions[stage3_best_index] - stage3_predictions[fallback_index])
                stage3_production_index = (
                    stage3_best_index
                    if stage3_margin >= STAGE3_PRODUCTION_MARGIN
                    else fallback_index
                )
                config_baseline_index = (
                    stage3_best_index
                    if stage3_margin >= self.config.reference_margin
                    else fallback_index
                )
                stage7_margin = float(
                    stage7_predictions[stage7_best_index]
                    - stage7_predictions[config_baseline_index]
                )
                selected_index = (
                    stage7_best_index
                    if stage7_margin >= self.config.hu_margin
                    else config_baseline_index
                )
                record = {
                    **self.context,
                    "config_id": self.config.config_id,
                    "hu_turn3_min_margin": self.config.hu_margin,
                    "hu_turn3_reference_min_margin": self.config.reference_margin,
                    "seat": self.seat,
                    "to_act_order": to_act_order,
                    "board": board_to_json(board),
                    "opponent_board": board_to_json(opponent_board),
                    "dead_cards": list(dead),
                    "dealt": list(dealt),
                    "legal_actions": len(actions),
                    "fallback_index": fallback_index,
                    "stage3_best_index": stage3_best_index,
                    "stage3_production_index": stage3_production_index,
                    "stage7_best_index": stage7_best_index,
                    "selected_index": selected_index,
                    "stage3_margin": stage3_margin,
                    "reference_margin": stage3_margin,
                    "stage7_predicted_margin": stage7_margin,
                    "override": selected_index != stage3_production_index,
                    "stage3_action": action_to_json(board, actions[stage3_production_index]),
                    "stage7_action": action_to_json(board, actions[selected_index]),
                    "fallback_action": action_to_json(board, actions[fallback_index]),
                }
                self.decision_records.append(record)
                return actions[selected_index]
        return super().choose_action(
            board,
            dealt,
            dead_cards=dead,
            opponent_board=opponent_board,
        )


def make_stage3_policy(parts: ModelParts, *, seed: int, seat: str, opening_lookahead_samples: int) -> RegularAiPolicy:
    return RegularAiPolicy(
        opening_model=parts.opening,
        turn1_model=parts.turn1,
        turn2_model=parts.turn2,
        turn3_model=parts.turn3,
        hu_turn3_model=parts.stage3,
        hu_turn3_min_margin=STAGE3_PRODUCTION_MARGIN,
        seat=seat,
        seed=seed,
        opening_lookahead_samples=opening_lookahead_samples,
    )


def trace_hand_with_records(
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
    discards = HuDiscardTracker()

    for player in (0, 1):
        dealt = deck[cursor : cursor + 5]
        cursor += 5
        action = policies[player].choose_action(
            boards[player],
            dealt,
            dead_cards=(*boards[1 - player].all_cards(), *discards.own_discards(player)),
            opponent_board=boards[1 - player],
        )
        boards[player] = boards[player].place(action.placements)
        discards.record(player, action.discards)

    for _round_index in range(1, 5):
        for player in (0, 1):
            dealt = deck[cursor : cursor + 3]
            cursor += 3
            action = policies[player].choose_action(
                boards[player],
                dealt,
                dead_cards=(*boards[1 - player].all_cards(), *discards.own_discards(player)),
                opponent_board=boards[1 - player],
            )
            boards[player] = boards[player].place(action.placements)
            discards.record(player, action.discards)

    score_p0, board_score_p0 = terminal_score(boards[0], boards[1], fl_ev=DEFAULT_FL_EV)
    _reverse_score, board_score_p1 = terminal_score(boards[1], boards[0], fl_ev=DEFAULT_FL_EV)
    return {
        "seed": seed,
        "profiles": {"p0": profile_p0, "p1": profile_p1},
        "score_p0": float(score_p0),
        "final": {"p0": board_to_json(boards[0]), "p1": board_to_json(boards[1])},
        "board_scores": {
            "p0": board_score_to_json(board_score_p0),
            "p1": board_score_to_json(board_score_p1),
        },
        "classifications": {
            "p0": classify_board(boards[0]),
            "p1": classify_board(boards[1]),
        },
    }


def summarize_values(values: list[float], prefix: str = "") -> dict[str, float]:
    if not values:
        return {
            f"{prefix}avg_score_per_hand_for_a": 0.0,
            f"{prefix}std_error": 0.0,
            f"{prefix}ci95_low": 0.0,
            f"{prefix}ci95_high": 0.0,
        }
    average = sum(values) / len(values)
    variance = sum((value - average) ** 2 for value in values) / max(len(values) - 1, 1)
    stderr = math.sqrt(variance / len(values))
    return {
        f"{prefix}avg_score_per_hand_for_a": average,
        f"{prefix}std_error": stderr,
        f"{prefix}ci95_low": average - 1.96 * stderr,
        f"{prefix}ci95_high": average + 1.96 * stderr,
    }


def evaluate_seat_swap_config(
    *,
    config: RuntimeConfig,
    seed: int,
    games: int,
    parts: ModelParts,
    opening_lookahead_samples: int,
    progress_every: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paired_scores: list[float] = []
    decisions: list[dict[str, Any]] = []
    wins = losses = ties = 0
    started_at = time.time()

    for index in range(games):
        hand_seed = seed + index
        context_ab = {
            "hand_seed": hand_seed,
            "paired_index": index,
            "seat_swap": "ab",
            "player": 0,
        }
        hand_ab = trace_hand_with_records(
            seed=hand_seed,
            profile_p0="stage7_candidate_A",
            profile_p1="stage3_baseline",
            policy_p0=Stage7TracingPolicy(
                config=config,
                parts=parts,
                decision_records=decisions,
                context=context_ab,
                seed=hand_seed * 4,
                seat="first",
                opening_lookahead_samples=opening_lookahead_samples,
            ),
            policy_p1=make_stage3_policy(
                parts,
                seed=hand_seed * 4 + 1,
                seat="second",
                opening_lookahead_samples=opening_lookahead_samples,
            ),
        )
        context_ba = {
            "hand_seed": hand_seed,
            "paired_index": index,
            "seat_swap": "ba",
            "player": 1,
        }
        hand_ba = trace_hand_with_records(
            seed=hand_seed,
            profile_p0="stage3_baseline",
            profile_p1="stage7_candidate_A",
            policy_p0=make_stage3_policy(
                parts,
                seed=hand_seed * 4 + 2,
                seat="first",
                opening_lookahead_samples=opening_lookahead_samples,
            ),
            policy_p1=Stage7TracingPolicy(
                config=config,
                parts=parts,
                decision_records=decisions,
                context=context_ba,
                seed=hand_seed * 4 + 3,
                seat="second",
                opening_lookahead_samples=opening_lookahead_samples,
            ),
        )
        paired_score = (float(hand_ab["score_p0"]) - float(hand_ba["score_p0"])) / 2.0
        paired_scores.append(paired_score)
        if paired_score > 0:
            wins += 1
        elif paired_score < 0:
            losses += 1
        else:
            ties += 1
        for record in decisions[-2:]:
            if record.get("hand_seed") == hand_seed:
                record["paired_score_for_a"] = paired_score
        if progress_every > 0 and (index + 1) % progress_every == 0:
            print(
                json.dumps(
                    {
                        "event": "seat_swap_progress",
                        "config_id": config.config_id,
                        "seed": seed,
                        "paired_seeds": index + 1,
                        **summarize_values(paired_scores),
                        "elapsed_seconds": time.time() - started_at,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )

    override_count = sum(1 for item in decisions if item["override"])
    summary = {
        "config_id": config.config_id,
        "hu_turn3_min_margin": config.hu_margin,
        "hu_turn3_reference_min_margin": config.reference_margin,
        "seed": seed,
        "paired_seeds": games,
        "hands": games * 2,
        **summarize_values(paired_scores),
        "paired_seed_wins": wins,
        "paired_seed_losses": losses,
        "paired_seed_ties": ties,
        "decision_count": len(decisions),
        "override_count": override_count,
        "override_rate": override_count / len(decisions) if decisions else 0.0,
        "elapsed_seconds": time.time() - started_at,
    }
    return summary, decisions


def action_key(action: dict[str, Any]) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    placements = tuple((str(card), str(row)) for card, row in action.get("placements", ()))
    discards = tuple(str(card) for card in action.get("discards", ()))
    return placements, discards


def bucket(value: float, buckets: list[tuple[str, float | None, float | None]]) -> str:
    for label, low, high in buckets:
        if low is not None and value < low:
            continue
        if high is not None and value >= high:
            continue
        return label
    return "other"


STAGE3_MARGIN_BUCKETS = [
    ("4-8", 4.0, 8.0),
    ("8-10", 8.0, 10.0),
    ("10-15", 10.0, 15.0),
    ("15-20", 15.0, 20.0),
    ("20-25", 20.0, 25.0),
    ("25+", 25.0, None),
    ("below4", None, 4.0),
]
STAGE7_MARGIN_BUCKETS = [
    ("0-2", 0.0, 2.0),
    ("2-4", 2.0, 4.0),
    ("4-6", 4.0, 6.0),
    ("6-10", 6.0, 10.0),
    ("10+", 10.0, None),
    ("negative", None, 0.0),
]
REFERENCE_MARGIN_BUCKETS = [
    ("0-5", 0.0, 5.0),
    ("5-10", 5.0, 10.0),
    ("10-15", 10.0, 15.0),
    ("15-20", 15.0, 20.0),
    ("20+", 20.0, None),
    ("negative", None, 0.0),
]
DELTA_BUCKETS = [
    ("negative", None, 0.0),
    ("0-0.05", 0.0, 0.05),
    ("0.05-0.25", 0.05, 0.25),
    ("0.25-1.0", 0.25, 1.0),
    ("1.0+", 1.0, None),
]


def action_field(action: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(action.get(key, default))
    except (TypeError, ValueError):
        return default


def fl_qualifies(action: dict[str, Any]) -> bool:
    board = action.get("next_board", {})
    top = board.get("top", ())
    if len(top) != 3:
        return False
    return bool(check_fl_entry(top).qualifies)


def relation_tags(stage3_action: dict[str, Any], stage7_action: dict[str, Any], delta: float) -> list[str]:
    tags = []
    future = max(action_field(stage3_action, "future_count"), action_field(stage7_action, "future_count"))
    if future > 0:
        stage3_non_bust = action_field(stage3_action, "non_bust_future_count")
        stage7_non_bust = action_field(stage7_action, "non_bust_future_count")
        if stage7_non_bust < stage3_non_bust:
            tags.append("foul_related")
    if fl_qualifies(stage3_action) != fl_qualifies(stage7_action):
        tags.append("fl_related")
    if abs(delta) >= 1.0:
        tags.append("royalty_or_large_ev_swing")
    if abs(delta) >= 3.0:
        tags.append("scoop_or_large_terminal_swing")
    if not tags:
        tags.append("none")
    return tags


def classify_failure(
    *,
    delta: float,
    stage7_margin: float,
    stage3_margin: float,
    stage3_action: dict[str, Any],
    stage7_action: dict[str, Any],
) -> str:
    if abs(delta) < 0.05:
        return "near_tie_noise"
    if stage7_margin < 4.0:
        return "false_positive_low_margin"
    if stage3_margin < STAGE3_PRODUCTION_MARGIN:
        return "reference_margin_miscalibrated"
    future = max(action_field(stage3_action, "future_count"), action_field(stage7_action, "future_count"))
    if future > 0 and action_field(stage7_action, "non_bust_future_count") < action_field(stage3_action, "non_bust_future_count"):
        return "foul_risk_underestimated"
    if fl_qualifies(stage7_action) and not fl_qualifies(stage3_action):
        return "FL_overvalued"
    if abs(delta) >= 1.0:
        return "royalty_overvalued"
    return "other"


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_teacher_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    overrides = [row for row in rows if row["override"]]
    deltas = [float(row["delta_vs_stage3"]) for row in overrides]
    positive = [delta for delta in deltas if delta > 0]
    negative = [-delta for delta in deltas if delta < 0]
    return {
        "teacher_samples": len(rows),
        "teacher_override_count": len(overrides),
        "teacher_override_rate": len(overrides) / len(rows) if rows else 0.0,
        "avg_gain_on_override": sum(deltas) / len(deltas) if deltas else 0.0,
        "median_gain_on_override": float(median(deltas)) if deltas else 0.0,
        "false_positive_override_rate": len(negative) / len(overrides) if overrides else 0.0,
        "avg_false_positive_cost": sum(negative) / len(negative) if negative else 0.0,
        "p90_loss": percentile(negative, 90),
        "p95_loss": percentile(negative, 95),
        "p99_loss": percentile(negative, 99),
        "max_loss": max(negative) if negative else 0.0,
        "avg_gain_when_positive": sum(positive) / len(positive) if positive else 0.0,
        "avg_loss_when_negative": sum(negative) / len(negative) if negative else 0.0,
        "count_positive_override": len(positive),
        "count_negative_override": len(negative),
    }


def build_teacher_prediction_records(
    *,
    samples: list[dict[str, Any]],
    parts: ModelParts,
    batch_samples: int,
) -> list[dict[str, Any]]:
    from .hu_turn3_model import sample_to_matrix

    records: list[dict[str, Any]] = []
    for start in range(0, len(samples), batch_samples):
        batch = samples[start : start + batch_samples]
        feature_blocks: list[np.ndarray] = []
        block_sizes: list[int] = []
        for sample in batch:
            features, _targets = sample_to_matrix(sample)
            feature_blocks.append(features)
            block_sizes.append(features.shape[0])
        if not feature_blocks:
            continue
        features = np.vstack(feature_blocks)
        stage3_predictions_all = parts.stage3.predict_matrix(features)
        stage7_predictions_all = parts.stage7.predict_matrix(features)
        cursor = 0
        for offset, (sample, block_size) in enumerate(zip(batch, block_sizes)):
            actions = sample["actions"]
            scores = [float(action.get("score", 0.0)) for action in actions]
            stage3_predictions = stage3_predictions_all[cursor : cursor + block_size]
            stage7_predictions = stage7_predictions_all[cursor : cursor + block_size]
            cursor += block_size

            ref = sample.get("reference_actions", {})
            fallback_index = None
            if isinstance(ref, dict):
                baseline = ref.get("baseline")
                if isinstance(baseline, dict) and baseline.get("sorted_index") is not None:
                    fallback_index = int(baseline["sorted_index"])
            if fallback_index is None:
                fallback_index = int(np.argmax([action.get("self_model_score", 0.0) for action in actions]))

            stage3_best_index = int(np.argmax(stage3_predictions))
            stage7_best_index = int(np.argmax(stage7_predictions))
            stage3_margin = float(stage3_predictions[stage3_best_index] - stage3_predictions[fallback_index])
            stage3_production_index = (
                stage3_best_index
                if stage3_margin >= STAGE3_PRODUCTION_MARGIN
                else fallback_index
            )
            records.append(
                {
                    "sample": sample,
                    "sample_id": sample.get("sample_id", start + offset),
                    "actions": actions,
                    "scores": scores,
                    "fallback_index": fallback_index,
                    "stage3_best_index": stage3_best_index,
                    "stage3_margin": stage3_margin,
                    "stage3_production_index": stage3_production_index,
                    "stage7_best_index": stage7_best_index,
                    "stage7_best_prediction": float(stage7_predictions[stage7_best_index]),
                    "stage7_prediction_at_stage3_best": float(stage7_predictions[stage3_best_index]),
                    "stage7_prediction_at_fallback": float(stage7_predictions[fallback_index]),
                }
            )
    return records


def teacher_config_rows(
    *,
    config: RuntimeConfig,
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for record in records:
            sample = record["sample"]
            actions = record["actions"]
            scores = record["scores"]
            fallback_index = int(record["fallback_index"])
            stage3_best_index = int(record["stage3_best_index"])
            stage3_margin = float(record["stage3_margin"])
            stage3_production_index = int(record["stage3_production_index"])
            stage7_best_index = int(record["stage7_best_index"])
            config_baseline_index = (
                stage3_best_index if stage3_margin >= config.reference_margin else fallback_index
            )
            if config_baseline_index == stage3_best_index:
                stage7_baseline_prediction = float(record["stage7_prediction_at_stage3_best"])
            else:
                stage7_baseline_prediction = float(record["stage7_prediction_at_fallback"])
            stage7_margin = float(record["stage7_best_prediction"] - stage7_baseline_prediction)
            selected_index = (
                stage7_best_index if stage7_margin >= config.hu_margin else config_baseline_index
            )
            delta = float(scores[selected_index] - scores[stage3_production_index])
            override = selected_index != stage3_production_index
            stage3_action = actions[stage3_production_index]
            stage7_action = actions[selected_index]
            tags = relation_tags(stage3_action, stage7_action, delta)
            row = {
                "config_id": config.config_id,
                "hu_turn3_min_margin": config.hu_margin,
                "hu_turn3_reference_min_margin": config.reference_margin,
                "sample_id": record["sample_id"],
                "override": override,
                "stage3_margin": stage3_margin,
                "reference_margin": stage3_margin,
                "stage7_predicted_margin": stage7_margin,
                "stage3_ev": scores[stage3_production_index],
                "stage7_ev": scores[selected_index],
                "delta_vs_stage3": delta,
                "stage3_margin_bucket": bucket(stage3_margin, STAGE3_MARGIN_BUCKETS),
                "stage7_predicted_margin_bucket": bucket(stage7_margin, STAGE7_MARGIN_BUCKETS),
                "reference_margin_bucket": bucket(stage3_margin, REFERENCE_MARGIN_BUCKETS),
                "actual_delta_vs_stage3_bucket": bucket(delta, DELTA_BUCKETS),
                "override_bucket": "override" if override else "no_override",
                "relation_tags": tags,
            }
            rows.append(row)
            if override and delta < 0:
                failures.append(
                    {
                        "config_id": config.config_id,
                        "sample_id": record["sample_id"],
                        "seed": sample.get("seed"),
                        "state": {
                            "seat": sample.get("seat"),
                            "to_act_order": sample.get("to_act_order"),
                            "board": sample.get("board"),
                            "opponent_board": sample.get("opponent_board"),
                            "dead_cards": sample.get("dead_cards"),
                            "dealt": sample.get("dealt"),
                        },
                        "hero_board": sample.get("board"),
                        "opponent_board": sample.get("opponent_board"),
                        "dead_cards": sample.get("dead_cards"),
                        "cards_to_place": sample.get("dealt"),
                        "legal_actions_count": len(actions),
                        "stage3_action": stage3_action,
                        "stage7_action": stage7_action,
                        "fallback_action": actions[fallback_index],
                        "predicted_margin": stage7_margin,
                        "reference_margin": stage3_margin,
                        "gate_probability": None,
                        "teacher_ev_per_action": [
                            {
                                "index": index,
                                "score": float(action.get("score", 0.0)),
                                "placements": action.get("placements", []),
                                "discards": action.get("discards", []),
                            }
                            for index, action in enumerate(actions)
                        ],
                        "stage3_ev": scores[stage3_production_index],
                        "stage7_ev": scores[selected_index],
                        "delta_vs_stage3": delta,
                        "realized_delta": None,
                        "foul_risk": {
                            "stage3_non_bust_future_count": action_field(stage3_action, "non_bust_future_count"),
                            "stage7_non_bust_future_count": action_field(stage7_action, "non_bust_future_count"),
                            "future_count": action_field(stage7_action, "future_count"),
                        },
                        "royalty_swing": delta,
                        "fl_related": "fl_related" in tags,
                        "reason_label": classify_failure(
                            delta=delta,
                            stage7_margin=stage7_margin,
                            stage3_margin=stage3_margin,
                            stage3_action=stage3_action,
                            stage7_action=stage7_action,
                        ),
                    }
                )
    summary = {
        "config_id": config.config_id,
        "hu_turn3_min_margin": config.hu_margin,
        "hu_turn3_reference_min_margin": config.reference_margin,
        **summarize_teacher_rows(rows),
    }
    return summary, rows, failures


def bucket_breakdown(config: RuntimeConfig, teacher_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    bucket_specs = [
        ("stage3_margin", "stage3_margin_bucket"),
        ("stage7_predicted_margin", "stage7_predicted_margin_bucket"),
        ("reference_margin", "reference_margin_bucket"),
        ("actual_delta_vs_stage3", "actual_delta_vs_stage3_bucket"),
        ("override", "override_bucket"),
    ]
    for bucket_type, field in bucket_specs:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in teacher_rows:
            grouped[str(row[field])].append(row)
        for label, rows in sorted(grouped.items()):
            result.append({"config_id": config.config_id, "bucket_type": bucket_type, "bucket": label, **summarize_teacher_rows(rows)})

    relation_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in teacher_rows:
        for tag in row["relation_tags"]:
            relation_grouped[tag].append(row)
    for label, rows in sorted(relation_grouped.items()):
        result.append({"config_id": config.config_id, "bucket_type": "foul_fl_royalty_scoop", "bucket": label, **summarize_teacher_rows(rows)})
    return result


def read_teacher_samples(path: Path, max_samples: int | None) -> list[dict[str, Any]]:
    samples = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            samples.append(json.loads(line))
            if max_samples is not None and len(samples) >= max_samples:
                break
    return samples


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_config_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["config_id"])].append(row)
    aggregated = {}
    for config_id, group in grouped.items():
        paired_scores: list[float] = []
        for row in group:
            paired_scores.extend([float(row["avg_score_per_hand_for_a"])] * int(row["paired_seeds"]))
        # Seed-level means are not enough to reconstruct per-hand CI; use weighted
        # variance over seed means as a conservative summary.
        total_games = sum(int(row["paired_seeds"]) for row in group)
        weighted_avg = (
            sum(float(row["avg_score_per_hand_for_a"]) * int(row["paired_seeds"]) for row in group)
            / max(total_games, 1)
        )
        seed_values = [float(row["avg_score_per_hand_for_a"]) for row in group]
        seed_stderr = (
            math.sqrt(sum((value - weighted_avg) ** 2 for value in seed_values) / max(len(seed_values) - 1, 1) / len(seed_values))
            if len(seed_values) > 1
            else float(group[0].get("std_error", 0.0))
        )
        decisions = sum(int(row["decision_count"]) for row in group)
        overrides = sum(int(row["override_count"]) for row in group)
        aggregated[config_id] = {
            "config_id": config_id,
            "hu_turn3_min_margin": group[0]["hu_turn3_min_margin"],
            "hu_turn3_reference_min_margin": group[0]["hu_turn3_reference_min_margin"],
            "hands": sum(int(row["hands"]) for row in group),
            "paired_seeds": total_games,
            "aggregate_ev_per_hand": weighted_avg,
            "aggregate_std_error_seed_means": seed_stderr,
            "aggregate_ci95_low_seed_means": weighted_avg - 1.96 * seed_stderr,
            "aggregate_ci95_high_seed_means": weighted_avg + 1.96 * seed_stderr,
            "seed_count": len(group),
            "decision_count": decisions,
            "override_count": overrides,
            "production_override_rate": overrides / decisions if decisions else 0.0,
            "paired_seed_wins": sum(int(row["paired_seed_wins"]) for row in group),
            "paired_seed_losses": sum(int(row["paired_seed_losses"]) for row in group),
            "paired_seed_ties": sum(int(row["paired_seed_ties"]) for row in group),
        }
    return aggregated


def merge_existing_matchups(path_values: list[str]) -> list[dict[str, Any]]:
    rows = []
    for value in path_values:
        path = Path(value)
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        model = data.get("models_a", {})
        margin = float(model.get("hu_turn3_min_margin", 0.0))
        reference = float(model.get("hu_turn3_reference_min_margin", 0.0))
        rows.append(
            {
                "config_id": RuntimeConfig(margin, reference).config_id,
                "hu_turn3_min_margin": margin,
                "hu_turn3_reference_min_margin": reference,
                "seed": int(data["seed"]),
                "paired_seeds": int(data["paired_seeds"]),
                "hands": int(data["hands"]),
                "avg_score_per_hand_for_a": float(data["avg_score_per_hand_for_a"]),
                "std_error": float(data["std_error"]),
                "ci95_low": float(data["ci95_low"]),
                "ci95_high": float(data["ci95_high"]),
                "paired_seed_wins": int(data["paired_seed_wins"]),
                "paired_seed_losses": int(data["paired_seed_losses"]),
                "paired_seed_ties": int(data["paired_seed_ties"]),
                "decision_count": 0,
                "override_count": 0,
                "override_rate": 0.0,
                "source": str(path),
            }
        )
    return rows


def choose_recommendation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    viable = [
        row
        for row in rows
        if row.get("aggregate_ev_per_hand", 0.0) > 0.0
        and row.get("avg_gain_on_override", 0.0) > 0.0
        and row.get("p95_loss", 999.0) <= 5.0
    ]
    if not viable:
        viable = rows[:]
    return max(
        viable,
        key=lambda row: (
            float(row.get("aggregate_ci95_low_seed_means", -999.0)),
            -float(row.get("false_positive_override_rate", 999.0)),
            -float(row.get("p95_loss", 999.0)),
            float(row.get("aggregate_ev_per_hand", -999.0)),
        ),
    )


def write_summary(
    *,
    path: Path,
    args: argparse.Namespace,
    configs: list[RuntimeConfig],
    seeds: list[int],
    grid_rows: list[dict[str, Any]],
    recommended: dict[str, Any],
    command: str,
) -> None:
    best_mean = max(grid_rows, key=lambda row: float(row.get("aggregate_ev_per_hand", -999.0)))
    lines = [
        "# Stage7 Candidate A Production Evaluation",
        "",
        "## Commands",
        "",
        f"```powershell\n{command}\n```",
        "",
        "## Artifacts",
        "",
        f"- Teacher data: `{args.teacher}`",
        f"- Stage7 model: `{args.stage7_model}`",
        f"- Stage3 model: `{args.stage3_model}`",
        f"- Output dir: `{args.output_dir}`",
        "",
        "## Comparison",
        "",
        "- Baseline: Stage3 production policy, HU T3 margin 10.0.",
        "- Candidate: Stage7 selective override with Stage3 reference fallback.",
        "",
        "## Seeds",
        "",
        ", ".join(str(seed) for seed in seeds),
        "",
        "## Threshold Grid",
        "",
        ", ".join(config.config_id for config in configs),
        "",
        "## Config Results",
        "",
        "| config | EV/hand | seed CI low | seed CI high | production override rate | teacher override rate | false positive | p95 loss | avg gain override |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(grid_rows, key=lambda item: float(item.get("aggregate_ev_per_hand", 0.0)), reverse=True):
        lines.append(
            "| {config_id} | {ev:.4f} | {low:.4f} | {high:.4f} | {por:.4f} | {tor:.4f} | {fp:.4f} | {p95:.4f} | {gain:.4f} |".format(
                config_id=row["config_id"],
                ev=float(row.get("aggregate_ev_per_hand", 0.0)),
                low=float(row.get("aggregate_ci95_low_seed_means", 0.0)),
                high=float(row.get("aggregate_ci95_high_seed_means", 0.0)),
                por=float(row.get("production_override_rate", 0.0)),
                tor=float(row.get("teacher_override_rate", 0.0)),
                fp=float(row.get("false_positive_override_rate", 0.0)),
                p95=float(row.get("p95_loss", 0.0)),
                gain=float(row.get("avg_gain_on_override", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Best Mean Config",
            "",
            f"- `{best_mean['config_id']}`: EV/hand `{float(best_mean.get('aggregate_ev_per_hand', 0.0)):.4f}`",
            "",
            "## Best Conservative Config",
            "",
            f"- `{recommended['config_id']}`",
            f"- EV/hand: `{float(recommended.get('aggregate_ev_per_hand', 0.0)):.4f}`",
            f"- seed-CI low: `{float(recommended.get('aggregate_ci95_low_seed_means', 0.0)):.4f}`",
            f"- false positive override rate: `{float(recommended.get('false_positive_override_rate', 0.0)):.4f}`",
            f"- p95 loss: `{float(recommended.get('p95_loss', 0.0)):.4f}`",
            "",
            "## Recommended Runtime",
            "",
            f"- `hu_turn3_min_margin = {recommended['hu_turn3_min_margin']}`",
            f"- `hu_turn3_reference_min_margin = {recommended['hu_turn3_reference_min_margin']}`",
            "",
            "## Adoption Decision",
            "",
        ]
    )
    if float(recommended.get("aggregate_ev_per_hand", 0.0)) > 0.0 and float(recommended.get("avg_gain_on_override", 0.0)) > 0.0:
        lines.append("Stage7_candidate_A is acceptable as a conservative selective override, not as a full Stage3 replacement.")
    else:
        lines.append("Stage7_candidate_A should not be promoted yet; keep Stage3 as production baseline.")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `margin0.25` is excluded from production candidates because the existing check was negative.",
            "- False positive and tail metrics are teacher-EV based on the fixed MC2048 50k dataset.",
            "- Seat-swap EV is the primary production metric.",
            "",
            "## Next Improvements",
            "",
            "1. Tune threshold/gate before adding more data.",
            "2. Inspect the top false positive override failures.",
            "3. Add more targeted teacher data only for buckets that still show tail risk.",
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
    teacher_samples = read_teacher_samples(args.teacher, args.teacher_max_samples)
    if not teacher_samples:
        raise SystemExit("no teacher samples")

    seed_rows = merge_existing_matchups(args.use_existing_matchup_json)
    all_decision_rows: list[dict[str, Any]] = []
    with _prediction_thread_context(args.prediction_threads):
        for config in configs:
            existing_keys = {
                (row["config_id"], row["seed"], row["paired_seeds"])
                for row in seed_rows
            }
            for seed in seeds:
                if (config.config_id, seed, args.games_per_seed) in existing_keys:
                    continue
                summary, decisions = evaluate_seat_swap_config(
                    config=config,
                    seed=seed,
                    games=args.games_per_seed,
                    parts=parts,
                    opening_lookahead_samples=args.opening_lookahead_samples,
                    progress_every=args.progress_every,
                )
                seed_rows.append(summary)
                all_decision_rows.extend(decisions)

    teacher_summaries: dict[str, dict[str, Any]] = {}
    bucket_rows: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    with _prediction_thread_context(args.prediction_threads):
        teacher_records = build_teacher_prediction_records(
            samples=teacher_samples,
            parts=parts,
            batch_samples=args.teacher_batch_samples,
        )
        for config in configs:
            summary, rows, failures = teacher_config_rows(
                config=config,
                records=teacher_records,
            )
            teacher_summaries[config.config_id] = summary
            bucket_rows.extend(bucket_breakdown(config, rows))
            all_failures.extend(failures)
            print(
                json.dumps(
                    {
                        "event": "teacher_config_progress",
                        "config_id": config.config_id,
                        "teacher_samples": summary["teacher_samples"],
                        "teacher_override_rate": summary["teacher_override_rate"],
                        "false_positive_override_rate": summary["false_positive_override_rate"],
                        "elapsed_seconds": time.time() - started_at,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                flush=True,
            )

    aggregated = aggregate_config_rows(seed_rows)
    grid_rows: list[dict[str, Any]] = []
    for config_id, row in aggregated.items():
        combined = dict(row)
        combined.update(teacher_summaries.get(config_id, {}))
        grid_rows.append(combined)
    grid_rows.sort(key=lambda row: (float(row.get("aggregate_ev_per_hand", 0.0)), -float(row.get("p95_loss", 0.0))), reverse=True)
    recommended = choose_recommendation(grid_rows)

    failure_top = sorted(all_failures, key=lambda item: float(item["delta_vs_stage3"]))[:30]
    recommended_runtime = {
        "candidate": "Stage7_candidate_A",
        "recommended": {
            "hu_turn3_model": str(args.stage7_model),
            "hu_turn3_reference_model": str(args.stage3_model),
            "hu_turn3_min_margin": recommended["hu_turn3_min_margin"],
            "hu_turn3_reference_min_margin": recommended["hu_turn3_reference_min_margin"],
        },
        "best_mean_config": max(grid_rows, key=lambda row: float(row.get("aggregate_ev_per_hand", -999.0))),
        "best_conservative_config": recommended,
        "margin0_25": "reject_for_production",
        "elapsed_seconds": time.time() - started_at,
    }

    grid_path = args.output_dir / "stage7_candidate_A_threshold_grid_results.csv"
    seed_path = args.output_dir / "stage7_candidate_A_seed_breakdown.csv"
    bucket_path = args.output_dir / "stage7_candidate_A_bucket_breakdown.csv"
    failure_path = args.output_dir / "stage7_candidate_A_override_failures_top30.jsonl"
    runtime_path = args.output_dir / "stage7_candidate_A_recommended_runtime.json"
    summary_path = args.output_dir / "stage7_candidate_A_production_eval_summary.md"
    decisions_path = args.output_dir / "stage7_candidate_A_production_override_decisions.jsonl"

    write_csv(grid_path, grid_rows)
    write_csv(seed_path, seed_rows)
    write_csv(bucket_path, bucket_rows)
    with failure_path.open("w", encoding="utf-8") as handle:
        for failure in failure_top:
            handle.write(json.dumps(failure, ensure_ascii=False, separators=(",", ":")) + "\n")
    with decisions_path.open("w", encoding="utf-8") as handle:
        for decision in all_decision_rows:
            handle.write(json.dumps(decision, ensure_ascii=False, separators=(",", ":")) + "\n")
    runtime_path.write_text(json.dumps(recommended_runtime, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    command = "python -m ofc_regular.evaluate_stage7_production_candidate " + " ".join(os.sys.argv[1:])
    write_summary(
        path=summary_path,
        args=args,
        configs=configs,
        seeds=seeds,
        grid_rows=grid_rows,
        recommended=recommended,
        command=command,
    )
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "grid_csv": str(grid_path),
                "seed_csv": str(seed_path),
                "bucket_csv": str(bucket_path),
                "failures_jsonl": str(failure_path),
                "recommended_runtime": str(runtime_path),
                "decisions_jsonl": str(decisions_path),
                "recommended": recommended_runtime["recommended"],
                "elapsed_seconds": time.time() - started_at,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
