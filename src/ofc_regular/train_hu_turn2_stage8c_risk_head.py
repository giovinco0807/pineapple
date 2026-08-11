"""Train a HU T2 Stage8c whole-game risk head from replay-ready TopK rows.

This is a research/smoke trainer. It intentionally keeps the whole-game risk
target separate from local EV/safe-LCB labels:

- positive rows: ``recommended_training_use == whole_game_risk_only``
- negative rows: ``recommended_training_use == whole_game_non_loss_control``
- excluded rows: ``local_ev_hard_negative`` and anything not replay-ready

The resulting model is not a production runtime policy by itself.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .analyze_hu_turn2_stage8b_risk_targets import (
    DEFAULT_TARGET_NAME,
    load_rows,
    recommended_use,
    replay_ready,
    safe_float,
    safe_int,
    target_paths,
    truthy,
)
from .cards import ALL_CARDS, RANK_VALUE, card_rank, card_suit
from .evaluator import (
    HAND_FLUSH,
    HAND_FULL_HOUSE,
    HAND_HIGH,
    HAND_PAIR,
    HAND_QUADS,
    HAND_STRAIGHT,
    HAND_STRAIGHT_FLUSH,
    HAND_TRIPS,
    HAND_TWO_PAIR,
    evaluate_3_card,
    evaluate_5_card,
    get_bottom_royalty,
    get_middle_royalty,
    get_top_royalty,
)
from .rules import check_fl_entry
from .state import Board
from .teacher import evaluate_two_turn_actions
from .hu_turn3_model import HU_FEATURE_DIM, sample_to_matrix
from .train_torch_action_value import parse_hidden_layers, select_device
from .turn3_model import _build_torch_mlp, _import_torch

RISK_LABEL_BY_USE = {
    "whole_game_non_loss_control": 0,
    "whole_game_risk_only": 1,
}
TOPK_CONFIRM_FIRE_LABEL_BY_USE = {
    "topk_confirm_realized_positive": 1,
    "topk_confirm_realized_loss": 0,
    "topk_confirm_rejected": 0,
    "topk_confirm_fire_selector_rejected": 0,
    "topk_confirm_topk_empty": 0,
    "topk_confirm_replay_positive": 1,
    "topk_confirm_replay_negative": 0,
}
TARGET_MODE_WHOLE_GAME_RISK = "whole_game_risk"
TARGET_MODE_LOCAL_EV_NEGATIVE = "local_ev_negative"
TARGET_MODE_TOPK_CONFIRM_FIRE = "topk_confirm_fire"
TARGET_MODE_REALIZED_WHOLE_GAME_LOSS = "realized_whole_game_loss"
TARGET_MODES = (
    TARGET_MODE_WHOLE_GAME_RISK,
    TARGET_MODE_LOCAL_EV_NEGATIVE,
    TARGET_MODE_TOPK_CONFIRM_FIRE,
    TARGET_MODE_REALIZED_WHOLE_GAME_LOSS,
)
SPLIT_NAME_TO_ID = {"train": 0, "val": 1, "test": 2}
SPLIT_ID_TO_NAME = {value: key for key, value in SPLIT_NAME_TO_ID.items()}
SPLIT_MODE_ROW_STRATIFIED = "row_stratified"
SPLIT_MODE_SOURCE_LOG = "source_log"
SPLIT_MODE_SOURCE_SEED = "source_seed"
SPLIT_MODES = (SPLIT_MODE_ROW_STRATIFIED, SPLIT_MODE_SOURCE_LOG, SPLIT_MODE_SOURCE_SEED)
MAX_ROWS_MODE_FIRST = "first"
MAX_ROWS_MODE_STRATIFIED = "stratified"
MAX_ROWS_MODES = (MAX_ROWS_MODE_FIRST, MAX_ROWS_MODE_STRATIFIED)
FEATURE_MODE_HU_ONLY = "hu_only"
FEATURE_MODE_HU_PLUS_RUNTIME_META = "hu_plus_runtime_meta"
FEATURE_MODE_RUNTIME_META_ONLY = "runtime_meta_only"
FEATURE_MODE_PRECONFIRM_META_ONLY = "preconfirm_meta_only"
FEATURE_MODE_HU_PLUS_RUNTIME_TAIL_META = "hu_plus_runtime_tail_meta"
FEATURE_MODE_RUNTIME_TAIL_META_ONLY = "runtime_tail_meta_only"
FEATURE_MODE_HU_DELTA_ONLY = "hu_delta_only"
FEATURE_MODE_HU_DELTA_PLUS_PRECONFIRM_META = "hu_delta_plus_preconfirm_meta"
FEATURE_MODE_HU_DELTA_PLUS_RUNTIME_TAIL_META = "hu_delta_plus_runtime_tail_meta"
FEATURE_MODE_OPPORTUNITY_PROXY_ONLY = "opportunity_proxy_only"
FEATURE_MODE_OPPORTUNITY_PROXY_PLUS_PRECONFIRM_META = "opportunity_proxy_plus_preconfirm_meta"
FEATURE_MODE_OPPORTUNITY_PROXY_PLUS_RUNTIME_TAIL_META = "opportunity_proxy_plus_runtime_tail_meta"
FEATURE_MODE_LOOKAHEAD_PROXY_ONLY = "lookahead_proxy_only"
FEATURE_MODE_LOOKAHEAD_PROXY_PLUS_PRECONFIRM_META = "lookahead_proxy_plus_preconfirm_meta"
FEATURE_MODE_LOOKAHEAD_PROXY_PLUS_RUNTIME_TAIL_META = "lookahead_proxy_plus_runtime_tail_meta"
FEATURE_MODE_OPPORTUNITY_LOOKAHEAD_PROXY_PLUS_PRECONFIRM_META = "opportunity_lookahead_proxy_plus_preconfirm_meta"
FEATURE_MODE_OPPORTUNITY_LOOKAHEAD_PROXY_PLUS_RUNTIME_TAIL_META = "opportunity_lookahead_proxy_plus_runtime_tail_meta"
FEATURE_MODES = (
    FEATURE_MODE_HU_ONLY,
    FEATURE_MODE_HU_PLUS_RUNTIME_META,
    FEATURE_MODE_RUNTIME_META_ONLY,
    FEATURE_MODE_PRECONFIRM_META_ONLY,
    FEATURE_MODE_HU_PLUS_RUNTIME_TAIL_META,
    FEATURE_MODE_RUNTIME_TAIL_META_ONLY,
    FEATURE_MODE_HU_DELTA_ONLY,
    FEATURE_MODE_HU_DELTA_PLUS_PRECONFIRM_META,
    FEATURE_MODE_HU_DELTA_PLUS_RUNTIME_TAIL_META,
    FEATURE_MODE_OPPORTUNITY_PROXY_ONLY,
    FEATURE_MODE_OPPORTUNITY_PROXY_PLUS_PRECONFIRM_META,
    FEATURE_MODE_OPPORTUNITY_PROXY_PLUS_RUNTIME_TAIL_META,
    FEATURE_MODE_LOOKAHEAD_PROXY_ONLY,
    FEATURE_MODE_LOOKAHEAD_PROXY_PLUS_PRECONFIRM_META,
    FEATURE_MODE_LOOKAHEAD_PROXY_PLUS_RUNTIME_TAIL_META,
    FEATURE_MODE_OPPORTUNITY_LOOKAHEAD_PROXY_PLUS_PRECONFIRM_META,
    FEATURE_MODE_OPPORTUNITY_LOOKAHEAD_PROXY_PLUS_RUNTIME_TAIL_META,
)
POS_WEIGHT_MODE_AUTO = "auto"
POS_WEIGHT_MODE_NONE = "none"
POS_WEIGHT_MODE_VALUE = "value"
POS_WEIGHT_MODES = (POS_WEIGHT_MODE_AUTO, POS_WEIGHT_MODE_NONE, POS_WEIGHT_MODE_VALUE)
DEFAULT_THRESHOLDS = (0.50, 0.70, 0.80, 0.90)
DEFAULT_TOPK_METRIC_COUNTS = (1, 3, 5, 10, 20, 50, 100)
RUNTIME_META_FEATURE_NAMES = (
    "meta_predicted_delta",
    "meta_abs_predicted_delta",
    "meta_gate_probability",
    "meta_logit_gate_probability",
    "meta_confirm_delta",
    "meta_confirm_delta_se",
    "meta_confirm_delta_z",
    "meta_confirm_minus_predicted_delta",
    "meta_candidate_ev_rank",
    "meta_candidate_ev_rank_inv",
    "meta_candidate_ev_rank_le1",
    "meta_candidate_ev_rank_le3",
    "meta_seat_is_second",
)
PRECONFIRM_META_FEATURE_NAMES = (
    "preconfirm_predicted_delta",
    "preconfirm_abs_predicted_delta",
    "preconfirm_gate_probability",
    "preconfirm_logit_gate_probability",
    "preconfirm_candidate_ev_rank",
    "preconfirm_candidate_ev_rank_inv",
    "preconfirm_candidate_ev_rank_le1",
    "preconfirm_candidate_ev_rank_le3",
    "preconfirm_model_score",
    "preconfirm_topk_score",
    "preconfirm_top_k_log1p",
    "preconfirm_seat_is_second",
)
BASE_CONFIRM_TAIL_FEATURE_NAMES = (
    "confirm_tail_count_log1p",
    "confirm_tail_mean",
    "confirm_tail_se",
    "confirm_tail_std",
    "confirm_tail_min",
    "confirm_tail_p01",
    "confirm_tail_p05",
    "confirm_tail_p25",
    "confirm_tail_p50",
    "confirm_tail_p75",
    "confirm_tail_p95",
    "confirm_tail_p99",
    "confirm_tail_max",
    "confirm_tail_mean_minus_p01",
    "confirm_tail_mean_minus_p05",
    "confirm_tail_mean_minus_p25",
    "confirm_tail_p75_minus_mean",
    "confirm_tail_p95_minus_mean",
    "confirm_tail_p99_minus_mean",
    "confirm_tail_range",
    "confirm_tail_min_loss",
    "confirm_tail_p01_loss",
    "confirm_tail_p05_loss",
    "confirm_tail_p25_loss",
    "confirm_tail_min_z",
    "confirm_tail_p01_z",
    "confirm_tail_p05_z",
    "confirm_tail_p25_z",
)
CONFIRM_TAIL_RATE_FEATURE_NAMES = (
    "confirm_tail_lt0_rate",
    "confirm_tail_le_neg6_rate",
    "confirm_tail_le_neg12_rate",
    "confirm_tail_le_neg20_rate",
)
CONFIRM_COMPONENT_DELTA_NAMES = (
    "terminal_score",
    "royalty_delta",
    "fl_delta",
    "line_score_delta",
    "scoop_delta",
    "foul_delta",
)
CONFIRM_COMPONENT_TAIL_STAT_NAMES = (
    "mean",
    "min",
    "p05",
    "p25",
    "p50",
    "p75",
    "p95",
    "max",
    "lt0_rate",
    "le_neg6_rate",
    "le_neg12_rate",
    "le_neg20_rate",
    "min_loss",
    "p05_loss",
    "p25_loss",
    "range",
)
CONFIRM_COMPONENT_TAIL_FEATURE_NAMES = tuple(
    f"confirm_component_{component}_{stat}"
    for component in CONFIRM_COMPONENT_DELTA_NAMES
    for stat in CONFIRM_COMPONENT_TAIL_STAT_NAMES
)
CONFIRM_TAIL_FEATURE_NAMES = (
    *BASE_CONFIRM_TAIL_FEATURE_NAMES,
    *CONFIRM_TAIL_RATE_FEATURE_NAMES,
    *CONFIRM_COMPONENT_TAIL_FEATURE_NAMES,
)
RUNTIME_TAIL_META_FEATURE_NAMES = RUNTIME_META_FEATURE_NAMES + CONFIRM_TAIL_FEATURE_NAMES
OPPORTUNITY_BOARD_FEATURE_NAMES = (
    "top_open",
    "middle_open",
    "bottom_open",
    "top_current_category",
    "top_possible_category",
    "top_pair_rank",
    "top_pair_qqplus_now",
    "top_pair_qqplus_possible",
    "top_trips_possible",
    "top_fl_entry_now",
    "top_fl_possible",
    "top_royalty_now",
    "top_royalty_potential",
    "middle_current_category",
    "middle_possible_category",
    "middle_pair_count",
    "middle_max_multiplicity",
    "middle_max_suit_count",
    "middle_made_royalty",
    "middle_royalty_potential",
    "middle_flush_possible",
    "middle_straight_possible",
    "middle_full_house_possible",
    "middle_quads_possible",
    "middle_straight_flush_possible",
    "bottom_current_category",
    "bottom_possible_category",
    "bottom_pair_count",
    "bottom_max_multiplicity",
    "bottom_max_suit_count",
    "bottom_made_royalty",
    "bottom_royalty_potential",
    "bottom_flush_possible",
    "bottom_straight_possible",
    "bottom_full_house_possible",
    "bottom_quads_possible",
    "bottom_straight_flush_possible",
    "total_made_royalty",
    "total_royalty_potential",
    "total_royalty_opportunity_gap",
    "inevitable_top_over_middle_proxy",
    "inevitable_middle_over_bottom_proxy",
    "top_over_middle_complete",
    "middle_over_bottom_complete",
    "top_category_minus_middle_possible",
    "middle_category_minus_bottom_possible",
)
OPPORTUNITY_PROXY_FEATURE_NAMES = (
    tuple(f"opportunity_candidate_{name}" for name in OPPORTUNITY_BOARD_FEATURE_NAMES)
    + tuple(f"opportunity_baseline_{name}" for name in OPPORTUNITY_BOARD_FEATURE_NAMES)
    + tuple(f"opportunity_delta_{name}" for name in OPPORTUNITY_BOARD_FEATURE_NAMES)
)
LOOKAHEAD_SAMPLES = 4
LOOKAHEAD_FINAL_SAMPLES = 4
LOOKAHEAD_BOARD_FEATURE_NAMES = (
    "sample_count_log1p",
    "score_mean",
    "score_std",
    "score_min",
    "score_p05",
    "score_p50",
    "score_p95",
    "score_max",
    "non_bust_rate_mean",
    "non_bust_rate_min",
    "missing_sample_rate",
)
LOOKAHEAD_PROXY_FEATURE_NAMES = (
    tuple(f"lookahead_candidate_{name}" for name in LOOKAHEAD_BOARD_FEATURE_NAMES)
    + tuple(f"lookahead_baseline_{name}" for name in LOOKAHEAD_BOARD_FEATURE_NAMES)
    + tuple(f"lookahead_delta_{name}" for name in LOOKAHEAD_BOARD_FEATURE_NAMES)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection-dir",
        type=Path,
        action="append",
        default=[],
        help=f"Directory containing {DEFAULT_TARGET_NAME}. Can be repeated.",
    )
    parser.add_argument("--input-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/hu_turn2_stage8c_whole_game_risk_head_smoke.pt"),
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-layer-sizes", default="256,128")
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026064517)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument(
        "--split-mode",
        choices=SPLIT_MODES,
        default=SPLIT_MODE_ROW_STRATIFIED,
        help="row_stratified preserves the old row-level split; source_* keeps groups out of multiple splits.",
    )
    parser.add_argument(
        "--fixed-val-groups",
        default="",
        help="Comma-separated split groups forced into validation for non-row split modes.",
    )
    parser.add_argument(
        "--fixed-test-groups",
        default="",
        help="Comma-separated split groups forced into test for non-row split modes; remaining groups are train.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        action="append",
        default=None,
        help=(
            "Threshold to report in risk_head_threshold_metrics.csv. Can be repeated. "
            f"Defaults to {', '.join(str(value) for value in DEFAULT_THRESHOLDS)}."
        ),
    )
    parser.add_argument("--max-rows", type=int)
    parser.add_argument(
        "--max-rows-mode",
        choices=MAX_ROWS_MODES,
        default=MAX_ROWS_MODE_FIRST,
        help="first preserves the historical prefix slice; stratified round-robins target/seat/source groups.",
    )
    parser.add_argument("--max-rows-seed", type=int, default=2026064518)
    parser.add_argument(
        "--exclude-recommended-use",
        action="append",
        default=[],
        help=(
            "Drop rows whose recommended_training_use/recommended_use matches this value before max-row sampling. "
            "Can be repeated; useful for excluding topk_confirm_topk_empty controls."
        ),
    )
    parser.add_argument(
        "--target-mode",
        choices=TARGET_MODES,
        default=TARGET_MODE_WHOLE_GAME_RISK,
        help=(
            "whole_game_risk learns realized downstream loss risk; local_ev_negative "
            "learns local replay negative candidates vs local positive-LCB controls."
        ),
    )
    parser.add_argument(
        "--feature-mode",
        choices=FEATURE_MODES,
        default=FEATURE_MODE_HU_ONLY,
        help="Use HU board/action features only, or append runtime-available TopK/confirm metadata.",
    )
    parser.add_argument(
        "--pos-weight-mode",
        choices=POS_WEIGHT_MODES,
        default=POS_WEIGHT_MODE_AUTO,
        help=(
            "BCE positive-class weighting. auto uses negative/positive ratio, "
            "none uses 1.0, value uses --pos-weight-value."
        ),
    )
    parser.add_argument("--pos-weight-value", type=float, default=1.0)
    parser.add_argument(
        "--topk-replay-negative-weight",
        type=float,
        default=1.0,
        help="Extra row weight for topk_confirm_replay_negative rows when target-mode=topk_confirm_fire.",
    )
    parser.add_argument(
        "--topk-realized-loss-weight",
        type=float,
        default=1.0,
        help="Extra row weight for topk_confirm_realized_loss rows when target-mode=topk_confirm_fire.",
    )
    parser.add_argument(
        "--topk-replay-positive-weight",
        type=float,
        default=1.0,
        help="Extra row weight for topk_confirm_replay_positive rows when target-mode=topk_confirm_fire.",
    )
    return parser.parse_args()


def resolved_thresholds(values: Iterable[float] | None) -> list[float]:
    if values is None:
        return [float(value) for value in DEFAULT_THRESHOLDS]
    return [float(value) for value in values]


def parse_fixed_groups(raw: str) -> set[str]:
    return {part.strip() for part in str(raw or "").split(",") if part.strip()}


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
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


def exclude_recommended_use_rows(
    rows: Iterable[dict[str, Any]],
    excluded_uses: Iterable[str],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    excluded = {str(value).strip() for value in excluded_uses if str(value).strip()}
    if not excluded:
        return list(rows), Counter()
    kept: list[dict[str, Any]] = []
    removed: Counter[str] = Counter()
    for row in rows:
        use = str(recommended_use(row) or "")
        if use in excluded:
            removed[use] += 1
            continue
        kept.append(row)
    return kept, removed


def risk_row_to_sample(row: dict[str, Any], *, action_key: str = "candidate_action") -> dict[str, Any]:
    return {
        "schema": "hu_turn2_stage8c_risk_head_sample_v1",
        "rule_set": "regular",
        "phase": "hu_turn2_stage8c_topk_risk",
        "seat": str(row.get("seat") or "first"),
        "to_act_order": str(row.get("to_act_order") or row.get("seat") or "first"),
        "board": row["hero_board"],
        "opponent_board": row.get("opponent_board", {"top": [], "middle": [], "bottom": []}),
        "dead_cards": list(row.get("dead_cards", ())),
        "dealt": list(row.get("cards_to_place", ())),
        "actions": [row[action_key]],
    }


def target_label_for_row(row: dict[str, Any], target_mode: str) -> tuple[int, str] | None:
    if target_mode == TARGET_MODE_WHOLE_GAME_RISK:
        use = recommended_use(row)
        if use not in RISK_LABEL_BY_USE:
            return None
        return RISK_LABEL_BY_USE[use], use
    if target_mode == TARGET_MODE_LOCAL_EV_NEGATIVE:
        if not replay_ready(row):
            return None
        local_label = str(row.get("local_replay_label", "")).strip().lower()
        local_bucket = str(row.get("local_replay_bucket", "")).strip().lower()
        if local_label == "negative" or local_bucket == "local_negative":
            return 1, "local_ev_negative"
        if local_label == "positive" and local_bucket == "local_positive_lcb":
            return 0, "local_ev_positive_lcb_control"
        return None
    if target_mode == TARGET_MODE_TOPK_CONFIRM_FIRE:
        use = recommended_use(row)
        if use not in TOPK_CONFIRM_FIRE_LABEL_BY_USE:
            return None
        return TOPK_CONFIRM_FIRE_LABEL_BY_USE[use], use
    if target_mode == TARGET_MODE_REALIZED_WHOLE_GAME_LOSS:
        if row.get("realized_delta") in (None, ""):
            return None
        realized_delta = safe_float(row.get("realized_delta"))
        if realized_delta < 0.0:
            return 1, "realized_whole_game_loss"
        return 0, "realized_whole_game_non_loss"
    raise ValueError(f"unknown target_mode={target_mode!r}")


def target_label_names(target_mode: str) -> tuple[str, str]:
    if target_mode == TARGET_MODE_WHOLE_GAME_RISK:
        return "whole_game_risk_only", "whole_game_non_loss_control"
    if target_mode == TARGET_MODE_LOCAL_EV_NEGATIVE:
        return "local_ev_negative", "local_ev_positive_lcb_control"
    if target_mode == TARGET_MODE_TOPK_CONFIRM_FIRE:
        return "topk_confirm_realized_positive", "topk_confirm_non_fire_or_loss"
    if target_mode == TARGET_MODE_REALIZED_WHOLE_GAME_LOSS:
        return "realized_whole_game_loss", "realized_whole_game_non_loss"
    raise ValueError(f"unknown target_mode={target_mode!r}")


def topk_confirm_training_ready(row: dict[str, Any]) -> bool:
    required = ("hero_board", "opponent_board", "dead_cards", "cards_to_place", "baseline_action", "candidate_action")
    return all(row.get(field) not in (None, "", [], {}) for field in required)


def row_ready_for_target(row: dict[str, Any], target_mode: str) -> bool:
    if target_mode in {TARGET_MODE_TOPK_CONFIRM_FIRE, TARGET_MODE_REALIZED_WHOLE_GAME_LOSS}:
        return topk_confirm_training_ready(row)
    target = target_label_for_row(row, target_mode)
    if target_mode == TARGET_MODE_WHOLE_GAME_RISK and target is not None:
        _label, target_group = target
        if target_group == "whole_game_non_loss_control":
            return topk_confirm_training_ready(row)
    return replay_ready(row)


def max_rows_group_key(row: dict[str, Any], target_mode: str) -> tuple[str, str, str]:
    target = target_label_for_row(row, target_mode)
    target_group = target[1] if target is not None else str(recommended_use(row) or "unlabeled")
    return (
        target_group,
        str(row.get("seat") or "unknown"),
        str(row.get("candidate_source") or "unknown"),
    )


def limit_rows_for_training(
    rows: list[dict[str, Any]],
    *,
    max_rows: int | None,
    mode: str,
    seed: int,
    target_mode: str,
) -> list[dict[str, Any]]:
    if max_rows is None or len(rows) <= max_rows:
        return rows
    limit = max(0, int(max_rows))
    if mode == MAX_ROWS_MODE_FIRST:
        return rows[:limit]
    if mode != MAX_ROWS_MODE_STRATIFIED:
        raise ValueError(f"unknown max_rows_mode={mode!r}")
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[max_rows_group_key(row, target_mode)].append(index)
    rng = np.random.default_rng(seed)
    shuffled: dict[tuple[str, str, str], list[int]] = {}
    for key, indices in groups.items():
        values = list(indices)
        rng.shuffle(values)
        shuffled[key] = values
    selected: set[int] = set()
    while len(selected) < limit:
        changed = False
        for key in sorted(shuffled):
            values = shuffled[key]
            if not values:
                continue
            selected.add(values.pop())
            changed = True
            if len(selected) >= limit:
                break
        if not changed:
            break
    return [row for index, row in enumerate(rows) if index in selected]


def clipped_probability(value: Any) -> float:
    return min(max(safe_float(value), 1e-6), 1.0 - 1e-6)


def runtime_meta_feature_vector(row: dict[str, Any]) -> np.ndarray:
    predicted_delta = safe_float(row.get("predicted_delta"))
    gate_probability = clipped_probability(row.get("gate_probability"))
    confirm_delta = safe_float(row.get("confirm_delta"))
    confirm_delta_se = max(safe_float(row.get("confirm_delta_se")), 1e-6)
    candidate_ev_rank = float(max(1, min(safe_int(row.get("candidate_ev_rank"), 9999), 9999)))
    seat_is_second = 1.0 if str(row.get("seat") or "").lower() == "second" else 0.0
    values = [
        predicted_delta,
        abs(predicted_delta),
        gate_probability,
        math.log(gate_probability / (1.0 - gate_probability)),
        confirm_delta,
        confirm_delta_se,
        confirm_delta / confirm_delta_se,
        confirm_delta - predicted_delta,
        candidate_ev_rank,
        1.0 / candidate_ev_rank,
        1.0 if candidate_ev_rank <= 1.0 else 0.0,
        1.0 if candidate_ev_rank <= 3.0 else 0.0,
        seat_is_second,
    ]
    return np.asarray(values, dtype=np.float32)


def preconfirm_meta_feature_vector(row: dict[str, Any]) -> np.ndarray:
    predicted_delta = safe_float(row.get("predicted_delta"))
    gate_probability = clipped_probability(row.get("gate_probability"))
    candidate_ev_rank = float(max(1, min(safe_int(row.get("candidate_ev_rank"), 9999), 9999)))
    top_k = float(max(0, safe_int(row.get("top_k"), 0)))
    seat_is_second = 1.0 if str(row.get("seat") or "").lower() == "second" else 0.0
    values = [
        predicted_delta,
        abs(predicted_delta),
        gate_probability,
        math.log(gate_probability / (1.0 - gate_probability)),
        candidate_ev_rank,
        1.0 / candidate_ev_rank,
        1.0 if candidate_ev_rank <= 1.0 else 0.0,
        1.0 if candidate_ev_rank <= 3.0 else 0.0,
        safe_float(row.get("model_score")),
        safe_float(row.get("topk_score")),
        math.log1p(top_k),
        seat_is_second,
    ]
    return np.asarray(values, dtype=np.float32)


def _summary_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def confirm_tail_summary(row: dict[str, Any]) -> dict[str, Any]:
    for key in (
        "confirm_paired_delta_summary",
        "paired_future_delta_summary",
        "stage_a_paired_delta_summary",
    ):
        summary = _summary_dict(row.get(key))
        if summary:
            return summary
    return {}


def _summary_float(summary: dict[str, Any], key: str, fallback: float) -> float:
    if key in summary:
        return safe_float(summary.get(key), fallback)
    return fallback


def confirm_tail_feature_vector(row: dict[str, Any]) -> np.ndarray:
    summary = confirm_tail_summary(row)
    mean = _summary_float(summary, "mean", safe_float(row.get("confirm_delta")))
    se = max(
        _summary_float(
            summary,
            "paired_delta_standard_error",
            _summary_float(summary, "standard_error", safe_float(row.get("confirm_delta_se"))),
        ),
        1e-6,
    )
    std = max(_summary_float(summary, "std", _summary_float(summary, "standard_deviation", 0.0)), 0.0)
    count = max(_summary_float(summary, "count", 0.0), 0.0)
    min_value = _summary_float(summary, "min", mean)
    p01 = _summary_float(summary, "p01", min_value)
    p05 = _summary_float(summary, "p05", p01)
    p25 = _summary_float(summary, "p25", mean)
    p50 = _summary_float(summary, "p50", mean)
    p75 = _summary_float(summary, "p75", mean)
    p95 = _summary_float(summary, "p95", p75)
    p99 = _summary_float(summary, "p99", p95)
    max_value = _summary_float(summary, "max", mean)
    values = [
        math.log1p(count),
        mean,
        se,
        std,
        min_value,
        p01,
        p05,
        p25,
        p50,
        p75,
        p95,
        p99,
        max_value,
        mean - p01,
        mean - p05,
        mean - p25,
        p75 - mean,
        p95 - mean,
        p99 - mean,
        max_value - min_value,
        max(0.0, -min_value),
        max(0.0, -p01),
        max(0.0, -p05),
        max(0.0, -p25),
        min_value / se,
        p01 / se,
        p05 / se,
        p25 / se,
    ]
    values.extend(
        [
            _summary_float(summary, "lt0_rate", 0.0),
            _summary_float(summary, "le_neg6_rate", 0.0),
            _summary_float(summary, "le_neg12_rate", 0.0),
            _summary_float(summary, "le_neg20_rate", 0.0),
        ]
    )
    component_summaries = summary.get("component_delta_summaries")
    if not isinstance(component_summaries, dict):
        component_summaries = {}
    for component in CONFIRM_COMPONENT_DELTA_NAMES:
        component_summary = _summary_dict(component_summaries.get(component))
        component_mean = _summary_float(component_summary, "mean", 0.0)
        component_min = _summary_float(component_summary, "min", component_mean)
        component_p05 = _summary_float(component_summary, "p05", component_min)
        component_p25 = _summary_float(component_summary, "p25", component_mean)
        component_p50 = _summary_float(component_summary, "p50", component_mean)
        component_p75 = _summary_float(component_summary, "p75", component_mean)
        component_p95 = _summary_float(component_summary, "p95", component_p75)
        component_max = _summary_float(component_summary, "max", component_mean)
        values.extend(
            [
                component_mean,
                component_min,
                component_p05,
                component_p25,
                component_p50,
                component_p75,
                component_p95,
                component_max,
                _summary_float(component_summary, "lt0_rate", 0.0),
                _summary_float(component_summary, "le_neg6_rate", 0.0),
                _summary_float(component_summary, "le_neg12_rate", 0.0),
                _summary_float(component_summary, "le_neg20_rate", 0.0),
                max(0.0, -component_min),
                max(0.0, -component_p05),
                max(0.0, -component_p25),
                component_max - component_min,
            ]
        )
    return np.asarray(values, dtype=np.float32)


def runtime_tail_meta_feature_vector(row: dict[str, Any]) -> np.ndarray:
    return np.concatenate([runtime_meta_feature_vector(row), confirm_tail_feature_vector(row)]).astype(np.float32, copy=False)


def hu_action_feature_vector(row: dict[str, Any], *, action_key: str) -> np.ndarray:
    sample = risk_row_to_sample(row, action_key=action_key)
    matrix, _targets = sample_to_matrix(sample)
    return matrix[0].astype(np.float32, copy=False)


def hu_delta_feature_vector(row: dict[str, Any]) -> np.ndarray:
    candidate = hu_action_feature_vector(row, action_key="candidate_action")
    baseline = hu_action_feature_vector(row, action_key="baseline_action")
    return (candidate - baseline).astype(np.float32, copy=False)


def _action_after_board(row: dict[str, Any], action_key: str) -> Board:
    board_rows = row.get("hero_board") or {}
    board = Board.from_rows(
        top=board_rows.get("top", ()),
        middle=board_rows.get("middle", ()),
        bottom=board_rows.get("bottom", ()),
    )
    action = row.get(action_key) or {}
    placements = tuple((str(card), str(target_row)) for card, target_row in action.get("placements", ()))
    return board.place(placements)


def _remaining_cards_for_t2_row(row: dict[str, Any]) -> set[str]:
    used: set[str] = set()
    for key in ("hero_board", "opponent_board"):
        board = row.get(key) or {}
        for cards in board.values():
            used.update(str(card) for card in cards)
    used.update(str(card) for card in row.get("dead_cards", ()))
    used.update(str(card) for card in row.get("cards_to_place", ()))
    return set(ALL_CARDS) - used


def _rank_counter(cards: Iterable[str]) -> Counter[int]:
    return Counter(card_rank(card) for card in cards)


def _suit_counter(cards: Iterable[str]) -> Counter[str]:
    return Counter(card_suit(card) for card in cards)


def _partial_duplicate_category(cards: tuple[str, ...]) -> int:
    counts = sorted(_rank_counter(cards).values(), reverse=True)
    if not counts:
        return HAND_HIGH
    if counts[0] >= 4:
        return HAND_QUADS
    if counts[0] >= 3:
        return HAND_TRIPS
    if sum(1 for count in counts if count >= 2) >= 2:
        return HAND_TWO_PAIR
    if counts[0] >= 2:
        return HAND_PAIR
    return HAND_HIGH


def _straight_runs() -> tuple[tuple[int, ...], ...]:
    return ((14, 5, 4, 3, 2),) + tuple(tuple(range(high - 4, high + 1)) for high in range(6, 15))


def _rank_available(remaining: set[str]) -> Counter[int]:
    return Counter(card_rank(card) for card in remaining)


def _suit_available(remaining: set[str]) -> Counter[str]:
    return Counter(card_suit(card) for card in remaining)


def _can_add_rank(rank: int, need: int, open_slots: int, available_by_rank: Counter[int]) -> bool:
    return need <= open_slots and available_by_rank.get(rank, 0) >= need


def _pair_possible(cards: tuple[str, ...], open_slots: int, remaining: set[str], *, min_rank: int = 2) -> bool:
    counts = _rank_counter(cards)
    available = _rank_available(remaining)
    for rank in range(min_rank, 15):
        need = max(0, 2 - counts.get(rank, 0))
        if _can_add_rank(rank, need, open_slots, available):
            return True
    return False


def _trips_possible(cards: tuple[str, ...], open_slots: int, remaining: set[str]) -> bool:
    counts = _rank_counter(cards)
    available = _rank_available(remaining)
    for rank in range(2, 15):
        need = max(0, 3 - counts.get(rank, 0))
        if _can_add_rank(rank, need, open_slots, available):
            return True
    return False


def _quads_possible(cards: tuple[str, ...], open_slots: int, remaining: set[str]) -> bool:
    counts = _rank_counter(cards)
    available = _rank_available(remaining)
    for rank in range(2, 15):
        need = max(0, 4 - counts.get(rank, 0))
        if _can_add_rank(rank, need, open_slots, available):
            return True
    return False


def _two_pair_possible(cards: tuple[str, ...], open_slots: int, remaining: set[str]) -> bool:
    counts = _rank_counter(cards)
    available = _rank_available(remaining)
    for first in range(2, 15):
        first_need = max(0, 2 - counts.get(first, 0))
        if first_need > open_slots or available.get(first, 0) < first_need:
            continue
        for second in range(first + 1, 15):
            second_need = max(0, 2 - counts.get(second, 0))
            if first_need + second_need <= open_slots and available.get(second, 0) >= second_need:
                return True
    return False


def _full_house_possible(cards: tuple[str, ...], open_slots: int, remaining: set[str]) -> bool:
    counts = _rank_counter(cards)
    available = _rank_available(remaining)
    for trips_rank in range(2, 15):
        trips_need = max(0, 3 - counts.get(trips_rank, 0))
        if trips_need > open_slots or available.get(trips_rank, 0) < trips_need:
            continue
        for pair_rank in range(2, 15):
            if pair_rank == trips_rank:
                continue
            pair_need = max(0, 2 - counts.get(pair_rank, 0))
            if trips_need + pair_need <= open_slots and available.get(pair_rank, 0) >= pair_need:
                return True
    return False


def _flush_possible(cards: tuple[str, ...], open_slots: int, remaining: set[str]) -> bool:
    suits = _suit_counter(cards)
    available = _suit_available(remaining)
    for suit in "hdcs":
        need = max(0, 5 - suits.get(suit, 0))
        if need <= open_slots and available.get(suit, 0) >= need:
            return True
    return False


def _straight_possible(cards: tuple[str, ...], open_slots: int, remaining: set[str]) -> bool:
    ranks = set(card_rank(card) for card in cards)
    available = _rank_available(remaining)
    for run in _straight_runs():
        need = 0
        ok = True
        for rank in run:
            if rank in ranks:
                continue
            if available.get(rank, 0) <= 0:
                ok = False
                break
            need += 1
        if ok and need <= open_slots:
            return True
    return False


def _straight_flush_possible(cards: tuple[str, ...], open_slots: int, remaining: set[str]) -> bool:
    card_set = set(cards)
    for suit in "hdcs":
        for run in _straight_runs():
            missing = 0
            ok = True
            for rank in run:
                card = f"{'23456789TJQKA'[rank - 2]}{suit}"
                if card in card_set:
                    continue
                if card not in remaining:
                    ok = False
                    break
                missing += 1
            if ok and missing <= open_slots:
                return True
    return False


def _royal_straight_flush_possible(cards: tuple[str, ...], open_slots: int, remaining: set[str]) -> bool:
    card_set = set(cards)
    for suit in "hdcs":
        missing = 0
        ok = True
        for rank_char in "TJQKA":
            card = f"{rank_char}{suit}"
            if card in card_set:
                continue
            if card not in remaining:
                ok = False
                break
            missing += 1
        if ok and missing <= open_slots:
            return True
    return False


def _top_potential_royalty(cards: tuple[str, ...], open_slots: int, remaining: set[str]) -> float:
    counts = _rank_counter(cards)
    available = _rank_available(remaining)
    best = 0
    for rank in range(2, 15):
        trips_need = max(0, 3 - counts.get(rank, 0))
        if _can_add_rank(rank, trips_need, open_slots, available):
            best = max(best, 10 + rank - 2)
        pair_need = max(0, 2 - counts.get(rank, 0))
        if rank >= RANK_VALUE["6"] and _can_add_rank(rank, pair_need, open_slots, available):
            best = max(best, rank - 5)
    return float(best)


def _five_card_possible_category(cards: tuple[str, ...], open_slots: int, remaining: set[str]) -> int:
    if len(cards) == 5:
        return int(evaluate_5_card(cards)[0])
    if _straight_flush_possible(cards, open_slots, remaining):
        return HAND_STRAIGHT_FLUSH
    if _quads_possible(cards, open_slots, remaining):
        return HAND_QUADS
    if _full_house_possible(cards, open_slots, remaining):
        return HAND_FULL_HOUSE
    if _flush_possible(cards, open_slots, remaining):
        return HAND_FLUSH
    if _straight_possible(cards, open_slots, remaining):
        return HAND_STRAIGHT
    if _trips_possible(cards, open_slots, remaining):
        return HAND_TRIPS
    if _two_pair_possible(cards, open_slots, remaining):
        return HAND_TWO_PAIR
    if _pair_possible(cards, open_slots, remaining):
        return HAND_PAIR
    return HAND_HIGH


def _five_card_potential_royalty(
    cards: tuple[str, ...],
    open_slots: int,
    remaining: set[str],
    *,
    row_name: str,
) -> float:
    if len(cards) == 5:
        return float(get_middle_royalty(cards) if row_name == "middle" else get_bottom_royalty(cards))
    royal_sf = _royal_straight_flush_possible(cards, open_slots, remaining)
    sf = _straight_flush_possible(cards, open_slots, remaining)
    quads = _quads_possible(cards, open_slots, remaining)
    full_house = _full_house_possible(cards, open_slots, remaining)
    flush = _flush_possible(cards, open_slots, remaining)
    straight = _straight_possible(cards, open_slots, remaining)
    trips = _trips_possible(cards, open_slots, remaining)
    if row_name == "middle":
        if royal_sf:
            return 50.0
        if sf:
            return 30.0
        if quads:
            return 20.0
        if full_house:
            return 12.0
        if flush:
            return 8.0
        if straight:
            return 4.0
        if trips:
            return 2.0
        return 0.0
    if royal_sf:
        return 25.0
    if sf:
        return 15.0
    if quads:
        return 10.0
    if full_house:
        return 6.0
    if flush:
        return 4.0
    if straight:
        return 2.0
    return 0.0


def _five_row_summary(cards: tuple[str, ...], open_slots: int, remaining: set[str], *, row_name: str) -> dict[str, float]:
    rank_counts = _rank_counter(cards)
    suit_counts = _suit_counter(cards)
    made_category = float(evaluate_5_card(cards)[0] if len(cards) == 5 else _partial_duplicate_category(cards))
    made_royalty = 0.0
    if len(cards) == 5:
        made_royalty = float(get_middle_royalty(cards) if row_name == "middle" else get_bottom_royalty(cards))
    return {
        "current_category": made_category,
        "possible_category": float(_five_card_possible_category(cards, open_slots, remaining)),
        "pair_count": float(sum(1 for count in rank_counts.values() if count >= 2)),
        "max_multiplicity": float(max(rank_counts.values(), default=0)),
        "max_suit_count": float(max(suit_counts.values(), default=0)),
        "made_royalty": made_royalty,
        "royalty_potential": _five_card_potential_royalty(cards, open_slots, remaining, row_name=row_name),
        "flush_possible": float(_flush_possible(cards, open_slots, remaining)),
        "straight_possible": float(_straight_possible(cards, open_slots, remaining)),
        "full_house_possible": float(_full_house_possible(cards, open_slots, remaining)),
        "quads_possible": float(_quads_possible(cards, open_slots, remaining)),
        "straight_flush_possible": float(_straight_flush_possible(cards, open_slots, remaining)),
    }


def _top_row_summary(cards: tuple[str, ...], open_slots: int, remaining: set[str]) -> dict[str, float]:
    counts = _rank_counter(cards)
    current_category = float(evaluate_3_card(cards)[0] if len(cards) == 3 else _partial_duplicate_category(cards))
    pair_rank = max((rank for rank, count in counts.items() if count >= 2), default=0)
    pair_qqplus_now = float(pair_rank >= RANK_VALUE["Q"])
    pair_qqplus_possible = float(_pair_possible(cards, open_slots, remaining, min_rank=RANK_VALUE["Q"]))
    trips_possible = float(_trips_possible(cards, open_slots, remaining))
    fl_entry_now = float(check_fl_entry(cards).qualifies) if len(cards) == 3 else 0.0
    fl_possible = float(bool(pair_qqplus_possible) or bool(trips_possible))
    return {
        "current_category": current_category,
        "possible_category": HAND_TRIPS if trips_possible else (HAND_PAIR if _pair_possible(cards, open_slots, remaining) else HAND_HIGH),
        "pair_rank": float(pair_rank),
        "pair_qqplus_now": pair_qqplus_now,
        "pair_qqplus_possible": pair_qqplus_possible,
        "trips_possible": trips_possible,
        "fl_entry_now": fl_entry_now,
        "fl_possible": fl_possible,
        "royalty_now": float(get_top_royalty(cards)) if len(cards) == 3 else 0.0,
        "royalty_potential": _top_potential_royalty(cards, open_slots, remaining),
    }


def _board_opportunity_vector(board: Board, remaining: set[str]) -> np.ndarray:
    top = tuple(board.top)
    middle = tuple(board.middle)
    bottom = tuple(board.bottom)
    top_open = board.open_slots("top")
    middle_open = board.open_slots("middle")
    bottom_open = board.open_slots("bottom")
    top_summary = _top_row_summary(top, top_open, remaining)
    middle_summary = _five_row_summary(middle, middle_open, remaining, row_name="middle")
    bottom_summary = _five_row_summary(bottom, bottom_open, remaining, row_name="bottom")
    total_made_royalty = (
        top_summary["royalty_now"] + middle_summary["made_royalty"] + bottom_summary["made_royalty"]
    )
    total_royalty_potential = (
        top_summary["royalty_potential"] + middle_summary["royalty_potential"] + bottom_summary["royalty_potential"]
    )
    top_over_middle_complete = 0.0
    middle_over_bottom_complete = 0.0
    if len(top) == 3 and len(middle) == 5:
        top_over_middle_complete = float(evaluate_3_card(top) > evaluate_5_card(middle))
    if len(middle) == 5 and len(bottom) == 5:
        middle_over_bottom_complete = float(evaluate_5_card(middle) > evaluate_5_card(bottom))
    top_category_minus_middle_possible = top_summary["current_category"] - middle_summary["possible_category"]
    middle_category_minus_bottom_possible = middle_summary["current_category"] - bottom_summary["possible_category"]
    values = [
        float(top_open),
        float(middle_open),
        float(bottom_open),
        top_summary["current_category"],
        float(top_summary["possible_category"]),
        top_summary["pair_rank"],
        top_summary["pair_qqplus_now"],
        top_summary["pair_qqplus_possible"],
        top_summary["trips_possible"],
        top_summary["fl_entry_now"],
        top_summary["fl_possible"],
        top_summary["royalty_now"],
        top_summary["royalty_potential"],
        middle_summary["current_category"],
        middle_summary["possible_category"],
        middle_summary["pair_count"],
        middle_summary["max_multiplicity"],
        middle_summary["max_suit_count"],
        middle_summary["made_royalty"],
        middle_summary["royalty_potential"],
        middle_summary["flush_possible"],
        middle_summary["straight_possible"],
        middle_summary["full_house_possible"],
        middle_summary["quads_possible"],
        middle_summary["straight_flush_possible"],
        bottom_summary["current_category"],
        bottom_summary["possible_category"],
        bottom_summary["pair_count"],
        bottom_summary["max_multiplicity"],
        bottom_summary["max_suit_count"],
        bottom_summary["made_royalty"],
        bottom_summary["royalty_potential"],
        bottom_summary["flush_possible"],
        bottom_summary["straight_possible"],
        bottom_summary["full_house_possible"],
        bottom_summary["quads_possible"],
        bottom_summary["straight_flush_possible"],
        total_made_royalty,
        total_royalty_potential,
        total_royalty_potential - total_made_royalty,
        float(top_category_minus_middle_possible > 0),
        float(middle_category_minus_bottom_possible > 0),
        top_over_middle_complete,
        middle_over_bottom_complete,
        top_category_minus_middle_possible,
        middle_category_minus_bottom_possible,
    ]
    return np.asarray(values, dtype=np.float32)


def opportunity_proxy_feature_vector(row: dict[str, Any]) -> np.ndarray:
    remaining = _remaining_cards_for_t2_row(row)
    candidate = _board_opportunity_vector(_action_after_board(row, "candidate_action"), remaining)
    baseline = _board_opportunity_vector(_action_after_board(row, "baseline_action"), remaining)
    return np.concatenate([candidate, baseline, candidate - baseline]).astype(np.float32, copy=False)


def _stable_int_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % (2**31 - 1)


def _stable_card_set_key(cards: Iterable[Any]) -> str:
    return ",".join(sorted(str(card) for card in cards))


def _lookahead_seed_key(row: dict[str, Any], board: Board, action_key: str) -> tuple[Any, ...]:
    action = row.get(action_key) or {}
    opponent = row.get("opponent_board") or {}
    opponent_cards: list[Any] = []
    for row_cards in opponent.values():
        opponent_cards.extend(row_cards)
    return (
        row.get("source_log", ""),
        row.get("hand_seed", ""),
        action_key,
        _stable_card_set_key(board.all_cards()),
        _stable_card_set_key(action.get("discards", ())),
        _stable_card_set_key(row.get("dead_cards", ())),
        _stable_card_set_key(row.get("cards_to_place", ())),
        _stable_card_set_key(opponent_cards),
    )


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction))))
    return float(ordered[index])


def _avg(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _sample_t3_deals(row: dict[str, Any], board: Board, *, sample_count: int, seed_suffix: str) -> tuple[tuple[str, str, str], ...]:
    used = set(board.all_cards())
    used.update(str(card) for card in row.get("dead_cards", ()))
    used.update(str(card) for card in row.get("cards_to_place", ()))
    opponent = row.get("opponent_board") or {}
    for cards in opponent.values():
        used.update(str(card) for card in cards)
    deck = tuple(card for card in ALL_CARDS if card not in used)
    deals = tuple(combinations(deck, 3))
    if len(deals) <= sample_count:
        return deals
    rng = np.random.default_rng(_stable_int_seed(*_lookahead_seed_key(row, board, seed_suffix)))
    indices = sorted(int(index) for index in rng.choice(len(deals), size=sample_count, replace=False))
    return tuple(deals[index] for index in indices)


def _dead_cards_for_lookahead(row: dict[str, Any], board: Board, action_key: str, t3_deal: tuple[str, str, str]) -> tuple[str, ...]:
    dead: list[str] = []
    action = row.get(action_key) or {}
    dead.extend(str(card) for card in row.get("dead_cards", ()))
    dead.extend(str(card) for card in action.get("discards", ()))
    opponent = row.get("opponent_board") or {}
    for cards in opponent.values():
        dead.extend(str(card) for card in cards)
    blocked = set(board.all_cards()) | set(t3_deal)
    unique: list[str] = []
    seen: set[str] = set()
    for card in dead:
        if card in blocked or card in seen:
            continue
        seen.add(card)
        unique.append(card)
    return tuple(unique)


def _short_lookahead_board_vector(row: dict[str, Any], action_key: str) -> np.ndarray:
    board = _action_after_board(row, action_key)
    if board.card_count() != 9:
        return np.zeros(len(LOOKAHEAD_BOARD_FEATURE_NAMES), dtype=np.float32)
    scores: list[float] = []
    non_bust_rates: list[float] = []
    deals = _sample_t3_deals(row, board, sample_count=LOOKAHEAD_SAMPLES, seed_suffix=action_key)
    for index, t3_deal in enumerate(deals):
        try:
            ranked = evaluate_two_turn_actions(
                board,
                t3_deal,
                opponent_board=None,
                dead_cards=_dead_cards_for_lookahead(row, board, action_key, t3_deal),
                max_future_deals=LOOKAHEAD_FINAL_SAMPLES,
                seed=_stable_int_seed(*_lookahead_seed_key(row, board, action_key), index),
            )
        except Exception:
            continue
        if not ranked:
            continue
        best = ranked[0]
        scores.append(float(best.score))
        non_bust_rates.append(float(best.non_bust_future_count / max(best.future_count, 1)))
    missing_rate = 1.0 - (len(scores) / max(len(deals), 1))
    if not scores:
        return np.asarray(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, missing_rate],
            dtype=np.float32,
        )
    values = [
        math.log1p(len(scores)),
        _avg(scores),
        float(np.std(np.asarray(scores, dtype=np.float32))),
        min(scores),
        _quantile(scores, 0.05),
        _quantile(scores, 0.50),
        _quantile(scores, 0.95),
        max(scores),
        _avg(non_bust_rates),
        min(non_bust_rates) if non_bust_rates else 0.0,
        missing_rate,
    ]
    return np.asarray(values, dtype=np.float32)


def lookahead_proxy_feature_vector(row: dict[str, Any]) -> np.ndarray:
    candidate = _short_lookahead_board_vector(row, "candidate_action")
    baseline = _short_lookahead_board_vector(row, "baseline_action")
    return np.concatenate([candidate, baseline, candidate - baseline]).astype(np.float32, copy=False)


def feature_column_names(feature_mode: str) -> list[str]:
    if feature_mode == FEATURE_MODE_RUNTIME_META_ONLY:
        return list(RUNTIME_META_FEATURE_NAMES)
    if feature_mode == FEATURE_MODE_PRECONFIRM_META_ONLY:
        return list(PRECONFIRM_META_FEATURE_NAMES)
    if feature_mode == FEATURE_MODE_RUNTIME_TAIL_META_ONLY:
        return list(RUNTIME_TAIL_META_FEATURE_NAMES)
    if feature_mode == FEATURE_MODE_HU_DELTA_ONLY:
        return [f"hu_delta_f{index}" for index in range(HU_FEATURE_DIM)]
    if feature_mode == FEATURE_MODE_HU_DELTA_PLUS_PRECONFIRM_META:
        return [f"hu_delta_f{index}" for index in range(HU_FEATURE_DIM)] + list(PRECONFIRM_META_FEATURE_NAMES)
    if feature_mode == FEATURE_MODE_HU_DELTA_PLUS_RUNTIME_TAIL_META:
        return [f"hu_delta_f{index}" for index in range(HU_FEATURE_DIM)] + list(RUNTIME_TAIL_META_FEATURE_NAMES)
    if feature_mode == FEATURE_MODE_OPPORTUNITY_PROXY_ONLY:
        return list(OPPORTUNITY_PROXY_FEATURE_NAMES)
    if feature_mode == FEATURE_MODE_OPPORTUNITY_PROXY_PLUS_PRECONFIRM_META:
        return list(OPPORTUNITY_PROXY_FEATURE_NAMES) + list(PRECONFIRM_META_FEATURE_NAMES)
    if feature_mode == FEATURE_MODE_OPPORTUNITY_PROXY_PLUS_RUNTIME_TAIL_META:
        return list(OPPORTUNITY_PROXY_FEATURE_NAMES) + list(RUNTIME_TAIL_META_FEATURE_NAMES)
    if feature_mode == FEATURE_MODE_LOOKAHEAD_PROXY_ONLY:
        return list(LOOKAHEAD_PROXY_FEATURE_NAMES)
    if feature_mode == FEATURE_MODE_LOOKAHEAD_PROXY_PLUS_PRECONFIRM_META:
        return list(LOOKAHEAD_PROXY_FEATURE_NAMES) + list(PRECONFIRM_META_FEATURE_NAMES)
    if feature_mode == FEATURE_MODE_LOOKAHEAD_PROXY_PLUS_RUNTIME_TAIL_META:
        return list(LOOKAHEAD_PROXY_FEATURE_NAMES) + list(RUNTIME_TAIL_META_FEATURE_NAMES)
    if feature_mode == FEATURE_MODE_OPPORTUNITY_LOOKAHEAD_PROXY_PLUS_PRECONFIRM_META:
        return list(OPPORTUNITY_PROXY_FEATURE_NAMES) + list(LOOKAHEAD_PROXY_FEATURE_NAMES) + list(
            PRECONFIRM_META_FEATURE_NAMES
        )
    if feature_mode == FEATURE_MODE_OPPORTUNITY_LOOKAHEAD_PROXY_PLUS_RUNTIME_TAIL_META:
        return list(OPPORTUNITY_PROXY_FEATURE_NAMES) + list(LOOKAHEAD_PROXY_FEATURE_NAMES) + list(
            RUNTIME_TAIL_META_FEATURE_NAMES
        )
    names = [f"hu_f{index}" for index in range(HU_FEATURE_DIM)]
    if feature_mode == FEATURE_MODE_HU_PLUS_RUNTIME_META:
        names.extend(RUNTIME_META_FEATURE_NAMES)
    elif feature_mode == FEATURE_MODE_HU_PLUS_RUNTIME_TAIL_META:
        names.extend(RUNTIME_TAIL_META_FEATURE_NAMES)
    elif feature_mode != FEATURE_MODE_HU_ONLY:
        raise ValueError(f"unknown feature_mode={feature_mode!r}")
    return names


def risk_feature_vector_for_row(row: dict[str, Any], feature_mode: str) -> np.ndarray:
    if feature_mode == FEATURE_MODE_RUNTIME_META_ONLY:
        return runtime_meta_feature_vector(row)
    if feature_mode == FEATURE_MODE_PRECONFIRM_META_ONLY:
        return preconfirm_meta_feature_vector(row)
    if feature_mode == FEATURE_MODE_RUNTIME_TAIL_META_ONLY:
        return runtime_tail_meta_feature_vector(row)
    if feature_mode in (
        FEATURE_MODE_HU_DELTA_ONLY,
        FEATURE_MODE_HU_DELTA_PLUS_PRECONFIRM_META,
        FEATURE_MODE_HU_DELTA_PLUS_RUNTIME_TAIL_META,
    ):
        feature_row = hu_delta_feature_vector(row)
    elif feature_mode in (
        FEATURE_MODE_OPPORTUNITY_PROXY_ONLY,
        FEATURE_MODE_OPPORTUNITY_PROXY_PLUS_PRECONFIRM_META,
        FEATURE_MODE_OPPORTUNITY_PROXY_PLUS_RUNTIME_TAIL_META,
    ):
        feature_row = opportunity_proxy_feature_vector(row)
    elif feature_mode in (
        FEATURE_MODE_LOOKAHEAD_PROXY_ONLY,
        FEATURE_MODE_LOOKAHEAD_PROXY_PLUS_PRECONFIRM_META,
        FEATURE_MODE_LOOKAHEAD_PROXY_PLUS_RUNTIME_TAIL_META,
    ):
        feature_row = lookahead_proxy_feature_vector(row)
    elif feature_mode == FEATURE_MODE_OPPORTUNITY_LOOKAHEAD_PROXY_PLUS_PRECONFIRM_META:
        feature_row = np.concatenate([opportunity_proxy_feature_vector(row), lookahead_proxy_feature_vector(row)]).astype(
            np.float32,
            copy=False,
        )
    elif feature_mode == FEATURE_MODE_OPPORTUNITY_LOOKAHEAD_PROXY_PLUS_RUNTIME_TAIL_META:
        feature_row = np.concatenate([opportunity_proxy_feature_vector(row), lookahead_proxy_feature_vector(row)]).astype(
            np.float32,
            copy=False,
        )
    else:
        sample = risk_row_to_sample(row)
        matrix, _targets = sample_to_matrix(sample)
        feature_row = matrix[0].astype(np.float32, copy=False)

    if feature_mode == FEATURE_MODE_HU_PLUS_RUNTIME_META:
        feature_row = np.concatenate([feature_row, runtime_meta_feature_vector(row)]).astype(np.float32, copy=False)
    elif feature_mode == FEATURE_MODE_HU_PLUS_RUNTIME_TAIL_META:
        feature_row = np.concatenate([feature_row, runtime_tail_meta_feature_vector(row)]).astype(np.float32, copy=False)
    elif feature_mode == FEATURE_MODE_HU_DELTA_PLUS_PRECONFIRM_META:
        feature_row = np.concatenate([feature_row, preconfirm_meta_feature_vector(row)]).astype(np.float32, copy=False)
    elif feature_mode == FEATURE_MODE_HU_DELTA_PLUS_RUNTIME_TAIL_META:
        feature_row = np.concatenate([feature_row, runtime_tail_meta_feature_vector(row)]).astype(np.float32, copy=False)
    elif feature_mode == FEATURE_MODE_OPPORTUNITY_PROXY_PLUS_PRECONFIRM_META:
        feature_row = np.concatenate([feature_row, preconfirm_meta_feature_vector(row)]).astype(np.float32, copy=False)
    elif feature_mode == FEATURE_MODE_OPPORTUNITY_PROXY_PLUS_RUNTIME_TAIL_META:
        feature_row = np.concatenate([feature_row, runtime_tail_meta_feature_vector(row)]).astype(np.float32, copy=False)
    elif feature_mode == FEATURE_MODE_LOOKAHEAD_PROXY_PLUS_PRECONFIRM_META:
        feature_row = np.concatenate([feature_row, preconfirm_meta_feature_vector(row)]).astype(np.float32, copy=False)
    elif feature_mode == FEATURE_MODE_LOOKAHEAD_PROXY_PLUS_RUNTIME_TAIL_META:
        feature_row = np.concatenate([feature_row, runtime_tail_meta_feature_vector(row)]).astype(np.float32, copy=False)
    elif feature_mode == FEATURE_MODE_OPPORTUNITY_LOOKAHEAD_PROXY_PLUS_PRECONFIRM_META:
        feature_row = np.concatenate([feature_row, preconfirm_meta_feature_vector(row)]).astype(np.float32, copy=False)
    elif feature_mode == FEATURE_MODE_OPPORTUNITY_LOOKAHEAD_PROXY_PLUS_RUNTIME_TAIL_META:
        feature_row = np.concatenate([feature_row, runtime_tail_meta_feature_vector(row)]).astype(np.float32, copy=False)
    elif feature_mode not in (
        FEATURE_MODE_HU_ONLY,
        FEATURE_MODE_RUNTIME_META_ONLY,
        FEATURE_MODE_PRECONFIRM_META_ONLY,
        FEATURE_MODE_RUNTIME_TAIL_META_ONLY,
        FEATURE_MODE_HU_DELTA_ONLY,
        FEATURE_MODE_OPPORTUNITY_PROXY_ONLY,
        FEATURE_MODE_LOOKAHEAD_PROXY_ONLY,
    ):
        raise ValueError(f"unknown feature_mode={feature_mode!r}")
    return feature_row.astype(np.float32, copy=False)


def materialize_training_rows(
    rows: list[dict[str, Any]],
    *,
    feature_mode: str = FEATURE_MODE_HU_ONLY,
    target_mode: str = TARGET_MODE_WHOLE_GAME_RISK,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    features: list[np.ndarray] = []
    labels: list[int] = []
    skipped: Counter[str] = Counter()
    for row in rows:
        target = target_label_for_row(row, target_mode)
        if target is None:
            skipped[f"target:{target_mode}:excluded"] += 1
            continue
        label, target_group = target
        if not row_ready_for_target(row, target_mode):
            skipped["not_replay_ready"] += 1
            continue
        try:
            feature_row = risk_feature_vector_for_row(row, feature_mode)
        except Exception as exc:  # pragma: no cover - surfaced through manifest.
            skipped[f"feature_error:{type(exc).__name__}"] += 1
            continue
        selected.append(row)
        features.append(feature_row)
        labels.append(label)
    if not selected:
        raise ValueError(f"no trainable rows for target_mode={target_mode}; skipped={dict(skipped)}")
    metadata = []
    for index, row in enumerate(selected):
        metadata.append(
            {
                "row_index": index,
                "source_log": row.get("source_log", ""),
                "config_id": row.get("config_id", ""),
                "hand_seed": row.get("hand_seed", ""),
                "split_group_source_log": split_group_key(row, SPLIT_MODE_SOURCE_LOG),
                "split_group_source_seed": split_group_key(row, SPLIT_MODE_SOURCE_SEED),
                "seat": row.get("seat", ""),
                "seat_swap": row.get("seat_swap", ""),
                "candidate_source": row.get("candidate_source", ""),
                "recommended_training_use": recommended_use(row),
                "risk_target_group": target_label_for_row(row, target_mode)[1],
                "label": labels[index],
                "state_signature": row.get("state_signature", ""),
                "action_signature": row.get("action_signature", ""),
                "baseline_action_signature": row.get("baseline_action_signature", ""),
                "candidate_index": safe_int(row.get("candidate_index", row.get("candidate_action_index")), -1),
                "baseline_index": safe_int(row.get("baseline_index", row.get("baseline_action_index")), -1),
                "realized_delta": safe_float(row.get("realized_delta")),
                "realized_delta_observed": int(
                    truthy(
                        row.get(
                            "realized_delta_observed",
                            truthy(row.get("override_fired")) and truthy(row.get("realized_delta_valid")),
                        )
                    )
                ),
                "realized_loss": max(0.0, -safe_float(row.get("realized_delta"))),
                "local_replay_bucket": row.get("local_replay_bucket", ""),
                "local_replay_label": row.get("local_replay_label", ""),
                "local_replay_delta": safe_float(row.get("local_replay_delta")),
                "confirm_delta": safe_float(row.get("confirm_delta")),
                "confirm_delta_se": safe_float(row.get("confirm_delta_se")),
                "predicted_delta": safe_float(row.get("predicted_delta")),
                "gate_probability": safe_float(row.get("gate_probability")),
                "candidate_ev_rank": safe_int(row.get("candidate_ev_rank"), 9999),
            }
        )
    return np.vstack(features).astype(np.float32), np.asarray(labels, dtype=np.float32), metadata


def split_group_key(row: dict[str, Any], mode: str) -> str:
    source_log = str(row.get("source_log") or "")
    if mode == SPLIT_MODE_SOURCE_LOG:
        return source_log or f"missing_source_log:{row.get('hand_seed', '')}"
    if mode == SPLIT_MODE_SOURCE_SEED:
        match = re.search(r"seed(\d+)", source_log)
        if match:
            return match.group(1)
        hand_seed = str(row.get("hand_seed") or "")
        if len(hand_seed) >= 10 and hand_seed[:10].isdigit():
            return hand_seed[:10]
        return source_log or f"missing_source_seed:{hand_seed}"
    if mode == SPLIT_MODE_ROW_STRATIFIED:
        return ""
    raise ValueError(f"unknown split mode {mode!r}")


def stratified_split(metadata: list[dict[str, Any]], labels: np.ndarray, *, seed: int, train_fraction: float, val_fraction: float) -> np.ndarray:
    if train_fraction <= 0.0 or val_fraction < 0.0 or train_fraction + val_fraction >= 1.0:
        raise ValueError("split fractions must satisfy train>0, val>=0, train+val<1")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        key = "|".join(
            [
                str(int(labels[index])),
                str(row.get("seat", "")),
                str(row.get("risk_target_group") or row.get("recommended_training_use", "")),
            ]
        )
        groups[key].append(index)
    split = np.full(len(metadata), SPLIT_NAME_TO_ID["train"], dtype=np.int8)
    rng = np.random.default_rng(seed)
    for indices in groups.values():
        values = np.asarray(indices, dtype=np.int64)
        rng.shuffle(values)
        n = int(values.size)
        val_n = int(round(n * val_fraction))
        test_n = int(round(n * (1.0 - train_fraction - val_fraction)))
        if n >= 3:
            val_n = max(1, val_n)
            test_n = max(1, test_n)
        if val_n + test_n >= n:
            overflow = val_n + test_n - n + 1
            test_n = max(0, test_n - overflow)
        split[values[:val_n]] = SPLIT_NAME_TO_ID["val"]
        split[values[val_n : val_n + test_n]] = SPLIT_NAME_TO_ID["test"]
    return split


def group_stratified_split(
    metadata: list[dict[str, Any]],
    labels: np.ndarray,
    *,
    seed: int,
    train_fraction: float,
    val_fraction: float,
    split_mode: str,
) -> np.ndarray:
    if split_mode == SPLIT_MODE_ROW_STRATIFIED:
        return stratified_split(
            metadata,
            labels,
            seed=seed,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
        )
    if train_fraction <= 0.0 or val_fraction < 0.0 or train_fraction + val_fraction >= 1.0:
        raise ValueError("split fractions must satisfy train>0, val>=0, train+val<1")
    group_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        key = str(row.get(f"split_group_{split_mode}") or row.get(split_mode) or "")
        if not key:
            key = f"row:{index}"
        group_to_indices[key].append(index)
    if len(group_to_indices) < 3:
        raise ValueError(f"split_mode={split_mode} needs at least 3 groups, got {len(group_to_indices)}")

    split_ids = (SPLIT_NAME_TO_ID["train"], SPLIT_NAME_TO_ID["val"], SPLIT_NAME_TO_ID["test"])
    fractions = {
        SPLIT_NAME_TO_ID["train"]: train_fraction,
        SPLIT_NAME_TO_ID["val"]: val_fraction,
        SPLIT_NAME_TO_ID["test"]: 1.0 - train_fraction - val_fraction,
    }
    total_rows = float(len(metadata))
    total_pos = float(np.sum(labels > 0.5))
    total_neg = float(len(metadata) - total_pos)
    row_targets = {split_id: max(1.0, total_rows * fractions[split_id]) for split_id in split_ids}
    pos_targets = {split_id: max(1.0, total_pos * fractions[split_id]) for split_id in split_ids}
    neg_targets = {split_id: max(1.0, total_neg * fractions[split_id]) for split_id in split_ids}
    row_counts = {split_id: 0.0 for split_id in split_ids}
    pos_counts = {split_id: 0.0 for split_id in split_ids}
    neg_counts = {split_id: 0.0 for split_id in split_ids}

    rng = np.random.default_rng(seed)
    group_items = list(group_to_indices.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)
    assignments: dict[str, int] = {}
    for group, indices in group_items:
        idx = np.asarray(indices, dtype=np.int64)
        group_rows = float(idx.size)
        group_pos = float(np.sum(labels[idx] > 0.5))
        group_neg = group_rows - group_pos
        best_split = max(
            split_ids,
            key=lambda split_id: (
                max(0.0, (row_targets[split_id] - row_counts[split_id]) / row_targets[split_id])
                + max(0.0, (pos_targets[split_id] - pos_counts[split_id]) / pos_targets[split_id])
                + max(0.0, (neg_targets[split_id] - neg_counts[split_id]) / neg_targets[split_id])
            ),
        )
        assignments[group] = best_split
        row_counts[best_split] += group_rows
        pos_counts[best_split] += group_pos
        neg_counts[best_split] += group_neg

    split = np.full(len(metadata), SPLIT_NAME_TO_ID["train"], dtype=np.int8)
    for group, indices in group_to_indices.items():
        split[np.asarray(indices, dtype=np.int64)] = assignments[group]
    return split


def fixed_group_split(
    metadata: list[dict[str, Any]],
    *,
    split_mode: str,
    val_groups: set[str],
    test_groups: set[str],
) -> np.ndarray:
    if split_mode == SPLIT_MODE_ROW_STRATIFIED:
        raise ValueError("fixed groups require source_log or source_seed split mode")
    overlap = val_groups & test_groups
    if overlap:
        raise ValueError(f"fixed val/test groups overlap: {sorted(overlap)}")
    split = np.full(len(metadata), SPLIT_NAME_TO_ID["train"], dtype=np.int8)
    present_groups: set[str] = set()
    for index, row in enumerate(metadata):
        group = str(row.get(f"split_group_{split_mode}") or row.get(split_mode) or "")
        present_groups.add(group)
        if group in val_groups:
            split[index] = SPLIT_NAME_TO_ID["val"]
        elif group in test_groups:
            split[index] = SPLIT_NAME_TO_ID["test"]
    missing = (val_groups | test_groups) - present_groups
    if missing:
        raise ValueError(f"fixed split groups not present: {sorted(missing)}")
    counts = Counter(int(value) for value in split)
    for split_name, split_id in SPLIT_NAME_TO_ID.items():
        if counts.get(split_id, 0) <= 0:
            raise ValueError(f"fixed split produced empty {split_name} split")
    return split


def average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float:
    positives = int(np.sum(labels > 0.5))
    if positives <= 0:
        return 0.0
    order = np.argsort(-probabilities)
    hits = 0
    score = 0.0
    for rank, index in enumerate(order, start=1):
        if labels[index] > 0.5:
            hits += 1
            score += hits / rank
    return float(score / positives)


def roc_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    positives = probabilities[labels > 0.5]
    negatives = probabilities[labels <= 0.5]
    if positives.size == 0 or negatives.size == 0:
        return 0.0
    combined = np.concatenate([positives, negatives])
    order = np.argsort(combined)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, combined.size + 1, dtype=np.float64)
    # Average ties.
    sorted_values = combined[order]
    start = 0
    while start < sorted_values.size:
        end = start + 1
        while end < sorted_values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        if end - start > 1:
            avg = float(np.mean(ranks[order[start:end]]))
            ranks[order[start:end]] = avg
        start = end
    pos_ranks = ranks[: positives.size]
    auc = (np.sum(pos_ranks) - positives.size * (positives.size + 1) / 2.0) / (positives.size * negatives.size)
    return float(auc)


def split_metrics(labels: np.ndarray, probabilities: np.ndarray, *, split_name: str) -> dict[str, Any]:
    if labels.size == 0:
        return {"split": split_name, "rows": 0}
    preds = (probabilities >= 0.5).astype(np.float32)
    return {
        "split": split_name,
        "rows": int(labels.size),
        "positives": int(np.sum(labels > 0.5)),
        "positive_rate": float(np.mean(labels)),
        "probability_mean": float(np.mean(probabilities)),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "accuracy_at_0p5": float(np.mean(preds == labels)),
        "average_precision": average_precision(labels, probabilities),
        "roc_auc": roc_auc(labels, probabilities),
    }


def threshold_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    split_name: str,
    thresholds: Iterable[float],
    realized_deltas: np.ndarray | None = None,
    realized_observed: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    if realized_deltas is None:
        realized_deltas = np.zeros(labels.shape, dtype=np.float32)
    else:
        realized_deltas = realized_deltas.astype(np.float32, copy=False)
        if realized_deltas.shape[0] != labels.shape[0]:
            raise ValueError("realized_deltas must match labels length")
    if realized_observed is None:
        realized_observed = np.ones(labels.shape, dtype=bool)
    else:
        realized_observed = realized_observed.astype(bool, copy=False)
        if realized_observed.shape[0] != labels.shape[0]:
            raise ValueError("realized_observed must match labels length")
    rows = []
    for threshold in thresholds:
        predicted = probabilities >= float(threshold)
        actual = labels > 0.5
        tp = int(np.sum(predicted & actual))
        fp = int(np.sum(predicted & ~actual))
        tn = int(np.sum(~predicted & ~actual))
        fn = int(np.sum(~predicted & actual))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        selected_deltas = realized_deltas[predicted]
        observed_selected = predicted & realized_observed
        observed_deltas = realized_deltas[observed_selected]
        selected_losses = np.maximum(0.0, -selected_deltas) if selected_deltas.size else np.zeros(0, dtype=np.float32)
        observed_losses = np.maximum(0.0, -observed_deltas) if observed_deltas.size else np.zeros(0, dtype=np.float32)
        selected_delta_sum = float(np.sum(selected_deltas)) if selected_deltas.size else 0.0
        observed_delta_sum = float(np.sum(observed_deltas)) if observed_deltas.size else 0.0
        rows.append(
            {
                "split": split_name,
                "threshold": float(threshold),
                "rows": int(labels.size),
                "selected_rows": int(np.sum(predicted)),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
                "fire_rate": float(np.mean(predicted)) if labels.size else 0.0,
                "selected_realized_delta_sum": selected_delta_sum,
                "selected_realized_delta_mean": float(np.mean(selected_deltas)) if selected_deltas.size else 0.0,
                "estimated_realized_delta_per_row": selected_delta_sum / labels.size if labels.size else 0.0,
                "selected_realized_loss_count": int(np.sum(selected_deltas < 0.0)) if selected_deltas.size else 0,
                "selected_realized_loss_mean": float(np.mean(selected_losses)) if selected_losses.size else 0.0,
                "selected_realized_max_loss": float(np.max(selected_losses)) if selected_losses.size else 0.0,
                "selected_observed_realized_delta_count": int(np.sum(observed_selected)),
                "selected_unknown_realized_delta_count": int(np.sum(predicted & ~realized_observed)),
                "selected_observed_realized_delta_sum": observed_delta_sum,
                "selected_observed_realized_delta_mean": float(np.mean(observed_deltas)) if observed_deltas.size else 0.0,
                "selected_observed_realized_loss_count": int(np.sum(observed_deltas < 0.0)) if observed_deltas.size else 0,
                "selected_observed_realized_loss_mean": float(np.mean(observed_losses)) if observed_losses.size else 0.0,
                "selected_observed_realized_max_loss": float(np.max(observed_losses)) if observed_losses.size else 0.0,
            }
        )
    return rows


def topk_selection_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    split_name: str,
    topk_counts: Iterable[int] = DEFAULT_TOPK_METRIC_COUNTS,
    realized_deltas: np.ndarray | None = None,
    realized_observed: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    if realized_deltas is None:
        realized_deltas = np.zeros(labels.shape, dtype=np.float32)
    else:
        realized_deltas = realized_deltas.astype(np.float32, copy=False)
        if realized_deltas.shape[0] != labels.shape[0]:
            raise ValueError("realized_deltas must match labels length")
    if realized_observed is None:
        realized_observed = np.ones(labels.shape, dtype=bool)
    else:
        realized_observed = realized_observed.astype(bool, copy=False)
        if realized_observed.shape[0] != labels.shape[0]:
            raise ValueError("realized_observed must match labels length")
    if labels.size == 0:
        return []
    order = np.argsort(-probabilities)
    positive_count = int(np.sum(labels > 0.5))
    rows = []
    for raw_count in topk_counts:
        count = int(raw_count)
        if count <= 0 or count > labels.size:
            continue
        selected_idx = order[:count]
        selected_labels = labels[selected_idx] > 0.5
        tp = int(np.sum(selected_labels))
        fp = int(count - tp)
        selected_deltas = realized_deltas[selected_idx]
        observed_mask = realized_observed[selected_idx]
        observed_deltas = selected_deltas[observed_mask]
        selected_losses = np.maximum(0.0, -selected_deltas) if selected_deltas.size else np.zeros(0, dtype=np.float32)
        observed_losses = np.maximum(0.0, -observed_deltas) if observed_deltas.size else np.zeros(0, dtype=np.float32)
        min_probability = float(probabilities[selected_idx[-1]]) if selected_idx.size else 0.0
        rows.append(
            {
                "split": split_name,
                "topk": count,
                "rows": int(labels.size),
                "positives": positive_count,
                "selected_rows": count,
                "tp": tp,
                "fp": fp,
                "precision": float(tp / count),
                "recall": float(tp / max(positive_count, 1)),
                "min_probability": min_probability,
                "selected_realized_delta_sum": float(np.sum(selected_deltas)) if selected_deltas.size else 0.0,
                "selected_realized_delta_mean": float(np.mean(selected_deltas)) if selected_deltas.size else 0.0,
                "estimated_realized_delta_per_row": float(np.sum(selected_deltas) / labels.size),
                "selected_realized_loss_count": int(np.sum(selected_deltas < 0.0)) if selected_deltas.size else 0,
                "selected_realized_loss_mean": float(np.mean(selected_losses)) if selected_losses.size else 0.0,
                "selected_realized_max_loss": float(np.max(selected_losses)) if selected_losses.size else 0.0,
                "selected_observed_realized_delta_count": int(np.sum(observed_mask)),
                "selected_unknown_realized_delta_count": int(count - np.sum(observed_mask)),
                "selected_observed_realized_delta_sum": float(np.sum(observed_deltas)) if observed_deltas.size else 0.0,
                "selected_observed_realized_delta_mean": float(np.mean(observed_deltas)) if observed_deltas.size else 0.0,
                "selected_observed_realized_loss_count": int(np.sum(observed_deltas < 0.0)) if observed_deltas.size else 0,
                "selected_observed_realized_loss_mean": float(np.mean(observed_losses)) if observed_losses.size else 0.0,
                "selected_observed_realized_max_loss": float(np.max(observed_losses)) if observed_losses.size else 0.0,
            }
        )
    return rows


def metadata_breakdown(metadata: list[dict[str, Any]], labels: np.ndarray, probabilities: np.ndarray, split: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, str, list[int]]] = [("overall", "all", list(range(len(metadata))))]
    for field in ("seat", "recommended_training_use", "local_replay_bucket", "local_replay_label"):
        for value in sorted({str(row.get(field, "")) for row in metadata}):
            groups.append((field, value, [i for i, row in enumerate(metadata) if str(row.get(field, "")) == value]))
    for split_id, split_name in SPLIT_ID_TO_NAME.items():
        groups.append(("split", split_name, [i for i, value in enumerate(split) if int(value) == split_id]))
    for field, value, indices in groups:
        if not indices:
            continue
        idx = np.asarray(indices, dtype=np.int64)
        rows.append(
            {
                "group_field": field,
                "group_value": value,
                "rows": int(idx.size),
                "positives": int(np.sum(labels[idx] > 0.5)),
                "positive_rate": float(np.mean(labels[idx])),
                "probability_mean": float(np.mean(probabilities[idx])),
                "average_precision": average_precision(labels[idx], probabilities[idx]),
                "roc_auc": roc_auc(labels[idx], probabilities[idx]),
            }
        )
    return rows


def sample_weights_for_metadata(
    metadata: list[dict[str, Any]],
    *,
    target_mode: str,
    topk_replay_negative_weight: float = 1.0,
    topk_realized_loss_weight: float = 1.0,
    topk_replay_positive_weight: float = 1.0,
) -> np.ndarray:
    weights = np.ones(len(metadata), dtype=np.float32)
    if target_mode != TARGET_MODE_TOPK_CONFIRM_FIRE:
        return weights
    for index, row in enumerate(metadata):
        group = str(row.get("risk_target_group") or row.get("recommended_training_use") or "")
        if group == "topk_confirm_replay_negative":
            weights[index] = float(topk_replay_negative_weight)
        elif group == "topk_confirm_realized_loss":
            weights[index] = float(topk_realized_loss_weight)
        elif group == "topk_confirm_replay_positive":
            weights[index] = float(topk_replay_positive_weight)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("sample weights must be positive and finite")
    return weights


def train_model(
    features: np.ndarray,
    labels: np.ndarray,
    split: np.ndarray,
    *,
    hidden_layer_sizes: tuple[int, ...],
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
    seed: int,
    device_choice: str,
    feature_mode: str = FEATURE_MODE_HU_ONLY,
    target_mode: str = TARGET_MODE_WHOLE_GAME_RISK,
    pos_weight_mode: str = POS_WEIGHT_MODE_AUTO,
    pos_weight_value: float = 1.0,
    sample_weights: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    torch = _import_torch()
    torch.manual_seed(seed)
    device = select_device(torch, device_choice)

    train_idx = np.where(split == SPLIT_NAME_TO_ID["train"])[0]
    val_idx = np.where(split == SPLIT_NAME_TO_ID["val"])[0]
    if train_idx.size == 0 or val_idx.size == 0:
        raise ValueError("train and val splits must both be non-empty")

    feature_mean = features[train_idx].mean(axis=0).astype(np.float32)
    feature_scale = features[train_idx].std(axis=0).astype(np.float32)
    feature_scale[feature_scale < 1e-6] = 1.0

    x_norm = ((features - feature_mean) / feature_scale).astype(np.float32)
    if sample_weights is None:
        sample_weights = np.ones(labels.shape, dtype=np.float32)
    else:
        sample_weights = sample_weights.astype(np.float32, copy=False)
        if sample_weights.shape != labels.shape:
            raise ValueError("sample_weights must match labels length")
        if not np.all(np.isfinite(sample_weights)) or np.any(sample_weights <= 0.0):
            raise ValueError("sample_weights must be positive and finite")
    net = _build_torch_mlp(torch, int(features.shape[1]), hidden_layer_sizes, dropout).to(device)
    positives = float(np.sum(labels[train_idx] > 0.5))
    negatives = float(train_idx.size - positives)
    if pos_weight_mode == POS_WEIGHT_MODE_AUTO:
        resolved_pos_weight = negatives / max(positives, 1.0)
    elif pos_weight_mode == POS_WEIGHT_MODE_NONE:
        resolved_pos_weight = 1.0
    elif pos_weight_mode == POS_WEIGHT_MODE_VALUE:
        if pos_weight_value <= 0.0 or not math.isfinite(pos_weight_value):
            raise ValueError("--pos-weight-value must be positive and finite")
        resolved_pos_weight = float(pos_weight_value)
    else:
        raise ValueError(f"unknown pos_weight_mode={pos_weight_mode!r}")
    pos_weight = torch.tensor([resolved_pos_weight], dtype=torch.float32, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    best_state = None
    best_val = float("inf")
    best_epoch = -1
    stale = 0
    history: list[dict[str, Any]] = []

    def split_loss(indices: np.ndarray) -> float:
        net.eval()
        loss_sum = 0.0
        weight_sum = 0.0
        with torch.inference_mode():
            for start in range(0, int(indices.size), batch_size):
                batch_idx = indices[start : start + batch_size]
                x = torch.from_numpy(x_norm[batch_idx]).to(device)
                y = torch.from_numpy(labels[batch_idx].astype(np.float32)).to(device)
                w = torch.from_numpy(sample_weights[batch_idx]).to(device)
                logits = net(x).squeeze(-1)
                losses = loss_fn(logits, y) * w
                loss_sum += float(torch.sum(losses).detach().cpu().item())
                weight_sum += float(torch.sum(w).detach().cpu().item())
        return float(loss_sum / max(weight_sum, 1e-12))

    for epoch in range(1, epochs + 1):
        net.train()
        shuffled = train_idx.copy()
        rng.shuffle(shuffled)
        train_losses = []
        for start in range(0, int(shuffled.size), batch_size):
            batch_idx = shuffled[start : start + batch_size]
            x = torch.from_numpy(x_norm[batch_idx]).to(device)
            y = torch.from_numpy(labels[batch_idx].astype(np.float32)).to(device)
            w = torch.from_numpy(sample_weights[batch_idx]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = net(x).squeeze(-1)
            loss = torch.sum(loss_fn(logits, y) * w) / torch.clamp(torch.sum(w), min=1e-12)
            loss.backward()
            optimizer.step()
            train_losses.append((float(loss.detach().cpu().item()), float(np.sum(sample_weights[batch_idx]))))
        train_loss = float(
            sum(loss_value * weight_sum for loss_value, weight_sum in train_losses)
            / max(sum(weight_sum for _loss_value, weight_sum in train_losses), 1e-12)
        )
        val_loss = split_loss(val_idx)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, features.shape[0], batch_size):
            end = min(start + batch_size, features.shape[0])
            x = torch.from_numpy(x_norm[start:end]).to(device)
            logits = net(x).squeeze(-1)
            probabilities.append(torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32))
    probs = np.concatenate(probabilities) if probabilities else np.zeros(0, dtype=np.float32)
    positive_label, negative_label = target_label_names(target_mode)
    payload = {
        "model_kind": "hu_turn2_stage8c_whole_game_risk_head_mlp",
        "feature_dim": int(features.shape[1]),
        "feature_mode": feature_mode,
        "feature_column_names": feature_column_names(feature_mode),
        "hidden_layer_sizes": list(hidden_layer_sizes),
        "dropout": float(dropout),
        "state_dict": best_state,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "positive_label": positive_label,
        "negative_label": negative_label,
        "target_mode": target_mode,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "pos_weight": float(pos_weight.detach().cpu().item()),
        "pos_weight_mode": pos_weight_mode,
    }
    return payload, probs, history


def write_summary(path: Path, manifest: dict[str, Any], metrics: list[dict[str, Any]]) -> None:
    target_mode = manifest.get("target_mode", TARGET_MODE_WHOLE_GAME_RISK)
    if target_mode == TARGET_MODE_WHOLE_GAME_RISK:
        target_title = "Whole-Game Risk Head"
    elif target_mode == TARGET_MODE_LOCAL_EV_NEGATIVE:
        target_title = "Local EV Negative Head"
    elif target_mode == TARGET_MODE_TOPK_CONFIRM_FIRE:
        target_title = "TopK Confirm Fire Head"
    elif target_mode == TARGET_MODE_REALIZED_WHOLE_GAME_LOSS:
        target_title = "Realized Whole-Game Loss Head"
    else:
        target_title = str(target_mode)
    lines = [
        f"# HU T2 Stage8c {target_title} Smoke",
        "",
        "This is a research smoke. It does not approve production, P2 fixed status, T1, or 50k teacher generation.",
        "",
        "## Inputs",
        "",
        f"- rows: `{manifest['trainable_rows']}`",
        f"- positives: `{manifest['positive_rows']}`",
        f"- negatives: `{manifest['negative_rows']}`",
        f"- excluded local EV hard negatives: `{manifest['excluded_local_ev_hard_negative_rows']}`",
        f"- feature mode: `{manifest['feature_mode']}`",
        f"- target mode: `{manifest.get('target_mode', TARGET_MODE_WHOLE_GAME_RISK)}`",
        f"- split mode: `{manifest['split_mode']}`",
        f"- positive class weight: `{manifest.get('pos_weight', 0.0):.4f}` ({manifest.get('pos_weight_mode', 'auto')})",
        (
            "- TopK row weights: "
            f"replay_negative `{manifest.get('topk_replay_negative_weight', 1.0):.2f}`, "
            f"realized_loss `{manifest.get('topk_realized_loss_weight', 1.0):.2f}`, "
            f"replay_positive `{manifest.get('topk_replay_positive_weight', 1.0):.2f}`"
        ),
        f"- model: `{manifest['model_output']}`",
        "",
        "## Metrics",
        "",
        "| split | rows | positives | AP | ROC AUC | Brier | acc@0.5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        if "average_precision" not in row:
            continue
        lines.append(
            "| {split} | {rows} | {positives} | {ap:.4f} | {auc:.4f} | {brier:.4f} | {acc:.4f} |".format(
                split=row.get("split", ""),
                rows=int(row.get("rows", 0)),
                positives=int(row.get("positives", 0)),
                ap=float(row.get("average_precision", 0.0)),
                auc=float(row.get("roc_auc", 0.0)),
                brier=float(row.get("brier", 0.0)),
                acc=float(row.get("accuracy_at_0p5", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- risk-head training smoke: `Pass`",
            "- production / P2 fixed: `No-Go`",
            "- 50k teacher: `No-Go`",
            "- T1 training: `No-Go`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    thresholds = resolved_thresholds(args.threshold)
    started = time.time()
    paths = target_paths(args.collection_dir, args.input_jsonl)
    rows, source_rows = load_rows(paths)
    loaded_rows = len(rows)
    rows, excluded_recommended_use_counts = exclude_recommended_use_rows(rows, args.exclude_recommended_use)
    rows_after_recommended_use_filter = len(rows)
    rows = limit_rows_for_training(
        rows,
        max_rows=args.max_rows,
        mode=args.max_rows_mode,
        seed=args.max_rows_seed,
        target_mode=args.target_mode,
    )
    features, labels, metadata = materialize_training_rows(
        rows,
        feature_mode=args.feature_mode,
        target_mode=args.target_mode,
    )
    sample_weights = sample_weights_for_metadata(
        metadata,
        target_mode=args.target_mode,
        topk_replay_negative_weight=args.topk_replay_negative_weight,
        topk_realized_loss_weight=args.topk_realized_loss_weight,
        topk_replay_positive_weight=args.topk_replay_positive_weight,
    )
    fixed_val_groups = parse_fixed_groups(args.fixed_val_groups)
    fixed_test_groups = parse_fixed_groups(args.fixed_test_groups)
    if fixed_val_groups or fixed_test_groups:
        split = fixed_group_split(
            metadata,
            split_mode=args.split_mode,
            val_groups=fixed_val_groups,
            test_groups=fixed_test_groups,
        )
    else:
        split = group_stratified_split(
            metadata,
            labels,
            seed=args.seed,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            split_mode=args.split_mode,
        )
    payload, probabilities, history = train_model(
        features,
        labels,
        split,
        hidden_layer_sizes=parse_hidden_layers(args.hidden_layer_sizes),
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
        device_choice=args.device,
        feature_mode=args.feature_mode,
        target_mode=args.target_mode,
        pos_weight_mode=args.pos_weight_mode,
        pos_weight_value=args.pos_weight_value,
        sample_weights=sample_weights,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    torch = _import_torch()
    torch.save(payload, args.model_output)

    prediction_rows = []
    for index, row in enumerate(metadata):
        prediction_rows.append(row | {"split": SPLIT_ID_TO_NAME[int(split[index])], "risk_probability": float(probabilities[index])})

    metrics = []
    threshold_rows = []
    topk_rows = []
    realized_deltas = np.asarray([safe_float(row.get("realized_delta")) for row in metadata], dtype=np.float32)
    realized_observed = np.asarray([truthy(row.get("realized_delta_observed")) for row in metadata], dtype=bool)
    for split_id, split_name in SPLIT_ID_TO_NAME.items():
        indices = np.where(split == split_id)[0]
        metrics.append(split_metrics(labels[indices], probabilities[indices], split_name=split_name))
        threshold_rows.extend(
            threshold_metrics(
                labels[indices],
                probabilities[indices],
                split_name=split_name,
                thresholds=thresholds,
                realized_deltas=realized_deltas[indices],
                realized_observed=realized_observed[indices],
            )
        )
        topk_rows.extend(
            topk_selection_metrics(
                labels[indices],
                probabilities[indices],
                split_name=split_name,
                realized_deltas=realized_deltas[indices],
                realized_observed=realized_observed[indices],
            )
        )
    metrics.append(split_metrics(labels, probabilities, split_name="all"))
    threshold_rows.extend(
        threshold_metrics(
            labels,
            probabilities,
            split_name="all",
            thresholds=thresholds,
            realized_deltas=realized_deltas,
            realized_observed=realized_observed,
        )
    )
    topk_rows.extend(
        topk_selection_metrics(
            labels,
            probabilities,
            split_name="all",
            realized_deltas=realized_deltas,
            realized_observed=realized_observed,
        )
    )

    label_counts = Counter(int(label) for label in labels)
    use_counts = Counter(recommended_use(row) for row in rows)
    target_group_counts = Counter(str(row.get("risk_target_group", "")) for row in metadata)
    manifest = {
        "schema": "hu_turn2_stage8c_risk_head_training_smoke_v1",
        "input_paths": [str(path) for path in paths],
        "source_rows": source_rows,
        "loaded_rows": int(loaded_rows),
        "exclude_recommended_use": [str(value) for value in args.exclude_recommended_use],
        "excluded_recommended_use_rows": int(sum(excluded_recommended_use_counts.values())),
        "excluded_recommended_use_counts": dict(excluded_recommended_use_counts),
        "rows_after_recommended_use_filter": int(rows_after_recommended_use_filter),
        "sampled_input_rows": int(len(rows)),
        "max_rows": args.max_rows,
        "max_rows_mode": args.max_rows_mode,
        "max_rows_seed": args.max_rows_seed,
        "output_dir": str(args.output_dir),
        "model_output": str(args.model_output),
        "feature_dim": int(features.shape[1]),
        "feature_mode": args.feature_mode,
        "feature_column_names": feature_column_names(args.feature_mode),
        "target_mode": args.target_mode,
        "split_mode": args.split_mode,
        "fixed_val_groups": sorted(fixed_val_groups),
        "fixed_test_groups": sorted(fixed_test_groups),
        "split_group_count": (
            len({str(row.get(f"split_group_{args.split_mode}") or "") for row in metadata})
            if args.split_mode != SPLIT_MODE_ROW_STRATIFIED
            else int(labels.size)
        ),
        "runtime_meta_feature_names": (
            list(RUNTIME_META_FEATURE_NAMES)
            if args.feature_mode
            in (
                FEATURE_MODE_HU_PLUS_RUNTIME_META,
                FEATURE_MODE_RUNTIME_META_ONLY,
                FEATURE_MODE_HU_PLUS_RUNTIME_TAIL_META,
                FEATURE_MODE_RUNTIME_TAIL_META_ONLY,
                FEATURE_MODE_HU_DELTA_PLUS_RUNTIME_TAIL_META,
                FEATURE_MODE_OPPORTUNITY_PROXY_PLUS_RUNTIME_TAIL_META,
                FEATURE_MODE_LOOKAHEAD_PROXY_PLUS_RUNTIME_TAIL_META,
                FEATURE_MODE_OPPORTUNITY_LOOKAHEAD_PROXY_PLUS_RUNTIME_TAIL_META,
            )
            else []
        ),
        "preconfirm_meta_feature_names": (
            list(PRECONFIRM_META_FEATURE_NAMES)
            if args.feature_mode
            in (
                FEATURE_MODE_PRECONFIRM_META_ONLY,
                FEATURE_MODE_HU_DELTA_PLUS_PRECONFIRM_META,
                FEATURE_MODE_OPPORTUNITY_PROXY_PLUS_PRECONFIRM_META,
                FEATURE_MODE_LOOKAHEAD_PROXY_PLUS_PRECONFIRM_META,
                FEATURE_MODE_OPPORTUNITY_LOOKAHEAD_PROXY_PLUS_PRECONFIRM_META,
            )
            else []
        ),
        "confirm_tail_feature_names": (
            list(CONFIRM_TAIL_FEATURE_NAMES)
            if args.feature_mode
            in (
                FEATURE_MODE_HU_PLUS_RUNTIME_TAIL_META,
                FEATURE_MODE_RUNTIME_TAIL_META_ONLY,
                FEATURE_MODE_HU_DELTA_PLUS_RUNTIME_TAIL_META,
                FEATURE_MODE_OPPORTUNITY_PROXY_PLUS_RUNTIME_TAIL_META,
                FEATURE_MODE_LOOKAHEAD_PROXY_PLUS_RUNTIME_TAIL_META,
                FEATURE_MODE_OPPORTUNITY_LOOKAHEAD_PROXY_PLUS_RUNTIME_TAIL_META,
            )
            else []
        ),
        "opportunity_proxy_feature_names": (
            list(OPPORTUNITY_PROXY_FEATURE_NAMES)
            if args.feature_mode
            in (
                FEATURE_MODE_OPPORTUNITY_PROXY_ONLY,
                FEATURE_MODE_OPPORTUNITY_PROXY_PLUS_PRECONFIRM_META,
                FEATURE_MODE_OPPORTUNITY_LOOKAHEAD_PROXY_PLUS_PRECONFIRM_META,
                FEATURE_MODE_OPPORTUNITY_PROXY_PLUS_RUNTIME_TAIL_META,
                FEATURE_MODE_OPPORTUNITY_LOOKAHEAD_PROXY_PLUS_RUNTIME_TAIL_META,
            )
            else []
        ),
        "lookahead_proxy_feature_names": (
            list(LOOKAHEAD_PROXY_FEATURE_NAMES)
            if args.feature_mode
            in (
                FEATURE_MODE_LOOKAHEAD_PROXY_ONLY,
                FEATURE_MODE_LOOKAHEAD_PROXY_PLUS_PRECONFIRM_META,
                FEATURE_MODE_LOOKAHEAD_PROXY_PLUS_RUNTIME_TAIL_META,
                FEATURE_MODE_OPPORTUNITY_LOOKAHEAD_PROXY_PLUS_PRECONFIRM_META,
                FEATURE_MODE_OPPORTUNITY_LOOKAHEAD_PROXY_PLUS_RUNTIME_TAIL_META,
            )
            else []
        ),
        "lookahead_samples": LOOKAHEAD_SAMPLES,
        "lookahead_final_samples": LOOKAHEAD_FINAL_SAMPLES,
        "trainable_rows": int(labels.size),
        "positive_rows": int(label_counts.get(1, 0)),
        "negative_rows": int(label_counts.get(0, 0)),
        "excluded_local_ev_hard_negative_rows": (
            int(use_counts.get("local_ev_hard_negative", 0))
            if args.target_mode == TARGET_MODE_WHOLE_GAME_RISK
            else 0
        ),
        "use_counts": dict(use_counts),
        "target_group_counts": dict(target_group_counts),
        "split_counts": dict(Counter(SPLIT_ID_TO_NAME[int(value)] for value in split)),
        "best_epoch": int(payload["best_epoch"]),
        "best_val_loss": float(payload["best_val_loss"]),
        "pos_weight_mode": args.pos_weight_mode,
        "pos_weight": float(payload["pos_weight"]),
        "topk_replay_negative_weight": float(args.topk_replay_negative_weight),
        "topk_realized_loss_weight": float(args.topk_realized_loss_weight),
        "topk_replay_positive_weight": float(args.topk_replay_positive_weight),
        "sample_weight_min": float(np.min(sample_weights)) if sample_weights.size else 0.0,
        "sample_weight_max": float(np.max(sample_weights)) if sample_weights.size else 0.0,
        "sample_weight_mean": float(np.mean(sample_weights)) if sample_weights.size else 0.0,
        "thresholds": thresholds,
        "topk_metric_counts": list(DEFAULT_TOPK_METRIC_COUNTS),
        "elapsed_seconds": time.time() - started,
        "production_p2_fixed": "No-Go",
        "t1_training": "No-Go",
        "teacher_50k": "No-Go",
    }
    (args.output_dir / "risk_head_training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "risk_head_training_history.csv", history)
    write_csv(args.output_dir / "risk_head_metrics.csv", metrics)
    write_csv(args.output_dir / "risk_head_threshold_metrics.csv", threshold_rows)
    write_csv(args.output_dir / "risk_head_topk_metrics.csv", topk_rows)
    write_csv(args.output_dir / "risk_head_predictions.csv", prediction_rows)
    write_csv(args.output_dir / "risk_head_breakdown.csv", metadata_breakdown(metadata, labels, probabilities, split))
    write_summary(args.output_dir / "risk_head_training_summary.md", manifest, metrics)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
