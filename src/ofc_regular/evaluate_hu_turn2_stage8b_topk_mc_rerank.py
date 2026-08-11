"""Validate HU T2 Stage8b as a TopK candidate generator with MC rerank.

This evaluator deliberately does not use Stage8b as a direct production gate.
Stage8b scores all legal HU T2 actions, keeps a small TopK candidate set plus
the current baseline action, then reranks that reduced set by MC rollout with a
configurable T3 continuation. Hidden-discard default is Stage3 reference direct;
Stage7_candidate_A m5_r10 must be opted in explicitly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .action_key import (
    ACTION_KEY_SCHEMA,
    action_key,
    canonical_argmax_index,
    canonical_descending_indices,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from .action_space import Action, generate_turn_actions
from .ai_profiles import DEFAULT_HU_TURN3_STAGE7_MIN_MARGIN, DEFAULT_HU_TURN3_STAGE7_REFERENCE_MIN_MARGIN
from .evaluate_hu_turn2_stage8_seat_swap import (
    DEFAULT_CALIBRATION_VALUES,
    DEFAULT_STAGE3_REFERENCE,
    DEFAULT_STAGE7_MODEL,
    DEFAULT_T2_STAGE8_MODEL,
    DEFAULT_T3_CONTINUATION,
    ModelParts,
    T3ContinuationMode,
    _t3_continuation_policy_name,
    _t3_policy_kwargs,
    load_parts,
    make_baseline_policy,
    parse_seeds,
    summarize_values,
    write_csv,
    write_jsonl,
)
from .evaluate_matchups import trace_hand
from .evaluator import score_board
from .final_turn_decision_cache import FinalTurnDecisionCache
from .hu_belief import sample_hidden_card_particles
from .hu_infoset import ActorObservation
from .hu_turn2_stage8_runtime import (
    _safe_predictions,
    hu_turn2_policy_sample,
    load_hu_turn2_stage8_model,
    sigmoid,
)
from .hu_turn2_teacher_data import (
    _t3_continuation_metadata as teacher_t3_continuation_metadata,
    _t3_continuation_policy_name as teacher_t3_continuation_policy_name,
    evaluate_hu_turn2_actions,
)
from .hu_turn3_batch_continuation import (
    HuTurn3ActionCache,
    HuTurn3DecisionCache,
    HuTurn3Stage3ReferenceCache,
    HuTurn3Stage7BatchConfig,
    Stage3StateFeatureCache,
)
from .hu_turn3_model import load_hu_action_value_model
from .hu_infoset import card_free_metadata
from .play_ai import _prediction_thread_context
from .policy import RegularAiPolicy, action_to_json, board_to_json, policy_sample
from .state import Board
from .teacher import DEFAULT_FL_EV
from .train_hu_turn2_stage8c_risk_head import (
    FEATURE_MODE_RUNTIME_META_ONLY,
    TARGET_MODE_LOCAL_EV_NEGATIVE,
    risk_feature_vector_for_row,
)
from .train_torch_action_value import select_device
from .turn3_model import _build_torch_mlp, load_action_value_model


DEFAULT_OUTPUT_DIR = Path("outputs/evals/hu_turn2_stage8b_topk_mc_rerank")
DEFAULT_TOPK_RERANK_CONFIGS = (
    "k3/mc64/d0.25/se0/pd0/seat=first,"
    "k5/mc64/d0.25/se0/pd0/seat=first"
)
TOPK_PERFORMANCE_METRIC_SOURCE = "realized_fired_whole_game_delta"
TOPK_PER_FIRE_PERFORMANCE_COLUMN = "per_override_delta_mean"
TOPK_HAND_EV_PERFORMANCE_COLUMN = "estimated_ev_per_hand"
TOPK_RERANK_DELTA_METRIC_ROLE = "gate_diagnostic_only"
TOPK_CONFIRM_DELTA_METRIC_ROLE = "gate_diagnostic_only"
TOPK_CONFIRM_DELTA_PERFORMANCE_CLAIM_ALLOWED = False
NON_FIRED_CANCELLATION_REQUIRED_FOR_CONDITIONAL_GO = True
CONDITIONAL_REALIZED_PER_FIRE_REQUIRED_FOR_CONDITIONAL_GO = True
CONFIRM_RERANK_DIAGNOSTICS_USED_FOR_CONDITIONAL_GO = False


@dataclass
class LocalEvRiskScorer:
    """Runtime scorer for the Stage8c local-EV-negative veto head.

    The score is a veto input only. It is not a realized performance metric.
    """

    path: Path
    state_dict: dict[str, Any]
    feature_dim: int
    hidden_layer_sizes: tuple[int, ...]
    dropout: float
    feature_mode: str
    target_mode: str
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    device: str = "auto"
    _torch: Any = field(default=None, init=False, repr=False)
    _net: Any = field(default=None, init=False, repr=False)
    _net_device: str | None = field(default=None, init=False, repr=False)

    @classmethod
    def load(cls, path: str | Path, *, device: str = "auto") -> "LocalEvRiskScorer":
        import torch

        model_path = Path(path)
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
        feature_mode = str(payload.get("feature_mode") or FEATURE_MODE_RUNTIME_META_ONLY)
        target_mode = str(payload.get("target_mode") or TARGET_MODE_LOCAL_EV_NEGATIVE)
        feature_dim = int(payload["feature_dim"])
        feature_mean = np.asarray(payload["feature_mean"], dtype=np.float32)
        feature_scale = np.asarray(payload["feature_scale"], dtype=np.float32)
        if feature_mean.shape[0] != feature_dim or feature_scale.shape[0] != feature_dim:
            raise ValueError("local EV risk scorer feature stats do not match feature_dim")
        return cls(
            path=model_path,
            state_dict=payload["state_dict"],
            feature_dim=feature_dim,
            hidden_layer_sizes=tuple(int(size) for size in payload.get("hidden_layer_sizes", ())),
            dropout=float(payload.get("dropout", 0.0)),
            feature_mode=feature_mode,
            target_mode=target_mode,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            device=device,
        )

    def predict_probability(self, row: dict[str, Any]) -> float:
        return self.predict_probabilities([row])[0]

    def predict_probabilities(self, rows: list[dict[str, Any]]) -> list[float]:
        if not rows:
            return []
        features = np.vstack(
            [risk_feature_vector_for_row(row, self.feature_mode) for row in rows]
        ).astype(np.float32, copy=False)
        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Stage8c risk feature dimension mismatch: got {features.shape[1]}, expected {self.feature_dim}"
            )
        torch = self._get_torch()
        device = self._select_device(torch)
        net = self._get_net(torch, device)
        normalized = (features - self.feature_mean) / self.feature_scale
        inference_guard = getattr(torch, "inference_mode", torch.no_grad)
        with inference_guard():
            tensor = torch.from_numpy(normalized.astype(np.float32, copy=False)).to(device)
            logits = net(tensor).squeeze(-1)
            probabilities = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1).astype(float)
        if probabilities.shape[0] != len(rows):
            raise ValueError("Stage8c risk scorer returned an unexpected probability count")
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("Stage8c risk scorer returned a non-finite probability")
        return [float(value) for value in probabilities]

    def _get_torch(self) -> Any:
        if self._torch is None:
            import torch

            self._torch = torch
        return self._torch

    def _select_device(self, torch: Any) -> str:
        return select_device(torch, self.device)

    def _get_net(self, torch: Any, device: str) -> Any:
        if self._net is None or self._net_device != device:
            net = _build_torch_mlp(
                torch,
                self.feature_dim,
                self.hidden_layer_sizes,
                self.dropout,
                output_dim=1,
            )
            net.load_state_dict(self.state_dict)
            net.to(device)
            net.eval()
            self._net = net
            self._net_device = device
        return self._net


def _board_from_json(data: dict[str, Any]) -> Board:
    return Board.from_rows(
        top=tuple(data.get("top") or ()),
        middle=tuple(data.get("middle") or ()),
        bottom=tuple(data.get("bottom") or ()),
    )


def _fl_value(score: Any) -> float:
    if score.busted:
        return 0.0
    return float(DEFAULT_FL_EV.get(score.fl_entry.card_count, 0.0))


def _line_value(own_value: tuple[int, tuple[int, ...]], opp_value: tuple[int, tuple[int, ...]]) -> int:
    if own_value > opp_value:
        return 1
    if own_value < opp_value:
        return -1
    return 0


def _terminal_breakdown(hero_board_json: dict[str, Any], opponent_board_json: dict[str, Any]) -> dict[str, Any]:
    hero_board = _board_from_json(hero_board_json)
    opponent_board = _board_from_json(opponent_board_json)
    hero = score_board(hero_board.top, hero_board.middle, hero_board.bottom)
    opponent = score_board(opponent_board.top, opponent_board.middle, opponent_board.bottom)
    hero_royalty = 0 if hero.busted else hero.total_royalty
    opponent_royalty = 0 if opponent.busted else opponent.total_royalty
    hero_fl_value = _fl_value(hero)
    opponent_fl_value = _fl_value(opponent)
    line_results = {"top": 0, "middle": 0, "bottom": 0}
    line_score_delta = 0
    scoop_delta = 0
    foul_delta = 0
    if hero.busted and opponent.busted:
        terminal = 0.0
    elif hero.busted:
        foul_delta = -6
        terminal = float(foul_delta - opponent_royalty - opponent_fl_value)
    elif opponent.busted:
        foul_delta = 6
        terminal = float(foul_delta + hero_royalty + hero_fl_value)
    else:
        line_results = {
            "top": _line_value(hero.top_value, opponent.top_value),
            "middle": _line_value(hero.middle_value, opponent.middle_value),
            "bottom": _line_value(hero.bottom_value, opponent.bottom_value),
        }
        line_score_delta = sum(line_results.values())
        scoop_delta = 3 if line_score_delta == 3 else (-3 if line_score_delta == -3 else 0)
        terminal = float(
            line_score_delta
            + scoop_delta
            + hero_royalty
            - opponent_royalty
            + hero_fl_value
            - opponent_fl_value
        )
    return {
        "final_board_hero": hero_board_json,
        "final_board_opponent": opponent_board_json,
        "hero_foul": bool(hero.busted),
        "opponent_foul": bool(opponent.busted),
        "hero_royalty": int(hero_royalty),
        "opponent_royalty": int(opponent_royalty),
        "royalty_delta": int(hero_royalty - opponent_royalty),
        "hero_fl_entry": bool((not hero.busted) and hero.fl_entry.qualifies),
        "hero_fl_card_count": int(0 if hero.busted else hero.fl_entry.card_count),
        "hero_fl_entry_type": None if hero.busted else hero.fl_entry.entry_type,
        "hero_fl_value": float(hero_fl_value),
        "hero_fl_stay": False,
        "opponent_fl_entry": bool((not opponent.busted) and opponent.fl_entry.qualifies),
        "opponent_fl_card_count": int(0 if opponent.busted else opponent.fl_entry.card_count),
        "opponent_fl_entry_type": None if opponent.busted else opponent.fl_entry.entry_type,
        "opponent_fl_value": float(opponent_fl_value),
        "opponent_fl_stay": False,
        "fl_delta": float(hero_fl_value - opponent_fl_value),
        "line_results": line_results,
        "line_score_delta": int(line_score_delta),
        "scoop_delta": int(scoop_delta),
        "foul_delta": int(foul_delta),
        "terminal_score": terminal,
    }


def _trace_outcome_for_player(hand: dict[str, Any], player: int) -> dict[str, Any]:
    hero_key = f"p{player}"
    opponent_key = f"p{1 - player}"
    final = hand.get("final") or {}
    if hero_key not in final or opponent_key not in final:
        return {}
    return _terminal_breakdown(final[hero_key], final[opponent_key])


def _seat_for_player(player: int) -> str:
    return "first" if player == 0 else "second"


def _compact_stage7_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "turn": record.get("turn") or record.get("street") or "T3",
        "seat": record.get("seat"),
        "override_fired": bool(record.get("override_fired")),
        "no_override_reason": record.get("no_override_reason", ""),
        "reference_margin": record.get("reference_margin"),
        "stage7_predicted_margin": record.get("stage7_predicted_margin"),
        "model_score": record.get("model_score"),
        "legality_check_result": record.get("legality_check_result"),
        "stage3_action": record.get("stage3_action"),
        "stage7_action": record.get("stage7_action"),
        "final_action": record.get("final_action"),
    }


def _trace_t3_decision_summary(
    hand: dict[str, Any],
    player: int,
    t3_decision_log: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    seat = _seat_for_player(player)
    turns = [turn for turn in hand.get("turns", []) if turn.get("turn") == "T3"]
    player_turns = [turn for turn in turns if int(turn.get("player", -1)) == player]
    opponent_turns = [turn for turn in turns if int(turn.get("player", -1)) == 1 - player]
    stage7_records = [
        _compact_stage7_record(record)
        for record in t3_decision_log
        if str(record.get("seat", "")) == seat and str(record.get("turn") or record.get("street") or "") == "T3"
    ]
    return {
        "player": player,
        "seat": seat,
        "player_t3_turn": player_turns[0] if player_turns else None,
        "opponent_t3_turn": opponent_turns[0] if opponent_turns else None,
        "stage7_record_count": len(stage7_records),
        "stage7_override_fired": any(record.get("override_fired") for record in stage7_records),
        "stage7_records": stage7_records,
    }


def _summary_override_fired(summary: dict[str, Any] | None) -> bool:
    if not isinstance(summary, dict):
        return False
    return bool(summary.get("stage7_override_fired"))


def _add_prefixed_outcome(row: dict[str, Any], prefix: str, outcome: dict[str, Any]) -> None:
    for key, value in outcome.items():
        row[f"{prefix}_{key}"] = value


def _add_prefixed_t3_summary(row: dict[str, Any], prefix: str, summary: dict[str, Any]) -> None:
    row[f"{prefix}_t3_decision_summary"] = summary
    row[f"{prefix}_downstream_override_fired"] = _summary_override_fired(summary)


def _add_t3_summary_delta(row: dict[str, Any]) -> None:
    candidate_summary = row.get("candidate_t3_decision_summary")
    baseline_summary = row.get("baseline_t3_decision_summary")
    if not isinstance(candidate_summary, dict):
        return
    if isinstance(baseline_summary, dict):
        row["t3_decision_summary"] = {
            "candidate": candidate_summary,
            "baseline": baseline_summary,
        }
        row["downstream_override_fired"] = bool(
            _summary_override_fired(candidate_summary) or _summary_override_fired(baseline_summary)
        )
    else:
        row["t3_decision_summary"] = {"candidate": candidate_summary}
        row["downstream_override_fired"] = _summary_override_fired(candidate_summary)


def _paired_delta_summary_for_action(
    sample: dict[str, Any] | None,
    candidate_index: int | None,
    baseline_index: int,
) -> dict[str, Any] | None:
    if sample is None or candidate_index is None:
        return None
    for item in sample.get("paired_delta_by_action") or ():
        if int(item.get("candidate_index", -1)) == int(candidate_index) and int(
            item.get("baseline_index", -1)
        ) == int(baseline_index):
            return dict(item)
    if (
        int(sample.get("paired_delta_candidate_index", -1)) == int(candidate_index)
        and int(sample.get("paired_delta_baseline_index", -1)) == int(baseline_index)
    ):
        return {
            "candidate_index": int(candidate_index),
            "baseline_index": int(baseline_index),
            "count": int(sample.get("paired_delta_count", 0) or 0),
            "mean": float(sample.get("paired_delta_mean", 0.0) or 0.0),
            "standard_error": float(sample.get("paired_delta_standard_error", 0.0) or 0.0),
            "std": float(sample.get("paired_delta_std", 0.0) or 0.0),
            "min": float(sample.get("paired_delta_min", 0.0) or 0.0),
            "p01": float(sample.get("paired_delta_p01", sample.get("paired_delta_p05", 0.0)) or 0.0),
            "p05": float(sample.get("paired_delta_p05", 0.0) or 0.0),
            "p25": float(sample.get("paired_delta_p25", 0.0) or 0.0),
            "p50": float(sample.get("paired_delta_p50", 0.0) or 0.0),
            "p75": float(sample.get("paired_delta_p75", 0.0) or 0.0),
            "p95": float(sample.get("paired_delta_p95", 0.0) or 0.0),
            "p99": float(sample.get("paired_delta_p99", sample.get("paired_delta_p95", 0.0)) or 0.0),
            "max": float(sample.get("paired_delta_max", 0.0) or 0.0),
            "lt0_rate": float(sample.get("paired_delta_lt0_rate", 0.0) or 0.0),
            "le_neg6_rate": float(sample.get("paired_delta_le_neg6_rate", 0.0) or 0.0),
            "le_neg12_rate": float(sample.get("paired_delta_le_neg12_rate", 0.0) or 0.0),
            "le_neg20_rate": float(sample.get("paired_delta_le_neg20_rate", 0.0) or 0.0),
        }
    return None


def _select_paired_future_delta_summary(
    confirm_summary: dict[str, Any] | None,
    stage_a_summary: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, str]:
    if confirm_summary is not None:
        return confirm_summary, "confirm"
    if stage_a_summary is not None:
        return stage_a_summary, "stage_a"
    return None, ""


def _add_outcome_delta(row: dict[str, Any]) -> None:
    candidate_score = row.get("candidate_terminal_score")
    baseline_score = row.get("baseline_terminal_score")
    if candidate_score in (None, "") or baseline_score in (None, ""):
        return
    for key in (
        "terminal_score",
        "royalty_delta",
        "fl_delta",
        "line_score_delta",
        "scoop_delta",
        "foul_delta",
        "hero_royalty",
        "opponent_royalty",
        "hero_fl_value",
        "opponent_fl_value",
    ):
        candidate_value = row.get(f"candidate_{key}")
        baseline_value = row.get(f"baseline_{key}")
        if candidate_value in (None, "") or baseline_value in (None, ""):
            continue
        row[f"{key}_vs_baseline"] = float(candidate_value) - float(baseline_value)


@dataclass(frozen=True)
class TopKMcRerankConfig:
    top_k: int
    mc_samples: int
    min_delta: float
    se_multiplier: float = 0.0
    confirm_mc_samples: int = 0
    confirm_se_multiplier: float = 0.0
    max_confirm_se: float | None = None
    min_confirm_delta: float | None = None
    first_min_confirm_delta: float | None = None
    second_min_confirm_delta: float | None = None
    allowed_seats: tuple[str, ...] = ()
    candidate_ev_rank_max: int | None = None
    min_gate_probability: float | None = None
    min_predicted_delta: float | None = None
    topk_score: str = "delta"

    @property
    def config_id(self) -> str:
        parts = [
            f"k{self.top_k}",
            f"mc{self.mc_samples}",
            f"d{self.min_delta:g}",
            f"se{self.se_multiplier:g}",
        ]
        if self.confirm_mc_samples > 0:
            parts.append(f"confirm{self.confirm_mc_samples:g}")
            parts.append(f"cse{self.confirm_se_multiplier:g}")
        if self.max_confirm_se is not None:
            parts.append(f"csemax{self.max_confirm_se:g}")
        if self.min_confirm_delta is not None:
            parts.append(f"cd{self.min_confirm_delta:g}")
        if self.first_min_confirm_delta is not None:
            parts.append(f"fcd{self.first_min_confirm_delta:g}")
        if self.second_min_confirm_delta is not None:
            parts.append(f"scd{self.second_min_confirm_delta:g}")
        if self.min_gate_probability is not None:
            parts.append(f"g{self.min_gate_probability:g}")
        if self.min_predicted_delta is not None:
            parts.append(f"pd{self.min_predicted_delta:g}")
        if self.allowed_seats:
            parts.append("seat" + "-".join(self.allowed_seats))
        if self.candidate_ev_rank_max is not None:
            parts.append(f"rank{self.candidate_ev_rank_max:g}")
        if self.topk_score != "delta":
            parts.append(f"by{self.topk_score}")
        return "_".join(parts)

    def min_confirm_delta_for_seat(self, seat: str) -> float | None:
        if seat == "first" and self.first_min_confirm_delta is not None:
            return self.first_min_confirm_delta
        if seat == "second" and self.second_min_confirm_delta is not None:
            return self.second_min_confirm_delta
        return self.min_confirm_delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games-per-seed", type=int, default=100)
    parser.add_argument(
        "--target-realized-overrides-per-seed",
        type=int,
        default=0,
        help=(
            "Optional early-stop target for realized fired decisions per config/seed. "
            "--games-per-seed remains the hard maximum paired-seed budget."
        ),
    )
    parser.add_argument(
        "--target-risk-vetoes-per-seed",
        type=int,
        default=0,
        help=(
            "Optional early-stop target for Stage8c risk-vetoed decisions per config/seed. "
            "Use this for fixed-threshold veto validation; --games-per-seed remains the hard maximum budget."
        ),
    )
    parser.add_argument(
        "--target-fire-selector-rejections-per-seed",
        type=int,
        default=0,
        help=(
            "Optional early-stop target for Stage8c fire-selector rejected decisions per config/seed. "
            "Use this for Stage9d distillation data collection; --games-per-seed remains the hard maximum budget."
        ),
    )
    parser.add_argument("--seeds", default="2026062101,2026062102,2026062103")
    parser.add_argument("--seed-stride", type=int, default=1_000_000)
    parser.add_argument(
        "--configs",
        default=DEFAULT_TOPK_RERANK_CONFIGS,
        help=(
            "Comma-separated configs. Example: "
            "k3/mc64/d0.25/se1.5/pd0/seat=first/rank3/g0.9/bydelta"
        ),
    )
    parser.add_argument("--opening-model", type=Path, default=Path("models/opening_stage7_torch_wide.pt"))
    parser.add_argument("--turn1-model", type=Path, default=Path("models/turn1_stage6_torch_wide.pt"))
    parser.add_argument("--turn2-baseline-model", type=Path, default=Path("models/turn2_stage8.pkl"))
    parser.add_argument("--turn3-model", type=Path, default=Path("models/turn3_stage6.pkl"))
    parser.add_argument("--hu-turn3-stage7-model", type=Path, default=DEFAULT_STAGE7_MODEL)
    parser.add_argument("--hu-turn3-reference-model", type=Path, default=DEFAULT_STAGE3_REFERENCE)
    parser.add_argument(
        "--t3-continuation",
        choices=("stage3_reference_default", "stage7_m5_r10"),
        default=DEFAULT_T3_CONTINUATION,
        help=(
            "T3 continuation used inside TopK MC rollouts. Hidden-discard default "
            "is stage3_reference_default; stage7_m5_r10 must be opted in explicitly."
        ),
    )
    parser.add_argument("--hu-turn2-stage8b-model", type=Path, default=DEFAULT_T2_STAGE8_MODEL)
    parser.add_argument(
        "--allow-missing-stage8b-model-fallback",
        action="store_true",
        help=(
            "Preflight-only safety check: if the Stage8b/Stage9f candidate "
            "model cannot be loaded, continue with the candidate model set to "
            "None so every HU T2 decision falls back to the baseline policy. "
            "Without this flag, model load failure remains fatal."
        ),
    )
    parser.add_argument("--calibration-values", type=Path, default=DEFAULT_CALIBRATION_VALUES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--prediction-threads", type=int, default=1)
    parser.add_argument("--opening-lookahead-samples", type=int, default=64)
    parser.add_argument("--batched-continuation-batch-size", type=int, default=8192)
    parser.add_argument("--stage3-feature-encoder-mode", default="rust_direct")
    parser.add_argument(
        "--local-ev-risk-model",
        type=Path,
        help=(
            "Deprecated alias for --stage8c-risk-model."
        ),
    )
    parser.add_argument(
        "--local-ev-risk-threshold",
        type=float,
        default=0.8,
        help="Deprecated alias for --stage8c-risk-threshold.",
    )
    parser.add_argument(
        "--stage8c-risk-model",
        type=Path,
        help=(
            "Optional Stage8c risk head. When set, the model is used only as a "
            "post-confirm veto before firing an override."
        ),
    )
    parser.add_argument(
        "--stage8c-risk-threshold",
        type=float,
        help="Veto the confirmed TopK action when Stage8c risk probability is at or above this threshold.",
    )
    parser.add_argument("--stage8c-risk-rank-min", type=int, help="Only apply the Stage8c risk veto at or above this EV rank.")
    parser.add_argument("--stage8c-risk-rank-max", type=int, help="Only apply the Stage8c risk veto at or below this EV rank.")
    parser.add_argument(
        "--stage8c-risk-audit-only",
        action="store_true",
        help=(
            "Score and log would-veto decisions without blocking the override. "
            "Use this to estimate fresh realized veto utility."
        ),
    )
    parser.add_argument(
        "--stage8c-fire-selector-model",
        type=Path,
        help=(
            "Optional Stage8c topk_confirm_fire head. This is a pre-confirm selector, "
            "not a risk veto and not a production gate."
        ),
    )
    parser.add_argument(
        "--stage8c-fire-selector-threshold",
        type=float,
        default=0.7,
        help="Keep TopK candidates whose Stage8c fire-selector probability is at or above this threshold.",
    )
    parser.add_argument(
        "--stage8c-fire-selector-audit-only",
        action="store_true",
        help="Score and log fire-selector probabilities without filtering candidates before MC.",
    )
    parser.add_argument(
        "--stage8c-fire-selector-direct-fire",
        action="store_true",
        help=(
            "Stage9e experimental mode: after the fire selector filters TopK candidates, "
            "choose the highest selector-probability candidate directly and skip MC/confirm. "
            "This is a latency/quality experiment, not a production default."
        ),
    )
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--write-decision-log", action="store_true")
    return parser.parse_args()


def parse_topk_configs(value: str) -> list[TopKMcRerankConfig]:
    configs: list[TopKMcRerankConfig] = []
    for item in [part.strip() for part in value.split(",") if part.strip()]:
        top_k: int | None = None
        mc_samples: int | None = None
        min_delta: float | None = None
        se_multiplier = 0.0
        confirm_mc_samples = 0
        confirm_se_multiplier = 0.0
        max_confirm_se: float | None = None
        min_confirm_delta: float | None = None
        first_min_confirm_delta: float | None = None
        second_min_confirm_delta: float | None = None
        allowed_seats: tuple[str, ...] = ()
        candidate_ev_rank_max: int | None = None
        min_gate_probability: float | None = None
        min_predicted_delta: float | None = None
        topk_score = "delta"
        for token in item.split("/"):
            key, sep, raw = token.partition("=")
            key = key.strip().lower()
            raw_value = raw.strip().lower() if sep else key
            if key.startswith("k") and not sep:
                top_k = int(float(key[1:]))
            elif key == "k" and sep:
                top_k = int(float(raw_value))
            elif key.startswith("mc") and not sep:
                mc_samples = int(float(key[2:]))
            elif key == "mc" and sep:
                mc_samples = int(float(raw_value))
            elif key.startswith("d") and not sep:
                min_delta = float(key[1:])
            elif key in {"d", "delta", "min_delta"} and sep:
                min_delta = float(raw_value)
            elif key.startswith("se") and not sep:
                se_multiplier = float(key[2:])
            elif key in {"se", "se_multiplier"} and sep:
                se_multiplier = float(raw_value)
            elif key.startswith("confirm") and not sep:
                confirm_mc_samples = int(float(key[7:]))
            elif key.startswith("cmc") and not sep:
                confirm_mc_samples = int(float(key[3:]))
            elif key in {"confirm", "confirm_mc", "confirm_mc_samples", "stage_b_mc"} and sep:
                confirm_mc_samples = int(float(raw_value))
            elif key.startswith("cse") and not sep:
                if key.startswith("csemax"):
                    max_confirm_se = float(key[6:])
                else:
                    confirm_se_multiplier = float(key[3:])
            elif key in {"confirm_se", "confirm_se_multiplier", "stage_b_se"} and sep:
                confirm_se_multiplier = float(raw_value)
            elif key in {"csemax", "confirm_se_max", "max_confirm_se", "stage_b_se_max"} and sep:
                max_confirm_se = float(raw_value)
            elif key.startswith("cd") and not sep:
                min_confirm_delta = float(key[2:])
            elif key in {"cd", "confirm_delta", "min_confirm_delta"} and sep:
                min_confirm_delta = float(raw_value)
            elif key.startswith("fcd") and not sep:
                first_min_confirm_delta = float(key[3:])
            elif key in {"fcd", "first_cd", "first_confirm_delta", "first_min_confirm_delta"} and sep:
                first_min_confirm_delta = float(raw_value)
            elif key.startswith("scd") and not sep:
                second_min_confirm_delta = float(key[3:])
            elif key in {"scd", "second_cd", "second_confirm_delta", "second_min_confirm_delta"} and sep:
                second_min_confirm_delta = float(raw_value)
            elif key.startswith("g") and not sep:
                min_gate_probability = float(key[1:])
            elif key in {"g", "gate", "min_gate_probability"} and sep:
                min_gate_probability = float(raw_value)
            elif key.startswith("pd") and not sep:
                min_predicted_delta = float(key[2:])
            elif key.startswith("pred") and not sep:
                min_predicted_delta = float(key[4:])
            elif key in {"pd", "pred", "predicted_delta", "min_predicted_delta"} and sep:
                min_predicted_delta = float(raw_value)
            elif key.startswith("rank") and not sep:
                candidate_ev_rank_max = int(float(key[4:]))
            elif key in {"rank", "rankguard", "candidate_ev_rank_max"} and sep:
                candidate_ev_rank_max = int(float(raw_value))
            elif key in {"seat", "seats", "allowed_seats"} and sep:
                if raw_value in {"all", "any", "*"}:
                    allowed_seats = ()
                else:
                    seats = tuple(seat for seat in raw_value.replace("|", "+").split("+") if seat)
                    invalid = [seat for seat in seats if seat not in {"first", "second"}]
                    if invalid:
                        raise ValueError(f"invalid seat filter in {item}: {invalid}")
                    allowed_seats = seats
            elif key.startswith("seat") and not sep and key not in {"seat"}:
                raw_seats = key[4:]
                allowed_seats = tuple(seat for seat in raw_seats.replace("-", "+").split("+") if seat)
            elif key.startswith("by") and not sep:
                topk_score = key[2:]
            elif key in {"by", "topk_score"} and sep:
                topk_score = raw_value
            else:
                raise ValueError(f"unknown TopK rerank config token in {item}: {token}")
        if top_k is None or mc_samples is None or min_delta is None:
            raise ValueError(f"config must include k, mc, and d: {item}")
        if top_k <= 0 or mc_samples <= 0:
            raise ValueError(f"top_k and mc_samples must be positive: {item}")
        if topk_score not in {"delta", "ev", "gate", "gate_delta"}:
            raise ValueError(f"unsupported topk_score in {item}: {topk_score}")
        configs.append(
            TopKMcRerankConfig(
                top_k=top_k,
                mc_samples=mc_samples,
                min_delta=min_delta,
                se_multiplier=se_multiplier,
                confirm_mc_samples=confirm_mc_samples,
                confirm_se_multiplier=confirm_se_multiplier,
                max_confirm_se=max_confirm_se,
                min_confirm_delta=min_confirm_delta,
                first_min_confirm_delta=first_min_confirm_delta,
                second_min_confirm_delta=second_min_confirm_delta,
                allowed_seats=allowed_seats,
                candidate_ev_rank_max=candidate_ev_rank_max,
                min_gate_probability=min_gate_probability,
                min_predicted_delta=min_predicted_delta,
                topk_score=topk_score,
            )
        )
    if not configs:
        raise ValueError("at least one TopK rerank config is required")
    return configs


class HuTurn2Stage8bTopKMcRerankPolicy(RegularAiPolicy):
    hu_turn2_stage8b_model: object | None
    topk_rerank_config: TopKMcRerankConfig
    topk_decision_log: list[dict[str, Any]] | None
    topk_context: dict[str, Any]

    def __init__(
        self,
        *,
        hu_turn2_stage8b_model: object | None,
        topk_rerank_config: TopKMcRerankConfig,
        topk_decision_log: list[dict[str, Any]] | None = None,
        topk_context: dict[str, Any] | None = None,
        batched_continuation_batch_size: int = 8192,
        stage3_feature_encoder_mode: str = "rust_direct",
        t3_continuation: T3ContinuationMode = DEFAULT_T3_CONTINUATION,
        local_ev_risk_scorer: LocalEvRiskScorer | None = None,
        local_ev_risk_threshold: float = 0.8,
        local_ev_risk_rank_min: int | None = None,
        local_ev_risk_rank_max: int | None = None,
        local_ev_risk_audit_only: bool = False,
        stage8c_fire_selector_scorer: LocalEvRiskScorer | None = None,
        stage8c_fire_selector_threshold: float = 0.7,
        stage8c_fire_selector_audit_only: bool = False,
        stage8c_fire_selector_direct_fire: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.hu_turn2_stage8b_model = hu_turn2_stage8b_model
        self.topk_rerank_config = topk_rerank_config
        self.topk_decision_log = topk_decision_log
        self.topk_context = card_free_metadata(topk_context)
        self.batched_continuation_batch_size = batched_continuation_batch_size
        self.t3_continuation = t3_continuation
        self.local_ev_risk_scorer = local_ev_risk_scorer
        self.local_ev_risk_threshold = float(local_ev_risk_threshold)
        self.local_ev_risk_rank_min = local_ev_risk_rank_min
        self.local_ev_risk_rank_max = local_ev_risk_rank_max
        self.local_ev_risk_audit_only = bool(local_ev_risk_audit_only)
        self.stage8c_fire_selector_scorer = stage8c_fire_selector_scorer
        self.stage8c_fire_selector_threshold = float(stage8c_fire_selector_threshold)
        self.stage8c_fire_selector_audit_only = bool(stage8c_fire_selector_audit_only)
        self.stage8c_fire_selector_direct_fire = bool(stage8c_fire_selector_direct_fire)
        self._batched_config = HuTurn3Stage7BatchConfig(
            stage7_enabled=t3_continuation == "stage7_m5_r10",
            hu_turn3_min_margin=DEFAULT_HU_TURN3_STAGE7_MIN_MARGIN,
            hu_turn3_reference_min_margin=(
                DEFAULT_HU_TURN3_STAGE7_REFERENCE_MIN_MARGIN
                if t3_continuation == "stage7_m5_r10"
                else 0.0
            ),
            batch_size=batched_continuation_batch_size,
            stage3_feature_encoder_mode=stage3_feature_encoder_mode,
        )
        self._t3_decision_cache = HuTurn3DecisionCache(max_size=200_000)
        self._t3_reference_cache = HuTurn3Stage3ReferenceCache(max_size=200_000)
        self._t3_state_feature_cache = Stage3StateFeatureCache(max_size=200_000)
        self._t3_action_cache = HuTurn3ActionCache(max_size=200_000)
        self._final_turn_cache = FinalTurnDecisionCache(max_size=200_000)

    def choose_action(
        self,
        board: Board,
        dealt_cards: Iterable[str],
        *,
        dead_cards: Iterable[str] = (),
        opponent_board: Board | None = None,
        hand_id: str | int | None = None,
        game_id: str | int | None = None,
        decision_seed: int | None = None,
        street: str | None = None,
    ) -> Action:
        if board.card_count() == 7 and opponent_board is not None and self.turn2_model is not None:
            action = self._choose_hu_turn2_topk_mc_action(
                board,
                tuple(dealt_cards),
                dead_cards=tuple(dead_cards),
                opponent_board=opponent_board,
                hand_id=hand_id,
                game_id=game_id,
                decision_seed=decision_seed,
                street=street,
            )
            if action is not None:
                return action
        return super().choose_action(
            board,
            dealt_cards,
            dead_cards=dead_cards,
            opponent_board=opponent_board,
            hand_id=hand_id,
            game_id=game_id,
            decision_seed=decision_seed,
            street=street,
        )

    def _choose_hu_turn2_topk_mc_action(
        self,
        board: Board,
        dealt: tuple[str, ...],
        *,
        dead_cards: tuple[str, ...],
        opponent_board: Board,
        hand_id: str | int | None,
        game_id: str | int | None,
        decision_seed: int | None,
        street: str | None,
    ) -> Action | None:
        started_at = time.perf_counter()
        config = self.topk_rerank_config
        actions = generate_turn_actions(board, dealt)
        if not actions:
            return None

        fallback_sample = policy_sample(board, dealt, actions)
        baseline_predictions, baseline_reason = _safe_predictions(self.turn2_model, fallback_sample, len(actions))
        if baseline_predictions is None:
            return None
        baseline_values = baseline_predictions.reshape(-1)
        baseline_index = canonical_argmax_index(baseline_values, actions)
        final_index = baseline_index
        no_override_reason = ""
        stage8b_top1_index: int | None = None
        rerank_best_index: int | None = None
        rerank_indices: list[int] = [baseline_index]
        predictions: np.ndarray | None = None
        predicted_delta: float | None = None
        predicted_ev: float | None = None
        gate_probability: float | None = None
        candidate_ev_rank: int | None = None
        rerank_delta: float | None = None
        rerank_delta_se: float | None = None
        rerank_best_ev: float | None = None
        rerank_baseline_ev: float | None = None
        rerank_latency_ms = 0.0
        stage_a_delta: float | None = None
        stage_a_delta_se: float | None = None
        stage_a_best_ev: float | None = None
        stage_a_baseline_ev: float | None = None
        stage_a_common_random_future_digest = ""
        stage_a_paired_delta_summary: dict[str, Any] | None = None
        confirm_delta: float | None = None
        confirm_delta_se: float | None = None
        confirm_delta_count = 0
        confirm_candidate_ev: float | None = None
        confirm_baseline_ev: float | None = None
        confirm_latency_ms = 0.0
        confirm_common_random_future_digest = ""
        confirm_paired_delta_summary: dict[str, Any] | None = None
        local_ev_risk_probability: float | None = None
        local_ev_risk_would_veto = False
        local_ev_risk_vetoed = False
        local_ev_risk_rank_guard_passed: bool | None = None
        stage8c_fire_selector_candidates: list[dict[str, Any]] = []
        stage8c_fire_selector_probability: float | None = None
        stage8c_fire_selector_max_probability: float | None = None
        stage8c_fire_selector_probability_by_index: dict[int, float] = {}
        stage8c_fire_selector_passed_count = 0
        stage8c_fire_selector_evaluated_count = 0
        stage8c_fire_selector_direct_fire_used = False
        stage8c_fire_selector_direct_fire_candidate_count = 0
        stage8c_fire_selector_direct_fire_probability: float | None = None
        evaluated_action_count = 0
        common_random_future_digest = ""

        if config.allowed_seats and self.seat not in config.allowed_seats:
            no_override_reason = "seat_not_allowed"
        elif self.hu_turn2_stage8b_model is None:
            no_override_reason = "model_load_failed"
        else:
            to_act_order = "second" if opponent_board.card_count() > board.card_count() else "first"
            sample = hu_turn2_policy_sample(
                board,
                dealt,
                actions,
                opponent_board=opponent_board,
                dead_cards=dead_cards,
                seat=self.seat,
                to_act_order=to_act_order,
            )
            predictions, prediction_reason = _safe_predictions(self.hu_turn2_stage8b_model, sample, len(actions))
            if predictions is None or predictions.ndim != 2 or predictions.shape[1] < 5:
                no_override_reason = prediction_reason or "illegal_candidate"
            else:
                stage8b_top1_index = canonical_argmax_index(predictions[:, 1], actions)
                ev_order = canonical_descending_indices(predictions[:, 0], actions)
                ev_ranks = np.empty(len(actions), dtype=np.int32)
                for rank, action_index in enumerate(ev_order, start=1):
                    ev_ranks[int(action_index)] = rank
                gate_values = np.asarray([sigmoid(float(value)) for value in predictions[:, 4]], dtype=np.float64)
                order = self._topk_order(predictions, gate_values, actions)
                selected: list[int] = []
                for index in order:
                    action_index = int(index)
                    if action_index == baseline_index:
                        continue
                    if (
                        config.min_predicted_delta is not None
                        and float(predictions[action_index, 1]) < config.min_predicted_delta
                    ):
                        continue
                    if config.candidate_ev_rank_max is not None and int(ev_ranks[action_index]) > config.candidate_ev_rank_max:
                        continue
                    if config.min_gate_probability is not None and float(gate_values[action_index]) < config.min_gate_probability:
                        continue
                    selected.append(action_index)
                    if len(selected) >= config.top_k:
                        break
                if not selected:
                    no_override_reason = "topk_empty"
                else:
                    if self.stage8c_fire_selector_scorer is not None:
                        filtered: list[int] = []
                        prediction_failed = False
                        selector_rows = [
                            self._stage8c_candidate_score_row(
                                board=board,
                                opponent_board=opponent_board,
                                dealt=dealt,
                                dead_cards=dead_cards,
                                actions=actions,
                                candidate_index=action_index,
                                baseline_index=baseline_index,
                                predictions=predictions,
                                gate_values=gate_values,
                                ev_ranks=ev_ranks,
                                decision_seed=decision_seed,
                                to_act_order=to_act_order,
                            )
                            for action_index in selected
                        ]
                        try:
                            batch_predict = getattr(
                                self.stage8c_fire_selector_scorer,
                                "predict_probabilities",
                                None,
                            )
                            if callable(batch_predict):
                                probabilities = list(batch_predict(selector_rows))
                            else:
                                probabilities = [
                                    self.stage8c_fire_selector_scorer.predict_probability(row)
                                    for row in selector_rows
                                ]
                            if len(probabilities) != len(selected):
                                raise ValueError("Stage8c fire selector returned an unexpected probability count")
                        except Exception:
                            probabilities = []
                            prediction_failed = True
                        for action_index, probability in zip(selected, probabilities):
                            probability = float(probability)
                            if not math.isfinite(probability):
                                prediction_failed = True
                                break
                            passed = probability >= self.stage8c_fire_selector_threshold
                            if passed or self.stage8c_fire_selector_audit_only:
                                filtered.append(action_index)
                            if passed:
                                stage8c_fire_selector_passed_count += 1
                            stage8c_fire_selector_probability_by_index[action_index] = probability
                            stage8c_fire_selector_candidates.append(
                                {
                                    "action_index": action_index,
                                    "action": action_to_json(board, actions[action_index]),
                                    "post_t2_board": board_to_json(board.place(actions[action_index].placements)),
                                    "probability": probability,
                                    "passed": passed,
                                    "predicted_delta": float(predictions[action_index, 1]),
                                    "model_score": float(predictions[action_index, 0]),
                                    "gate_probability": float(gate_values[action_index]),
                                    "candidate_ev_rank": int(ev_ranks[action_index]),
                                }
                            )
                        stage8c_fire_selector_evaluated_count = len(stage8c_fire_selector_candidates)
                        if stage8c_fire_selector_candidates:
                            stage8c_fire_selector_max_probability = max(
                                float(item["probability"]) for item in stage8c_fire_selector_candidates
                            )
                        if prediction_failed:
                            selected = []
                            no_override_reason = "stage8c_fire_selector_prediction_failed"
                        elif not self.stage8c_fire_selector_audit_only:
                            selected = filtered
                            if not selected:
                                no_override_reason = "below_fire_selector_threshold"
                    if not selected:
                        pass
                    elif no_override_reason:
                        pass
                    elif self.stage8c_fire_selector_direct_fire:
                        direct_candidates = [index for index in selected if index != baseline_index]
                        stage8c_fire_selector_direct_fire_candidate_count = len(direct_candidates)
                        if not direct_candidates:
                            no_override_reason = "direct_fire_empty"
                        else:
                            rerank_best_index = max(
                                direct_candidates,
                                key=lambda index: (
                                    stage8c_fire_selector_probability_by_index.get(index, float("-inf")),
                                    float(predictions[index, 1]),
                                    -int(ev_ranks[index]),
                                ),
                            )
                            final_index = rerank_best_index
                            rerank_indices = sorted({baseline_index, rerank_best_index})
                            evaluated_action_count = len(rerank_indices)
                            rerank_best_ev = float(predictions[rerank_best_index, 0])
                            rerank_baseline_ev = float(predictions[baseline_index, 0])
                            rerank_delta = float(predictions[rerank_best_index, 1])
                            rerank_delta_se = 0.0
                            stage8c_fire_selector_direct_fire_used = True
                            stage8c_fire_selector_direct_fire_probability = (
                                stage8c_fire_selector_probability_by_index.get(rerank_best_index)
                            )
                    else:
                        rerank_indices = sorted({baseline_index, *selected})
                        rerank_started = time.perf_counter()
                        sample_mc = self._rerank_sample(
                            board=board,
                            dealt=dealt,
                            opponent_board=opponent_board,
                            dead_cards=dead_cards,
                            action_indices=rerank_indices,
                            seed=self._future_rollout_seed(
                                board=board,
                                opponent_board=opponent_board,
                                dealt=dealt,
                                decision_seed=decision_seed,
                                hand_id=hand_id,
                                game_id=game_id,
                                phase="stage_a",
                            ),
                            future_samples=config.mc_samples,
                        )
                        rerank_latency_ms = (time.perf_counter() - rerank_started) * 1000.0
                        if sample_mc is None:
                            no_override_reason = "mc_rerank_failed"
                        else:
                            actions_by_original = {
                                int(action.get("original_index", -1)): action
                                for action in list(sample_mc.get("actions") or ())
                            }
                            best_action = (sample_mc.get("actions") or [None])[0]
                            baseline_action = actions_by_original.get(baseline_index)
                            if best_action is None or baseline_action is None:
                                no_override_reason = "mc_rerank_missing_baseline"
                            else:
                                rerank_best_index = int(best_action.get("original_index", -1))
                                stage_a_best_ev = float(best_action.get("score", best_action.get("ev", 0.0)))
                                stage_a_baseline_ev = float(baseline_action.get("score", baseline_action.get("ev", 0.0)))
                                stage_a_delta = stage_a_best_ev - stage_a_baseline_ev
                                best_se = float(best_action.get("ev_standard_error", best_action.get("standard_error", 0.0)))
                                baseline_se = float(baseline_action.get("ev_standard_error", baseline_action.get("standard_error", 0.0)))
                                stage_a_delta_se = math.sqrt(best_se * best_se + baseline_se * baseline_se)
                                rerank_best_ev = stage_a_best_ev
                                rerank_baseline_ev = stage_a_baseline_ev
                                rerank_delta = stage_a_delta
                                rerank_delta_se = stage_a_delta_se
                                evaluated_action_count = len(actions_by_original)
                                stage_a_common_random_future_digest = str(sample_mc.get("common_random_future_digest", ""))
                                stage_a_paired_delta_summary = _paired_delta_summary_for_action(
                                    sample_mc,
                                    rerank_best_index,
                                    baseline_index,
                                )
                                common_random_future_digest = stage_a_common_random_future_digest
                                if rerank_best_index == baseline_index:
                                    no_override_reason = "mc_best_is_baseline"
                                elif (
                                    config.min_predicted_delta is not None
                                    and float(predictions[rerank_best_index, 1]) < config.min_predicted_delta
                                ):
                                    no_override_reason = "below_predicted_delta"
                                elif config.confirm_mc_samples > 0:
                                    confirm_started = time.perf_counter()
                                    confirm_sample = self._rerank_sample(
                                        board=board,
                                        dealt=dealt,
                                        opponent_board=opponent_board,
                                        dead_cards=dead_cards,
                                        action_indices=sorted({baseline_index, rerank_best_index}),
                                        seed=self._future_rollout_seed(
                                            board=board,
                                            opponent_board=opponent_board,
                                            dealt=dealt,
                                            decision_seed=decision_seed,
                                            hand_id=hand_id,
                                            game_id=game_id,
                                            phase="stage_b_confirm",
                                        ),
                                        future_samples=config.confirm_mc_samples,
                                    )
                                    confirm_latency_ms = (time.perf_counter() - confirm_started) * 1000.0
                                    if confirm_sample is None:
                                        no_override_reason = "confirm_mc_failed"
                                    else:
                                        confirm_actions = {
                                            int(action.get("original_index", -1)): action
                                            for action in list(confirm_sample.get("actions") or ())
                                        }
                                        confirm_candidate = confirm_actions.get(rerank_best_index)
                                        confirm_baseline = confirm_actions.get(baseline_index)
                                        if confirm_candidate is None or confirm_baseline is None:
                                            no_override_reason = "confirm_mc_missing_baseline"
                                        else:
                                            confirm_candidate_ev = float(
                                                confirm_candidate.get("score", confirm_candidate.get("ev", 0.0))
                                            )
                                            confirm_baseline_ev = float(
                                                confirm_baseline.get("score", confirm_baseline.get("ev", 0.0))
                                            )
                                            confirm_delta = confirm_candidate_ev - confirm_baseline_ev
                                            candidate_se = float(
                                                confirm_candidate.get(
                                                    "ev_standard_error",
                                                    confirm_candidate.get("standard_error", 0.0),
                                                )
                                            )
                                            confirm_base_se = float(
                                                confirm_baseline.get(
                                                    "ev_standard_error",
                                                    confirm_baseline.get("standard_error", 0.0),
                                                )
                                            )
                                            confirm_delta_se = math.sqrt(
                                                candidate_se * candidate_se + confirm_base_se * confirm_base_se
                                            )
                                            if (
                                                int(confirm_sample.get("paired_delta_candidate_index", -1)) == rerank_best_index
                                                and int(confirm_sample.get("paired_delta_baseline_index", -1)) == baseline_index
                                            ):
                                                confirm_delta = float(confirm_sample.get("paired_delta_mean", confirm_delta))
                                                confirm_delta_se = float(
                                                    confirm_sample.get("paired_delta_standard_error", confirm_delta_se)
                                                )
                                                confirm_delta_count = int(confirm_sample.get("paired_delta_count", 0) or 0)
                                            confirm_paired_delta_summary = _paired_delta_summary_for_action(
                                                confirm_sample,
                                                rerank_best_index,
                                                baseline_index,
                                            )
                                            if confirm_paired_delta_summary is not None:
                                                confirm_delta = float(
                                                    confirm_paired_delta_summary.get("mean", confirm_delta)
                                                )
                                                confirm_delta_se = float(
                                                    confirm_paired_delta_summary.get(
                                                        "standard_error",
                                                        confirm_delta_se,
                                                    )
                                                )
                                                confirm_delta_count = int(
                                                    confirm_paired_delta_summary.get("count", confirm_delta_count) or 0
                                                )
                                            confirm_common_random_future_digest = str(
                                                confirm_sample.get("common_random_future_digest", "")
                                            )
                                            common_random_future_digest = confirm_common_random_future_digest
                                            rerank_best_ev = confirm_candidate_ev
                                            rerank_baseline_ev = confirm_baseline_ev
                                            rerank_delta = confirm_delta
                                            rerank_delta_se = confirm_delta_se
                                            seat_min_confirm_delta = config.min_confirm_delta_for_seat(self.seat)
                                            confirm_delta_floor = (
                                                seat_min_confirm_delta
                                                if seat_min_confirm_delta is not None
                                                else config.min_delta
                                            )
                                            if confirm_delta < confirm_delta_floor:
                                                no_override_reason = "below_confirm_delta"
                                            elif (
                                                config.max_confirm_se is not None
                                                and confirm_delta_se > config.max_confirm_se
                                            ):
                                                no_override_reason = "above_confirm_se"
                                            elif (
                                                config.confirm_se_multiplier > 0.0
                                                and confirm_delta < config.confirm_se_multiplier * confirm_delta_se
                                            ):
                                                no_override_reason = "below_confirm_se"
                                            elif self.local_ev_risk_scorer is not None:
                                                try:
                                                    candidate_rank = int(ev_ranks[rerank_best_index])
                                                    local_ev_risk_rank_guard_passed = (
                                                        (self.local_ev_risk_rank_min is None or candidate_rank >= self.local_ev_risk_rank_min)
                                                        and (self.local_ev_risk_rank_max is None or candidate_rank <= self.local_ev_risk_rank_max)
                                                    )
                                                    if not local_ev_risk_rank_guard_passed:
                                                        final_index = rerank_best_index
                                                    else:
                                                        candidate_action_json = action_to_json(board, actions[rerank_best_index])
                                                        baseline_action_json = action_to_json(board, actions[baseline_index])
                                                        local_ev_risk_probability = self.local_ev_risk_scorer.predict_probability(
                                                            {
                                                                "source_log": self.topk_context.get("source_log", ""),
                                                                "hand_seed": decision_seed if decision_seed is not None else self.seed,
                                                                "seat": self.seat,
                                                                "to_act_order": to_act_order,
                                                                "hero_board": board_to_json(board),
                                                                "opponent_board": board_to_json(opponent_board),
                                                                "dead_cards": list(dead_cards),
                                                                "cards_to_place": list(dealt),
                                                                "candidate_action": candidate_action_json,
                                                                "baseline_action": baseline_action_json,
                                                                "predicted_delta": float(predictions[rerank_best_index, 1]),
                                                                "gate_probability": float(gate_values[rerank_best_index]),
                                                                "confirm_delta": confirm_delta,
                                                                "confirm_delta_se": confirm_delta_se,
                                                                "confirm_paired_delta_summary": confirm_paired_delta_summary,
                                                                "paired_future_delta_summary": confirm_paired_delta_summary,
                                                                "candidate_ev_rank": candidate_rank,
                                                            }
                                                        )
                                                except Exception:
                                                    no_override_reason = "local_ev_risk_prediction_failed"
                                                else:
                                                    if local_ev_risk_probability is not None and local_ev_risk_probability >= self.local_ev_risk_threshold:
                                                        local_ev_risk_would_veto = True
                                                        if self.local_ev_risk_audit_only:
                                                            final_index = rerank_best_index
                                                        else:
                                                            local_ev_risk_vetoed = True
                                                            no_override_reason = "local_ev_risk_veto"
                                                    elif local_ev_risk_rank_guard_passed:
                                                        final_index = rerank_best_index
                                            else:
                                                final_index = rerank_best_index
                                elif rerank_delta < config.min_delta:
                                    no_override_reason = "below_rerank_delta"
                                elif config.se_multiplier > 0.0 and rerank_delta < config.se_multiplier * rerank_delta_se:
                                    no_override_reason = "below_rerank_se"
                                else:
                                    final_index = rerank_best_index
                candidate_for_logging = rerank_best_index if rerank_best_index is not None else stage8b_top1_index
                if candidate_for_logging is not None and 0 <= candidate_for_logging < len(actions):
                    predicted_delta = float(predictions[candidate_for_logging, 1])
                    predicted_ev = float(predictions[candidate_for_logging, 0])
                    gate_probability = float(gate_values[candidate_for_logging])
                    candidate_ev_rank = int(ev_ranks[candidate_for_logging])
                    for item in stage8c_fire_selector_candidates:
                        if int(item.get("action_index", -1)) == candidate_for_logging:
                            stage8c_fire_selector_probability = float(item["probability"])
                            break

        override_fired = final_index != baseline_index
        if not no_override_reason and not override_fired:
            no_override_reason = "fallback_to_baseline"
        visible_dead_cards = list(dead_cards)
        opponent_public = set(opponent_board.all_cards())
        hero_private_discards = [
            card for card in dead_cards if card not in opponent_public
        ]
        paired_future_delta_summary, paired_future_delta_source = _select_paired_future_delta_summary(
            confirm_paired_delta_summary,
            stage_a_paired_delta_summary,
        )
        self._log_topk_decision(
            {
                **self.topk_context,
                "hand_id": hand_id,
                "game_id": game_id,
                "seed": decision_seed if decision_seed is not None else self.seed,
                "street": street or "T2",
                "turn": street or "T2",
                "seat": self.seat,
                "t3_continuation": self.t3_continuation,
                "t3_continuation_policy": _t3_continuation_policy_name(self.t3_continuation),
                "hero_board": board_to_json(board),
                "opponent_board": board_to_json(opponent_board),
                "cards_to_place": list(dealt),
                "dead_cards": list(dead_cards),
                "visible_dead_cards": list(visible_dead_cards),
                "true_dead_cards": [],
                "hero_private_discards": hero_private_discards,
                "visibility_model": "actor_observation_v1",
                "discard_visibility": "own_private_only",
                "replay_ready": False,
                "action_key_schema": ACTION_KEY_SCHEMA,
                "legal_action_set_digest": legal_action_set_digest(actions),
                "legal_action_order_digest": ordered_action_mapping_digest(actions),
                "baseline_action": action_to_json(board, actions[baseline_index]),
                "stage8b_top1_action": action_to_json(board, actions[stage8b_top1_index]) if stage8b_top1_index is not None else None,
                "rerank_best_action": action_to_json(board, actions[rerank_best_index]) if rerank_best_index is not None and 0 <= rerank_best_index < len(actions) else None,
                "fallback_action": action_to_json(board, actions[baseline_index]),
                "final_action": action_to_json(board, actions[final_index]),
                "post_t2_baseline_board": board_to_json(board.place(actions[baseline_index].placements)),
                "post_t2_candidate_board": board_to_json(board.place(actions[final_index].placements)),
                "post_t2_rerank_best_board": (
                    board_to_json(board.place(actions[rerank_best_index].placements))
                    if rerank_best_index is not None and 0 <= rerank_best_index < len(actions)
                    else None
                ),
                "baseline_action_index": baseline_index,
                "stage8b_top1_index": stage8b_top1_index,
                "rerank_best_index": rerank_best_index,
                "final_action_index": final_index,
                "rerank_action_indices": rerank_indices,
                "rerank_action_keys": [
                    action_key(actions[index]).to_token() for index in rerank_indices
                ],
                "baseline_action_key": action_key(actions[baseline_index]).to_token(),
                "stage8b_top1_action_key": (
                    action_key(actions[stage8b_top1_index]).to_token()
                    if stage8b_top1_index is not None
                    else None
                ),
                "rerank_best_action_key": (
                    action_key(actions[rerank_best_index]).to_token()
                    if rerank_best_index is not None
                    and 0 <= rerank_best_index < len(actions)
                    else None
                ),
                "final_action_key": action_key(actions[final_index]).to_token(),
                "override_fired": override_fired,
                "no_override_reason": no_override_reason or "",
                "top_k": config.top_k,
                "mc_samples": config.mc_samples,
                "min_rerank_delta": config.min_delta,
                "se_multiplier": config.se_multiplier,
                "confirm_mc_samples": config.confirm_mc_samples,
                "confirm_se_multiplier": config.confirm_se_multiplier,
                "max_confirm_se": config.max_confirm_se,
                "min_confirm_delta": config.min_confirm_delta,
                "first_min_confirm_delta": config.first_min_confirm_delta,
                "second_min_confirm_delta": config.second_min_confirm_delta,
                "seat_min_confirm_delta": config.min_confirm_delta_for_seat(self.seat),
                "topk_score": config.topk_score,
                "allowed_seats": list(config.allowed_seats),
                "candidate_ev_rank_max": config.candidate_ev_rank_max,
                "min_gate_probability": config.min_gate_probability,
                "min_predicted_delta": config.min_predicted_delta,
                "predicted_delta": predicted_delta,
                "gate_probability": gate_probability,
                "model_score": predicted_ev,
                "candidate_ev_rank": candidate_ev_rank,
                "rerank_delta": rerank_delta,
                "rerank_delta_se": rerank_delta_se,
                "rerank_best_ev": rerank_best_ev,
                "rerank_baseline_ev": rerank_baseline_ev,
                "stage_a_delta": stage_a_delta,
                "stage_a_delta_se": stage_a_delta_se,
                "stage_a_best_ev": stage_a_best_ev,
                "stage_a_baseline_ev": stage_a_baseline_ev,
                "stage_a_common_random_future_digest": stage_a_common_random_future_digest,
                "stage_a_paired_delta_summary": stage_a_paired_delta_summary,
                "confirm_delta": confirm_delta,
                "confirm_delta_se": confirm_delta_se,
                "confirm_delta_count": confirm_delta_count,
                "confirm_candidate_ev": confirm_candidate_ev,
                "confirm_baseline_ev": confirm_baseline_ev,
                "confirm_common_random_future_digest": confirm_common_random_future_digest,
                "confirm_paired_delta_summary": confirm_paired_delta_summary,
                "local_ev_risk_model": (
                    str(self.local_ev_risk_scorer.path) if self.local_ev_risk_scorer is not None else ""
                ),
                "local_ev_risk_enabled": self.local_ev_risk_scorer is not None,
                "local_ev_risk_threshold": (
                    self.local_ev_risk_threshold if self.local_ev_risk_scorer is not None else None
                ),
                "local_ev_risk_rank_min": self.local_ev_risk_rank_min,
                "local_ev_risk_rank_max": self.local_ev_risk_rank_max,
                "local_ev_risk_audit_only": self.local_ev_risk_audit_only,
                "local_ev_risk_rank_guard_passed": local_ev_risk_rank_guard_passed,
                "local_ev_risk_probability": local_ev_risk_probability,
                "local_ev_risk_would_veto": local_ev_risk_would_veto,
                "local_ev_risk_vetoed": local_ev_risk_vetoed,
                "stage8c_fire_selector_model": (
                    str(self.stage8c_fire_selector_scorer.path)
                    if self.stage8c_fire_selector_scorer is not None
                    else ""
                ),
                "stage8c_fire_selector_enabled": self.stage8c_fire_selector_scorer is not None,
                "stage8c_fire_selector_threshold": (
                    self.stage8c_fire_selector_threshold
                    if self.stage8c_fire_selector_scorer is not None
                    else None
                ),
                "stage8c_fire_selector_audit_only": self.stage8c_fire_selector_audit_only,
                "stage8c_fire_selector_direct_fire_enabled": self.stage8c_fire_selector_direct_fire,
                "stage8c_fire_selector_direct_fire_used": stage8c_fire_selector_direct_fire_used,
                "stage8c_fire_selector_direct_fire_candidate_count": (
                    stage8c_fire_selector_direct_fire_candidate_count
                ),
                "stage8c_fire_selector_direct_fire_probability": (
                    stage8c_fire_selector_direct_fire_probability
                ),
                "stage8c_fire_selector_probability": stage8c_fire_selector_probability,
                "stage8c_fire_selector_max_probability": stage8c_fire_selector_max_probability,
                "stage8c_fire_selector_evaluated_count": stage8c_fire_selector_evaluated_count,
                "stage8c_fire_selector_passed_count": stage8c_fire_selector_passed_count,
                "stage8c_fire_selector_candidates": stage8c_fire_selector_candidates,
                "paired_future_delta_summary": paired_future_delta_summary,
                "paired_future_delta_source": paired_future_delta_source,
                "evaluated_action_count": evaluated_action_count,
                "common_random_future_digest": common_random_future_digest,
                "runtime_latency_ms": (time.perf_counter() - started_at) * 1000.0,
                "mc_rerank_latency_ms": rerank_latency_ms,
                "confirm_mc_latency_ms": confirm_latency_ms,
            }
        )
        return actions[final_index]

    def _stage8c_candidate_score_row(
        self,
        *,
        board: Board,
        opponent_board: Board,
        dealt: tuple[str, ...],
        dead_cards: tuple[str, ...],
        actions: list[Action],
        candidate_index: int,
        baseline_index: int,
        predictions: np.ndarray,
        gate_values: np.ndarray,
        ev_ranks: np.ndarray,
        decision_seed: int | None,
        to_act_order: str,
    ) -> dict[str, Any]:
        return {
            "source_log": self.topk_context.get("source_log", ""),
            "hand_seed": decision_seed if decision_seed is not None else self.seed,
            "seat": self.seat,
            "to_act_order": to_act_order,
            "hero_board": board_to_json(board),
            "opponent_board": board_to_json(opponent_board),
            "dead_cards": list(dead_cards),
            "cards_to_place": list(dealt),
            "candidate_action": action_to_json(board, actions[candidate_index]),
            "baseline_action": action_to_json(board, actions[baseline_index]),
            "predicted_delta": float(predictions[candidate_index, 1]),
            "gate_probability": float(gate_values[candidate_index]),
            "candidate_ev_rank": int(ev_ranks[candidate_index]),
            "model_score": float(predictions[candidate_index, 0]),
            # Keep the historical distillation feature behavior: topk_score is
            # a config string in runtime logs and becomes 0.0 under safe_float.
            "topk_score": self.topk_rerank_config.topk_score,
            "top_k": self.topk_rerank_config.top_k,
        }

    def _topk_order(
        self,
        predictions: np.ndarray,
        gate_values: np.ndarray,
        actions: list[Action],
    ) -> np.ndarray:
        config = self.topk_rerank_config
        if config.topk_score == "ev":
            scores = predictions[:, 0]
        elif config.topk_score == "gate":
            scores = gate_values
        elif config.topk_score == "gate_delta":
            scores = predictions[:, 1] * gate_values
        else:
            scores = predictions[:, 1]
        return np.asarray(canonical_descending_indices(scores, actions), dtype=np.int64)

    def _rerank_sample(
        self,
        *,
        board: Board,
        dealt: tuple[str, ...],
        opponent_board: Board,
        dead_cards: tuple[str, ...],
        action_indices: list[int],
        seed: int,
        future_samples: int | None = None,
    ) -> dict[str, Any] | None:
        excluded = set(board.all_cards()) | set(opponent_board.all_cards()) | set(dealt)
        rollout_dead = tuple(card for card in dead_cards if card not in excluded)
        rollout_count = future_samples or self.topk_rerank_config.mc_samples
        observation = ActorObservation(
            hero_board=board,
            opponent_public_board=opponent_board,
            dealt_cards=dealt,
            hero_private_discards=rollout_dead,
            seat=self.seat,  # type: ignore[arg-type]
            street="T2",
            to_act_order=(
                "second" if opponent_board.card_count() > board.card_count() else "first"
            ),
        )
        belief_batch = sample_hidden_card_particles(
            observation,
            base_seed=seed,
            run_id="hu_turn2_topk_rerank",
            sample_count=rollout_count,
        )
        hero_policy = self._build_rollout_policy(seed=seed * 4 + 1, seat=self.seat)
        opponent_seat = "second" if self.seat == "first" else "first"
        opponent_policy = self._build_rollout_policy(seed=seed * 4 + 2, seat=opponent_seat)
        return evaluate_hu_turn2_actions(
            board=board,
            dealt_cards=dealt,
            opponent_board=opponent_board,
            hero_seat=self.seat,
            continuation_policy=hero_policy,
            opponent_policy=opponent_policy,
            continuation_policy_name=teacher_t3_continuation_policy_name(self.t3_continuation),
            continuation_metadata=teacher_t3_continuation_metadata(self.t3_continuation),
            baseline_turn2_model=self.turn2_model,
            future_samples=rollout_count,
            future_rollout_seed=seed,
            action_indices=action_indices,
            use_batched_continuation=True,
            batched_continuation_config=self._batched_config,
            batched_continuation_cache=self._t3_decision_cache,
            batched_reference_cache=self._t3_reference_cache,
            batched_state_feature_cache=self._t3_state_feature_cache,
            batched_action_cache=self._t3_action_cache,
            batched_continuation_batch_size=self.batched_continuation_batch_size,
            final_turn_cache=self._final_turn_cache,
            use_final_turn_cache=True,
            observation=observation,
            belief_batch=belief_batch,
        )

    def _build_rollout_policy(self, *, seed: int, seat: str) -> RegularAiPolicy:
        return RegularAiPolicy(
            opening_model=self.opening_model,
            turn1_model=self.turn1_model,
            turn2_model=self.turn2_model,
            turn3_model=self.turn3_model,
            **_t3_policy_kwargs(
                ModelParts(
                    opening=self.opening_model,
                    turn1=self.turn1_model,
                    turn2_baseline=self.turn2_model,
                    turn3=self.turn3_model,
                    hu_turn3_stage7=self.hu_turn3_model,
                    hu_turn3_reference=self.hu_turn3_reference_model,
                    hu_turn2_stage8=None,
                ),
                self.t3_continuation,
            ),
            seed=seed,
            seat=seat,
            opening_lookahead_samples=self.opening_lookahead_samples,
        )

    def _future_rollout_seed(
        self,
        *,
        board: Board,
        opponent_board: Board,
        dealt: tuple[str, ...],
        decision_seed: int | None,
        hand_id: str | int | None,
        game_id: str | int | None,
        phase: str = "stage_a",
    ) -> int:
        payload = {
            "phase": phase,
            "decision_seed": decision_seed if decision_seed is not None else self.seed,
            "hand_id": hand_id,
            "game_id": game_id,
            "seat": self.seat,
            "board": board_to_json(board),
            "opponent": board_to_json(opponent_board),
            "dealt": list(dealt),
            "config_id": self.topk_rerank_config.config_id,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return int(digest[:16], 16) % 2_147_483_647

    def _log_topk_decision(self, record: dict[str, Any]) -> None:
        if self.topk_decision_log is not None:
            self.topk_decision_log.append(record)


def make_topk_policy(
    parts: ModelParts,
    *,
    stage8b_model: object,
    config: TopKMcRerankConfig,
    decisions: list[dict[str, Any]],
    context: dict[str, Any],
    seed: int,
    seat: str,
    opening_lookahead_samples: int,
    batched_continuation_batch_size: int,
    stage3_feature_encoder_mode: str,
    t3_continuation: T3ContinuationMode,
    local_ev_risk_scorer: LocalEvRiskScorer | None = None,
    local_ev_risk_threshold: float = 0.8,
    local_ev_risk_rank_min: int | None = None,
    local_ev_risk_rank_max: int | None = None,
    local_ev_risk_audit_only: bool = False,
    stage8c_fire_selector_scorer: LocalEvRiskScorer | None = None,
    stage8c_fire_selector_threshold: float = 0.7,
    stage8c_fire_selector_audit_only: bool = False,
    stage8c_fire_selector_direct_fire: bool = False,
    hu_turn3_decision_log: list[dict[str, Any]] | None = None,
) -> HuTurn2Stage8bTopKMcRerankPolicy:
    return HuTurn2Stage8bTopKMcRerankPolicy(
        opening_model=parts.opening,
        turn1_model=parts.turn1,
        turn2_model=parts.turn2_baseline,
        turn3_model=parts.turn3,
        **_t3_policy_kwargs(parts, t3_continuation),
        hu_turn3_decision_log=hu_turn3_decision_log,
        hu_turn2_stage8b_model=stage8b_model,
        topk_rerank_config=config,
        topk_decision_log=decisions,
        topk_context=context,
        seed=seed,
        seat=seat,
        opening_lookahead_samples=opening_lookahead_samples,
        batched_continuation_batch_size=batched_continuation_batch_size,
        stage3_feature_encoder_mode=stage3_feature_encoder_mode,
        t3_continuation=t3_continuation,
        local_ev_risk_scorer=local_ev_risk_scorer,
        local_ev_risk_threshold=local_ev_risk_threshold,
        local_ev_risk_rank_min=local_ev_risk_rank_min,
        local_ev_risk_rank_max=local_ev_risk_rank_max,
        local_ev_risk_audit_only=local_ev_risk_audit_only,
        stage8c_fire_selector_scorer=stage8c_fire_selector_scorer,
        stage8c_fire_selector_threshold=stage8c_fire_selector_threshold,
        stage8c_fire_selector_audit_only=stage8c_fire_selector_audit_only,
        stage8c_fire_selector_direct_fire=stage8c_fire_selector_direct_fire,
    )


def load_topk_parts(args: argparse.Namespace) -> ModelParts:
    """Load TopK eval model parts, optionally allowing Stage9f model fallback.

    Baseline and T3-continuation model failures remain fatal. Only the HU T2
    candidate generator can be set to None, and only when the explicit preflight
    flag is present.
    """
    if not getattr(args, "allow_missing_stage8b_model_fallback", False):
        args.stage8b_model_load_failed = False
        args.stage8b_model_load_error = ""
        return load_parts(
            argparse.Namespace(
                opening_model=args.opening_model,
                turn1_model=args.turn1_model,
                turn2_baseline_model=args.turn2_baseline_model,
                turn3_model=args.turn3_model,
                hu_turn3_stage7_model=args.hu_turn3_stage7_model,
                hu_turn3_reference_model=args.hu_turn3_reference_model,
                hu_turn2_stage8_model=args.hu_turn2_stage8b_model,
                device=args.device,
            )
        )

    candidate_model: object | None
    try:
        candidate_model = load_hu_turn2_stage8_model(args.hu_turn2_stage8b_model, device=args.device)
        args.stage8b_model_load_failed = False
        args.stage8b_model_load_error = ""
    except Exception as exc:
        candidate_model = None
        args.stage8b_model_load_failed = True
        args.stage8b_model_load_error = str(exc)

    return ModelParts(
        opening=load_action_value_model(args.opening_model),
        turn1=load_action_value_model(args.turn1_model),
        turn2_baseline=load_action_value_model(args.turn2_baseline_model),
        turn3=load_action_value_model(args.turn3_model),
        hu_turn3_stage7=load_hu_action_value_model(args.hu_turn3_stage7_model),
        hu_turn3_reference=load_hu_action_value_model(args.hu_turn3_reference_model),
        hu_turn2_stage8=candidate_model,
    )


def realized_override_count(decisions: Iterable[dict[str, Any]]) -> int:
    return sum(
        1
        for row in decisions
        if row.get("override_fired")
        and row.get("realized_delta_valid")
        and row.get("realized_candidate_seat_delta") not in (None, "")
    )


def risk_veto_count(decisions: Iterable[dict[str, Any]]) -> int:
    return sum(1 for row in decisions if row.get("local_ev_risk_would_veto", row.get("local_ev_risk_vetoed")))


def fire_selector_rejection_count(decisions: Iterable[dict[str, Any]]) -> int:
    return sum(1 for row in decisions if row.get("no_override_reason") == "below_fire_selector_threshold")


def evaluate_config_seed(
    *,
    config: TopKMcRerankConfig,
    seed: int,
    seed_stride: int,
    games: int,
    target_realized_overrides: int,
    target_risk_vetoes: int,
    target_fire_selector_rejections: int,
    parts: ModelParts,
    stage8b_model: object,
    opening_lookahead_samples: int,
    batched_continuation_batch_size: int,
    stage3_feature_encoder_mode: str,
    t3_continuation: T3ContinuationMode,
    local_ev_risk_scorer: LocalEvRiskScorer | None,
    local_ev_risk_threshold: float,
    local_ev_risk_rank_min: int | None,
    local_ev_risk_rank_max: int | None,
    local_ev_risk_audit_only: bool,
    stage8c_fire_selector_scorer: LocalEvRiskScorer | None,
    stage8c_fire_selector_threshold: float,
    stage8c_fire_selector_audit_only: bool,
    stage8c_fire_selector_direct_fire: bool,
    progress_every: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    paired_scores: list[float] = []
    position_scores: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    wins = losses = ties = 0
    started_at = time.time()
    stop_reason = "max_games_reached"
    for index in range(games):
        hand_seed = seed * seed_stride + index
        before = len(decisions)
        ab_t3_decision_log: list[dict[str, Any]] = []
        hand_ab = trace_hand(
            seed=hand_seed,
            profile_p0="stage8b_topk_mc",
            profile_p1="baseline",
            policy_p0=make_topk_policy(
                parts,
                stage8b_model=stage8b_model,
                config=config,
                decisions=decisions,
                context={"config_id": config.config_id, "paired_index": index, "seat_swap": "ab"},
                seed=hand_seed * 4,
                seat="first",
                opening_lookahead_samples=opening_lookahead_samples,
                batched_continuation_batch_size=batched_continuation_batch_size,
                stage3_feature_encoder_mode=stage3_feature_encoder_mode,
                t3_continuation=t3_continuation,
                local_ev_risk_scorer=local_ev_risk_scorer,
                local_ev_risk_threshold=local_ev_risk_threshold,
                local_ev_risk_rank_min=local_ev_risk_rank_min,
                local_ev_risk_rank_max=local_ev_risk_rank_max,
                local_ev_risk_audit_only=local_ev_risk_audit_only,
                stage8c_fire_selector_scorer=stage8c_fire_selector_scorer,
                stage8c_fire_selector_threshold=stage8c_fire_selector_threshold,
                stage8c_fire_selector_audit_only=stage8c_fire_selector_audit_only,
                stage8c_fire_selector_direct_fire=stage8c_fire_selector_direct_fire,
                hu_turn3_decision_log=ab_t3_decision_log,
            ),
            policy_p1=make_baseline_policy(
                parts,
                seed=hand_seed * 4 + 1,
                seat="second",
                opening_lookahead_samples=opening_lookahead_samples,
                t3_continuation=t3_continuation,
                hu_turn3_decision_log=ab_t3_decision_log,
            ),
        )
        ab_rows = decisions[before:]
        ab_candidate_outcome = _trace_outcome_for_player(hand_ab, 0)
        ab_candidate_t3_summary = _trace_t3_decision_summary(hand_ab, 0, ab_t3_decision_log)
        for row in ab_rows:
            row["candidate_seat_score"] = float(hand_ab["score_p0"])
            row["paired_index"] = index
            row["hand_seed"] = hand_seed
            _add_prefixed_outcome(row, "candidate", ab_candidate_outcome)
            _add_prefixed_t3_summary(row, "candidate", ab_candidate_t3_summary)
            _add_t3_summary_delta(row)
        position_scores.append({"config_id": config.config_id, "seed": seed, "seat": "first", "score": float(hand_ab["score_p0"])})

        before = len(decisions)
        ba_t3_decision_log: list[dict[str, Any]] = []
        hand_ba = trace_hand(
            seed=hand_seed,
            profile_p0="baseline",
            profile_p1="stage8b_topk_mc",
            policy_p0=make_baseline_policy(
                parts,
                seed=hand_seed * 4 + 2,
                seat="first",
                opening_lookahead_samples=opening_lookahead_samples,
                t3_continuation=t3_continuation,
                hu_turn3_decision_log=ba_t3_decision_log,
            ),
            policy_p1=make_topk_policy(
                parts,
                stage8b_model=stage8b_model,
                config=config,
                decisions=decisions,
                context={"config_id": config.config_id, "paired_index": index, "seat_swap": "ba"},
                seed=hand_seed * 4 + 3,
                seat="second",
                opening_lookahead_samples=opening_lookahead_samples,
                batched_continuation_batch_size=batched_continuation_batch_size,
                stage3_feature_encoder_mode=stage3_feature_encoder_mode,
                t3_continuation=t3_continuation,
                local_ev_risk_scorer=local_ev_risk_scorer,
                local_ev_risk_threshold=local_ev_risk_threshold,
                local_ev_risk_rank_min=local_ev_risk_rank_min,
                local_ev_risk_rank_max=local_ev_risk_rank_max,
                local_ev_risk_audit_only=local_ev_risk_audit_only,
                stage8c_fire_selector_scorer=stage8c_fire_selector_scorer,
                stage8c_fire_selector_threshold=stage8c_fire_selector_threshold,
                stage8c_fire_selector_audit_only=stage8c_fire_selector_audit_only,
                stage8c_fire_selector_direct_fire=stage8c_fire_selector_direct_fire,
                hu_turn3_decision_log=ba_t3_decision_log,
            ),
        )
        candidate_second_score = -float(hand_ba["score_p0"])
        ba_rows = decisions[before:]
        ba_candidate_outcome = _trace_outcome_for_player(hand_ba, 1)
        ba_candidate_t3_summary = _trace_t3_decision_summary(hand_ba, 1, ba_t3_decision_log)
        for row in ba_rows:
            row["candidate_seat_score"] = candidate_second_score
            row["paired_index"] = index
            row["hand_seed"] = hand_seed
            _add_prefixed_outcome(row, "candidate", ba_candidate_outcome)
            _add_prefixed_t3_summary(row, "candidate", ba_candidate_t3_summary)
            _add_t3_summary_delta(row)
        position_scores.append({"config_id": config.config_id, "seed": seed, "seat": "second", "score": candidate_second_score})

        # A seat-specific realized delta can be inferred from the opposite
        # seat-swap trace only when that opposite trace remained baseline-like.
        # This is the unbiased per-fire evaluation target; confirm MC deltas
        # are gate inputs only and are intentionally not used as realized gain.
        first_baseline_score = float(hand_ba["score_p0"])
        second_baseline_score = -float(hand_ab["score_p0"])
        ab_has_override = any(bool(row.get("override_fired")) for row in ab_rows)
        ba_has_override = any(bool(row.get("override_fired")) for row in ba_rows)
        first_baseline_outcome = _trace_outcome_for_player(hand_ba, 0)
        second_baseline_outcome = _trace_outcome_for_player(hand_ab, 1)
        first_baseline_t3_summary = _trace_t3_decision_summary(hand_ba, 0, ba_t3_decision_log)
        second_baseline_t3_summary = _trace_t3_decision_summary(hand_ab, 1, ab_t3_decision_log)
        for row in ab_rows:
            row["baseline_seat_score"] = first_baseline_score
            row["realized_delta_valid"] = not ba_has_override
            row["realized_delta_basis"] = "opposite_seat_swap_counterfactual"
            if row["realized_delta_valid"]:
                row["realized_candidate_seat_delta"] = float(row["candidate_seat_score"]) - first_baseline_score
                _add_prefixed_outcome(row, "baseline", first_baseline_outcome)
                _add_prefixed_t3_summary(row, "baseline", first_baseline_t3_summary)
                _add_t3_summary_delta(row)
                _add_outcome_delta(row)
        for row in ba_rows:
            row["baseline_seat_score"] = second_baseline_score
            row["realized_delta_valid"] = not ab_has_override
            row["realized_delta_basis"] = "opposite_seat_swap_counterfactual"
            if row["realized_delta_valid"]:
                row["realized_candidate_seat_delta"] = float(row["candidate_seat_score"]) - second_baseline_score
                _add_prefixed_outcome(row, "baseline", second_baseline_outcome)
                _add_prefixed_t3_summary(row, "baseline", second_baseline_t3_summary)
                _add_t3_summary_delta(row)
                _add_outcome_delta(row)

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
                        "event": "topk_mc_rerank_progress",
                        "config_id": config.config_id,
                        "t3_continuation": t3_continuation,
                        "t3_continuation_policy": _t3_continuation_policy_name(t3_continuation),
                        "seed": seed,
                        "paired_seeds": index + 1,
                        "target_realized_overrides": target_realized_overrides,
                        "target_risk_vetoes": target_risk_vetoes,
                        "target_fire_selector_rejections": target_fire_selector_rejections,
                        "realized_override_count": realized_override_count(decisions),
                        "risk_veto_count": risk_veto_count(decisions),
                        "fire_selector_rejection_count": fire_selector_rejection_count(decisions),
                        **summarize_values(paired_scores, "ev_per_hand_"),
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
        if target_realized_overrides > 0 and realized_override_count(decisions) >= target_realized_overrides:
            stop_reason = "target_realized_overrides_reached"
            break
        if target_risk_vetoes > 0 and risk_veto_count(decisions) >= target_risk_vetoes:
            stop_reason = "target_risk_vetoes_reached"
            break
        if (
            target_fire_selector_rejections > 0
            and fire_selector_rejection_count(decisions) >= target_fire_selector_rejections
        ):
            stop_reason = "target_fire_selector_rejections_reached"
            break

    no_override = Counter(str(row.get("no_override_reason", "")) for row in decisions if not row.get("override_fired"))
    override_count = sum(1 for row in decisions if row.get("override_fired"))
    local_ev_risk_evaluated_count = sum(
        1 for row in decisions if row.get("local_ev_risk_probability") not in (None, "")
    )
    local_ev_risk_would_veto_count = risk_veto_count(decisions)
    local_ev_risk_veto_count = sum(1 for row in decisions if row.get("local_ev_risk_vetoed"))
    stage8c_fire_selector_evaluated_count = sum(
        int(row.get("stage8c_fire_selector_evaluated_count", 0) or 0) for row in decisions
    )
    stage8c_fire_selector_passed_count = sum(
        int(row.get("stage8c_fire_selector_passed_count", 0) or 0) for row in decisions
    )
    stage8c_fire_selector_direct_fire_count = sum(
        1 for row in decisions if row.get("stage8c_fire_selector_direct_fire_used")
    )
    stage8c_fire_selector_blocked_count = sum(
        1 for row in decisions if row.get("no_override_reason") == "below_fire_selector_threshold"
    )
    fired = [row for row in decisions if row.get("override_fired")]
    rerank_deltas = [float(row.get("rerank_delta", 0.0) or 0.0) for row in fired]
    rerank_losses = [max(0.0, -delta) for delta in rerank_deltas]
    confirm_deltas = [stage_b_delta(row) for row in fired]
    confirm_losses = [max(0.0, -delta) for delta in confirm_deltas]
    realized_deltas = [
        float(row.get("realized_candidate_seat_delta", 0.0) or 0.0)
        for row in fired
        if row.get("realized_delta_valid") and row.get("realized_candidate_seat_delta") not in (None, "")
    ]
    summary = {
        "config_id": config.config_id,
        "t3_continuation": t3_continuation,
        "t3_continuation_policy": _t3_continuation_policy_name(t3_continuation),
        "top_k": config.top_k,
        "mc_samples": config.mc_samples,
        "min_rerank_delta": config.min_delta,
        "se_multiplier": config.se_multiplier,
        "confirm_mc_samples": config.confirm_mc_samples,
        "confirm_se_multiplier": config.confirm_se_multiplier,
        "max_confirm_se": "" if config.max_confirm_se is None else config.max_confirm_se,
        "min_confirm_delta": "" if config.min_confirm_delta is None else config.min_confirm_delta,
        "first_min_confirm_delta": "" if config.first_min_confirm_delta is None else config.first_min_confirm_delta,
        "second_min_confirm_delta": "" if config.second_min_confirm_delta is None else config.second_min_confirm_delta,
        "allowed_seats": "+".join(config.allowed_seats),
        "candidate_ev_rank_max": "" if config.candidate_ev_rank_max is None else config.candidate_ev_rank_max,
        "min_gate_probability": "" if config.min_gate_probability is None else config.min_gate_probability,
        "min_predicted_delta": "" if config.min_predicted_delta is None else config.min_predicted_delta,
        "topk_score": config.topk_score,
        "stage8b_model_loaded": stage8b_model is not None,
        "stage8b_model_load_failed": stage8b_model is None,
        "local_ev_risk_model": "" if local_ev_risk_scorer is None else str(local_ev_risk_scorer.path),
        "local_ev_risk_threshold": "" if local_ev_risk_scorer is None else local_ev_risk_threshold,
        "stage8c_risk_model": "" if local_ev_risk_scorer is None else str(local_ev_risk_scorer.path),
        "stage8c_risk_threshold": "" if local_ev_risk_scorer is None else local_ev_risk_threshold,
        "stage8c_risk_rank_min": "" if local_ev_risk_scorer is None or local_ev_risk_rank_min is None else local_ev_risk_rank_min,
        "stage8c_risk_rank_max": "" if local_ev_risk_scorer is None or local_ev_risk_rank_max is None else local_ev_risk_rank_max,
        "stage8c_risk_audit_only": bool(local_ev_risk_audit_only),
        "stage8c_fire_selector_model": (
            "" if stage8c_fire_selector_scorer is None else str(stage8c_fire_selector_scorer.path)
        ),
        "stage8c_fire_selector_threshold": (
            "" if stage8c_fire_selector_scorer is None else stage8c_fire_selector_threshold
        ),
        "stage8c_fire_selector_audit_only": bool(stage8c_fire_selector_audit_only),
        "stage8c_fire_selector_direct_fire_enabled": bool(stage8c_fire_selector_direct_fire),
        "seed": seed,
        "paired_seeds": len(paired_scores),
        "max_paired_seeds": games,
        "hands": len(paired_scores) * 2,
        "target_realized_overrides": target_realized_overrides,
        "target_risk_vetoes": target_risk_vetoes,
        "target_fire_selector_rejections": target_fire_selector_rejections,
        "target_realized_overrides_reached": bool(
            target_realized_overrides > 0 and len(realized_deltas) >= target_realized_overrides
        ),
        "target_risk_vetoes_reached": bool(
            target_risk_vetoes > 0 and local_ev_risk_would_veto_count >= target_risk_vetoes
        ),
        "target_fire_selector_rejections_reached": bool(
            target_fire_selector_rejections > 0
            and stage8c_fire_selector_blocked_count >= target_fire_selector_rejections
        ),
        "stop_reason": stop_reason,
        "ev_per_hand": float(np.mean(paired_scores)) if paired_scores else 0.0,
        **summarize_values(paired_scores, "paired_score_"),
        "paired_seed_wins": wins,
        "paired_seed_losses": losses,
        "paired_seed_ties": ties,
        "decision_count": len(decisions),
        "override_count": override_count,
        "realized_override_count": len(realized_deltas),
        "override_rate": override_count / max(len(decisions), 1),
        "local_ev_risk_evaluated_count": local_ev_risk_evaluated_count,
        "local_ev_risk_would_veto_count": local_ev_risk_would_veto_count,
        "local_ev_risk_veto_count": local_ev_risk_veto_count,
        "stage8c_fire_selector_evaluated_count": stage8c_fire_selector_evaluated_count,
        "stage8c_fire_selector_passed_count": stage8c_fire_selector_passed_count,
        "stage8c_fire_selector_direct_fire_count": stage8c_fire_selector_direct_fire_count,
        "stage8c_fire_selector_blocked_count": stage8c_fire_selector_blocked_count,
        "stage8c_fire_selector_pass_rate": stage8c_fire_selector_passed_count
        / max(stage8c_fire_selector_evaluated_count, 1),
        "avg_rerank_delta_on_override": float(np.mean(rerank_deltas)) if rerank_deltas else 0.0,
        "median_rerank_delta_on_override": float(np.median(rerank_deltas)) if rerank_deltas else 0.0,
        "avg_confirm_delta_on_override": float(np.mean(confirm_deltas)) if confirm_deltas else 0.0,
        "median_confirm_delta_on_override": float(np.median(confirm_deltas)) if confirm_deltas else 0.0,
        "avg_realized_delta_on_override": float(np.mean(realized_deltas)) if realized_deltas else 0.0,
        "median_realized_delta_on_override": float(np.median(realized_deltas)) if realized_deltas else 0.0,
        "performance_metric_source": TOPK_PERFORMANCE_METRIC_SOURCE,
        "per_fire_performance_column": "avg_realized_delta_on_override",
        "hand_ev_performance_column": "ev_per_hand",
        "rerank_delta_metric_role": TOPK_RERANK_DELTA_METRIC_ROLE,
        "confirm_delta_metric_role": TOPK_CONFIRM_DELTA_METRIC_ROLE,
        "confirm_delta_performance_claim_allowed": TOPK_CONFIRM_DELTA_PERFORMANCE_CLAIM_ALLOWED,
        "p95_rerank_loss": percentile(rerank_losses, 95),
        "max_rerank_loss": max(rerank_losses) if rerank_losses else 0.0,
        "p95_confirm_loss": percentile(confirm_losses, 95),
        "max_confirm_loss": max(confirm_losses) if confirm_losses else 0.0,
        "no_override_reason_counts": json.dumps(dict(no_override), sort_keys=True),
        "elapsed_seconds": time.time() - started_at,
    }
    return summary, decisions, position_scores


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), q / 100.0))


def stage_b_delta(row: dict[str, Any]) -> float:
    """Return the independent confirm-MC delta when present, else legacy rerank delta."""

    value = row.get("confirm_delta")
    if value not in (None, ""):
        return float(value)
    return float(row.get("rerank_delta", 0.0) or 0.0)


def stage_b_delta_se(row: dict[str, Any]) -> float:
    value = row.get("confirm_delta_se")
    if value not in (None, ""):
        return float(value)
    return float(row.get("rerank_delta_se", 0.0) or 0.0)


def aggregate_topk_seed_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        by_config[str(row["config_id"])].append(row)
    rows: list[dict[str, Any]] = []
    for config_id, group in by_config.items():
        total_games = sum(int(row["paired_seeds"]) for row in group)
        mean = sum(float(row["ev_per_hand"]) * int(row["paired_seeds"]) for row in group) / max(total_games, 1)
        seed_values = [float(row["ev_per_hand"]) for row in group]
        stderr = float(np.std(seed_values, ddof=1) / math.sqrt(len(seed_values))) if len(seed_values) > 1 else float(group[0].get("paired_score_std_error", 0.0))
        decisions = sum(int(item["decision_count"]) for item in group)
        overrides = sum(int(item["override_count"]) for item in group)
        realized_overrides = sum(int(item.get("realized_override_count", 0) or 0) for item in group)
        local_ev_risk_evaluated = sum(int(item.get("local_ev_risk_evaluated_count", 0) or 0) for item in group)
        local_ev_risk_would_vetoes = sum(
            int(item.get("local_ev_risk_would_veto_count", item.get("local_ev_risk_veto_count", 0)) or 0)
            for item in group
        )
        local_ev_risk_vetoes = sum(int(item.get("local_ev_risk_veto_count", 0) or 0) for item in group)
        stage8c_fire_selector_evaluated = sum(
            int(item.get("stage8c_fire_selector_evaluated_count", 0) or 0) for item in group
        )
        stage8c_fire_selector_passed = sum(
            int(item.get("stage8c_fire_selector_passed_count", 0) or 0) for item in group
        )
        stage8c_fire_selector_direct_fire = sum(
            int(item.get("stage8c_fire_selector_direct_fire_count", 0) or 0) for item in group
        )
        stage8c_fire_selector_blocked = sum(
            int(item.get("stage8c_fire_selector_blocked_count", 0) or 0) for item in group
        )
        realized_deltas = []
        confirm_deltas = []
        for item in group:
            if int(item.get("realized_override_count", 0) or 0):
                realized_deltas.extend(
                    [float(item.get("avg_realized_delta_on_override", 0.0))]
                    * int(item.get("realized_override_count", 0) or 0)
                )
            if int(item.get("override_count", 0) or 0):
                confirm_deltas.extend(
                    [float(item.get("avg_confirm_delta_on_override", item.get("avg_rerank_delta_on_override", 0.0)))]
                    * int(item.get("override_count", 0) or 0)
                )
        rows.append(
            {
                "config_id": config_id,
                "t3_continuation": group[0].get("t3_continuation", ""),
                "t3_continuation_policy": group[0].get("t3_continuation_policy", ""),
                "top_k": group[0]["top_k"],
                "mc_samples": group[0]["mc_samples"],
                "min_rerank_delta": group[0]["min_rerank_delta"],
                "se_multiplier": group[0]["se_multiplier"],
                "confirm_mc_samples": group[0].get("confirm_mc_samples", 0),
                "confirm_se_multiplier": group[0].get("confirm_se_multiplier", 0.0),
                "max_confirm_se": group[0].get("max_confirm_se", ""),
                "min_confirm_delta": group[0].get("min_confirm_delta", ""),
                "first_min_confirm_delta": group[0].get("first_min_confirm_delta", ""),
                "second_min_confirm_delta": group[0].get("second_min_confirm_delta", ""),
                "allowed_seats": group[0]["allowed_seats"],
                "candidate_ev_rank_max": group[0]["candidate_ev_rank_max"],
                "min_gate_probability": group[0]["min_gate_probability"],
                "min_predicted_delta": group[0]["min_predicted_delta"],
                "topk_score": group[0]["topk_score"],
                "stage8b_model_loaded": group[0].get("stage8b_model_loaded", ""),
                "stage8b_model_load_failed": group[0].get("stage8b_model_load_failed", ""),
                "local_ev_risk_model": group[0].get("local_ev_risk_model", ""),
                "local_ev_risk_threshold": group[0].get("local_ev_risk_threshold", ""),
                "stage8c_risk_model": group[0].get("stage8c_risk_model", group[0].get("local_ev_risk_model", "")),
                "stage8c_risk_threshold": group[0].get(
                    "stage8c_risk_threshold",
                    group[0].get("local_ev_risk_threshold", ""),
                ),
                "stage8c_risk_rank_min": group[0].get("stage8c_risk_rank_min", ""),
                "stage8c_risk_rank_max": group[0].get("stage8c_risk_rank_max", ""),
                "stage8c_risk_audit_only": group[0].get("stage8c_risk_audit_only", False),
                "stage8c_fire_selector_model": group[0].get("stage8c_fire_selector_model", ""),
                "stage8c_fire_selector_threshold": group[0].get("stage8c_fire_selector_threshold", ""),
                "stage8c_fire_selector_audit_only": group[0].get("stage8c_fire_selector_audit_only", False),
                "stage8c_fire_selector_direct_fire_enabled": group[0].get(
                    "stage8c_fire_selector_direct_fire_enabled",
                    False,
                ),
                "paired_seeds": total_games,
                "max_paired_seeds": sum(int(row.get("max_paired_seeds", row["paired_seeds"])) for row in group),
                "target_realized_overrides": group[0].get("target_realized_overrides", 0),
                "target_risk_vetoes": group[0].get("target_risk_vetoes", 0),
                "target_fire_selector_rejections": group[0].get("target_fire_selector_rejections", 0),
                "target_realized_overrides_reached_count": sum(
                    1 for row in group if str(row.get("target_realized_overrides_reached", "")).lower() in {"true", "1"}
                ),
                "target_risk_vetoes_reached_count": sum(
                    1 for row in group if str(row.get("target_risk_vetoes_reached", "")).lower() in {"true", "1"}
                ),
                "target_fire_selector_rejections_reached_count": sum(
                    1
                    for row in group
                    if str(row.get("target_fire_selector_rejections_reached", "")).lower() in {"true", "1"}
                ),
                "stop_reason_counts": json.dumps(
                    dict(Counter(str(row.get("stop_reason", "")) for row in group)),
                    sort_keys=True,
                ),
                "hands": sum(int(row["hands"]) for row in group),
                "aggregate_ev_per_hand": mean,
                "std_error_seed_means": stderr,
                "ci95_low_seed_means": mean - 1.96 * stderr,
                "ci95_high_seed_means": mean + 1.96 * stderr,
                "seed_count": len(group),
                "decision_count": decisions,
                "override_count": overrides,
                "realized_override_count": realized_overrides,
                "runtime_override_rate": overrides / max(decisions, 1),
                "local_ev_risk_evaluated_count": local_ev_risk_evaluated,
                "local_ev_risk_would_veto_count": local_ev_risk_would_vetoes,
                "local_ev_risk_veto_count": local_ev_risk_vetoes,
                "local_ev_risk_would_veto_rate_on_evaluated": local_ev_risk_would_vetoes / max(local_ev_risk_evaluated, 1),
                "local_ev_risk_veto_rate_on_evaluated": local_ev_risk_vetoes / max(local_ev_risk_evaluated, 1),
                "stage8c_fire_selector_evaluated_count": stage8c_fire_selector_evaluated,
                "stage8c_fire_selector_passed_count": stage8c_fire_selector_passed,
                "stage8c_fire_selector_direct_fire_count": stage8c_fire_selector_direct_fire,
                "stage8c_fire_selector_blocked_count": stage8c_fire_selector_blocked,
                "stage8c_fire_selector_pass_rate": stage8c_fire_selector_passed
                / max(stage8c_fire_selector_evaluated, 1),
                "paired_seed_wins": sum(int(row["paired_seed_wins"]) for row in group),
                "paired_seed_losses": sum(int(row["paired_seed_losses"]) for row in group),
                "paired_seed_ties": sum(int(row["paired_seed_ties"]) for row in group),
                "avg_gain_on_override": float(np.mean(realized_deltas)) if realized_deltas else 0.0,
                "avg_confirm_delta_on_override": float(np.mean(confirm_deltas)) if confirm_deltas else 0.0,
                "performance_metric_source": TOPK_PERFORMANCE_METRIC_SOURCE,
                "per_fire_performance_column": "avg_gain_on_override",
                "hand_ev_performance_column": "aggregate_ev_per_hand",
                "rerank_delta_metric_role": TOPK_RERANK_DELTA_METRIC_ROLE,
                "confirm_delta_metric_role": TOPK_CONFIRM_DELTA_METRIC_ROLE,
                "confirm_delta_performance_claim_allowed": TOPK_CONFIRM_DELTA_PERFORMANCE_CLAIM_ALLOWED,
            }
        )
    rows.sort(key=lambda item: float(item.get("aggregate_ev_per_hand", 0.0)), reverse=True)
    return rows


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
        overrides = sum(1 for row in group_decisions if row.get("override_fired"))
        output.append(
            {
                "config_id": key[0],
                "seat": key[1],
                "hands": len(scores),
                "ev_per_hand": float(np.mean(scores)) if scores else 0.0,
                "override_count": overrides,
                "decision_count": len(group_decisions),
                "override_rate": overrides / max(len(group_decisions), 1),
            }
        )
    return output


def latency_breakdown(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for config_id, group in group_by(decisions, "config_id").items():
        fired = [row for row in group if row.get("override_fired")]
        reranked = [row for row in group if float(row.get("mc_rerank_latency_ms", 0.0) or 0.0) > 0.0]
        for label, rows in (("all_decisions", group), ("reranked", reranked), ("override_fired", fired)):
            runtime = [float(row.get("runtime_latency_ms", 0.0) or 0.0) for row in rows]
            rerank = [float(row.get("mc_rerank_latency_ms", 0.0) or 0.0) for row in rows]
            confirm = [float(row.get("confirm_mc_latency_ms", 0.0) or 0.0) for row in rows]
            action_counts = [float(row.get("evaluated_action_count", 0.0) or 0.0) for row in rows]
            mc_samples = [float(row.get("mc_samples", 0.0) or 0.0) for row in rows]
            denom = sum(a * m for a, m in zip(action_counts, mc_samples))
            output.append(
                {
                    "config_id": config_id,
                    "scope": label,
                    "decision_count": len(rows),
                    "runtime_latency_ms_mean": float(np.mean(runtime)) if runtime else 0.0,
                    "runtime_latency_ms_p95": percentile(runtime, 95),
                    "mc_rerank_latency_ms_mean": float(np.mean(rerank)) if rerank else 0.0,
                    "mc_rerank_latency_ms_p95": percentile(rerank, 95),
                    "confirm_mc_latency_ms_mean": float(np.mean(confirm)) if confirm else 0.0,
                    "confirm_mc_latency_ms_p95": percentile(confirm, 95),
                    "evaluated_action_count_mean": float(np.mean(action_counts)) if action_counts else 0.0,
                    "ms_per_mc_action": sum(rerank) / max(denom, 1.0),
                }
            )
    return output


def conditional_override_metrics(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for config_id, group in group_by(decisions, "config_id").items():
        fired = [row for row in group if row.get("override_fired")]
        realized_rows = [
            row
            for row in fired
            if row.get("realized_delta_valid") and row.get("realized_candidate_seat_delta") not in (None, "")
        ]
        deltas = [float(row.get("realized_candidate_seat_delta", 0.0) or 0.0) for row in realized_rows]
        confirm_deltas = [stage_b_delta(row) for row in fired]
        override_rate = len(fired) / max(len(group), 1)
        if deltas:
            mean = float(np.mean(np.asarray(deltas, dtype=np.float64)))
            stderr = float(np.std(np.asarray(deltas, dtype=np.float64), ddof=1) / math.sqrt(len(deltas))) if len(deltas) > 1 else 0.0
        else:
            mean = 0.0
            stderr = 0.0
        output.append(
            {
                "config_id": config_id,
                "decision_count": len(group),
                "override_count": len(fired),
                "realized_override_count": len(deltas),
                "override_rate": override_rate,
                "per_override_delta_mean": mean,
                "per_override_delta_std_error": stderr,
                "per_override_delta_ci95_low": mean - 1.96 * stderr,
                "per_override_delta_ci95_high": mean + 1.96 * stderr,
                "estimated_ev_per_hand": override_rate * mean,
                "estimated_ev_per_hand_ci95_low": override_rate * (mean - 1.96 * stderr),
                "estimated_ev_per_hand_ci95_high": override_rate * (mean + 1.96 * stderr),
                "p95_loss": percentile([max(0.0, -delta) for delta in deltas], 95),
                "max_loss": max([max(0.0, -delta) for delta in deltas], default=0.0),
                "confirm_delta_mean_on_fired": float(np.mean(np.asarray(confirm_deltas, dtype=np.float64))) if confirm_deltas else 0.0,
                "performance_metric_source": TOPK_PERFORMANCE_METRIC_SOURCE,
                "per_fire_performance_column": TOPK_PER_FIRE_PERFORMANCE_COLUMN,
                "hand_ev_performance_column": TOPK_HAND_EV_PERFORMANCE_COLUMN,
                "confirm_delta_metric_role": TOPK_CONFIRM_DELTA_METRIC_ROLE,
                "confirm_delta_performance_claim_allowed": TOPK_CONFIRM_DELTA_PERFORMANCE_CLAIM_ALLOWED,
            }
        )
    output.sort(key=lambda item: float(item.get("estimated_ev_per_hand", 0.0)), reverse=True)
    return output


def risk_veto_candidate_metrics(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for config_id, group in group_by(decisions, "config_id").items():
        selected = [
            row
            for row in group
            if row.get("local_ev_risk_would_veto", row.get("local_ev_risk_vetoed"))
        ]
        realized_rows = [
            row
            for row in selected
            if row.get("realized_delta_valid")
            and row.get("realized_candidate_seat_delta") not in (None, "")
        ]
        deltas = [float(row.get("realized_candidate_seat_delta", 0.0) or 0.0) for row in realized_rows]
        losses = [max(0.0, -delta) for delta in deltas]
        would_veto_rate = len(selected) / max(len(group), 1)
        if deltas:
            mean = float(np.mean(np.asarray(deltas, dtype=np.float64)))
            stderr = float(np.std(np.asarray(deltas, dtype=np.float64), ddof=1) / math.sqrt(len(deltas))) if len(deltas) > 1 else 0.0
        else:
            mean = 0.0
            stderr = 0.0
        veto_utility = -mean
        output.append(
            {
                "config_id": config_id,
                "decision_count": len(group),
                "would_veto_count": len(selected),
                "audit_only_would_veto_count": sum(1 for row in selected if row.get("local_ev_risk_audit_only")),
                "actual_veto_count": sum(1 for row in selected if row.get("local_ev_risk_vetoed")),
                "realized_would_veto_count": len(deltas),
                "would_veto_rate": would_veto_rate,
                "realized_candidate_delta_mean": mean,
                "realized_candidate_delta_std_error": stderr,
                "realized_candidate_delta_ci95_low": mean - 1.96 * stderr,
                "realized_candidate_delta_ci95_high": mean + 1.96 * stderr,
                "veto_utility_per_veto": veto_utility,
                "estimated_veto_utility_per_hand": would_veto_rate * veto_utility,
                "candidate_p95_loss": percentile(losses, 95),
                "candidate_max_loss": max(losses, default=0.0),
            }
        )
    output.sort(key=lambda item: float(item.get("estimated_veto_utility_per_hand", 0.0)), reverse=True)
    return output


def predicted_delta_safety_audit(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Audit the known bad pattern where MC overrides model-negative actions."""
    output: list[dict[str, Any]] = []
    for config_id, group in group_by(decisions, "config_id").items():
        fired = [row for row in group if row.get("override_fired")]
        negative_predicted = [
            row
            for row in fired
            if row.get("predicted_delta") not in (None, "") and float(row.get("predicted_delta", 0.0) or 0.0) < 0.0
        ]
        realized_negative_predicted = [
            row
            for row in negative_predicted
            if row.get("realized_delta_valid")
            and row.get("realized_candidate_seat_delta") not in (None, "")
            and float(row.get("realized_candidate_seat_delta", 0.0) or 0.0) < 0.0
        ]
        realized_fired = [
            row
            for row in fired
            if row.get("realized_delta_valid") and row.get("realized_candidate_seat_delta") not in (None, "")
        ]
        realized_losses = [
            row
            for row in realized_fired
            if float(row.get("realized_candidate_seat_delta", 0.0) or 0.0) < 0.0
        ]
        negative_predicted_losses = [
            max(0.0, -float(row.get("realized_candidate_seat_delta", 0.0) or 0.0))
            for row in realized_negative_predicted
        ]
        output.append(
            {
                "config_id": config_id,
                "decision_count": len(group),
                "override_count": len(fired),
                "realized_override_count": len(realized_fired),
                "negative_predicted_delta_override_count": len(negative_predicted),
                "negative_predicted_delta_override_rate": len(negative_predicted) / max(len(fired), 1),
                "realized_loss_count": len(realized_losses),
                "negative_predicted_delta_realized_loss_count": len(realized_negative_predicted),
                "negative_predicted_delta_realized_loss_sum": -float(sum(negative_predicted_losses)),
                "negative_predicted_delta_max_loss": max(negative_predicted_losses, default=0.0),
                "safety_blocker": int(bool(realized_negative_predicted)),
            }
        )
    return output


def has_safety_blocker(rows: Iterable[dict[str, Any]]) -> bool:
    return any(int(row.get("safety_blocker", 0) or 0) for row in rows)


def has_safety_blocker_for_config(best: dict[str, Any], rows: Iterable[dict[str, Any]]) -> bool:
    config_id = str(best.get("config_id", ""))
    if not config_id:
        return False
    return any(
        str(row.get("config_id", "")) == config_id and int(row.get("safety_blocker", 0) or 0)
        for row in rows
    )


def cancellation_audit(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for config_id, group in group_by(decisions, "config_id").items():
        valid = [
            row
            for row in group
            if row.get("realized_delta_valid") and row.get("realized_candidate_seat_delta") not in (None, "")
        ]
        non_fired = [row for row in valid if not row.get("override_fired")]
        fired = [row for row in valid if row.get("override_fired")]
        non_fired_deltas = [float(row.get("realized_candidate_seat_delta", 0.0) or 0.0) for row in non_fired]
        fired_deltas = [float(row.get("realized_candidate_seat_delta", 0.0) or 0.0) for row in fired]
        nonzero = [delta for delta in non_fired_deltas if abs(delta) > 1e-9]
        output.append(
            {
                "config_id": config_id,
                "decision_count": len(group),
                "valid_realized_delta_count": len(valid),
                "non_fired_count": len(non_fired),
                "non_fired_nonzero_count": len(nonzero),
                "non_fired_delta_sum": float(sum(non_fired_deltas)),
                "non_fired_delta_max_abs": max([abs(delta) for delta in non_fired_deltas], default=0.0),
                "fired_count": len(fired),
                "fired_delta_sum": float(sum(fired_deltas)),
                "fired_delta_mean": float(np.mean(np.asarray(fired_deltas, dtype=np.float64))) if fired_deltas else 0.0,
            }
        )
    return output


def cancellation_is_clean(rows: Iterable[dict[str, Any]]) -> bool:
    rows = list(rows)
    if not rows:
        return False
    return all(
        int(row.get("non_fired_nonzero_count", 0) or 0) == 0
        and abs(float(row.get("non_fired_delta_sum", 0.0) or 0.0)) <= 1e-9
        and float(row.get("non_fired_delta_max_abs", 0.0) or 0.0) <= 1e-9
        for row in rows
    )


def cancellation_row_for_config(best: dict[str, Any], rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    config_id = str(best.get("config_id", ""))
    if not config_id:
        return None
    for row in rows:
        if str(row.get("config_id", "")) == config_id:
            return row
    return None


def cancellation_is_clean_for_config(best: dict[str, Any], rows: Iterable[dict[str, Any]]) -> bool:
    row = cancellation_row_for_config(best, rows)
    if row is None:
        return False
    return bool(
        int(row.get("non_fired_nonzero_count", 0) or 0) == 0
        and abs(float(row.get("non_fired_delta_sum", 0.0) or 0.0)) <= 1e-9
        and float(row.get("non_fired_delta_max_abs", 0.0) or 0.0) <= 1e-9
    )


def conditional_per_fire_row_for_config(
    best: dict[str, Any],
    conditional_rows: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    config_id = str(best.get("config_id", ""))
    if not config_id:
        return None
    for row in conditional_rows:
        if str(row.get("config_id", "")) == config_id:
            return row
    return None


def conditional_per_fire_is_positive(
    best: dict[str, Any],
    conditional_rows: Iterable[dict[str, Any]],
) -> bool:
    row = conditional_per_fire_row_for_config(best, conditional_rows)
    if row is None:
        return False
    return bool(
        int(row.get("realized_override_count", 0) or 0) > 0
        and float(row.get("per_override_delta_mean", 0.0) or 0.0) > 0.0
        and float(row.get("estimated_ev_per_hand", 0.0) or 0.0) > 0.0
    )


def topk_conditional_go(
    best: dict[str, Any],
    *,
    conditional_rows: Iterable[dict[str, Any]],
    cancellation_rows: Iterable[dict[str, Any]],
    safety_rows: Iterable[dict[str, Any]],
) -> bool:
    return bool(
        best
        and conditional_per_fire_is_positive(best, conditional_rows)
        and cancellation_is_clean_for_config(best, cancellation_rows)
        and not has_safety_blocker_for_config(best, safety_rows)
        and float(best.get("aggregate_ev_per_hand", 0.0)) > 0.0
        and float(best.get("ci95_low_seed_means", -999.0)) > -0.05
    )


def runtime_decision_buckets(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for config_id, group in group_by(decisions, "config_id").items():
        for field in ("no_override_reason", "override_fired"):
            for label, rows in group_by(group, field).items():
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


def failure_rows(decisions: list[dict[str, Any]], limit: int = 30) -> list[dict[str, Any]]:
    failures = [row for row in decisions if row.get("override_fired")]
    failures.sort(key=lambda row: float(row.get("candidate_seat_score", 0.0) or 0.0))
    output: list[dict[str, Any]] = []
    for row in failures[:limit]:
        output.append(
            {
                "config_id": row.get("config_id"),
                "seed": row.get("seed"),
                "hand_id": row.get("hand_id"),
                "hand_seed": row.get("hand_seed"),
                "seat": row.get("seat"),
                "seat_swap": row.get("seat_swap"),
                "candidate_seat_score": row.get("candidate_seat_score"),
                "hero_board": row.get("hero_board"),
                "opponent_board": row.get("opponent_board"),
                "dead_cards": row.get("dead_cards"),
                "cards_to_place": row.get("cards_to_place"),
                "baseline_action": row.get("baseline_action"),
                "stage8b_top1_action": row.get("stage8b_top1_action"),
                "rerank_best_action": row.get("rerank_best_action"),
                "final_action": row.get("final_action"),
                "predicted_delta": row.get("predicted_delta"),
                "gate_probability": row.get("gate_probability"),
                "candidate_ev_rank": row.get("candidate_ev_rank"),
                "confirm_delta": row.get("confirm_delta"),
                "confirm_delta_se": row.get("confirm_delta_se"),
                "local_ev_risk_probability": row.get("local_ev_risk_probability"),
                "local_ev_risk_threshold": row.get("local_ev_risk_threshold"),
                "local_ev_risk_would_veto": row.get("local_ev_risk_would_veto"),
                "local_ev_risk_vetoed": row.get("local_ev_risk_vetoed"),
                "local_ev_risk_audit_only": row.get("local_ev_risk_audit_only"),
                "rerank_delta": row.get("rerank_delta"),
                "rerank_delta_se": row.get("rerank_delta_se"),
                "failure_label": classify_failure(row),
            }
        )
    return output


def classify_failure(row: dict[str, Any]) -> str:
    if stage_b_delta(row) < float(row.get("min_rerank_delta", 0.0) or 0.0):
        return "mc_rerank_threshold_leak"
    if float(row.get("candidate_seat_score", 0.0) or 0.0) < -20.0:
        return "tail_loss_audit_required"
    return "other"


def group_by(rows: Iterable[dict[str, Any]], field: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field, "unknown"))].append(row)
    return grouped


def write_summary(
    path: Path,
    rows: list[dict[str, Any]],
    conditional_rows: list[dict[str, Any]],
    risk_veto_rows: list[dict[str, Any]],
    cancellation_rows: list[dict[str, Any]],
    safety_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    best = rows[0] if rows else {}
    cancellation_clean = cancellation_is_clean_for_config(best, cancellation_rows)
    cancellation_clean_all_configs = cancellation_is_clean(cancellation_rows)
    lines = [
        "# HU T2 Stage8b TopK + MC Rerank Validation",
        "",
        "This is validation-only. It does not authorize 50k teacher generation, T1 training, production training, P2 fixed status, or production runtime changes.",
        "",
        "## Inputs",
        "",
        f"- T3 continuation: `{_t3_continuation_policy_name(args.t3_continuation)}`",
        f"- t3_continuation: `{args.t3_continuation}`",
        "- Stage8b role: candidate generator only, not direct override",
        f"- model: `{args.hu_turn2_stage8b_model}`",
        f"- allow missing Stage8b model fallback: `{getattr(args, 'allow_missing_stage8b_model_fallback', False)}`",
        f"- Stage8b model load failed: `{getattr(args, 'stage8b_model_load_failed', False)}`",
        f"- Stage8b model load error: `{getattr(args, 'stage8b_model_load_error', '')}`",
        f"- seeds: `{args.seeds}`",
        f"- games_per_seed: `{args.games_per_seed}`",
        f"- target_realized_overrides_per_seed: `{args.target_realized_overrides_per_seed}`",
        f"- target_risk_vetoes_per_seed: `{args.target_risk_vetoes_per_seed}`",
        f"- target_fire_selector_rejections_per_seed: `{getattr(args, 'target_fire_selector_rejections_per_seed', 0)}`",
        f"- seed_stride: `{args.seed_stride}`",
        "- Two-stage configs use `confirm_mc_samples > 0`: Stage A selects a champion, Stage B rerolls champion vs baseline with an independent future seed.",
        f"- Stage8c risk veto model: `{getattr(args, 'stage8c_risk_model_resolved', None) or getattr(args, 'stage8c_risk_model', None) or getattr(args, 'local_ev_risk_model', None) or ''}`",
        f"- Stage8c risk veto threshold: `{getattr(args, 'stage8c_risk_threshold_resolved', None) if (getattr(args, 'stage8c_risk_model_resolved', None) or getattr(args, 'stage8c_risk_model', None) or getattr(args, 'local_ev_risk_model', None)) else ''}`",
        f"- Stage8c risk veto rank guard: `{getattr(args, 'stage8c_risk_rank_min', None) or ''}` to `{getattr(args, 'stage8c_risk_rank_max', None) or ''}`",
        f"- Stage8c risk audit only: `{getattr(args, 'stage8c_risk_audit_only', False)}`",
        "- Stage8c risk probability is a runtime veto input only, not a realized performance estimate.",
        f"- Stage8c fire selector model: `{getattr(args, 'stage8c_fire_selector_model', None) or ''}`",
        f"- Stage8c fire selector threshold: `{getattr(args, 'stage8c_fire_selector_threshold', '') if getattr(args, 'stage8c_fire_selector_model', None) else ''}`",
        f"- Stage8c fire selector audit only: `{getattr(args, 'stage8c_fire_selector_audit_only', False)}`",
        f"- Stage8c fire selector direct-fire: `{getattr(args, 'stage8c_fire_selector_direct_fire', False)}`",
        (
            "- Stage9e direct-fire skips MC/confirm and is a latency experiment; "
            "realized whole-game deltas remain the only performance metric."
            if getattr(args, "stage8c_fire_selector_direct_fire", False)
            else "- Stage8c fire selector probability is a pre-confirm candidate filter only; confirm/realized deltas remain the evaluation gates."
        ),
        f"- performance metric source: `{TOPK_PERFORMANCE_METRIC_SOURCE}`",
        f"- aggregate hand EV performance column: `aggregate_ev_per_hand`",
        f"- conditional per-fire performance column: `{TOPK_PER_FIRE_PERFORMANCE_COLUMN}`",
        f"- conditional hand EV performance column: `{TOPK_HAND_EV_PERFORMANCE_COLUMN}`",
        f"- rerank delta role: `{TOPK_RERANK_DELTA_METRIC_ROLE}`",
        f"- confirm delta role: `{TOPK_CONFIRM_DELTA_METRIC_ROLE}`",
        f"- confirm delta performance claim allowed: `{TOPK_CONFIRM_DELTA_PERFORMANCE_CLAIM_ALLOWED}`",
        f"- non-fired cancellation required for Conditional-Go: `{NON_FIRED_CANCELLATION_REQUIRED_FOR_CONDITIONAL_GO}`",
        f"- realized per-fire required for Conditional-Go: `{CONDITIONAL_REALIZED_PER_FIRE_REQUIRED_FOR_CONDITIONAL_GO}`",
        f"- confirm/rerank diagnostics used for Conditional-Go: `{CONFIRM_RERANK_DIAGNOSTICS_USED_FOR_CONDITIONAL_GO}`",
        f"- cancellation clean for best config: `{cancellation_clean}`",
        f"- cancellation clean in all configs: `{cancellation_clean_all_configs}`",
        "",
        "## Results",
        "",
        "| config | paired | EV/hand | CI low | CI high | overrides | risk vetoes | override rate | avg realized gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
                "| {config_id} | {paired} | {ev:.4f} | {low:.4f} | {high:.4f} | {overrides} | {vetoes} | {rate:.4f} | {gain:.4f} |".format(
                    config_id=row.get("config_id", ""),
                    paired=row.get("paired_seeds", ""),
                    ev=float(row.get("aggregate_ev_per_hand", 0.0)),
                    low=float(row.get("ci95_low_seed_means", 0.0)),
                    high=float(row.get("ci95_high_seed_means", 0.0)),
                    overrides=row.get("override_count", ""),
                    vetoes=row.get("local_ev_risk_veto_count", 0),
                    rate=float(row.get("runtime_override_rate", 0.0)),
                    gain=float(row.get("avg_gain_on_override", 0.0)),
                )
        )
    if conditional_rows:
        lines.extend(
            [
                "",
                "## Conditional Override Estimate",
                "",
                "This uses fired decisions only, but the delta is the realized seat-swap counterfactual delta, not the confirm MC gate estimate.",
                "`estimated_EV/hand = override_rate * mean(realized_per_override_delta)`.",
                "The `confirm_delta_mean_on_fired` column is a gate diagnostic and must not be cited as realized performance.",
                "",
                "| config | overrides | realized | override rate | realized per-fire delta | confirm delta mean | estimated EV/hand |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in conditional_rows:
            lines.append(
                "| {config_id} | {overrides} | {realized} | {rate:.4f} | {mean:.4f} | {confirm:.4f} | {ev:.4f} |".format(
                    config_id=row.get("config_id", ""),
                    overrides=row.get("override_count", ""),
                    realized=row.get("realized_override_count", ""),
                    rate=float(row.get("override_rate", 0.0)),
                    mean=float(row.get("per_override_delta_mean", 0.0)),
                    confirm=float(row.get("confirm_delta_mean_on_fired", 0.0)),
                    ev=float(row.get("estimated_ev_per_hand", 0.0)),
                )
            )
    if risk_veto_rows:
        lines.extend(
            [
                "",
                "## Risk Veto Candidate Estimate",
                "",
                "This uses rows where the risk gate would veto. In audit-only mode these rows still fire, so the realized candidate delta can estimate veto utility.",
                "",
                "| config | would-veto | realized | would-veto rate | realized candidate delta | veto utility/veto | est veto utility/hand | actual vetoes |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in risk_veto_rows:
            lines.append(
                "| {config_id} | {would_veto} | {realized} | {rate:.4f} | {delta:.4f} | {utility:.4f} | {ev:.4f} | {actual} |".format(
                    config_id=row.get("config_id", ""),
                    would_veto=row.get("would_veto_count", 0),
                    realized=row.get("realized_would_veto_count", 0),
                    rate=float(row.get("would_veto_rate", 0.0)),
                    delta=float(row.get("realized_candidate_delta_mean", 0.0)),
                    utility=float(row.get("veto_utility_per_veto", 0.0)),
                    ev=float(row.get("estimated_veto_utility_per_hand", 0.0)),
                    actual=row.get("actual_veto_count", 0),
                )
            )
    if safety_rows:
        lines.extend(
            [
                "",
                "## Safety Audit",
                "",
                "Known bad pattern: MC/confirm rerank overriding an action whose model `predicted_delta` is negative.",
                "",
                "| config | overrides | negative-pred overrides | realized losses from negative-pred | safety blocker |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in safety_rows:
            lines.append(
                "| {config_id} | {overrides} | {neg} | {losses} | {blocker} |".format(
                    config_id=row.get("config_id", ""),
                    overrides=row.get("override_count", 0),
                    neg=row.get("negative_predicted_delta_override_count", 0),
                    losses=row.get("negative_predicted_delta_realized_loss_count", 0),
                    blocker=row.get("safety_blocker", 0),
                )
            )
    safety_blocker = has_safety_blocker(safety_rows)
    best_safety_blocker = has_safety_blocker_for_config(best, safety_rows)
    conditional_per_fire_positive = conditional_per_fire_is_positive(best, conditional_rows)
    conditional_best = conditional_per_fire_row_for_config(best, conditional_rows) or {}
    go = topk_conditional_go(
        best,
        conditional_rows=conditional_rows,
        cancellation_rows=cancellation_rows,
        safety_rows=safety_rows,
    )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- TopK+MC rerank validation: `{'Conditional-Go' if go else 'No-Go'}`",
            f"- conditional realized per-fire positive: `{'yes' if conditional_per_fire_positive else 'no'}`",
            f"- conditional realized overrides for best config: `{conditional_best.get('realized_override_count', 0)}`",
            f"- non-fired cancellation clean for best config: `{'yes' if cancellation_clean else 'no'}`",
            f"- non-fired cancellation clean in all configs: `{'yes' if cancellation_clean_all_configs else 'no'}`",
            f"- known negative-predicted-delta safety blocker for best config: `{'yes' if best_safety_blocker else 'no'}`",
            f"- known negative-predicted-delta safety blocker in any config: `{'yes' if safety_blocker else 'no'}`",
            "- production / P2 fixed: `No-Go`",
            "- 50k teacher: `No-Go`",
            "- T1 training: `No-Go`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_go_nogo(
    path: Path,
    result_rows: list[dict[str, Any]],
    conditional_rows: list[dict[str, Any]],
    cancellation_rows: list[dict[str, Any]],
    safety_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    stage8c_risk_model: str | Path | None,
    stage8c_risk_threshold: float,
    elapsed_seconds: float,
) -> None:
    best = result_rows[0] if result_rows else {}
    safety_blocker = has_safety_blocker(safety_rows)
    best_safety_blocker = has_safety_blocker_for_config(best, safety_rows)
    cancellation_clean = cancellation_is_clean_for_config(best, cancellation_rows)
    cancellation_clean_all_configs = cancellation_is_clean(cancellation_rows)
    conditional_per_fire_positive = conditional_per_fire_is_positive(best, conditional_rows)
    conditional_best = conditional_per_fire_row_for_config(best, conditional_rows) or {}
    go = topk_conditional_go(
        best,
        conditional_rows=conditional_rows,
        cancellation_rows=cancellation_rows,
        safety_rows=safety_rows,
    )
    path.write_text(
        "\n".join(
            [
                "# HU T2 Stage8b TopK + MC Rerank Go / No-Go",
                "",
                f"- execution: `Pass`",
                f"- decision: `{'Conditional-Go' if go else 'No-Go'}`",
                f"- t3_continuation: `{args.t3_continuation}`",
                f"- t3_continuation_policy: `{_t3_continuation_policy_name(args.t3_continuation)}`",
                f"- allow_missing_stage8b_model_fallback: `{getattr(args, 'allow_missing_stage8b_model_fallback', False)}`",
                f"- stage8b_model_load_failed: `{getattr(args, 'stage8b_model_load_failed', False)}`",
                f"- stage8c_risk_model: `{stage8c_risk_model or ''}`",
                f"- stage8c_risk_threshold: `{stage8c_risk_threshold if stage8c_risk_model else ''}`",
                f"- stage8c_risk_rank_min: `{args.stage8c_risk_rank_min if stage8c_risk_model else ''}`",
                f"- stage8c_risk_rank_max: `{args.stage8c_risk_rank_max if stage8c_risk_model else ''}`",
                f"- stage8c_fire_selector_direct_fire: `{getattr(args, 'stage8c_fire_selector_direct_fire', False)}`",
                f"- known_negative_predicted_delta_safety_blocker_for_best_config: `{'yes' if best_safety_blocker else 'no'}`",
                f"- known_negative_predicted_delta_safety_blocker_any_config: `{'yes' if safety_blocker else 'no'}`",
                f"- performance_metric_source: `{TOPK_PERFORMANCE_METRIC_SOURCE}`",
                f"- aggregate_hand_ev_performance_column: `aggregate_ev_per_hand`",
                f"- conditional_per_fire_performance_column: `{TOPK_PER_FIRE_PERFORMANCE_COLUMN}`",
                f"- conditional_hand_ev_performance_column: `{TOPK_HAND_EV_PERFORMANCE_COLUMN}`",
                f"- rerank_delta_metric_role: `{TOPK_RERANK_DELTA_METRIC_ROLE}`",
                f"- confirm_delta_metric_role: `{TOPK_CONFIRM_DELTA_METRIC_ROLE}`",
                f"- confirm_delta_performance_claim_allowed: `{TOPK_CONFIRM_DELTA_PERFORMANCE_CLAIM_ALLOWED}`",
                f"- conditional_realized_per_fire_positive: `{'yes' if conditional_per_fire_positive else 'no'}`",
                f"- conditional_realized_overrides_for_best_config: `{conditional_best.get('realized_override_count', 0)}`",
                f"- conditional_realized_per_fire_required_for_conditional_go: `{CONDITIONAL_REALIZED_PER_FIRE_REQUIRED_FOR_CONDITIONAL_GO}`",
                f"- confirm_rerank_diagnostics_used_for_conditional_go: `{CONFIRM_RERANK_DIAGNOSTICS_USED_FOR_CONDITIONAL_GO}`",
                f"- non_fired_cancellation_required_for_conditional_go: `{NON_FIRED_CANCELLATION_REQUIRED_FOR_CONDITIONAL_GO}`",
                f"- cancellation_clean_for_best_config: `{cancellation_clean}`",
                f"- cancellation_clean_all_configs: `{cancellation_clean_all_configs}`",
                f"- cancellation_clean: `{cancellation_clean}`",
                "- production / P2 fixed: `No-Go`",
                "- 50k teacher: `No-Go`",
                "- T1 training: `No-Go`",
                f"- elapsed_seconds: `{elapsed_seconds:.2f}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.games_per_seed <= 0:
        raise SystemExit("--games-per-seed must be positive")
    if args.target_realized_overrides_per_seed < 0:
        raise SystemExit("--target-realized-overrides-per-seed must be non-negative")
    if args.target_risk_vetoes_per_seed < 0:
        raise SystemExit("--target-risk-vetoes-per-seed must be non-negative")
    if args.target_fire_selector_rejections_per_seed < 0:
        raise SystemExit("--target-fire-selector-rejections-per-seed must be non-negative")
    stage8c_risk_model = args.stage8c_risk_model or args.local_ev_risk_model
    stage8c_risk_threshold = (
        float(args.stage8c_risk_threshold)
        if args.stage8c_risk_threshold is not None
        else float(args.local_ev_risk_threshold)
    )
    if not 0.0 <= stage8c_risk_threshold <= 1.0:
        raise SystemExit("--stage8c-risk-threshold must be in [0, 1]")
    if not 0.0 <= float(args.stage8c_fire_selector_threshold) <= 1.0:
        raise SystemExit("--stage8c-fire-selector-threshold must be in [0, 1]")
    if args.stage8c_fire_selector_direct_fire and args.stage8c_fire_selector_model is None:
        raise SystemExit("--stage8c-fire-selector-direct-fire requires --stage8c-fire-selector-model")
    if args.stage8c_fire_selector_direct_fire and args.stage8c_fire_selector_audit_only:
        raise SystemExit("--stage8c-fire-selector-direct-fire cannot be combined with --stage8c-fire-selector-audit-only")
    if args.stage8c_risk_rank_min is not None and args.stage8c_risk_rank_min <= 0:
        raise SystemExit("--stage8c-risk-rank-min must be positive")
    if args.stage8c_risk_rank_max is not None and args.stage8c_risk_rank_max <= 0:
        raise SystemExit("--stage8c-risk-rank-max must be positive")
    if (
        args.stage8c_risk_rank_min is not None
        and args.stage8c_risk_rank_max is not None
        and args.stage8c_risk_rank_min > args.stage8c_risk_rank_max
    ):
        raise SystemExit("--stage8c-risk-rank-min must be <= --stage8c-risk-rank-max")
    if args.prediction_threads > 0:
        os.environ.setdefault("OMP_NUM_THREADS", str(args.prediction_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.prediction_threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.prediction_threads))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configs = parse_topk_configs(args.configs)
    seeds = parse_seeds(args.seeds)
    parts = load_topk_parts(args)
    stage8b_model = parts.hu_turn2_stage8
    stage8c_risk_scorer = (
        LocalEvRiskScorer.load(stage8c_risk_model, device=args.device)
        if stage8c_risk_model is not None
        else None
    )
    stage8c_fire_selector_scorer = (
        LocalEvRiskScorer.load(args.stage8c_fire_selector_model, device=args.device)
        if args.stage8c_fire_selector_model is not None
        else None
    )
    started_at = time.time()
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
                    target_realized_overrides=args.target_realized_overrides_per_seed,
                    target_risk_vetoes=args.target_risk_vetoes_per_seed,
                    target_fire_selector_rejections=args.target_fire_selector_rejections_per_seed,
                    parts=parts,
                    stage8b_model=stage8b_model,
                    opening_lookahead_samples=args.opening_lookahead_samples,
                    batched_continuation_batch_size=args.batched_continuation_batch_size,
                    stage3_feature_encoder_mode=args.stage3_feature_encoder_mode,
                    t3_continuation=args.t3_continuation,
                    local_ev_risk_scorer=stage8c_risk_scorer,
                    local_ev_risk_threshold=stage8c_risk_threshold,
                    local_ev_risk_rank_min=args.stage8c_risk_rank_min,
                    local_ev_risk_rank_max=args.stage8c_risk_rank_max,
                    local_ev_risk_audit_only=args.stage8c_risk_audit_only,
                    stage8c_fire_selector_scorer=stage8c_fire_selector_scorer,
                    stage8c_fire_selector_threshold=float(args.stage8c_fire_selector_threshold),
                    stage8c_fire_selector_audit_only=args.stage8c_fire_selector_audit_only,
                    stage8c_fire_selector_direct_fire=args.stage8c_fire_selector_direct_fire,
                    progress_every=args.progress_every,
                )
                seed_rows.append(summary)
                decision_rows.extend(decisions)
                position_scores.extend(scores)

    result_rows = aggregate_topk_seed_rows(seed_rows)
    conditional_rows = conditional_override_metrics(decision_rows)
    risk_veto_rows = risk_veto_candidate_metrics(decision_rows)
    cancellation_rows = cancellation_audit(decision_rows)
    safety_rows = predicted_delta_safety_audit(decision_rows)
    write_csv(args.output_dir / "topk_rerank_results.csv", result_rows)
    write_csv(args.output_dir / "seed_breakdown.csv", seed_rows)
    write_csv(args.output_dir / "position_breakdown.csv", runtime_position_breakdown(position_scores, decision_rows))
    write_csv(args.output_dir / "latency_breakdown.csv", latency_breakdown(decision_rows))
    write_csv(args.output_dir / "no_override_reason_counts.csv", runtime_decision_buckets(decision_rows))
    write_csv(args.output_dir / "conditional_override_metrics.csv", conditional_rows)
    write_csv(args.output_dir / "risk_veto_candidate_metrics.csv", risk_veto_rows)
    write_csv(args.output_dir / "cancellation_audit.csv", cancellation_rows)
    write_csv(args.output_dir / "predicted_delta_safety_audit.csv", safety_rows)
    write_jsonl(args.output_dir / "failure_top30.jsonl", failure_rows(decision_rows))
    if args.write_decision_log:
        write_jsonl(args.output_dir / "runtime_decisions.jsonl", decision_rows)
    args.stage8c_risk_model_resolved = stage8c_risk_model
    args.stage8c_risk_threshold_resolved = stage8c_risk_threshold
    write_summary(
        args.output_dir / "topk_rerank_summary.md",
        result_rows,
        conditional_rows,
        risk_veto_rows,
        cancellation_rows,
        safety_rows,
        args,
    )
    write_go_nogo(
        args.output_dir / "go_nogo.md",
        result_rows,
        conditional_rows,
        cancellation_rows,
        safety_rows,
        args,
        stage8c_risk_model=stage8c_risk_model,
        stage8c_risk_threshold=stage8c_risk_threshold,
        elapsed_seconds=time.time() - started_at,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "configs": len(configs), "seeds": len(seeds), "elapsed_seconds": time.time() - started_at}, indent=2))


if __name__ == "__main__":
    main()
