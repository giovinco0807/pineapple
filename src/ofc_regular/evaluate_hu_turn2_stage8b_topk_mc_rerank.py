"""Validate HU T2 Stage8b as a TopK candidate generator with MC rerank.

This evaluator deliberately does not use Stage8b as a direct production gate.
Stage8b scores all legal HU T2 actions, keeps a small TopK candidate set plus
the current baseline action, then reranks that reduced set by MC rollout with
the fixed Stage7_candidate_A m5_r10 T3 continuation.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .action_space import Action, generate_turn_actions
from .ai_profiles import DEFAULT_HU_TURN3_STAGE7_MIN_MARGIN, DEFAULT_HU_TURN3_STAGE7_REFERENCE_MIN_MARGIN
from .evaluate_hu_turn2_stage8_seat_swap import (
    DEFAULT_CALIBRATION_VALUES,
    DEFAULT_STAGE3_REFERENCE,
    DEFAULT_STAGE7_MODEL,
    DEFAULT_T2_STAGE8_MODEL,
    ModelParts,
    load_parts,
    make_baseline_policy,
    parse_seeds,
    summarize_values,
    write_csv,
    write_jsonl,
)
from .evaluate_matchups import trace_hand
from .final_turn_decision_cache import FinalTurnDecisionCache
from .hu_turn2_stage8_runtime import _safe_predictions, hu_turn2_policy_sample, sigmoid
from .hu_turn2_teacher_data import evaluate_hu_turn2_actions
from .hu_turn3_batch_continuation import (
    HuTurn3ActionCache,
    HuTurn3DecisionCache,
    HuTurn3Stage3ReferenceCache,
    HuTurn3Stage7BatchConfig,
    Stage3StateFeatureCache,
)
from .play_ai import _prediction_thread_context
from .policy import RegularAiPolicy, action_to_json, board_to_json, policy_sample
from .state import Board
from .turn3_model import load_action_value_model


DEFAULT_OUTPUT_DIR = Path("outputs/evals/hu_turn2_stage8b_topk_mc_rerank")


@dataclass(frozen=True)
class TopKMcRerankConfig:
    top_k: int
    mc_samples: int
    min_delta: float
    se_multiplier: float = 0.0
    allowed_seats: tuple[str, ...] = ()
    candidate_ev_rank_max: int | None = None
    min_gate_probability: float | None = None
    topk_score: str = "delta"
    # Confirmation MC samples for the two-stage rerank. The selection MC
    # picks the candidate; an independent-stream confirmation MC re-checks
    # candidate vs baseline so the winner's selection noise cannot inflate
    # the gate delta. -1 = same as mc_samples, 0 = disabled (legacy
    # single-stage gate, selection-biased).
    confirm_mc_samples: int = -1

    @property
    def resolved_confirm_mc_samples(self) -> int:
        if self.confirm_mc_samples < 0:
            return self.mc_samples
        return self.confirm_mc_samples

    @property
    def config_id(self) -> str:
        parts = [
            f"k{self.top_k}",
            f"mc{self.mc_samples}",
            f"d{self.min_delta:g}",
            f"se{self.se_multiplier:g}",
        ]
        if self.confirm_mc_samples == 0:
            parts.append("cmc0")
        elif self.confirm_mc_samples > 0:
            parts.append(f"cmc{self.confirm_mc_samples}")
        if self.min_gate_probability is not None:
            parts.append(f"g{self.min_gate_probability:g}")
        if self.allowed_seats:
            parts.append("seat" + "-".join(self.allowed_seats))
        if self.candidate_ev_rank_max is not None:
            parts.append(f"rank{self.candidate_ev_rank_max:g}")
        if self.topk_score != "delta":
            parts.append(f"by{self.topk_score}")
        return "_".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games-per-seed", type=int, default=100)
    parser.add_argument("--seeds", default="2026062101,2026062102,2026062103")
    parser.add_argument("--seed-stride", type=int, default=1_000_000)
    parser.add_argument(
        "--configs",
        default="k3/mc64/d0.25/se0/seat=first,k5/mc64/d0.25/se0/seat=first",
        help=(
            "Comma-separated configs. Example: "
            "k3/mc64/d0.25/se1.5/seat=first/rank3/g0.9/bydelta"
        ),
    )
    parser.add_argument("--opening-model", type=Path, default=Path("models/opening_stage7_torch_wide.pt"))
    parser.add_argument("--turn1-model", type=Path, default=Path("models/turn1_stage6_torch_wide.pt"))
    parser.add_argument("--turn2-baseline-model", type=Path, default=Path("models/turn2_stage8.pkl"))
    parser.add_argument("--turn3-model", type=Path, default=Path("models/turn3_stage6.pkl"))
    parser.add_argument("--hu-turn3-stage7-model", type=Path, default=DEFAULT_STAGE7_MODEL)
    parser.add_argument("--hu-turn3-reference-model", type=Path, default=DEFAULT_STAGE3_REFERENCE)
    parser.add_argument("--hu-turn2-stage8b-model", type=Path, default=DEFAULT_T2_STAGE8_MODEL)
    parser.add_argument("--calibration-values", type=Path, default=DEFAULT_CALIBRATION_VALUES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--prediction-threads", type=int, default=1)
    parser.add_argument("--opening-lookahead-samples", type=int, default=64)
    parser.add_argument("--batched-continuation-batch-size", type=int, default=8192)
    parser.add_argument("--stage3-feature-encoder-mode", default="rust_direct")
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
        confirm_mc_samples = -1
        allowed_seats: tuple[str, ...] = ()
        candidate_ev_rank_max: int | None = None
        min_gate_probability: float | None = None
        topk_score = "delta"
        for token in item.split("/"):
            key, sep, raw = token.partition("=")
            key = key.strip().lower()
            raw_value = raw.strip().lower() if sep else key
            if key.startswith("k") and not sep:
                top_k = int(float(key[1:]))
            elif key == "k" and sep:
                top_k = int(float(raw_value))
            elif key.startswith("cmc") and not sep:
                confirm_mc_samples = int(float(key[3:]))
            elif key in {"cmc", "confirm_mc", "confirm_mc_samples"} and sep:
                confirm_mc_samples = int(float(raw_value))
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
            elif key.startswith("g") and not sep:
                min_gate_probability = float(key[1:])
            elif key in {"g", "gate", "min_gate_probability"} and sep:
                min_gate_probability = float(raw_value)
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
                allowed_seats=allowed_seats,
                candidate_ev_rank_max=candidate_ev_rank_max,
                min_gate_probability=min_gate_probability,
                topk_score=topk_score,
                confirm_mc_samples=confirm_mc_samples,
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
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.hu_turn2_stage8b_model = hu_turn2_stage8b_model
        self.topk_rerank_config = topk_rerank_config
        self.topk_decision_log = topk_decision_log
        self.topk_context = dict(topk_context or {})
        self.batched_continuation_batch_size = batched_continuation_batch_size
        self._batched_config = HuTurn3Stage7BatchConfig(
            hu_turn3_min_margin=DEFAULT_HU_TURN3_STAGE7_MIN_MARGIN,
            hu_turn3_reference_min_margin=DEFAULT_HU_TURN3_STAGE7_REFERENCE_MIN_MARGIN,
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
        baseline_index = int(np.argmax(baseline_values))
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
        confirm_delta: float | None = None
        confirm_delta_se: float | None = None
        confirm_best_ev: float | None = None
        confirm_baseline_ev: float | None = None
        rerank_latency_ms = 0.0
        confirm_latency_ms = 0.0
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
                stage8b_top1_index = int(np.argmax(predictions[:, 1]))
                ev_order = np.argsort(-predictions[:, 0], kind="mergesort")
                ev_ranks = np.empty(len(actions), dtype=np.int32)
                for rank, action_index in enumerate(ev_order, start=1):
                    ev_ranks[int(action_index)] = rank
                gate_values = np.asarray([sigmoid(float(value)) for value in predictions[:, 4]], dtype=np.float64)
                order = self._topk_order(predictions, gate_values)
                selected: list[int] = []
                for index in order:
                    action_index = int(index)
                    if action_index == baseline_index:
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
                        ),
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
                            rerank_best_ev = float(best_action.get("score", best_action.get("ev", 0.0)))
                            rerank_baseline_ev = float(baseline_action.get("score", baseline_action.get("ev", 0.0)))
                            rerank_delta = rerank_best_ev - rerank_baseline_ev
                            best_se = float(best_action.get("ev_standard_error", best_action.get("standard_error", 0.0)))
                            baseline_se = float(baseline_action.get("ev_standard_error", baseline_action.get("standard_error", 0.0)))
                            rerank_delta_se = math.sqrt(best_se * best_se + baseline_se * baseline_se)
                            evaluated_action_count = len(actions_by_original)
                            common_random_future_digest = str(sample_mc.get("common_random_future_digest", ""))
                            if rerank_best_index == baseline_index:
                                no_override_reason = "mc_best_is_baseline"
                            elif rerank_delta < config.min_delta:
                                no_override_reason = "below_rerank_delta"
                            elif config.se_multiplier > 0.0 and rerank_delta < config.se_multiplier * rerank_delta_se:
                                no_override_reason = "below_rerank_se"
                            elif config.resolved_confirm_mc_samples <= 0:
                                final_index = rerank_best_index
                            else:
                                # Two-stage rerank: confirm the selected
                                # candidate against the baseline on an
                                # independent random stream so selection
                                # noise cannot inflate the gate delta.
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
                                        stage="confirm",
                                    ),
                                    mc_samples=config.resolved_confirm_mc_samples,
                                )
                                confirm_latency_ms = (time.perf_counter() - confirm_started) * 1000.0
                                confirm_actions = {
                                    int(item.get("original_index", -1)): item
                                    for item in list((confirm_sample or {}).get("actions") or ())
                                }
                                confirm_candidate = confirm_actions.get(rerank_best_index)
                                confirm_baseline = confirm_actions.get(baseline_index)
                                if confirm_sample is None:
                                    no_override_reason = "confirm_failed"
                                elif confirm_candidate is None or confirm_baseline is None:
                                    no_override_reason = "confirm_missing_action"
                                else:
                                    confirm_best_ev = float(confirm_candidate.get("score", confirm_candidate.get("ev", 0.0)))
                                    confirm_baseline_ev = float(confirm_baseline.get("score", confirm_baseline.get("ev", 0.0)))
                                    confirm_delta = confirm_best_ev - confirm_baseline_ev
                                    confirm_best_se = float(
                                        confirm_candidate.get("ev_standard_error", confirm_candidate.get("standard_error", 0.0))
                                    )
                                    confirm_baseline_se = float(
                                        confirm_baseline.get("ev_standard_error", confirm_baseline.get("standard_error", 0.0))
                                    )
                                    confirm_delta_se = math.sqrt(
                                        confirm_best_se * confirm_best_se + confirm_baseline_se * confirm_baseline_se
                                    )
                                    if confirm_delta < config.min_delta:
                                        no_override_reason = "below_confirm_delta"
                                    elif config.se_multiplier > 0.0 and confirm_delta < config.se_multiplier * confirm_delta_se:
                                        no_override_reason = "below_confirm_se"
                                    else:
                                        final_index = rerank_best_index
                candidate_for_logging = rerank_best_index if rerank_best_index is not None else stage8b_top1_index
                if candidate_for_logging is not None and 0 <= candidate_for_logging < len(actions):
                    predicted_delta = float(predictions[candidate_for_logging, 1])
                    predicted_ev = float(predictions[candidate_for_logging, 0])
                    gate_probability = float(gate_values[candidate_for_logging])
                    candidate_ev_rank = int(ev_ranks[candidate_for_logging])

        override_fired = final_index != baseline_index
        if not no_override_reason and not override_fired:
            no_override_reason = "fallback_to_baseline"
        self._log_topk_decision(
            {
                **self.topk_context,
                "hand_id": hand_id,
                "game_id": game_id,
                "seed": decision_seed if decision_seed is not None else self.seed,
                "street": street or "T2",
                "turn": street or "T2",
                "seat": self.seat,
                "hero_board": board_to_json(board),
                "opponent_board": board_to_json(opponent_board),
                "cards_to_place": list(dealt),
                "dead_cards": list(dead_cards),
                "baseline_action": action_to_json(board, actions[baseline_index]),
                "stage8b_top1_action": action_to_json(board, actions[stage8b_top1_index]) if stage8b_top1_index is not None else None,
                "rerank_best_action": action_to_json(board, actions[rerank_best_index]) if rerank_best_index is not None and 0 <= rerank_best_index < len(actions) else None,
                "fallback_action": action_to_json(board, actions[baseline_index]),
                "final_action": action_to_json(board, actions[final_index]),
                "baseline_action_index": baseline_index,
                "stage8b_top1_index": stage8b_top1_index,
                "rerank_best_index": rerank_best_index,
                "final_action_index": final_index,
                "rerank_action_indices": rerank_indices,
                "override_fired": override_fired,
                "no_override_reason": no_override_reason or "",
                "top_k": config.top_k,
                "mc_samples": config.mc_samples,
                "min_rerank_delta": config.min_delta,
                "se_multiplier": config.se_multiplier,
                "topk_score": config.topk_score,
                "allowed_seats": list(config.allowed_seats),
                "candidate_ev_rank_max": config.candidate_ev_rank_max,
                "min_gate_probability": config.min_gate_probability,
                "predicted_delta": predicted_delta,
                "gate_probability": gate_probability,
                "model_score": predicted_ev,
                "candidate_ev_rank": candidate_ev_rank,
                "rerank_delta": rerank_delta,
                "rerank_delta_se": rerank_delta_se,
                "rerank_best_ev": rerank_best_ev,
                "rerank_baseline_ev": rerank_baseline_ev,
                "confirm_mc_samples": config.resolved_confirm_mc_samples,
                "confirm_delta": confirm_delta,
                "confirm_delta_se": confirm_delta_se,
                "confirm_best_ev": confirm_best_ev,
                "confirm_baseline_ev": confirm_baseline_ev,
                "evaluated_action_count": evaluated_action_count,
                "common_random_future_digest": common_random_future_digest,
                "runtime_latency_ms": (time.perf_counter() - started_at) * 1000.0,
                "mc_rerank_latency_ms": rerank_latency_ms,
                "mc_confirm_latency_ms": confirm_latency_ms,
            }
        )
        return actions[final_index]

    def _topk_order(self, predictions: np.ndarray, gate_values: np.ndarray) -> np.ndarray:
        config = self.topk_rerank_config
        if config.topk_score == "ev":
            scores = predictions[:, 0]
        elif config.topk_score == "gate":
            scores = gate_values
        elif config.topk_score == "gate_delta":
            scores = predictions[:, 1] * gate_values
        else:
            scores = predictions[:, 1]
        return np.argsort(-scores, kind="mergesort")

    def _rerank_sample(
        self,
        *,
        board: Board,
        dealt: tuple[str, ...],
        opponent_board: Board,
        dead_cards: tuple[str, ...],
        action_indices: list[int],
        seed: int,
        mc_samples: int | None = None,
    ) -> dict[str, Any] | None:
        dead_set = set(dead_cards)
        excluded = set(board.all_cards()) | set(opponent_board.all_cards()) | set(dealt)
        rollout_dead = tuple(card for card in dead_cards if card not in excluded)
        hero_policy = self._build_rollout_policy(seed=seed * 4 + 1, seat=self.seat)
        opponent_seat = "second" if self.seat == "first" else "first"
        opponent_policy = self._build_rollout_policy(seed=seed * 4 + 2, seat=opponent_seat)
        return evaluate_hu_turn2_actions(
            board=board,
            dealt_cards=dealt,
            opponent_board=opponent_board,
            dead_cards=rollout_dead,
            hero_seat=self.seat,
            continuation_policy=hero_policy,
            opponent_policy=opponent_policy,
            baseline_turn2_model=self.turn2_model,
            future_samples=mc_samples if mc_samples is not None else self.topk_rerank_config.mc_samples,
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
        )

    def _build_rollout_policy(self, *, seed: int, seat: str) -> RegularAiPolicy:
        return RegularAiPolicy(
            opening_model=self.opening_model,
            turn1_model=self.turn1_model,
            turn2_model=self.turn2_model,
            turn3_model=self.turn3_model,
            hu_turn3_model=self.hu_turn3_model,
            hu_turn3_reference_model=self.hu_turn3_reference_model,
            hu_turn3_min_margin=self.hu_turn3_min_margin,
            hu_turn3_reference_min_margin=self.hu_turn3_reference_min_margin,
            hu_turn3_stage7_enabled=self.hu_turn3_stage7_enabled,
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
        stage: str = "select",
    ) -> int:
        payload = {
            "decision_seed": decision_seed if decision_seed is not None else self.seed,
            "hand_id": hand_id,
            "game_id": game_id,
            "seat": self.seat,
            "board": board_to_json(board),
            "opponent": board_to_json(opponent_board),
            "dealt": list(dealt),
            "config_id": self.topk_rerank_config.config_id,
            "stage": stage,
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
) -> HuTurn2Stage8bTopKMcRerankPolicy:
    return HuTurn2Stage8bTopKMcRerankPolicy(
        opening_model=parts.opening,
        turn1_model=parts.turn1,
        turn2_model=parts.turn2_baseline,
        turn3_model=parts.turn3,
        hu_turn3_model=parts.hu_turn3_stage7,
        hu_turn3_reference_model=parts.hu_turn3_reference,
        hu_turn3_min_margin=5.0,
        hu_turn3_reference_min_margin=10.0,
        hu_turn2_stage8b_model=stage8b_model,
        topk_rerank_config=config,
        topk_decision_log=decisions,
        topk_context=context,
        seed=seed,
        seat=seat,
        opening_lookahead_samples=opening_lookahead_samples,
        batched_continuation_batch_size=batched_continuation_batch_size,
        stage3_feature_encoder_mode=stage3_feature_encoder_mode,
    )


def evaluate_config_seed(
    *,
    config: TopKMcRerankConfig,
    seed: int,
    seed_stride: int,
    games: int,
    parts: ModelParts,
    stage8b_model: object,
    opening_lookahead_samples: int,
    batched_continuation_batch_size: int,
    stage3_feature_encoder_mode: str,
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
            profile_p1="stage8b_topk_mc",
            policy_p0=make_baseline_policy(
                parts,
                seed=hand_seed * 4 + 2,
                seat="first",
                opening_lookahead_samples=opening_lookahead_samples,
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
                        "event": "topk_mc_rerank_progress",
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
    override_count = sum(1 for row in decisions if row.get("override_fired"))
    fired = [row for row in decisions if row.get("override_fired")]
    rerank_deltas = [float(row.get("rerank_delta", 0.0) or 0.0) for row in fired]
    rerank_losses = [max(0.0, -delta) for delta in rerank_deltas]
    confirm_deltas = [
        float(row["confirm_delta"]) for row in fired if row.get("confirm_delta") is not None
    ]
    summary = {
        "config_id": config.config_id,
        "top_k": config.top_k,
        "mc_samples": config.mc_samples,
        "confirm_mc_samples": config.resolved_confirm_mc_samples,
        "min_rerank_delta": config.min_delta,
        "se_multiplier": config.se_multiplier,
        "allowed_seats": "+".join(config.allowed_seats),
        "candidate_ev_rank_max": "" if config.candidate_ev_rank_max is None else config.candidate_ev_rank_max,
        "min_gate_probability": "" if config.min_gate_probability is None else config.min_gate_probability,
        "topk_score": config.topk_score,
        "seed": seed,
        "paired_seeds": games,
        "hands": games * 2,
        "ev_per_hand": float(np.mean(paired_scores)) if paired_scores else 0.0,
        **summarize_values(paired_scores, "paired_score_"),
        "paired_seed_wins": wins,
        "paired_seed_losses": losses,
        "paired_seed_ties": ties,
        "decision_count": len(decisions),
        "override_count": override_count,
        "override_rate": override_count / max(len(decisions), 1),
        "avg_rerank_delta_on_override": float(np.mean(rerank_deltas)) if rerank_deltas else 0.0,
        "median_rerank_delta_on_override": float(np.median(rerank_deltas)) if rerank_deltas else 0.0,
        "avg_confirm_delta_on_override": float(np.mean(confirm_deltas)) if confirm_deltas else 0.0,
        "median_confirm_delta_on_override": float(np.median(confirm_deltas)) if confirm_deltas else 0.0,
        "confirmed_override_count": len(confirm_deltas),
        "p95_rerank_loss": percentile(rerank_losses, 95),
        "max_rerank_loss": max(rerank_losses) if rerank_losses else 0.0,
        "no_override_reason_counts": json.dumps(dict(no_override), sort_keys=True),
        "elapsed_seconds": time.time() - started_at,
    }
    return summary, decisions, position_scores


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), q / 100.0))


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
        fired_deltas = []
        confirm_deltas = []
        for item in group:
            if float(item.get("avg_rerank_delta_on_override", 0.0)) and int(item.get("override_count", 0)):
                fired_deltas.extend([float(item["avg_rerank_delta_on_override"])] * int(item["override_count"]))
            confirmed = int(item.get("confirmed_override_count", 0) or 0)
            if confirmed and item.get("avg_confirm_delta_on_override") not in (None, ""):
                confirm_deltas.extend([float(item["avg_confirm_delta_on_override"])] * confirmed)
        rows.append(
            {
                "config_id": config_id,
                "top_k": group[0]["top_k"],
                "mc_samples": group[0]["mc_samples"],
                "min_rerank_delta": group[0]["min_rerank_delta"],
                "se_multiplier": group[0]["se_multiplier"],
                "allowed_seats": group[0]["allowed_seats"],
                "candidate_ev_rank_max": group[0]["candidate_ev_rank_max"],
                "min_gate_probability": group[0]["min_gate_probability"],
                "topk_score": group[0]["topk_score"],
                "paired_seeds": total_games,
                "hands": sum(int(row["hands"]) for row in group),
                "aggregate_ev_per_hand": mean,
                "std_error_seed_means": stderr,
                "ci95_low_seed_means": mean - 1.96 * stderr,
                "ci95_high_seed_means": mean + 1.96 * stderr,
                "seed_count": len(group),
                "decision_count": decisions,
                "override_count": overrides,
                "runtime_override_rate": overrides / max(decisions, 1),
                "paired_seed_wins": sum(int(row["paired_seed_wins"]) for row in group),
                "paired_seed_losses": sum(int(row["paired_seed_losses"]) for row in group),
                "paired_seed_ties": sum(int(row["paired_seed_ties"]) for row in group),
                "avg_gain_on_override": float(np.mean(fired_deltas)) if fired_deltas else 0.0,
                "avg_confirm_gain_on_override": float(np.mean(confirm_deltas)) if confirm_deltas else 0.0,
                "confirmed_override_count": len(confirm_deltas),
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
                    "evaluated_action_count_mean": float(np.mean(action_counts)) if action_counts else 0.0,
                    "ms_per_mc_action": sum(rerank) / max(denom, 1.0),
                }
            )
    return output


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
                "rerank_delta": row.get("rerank_delta"),
                "rerank_delta_se": row.get("rerank_delta_se"),
                "confirm_delta": row.get("confirm_delta"),
                "confirm_delta_se": row.get("confirm_delta_se"),
                "failure_label": classify_failure(row),
            }
        )
    return output


def classify_failure(row: dict[str, Any]) -> str:
    if float(row.get("rerank_delta", 0.0) or 0.0) < float(row.get("min_rerank_delta", 0.0) or 0.0):
        return "mc_rerank_threshold_leak"
    if float(row.get("candidate_seat_score", 0.0) or 0.0) < -20.0:
        return "tail_loss_audit_required"
    return "other"


def group_by(rows: Iterable[dict[str, Any]], field: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field, "unknown"))].append(row)
    return grouped


def write_summary(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# HU T2 Stage8b TopK + MC Rerank Validation",
        "",
        "This is validation-only. It does not authorize 50k teacher generation, T1 training, production training, P2 fixed status, or production runtime changes.",
        "",
        "## Inputs",
        "",
        "- T3 continuation: `Stage7_candidate_A m5_r10` fixed",
        "- Stage8b role: candidate generator only, not direct override",
        f"- model: `{args.hu_turn2_stage8b_model}`",
        f"- seeds: `{args.seeds}`",
        f"- games_per_seed: `{args.games_per_seed}`",
        f"- seed_stride: `{args.seed_stride}`",
        "",
        "## Results",
        "",
        "| config | paired | EV/hand | CI low | CI high | overrides | override rate | avg MC gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {config_id} | {paired} | {ev:.4f} | {low:.4f} | {high:.4f} | {overrides} | {rate:.4f} | {gain:.4f} |".format(
                config_id=row.get("config_id", ""),
                paired=row.get("paired_seeds", ""),
                ev=float(row.get("aggregate_ev_per_hand", 0.0)),
                low=float(row.get("ci95_low_seed_means", 0.0)),
                high=float(row.get("ci95_high_seed_means", 0.0)),
                overrides=row.get("override_count", ""),
                rate=float(row.get("runtime_override_rate", 0.0)),
                gain=float(row.get("avg_gain_on_override", 0.0)),
            )
        )
    best = rows[0] if rows else {}
    go = bool(best and float(best.get("aggregate_ev_per_hand", 0.0)) > 0.0 and float(best.get("ci95_low_seed_means", -999.0)) > -0.05)
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- TopK+MC rerank validation: `{'Conditional-Go' if go else 'No-Go'}`",
            "- production / P2 fixed: `No-Go`",
            "- 50k teacher: `No-Go`",
            "- T1 training: `No-Go`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.games_per_seed <= 0:
        raise SystemExit("--games-per-seed must be positive")
    if args.prediction_threads > 0:
        os.environ.setdefault("OMP_NUM_THREADS", str(args.prediction_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.prediction_threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.prediction_threads))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configs = parse_topk_configs(args.configs)
    seeds = parse_seeds(args.seeds)
    parts = load_parts(
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
    stage8b_model = parts.hu_turn2_stage8
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
                    parts=parts,
                    stage8b_model=stage8b_model,
                    opening_lookahead_samples=args.opening_lookahead_samples,
                    batched_continuation_batch_size=args.batched_continuation_batch_size,
                    stage3_feature_encoder_mode=args.stage3_feature_encoder_mode,
                    progress_every=args.progress_every,
                )
                seed_rows.append(summary)
                decision_rows.extend(decisions)
                position_scores.extend(scores)

    result_rows = aggregate_topk_seed_rows(seed_rows)
    write_csv(args.output_dir / "topk_rerank_results.csv", result_rows)
    write_csv(args.output_dir / "seed_breakdown.csv", seed_rows)
    write_csv(args.output_dir / "position_breakdown.csv", runtime_position_breakdown(position_scores, decision_rows))
    write_csv(args.output_dir / "latency_breakdown.csv", latency_breakdown(decision_rows))
    write_csv(args.output_dir / "no_override_reason_counts.csv", runtime_decision_buckets(decision_rows))
    write_jsonl(args.output_dir / "failure_top30.jsonl", failure_rows(decision_rows))
    if args.write_decision_log:
        write_jsonl(args.output_dir / "runtime_decisions.jsonl", decision_rows)
    write_summary(args.output_dir / "topk_rerank_summary.md", result_rows, args)
    best = result_rows[0] if result_rows else {}
    go = bool(best and float(best.get("aggregate_ev_per_hand", 0.0)) > 0.0 and float(best.get("ci95_low_seed_means", -999.0)) > -0.05)
    (args.output_dir / "go_nogo.md").write_text(
        "\n".join(
            [
                "# HU T2 Stage8b TopK + MC Rerank Go / No-Go",
                "",
                f"- execution: `Pass`",
                f"- decision: `{'Conditional-Go' if go else 'No-Go'}`",
                "- production / P2 fixed: `No-Go`",
                "- 50k teacher: `No-Go`",
                "- T1 training: `No-Go`",
                f"- elapsed_seconds: `{time.time() - started_at:.2f}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "configs": len(configs), "seeds": len(seeds), "elapsed_seconds": time.time() - started_at}, indent=2))


if __name__ == "__main__":
    main()
