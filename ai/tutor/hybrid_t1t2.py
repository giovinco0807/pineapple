"""Hybrid T1/T2 tutor evaluation.

This module uses the action-value reranker as a fast candidate generator, then
optionally refines a small T2 subset by expanding sampled T3 deals through the
Rust T3 exact solver.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.action_space import Action, encode_action, get_turn_actions
from ai.engine.encoding import ALL_CARDS, Board, Observation
from ai.engine.game_engine import RANK_VALUES, evaluate_board_with_joker_constraint, evaluate_hand, hand_category
from ai.engine.turn_order import normalize_position, validate_decision_board_counts
from ai.mcts.rollout_evaluator import RolloutEvaluator
from ai.models.action_value_reranker import (
    ActionValueReranker,
    BlendedActionValueReranker,
    CascadeSwitchActionValueReranker,
    ConditionalBlendedActionValueReranker,
    ConditionalSwitchActionValueReranker,
)

FL_TYPE_KEYS = ("qq", "kk", "aa", "trips")
ROWS = ("top", "middle", "bottom")
BLOCKER_RANKS = ("A", "K", "Q", "J", "T", "2")


def enrich_override_gate_features(features: dict[str, float]) -> dict[str, float]:
    """Add deterministic derived features for T1 override gate models."""
    out = {str(key): float(value) for key, value in features.items()}
    refined_delta = float(out.get("refined_delta", 0.0))
    model_score_delta = float(out.get("model_score_delta", 0.0))
    model_rank_gap = float(out.get("model_rank_gap", 0.0))
    predicted_bust_delta = float(out.get("predicted_bust_delta", 0.0))
    predicted_fl_delta = float(out.get("predicted_fl_delta", 0.0))
    predicted_qq_delta = float(out.get("predicted_qq_delta", 0.0))
    predicted_kk_delta = float(out.get("predicted_kk_delta", 0.0))
    predicted_aa_delta = float(out.get("predicted_aa_delta", 0.0))
    predicted_trips_delta = float(out.get("predicted_trips_delta", 0.0))
    best_refined_score = float(out.get("best_refined_score", 0.0))
    model_refined_score = float(out.get("model_refined_score", 0.0))
    best_model_score = float(out.get("best_model_score", 0.0))
    model_top1_score = float(out.get("model_top1_score", 0.0))
    rank_denominator = max(abs(model_rank_gap), 1.0)

    out.update(
        {
            "abs_refined_delta": abs(refined_delta),
            "abs_model_score_delta": abs(model_score_delta),
            "refined_delta_per_rank_gap": refined_delta / rank_denominator,
            "refined_delta_minus_abs_model_score_delta": refined_delta - abs(model_score_delta),
            "refined_delta_plus_model_score_delta": refined_delta + model_score_delta,
            "refined_delta_x_rank_gap": refined_delta * model_rank_gap,
            "refined_delta_x_model_score_delta": refined_delta * model_score_delta,
            "refined_delta_x_predicted_bust_delta": refined_delta * predicted_bust_delta,
            "refined_delta_x_predicted_fl_delta": refined_delta * predicted_fl_delta,
            "rank_gap_is_one": 1.0 if abs(model_rank_gap) <= 1.0 else 0.0,
            "rank_gap_is_two_plus": 1.0 if abs(model_rank_gap) >= 2.0 else 0.0,
            "best_refined_minus_best_model_score": best_refined_score - best_model_score,
            "model_refined_minus_model_top1_score": model_refined_score - model_top1_score,
            "fl_type_delta_sum": predicted_qq_delta
            + predicted_kk_delta
            + predicted_aa_delta
            + predicted_trips_delta,
            "premium_fl_delta_sum": predicted_kk_delta + predicted_aa_delta + predicted_trips_delta,
            "risk_adjusted_refined_delta": refined_delta - 10.0 * max(predicted_bust_delta, 0.0),
            "fl_adjusted_refined_delta": refined_delta + 5.0 * predicted_fl_delta,
        }
    )
    return out


class DummyPolicy(torch.nn.Module):
    def forward(self, state, mask=None):
        width = mask.shape[-1] if mask is not None else 1
        return torch.ones((state.shape[0], width), dtype=torch.float32, device=state.device)


@dataclass(frozen=True)
class HybridConfig:
    shortlist_k: int = 15
    insurance_k: int = 5
    sync_exact_k: int = 3
    t2_sync_exact_k: int = 0
    t1_adaptive_sync_exact_k: int = 0
    t1_adaptive_model_rank_k: int = 0
    t1_sync_model_insurance_k: int = 0
    t1_sync_tactical_insurance_k: int = 0
    t2_sync_model_insurance_k: int = 0
    t2_sync_model_insurance_top1_kk_min: float = 0.0
    t2_tactical_insurance_k: int = 0
    t1_aux_shortlist_k: int = 0
    t2_aux_shortlist_k: int = 0
    max_pool: int = 20
    time_budget_ms: int = 5000
    mode: str = "fast"
    high_bust_threshold: float = 0.999
    t2_extra_margin: float = 1.0
    t2_initial_samples_per_candidate: int = 3
    t2_extra_samples_per_round: int = 2
    t2_max_samples_per_candidate: int = 24
    t2_close_candidate_limit: int = 4
    t2_baseline_min_samples: int = 3
    t2_model_rank_min_samples_k: int = 0
    t2_model_rank_min_samples: int = 0
    t2_model_rank_min_samples_min_remaining_ms: int = 0
    t2_adaptive_shortlist_k: int = 0
    t2_adaptive_sync_exact_k: int = 0
    t2_adaptive_max_model_score: float | None = None
    t2_post_refine_sync_exact_k: int = 0
    t2_post_refine_min_remaining_ms: int = 0
    t2_post_refine_policy: str = "always"
    t2_post_refine_extra_model_score_min: float = -1.0
    t1_refinement: str = "none"
    t1_mc_sims: int = 32
    t1_recursive_beam: int = 5
    t1_recursive_child_sims: int = 2
    t1_refinement_first_batch_k: int = 0
    t1_refinement_tail_min_remaining_ms: int = 0
    t1_refinement_timeout_headroom_ms: int = 0
    t1_extra_refine_top_k: int = 0
    t1_extra_refine_margin: float = 0.0
    t1_extra_refine_sims: int = 0
    t1_sync_selection_policy: str = "rank"
    t1_sync_selector: dict[str, Any] | None = None
    t2_sync_selection_policy: str = "rank"
    t2_sync_selector: dict[str, Any] | None = None
    t2_selection_policy: str = "refined_score"
    t2_selection_model_weight: float = 0.0
    t2_selection_structured_top_model_weight: float = -1.0
    t2_final_selector: dict[str, Any] | None = None
    t2_model_top1_rescue_fl_min: float = -1.0
    t2_model_top1_rescue_bust_max: float = -1.0
    t2_model_top1_rescue_selected_model_rank_min: int = 0
    t2_model_top1_rescue_refined_delta_max: float = -1.0
    t2_model_top1_rescue2_fl_min: float = -1.0
    t2_model_top1_rescue2_fl_max: float = -1.0
    t2_model_top1_rescue2_bust_max: float = -1.0
    t2_model_top1_rescue2_selected_model_rank_min: int = 0
    t2_model_top1_rescue2_refined_delta_max: float = -1.0
    t2_model_top1_bust_rescue_selected_model_rank_min: int = 0
    t2_model_top1_bust_rescue_refined_delta_max: float = -1.0
    t2_model_top1_bust_rescue_model_delta_min: float = -1.0
    t2_model_top1_bust_rescue_bust_delta_min: float = -1.0
    t2_model_top1_bust_rescue_top_bust_max: float = -1.0
    t2_model_top1_bust_rescue_top_fl_min: float = -1.0
    t2_model_top1_bust_rescue_current_fl_min: float = -1.0
    t2_model_top1_bust_rescue_fl_delta_min: float = -1.0
    t2_middle_fill_bottom_shift_rescue_selected_model_rank_max: int = 0
    t2_middle_fill_bottom_shift_rescue_challenger_model_rank_max: int = 0
    t2_middle_fill_bottom_shift_rescue_model_gap_max: float = -1.0
    t2_middle_fill_bottom_shift_rescue_bust_delta_min: float = -1.0
    t2_middle_fill_bottom_shift_rescue_fl_delta_min: float = -1.0
    t2_middle_fill_bottom_shift_rescue_challenger_bust_max: float = -1.0
    t2_middle_fill_bottom_shift_rescue_challenger_fl_min: float = -1.0
    t2_middle_fill_bottom_shift_rescue_selected_bust_min: float = -1.0
    t2_model_rank_rescue_k: int = 0
    t2_model_rank_rescue_selected_model_rank_min: int = 0
    t2_model_rank_rescue_refined_delta_max: float = -1.0
    t2_model_rank_rescue_model_delta_min: float = -1.0
    t2_model_rank_rescue_min_refined_score: float = -1.0
    t1_selection_policy: str = "refined_score"
    t1_selection_model_weight: float = 0.0
    t1_final_selector: dict[str, Any] | None = None
    t1_arbitration_selector: dict[str, Any] | None = None
    t1_arbitration_challenger_selector: dict[str, Any] | None = None
    t1_arbitration_policy: str = "none"
    t1_arbitration_dual_gate_policy: str = "none"
    t1_final_challenger_selector: dict[str, Any] | None = None
    t1_final_challenger_gate: dict[str, Any] | None = None
    t1_final_challenger_gate_policy: str = "none"
    t1_refined_challenger_bust_max: float = -1.0
    t1_refined_challenger_refined_delta_max: float = -1.0
    t1_blend_challenger_weight: float = -1.0
    t1_blend_challenger_bust_min: float = -1.0
    t1_blend_challenger_premium_fl_delta_min: float = -1.0
    t1_low_risk_blend_challenger_weight: float = -1.0
    t1_low_risk_blend_challenger_fl_max: float = -1.0
    t1_low_risk_blend_challenger_bust_max: float = -1.0
    t1_model_rescue_qq_delta_min: float = -1.0
    t1_model_rescue_premium_delta_min: float = -1.0
    t1_model_rescue_candidate_fl_delta_min: float = -1.0
    t1_model_rescue_model_score_min: float = -1.0
    t1_model_rescue_model_bust_max: float = -1.0
    t1_model_rescue_refined_delta_max: float = -1.0
    t1_final_bottom_sparse_rescue_margin: float = -1.0
    t1_no_refine_fallback_policy: str = "model"
    t1_override_margin: float = 0.0
    t1_override_gate: dict[str, Any] | None = None
    t1_override_gate_threshold: float = 0.5
    t2_refinement: str = "mc_board"
    t2_exact_backend: str = "full_exact"
    t3_pool_config: str = "ai/config/t3_ev_loss_fresh_pool_20260607.json"
    t3_pool_k: int = -1
    t2_mc_sims: int = 300
    enable_sync_refinement: bool = True


def load_t1_override_gate(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    gate_path = Path(path)
    if not gate_path.exists():
        raise FileNotFoundError(f"T1 override gate not found: {gate_path}")
    payload = json.loads(gate_path.read_text(encoding="utf-8-sig"))
    gate = payload.get("gate") if isinstance(payload, dict) else None
    if isinstance(gate, dict):
        return gate
    return payload


def load_t1_runtime_gate(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    gate_path = Path(path)
    if not gate_path.exists():
        raise FileNotFoundError(f"T1 runtime gate not found: {gate_path}")
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    gate = payload.get("gate") if isinstance(payload, dict) else None
    if isinstance(gate, dict):
        return gate
    return payload


def load_t1_sync_selector(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    selector_path = Path(path)
    if not selector_path.exists():
        raise FileNotFoundError(f"T1 sync selector not found: {selector_path}")
    return json.loads(selector_path.read_text(encoding="utf-8"))


def load_t2_sync_selector(path: str | Path | None) -> dict[str, Any] | None:
    return load_t1_sync_selector(path)


def load_t1_final_selector(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    selector_path = Path(path)
    if not selector_path.exists():
        raise FileNotFoundError(f"T1 final selector not found: {selector_path}")
    return json.loads(selector_path.read_text(encoding="utf-8"))


def _row(board: dict[str, Any] | None, *names: str) -> list[str]:
    out: list[str] = []
    if not board:
        return out
    for name in names:
        out.extend(str(card) for card in (board.get(name, []) or []) if card)
    return out


def board_from_payload(board: dict[str, Any] | None) -> Board:
    board = board or {}
    return Board(
        top=_row(board, "top"),
        middle=_row(board, "middle", "mid"),
        bottom=_row(board, "bottom", "bot"),
    )


def observation_from_payload(payload: dict[str, Any]) -> Observation:
    board_self = board_from_payload(payload.get("board"))
    board_opponent = board_from_payload(payload.get("opponent_board"))
    raw_position = payload.get("position")
    if raw_position in (None, ""):
        raw_position = payload.get("player_position")
    has_position = raw_position not in (None, "")
    has_is_btn = "is_btn" in payload and payload.get("is_btn") is not None
    if not has_position and not has_is_btn:
        raise ValueError("HU decision requires explicit position or is_btn")
    position = normalize_position(
        raw_position if has_position else None,
        is_btn=payload["is_btn"] if has_is_btn else None,
    )
    turn = int(payload.get("turn", 0))
    validate_decision_board_counts(
        turn,
        position,
        len(board_self.all_cards()),
        len(board_opponent.all_cards()),
    )
    return Observation(
        board_self=board_self,
        board_opponent=board_opponent,
        dealt_cards=[str(card) for card in (payload.get("dealt") or []) if card],
        known_discards_self=[str(card) for card in (payload.get("known_discards") or []) if card],
        turn=turn,
        is_btn=position == "btn",
        is_fl=bool(payload.get("is_fl", False)),
        opp_is_fl=bool(payload.get("opp_is_fl", False)),
        chips_self=int(payload.get("chips_self", 200) or 200),
        chips_opponent=int(payload.get("chips_opponent", 200) or 200),
    )


def action_to_dict(action: Action) -> dict[str, Any]:
    return {
        "placements": [[card, row] for card, row in action.placements],
        "discard": action.discard,
    }


def _override_gate_features(model_top1: dict[str, Any], best_candidate: dict[str, Any]) -> dict[str, float]:
    model_refined = model_top1.get("refined_score")
    best_refined = best_candidate.get("refined_score")
    features = {
        "bias": 1.0,
        "refined_delta": 0.0
        if model_refined is None or best_refined is None
        else float(best_refined) - float(model_refined),
        "model_score_delta": float(best_candidate.get("model_score", 0.0))
        - float(model_top1.get("model_score", 0.0)),
        "model_raw_score_delta": float(best_candidate.get("model_raw_score", 0.0))
        - float(model_top1.get("model_raw_score", 0.0)),
        "model_rank_gap": float(best_candidate.get("model_rank", 0))
        - float(model_top1.get("model_rank", 0)),
        "predicted_bust_delta": float(best_candidate.get("predicted_bust", 0.0))
        - float(model_top1.get("predicted_bust", 0.0)),
        "predicted_fl_delta": float(best_candidate.get("predicted_fl", 0.0))
        - float(model_top1.get("predicted_fl", 0.0)),
        "best_refined_score": 0.0 if best_refined is None else float(best_refined),
        "model_refined_score": 0.0 if model_refined is None else float(model_refined),
        "best_model_score": float(best_candidate.get("model_score", 0.0)),
        "model_top1_score": float(model_top1.get("model_score", 0.0)),
        "best_predicted_bust": float(best_candidate.get("predicted_bust", 0.0)),
        "model_predicted_bust": float(model_top1.get("predicted_bust", 0.0)),
        "best_predicted_fl": float(best_candidate.get("predicted_fl", 0.0)),
        "model_predicted_fl": float(model_top1.get("predicted_fl", 0.0)),
    }
    best_types = best_candidate.get("predicted_fl_types") or {}
    model_types = model_top1.get("predicted_fl_types") or {}
    for key in FL_TYPE_KEYS:
        features[f"predicted_{key}_delta"] = float(best_types.get(key, 0.0)) - float(model_types.get(key, 0.0))
    return enrich_override_gate_features(features)


def _linear_gate_probability(gate: dict[str, Any], features: dict[str, float]) -> float:
    import math

    feature_names = list(gate.get("features") or [])
    weights = list(gate.get("weights") or [])
    means = dict(gate.get("means") or {})
    scales = dict(gate.get("scales") or {})
    z = float(gate.get("intercept", 0.0))
    for name, weight in zip(feature_names, weights):
        raw = float(features.get(name, 0.0))
        if name != "bias":
            raw = (raw - float(means.get(name, 0.0))) / max(float(scales.get(name, 1.0)), 1e-9)
        z += float(weight) * raw
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _rule_condition_accepts(condition: dict[str, Any], features: dict[str, float]) -> bool:
    name = str(condition.get("feature") or "")
    direction = str(condition.get("direction") or "ge")
    threshold = float(condition.get("threshold", 0.0) or 0.0)
    value = float(features.get(name, 0.0))
    if direction == "ge":
        return value >= threshold
    if direction == "le":
        return value <= threshold
    raise ValueError(f"unsupported gate condition direction: {direction}")


def _rule_gate_accepts(gate: dict[str, Any], features: dict[str, float]) -> bool:
    conditions = list(gate.get("conditions") or [])
    for condition in conditions:
        if not _rule_condition_accepts(condition, features):
            return False
    for group in list(gate.get("reject_condition_groups") or []):
        group_conditions = list((group or {}).get("conditions") or [])
        if group_conditions and all(_rule_condition_accepts(condition, features) for condition in group_conditions):
            return False
    for condition in list(gate.get("reject_conditions") or []):
        name = str(condition.get("feature") or "")
        if name and _rule_condition_accepts(condition, features):
            return False
    return True


def _gate_accepts_override(
    gate: dict[str, Any] | None,
    features: dict[str, float],
    *,
    threshold: float,
) -> tuple[bool, float | None]:
    if not gate:
        return True, None
    if str(gate.get("kind") or "") in {"threshold_rule", "rule"}:
        accepted = _rule_gate_accepts(gate, features)
        return accepted, 1.0 if accepted else 0.0
    probability = _linear_gate_probability(gate, features)
    return probability >= float(threshold), probability


def _margin_accepts_override(margin: float, refined_delta: float) -> bool:
    if float(margin) == 0.0:
        return True
    return float(refined_delta) >= float(margin)


def apply_action(board: Board, action: Action) -> Board:
    out = board.copy()
    for card, row in action.placements:
        getattr(out, row).append(card)
    return out


def load_action_value_model(path: str | Path, device: str | torch.device = "cpu") -> ActionValueReranker:
    model = ActionValueReranker.from_checkpoint(path, map_location=device)
    model.to(device)
    model.eval()
    return model


def load_blended_action_value_model(
    model_a_path: str | Path,
    model_b_path: str | Path,
    model_b_weight: float,
    device: str | torch.device = "cpu",
) -> BlendedActionValueReranker:
    model = BlendedActionValueReranker.from_checkpoints(
        model_a_path,
        model_b_path,
        model_b_weight=model_b_weight,
        map_location=device,
    )
    model.to(device)
    model.eval()
    return model


def load_weighted_action_value_ensemble(
    model_paths: list[str | Path],
    weights: list[float],
    device: str | torch.device = "cpu",
) -> BlendedActionValueReranker:
    model = BlendedActionValueReranker.from_weighted_checkpoints(
        model_paths,
        weights,
        map_location=device,
    )
    model.to(device)
    model.eval()
    return model


def load_conditional_action_value_ensemble(
    model_paths: list[str | Path],
    weights: list[float],
    specialist_path: str | Path,
    specialist_weight: float,
    gate: str,
    device: str | torch.device = "cpu",
) -> ConditionalBlendedActionValueReranker:
    model = ConditionalBlendedActionValueReranker.from_checkpoints(
        list(model_paths),
        list(weights),
        specialist_path,
        specialist_weight=float(specialist_weight),
        gate=gate,
        map_location=device,
    )
    model.to(device)
    model.eval()
    return model


def load_conditional_switch_action_value_ensemble(
    base_paths: list[str | Path],
    base_weights: list[float],
    challenger_paths: list[str | Path],
    challenger_weights: list[float],
    gate: str,
    device: str | torch.device = "cpu",
) -> ConditionalSwitchActionValueReranker:
    model = ConditionalSwitchActionValueReranker.from_checkpoints(
        list(base_paths),
        list(base_weights),
        list(challenger_paths),
        list(challenger_weights),
        gate=gate,
        map_location=device,
    )
    model.to(device)
    model.eval()
    return model


def load_cascade_switch_action_value_ensemble(
    base_paths: list[str | Path],
    base_weights: list[float],
    switch_specs: list[tuple[str, list[str | Path], list[float]]],
    device: str | torch.device = "cpu",
) -> CascadeSwitchActionValueReranker:
    model = CascadeSwitchActionValueReranker.from_checkpoints(
        list(base_paths),
        list(base_weights),
        switch_specs,
        map_location=device,
    )
    model.to(device)
    model.eval()
    return model


def make_action_value_evaluator(
    model: torch.nn.Module | None = None,
    *,
    model_path: str | Path | None = None,
    models_by_turn: dict[int, torch.nn.Module] | None = None,
    model_paths_by_turn: dict[int, str | Path] | None = None,
    model_blends_by_turn: dict[int, tuple[str | Path, str | Path, float]] | None = None,
    model_ensembles_by_turn: dict[int, tuple[list[str | Path], list[float]]] | None = None,
    model_conditional_ensembles_by_turn: dict[int, tuple[list[str | Path], list[float], str | Path, float, str]] | None = None,
    model_conditional_switch_ensembles_by_turn: dict[int, tuple[list[str | Path], list[float], list[str | Path], list[float], str]] | None = None,
    model_cascade_switch_ensembles_by_turn: dict[int, tuple[list[str | Path], list[float], list[tuple[str, list[str | Path], list[float]]]]] | None = None,
    device: str | None = None,
    suit_ensemble_turns: Iterable[int] | None = None,
    suit_ensemble_size: int = 8,
) -> RolloutEvaluator:
    actual_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        if model_path is None:
            raise ValueError("model or model_path is required")
        model = load_action_value_model(model_path, actual_device)
    model.to(actual_device)
    model.eval()
    turn_models: dict[int, torch.nn.Module] = {}
    for turn, turn_model in (models_by_turn or {}).items():
        turn_model.to(actual_device)
        turn_model.eval()
        turn_models[int(turn)] = turn_model
    for turn, path in (model_paths_by_turn or {}).items():
        turn_models[int(turn)] = load_action_value_model(path, actual_device)
    for turn, spec in (model_blends_by_turn or {}).items():
        model_a_path, model_b_path, model_b_weight = spec
        turn_models[int(turn)] = load_blended_action_value_model(
            model_a_path,
            model_b_path,
            model_b_weight,
            actual_device,
        )
    for turn, spec in (model_ensembles_by_turn or {}).items():
        model_paths, weights = spec
        turn_models[int(turn)] = load_weighted_action_value_ensemble(
            list(model_paths),
            list(weights),
            actual_device,
        )
    for turn, spec in (model_conditional_ensembles_by_turn or {}).items():
        model_paths, weights, specialist_path, specialist_weight, gate = spec
        turn_models[int(turn)] = load_conditional_action_value_ensemble(
            list(model_paths),
            list(weights),
            specialist_path,
            float(specialist_weight),
            gate,
            actual_device,
        )
    for turn, spec in (model_conditional_switch_ensembles_by_turn or {}).items():
        base_paths, base_weights, challenger_paths, challenger_weights, gate = spec
        turn_models[int(turn)] = load_conditional_switch_action_value_ensemble(
            list(base_paths),
            list(base_weights),
            list(challenger_paths),
            list(challenger_weights),
            gate,
            actual_device,
        )
    for turn, spec in (model_cascade_switch_ensembles_by_turn or {}).items():
        base_paths, base_weights, switch_specs = spec
        turn_models[int(turn)] = load_cascade_switch_action_value_ensemble(
            list(base_paths),
            list(base_weights),
            list(switch_specs),
            actual_device,
        )
    return RolloutEvaluator(
        policy_net=DummyPolicy(),
        action_value_net=model,
        action_value_nets_by_turn=turn_models,
        device=actual_device,
        full_width=True,
        n_rollouts=0,
        action_value_suit_ensemble_turns=suit_ensemble_turns,
        action_value_suit_ensemble_size=suit_ensemble_size,
    )


def _candidate_fl_score(candidate: dict[str, Any]) -> float:
    fl_types = candidate["predicted_fl_types"]
    return max(
        [float(candidate["predicted_fl"])]
        + [float(fl_types.get(key, 0.0)) for key in FL_TYPE_KEYS]
    )


def _card_rank(card: str) -> str:
    card = str(card)
    if card.startswith("X"):
        return "X"
    return card[:-1]


def _row_cards(board: dict[str, Any], row: str) -> list[str]:
    if row == "middle":
        return [str(card) for card in (board.get("middle", []) or board.get("mid", []) or [])]
    if row == "bottom":
        return [str(card) for card in (board.get("bottom", []) or board.get("bot", []) or [])]
    return [str(card) for card in (board.get(row, []) or [])]


def _placed_cards(candidate: dict[str, Any], row: str) -> list[str]:
    action = candidate.get("action") or {}
    return [str(card) for card, placed_row in (action.get("placements") or []) if placed_row == row]


def _kind_completed_by_placement(row_cards: list[str], placed_cards: list[str], n: int) -> bool:
    if len(row_cards) < n or not placed_cards:
        return False
    jokers = sum(1 for card in row_cards if _card_rank(card) == "X")
    counts: dict[str, int] = {}
    for card in row_cards:
        rank = _card_rank(card)
        if rank != "X":
            counts[rank] = counts.get(rank, 0) + 1
    for card in placed_cards:
        rank = _card_rank(card)
        if rank == "X":
            return any(count + jokers >= n for count in counts.values())
        if counts.get(rank, 0) + jokers >= n:
            return True
    return False


def _has_straight(cards: list[str]) -> bool:
    if len(cards) < 5:
        return False
    jokers = sum(1 for card in cards if _card_rank(card) == "X")
    rank_values = {
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "T": 10,
        "J": 11,
        "Q": 12,
        "K": 13,
        "A": 14,
    }
    values = {rank_values[rank] for rank in (_card_rank(card) for card in cards) if rank in rank_values}
    if 14 in values:
        values.add(1)
    for start in range(1, 11):
        needed = {start + offset for offset in range(5)}
        if len(needed - values) <= jokers:
            return True
    return False


def _has_flush(cards: list[str]) -> bool:
    if len(cards) < 5:
        return False
    jokers = sum(1 for card in cards if _card_rank(card) == "X")
    suits: dict[str, int] = {}
    for card in cards:
        card = str(card)
        if card.startswith("X") or len(card) < 2:
            continue
        suit = card[-1]
        suits[suit] = suits.get(suit, 0) + 1
    return any(count + jokers >= 5 for count in suits.values())


def _has_natural_kind(cards: list[str], n: int) -> bool:
    counts: dict[str, int] = {}
    for card in cards:
        rank = _card_rank(card)
        if rank != "X":
            counts[rank] = counts.get(rank, 0) + 1
    return any(count >= n for count in counts.values())


def _highest_rank_value(cards: list[str]) -> int:
    best = 0
    for card in cards:
        rank = _card_rank(card)
        if rank == "X":
            best = max(best, 15)
        else:
            best = max(best, int(RANK_VALUES.get(rank, 0)))
    return best


def _t1_tactical_score(candidate: dict[str, Any]) -> tuple[int, float, float, int]:
    board = candidate.get("board") or {}
    top_cards = _row_cards(board, "top")
    middle_cards = _row_cards(board, "middle")
    bottom_cards = _row_cards(board, "bottom")
    placed_top = _placed_cards(candidate, "top")
    placed_middle = _placed_cards(candidate, "middle")
    placed_bottom = _placed_cards(candidate, "bottom")

    priority = 0
    predicted_bust = float(candidate.get("predicted_bust", 1.0) or 1.0)
    if placed_top and _has_natural_kind(top_cards, 3):
        priority = max(priority, 540)
    elif _kind_completed_by_placement(top_cards, placed_top, 3):
        priority = max(priority, 470 if predicted_bust <= 0.5 else 300)
    if _has_straight(bottom_cards) and placed_bottom:
        priority = max(priority, 420)
    if _kind_completed_by_placement(bottom_cards, placed_bottom, 3):
        priority = max(priority, 360)
    existing_bottom_count = max(0, len(bottom_cards) - len(placed_bottom))
    if existing_bottom_count <= 1 and len(placed_bottom) == 2:
        priority = max(priority, 360)
    if _kind_completed_by_placement(middle_cards, placed_middle, 3):
        priority = max(priority, 320)
    if _kind_completed_by_placement(bottom_cards, placed_bottom, 2):
        priority = max(priority, 260)
    return (
        priority,
        -predicted_bust,
        float(candidate.get("model_score", float("-inf"))),
        -int(candidate.get("model_rank", 999999)),
    )


def _t2_tactical_score(candidate: dict[str, Any]) -> tuple[int, float, float, int]:
    board = candidate.get("board") or {}
    top_cards = _row_cards(board, "top")
    middle_cards = _row_cards(board, "middle")
    bottom_cards = _row_cards(board, "bottom")
    placed_top = _placed_cards(candidate, "top")
    placed_middle = _placed_cards(candidate, "middle")
    placed_bottom = _placed_cards(candidate, "bottom")

    priority = 0
    predicted_bust = float(candidate.get("predicted_bust", 1.0) or 1.0)
    if placed_bottom and _has_flush(bottom_cards):
        priority = max(priority, 620)
    if placed_bottom and _has_straight(bottom_cards):
        priority = max(priority, 560)
    if placed_top and _kind_completed_by_placement(top_cards, placed_top, 3):
        priority = max(priority, 540)
    if placed_top and _kind_completed_by_placement(top_cards, placed_top, 2):
        top_rank = _highest_rank_value(top_cards)
        priority = max(priority, 520 if top_rank >= RANK_VALUES["Q"] else 380)
    if _kind_completed_by_placement(middle_cards, placed_middle, 3):
        priority = max(priority, 480)
    if _kind_completed_by_placement(bottom_cards, placed_bottom, 3):
        priority = max(priority, 440)
    if _kind_completed_by_placement(middle_cards, placed_middle, 2):
        priority = max(priority, 340)
    if _kind_completed_by_placement(bottom_cards, placed_bottom, 2):
        priority = max(priority, 300)
    if placed_top and len(top_cards) <= 2:
        existing_top = [card for card in top_cards if card not in placed_top]
        placed_rank = _highest_rank_value(placed_top)
        existing_rank = _highest_rank_value(existing_top)
        if existing_rank >= RANK_VALUES["Q"] and placed_rank >= RANK_VALUES["9"] and predicted_bust <= 0.65:
            priority = max(priority, 280)

    return (
        priority,
        -predicted_bust,
        float(candidate.get("model_score", float("-inf"))),
        -int(candidate.get("model_rank", 999999)),
    )


def _t2_tactical_insurance_candidates(
    ranked: list[dict[str, Any]],
    config: HybridConfig,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    limit = max(0, min(int(config.t2_tactical_insurance_k), int(limit)))
    if limit <= 0:
        return []
    has_safe = any(float(item.get("predicted_bust", 1.0)) < config.high_bust_threshold for item in ranked)
    candidates = []
    for item in ranked:
        if has_safe and float(item.get("predicted_bust", 1.0)) >= config.high_bust_threshold:
            continue
        score = _t2_tactical_score(item)
        if score[0] <= 0:
            continue
        candidates.append((score, item))
    selected = [item for _, item in sorted(candidates, key=lambda row: row[0], reverse=True)[:limit]]
    for item in selected:
        item["insurance_reason"] = "t2_tactical"
    return selected


def _t1_tactical_insurance_candidates(
    ranked: list[dict[str, Any]],
    config: HybridConfig,
) -> list[dict[str, Any]]:
    limit = max(0, int(config.t1_sync_tactical_insurance_k))
    if limit <= 0:
        return []
    has_safe = any(float(item.get("predicted_bust", 1.0)) < config.high_bust_threshold for item in ranked)
    candidates = []
    for item in ranked:
        if has_safe and float(item.get("predicted_bust", 1.0)) >= config.high_bust_threshold:
            continue
        score = _t1_tactical_score(item)
        if score[0] <= 0:
            continue
        candidates.append((score, item))
    selected = [item for _, item in sorted(candidates, key=lambda row: row[0], reverse=True)[:limit]]
    for item in selected:
        item["sync_insurance_reason"] = "tactical"
    return selected


def _sync_tactical_insurance_candidates(
    ranked: list[dict[str, Any]],
    config: HybridConfig,
    *,
    turn: int,
    limit: int,
) -> list[dict[str, Any]]:
    if int(turn) == 1:
        return _t1_tactical_insurance_candidates(ranked, config)[: max(0, int(limit))]
    if int(turn) == 2:
        return _t2_tactical_insurance_candidates(ranked, config, limit=max(0, int(limit)))
    return []


def _sync_pool_insurance_candidates(
    ranked: list[dict[str, Any]],
    config: HybridConfig,
    *,
    turn: int,
    limit: int,
) -> list[dict[str, Any]]:
    if int(turn) != 2 or int(config.t2_tactical_insurance_k) <= 0:
        return []
    reason_priority = {
        "t2_tactical": 0,
        "low_bust": 1,
        "fl": 2,
    }
    candidates = [
        item
        for item in ranked
        if str(item.get("insurance_reason") or "") in reason_priority
    ]
    candidates.sort(
        key=lambda item: (
            reason_priority[str(item.get("insurance_reason") or "")],
            float(item.get("predicted_bust", 1.0) or 1.0),
            -_candidate_fl_score(item),
            int(item.get("model_rank", 999999)),
        )
    )
    return candidates[: max(0, int(limit))]


def _sync_insurance_candidates(
    ranked: list[dict[str, Any]],
    config: HybridConfig,
    *,
    turn: int,
    limit: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for source in (
        _sync_tactical_insurance_candidates(ranked, config, turn=turn, limit=limit),
        _sync_pool_insurance_candidates(ranked, config, turn=turn, limit=limit),
    ):
        for item in source:
            action_idx = int(item["action_idx"])
            if action_idx in seen:
                continue
            out.append(item)
            seen.add(action_idx)
            if len(out) >= max(0, int(limit)):
                return out
    return out


def _has_kind_with_jokers(cards: list[str], n: int) -> bool:
    jokers = sum(1 for card in cards if _card_rank(card) == "X")
    counts: dict[str, int] = {}
    for card in cards:
        rank = _card_rank(card)
        if rank != "X":
            counts[rank] = counts.get(rank, 0) + 1
    return any(count + jokers >= n for count in counts.values())


def _t2_post_refine_policy_accepts(
    obs: Observation,
    extra_candidates: list[dict[str, Any]],
    config: HybridConfig,
) -> tuple[bool, dict[str, Any]]:
    policy = str(config.t2_post_refine_policy or "always")
    max_extra_model_score = max(
        (float(item.get("model_score", float("-inf"))) for item in extra_candidates),
        default=float("-inf"),
    )
    details: dict[str, Any] = {
        "policy": policy,
        "extra_candidate_count": int(len(extra_candidates)),
        "max_extra_model_score": max_extra_model_score,
    }
    if policy in ("", "always"):
        details["accepted"] = True
        return True, details
    if policy == "structured_top_model_min":
        own_top_structured = _t2_own_top_structured(obs)
        score_min = float(config.t2_post_refine_extra_model_score_min)
        details.update(
            {
                "own_top_structured": bool(own_top_structured),
                "extra_model_score_min": score_min,
            }
        )
        if not own_top_structured:
            details["accepted"] = False
            details["reject_reason"] = "own_top_not_structured"
            return False, details
        if max_extra_model_score < score_min:
            details["accepted"] = False
            details["reject_reason"] = "extra_model_score_below_min"
            return False, details
        details["accepted"] = True
        return True, details
    details["accepted"] = False
    details["reject_reason"] = f"unknown_policy:{policy}"
    return False, details


def _t2_own_top_structured(obs: Observation) -> bool:
    top_cards = [str(card) for card in (getattr(obs.board_self, "top", []) or []) if card]
    return any(_card_rank(card) == "X" for card in top_cards) or _has_kind_with_jokers(top_cards, 2)


def t1_blocker_features(candidate: dict[str, Any]) -> dict[str, float]:
    board = candidate.get("board") or {}
    opponent_board = candidate.get("opponent_board") or {}
    known_discards = [str(card) for card in (candidate.get("known_discards") or []) if card]
    action = candidate.get("action") or {}
    discard = str(action.get("discard") or "")

    own_cards = _row_cards(board, "top") + _row_cards(board, "middle") + _row_cards(board, "bottom")
    opp_top = _row_cards(opponent_board, "top")
    opp_middle = _row_cards(opponent_board, "middle")
    opp_bottom = _row_cards(opponent_board, "bottom")
    opp_cards = opp_top + opp_middle + opp_bottom
    dead_cards = list(own_cards) + list(opp_cards) + list(known_discards)
    if discard:
        dead_cards.append(discard)

    def count_rank(cards: list[str], rank: str) -> float:
        return float(sum(1 for card in cards if _card_rank(card) == rank))

    features = {
        "opp_top_count": float(len(opp_top)),
        "opp_middle_count": float(len(opp_middle)),
        "opp_bottom_count": float(len(opp_bottom)),
        "opp_top_pair": 1.0 if _has_kind_with_jokers(opp_top, 2) else 0.0,
        "opp_top_trips": 1.0 if _has_kind_with_jokers(opp_top, 3) else 0.0,
        "opp_middle_pair": 1.0 if _has_kind_with_jokers(opp_middle, 2) else 0.0,
        "opp_middle_trips": 1.0 if _has_kind_with_jokers(opp_middle, 3) else 0.0,
        "opp_bottom_pair": 1.0 if _has_kind_with_jokers(opp_bottom, 2) else 0.0,
        "opp_bottom_trips": 1.0 if _has_kind_with_jokers(opp_bottom, 3) else 0.0,
    }
    for rank in BLOCKER_RANKS:
        key = rank.lower()
        dead = count_rank(dead_cards, rank)
        own = count_rank(own_cards, rank)
        opp = count_rank(opp_cards, rank)
        features[f"dead_{key}"] = dead
        features[f"own_visible_{key}"] = own
        features[f"opp_visible_{key}"] = opp
        features[f"remaining_{key}"] = max(0.0, 4.0 - dead)
    return features


def t1_action_shape_features(candidate: dict[str, Any]) -> dict[str, float]:
    board = candidate.get("board") or {}
    action = candidate.get("action") or {}
    discard = str(action.get("discard") or "")
    placed_top = _placed_cards(candidate, "top")
    placed_middle = _placed_cards(candidate, "middle")
    placed_bottom = _placed_cards(candidate, "bottom")
    top_cards = _row_cards(board, "top")
    middle_cards = _row_cards(board, "middle")
    bottom_cards = _row_cards(board, "bottom")

    placed_top_count = float(len(placed_top))
    placed_middle_count = float(len(placed_middle))
    placed_bottom_count = float(len(placed_bottom))
    top_count = float(len(top_cards))
    middle_count = float(len(middle_cards))
    bottom_count = float(len(bottom_cards))
    existing_top_count = max(0.0, top_count - placed_top_count)
    existing_middle_count = max(0.0, middle_count - placed_middle_count)
    existing_bottom_count = max(0.0, bottom_count - placed_bottom_count)
    placed_ranks = [_card_rank(card) for card in placed_top + placed_middle + placed_bottom]
    discard_rank = _card_rank(discard) if discard else ""

    return {
        "placed_top_count": placed_top_count,
        "placed_middle_count": placed_middle_count,
        "placed_bottom_count": placed_bottom_count,
        "final_top_count": top_count,
        "final_middle_count": middle_count,
        "final_bottom_count": bottom_count,
        "existing_top_count": existing_top_count,
        "existing_middle_count": existing_middle_count,
        "existing_bottom_count": existing_bottom_count,
        "bottom_sparse_two_placed": 1.0
        if existing_bottom_count <= 1.0 and placed_bottom_count == 2.0
        else 0.0,
        "middle_sparse_two_placed": 1.0
        if existing_middle_count <= 1.0 and placed_middle_count == 2.0
        else 0.0,
        "top_full_after": 1.0 if top_count >= 3.0 else 0.0,
        "middle_full_after": 1.0 if middle_count >= 5.0 else 0.0,
        "bottom_full_after": 1.0 if bottom_count >= 5.0 else 0.0,
        "final_top_pair": 1.0 if _has_kind_with_jokers(top_cards, 2) else 0.0,
        "final_top_trips": 1.0 if _has_kind_with_jokers(top_cards, 3) else 0.0,
        "final_middle_pair": 1.0 if _has_kind_with_jokers(middle_cards, 2) else 0.0,
        "final_middle_trips": 1.0 if _has_kind_with_jokers(middle_cards, 3) else 0.0,
        "final_bottom_pair": 1.0 if _has_kind_with_jokers(bottom_cards, 2) else 0.0,
        "final_bottom_trips": 1.0 if _has_kind_with_jokers(bottom_cards, 3) else 0.0,
        "placed_top_has_a": 1.0 if "A" in [_card_rank(card) for card in placed_top] else 0.0,
        "placed_top_has_k": 1.0 if "K" in [_card_rank(card) for card in placed_top] else 0.0,
        "placed_top_has_q": 1.0 if "Q" in [_card_rank(card) for card in placed_top] else 0.0,
        "placed_has_joker": 1.0 if "X" in placed_ranks else 0.0,
        "discard_a": 1.0 if discard_rank == "A" else 0.0,
        "discard_k": 1.0 if discard_rank == "K" else 0.0,
        "discard_q": 1.0 if discard_rank == "Q" else 0.0,
        "discard_joker": 1.0 if discard_rank == "X" else 0.0,
    }


def _is_t1_bottom_sparse_two_placed(candidate: dict[str, Any]) -> bool:
    return bool(t1_action_shape_features(candidate).get("bottom_sparse_two_placed", 0.0) >= 1.0)


def _t1_bottom_sparse_rescue_candidate(
    best_candidate: dict[str, Any] | None,
    refined_candidates: list[dict[str, Any]],
    *,
    margin: float,
) -> tuple[dict[str, Any] | None, float | None]:
    if (
        best_candidate is None
        or best_candidate.get("refined_score") is None
        or margin < 0.0
        or _is_t1_bottom_sparse_two_placed(best_candidate)
    ):
        return None, None
    sparse_candidates = [
        item
        for item in refined_candidates
        if item.get("refined_score") is not None and _is_t1_bottom_sparse_two_placed(item)
    ]
    if not sparse_candidates:
        return None, None
    sparse_best = max(
        sparse_candidates,
        key=lambda item: (
            float(item.get("refined_score", float("-inf"))),
            float(item.get("model_score", float("-inf"))),
            -int(item.get("model_rank", 999999)),
        ),
    )
    rescue_delta = float(best_candidate.get("refined_score", 0.0) or 0.0) - float(
        sparse_best.get("refined_score", 0.0) or 0.0
    )
    if rescue_delta <= margin:
        return sparse_best, rescue_delta
    return None, rescue_delta


def _t1_no_refine_fallback_candidate(
    sync_candidates: list[dict[str, Any]],
    output_candidates: list[dict[str, Any]],
    *,
    policy: str,
) -> dict[str, Any] | None:
    if not output_candidates:
        return None
    if policy == "model" or not sync_candidates:
        return output_candidates[0]
    if policy == "fl_safe":
        return max(
            sync_candidates,
            key=lambda item: (
                float(item.get("predicted_fl", 0.0) or 0.0)
                - float(item.get("predicted_bust", 0.0) or 0.0),
                float(item.get("model_score", float("-inf"))),
                -int(item.get("model_rank", 999999)),
            ),
        )
    if policy == "low_bust":
        return min(
            sync_candidates,
            key=lambda item: (
                float(item.get("predicted_bust", 1.0) or 1.0),
                -float(item.get("model_score", float("-inf"))),
                int(item.get("model_rank", 999999)),
            ),
        )
    if policy == "model_bust":
        return max(
            sync_candidates,
            key=lambda item: (
                float(item.get("model_score", float("-inf"))) - 10.0 * float(item.get("predicted_bust", 0.0) or 0.0),
                -int(item.get("model_rank", 999999)),
            ),
        )
    raise ValueError(f"unknown T1 no-refine fallback policy: {policy}")


def _inject_sync_insurance(
    selected: list[dict[str, Any]],
    insurance: list[dict[str, Any]],
    *,
    limit: int,
    prepend: bool = True,
) -> list[dict[str, Any]]:
    if limit <= 0 or not insurance:
        return selected[:limit]
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    first = insurance if prepend else selected
    second = selected if prepend else insurance
    for item in first:
        if len(out) >= limit:
            break
        action_idx = int(item["action_idx"])
        if action_idx not in seen:
            out.append(item)
            seen.add(action_idx)
    for item in second:
        if len(out) >= limit:
            break
        action_idx = int(item["action_idx"])
        if action_idx not in seen:
            out.append(item)
            seen.add(action_idx)
    return out[:limit]


def _rank_feature(candidate: dict[str, Any], name: str, default: int = 999) -> float:
    try:
        return float(candidate.get(name, default) or default)
    except (TypeError, ValueError):
        return float(default)


def t1_sync_selector_features(candidate: dict[str, Any]) -> dict[str, float]:
    """Features available before recursive T1 refinement starts."""
    model_rank = max(_rank_feature(candidate, "model_rank"), 1.0)
    refinement_rank = max(_rank_feature(candidate, "refinement_rank"), 1.0)
    aux_rank = max(_rank_feature(candidate, "aux_model_rank"), 1.0)
    model_score = float(candidate.get("model_score", 0.0) or 0.0)
    model_raw_score = float(candidate.get("model_raw_score", 0.0) or 0.0)
    predicted_bust = float(candidate.get("predicted_bust", 0.0) or 0.0)
    predicted_fl = float(candidate.get("predicted_fl", 0.0) or 0.0)
    fl_types = candidate.get("predicted_fl_types") or {}
    predicted_qq = float(fl_types.get("qq", 0.0) or 0.0)
    predicted_kk = float(fl_types.get("kk", 0.0) or 0.0)
    predicted_aa = float(fl_types.get("aa", 0.0) or 0.0)
    predicted_trips = float(fl_types.get("trips", 0.0) or 0.0)
    fl_score = max(predicted_fl, predicted_qq, predicted_kk, predicted_aa, predicted_trips)
    insurance_reason = str(candidate.get("insurance_reason") or "")
    has_aux = 1.0 if "aux_model_score" in candidate else 0.0
    aux_model_score = float(candidate.get("aux_model_score", 0.0) or 0.0)

    return {
        "bias": 1.0,
        "model_score": model_score,
        "model_raw_score": model_raw_score,
        "neg_model_rank": -model_rank,
        "inv_model_rank": 1.0 / model_rank,
        "rank_le_1": 1.0 if model_rank <= 1.0 else 0.0,
        "rank_le_3": 1.0 if model_rank <= 3.0 else 0.0,
        "rank_le_6": 1.0 if model_rank <= 6.0 else 0.0,
        "rank_le_10": 1.0 if model_rank <= 10.0 else 0.0,
        "rank_le_15": 1.0 if model_rank <= 15.0 else 0.0,
        "neg_refinement_rank": -refinement_rank,
        "inv_refinement_rank": 1.0 / refinement_rank,
        "refinement_rank_le_6": 1.0 if refinement_rank <= 6.0 else 0.0,
        "refinement_rank_le_10": 1.0 if refinement_rank <= 10.0 else 0.0,
        "predicted_bust": predicted_bust,
        "predicted_safe": 1.0 - predicted_bust,
        "predicted_fl": predicted_fl,
        "candidate_fl_score": fl_score,
        "predicted_qq": predicted_qq,
        "predicted_kk": predicted_kk,
        "predicted_aa": predicted_aa,
        "predicted_trips": predicted_trips,
        "premium_fl_sum": predicted_kk + predicted_aa + predicted_trips,
        "insurance_low_bust": 1.0 if insurance_reason == "low_bust" else 0.0,
        "insurance_fl": 1.0 if insurance_reason == "fl" else 0.0,
        "insurance_aux": 1.0 if insurance_reason == "aux_shortlist" else 0.0,
        "has_aux_score": has_aux,
        "aux_model_score": aux_model_score,
        "neg_aux_model_rank": -aux_rank if has_aux else 0.0,
        "aux_rank_le_10": 1.0 if has_aux and aux_rank <= 10.0 else 0.0,
        "model_score_x_inv_rank": model_score / model_rank,
        "fl_score_x_safe": fl_score * (1.0 - predicted_bust),
    }


def t2_sync_selector_features(candidate: dict[str, Any]) -> dict[str, float]:
    features = t1_sync_selector_features(candidate)
    features.update(t1_action_shape_features(candidate))
    features.update(t1_blocker_features(candidate))
    forced_bust = 1.0 if bool(candidate.get("forced_bust", False)) else 0.0
    features.update(
        {
            "forced_bust": forced_bust,
            "not_forced_bust": 1.0 - forced_bust,
            "model_score_x_not_forced_bust": features["model_score"] * (1.0 - forced_bust),
        }
    )
    return features


def _linear_selector_score(selector: dict[str, Any], features: dict[str, float]) -> float:
    feature_names = list(selector.get("features") or [])
    weights = list(selector.get("weights") or [])
    means = dict(selector.get("means") or {})
    scales = dict(selector.get("scales") or {})
    value = float(selector.get("intercept", 0.0))
    for name, weight in zip(feature_names, weights):
        raw = float(features.get(name, 0.0))
        if name != "bias":
            raw = (raw - float(means.get(name, 0.0))) / max(float(scales.get(name, 1.0)), 1e-9)
        value += float(weight) * raw
    return value


def t1_final_selector_features(candidate: dict[str, Any]) -> dict[str, float]:
    """Features available after T1 recursive refinement."""
    features = t1_sync_selector_features(candidate)
    refined_score = float(candidate.get("refined_score", 0.0) or 0.0)
    model_score = float(candidate.get("model_score", 0.0) or 0.0)
    refined_rank = max(_rank_feature(candidate, "refined_rank"), 1.0)
    samples = float(candidate.get("samples", 0.0) or 0.0)
    elapsed_ms = float(candidate.get("elapsed_ms", 0.0) or 0.0)
    features.update(
        {
            "refined_score": refined_score,
            "refined_minus_model": refined_score - model_score,
            "refined_plus_model": refined_score + model_score,
            "neg_refined_rank": -refined_rank,
            "inv_refined_rank": 1.0 / refined_rank,
            "refined_rank_le_1": 1.0 if refined_rank <= 1.0 else 0.0,
            "refined_rank_le_3": 1.0 if refined_rank <= 3.0 else 0.0,
            "refined_rank_le_5": 1.0 if refined_rank <= 5.0 else 0.0,
            "samples": samples,
            "log_samples": 0.0 if samples <= 0.0 else math.log1p(samples),
            "elapsed_ms": elapsed_ms,
            "sync_selector_score": float(candidate.get("sync_selector_score", 0.0) or 0.0),
            "refined_x_safe": refined_score * float(features.get("predicted_safe", 0.0)),
            "refined_x_fl": refined_score * float(features.get("candidate_fl_score", 0.0)),
        }
    )
    features.update(t1_action_shape_features(candidate))
    features.update(t1_blocker_features(candidate))
    return features


def t2_final_selector_features(candidate: dict[str, Any]) -> dict[str, float]:
    """Features available after T2 partial exact refinement."""
    features = t1_final_selector_features(candidate)
    features.update(t2_sync_selector_features(candidate))
    forced_bust = 1.0 if bool(candidate.get("forced_bust", False)) else 0.0
    refined_score = float(candidate.get("refined_score", 0.0) or 0.0)
    model_score = float(candidate.get("model_score", 0.0) or 0.0)
    model_rank = max(_rank_feature(candidate, "model_rank"), 1.0)
    refined_rank = max(_rank_feature(candidate, "refined_rank"), 1.0)
    features.update(
        {
            "forced_bust": forced_bust,
            "not_forced_bust": 1.0 - forced_bust,
            "model_score_x_not_forced_bust": model_score * (1.0 - forced_bust),
            "refined_score_x_not_forced_bust": refined_score * (1.0 - forced_bust),
            "refined_minus_model_per_rank": (refined_score - model_score) / model_rank,
            "refined_rank_minus_model_rank": refined_rank - model_rank,
        }
    )
    return features


def _candidate_fl_type(candidate: dict[str, Any], key: str) -> float:
    predicted_types = candidate.get("predicted_fl_types") or {}
    if key in predicted_types:
        return float(predicted_types.get(key, 0.0) or 0.0)
    return float(candidate.get(f"predicted_{key}", 0.0) or 0.0)


def _t1_arbitration_candidate(
    current: dict[str, Any],
    refined_candidates: list[dict[str, Any]],
    *,
    selector: dict[str, Any] | None,
    policy: str,
    score_key: str = "arbitration_selector_score",
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Optionally replace the current T1 final selector pick.

    Supported policies are deliberately narrow and must be enabled explicitly.
    They are offline-derived gates for testing challenger final selectors
    without changing the default T1 runtime.
    """
    if selector is None or policy == "none" or not refined_candidates:
        return current, None
    supported = {
        "current_refined_rank_ge2_and_qq_nonnegative",
        "pairwise_refined_delta_ge_m086_current_bust_ge_0026",
        "refined_score_model_rank_ge6_conflict_ge_m086",
        "mixed_kk_delta_le_0043_model_delta_le_1261",
        "mixed_kk_delta_le_0043_model_delta_le_1111_conflict_ge_m0483",
        "remain85_current_margin_le_0664",
    }
    if policy not in supported:
        raise ValueError(f"unsupported T1 arbitration policy: {policy}")

    scored: list[tuple[float, dict[str, Any]]] = []
    for item in refined_candidates:
        score = _linear_selector_score(selector, t1_final_selector_features(item))
        item[score_key] = score
        scored.append((score, item))
    if policy == "refined_score_model_rank_ge6_conflict_ge_m086":
        challenger = max(
            refined_candidates,
            key=lambda item: (
                float(item.get("refined_score", float("-inf"))),
                float(item.get("model_score", float("-inf"))),
                -int(item.get("model_rank", 999)),
            ),
        )
        challenger_score = float(challenger.get(score_key, 0.0) or 0.0)
    else:
        challenger_score, challenger = max(
            scored,
            key=lambda pair: (
                pair[0],
                float(pair[1].get("refined_score", float("-inf"))),
                float(pair[1].get("model_score", float("-inf"))),
                -int(pair[1].get("model_rank", 999)),
            ),
        )
    current_score = float(current.get(score_key, float("nan")))
    current_under_current = float(current.get("final_selector_score", 0.0) or 0.0)
    challenger_under_current = float(challenger.get("final_selector_score", 0.0) or 0.0)
    current_margin_under_current = current_under_current - challenger_under_current
    current_refined_rank = max(_rank_feature(current, "refined_rank"), 1.0)
    refined_score_delta = float(challenger.get("refined_score", 0.0) or 0.0) - float(
        current.get("refined_score", 0.0) or 0.0
    )
    current_predicted_bust = float(current.get("predicted_bust", 0.0) or 0.0)
    predicted_qq_delta = _candidate_fl_type(challenger, "qq") - _candidate_fl_type(current, "qq")
    predicted_kk_delta = _candidate_fl_type(challenger, "kk") - _candidate_fl_type(current, "kk")
    model_score_delta = float(challenger.get("model_score", 0.0) or 0.0) - float(
        current.get("model_score", 0.0) or 0.0
    )
    same_action = int(current.get("action_idx", -1)) == int(challenger.get("action_idx", -2))
    challenger_model_rank = max(_rank_feature(challenger, "model_rank"), 1.0)
    selector_conflict_margin = refined_score_delta - (
        current_score - float(challenger.get(score_key, 0.0) or 0.0)
    )
    if policy == "current_refined_rank_ge2_and_qq_nonnegative":
        accepted = (
            not same_action
            and current_refined_rank >= 2.0
            and predicted_qq_delta >= 0.0
        )
    elif policy == "pairwise_refined_delta_ge_m086_current_bust_ge_0026":
        accepted = (
            not same_action
            and refined_score_delta >= -0.8597222222222314
            and current_predicted_bust >= 0.02634526789188385
        )
    elif policy == "refined_score_model_rank_ge6_conflict_ge_m086":
        accepted = (
            not same_action
            and challenger_model_rank >= 6.0
            and selector_conflict_margin >= -0.8622868813378028
        )
    elif policy == "mixed_kk_delta_le_0043_model_delta_le_1261":
        accepted = (
            not same_action
            and predicted_kk_delta <= 0.043167173862457275
            and model_score_delta <= 1.261122226715088
        )
    elif policy == "mixed_kk_delta_le_0043_model_delta_le_1111_conflict_ge_m0483":
        accepted = (
            not same_action
            and predicted_kk_delta <= 0.043167173862457275
            and model_score_delta <= 1.110815
            and selector_conflict_margin >= -0.48343
        )
    else:
        accepted = (
            not same_action
            and current_margin_under_current <= 0.664098569143615
        )
    details = {
        "policy": policy,
        "accepted": bool(accepted),
        "current_action_idx": int(current.get("action_idx", -1)),
        "challenger_action_idx": int(challenger.get("action_idx", -1)),
        "current_refined_rank": float(current_refined_rank),
        "challenger_model_rank": float(challenger_model_rank),
        "refined_score_delta": float(refined_score_delta),
        "selector_conflict_margin": float(selector_conflict_margin),
        "current_margin_under_current": float(current_margin_under_current),
        "current_predicted_bust": float(current_predicted_bust),
        "predicted_qq_delta": float(predicted_qq_delta),
        "predicted_kk_delta": float(predicted_kk_delta),
        "model_score_delta": float(model_score_delta),
        "challenger_selector_score": float(challenger_score),
        "current_selector_score": current_score,
        "selector_score_key": score_key,
        "same_action": bool(same_action),
    }
    return (challenger if accepted else current), details


def _t1_dual_arbitration_candidate(
    base_current: dict[str, Any],
    primary_candidate: dict[str, Any],
    primary_details: dict[str, Any] | None,
    challenger_candidate: dict[str, Any],
    challenger_details: dict[str, Any] | None,
    *,
    model_top1: dict[str, Any] | None,
    policy: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if policy == "none":
        return primary_candidate, primary_details
    supported = {"new_if_kk_and_premium_delta_nonpositive"}
    if policy not in supported:
        raise ValueError(f"unsupported T1 dual arbitration gate policy: {policy}")
    if model_top1 is None or challenger_details is None:
        return primary_candidate, primary_details

    primary_features = _override_gate_features(model_top1, primary_candidate)
    challenger_features = _override_gate_features(model_top1, challenger_candidate)
    predicted_kk_delta_delta = (
        float(challenger_features.get("predicted_kk_delta", 0.0))
        - float(primary_features.get("predicted_kk_delta", 0.0))
    )
    premium_fl_delta_sum_delta = (
        float(challenger_features.get("premium_fl_delta_sum", 0.0))
        - float(primary_features.get("premium_fl_delta_sum", 0.0))
    )
    same_action = int(primary_candidate.get("action_idx", -1)) == int(
        challenger_candidate.get("action_idx", -2)
    )
    accepted = (
        not same_action
        and predicted_kk_delta_delta <= -1.5966406863299198e-07
        and premium_fl_delta_sum_delta <= 0.0
    )
    chosen = challenger_candidate if accepted else primary_candidate
    details = {
        "policy": policy,
        "accepted": int(chosen.get("action_idx", -1)) != int(base_current.get("action_idx", -2)),
        "dual_gate_accepted": bool(accepted),
        "base_action_idx": int(base_current.get("action_idx", -1)),
        "primary_action_idx": int(primary_candidate.get("action_idx", -1)),
        "challenger_action_idx": int(challenger_candidate.get("action_idx", -1)),
        "chosen_action_idx": int(chosen.get("action_idx", -1)),
        "same_action": bool(same_action),
        "predicted_kk_delta_delta": float(predicted_kk_delta_delta),
        "premium_fl_delta_sum_delta": float(premium_fl_delta_sum_delta),
        "primary_accepted": bool((primary_details or {}).get("accepted", False)),
        "challenger_accepted": bool((challenger_details or {}).get("accepted", False)),
        "primary_refined_score_delta": float((primary_details or {}).get("refined_score_delta", 0.0) or 0.0),
        "challenger_refined_score_delta": float(
            (challenger_details or {}).get("refined_score_delta", 0.0) or 0.0
        ),
        "primary_predicted_kk_delta": float(primary_features.get("predicted_kk_delta", 0.0)),
        "challenger_predicted_kk_delta": float(challenger_features.get("predicted_kk_delta", 0.0)),
        "primary_premium_fl_delta_sum": float(primary_features.get("premium_fl_delta_sum", 0.0)),
        "challenger_premium_fl_delta_sum": float(challenger_features.get("premium_fl_delta_sum", 0.0)),
    }
    return chosen, details


def _t1_final_challenger_candidate(
    current: dict[str, Any],
    refined_candidates: list[dict[str, Any]],
    *,
    selector: dict[str, Any] | None,
    gate: dict[str, Any] | None,
    model_top1: dict[str, Any] | None,
    override_gate: dict[str, Any] | None,
    override_gate_threshold: float,
    policy: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Optionally switch from the completed T1 stack to a challenger selector.

    This runs after the normal final-selector/arbitration/override/rescue
    stack.  The challenger is selected from the same refined candidates but is
    not passed through the existing arbitration or override stack again.
    """
    if selector is None or policy == "none" or not refined_candidates:
        return current, None
    supported = {
        "guard2_bust_delta_ge_m066_rank_gap_delta_ge_m1",
        "guard2_bust_rank_refined_delta_ge_m0778",
        "guard2_bust_delta_ge_m066",
        "logistic_gate",
        "logistic_gate_refined_override_delta_ge_0677",
    }
    if policy not in supported:
        raise ValueError(f"unsupported T1 final challenger gate policy: {policy}")
    if model_top1 is None:
        return current, None

    scored: list[tuple[float, dict[str, Any]]] = []
    for item in refined_candidates:
        score = _linear_selector_score(selector, t1_final_selector_features(item))
        item["final_challenger_selector_score"] = score
        scored.append((score, item))
    challenger_score, challenger = max(
        scored,
        key=lambda pair: (
            pair[0],
            float(pair[1].get("refined_score", float("-inf"))),
            float(pair[1].get("model_score", float("-inf"))),
            -int(pair[1].get("model_rank", 999)),
        ),
    )
    same_action = int(current.get("action_idx", -1)) == int(challenger.get("action_idx", -2))
    current_features = _override_gate_features(model_top1, current)
    challenger_features = _override_gate_features(model_top1, challenger)
    best_predicted_bust_delta = (
        float(challenger_features.get("best_predicted_bust", 0.0))
        - float(current_features.get("best_predicted_bust", 0.0))
    )
    model_rank_gap_delta = (
        float(challenger_features.get("model_rank_gap", 0.0))
        - float(current_features.get("model_rank_gap", 0.0))
    )
    best_model_score_delta = (
        float(challenger_features.get("best_model_score", 0.0))
        - float(current_features.get("best_model_score", 0.0))
    )
    refined_score_delta = float(challenger.get("refined_score", 0.0) or 0.0) - float(
        current.get("refined_score", 0.0) or 0.0
    )
    gate_features: dict[str, float] | None = None
    gate_probability: float | None = None
    gate_threshold = None
    if policy in {"logistic_gate", "logistic_gate_refined_override_delta_ge_0677"}:
        if gate is None:
            return current, None
        current_probability = (
            _linear_gate_probability(override_gate, current_features)
            if override_gate is not None
            else 0.0
        )
        challenger_probability = (
            _linear_gate_probability(override_gate, challenger_features)
            if override_gate is not None
            else 0.0
        )
        current_overridden = int(current.get("action_idx", -1)) != int(model_top1.get("action_idx", -2))
        challenger_overridden = int(challenger.get("action_idx", -1)) != int(model_top1.get("action_idx", -2))
        gate_features = {
            "same_action": 1.0 if same_action else 0.0,
            "refined_score_delta": float(refined_score_delta),
            "refined_override_delta_delta": float(refined_score_delta),
            "best_refined_score_delta": float(refined_score_delta),
            "model_top1_refined_score_delta": 0.0,
            "override_gate_probability_delta": float(challenger_probability - current_probability),
            "current_override_gate_probability": float(current_probability),
            "challenger_override_gate_probability": float(challenger_probability),
            "current_model_top1_overridden": 1.0 if current_overridden else 0.0,
            "challenger_model_top1_overridden": 1.0 if challenger_overridden else 0.0,
            "current_override_accepted": 1.0 if current_overridden and current_probability >= override_gate_threshold else 0.0,
            "challenger_override_accepted": 1.0
            if challenger_overridden and challenger_probability >= override_gate_threshold
            else 0.0,
            "current_refined_delta": float(current_features.get("refined_delta", 0.0)),
            "challenger_refined_delta": float(challenger_features.get("refined_delta", 0.0)),
            "t1_predicted_bust_delta_delta": float(
                challenger_features.get("predicted_bust_delta", 0.0)
                - current_features.get("predicted_bust_delta", 0.0)
            ),
            "t1_predicted_fl_delta_delta": float(
                challenger_features.get("predicted_fl_delta", 0.0)
                - current_features.get("predicted_fl_delta", 0.0)
            ),
            "t1_predicted_qq_delta_delta": float(
                challenger_features.get("predicted_qq_delta", 0.0)
                - current_features.get("predicted_qq_delta", 0.0)
            ),
            "t1_predicted_kk_delta_delta": float(
                challenger_features.get("predicted_kk_delta", 0.0)
                - current_features.get("predicted_kk_delta", 0.0)
            ),
            "t1_predicted_aa_delta_delta": float(
                challenger_features.get("predicted_aa_delta", 0.0)
                - current_features.get("predicted_aa_delta", 0.0)
            ),
            "t1_premium_fl_delta_sum_delta": float(
                challenger_features.get("premium_fl_delta_sum", 0.0)
                - current_features.get("premium_fl_delta_sum", 0.0)
            ),
            "t1_model_rank_gap_delta": float(model_rank_gap_delta),
            "t1_best_predicted_bust_delta": float(best_predicted_bust_delta),
            "t1_best_predicted_fl_delta": float(
                challenger_features.get("best_predicted_fl", 0.0)
                - current_features.get("best_predicted_fl", 0.0)
            ),
            "t1_best_model_score_delta": float(best_model_score_delta),
            "challenger_arbitration_accepted": 0.0,
            "challenger_arbitration_current_margin_under_current": 0.0,
            "challenger_arbitration_refined_score_delta": 0.0,
            "challenger_arbitration_model_score_delta": 0.0,
            "challenger_arbitration_predicted_kk_delta": 0.0,
        }
        gate_probability = _linear_gate_probability(gate, gate_features)
        gate_threshold = float(gate.get("threshold", 0.5))
        accepted = not same_action and gate_probability >= gate_threshold
        if policy == "logistic_gate_refined_override_delta_ge_0677":
            accepted = accepted and refined_score_delta >= 0.6770833333333357
    elif policy == "guard2_bust_delta_ge_m066":
        accepted = (
            not same_action
            and best_predicted_bust_delta >= -0.06600075960159302
        )
    elif policy == "guard2_bust_rank_refined_delta_ge_m0778":
        accepted = (
            not same_action
            and best_predicted_bust_delta >= -0.06600075960159302
            and model_rank_gap_delta >= -1.0
            and refined_score_delta >= -0.7777777777777772
        )
    else:
        accepted = (
            not same_action
            and best_predicted_bust_delta >= -0.06600075960159302
            and model_rank_gap_delta >= -1.0
        )
    chosen = challenger if accepted else current
    details = {
        "policy": policy,
        "accepted": bool(accepted),
        "current_action_idx": int(current.get("action_idx", -1)),
        "challenger_action_idx": int(challenger.get("action_idx", -1)),
        "chosen_action_idx": int(chosen.get("action_idx", -1)),
        "same_action": bool(same_action),
        "best_predicted_bust_delta": float(best_predicted_bust_delta),
        "model_rank_gap_delta": float(model_rank_gap_delta),
        "best_model_score_delta": float(best_model_score_delta),
        "refined_score_delta": float(refined_score_delta),
        "gate_probability": gate_probability,
        "gate_threshold": gate_threshold,
        "gate_features": gate_features,
        "current_best_predicted_bust": float(current_features.get("best_predicted_bust", 0.0)),
        "challenger_best_predicted_bust": float(challenger_features.get("best_predicted_bust", 0.0)),
        "current_model_rank_gap": float(current_features.get("model_rank_gap", 0.0)),
        "challenger_model_rank_gap": float(challenger_features.get("model_rank_gap", 0.0)),
        "current_best_model_score": float(current_features.get("best_model_score", 0.0)),
        "challenger_best_model_score": float(challenger_features.get("best_model_score", 0.0)),
        "challenger_selector_score": float(challenger_score),
    }
    return chosen, details


def _t1_refined_challenger_candidate(
    current: dict[str, Any],
    refined_candidates: list[dict[str, Any]],
    *,
    bust_max: float,
    refined_delta_max: float,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if float(bust_max) < 0.0 or float(refined_delta_max) < 0.0 or not refined_candidates:
        return current, None
    challenger = max(
        refined_candidates,
        key=lambda item: (
            float(item.get("refined_score", float("-inf"))),
            float(item.get("model_score", float("-inf"))),
            -int(item.get("model_rank", 999)),
        ),
    )
    same_action = int(current.get("action_idx", -1)) == int(challenger.get("action_idx", -2))
    refined_score_delta = float(challenger.get("refined_score", 0.0) or 0.0) - float(
        current.get("refined_score", 0.0) or 0.0
    )
    challenger_predicted_bust = float(challenger.get("predicted_bust", 0.0) or 0.0)
    current_predicted_bust = float(current.get("predicted_bust", 0.0) or 0.0)
    accepted = (
        not same_action
        and challenger_predicted_bust <= float(bust_max)
        and refined_score_delta <= float(refined_delta_max)
    )
    chosen = challenger if accepted else current
    details = {
        "policy": "refined_score_low_bust_close_delta",
        "accepted": bool(accepted),
        "current_action_idx": int(current.get("action_idx", -1)),
        "challenger_action_idx": int(challenger.get("action_idx", -1)),
        "chosen_action_idx": int(chosen.get("action_idx", -1)),
        "same_action": bool(same_action),
        "current_refined_score": float(current.get("refined_score", 0.0) or 0.0),
        "challenger_refined_score": float(challenger.get("refined_score", 0.0) or 0.0),
        "refined_score_delta": float(refined_score_delta),
        "refined_delta_max": float(refined_delta_max),
        "current_predicted_bust": float(current_predicted_bust),
        "challenger_predicted_bust": float(challenger_predicted_bust),
        "bust_max": float(bust_max),
    }
    return chosen, details


def _premium_fl_sum(candidate: dict[str, Any]) -> float:
    return (
        _candidate_fl_type(candidate, "aa")
        + _candidate_fl_type(candidate, "kk")
        + _candidate_fl_type(candidate, "trips")
    )


def _t1_blend_challenger_candidate(
    current: dict[str, Any],
    refined_candidates: list[dict[str, Any]],
    *,
    weight: float,
    bust_min: float,
    premium_fl_delta_min: float,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if (
        float(weight) < 0.0
        or float(bust_min) < 0.0
        or float(premium_fl_delta_min) < 0.0
        or not refined_candidates
    ):
        return current, None
    challenger = max(
        refined_candidates,
        key=lambda item: (
            float(item.get("refined_score", float("-inf")))
            + float(weight) * float(item.get("model_score", float("-inf"))),
            float(item.get("refined_score", float("-inf"))),
            float(item.get("model_score", float("-inf"))),
            -int(item.get("model_rank", 999)),
        ),
    )
    same_action = int(current.get("action_idx", -1)) == int(challenger.get("action_idx", -2))
    current_premium_fl = _premium_fl_sum(current)
    challenger_premium_fl = _premium_fl_sum(challenger)
    premium_fl_delta = challenger_premium_fl - current_premium_fl
    challenger_predicted_bust = float(challenger.get("predicted_bust", 0.0) or 0.0)
    current_predicted_bust = float(current.get("predicted_bust", 0.0) or 0.0)
    current_blend_score = float(current.get("refined_score", 0.0) or 0.0) + float(weight) * float(
        current.get("model_score", 0.0) or 0.0
    )
    challenger_blend_score = float(challenger.get("refined_score", 0.0) or 0.0) + float(weight) * float(
        challenger.get("model_score", 0.0) or 0.0
    )
    accepted = (
        not same_action
        and challenger_predicted_bust >= float(bust_min)
        and premium_fl_delta >= float(premium_fl_delta_min)
    )
    chosen = challenger if accepted else current
    details = {
        "policy": "blend_high_bust_premium_fl",
        "accepted": bool(accepted),
        "current_action_idx": int(current.get("action_idx", -1)),
        "challenger_action_idx": int(challenger.get("action_idx", -1)),
        "chosen_action_idx": int(chosen.get("action_idx", -1)),
        "same_action": bool(same_action),
        "weight": float(weight),
        "current_blend_score": float(current_blend_score),
        "challenger_blend_score": float(challenger_blend_score),
        "blend_score_delta": float(challenger_blend_score - current_blend_score),
        "current_refined_score": float(current.get("refined_score", 0.0) or 0.0),
        "challenger_refined_score": float(challenger.get("refined_score", 0.0) or 0.0),
        "current_model_score": float(current.get("model_score", 0.0) or 0.0),
        "challenger_model_score": float(challenger.get("model_score", 0.0) or 0.0),
        "current_predicted_bust": float(current_predicted_bust),
        "challenger_predicted_bust": float(challenger_predicted_bust),
        "bust_min": float(bust_min),
        "current_premium_fl": float(current_premium_fl),
        "challenger_premium_fl": float(challenger_premium_fl),
        "premium_fl_delta": float(premium_fl_delta),
        "premium_fl_delta_min": float(premium_fl_delta_min),
    }
    return chosen, details


def _candidate_fl_max(candidate: dict[str, Any]) -> float:
    return max(
        _candidate_fl_type(candidate, "fl"),
        _candidate_fl_type(candidate, "aa"),
        _candidate_fl_type(candidate, "kk"),
        _candidate_fl_type(candidate, "qq"),
        _candidate_fl_type(candidate, "trips"),
    )


def _t1_low_risk_blend_challenger_candidate(
    current: dict[str, Any],
    refined_candidates: list[dict[str, Any]],
    *,
    weight: float,
    fl_max: float,
    bust_max: float,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if float(weight) < 0.0 or float(fl_max) < 0.0 or float(bust_max) < 0.0 or not refined_candidates:
        return current, None
    challenger = max(
        refined_candidates,
        key=lambda item: (
            float(item.get("refined_score", float("-inf")))
            + float(weight) * float(item.get("model_score", float("-inf"))),
            float(item.get("refined_score", float("-inf"))),
            float(item.get("model_score", float("-inf"))),
            -int(item.get("model_rank", 999)),
        ),
    )
    same_action = int(current.get("action_idx", -1)) == int(challenger.get("action_idx", -2))
    challenger_predicted_fl = float(challenger.get("predicted_fl", 0.0) or 0.0)
    challenger_predicted_bust = float(challenger.get("predicted_bust", 0.0) or 0.0)
    accepted = (
        not same_action
        and challenger_predicted_fl <= float(fl_max)
        and challenger_predicted_bust <= float(bust_max)
    )
    chosen = challenger if accepted else current
    details = {
        "policy": "low_risk_blend",
        "accepted": bool(accepted),
        "current_action_idx": int(current.get("action_idx", -1)),
        "challenger_action_idx": int(challenger.get("action_idx", -1)),
        "chosen_action_idx": int(chosen.get("action_idx", -1)),
        "same_action": bool(same_action),
        "weight": float(weight),
        "current_refined_score": float(current.get("refined_score", 0.0) or 0.0),
        "challenger_refined_score": float(challenger.get("refined_score", 0.0) or 0.0),
        "current_model_score": float(current.get("model_score", 0.0) or 0.0),
        "challenger_model_score": float(challenger.get("model_score", 0.0) or 0.0),
        "challenger_predicted_fl": float(challenger_predicted_fl),
        "fl_max": float(fl_max),
        "challenger_predicted_bust": float(challenger_predicted_bust),
        "bust_max": float(bust_max),
    }
    return chosen, details


def _t1_model_rescue_candidate(
    current: dict[str, Any],
    model_top1: dict[str, Any] | None,
    *,
    qq_delta_min: float,
    candidate_fl_delta_min: float,
    premium_delta_min: float = -1.0,
    model_score_min: float = -1.0,
    model_bust_max: float = -1.0,
    refined_delta_max: float = -1.0,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if (
        model_top1 is None
        or (float(qq_delta_min) < 0.0 and float(premium_delta_min) < 0.0)
        or float(candidate_fl_delta_min) < 0.0
    ):
        return current, None
    same_action = int(current.get("action_idx", -1)) == int(model_top1.get("action_idx", -2))
    predicted_qq_delta = _candidate_fl_type(model_top1, "qq") - _candidate_fl_type(current, "qq")
    premium_delta = _premium_fl_sum(model_top1) - _premium_fl_sum(current)
    candidate_fl_delta = _candidate_fl_max(model_top1) - _candidate_fl_max(current)
    model_score = float(model_top1.get("model_score", 0.0) or 0.0)
    model_bust = float(model_top1.get("predicted_bust", 0.0) or 0.0)
    current_refined_score = current.get("refined_score")
    model_refined_score = model_top1.get("refined_score")
    refined_delta = None
    if current_refined_score is not None and model_refined_score is not None:
        refined_delta = float(current_refined_score) - float(model_refined_score)
    qq_condition = float(qq_delta_min) >= 0.0 and predicted_qq_delta >= float(qq_delta_min)
    premium_condition = float(premium_delta_min) >= 0.0 and premium_delta >= float(premium_delta_min)
    model_score_condition = float(model_score_min) < 0.0 or model_score >= float(model_score_min)
    model_bust_condition = float(model_bust_max) < 0.0 or model_bust <= float(model_bust_max)
    refined_delta_condition = (
        float(refined_delta_max) < 0.0
        or refined_delta is None
        or refined_delta <= float(refined_delta_max)
    )
    accepted = (
        not same_action
        and (qq_condition or premium_condition)
        and candidate_fl_delta >= float(candidate_fl_delta_min)
        and model_score_condition
        and model_bust_condition
        and refined_delta_condition
    )
    chosen = model_top1 if accepted else current
    details = {
        "policy": "model_top1_fl_qq_rescue",
        "accepted": bool(accepted),
        "current_action_idx": int(current.get("action_idx", -1)),
        "challenger_action_idx": int(model_top1.get("action_idx", -1)),
        "chosen_action_idx": int(chosen.get("action_idx", -1)),
        "same_action": bool(same_action),
        "predicted_qq_delta": float(predicted_qq_delta),
        "qq_delta_min": float(qq_delta_min),
        "premium_delta": float(premium_delta),
        "premium_delta_min": float(premium_delta_min),
        "qq_condition": bool(qq_condition),
        "premium_condition": bool(premium_condition),
        "candidate_fl_delta": float(candidate_fl_delta),
        "candidate_fl_delta_min": float(candidate_fl_delta_min),
        "model_score": float(model_score),
        "model_score_min": float(model_score_min),
        "model_score_condition": bool(model_score_condition),
        "model_bust": float(model_bust),
        "model_bust_max": float(model_bust_max),
        "model_bust_condition": bool(model_bust_condition),
        "refined_delta": float(refined_delta) if refined_delta is not None else None,
        "refined_delta_max": float(refined_delta_max),
        "refined_delta_condition": bool(refined_delta_condition),
        "current_candidate_fl": float(_candidate_fl_max(current)),
        "challenger_candidate_fl": float(_candidate_fl_max(model_top1)),
        "current_premium_fl": float(_premium_fl_sum(current)),
        "challenger_premium_fl": float(_premium_fl_sum(model_top1)),
    }
    return chosen, details


def _t2_model_top1_rescue_candidate(
    current: dict[str, Any],
    model_top1: dict[str, Any] | None,
    *,
    fl_min: float,
    refined_delta_max: float,
    bust_max: float = -1.0,
    selected_model_rank_min: int = 0,
    secondary_fl_min: float = -1.0,
    secondary_fl_max: float = -1.0,
    secondary_bust_max: float = -1.0,
    secondary_selected_model_rank_min: int = 0,
    secondary_refined_delta_max: float = -1.0,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    primary_enabled = float(fl_min) >= 0.0 and float(refined_delta_max) >= 0.0
    secondary_enabled = (
        float(secondary_fl_min) >= 0.0
        and float(secondary_fl_max) >= 0.0
        and float(secondary_bust_max) >= 0.0
        and int(secondary_selected_model_rank_min) > 0
        and float(secondary_refined_delta_max) >= 0.0
    )
    if model_top1 is None or not (primary_enabled or secondary_enabled):
        return current, None
    same_action = int(current.get("action_idx", -1)) == int(model_top1.get("action_idx", -2))
    current_refined_score = current.get("refined_score")
    model_refined_score = model_top1.get("refined_score")
    refined_delta = None
    if current_refined_score is not None and model_refined_score is not None:
        refined_delta = float(current_refined_score) - float(model_refined_score)
    predicted_fl = _candidate_fl_max(model_top1)
    predicted_bust = float(model_top1.get("predicted_bust", 0.0) or 0.0)
    current_model_rank = int(current.get("model_rank", 999999) or 999999)
    fl_condition = primary_enabled and predicted_fl >= float(fl_min)
    bust_condition = primary_enabled and (
        float(bust_max) < 0.0 or predicted_bust <= float(bust_max)
    )
    primary_rank_condition = primary_enabled and current_model_rank >= int(selected_model_rank_min)
    refined_delta_condition = (
        primary_enabled and refined_delta is not None and refined_delta <= float(refined_delta_max)
    )
    primary_accepted = (
        not same_action
        and fl_condition
        and bust_condition
        and primary_rank_condition
        and refined_delta_condition
    )
    secondary_fl_condition = (
        secondary_enabled
        and predicted_fl >= float(secondary_fl_min)
        and predicted_fl <= float(secondary_fl_max)
    )
    secondary_bust_condition = secondary_enabled and predicted_bust <= float(secondary_bust_max)
    secondary_rank_condition = (
        secondary_enabled and current_model_rank >= int(secondary_selected_model_rank_min)
    )
    secondary_delta_condition = (
        secondary_enabled
        and refined_delta is not None
        and refined_delta <= float(secondary_refined_delta_max)
    )
    secondary_accepted = (
        not same_action
        and secondary_fl_condition
        and secondary_bust_condition
        and secondary_rank_condition
        and secondary_delta_condition
    )
    accepted = primary_accepted or secondary_accepted
    chosen = model_top1 if accepted else current
    details = {
        "policy": "t2_model_top1_rescue",
        "accepted": bool(accepted),
        "accepted_gate": "primary" if primary_accepted else ("secondary" if secondary_accepted else None),
        "current_action_idx": int(current.get("action_idx", -1)),
        "challenger_action_idx": int(model_top1.get("action_idx", -1)),
        "chosen_action_idx": int(chosen.get("action_idx", -1)),
        "same_action": bool(same_action),
        "challenger_predicted_fl": float(predicted_fl),
        "challenger_predicted_bust": float(predicted_bust),
        "current_model_rank": int(current_model_rank),
        "fl_min": float(fl_min),
        "fl_condition": bool(fl_condition),
        "bust_max": float(bust_max),
        "bust_condition": bool(bust_condition),
        "selected_model_rank_min": int(selected_model_rank_min),
        "primary_rank_condition": bool(primary_rank_condition),
        "current_refined_score": float(current_refined_score) if current_refined_score is not None else None,
        "challenger_refined_score": float(model_refined_score) if model_refined_score is not None else None,
        "refined_delta": float(refined_delta) if refined_delta is not None else None,
        "refined_delta_max": float(refined_delta_max),
        "refined_delta_condition": bool(refined_delta_condition),
        "secondary_enabled": bool(secondary_enabled),
        "secondary_fl_min": float(secondary_fl_min),
        "secondary_fl_max": float(secondary_fl_max),
        "secondary_fl_condition": bool(secondary_fl_condition),
        "secondary_bust_max": float(secondary_bust_max),
        "secondary_bust_condition": bool(secondary_bust_condition),
        "secondary_selected_model_rank_min": int(secondary_selected_model_rank_min),
        "secondary_rank_condition": bool(secondary_rank_condition),
        "secondary_refined_delta_max": float(secondary_refined_delta_max),
        "secondary_delta_condition": bool(secondary_delta_condition),
    }
    return chosen, details


def _t2_model_top1_bust_rescue_candidate(
    current: dict[str, Any],
    model_top1: dict[str, Any] | None,
    *,
    selected_model_rank_min: int,
    refined_delta_max: float,
    model_delta_min: float,
    bust_delta_min: float,
    top_bust_max: float,
    top_fl_min: float,
    current_fl_min: float,
    fl_delta_min: float,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    enabled = (
        int(selected_model_rank_min) > 0
        and float(refined_delta_max) >= 0.0
        and float(model_delta_min) >= 0.0
        and float(bust_delta_min) >= 0.0
        and float(top_bust_max) >= 0.0
        and float(top_fl_min) >= 0.0
        and float(current_fl_min) >= 0.0
        and float(fl_delta_min) >= 0.0
    )
    if model_top1 is None or not enabled:
        return current, None

    same_action = int(current.get("action_idx", -1)) == int(model_top1.get("action_idx", -2))
    current_refined_score = current.get("refined_score")
    model_refined_score = model_top1.get("refined_score")
    refined_delta = None
    if current_refined_score is not None and model_refined_score is not None:
        refined_delta = float(current_refined_score) - float(model_refined_score)

    current_model_rank = int(current.get("model_rank", 999999) or 999999)
    current_model_score = float(current.get("model_score", 0.0) or 0.0)
    top_model_score = float(model_top1.get("model_score", 0.0) or 0.0)
    model_delta = top_model_score - current_model_score
    current_bust = float(current.get("predicted_bust", 0.0) or 0.0)
    top_bust = float(model_top1.get("predicted_bust", 0.0) or 0.0)
    bust_delta = current_bust - top_bust
    current_fl = _candidate_fl_max(current)
    top_fl = _candidate_fl_max(model_top1)
    fl_delta = top_fl - current_fl

    rank_condition = current_model_rank >= int(selected_model_rank_min)
    refined_delta_condition = refined_delta is not None and refined_delta <= float(refined_delta_max)
    model_delta_condition = model_delta >= float(model_delta_min)
    bust_delta_condition = bust_delta >= float(bust_delta_min)
    top_bust_condition = top_bust <= float(top_bust_max)
    top_fl_condition = top_fl >= float(top_fl_min)
    current_fl_condition = current_fl >= float(current_fl_min)
    fl_delta_condition = fl_delta >= float(fl_delta_min)
    accepted = (
        not same_action
        and rank_condition
        and refined_delta_condition
        and model_delta_condition
        and bust_delta_condition
        and top_bust_condition
        and top_fl_condition
        and current_fl_condition
        and fl_delta_condition
    )
    chosen = model_top1 if accepted else current
    details = {
        "policy": "t2_model_top1_bust_rescue",
        "accepted": bool(accepted),
        "current_action_idx": int(current.get("action_idx", -1)),
        "challenger_action_idx": int(model_top1.get("action_idx", -1)),
        "chosen_action_idx": int(chosen.get("action_idx", -1)),
        "same_action": bool(same_action),
        "current_model_rank": int(current_model_rank),
        "challenger_model_rank": int(model_top1.get("model_rank", -1)),
        "selected_model_rank_min": int(selected_model_rank_min),
        "rank_condition": bool(rank_condition),
        "current_refined_score": float(current_refined_score) if current_refined_score is not None else None,
        "challenger_refined_score": float(model_refined_score) if model_refined_score is not None else None,
        "refined_delta": float(refined_delta) if refined_delta is not None else None,
        "refined_delta_max": float(refined_delta_max),
        "refined_delta_condition": bool(refined_delta_condition),
        "current_model_score": float(current_model_score),
        "challenger_model_score": float(top_model_score),
        "model_delta": float(model_delta),
        "model_delta_min": float(model_delta_min),
        "model_delta_condition": bool(model_delta_condition),
        "current_predicted_bust": float(current_bust),
        "challenger_predicted_bust": float(top_bust),
        "bust_delta": float(bust_delta),
        "bust_delta_min": float(bust_delta_min),
        "bust_delta_condition": bool(bust_delta_condition),
        "top_bust_max": float(top_bust_max),
        "top_bust_condition": bool(top_bust_condition),
        "current_candidate_fl": float(current_fl),
        "challenger_candidate_fl": float(top_fl),
        "top_fl_min": float(top_fl_min),
        "top_fl_condition": bool(top_fl_condition),
        "current_fl_min": float(current_fl_min),
        "current_fl_condition": bool(current_fl_condition),
        "fl_delta": float(fl_delta),
        "fl_delta_min": float(fl_delta_min),
        "fl_delta_condition": bool(fl_delta_condition),
    }
    return chosen, details


def _action_cards_to_row(action: dict[str, Any], row: str) -> set[str]:
    placements = action.get("placements") if isinstance(action, dict) else None
    if not isinstance(placements, list):
        return set()
    return {
        str(card)
        for card, target_row in placements
        if isinstance(card, str) and str(target_row) == row
    }


def _t2_middle_fill_bottom_shift_structure(
    obs: Observation,
    current: dict[str, Any],
    challenger: dict[str, Any],
) -> tuple[bool, str | None]:
    current_action = current.get("action") if isinstance(current.get("action"), dict) else {}
    challenger_action = challenger.get("action") if isinstance(challenger.get("action"), dict) else {}
    if current_action.get("discard") != challenger_action.get("discard"):
        return False, None
    if len(obs.board_self.middle) != 4 or len(obs.board_self.bottom) > 2:
        return False, None

    current_middle = _action_cards_to_row(current_action, "middle")
    challenger_bottom = _action_cards_to_row(challenger_action, "bottom")
    moved_cards = current_middle & challenger_bottom
    if len(moved_cards) != 1:
        return False, None
    moved_card = next(iter(moved_cards))

    if _action_cards_to_row(current_action, "top") != _action_cards_to_row(challenger_action, "top"):
        return False, None
    if not _action_cards_to_row(current_action, "top"):
        return False, None

    current_other = {
        (str(card), str(row))
        for card, row in current_action.get("placements", [])
        if str(card) != moved_card
    }
    challenger_other = {
        (str(card), str(row))
        for card, row in challenger_action.get("placements", [])
        if str(card) != moved_card
    }
    if current_other != challenger_other:
        return False, None
    return True, moved_card


def _t2_middle_fill_bottom_shift_rescue_candidate(
    obs: Observation,
    current: dict[str, Any],
    refined_candidates: list[dict[str, Any]],
    *,
    selected_model_rank_max: int,
    challenger_model_rank_max: int,
    model_gap_max: float,
    bust_delta_min: float,
    fl_delta_min: float,
    challenger_bust_max: float,
    challenger_fl_min: float,
    selected_bust_min: float,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    enabled = (
        int(selected_model_rank_max) > 0
        and int(challenger_model_rank_max) > 0
        and float(model_gap_max) >= 0.0
        and float(bust_delta_min) >= 0.0
        and float(fl_delta_min) >= 0.0
        and float(challenger_bust_max) >= 0.0
        and float(challenger_fl_min) >= 0.0
        and float(selected_bust_min) >= 0.0
    )
    if not enabled:
        return current, None

    current_rank = int(current.get("model_rank", 999999) or 999999)
    current_model = float(current.get("model_score", 0.0) or 0.0)
    current_bust = float(current.get("predicted_bust", 0.0) or 0.0)
    current_fl = _candidate_fl_max(current)
    if current_rank > int(selected_model_rank_max) or current_bust < float(selected_bust_min):
        return current, {
            "policy": "t2_middle_fill_bottom_shift_rescue",
            "accepted": False,
            "reject_reason": "selected_rank_or_bust",
            "current_action_idx": int(current.get("action_idx", -1)),
            "current_model_rank": int(current_rank),
            "selected_model_rank_max": int(selected_model_rank_max),
            "current_predicted_bust": float(current_bust),
            "selected_bust_min": float(selected_bust_min),
        }

    challengers: list[tuple[float, float, float, int, str, dict[str, Any]]] = []
    inspected_count = 0
    structural_count = 0
    for candidate in refined_candidates:
        if int(candidate.get("action_idx", -1)) == int(current.get("action_idx", -2)):
            continue
        candidate_rank = int(candidate.get("model_rank", 999999) or 999999)
        if candidate_rank <= current_rank or candidate_rank > int(challenger_model_rank_max):
            continue
        inspected_count += 1
        structural_ok, moved_card = _t2_middle_fill_bottom_shift_structure(obs, current, candidate)
        if not structural_ok or moved_card is None:
            continue
        structural_count += 1
        candidate_model = float(candidate.get("model_score", 0.0) or 0.0)
        model_gap = current_model - candidate_model
        candidate_bust = float(candidate.get("predicted_bust", 0.0) or 0.0)
        bust_delta = current_bust - candidate_bust
        candidate_fl = _candidate_fl_max(candidate)
        fl_delta = candidate_fl - current_fl
        if model_gap > float(model_gap_max):
            continue
        if bust_delta < float(bust_delta_min):
            continue
        if fl_delta < float(fl_delta_min):
            continue
        if candidate_bust > float(challenger_bust_max):
            continue
        if candidate_fl < float(challenger_fl_min):
            continue
        rescue_score = fl_delta + bust_delta - max(0.0, model_gap) * 0.01
        challengers.append((rescue_score, -model_gap, -candidate_rank, int(candidate["action_idx"]), moved_card, candidate))

    challengers.sort(reverse=True, key=lambda item: item[:4])
    challenger = challengers[0][5] if challengers else None
    accepted = challenger is not None
    chosen = challenger if challenger is not None else current
    moved_card = challengers[0][4] if challengers else None
    challenger_model = float(challenger.get("model_score", 0.0) or 0.0) if challenger is not None else None
    challenger_bust = float(challenger.get("predicted_bust", 0.0) or 0.0) if challenger is not None else None
    challenger_fl = _candidate_fl_max(challenger) if challenger is not None else None
    details = {
        "policy": "t2_middle_fill_bottom_shift_rescue",
        "accepted": bool(accepted),
        "reject_reason": None if accepted else "no_matching_challenger",
        "current_action_idx": int(current.get("action_idx", -1)),
        "challenger_action_idx": int(challenger.get("action_idx", -1)) if challenger is not None else None,
        "chosen_action_idx": int(chosen.get("action_idx", -1)),
        "moved_card": moved_card,
        "current_model_rank": int(current_rank),
        "challenger_model_rank": int(challenger.get("model_rank", -1)) if challenger is not None else None,
        "selected_model_rank_max": int(selected_model_rank_max),
        "challenger_model_rank_max": int(challenger_model_rank_max),
        "current_model_score": float(current_model),
        "challenger_model_score": float(challenger_model) if challenger_model is not None else None,
        "model_gap": (
            float(current_model) - float(challenger_model)
            if challenger_model is not None
            else None
        ),
        "model_gap_max": float(model_gap_max),
        "current_predicted_bust": float(current_bust),
        "challenger_predicted_bust": float(challenger_bust) if challenger_bust is not None else None,
        "bust_delta": (
            float(current_bust) - float(challenger_bust)
            if challenger_bust is not None
            else None
        ),
        "bust_delta_min": float(bust_delta_min),
        "challenger_bust_max": float(challenger_bust_max),
        "selected_bust_min": float(selected_bust_min),
        "current_candidate_fl": float(current_fl),
        "challenger_candidate_fl": float(challenger_fl) if challenger_fl is not None else None,
        "fl_delta": (
            float(challenger_fl) - float(current_fl)
            if challenger_fl is not None
            else None
        ),
        "fl_delta_min": float(fl_delta_min),
        "challenger_fl_min": float(challenger_fl_min),
        "inspected_count": int(inspected_count),
        "structural_count": int(structural_count),
        "candidate_count": int(len(challengers)),
    }
    return chosen, details


def _t2_model_rank_rescue_candidate(
    current: dict[str, Any],
    refined_candidates: list[dict[str, Any]],
    *,
    rank_k: int,
    selected_model_rank_min: int,
    refined_delta_max: float,
    model_delta_min: float,
    min_refined_score: float = -1.0,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    enabled = (
        int(rank_k) > 0
        and int(selected_model_rank_min) > 0
        and float(refined_delta_max) >= 0.0
        and float(model_delta_min) >= 0.0
    )
    if not enabled:
        return current, None

    current_rank = int(current.get("model_rank", 999999) or 999999)
    current_refined = current.get("refined_score")
    current_model = float(current.get("model_score", 0.0) or 0.0)
    if current_refined is None or current_rank < int(selected_model_rank_min):
        return current, {
            "policy": "t2_model_rank_rescue",
            "accepted": False,
            "reject_reason": "current_rank_or_refined",
            "current_action_idx": int(current.get("action_idx", -1)),
            "current_model_rank": int(current_rank),
            "selected_model_rank_min": int(selected_model_rank_min),
        }

    challengers: list[tuple[float, float, float, int, dict[str, Any]]] = []
    for candidate in refined_candidates:
        if int(candidate.get("action_idx", -1)) == int(current.get("action_idx", -2)):
            continue
        model_rank = int(candidate.get("model_rank", 999999) or 999999)
        if model_rank > int(rank_k):
            continue
        candidate_refined = candidate.get("refined_score")
        if candidate_refined is None:
            continue
        candidate_refined_f = float(candidate_refined)
        refined_delta = float(current_refined) - candidate_refined_f
        model_delta = float(candidate.get("model_score", 0.0) or 0.0) - current_model
        if float(min_refined_score) >= 0.0 and candidate_refined_f < float(min_refined_score):
            continue
        if refined_delta > float(refined_delta_max):
            continue
        if model_delta < float(model_delta_min):
            continue
        rescue_score = model_delta - max(0.0, refined_delta)
        challengers.append((rescue_score, model_delta, -refined_delta, -model_rank, candidate))

    challengers.sort(reverse=True, key=lambda item: item[:4])
    challenger = challengers[0][4] if challengers else None
    accepted = challenger is not None
    chosen = challenger if challenger is not None else current
    challenger_refined = challenger.get("refined_score") if challenger is not None else None
    challenger_model = float(challenger.get("model_score", 0.0) or 0.0) if challenger is not None else None
    details = {
        "policy": "t2_model_rank_rescue",
        "accepted": bool(accepted),
        "reject_reason": None if accepted else "no_matching_challenger",
        "current_action_idx": int(current.get("action_idx", -1)),
        "challenger_action_idx": int(challenger.get("action_idx", -1)) if challenger is not None else None,
        "chosen_action_idx": int(chosen.get("action_idx", -1)),
        "rank_k": int(rank_k),
        "current_model_rank": int(current_rank),
        "challenger_model_rank": int(challenger.get("model_rank", -1)) if challenger is not None else None,
        "selected_model_rank_min": int(selected_model_rank_min),
        "current_refined_score": float(current_refined),
        "challenger_refined_score": float(challenger_refined) if challenger_refined is not None else None,
        "refined_delta": (
            float(current_refined) - float(challenger_refined)
            if challenger_refined is not None
            else None
        ),
        "refined_delta_max": float(refined_delta_max),
        "current_model_score": float(current_model),
        "challenger_model_score": float(challenger_model) if challenger_model is not None else None,
        "model_delta": (
            float(challenger_model) - float(current_model)
            if challenger_model is not None
            else None
        ),
        "model_delta_min": float(model_delta_min),
        "min_refined_score": float(min_refined_score),
        "candidate_count": int(len(challengers)),
    }
    return chosen, details


def _add_unique(pool: list[dict[str, Any]], candidate: dict[str, Any], seen: set[int], limit: int) -> None:
    if len(pool) >= limit:
        return
    action_idx = int(candidate["action_idx"])
    if action_idx in seen:
        return
    pool.append(candidate)
    seen.add(action_idx)


def build_shortlist(
    candidates: list[dict[str, Any]],
    *,
    config: HybridConfig = HybridConfig(),
) -> list[dict[str, Any]]:
    ordered = sorted(candidates, key=lambda item: float(item["model_score"]), reverse=True)
    for rank, candidate in enumerate(ordered, start=1):
        candidate["model_rank"] = rank
        candidate["refinement_rank"] = rank

    turn = int(next((item.get("turn") for item in candidates if item.get("turn") is not None), 1) or 1)
    aux_shortlist_k = int(config.t2_aux_shortlist_k if turn == 2 else config.t1_aux_shortlist_k)
    aux_ordered: list[dict[str, Any]] = []
    if aux_shortlist_k > 0 and any("aux_model_score" in item for item in candidates):
        aux_ordered = sorted(
            [item for item in candidates if "aux_model_score" in item],
            key=lambda item: float(item["aux_model_score"]),
            reverse=True,
        )
        for rank, candidate in enumerate(aux_ordered, start=1):
            candidate["aux_model_rank"] = rank
            candidate["refinement_rank"] = min(int(candidate["refinement_rank"]), rank)

    pool: list[dict[str, Any]] = []
    seen: set[int] = set()
    for candidate in ordered[: config.shortlist_k]:
        _add_unique(pool, candidate, seen, config.max_pool)

    for candidate in aux_ordered[:aux_shortlist_k]:
        before = len(pool)
        _add_unique(pool, candidate, seen, config.max_pool)
        if len(pool) > before:
            candidate["insurance_reason"] = "aux_shortlist"

    insurance_budget = max(0, config.insurance_k)
    if any(int(item.get("turn", 0) or 0) == 2 for item in ordered):
        for candidate in _t2_tactical_insurance_candidates(ordered, config, limit=insurance_budget):
            before = len(pool)
            _add_unique(pool, candidate, seen, config.max_pool)
            if len(pool) > before:
                insurance_budget -= 1

    bust_slots = min(3, insurance_budget)
    fl_slots = max(0, insurance_budget - bust_slots)

    for candidate in sorted(ordered, key=lambda item: (float(item["predicted_bust"]), -float(item["model_score"]))):
        if bust_slots <= 0:
            break
        before = len(pool)
        _add_unique(pool, candidate, seen, config.max_pool)
        if len(pool) > before:
            candidate["insurance_reason"] = "low_bust"
            bust_slots -= 1

    for candidate in sorted(ordered, key=lambda item: (_candidate_fl_score(item), float(item["model_score"])), reverse=True):
        if fl_slots <= 0:
            break
        before = len(pool)
        _add_unique(pool, candidate, seen, config.max_pool)
        if len(pool) > before:
            candidate["insurance_reason"] = "fl"
            fl_slots -= 1

    return pool[: config.max_pool]


def _should_expand_t2_sync_candidates(
    candidates: list[dict[str, Any]],
    config: HybridConfig,
    *,
    turn: int,
) -> bool:
    if int(turn) != 2:
        return False
    if int(config.t2_adaptive_shortlist_k) <= int(config.shortlist_k):
        return False
    if int(config.t2_adaptive_sync_exact_k) <= max(int(config.sync_exact_k), int(config.t2_sync_exact_k)):
        return False
    if config.t2_adaptive_max_model_score is None:
        return False
    if not candidates:
        return False
    top_model_score = max(float(item["model_score"]) for item in candidates)
    return top_model_score <= float(config.t2_adaptive_max_model_score)


def _sync_refinement_candidates(
    pool: list[dict[str, Any]],
    config: HybridConfig,
    *,
    turn: int,
) -> list[dict[str, Any]]:
    def t2_model_insurance_enabled() -> bool:
        if int(turn) != 2 or int(config.t2_sync_model_insurance_k) <= 0:
            return False
        kk_min = float(config.t2_sync_model_insurance_top1_kk_min)
        if kk_min <= 0.0:
            return True
        if not ranked:
            return False
        model_top = min(ranked, key=lambda candidate: int(candidate.get("model_rank", 999999)))
        fl_types = model_top.get("predicted_fl_types") or {}
        try:
            predicted_kk = float(fl_types.get("kk", 0.0) or 0.0)
        except (TypeError, ValueError):
            predicted_kk = 0.0
        return predicted_kk >= kk_min

    def refinement_rank(item: dict[str, Any]) -> tuple[float, int, int]:
        if (
            (int(turn) == 1 and config.t1_sync_selection_policy == "selector")
            or (int(turn) == 2 and config.t2_sync_selection_policy == "selector")
        ) and "sync_selector_score" in item:
            return (
                -float(item.get("sync_selector_score", float("-inf"))),
                int(item.get("refinement_rank", item["model_rank"])),
                int(item["model_rank"]),
            )
        return (
            float(item.get("refinement_rank", item["model_rank"])),
            int(item.get("refinement_rank", item["model_rank"])),
            int(item["model_rank"]),
        )

    ranked = sorted(pool, key=refinement_rank)
    non_forced_ranked = [item for item in ranked if not bool(item.get("forced_bust", False))]
    if int(turn) == 2 and non_forced_ranked:
        ranked = non_forced_ranked
    sync_exact_k = max(0, int(config.sync_exact_k))
    if (
        int(turn) == 1
        and int(config.t1_adaptive_sync_exact_k) > sync_exact_k
        and int(config.t1_adaptive_model_rank_k) > 0
    ):
        base_action_indices = {int(item["action_idx"]) for item in ranked[:sync_exact_k]}
        needs_adaptive = any(
            int(item.get("model_rank", 999)) <= int(config.t1_adaptive_model_rank_k)
            and int(item["action_idx"]) not in base_action_indices
            for item in ranked
        )
        if needs_adaptive:
            sync_exact_k = min(len(ranked), int(config.t1_adaptive_sync_exact_k))

    if int(turn) == 1 and int(config.t1_sync_model_insurance_k) > 0 and sync_exact_k > 0:
        selected: list[dict[str, Any]] = []
        seen: set[int] = set()
        for item in sorted(ranked, key=lambda candidate: int(candidate.get("model_rank", 999))):
            if len(selected) >= min(int(config.t1_sync_model_insurance_k), sync_exact_k):
                break
            action_idx = int(item["action_idx"])
            if action_idx not in seen:
                selected.append(item)
                seen.add(action_idx)
        for item in ranked:
            if len(selected) >= sync_exact_k:
                break
            action_idx = int(item["action_idx"])
            if action_idx not in seen:
                selected.append(item)
                seen.add(action_idx)
        return _inject_sync_insurance(
            selected,
            _sync_insurance_candidates(ranked, config, turn=int(turn), limit=sync_exact_k),
            limit=sync_exact_k,
        )

    if t2_model_insurance_enabled() and sync_exact_k > 0:
        model_selected: list[dict[str, Any]] = []
        seen: set[int] = set()
        for item in sorted(ranked, key=lambda candidate: int(candidate.get("model_rank", 999))):
            if len(model_selected) >= min(int(config.t2_sync_model_insurance_k), sync_exact_k):
                break
            action_idx = int(item["action_idx"])
            if action_idx not in seen:
                model_selected.append(item)
                seen.add(action_idx)
        ranked_fill: list[dict[str, Any]] = []
        for item in ranked:
            if len(model_selected) + len(ranked_fill) >= sync_exact_k:
                break
            action_idx = int(item["action_idx"])
            if action_idx not in seen:
                ranked_fill.append(item)
                seen.add(action_idx)

        insurance_limit = max(0, min(int(config.t2_tactical_insurance_k), int(config.insurance_k), sync_exact_k))
        insurance = _sync_insurance_candidates(ranked, config, turn=int(turn), limit=insurance_limit)
        limit = min(len(ranked), sync_exact_k + insurance_limit)
        out: list[dict[str, Any]] = []
        out_seen: set[int] = set()

        def add_candidate(item: dict[str, Any]) -> None:
            if len(out) >= limit:
                return
            action_idx = int(item["action_idx"])
            if action_idx in out_seen:
                return
            out.append(item)
            out_seen.add(action_idx)

        interleave_len = max(len(model_selected), len(insurance))
        for i in range(interleave_len):
            if i < len(model_selected):
                add_candidate(model_selected[i])
            if i < len(insurance):
                add_candidate(insurance[i])
        for item in ranked_fill:
            add_candidate(item)
        for item in insurance:
            add_candidate(item)
        for item in ranked:
            add_candidate(item)
        return out[:limit]

    top = ranked[:sync_exact_k]
    selected = list(top)

    if top and all(float(item["predicted_bust"]) >= config.high_bust_threshold for item in top) and ranked:
        safest = min(ranked, key=lambda item: float(item["predicted_bust"]))
        if int(safest["action_idx"]) not in {int(item["action_idx"]) for item in selected}:
            selected.append(safest)

    return _inject_sync_insurance(
        selected,
        _sync_insurance_candidates(ranked, config, turn=int(turn), limit=sync_exact_k),
        limit=min(len(selected), sync_exact_k + 1),
    )


def _remaining_deck_after_candidate(obs: Observation, candidate: dict[str, Any], board: Board) -> list[str]:
    action = candidate.get("_action")
    discard = getattr(action, "discard", None) if action is not None else None
    used = set(board.all_cards())
    used.update(obs.board_opponent.all_cards())
    used.update(obs.known_discards_self)
    if discard:
        used.add(str(discard))
    return [card for card in ALL_CARDS if card not in used]


def _board_completion_exists(
    board: Board,
    deck: list[str],
    *,
    max_middle_checks: int = 1000,
    max_final_checks: int = 200,
) -> bool:
    top_need = 3 - len(board.top)
    middle_need = 5 - len(board.middle)
    bottom_need = 5 - len(board.bottom)
    needs = (top_need, middle_need, bottom_need)
    if any(need < 0 for need in needs):
        return False
    total_need = sum(needs)
    if total_need == 0:
        return not bool(evaluate_board_with_joker_constraint(board.top, board.middle, board.bottom)["busted"])
    if total_need > 4 or len(deck) < total_need:
        return True

    # Fast runtime guard: only prove forced bust when bottom is already fixed.
    # If bottom is still open, there are too many constructive completions to
    # exhaust cheaply, so leave the candidate to the model/refinement path.
    if bottom_need != 0:
        return True

    bottom = list(board.bottom)
    bottom_value = evaluate_hand(bottom, 5)
    if (
        hand_category(bottom_value) == 0
        and not any(card in ("X1", "X2", "JK") for card in board.middle + bottom)
        and board.middle
    ):
        bottom_high = max(RANK_VALUES.get(str(card)[0], 0) for card in bottom)
        middle_ranks = [RANK_VALUES.get(str(card)[0], 0) for card in board.middle]
        if max(middle_ranks) > bottom_high or len(set(middle_ranks)) < len(middle_ranks):
            return False
    middle_checks = 0
    final_checks = 0
    for middle_extra in itertools.combinations(deck, middle_need):
        middle_checks += 1
        if middle_checks > max_middle_checks:
            return True
        middle = list(board.middle) + list(middle_extra)
        middle_has_joker = any(card in ("X1", "X2", "JK") for card in middle)
        if not middle_has_joker and evaluate_hand(middle, 5) > bottom_value:
            continue
        middle_extra_set = set(middle_extra)
        remaining = [card for card in deck if card not in middle_extra_set]
        for top_extra in itertools.combinations(remaining, top_need):
            final_checks += 1
            if final_checks > max_final_checks:
                return True
            top = list(board.top) + list(top_extra)
            if middle_has_joker or any(card in ("X1", "X2", "JK") for card in top + bottom):
                if not bool(evaluate_board_with_joker_constraint(top, middle, bottom)["busted"]):
                    return True
            else:
                middle_value = evaluate_hand(middle, 5)
                if evaluate_hand(top, 3) <= middle_value <= bottom_value:
                    return True
    return False


def annotate_t2_forced_bust_candidates(obs: Observation, candidates: list[dict[str, Any]]) -> None:
    """Mark T2 candidates that cannot complete into a legal board.

    This is deliberately conservative.  It only proves common forced-bust T2
    cases where the bottom row is already fixed and too weak for any legal
    middle/top completion.  Other candidates are left to model/refinement.
    """
    if int(obs.turn) != 2:
        return
    for candidate in candidates:
        board = board_from_payload(candidate.get("board") or {})
        deck = _remaining_deck_after_candidate(obs, candidate, board)
        forced = not _board_completion_exists(board, deck)
        candidate["forced_bust"] = bool(forced)
        if forced:
            candidate["forced_bust_reason"] = "no_legal_completion"


def _t1_extra_refinement_candidates(
    refined_candidates: list[dict[str, Any]],
    config: HybridConfig,
) -> list[dict[str, Any]]:
    top_k = max(0, int(config.t1_extra_refine_top_k))
    extra_sims = max(0, int(config.t1_extra_refine_sims))
    if top_k <= 0 or extra_sims <= 0:
        return []
    ranked = sorted(
        [item for item in refined_candidates if item.get("refined_score") is not None],
        key=lambda item: (
            float(item.get("refined_score", float("-inf"))),
            float(item.get("model_score", float("-inf"))),
            -int(item.get("model_rank", 999999)),
        ),
        reverse=True,
    )
    if not ranked:
        return []
    best_score = float(ranked[0].get("refined_score", 0.0) or 0.0)
    margin = max(0.0, float(config.t1_extra_refine_margin))
    selected: list[dict[str, Any]] = []
    for item in ranked:
        if len(selected) >= top_k:
            break
        refined_score = float(item.get("refined_score", float("-inf")))
        if best_score - refined_score <= margin:
            selected.append(item)
    return selected


def _remaining_deck_after_action(obs: Observation, action: Action) -> list[str]:
    board = apply_action(obs.board_self, action)
    used = set(board.all_cards())
    used.update(obs.board_opponent.all_cards())
    used.update(obs.known_discards_self)
    if action.discard:
        used.add(action.discard)
    return [card for card in ALL_CARDS if card not in used]


def _stable_seed(obs: Observation, action_idx: int) -> int:
    payload = {
        "board": obs.board_self.to_dict(),
        "opponent_board": obs.board_opponent.to_dict(),
        "dealt": list(obs.dealt_cards),
        "known_discards": list(obs.known_discards_self),
        "turn": int(obs.turn),
        "action_idx": int(action_idx),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _sample_t3_draws(
    obs: Observation,
    action: Action,
    action_idx: int,
    sample_count: int,
    *,
    sample_offset: int = 0,
) -> list[list[str]]:
    deck = _remaining_deck_after_action(obs, action)
    combos = list(itertools.combinations(deck, 3))
    if not combos:
        return []
    rng = random.Random(_stable_seed(obs, action_idx))
    order = list(range(len(combos)))
    rng.shuffle(order)
    sample_offset = max(0, sample_offset)
    available = max(0, len(order) - sample_offset)
    if available == 0:
        return []
    sample_count = min(max(1, sample_count), available)
    selected = order[sample_offset : sample_offset + sample_count]
    return [list(combos[index]) for index in selected]


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def default_rust_solver_path() -> Path:
    suffix = ".exe" if sys.platform.startswith("win") else ""
    return _repo_root() / "ai" / "rust_solver" / "target" / "release" / f"t3_exact_solver{suffix}"


def default_prob_engine_path() -> Path:
    suffix = ".exe" if sys.platform.startswith("win") else ""
    return _repo_root() / "ai" / "rust_solver" / "target" / "release" / f"prob_engine{suffix}"


def _cards_arg(cards: Iterable[str]) -> str:
    return ",".join(str(card) for card in cards if card)


def _mc_exclude_for_candidate(obs: Observation, candidate: dict[str, Any]) -> list[str]:
    action = candidate["_action"]
    exclude = list(obs.board_opponent.all_cards())
    exclude.extend(obs.known_discards_self)
    if action.discard:
        exclude.append(action.discard)
    return exclude


def _t3_payload_for_candidate(obs: Observation, candidate: dict[str, Any], draw: list[str]) -> dict[str, Any]:
    action = candidate["_action"]
    return {
        "turn": 3,
        "board": apply_action(obs.board_self, action).to_dict(),
        "opponent_board": obs.board_opponent.to_dict(),
        "dealt": list(draw),
        "known_discards": list(obs.known_discards_self) + ([action.discard] if action.discard else []),
        "exclude": [],
        "is_btn": bool(obs.is_btn),
    }


def refine_t2_candidates_exact_partial(
    obs: Observation,
    candidates: list[dict[str, Any]],
    *,
    config: HybridConfig = HybridConfig(),
    rust_solver_path: str | Path | None = None,
    started_at: float | None = None,
    t3_pooler: Any | None = None,
) -> dict[str, Any]:
    if not candidates or not config.enable_sync_refinement:
        return {"exact_evaluated": 0, "error": None}
    solver = Path(rust_solver_path) if rust_solver_path is not None else default_rust_solver_path()
    if not solver.exists():
        return {"exact_evaluated": 0, "error": f"missing_rust_solver:{solver}"}

    started = started_at or time.perf_counter()
    remaining = max(0.0, config.time_budget_ms / 1000.0 - (time.perf_counter() - started))
    if remaining <= 0.25:
        return {"exact_evaluated": 0, "error": "time_budget_exhausted"}

    backend = str(config.t2_exact_backend or "full_exact")
    if backend not in {"full_exact", "t3_union"}:
        return {"exact_evaluated": 0, "error": f"unsupported_t2_exact_backend:{backend}"}
    backend_label = (
        "t3_all_legal_exact"
        if backend == "t3_union" and int(config.t3_pool_k) <= 0
        else backend
    )

    exact_evaluated = 0
    score_totals: dict[int, float] = {}
    sample_counts: dict[int, int] = {}
    elapsed_totals: dict[int, float] = {}
    t3_pool_sizes: list[int] = []
    t3_exact_evaluated: list[int] = []
    model_rank_min_sample_batches = 0
    shared_t3_pooler = t3_pooler
    error: str | None = None

    max_samples_per_candidate = max(1, int(config.t2_max_samples_per_candidate))
    deadline_s = started + max(0.0, float(config.time_budget_ms) / 1000.0)

    def apply_refined_scores() -> None:
        for candidate in candidates:
            action_idx = int(candidate["action_idx"])
            count = sample_counts.get(action_idx, 0)
            if count <= 0:
                continue
            candidate["refined_score"] = score_totals[action_idx] / count
            candidate["refinement_source"] = "exact_partial"
            candidate["refinement_backend"] = backend_label
            candidate["samples"] = int(candidate.get("samples", 0)) + count
            candidate["elapsed_ms"] = float(candidate.get("elapsed_ms", 0.0)) + elapsed_totals.get(action_idx, 0.0)

    def t3_union_summary_payload() -> dict[str, Any] | None:
        if not t3_pool_sizes:
            return None
        return {
            "samples": len(t3_pool_sizes),
            "pool_size_min": min(t3_pool_sizes),
            "pool_size_mean": sum(t3_pool_sizes) / len(t3_pool_sizes),
            "pool_size_max": max(t3_pool_sizes),
            "exact_evaluated_min": min(t3_exact_evaluated) if t3_exact_evaluated else 0,
            "exact_evaluated_mean": (
                sum(t3_exact_evaluated) / len(t3_exact_evaluated) if t3_exact_evaluated else 0.0
            ),
            "exact_evaluated_max": max(t3_exact_evaluated) if t3_exact_evaluated else 0,
        }

    def result_payload(error_value: str | None) -> dict[str, Any]:
        apply_refined_scores()
        return {
            "exact_evaluated": exact_evaluated,
            "error": error_value,
            "t2_exact_backend": backend_label,
            "t2_exact_backend_internal": backend,
            "samples_by_action": {str(k): int(v) for k, v in sorted(sample_counts.items())},
            "initial_samples_per_candidate": max(1, int(config.t2_initial_samples_per_candidate)),
            "extra_samples_per_round": max(1, int(config.t2_extra_samples_per_round)),
            "max_samples_per_candidate": max_samples_per_candidate,
            "close_candidate_limit": max(0, int(config.t2_close_candidate_limit)),
            "model_rank_min_samples_k": max(0, int(config.t2_model_rank_min_samples_k)),
            "model_rank_min_samples": max(0, int(config.t2_model_rank_min_samples)),
            "model_rank_min_sample_batches": int(model_rank_min_sample_batches),
            "t3_pool_config": str(config.t3_pool_config) if backend == "t3_union" else None,
            "t3_pool_k": int(config.t3_pool_k) if backend == "t3_union" else None,
            "t3_union_summary": t3_union_summary_payload(),
        }

    def run_t3_union_batch(rows: list[dict[str, Any]], row_to_candidate: list[dict[str, Any]]) -> bool:
        nonlocal exact_evaluated, error, shared_t3_pooler
        if not rows:
            return True
        try:
            from ai.tutor.t3_runtime import T3UnionCandidatePool, evaluate_t3_position
        except Exception as exc:
            error = f"t3_union_import_failed:{type(exc).__name__}:{exc}"
            return False

        config_path = Path(config.t3_pool_config)
        if not config_path.is_absolute():
            config_path = _repo_root() / config_path
        if shared_t3_pooler is None:
            try:
                shared_t3_pooler = T3UnionCandidatePool(config_path=config_path, device="auto")
            except Exception as exc:
                error = f"t3_union_pooler_failed:{type(exc).__name__}:{exc}"
                return False

        for row, candidate in zip(rows, row_to_candidate):
            batch_remaining = max(0.0, config.time_budget_ms / 1000.0 - (time.perf_counter() - started))
            if batch_remaining <= 0.25:
                error = "time_budget_exhausted"
                return False
            before = time.perf_counter()
            try:
                result = evaluate_t3_position(
                    row,
                    pooler=shared_t3_pooler,
                    per_source_top_k=int(config.t3_pool_k) if int(config.t3_pool_k) != 0 else None,
                    rust_solver_path=solver,
                    rust_timeout_s=max(0.25, batch_remaining),
                    deadline_s=deadline_s,
                )
            except TimeoutError as exc:
                error = f"t3_union_time_budget_exhausted:{exc}"
                return False
            except Exception as exc:
                error = f"t3_union_failed:{type(exc).__name__}:{exc}"
                return False

            score = float(((result.get("best") or {}).get("metrics") or {}).get("score", candidate["model_score"]))
            action_idx = int(candidate["action_idx"])
            score_totals[action_idx] = score_totals.get(action_idx, 0.0) + score
            sample_counts[action_idx] = sample_counts.get(action_idx, 0) + 1
            elapsed_totals[action_idx] = elapsed_totals.get(action_idx, 0.0) + (
                float(result.get("elapsed_ms", 0.0) or 0.0) or ((time.perf_counter() - before) * 1000.0)
            )
            t3_pool_sizes.append(int(result.get("candidate_pool_size", 0) or 0))
            t3_exact_evaluated.append(int(result.get("exact_evaluated", 0) or 0))
            exact_evaluated += 1
        return True

    def run_sample_batch(batch_candidates: list[dict[str, Any]], samples_per_candidate: int) -> bool:
        nonlocal exact_evaluated, error
        rows = []
        row_to_candidate: list[dict[str, Any]] = []
        samples_per_candidate = max(1, int(samples_per_candidate))
        for candidate in batch_candidates:
            action_idx = int(candidate["action_idx"])
            current_samples = sample_counts.get(action_idx, 0)
            if current_samples >= max_samples_per_candidate:
                continue
            requested_samples = min(samples_per_candidate, max_samples_per_candidate - current_samples)
            draws = _sample_t3_draws(
                obs,
                candidate["_action"],
                action_idx,
                requested_samples,
                sample_offset=current_samples,
            )
            for draw in draws:
                rows.append(_t3_payload_for_candidate(obs, candidate, draw))
                row_to_candidate.append(candidate)

        if not rows:
            return True

        batch_remaining = max(0.0, config.time_budget_ms / 1000.0 - (time.perf_counter() - started))
        if batch_remaining <= 0.25:
            error = "time_budget_exhausted"
            return False
        if backend == "t3_union":
            return run_t3_union_batch(rows, row_to_candidate)

        with tempfile.TemporaryDirectory(prefix="ofc_hybrid_t2_") as tmp:
            input_path = Path(tmp) / "input.jsonl"
            output_path = Path(tmp) / "output.jsonl"
            input_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            cmd = [
                str(solver),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--top-n",
                "1",
            ]
            before = time.perf_counter()
            try:
                subprocess.run(
                    cmd,
                    cwd=str(_repo_root()),
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=max(0.25, batch_remaining),
                )
            except subprocess.TimeoutExpired:
                error = "time_budget_timeout"
                return False
            except subprocess.CalledProcessError as exc:
                error = f"rust_solver_failed:{exc.stderr[-500:]}"
                return False
            elapsed_ms = (time.perf_counter() - before) * 1000.0
            lines = output_path.read_text(encoding="utf-8").splitlines() if output_path.exists() else []

        per_row_elapsed_ms = elapsed_ms / max(len(row_to_candidate), 1)
        for line, candidate in zip(lines, row_to_candidate):
            if not line.strip():
                continue
            result = json.loads(line)
            score = float(((result.get("best") or {}).get("metrics") or {}).get("score", candidate["model_score"]))
            action_idx = int(candidate["action_idx"])
            score_totals[action_idx] = score_totals.get(action_idx, 0.0) + score
            sample_counts[action_idx] = sample_counts.get(action_idx, 0) + 1
            elapsed_totals[action_idx] = elapsed_totals.get(action_idx, 0.0) + per_row_elapsed_ms
            exact_evaluated += 1
        return True

    initial_samples = max(1, int(config.t2_initial_samples_per_candidate))
    extra_samples_per_round = max(1, int(config.t2_extra_samples_per_round))
    close_candidate_limit = max(0, int(config.t2_close_candidate_limit))
    model_rank_min_samples_k = max(0, int(config.t2_model_rank_min_samples_k))
    model_rank_min_samples = max(0, min(max_samples_per_candidate, int(config.t2_model_rank_min_samples)))
    model_rank_min_remaining_ms = max(0, int(config.t2_model_rank_min_samples_min_remaining_ms))

    if not run_sample_batch(candidates, initial_samples):
        return result_payload(error)
    if exact_evaluated == 0:
        return result_payload("no_t3_draws")

    if model_rank_min_samples_k > 0 and model_rank_min_samples > initial_samples:
        while True:
            remaining_ms = int(config.time_budget_ms) - int((time.perf_counter() - started) * 1000.0)
            if remaining_ms < model_rank_min_remaining_ms:
                break
            model_rank_candidates = [
                candidate
                for candidate in candidates
                if int(candidate.get("model_rank", 999999)) <= model_rank_min_samples_k
                and sample_counts.get(int(candidate["action_idx"]), 0) < model_rank_min_samples
                and sample_counts.get(int(candidate["action_idx"]), 0) < max_samples_per_candidate
            ]
            if not model_rank_candidates:
                break
            samples_to_add = min(
                model_rank_min_samples - sample_counts.get(int(candidate["action_idx"]), 0)
                for candidate in model_rank_candidates
            )
            before_count = exact_evaluated
            if not run_sample_batch(model_rank_candidates, samples_to_add):
                return result_payload(error)
            model_rank_min_sample_batches += 1
            if exact_evaluated == before_count:
                break

    baseline_candidate = min(candidates, key=lambda item: int(item["model_rank"]), default=None)

    while True:
        remaining = max(0.0, config.time_budget_ms / 1000.0 - (time.perf_counter() - started))
        if remaining <= 0.25:
            break
        scored = [
            candidate
            for candidate in candidates
            if sample_counts.get(int(candidate["action_idx"]), 0) > 0
        ]
        if not scored:
            break

        def average_score(item: dict[str, Any]) -> float:
            action_idx = int(item["action_idx"])
            return score_totals[action_idx] / sample_counts[action_idx]

        best_score = max(average_score(candidate) for candidate in scored)
        close_candidates = [
            candidate
            for candidate in sorted(scored, key=average_score, reverse=True)
            if best_score - average_score(candidate) <= config.t2_extra_margin
            and sample_counts.get(int(candidate["action_idx"]), 0) < max_samples_per_candidate
        ]
        if close_candidate_limit > 0:
            close_candidates = close_candidates[:close_candidate_limit]
        if baseline_candidate is not None:
            baseline_idx = int(baseline_candidate["action_idx"])
            if (
                sample_counts.get(baseline_idx, 0) < max(1, int(config.t2_baseline_min_samples))
                and sample_counts.get(baseline_idx, 0) < max_samples_per_candidate
            ):
                if baseline_idx not in {int(candidate["action_idx"]) for candidate in close_candidates}:
                    close_candidates.append(baseline_candidate)
        before_count = exact_evaluated
        if not close_candidates or not run_sample_batch(close_candidates, extra_samples_per_round):
            break
        if exact_evaluated == before_count:
            break

    return result_payload(error)


def refine_t2_candidates_mc_board(
    obs: Observation,
    candidates: list[dict[str, Any]],
    *,
    config: HybridConfig = HybridConfig(),
    prob_engine_path: str | Path | None = None,
    started_at: float | None = None,
) -> dict[str, Any]:
    if not candidates or not config.enable_sync_refinement:
        return {"exact_evaluated": 0, "error": None}
    engine = Path(prob_engine_path) if prob_engine_path is not None else default_prob_engine_path()
    if not engine.exists():
        return {"exact_evaluated": 0, "error": f"missing_prob_engine:{engine}"}

    started = started_at or time.perf_counter()
    remaining = max(0.0, config.time_budget_ms / 1000.0 - (time.perf_counter() - started))
    if remaining <= 0.25:
        return {"exact_evaluated": 0, "error": "time_budget_exhausted"}

    rows = []
    by_action_idx = {int(candidate["action_idx"]): candidate for candidate in candidates}
    for candidate in candidates:
        board = candidate["board"]
        rows.append(
            {
                "id": int(candidate["action_idx"]),
                "mode": "board_mc",
                "top": _cards_arg(board.get("top", [])),
                "mid": _cards_arg(board.get("middle", []) or board.get("mid", [])),
                "bot": _cards_arg(board.get("bottom", []) or board.get("bot", [])),
                "exclude": _cards_arg(_mc_exclude_for_candidate(obs, candidate)),
                "turn": 3,
                "sims": int(config.t2_mc_sims),
            }
        )

    with tempfile.TemporaryDirectory(prefix="ofc_hybrid_t2_mc_") as tmp:
        input_path = Path(tmp) / "input.jsonl"
        output_path = Path(tmp) / "output.jsonl"
        input_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        cmd = [
            str(engine),
            "--mode",
            "batch",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
        before = time.perf_counter()
        try:
            subprocess.run(
                cmd,
                cwd=str(_repo_root()),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=max(0.25, remaining),
            )
        except subprocess.TimeoutExpired:
            return {"exact_evaluated": 0, "error": "time_budget_timeout"}
        except subprocess.CalledProcessError as exc:
            return {"exact_evaluated": 0, "error": f"prob_engine_failed:{exc.stderr[-500:]}"}
        elapsed_ms = (time.perf_counter() - before) * 1000.0
        lines = output_path.read_text(encoding="utf-8").splitlines() if output_path.exists() else []

    evaluated = 0
    per_candidate_elapsed_ms = elapsed_ms / max(len(candidates), 1)
    for line in lines:
        if not line.strip():
            continue
        response = json.loads(line)
        if not response.get("ok"):
            continue
        action_idx = int(response.get("id"))
        candidate = by_action_idx.get(action_idx)
        result = response.get("result") or {}
        mc = result.get("mc") or {}
        if candidate is None or "avg_score" not in mc:
            continue
        candidate["refined_score"] = float(mc["avg_score"])
        candidate["refinement_source"] = "mc_board"
        candidate["samples"] = int(candidate.get("samples", 0)) + int(mc.get("simulations", config.t2_mc_sims) or 0)
        candidate["elapsed_ms"] = float(candidate.get("elapsed_ms", 0.0)) + per_candidate_elapsed_ms
        candidate["refined_bust"] = float(mc.get("bust_rate", candidate["predicted_bust"]))
        candidate["refined_fl"] = float(mc.get("fl_rate", candidate["predicted_fl"]))
        evaluated += 1

    return {"exact_evaluated": evaluated, "error": None if evaluated else "no_mc_results"}


def refine_t1_candidates_recursive_mc(
    obs: Observation,
    candidates: list[dict[str, Any]],
    *,
    config: HybridConfig = HybridConfig(),
    prob_engine_path: str | Path | None = None,
    started_at: float | None = None,
) -> dict[str, Any]:
    if not candidates or not config.enable_sync_refinement or config.t1_refinement == "none":
        return {"exact_evaluated": 0, "error": None}
    engine = Path(prob_engine_path) if prob_engine_path is not None else default_prob_engine_path()
    if not engine.exists():
        return {"exact_evaluated": 0, "error": f"missing_prob_engine:{engine}"}

    started = started_at or time.perf_counter()
    remaining = max(0.0, config.time_budget_ms / 1000.0 - (time.perf_counter() - started))
    if remaining <= 0.25:
        return {"exact_evaluated": 0, "error": "time_budget_exhausted"}

    mode = "board_recursive_mc" if config.t1_refinement == "recursive_mc" else "board_mc"

    def run_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
        batch_remaining = max(0.0, config.time_budget_ms / 1000.0 - (time.perf_counter() - started))
        timeout_headroom = max(0.0, float(config.t1_refinement_timeout_headroom_ms) / 1000.0)
        subprocess_timeout = batch_remaining - timeout_headroom
        if subprocess_timeout <= 0.25:
            return {"exact_evaluated": 0, "error": "time_budget_exhausted"}
        rows = []
        by_action_idx = {int(candidate["action_idx"]): candidate for candidate in batch}
        for candidate in batch:
            board = candidate["board"]
            rows.append(
                {
                    "id": int(candidate["action_idx"]),
                    "mode": mode,
                    "top": _cards_arg(board.get("top", [])),
                    "mid": _cards_arg(board.get("middle", []) or board.get("mid", [])),
                    "bot": _cards_arg(board.get("bottom", []) or board.get("bot", [])),
                    "exclude": _cards_arg(_mc_exclude_for_candidate(obs, candidate)),
                    "turn": 2,
                    "sims": int(config.t1_mc_sims),
                    "recursive_beam": int(config.t1_recursive_beam),
                    "recursive_child_sims": int(config.t1_recursive_child_sims),
                }
            )

        with tempfile.TemporaryDirectory(prefix="ofc_hybrid_t1_mc_") as tmp:
            input_path = Path(tmp) / "input.jsonl"
            output_path = Path(tmp) / "output.jsonl"
            input_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            cmd = [
                str(engine),
                "--mode",
                "batch",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ]
            before = time.perf_counter()
            try:
                subprocess.run(
                    cmd,
                    cwd=str(_repo_root()),
                    check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=max(0.25, subprocess_timeout),
            )
            except subprocess.TimeoutExpired:
                return {"exact_evaluated": 0, "error": "time_budget_timeout"}
            except subprocess.CalledProcessError as exc:
                return {"exact_evaluated": 0, "error": f"prob_engine_failed:{exc.stderr[-500:]}"}
            elapsed_ms = (time.perf_counter() - before) * 1000.0
            lines = output_path.read_text(encoding="utf-8").splitlines() if output_path.exists() else []

        evaluated = 0
        per_candidate_elapsed_ms = elapsed_ms / max(len(batch), 1)
        for line in lines:
            if not line.strip():
                continue
            response = json.loads(line)
            if not response.get("ok"):
                continue
            action_idx = int(response.get("id"))
            candidate = by_action_idx.get(action_idx)
            result = response.get("result") or {}
            mc = result.get("mc") or {}
            if candidate is None or "avg_score" not in mc:
                continue
            new_samples = int(mc.get("simulations", config.t1_mc_sims) or 0)
            old_samples = int(candidate.get("samples", 0) or 0)
            old_score = candidate.get("refined_score")
            if old_score is not None and old_samples > 0 and new_samples > 0:
                candidate["refined_score"] = (
                    float(old_score) * old_samples + float(mc["avg_score"]) * new_samples
                ) / (old_samples + new_samples)
            else:
                candidate["refined_score"] = float(mc["avg_score"])
            candidate["refinement_source"] = config.t1_refinement
            candidate["samples"] = old_samples + new_samples
            candidate["elapsed_ms"] = float(candidate.get("elapsed_ms", 0.0)) + per_candidate_elapsed_ms
            candidate["refined_bust"] = float(mc.get("bust_rate", candidate["predicted_bust"]))
            candidate["refined_fl"] = float(mc.get("fl_rate", candidate["predicted_fl"]))
            evaluated += 1

        return {"exact_evaluated": evaluated, "error": None if evaluated else "no_mc_results"}

    first_batch_k = max(0, int(config.t1_refinement_first_batch_k))
    if first_batch_k <= 0 or first_batch_k >= len(candidates):
        return run_batch(candidates)

    first_result = run_batch(candidates[:first_batch_k])
    exact_evaluated = int(first_result.get("exact_evaluated", 0) or 0)
    errors = [str(first_result["error"])] if first_result.get("error") else []
    if first_result.get("error") in {"time_budget_timeout", "time_budget_exhausted"}:
        return {"exact_evaluated": exact_evaluated, "error": first_result.get("error")}

    tail = candidates[first_batch_k:]
    remaining_ms = config.time_budget_ms - int((time.perf_counter() - started) * 1000.0)
    min_tail_remaining_ms = max(0, int(config.t1_refinement_tail_min_remaining_ms))
    if tail and remaining_ms >= min_tail_remaining_ms:
        tail_result = run_batch(tail)
        exact_evaluated += int(tail_result.get("exact_evaluated", 0) or 0)
        if tail_result.get("error"):
            errors.append(str(tail_result["error"]))
            for candidate in tail:
                if candidate.get("refined_score") is None:
                    candidate["refinement_source"] = "exact_background"
                    candidate["refinement_skip_reason"] = str(tail_result["error"])
    elif tail:
        for candidate in tail:
            if candidate.get("refined_score") is None:
                candidate["refinement_source"] = "exact_background"
                candidate["refinement_skip_reason"] = "tail_skipped_headroom"
        errors.append("tail_skipped_headroom")

    return {"exact_evaluated": exact_evaluated, "error": ";".join(errors) if errors else None}


def evaluate_hybrid_position(
    payload_or_obs: dict[str, Any] | Observation,
    *,
    evaluator: RolloutEvaluator | None = None,
    shortlist_evaluator: RolloutEvaluator | None = None,
    model: torch.nn.Module | None = None,
    model_path: str | Path | None = None,
    device: str | None = None,
    config: HybridConfig = HybridConfig(),
    rust_solver_path: str | Path | None = None,
    t3_pooler: Any | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    obs = payload_or_obs if isinstance(payload_or_obs, Observation) else observation_from_payload(payload_or_obs)
    if int(obs.turn) not in (1, 2):
        raise ValueError("evaluate_hybrid_position only supports T1/T2")
    if len(obs.dealt_cards) != 3:
        raise ValueError("T1/T2 hybrid evaluation requires exactly 3 dealt cards")
    if evaluator is None:
        evaluator = make_action_value_evaluator(model=model, model_path=model_path, device=device)

    valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)
    indexed_actions = list(enumerate(valid_actions))
    details = evaluator.score_candidates_action_value_details(obs, indexed_actions)
    aux_details = None
    aux_shortlist_enabled = (
        (int(obs.turn) == 1 and config.t1_aux_shortlist_k > 0)
        or (int(obs.turn) == 2 and config.t2_aux_shortlist_k > 0)
    )
    if aux_shortlist_enabled and shortlist_evaluator is not None:
        aux_details = shortlist_evaluator.score_candidates_action_value_details(obs, indexed_actions)
    candidates: list[dict[str, Any]] = []
    for i, (action_index, action) in enumerate(indexed_actions):
        fl_types = details["fl_type_probs"][i] if len(details["fl_type_probs"]) > i else [0.0] * len(FL_TYPE_KEYS)
        candidate = {
            "action_idx": int(encode_action(action, valid_actions, turn=int(obs.turn), dealt_cards=obs.dealt_cards)),
            "action": action_to_dict(action),
            "turn": int(obs.turn),
            "board": apply_action(obs.board_self, action).to_dict(),
            "opponent_board": obs.board_opponent.to_dict(),
            "known_discards": list(obs.known_discards_self),
            "model_score": float(details["model_score"][i]),
            "model_raw_score": float(details["raw_score"][i]),
            "model_rank": 0,
            "refinement_rank": 0,
            "predicted_bust": float(details["bust_prob"][i]),
            "predicted_fl": float(details["fl_prob"][i]),
            "predicted_fl_types": {
                key: float(fl_types[j]) if len(fl_types) > j else 0.0 for j, key in enumerate(FL_TYPE_KEYS)
            },
            "refined_score": None,
            "refinement_source": "none",
            "samples": 0,
            "elapsed_ms": 0.0,
            "_action": action,
            "_list_index": int(action_index),
        }
        if aux_details is not None:
            candidate["aux_model_score"] = float(aux_details["model_score"][i])
            candidate["aux_model_raw_score"] = float(aux_details["raw_score"][i])
        candidates.append(candidate)

    shortlist_config = config
    t2_adaptive_sync_applied = _should_expand_t2_sync_candidates(
        candidates,
        config,
        turn=int(obs.turn),
    )
    if t2_adaptive_sync_applied:
        shortlist_config = replace(
            config,
            shortlist_k=max(int(config.shortlist_k), int(config.t2_adaptive_shortlist_k)),
        )

    pool = build_shortlist(candidates, config=shortlist_config)
    if int(obs.turn) == 2:
        annotate_t2_forced_bust_candidates(obs, pool)
    if int(obs.turn) == 1 and config.t1_sync_selector is not None:
        for candidate in pool:
            candidate["sync_selector_score"] = _linear_selector_score(
                config.t1_sync_selector,
                t1_sync_selector_features(candidate),
            )
    if int(obs.turn) == 2 and config.t2_sync_selector is not None:
        for candidate in pool:
            candidate["sync_selector_score"] = _linear_selector_score(
                config.t2_sync_selector,
                t2_sync_selector_features(candidate),
            )
    refinement_config = shortlist_config
    if int(obs.turn) == 2 and int(config.t2_sync_exact_k) > 0:
        t2_sync_exact_k = int(config.t2_sync_exact_k)
        if t2_adaptive_sync_applied:
            t2_sync_exact_k = max(t2_sync_exact_k, int(config.t2_adaptive_sync_exact_k))
        refinement_config = replace(shortlist_config, sync_exact_k=t2_sync_exact_k)
    sync_candidates = _sync_refinement_candidates(pool, refinement_config, turn=int(obs.turn))
    sync_action_indices = {int(item["action_idx"]) for item in sync_candidates}
    for candidate in pool:
        if int(candidate["action_idx"]) not in sync_action_indices:
            candidate["refinement_source"] = "exact_background"

    exact_evaluated = 0
    refinement_error = None
    refinement_backend = None
    refinement_sample_counts = None
    t3_union_summary = None
    t2_post_refine_expanded = False
    t2_post_refine_candidate_count = 0
    t2_post_refine_error = None
    t2_post_refine_policy_details = None
    t2_post_refine_skipped_reason = None
    t1_extra_refinement_candidate_count = 0
    t1_extra_refinement_error = None

    def merge_sample_counts(
        base: dict[str, int] | None,
        extra: dict[str, int] | None,
    ) -> dict[str, int] | None:
        if not base and not extra:
            return None
        merged: dict[str, int] = {}
        for source in (base or {}, extra or {}):
            for key, value in source.items():
                merged[str(key)] = merged.get(str(key), 0) + int(value)
        return merged

    def merge_t3_union_summary(
        base: dict[str, Any] | None,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not base:
            return extra
        if not extra:
            return base
        base_samples = int(base.get("samples", 0) or 0)
        extra_samples = int(extra.get("samples", 0) or 0)
        total = base_samples + extra_samples
        if total <= 0:
            return base

        def weighted_mean(key: str) -> float:
            return (
                float(base.get(key, 0.0) or 0.0) * base_samples
                + float(extra.get(key, 0.0) or 0.0) * extra_samples
            ) / total

        return {
            "samples": total,
            "pool_size_min": min(int(base.get("pool_size_min", 0) or 0), int(extra.get("pool_size_min", 0) or 0)),
            "pool_size_mean": weighted_mean("pool_size_mean"),
            "pool_size_max": max(int(base.get("pool_size_max", 0) or 0), int(extra.get("pool_size_max", 0) or 0)),
            "exact_evaluated_min": min(
                int(base.get("exact_evaluated_min", 0) or 0),
                int(extra.get("exact_evaluated_min", 0) or 0),
            ),
            "exact_evaluated_mean": weighted_mean("exact_evaluated_mean"),
            "exact_evaluated_max": max(
                int(base.get("exact_evaluated_max", 0) or 0),
                int(extra.get("exact_evaluated_max", 0) or 0),
            ),
        }

    if int(obs.turn) == 2 and sync_candidates:
        if config.t2_refinement == "mc_board":
            refine_result = refine_t2_candidates_mc_board(
                obs,
                sync_candidates,
                config=config,
                prob_engine_path=None,
                started_at=started,
            )
        else:
            refine_result = refine_t2_candidates_exact_partial(
                obs,
                sync_candidates,
                config=config,
                rust_solver_path=rust_solver_path,
                started_at=started,
                t3_pooler=t3_pooler,
            )
        exact_evaluated = int(refine_result.get("exact_evaluated", 0) or 0)
        refinement_error = refine_result.get("error")
        refinement_backend = refine_result.get("t2_exact_backend")
        refinement_sample_counts = refine_result.get("samples_by_action")
        t3_union_summary = refine_result.get("t3_union_summary")
        post_refine_k = int(config.t2_post_refine_sync_exact_k)
        remaining_ms = int(config.time_budget_ms) - int((time.perf_counter() - started) * 1000.0)
        if (
            config.t2_refinement != "mc_board"
            and post_refine_k > int(refinement_config.sync_exact_k)
            and remaining_ms >= int(config.t2_post_refine_min_remaining_ms)
            and refinement_error is None
        ):
            expanded_config = replace(refinement_config, sync_exact_k=post_refine_k)
            expanded_candidates = _sync_refinement_candidates(pool, expanded_config, turn=int(obs.turn))
            extra_candidates = [
                item for item in expanded_candidates if item.get("refined_score") is None
            ]
            if extra_candidates:
                accepted, t2_post_refine_policy_details = _t2_post_refine_policy_accepts(
                    obs,
                    extra_candidates,
                    config,
                )
                if accepted:
                    t2_post_refine_expanded = True
                    t2_post_refine_candidate_count = len(extra_candidates)
                    extra_result = refine_t2_candidates_exact_partial(
                        obs,
                        extra_candidates,
                        config=config,
                        rust_solver_path=rust_solver_path,
                        started_at=started,
                        t3_pooler=t3_pooler,
                    )
                    exact_evaluated += int(extra_result.get("exact_evaluated", 0) or 0)
                    t2_post_refine_error = extra_result.get("error")
                    refinement_sample_counts = merge_sample_counts(
                        refinement_sample_counts,
                        extra_result.get("samples_by_action"),
                    )
                    t3_union_summary = merge_t3_union_summary(
                        t3_union_summary,
                        extra_result.get("t3_union_summary"),
                    )
                    if refinement_backend is None:
                        refinement_backend = extra_result.get("t2_exact_backend")
                else:
                    t2_post_refine_skipped_reason = str(
                        (t2_post_refine_policy_details or {}).get("reject_reason") or "policy_rejected"
                    )
    elif int(obs.turn) == 1 and sync_candidates:
        refine_result = refine_t1_candidates_recursive_mc(
            obs,
            sync_candidates,
            config=config,
            prob_engine_path=None,
            started_at=started,
        )
        exact_evaluated = int(refine_result.get("exact_evaluated", 0) or 0)
        refinement_error = refine_result.get("error")
        refinement_sample_counts = refine_result.get("samples_by_action")
        extra_candidates = _t1_extra_refinement_candidates(
            [item for item in pool if item.get("refined_score") is not None],
            config,
        )
        if extra_candidates:
            t1_extra_refinement_candidate_count = len(extra_candidates)
            extra_config = replace(config, t1_mc_sims=int(config.t1_extra_refine_sims))
            extra_result = refine_t1_candidates_recursive_mc(
                obs,
                extra_candidates,
                config=extra_config,
                prob_engine_path=None,
                started_at=started,
            )
            exact_evaluated += int(extra_result.get("exact_evaluated", 0) or 0)
            t1_extra_refinement_error = extra_result.get("error")

    # Refined exact/partial-exact scores are on the teacher EV scale, while raw
    # model scores are uncalibrated.  Do not mix them when choosing the
    # synchronous answer.  Background candidates remain in the response for UI
    # and logging, but once any candidate has been refined the answer is chosen
    # from refined candidates only.
    refined_candidates = [item for item in pool if item.get("refined_score") is not None]

    def output_sort_key(item: dict[str, Any]) -> tuple[int, float, float, int]:
        refined = item.get("refined_score")
        if refined is not None:
            return (1, float(refined), float(item["model_score"]), -int(item["model_rank"]))
        return (0, float(item["model_score"]), float(item["model_score"]), -int(item["model_rank"]))

    output_candidates = sorted(pool, key=output_sort_key, reverse=True)
    refined_ranked_candidates = [
        item for item in output_candidates if item.get("refined_score") is not None
    ]
    for refined_rank, candidate in enumerate(refined_ranked_candidates, start=1):
        candidate["refined_rank"] = refined_rank
    model_top1 = min(pool, key=lambda item: int(item["model_rank"]), default=None)
    t1_override_features = None
    t1_override_gate_probability = None
    t1_override_gate_accepted = None
    t1_override_reject_reason = None
    t1_final_bottom_sparse_rescue_applied = False
    t1_final_bottom_sparse_rescue_action_idx = None
    t1_final_bottom_sparse_rescue_delta = None
    t1_no_refine_fallback_action_idx = None
    t1_no_refine_fallback_applied = False
    t1_arbitration_details = None
    t1_final_challenger_details = None
    t1_refined_challenger_details = None
    t1_blend_challenger_details = None
    t1_low_risk_blend_challenger_details = None
    t1_model_rescue_details = None
    t2_model_top1_rescue_details = None
    t2_model_top1_bust_rescue_details = None
    t2_middle_fill_bottom_shift_rescue_details = None
    t2_model_rank_rescue_details = None

    def refined_selection_key(item: dict[str, Any]) -> tuple[float, float, float, int]:
        refined = float(item.get("refined_score", float("-inf")))
        model_score = float(item.get("model_score", float("-inf")))
        if int(obs.turn) == 1 and config.t1_selection_policy == "selector" and config.t1_final_selector is not None:
            selector_score = _linear_selector_score(config.t1_final_selector, t1_final_selector_features(item))
            item["final_selector_score"] = selector_score
        elif int(obs.turn) == 1 and config.t1_selection_policy == "refined_plus_model":
            selector_score = refined + float(config.t1_selection_model_weight) * model_score
        elif int(obs.turn) == 2 and config.t2_selection_policy == "selector" and config.t2_final_selector is not None:
            selector_score = _linear_selector_score(config.t2_final_selector, t2_final_selector_features(item))
            item["t2_final_selector_score"] = selector_score
        elif int(obs.turn) == 2 and config.t2_selection_policy == "refined_plus_model":
            model_weight = float(config.t2_selection_model_weight)
            structured_top_weight = float(config.t2_selection_structured_top_model_weight)
            if structured_top_weight >= 0.0 and _t2_own_top_structured(obs):
                model_weight = structured_top_weight
            selector_score = refined + model_weight * model_score
            item["t2_selection_model_weight_used"] = model_weight
        else:
            selector_score = refined
        return (selector_score, refined, model_score, -int(item["model_rank"]))

    if refined_candidates:
        best_candidate = max(
            refined_candidates,
            key=refined_selection_key,
        )
        if int(obs.turn) == 1 and best_candidate is not None:
            base_arbitration_candidate = best_candidate
            best_candidate, t1_arbitration_details = _t1_arbitration_candidate(
                base_arbitration_candidate,
                refined_candidates,
                selector=config.t1_arbitration_selector,
                policy=str(config.t1_arbitration_policy),
            )
            if (
                config.t1_arbitration_challenger_selector is not None
                and str(config.t1_arbitration_dual_gate_policy) != "none"
            ):
                challenger_candidate, challenger_details = _t1_arbitration_candidate(
                    base_arbitration_candidate,
                    refined_candidates,
                    selector=config.t1_arbitration_challenger_selector,
                    policy=str(config.t1_arbitration_policy),
                    score_key="arbitration_challenger_selector_score",
                )
                best_candidate, t1_arbitration_details = _t1_dual_arbitration_candidate(
                    base_arbitration_candidate,
                    best_candidate,
                    t1_arbitration_details,
                    challenger_candidate,
                    challenger_details,
                    model_top1=model_top1,
                    policy=str(config.t1_arbitration_dual_gate_policy),
                )
        if (
            int(obs.turn) == 1
            and model_top1 is not None
            and int(best_candidate["action_idx"]) != int(model_top1["action_idx"])
        ):
            t1_override_features = _override_gate_features(model_top1, best_candidate)
            if not _margin_accepts_override(
                config.t1_override_margin,
                t1_override_features["refined_delta"],
            ):
                t1_override_gate_accepted = False
                t1_override_reject_reason = "margin"
                best_candidate = model_top1
            else:
                accepted, probability = _gate_accepts_override(
                    config.t1_override_gate,
                    t1_override_features,
                    threshold=config.t1_override_gate_threshold,
                )
                t1_override_gate_probability = probability
                t1_override_gate_accepted = accepted
                if not accepted:
                    t1_override_reject_reason = "gate"
                    best_candidate = model_top1
        if (
            int(obs.turn) == 1
            and float(config.t1_final_bottom_sparse_rescue_margin) >= 0.0
            and best_candidate is not None
            and best_candidate.get("refined_score") is not None
        ):
            sparse_best, rescue_delta = _t1_bottom_sparse_rescue_candidate(
                best_candidate,
                refined_candidates,
                margin=float(config.t1_final_bottom_sparse_rescue_margin),
            )
            if sparse_best is not None:
                best_candidate = sparse_best
                t1_final_bottom_sparse_rescue_applied = True
                t1_final_bottom_sparse_rescue_action_idx = int(sparse_best["action_idx"])
                t1_final_bottom_sparse_rescue_delta = rescue_delta
        if (
            int(obs.turn) == 1
            and best_candidate is not None
            and config.t1_final_challenger_selector is not None
            and str(config.t1_final_challenger_gate_policy) != "none"
        ):
            best_candidate, t1_final_challenger_details = _t1_final_challenger_candidate(
                best_candidate,
                refined_candidates,
                selector=config.t1_final_challenger_selector,
                gate=config.t1_final_challenger_gate,
                model_top1=model_top1,
                override_gate=config.t1_override_gate,
                override_gate_threshold=config.t1_override_gate_threshold,
                policy=str(config.t1_final_challenger_gate_policy),
            )
        if (
            int(obs.turn) == 1
            and best_candidate is not None
            and float(config.t1_refined_challenger_bust_max) >= 0.0
            and float(config.t1_refined_challenger_refined_delta_max) >= 0.0
        ):
            best_candidate, t1_refined_challenger_details = _t1_refined_challenger_candidate(
                best_candidate,
                refined_candidates,
                bust_max=float(config.t1_refined_challenger_bust_max),
                refined_delta_max=float(config.t1_refined_challenger_refined_delta_max),
            )
        if (
            int(obs.turn) == 1
            and best_candidate is not None
            and float(config.t1_blend_challenger_weight) >= 0.0
            and float(config.t1_blend_challenger_bust_min) >= 0.0
            and float(config.t1_blend_challenger_premium_fl_delta_min) >= 0.0
        ):
            best_candidate, t1_blend_challenger_details = _t1_blend_challenger_candidate(
                best_candidate,
                refined_candidates,
                weight=float(config.t1_blend_challenger_weight),
                bust_min=float(config.t1_blend_challenger_bust_min),
                premium_fl_delta_min=float(config.t1_blend_challenger_premium_fl_delta_min),
            )
        if (
            int(obs.turn) == 1
            and best_candidate is not None
            and float(config.t1_low_risk_blend_challenger_weight) >= 0.0
            and float(config.t1_low_risk_blend_challenger_fl_max) >= 0.0
            and float(config.t1_low_risk_blend_challenger_bust_max) >= 0.0
        ):
            best_candidate, t1_low_risk_blend_challenger_details = _t1_low_risk_blend_challenger_candidate(
                best_candidate,
                refined_candidates,
                weight=float(config.t1_low_risk_blend_challenger_weight),
                fl_max=float(config.t1_low_risk_blend_challenger_fl_max),
                bust_max=float(config.t1_low_risk_blend_challenger_bust_max),
            )
        if (
            int(obs.turn) == 1
            and best_candidate is not None
            and (
                float(config.t1_model_rescue_qq_delta_min) >= 0.0
                or float(config.t1_model_rescue_premium_delta_min) >= 0.0
            )
            and float(config.t1_model_rescue_candidate_fl_delta_min) >= 0.0
        ):
            best_candidate, t1_model_rescue_details = _t1_model_rescue_candidate(
                best_candidate,
                model_top1,
                qq_delta_min=float(config.t1_model_rescue_qq_delta_min),
                premium_delta_min=float(config.t1_model_rescue_premium_delta_min),
                candidate_fl_delta_min=float(config.t1_model_rescue_candidate_fl_delta_min),
                model_score_min=float(config.t1_model_rescue_model_score_min),
                model_bust_max=float(config.t1_model_rescue_model_bust_max),
                refined_delta_max=float(config.t1_model_rescue_refined_delta_max),
            )
        if (
            int(obs.turn) == 2
            and best_candidate is not None
            and float(config.t2_model_top1_rescue_fl_min) >= 0.0
            and float(config.t2_model_top1_rescue_refined_delta_max) >= 0.0
        ):
            best_candidate, t2_model_top1_rescue_details = _t2_model_top1_rescue_candidate(
                best_candidate,
                model_top1,
                fl_min=float(config.t2_model_top1_rescue_fl_min),
                refined_delta_max=float(config.t2_model_top1_rescue_refined_delta_max),
                bust_max=float(config.t2_model_top1_rescue_bust_max),
                selected_model_rank_min=int(config.t2_model_top1_rescue_selected_model_rank_min),
                secondary_fl_min=float(config.t2_model_top1_rescue2_fl_min),
                secondary_fl_max=float(config.t2_model_top1_rescue2_fl_max),
                secondary_bust_max=float(config.t2_model_top1_rescue2_bust_max),
                secondary_selected_model_rank_min=int(config.t2_model_top1_rescue2_selected_model_rank_min),
                secondary_refined_delta_max=float(config.t2_model_top1_rescue2_refined_delta_max),
            )
        if (
            int(obs.turn) == 2
            and best_candidate is not None
            and int(config.t2_model_top1_bust_rescue_selected_model_rank_min) > 0
        ):
            best_candidate, t2_model_top1_bust_rescue_details = _t2_model_top1_bust_rescue_candidate(
                best_candidate,
                model_top1,
                selected_model_rank_min=int(config.t2_model_top1_bust_rescue_selected_model_rank_min),
                refined_delta_max=float(config.t2_model_top1_bust_rescue_refined_delta_max),
                model_delta_min=float(config.t2_model_top1_bust_rescue_model_delta_min),
                bust_delta_min=float(config.t2_model_top1_bust_rescue_bust_delta_min),
                top_bust_max=float(config.t2_model_top1_bust_rescue_top_bust_max),
                top_fl_min=float(config.t2_model_top1_bust_rescue_top_fl_min),
                current_fl_min=float(config.t2_model_top1_bust_rescue_current_fl_min),
                fl_delta_min=float(config.t2_model_top1_bust_rescue_fl_delta_min),
            )
        if (
            int(obs.turn) == 2
            and best_candidate is not None
            and int(config.t2_middle_fill_bottom_shift_rescue_selected_model_rank_max) > 0
        ):
            best_candidate, t2_middle_fill_bottom_shift_rescue_details = (
                _t2_middle_fill_bottom_shift_rescue_candidate(
                    obs,
                    best_candidate,
                    refined_candidates,
                    selected_model_rank_max=int(
                        config.t2_middle_fill_bottom_shift_rescue_selected_model_rank_max
                    ),
                    challenger_model_rank_max=int(
                        config.t2_middle_fill_bottom_shift_rescue_challenger_model_rank_max
                    ),
                    model_gap_max=float(config.t2_middle_fill_bottom_shift_rescue_model_gap_max),
                    bust_delta_min=float(config.t2_middle_fill_bottom_shift_rescue_bust_delta_min),
                    fl_delta_min=float(config.t2_middle_fill_bottom_shift_rescue_fl_delta_min),
                    challenger_bust_max=float(
                        config.t2_middle_fill_bottom_shift_rescue_challenger_bust_max
                    ),
                    challenger_fl_min=float(
                        config.t2_middle_fill_bottom_shift_rescue_challenger_fl_min
                    ),
                    selected_bust_min=float(
                        config.t2_middle_fill_bottom_shift_rescue_selected_bust_min
                    ),
                )
            )
        if (
            int(obs.turn) == 2
            and best_candidate is not None
            and int(config.t2_model_rank_rescue_k) > 0
        ):
            best_candidate, t2_model_rank_rescue_details = _t2_model_rank_rescue_candidate(
                best_candidate,
                refined_candidates,
                rank_k=int(config.t2_model_rank_rescue_k),
                selected_model_rank_min=int(config.t2_model_rank_rescue_selected_model_rank_min),
                refined_delta_max=float(config.t2_model_rank_rescue_refined_delta_max),
                model_delta_min=float(config.t2_model_rank_rescue_model_delta_min),
                min_refined_score=float(config.t2_model_rank_rescue_min_refined_score),
            )
    else:
        fallback_policy = str(config.t1_no_refine_fallback_policy) if int(obs.turn) == 1 else "model"
        best_candidate = _t1_no_refine_fallback_candidate(
            sync_candidates,
            output_candidates,
            policy=fallback_policy,
        )
        if best_candidate is not None:
            t1_no_refine_fallback_action_idx = int(best_candidate["action_idx"])
            t1_no_refine_fallback_applied = (
                int(obs.turn) == 1
                and fallback_policy != "model"
                and bool(sync_candidates)
            )
    model_top1_action_idx = int(model_top1["action_idx"]) if model_top1 is not None else None
    best_action_idx = int(best_candidate["action_idx"]) if best_candidate is not None else None
    model_top1_refined_score = (
        float(model_top1["refined_score"])
        if model_top1 is not None and model_top1.get("refined_score") is not None
        else None
    )
    best_refined_score = (
        float(best_candidate["refined_score"])
        if best_candidate is not None and best_candidate.get("refined_score") is not None
        else None
    )
    for candidate in output_candidates:
        candidate.pop("_action", None)
        candidate.pop("_list_index", None)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "turn": int(obs.turn),
        "mode": config.mode,
        "estimated": True,
        "exact_partial": exact_evaluated > 0 and config.t2_refinement == "exact_partial",
        "refinement_method": config.t2_refinement if int(obs.turn) == 2 else config.t1_refinement,
        "refinement_backend": refinement_backend,
        "refinement_evaluated": exact_evaluated,
        "refinement_samples_by_action": refinement_sample_counts,
        "t3_union_summary": t3_union_summary,
        "time_budget_ms": int(config.time_budget_ms),
        "shortlist_k": int(config.shortlist_k),
        "effective_shortlist_k": int(shortlist_config.shortlist_k),
        "insurance_k": int(config.insurance_k),
        "sync_exact_k": int(refinement_config.sync_exact_k),
        "base_sync_exact_k": int(config.sync_exact_k),
        "t2_sync_exact_k": int(config.t2_sync_exact_k),
        "t2_initial_samples_per_candidate": int(config.t2_initial_samples_per_candidate),
        "t2_extra_samples_per_round": int(config.t2_extra_samples_per_round),
        "t2_max_samples_per_candidate": int(config.t2_max_samples_per_candidate),
        "t2_close_candidate_limit": int(config.t2_close_candidate_limit),
        "t2_model_rank_min_samples_k": int(config.t2_model_rank_min_samples_k),
        "t2_model_rank_min_samples": int(config.t2_model_rank_min_samples),
        "t2_model_rank_min_samples_min_remaining_ms": int(
            config.t2_model_rank_min_samples_min_remaining_ms
        ),
        "t2_adaptive_sync_applied": bool(t2_adaptive_sync_applied),
        "t2_adaptive_shortlist_k": int(config.t2_adaptive_shortlist_k),
        "t2_adaptive_sync_exact_k": int(config.t2_adaptive_sync_exact_k),
        "t2_adaptive_max_model_score": config.t2_adaptive_max_model_score,
        "t2_post_refine_sync_exact_k": int(config.t2_post_refine_sync_exact_k),
        "t2_post_refine_min_remaining_ms": int(config.t2_post_refine_min_remaining_ms),
        "t2_post_refine_policy": str(config.t2_post_refine_policy),
        "t2_post_refine_extra_model_score_min": float(config.t2_post_refine_extra_model_score_min),
        "t2_post_refine_expanded": bool(t2_post_refine_expanded),
        "t2_post_refine_candidate_count": int(t2_post_refine_candidate_count),
        "t2_post_refine_error": t2_post_refine_error,
        "t2_post_refine_policy_details": t2_post_refine_policy_details,
        "t2_post_refine_skipped_reason": t2_post_refine_skipped_reason,
        "t2_exact_backend": str(config.t2_exact_backend),
        "t3_pool_config": str(config.t3_pool_config),
        "t3_pool_k": int(config.t3_pool_k),
        "t1_adaptive_sync_exact_k": int(config.t1_adaptive_sync_exact_k),
        "t1_adaptive_model_rank_k": int(config.t1_adaptive_model_rank_k),
        "t1_sync_model_insurance_k": int(config.t1_sync_model_insurance_k),
        "t1_sync_tactical_insurance_k": int(config.t1_sync_tactical_insurance_k),
        "t2_sync_model_insurance_k": int(config.t2_sync_model_insurance_k),
        "t2_sync_model_insurance_top1_kk_min": float(config.t2_sync_model_insurance_top1_kk_min),
        "t2_tactical_insurance_k": int(config.t2_tactical_insurance_k),
        "t1_aux_shortlist_k": int(config.t1_aux_shortlist_k),
        "t2_aux_shortlist_k": int(config.t2_aux_shortlist_k),
        "t1_extra_refine_top_k": int(config.t1_extra_refine_top_k),
        "t1_extra_refine_margin": float(config.t1_extra_refine_margin),
        "t1_extra_refine_sims": int(config.t1_extra_refine_sims),
        "t1_refinement_first_batch_k": int(config.t1_refinement_first_batch_k),
        "t1_refinement_tail_min_remaining_ms": int(config.t1_refinement_tail_min_remaining_ms),
        "t1_refinement_timeout_headroom_ms": int(config.t1_refinement_timeout_headroom_ms),
        "t1_sync_selection_policy": str(config.t1_sync_selection_policy),
        "t1_sync_selector_name": str((config.t1_sync_selector or {}).get("name", "")),
        "t2_sync_selection_policy": str(config.t2_sync_selection_policy),
        "t2_sync_selector_name": str((config.t2_sync_selector or {}).get("name", "")),
        "t2_selection_policy": str(config.t2_selection_policy),
        "t2_final_selector_name": str((config.t2_final_selector or {}).get("name", "")),
        "t2_selection_model_weight": float(config.t2_selection_model_weight),
        "t2_selection_structured_top_model_weight": float(
            config.t2_selection_structured_top_model_weight
        ),
        "t2_own_top_structured": bool(_t2_own_top_structured(obs)) if int(obs.turn) == 2 else False,
        "t2_model_top1_rescue_fl_min": float(config.t2_model_top1_rescue_fl_min),
        "t2_model_top1_rescue_bust_max": float(config.t2_model_top1_rescue_bust_max),
        "t2_model_top1_rescue_selected_model_rank_min": int(
            config.t2_model_top1_rescue_selected_model_rank_min
        ),
        "t2_model_top1_rescue_refined_delta_max": float(
            config.t2_model_top1_rescue_refined_delta_max
        ),
        "t2_model_top1_rescue2_fl_min": float(config.t2_model_top1_rescue2_fl_min),
        "t2_model_top1_rescue2_fl_max": float(config.t2_model_top1_rescue2_fl_max),
        "t2_model_top1_rescue2_bust_max": float(config.t2_model_top1_rescue2_bust_max),
        "t2_model_top1_rescue2_selected_model_rank_min": int(
            config.t2_model_top1_rescue2_selected_model_rank_min
        ),
        "t2_model_top1_rescue2_refined_delta_max": float(
            config.t2_model_top1_rescue2_refined_delta_max
        ),
        "t2_model_top1_bust_rescue_selected_model_rank_min": int(
            config.t2_model_top1_bust_rescue_selected_model_rank_min
        ),
        "t2_model_top1_bust_rescue_refined_delta_max": float(
            config.t2_model_top1_bust_rescue_refined_delta_max
        ),
        "t2_model_top1_bust_rescue_model_delta_min": float(
            config.t2_model_top1_bust_rescue_model_delta_min
        ),
        "t2_model_top1_bust_rescue_bust_delta_min": float(
            config.t2_model_top1_bust_rescue_bust_delta_min
        ),
        "t2_model_top1_bust_rescue_top_bust_max": float(
            config.t2_model_top1_bust_rescue_top_bust_max
        ),
        "t2_model_top1_bust_rescue_top_fl_min": float(
            config.t2_model_top1_bust_rescue_top_fl_min
        ),
        "t2_model_top1_bust_rescue_current_fl_min": float(
            config.t2_model_top1_bust_rescue_current_fl_min
        ),
        "t2_model_top1_bust_rescue_fl_delta_min": float(
            config.t2_model_top1_bust_rescue_fl_delta_min
        ),
        "t2_middle_fill_bottom_shift_rescue_selected_model_rank_max": int(
            config.t2_middle_fill_bottom_shift_rescue_selected_model_rank_max
        ),
        "t2_middle_fill_bottom_shift_rescue_challenger_model_rank_max": int(
            config.t2_middle_fill_bottom_shift_rescue_challenger_model_rank_max
        ),
        "t2_middle_fill_bottom_shift_rescue_model_gap_max": float(
            config.t2_middle_fill_bottom_shift_rescue_model_gap_max
        ),
        "t2_middle_fill_bottom_shift_rescue_bust_delta_min": float(
            config.t2_middle_fill_bottom_shift_rescue_bust_delta_min
        ),
        "t2_middle_fill_bottom_shift_rescue_fl_delta_min": float(
            config.t2_middle_fill_bottom_shift_rescue_fl_delta_min
        ),
        "t2_middle_fill_bottom_shift_rescue_challenger_bust_max": float(
            config.t2_middle_fill_bottom_shift_rescue_challenger_bust_max
        ),
        "t2_middle_fill_bottom_shift_rescue_challenger_fl_min": float(
            config.t2_middle_fill_bottom_shift_rescue_challenger_fl_min
        ),
        "t2_middle_fill_bottom_shift_rescue_selected_bust_min": float(
            config.t2_middle_fill_bottom_shift_rescue_selected_bust_min
        ),
        "t2_model_rank_rescue_k": int(config.t2_model_rank_rescue_k),
        "t2_model_rank_rescue_selected_model_rank_min": int(
            config.t2_model_rank_rescue_selected_model_rank_min
        ),
        "t2_model_rank_rescue_refined_delta_max": float(
            config.t2_model_rank_rescue_refined_delta_max
        ),
        "t2_model_rank_rescue_model_delta_min": float(
            config.t2_model_rank_rescue_model_delta_min
        ),
        "t2_model_rank_rescue_min_refined_score": float(
            config.t2_model_rank_rescue_min_refined_score
        ),
        "max_pool": int(config.max_pool),
        "t1_override_margin": float(config.t1_override_margin),
        "t1_selection_policy": str(config.t1_selection_policy),
        "t1_selection_model_weight": float(config.t1_selection_model_weight),
        "t1_arbitration_policy": str(config.t1_arbitration_policy),
        "t1_arbitration_selector_name": str((config.t1_arbitration_selector or {}).get("name", "")),
        "t1_arbitration_challenger_selector_name": str(
            (config.t1_arbitration_challenger_selector or {}).get("name", "")
        ),
        "t1_arbitration_dual_gate_policy": str(config.t1_arbitration_dual_gate_policy),
        "t1_final_challenger_selector_name": str(
            (config.t1_final_challenger_selector or {}).get("name", "")
        ),
        "t1_final_challenger_gate_policy": str(config.t1_final_challenger_gate_policy),
        "t1_refined_challenger_bust_max": float(config.t1_refined_challenger_bust_max),
        "t1_refined_challenger_refined_delta_max": float(
            config.t1_refined_challenger_refined_delta_max
        ),
        "t1_blend_challenger_weight": float(config.t1_blend_challenger_weight),
        "t1_blend_challenger_bust_min": float(config.t1_blend_challenger_bust_min),
        "t1_blend_challenger_premium_fl_delta_min": float(
            config.t1_blend_challenger_premium_fl_delta_min
        ),
        "t1_low_risk_blend_challenger_weight": float(config.t1_low_risk_blend_challenger_weight),
        "t1_low_risk_blend_challenger_fl_max": float(config.t1_low_risk_blend_challenger_fl_max),
        "t1_low_risk_blend_challenger_bust_max": float(config.t1_low_risk_blend_challenger_bust_max),
        "t1_model_rescue_qq_delta_min": float(config.t1_model_rescue_qq_delta_min),
        "t1_model_rescue_premium_delta_min": float(config.t1_model_rescue_premium_delta_min),
        "t1_model_rescue_candidate_fl_delta_min": float(
            config.t1_model_rescue_candidate_fl_delta_min
        ),
        "t1_model_rescue_model_score_min": float(config.t1_model_rescue_model_score_min),
        "t1_model_rescue_model_bust_max": float(config.t1_model_rescue_model_bust_max),
        "t1_model_rescue_refined_delta_max": float(config.t1_model_rescue_refined_delta_max),
        "t1_final_bottom_sparse_rescue_margin": float(config.t1_final_bottom_sparse_rescue_margin),
        "t1_no_refine_fallback_policy": str(config.t1_no_refine_fallback_policy),
        "t1_final_selector_name": str((config.t1_final_selector or {}).get("name", "")),
        "legal_actions": len(valid_actions),
        "candidate_pool_size": len(output_candidates),
        "exact_evaluated": exact_evaluated,
        "sync_refinement_candidate_count": len(sync_candidates),
        "t1_extra_refinement_candidate_count": t1_extra_refinement_candidate_count,
        "sync_exact_action_indices": sorted(sync_action_indices),
        "model_top1_action_idx": model_top1_action_idx,
        "best_action_idx": best_action_idx,
        "model_top1_refined_score": model_top1_refined_score,
        "best_refined_score": best_refined_score,
        "refined_override_delta": (
            best_refined_score - model_top1_refined_score
            if best_refined_score is not None and model_top1_refined_score is not None
            else None
        ),
        "t1_override_features": t1_override_features,
        "t1_override_gate_probability": t1_override_gate_probability,
        "t1_override_gate_accepted": t1_override_gate_accepted,
        "t1_override_reject_reason": t1_override_reject_reason,
        "t1_arbitration_details": t1_arbitration_details,
        "t1_arbitration_applied": bool((t1_arbitration_details or {}).get("accepted", False)),
        "t1_final_challenger_details": t1_final_challenger_details,
        "t1_final_challenger_applied": bool((t1_final_challenger_details or {}).get("accepted", False)),
        "t1_refined_challenger_details": t1_refined_challenger_details,
        "t1_refined_challenger_applied": bool(
            (t1_refined_challenger_details or {}).get("accepted", False)
        ),
        "t1_blend_challenger_details": t1_blend_challenger_details,
        "t1_blend_challenger_applied": bool(
            (t1_blend_challenger_details or {}).get("accepted", False)
        ),
        "t1_low_risk_blend_challenger_details": t1_low_risk_blend_challenger_details,
        "t1_low_risk_blend_challenger_applied": bool(
            (t1_low_risk_blend_challenger_details or {}).get("accepted", False)
        ),
        "t1_model_rescue_details": t1_model_rescue_details,
        "t1_model_rescue_applied": bool((t1_model_rescue_details or {}).get("accepted", False)),
        "t2_model_top1_rescue_details": t2_model_top1_rescue_details,
        "t2_model_top1_rescue_applied": bool(
            (t2_model_top1_rescue_details or {}).get("accepted", False)
        ),
        "t2_model_top1_bust_rescue_details": t2_model_top1_bust_rescue_details,
        "t2_model_top1_bust_rescue_applied": bool(
            (t2_model_top1_bust_rescue_details or {}).get("accepted", False)
        ),
        "t2_middle_fill_bottom_shift_rescue_details": t2_middle_fill_bottom_shift_rescue_details,
        "t2_middle_fill_bottom_shift_rescue_applied": bool(
            (t2_middle_fill_bottom_shift_rescue_details or {}).get("accepted", False)
        ),
        "t2_model_rank_rescue_details": t2_model_rank_rescue_details,
        "t2_model_rank_rescue_applied": bool(
            (t2_model_rank_rescue_details or {}).get("accepted", False)
        ),
        "t1_final_bottom_sparse_rescue_applied": t1_final_bottom_sparse_rescue_applied,
        "t1_final_bottom_sparse_rescue_action_idx": t1_final_bottom_sparse_rescue_action_idx,
        "t1_final_bottom_sparse_rescue_delta": t1_final_bottom_sparse_rescue_delta,
        "t1_no_refine_fallback_applied": t1_no_refine_fallback_applied,
        "t1_no_refine_fallback_action_idx": t1_no_refine_fallback_action_idx,
        "model_top1_overridden": bool(
            exact_evaluated > 0
            and model_top1_action_idx is not None
            and best_action_idx is not None
            and best_action_idx != model_top1_action_idx
        ),
        "elapsed_ms": elapsed_ms,
        "refinement_error": refinement_error,
        "t1_extra_refinement_error": t1_extra_refinement_error,
        "best": best_candidate,
        "candidates": output_candidates,
    }


def _iter_jsonl(path: Path, limit: int = 0) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for count, line in enumerate(f, start=1):
            if limit > 0 and count > limit:
                break
            if line.strip():
                yield json.loads(line)


def parse_turn_model_specs(specs: Iterable[str]) -> dict[int, str]:
    out: dict[int, str] = {}
    for raw in specs:
        if not raw:
            continue
        if "=" not in raw:
            raise ValueError(f"turn model must use TURN=PATH: {raw}")
        turn_raw, path = raw.split("=", 1)
        turn = int(turn_raw.strip())
        if not path.strip():
            raise ValueError(f"empty model path for turn {turn}")
        out[turn] = path.strip()
    return out


def parse_turn_model_blend_specs(specs: Iterable[str]) -> dict[int, tuple[str, str, float]]:
    out: dict[int, tuple[str, str, float]] = {}
    for raw in specs:
        if not raw:
            continue
        if "=" not in raw:
            raise ValueError(f"turn model blend must use TURN=PATH_A,PATH_B,WEIGHT_B: {raw}")
        turn_raw, spec = raw.split("=", 1)
        turn = int(turn_raw.strip())
        parts = [part.strip() for part in spec.split(",")]
        if len(parts) != 3 or not parts[0] or not parts[1] or not parts[2]:
            raise ValueError(f"turn model blend must use TURN=PATH_A,PATH_B,WEIGHT_B: {raw}")
        weight = float(parts[2])
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"turn model blend weight must be between 0 and 1: {raw}")
        out[turn] = (parts[0], parts[1], weight)
    return out


def parse_turn_model_ensemble_specs(specs: Iterable[str]) -> dict[int, tuple[list[str], list[float]]]:
    out: dict[int, tuple[list[str], list[float]]] = {}
    for raw in specs:
        if not raw:
            continue
        if "=" not in raw:
            raise ValueError(f"turn model ensemble must use TURN=PATHS;WEIGHTS: {raw}")
        turn_raw, spec = raw.split("=", 1)
        turn = int(turn_raw.strip())
        if ";" in spec:
            paths_raw, weights_raw = spec.split(";", 1)
            paths = [part.strip() for part in paths_raw.split(",") if part.strip()]
            weights = [float(part.strip()) for part in weights_raw.split(",") if part.strip()]
        else:
            paths = []
            weights = []
            for part in [part.strip() for part in spec.split(",") if part.strip()]:
                if "@" not in part:
                    raise ValueError(f"turn model ensemble item must use PATH@WEIGHT: {raw}")
                path, weight_raw = part.rsplit("@", 1)
                if not path.strip() or not weight_raw.strip():
                    raise ValueError(f"turn model ensemble item must use PATH@WEIGHT: {raw}")
                paths.append(path.strip())
                weights.append(float(weight_raw))
        if len(paths) < 2:
            raise ValueError(f"turn model ensemble needs at least two paths: {raw}")
        if len(paths) != len(weights):
            raise ValueError(f"turn model ensemble paths and weights differ: {raw}")
        if any(weight < 0.0 for weight in weights):
            raise ValueError(f"turn model ensemble weights must be non-negative: {raw}")
        if sum(weights) <= 0.0:
            raise ValueError(f"turn model ensemble weights must sum positive: {raw}")
        out[turn] = (paths, weights)
    return out


def parse_turn_model_conditional_ensemble_specs(
    specs: Iterable[str],
) -> dict[int, tuple[list[str], list[float], str, float, str]]:
    out: dict[int, tuple[list[str], list[float], str, float, str]] = {}
    for raw in specs:
        if not raw:
            continue
        if "=" not in raw:
            raise ValueError(
                "turn conditional ensemble must use TURN=PATHS;WEIGHTS;SPECIALIST;WEIGHT;GATE: "
                f"{raw}"
            )
        turn_raw, spec = raw.split("=", 1)
        turn = int(turn_raw.strip())
        parts = [part.strip() for part in spec.split(";")]
        if len(parts) != 5:
            raise ValueError(
                "turn conditional ensemble must use TURN=PATHS;WEIGHTS;SPECIALIST;WEIGHT;GATE: "
                f"{raw}"
            )
        paths = [part.strip() for part in parts[0].split(",") if part.strip()]
        weights = [float(part.strip()) for part in parts[1].split(",") if part.strip()]
        specialist_path = parts[2]
        specialist_weight = float(parts[3])
        gate = parts[4] or "target_like_strict_state"
        if len(paths) < 1:
            raise ValueError(f"turn conditional ensemble needs at least one base path: {raw}")
        if len(paths) != len(weights):
            raise ValueError(f"turn conditional ensemble paths and weights differ: {raw}")
        if any(weight < 0.0 for weight in weights):
            raise ValueError(f"turn conditional ensemble base weights must be non-negative: {raw}")
        if sum(weights) <= 0.0:
            raise ValueError(f"turn conditional ensemble base weights must sum positive: {raw}")
        if not specialist_path:
            raise ValueError(f"turn conditional ensemble specialist path is empty: {raw}")
        if not 0.0 <= specialist_weight <= 1.0:
            raise ValueError(f"turn conditional ensemble specialist weight must be 0..1: {raw}")
        out[turn] = (paths, weights, specialist_path, specialist_weight, gate)
    return out


def parse_turn_model_conditional_switch_ensemble_specs(
    specs: Iterable[str],
) -> dict[int, tuple[list[str], list[float], list[str], list[float], str]]:
    out: dict[int, tuple[list[str], list[float], list[str], list[float], str]] = {}
    for raw in specs:
        if not raw:
            continue
        if "=" not in raw:
            raise ValueError(
                "turn conditional switch ensemble must use "
                "TURN=BASE_PATHS;BASE_WEIGHTS;CHALLENGER_PATHS;CHALLENGER_WEIGHTS;GATE: "
                f"{raw}"
            )
        turn_raw, spec = raw.split("=", 1)
        turn = int(turn_raw.strip())
        parts = [part.strip() for part in spec.split(";")]
        if len(parts) != 5:
            raise ValueError(
                "turn conditional switch ensemble must use "
                "TURN=BASE_PATHS;BASE_WEIGHTS;CHALLENGER_PATHS;CHALLENGER_WEIGHTS;GATE: "
                f"{raw}"
            )
        base_paths = [part.strip() for part in parts[0].split(",") if part.strip()]
        base_weights = [float(part.strip()) for part in parts[1].split(",") if part.strip()]
        challenger_paths = [part.strip() for part in parts[2].split(",") if part.strip()]
        challenger_weights = [float(part.strip()) for part in parts[3].split(",") if part.strip()]
        gate = parts[4] or "target_like_strict_state"
        if len(base_paths) < 1 or len(challenger_paths) < 1:
            raise ValueError(f"turn conditional switch ensemble paths cannot be empty: {raw}")
        if len(base_paths) != len(base_weights):
            raise ValueError(f"turn conditional switch base paths and weights differ: {raw}")
        if len(challenger_paths) != len(challenger_weights):
            raise ValueError(f"turn conditional switch challenger paths and weights differ: {raw}")
        if any(weight < 0.0 for weight in [*base_weights, *challenger_weights]):
            raise ValueError(f"turn conditional switch ensemble weights must be non-negative: {raw}")
        if sum(base_weights) <= 0.0 or sum(challenger_weights) <= 0.0:
            raise ValueError(f"turn conditional switch ensemble weights must sum positive: {raw}")
        out[turn] = (base_paths, base_weights, challenger_paths, challenger_weights, gate)
    return out


def parse_turn_model_cascade_switch_ensemble_specs(
    specs: Iterable[str],
) -> dict[int, tuple[list[str], list[float], list[tuple[str, list[str], list[float]]]]]:
    out: dict[int, tuple[list[str], list[float], list[tuple[str, list[str], list[float]]]]] = {}
    for raw in specs:
        if not raw:
            continue
        if "=" not in raw:
            raise ValueError(
                "turn cascade switch ensemble must use "
                "TURN=BASE_PATHS;BASE_WEIGHTS;GATE|PATHS|WEIGHTS[;GATE|PATHS|WEIGHTS...]: "
                f"{raw}"
            )
        turn_raw, spec = raw.split("=", 1)
        turn = int(turn_raw.strip())
        parts = [part.strip() for part in spec.split(";") if part.strip()]
        if len(parts) < 3:
            raise ValueError(
                "turn cascade switch ensemble must use "
                "TURN=BASE_PATHS;BASE_WEIGHTS;GATE|PATHS|WEIGHTS[;GATE|PATHS|WEIGHTS...]: "
                f"{raw}"
            )
        base_paths = [part.strip() for part in parts[0].split(",") if part.strip()]
        base_weights = [float(part.strip()) for part in parts[1].split(",") if part.strip()]
        if len(base_paths) < 1:
            raise ValueError(f"turn cascade switch base paths cannot be empty: {raw}")
        if len(base_paths) != len(base_weights):
            raise ValueError(f"turn cascade switch base paths and weights differ: {raw}")
        if any(weight < 0.0 for weight in base_weights) or sum(base_weights) <= 0.0:
            raise ValueError(f"turn cascade switch base weights must be non-negative and sum positive: {raw}")
        switches: list[tuple[str, list[str], list[float]]] = []
        for switch_part in parts[2:]:
            switch_bits = [part.strip() for part in switch_part.split("|")]
            if len(switch_bits) != 3:
                raise ValueError(f"cascade switch item must use GATE|PATHS|WEIGHTS: {raw}")
            gate = switch_bits[0] or "all"
            paths = [part.strip() for part in switch_bits[1].split(",") if part.strip()]
            weights = [float(part.strip()) for part in switch_bits[2].split(",") if part.strip()]
            if len(paths) < 1:
                raise ValueError(f"cascade switch paths cannot be empty: {raw}")
            if len(paths) != len(weights):
                raise ValueError(f"cascade switch paths and weights differ: {raw}")
            if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
                raise ValueError(f"cascade switch weights must be non-negative and sum positive: {raw}")
            switches.append((gate, paths, weights))
        out[turn] = (base_paths, base_weights, switches)
    return out


def _option_dest(option: str) -> str:
    name = option.split("=", 1)[0]
    return name.lstrip("-").replace("-", "_")


def _provided_cli_dests(argv: Iterable[str]) -> set[str]:
    return {_option_dest(token) for token in argv if token.startswith("--")}


def _runtime_config_mode_payload(payload: dict[str, Any], mode_name: str | None = None) -> tuple[str, dict[str, Any]]:
    selected = mode_name or str(payload.get("default_mode") or "")
    if not selected:
        raise ValueError("Runtime config must define default_mode or --runtime-mode")

    modes: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict) and (key.startswith("k") or key.endswith("_mode")):
            modes[key] = value
        elif isinstance(value, dict) and key not in {"models", "optional_modes"}:
            if any(
                mode_key in value
                for mode_key in (
                    "inherits",
                    "shortlist_k",
                    "t1_refinement",
                    "t2_refinement",
                    "time_budget_ms",
                )
            ):
                modes[key] = value
    optional_modes = payload.get("optional_modes")
    if isinstance(optional_modes, dict):
        modes.update({str(key): value for key, value in optional_modes.items() if isinstance(value, dict)})
    if selected not in modes and isinstance(payload.get(selected), dict):
        modes[selected] = payload[selected]

    if selected not in modes:
        raise ValueError(f"Runtime mode not found in config: {selected}")

    resolving: set[str] = set()

    def resolve_mode(name: str) -> dict[str, Any]:
        if name in resolving:
            chain = " -> ".join([*resolving, name])
            raise ValueError(f"Runtime mode inheritance cycle: {chain}")
        if name not in modes:
            raise ValueError(f"Runtime mode not found in config: {name}")
        resolving.add(name)
        current = dict(modes[name])
        inherits = str(current.pop("inherits", "") or "")
        if inherits:
            inherited = resolve_mode(inherits)
            inherited.update(current)
            current = inherited
        resolving.remove(name)
        return current

    return selected, resolve_mode(selected)


def _apply_runtime_default(args: argparse.Namespace, provided: set[str], name: str, value: Any) -> None:
    if name in provided or not hasattr(args, name):
        return
    setattr(args, name, value)


def _turn_model_ensemble_defaults(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    out: list[str] = []
    for turn_raw, spec in payload.items():
        if not isinstance(spec, dict):
            raise ValueError(f"turn_model_ensembles entry must be an object: {turn_raw}")
        paths = spec.get("checkpoints") or spec.get("paths")
        weights = spec.get("weights")
        if not isinstance(paths, list) or not isinstance(weights, list):
            raise ValueError(f"turn_model_ensembles entry needs checkpoints and weights: {turn_raw}")
        if len(paths) != len(weights) or len(paths) < 2:
            raise ValueError(f"turn_model_ensembles checkpoints/weights mismatch: {turn_raw}")
        path_part = ",".join(str(path) for path in paths)
        weight_part = ",".join(str(weight) for weight in weights)
        out.append(f"{int(turn_raw)}={path_part};{weight_part}")
    return out


def _turn_model_conditional_ensemble_defaults(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    out: list[str] = []
    for turn_raw, spec in payload.items():
        if not isinstance(spec, dict):
            raise ValueError(f"turn_model_conditional_ensembles entry must be an object: {turn_raw}")
        paths = spec.get("checkpoints") or spec.get("paths") or spec.get("base_checkpoints")
        weights = spec.get("weights") or spec.get("base_weights")
        specialist = spec.get("specialist") or spec.get("specialist_checkpoint")
        specialist_weight = spec.get("specialist_weight")
        gate = str(spec.get("gate") or "target_like_strict_state")
        if not isinstance(paths, list) or not isinstance(weights, list):
            raise ValueError(
                f"turn_model_conditional_ensembles entry needs checkpoints and weights: {turn_raw}"
            )
        if len(paths) != len(weights) or len(paths) < 1:
            raise ValueError(
                f"turn_model_conditional_ensembles checkpoints/weights mismatch: {turn_raw}"
            )
        if specialist in (None, "") or specialist_weight is None:
            raise ValueError(
                f"turn_model_conditional_ensembles entry needs specialist and specialist_weight: {turn_raw}"
            )
        path_part = ",".join(str(path) for path in paths)
        weight_part = ",".join(str(weight) for weight in weights)
        out.append(f"{int(turn_raw)}={path_part};{weight_part};{specialist};{specialist_weight};{gate}")
    return out


def _turn_model_conditional_switch_ensemble_defaults(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    out: list[str] = []
    for turn_raw, spec in payload.items():
        if not isinstance(spec, dict):
            raise ValueError(f"turn_model_conditional_switch_ensembles entry must be an object: {turn_raw}")
        base_paths = spec.get("base_checkpoints") or spec.get("base_paths")
        base_weights = spec.get("base_weights")
        challenger_paths = spec.get("challenger_checkpoints") or spec.get("challenger_paths")
        challenger_weights = spec.get("challenger_weights")
        gate = str(spec.get("gate") or "target_like_strict_state")
        if not isinstance(base_paths, list) or not isinstance(base_weights, list):
            raise ValueError(
                f"turn_model_conditional_switch_ensembles entry needs base checkpoints and weights: {turn_raw}"
            )
        if not isinstance(challenger_paths, list) or not isinstance(challenger_weights, list):
            raise ValueError(
                f"turn_model_conditional_switch_ensembles entry needs challenger checkpoints and weights: {turn_raw}"
            )
        if len(base_paths) != len(base_weights) or len(base_paths) < 1:
            raise ValueError(
                f"turn_model_conditional_switch_ensembles base checkpoints/weights mismatch: {turn_raw}"
            )
        if len(challenger_paths) != len(challenger_weights) or len(challenger_paths) < 1:
            raise ValueError(
                f"turn_model_conditional_switch_ensembles challenger checkpoints/weights mismatch: {turn_raw}"
            )
        base_path_part = ",".join(str(path) for path in base_paths)
        base_weight_part = ",".join(str(weight) for weight in base_weights)
        challenger_path_part = ",".join(str(path) for path in challenger_paths)
        challenger_weight_part = ",".join(str(weight) for weight in challenger_weights)
        out.append(
            f"{int(turn_raw)}={base_path_part};{base_weight_part};"
            f"{challenger_path_part};{challenger_weight_part};{gate}"
        )
    return out


def _turn_model_cascade_switch_ensemble_defaults(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    out: list[str] = []
    for turn_raw, spec in payload.items():
        if not isinstance(spec, dict):
            raise ValueError(f"turn_model_cascade_switch_ensembles entry must be an object: {turn_raw}")
        base_paths = spec.get("base_checkpoints") or spec.get("base_paths")
        base_weights = spec.get("base_weights")
        switches = spec.get("switches")
        if not isinstance(base_paths, list) or not isinstance(base_weights, list):
            raise ValueError(
                f"turn_model_cascade_switch_ensembles entry needs base checkpoints and weights: {turn_raw}"
            )
        if len(base_paths) != len(base_weights) or len(base_paths) < 1:
            raise ValueError(
                f"turn_model_cascade_switch_ensembles base checkpoints/weights mismatch: {turn_raw}"
            )
        if not isinstance(switches, list) or not switches:
            raise ValueError(f"turn_model_cascade_switch_ensembles entry needs switches: {turn_raw}")
        base_path_part = ",".join(str(path) for path in base_paths)
        base_weight_part = ",".join(str(weight) for weight in base_weights)
        switch_parts: list[str] = []
        for switch in switches:
            if not isinstance(switch, dict):
                raise ValueError(f"cascade switch entry must be an object: {turn_raw}")
            gate = str(switch.get("gate") or "all")
            paths = switch.get("checkpoints") or switch.get("paths") or switch.get("challenger_checkpoints")
            weights = switch.get("weights") or switch.get("challenger_weights")
            if not isinstance(paths, list) or not isinstance(weights, list):
                raise ValueError(f"cascade switch needs checkpoints and weights: {turn_raw}")
            if len(paths) != len(weights) or len(paths) < 1:
                raise ValueError(f"cascade switch checkpoints/weights mismatch: {turn_raw}")
            path_part = ",".join(str(path) for path in paths)
            weight_part = ",".join(str(weight) for weight in weights)
            switch_parts.append(f"{gate}|{path_part}|{weight_part}")
        out.append(f"{int(turn_raw)}={base_path_part};{base_weight_part};" + ";".join(switch_parts))
    return out


def apply_hybrid_runtime_config(
    args: argparse.Namespace,
    *,
    provided_dests: set[str],
) -> tuple[str, dict[str, Any]] | None:
    runtime_config = str(getattr(args, "runtime_config", "") or "")
    if not runtime_config:
        return None

    path = Path(runtime_config)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Runtime config must be a JSON object: {path}")
    mode_name, mode = _runtime_config_mode_payload(payload, str(getattr(args, "runtime_mode", "") or "") or None)

    models = payload.get("models")
    mode_models = mode.get("models")
    if isinstance(mode_models, dict):
        merged_models: dict[str, Any] = {}
        if isinstance(models, dict):
            merged_models.update(models)
        merged_models.update(mode_models)
        models = merged_models
    if isinstance(models, dict):
        model_defaults = {
            "model": models.get("action_value") or models.get("t1_action_value") or models.get("t2_action_value"),
            "t1_aux_shortlist_model": models.get("t1_aux_shortlist"),
            "t2_aux_shortlist_model": models.get("t2_aux_shortlist"),
            "t1_sync_selector": models.get("t1_sync_selector"),
            "t2_sync_selector": models.get("t2_sync_selector"),
            "t2_final_selector": models.get("t2_final_selector"),
            "t1_final_selector": models.get("t1_final_selector"),
            "t1_override_gate": models.get("t1_override_gate"),
            "t1_arbitration_selector": models.get("t1_arbitration_selector"),
            "t1_arbitration_challenger_selector": models.get("t1_arbitration_challenger_selector"),
            "t1_final_challenger_selector": models.get("t1_final_challenger_selector"),
            "t1_final_challenger_gate": models.get("t1_final_challenger_gate"),
        }
        turn_model_ensembles = _turn_model_ensemble_defaults(
            models.get("turn_model_ensembles") or models.get("action_value_ensembles_by_turn")
        )
        if turn_model_ensembles:
            model_defaults["turn_model_ensemble"] = turn_model_ensembles
        turn_model_conditional_ensembles = _turn_model_conditional_ensemble_defaults(
            models.get("turn_model_conditional_ensembles")
            or models.get("conditional_action_value_ensembles_by_turn")
        )
        if turn_model_conditional_ensembles:
            model_defaults["turn_model_conditional_ensemble"] = turn_model_conditional_ensembles
        turn_model_conditional_switch_ensembles = _turn_model_conditional_switch_ensemble_defaults(
            models.get("turn_model_conditional_switch_ensembles")
            or models.get("conditional_switch_action_value_ensembles_by_turn")
        )
        if turn_model_conditional_switch_ensembles:
            model_defaults["turn_model_conditional_switch_ensemble"] = turn_model_conditional_switch_ensembles
        turn_model_cascade_switch_ensembles = _turn_model_cascade_switch_ensemble_defaults(
            models.get("turn_model_cascade_switch_ensembles")
            or models.get("cascade_switch_action_value_ensembles_by_turn")
        )
        if turn_model_cascade_switch_ensembles:
            model_defaults["turn_model_cascade_switch_ensemble"] = turn_model_cascade_switch_ensembles
        for name, value in model_defaults.items():
            if value not in (None, ""):
                _apply_runtime_default(args, provided_dests, name, value)

    for key, value in mode.items():
        if key in {"use_case", "note", "description", "models"}:
            continue
        _apply_runtime_default(args, provided_dests, str(key), value)
    return mode_name, mode


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run T1/T2 hybrid model shortlist + optional exact refinement")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--runtime-config",
        default="",
        help="Optional JSON runtime config. Values are used as defaults unless the same CLI option is provided.",
    )
    parser.add_argument(
        "--runtime-mode",
        default="",
        help="Optional mode inside --runtime-config. Defaults to the config's default_mode.",
    )
    parser.add_argument(
        "--turn-model",
        action="append",
        default=[],
        help="Optional per-turn model override, e.g. 2=path/to/action_value_best.pt",
    )
    parser.add_argument(
        "--turn-model-blend",
        action="append",
        default=[],
        help="Optional per-turn two-model blend, e.g. 3=old.pt,new.pt,0.25. Overrides --turn-model for that turn.",
    )
    parser.add_argument(
        "--turn-model-ensemble",
        action="append",
        default=[],
        help=(
            "Optional per-turn weighted model ensemble, e.g. "
            "'2=old.pt,ft204.pt,ft284.pt;0.50,0.15,0.35'. "
            "Overrides --turn-model and --turn-model-blend for that turn."
        ),
    )
    parser.add_argument(
        "--turn-model-conditional-ensemble",
        action="append",
        default=[],
        help=(
            "Optional per-turn conditional specialist ensemble, e.g. "
            "'2=base1.pt,base2.pt;0.7,0.3;specialist.pt;0.35;target_like_strict_state'. "
            "Overrides --turn-model, --turn-model-blend, and --turn-model-ensemble for that turn."
        ),
    )
    parser.add_argument(
        "--turn-model-conditional-switch-ensemble",
        action="append",
        default=[],
        help=(
            "Optional per-turn conditional ensemble switch, e.g. "
            "'2=base1.pt,base2.pt;0.7,0.3;challenger1.pt,challenger2.pt;0.6,0.4;opp_top_len_le_1_context'. "
            "Overrides other turn model specs for that turn."
        ),
    )
    parser.add_argument(
        "--turn-model-cascade-switch-ensemble",
        action="append",
        default=[],
        help=(
            "Optional ordered per-turn ensemble switches, e.g. "
            "'2=base1.pt,base2.pt;0.7,0.3;gate_a|a.pt|1.0;gate_b|b.pt|1.0'. "
            "Later switches override earlier switches for matching candidates."
        ),
    )
    parser.add_argument(
        "--t1-aux-shortlist-model",
        default="",
        help="Optional T1-only auxiliary model used only to add candidates to the shortlist.",
    )
    parser.add_argument("--t1-aux-shortlist-k", type=int, default=0)
    parser.add_argument(
        "--t2-aux-shortlist-model",
        default="",
        help="Optional T2-only auxiliary model used only to add candidates to the shortlist.",
    )
    parser.add_argument("--t2-aux-shortlist-k", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--turns", default="1,2")
    parser.add_argument("--shortlist-k", type=int, default=15)
    parser.add_argument("--insurance-k", type=int, default=5)
    parser.add_argument("--sync-exact-k", type=int, default=3)
    parser.add_argument("--t2-sync-exact-k", type=int, default=0)
    parser.add_argument("--t1-adaptive-sync-exact-k", type=int, default=0)
    parser.add_argument("--t1-adaptive-model-rank-k", type=int, default=0)
    parser.add_argument("--t1-sync-model-insurance-k", type=int, default=0)
    parser.add_argument("--t1-sync-tactical-insurance-k", type=int, default=0)
    parser.add_argument("--t2-sync-model-insurance-k", type=int, default=0)
    parser.add_argument("--t2-sync-model-insurance-top1-kk-min", type=float, default=0.0)
    parser.add_argument("--t2-tactical-insurance-k", type=int, default=0)
    parser.add_argument("--max-pool", type=int, default=20)
    parser.add_argument("--time-budget-ms", type=int, default=5000)
    parser.add_argument("--t1-refinement", default="none", choices=["none", "mc_board", "recursive_mc"])
    parser.add_argument("--t1-mc-sims", type=int, default=32)
    parser.add_argument("--t1-recursive-beam", type=int, default=5)
    parser.add_argument("--t1-recursive-child-sims", type=int, default=2)
    parser.add_argument("--t1-refinement-first-batch-k", type=int, default=0)
    parser.add_argument("--t1-refinement-tail-min-remaining-ms", type=int, default=0)
    parser.add_argument("--t1-refinement-timeout-headroom-ms", type=int, default=0)
    parser.add_argument("--t1-extra-refine-top-k", type=int, default=0)
    parser.add_argument("--t1-extra-refine-margin", type=float, default=0.0)
    parser.add_argument("--t1-extra-refine-sims", type=int, default=0)
    parser.add_argument("--t1-sync-selection-policy", default="rank", choices=["rank", "selector"])
    parser.add_argument("--t1-sync-selector", default="")
    parser.add_argument("--t2-sync-selection-policy", default="rank", choices=["rank", "selector"])
    parser.add_argument("--t2-sync-selector", default="")
    parser.add_argument("--t2-selection-policy", default="refined_score", choices=["refined_score", "refined_plus_model", "selector"])
    parser.add_argument("--t2-selection-model-weight", type=float, default=0.0)
    parser.add_argument("--t2-selection-structured-top-model-weight", type=float, default=-1.0)
    parser.add_argument("--t2-model-top1-rescue-fl-min", type=float, default=-1.0)
    parser.add_argument("--t2-model-top1-rescue-bust-max", type=float, default=-1.0)
    parser.add_argument("--t2-model-top1-rescue-selected-model-rank-min", type=int, default=0)
    parser.add_argument("--t2-model-top1-rescue-refined-delta-max", type=float, default=-1.0)
    parser.add_argument("--t2-model-top1-rescue2-fl-min", type=float, default=-1.0)
    parser.add_argument("--t2-model-top1-rescue2-fl-max", type=float, default=-1.0)
    parser.add_argument("--t2-model-top1-rescue2-bust-max", type=float, default=-1.0)
    parser.add_argument("--t2-model-top1-rescue2-selected-model-rank-min", type=int, default=0)
    parser.add_argument("--t2-model-top1-rescue2-refined-delta-max", type=float, default=-1.0)
    parser.add_argument("--t2-model-top1-bust-rescue-selected-model-rank-min", type=int, default=0)
    parser.add_argument("--t2-model-top1-bust-rescue-refined-delta-max", type=float, default=-1.0)
    parser.add_argument("--t2-model-top1-bust-rescue-model-delta-min", type=float, default=-1.0)
    parser.add_argument("--t2-model-top1-bust-rescue-bust-delta-min", type=float, default=-1.0)
    parser.add_argument("--t2-model-top1-bust-rescue-top-bust-max", type=float, default=-1.0)
    parser.add_argument("--t2-model-top1-bust-rescue-top-fl-min", type=float, default=-1.0)
    parser.add_argument("--t2-model-top1-bust-rescue-current-fl-min", type=float, default=-1.0)
    parser.add_argument("--t2-model-top1-bust-rescue-fl-delta-min", type=float, default=-1.0)
    parser.add_argument("--t2-middle-fill-bottom-shift-rescue-selected-model-rank-max", type=int, default=0)
    parser.add_argument("--t2-middle-fill-bottom-shift-rescue-challenger-model-rank-max", type=int, default=0)
    parser.add_argument("--t2-middle-fill-bottom-shift-rescue-model-gap-max", type=float, default=-1.0)
    parser.add_argument("--t2-middle-fill-bottom-shift-rescue-bust-delta-min", type=float, default=-1.0)
    parser.add_argument("--t2-middle-fill-bottom-shift-rescue-fl-delta-min", type=float, default=-1.0)
    parser.add_argument("--t2-middle-fill-bottom-shift-rescue-challenger-bust-max", type=float, default=-1.0)
    parser.add_argument("--t2-middle-fill-bottom-shift-rescue-challenger-fl-min", type=float, default=-1.0)
    parser.add_argument("--t2-middle-fill-bottom-shift-rescue-selected-bust-min", type=float, default=-1.0)
    parser.add_argument("--t2-model-rank-rescue-k", type=int, default=0)
    parser.add_argument("--t2-model-rank-rescue-selected-model-rank-min", type=int, default=0)
    parser.add_argument("--t2-model-rank-rescue-refined-delta-max", type=float, default=-1.0)
    parser.add_argument("--t2-model-rank-rescue-model-delta-min", type=float, default=-1.0)
    parser.add_argument("--t2-model-rank-rescue-min-refined-score", type=float, default=-1.0)
    parser.add_argument("--t1-selection-policy", default="refined_score", choices=["refined_score", "refined_plus_model", "selector"])
    parser.add_argument("--t1-selection-model-weight", type=float, default=0.0)
    parser.add_argument("--t1-final-selector", default="")
    parser.add_argument("--t2-final-selector", default="")
    parser.add_argument("--t1-arbitration-selector", default="")
    parser.add_argument("--t1-arbitration-challenger-selector", default="")
    parser.add_argument(
        "--t1-arbitration-policy",
        default="none",
        choices=[
            "none",
            "current_refined_rank_ge2_and_qq_nonnegative",
            "pairwise_refined_delta_ge_m086_current_bust_ge_0026",
            "refined_score_model_rank_ge6_conflict_ge_m086",
            "mixed_kk_delta_le_0043_model_delta_le_1261",
            "mixed_kk_delta_le_0043_model_delta_le_1111_conflict_ge_m0483",
            "remain85_current_margin_le_0664",
        ],
    )
    parser.add_argument(
        "--t1-arbitration-dual-gate-policy",
        default="none",
        choices=["none", "new_if_kk_and_premium_delta_nonpositive"],
    )
    parser.add_argument("--t1-final-challenger-selector", default="")
    parser.add_argument("--t1-final-challenger-gate", default="")
    parser.add_argument(
        "--t1-final-challenger-gate-policy",
        default="none",
        choices=[
            "none",
            "guard2_bust_delta_ge_m066_rank_gap_delta_ge_m1",
            "guard2_bust_rank_refined_delta_ge_m0778",
            "guard2_bust_delta_ge_m066",
            "logistic_gate",
            "logistic_gate_refined_override_delta_ge_0677",
        ],
    )
    parser.add_argument("--t1-refined-challenger-bust-max", type=float, default=-1.0)
    parser.add_argument("--t1-refined-challenger-refined-delta-max", type=float, default=-1.0)
    parser.add_argument("--t1-blend-challenger-weight", type=float, default=-1.0)
    parser.add_argument("--t1-blend-challenger-bust-min", type=float, default=-1.0)
    parser.add_argument("--t1-blend-challenger-premium-fl-delta-min", type=float, default=-1.0)
    parser.add_argument("--t1-low-risk-blend-challenger-weight", type=float, default=-1.0)
    parser.add_argument("--t1-low-risk-blend-challenger-fl-max", type=float, default=-1.0)
    parser.add_argument("--t1-low-risk-blend-challenger-bust-max", type=float, default=-1.0)
    parser.add_argument("--t1-model-rescue-qq-delta-min", type=float, default=-1.0)
    parser.add_argument("--t1-model-rescue-premium-delta-min", type=float, default=-1.0)
    parser.add_argument("--t1-model-rescue-candidate-fl-delta-min", type=float, default=-1.0)
    parser.add_argument("--t1-model-rescue-model-score-min", type=float, default=-1.0)
    parser.add_argument("--t1-model-rescue-model-bust-max", type=float, default=-1.0)
    parser.add_argument("--t1-model-rescue-refined-delta-max", type=float, default=-1.0)
    parser.add_argument("--t1-final-bottom-sparse-rescue-margin", type=float, default=-1.0)
    parser.add_argument(
        "--t1-no-refine-fallback-policy",
        default="model",
        choices=["model", "fl_safe", "low_bust", "model_bust"],
    )
    parser.add_argument("--t1-override-margin", type=float, default=0.0)
    parser.add_argument("--t1-override-gate", default="")
    parser.add_argument("--t1-override-gate-threshold", type=float, default=0.5)
    parser.add_argument("--t2-initial-samples-per-candidate", type=int, default=3)
    parser.add_argument("--t2-extra-samples-per-round", type=int, default=2)
    parser.add_argument("--t2-max-samples-per-candidate", type=int, default=24)
    parser.add_argument("--t2-close-candidate-limit", type=int, default=4)
    parser.add_argument("--t2-baseline-min-samples", type=int, default=3)
    parser.add_argument("--t2-model-rank-min-samples-k", type=int, default=0)
    parser.add_argument("--t2-model-rank-min-samples", type=int, default=0)
    parser.add_argument("--t2-model-rank-min-samples-min-remaining-ms", type=int, default=0)
    parser.add_argument("--t2-adaptive-shortlist-k", type=int, default=0)
    parser.add_argument("--t2-adaptive-sync-exact-k", type=int, default=0)
    parser.add_argument("--t2-adaptive-max-model-score", type=float, default=None)
    parser.add_argument("--t2-post-refine-sync-exact-k", type=int, default=0)
    parser.add_argument("--t2-post-refine-min-remaining-ms", type=int, default=0)
    parser.add_argument(
        "--t2-post-refine-policy",
        default="always",
        choices=["always", "structured_top_model_min"],
    )
    parser.add_argument("--t2-post-refine-extra-model-score-min", type=float, default=-1.0)
    parser.add_argument("--t2-refinement", default="mc_board", choices=["exact_partial", "mc_board"])
    parser.add_argument(
        "--t2-exact-backend",
        default="full_exact",
        choices=["full_exact", "t3_union"],
        help="Backend for --t2-refinement exact_partial. t3_union uses model TopK plus Rust exact rerank at T3.",
    )
    parser.add_argument("--t3-pool-config", default="ai/config/t3_ev_loss_fresh_pool_20260607.json")
    parser.add_argument("--t3-pool-k", type=int, default=10)
    parser.add_argument("--t2-mc-sims", type=int, default=300)
    parser.add_argument("--disable-refinement", action="store_true")
    parser.add_argument("--rust-solver", default="")
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(raw_argv)
    provided_dests = _provided_cli_dests(raw_argv)
    runtime_config_result = apply_hybrid_runtime_config(args, provided_dests=provided_dests)
    if runtime_config_result is not None:
        runtime_mode, _runtime_mode_config = runtime_config_result
        args.runtime_mode = runtime_mode
    if not args.model:
        parser.error("--model is required unless supplied by --runtime-config")

    turns = {int(part) for part in args.turns.split(",") if part.strip()}
    config = HybridConfig(
        shortlist_k=args.shortlist_k,
        insurance_k=args.insurance_k,
        sync_exact_k=args.sync_exact_k,
        t2_sync_exact_k=args.t2_sync_exact_k,
        t1_adaptive_sync_exact_k=args.t1_adaptive_sync_exact_k,
        t1_adaptive_model_rank_k=args.t1_adaptive_model_rank_k,
        t1_sync_model_insurance_k=args.t1_sync_model_insurance_k,
        t1_sync_tactical_insurance_k=args.t1_sync_tactical_insurance_k,
        t2_sync_model_insurance_k=args.t2_sync_model_insurance_k,
        t2_sync_model_insurance_top1_kk_min=args.t2_sync_model_insurance_top1_kk_min,
        t2_tactical_insurance_k=args.t2_tactical_insurance_k,
        t1_aux_shortlist_k=args.t1_aux_shortlist_k,
        t2_aux_shortlist_k=args.t2_aux_shortlist_k,
        max_pool=args.max_pool,
        time_budget_ms=args.time_budget_ms,
        t1_refinement=args.t1_refinement,
        t1_mc_sims=args.t1_mc_sims,
        t1_recursive_beam=args.t1_recursive_beam,
        t1_recursive_child_sims=args.t1_recursive_child_sims,
        t1_refinement_first_batch_k=args.t1_refinement_first_batch_k,
        t1_refinement_tail_min_remaining_ms=args.t1_refinement_tail_min_remaining_ms,
        t1_refinement_timeout_headroom_ms=args.t1_refinement_timeout_headroom_ms,
        t1_extra_refine_top_k=args.t1_extra_refine_top_k,
        t1_extra_refine_margin=args.t1_extra_refine_margin,
        t1_extra_refine_sims=args.t1_extra_refine_sims,
        t1_sync_selection_policy=args.t1_sync_selection_policy,
        t1_sync_selector=load_t1_sync_selector(args.t1_sync_selector),
        t2_sync_selection_policy=args.t2_sync_selection_policy,
        t2_sync_selector=load_t2_sync_selector(args.t2_sync_selector),
        t2_selection_policy=args.t2_selection_policy,
        t2_selection_model_weight=args.t2_selection_model_weight,
        t2_selection_structured_top_model_weight=args.t2_selection_structured_top_model_weight,
        t2_final_selector=load_t2_sync_selector(args.t2_final_selector),
        t2_model_top1_rescue_fl_min=args.t2_model_top1_rescue_fl_min,
        t2_model_top1_rescue_bust_max=args.t2_model_top1_rescue_bust_max,
        t2_model_top1_rescue_selected_model_rank_min=args.t2_model_top1_rescue_selected_model_rank_min,
        t2_model_top1_rescue_refined_delta_max=args.t2_model_top1_rescue_refined_delta_max,
        t2_model_top1_rescue2_fl_min=args.t2_model_top1_rescue2_fl_min,
        t2_model_top1_rescue2_fl_max=args.t2_model_top1_rescue2_fl_max,
        t2_model_top1_rescue2_bust_max=args.t2_model_top1_rescue2_bust_max,
        t2_model_top1_rescue2_selected_model_rank_min=args.t2_model_top1_rescue2_selected_model_rank_min,
        t2_model_top1_rescue2_refined_delta_max=args.t2_model_top1_rescue2_refined_delta_max,
        t2_model_top1_bust_rescue_selected_model_rank_min=args.t2_model_top1_bust_rescue_selected_model_rank_min,
        t2_model_top1_bust_rescue_refined_delta_max=args.t2_model_top1_bust_rescue_refined_delta_max,
        t2_model_top1_bust_rescue_model_delta_min=args.t2_model_top1_bust_rescue_model_delta_min,
        t2_model_top1_bust_rescue_bust_delta_min=args.t2_model_top1_bust_rescue_bust_delta_min,
        t2_model_top1_bust_rescue_top_bust_max=args.t2_model_top1_bust_rescue_top_bust_max,
        t2_model_top1_bust_rescue_top_fl_min=args.t2_model_top1_bust_rescue_top_fl_min,
        t2_model_top1_bust_rescue_current_fl_min=args.t2_model_top1_bust_rescue_current_fl_min,
        t2_model_top1_bust_rescue_fl_delta_min=args.t2_model_top1_bust_rescue_fl_delta_min,
        t2_middle_fill_bottom_shift_rescue_selected_model_rank_max=(
            args.t2_middle_fill_bottom_shift_rescue_selected_model_rank_max
        ),
        t2_middle_fill_bottom_shift_rescue_challenger_model_rank_max=(
            args.t2_middle_fill_bottom_shift_rescue_challenger_model_rank_max
        ),
        t2_middle_fill_bottom_shift_rescue_model_gap_max=(
            args.t2_middle_fill_bottom_shift_rescue_model_gap_max
        ),
        t2_middle_fill_bottom_shift_rescue_bust_delta_min=(
            args.t2_middle_fill_bottom_shift_rescue_bust_delta_min
        ),
        t2_middle_fill_bottom_shift_rescue_fl_delta_min=(
            args.t2_middle_fill_bottom_shift_rescue_fl_delta_min
        ),
        t2_middle_fill_bottom_shift_rescue_challenger_bust_max=(
            args.t2_middle_fill_bottom_shift_rescue_challenger_bust_max
        ),
        t2_middle_fill_bottom_shift_rescue_challenger_fl_min=(
            args.t2_middle_fill_bottom_shift_rescue_challenger_fl_min
        ),
        t2_middle_fill_bottom_shift_rescue_selected_bust_min=(
            args.t2_middle_fill_bottom_shift_rescue_selected_bust_min
        ),
        t2_model_rank_rescue_k=args.t2_model_rank_rescue_k,
        t2_model_rank_rescue_selected_model_rank_min=args.t2_model_rank_rescue_selected_model_rank_min,
        t2_model_rank_rescue_refined_delta_max=args.t2_model_rank_rescue_refined_delta_max,
        t2_model_rank_rescue_model_delta_min=args.t2_model_rank_rescue_model_delta_min,
        t2_model_rank_rescue_min_refined_score=args.t2_model_rank_rescue_min_refined_score,
        t1_selection_policy=args.t1_selection_policy,
        t1_selection_model_weight=args.t1_selection_model_weight,
        t1_final_selector=load_t1_final_selector(args.t1_final_selector),
        t1_arbitration_selector=load_t1_final_selector(args.t1_arbitration_selector),
        t1_arbitration_challenger_selector=load_t1_final_selector(args.t1_arbitration_challenger_selector),
        t1_arbitration_policy=args.t1_arbitration_policy,
        t1_arbitration_dual_gate_policy=args.t1_arbitration_dual_gate_policy,
        t1_final_challenger_selector=load_t1_final_selector(args.t1_final_challenger_selector),
        t1_final_challenger_gate=load_t1_runtime_gate(args.t1_final_challenger_gate),
        t1_final_challenger_gate_policy=args.t1_final_challenger_gate_policy,
        t1_refined_challenger_bust_max=args.t1_refined_challenger_bust_max,
        t1_refined_challenger_refined_delta_max=args.t1_refined_challenger_refined_delta_max,
        t1_blend_challenger_weight=args.t1_blend_challenger_weight,
        t1_blend_challenger_bust_min=args.t1_blend_challenger_bust_min,
        t1_blend_challenger_premium_fl_delta_min=args.t1_blend_challenger_premium_fl_delta_min,
        t1_low_risk_blend_challenger_weight=args.t1_low_risk_blend_challenger_weight,
        t1_low_risk_blend_challenger_fl_max=args.t1_low_risk_blend_challenger_fl_max,
        t1_low_risk_blend_challenger_bust_max=args.t1_low_risk_blend_challenger_bust_max,
        t1_model_rescue_qq_delta_min=args.t1_model_rescue_qq_delta_min,
        t1_model_rescue_premium_delta_min=args.t1_model_rescue_premium_delta_min,
        t1_model_rescue_candidate_fl_delta_min=args.t1_model_rescue_candidate_fl_delta_min,
        t1_model_rescue_model_score_min=args.t1_model_rescue_model_score_min,
        t1_model_rescue_model_bust_max=args.t1_model_rescue_model_bust_max,
        t1_model_rescue_refined_delta_max=args.t1_model_rescue_refined_delta_max,
        t1_final_bottom_sparse_rescue_margin=args.t1_final_bottom_sparse_rescue_margin,
        t1_no_refine_fallback_policy=args.t1_no_refine_fallback_policy,
        t1_override_margin=args.t1_override_margin,
        t1_override_gate=load_t1_override_gate(args.t1_override_gate),
        t1_override_gate_threshold=args.t1_override_gate_threshold,
        t2_initial_samples_per_candidate=args.t2_initial_samples_per_candidate,
        t2_extra_samples_per_round=args.t2_extra_samples_per_round,
        t2_max_samples_per_candidate=args.t2_max_samples_per_candidate,
        t2_close_candidate_limit=args.t2_close_candidate_limit,
        t2_baseline_min_samples=args.t2_baseline_min_samples,
        t2_model_rank_min_samples_k=args.t2_model_rank_min_samples_k,
        t2_model_rank_min_samples=args.t2_model_rank_min_samples,
        t2_model_rank_min_samples_min_remaining_ms=args.t2_model_rank_min_samples_min_remaining_ms,
        t2_adaptive_shortlist_k=args.t2_adaptive_shortlist_k,
        t2_adaptive_sync_exact_k=args.t2_adaptive_sync_exact_k,
        t2_adaptive_max_model_score=args.t2_adaptive_max_model_score,
        t2_post_refine_sync_exact_k=args.t2_post_refine_sync_exact_k,
        t2_post_refine_min_remaining_ms=args.t2_post_refine_min_remaining_ms,
        t2_post_refine_policy=args.t2_post_refine_policy,
        t2_post_refine_extra_model_score_min=args.t2_post_refine_extra_model_score_min,
        t2_refinement=args.t2_refinement,
        t2_exact_backend=args.t2_exact_backend,
        t3_pool_config=args.t3_pool_config,
        t3_pool_k=args.t3_pool_k,
        t2_mc_sims=args.t2_mc_sims,
        enable_sync_refinement=not args.disable_refinement,
    )
    evaluator = make_action_value_evaluator(
        model_path=args.model,
        model_paths_by_turn=parse_turn_model_specs(args.turn_model),
        model_blends_by_turn=parse_turn_model_blend_specs(args.turn_model_blend),
        model_ensembles_by_turn=parse_turn_model_ensemble_specs(args.turn_model_ensemble),
        model_conditional_ensembles_by_turn=parse_turn_model_conditional_ensemble_specs(
            args.turn_model_conditional_ensemble
        ),
        model_conditional_switch_ensembles_by_turn=parse_turn_model_conditional_switch_ensemble_specs(
            args.turn_model_conditional_switch_ensemble
        ),
        model_cascade_switch_ensembles_by_turn=parse_turn_model_cascade_switch_ensemble_specs(
            args.turn_model_cascade_switch_ensemble
        ),
        device=args.device or None,
    )
    aux_model_specs: list[str] = []
    aux_base_model = ""
    if args.t1_aux_shortlist_model and args.t1_aux_shortlist_k > 0:
        aux_base_model = args.t1_aux_shortlist_model
        aux_model_specs.append(f"1={args.t1_aux_shortlist_model}")
    if args.t2_aux_shortlist_model and args.t2_aux_shortlist_k > 0:
        aux_base_model = aux_base_model or args.t2_aux_shortlist_model
        aux_model_specs.append(f"2={args.t2_aux_shortlist_model}")
    shortlist_evaluator = (
        make_action_value_evaluator(
            model_path=aux_base_model,
            model_paths_by_turn=parse_turn_model_specs(aux_model_specs),
            device=args.device or None,
        )
        if aux_base_model
        else None
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", encoding="utf-8") as f:
        for payload in _iter_jsonl(Path(args.input), args.limit):
            if int(payload.get("turn", -1)) not in turns:
                continue
            result = evaluate_hybrid_position(
                payload,
                evaluator=evaluator,
                shortlist_evaluator=shortlist_evaluator,
                config=config,
                rust_solver_path=args.rust_solver or None,
            )
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            written += 1
    print(json.dumps({"output": str(output), "written": written}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
