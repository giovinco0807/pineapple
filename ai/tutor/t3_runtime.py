"""Runtime T3 candidate pooling plus exact reranking.

The service path for late-turn T3 should not trust a model Top1 directly.  This
module builds a union candidate pool from independent action-value and set
rerankers, then sends only that pool to the Rust exact solver.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.action_space import Action, encode_action, get_turn_actions
from ai.engine.encoding import Board, Observation, encode_state
from ai.engine.turn_order import normalize_position, validate_decision_board_counts
from ai.models.action_value_reranker import ActionValueReranker
from ai.models.action_value_set_reranker import ActionValueSetReranker
from ai.training.action_feature_encoding import adapt_np_state_with_action
from ai.tutor.exact_late import (
    action_key,
    action_to_dict,
    apply_action,
    board_card_count,
    board_to_dict,
    evaluate_late_position,
    normalize_position_payload,
)

DEFAULT_CONFIG = Path("ai/config/t3_ev_loss_fresh_pool_20260607.json")
FL_TYPE_KEYS = ("qq", "kk", "aa", "trips")
RANK_VALUE = {rank: value for value, rank in enumerate("23456789TJQKA", start=2)}


def _check_deadline(deadline_s: float | None, stage: str, *, min_remaining_s: float = 0.0) -> None:
    if deadline_s is None:
        return
    remaining = float(deadline_s) - time.perf_counter()
    if remaining <= float(min_remaining_s):
        raise TimeoutError(f"{stage}:time_budget_exhausted")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _resolve_device(device: str | None) -> torch.device:
    if not device or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _iter_jsonl(path: Path, limit: int = 0) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for count, line in enumerate(f, start=1):
            if limit > 0 and count > limit:
                break
            if line.strip():
                yield json.loads(line)


def _position_from_payload(payload: dict[str, Any]) -> str:
    raw = payload.get("position") or payload.get("player_position")
    is_btn = payload["is_btn"] if "is_btn" in payload else None
    return normalize_position(raw, is_btn=is_btn)


def _row_name(row: Any) -> str:
    return {"mid": "middle", "bot": "bottom"}.get(str(row), str(row))


def _rank_value(card: Any) -> int:
    text = str(card or "")
    if text.startswith("X"):
        return 15
    return RANK_VALUE.get(text[:1], 0)


def _low_card_score(card: Any) -> float:
    value = _rank_value(card)
    if value <= 0:
        return 0.0
    return max(0.0, 15.0 - float(value))


def _insurance_priority(action_payload: dict[str, Any]) -> float:
    placements = [(str(card), _row_name(row)) for card, row in (action_payload.get("placements") or [])]
    discard = action_payload.get("discard")
    row_counts = {"top": 0, "middle": 0, "bottom": 0}
    row_low = {"top": 0.0, "middle": 0.0, "bottom": 0.0}
    row_high = {"top": 0.0, "middle": 0.0, "bottom": 0.0}
    for card, row in placements:
        if row not in row_counts:
            continue
        value = _rank_value(card)
        row_counts[row] += 1
        row_low[row] += _low_card_score(card)
        row_high[row] += max(0.0, float(value) - 10.0)

    priority = 0.0
    if row_counts["top"] == 0:
        priority += 100.0
    if row_counts["bottom"] == 2:
        priority += 35.0
    if row_counts["middle"] == 2:
        priority += 12.0
    priority += row_low["bottom"] * 2.0
    priority += row_low["middle"]
    priority -= row_high["top"] * 4.0
    priority -= row_counts["top"] * 8.0
    priority += max(0.0, _rank_value(discard) - 11.0) * 0.5
    return priority


def _high_top_anchor_priority(action_payload: dict[str, Any]) -> float:
    placements = [(str(card), _row_name(row)) for card, row in (action_payload.get("placements") or [])]
    top_values = [_rank_value(card) for card, row in placements if row == "top"]
    side_cards = [(card, row) for card, row in placements if row in {"middle", "bottom"}]
    if not top_values or not side_cards or max(top_values) < 11:
        return 0.0
    priority = 100.0 + max(top_values) * 2.0
    for card, row in side_cards:
        if row == "bottom":
            priority += 8.0
        priority += _low_card_score(card) * 1.5
    return priority


def _low_top_side_priority(action_payload: dict[str, Any]) -> float:
    placements = [(str(card), _row_name(row)) for card, row in (action_payload.get("placements") or [])]
    top_values = [_rank_value(card) for card, row in placements if row == "top"]
    side_cards = [(card, row) for card, row in placements if row in {"middle", "bottom"}]
    if not top_values or not side_cards or min(top_values) > 8:
        return 0.0
    priority = 100.0 + sum(max(0.0, 9.0 - float(value)) for value in top_values)
    for card, row in side_cards:
        value = _rank_value(card)
        if row == "bottom":
            priority += 5.0
        priority += max(0.0, float(value) - 8.0) * 1.5
    return priority


def _pair_middle_bottom_priority(action_payload: dict[str, Any]) -> float:
    placements = [(str(card), _row_name(row)) for card, row in (action_payload.get("placements") or [])]
    if len(placements) != 2:
        return 0.0
    if {row for _card, row in placements} != {"middle", "bottom"}:
        return 0.0
    return 100.0 + sum(_low_card_score(card) for card, _row in placements)


def _select_insurance_indices(action_payloads: list[dict[str, Any]], count: int, mode: str = "general") -> list[int]:
    if count <= 0:
        return []
    if mode != "category":
        return [
            int(local_idx)
            for local_idx in sorted(
                range(len(action_payloads)),
                key=lambda idx: (-_insurance_priority(action_payloads[int(idx)]), action_key(action_payloads[int(idx)])),
            )[:count]
        ]

    selected: list[int] = []
    selected_set: set[int] = set()
    categories = (
        _insurance_priority,
        _high_top_anchor_priority,
        _low_top_side_priority,
        _pair_middle_bottom_priority,
    )
    for scorer in categories:
        if len(selected) >= count:
            break
        ranked = sorted(
            range(len(action_payloads)),
            key=lambda local_idx: (-scorer(action_payloads[int(local_idx)]), action_key(action_payloads[int(local_idx)])),
        )
        for local_idx in ranked:
            local_idx = int(local_idx)
            if local_idx in selected_set:
                continue
            if scorer(action_payloads[local_idx]) <= 0.0:
                break
            selected.append(local_idx)
            selected_set.add(local_idx)
            break

    if len(selected) < count:
        for local_idx in sorted(
            range(len(action_payloads)),
            key=lambda idx: (-_insurance_priority(action_payloads[int(idx)]), action_key(action_payloads[int(idx)])),
        ):
            local_idx = int(local_idx)
            if local_idx in selected_set:
                continue
            selected.append(local_idx)
            selected_set.add(local_idx)
            if len(selected) >= count:
                break
    return selected


def _post_action_obs(obs: Observation, action: Action) -> Observation:
    new_board = obs.board_self.copy()
    for card, row in action.placements:
        getattr(new_board, row).append(card)

    unavailable: list[str] = []
    seen = set(new_board.all_cards())
    seen.update(obs.board_opponent.all_cards())
    for card in list(obs.known_discards_self) + obs.board_opponent.all_cards():
        if card not in seen and card not in unavailable:
            unavailable.append(card)
    if action.discard and action.discard not in unavailable:
        unavailable.append(action.discard)

    return Observation(
        board_self=new_board,
        board_opponent=obs.board_opponent,
        dealt_cards=[],
        known_discards_self=unavailable,
        turn=obs.turn,
        is_btn=obs.is_btn,
        is_fl=False,
        opp_is_fl=False,
        chips_self=obs.chips_self,
        chips_opponent=obs.chips_opponent,
    )


def _score_action_value_model(
    model: ActionValueReranker,
    states: np.ndarray,
    *,
    turn: int,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            batch = torch.as_tensor(states[start : start + batch_size], dtype=torch.float32, device=device)
            if hasattr(model, "predict_components"):
                turn_tensor = torch.full((batch.shape[0],), int(turn), dtype=torch.long, device=device)
                try:
                    out = model.predict_components(batch, turn=turn_tensor)
                except TypeError:
                    out = model.predict_components(batch)
                score = out["score"]
            else:
                raw = model(batch)
                score = raw["score"] if isinstance(raw, dict) else raw.squeeze(-1)
            scores.append(score.detach().cpu().numpy())
    return np.concatenate(scores).astype(np.float32, copy=False)


def _score_set_model(
    model: ActionValueSetReranker,
    states: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        state_tensor = torch.as_tensor(states[None, :, :], dtype=torch.float32, device=device)
        mask = torch.ones((1, states.shape[0]), dtype=torch.bool, device=device)
        return model.predict_scores(state_tensor, mask).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)


@dataclass
class PositionModels:
    action_value: list[tuple[str, ActionValueReranker]]
    set_models: list[tuple[str, ActionValueSetReranker]]


class T3UnionCandidatePool:
    """Build a union pool from configured T3 candidate-source checkpoints."""

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG,
        device: str | torch.device | None = "auto",
        pool_policy: str = "fast",
    ):
        self.config_path = Path(config_path)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8-sig"))
        self.device = _resolve_device(str(device) if device is not None else "auto")
        self.pool_policy = str(pool_policy or "fast")
        self._models: dict[str, PositionModels] = {}

    def _policy_config(self) -> dict[str, Any]:
        policy = self.config.get("candidate_pool_policy") or {}
        return (
            policy.get(self.pool_policy)
            or policy.get("fast")
            or policy.get("fast_candidate")
            or {}
        )

    def default_pool_k(self) -> int:
        config = self._policy_config()
        return int(config.get("per_source_top_k", 10) or 10)

    def default_insurance_candidates(self) -> int:
        config = self._policy_config()
        return int(config.get("insurance_candidates", config.get("insurance_k", 0)) or 0)

    def default_insurance_mode(self) -> str:
        config = self._policy_config()
        return str(config.get("insurance_mode", "general") or "general")

    def default_near_full_expand_gap(self) -> int:
        config = self._policy_config()
        return int(config.get("near_full_expand_gap", 0) or 0)

    def _checkpoint_name(self, path: str) -> str:
        p = Path(path)
        return p.parent.name or p.stem

    def _load_position_models(self, position: str) -> PositionModels:
        position = str(position).lower()
        if position in self._models:
            return self._models[position]

        action_value_paths = list((self.config.get("action_value_checkpoints") or {}).get(position) or [])
        set_paths = list((self.config.get("set_checkpoints_union8") or {}).get(position) or [])
        if not action_value_paths and not set_paths:
            raise ValueError(f"No T3 candidate-source checkpoints configured for position: {position}")

        action_value: list[tuple[str, ActionValueReranker]] = []
        for path in action_value_paths:
            model = ActionValueReranker.from_checkpoint(path, map_location=self.device).to(self.device)
            model.eval()
            action_value.append((self._checkpoint_name(path), model))

        set_models: list[tuple[str, ActionValueSetReranker]] = []
        for path in set_paths:
            model = ActionValueSetReranker.from_checkpoint(path, map_location=self.device).to(self.device)
            model.eval()
            set_models.append((self._checkpoint_name(path), model))

        models = PositionModels(action_value=action_value, set_models=set_models)
        self._models[position] = models
        return models

    def build_pool(
        self,
        obs: Observation,
        *,
        position: str,
        per_source_top_k: int | None = None,
        insurance_candidates: int | None = None,
        insurance_mode: str | None = None,
        near_full_expand_gap: int | None = None,
        deadline_s: float | None = None,
    ) -> dict[str, Any]:
        if int(obs.turn) != 3:
            raise ValueError("T3UnionCandidatePool only supports turn=3")
        _check_deadline(deadline_s, "t3_pool:start", min_remaining_s=0.01)
        valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)
        indexed_actions = list(enumerate(valid_actions))
        if per_source_top_k is not None and int(per_source_top_k) <= 0:
            _check_deadline(deadline_s, "t3_pool:all_legal", min_remaining_s=0.01)
            candidates = [
                {
                    "action_idx": int(
                        encode_action(
                            action,
                            valid_actions,
                            turn=int(obs.turn),
                            dealt_cards=obs.dealt_cards,
                        )
                    ),
                    "list_index": int(action_index),
                    "action": action_to_dict(action),
                    "board": board_to_dict(apply_action(obs.board_self, action)),
                    "model_score": 0.0,
                    "best_source_rank": int(action_index + 1),
                    "source_ranks": {},
                    "source_scores": {},
                }
                for action_index, action in indexed_actions
            ]
            return {
                "position": position,
                "pool_policy": self.pool_policy,
                "per_source_top_k": int(per_source_top_k),
                "legal_actions": len(valid_actions),
                "candidate_pool_size": len(candidates),
                "sources": [
                    {
                        "name": "all_legal_actions",
                        "kind": "exact",
                        "top_k": len(candidates),
                        "selected_action_indices": [int(index) for index, _action in indexed_actions],
                    }
                ],
                "candidates": candidates,
            }
        decision = {
            "turn": int(obs.turn),
            "board": board_to_dict(obs.board_self),
            "opponent_board": board_to_dict(obs.board_opponent),
            "dealt": list(obs.dealt_cards),
            "known_discards": list(obs.known_discards_self),
            "exclude": list(obs.known_discards_self),
            "is_btn": bool(obs.is_btn),
        }
        base_states: list[np.ndarray] = []
        action_payloads: list[dict[str, Any]] = []
        state_cache: dict[int, np.ndarray] = {}
        for _idx, action in indexed_actions:
            action_payload = action_to_dict(action)
            base_states.append(np.asarray(encode_state(_post_action_obs(obs, action)), dtype=np.float32))
            action_payloads.append(action_payload)

        def states_for_model(model: Any) -> np.ndarray:
            target_dim = int(getattr(model, "input_dim", 520) or 520)
            cached = state_cache.get(target_dim)
            if cached is not None:
                return cached
            states = np.asarray(
                [
                    adapt_np_state_with_action(base_state, target_dim, decision, action_payload)
                    for base_state, action_payload in zip(base_states, action_payloads)
                ],
                dtype=np.float32,
            )
            state_cache[target_dim] = states
            return states

        states_520 = np.asarray(base_states, dtype=np.float32)
        models = self._load_position_models(position)
        k = int(per_source_top_k or self.default_pool_k())
        insurance_k = int(self.default_insurance_candidates() if insurance_candidates is None else insurance_candidates)
        insurance_mode_value = str(self.default_insurance_mode() if insurance_mode is None else insurance_mode)
        near_full_gap = int(self.default_near_full_expand_gap() if near_full_expand_gap is None else near_full_expand_gap)
        selected: set[int] = set()
        source_summaries: list[dict[str, Any]] = []
        per_candidate: list[dict[str, Any]] = [
            {"source_ranks": {}, "source_scores": {}} for _ in indexed_actions
        ]

        def add_source(name: str, kind: str, scores: np.ndarray) -> None:
            order = np.argsort(-scores)
            limit = min(k, len(order))
            for rank, local_idx in enumerate(order, start=1):
                if rank <= limit:
                    selected.add(int(local_idx))
                info = per_candidate[int(local_idx)]
                source_key = f"{kind}:{name}"
                info["source_ranks"][source_key] = int(rank)
                info["source_scores"][source_key] = float(scores[int(local_idx)])
            source_summaries.append(
                {
                    "name": name,
                    "kind": kind,
                    "top_k": limit,
                    "selected_action_indices": [int(indexed_actions[int(i)][0]) for i in order[:limit]],
                }
            )

        for name, model in models.action_value:
            _check_deadline(deadline_s, f"t3_pool:action_value:{name}", min_remaining_s=0.01)
            add_source(
                name,
                "action_value",
                _score_action_value_model(model, states_for_model(model), turn=int(obs.turn), device=self.device),
            )
        for name, model in models.set_models:
            _check_deadline(deadline_s, f"t3_pool:set:{name}", min_remaining_s=0.01)
            add_source(name, "set", _score_set_model(model, states_520, device=self.device))

        if insurance_k > 0:
            _check_deadline(deadline_s, "t3_pool:insurance", min_remaining_s=0.01)
            added = []
            for local_idx in _select_insurance_indices(action_payloads, insurance_k, mode=insurance_mode_value):
                local_idx = int(local_idx)
                if local_idx in selected:
                    continue
                selected.add(local_idx)
                added.append(local_idx)
                if len(added) >= insurance_k:
                    break
            source_summaries.append(
                {
                    "name": "shape_insurance",
                    "kind": "heuristic",
                    "mode": insurance_mode_value,
                    "top_k": int(insurance_k),
                    "selected_action_indices": [int(indexed_actions[int(i)][0]) for i in added],
                }
            )

        if near_full_gap > 0 and len(selected) >= max(0, len(indexed_actions) - near_full_gap):
            _check_deadline(deadline_s, "t3_pool:near_full_expand", min_remaining_s=0.01)
            before = len(selected)
            selected.update(range(len(indexed_actions)))
            source_summaries.append(
                {
                    "name": "near_full_expand",
                    "kind": "heuristic",
                    "gap": int(near_full_gap),
                    "added": int(len(selected) - before),
                    "selected_action_indices": [
                        int(indexed_actions[int(i)][0])
                        for i in range(len(indexed_actions))
                    ],
                }
            )

        candidates = []
        _check_deadline(deadline_s, "t3_pool:materialize", min_remaining_s=0.01)
        for local_idx in sorted(selected):
            action_index, action = indexed_actions[int(local_idx)]
            info = per_candidate[int(local_idx)]
            source_ranks = info["source_ranks"]
            source_scores = info["source_scores"]
            model_score = max(source_scores.values()) if source_scores else float("-inf")
            best_source_rank = min(source_ranks.values()) if source_ranks else 999999
            candidates.append(
                {
                    "action_idx": int(encode_action(action, valid_actions, turn=int(obs.turn), dealt_cards=obs.dealt_cards)),
                    "list_index": int(action_index),
                    "action": action_to_dict(action),
                    "board": board_to_dict(apply_action(obs.board_self, action)),
                    "model_score": float(model_score),
                    "best_source_rank": int(best_source_rank),
                    "source_ranks": source_ranks,
                    "source_scores": source_scores,
                }
            )
        candidates.sort(key=lambda item: (int(item["best_source_rank"]), -float(item["model_score"]), int(item["list_index"])))
        return {
            "position": position,
            "pool_policy": self.pool_policy,
            "per_source_top_k": k,
            "insurance_candidates": int(max(0, insurance_k)),
            "insurance_mode": insurance_mode_value,
            "near_full_expand_gap": int(max(0, near_full_gap)),
            "legal_actions": len(valid_actions),
            "candidate_pool_size": len(candidates),
            "sources": source_summaries,
            "candidates": candidates,
        }


def observation_from_t3_payload(payload: dict[str, Any]) -> tuple[Observation, Board, Board, list[str], list[str], str]:
    board, opponent, dealt, exclude = normalize_position_payload(payload)
    if board_card_count(board) != 9:
        raise ValueError(f"T3 runtime requires a 9-card board, got {board_card_count(board)}")
    if len(dealt) != 3:
        raise ValueError(f"T3 runtime requires exactly 3 dealt cards, got {len(dealt)}")
    position = _position_from_payload(payload)
    validate_decision_board_counts(
        3,
        position,
        board_card_count(board),
        board_card_count(opponent),
    )
    obs = Observation(
        board_self=board,
        board_opponent=opponent,
        dealt_cards=dealt,
        known_discards_self=exclude,
        turn=3,
        is_btn=(position == "btn"),
        is_fl=bool(payload.get("is_fl", False)),
        opp_is_fl=bool(payload.get("opp_is_fl", False)),
        chips_self=int(payload.get("chips_self", 200) or 200),
        chips_opponent=int(payload.get("chips_opponent", 200) or 200),
    )
    return obs, board, opponent, dealt, exclude, position


def evaluate_t3_position(
    payload: dict[str, Any],
    *,
    pooler: T3UnionCandidatePool | None = None,
    config_path: str | Path = DEFAULT_CONFIG,
    per_source_top_k: int | None = None,
    insurance_candidates: int | None = None,
    insurance_mode: str | None = None,
    pool_policy: str = "fast",
    near_full_expand_gap: int | None = None,
    rust_solver_path: str | Path | None = None,
    rust_timeout_s: float = 5.0,
    deadline_s: float | None = None,
    device: str | torch.device | None = "auto",
) -> dict[str, Any]:
    started = time.perf_counter()
    _check_deadline(deadline_s, "t3:start", min_remaining_s=0.01)
    obs, board, opponent, dealt, exclude, position = observation_from_t3_payload(payload)
    pooler = pooler or T3UnionCandidatePool(config_path=config_path, device=device, pool_policy=pool_policy)
    pool_started = time.perf_counter()
    pool = pooler.build_pool(
        obs,
        position=position,
        per_source_top_k=per_source_top_k,
        insurance_candidates=insurance_candidates,
        insurance_mode=insurance_mode,
        near_full_expand_gap=near_full_expand_gap,
        deadline_s=deadline_s,
    )
    pool_elapsed_ms = (time.perf_counter() - pool_started) * 1000.0
    candidate_actions = [candidate["action"] for candidate in pool["candidates"]]
    if deadline_s is not None:
        remaining = float(deadline_s) - time.perf_counter()
        if remaining <= 0.25:
            raise TimeoutError("t3_exact:time_budget_exhausted")
        rust_timeout_s = min(float(rust_timeout_s), remaining)
    exact = evaluate_late_position(
        board,
        dealt,
        3,
        opponent_board=opponent,
        exclude=exclude,
        candidate_actions=candidate_actions,
        top_n=max(1, len(candidate_actions)),
        prefer_rust=True,
        rust_solver_path=rust_solver_path,
        rust_timeout_s=rust_timeout_s,
        fallback_on_rust_error=False,
    )
    by_action = {action_key(candidate["action"]): candidate for candidate in pool["candidates"]}
    for candidate in exact.get("candidates") or []:
        model_candidate = by_action.get(action_key(candidate.get("action") or {}))
        if model_candidate is not None:
            candidate["model_pool"] = model_candidate
    if exact.get("best"):
        model_best = by_action.get(action_key((exact.get("best") or {}).get("action") or {}))
        if model_best is not None:
            exact["best"]["model_pool"] = model_best
    exact_elapsed_ms = float(exact.get("elapsed_ms", 0.0) or 0.0)
    return {
        "turn": 3,
        "mode": "t3_union_exact",
        "estimated": False,
        "exact": True,
        "exact_scope": str(exact.get("exact_scope") or "t3_self_board_all_t4_draws_best_t4"),
        "hu_exact": False,
        "position": position,
        "config": str(pooler.config_path),
        "pool_policy": str(pool.get("pool_policy", pool_policy) or pool_policy),
        "per_source_top_k": int(pool["per_source_top_k"]),
        "insurance_candidates": int(pool.get("insurance_candidates", 0) or 0),
        "insurance_mode": str(pool.get("insurance_mode", "general") or "general"),
        "near_full_expand_gap": int(pool.get("near_full_expand_gap", 0) or 0),
        "legal_actions": int(pool["legal_actions"]),
        "candidate_pool_size": int(pool["candidate_pool_size"]),
        "exact_evaluated": int(exact.get("evaluated_actions", len(candidate_actions)) or 0),
        "requested_actions": int(exact.get("requested_actions", len(candidate_actions)) or 0),
        "pool_elapsed_ms": pool_elapsed_ms,
        "exact_elapsed_ms": exact_elapsed_ms,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "rust_solver": exact.get("rust_solver"),
        "best": exact.get("best"),
        "candidates": exact.get("candidates") or [],
        "candidate_pool": pool,
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate T3 with union model pool plus Rust exact rerank")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--pool-policy", default="fast")
    parser.add_argument("--pool-k", type=int, default=0, help="Per-source TopK. Defaults to config policy.")
    parser.add_argument("--insurance-candidates", type=int, default=-1, help="Shape-insurance candidates. Defaults to config policy.")
    parser.add_argument("--insurance-mode", choices=("default", "general", "category"), default="default")
    parser.add_argument("--near-full-expand-gap", type=int, default=-1, help="Expand to all legal actions when the pool is within this gap of full. Defaults to config policy.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rust-solver", default="")
    parser.add_argument("--rust-timeout-s", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(list(argv) if argv is not None else None)

    pooler = T3UnionCandidatePool(config_path=args.config, device=args.device, pool_policy=args.pool_policy)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", encoding="utf-8") as f:
        for payload in _iter_jsonl(Path(args.input), args.limit):
            result = evaluate_t3_position(
                payload,
                pooler=pooler,
                per_source_top_k=args.pool_k or None,
                insurance_candidates=(None if int(args.insurance_candidates) < 0 else int(args.insurance_candidates)),
                insurance_mode=(None if args.insurance_mode == "default" else str(args.insurance_mode)),
                pool_policy=args.pool_policy,
                near_full_expand_gap=(None if int(args.near_full_expand_gap) < 0 else int(args.near_full_expand_gap)),
                rust_solver_path=args.rust_solver or None,
                rust_timeout_s=args.rust_timeout_s,
                device=args.device,
            )
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            written += 1
    print(json.dumps({"output": str(output), "written": written}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
