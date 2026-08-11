"""Runtime helpers for HU Turn2 Stage8 selective overrides."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .action_key import (
    ACTION_KEY_SCHEMA,
    action_key,
    action_key_from_payload,
    canonical_argmax_index,
    canonical_descending_indices,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from .action_space import Action, generate_turn_actions
from .hu_infoset import card_free_metadata
from .hu_turn3_model import sample_to_matrix
from .policy import RegularAiPolicy, action_to_json, board_to_json, policy_sample
from .state import Board
from .train_torch_action_value import parse_hidden_layers, select_device
from .turn3_model import _build_torch_mlp


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


@dataclass(frozen=True)
class HuTurn2Stage8RuntimeConfig:
    min_margin: float
    reference_min_margin: float
    gate_threshold: float
    enabled: bool = True
    min_model_score: float | None = None
    allowed_seats: tuple[str, ...] = ()
    candidate_ev_rank_max: int | None = None

    @property
    def config_id(self) -> str:
        suffix = ""
        if self.allowed_seats:
            suffix += "_seat" + "-".join(self.allowed_seats)
        if self.min_model_score is not None:
            suffix += f"_s{self.min_model_score:g}"
        if self.candidate_ev_rank_max is not None:
            suffix += f"_k{self.candidate_ev_rank_max:g}"
        return f"m{self.min_margin:g}_r{self.reference_min_margin:g}_g{self.gate_threshold:g}{suffix}"


@dataclass
class HuTurn2Stage8MultiheadModel:
    """Five-head T2 model trained by ``train_hu_turn2_pilot_model``."""

    state_dict: dict[str, Any]
    feature_dim: int
    hidden_layer_sizes: tuple[int, ...]
    dropout: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    batch_size: int = 8192
    device: str = "auto"
    _torch: Any = field(default=None, init=False, repr=False)
    _net: Any = field(default=None, init=False, repr=False)
    _net_device: str | None = field(default=None, init=False, repr=False)

    @classmethod
    def load(cls, path: str | Path, *, device: str = "auto", batch_size: int = 8192) -> "HuTurn2Stage8MultiheadModel":
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("model_kind") != "hu_turn2_pilot_multihead_mlp":
            raise TypeError(f"unsupported HU T2 model kind: {payload.get('model_kind')}")
        return cls(
            state_dict=payload["state_dict"],
            feature_dim=int(payload["feature_dim"]),
            hidden_layer_sizes=tuple(int(size) for size in payload["hidden_layer_sizes"]),
            dropout=float(payload.get("dropout", 0.0)),
            feature_mean=np.asarray(payload["feature_mean"], dtype=np.float32),
            feature_scale=np.asarray(payload["feature_scale"], dtype=np.float32),
            target_mean=np.asarray(payload["target_mean"], dtype=np.float32),
            target_scale=np.asarray(payload["target_scale"], dtype=np.float32),
            batch_size=batch_size,
            device=device,
        )

    def predict_matrix(self, features: np.ndarray) -> np.ndarray:
        torch = self._get_torch()
        device = self._select_device(torch)
        net = self._get_net(torch, device)
        array = features.astype(np.float32, copy=False)
        if array.shape[1] != self.feature_dim:
            if array.shape[1] > self.feature_dim:
                array = array[:, : self.feature_dim]
            else:
                padding = np.zeros((array.shape[0], self.feature_dim - array.shape[1]), dtype=np.float32)
                array = np.hstack([array, padding])
        outputs: list[np.ndarray] = []
        inference_guard = getattr(torch, "inference_mode", torch.no_grad)
        with inference_guard():
            for start in range(0, array.shape[0], self.batch_size):
                end = min(start + self.batch_size, array.shape[0])
                batch = (array[start:end] - self.feature_mean) / self.feature_scale
                tensor = torch.from_numpy(batch.astype(np.float32, copy=False)).to(device)
                out = net(tensor).detach().cpu().numpy().astype(np.float32)
                out[:, :4] = out[:, :4] * self.target_scale + self.target_mean
                outputs.append(out.astype(np.float64, copy=False))
        return np.vstack(outputs) if outputs else np.zeros((0, 5), dtype=np.float64)

    def predict_sample(self, sample: dict[str, Any]) -> np.ndarray:
        features, _targets = sample_to_matrix(sample)
        return self.predict_matrix(features)

    def choose_action_index(self, sample: dict[str, Any]) -> int:
        predictions = self.predict_sample(sample)
        values = predictions[:, 0]
        best = float(np.max(values))
        tied = [index for index, value in enumerate(values) if float(value) == best]
        return min(
            tied,
            key=lambda index: action_key_from_payload(sample["actions"][index]).sort_key(),
        )

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
                output_dim=5,
            )
            net.load_state_dict(self.state_dict)
            net.to(device)
            net.eval()
            self._net = net
            self._net_device = device
        return self._net


def load_hu_turn2_stage8_model(
    path: str | Path,
    *,
    device: str = "auto",
    batch_size: int = 8192,
) -> HuTurn2Stage8MultiheadModel:
    return HuTurn2Stage8MultiheadModel.load(path, device=device, batch_size=batch_size)


def hu_turn2_policy_sample(
    board: Board,
    dealt_cards: Iterable[str],
    actions: list[Action],
    *,
    opponent_board: Board,
    dead_cards: Iterable[str],
    seat: str,
    to_act_order: str,
) -> dict[str, Any]:
    return {
        "rule_set": "regular",
        "schema": "hu_stage1",
        "phase": "hu_turn2_7card",
        "seat": seat,
        "to_act_order": to_act_order,
        "board": board_to_json(board),
        "opponent_board": board_to_json(opponent_board),
        "dead_cards": list(dead_cards),
        "dealt": list(dealt_cards),
        "best_action": 0,
        "score_gap": 0.0,
        "actions": [action_to_json(board, action) for action in actions],
    }


def _safe_predictions(model: object | None, sample: dict[str, Any], expected_len: int) -> tuple[np.ndarray | None, str]:
    if model is None:
        return None, "model_load_failed"
    try:
        predictions = np.asarray(model.predict_sample(sample), dtype=np.float64)
    except Exception:
        return None, "fallback_to_baseline"
    if predictions.ndim == 1:
        if predictions.shape[0] != expected_len:
            return None, "illegal_candidate"
    elif predictions.ndim == 2:
        if predictions.shape[0] != expected_len:
            return None, "illegal_candidate"
    else:
        return None, "illegal_candidate"
    if not np.isfinite(predictions).all():
        return None, "nan_prediction"
    return predictions, ""


def _top_margin(values: np.ndarray, best_index: int) -> float:
    if values.size <= 1:
        return 0.0
    best = float(values[best_index])
    second = max(float(value) for index, value in enumerate(values) if index != best_index)
    return best - second


class HuTurn2Stage8SelectiveOverridePolicy(RegularAiPolicy):
    """Regular AI with HU T2 Stage8 used only as a selective override."""

    hu_turn2_stage8_model: object | None
    hu_turn2_stage8_config: HuTurn2Stage8RuntimeConfig
    hu_turn2_decision_log: list[dict[str, Any]] | None
    hu_turn2_context: dict[str, Any]

    def __init__(
        self,
        *,
        hu_turn2_stage8_model: object | None = None,
        hu_turn2_stage8_config: HuTurn2Stage8RuntimeConfig | None = None,
        hu_turn2_decision_log: list[dict[str, Any]] | None = None,
        hu_turn2_context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.hu_turn2_stage8_model = hu_turn2_stage8_model
        self.hu_turn2_stage8_config = hu_turn2_stage8_config or HuTurn2Stage8RuntimeConfig(0.0, 0.0, 0.0, False)
        self.hu_turn2_decision_log = hu_turn2_decision_log
        self.hu_turn2_context = card_free_metadata(hu_turn2_context)

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
            action = self._choose_hu_turn2_stage8_action(
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

    def _choose_hu_turn2_stage8_action(
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
        actions = generate_turn_actions(board, dealt)
        if not actions:
            return None

        fallback_sample = policy_sample(board, dealt, actions)
        baseline_predictions, baseline_reason = _safe_predictions(self.turn2_model, fallback_sample, len(actions))
        if baseline_predictions is None:
            return None
        baseline_values = baseline_predictions.reshape(-1)
        baseline_index = canonical_argmax_index(baseline_values, actions)
        reference_margin = _top_margin(baseline_values, baseline_index)
        final_index = baseline_index
        candidate_index: int | None = None
        predicted_delta: float | None = None
        gate_probability: float | None = None
        predicted_ev: float | None = None
        candidate_ev_rank: int | None = None
        no_override_reason = ""

        if not self.hu_turn2_stage8_config.enabled:
            no_override_reason = "stage8_disabled"
        elif self.hu_turn2_stage8_config.allowed_seats and self.seat not in self.hu_turn2_stage8_config.allowed_seats:
            no_override_reason = "seat_not_allowed"
        elif self.hu_turn2_stage8_model is None:
            no_override_reason = "model_load_failed"
        elif reference_margin < self.hu_turn2_stage8_config.reference_min_margin:
            no_override_reason = "below_reference_margin"
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
            predictions, prediction_reason = _safe_predictions(self.hu_turn2_stage8_model, sample, len(actions))
            if predictions is None:
                no_override_reason = prediction_reason
            else:
                if predictions.ndim != 2 or predictions.shape[1] < 5:
                    no_override_reason = "illegal_candidate"
                else:
                    candidate_index = canonical_argmax_index(predictions[:, 1], actions)
                    predicted_delta = float(predictions[candidate_index, 1])
                    predicted_ev = float(predictions[candidate_index, 0])
                    ev_order = canonical_descending_indices(predictions[:, 0], actions)
                    candidate_ev_rank = ev_order.index(candidate_index) + 1
                    gate_probability = sigmoid(float(np.mean(predictions[:, 4])))
                    legality_check_result = self._candidate_legality_result(board, actions, candidate_index)
                    if legality_check_result != "legal":
                        no_override_reason = "illegal_candidate"
                    elif candidate_index == baseline_index:
                        no_override_reason = "same_as_baseline"
                    elif (
                        self.hu_turn2_stage8_config.min_model_score is not None
                        and predicted_ev < self.hu_turn2_stage8_config.min_model_score
                    ):
                        no_override_reason = "below_model_score"
                    elif (
                        self.hu_turn2_stage8_config.candidate_ev_rank_max is not None
                        and candidate_ev_rank > self.hu_turn2_stage8_config.candidate_ev_rank_max
                    ):
                        no_override_reason = "below_candidate_ev_rank"
                    elif predicted_delta < self.hu_turn2_stage8_config.min_margin:
                        no_override_reason = "below_stage8_margin"
                    elif gate_probability < self.hu_turn2_stage8_config.gate_threshold:
                        no_override_reason = "below_gate_threshold"
                    else:
                        final_index = candidate_index

        override_fired = final_index != baseline_index
        if not no_override_reason and not override_fired:
            no_override_reason = "fallback_to_baseline"
        visible_dead_cards = list(dead_cards)
        opponent_public = set(opponent_board.all_cards())
        hero_private_discards = [
            card for card in dead_cards if card not in opponent_public
        ]
        self._log_hu_turn2_decision(
            {
                **self.hu_turn2_context,
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
                "stage8_action": action_to_json(board, actions[candidate_index]) if candidate_index is not None else None,
                "fallback_action": action_to_json(board, actions[baseline_index]),
                "final_action": action_to_json(board, actions[final_index]),
                "baseline_action_index": baseline_index,
                "stage8_action_index": candidate_index,
                "final_action_index": final_index,
                "baseline_action_key": action_key(actions[baseline_index]).to_token(),
                "stage8_action_key": (
                    action_key(actions[candidate_index]).to_token()
                    if candidate_index is not None
                    else None
                ),
                "final_action_key": action_key(actions[final_index]).to_token(),
                "override_fired": override_fired,
                "no_override_reason": no_override_reason or "",
                "hu_turn2_min_margin": self.hu_turn2_stage8_config.min_margin,
                "hu_turn2_reference_min_margin": self.hu_turn2_stage8_config.reference_min_margin,
                "hu_turn2_gate_threshold": self.hu_turn2_stage8_config.gate_threshold,
                "hu_turn2_min_model_score": self.hu_turn2_stage8_config.min_model_score,
                "hu_turn2_allowed_seats": list(self.hu_turn2_stage8_config.allowed_seats),
                "hu_turn2_candidate_ev_rank_max": self.hu_turn2_stage8_config.candidate_ev_rank_max,
                "predicted_delta": predicted_delta,
                "gate_probability": gate_probability,
                "reference_margin_raw": reference_margin,
                "model_score": predicted_ev,
                "candidate_ev_rank": candidate_ev_rank,
                "legality_check_result": "legal" if candidate_index is not None else "not_evaluated",
                "runtime_latency_ms": (time.perf_counter() - started_at) * 1000.0,
            }
        )
        return actions[final_index]

    def _log_hu_turn2_decision(self, record: dict[str, Any]) -> None:
        if self.hu_turn2_decision_log is not None:
            self.hu_turn2_decision_log.append(record)
        log_path = self.hu_turn2_context.get("decision_log_path")
        if not log_path:
            return
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
