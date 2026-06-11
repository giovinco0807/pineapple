"""HU-aware Turn3 action-value model for regular OFC."""

from __future__ import annotations

import pickle
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .cards import ALL_CARDS, RANKS, RANK_VALUE, card_rank, card_suit
from .evaluator import (
    evaluate_3_card,
    evaluate_5_card,
    get_bottom_royalty,
    get_middle_royalty,
    get_top_royalty,
    score_board,
)
from .rules import check_fl_entry
from .state import Board, ROWS
from .turn3_model import (
    CARD_INDEX,
    CURRENT_OFFSET,
    DEALT_OFFSET,
    DISCARD_OFFSET,
    FEATURE_DIM as SELF_FEATURE_DIM,
    NEXT_OFFSET,
    PLACEMENT_OFFSET,
    RANK_INDEX,
    ROYALTY_SCALE,
    ROW_CAPACITY,
    encode_action,
    read_teacher_samples,
    split_samples,
    _align_feature_matrix,
    _encode_board as _encode_self_board,
    _encode_cards as _encode_self_cards,
    _encode_extra_stats as _encode_self_extra_stats,
    _encode_row_stats as _encode_self_row_stats,
    _set_card_row as _set_self_card_row,
    _build_torch_mlp,
    _estimator_feature_dim,
    _import_torch,
    _straight_potential,
)

OPPONENT_OFFSET = SELF_FEATURE_DIM
DEAD_OFFSET = OPPONENT_OFFSET + 3 * 52
SEAT_OFFSET = DEAD_OFFSET + 52
ORDER_OFFSET = SEAT_OFFSET + 2
OPP_ROW_LEN_OFFSET = ORDER_OFFSET + 2
OPP_RANK_COUNT_OFFSET = OPP_ROW_LEN_OFFSET + 3
OPP_SUIT_COUNT_OFFSET = OPP_RANK_COUNT_OFFSET + 3 * 13
GLOBAL_OFFSET = OPP_SUIT_COUNT_OFFSET + 3 * 4
GLOBAL_DIM = 12
HU_DERIVED_OFFSET = GLOBAL_OFFSET + GLOBAL_DIM
HU_DERIVED_DIM = 48
HU_MATCHUP_OFFSET = HU_DERIVED_OFFSET + HU_DERIVED_DIM
HU_MATCHUP_DIM = 48
HU_FEATURE_DIM = HU_MATCHUP_OFFSET + HU_MATCHUP_DIM

SEAT_INDEX = {"first": 0, "second": 1}
SUMMARY_CACHE_SIZE = 1_000_000


@dataclass(frozen=True)
class HuSklearnActionValueModel:
    estimator: Any

    def predict_matrix(self, features: np.ndarray) -> np.ndarray:
        expected_dim = _estimator_feature_dim(self.estimator)
        if expected_dim is not None:
            features = _align_feature_matrix(features, expected_dim)
        return np.asarray(self.estimator.predict(features.astype(np.float32)), dtype=np.float64)

    def predict_sample(self, sample: dict[str, Any]) -> np.ndarray:
        features, _targets = sample_to_matrix(sample)
        return self.predict_matrix(features)

    def choose_action_index(self, sample: dict[str, Any]) -> int:
        return int(np.argmax(self.predict_sample(sample)))

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "HuSklearnActionValueModel":
        with Path(path).open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError(f"expected {cls.__name__}, got {type(model).__name__}")
        return model


@dataclass
class HuTorchActionValueModel:
    state_dict: dict[str, Any]
    hidden_layer_sizes: tuple[int, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: float
    target_scale: float
    dropout: float = 0.0
    batch_size: int = 8192
    _net: Any = field(default=None, init=False, repr=False)
    _net_device: str | None = field(default=None, init=False, repr=False)

    def predict_matrix(self, features: np.ndarray) -> np.ndarray:
        torch = _import_torch()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = self._get_net(torch, device)

        feature_array = _align_feature_matrix(features, self.feature_mean.shape[0]).astype(
            np.float32,
            copy=False,
        )
        predictions: list[np.ndarray] = []
        inference_guard = getattr(torch, "inference_mode", torch.no_grad)
        with inference_guard():
            for start in range(0, feature_array.shape[0], self.batch_size):
                end = min(start + self.batch_size, feature_array.shape[0])
                batch = (feature_array[start:end] - self.feature_mean) / self.feature_scale
                tensor = torch.from_numpy(batch.astype(np.float32, copy=False)).to(device)
                output = net(tensor).squeeze(-1).detach().cpu().numpy().astype(np.float64)
                predictions.append(output * self.target_scale + self.target_mean)
        if not predictions:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate(predictions)

    def predict_sample(self, sample: dict[str, Any]) -> np.ndarray:
        features, _targets = sample_to_matrix(sample)
        return self.predict_matrix(features)

    def choose_action_index(self, sample: dict[str, Any]) -> int:
        return int(np.argmax(self.predict_sample(sample)))

    def save(self, path: str | Path) -> None:
        torch = _import_torch()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_kind": "hu_torch_action_value_mlp",
                "feature_dim": int(self.feature_mean.shape[0]),
                "hidden_layer_sizes": list(self.hidden_layer_sizes),
                "dropout": self.dropout,
                "batch_size": self.batch_size,
                "state_dict": self.state_dict,
                "feature_mean": self.feature_mean.astype(np.float32),
                "feature_scale": self.feature_scale.astype(np.float32),
                "target_mean": float(self.target_mean),
                "target_scale": float(self.target_scale),
            },
            output,
        )

    @classmethod
    def load(cls, path: str | Path) -> "HuTorchActionValueModel":
        torch = _import_torch()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("model_kind") != "hu_torch_action_value_mlp":
            raise TypeError(f"unsupported HU torch action-value model: {payload.get('model_kind')}")
        feature_dim = int(payload.get("feature_dim", -1))
        if feature_dim > HU_FEATURE_DIM:
            raise ValueError(f"model feature dim {feature_dim} is newer than local dim {HU_FEATURE_DIM}")
        return cls(
            state_dict=payload["state_dict"],
            hidden_layer_sizes=tuple(int(size) for size in payload["hidden_layer_sizes"]),
            dropout=float(payload.get("dropout", 0.0)),
            batch_size=int(payload.get("batch_size", 8192)),
            feature_mean=np.asarray(payload["feature_mean"], dtype=np.float32),
            feature_scale=np.asarray(payload["feature_scale"], dtype=np.float32),
            target_mean=float(payload["target_mean"]),
            target_scale=float(payload["target_scale"]),
        )

    def _get_net(self, torch: Any, device: str) -> Any:
        if self._net is None or self._net_device != device:
            net = _build_torch_mlp(
                torch,
                int(self.feature_mean.shape[0]),
                self.hidden_layer_sizes,
                self.dropout,
            )
            net.load_state_dict(self.state_dict)
            net.to(device)
            net.eval()
            self._net = net
            self._net_device = device
        return self._net


def load_hu_action_value_model(path: str | Path) -> HuSklearnActionValueModel | HuTorchActionValueModel:
    model_path = Path(path)
    if model_path.suffix in {".pkl", ".pickle"}:
        return HuSklearnActionValueModel.load(model_path)
    if model_path.suffix in {".pt", ".pth"}:
        return HuTorchActionValueModel.load(model_path)
    raise ValueError(f"unsupported HU model file extension: {model_path.suffix}")


def hu_policy_sample(
    board: Board,
    dealt_cards: Iterable[str],
    actions: Sequence[Any],
    *,
    opponent_board: Board,
    dead_cards: Iterable[str],
    seat: str,
    to_act_order: str,
) -> dict[str, Any]:
    from .policy import action_to_json, board_to_json

    return {
        "rule_set": "regular",
        "schema": "hu_stage1",
        "phase": "hu_turn3_9card",
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


def samples_to_matrix(samples: Iterable[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    materialized = list(samples)
    if not materialized:
        raise ValueError("no HU teacher samples")
    total_actions = sum(len(sample.get("actions", ())) for sample in materialized)
    if total_actions <= 0:
        raise ValueError("HU teacher samples have no actions")

    features = np.empty((total_actions, HU_FEATURE_DIM), dtype=np.float32)
    targets = np.empty(total_actions, dtype=np.float64)
    cursor = 0
    for sample in materialized:
        sample_features, sample_targets = sample_to_matrix(sample)
        end = cursor + sample_features.shape[0]
        features[cursor:end] = sample_features
        targets[cursor:end] = sample_targets
        cursor = end
    return features, targets


def sample_to_matrix(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    actions = sample.get("actions", ())
    if not actions:
        raise ValueError("HU teacher sample has no actions")
    features = np.zeros((len(actions), HU_FEATURE_DIM), dtype=np.float32)
    targets = np.zeros(len(actions), dtype=np.float64)
    context = _build_sample_encode_context(sample, actions)
    for row_index, action in enumerate(actions):
        _encode_action_into_with_context(features[row_index], sample, action, context)
        targets[row_index] = float(action.get("score", 0.0))
    return features, targets


def evaluate_model(
    model: HuSklearnActionValueModel,
    samples: Sequence[dict[str, Any]],
    *,
    tie_tolerance: float = 1e-9,
    batch_samples: int = 512,
) -> dict[str, float]:
    if not samples:
        return {
            "samples": 0,
            "actions": 0,
            "mse": 0.0,
            "top1_accuracy": 0.0,
            "top3_accuracy": 0.0,
            "avg_regret": 0.0,
        }
    squared_error_sum = 0.0
    regret_sum = 0.0
    correct = 0
    top3_correct = 0
    action_count = 0
    for start in range(0, len(samples), batch_samples):
        batch = samples[start : start + batch_samples]
        feature_blocks: list[np.ndarray] = []
        target_blocks: list[np.ndarray] = []
        block_sizes: list[int] = []
        for sample in batch:
            features, targets = sample_to_matrix(sample)
            feature_blocks.append(features)
            target_blocks.append(targets)
            block_sizes.append(len(targets))

        features = np.vstack(feature_blocks)
        targets = np.concatenate(target_blocks)
        predictions = model.predict_matrix(features)
        errors = predictions - targets
        squared_error_sum += float(np.dot(errors, errors))
        action_count += int(targets.shape[0])

        cursor = 0
        for block_size in block_sizes:
            block_predictions = predictions[cursor : cursor + block_size]
            block_targets = targets[cursor : cursor + block_size]
            predicted_idx = int(np.argmax(block_predictions))
            true_best_score = float(np.max(block_targets))
            correct += int(block_targets[predicted_idx] >= true_best_score - tie_tolerance)
            top_k = min(3, block_size)
            top_indices = np.argpartition(block_predictions, -top_k)[-top_k:]
            top3_correct += int(
                np.any(block_targets[top_indices] >= true_best_score - tie_tolerance)
            )
            regret_sum += float(true_best_score - block_targets[predicted_idx])
            cursor += block_size

    return {
        "samples": float(len(samples)),
        "actions": float(action_count),
        "mse": float(squared_error_sum / action_count),
        "top1_accuracy": float(correct / len(samples)),
        "top3_accuracy": float(top3_correct / len(samples)),
        "avg_regret": float(regret_sum / len(samples)),
    }


def _encode_action_into(vector: np.ndarray, sample: dict[str, Any], action: dict[str, Any]) -> None:
    base = encode_action(sample, action)
    vector[:SELF_FEATURE_DIM] = base

    opponent_board = sample.get("opponent_board", {})
    _encode_board(vector, opponent_board, OPPONENT_OFFSET)
    _encode_cards(vector, sample.get("dead_cards", ()), DEAD_OFFSET)
    _encode_binary(vector, sample.get("seat", "first"), SEAT_OFFSET)
    _encode_binary(vector, sample.get("to_act_order", "first"), ORDER_OFFSET)
    _encode_row_stats(vector, opponent_board)
    _encode_global_stats(vector, sample, action)
    _encode_hu_derived_stats(vector, sample, action)
    _encode_hu_matchup_stats(vector, sample, action)


def _build_sample_encode_context(
    sample: dict[str, Any],
    actions: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    common = np.zeros(HU_FEATURE_DIM, dtype=np.float32)
    self_common = np.zeros(SELF_FEATURE_DIM, dtype=np.float32)
    _encode_self_board(self_common, sample["board"], CURRENT_OFFSET)
    _encode_self_cards(self_common, sample.get("dealt", ()), DEALT_OFFSET)

    opponent_board = sample.get("opponent_board", {})
    _encode_board(common, opponent_board, OPPONENT_OFFSET)
    _encode_cards(common, sample.get("dead_cards", ()), DEAD_OFFSET)
    _encode_binary(common, sample.get("seat", "first"), SEAT_OFFSET)
    _encode_binary(common, sample.get("to_act_order", "first"), ORDER_OFFSET)
    _encode_row_stats(common, opponent_board)

    available_by_rank = _available_by_rank(_visible_cards(sample, actions[0]))
    opponent_top = _top_summary(opponent_board.get("top", ()), available_by_rank)
    opponent_rows = _complete_row_summary(opponent_board)
    opponent_matchups = {
        row: _row_matchup_summary(row, opponent_board.get(row, ()), available_by_rank)
        for row in ROWS
    }
    opponent_needed_top = _opponent_needed_top_ranks(opponent_board, available_by_rank)
    opponent_count = _card_count(opponent_board)
    opponent_terminal: dict[str, float] | None = None
    if opponent_count == 13:
        try:
            score = score_board(
                opponent_board.get("top", ()),
                opponent_board.get("middle", ()),
                opponent_board.get("bottom", ()),
            )
            opponent_terminal = {
                "busted": 1.0 if score.busted else 0.0,
                "royalty": score.total_royalty / 100.0,
                "fl_entry": 1.0 if score.fl_entry.qualifies else 0.0,
            }
        except ValueError:
            opponent_terminal = None

    return {
        "common": common,
        "self_common": self_common,
        "opponent_board": opponent_board,
        "available_by_rank": available_by_rank,
        "opponent_top": opponent_top,
        "opponent_rows": opponent_rows,
        "opponent_matchups": opponent_matchups,
        "opponent_needed_top": opponent_needed_top,
        "opponent_count": opponent_count,
        "opponent_terminal": opponent_terminal,
    }


def _encode_action_into_with_context(
    vector: np.ndarray,
    sample: dict[str, Any],
    action: dict[str, Any],
    context: dict[str, Any],
) -> None:
    vector[:] = context["common"]
    _encode_self_action_into_with_context(vector[:SELF_FEATURE_DIM], sample, action, context)
    _encode_global_stats_with_context(vector, sample, action, context)
    _encode_hu_derived_stats_with_context(vector, sample, action, context)
    _encode_hu_matchup_stats_with_context(vector, sample, action, context)


def _encode_self_action_into_with_context(
    vector: np.ndarray,
    sample: dict[str, Any],
    action: dict[str, Any],
    context: dict[str, Any],
) -> None:
    vector[:] = context["self_common"]
    for card, row in action.get("placements", ()):
        _set_self_card_row(vector, card, row, PLACEMENT_OFFSET)
    _encode_self_cards(vector, action.get("discards", ()), DISCARD_OFFSET)

    next_board = action.get("next_board", sample["board"])
    _encode_self_board(vector, next_board, NEXT_OFFSET)
    _encode_self_row_stats(vector, next_board)
    _encode_self_extra_stats(vector, next_board)


def _encode_global_stats_with_context(
    vector: np.ndarray,
    sample: dict[str, Any],
    action: dict[str, Any],
    context: dict[str, Any],
) -> None:
    next_board = action.get("next_board", sample.get("board", {}))
    opponent_board = context["opponent_board"]
    hero_count = _card_count(next_board)
    opp_count = int(context["opponent_count"])
    vector[GLOBAL_OFFSET + 0] = hero_count / 13.0
    vector[GLOBAL_OFFSET + 1] = opp_count / 13.0
    for row_index, row in enumerate(ROWS):
        hero_open = ROW_CAPACITY[row] - len(next_board.get(row, ()))
        opp_open = ROW_CAPACITY[row] - len(opponent_board.get(row, ()))
        vector[GLOBAL_OFFSET + 2 + row_index] = hero_open / ROW_CAPACITY[row]
        vector[GLOBAL_OFFSET + 5 + row_index] = opp_open / ROW_CAPACITY[row]

    terminal = context["opponent_terminal"]
    if terminal is not None:
        vector[GLOBAL_OFFSET + 8] = 1.0
        vector[GLOBAL_OFFSET + 9] = terminal["busted"]
        vector[GLOBAL_OFFSET + 10] = terminal["royalty"]
        vector[GLOBAL_OFFSET + 11] = terminal["fl_entry"]


def _encode_hu_derived_stats_with_context(
    vector: np.ndarray,
    sample: dict[str, Any],
    action: dict[str, Any],
    context: dict[str, Any],
) -> None:
    next_board = action.get("next_board", sample.get("board", {}))
    available_by_rank = context["available_by_rank"]

    hero_top = _top_summary(next_board.get("top", ()), available_by_rank)
    opponent_top = context["opponent_top"]
    hero_rows = _complete_row_summary(next_board)
    opponent_rows = context["opponent_rows"]

    offset = HU_DERIVED_OFFSET
    _write_top_summary(vector, offset, hero_top)
    _write_top_summary(vector, offset + 12, opponent_top)

    vector[offset + 24] = hero_rows["top_royalty"] / 22.0
    vector[offset + 25] = hero_rows["middle_royalty"] / 50.0
    vector[offset + 26] = hero_rows["bottom_royalty"] / 25.0
    vector[offset + 27] = hero_rows["top_middle_order"]
    vector[offset + 28] = hero_rows["middle_bottom_order"]
    vector[offset + 29] = opponent_rows["top_royalty"] / 22.0
    vector[offset + 30] = opponent_rows["middle_royalty"] / 50.0
    vector[offset + 31] = opponent_rows["bottom_royalty"] / 25.0
    vector[offset + 32] = opponent_rows["top_middle_order"]
    vector[offset + 33] = opponent_rows["middle_bottom_order"]

    vector[offset + 34] = hero_top["fl_potential"] - opponent_top["fl_potential"]
    vector[offset + 35] = hero_top["high_pair_rank"] - opponent_top["high_pair_rank"]
    vector[offset + 36] = hero_rows["total_royalty"] / 100.0 - opponent_rows["total_royalty"] / 100.0

    discards = tuple(action.get("discards", ()))
    vector[offset + 37] = sum(1 for card in discards if card[0] in "QKA") / max(len(discards), 1)
    vector[offset + 38] = sum(
        1 for card in discards if card[0] in context["opponent_needed_top"]
    ) / max(len(discards), 1)

    for index, rank in enumerate(("Q", "K", "A")):
        vector[offset + 39 + index] = available_by_rank[rank] / 4.0
        vector[offset + 42 + index] = opponent_top[f"needs_{rank.lower()}"]
        vector[offset + 45 + index] = hero_top[f"needs_{rank.lower()}"]


def _encode_hu_matchup_stats_with_context(
    vector: np.ndarray,
    sample: dict[str, Any],
    action: dict[str, Any],
    context: dict[str, Any],
) -> None:
    next_board = action.get("next_board", sample.get("board", {}))
    opponent_board = context["opponent_board"]
    available_by_rank = context["available_by_rank"]

    hero_summaries = {
        row: _row_matchup_summary(row, next_board.get(row, ()), available_by_rank)
        for row in ROWS
    }
    opponent_summaries = context["opponent_matchups"]

    offset = HU_MATCHUP_OFFSET
    for row_index, row in enumerate(ROWS):
        hero = hero_summaries[row]
        opponent = opponent_summaries[row]
        row_offset = offset + row_index * 12
        vector[row_offset + 0] = hero["count"]
        vector[row_offset + 1] = opponent["count"]
        vector[row_offset + 2] = hero["count"] - opponent["count"]
        vector[row_offset + 3] = hero["slots"]
        vector[row_offset + 4] = opponent["slots"]
        vector[row_offset + 5] = hero["category"]
        vector[row_offset + 6] = opponent["category"]
        vector[row_offset + 7] = hero["category"] - opponent["category"]
        vector[row_offset + 8] = hero["royalty"]
        vector[row_offset + 9] = opponent["royalty"]
        vector[row_offset + 10] = hero["total_royalty"] / 100.0 - opponent["total_royalty"] / 100.0
        vector[row_offset + 11] = hero["premium_potential"] - opponent["premium_potential"]

    hero_rows = _complete_row_summary(next_board)
    opponent_rows = context["opponent_rows"]
    completed_hero = sum(
        int(len(next_board.get(row, ())) == ROW_CAPACITY[row])
        for row in ROWS
    )
    completed_opponent = sum(
        int(len(opponent_board.get(row, ())) == ROW_CAPACITY[row])
        for row in ROWS
    )
    global_offset = offset + 36
    vector[global_offset + 0] = hero_rows["total_royalty"] / 100.0 - opponent_rows["total_royalty"] / 100.0
    vector[global_offset + 1] = (completed_hero - completed_opponent) / 3.0
    vector[global_offset + 2] = hero_rows["top_middle_order"]
    vector[global_offset + 3] = hero_rows["middle_bottom_order"]
    vector[global_offset + 4] = opponent_rows["top_middle_order"]
    vector[global_offset + 5] = opponent_rows["middle_bottom_order"]
    vector[global_offset + 6] = hero_rows["top_middle_order"] - opponent_rows["top_middle_order"]
    vector[global_offset + 7] = hero_rows["middle_bottom_order"] - opponent_rows["middle_bottom_order"]
    vector[global_offset + 8] = hero_summaries["top"]["premium_potential"] - opponent_summaries["top"]["premium_potential"]
    vector[global_offset + 9] = hero_summaries["middle"]["premium_potential"] - opponent_summaries["middle"]["premium_potential"]
    vector[global_offset + 10] = hero_summaries["bottom"]["premium_potential"] - opponent_summaries["bottom"]["premium_potential"]
    vector[global_offset + 11] = sum(
        hero_summaries[row]["premium_potential"] - opponent_summaries[row]["premium_potential"]
        for row in ROWS
    ) / 3.0


def _encode_board(vector: np.ndarray, board: dict[str, Sequence[str]], offset: int) -> None:
    for row in ROWS:
        row_index = ROWS.index(row)
        for card in board.get(row, ()):
            vector[offset + row_index * 52 + CARD_INDEX[card]] = 1.0


def _encode_cards(vector: np.ndarray, cards: Iterable[str], offset: int) -> None:
    for card in cards:
        if card in CARD_INDEX:
            vector[offset + CARD_INDEX[card]] = 1.0


def _encode_binary(vector: np.ndarray, value: str, offset: int) -> None:
    vector[offset + SEAT_INDEX.get(value, 0)] = 1.0


def _encode_row_stats(vector: np.ndarray, board: dict[str, Sequence[str]]) -> None:
    for row_index, row in enumerate(ROWS):
        cards = tuple(board.get(row, ()))
        capacity = ROW_CAPACITY[row]
        vector[OPP_ROW_LEN_OFFSET + row_index] = len(cards) / capacity
        for card in cards:
            vector[OPP_RANK_COUNT_OFFSET + row_index * 13 + RANK_INDEX[card[0]]] += 1.0 / 4.0
            vector[OPP_SUIT_COUNT_OFFSET + row_index * 4 + "hdcs".index(card[1])] += 1.0 / 5.0


def _encode_global_stats(vector: np.ndarray, sample: dict[str, Any], action: dict[str, Any]) -> None:
    next_board = action.get("next_board", sample.get("board", {}))
    opponent_board = sample.get("opponent_board", {})
    hero_count = _card_count(next_board)
    opp_count = _card_count(opponent_board)
    vector[GLOBAL_OFFSET + 0] = hero_count / 13.0
    vector[GLOBAL_OFFSET + 1] = opp_count / 13.0
    for row_index, row in enumerate(ROWS):
        hero_open = ROW_CAPACITY[row] - len(next_board.get(row, ()))
        opp_open = ROW_CAPACITY[row] - len(opponent_board.get(row, ()))
        vector[GLOBAL_OFFSET + 2 + row_index] = hero_open / ROW_CAPACITY[row]
        vector[GLOBAL_OFFSET + 5 + row_index] = opp_open / ROW_CAPACITY[row]

    if opp_count == 13:
        try:
            score = score_board(
                opponent_board.get("top", ()),
                opponent_board.get("middle", ()),
                opponent_board.get("bottom", ()),
            )
        except ValueError:
            return
        vector[GLOBAL_OFFSET + 8] = 1.0
        vector[GLOBAL_OFFSET + 9] = 1.0 if score.busted else 0.0
        vector[GLOBAL_OFFSET + 10] = score.total_royalty / 100.0
        vector[GLOBAL_OFFSET + 11] = 1.0 if score.fl_entry.qualifies else 0.0


def _encode_hu_derived_stats(vector: np.ndarray, sample: dict[str, Any], action: dict[str, Any]) -> None:
    next_board = action.get("next_board", sample.get("board", {}))
    opponent_board = sample.get("opponent_board", {})
    visible_cards = _visible_cards(sample, action)
    available_by_rank = _available_by_rank(visible_cards)

    hero_top = _top_summary(next_board.get("top", ()), available_by_rank)
    opponent_top = _top_summary(opponent_board.get("top", ()), available_by_rank)
    hero_rows = _complete_row_summary(next_board)
    opponent_rows = _complete_row_summary(opponent_board)

    offset = HU_DERIVED_OFFSET
    _write_top_summary(vector, offset, hero_top)
    _write_top_summary(vector, offset + 12, opponent_top)

    vector[offset + 24] = hero_rows["top_royalty"] / 22.0
    vector[offset + 25] = hero_rows["middle_royalty"] / 50.0
    vector[offset + 26] = hero_rows["bottom_royalty"] / 25.0
    vector[offset + 27] = hero_rows["top_middle_order"]
    vector[offset + 28] = hero_rows["middle_bottom_order"]
    vector[offset + 29] = opponent_rows["top_royalty"] / 22.0
    vector[offset + 30] = opponent_rows["middle_royalty"] / 50.0
    vector[offset + 31] = opponent_rows["bottom_royalty"] / 25.0
    vector[offset + 32] = opponent_rows["top_middle_order"]
    vector[offset + 33] = opponent_rows["middle_bottom_order"]

    vector[offset + 34] = hero_top["fl_potential"] - opponent_top["fl_potential"]
    vector[offset + 35] = hero_top["high_pair_rank"] - opponent_top["high_pair_rank"]
    vector[offset + 36] = hero_rows["total_royalty"] / 100.0 - opponent_rows["total_royalty"] / 100.0

    discards = tuple(action.get("discards", ()))
    vector[offset + 37] = sum(1 for card in discards if card[0] in "QKA") / max(len(discards), 1)
    vector[offset + 38] = sum(1 for card in discards if card[0] in _opponent_needed_top_ranks(opponent_board, available_by_rank)) / max(len(discards), 1)

    for index, rank in enumerate(("Q", "K", "A")):
        vector[offset + 39 + index] = available_by_rank[rank] / 4.0
        vector[offset + 42 + index] = opponent_top[f"needs_{rank.lower()}"]
        vector[offset + 45 + index] = hero_top[f"needs_{rank.lower()}"]


def _encode_hu_matchup_stats(vector: np.ndarray, sample: dict[str, Any], action: dict[str, Any]) -> None:
    next_board = action.get("next_board", sample.get("board", {}))
    opponent_board = sample.get("opponent_board", {})
    visible_cards = _visible_cards(sample, action)
    available_by_rank = _available_by_rank(visible_cards)

    hero_summaries = {
        row: _row_matchup_summary(row, next_board.get(row, ()), available_by_rank)
        for row in ROWS
    }
    opponent_summaries = {
        row: _row_matchup_summary(row, opponent_board.get(row, ()), available_by_rank)
        for row in ROWS
    }

    offset = HU_MATCHUP_OFFSET
    for row_index, row in enumerate(ROWS):
        hero = hero_summaries[row]
        opponent = opponent_summaries[row]
        row_offset = offset + row_index * 12
        vector[row_offset + 0] = hero["count"]
        vector[row_offset + 1] = opponent["count"]
        vector[row_offset + 2] = hero["count"] - opponent["count"]
        vector[row_offset + 3] = hero["slots"]
        vector[row_offset + 4] = opponent["slots"]
        vector[row_offset + 5] = hero["category"]
        vector[row_offset + 6] = opponent["category"]
        vector[row_offset + 7] = hero["category"] - opponent["category"]
        vector[row_offset + 8] = hero["royalty"]
        vector[row_offset + 9] = opponent["royalty"]
        vector[row_offset + 10] = hero["total_royalty"] / 100.0 - opponent["total_royalty"] / 100.0
        vector[row_offset + 11] = hero["premium_potential"] - opponent["premium_potential"]

    hero_rows = _complete_row_summary(next_board)
    opponent_rows = _complete_row_summary(opponent_board)
    completed_hero = sum(
        int(len(next_board.get(row, ())) == ROW_CAPACITY[row])
        for row in ROWS
    )
    completed_opponent = sum(
        int(len(opponent_board.get(row, ())) == ROW_CAPACITY[row])
        for row in ROWS
    )
    global_offset = offset + 36
    vector[global_offset + 0] = hero_rows["total_royalty"] / 100.0 - opponent_rows["total_royalty"] / 100.0
    vector[global_offset + 1] = (completed_hero - completed_opponent) / 3.0
    vector[global_offset + 2] = hero_rows["top_middle_order"]
    vector[global_offset + 3] = hero_rows["middle_bottom_order"]
    vector[global_offset + 4] = opponent_rows["top_middle_order"]
    vector[global_offset + 5] = opponent_rows["middle_bottom_order"]
    vector[global_offset + 6] = hero_rows["top_middle_order"] - opponent_rows["top_middle_order"]
    vector[global_offset + 7] = hero_rows["middle_bottom_order"] - opponent_rows["middle_bottom_order"]
    vector[global_offset + 8] = hero_summaries["top"]["premium_potential"] - opponent_summaries["top"]["premium_potential"]
    vector[global_offset + 9] = hero_summaries["middle"]["premium_potential"] - opponent_summaries["middle"]["premium_potential"]
    vector[global_offset + 10] = hero_summaries["bottom"]["premium_potential"] - opponent_summaries["bottom"]["premium_potential"]
    vector[global_offset + 11] = sum(
        hero_summaries[row]["premium_potential"] - opponent_summaries[row]["premium_potential"]
        for row in ROWS
    ) / 3.0


def _row_matchup_summary(
    row: str,
    cards: Sequence[str],
    available_by_rank: dict[str, int],
) -> dict[str, float]:
    return dict(
        _row_matchup_summary_cached(
            row,
            _cards_cache_key(cards),
            _available_cache_key(available_by_rank),
        )
    )


@lru_cache(maxsize=SUMMARY_CACHE_SIZE)
def _row_matchup_summary_cached(
    row: str,
    cards: tuple[str, ...],
    available_key: tuple[int, ...],
) -> tuple[tuple[str, float], ...]:
    available_by_rank = _available_from_cache_key(available_key)
    capacity = ROW_CAPACITY[row]
    slots = max(0, capacity - len(cards))
    complete = len(cards) == capacity
    category = 0
    royalty = 0
    if complete:
        if row == "top":
            category, _made_ranks = evaluate_3_card(cards)
            royalty = get_top_royalty(cards)
        elif row == "middle":
            category, _made_ranks = evaluate_5_card(cards)
            royalty = get_middle_royalty(cards)
        else:
            category, _made_ranks = evaluate_5_card(cards)
            royalty = get_bottom_royalty(cards)

    return tuple(
        {
        "count": len(cards) / capacity,
        "slots": slots / capacity,
        "category": category / 8.0,
        "royalty": royalty / ROYALTY_SCALE[row],
        "total_royalty": float(royalty),
        "premium_potential": _row_premium_potential(
            row,
            cards,
            slots,
            category=category,
            available_by_rank=available_by_rank,
        ),
        }.items()
    )


def _row_premium_potential(
    row: str,
    cards: Sequence[str],
    slots: int,
    *,
    category: int,
    available_by_rank: dict[str, int],
) -> float:
    if row == "top":
        return _top_summary(cards, available_by_rank)["fl_potential"]

    ranks = [card_rank(card) for card in cards]
    rank_counts = Counter(ranks)
    suit_counts = Counter(card_suit(card) for card in cards)
    made_score = category / 8.0 if len(cards) == ROW_CAPACITY[row] else 0.0
    max_multiplicity = max(rank_counts.values(), default=0)
    pair_count = sum(1 for count in rank_counts.values() if count >= 2)
    if max_multiplicity + slots >= 4:
        multiplicity_score = 1.0
    elif max_multiplicity >= 3 and pair_count >= 1:
        multiplicity_score = 0.85
    elif max_multiplicity + slots >= 3:
        multiplicity_score = 0.65
    elif pair_count >= 2:
        multiplicity_score = 0.5
    elif pair_count >= 1:
        multiplicity_score = 0.35
    else:
        multiplicity_score = 0.0

    suit_max = max(suit_counts.values(), default=0)
    flush_score = suit_max / 5.0 if suit_max + slots >= 5 else 0.0
    straight_score, _straight_high = _straight_potential(ranks, slots)
    return max(made_score, multiplicity_score, flush_score, straight_score)


def _write_top_summary(vector: np.ndarray, offset: int, summary: dict[str, float]) -> None:
    vector[offset + 0] = summary["count"]
    vector[offset + 1] = summary["slots"]
    vector[offset + 2] = summary["complete_fl"]
    vector[offset + 3] = summary["fl_potential"]
    vector[offset + 4] = summary["high_pair_made"]
    vector[offset + 5] = summary["high_pair_rank"]
    vector[offset + 6] = summary["any_pair"]
    vector[offset + 7] = summary["any_trips"]
    vector[offset + 8] = summary["trip_potential"]
    vector[offset + 9] = summary["pair_or_better_potential"]
    vector[offset + 10] = summary["max_multiplicity"]
    vector[offset + 11] = summary["top_royalty"]


def _visible_cards(sample: dict[str, Any], action: dict[str, Any]) -> tuple[str, ...]:
    cards: list[str] = []
    for board_key in ("board", "opponent_board"):
        board = sample.get(board_key, {})
        for row in ROWS:
            cards.extend(board.get(row, ()))
    for row in ROWS:
        cards.extend(action.get("next_board", {}).get(row, ()))
    cards.extend(sample.get("dealt", ()))
    cards.extend(sample.get("dead_cards", ()))
    cards.extend(action.get("discards", ()))
    return tuple(dict.fromkeys(card for card in cards if card in CARD_INDEX))


def _available_by_rank(visible_cards: Iterable[str]) -> dict[str, int]:
    used = Counter(card[0] for card in visible_cards)
    return {rank: max(0, 4 - used.get(rank, 0)) for rank in RANKS}


def _top_summary(cards: Sequence[str], available_by_rank: dict[str, int]) -> dict[str, float]:
    return dict(_top_summary_cached(_cards_cache_key(cards), _available_cache_key(available_by_rank)))


@lru_cache(maxsize=SUMMARY_CACHE_SIZE)
def _top_summary_cached(
    cards: tuple[str, ...],
    available_key: tuple[int, ...],
) -> tuple[tuple[str, float], ...]:
    available_by_rank = _available_from_cache_key(available_key)
    slots = max(0, 3 - len(cards))
    counts = Counter(card[0] for card in cards)
    pair_ranks = [rank for rank, count in counts.items() if count >= 2]
    high_pair_ranks = [rank for rank in ("Q", "K", "A") if counts.get(rank, 0) >= 2]
    high_pair_rank_value = max((RANK_VALUE[rank] for rank in high_pair_ranks), default=0)
    max_multiplicity = max(counts.values(), default=0)

    complete_fl = 1.0 if len(cards) == 3 and check_fl_entry(cards).qualifies else 0.0
    high_pair_potential = any(
        counts.get(rank, 0) + min(slots, available_by_rank[rank]) >= 2
        for rank in ("Q", "K", "A")
    )
    trip_potential = any(
        counts.get(rank, 0) + min(slots, available_by_rank[rank]) >= 3
        for rank in RANKS
    )
    fl_potential = 1.0 if complete_fl or trip_potential else (0.75 if high_pair_potential else 0.0)

    royalty = 0
    if len(cards) == 3:
        from .evaluator import get_top_royalty

        royalty = get_top_royalty(cards)

    return tuple(
        {
        "count": len(cards) / 3.0,
        "slots": slots / 3.0,
        "complete_fl": complete_fl,
        "fl_potential": fl_potential,
        "high_pair_made": 1.0 if high_pair_ranks else 0.0,
        "high_pair_rank": high_pair_rank_value / 14.0,
        "any_pair": 1.0 if pair_ranks else 0.0,
        "any_trips": 1.0 if max_multiplicity >= 3 else 0.0,
        "trip_potential": 1.0 if trip_potential else 0.0,
        "pair_or_better_potential": 1.0 if high_pair_potential or trip_potential else 0.0,
        "max_multiplicity": max_multiplicity / 3.0,
        "top_royalty": royalty / 22.0,
        "needs_q": _needs_rank_for_top_pair("Q", counts, slots, available_by_rank),
        "needs_k": _needs_rank_for_top_pair("K", counts, slots, available_by_rank),
        "needs_a": _needs_rank_for_top_pair("A", counts, slots, available_by_rank),
        }.items()
    )


def _needs_rank_for_top_pair(
    rank: str,
    counts: Counter[str],
    slots: int,
    available_by_rank: dict[str, int],
) -> float:
    if slots <= 0 or counts.get(rank, 0) >= 2:
        return 0.0
    missing = 2 - counts.get(rank, 0)
    if missing <= min(slots, available_by_rank[rank]):
        return 1.0
    return 0.0


def _opponent_needed_top_ranks(
    opponent_board: dict[str, Sequence[str]],
    available_by_rank: dict[str, int],
) -> set[str]:
    top = tuple(opponent_board.get("top", ()))
    slots = max(0, 3 - len(top))
    counts = Counter(card[0] for card in top)
    return {
        rank
        for rank in ("Q", "K", "A")
        if _needs_rank_for_top_pair(rank, counts, slots, available_by_rank) > 0.0
    }


def _complete_row_summary(board: dict[str, Sequence[str]]) -> dict[str, float]:
    return dict(
        _complete_row_summary_cached(
            _cards_cache_key(board.get("top", ())),
            _cards_cache_key(board.get("middle", ())),
            _cards_cache_key(board.get("bottom", ())),
        )
    )


@lru_cache(maxsize=SUMMARY_CACHE_SIZE)
def _complete_row_summary_cached(
    top: tuple[str, ...],
    middle: tuple[str, ...],
    bottom: tuple[str, ...],
) -> tuple[tuple[str, float], ...]:
    top_royalty = middle_royalty = bottom_royalty = 0
    top_value = middle_value = bottom_value = None

    if len(top) == 3:
        from .evaluator import evaluate_3_card, get_top_royalty

        top_value = evaluate_3_card(top)
        top_royalty = get_top_royalty(top)
    if len(middle) == 5:
        from .evaluator import evaluate_5_card, get_middle_royalty

        middle_value = evaluate_5_card(middle)
        middle_royalty = get_middle_royalty(middle)
    if len(bottom) == 5:
        from .evaluator import evaluate_5_card, get_bottom_royalty

        bottom_value = evaluate_5_card(bottom)
        bottom_royalty = get_bottom_royalty(bottom)

    return tuple(
        {
        "top_royalty": float(top_royalty),
        "middle_royalty": float(middle_royalty),
        "bottom_royalty": float(bottom_royalty),
        "total_royalty": float(top_royalty + middle_royalty + bottom_royalty),
        "top_middle_order": _order_value(top_value, middle_value),
        "middle_bottom_order": _order_value(middle_value, bottom_value),
        }.items()
    )


def _cards_cache_key(cards: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(tuple(cards), key=lambda card: CARD_INDEX.get(card, -1)))


def _available_cache_key(available_by_rank: dict[str, int]) -> tuple[int, ...]:
    return tuple(int(available_by_rank.get(rank, 0)) for rank in RANKS)


def _available_from_cache_key(key: tuple[int, ...]) -> dict[str, int]:
    return {rank: int(key[index]) for index, rank in enumerate(RANKS)}


def _order_value(left: tuple[int, tuple[int, ...]] | None, right: tuple[int, tuple[int, ...]] | None) -> float:
    if left is None or right is None:
        return 0.0
    return 1.0 if left <= right else -1.0


def _card_count(board: dict[str, Sequence[str]]) -> int:
    return sum(len(board.get(row, ())) for row in ROWS)
