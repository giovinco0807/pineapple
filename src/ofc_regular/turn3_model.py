"""Turn3 action-value model for 9-card regular OFC states."""

from __future__ import annotations

import json
import pickle
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .cards import ALL_CARDS, RANKS, RANK_VALUE, card_rank, card_suit
from .evaluator import (
    HAND_FLUSH,
    HAND_QUADS,
    evaluate_3_card,
    evaluate_5_card,
    get_bottom_royalty,
    get_middle_royalty,
    get_top_royalty,
)
from .rules import check_fl_entry

ROWS = ("top", "middle", "bottom")
CARD_INDEX = {card: idx for idx, card in enumerate(ALL_CARDS)}
RANK_INDEX = {rank: idx for idx, rank in enumerate(RANKS)}

CURRENT_OFFSET = 0
DEALT_OFFSET = CURRENT_OFFSET + 3 * 52
PLACEMENT_OFFSET = DEALT_OFFSET + 52
DISCARD_OFFSET = PLACEMENT_OFFSET + 3 * 52
NEXT_OFFSET = DISCARD_OFFSET + 52
ROW_LEN_OFFSET = NEXT_OFFSET + 3 * 52
RANK_COUNT_OFFSET = ROW_LEN_OFFSET + 3
SUIT_COUNT_OFFSET = RANK_COUNT_OFFSET + 3 * 13
BASE_FEATURE_DIM = SUIT_COUNT_OFFSET + 3 * 4
ROW_EXTRA_OFFSET = BASE_FEATURE_DIM
ROW_EXTRA_DIM = 24
GLOBAL_EXTRA_OFFSET = ROW_EXTRA_OFFSET + 3 * ROW_EXTRA_DIM
GLOBAL_EXTRA_DIM = 4
FEATURE_DIM = GLOBAL_EXTRA_OFFSET + GLOBAL_EXTRA_DIM

ROW_CAPACITY = {"top": 3, "middle": 5, "bottom": 5}
SUMMARY_CACHE_SIZE = 1_000_000
ROYALTY_SCALE = {"top": 22.0, "middle": 50.0, "bottom": 25.0}


@dataclass(frozen=True)
class Turn3RidgeModel:
    weights: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: float

    def predict_matrix(self, features: np.ndarray) -> np.ndarray:
        features = _align_feature_matrix(features, self.feature_mean.shape[0])
        x = (features.astype(np.float64) - self.feature_mean) / self.feature_scale
        return x @ self.weights + self.target_mean

    def predict_sample(self, sample: dict[str, Any]) -> np.ndarray:
        features, _targets = sample_to_matrix(sample)
        return self.predict_matrix(features)

    def choose_action_index(self, sample: dict[str, Any]) -> int:
        return int(np.argmax(self.predict_sample(sample)))

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            weights=self.weights,
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            target_mean=np.array([self.target_mean], dtype=np.float64),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Turn3RidgeModel":
        data = np.load(path)
        return cls(
            weights=data["weights"],
            feature_mean=data["feature_mean"],
            feature_scale=data["feature_scale"],
            target_mean=float(data["target_mean"][0]),
        )


@dataclass(frozen=True)
class SklearnActionValueModel:
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
    def load(cls, path: str | Path) -> "SklearnActionValueModel":
        with Path(path).open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError(f"expected {cls.__name__}, got {type(model).__name__}")
        return model


@dataclass
class TorchActionValueModel:
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
                "model_kind": "torch_action_value_mlp",
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
    def load(cls, path: str | Path) -> "TorchActionValueModel":
        torch = _import_torch()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("model_kind") != "torch_action_value_mlp":
            raise TypeError(f"unsupported torch action-value model: {payload.get('model_kind')}")
        feature_dim = int(payload.get("feature_dim", -1))
        if feature_dim > FEATURE_DIM:
            raise ValueError(f"model feature dim {feature_dim} is newer than local dim {FEATURE_DIM}")
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


def load_action_value_model(path: str | Path) -> Turn3RidgeModel | SklearnActionValueModel | TorchActionValueModel:
    model_path = Path(path)
    if model_path.suffix == ".npz":
        return Turn3RidgeModel.load(model_path)
    if model_path.suffix in {".pkl", ".pickle"}:
        return SklearnActionValueModel.load(model_path)
    if model_path.suffix in {".pt", ".pth"}:
        return TorchActionValueModel.load(model_path)
    raise ValueError(f"unsupported model file extension: {model_path.suffix}")


def _align_feature_matrix(features: np.ndarray, expected_dim: int) -> np.ndarray:
    if features.shape[1] == expected_dim:
        return features
    if features.shape[1] > expected_dim:
        return features[:, :expected_dim]
    padding = np.zeros((features.shape[0], expected_dim - features.shape[1]), dtype=features.dtype)
    return np.hstack([features, padding])


def _estimator_feature_dim(estimator: Any) -> int | None:
    value = getattr(estimator, "n_features_in_", None)
    if value is not None:
        return int(value)
    steps = getattr(estimator, "steps", None)
    if steps:
        for _name, step in reversed(steps):
            value = getattr(step, "n_features_in_", None)
            if value is not None:
                return int(value)
    return None


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional local GPU stack
        raise RuntimeError("PyTorch is required to load or run .pt action-value models") from exc
    return torch


def _build_torch_mlp(
    torch: Any,
    input_dim: int,
    hidden_layer_sizes: Sequence[int],
    dropout: float,
    output_dim: int = 1,
) -> Any:
    layers: list[Any] = []
    last_dim = input_dim
    for hidden_dim in hidden_layer_sizes:
        layers.append(torch.nn.Linear(last_dim, int(hidden_dim)))
        layers.append(torch.nn.ReLU())
        if dropout > 0.0:
            layers.append(torch.nn.Dropout(dropout))
        last_dim = int(hidden_dim)
    layers.append(torch.nn.Linear(last_dim, int(output_dim)))
    return torch.nn.Sequential(*layers)


def read_teacher_samples(
    path: str | Path,
    max_samples: int | None = None,
    *,
    skip_samples: int = 0,
) -> list[dict[str, Any]]:
    if skip_samples < 0:
        raise ValueError("skip_samples must be non-negative")
    samples: list[dict[str, Any]] = []
    seen = 0
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            if seen < skip_samples:
                seen += 1
                continue
            samples.append(json.loads(line))
            seen += 1
            if max_samples is not None and len(samples) >= max_samples:
                break
    return samples


def split_samples(
    samples: Sequence[dict[str, Any]],
    *,
    holdout_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 <= holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in [0, 1)")
    indices = np.arange(len(samples))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    holdout_count = int(round(len(samples) * holdout_fraction))
    holdout_idx = set(indices[:holdout_count].tolist())
    train = [sample for idx, sample in enumerate(samples) if idx not in holdout_idx]
    holdout = [sample for idx, sample in enumerate(samples) if idx in holdout_idx]
    return train, holdout


def samples_to_matrix(samples: Iterable[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    materialized = list(samples)
    if not materialized:
        raise ValueError("no teacher samples")

    total_actions = 0
    for sample in materialized:
        actions = sample.get("actions", [])
        if not actions:
            raise ValueError("teacher sample has no actions")
        total_actions += len(actions)

    features = np.empty((total_actions, FEATURE_DIM), dtype=np.float32)
    targets = np.empty(total_actions, dtype=np.float64)
    cursor = 0
    for sample in materialized:
        for action in sample["actions"]:
            row = features[cursor]
            row.fill(0.0)
            _encode_action_into(row, sample, action)
            targets[cursor] = float(action["score"])
            cursor += 1
    return features, targets


def sample_to_matrix(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    actions = sample.get("actions", [])
    if not actions:
        raise ValueError("teacher sample has no actions")
    features = np.zeros((len(actions), FEATURE_DIM), dtype=np.float32)
    targets = np.zeros(len(actions), dtype=np.float64)
    for row, action in enumerate(actions):
        _encode_action_into(features[row], sample, action)
        targets[row] = float(action["score"])
    return features, targets


def predict_sample_maxes(
    model: Turn3RidgeModel | SklearnActionValueModel,
    samples: Iterable[dict[str, Any]],
) -> list[float]:
    """Predict max action value for each sample with one batched model call."""
    feature_blocks: list[np.ndarray] = []
    block_sizes: list[int] = []
    for sample in samples:
        features, _targets = sample_to_matrix(sample)
        feature_blocks.append(features)
        block_sizes.append(features.shape[0])
    if not feature_blocks:
        return []

    predictions = model.predict_matrix(np.vstack(feature_blocks))
    maxes: list[float] = []
    cursor = 0
    for block_size in block_sizes:
        block = predictions[cursor : cursor + block_size]
        maxes.append(float(block.max()))
        cursor += block_size
    return maxes


def encode_action(sample: dict[str, Any], action: dict[str, Any]) -> np.ndarray:
    vector = np.zeros(FEATURE_DIM, dtype=np.float32)
    _encode_action_into(vector, sample, action)
    return vector


def _encode_action_into(vector: np.ndarray, sample: dict[str, Any], action: dict[str, Any]) -> None:
    _encode_board(vector, sample["board"], CURRENT_OFFSET)
    _encode_cards(vector, sample.get("dealt", ()), DEALT_OFFSET)

    for card, row in action.get("placements", ()):
        _set_card_row(vector, card, row, PLACEMENT_OFFSET)
    _encode_cards(vector, action.get("discards", ()), DISCARD_OFFSET)

    next_board = action.get("next_board", sample["board"])
    _encode_board(vector, next_board, NEXT_OFFSET)
    _encode_row_stats(vector, next_board)
    _encode_extra_stats(vector, next_board)


def train_ridge_model(
    samples: Sequence[dict[str, Any]],
    *,
    l2: float = 1.0,
) -> Turn3RidgeModel:
    if l2 <= 0:
        raise ValueError("l2 must be positive")
    features, targets = samples_to_matrix(samples)
    feature_mean = features.mean(axis=0, dtype=np.float64)
    feature_scale = features.std(axis=0, dtype=np.float64)
    feature_scale[feature_scale < 1e-6] = 1.0
    target_mean = float(targets.mean())

    x = (features.astype(np.float64) - feature_mean) / feature_scale
    y = targets - target_mean
    xtx = x.T @ x
    xtx.flat[:: xtx.shape[0] + 1] += l2
    weights = np.linalg.solve(xtx, x.T @ y)
    return Turn3RidgeModel(
        weights=weights,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_mean=target_mean,
    )


def evaluate_model(
    model: Turn3RidgeModel,
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
    if batch_samples <= 0:
        raise ValueError("batch_samples must be positive")

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


def _encode_board(vector: np.ndarray, board: dict[str, Sequence[str]], offset: int) -> None:
    for row in ROWS:
        for card in board.get(row, ()):
            _set_card_row(vector, card, row, offset)


def _set_card_row(vector: np.ndarray, card: str, row: str, offset: int) -> None:
    row_index = ROWS.index(row)
    vector[offset + row_index * 52 + CARD_INDEX[card]] = 1.0


def _encode_cards(vector: np.ndarray, cards: Iterable[str], offset: int) -> None:
    for card in cards:
        vector[offset + CARD_INDEX[card]] = 1.0


def _encode_row_stats(vector: np.ndarray, board: dict[str, Sequence[str]]) -> None:
    for row_index, row in enumerate(ROWS):
        cards = tuple(board.get(row, ()))
        vector[ROW_LEN_OFFSET + row_index] = len(cards) / 5.0
        for card in cards:
            rank = card[0]
            suit = card[1]
            vector[RANK_COUNT_OFFSET + row_index * 13 + RANK_INDEX[rank]] += 1.0 / 4.0
            vector[SUIT_COUNT_OFFSET + row_index * 4 + "hdcs".index(suit)] += 1.0 / 5.0


def _encode_extra_stats(vector: np.ndarray, board: dict[str, Sequence[str]]) -> None:
    for row_index, row in enumerate(ROWS):
        _encode_row_extra(vector, row_index, row, tuple(board.get(row, ())))
    _encode_global_extra(vector, board)


def _encode_row_extra(vector: np.ndarray, row_index: int, row: str, cards: Sequence[str]) -> None:
    offset = ROW_EXTRA_OFFSET + row_index * ROW_EXTRA_DIM
    vector[offset : offset + ROW_EXTRA_DIM] = _row_extra_values(row, _cards_cache_key(cards))


@lru_cache(maxsize=SUMMARY_CACHE_SIZE)
def _row_extra_values(row: str, cards: tuple[str, ...]) -> tuple[float, ...]:
    values = np.zeros(ROW_EXTRA_DIM, dtype=np.float32)
    capacity = ROW_CAPACITY[row]
    slots = capacity - len(cards)
    ranks = [card_rank(card) for card in cards]
    rank_counts = Counter(ranks)
    suit_counts = Counter(card_suit(card) for card in cards)

    values[0] = len(cards) / capacity
    values[1] = max(slots, 0) / capacity
    if ranks:
        values[2] = sum(ranks) / (14.0 * capacity)
        values[3] = max(ranks) / 14.0
        values[4] = min(ranks) / 14.0
    values[5] = len(rank_counts) / capacity

    counts = list(rank_counts.values())
    max_multiplicity = max(counts, default=0)
    pairs = [rank for rank, count in rank_counts.items() if count >= 2]
    trips = [rank for rank, count in rank_counts.items() if count >= 3]
    quads = [rank for rank, count in rank_counts.items() if count >= 4]
    values[6] = max_multiplicity / capacity
    values[7] = min(len(pairs), 2) / 2.0
    values[8] = 1.0 if trips else 0.0
    values[9] = 1.0 if quads else 0.0
    values[10] = (max(pairs) / 14.0) if pairs else 0.0
    values[11] = (max(trips) / 14.0) if trips else 0.0

    suit_max = max(suit_counts.values(), default=0)
    values[12] = suit_max / capacity
    if row != "top" and suit_max + slots >= 5:
        values[13] = suit_max / 5.0

    straight_score, straight_high = _straight_potential(ranks, slots) if row != "top" else (0.0, 0.0)
    values[14] = straight_score
    values[15] = straight_high

    category = 0
    made_ranks: tuple[int, ...] = ()
    royalty = 0
    complete = len(cards) == capacity
    if complete:
        if row == "top":
            category, made_ranks = evaluate_3_card(cards)
            royalty = get_top_royalty(cards)
        elif row == "middle":
            category, made_ranks = evaluate_5_card(cards)
            royalty = get_middle_royalty(cards)
        else:
            category, made_ranks = evaluate_5_card(cards)
            royalty = get_bottom_royalty(cards)
    values[16] = category / 8.0
    values[17] = (made_ranks[0] / 14.0) if made_ranks else 0.0
    values[18] = (made_ranks[1] / 14.0) if len(made_ranks) > 1 else 0.0
    values[19] = royalty / ROYALTY_SCALE[row]

    if row == "top":
        values[20] = 1.0 if complete and check_fl_entry(cards).qualifies else 0.0
        values[21] = _top_fl_potential(cards, slots)
    values[22] = 1.0 if complete else 0.0
    if row != "top" and complete:
        values[23] = 1.0 if category >= HAND_FLUSH else 0.0
    return tuple(float(value) for value in values)


def _encode_global_extra(vector: np.ndarray, board: dict[str, Sequence[str]]) -> None:
    vector[GLOBAL_EXTRA_OFFSET : GLOBAL_EXTRA_OFFSET + 4] = _global_extra_values(
        _cards_cache_key(board.get("top", ())),
        _cards_cache_key(board.get("middle", ())),
        _cards_cache_key(board.get("bottom", ())),
    )


@lru_cache(maxsize=SUMMARY_CACHE_SIZE)
def _global_extra_values(
    top: tuple[str, ...],
    middle: tuple[str, ...],
    bottom: tuple[str, ...],
) -> tuple[float, float, float, float]:
    values = np.zeros(4, dtype=np.float32)
    values[0] = _row_order_feature("top", top, "middle", middle)
    values[1] = _row_order_feature("middle", middle, "bottom", bottom)
    completed_rows = int(len(top) == 3) + int(len(middle) == 5) + int(len(bottom) == 5)
    values[2] = completed_rows / 3.0
    royalty = 0
    if len(top) == 3:
        royalty += get_top_royalty(top)
    if len(middle) == 5:
        royalty += get_middle_royalty(middle)
    if len(bottom) == 5:
        royalty += get_bottom_royalty(bottom)
    values[3] = royalty / 100.0
    return tuple(float(value) for value in values)


def _cards_cache_key(cards: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(tuple(cards), key=lambda card: CARD_INDEX.get(card, -1)))


def _row_order_feature(left_row: str, left_cards: Sequence[str], right_row: str, right_cards: Sequence[str]) -> float:
    if len(left_cards) != ROW_CAPACITY[left_row] or len(right_cards) != ROW_CAPACITY[right_row]:
        return 0.0
    left_value = evaluate_3_card(left_cards) if left_row == "top" else evaluate_5_card(left_cards)
    right_value = evaluate_3_card(right_cards) if right_row == "top" else evaluate_5_card(right_cards)
    return 1.0 if left_value <= right_value else -1.0


def _top_fl_potential(cards: Sequence[str], slots: int) -> float:
    if slots < 0:
        return 0.0
    counts = Counter(card[0] for card in cards)
    if any(count + slots >= 3 for count in counts.values()):
        return 1.0
    if slots >= 2 and len(cards) <= 1:
        return 0.75
    for rank in ("A", "K", "Q"):
        if counts.get(rank, 0) + slots >= 2:
            return 0.75
    return 0.0


def _straight_potential(ranks: Sequence[int], slots: int) -> tuple[float, float]:
    if slots < 0:
        return 0.0, 0.0
    rank_set = set(ranks)
    sequences = [({14, 5, 4, 3, 2}, 5)]
    sequences.extend((set(range(high - 4, high + 1)), high) for high in range(6, 15))
    best_score = 0.0
    best_high = 0
    for sequence, high in sequences:
        missing = len(sequence - rank_set)
        if missing <= slots:
            score = (5 - missing) / 5.0
            if score > best_score or (score == best_score and high > best_high):
                best_score = score
                best_high = high
    return best_score, best_high / 14.0 if best_high else 0.0
