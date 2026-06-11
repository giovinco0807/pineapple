"""Gate model for HU Turn3 overrides."""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .evaluator import get_bottom_royalty, get_middle_royalty, get_top_royalty, score_board
from .rules import check_fl_entry
from .state import ROWS
from .turn3_model import ROW_CAPACITY


GATE_FEATURE_NAMES = (
    "predicted_margin",
    "self_regret",
    "hu_chosen_score",
    "hu_baseline_score",
    "self_chosen_score",
    "self_baseline_score",
    "actions_count",
    "dead_count",
    "seat_second",
    "to_act_second",
    "hero_top_count",
    "opponent_top_count",
    "chosen_top_count",
    "baseline_top_count",
    "chosen_top_fl",
    "baseline_top_fl",
    "chosen_top_royalty",
    "baseline_top_royalty",
    "chosen_total_royalty",
    "baseline_total_royalty",
    "chosen_busted_if_complete",
    "baseline_busted_if_complete",
    "chosen_high_discards",
    "baseline_high_discards",
    "top_len_delta",
    "middle_len_delta",
    "bottom_len_delta",
    "chosen_top_middle_order",
    "baseline_top_middle_order",
    "chosen_middle_bottom_order",
    "baseline_middle_bottom_order",
    "opponent_complete",
)


@dataclass(frozen=True)
class HuTurn3GateModel:
    estimator: Any
    feature_names: tuple[str, ...] = GATE_FEATURE_NAMES

    def predict_accept_probability(
        self,
        sample: dict[str, Any],
        *,
        chosen_index: int,
        baseline_index: int,
        hu_predictions: Sequence[float],
        self_predictions: Sequence[float],
    ) -> float:
        features = decision_gate_features(
            sample,
            chosen_index=chosen_index,
            baseline_index=baseline_index,
            hu_predictions=hu_predictions,
            self_predictions=self_predictions,
        )
        matrix = features.reshape(1, -1)
        if hasattr(self.estimator, "predict_proba"):
            probabilities = self.estimator.predict_proba(matrix)
            classes = list(getattr(self.estimator, "classes_", [0, 1]))
            positive_index = classes.index(1) if 1 in classes else len(classes) - 1
            return float(probabilities[0][positive_index])
        if hasattr(self.estimator, "decision_function"):
            score = float(np.asarray(self.estimator.decision_function(matrix)).reshape(-1)[0])
            return float(1.0 / (1.0 + math.exp(-score)))
        prediction = float(np.asarray(self.estimator.predict(matrix)).reshape(-1)[0])
        return float(min(1.0, max(0.0, prediction)))

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "HuTurn3GateModel":
        with Path(path).open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError(f"expected {cls.__name__}, got {type(model).__name__}")
        return model


def load_hu_turn3_gate_model(path: str | Path) -> HuTurn3GateModel:
    return HuTurn3GateModel.load(path)


def decision_gate_features(
    sample: dict[str, Any],
    *,
    chosen_index: int,
    baseline_index: int,
    hu_predictions: Sequence[float],
    self_predictions: Sequence[float],
) -> np.ndarray:
    actions = sample.get("actions", ())
    if not actions:
        raise ValueError("HU gate sample has no actions")
    if chosen_index >= len(actions) or baseline_index >= len(actions):
        raise IndexError("HU gate action index out of range")

    chosen = actions[chosen_index]
    baseline = actions[baseline_index]
    chosen_board = chosen.get("next_board", sample.get("board", {}))
    baseline_board = baseline.get("next_board", sample.get("board", {}))
    opponent_board = sample.get("opponent_board", {})
    chosen_summary = _board_summary(chosen_board)
    baseline_summary = _board_summary(baseline_board)

    values = [
        float(hu_predictions[chosen_index]) - float(hu_predictions[baseline_index]),
        float(self_predictions[baseline_index]) - float(self_predictions[chosen_index]),
        float(hu_predictions[chosen_index]),
        float(hu_predictions[baseline_index]),
        float(self_predictions[chosen_index]),
        float(self_predictions[baseline_index]),
        len(actions) / 24.0,
        len(sample.get("dead_cards", ())) / 52.0,
        1.0 if sample.get("seat") == "second" else 0.0,
        1.0 if sample.get("to_act_order") == "second" else 0.0,
        len(sample.get("board", {}).get("top", ())) / 3.0,
        len(opponent_board.get("top", ())) / 3.0,
        len(chosen_board.get("top", ())) / 3.0,
        len(baseline_board.get("top", ())) / 3.0,
        1.0 if check_fl_entry(tuple(chosen_board.get("top", ()))).qualifies else 0.0,
        1.0 if check_fl_entry(tuple(baseline_board.get("top", ()))).qualifies else 0.0,
        _top_royalty(chosen_board) / 22.0,
        _top_royalty(baseline_board) / 22.0,
        chosen_summary["total_royalty"] / 100.0,
        baseline_summary["total_royalty"] / 100.0,
        chosen_summary["busted"],
        baseline_summary["busted"],
        _high_discard_fraction(chosen),
        _high_discard_fraction(baseline),
    ]
    for row in ROWS:
        capacity = ROW_CAPACITY[row]
        values.append(
            (
                len(chosen_board.get(row, ()))
                - len(baseline_board.get(row, ()))
            )
            / capacity
        )
    values.extend(
        [
            chosen_summary["top_middle_order"],
            baseline_summary["top_middle_order"],
            chosen_summary["middle_bottom_order"],
            baseline_summary["middle_bottom_order"],
            1.0 if all(len(opponent_board.get(row, ())) == ROW_CAPACITY[row] for row in ROWS) else 0.0,
        ]
    )
    return np.asarray(values, dtype=np.float32)


def _top_royalty(board: dict[str, Sequence[str]]) -> float:
    top = tuple(board.get("top", ()))
    return float(get_top_royalty(top)) if len(top) == 3 else 0.0


def _high_discard_fraction(action: dict[str, Any]) -> float:
    discards = tuple(action.get("discards", ()))
    if not discards:
        return 0.0
    return sum(1 for card in discards if str(card)[0] in "TJQKA") / len(discards)


def _board_summary(board: dict[str, Sequence[str]]) -> dict[str, float]:
    top = tuple(board.get("top", ()))
    middle = tuple(board.get("middle", ()))
    bottom = tuple(board.get("bottom", ()))
    top_royalty = get_top_royalty(top) if len(top) == 3 else 0
    middle_royalty = get_middle_royalty(middle) if len(middle) == 5 else 0
    bottom_royalty = get_bottom_royalty(bottom) if len(bottom) == 5 else 0
    busted = 0.0
    top_middle_order = 0.0
    middle_bottom_order = 0.0
    if len(top) == 3 and len(middle) == 5 and len(bottom) == 5:
        board_score = score_board(top, middle, bottom)
        busted = 1.0 if board_score.busted else 0.0
        top_middle_order = 1.0
        middle_bottom_order = 1.0
    return {
        "total_royalty": float(top_royalty + middle_royalty + bottom_royalty),
        "busted": busted,
        "top_middle_order": top_middle_order,
        "middle_bottom_order": middle_bottom_order,
    }
