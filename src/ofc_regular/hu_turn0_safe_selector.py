"""Runtime loading and scoring for HU T0 safe-override selectors."""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .train_hu_turn0_safe_override_selector import row_to_feature_vector


MODEL_KIND = "hu_turn0_safe_override_selector_sklearn"


def load_hu_turn0_safe_selector_model(path: str | Path) -> dict[str, Any]:
    model_path = Path(path)
    with model_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or payload.get("model_kind") != MODEL_KIND:
        raise ValueError(f"unsupported HU T0 safe selector: {model_path}")
    if payload.get("estimator") is None or payload.get("feature_mode") is None:
        raise ValueError(f"HU T0 safe selector is incomplete: {model_path}")
    return payload


def score_hu_turn0_safe_selector(model_payload: Any, row: dict[str, Any]) -> float:
    if not isinstance(model_payload, dict) or model_payload.get("model_kind") != MODEL_KIND:
        raise ValueError("invalid HU T0 safe selector payload")
    estimator = model_payload.get("estimator")
    feature_mode = str(model_payload.get("feature_mode"))
    if estimator is None:
        raise ValueError("HU T0 safe selector has no estimator")
    vector = row_to_feature_vector(row, feature_mode=feature_mode).reshape(1, -1)
    if hasattr(estimator, "predict_proba"):
        probabilities = np.asarray(estimator.predict_proba(vector), dtype=np.float64)
        score = float(probabilities[0, 1])
    elif hasattr(estimator, "decision_function"):
        decision = float(np.asarray(estimator.decision_function(vector)).reshape(-1)[0])
        score = 1.0 / (1.0 + math.exp(-decision))
    else:
        score = float(np.asarray(estimator.predict(vector)).reshape(-1)[0])
    if not math.isfinite(score):
        raise ValueError("HU T0 safe selector produced non-finite score")
    return score
