"""Runtime helpers for HU Turn1 safe-override selector models."""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .train_hu_turn1_safe_override_selector import row_to_feature_vector


def load_hu_turn1_safe_selector_model(path: Path) -> Any:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"HU Turn1 safe selector must be a dict payload: {path}")
    if payload.get("model_kind") != "hu_turn1_safe_override_selector_sklearn":
        raise ValueError(f"unsupported HU Turn1 safe selector kind: {payload.get('model_kind')!r}")
    if "estimator" not in payload or "feature_mode" not in payload:
        raise ValueError(f"HU Turn1 safe selector is missing estimator/feature_mode: {path}")
    return payload


def score_hu_turn1_safe_selector(model_payload: Any, row: dict[str, Any]) -> float:
    if not isinstance(model_payload, dict):
        raise ValueError("HU Turn1 safe selector payload must be a dict")
    feature_mode = str(model_payload.get("feature_mode", "delta_plus_meta"))
    estimator = model_payload.get("estimator")
    if estimator is None:
        raise ValueError("HU Turn1 safe selector payload is missing estimator")
    x = row_to_feature_vector(row, feature_mode=feature_mode).reshape(1, -1)
    if hasattr(estimator, "predict_proba"):
        score = float(estimator.predict_proba(x)[0, 1])
    elif hasattr(estimator, "decision_function"):
        decision = float(estimator.decision_function(x)[0])
        score = 1.0 / (1.0 + math.exp(-decision))
    else:
        prediction = np.asarray(estimator.predict(x), dtype=np.float64).reshape(-1)
        if prediction.size == 0:
            raise ValueError("HU Turn1 safe selector produced empty prediction")
        score = float(prediction[0])
    if not math.isfinite(score):
        raise ValueError("HU Turn1 safe selector produced non-finite score")
    return score
