"""Candidate-model loading helpers for HU T0 search.

T0 candidate unions may mix the legacy self-board opening model with newer
HU-aware action-value models. Both expose ``predict_sample`` but use different
feature encoders and serialization formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .hu_turn3_model import load_hu_action_value_model
from .turn3_model import load_action_value_model


def load_turn0_candidate_model(path: str | Path) -> Any:
    model_path = Path(path)
    hu_error: Exception | None = None
    try:
        return load_hu_action_value_model(model_path)
    except Exception as exc:  # The legacy self-board format is intentionally different.
        hu_error = exc
    try:
        return load_action_value_model(model_path)
    except Exception as self_error:
        raise RuntimeError(
            f"failed to load T0 candidate model {model_path} as HU-aware or self-board model; "
            f"HU error={hu_error!r}; self-board error={self_error!r}"
        ) from self_error
