import numpy as np

from ai.training.action_feature_encoding import (
    ACTION_FEATURE_DIM,
    action_feature_vector,
    adapt_np_state_with_action,
)


def test_action_feature_vector_encodes_regular_action_identity() -> None:
    decision = {
        "turn": 3,
        "dealt": ["Ah", "4s", "8s"],
        "board": {
            "top": ["Ad", "Ac"],
            "middle": ["6d", "7s", "7c", "4c"],
            "bottom": ["9h", "Kc", "9s"],
        },
    }
    candidate = {
        "placements": [["4s", "middle"], ["8s", "top"]],
        "discard": "Ah",
    }

    features = action_feature_vector(decision, candidate)

    assert features.shape == (ACTION_FEATURE_DIM,)
    assert features[:27].sum() == 1.0
    assert features[27:41].sum() == 1.0
    assert features[41:46].sum() == 1.0
    assert np.isclose(features[46:49].sum(), 1.0)
    assert np.isclose(features[49:91].sum(), 1.0)


def test_adapt_np_state_with_action_preserves_legacy_dim_and_appends_for_augmented_dim() -> None:
    state = np.ones(522, dtype=np.float32)
    decision = {"turn": 3, "dealt": ["Ah", "4s", "8s"], "board": {}}
    candidate = {"placements": [["4s", "middle"], ["8s", "top"]], "discard": "Ah"}

    legacy = adapt_np_state_with_action(state, 520, decision, candidate)
    augmented = adapt_np_state_with_action(state, 520 + ACTION_FEATURE_DIM, decision, candidate)

    assert legacy.shape == (520,)
    assert augmented.shape == (520 + ACTION_FEATURE_DIM,)
    assert np.allclose(augmented[:520], legacy)
    assert augmented[520:].sum() > 0.0
