import argparse

import numpy as np
import pytest

from ai.tutor.audit_hybrid_shortlist import (
    effective_sync_weights,
    load_jsonl_index,
    sync_adjusted_model_score,
)


def _args(**overrides):
    values = {
        "bust_weight": 0.0,
        "fl_any_weight": 0.0,
        "fl_qq_weight": 0.0,
        "fl_kk_weight": 0.0,
        "fl_aa_weight": 0.0,
        "fl_trips_weight": 0.0,
        "sync_bust_weight": None,
        "sync_fl_any_weight": None,
        "sync_fl_qq_weight": None,
        "sync_fl_kk_weight": None,
        "sync_fl_aa_weight": None,
        "sync_fl_trips_weight": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_load_jsonl_index_ignores_directory_source(tmp_path):
    assert load_jsonl_index(tmp_path) == {}


def test_sync_adjusted_score_can_differ_from_pool_score():
    preds = {
        "score": np.array([10.0, 9.0], dtype=np.float32),
        "bust": np.array([0.0, 0.5], dtype=np.float32),
        "fl": np.array([0.0, 0.8], dtype=np.float32),
        "fl_types": np.zeros((2, 4), dtype=np.float32),
    }
    args = _args(sync_bust_weight=2.0, sync_fl_any_weight=2.0)

    adjusted = sync_adjusted_model_score(preds, args)

    assert adjusted.tolist() == pytest.approx([10.0, 9.6])
    assert effective_sync_weights(args)["bust"] == 2.0
    assert effective_sync_weights(args)["fl_any"] == 2.0


def test_sync_adjusted_score_defaults_to_pool_weights():
    preds = {
        "score": np.array([10.0, 9.0], dtype=np.float32),
        "bust": np.array([0.0, 0.5], dtype=np.float32),
        "fl": np.array([0.0, 0.8], dtype=np.float32),
        "fl_types": np.zeros((2, 4), dtype=np.float32),
    }
    args = _args(bust_weight=1.0, fl_any_weight=0.5)

    adjusted = sync_adjusted_model_score(preds, args)

    assert adjusted.tolist() == pytest.approx([10.0, 8.9])
    assert effective_sync_weights(args)["bust"] == 1.0
    assert effective_sync_weights(args)["fl_any"] == 0.5
