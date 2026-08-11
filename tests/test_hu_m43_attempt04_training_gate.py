from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ofc_regular.hu_m43_attempt04_training import (
    M43_ATTEMPT04_PROFILES,
    _metrics,
    _passes_scaled_dev900_gate,
)


ROOT = Path(__file__).resolve().parents[1]
FINAL_DEV_REPORT = (
    ROOT
    / "outputs/hu_joint_policy/m43_attempt04_dev900/"
    "meta_all_gain_conformal_final_pilot.json"
)


def _passing_gate_row() -> dict:
    return {
        "fires": 90,
        "false_positive_rate": 0.30,
        "mean_delta_per_state": 0.01,
        "cluster_lcb90_per_fire": 0.01,
        "actual_selected_tail_maxima": {"p95": 25.0, "p99": 40.0, "max": 50.0},
        "profile": {
            profile: {
                "states": 180,
                "fires": 18,
                "mean_delta_per_fire": 0.0,
                "mean_delta_per_state": 0.0,
            }
            for profile in M43_ATTEMPT04_PROFILES
        },
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fires", 8),
        ("mean_delta_per_fire", -0.001),
        ("mean_delta_per_state", -0.001),
    ],
)
def test_scaled_dev_gate_requires_nonnegative_metrics_in_every_profile(field, value):
    row = _passing_gate_row()
    assert _passes_scaled_dev900_gate(row) is True
    row["profile"][M43_ATTEMPT04_PROFILES[-1]][field] = value
    assert _passes_scaled_dev900_gate(row) is False


def test_metrics_reports_profile_delta_per_state_over_all_profile_states():
    samples = [
        SimpleNamespace(
            teacher_paired_delta_mean=np.asarray([float(index + 1)]),
            downside_loss_p95=np.asarray([1.0]),
            downside_loss_p99=np.asarray([2.0]),
            downside_loss_max=np.asarray([3.0]),
        )
        for index in range(5)
    ]
    result = _metrics(
        samples,
        M43_ATTEMPT04_PROFILES,
        [0, 0, 0, 0, 0],
        np.asarray([True, False, True, False, True]),
    )
    assert result["profile"][M43_ATTEMPT04_PROFILES[0]][
        "mean_delta_per_state"
    ] == 1.0
    assert result["profile"][M43_ATTEMPT04_PROFILES[1]][
        "mean_delta_per_state"
    ] == 0.0
    assert result["profile"][M43_ATTEMPT04_PROFILES[2]][
        "mean_delta_per_state"
    ] == 3.0


def test_attempt04_final_dev_report_remains_zero_eligible_under_corrected_gate():
    payload = json.loads(FINAL_DEV_REPORT.read_text(encoding="utf-8"))
    payload = payload["final_bounded_comparison"]
    assert payload["eligible_count"] == 0
    eligible = []
    for model in payload["models"].values():
        for quantile in model["quantiles"].values():
            for original in quantile["rows"]:
                row = deepcopy(original)
                for profile in row["profile"].values():
                    mean = profile.get("mean_delta_per_fire")
                    fires = int(profile["fires"])
                    states = int(profile["states"])
                    profile["mean_delta_per_state"] = (
                        float(mean) * fires / states if mean is not None else 0.0
                    )
                if _passes_scaled_dev900_gate(row):
                    eligible.append(row)
    assert eligible == []
