from __future__ import annotations

import pytest

from ofc_regular.validate_hu_m31_t3_convergence import REFERENCE_BUDGET
from ofc_regular.profile_hu_m31_t3_parallelism import (
    ParallelProfileConfig,
    _validate_source_reports,
    project_local100_parallel_wall_time,
)


def _reports():
    local10 = {
        "schema": "hu_m31_t3_step3_local10_profile_v1",
        "all_gates_passed": True,
        "engine": {"library_sha256": "a" * 64},
        "local100_projection": {"projected_total_seconds": 1800.0},
    }
    convergence = {
        "schema": "hu_m31_t3_step3_budget_convergence_v1",
        "all_gates_passed": True,
        "engine": {"library_sha256": "a" * 64},
        "local100_authorization": {
            "authorized": True,
            "budget": REFERENCE_BUDGET.to_dict(),
        },
        "budget_runs": [
            {
                "budget": {"label": "baseline_1_1_1"},
                "batch_wall_seconds": 10.0,
            },
            {
                "budget": {"label": "teacher_default_4_8_2"},
                "batch_wall_seconds": 40.0,
            },
        ],
    }
    return local10, convergence


def test_projection_uses_measured_effective_parallelism():
    local10, convergence = _reports()
    rows = [{"wall_seconds": 100.0} for _ in range(8)]
    projection = project_local100_parallel_wall_time(
        local10_report=local10,
        convergence_report=convergence,
        worker_rows=rows,
        parallel_wall_seconds=120.0,
        gate_seconds=3600.0,
    )

    assert projection["sample_budget_cost_ratio"] == 4.0
    assert projection["measured_effective_parallelism"] == pytest.approx(800 / 120)
    assert projection["projected_parallel_seconds"] == pytest.approx(1080.0)


def test_source_reports_fail_closed_on_wrong_budget():
    local10, convergence = _reports()
    convergence["local100_authorization"]["budget"] = {
        **REFERENCE_BUDGET.to_dict(),
        "candidate_samples": 3,
    }
    with pytest.raises(ValueError, match="unexpected budget"):
        _validate_source_reports(local10, convergence)


def test_process_pool_contract_reuses_only_configured_worker_count():
    config = ParallelProfileConfig(workers=2, hand_count=4)
    process_ids = {101, 202, 101, 202}
    assert len(process_ids) == config.workers
    assert len(process_ids) != config.hand_count


@pytest.mark.parametrize(
    ("kwargs", "error"),
    (
        ({"workers": 0}, "positive"),
        ({"workers": 3, "hand_count": 2}, "must not exceed"),
        ({"max_projected_seconds": 0.0}, "positive"),
    ),
)
def test_parallel_profile_config_fails_closed(kwargs, error):
    with pytest.raises((TypeError, ValueError), match=error):
        ParallelProfileConfig(**kwargs)
