from __future__ import annotations

import json

import pytest

from ofc_regular.validate_hu_m31_t3_convergence import REFERENCE_BUDGET
from ofc_regular.validate_hu_m31_t3_local100 import (
    LOCAL100_SUMMARY_SCHEMA,
    Local100Config,
    _percentile,
    _validate_authorization,
    _write_json_atomic,
    build_task_specs,
    schedule_task_specs,
)


def test_frozen_local100_task_matrix_is_complete_and_unique():
    config = Local100Config()
    specs = build_task_specs(config)
    scheduled = schedule_task_specs(config)

    assert len(specs) == 70
    assert len({spec.task_id for spec in specs}) == 70
    assert {spec.task_id for spec in scheduled} == {spec.task_id for spec in specs}
    assert sum(spec.kind == "primary" for spec in specs) == 50
    assert sum(spec.kind == "determinism" for spec in specs) == 10
    assert sum(spec.kind == "permutation" for spec in specs) == 10


def test_authorization_pins_budget_workers_library_and_projection():
    config = Local100Config()
    report = {
        "schema": "hu_m31_t3_step3_parallel_profile_v1",
        "all_gates_passed": True,
        "local100_authorized": True,
        "engine": {"library_sha256": "a" * 64},
        "config": {"workers": 2, "budget": REFERENCE_BUDGET.to_dict()},
        "projection": {"projected_parallel_seconds": 3599.0},
    }
    _validate_authorization(report, expected_library_sha256="a" * 64, config=config)

    report["projection"]["projected_parallel_seconds"] = 3600.1
    with pytest.raises(ValueError, match="exceeds"):
        _validate_authorization(
            report,
            expected_library_sha256="a" * 64,
            config=config,
        )


def test_percentile_uses_frozen_nearest_rank_rule():
    values = list(range(1, 101))
    assert _percentile(values, 0.50) == 50.0
    assert _percentile(values, 0.95) == 95.0
    assert _percentile(values, 0.99) == 99.0


def test_local100_write_once_output(tmp_path):
    path = tmp_path / "summary.json"
    _write_json_atomic(path, {"schema": LOCAL100_SUMMARY_SCHEMA})
    assert json.loads(path.read_text(encoding="utf-8"))["schema"] == (
        LOCAL100_SUMMARY_SCHEMA
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _write_json_atomic(path, {"schema": "changed"})


@pytest.mark.parametrize(
    ("kwargs", "error"),
    (
        ({"root_count": 98}, "exactly 100"),
        ({"worker_count": 3}, "frozen at two"),
        ({"rayon_threads_per_worker": 7}, "frozen at eight"),
    ),
)
def test_local100_config_fails_closed(kwargs, error):
    with pytest.raises((TypeError, ValueError), match=error):
        Local100Config(**kwargs)
