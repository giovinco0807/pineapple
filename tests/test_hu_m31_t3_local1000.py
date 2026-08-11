from __future__ import annotations

import json

import pytest

from ofc_regular.action_space import generate_turn_actions
from ofc_regular.validate_hu_m31_t3_profile import _generate_hand_t3_roots
from ofc_regular.validate_hu_m31_t3_local1000 import (
    LOCAL1000_SUMMARY_SCHEMA,
    Local1000Config,
    _percentile,
    _write_json_atomic,
    build_task_specs,
    projected_local1000_seconds,
    schedule_task_specs,
)


def test_frozen_step4_task_matrix_is_complete_and_unique():
    config = Local1000Config()
    specs = build_task_specs(config)
    scheduled = schedule_task_specs(config)

    assert len(specs) == 510
    assert len({spec.task_id for spec in specs}) == 510
    assert {spec.task_id for spec in scheduled} == {spec.task_id for spec in specs}
    assert sum(spec.kind == "primary" for spec in specs) == 500
    assert sum(spec.kind == "determinism" for spec in specs) == 10


def test_scheduler_places_expensive_first_seat_geometries_first():
    config = Local1000Config()
    scheduled = [spec for spec in schedule_task_specs(config) if spec.kind == "primary"]
    first_counts = []
    for spec in scheduled:
        observation = _generate_hand_t3_roots(config.hand_seed(spec.hand_index))[0]
        first_counts.append(
            len(
                generate_turn_actions(
                    observation.hero_board,
                    observation.dealt_cards,
                )
            )
        )
    assert first_counts == sorted(first_counts, reverse=True)


def test_projection_is_frozen_below_three_point_six_hours():
    summary = {
        "performance": {
            "primary_scalar_seconds_sum": 2265.614876499967,
            "determinism_seconds_sum": 545.8078172999958,
        }
    }
    projection = projected_local1000_seconds(summary)
    assert projection["projected_seconds"] == pytest.approx(12761.076120264817)
    assert projection["gate_seconds"] == 12960.0
    assert projection["passed"] is True


def test_step4_percentile_uses_nearest_rank():
    values = list(range(1, 1001))
    assert _percentile(values, 0.50) == 500.0
    assert _percentile(values, 0.95) == 950.0
    assert _percentile(values, 0.99) == 990.0


def test_step4_write_once_artifact(tmp_path):
    path = tmp_path / "summary.json"
    _write_json_atomic(path, {"schema": LOCAL1000_SUMMARY_SCHEMA})
    assert json.loads(path.read_text(encoding="utf-8"))["schema"] == (
        LOCAL1000_SUMMARY_SCHEMA
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _write_json_atomic(path, {"schema": "changed"})


@pytest.mark.parametrize(
    ("kwargs", "error"),
    (
        ({"root_count": 998}, "exactly 1,000"),
        ({"seed_start": 1}, "seed_start is frozen"),
        ({"worker_count": 4}, "frozen at two"),
        ({"max_wall_seconds": 13000.0}, "frozen at 12,960"),
    ),
)
def test_step4_config_fails_closed(kwargs, error):
    with pytest.raises((TypeError, ValueError), match=error):
        Local1000Config(**kwargs)
