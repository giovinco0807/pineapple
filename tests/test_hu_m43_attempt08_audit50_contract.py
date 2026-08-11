from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from ofc_regular.hu_m43_attempt08_audit50_contract import (
    AUDIT50_PLAN_SHA256,
    PROFILES,
    ROOT_FIRST,
    ROOT_LAST,
    TOTAL_SHARDS,
    build_audit50_schedule,
    load_and_validate_audit50_plan,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/hu_joint_policy_m43_attempt08_audit50.json"


def test_audit50_plan_and_schedule_are_exactly_frozen() -> None:
    plan = load_and_validate_audit50_plan(PLAN)
    schedule = build_audit50_schedule()
    assert sha256_file(PLAN) == AUDIT50_PLAN_SHA256
    assert plan["population"]["roots"] == TOTAL_SHARDS == 50
    assert [row["root_index"] for row in schedule] == list(
        range(ROOT_FIRST, ROOT_LAST + 1)
    )
    assert [row["shard"] for row in schedule] == list(range(TOTAL_SHARDS))
    assert Counter(row["root_profile"] for row in schedule) == Counter(
        {profile: 10 for profile in PROFILES}
    )
    assert all(row["seed_material_opened"] is False for row in schedule)
    assert all(row["current_profile_resolved"] is False for row in schedule)


def test_audit50_plan_byte_tamper_fails_before_semantic_use(tmp_path: Path) -> None:
    tampered = tmp_path / PLAN.name
    tampered.write_bytes(PLAN.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256 changed"):
        load_and_validate_audit50_plan(tampered)


def test_audit50_has_no_development_or_preflight_root_overlap() -> None:
    roots = {row["root_index"] for row in build_audit50_schedule()}
    assert roots.isdisjoint(range(0, 200))

