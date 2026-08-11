from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import run_hu_m31_t3_step6d_performance_v2 as runner


def _all_seed_values(contract: dict) -> set[int]:
    values: set[int] = set()
    for index in runner.CONTRACT_HAND_INDICES:
        values.update(
            runner.candidate02_performance_lock_recovery_v3_seed_values(
                index
            ).values()
        )
    return values


def test_recovery_v3_seed_schedule_is_fresh_and_disjoint() -> None:
    contract = runner.candidate02_performance_lock_recovery_v3_seed_contract()
    values = _all_seed_values(contract)
    v1 = {
        value
        for index in runner.CONTRACT_HAND_INDICES
        for value in runner.candidate02_performance_lock_seed_values(index).values()
    }
    rearm1 = {
        value
        for index in runner.CONTRACT_HAND_INDICES
        for value in (
            runner.candidate02_performance_lock_recovery_seed_values(index).values()
        )
    }

    assert len(values) == 600
    assert not values & v1
    assert not values & rearm1
    assert contract["seed_min"] == 710_108_071_901
    assert contract["seed_max"] == 715_207_072_198
    assert contract["performance_lock_v1_overlap_count"] == 0
    assert contract["performance_lock_rearm1_overlap_count"] == 0
    assert contract["candidate02_development_overlap_count"] == 0
    assert contract["existing_step6d_union_overlap_count"] == 0
    assert contract["all_values_unique"] is True
    assert contract["locked_before_content_read"] is True
    assert contract["seed_set_sha256"] == runner.contract_canonical_sha256(
        sorted(values)
    )


def test_recovery_v3_contract_and_manifest_are_exactly_typed() -> None:
    contract = runner.build_run_contract(
        candidate_library_sha256=(
            runner.CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_CANDIDATE_LIBRARY_SHA256
        ),
        reference_library_sha256=(
            runner.CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_REFERENCE_LIBRARY_SHA256
        ),
        variant=runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_VARIANT,
    )
    validated = runner.validate_run_contract(contract)
    manifest = runner.build_shard_manifest(
        run_contract=validated,
        source_role="candidate",
        work_hand_indices=range(10),
    )

    assert runner.contract_variant(validated) == (
        runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_VARIANT
    )
    assert validated["schema"] == (
        runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_RUN_CONTRACT_SCHEMA
    )
    assert validated["schedule"] == (
        runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_SCHEDULE
    )
    assert validated["seed_contract"] == (
        runner.candidate02_performance_lock_recovery_v3_seed_contract()
    )
    assert manifest["schema"] == (
        runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_SHARD_MANIFEST_SCHEMA
    )
    assert manifest["run_contract_digest"] == runner.canonical_sha256(validated)
    assert runner._source_hand_schema(validated) == (
        runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_SOURCE_HAND_SCHEMA
    )
    assert runner._done_schema(validated) == (
        runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_DONE_SCHEMA
    )

    changed = deepcopy(validated)
    changed["seed_contract"]["namespace_bases"]["hand"] += 1
    with pytest.raises(ValueError):
        runner.validate_run_contract(changed)


def test_v1_and_rearm1_contract_bytes_remain_unchanged() -> None:
    v1 = runner.build_run_contract(
        candidate_library_sha256=(
            runner.CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_CANDIDATE_LIBRARY_SHA256
        ),
        reference_library_sha256=(
            runner.CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_REFERENCE_LIBRARY_SHA256
        ),
        variant=runner.CANDIDATE02_PERFORMANCE_LOCK_VARIANT,
    )
    rearm1 = runner.build_run_contract(
        candidate_library_sha256=(
            runner.CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_CANDIDATE_LIBRARY_SHA256
        ),
        reference_library_sha256=(
            runner.CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_REFERENCE_LIBRARY_SHA256
        ),
        variant=runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT,
    )

    assert runner.canonical_sha256(v1) == (
        "e73c2b06279c1f1e91c38b2887ee465acf85f8f1072afef636512283252c34a4"
    )
    assert runner.canonical_sha256(rearm1) == (
        "f02e8845401eef92ba617f2c832204bdc74c20aa9e91810c52bf99687c4886b5"
    )


def test_recovery_v3_root_materialization_routes_with_v3_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = runner.build_run_contract(
        candidate_library_sha256=(
            runner.CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_CANDIDATE_LIBRARY_SHA256
        ),
        reference_library_sha256=(
            runner.CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_REFERENCE_LIBRARY_SHA256
        ),
        variant=runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_VARIANT,
    )
    sentinel = [{"hand_index": 3, "source": "recovery-v3"}]
    calls: list[dict[str, Any]] = []

    def materialize(**kwargs: Any) -> list[dict[str, Any]]:
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(
        runner,
        "_materialize_candidate02_performance_lock_recovery_roots",
        materialize,
    )
    result = runner._materialize_roots(
        contract=contract,
        repository_root=tmp_path,
        output_dir=tmp_path / "out",
        indices=[3],
    )
    assert result is sentinel
    assert calls == [
        {
            "repository_root": tmp_path,
            "output_dir": tmp_path / "out",
            "indices": [3],
            "recovery_v3": True,
        }
    ]
    row = runner.candidate02_performance_lock_recovery_v3_schedule_row(3)
    assert row["schedule"] == (
        runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_SCHEDULE
    )
    assert row["seeds"] == (
        runner.candidate02_performance_lock_recovery_v3_seed_values(3)
    )
