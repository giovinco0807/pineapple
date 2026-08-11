from __future__ import annotations

from copy import deepcopy

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_candidate02_performance_lock_rearm2_plan as subject,
)
from ofc_regular import run_hu_m31_t3_step6d_performance_v2 as runner


@pytest.fixture
def incident(monkeypatch: pytest.MonkeyPatch) -> dict:
    value = {
        "schema": subject.closeout.RECEIPT_SCHEMA,
        "status": subject.closeout.RECEIPT_STATUS,
        "run_name": subject.closeout.RUN_NAME,
        "run_contract_digest": subject.closeout.RUN_CONTRACT_DIGEST,
        "startup_logs": {
            "count": 20,
            "job_ids": list(subject.closeout.JOB_IDS),
            "aggregate_sha256": "a" * 64,
            "all_exact_schema_failures": True,
        },
        "control_plane_counts": {
            key: 0 for key in subject.closeout.COUNT_KEYS
        },
        "incident": {
            "attempt_index": 0,
            "failed_jobs": 20,
            "error": subject.closeout.ERROR_LINE,
            "actual_job_schema": (
                runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SHARD_MANIFEST_SCHEMA
            ),
            "startup_expected_job_schema": runner.SHARD_MANIFEST_SCHEMA,
            "deterministic_with_claimed_bytes": True,
        },
        "content_access": {
            "root_content_opened": False,
            "hand_content_opened": False,
            "result_content_opened": False,
            "cloud_mutated": False,
        },
        "disposition": {
            "attempt0_consumed": True,
            "current_run_irrecoverable": True,
            "rearm1_attempt1_authorized": False,
            "rearm1_package_reuse_authorized": False,
            "rearm1_root_reuse_authorized": False,
            "rearm1_seed_reuse_authorized": False,
            "rearm1_claim_reuse_authorized": False,
            "fresh_rearm2_plan_authorized": False,
            "training_eligible": False,
            "current_profile_changed": False,
        },
    }
    monkeypatch.setattr(
        subject, "STARTUP_FAILURE_RECEIPT_SHA256", subject.canonical_sha256(value)
    )
    return value


def _freeze(
    monkeypatch: pytest.MonkeyPatch, incident: dict
) -> tuple[dict, str]:
    monkeypatch.setattr(
        subject,
        "PRECONTENT_PLAN_SHA256",
        subject.PRECONTENT_PLAN_SHA256_PLACEHOLDER,
    )
    digest = subject.compute_precontent_plan_sha256(incident_receipt=incident)
    monkeypatch.setattr(subject, "PRECONTENT_PLAN_SHA256", digest)
    return subject.build_precontent_plan(incident_receipt=incident), digest


def test_rearm2_plan_uses_fresh_v3_seed_schedule_and_preserves_binaries(
    incident: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, digest = _freeze(monkeypatch, incident)
    contract = plan["run_contract"]
    seeds = contract["seed_contract"]

    assert subject.canonical_sha256(plan) == digest
    assert plan["schema"] == subject.PLAN_SCHEMA
    assert plan["scope"] == subject.PLAN_SCOPE
    assert plan["candidate_variant"] == (
        runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_VARIANT
    )
    assert runner.contract_variant(contract) == plan["candidate_variant"]
    assert plan["run_contract_digest"] == subject.RECOVERY_V3_RUN_CONTRACT_DIGEST
    assert seeds["seed_min"] == 710_108_071_901
    assert seeds["performance_lock_v1_overlap_count"] == 0
    assert seeds["performance_lock_rearm1_overlap_count"] == 0
    assert seeds["existing_step6d_union_overlap_count"] == 0
    assert contract["candidate_library_sha256"] == subject.CANDIDATE_LIBRARY_SHA256
    assert contract["reference_library_sha256"] == subject.REFERENCE_LIBRARY_SHA256
    assert plan["current_profile_changed"] is False


def test_rearm1_is_terminal_and_every_identity_reuse_is_forbidden(
    incident: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _ = _freeze(monkeypatch, incident)
    old = plan["rearm1_disposition"]
    authority = plan["fresh_lock_authority"]

    assert old["attempt0_consumed"] is True
    assert old["rearm1_attempt1_authorized"] is False
    assert old["rearm1_package_reuse_authorized"] is False
    assert old["rearm1_root_reuse_authorized"] is False
    assert old["rearm1_seed_reuse_authorized"] is False
    assert old["rearm1_claim_reuse_authorized"] is False
    assert authority["fresh_lock_ordinal"] == "rearm2"
    assert authority["fresh_root_set_required"] is True
    assert authority["fresh_seed_set_required"] is True
    assert authority["fresh_global_claim_required_before_root_content"] is True
    assert authority["rearm1_attempt1_authorized"] is False
    assert authority["rearm1_package_authorized"] is False
    assert authority["rearm1_roots_authorized"] is False
    assert authority["rearm1_seeds_authorized"] is False


def test_actual_package_exhaustive_smoke_is_required_before_authorization(
    incident: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _ = _freeze(monkeypatch, incident)
    smoke = plan["actual_package_smoke_requirement"]
    authority = plan["fresh_lock_authority"]

    assert smoke == {
        "schema": subject.ACTUAL_PACKAGE_SMOKE_SCHEMA,
        "status": subject.ACTUAL_PACKAGE_SMOKE_STATUS,
        "authorization_requires_exhaustive_actual_package_smoke": True,
        "write_once_receipt_required": True,
        "verified_job_count": 20,
        "startup_invocation_count": 20,
        "root_read": False,
        "cloud_mutated": False,
        "global_spot_claim_and_authorization_bind_receipt_sha256": True,
    }
    assert authority["fresh_spot_execution_authorized"] is False
    assert authority["authorization_before_actual_package_smoke_allowed"] is False
    assert plan["cloud_started"] is False


def test_jobs_are_balanced_and_use_the_v3_shard_schema(
    incident: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _ = _freeze(monkeypatch, incident)
    jobs = plan["jobs"]
    assert len(jobs) == 20
    assert [row["job_id"] for row in jobs] == list(subject.JOB_IDS)
    assert all(
        runner.build_shard_manifest(
            run_contract=plan["run_contract"],
            source_role=row["source_role"],
            work_hand_indices=row["work_hand_indices"],
        )["schema"]
        == runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_SHARD_MANIFEST_SCHEMA
        for row in jobs
    )


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("rearm1_disposition", "rearm1_attempt1_authorized"), True),
        (("fresh_lock_authority", "rearm1_package_authorized"), True),
        (
            (
                "actual_package_smoke_requirement",
                "authorization_requires_exhaustive_actual_package_smoke",
            ),
            False,
        ),
        (("run_contract", "seed_contract", "performance_lock_rearm1_overlap_count"), 1),
        (("current_profile_changed",), True),
    ),
)
def test_authority_seed_smoke_and_current_tamper_fail_closed(
    path: tuple[object, ...],
    value: object,
    incident: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, _ = _freeze(monkeypatch, incident)
    mutated = deepcopy(plan)
    target: object = mutated
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    with pytest.raises(ValueError):
        subject.validate_precontent_plan(mutated, incident_receipt=incident)
