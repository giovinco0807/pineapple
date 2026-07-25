from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any, Mapping

import pytest

from ofc_regular import hu_m31_t3_step6d_full100_wave_cloud_v2 as subject
from ofc_regular import hu_m31_t3_step6d_full100_wave_plan_v2 as wave_v2


RUN_NAME = "regular-hu-m31-c02-f100wv2-20260722-004"
SALT = "1234567890abcdef1234567890abcdef"
PACKAGE_SHA = "2" * 64
IMAGE_DIGEST = "sha256:" + "3" * 64
CONTENT_SHA = "4" * 64
WRONG_CONTENT_SHA = "5" * 64
PROJECT = "ofc-project-123"
ZONE = "asia-northeast1-b"
NONCE = "12345678-1234-4234-9234-1234567890ab"


class FakeAtomicBackend:
    def __init__(self) -> None:
        self.objects: dict[str, dict[str, Any]] = {}
        self.put_count = 0
        self.get_count = 0

    def put_if_absent(
        self, *, object_name: str, payload: bytes
    ) -> Mapping[str, Any]:
        self.put_count += 1
        if object_name in self.objects:
            raise FileExistsError(object_name)
        generation = str(1000 + self.put_count)
        etag = f"etag-create-{generation}"
        row = {
            "object_name": object_name,
            "generation": generation,
            "etag": etag,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "payload": payload,
        }
        self.objects[object_name] = row
        return {"created": True, **{key: row[key] for key in row if key != "payload"}}

    def get_object(self, *, object_name: str) -> Mapping[str, Any]:
        self.get_count += 1
        return deepcopy(self.objects[object_name])


@pytest.fixture()
def context() -> tuple[dict, dict, dict]:
    plan = wave_v2.build_wave_plan(
        run_name=RUN_NAME,
        identity_salt=SALT,
        package_sha256=PACKAGE_SHA,
        image_digest=IMAGE_DIGEST,
    )
    baseline = wave_v2.build_observed_transition(
        plan,
        project_id=PROJECT,
        zone=ZONE,
        observed_at_utc="2026-07-22T01:00:00Z",
        previous_transition_digest=None,
        attempt_history=wave_v2.empty_attempt_history(plan),
    )
    ledger = wave_v2.build_attempt_ledger(plan, transitions=[baseline])
    resume = wave_v2.build_resume_plan(plan, attempt_ledger=ledger)
    return plan, ledger, resume


def _quota_rows(available: int = 128) -> list[dict[str, Any]]:
    return [
        {
            "metric": metric,
            "limit_vcpus": 256,
            "usage_vcpus": 256 - available,
            "available_vcpus": available,
            "readback_complete": True,
        }
        for metric in subject.QUOTA_METRICS
    ]


def _quota(plan: dict, ledger: dict, resume: dict, available: int = 128) -> dict:
    return subject.build_live_quota_headroom_receipt(
        plan,
        ledger,
        resume,
        immutable_content_sha256=CONTENT_SHA,
        project_id=PROJECT,
        zone=ZONE,
        observed_at_utc="2026-07-22T01:00:01Z",
        expires_at_utc="2026-07-22T01:05:01Z",
        quota_metrics=_quota_rows(available),
        readback_source="fake_backend_fixture",
    )


def _claim(
    plan: dict, ledger: dict, resume: dict, backend: FakeAtomicBackend
) -> dict:
    return subject.create_persistent_atomic_launch_claim(
        plan,
        ledger,
        resume,
        immutable_content_sha256=CONTENT_SHA,
        project_id=PROJECT,
        zone=ZONE,
        claim_nonce=NONCE,
        claimed_at_utc="2026-07-22T01:00:02Z",
        backend=backend,
    )


def _absence_rows(resume: dict) -> list[dict[str, Any]]:
    return [
        {
            "instance_id": row["instance_id"],
            "instance_absent": True,
            "boot_disk_absent": True,
            "readback_complete": True,
        }
        for row in resume["selected_attempts"]
    ]


def _mapping(plan: dict, ledger: dict, resume: dict) -> dict:
    return subject.build_planned_launch_mapping_receipt(
        plan,
        ledger,
        resume,
        immutable_content_sha256=CONTENT_SHA,
        project_id=PROJECT,
        zone=ZONE,
        observed_at_utc="2026-07-22T01:00:03Z",
        expires_at_utc="2026-07-22T01:05:03Z",
        instance_absence_readbacks=_absence_rows(resume),
        readback_source="fake_backend_fixture",
    )


def _authorization(
    plan: dict,
    ledger: dict,
    resume: dict,
    quota: dict,
    claim: dict,
    mapping: dict,
) -> dict:
    return subject.build_prelaunch_authorization(
        plan,
        ledger,
        resume,
        immutable_content_sha256=CONTENT_SHA,
        quota_receipt=quota,
        persistent_claim_receipt=claim,
        planned_mapping_receipt=mapping,
        raw_claim_nonce=NONCE,
        authorized_at_utc="2026-07-22T01:00:04Z",
        expires_at_utc="2026-07-22T01:03:04Z",
        explicit_launch_authorized=True,
    )


def _launch_observations(mapping: dict) -> list[dict[str, Any]]:
    return [
        {
            "job_id": row["job_id"],
            "source_role": row["source_role"],
            "attempt_id": row["attempt_id"],
            "instance_id": row["instance_id"],
            "operation_id": f"insert-{index:02d}-{row['instance_id']}",
            "operation_status": "DONE",
            "instance_status": "RUNNING",
            "ownership_label": row["ownership_label"],
            "machine_type": row["machine_type"],
            "vcpus": row["vcpus"],
            "instance_readback_complete": True,
        }
        for index, row in enumerate(mapping["rows"])
    ]


def _launch(
    plan: dict,
    ledger: dict,
    resume: dict,
    quota: dict,
    claim: dict,
    mapping: dict,
    authorization: dict,
) -> dict:
    return subject.build_actual_launch_receipt(
        plan,
        ledger,
        resume,
        immutable_content_sha256=CONTENT_SHA,
        quota_receipt=quota,
        persistent_claim_receipt=claim,
        planned_mapping_receipt=mapping,
        prelaunch_authorization=authorization,
        launch_started_at_utc="2026-07-22T01:00:05Z",
        observed_at_utc="2026-07-22T01:00:20Z",
        instance_create_readbacks=_launch_observations(mapping),
        readback_source="fake_backend_fixture",
    )


def _cleanup_observations(launch: dict) -> list[dict[str, Any]]:
    return [
        {
            "instance_id": row["instance_id"],
            "delete_operation_id": f"delete-{index:02d}-{row['instance_id']}",
            "delete_operation_status": "DONE",
            "instance_absent": True,
            "boot_disk_absent": True,
            "absence_readback_complete": True,
            "absence_readback_attempt_count": 2,
        }
        for index, row in enumerate(launch["rows"])
    ]


def _rehash(value: dict, digest_field: str) -> None:
    value[digest_field] = subject.canonical_sha256(
        {key: item for key, item in value.items() if key != digest_field}
    )


def _full_chain(context: tuple[dict, dict, dict]) -> dict[str, dict]:
    plan, ledger, resume = context
    backend = FakeAtomicBackend()
    quota = _quota(plan, ledger, resume)
    claim = _claim(plan, ledger, resume, backend)
    mapping = _mapping(plan, ledger, resume)
    authorization = _authorization(
        plan, ledger, resume, quota, claim, mapping
    )
    launch = _launch(
        plan, ledger, resume, quota, claim, mapping, authorization
    )
    cleanup = subject.build_cleanup_absence_receipt(
        plan,
        ledger,
        resume,
        immutable_content_sha256=CONTENT_SHA,
        quota_receipt=quota,
        persistent_claim_receipt=claim,
        planned_mapping_receipt=mapping,
        prelaunch_authorization=authorization,
        actual_launch_receipt=launch,
        observed_at_utc="2026-07-22T01:01:00Z",
        cleanup_readbacks=_cleanup_observations(launch),
        readback_source="fake_backend_fixture",
        owned_root_verified=True,
        parent_directory_fsync_completed=True,
    )
    return {
        "quota": quota,
        "claim": claim,
        "mapping": mapping,
        "authorization": authorization,
        "launch": launch,
        "cleanup": cleanup,
    }


def test_live_quota_receipt_is_exact_and_insufficient_headroom_never_authorizes(
    context: tuple[dict, dict, dict],
) -> None:
    plan, ledger, resume = context
    quota = _quota(plan, ledger, resume)
    assert quota["required_vcpus"] == 8 * 16
    assert quota["quota_sufficient"] is True
    assert subject.validate_live_quota_headroom_receipt(
        plan,
        ledger,
        resume,
        quota,
        immutable_content_sha256=CONTENT_SHA,
        now_utc="2026-07-22T01:04:00Z",
    ) == quota

    bad_arithmetic = deepcopy(quota)
    bad_arithmetic["quota_metrics"][0]["available_vcpus"] -= 1
    _rehash(bad_arithmetic, "receipt_sha256")
    with pytest.raises(ValueError, match="arithmetic"):
        subject.validate_live_quota_headroom_receipt(
            plan,
            ledger,
            resume,
            bad_arithmetic,
            immutable_content_sha256=CONTENT_SHA,
        )

    insufficient = _quota(plan, ledger, resume, available=112)
    assert insufficient["quota_sufficient"] is False
    backend = FakeAtomicBackend()
    claim = _claim(plan, ledger, resume, backend)
    mapping = _mapping(plan, ledger, resume)
    with pytest.raises(PermissionError, match="insufficient"):
        _authorization(plan, ledger, resume, insufficient, claim, mapping)
    with pytest.raises(PermissionError, match="stale"):
        subject.validate_live_quota_headroom_receipt(
            plan,
            ledger,
            resume,
            quota,
            immutable_content_sha256=CONTENT_SHA,
            now_utc="2026-07-22T01:05:02Z",
        )
    malformed = _quota_rows()
    malformed[0]["available_vcpus"] = None
    with pytest.raises(ValueError):
        subject.build_live_quota_headroom_receipt(
            plan,
            ledger,
            resume,
            immutable_content_sha256=CONTENT_SHA,
            project_id=PROJECT,
            zone=ZONE,
            observed_at_utc="2026-07-22T01:00:01Z",
            expires_at_utc="2026-07-22T01:05:01Z",
            quota_metrics=malformed,
            readback_source="fake_backend_fixture",
        )


def test_atomic_claim_is_create_only_and_binds_generation_etag_and_nonce(
    context: tuple[dict, dict, dict],
) -> None:
    plan, ledger, resume = context
    backend = FakeAtomicBackend()
    claim = _claim(plan, ledger, resume, backend)
    assert backend.put_count == 1
    assert backend.get_count == 1
    assert claim["generation"] == claim["readback_generation"]
    assert claim["etag"] == claim["readback_etag"]
    assert claim["existing_object_accepted"] is False
    assert subject.validate_persistent_atomic_launch_claim_receipt(
        plan,
        ledger,
        resume,
        claim,
        immutable_content_sha256=CONTENT_SHA,
        raw_claim_nonce=NONCE,
    ) == claim

    with pytest.raises(FileExistsError):
        _claim(plan, ledger, resume, backend)
    with pytest.raises(FileExistsError):
        subject.create_persistent_atomic_launch_claim(
            plan,
            ledger,
            resume,
            immutable_content_sha256=WRONG_CONTENT_SHA,
            project_id=PROJECT,
            zone=ZONE,
            claim_nonce="12345678-1234-4234-9234-1234567890ac",
            claimed_at_utc="2026-07-22T01:00:03Z",
            backend=backend,
        )
    assert claim["immutable_content_sha256"] == CONTENT_SHA
    assert claim["claim_payload"]["immutable_content_sha256"] == CONTENT_SHA

    wrong_nonce = "12345678-1234-4234-9234-1234567890ac"
    with pytest.raises(ValueError, match="does not match"):
        subject.validate_persistent_atomic_launch_claim_receipt(
            plan,
            ledger,
            resume,
            claim,
            immutable_content_sha256=CONTENT_SHA,
            raw_claim_nonce=wrong_nonce,
        )
    tampered = deepcopy(claim)
    tampered["readback_etag"] = "etag-create-different"
    _rehash(tampered, "receipt_sha256")
    with pytest.raises(ValueError, match="contract changed"):
        subject.validate_persistent_atomic_launch_claim_receipt(
            plan,
            ledger,
            resume,
            tampered,
            immutable_content_sha256=CONTENT_SHA,
        )


def test_planned_mapping_and_prelaunch_authorization_are_exact_one_shot(
    context: tuple[dict, dict, dict],
) -> None:
    plan, ledger, resume = context
    backend = FakeAtomicBackend()
    quota = _quota(plan, ledger, resume)
    claim = _claim(plan, ledger, resume, backend)
    mapping = _mapping(plan, ledger, resume)
    assert mapping["one_vm_one_job_one_role"] is True
    assert len(mapping["rows"]) == 8
    assert len({row["instance_id"] for row in mapping["rows"]}) == 8

    authorization = _authorization(
        plan, ledger, resume, quota, claim, mapping
    )
    assert authorization["authorized_create_count"] == 8
    assert authorization["authorized_vcpus"] == 128
    assert authorization["one_shot"] is True
    assert authorization["reuse_authorized"] is False
    assert authorization["unlisted_instance_create_authorized"] is False
    assert authorization["worker_iam_evidence_required_before_transport_create"] is True
    assert authorization["worker_iam_evidence_embedded"] is False
    assert subject.validate_prelaunch_authorization(
        plan,
        ledger,
        resume,
        immutable_content_sha256=CONTENT_SHA,
        quota_receipt=quota,
        persistent_claim_receipt=claim,
        planned_mapping_receipt=mapping,
        value=authorization,
        raw_claim_nonce=NONCE,
        now_utc="2026-07-22T01:01:00Z",
    ) == authorization

    changed_role = deepcopy(mapping)
    changed_role["rows"][0]["source_role"] = "reference"
    changed_role["mapping_sha256"] = subject.canonical_sha256(changed_role["rows"])
    _rehash(changed_role, "receipt_sha256")
    with pytest.raises(ValueError, match="one VM"):
        subject.validate_planned_launch_mapping_receipt(
            plan,
            ledger,
            resume,
            changed_role,
            immutable_content_sha256=CONTENT_SHA,
        )

    changed_evidence = deepcopy(authorization)
    changed_evidence["quota_receipt_sha256"] = "f" * 64
    _rehash(changed_evidence, "authorization_sha256")
    with pytest.raises(ValueError, match="evidence chain"):
        subject.validate_prelaunch_authorization(
            plan,
            ledger,
            resume,
            immutable_content_sha256=CONTENT_SHA,
            quota_receipt=quota,
            persistent_claim_receipt=claim,
            planned_mapping_receipt=mapping,
            value=changed_evidence,
        )
    before_evidence = deepcopy(authorization)
    before_evidence["authorized_at_utc"] = "2026-07-22T01:00:00Z"
    before_evidence["expires_at_utc"] = "2026-07-22T01:03:00Z"
    _rehash(before_evidence, "authorization_sha256")
    with pytest.raises(PermissionError, match="not live"):
        subject.validate_prelaunch_authorization(
            plan,
            ledger,
            resume,
            immutable_content_sha256=CONTENT_SHA,
            quota_receipt=quota,
            persistent_claim_receipt=claim,
            planned_mapping_receipt=mapping,
            value=before_evidence,
        )


def test_actual_launch_receipt_cannot_escape_one_vm_one_job_one_role(
    context: tuple[dict, dict, dict],
) -> None:
    plan, ledger, resume = context
    backend = FakeAtomicBackend()
    quota = _quota(plan, ledger, resume)
    claim = _claim(plan, ledger, resume, backend)
    mapping = _mapping(plan, ledger, resume)
    authorization = _authorization(
        plan, ledger, resume, quota, claim, mapping
    )
    launch = _launch(
        plan, ledger, resume, quota, claim, mapping, authorization
    )
    assert launch["created_instance_count"] == 8
    assert launch["additional_create_authorized"] is False
    assert subject.validate_actual_launch_receipt(
        plan,
        ledger,
        resume,
        immutable_content_sha256=CONTENT_SHA,
        quota_receipt=quota,
        persistent_claim_receipt=claim,
        planned_mapping_receipt=mapping,
        prelaunch_authorization=authorization,
        value=launch,
    ) == launch

    wrong_role = _launch_observations(mapping)
    wrong_role[0]["source_role"] = "reference"
    with pytest.raises(ValueError, match="planned job/role"):
        subject.build_actual_launch_receipt(
            plan,
            ledger,
            resume,
            immutable_content_sha256=CONTENT_SHA,
            quota_receipt=quota,
            persistent_claim_receipt=claim,
            planned_mapping_receipt=mapping,
            prelaunch_authorization=authorization,
            launch_started_at_utc="2026-07-22T01:00:05Z",
            observed_at_utc="2026-07-22T01:00:20Z",
            instance_create_readbacks=wrong_role,
            readback_source="fake_backend_fixture",
        )
    with pytest.raises(ValueError, match="cardinality"):
        subject.build_actual_launch_receipt(
            plan,
            ledger,
            resume,
            immutable_content_sha256=CONTENT_SHA,
            quota_receipt=quota,
            persistent_claim_receipt=claim,
            planned_mapping_receipt=mapping,
            prelaunch_authorization=authorization,
            launch_started_at_utc="2026-07-22T01:00:05Z",
            observed_at_utc="2026-07-22T01:00:20Z",
            instance_create_readbacks=_launch_observations(mapping)[:-1],
            readback_source="fake_backend_fixture",
        )
    after_expiry = deepcopy(launch)
    after_expiry["launch_started_at_utc"] = "2026-07-22T01:03:05Z"
    after_expiry["observed_at_utc"] = "2026-07-22T01:03:06Z"
    _rehash(after_expiry, "receipt_sha256")
    with pytest.raises(ValueError, match="contract changed"):
        subject.validate_actual_launch_receipt(
            plan,
            ledger,
            resume,
            immutable_content_sha256=CONTENT_SHA,
            quota_receipt=quota,
            persistent_claim_receipt=claim,
            planned_mapping_receipt=mapping,
            prelaunch_authorization=authorization,
            value=after_expiry,
        )


def test_cleanup_is_exact_owned_absence_and_lifecycle_grants_no_create(
    context: tuple[dict, dict, dict],
) -> None:
    plan, ledger, resume = context
    evidence = _full_chain(context)
    cleanup = evidence["cleanup"]
    assert cleanup["deleted_instance_count"] == 8
    assert cleanup["exact_owned_only"] is True
    assert cleanup["wildcard_delete_used"] is False
    assert cleanup["unrelated_instance_touched"] is False
    assert cleanup["owned_root_verified"] is True
    assert cleanup["parent_directory_fsync_completed"] is True

    attestation = subject.build_lifecycle_attestation(
        plan,
        ledger,
        resume,
        immutable_content_sha256=CONTENT_SHA,
        quota_receipt=evidence["quota"],
        persistent_claim_receipt=evidence["claim"],
        planned_mapping_receipt=evidence["mapping"],
        prelaunch_authorization=evidence["authorization"],
        actual_launch_receipt=evidence["launch"],
        cleanup_absence_receipt=cleanup,
        attested_at_utc="2026-07-22T01:01:01Z",
    )
    assert attestation["lifecycle_complete"] is True
    assert attestation["additional_create_authorized"] is False
    assert attestation["next_wave_requires_fresh_evidence"] is True

    wildcard = deepcopy(cleanup)
    wildcard["wildcard_delete_used"] = True
    _rehash(wildcard, "receipt_sha256")
    with pytest.raises(ValueError, match="contract changed"):
        subject.validate_cleanup_absence_receipt(
            plan,
            ledger,
            resume,
            immutable_content_sha256=CONTENT_SHA,
            quota_receipt=evidence["quota"],
            persistent_claim_receipt=evidence["claim"],
            planned_mapping_receipt=evidence["mapping"],
            prelaunch_authorization=evidence["authorization"],
            actual_launch_receipt=evidence["launch"],
            value=wildcard,
        )

    changed = deepcopy(attestation)
    changed["additional_create_authorized"] = True
    _rehash(changed, "attestation_sha256")
    with pytest.raises(ValueError, match="chain changed"):
        subject.validate_lifecycle_attestation(
            plan,
            ledger,
            resume,
            immutable_content_sha256=CONTENT_SHA,
            quota_receipt=evidence["quota"],
            persistent_claim_receipt=evidence["claim"],
            planned_mapping_receipt=evidence["mapping"],
            prelaunch_authorization=evidence["authorization"],
            actual_launch_receipt=evidence["launch"],
            cleanup_absence_receipt=cleanup,
            value=changed,
        )


def test_evidence_from_another_execution_identity_is_rejected(
    context: tuple[dict, dict, dict],
) -> None:
    plan, ledger, resume = context
    quota = _quota(plan, ledger, resume)
    other_plan = wave_v2.build_wave_plan(
        run_name=RUN_NAME,
        identity_salt="fedcba0987654321fedcba0987654321",
        package_sha256=PACKAGE_SHA,
        image_digest=IMAGE_DIGEST,
    )
    other_baseline = wave_v2.build_observed_transition(
        other_plan,
        project_id=PROJECT,
        zone=ZONE,
        observed_at_utc="2026-07-22T01:00:00Z",
        previous_transition_digest=None,
        attempt_history=wave_v2.empty_attempt_history(other_plan),
    )
    other_ledger = wave_v2.build_attempt_ledger(
        other_plan, transitions=[other_baseline]
    )
    other_resume = wave_v2.build_resume_plan(
        other_plan, attempt_ledger=other_ledger
    )
    with pytest.raises(ValueError, match="binding"):
        subject.validate_live_quota_headroom_receipt(
            other_plan,
            other_ledger,
            other_resume,
            quota,
            immutable_content_sha256=CONTENT_SHA,
        )


def test_wrong_outer_immutable_content_fails_closed_at_every_phase(
    context: tuple[dict, dict, dict],
) -> None:
    plan, ledger, resume = context
    evidence = _full_chain(context)
    attestation = subject.build_lifecycle_attestation(
        plan,
        ledger,
        resume,
        immutable_content_sha256=CONTENT_SHA,
        quota_receipt=evidence["quota"],
        persistent_claim_receipt=evidence["claim"],
        planned_mapping_receipt=evidence["mapping"],
        prelaunch_authorization=evidence["authorization"],
        actual_launch_receipt=evidence["launch"],
        cleanup_absence_receipt=evidence["cleanup"],
        attested_at_utc="2026-07-22T01:01:01Z",
    )
    assert all(
        receipt["immutable_content_sha256"] == CONTENT_SHA
        for receipt in [*evidence.values(), attestation]
    )

    with pytest.raises(ValueError, match="binding"):
        subject.validate_live_quota_headroom_receipt(
            plan,
            ledger,
            resume,
            evidence["quota"],
            immutable_content_sha256=WRONG_CONTENT_SHA,
        )
    with pytest.raises(ValueError, match="binding"):
        subject.validate_persistent_atomic_launch_claim_receipt(
            plan,
            ledger,
            resume,
            evidence["claim"],
            immutable_content_sha256=WRONG_CONTENT_SHA,
        )
    with pytest.raises(ValueError, match="binding"):
        subject.validate_planned_launch_mapping_receipt(
            plan,
            ledger,
            resume,
            evidence["mapping"],
            immutable_content_sha256=WRONG_CONTENT_SHA,
        )
    with pytest.raises(ValueError, match="binding"):
        subject.validate_prelaunch_authorization(
            plan,
            ledger,
            resume,
            immutable_content_sha256=WRONG_CONTENT_SHA,
            quota_receipt=evidence["quota"],
            persistent_claim_receipt=evidence["claim"],
            planned_mapping_receipt=evidence["mapping"],
            value=evidence["authorization"],
        )
    with pytest.raises(ValueError, match="binding"):
        subject.validate_actual_launch_receipt(
            plan,
            ledger,
            resume,
            immutable_content_sha256=WRONG_CONTENT_SHA,
            quota_receipt=evidence["quota"],
            persistent_claim_receipt=evidence["claim"],
            planned_mapping_receipt=evidence["mapping"],
            prelaunch_authorization=evidence["authorization"],
            value=evidence["launch"],
        )
    with pytest.raises(ValueError, match="binding"):
        subject.validate_cleanup_absence_receipt(
            plan,
            ledger,
            resume,
            immutable_content_sha256=WRONG_CONTENT_SHA,
            quota_receipt=evidence["quota"],
            persistent_claim_receipt=evidence["claim"],
            planned_mapping_receipt=evidence["mapping"],
            prelaunch_authorization=evidence["authorization"],
            actual_launch_receipt=evidence["launch"],
            value=evidence["cleanup"],
        )
    with pytest.raises(ValueError, match="binding"):
        subject.validate_lifecycle_attestation(
            plan,
            ledger,
            resume,
            immutable_content_sha256=WRONG_CONTENT_SHA,
            quota_receipt=evidence["quota"],
            persistent_claim_receipt=evidence["claim"],
            planned_mapping_receipt=evidence["mapping"],
            prelaunch_authorization=evidence["authorization"],
            actual_launch_receipt=evidence["launch"],
            cleanup_absence_receipt=evidence["cleanup"],
            value=attestation,
        )
