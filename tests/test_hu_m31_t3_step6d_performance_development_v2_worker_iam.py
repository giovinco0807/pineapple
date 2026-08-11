from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_performance_development_v2_worker_iam as subject,
)


ISSUED = 1_800_000_000
RAW_NONCE = "perfdev-v2-worker-iam-one-shot-nonce-0001"
NONCE_SHA = hashlib.sha256(RAW_NONCE.encode("utf-8")).hexdigest()
EXPIRED_CLEANUP_NONCE = "550e8400-e29b-41d4-a716-446655440000"
CONDITION_EXPIRES = ISSUED + subject.CONDITION_LIFETIME_SECONDS
EXPIRED_CLEANUP_NOT_BEFORE = (
    CONDITION_EXPIRES + subject.EXPIRED_EXACT2_GRACE_SECONDS
)


def _execution_plan() -> dict[str, Any]:
    run_name = "regular-hu-m31-c02-perfdev-v2-20300115-iam01"
    return {
        "project": subject.PROJECT,
        "region": "asia-northeast1",
        "run_name": run_name,
        "result_prefix": f"hu-m31-t3/perfdev-v2/{run_name}/",
        "worker_identity": {
            "service_account": subject.WORKER_SERVICE_ACCOUNT,
        },
        "limits": {"vm_ttl_seconds": 4_500},
        "instances": [
            {
                "source_role": "candidate",
                "instance_name": "m31-pdv2-candidate",
                "ownership_label": "pdv2-00000000000000000000",
            },
            {
                "source_role": "reference",
                "instance_name": "m31-pdv2-reference",
                "ownership_label": "pdv2-00000000000000000000",
            },
        ],
    }


@pytest.fixture(autouse=True)
def _stub_execution_plan_validator(monkeypatch: pytest.MonkeyPatch) -> None:
    def validate(value: Mapping[str, Any], **_: Any) -> dict[str, Any]:
        plan = copy.deepcopy(dict(value))
        if plan != _execution_plan():
            raise ValueError("fixture execution plan changed")
        return plan

    monkeypatch.setattr(subject.cloud, "validate_execution_plan", validate)


def _plan() -> dict[str, Any]:
    return subject.build_iam_plan(
        execution_plan=_execution_plan(),
        one_shot_nonce_sha256=NONCE_SHA,
        issued_at_unix_seconds=ISSUED,
    )


def _role(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "title": name.rsplit("/", 1)[-1],
        "description": "frozen test role",
        "includedPermissions": list(subject.ROLE_PERMISSIONS[name]),
        "stage": "GA",
        "etag": "BwWRoleEtag==",
        "deleted": False,
    }


def _bucket_policy() -> dict[str, Any]:
    return {
        "kind": "storage#policy",
        "resourceId": f"projects/_/buckets/{subject.BUCKET}",
        "version": 3,
        "etag": "BwWBucketEtag==",
        "bindings": [
            {
                "role": "roles/storage.legacyBucketReader",
                "members": ["projectViewer:unrelated-project"],
            },
            {
                "role": "roles/storage.objectViewer",
                "members": ["user:unrelated@example.com"],
                "condition": {
                    "title": "unrelated-condition",
                    "expression": (
                        'resource.name.startsWith("projects/_/buckets/other/objects/x/")'
                    ),
                },
            },
        ],
    }


def _project_policy() -> dict[str, Any]:
    return {
        "version": 1,
        "etag": "BwWProjectEtag==",
        "bindings": [
            {
                "role": "roles/owner",
                "members": ["user:controller@example.com"],
            }
        ],
        "auditConfigs": [],
    }


class FakeTransport:
    def __init__(self) -> None:
        self.roles = {
            name: _role(name) for name in (subject.READER_ROLE, subject.CREATOR_ROLE)
        }
        self.bucket_policy = _bucket_policy()
        self.project_policy = _project_policy()
        self.set_count = 0
        self.set_inputs: list[dict[str, Any]] = []

    def get_role(self, *, role_name: str) -> Mapping[str, Any]:
        return copy.deepcopy(self.roles[role_name])

    def get_bucket_policy(self) -> Mapping[str, Any]:
        return copy.deepcopy(self.bucket_policy)

    def get_project_policy(self) -> Mapping[str, Any]:
        return copy.deepcopy(self.project_policy)

    def set_bucket_policy(self, *, policy: Mapping[str, Any]) -> Mapping[str, Any]:
        supplied = copy.deepcopy(dict(policy))
        assert supplied["etag"] == self.bucket_policy["etag"]
        self.set_count += 1
        self.set_inputs.append(supplied)
        supplied["etag"] = f"BwWBucketEtag{self.set_count}=="
        self.bucket_policy = supplied
        return copy.deepcopy(supplied)


def _installed(
    fake: FakeTransport,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan = _plan()
    prepare = subject.prepare_worker_iam(
        iam_plan=plan, transport=fake, now_unix_seconds=ISSUED
    )
    authorization = subject.build_explicit_authorization(
        iam_plan=plan, raw_nonce=RAW_NONCE, operation="apply"
    )
    install = subject.apply_worker_iam(
        iam_plan=plan,
        prepare_receipt=prepare,
        authorization=authorization,
        raw_nonce=RAW_NONCE,
        transport=fake,
        execute=True,
        now_unix_seconds=ISSUED,
    )
    return plan, prepare, install


def _readback(
    fake: FakeTransport,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan, prepare, install = _installed(fake)
    receipt = subject.readback_worker_iam(
        iam_plan=plan,
        prepare_receipt=prepare,
        install_receipt=install,
        transport=fake,
        now_unix_seconds=ISSUED,
    )
    return plan, prepare, install, receipt


def test_plan_is_bucket_scoped_time_bounded_and_nonce_sealed() -> None:
    plan = _plan()
    assert plan["bucket_iam_only"] is True
    assert plan["service_account_act_as_binding_managed"] is False
    assert plan["condition_lifetime_seconds"] == 5_400
    assert plan["expires_at_utc"] == subject._rfc3339(ISSUED + 5_400)
    assert plan["one_shot_nonce_sha256"] == NONCE_SHA
    assert RAW_NONCE not in json.dumps(plan)
    reader, creator = plan["expected_bindings"]
    assert reader["role"] == subject.READER_ROLE
    assert creator["role"] == subject.CREATOR_ROLE
    assert f"projects/_/buckets/{subject.BUCKET}/objects/" in reader["condition"]["expression"]
    assert plan["result_prefix"] + "control/" in reader["condition"]["expression"]
    assert plan["result_prefix"] in creator["condition"]["expression"]
    for row in (reader, creator):
        expression = row["condition"]["expression"]
        assert " && request.time < timestamp(" in expression
        assert plan["expires_at_utc"] in expression
        assert "*" not in expression
    assert subject.validate_iam_plan(plan) == plan


def test_plan_tamper_and_unsafe_live_window_fail_closed() -> None:
    plan = _plan()
    changed = copy.deepcopy(plan)
    changed["expected_bindings"][0]["condition"]["expression"] = "true"
    with pytest.raises(ValueError, match="digest"):
        subject.validate_iam_plan(changed)
    fake = FakeTransport()
    with pytest.raises(ValueError, match="launch-safe"):
        subject.prepare_worker_iam(
            iam_plan=plan,
            transport=fake,
            now_unix_seconds=ISSUED + 601,
        )
    assert fake.set_count == 0


def test_prepare_validates_roles_bucket_zero_and_project_direct_zero() -> None:
    fake = FakeTransport()
    receipt = subject.prepare_worker_iam(
        iam_plan=_plan(), transport=fake, now_unix_seconds=ISSUED
    )
    assert receipt["targeted_binding_count"] == 0
    assert receipt["project_worker_zero_readback"]["direct_worker_binding_count"] == 0
    assert receipt["cloud_mutation_performed"] is False
    assert "etag" not in json.dumps(receipt).lower().replace("etag_sha256", "")
    assert fake.set_count == 0


@pytest.mark.parametrize("drift", ["permission", "stage", "deleted"])
def test_prepare_rejects_custom_role_drift(drift: str) -> None:
    fake = FakeTransport()
    role = fake.roles[subject.READER_ROLE]
    if drift == "permission":
        role["includedPermissions"] = ["storage.objects.list"]
    elif drift == "stage":
        role["stage"] = "BETA"
    else:
        role["deleted"] = True
    with pytest.raises(ValueError, match="custom role"):
        subject.prepare_worker_iam(
            iam_plan=_plan(), transport=fake, now_unix_seconds=ISSUED
        )
    assert fake.set_count == 0


def test_prepare_rejects_any_excess_worker_binding_on_bucket_or_project() -> None:
    plan = _plan()
    fake = FakeTransport()
    fake.bucket_policy["bindings"].append(
        {"role": "roles/storage.objectAdmin", "members": [subject.WORKER_PRINCIPAL]}
    )
    with pytest.raises(ValueError, match="stale or drifted"):
        subject.prepare_worker_iam(
            iam_plan=plan, transport=fake, now_unix_seconds=ISSUED
        )
    fake = FakeTransport()
    fake.project_policy["bindings"].append(
        {"role": "roles/viewer", "members": [subject.WORKER_PRINCIPAL]}
    )
    with pytest.raises(ValueError, match="direct project"):
        subject.prepare_worker_iam(
            iam_plan=plan, transport=fake, now_unix_seconds=ISSUED
        )


def test_apply_dry_run_requires_authorization_and_performs_zero_mutations() -> None:
    fake = FakeTransport()
    plan = _plan()
    prepare = subject.prepare_worker_iam(
        iam_plan=plan, transport=fake, now_unix_seconds=ISSUED
    )
    authorization = subject.build_explicit_authorization(
        iam_plan=plan, raw_nonce=RAW_NONCE, operation="apply"
    )
    receipt = subject.apply_worker_iam(
        iam_plan=plan,
        prepare_receipt=prepare,
        authorization=authorization,
        raw_nonce=RAW_NONCE,
        transport=fake,
        execute=False,
        now_unix_seconds=ISSUED,
    )
    assert receipt["status"] == "dry_run_validated_no_mutation"
    assert receipt["set_attempt_count"] == 0
    assert receipt["cloud_mutation_performed"] is False
    assert fake.set_count == 0
    with pytest.raises(ValueError, match="nonce"):
        subject.apply_worker_iam(
            iam_plan=plan,
            prepare_receipt=prepare,
            authorization=authorization,
            raw_nonce="wrong-one-shot-nonce-value",
            transport=fake,
            execute=True,
            now_unix_seconds=ISSUED,
        )
    assert fake.set_count == 0


def test_apply_detects_etag_or_policy_drift_before_single_cas() -> None:
    fake = FakeTransport()
    plan = _plan()
    prepare = subject.prepare_worker_iam(
        iam_plan=plan, transport=fake, now_unix_seconds=ISSUED
    )
    authorization = subject.build_explicit_authorization(
        iam_plan=plan, raw_nonce=RAW_NONCE, operation="apply"
    )
    fake.bucket_policy["etag"] = "BwWUnexpectedEtag=="
    with pytest.raises(ValueError, match="ETag"):
        subject.apply_worker_iam(
            iam_plan=plan,
            prepare_receipt=prepare,
            authorization=authorization,
            raw_nonce=RAW_NONCE,
            transport=fake,
            execute=True,
            now_unix_seconds=ISSUED,
        )
    assert fake.set_count == 0


def test_apply_installs_exact_two_and_preserves_every_unrelated_binding() -> None:
    fake = FakeTransport()
    before = copy.deepcopy(fake.bucket_policy["bindings"])
    plan, prepare, install = _installed(fake)
    assert fake.set_count == 1
    assert install["installed_binding_count"] == 2
    assert install["set_attempt_count"] == 1
    assert install["project_worker_zero_readback"]["direct_worker_binding_count"] == 0
    assert fake.bucket_policy["bindings"][: len(before)] == before
    assert fake.bucket_policy["bindings"][len(before) :] == plan["expected_bindings"]
    assert install["unrelated_policy_fingerprint_sha256"] == prepare[
        "unrelated_policy_fingerprint_sha256"
    ]


def test_readback_matches_cloud_launch_worker_iam_contract() -> None:
    fake = FakeTransport()
    plan, _prepare, _install, receipt = _readback(fake)
    launch = receipt["launch_preflight"]
    assert launch == {
        "read_only": True,
        "service_account": subject.WORKER_SERVICE_ACCOUNT,
        "bucket": subject.BUCKET,
        "required_reader_prefix": f"{plan['result_prefix']}control/",
        "required_creator_prefix": plan["result_prefix"],
        "exact_conditional_binding_count": 2,
        "reader_binding_count": 1,
        "creator_binding_count": 1,
        "excess_worker_binding_count": 0,
        "iam_expiry_unix_seconds": ISSUED + 5_400,
        "object_get_allowed": True,
        "object_create_allowed": True,
        "object_list_required": False,
        "exact_prefix_condition": True,
        "all_required_permissions_present": True,
    }
    assert receipt["project_worker_zero_readback"]["direct_worker_binding_count"] == 0


def test_readback_worker_view_is_accepted_by_cloud_namespace_gate() -> None:
    fake = FakeTransport()
    plan, _prepare, _install, receipt = _readback(fake)
    instance_names = [row["instance_name"] for row in plan["execution_plan"]["instances"]]
    observation = {
        "read_only": True,
        "cloud_mutated": False,
        "run_name_collision_count": 0,
        "identity_collision_count": 0,
        "result_prefix_object_count": 0,
        "instance_collision_counts": {name: 0 for name in instance_names},
        "worker_iam": receipt["launch_preflight"],
        "network_path": {
            "read_only": True,
            "router_name": subject.cloud.NAT_ROUTER_NAME,
            "router_region": plan["execution_plan"]["region"],
            "router_network_exact": True,
            "nat_name": subject.cloud.NAT_NAME,
            "nat_count_with_name": 1,
            "nat_ip_allocate_option": "AUTO_ONLY",
            "source_subnetwork_ip_ranges_to_nat": "ALL_SUBNETWORKS_ALL_IP_RANGES",
            "external_ipv4_on_vm": False,
            "path_ready": True,
        },
    }
    assert subject.cloud._validate_namespace_observation(
        observation, plan["execution_plan"]
    ) == observation


def test_readback_rejects_condition_drift_duplicate_or_excess_binding() -> None:
    fake = FakeTransport()
    plan, prepare, install = _installed(fake)
    fake.bucket_policy["bindings"][-1]["condition"]["expression"] += " "
    with pytest.raises(ValueError, match="drifted"):
        subject.readback_worker_iam(
            iam_plan=plan,
            prepare_receipt=prepare,
            install_receipt=install,
            transport=fake,
            now_unix_seconds=ISSUED,
        )
    fake = FakeTransport()
    plan, prepare, install = _installed(fake)
    duplicate = copy.deepcopy(plan["expected_bindings"][0])
    duplicate["members"] = ["user:someone-else@example.com"]
    fake.bucket_policy["bindings"].append(duplicate)
    with pytest.raises(ValueError, match="duplicate"):
        subject.readback_worker_iam(
            iam_plan=plan,
            prepare_receipt=prepare,
            install_receipt=install,
            transport=fake,
            now_unix_seconds=ISSUED,
        )


def _collection(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_name": plan["run_name"],
        "execution_plan_sha256": plan["execution_plan_sha256"],
        "receipt_content_sha256": "a" * 64,
    }


def test_cleanup_requires_terminal_evidence_and_dry_run_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeTransport()
    plan, prepare, install, readback = _readback(fake)
    authorization = subject.build_explicit_authorization(
        iam_plan=plan, raw_nonce=RAW_NONCE, operation="cleanup"
    )
    with pytest.raises(ValueError, match="exactly one"):
        subject.cleanup_worker_iam(
            iam_plan=plan,
            prepare_receipt=prepare,
            install_receipt=install,
            readback_receipt=readback,
            authorization=authorization,
            raw_nonce=RAW_NONCE,
            transport=fake,
        )
    monkeypatch.setattr(subject.cloud, "_validate_collection_receipt", lambda value: dict(value))
    receipt = subject.cleanup_worker_iam(
        iam_plan=plan,
        prepare_receipt=prepare,
        install_receipt=install,
        readback_receipt=readback,
        authorization=authorization,
        raw_nonce=RAW_NONCE,
        transport=fake,
        collection_receipt=_collection(plan),
        execute=False,
    )
    assert receipt["status"] == "dry_run_validated_no_mutation"
    assert receipt["set_attempt_count"] == 0
    assert fake.set_count == 1


def test_cleanup_removes_only_owned_two_and_post_readback_is_worker_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeTransport()
    unrelated = copy.deepcopy(fake.bucket_policy["bindings"])
    plan, prepare, install, readback = _readback(fake)
    authorization = subject.build_explicit_authorization(
        iam_plan=plan, raw_nonce=RAW_NONCE, operation="cleanup"
    )
    monkeypatch.setattr(subject.cloud, "_validate_collection_receipt", lambda value: dict(value))
    receipt = subject.cleanup_worker_iam(
        iam_plan=plan,
        prepare_receipt=prepare,
        install_receipt=install,
        readback_receipt=readback,
        authorization=authorization,
        raw_nonce=RAW_NONCE,
        transport=fake,
        collection_receipt=_collection(plan),
        execute=True,
    )
    assert fake.set_count == 2
    assert fake.bucket_policy["bindings"] == unrelated
    assert receipt["removed_binding_count"] == 2
    assert receipt["remaining_targeted_binding_count"] == 0
    assert receipt["project_worker_zero_readback_before"]["direct_worker_binding_count"] == 0
    assert receipt["project_worker_zero_readback_after"]["direct_worker_binding_count"] == 0


def test_cleanup_accepts_only_cloud_validated_owned_partial_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeTransport()
    plan, prepare, install, readback = _readback(fake)
    authorization = subject.build_explicit_authorization(
        iam_plan=plan, raw_nonce=RAW_NONCE, operation="cleanup"
    )
    partial = {"receipt_content_sha256": "b" * 64}
    calls: list[dict[str, Any]] = []

    def validate(value: Mapping[str, Any], *, execution_plan: Mapping[str, Any]) -> dict[str, Any]:
        assert dict(execution_plan) == plan["execution_plan"]
        calls.append(dict(value))
        return dict(value)

    monkeypatch.setattr(subject.cloud, "_validate_partial_launch_receipt", validate)
    receipt = subject.cleanup_worker_iam(
        iam_plan=plan,
        prepare_receipt=prepare,
        install_receipt=install,
        readback_receipt=readback,
        authorization=authorization,
        raw_nonce=RAW_NONCE,
        transport=fake,
        partial_launch_receipt=partial,
        execute=False,
    )
    assert calls == [partial]
    assert receipt["terminal_evidence_kind"] == "owned_partial_launch"


def test_cleanup_accepts_cloud_validated_owned_launch_failure_closeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeTransport()
    plan, prepare, install, readback = _readback(fake)
    authorization = subject.build_explicit_authorization(
        iam_plan=plan, raw_nonce=RAW_NONCE, operation="cleanup"
    )
    closeout = {
        "schema": "hu_m31_t3_step6d_perfdev_v2_owned_launch_failure_closeout_v1",
        "run_name": plan["run_name"],
        "execution_plan_sha256": plan["execution_plan_sha256"],
        "operator_abort": True,
        "scientific_result_claimed": False,
        "all_owned_instances_deleted": True,
        "receipt_content_sha256": "c" * 64,
    }
    calls: list[dict[str, Any]] = []

    def validate(
        value: Mapping[str, Any], *, execution_plan: Mapping[str, Any]
    ) -> dict[str, Any]:
        assert dict(execution_plan) == plan["execution_plan"]
        calls.append(dict(value))
        return dict(value)

    monkeypatch.setattr(
        subject.cloud,
        "_validate_owned_launch_failure_closeout_receipt",
        validate,
        raising=False,
    )
    receipt = subject.cleanup_worker_iam(
        iam_plan=plan,
        prepare_receipt=prepare,
        install_receipt=install,
        readback_receipt=readback,
        authorization=authorization,
        raw_nonce=RAW_NONCE,
        transport=fake,
        owned_launch_failure_closeout_receipt=closeout,
        execute=False,
    )
    assert calls == [closeout]
    assert receipt["terminal_evidence_kind"] == "owned_launch_failure_closeout"
    assert receipt["terminal_evidence_sha256"] == "c" * 64
    assert fake.set_count == 1

    changed = dict(closeout)
    changed["all_owned_instances_deleted"] = False
    with pytest.raises(ValueError, match="closeout evidence"):
        subject.cleanup_worker_iam(
            iam_plan=plan,
            prepare_receipt=prepare,
            install_receipt=install,
            readback_receipt=readback,
            authorization=authorization,
            raw_nonce=RAW_NONCE,
            transport=fake,
            owned_launch_failure_closeout_receipt=changed,
            execute=False,
        )


def test_cleanup_accepts_expired_exact2_only_after_expiry_grace() -> None:
    fake = FakeTransport()
    plan, prepare, install, readback = _readback(fake)
    authorization = subject.build_expired_exact2_cleanup_authorization(
        iam_plan=plan,
        raw_cleanup_nonce=EXPIRED_CLEANUP_NONCE,
        issued_at_unix_seconds=EXPIRED_CLEANUP_NOT_BEFORE,
    )
    receipt = subject.cleanup_worker_iam(
        iam_plan=plan,
        prepare_receipt=prepare,
        install_receipt=install,
        readback_receipt=readback,
        authorization=authorization,
        raw_nonce=None,
        transport=fake,
        expired_exact2=True,
        raw_expired_cleanup_nonce=EXPIRED_CLEANUP_NONCE,
        execute=False,
        now_unix_seconds=EXPIRED_CLEANUP_NOT_BEFORE,
    )
    assert receipt["terminal_evidence_kind"] == "expired_exact2_condition_inert"
    assert (
        receipt["terminal_evidence_sha256"]
        == authorization["authorization_sha256"]
    )
    assert receipt["condition_effective_during_cleanup"] is False
    assert receipt["cleanup_not_before_unix_seconds"] == EXPIRED_CLEANUP_NOT_BEFORE
    assert receipt["cleanup_observed_unix_seconds"] == EXPIRED_CLEANUP_NOT_BEFORE
    assert receipt["zero_created_claimed"] is False
    assert fake.set_count == 1


def test_expired_exact2_cleanup_rejects_early_stale_or_wrong_nonce() -> None:
    fake = FakeTransport()
    plan, prepare, install, readback = _readback(fake)
    with pytest.raises(ValueError, match="precedes expiry grace"):
        subject.build_expired_exact2_cleanup_authorization(
            iam_plan=plan,
            raw_cleanup_nonce=EXPIRED_CLEANUP_NONCE,
            issued_at_unix_seconds=EXPIRED_CLEANUP_NOT_BEFORE - 1,
        )
    launch_nonce_reuse_plan = subject.build_iam_plan(
        execution_plan=_execution_plan(),
        one_shot_nonce_sha256=hashlib.sha256(
            EXPIRED_CLEANUP_NONCE.encode("utf-8")
        ).hexdigest(),
        issued_at_unix_seconds=ISSUED,
    )
    with pytest.raises(ValueError, match="reused the launch nonce"):
        subject.build_expired_exact2_cleanup_authorization(
            iam_plan=launch_nonce_reuse_plan,
            raw_cleanup_nonce=EXPIRED_CLEANUP_NONCE,
            issued_at_unix_seconds=EXPIRED_CLEANUP_NOT_BEFORE,
        )
    authorization = subject.build_expired_exact2_cleanup_authorization(
        iam_plan=plan,
        raw_cleanup_nonce=EXPIRED_CLEANUP_NONCE,
        issued_at_unix_seconds=EXPIRED_CLEANUP_NOT_BEFORE,
    )
    with pytest.raises(ValueError, match="not live"):
        subject.cleanup_worker_iam(
            iam_plan=plan,
            prepare_receipt=prepare,
            install_receipt=install,
            readback_receipt=readback,
            authorization=authorization,
            raw_nonce=None,
            transport=fake,
            expired_exact2=True,
            raw_expired_cleanup_nonce=EXPIRED_CLEANUP_NONCE,
            execute=False,
            now_unix_seconds=EXPIRED_CLEANUP_NOT_BEFORE - 1,
        )
    with pytest.raises(ValueError, match="not live"):
        subject.cleanup_worker_iam(
            iam_plan=plan,
            prepare_receipt=prepare,
            install_receipt=install,
            readback_receipt=readback,
            authorization=authorization,
            raw_nonce=None,
            transport=fake,
            expired_exact2=True,
            raw_expired_cleanup_nonce=EXPIRED_CLEANUP_NONCE,
            execute=False,
            now_unix_seconds=(
                EXPIRED_CLEANUP_NOT_BEFORE
                + subject.EXPIRED_EXACT2_AUTHORIZATION_LIFETIME_SECONDS
                + 1
            ),
        )
    with pytest.raises(ValueError, match="not live"):
        subject.cleanup_worker_iam(
            iam_plan=plan,
            prepare_receipt=prepare,
            install_receipt=install,
            readback_receipt=readback,
            authorization=authorization,
            raw_nonce=None,
            transport=fake,
            expired_exact2=True,
            raw_expired_cleanup_nonce="123e4567-e89b-42d3-a456-426614174000",
            execute=False,
            now_unix_seconds=EXPIRED_CLEANUP_NOT_BEFORE,
        )


def test_expired_exact2_execute_removes_only_plan_owned_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeTransport()
    unrelated = copy.deepcopy(fake.bucket_policy["bindings"])
    plan, prepare, install, readback = _readback(fake)
    authorization = subject.build_expired_exact2_cleanup_authorization(
        iam_plan=plan,
        raw_cleanup_nonce=EXPIRED_CLEANUP_NONCE,
        issued_at_unix_seconds=EXPIRED_CLEANUP_NOT_BEFORE,
    )
    monkeypatch.setattr(subject.time, "time", lambda: EXPIRED_CLEANUP_NOT_BEFORE)
    with pytest.raises(ValueError, match="cannot use an injected clock"):
        subject.cleanup_worker_iam(
            iam_plan=plan,
            prepare_receipt=prepare,
            install_receipt=install,
            readback_receipt=readback,
            authorization=authorization,
            raw_nonce=None,
            transport=fake,
            expired_exact2=True,
            raw_expired_cleanup_nonce=EXPIRED_CLEANUP_NONCE,
            execute=True,
            now_unix_seconds=EXPIRED_CLEANUP_NOT_BEFORE,
        )
    receipt = subject.cleanup_worker_iam(
        iam_plan=plan,
        prepare_receipt=prepare,
        install_receipt=install,
        readback_receipt=readback,
        authorization=authorization,
        raw_nonce=None,
        transport=fake,
        expired_exact2=True,
        raw_expired_cleanup_nonce=EXPIRED_CLEANUP_NONCE,
        execute=True,
    )
    assert fake.set_count == 2
    assert fake.bucket_policy["bindings"] == unrelated
    assert receipt["removed_binding_count"] == 2
    assert receipt["remaining_targeted_binding_count"] == 0
    assert receipt["condition_effective_during_cleanup"] is False


def test_cleanup_terminal_evidence_is_exactly_one_of_four(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeTransport()
    plan, prepare, install, readback = _readback(fake)
    authorization = subject.build_explicit_authorization(
        iam_plan=plan, raw_nonce=RAW_NONCE, operation="cleanup"
    )
    closeout = {
        "run_name": plan["run_name"],
        "execution_plan_sha256": plan["execution_plan_sha256"],
        "operator_abort": True,
        "scientific_result_claimed": False,
        "all_owned_instances_deleted": True,
        "receipt_content_sha256": "d" * 64,
    }
    monkeypatch.setattr(
        subject.cloud,
        "_validate_collection_receipt",
        lambda value: dict(value),
    )
    monkeypatch.setattr(
        subject.cloud,
        "_validate_owned_launch_failure_closeout_receipt",
        lambda value, *, execution_plan: dict(value),
        raising=False,
    )
    with pytest.raises(ValueError, match="exactly one"):
        subject.cleanup_worker_iam(
            iam_plan=plan,
            prepare_receipt=prepare,
            install_receipt=install,
            readback_receipt=readback,
            authorization=authorization,
            raw_nonce=RAW_NONCE,
            transport=fake,
            collection_receipt=_collection(plan),
            owned_launch_failure_closeout_receipt=closeout,
            execute=False,
        )

    expired_authorization = subject.build_expired_exact2_cleanup_authorization(
        iam_plan=plan,
        raw_cleanup_nonce=EXPIRED_CLEANUP_NONCE,
        issued_at_unix_seconds=EXPIRED_CLEANUP_NOT_BEFORE,
    )
    with pytest.raises(ValueError, match="exactly one"):
        subject.cleanup_worker_iam(
            iam_plan=plan,
            prepare_receipt=prepare,
            install_receipt=install,
            readback_receipt=readback,
            authorization=expired_authorization,
            raw_nonce=None,
            transport=fake,
            collection_receipt=_collection(plan),
            expired_exact2=True,
            raw_expired_cleanup_nonce=EXPIRED_CLEANUP_NONCE,
            execute=False,
            now_unix_seconds=EXPIRED_CLEANUP_NOT_BEFORE,
        )


def test_cleanup_cli_exposes_owned_launch_failure_closeout_option() -> None:
    args = subject._parser().parse_args(
        [
            "cleanup",
            "--iam-plan",
            "iam.json",
            "--prepare",
            "prepare.json",
            "--install",
            "install.json",
            "--readback",
            "readback.json",
            "--authorization",
            "cleanup-auth.json",
            "--owned-launch-failure-closeout",
            "abort-closeout.json",
            "--output",
            "cleanup.json",
        ]
    )
    assert args.owned_launch_failure_closeout == "abort-closeout.json"


def test_cleanup_cli_exposes_expired_exact2_and_removes_zero_receipt_option() -> None:
    args = subject._parser().parse_args(
        [
            "cleanup",
            "--iam-plan", "iam.json",
            "--prepare", "prepare.json",
            "--install", "install.json",
            "--readback", "readback.json",
            "--authorization", "cleanup-auth.json",
            "--expired-exact2",
            "--output", "cleanup.json",
        ]
    )
    assert args.expired_exact2 is True
    with pytest.raises(SystemExit):
        subject._parser().parse_args(
            [
                "cleanup",
                "--iam-plan", "iam.json",
                "--prepare", "prepare.json",
                "--install", "install.json",
                "--readback", "readback.json",
                "--authorization", "cleanup-auth.json",
                "--zero-created-closeout", "zero-closeout.json",
                "--output", "cleanup.json",
            ]
        )


def test_owned_launch_failure_closeout_fails_closed_without_cloud_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeTransport()
    plan, prepare, install, readback = _readback(fake)
    authorization = subject.build_explicit_authorization(
        iam_plan=plan, raw_nonce=RAW_NONCE, operation="cleanup"
    )
    monkeypatch.delattr(
        subject.cloud,
        "_validate_owned_launch_failure_closeout_receipt",
        raising=False,
    )
    with pytest.raises(RuntimeError, match="validator is unavailable"):
        subject.cleanup_worker_iam(
            iam_plan=plan,
            prepare_receipt=prepare,
            install_receipt=install,
            readback_receipt=readback,
            authorization=authorization,
            raw_nonce=RAW_NONCE,
            transport=fake,
            owned_launch_failure_closeout_receipt={
                "run_name": plan["run_name"],
                "execution_plan_sha256": plan["execution_plan_sha256"],
            },
            execute=False,
        )
    assert fake.set_count == 1


def test_real_adapter_has_fixed_env_token_and_exact_get_post_put_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "secret-token-value-that-must-never-be-returned"
    monkeypatch.setenv(subject.TOKEN_ENVIRONMENT_VARIABLE, token)
    calls: list[tuple[str, str, Mapping[str, str], bytes | None]] = []

    def requester(
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: int,
    ) -> subject.HttpResponse:
        assert timeout == subject.REQUEST_TIMEOUT_SECONDS
        calls.append((method, url, dict(headers), body))
        if "iam.googleapis.com" in url:
            role = subject.READER_ROLE if url.endswith(subject.READER_ROLE) else subject.CREATOR_ROLE
            value = _role(role)
        elif method == "POST":
            value = _project_policy()
        else:
            value = _bucket_policy()
        return subject.HttpResponse(200, json.dumps(value).encode("utf-8"), {})

    adapter = subject.GcpBucketIamTransport(requester=requester)
    adapter.get_role(role_name=subject.READER_ROLE)
    adapter.get_bucket_policy()
    adapter.get_project_policy()
    adapter.set_bucket_policy(policy=_bucket_policy())
    assert [row[0] for row in calls] == ["GET", "GET", "POST", "PUT"]
    assert calls[1][1].endswith("/iam?optionsRequestedPolicyVersion=3")
    assert calls[2][1].endswith(f"projects/{subject.PROJECT}:getIamPolicy")
    assert json.loads(calls[2][3]) == {"options": {"requestedPolicyVersion": 3}}
    assert json.loads(calls[3][3]) == _bucket_policy()
    serialized = json.dumps([row[1] for row in calls])
    assert token not in serialized


def test_real_adapter_412_is_one_shot_and_error_is_secret_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "secret-token-value-that-must-never-be-returned"
    monkeypatch.setenv(subject.TOKEN_ENVIRONMENT_VARIABLE, token)
    count = 0

    def requester(*_args: Any) -> subject.HttpResponse:
        nonlocal count
        count += 1
        return subject.HttpResponse(412, b'{"error":"body-secret"}', {})

    adapter = subject.GcpBucketIamTransport(requester=requester)
    with pytest.raises(subject.WorkerIamError) as caught:
        adapter.set_bucket_policy(policy=_bucket_policy())
    assert count == 1
    assert str(caught.value) == "worker_iam_rest_request_rejected"
    assert token not in str(caught.value)
    assert "body-secret" not in str(caught.value)


def test_write_once_is_canonical_and_immutable(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    value = _plan()
    subject.write_json_once(path, value)
    assert path.read_bytes() == subject.canonical_bytes(value)
    assert subject.read_canonical_json(path, "plan") == value
    with pytest.raises(FileExistsError):
        subject.write_json_once(path, value)


@pytest.mark.parametrize("operation", ["apply", "cleanup"])
def test_cli_authorize_uses_fixed_nonce_env_and_writes_no_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    plan = _plan()
    plan_path = tmp_path / "iam-plan.json"
    output = tmp_path / f"{operation}-authorization.json"
    subject.write_json_once(plan_path, plan)
    monkeypatch.setenv(subject.NONCE_ENVIRONMENT_VARIABLE, RAW_NONCE)
    assert subject.main(
        [
            "authorize",
            "--iam-plan",
            str(plan_path),
            "--operation",
            operation,
            "--output",
            str(output),
        ]
    ) == 0
    value = subject.read_canonical_json(output, "authorization")
    assert value == subject.build_explicit_authorization(
        iam_plan=plan, raw_nonce=RAW_NONCE, operation=operation
    )
    assert value["operation"] == operation
    assert value["one_shot_nonce_sha256"] == NONCE_SHA
    assert RAW_NONCE.encode("utf-8") not in output.read_bytes()
    with pytest.raises(FileExistsError):
        subject.main(
            [
                "authorize",
                "--iam-plan",
                str(plan_path),
                "--operation",
                operation,
                "--output",
                str(output),
            ]
        )


def test_cli_authorize_expired_exact2_uses_fresh_independent_nonce(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan()
    plan_path = tmp_path / "iam-plan.json"
    output = tmp_path / "expired-exact2-authorization.json"
    subject.write_json_once(plan_path, plan)
    monkeypatch.delenv(subject.NONCE_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.setenv(
        subject.EXPIRED_CLEANUP_NONCE_ENVIRONMENT_VARIABLE,
        EXPIRED_CLEANUP_NONCE,
    )
    monkeypatch.setattr(subject.time, "time", lambda: EXPIRED_CLEANUP_NOT_BEFORE)
    assert subject.main(
        [
            "authorize",
            "--iam-plan", str(plan_path),
            "--operation", "cleanup-expired-exact2",
            "--output", str(output),
        ]
    ) == 0
    value = subject.read_canonical_json(output, "expired authorization")
    assert value == subject.build_expired_exact2_cleanup_authorization(
        iam_plan=plan,
        raw_cleanup_nonce=EXPIRED_CLEANUP_NONCE,
        issued_at_unix_seconds=EXPIRED_CLEANUP_NOT_BEFORE,
    )
    assert value["zero_created_claimed"] is False
    assert RAW_NONCE.encode("utf-8") not in output.read_bytes()
    assert EXPIRED_CLEANUP_NONCE.encode("utf-8") not in output.read_bytes()

    reused_plan = subject.build_iam_plan(
        execution_plan=_execution_plan(),
        one_shot_nonce_sha256=hashlib.sha256(
            EXPIRED_CLEANUP_NONCE.encode("utf-8")
        ).hexdigest(),
        issued_at_unix_seconds=ISSUED,
    )
    reused_plan_path = tmp_path / "reused-launch-nonce-plan.json"
    subject.write_json_once(reused_plan_path, reused_plan)
    with pytest.raises(ValueError, match="reused the launch nonce"):
        subject.main(
            [
                "authorize",
                "--iam-plan", str(reused_plan_path),
                "--operation", "cleanup-expired-exact2",
                "--output", str(tmp_path / "must-not-exist.json"),
            ]
        )


def test_cli_authorize_requires_fixed_nonce_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = tmp_path / "iam-plan.json"
    subject.write_json_once(plan_path, _plan())
    monkeypatch.delenv(subject.NONCE_ENVIRONMENT_VARIABLE, raising=False)
    with pytest.raises(ValueError, match="nonce environment"):
        subject.main(
            [
                "authorize",
                "--iam-plan",
                str(plan_path),
                "--operation",
                "apply",
                "--output",
                str(tmp_path / "authorization.json"),
            ]
        )
