from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Mapping
from unittest.mock import patch

import pytest

from ofc_regular import hu_m31_t3_step6d_full100_wave_cloud_v2 as cloud_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_gcp_adapter_v2 as gcp_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_launch_bundle_v2 as subject
from ofc_regular import hu_m31_t3_step6d_full100_wave_package_v2 as package_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_plan_v2 as wave_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_runtime_preflight_v2 as runtime_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_runtime_gcp_adapter_v2 as runtime_gcp_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_worker_identity_v2 as identity_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_worker_iam_v2 as iam_v2
from ofc_regular.hu_m31_t3_step6d_performance_development_v2_cloud_execution_v1 import HttpResponse


RUN_NAME = "regular-hu-m31-c02-f100wv2-20260722-099"
SALT = "1234567890abcdef1234567890abcdef"
PACKAGE_SHA = "2" * 64
IMAGE_DIGEST = (
    "sha256:9dd85299f559ea3b143b1a764a9c69e0e535672036c2b45bf1cff25b88da3c0d"
)
EXPECTED_STARTUP_SHA = (
    "204a12c40b56dda643b7228e1687818a618bf1cb4a1f50359187b113c82a7d87"
)
ZONE = "asia-northeast1-b"
NONCE = "12345678-1234-4234-9234-1234567890ab"
NOW = "2026-07-22T01:00:05Z"


class FakeAtomicBackend:
    def __init__(self) -> None:
        self.objects: dict[str, dict[str, Any]] = {}

    def put_if_absent(
        self, *, object_name: str, payload: bytes
    ) -> Mapping[str, Any]:
        if object_name in self.objects:
            raise FileExistsError(object_name)
        row = {
            "object_name": object_name,
            "generation": "1001",
            "etag": "etag-create-1001",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "payload": payload,
        }
        self.objects[object_name] = row
        return {
            "created": True,
            **{key: value for key, value in row.items() if key != "payload"},
        }

    def get_object(self, *, object_name: str) -> Mapping[str, Any]:
        return deepcopy(self.objects[object_name])


class FakeIamBackend:
    def __init__(self) -> None:
        self.policy: dict[str, Any] = {
            "kind": "storage#policy",
            "resourceId": f"projects/_/buckets/{iam_v2.BUCKET}",
            "version": 3,
            "etag": "BwWInitialEtag==",
            "bindings": [],
            "auditConfigs": [],
        }
        self.set_count = 0

    def get_bucket_policy(self) -> Mapping[str, Any]:
        return deepcopy(self.policy)

    def set_bucket_policy(self, *, policy: Mapping[str, Any]) -> Mapping[str, Any]:
        supplied = deepcopy(dict(policy))
        if supplied["etag"] != self.policy["etag"]:
            raise iam_v2.WorkerIamCasError("stale fake ETag")
        self.set_count += 1
        supplied["etag"] = f"BwWAfterSet{self.set_count}=="
        self.policy = supplied
        return deepcopy(self.policy)


def _entry(
    relative_path: str,
    kind: str,
    sha256: str,
    byte_count: int,
    *,
    job_id: str | None = None,
    source_role: str | None = None,
    shard_index: int | None = None,
    work_hand_indices: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "relative_path": relative_path,
        "object_name": "pending",
        "kind": kind,
        "sha256": sha256,
        "bytes": byte_count,
        "job_id": job_id,
        "source_role": source_role,
        "shard_index": shard_index,
        "work_hand_indices": work_hand_indices,
    }


def _outer_manifest(plan: dict[str, Any]) -> dict[str, Any]:
    source_bytes = 25
    wave_bytes = wave_v2.canonical_bytes(plan)
    entries = [
        _entry(
            package_v2.SOURCE_PATH,
            "scientific_source_archive",
            PACKAGE_SHA,
            source_bytes,
        ),
        _entry(
            package_v2.SCIENTIFIC_MANIFEST_PATH,
            "scientific_source_manifest",
            "4" * 64,
            11,
        ),
        _entry(
            package_v2.WHEELHOUSE_PATH,
            "offline_wheelhouse_archive",
            "5" * 64,
            12,
        ),
        _entry(
            package_v2.WHEELHOUSE_MANIFEST_PATH,
            "offline_wheelhouse_manifest",
            "6" * 64,
            13,
        ),
        _entry(
            package_v2.STARTUP_PATH,
            "wave_v2_startup",
            EXPECTED_STARTUP_SHA,
            14,
        ),
        _entry(
            package_v2.WAVE_PLAN_PATH,
            "wave_plan",
            hashlib.sha256(wave_bytes).hexdigest(),
            len(wave_bytes),
        ),
    ]
    frozen_by_job = {
        row["job_id"]: row for row in plan["full100_plan"]["jobs"]
    }
    for index, job_id in enumerate(plan["coverage"]["job_ids"]):
        frozen = frozen_by_job[job_id]
        entries.append(
            _entry(
                package_v2.JOB_PATH_TEMPLATE.format(job_id=frozen["job_id"]),
                "job_manifest",
                frozen["shard_manifest_sha256"],
                100 + index,
                job_id=frozen["job_id"],
                source_role=frozen["source_role"],
                shard_index=frozen["shard_index"],
                work_hand_indices=list(frozen["work_hand_indices"]),
            )
        )
    records = [
        {key: value for key, value in row.items() if key != "object_name"}
        for row in entries
    ]
    content_sha = package_v2.canonical_sha256(records)
    prefix = f"{package_v2.CONTENT_PREFIX_ROOT}/{content_sha}"
    for row in entries:
        row["object_name"] = f"{prefix}/{row['relative_path']}"
    value: dict[str, Any] = {
        "schema": package_v2.OUTER_MANIFEST_SCHEMA,
        "status": "immutable_outer_content_ready_cloud_not_authorized",
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_plan_sha256": plan["schedule_sha256"],
        "full100_plan_sha256": plan["full100_plan_sha256"],
        "run_contract_digest": plan["run_contract_digest"],
        "scientific_lineage": {
            "legacy_run_name": "regular-hu-m31-c02-full100-dev-20260717-002",
            "legacy_manifest_sha256": "4" * 64,
            "legacy_ready_sha256": "7" * 64,
            "legacy_manifest_bytes": 11,
            "source_sha256": PACKAGE_SHA,
            "source_bytes": source_bytes,
            "startup_ignored": True,
            "launch_authorization_ignored": True,
            "launcher_reuse_forbidden": True,
        },
        "runtime_binding": deepcopy(plan["runtime_binding"]),
        "entries": entries,
        "entry_count": len(entries),
        "wheelhouse_binding": {
            "archive_sha256": "5" * 64,
            "archive_bytes": 12,
            "manifest_sha256": "6" * 64,
            "manifest_bytes": 13,
            "requirements_sha256": "8" * 64,
            "offline_install_only": True,
        },
        "expected_startup_sha256": EXPECTED_STARTUP_SHA,
        "content_payload_sha256": content_sha,
        "content_prefix": prefix,
        "content_create_only": True,
        "legacy_launcher_authorized": False,
        "cloud_launch_authorized": False,
        "cloud_started": False,
        "performance_lock_authorized": False,
        "quality_pilot_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
    }
    value["manifest_sha256"] = package_v2.canonical_sha256(value)
    return package_v2.validate_outer_manifest(
        plan,
        value,
        expected_startup_sha256=EXPECTED_STARTUP_SHA,
    )


def _quota_rows() -> list[dict[str, Any]]:
    return [
        {
            "metric": metric,
            "limit_vcpus": 256,
            "usage_vcpus": 128,
            "available_vcpus": 128,
            "readback_complete": True,
        }
        for metric in cloud_v2.QUOTA_METRICS
    ]


def _cloud_chain(
    plan: dict[str, Any],
    ledger: dict[str, Any],
    resume: dict[str, Any],
    content_sha: str,
) -> dict[str, dict[str, Any]]:
    quota = cloud_v2.build_live_quota_headroom_receipt(
        plan,
        ledger,
        resume,
        immutable_content_sha256=content_sha,
        project_id=iam_v2.PROJECT,
        zone=ZONE,
        observed_at_utc="2026-07-22T01:00:01Z",
        expires_at_utc="2026-07-22T01:05:01Z",
        quota_metrics=_quota_rows(),
        readback_source="fake_backend_fixture",
    )
    claim = cloud_v2.create_persistent_atomic_launch_claim(
        plan,
        ledger,
        resume,
        immutable_content_sha256=content_sha,
        project_id=iam_v2.PROJECT,
        zone=ZONE,
        claim_nonce=NONCE,
        claimed_at_utc="2026-07-22T01:00:02Z",
        backend=FakeAtomicBackend(),
    )
    mapping = cloud_v2.build_planned_launch_mapping_receipt(
        plan,
        ledger,
        resume,
        immutable_content_sha256=content_sha,
        project_id=iam_v2.PROJECT,
        zone=ZONE,
        observed_at_utc="2026-07-22T01:00:01Z",
        expires_at_utc="2026-07-22T01:05:01Z",
        instance_absence_readbacks=[
            {
                "instance_id": row["instance_id"],
                "instance_absent": True,
                "boot_disk_absent": True,
                "readback_complete": True,
            }
            for row in resume["selected_attempts"]
        ],
        readback_source="fake_backend_fixture",
    )
    authorization = cloud_v2.build_prelaunch_authorization(
        plan,
        ledger,
        resume,
        immutable_content_sha256=content_sha,
        quota_receipt=quota,
        persistent_claim_receipt=claim,
        planned_mapping_receipt=mapping,
        raw_claim_nonce=NONCE,
        authorized_at_utc="2026-07-22T01:00:04Z",
        expires_at_utc="2026-07-22T01:03:04Z",
        explicit_launch_authorized=True,
    )
    return {
        "quota": quota,
        "claim": claim,
        "mapping": mapping,
        "authorization": authorization,
    }


def _identity_chain(
    plan: dict[str, Any], ledger: dict[str, Any], resume: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    identity = identity_v2.build_worker_identity_plan(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        wave_index=resume["resume_wave_index"],
    )
    observed_accounts = []
    for index, account in enumerate(identity_v2.fixed_pool_accounts()):
        observed_accounts.append(
            {
                "account_id": account["account_id"],
                "email": account["email"],
                "name": account["name"],
                "project_id": identity_v2.PROJECT,
                "unique_id": str(100_000_000 + index),
                "disabled": False,
                "exists": True,
            }
        )
    inventory = identity_v2.build_inventory_receipt(
        identity_plan=identity,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        observed_accounts=observed_accounts,
        observed_at_utc="2026-07-22T01:00:01Z",
        provider_source="fake-provider-test-v1",
    )
    actas = identity_v2.build_act_as_receipt(
        identity_plan=identity,
        inventory_receipt=inventory,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        test_iam_permissions_observations=[
            {
                "account_id": worker["account_id"],
                "email": worker["service_account_email"],
                "name": worker["service_account_name"],
                "http_method": "POST",
                "requested_permissions": [identity_v2.ACT_AS_PERMISSION],
                "granted_permissions": [identity_v2.ACT_AS_PERMISSION],
            }
            for worker in identity["selected_workers"]
        ],
        tested_at_utc="2026-07-22T01:00:03Z",
        provider_source="fake-provider-test-v1",
    )
    project_scan = identity_v2.build_project_iam_scan_receipt(
        identity_plan=identity,
        inventory_receipt=inventory,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        project_iam_policy={
            "version": 3,
            "etag": "project-policy-etag",
            "bindings": [],
            "auditConfigs": [],
        },
        observed_at_utc="2026-07-22T01:00:01Z",
        provider_source="fake-provider-test-v1",
    )
    return {
        "identity_plan": identity,
        "identity_inventory": inventory,
        "identity_actas": actas,
        "project_scan": project_scan,
    }


def _gcp_read_chain(
    *,
    plan: dict[str, Any],
    ledger: dict[str, Any],
    resume: dict[str, Any],
    outer: dict[str, Any],
    quota: dict[str, Any],
    mapping: dict[str, Any],
    iam_plan: dict[str, Any],
    identity_inventory: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    by_email = {
        row["email"]: row for row in identity_inventory["rows"]
    }
    accounts = []
    for worker in iam_plan["workers"]:
        inventory = by_email[worker["service_account"]]
        accounts.append(
            {
                "job_id": worker["job_id"],
                "source_role": worker["source_role"],
                "service_account": worker["service_account"],
                "resource_name": (
                    f"projects/{iam_v2.PROJECT}/serviceAccounts/"
                    f"{inventory['unique_id']}"
                ),
                "unique_id": inventory["unique_id"],
                "disabled": False,
                "exists": True,
            }
        )
    custom_roles = [
        {
            "role_name": role,
            "included_permissions": list(permissions),
            "stage": "GA",
            "deleted": False,
            "etag_sha256": hashlib.sha256(
                f"etag-{role}".encode("utf-8")
            ).hexdigest(),
        }
        for role, permissions in gcp_v2.CUSTOM_ROLE_EXPECTATIONS
    ]
    result_rows = [
        {
            "job_id": selected["job_id"],
            "source_role": selected["source_role"],
            "attempt_id": selected["attempt_id"],
            "artifact_prefix": selected["artifact_prefix"],
            "acceptance_path": plan["artifact_contract"][
                "job_acceptance_path_template"
            ].format(job_id=selected["job_id"]),
            "list_page_count": 1,
            "listed_object_count": 0,
            "artifact_prefix_absent": True,
            "acceptance_object_absent": True,
        }
        for selected in resume["selected_attempts"]
    ]
    result_core = {
        "schema": gcp_v2.SELECTED_RESULT_PREFLIGHT_RECEIPT_SCHEMA,
        "status": "selected_attempt_prefixes_and_acceptance_objects_absent",
        "bucket": iam_v2.BUCKET,
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_plan_sha256": plan["schedule_sha256"],
        "attempt_ledger_sha256": ledger["ledger_sha256"],
        "resume_plan_sha256": resume["resume_sha256"],
        "wave_index": resume["resume_wave_index"],
        "selected_attempts_sha256": gcp_v2.canonical_sha256(
            resume["selected_attempts"]
        ),
        "rows": result_rows,
        "selected_attempt_count": len(result_rows),
        "http_get_count": 2 * len(result_rows),
        "pagination_observed": False,
        "all_selected_artifact_prefixes_absent": True,
        "all_selected_acceptance_objects_absent": True,
        "read_only": True,
        "cloud_mutation_performed": False,
        "observed_at_utc": "2026-07-22T01:00:01Z",
        "current_profile_changed": False,
    }
    result_preflight = {
        **result_core,
        "receipt_sha256": gcp_v2.canonical_sha256(result_core),
    }
    read_core = {
        "schema": gcp_v2.READ_RECEIPT_SCHEMA,
        "status": "fresh_phase_a_readback_complete",
        "project": iam_v2.PROJECT,
        "region": gcp_v2.REGION,
        "zone": ZONE,
        "bucket": iam_v2.BUCKET,
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_plan_sha256": plan["schedule_sha256"],
        "attempt_ledger_sha256": ledger["ledger_sha256"],
        "resume_plan_sha256": resume["resume_sha256"],
        "wave_index": resume["resume_wave_index"],
        "immutable_content_prefix": outer["content_prefix"],
        "content_payload_sha256": outer["content_payload_sha256"],
        "outer_manifest_sha256": outer["manifest_sha256"],
        "iam_plan_sha256": iam_plan["plan_sha256"],
        "provider_quota_metrics": {
            "c4": gcp_v2.C4_QUOTA_METRIC,
            "c4_quota_id": gcp_v2.C4_QUOTA_ID,
            "c4_project_number": "123456789",
            "c4_usage_inventory_sha256": "9" * 64,
            "spot": "PREEMPTIBLE_CPUS",
            "global": gcp_v2.GLOBAL_QUOTA_METRIC,
            "global_quota_id": gcp_v2.GLOBAL_QUOTA_ID,
            "global_usage_inventory_sha256": "8" * 64,
        },
        "quota_receipt": quota,
        "planned_mapping_receipt": mapping,
        "custom_roles": custom_roles,
        "custom_roles_sha256": gcp_v2.canonical_sha256(custom_roles),
        "custom_role_count": len(custom_roles),
        "selected_result_preflight_receipt": result_preflight,
        "observed_at_utc": "2026-07-22T01:00:01Z",
        "service_accounts": accounts,
        "selected_vm_count": len(accounts),
        "http_get_count": (
            7 + 2 * len(accounts) + len(custom_roles) + len(accounts)
            + result_preflight["http_get_count"]
        ),
        "all_selected_instances_absent": True,
        "all_selected_boot_disks_absent": True,
        "all_service_accounts_exist": True,
        "all_custom_roles_exact_ga_not_deleted": True,
        "pagination_observed": False,
        "read_only": True,
        "cloud_mutation_performed": False,
        "current_profile_changed": False,
    }
    gcp_read = {
        **read_core,
        "receipt_sha256": gcp_v2.canonical_sha256(read_core),
    }
    gcp_read = gcp_v2.validate_read_receipt(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        immutable_content_prefix=outer["content_prefix"],
        content_payload_sha256=outer["content_payload_sha256"],
        outer_manifest_sha256=outer["manifest_sha256"],
        iam_plan=iam_plan,
        value=gcp_read,
    )
    actas_rows = [
        {
            "job_id": account["job_id"],
            "source_role": account["source_role"],
            "service_account": account["service_account"],
            "resource_name": account["resource_name"],
            "unique_id": account["unique_id"],
            "requested_permissions": [gcp_v2.ACT_AS_PERMISSION],
            "granted_permissions": [gcp_v2.ACT_AS_PERMISSION],
            "act_as_granted": True,
            "test_complete": True,
        }
        for account in accounts
    ]
    actas_core = {
        "schema": gcp_v2.SERVICE_ACCOUNT_ACTAS_RECEIPT_SCHEMA,
        "status": "all_selected_worker_act_as_and_provider_permissions_granted",
        "project": iam_v2.PROJECT,
        "region": gcp_v2.REGION,
        "zone": ZONE,
        "bucket": iam_v2.BUCKET,
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_plan_sha256": plan["schedule_sha256"],
        "attempt_ledger_sha256": ledger["ledger_sha256"],
        "resume_plan_sha256": resume["resume_sha256"],
        "wave_index": resume["resume_wave_index"],
        "immutable_content_prefix": outer["content_prefix"],
        "content_payload_sha256": outer["content_payload_sha256"],
        "outer_manifest_sha256": outer["manifest_sha256"],
        "iam_plan_sha256": iam_plan["plan_sha256"],
        "gcp_read_receipt_sha256": gcp_read["receipt_sha256"],
        "checked_at_utc": "2026-07-22T01:00:03Z",
        "expires_at_utc": "2026-07-22T01:05:01Z",
        "permission": gcp_v2.ACT_AS_PERMISSION,
        "rows": actas_rows,
        "selected_worker_count": len(accounts),
        "http_post_count": len(accounts) + 1,
        "all_selected_workers_act_as_granted": True,
        "provider_project_permissions_requested": list(
            gcp_v2.PROVIDER_PROJECT_PERMISSIONS
        ),
        "provider_project_permissions_granted": list(
            gcp_v2.PROVIDER_PROJECT_PERMISSIONS
        ),
        "provider_project_permission_count": len(
            gcp_v2.PROVIDER_PROJECT_PERMISSIONS
        ),
        "all_provider_project_permissions_granted": True,
        "permission_test_only": True,
        "cloud_mutation_performed": False,
        "current_profile_changed": False,
    }
    provider_actas = {
        **actas_core,
        "receipt_sha256": gcp_v2.canonical_sha256(actas_core),
    }
    provider_actas = gcp_v2.validate_service_account_actas_receipt(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        immutable_content_prefix=outer["content_prefix"],
        content_payload_sha256=outer["content_payload_sha256"],
        outer_manifest_sha256=outer["manifest_sha256"],
        iam_plan=iam_plan,
        gcp_read_receipt=gcp_read,
        value=provider_actas,
        now_utc=NOW,
    )
    return {"gcp_read": gcp_read, "provider_actas": provider_actas}


def _runtime_gcp_read(plan: dict[str, Any]) -> dict[str, Any]:
    image_name = "debian-12-bookworm-v20260721"
    observations = {
        "image_family": {
            "kind": "compute#image",
            "id": "9021508813201755912",
            "name": image_name,
            "family": "debian-12",
            "status": "READY",
            "architecture": "X86_64",
            "guestOsFeatures": [
                {"type": "GVNIC"},
                {"type": "SEV_CAPABLE"},
                {"type": "SEV_LIVE_MIGRATABLE_V2"},
                {"type": "UEFI_COMPATIBLE"},
                {"type": "VIRTIO_SCSI_MULTIQUEUE"},
            ],
            "storageLocations": ["asia-northeast1"],
            "selfLink": (
                "https://www.googleapis.com/compute/v1/projects/debian-cloud/"
                f"global/images/{image_name}"
            ),
        },
        "machine_type": {
            "kind": "compute#machineType",
            "id": "16001",
            "name": runtime_v2.MACHINE_TYPE,
            "guestCpus": 16,
            "memoryMb": 61_440,
            "architecture": "X86_64",
            "zone": runtime_v2.ZONE_SELF_LINK,
            "selfLink": runtime_v2.MACHINE_TYPE_SELF_LINK,
        },
        "network": {
            "kind": "compute#network",
            "id": "10001",
            "name": "default",
            "selfLink": runtime_v2.NETWORK_SELF_LINK,
            "autoCreateSubnetworks": True,
        },
        "subnetwork": {
            "kind": "compute#subnetwork",
            "id": "10002",
            "name": "default",
            "region": runtime_v2.REGION_SELF_LINK,
            "network": runtime_v2.NETWORK_SELF_LINK,
            "selfLink": runtime_v2.SUBNETWORK_SELF_LINK,
            "stackType": "IPV4_ONLY",
        },
        "router_nat": {
            "kind": "compute#router",
            "id": "10003",
            "name": runtime_v2.NAT_ROUTER_NAME,
            "region": runtime_v2.REGION_SELF_LINK,
            "network": runtime_v2.NETWORK_SELF_LINK,
            "selfLink": runtime_v2.ROUTER_SELF_LINK,
            "nats": [
                {
                    "name": runtime_v2.NAT_NAME,
                    "natIpAllocateOption": "AUTO_ONLY",
                    "sourceSubnetworkIpRangesToNat": (
                        "ALL_SUBNETWORKS_ALL_IP_RANGES"
                    ),
                }
            ],
        },
        "bucket": {
            "kind": "storage#bucket",
            "id": runtime_v2.BUCKET,
            "name": runtime_v2.BUCKET,
            "projectNumber": runtime_v2.PROJECT_NUMBER,
            "location": runtime_v2.BUCKET_LOCATION,
            "locationType": "region",
            "storageClass": "STANDARD",
            "iamConfiguration": {
                "uniformBucketLevelAccess": {"enabled": True}
            },
        },
    }
    by_url = {url: resource for resource, url in runtime_gcp_v2.ENDPOINTS}

    def requester(method, url, headers, body, timeout):
        del headers, timeout
        assert method == "GET" and body is None
        return HttpResponse(
            status=200,
            body=json.dumps(
                observations[by_url[url]], sort_keys=True, separators=(",", ":")
            ).encode("ascii"),
            headers={},
        )

    with patch.dict(
        "os.environ",
        {runtime_gcp_v2.TOKEN_ENV: "test-token-1234567890-abcdef"},
    ):
        return runtime_gcp_v2.RuntimeGcpReadAdapterV2(
            wave_plan=plan, requester=requester
        ).read(
            observed_at_utc="2026-07-22T01:00:00Z",
            current_utc="2026-07-22T01:00:04Z",
        )


def _single_job_context() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any]
]:
    """Derive a real one-job resume without weakening the normal 8+8+4 plan."""

    plan = wave_v2.build_wave_plan(
        run_name=RUN_NAME,
        identity_salt=SALT,
        package_sha256=PACKAGE_SHA,
        image_digest=IMAGE_DIGEST,
    )
    baseline = wave_v2.build_observed_transition(
        plan,
        project_id=iam_v2.PROJECT,
        zone=ZONE,
        observed_at_utc="2026-07-22T00:59:58Z",
        previous_transition_digest=None,
        attempt_history=wave_v2.empty_attempt_history(plan),
    )
    histories = {
        row["job_id"]: deepcopy(row) for row in baseline["attempt_history"]
    }
    frozen_jobs = {
        row["job_id"]: row for row in plan["full100_plan"]["jobs"]
    }
    instance_ids: dict[str, dict[str, str]] = {}
    for pair in plan["waves"][0]["candidate_reference_pairs"]:
        instance_ids[pair["candidate_job_id"]] = pair[
            "candidate_attempt_instance_ids"
        ]
        instance_ids[pair["reference_job_id"]] = pair[
            "reference_attempt_instance_ids"
        ]
    keep = "candidate-shard-00"
    done: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    for ordinal, job_id in enumerate(plan["waves"][0]["job_ids"]):
        if job_id == keep:
            continue
        role = histories[job_id]["source_role"]
        histories[job_id]["attempts"].append(
            {
                "attempt_id": "a00",
                "instance_id": instance_ids[job_id]["a00"],
                "launch_receipt_sha256": hashlib.sha256(
                    f"single-canary-launch-{job_id}".encode("utf-8")
                ).hexdigest(),
                "terminal_status": "accepted",
            }
        )
        roots = [
            {
                "hand_index": hand,
                "sha256": hashlib.sha256(
                    f"single-canary-root-{hand:03d}".encode("utf-8")
                ).hexdigest(),
            }
            for hand in frozen_jobs[job_id]["work_hand_indices"]
        ]
        root_digest = wave_v2.canonical_sha256(roots)
        generation = 10_000 + ordinal * 10
        done_sha = hashlib.sha256(
            f"single-canary-done-{job_id}".encode("utf-8")
        ).hexdigest()
        attempt_prefix = plan["artifact_contract"][
            "attempt_path_template"
        ].format(job_id=job_id, attempt_id="a00")
        done.append(
            {
                "job_id": job_id,
                "source_role": role,
                "attempt_id": "a00",
                "path": f"{attempt_prefix}/DONE.json",
                "generation": generation,
                "bytes": 200 + ordinal,
                "sha256": done_sha,
                "done_identity_sha256": wave_v2.expected_done_identity_sha256(
                    plan,
                    job_id=job_id,
                    attempt_id="a00",
                    root_digest=root_digest,
                ),
                "package_sha256": plan["runtime_binding"]["package_sha256"],
                "image_digest": plan["runtime_binding"]["image_digest"],
                "binary_sha256": plan["runtime_binding"][
                    "binary_sha256_by_role"
                ][role],
                "allocation_digest": plan["runtime_binding"][
                    "allocation_digest"
                ],
                "root_digest": root_digest,
            }
        )
        accepted.append(
            {
                "job_id": job_id,
                "source_role": role,
                "attempt_id": "a00",
                "path": plan["artifact_contract"][
                    "job_acceptance_path_template"
                ].format(job_id=job_id),
                "generation": generation + 1,
                "bytes": 100 + ordinal,
                "sha256": hashlib.sha256(
                    f"single-canary-accepted-{job_id}".encode("utf-8")
                ).hexdigest(),
                "done_generation": generation,
                "done_sha256": done_sha,
                "create_only": True,
            }
        )
    observed = wave_v2.build_observed_transition(
        plan,
        project_id=iam_v2.PROJECT,
        zone=ZONE,
        observed_at_utc="2026-07-22T00:59:59Z",
        previous_transition_digest=baseline["transition_digest"],
        attempt_history=[
            histories[job_id] for job_id in plan["coverage"]["job_ids"]
        ],
        done_objects=done,
        acceptance_records=accepted,
    )
    ledger = wave_v2.build_attempt_ledger(
        plan,
        transitions=[baseline, observed],
        consumed_transition_digests=[baseline["transition_digest"]],
    )
    resume = wave_v2.build_resume_plan(plan, attempt_ledger=ledger)
    assert resume["selected_attempts"] == [
        {
            "job_id": keep,
            "source_role": "candidate",
            "attempt_id": "a00",
            "instance_id": instance_ids[keep]["a00"],
            "artifact_prefix": plan["artifact_contract"][
                "attempt_path_template"
            ].format(job_id=keep, attempt_id="a00"),
        }
    ]
    return plan, ledger, resume


def _evidence_for_context(
    plan: dict[str, Any],
    ledger: dict[str, Any],
    resume: dict[str, Any],
) -> dict[str, Any]:
    runtime_gcp_read = _runtime_gcp_read(plan)
    runtime_preflight = runtime_gcp_read["runtime_preflight_receipt"]
    outer = _outer_manifest(plan)
    cloud = _cloud_chain(plan, ledger, resume, outer["content_payload_sha256"])
    issued_at = int(
        datetime(2026, 7, 22, 1, 0, 0, tzinfo=timezone.utc).timestamp()
    )
    iam_plan = iam_v2.build_worker_iam_plan(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        wave_index=resume["resume_wave_index"],
        immutable_content_prefix=outer["content_prefix"],
        content_payload_sha256=outer["content_payload_sha256"],
        outer_manifest_sha256=outer["manifest_sha256"],
        issued_at_unix_seconds=issued_at,
    )
    iam_backend = FakeIamBackend()
    content_kwargs = {
        "immutable_content_prefix": outer["content_prefix"],
        "content_payload_sha256": outer["content_payload_sha256"],
        "outer_manifest_sha256": outer["manifest_sha256"],
    }
    prepare = iam_v2.prepare_worker_iam(
        iam_plan=iam_plan,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        **content_kwargs,
        backend=iam_backend,
    )
    install = iam_v2.install_worker_iam(
        iam_plan=iam_plan,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        **content_kwargs,
        prepare_receipt=prepare,
        backend=iam_backend,
    )
    readback = iam_v2.readback_worker_iam(
        iam_plan=iam_plan,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        **content_kwargs,
        prepare_receipt=prepare,
        install_receipt=install,
        backend=iam_backend,
    )
    identity = _identity_chain(plan, ledger, resume)
    gcp = _gcp_read_chain(
        plan=plan,
        ledger=ledger,
        resume=resume,
        outer=outer,
        quota=cloud["quota"],
        mapping=cloud["mapping"],
        iam_plan=iam_plan,
        identity_inventory=identity["identity_inventory"],
    )
    return {
        "plan": plan,
        "ledger": ledger,
        "resume": resume,
        "runtime_preflight": runtime_preflight,
        "runtime_gcp_read": runtime_gcp_read,
        "outer": outer,
        **cloud,
        "iam_plan": iam_plan,
        "prepare": prepare,
        "install": install,
        "readback": readback,
        **identity,
        **gcp,
    }


@pytest.fixture(scope="module")
def evidence() -> dict[str, Any]:
    plan = wave_v2.build_wave_plan(
        run_name=RUN_NAME,
        identity_salt=SALT,
        package_sha256=PACKAGE_SHA,
        image_digest=IMAGE_DIGEST,
    )
    transition = wave_v2.build_observed_transition(
        plan,
        project_id=iam_v2.PROJECT,
        zone=ZONE,
        observed_at_utc="2026-07-22T01:00:00Z",
        previous_transition_digest=None,
        attempt_history=wave_v2.empty_attempt_history(plan),
    )
    ledger = wave_v2.build_attempt_ledger(plan, transitions=[transition])
    resume = wave_v2.build_resume_plan(plan, attempt_ledger=ledger)
    return _evidence_for_context(plan, ledger, resume)


def _evidence_with_iam_remaining_seconds(
    evidence: dict[str, Any], remaining_seconds: int
) -> dict[str, Any]:
    now_unix_seconds = int(
        datetime.fromisoformat(NOW.removesuffix("Z") + "+00:00").timestamp()
    )
    issued_at_unix_seconds = (
        now_unix_seconds
        + remaining_seconds
        - iam_v2.CONDITION_LIFETIME_SECONDS
    )
    iam_plan = iam_v2.build_worker_iam_plan(
        wave_plan=evidence["plan"],
        attempt_ledger=evidence["ledger"],
        resume_plan=evidence["resume"],
        wave_index=evidence["resume"]["resume_wave_index"],
        immutable_content_prefix=evidence["outer"]["content_prefix"],
        content_payload_sha256=evidence["outer"]["content_payload_sha256"],
        outer_manifest_sha256=evidence["outer"]["manifest_sha256"],
        issued_at_unix_seconds=issued_at_unix_seconds,
    )
    iam_backend = FakeIamBackend()
    content_kwargs = {
        "immutable_content_prefix": evidence["outer"]["content_prefix"],
        "content_payload_sha256": evidence["outer"]["content_payload_sha256"],
        "outer_manifest_sha256": evidence["outer"]["manifest_sha256"],
    }
    prepare = iam_v2.prepare_worker_iam(
        iam_plan=iam_plan,
        wave_plan=evidence["plan"],
        attempt_ledger=evidence["ledger"],
        resume_plan=evidence["resume"],
        **content_kwargs,
        backend=iam_backend,
    )
    install = iam_v2.install_worker_iam(
        iam_plan=iam_plan,
        wave_plan=evidence["plan"],
        attempt_ledger=evidence["ledger"],
        resume_plan=evidence["resume"],
        **content_kwargs,
        prepare_receipt=prepare,
        backend=iam_backend,
    )
    readback = iam_v2.readback_worker_iam(
        iam_plan=iam_plan,
        wave_plan=evidence["plan"],
        attempt_ledger=evidence["ledger"],
        resume_plan=evidence["resume"],
        **content_kwargs,
        prepare_receipt=prepare,
        install_receipt=install,
        backend=iam_backend,
    )
    gcp = _gcp_read_chain(
        plan=evidence["plan"],
        ledger=evidence["ledger"],
        resume=evidence["resume"],
        outer=evidence["outer"],
        quota=evidence["quota"],
        mapping=evidence["mapping"],
        iam_plan=iam_plan,
        identity_inventory=evidence["identity_inventory"],
    )
    return {
        **evidence,
        "iam_plan": iam_plan,
        "prepare": prepare,
        "install": install,
        "readback": readback,
        **gcp,
    }


def _kwargs(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "wave_plan": evidence["plan"],
        "attempt_ledger": evidence["ledger"],
        "resume_plan": evidence["resume"],
        "outer_manifest": evidence["outer"],
        "expected_startup_sha256": EXPECTED_STARTUP_SHA,
        "quota_receipt": evidence["quota"],
        "persistent_claim_receipt": evidence["claim"],
        "planned_mapping_receipt": evidence["mapping"],
        "prelaunch_authorization": evidence["authorization"],
        "raw_claim_nonce": NONCE,
        "current_time_utc": NOW,
        "runtime_preflight_receipt": evidence["runtime_preflight"],
        "runtime_gcp_read_receipt": evidence["runtime_gcp_read"],
        "gcp_read_receipt": evidence["gcp_read"],
        "service_account_actas_receipt": evidence["provider_actas"],
        "worker_identity_plan": evidence["identity_plan"],
        "worker_identity_inventory_receipt": evidence[
            "identity_inventory"
        ],
        "worker_identity_act_as_receipt": evidence["identity_actas"],
        "project_iam_scan_receipt": evidence["project_scan"],
        "worker_iam_plan": evidence["iam_plan"],
        "worker_iam_prepare_receipt": evidence["prepare"],
        "worker_iam_install_receipt": evidence["install"],
        "worker_iam_readback_receipt": evidence["readback"],
    }


@pytest.fixture(scope="module")
def bundle(evidence: dict[str, Any]) -> dict[str, Any]:
    return subject.build_launch_bundle(**_kwargs(evidence))


def _reseal_bundle(value: dict[str, Any]) -> None:
    value["bundle_sha256"] = subject.canonical_sha256(
        {key: item for key, item in value.items() if key != "bundle_sha256"}
    )


def _reseal_iam_plan(value: dict[str, Any]) -> None:
    value["plan_sha256"] = iam_v2.canonical_sha256(
        {key: item for key, item in value.items() if key != "plan_sha256"}
    )


def _reseal_gcp_receipt(value: dict[str, Any]) -> None:
    value["receipt_sha256"] = gcp_v2.canonical_sha256(
        {key: item for key, item in value.items() if key != "receipt_sha256"}
    )


def _reseal_identity_receipt(value: dict[str, Any]) -> None:
    value["receipt_sha256"] = identity_v2.canonical_sha256(
        {key: item for key, item in value.items() if key != "receipt_sha256"}
    )


def test_complete_chain_builds_exact_one_shot_bootstrap_inventory(
    evidence: dict[str, Any], bundle: dict[str, Any]
) -> None:
    kwargs = _kwargs(evidence)
    assert subject.validate_launch_bundle(**kwargs, value=bundle) == bundle
    selected = evidence["resume"]["selected_attempts"]
    workers = evidence["iam_plan"]["workers"]
    assert bundle["selected_vm_count"] == 8 == len(selected)
    assert len(bundle["bootstrap_inventory"]) == len(selected)
    assert bundle["cloud_create_authorized"] is True
    assert bundle["exact_selected_create_authorized"] is True
    assert bundle["one_shot"] is True
    assert bundle["worker_iam_evidence_embedded"] is True
    assert bundle["worker_iam_readback_complete"] is True
    assert bundle["runtime_preflight_complete"] is True
    assert bundle["runtime_gcp_live_source_complete"] is True
    assert bundle["runtime_gcp_read_receipt_sha256"] == evidence[
        "runtime_gcp_read"
    ]["receipt_sha256"]
    assert bundle["gcp_readback_complete"] is True
    assert bundle["gcp_custom_roles_exact_ga_not_deleted"] is True
    assert bundle["gcp_custom_roles_sha256"] == evidence["gcp_read"][
        "custom_roles_sha256"
    ]
    assert bundle["service_account_act_as_complete"] is True
    assert bundle["worker_identity_evidence_complete"] is True
    assert bundle["project_iam_zero_roles"] is True
    assert bundle["runtime_preflight_receipt_sha256"] == evidence[
        "runtime_preflight"
    ]["receipt_sha256"]
    assert bundle["gcp_read_receipt_sha256"] == evidence["gcp_read"][
        "receipt_sha256"
    ]
    assert bundle["service_account_actas_receipt_sha256"] == evidence[
        "provider_actas"
    ]["receipt_sha256"]
    assert bundle["gcp_selected_result_preflight_receipt_sha256"] == evidence[
        "gcp_read"
    ]["selected_result_preflight_receipt"]["receipt_sha256"]
    assert bundle["gcp_selected_result_preflight_absence_complete"] is True
    assert bundle["provider_project_permissions_complete"] is True
    assert bundle["worker_identity_plan_sha256"] == evidence[
        "identity_plan"
    ]["plan_sha256"]
    assert bundle["worker_identity_inventory_receipt_sha256"] == evidence[
        "identity_inventory"
    ]["receipt_sha256"]
    assert bundle["worker_identity_act_as_receipt_sha256"] == evidence[
        "identity_actas"
    ]["receipt_sha256"]
    assert bundle["project_iam_scan_receipt_sha256"] == evidence[
        "project_scan"
    ]["receipt_sha256"]
    assert bundle["cloud_started"] is False
    assert bundle["legacy_launcher_authorized"] is False
    assert bundle["current_profile_changed"] is False
    assert bundle["named_profile_added"] is False
    assert bundle["runtime_policy_activated"] is False
    for item, selected_row, worker in zip(
        bundle["bootstrap_inventory"], selected, workers, strict=True
    ):
        bootstrap = item["bootstrap"]
        assert item["job_id"] == selected_row["job_id"]
        assert item["source_role"] == selected_row["source_role"]
        assert item["instance_id"] == selected_row["instance_id"]
        assert item["service_account"] == worker["service_account"]
        assert bootstrap["bucket"] == iam_v2.BUCKET
        assert bootstrap["worker_principal"] == worker["service_account"]
        assert (
            bootstrap["prelaunch_authorization_sha256"]
            == evidence["authorization"]["authorization_sha256"]
        )


def test_valid_single_job_resume_crosses_identity_iam_and_launch_bundle_exactly_once(
) -> None:
    plan, ledger, resume = _single_job_context()
    assert len(plan["waves"][0]["job_ids"]) == 8
    assert len(resume["selected_attempts"]) == 1

    evidence = _evidence_for_context(plan, ledger, resume)
    bundle = subject.build_launch_bundle(**_kwargs(evidence))

    assert subject.validate_launch_bundle(
        **_kwargs(evidence), value=bundle
    ) == bundle
    selected = resume["selected_attempts"][0]
    worker = evidence["iam_plan"]["workers"][0]
    inventory = bundle["bootstrap_inventory"][0]
    assert evidence["identity_plan"]["selected_count"] == 1
    assert evidence["iam_plan"]["worker_count"] == 1
    assert evidence["iam_plan"]["exact_binding_count"] == 2
    assert bundle["selected_vm_count"] == 1
    assert bundle["selected_job_ids"] == ["candidate-shard-00"]
    assert bundle["selected_attempt_ids"] == ["a00"]
    assert bundle["selected_instance_ids"] == [selected["instance_id"]]
    assert bundle["selected_service_accounts"] == [worker["service_account"]]
    assert inventory["job_id"] == selected["job_id"]
    assert inventory["bootstrap"]["job_id"] == selected["job_id"]
    assert inventory["bootstrap"]["instance_name"] == selected["instance_id"]
    assert inventory["bootstrap"]["worker_principal"] == worker["service_account"]


def test_worker_and_provider_actas_accept_distinct_monotonic_observation_times(
    evidence: dict[str, Any],
) -> None:
    sequential = deepcopy(evidence)
    sequential["identity_actas"]["tested_at_utc"] = "2026-07-22T01:00:02Z"
    _reseal_identity_receipt(sequential["identity_actas"])
    assert subject.build_launch_bundle(**_kwargs(sequential))[
        "cloud_create_authorized"
    ] is True


def test_worker_actas_after_provider_actas_is_rejected(
    evidence: dict[str, Any],
) -> None:
    reversed_chain = deepcopy(evidence)
    reversed_chain["identity_actas"]["tested_at_utc"] = "2026-07-22T01:00:04Z"
    _reseal_identity_receipt(reversed_chain["identity_actas"])
    with pytest.raises(
        ValueError, match="provider and worker-identity evidence chain changed"
    ):
        subject.build_launch_bundle(**_kwargs(reversed_chain))


def test_worker_iam_exact_minimum_remaining_window_builds_and_validates(
    evidence: dict[str, Any],
) -> None:
    assert subject.STARTUP_WATCHDOG_SECONDS == 4_200
    assert subject.IAM_PROVISION_UPLOAD_MARGIN_SECONDS == 300
    assert subject.IAM_MIN_REMAINING_SECONDS == 4_500
    boundary = _evidence_with_iam_remaining_seconds(
        evidence, subject.IAM_MIN_REMAINING_SECONDS
    )
    kwargs = _kwargs(boundary)
    built = subject.build_launch_bundle(**kwargs)
    assert subject.validate_launch_bundle(**kwargs, value=built) == built


def test_worker_iam_one_second_short_never_builds_or_validates(
    evidence: dict[str, Any],
) -> None:
    remaining = subject.IAM_MIN_REMAINING_SECONDS - 1
    insufficient = _evidence_with_iam_remaining_seconds(evidence, remaining)
    kwargs = _kwargs(insufficient)
    message = "full provision, runtime, and upload window"
    with pytest.raises(PermissionError, match=message):
        subject.build_launch_bundle(**kwargs)

    # Produce the otherwise exact bundle under the former, one-second-short
    # threshold so the public validation path must independently reapply the
    # current 4,500-second contract instead of trusting embedded booleans.
    with patch.object(subject, "IAM_MIN_REMAINING_SECONDS", remaining):
        formerly_valid = subject.build_launch_bundle(**kwargs)
    with pytest.raises(PermissionError, match=message):
        subject.validate_launch_bundle(**kwargs, value=formerly_valid)


def test_custom_role_proof_is_mandatory_and_cannot_be_resealed(
    evidence: dict[str, Any], bundle: dict[str, Any]
) -> None:
    forged_read = deepcopy(evidence["gcp_read"])
    forged_read["custom_roles"][0]["included_permissions"].append(
        "storage.objects.list"
    )
    forged_read["custom_roles_sha256"] = gcp_v2.canonical_sha256(
        forged_read["custom_roles"]
    )
    _reseal_gcp_receipt(forged_read)
    with pytest.raises(ValueError, match="custom-role"):
        subject.build_launch_bundle(
            **{**_kwargs(evidence), "gcp_read_receipt": forged_read}
        )

    forged_bundle = deepcopy(bundle)
    forged_bundle["gcp_custom_roles_sha256"] = "f" * 64
    _reseal_bundle(forged_bundle)
    with pytest.raises(ValueError, match="surface evidence"):
        subject.validate_launch_bundle(
            **_kwargs(evidence), value=forged_bundle
        )


def test_selected_result_preflight_is_mandatory_and_cannot_be_resealed(
    evidence: dict[str, Any], bundle: dict[str, Any]
) -> None:
    forged_read = deepcopy(evidence["gcp_read"])
    result = forged_read["selected_result_preflight_receipt"]
    result["resume_plan_sha256"] = "f" * 64
    _reseal_gcp_receipt(result)
    _reseal_gcp_receipt(forged_read)
    with pytest.raises(ValueError, match="selected result preflight"):
        subject.build_launch_bundle(
            **{**_kwargs(evidence), "gcp_read_receipt": forged_read}
        )

    forged_bundle = deepcopy(bundle)
    forged_bundle["gcp_selected_result_preflight_receipt_sha256"] = "f" * 64
    _reseal_bundle(forged_bundle)
    with pytest.raises(ValueError, match="surface evidence"):
        subject.validate_launch_bundle(
            **_kwargs(evidence), value=forged_bundle
        )


def test_selected_result_preflight_stale_self_reseal_fails_time_chain(
    evidence: dict[str, Any],
) -> None:
    forged_read = deepcopy(evidence["gcp_read"])
    result = forged_read["selected_result_preflight_receipt"]
    result["observed_at_utc"] = "2026-07-22T00:55:00Z"
    _reseal_gcp_receipt(result)
    _reseal_gcp_receipt(forged_read)

    with pytest.raises(ValueError, match="observation-time evidence chain"):
        subject.build_launch_bundle(
            **{**_kwargs(evidence), "gcp_read_receipt": forged_read}
        )


def test_wrong_nonce_authorization_and_outer_content_fail_closed(
    evidence: dict[str, Any],
) -> None:
    kwargs = _kwargs(evidence)
    with pytest.raises(ValueError, match="nonce"):
        subject.build_launch_bundle(
            **{**kwargs, "raw_claim_nonce": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"}
        )

    wrong_auth = deepcopy(evidence["authorization"])
    wrong_auth["authorization_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="authorization"):
        subject.build_launch_bundle(
            **{**kwargs, "prelaunch_authorization": wrong_auth}
        )

    wrong_cloud = _cloud_chain(
        evidence["plan"], evidence["ledger"], evidence["resume"], "f" * 64
    )
    with pytest.raises(ValueError, match="binding"):
        subject.build_launch_bundle(
            **{
                **kwargs,
                "quota_receipt": wrong_cloud["quota"],
                "persistent_claim_receipt": wrong_cloud["claim"],
                "planned_mapping_receipt": wrong_cloud["mapping"],
                "prelaunch_authorization": wrong_cloud["authorization"],
            }
        )


def test_duplicate_service_account_is_rejected_before_bundle_authorization(
    evidence: dict[str, Any],
) -> None:
    forged = deepcopy(evidence["iam_plan"])
    forged["workers"][1]["service_account"] = forged["workers"][0][
        "service_account"
    ]
    forged["workers"][1]["principal"] = forged["workers"][0]["principal"]
    _reseal_iam_plan(forged)
    with pytest.raises(ValueError, match="duplicated"):
        subject.build_launch_bundle(
            **{**_kwargs(evidence), "worker_iam_plan": forged}
        )


@pytest.mark.parametrize(
    "field,mutate",
    [
        ("bucket", lambda bundle: bundle.__setitem__("bucket", "wrong-bucket")),
        (
            "job",
            lambda bundle: bundle["bootstrap_inventory"][0].__setitem__(
                "job_id", bundle["bootstrap_inventory"][1]["job_id"]
            ),
        ),
        (
            "VM",
            lambda bundle: bundle["selected_instance_ids"].__setitem__(
                1, bundle["selected_instance_ids"][0]
            ),
        ),
        (
            "authorization",
            lambda bundle: bundle["bootstrap_inventory"][0].__setitem__(
                "prelaunch_authorization_sha256", "f" * 64
            ),
        ),
    ],
)
def test_resealed_bundle_cannot_change_bucket_job_vm_or_authorization(
    evidence: dict[str, Any], bundle: dict[str, Any], field: str, mutate: Any
) -> None:
    kwargs = _kwargs(evidence)
    forged = deepcopy(bundle)
    mutate(forged)
    forged["bootstrap_inventory_sha256"] = subject.canonical_sha256(
        forged["bootstrap_inventory"]
    )
    _reseal_bundle(forged)
    with pytest.raises(ValueError, match="launch bundle"):
        subject.validate_launch_bundle(**kwargs, value=forged)


def test_stale_create_window_or_readback_receipt_never_authorizes(
    evidence: dict[str, Any],
) -> None:
    kwargs = _kwargs(evidence)
    with pytest.raises(PermissionError, match="stale"):
        subject.build_launch_bundle(
            **{**kwargs, "current_time_utc": "2026-07-22T01:03:05Z"}
        )
    forged_readback = deepcopy(evidence["readback"])
    forged_readback["receipt_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="readback"):
        subject.build_launch_bundle(
            **{**kwargs, "worker_iam_readback_receipt": forged_readback}
        )


@pytest.mark.parametrize(
    "required_key",
    [
        "runtime_preflight_receipt",
        "gcp_read_receipt",
        "service_account_actas_receipt",
        "worker_identity_plan",
        "worker_identity_inventory_receipt",
        "worker_identity_act_as_receipt",
        "project_iam_scan_receipt",
    ],
)
def test_provider_and_identity_evidence_are_mandatory(
    evidence: dict[str, Any], required_key: str
) -> None:
    kwargs = _kwargs(evidence)
    kwargs.pop(required_key)
    with pytest.raises(TypeError, match=required_key):
        subject.build_launch_bundle(**kwargs)


def test_gcp_read_must_embed_the_separately_supplied_quota_and_mapping(
    evidence: dict[str, Any],
) -> None:
    alternate_quota = cloud_v2.build_live_quota_headroom_receipt(
        evidence["plan"],
        evidence["ledger"],
        evidence["resume"],
        immutable_content_sha256=evidence["outer"]["content_payload_sha256"],
        project_id=iam_v2.PROJECT,
        zone=ZONE,
        observed_at_utc="2026-07-22T01:00:01Z",
        expires_at_utc="2026-07-22T01:05:01Z",
        quota_metrics=[
            {
                **row,
                "limit_vcpus": row["limit_vcpus"] + 1,
                "usage_vcpus": row["usage_vcpus"] + 1,
            }
            for row in _quota_rows()
        ],
        readback_source="fake_backend_fixture",
    )
    gcp_read = deepcopy(evidence["gcp_read"])
    gcp_read["quota_receipt"] = alternate_quota
    _reseal_gcp_receipt(gcp_read)
    provider_actas = deepcopy(evidence["provider_actas"])
    provider_actas["gcp_read_receipt_sha256"] = gcp_read["receipt_sha256"]
    _reseal_gcp_receipt(provider_actas)
    with pytest.raises(ValueError, match="evidence chain"):
        subject.build_launch_bundle(
            **{
                **_kwargs(evidence),
                "gcp_read_receipt": gcp_read,
                "service_account_actas_receipt": provider_actas,
            }
        )


def test_provider_unique_id_drift_cannot_escape_identity_inventory(
    evidence: dict[str, Any],
) -> None:
    gcp_read = deepcopy(evidence["gcp_read"])
    gcp_read["service_accounts"][0]["unique_id"] = "999999999"
    gcp_read["service_accounts"][0]["resource_name"] = (
        f"projects/{iam_v2.PROJECT}/serviceAccounts/999999999"
    )
    _reseal_gcp_receipt(gcp_read)
    provider_actas = deepcopy(evidence["provider_actas"])
    provider_actas["gcp_read_receipt_sha256"] = gcp_read["receipt_sha256"]
    provider_actas["rows"][0]["unique_id"] = "999999999"
    provider_actas["rows"][0]["resource_name"] = (
        f"projects/{iam_v2.PROJECT}/serviceAccounts/999999999"
    )
    _reseal_gcp_receipt(provider_actas)
    with pytest.raises(ValueError, match="provider unique ID"):
        subject.build_launch_bundle(
            **{
                **_kwargs(evidence),
                "gcp_read_receipt": gcp_read,
                "service_account_actas_receipt": provider_actas,
            }
        )


def test_stale_worker_identity_evidence_cannot_authorize_create(
    evidence: dict[str, Any],
) -> None:
    inventory = deepcopy(evidence["identity_inventory"])
    inventory["observed_at_utc"] = "2026-07-22T00:54:00Z"
    _reseal_identity_receipt(inventory)
    actas = deepcopy(evidence["identity_actas"])
    actas["inventory_receipt_sha256"] = inventory["receipt_sha256"]
    _reseal_identity_receipt(actas)
    scan = deepcopy(evidence["project_scan"])
    scan["inventory_receipt_sha256"] = inventory["receipt_sha256"]
    scan["observed_at_utc"] = "2026-07-22T00:54:00Z"
    _reseal_identity_receipt(scan)
    with pytest.raises(PermissionError, match="stale"):
        subject.build_launch_bundle(
            **{
                **_kwargs(evidence),
                "worker_identity_inventory_receipt": inventory,
                "worker_identity_act_as_receipt": actas,
                "project_iam_scan_receipt": scan,
            }
        )


def test_runtime_preflight_drift_cannot_authorize_create(
    evidence: dict[str, Any],
) -> None:
    runtime = deepcopy(evidence["runtime_preflight"])
    runtime["launch_network_design"]["external_ipv4"] = True
    runtime["receipt_sha256"] = runtime_v2.canonical_sha256(
        {key: item for key, item in runtime.items() if key != "receipt_sha256"}
    )
    with pytest.raises(ValueError, match="runtime preflight"):
        subject.build_launch_bundle(
            **{**_kwargs(evidence), "runtime_preflight_receipt": runtime}
        )


def test_manual_runtime_preflight_without_live_source_cannot_authorize_create(
    evidence: dict[str, Any],
) -> None:
    fabricated_source = {
        "runtime_preflight_receipt": deepcopy(evidence["runtime_preflight"]),
        "receipt_sha256": evidence["runtime_preflight"]["receipt_sha256"],
    }
    with pytest.raises(ValueError, match="runtime GCP read receipt fields"):
        subject.build_launch_bundle(
            **{
                **_kwargs(evidence),
                "runtime_gcp_read_receipt": fabricated_source,
            }
        )
