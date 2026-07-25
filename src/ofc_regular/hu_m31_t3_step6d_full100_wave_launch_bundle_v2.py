"""Pure fail-closed launch bundle for one full100 wave-v2 create window.

This module does not call a cloud transport.  It joins the independently
validated wave, immutable package, cloud prelaunch, and worker-IAM evidence
into the only object that may authorize the selected VM creates.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Mapping

from . import hu_m31_t3_step6d_full100_wave_cloud_v2 as cloud_v2
from . import hu_m31_t3_step6d_full100_wave_gcp_adapter_v2 as gcp_adapter_v2
from . import hu_m31_t3_step6d_full100_wave_package_v2 as package_v2
from . import hu_m31_t3_step6d_full100_wave_plan_v2 as wave_v2
from . import hu_m31_t3_step6d_full100_wave_runtime_preflight_v2 as runtime_v2
from . import hu_m31_t3_step6d_full100_wave_runtime_gcp_adapter_v2 as runtime_gcp_v2
from . import hu_m31_t3_step6d_full100_wave_science_registry_v2 as science_registry
from . import hu_m31_t3_step6d_full100_wave_worker_identity_v2 as worker_identity_v2
from . import hu_m31_t3_step6d_full100_wave_worker_iam_v2 as worker_iam_v2


LAUNCH_BUNDLE_SCHEMA = "hu_m31_t3_step6d_full100_wave_launch_bundle_v2"
LAUNCH_BUNDLE_STATUS = "exact_selected_wave_launch_bundle_ready"
EXPECTED_STARTUP_SHA256 = (
    "204a12c40b56dda643b7228e1687818a618bf1cb4a1f50359187b113c82a7d87"
)
MAX_IDENTITY_EVIDENCE_AGE_SECONDS = 300
STARTUP_WATCHDOG_SECONDS = 4_200
IAM_PROVISION_UPLOAD_MARGIN_SECONDS = 300
IAM_MIN_REMAINING_SECONDS = (
    STARTUP_WATCHDOG_SECONDS + IAM_PROVISION_UPLOAD_MARGIN_SECONDS
)

_INVENTORY_ROW_KEYS = frozenset(
    {
        "job_id",
        "source_role",
        "attempt_id",
        "instance_id",
        "service_account",
        "prelaunch_authorization_sha256",
        "bootstrap_sha256",
        "bootstrap",
    }
)
_BUNDLE_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "execution_identity_sha256",
        "wave_plan_sha256",
        "attempt_ledger_sha256",
        "resume_plan_sha256",
        "outer_manifest_sha256",
        "immutable_content_sha256",
        "immutable_content_prefix",
        "project_id",
        "zone",
        "bucket",
        "wave_index",
        "quota_receipt_sha256",
        "persistent_claim_receipt_sha256",
        "planned_mapping_receipt_sha256",
        "prelaunch_authorization_sha256",
        "runtime_preflight_receipt_sha256",
        "runtime_gcp_read_receipt_sha256",
        "gcp_read_receipt_sha256",
        "gcp_custom_roles_sha256",
        "gcp_selected_result_preflight_receipt_sha256",
        "service_account_actas_receipt_sha256",
        "worker_identity_plan_sha256",
        "worker_identity_inventory_receipt_sha256",
        "worker_identity_act_as_receipt_sha256",
        "project_iam_scan_receipt_sha256",
        "worker_iam_plan_sha256",
        "worker_iam_prepare_receipt_sha256",
        "worker_iam_install_receipt_sha256",
        "worker_iam_readback_receipt_sha256",
        "current_time_utc",
        "selected_vm_count",
        "selected_job_ids",
        "selected_source_roles",
        "selected_attempt_ids",
        "selected_instance_ids",
        "selected_service_accounts",
        "selected_identity_sha256",
        "bootstrap_inventory",
        "bootstrap_inventory_sha256",
        "worker_iam_evidence_embedded",
        "worker_iam_readback_complete",
        "runtime_preflight_complete",
        "runtime_gcp_live_source_complete",
        "gcp_readback_complete",
        "gcp_custom_roles_exact_ga_not_deleted",
        "gcp_selected_result_preflight_absence_complete",
        "service_account_act_as_complete",
        "provider_project_permissions_complete",
        "worker_identity_evidence_complete",
        "project_iam_zero_roles",
        "one_vm_one_job_one_role",
        "cloud_create_authorized",
        "exact_selected_create_authorized",
        "one_shot",
        "reuse_authorized",
        "additional_create_authorized",
        "unlisted_instance_create_authorized",
        "legacy_launcher_authorized",
        "cloud_started",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
        "hidden_truth_exposed",
        "bundle_sha256",
    }
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields changed")


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} is invalid") from exc
    if parsed.tzinfo != timezone.utc:
        raise ValueError(f"{label} is not UTC")
    return parsed


def _require_fresh_identity_time(
    observed_at_utc: Any, current_time_utc: str, label: str
) -> None:
    age = (
        _parse_utc(current_time_utc, "current time")
        - _parse_utc(observed_at_utc, label)
    ).total_seconds()
    if not 0 <= age <= MAX_IDENTITY_EVIDENCE_AGE_SECONDS:
        raise PermissionError(f"{label} is stale or future-dated")


def _validated_evidence(
    *,
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    outer_manifest: Mapping[str, Any],
    expected_startup_sha256: str,
    quota_receipt: Mapping[str, Any],
    persistent_claim_receipt: Mapping[str, Any],
    planned_mapping_receipt: Mapping[str, Any],
    prelaunch_authorization: Mapping[str, Any],
    raw_claim_nonce: str,
    current_time_utc: str,
    runtime_preflight_receipt: Mapping[str, Any],
    runtime_gcp_read_receipt: Mapping[str, Any],
    gcp_read_receipt: Mapping[str, Any],
    service_account_actas_receipt: Mapping[str, Any],
    worker_identity_plan: Mapping[str, Any],
    worker_identity_inventory_receipt: Mapping[str, Any],
    worker_identity_act_as_receipt: Mapping[str, Any],
    project_iam_scan_receipt: Mapping[str, Any],
    worker_iam_plan: Mapping[str, Any],
    worker_iam_prepare_receipt: Mapping[str, Any],
    worker_iam_install_receipt: Mapping[str, Any],
    worker_iam_readback_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    plan = wave_v2.validate_wave_plan(wave_plan)
    ledger = wave_v2.validate_attempt_ledger(plan, attempt_ledger)
    resume = wave_v2.validate_resume_plan(plan, ledger, resume_plan)
    if resume["all_jobs_complete"] is True or not resume["selected_attempts"]:
        raise PermissionError("completed or empty wave has no create authorization")
    runtime_gcp_read = runtime_gcp_v2.validate_runtime_gcp_read_receipt(
        wave_plan=plan,
        value=runtime_gcp_read_receipt,
        current_utc=current_time_utc,
    )
    runtime_preflight = runtime_v2.validate_runtime_preflight_receipt(
        wave_plan=plan,
        value=runtime_preflight_receipt,
        current_utc=current_time_utc,
    )
    if runtime_gcp_read["runtime_preflight_receipt"] != runtime_preflight:
        raise ValueError("runtime preflight is not the live GCP-derived receipt")

    manifest = package_v2.validate_outer_manifest(
        plan,
        outer_manifest,
        expected_startup_sha256=expected_startup_sha256,
    )
    content_sha = manifest["content_payload_sha256"]
    quota = cloud_v2.validate_live_quota_headroom_receipt(
        plan,
        ledger,
        resume,
        quota_receipt,
        immutable_content_sha256=content_sha,
        now_utc=current_time_utc,
    )
    claim = cloud_v2.validate_persistent_atomic_launch_claim_receipt(
        plan,
        ledger,
        resume,
        persistent_claim_receipt,
        immutable_content_sha256=content_sha,
        raw_claim_nonce=raw_claim_nonce,
    )
    mapping = cloud_v2.validate_planned_launch_mapping_receipt(
        plan,
        ledger,
        resume,
        planned_mapping_receipt,
        immutable_content_sha256=content_sha,
        now_utc=current_time_utc,
    )
    authorization = cloud_v2.validate_prelaunch_authorization(
        plan,
        ledger,
        resume,
        immutable_content_sha256=content_sha,
        quota_receipt=quota,
        persistent_claim_receipt=claim,
        planned_mapping_receipt=mapping,
        value=prelaunch_authorization,
        raw_claim_nonce=raw_claim_nonce,
        now_utc=current_time_utc,
    )

    iam_plan = worker_iam_v2.validate_worker_iam_plan(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        immutable_content_prefix=manifest["content_prefix"],
        content_payload_sha256=content_sha,
        outer_manifest_sha256=manifest["manifest_sha256"],
        value=worker_iam_plan,
    )
    iam_kwargs = {
        "iam_plan": iam_plan,
        "wave_plan": plan,
        "attempt_ledger": ledger,
        "resume_plan": resume,
        "immutable_content_prefix": manifest["content_prefix"],
        "content_payload_sha256": content_sha,
        "outer_manifest_sha256": manifest["manifest_sha256"],
    }
    prepare = worker_iam_v2.validate_prepare_receipt(
        **iam_kwargs,
        value=worker_iam_prepare_receipt,
    )
    install = worker_iam_v2.validate_install_receipt(
        **iam_kwargs,
        prepare_receipt=prepare,
        value=worker_iam_install_receipt,
    )
    readback = worker_iam_v2.validate_readback_receipt(
        **iam_kwargs,
        prepare_receipt=prepare,
        install_receipt=install,
        value=worker_iam_readback_receipt,
    )

    gcp_read = gcp_adapter_v2.validate_read_receipt(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        immutable_content_prefix=manifest["content_prefix"],
        content_payload_sha256=content_sha,
        outer_manifest_sha256=manifest["manifest_sha256"],
        iam_plan=iam_plan,
        value=gcp_read_receipt,
    )
    provider_actas = gcp_adapter_v2.validate_service_account_actas_receipt(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        immutable_content_prefix=manifest["content_prefix"],
        content_payload_sha256=content_sha,
        outer_manifest_sha256=manifest["manifest_sha256"],
        iam_plan=iam_plan,
        gcp_read_receipt=gcp_read,
        value=service_account_actas_receipt,
        now_utc=current_time_utc,
    )
    identity_plan = worker_identity_v2.validate_worker_identity_plan(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        value=worker_identity_plan,
    )
    identity_inventory = worker_identity_v2.validate_inventory_receipt(
        identity_plan=identity_plan,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        value=worker_identity_inventory_receipt,
    )
    identity_actas = worker_identity_v2.validate_act_as_receipt(
        identity_plan=identity_plan,
        inventory_receipt=identity_inventory,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        value=worker_identity_act_as_receipt,
    )
    project_scan = worker_identity_v2.validate_project_iam_scan_receipt(
        identity_plan=identity_plan,
        inventory_receipt=identity_inventory,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        value=project_iam_scan_receipt,
    )

    if (
        gcp_read["quota_receipt"]["receipt_sha256"]
        != quota["receipt_sha256"]
        or gcp_read["planned_mapping_receipt"]["receipt_sha256"]
        != mapping["receipt_sha256"]
        or provider_actas["gcp_read_receipt_sha256"]
        != gcp_read["receipt_sha256"]
        or identity_inventory["identity_plan_sha256"]
        != identity_plan["plan_sha256"]
        or identity_actas["identity_plan_sha256"]
        != identity_plan["plan_sha256"]
        or identity_actas["inventory_receipt_sha256"]
        != identity_inventory["receipt_sha256"]
        # These are two independent live POST checks separated by the project
        # IAM scan and the Phase-A reads.  Requiring byte-identical seconds is
        # invalid once the live reads cross a second boundary.  Preserve the
        # actual chain direction; freshness for both receipts is validated
        # independently below and by the provider receipt validator.
        or _parse_utc(
            identity_actas["tested_at_utc"], "worker identity actAs time"
        )
        > _parse_utc(
            provider_actas["checked_at_utc"], "provider actAs time"
        )
        or project_scan["identity_plan_sha256"]
        != identity_plan["plan_sha256"]
        or project_scan["inventory_receipt_sha256"]
        != identity_inventory["receipt_sha256"]
        or project_scan["pool_role_membership_count"] != 0
        or project_scan["all_pool_accounts_have_zero_project_roles"] is not True
    ):
        raise ValueError("provider and worker-identity evidence chain changed")
    if not (
        gcp_read["observed_at_utc"]
        == gcp_read["selected_result_preflight_receipt"]["observed_at_utc"]
        == quota["observed_at_utc"]
        == mapping["observed_at_utc"]
    ):
        raise ValueError("GCP Phase-A observation-time evidence chain changed")
    _require_fresh_identity_time(
        gcp_read["observed_at_utc"],
        current_time_utc,
        "GCP Phase-A read time",
    )
    _require_fresh_identity_time(
        identity_inventory["observed_at_utc"],
        current_time_utc,
        "worker identity inventory time",
    )
    _require_fresh_identity_time(
        identity_actas["tested_at_utc"],
        current_time_utc,
        "worker identity actAs time",
    )
    _require_fresh_identity_time(
        project_scan["observed_at_utc"],
        current_time_utc,
        "project IAM scan time",
    )

    latest_transition = ledger["transitions"][-1]
    if (
        authorization["project_id"] != latest_transition["project_id"]
        or authorization["zone"] != latest_transition["zone"]
        or authorization["project_id"] != iam_plan["project"]
        or runtime_preflight["project"] != authorization["project_id"]
        or runtime_preflight["zone"] != authorization["zone"]
        or runtime_preflight["bucket"]["name"] != iam_plan["bucket"]
        or runtime_preflight["runtime_image_digest"]
        != plan["runtime_binding"]["image_digest"]
        or gcp_read["project"] != authorization["project_id"]
        or gcp_read["zone"] != authorization["zone"]
        or gcp_read["bucket"] != iam_plan["bucket"]
        or identity_plan["project"] != iam_plan["project"]
        or identity_inventory["project"] != iam_plan["project"]
        or identity_actas["project"] != iam_plan["project"]
        or project_scan["project"] != iam_plan["project"]
        or iam_plan["bucket"] != worker_iam_v2.BUCKET
        or authorization["wave_index"] != resume["resume_wave_index"]
        or iam_plan["wave_index"] != resume["resume_wave_index"]
    ):
        raise ValueError("project, zone, bucket, or wave identity changed")

    now = _parse_utc(current_time_utc, "current time")
    iam_issued = _parse_utc(iam_plan["issued_at_utc"], "IAM issue time")
    iam_expires = _parse_utc(iam_plan["expires_at_utc"], "IAM expiry")
    auth_expires = _parse_utc(
        authorization["expires_at_utc"], "prelaunch authorization expiry"
    )
    iam_remaining_seconds = (iam_expires - now).total_seconds()
    if (
        iam_issued > now
        or iam_remaining_seconds < IAM_MIN_REMAINING_SECONDS
        or auth_expires > iam_expires
    ):
        raise PermissionError(
            "worker IAM is not live for the full provision, runtime, and upload window"
        )

    selected = resume["selected_attempts"]
    mapping_rows = mapping["rows"]
    workers = iam_plan["workers"]
    identity_workers = identity_plan["selected_workers"]
    gcp_accounts = gcp_read["service_accounts"]
    provider_actas_rows = provider_actas["rows"]
    identity_actas_rows = identity_actas["rows"]
    if not (
        len(selected)
        == len(mapping_rows)
        == len(workers)
        == len(identity_workers)
        == len(gcp_accounts)
        == len(provider_actas_rows)
        == len(identity_actas_rows)
    ):
        raise ValueError("selected job, mapping, and IAM cardinality changed")

    selected_identity = [
        {
            "job_id": row["job_id"],
            "source_role": row["source_role"],
            "attempt_id": row["attempt_id"],
            "instance_id": row["instance_id"],
            "artifact_prefix": row["artifact_prefix"],
        }
        for row in selected
    ]
    mapping_identity = [
        {key: row[key] for key in selected_identity[index]}
        for index, row in enumerate(mapping_rows)
    ]
    worker_identity = [
        {
            "job_id": row["job_id"],
            "source_role": row["source_role"],
            "attempt_id": row["attempt_id"],
            "instance_id": row["vm_instance_id"],
        }
        for row in workers
    ]
    selected_worker_identity = [
        {key: row[key] for key in worker_identity[index]}
        for index, row in enumerate(selected_identity)
    ]
    if (
        mapping_identity != selected_identity
        or worker_identity != selected_worker_identity
    ):
        raise ValueError("job, role, attempt, VM, or artifact mapping changed")

    inventory_by_email = {
        row["email"]: row for row in identity_inventory["rows"]
    }
    for index, (
        selected_row,
        iam_worker,
        identity_worker,
        gcp_account,
        provider_row,
        pure_actas_row,
    ) in enumerate(
        zip(
            selected,
            workers,
            identity_workers,
            gcp_accounts,
            provider_actas_rows,
            identity_actas_rows,
            strict=True,
        )
    ):
        email = identity_worker["service_account_email"]
        inventory_row = inventory_by_email.get(email)
        if (
            identity_worker["worker_slot"] != index
            or identity_worker["job_id"] != selected_row["job_id"]
            or identity_worker["source_role"] != selected_row["source_role"]
            or identity_worker["attempt_id"] != selected_row["attempt_id"]
            or identity_worker["vm_instance_id"] != selected_row["instance_id"]
            or iam_worker["service_account"] != email
            or gcp_account["service_account"] != email
            or provider_row["service_account"] != email
            or pure_actas_row["email"] != email
            or pure_actas_row["account_id"] != identity_worker["account_id"]
            or pure_actas_row["name"]
            != identity_worker["service_account_name"]
            or inventory_row is None
            or gcp_account["unique_id"] != inventory_row["unique_id"]
            or provider_row["unique_id"] != inventory_row["unique_id"]
        ):
            raise ValueError(
                "selected worker account order or provider unique ID changed"
            )

    jobs = [row["job_id"] for row in selected]
    instances = [row["instance_id"] for row in selected]
    accounts = [row["service_account"] for row in workers]
    principals = [row["principal"] for row in workers]
    if (
        len(set(jobs)) != len(jobs)
        or len(set(instances)) != len(instances)
        or len(set(accounts)) != len(accounts)
        or len(set(principals)) != len(principals)
        or any(
            principal != f"serviceAccount:{account}"
            for principal, account in zip(principals, accounts, strict=True)
        )
    ):
        raise ValueError("job, VM, or service-account identity is duplicated")

    return (
        plan,
        ledger,
        resume,
        runtime_preflight,
        runtime_gcp_read,
        manifest,
        quota,
        claim,
        mapping,
        authorization,
        gcp_read,
        provider_actas,
        identity_plan,
        identity_inventory,
        identity_actas,
        project_scan,
        iam_plan,
        prepare,
        install,
        readback,
    )


def _build_core(
    *,
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    outer_manifest: Mapping[str, Any],
    expected_startup_sha256: str | None,
    quota_receipt: Mapping[str, Any],
    persistent_claim_receipt: Mapping[str, Any],
    planned_mapping_receipt: Mapping[str, Any],
    prelaunch_authorization: Mapping[str, Any],
    raw_claim_nonce: str,
    current_time_utc: str,
    runtime_preflight_receipt: Mapping[str, Any],
    runtime_gcp_read_receipt: Mapping[str, Any],
    gcp_read_receipt: Mapping[str, Any],
    service_account_actas_receipt: Mapping[str, Any],
    worker_identity_plan: Mapping[str, Any],
    worker_identity_inventory_receipt: Mapping[str, Any],
    worker_identity_act_as_receipt: Mapping[str, Any],
    project_iam_scan_receipt: Mapping[str, Any],
    worker_iam_plan: Mapping[str, Any],
    worker_iam_prepare_receipt: Mapping[str, Any],
    worker_iam_install_receipt: Mapping[str, Any],
    worker_iam_readback_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    startup_sha256 = science_registry.resolve_startup_sha256(
        wave_plan, expected_startup_sha256
    )
    (
        plan,
        ledger,
        resume,
        runtime_preflight,
        runtime_gcp_read,
        manifest,
        quota,
        claim,
        mapping,
        authorization,
        gcp_read,
        provider_actas,
        identity_plan,
        identity_inventory,
        identity_actas,
        project_scan,
        iam_plan,
        prepare,
        install,
        readback,
    ) = _validated_evidence(
        wave_plan=wave_plan,
        attempt_ledger=attempt_ledger,
        resume_plan=resume_plan,
        outer_manifest=outer_manifest,
        expected_startup_sha256=startup_sha256,
        quota_receipt=quota_receipt,
        persistent_claim_receipt=persistent_claim_receipt,
        planned_mapping_receipt=planned_mapping_receipt,
        prelaunch_authorization=prelaunch_authorization,
        raw_claim_nonce=raw_claim_nonce,
        current_time_utc=current_time_utc,
        runtime_preflight_receipt=runtime_preflight_receipt,
        runtime_gcp_read_receipt=runtime_gcp_read_receipt,
        gcp_read_receipt=gcp_read_receipt,
        service_account_actas_receipt=service_account_actas_receipt,
        worker_identity_plan=worker_identity_plan,
        worker_identity_inventory_receipt=worker_identity_inventory_receipt,
        worker_identity_act_as_receipt=worker_identity_act_as_receipt,
        project_iam_scan_receipt=project_iam_scan_receipt,
        worker_iam_plan=worker_iam_plan,
        worker_iam_prepare_receipt=worker_iam_prepare_receipt,
        worker_iam_install_receipt=worker_iam_install_receipt,
        worker_iam_readback_receipt=worker_iam_readback_receipt,
    )

    selected = resume["selected_attempts"]
    workers = iam_plan["workers"]
    inventory: list[dict[str, Any]] = []
    for selected_row, worker in zip(selected, workers, strict=True):
        bootstrap = package_v2.build_job_bootstrap_metadata(
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            outer_manifest=manifest,
            job_id=selected_row["job_id"],
            bucket=iam_plan["bucket"],
            worker_principal=worker["service_account"],
            prelaunch_authorization_sha256=authorization[
                "authorization_sha256"
            ],
            expected_startup_sha256=startup_sha256,
        )
        if (
            bootstrap["job_id"] != selected_row["job_id"]
            or bootstrap["source_role"] != selected_row["source_role"]
            or bootstrap["attempt_id"] != selected_row["attempt_id"]
            or bootstrap["instance_name"] != selected_row["instance_id"]
            or bootstrap["bucket"] != iam_plan["bucket"]
            or bootstrap["worker_principal"] != worker["service_account"]
            or bootstrap["prelaunch_authorization_sha256"]
            != authorization["authorization_sha256"]
        ):
            raise ValueError("generated job bootstrap identity changed")
        inventory.append(
            {
                "job_id": selected_row["job_id"],
                "source_role": selected_row["source_role"],
                "attempt_id": selected_row["attempt_id"],
                "instance_id": selected_row["instance_id"],
                "service_account": worker["service_account"],
                "prelaunch_authorization_sha256": authorization[
                    "authorization_sha256"
                ],
                "bootstrap_sha256": bootstrap["bootstrap_sha256"],
                "bootstrap": bootstrap,
            }
        )

    for row in inventory:
        _exact_keys(row, _INVENTORY_ROW_KEYS, "bootstrap inventory row")
    selected_identity = [
        {
            "job_id": row["job_id"],
            "source_role": row["source_role"],
            "attempt_id": row["attempt_id"],
            "instance_id": row["instance_id"],
            "service_account": worker["service_account"],
        }
        for row, worker in zip(selected, workers, strict=True)
    ]
    core: dict[str, Any] = {
        "schema": LAUNCH_BUNDLE_SCHEMA,
        "status": LAUNCH_BUNDLE_STATUS,
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_plan_sha256": plan["schedule_sha256"],
        "attempt_ledger_sha256": ledger["ledger_sha256"],
        "resume_plan_sha256": resume["resume_sha256"],
        "outer_manifest_sha256": manifest["manifest_sha256"],
        "immutable_content_sha256": manifest["content_payload_sha256"],
        "immutable_content_prefix": manifest["content_prefix"],
        "project_id": authorization["project_id"],
        "zone": authorization["zone"],
        "bucket": iam_plan["bucket"],
        "wave_index": resume["resume_wave_index"],
        "quota_receipt_sha256": quota["receipt_sha256"],
        "persistent_claim_receipt_sha256": claim["receipt_sha256"],
        "planned_mapping_receipt_sha256": mapping["receipt_sha256"],
        "prelaunch_authorization_sha256": authorization[
            "authorization_sha256"
        ],
        "runtime_preflight_receipt_sha256": runtime_preflight[
            "receipt_sha256"
        ],
        "runtime_gcp_read_receipt_sha256": runtime_gcp_read["receipt_sha256"],
        "gcp_read_receipt_sha256": gcp_read["receipt_sha256"],
        "gcp_custom_roles_sha256": gcp_read["custom_roles_sha256"],
        "gcp_selected_result_preflight_receipt_sha256": gcp_read[
            "selected_result_preflight_receipt"
        ]["receipt_sha256"],
        "service_account_actas_receipt_sha256": provider_actas[
            "receipt_sha256"
        ],
        "worker_identity_plan_sha256": identity_plan["plan_sha256"],
        "worker_identity_inventory_receipt_sha256": identity_inventory[
            "receipt_sha256"
        ],
        "worker_identity_act_as_receipt_sha256": identity_actas[
            "receipt_sha256"
        ],
        "project_iam_scan_receipt_sha256": project_scan["receipt_sha256"],
        "worker_iam_plan_sha256": iam_plan["plan_sha256"],
        "worker_iam_prepare_receipt_sha256": prepare["receipt_sha256"],
        "worker_iam_install_receipt_sha256": install["receipt_sha256"],
        "worker_iam_readback_receipt_sha256": readback["receipt_sha256"],
        "current_time_utc": current_time_utc,
        "selected_vm_count": len(selected),
        "selected_job_ids": [row["job_id"] for row in selected],
        "selected_source_roles": [row["source_role"] for row in selected],
        "selected_attempt_ids": [row["attempt_id"] for row in selected],
        "selected_instance_ids": [row["instance_id"] for row in selected],
        "selected_service_accounts": [
            row["service_account"] for row in workers
        ],
        "selected_identity_sha256": canonical_sha256(selected_identity),
        "bootstrap_inventory": inventory,
        "bootstrap_inventory_sha256": canonical_sha256(inventory),
        "worker_iam_evidence_embedded": True,
        "worker_iam_readback_complete": True,
        "runtime_preflight_complete": True,
        "runtime_gcp_live_source_complete": True,
        "gcp_readback_complete": True,
        "gcp_custom_roles_exact_ga_not_deleted": True,
        "gcp_selected_result_preflight_absence_complete": True,
        "service_account_act_as_complete": True,
        "provider_project_permissions_complete": True,
        "worker_identity_evidence_complete": True,
        "project_iam_zero_roles": True,
        "one_vm_one_job_one_role": True,
        "cloud_create_authorized": True,
        "exact_selected_create_authorized": True,
        "one_shot": True,
        "reuse_authorized": False,
        "additional_create_authorized": False,
        "unlisted_instance_create_authorized": False,
        "legacy_launcher_authorized": False,
        "cloud_started": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "hidden_truth_exposed": False,
    }
    return core


def _validate_bundle_surface(
    payload: Mapping[str, Any],
    *,
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    outer_manifest: Mapping[str, Any],
    quota_receipt: Mapping[str, Any],
    persistent_claim_receipt: Mapping[str, Any],
    planned_mapping_receipt: Mapping[str, Any],
    prelaunch_authorization: Mapping[str, Any],
    current_time_utc: str,
    runtime_preflight_receipt: Mapping[str, Any],
    runtime_gcp_read_receipt: Mapping[str, Any],
    gcp_read_receipt: Mapping[str, Any],
    service_account_actas_receipt: Mapping[str, Any],
    worker_identity_plan: Mapping[str, Any],
    worker_identity_inventory_receipt: Mapping[str, Any],
    worker_identity_act_as_receipt: Mapping[str, Any],
    project_iam_scan_receipt: Mapping[str, Any],
    worker_iam_plan: Mapping[str, Any],
    worker_iam_prepare_receipt: Mapping[str, Any],
    worker_iam_install_receipt: Mapping[str, Any],
    worker_iam_readback_receipt: Mapping[str, Any],
) -> None:
    """Reject self-consistent bundle tampering before expensive re-derivation."""

    evidence_objects = (
        attempt_ledger,
        resume_plan,
        outer_manifest,
        quota_receipt,
        persistent_claim_receipt,
        planned_mapping_receipt,
        prelaunch_authorization,
        runtime_preflight_receipt,
        runtime_gcp_read_receipt,
        gcp_read_receipt,
        service_account_actas_receipt,
        worker_identity_plan,
        worker_identity_inventory_receipt,
        worker_identity_act_as_receipt,
        project_iam_scan_receipt,
        worker_iam_plan,
        worker_iam_prepare_receipt,
        worker_iam_install_receipt,
        worker_iam_readback_receipt,
    )
    if any(not isinstance(value, Mapping) for value in evidence_objects):
        raise ValueError("launch bundle surface evidence is not an object")
    selected_result_preflight = gcp_read_receipt.get(
        "selected_result_preflight_receipt"
    )
    if not isinstance(selected_result_preflight, Mapping):
        raise ValueError("launch bundle selected-result preflight is missing")
    selected = resume_plan.get("selected_attempts")
    workers = worker_iam_plan.get("workers")
    inventory = payload.get("bootstrap_inventory")
    if (
        not isinstance(selected, list)
        or not isinstance(workers, list)
        or not isinstance(inventory, list)
        or not selected
        or not (len(selected) == len(workers) == len(inventory))
        or any(not isinstance(row, Mapping) for row in selected)
        or any(not isinstance(row, Mapping) for row in workers)
    ):
        raise ValueError("launch bundle surface cardinality changed")
    if (
        payload.get("schema") != LAUNCH_BUNDLE_SCHEMA
        or payload.get("status") != LAUNCH_BUNDLE_STATUS
        or payload.get("attempt_ledger_sha256")
        != attempt_ledger.get("ledger_sha256")
        or payload.get("resume_plan_sha256") != resume_plan.get("resume_sha256")
        or payload.get("outer_manifest_sha256")
        != outer_manifest.get("manifest_sha256")
        or payload.get("immutable_content_sha256")
        != outer_manifest.get("content_payload_sha256")
        or payload.get("immutable_content_prefix")
        != outer_manifest.get("content_prefix")
        or payload.get("project_id")
        != prelaunch_authorization.get("project_id")
        or payload.get("zone") != prelaunch_authorization.get("zone")
        or payload.get("bucket") != worker_iam_plan.get("bucket")
        or payload.get("wave_index") != resume_plan.get("resume_wave_index")
        or payload.get("quota_receipt_sha256")
        != quota_receipt.get("receipt_sha256")
        or payload.get("persistent_claim_receipt_sha256")
        != persistent_claim_receipt.get("receipt_sha256")
        or payload.get("planned_mapping_receipt_sha256")
        != planned_mapping_receipt.get("receipt_sha256")
        or payload.get("prelaunch_authorization_sha256")
        != prelaunch_authorization.get("authorization_sha256")
        or payload.get("runtime_preflight_receipt_sha256")
        != runtime_preflight_receipt.get("receipt_sha256")
        or payload.get("runtime_gcp_read_receipt_sha256")
        != runtime_gcp_read_receipt.get("receipt_sha256")
        or payload.get("gcp_read_receipt_sha256")
        != gcp_read_receipt.get("receipt_sha256")
        or payload.get("gcp_custom_roles_sha256")
        != gcp_read_receipt.get("custom_roles_sha256")
        or payload.get("gcp_selected_result_preflight_receipt_sha256")
        != selected_result_preflight.get("receipt_sha256")
        or payload.get("service_account_actas_receipt_sha256")
        != service_account_actas_receipt.get("receipt_sha256")
        or payload.get("worker_identity_plan_sha256")
        != worker_identity_plan.get("plan_sha256")
        or payload.get("worker_identity_inventory_receipt_sha256")
        != worker_identity_inventory_receipt.get("receipt_sha256")
        or payload.get("worker_identity_act_as_receipt_sha256")
        != worker_identity_act_as_receipt.get("receipt_sha256")
        or payload.get("project_iam_scan_receipt_sha256")
        != project_iam_scan_receipt.get("receipt_sha256")
        or payload.get("worker_iam_plan_sha256")
        != worker_iam_plan.get("plan_sha256")
        or payload.get("worker_iam_prepare_receipt_sha256")
        != worker_iam_prepare_receipt.get("receipt_sha256")
        or payload.get("worker_iam_install_receipt_sha256")
        != worker_iam_install_receipt.get("receipt_sha256")
        or payload.get("worker_iam_readback_receipt_sha256")
        != worker_iam_readback_receipt.get("receipt_sha256")
        or payload.get("current_time_utc") != current_time_utc
    ):
        raise ValueError("launch bundle surface evidence binding changed")
    required_true = (
        "worker_iam_evidence_embedded",
        "worker_iam_readback_complete",
        "runtime_preflight_complete",
        "runtime_gcp_live_source_complete",
        "gcp_readback_complete",
        "gcp_custom_roles_exact_ga_not_deleted",
        "gcp_selected_result_preflight_absence_complete",
        "service_account_act_as_complete",
        "provider_project_permissions_complete",
        "worker_identity_evidence_complete",
        "project_iam_zero_roles",
        "one_vm_one_job_one_role",
        "cloud_create_authorized",
        "exact_selected_create_authorized",
        "one_shot",
    )
    required_false = (
        "reuse_authorized",
        "additional_create_authorized",
        "unlisted_instance_create_authorized",
        "legacy_launcher_authorized",
        "cloud_started",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
        "hidden_truth_exposed",
    )
    if any(payload.get(field) is not True for field in required_true) or any(
        payload.get(field) is not False for field in required_false
    ):
        raise ValueError("launch bundle authorization boundary changed")

    expected_jobs = [row.get("job_id") for row in selected]
    expected_roles = [row.get("source_role") for row in selected]
    expected_attempts = [row.get("attempt_id") for row in selected]
    expected_instances = [row.get("instance_id") for row in selected]
    expected_accounts = [row.get("service_account") for row in workers]
    selected_identity = [
        {
            "job_id": selected[index].get("job_id"),
            "source_role": selected[index].get("source_role"),
            "attempt_id": selected[index].get("attempt_id"),
            "instance_id": selected[index].get("instance_id"),
            "service_account": workers[index].get("service_account"),
        }
        for index in range(len(selected))
    ]
    if (
        payload.get("selected_vm_count") != len(selected)
        or payload.get("selected_job_ids") != expected_jobs
        or payload.get("selected_source_roles") != expected_roles
        or payload.get("selected_attempt_ids") != expected_attempts
        or payload.get("selected_instance_ids") != expected_instances
        or payload.get("selected_service_accounts") != expected_accounts
        or payload.get("selected_identity_sha256")
        != canonical_sha256(selected_identity)
        or len(set(expected_jobs)) != len(expected_jobs)
        or len(set(expected_instances)) != len(expected_instances)
        or len(set(expected_accounts)) != len(expected_accounts)
        or payload.get("bootstrap_inventory_sha256")
        != canonical_sha256(inventory)
    ):
        raise ValueError("launch bundle selected identity changed")
    auth_sha = payload["prelaunch_authorization_sha256"]
    for index, raw in enumerate(inventory):
        if not isinstance(raw, Mapping):
            raise ValueError("bootstrap inventory row is not an object")
        row = dict(raw)
        _exact_keys(row, _INVENTORY_ROW_KEYS, "bootstrap inventory row")
        bootstrap = row.get("bootstrap")
        if not isinstance(bootstrap, Mapping):
            raise ValueError("bootstrap inventory metadata is missing")
        expected = selected[index]
        account = workers[index].get("service_account")
        if (
            row.get("job_id") != expected.get("job_id")
            or row.get("source_role") != expected.get("source_role")
            or row.get("attempt_id") != expected.get("attempt_id")
            or row.get("instance_id") != expected.get("instance_id")
            or row.get("service_account") != account
            or row.get("prelaunch_authorization_sha256") != auth_sha
            or row.get("bootstrap_sha256")
            != bootstrap.get("bootstrap_sha256")
            or bootstrap.get("job_id") != expected.get("job_id")
            or bootstrap.get("source_role") != expected.get("source_role")
            or bootstrap.get("attempt_id") != expected.get("attempt_id")
            or bootstrap.get("instance_name") != expected.get("instance_id")
            or bootstrap.get("bucket") != payload.get("bucket")
            or bootstrap.get("worker_principal") != account
            or bootstrap.get("prelaunch_authorization_sha256") != auth_sha
            or bootstrap.get("bootstrap_sha256")
            != package_v2.canonical_sha256(
                {
                    key: value
                    for key, value in bootstrap.items()
                    if key != "bootstrap_sha256"
                }
            )
        ):
            raise ValueError("launch bundle bootstrap identity changed")


def build_launch_bundle(
    *,
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    outer_manifest: Mapping[str, Any],
    expected_startup_sha256: str | None = None,
    quota_receipt: Mapping[str, Any],
    persistent_claim_receipt: Mapping[str, Any],
    planned_mapping_receipt: Mapping[str, Any],
    prelaunch_authorization: Mapping[str, Any],
    raw_claim_nonce: str,
    current_time_utc: str,
    runtime_preflight_receipt: Mapping[str, Any],
    runtime_gcp_read_receipt: Mapping[str, Any],
    gcp_read_receipt: Mapping[str, Any],
    service_account_actas_receipt: Mapping[str, Any],
    worker_identity_plan: Mapping[str, Any],
    worker_identity_inventory_receipt: Mapping[str, Any],
    worker_identity_act_as_receipt: Mapping[str, Any],
    project_iam_scan_receipt: Mapping[str, Any],
    worker_iam_plan: Mapping[str, Any],
    worker_iam_prepare_receipt: Mapping[str, Any],
    worker_iam_install_receipt: Mapping[str, Any],
    worker_iam_readback_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the pure one-shot bundle after every independent proof validates."""

    core = _build_core(
        wave_plan=wave_plan,
        attempt_ledger=attempt_ledger,
        resume_plan=resume_plan,
        outer_manifest=outer_manifest,
        expected_startup_sha256=expected_startup_sha256,
        quota_receipt=quota_receipt,
        persistent_claim_receipt=persistent_claim_receipt,
        planned_mapping_receipt=planned_mapping_receipt,
        prelaunch_authorization=prelaunch_authorization,
        raw_claim_nonce=raw_claim_nonce,
        current_time_utc=current_time_utc,
        runtime_preflight_receipt=runtime_preflight_receipt,
        runtime_gcp_read_receipt=runtime_gcp_read_receipt,
        gcp_read_receipt=gcp_read_receipt,
        service_account_actas_receipt=service_account_actas_receipt,
        worker_identity_plan=worker_identity_plan,
        worker_identity_inventory_receipt=worker_identity_inventory_receipt,
        worker_identity_act_as_receipt=worker_identity_act_as_receipt,
        project_iam_scan_receipt=project_iam_scan_receipt,
        worker_iam_plan=worker_iam_plan,
        worker_iam_prepare_receipt=worker_iam_prepare_receipt,
        worker_iam_install_receipt=worker_iam_install_receipt,
        worker_iam_readback_receipt=worker_iam_readback_receipt,
    )
    return {**core, "bundle_sha256": canonical_sha256(core)}


def validate_launch_bundle(
    *,
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    outer_manifest: Mapping[str, Any],
    expected_startup_sha256: str | None = None,
    quota_receipt: Mapping[str, Any],
    persistent_claim_receipt: Mapping[str, Any],
    planned_mapping_receipt: Mapping[str, Any],
    prelaunch_authorization: Mapping[str, Any],
    raw_claim_nonce: str,
    current_time_utc: str,
    runtime_preflight_receipt: Mapping[str, Any],
    runtime_gcp_read_receipt: Mapping[str, Any],
    gcp_read_receipt: Mapping[str, Any],
    service_account_actas_receipt: Mapping[str, Any],
    worker_identity_plan: Mapping[str, Any],
    worker_identity_inventory_receipt: Mapping[str, Any],
    worker_identity_act_as_receipt: Mapping[str, Any],
    project_iam_scan_receipt: Mapping[str, Any],
    worker_iam_plan: Mapping[str, Any],
    worker_iam_prepare_receipt: Mapping[str, Any],
    worker_iam_install_receipt: Mapping[str, Any],
    worker_iam_readback_receipt: Mapping[str, Any],
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild and compare the complete bundle; caller-supplied fields are inert."""

    if not isinstance(value, Mapping):
        raise ValueError("launch bundle must be an object")
    payload = deepcopy(dict(value))
    _exact_keys(payload, _BUNDLE_KEYS, "launch bundle")
    digest = payload.pop("bundle_sha256", None)
    if digest != canonical_sha256(payload):
        raise ValueError("launch bundle digest changed")
    _validate_bundle_surface(
        payload,
        attempt_ledger=attempt_ledger,
        resume_plan=resume_plan,
        outer_manifest=outer_manifest,
        quota_receipt=quota_receipt,
        persistent_claim_receipt=persistent_claim_receipt,
        planned_mapping_receipt=planned_mapping_receipt,
        prelaunch_authorization=prelaunch_authorization,
        current_time_utc=current_time_utc,
        runtime_preflight_receipt=runtime_preflight_receipt,
        runtime_gcp_read_receipt=runtime_gcp_read_receipt,
        gcp_read_receipt=gcp_read_receipt,
        service_account_actas_receipt=service_account_actas_receipt,
        worker_identity_plan=worker_identity_plan,
        worker_identity_inventory_receipt=worker_identity_inventory_receipt,
        worker_identity_act_as_receipt=worker_identity_act_as_receipt,
        project_iam_scan_receipt=project_iam_scan_receipt,
        worker_iam_plan=worker_iam_plan,
        worker_iam_prepare_receipt=worker_iam_prepare_receipt,
        worker_iam_install_receipt=worker_iam_install_receipt,
        worker_iam_readback_receipt=worker_iam_readback_receipt,
    )
    expected = _build_core(
        wave_plan=wave_plan,
        attempt_ledger=attempt_ledger,
        resume_plan=resume_plan,
        outer_manifest=outer_manifest,
        expected_startup_sha256=expected_startup_sha256,
        quota_receipt=quota_receipt,
        persistent_claim_receipt=persistent_claim_receipt,
        planned_mapping_receipt=planned_mapping_receipt,
        prelaunch_authorization=prelaunch_authorization,
        raw_claim_nonce=raw_claim_nonce,
        current_time_utc=current_time_utc,
        runtime_preflight_receipt=runtime_preflight_receipt,
        runtime_gcp_read_receipt=runtime_gcp_read_receipt,
        gcp_read_receipt=gcp_read_receipt,
        service_account_actas_receipt=service_account_actas_receipt,
        worker_identity_plan=worker_identity_plan,
        worker_identity_inventory_receipt=worker_identity_inventory_receipt,
        worker_identity_act_as_receipt=worker_identity_act_as_receipt,
        project_iam_scan_receipt=project_iam_scan_receipt,
        worker_iam_plan=worker_iam_plan,
        worker_iam_prepare_receipt=worker_iam_prepare_receipt,
        worker_iam_install_receipt=worker_iam_install_receipt,
        worker_iam_readback_receipt=worker_iam_readback_receipt,
    )
    if payload != expected:
        raise ValueError("launch bundle no longer derives from validated evidence")
    return {**payload, "bundle_sha256": digest}


__all__ = [
    "EXPECTED_STARTUP_SHA256",
    "IAM_MIN_REMAINING_SECONDS",
    "IAM_PROVISION_UPLOAD_MARGIN_SECONDS",
    "LAUNCH_BUNDLE_SCHEMA",
    "LAUNCH_BUNDLE_STATUS",
    "MAX_IDENTITY_EVIDENCE_AGE_SECONDS",
    "STARTUP_WATCHDOG_SECONDS",
    "build_launch_bundle",
    "canonical_bytes",
    "canonical_sha256",
    "validate_launch_bundle",
]
