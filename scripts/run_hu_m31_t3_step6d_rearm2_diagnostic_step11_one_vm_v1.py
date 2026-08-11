#!/usr/bin/env python3
"""Run the explicitly authorized Step11 fix3 one-VM lifecycle smoke once.

The RSA private key and both OAuth tokens exist only in this process.  This
script cannot launch a second VM or attempt1, and every durable output is
public/audit evidence only.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_10c2_real_readonly_preflight_v1
    as real_preflight,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as transport,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_plan as canary_plan,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_step11_iam_capacity_gate_v1
    as iam_gate,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_cloud_controller_v1
    as cloud,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_controller_v1
    as controller,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_launch_contract_v1
    as launch,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_rest_iam_admin_v1
    as rest_iam,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = (
    REPO_ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m31_t3_step6d"
    / "rearm2_diagnostic_cloud_worker_package_v1"
)
WHEEL_PATH = (
    REPO_ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m31_t3_step6d"
    / "rearm2_diagnostic_10c2_local_preflight_v4"
    / "outer"
    / "wheels"
    / transport.EXPECTED_NUMPY_WHEEL_FILENAME
)
STARTUP_PATH = (
    REPO_ROOT
    / "scripts"
    / "bootstrap_hu_m31_t3_step6d_rearm2_diagnostic_step11_v1.sh"
)
PREBOOTSTRAP_PATH = (
    REPO_ROOT
    / "scripts"
    / "prebootstrap_hu_m31_t3_step6d_rearm2_diagnostic_step11_v1.py"
)
OUTPUT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m31_t3_step6d"
    / "step11_one_vm_v12_fix3_actual"
)
EXPECTED_OUTER_IDENTITY = (
    "593a1f1caaa8710811fc3044c32c66d681ccfe484095b9c94b3c6ba886aa6548"
)
EXPECTED_DIRECT_IDENTITY = (
    "8e27e55dab68ae9697c1c40c80353e8905ad1339423c52d51070f559fbceb44a"
)
EXPECTED_INSTANCE_NAME = "r2d-10c2-s1-candidate-00-a0-0ccd956a"
AUTHORIZATION_EXPIRY = "2026-07-19T07:30:00Z"
INITIATING_PRINCIPAL = "user:giovinco.080807@gmail.com"
EXPECTED_POLICY_REGISTRY_SHA256 = (
    "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
)
POLICY_REGISTRY_PATH = REPO_ROOT / "src" / "ofc_regular" / "ai_profiles.py"
IAM_ARTIFACT_ROOT = (
    REPO_ROOT / "outputs" / "hu_joint_policy" / "m31_t3_step6d"
)
TIME_CONDITION_PATH = IAM_ARTIFACT_ROOT / "step11_v12_fix3_time_condition.json"
INSTANCE_CONDITION_PATH = (
    IAM_ARTIFACT_ROOT / "step11_v12_fix3_instance_condition.json"
)
CONTROLLER_PACKAGE_READ_CONDITION_PATH = (
    IAM_ARTIFACT_ROOT
    / "step11_v12_fix3_controller_package_read_condition.json"
)
CONTROLLER_PACKAGE_CREATE_CONDITION_PATH = (
    IAM_ARTIFACT_ROOT
    / "step11_v12_fix3_controller_package_create_condition.json"
)
WORKER_OBJECT_READ_CONDITION_PATH = (
    IAM_ARTIFACT_ROOT / "step11_v12_fix3_worker_object_read_condition.json"
)
WORKER_RESULT_CREATE_CONDITION_PATH = (
    IAM_ARTIFACT_ROOT / "step11_v12_fix3_worker_result_create_condition.json"
)
NAT_ROUTER_RESOURCE = (
    "projects/ofc-solver-485418/regions/asia-northeast1/routers/"
    "ofc-t3-nat-router-asia-northeast1"
)

_REST_ADMIN: rest_iam.Step11RestIamAdmin | None = None
_ACTIVE_USER_TOKEN: "ActiveUserToken | None" = None
_ACTIVE_USER_CLOUD_CLIENT: "ActiveUserExactCloudClient | None" = None


def _write(path: Path, value: Mapping[str, Any]) -> None:
    controller.exclusive_write_json(path, value)


def _load_condition(path: Path) -> dict[str, str]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"IAM condition must be a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or set(value) not in (
            {"title", "expression"},
            {"title", "expression", "description"},
        )
        or not isinstance(value.get("title"), str)
        or not value["title"]
        or not isinstance(value.get("expression"), str)
        or not value["expression"]
        or (
            "description" in value
            and (
                not isinstance(value["description"], str)
                or not value["description"]
            )
        )
    ):
        raise ValueError(f"IAM condition shape changed: {path}")
    return value


def _validate_fix3_condition_files(
    *,
    contract: Mapping[str, Any],
    gate_plan: Mapping[str, Any],
) -> None:
    checked = transport.validate_job_contract(contract)
    checked_gate = iam_gate.validate_step11_gate_plan(gate_plan)
    worker_bindings = checked_gate["iam_contract"]["worker_bindings"]
    controller_bindings = checked_gate["iam_contract"][
        "controller_bindings"
    ]
    worker_read = worker_bindings[0]["condition"]
    worker_result = worker_bindings[1]["condition"]
    instance_expression = controller_bindings[1]["condition"]["expression"]
    time_expression = controller_bindings[0]["condition"]["expression"]
    package_prefix = checked["remote_layout"]["package_prefix"]
    gs_prefix = f"gs://{transport.BUCKET}/"
    if not package_prefix.startswith(gs_prefix):
        raise ValueError("fix3 package prefix escaped expected bucket")
    package_object_prefix = package_prefix[len(gs_prefix) :]
    package_create_expression = (
        "resource.name.startsWith("
        f"\"projects/_/buckets/{transport.BUCKET}/objects/"
        f"{package_object_prefix}/\") && {time_expression}"
    )
    expected = {
        TIME_CONDITION_PATH: (
            "ofc-m31-step11-time-v12-fix3",
            time_expression,
        ),
        INSTANCE_CONDITION_PATH: (
            "ofc-m31-step11-exact-instance-v12-fix3",
            instance_expression,
        ),
        CONTROLLER_PACKAGE_READ_CONDITION_PATH: (
            "ofc-m31-step11-controller-read-v12-fix3",
            f"{worker_read['expression']} && {time_expression}",
        ),
        CONTROLLER_PACKAGE_CREATE_CONDITION_PATH: (
            "ofc-m31-step11-controller-package-create-v12-fix3",
            package_create_expression,
        ),
        WORKER_OBJECT_READ_CONDITION_PATH: (
            worker_read["title"],
            worker_read["expression"],
        ),
        WORKER_RESULT_CREATE_CONDITION_PATH: (
            worker_result["title"],
            worker_result["expression"],
        ),
    }
    if len(expected) != 6:
        raise AssertionError("fix3 IAM condition paths are not distinct")
    for path, (title, expression) in expected.items():
        condition = _load_condition(path)
        if (
            condition["title"] != title
            or condition["expression"] != expression
        ):
            raise ValueError(f"fix3 IAM condition drifted: {path}")


def _assert_policy_registry_unchanged() -> None:
    if (
        not POLICY_REGISTRY_PATH.is_file()
        or POLICY_REGISTRY_PATH.is_symlink()
        or hashlib.sha256(POLICY_REGISTRY_PATH.read_bytes()).hexdigest()
        != EXPECTED_POLICY_REGISTRY_SHA256
    ):
        raise RuntimeError("current/profile registry hash changed")


def _step11_binding_specs(
    gate_plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    checked = iam_gate.validate_step11_gate_plan(gate_plan)
    project = transport.PROJECT
    controller_member = (
        f"serviceAccount:{controller.CONTROLLER_SERVICE_ACCOUNT}"
    )
    worker_member = f"serviceAccount:{transport.WORKER_SERVICE_ACCOUNT}"
    time_condition = _load_condition(TIME_CONDITION_PATH)
    instance_condition = _load_condition(INSTANCE_CONDITION_PATH)
    controller_read_condition = _load_condition(
        CONTROLLER_PACKAGE_READ_CONDITION_PATH
    )
    controller_create_condition = _load_condition(
        CONTROLLER_PACKAGE_CREATE_CONDITION_PATH
    )
    worker_read_condition = _load_condition(WORKER_OBJECT_READ_CONDITION_PATH)
    worker_result_condition = _load_condition(
        WORKER_RESULT_CREATE_CONDITION_PATH
    )
    custom = checked["iam_contract"]["custom_roles"]
    specs = [
        {
            "purpose": "worker_self_delete",
            "target": rest_iam.PolicyTarget.PROJECT,
            "role": custom["worker_self_delete"]["name"],
            "member": worker_member,
            "condition": instance_condition,
        },
        {
            "purpose": "controller_vm_launch",
            "target": rest_iam.PolicyTarget.PROJECT,
            "role": custom["controller_vm_launch"]["name"],
            "member": controller_member,
            "condition": time_condition,
        },
        {
            "purpose": "controller_instance_lifecycle",
            "target": rest_iam.PolicyTarget.PROJECT,
            "role": custom["controller_instance_lifecycle"]["name"],
            "member": controller_member,
            "condition": instance_condition,
        },
        {
            "purpose": "controller_zone_operation_reader",
            "target": rest_iam.PolicyTarget.PROJECT,
            "role": custom["controller_zone_operation_reader"]["name"],
            "member": controller_member,
            "condition": time_condition,
        },
        {
            "purpose": "controller_service_usage",
            "target": rest_iam.PolicyTarget.PROJECT,
            "role": "roles/serviceusage.serviceUsageConsumer",
            "member": controller_member,
            "condition": time_condition,
        },
        {
            "purpose": "controller_package_and_result_reader",
            "target": rest_iam.PolicyTarget.BUCKET,
            "role": custom["worker_object_reader"]["name"],
            "member": controller_member,
            "condition": controller_read_condition,
        },
        {
            "purpose": "controller_package_creator",
            "target": rest_iam.PolicyTarget.BUCKET,
            "role": custom["worker_result_creator"]["name"],
            "member": controller_member,
            "condition": controller_create_condition,
        },
        {
            "purpose": "worker_package_and_result_reader",
            "target": rest_iam.PolicyTarget.BUCKET,
            "role": custom["worker_object_reader"]["name"],
            "member": worker_member,
            "condition": worker_read_condition,
        },
        {
            "purpose": "worker_result_creator",
            "target": rest_iam.PolicyTarget.BUCKET,
            "role": custom["worker_result_creator"]["name"],
            "member": worker_member,
            "condition": worker_result_condition,
        },
        {
            "purpose": "controller_worker_act_as",
            "target": rest_iam.PolicyTarget.WORKER_SERVICE_ACCOUNT,
            "role": "roles/iam.serviceAccountUser",
            "member": controller_member,
            "condition": time_condition,
        },
        {
            "purpose": "initiator_controller_token_creator",
            "target": rest_iam.PolicyTarget.CONTROLLER_SERVICE_ACCOUNT,
            "role": "roles/iam.serviceAccountTokenCreator",
            "member": INITIATING_PRINCIPAL,
            "condition": time_condition,
        },
    ]
    if (
        len(specs) != 11
        or len({spec["purpose"] for spec in specs}) != len(specs)
        or any(
            spec["role"]
            != (
                f"projects/{project}/roles/"
                + spec["role"].rsplit("/", 1)[1]
            )
            for spec in specs
            if spec["role"].startswith("projects/")
        )
    ):
        raise AssertionError("Step11 authorization binding set changed")
    return specs


def _target_principals(
    target: rest_iam.PolicyTarget,
) -> frozenset[str]:
    controller_member = (
        f"serviceAccount:{controller.CONTROLLER_SERVICE_ACCOUNT}"
    )
    worker_member = f"serviceAccount:{transport.WORKER_SERVICE_ACCOUNT}"
    if target in {
        rest_iam.PolicyTarget.PROJECT,
        rest_iam.PolicyTarget.BUCKET,
    }:
        return frozenset({controller_member, worker_member})
    if target is rest_iam.PolicyTarget.WORKER_SERVICE_ACCOUNT:
        return frozenset({controller_member})
    if target is rest_iam.PolicyTarget.CONTROLLER_SERVICE_ACCOUNT:
        return frozenset({INITIATING_PRINCIPAL})
    raise AssertionError("unknown Step11 IAM target")


def _matching_targeted_bindings(
    policy: Mapping[str, Any],
    *,
    target: rest_iam.PolicyTarget,
) -> list[dict[str, Any]]:
    principals = _target_principals(target)
    matches: list[dict[str, Any]] = []
    for raw in policy.get("bindings", []):
        members = raw.get("members", [])
        if any(member in principals for member in members):
            matches.append(
                {
                    "role": raw.get("role"),
                    "members": sorted(
                        member for member in members if member in principals
                    ),
                    "condition": raw.get("condition"),
                }
            )
    return matches


def _clear_stale_step11_bindings(
    admin: rest_iam.Step11RestIamAdmin,
    gate_plan: Mapping[str, Any],
) -> dict[str, Any]:
    specs = _step11_binding_specs(gate_plan)
    allowed = {
        (spec["target"], spec["role"], spec["member"])
        for spec in specs
    }
    removed: list[dict[str, Any]] = []
    for target in rest_iam.PolicyTarget:
        policy = admin.get_policy(target)
        for raw in list(policy.get("bindings", [])):
            role = raw.get("role")
            condition = raw.get("condition")
            for member in list(raw.get("members", [])):
                if (target, role, member) not in allowed:
                    continue
                result = admin.remove_binding(
                    target,
                    role=role,
                    member=member,
                    condition=condition,
                )
                if result.changed:
                    removed.append(
                        {
                            "target": target.value,
                            "role": role,
                            "member": member,
                            "condition_sha256": controller.canonical_sha256(
                                condition
                            )
                            if condition is not None
                            else None,
                        }
                    )
    residual: dict[str, Any] = {}
    for target in rest_iam.PolicyTarget:
        matches = _matching_targeted_bindings(
            admin.get_policy(target),
            target=target,
        )
        if matches:
            residual[target.value] = matches
    if residual:
        raise RuntimeError(
            "non-Step11 targeted IAM bindings prevent fresh authorization"
        )
    body = {
        "schema": (
            "hu_m31_t3_step6d_rearm2_diagnostic_"
            "step11_targeted_binding_cleanup_v1"
        ),
        "status": "all_step11_target_principals_absent_after_cleanup",
        "removed_bindings": removed,
        "removed_binding_count": len(removed),
        "readback_verified": True,
        "cloud_mutation_performed": bool(removed),
    }
    return {**body, "receipt_sha256": controller.canonical_sha256(body)}


def _install_step11_authorization(
    admin: rest_iam.Step11RestIamAdmin,
    gate_plan: Mapping[str, Any],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for spec in _step11_binding_specs(gate_plan):
        result = admin.add_binding(
            spec["target"],
            role=spec["role"],
            member=spec["member"],
            condition=spec["condition"],
        )
        if result.changed is not True:
            raise RuntimeError("Step11 authorization was not fresh")
        records.append(
            {
                "purpose": spec["purpose"],
                "target": spec["target"].value,
                "role": spec["role"],
                "member": spec["member"],
                "condition_sha256": controller.canonical_sha256(
                    spec["condition"]
                ),
                "created": True,
                "attempts": result.attempts,
            }
        )
    if len(records) != 11 or not all(row["created"] for row in records):
        raise RuntimeError("Step11 authorization binding count changed")
    body = {
        "schema": (
            "hu_m31_t3_step6d_rearm2_diagnostic_"
            "step11_fix3_authorization_v1"
        ),
        "status": "fresh_exact_one_vm_authorization_installed",
        "outer_package_identity_sha256": EXPECTED_OUTER_IDENTITY,
        "direct_stage_identity_sha256": EXPECTED_DIRECT_IDENTITY,
        "instance_name": EXPECTED_INSTANCE_NAME,
        "expires_at": AUTHORIZATION_EXPIRY,
        "binding_count": len(records),
        "bindings": records,
        "vm_limit": 1,
        "attempt_limit": 1,
        "current_profile_changed": False,
        "cloud_mutation_performed": True,
    }
    return {**body, "receipt_sha256": controller.canonical_sha256(body)}


def _gcloud_executable() -> str:
    candidates = ("gcloud.cmd", "gcloud") if os.name == "nt" else ("gcloud",)
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved is not None:
            return resolved
    raise FileNotFoundError("gcloud executable was not found")


def _gcloud_json(arguments: list[str]) -> Any:
    completed = subprocess.run(
        [_gcloud_executable(), *arguments, "--format=json", "--quiet"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("gcloud GET-only evidence is not JSON") from error


def _run_gcloud(arguments: list[str]) -> None:
    subprocess.run(
        [_gcloud_executable(), *arguments, "--quiet"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )


def _run_gcloud_idempotent_remove(arguments: list[str]) -> int:
    completed = subprocess.run(
        [_gcloud_executable(), *arguments, "--quiet"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    return int(completed.returncode)


def _require_rest_admin() -> rest_iam.Step11RestIamAdmin:
    if _REST_ADMIN is None:
        raise RuntimeError("Step11 REST IAM administrator is unavailable")
    return _REST_ADMIN


def _revoke_controller_package_create() -> dict[str, Any]:
    admin = _require_rest_admin()
    member = f"serviceAccount:{controller.CONTROLLER_SERVICE_ACCOUNT}"
    role = f"projects/{transport.PROJECT}/roles/ofcM31T3ResultCreatorV1"
    result = admin.remove_binding(
        rest_iam.PolicyTarget.BUCKET,
        role=role,
        member=member,
        condition=_load_condition(CONTROLLER_PACKAGE_CREATE_CONDITION_PATH),
    )
    if result.changed is not True:
        raise RuntimeError("controller package-create binding was not live")
    policy = admin.get_policy(rest_iam.PolicyTarget.BUCKET)
    if any(
        isinstance(binding, Mapping)
        and binding.get("role") == role
        and member in binding.get("members", [])
        for binding in policy.get("bindings", [])
    ):
        raise RuntimeError("controller package-create binding survived revoke")
    body = {
        "schema": (
            "hu_m31_t3_step6d_rearm2_diagnostic_"
            "step11_package_create_revoke_v1"
        ),
        "status": "controller_package_create_removed_after_readback",
        "member": member,
        "role": role,
        "condition_sha256": controller.sha256_bytes(
            CONTROLLER_PACKAGE_CREATE_CONDITION_PATH.read_bytes()
        ),
        "readback_verified": True,
        "cloud_mutation_performed": True,
    }
    return {**body, "receipt_sha256": controller.canonical_sha256(body)}


def _post_claim_revoke(
    callback_input: Mapping[str, Any],
) -> dict[str, Any]:
    admin = _require_rest_admin()
    controller_member = (
        f"serviceAccount:{controller.CONTROLLER_SERVICE_ACCOUNT}"
    )
    launch_role = (
        f"projects/{transport.PROJECT}/roles/ofcM31T3VmLaunchV1"
    )
    time_condition = _load_condition(TIME_CONDITION_PATH)
    launch_result = admin.remove_binding(
        rest_iam.PolicyTarget.PROJECT,
        role=launch_role,
        member=controller_member,
        condition=time_condition,
    )
    actas_result = admin.remove_binding(
        rest_iam.PolicyTarget.WORKER_SERVICE_ACCOUNT,
        role="roles/iam.serviceAccountUser",
        member=controller_member,
        condition=time_condition,
    )
    if launch_result.changed is not True or actas_result.changed is not True:
        raise RuntimeError("post-claim launch authority was not live")
    project_policy = admin.get_policy(rest_iam.PolicyTarget.PROJECT)
    worker_policy = admin.get_policy(
        rest_iam.PolicyTarget.WORKER_SERVICE_ACCOUNT
    )
    if any(
        binding.get("role") == launch_role
        and controller_member in binding.get("members", [])
        for binding in project_policy.get("bindings", [])
    ) or any(
        binding.get("role") == "roles/iam.serviceAccountUser"
        and controller_member in binding.get("members", [])
        for binding in worker_policy.get("bindings", [])
    ):
        raise RuntimeError("post-claim launch authority survived revoke")
    body = {
        "schema": cloud.POST_CLAIM_REVOKE_SCHEMA,
        "status": "launch_and_worker_actas_removed_after_claim",
        "instance_name": callback_input["instance_name"],
        "provider_instance_id": callback_input["provider_instance_id"],
        "claim_sha256": callback_input["claim_sha256"],
        "launch_binding_removed": True,
        "worker_actas_binding_removed": True,
        "readback_verified": True,
        "cloud_mutation_performed": True,
    }
    receipt = {
        **body,
        "receipt_sha256": cloud.canonical_sha256(body),
    }
    _write(OUTPUT_ROOT / "post_claim_revoke_receipt.json", receipt)
    return receipt


def _legacy_gcloud_final_cleanup_unused() -> dict[str, Any]:
    project = transport.PROJECT
    bucket = f"gs://{transport.BUCKET}"
    controller_email = controller.CONTROLLER_SERVICE_ACCOUNT
    worker_email = transport.WORKER_SERVICE_ACCOUNT
    controller_member = f"serviceAccount:{controller_email}"
    worker_member = f"serviceAccount:{worker_email}"
    owner_member = "user:giovinco.080807@gmail.com"

    instance_inventory = _gcloud_json(
        [
            "compute",
            "instances",
            "list",
            f"--project={project}",
            f"--zones={transport.ZONE}",
            f"--filter=name={EXPECTED_INSTANCE_NAME}",
        ]
    )
    if not isinstance(instance_inventory, list):
        raise RuntimeError("cleanup instance inventory changed")
    if instance_inventory:
        if (
            len(instance_inventory) != 1
            or instance_inventory[0].get("name") != EXPECTED_INSTANCE_NAME
        ):
            raise RuntimeError("cleanup instance inventory escaped exact target")
        _run_gcloud(
            [
                "compute",
                "instances",
                "delete",
                EXPECTED_INSTANCE_NAME,
                f"--project={project}",
                f"--zone={transport.ZONE}",
            ]
        )
    disk_inventory = _gcloud_json(
        [
            "compute",
            "disks",
            "list",
            f"--project={project}",
            f"--zones={transport.ZONE}",
            f"--filter=name={EXPECTED_INSTANCE_NAME}",
        ]
    )
    if not isinstance(disk_inventory, list):
        raise RuntimeError("cleanup disk inventory changed")
    if disk_inventory:
        if (
            len(disk_inventory) != 1
            or disk_inventory[0].get("name") != EXPECTED_INSTANCE_NAME
        ):
            raise RuntimeError("cleanup disk inventory escaped exact target")
        _run_gcloud(
            [
                "compute",
                "disks",
                "delete",
                EXPECTED_INSTANCE_NAME,
                f"--project={project}",
                f"--zone={transport.ZONE}",
            ]
        )

    bucket_removals = [
        (
            controller_member,
            f"projects/{project}/roles/ofcM31T3ObjectReaderV1",
            CONTROLLER_PACKAGE_READ_CONDITION_PATH,
        ),
        (
            controller_member,
            f"projects/{project}/roles/ofcM31T3ResultCreatorV1",
            CONTROLLER_PACKAGE_CREATE_CONDITION_PATH,
        ),
        (
            worker_member,
            f"projects/{project}/roles/ofcM31T3ObjectReaderV1",
            WORKER_OBJECT_READ_CONDITION_PATH,
        ),
        (
            worker_member,
            f"projects/{project}/roles/ofcM31T3ResultCreatorV1",
            WORKER_RESULT_CREATE_CONDITION_PATH,
        ),
    ]
    for member, role, condition_path in bucket_removals:
        _run_gcloud_idempotent_remove(
            [
                "storage",
                "buckets",
                "remove-iam-policy-binding",
                bucket,
                f"--member={member}",
                f"--role={role}",
                "--condition-from-file=" + str(condition_path),
            ]
        )

    project_removals = [
        (
            worker_member,
            f"projects/{project}/roles/ofcM31T3SelfDeleteV1",
            INSTANCE_CONDITION_PATH,
        ),
        (
            controller_member,
            f"projects/{project}/roles/ofcM31T3VmLaunchV1",
            TIME_CONDITION_PATH,
        ),
        (
            controller_member,
            f"projects/{project}/roles/ofcM31T3VmLifecycleV1",
            INSTANCE_CONDITION_PATH,
        ),
        (
            controller_member,
            f"projects/{project}/roles/ofcM31T3ZoneOperationReaderV1",
            TIME_CONDITION_PATH,
        ),
        (
            controller_member,
            "roles/serviceusage.serviceUsageConsumer",
            TIME_CONDITION_PATH,
        ),
    ]
    for member, role, condition_path in project_removals:
        _run_gcloud_idempotent_remove(
            [
                "projects",
                "remove-iam-policy-binding",
                project,
                f"--member={member}",
                f"--role={role}",
                "--condition-from-file=" + str(condition_path),
            ]
        )
    _run_gcloud_idempotent_remove(
        [
            "iam",
            "service-accounts",
            "remove-iam-policy-binding",
            worker_email,
            f"--project={project}",
            f"--member={controller_member}",
            "--role=roles/iam.serviceAccountUser",
            "--condition-from-file=" + str(TIME_CONDITION_PATH),
        ]
    )
    _run_gcloud_idempotent_remove(
        [
            "iam",
            "service-accounts",
            "remove-iam-policy-binding",
            controller_email,
            f"--project={project}",
            f"--member={owner_member}",
            "--role=roles/iam.serviceAccountTokenCreator",
            "--condition-from-file=" + str(TIME_CONDITION_PATH),
        ]
    )

    final_instances = _gcloud_json(
        [
            "compute",
            "instances",
            "list",
            f"--project={project}",
            f"--zones={transport.ZONE}",
            f"--filter=name={EXPECTED_INSTANCE_NAME}",
        ]
    )
    final_disks = _gcloud_json(
        [
            "compute",
            "disks",
            "list",
            f"--project={project}",
            f"--zones={transport.ZONE}",
            f"--filter=name={EXPECTED_INSTANCE_NAME}",
        ]
    )
    project_policy = _gcloud_json(
        ["projects", "get-iam-policy", project]
    )
    bucket_policy = _gcloud_json(
        ["storage", "buckets", "get-iam-policy", bucket]
    )
    worker_policy = _gcloud_json(
        [
            "iam",
            "service-accounts",
            "get-iam-policy",
            worker_email,
            f"--project={project}",
        ]
    )
    controller_policy = _gcloud_json(
        [
            "iam",
            "service-accounts",
            "get-iam-policy",
            controller_email,
            f"--project={project}",
        ]
    )
    project_targeted = [
        binding
        for binding in project_policy.get("bindings", [])
        if any(
            member in binding.get("members", [])
            for member in (controller_member, worker_member)
        )
    ]
    bucket_targeted = [
        binding
        for binding in bucket_policy.get("bindings", [])
        if any(
            member in binding.get("members", [])
            for member in (controller_member, worker_member)
        )
    ]
    worker_targeted = [
        binding
        for binding in worker_policy.get("bindings", [])
        if controller_member in binding.get("members", [])
    ]
    controller_targeted = [
        binding
        for binding in controller_policy.get("bindings", [])
        if owner_member in binding.get("members", [])
    ]
    if (
        final_instances != []
        or final_disks != []
        or project_targeted
        or bucket_targeted
        or worker_targeted
        or controller_targeted
    ):
        raise RuntimeError("Step11 final cleanup readback did not converge")
    body = {
        "schema": (
            "hu_m31_t3_step6d_rearm2_diagnostic_"
            "step11_final_cloud_cleanup_v1"
        ),
        "status": "exact_vm_disk_and_all_step11_bindings_absent",
        "instance_name": EXPECTED_INSTANCE_NAME,
        "zone": transport.ZONE,
        "instance_inventory": final_instances,
        "disk_inventory": final_disks,
        "project_targeted_bindings": project_targeted,
        "bucket_targeted_bindings": bucket_targeted,
        "worker_service_account_targeted_bindings": worker_targeted,
        "controller_service_account_targeted_bindings": controller_targeted,
        "custom_roles_retained_unbound": True,
        "package_and_results_retained": True,
        "current_profile_changed": False,
        "cloud_mutation_performed": True,
    }
    receipt = {
        **body,
        "receipt_sha256": cloud.canonical_sha256(body),
    }
    _write(OUTPUT_ROOT / "final_cloud_cleanup_receipt.json", receipt)
    return receipt


def _collect_live_iam_evidence(
    gate_plan: Mapping[str, Any],
) -> dict[str, Any]:
    admin = _require_rest_admin()
    if _ACTIVE_USER_TOKEN is None:
        raise RuntimeError("active-user token is unavailable")
    project = transport.PROJECT
    controller_email = controller.CONTROLLER_SERVICE_ACCOUNT
    worker_email = transport.WORKER_SERVICE_ACCOUNT
    project_policy = admin.get_policy(rest_iam.PolicyTarget.PROJECT)
    bucket_policy = admin.get_policy(rest_iam.PolicyTarget.BUCKET)
    worker_policy = admin.get_policy(
        rest_iam.PolicyTarget.WORKER_SERVICE_ACCOUNT
    )
    controller_policy = admin.get_policy(
        rest_iam.PolicyTarget.CONTROLLER_SERVICE_ACCOUNT
    )
    evidence_client = ActiveUserEvidenceClient(_ACTIVE_USER_TOKEN)
    roles: dict[str, Any] = {}
    for key, expected in sorted(
        gate_plan["iam_contract"]["custom_roles"].items()
    ):
        role_id = expected["name"].rsplit("/", 1)[1]
        roles[key] = evidence_client.get_json(
            "https://iam.googleapis.com/v1/projects/"
            f"{project}/roles/{urllib.parse.quote(role_id, safe='')}"
        )
    raw_services: list[Mapping[str, Any]] = []
    seen_page_tokens: set[str] = set()
    page_token: str | None = None
    for _ in range(10):
        query = {
            "filter": "state:ENABLED",
            "pageSize": "200",
        }
        if page_token is not None:
            query["pageToken"] = page_token
        service_page = evidence_client.get_json(
            "https://serviceusage.googleapis.com/v1/projects/"
            f"{project}/services?{urllib.parse.urlencode(query)}"
        )
        page_rows = service_page.get("services", [])
        if not isinstance(page_rows, list) or any(
            not isinstance(row, Mapping) for row in page_rows
        ):
            raise RuntimeError("enabled service inventory changed")
        raw_services.extend(page_rows)
        next_token = service_page.get("nextPageToken")
        if next_token is None:
            break
        if (
            not isinstance(next_token, str)
            or not next_token
            or next_token in seen_page_tokens
        ):
            raise RuntimeError("enabled service pagination changed")
        seen_page_tokens.add(next_token)
        page_token = next_token
    else:
        raise RuntimeError("enabled service pagination exceeded bound")
    enabled_services = sorted(
        {
            row.get("config", {}).get("name")
            for row in raw_services
            if isinstance(row, Mapping)
            and row.get("state") == "ENABLED"
            and isinstance(row.get("config"), Mapping)
            and isinstance(row["config"].get("name"), str)
        }
    )
    return cloud.seal_shared_project_live_evidence(
        {
            "schema": cloud.SHARED_PROJECT_LIVE_EVIDENCE_SCHEMA,
            "collected_at_unix_seconds": int(time.time()),
            "collected_via_get_only": True,
            "cloud_mutation_performed": False,
            "project": project,
            "bucket": transport.BUCKET,
            "worker_service_account": worker_email,
            "controller_service_account": controller_email,
            "initiating_principal": INITIATING_PRINCIPAL,
            "project_policy": project_policy,
            "bucket_policy": bucket_policy,
            "worker_service_account_policy": worker_policy,
            "controller_service_account_policy": controller_policy,
            "custom_roles": roles,
            "enabled_services": enabled_services,
        }
    )


class CachedControllerToken:
    controller_principal = controller.CONTROLLER_SERVICE_ACCOUNT

    def __init__(self, admin: rest_iam.Step11RestIamAdmin) -> None:
        self._admin = admin
        self._token: rest_iam.GeneratedControllerAccessToken | None = None
        self._refresh_at_monotonic = 0.0

    def access_token(self) -> str:
        if (
            self._token is None
            or time.monotonic() >= self._refresh_at_monotonic
        ):
            self._token = self._admin.generate_controller_access_token(
                lifetime_seconds=3_600
            )
            # The requested token lifetime is one hour. Refresh early so the
            # bounded cleanup path still has a valid controller credential.
            self._refresh_at_monotonic = time.monotonic() + 2_700
        return self._token.access_token()


def _wait_for_controller_token(
    source: CachedControllerToken,
) -> None:
    for attempt in range(1, 9):
        try:
            source.access_token()
            return
        except rest_iam.RestIamAdminError as error:
            if error.status_code != 403 or attempt == 8:
                raise
            time.sleep(min(8, 2 ** (attempt - 1)))
    raise AssertionError("controller token propagation loop escaped")


class PackagePropagationRetryClient:
    """Retry only pre-VM GCS 403 responses using the identical request."""

    def __init__(self, inner: controller.GoogleJsonClient) -> None:
        self._inner = inner

    def request(
        self,
        *,
        method: str,
        url: str,
        body: bytes | None = None,
        content_type: str | None = None,
        timeout_seconds: int = 60,
    ) -> controller.HttpResponse:
        if not url.startswith("https://storage.googleapis.com/"):
            raise ValueError("package propagation retry escaped GCS")
        for attempt in range(1, 9):
            response = self._inner.request(
                method=method,
                url=url,
                body=body,
                content_type=content_type,
                timeout_seconds=timeout_seconds,
            )
            if response.status != 403 or attempt == 8:
                return response
            time.sleep(min(8, 2 ** (attempt - 1)))
        raise AssertionError("package propagation retry loop escaped")


class ActiveUserToken:
    def __init__(self) -> None:
        self._token: str | None = None
        self._refresh_at_monotonic = 0.0

    def access_token(self) -> str:
        if (
            self._token is None
            or time.monotonic() >= self._refresh_at_monotonic
        ):
            completed = subprocess.run(
                [
                    _gcloud_executable(),
                    "auth",
                    "print-access-token",
                    "--quiet",
                ],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=60,
            )
            token = completed.stdout.strip()
            if (
                not token
                or len(token) > 16_384
                or any(character.isspace() for character in token)
            ):
                raise RuntimeError("collector token shape changed")
            self._token = token
            self._refresh_at_monotonic = time.monotonic() + 2_700
        return self._token


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Mapping[str, str],
        newurl: str,
    ) -> None:
        return None


class ActiveUserExactCloudClient:
    """User-token client restricted to the exact Step11 VM, disk, and ops."""

    def __init__(self, source: ActiveUserToken) -> None:
        self._source = source
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect()
        )

    @staticmethod
    def _allowed(method: str, url: str) -> bool:
        parsed = urllib.parse.urlsplit(url)
        root = (
            f"/compute/v1/projects/{transport.PROJECT}/zones/"
            f"{transport.ZONE}/"
        )
        exact_resources = {
            root + f"instances/{EXPECTED_INSTANCE_NAME}",
            root + f"disks/{EXPECTED_INSTANCE_NAME}",
        }
        if (
            parsed.scheme != "https"
            or parsed.netloc != "compute.googleapis.com"
            or parsed.fragment
        ):
            return False
        if parsed.path in exact_resources:
            if method == "GET":
                return not parsed.query
            if method == "DELETE":
                query = urllib.parse.parse_qs(
                    parsed.query,
                    keep_blank_values=True,
                    strict_parsing=True,
                )
                return (
                    set(query) == {"requestId"}
                    and len(query["requestId"]) == 1
                    and re.fullmatch(
                        r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
                        r"[89ab][0-9a-f]{3}-[0-9a-f]{12}",
                        query["requestId"][0],
                    )
                    is not None
                )
            return False
        operation_prefix = root + "operations/"
        return (
            method == "GET"
            and not parsed.query
            and parsed.path.startswith(operation_prefix)
            and re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}",
                parsed.path[len(operation_prefix) :],
            )
            is not None
        )

    def request(
        self,
        *,
        method: str,
        url: str,
        body: bytes | None = None,
        content_type: str | None = None,
        timeout_seconds: int = 60,
    ) -> controller.HttpResponse:
        if (
            not self._allowed(method, url)
            or body is not None
            or content_type is not None
            or type(timeout_seconds) is not int
            or not 1 <= timeout_seconds <= 120
        ):
            raise ValueError("active-user compute request escaped exact surface")
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self._source.access_token()}"
            },
            method=method,
        )
        try:
            with self._opener.open(
                request, timeout=timeout_seconds
            ) as response:
                return controller.HttpResponse(
                    status=int(response.status),
                    headers=dict(response.headers.items()),
                    body=response.read(4 * 1024 * 1024 + 1),
                )
        except urllib.error.HTTPError as error:
            return controller.HttpResponse(
                status=int(error.code),
                headers=(
                    dict(error.headers.items()) if error.headers else {}
                ),
                body=error.read(4 * 1024 * 1024 + 1),
            )


class ActiveUserEvidenceClient:
    """GET-only client for the exact custom-role and enabled-service evidence."""

    def __init__(self, source: ActiveUserToken) -> None:
        self._source = source
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect()
        )

    @staticmethod
    def _allowed(url: str) -> bool:
        parsed = urllib.parse.urlsplit(url)
        role_prefix = f"/v1/projects/{transport.PROJECT}/roles/"
        service_path = f"/v1/projects/{transport.PROJECT}/services"
        if parsed.scheme != "https" or parsed.fragment:
            return False
        if (
            parsed.netloc == "iam.googleapis.com"
            and not parsed.query
            and parsed.path.startswith(role_prefix)
            and re.fullmatch(
                r"[A-Za-z0-9_.]{1,64}",
                parsed.path[len(role_prefix) :],
            )
            is not None
        ):
            return True
        if (
            parsed.netloc != "serviceusage.googleapis.com"
            or parsed.path != service_path
        ):
            return False
        query = urllib.parse.parse_qs(
            parsed.query,
            keep_blank_values=True,
            strict_parsing=True,
        )
        if set(query) not in ({"filter", "pageSize"}, {"filter", "pageSize", "pageToken"}):
            return False
        return (
            query.get("filter") == ["state:ENABLED"]
            and query.get("pageSize") == ["200"]
            and (
                "pageToken" not in query
                or (
                    len(query["pageToken"]) == 1
                    and re.fullmatch(
                        r"[A-Za-z0-9._~+/=-]{1,2048}",
                        query["pageToken"][0],
                    )
                    is not None
                )
            )
        )

    def get_json(self, url: str) -> Mapping[str, Any]:
        if not self._allowed(url):
            raise ValueError("live-evidence request escaped GET-only surface")
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._source.access_token()}",
            },
            method="GET",
        )
        try:
            with self._opener.open(request, timeout=60) as response:
                status = int(response.status)
                raw = response.read(4 * 1024 * 1024 + 1)
        except urllib.error.HTTPError as error:
            status = int(error.code)
            raw = error.read(4 * 1024 * 1024 + 1)
        if status != 200 or len(raw) > 4 * 1024 * 1024:
            raise RuntimeError(
                f"live-evidence GET returned HTTP {status}"
            )
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise RuntimeError("live-evidence response is not JSON") from None
        if not isinstance(value, Mapping):
            raise RuntimeError("live-evidence response shape changed")
        return value


class ActiveUserReadOnlyClient:
    def __init__(self, source: ActiveUserToken) -> None:
        self._source = source
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect()
        )

    def request(
        self,
        *,
        method: str,
        url: str,
        body: bytes | None = None,
        content_type: str | None = None,
        timeout_seconds: int = 60,
    ) -> controller.HttpResponse:
        if (
            method != "GET"
            or body is not None
            or content_type is not None
            or not url.startswith("https://storage.googleapis.com/")
        ):
            raise ValueError("collector escaped GET-only GCS surface")
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self._source.access_token()}"
            },
            method="GET",
        )
        try:
            with self._opener.open(
                request, timeout=timeout_seconds
            ) as response:
                return controller.HttpResponse(
                    status=int(response.status),
                    headers=dict(response.headers.items()),
                    body=response.read(),
                )
        except urllib.error.HTTPError as error:
            return controller.HttpResponse(
                status=int(error.code),
                headers=(
                    dict(error.headers.items()) if error.headers else {}
                ),
                body=error.read(),
            )


def _exact_compute_url(resource_kind: str) -> str:
    if resource_kind not in {"instances", "disks"}:
        raise ValueError("cleanup resource kind changed")
    return (
        "https://compute.googleapis.com/compute/v1/projects/"
        f"{transport.PROJECT}/zones/{transport.ZONE}/"
        f"{resource_kind}/{EXPECTED_INSTANCE_NAME}"
    )


def _validate_exact_compute_resource(
    response: controller.HttpResponse,
    *,
    resource_kind: str,
) -> Mapping[str, Any]:
    if response.status != 200 or len(response.body) > 4 * 1024 * 1024:
        raise RuntimeError("exact compute resource read failed")
    try:
        value = json.loads(response.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError("exact compute resource response changed") from None
    expected_url = _exact_compute_url(resource_kind)
    expected_path = urllib.parse.urlsplit(expected_url).path
    self_link = value.get("selfLink") if isinstance(value, Mapping) else None
    parsed_link = (
        urllib.parse.urlsplit(self_link)
        if isinstance(self_link, str)
        else None
    )
    if (
        not isinstance(value, Mapping)
        or value.get("name") != EXPECTED_INSTANCE_NAME
        or parsed_link is None
        or parsed_link.scheme != "https"
        or parsed_link.netloc
        not in {"compute.googleapis.com", "www.googleapis.com"}
        or parsed_link.path != expected_path
        or parsed_link.query
        or parsed_link.fragment
    ):
        raise RuntimeError("exact compute resource identity changed")
    return value


def _wait_exact_compute_absence(
    client: ActiveUserExactCloudClient,
    *,
    resource_kind: str,
    timeout_seconds: int = 300,
) -> int:
    deadline = time.monotonic() + timeout_seconds
    url = _exact_compute_url(resource_kind)
    while True:
        response = client.request(method="GET", url=url)
        if response.status == 404:
            return 404
        if response.status != 200:
            raise RuntimeError(
                f"exact {resource_kind} absence GET returned "
                f"HTTP {response.status}"
            )
        _validate_exact_compute_resource(
            response,
            resource_kind=resource_kind,
        )
        if time.monotonic() >= deadline:
            raise TimeoutError(f"exact {resource_kind} deletion timed out")
        time.sleep(2)


def _delete_exact_disk_if_present(
    client: ActiveUserExactCloudClient,
) -> dict[str, Any]:
    url = _exact_compute_url("disks")
    initial = client.request(method="GET", url=url)
    if initial.status == 404:
        return {
            "disk_name": EXPECTED_INSTANCE_NAME,
            "delete_requested": False,
            "already_absent": True,
            "provider_get_status": 404,
        }
    _validate_exact_compute_resource(initial, resource_kind="disks")
    response = client.request(
        method="DELETE",
        url=f"{url}?requestId={uuid.uuid4()}",
    )
    if response.status == 404:
        return {
            "disk_name": EXPECTED_INSTANCE_NAME,
            "delete_requested": False,
            "already_absent": True,
            "provider_get_status": 404,
        }
    if response.status not in {200, 202}:
        raise RuntimeError(
            f"exact disk delete returned HTTP {response.status}"
        )
    try:
        operation_value = json.loads(response.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError("exact disk delete operation changed") from None
    if not isinstance(operation_value, Mapping):
        raise RuntimeError("exact disk delete operation changed")
    operation = cloud.wait_zone_operation(
        client=client,
        initial=operation_value,
        expected_instance_url=url,
    )
    status = _wait_exact_compute_absence(
        client,
        resource_kind="disks",
    )
    return {
        "disk_name": EXPECTED_INSTANCE_NAME,
        "delete_requested": True,
        "already_absent": False,
        "operation": operation,
        "provider_get_status": status,
    }


def _final_cleanup() -> dict[str, Any]:
    cleanup_errors: list[BaseException] = []
    try:
        _assert_policy_registry_unchanged()
    except BaseException as error:
        cleanup_errors.append(error)
    if _REST_ADMIN is None or _ACTIVE_USER_CLOUD_CLIENT is None:
        if cleanup_errors:
            raise BaseExceptionGroup(
                "Step11 pre-cloud cleanup invariant failed",
                cleanup_errors,
            )
        body = {
            "schema": (
                "hu_m31_t3_step6d_rearm2_diagnostic_"
                "step11_final_cloud_cleanup_v1"
            ),
            "status": "no_cloud_administrator_created_before_failure",
            "instance_name": EXPECTED_INSTANCE_NAME,
            "zone": transport.ZONE,
            "instance_inventory": [],
            "disk_inventory": [],
            "project_targeted_bindings": [],
            "bucket_targeted_bindings": [],
            "worker_service_account_targeted_bindings": [],
            "controller_service_account_targeted_bindings": [],
            "custom_roles_retained_unbound": True,
            "package_and_results_retained": True,
            "current_profile_changed": False,
            "cloud_mutation_performed": False,
        }
        return {**body, "receipt_sha256": cloud.canonical_sha256(body)}

    instance_cleanup: Mapping[str, Any] | None = None
    disk_cleanup: Mapping[str, Any] | None = None
    binding_cleanup: Mapping[str, Any] | None = None
    targeted: dict[str, list[dict[str, Any]]] = {}
    final_instance_status: int | None = None
    final_disk_status: int | None = None

    # Revoke authority first. Compute cleanup uses the initiating user and does
    # not need any of the temporary controller/worker bindings.
    try:
        final_gate_plan = json.loads(
            (OUTPUT_ROOT / "iam_gate_plan.json").read_text(encoding="utf-8")
        )
        binding_cleanup = _clear_stale_step11_bindings(
            _REST_ADMIN,
            final_gate_plan,
        )
    except BaseException as error:
        cleanup_errors.append(error)

    try:
        initial_instance = _ACTIVE_USER_CLOUD_CLIENT.request(
            method="GET",
            url=_exact_compute_url("instances"),
        )
        if initial_instance.status == 404:
            instance_cleanup = {
                "instance_name": EXPECTED_INSTANCE_NAME,
                "delete_requested": False,
                "already_absent": True,
                "provider_get_status": 404,
            }
        elif initial_instance.status == 200:
            _validate_exact_compute_resource(
                initial_instance,
                resource_kind="instances",
            )
            instance_cleanup = cloud.delete_exact_instance(
                client=_ACTIVE_USER_CLOUD_CLIENT,
                instance_name=EXPECTED_INSTANCE_NAME,
            )
        else:
            raise RuntimeError(
                "exact instance cleanup GET returned "
                f"HTTP {initial_instance.status}"
            )
    except BaseException as error:
        cleanup_errors.append(error)

    try:
        disk_cleanup = _delete_exact_disk_if_present(
            _ACTIVE_USER_CLOUD_CLIENT
        )
    except BaseException as error:
        cleanup_errors.append(error)

    try:
        for target in rest_iam.PolicyTarget:
            targeted[target.value] = _matching_targeted_bindings(
                _REST_ADMIN.get_policy(target),
                target=target,
            )
        if any(targeted.values()):
            raise RuntimeError("Step11 IAM cleanup readback did not converge")
    except BaseException as error:
        cleanup_errors.append(error)

    try:
        final_instance_status = _wait_exact_compute_absence(
            _ACTIVE_USER_CLOUD_CLIENT,
            resource_kind="instances",
        )
        if final_instance_status != 404:
            raise RuntimeError("Step11 instance cleanup did not converge")
    except BaseException as error:
        cleanup_errors.append(error)
    try:
        final_disk_status = _wait_exact_compute_absence(
            _ACTIVE_USER_CLOUD_CLIENT,
            resource_kind="disks",
        )
        if final_disk_status != 404:
            raise RuntimeError("Step11 disk cleanup did not converge")
    except BaseException as error:
        cleanup_errors.append(error)
    try:
        _assert_policy_registry_unchanged()
    except BaseException as error:
        cleanup_errors.append(error)

    if cleanup_errors:
        raise BaseExceptionGroup(
            "Step11 final cleanup did not fully converge",
            cleanup_errors,
        )
    if (
        instance_cleanup is None
        or disk_cleanup is None
        or binding_cleanup is None
        or set(targeted)
        != {target.value for target in rest_iam.PolicyTarget}
    ):
        raise AssertionError("Step11 cleanup evidence is incomplete")
    body = {
        "schema": (
            "hu_m31_t3_step6d_rearm2_diagnostic_"
            "step11_final_cloud_cleanup_v1"
        ),
        "status": "exact_vm_disk_and_all_step11_bindings_absent",
        "instance_name": EXPECTED_INSTANCE_NAME,
        "zone": transport.ZONE,
        "instance_inventory": [],
        "disk_inventory": [],
        "instance_cleanup": instance_cleanup,
        "disk_cleanup": disk_cleanup,
        "binding_cleanup_receipt_sha256": binding_cleanup["receipt_sha256"],
        "project_targeted_bindings": targeted[
            rest_iam.PolicyTarget.PROJECT.value
        ],
        "bucket_targeted_bindings": targeted[
            rest_iam.PolicyTarget.BUCKET.value
        ],
        "worker_service_account_targeted_bindings": targeted[
            rest_iam.PolicyTarget.WORKER_SERVICE_ACCOUNT.value
        ],
        "controller_service_account_targeted_bindings": targeted[
            rest_iam.PolicyTarget.CONTROLLER_SERVICE_ACCOUNT.value
        ],
        "custom_roles_retained_unbound": True,
        "package_and_results_retained": True,
        "current_profile_changed": False,
        "cloud_mutation_performed": True,
    }
    receipt = {
        **body,
        "receipt_sha256": cloud.canonical_sha256(body),
    }
    _write(OUTPUT_ROOT / "final_cloud_cleanup_receipt.json", receipt)
    return receipt


def main() -> int:
    global _REST_ADMIN, _ACTIVE_USER_TOKEN, _ACTIVE_USER_CLOUD_CLIENT

    _assert_policy_registry_unchanged()
    now_seconds = int(time.time())
    expiry_seconds = int(
        datetime.strptime(
            AUTHORIZATION_EXPIRY, "%Y-%m-%dT%H:%M:%SZ"
        )
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )
    if expiry_seconds - now_seconds < cloud.MIN_EXECUTION_REMAINING_SECONDS:
        raise RuntimeError(
            "Step11 authorization lacks the full runtime and cleanup margin"
        )
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise FileExistsError(f"Step11 output root is not fresh: {OUTPUT_ROOT}")
    OUTPUT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir()

    signer = controller.generate_ephemeral_controller_key(key_size=3_072)
    wheel_record = transport.build_offline_wheel_record(WHEEL_PATH)
    direct_contract = transport.build_job_contract(
        package_dir=PACKAGE_DIR,
        stage_id=canary_plan.STAGE1_ID,
        job_id=canary_plan.STAGE1_JOB_IDS[0],
        offline_wheel_record=wheel_record,
        controller_public_key_record=signer.public_record,
    )
    if (
        direct_contract["outer_package_manifest"][
            "outer_package_identity_sha256"
        ]
        != EXPECTED_OUTER_IDENTITY
        or direct_contract["direct_stage_identity_sha256"]
        != EXPECTED_DIRECT_IDENTITY
        or direct_contract["metadata_binding"]["instance_name"]
        != EXPECTED_INSTANCE_NAME
    ):
        raise RuntimeError("final fix3 cloud identity changed")
    _write(OUTPUT_ROOT / "transport_contract.json", direct_contract)
    _write(OUTPUT_ROOT / "controller_public_key.json", signer.public_record)

    local_root = OUTPUT_ROOT / "local_preflight"
    local_receipt = transport.local_preflight(
        contract=direct_contract,
        package_mirror=PACKAGE_DIR,
        fresh_work_root=local_root,
        offline_wheel_mirror=WHEEL_PATH,
        offline_install_smoke=False,
    )
    _write(OUTPUT_ROOT / "local_preflight_receipt.json", local_receipt)
    outer_root = local_root / "outer"

    launch_contract = launch.build_launch_contract(
        transport_contract=direct_contract,
        controller_public_key_record=signer.public_record,
        prebootstrap_path=PREBOOTSTRAP_PATH,
        startup_path=STARTUP_PATH,
    )
    _write(OUTPUT_ROOT / "launch_contract.json", launch_contract)

    now_seconds = int(time.time())
    gate_plan = iam_gate.build_step11_gate_plan(
        direct_contract,
        issued_at_unix_seconds=now_seconds,
        expires_at_unix_seconds=expiry_seconds,
        nat_router_resource=NAT_ROUTER_RESOURCE,
    )
    _validate_fix3_condition_files(
        contract=direct_contract,
        gate_plan=gate_plan,
    )
    _write(OUTPUT_ROOT / "iam_gate_plan.json", gate_plan)

    user_token = ActiveUserToken()
    user_token.access_token()
    admin = rest_iam.Step11RestIamAdmin(
        http_client=rest_iam.StdlibJsonHttpsClient(),
        user_token_source=user_token,
        project=transport.PROJECT,
        bucket=transport.BUCKET,
        worker_service_account=transport.WORKER_SERVICE_ACCOUNT,
        controller_service_account=controller.CONTROLLER_SERVICE_ACCOUNT,
    )
    user_cloud_client = ActiveUserExactCloudClient(user_token)
    _ACTIVE_USER_TOKEN = user_token
    _REST_ADMIN = admin
    _ACTIVE_USER_CLOUD_CLIENT = user_cloud_client
    prelaunch_status: dict[str, int] = {}
    for resource_kind in ("instances", "disks"):
        response = user_cloud_client.request(
            method="GET",
            url=_exact_compute_url(resource_kind),
        )
        prelaunch_status[resource_kind] = response.status
        if response.status != 404:
            if response.status == 200:
                _validate_exact_compute_resource(
                    response,
                    resource_kind=resource_kind,
                )
            raise RuntimeError(
                f"fresh Step11 exact {resource_kind} target is not absent"
            )
    cloud_empty_body = {
        "schema": (
            "hu_m31_t3_step6d_rearm2_diagnostic_"
            "step11_prelaunch_cloud_empty_v1"
        ),
        "status": "exact_instance_and_disk_absent_before_authorization",
        "instance_name": EXPECTED_INSTANCE_NAME,
        "zone": transport.ZONE,
        "provider_status": prelaunch_status,
        "provider_get_404_observed": all(
            status == 404 for status in prelaunch_status.values()
        ),
        "cloud_mutation_performed": False,
    }
    _write(
        OUTPUT_ROOT / "prelaunch_cloud_empty_receipt.json",
        {
            **cloud_empty_body,
            "receipt_sha256": cloud.canonical_sha256(cloud_empty_body),
        },
    )
    stale_cleanup = _clear_stale_step11_bindings(admin, gate_plan)
    _write(
        OUTPUT_ROOT / "stale_binding_cleanup_receipt.json",
        stale_cleanup,
    )
    authorization = _install_step11_authorization(admin, gate_plan)
    _write(OUTPUT_ROOT / "authorization_receipt.json", authorization)

    controller_token = CachedControllerToken(admin)
    _wait_for_controller_token(controller_token)
    controller_client = controller.GoogleJsonClient(controller_token)
    package_client = PackagePropagationRetryClient(controller_client)
    package_plan = controller.build_package_provision_plan(
        contract=direct_contract, outer_root=outer_root
    )
    _write(OUTPUT_ROOT / "package_provision_plan.json", package_plan)
    package_receipt = controller.provision_package(
        client=package_client,
        plan=package_plan,
        outer_root=outer_root,
    )
    if (
        len(package_receipt["records"]) != package_plan["object_count"]
        or not all(
            record["created"] is True
            for record in package_receipt["records"]
        )
    ):
        raise FileExistsError("fix3 package prefix was not exactly fresh")
    _write(OUTPUT_ROOT / "package_provision_receipt.json", package_receipt)
    package_create_revoke = _revoke_controller_package_create()
    _write(
        OUTPUT_ROOT / "package_create_revoke_receipt.json",
        package_create_revoke,
    )
    package_create_body = {
        "schema": (
            "hu_m31_t3_step6d_rearm2_diagnostic_"
            "step11_fix3_package_provision_v1"
        ),
        "status": "fresh_fix3_package_created_and_generation_pinned",
        "package_receipt_sha256": package_receipt["receipt_sha256"],
        "contract_sha256": package_receipt["contract_sha256"],
        "package_generations_sha256": package_receipt[
            "package_generations_sha256"
        ],
        "package_object_count": package_plan["object_count"],
        "all_objects_created": all(
            record["created"] for record in package_receipt["records"]
        ),
        "package_create_binding_revoked": True,
        "fresh_live_readback_required_before_insert": True,
        "vm_created": False,
        "cloud_mutation_performed": True,
    }
    _write(
        OUTPUT_ROOT / "package_create_receipt.json",
        {
            **package_create_body,
            "receipt_sha256": controller.canonical_sha256(
                package_create_body
            ),
        },
    )

    preflight = real_preflight.run_actual_read_only_preflight(
        package_dir=PACKAGE_DIR,
        pinned_wheel_path=WHEEL_PATH,
        stage_id=canary_plan.STAGE1_ID,
        output_path=OUTPUT_ROOT / "actual_readonly_preflight.json",
        token_source=user_token,
        transport=real_preflight.StdlibGoogleJsonReadOnlyTransport(),
        expected_worker_storage_bindings=gate_plan["iam_contract"][
            "worker_bindings"
        ][:2],
    )
    if preflight["status"] != (
        "pass_read_only_preflight_launch_still_unauthorized"
    ):
        raise RuntimeError("fresh post-provision GET-only preflight did not pass")
    package_preflight_body = {
        "schema": (
            "hu_m31_t3_step6d_rearm2_diagnostic_"
            "step11_fix3_package_preflight_v1"
        ),
        "status": "fresh_fix3_package_live_readback_passed",
        "source_receipt_sha256": package_receipt["receipt_sha256"],
        "actual_preflight_artifact_sha256": preflight["artifact_sha256"],
        "package_create_binding_required": False,
        "vm_created": False,
        "cloud_mutation_performed": True,
    }
    _write(
        OUTPUT_ROOT / "package_preflight_receipt.json",
        {
            **package_preflight_body,
            "receipt_sha256": controller.canonical_sha256(
                package_preflight_body
            ),
        },
    )

    live_iam_evidence = _collect_live_iam_evidence(gate_plan)
    _write(OUTPUT_ROOT / "live_iam_evidence.json", live_iam_evidence)
    exception_now = int(time.time())
    exception = cloud.build_shared_project_exception(
        transport_contract=direct_contract,
        actual_preflight_artifact=preflight,
        gate_plan=gate_plan,
        live_iam_evidence=live_iam_evidence,
        now_unix_seconds=exception_now,
    )
    _write(OUTPUT_ROOT / "shared_project_exception.json", exception)
    if (
        expiry_seconds - int(time.time())
        < cloud.MIN_EXECUTION_REMAINING_SECONDS
    ):
        raise RuntimeError(
            "Step11 authorization margin expired before instance insert"
        )

    receipt = cloud.execute_step11_attempt0(
        execute=True,
        execution_confirmation=cloud.EXECUTION_CONFIRMATION,
        client=controller_client,
        collector_client=ActiveUserReadOnlyClient(user_token),
        transport_contract=direct_contract,
        launch_contract=launch_contract,
        signer=signer,
        package_provision_receipt=package_receipt,
        outer_root=outer_root,
        actual_preflight_artifact=preflight,
        gate_plan=gate_plan,
        gate_observation=None,
        live_iam_evidence=live_iam_evidence,
        post_claim_callback=_post_claim_revoke,
        shared_project_exception=exception,
        startup_path=STARTUP_PATH,
        prebootstrap_path=PREBOOTSTRAP_PATH,
        destination_root=OUTPUT_ROOT / "received",
        final_receipt_path=OUTPUT_ROOT / "final_lifecycle_receipt.json",
    )
    return 0


def _run_with_cleanup() -> int:
    try:
        result = main()
    except BaseException as primary_error:
        if OUTPUT_ROOT.is_dir() and not OUTPUT_ROOT.is_symlink():
            try:
                _final_cleanup()
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "Step11 failed and final cloud cleanup also failed",
                    [primary_error, cleanup_error],
                ) from None
        raise
    cleanup = _final_cleanup()
    receipt = json.loads(
        (OUTPUT_ROOT / "final_lifecycle_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    sys.stdout.write(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt_sha256": receipt["receipt_sha256"],
                "cleanup_receipt_sha256": cleanup["receipt_sha256"],
                "instance_name": EXPECTED_INSTANCE_NAME,
                "provider_get_404_observed": receipt[
                    "provider_get_404_observed"
                ],
                "all_step11_iam_removed": True,
                "output_root": str(OUTPUT_ROOT),
            },
            sort_keys=True,
        )
        + "\n"
    )
    return result


if __name__ == "__main__":
    raise SystemExit(_run_with_cleanup())
