#!/usr/bin/env python3
"""Read-only closeout of the existing Step12b token-barrier failure.

This command has no execute flag and no mutation adapter.  It performs only
provider GET/list/generation reads, then writes one deterministic local
receipt.  Re-running it accepts an identical existing receipt and otherwise
fails closed; it never overwrites the receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as transport,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_rest_iam_admin_v1
    as rest_iam,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_failure_closeout_v1
    as closeout,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_live_cloud_adapters_v2
    as live_cloud,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_live_preflight_collectors_v2
    as live_collectors,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_run_scoped_controller_sa_v2
    as controller_sa,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
STEP6D_ROOT = (
    REPO_ROOT / "outputs" / "hu_joint_policy" / "m31_t3_step6d"
)
OUTPUT_ROOT = STEP6D_ROOT / "step12b_pair_v2_actual"
RECEIPT_PATH = OUTPUT_ROOT / "token_barrier_failure_closeout_receipt.json"
POLICY_REGISTRY_PATH = REPO_ROOT / "src" / "ofc_regular" / "ai_profiles.py"
OLD_TREE_ROOTS = {
    "step11_one_vm_v12_fix3_actual": (
        STEP6D_ROOT / "step11_one_vm_v12_fix3_actual"
    ),
    "rearm2_diagnostic_cloud_worker_package_v1": (
        STEP6D_ROOT / "rearm2_diagnostic_cloud_worker_package_v1"
    ),
    "step12_pair_v1_actual": STEP6D_ROOT / "step12_pair_v1_actual",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"required regular JSON file missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"required JSON object changed: {path}")
    return value


class GcloudReadOnlyInspector:
    """Narrow CLI adapter exposing policy and Cloud Audit reads only."""

    _MAX_JSON_BYTES = 8 * 1024 * 1024
    _AUDIT_FORMAT = (
        "json(timestamp,protoPayload.serviceName,"
        "protoPayload.methodName,protoPayload.resourceName,"
        "protoPayload.status,protoPayload.response)"
    )

    def __init__(
        self,
        *,
        project: str,
        bucket: str,
        worker_service_account: str,
        executable: str | None = None,
        command_runner: Callable[..., Any] = subprocess.run,
    ) -> None:
        selected = executable or shutil.which(
            "gcloud.cmd" if os.name == "nt" else "gcloud"
        )
        if not isinstance(selected, str) or not selected:
            raise RuntimeError("gcloud executable is unavailable")
        if not callable(command_runner):
            raise ValueError("gcloud command runner changed")
        self._executable = selected
        self._run = command_runner
        self._project = project
        self._bucket = bucket
        self._worker_service_account = worker_service_account

    def _read_json(
        self, operation: str, arguments: Sequence[str]
    ) -> Any:
        if (
            not isinstance(operation, str)
            or not operation
            or isinstance(arguments, (str, bytes))
            or not isinstance(arguments, Sequence)
            or any(
                not isinstance(argument, str) or not argument
                for argument in arguments
            )
        ):
            raise ValueError("read-only gcloud command changed")
        completed = self._run(
            [self._executable, *arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=90,
            check=False,
        )
        if (
            getattr(completed, "returncode", None) != 0
            or not isinstance(getattr(completed, "stdout", None), str)
        ):
            raise RuntimeError(f"{operation}_failed")
        raw = completed.stdout.encode("utf-8")
        if not raw or len(raw) > self._MAX_JSON_BYTES:
            raise RuntimeError(f"{operation}_response_changed")
        try:
            return json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise RuntimeError(
                f"{operation}_response_changed"
            ) from None

    def get_policy(
        self, target: rest_iam.PolicyTarget
    ) -> Mapping[str, Any]:
        if target is rest_iam.PolicyTarget.PROJECT:
            arguments = [
                "projects",
                "get-iam-policy",
                self._project,
                "--format=json",
            ]
        elif target is rest_iam.PolicyTarget.BUCKET:
            arguments = [
                "storage",
                "buckets",
                "get-iam-policy",
                f"gs://{self._bucket}",
                "--format=json",
            ]
        elif target is rest_iam.PolicyTarget.WORKER_SERVICE_ACCOUNT:
            arguments = [
                "iam",
                "service-accounts",
                "get-iam-policy",
                self._worker_service_account,
                f"--project={self._project}",
                "--format=json",
            ]
        else:
            raise ValueError("policy read escaped Phase2 target set")
        value = self._read_json("iam_policy_get", arguments)
        if not isinstance(value, Mapping):
            raise RuntimeError("iam_policy_get_response_changed")
        return dict(value)

    def read_token_barrier_audit_events(
        self, *, controller_unique_id: str
    ) -> list[dict[str, Any]]:
        if (
            not isinstance(controller_unique_id, str)
            or not controller_unique_id.isdecimal()
        ):
            raise ValueError("controller unique ID changed")
        resource = (
            "projects/-/serviceAccounts/" + controller_unique_id
        )
        query = (
            'timestamp>="2026-07-20T09:43:40Z" AND '
            'timestamp<="2026-07-20T09:44:20Z" AND '
            f'protoPayload.resourceName="{resource}" AND '
            '(protoPayload.methodName="GenerateAccessToken" OR '
            "protoPayload.methodName="
            '"google.iam.admin.v1.SetIAMPolicy" OR '
            "protoPayload.methodName="
            '"google.iam.admin.v1.DeleteServiceAccount")'
        )
        value = self._read_json(
            "cloud_audit_log_read",
            [
                "logging",
                "read",
                query,
                f"--project={self._project}",
                "--limit=20",
                "--order=asc",
                f"--format={self._AUDIT_FORMAT}",
            ],
        )
        if not isinstance(value, list) or len(value) != 3:
            raise RuntimeError("cloud_audit_log_read_response_changed")
        event_roles = {
            "GenerateAccessToken": "controller_token_mint",
            "google.iam.admin.v1.SetIAMPolicy": (
                "token_creator_revoke"
            ),
            "google.iam.admin.v1.DeleteServiceAccount": (
                "controller_service_account_delete"
            ),
        }
        rows = []
        for raw in value:
            if (
                not isinstance(raw, Mapping)
                or set(raw) != {"timestamp", "protoPayload"}
                or not isinstance(raw["timestamp"], str)
                or not isinstance(raw["protoPayload"], Mapping)
            ):
                raise RuntimeError(
                    "cloud_audit_log_read_response_changed"
                )
            payload = raw["protoPayload"]
            method = payload.get("methodName")
            status = payload.get("status")
            if (
                method not in event_roles
                or payload.get("resourceName") != resource
                or status != {}
            ):
                raise RuntimeError(
                    "cloud_audit_log_read_response_changed"
                )
            response = payload.get("response")
            is_revoke = (
                method == "google.iam.admin.v1.SetIAMPolicy"
            )
            if is_revoke:
                if (
                    not isinstance(response, Mapping)
                    or set(response) != {"@type", "etag", "version"}
                    or response.get("@type")
                    != "type.googleapis.com/google.iam.v1.Policy"
                    or not isinstance(response.get("etag"), str)
                    or not response["etag"]
                    or response.get("version") != 1
                    or "bindings" in response
                ):
                    raise RuntimeError(
                        "cloud_audit_log_read_response_changed"
                    )
                response_fields = {
                    "response_policy_present": True,
                    "response_policy_etag_present": True,
                    "response_policy_etag_sha256": hashlib.sha256(
                        response["etag"].encode("ascii")
                    ).hexdigest(),
                    "response_policy_version": 1,
                    "response_policy_bindings_field_present": False,
                    "response_policy_binding_count": 0,
                }
            else:
                if response is not None:
                    raise RuntimeError(
                        "cloud_audit_log_read_response_changed"
                    )
                response_fields = {
                    "response_policy_present": False,
                    "response_policy_etag_present": False,
                    "response_policy_etag_sha256": None,
                    "response_policy_version": None,
                    "response_policy_bindings_field_present": None,
                    "response_policy_binding_count": None,
                }
            rows.append(
                {
                    "event_role": event_roles[method],
                    "timestamp": raw["timestamp"],
                    "service_name": payload.get("serviceName"),
                    "method_name": method,
                    "resource_name": resource,
                    "status_code": 0,
                    "successful": True,
                    **response_fields,
                }
            )
        return rows


class LiveReadOnlyCloseoutObserver:
    """GET-only adapters bound to the one existing Step12b identity."""

    def __init__(
        self,
        *,
        deployment: Mapping[str, Any],
        phase2_plan: Mapping[str, Any],
        source_receipt: Mapping[str, Any],
    ) -> None:
        self._deployment = dict(deployment)
        self._plan = dict(phase2_plan)
        self._source = dict(source_receipt)
        self._log: list[str] = []
        self._tokens = live_cloud.GcloudUserAccessTokenSource()
        self._http = live_cloud.StdlibCloudHttpsClient()
        names = [
            row["instance_name"] for row in deployment["instances"]
        ]
        self._compute = live_cloud.ExactPairComputeClient(
            token_source=self._tokens,
            instance_names=names,
            principal="user:step12b-read-only-closeout",
            credential_kind="user",
            http_client=self._http,
            allow_insert=False,
        )
        identity = controller_sa.validate_controller_identity(
            deployment["controller_service_account"]
        )
        self._controller = (
            controller_sa.RunScopedControllerServiceAccountAdmin(
                http_client=self._http,
                user_token_source=self._tokens,
                identity=identity,
            )
        )
        self._gcloud = GcloudReadOnlyInspector(
            project=transport.PROJECT,
            bucket=transport.BUCKET,
            worker_service_account=transport.WORKER_SERVICE_ACCOUNT,
        )
        source_plan = {
            "source_prefix": source_receipt["source_prefix"],
            "objects": [
                {"uri": row["uri"]} for row in source_receipt["records"]
            ],
        }
        self._source_store = (
            live_cloud.GenerationPinnedBootstrapSourceStore(
                source_plan=source_plan,
                http_client=self._http,
                user_token_source=self._tokens,
            )
        )

    def verify_exact_compute_absent(
        self,
        *,
        instance_names: Sequence[str],
        disk_names: Sequence[str],
    ) -> Mapping[str, Any]:
        self._log.append("compute_exact_four_get")
        return self._compute.verify_exact_instances_and_disks_absent(
            instance_names=instance_names,
            disk_names=disk_names,
        )

    def get_controller_service_account(
        self, *, email: str
    ) -> Mapping[str, Any] | None:
        if email != self._controller.identity.email:
            raise ValueError("controller service-account read escaped identity")
        self._log.append("controller_service_account_get")
        return self._controller.get()

    def get_iam_policy(
        self, *, target: str
    ) -> Mapping[str, Any] | None:
        checked = rest_iam.PolicyTarget(target)
        self._log.append("iam_policy_get")
        return self._gcloud.get_policy(checked)

    def list_direct_v2_stage_objects(
        self, *, stage_prefix: str
    ) -> Sequence[str]:
        if stage_prefix != self._deployment["remote_layout"]["stage_prefix"]:
            raise ValueError("direct-v2 stage read escaped identity")
        self._log.append("direct_v2_stage_list")
        receipt = live_collectors.collect_direct_v2_prefix_empty_receipt(
            self._deployment,
            http_client=self._http,
            token_source=self._tokens,
            observed_at_unix_seconds=int(time.time()),
        )
        if (
            receipt.get("object_count") != 0
            or receipt.get("collected_via_get_only") is not True
            or receipt.get("cloud_mutation_performed") is not False
        ):
            raise ValueError("direct-v2 stage empty readback changed")
        return []

    def list_bootstrap_source_objects(
        self, *, source_prefix: str
    ) -> Sequence[str]:
        self._log.append("bootstrap_source_list")
        return self._source_store.list_objects(prefix=source_prefix)

    def read_bootstrap_source_generation(
        self, *, uri: str, generation: int
    ) -> Mapping[str, Any]:
        self._log.append("bootstrap_source_generation_get")
        raw = self._source_store.generation_pinned_get(
            uri=uri, generation=generation
        )
        return {
            "uri": uri,
            "generation": generation,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "readback_complete": True,
        }

    def read_token_barrier_audit_events(
        self, *, controller_unique_id: str
    ) -> Sequence[Mapping[str, Any]]:
        self._log.append("cloud_audit_log_read")
        return self._gcloud.read_token_barrier_audit_events(
            controller_unique_id=controller_unique_id
        )

    def audit_log(self) -> Sequence[str]:
        return tuple(self._log)


def build_live_closeout_receipt() -> dict[str, Any]:
    deployment = _read_json(OUTPUT_ROOT / "deployment_contract.json")
    plan = _read_json(OUTPUT_ROOT / "phase2_iam_plan.json")
    source = _read_json(
        OUTPUT_ROOT / "bootstrap_source_provision_receipt.json"
    )
    observer = LiveReadOnlyCloseoutObserver(
        deployment=deployment,
        phase2_plan=plan,
        source_receipt=source,
    )
    return closeout.verify_token_barrier_failure_closeout(
        output_root=OUTPUT_ROOT,
        observer=observer,
        policy_registry_path=POLICY_REGISTRY_PATH,
        old_tree_roots=OLD_TREE_ROOTS,
        expected_old_tree_digests=closeout.EXPECTED_OLD_TREE_DIGESTS,
    )


def write_once_or_validate_identical(
    path: Path, receipt: Mapping[str, Any]
) -> dict[str, Any]:
    checked = closeout.validate_closeout_receipt(receipt)
    raw = closeout.canonical_bytes(checked) + b"\n"
    if path.exists() or path.is_symlink():
        if not path.is_file() or path.is_symlink():
            raise ValueError("closeout receipt path changed")
        existing = path.read_bytes()
        if existing != raw:
            raise FileExistsError(
                "different closeout receipt already exists"
            )
        return closeout.validate_closeout_receipt(
            json.loads(existing.decode("ascii"))
        )
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
    return checked


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only Step12b token-barrier failure closeout verifier"
        )
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="perform all readbacks without creating the local receipt",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = build_live_closeout_receipt()
    if not args.verify_only:
        receipt = write_once_or_validate_identical(
            RECEIPT_PATH, receipt
        )
    sys.stdout.write(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt_sha256": receipt["receipt_sha256"],
                "receipt_path": (
                    None if args.verify_only else str(RECEIPT_PATH)
                ),
                "cloud_read_only": True,
                "cloud_mutation_performed": False,
                "automatic_retry_performed": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
