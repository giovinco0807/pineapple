"""Mode-separated GCP REST adapter for full-100 wave Phase A.

The adapter exposes only four capabilities:

* ``read``: quota, selected instance/disk absence, and worker-SA GETs;
* ``claim``: one exact GCS create-if-absent followed by exact GET readback;
* ``bucket-iam-install``: one validated bucket IAM add CAS;
* ``bucket-iam-cleanup``: one validated bucket IAM removal CAS.

There is deliberately no GCE create/delete surface here.  HTTP is injectable
for deterministic tests.  Bearer credentials are read from the environment
for every request and are never retained or emitted in receipts/errors.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.parse
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from . import hu_m31_t3_step6d_full100_wave_cloud_v2 as cloud_v2
from . import hu_m31_t3_step6d_full100_wave_plan_v2 as wave_v2
from . import hu_m31_t3_step6d_full100_wave_worker_iam_v2 as worker_iam_v2
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_10c2_real_readonly_preflight_v1
    as proven_readonly_v1,
)
from .hu_m31_t3_step6d_performance_development_v2_cloud_execution_v1 import (
    HttpResponse,
    _stdlib_http_request,
)


READ_RECEIPT_SCHEMA = "hu_m31_t3_step6d_full100_wave_gcp_read_receipt_v2"
SERVICE_ACCOUNT_ACTAS_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_service_account_actas_receipt_v2"
)
SELECTED_RESULT_PREFLIGHT_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_selected_result_preflight_receipt_v2"
)

PROJECT = worker_iam_v2.PROJECT
REGION = "asia-northeast1"
ZONE = "asia-northeast1-b"
BUCKET = worker_iam_v2.BUCKET
MACHINE_TYPE = "c4-standard-16"
TOKEN_ENV = "GOOGLE_OAUTH_ACCESS_TOKEN"
ACT_AS_PERMISSION = "iam.serviceAccounts.actAs"
PROVIDER_PROJECT_PERMISSIONS = (
    "compute.disks.create",
    "compute.disks.delete",
    "compute.disks.get",
    "compute.disks.setLabels",
    "compute.instances.create",
    "compute.instances.delete",
    "compute.instances.get",
    "compute.instances.setMetadata",
    "compute.instances.setServiceAccount",
    "compute.networks.use",
    "compute.subnetworks.use",
    "compute.zoneOperations.get",
)
MAX_ACTAS_RECEIPT_LIFETIME_SECONDS = 300

C4_QUOTA_ID = "CPUS-PER-VM-FAMILY-per-project-region"
C4_QUOTA_METRIC = "compute.googleapis.com/cpus_per_vm_family"
C4_QUOTA_VM_FAMILY = "C4"
GLOBAL_QUOTA_ID = "CPUS-ALL-REGIONS-per-project"
GLOBAL_QUOTA_METRIC = "compute.googleapis.com/cpus_all_regions"
SPOT_QUOTA_METRICS = ("PREEMPTIBLE_CPUS", "SPOT_CPUS")
CUSTOM_ROLE_EXPECTATIONS = (
    (worker_iam_v2.READER_ROLE, worker_iam_v2.ROLE_PERMISSIONS[worker_iam_v2.READER_ROLE]),
    (
        worker_iam_v2.CREATOR_ROLE,
        worker_iam_v2.ROLE_PERMISSIONS[worker_iam_v2.CREATOR_ROLE],
    ),
)

_MODES = frozenset(
    {
        "read",
        "actas-check",
        "claim",
        "bucket-iam-prepare",
        "bucket-iam-install",
        "bucket-iam-reconcile-install",
        "bucket-iam-readback",
        "bucket-iam-cleanup",
        "bucket-iam-reconcile-cleanup",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_PROJECT = re.compile(r"^[a-z][a-z0-9-]{4,61}[a-z0-9]$")
_SAFE_REGION = re.compile(r"^[a-z]+-[a-z0-9]+[0-9]$")
_SAFE_ZONE = re.compile(r"^[a-z]+-[a-z0-9]+[0-9]-[a-z]$")
_SAFE_BUCKET = re.compile(r"^[a-z0-9][a-z0-9._-]{1,221}[a-z0-9]$")
_SERVICE_ACCOUNT = re.compile(
    rf"^[a-z][a-z0-9-]{{4,28}}[a-z0-9]@{re.escape(PROJECT)}"
    r"\.iam\.gserviceaccount\.com$"
)
_UNIQUE_ID = re.compile(r"^[1-9][0-9]{5,31}$")

_CLAIM_PAYLOAD_KEYS = frozenset(
    {
        "schema",
        "wave_plan_sha256",
        "attempt_ledger_sha256",
        "resume_plan_sha256",
        "immutable_content_sha256",
        "run_name",
        "execution_identity_sha256",
        "wave_index",
        "selected_attempts_sha256",
        "selected_vm_count",
        "project_id",
        "zone",
        "claim_nonce_sha256",
        "claimed_at_utc",
        "create_only_precondition_generation",
    }
)
_SERVICE_ACCOUNT_ROW_KEYS = frozenset(
    {
        "job_id",
        "source_role",
        "service_account",
        "resource_name",
        "unique_id",
        "disabled",
        "exists",
    }
)
_CUSTOM_ROLE_ROW_KEYS = frozenset(
    {"role_name", "included_permissions", "stage", "deleted", "etag_sha256"}
)
_RESULT_PREFLIGHT_ROW_KEYS = frozenset(
    {
        "job_id", "source_role", "attempt_id", "artifact_prefix",
        "acceptance_path", "list_page_count", "listed_object_count",
        "artifact_prefix_absent", "acceptance_object_absent",
    }
)
_RESULT_PREFLIGHT_KEYS = frozenset(
    {
        "schema", "status", "bucket", "run_name",
        "execution_identity_sha256", "wave_plan_sha256",
        "attempt_ledger_sha256", "resume_plan_sha256", "wave_index",
        "selected_attempts_sha256", "rows", "selected_attempt_count",
        "http_get_count", "pagination_observed",
        "all_selected_artifact_prefixes_absent",
        "all_selected_acceptance_objects_absent", "read_only",
        "cloud_mutation_performed", "observed_at_utc",
        "current_profile_changed", "receipt_sha256",
    }
)
_PROVIDER_QUOTA_KEYS = frozenset(
    {
        "c4",
        "c4_quota_id",
        "c4_project_number",
        "c4_usage_inventory_sha256",
        "spot",
        "global",
        "global_quota_id",
        "global_usage_inventory_sha256",
    }
)
_ACTAS_ROW_KEYS = frozenset(
    {
        "job_id",
        "source_role",
        "service_account",
        "resource_name",
        "unique_id",
        "requested_permissions",
        "granted_permissions",
        "act_as_granted",
        "test_complete",
    }
)
_ACTAS_KEYS = frozenset(
    {
        "schema",
        "status",
        "project",
        "region",
        "zone",
        "bucket",
        "run_name",
        "execution_identity_sha256",
        "wave_plan_sha256",
        "attempt_ledger_sha256",
        "resume_plan_sha256",
        "wave_index",
        "immutable_content_prefix",
        "content_payload_sha256",
        "outer_manifest_sha256",
        "iam_plan_sha256",
        "gcp_read_receipt_sha256",
        "checked_at_utc",
        "expires_at_utc",
        "permission",
        "rows",
        "selected_worker_count",
        "http_post_count",
        "all_selected_workers_act_as_granted",
        "provider_project_permissions_requested",
        "provider_project_permissions_granted",
        "provider_project_permission_count",
        "all_provider_project_permissions_granted",
        "permission_test_only",
        "cloud_mutation_performed",
        "current_profile_changed",
        "receipt_sha256",
    }
)
_READ_KEYS = frozenset(
    {
        "schema",
        "status",
        "project",
        "region",
        "zone",
        "bucket",
        "run_name",
        "execution_identity_sha256",
        "wave_plan_sha256",
        "attempt_ledger_sha256",
        "resume_plan_sha256",
        "wave_index",
        "immutable_content_prefix",
        "content_payload_sha256",
        "outer_manifest_sha256",
        "iam_plan_sha256",
        "provider_quota_metrics",
        "quota_receipt",
        "planned_mapping_receipt",
        "custom_roles",
        "custom_roles_sha256",
        "custom_role_count",
        "selected_result_preflight_receipt",
        "observed_at_utc",
        "service_accounts",
        "selected_vm_count",
        "http_get_count",
        "all_selected_instances_absent",
        "all_selected_boot_disks_absent",
        "all_service_accounts_exist",
        "all_custom_roles_exact_ga_not_deleted",
        "pagination_observed",
        "read_only",
        "cloud_mutation_performed",
        "current_profile_changed",
        "receipt_sha256",
    }
)

HttpRequester = Callable[
    [str, str, Mapping[str, str], bytes | None, int], HttpResponse
]


class GcpPhaseATransportError(RuntimeError):
    """The provider may have committed a mutation but returned no status."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _require_sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or _SHA256.fullmatch(value) is None
        or value == "0" * 64
    ):
        raise ValueError(f"{label} must be a nonzero lowercase SHA-256")
    return value


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = deepcopy(dict(value))
    if "receipt_sha256" in body:
        raise ValueError("GCP read receipt was already sealed")
    return {**body, "receipt_sha256": canonical_sha256(body)}


def _validate_scope(
    *, project: Any, region: Any, zone: Any, bucket: Any
) -> tuple[str, str, str, str]:
    if (
        not isinstance(project, str)
        or _SAFE_PROJECT.fullmatch(project) is None
        or project != PROJECT
        or not isinstance(region, str)
        or _SAFE_REGION.fullmatch(region) is None
        or region != REGION
        or not isinstance(zone, str)
        or _SAFE_ZONE.fullmatch(zone) is None
        or zone != ZONE
        or not isinstance(bucket, str)
        or _SAFE_BUCKET.fullmatch(bucket) is None
        or bucket != BUCKET
    ):
        raise ValueError("GCP Phase A target escaped the fixed project scope")
    return project, region, zone, bucket


def _validate_content_binding(
    *,
    immutable_content_prefix: Any,
    content_payload_sha256: Any,
    outer_manifest_sha256: Any,
) -> tuple[str, str, str]:
    payload_sha = _require_sha(content_payload_sha256, "content payload")
    manifest_sha = _require_sha(outer_manifest_sha256, "outer manifest")
    expected = (
        f"{worker_iam_v2.IMMUTABLE_CONTENT_PREFIX_ROOT}/{payload_sha}"
    )
    if immutable_content_prefix != expected:
        raise ValueError("GCP adapter immutable content prefix changed")
    return expected, payload_sha, manifest_sha


def _validated_context(
    *,
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    immutable_content_prefix: str,
    content_payload_sha256: str,
    outer_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str, str, str]:
    plan = wave_v2.validate_wave_plan(wave_plan)
    ledger = wave_v2.validate_attempt_ledger(plan, attempt_ledger)
    resume = wave_v2.validate_resume_plan(plan, ledger, resume_plan)
    if (
        resume["all_jobs_complete"] is not False
        or resume["resume_wave_index"] is None
        or not 1 <= len(resume["selected_attempts"]) <= wave_v2.MAX_CONCURRENT_VMS
        or plan["quota_contract"]["region"] != REGION
        or plan["quota_contract"]["machine_type"] != MACHINE_TYPE
    ):
        raise ValueError("GCP adapter wave context is not launchable")
    content_prefix, content_sha, manifest_sha = _validate_content_binding(
        immutable_content_prefix=immutable_content_prefix,
        content_payload_sha256=content_payload_sha256,
        outer_manifest_sha256=outer_manifest_sha256,
    )
    return plan, ledger, resume, content_prefix, content_sha, manifest_sha


def _claim_path(plan: Mapping[str, Any], resume: Mapping[str, Any]) -> str:
    wave_id = plan["waves"][resume["resume_wave_index"]]["wave_id"]
    return (
        f"{plan['artifact_contract']['prefix']}/control/waves/{wave_id}/claims/"
        f"{resume['resume_sha256']}.json"
    )


def _numeric_quota(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} quota value changed")
    if value < 0 or int(value) != value:
        raise RuntimeError(f"{label} quota value changed")
    return int(value)


def _utc_seconds(value: Any, label: str) -> int:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} is invalid") from exc
    if parsed.tzinfo != timezone.utc:
        raise ValueError(f"{label} is not UTC")
    return int(parsed.timestamp())


def _provider_policy_response_matches_exact_mutation(
    *, supplied: Mapping[str, Any], result: Mapping[str, Any]
) -> bool:
    """Accept only GCS's condition-free policy-version canonicalization."""

    if worker_iam_v2._policy_fingerprint(result) == worker_iam_v2._policy_fingerprint(
        supplied
    ):
        return True
    if supplied.get("version") != 3 or result.get("version") != 1:
        return False
    if any("condition" in binding for binding in supplied.get("bindings", [])):
        return False
    if any("condition" in binding for binding in result.get("bindings", [])):
        return False
    normalized = deepcopy(dict(result))
    normalized["version"] = 3
    return worker_iam_v2._policy_fingerprint(
        normalized
    ) == worker_iam_v2._policy_fingerprint(supplied)


class GcpWavePhaseAAdapter:
    """Strict mode-specific REST boundary for prelaunch operations."""

    def __init__(
        self,
        *,
        mode: str,
        project: str,
        region: str,
        zone: str,
        bucket: str,
        wave_plan: Mapping[str, Any],
        attempt_ledger: Mapping[str, Any],
        resume_plan: Mapping[str, Any],
        immutable_content_prefix: str,
        content_payload_sha256: str,
        outer_manifest_sha256: str,
        iam_plan: Mapping[str, Any] | None = None,
        gcp_read_receipt: Mapping[str, Any] | None = None,
        prepare_receipt: Mapping[str, Any] | None = None,
        install_receipt: Mapping[str, Any] | None = None,
        readback_receipt: Mapping[str, Any] | None = None,
        requester: HttpRequester | None = None,
    ) -> None:
        if mode not in _MODES:
            raise ValueError("GCP Phase A adapter mode is invalid")
        self.project, self.region, self.zone, self.bucket = _validate_scope(
            project=project, region=region, zone=zone, bucket=bucket
        )
        (
            self._plan,
            self._ledger,
            self._resume,
            self._content_prefix,
            self._content_sha,
            self._manifest_sha,
        ) = _validated_context(
            wave_plan=wave_plan,
            attempt_ledger=attempt_ledger,
            resume_plan=resume_plan,
            immutable_content_prefix=immutable_content_prefix,
            content_payload_sha256=content_payload_sha256,
            outer_manifest_sha256=outer_manifest_sha256,
        )
        self.mode = mode
        self._requester = requester or _stdlib_http_request
        if not callable(self._requester):
            raise ValueError("GCP Phase A requester is invalid")
        self._iam_plan: dict[str, Any] | None = None
        self._gcp_read: dict[str, Any] | None = None
        self._prepare: dict[str, Any] | None = None
        self._install: dict[str, Any] | None = None
        self._readback: dict[str, Any] | None = None
        self._last_policy: dict[str, Any] | None = None
        self._set_count = 0
        self._http_count = 0

        content_kwargs = self._content_kwargs()
        if mode == "read":
            if iam_plan is None:
                raise PermissionError("read mode requires a validated worker IAM plan")
            self._iam_plan = worker_iam_v2.validate_worker_iam_plan(
                wave_plan=self._plan,
                attempt_ledger=self._ledger,
                resume_plan=self._resume,
                **content_kwargs,
                value=iam_plan,
            )
            if any(
                value is not None
                for value in (
                    gcp_read_receipt,
                    prepare_receipt,
                    install_receipt,
                    readback_receipt,
                )
            ):
                raise ValueError("read mode received later evidence")
        elif mode == "actas-check":
            if iam_plan is None or gcp_read_receipt is None:
                raise PermissionError(
                    "actAs check mode requires IAM plan and GCP read receipt"
                )
            if any(
                value is not None
                for value in (prepare_receipt, install_receipt, readback_receipt)
            ):
                raise ValueError("actAs check mode received mutation receipts")
            self._iam_plan = worker_iam_v2.validate_worker_iam_plan(
                wave_plan=self._plan,
                attempt_ledger=self._ledger,
                resume_plan=self._resume,
                **content_kwargs,
                value=iam_plan,
            )
            self._gcp_read = validate_read_receipt(
                wave_plan=self._plan,
                attempt_ledger=self._ledger,
                resume_plan=self._resume,
                **content_kwargs,
                iam_plan=self._iam_plan,
                value=gcp_read_receipt,
            )
        elif mode == "claim":
            if any(
                value is not None
                for value in (
                    iam_plan,
                    gcp_read_receipt,
                    prepare_receipt,
                    install_receipt,
                    readback_receipt,
                )
            ):
                raise ValueError("claim mode received IAM capabilities")
        elif mode == "bucket-iam-prepare":
            if iam_plan is None:
                raise PermissionError(
                    "bucket IAM prepare mode requires a validated IAM plan"
                )
            if any(
                value is not None
                for value in (
                    gcp_read_receipt,
                    prepare_receipt,
                    install_receipt,
                    readback_receipt,
                )
            ):
                raise ValueError("bucket IAM prepare mode received later evidence")
            self._iam_plan = worker_iam_v2.validate_worker_iam_plan(
                wave_plan=self._plan,
                attempt_ledger=self._ledger,
                resume_plan=self._resume,
                **content_kwargs,
                value=iam_plan,
            )
        elif mode in {"bucket-iam-install", "bucket-iam-reconcile-install"}:
            if iam_plan is None or prepare_receipt is None:
                raise PermissionError(
                    "bucket IAM install mode requires plan and prepare receipt"
                )
            if (
                gcp_read_receipt is not None
                or install_receipt is not None
                or readback_receipt is not None
            ):
                raise ValueError("bucket IAM install mode received later receipts")
            self._iam_plan = worker_iam_v2.validate_worker_iam_plan(
                wave_plan=self._plan,
                attempt_ledger=self._ledger,
                resume_plan=self._resume,
                **content_kwargs,
                value=iam_plan,
            )
            self._prepare = worker_iam_v2.validate_prepare_receipt(
                iam_plan=self._iam_plan,
                wave_plan=self._plan,
                attempt_ledger=self._ledger,
                resume_plan=self._resume,
                **content_kwargs,
                value=prepare_receipt,
            )
        elif mode == "bucket-iam-readback":
            if (
                iam_plan is None
                or prepare_receipt is None
                or install_receipt is None
                or gcp_read_receipt is not None
                or readback_receipt is not None
            ):
                raise PermissionError(
                    "bucket IAM readback mode requires plan, prepare, and install only"
                )
            self._iam_plan = worker_iam_v2.validate_worker_iam_plan(
                wave_plan=self._plan,
                attempt_ledger=self._ledger,
                resume_plan=self._resume,
                **content_kwargs,
                value=iam_plan,
            )
            self._prepare = worker_iam_v2.validate_prepare_receipt(
                iam_plan=self._iam_plan,
                wave_plan=self._plan,
                attempt_ledger=self._ledger,
                resume_plan=self._resume,
                **content_kwargs,
                value=prepare_receipt,
            )
            self._install = worker_iam_v2.validate_install_receipt(
                iam_plan=self._iam_plan,
                wave_plan=self._plan,
                attempt_ledger=self._ledger,
                resume_plan=self._resume,
                **content_kwargs,
                prepare_receipt=self._prepare,
                value=install_receipt,
            )
        else:
            if gcp_read_receipt is not None:
                raise ValueError("bucket IAM cleanup mode received GCP read evidence")
            if any(
                value is None
                for value in (
                    iam_plan,
                    prepare_receipt,
                    install_receipt,
                    readback_receipt,
                )
            ):
                raise PermissionError(
                    "bucket IAM cleanup mode requires the complete receipt chain"
                )
            self._iam_plan = worker_iam_v2.validate_worker_iam_plan(
                wave_plan=self._plan,
                attempt_ledger=self._ledger,
                resume_plan=self._resume,
                **content_kwargs,
                value=iam_plan,
            )
            self._prepare = worker_iam_v2.validate_prepare_receipt(
                iam_plan=self._iam_plan,
                wave_plan=self._plan,
                attempt_ledger=self._ledger,
                resume_plan=self._resume,
                **content_kwargs,
                value=prepare_receipt,
            )
            self._install = worker_iam_v2.validate_install_receipt(
                iam_plan=self._iam_plan,
                wave_plan=self._plan,
                attempt_ledger=self._ledger,
                resume_plan=self._resume,
                **content_kwargs,
                prepare_receipt=self._prepare,
                value=install_receipt,
            )
            self._readback = worker_iam_v2.validate_readback_receipt(
                iam_plan=self._iam_plan,
                wave_plan=self._plan,
                attempt_ledger=self._ledger,
                resume_plan=self._resume,
                **content_kwargs,
                prepare_receipt=self._prepare,
                install_receipt=self._install,
                value=readback_receipt,
            )

    def _content_kwargs(self) -> dict[str, str]:
        return {
            "immutable_content_prefix": self._content_prefix,
            "content_payload_sha256": self._content_sha,
            "outer_manifest_sha256": self._manifest_sha,
        }

    def _token_headers(self, *, content_type: str | None = None) -> dict[str, str]:
        token = os.environ.get(TOKEN_ENV)
        if not isinstance(token, str) or len(token) < 20 or any(
            character.isspace() for character in token
        ):
            raise PermissionError(
                f"Bearer token must be supplied only through {TOKEN_ENV}"
            )
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        if content_type is not None:
            headers["Content-Type"] = content_type
        return headers

    def _http(
        self,
        *,
        method: str,
        url: str,
        body: bytes | None = None,
        content_type: str | None = None,
        allowed_statuses: Sequence[int] = (200,),
        timeout_seconds: int = 60,
    ) -> HttpResponse:
        allowed_methods = {
            "read": {"GET"},
            "actas-check": {"POST"},
            "claim": {"GET", "POST"},
            "bucket-iam-prepare": {"GET"},
            "bucket-iam-install": {"GET", "PUT"},
            "bucket-iam-reconcile-install": {"GET"},
            "bucket-iam-readback": {"GET"},
            "bucket-iam-cleanup": {"GET", "PUT"},
            "bucket-iam-reconcile-cleanup": {"GET"},
        }[self.mode]
        if method not in allowed_methods:
            raise PermissionError("HTTP method escaped the fixed adapter mode")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int)
            or not 1 <= timeout_seconds <= 600
        ):
            raise ValueError("GCP Phase A timeout escaped the fixed maximum")
        self._http_count += 1
        headers = self._token_headers(content_type=content_type)
        try:
            response = self._requester(
                method,
                url,
                headers,
                body,
                timeout_seconds,
            )
        except Exception as exc:
            raise GcpPhaseATransportError(
                f"GCP Phase A {method} transport failed without provider status"
            ) from exc
        if not isinstance(response, HttpResponse):
            raise RuntimeError("GCP Phase A requester returned an invalid response")
        if response.status not in allowed_statuses:
            raise RuntimeError(
                f"GCP Phase A {method} failed with status {response.status}"
            )
        return response

    @staticmethod
    def _json(response: HttpResponse, label: str) -> dict[str, Any]:
        try:
            value = json.loads(response.body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"{label} response was not JSON") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"{label} response was not an object")
        if value.get("nextPageToken") is not None:
            raise RuntimeError(f"{label} pagination is not authorized")
        return value

    def _quota_rows(
        self, response: HttpResponse, *, scope_name: str, label: str
    ) -> list[dict[str, Any]]:
        value = self._json(response, label)
        if frozenset(value) != frozenset({"name", "quotas"}):
            raise RuntimeError(f"{label} top-level fields changed")
        if value.get("name") != scope_name:
            raise RuntimeError(f"{label} scope changed")
        quotas = value.get("quotas")
        if not isinstance(quotas, list):
            raise RuntimeError(f"{label} quotas changed")
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in quotas:
            if not isinstance(raw, Mapping) or frozenset(raw) != frozenset(
                {"metric", "limit", "usage"}
            ):
                raise RuntimeError(f"{label} quota entry fields changed")
            if not isinstance(raw.get("metric"), str):
                raise RuntimeError(f"{label} quota entry changed")
            metric = raw["metric"]
            if metric in seen:
                raise RuntimeError(f"{label} returned a duplicate quota metric")
            seen.add(metric)
            limit = _numeric_quota(raw.get("limit"), f"{label} {metric} limit")
            usage = _numeric_quota(raw.get("usage"), f"{label} {metric} usage")
            if usage > limit:
                raise RuntimeError(f"{label} quota usage exceeds its limit")
            rows.append(
                {
                    "metric": metric,
                    "limit_vcpus": limit,
                    "usage_vcpus": usage,
                    "available_vcpus": limit - usage,
                }
            )
        return rows

    @staticmethod
    def _one_quota(
        rows: Sequence[Mapping[str, Any]], metrics: Sequence[str], label: str
    ) -> dict[str, Any]:
        matches = [row for row in rows if row["metric"] in metrics]
        if len(matches) != 1:
            raise RuntimeError(f"{label} quota metric is missing or ambiguous")
        return deepcopy(dict(matches[0]))

    def _c4_cloud_quota(self, response: HttpResponse) -> dict[str, Any]:
        value = self._json(response, "Cloud Quotas C4 quotaInfo")
        name = value.get("name")
        expected_suffix = (
            "/locations/global/services/compute.googleapis.com/quotaInfos/"
            f"{C4_QUOTA_ID}"
        )
        if (
            not isinstance(name, str)
            or not name.startswith("projects/")
            or not name.endswith(expected_suffix)
        ):
            raise RuntimeError("Cloud Quotas C4 resource name changed")
        project_number = name[len("projects/") : -len(expected_suffix)]
        if not project_number.isdigit() or int(project_number) <= 0:
            raise RuntimeError("Cloud Quotas C4 project number changed")
        if (
            value.get("quotaId") != C4_QUOTA_ID
            or value.get("metric") != C4_QUOTA_METRIC
            or value.get("service") != "compute.googleapis.com"
            or value.get("isPrecise") is not True
            or value.get("containerType") != "PROJECT"
            or value.get("metricUnit") != "1"
            or value.get("dimensions") != ["region", "vm_family"]
        ):
            raise RuntimeError("Cloud Quotas C4 identity changed")
        dimension_infos = value.get("dimensionsInfos")
        if not isinstance(dimension_infos, list):
            raise RuntimeError("Cloud Quotas C4 dimensions changed")
        matches = [
            row
            for row in dimension_infos
            if isinstance(row, Mapping)
            and row.get("dimensions")
            == {"region": self.region, "vm_family": C4_QUOTA_VM_FAMILY}
        ]
        if len(matches) != 1:
            raise RuntimeError("Cloud Quotas C4 dimension is missing or ambiguous")
        match = matches[0]
        details = match.get("details")
        limit_value = details.get("value") if isinstance(details, Mapping) else None
        if (
            match.get("applicableLocations") != [self.region]
            or not isinstance(limit_value, str)
            or not limit_value.isdigit()
            or int(limit_value) <= 0
        ):
            raise RuntimeError("Cloud Quotas C4 limit changed")
        return {
            "metric": C4_QUOTA_METRIC,
            "quota_id": C4_QUOTA_ID,
            "project_number": project_number,
            "limit_vcpus": int(limit_value),
        }

    def _proven_inventory_target(self) -> Any:
        if self._iam_plan is None:
            raise PermissionError("inventory target requires a worker IAM plan")
        return proven_readonly_v1._Target(
            project=self.project,
            bucket=self.bucket,
            region=self.region,
            zone=self.zone,
            machine_type=MACHINE_TYPE,
            package_object_prefix=self._content_prefix,
            stage_object_prefix=self._plan["artifact_contract"]["prefix"],
            expected_instance_names=tuple(
                row["instance_id"] for row in self._resume["selected_attempts"]
            ),
            service_account_email=self._iam_plan["workers"][0][
                "service_account"
            ],
            image_project=proven_readonly_v1.IMAGE_PROJECT,
            image_name=proven_readonly_v1.IMAGE_NAME,
            image_id=proven_readonly_v1.IMAGE_ID,
            image_self_link=proven_readonly_v1.IMAGE_SELF_LINK,
        )

    def _proven_usage_inventory(self) -> tuple[Any, list[dict[str, Any]]]:
        target = self._proven_inventory_target()
        facts: list[dict[str, Any]] = []
        for endpoint_id in proven_readonly_v1.GLOBAL_CPU_EMPTY_INVENTORY_ENDPOINTS:
            url = proven_readonly_v1._request_url(
                endpoint_id,
                target,
                page_token=None,
            )
            value = self._json(
                self._http(method="GET", url=url),
                f"{endpoint_id} inventory",
            )
            facts.append(
                proven_readonly_v1._empty_compute_inventory_facts(
                    value,
                    endpoint_id=endpoint_id,
                    target=target,
                )
            )
        return target, facts

    def _selected_result_preflight(self, *, observed_at_utc: str) -> dict[str, Any]:
        """Read only the selected attempt prefixes and selected job accepts."""

        start_count = self._http_count
        rows: list[dict[str, Any]] = []
        pagination_observed = False
        acceptance_template = self._plan["artifact_contract"][
            "job_acceptance_path_template"
        ]
        list_fields = urllib.parse.quote(
            "items(name,generation,etag,size),nextPageToken", safe="(),"
        )
        for selected in self._resume["selected_attempts"]:
            prefix = f"{selected['artifact_prefix']}/"
            page_token: str | None = None
            seen_tokens: set[str] = set()
            page_count = 0
            listed_count = 0
            while True:
                page_count += 1
                if page_count > 100:
                    raise RuntimeError("selected result prefix pagination exceeded bound")
                query = (
                    f"prefix={urllib.parse.quote(prefix, safe='')}"
                    f"&maxResults=1000&fields={list_fields}"
                )
                if page_token is not None:
                    query += f"&pageToken={urllib.parse.quote(page_token, safe='')}"
                response = self._http(
                    method="GET",
                    url=(
                        "https://storage.googleapis.com/storage/v1/b/"
                        f"{self.bucket}/o?{query}"
                    ),
                )
                try:
                    page = json.loads(response.body)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise RuntimeError(
                        "selected result prefix list response was not JSON"
                    ) from exc
                if not isinstance(page, dict) or not set(page).issubset(
                    {"items", "nextPageToken"}
                ):
                    raise RuntimeError("selected result prefix list shape changed")
                items = page.get("items", [])
                if not isinstance(items, list):
                    raise RuntimeError("selected result prefix list items changed")
                for item in items:
                    if (
                        not isinstance(item, Mapping)
                        or not isinstance(item.get("name"), str)
                        or not item["name"].startswith(prefix)
                    ):
                        raise RuntimeError("selected result prefix list identity changed")
                listed_count += len(items)
                if listed_count:
                    raise FileExistsError(
                        "selected attempt result prefix is not empty"
                    )
                next_token = page.get("nextPageToken")
                if next_token is None:
                    break
                if (
                    not isinstance(next_token, str)
                    or not next_token
                    or next_token in seen_tokens
                ):
                    raise RuntimeError("selected result prefix pagination token changed")
                pagination_observed = True
                seen_tokens.add(next_token)
                page_token = next_token

            acceptance_path = acceptance_template.format(
                job_id=selected["job_id"]
            )
            acceptance_url = (
                "https://storage.googleapis.com/storage/v1/b/"
                f"{self.bucket}/o/"
                f"{urllib.parse.quote(acceptance_path, safe='')}"
                "?fields=name%2Cgeneration%2Cetag%2Csize"
            )
            acceptance = self._http(
                method="GET", url=acceptance_url, allowed_statuses=(200, 404)
            )
            if acceptance.status != 404:
                raise FileExistsError("selected job already has ACCEPTED.json")
            rows.append(
                {
                    "job_id": selected["job_id"],
                    "source_role": selected["source_role"],
                    "attempt_id": selected["attempt_id"],
                    "artifact_prefix": selected["artifact_prefix"],
                    "acceptance_path": acceptance_path,
                    "list_page_count": page_count,
                    "listed_object_count": 0,
                    "artifact_prefix_absent": True,
                    "acceptance_object_absent": True,
                }
            )
        core = {
            "schema": SELECTED_RESULT_PREFLIGHT_RECEIPT_SCHEMA,
            "status": "selected_attempt_prefixes_and_acceptance_objects_absent",
            "bucket": self.bucket,
            "run_name": self._plan["run_name"],
            "execution_identity_sha256": self._plan["execution_identity_sha256"],
            "wave_plan_sha256": self._plan["schedule_sha256"],
            "attempt_ledger_sha256": self._ledger["ledger_sha256"],
            "resume_plan_sha256": self._resume["resume_sha256"],
            "wave_index": self._resume["resume_wave_index"],
            "selected_attempts_sha256": canonical_sha256(
                self._resume["selected_attempts"]
            ),
            "rows": rows,
            "selected_attempt_count": len(rows),
            "http_get_count": self._http_count - start_count,
            "pagination_observed": pagination_observed,
            "all_selected_artifact_prefixes_absent": True,
            "all_selected_acceptance_objects_absent": True,
            "read_only": True,
            "cloud_mutation_performed": False,
            "observed_at_utc": observed_at_utc,
            "current_profile_changed": False,
        }
        return _seal(core)

    def read_prelaunch(
        self, *, observed_at_utc: str, expires_at_utc: str
    ) -> dict[str, Any]:
        if self.mode != "read" or self._iam_plan is None:
            raise PermissionError("prelaunch observation requires read mode")
        start_count = self._http_count
        cloud_quota_url = (
            "https://cloudquotas.googleapis.com/v1/projects/"
            f"{self.project}/locations/global/services/compute.googleapis.com/"
            f"quotaInfos/{C4_QUOTA_ID}"
        )
        c4_response = self._http(method="GET", url=cloud_quota_url)
        c4_limit = self._c4_cloud_quota(c4_response)
        c4_quota_info = self._json(c4_response, "Cloud Quotas C4 quotaInfo")
        global_quota_url = (
            "https://cloudquotas.googleapis.com/v1/projects/"
            f"{self.project}/locations/global/services/compute.googleapis.com/"
            f"quotaInfos/{GLOBAL_QUOTA_ID}"
        )
        global_quota_info = self._json(
            self._http(method="GET", url=global_quota_url),
            "Cloud Quotas global quotaInfo",
        )
        inventory_target, inventory_facts = self._proven_usage_inventory()
        global_facts = proven_readonly_v1._global_cpu_quota_facts(
            global_quota_info,
            inventory_facts,
            target=inventory_target,
            project_number=c4_limit["project_number"],
        )
        c4_facts = proven_readonly_v1._regional_c4_quota_facts(
            c4_quota_info,
            target=inventory_target,
            project_number=c4_limit["project_number"],
            usage_inventory_proof=inventory_facts,
        )
        fields = urllib.parse.quote("name,quotas", safe=",")
        region_url = (
            "https://compute.googleapis.com/compute/v1/projects/"
            f"{self.project}/regions/{self.region}?fields={fields}"
        )
        region_rows = self._quota_rows(
            self._http(method="GET", url=region_url),
            scope_name=self.region,
            label="regional quota",
        )
        spot = self._one_quota(region_rows, SPOT_QUOTA_METRICS, "Spot")
        if spot["usage_vcpus"] < c4_facts["target_region_spot_vcpu_observed"]:
            raise RuntimeError("regional Spot usage is below proven inventory")
        c4 = {
            "limit_vcpus": int(c4_facts["limit"]),
            "usage_vcpus": int(c4_facts["usage"]),
            "available_vcpus": int(c4_facts["available"]),
        }
        global_cpu = {
            "limit_vcpus": int(global_facts["limit"]),
            "usage_vcpus": int(global_facts["usage"]),
            "available_vcpus": int(global_facts["available"]),
        }
        quota_metrics = [
            {
                "metric": "c4_family_vcpus",
                **{key: c4[key] for key in (
                    "limit_vcpus", "usage_vcpus", "available_vcpus"
                )},
                "readback_complete": True,
            },
            {
                "metric": "spot_vcpus",
                **{key: spot[key] for key in (
                    "limit_vcpus", "usage_vcpus", "available_vcpus"
                )},
                "readback_complete": True,
            },
            {
                "metric": "global_vcpus",
                **{key: global_cpu[key] for key in (
                    "limit_vcpus", "usage_vcpus", "available_vcpus"
                )},
                "readback_complete": True,
            },
        ]
        quota_receipt = cloud_v2.build_live_quota_headroom_receipt(
            self._plan,
            self._ledger,
            self._resume,
            immutable_content_sha256=self._content_sha,
            project_id=self.project,
            zone=self.zone,
            observed_at_utc=observed_at_utc,
            expires_at_utc=expires_at_utc,
            quota_metrics=quota_metrics,
            readback_source="cloud_quotas_api_and_compute_inventory",
        )

        absence_rows: list[dict[str, Any]] = []
        for selected in self._resume["selected_attempts"]:
            encoded = urllib.parse.quote(selected["instance_id"], safe="")
            instance_url = (
                "https://compute.googleapis.com/compute/v1/projects/"
                f"{self.project}/zones/{self.zone}/instances/{encoded}"
            )
            instance = self._http(
                method="GET", url=instance_url, allowed_statuses=(200, 404)
            )
            if instance.status != 404:
                raise FileExistsError("selected GCE instance name is not absent")
            disk_url = (
                "https://compute.googleapis.com/compute/v1/projects/"
                f"{self.project}/zones/{self.zone}/disks/{encoded}"
            )
            disk = self._http(
                method="GET", url=disk_url, allowed_statuses=(200, 404)
            )
            if disk.status != 404:
                raise FileExistsError("selected GCE boot disk name is not absent")
            absence_rows.append(
                {
                    "instance_id": selected["instance_id"],
                    "instance_absent": True,
                    "boot_disk_absent": True,
                    "readback_complete": True,
                }
            )
        mapping_receipt = cloud_v2.build_planned_launch_mapping_receipt(
            self._plan,
            self._ledger,
            self._resume,
            immutable_content_sha256=self._content_sha,
            project_id=self.project,
            zone=self.zone,
            observed_at_utc=observed_at_utc,
            expires_at_utc=expires_at_utc,
            instance_absence_readbacks=absence_rows,
            readback_source="compute_instances_and_disks_api",
        )
        result_preflight = self._selected_result_preflight(
            observed_at_utc=observed_at_utc
        )

        custom_roles: list[dict[str, Any]] = []
        role_fields = urllib.parse.quote(
            "name,includedPermissions,stage,deleted,etag", safe=","
        )
        for role_name, expected_permissions in CUSTOM_ROLE_EXPECTATIONS:
            role_id = role_name.rsplit("/", 1)[-1]
            role_url = (
                "https://iam.googleapis.com/v1/projects/"
                f"{self.project}/roles/{urllib.parse.quote(role_id, safe='')}"
                f"?fields={role_fields}"
            )
            value = self._json(
                self._http(method="GET", url=role_url),
                f"custom role {role_id}",
            )
            allowed_fields = {
                "name", "includedPermissions", "stage", "deleted", "etag"
            }
            if (
                not set(value).issubset(allowed_fields)
                or not {"name", "includedPermissions", "stage", "etag"}.issubset(
                    value
                )
                or value.get("name") != role_name
                or value.get("includedPermissions") != list(expected_permissions)
                or value.get("stage") != "GA"
                or value.get("deleted", False) is not False
                or not isinstance(value.get("etag"), str)
                or not value["etag"]
            ):
                raise RuntimeError("custom role permission, stage, or deletion state changed")
            custom_roles.append(
                {
                    "role_name": role_name,
                    "included_permissions": list(expected_permissions),
                    "stage": "GA",
                    "deleted": False,
                    "etag_sha256": hashlib.sha256(
                        value["etag"].encode("utf-8")
                    ).hexdigest(),
                }
            )

        account_rows: list[dict[str, Any]] = []
        for worker in self._iam_plan["workers"]:
            account = worker["service_account"]
            encoded_account = urllib.parse.quote(account, safe="")
            account_url = (
                "https://iam.googleapis.com/v1/projects/"
                f"{self.project}/serviceAccounts/{encoded_account}"
            )
            response = self._http(method="GET", url=account_url)
            value = self._json(response, "service account")
            unique_id = value.get("uniqueId")
            name = value.get("name")
            allowed_names = {
                f"projects/{self.project}/serviceAccounts/{account}",
                f"projects/{self.project}/serviceAccounts/{unique_id}",
            }
            if (
                value.get("email") != account
                or value.get("projectId") != self.project
                or value.get("disabled", False) is not False
                or not isinstance(unique_id, str)
                or _UNIQUE_ID.fullmatch(unique_id) is None
                or not isinstance(name, str)
                or name not in allowed_names
            ):
                raise RuntimeError("service account identity changed")
            account_rows.append(
                {
                    "job_id": worker["job_id"],
                    "source_role": worker["source_role"],
                    "service_account": account,
                    "resource_name": name,
                    "unique_id": unique_id,
                    "disabled": False,
                    "exists": True,
                }
            )
        core = {
            "schema": READ_RECEIPT_SCHEMA,
            "status": "fresh_phase_a_readback_complete",
            "project": self.project,
            "region": self.region,
            "zone": self.zone,
            "bucket": self.bucket,
            "run_name": self._plan["run_name"],
            "execution_identity_sha256": self._plan[
                "execution_identity_sha256"
            ],
            "wave_plan_sha256": self._plan["schedule_sha256"],
            "attempt_ledger_sha256": self._ledger["ledger_sha256"],
            "resume_plan_sha256": self._resume["resume_sha256"],
            "wave_index": self._resume["resume_wave_index"],
            "immutable_content_prefix": self._content_prefix,
            "content_payload_sha256": self._content_sha,
            "outer_manifest_sha256": self._manifest_sha,
            "iam_plan_sha256": self._iam_plan["plan_sha256"],
            "provider_quota_metrics": {
                "c4": C4_QUOTA_METRIC,
                "c4_quota_id": C4_QUOTA_ID,
                "c4_project_number": c4_limit["project_number"],
                "c4_usage_inventory_sha256": c4_facts[
                    "usage_inventory_proof_sha256"
                ],
                "spot": spot["metric"],
                "global": GLOBAL_QUOTA_METRIC,
                "global_quota_id": GLOBAL_QUOTA_ID,
                "global_usage_inventory_sha256": (
                    proven_readonly_v1.canonical_sha256(
                        global_facts["usage_inventory_proof"]
                    )
                ),
            },
            "quota_receipt": quota_receipt,
            "planned_mapping_receipt": mapping_receipt,
            "custom_roles": custom_roles,
            "custom_roles_sha256": canonical_sha256(custom_roles),
            "custom_role_count": len(custom_roles),
            "selected_result_preflight_receipt": result_preflight,
            "observed_at_utc": observed_at_utc,
            "service_accounts": account_rows,
            "selected_vm_count": len(self._resume["selected_attempts"]),
            "http_get_count": self._http_count - start_count,
            "all_selected_instances_absent": True,
            "all_selected_boot_disks_absent": True,
            "all_service_accounts_exist": True,
            "all_custom_roles_exact_ga_not_deleted": True,
            "pagination_observed": result_preflight["pagination_observed"],
            "read_only": True,
            "cloud_mutation_performed": False,
            "current_profile_changed": False,
        }
        return validate_read_receipt(
            wave_plan=self._plan,
            attempt_ledger=self._ledger,
            resume_plan=self._resume,
            immutable_content_prefix=self._content_prefix,
            content_payload_sha256=self._content_sha,
            outer_manifest_sha256=self._manifest_sha,
            iam_plan=self._iam_plan,
            value=_seal(core),
        )

    def check_service_account_act_as(
        self, *, checked_at_utc: str, expires_at_utc: str
    ) -> dict[str, Any]:
        """Prove the current caller can attach every selected worker account."""

        if (
            self.mode != "actas-check"
            or self._iam_plan is None
            or self._gcp_read is None
        ):
            raise PermissionError(
                "service-account permission test requires actAs check mode"
            )
        checked = _utc_seconds(checked_at_utc, "actAs checked time")
        expires = _utc_seconds(expires_at_utc, "actAs expiry")
        if not 1 <= expires - checked <= MAX_ACTAS_RECEIPT_LIFETIME_SECONDS:
            raise ValueError("actAs receipt lifetime changed")
        read_observed = max(
            _utc_seconds(
                self._gcp_read[name]["observed_at_utc"],
                f"{name} observed time",
            )
            for name in ("quota_receipt", "planned_mapping_receipt")
        )
        read_expires = min(
            _utc_seconds(
                self._gcp_read[name]["expires_at_utc"],
                f"{name} expiry",
            )
            for name in ("quota_receipt", "planned_mapping_receipt")
        )
        if checked < read_observed or expires > read_expires:
            raise PermissionError("actAs check is outside the GCP read window")

        start_count = self._http_count
        request_body = canonical_bytes({"permissions": [ACT_AS_PERMISSION]})
        rows: list[dict[str, Any]] = []
        for account in self._gcp_read["service_accounts"]:
            encoded = urllib.parse.quote(account["service_account"], safe="")
            url = (
                "https://iam.googleapis.com/v1/projects/"
                f"{self.project}/serviceAccounts/{encoded}:testIamPermissions"
            )
            response = self._json(
                self._http(
                    method="POST",
                    url=url,
                    body=request_body,
                    content_type="application/json",
                ),
                "service-account testIamPermissions",
            )
            if not set(response).issubset({"permissions"}):
                raise RuntimeError(
                    "service-account permission response fields changed"
                )
            granted = response.get("permissions", [])
            if granted != [ACT_AS_PERMISSION]:
                raise PermissionError(
                    "caller lacks iam.serviceAccounts.actAs for a selected worker"
                )
            rows.append(
                {
                    "job_id": account["job_id"],
                    "source_role": account["source_role"],
                    "service_account": account["service_account"],
                    "resource_name": account["resource_name"],
                    "unique_id": account["unique_id"],
                    "requested_permissions": [ACT_AS_PERMISSION],
                    "granted_permissions": [ACT_AS_PERMISSION],
                    "act_as_granted": True,
                    "test_complete": True,
                }
            )
        provider_body = canonical_bytes(
            {"permissions": list(PROVIDER_PROJECT_PERMISSIONS)}
        )
        provider_url = (
            "https://cloudresourcemanager.googleapis.com/v1/projects/"
            f"{self.project}:testIamPermissions"
        )
        provider_response = self._json(
            self._http(
                method="POST",
                url=provider_url,
                body=provider_body,
                content_type="application/json",
            ),
            "provider project testIamPermissions",
        )
        if not set(provider_response).issubset({"permissions"}):
            raise RuntimeError("provider project permission response fields changed")
        provider_granted = provider_response.get("permissions", [])
        if provider_granted != list(PROVIDER_PROJECT_PERMISSIONS):
            raise PermissionError(
                "caller lacks an exact required GCE provider project permission"
            )
        core = {
            "schema": SERVICE_ACCOUNT_ACTAS_RECEIPT_SCHEMA,
            "status": (
                "all_selected_worker_act_as_and_provider_permissions_granted"
            ),
            "project": self.project,
            "region": self.region,
            "zone": self.zone,
            "bucket": self.bucket,
            "run_name": self._plan["run_name"],
            "execution_identity_sha256": self._plan[
                "execution_identity_sha256"
            ],
            "wave_plan_sha256": self._plan["schedule_sha256"],
            "attempt_ledger_sha256": self._ledger["ledger_sha256"],
            "resume_plan_sha256": self._resume["resume_sha256"],
            "wave_index": self._resume["resume_wave_index"],
            "immutable_content_prefix": self._content_prefix,
            "content_payload_sha256": self._content_sha,
            "outer_manifest_sha256": self._manifest_sha,
            "iam_plan_sha256": self._iam_plan["plan_sha256"],
            "gcp_read_receipt_sha256": self._gcp_read["receipt_sha256"],
            "checked_at_utc": checked_at_utc,
            "expires_at_utc": expires_at_utc,
            "permission": ACT_AS_PERMISSION,
            "rows": rows,
            "selected_worker_count": len(rows),
            "http_post_count": self._http_count - start_count,
            "all_selected_workers_act_as_granted": True,
            "provider_project_permissions_requested": list(
                PROVIDER_PROJECT_PERMISSIONS
            ),
            "provider_project_permissions_granted": list(
                PROVIDER_PROJECT_PERMISSIONS
            ),
            "provider_project_permission_count": len(
                PROVIDER_PROJECT_PERMISSIONS
            ),
            "all_provider_project_permissions_granted": True,
            "permission_test_only": True,
            "cloud_mutation_performed": False,
            "current_profile_changed": False,
        }
        return validate_service_account_actas_receipt(
            wave_plan=self._plan,
            attempt_ledger=self._ledger,
            resume_plan=self._resume,
            immutable_content_prefix=self._content_prefix,
            content_payload_sha256=self._content_sha,
            outer_manifest_sha256=self._manifest_sha,
            iam_plan=self._iam_plan,
            gcp_read_receipt=self._gcp_read,
            value=_seal(core),
        )

    def _validate_claim_payload(self, payload: bytes) -> dict[str, Any]:
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("launch claim payload is not canonical JSON") from exc
        if not isinstance(value, dict) or set(value) != _CLAIM_PAYLOAD_KEYS:
            raise ValueError("launch claim payload fields changed")
        expected = {
            "schema": cloud_v2.LAUNCH_CLAIM_PAYLOAD_SCHEMA,
            "wave_plan_sha256": self._plan["schedule_sha256"],
            "attempt_ledger_sha256": self._ledger["ledger_sha256"],
            "resume_plan_sha256": self._resume["resume_sha256"],
            "immutable_content_sha256": self._content_sha,
            "run_name": self._plan["run_name"],
            "execution_identity_sha256": self._plan[
                "execution_identity_sha256"
            ],
            "wave_index": self._resume["resume_wave_index"],
            "selected_attempts_sha256": cloud_v2.canonical_sha256(
                self._resume["selected_attempts"]
            ),
            "selected_vm_count": len(self._resume["selected_attempts"]),
            "project_id": self.project,
            "zone": self.zone,
            "create_only_precondition_generation": 0,
        }
        if any(value.get(key) != expected_value for key, expected_value in expected.items()):
            raise ValueError("launch claim payload escaped wave/content context")
        _require_sha(value.get("claim_nonce_sha256"), "launch claim nonce")
        if (
            not isinstance(value.get("claimed_at_utc"), str)
            or not value["claimed_at_utc"].endswith("Z")
            or cloud_v2.canonical_bytes(value) != payload
        ):
            raise ValueError("launch claim payload is not canonical")
        return value

    def put_if_absent(
        self, *, object_name: str, payload: bytes
    ) -> Mapping[str, Any]:
        if self.mode != "claim":
            raise PermissionError("GCS claim creation requires claim mode")
        if object_name != _claim_path(self._plan, self._resume):
            raise ValueError("GCS claim path escaped the exact resume plan")
        self._validate_claim_payload(payload)
        query = urllib.parse.urlencode(
            {
                "uploadType": "media",
                "ifGenerationMatch": "0",
                "name": object_name,
            }
        )
        url = (
            "https://storage.googleapis.com/upload/storage/v1/b/"
            f"{urllib.parse.quote(self.bucket, safe='')}/o?{query}"
        )
        try:
            response = self._http(
                method="POST",
                url=url,
                body=payload,
                content_type="application/octet-stream",
                allowed_statuses=(200, 412),
            )
        except GcpPhaseATransportError as exc:
            try:
                recovered = self.get_object(object_name=object_name)
            except Exception:
                raise exc
            if (
                recovered.get("payload") != payload
                or recovered.get("sha256") != hashlib.sha256(payload).hexdigest()
                or recovered.get("bytes") != len(payload)
                or not isinstance(recovered.get("generation"), str)
                or not isinstance(recovered.get("etag"), str)
            ):
                raise RuntimeError(
                    "persistent launch claim transport recovery identity changed"
                ) from exc
            return {
                "created": True,
                "object_name": object_name,
                "generation": recovered["generation"],
                "etag": recovered["etag"],
                "sha256": recovered["sha256"],
                "bytes": recovered["bytes"],
            }
        if response.status == 412:
            raise FileExistsError("persistent launch claim already exists")
        value = self._json(response, "GCS claim create")
        generation = value.get("generation")
        etag = value.get("etag")
        if (
            value.get("name") != object_name
            or not isinstance(generation, str)
            or not generation.isdigit()
            or int(generation) <= 0
            or not isinstance(etag, str)
            or not etag
            or int(value.get("size", -1)) != len(payload)
        ):
            raise RuntimeError("GCS claim create identity changed")
        return {
            "created": True,
            "object_name": object_name,
            "generation": generation,
            "etag": etag,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }

    def get_object(self, *, object_name: str) -> Mapping[str, Any]:
        if self.mode != "claim":
            raise PermissionError("GCS claim readback requires claim mode")
        if object_name != _claim_path(self._plan, self._resume):
            raise ValueError("GCS claim readback escaped the exact resume plan")
        encoded = urllib.parse.quote(object_name, safe="")
        fields = urllib.parse.quote("name,generation,etag,size", safe=",")
        metadata_url = (
            "https://storage.googleapis.com/storage/v1/b/"
            f"{urllib.parse.quote(self.bucket, safe='')}/o/{encoded}?fields={fields}"
        )
        metadata = self._json(
            self._http(method="GET", url=metadata_url), "GCS claim metadata"
        )
        generation = metadata.get("generation")
        etag = metadata.get("etag")
        size = metadata.get("size")
        if (
            metadata.get("name") != object_name
            or not isinstance(generation, str)
            or not generation.isdigit()
            or int(generation) <= 0
            or not isinstance(etag, str)
            or not etag
            or not isinstance(size, str)
            or not size.isdigit()
        ):
            raise RuntimeError("GCS claim metadata identity changed")
        media_query = urllib.parse.urlencode(
            {"alt": "media", "ifGenerationMatch": generation}
        )
        media_url = (
            "https://storage.googleapis.com/storage/v1/b/"
            f"{urllib.parse.quote(self.bucket, safe='')}/o/{encoded}?{media_query}"
        )
        media = self._http(method="GET", url=media_url)
        if len(media.body) != int(size):
            raise RuntimeError("GCS claim bytes changed on readback")
        self._validate_claim_payload(media.body)
        return {
            "object_name": object_name,
            "generation": generation,
            "etag": etag,
            "sha256": hashlib.sha256(media.body).hexdigest(),
            "bytes": len(media.body),
            "payload": media.body,
        }

    def _bucket_iam_url(self) -> str:
        return (
            "https://storage.googleapis.com/storage/v1/b/"
            f"{urllib.parse.quote(self.bucket, safe='')}/iam"
            "?optionsRequestedPolicyVersion=3"
        )

    def get_bucket_policy(self) -> Mapping[str, Any]:
        if self.mode not in {
            "bucket-iam-prepare", "bucket-iam-install",
            "bucket-iam-reconcile-install", "bucket-iam-readback",
            "bucket-iam-cleanup", "bucket-iam-reconcile-cleanup",
        }:
            raise PermissionError("bucket IAM GET requires an IAM mode")
        value = self._json(
            self._http(method="GET", url=self._bucket_iam_url()),
            "bucket IAM",
        )
        policy = worker_iam_v2._validated_policy(value)
        self._last_policy = deepcopy(policy)
        return policy

    def set_bucket_policy(
        self, *, policy: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if self.mode not in {"bucket-iam-install", "bucket-iam-cleanup"}:
            raise PermissionError("bucket IAM PUT requires an IAM mode")
        if self._iam_plan is None or self._last_policy is None:
            raise PermissionError("bucket IAM PUT requires an exact preceding GET")
        if self._set_count != 0:
            raise PermissionError("bucket IAM adapter permits exactly one CAS attempt")
        supplied = worker_iam_v2._validated_policy(policy)
        install = self.mode == "bucket-iam-install"
        expected = worker_iam_v2._mutation_policy(
            self._last_policy, self._iam_plan, install=install
        )
        if supplied != expected:
            raise ValueError("bucket IAM PUT escaped the validated exact mutation")
        if supplied["etag"] != self._last_policy["etag"]:
            raise ValueError("bucket IAM PUT lost its exact ETag CAS")
        expected_etag_sha = (
            self._prepare["pre_policy_etag_sha256"]
            if install and self._prepare is not None
            else self._readback["observed_policy_etag_sha256"]
            if self._readback is not None
            else None
        )
        if (
            expected_etag_sha is None
            or hashlib.sha256(supplied["etag"].encode()).hexdigest()
            != expected_etag_sha
        ):
            raise ValueError("bucket IAM PUT ETag differs from the bound receipt")
        self._set_count += 1
        response = self._http(
            method="PUT",
            url=self._bucket_iam_url(),
            body=canonical_bytes(supplied),
            content_type="application/json",
            allowed_statuses=(200, 409, 412),
        )
        if response.status in {409, 412}:
            raise worker_iam_v2.WorkerIamCasError(
                "bucket IAM ETag CAS failed"
            )
        result = worker_iam_v2._validated_policy(
            self._json(response, "bucket IAM PUT")
        )
        if (
            result["etag"] == supplied["etag"]
            or not _provider_policy_response_matches_exact_mutation(
                supplied=supplied, result=result
            )
        ):
            raise RuntimeError("bucket IAM PUT readback changed the exact mutation")
        self._last_policy = deepcopy(result)
        return {
            "set": True,
            "policy_etag_sha256": hashlib.sha256(
                result["etag"].encode()
            ).hexdigest(),
            "body_emitted": False,
            "token_emitted": False,
        }

    # Explicitly refuse the Phase B surface even if a caller guesses names.
    def create_instance(self, *_: Any, **__: Any) -> Mapping[str, Any]:
        raise PermissionError("GCE create belongs to Phase B")

    def delete_instance_exact(self, *_: Any, **__: Any) -> Mapping[str, Any]:
        raise PermissionError("GCE delete belongs to Phase B")


def _validate_selected_result_preflight_receipt(
    *,
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _RESULT_PREFLIGHT_KEYS:
        raise ValueError("selected result preflight receipt fields changed")
    receipt = deepcopy(dict(value))
    digest = receipt.pop("receipt_sha256", None)
    if digest != canonical_sha256(receipt):
        raise ValueError("selected result preflight receipt digest changed")
    receipt["receipt_sha256"] = digest
    rows = receipt.get("rows")
    selected = resume["selected_attempts"]
    if not isinstance(rows, list) or len(rows) != len(selected):
        raise ValueError("selected result preflight row cardinality changed")
    expected_get_count = 0
    pagination = False
    acceptance_template = plan["artifact_contract"][
        "job_acceptance_path_template"
    ]
    for selected_row, raw in zip(selected, rows, strict=True):
        if not isinstance(raw, Mapping) or set(raw) != _RESULT_PREFLIGHT_ROW_KEYS:
            raise ValueError("selected result preflight row fields changed")
        page_count = raw.get("list_page_count")
        if (
            raw.get("job_id") != selected_row["job_id"]
            or raw.get("source_role") != selected_row["source_role"]
            or raw.get("attempt_id") != selected_row["attempt_id"]
            or raw.get("artifact_prefix") != selected_row["artifact_prefix"]
            or raw.get("acceptance_path")
            != acceptance_template.format(job_id=selected_row["job_id"])
            or isinstance(page_count, bool)
            or not isinstance(page_count, int)
            or not 1 <= page_count <= 100
            or raw.get("listed_object_count") != 0
            or raw.get("artifact_prefix_absent") is not True
            or raw.get("acceptance_object_absent") is not True
        ):
            raise ValueError("selected result preflight row identity changed")
        expected_get_count += page_count + 1
        pagination = pagination or page_count > 1
    _utc_seconds(receipt.get("observed_at_utc"), "result preflight observed time")
    if (
        receipt.get("schema") != SELECTED_RESULT_PREFLIGHT_RECEIPT_SCHEMA
        or receipt.get("status")
        != "selected_attempt_prefixes_and_acceptance_objects_absent"
        or receipt.get("bucket") != BUCKET
        or receipt.get("run_name") != plan["run_name"]
        or receipt.get("execution_identity_sha256")
        != plan["execution_identity_sha256"]
        or receipt.get("wave_plan_sha256") != plan["schedule_sha256"]
        or receipt.get("attempt_ledger_sha256") != ledger["ledger_sha256"]
        or receipt.get("resume_plan_sha256") != resume["resume_sha256"]
        or receipt.get("wave_index") != resume["resume_wave_index"]
        or receipt.get("selected_attempts_sha256")
        != canonical_sha256(selected)
        or receipt.get("selected_attempt_count") != len(selected)
        or receipt.get("http_get_count") != expected_get_count
        or receipt.get("pagination_observed") is not pagination
        or receipt.get("all_selected_artifact_prefixes_absent") is not True
        or receipt.get("all_selected_acceptance_objects_absent") is not True
        or receipt.get("read_only") is not True
        or receipt.get("cloud_mutation_performed") is not False
        or receipt.get("current_profile_changed") is not False
    ):
        raise ValueError("selected result preflight receipt evidence changed")
    return receipt


def validate_read_receipt(
    *,
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    immutable_content_prefix: str,
    content_payload_sha256: str,
    outer_manifest_sha256: str,
    iam_plan: Mapping[str, Any],
    value: Mapping[str, Any],
) -> dict[str, Any]:
    (
        plan,
        ledger,
        resume,
        content_prefix,
        content_sha,
        manifest_sha,
    ) = _validated_context(
        wave_plan=wave_plan,
        attempt_ledger=attempt_ledger,
        resume_plan=resume_plan,
        immutable_content_prefix=immutable_content_prefix,
        content_payload_sha256=content_payload_sha256,
        outer_manifest_sha256=outer_manifest_sha256,
    )
    iam = worker_iam_v2.validate_worker_iam_plan(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        immutable_content_prefix=content_prefix,
        content_payload_sha256=content_sha,
        outer_manifest_sha256=manifest_sha,
        value=iam_plan,
    )
    if not isinstance(value, Mapping) or set(value) != _READ_KEYS:
        raise ValueError("GCP read receipt fields changed")
    receipt = deepcopy(dict(value))
    digest = receipt.pop("receipt_sha256", None)
    if digest != canonical_sha256(receipt):
        raise ValueError("GCP read receipt digest changed")
    receipt["receipt_sha256"] = digest
    quota = cloud_v2.validate_live_quota_headroom_receipt(
        plan,
        ledger,
        resume,
        receipt.get("quota_receipt", {}),
        immutable_content_sha256=content_sha,
    )
    mapping = cloud_v2.validate_planned_launch_mapping_receipt(
        plan,
        ledger,
        resume,
        receipt.get("planned_mapping_receipt", {}),
        immutable_content_sha256=content_sha,
    )
    result_preflight = _validate_selected_result_preflight_receipt(
        plan=plan,
        ledger=ledger,
        resume=resume,
        value=receipt.get("selected_result_preflight_receipt", {}),
    )
    observed_at_utc = receipt.get("observed_at_utc")
    _utc_seconds(observed_at_utc, "GCP read observed time")
    if not (
        observed_at_utc
        == result_preflight["observed_at_utc"]
        == quota["observed_at_utc"]
        == mapping["observed_at_utc"]
    ):
        raise ValueError("GCP read observation-time evidence chain changed")
    provider_metrics = receipt.get("provider_quota_metrics")
    if (
        not isinstance(provider_metrics, Mapping)
        or set(provider_metrics) != _PROVIDER_QUOTA_KEYS
    ):
        raise ValueError("GCP provider quota metric fields changed")
    accounts = receipt.get("service_accounts")
    if not isinstance(accounts, list) or len(accounts) != len(iam["workers"]):
        raise ValueError("GCP read service-account set changed")
    for worker, raw in zip(iam["workers"], accounts, strict=True):
        if not isinstance(raw, Mapping) or set(raw) != _SERVICE_ACCOUNT_ROW_KEYS:
            raise ValueError("GCP read service-account row fields changed")
        if (
            raw.get("job_id") != worker["job_id"]
            or raw.get("source_role") != worker["source_role"]
            or raw.get("service_account") != worker["service_account"]
            or not isinstance(raw.get("unique_id"), str)
            or _UNIQUE_ID.fullmatch(raw["unique_id"]) is None
            or raw.get("resource_name")
            not in {
                f"projects/{PROJECT}/serviceAccounts/{raw.get('service_account')}",
                f"projects/{PROJECT}/serviceAccounts/{raw['unique_id']}",
            }
            or raw.get("disabled") is not False
            or raw.get("exists") is not True
        ):
            raise ValueError("GCP read service-account identity changed")
    roles = receipt.get("custom_roles")
    if not isinstance(roles, list) or len(roles) != len(CUSTOM_ROLE_EXPECTATIONS):
        raise ValueError("GCP custom-role set changed")
    for (role_name, permissions), raw in zip(
        CUSTOM_ROLE_EXPECTATIONS, roles, strict=True
    ):
        if (
            not isinstance(raw, Mapping)
            or set(raw) != _CUSTOM_ROLE_ROW_KEYS
            or raw.get("role_name") != role_name
            or raw.get("included_permissions") != list(permissions)
            or raw.get("stage") != "GA"
            or raw.get("deleted") is not False
            or _SHA256.fullmatch(str(raw.get("etag_sha256"))) is None
        ):
            raise ValueError("GCP custom-role permission or lifecycle changed")
    expected_get_count = (
        7
        + 2 * len(resume["selected_attempts"])
        + len(CUSTOM_ROLE_EXPECTATIONS)
        + len(accounts)
        + result_preflight["http_get_count"]
    )
    if (
        receipt.get("schema") != READ_RECEIPT_SCHEMA
        or receipt.get("status") != "fresh_phase_a_readback_complete"
        or receipt.get("project") != PROJECT
        or receipt.get("region") != REGION
        or receipt.get("zone") != ZONE
        or receipt.get("bucket") != BUCKET
        or receipt.get("run_name") != plan["run_name"]
        or receipt.get("execution_identity_sha256")
        != plan["execution_identity_sha256"]
        or receipt.get("wave_plan_sha256") != plan["schedule_sha256"]
        or receipt.get("attempt_ledger_sha256") != ledger["ledger_sha256"]
        or receipt.get("resume_plan_sha256") != resume["resume_sha256"]
        or receipt.get("wave_index") != resume["resume_wave_index"]
        or receipt.get("immutable_content_prefix") != content_prefix
        or receipt.get("content_payload_sha256") != content_sha
        or receipt.get("outer_manifest_sha256") != manifest_sha
        or receipt.get("iam_plan_sha256") != iam["plan_sha256"]
        or receipt.get("custom_roles_sha256") != canonical_sha256(roles)
        or receipt.get("custom_role_count") != len(CUSTOM_ROLE_EXPECTATIONS)
        or provider_metrics["c4"] != C4_QUOTA_METRIC
        or provider_metrics["c4_quota_id"] != C4_QUOTA_ID
        or not isinstance(provider_metrics["c4_project_number"], str)
        or not provider_metrics["c4_project_number"].isdigit()
        or int(provider_metrics["c4_project_number"]) <= 0
        or _SHA256.fullmatch(
            str(provider_metrics["c4_usage_inventory_sha256"])
        )
        is None
        or provider_metrics["spot"] not in SPOT_QUOTA_METRICS
        or provider_metrics["global"] != GLOBAL_QUOTA_METRIC
        or provider_metrics["global_quota_id"] != GLOBAL_QUOTA_ID
        or _SHA256.fullmatch(
            str(provider_metrics["global_usage_inventory_sha256"])
        )
        is None
        or receipt.get("selected_vm_count") != len(resume["selected_attempts"])
        or receipt.get("http_get_count") != expected_get_count
        or receipt.get("all_selected_instances_absent") is not True
        or receipt.get("all_selected_boot_disks_absent") is not True
        or receipt.get("all_service_accounts_exist") is not True
        or receipt.get("all_custom_roles_exact_ga_not_deleted") is not True
        or receipt.get("pagination_observed")
        is not result_preflight["pagination_observed"]
        or receipt.get("read_only") is not True
        or receipt.get("cloud_mutation_performed") is not False
        or receipt.get("current_profile_changed") is not False
        or quota["quota_sufficient"] is not True
        or mapping["one_vm_one_job_one_role"] is not True
    ):
        raise ValueError("GCP read receipt evidence changed")
    return receipt


def validate_service_account_actas_receipt(
    *,
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    immutable_content_prefix: str,
    content_payload_sha256: str,
    outer_manifest_sha256: str,
    iam_plan: Mapping[str, Any],
    gcp_read_receipt: Mapping[str, Any],
    value: Mapping[str, Any],
    now_utc: str | None = None,
) -> dict[str, Any]:
    """Validate the non-mutating actAs proof consumed by a launch bundle."""

    (
        plan,
        ledger,
        resume,
        content_prefix,
        content_sha,
        manifest_sha,
    ) = _validated_context(
        wave_plan=wave_plan,
        attempt_ledger=attempt_ledger,
        resume_plan=resume_plan,
        immutable_content_prefix=immutable_content_prefix,
        content_payload_sha256=content_payload_sha256,
        outer_manifest_sha256=outer_manifest_sha256,
    )
    iam = worker_iam_v2.validate_worker_iam_plan(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        immutable_content_prefix=content_prefix,
        content_payload_sha256=content_sha,
        outer_manifest_sha256=manifest_sha,
        value=iam_plan,
    )
    read = validate_read_receipt(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        immutable_content_prefix=content_prefix,
        content_payload_sha256=content_sha,
        outer_manifest_sha256=manifest_sha,
        iam_plan=iam,
        value=gcp_read_receipt,
    )
    if not isinstance(value, Mapping) or set(value) != _ACTAS_KEYS:
        raise ValueError("service-account actAs receipt fields changed")
    receipt = deepcopy(dict(value))
    digest = receipt.pop("receipt_sha256", None)
    if digest != canonical_sha256(receipt):
        raise ValueError("service-account actAs receipt digest changed")
    receipt["receipt_sha256"] = digest
    rows = receipt.get("rows")
    accounts = read["service_accounts"]
    if not isinstance(rows, list) or len(rows) != len(accounts):
        raise ValueError("service-account actAs receipt cardinality changed")
    for account, raw in zip(accounts, rows, strict=True):
        if not isinstance(raw, Mapping) or set(raw) != _ACTAS_ROW_KEYS:
            raise ValueError("service-account actAs row fields changed")
        if (
            raw.get("job_id") != account["job_id"]
            or raw.get("source_role") != account["source_role"]
            or raw.get("service_account") != account["service_account"]
            or raw.get("resource_name") != account["resource_name"]
            or raw.get("unique_id") != account["unique_id"]
            or raw.get("requested_permissions") != [ACT_AS_PERMISSION]
            or raw.get("granted_permissions") != [ACT_AS_PERMISSION]
            or raw.get("act_as_granted") is not True
            or raw.get("test_complete") is not True
        ):
            raise ValueError("service-account actAs row identity changed")
    checked = _utc_seconds(receipt.get("checked_at_utc"), "actAs checked time")
    expires = _utc_seconds(receipt.get("expires_at_utc"), "actAs expiry")
    read_observed = max(
        _utc_seconds(read[name]["observed_at_utc"], f"{name} observed time")
        for name in ("quota_receipt", "planned_mapping_receipt")
    )
    read_expires = min(
        _utc_seconds(read[name]["expires_at_utc"], f"{name} expiry")
        for name in ("quota_receipt", "planned_mapping_receipt")
    )
    if (
        not 1 <= expires - checked <= MAX_ACTAS_RECEIPT_LIFETIME_SECONDS
        or checked < read_observed
        or expires > read_expires
    ):
        raise PermissionError("service-account actAs receipt is outside its read window")
    if now_utc is not None:
        now = _utc_seconds(now_utc, "actAs validation time")
        if not checked <= now <= expires:
            raise PermissionError("service-account actAs receipt is stale or future-dated")
    if (
        receipt.get("schema") != SERVICE_ACCOUNT_ACTAS_RECEIPT_SCHEMA
        or receipt.get("status")
        != "all_selected_worker_act_as_and_provider_permissions_granted"
        or receipt.get("project") != PROJECT
        or receipt.get("region") != REGION
        or receipt.get("zone") != ZONE
        or receipt.get("bucket") != BUCKET
        or receipt.get("run_name") != plan["run_name"]
        or receipt.get("execution_identity_sha256")
        != plan["execution_identity_sha256"]
        or receipt.get("wave_plan_sha256") != plan["schedule_sha256"]
        or receipt.get("attempt_ledger_sha256") != ledger["ledger_sha256"]
        or receipt.get("resume_plan_sha256") != resume["resume_sha256"]
        or receipt.get("wave_index") != resume["resume_wave_index"]
        or receipt.get("immutable_content_prefix") != content_prefix
        or receipt.get("content_payload_sha256") != content_sha
        or receipt.get("outer_manifest_sha256") != manifest_sha
        or receipt.get("iam_plan_sha256") != iam["plan_sha256"]
        or receipt.get("gcp_read_receipt_sha256") != read["receipt_sha256"]
        or receipt.get("permission") != ACT_AS_PERMISSION
        or receipt.get("selected_worker_count") != len(accounts)
        or receipt.get("http_post_count") != len(accounts) + 1
        or receipt.get("all_selected_workers_act_as_granted") is not True
        or receipt.get("provider_project_permissions_requested")
        != list(PROVIDER_PROJECT_PERMISSIONS)
        or receipt.get("provider_project_permissions_granted")
        != list(PROVIDER_PROJECT_PERMISSIONS)
        or receipt.get("provider_project_permission_count")
        != len(PROVIDER_PROJECT_PERMISSIONS)
        or receipt.get("all_provider_project_permissions_granted") is not True
        or receipt.get("permission_test_only") is not True
        or receipt.get("cloud_mutation_performed") is not False
        or receipt.get("current_profile_changed") is not False
    ):
        raise ValueError("service-account actAs receipt evidence changed")
    return receipt


__all__ = [
    "ACT_AS_PERMISSION",
    "BUCKET",
    "C4_QUOTA_ID",
    "C4_QUOTA_METRIC",
    "C4_QUOTA_VM_FAMILY",
    "CUSTOM_ROLE_EXPECTATIONS",
    "GLOBAL_QUOTA_ID",
    "GLOBAL_QUOTA_METRIC",
    "GcpWavePhaseAAdapter",
    "GcpPhaseATransportError",
    "HttpResponse",
    "MAX_ACTAS_RECEIPT_LIFETIME_SECONDS",
    "PROVIDER_PROJECT_PERMISSIONS",
    "PROJECT",
    "READ_RECEIPT_SCHEMA",
    "REGION",
    "SELECTED_RESULT_PREFLIGHT_RECEIPT_SCHEMA",
    "SERVICE_ACCOUNT_ACTAS_RECEIPT_SCHEMA",
    "SPOT_QUOTA_METRICS",
    "TOKEN_ENV",
    "ZONE",
    "canonical_bytes",
    "canonical_sha256",
    "validate_read_receipt",
    "validate_service_account_actas_receipt",
]
