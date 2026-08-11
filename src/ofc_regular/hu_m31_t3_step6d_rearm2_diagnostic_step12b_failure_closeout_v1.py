"""Read-only closeout verifier for the Step12b token-barrier failure.

The verifier consumes the durable ``step12b_pair_v2_actual`` failure
artifacts plus an injected, GET-only cloud observer.  It does not expose a
create, update, delete, retry, or VM-launch operation.  A successful receipt
proves that the failed attempt is closed without deleting the three fresh
bootstrap source objects or touching either immutable predecessor tree.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_token_barrier_failure_closeout_v1"
)
STATUS = "token_barrier_failure_closed_by_read_only_verification"
ROOT_CAUSE_CLASSIFICATION = (
    "empty_iam_policy_response_bindings_omission_normalization_mismatch"
)
FAILURE_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_pair_v2_actual_runner_failure_v1"
)
EXPECTED_POLICY_REGISTRY_SHA256 = (
    "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
)
EXPECTED_TOKEN_MINT_TIMESTAMP = "2026-07-20T09:43:49.190306825Z"
EXPECTED_TOKEN_CREATOR_REVOKE_TIMESTAMP = (
    "2026-07-20T09:43:56.190854856Z"
)
EXPECTED_CONTROLLER_DELETE_TIMESTAMP = "2026-07-20T09:44:12.574156296Z"
EXPECTED_REVOKE_POLICY_ETAG_SHA256 = (
    "5217d9966ccf5829355374286dbb5b8456e32b0bc968a6bb3f6ce176bc4d7015"
)
EXPECTED_OLD_TREE_KEYS = frozenset(
    {
        "step11_one_vm_v12_fix3_actual",
        "rearm2_diagnostic_cloud_worker_package_v1",
        "step12_pair_v1_actual",
    }
)
EXPECTED_OLD_TREE_DIGESTS = {
    "step11_one_vm_v12_fix3_actual": (
        "b019173b41aa4d12c49092dea68ddec98a563378f5547ffa84e236cd181108c1"
    ),
    "rearm2_diagnostic_cloud_worker_package_v1": (
        "2da83238f87083551ed45df95fb0cce10d4acf6fffd509ef51a22427b7f9248b"
    ),
    "step12_pair_v1_actual": (
        "5686398642c2795f60ae67bef99f3a3a3c650a69611f2c68748fcf7caeab8f5d"
    ),
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FAILURE_FIELDS = frozenset(
    {
        "schema",
        "status",
        "deployment_contract_sha256",
        "failure_stage",
        "exception_type",
        "failure_evidence_sha256",
        "pair_controller_owned_cleanup",
        "outer_cleanup_always_attempted",
        "outer_cleanup_call_returned",
        "outer_cleanup_verified",
        "outer_failure_cleanup_receipt_path",
        "outer_failure_cleanup_receipt_exists",
        "outer_cleanup_unverified_operations",
        "outer_cleanup_receipt_sha256",
        "pair_controller_failure_receipt_path",
        "pair_controller_failure_receipt_exists",
        "pair_controller_failure_receipt_sha256",
        "automatic_retry_performed",
        "attempt1_authorized",
        "third_vm_authorized",
        "policy_registry_sha256_before",
        "policy_registry_sha256_after",
        "current_profile_changed",
        "exception_message_stored",
        "private_key_stored",
        "access_token_stored",
        "receipt_sha256",
    }
)
_OUTER_CLEANUP_FIELDS = frozenset(
    {
        "schema",
        "status",
        "failure_evidence_sha256",
        "records",
        "records_sha256",
        "cleanup_order",
        "mandatory_verification_complete",
        "phase2_principals_zero_verified",
        "token_creator_zero_verified",
        "controller_service_account_get404_verified",
        "exact_instances_and_disks_get404_verified",
        "automatic_retry_performed",
        "attempt1_authorized",
        "third_vm_authorized",
        "current_profile_changed",
        "receipt_sha256",
    }
)
_FORBIDDEN_SECRET_FIELDS = frozenset(
    {
        "access_token",
        "authorization_header",
        "private_key",
        "private_key_pem",
        "raw_policy",
        "raw_response_body",
        "raw_token",
    }
)


class ReadOnlyCloseoutObserver(Protocol):
    """The complete cloud surface accepted by the closeout verifier."""

    def verify_exact_compute_absent(
        self,
        *,
        instance_names: Sequence[str],
        disk_names: Sequence[str],
    ) -> Mapping[str, Any]: ...

    def get_controller_service_account(
        self, *, email: str
    ) -> Mapping[str, Any] | None: ...

    def get_iam_policy(
        self, *, target: str
    ) -> Mapping[str, Any] | None: ...

    def list_direct_v2_stage_objects(
        self, *, stage_prefix: str
    ) -> Sequence[str]: ...

    def list_bootstrap_source_objects(
        self, *, source_prefix: str
    ) -> Sequence[str]: ...

    def read_bootstrap_source_generation(
        self, *, uri: str, generation: int
    ) -> Mapping[str, Any]: ...

    def read_token_barrier_audit_events(
        self, *, controller_unique_id: str
    ) -> Sequence[Mapping[str, Any]]: ...

    def audit_log(self) -> Sequence[str]: ...


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


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or _SHA256.fullmatch(value) is None
        or value == "0" * 64
    ):
        raise ValueError(f"{label} is not a nonzero lowercase SHA-256")
    return value


def _reject_secret_surface(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string field")
            if (
                key.lower() in _FORBIDDEN_SECRET_FIELDS
                and child is not False
                and child is not None
            ):
                raise ValueError(f"{path}.{key} contains secret material")
            _reject_secret_surface(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_secret_surface(child, f"{path}[{index}]")
    elif isinstance(value, str):
        if (
            "-----BEGIN PRIVATE KEY-----" in value
            or value.lower().startswith("bearer ")
        ):
            raise ValueError(f"{path} contains secret material")


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = copy.deepcopy(dict(value))
    _reject_secret_surface(body)
    return {**body, "receipt_sha256": canonical_sha256(body)}


def validate_sealed_receipt(
    value: Mapping[str, Any],
    *,
    label: str,
    exact_fields: frozenset[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not a receipt mapping")
    checked = copy.deepcopy(dict(value))
    if exact_fields is not None and set(checked) != exact_fields:
        raise ValueError(f"{label} fields changed")
    supplied = _sha(checked.pop("receipt_sha256", None), label)
    if canonical_sha256(checked) != supplied:
        raise ValueError(f"{label} sealed digest changed")
    restored = {**checked, "receipt_sha256": supplied}
    _reject_secret_surface(restored)
    return restored


def validate_closeout_receipt(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    checked = validate_sealed_receipt(value, label="closeout receipt")
    if (
        checked.get("schema") != SCHEMA
        or checked.get("status") != STATUS
        or checked.get("all_closeout_gates_passed") is not True
        or checked.get("cloud_read_only") is not True
        or checked.get("cloud_mutation_performed") is not False
        or checked.get("automatic_retry_performed") is not False
        or checked.get("attempt1_authorized") is not False
        or checked.get("third_vm_authorized") is not False
        or checked.get("source_objects_deleted") is not False
        or checked.get("old_v1_touched") is not False
        or checked.get("current_profile_changed") is not False
        or checked.get("root_cause_classification")
        != ROOT_CAUSE_CLASSIFICATION
        or checked.get("root_cause_verified") is not True
        or checked.get("root_cause_permission_propagation_failure")
        is not False
    ):
        raise ValueError("closeout receipt boundary changed")
    return checked


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label} is not a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _validate_self_digest(
    value: Mapping[str, Any],
    *,
    digest_field: str,
    label: str,
) -> dict[str, Any]:
    checked = copy.deepcopy(dict(value))
    supplied = _sha(checked.pop(digest_field, None), label)
    if canonical_sha256(checked) != supplied:
        raise ValueError(f"{label} digest changed")
    return {**checked, digest_field: supplied}


def _validate_local_failure_artifacts(
    output_root: Path,
) -> dict[str, Any]:
    failure = validate_sealed_receipt(
        _read_json(output_root / "FAILURE.json", "FAILURE"),
        label="FAILURE",
        exact_fields=_FAILURE_FIELDS,
    )
    cleanup = validate_sealed_receipt(
        _read_json(
            output_root / "outer_failure_cleanup_receipt.json",
            "outer cleanup receipt",
        ),
        label="outer cleanup receipt",
        exact_fields=_OUTER_CLEANUP_FIELDS,
    )
    deployment = _validate_self_digest(
        _read_json(
            output_root / "deployment_contract.json",
            "deployment contract",
        ),
        digest_field="deployment_contract_sha256",
        label="deployment contract",
    )
    plan = _validate_self_digest(
        _read_json(output_root / "phase2_iam_plan.json", "Phase2 plan"),
        digest_field="plan_sha256",
        label="Phase2 plan",
    )
    source = validate_sealed_receipt(
        _read_json(
            output_root / "bootstrap_source_provision_receipt.json",
            "bootstrap source provision receipt",
        ),
        label="bootstrap source provision receipt",
    )
    controller_create = validate_sealed_receipt(
        _read_json(
            output_root / "controller_service_account_create_receipt.json",
            "controller service-account create receipt",
        ),
        label="controller service-account create receipt",
    )
    manifests = validate_sealed_receipt(
        _read_json(
            output_root / "role_bootstrap_manifest_digests.json",
            "role bootstrap manifest digests",
        ),
        label="role bootstrap manifest digests",
    )

    deployment_sha = deployment["deployment_contract_sha256"]
    expected_cleanup_order = [
        "phase2_controller_then_worker_zero",
        "token_creator_residual_zero",
        "controller_service_account_absent",
        "exact_instances_and_disks_absent",
    ]
    records = cleanup.get("records")
    if (
        failure["schema"] != FAILURE_SCHEMA
        or failure["status"]
        != "step12b_exact_pair_stopped_without_retry"
        or failure["failure_stage"] != "token_barrier"
        or failure["exception_type"] != "TokenBarrierFailure"
        or failure["deployment_contract_sha256"] != deployment_sha
        or failure["outer_cleanup_receipt_sha256"]
        != cleanup["receipt_sha256"]
        or failure["outer_cleanup_unverified_operations"]
        != ["token_creator_residual_zero"]
        or failure["outer_cleanup_verified"] is not False
        or failure["outer_cleanup_always_attempted"] is not True
        or failure["outer_cleanup_call_returned"] is not True
        or failure["pair_controller_owned_cleanup"] is not False
        or failure["pair_controller_failure_receipt_exists"] is not False
        or failure["automatic_retry_performed"] is not False
        or failure["attempt1_authorized"] is not False
        or failure["third_vm_authorized"] is not False
        or failure["current_profile_changed"] is not False
        or cleanup["schema"] != FAILURE_SCHEMA
        or cleanup["status"] != "mandatory_failure_cleanup_incomplete"
        or cleanup["failure_evidence_sha256"]
        != failure["failure_evidence_sha256"]
        or cleanup["cleanup_order"] != expected_cleanup_order
        or cleanup["mandatory_verification_complete"] is not False
        or cleanup["token_creator_zero_verified"] is not False
        or cleanup["phase2_principals_zero_verified"] is not True
        or cleanup["controller_service_account_get404_verified"]
        is not True
        or cleanup["exact_instances_and_disks_get404_verified"]
        is not True
        or cleanup["automatic_retry_performed"] is not False
        or cleanup["attempt1_authorized"] is not False
        or cleanup["third_vm_authorized"] is not False
        or not isinstance(records, list)
        or canonical_sha256(records) != cleanup["records_sha256"]
        or [
            row.get("operation") if isinstance(row, Mapping) else None
            for row in records
        ]
        != [
            "phase2_not_installed",
            "token_creator_residual_zero",
            "controller_service_account_absent",
            "exact_instances_and_disks_absent",
        ]
        or [
            row.get("completed") if isinstance(row, Mapping) else None
            for row in records
        ]
        != [True, False, True, True]
    ):
        raise ValueError("token-barrier failure/cleanup evidence changed")

    instances = deployment.get("instances")
    phase2_bindings = plan.get("phase2_bindings")
    if (
        deployment.get("schema")
        != (
            "hu_m31_t3_step6d_rearm2_diagnostic_"
            "step12b_deployment_contract_v2"
        )
        or deployment.get("vm_count") != 2
        or deployment.get("attempt_index") != 0
        or not isinstance(instances, list)
        or len(instances) != 2
        or len(
            {
                row.get("instance_name")
                for row in instances
                if isinstance(row, Mapping)
            }
        )
        != 2
        or plan.get("source_deployment", {}).get(
            "deployment_contract_sha256"
        )
        != deployment_sha
        or not isinstance(phase2_bindings, Mapping)
        or not isinstance(phase2_bindings.get("controller"), list)
        or not isinstance(phase2_bindings.get("worker"), list)
        or len(phase2_bindings["controller"]) != 5
        or len(phase2_bindings["worker"]) != 3
        or phase2_bindings.get("controller_binding_count") != 5
        or phase2_bindings.get("worker_binding_count") != 3
        or phase2_bindings.get("total_binding_count") != 8
        or phase2_bindings.get("token_creator_role_forbidden")
        != "roles/iam.serviceAccountTokenCreator"
        or phase2_bindings.get("phase1_token_creator_binding_included")
        is not False
    ):
        raise ValueError("deployment/Phase2 failure identity changed")

    source_records = source.get("records")
    source_prefix = source.get("source_prefix")
    if (
        source.get("deployment_contract_sha256") != deployment_sha
        or source.get("status")
        != "fresh_direct_v2_bootstrap_sources_created_and_read_back"
        or source.get("object_count") != 3
        or source.get("all_generation_bound") is not True
        or source.get("all_bytes_and_sha256_read_back") is not True
        or source.get("direct_v2_result_write_count") != 0
        or not isinstance(source_prefix, str)
        or source_prefix
        != (
            "gs://pokerhu-ofc-solver-485418-training/"
            "hu-m31-r2diag-direct-v2/bootstrap-sources/"
            f"{deployment_sha}"
        )
        or not isinstance(source_records, list)
        or len(source_records) != 3
        or canonical_sha256(source_records) != source["records_sha256"]
        or len(
            {
                row.get("uri")
                for row in source_records
                if isinstance(row, Mapping)
            }
        )
        != 3
    ):
        raise ValueError("bootstrap source provision evidence changed")
    for row in source_records:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("uri"), str)
            or not row["uri"].startswith(source_prefix.rstrip("/") + "/")
            or type(row.get("generation")) is not int
            or row["generation"] <= 0
            or type(row.get("bytes")) is not int
            or row["bytes"] <= 0
            or _SHA256.fullmatch(str(row.get("sha256"))) is None
            or row.get("readback_verified") is not True
        ):
            raise ValueError("bootstrap source record changed")

    controller_identity = deployment.get("controller_service_account")
    if (
        not isinstance(controller_identity, Mapping)
        or controller_identity.get("run_scoped") is not True
        or controller_create.get("email") != controller_identity.get("email")
        or controller_create.get("status")
        != "run_scoped_controller_service_account_created"
        or controller_create.get("readback_verified") is not True
        or manifests.get("manifest_count") != 2
        or manifests.get("opponent_payload_embedded_in_role_metadata")
        is not False
    ):
        raise ValueError("controller/source manifest evidence changed")

    return {
        "failure": failure,
        "cleanup": cleanup,
        "deployment": deployment,
        "plan": plan,
        "source": source,
        "controller_create": controller_create,
        "manifests": manifests,
    }


def _condition_title(value: Any) -> str:
    if (
        not isinstance(value, Mapping)
        or not isinstance(value.get("title"), str)
        or not value["title"]
    ):
        raise ValueError("Phase2 condition title changed")
    return value["title"]


def _policy_binding_rows(
    observer: ReadOnlyCloseoutObserver,
    plan: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str | None]]:
    expected = [
        *plan["phase2_bindings"]["controller"],
        *plan["phase2_bindings"]["worker"],
    ]
    targets = sorted({row["target"] for row in expected})
    policies: dict[str, Mapping[str, Any] | None] = {}
    policy_shas: dict[str, str | None] = {}
    for target in targets:
        policy = observer.get_iam_policy(target=target)
        if policy is not None and not isinstance(policy, Mapping):
            raise ValueError("IAM policy readback changed")
        policies[target] = policy
        policy_shas[target] = (
            canonical_sha256(dict(policy)) if policy is not None else None
        )

    rows = []
    seen_titles: set[str] = set()
    for binding in expected:
        if not isinstance(binding, Mapping):
            raise ValueError("Phase2 binding identity changed")
        title = _condition_title(binding.get("condition"))
        if title in seen_titles:
            raise ValueError("Phase2 condition title collided")
        seen_titles.add(title)
        policy = policies[binding["target"]]
        occurrences = 0
        if policy is not None:
            raw_bindings = policy.get("bindings", [])
            if not isinstance(raw_bindings, list):
                raise ValueError("IAM policy bindings changed")
            for raw in raw_bindings:
                if not isinstance(raw, Mapping):
                    raise ValueError("IAM policy binding changed")
                members = raw.get("members", [])
                condition = raw.get("condition")
                if not isinstance(members, list) or any(
                    not isinstance(member, str) for member in members
                ):
                    raise ValueError("IAM policy members changed")
                raw_title = (
                    condition.get("title")
                    if isinstance(condition, Mapping)
                    else None
                )
                if (
                    raw.get("role") == binding["role"]
                    and raw_title == title
                ):
                    occurrences += members.count(binding["member"])
        if occurrences != 0:
            raise ValueError(
                f"Phase2 condition-title binding remained: {title}"
            )
        rows.append(
            {
                "purpose": binding["purpose"],
                "target": binding["target"],
                "role": binding["role"],
                "member": binding["member"],
                "condition_title": title,
                "binding_sha256": binding["binding_sha256"],
                "member_occurrences": 0,
                "target_policy_get_status": (
                    200 if policy is not None else 404
                ),
                "readback_complete": True,
            }
        )
    if len(rows) != 8:
        raise ValueError("Phase2 condition-title readback count changed")
    return rows, policy_shas


def build_local_tree_snapshot(root: Path) -> dict[str, Any]:
    resolved = Path(root).resolve()
    if not resolved.is_dir() or resolved.is_symlink():
        raise ValueError(f"immutable tree is not a regular directory: {root}")
    rows = []
    for path in sorted(resolved.rglob("*"), key=lambda row: row.as_posix()):
        if path.is_symlink():
            raise ValueError(f"immutable tree contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"immutable tree contains a special file: {path}")
        raw = path.read_bytes()
        rows.append(
            {
                "path": path.relative_to(resolved).as_posix(),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    if not rows:
        raise ValueError(f"immutable tree is empty: {root}")
    return {
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "tree_sha256": canonical_sha256(rows),
        "regular_files_only": True,
        "symlink_count": 0,
    }


def _verify_old_trees(
    *,
    old_tree_roots: Mapping[str, Path],
    expected_old_tree_digests: Mapping[str, str],
) -> list[dict[str, Any]]:
    if (
        set(old_tree_roots) != EXPECTED_OLD_TREE_KEYS
        or set(expected_old_tree_digests) != EXPECTED_OLD_TREE_KEYS
    ):
        raise ValueError("immutable predecessor tree set changed")
    records = []
    for name in sorted(EXPECTED_OLD_TREE_KEYS):
        expected = _sha(
            expected_old_tree_digests[name],
            f"{name} expected tree",
        )
        snapshot = build_local_tree_snapshot(old_tree_roots[name])
        if snapshot["tree_sha256"] != expected:
            raise ValueError(f"immutable predecessor tree changed: {name}")
        records.append(
            {
                "name": name,
                **snapshot,
                "expected_tree_sha256": expected,
                "unchanged": True,
            }
        )
    return records


def _verify_source_objects(
    observer: ReadOnlyCloseoutObserver,
    source_receipt: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = list(source_receipt["records"])
    expected_uris = sorted(row["uri"] for row in records)
    listed = observer.list_bootstrap_source_objects(
        source_prefix=source_receipt["source_prefix"]
    )
    if (
        isinstance(listed, (str, bytes))
        or not isinstance(listed, Sequence)
        or sorted(listed) != expected_uris
        or len(set(listed)) != 3
    ):
        raise ValueError("fresh bootstrap source object inventory changed")
    readbacks = []
    by_uri = {row["uri"]: row for row in records}
    for uri in expected_uris:
        expected = by_uri[uri]
        observed = observer.read_bootstrap_source_generation(
            uri=uri, generation=expected["generation"]
        )
        if not isinstance(observed, Mapping):
            raise ValueError("bootstrap source generation readback changed")
        if (
            observed.get("uri") != uri
            or observed.get("generation") != expected["generation"]
            or observed.get("bytes") != expected["bytes"]
            or observed.get("sha256") != expected["sha256"]
            or observed.get("readback_complete") is not True
        ):
            raise ValueError("bootstrap source generation identity changed")
        readbacks.append(
            {
                "uri": uri,
                "generation": expected["generation"],
                "bytes": expected["bytes"],
                "sha256": expected["sha256"],
                "retained": True,
                "generation_pinned_readback_complete": True,
            }
        )
    return readbacks


def _expected_root_cause_audit_events(
    controller_unique_id: str,
) -> list[dict[str, Any]]:
    if (
        not isinstance(controller_unique_id, str)
        or not controller_unique_id.isdecimal()
        or not 6 <= len(controller_unique_id) <= 32
    ):
        raise ValueError("controller service-account unique ID changed")
    resource = (
        "projects/-/serviceAccounts/" + controller_unique_id
    )
    empty_response = {
        "response_policy_present": False,
        "response_policy_etag_present": False,
        "response_policy_etag_sha256": None,
        "response_policy_version": None,
        "response_policy_bindings_field_present": None,
        "response_policy_binding_count": None,
    }
    return [
        {
            "event_role": "controller_token_mint",
            "timestamp": EXPECTED_TOKEN_MINT_TIMESTAMP,
            "service_name": "iamcredentials.googleapis.com",
            "method_name": "GenerateAccessToken",
            "resource_name": resource,
            "status_code": 0,
            "successful": True,
            **empty_response,
        },
        {
            "event_role": "token_creator_revoke",
            "timestamp": EXPECTED_TOKEN_CREATOR_REVOKE_TIMESTAMP,
            "service_name": "iam.googleapis.com",
            "method_name": "google.iam.admin.v1.SetIAMPolicy",
            "resource_name": resource,
            "status_code": 0,
            "successful": True,
            "response_policy_present": True,
            "response_policy_etag_present": True,
            "response_policy_etag_sha256": (
                EXPECTED_REVOKE_POLICY_ETAG_SHA256
            ),
            "response_policy_version": 1,
            "response_policy_bindings_field_present": False,
            "response_policy_binding_count": 0,
        },
        {
            "event_role": "controller_service_account_delete",
            "timestamp": EXPECTED_CONTROLLER_DELETE_TIMESTAMP,
            "service_name": "iam.googleapis.com",
            "method_name": (
                "google.iam.admin.v1.DeleteServiceAccount"
            ),
            "resource_name": resource,
            "status_code": 0,
            "successful": True,
            **empty_response,
        },
    ]


def _verify_root_cause_audit(
    observer: ReadOnlyCloseoutObserver,
    controller_create_receipt: Mapping[str, Any],
) -> list[dict[str, Any]]:
    provider = controller_create_receipt.get("provider")
    if not isinstance(provider, Mapping):
        raise ValueError("controller create provider identity changed")
    unique_id = provider.get("unique_id")
    expected = _expected_root_cause_audit_events(unique_id)
    observed = observer.read_token_barrier_audit_events(
        controller_unique_id=unique_id
    )
    if (
        isinstance(observed, (str, bytes))
        or not isinstance(observed, Sequence)
        or any(not isinstance(row, Mapping) for row in observed)
    ):
        raise ValueError("token-barrier Cloud Audit evidence changed")
    normalized = [copy.deepcopy(dict(row)) for row in observed]
    if normalized != expected:
        raise ValueError("token-barrier Cloud Audit evidence changed")
    return normalized


def verify_token_barrier_failure_closeout(
    *,
    output_root: Path,
    observer: ReadOnlyCloseoutObserver,
    policy_registry_path: Path,
    old_tree_roots: Mapping[str, Path],
    expected_old_tree_digests: Mapping[str, str],
) -> dict[str, Any]:
    """Reconcile every required surface without a cloud mutation or retry."""

    root = Path(output_root).resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Step12b failure output root changed")
    local = _validate_local_failure_artifacts(root)
    deployment = local["deployment"]
    plan = local["plan"]
    source = local["source"]
    failure = local["failure"]

    instance_names = [
        row["instance_name"] for row in deployment["instances"]
    ]
    compute = observer.verify_exact_compute_absent(
        instance_names=instance_names,
        disk_names=instance_names,
    )
    if (
        not isinstance(compute, Mapping)
        or compute.get("instance_final_statuses") != [404, 404]
        or compute.get("disk_final_statuses") != [404, 404]
        or compute.get("all_four_targets_get404_verified") is not True
    ):
        raise ValueError("exact two VM/two disk GET-404 proof changed")
    compute_sha = canonical_sha256(dict(compute))

    controller_email = deployment["controller_service_account"]["email"]
    if observer.get_controller_service_account(email=controller_email) is not None:
        raise ValueError("run-scoped controller service account still exists")

    binding_rows, policy_shas = _policy_binding_rows(observer, plan)

    stage_prefix = deployment["remote_layout"]["stage_prefix"]
    result_objects = observer.list_direct_v2_stage_objects(
        stage_prefix=stage_prefix
    )
    if (
        isinstance(result_objects, (str, bytes))
        or not isinstance(result_objects, Sequence)
        or list(result_objects) != []
    ):
        raise ValueError("direct-v2 stage/result objects are not zero")

    source_readbacks = _verify_source_objects(observer, source)
    root_cause_audit = _verify_root_cause_audit(
        observer, local["controller_create"]
    )

    profile = Path(policy_registry_path)
    if not profile.is_file() or profile.is_symlink():
        raise ValueError("current profile registry is not a regular file")
    profile_sha = hashlib.sha256(profile.read_bytes()).hexdigest()
    if (
        profile_sha != EXPECTED_POLICY_REGISTRY_SHA256
        or profile_sha != deployment["current_profile_sha256"]
        or profile_sha != failure["policy_registry_sha256_before"]
        or profile_sha != failure["policy_registry_sha256_after"]
    ):
        raise ValueError("current profile registry changed")

    old_trees = _verify_old_trees(
        old_tree_roots=old_tree_roots,
        expected_old_tree_digests=expected_old_tree_digests,
    )

    audit = observer.audit_log()
    allowed_operations = {
        "compute_exact_four_get",
        "controller_service_account_get",
        "iam_policy_get",
        "direct_v2_stage_list",
        "bootstrap_source_list",
        "bootstrap_source_generation_get",
        "cloud_audit_log_read",
    }
    if (
        isinstance(audit, (str, bytes))
        or not isinstance(audit, Sequence)
        or not audit
        or any(
            not isinstance(operation, str)
            or operation not in allowed_operations
            for operation in audit
        )
    ):
        raise ValueError("closeout observer operation audit changed")

    body = {
        "schema": SCHEMA,
        "status": STATUS,
        "deployment_contract_sha256": deployment[
            "deployment_contract_sha256"
        ],
        "failure_receipt_sha256": failure["receipt_sha256"],
        "outer_failure_cleanup_receipt_sha256": local["cleanup"][
            "receipt_sha256"
        ],
        "failure_evidence_sha256": failure["failure_evidence_sha256"],
        "failure_stage": "token_barrier",
        "original_unverified_operation": "token_creator_residual_zero",
        "root_cause_classification": ROOT_CAUSE_CLASSIFICATION,
        "root_cause_verified": True,
        "root_cause_permission_propagation_failure": False,
        "root_cause_token_mint_succeeded": True,
        "root_cause_token_creator_revoke_succeeded": True,
        "root_cause_empty_policy_bindings_omitted": True,
        "root_cause_controller_service_account_delete_succeeded": True,
        "root_cause_required_code_correction": (
            "normalize_missing_iam_policy_bindings_to_empty_list"
        ),
        "root_cause_cloud_audit_events": root_cause_audit,
        "root_cause_cloud_audit_events_sha256": canonical_sha256(
            root_cause_audit
        ),
        "token_creator_zero_proven_by_controller_sa_get404": True,
        "controller_service_account": controller_email,
        "controller_service_account_get_status": 404,
        "controller_service_account_absent": True,
        "instance_names": instance_names,
        "disk_names": instance_names,
        "compute_absence_readback_sha256": compute_sha,
        "instance_final_statuses": [404, 404],
        "disk_final_statuses": [404, 404],
        "exact_two_instances_and_two_disks_absent": True,
        "phase2_iam_plan_sha256": plan["plan_sha256"],
        "phase2_condition_title_readbacks": binding_rows,
        "phase2_condition_title_readbacks_sha256": canonical_sha256(
            binding_rows
        ),
        "phase2_target_policy_sha256s": policy_shas,
        "phase2_condition_title_binding_count": 8,
        "phase2_controller_binding_count": 5,
        "phase2_worker_binding_count": 3,
        "all_phase2_condition_title_bindings_zero": True,
        "direct_v2_stage_prefix": stage_prefix,
        "direct_v2_result_object_count": 0,
        "direct_v2_result_objects_zero": True,
        "bootstrap_source_provision_receipt_sha256": source[
            "receipt_sha256"
        ],
        "bootstrap_source_prefix": source["source_prefix"],
        "bootstrap_source_readbacks": source_readbacks,
        "bootstrap_source_readbacks_sha256": canonical_sha256(
            source_readbacks
        ),
        "bootstrap_source_object_count": 3,
        "bootstrap_source_objects_retained": True,
        "bootstrap_source_generations_pinned": True,
        "source_objects_deleted": False,
        "old_tree_readbacks": old_trees,
        "old_tree_readbacks_sha256": canonical_sha256(old_trees),
        "old_immutable_local_trees_unchanged": True,
        "old_v1_touched": False,
        "policy_registry_sha256": profile_sha,
        "current_profile_changed": False,
        "read_only_operations": list(audit),
        "read_only_operation_count": len(audit),
        "cloud_read_only": True,
        "cloud_mutation_performed": False,
        "vm_insert_attempt_count": 0,
        "automatic_retry_performed": False,
        "attempt1_authorized": False,
        "third_vm_authorized": False,
        "all_closeout_gates_passed": True,
        "diagnostic_only": True,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
        "access_token_stored": False,
        "authorization_header_stored": False,
        "private_key_stored": False,
        "raw_policy_stored": False,
    }
    return _seal(body)


__all__ = [
    "EXPECTED_OLD_TREE_KEYS",
    "EXPECTED_OLD_TREE_DIGESTS",
    "EXPECTED_POLICY_REGISTRY_SHA256",
    "FAILURE_SCHEMA",
    "ROOT_CAUSE_CLASSIFICATION",
    "ReadOnlyCloseoutObserver",
    "SCHEMA",
    "STATUS",
    "build_local_tree_snapshot",
    "canonical_bytes",
    "canonical_sha256",
    "validate_closeout_receipt",
    "validate_sealed_receipt",
    "verify_token_barrier_failure_closeout",
]
