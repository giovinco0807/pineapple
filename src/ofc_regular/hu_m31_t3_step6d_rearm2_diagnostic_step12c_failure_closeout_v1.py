"""Independent read-only closeout for the terminal Step12c Phase2 failure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_failure_closeout_v1
    as base_closeout,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_phase2_iam_lifecycle_v2
    as phase2_lifecycle,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_token_barrier_v1
    as token_barrier,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12c_fresh_identity_v1
    as fresh_identity,
)


SCHEMA = "hu_m31_t3_step6d_step12c_phase2_schema_failure_closeout_v1"
STATUS = "phase2_token_receipt_schema_failure_closed_read_only"
EXPECTED_PROFILE_SHA256 = (
    "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
)


class ReadOnlyObserver(Protocol):
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


def _seal(body: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(body)
    return {**copied, "receipt_sha256": canonical_sha256(copied)}


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label} is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} changed")
    return value


def validate_closeout_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    checked = base_closeout.validate_sealed_receipt(
        value, label="Step12c closeout receipt"
    )
    if (
        checked.get("schema") != SCHEMA
        or checked.get("status") != STATUS
        or checked.get("all_closeout_gates_passed") is not True
        or checked.get("all_phase2_condition_title_bindings_zero") is not True
        or checked.get("exact_two_instances_and_two_disks_absent") is not True
        or checked.get("controller_service_account_absent") is not True
        or checked.get("direct_v2_result_objects_zero") is not True
        or checked.get("bootstrap_source_object_count") != 3
        or checked.get("performance_evidence") is not False
        or checked.get("quality_evidence") is not False
        or checked.get("training_eligible") is not False
        or checked.get("promotion_evidence") is not False
        or checked.get("cloud_mutation_performed_by_closeout") is not False
        or checked.get("policy_registry_sha256") != EXPECTED_PROFILE_SHA256
        or checked.get("current_profile_changed") is not False
        or checked.get("attempt1_authorized") is not False
        or checked.get("automatic_retry_performed") is not False
        or checked.get("third_vm_authorized") is not False
        or checked.get("read_only_operation_count")
        != len(checked.get("read_only_operations", []))
    ):
        raise ValueError("Step12c closeout boundary changed")
    return checked


def build_closeout_receipt(
    *,
    output_root: Path,
    observer: ReadOnlyObserver,
    policy_registry_path: Path,
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    if root != fresh_identity.EXPECTED_OUTPUT_ROOT.resolve():
        raise ValueError("Step12c closeout root changed")
    deployment = _read_json(root / "deployment_contract.json", "deployment")
    plan = _read_json(root / "phase2_iam_plan.json", "Phase2 plan")
    failure = base_closeout.validate_sealed_receipt(
        _read_json(root / "FAILURE.json", "runner failure"),
        label="Step12c runner failure",
    )
    wrapper_failure = base_closeout.validate_sealed_receipt(
        _read_json(root / "STEP12C_FAILURE.json", "wrapper failure"),
        label="Step12c wrapper failure",
    )
    outer_cleanup = base_closeout.validate_sealed_receipt(
        _read_json(root / "outer_failure_cleanup_receipt.json", "outer cleanup"),
        label="Step12c outer cleanup",
    )
    source = base_closeout.validate_sealed_receipt(
        _read_json(
            root / "bootstrap_source_provision_receipt.json",
            "bootstrap source receipt",
        ),
        label="Step12c bootstrap source receipt",
    )
    token_receipt = base_closeout.validate_sealed_receipt(
        _read_json(root / "token_barrier_receipt.json", "token receipt"),
        label="Step12c token receipt",
    )
    if (
        failure.get("failure_stage") != "phase2_install"
        or failure.get("exception_type") != "ValueError"
        or failure.get("automatic_retry_performed") is not False
        or failure.get("attempt1_authorized") is not False
        or failure.get("third_vm_authorized") is not False
        or wrapper_failure.get("status") != "step12c_stopped_without_retry"
        or wrapper_failure.get("exception_message_stored") is not False
        or (root / "phase2_install_receipt.json").exists()
        or (root / "pair_controller_receipt.json").exists()
        or (root / "FINAL.json").exists()
    ):
        raise ValueError("Step12c local failure boundary changed")
    outcome = token_barrier.TokenBarrierOutcome(object(), token_receipt)
    token_sha = phase2_lifecycle._validate_token_barrier(outcome, plan)

    names = [row["instance_name"] for row in deployment["instances"]]
    compute = observer.verify_exact_compute_absent(
        instance_names=names, disk_names=names
    )
    if (
        compute.get("all_four_targets_get404_verified") is not True
        or compute.get("instance_final_statuses") != [404, 404]
        or compute.get("disk_final_statuses") != [404, 404]
    ):
        raise ValueError("Step12c compute absence changed")
    controller_email = deployment["controller_service_account"]["email"]
    if observer.get_controller_service_account(email=controller_email) is not None:
        raise ValueError("Step12c controller service account remained")
    binding_rows, policy_shas = base_closeout._policy_binding_rows(
        observer, plan
    )
    stage_objects = observer.list_direct_v2_stage_objects(
        stage_prefix=deployment["remote_layout"]["stage_prefix"]
    )
    if list(stage_objects) != []:
        raise ValueError("Step12c result objects remained")
    source_readbacks = base_closeout._verify_source_objects(observer, source)

    if (
        not policy_registry_path.is_file()
        or policy_registry_path.is_symlink()
    ):
        raise ValueError("profile registry changed")
    profile_sha = hashlib.sha256(policy_registry_path.read_bytes()).hexdigest()
    if profile_sha != EXPECTED_PROFILE_SHA256:
        raise ValueError("profile registry hash changed")
    terminal_after = fresh_identity.terminal_tree_snapshot()
    if terminal_after != wrapper_failure.get("terminal_tree"):
        raise ValueError("terminal Step12b tree changed")
    zero = token_receipt["token_creator_revoke_zero_readback"]
    operations = list(observer.audit_log())

    body = {
        "schema": SCHEMA,
        "status": STATUS,
        "all_closeout_gates_passed": True,
        "deployment_contract_sha256": deployment[
            "deployment_contract_sha256"
        ],
        "failure_receipt_sha256": failure["receipt_sha256"],
        "wrapper_failure_receipt_sha256": wrapper_failure[
            "receipt_sha256"
        ],
        "outer_failure_cleanup_receipt_sha256": outer_cleanup[
            "receipt_sha256"
        ],
        "outer_cleanup_originally_complete": outer_cleanup.get(
            "mandatory_verification_complete"
        ),
        "root_cause_classification": (
            "token_success_receipt_zero_readback_fields_missing_from_"
            "phase2_validator_allowlist"
        ),
        "required_code_correction": (
            "validate_token_creator_zero_readback_fields_before_phase2"
        ),
        "token_barrier_receipt_sha256": token_sha,
        "token_creator_zero_observed": zero["zero_observed"],
        "token_creator_second_add_performed": zero[
            "second_add_performed"
        ],
        "controller_token_reminted": zero["controller_token_reminted"],
        "phase2_install_receipt_present": False,
        "phase2_condition_title_readbacks": binding_rows,
        "phase2_condition_title_readbacks_sha256": canonical_sha256(
            binding_rows
        ),
        "phase2_target_policy_sha256s": policy_shas,
        "phase2_condition_title_binding_count": 8,
        "all_phase2_condition_title_bindings_zero": True,
        "instance_names": names,
        "disk_names": names,
        "instance_final_statuses": [404, 404],
        "disk_final_statuses": [404, 404],
        "exact_two_instances_and_two_disks_absent": True,
        "controller_service_account": controller_email,
        "controller_service_account_absent": True,
        "direct_v2_stage_prefix": deployment["remote_layout"][
            "stage_prefix"
        ],
        "direct_v2_result_object_count": 0,
        "direct_v2_result_objects_zero": True,
        "bootstrap_source_prefix": source["source_prefix"],
        "bootstrap_source_object_count": len(source_readbacks),
        "bootstrap_source_readbacks": source_readbacks,
        "bootstrap_source_readbacks_sha256": canonical_sha256(
            source_readbacks
        ),
        "bootstrap_source_objects_retained": True,
        "terminal_step12b_tree": terminal_after,
        "terminal_step12b_tree_unchanged": True,
        "policy_registry_sha256": profile_sha,
        "read_only_operations": operations,
        "read_only_operation_count": len(operations),
        "cloud_mutation_performed_by_closeout": False,
        "automatic_retry_performed": False,
        "attempt1_authorized": False,
        "third_vm_authorized": False,
        "performance_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
        "access_token_stored": False,
        "authorization_header_stored": False,
        "private_key_stored": False,
        "current_profile_changed": False,
    }
    return validate_closeout_receipt(_seal(body))


__all__ = [
    "EXPECTED_PROFILE_SHA256",
    "SCHEMA",
    "STATUS",
    "build_closeout_receipt",
    "canonical_sha256",
    "validate_closeout_receipt",
]
