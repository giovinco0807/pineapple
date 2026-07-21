"""Cloud-client-neutral lifecycle for the exact Step12b Phase-2 IAM set.

The caller injects the already narrow :class:`Step11RestIamAdmin`.  This
module performs no work at import time and has no VM-insert or TokenCreator
surface.

The sequence is intentionally rigid:

1. validate a successful token barrier whose TokenCreator binding is gone;
2. prove all targeted controller/worker bindings are initially absent;
3. install the three worker and five controller bindings (launch last);
4. GET-read back the exact eight-binding set;
5. after both provider claim-CAS readbacks, remove all five controller
   bindings and produce the receipt consumed by ``pair_release``;
6. remove all three worker bindings only at final cleanup.

Any failure after mutation begins performs best-effort fail-closed cleanup in
the order controller bindings first, worker bindings second.  A stale initial
binding is never cleared-and-continued.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_rest_iam_admin_v1
    as rest_iam,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_pair_release_v2
    as pair_release,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_phase2_iam_plan_v2
    as phase2_iam,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_token_barrier_v1
    as token_barrier,
)


INSTALL_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_phase2_iam_install_receipt_v2"
)
CONTROLLER_ZERO_LIFECYCLE_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_phase2_controller_zero_lifecycle_receipt_v2"
)
WORKER_ZERO_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_phase2_worker_zero_receipt_v2"
)
FAILURE_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_phase2_iam_lifecycle_failure_v2"
)
PAIR_CLAIMS_CAPABILITY_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_pair_claim_cas_readbacks_capability_v2"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_INSTALLED_SEAL = object()
_PAIR_CLAIMS_SEAL = object()
_CONTROLLER_ZERO_SEAL = object()
_WORKER_ZERO_SEAL = object()

_TOKEN_SUCCESS_FIELDS = {
    "schema",
    "status",
    "controller_service_account",
    "run_scoped_controller_required",
    "binding_purpose",
    "binding_role",
    "binding_member",
    "binding_condition_sha256",
    "binding_add_attempts",
    "binding_add_outcome",
    "binding_add_readback_sha256",
    "token_attempt_count",
    "failed_token_attempt_count",
    "token_attempts",
    "started_at_unix_seconds",
    "finished_at_unix_seconds",
    "elapsed_seconds",
    "maximum_propagation_seconds",
    "token_lifetime_seconds",
    "token_expire_time",
    "fixed_nonrefreshing_controller_credential",
    "controller_token_remint_forbidden",
    "token_creator_revoke_attempts",
    "token_creator_revoke_changed",
    "token_creator_revoke_readback_sha256",
    "token_creator_revoke_zero_readback",
    "token_creator_revoke_zero_readback_evidence_sha256",
    "token_creator_live_after_barrier",
    "phase2_started",
    "phase2_must_exclude_token_creator",
    "run_scoped_controller_delete_after_pair_claim_required",
    "vm_insert_attempt_count",
    "access_token_stored",
    "authorization_header_stored",
    "response_body_stored",
    "cloud_mutation_performed",
    "current_profile_changed",
    "receipt_sha256",
}


@dataclass(frozen=True)
class Phase2IamInstalledCapability:
    deployment_contract_sha256: str
    phase2_iam_plan_sha256: str
    token_barrier_receipt_sha256: str
    exact_readback_receipt_sha256: str
    install_receipt: Mapping[str, Any]
    _exact_readback_receipt_bytes: bytes = field(
        repr=False, compare=False
    )
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _INSTALLED_SEAL:
            raise ValueError("Phase2IamInstalledCapability cannot be forged")
        if not isinstance(self._exact_readback_receipt_bytes, bytes):
            raise ValueError("Phase2 IAM exact readback storage changed")
        try:
            stored = json.loads(
                self._exact_readback_receipt_bytes.decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                "Phase2 IAM exact readback storage changed"
            ) from error
        if (
            not isinstance(stored, dict)
            or phase2_iam.canonical_bytes(stored)
            != self._exact_readback_receipt_bytes
            or stored.get("receipt_sha256")
            != self.exact_readback_receipt_sha256
        ):
            raise ValueError("Phase2 IAM exact readback storage changed")


@dataclass(frozen=True)
class PairClaimCasReadbacksCapability:
    deployment_contract_sha256: str
    external_job_ids: tuple[str, str]
    claim_readback_receipt_sha256s: tuple[str, str]
    capability_sha256: str
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _PAIR_CLAIMS_SEAL:
            raise ValueError(
                "PairClaimCasReadbacksCapability cannot be forged"
            )


@dataclass(frozen=True)
class ControllerBindingsZeroCapability:
    deployment_contract_sha256: str
    phase2_iam_plan_sha256: str
    phase2_zero_receipt: Mapping[str, Any]
    lifecycle_receipt: Mapping[str, Any]
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _CONTROLLER_ZERO_SEAL:
            raise ValueError(
                "ControllerBindingsZeroCapability cannot be forged"
            )


@dataclass(frozen=True)
class WorkerBindingsZeroCapability:
    deployment_contract_sha256: str
    phase2_iam_plan_sha256: str
    receipt: Mapping[str, Any]
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _WORKER_ZERO_SEAL:
            raise ValueError(
                "WorkerBindingsZeroCapability cannot be forged"
            )


class Phase2IamLifecycleError(RuntimeError):
    """Fail-closed lifecycle error containing only a sealed receipt."""

    def __init__(self, receipt: Mapping[str, Any]) -> None:
        super().__init__("step12b_phase2_iam_lifecycle_failed")
        self.receipt = copy.deepcopy(dict(receipt))

    def __str__(self) -> str:
        return "step12b_phase2_iam_lifecycle_failed"


class _LifecycleAbort(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def canonical_sha256(value: Any) -> str:
    return phase2_iam.canonical_sha256(value)


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or _SHA256.fullmatch(value) is None
        or value == "0" * 64
    ):
        raise ValueError(f"{label} is not a nonzero lowercase SHA-256")
    return value


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = copy.deepcopy(dict(value))
    return {**body, "receipt_sha256": canonical_sha256(body)}


def _validate_sealed(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    checked = copy.deepcopy(dict(value))
    supplied = _sha(checked.pop("receipt_sha256", None), label)
    if canonical_sha256(checked) != supplied:
        raise ValueError(f"{label} digest changed")
    return dict(value)


def _reject_sensitive_readback_fields(
    value: Any, path: str = "$"
) -> None:
    forbidden = {
        "access_token",
        "authorization_header",
        "private_key",
        "raw_policy",
        "raw_response_body",
        "response_body",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("Phase2 IAM readback key changed")
            if key.lower() in forbidden:
                raise ValueError(
                    f"sensitive Phase2 IAM readback field at {path}.{key}"
                )
            _reject_sensitive_readback_fields(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for position, child in enumerate(value):
            _reject_sensitive_readback_fields(
                child, f"{path}[{position}]"
            )


def _validate_plan(
    phase2_iam_plan: Mapping[str, Any],
    *,
    deployment_contract: Mapping[str, Any],
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
) -> dict[str, Any]:
    return phase2_iam.validate_step12b_phase2_iam_plan(
        phase2_iam_plan,
        deployment_contract=deployment_contract,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
    )


def get_step12b_phase2_exact_readback_receipt(
    installed: Phase2IamInstalledCapability,
    *,
    phase2_iam_plan: Mapping[str, Any],
    deployment_contract: Mapping[str, Any],
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
) -> dict[str, Any]:
    """Return a fresh, fully revalidated copy of the install readback."""

    plan = _validate_plan(
        phase2_iam_plan,
        deployment_contract=deployment_contract,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
    )
    if (
        not isinstance(installed, Phase2IamInstalledCapability)
        or installed._seal is not _INSTALLED_SEAL
        or installed.deployment_contract_sha256
        != plan["source_deployment"]["deployment_contract_sha256"]
        or installed.phase2_iam_plan_sha256 != plan["plan_sha256"]
    ):
        raise ValueError("Phase2 installed capability changed")
    try:
        stored = json.loads(
            installed._exact_readback_receipt_bytes.decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Phase2 IAM exact readback storage changed") from error
    if not isinstance(stored, dict):
        raise ValueError("Phase2 IAM exact readback storage changed")
    _reject_sensitive_readback_fields(stored)
    checked = phase2_iam.validate_step12b_phase2_iam_readback_receipt(
        stored,
        plan=plan,
        deployment_contract=deployment_contract,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
    )
    if (
        checked["receipt_sha256"]
        != installed.exact_readback_receipt_sha256
        or installed.install_receipt.get(
            "exact_readback_receipt_sha256"
        )
        != checked["receipt_sha256"]
    ):
        raise ValueError("Phase2 IAM exact readback digest changed")
    return copy.deepcopy(checked)


def _validate_admin(
    iam_admin: rest_iam.Step11RestIamAdmin,
    plan: Mapping[str, Any],
) -> None:
    controller = plan["principals"]["controller_service_account"]
    if (
        getattr(iam_admin, "project", None) != phase2_iam.PROJECT
        or getattr(iam_admin, "bucket", None) != phase2_iam.BUCKET
        or getattr(iam_admin, "worker_service_account", None)
        != phase2_iam.WORKER_SERVICE_ACCOUNT
        or getattr(iam_admin, "controller_service_account", None)
        != controller["email"]
    ):
        raise ValueError("Phase2 IAM admin resource binding changed")


def _validate_token_barrier(
    outcome: token_barrier.TokenBarrierOutcome,
    plan: Mapping[str, Any],
) -> str:
    if not isinstance(outcome, token_barrier.TokenBarrierOutcome):
        raise ValueError("successful TokenBarrierOutcome is required")
    receipt = copy.deepcopy(dict(outcome.receipt))
    if set(receipt) != _TOKEN_SUCCESS_FIELDS:
        raise ValueError("token barrier success receipt fields changed")
    supplied = _sha(
        receipt.pop("receipt_sha256", None),
        "token barrier receipt",
    )
    if canonical_sha256(receipt) != supplied:
        raise ValueError("token barrier receipt digest changed")
    controller = plan["principals"]["controller_service_account"]
    if (
        receipt["schema"] != token_barrier.SCHEMA
        or receipt["status"] != token_barrier.SUCCESS_STATUS
        or receipt["controller_service_account"] != controller["email"]
        or receipt["run_scoped_controller_required"] is not True
        or receipt["binding_role"] != phase2_iam.TOKEN_CREATOR_ROLE
        or receipt["binding_add_outcome"] != "confirmed_changed"
        or receipt["token_creator_revoke_changed"] is not True
        or receipt["token_creator_live_after_barrier"] is not False
        or receipt["phase2_started"] is not False
        or receipt["phase2_must_exclude_token_creator"] is not True
        or receipt["fixed_nonrefreshing_controller_credential"] is not True
        or receipt["controller_token_remint_forbidden"] is not True
        or receipt["vm_insert_attempt_count"] != 0
        or receipt["access_token_stored"] is not False
        or receipt["authorization_header_stored"] is not False
        or receipt["response_body_stored"] is not False
        or receipt["current_profile_changed"] is not False
    ):
        raise ValueError("token barrier did not close Phase 1")
    _sha(
        receipt["token_creator_revoke_readback_sha256"],
        "TokenCreator revoke readback",
    )
    zero = receipt["token_creator_revoke_zero_readback"]
    if not isinstance(zero, Mapping):
        raise ValueError("TokenCreator zero readback changed")
    expected_zero_fields = {
        "zero_observed",
        "poll_count",
        "stale_present_count",
        "observation_sha256s",
        "observation_sha256s_sha256",
        "sleep_delays_seconds",
        "elapsed_seconds",
        "maximum_seconds",
        "failure_reason",
        "second_add_performed",
        "controller_token_reminted",
    }
    observations = zero.get("observation_sha256s")
    delays = zero.get("sleep_delays_seconds")
    poll_count = zero.get("poll_count")
    stale_count = zero.get("stale_present_count")
    if (
        set(zero) != expected_zero_fields
        or zero.get("zero_observed") is not True
        or type(poll_count) is not int
        or poll_count < 1
        or type(stale_count) is not int
        or not 0 <= stale_count < poll_count
        or not isinstance(observations, list)
        or len(observations) != poll_count
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in observations
        )
        or zero.get("observation_sha256s_sha256")
        != canonical_sha256(observations)
        or not isinstance(delays, list)
        or len(delays) != poll_count - 1
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or float(value) <= 0
            for value in delays
        )
        or isinstance(zero.get("elapsed_seconds"), bool)
        or not isinstance(zero.get("elapsed_seconds"), (int, float))
        or not 0 <= float(zero["elapsed_seconds"])
        <= token_barrier.MAX_PROPAGATION_SECONDS
        or zero.get("maximum_seconds")
        != token_barrier.MAX_PROPAGATION_SECONDS
        or zero.get("failure_reason") is not None
        or zero.get("second_add_performed") is not False
        or zero.get("controller_token_reminted") is not False
        or receipt["token_creator_revoke_zero_readback_evidence_sha256"]
        != canonical_sha256(dict(zero))
    ):
        raise ValueError("TokenCreator zero readback changed")
    return supplied


def _target(value: str) -> rest_iam.PolicyTarget:
    try:
        return rest_iam.PolicyTarget(value)
    except ValueError:
        raise ValueError("Phase2 IAM target escaped Step11RestIamAdmin") from None


def _bindings(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [
        *plan["phase2_bindings"]["controller"],
        *plan["phase2_bindings"]["worker"],
    ]
    if (
        len(rows)
        != (
            phase2_iam.CONTROLLER_BINDING_COUNT
            + phase2_iam.WORKER_BINDING_COUNT
        )
        or any(row["role"] == phase2_iam.TOKEN_CREATOR_ROLE for row in rows)
    ):
        raise ValueError("Phase2 IAM binding set changed")
    return [copy.deepcopy(dict(row)) for row in rows]


def _condition_key(value: Mapping[str, Any] | None) -> str | None:
    if value is None:
        return None
    return canonical_sha256(dict(value))


def _read_policies(
    iam_admin: rest_iam.Step11RestIamAdmin,
    plan: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    targets = sorted({row["target"] for row in _bindings(plan)})
    policies = {}
    for target in targets:
        policy = iam_admin.get_policy(_target(target))
        if not isinstance(policy, Mapping):
            raise _LifecycleAbort("policy_readback_changed")
        policies[target] = copy.deepcopy(dict(policy))
    return policies


def _targeted_memberships(
    policies: Mapping[str, Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    principals = {
        plan["principals"]["controller_principal"],
        plan["principals"]["worker_principal"],
    }
    rows = []
    for target, policy in policies.items():
        bindings = policy.get("bindings")
        # An empty IAM policy is returned by GCP with no "bindings" key (or
        # an explicit null), which legitimately means zero memberships - the
        # exact initial-zero state this check confirms. Treat it as empty
        # rather than a shape violation; still reject genuinely malformed
        # non-list, non-null shapes.
        if bindings is None:
            continue
        if not isinstance(bindings, list):
            raise _LifecycleAbort("policy_binding_shape_changed")
        for binding in bindings:
            if not isinstance(binding, Mapping):
                raise _LifecycleAbort("policy_binding_shape_changed")
            members = binding.get("members")
            role = binding.get("role")
            condition = binding.get("condition")
            if (
                not isinstance(members, list)
                or not isinstance(role, str)
                or (
                    condition is not None
                    and not isinstance(condition, Mapping)
                )
            ):
                raise _LifecycleAbort("policy_binding_shape_changed")
            for member in members:
                if member in principals:
                    rows.append(
                        {
                            "target": target,
                            "role": role,
                            "member": member,
                            "condition": (
                                None
                                if condition is None
                                else copy.deepcopy(dict(condition))
                            ),
                        }
                    )
    rows.sort(
        key=lambda row: (
            row["target"],
            row["role"],
            row["member"],
            _condition_key(row["condition"]) or "",
        )
    )
    return rows


def _expected_membership_keys(
    plan: Mapping[str, Any],
) -> set[tuple[str, str, str, str]]:
    return {
        (
            row["target"],
            row["role"],
            row["member"],
            _condition_key(row["condition"]) or "",
        )
        for row in _bindings(plan)
    }


def _observed_membership_keys(
    memberships: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str, str, str]]:
    return [
        (
            row["target"],
            row["role"],
            row["member"],
            _condition_key(row["condition"]) or "",
        )
        for row in memberships
    ]


def _require_initial_zero(
    iam_admin: rest_iam.Step11RestIamAdmin,
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    memberships = _targeted_memberships(
        _read_policies(iam_admin, plan), plan
    )
    if memberships:
        raise Phase2IamLifecycleError(
            _seal(
                {
                    "schema": FAILURE_RECEIPT_SCHEMA,
                    "stage": "initial_exact_zero",
                    "failure_reason": "stale_targeted_binding_present",
                    "phase2_iam_plan_sha256": plan["plan_sha256"],
                    "targeted_binding_count": len(memberships),
                    "targeted_binding_observation_sha256": canonical_sha256(
                        memberships
                    ),
                    "stale_binding_cleared": False,
                    "clear_and_continue_forbidden": True,
                    "iam_mutation_attempted": False,
                    "cleanup_attempted": False,
                    "access_token_stored": False,
                    "authorization_header_stored": False,
                    "raw_policy_stored": False,
                    "current_profile_changed": False,
                }
            )
        )
    return []


def _validate_custom_roles(
    plan: Mapping[str, Any],
    readbacks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    requirements = {
        row["name"]: row
        for row in plan["custom_role_readback_contract"]["requirements"]
    }
    if (
        isinstance(readbacks, (str, bytes))
        or not isinstance(readbacks, Sequence)
        or len(readbacks) != len(requirements)
    ):
        raise ValueError("custom-role readback count changed")
    normalized = []
    for raw in readbacks:
        row = copy.deepcopy(dict(raw))
        if set(row) != {
            "name",
            "stage",
            "included_permissions",
            "deleted",
            "get_status",
            "readback_complete",
        }:
            raise ValueError("custom-role readback fields changed")
        expected = requirements.get(row["name"])
        if (
            expected is None
            or row["stage"] != expected["stage"]
            or row["included_permissions"]
            != expected["included_permissions"]
            or row["deleted"] is not False
            or row["get_status"] != 200
            or row["readback_complete"] is not True
        ):
            raise ValueError("custom-role readback changed")
        normalized.append(row)
    normalized.sort(key=lambda row: row["name"])
    if [row["name"] for row in normalized] != sorted(requirements):
        raise ValueError("custom-role readback identity changed")
    return normalized


def _install_order(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_purpose = {row["purpose"]: row for row in _bindings(plan)}
    purposes = [
        "worker_package_and_result_reader",
        "worker_result_creator",
        "worker_self_delete",
        "controller_service_usage",
        "controller_zone_operation_reader",
        "controller_instance_lifecycle",
        "controller_worker_act_as",
        "controller_vm_launch",
    ]
    if set(by_purpose) != set(purposes):
        raise ValueError("Phase2 install purpose set changed")
    return [by_purpose[purpose] for purpose in purposes]


def _controller_remove_order(
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_purpose = {
        row["purpose"]: row
        for row in plan["phase2_bindings"]["controller"]
    }
    purposes = [
        "controller_vm_launch",
        "controller_worker_act_as",
        "controller_instance_lifecycle",
        "controller_service_usage",
        "controller_zone_operation_reader",
    ]
    if set(by_purpose) != set(purposes):
        raise ValueError("controller removal purpose set changed")
    return [by_purpose[purpose] for purpose in purposes]


def _worker_remove_order(
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_purpose = {
        row["purpose"]: row
        for row in plan["phase2_bindings"]["worker"]
    }
    purposes = [
        "worker_result_creator",
        "worker_self_delete",
        "worker_package_and_result_reader",
    ]
    if set(by_purpose) != set(purposes):
        raise ValueError("worker removal purpose set changed")
    return [by_purpose[purpose] for purpose in purposes]


def _mutation_record(
    binding: Mapping[str, Any],
    result: Any,
    *,
    operation: str,
    require_changed: bool,
) -> dict[str, Any]:
    changed = getattr(result, "changed", None)
    attempts = getattr(result, "attempts", None)
    target = getattr(result, "target", None)
    target_value = getattr(target, "value", target)
    if (
        type(changed) is not bool
        or type(attempts) is not int
        or attempts < 1
        or target_value != binding["target"]
        or (require_changed and changed is not True)
    ):
        raise _LifecycleAbort(f"{operation}_mutation_result_changed")
    return {
        "operation": operation,
        "purpose": binding["purpose"],
        "target": binding["target"],
        "binding_sha256": binding["binding_sha256"],
        "changed": changed,
        "attempts": attempts,
    }


def _mutate(
    iam_admin: rest_iam.Step11RestIamAdmin,
    binding: Mapping[str, Any],
    *,
    add: bool,
    require_changed: bool,
) -> dict[str, Any]:
    method = iam_admin.add_binding if add else iam_admin.remove_binding
    result = method(
        _target(binding["target"]),
        role=binding["role"],
        member=binding["member"],
        condition=binding["condition"],
    )
    return _mutation_record(
        binding,
        result,
        operation="add" if add else "remove",
        require_changed=require_changed,
    )


def _binding_readbacks(
    policies: Mapping[str, Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    memberships = _targeted_memberships(policies, plan)
    observed_keys = _observed_membership_keys(memberships)
    expected_keys = _expected_membership_keys(plan)
    unexpected = [
        row
        for row, key in zip(memberships, observed_keys, strict=True)
        if key not in expected_keys
    ]
    counts: dict[tuple[str, str, str, str], int] = {}
    for key in observed_keys:
        counts[key] = counts.get(key, 0) + 1
    readbacks = []
    for row in _bindings(plan):
        key = (
            row["target"],
            row["role"],
            row["member"],
            _condition_key(row["condition"]) or "",
        )
        readbacks.append(
            {
                "purpose": row["purpose"],
                "target": row["target"],
                "resource": row["resource"],
                "role": row["role"],
                "member": row["member"],
                "condition": copy.deepcopy(row["condition"]),
                "binding_sha256": row["binding_sha256"],
                "member_occurrences": counts.get(key, 0),
                "readback_complete": True,
            }
        )
    return readbacks, unexpected


def _zero_readbacks(
    policies: Mapping[str, Mapping[str, Any]],
    identities: Sequence[Mapping[str, Any]],
    *,
    principal: str,
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    memberships = _targeted_memberships(policies, plan)
    if any(row["member"] == principal for row in memberships):
        raise _LifecycleAbort("principal_binding_not_zero")
    return [
        {
            **copy.deepcopy(dict(identity)),
            "controller_member_count": 0,
            "controller_member_present": False,
            "readback_complete": True,
        }
        for identity in identities
    ]


def _cleanup_all(
    iam_admin: rest_iam.Step11RestIamAdmin,
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = []
    for group, rows in (
        ("controller", _controller_remove_order(plan)),
        ("worker", _worker_remove_order(plan)),
    ):
        for binding in rows:
            try:
                record = _mutate(
                    iam_admin,
                    binding,
                    add=False,
                    require_changed=False,
                )
                records.append({**record, "cleanup_group": group})
            except Exception:
                records.append(
                    {
                        "operation": "remove",
                        "purpose": binding["purpose"],
                        "target": binding["target"],
                        "binding_sha256": binding["binding_sha256"],
                        "changed": None,
                        "attempts": None,
                        "cleanup_group": group,
                        "sanitized_failure": "iam_remove_failed",
                    }
                )
    return records


def _failure(
    *,
    stage: str,
    reason: str,
    plan: Mapping[str, Any],
    mutation_records: Sequence[Mapping[str, Any]],
    cleanup_records: Sequence[Mapping[str, Any]],
) -> Phase2IamLifecycleError:
    return Phase2IamLifecycleError(
        _seal(
            {
                "schema": FAILURE_RECEIPT_SCHEMA,
                "stage": stage,
                "failure_reason": reason,
                "deployment_contract_sha256": plan[
                    "source_deployment"
                ]["deployment_contract_sha256"],
                "phase2_iam_plan_sha256": plan["plan_sha256"],
                "mutation_records": [
                    copy.deepcopy(dict(row)) for row in mutation_records
                ],
                "mutation_records_sha256": canonical_sha256(
                    list(mutation_records)
                ),
                "cleanup_records": [
                    copy.deepcopy(dict(row)) for row in cleanup_records
                ],
                "cleanup_records_sha256": canonical_sha256(
                    list(cleanup_records)
                ),
                "cleanup_controller_before_worker": True,
                "phase2_install_completed": False,
                "pair_release_authorized": False,
                "vm_insert_attempt_count": 0,
                "access_token_stored": False,
                "authorization_header_stored": False,
                "raw_policy_stored": False,
                "current_profile_changed": False,
            }
        )
    )


def install_step12b_phase2_iam(
    *,
    iam_admin: rest_iam.Step11RestIamAdmin,
    phase2_iam_plan: Mapping[str, Any],
    deployment_contract: Mapping[str, Any],
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
    token_barrier_outcome: token_barrier.TokenBarrierOutcome,
    custom_role_readbacks: Sequence[Mapping[str, Any]],
    observed_at_unix_seconds: int,
) -> Phase2IamInstalledCapability:
    """Install exactly eight bindings after a closed Phase-1 barrier."""

    plan = _validate_plan(
        phase2_iam_plan,
        deployment_contract=deployment_contract,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
    )
    _validate_admin(iam_admin, plan)
    barrier_sha = _validate_token_barrier(token_barrier_outcome, plan)
    normalized_roles = _validate_custom_roles(
        plan, custom_role_readbacks
    )
    _require_initial_zero(iam_admin, plan)

    mutation_records: list[dict[str, Any]] = []
    try:
        for binding in _install_order(plan):
            mutation_records.append(
                _mutate(
                    iam_admin,
                    binding,
                    add=True,
                    require_changed=True,
                )
            )
        policies = _read_policies(iam_admin, plan)
        binding_readbacks, unexpected = _binding_readbacks(policies, plan)
        exact_readback = (
            phase2_iam.build_step12b_phase2_iam_readback_receipt(
                plan,
                deployment_contract=deployment_contract,
                candidate_payload_contract=candidate_payload_contract,
                reference_payload_contract=reference_payload_contract,
                controller_public_key_record=controller_public_key_record,
                run_nonce=run_nonce,
                observed_at_unix_seconds=observed_at_unix_seconds,
                custom_role_readbacks=normalized_roles,
                binding_readbacks=binding_readbacks,
                unexpected_targeted_bindings=unexpected,
            )
        )
        _reject_sensitive_readback_fields(exact_readback)
    except Exception as error:
        cleanup = _cleanup_all(iam_admin, plan)
        reason = (
            error.code
            if isinstance(error, _LifecycleAbort)
            else "phase2_install_or_readback_failed"
        )
        raise _failure(
            stage="install_and_exact_readback",
            reason=reason,
            plan=plan,
            mutation_records=mutation_records,
            cleanup_records=cleanup,
        ) from None

    receipt = _seal(
        {
            "schema": INSTALL_RECEIPT_SCHEMA,
            "deployment_contract_sha256": plan["source_deployment"][
                "deployment_contract_sha256"
            ],
            "phase2_iam_plan_sha256": plan["plan_sha256"],
            "token_barrier_receipt_sha256": barrier_sha,
            "initial_targeted_binding_count": 0,
            "initial_exact_zero_readback": True,
            "mutation_records": mutation_records,
            "mutation_records_sha256": canonical_sha256(
                mutation_records
            ),
            "installed_controller_binding_count": (
                phase2_iam.CONTROLLER_BINDING_COUNT
            ),
            "installed_worker_binding_count": (
                phase2_iam.WORKER_BINDING_COUNT
            ),
            "exact_readback_receipt_sha256": exact_readback[
                "receipt_sha256"
            ],
            "all_phase2_bindings_present_exactly_once": True,
            "phase2_started": True,
            "token_creator_binding_included": False,
            "vm_insert_attempt_count": 0,
            "access_token_stored": False,
            "authorization_header_stored": False,
            "raw_policy_stored": False,
            "current_profile_changed": False,
        }
    )
    return Phase2IamInstalledCapability(
        deployment_contract_sha256=receipt[
            "deployment_contract_sha256"
        ],
        phase2_iam_plan_sha256=plan["plan_sha256"],
        token_barrier_receipt_sha256=barrier_sha,
        exact_readback_receipt_sha256=exact_readback["receipt_sha256"],
        install_receipt=receipt,
        _exact_readback_receipt_bytes=phase2_iam.canonical_bytes(
            exact_readback
        ),
        _seal=_INSTALLED_SEAL,
    )


def validate_pair_claim_cas_readbacks_capability(
    *,
    deployment_contract: Mapping[str, Any],
    claim_cas_readback_receipts: Sequence[Mapping[str, Any]],
) -> PairClaimCasReadbacksCapability:
    """Create a sealed pair-wide capability from exactly two claim readbacks."""

    if (
        isinstance(claim_cas_readback_receipts, (str, bytes))
        or not isinstance(claim_cas_readback_receipts, Sequence)
        or len(claim_cas_readback_receipts) != phase2_iam.VM_COUNT
    ):
        raise ValueError("exactly two claim CAS readbacks are required")
    checked = [
        pair_release.validate_claim_cas_readback_receipt(
            row, deployment_contract=deployment_contract
        )
        for row in claim_cas_readback_receipts
    ]
    selected = list(deployment_contract["selected_job_ids"])
    by_job = {row["external_job_id"]: row for row in checked}
    if len(by_job) != phase2_iam.VM_COUNT or set(by_job) != set(selected):
        raise ValueError("claim CAS readback pair mapping changed")
    ordered = [by_job[job_id] for job_id in selected]
    receipt_shas = tuple(row["receipt_sha256"] for row in ordered)
    body = {
        "schema": PAIR_CLAIMS_CAPABILITY_SCHEMA,
        "deployment_contract_sha256": deployment_contract[
            "deployment_contract_sha256"
        ],
        "external_job_ids": selected,
        "claim_readback_receipt_sha256s": list(receipt_shas),
        "both_provider_claim_cas_readbacks_complete": True,
    }
    return PairClaimCasReadbacksCapability(
        deployment_contract_sha256=body[
            "deployment_contract_sha256"
        ],
        external_job_ids=(selected[0], selected[1]),
        claim_readback_receipt_sha256s=(
            receipt_shas[0],
            receipt_shas[1],
        ),
        capability_sha256=canonical_sha256(body),
        _seal=_PAIR_CLAIMS_SEAL,
    )


def remove_step12b_phase2_controller_bindings(
    *,
    iam_admin: rest_iam.Step11RestIamAdmin,
    installed: Phase2IamInstalledCapability,
    pair_claims: PairClaimCasReadbacksCapability,
    phase2_iam_plan: Mapping[str, Any],
    deployment_contract: Mapping[str, Any],
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
) -> ControllerBindingsZeroCapability:
    """Remove five controller bindings only after both provider claims."""

    plan = _validate_plan(
        phase2_iam_plan,
        deployment_contract=deployment_contract,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
    )
    _validate_admin(iam_admin, plan)
    if (
        not isinstance(installed, Phase2IamInstalledCapability)
        or installed._seal is not _INSTALLED_SEAL
        or installed.deployment_contract_sha256
        != plan["source_deployment"]["deployment_contract_sha256"]
        or installed.phase2_iam_plan_sha256 != plan["plan_sha256"]
        or not isinstance(pair_claims, PairClaimCasReadbacksCapability)
        or pair_claims._seal is not _PAIR_CLAIMS_SEAL
        or pair_claims.deployment_contract_sha256
        != installed.deployment_contract_sha256
        or pair_claims.external_job_ids
        != tuple(plan["source_deployment"]["selected_job_ids"])
    ):
        raise ValueError("Phase2 controller removal capability changed")

    mutation_records: list[dict[str, Any]] = []
    try:
        for binding in _controller_remove_order(plan):
            mutation_records.append(
                _mutate(
                    iam_admin,
                    binding,
                    add=False,
                    require_changed=True,
                )
            )
        policies = _read_policies(iam_admin, plan)
        identities = plan["controller_release_removal_group"][
            "binding_identities"
        ]
        zero_readbacks = _zero_readbacks(
            policies,
            identities,
            principal=plan["principals"]["controller_principal"],
            plan=plan,
        )
        phase2_zero = pair_release.build_phase2_zero_receipt(
            deployment_contract=deployment_contract,
            candidate_payload_contract=candidate_payload_contract,
            reference_payload_contract=reference_payload_contract,
            controller_public_key_record=controller_public_key_record,
            run_nonce=run_nonce,
            phase2_iam_plan=plan,
            binding_readbacks=zero_readbacks,
        )
    except Exception as error:
        cleanup = _cleanup_all(iam_admin, plan)
        reason = (
            error.code
            if isinstance(error, _LifecycleAbort)
            else "controller_remove_or_zero_readback_failed"
        )
        raise _failure(
            stage="controller_remove_after_pair_claims",
            reason=reason,
            plan=plan,
            mutation_records=mutation_records,
            cleanup_records=cleanup,
        ) from None

    lifecycle_receipt = _seal(
        {
            "schema": CONTROLLER_ZERO_LIFECYCLE_SCHEMA,
            "deployment_contract_sha256": (
                installed.deployment_contract_sha256
            ),
            "phase2_iam_plan_sha256": plan["plan_sha256"],
            "pair_claims_capability_sha256": pair_claims.capability_sha256,
            "mutation_records": mutation_records,
            "mutation_records_sha256": canonical_sha256(
                mutation_records
            ),
            "removed_controller_binding_count": (
                phase2_iam.CONTROLLER_BINDING_COUNT
            ),
            "phase2_zero_receipt_sha256": phase2_zero[
                "receipt_sha256"
            ],
            "all_controller_bindings_zero": True,
            "worker_bindings_removed": False,
            "pair_release_prerequisite_satisfied": True,
            "access_token_stored": False,
            "authorization_header_stored": False,
            "raw_policy_stored": False,
            "current_profile_changed": False,
        }
    )
    return ControllerBindingsZeroCapability(
        deployment_contract_sha256=installed.deployment_contract_sha256,
        phase2_iam_plan_sha256=plan["plan_sha256"],
        phase2_zero_receipt=phase2_zero,
        lifecycle_receipt=lifecycle_receipt,
        _seal=_CONTROLLER_ZERO_SEAL,
    )


def remove_step12b_phase2_worker_bindings_final(
    *,
    iam_admin: rest_iam.Step11RestIamAdmin,
    installed: Phase2IamInstalledCapability,
    controller_zero: ControllerBindingsZeroCapability,
    completion_kind: str,
    completion_evidence_sha256: str,
    phase2_iam_plan: Mapping[str, Any],
    deployment_contract: Mapping[str, Any],
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
) -> WorkerBindingsZeroCapability:
    """Remove three worker bindings at final DONE or failure cleanup."""

    plan = _validate_plan(
        phase2_iam_plan,
        deployment_contract=deployment_contract,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
    )
    _validate_admin(iam_admin, plan)
    evidence_sha = _sha(
        completion_evidence_sha256, "worker cleanup evidence"
    )
    if completion_kind not in {
        "pair_done_readback",
        "failure_cleanup",
    }:
        raise ValueError("worker cleanup completion kind changed")
    if (
        not isinstance(installed, Phase2IamInstalledCapability)
        or installed._seal is not _INSTALLED_SEAL
        or not isinstance(
            controller_zero, ControllerBindingsZeroCapability
        )
        or controller_zero._seal is not _CONTROLLER_ZERO_SEAL
        or controller_zero.deployment_contract_sha256
        != installed.deployment_contract_sha256
        or controller_zero.phase2_iam_plan_sha256
        != installed.phase2_iam_plan_sha256
        or installed.phase2_iam_plan_sha256 != plan["plan_sha256"]
    ):
        raise ValueError("worker cleanup capability changed")

    mutation_records: list[dict[str, Any]] = []
    try:
        for binding in _worker_remove_order(plan):
            mutation_records.append(
                _mutate(
                    iam_admin,
                    binding,
                    add=False,
                    require_changed=True,
                )
            )
        policies = _read_policies(iam_admin, plan)
        identities = plan["worker_final_cleanup_removal_group"][
            "binding_identities"
        ]
        memberships = _targeted_memberships(policies, plan)
        if memberships:
            raise _LifecycleAbort("final_targeted_binding_set_not_zero")
        zero_rows = [
            {
                **copy.deepcopy(dict(identity)),
                "worker_member_count": 0,
                "worker_member_present": False,
                "readback_complete": True,
            }
            for identity in identities
        ]
    except Exception as error:
        cleanup = _cleanup_all(iam_admin, plan)
        reason = (
            error.code
            if isinstance(error, _LifecycleAbort)
            else "worker_remove_or_zero_readback_failed"
        )
        raise _failure(
            stage="worker_final_remove",
            reason=reason,
            plan=plan,
            mutation_records=mutation_records,
            cleanup_records=cleanup,
        ) from None

    receipt = _seal(
        {
            "schema": WORKER_ZERO_RECEIPT_SCHEMA,
            "deployment_contract_sha256": (
                installed.deployment_contract_sha256
            ),
            "phase2_iam_plan_sha256": plan["plan_sha256"],
            "completion_kind": completion_kind,
            "completion_evidence_sha256": evidence_sha,
            "mutation_records": mutation_records,
            "mutation_records_sha256": canonical_sha256(
                mutation_records
            ),
            "worker_zero_readbacks": zero_rows,
            "worker_zero_readbacks_sha256": canonical_sha256(zero_rows),
            "removed_worker_binding_count": (
                phase2_iam.WORKER_BINDING_COUNT
            ),
            "all_worker_bindings_zero": True,
            "all_controller_bindings_already_zero": True,
            "final_targeted_binding_count": 0,
            "access_token_stored": False,
            "authorization_header_stored": False,
            "raw_policy_stored": False,
            "current_profile_changed": False,
        }
    )
    return WorkerBindingsZeroCapability(
        deployment_contract_sha256=installed.deployment_contract_sha256,
        phase2_iam_plan_sha256=plan["plan_sha256"],
        receipt=receipt,
        _seal=_WORKER_ZERO_SEAL,
    )


def cleanup_step12b_phase2_iam_on_failure(
    *,
    iam_admin: rest_iam.Step11RestIamAdmin,
    phase2_iam_plan: Mapping[str, Any],
    deployment_contract: Mapping[str, Any],
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
    failure_evidence_sha256: str,
) -> dict[str, Any]:
    """Explicit best-effort IAM cleanup: controller first, then worker."""

    plan = _validate_plan(
        phase2_iam_plan,
        deployment_contract=deployment_contract,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
    )
    _validate_admin(iam_admin, plan)
    evidence_sha = _sha(
        failure_evidence_sha256, "Phase2 failure cleanup evidence"
    )
    records = _cleanup_all(iam_admin, plan)
    return _seal(
        {
            "schema": FAILURE_RECEIPT_SCHEMA,
            "stage": "explicit_failure_cleanup",
            "failure_reason": "caller_confirmed_failure",
            "deployment_contract_sha256": plan[
                "source_deployment"
            ]["deployment_contract_sha256"],
            "phase2_iam_plan_sha256": plan["plan_sha256"],
            "failure_evidence_sha256": evidence_sha,
            "cleanup_records": records,
            "cleanup_records_sha256": canonical_sha256(records),
            "cleanup_controller_before_worker": True,
            "cleanup_best_effort": True,
            "pair_release_authorized": False,
            "vm_insert_attempt_count": 0,
            "access_token_stored": False,
            "authorization_header_stored": False,
            "raw_policy_stored": False,
            "current_profile_changed": False,
        }
    )


__all__ = [
    "CONTROLLER_ZERO_LIFECYCLE_SCHEMA",
    "ControllerBindingsZeroCapability",
    "FAILURE_RECEIPT_SCHEMA",
    "INSTALL_RECEIPT_SCHEMA",
    "PAIR_CLAIMS_CAPABILITY_SCHEMA",
    "PairClaimCasReadbacksCapability",
    "Phase2IamInstalledCapability",
    "Phase2IamLifecycleError",
    "WORKER_ZERO_RECEIPT_SCHEMA",
    "WorkerBindingsZeroCapability",
    "cleanup_step12b_phase2_iam_on_failure",
    "get_step12b_phase2_exact_readback_receipt",
    "install_step12b_phase2_iam",
    "remove_step12b_phase2_controller_bindings",
    "remove_step12b_phase2_worker_bindings_final",
    "validate_pair_claim_cas_readbacks_capability",
]
