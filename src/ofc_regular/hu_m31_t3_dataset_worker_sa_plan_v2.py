"""Cloud-neutral service-account inventory/provisioning plan for transport v2.

The live project currently has worker accounts 00..07.  Parallel-20 requires
the exact pool 00..19.  This module records the missing 08..19 actions but
does not authorize or perform account creation.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from . import hu_m31_t3_dataset_gcp_provider_v2 as provider


PLAN_SCHEMA = "hu_m31_t3_dataset_worker_sa_provisioning_plan_v2"
INVENTORY_RECEIPT_SCHEMA = "hu_m31_t3_dataset_worker_sa_inventory_receipt_v2"
KNOWN_EXISTING = provider.EXPECTED_WORKER_SERVICE_ACCOUNTS[:8]
EXPECTED_MISSING = provider.EXPECTED_WORKER_SERVICE_ACCOUNTS[8:]


def canonical_sha256(value: Any) -> str:
    return provider.canonical_sha256(value)


def _self_digest(value: Mapping[str, Any], field: str) -> str:
    copied = deepcopy(dict(value))
    copied.pop(field, None)
    return canonical_sha256(copied)


def build_provisioning_plan(
    *, observed_existing_service_accounts: Sequence[str] = KNOWN_EXISTING
) -> dict[str, Any]:
    observed = tuple(observed_existing_service_accounts)
    required = provider.EXPECTED_WORKER_SERVICE_ACCOUNTS
    if (
        len(set(observed)) != len(observed)
        or any(email not in required for email in observed)
        or not set(KNOWN_EXISTING).issubset(observed)
    ):
        raise ValueError("parallel20 worker SA inventory is unexpected")
    missing = tuple(email for email in required if email not in observed)
    core = {
        "schema": PLAN_SCHEMA,
        "status": "requires_separate_authorization_before_provisioning",
        "project": provider.PROJECT,
        "required_service_accounts": list(required),
        "observed_existing_service_accounts": list(observed),
        "missing_service_accounts": list(missing),
        "missing_count": len(missing),
        "expected_initial_missing_service_accounts": list(EXPECTED_MISSING),
        "actions": [
            {
                "ordinal": index,
                "email": email,
                "account_id": email.split("@", 1)[0],
                "display_name": f"OFC M3.1 dataset worker {email[-39:-37]}",
                "operation": "create_service_account",
            }
            for index, email in enumerate(missing)
        ],
        "service_account_creation_authorized": False,
        "iam_role_grants_authorized": False,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    return {**core, "plan_sha256": canonical_sha256(core)}


def validate_provisioning_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    plan = deepcopy(dict(value))
    if (
        plan.get("plan_sha256") != _self_digest(plan, "plan_sha256")
        or plan.get("schema") != PLAN_SCHEMA
        or plan.get("required_service_accounts")
        != list(provider.EXPECTED_WORKER_SERVICE_ACCOUNTS)
        or plan.get("missing_count") != len(plan.get("missing_service_accounts", []))
        or plan.get("service_account_creation_authorized") is not False
        or plan.get("iam_role_grants_authorized") is not False
        or plan.get("cloud_mutated") is not False
        or plan.get("current_profile_changed") is not False
    ):
        raise ValueError("parallel20 worker SA provisioning plan changed")
    return plan


def build_inventory_receipt(
    *,
    provisioning_plan: Mapping[str, Any],
    observed_existing_service_accounts: Sequence[str],
) -> dict[str, Any]:
    plan = validate_provisioning_plan(provisioning_plan)
    observed = tuple(observed_existing_service_accounts)
    if (
        len(set(observed)) != len(observed)
        or any(
            email not in provider.EXPECTED_WORKER_SERVICE_ACCOUNTS
            for email in observed
        )
    ):
        raise ValueError("parallel20 worker SA inventory readback changed")
    missing = [
        email
        for email in provider.EXPECTED_WORKER_SERVICE_ACCOUNTS
        if email not in observed
    ]
    ready = not missing
    core = {
        "schema": INVENTORY_RECEIPT_SCHEMA,
        "status": (
            "exact_parallel20_worker_pool_ready"
            if ready
            else "parallel20_worker_pool_incomplete_no_go"
        ),
        "provisioning_plan_sha256": plan["plan_sha256"],
        "required_service_accounts": list(
            provider.EXPECTED_WORKER_SERVICE_ACCOUNTS
        ),
        "observed_existing_service_accounts": list(observed),
        "missing_service_accounts": missing,
        "worker_pool_ready": ready,
        "service_accounts_created_by_receipt": False,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def validate_inventory_receipt(
    value: Mapping[str, Any], *, provisioning_plan: Mapping[str, Any]
) -> dict[str, Any]:
    plan = validate_provisioning_plan(provisioning_plan)
    receipt = deepcopy(dict(value))
    missing = receipt.get("missing_service_accounts")
    ready = receipt.get("worker_pool_ready")
    if (
        receipt.get("receipt_sha256") != _self_digest(receipt, "receipt_sha256")
        or receipt.get("schema") != INVENTORY_RECEIPT_SCHEMA
        or receipt.get("provisioning_plan_sha256") != plan["plan_sha256"]
        or receipt.get("required_service_accounts")
        != list(provider.EXPECTED_WORKER_SERVICE_ACCOUNTS)
        or not isinstance(missing, list)
        or ready is not (len(missing) == 0)
        or receipt.get("status")
        != (
            "exact_parallel20_worker_pool_ready"
            if ready
            else "parallel20_worker_pool_incomplete_no_go"
        )
        or receipt.get("service_accounts_created_by_receipt") is not False
        or receipt.get("cloud_mutated") is not False
        or receipt.get("current_profile_changed") is not False
    ):
        raise ValueError("parallel20 worker SA inventory receipt changed")
    return receipt


__all__ = [
    "EXPECTED_MISSING",
    "INVENTORY_RECEIPT_SCHEMA",
    "KNOWN_EXISTING",
    "PLAN_SCHEMA",
    "build_inventory_receipt",
    "build_provisioning_plan",
    "validate_inventory_receipt",
    "validate_provisioning_plan",
]
