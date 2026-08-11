"""Quota-bound production transport selection for the M3.1 dataset.

The scientific dataset is identical under both transports.  The current live
asia-northeast1 C4-family limit is 128 vCPUs, so production must use the
existing v1 8-VM / 45-wave transport.  Parallel-20 v2 remains dormant until a
fresh quota readback can satisfy 320 vCPUs plus its 35-vCPU reserve.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from . import hu_m31_t3_dataset_gcp_provider_v2 as provider_v2
from . import hu_m31_t3_dataset_gcp_transport_v1 as transport_v1
from . import hu_m31_t3_dataset_gcp_transport_v2 as transport_v2


SELECTION_SCHEMA = "hu_m31_t3_dataset_transport_selection_v1"
REHEARSAL_SCHEMA = "hu_m31_t3_dataset_transport_wave0_rehearsal_v1"


def canonical_sha256(value: Any) -> str:
    return transport_v1.canonical_sha256(value)


def _self_digest(value: Mapping[str, Any], field: str) -> str:
    copied = deepcopy(dict(value))
    copied.pop(field, None)
    return canonical_sha256(copied)


def build_selection_receipt(
    *,
    c4_limit_vcpus: int,
    c4_usage_vcpus: int,
    spot_limit_vcpus: int,
    spot_usage_vcpus: int,
    global_limit_vcpus: int,
    global_usage_vcpus: int,
    observed_worker_service_account_count: int,
    evidence_source: str,
    evidence_sha256: str,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    values = (
        c4_limit_vcpus,
        c4_usage_vcpus,
        spot_limit_vcpus,
        spot_usage_vcpus,
        global_limit_vcpus,
        global_usage_vcpus,
        observed_worker_service_account_count,
    )
    if (
        any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in values
        )
        or not isinstance(evidence_source, str)
        or not evidence_source
        or len(evidence_sha256) != 64
        or any(len(value) != 64 for value in source_hashes.values())
    ):
        raise ValueError("transport selection evidence is invalid")
    c4_headroom = c4_limit_vcpus - c4_usage_vcpus
    spot_headroom = spot_limit_vcpus - spot_usage_vcpus
    global_headroom = global_limit_vcpus - global_usage_vcpus
    v1_required = transport_v1.MAX_CONCURRENT_VMS * transport_v1.VCPUS_PER_VM
    v2_required = transport_v2.MAX_CONCURRENT_VCPUS
    v2_required_with_reserve = v2_required + provider_v2.MIN_POST_LAUNCH_RESERVE_VCPUS
    v1_pass = all(
        headroom >= v1_required
        for headroom in (c4_headroom, spot_headroom, global_headroom)
    )
    v2_pass = (
        all(
            headroom >= v2_required_with_reserve
            for headroom in (c4_headroom, spot_headroom, global_headroom)
        )
        and observed_worker_service_account_count == transport_v2.MAX_CONCURRENT_VMS
    )
    if not v1_pass or v2_pass:
        raise ValueError("live quota no longer selects v1-only production")
    core = {
        "schema": SELECTION_SCHEMA,
        "status": "v1_8vm_selected_v2_parallel20_no_go",
        "evidence_source": evidence_source,
        "evidence_sha256": evidence_sha256,
        "live_quota": {
            "region": transport_v1.REGION,
            "c4_family": {
                "limit_vcpus": c4_limit_vcpus,
                "usage_vcpus": c4_usage_vcpus,
                "headroom_vcpus": c4_headroom,
            },
            "spot": {
                "limit_vcpus": spot_limit_vcpus,
                "usage_vcpus": spot_usage_vcpus,
                "headroom_vcpus": spot_headroom,
            },
            "global": {
                "limit_vcpus": global_limit_vcpus,
                "usage_vcpus": global_usage_vcpus,
                "headroom_vcpus": global_headroom,
            },
        },
        "v1": {
            "selected_for_m31_production": True,
            "max_concurrent_vms": transport_v1.MAX_CONCURRENT_VMS,
            "required_vcpus": v1_required,
            "wave_count": transport_v1.WAVE_COUNT,
            "quota_gate_pass": True,
            "scientific_bytes_changed": False,
        },
        "v2": {
            "selected_for_m31_production": False,
            "max_concurrent_vms": transport_v2.MAX_CONCURRENT_VMS,
            "required_vcpus": v2_required,
            "minimum_post_launch_reserve_vcpus": (
                provider_v2.MIN_POST_LAUNCH_RESERVE_VCPUS
            ),
            "required_with_reserve_vcpus": v2_required_with_reserve,
            "wave_count": transport_v2.WAVE_COUNT,
            "quota_gate_pass": False,
            "c4_vcpu_deficit": v2_required_with_reserve - c4_headroom,
            "observed_worker_service_account_count": (
                observed_worker_service_account_count
            ),
            "missing_worker_service_account_count": max(
                0,
                transport_v2.MAX_CONCURRENT_VMS - observed_worker_service_account_count,
            ),
            "future_only_after_fresh_quota_and_inventory_pass": True,
        },
        "fallback_policy": {
            "selected": "v1_c4_standard_16_8vm_45waves",
            "different_vm_family_authorized": False,
            "different_vm_family_requires_small_parity_and_performance_smoke": True,
            "quota_increase_required_before_v2": True,
        },
        "service_account_08_19_creation_authorized": False,
        "service_account_08_19_creation_required_for_selected_v1": False,
        "source_hashes": dict(sorted(source_hashes.items())),
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def validate_selection_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = deepcopy(dict(value))
    if (
        receipt.get("receipt_sha256") != _self_digest(receipt, "receipt_sha256")
        or receipt.get("schema") != SELECTION_SCHEMA
        or receipt.get("status") != "v1_8vm_selected_v2_parallel20_no_go"
        or receipt.get("v1", {}).get("selected_for_m31_production") is not True
        or receipt.get("v1", {}).get("max_concurrent_vms") != 8
        or receipt.get("v1", {}).get("wave_count") != 45
        or receipt.get("v2", {}).get("selected_for_m31_production") is not False
        or receipt.get("v2", {}).get("max_concurrent_vms") != 20
        or receipt.get("v2", {}).get("quota_gate_pass") is not False
        or receipt.get("service_account_08_19_creation_authorized") is not False
        or receipt.get("cloud_mutated") is not False
        or receipt.get("current_profile_changed") is not False
    ):
        raise ValueError("transport selection receipt changed")
    return receipt


def build_wave0_rehearsal_receipt(
    *,
    selection_receipt: Mapping[str, Any],
    v1_wave_request: Mapping[str, Any],
    v2_wave_request: Mapping[str, Any],
    fixture_kind: str,
) -> dict[str, Any]:
    selection = validate_selection_receipt(selection_receipt)
    v1_selected = v1_wave_request.get("selected_attempts")
    v2_selected = v2_wave_request.get("selected_attempts")
    if (
        fixture_kind not in {"synthetic_local", "production_sources_local"}
        or not isinstance(v1_selected, list)
        or len(v1_selected) != 8
        or not isinstance(v2_selected, list)
        or len(v2_selected) != 20
        or v1_wave_request.get("wave_index") != 0
        or v2_wave_request.get("wave_index") != 0
        or [row["shard_id"] for row in v1_selected]
        != [row["shard_id"] for row in v2_selected[:8]]
    ):
        raise ValueError("wave0 rehearsal requests are not aligned")
    core = {
        "schema": REHEARSAL_SCHEMA,
        "status": "local_wave0_requests_generated_v1_selected",
        "selection_receipt_sha256": selection["receipt_sha256"],
        "fixture_kind": fixture_kind,
        "v1_wave_request_sha256": v1_wave_request["request_sha256"],
        "v1_selected_count": len(v1_selected),
        "v1_required_vcpus": 8 * transport_v1.VCPUS_PER_VM,
        "v2_wave_request_sha256": v2_wave_request["request_sha256"],
        "v2_selected_count": len(v2_selected),
        "v2_required_vcpus": transport_v2.MAX_CONCURRENT_VCPUS,
        "first_eight_shards_identical": True,
        "selected_for_execution": "v1",
        "v2_execution_authorized": False,
        "service_account_08_19_creation_authorized": False,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


__all__ = [
    "REHEARSAL_SCHEMA",
    "SELECTION_SCHEMA",
    "build_selection_receipt",
    "build_wave0_rehearsal_receipt",
    "canonical_sha256",
    "validate_selection_receipt",
]
