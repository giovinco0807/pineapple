"""Byte-locked, local-only C4 formal benchmark contract for HU RL.

The harness never creates or inspects cloud resources.  An external controller
must provide a canonical machine-attestation receipt for an already provisioned
``c4-standard-16`` instance.  The harness then verifies the exact installed
Linux wheel and source snapshot, reuses packed benchmark v3 for the throughput
measurement, and runs one additional 4,096-lane combined-packed RSS probe.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final

from .hu_rl_native import NativeBatchHuRlEnvV1
from .hu_rl_native_benchmark import (
    BENCHMARK_LANE_COUNTS,
    DECISIONS_PER_HAND,
    HU_RL_NATIVE_BATCH_BENCHMARK_SCHEMA,
    HU_RL_NATIVE_BATCH_LANE_RESULT_SCHEMA,
    RL_READY_DIAGNOSTIC_TARGET_DECISIONS_PER_SECOND,
    HuRlNativeBenchmarkError,
    _SOURCE_HASH_PATHS,
    _run_lane_benchmark,
    _validate_lane_result,
    _validate_provenance,
    collect_native_benchmark_provenance,
    read_process_memory,
    run_native_batch_mechanics_benchmark,
    validate_native_batch_benchmark_document,
)


HU_RL_C4_FORMAL_MANIFEST_SCHEMA: Final = (
    "regular_ofc_hu_rl_c4_formal_benchmark_manifest_v2"
)
HU_RL_C4_MACHINE_ATTESTATION_SCHEMA: Final = (
    "regular_ofc_hu_rl_c4_external_machine_attestation_v2"
)
HU_RL_C4_EXTERNAL_GCE_OBSERVATION_SCHEMA: Final = (
    "regular_ofc_hu_rl_c4_external_gce_observation_v1"
)
HU_RL_C4_MEMORY_PROBE_SCHEMA: Final = "regular_ofc_hu_rl_c4_4096_lane_memory_probe_v1"
HU_RL_C4_FORMAL_RESULT_SCHEMA: Final = "regular_ofc_hu_rl_c4_formal_benchmark_result_v2"

TARGET_PROVIDER: Final = "gcp"
TARGET_PROJECT_ID: Final = "ofc-solver-485418"
TARGET_ZONE: Final = "asia-northeast1-b"
TARGET_MACHINE_TYPE: Final = "c4-standard-16"
TARGET_VCPU_COUNT: Final = 16
TARGET_SYSTEM: Final = "Linux"
TARGET_ARCHITECTURE: Final = "x86_64"
TARGET_PYTHON_IMPLEMENTATION: Final = "CPython"
TARGET_PYTHON_MAJOR: Final = 3
TARGET_PYTHON_MINOR: Final = 11
TARGET_PYTHON_ABI: Final = "cp311"
TARGET_PROVISIONING_MODEL: Final = "SPOT"
TARGET_NIC_TYPE: Final = "GVNIC"
TARGET_BOOT_DISK_TYPE: Final = "hyperdisk-balanced"
EXTERNAL_GCE_OBSERVATION_SOURCE: Final = (
    "external_gce_instances_get_machine_types_get_disks_get_receipt"
)
MACHINE_ATTESTATION_SOURCE: Final = "external_canonical_gce_observation_receipt"
FORMAL_RATE_TARGET: Final = 20_000.0
FORMAL_MEMORY_LANE_COUNT: Final = 4_096
FORMAL_RSS_LIMIT_BYTES: Final = 2 * 1024 * 1024 * 1024
FORMAL_THREAD_COUNT: Final = 16
FORMAL_CHUNK_WIDTH: Final = 64
FORMAL_SEED: Final = 20260722
FORMAL_BOUNDARY_MODE: Final = "combined_packed"
FORMAL_MANIFEST_FILENAME: Final = "hu_rl_c4_formal_benchmark_manifest.json"
FORMAL_RESULT_FILENAME: Final = "hu_rl_c4_formal_benchmark_result.json"

STARTUP_SCRIPT_PATH: Final = "scripts/startup_hu_rl_c4_formal_benchmark.sh"
RUNNER_SCRIPT_PATH: Final = "scripts/run_hu_rl_c4_formal_benchmark.py"
HARNESS_MODULE_PATH: Final = "src/ofc_regular/hu_rl_c4_formal_benchmark.py"
HARNESS_SOURCE_PATHS: Final = frozenset(
    {STARTUP_SCRIPT_PATH, RUNNER_SCRIPT_PATH, HARNESS_MODULE_PATH}
)
STARTUP_ARGV_TEMPLATE: Final = (
    "python3",
    "-B",
    "-s",
    RUNNER_SCRIPT_PATH,
    "execute",
    "--manifest",
    FORMAL_MANIFEST_FILENAME,
    "--machine-attestation",
    "${OFC_HU_RL_C4_MACHINE_ATTESTATION}",
    "--output",
    "${OFC_HU_RL_C4_RESULT}",
)
STARTUP_REQUIRED_ENVIRONMENT: Final = (
    "OFC_HU_RL_C4_PACKAGE_ROOT",
    "OFC_HU_RL_C4_MACHINE_ATTESTATION",
    "OFC_HU_RL_C4_RESULT",
)

MAX_CANONICAL_INPUT_BYTES: Final = 8 * 1024 * 1024

_MANIFEST_FIELDS = {
    "schema",
    "status",
    "artifact_role",
    "scope",
    "cloud_operations_authorized",
    "network_access_required",
    "target",
    "execution",
    "gates",
    "benchmark_v3_contract",
    "startup",
    "benchmark_provenance",
    "harness_source_sha256",
    "result_schema",
    "manifest_sha256",
}
_TARGET_FIELDS = {
    "provider",
    "project_id",
    "zone",
    "machine_type",
    "vcpu_count",
    "system",
    "architecture",
    "python_implementation",
    "python_major",
    "python_minor",
    "python_abi",
    "provisioning_model",
    "nic_type",
    "boot_disk_type",
    "attestation_schema",
    "attestation_source",
    "external_attestation_required",
    "harness_cloud_lookup_performed",
}
_EXECUTION_FIELDS = {
    "seed",
    "chunk_width",
    "thread_count",
    "decision_count_per_lane",
    "rate_lane_counts",
    "rate_boundary_mode",
    "memory_lane_count",
}
_GATE_FIELDS = {
    "actor_decisions_per_second_minimum",
    "rss_bytes_maximum",
    "all_required",
}
_V3_CONTRACT_FIELDS = {
    "schema",
    "native_engine_required",
    "performance_gate_evaluated_by_v3",
    "lane_counts",
    "separate_combined_byte_exact_required",
}
_STARTUP_FIELDS = {
    "script_path",
    "script_sha256",
    "argv_template",
    "argv_template_sha256",
    "required_environment",
    "creates_cloud_resources",
    "performs_network_calls",
}
_ATTESTATION_FIELDS = {
    "schema",
    "source",
    "provider",
    "project_id",
    "zone",
    "instance_name",
    "instance_id",
    "status",
    "machine_type",
    "machine_type_uri",
    "vcpu_count",
    "cpu_platform",
    "provisioning_model",
    "preemptible",
    "deletion_protection",
    "nic_type",
    "boot_disk_type",
    "image_self_link",
    "image_id",
    "observed_at_utc",
    "controller_principal_sha256",
    "external_observation_schema",
    "external_observation_sha256",
    "manifest_sha256",
    "instance_identity_sha256",
    "attestation_sha256",
}
_EXTERNAL_OBSERVATION_FIELDS = {
    "schema",
    "source",
    "provider",
    "project_id",
    "zone",
    "instance_name",
    "instance_id",
    "status",
    "machine_type",
    "machine_type_uri",
    "vcpu_count",
    "cpu_platform",
    "provisioning_model",
    "preemptible",
    "deletion_protection",
    "nic_type",
    "boot_disk_type",
    "image_self_link",
    "image_id",
    "observed_at_utc",
    "controller_principal_sha256",
    "observation_sha256",
}
_MEMORY_PROBE_FIELDS = {
    "schema",
    "boundary_mode",
    "lane_count",
    "actor_decisions",
    "decision_counts_digest",
    "all_done",
    "v3_separate_combined_preflight_passed",
    "timings_seconds",
    "actor_decisions_per_second",
    "end_to_end_actor_decisions_per_second",
    "memory",
}
_RESULT_FIELDS = {
    "schema",
    "status",
    "artifact_role",
    "manifest_sha256",
    "machine_attestation",
    "benchmark_v3",
    "memory_probe_4096",
    "measurements",
    "gates",
    "runtime_binding",
    "cloud_mutations_performed",
    "harness_cloud_lookup_performed",
    "result_sha256",
}
_MEASUREMENT_FIELDS = {
    "minimum_combined_actor_decisions_per_second",
    "combined_rate_by_lane",
    "peak_rss_bytes_at_4096_lanes",
}
_RESULT_GATE_FIELDS = {
    "actor_decisions_per_second_minimum",
    "rss_bytes_maximum",
    "rate_pass",
    "rss_pass",
    "overall_pass",
}
_RUNTIME_FIELDS = {
    "system",
    "machine",
    "processor",
    "cpu_count",
    "python_version",
    "python_implementation",
    "topology_matches_contract",
}


class HuRlC4FormalBenchmarkError(ValueError):
    """The formal package, attestation, result, or local binding is invalid."""


ProvenanceProvider = Callable[[], Mapping[str, Any]]
HashProvider = Callable[[], Mapping[str, str]]
BenchmarkRunner = Callable[..., Mapping[str, Any]]
MemoryProbeRunner = Callable[..., Mapping[str, Any]]
RuntimeProvider = Callable[[], Mapping[str, Any]]


def build_c4_formal_benchmark_manifest(
    *,
    benchmark_provenance: Mapping[str, Any] | None = None,
    harness_source_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Freeze one local execution contract without launching anything."""

    provenance = dict(
        benchmark_provenance
        if benchmark_provenance is not None
        else collect_native_benchmark_provenance()
    )
    harness_hashes = dict(
        harness_source_sha256
        if harness_source_sha256 is not None
        else collect_c4_harness_source_hashes()
    )
    _validate_linux_native_provenance(provenance)
    _validate_harness_source_hashes(harness_hashes)
    manifest: dict[str, Any] = {
        "schema": HU_RL_C4_FORMAL_MANIFEST_SCHEMA,
        "status": "local_execution_package_manifest_ready",
        "artifact_role": "c4_standard_16_formal_benchmark_contract",
        "scope": "local_only_no_cloud_lifecycle_or_network",
        "cloud_operations_authorized": False,
        "network_access_required": False,
        "target": {
            "provider": TARGET_PROVIDER,
            "project_id": TARGET_PROJECT_ID,
            "zone": TARGET_ZONE,
            "machine_type": TARGET_MACHINE_TYPE,
            "vcpu_count": TARGET_VCPU_COUNT,
            "system": TARGET_SYSTEM,
            "architecture": TARGET_ARCHITECTURE,
            "python_implementation": TARGET_PYTHON_IMPLEMENTATION,
            "python_major": TARGET_PYTHON_MAJOR,
            "python_minor": TARGET_PYTHON_MINOR,
            "python_abi": TARGET_PYTHON_ABI,
            "provisioning_model": TARGET_PROVISIONING_MODEL,
            "nic_type": TARGET_NIC_TYPE,
            "boot_disk_type": TARGET_BOOT_DISK_TYPE,
            "attestation_schema": HU_RL_C4_MACHINE_ATTESTATION_SCHEMA,
            "attestation_source": MACHINE_ATTESTATION_SOURCE,
            "external_attestation_required": True,
            "harness_cloud_lookup_performed": False,
        },
        "execution": {
            "seed": FORMAL_SEED,
            "chunk_width": FORMAL_CHUNK_WIDTH,
            "thread_count": FORMAL_THREAD_COUNT,
            "decision_count_per_lane": DECISIONS_PER_HAND,
            "rate_lane_counts": list(BENCHMARK_LANE_COUNTS),
            "rate_boundary_mode": FORMAL_BOUNDARY_MODE,
            "memory_lane_count": FORMAL_MEMORY_LANE_COUNT,
        },
        "gates": {
            "actor_decisions_per_second_minimum": FORMAL_RATE_TARGET,
            "rss_bytes_maximum": FORMAL_RSS_LIMIT_BYTES,
            "all_required": True,
        },
        "benchmark_v3_contract": {
            "schema": HU_RL_NATIVE_BATCH_BENCHMARK_SCHEMA,
            "native_engine_required": True,
            "performance_gate_evaluated_by_v3": False,
            "lane_counts": list(BENCHMARK_LANE_COUNTS),
            "separate_combined_byte_exact_required": True,
        },
        "startup": {
            "script_path": STARTUP_SCRIPT_PATH,
            "script_sha256": harness_hashes[STARTUP_SCRIPT_PATH],
            "argv_template": list(STARTUP_ARGV_TEMPLATE),
            "argv_template_sha256": _canonical_digest(list(STARTUP_ARGV_TEMPLATE)),
            "required_environment": list(STARTUP_REQUIRED_ENVIRONMENT),
            "creates_cloud_resources": False,
            "performs_network_calls": False,
        },
        "benchmark_provenance": provenance,
        "harness_source_sha256": harness_hashes,
        "result_schema": HU_RL_C4_FORMAL_RESULT_SCHEMA,
        "manifest_sha256": None,
    }
    manifest["manifest_sha256"] = _self_digest(manifest, "manifest_sha256")
    validate_c4_formal_benchmark_manifest(manifest)
    return manifest


def validate_c4_formal_benchmark_manifest(manifest: Mapping[str, Any]) -> None:
    _require_mapping(manifest, "C4 formal manifest")
    _require_exact_fields(manifest, _MANIFEST_FIELDS, "C4 formal manifest")
    expected = {
        "schema": HU_RL_C4_FORMAL_MANIFEST_SCHEMA,
        "status": "local_execution_package_manifest_ready",
        "artifact_role": "c4_standard_16_formal_benchmark_contract",
        "scope": "local_only_no_cloud_lifecycle_or_network",
        "cloud_operations_authorized": False,
        "network_access_required": False,
        "result_schema": HU_RL_C4_FORMAL_RESULT_SCHEMA,
    }
    _require_literals(manifest, expected, "C4 formal manifest")

    target = manifest["target"]
    _require_mapping(target, "C4 target")
    _require_exact_fields(target, _TARGET_FIELDS, "C4 target")
    _require_literals(
        target,
        {
            "provider": TARGET_PROVIDER,
            "project_id": TARGET_PROJECT_ID,
            "zone": TARGET_ZONE,
            "machine_type": TARGET_MACHINE_TYPE,
            "vcpu_count": TARGET_VCPU_COUNT,
            "system": TARGET_SYSTEM,
            "architecture": TARGET_ARCHITECTURE,
            "python_implementation": TARGET_PYTHON_IMPLEMENTATION,
            "python_major": TARGET_PYTHON_MAJOR,
            "python_minor": TARGET_PYTHON_MINOR,
            "python_abi": TARGET_PYTHON_ABI,
            "provisioning_model": TARGET_PROVISIONING_MODEL,
            "nic_type": TARGET_NIC_TYPE,
            "boot_disk_type": TARGET_BOOT_DISK_TYPE,
            "attestation_schema": HU_RL_C4_MACHINE_ATTESTATION_SCHEMA,
            "attestation_source": MACHINE_ATTESTATION_SOURCE,
            "external_attestation_required": True,
            "harness_cloud_lookup_performed": False,
        },
        "C4 target",
    )
    execution = manifest["execution"]
    _require_mapping(execution, "C4 execution")
    _require_exact_fields(execution, _EXECUTION_FIELDS, "C4 execution")
    _require_literals(
        execution,
        {
            "seed": FORMAL_SEED,
            "chunk_width": FORMAL_CHUNK_WIDTH,
            "thread_count": FORMAL_THREAD_COUNT,
            "decision_count_per_lane": DECISIONS_PER_HAND,
            "rate_lane_counts": list(BENCHMARK_LANE_COUNTS),
            "rate_boundary_mode": FORMAL_BOUNDARY_MODE,
            "memory_lane_count": FORMAL_MEMORY_LANE_COUNT,
        },
        "C4 execution",
    )
    gates = manifest["gates"]
    _require_mapping(gates, "C4 gates")
    _require_exact_fields(gates, _GATE_FIELDS, "C4 gates")
    _require_literals(
        gates,
        {
            "actor_decisions_per_second_minimum": FORMAL_RATE_TARGET,
            "rss_bytes_maximum": FORMAL_RSS_LIMIT_BYTES,
            "all_required": True,
        },
        "C4 gates",
    )
    v3 = manifest["benchmark_v3_contract"]
    _require_mapping(v3, "C4 benchmark v3 contract")
    _require_exact_fields(v3, _V3_CONTRACT_FIELDS, "C4 benchmark v3 contract")
    _require_literals(
        v3,
        {
            "schema": HU_RL_NATIVE_BATCH_BENCHMARK_SCHEMA,
            "native_engine_required": True,
            "performance_gate_evaluated_by_v3": False,
            "lane_counts": list(BENCHMARK_LANE_COUNTS),
            "separate_combined_byte_exact_required": True,
        },
        "C4 benchmark v3 contract",
    )
    harness_hashes = manifest["harness_source_sha256"]
    _validate_harness_source_hashes(harness_hashes)
    startup = manifest["startup"]
    _require_mapping(startup, "C4 startup")
    _require_exact_fields(startup, _STARTUP_FIELDS, "C4 startup")
    _require_literals(
        startup,
        {
            "script_path": STARTUP_SCRIPT_PATH,
            "script_sha256": harness_hashes[STARTUP_SCRIPT_PATH],
            "argv_template": list(STARTUP_ARGV_TEMPLATE),
            "argv_template_sha256": _canonical_digest(list(STARTUP_ARGV_TEMPLATE)),
            "required_environment": list(STARTUP_REQUIRED_ENVIRONMENT),
            "creates_cloud_resources": False,
            "performs_network_calls": False,
        },
        "C4 startup",
    )
    _validate_linux_native_provenance(manifest["benchmark_provenance"])
    if not _is_sha256(manifest["manifest_sha256"]) or manifest[
        "manifest_sha256"
    ] != _self_digest(manifest, "manifest_sha256"):
        raise HuRlC4FormalBenchmarkError("C4 formal manifest digest mismatch")


def build_c4_machine_attestation(
    *, manifest_sha256: str, external_observation: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind a controller-produced GCE observation to one frozen manifest.

    This function performs no cloud lookup.  The external controller remains
    responsible for producing the canonical observation from authenticated
    read-only GCE API responses immediately before execution.
    """

    if not _is_sha256(manifest_sha256):
        raise HuRlC4FormalBenchmarkError("C4 manifest digest is invalid")
    validate_c4_external_gce_observation(external_observation)
    copied_fields = (
        "project_id",
        "zone",
        "instance_name",
        "instance_id",
        "status",
        "machine_type",
        "machine_type_uri",
        "vcpu_count",
        "cpu_platform",
        "provisioning_model",
        "preemptible",
        "deletion_protection",
        "nic_type",
        "boot_disk_type",
        "image_self_link",
        "image_id",
        "observed_at_utc",
        "controller_principal_sha256",
    )
    attestation: dict[str, Any] = {
        "schema": HU_RL_C4_MACHINE_ATTESTATION_SCHEMA,
        "source": MACHINE_ATTESTATION_SOURCE,
        "provider": TARGET_PROVIDER,
        **{field: external_observation[field] for field in copied_fields},
        "external_observation_schema": HU_RL_C4_EXTERNAL_GCE_OBSERVATION_SCHEMA,
        "external_observation_sha256": external_observation["observation_sha256"],
        "manifest_sha256": manifest_sha256,
        "instance_identity_sha256": _canonical_digest(
            {
                field: external_observation[field]
                for field in (
                    "project_id",
                    "zone",
                    "instance_name",
                    "instance_id",
                    "machine_type_uri",
                    "image_self_link",
                    "image_id",
                )
            }
        ),
        "attestation_sha256": None,
    }
    attestation["attestation_sha256"] = _self_digest(attestation, "attestation_sha256")
    validate_c4_machine_attestation(attestation, manifest_sha256=manifest_sha256)
    return attestation


def validate_c4_external_gce_observation(
    observation: Mapping[str, Any],
) -> None:
    _require_mapping(observation, "C4 external GCE observation")
    _require_exact_fields(
        observation,
        _EXTERNAL_OBSERVATION_FIELDS,
        "C4 external GCE observation",
    )
    _require_literals(
        observation,
        {
            "schema": HU_RL_C4_EXTERNAL_GCE_OBSERVATION_SCHEMA,
            "source": EXTERNAL_GCE_OBSERVATION_SOURCE,
            "provider": TARGET_PROVIDER,
            "project_id": TARGET_PROJECT_ID,
            "zone": TARGET_ZONE,
            "status": "RUNNING",
            "machine_type": TARGET_MACHINE_TYPE,
            "vcpu_count": TARGET_VCPU_COUNT,
            "provisioning_model": TARGET_PROVISIONING_MODEL,
            "preemptible": True,
            "deletion_protection": False,
            "nic_type": TARGET_NIC_TYPE,
            "boot_disk_type": TARGET_BOOT_DISK_TYPE,
        },
        "C4 external GCE observation",
    )
    instance_name = observation["instance_name"]
    instance_id = observation["instance_id"]
    machine_type_uri = observation["machine_type_uri"]
    image_self_link = observation["image_self_link"]
    image_id = observation["image_id"]
    observed_at = observation["observed_at_utc"]
    if (
        not isinstance(instance_name, str)
        or re.fullmatch(r"[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?", instance_name) is None
    ):
        raise HuRlC4FormalBenchmarkError("C4 observed instance name is invalid")
    if (
        not isinstance(instance_id, str)
        or re.fullmatch(r"[1-9][0-9]*", instance_id) is None
    ):
        raise HuRlC4FormalBenchmarkError("C4 observed instance id is invalid")
    expected_machine_suffix = (
        f"/projects/{TARGET_PROJECT_ID}/zones/{TARGET_ZONE}/machineTypes/"
        f"{TARGET_MACHINE_TYPE}"
    )
    if not isinstance(machine_type_uri, str) or not machine_type_uri.endswith(
        expected_machine_suffix
    ):
        raise HuRlC4FormalBenchmarkError("C4 observed machine type URI is invalid")
    if (
        not isinstance(observation["cpu_platform"], str)
        or not observation["cpu_platform"].strip()
    ):
        raise HuRlC4FormalBenchmarkError("C4 observed CPU platform is invalid")
    if (
        not isinstance(image_self_link, str)
        or not image_self_link.startswith(
            "https://www.googleapis.com/compute/v1/projects/"
        )
        or "/global/images/" not in image_self_link
    ):
        raise HuRlC4FormalBenchmarkError("C4 observed image self link is invalid")
    if not isinstance(image_id, str) or re.fullmatch(r"[1-9][0-9]*", image_id) is None:
        raise HuRlC4FormalBenchmarkError("C4 observed image id is invalid")
    if (
        not isinstance(observed_at, str)
        or re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,9})?Z",
            observed_at,
        )
        is None
    ):
        raise HuRlC4FormalBenchmarkError("C4 observation timestamp is invalid")
    if not _is_sha256(observation["controller_principal_sha256"]):
        raise HuRlC4FormalBenchmarkError("C4 controller principal digest is invalid")
    if not _is_sha256(observation["observation_sha256"]) or observation[
        "observation_sha256"
    ] != _self_digest(observation, "observation_sha256"):
        raise HuRlC4FormalBenchmarkError("C4 external observation digest mismatch")


def validate_c4_machine_attestation(
    attestation: Mapping[str, Any], *, manifest_sha256: str
) -> None:
    _require_mapping(attestation, "C4 machine attestation")
    _require_exact_fields(attestation, _ATTESTATION_FIELDS, "C4 machine attestation")
    _require_literals(
        attestation,
        {
            "schema": HU_RL_C4_MACHINE_ATTESTATION_SCHEMA,
            "source": MACHINE_ATTESTATION_SOURCE,
            "provider": TARGET_PROVIDER,
            "project_id": TARGET_PROJECT_ID,
            "zone": TARGET_ZONE,
            "status": "RUNNING",
            "machine_type": TARGET_MACHINE_TYPE,
            "vcpu_count": TARGET_VCPU_COUNT,
            "provisioning_model": TARGET_PROVISIONING_MODEL,
            "preemptible": True,
            "deletion_protection": False,
            "nic_type": TARGET_NIC_TYPE,
            "boot_disk_type": TARGET_BOOT_DISK_TYPE,
            "external_observation_schema": HU_RL_C4_EXTERNAL_GCE_OBSERVATION_SCHEMA,
            "manifest_sha256": manifest_sha256,
        },
        "C4 machine attestation",
    )
    reconstructed_observation = {
        "schema": attestation["external_observation_schema"],
        "source": EXTERNAL_GCE_OBSERVATION_SOURCE,
        "provider": attestation["provider"],
        **{
            field: attestation[field]
            for field in (
                "project_id",
                "zone",
                "instance_name",
                "instance_id",
                "status",
                "machine_type",
                "machine_type_uri",
                "vcpu_count",
                "cpu_platform",
                "provisioning_model",
                "preemptible",
                "deletion_protection",
                "nic_type",
                "boot_disk_type",
                "image_self_link",
                "image_id",
                "observed_at_utc",
                "controller_principal_sha256",
            )
        },
        "observation_sha256": attestation["external_observation_sha256"],
    }
    validate_c4_external_gce_observation(reconstructed_observation)
    expected_identity = _canonical_digest(
        {
            field: attestation[field]
            for field in (
                "project_id",
                "zone",
                "instance_name",
                "instance_id",
                "machine_type_uri",
                "image_self_link",
                "image_id",
            )
        }
    )
    if attestation["instance_identity_sha256"] != expected_identity:
        raise HuRlC4FormalBenchmarkError("C4 instance identity digest is invalid")
    if not _is_sha256(attestation["attestation_sha256"]) or attestation[
        "attestation_sha256"
    ] != _self_digest(attestation, "attestation_sha256"):
        raise HuRlC4FormalBenchmarkError("C4 machine attestation digest mismatch")


def run_c4_formal_benchmark(
    *,
    manifest: Mapping[str, Any],
    machine_attestation: Mapping[str, Any],
    _provenance_provider: ProvenanceProvider = collect_native_benchmark_provenance,
    _harness_hash_provider: HashProvider | None = None,
    _benchmark_runner: BenchmarkRunner = run_native_batch_mechanics_benchmark,
    _memory_probe_runner: MemoryProbeRunner | None = None,
    _runtime_provider: RuntimeProvider | None = None,
) -> dict[str, Any]:
    """Execute the fixed formal gate on an already provisioned local host."""

    validate_c4_formal_benchmark_manifest(manifest)
    manifest_sha256 = manifest["manifest_sha256"]
    validate_c4_machine_attestation(
        machine_attestation, manifest_sha256=manifest_sha256
    )
    hash_provider = _harness_hash_provider or collect_c4_harness_source_hashes
    runtime_provider = _runtime_provider or collect_c4_runtime_binding
    probe_runner = _memory_probe_runner or _run_c4_memory_probe
    _validate_current_package_binding(
        manifest,
        provenance_provider=_provenance_provider,
        harness_hash_provider=hash_provider,
    )
    runtime = dict(runtime_provider())
    _validate_runtime_binding(runtime)

    execution = manifest["execution"]
    benchmark_v3 = dict(
        _benchmark_runner(
            seed=execution["seed"],
            chunk_width=execution["chunk_width"],
            thread_count=execution["thread_count"],
        )
    )
    validate_native_batch_benchmark_document(benchmark_v3, require_native_engine=True)
    _validate_v3_against_manifest(benchmark_v3, manifest)
    probe = dict(
        probe_runner(
            seed=execution["seed"],
            chunk_width=execution["chunk_width"],
            thread_count=execution["thread_count"],
            v3_preflight_passed=True,
        )
    )
    validate_c4_memory_probe(probe)
    _validate_current_package_binding(
        manifest,
        provenance_provider=_provenance_provider,
        harness_hash_provider=hash_provider,
    )

    combined_rates = {
        str(result["lane_count"]): float(result["actor_decisions_per_second"])
        for result in benchmark_v3["results"]
        if result["boundary_mode"] == FORMAL_BOUNDARY_MODE
    }
    minimum_rate = min(combined_rates.values())
    peak_rss = probe["memory"]["peak_rss_bytes"]
    rate_pass = minimum_rate >= FORMAL_RATE_TARGET
    rss_pass = peak_rss <= FORMAL_RSS_LIMIT_BYTES
    overall_pass = rate_pass and rss_pass
    result: dict[str, Any] = {
        "schema": HU_RL_C4_FORMAL_RESULT_SCHEMA,
        "status": "formal_gate_pass" if overall_pass else "formal_gate_no_go",
        "artifact_role": "c4_standard_16_formal_performance_gate",
        "manifest_sha256": manifest_sha256,
        "machine_attestation": dict(machine_attestation),
        "benchmark_v3": benchmark_v3,
        "memory_probe_4096": probe,
        "measurements": {
            "minimum_combined_actor_decisions_per_second": minimum_rate,
            "combined_rate_by_lane": combined_rates,
            "peak_rss_bytes_at_4096_lanes": peak_rss,
        },
        "gates": {
            "actor_decisions_per_second_minimum": FORMAL_RATE_TARGET,
            "rss_bytes_maximum": FORMAL_RSS_LIMIT_BYTES,
            "rate_pass": rate_pass,
            "rss_pass": rss_pass,
            "overall_pass": overall_pass,
        },
        "runtime_binding": runtime,
        "cloud_mutations_performed": False,
        "harness_cloud_lookup_performed": False,
        "result_sha256": None,
    }
    result["result_sha256"] = _self_digest(result, "result_sha256")
    validate_c4_formal_benchmark_result(result, manifest=manifest)
    return result


def validate_c4_formal_benchmark_result(
    result: Mapping[str, Any], *, manifest: Mapping[str, Any]
) -> None:
    validate_c4_formal_benchmark_manifest(manifest)
    _require_mapping(result, "C4 formal result")
    _require_exact_fields(result, _RESULT_FIELDS, "C4 formal result")
    _require_literals(
        result,
        {
            "schema": HU_RL_C4_FORMAL_RESULT_SCHEMA,
            "artifact_role": "c4_standard_16_formal_performance_gate",
            "manifest_sha256": manifest["manifest_sha256"],
            "cloud_mutations_performed": False,
            "harness_cloud_lookup_performed": False,
        },
        "C4 formal result",
    )
    validate_c4_machine_attestation(
        result["machine_attestation"],
        manifest_sha256=manifest["manifest_sha256"],
    )
    benchmark_v3 = result["benchmark_v3"]
    validate_native_batch_benchmark_document(benchmark_v3, require_native_engine=True)
    _validate_v3_against_manifest(benchmark_v3, manifest)
    probe = result["memory_probe_4096"]
    validate_c4_memory_probe(probe)
    runtime = result["runtime_binding"]
    _validate_runtime_binding(runtime)

    combined_rates = {
        str(item["lane_count"]): float(item["actor_decisions_per_second"])
        for item in benchmark_v3["results"]
        if item["boundary_mode"] == FORMAL_BOUNDARY_MODE
    }
    if set(combined_rates) != {str(lane) for lane in BENCHMARK_LANE_COUNTS}:
        raise HuRlC4FormalBenchmarkError("C4 combined rate lanes changed")
    minimum_rate = min(combined_rates.values())
    peak_rss = probe["memory"]["peak_rss_bytes"]
    measurements = result["measurements"]
    _require_mapping(measurements, "C4 measurements")
    _require_exact_fields(measurements, _MEASUREMENT_FIELDS, "C4 measurements")
    if measurements["combined_rate_by_lane"] != combined_rates:
        raise HuRlC4FormalBenchmarkError("C4 combined rate summary mismatch")
    if not _same_finite_number(
        measurements["minimum_combined_actor_decisions_per_second"],
        minimum_rate,
    ):
        raise HuRlC4FormalBenchmarkError("C4 minimum throughput summary mismatch")
    if measurements["peak_rss_bytes_at_4096_lanes"] != peak_rss:
        raise HuRlC4FormalBenchmarkError("C4 peak RSS summary mismatch")

    rate_pass = minimum_rate >= FORMAL_RATE_TARGET
    rss_pass = peak_rss <= FORMAL_RSS_LIMIT_BYTES
    overall_pass = rate_pass and rss_pass
    gates = result["gates"]
    _require_mapping(gates, "C4 result gates")
    _require_exact_fields(gates, _RESULT_GATE_FIELDS, "C4 result gates")
    _require_literals(
        gates,
        {
            "actor_decisions_per_second_minimum": FORMAL_RATE_TARGET,
            "rss_bytes_maximum": FORMAL_RSS_LIMIT_BYTES,
            "rate_pass": rate_pass,
            "rss_pass": rss_pass,
            "overall_pass": overall_pass,
        },
        "C4 result gates",
    )
    expected_status = "formal_gate_pass" if overall_pass else "formal_gate_no_go"
    if result["status"] != expected_status:
        raise HuRlC4FormalBenchmarkError("C4 formal status disagrees with gates")
    if not _is_sha256(result["result_sha256"]) or result[
        "result_sha256"
    ] != _self_digest(result, "result_sha256"):
        raise HuRlC4FormalBenchmarkError("C4 formal result digest mismatch")


def validate_c4_memory_probe(probe: Mapping[str, Any]) -> None:
    _require_mapping(probe, "C4 memory probe")
    _require_exact_fields(probe, _MEMORY_PROBE_FIELDS, "C4 memory probe")
    _require_literals(
        probe,
        {
            "schema": HU_RL_C4_MEMORY_PROBE_SCHEMA,
            "boundary_mode": FORMAL_BOUNDARY_MODE,
            "lane_count": FORMAL_MEMORY_LANE_COUNT,
            "actor_decisions": FORMAL_MEMORY_LANE_COUNT * DECISIONS_PER_HAND,
            "all_done": True,
            "v3_separate_combined_preflight_passed": True,
        },
        "C4 memory probe",
    )
    raw_v3_lane_result = {
        "schema": HU_RL_NATIVE_BATCH_LANE_RESULT_SCHEMA,
        "boundary_mode": FORMAL_BOUNDARY_MODE,
        "lane_count": probe["lane_count"],
        "actor_decisions": probe["actor_decisions"],
        "decision_counts_digest": probe["decision_counts_digest"],
        "all_done": probe["all_done"],
        "separate_packed_byte_exact": probe["v3_separate_combined_preflight_passed"],
        "timings_seconds": probe["timings_seconds"],
        "actor_decisions_per_second": probe["actor_decisions_per_second"],
        "end_to_end_actor_decisions_per_second": probe[
            "end_to_end_actor_decisions_per_second"
        ],
        "diagnostic_disposition": (
            "rate_target_met_but_gate_not_evaluated"
            if probe["actor_decisions_per_second"]
            >= RL_READY_DIAGNOSTIC_TARGET_DECISIONS_PER_SECOND
            else "below_rate_target_no_go"
        ),
        "memory": probe["memory"],
    }
    try:
        _validate_lane_result(
            raw_v3_lane_result,
            expected_lane_count=FORMAL_MEMORY_LANE_COUNT,
            expected_boundary_mode=FORMAL_BOUNDARY_MODE,
        )
    except HuRlNativeBenchmarkError as exc:
        raise HuRlC4FormalBenchmarkError("C4 memory probe is invalid") from exc
    if probe["memory"]["source"] != "posix_rusage_statm":
        raise HuRlC4FormalBenchmarkError("C4 memory probe source is invalid")


def collect_c4_harness_source_hashes() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[2]
    result = {
        relative: _sha256_file(repository / Path(relative))
        for relative in sorted(HARNESS_SOURCE_PATHS)
    }
    _validate_harness_source_hashes(result)
    return result


def collect_c4_runtime_binding() -> dict[str, Any]:
    cpu_count = os.cpu_count() or 0
    system = platform.system()
    machine = platform.machine() or "unknown"
    return {
        "system": system,
        "machine": machine,
        "processor": platform.processor() or "unknown",
        "cpu_count": cpu_count,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "topology_matches_contract": (
            system == TARGET_SYSTEM
            and machine in {TARGET_ARCHITECTURE, "AMD64"}
            and cpu_count == TARGET_VCPU_COUNT
            and platform.python_implementation() == TARGET_PYTHON_IMPLEMENTATION
            and sys.version_info[:2] == (TARGET_PYTHON_MAJOR, TARGET_PYTHON_MINOR)
        ),
    }


def canonical_c4_json(value: Mapping[str, Any]) -> str:
    return _canonical_json(value)


def load_canonical_c4_json(path: Path, *, context: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise HuRlC4FormalBenchmarkError(f"{context} could not be read") from exc
    if not raw or len(raw) > MAX_CANONICAL_INPUT_BYTES:
        raise HuRlC4FormalBenchmarkError(f"{context} byte size is invalid")
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HuRlC4FormalBenchmarkError(f"{context} is not JSON") from exc
    _require_mapping(value, context)
    expected = (_canonical_json(value) + "\n").encode("ascii")
    if raw != expected:
        raise HuRlC4FormalBenchmarkError(f"{context} is not canonical byte-locked JSON")
    return dict(value)


def _run_c4_memory_probe(
    *,
    seed: int,
    chunk_width: int,
    thread_count: int,
    v3_preflight_passed: bool,
) -> dict[str, Any]:
    if v3_preflight_passed is not True:
        raise HuRlC4FormalBenchmarkError("C4 v3 preflight did not pass")
    try:
        raw = _run_lane_benchmark(
            boundary_mode=FORMAL_BOUNDARY_MODE,
            lane_count=FORMAL_MEMORY_LANE_COUNT,
            seed=seed,
            chunk_width=chunk_width,
            thread_count=thread_count,
            env_factory=NativeBatchHuRlEnvV1,
            memory_reader=read_process_memory,
            clock=time.perf_counter,
            separate_packed_byte_exact=True,
        )
    except HuRlNativeBenchmarkError as exc:
        raise HuRlC4FormalBenchmarkError("C4 4096-lane probe failed") from exc
    return {
        "schema": HU_RL_C4_MEMORY_PROBE_SCHEMA,
        "boundary_mode": raw["boundary_mode"],
        "lane_count": raw["lane_count"],
        "actor_decisions": raw["actor_decisions"],
        "decision_counts_digest": raw["decision_counts_digest"],
        "all_done": raw["all_done"],
        "v3_separate_combined_preflight_passed": True,
        "timings_seconds": raw["timings_seconds"],
        "actor_decisions_per_second": raw["actor_decisions_per_second"],
        "end_to_end_actor_decisions_per_second": raw[
            "end_to_end_actor_decisions_per_second"
        ],
        "memory": raw["memory"],
    }


def _validate_current_package_binding(
    manifest: Mapping[str, Any],
    *,
    provenance_provider: ProvenanceProvider,
    harness_hash_provider: HashProvider,
) -> None:
    current_provenance = dict(provenance_provider())
    current_hashes = dict(harness_hash_provider())
    if current_provenance != manifest["benchmark_provenance"]:
        raise HuRlC4FormalBenchmarkError("C4 benchmark wheel/source binding changed")
    if current_hashes != manifest["harness_source_sha256"]:
        raise HuRlC4FormalBenchmarkError("C4 harness source binding changed")


def _validate_v3_against_manifest(
    benchmark_v3: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    execution = manifest["execution"]
    benchmark_runtime = benchmark_v3["runtime"]
    if (
        benchmark_v3["seed"] != execution["seed"]
        or benchmark_v3["chunk_width"] != execution["chunk_width"]
        or benchmark_v3["thread_count"] != execution["thread_count"]
        or benchmark_v3["lane_counts"] != execution["rate_lane_counts"]
        or benchmark_v3["provenance"] != manifest["benchmark_provenance"]
        or benchmark_v3["performance_gate_evaluated"] is not False
        or benchmark_v3["performance_gate_pass"] is not None
        or benchmark_runtime["cpu_count"] != TARGET_VCPU_COUNT
        or not benchmark_runtime["platform"].startswith("Linux-")
        or benchmark_runtime["machine"] not in {TARGET_ARCHITECTURE, "AMD64"}
        or benchmark_runtime["python_implementation"] != TARGET_PYTHON_IMPLEMENTATION
        or not _is_target_python_version(benchmark_runtime["python_version"])
        or benchmark_runtime["memory_sources"] != ["posix_rusage_statm"]
        or any(
            value is not True
            for value in benchmark_v3["separate_packed_byte_exact_by_lane"].values()
        )
    ):
        raise HuRlC4FormalBenchmarkError(
            "packed benchmark v3 disagrees with C4 manifest"
        )


def _validate_runtime_binding(runtime: Mapping[str, Any]) -> None:
    _require_mapping(runtime, "C4 runtime binding")
    _require_exact_fields(runtime, _RUNTIME_FIELDS, "C4 runtime binding")
    for field in (
        "system",
        "machine",
        "processor",
        "python_version",
        "python_implementation",
    ):
        if not isinstance(runtime[field], str) or not runtime[field]:
            raise HuRlC4FormalBenchmarkError(f"C4 runtime {field} is invalid")
    if (
        runtime["system"] != TARGET_SYSTEM
        or runtime["machine"] not in {TARGET_ARCHITECTURE, "AMD64"}
        or type(runtime["cpu_count"]) is not int
        or runtime["cpu_count"] != TARGET_VCPU_COUNT
        or runtime["python_implementation"] != TARGET_PYTHON_IMPLEMENTATION
        or not _is_target_python_version(runtime["python_version"])
        or runtime["topology_matches_contract"] is not True
    ):
        raise HuRlC4FormalBenchmarkError("C4 runtime topology mismatch")


def _validate_linux_native_provenance(value: object) -> None:
    try:
        _validate_provenance(value)
    except HuRlNativeBenchmarkError as exc:
        raise HuRlC4FormalBenchmarkError("C4 benchmark provenance is invalid") from exc
    assert isinstance(value, Mapping)
    wheel = value["wheel_filename"]
    extension = value["native_extension_filename"]
    if (
        value["package_name"] != "ofc-hu-rl-engine-native"
        or value["package_version"] != "0.1.0"
        or not isinstance(wheel, str)
        or re.fullmatch(
            r"ofc_hu_rl_engine_native-0\.1\.0-cp311-cp311-"
            r"manylinux(?:_[0-9_]+)?_x86_64\.whl",
            wheel,
        )
        is None
        or not isinstance(extension, str)
        or extension != "_ofc_hu_rl_engine.cpython-311-x86_64-linux-gnu.so"
    ):
        raise HuRlC4FormalBenchmarkError(
            "C4 manifest requires the exact CPython 3.11 manylinux x86_64 native wheel"
        )
    if set(value["source_sha256"]) != set(_SOURCE_HASH_PATHS):
        raise HuRlC4FormalBenchmarkError("C4 benchmark source set changed")


def _validate_harness_source_hashes(value: object) -> None:
    _require_mapping(value, "C4 harness source hashes")
    if set(value) != set(HARNESS_SOURCE_PATHS) or any(
        not _is_sha256(digest) for digest in value.values()
    ):
        raise HuRlC4FormalBenchmarkError("C4 harness source hashes are invalid")


def _is_target_python_version(value: object) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(r"3\.11\.[0-9]+(?:[+._-][A-Za-z0-9._-]+)?", value) is not None
    )


def _require_literals(
    value: Mapping[str, Any], expected: Mapping[str, Any], context: str
) -> None:
    for field, required in expected.items():
        actual = value[field]
        if actual != required or (
            isinstance(required, bool) and type(actual) is not bool
        ):
            raise HuRlC4FormalBenchmarkError(f"{context} {field} is invalid")


def _same_finite_number(actual: object, expected: float) -> bool:
    return (
        type(actual) in (int, float)
        and math.isfinite(float(actual))
        and math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=0.0)
    )


def _self_digest(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload[field] = None
    return _canonical_digest(payload)


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise HuRlC4FormalBenchmarkError("C4 artifact is not canonical JSON") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise HuRlC4FormalBenchmarkError("C4 source file could not be read") from exc
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_mapping(value: object, context: str) -> None:
    if not isinstance(value, Mapping):
        raise HuRlC4FormalBenchmarkError(f"{context} must be a mapping")


def _require_exact_fields(
    value: Mapping[str, Any], expected: set[str], context: str
) -> None:
    if set(value) != expected:
        raise HuRlC4FormalBenchmarkError(f"{context} fields are invalid")
