from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from ofc_regular.hu_rl_c4_formal_benchmark import (
    FORMAL_CHUNK_WIDTH,
    FORMAL_MEMORY_LANE_COUNT,
    FORMAL_RATE_TARGET,
    FORMAL_RSS_LIMIT_BYTES,
    FORMAL_SEED,
    FORMAL_THREAD_COUNT,
    HARNESS_SOURCE_PATHS,
    HU_RL_C4_EXTERNAL_GCE_OBSERVATION_SCHEMA,
    HU_RL_C4_FORMAL_RESULT_SCHEMA,
    HU_RL_C4_MACHINE_ATTESTATION_SCHEMA,
    HU_RL_C4_MEMORY_PROBE_SCHEMA,
    RUNNER_SCRIPT_PATH,
    STARTUP_SCRIPT_PATH,
    TARGET_BOOT_DISK_TYPE,
    TARGET_MACHINE_TYPE,
    TARGET_NIC_TYPE,
    TARGET_PROJECT_ID,
    TARGET_ZONE,
    HuRlC4FormalBenchmarkError,
    _self_digest,
    build_c4_formal_benchmark_manifest,
    build_c4_machine_attestation,
    canonical_c4_json,
    collect_c4_harness_source_hashes,
    load_canonical_c4_json,
    run_c4_formal_benchmark,
    validate_c4_formal_benchmark_manifest,
    validate_c4_formal_benchmark_result,
    validate_c4_external_gce_observation,
    validate_c4_machine_attestation,
    validate_c4_memory_probe,
)
from ofc_regular.hu_rl_native_benchmark import (
    BENCHMARK_BOUNDARY_MODES,
    BENCHMARK_LANE_COUNTS,
    DECISIONS_PER_HAND,
    HU_RL_NATIVE_BATCH_BENCHMARK_SCHEMA,
    HU_RL_NATIVE_BATCH_LANE_RESULT_SCHEMA,
    HU_RL_NATIVE_BATCH_PROVENANCE_SCHEMA,
    _SOURCE_HASH_PATHS,
    _decision_counts_digest,
    _receipt_digest,
    validate_native_batch_benchmark_document,
)


def _provenance() -> dict[str, Any]:
    return {
        "schema": HU_RL_NATIVE_BATCH_PROVENANCE_SCHEMA,
        "package_name": "ofc-hu-rl-engine-native",
        "package_version": "0.1.0",
        "wheel_filename": (
            "ofc_hu_rl_engine_native-0.1.0-cp311-cp311-" "manylinux_2_34_x86_64.whl"
        ),
        "wheel_sha256": "1" * 64,
        "native_extension_filename": (
            "_ofc_hu_rl_engine.cpython-311-x86_64-linux-gnu.so"
        ),
        "native_extension_sha256": "2" * 64,
        "source_sha256": {path: "3" * 64 for path in _SOURCE_HASH_PATHS},
    }


def _harness_hashes() -> dict[str, str]:
    return {path: "4" * 64 for path in HARNESS_SOURCE_PATHS}


def _manifest() -> dict[str, Any]:
    return build_c4_formal_benchmark_manifest(
        benchmark_provenance=_provenance(),
        harness_source_sha256=_harness_hashes(),
    )


def _observation() -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": HU_RL_C4_EXTERNAL_GCE_OBSERVATION_SCHEMA,
        "source": "external_gce_instances_get_machine_types_get_disks_get_receipt",
        "provider": "gcp",
        "project_id": TARGET_PROJECT_ID,
        "zone": TARGET_ZONE,
        "instance_name": "hu-rl-c4-formal-20260722",
        "instance_id": "1234567890123456789",
        "status": "RUNNING",
        "machine_type": TARGET_MACHINE_TYPE,
        "machine_type_uri": (
            "https://www.googleapis.com/compute/v1/projects/"
            f"{TARGET_PROJECT_ID}/zones/{TARGET_ZONE}/machineTypes/"
            f"{TARGET_MACHINE_TYPE}"
        ),
        "vcpu_count": 16,
        "cpu_platform": "Intel Emerald Rapids",
        "provisioning_model": "SPOT",
        "preemptible": True,
        "deletion_protection": False,
        "nic_type": TARGET_NIC_TYPE,
        "boot_disk_type": TARGET_BOOT_DISK_TYPE,
        "image_self_link": (
            "https://www.googleapis.com/compute/v1/projects/debian-cloud/"
            "global/images/debian-12-bookworm-v20260721"
        ),
        "image_id": "9876543210987654321",
        "observed_at_utc": "2026-07-22T08:00:00Z",
        "controller_principal_sha256": "5" * 64,
        "observation_sha256": None,
    }
    value["observation_sha256"] = _self_digest(value, "observation_sha256")
    return value


def _attestation(manifest: dict[str, Any]) -> dict[str, Any]:
    return build_c4_machine_attestation(
        manifest_sha256=manifest["manifest_sha256"],
        external_observation=_observation(),
    )


def _memory(*, peak_rss: int) -> dict[str, Any]:
    baseline_rss = 64 * 1024 * 1024
    baseline_peak = 72 * 1024 * 1024
    max_sampled = min(peak_rss, max(baseline_rss, peak_rss - 1024))
    final_rss = min(max_sampled, max(baseline_rss, max_sampled - 1024))
    return {
        "source": "posix_rusage_statm",
        "sample_count": 24,
        "baseline_rss_bytes": baseline_rss,
        "baseline_peak_rss_bytes": baseline_peak,
        "final_rss_bytes": final_rss,
        "max_sampled_rss_bytes": max_sampled,
        "peak_rss_bytes": peak_rss,
        "peak_growth_bytes": peak_rss - baseline_peak,
    }


def _timings(*, actor_decisions: int, rate: float) -> dict[str, float]:
    boundary = actor_decisions / rate
    actor = boundary * 0.7
    step = boundary - actor
    decision_loop = boundary * 1.1
    deck_generation = 0.01
    construct = 0.01
    reset = 0.01
    return {
        "deck_generation": deck_generation,
        "construct": construct,
        "reset": reset,
        "actor_decision_total": actor,
        "step_total": step,
        "decision_loop": decision_loop,
        "total": deck_generation + construct + reset + decision_loop + 0.01,
    }


def _lane_result(
    *, lane_count: int, boundary_mode: str, rate: float, peak_rss: int
) -> dict[str, Any]:
    actor_decisions = lane_count * DECISIONS_PER_HAND
    timings = _timings(actor_decisions=actor_decisions, rate=rate)
    return {
        "schema": HU_RL_NATIVE_BATCH_LANE_RESULT_SCHEMA,
        "boundary_mode": boundary_mode,
        "lane_count": lane_count,
        "actor_decisions": actor_decisions,
        "decision_counts_digest": _decision_counts_digest(lane_count),
        "all_done": True,
        "separate_packed_byte_exact": True,
        "timings_seconds": timings,
        "actor_decisions_per_second": rate,
        "end_to_end_actor_decisions_per_second": (
            actor_decisions / timings["decision_loop"]
        ),
        "diagnostic_disposition": (
            "rate_target_met_but_gate_not_evaluated"
            if rate >= FORMAL_RATE_TARGET
            else "below_rate_target_no_go"
        ),
        "memory": _memory(peak_rss=peak_rss),
    }


def _benchmark_document(*, combined_rate: float = 30_000.0) -> dict[str, Any]:
    results = [
        _lane_result(
            lane_count=lane_count,
            boundary_mode=boundary_mode,
            rate=combined_rate if boundary_mode == "combined_packed" else 25_000.0,
            peak_rss=256 * 1024 * 1024,
        )
        for lane_count in BENCHMARK_LANE_COUNTS
        for boundary_mode in BENCHMARK_BOUNDARY_MODES
    ]
    document: dict[str, Any] = {
        "schema": HU_RL_NATIVE_BATCH_BENCHMARK_SCHEMA,
        "status": "local_diagnostic_complete",
        "artifact_role": "local_batch_mechanics_diagnostic_only",
        "scope": "strictly_local_no_cloud",
        "engine": "pyo3_native_combined_vs_separate_packed",
        "cloud_executable": False,
        "launch_authorized": False,
        "performance_gate_evaluated": False,
        "performance_gate_pass": None,
        "diagnostic_target_actor_decisions_per_second": FORMAL_RATE_TARGET,
        "diagnostic_disposition": (
            "rate_target_met_but_gate_not_evaluated"
            if combined_rate >= FORMAL_RATE_TARGET
            else "below_rate_target_no_go"
        ),
        "seed": FORMAL_SEED,
        "chunk_width": FORMAL_CHUNK_WIDTH,
        "thread_count": FORMAL_THREAD_COUNT,
        "decision_count_per_lane": DECISIONS_PER_HAND,
        "lane_counts": list(BENCHMARK_LANE_COUNTS),
        "separate_packed_byte_exact_by_lane": {
            str(lane): True for lane in BENCHMARK_LANE_COUNTS
        },
        "results": results,
        "provenance": _provenance(),
        "runtime": {
            "python_version": "3.11.13",
            "python_implementation": "CPython",
            "platform": "Linux-fixture",
            "machine": "x86_64",
            "processor": "fixture-cpu",
            "cpu_count": 16,
            "process_id": 123,
            "clock": "time.perf_counter",
            "native_available": True,
            "memory_sources": ["posix_rusage_statm"],
        },
        "receipt_sha256": None,
    }
    document["receipt_sha256"] = _receipt_digest(document)
    validate_native_batch_benchmark_document(document, require_native_engine=True)
    return document


def _probe(*, peak_rss: int = 1024 * 1024 * 1024) -> dict[str, Any]:
    actor_decisions = FORMAL_MEMORY_LANE_COUNT * DECISIONS_PER_HAND
    rate = 28_000.0
    timings = _timings(actor_decisions=actor_decisions, rate=rate)
    value = {
        "schema": HU_RL_C4_MEMORY_PROBE_SCHEMA,
        "boundary_mode": "combined_packed",
        "lane_count": FORMAL_MEMORY_LANE_COUNT,
        "actor_decisions": actor_decisions,
        "decision_counts_digest": _decision_counts_digest(FORMAL_MEMORY_LANE_COUNT),
        "all_done": True,
        "v3_separate_combined_preflight_passed": True,
        "timings_seconds": timings,
        "actor_decisions_per_second": rate,
        "end_to_end_actor_decisions_per_second": (
            actor_decisions / timings["decision_loop"]
        ),
        "memory": _memory(peak_rss=peak_rss),
    }
    validate_c4_memory_probe(value)
    return value


def _runtime() -> dict[str, Any]:
    return {
        "system": "Linux",
        "machine": "x86_64",
        "processor": "fixture-cpu",
        "cpu_count": 16,
        "python_version": "3.11.13",
        "python_implementation": "CPython",
        "topology_matches_contract": True,
    }


def _run(
    *, combined_rate: float = 30_000.0, peak_rss: int = 1024 * 1024 * 1024
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _manifest()
    result = run_c4_formal_benchmark(
        manifest=manifest,
        machine_attestation=_attestation(manifest),
        _provenance_provider=lambda: deepcopy(_provenance()),
        _harness_hash_provider=lambda: deepcopy(_harness_hashes()),
        _benchmark_runner=lambda **_kwargs: deepcopy(
            _benchmark_document(combined_rate=combined_rate)
        ),
        _memory_probe_runner=lambda **_kwargs: deepcopy(_probe(peak_rss=peak_rss)),
        _runtime_provider=lambda: deepcopy(_runtime()),
    )
    return manifest, result


def test_manifest_locks_target_startup_thresholds_and_local_only_scope() -> None:
    manifest = _manifest()
    validate_c4_formal_benchmark_manifest(manifest)
    assert manifest["target"]["machine_type"] == "c4-standard-16"
    assert manifest["target"]["vcpu_count"] == 16
    assert manifest["target"]["project_id"] == TARGET_PROJECT_ID
    assert manifest["target"]["zone"] == TARGET_ZONE
    assert manifest["target"]["python_implementation"] == "CPython"
    assert manifest["target"]["python_major"] == 3
    assert manifest["target"]["python_minor"] == 11
    assert manifest["target"]["python_abi"] == "cp311"
    assert manifest["target"]["provisioning_model"] == "SPOT"
    assert manifest["target"]["nic_type"] == "GVNIC"
    assert manifest["target"]["boot_disk_type"] == "hyperdisk-balanced"
    assert manifest["execution"]["thread_count"] == 16
    assert manifest["execution"]["memory_lane_count"] == 4096
    assert manifest["gates"]["actor_decisions_per_second_minimum"] == 20_000.0
    assert manifest["gates"]["rss_bytes_maximum"] == 2 * 1024**3
    assert manifest["cloud_operations_authorized"] is False
    assert manifest["network_access_required"] is False
    assert manifest["startup"]["script_path"] == STARTUP_SCRIPT_PATH
    assert RUNNER_SCRIPT_PATH in manifest["startup"]["argv_template"]
    assert manifest["startup"]["creates_cloud_resources"] is False
    assert manifest["startup"]["performs_network_calls"] is False


def test_manifest_requires_linux_native_wheel() -> None:
    provenance = _provenance()
    provenance["wheel_filename"] = "ofc_hu_rl_engine_native-cp313-win_amd64.whl"
    provenance["native_extension_filename"] = "_ofc_hu_rl_engine.pyd"
    with pytest.raises(HuRlC4FormalBenchmarkError, match="exact CPython 3.11"):
        build_c4_formal_benchmark_manifest(
            benchmark_provenance=provenance,
            harness_source_sha256=_harness_hashes(),
        )


def test_manifest_rejects_linux_cp313_or_generic_linux_wheel() -> None:
    for wheel, extension in (
        (
            "ofc_hu_rl_engine_native-0.1.0-cp313-cp313-manylinux_2_34_x86_64.whl",
            "_ofc_hu_rl_engine.cpython-313-x86_64-linux-gnu.so",
        ),
        (
            "ofc_hu_rl_engine_native-0.1.0-cp311-cp311-linux_x86_64.whl",
            "_ofc_hu_rl_engine.cpython-311-x86_64-linux-gnu.so",
        ),
    ):
        provenance = _provenance()
        provenance["wheel_filename"] = wheel
        provenance["native_extension_filename"] = extension
        with pytest.raises(HuRlC4FormalBenchmarkError, match="exact CPython 3.11"):
            build_c4_formal_benchmark_manifest(
                benchmark_provenance=provenance,
                harness_source_sha256=_harness_hashes(),
            )


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda value: value["target"].__setitem__("machine_type", "n2-standard-16"),
            "machine_type",
        ),
        (
            lambda value: value["gates"].__setitem__(
                "actor_decisions_per_second_minimum", 19_999.0
            ),
            "actor_decisions_per_second_minimum",
        ),
        (
            lambda value: value["startup"].__setitem__("performs_network_calls", True),
            "performs_network_calls",
        ),
        (
            lambda value: value.__setitem__("manifest_sha256", "0" * 64),
            "digest",
        ),
    ],
)
def test_manifest_tampering_fails_closed(mutator, match: str) -> None:
    manifest = _manifest()
    mutator(manifest)
    with pytest.raises(HuRlC4FormalBenchmarkError, match=match):
        validate_c4_formal_benchmark_manifest(manifest)


def test_external_attestation_is_required_and_bound_to_manifest() -> None:
    manifest = _manifest()
    attestation = _attestation(manifest)
    validate_c4_machine_attestation(
        attestation, manifest_sha256=manifest["manifest_sha256"]
    )
    attestation["machine_type"] = "c3-standard-176"
    with pytest.raises(HuRlC4FormalBenchmarkError, match="machine_type"):
        validate_c4_machine_attestation(
            attestation, manifest_sha256=manifest["manifest_sha256"]
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("provisioning_model", "STANDARD", "provisioning_model"),
        ("nic_type", "VIRTIO_NET", "nic_type"),
        ("boot_disk_type", "pd-balanced", "boot_disk_type"),
        ("status", "TERMINATED", "status"),
        ("vcpu_count", 8, "vcpu_count"),
    ],
)
def test_external_gce_observation_locks_spot_c4_platform(
    field: str, value: object, match: str
) -> None:
    observation = _observation()
    observation[field] = value
    observation["observation_sha256"] = _self_digest(observation, "observation_sha256")
    with pytest.raises(HuRlC4FormalBenchmarkError, match=match):
        validate_c4_external_gce_observation(observation)


def test_attestation_identity_is_derived_not_caller_selected() -> None:
    manifest = _manifest()
    attestation = _attestation(manifest)
    attestation["instance_identity_sha256"] = "6" * 64
    attestation["attestation_sha256"] = _self_digest(attestation, "attestation_sha256")
    with pytest.raises(HuRlC4FormalBenchmarkError, match="instance identity"):
        validate_c4_machine_attestation(
            attestation, manifest_sha256=manifest["manifest_sha256"]
        )


def test_formal_pass_requires_both_rate_and_4096_rss_gate() -> None:
    manifest, result = _run()
    validate_c4_formal_benchmark_result(result, manifest=manifest)
    assert result["schema"] == HU_RL_C4_FORMAL_RESULT_SCHEMA
    assert result["status"] == "formal_gate_pass"
    assert result["gates"] == {
        "actor_decisions_per_second_minimum": 20_000.0,
        "rss_bytes_maximum": 2 * 1024**3,
        "rate_pass": True,
        "rss_pass": True,
        "overall_pass": True,
    }
    assert result["cloud_mutations_performed"] is False
    assert result["harness_cloud_lookup_performed"] is False


@pytest.mark.parametrize(
    ("combined_rate", "peak_rss", "failed_gate"),
    [
        (19_999.0, 1024 * 1024 * 1024, "rate_pass"),
        (30_000.0, FORMAL_RSS_LIMIT_BYTES + 1, "rss_pass"),
    ],
)
def test_measured_gate_failure_is_valid_no_go_not_false_pass(
    combined_rate: float, peak_rss: int, failed_gate: str
) -> None:
    manifest, result = _run(combined_rate=combined_rate, peak_rss=peak_rss)
    validate_c4_formal_benchmark_result(result, manifest=manifest)
    assert result["status"] == "formal_gate_no_go"
    assert result["gates"][failed_gate] is False
    assert result["gates"]["overall_pass"] is False


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda value: value["gates"].__setitem__("overall_pass", False),
            "overall_pass",
        ),
        (
            lambda value: value["measurements"].__setitem__(
                "peak_rss_bytes_at_4096_lanes", 1
            ),
            "peak RSS summary",
        ),
        (
            lambda value: value["runtime_binding"].__setitem__("cpu_count", 15),
            "topology mismatch",
        ),
        (
            lambda value: value["runtime_binding"].__setitem__(
                "python_version", "3.13.5"
            ),
            "topology mismatch",
        ),
        (
            lambda value: value.__setitem__("result_sha256", "0" * 64),
            "result digest",
        ),
    ],
)
def test_result_tampering_fails_closed(mutator, match: str) -> None:
    manifest, result = _run()
    mutator(result)
    with pytest.raises(HuRlC4FormalBenchmarkError, match=match):
        validate_c4_formal_benchmark_result(result, manifest=manifest)


def test_package_binding_and_runtime_are_checked_before_measurement() -> None:
    manifest = _manifest()
    called = False

    def forbidden_runner(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("benchmark must not start")

    changed = _provenance()
    changed["wheel_sha256"] = "9" * 64
    with pytest.raises(HuRlC4FormalBenchmarkError, match="binding changed"):
        run_c4_formal_benchmark(
            manifest=manifest,
            machine_attestation=_attestation(manifest),
            _provenance_provider=lambda: changed,
            _harness_hash_provider=lambda: _harness_hashes(),
            _benchmark_runner=forbidden_runner,
            _runtime_provider=lambda: _runtime(),
        )
    assert called is False

    with pytest.raises(HuRlC4FormalBenchmarkError, match="topology mismatch"):
        run_c4_formal_benchmark(
            manifest=manifest,
            machine_attestation=_attestation(manifest),
            _provenance_provider=lambda: _provenance(),
            _harness_hash_provider=lambda: _harness_hashes(),
            _benchmark_runner=forbidden_runner,
            _runtime_provider=lambda: {**_runtime(), "cpu_count": 8},
        )
    assert called is False


def test_canonical_loader_rejects_noncanonical_or_modified_bytes(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    path = tmp_path / "manifest.json"
    path.write_bytes((canonical_c4_json(manifest) + "\n").encode("ascii"))
    loaded = load_canonical_c4_json(path, context="fixture manifest")
    assert loaded == manifest
    path.write_text(json.dumps(manifest, indent=2), encoding="ascii")
    with pytest.raises(HuRlC4FormalBenchmarkError, match="byte-locked"):
        load_canonical_c4_json(path, context="fixture manifest")


def test_startup_is_local_only_and_harness_sources_are_hashable() -> None:
    repository = Path(__file__).resolve().parents[1]
    startup = (repository / STARTUP_SCRIPT_PATH).read_text(encoding="utf-8")
    lowered = startup.lower()
    for forbidden in (
        "gcloud",
        "gsutil",
        "curl",
        "metadata.google.internal",
        "iam",
        "instances create",
        "instances delete",
    ):
        assert forbidden not in lowered
    hashes = collect_c4_harness_source_hashes()
    assert set(hashes) == set(HARNESS_SOURCE_PATHS)
    assert all(len(digest) == 64 for digest in hashes.values())


def test_result_contains_no_hidden_world_or_cloud_mutation_payload() -> None:
    _, result = _run()
    lowered = canonical_c4_json(result).lower()
    for forbidden in (
        "deck_tail",
        "opponent_private_discard",
        "world_state",
        "gcloud",
        "service_account_key",
        "access_token",
    ):
        assert forbidden not in lowered


def test_validate_cli_crosses_process_boundary_without_cloud_calls(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    manifest, result = _run()
    manifest_path = tmp_path / "manifest.json"
    result_path = tmp_path / "result.json"
    manifest_path.write_bytes((canonical_c4_json(manifest) + "\n").encode("ascii"))
    result_path.write_bytes((canonical_c4_json(result) + "\n").encode("ascii"))
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / RUNNER_SCRIPT_PATH),
            "validate",
            "--manifest",
            str(manifest_path),
            "--result",
            str(result_path),
        ],
        cwd=repository,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["status"] == "formal_gate_pass"
    assert summary["manifest_sha256"] == manifest["manifest_sha256"]
    assert summary["result_sha256"] == result["result_sha256"]


def test_attest_cli_crosses_process_boundary_without_cloud_calls(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    manifest = _manifest()
    observation = _observation()
    manifest_path = tmp_path / "manifest.json"
    observation_path = tmp_path / "observation.json"
    attestation_path = tmp_path / "attestation.json"
    manifest_path.write_bytes((canonical_c4_json(manifest) + "\n").encode("ascii"))
    observation_path.write_bytes(
        (canonical_c4_json(observation) + "\n").encode("ascii")
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / RUNNER_SCRIPT_PATH),
            "attest",
            "--manifest",
            str(manifest_path),
            "--external-observation",
            str(observation_path),
            "--output",
            str(attestation_path),
        ],
        cwd=repository,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    attestation = load_canonical_c4_json(
        attestation_path, context="C4 machine attestation"
    )
    validate_c4_machine_attestation(
        attestation, manifest_sha256=manifest["manifest_sha256"]
    )
    assert (
        attestation["external_observation_sha256"] == observation["observation_sha256"]
    )
