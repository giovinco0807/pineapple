"""Freeze Attempt07 preflight correctness and operational Go/No-Go.

This is the only producer of a development-Spot authorization.  It consumes
the five value-redacted proofs through their aggregate plus the five external
DONE records.  Timing and RSS decide only whether the bounded development100
run is operationally feasible; they are never policy-strength evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .aggregate_hu_m43_attempt07_preflight import validate_preflight_go_aggregate
from .hu_m43_attempt06_spot import (
    PINNED_MODEL_MANIFEST_SHA256,
    PINNED_NATIVE_MANIFEST_SHA256,
)
from .hu_m43_attempt06_teacher import ATTEMPT06_FROZEN_MODEL_SHA256
from .hu_m43_attempt07_contract import (
    AI_PROFILES_SHA256,
    M43_ATTEMPT07_PLAN_SHA256,
    load_and_validate_attempt07_plan,
)
from .hu_m43_attempt07_preflight_spot import (
    ATTEMPT06_BASE_MANIFEST_SHA256,
    ATTEMPT07_JOB_COUNT,
    ATTEMPT07_MACHINE_TYPE,
    DONE_SCHEMA,
    LAUNCH_AUTHORIZATION_SCHEMA,
    PACKAGE_MANIFEST_SCHEMA,
    _validate_authorization,
    build_preflight_schedule,
)
from .hu_m43_attempt07_spot import (
    EXPECTED_SHARDS as DEVELOPMENT_SHARDS,
    MAX_WAVE_SHARDS as DEVELOPMENT_MAX_WAVE_SHARDS,
    NATIVE_BATCH_THREADS as DEVELOPMENT_NATIVE_BATCH_THREADS,
    PACKAGE_MANIFEST_SCHEMA as DEVELOPMENT_PACKAGE_MANIFEST_SCHEMA,
    ROOTS_PER_SHARD as DEVELOPMENT_ROOTS_PER_SHARD,
    build_attempt07_spot_schedule,
)
from .run_hu_m43_attempt07_preflight import (
    ATTEMPT07_PREFLIGHT_SOURCE_SHA256,
    DEFAULT_ATTEMPT07_PLAN_PATH,
    DEFAULT_PREFLIGHT_PLAN_PATH,
    canonical_json_bytes,
    load_preflight_plan,
)


AUTHORIZATION_SCHEMA = "hu_m43_attempt07_development_spot_authorization_v1"
AUTHORIZED_STATUS = "authorized_after_attempt07_preflight"
NOT_AUTHORIZED_STATUS = "not_authorized_after_attempt07_preflight"

_SLOTS = (
    "root0_batch_a",
    "root0_batch_b",
    "root0_scalar",
    "root1_batch",
    "root2_batch",
)
_DONE_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "job_index",
        "job_id",
        "source_root_index",
        "batch_child_selectors",
        "native_batch_threads",
        "output_prefix",
        "output_sha256",
        "checkpoint_sha256",
        "heartbeat_sha256",
        "summary_sha256",
        "run_log_sha256",
        "manifest_sha256",
        "authorization_sha256",
        "schedule_sha256",
        "attempt07_plan_sha256",
        "source_merged_sha256",
        "model_sha256",
        "ai_profiles_sha256",
        "elapsed_seconds",
        "peak_rss_bytes",
        "teacher_values_exported",
        "arm_selection_performed",
        "new_root_generated",
        "current_profile_resolved",
        "current_profile_mutated",
        "runtime_policy_activated",
    }
)
_SHA_CHARS = frozenset("0123456789abcdef")
_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "jobs",
        "source_roots",
        "machine_type",
        "native_batch_threads",
        "base_attempt06_manifest_sha256",
        "base_attempt06_package_tree_sha256",
        "base_attempt06_source_zip_sha256",
        "package_tree_sha256",
        "overlay_closure_sha256",
        "source_zip_sha256",
        "source_zip_bytes",
        "startup_sha256",
        "schedule_sha256",
        "preflight_plan_sha256",
        "attempt07_plan_sha256",
        "source_merged_sha256",
        "model_sha256",
        "ai_profiles_sha256",
        "new_root_generated",
        "teacher_executed",
        "gcloud_invoked",
        "instances_created",
        "arm_selection_performed",
        "current_profile_resolved",
        "current_profile_mutated",
        "runtime_policy_activated",
    }
)
_LAUNCH_AUTHORIZATION_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "manifest_sha256",
        "preflight_plan_sha256",
        "attempt07_plan_sha256",
        "source_merged_sha256",
        "model_sha256",
        "ai_profiles_sha256",
        "machine_type",
        "native_batch_threads",
        "jobs",
        "source_roots",
        "local_gates",
        "local_evidence",
        "actual_scalar_batch_result",
        "actual_operational_go_no_go",
        "spot_authorized",
        "new_root_generation_allowed",
        "arm_selection_allowed",
        "current_profile_resolved",
        "current_profile_mutated",
        "runtime_policy_activated",
    }
)
_DEVELOPMENT_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "plan_sha256",
        "status_sha256",
        "model_sha256",
        "ai_profiles_sha256",
        "preflight_aggregate_sha256",
        "preflight_plan_sha256",
        "preflight_manifest_sha256",
        "preflight_schedule_sha256",
        "preflight_launch_authorization_sha256",
        "preflight_done_sha256",
        "schedule_sha256",
        "source_closure_sha256",
        "source_zip_sha256",
        "startup_sha256",
        "source_model_manifest_sha256",
        "source_native_manifest_sha256",
        "total_roots",
        "total_shards",
        "roots_per_shard",
        "root_profile_assignment",
        "batch_child_selectors",
        "native_batch_threads",
        "recommended_machine_type",
        "recommended_wave_shards",
        "fresh_root_opened",
        "teacher_executed",
        "gcloud_invoked",
        "instances_created",
        "current_profile_mutated",
        "runtime_policy_activated",
    }
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_canonical(path: str | Path, label: str) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not JSON") from exc
    if not isinstance(payload, dict) or raw != canonical_json_bytes(payload):
        raise ValueError(f"{label} is not canonical JSON")
    return payload


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA_CHARS for character in value)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def _elapsed(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _rss(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _atomic_write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Attempt07 preflight finalization exists: {path}")
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Attempt07 preflight finalization exists: {path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _distinct(paths: Mapping[str, Path]) -> None:
    identities = [
        os.path.normcase(str(path.resolve(strict=False))) for path in paths.values()
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("Attempt07 finalizer inputs must be distinct")
    items = tuple(paths.items())
    for left_index, (_, left) in enumerate(items):
        for _, right in items[left_index + 1 :]:
            if left.exists() and right.exists() and os.path.samefile(left, right):
                raise ValueError("Attempt07 finalizer inputs must be physically distinct")


def finalize_attempt07_preflight(
    *,
    proof_aggregate: str | Path,
    preflight_manifest: str | Path,
    preflight_schedule: str | Path,
    preflight_launch_authorization: str | Path,
    development_manifest: str | Path,
    development_schedule: str | Path,
    done_paths: Mapping[str, str | Path],
    output: str | Path,
    preflight_plan: str | Path = DEFAULT_PREFLIGHT_PLAN_PATH,
    attempt07_plan: str | Path = DEFAULT_ATTEMPT07_PLAN_PATH,
) -> dict[str, Any]:
    """Write one immutable operational authorization or No-Go receipt."""

    if set(done_paths) != set(_SLOTS):
        raise ValueError("Attempt07 finalizer requires exactly five frozen DONE slots")
    inputs = {
        "proof_aggregate": Path(proof_aggregate),
        "preflight_manifest": Path(preflight_manifest),
        "preflight_schedule": Path(preflight_schedule),
        "preflight_launch_authorization": Path(preflight_launch_authorization),
        "preflight_plan": Path(preflight_plan),
        "attempt07_plan": Path(attempt07_plan),
        "development_manifest": Path(development_manifest),
        "development_schedule": Path(development_schedule),
        **{f"done:{slot}": Path(path) for slot, path in done_paths.items()},
    }
    output_path = Path(output)
    _distinct({**inputs, "output": output_path})
    if output_path.exists():
        raise FileExistsError(f"Attempt07 preflight finalization exists: {output_path}")

    plan = load_preflight_plan(preflight_plan)
    if _sha256_file(attempt07_plan) != M43_ATTEMPT07_PLAN_SHA256:
        raise ValueError("Attempt07 development plan changed")
    limits = plan["operational_go_no_go"]
    manifest = _load_canonical(preflight_manifest, "preflight manifest")
    schedule_raw = Path(preflight_schedule).read_bytes()
    try:
        schedule = tuple(
            json.loads(line)
            for line in schedule_raw.decode("utf-8").splitlines()
            if line.strip()
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Attempt07 preflight schedule is not canonical JSONL") from exc
    expected_schedule = build_preflight_schedule()
    if (
        schedule != expected_schedule
        or schedule_raw != b"".join(canonical_json_bytes(row) for row in schedule)
    ):
        raise ValueError("Attempt07 preflight schedule changed")
    manifest_sha = _sha256_file(preflight_manifest)
    schedule_sha = _sha256_file(preflight_schedule)
    preflight_plan_sha = _sha256_file(preflight_plan)
    if (
        set(manifest) != _MANIFEST_KEYS
        or manifest.get("schema") != PACKAGE_MANIFEST_SCHEMA
        or manifest.get("status")
        != "packaged_attempt06_copy_plus_overlay_without_execution"
        or not isinstance(manifest.get("run_name"), str)
        or not manifest["run_name"]
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
            for character in manifest["run_name"]
        )
        or type(manifest.get("jobs")) is not int
        or manifest.get("jobs") != ATTEMPT07_JOB_COUNT
        or manifest.get("source_roots") != [0, 1, 2]
        or manifest.get("machine_type") != ATTEMPT07_MACHINE_TYPE
        or type(manifest.get("native_batch_threads")) is not int
        or manifest.get("native_batch_threads") != 4
        or manifest.get("base_attempt06_manifest_sha256")
        != ATTEMPT06_BASE_MANIFEST_SHA256
        or manifest.get("attempt07_plan_sha256") != M43_ATTEMPT07_PLAN_SHA256
        or manifest.get("preflight_plan_sha256") != preflight_plan_sha
        or manifest.get("schedule_sha256") != schedule_sha
        or manifest.get("source_merged_sha256") != ATTEMPT07_PREFLIGHT_SOURCE_SHA256
        or manifest.get("model_sha256") != ATTEMPT06_FROZEN_MODEL_SHA256
        or manifest.get("ai_profiles_sha256") != AI_PROFILES_SHA256
        or manifest.get("new_root_generated") is not False
        or manifest.get("teacher_executed") is not False
        or manifest.get("gcloud_invoked") is not False
        or manifest.get("instances_created") is not False
        or manifest.get("arm_selection_performed") is not False
        or manifest.get("current_profile_resolved") is not False
        or manifest.get("current_profile_mutated") is not False
        or manifest.get("runtime_policy_activated") is not False
        or type(manifest.get("source_zip_bytes")) is not int
        or manifest.get("source_zip_bytes", 0) <= 0
    ):
        raise ValueError("Attempt07 preflight package manifest changed")
    for key in (
        "base_attempt06_package_tree_sha256",
        "base_attempt06_source_zip_sha256",
        "package_tree_sha256",
        "overlay_closure_sha256",
        "source_zip_sha256",
        "startup_sha256",
        "schedule_sha256",
        "preflight_plan_sha256",
        "attempt07_plan_sha256",
        "source_merged_sha256",
        "model_sha256",
        "ai_profiles_sha256",
    ):
        _sha(manifest.get(key), f"preflight manifest.{key}")
    launch_authorization_sha = _validate_authorization(
        Path(preflight_launch_authorization),
        manifest=manifest,
        manifest_sha256=manifest_sha,
    )
    launch_authorization = _load_canonical(
        preflight_launch_authorization, "preflight launch authorization"
    )
    if (
        set(launch_authorization) != _LAUNCH_AUTHORIZATION_KEYS
        or launch_authorization.get("source_roots") != [0, 1, 2]
    ):
        raise ValueError("Attempt07 launch authorization fields changed")
    evidence = launch_authorization.get("local_evidence")
    expected_passes = {"attempt07_pytest": 99, "rust_parity": 9}
    if not isinstance(evidence, Mapping) or set(evidence) != {
        "attempt07_pytest",
        "rust_parity",
        "package_tests",
    }:
        raise ValueError("Attempt07 launch authorization evidence changed")
    for name, row in evidence.items():
        if (
            not isinstance(row, Mapping)
            or set(row) != {"receipt_sha256", "passed", "failed"}
            or type(row.get("passed")) is not int
            or row.get("passed", 0) < 1
            or row.get("passed") != expected_passes.get(name, row.get("passed"))
            or type(row.get("failed")) is not int
            or row.get("failed") != 0
        ):
            raise ValueError("Attempt07 launch authorization evidence changed")
        _sha(row.get("receipt_sha256"), f"launch authorization evidence.{name}")

    aggregate = _load_canonical(proof_aggregate, "preflight proof aggregate")
    validate_preflight_go_aggregate(
        aggregate,
        preflight_plan_sha256=preflight_plan_sha,
        require_operational_jobs=True,
    )
    aggregate_sha = _sha256_file(proof_aggregate)

    done_sha: dict[str, str] = {}
    elapsed: dict[str, float] = {}
    peak_rss: dict[str, int] = {}
    proof_hashes = aggregate.get("proof_file_sha256")
    if not isinstance(proof_hashes, Mapping) or set(proof_hashes) != set(_SLOTS):
        raise ValueError("Attempt07 aggregate proof hashes changed")
    run_name = manifest.get("run_name")
    for spec, slot in zip(expected_schedule, _SLOTS, strict=True):
        done = _load_canonical(done_paths[slot], f"DONE {slot}")
        if set(done) != _DONE_KEYS:
            raise ValueError(f"Attempt07 DONE fields changed: {slot}")
        fixed = {
            "schema": DONE_SCHEMA,
            "status": "complete",
            "run_name": run_name,
            "job_index": spec["job_index"],
            "job_id": spec["job_id"],
            "source_root_index": spec["source_root_index"],
            "batch_child_selectors": spec["batch_child_selectors"],
            "native_batch_threads": 4,
            "output_prefix": spec["output_prefix"],
            "manifest_sha256": manifest_sha,
            "authorization_sha256": launch_authorization_sha,
            "schedule_sha256": schedule_sha,
            "attempt07_plan_sha256": M43_ATTEMPT07_PLAN_SHA256,
            "source_merged_sha256": ATTEMPT07_PREFLIGHT_SOURCE_SHA256,
            "model_sha256": ATTEMPT06_FROZEN_MODEL_SHA256,
            "ai_profiles_sha256": AI_PROFILES_SHA256,
            "teacher_values_exported": False,
            "arm_selection_performed": False,
            "new_root_generated": False,
            "current_profile_resolved": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        }
        for key, expected in fixed.items():
            if done.get(key) != expected or type(done.get(key)) is not type(expected):
                raise ValueError(f"Attempt07 DONE identity changed: {slot}.{key}")
        for key in (
            "output_sha256",
            "checkpoint_sha256",
            "heartbeat_sha256",
            "summary_sha256",
            "run_log_sha256",
        ):
            _sha(done.get(key), f"{slot}.{key}")
        if proof_hashes[slot] != done["output_sha256"]:
            raise ValueError(f"Attempt07 aggregate/DONE proof mismatch: {slot}")
        elapsed[slot] = _elapsed(done.get("elapsed_seconds"), f"{slot}.elapsed")
        peak_rss[slot] = _rss(done.get("peak_rss_bytes"), f"{slot}.rss")
        done_sha[slot] = _sha256_file(done_paths[slot])

    operational = aggregate.get("operational_diagnostics")
    if (
        not isinstance(operational, Mapping)
        or operational.get("status") != "ok"
        or operational.get("job_count") != ATTEMPT07_JOB_COUNT
        or operational.get("science_decision_input") is not False
        or set(operational.get("jobs", {})) != set(_SLOTS)
    ):
        raise ValueError("Attempt07 aggregate lacks all five operational records")
    for slot in _SLOTS:
        if operational["jobs"][slot] != {
            "elapsed_seconds": elapsed[slot],
            "peak_rss_bytes": peak_rss[slot],
        }:
            raise ValueError(f"Attempt07 aggregate/DONE operational mismatch: {slot}")

    development_manifest_payload = _load_canonical(
        development_manifest, "Attempt07 development package manifest"
    )
    development_schedule_raw = Path(development_schedule).read_bytes()
    try:
        development_schedule_rows = [
            json.loads(line)
            for line in development_schedule_raw.decode("utf-8").splitlines()
            if line.strip()
        ]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Attempt07 development schedule is not canonical JSONL") from exc
    attempt07_plan_payload = load_and_validate_attempt07_plan(attempt07_plan)
    expected_development_schedule = build_attempt07_spot_schedule(
        attempt07_plan_payload
    )
    if (
        development_schedule_rows != expected_development_schedule
        or development_schedule_raw
        != b"".join(
            canonical_json_bytes(row) for row in development_schedule_rows
        )
    ):
        raise ValueError("Attempt07 development schedule changed")
    development_schedule_sha = _sha256_file(development_schedule)
    development_manifest_sha = _sha256_file(development_manifest)
    development_run_name = development_manifest_payload.get("run_name")
    if (
        set(development_manifest_payload) != _DEVELOPMENT_MANIFEST_KEYS
        or development_manifest_payload.get("schema")
        != DEVELOPMENT_PACKAGE_MANIFEST_SCHEMA
        or development_manifest_payload.get("status")
        != "frozen_package_only_no_root_opened"
        or not isinstance(development_run_name, str)
        or not development_run_name
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
            for character in development_run_name
        )
        or development_manifest_payload.get("plan_sha256")
        != M43_ATTEMPT07_PLAN_SHA256
        or development_manifest_payload.get("model_sha256")
        != ATTEMPT06_FROZEN_MODEL_SHA256
        or development_manifest_payload.get("ai_profiles_sha256")
        != AI_PROFILES_SHA256
        or development_manifest_payload.get("preflight_aggregate_sha256")
        != aggregate_sha
        or development_manifest_payload.get("preflight_plan_sha256")
        != preflight_plan_sha
        or development_manifest_payload.get("preflight_manifest_sha256")
        != manifest_sha
        or development_manifest_payload.get("preflight_schedule_sha256")
        != schedule_sha
        or development_manifest_payload.get(
            "preflight_launch_authorization_sha256"
        )
        != launch_authorization_sha
        or development_manifest_payload.get("preflight_done_sha256") != done_sha
        or development_manifest_payload.get("schedule_sha256")
        != development_schedule_sha
        or development_manifest_payload.get("source_model_manifest_sha256")
        != PINNED_MODEL_MANIFEST_SHA256
        or development_manifest_payload.get("source_native_manifest_sha256")
        != PINNED_NATIVE_MANIFEST_SHA256
        or type(development_manifest_payload.get("total_roots")) is not int
        or development_manifest_payload.get("total_roots") != DEVELOPMENT_SHARDS
        or type(development_manifest_payload.get("total_shards")) is not int
        or development_manifest_payload.get("total_shards") != DEVELOPMENT_SHARDS
        or type(development_manifest_payload.get("roots_per_shard")) is not int
        or development_manifest_payload.get("roots_per_shard")
        != DEVELOPMENT_ROOTS_PER_SHARD
        or development_manifest_payload.get("root_profile_assignment")
        != "root_index_mod_5_in_frozen_profile_order"
        or development_manifest_payload.get("batch_child_selectors") is not True
        or type(development_manifest_payload.get("native_batch_threads")) is not int
        or development_manifest_payload.get("native_batch_threads")
        != DEVELOPMENT_NATIVE_BATCH_THREADS
        or development_manifest_payload.get("recommended_machine_type")
        != "c4-standard-4"
        or type(development_manifest_payload.get("recommended_wave_shards"))
        is not int
        or development_manifest_payload.get("recommended_wave_shards")
        != DEVELOPMENT_MAX_WAVE_SHARDS
        or development_manifest_payload.get("fresh_root_opened") is not False
        or development_manifest_payload.get("teacher_executed") is not False
        or development_manifest_payload.get("gcloud_invoked") is not False
        or development_manifest_payload.get("instances_created") is not False
        or development_manifest_payload.get("current_profile_mutated") is not False
        or development_manifest_payload.get("runtime_policy_activated") is not False
    ):
        raise ValueError("Attempt07 development package manifest changed")
    for key in (
        "status_sha256",
        "schedule_sha256",
        "source_closure_sha256",
        "source_zip_sha256",
        "startup_sha256",
        "source_model_manifest_sha256",
        "source_native_manifest_sha256",
    ):
        _sha(
            development_manifest_payload.get(key),
            f"development manifest.{key}",
        )

    root0_batch = [elapsed["root0_batch_a"], elapsed["root0_batch_b"]]
    batch_median = float(statistics.median(root0_batch))
    replicate_ratio = max(root0_batch) / min(root0_batch)
    speedup = elapsed["root0_scalar"] / batch_median
    batch_elapsed_max = max(
        elapsed[slot]
        for slot in ("root0_batch_a", "root0_batch_b", "root1_batch", "root2_batch")
    )
    peak_rss_max = max(peak_rss.values())
    metrics = {
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": peak_rss,
        "root0_batch_median_elapsed_seconds": batch_median,
        "root0_batch_replicate_elapsed_ratio": replicate_ratio,
        "scalar_to_root0_batch_median_speedup": speedup,
        "batch_elapsed_seconds_max": batch_elapsed_max,
        "peak_rss_bytes_max": peak_rss_max,
    }

    def gate(name: str, passed: bool, observed: Any, requirement: str) -> dict[str, Any]:
        return {
            "name": name,
            "passed": bool(passed),
            "observed": observed,
            "requirement": requirement,
        }

    gates = {
        "proof_aggregate_go": gate(
            "proof_aggregate_go", True, "go", "go"
        ),
        "all_five_done_metadata": gate(
            "all_five_done_metadata", True, len(done_sha), "= 5"
        ),
        "batch_elapsed_seconds_per_root": gate(
            "batch_elapsed_seconds_per_root",
            batch_elapsed_max <= float(limits["batch_elapsed_seconds_per_root_max"]),
            batch_elapsed_max,
            f"<= {limits['batch_elapsed_seconds_per_root_max']}",
        ),
        "scalar_elapsed_seconds_root0": gate(
            "scalar_elapsed_seconds_root0",
            elapsed["root0_scalar"] <= float(limits["scalar_elapsed_seconds_root0_max"]),
            elapsed["root0_scalar"],
            f"<= {limits['scalar_elapsed_seconds_root0_max']}",
        ),
        "scalar_to_root0_batch_median_speedup": gate(
            "scalar_to_root0_batch_median_speedup",
            speedup >= float(limits["scalar_to_root0_batch_median_speedup_min"]),
            speedup,
            f">= {limits['scalar_to_root0_batch_median_speedup_min']}",
        ),
        "root0_batch_replicate_elapsed_ratio": gate(
            "root0_batch_replicate_elapsed_ratio",
            replicate_ratio <= float(limits["root0_batch_replicate_elapsed_ratio_max"]),
            replicate_ratio,
            f"<= {limits['root0_batch_replicate_elapsed_ratio_max']}",
        ),
        "peak_rss_bytes_per_job": gate(
            "peak_rss_bytes_per_job",
            peak_rss_max <= int(limits["peak_rss_bytes_per_job_max"]),
            peak_rss_max,
            f"<= {limits['peak_rss_bytes_per_job_max']}",
        ),
    }
    all_passed = all(item["passed"] for item in gates.values())
    report = {
        "schema": AUTHORIZATION_SCHEMA,
        "status": AUTHORIZED_STATUS if all_passed else NOT_AUTHORIZED_STATUS,
        "spot_authorized": all_passed,
        "attempt07_plan_sha256": M43_ATTEMPT07_PLAN_SHA256,
        "preflight_plan_sha256": preflight_plan_sha,
        "preflight_aggregate_sha256": aggregate_sha,
        "preflight_manifest_sha256": manifest_sha,
        "preflight_schedule_sha256": schedule_sha,
        "preflight_launch_authorization_sha256": launch_authorization_sha,
        "done_sha256": done_sha,
        "development_run_name": development_run_name,
        "development_manifest_sha256": development_manifest_sha,
        "development_schedule_sha256": development_schedule_sha,
        "development_source_closure_sha256": development_manifest_payload[
            "source_closure_sha256"
        ],
        "development_source_zip_sha256": development_manifest_payload[
            "source_zip_sha256"
        ],
        "development_startup_sha256": development_manifest_payload[
            "startup_sha256"
        ],
        "development_status_sha256": development_manifest_payload[
            "status_sha256"
        ],
        "development_source_model_manifest_sha256": development_manifest_payload[
            "source_model_manifest_sha256"
        ],
        "development_source_native_manifest_sha256": development_manifest_payload[
            "source_native_manifest_sha256"
        ],
        "development_total_roots": DEVELOPMENT_SHARDS,
        "development_total_shards": DEVELOPMENT_SHARDS,
        "development_roots_per_shard": DEVELOPMENT_ROOTS_PER_SHARD,
        "development_native_batch_threads": DEVELOPMENT_NATIVE_BATCH_THREADS,
        "development_max_wave_shards": DEVELOPMENT_MAX_WAVE_SHARDS,
        "development_machine_type": "c4-standard-4",
        "development_root_profile_assignment": (
            "root_index_mod_5_in_frozen_profile_order"
        ),
        "development_batch_child_selectors": True,
        "development_package_frozen_before_authorization": True,
        "operational_metrics": metrics,
        "operational_gates": gates,
        "all_gates_passed": all_passed,
        "development_started": False,
        "fresh_development_root_opened": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    _atomic_write_once(output_path, canonical_json_bytes(report))
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof-aggregate", required=True)
    parser.add_argument("--preflight-manifest", required=True)
    parser.add_argument("--preflight-schedule", required=True)
    parser.add_argument("--preflight-launch-authorization", required=True)
    parser.add_argument("--development-manifest", required=True)
    parser.add_argument("--development-schedule", required=True)
    parser.add_argument("--preflight-plan", default=str(DEFAULT_PREFLIGHT_PLAN_PATH))
    parser.add_argument("--attempt07-plan", default=str(DEFAULT_ATTEMPT07_PLAN_PATH))
    for slot in _SLOTS:
        parser.add_argument(f"--done-{slot.replace('_', '-')}", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    done = {
        slot: getattr(args, f"done_{slot}")
        for slot in _SLOTS
    }
    finalize_attempt07_preflight(
        proof_aggregate=args.proof_aggregate,
        preflight_manifest=args.preflight_manifest,
        preflight_schedule=args.preflight_schedule,
        preflight_launch_authorization=args.preflight_launch_authorization,
        development_manifest=args.development_manifest,
        development_schedule=args.development_schedule,
        preflight_plan=args.preflight_plan,
        attempt07_plan=args.attempt07_plan,
        done_paths=done,
        output=args.output,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "AUTHORIZATION_SCHEMA",
    "AUTHORIZED_STATUS",
    "NOT_AUTHORIZED_STATUS",
    "finalize_attempt07_preflight",
    "main",
]
