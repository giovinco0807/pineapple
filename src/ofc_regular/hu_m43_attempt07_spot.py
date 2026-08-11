"""Immutable Spot packaging and receive validation for Attempt07 development100.

This module does not launch VMs.  Packaging copies only frozen source and
runtime artifacts and never opens a development root.  Workers must acquire a
create-only GCS root claim before invoking the one-root development runner.
Receive validation accepts content only after all 100 immutable DONE records
exist; arm selection remains a separate explicit command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .aggregate_hu_m43_attempt07_preflight import (
    ATTEMPT07_PREFLIGHT_AGGREGATE_SCHEMA,
    validate_preflight_go_aggregate,
)
from .hu_m43_attempt06_spot import (
    PINNED_MODEL_MANIFEST_SHA256,
    PINNED_NATIVE_MANIFEST_SHA256,
    PINNED_TEMPLATE_RUN,
    REQUIREMENTS,
    _copy_file,
    _copy_manifest_files,
    _copy_source_tree,
    _verify_pinned_runtime_closure,
    _write_deterministic_zip,
)
from .hu_m43_attempt06_teacher import ATTEMPT06_FROZEN_MODEL_SHA256
from .hu_m43_attempt07_contract import (
    AI_PROFILES_SHA256,
    M43_ATTEMPT07_PLAN_SHA256,
    M43_ATTEMPT07_PROFILES,
    enumerate_attempt07_seed_schedules,
    load_and_validate_attempt07_plan,
    load_and_validate_attempt07_status,
)
from .run_hu_m43_attempt07_development import (
    ATTEMPT07_CHECKPOINT_SCHEMA,
    ATTEMPT07_HEARTBEAT_SCHEMA,
    ATTEMPT07_SHARD_ROW_SCHEMA,
    ATTEMPT07_SHARD_SUMMARY_SCHEMA,
)
from .run_hu_m43_attempt07_preflight import (
    ATTEMPT07_PREFLIGHT_SOURCE_SHA256,
    load_preflight_plan,
)
from .select_hu_m43_attempt07_development_arm import _validate_row


PACKAGE_MANIFEST_SCHEMA = "hu_m43_attempt07_development_spot_package_v1"
SOURCE_CLOSURE_SCHEMA = "hu_m43_attempt07_development_source_closure_v1"
SHARD_SCHEMA = "hu_m43_attempt07_development_spot_shard_v1"
PACKAGE_RESULT_SCHEMA = "hu_m43_attempt07_development_package_result_v1"
DONE_SCHEMA = "hu_m43_attempt07_development_spot_done_v1"
GLOBAL_CLAIM_SCHEMA = "hu_m43_attempt07_development_global_claim_v1"
ROOT_CLAIM_SCHEMA = "hu_m43_attempt07_development_root_claim_v1"
RECEIVE_AUDIT_SCHEMA = "hu_m43_attempt07_development_received_shard_v1"
RECEIVE_MERGE_SCHEMA = "hu_m43_attempt07_development_receive_merge_v1"
AUTHORIZATION_SCHEMA = "hu_m43_attempt07_development_spot_authorization_v1"
OUTPUT_CONSUMPTION_SCHEMA = "hu_m43_attempt07_development_output_consumption_v1"

EXPECTED_SHARDS = 100
ROOTS_PER_SHARD = 1
NATIVE_BATCH_THREADS = 4
MAX_WAVE_SHARDS = 50
ROOT_REMATERIALIZATION_MODE = (
    "same_root_index_same_six_seeds_same_frozen_closure_after_matching_claim"
)
RUN_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,126}[a-z0-9]$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_AUTHORIZATION_KEYS = frozenset(
    {
        "schema",
        "status",
        "spot_authorized",
        "attempt07_plan_sha256",
        "preflight_plan_sha256",
        "preflight_aggregate_sha256",
        "preflight_manifest_sha256",
        "preflight_schedule_sha256",
        "preflight_launch_authorization_sha256",
        "done_sha256",
        "development_run_name",
        "development_manifest_sha256",
        "development_schedule_sha256",
        "development_source_closure_sha256",
        "development_source_zip_sha256",
        "development_startup_sha256",
        "development_status_sha256",
        "development_source_model_manifest_sha256",
        "development_source_native_manifest_sha256",
        "development_total_roots",
        "development_total_shards",
        "development_roots_per_shard",
        "development_native_batch_threads",
        "development_max_wave_shards",
        "development_machine_type",
        "development_root_profile_assignment",
        "development_batch_child_selectors",
        "development_package_frozen_before_authorization",
        "operational_metrics",
        "operational_gates",
        "all_gates_passed",
        "development_started",
        "fresh_development_root_opened",
        "current_profile_mutated",
        "runtime_policy_activated",
    }
)
_PREFLIGHT_SLOTS = frozenset(
    {
        "root0_batch_a",
        "root0_batch_b",
        "root0_scalar",
        "root1_batch",
        "root2_batch",
    }
)
_OPERATIONAL_GATE_NAMES = frozenset(
    {
        "proof_aggregate_go",
        "all_five_done_metadata",
        "batch_elapsed_seconds_per_root",
        "scalar_elapsed_seconds_root0",
        "scalar_to_root0_batch_median_speedup",
        "root0_batch_replicate_elapsed_ratio",
        "peak_rss_bytes_per_job",
    }
)
_OPERATIONAL_METRIC_KEYS = frozenset(
    {
        "elapsed_seconds",
        "peak_rss_bytes",
        "root0_batch_median_elapsed_seconds",
        "root0_batch_replicate_elapsed_ratio",
        "scalar_to_root0_batch_median_speedup",
        "batch_elapsed_seconds_max",
        "peak_rss_bytes_max",
    }
)
_PACKAGE_MANIFEST_KEYS = frozenset(
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
_OUTPUT_CONSUMPTION_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "manifest_sha256",
        "schedule_sha256",
        "authorization_sha256",
        "done_sha256",
        "expected_shards",
        "expected_roots",
        "all_done_markers_verified",
        "result_objects_addressed_when_claimed",
        "selector_executed",
        "fit_performed",
        "threshold_selected",
        "current_profile_mutated",
        "runtime_policy_activated",
    }
)
_DONE_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "run_id",
        "shard",
        "root_index",
        "root_profile",
        "seeds",
        "output_prefix",
        "manifest_sha256",
        "source_sha256",
        "startup_sha256",
        "schedule_sha256",
        "plan_sha256",
        "status_sha256",
        "model_sha256",
        "ai_profiles_sha256",
        "preflight_plan_sha256",
        "preflight_aggregate_sha256",
        "source_closure_sha256",
        "source_model_manifest_sha256",
        "source_native_manifest_sha256",
        "authorization_sha256",
        "global_claim_sha256",
        "root_claim_sha256",
        "output_sha256",
        "checkpoint_sha256",
        "heartbeat_sha256",
        "generator_summary_sha256",
        "run_log_sha256",
        "config_sha256",
        "elapsed_seconds",
        "peak_rss_bytes",
        "native_batch_threads",
        "teacher_values_are_realized_match_ev",
        "current_profile_mutated",
        "runtime_policy_activated",
    }
)
_CLAIM_CLOSURE_KEYS = frozenset(
    {
        "run_name",
        "manifest_sha256",
        "source_sha256",
        "startup_sha256",
        "schedule_sha256",
        "plan_sha256",
        "status_sha256",
        "model_sha256",
        "ai_profiles_sha256",
        "preflight_plan_sha256",
        "preflight_aggregate_sha256",
        "source_closure_sha256",
        "source_model_manifest_sha256",
        "source_native_manifest_sha256",
        "authorization_sha256",
        "native_batch_threads",
    }
)
_GLOBAL_CLAIM_KEYS = _CLAIM_CLOSURE_KEYS | frozenset(
    {
        "schema",
        "status",
        "roots",
        "shards",
        "roots_per_shard",
        "development_started_when_claimed",
        "current_profile_mutated",
        "runtime_policy_activated",
    }
)
_ROOT_CLAIM_KEYS = _CLAIM_CLOSURE_KEYS | frozenset(
    {
        "schema",
        "status",
        "run_id",
        "root_index",
        "root_profile",
        "seeds",
        "output_prefix",
        "alternate_seed_or_result_retry_allowed",
        "deterministic_recompute_allowed",
        "deterministic_recompute_mode",
    }
)
_PREFLIGHT_DONE_KEYS = frozenset(
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

_AUTHORIZATION_LAYOUT_SENTINELS: dict[str, tuple[str, str]] = {
    "package_src": ("package_src", "dir"),
    "configs": ("configs", "dir"),
    "frozen": ("frozen", "dir"),
    "preflight_closure": ("preflight_closure", "dir"),
    "top_plan": ("hu_joint_policy_m43_attempt07.json", "file"),
    "top_status": ("hu_joint_policy_m43_attempt07_status.json", "file"),
    "top_preflight_plan": (
        "hu_joint_policy_m43_attempt07_preflight.json",
        "file",
    ),
    "top_preflight_aggregate": ("attempt07_preflight_aggregate.json", "file"),
    "top_startup": ("startup_hu_m43_attempt07_development.sh", "file"),
    "top_source_zip": (
        "ofc_regular_hu_m43_attempt07_development_source.zip",
        "file",
    ),
    "top_model_manifest": ("source_model_manifest.json", "file"),
    "top_native_manifest": ("source_native_manifest.json", "file"),
}
_AUTHORIZATION_LAYOUT_SIGNATURES: dict[str, frozenset[str]] = {
    "full_package_run": frozenset(
        {
            "package_src",
            "preflight_closure",
            "top_plan",
            "top_status",
            "top_preflight_plan",
            "top_preflight_aggregate",
            "top_startup",
            "top_source_zip",
        }
    ),
    "extracted_package_worker": frozenset(
        {"configs", "frozen", "top_model_manifest", "top_native_manifest"}
    ),
    "received_closure": frozenset(
        {
            "preflight_closure",
            "top_plan",
            "top_status",
            "top_preflight_plan",
            "top_preflight_aggregate",
            "top_startup",
            "top_model_manifest",
            "top_native_manifest",
        }
    ),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable file: {path}")
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"immutable file concurrently created: {path}"
            ) from exc
        os.unlink(temporary)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(path, canonical_json_bytes(payload))


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a mapping")
    return payload


def _load_canonical_mapping(path: Path, label: str) -> dict[str, Any]:
    raw = path.read_bytes()
    payload = _load_mapping(path, label)
    if raw != canonical_json_bytes(payload):
        raise ValueError(f"{label} is not canonical JSON")
    return payload


def _require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"{label} SHA-256 changed")


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_nonnegative_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _closure_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def _resolve_attempt07_authorization_layout(root: Path) -> dict[str, Any]:
    """Resolve one exact, known physical layout for authorization validation."""

    if not root.is_dir():
        raise ValueError("Attempt07 authorization layout root is missing")
    for name in ("manifest.json", "shards_manifest.jsonl", "source_closure_manifest.json"):
        if not (root / name).is_file():
            raise ValueError(f"Attempt07 authorization layout is missing {name}")

    observed: set[str] = set()
    for name, (relative, kind) in _AUTHORIZATION_LAYOUT_SENTINELS.items():
        path = root / relative
        if not path.exists():
            continue
        if (kind == "file" and not path.is_file()) or (
            kind == "dir" and not path.is_dir()
        ):
            raise ValueError(f"Attempt07 authorization layout sentinel changed: {name}")
        observed.add(name)
    matches = [
        name
        for name, signature in _AUTHORIZATION_LAYOUT_SIGNATURES.items()
        if observed == set(signature)
    ]
    if len(matches) != 1:
        raise ValueError(
            "Attempt07 authorization layout is unknown or ambiguous: "
            + ",".join(sorted(observed))
        )
    name = matches[0]

    if name == "full_package_run":
        paths: dict[str, Any] = {
            "plan": root / "hu_joint_policy_m43_attempt07.json",
            "status": root / "hu_joint_policy_m43_attempt07_status.json",
            "preflight_plan": root / "hu_joint_policy_m43_attempt07_preflight.json",
            "preflight_aggregate": root / "attempt07_preflight_aggregate.json",
            "preflight_closure": root / "preflight_closure",
            "source_model_manifest": root / "package_src/source_model_manifest.json",
            "source_native_manifest": root / "package_src/source_native_manifest.json",
            "startup": root / "startup_hu_m43_attempt07_development.sh",
            "source_zip": root
            / "ofc_regular_hu_m43_attempt07_development_source.zip",
            "worker_done_layout": False,
        }
    elif name == "extracted_package_worker":
        paths = {
            "plan": root / "configs/hu_joint_policy_m43_attempt07.json",
            "status": root / "configs/hu_joint_policy_m43_attempt07_status.json",
            "preflight_plan": root
            / "configs/hu_joint_policy_m43_attempt07_preflight.json",
            "preflight_aggregate": root
            / "frozen/attempt07_preflight_aggregate.json",
            "preflight_closure": root / "frozen/preflight",
            "source_model_manifest": root / "source_model_manifest.json",
            "source_native_manifest": root / "source_native_manifest.json",
            "startup": None,
            "source_zip": None,
            "worker_done_layout": True,
        }
    else:
        paths = {
            "plan": root / "hu_joint_policy_m43_attempt07.json",
            "status": root / "hu_joint_policy_m43_attempt07_status.json",
            "preflight_plan": root / "hu_joint_policy_m43_attempt07_preflight.json",
            "preflight_aggregate": root / "attempt07_preflight_aggregate.json",
            "preflight_closure": root / "preflight_closure",
            "source_model_manifest": root / "source_model_manifest.json",
            "source_native_manifest": root / "source_native_manifest.json",
            "startup": root / "startup_hu_m43_attempt07_development.sh",
            "source_zip": None,
            "worker_done_layout": False,
        }
    required_files = (
        "plan",
        "status",
        "preflight_plan",
        "preflight_aggregate",
        "source_model_manifest",
        "source_native_manifest",
    )
    for field in required_files:
        if not paths[field].is_file():
            raise ValueError(
                f"Attempt07 {name} authorization layout is missing {field}"
            )
    if not paths["preflight_closure"].is_dir():
        raise ValueError(f"Attempt07 {name} preflight closure is missing")
    for field in ("startup", "source_zip"):
        path = paths[field]
        if path is not None and not path.is_file():
            raise ValueError(f"Attempt07 {name} authorization layout is missing {field}")
    return {"name": name, **paths}


def build_attempt07_spot_schedule(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the exact balanced one-root development schedule."""

    population = plan.get("development_population")
    if not isinstance(population, Mapping) or population != {
        "classification": "new_balanced_development_only",
        "roots": 100,
        "root_index_first": 0,
        "root_index_last": 99,
        "profiles": 5,
        "roots_per_profile": 20,
        "profile_assignment": "root_index_mod_5_in_frozen_profile_order",
        "candidate_arm_count": 4,
        "winner_count_max": 1,
        "fresh_generalization_claim_allowed": False,
        "fit_allowed": False,
        "threshold_selection_allowed": False,
        "runtime_activation_allowed": False,
    }:
        raise ValueError("Attempt07 development population changed")
    schedules = enumerate_attempt07_seed_schedules(plan, population="development")
    rows: list[dict[str, Any]] = []
    for root_index in range(EXPECTED_SHARDS):
        seeds = {domain: values[root_index] for domain, values in schedules.items()}
        rows.append(
            {
                "schema": SHARD_SCHEMA,
                "shard": root_index,
                "root_index": root_index,
                "roots": 1,
                "root_profile": M43_ATTEMPT07_PROFILES[
                    root_index % len(M43_ATTEMPT07_PROFILES)
                ],
                "seeds": seeds,
                "baseline_profile": "stage18_p1",
                "continuation_profile": "stage9f_p2",
                "batch_child_selectors": True,
                "native_batch_threads": NATIVE_BATCH_THREADS,
                "output_prefix": f"shard_{root_index:03d}",
            }
        )
    if any(len(set(row["seeds"].values())) != 6 for row in rows):
        raise AssertionError("Attempt07 schedule contains an intra-root seed collision")
    for domain in ("hand", "screen", "rerank", "veto", "assessment", "child"):
        if len({row["seeds"][domain] for row in rows}) != EXPECTED_SHARDS:
            raise AssertionError(f"Attempt07 schedule duplicated {domain} seeds")
    counts = {profile: 0 for profile in M43_ATTEMPT07_PROFILES}
    for row in rows:
        counts[str(row["root_profile"])] += 1
    if set(counts.values()) != {20}:
        raise AssertionError("Attempt07 schedule lost balanced profiles")
    return rows


def _schedule_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def _validate_preflight_go(
    path: Path, *, preflight_plan_sha256: str
) -> dict[str, Any]:
    payload = _load_canonical_mapping(path, "Attempt07 preflight aggregate")
    validate_preflight_go_aggregate(
        payload,
        preflight_plan_sha256=preflight_plan_sha256,
        require_operational_jobs=True,
    )
    gates = payload.get("proof_gates")
    science = payload.get("science_boundary")
    contract = payload.get("contract")
    expected_slots = sorted(_PREFLIGHT_SLOTS)
    if (
        payload.get("schema") != ATTEMPT07_PREFLIGHT_AGGREGATE_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("decision") != "go"
        or payload.get("reasons") != ["all_preflight_proof_gates_passed"]
        or payload.get("proof_input_count") != 5
        or payload.get("valid_proof_count") != 5
        or sorted(payload.get("expected_slots", [])) != expected_slots
        or payload.get("root_coverage") != [0, 1, 2]
        or not isinstance(gates, Mapping)
        or not gates
        or any(value is not True for value in gates.values())
        or not isinstance(science, Mapping)
        or science.get("proof_only_no_teacher_values_or_arm_details") is not True
        or science.get("arm_selection_allowed") is not False
        or science.get("fit_allowed") is not False
        or science.get("threshold_selection_allowed") is not False
        or science.get("current_profile_mutated") is not False
        or science.get("runtime_activation_allowed") is not False
        or science.get("current_profile_resolved") is not False
        or science.get("fresh_seed_or_root_opened") is not False
        or science.get("done_metadata_is_science_input") is not False
        or not isinstance(contract, Mapping)
        or contract.get("attempt07_plan_sha256") != M43_ATTEMPT07_PLAN_SHA256
        or contract.get("model_sha256") != ATTEMPT06_FROZEN_MODEL_SHA256
        or contract.get("ai_profiles_sha256") != AI_PROFILES_SHA256
    ):
        raise ValueError("Attempt07 preflight aggregate is not an immutable Go")
    _require_sha256(
        contract.get("preflight_plan_sha256"),
        "Attempt07 aggregate preflight_plan_sha256",
    )
    diagnostics = payload.get("operational_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError("Attempt07 preflight aggregate lacks operational diagnostics")
    return payload


def validate_attempt07_spot_authorization(
    *,
    authorization_path: str | Path,
    manifest_path: str | Path,
    preflight_aggregate_path: str | Path,
) -> dict[str, Any]:
    """Validate the exact result-before-development launch authorization."""

    authorization_file = Path(authorization_path)
    manifest_file = Path(manifest_path)
    preflight_file = Path(preflight_aggregate_path)
    authorization = _load_canonical_mapping(
        authorization_file, "Attempt07 Spot authorization"
    )
    manifest = _load_canonical_mapping(manifest_file, "Attempt07 package manifest")
    if (
        set(manifest) != _PACKAGE_MANIFEST_KEYS
        or manifest.get("schema") != PACKAGE_MANIFEST_SCHEMA
        or manifest.get("status") != "frozen_package_only_no_root_opened"
        or not isinstance(manifest.get("run_name"), str)
        or RUN_NAME_RE.fullmatch(str(manifest.get("run_name"))) is None
        or manifest.get("plan_sha256") != M43_ATTEMPT07_PLAN_SHA256
        or manifest.get("model_sha256") != ATTEMPT06_FROZEN_MODEL_SHA256
        or manifest.get("ai_profiles_sha256") != AI_PROFILES_SHA256
        or manifest.get("total_roots") != EXPECTED_SHARDS
        or manifest.get("total_shards") != EXPECTED_SHARDS
        or manifest.get("roots_per_shard") != ROOTS_PER_SHARD
        or manifest.get("root_profile_assignment")
        != "root_index_mod_5_in_frozen_profile_order"
        or manifest.get("batch_child_selectors") is not True
        or manifest.get("native_batch_threads") != NATIVE_BATCH_THREADS
        or manifest.get("recommended_machine_type") != "c4-standard-4"
        or manifest.get("recommended_wave_shards") != MAX_WAVE_SHARDS
        or manifest.get("fresh_root_opened") is not False
        or manifest.get("teacher_executed") is not False
        or manifest.get("gcloud_invoked") is not False
        or manifest.get("instances_created") is not False
        or manifest.get("current_profile_mutated") is not False
        or manifest.get("runtime_policy_activated") is not False
    ):
        raise ValueError("Attempt07 development package manifest changed")
    for field in _PACKAGE_MANIFEST_KEYS & {
        key for key in _PACKAGE_MANIFEST_KEYS if key.endswith("_sha256")
    }:
        if field != "preflight_done_sha256":
            _require_sha256(manifest.get(field), f"Attempt07 manifest {field}")
    package_root = manifest_file.parent
    if manifest_file.name != "manifest.json":
        raise ValueError("Attempt07 authorization manifest path is not canonical")
    layout = _resolve_attempt07_authorization_layout(package_root)
    expected_preflight_file = Path(layout["preflight_aggregate"])
    if preflight_file.resolve() != expected_preflight_file.resolve():
        raise ValueError("Attempt07 authorization preflight aggregate layout changed")
    preflight = _validate_preflight_go(
        preflight_file,
        preflight_plan_sha256=str(manifest.get("preflight_plan_sha256")),
    )
    preflight_plan_file = Path(layout["preflight_plan"])
    # Hash agreement is not enough: a self-consistent package must still use
    # the producer-owned, byte-for-byte frozen thresholds and proof contract.
    preflight_plan = load_preflight_plan(preflight_plan_file)
    limits = preflight_plan.get("operational_go_no_go")
    if not isinstance(limits, Mapping):
        raise ValueError("Attempt07 frozen operational thresholds are missing")
    physical_development_files = {
        "schedule_sha256": package_root / "shards_manifest.jsonl",
        "source_closure_sha256": package_root / "source_closure_manifest.json",
        "plan_sha256": Path(layout["plan"]),
        "status_sha256": Path(layout["status"]),
        "preflight_plan_sha256": preflight_plan_file,
        "source_model_manifest_sha256": Path(layout["source_model_manifest"]),
        "source_native_manifest_sha256": Path(layout["source_native_manifest"]),
    }
    if layout["source_zip"] is not None:
        physical_development_files["source_zip_sha256"] = Path(layout["source_zip"])
    if layout["startup"] is not None:
        physical_development_files["startup_sha256"] = Path(layout["startup"])
    for field, path in physical_development_files.items():
        if sha256_file(path) != manifest.get(field):
            raise ValueError(f"Attempt07 development physical {field} changed")
    if set(authorization) != _AUTHORIZATION_KEYS:
        raise ValueError("Attempt07 Spot authorization top-level fields changed")
    done_sha256 = authorization.get("done_sha256")
    operational_metrics = authorization.get("operational_metrics")
    operational_gates = authorization.get("operational_gates")
    if (
        authorization.get("schema") != AUTHORIZATION_SCHEMA
        or authorization.get("status") != "authorized_after_attempt07_preflight"
        or authorization.get("spot_authorized") is not True
        or authorization.get("attempt07_plan_sha256")
        != manifest.get("plan_sha256")
        or authorization.get("attempt07_plan_sha256")
        != M43_ATTEMPT07_PLAN_SHA256
        or authorization.get("preflight_plan_sha256")
        != manifest.get("preflight_plan_sha256")
        or authorization.get("preflight_plan_sha256")
        != sha256_file(preflight_plan_file)
        or authorization.get("preflight_aggregate_sha256")
        != sha256_file(preflight_file)
        or authorization.get("preflight_aggregate_sha256")
        != manifest.get("preflight_aggregate_sha256")
        or authorization.get("development_run_name") != manifest.get("run_name")
        or authorization.get("development_manifest_sha256")
        != sha256_file(manifest_file)
        or authorization.get("development_schedule_sha256")
        != manifest.get("schedule_sha256")
        or authorization.get("development_source_closure_sha256")
        != manifest.get("source_closure_sha256")
        or authorization.get("development_source_zip_sha256")
        != manifest.get("source_zip_sha256")
        or authorization.get("development_startup_sha256")
        != manifest.get("startup_sha256")
        or authorization.get("development_status_sha256")
        != manifest.get("status_sha256")
        or authorization.get("development_source_model_manifest_sha256")
        != manifest.get("source_model_manifest_sha256")
        or authorization.get("development_source_native_manifest_sha256")
        != manifest.get("source_native_manifest_sha256")
        or type(authorization.get("development_total_roots")) is not int
        or authorization.get("development_total_roots") != EXPECTED_SHARDS
        or type(authorization.get("development_total_shards")) is not int
        or authorization.get("development_total_shards") != EXPECTED_SHARDS
        or type(authorization.get("development_roots_per_shard")) is not int
        or authorization.get("development_roots_per_shard") != ROOTS_PER_SHARD
        or type(authorization.get("development_native_batch_threads")) is not int
        or authorization.get("development_native_batch_threads")
        != NATIVE_BATCH_THREADS
        or type(authorization.get("development_max_wave_shards")) is not int
        or authorization.get("development_max_wave_shards") != MAX_WAVE_SHARDS
        or authorization.get("development_machine_type") != "c4-standard-4"
        or authorization.get("development_root_profile_assignment")
        != "root_index_mod_5_in_frozen_profile_order"
        or authorization.get("development_batch_child_selectors") is not True
        or authorization.get("development_package_frozen_before_authorization")
        is not True
        or not isinstance(done_sha256, Mapping)
        or set(done_sha256) != _PREFLIGHT_SLOTS
        or not isinstance(operational_metrics, Mapping)
        or not isinstance(operational_gates, Mapping)
        or set(operational_metrics) != _OPERATIONAL_METRIC_KEYS
        or set(operational_gates) != _OPERATIONAL_GATE_NAMES
        or any(
            not isinstance(value, Mapping)
            or set(value) != {"name", "passed", "observed", "requirement"}
            or value.get("name") != name
            or value.get("passed") is not True
            for name, value in operational_gates.items()
        )
        or authorization.get("all_gates_passed") is not True
        or authorization.get("development_started") is not False
        or authorization.get("fresh_development_root_opened") is not False
        or authorization.get("current_profile_mutated") is not False
        or authorization.get("runtime_policy_activated") is not False
    ):
        raise ValueError("Attempt07 Spot authorization identity or gates changed")
    for label, value in done_sha256.items():
        _require_sha256(value, f"Attempt07 authorization DONE {label}")
    for field in (
        "preflight_plan_sha256",
        "preflight_manifest_sha256",
        "preflight_schedule_sha256",
        "preflight_launch_authorization_sha256",
    ):
        _require_sha256(authorization.get(field), f"Attempt07 authorization {field}")

    preflight_closure_root = Path(layout["preflight_closure"])
    packaged_source_layout = bool(layout["worker_done_layout"])
    frozen_preflight_files = {
        "preflight_manifest_sha256": preflight_closure_root / "manifest.json",
        "preflight_schedule_sha256": preflight_closure_root
        / "shards_manifest.jsonl",
        "preflight_launch_authorization_sha256": preflight_closure_root
        / "launch_authorization.json",
    }
    for field, path in frozen_preflight_files.items():
        if (
            sha256_file(path) != manifest.get(field)
            or authorization.get(field) != manifest.get(field)
        ):
            raise ValueError(f"Attempt07 authorization {field} closure changed")
    manifest_done = manifest.get("preflight_done_sha256")
    if not isinstance(manifest_done, Mapping) or set(manifest_done) != _PREFLIGHT_SLOTS:
        raise ValueError("Attempt07 manifest preflight DONE closure changed")
    from .hu_m43_attempt07_preflight_spot import (
        DONE_SCHEMA as PREFLIGHT_DONE_SCHEMA,
        _validate_authorization as validate_preflight_launch_authorization,
        _validate_package_manifest as validate_preflight_package_manifest,
        build_preflight_schedule,
    )

    preflight_manifest_path = frozen_preflight_files["preflight_manifest_sha256"]
    preflight_schedule_path = frozen_preflight_files["preflight_schedule_sha256"]
    preflight_launch_path = frozen_preflight_files[
        "preflight_launch_authorization_sha256"
    ]
    preflight_manifest_payload = _load_canonical_mapping(
        preflight_manifest_path, "packaged Attempt07 preflight manifest"
    )
    validate_preflight_package_manifest(
        preflight_manifest_payload, preflight_manifest_path
    )
    preflight_manifest_sha256 = sha256_file(preflight_manifest_path)
    _load_canonical_mapping(
        preflight_launch_path, "packaged Attempt07 preflight launch authorization"
    )
    preflight_launch_sha256 = validate_preflight_launch_authorization(
        preflight_launch_path,
        manifest=preflight_manifest_payload,
        manifest_sha256=preflight_manifest_sha256,
    )
    preflight_schedule_raw = preflight_schedule_path.read_bytes()
    try:
        preflight_schedule = [
            json.loads(line)
            for line in preflight_schedule_raw.decode("utf-8").splitlines()
            if line.strip()
        ]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("packaged Attempt07 preflight schedule is invalid") from exc
    expected_preflight_schedule = list(build_preflight_schedule())
    if (
        preflight_schedule != expected_preflight_schedule
        or preflight_schedule_raw
        != b"".join(canonical_json_bytes(row) for row in preflight_schedule)
    ):
        raise ValueError("packaged Attempt07 preflight schedule changed")
    preflight_schedule_sha256 = sha256_file(preflight_schedule_path)
    proof_hashes = preflight.get("proof_file_sha256")
    if not isinstance(proof_hashes, Mapping) or set(proof_hashes) != _PREFLIGHT_SLOTS:
        raise ValueError("Attempt07 preflight proof hashes changed")
    raw_elapsed: dict[str, float] = {}
    raw_rss: dict[str, int] = {}
    for spec in expected_preflight_schedule:
        label = str(spec["job_id"])
        path = (
            preflight_closure_root / "done" / f"{label}.json"
            if packaged_source_layout
            else preflight_closure_root / f"{label}.DONE.json"
        )
        done = _load_canonical_mapping(path, f"packaged preflight DONE {label}")
        fixed = {
            "schema": PREFLIGHT_DONE_SCHEMA,
            "status": "complete",
            "run_name": preflight_manifest_payload["run_name"],
            "job_index": spec["job_index"],
            "job_id": label,
            "source_root_index": spec["source_root_index"],
            "batch_child_selectors": spec["batch_child_selectors"],
            "native_batch_threads": NATIVE_BATCH_THREADS,
            "output_prefix": spec["output_prefix"],
            "manifest_sha256": preflight_manifest_sha256,
            "authorization_sha256": preflight_launch_sha256,
            "schedule_sha256": preflight_schedule_sha256,
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
        if set(done) != _PREFLIGHT_DONE_KEYS or any(
            done.get(key) != value or type(done.get(key)) is not type(value)
            for key, value in fixed.items()
        ):
            raise ValueError(f"Attempt07 packaged preflight DONE changed: {label}")
        for field in (
            "output_sha256",
            "checkpoint_sha256",
            "heartbeat_sha256",
            "summary_sha256",
            "run_log_sha256",
        ):
            _require_sha256(done.get(field), f"preflight DONE {label}.{field}")
        elapsed_value = done.get("elapsed_seconds")
        rss_value = done.get("peak_rss_bytes")
        elapsed = _require_nonnegative_finite(
            elapsed_value, f"preflight DONE {label}.elapsed_seconds"
        )
        if elapsed <= 0.0 or type(rss_value) is not int or rss_value <= 0:
            raise ValueError(f"Attempt07 preflight DONE metrics changed: {label}")
        actual = sha256_file(path)
        if (
            actual != manifest_done[label]
            or actual != done_sha256[label]
            or done["output_sha256"] != proof_hashes[label]
        ):
            raise ValueError(f"Attempt07 authorization DONE closure changed: {label}")
        raw_elapsed[label] = elapsed
        raw_rss[label] = rss_value

    # The finalizer owns the operational formulas.  The launcher independently
    # requires the five exact metric jobs to match the already-frozen aggregate,
    # so a boolean-only authorization cannot bypass the measured preflight.
    diagnostics = preflight.get("operational_diagnostics")
    jobs = diagnostics.get("jobs") if isinstance(diagnostics, Mapping) else None
    if (
        diagnostics.get("status") != "ok"
        or diagnostics.get("science_decision_input") is not False
        or diagnostics.get("job_count") != 5
        or diagnostics.get("invalid_labels") != []
        or not isinstance(jobs, Mapping)
        or set(jobs) != _PREFLIGHT_SLOTS
    ):
        raise ValueError("Attempt07 authorization lacks all five operational jobs")
    elapsed_metrics = operational_metrics.get("elapsed_seconds")
    rss_metrics = operational_metrics.get("peak_rss_bytes")
    if (
        not isinstance(elapsed_metrics, Mapping)
        or set(elapsed_metrics) != _PREFLIGHT_SLOTS
        or not isinstance(rss_metrics, Mapping)
        or set(rss_metrics) != _PREFLIGHT_SLOTS
    ):
        raise ValueError("Attempt07 authorization operational maps changed")
    for label in sorted(_PREFLIGHT_SLOTS):
        expected = jobs[label]
        if not isinstance(expected, Mapping):
            raise ValueError("Attempt07 operational metric row changed")
        expected_elapsed = _require_nonnegative_finite(
            expected.get("elapsed_seconds"), f"aggregate {label}.elapsed_seconds"
        )
        actual_elapsed = _require_nonnegative_finite(
            elapsed_metrics[label], f"authorization {label}.elapsed_seconds"
        )
        expected_rss = _require_nonnegative_finite(
            expected.get("peak_rss_bytes"), f"aggregate {label}.peak_rss_bytes"
        )
        actual_rss = _require_nonnegative_finite(
            rss_metrics[label], f"authorization {label}.peak_rss_bytes"
        )
        if (
            actual_elapsed != expected_elapsed
            or actual_rss != expected_rss
            or actual_elapsed != raw_elapsed[label]
            or int(actual_rss) != raw_rss[label]
        ):
            raise ValueError("Attempt07 authorization operational metrics changed")
        if actual_elapsed <= 0.0 or actual_rss <= 0.0 or not actual_rss.is_integer():
            raise ValueError("Attempt07 authorization operational metrics are invalid")
    for field in _OPERATIONAL_METRIC_KEYS - {
        "elapsed_seconds",
        "peak_rss_bytes",
    }:
        _require_nonnegative_finite(
            operational_metrics[field], f"authorization operational {field}"
        )
    elapsed = raw_elapsed
    rss = raw_rss
    root0_batch = [elapsed["root0_batch_a"], elapsed["root0_batch_b"]]
    batch_median = sum(root0_batch) / 2.0
    replicate_ratio = max(root0_batch) / min(root0_batch)
    speedup = elapsed["root0_scalar"] / batch_median
    batch_max = max(
        elapsed[label]
        for label in (
            "root0_batch_a",
            "root0_batch_b",
            "root1_batch",
            "root2_batch",
        )
    )
    rss_max = max(rss.values())
    recomputed_metrics = {
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": rss,
        "root0_batch_median_elapsed_seconds": batch_median,
        "root0_batch_replicate_elapsed_ratio": replicate_ratio,
        "scalar_to_root0_batch_median_speedup": speedup,
        "batch_elapsed_seconds_max": batch_max,
        "peak_rss_bytes_max": rss_max,
    }
    if operational_metrics != recomputed_metrics:
        raise ValueError("Attempt07 authorization operational formulas changed")

    def expected_gate(
        name: str, passed: bool, observed: Any, requirement: str
    ) -> dict[str, Any]:
        return {
            "name": name,
            "passed": passed,
            "observed": observed,
            "requirement": requirement,
        }

    expected_gates = {
        "proof_aggregate_go": expected_gate(
            "proof_aggregate_go", True, "go", "go"
        ),
        "all_five_done_metadata": expected_gate(
            "all_five_done_metadata", True, 5, "= 5"
        ),
        "batch_elapsed_seconds_per_root": expected_gate(
            "batch_elapsed_seconds_per_root",
            batch_max <= float(limits["batch_elapsed_seconds_per_root_max"]),
            batch_max,
            f"<= {limits['batch_elapsed_seconds_per_root_max']}",
        ),
        "scalar_elapsed_seconds_root0": expected_gate(
            "scalar_elapsed_seconds_root0",
            elapsed["root0_scalar"]
            <= float(limits["scalar_elapsed_seconds_root0_max"]),
            elapsed["root0_scalar"],
            f"<= {limits['scalar_elapsed_seconds_root0_max']}",
        ),
        "scalar_to_root0_batch_median_speedup": expected_gate(
            "scalar_to_root0_batch_median_speedup",
            speedup >= float(limits["scalar_to_root0_batch_median_speedup_min"]),
            speedup,
            f">= {limits['scalar_to_root0_batch_median_speedup_min']}",
        ),
        "root0_batch_replicate_elapsed_ratio": expected_gate(
            "root0_batch_replicate_elapsed_ratio",
            replicate_ratio
            <= float(limits["root0_batch_replicate_elapsed_ratio_max"]),
            replicate_ratio,
            f"<= {limits['root0_batch_replicate_elapsed_ratio_max']}",
        ),
        "peak_rss_bytes_per_job": expected_gate(
            "peak_rss_bytes_per_job",
            rss_max <= int(limits["peak_rss_bytes_per_job_max"]),
            rss_max,
            f"<= {limits['peak_rss_bytes_per_job_max']}",
        ),
    }
    if operational_gates != expected_gates or not all(
        gate["passed"] for gate in expected_gates.values()
    ):
        raise ValueError("Attempt07 authorization fixed operational gates changed")
    return {
        "schema": AUTHORIZATION_SCHEMA,
        "status": "pass",
        "authorization_sha256": sha256_file(authorization_file),
        "preflight_aggregate_sha256": sha256_file(preflight_file),
        "all_gates_passed": True,
        "development_started": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }


def _validate_existing_package(run_dir: Path, run_name: str) -> dict[str, Any]:
    manifest = _load_canonical_mapping(
        run_dir / "manifest.json", "Attempt07 manifest"
    )
    if (
        set(manifest) != _PACKAGE_MANIFEST_KEYS
        or manifest.get("schema") != PACKAGE_MANIFEST_SCHEMA
        or manifest.get("status") != "frozen_package_only_no_root_opened"
        or manifest.get("run_name") != run_name
        or manifest.get("plan_sha256") != M43_ATTEMPT07_PLAN_SHA256
        or manifest.get("model_sha256") != ATTEMPT06_FROZEN_MODEL_SHA256
        or manifest.get("ai_profiles_sha256") != AI_PROFILES_SHA256
        or manifest.get("total_shards") != EXPECTED_SHARDS
        or manifest.get("total_roots") != EXPECTED_SHARDS
        or manifest.get("roots_per_shard") != ROOTS_PER_SHARD
        or manifest.get("native_batch_threads") != NATIVE_BATCH_THREADS
        or manifest.get("fresh_root_opened") is not False
        or manifest.get("teacher_executed") is not False
        or manifest.get("gcloud_invoked") is not False
        or manifest.get("instances_created") is not False
        or manifest.get("current_profile_mutated") is not False
        or manifest.get("runtime_policy_activated") is not False
    ):
        raise ValueError("existing Attempt07 package changed")
    bindings = {
        "shards_manifest.jsonl": "schedule_sha256",
        "source_closure_manifest.json": "source_closure_sha256",
        "ofc_regular_hu_m43_attempt07_development_source.zip": "source_zip_sha256",
        "startup_hu_m43_attempt07_development.sh": "startup_sha256",
        "hu_joint_policy_m43_attempt07.json": "plan_sha256",
        "hu_joint_policy_m43_attempt07_status.json": "status_sha256",
        "attempt07_preflight_aggregate.json": "preflight_aggregate_sha256",
        "hu_joint_policy_m43_attempt07_preflight.json": "preflight_plan_sha256",
        "preflight_closure/manifest.json": "preflight_manifest_sha256",
        "preflight_closure/shards_manifest.jsonl": "preflight_schedule_sha256",
        "preflight_closure/launch_authorization.json": (
            "preflight_launch_authorization_sha256"
        ),
    }
    for name, field in bindings.items():
        _require_hash(run_dir / name, str(manifest[field]), f"existing {name}")
    done_hashes = manifest.get("preflight_done_sha256")
    if not isinstance(done_hashes, Mapping) or set(done_hashes) != _PREFLIGHT_SLOTS:
        raise ValueError("existing Attempt07 preflight DONE closure changed")
    for label, expected in done_hashes.items():
        _require_hash(
            run_dir / "preflight_closure" / f"{label}.DONE.json",
            _require_sha256(expected, f"existing preflight DONE {label}"),
            f"existing preflight DONE {label}",
        )
    return manifest


def package_attempt07_spot(
    *,
    repo_root: str | Path,
    run_dir: str | Path,
    run_name: str,
    plan_path: str | Path,
    status_path: str | Path,
    model_path: str | Path,
    startup_path: str | Path,
    preflight_aggregate_path: str | Path,
    preflight_plan_path: str | Path,
    preflight_manifest_path: str | Path,
    preflight_schedule_path: str | Path,
    preflight_launch_authorization_path: str | Path,
    preflight_done_paths: Mapping[str, str | Path],
    resume_existing: bool = False,
) -> dict[str, Any]:
    """Build an immutable package without generating or reading a root."""

    if RUN_NAME_RE.fullmatch(run_name) is None:
        raise ValueError("RunName is not a safe GCP identity")
    root = Path(repo_root).resolve()
    destination = Path(run_dir).resolve()
    expected = (root / "outputs" / "gcp_runs" / run_name).resolve()
    if os.path.normcase(str(destination)) != os.path.normcase(str(expected)):
        raise ValueError("Attempt07 run directory must be outputs/gcp_runs/<run_name>")
    if destination.exists():
        if not resume_existing:
            raise FileExistsError(f"Attempt07 run directory exists: {destination}")
        manifest = _validate_existing_package(destination, run_name)
        return _package_result(destination, manifest, resumed=True)

    plan_file = Path(plan_path).resolve()
    status_file = Path(status_path).resolve()
    model_file = Path(model_path).resolve()
    startup_file = Path(startup_path).resolve()
    preflight_file = Path(preflight_aggregate_path).resolve()
    preflight_plan_file = Path(preflight_plan_path).resolve()
    preflight_manifest_file = Path(preflight_manifest_path).resolve()
    preflight_schedule_file = Path(preflight_schedule_path).resolve()
    preflight_launch_authorization_file = Path(
        preflight_launch_authorization_path
    ).resolve()
    if set(preflight_done_paths) != _PREFLIGHT_SLOTS:
        raise ValueError("Attempt07 package requires exactly five preflight DONE files")
    preflight_done_files = {
        label: Path(path).resolve() for label, path in preflight_done_paths.items()
    }
    plan = load_and_validate_attempt07_plan(plan_file)
    load_and_validate_attempt07_status(status_file)
    _require_hash(plan_file, M43_ATTEMPT07_PLAN_SHA256, "Attempt07 plan")
    _require_hash(model_file, ATTEMPT06_FROZEN_MODEL_SHA256, "Lambda ranker")
    _require_hash(root / "src/ofc_regular/ai_profiles.py", AI_PROFILES_SHA256, "AI profiles")
    preflight = _validate_preflight_go(
        preflight_file,
        preflight_plan_sha256=sha256_file(preflight_plan_file),
    )
    if (
        preflight["contract"]["preflight_plan_sha256"]
        != sha256_file(preflight_plan_file)
    ):
        raise ValueError("Attempt07 preflight plan changed after proof aggregation")
    preflight_manifest = _load_canonical_mapping(
        preflight_manifest_file, "Attempt07 preflight package manifest"
    )
    preflight_authorization = _load_canonical_mapping(
        preflight_launch_authorization_file,
        "Attempt07 preflight launch authorization",
    )
    preflight_schedule_raw = preflight_schedule_file.read_bytes()
    try:
        preflight_schedule = [
            json.loads(line)
            for line in preflight_schedule_raw.decode("utf-8").splitlines()
            if line.strip()
        ]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Attempt07 preflight schedule is not JSONL") from exc
    from .hu_m43_attempt07_preflight_spot import (  # local to avoid a cycle
        LAUNCH_AUTHORIZATION_SCHEMA as PREFLIGHT_AUTHORIZATION_SCHEMA,
        PACKAGE_MANIFEST_SCHEMA as PREFLIGHT_PACKAGE_MANIFEST_SCHEMA,
        build_preflight_schedule,
    )

    expected_preflight_schedule = list(build_preflight_schedule())
    if (
        preflight_schedule != expected_preflight_schedule
        or preflight_schedule_raw
        != b"".join(canonical_json_bytes(row) for row in preflight_schedule)
        or preflight_manifest.get("schema") != PREFLIGHT_PACKAGE_MANIFEST_SCHEMA
        or preflight_manifest.get("status")
        != "packaged_attempt06_copy_plus_overlay_without_execution"
        or preflight_manifest.get("schedule_sha256")
        != sha256_file(preflight_schedule_file)
        or preflight_manifest.get("preflight_plan_sha256")
        != sha256_file(preflight_plan_file)
        or preflight_manifest.get("attempt07_plan_sha256")
        != M43_ATTEMPT07_PLAN_SHA256
        or preflight_manifest.get("model_sha256")
        != ATTEMPT06_FROZEN_MODEL_SHA256
        or preflight_manifest.get("ai_profiles_sha256") != AI_PROFILES_SHA256
        or preflight_manifest.get("new_root_generated") is not False
        or preflight_manifest.get("teacher_executed") is not False
        or preflight_manifest.get("current_profile_mutated") is not False
        or preflight_authorization.get("schema")
        != PREFLIGHT_AUTHORIZATION_SCHEMA
        or preflight_authorization.get("status")
        != "authorized_for_bounded_spot_preflight"
        or preflight_authorization.get("manifest_sha256")
        != sha256_file(preflight_manifest_file)
        or preflight_authorization.get("spot_authorized") is not True
        or preflight_authorization.get("current_profile_mutated") is not False
        or preflight_authorization.get("runtime_policy_activated") is not False
    ):
        raise ValueError("Attempt07 exact preflight package closure changed")
    preflight_done_sha256: dict[str, str] = {}
    for label in sorted(_PREFLIGHT_SLOTS):
        done = _load_canonical_mapping(
            preflight_done_files[label], f"Attempt07 preflight DONE {label}"
        )
        if (
            done.get("schema") != "hu_m43_attempt07_preflight_spot_done_v1"
            or done.get("status") != "complete"
            or done.get("job_id") != label
            or done.get("manifest_sha256") != sha256_file(preflight_manifest_file)
            or done.get("authorization_sha256")
            != sha256_file(preflight_launch_authorization_file)
            or done.get("schedule_sha256") != sha256_file(preflight_schedule_file)
            or done.get("current_profile_mutated") is not False
            or done.get("runtime_policy_activated") is not False
        ):
            raise ValueError(f"Attempt07 preflight DONE identity changed: {label}")
        preflight_done_sha256[label] = sha256_file(preflight_done_files[label])
    schedule = build_attempt07_spot_schedule(plan)

    template_root = root / "outputs/gcp_runs" / PINNED_TEMPLATE_RUN / "package_src"
    model_manifest, native_manifest = _verify_pinned_runtime_closure(template_root)
    staging = destination.with_name(destination.name + ".building")
    if staging.exists():
        raise FileExistsError(f"stale Attempt07 staging directory: {staging}")
    staging.mkdir(parents=True)
    try:
        package_root = staging / "package_src"
        package_root.mkdir()
        _copy_source_tree(root / "src/ofc_regular", package_root / "src/ofc_regular")
        _copy_manifest_files(template_root, package_root, model_manifest["models"])
        _copy_manifest_files(template_root, package_root, native_manifest["binaries"])
        _copy_file(
            template_root / "source_model_manifest.json",
            package_root / "source_model_manifest.json",
        )
        _copy_file(
            template_root / "source_native_manifest.json",
            package_root / "source_native_manifest.json",
        )
        _copy_file(model_file, package_root / "artifacts/lambda_rank_candidate.pkl")
        _copy_file(plan_file, package_root / "configs/hu_joint_policy_m43_attempt07.json")
        _copy_file(
            status_file,
            package_root / "configs/hu_joint_policy_m43_attempt07_status.json",
        )
        _copy_file(
            preflight_plan_file,
            package_root / "configs/hu_joint_policy_m43_attempt07_preflight.json",
        )
        _copy_file(
            preflight_file,
            package_root / "frozen/attempt07_preflight_aggregate.json",
        )
        _copy_file(
            preflight_manifest_file,
            package_root / "frozen/preflight/manifest.json",
        )
        _copy_file(
            preflight_schedule_file,
            package_root / "frozen/preflight/shards_manifest.jsonl",
        )
        _copy_file(
            preflight_launch_authorization_file,
            package_root / "frozen/preflight/launch_authorization.json",
        )
        for label in sorted(_PREFLIGHT_SLOTS):
            _copy_file(
                preflight_done_files[label],
                package_root / f"frozen/preflight/done/{label}.json",
            )
        _copy_file(
            root / "src/ofc_regular/ai_profiles.py",
            package_root / "frozen/ai_profiles.py",
        )
        _atomic_write(package_root / "shards_manifest.jsonl", _schedule_bytes(schedule))
        _atomic_write(
            package_root / "requirements-attempt07.txt", REQUIREMENTS.encode("ascii")
        )
        closure = {
            "schema": SOURCE_CLOSURE_SCHEMA,
            "status": "closed_before_any_development_root",
            "run_name": run_name,
            "plan_sha256": sha256_file(plan_file),
            "status_sha256": sha256_file(status_file),
            "model_sha256": sha256_file(model_file),
            "ai_profiles_sha256": AI_PROFILES_SHA256,
            "preflight_aggregate_sha256": sha256_file(preflight_file),
            "preflight_plan_sha256": sha256_file(preflight_plan_file),
            "preflight_manifest_sha256": sha256_file(preflight_manifest_file),
            "preflight_schedule_sha256": sha256_file(preflight_schedule_file),
            "preflight_launch_authorization_sha256": sha256_file(
                preflight_launch_authorization_file
            ),
            "preflight_done_sha256": preflight_done_sha256,
            "files": _closure_rows(package_root),
            "fresh_root_opened": False,
            "teacher_executed": False,
            "current_profile_mutated": False,
        }
        _write_json(package_root / "source_closure_manifest.json", closure)

        schedule_path = staging / "shards_manifest.jsonl"
        startup_copy = staging / "startup_hu_m43_attempt07_development.sh"
        closure_copy = staging / "source_closure_manifest.json"
        plan_copy = staging / "hu_joint_policy_m43_attempt07.json"
        status_copy = staging / "hu_joint_policy_m43_attempt07_status.json"
        preflight_copy = staging / "attempt07_preflight_aggregate.json"
        preflight_plan_copy = staging / "hu_joint_policy_m43_attempt07_preflight.json"
        preflight_closure = staging / "preflight_closure"
        preflight_closure.mkdir()
        _copy_file(package_root / "shards_manifest.jsonl", schedule_path)
        _copy_file(startup_file, startup_copy)
        _copy_file(package_root / "source_closure_manifest.json", closure_copy)
        _copy_file(plan_file, plan_copy)
        _copy_file(status_file, status_copy)
        _copy_file(preflight_file, preflight_copy)
        _copy_file(preflight_plan_file, preflight_plan_copy)
        _copy_file(preflight_manifest_file, preflight_closure / "manifest.json")
        _copy_file(
            preflight_schedule_file, preflight_closure / "shards_manifest.jsonl"
        )
        _copy_file(
            preflight_launch_authorization_file,
            preflight_closure / "launch_authorization.json",
        )
        for label in sorted(_PREFLIGHT_SLOTS):
            _copy_file(
                preflight_done_files[label],
                preflight_closure / f"{label}.DONE.json",
            )
        source_zip = staging / "ofc_regular_hu_m43_attempt07_development_source.zip"
        _write_deterministic_zip(package_root, source_zip)
        manifest = {
            "schema": PACKAGE_MANIFEST_SCHEMA,
            "status": "frozen_package_only_no_root_opened",
            "run_name": run_name,
            "plan_sha256": sha256_file(plan_file),
            "status_sha256": sha256_file(status_file),
            "model_sha256": sha256_file(model_file),
            "ai_profiles_sha256": AI_PROFILES_SHA256,
            "preflight_aggregate_sha256": sha256_file(preflight_file),
            "preflight_plan_sha256": sha256_file(preflight_plan_file),
            "preflight_manifest_sha256": sha256_file(preflight_manifest_file),
            "preflight_schedule_sha256": sha256_file(preflight_schedule_file),
            "preflight_launch_authorization_sha256": sha256_file(
                preflight_launch_authorization_file
            ),
            "preflight_done_sha256": preflight_done_sha256,
            "schedule_sha256": sha256_file(schedule_path),
            "source_closure_sha256": sha256_file(closure_copy),
            "source_zip_sha256": sha256_file(source_zip),
            "startup_sha256": sha256_file(startup_copy),
            "source_model_manifest_sha256": PINNED_MODEL_MANIFEST_SHA256,
            "source_native_manifest_sha256": PINNED_NATIVE_MANIFEST_SHA256,
            "total_roots": EXPECTED_SHARDS,
            "total_shards": EXPECTED_SHARDS,
            "roots_per_shard": ROOTS_PER_SHARD,
            "root_profile_assignment": "root_index_mod_5_in_frozen_profile_order",
            "batch_child_selectors": True,
            "native_batch_threads": NATIVE_BATCH_THREADS,
            "recommended_machine_type": "c4-standard-4",
            "recommended_wave_shards": MAX_WAVE_SHARDS,
            "fresh_root_opened": False,
            "teacher_executed": False,
            "gcloud_invoked": False,
            "instances_created": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        }
        _write_json(staging / "manifest.json", manifest)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return _package_result(destination, manifest, resumed=False)


def _package_result(
    run_dir: Path, manifest: Mapping[str, Any], *, resumed: bool
) -> dict[str, Any]:
    return {
        "schema": PACKAGE_RESULT_SCHEMA,
        "status": "packaged_without_root_or_gcloud",
        "run_name": manifest["run_name"],
        "run_dir": str(run_dir),
        "manifest": str(run_dir / "manifest.json"),
        "manifest_sha256": sha256_file(run_dir / "manifest.json"),
        "total_shards": EXPECTED_SHARDS,
        "roots_per_shard": ROOTS_PER_SHARD,
        "resumed_existing_package": resumed,
        "fresh_root_opened": False,
        "teacher_executed": False,
        "gcloud_invoked": False,
        "instances_created": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }


def load_schedule(path: str | Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if len(rows) != EXPECTED_SHARDS:
        raise ValueError("Attempt07 schedule must contain 100 rows")
    for index, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or row.get("schema") != SHARD_SCHEMA
            or row.get("shard") != index
            or row.get("root_index") != index
            or row.get("output_prefix") != f"shard_{index:03d}"
        ):
            raise ValueError("Attempt07 schedule ordering changed")
    return rows


def create_attempt07_output_consumption_claim(
    *,
    done_root: str | Path,
    schedule_path: str | Path,
    manifest_path: str | Path,
    authorization_path: str | Path,
    preflight_aggregate_path: str | Path,
    output: str | Path,
    resume_existing: bool = False,
) -> dict[str, Any]:
    """Claim complete development output after DONE-only verification.

    This function intentionally has no parameter for a result-content path.
    It reads exactly the 100 operational DONE records and package closure.
    """

    destination = Path(output)
    manifest_file = Path(manifest_path)
    schedule_file = Path(schedule_path)
    authorization_file = Path(authorization_path)
    manifest = _load_canonical_mapping(manifest_file, "Attempt07 claim manifest")
    if (
        set(manifest) != _PACKAGE_MANIFEST_KEYS
        or manifest.get("schema") != PACKAGE_MANIFEST_SCHEMA
        or manifest.get("status") != "frozen_package_only_no_root_opened"
        or manifest.get("total_shards") != EXPECTED_SHARDS
        or manifest.get("run_name") is None
        or sha256_file(schedule_file) != manifest.get("schedule_sha256")
    ):
        raise ValueError("Attempt07 claim manifest closure changed")
    validate_attempt07_spot_authorization(
        authorization_path=authorization_file,
        manifest_path=manifest_file,
        preflight_aggregate_path=preflight_aggregate_path,
    )
    authorization_sha256 = sha256_file(authorization_file)
    schedule = load_schedule(schedule_file)
    done_hashes: dict[str, str] = {}
    root = Path(done_root)
    for spec in schedule:
        shard = int(spec["shard"])
        path = root / f"DONE-{shard:03d}.json"
        done = _load_canonical_mapping(path, f"Attempt07 DONE {shard}")
        if set(done) != _DONE_KEYS:
            raise ValueError(f"Attempt07 DONE fields changed: {shard}")
        expected = {
            "schema": DONE_SCHEMA,
            "status": "complete",
            "run_name": manifest["run_name"],
            "run_id": f"{manifest['run_name']}:shard={shard}",
            "shard": shard,
            "root_index": shard,
            "root_profile": spec["root_profile"],
            "seeds": spec["seeds"],
            "output_prefix": spec["output_prefix"],
            "manifest_sha256": sha256_file(manifest_file),
            "source_sha256": manifest["source_zip_sha256"],
            "startup_sha256": manifest["startup_sha256"],
            "authorization_sha256": authorization_sha256,
            "schedule_sha256": manifest["schedule_sha256"],
            "plan_sha256": manifest["plan_sha256"],
            "status_sha256": manifest["status_sha256"],
            "model_sha256": manifest["model_sha256"],
            "ai_profiles_sha256": manifest["ai_profiles_sha256"],
            "preflight_plan_sha256": manifest["preflight_plan_sha256"],
            "preflight_aggregate_sha256": manifest[
                "preflight_aggregate_sha256"
            ],
            "source_closure_sha256": manifest["source_closure_sha256"],
            "source_model_manifest_sha256": manifest[
                "source_model_manifest_sha256"
            ],
            "source_native_manifest_sha256": manifest[
                "source_native_manifest_sha256"
            ],
            "native_batch_threads": NATIVE_BATCH_THREADS,
            "teacher_values_are_realized_match_ev": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        }
        if any(
            done.get(key) != value or type(done.get(key)) is not type(value)
            for key, value in expected.items()
        ):
            raise ValueError(f"Attempt07 DONE-only claim identity changed: {shard}")
        for field in (
            "output_sha256",
            "checkpoint_sha256",
            "heartbeat_sha256",
            "generator_summary_sha256",
            "run_log_sha256",
            "config_sha256",
            "global_claim_sha256",
            "root_claim_sha256",
        ):
            _require_sha256(done.get(field), f"Attempt07 DONE {shard}.{field}")
        _require_nonnegative_finite(
            done.get("elapsed_seconds"), f"Attempt07 DONE {shard}.elapsed"
        )
        peak_rss = _require_nonnegative_finite(
            done.get("peak_rss_bytes"), f"Attempt07 DONE {shard}.rss"
        )
        if not peak_rss.is_integer():
            raise ValueError(f"Attempt07 DONE {shard}.rss must be an integer")
        done_hashes[f"{shard:03d}"] = sha256_file(path)
    payload = {
        "schema": OUTPUT_CONSUMPTION_SCHEMA,
        "status": "claimed_after_all_done_before_any_teacher_read",
        "run_name": manifest["run_name"],
        "manifest_sha256": sha256_file(manifest_file),
        "schedule_sha256": sha256_file(schedule_file),
        "authorization_sha256": authorization_sha256,
        "done_sha256": done_hashes,
        "expected_shards": EXPECTED_SHARDS,
        "expected_roots": EXPECTED_SHARDS,
        "all_done_markers_verified": True,
        "result_objects_addressed_when_claimed": False,
        "selector_executed": False,
        "fit_performed": False,
        "threshold_selected": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    encoded = canonical_json_bytes(payload)
    if destination.exists():
        if not resume_existing or destination.read_bytes() != encoded:
            raise FileExistsError("Attempt07 output consumption claim differs or is owned")
    else:
        _atomic_write(destination, encoded)
    return payload


def validate_received_attempt07_shard(
    *,
    shard_dir: str | Path,
    schedule_path: str | Path,
    manifest_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Validate one downloaded shard against DONE, claims, and producer row."""

    directory = Path(shard_dir)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(destination)
    manifest = _load_canonical_mapping(
        Path(manifest_path), "Attempt07 package manifest"
    )
    if (
        set(manifest) != _PACKAGE_MANIFEST_KEYS
        or manifest.get("schema") != PACKAGE_MANIFEST_SCHEMA
        or manifest.get("status") != "frozen_package_only_no_root_opened"
        or manifest.get("plan_sha256") != M43_ATTEMPT07_PLAN_SHA256
        or manifest.get("model_sha256") != ATTEMPT06_FROZEN_MODEL_SHA256
        or manifest.get("ai_profiles_sha256") != AI_PROFILES_SHA256
        or manifest.get("total_shards") != EXPECTED_SHARDS
        or manifest.get("total_roots") != EXPECTED_SHARDS
        or manifest.get("roots_per_shard") != ROOTS_PER_SHARD
        or manifest.get("native_batch_threads") != NATIVE_BATCH_THREADS
        or manifest.get("fresh_root_opened") is not False
        or manifest.get("teacher_executed") is not False
        or manifest.get("current_profile_mutated") is not False
        or manifest.get("runtime_policy_activated") is not False
    ):
        raise ValueError("Attempt07 package manifest schema changed")
    if sha256_file(schedule_path) != manifest.get("schedule_sha256"):
        raise ValueError("Attempt07 receive schedule disagrees with manifest")
    schedule = load_schedule(schedule_path)
    done = _load_canonical_mapping(directory / "DONE.json", "Attempt07 DONE")
    if set(done) != _DONE_KEYS:
        raise ValueError("Attempt07 DONE fields changed")
    shard = done.get("shard")
    if type(shard) is not int or not 0 <= shard < EXPECTED_SHARDS:
        raise ValueError("Attempt07 DONE shard is invalid")
    spec = schedule[shard]
    closure = {
        "schema": DONE_SCHEMA,
        "status": "complete",
        "run_name": manifest["run_name"],
        "run_id": f"{manifest['run_name']}:shard={shard}",
        "shard": shard,
        "root_index": shard,
        "root_profile": spec["root_profile"],
        "seeds": spec["seeds"],
        "output_prefix": spec["output_prefix"],
        "manifest_sha256": sha256_file(manifest_path),
        "source_sha256": manifest["source_zip_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "schedule_sha256": manifest["schedule_sha256"],
        "plan_sha256": manifest["plan_sha256"],
        "status_sha256": manifest["status_sha256"],
        "model_sha256": manifest["model_sha256"],
        "ai_profiles_sha256": manifest["ai_profiles_sha256"],
        "preflight_aggregate_sha256": manifest["preflight_aggregate_sha256"],
        "preflight_plan_sha256": manifest["preflight_plan_sha256"],
        "source_closure_sha256": manifest["source_closure_sha256"],
        "source_model_manifest_sha256": manifest[
            "source_model_manifest_sha256"
        ],
        "source_native_manifest_sha256": manifest[
            "source_native_manifest_sha256"
        ],
        "native_batch_threads": NATIVE_BATCH_THREADS,
        "teacher_values_are_realized_match_ev": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    if any(done.get(key) != value for key, value in closure.items()):
        raise ValueError("Attempt07 DONE closure changed")
    for field in (
        "authorization_sha256",
        "global_claim_sha256",
        "root_claim_sha256",
        "output_sha256",
        "checkpoint_sha256",
        "heartbeat_sha256",
        "generator_summary_sha256",
        "run_log_sha256",
        "config_sha256",
    ):
        _require_sha256(done.get(field), f"Attempt07 DONE {field}")
    _require_nonnegative_finite(done.get("elapsed_seconds"), "DONE elapsed_seconds")
    peak_rss = _require_nonnegative_finite(
        done.get("peak_rss_bytes"), "DONE peak_rss_bytes"
    )
    if not float(peak_rss).is_integer():
        raise ValueError("DONE peak_rss_bytes must be an integer")
    artifact_fields = {
        "teacher.jsonl": "output_sha256",
        "checkpoint.json": "checkpoint_sha256",
        "heartbeat.json": "heartbeat_sha256",
        "generator_summary.json": "generator_summary_sha256",
        "run.log": "run_log_sha256",
        "global_claim.json": "global_claim_sha256",
        "root_claim.json": "root_claim_sha256",
        "authorization.json": "authorization_sha256",
    }
    for name, field in artifact_fields.items():
        _require_hash(directory / name, str(done[field]), f"received {name}")
    global_claim = _load_canonical_mapping(
        directory / "global_claim.json", "global claim"
    )
    root_claim = _load_canonical_mapping(directory / "root_claim.json", "root claim")
    authorization_audit = validate_attempt07_spot_authorization(
        authorization_path=directory / "authorization.json",
        manifest_path=manifest_path,
        preflight_aggregate_path=(
            Path(manifest_path).parent / "attempt07_preflight_aggregate.json"
        ),
    )
    if authorization_audit["authorization_sha256"] != done["authorization_sha256"]:
        raise ValueError("Attempt07 authorization SHA disagrees with DONE")
    common_claim = {
        "run_name": manifest["run_name"],
        "manifest_sha256": sha256_file(manifest_path),
        "source_sha256": manifest["source_zip_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "schedule_sha256": manifest["schedule_sha256"],
        "plan_sha256": manifest["plan_sha256"],
        "status_sha256": manifest["status_sha256"],
        "model_sha256": manifest["model_sha256"],
        "ai_profiles_sha256": manifest["ai_profiles_sha256"],
        "preflight_aggregate_sha256": manifest["preflight_aggregate_sha256"],
        "preflight_plan_sha256": manifest["preflight_plan_sha256"],
        "source_closure_sha256": manifest["source_closure_sha256"],
        "source_model_manifest_sha256": manifest[
            "source_model_manifest_sha256"
        ],
        "source_native_manifest_sha256": manifest[
            "source_native_manifest_sha256"
        ],
        "authorization_sha256": done["authorization_sha256"],
        "native_batch_threads": NATIVE_BATCH_THREADS,
    }
    if (
        set(global_claim) != _GLOBAL_CLAIM_KEYS
        or global_claim.get("schema") != GLOBAL_CLAIM_SCHEMA
        or global_claim.get("status") != "claimed_before_any_development_root"
        or global_claim.get("roots") != EXPECTED_SHARDS
        or global_claim.get("shards") != EXPECTED_SHARDS
        or global_claim.get("roots_per_shard") != ROOTS_PER_SHARD
        or global_claim.get("development_started_when_claimed") is not False
        or global_claim.get("current_profile_mutated") is not False
        or global_claim.get("runtime_policy_activated") is not False
        or any(global_claim.get(key) != value for key, value in common_claim.items())
    ):
        raise ValueError("Attempt07 global claim changed")
    if (
        set(root_claim) != _ROOT_CLAIM_KEYS
        or root_claim.get("schema") != ROOT_CLAIM_SCHEMA
        or root_claim.get("status") != "claimed_before_root_generation"
        or root_claim.get("run_id") != f"{manifest['run_name']}:shard={shard}"
        or root_claim.get("root_index") != shard
        or root_claim.get("root_profile") != spec["root_profile"]
        or root_claim.get("seeds") != spec["seeds"]
        or root_claim.get("output_prefix") != spec["output_prefix"]
        or root_claim.get("alternate_seed_or_result_retry_allowed") is not False
        or root_claim.get("deterministic_recompute_allowed") is not True
        or root_claim.get("deterministic_recompute_mode")
        != ROOT_REMATERIALIZATION_MODE
        or any(root_claim.get(key) != value for key, value in common_claim.items())
    ):
        raise ValueError("Attempt07 root claim changed")

    raw = (directory / "teacher.jsonl").read_bytes()
    lines = [line for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("Attempt07 shard must contain one teacher row")
    row = json.loads(lines[0])
    if not isinstance(row, dict) or row.get("schema") != ATTEMPT07_SHARD_ROW_SCHEMA:
        raise ValueError("Attempt07 received row schema changed")
    if raw != canonical_json_bytes(row):
        raise ValueError("Attempt07 received row is not canonical JSONL")
    provenance = row.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("run_id")
        != f"{manifest['run_name']}:shard={shard}"
    ):
        raise ValueError("Attempt07 received row run_id changed")
    validated = _validate_row(row, root_index=shard, expected_seeds=spec["seeds"])
    if validated["config_sha256"] != done["config_sha256"]:
        raise ValueError("Attempt07 DONE config SHA disagrees with row provenance")
    checkpoint = _load_mapping(directory / "checkpoint.json", "checkpoint")
    heartbeat = _load_mapping(directory / "heartbeat.json", "heartbeat")
    generator_summary = _load_mapping(
        directory / "generator_summary.json", "generator summary"
    )
    if (
        checkpoint.get("schema") != ATTEMPT07_CHECKPOINT_SCHEMA
        or checkpoint.get("completed_roots") != 1
        or checkpoint.get("root_index") != shard
        or checkpoint.get("config_sha256") != done["config_sha256"]
        or heartbeat.get("schema") != ATTEMPT07_HEARTBEAT_SCHEMA
        or heartbeat.get("status") != "complete"
        or heartbeat.get("root_index") != shard
        or heartbeat.get("output_sha256") != done["output_sha256"]
        or generator_summary.get("schema") != ATTEMPT07_SHARD_SUMMARY_SCHEMA
        or generator_summary.get("status") != "complete"
        or generator_summary.get("root_index") != shard
        or generator_summary.get("config_sha256") != done["config_sha256"]
        or generator_summary.get("output_sha256") != done["output_sha256"]
    ):
        raise ValueError("Attempt07 checkpoint/heartbeat closure changed")
    audit = {
        "schema": RECEIVE_AUDIT_SCHEMA,
        "status": "pass",
        "shard": shard,
        "root_index": shard,
        "root_profile": spec["root_profile"],
        "observation_fingerprint": validated["fingerprint"],
        "config_sha256": validated["config_sha256"],
        "output_sha256": done["output_sha256"],
        "manifest_sha256": sha256_file(manifest_path),
        "authorization_sha256": done["authorization_sha256"],
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    _write_json(destination, audit)
    return audit


def merge_received_attempt07_shards(
    *, received_root: str | Path,
    schedule_path: str | Path,
    manifest_path: str | Path,
    consumption_claim_path: str | Path,
    output: str | Path,
    receipt: str | Path,
) -> dict[str, Any]:
    """Merge exactly 100 audited rows in root order without selecting an arm."""

    root = Path(received_root)
    destination = Path(output)
    receipt_path = Path(receipt)
    if destination.exists() or receipt_path.exists():
        raise FileExistsError("Attempt07 merge output or receipt already exists")
    manifest_file = Path(manifest_path)
    manifest_sha256 = sha256_file(manifest_file)
    manifest = _load_canonical_mapping(manifest_file, "Attempt07 merge manifest")
    if (
        set(manifest) != _PACKAGE_MANIFEST_KEYS
        or manifest.get("schema") != PACKAGE_MANIFEST_SCHEMA
        or manifest.get("status") != "frozen_package_only_no_root_opened"
        or manifest.get("total_shards") != EXPECTED_SHARDS
        or manifest.get("plan_sha256") != M43_ATTEMPT07_PLAN_SHA256
        or manifest.get("model_sha256") != ATTEMPT06_FROZEN_MODEL_SHA256
        or manifest.get("ai_profiles_sha256") != AI_PROFILES_SHA256
        or sha256_file(schedule_path) != manifest.get("schedule_sha256")
    ):
        raise ValueError("Attempt07 merge manifest closure changed")
    claim_file = Path(consumption_claim_path)
    claim = _load_canonical_mapping(
        claim_file, "Attempt07 output consumption claim"
    )
    claim_done = claim.get("done_sha256")
    if (
        set(claim) != _OUTPUT_CONSUMPTION_KEYS
        or claim.get("schema") != OUTPUT_CONSUMPTION_SCHEMA
        or claim.get("status") != "claimed_after_all_done_before_any_teacher_read"
        or claim.get("run_name") != manifest.get("run_name")
        or claim.get("manifest_sha256") != manifest_sha256
        or claim.get("schedule_sha256") != manifest.get("schedule_sha256")
        or claim.get("all_done_markers_verified") is not True
        or claim.get("expected_shards") != EXPECTED_SHARDS
        or claim.get("expected_roots") != EXPECTED_SHARDS
        or not isinstance(claim_done, Mapping)
        or set(claim_done) != {f"{index:03d}" for index in range(EXPECTED_SHARDS)}
        or any(SHA256_RE.fullmatch(str(value)) is None for value in claim_done.values())
        or claim.get("result_objects_addressed_when_claimed") is not False
        or claim.get("selector_executed") is not False
        or claim.get("fit_performed") is not False
        or claim.get("threshold_selected") is not False
        or claim.get("current_profile_mutated") is not False
        or claim.get("runtime_policy_activated") is not False
    ):
        raise ValueError("Attempt07 output consumption claim changed")
    claim_sha256 = sha256_file(claim_file)
    schedule = load_schedule(schedule_path)
    encoded_rows: list[bytes] = []
    audit_hashes: list[str] = []
    for spec in schedule:
        directory = root / str(spec["output_prefix"])
        audit = _load_mapping(directory / "received_audit.json", "received audit")
        done_path = directory / "DONE.json"
        if (
            audit.get("schema") != RECEIVE_AUDIT_SCHEMA
            or audit.get("status") != "pass"
            or audit.get("root_index") != spec["root_index"]
            or audit.get("manifest_sha256") != manifest_sha256
            or audit.get("authorization_sha256") != claim.get("authorization_sha256")
            or sha256_file(done_path)
            != claim_done[f"{int(spec['root_index']):03d}"]
        ):
            raise ValueError("Attempt07 merge found an unaudited shard")
        teacher_path = directory / "teacher.jsonl"
        raw = teacher_path.read_bytes()
        rows = [line for line in raw.decode("utf-8").splitlines() if line.strip()]
        if len(rows) != 1:
            raise ValueError("Attempt07 merge shard row count changed")
        row = json.loads(rows[0])
        if row.get("root_index") != spec["root_index"]:
            raise ValueError("Attempt07 merge root ordering changed")
        canonical = canonical_json_bytes(row)
        if hashlib.sha256(canonical).hexdigest() != audit.get("output_sha256"):
            raise ValueError("Attempt07 merge row hash changed after audit")
        encoded_rows.append(canonical)
        audit_hashes.append(sha256_file(directory / "received_audit.json"))
    _atomic_write(destination, b"".join(encoded_rows))
    result = {
        "schema": RECEIVE_MERGE_SCHEMA,
        "status": "complete_no_selection",
        "roots": EXPECTED_SHARDS,
        "root_indices": "0..99",
        "profiles": {profile: 20 for profile in M43_ATTEMPT07_PROFILES},
        "merged_sha256": sha256_file(destination),
        "schedule_sha256": sha256_file(schedule_path),
        "manifest_sha256": manifest_sha256,
        "consumption_claim_sha256": claim_sha256,
        "received_audit_sha256": audit_hashes,
        "selector_executed": False,
        "selector_command_required": (
            "python -m ofc_regular.select_hu_m43_attempt07_development_arm"
        ),
        "fit_performed": False,
        "threshold_selected": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    _write_json(receipt_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    package = commands.add_parser("package")
    for name in (
        "repo-root",
        "run-dir",
        "run-name",
        "plan",
        "status",
        "model",
        "startup",
        "preflight-aggregate",
        "preflight-plan",
        "preflight-manifest",
        "preflight-schedule",
        "preflight-launch-authorization",
    ):
        package.add_argument(f"--{name}", required=True)
    package.add_argument("--resume-existing", action="store_true")
    for slot in sorted(_PREFLIGHT_SLOTS):
        package.add_argument(f"--preflight-done-{slot.replace('_', '-')}", required=True)
    authorization = commands.add_parser("validate-authorization")
    authorization.add_argument("--authorization", required=True)
    authorization.add_argument("--manifest", required=True)
    authorization.add_argument("--preflight-aggregate", required=True)
    claim = commands.add_parser("claim-complete-output")
    claim.add_argument("--done-root", required=True)
    claim.add_argument("--schedule", required=True)
    claim.add_argument("--manifest", required=True)
    claim.add_argument("--authorization", required=True)
    claim.add_argument("--preflight-aggregate", required=True)
    claim.add_argument("--output", required=True)
    claim.add_argument("--resume-existing", action="store_true")
    validate = commands.add_parser("validate-received-shard")
    validate.add_argument("--shard-dir", required=True)
    validate.add_argument("--schedule", required=True)
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--output", required=True)
    merge = commands.add_parser("merge-received")
    merge.add_argument("--received-root", required=True)
    merge.add_argument("--schedule", required=True)
    merge.add_argument("--manifest", required=True)
    merge.add_argument("--consumption-claim", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "package":
        result = package_attempt07_spot(
            repo_root=args.repo_root,
            run_dir=args.run_dir,
            run_name=args.run_name,
            plan_path=args.plan,
            status_path=args.status,
            model_path=args.model,
            startup_path=args.startup,
            preflight_aggregate_path=args.preflight_aggregate,
            preflight_plan_path=args.preflight_plan,
            preflight_manifest_path=args.preflight_manifest,
            preflight_schedule_path=args.preflight_schedule,
            preflight_launch_authorization_path=(
                args.preflight_launch_authorization
            ),
            preflight_done_paths={
                slot: getattr(args, f"preflight_done_{slot}")
                for slot in _PREFLIGHT_SLOTS
            },
            resume_existing=args.resume_existing,
        )
    elif args.command == "validate-authorization":
        result = validate_attempt07_spot_authorization(
            authorization_path=args.authorization,
            manifest_path=args.manifest,
            preflight_aggregate_path=args.preflight_aggregate,
        )
    elif args.command == "claim-complete-output":
        result = create_attempt07_output_consumption_claim(
            done_root=args.done_root,
            schedule_path=args.schedule,
            manifest_path=args.manifest,
            authorization_path=args.authorization,
            preflight_aggregate_path=args.preflight_aggregate,
            output=args.output,
            resume_existing=args.resume_existing,
        )
    elif args.command == "validate-received-shard":
        result = validate_received_attempt07_shard(
            shard_dir=args.shard_dir,
            schedule_path=args.schedule,
            manifest_path=args.manifest,
            output=args.output,
        )
    else:
        result = merge_received_attempt07_shards(
            received_root=args.received_root,
            schedule_path=args.schedule,
            manifest_path=args.manifest,
            consumption_claim_path=args.consumption_claim,
            output=args.output,
            receipt=args.receipt,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "AUTHORIZATION_SCHEMA",
    "DONE_SCHEMA",
    "EXPECTED_SHARDS",
    "MAX_WAVE_SHARDS",
    "NATIVE_BATCH_THREADS",
    "PACKAGE_MANIFEST_SCHEMA",
    "ROOT_CLAIM_SCHEMA",
    "ROOT_REMATERIALIZATION_MODE",
    "SHARD_SCHEMA",
    "build_attempt07_spot_schedule",
    "create_attempt07_output_consumption_claim",
    "merge_received_attempt07_shards",
    "package_attempt07_spot",
    "validate_attempt07_spot_authorization",
    "validate_received_attempt07_shard",
]
