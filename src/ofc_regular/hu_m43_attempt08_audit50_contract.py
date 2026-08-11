"""Immutable outer contract for the Attempt08 audit50 overlay.

This module is intentionally not part of the frozen Attempt08 development
runtime closure.  Audit50 is an independently hashed overlay which consumes
the already-frozen development package without modifying it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


AUDIT50_PLAN_SCHEMA = "hu_m43_attempt08_audit50_plan_v1"
AUDIT50_PLAN_SHA256 = (
    "ce39207b88c6ccdeecb07d805f0c881fcf6c6cba6b2beb221d4537602dd0cc2b"
)
SOURCE_PLAN_SHA256 = (
    "f0b8d7c2af40aeeb9b1a1c4e446237afb70ba04713b3d4bce555f26e4617ba51"
)
LAMBDA_MODEL_SHA256 = (
    "e7fa728308c8973e2e202baf79309e30b9b6f58c37431d8ea55850390abc28c3"
)
AI_PROFILES_SHA256 = (
    "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
)
DEVELOPMENT_FUTURE_RUNNER_SHA256 = (
    "b7ab4f08f9725e2fe90510934d9aa0d549a83bab2761ed8baa63d01b4c8601dd"
)
GCP_IMAGE_NAME = "debian-12-bookworm-v20260609"
GCP_IMAGE_ID = "1449487925682397051"
GCP_IMAGE_SELF_LINK = (
    "projects/debian-cloud/global/images/debian-12-bookworm-v20260609"
)
MACHINE_TYPE = "c4-highmem-4"
NATIVE_BATCH_THREADS = 4
MAX_WAVE_SHARDS = 25
ROOT_FIRST = 200
ROOT_LAST = 249
TOTAL_SHARDS = 50
PROFILES = (
    "stage19_p0",
    "stage9f_p2",
    "stage7_m5_r10",
    "stage3_baseline",
    "random_exact_final",
)
BASELINE_PROFILE = "stage18_p1"
CONTINUATION_PROFILE = "stage9f_p2"
SEED_DOMAINS = ("hand", "rerank", "veto", "stress", "assessment", "child")
RUN_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,126}[a-z0-9]$")
HEX = frozenset("0123456789abcdef")

EXPECTED_GATES: Mapping[str, Any] = {
    "states_denominator": 50,
    "fires_total_min": 10,
    "fires_each_profile_min": 1,
    "mean_delta_per_state_strictly_greater_than": 0.0,
    "mean_delta_per_fire_strictly_greater_than": 0.0,
    "false_positive_rate_per_fire_max": 0.4,
    "override_loss_p95_max": 25.0,
    "override_loss_p99_max": 40.0,
    "override_loss_max": 50.0,
    "action_mapping_violation_count_max": 0,
    "rng_domain_violation_count_max": 0,
    "hidden_information_violation_count_max": 0,
    "risk_reserve_contract_violation_count_max": 0,
    "locked_action_change_violation_count_max": 0,
    "nonfire_exact_baseline_action_fallback_required": True,
    "teacher_values_reported_as_realized_match_ev": False,
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def load_json_mapping(
    path: str | Path, label: str, *, canonical: bool = True
) -> dict[str, Any]:
    source = Path(path)
    try:
        raw = source.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {source}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be one JSON mapping")
    if canonical and raw != canonical_json_bytes(value):
        raise ValueError(f"{label} must be canonical JSON")
    return value


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) > HEX:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def atomic_create(path: str | Path, data: bytes, *, allow_identical: bool = False) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if allow_identical and destination.read_bytes() == data:
            return
        raise FileExistsError(f"refusing to overwrite immutable file: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            if allow_identical and destination.read_bytes() == data:
                return
            raise FileExistsError(
                f"immutable file concurrently created: {destination}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def write_canonical_json(
    path: str | Path, value: Mapping[str, Any], *, allow_identical: bool = False
) -> None:
    atomic_create(path, canonical_json_bytes(value), allow_identical=allow_identical)


def load_and_validate_audit50_plan(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if sha256_file(source) != AUDIT50_PLAN_SHA256:
        raise ValueError("Attempt08 audit50 plan SHA-256 changed")
    plan = load_json_mapping(source, "Attempt08 audit50 plan", canonical=False)
    population = plan.get("population")
    execution = plan.get("execution")
    prerequisite = plan.get("development_prerequisite")
    gates = plan.get("audit_go_no_go")
    decision = plan.get("decision_contract")
    guards = plan.get("activation_guards")
    if (
        plan.get("schema") != AUDIT50_PLAN_SCHEMA
        or plan.get("status") != "frozen_before_reserved_audit_open"
        or plan.get("classification")
        != "one_shot_disjoint_search_quality_audit_only"
        or not isinstance(population, Mapping)
        or population.get("name") != "future_audit"
        or population.get("roots") != TOTAL_SHARDS
        or population.get("root_index_first") != ROOT_FIRST
        or population.get("root_index_last") != ROOT_LAST
        or tuple(population.get("profiles", ())) != PROFILES
        or population.get("roots_per_profile") != 10
        or population.get("profile_assignment")
        != "root_index_mod_5_in_frozen_profile_order"
        or population.get("development_or_preflight_seed_reuse_allowed") is not False
        or population.get("alternate_root_seed_or_result_retry_allowed") is not False
        or not isinstance(execution, Mapping)
        or execution.get("shard_layout") != "one_root_per_shard"
        or execution.get("batch_child_selectors") is not True
        or execution.get("native_batch_threads") != NATIVE_BATCH_THREADS
        or execution.get("machine_type") != MACHINE_TYPE
        or execution.get("max_wave_shards") != MAX_WAVE_SHARDS
        or execution.get("heartbeat_required") is not True
        or execution.get("done_published_last") is not True
        or execution.get("current_profile_resolution_allowed") is not False
        or not isinstance(prerequisite, Mapping)
        or prerequisite.get("decision") != "go"
        or prerequisite.get("immutable_development_pass_freeze_required") is not True
        or prerequisite.get("separate_future_audit_open_authorization_required")
        is not True
        or prerequisite.get("audit_plan_sha256_bound_by_launch_authorization")
        is not True
    ):
        raise ValueError("Attempt08 audit50 population/execution contract changed")
    source_plan = plan.get("source_search_plan")
    if (
        not isinstance(source_plan, Mapping)
        or source_plan.get("sha256") != SOURCE_PLAN_SHA256
    ):
        raise ValueError("Attempt08 audit50 source-plan binding changed")
    if not isinstance(gates, Mapping):
        raise ValueError("Attempt08 audit50 gates are missing")
    for key, expected in EXPECTED_GATES.items():
        if gates.get(key) != expected or type(gates.get(key)) is not type(expected):
            raise ValueError(f"Attempt08 audit50 frozen gate changed: {key}")
    if (
        gates.get("all_gates_required") is not True
        or gates.get("assessment_source")
        != "disjoint_A256_locked_final_nonbaseline_output_vs_explicit_baseline"
        or gates.get("quantile_method") != "numpy_linear"
        or gates.get("tail_semantics")
        != "maximum_across_fired_roots_of_each_A256_per_root_loss_metric"
        or not isinstance(decision, Mapping)
        or decision.get("gate_evaluation_count") != 1
        or decision.get("threshold_search_on_audit_allowed") is not False
        or decision.get("fit_on_audit_allowed") is not False
        or decision.get("profile_exclusion_allowed") is not False
        or decision.get("go_action")
        != "authorize_distillation_from_development200_only"
        or decision.get("no_go_action")
        != "close_attempt08_without_fit_threshold_or_runtime_activation"
        or decision.get("realized_population_acceptance_still_required_after_go")
        is not True
        or not isinstance(guards, Mapping)
        or any(value is not False for value in guards.values())
    ):
        raise ValueError("Attempt08 audit50 gate/decision boundary changed")
    return plan


def build_audit50_schedule() -> list[dict[str, Any]]:
    return [
        {
            "schema": "hu_m43_attempt08_audit50_spot_shard_v1",
            "shard": shard,
            "root_index": ROOT_FIRST + shard,
            "roots": 1,
            "root_profile": PROFILES[(ROOT_FIRST + shard) % len(PROFILES)],
            "baseline_profile": BASELINE_PROFILE,
            "continuation_profile": CONTINUATION_PROFILE,
            "output_prefix": f"shard_{shard:03d}",
            "batch_child_selectors": True,
            "native_batch_threads": NATIVE_BATCH_THREADS,
            "machine_type": MACHINE_TYPE,
            "seed_material_opened": False,
            "current_profile_resolved": False,
        }
        for shard in range(TOTAL_SHARDS)
    ]


def schedule_bytes(schedule: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in schedule)


def validate_schedule_bytes(raw: bytes) -> list[dict[str, Any]]:
    if not raw.endswith(b"\n") or b"\r" in raw:
        raise ValueError("Attempt08 audit50 schedule must use canonical LF JSONL")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(raw.splitlines()):
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Attempt08 audit50 schedule row {index} is invalid") from exc
        if not isinstance(value, dict):
            raise ValueError("Attempt08 audit50 schedule rows must be mappings")
        rows.append(value)
    expected = build_audit50_schedule()
    if rows != expected or raw != schedule_bytes(expected):
        raise ValueError("Attempt08 audit50 schedule changed")
    return rows


__all__ = [
    "AI_PROFILES_SHA256",
    "AUDIT50_PLAN_SHA256",
    "BASELINE_PROFILE",
    "CONTINUATION_PROFILE",
    "DEVELOPMENT_FUTURE_RUNNER_SHA256",
    "EXPECTED_GATES",
    "GCP_IMAGE_ID",
    "GCP_IMAGE_NAME",
    "GCP_IMAGE_SELF_LINK",
    "LAMBDA_MODEL_SHA256",
    "MACHINE_TYPE",
    "MAX_WAVE_SHARDS",
    "NATIVE_BATCH_THREADS",
    "PROFILES",
    "ROOT_FIRST",
    "ROOT_LAST",
    "RUN_NAME_RE",
    "SEED_DOMAINS",
    "SOURCE_PLAN_SHA256",
    "TOTAL_SHARDS",
    "atomic_create",
    "build_audit50_schedule",
    "canonical_json_bytes",
    "canonical_sha256",
    "load_and_validate_audit50_plan",
    "load_json_mapping",
    "require_sha256",
    "schedule_bytes",
    "sha256_file",
    "validate_schedule_bytes",
    "write_canonical_json",
]
