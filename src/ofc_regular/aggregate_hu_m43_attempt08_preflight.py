"""Aggregate the five redacted Attempt08 correctness-preflight proofs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hu_m43_attempt06_teacher import (
    ATTEMPT06_SHARD_ROW_SCHEMA,
    ATTEMPT06_TEACHER_SCHEMA,
)
from .hu_m43_attempt08_contract import (
    AI_PROFILES_SHA256,
    ATTEMPT08_LAMBDA_MODEL_SHA256,
    M43_ATTEMPT08_PLAN_SCHEMA,
    M43_ATTEMPT08_PLAN_SHA256,
)
from .hu_m43_attempt08_teacher import ATTEMPT08_SOLVER_ID, ATTEMPT08_TEACHER_SCHEMA
from .hu_m43_attempt08_runtime_anchor import ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256
from .hu_m43_attempt08_runtime_anchor_contract import (
    ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256,
    ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256,
)
from .hu_m43_attempt08_runtime_identity import (
    ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
    ATTEMPT08_GCP_IMAGE_ID,
    ATTEMPT08_GCP_IMAGE_NAME,
)
from .run_hu_m43_attempt07_preflight import _atomic_write_once, _sha256_value
from .run_hu_m43_attempt08_preflight import (
    ATTEMPT08_PREFLIGHT_NATIVE_BATCH_THREADS,
    ATTEMPT08_PREFLIGHT_PLAN_SCHEMA,
    ATTEMPT08_PREFLIGHT_PLAN_SHA256,
    ATTEMPT08_PREFLIGHT_PROOF_SCHEMA,
    ATTEMPT08_PREFLIGHT_PROOF_STATUS,
    ATTEMPT08_PREFLIGHT_SLOTS,
    DEFAULT_PREFLIGHT_PLAN_PATH,
    DEFAULT_SOURCE_PATH,
    attempt08_preflight_seeds,
    canonical_json_bytes,
    load_preflight_plan,
    load_source_row,
)


ATTEMPT08_PREFLIGHT_AGGREGATE_SCHEMA = "hu_m43_attempt08_preflight_aggregate_v1"
ATTEMPT08_PREFLIGHT_AGGREGATE_STATUS = "complete_proof_only"

_PROOF_TOP_KEYS = {
    "schema",
    "status",
    "slot",
    "source",
    "contract",
    "execution",
    "result_proof",
    "science_boundary",
}
_CORRECTNESS_FLAGS = (
    "exact_actionkey_reference_parity_verified",
    "hidden_information_safety_verified",
    "rng_domain_separation_verified",
    "conditional_X_A_skip_contract_verified",
)
_PROOF_EXECUTION_KEYS = {
    "batch_child_selectors",
    "native_batch_threads",
    "run_id",
    "teacher_elapsed_seconds",
    "process_peak_rss_bytes",
    "runtime_fingerprint_sha256",
    "measurement_scope",
}
_PROOF_RESULT_KEYS = {
    "teacher_schema",
    "solver_id",
    "opaque_teacher_sha256",
    "semantic_parity_sha256",
    *_CORRECTNESS_FLAGS,
    "teacher_action_or_value_details_exported",
}
_PROOF_GATE_NAMES = (
    "five_proof_artifact_paths_distinct",
    "all_five_canonical_proofs_valid",
    "source_root_coverage_exact_0_1_2",
    "root0_batch_a_b_exact_teacher_determinism",
    "root0_scalar_batch_semantic_parity",
    "exact_actionkey_reference_parity_each_run",
    "hidden_information_safety_each_run",
    "rng_domain_separation_each_run",
    "conditional_X_A_skip_contract_each_run",
    "runtime_fingerprint_identical_all_five",
)
_OPERATIONAL_GATE_NAMES = (
    "teacher_elapsed_seconds_each_run_max_2400",
    "process_peak_rss_bytes_each_run_max_28GiB",
    "spot_operational_evidence_bound",
)
_OPERATIONAL_KEYS = {
    "science_decision_input",
    "jobs",
    "teacher_elapsed_seconds_max",
    "process_peak_rss_bytes_max",
    "root0_scalar_to_batch_median_speedup_diagnostic",
}
_AGGREGATE_TOP_KEYS = {
    "schema",
    "status",
    "decision",
    "reasons",
    "contract",
    "valid_proof_count",
    "root_coverage",
    "proof_file_sha256",
    "proof_evidence_sha256",
    "runtime_fingerprint_sha256",
    "spot_operational_evidence_sha256",
    "proof_gates",
    "operational_gates",
    "operational_diagnostics",
    "science_boundary",
}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Attempt08 aggregate {label} must be a mapping")
    return value


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Attempt08 aggregate {label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"Attempt08 aggregate {label} must be positive and finite")
    return result


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"Attempt08 aggregate {label} must be a positive integer")
    return value


def _load_canonical_json(path: str | Path, *, label: str) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    payload = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(payload, dict) or raw != canonical_json_bytes(payload):
        raise ValueError(f"Attempt08 {label} is not canonical JSON")
    return payload


def _validate_proof(
    payload: Mapping[str, Any],
    *,
    slot: str,
    source: str | Path,
) -> dict[str, Any]:
    if set(payload) != _PROOF_TOP_KEYS:
        raise ValueError(f"Attempt08 proof {slot} fields changed")
    source_root, batch = ATTEMPT08_PREFLIGHT_SLOTS[slot]
    source_row, observation = load_source_row(source, source_root)
    source_block = _mapping(payload.get("source"), f"{slot}.source")
    contract = _mapping(payload.get("contract"), f"{slot}.contract")
    execution = _mapping(payload.get("execution"), f"{slot}.execution")
    result = _mapping(payload.get("result_proof"), f"{slot}.result_proof")
    science = _mapping(payload.get("science_boundary"), f"{slot}.science")
    expected_source = {
        "merged_sha256": (
            "6b1063589aa2ee4f3e65d9489384176a85c69abe1dfefe30ffb4474f96964903"
        ),
        "source_root_index": source_root,
        "source_row_sha256": _sha256_value(source_row),
        "wrapper_schema": ATTEMPT06_SHARD_ROW_SCHEMA,
        "teacher_schema": ATTEMPT06_TEACHER_SCHEMA,
        "observation_fingerprint": observation.fingerprint(),
        "source_already_consumed": True,
        "new_root_generated": False,
    }
    seeds = attempt08_preflight_seeds(source_root)
    expected_contract = {
        "attempt08_plan_schema": M43_ATTEMPT08_PLAN_SCHEMA,
        "attempt08_plan_sha256": M43_ATTEMPT08_PLAN_SHA256,
        "preflight_plan_schema": ATTEMPT08_PREFLIGHT_PLAN_SCHEMA,
        "preflight_plan_sha256": ATTEMPT08_PREFLIGHT_PLAN_SHA256,
        "model_sha256": ATTEMPT08_LAMBDA_MODEL_SHA256,
        "ai_profiles_sha256": AI_PROFILES_SHA256,
        "runtime_semantic_anchor_sha256": (
            ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256
        ),
        "runtime_source_closure_sha256": ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256,
        "runtime_requirements_sha256": ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
        "gcp_image_name": ATTEMPT08_GCP_IMAGE_NAME,
        "gcp_image_id": ATTEMPT08_GCP_IMAGE_ID,
        "t2_policy_id": "stage9f_p2",
        "t2_resolution": "explicit_profile_never_current",
        "seeds": seeds,
        "continuation_policy_seeds": {
            "first": seeds["child"],
            "second": seeds["child"] + 1,
        },
    }
    expected_science = {
        "proof_only_not_policy_science": True,
        "development200_authorized": False,
        "future_audit_authorized": False,
        "fit_allowed": False,
        "threshold_selection_allowed": False,
        "runtime_activation_allowed": False,
        "current_profile_resolved": False,
        "current_profile_mutated": False,
        "fresh_seed_or_root_opened": False,
    }
    if (
        payload.get("schema") != ATTEMPT08_PREFLIGHT_PROOF_SCHEMA
        or payload.get("status") != ATTEMPT08_PREFLIGHT_PROOF_STATUS
        or payload.get("slot") != slot
        or dict(source_block) != expected_source
        or dict(contract) != expected_contract
        or set(execution) != _PROOF_EXECUTION_KEYS
        or execution.get("batch_child_selectors") is not batch
        or execution.get("native_batch_threads")
        != ATTEMPT08_PREFLIGHT_NATIVE_BATCH_THREADS
        or execution.get("run_id")
        != (
            f"attempt08-preflight:source-root={source_root}:"
            f"obs={observation.fingerprint()}"
        )
        or execution.get("measurement_scope")
        != "elapsed_is_teacher_call_only_rss_is_one_root_process_high_water"
        or execution.get("runtime_fingerprint_sha256")
        != ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256
        or result.get("teacher_schema") != ATTEMPT08_TEACHER_SCHEMA
        or result.get("solver_id") != ATTEMPT08_SOLVER_ID
        or set(result) != _PROOF_RESULT_KEYS
        or result.get("teacher_action_or_value_details_exported") is not False
        or any(result.get(flag) is not True for flag in _CORRECTNESS_FLAGS)
        or not _is_sha256(result.get("opaque_teacher_sha256"))
        or not _is_sha256(result.get("semantic_parity_sha256"))
        or dict(science) != expected_science
    ):
        raise ValueError(f"Attempt08 proof {slot} boundary changed")
    elapsed = _positive_number(
        execution.get("teacher_elapsed_seconds"), f"{slot}.elapsed"
    )
    rss = _positive_int(execution.get("process_peak_rss_bytes"), f"{slot}.rss")
    encoded = json.dumps(payload, sort_keys=True)
    for forbidden in (
        '"selected_action_key"',
        '"raw_paired_deltas',
        '"paired_delta_vs_baseline"',
        '"legal_actions"',
    ):
        if forbidden in encoded:
            raise ValueError(f"Attempt08 proof {slot} leaked teacher details")
    return {
        "root": source_root,
        "batch": batch,
        "opaque": str(result["opaque_teacher_sha256"]),
        "semantic": str(result["semantic_parity_sha256"]),
        "elapsed": elapsed,
        "rss": rss,
        "runtime_fingerprint": str(execution["runtime_fingerprint_sha256"]),
    }


def _proof_evidence_sha256(proof_file_sha256: Mapping[str, Any]) -> str:
    return _sha256_value(dict(proof_file_sha256))


def aggregate_preflight_proofs(
    *,
    root0_batch_a: str | Path,
    root0_batch_b: str | Path,
    root0_scalar: str | Path,
    root1_batch: str | Path,
    root2_batch: str | Path,
    output: str | Path,
    source: str | Path = DEFAULT_SOURCE_PATH,
    preflight_plan: str | Path = DEFAULT_PREFLIGHT_PLAN_PATH,
    spot_operational_evidence_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate five proof slots and publish one value-redacted Go/No-Go."""

    load_preflight_plan(preflight_plan)
    paths = {
        "root0_batch_a": Path(root0_batch_a),
        "root0_batch_b": Path(root0_batch_b),
        "root0_scalar": Path(root0_scalar),
        "root1_batch": Path(root1_batch),
        "root2_batch": Path(root2_batch),
    }
    output_path = Path(output)
    output_identity = os.path.normcase(str(output_path.resolve(strict=False)))
    input_identities = [
        os.path.normcase(str(path.resolve(strict=False))) for path in paths.values()
    ]
    if output_identity in input_identities:
        raise ValueError("Attempt08 aggregate output aliases a proof input")
    if output_path.exists():
        raise FileExistsError(f"Attempt08 aggregate output exists: {output_path}")

    distinct_paths = len(set(input_identities)) == len(input_identities)
    proof_hashes: dict[str, str | None] = {}
    valid: dict[str, dict[str, Any]] = {}
    reasons: list[str] = []
    for slot, path in paths.items():
        try:
            proof_hashes[slot] = _sha256_file(path)
            payload = _load_canonical_json(path, label=f"proof {slot}")
            valid[slot] = _validate_proof(payload, slot=slot, source=source)
        except (OSError, ValueError, json.JSONDecodeError):
            proof_hashes.setdefault(slot, None)
            reasons.append(f"proof_validation_failed:{slot}")
    if not distinct_paths:
        reasons.append("five_proof_artifact_paths_not_distinct")

    all_five = len(valid) == 5
    root_coverage = sorted({int(row["root"]) for row in valid.values()})
    batch_determinism = (
        all_five
        and valid["root0_batch_a"]["opaque"]
        == valid["root0_batch_b"]["opaque"]
    )
    scalar_batch_parity = (
        all_five
        and valid["root0_scalar"]["semantic"]
        == valid["root0_batch_a"]["semantic"]
        == valid["root0_batch_b"]["semantic"]
    )
    runtime_fingerprints = {
        row["runtime_fingerprint"] for row in valid.values()
    }
    runtime_fingerprint_match = all_five and len(runtime_fingerprints) == 1
    runtime_fingerprint = (
        next(iter(runtime_fingerprints)) if runtime_fingerprint_match else None
    )
    correctness_each = all_five
    proof_gates = {
        "five_proof_artifact_paths_distinct": distinct_paths,
        "all_five_canonical_proofs_valid": all_five,
        "source_root_coverage_exact_0_1_2": root_coverage == [0, 1, 2],
        "root0_batch_a_b_exact_teacher_determinism": batch_determinism,
        "root0_scalar_batch_semantic_parity": scalar_batch_parity,
        "exact_actionkey_reference_parity_each_run": correctness_each,
        "hidden_information_safety_each_run": correctness_each,
        "rng_domain_separation_each_run": correctness_each,
        "conditional_X_A_skip_contract_each_run": correctness_each,
        "runtime_fingerprint_identical_all_five": runtime_fingerprint_match,
    }
    if not batch_determinism:
        reasons.append("root0_batch_teacher_determinism_failed")
    if not scalar_batch_parity:
        reasons.append("root0_scalar_batch_semantic_parity_failed")

    jobs = {
        slot: {
            "teacher_elapsed_seconds": float(row["elapsed"]),
            "process_peak_rss_bytes": int(row["rss"]),
        }
        for slot, row in valid.items()
    }
    elapsed_ok = all_five and all(
        row["teacher_elapsed_seconds"] <= 2400.0 for row in jobs.values()
    )
    rss_ok = all_five and all(
        row["process_peak_rss_bytes"] <= 30_064_771_072 for row in jobs.values()
    )
    operational_gates = {
        "teacher_elapsed_seconds_each_run_max_2400": elapsed_ok,
        "process_peak_rss_bytes_each_run_max_28GiB": rss_ok,
        "spot_operational_evidence_bound": _is_sha256(
            spot_operational_evidence_sha256
        ),
    }
    if not elapsed_ok:
        reasons.append("teacher_elapsed_limit_failed")
    if not rss_ok:
        reasons.append("process_peak_rss_limit_failed")
    speedup: float | None = None
    if all_five:
        batch_median = statistics.median(
            (
                jobs["root0_batch_a"]["teacher_elapsed_seconds"],
                jobs["root0_batch_b"]["teacher_elapsed_seconds"],
            )
        )
        speedup = jobs["root0_scalar"]["teacher_elapsed_seconds"] / batch_median
    operational = {
        "science_decision_input": False,
        "jobs": jobs,
        "teacher_elapsed_seconds_max": max(
            (row["teacher_elapsed_seconds"] for row in jobs.values()), default=0.0
        ),
        "process_peak_rss_bytes_max": max(
            (row["process_peak_rss_bytes"] for row in jobs.values()), default=0
        ),
        "root0_scalar_to_batch_median_speedup_diagnostic": speedup,
    }
    decision = (
        "go"
        if all(proof_gates.values()) and all(operational_gates.values())
        else "no_go"
    )
    if decision == "go":
        reasons = ["all_correctness_and_operational_preflight_gates_passed"]
    else:
        reasons = sorted([
            *(f"gate_failed:{name}" for name, passed in proof_gates.items() if not passed),
            *(
                f"gate_failed:{name}"
                for name, passed in operational_gates.items()
                if not passed
            ),
        ])
    report = {
        "schema": ATTEMPT08_PREFLIGHT_AGGREGATE_SCHEMA,
        "status": ATTEMPT08_PREFLIGHT_AGGREGATE_STATUS,
        "decision": decision,
        "reasons": reasons,
        "contract": {
            "attempt08_plan_schema": M43_ATTEMPT08_PLAN_SCHEMA,
            "attempt08_plan_sha256": M43_ATTEMPT08_PLAN_SHA256,
            "preflight_plan_schema": ATTEMPT08_PREFLIGHT_PLAN_SCHEMA,
            "preflight_plan_sha256": ATTEMPT08_PREFLIGHT_PLAN_SHA256,
            "teacher_schema": ATTEMPT08_TEACHER_SCHEMA,
            "model_sha256": ATTEMPT08_LAMBDA_MODEL_SHA256,
            "ai_profiles_sha256": AI_PROFILES_SHA256,
            "runtime_semantic_anchor_sha256": (
                ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256
            ),
            "runtime_source_closure_sha256": (
                ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256
            ),
            "runtime_requirements_sha256": ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
            "gcp_image_name": ATTEMPT08_GCP_IMAGE_NAME,
            "gcp_image_id": ATTEMPT08_GCP_IMAGE_ID,
        },
        "valid_proof_count": len(valid),
        "root_coverage": root_coverage,
        "proof_file_sha256": proof_hashes,
        "proof_evidence_sha256": _proof_evidence_sha256(proof_hashes),
        "runtime_fingerprint_sha256": runtime_fingerprint,
        "spot_operational_evidence_sha256": spot_operational_evidence_sha256,
        "proof_gates": proof_gates,
        "operational_gates": operational_gates,
        "operational_diagnostics": operational,
        "science_boundary": {
            "proof_only_not_policy_science": True,
            "opaque_teacher_hashes_exported": False,
            "teacher_action_or_value_details_exported": False,
            "development200_authorized": False,
            "future_audit_authorized": False,
            "fit_allowed": False,
            "threshold_selection_allowed": False,
            "runtime_activation_allowed": False,
            "current_profile_changed": False,
        },
    }
    validate_preflight_aggregate(report)
    _atomic_write_once(output_path, canonical_json_bytes(report))
    return report


def validate_preflight_aggregate(payload: Mapping[str, Any]) -> None:
    """Fail closed on aggregate truncation, inconsistency, or value leakage."""

    if set(payload) != _AGGREGATE_TOP_KEYS:
        raise ValueError("Attempt08 aggregate fields changed")
    contract = _mapping(payload.get("contract"), "contract")
    proof_hashes = _mapping(payload.get("proof_file_sha256"), "proof hashes")
    proof_gates = _mapping(payload.get("proof_gates"), "proof gates")
    operational_gates = _mapping(payload.get("operational_gates"), "operational gates")
    operational = _mapping(payload.get("operational_diagnostics"), "operations")
    science = _mapping(payload.get("science_boundary"), "science")
    decision = payload.get("decision")
    reasons = payload.get("reasons")
    runtime_fingerprint = payload.get("runtime_fingerprint_sha256")
    spot_evidence_sha256 = payload.get("spot_operational_evidence_sha256")
    expected_contract = {
        "attempt08_plan_schema": M43_ATTEMPT08_PLAN_SCHEMA,
        "attempt08_plan_sha256": M43_ATTEMPT08_PLAN_SHA256,
        "preflight_plan_schema": ATTEMPT08_PREFLIGHT_PLAN_SCHEMA,
        "preflight_plan_sha256": ATTEMPT08_PREFLIGHT_PLAN_SHA256,
        "teacher_schema": ATTEMPT08_TEACHER_SCHEMA,
        "model_sha256": ATTEMPT08_LAMBDA_MODEL_SHA256,
        "ai_profiles_sha256": AI_PROFILES_SHA256,
        "runtime_semantic_anchor_sha256": (
            ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256
        ),
        "runtime_source_closure_sha256": ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256,
        "runtime_requirements_sha256": ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
        "gcp_image_name": ATTEMPT08_GCP_IMAGE_NAME,
        "gcp_image_id": ATTEMPT08_GCP_IMAGE_ID,
    }
    expected_science = {
        "proof_only_not_policy_science": True,
        "opaque_teacher_hashes_exported": False,
        "teacher_action_or_value_details_exported": False,
        "development200_authorized": False,
        "future_audit_authorized": False,
        "fit_allowed": False,
        "threshold_selection_allowed": False,
        "runtime_activation_allowed": False,
        "current_profile_changed": False,
    }
    valid_count = payload.get("valid_proof_count")
    root_coverage = payload.get("root_coverage")
    if type(valid_count) is not int or not 0 <= valid_count <= 5:
        raise ValueError("Attempt08 aggregate valid proof count changed")
    if (
        not isinstance(root_coverage, list)
        or any(type(value) is not int for value in root_coverage)
        or root_coverage != sorted(set(root_coverage))
        or not set(root_coverage).issubset({0, 1, 2})
    ):
        raise ValueError("Attempt08 aggregate root coverage changed")
    if (
        payload.get("schema") != ATTEMPT08_PREFLIGHT_AGGREGATE_SCHEMA
        or payload.get("status") != ATTEMPT08_PREFLIGHT_AGGREGATE_STATUS
        or decision not in {"go", "no_go"}
        or not isinstance(reasons, list)
        or not all(isinstance(reason, str) and reason for reason in reasons)
        or dict(contract) != expected_contract
        or set(proof_hashes) != set(ATTEMPT08_PREFLIGHT_SLOTS)
        or not all(value is None or _is_sha256(value) for value in proof_hashes.values())
        or payload.get("proof_evidence_sha256")
        != _proof_evidence_sha256(dict(proof_hashes))
        or runtime_fingerprint
        not in {None, ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256}
        or (spot_evidence_sha256 is not None and not _is_sha256(spot_evidence_sha256))
        or set(proof_gates) != set(_PROOF_GATE_NAMES)
        or not all(type(value) is bool for value in proof_gates.values())
        or set(operational_gates) != set(_OPERATIONAL_GATE_NAMES)
        or not all(type(value) is bool for value in operational_gates.values())
        or set(operational) != _OPERATIONAL_KEYS
        or operational.get("science_decision_input") is not False
        or dict(science) != expected_science
    ):
        raise ValueError("Attempt08 aggregate boundary changed")
    jobs = _mapping(operational.get("jobs"), "operational jobs")
    if not set(jobs).issubset(ATTEMPT08_PREFLIGHT_SLOTS):
        raise ValueError("Attempt08 aggregate operational job slots changed")
    normalized_jobs: dict[str, dict[str, float | int]] = {}
    for slot, raw_job in jobs.items():
        job = _mapping(raw_job, f"operational job {slot}")
        if set(job) != {"teacher_elapsed_seconds", "process_peak_rss_bytes"}:
            raise ValueError("Attempt08 aggregate operational job fields changed")
        normalized_jobs[str(slot)] = {
            "teacher_elapsed_seconds": _positive_number(
                job.get("teacher_elapsed_seconds"), f"{slot}.elapsed"
            ),
            "process_peak_rss_bytes": _positive_int(
                job.get("process_peak_rss_bytes"), f"{slot}.rss"
            ),
        }
    elapsed_max = max(
        (float(job["teacher_elapsed_seconds"]) for job in normalized_jobs.values()),
        default=0.0,
    )
    rss_max = max(
        (int(job["process_peak_rss_bytes"]) for job in normalized_jobs.values()),
        default=0,
    )
    all_jobs = set(normalized_jobs) == set(ATTEMPT08_PREFLIGHT_SLOTS)
    expected_root_coverage = sorted(
        {
            ATTEMPT08_PREFLIGHT_SLOTS[slot][0]
            for slot in normalized_jobs
        }
    )
    if valid_count != len(normalized_jobs) or root_coverage != expected_root_coverage:
        raise ValueError("Attempt08 aggregate valid proof identity changed")
    expected_speedup: float | None = None
    if all_jobs:
        batch_median = statistics.median(
            (
                float(normalized_jobs["root0_batch_a"]["teacher_elapsed_seconds"]),
                float(normalized_jobs["root0_batch_b"]["teacher_elapsed_seconds"]),
            )
        )
        expected_speedup = (
            float(normalized_jobs["root0_scalar"]["teacher_elapsed_seconds"])
            / batch_median
        )
    if (
        operational.get("teacher_elapsed_seconds_max") != elapsed_max
        or operational.get("process_peak_rss_bytes_max") != rss_max
        or operational.get("root0_scalar_to_batch_median_speedup_diagnostic")
        != expected_speedup
    ):
        raise ValueError("Attempt08 aggregate operational summaries changed")
    expected_operational_gates = {
        "teacher_elapsed_seconds_each_run_max_2400": all_jobs
        and all(
            float(job["teacher_elapsed_seconds"]) <= 2400.0
            for job in normalized_jobs.values()
        ),
        "process_peak_rss_bytes_each_run_max_28GiB": all_jobs
        and all(
            int(job["process_peak_rss_bytes"]) <= 30_064_771_072
            for job in normalized_jobs.values()
        ),
        "spot_operational_evidence_bound": _is_sha256(spot_evidence_sha256),
    }
    if dict(operational_gates) != expected_operational_gates:
        raise ValueError("Attempt08 aggregate operational gates were not recomputed")
    all_valid = valid_count == 5
    if (
        proof_gates["all_five_canonical_proofs_valid"] is not all_valid
        or proof_gates["source_root_coverage_exact_0_1_2"]
        is not (root_coverage == [0, 1, 2])
        or any(
            proof_gates[name] is not all_valid
            for name in (
                "exact_actionkey_reference_parity_each_run",
                "hidden_information_safety_each_run",
                "rng_domain_separation_each_run",
                "conditional_X_A_skip_contract_each_run",
            )
        )
    ):
        raise ValueError("Attempt08 aggregate proof gates were not recomputed")
    if proof_gates["runtime_fingerprint_identical_all_five"] is not (
        all_valid
        and runtime_fingerprint == ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256
    ):
        raise ValueError("Attempt08 aggregate runtime fingerprint gate changed")
    expected_go = all(proof_gates.values()) and all(operational_gates.values())
    if decision == "go" and (
        valid_count != 5
        or root_coverage != [0, 1, 2]
        or not all(_is_sha256(value) for value in proof_hashes.values())
        or not all_jobs
        or runtime_fingerprint != ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256
        or not _is_sha256(spot_evidence_sha256)
    ):
        raise ValueError("Attempt08 aggregate Go lacks exact five-proof evidence")
    if (decision == "go") is not expected_go:
        raise ValueError("Attempt08 aggregate decision disagrees with gates")
    expected_reasons = (
        ["all_correctness_and_operational_preflight_gates_passed"]
        if decision == "go"
        else sorted([
            *(f"gate_failed:{name}" for name, passed in proof_gates.items() if not passed),
            *(
                f"gate_failed:{name}"
                for name, passed in operational_gates.items()
                if not passed
            ),
        ])
    )
    if reasons != expected_reasons:
        raise ValueError("Attempt08 aggregate reasons disagree with gates")
    encoded = json.dumps(payload, sort_keys=True)
    for forbidden in (
        '"opaque_teacher_sha256"',
        '"semantic_parity_sha256"',
        '"selected_action_key"',
        '"raw_paired_deltas',
        '"paired_delta_vs_baseline"',
        '"legal_actions"',
    ):
        if forbidden in encoded:
            raise ValueError("Attempt08 aggregate leaked teacher details")


def validate_preflight_aggregate_with_proofs(
    payload: Mapping[str, Any],
    *,
    proof_paths: Mapping[str, str | Path],
    source: str | Path = DEFAULT_SOURCE_PATH,
) -> None:
    """Rebuild every Go-relevant gate from the five producer proofs.

    The value-redacted aggregate intentionally does not carry the opaque or
    semantic teacher digests.  Consequently an aggregate by itself cannot
    prove root-0 determinism or scalar/batch parity.  Authorization consumers
    must call this validator with the exact five proof artifacts rather than
    trusting mutable aggregate booleans.
    """

    validate_preflight_aggregate(payload)
    spot_evidence_sha256 = payload.get("spot_operational_evidence_sha256")
    expected_slots = tuple(ATTEMPT08_PREFLIGHT_SLOTS)
    if set(proof_paths) != set(expected_slots):
        raise ValueError("Attempt08 aggregate requires the exact five proof paths")
    normalized = {slot: Path(proof_paths[slot]).resolve() for slot in expected_slots}
    identities = [os.path.normcase(str(normalized[slot])) for slot in expected_slots]
    if len(set(identities)) != len(expected_slots):
        raise ValueError("Attempt08 aggregate proof paths must be distinct")

    proof_hashes = _mapping(payload.get("proof_file_sha256"), "proof hashes")
    validated: dict[str, dict[str, Any]] = {}
    for slot in expected_slots:
        path = normalized[slot]
        actual_sha256 = _sha256_file(path)
        if proof_hashes.get(slot) != actual_sha256:
            raise ValueError(f"Attempt08 aggregate proof hash changed: {slot}")
        proof = _load_canonical_json(path, label=f"authorization proof {slot}")
        validated[slot] = _validate_proof(proof, slot=slot, source=source)

    root_coverage = sorted({int(row["root"]) for row in validated.values()})
    determinism = (
        validated["root0_batch_a"]["opaque"]
        == validated["root0_batch_b"]["opaque"]
    )
    scalar_batch_parity = (
        validated["root0_scalar"]["semantic"]
        == validated["root0_batch_a"]["semantic"]
        == validated["root0_batch_b"]["semantic"]
    )
    proof_runtime_fingerprints = {
        row["runtime_fingerprint"] for row in validated.values()
    }
    runtime_fingerprint_match = len(proof_runtime_fingerprints) == 1
    runtime_fingerprint = (
        next(iter(proof_runtime_fingerprints))
        if runtime_fingerprint_match
        else None
    )
    expected_proof_gates = {
        "five_proof_artifact_paths_distinct": True,
        "all_five_canonical_proofs_valid": True,
        "source_root_coverage_exact_0_1_2": root_coverage == [0, 1, 2],
        "root0_batch_a_b_exact_teacher_determinism": determinism,
        "root0_scalar_batch_semantic_parity": scalar_batch_parity,
        "exact_actionkey_reference_parity_each_run": True,
        "hidden_information_safety_each_run": True,
        "rng_domain_separation_each_run": True,
        "conditional_X_A_skip_contract_each_run": True,
        "runtime_fingerprint_identical_all_five": runtime_fingerprint_match,
    }
    jobs = {
        slot: {
            "teacher_elapsed_seconds": float(row["elapsed"]),
            "process_peak_rss_bytes": int(row["rss"]),
        }
        for slot, row in validated.items()
    }
    expected_operational_gates = {
        "teacher_elapsed_seconds_each_run_max_2400": all(
            row["teacher_elapsed_seconds"] <= 2400.0 for row in jobs.values()
        ),
        "process_peak_rss_bytes_each_run_max_28GiB": all(
            row["process_peak_rss_bytes"] <= 30_064_771_072
            for row in jobs.values()
        ),
        "spot_operational_evidence_bound": _is_sha256(spot_evidence_sha256),
    }
    batch_median = statistics.median(
        (
            jobs["root0_batch_a"]["teacher_elapsed_seconds"],
            jobs["root0_batch_b"]["teacher_elapsed_seconds"],
        )
    )
    expected_operational = {
        "science_decision_input": False,
        "jobs": jobs,
        "teacher_elapsed_seconds_max": max(
            row["teacher_elapsed_seconds"] for row in jobs.values()
        ),
        "process_peak_rss_bytes_max": max(
            row["process_peak_rss_bytes"] for row in jobs.values()
        ),
        "root0_scalar_to_batch_median_speedup_diagnostic": (
            jobs["root0_scalar"]["teacher_elapsed_seconds"] / batch_median
        ),
    }
    expected_decision = (
        "go"
        if all(expected_proof_gates.values())
        and all(expected_operational_gates.values())
        else "no_go"
    )
    expected_reasons = (
        ["all_correctness_and_operational_preflight_gates_passed"]
        if expected_decision == "go"
        else sorted(
            [
                *(
                    f"gate_failed:{name}"
                    for name, passed in expected_proof_gates.items()
                    if not passed
                ),
                *(
                    f"gate_failed:{name}"
                    for name, passed in expected_operational_gates.items()
                    if not passed
                ),
            ]
        )
    )
    if (
        payload.get("valid_proof_count") != 5
        or payload.get("root_coverage") != root_coverage
        or payload.get("proof_evidence_sha256")
        != _proof_evidence_sha256(dict(proof_hashes))
        or payload.get("runtime_fingerprint_sha256") != runtime_fingerprint
        or dict(_mapping(payload.get("proof_gates"), "proof gates"))
        != expected_proof_gates
        or dict(_mapping(payload.get("operational_gates"), "operational gates"))
        != expected_operational_gates
        or dict(_mapping(payload.get("operational_diagnostics"), "operations"))
        != expected_operational
        or payload.get("decision") != expected_decision
        or payload.get("reasons") != expected_reasons
    ):
        raise ValueError("Attempt08 aggregate disagrees with producer proof evidence")


def load_and_validate_preflight_aggregate(path: str | Path) -> dict[str, Any]:
    payload = _load_canonical_json(path, label="aggregate")
    validate_preflight_aggregate(payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for slot in ATTEMPT08_PREFLIGHT_SLOTS:
        parser.add_argument(f"--{slot.replace('_', '-')}", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE_PATH))
    parser.add_argument("--preflight-plan", default=str(DEFAULT_PREFLIGHT_PLAN_PATH))
    parser.add_argument("--spot-operational-evidence-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    aggregate_preflight_proofs(
        root0_batch_a=args.root0_batch_a,
        root0_batch_b=args.root0_batch_b,
        root0_scalar=args.root0_scalar,
        root1_batch=args.root1_batch,
        root2_batch=args.root2_batch,
        output=args.output,
        source=args.source,
        preflight_plan=args.preflight_plan,
        spot_operational_evidence_sha256=args.spot_operational_evidence_sha256,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ATTEMPT08_PREFLIGHT_AGGREGATE_SCHEMA",
    "ATTEMPT08_PREFLIGHT_AGGREGATE_STATUS",
    "aggregate_preflight_proofs",
    "load_and_validate_preflight_aggregate",
    "validate_preflight_aggregate",
    "validate_preflight_aggregate_with_proofs",
]
