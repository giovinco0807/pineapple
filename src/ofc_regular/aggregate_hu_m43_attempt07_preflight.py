"""Validate and aggregate the five bounded Attempt07 preflight proofs.

The aggregator never evaluates a poker state.  It validates canonical,
value-redacted proof rows for root0 batch A/B/scalar and roots1/2 batch,
checks within-mode determinism and scalar/batch semantic parity, and emits a
single Go/No-Go receipt.  Optional external DONE metadata is projected into a
separate operational section containing only elapsed time and peak RSS; it
cannot affect the scientific decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hu_m43_attempt06_teacher import (
    ATTEMPT06_FROZEN_MODEL_SHA256,
    ATTEMPT06_SHARD_ROW_SCHEMA,
    ATTEMPT06_T2_POLICY_ID,
    ATTEMPT06_TEACHER_SCHEMA,
)
from .hu_m43_attempt07_contract import (
    AI_PROFILES_SHA256,
    M43_ATTEMPT07_PLAN_SHA256,
    load_and_validate_attempt07_plan,
)
from .hu_m43_attempt07_teacher import (
    ATTEMPT07_SOLVER_ID,
    ATTEMPT07_TEACHER_SCHEMA,
)
from .run_hu_m43_attempt07_preflight import (
    ATTEMPT06_SOURCE_PLAN_SHA256,
    ATTEMPT07_PREFLIGHT_NATIVE_BATCH_THREADS,
    ATTEMPT07_PREFLIGHT_PLAN_SCHEMA,
    ATTEMPT07_PREFLIGHT_ROW_SCHEMA,
    ATTEMPT07_PREFLIGHT_SOURCE_SHA256,
    ATTEMPT07_PREFLIGHT_STATUS,
    DEFAULT_ATTEMPT07_PLAN_PATH,
    DEFAULT_PREFLIGHT_PLAN_PATH,
    DEFAULT_SOURCE_PATH,
    _sha256_value,
    attempt07_preflight_seeds,
    canonical_json_bytes,
    load_preflight_plan,
    load_source_row,
)


ATTEMPT07_PREFLIGHT_AGGREGATE_SCHEMA = (
    "hu_m43_attempt07_preflight_proof_aggregate_v1"
)

_SLOT_SPECS: tuple[tuple[str, int, bool], ...] = (
    ("root0_batch_a", 0, True),
    ("root0_batch_b", 0, True),
    ("root0_scalar", 0, False),
    ("root1_batch", 1, True),
    ("root2_batch", 2, True),
)
_SLOT_NAMES = tuple(name for name, _, _ in _SLOT_SPECS)
_SHA256_CHARS = frozenset("0123456789abcdef")
_RESULT_PROOF_KEYS = frozenset(
    {
        "teacher_schema",
        "solver_id",
        "opaque_teacher_sha256",
        "opaque_teacher_sha256_purpose",
        "semantic_parity_sha256",
        "semantic_parity_sha256_purpose",
        "teacher_values_exported",
        "arm_details_exported",
    }
)
_EXPECTED_SCIENCE_BOUNDARY = {
    "arm_selection_allowed": False,
    "fit_allowed": False,
    "threshold_selection_allowed": False,
    "runtime_activation_allowed": False,
    "current_profile_resolved": False,
    "current_profile_mutated": False,
    "fresh_seed_or_root_opened": False,
}

_EXPECTED_AGGREGATE_SCIENCE_BOUNDARY = {
    "proof_only_no_teacher_values_or_arm_details": True,
    "arm_selection_allowed": False,
    "fit_allowed": False,
    "threshold_selection_allowed": False,
    "runtime_activation_allowed": False,
    "current_profile_resolved": False,
    "current_profile_mutated": False,
    "fresh_seed_or_root_opened": False,
    "done_metadata_is_science_input": False,
}

_PROOF_GATE_NAMES = frozenset(
    {
        "frozen_context_valid",
        "five_proof_artifact_paths_distinct",
        "all_five_canonical_proofs_valid",
        "exact_root_and_mode_coverage",
        "root0_batch_a_b_canonical_rows_identical",
        "root0_batch_opaque_mode_determinism",
        "root0_scalar_batch_semantic_parity",
        "three_source_roots_covered",
        "teacher_values_absent",
        "arm_details_absent",
        "current_profile_unchanged",
    }
)

_AGGREGATE_KEYS = frozenset(
    {
        "schema",
        "status",
        "decision",
        "reasons",
        "proof_input_count",
        "valid_proof_count",
        "expected_slots",
        "root_coverage",
        "proof_gates",
        "proof_file_sha256",
        "cross_mode_opaque_teacher_hash_compared",
        "science_boundary",
        "contract",
        "operational_diagnostics",
    }
)


@dataclass(frozen=True)
class _ValidatedProof:
    row: dict[str, Any]
    canonical_bytes: bytes
    file_sha256: str


@dataclass(frozen=True)
class _ExpectedSource:
    source_row_sha256: str
    observation_fingerprint: str
    baseline_action_key: str


def _require_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise ValueError(f"Attempt07 preflight {field} is not a SHA-256")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_canonical_proof(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Attempt07 preflight proof is not canonical JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Attempt07 preflight proof must be a JSON mapping")
    canonical = canonical_json_bytes(payload)
    if raw != canonical:
        raise ValueError("Attempt07 preflight proof bytes are not canonical")
    return payload, canonical


def _expected_sources(source: Path) -> dict[int, _ExpectedSource]:
    expected: dict[int, _ExpectedSource] = {}
    for root_index in (0, 1, 2):
        row, observation = load_source_row(source, root_index)
        expected[root_index] = _ExpectedSource(
            source_row_sha256=_sha256_value(row),
            observation_fingerprint=observation.fingerprint(),
            baseline_action_key=str(row["baseline_action_key"]),
        )
    return expected


def _validate_proof_row(
    path: Path,
    *,
    expected_root: int,
    expected_batch: bool,
    expected_source: _ExpectedSource,
    preflight_plan_sha256: str,
) -> _ValidatedProof:
    row, canonical = _load_canonical_proof(path)
    if set(row) != {
        "schema",
        "status",
        "source",
        "contract",
        "execution",
        "result_proof",
        "science_boundary",
    }:
        raise ValueError("Attempt07 preflight proof top-level fields changed")
    source = row.get("source")
    contract = row.get("contract")
    execution = row.get("execution")
    result = row.get("result_proof")
    science = row.get("science_boundary")
    if not all(
        isinstance(value, Mapping)
        for value in (source, contract, execution, result, science)
    ):
        raise ValueError("Attempt07 preflight proof sections are incomplete")
    if set(source) != {
        "merged_sha256",
        "source_root_index",
        "source_row_sha256",
        "wrapper_schema",
        "teacher_schema",
        "observation_fingerprint",
        "baseline_action_key",
        "new_root_generated",
    }:
        raise ValueError("Attempt07 preflight source proof fields changed")
    if set(contract) != {
        "plan_sha256",
        "preflight_plan_sha256",
        "model_sha256",
        "ai_profiles_sha256",
        "t2_policy_id",
        "t2_resolution",
        "seeds",
        "continuation_policy_seeds",
    }:
        raise ValueError("Attempt07 preflight contract proof fields changed")
    if set(execution) != {
        "batch_child_selectors",
        "native_batch_threads",
        "run_id",
    }:
        raise ValueError("Attempt07 preflight execution proof fields changed")
    if set(result) != _RESULT_PROOF_KEYS:
        raise ValueError("Attempt07 preflight result proof fields changed")
    if dict(science) != _EXPECTED_SCIENCE_BOUNDARY:
        raise ValueError("Attempt07 preflight science boundary changed")

    seeds = attempt07_preflight_seeds(expected_root)
    expected_run_id = (
        f"attempt07-preflight:source-root={expected_root}:"
        f"obs={expected_source.observation_fingerprint}"
    )
    expected_source_fields = {
        "merged_sha256": ATTEMPT07_PREFLIGHT_SOURCE_SHA256,
        "source_root_index": expected_root,
        "source_row_sha256": expected_source.source_row_sha256,
        "wrapper_schema": ATTEMPT06_SHARD_ROW_SCHEMA,
        "teacher_schema": ATTEMPT06_TEACHER_SCHEMA,
        "observation_fingerprint": expected_source.observation_fingerprint,
        "baseline_action_key": expected_source.baseline_action_key,
        "new_root_generated": False,
    }
    expected_contract = {
        "plan_sha256": M43_ATTEMPT07_PLAN_SHA256,
        "preflight_plan_sha256": preflight_plan_sha256,
        "model_sha256": ATTEMPT06_FROZEN_MODEL_SHA256,
        "ai_profiles_sha256": AI_PROFILES_SHA256,
        "t2_policy_id": ATTEMPT06_T2_POLICY_ID,
        "t2_resolution": "explicit_profile_never_current",
        "seeds": seeds,
        "continuation_policy_seeds": {
            "first": seeds["child"],
            "second": seeds["child"] + 1,
        },
    }
    expected_execution = {
        "batch_child_selectors": expected_batch,
        "native_batch_threads": ATTEMPT07_PREFLIGHT_NATIVE_BATCH_THREADS,
        "run_id": expected_run_id,
    }
    expected_result_fixed = {
        "teacher_schema": ATTEMPT07_TEACHER_SCHEMA,
        "solver_id": ATTEMPT07_SOLVER_ID,
        "opaque_teacher_sha256_purpose": (
            "determinism_only_not_arm_selection"
        ),
        "semantic_parity_sha256_purpose": (
            "scalar_batch_exact_parity_after_normalizing_only_"
            "batch_execution_metadata"
        ),
        "teacher_values_exported": False,
        "arm_details_exported": False,
    }
    if (
        row.get("schema") != ATTEMPT07_PREFLIGHT_ROW_SCHEMA
        or row.get("status") != ATTEMPT07_PREFLIGHT_STATUS
        or dict(source) != expected_source_fields
        or dict(contract) != expected_contract
        or dict(execution) != expected_execution
        or any(result.get(key) != value for key, value in expected_result_fixed.items())
    ):
        raise ValueError("Attempt07 preflight proof identity or contract changed")
    _require_sha256(
        result.get("opaque_teacher_sha256"), field="opaque_teacher_sha256"
    )
    _require_sha256(
        result.get("semantic_parity_sha256"),
        field="semantic_parity_sha256",
    )
    return _ValidatedProof(
        row=dict(row),
        canonical_bytes=canonical,
        file_sha256=hashlib.sha256(canonical).hexdigest(),
    )


def _atomic_write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            if path.read_bytes() != payload:
                raise FileExistsError(
                    f"Attempt07 preflight aggregate already differs: {path}"
                ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _number(value: Any, *, field: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"DONE {field} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"DONE {field} must be finite and non-negative")
    return value


def _operational_diagnostics(
    done_metadata: Mapping[str, str | Path] | None,
) -> dict[str, Any]:
    if not done_metadata:
        return {
            "status": "not_provided",
            "science_decision_input": False,
            "allowed_fields": ["elapsed_seconds", "peak_rss_bytes"],
            "jobs": {},
        }
    diagnostics: dict[str, dict[str, float | int]] = {}
    invalid: list[str] = []
    for label, raw_path in done_metadata.items():
        if label not in _SLOT_NAMES:
            invalid.append(label)
            continue
        try:
            payload = json.loads(Path(raw_path).read_text(encoding="utf-8-sig"))
            if not isinstance(payload, Mapping):
                raise ValueError("DONE metadata must be a mapping")
            elapsed = _number(payload.get("elapsed_seconds"), field="elapsed_seconds")
            peak_rss = _number(payload.get("peak_rss_bytes"), field="peak_rss_bytes")
            diagnostics[label] = {
                "elapsed_seconds": elapsed,
                "peak_rss_bytes": peak_rss,
            }
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            invalid.append(label)
    elapsed_values = [float(row["elapsed_seconds"]) for row in diagnostics.values()]
    rss_values = [int(row["peak_rss_bytes"]) for row in diagnostics.values()]
    return {
        "status": "ok" if not invalid else "partial_invalid_not_used_for_science",
        "science_decision_input": False,
        "allowed_fields": ["elapsed_seconds", "peak_rss_bytes"],
        "job_count": len(diagnostics),
        "invalid_labels": sorted(set(invalid)),
        "jobs": {key: diagnostics[key] for key in sorted(diagnostics)},
        "elapsed_seconds_sum": sum(elapsed_values),
        "elapsed_seconds_max": max(elapsed_values, default=0.0),
        "peak_rss_bytes_max": max(rss_values, default=0),
    }


def _proof_paths_are_distinct(paths: Mapping[str, Path]) -> bool:
    identities = [
        os.path.normcase(str(path.resolve(strict=False))) for path in paths.values()
    ]
    if len(set(identities)) != len(identities):
        return False
    items = tuple(paths.items())
    for left_index, (_, left) in enumerate(items):
        if not left.exists():
            continue
        for _, right in items[left_index + 1 :]:
            if right.exists():
                try:
                    if os.path.samefile(left, right):
                        return False
                except OSError:
                    return False
    return True


def validate_preflight_go_aggregate(
    payload: Mapping[str, Any],
    *,
    preflight_plan_sha256: str,
    require_operational_jobs: bool = True,
) -> None:
    """Fail closed unless *payload* is the exact frozen Go aggregate shape.

    This validates the aggregate receipt itself, not merely a caller-provided
    subset of truthy gates.  The five proof files and DONE records still need
    to be hash-bound independently by the consumer.
    """

    if set(payload) != _AGGREGATE_KEYS:
        raise ValueError("Attempt07 aggregate top-level fields changed")
    if (
        payload.get("schema") != ATTEMPT07_PREFLIGHT_AGGREGATE_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("decision") != "go"
        or payload.get("reasons") != ["all_preflight_proof_gates_passed"]
        or type(payload.get("proof_input_count")) is not int
        or payload.get("proof_input_count") != len(_SLOT_SPECS)
        or type(payload.get("valid_proof_count")) is not int
        or payload.get("valid_proof_count") != len(_SLOT_SPECS)
        or payload.get("expected_slots") != list(_SLOT_NAMES)
        or payload.get("root_coverage") != [0, 1, 2]
        or payload.get("cross_mode_opaque_teacher_hash_compared") is not False
    ):
        raise ValueError("Attempt07 aggregate identity is not the frozen Go receipt")

    proof_gates = payload.get("proof_gates")
    if (
        not isinstance(proof_gates, Mapping)
        or set(proof_gates) != _PROOF_GATE_NAMES
        or any(type(value) is not bool or value is not True for value in proof_gates.values())
    ):
        raise ValueError("Attempt07 aggregate proof gates changed")

    proof_hashes = payload.get("proof_file_sha256")
    if not isinstance(proof_hashes, Mapping) or set(proof_hashes) != set(_SLOT_NAMES):
        raise ValueError("Attempt07 aggregate proof hashes changed")
    for slot in _SLOT_NAMES:
        _require_sha256(proof_hashes[slot], field=f"proof_file_sha256.{slot}")
    if proof_hashes["root0_batch_a"] != proof_hashes["root0_batch_b"]:
        raise ValueError("Attempt07 aggregate batch determinism hashes changed")

    science = payload.get("science_boundary")
    if not isinstance(science, Mapping) or dict(science) != _EXPECTED_AGGREGATE_SCIENCE_BOUNDARY:
        raise ValueError("Attempt07 aggregate science boundary changed")

    expected_contract = {
        "preflight_plan_schema": ATTEMPT07_PREFLIGHT_PLAN_SCHEMA,
        "preflight_plan_sha256": preflight_plan_sha256,
        "attempt06_source_plan_sha256": ATTEMPT06_SOURCE_PLAN_SHA256,
        "attempt06_merged_source_sha256": ATTEMPT07_PREFLIGHT_SOURCE_SHA256,
        "attempt06_wrapper_schema": ATTEMPT06_SHARD_ROW_SCHEMA,
        "attempt06_teacher_schema": ATTEMPT06_TEACHER_SCHEMA,
        "attempt07_plan_sha256": M43_ATTEMPT07_PLAN_SHA256,
        "attempt07_teacher_schema": ATTEMPT07_TEACHER_SCHEMA,
        "model_sha256": ATTEMPT06_FROZEN_MODEL_SHA256,
        "ai_profiles_sha256": AI_PROFILES_SHA256,
    }
    contract = payload.get("contract")
    if not isinstance(contract, Mapping) or dict(contract) != expected_contract:
        raise ValueError("Attempt07 aggregate contract changed")

    operational = payload.get("operational_diagnostics")
    if not isinstance(operational, Mapping):
        raise ValueError("Attempt07 aggregate operational diagnostics are missing")
    if require_operational_jobs:
        expected_keys = {
            "status",
            "science_decision_input",
            "allowed_fields",
            "job_count",
            "invalid_labels",
            "jobs",
            "elapsed_seconds_sum",
            "elapsed_seconds_max",
            "peak_rss_bytes_max",
        }
        if (
            set(operational) != expected_keys
            or operational.get("status") != "ok"
            or operational.get("science_decision_input") is not False
            or operational.get("allowed_fields")
            != ["elapsed_seconds", "peak_rss_bytes"]
            or type(operational.get("job_count")) is not int
            or operational.get("job_count") != len(_SLOT_SPECS)
            or operational.get("invalid_labels") != []
        ):
            raise ValueError("Attempt07 aggregate operational identity changed")
        jobs = operational.get("jobs")
        if not isinstance(jobs, Mapping) or set(jobs) != set(_SLOT_NAMES):
            raise ValueError("Attempt07 aggregate operational jobs changed")
        elapsed_values: list[float] = []
        rss_values: list[int] = []
        for slot in _SLOT_NAMES:
            job = jobs[slot]
            if not isinstance(job, Mapping) or set(job) != {
                "elapsed_seconds",
                "peak_rss_bytes",
            }:
                raise ValueError(f"Attempt07 aggregate operational job changed: {slot}")
            elapsed = _number(job["elapsed_seconds"], field=f"{slot}.elapsed_seconds")
            rss = _number(job["peak_rss_bytes"], field=f"{slot}.peak_rss_bytes")
            elapsed_values.append(float(elapsed))
            rss_values.append(int(rss))
        if (
            isinstance(operational.get("elapsed_seconds_sum"), bool)
            or not isinstance(operational.get("elapsed_seconds_sum"), (int, float))
            or float(operational["elapsed_seconds_sum"]) != sum(elapsed_values)
            or isinstance(operational.get("elapsed_seconds_max"), bool)
            or not isinstance(operational.get("elapsed_seconds_max"), (int, float))
            or float(operational["elapsed_seconds_max"]) != max(elapsed_values)
            or type(operational.get("peak_rss_bytes_max")) is not int
            or operational["peak_rss_bytes_max"] != max(rss_values)
        ):
            raise ValueError("Attempt07 aggregate operational summaries changed")


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
    attempt07_plan: str | Path = DEFAULT_ATTEMPT07_PLAN_PATH,
    done_metadata: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Validate five redacted proofs and atomically publish one receipt."""

    proof_paths = {
        "root0_batch_a": Path(root0_batch_a),
        "root0_batch_b": Path(root0_batch_b),
        "root0_scalar": Path(root0_scalar),
        "root1_batch": Path(root1_batch),
        "root2_batch": Path(root2_batch),
    }
    output_path = Path(output)
    protected = {
        **proof_paths,
        "source": Path(source),
        "preflight_plan": Path(preflight_plan),
        "attempt07_plan": Path(attempt07_plan),
        **{
            f"done:{label}": Path(path)
            for label, path in (done_metadata or {}).items()
        },
    }
    output_identity = os.path.normcase(str(output_path.resolve(strict=False)))
    if any(
        output_identity == os.path.normcase(str(path.resolve(strict=False)))
        for path in protected.values()
    ):
        raise ValueError("Attempt07 preflight aggregate output aliases an input")
    failures: list[str] = []
    validated: dict[str, _ValidatedProof] = {}
    proof_paths_distinct = _proof_paths_are_distinct(proof_paths)
    if not proof_paths_distinct:
        failures.append("five_proof_artifact_paths_not_distinct")
    context_valid = True
    try:
        load_preflight_plan(preflight_plan)
        plan = load_and_validate_attempt07_plan(attempt07_plan)
        if _sha256_file(Path(attempt07_plan)) != M43_ATTEMPT07_PLAN_SHA256:
            raise ValueError("Attempt07 plan SHA changed")
        del plan
        preflight_plan_sha256 = _sha256_file(Path(preflight_plan))
        sources = _expected_sources(Path(source))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        context_valid = False
        preflight_plan_sha256 = ""
        sources = {}
        failures.append("frozen_context_invalid")

    if context_valid:
        for label, expected_root, expected_batch in _SLOT_SPECS:
            try:
                validated[label] = _validate_proof_row(
                    proof_paths[label],
                    expected_root=expected_root,
                    expected_batch=expected_batch,
                    expected_source=sources[expected_root],
                    preflight_plan_sha256=preflight_plan_sha256,
                )
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                ValueError,
            ):
                failures.append(f"proof_validation_failed:{label}")

    all_five_valid = len(validated) == len(_SLOT_SPECS)
    root_coverage = all_five_valid and {
        int(proof.row["source"]["source_root_index"])
        for proof in validated.values()
    } == {0, 1, 2}
    batch_bytes_equal = (
        "root0_batch_a" in validated
        and "root0_batch_b" in validated
        and validated["root0_batch_a"].canonical_bytes
        == validated["root0_batch_b"].canonical_bytes
    )
    batch_opaque_equal = (
        "root0_batch_a" in validated
        and "root0_batch_b" in validated
        and validated["root0_batch_a"].row["result_proof"][
            "opaque_teacher_sha256"
        ]
        == validated["root0_batch_b"].row["result_proof"][
            "opaque_teacher_sha256"
        ]
    )
    semantic_parity = (
        "root0_scalar" in validated
        and "root0_batch_a" in validated
        and validated["root0_scalar"].row["result_proof"][
            "semantic_parity_sha256"
        ]
        == validated["root0_batch_a"].row["result_proof"][
            "semantic_parity_sha256"
        ]
    )
    if all_five_valid and not root_coverage:
        failures.append("root_coverage_or_mode_mapping_failed")
    if all_five_valid and not batch_bytes_equal:
        failures.append("root0_batch_canonical_determinism_failed")
    if all_five_valid and not batch_opaque_equal:
        failures.append("root0_batch_opaque_determinism_failed")
    if all_five_valid and not semantic_parity:
        failures.append("root0_scalar_batch_semantic_parity_failed")

    proof_gates = {
        "frozen_context_valid": context_valid,
        "five_proof_artifact_paths_distinct": proof_paths_distinct,
        "all_five_canonical_proofs_valid": all_five_valid,
        "exact_root_and_mode_coverage": root_coverage,
        "root0_batch_a_b_canonical_rows_identical": batch_bytes_equal,
        "root0_batch_opaque_mode_determinism": batch_opaque_equal,
        "root0_scalar_batch_semantic_parity": semantic_parity,
        "three_source_roots_covered": root_coverage,
        "teacher_values_absent": all_five_valid,
        "arm_details_absent": all_five_valid,
        "current_profile_unchanged": all_five_valid,
    }
    go = all(proof_gates.values()) and not failures
    reasons = ["all_preflight_proof_gates_passed"] if go else sorted(set(failures))
    report = {
        "schema": ATTEMPT07_PREFLIGHT_AGGREGATE_SCHEMA,
        "status": "complete",
        "decision": "go" if go else "no_go",
        "reasons": reasons,
        "proof_input_count": len(_SLOT_SPECS),
        "valid_proof_count": len(validated),
        "expected_slots": list(_SLOT_NAMES),
        "root_coverage": [0, 1, 2] if root_coverage else [],
        "proof_gates": proof_gates,
        "proof_file_sha256": {
            label: validated[label].file_sha256 for label in sorted(validated)
        },
        "cross_mode_opaque_teacher_hash_compared": False,
        "science_boundary": {
            "proof_only_no_teacher_values_or_arm_details": True,
            "arm_selection_allowed": False,
            "fit_allowed": False,
            "threshold_selection_allowed": False,
            "runtime_activation_allowed": False,
            "current_profile_resolved": False,
            "current_profile_mutated": False,
            "fresh_seed_or_root_opened": False,
            "done_metadata_is_science_input": False,
        },
        "contract": {
            "preflight_plan_schema": ATTEMPT07_PREFLIGHT_PLAN_SCHEMA,
            "preflight_plan_sha256": preflight_plan_sha256,
            "attempt06_source_plan_sha256": ATTEMPT06_SOURCE_PLAN_SHA256,
            "attempt06_merged_source_sha256": ATTEMPT07_PREFLIGHT_SOURCE_SHA256,
            "attempt06_wrapper_schema": ATTEMPT06_SHARD_ROW_SCHEMA,
            "attempt06_teacher_schema": ATTEMPT06_TEACHER_SCHEMA,
            "attempt07_plan_sha256": M43_ATTEMPT07_PLAN_SHA256,
            "attempt07_teacher_schema": ATTEMPT07_TEACHER_SCHEMA,
            "model_sha256": ATTEMPT06_FROZEN_MODEL_SHA256,
            "ai_profiles_sha256": AI_PROFILES_SHA256,
        },
        "operational_diagnostics": _operational_diagnostics(done_metadata),
    }
    _atomic_write_once(output_path, canonical_json_bytes(report))
    return report


def _parse_done_metadata(values: Sequence[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or label not in _SLOT_NAMES or not raw_path:
            raise ValueError("--done-metadata must be SLOT=PATH for a frozen slot")
        if label in parsed:
            raise ValueError(f"duplicate DONE metadata slot: {label}")
        parsed[label] = Path(raw_path)
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for label in _SLOT_NAMES:
        parser.add_argument(f"--{label.replace('_', '-')}", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE_PATH))
    parser.add_argument("--preflight-plan", default=str(DEFAULT_PREFLIGHT_PLAN_PATH))
    parser.add_argument("--attempt07-plan", default=str(DEFAULT_ATTEMPT07_PLAN_PATH))
    parser.add_argument("--done-metadata", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    done = _parse_done_metadata(args.done_metadata)
    aggregate_preflight_proofs(
        root0_batch_a=args.root0_batch_a,
        root0_batch_b=args.root0_batch_b,
        root0_scalar=args.root0_scalar,
        root1_batch=args.root1_batch,
        root2_batch=args.root2_batch,
        output=args.output,
        source=args.source,
        preflight_plan=args.preflight_plan,
        attempt07_plan=args.attempt07_plan,
        done_metadata=done,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ATTEMPT07_PREFLIGHT_AGGREGATE_SCHEMA",
    "aggregate_preflight_proofs",
    "validate_preflight_go_aggregate",
    "main",
]
