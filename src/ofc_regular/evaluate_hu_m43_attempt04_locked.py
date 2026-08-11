"""Consume and evaluate the Attempt04 locked200 split exactly once.

The global marker is durably created before the first locked shard stat, hash,
open, parse, or model evaluation.  The frozen v6 model and absolute threshold
are diagnostic gates for launching fresh population evaluation only; this
module cannot train, retune, promote, activate ``current``, or report teacher
values as realized match EV.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from .hu_m43_attempt02_contract import _identity_digest, _row_identity
from .hu_m43_attempt04_runtime import (
    M43_ATTEMPT04_LOCKED_MARKER_SCHEMA,
    M43_ATTEMPT04_LOCKED_RECEIPT_SCHEMA,
    M43_ATTEMPT04_LOCKED_RECORDS,
    file_sha256,
    load_bound_attempt04_v6_model,
    read_json_mapping,
    self_digest,
    validate_attempt04_runtime_artifact_files,
    validate_attempt04_runtime_freeze,
)
from .train_hu_m4_joint_model import prepare_teacher_samples, read_teacher_jsonl


ATTEMPT04_PROFILES = (
    "stage19_p0",
    "stage9f_p2",
    "stage7_m5_r10",
    "stage3_baseline",
    "random_exact_final",
)
ATTEMPT04_LOCKED_GATES: dict[str, float | int] = {
    "states": 200,
    "states_per_profile": 40,
    "raw_positive_rate": 0.40,
    "raw_positive_rate_per_profile": 0.30,
    "minimum_fires": 30,
    "minimum_fires_per_profile": 3,
    "false_positive_rate": 0.30,
    "p95_loss": 25.0,
    "p99_loss": 40.0,
    "max_loss": 50.0,
}


def evaluate_attempt04_locked_once(
    *,
    model_path: str | Path,
    final_training_manifest_path: str | Path,
    threshold_lock_path: str | Path,
    runtime_freeze_path: str | Path,
    attempt04_plan_path: str | Path,
    population_plan_path: str | Path,
    repo_root: str | Path,
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Claim locked200, evaluate one frozen policy, and write a Go/No-Go receipt."""

    root = Path(repo_root).resolve()
    paths = {
        "model": Path(model_path).resolve(),
        "training": Path(final_training_manifest_path).resolve(),
        "threshold": Path(threshold_lock_path).resolve(),
        "freeze": Path(runtime_freeze_path).resolve(),
        "attempt04_plan": Path(attempt04_plan_path).resolve(),
        "population_plan": Path(population_plan_path).resolve(),
    }
    receipt_output = Path(receipt_path).resolve()
    if len(set(paths.values())) != len(paths) or receipt_output in set(paths.values()):
        raise ValueError("Attempt04 locked lifecycle inputs/receipt are aliased")
    if not all(path.is_file() for path in paths.values()):
        raise ValueError("Attempt04 locked lifecycle input is missing")
    if receipt_output.exists():
        raise FileExistsError("Attempt04 locked200 receipt already exists")

    freeze = read_json_mapping(paths["freeze"], "Attempt04 runtime freeze")
    training = read_json_mapping(paths["training"], "Attempt04 final manifest")
    threshold = read_json_mapping(paths["threshold"], "Attempt04 threshold lock")
    validate_attempt04_runtime_freeze(
        freeze,
        final_training_manifest=training,
        threshold_lock=threshold,
    )
    frozen_model = _mapping(freeze.get("model"), "freeze.model")
    locked = _mapping(freeze.get("locked200"), "freeze.locked200")
    implementation_path = root / "src/ofc_regular/hu_m43_joint_model_v6.py"
    actual_hashes = validate_attempt04_runtime_artifact_files(
        freeze,
        model_path=paths["model"],
        final_training_manifest_path=paths["training"],
        threshold_lock_path=paths["threshold"],
        attempt04_plan_path=paths["attempt04_plan"],
        population_plan_path=paths["population_plan"],
        v6_implementation_path=implementation_path,
    )
    actual_hashes["freeze"] = file_sha256(paths["freeze"])
    model = load_bound_attempt04_v6_model(
        paths["model"],
        expected_sha256=str(frozen_model["file_sha256"]),
        runtime_freeze=freeze,
        final_training_manifest_path=paths["training"],
        threshold_lock_path=paths["threshold"],
    )

    # Metadata-only path resolution is allowed.  No locked shard is stat'ed,
    # hashed, opened, or parsed until the exclusive marker is durable below.
    shard_specs = tuple(
        _mapping(row, f"locked200.ordered_shards[{index}]")
        for index, row in enumerate(locked["ordered_shards"])
    )
    shard_paths = tuple(
        _resolve(root, row.get("path"), "locked200 shard path")
        for row in shard_specs
    )
    marker_output = _resolve(
        root,
        locked.get("global_consumption_marker"),
        "locked200 global marker",
    )
    all_lifecycle_paths = set(paths.values()) | {receipt_output}
    if (
        len(set(shard_paths)) != len(shard_paths)
        or any(path in all_lifecycle_paths for path in shard_paths)
        or marker_output in all_lifecycle_paths
        or marker_output in set(shard_paths)
    ):
        raise ValueError("Attempt04 locked200 paths alias lifecycle artifacts")
    identity = str(locked["identity_sha256"])
    marker_hits = _find_identity_markers(root, identity)
    if marker_hits or marker_output.exists():
        raise FileExistsError("Attempt04 locked200 identity is already consumed")

    marker: dict[str, Any] = {
        "schema": M43_ATTEMPT04_LOCKED_MARKER_SCHEMA,
        "status": "claimed_before_locked200_content_read",
        "runtime_freeze_sha256": freeze["freeze_sha256"],
        "runtime_freeze_file_sha256": actual_hashes["freeze"],
        "model_sha256": actual_hashes["model"],
        "final_training_manifest_file_sha256": actual_hashes["training"],
        "threshold_lock_file_sha256": actual_hashes["threshold"],
        "attempt04_plan_file_sha256": actual_hashes["attempt04_plan"],
        "population_plan_file_sha256": actual_hashes["population_plan"],
        "locked200_identity_sha256": identity,
        "locked200_records": M43_ATTEMPT04_LOCKED_RECORDS,
        "evaluation_pass_count": 1,
        "claim_is_consuming_even_on_crash": True,
        "threshold_reselection_performed": False,
        "current_profile_resolved": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    marker["marker_sha256"] = self_digest(marker, "marker_sha256")
    _write_json_exclusive(marker_output, marker)

    # First permitted locked-content access.
    rows, samples, audited = _audit_and_read_locked200(
        shard_paths,
        shard_specs=shard_specs,
        expected_identity_sha256=identity,
    )
    diagnostic = evaluate_attempt04_fixed_threshold(model, rows, samples)
    population_allowed = diagnostic["status"] == "go_locked200"
    receipt: dict[str, Any] = {
        "schema": M43_ATTEMPT04_LOCKED_RECEIPT_SCHEMA,
        "status": (
            "go_locked200_population_launch_eligible"
            if population_allowed
            else "no_go_locked200"
        ),
        "promotion_status": (
            "eligible_to_launch_fresh_population"
            if population_allowed
            else "no_go_locked200"
        ),
        "population_launch_allowed": population_allowed,
        "runtime_freeze_sha256": freeze["freeze_sha256"],
        "runtime_freeze_file_sha256": actual_hashes["freeze"],
        "model_sha256": actual_hashes["model"],
        "final_training_manifest_file_sha256": actual_hashes["training"],
        "threshold_lock_file_sha256": actual_hashes["threshold"],
        "attempt04_plan_file_sha256": actual_hashes["attempt04_plan"],
        "population_plan_file_sha256": actual_hashes["population_plan"],
        "frozen_threshold": float(model.safety_threshold),
        "locked200_identity_sha256": audited["identity_sha256"],
        "locked200_records": audited["records"],
        "locked200_ordered_shards": audited["ordered_shards"],
        "consumption_marker_resolved_path": str(marker_output),
        "consumption_marker_file_sha256": file_sha256(marker_output),
        "consumption_marker_canonical_sha256": marker["marker_sha256"],
        "evaluation_pass_count": 1,
        "threshold_search_performed": False,
        "threshold_reselection_performed": False,
        "model_selection_performed": False,
        "feature_selection_performed": False,
        "algorithm_or_gate_change_after_result_allowed": False,
        "current_profile_resolved": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "policy_promoted": False,
        "requires_fresh_population_acceptance": population_allowed,
        "minimum_population_valid_overrides": 300,
        "teacher_value_status": (
            "diagnostic_only_not_realized_match_ev_not_runtime_gate"
        ),
        "locked_diagnostic": diagnostic,
    }
    receipt["receipt_sha256"] = self_digest(receipt, "receipt_sha256")
    validate_attempt04_locked_receipt(
        receipt,
        runtime_freeze=freeze,
        marker=marker,
    )
    _write_json_exclusive(receipt_output, receipt)
    return receipt


def evaluate_attempt04_fixed_threshold(
    model: Any,
    raw_rows: Sequence[Mapping[str, Any]],
    samples: Sequence[Any],
) -> dict[str, Any]:
    if len(raw_rows) != len(samples):
        raise ValueError("Attempt04 locked raw/prepared row count mismatch")
    profile_counts: Counter[str] = Counter()
    result_rows: list[dict[str, Any]] = []
    mapping_exact = True
    baseline_exact = True
    for state_index, (raw, sample) in enumerate(zip(raw_rows, samples, strict=True)):
        profile = _root_profile(raw)
        profile_counts[profile] += 1
        _require_targets(sample)
        baseline = int(sample.baseline_index)
        heads = model.predict_heads_sample(
            sample.policy_sample, baseline_index=baseline
        )
        decision = model.select_action_index(
            sample.policy_sample, baseline_index=baseline
        )
        proposal = int(heads.proposal_index)
        mapping_exact = bool(
            mapping_exact
            and proposal != baseline
            and 0 <= proposal < len(sample.policy_sample["actions"])
            and decision.proposal_index == proposal
            and decision.baseline_index == baseline
            and decision.selected_index in {baseline, proposal}
        )
        baseline_exact = bool(
            baseline_exact
            and float(heads.action_score[baseline]) == 0.0
            and float(heads.base_delta[baseline]) == 0.0
            and float(heads.delta_disagreement[baseline]) == 0.0
            and not bool(heads.risk_eligible_mask[baseline])
        )
        paired = np.asarray(sample.teacher_paired_delta_mean, dtype=np.float64)
        teacher_delta = float(paired[proposal])
        row = {
            "state_index": state_index,
            "profile": profile,
            "seat": str(sample.seat),
            "proposal_index": proposal,
            "baseline_index": baseline,
            "risk_eligible": bool(heads.proposal_risk_eligible),
            "safety_probability": float(decision.safety_probability),
            "fired": bool(decision.override_fired),
            "teacher_delta": teacher_delta,
            "downside_loss_p95": float(sample.downside_loss_p95[proposal]),
            "downside_loss_p99": float(sample.downside_loss_p99[proposal]),
            "downside_loss_max": float(sample.downside_loss_max[proposal]),
        }
        if not all(
            math.isfinite(float(row[field]))
            for field in (
                "safety_probability",
                "teacher_delta",
                "downside_loss_p95",
                "downside_loss_p99",
                "downside_loss_max",
            )
        ):
            raise ValueError("Attempt04 locked evaluation produced non-finite metrics")
        result_rows.append(row)

    raw = _raw_metrics(result_rows)
    selected = _selected_metrics(result_rows)
    by_profile = {
        profile: {
            "raw": _raw_metrics(
                [row for row in result_rows if row["profile"] == profile]
            ),
            "selected": _selected_metrics(
                [row for row in result_rows if row["profile"] == profile]
            ),
        }
        for profile in ATTEMPT04_PROFILES
    }
    by_seat = {
        seat: {
            "raw": _raw_metrics(
                [row for row in result_rows if row["seat"] == seat]
            ),
            "selected": _selected_metrics(
                [row for row in result_rows if row["seat"] == seat]
            ),
        }
        for seat in ("first", "second")
    }
    expected_profile_counts = {
        profile: int(ATTEMPT04_LOCKED_GATES["states_per_profile"])
        for profile in ATTEMPT04_PROFILES
    }
    gates = {
        "exact_locked200_states": len(result_rows)
        == ATTEMPT04_LOCKED_GATES["states"],
        "exact_five_profile_balance": dict(profile_counts)
        == expected_profile_counts,
        "semantic_action_mapping_exact": mapping_exact,
        "baseline_quantities_exact_zero": baseline_exact,
        "raw_proposal_every_state": raw["proposals"] == len(result_rows),
        "raw_proposal_positive_rate_at_least_0_40": raw["positive_rate"]
        >= ATTEMPT04_LOCKED_GATES["raw_positive_rate"],
        "raw_proposal_positive_rate_each_profile_at_least_0_30": all(
            by_profile[profile]["raw"]["positive_rate"]
            >= ATTEMPT04_LOCKED_GATES["raw_positive_rate_per_profile"]
            for profile in ATTEMPT04_PROFILES
        ),
        # Raw forced proposals are mandatory diagnostics, but their mean/LCB
        # are not Go gates: v6 is a selective fallback policy, not a full
        # replacement.  The actually fired subset retains strict mean/LCB.
        "fixed_threshold_fires_at_least_30": selected["fires"]
        >= ATTEMPT04_LOCKED_GATES["minimum_fires"],
        "fixed_threshold_fires_each_profile_at_least_3": all(
            by_profile[profile]["selected"]["fires"]
            >= ATTEMPT04_LOCKED_GATES["minimum_fires_per_profile"]
            for profile in ATTEMPT04_PROFILES
        ),
        "selected_mean_delta_per_fire_strictly_positive": selected[
            "mean_delta_per_fire"
        ]
        > 0.0,
        "selected_delta_per_state_strictly_positive": selected[
            "delta_per_state"
        ]
        > 0.0,
        "selected_one_sided_student_t_90_lcb_strictly_positive": _positive_lcb(
            selected
        ),
        "selected_false_positive_rate_at_most_0_30": selected[
            "false_positive_rate"
        ]
        <= ATTEMPT04_LOCKED_GATES["false_positive_rate"],
        "selected_delta_per_state_each_profile_nonnegative": all(
            by_profile[profile]["selected"]["delta_per_state"] >= 0.0
            for profile in ATTEMPT04_PROFILES
        ),
        "selected_p95_loss_at_most_25": selected["loss_tail"]["p95"]
        <= ATTEMPT04_LOCKED_GATES["p95_loss"],
        "selected_p99_loss_at_most_40": selected["loss_tail"]["p99"]
        <= ATTEMPT04_LOCKED_GATES["p99_loss"],
        "selected_max_loss_at_most_50": selected["loss_tail"]["max"]
        <= ATTEMPT04_LOCKED_GATES["max_loss"],
    }
    status = "go_locked200" if all(gates.values()) else "no_go_locked200"
    row_digest_payload = [
        {
            "state_index": row["state_index"],
            "profile": row["profile"],
            "seat": row["seat"],
            "proposal_index": row["proposal_index"],
            "baseline_index": row["baseline_index"],
            "risk_eligible": row["risk_eligible"],
            "safety_probability": row["safety_probability"],
            "fired": row["fired"],
            "teacher_delta": row["teacher_delta"],
            "downside_loss_p95": row["downside_loss_p95"],
            "downside_loss_p99": row["downside_loss_p99"],
            "downside_loss_max": row["downside_loss_max"],
        }
        for row in result_rows
    ]
    return {
        "schema": "hu_m43_attempt04_locked200_diagnostic_v1",
        "status": status,
        "split": "fresh_locked200",
        "states": len(result_rows),
        "profile_counts": dict(profile_counts),
        "frozen_threshold": float(model.safety_threshold),
        "safety_enabled": bool(model.safety_enabled),
        "raw_proposal_metrics": raw,
        "fixed_threshold_metrics": selected,
        "by_profile": by_profile,
        "by_seat": by_seat,
        "evaluation_rows_sha256": self_digest(
            {"rows": row_digest_payload}, "unused"
        ),
        "fixed_gates": dict(ATTEMPT04_LOCKED_GATES),
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "population_launch_allowed": status == "go_locked200",
        "threshold_search_performed": False,
        "threshold_reselection_performed": False,
        "teacher_value_status": (
            "diagnostic_only_not_realized_match_ev_not_runtime_gate"
        ),
    }


def validate_attempt04_locked_receipt(
    receipt: Mapping[str, Any],
    *,
    runtime_freeze: Mapping[str, Any],
    marker: Mapping[str, Any],
) -> None:
    if receipt.get("schema") != M43_ATTEMPT04_LOCKED_RECEIPT_SCHEMA:
        raise ValueError("Attempt04 locked200 receipt schema mismatch")
    if receipt.get("receipt_sha256") != self_digest(receipt, "receipt_sha256"):
        raise ValueError("Attempt04 locked200 receipt self digest mismatch")
    diagnostic = _mapping(receipt.get("locked_diagnostic"), "locked_diagnostic")
    gates = _mapping(diagnostic.get("gates"), "locked_diagnostic.gates")
    all_pass = bool(gates) and all(value is True for value in gates.values())
    go = receipt.get("population_launch_allowed") is True
    if go != all_pass or go != (diagnostic.get("status") == "go_locked200"):
        raise ValueError("Attempt04 locked200 receipt Go does not match its gates")
    if (
        receipt.get("runtime_freeze_sha256") != runtime_freeze.get("freeze_sha256")
        or receipt.get("model_sha256") != marker.get("model_sha256")
        or receipt.get("consumption_marker_canonical_sha256")
        != marker.get("marker_sha256")
        or receipt.get("locked200_identity_sha256")
        != marker.get("locked200_identity_sha256")
        or receipt.get("locked200_records") != M43_ATTEMPT04_LOCKED_RECORDS
        or receipt.get("evaluation_pass_count") != 1
    ):
        raise ValueError("Attempt04 locked200 receipt lifecycle binding mismatch")
    if go:
        if (
            receipt.get("status")
            != "go_locked200_population_launch_eligible"
            or receipt.get("promotion_status")
            != "eligible_to_launch_fresh_population"
            or receipt.get("requires_fresh_population_acceptance") is not True
        ):
            raise ValueError("Attempt04 locked200 Go status mismatch")
    elif (
        receipt.get("status") != "no_go_locked200"
        or receipt.get("promotion_status") != "no_go_locked200"
        or receipt.get("requires_fresh_population_acceptance") is not False
    ):
        raise ValueError("Attempt04 locked200 No-Go status mismatch")
    if any(
        receipt.get(field) is not False
        for field in (
            "threshold_search_performed",
            "threshold_reselection_performed",
            "model_selection_performed",
            "feature_selection_performed",
            "algorithm_or_gate_change_after_result_allowed",
            "current_profile_resolved",
            "current_profile_mutated",
            "runtime_policy_activated",
            "policy_promoted",
        )
    ):
        raise ValueError("Attempt04 locked200 receipt contains an unsafe flag")


def _audit_and_read_locked200(
    paths: Sequence[Path],
    *,
    shard_specs: Sequence[Mapping[str, Any]],
    expected_identity_sha256: str,
) -> tuple[list[dict[str, Any]], list[Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    audited_shards: list[dict[str, Any]] = []
    for index, (path, spec) in enumerate(zip(paths, shard_specs, strict=True)):
        if not path.is_file():
            raise ValueError("Attempt04 locked200 shard is missing after claim")
        byte_size = path.stat().st_size
        if byte_size != int(spec["bytes"]):
            raise ValueError("Attempt04 locked200 shard byte size changed after claim")
        actual_sha = file_sha256(path)
        if actual_sha != spec.get("file_sha256"):
            raise ValueError("Attempt04 locked200 shard SHA changed after claim")
        shard_rows = read_teacher_jsonl(path)
        if len(shard_rows) != int(spec["records"]):
            raise ValueError("Attempt04 locked200 shard record count changed")
        rows.extend(shard_rows)
        audited_shards.append(
            {
                "index": index,
                "path": str(path),
                "file_sha256": actual_sha,
                "bytes": byte_size,
                "records": len(shard_rows),
            }
        )
    if len(rows) != M43_ATTEMPT04_LOCKED_RECORDS:
        raise ValueError("Attempt04 locked200 total record count changed")
    identities = [_row_identity(row, "Attempt04 locked200") for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Attempt04 locked200 identities are duplicated")
    identity_sha = _identity_digest(set(identities))
    if identity_sha != expected_identity_sha256:
        raise ValueError("Attempt04 locked200 identity digest changed")
    samples = prepare_teacher_samples(rows)
    return rows, samples, {
        "records": len(rows),
        "identity_sha256": identity_sha,
        "ordered_shards": audited_shards,
    }


def _raw_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deltas = np.asarray([float(row["teacher_delta"]) for row in rows], dtype=np.float64)
    lcb = _student_t_lcb(deltas)
    return {
        "states": len(rows),
        "proposals": len(rows),
        "positive_count": int(np.sum(deltas > 0.0)),
        "positive_rate": float(np.mean(deltas > 0.0)) if deltas.size else 0.0,
        "mean_delta": float(np.mean(deltas)) if deltas.size else 0.0,
        "delta_per_state": float(np.sum(deltas) / len(rows)) if rows else 0.0,
        "mean_delta_lcb": lcb,
    }


def _selected_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fired = [row for row in rows if row.get("fired") is True]
    deltas = np.asarray(
        [float(row["teacher_delta"]) for row in fired], dtype=np.float64
    )
    false_positives = int(np.sum(deltas <= 0.0))
    return {
        "states": len(rows),
        "risk_eligible_proposals": sum(
            bool(row.get("risk_eligible")) for row in rows
        ),
        "fires": len(fired),
        "fire_rate": float(len(fired) / len(rows)) if rows else 0.0,
        "false_positives": false_positives,
        "false_positive_rate": (
            float(false_positives / len(fired)) if fired else 0.0
        ),
        "teacher_delta_sum": float(np.sum(deltas)) if deltas.size else 0.0,
        "mean_delta_per_fire": float(np.mean(deltas)) if deltas.size else 0.0,
        "delta_per_state": float(np.sum(deltas) / len(rows)) if rows else 0.0,
        "mean_delta_lcb": _student_t_lcb(deltas),
        "loss_tail": {
            "p95": max(
                (float(row["downside_loss_p95"]) for row in fired), default=0.0
            ),
            "p99": max(
                (float(row["downside_loss_p99"]) for row in fired), default=0.0
            ),
            "max": max(
                (float(row["downside_loss_max"]) for row in fired), default=0.0
            ),
        },
    }


def _student_t_lcb(values: np.ndarray) -> dict[str, Any]:
    count = int(values.size)
    if count < 2:
        return {
            "method": "one_sided_student_t_over_independent_state_clusters",
            "confidence": 0.90,
            "clusters": count,
            "degrees_of_freedom": None,
            "sample_standard_error": None,
            "critical_value": None,
            "lower_bound": None,
        }
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / math.sqrt(count))
    critical = float(student_t.ppf(0.90, count - 1))
    return {
        "method": "one_sided_student_t_over_independent_state_clusters",
        "confidence": 0.90,
        "clusters": count,
        "degrees_of_freedom": count - 1,
        "sample_standard_error": standard_error,
        "critical_value": critical,
        "lower_bound": mean - critical * standard_error,
    }


def _positive_lcb(metrics: Mapping[str, Any]) -> bool:
    lcb = _mapping(metrics.get("mean_delta_lcb"), "mean_delta_lcb")
    value = lcb.get("lower_bound")
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0.0


def _require_targets(sample: Any) -> None:
    arrays = (
        sample.teacher_paired_delta_mean,
        sample.downside_loss_p95,
        sample.downside_loss_p99,
        sample.downside_loss_max,
    )
    if any(value is None for value in arrays):
        raise ValueError("Attempt04 locked200 row is missing paired teacher targets")


def _root_profile(row: Mapping[str, Any]) -> str:
    provenance = _mapping(row.get("provenance"), "locked200 provenance")
    profile = provenance.get("root_profile")
    if profile not in ATTEMPT04_PROFILES:
        raise ValueError("Attempt04 locked200 root profile is invalid")
    return str(profile)


def _find_identity_markers(root: Path, identity_sha256: str) -> list[Path]:
    matches: list[Path] = []
    for path in root.rglob("M43_ATTEMPT04_LOCKED200_CONSUMED.json"):
        try:
            payload = read_json_mapping(path, "Attempt04 locked200 marker")
        except (OSError, json.JSONDecodeError) as exc:  # defensive for concurrent writes
            raise ValueError(f"invalid Attempt04 locked200 marker: {path}") from exc
        if payload.get("locked200_identity_sha256") == identity_sha256:
            matches.append(path.resolve())
    return sorted(matches)


def _resolve(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is missing")
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--final-training-manifest", type=Path, required=True)
    parser.add_argument("--threshold-lock", type=Path, required=True)
    parser.add_argument("--runtime-freeze", type=Path, required=True)
    parser.add_argument("--attempt04-plan", type=Path, required=True)
    parser.add_argument("--population-plan", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = evaluate_attempt04_locked_once(
        model_path=args.model,
        final_training_manifest_path=args.final_training_manifest,
        threshold_lock_path=args.threshold_lock,
        runtime_freeze_path=args.runtime_freeze,
        attempt04_plan_path=args.attempt04_plan,
        population_plan_path=args.population_plan,
        repo_root=args.repo_root,
        receipt_path=args.receipt,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ATTEMPT04_LOCKED_GATES",
    "ATTEMPT04_PROFILES",
    "evaluate_attempt04_fixed_threshold",
    "evaluate_attempt04_locked_once",
    "validate_attempt04_locked_receipt",
]
