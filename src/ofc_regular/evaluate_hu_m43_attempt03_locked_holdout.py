"""Consume the inherited Attempt03 locked holdout exactly once.

The canonical global marker is created before the locked file is stat'ed,
hashed, parsed, or evaluated.  The fixed v5 model and threshold are diagnostic
only here: no model, feature, gate, or threshold selection is allowed, and the
result is not reported as realized match EV.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .hu_m43_attempt02_contract import _find_consumption_markers
from .hu_m43_attempt03_contract import _identity_digest, _row_identity
from .hu_m43_attempt03_runtime import (
    M43_ATTEMPT03_LOCKED_MARKER_SCHEMA,
    M43_ATTEMPT03_LOCKED_RECEIPT_SCHEMA,
    file_sha256,
    load_bound_attempt03_v5_model,
    read_json_mapping,
    self_digest,
    validate_attempt03_precalibration_receipt,
    validate_attempt03_runtime_freeze,
)
from .hu_m43_attempt03_training import load_attempt03_training_freeze
from .train_hu_m4_joint_model import prepare_teacher_samples, read_teacher_jsonl
from .validate_hu_m43_attempt03_population import (
    load_and_validate_attempt03_population_plan,
)


def evaluate_attempt03_locked_holdout_once(
    *,
    model_path: str | Path,
    final_training_manifest_path: str | Path,
    runtime_freeze_path: str | Path,
    training_freeze_path: str | Path,
    precalibration_receipt_path: str | Path,
    model_freeze_path: str | Path,
    attempt03_plan_path: str | Path,
    population_plan_path: str | Path,
    repo_root: str | Path,
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Claim the inherited lock, then run one fixed-threshold diagnostic."""

    root = Path(repo_root).resolve()
    paths = {
        "model": Path(model_path).resolve(),
        "training": Path(final_training_manifest_path).resolve(),
        "freeze": Path(runtime_freeze_path).resolve(),
        "training_freeze": Path(training_freeze_path).resolve(),
        "precalibration": Path(precalibration_receipt_path).resolve(),
        "model_freeze": Path(model_freeze_path).resolve(),
        "attempt03_plan": Path(attempt03_plan_path).resolve(),
        "population_plan": Path(population_plan_path).resolve(),
    }
    receipt_output = Path(receipt_path).resolve()
    if len(set(paths.values())) != len(paths) or receipt_output in set(paths.values()):
        raise ValueError("Attempt03 locked lifecycle inputs/receipt are aliased")
    if not all(path.is_file() for path in paths.values()):
        raise ValueError("Attempt03 locked lifecycle input is missing")
    if receipt_output.exists():
        raise FileExistsError("Attempt03 locked receipt already exists")

    freeze = read_json_mapping(paths["freeze"], "Attempt03 model-threshold freeze")
    training = read_json_mapping(paths["training"], "Attempt03 final manifest")
    validate_attempt03_runtime_freeze(freeze, final_training_manifest=training)
    frozen_model = _mapping(freeze.get("model"), "freeze.model")
    frozen_training = _mapping(
        freeze.get("training_manifest"), "freeze.training_manifest"
    )
    frozen_plan = _mapping(freeze.get("attempt03_plan"), "freeze.attempt03_plan")
    frozen_executable = _mapping(
        freeze.get("executable_model_freeze"),
        "freeze.executable_model_freeze",
    )
    frozen_training_pipeline = _mapping(
        freeze.get("training_pipeline_freeze"),
        "freeze.training_pipeline_freeze",
    )
    locked = _mapping(freeze.get("inherited_locked"), "freeze.inherited_locked")
    if (
        file_sha256(paths["model"]) != frozen_model.get("file_sha256")
        or file_sha256(paths["training"]) != frozen_training.get("file_sha256")
        or file_sha256(paths["attempt03_plan"])
        != frozen_plan.get("file_sha256")
        or file_sha256(paths["model_freeze"])
        != frozen_executable.get("file_sha256")
        or file_sha256(paths["training_freeze"])
        != frozen_training_pipeline.get("file_sha256")
    ):
        raise ValueError("Attempt03 locked lifecycle artifact hash changed")
    load_attempt03_training_freeze(
        paths["training_freeze"],
        repo_root=root,
        expected_model_freeze_path=paths["model_freeze"],
    )
    precalibration = read_json_mapping(
        paths["precalibration"], "Attempt03 pre-calibration receipt"
    )
    validate_attempt03_precalibration_receipt(precalibration)
    if precalibration.get("receipt_sha256") != training.get(
        "precalibration_receipt_sha256"
    ):
        raise ValueError("Attempt03 pre-calibration receipt binding changed")
    model = load_bound_attempt03_v5_model(
        paths["model"],
        expected_sha256=str(frozen_model["file_sha256"]),
        runtime_freeze=freeze,
        final_training_manifest_path=paths["training"],
    )
    population_plan = load_and_validate_attempt03_population_plan(
        paths["population_plan"], repo_root=root
    )
    population_plan.pop("_freshness_counts", None)
    population_plan_sha = file_sha256(paths["population_plan"])

    # Resolve metadata-only paths.  Do not stat/hash/open locked_path before
    # the marker is written below.
    locked_path = _resolve(root, locked.get("path"), "inherited locked path")
    marker_output = _resolve(
        root,
        locked.get("global_consumption_marker"),
        "canonical locked-consumption marker",
    )
    if marker_output in set(paths.values()) or marker_output == receipt_output:
        raise ValueError("Attempt03 canonical marker aliases another lifecycle path")
    identity = str(locked.get("identity_sha256", ""))
    marker_hits = _find_consumption_markers(root, identity)
    if marker_hits or marker_output.exists():
        raise FileExistsError("Attempt03 inherited locked identity is already consumed")

    marker: dict[str, Any] = {
        "schema": M43_ATTEMPT03_LOCKED_MARKER_SCHEMA,
        "status": "claimed_before_inherited_locked_content_read",
        "runtime_freeze_sha256": freeze["freeze_sha256"],
        "runtime_freeze_file_sha256": file_sha256(paths["freeze"]),
        "population_plan_file_sha256": population_plan_sha,
        "model_sha256": frozen_model["file_sha256"],
        "model_freeze_file_sha256": file_sha256(paths["model_freeze"]),
        "training_freeze_file_sha256": file_sha256(paths["training_freeze"]),
        "precalibration_receipt_file_sha256": file_sha256(
            paths["precalibration"]
        ),
        "locked_identity_sha256": identity,
        "locked_file_sha256": locked["file_sha256"],
        "locked_records": locked["records"],
        "evaluation_pass_count": 1,
        "claim_is_consuming_even_on_crash": True,
        "current_profile_resolved": False,
        "current_profile_changed": False,
        "runtime_policy_activated": False,
    }
    marker["marker_sha256"] = self_digest(marker, "marker_sha256")
    _write_json_exclusive(marker_output, marker)

    # This is the first operation permitted to touch the locked JSONL.  A crash
    # from here onward intentionally leaves the canonical marker consumed.
    samples, audited = _audit_and_read_locked(
        locked_path,
        expected=locked,
    )
    diagnostic = _evaluate_v5_fixed_threshold(model, samples)
    receipt: dict[str, Any] = {
        "schema": M43_ATTEMPT03_LOCKED_RECEIPT_SCHEMA,
        "status": "evaluated_once_diagnostic_only_no_activation",
        "runtime_freeze_sha256": freeze["freeze_sha256"],
        "runtime_freeze_file_sha256": file_sha256(paths["freeze"]),
        "population_plan_file_sha256": population_plan_sha,
        "model_sha256": frozen_model["file_sha256"],
        "model_freeze_file_sha256": file_sha256(paths["model_freeze"]),
        "training_freeze_file_sha256": file_sha256(paths["training_freeze"]),
        "precalibration_receipt_file_sha256": file_sha256(
            paths["precalibration"]
        ),
        "frozen_threshold": float(frozen_model["safety_threshold"]),
        "locked_identity_sha256": audited["identity_sha256"],
        "locked_file_sha256": audited["file_sha256"],
        "locked_records": audited["records"],
        "consumption_marker_resolved_path": str(marker_output),
        "consumption_marker_file_sha256": file_sha256(marker_output),
        "consumption_marker_canonical_sha256": marker["marker_sha256"],
        "evaluation_pass_count": 1,
        "threshold_search_performed": False,
        "threshold_reselection_performed": False,
        "model_selection_performed": False,
        "feature_selection_performed": False,
        "current_profile_resolved": False,
        "current_profile_changed": False,
        "runtime_policy_activated": False,
        "policy_promoted": False,
        "requires_fresh_population_acceptance": True,
        "minimum_population_valid_overrides": int(
            population_plan["minimum_valid_overrides"]
        ),
        "teacher_value_status": (
            "diagnostic_only_not_realized_match_ev_not_runtime_gate"
        ),
        "locked_diagnostic": diagnostic,
    }
    receipt["receipt_sha256"] = self_digest(receipt, "receipt_sha256")
    _write_json_exclusive(receipt_output, receipt)
    return receipt


def _audit_and_read_locked(
    path: Path, *, expected: Mapping[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    if not path.is_file():
        raise ValueError("Attempt03 inherited locked file is missing after claim")
    actual_sha = file_sha256(path)
    if actual_sha != expected.get("file_sha256"):
        raise ValueError("Attempt03 inherited locked file SHA changed after claim")
    rows = read_teacher_jsonl(path)
    expected_records = int(expected.get("records", -1))
    if len(rows) != expected_records:
        raise ValueError("Attempt03 inherited locked record count changed after claim")
    identities = [_row_identity(row, str(path)) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Attempt03 inherited locked identities are duplicated")
    identity_sha = _identity_digest(set(identities))
    if identity_sha != expected.get("identity_sha256"):
        raise ValueError("Attempt03 inherited locked identity digest changed")
    return prepare_teacher_samples(rows), {
        "records": len(rows),
        "file_sha256": actual_sha,
        "identity_sha256": identity_sha,
    }


def _evaluate_v5_fixed_threshold(model: Any, samples: Sequence[Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    top1 = 0
    for sample in samples:
        baseline = sample.baseline_index
        heads = model.predict_heads_sample(
            sample.policy_sample, baseline_index=baseline
        )
        proposal = int(heads.proposal_index)
        eligible = bool(heads.proposal_eligible)
        probability = (
            float(
                model.predict_safety_probability(
                    sample.policy_sample,
                    candidate_index=proposal,
                    baseline_index=baseline,
                )
            )
            if eligible
            else 0.0
        )
        fired = bool(
            eligible
            and model.safety_enabled
            and probability >= float(model.safety_threshold)
        )
        paired = np.asarray(sample.teacher_paired_delta_mean, dtype=np.float64)
        top1 += int(float(paired[proposal]) >= float(np.max(paired)) - 1.0e-9)
        rows.append(
            {
                "seat": sample.seat,
                "proposal_index": proposal,
                "baseline_index": baseline,
                "eligible": eligible,
                "positive_vote_count": int(heads.delta_positive_votes[proposal]),
                "safety_probability": probability,
                "fired": fired,
                "teacher_delta": float(paired[proposal]),
                "downside_loss_p95": float(sample.downside_loss_p95[proposal]),
                "downside_loss_p99": float(sample.downside_loss_p99[proposal]),
                "downside_loss_max": float(sample.downside_loss_max[proposal]),
            }
        )
    return {
        "split": "inherited_locked_holdout",
        "samples": len(samples),
        "proposal_top1_accuracy": float(top1 / len(samples)) if samples else 0.0,
        "frozen_threshold": float(model.safety_threshold),
        "safety_enabled": bool(model.safety_enabled),
        "ineligible_top_proposal_rerank_allowed": False,
        "override_metrics": _fixed_threshold_metrics(rows),
        "seat_override_metrics": {
            seat: _fixed_threshold_metrics(
                [row for row in rows if row["seat"] == seat]
            )
            for seat in ("first", "second")
        },
        "threshold_search_performed": False,
        "teacher_value_status": (
            "diagnostic_only_not_realized_match_ev_not_runtime_gate"
        ),
    }


def _fixed_threshold_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fired = [row for row in rows if row.get("fired") is True]
    deltas = np.asarray(
        [float(row["teacher_delta"]) for row in fired], dtype=np.float64
    )
    false_positives = deltas[deltas <= 0.0]
    return {
        "states": len(rows),
        "eligible_proposals": sum(bool(row.get("eligible")) for row in rows),
        "fires": len(fired),
        "fire_rate": float(len(fired) / len(rows)) if rows else 0.0,
        "false_positives": int(false_positives.size),
        "false_positive_rate": (
            float(false_positives.size / len(fired)) if fired else 0.0
        ),
        "teacher_delta_sum": float(np.sum(deltas)) if deltas.size else 0.0,
        "teacher_mean_delta_per_fire": (
            float(np.mean(deltas)) if deltas.size else 0.0
        ),
        "teacher_p05_delta": (
            float(np.quantile(deltas, 0.05)) if deltas.size else 0.0
        ),
        "teacher_min_delta": float(np.min(deltas)) if deltas.size else 0.0,
        "p95_loss": (
            max(float(row["downside_loss_p95"]) for row in fired)
            if fired
            else 0.0
        ),
        "p99_loss": (
            max(float(row["downside_loss_p99"]) for row in fired)
            if fired
            else 0.0
        ),
        "max_loss": (
            max(float(row["downside_loss_max"]) for row in fired)
            if fired
            else 0.0
        ),
    }


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
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--final-training-manifest", type=Path, required=True)
    parser.add_argument("--runtime-freeze", type=Path, required=True)
    parser.add_argument("--training-freeze", type=Path, required=True)
    parser.add_argument("--precalibration-receipt", type=Path, required=True)
    parser.add_argument("--model-freeze", type=Path, required=True)
    parser.add_argument("--attempt03-plan", type=Path, required=True)
    parser.add_argument("--population-plan", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = evaluate_attempt03_locked_holdout_once(
        model_path=args.model,
        final_training_manifest_path=args.final_training_manifest,
        runtime_freeze_path=args.runtime_freeze,
        training_freeze_path=args.training_freeze,
        precalibration_receipt_path=args.precalibration_receipt,
        model_freeze_path=args.model_freeze,
        attempt03_plan_path=args.attempt03_plan,
        population_plan_path=args.population_plan,
        repo_root=args.repo_root,
        receipt_path=args.receipt,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["evaluate_attempt03_locked_holdout_once"]
