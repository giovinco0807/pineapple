"""Freeze an already-trained M4.3 model and threshold before holdout access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hu_m43_pilot_contract import (
    M43_FREEZE_SCHEMA,
    load_and_validate_plan,
    validate_data_contract_binding,
    validate_model_threshold_freeze,
)
from .hu_m4_joint_model import (
    PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE,
    HuM4JointActionModel,
)


def build_model_threshold_freeze(
    *,
    model_path: str | Path,
    training_manifest_path: str | Path,
    data_contract_path: str | Path,
    plan_path: str | Path,
    repo_root: str | Path,
    train: Sequence[str | Path],
    calibration: Sequence[str | Path],
) -> dict[str, Any]:
    """Validate training inputs and return an immutable pre-holdout freeze."""

    model_source = Path(model_path).resolve()
    manifest_source = Path(training_manifest_path).resolve()
    contract_source = Path(data_contract_path).resolve()
    if len({model_source, manifest_source, contract_source}) != 3:
        raise ValueError("M4.3 model, manifest, and data contract must be distinct")
    model_sha = _sha256(model_source)
    training_manifest_sha = _sha256(manifest_source)
    training_manifest = _read_mapping(manifest_source)
    data_contract = _read_mapping(contract_source)
    plan = load_and_validate_plan(plan_path)
    binding = validate_data_contract_binding(
        data_contract,
        plan_path=plan_path,
        repo_root=repo_root,
        train=train,
        calibration=calibration,
    )

    formula = _mapping(training_manifest.get("action_score_formula"), "action_score_formula")
    if formula.get("mode") != PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE:
        raise ValueError("M4.3 freeze requires the paired-delta risk ensemble")
    if formula.get("baseline_action_score_exact_zero") is not True:
        raise ValueError("M4.3 freeze requires exact-zero baseline scoring")
    if formula.get("runtime_model") != "stored_crossfit_fold_ensemble":
        raise ValueError("M4.3 freeze requires the stored cross-fit fold ensemble")
    if formula.get("teacher_value_runtime_input") is not False:
        raise ValueError("M4.3 freeze forbids teacher value runtime input")
    if formula.get("teacher_lcb_runtime_gate") is not False:
        raise ValueError("M4.3 freeze forbids teacher LCB runtime gates")

    if training_manifest.get("locked_holdout_used_for_threshold_or_training") is not False:
        raise ValueError("M4.3 trainer used locked holdout before freeze")
    locked_status = _mapping(
        training_manifest.get("locked_holdout"), "locked_holdout"
    )
    if locked_status != {"status": "not_evaluated_pre_freeze"}:
        raise ValueError("M4.3 trainer must leave locked holdout unevaluated")
    inputs = _mapping(training_manifest.get("inputs"), "inputs")
    if set(inputs) != {"train", "calibration"}:
        raise ValueError("M4.3 training manifest may contain only train/calibration inputs")

    declared_binding = _mapping(
        training_manifest.get("m43_data_contract"), "m43_data_contract"
    )
    for key in (
        "contract_sha256",
        "plan_sha256",
        "prior_freshness_audit_sha256",
        "base_audit_sha256",
        "teacher_shards_all_splits_sha256",
    ):
        if declared_binding.get(key) != binding.get(key):
            raise ValueError(f"M4.3 trainer data binding mismatch: {key}")
    if declared_binding.get("verified_splits") != binding.get("verified_splits"):
        raise ValueError("M4.3 trainer verified-split binding mismatch")

    calibration_report = _mapping(
        training_manifest.get("calibration"), "calibration"
    )
    selected_threshold = float(calibration_report.get("selected_threshold"))
    if calibration_report.get("threshold_selection_source") != (
        "calibration.threshold_lock"
    ):
        raise ValueError("M4.3 threshold source is not threshold-lock")
    if calibration_report.get("safety_calibrator_sources") != [
        "train_oof",
        "calibration.safety_fit",
    ]:
        raise ValueError("M4.3 safety calibrator source declaration is unsafe")
    runtime_lock = _mapping(training_manifest.get("runtime_lock"), "runtime_lock")
    if runtime_lock.get("single_joint_artifact") is not True:
        raise ValueError("M4.3 runtime lock is not a single joint artifact")
    if runtime_lock.get("candidate_model_sha256") != model_sha:
        raise ValueError("M4.3 candidate artifact SHA mismatch")
    if runtime_lock.get("safety_model_sha256") != model_sha:
        raise ValueError("M4.3 safety artifact SHA mismatch")
    if float(runtime_lock.get("safety_threshold")) != selected_threshold:
        raise ValueError("M4.3 runtime/calibration threshold mismatch")

    model = HuM4JointActionModel.load(model_source)
    if model.action_score_mode != PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE:
        raise ValueError("M4.3 saved artifact action-score mode mismatch")
    if float(model.safety_threshold) != selected_threshold:
        raise ValueError("M4.3 saved artifact threshold mismatch")

    freeze: dict[str, Any] = {
        "schema": M43_FREEZE_SCHEMA,
        "status": "model_and_threshold_frozen_locked_unopened",
        "plan_sha256": binding["plan_sha256"],
        "data_contract_sha256": binding["contract_sha256"],
        "data_contract_resolved_path": str(contract_source),
        "locked_consumption_marker_resolved_path": str(
            contract_source.with_name("M43_LOCKED_CONSUMED.json").resolve()
        ),
        "prior_freshness_audit_sha256": binding[
            "prior_freshness_audit_sha256"
        ],
        "base_audit_sha256": binding["base_audit_sha256"],
        "teacher_shards_all_splits_sha256": binding[
            "teacher_shards_all_splits_sha256"
        ],
        "training_manifest_sha256": training_manifest_sha,
        "model_sha256": model_sha,
        "model_id": model.model_id,
        "frozen_threshold": selected_threshold,
        "safety_enabled": bool(model.safety_enabled),
        "calibration_status": str(calibration_report.get("status", "")),
        "ranker_training_sources": ["train"],
        "ranker_crossfit_source": "train",
        "safety_calibrator_sources": ["train_oof", "calibration.safety_fit"],
        "threshold_selection_source": "calibration.threshold_lock",
        "model_selection_sources": ["train"],
        "locked_holdout_used_for_training_selection_or_threshold": False,
        "model_or_threshold_locked_label_access_count_at_freeze": 0,
        "current_profile_resolved": False,
        "runtime_policy_activated": False,
        "pilot_can_promote_policy": False,
        "fresh_population_acceptance_required": True,
        "minimum_population_valid_overrides": int(
            plan["promotion_boundary"]["minimum_population_valid_overrides"]
        ),
    }
    validate_model_threshold_freeze(freeze, data_contract=data_contract)
    return freeze


def _read_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"M4.3 JSON artifact must be a mapping: {path}")
    return value


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"M4.3 {location} must be a mapping")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--data-contract", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--train", type=Path, action="append", required=True)
    parser.add_argument("--calibration", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output.resolve()
    inputs = {
        args.model.resolve(),
        args.training_manifest.resolve(),
        args.data_contract.resolve(),
        args.plan.resolve(),
        *(path.resolve() for path in args.train),
        *(path.resolve() for path in args.calibration),
    }
    if output in inputs:
        raise ValueError("M4.3 freeze output must be distinct from every input")
    freeze = build_model_threshold_freeze(
        model_path=args.model,
        training_manifest_path=args.training_manifest,
        data_contract_path=args.data_contract,
        plan_path=args.plan,
        repo_root=args.repo_root,
        train=args.train,
        calibration=args.calibration,
    )
    _atomic_json(output, freeze)
    print(json.dumps(freeze, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["build_model_threshold_freeze"]
