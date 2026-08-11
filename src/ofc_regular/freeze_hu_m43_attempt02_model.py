"""Freeze an M4.3 Attempt02 v4 model before inherited-lock access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hu_m43_attempt02_contract import (
    _find_consumption_markers,
    load_and_validate_attempt02_plan,
)
from .hu_m43_attempt02_lifecycle import (
    M43_ATTEMPT02_FREEZE_SCHEMA,
    M43_ATTEMPT02_FREEZE_STATUS,
    build_calibration_role_binding,
    build_fresh_bindings,
    build_fresh_identity_summaries,
    file_sha256,
    read_json_mapping,
    resolve_contract_path,
    self_digest,
    validate_attempt02_data_contract,
    validate_attempt02_freeze,
    validate_v4_training_manifest,
    write_immutable_json,
)
from .hu_m43_joint_model_v4 import HuM43JointModelV4


def build_attempt02_model_threshold_freeze(
    *,
    model_path: str | Path,
    training_manifest_path: str | Path,
    data_contract_path: str | Path,
    plan_path: str | Path,
    population_plan_path: str | Path,
    repo_root: str | Path,
    train: Sequence[str | Path],
    calibration: Sequence[str | Path],
) -> dict[str, Any]:
    """Validate the complete fresh-data/model chain without opening locked JSONL."""

    root = Path(repo_root).resolve()
    model_source = Path(model_path).resolve()
    manifest_source = Path(training_manifest_path).resolve()
    contract_source = Path(data_contract_path).resolve()
    plan_source = Path(plan_path).resolve()
    population_plan_source = Path(population_plan_path).resolve()
    all_sources = {
        model_source,
        manifest_source,
        contract_source,
        plan_source,
        population_plan_source,
        *(Path(path).resolve() for path in train),
        *(Path(path).resolve() for path in calibration),
    }
    expected_sources = 5 + len(train) + len(calibration)
    if len(all_sources) != expected_sources:
        raise ValueError("Attempt02 freeze inputs must be distinct")
    if not all(path.is_file() for path in all_sources):
        raise ValueError("Attempt02 freeze input file is missing")

    plan = load_and_validate_attempt02_plan(plan_source)
    plan_sha = file_sha256(plan_source)
    # Local import avoids coupling the locked-free lifecycle module to the
    # population evaluator at import time.  The full acceptance plan must be
    # fixed before the one-shot inherited holdout can reveal diagnostics.
    from .validate_hu_m43_attempt02_acceptance import (
        load_and_validate_attempt02_population_plan,
    )

    population_plan = load_and_validate_attempt02_population_plan(
        population_plan_source
    )
    if population_plan["attempt02_training_plan"]["file_sha256"] != plan_sha:
        raise ValueError("Attempt02 population plan targets another training plan")
    population_plan_sha = file_sha256(population_plan_source)
    contract = read_json_mapping(contract_source, "Attempt02 data contract")
    fresh_bindings = build_fresh_bindings(
        repo_root=root,
        train=train,
        calibration=calibration,
    )
    fresh_identities = build_fresh_identity_summaries(
        train=train,
        calibration=calibration,
    )
    roles = build_calibration_role_binding(plan=plan, calibration=calibration)
    training_binding = validate_attempt02_data_contract(
        contract,
        plan=plan,
        plan_sha256=plan_sha,
        fresh_bindings=fresh_bindings,
        fresh_identity_summaries=fresh_identities,
        calibration_role_binding=roles,
    )

    canonical_marker = resolve_contract_path(
        root,
        contract["global_locked_consumption"]["canonical_marker_path"],
        "Attempt02 canonical consumption marker",
    )
    if canonical_marker.exists():
        raise FileExistsError(
            f"Attempt02 inherited locked holdout is already consumed: {canonical_marker}"
        )
    locked_identity = str(contract["inherited_locked"]["identity_sha256"])
    marker_hits = _find_consumption_markers(root, locked_identity)
    if marker_hits:
        raise FileExistsError(
            "Attempt02 inherited locked identity already has a global marker"
        )

    model_sha = file_sha256(model_source)
    training = read_json_mapping(manifest_source, "Attempt02 v4 training manifest")
    threshold = validate_v4_training_manifest(
        training,
        model_sha256=model_sha,
        expected_binding=training_binding,
        data_contract_file_sha256=file_sha256(contract_source),
        contract=contract,
        plan=plan,
    )
    model = HuM43JointModelV4.load(model_source, expected_sha256=model_sha)
    if model.model_id != training.get("model_id"):
        raise ValueError("Attempt02 v4 model_id disagrees with training manifest")
    if model.safety_enabled is not True or model.safety_estimator is None:
        raise ValueError("Attempt02 v4 artifact safety is not enabled")
    if float(model.safety_threshold) != float(threshold):
        raise ValueError("Attempt02 v4 artifact threshold disagrees with manifest")
    model_binding = model.manifest
    if not isinstance(model_binding, Mapping) or (
        model_binding.get("schema") != "hu_m43_attempt02_v4_model_binding_v1"
        or model_binding.get("data_contract_sha256")
        != contract.get("contract_sha256")
        or model_binding.get("calibration_binding_schema")
        != "hu_m43_attempt02_v4_calibration_binding_v1"
        or model_binding.get("current_profile_mutated") is not False
        or model_binding.get("runtime_policy_activated") is not False
        or model_binding.get("full_replacement") is not False
        or model_binding.get("runtime_teacher_inputs") is not False
    ):
        raise ValueError("Attempt02 v4 embedded model binding mismatch")

    selected = training["threshold_selection"]["selected_metrics"]
    inherited = contract["inherited_locked"]
    freeze: dict[str, Any] = {
        "schema": M43_ATTEMPT02_FREEZE_SCHEMA,
        "status": M43_ATTEMPT02_FREEZE_STATUS,
        "plan_sha256": plan_sha,
        "population_plan_file_sha256": population_plan_sha,
        "population_plan_resolved_path": str(population_plan_source),
        "data_contract_sha256": contract["contract_sha256"],
        "data_contract_file_sha256": file_sha256(contract_source),
        "data_contract_resolved_path": str(contract_source),
        "training_manifest_sha256": file_sha256(manifest_source),
        "training_manifest_resolved_path": str(manifest_source),
        "model_sha256": model_sha,
        "model_id": model.model_id,
        "model_schema": model.schema,
        "frozen_threshold": float(threshold),
        "safety_enabled": True,
        "precalibration_status": "go",
        "threshold_lock_status": "go",
        "minimum_threshold_lock_fires": int(
            plan["calibration_partition"][
                "minimum_threshold_lock_fires_for_positive_pilot_signal"
            ]
        ),
        "selected_threshold_metrics": dict(selected),
        "fresh_teacher_binding": training_binding,
        "inherited_locked_binding": {
            key: inherited[key]
            for key in (
                "path",
                "records",
                "bytes",
                "file_sha256",
                "canonical_rows_sha256",
                "identity_sha256",
                "ordered_shards_sha256",
            )
        },
        "locked_status": "inherited_unopened",
        "locked_content_access_count_at_freeze": 0,
        "canonical_consumption_marker_relative_path": contract[
            "global_locked_consumption"
        ]["canonical_marker_path"],
        "canonical_consumption_marker_resolved_path": str(canonical_marker),
        "activation_guards": {
            "current_profile_changed": False,
            "current_profile_resolved": False,
            "runtime_policy_activated": False,
            "full_replacement_enabled": False,
        },
        "fresh_population_acceptance_required": True,
        "minimum_population_valid_overrides": 300,
    }
    freeze["freeze_sha256"] = self_digest(freeze, "freeze_sha256")
    validate_attempt02_freeze(freeze, contract=contract)
    return freeze


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--data-contract", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--population-plan", type=Path, required=True)
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
        args.population_plan.resolve(),
        *(path.resolve() for path in args.train),
        *(path.resolve() for path in args.calibration),
    }
    if output in inputs:
        raise ValueError("Attempt02 freeze output must be distinct from every input")
    freeze = build_attempt02_model_threshold_freeze(
        model_path=args.model,
        training_manifest_path=args.training_manifest,
        data_contract_path=args.data_contract,
        plan_path=args.plan,
        population_plan_path=args.population_plan,
        repo_root=args.repo_root,
        train=args.train,
        calibration=args.calibration,
    )
    write_immutable_json(output, freeze)
    print(json.dumps(freeze, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["build_attempt02_model_threshold_freeze"]
