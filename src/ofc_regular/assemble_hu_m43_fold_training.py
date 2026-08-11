"""Assemble the exact 30 M4.3 fold artifacts on the trusted local host.

Fold workers never receive the sealed data contract or the aborted-attempt
receipt.  This command rebinds the redacted cloud projection to those local
files, verifies the immutable Spot run and every fold artifact, then performs
the single OOF safety/calibration pass.  The final manifest is published last.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hu_m43_fold_training import (
    M43_FOLD_JOB_COUNT,
    TRAINING_HYPERPARAMETER_KEYS,
    _integer,
    _mapping,
    _reject_sensitive_strings,
    _require_sha256,
    _sequence,
    _training_hyperparameters,
    _verify_frozen_dependencies,
    _verify_frozen_process_environment,
    load_fold_artifact_provider,
    load_fold_cloud_contract,
    verify_local_rebind_manifest,
    verify_fold_cloud_contract_against_local,
)
from .hu_m43_pilot_contract import canonical_manifest_sha256
from .train_hu_m4_joint_model import (
    HU_M4_JOINT_TRAINING_MANIFEST_SCHEMA,
    _sha256,
    parse_thresholds,
    train_from_files,
)


M43_FOLD_SPOT_RUN_MANIFEST_SCHEMA = "hu_m43_fold_spot_run_manifest_v1"


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_run_inputs(
    run_inputs: Mapping[str, Any], projection: Mapping[str, Any]
) -> None:
    if set(run_inputs) != {"train", "calibration", "input_bundle_sha256"}:
        raise ValueError("M4.3 Spot run input split/key set changed")
    if run_inputs.get("input_bundle_sha256") != projection.get(
        "input_bundle_sha256"
    ):
        raise ValueError("M4.3 Spot run input bundle digest mismatch")
    projection_inputs = _mapping(projection.get("inputs"), "projection.inputs")
    for split in ("train", "calibration"):
        expected_rows = _sequence(
            projection_inputs.get(split), f"projection.inputs.{split}"
        )
        actual_rows = _sequence(run_inputs.get(split), f"run.inputs.{split}")
        if len(actual_rows) != len(expected_rows):
            raise ValueError(f"M4.3 Spot run {split} shard coverage changed")
        for index, (actual, expected) in enumerate(
            zip(actual_rows, expected_rows, strict=True)
        ):
            actual_map = _mapping(actual, f"run.inputs.{split}[{index}]")
            expected_map = _mapping(expected, f"projection.inputs.{split}[{index}]")
            if set(actual_map) != set(expected_map) | {"uri"}:
                raise ValueError(
                    f"M4.3 Spot run {split} shard key set changed at {index}"
                )
            for key in ("index", "bytes", "rows"):
                if _integer(
                    actual_map.get(key), f"run.inputs.{split}[{index}].{key}"
                ) != _integer(
                    expected_map.get(key),
                    f"projection.inputs.{split}[{index}].{key}",
                ):
                    raise ValueError(
                        f"M4.3 Spot run {split} integer provenance changed at {index}"
                    )
            _require_sha256(
                actual_map.get("sha256"),
                f"run.inputs.{split}[{index}].sha256",
            )
            if {key: actual_map.get(key) for key in expected_map} != dict(
                expected_map
            ):
                raise ValueError(
                    f"M4.3 Spot run {split} shard provenance changed at {index}"
                )
            uri = actual_map.get("uri")
            if not isinstance(uri, str) or not uri.startswith("gs://"):
                raise ValueError(
                    f"M4.3 Spot run {split} transport URI invalid at {index}"
                )


def _verify_run_jobs(
    manifest: Mapping[str, Any], projection: Mapping[str, Any]
) -> None:
    jobs = _sequence(manifest.get("jobs"), "run.jobs")
    fold_plan = _mapping(projection.get("fold_plan"), "projection.fold_plan")
    expected_jobs = _sequence(fold_plan.get("jobs"), "projection.fold_plan.jobs")
    if len(jobs) != M43_FOLD_JOB_COUNT or len(expected_jobs) != M43_FOLD_JOB_COUNT:
        raise ValueError("M4.3 Spot run must bind exact 30-job coverage")
    run_name = manifest.get("run_name")
    bucket = manifest.get("bucket")
    for index, (raw_job, raw_expected) in enumerate(
        zip(jobs, expected_jobs, strict=True)
    ):
        job = _mapping(raw_job, f"run.jobs[{index}]")
        expected = _mapping(raw_expected, f"projection.jobs[{index}]")
        expected_spec = dict(expected)
        expected_spec_sha = _require_sha256(
            expected_spec.pop("job_spec_sha256", None),
            f"projection.jobs[{index}].job_spec_sha256",
        )
        if canonical_manifest_sha256(expected_spec) != expected_spec_sha:
            raise ValueError(f"M4.3 projection job spec digest invalid at {index}")
        if _integer(job.get("job_index"), f"run.jobs[{index}].job_index") != index:
            raise ValueError(f"M4.3 Spot run job index changed at {index}")
        actual_job_spec = _mapping(
            job.get("job_spec"), f"run.jobs[{index}].job_spec"
        )
        expected_inner = expected_spec.get("inner_fold")
        actual_inner = job.get("inner_fold")
        inner_matches = (
            actual_inner is None
            if expected_inner is None
            else _integer(actual_inner, f"run.jobs[{index}].inner_fold")
            == _integer(expected_inner, f"projection.jobs[{index}].inner_fold")
        )
        if (
            job.get("job_kind") != expected_spec.get("kind")
            or _integer(job.get("outer_fold"), f"run.jobs[{index}].outer_fold")
            != expected_spec.get("outer_fold")
            or not inner_matches
            or actual_job_spec != expected_spec
            or canonical_manifest_sha256(actual_job_spec) != expected_spec_sha
            or job.get("job_spec_sha256") != expected_spec_sha
            or _integer(job.get("shard_index"), f"run.jobs[{index}].shard_index")
            != index // 4
        ):
            raise ValueError(f"M4.3 Spot run full job identity changed at {index}")
        expected_prefix = (
            f"gs://{bucket}/runs/{run_name}/results/job-{index:02d}"
        )
        if (
            job.get("result_prefix") != expected_prefix
            or job.get("done_uri") != f"{expected_prefix}/DONE.json"
        ):
            raise ValueError(f"M4.3 Spot run job URI changed at {index}")


def verify_spot_run_manifest(
    *,
    run_manifest_path: str | Path,
    fold_cloud_contract_path: str | Path,
    expected_run_name: str,
    expected_run_manifest_sha256: str,
    expected_source_sha256: str,
    hyperparameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the external immutable run before consuming any estimator."""

    source = Path(run_manifest_path).resolve()
    projection_path = Path(fold_cloud_contract_path).resolve()
    expected_run_manifest_sha256 = _require_sha256(
        expected_run_manifest_sha256, "expected run manifest SHA-256"
    )
    expected_source_sha256 = _require_sha256(
        expected_source_sha256, "expected source SHA-256"
    )
    if _sha256(source) != expected_run_manifest_sha256:
        raise ValueError("M4.3 Spot run manifest file SHA-256 mismatch")
    manifest = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict):
        raise ValueError("M4.3 Spot run manifest must be an object")
    _reject_sensitive_strings(manifest, label="M4.3 Spot run manifest")
    if (
        manifest.get("schema") != M43_FOLD_SPOT_RUN_MANIFEST_SCHEMA
        or manifest.get("status") != "frozen"
        or manifest.get("run_name") != expected_run_name
        or manifest.get("current_profile_mutated") is not False
        or manifest.get("no_runtime_activation") is not True
        or _integer(manifest.get("job_count"), "run.job_count")
        != M43_FOLD_JOB_COUNT
    ):
        raise ValueError("M4.3 Spot run lifecycle identity is invalid")
    if not isinstance(manifest.get("bucket"), str) or not manifest.get("bucket"):
        raise ValueError("M4.3 Spot run bucket is missing")
    source_binding = _mapping(manifest.get("source"), "run.source")
    if source_binding.get("sha256") != expected_source_sha256:
        raise ValueError("M4.3 Spot run source SHA-256 mismatch")
    _require_sha256(source_binding.get("fold_worker_sha256"), "fold worker SHA-256")
    _require_sha256(source_binding.get("assembler_sha256"), "assembler SHA-256")

    projection = load_fold_cloud_contract(projection_path)
    cloud_binding = _mapping(manifest.get("cloud_contract"), "run.cloud_contract")
    if set(cloud_binding) != {
        "uri",
        "bytes",
        "file_sha256",
        "contract_sha256",
        "predeclared_training_receipt_sha256",
        "train_identity_sha256",
        "fold_plan_sha256",
    }:
        raise ValueError("M4.3 Spot run cloud binding key set changed")
    fold_plan = _mapping(projection.get("fold_plan"), "projection.fold_plan")
    expected_cloud_binding = {
        "file_sha256": _sha256(projection_path),
        "contract_sha256": projection.get("contract_sha256"),
        "predeclared_training_receipt_sha256": projection.get(
            "predeclared_training_receipt_sha256"
        ),
        "train_identity_sha256": fold_plan.get("train_identity_sha256"),
        "fold_plan_sha256": fold_plan.get("fold_plan_sha256"),
    }
    for key, expected in expected_cloud_binding.items():
        if cloud_binding.get(key) != expected:
            raise ValueError(f"M4.3 Spot run cloud binding changed: {key}")
    normalized_hyperparameters = _training_hyperparameters(hyperparameters)
    projection_dependencies = _verify_frozen_dependencies(
        projection.get("dependencies")
    )
    if manifest.get("dependencies") != projection_dependencies:
        raise ValueError("M4.3 Spot run dependency versions changed")
    if manifest.get("process_environment") != projection.get(
        "process_environment"
    ):
        raise ValueError("M4.3 Spot run process environment changed")
    if manifest.get("training_config") != normalized_hyperparameters:
        raise ValueError("M4.3 Spot run full training configuration changed")
    if manifest.get("training_config_sha256") != canonical_manifest_sha256(
        normalized_hyperparameters
    ):
        raise ValueError("M4.3 Spot run training configuration digest mismatch")
    _verify_run_inputs(
        _mapping(manifest.get("inputs"), "run.inputs"), projection
    )
    _verify_run_jobs(manifest, projection)
    compute = _mapping(manifest.get("compute"), "run.compute")
    if (
        _integer(compute.get("shard_count"), "compute.shard_count") != 8
        or _integer(compute.get("jobs_per_shard"), "compute.jobs_per_shard") != 4
        or compute.get("spot") is not True
        or compute.get("auto_delete") is not True
        or compute.get("termination_action") != "DELETE"
    ):
        raise ValueError("M4.3 Spot compute lifecycle contract changed")
    return manifest


def assemble_from_fold_artifacts(
    *,
    fold_artifacts_dir: str | Path,
    train_path: str | Path | Sequence[str | Path],
    calibration_path: str | Path | Sequence[str | Path],
    data_contract_path: str | Path,
    plan_path: str | Path,
    fold_cloud_contract_path: str | Path,
    repo_root: str | Path,
    run_manifest_path: str | Path,
    predeclared_receipt_path: str | Path,
    local_rebind_manifest_path: str | Path,
    source_archive_path: str | Path,
    output_model: str | Path,
    manifest_output: str | Path,
    run_name: str,
    run_manifest_sha256: str,
    source_sha256: str,
    hyperparameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble one immutable run, publishing the external manifest last."""

    normalized = _training_hyperparameters(hyperparameters)
    _verify_frozen_process_environment(
        load_fold_cloud_contract(fold_cloud_contract_path).get(
            "process_environment"
        )
    )
    projection = verify_fold_cloud_contract_against_local(
        fold_cloud_contract_path=fold_cloud_contract_path,
        train_path=train_path,
        calibration_path=calibration_path,
        data_contract_path=data_contract_path,
        plan_path=plan_path,
        repo_root=repo_root,
        predeclared_receipt_path=predeclared_receipt_path,
        hyperparameters=normalized,
    )
    run_manifest = verify_spot_run_manifest(
        run_manifest_path=run_manifest_path,
        fold_cloud_contract_path=fold_cloud_contract_path,
        expected_run_name=run_name,
        expected_run_manifest_sha256=run_manifest_sha256,
        expected_source_sha256=source_sha256,
        hyperparameters=normalized,
    )
    local_rebind = verify_local_rebind_manifest(
        local_rebind_manifest_path=local_rebind_manifest_path,
        run_name=run_name,
        train_path=train_path,
        calibration_path=calibration_path,
        data_contract_path=data_contract_path,
        plan_path=plan_path,
        repo_root=repo_root,
        predeclared_receipt_path=predeclared_receipt_path,
        fold_cloud_contract_path=fold_cloud_contract_path,
        run_manifest_path=run_manifest_path,
        source_archive_path=source_archive_path,
        hyperparameters=normalized,
    )
    provider = load_fold_artifact_provider(
        artifacts_dir=fold_artifacts_dir,
        fold_cloud_contract_path=fold_cloud_contract_path,
        train_path=train_path,
        calibration_path=calibration_path,
    )
    assembly = dict(provider.assembly_receipt)
    if (
        assembly.get("run_name") != run_name
        or assembly.get("run_manifest_sha256") != run_manifest_sha256
        or assembly.get("source_sha256") != source_sha256
    ):
        raise ValueError("M4.3 fold artifacts do not belong to the verified run")
    assembly.update(
        {
            "run_manifest_schema": run_manifest.get("schema"),
            "training_config_sha256": run_manifest.get(
                "training_config_sha256"
            ),
            "predeclared_training_receipt_sha256": projection.get(
                "predeclared_training_receipt_sha256"
            ),
            "fold_plan_sha256": _mapping(
                projection.get("fold_plan"), "projection.fold_plan"
            ).get("fold_plan_sha256"),
            "local_sealed_rebind_manifest_sha256": local_rebind.get(
                "rebind_sha256"
            ),
        }
    )

    final_model = Path(output_model).resolve()
    final_manifest = Path(manifest_output).resolve()
    if final_model == final_manifest:
        raise ValueError("M4.3 model and manifest outputs must be distinct")
    if final_model.exists() or final_manifest.exists():
        raise FileExistsError("M4.3 assembler refuses to overwrite an output")
    final_model.parent.mkdir(parents=True, exist_ok=True)
    final_manifest.parent.mkdir(parents=True, exist_ok=True)
    staging = final_manifest.parent / f".m43-assemble-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"M4.3 assembler staging already exists: {staging}")
    staging.mkdir()
    staged_model = staging / "model.pkl"
    staged_manifest = staging / "training_manifest.json"
    try:
        result = train_from_files(
            train_path=train_path,
            calibration_path=calibration_path,
            locked_holdout_path=None,
            output_model=staged_model,
            manifest_output=staged_manifest,
            m43_data_contract_path=data_contract_path,
            m43_plan_path=plan_path,
            m43_repo_root=repo_root,
            m43_fold_estimator_provider=provider,
            m43_distributed_fold_assembly=assembly,
            model_id=str(normalized["model_id"]),
            near_best_margin=float(normalized["near_best_margin"]),
            minimum_safe_teacher_gain=float(
                normalized["minimum_safe_teacher_gain"]
            ),
            iterations=int(normalized["iterations"]),
            max_leaf_nodes=int(normalized["max_leaf_nodes"]),
            learning_rate=float(normalized["learning_rate"]),
            l2_regularization=float(normalized["l2_regularization"]),
            seed=int(normalized["seed"]),
            action_score_mode=str(normalized["action_score_mode"]),
            cross_fit_folds=int(normalized["cross_fit_folds"]),
            paired_se_floor=float(normalized["paired_se_floor"]),
            paired_huber_alpha=float(normalized["paired_huber_alpha"]),
            downside_quantile=float(normalized["downside_quantile"]),
            positive_gain_score_weight=float(
                normalized["positive_gain_score_weight"]
            ),
            downside_risk_score_weight=float(
                normalized["downside_risk_score_weight"]
            ),
            ensemble_disagreement_score_weight=float(
                normalized["ensemble_disagreement_score_weight"]
            ),
            safety_calibrator_c=float(normalized["safety_calibrator_c"]),
            safety_fit_ratio=float(normalized["safety_fit_ratio"]),
            safety_split_seed=int(normalized["safety_split_seed"]),
            minimum_safety_fit_samples=int(
                normalized["minimum_safety_fit_samples"]
            ),
            minimum_threshold_lock_samples=int(
                normalized["minimum_threshold_lock_samples"]
            ),
            thresholds=tuple(float(value) for value in normalized["thresholds"]),
            minimum_calibration_fires=int(
                normalized["minimum_calibration_fires"]
            ),
            maximum_false_positive_rate=float(
                normalized["maximum_false_positive_rate"]
            ),
            maximum_p95_loss=float(normalized["maximum_p95_loss"]),
            maximum_p99_loss=float(normalized["maximum_p99_loss"]),
            maximum_max_loss=float(normalized["maximum_max_loss"]),
        )
        provider.assert_complete()
        if result.get("schema") != HU_M4_JOINT_TRAINING_MANIFEST_SCHEMA:
            raise ValueError("M4.3 assembled training manifest schema mismatch")
        if result.get("distributed_fold_assembly") != assembly:
            raise ValueError("M4.3 assembled manifest lost fold provenance")
        if result.get("locked_holdout_used_for_threshold_or_training") is not False:
            raise ValueError("M4.3 assembler unexpectedly consumed sealed labels")
        runtime_lock = _mapping(result.get("runtime_lock"), "runtime_lock")
        staged_model_sha = _sha256(staged_model)
        if (
            runtime_lock.get("candidate_model_sha256") != staged_model_sha
            or runtime_lock.get("safety_model_sha256") != staged_model_sha
        ):
            raise ValueError("M4.3 assembled model runtime hash mismatch")
        os.replace(staged_model, final_model)
        final_result = dict(result)
        final_runtime_lock = dict(runtime_lock)
        final_runtime_lock["model_path"] = str(final_model)
        final_result["runtime_lock"] = final_runtime_lock
        _write_json_atomic(final_manifest, final_result)
        return final_result
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-artifacts-dir", type=Path, required=True)
    parser.add_argument("--train", type=Path, action="append", required=True)
    parser.add_argument("--calibration", type=Path, action="append", required=True)
    parser.add_argument("--m43-data-contract", type=Path, required=True)
    parser.add_argument("--m43-plan", type=Path, required=True)
    parser.add_argument("--fold-cloud-contract", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--predeclared-receipt", type=Path, required=True)
    parser.add_argument(
        "--local-sealed-rebind-manifest", type=Path, required=True
    )
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-manifest-sha256", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--action-score-mode", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--cross-fit-folds", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--max-leaf-nodes", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--l2-regularization", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--paired-se-floor", type=float, required=True)
    parser.add_argument("--paired-huber-alpha", type=float, required=True)
    parser.add_argument("--downside-quantile", type=float, required=True)
    parser.add_argument("--positive-gain-score-weight", type=float, required=True)
    parser.add_argument("--downside-risk-score-weight", type=float, required=True)
    parser.add_argument(
        "--ensemble-disagreement-score-weight", type=float, required=True
    )
    parser.add_argument("--near-best-margin", type=float, required=True)
    parser.add_argument("--minimum-safe-teacher-gain", type=float, required=True)
    parser.add_argument("--safety-calibrator-c", type=float, required=True)
    parser.add_argument("--safety-fit-ratio", type=float, required=True)
    parser.add_argument("--safety-split-seed", type=int, required=True)
    parser.add_argument("--minimum-safety-fit-samples", type=int, required=True)
    parser.add_argument("--minimum-threshold-lock-samples", type=int, required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--minimum-calibration-fires", type=int, required=True)
    parser.add_argument("--maximum-false-positive-rate", type=float, required=True)
    parser.add_argument("--maximum-p95-loss", type=float, required=True)
    parser.add_argument("--maximum-p99-loss", type=float, required=True)
    parser.add_argument("--maximum-max-loss", type=float, required=True)
    return parser


def _hyperparameters(args: argparse.Namespace) -> dict[str, Any]:
    values = {key: getattr(args, key) for key in TRAINING_HYPERPARAMETER_KEYS}
    values["thresholds"] = list(parse_thresholds(args.thresholds))
    return _training_hyperparameters(values)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    result = assemble_from_fold_artifacts(
        fold_artifacts_dir=args.fold_artifacts_dir,
        train_path=args.train,
        calibration_path=args.calibration,
        data_contract_path=args.m43_data_contract,
        plan_path=args.m43_plan,
        fold_cloud_contract_path=args.fold_cloud_contract,
        repo_root=args.repo_root,
        run_manifest_path=args.run_manifest,
        predeclared_receipt_path=args.predeclared_receipt,
        local_rebind_manifest_path=args.local_sealed_rebind_manifest,
        source_archive_path=args.source_archive,
        output_model=args.output_model,
        manifest_output=args.manifest_output,
        run_name=args.run_name,
        run_manifest_sha256=args.run_manifest_sha256,
        source_sha256=args.source_sha256,
        hyperparameters=_hyperparameters(args),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "M43_FOLD_SPOT_RUN_MANIFEST_SCHEMA",
    "assemble_from_fold_artifacts",
    "verify_spot_run_manifest",
]
