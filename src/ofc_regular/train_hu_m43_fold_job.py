"""Prepare the redacted M4.3 contract or fit one deterministic fold job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .hu_m4_joint_model import PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE
from .hu_m43_fold_training import (
    run_fold_job,
    write_fold_cloud_contract,
    write_local_rebind_manifest,
)
from .train_hu_m4_joint_model import parse_thresholds


def _add_training_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--action-score-mode",
        default=PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE,
        choices=[PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE],
    )
    parser.add_argument("--cross-fit-folds", type=int, default=5)
    parser.add_argument("--near-best-margin", type=float, default=0.5)
    parser.add_argument("--minimum-safe-teacher-gain", type=float, default=0.0)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-regularization", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026071801)
    parser.add_argument("--paired-se-floor", type=float, default=0.5)
    parser.add_argument("--paired-huber-alpha", type=float, default=0.9)
    parser.add_argument("--downside-quantile", type=float, default=0.9)
    parser.add_argument("--positive-gain-score-weight", type=float, default=0.25)
    parser.add_argument("--downside-risk-score-weight", type=float, default=0.5)
    parser.add_argument(
        "--ensemble-disagreement-score-weight", type=float, default=0.25
    )
    parser.add_argument("--safety-calibrator-c", type=float, default=0.25)
    parser.add_argument("--safety-fit-ratio", type=float, default=0.5)
    parser.add_argument("--safety-split-seed", type=int, default=2026071802)
    parser.add_argument("--minimum-safety-fit-samples", type=int, default=30)
    parser.add_argument("--minimum-threshold-lock-samples", type=int, default=30)
    parser.add_argument(
        "--thresholds",
        default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.975,0.99,1",
    )
    parser.add_argument("--minimum-calibration-fires", type=int, default=10)
    parser.add_argument("--maximum-false-positive-rate", type=float, default=0.30)
    parser.add_argument("--maximum-p95-loss", type=float, default=25.0)
    parser.add_argument("--maximum-p99-loss", type=float, default=40.0)
    parser.add_argument("--maximum-max-loss", type=float, default=50.0)


def _training_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "cross_fit_folds": args.cross_fit_folds,
        "iterations": args.iterations,
        "max_leaf_nodes": args.max_leaf_nodes,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "paired_se_floor": args.paired_se_floor,
        "paired_huber_alpha": args.paired_huber_alpha,
        "downside_quantile": args.downside_quantile,
        "positive_gain_score_weight": args.positive_gain_score_weight,
        "downside_risk_score_weight": args.downside_risk_score_weight,
        "ensemble_disagreement_score_weight": (
            args.ensemble_disagreement_score_weight
        ),
        "action_score_mode": args.action_score_mode,
        "model_id": args.model_id,
        "near_best_margin": args.near_best_margin,
        "minimum_safe_teacher_gain": args.minimum_safe_teacher_gain,
        "l2_regularization": args.l2_regularization,
        "safety_calibrator_c": args.safety_calibrator_c,
        "safety_fit_ratio": args.safety_fit_ratio,
        "safety_split_seed": args.safety_split_seed,
        "minimum_safety_fit_samples": args.minimum_safety_fit_samples,
        "minimum_threshold_lock_samples": args.minimum_threshold_lock_samples,
        "thresholds": list(parse_thresholds(args.thresholds)),
        "minimum_calibration_fires": args.minimum_calibration_fires,
        "maximum_false_positive_rate": args.maximum_false_positive_rate,
        "maximum_p95_loss": args.maximum_p95_loss,
        "maximum_p99_loss": args.maximum_p99_loss,
        "maximum_max_loss": args.maximum_max_loss,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--train", type=Path, action="append", required=True)
    prepare.add_argument("--calibration", type=Path, action="append", required=True)
    prepare.add_argument("--m43-data-contract", type=Path, required=True)
    prepare.add_argument("--m43-plan", type=Path, required=True)
    prepare.add_argument("--repo-root", type=Path, required=True)
    prepare.add_argument("--predeclared-receipt", type=Path, required=True)
    prepare.add_argument("--output-contract", type=Path, required=True)
    _add_training_config(prepare)

    rebind = commands.add_parser("rebind")
    rebind.add_argument("--run-name", required=True)
    rebind.add_argument("--train", type=Path, action="append", required=True)
    rebind.add_argument("--calibration", type=Path, action="append", required=True)
    rebind.add_argument("--m43-data-contract", type=Path, required=True)
    rebind.add_argument("--m43-plan", type=Path, required=True)
    rebind.add_argument("--repo-root", type=Path, required=True)
    rebind.add_argument("--predeclared-receipt", type=Path, required=True)
    rebind.add_argument("--fold-cloud-contract", type=Path, required=True)
    rebind.add_argument("--run-manifest", type=Path, required=True)
    rebind.add_argument("--source-archive", type=Path, required=True)
    rebind.add_argument("--output-rebind-manifest", type=Path, required=True)
    _add_training_config(rebind)

    run = commands.add_parser("run")
    run.add_argument("--job-index", type=int, required=True)
    run.add_argument("--train", type=Path, action="append", required=True)
    run.add_argument("--calibration", type=Path, action="append", required=True)
    run.add_argument("--fold-cloud-contract", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--run-name", required=True)
    run.add_argument("--source-sha256", required=True)
    run.add_argument("--run-manifest-sha256", required=True)
    run.add_argument("--input-bundle-sha256", required=True)
    run.add_argument("--job-spec-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        payload = write_fold_cloud_contract(
            args.output_contract,
            train_path=args.train,
            calibration_path=args.calibration,
            data_contract_path=args.m43_data_contract,
            plan_path=args.m43_plan,
            repo_root=args.repo_root,
            predeclared_receipt_path=args.predeclared_receipt,
            hyperparameters=_training_config(args),
        )
    elif args.command == "rebind":
        payload = write_local_rebind_manifest(
            args.output_rebind_manifest,
            run_name=args.run_name,
            train_path=args.train,
            calibration_path=args.calibration,
            data_contract_path=args.m43_data_contract,
            plan_path=args.m43_plan,
            repo_root=args.repo_root,
            predeclared_receipt_path=args.predeclared_receipt,
            fold_cloud_contract_path=args.fold_cloud_contract,
            run_manifest_path=args.run_manifest,
            source_archive_path=args.source_archive,
            hyperparameters=_training_config(args),
        )
    else:
        payload = run_fold_job(
            job_index=args.job_index,
            train_path=args.train,
            calibration_path=args.calibration,
            fold_cloud_contract_path=args.fold_cloud_contract,
            output_dir=args.output_dir,
            run_name=args.run_name,
            source_sha256=args.source_sha256,
            run_manifest_sha256=args.run_manifest_sha256,
            input_bundle_sha256=args.input_bundle_sha256,
            job_spec_sha256=args.job_spec_sha256,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
