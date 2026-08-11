"""Prepare or run one train-only M4.3 Attempt02 v4 fold job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .hu_m43_attempt02_fold_training import (
    Attempt02V4TrainingConfig,
    run_attempt02_fold_job,
    write_attempt02_fold_cloud_contract,
)
from .hu_m43_joint_model_v4 import V4FoldWorkerConfig


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--train", type=Path, action="append", required=True)
    prepare.add_argument("--output-contract", type=Path, required=True)
    prepare.add_argument("--model-id", default="hu-m43-t1-v4-attempt02")
    prepare.add_argument("--iterations", type=int, default=150)
    prepare.add_argument("--max-leaf-nodes", type=int, default=31)
    prepare.add_argument("--learning-rate", type=float, default=0.05)
    prepare.add_argument("--paired-se-floor", type=float, default=0.50)
    prepare.add_argument("--huber-alpha", type=float, default=0.90)

    run = commands.add_parser("run")
    run.add_argument("--job-index", type=int, required=True)
    run.add_argument("--train", type=Path, action="append", required=True)
    run.add_argument("--fold-cloud-contract", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--run-name", required=True)
    run.add_argument("--source-sha256", required=True)
    run.add_argument("--run-manifest-sha256", required=True)
    run.add_argument("--input-bundle-sha256", required=True)
    run.add_argument("--job-spec-sha256", required=True)
    return parser.parse_args(argv)


def _config(args: argparse.Namespace) -> Attempt02V4TrainingConfig:
    return Attempt02V4TrainingConfig(
        model_id=args.model_id,
        worker=V4FoldWorkerConfig(
            paired_se_floor=args.paired_se_floor,
            huber_alpha=args.huber_alpha,
            iterations=args.iterations,
            max_leaf_nodes=args.max_leaf_nodes,
            learning_rate=args.learning_rate,
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        result = write_attempt02_fold_cloud_contract(
            args.output_contract,
            train_path=args.train,
            config=_config(args),
        )
    else:
        result = run_attempt02_fold_job(
            job_index=args.job_index,
            train_path=args.train,
            fold_cloud_contract_path=args.fold_cloud_contract,
            output_dir=args.output_dir,
            run_name=args.run_name,
            source_sha256=args.source_sha256,
            run_manifest_sha256=args.run_manifest_sha256,
            input_bundle_sha256=args.input_bundle_sha256,
            job_spec_sha256=args.job_spec_sha256,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main", "parse_args"]
