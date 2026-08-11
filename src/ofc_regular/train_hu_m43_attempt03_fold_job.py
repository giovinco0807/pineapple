"""Run one immutable Attempt03 v5 base-fold job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .hu_m43_attempt03_training import run_attempt03_fold_job


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-index", type=int, required=True)
    parser.add_argument("--inherited-train", type=Path, required=True)
    parser.add_argument("--fresh-train-fit", type=Path, required=True)
    parser.add_argument("--fold-cloud-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--run-manifest-sha256", required=True)
    parser.add_argument("--input-bundle-sha256", required=True)
    parser.add_argument("--job-spec-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_attempt03_fold_job(
        job_index=args.job_index,
        inherited_train_path=args.inherited_train,
        fresh_train_fit_path=args.fresh_train_fit,
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
