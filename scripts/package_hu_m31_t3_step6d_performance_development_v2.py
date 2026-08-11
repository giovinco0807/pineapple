"""Audit or locally package the T3 performance-development v2 tail probe.

This command has no cloud adapter, authorization command, or launch command.
``--inventory-only`` is read-only.  Package mode creates an immutable local
directory whose startup marker exits immediately and whose manifest states
``cloud_executable=false``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ofc_regular import (
    hu_m31_t3_step6d_performance_development_v2_local_package as package_v1,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--output-parent", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument(
        "--candidate-library", type=Path, default=package_v1.DEFAULT_CANDIDATE_PATH
    )
    parser.add_argument(
        "--reference-library", type=Path, default=package_v1.DEFAULT_REFERENCE_PATH
    )
    parser.add_argument(
        "--feature-encoder", type=Path, default=package_v1.DEFAULT_FEATURE_PATH
    )
    parser.add_argument(
        "--full100-plan", type=Path, default=package_v1.DEFAULT_PLAN_PATH
    )
    parser.add_argument(
        "--accepted-summary",
        type=Path,
        default=package_v1.DEFAULT_ACCEPTED_SUMMARY_PATH,
    )
    parser.add_argument(
        "--accepted-validation",
        type=Path,
        default=package_v1.DEFAULT_ACCEPTED_VALIDATION_PATH,
    )
    parser.add_argument("--root-dir", type=Path, default=package_v1.DEFAULT_ROOT_DIR)
    parser.add_argument(
        "--rearm2-root-dir", type=Path, default=package_v1.DEFAULT_REARM2_ROOT_DIR
    )
    parser.add_argument("--dry-run-receipt", type=Path)
    parser.add_argument(
        "--repository-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    common = {
        "candidate_library": args.candidate_library,
        "reference_library": args.reference_library,
        "feature_encoder": args.feature_encoder,
        "full100_plan_path": args.full100_plan,
        "accepted_summary_path": args.accepted_summary,
        "accepted_validation_path": args.accepted_validation,
        "root_dir": args.root_dir,
        "rearm2_root_dir": args.rearm2_root_dir,
    }
    if args.inventory_only:
        if args.output_parent is not None or args.run_name is not None:
            raise ValueError("inventory-only mode does not accept an output or run name")
        result = package_v1.audit_source_inventory(
            **common,
            dry_run_receipt_path=args.dry_run_receipt,
        )
    else:
        if args.output_parent is None or args.run_name is None:
            raise ValueError("package mode requires --output-parent and --run-name")
        if args.dry_run_receipt is None:
            raise ValueError(
                "package mode requires an explicit fresh --dry-run-receipt"
            )
        result = package_v1.package_local(
            **common,
            output_parent=args.output_parent,
            run_name=args.run_name,
            dry_run_receipt_path=args.dry_run_receipt,
            repository_root=args.repository_root,
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
