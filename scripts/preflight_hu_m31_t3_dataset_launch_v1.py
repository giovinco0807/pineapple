"""Write the cloud-neutral M3.1 dataset launch preflight receipt."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from ofc_regular import hu_m31_t3_dataset_launch_preflight_v1 as preflight


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create-only M3.1 dataset v1 launch preflight"
    )
    parser.add_argument("--dataset-run-name", required=True)
    parser.add_argument("--selection-receipt", required=True)
    parser.add_argument("--planned-source-binding", required=True)
    parser.add_argument("--planned-controller-root", required=True)
    parser.add_argument("--planned-supervisor-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = preflight.write_launch_preflight(
            output_path=args.output,
            dataset_run_name=args.dataset_run_name,
            selection_receipt_path=args.selection_receipt,
            planned_source_binding_path=args.planned_source_binding,
            planned_controller_root=args.planned_controller_root,
            planned_supervisor_root=args.planned_supervisor_root,
        )
        print(preflight.canonical_bytes(receipt).decode("ascii"))
    except Exception as exc:  # pragma: no cover - CLI boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
