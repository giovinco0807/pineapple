"""Build a local-only T3 performance-development v2 dry-run receipt.

Both inputs are caller-supplied read-only observations.  This entrypoint has no
cloud adapter and cannot package or launch a VM.  ``--output`` is optional and,
when used, is an immutable local receipt created exactly once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ofc_regular import (
    hu_m31_t3_step6d_performance_development_v2_contract as contract_v1,
)
from ofc_regular import (
    hu_m31_t3_step6d_performance_development_v2_preflight as preflight,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-observation", type=Path, required=True)
    parser.add_argument("--runtime-observation", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = preflight.build_dry_run_receipt_from_files(
        image_path=args.image_observation,
        runtime_path=args.runtime_observation,
    )
    if args.output is not None:
        preflight.write_once(args.output, receipt)
    print(
        json.dumps(
            {
                "schema": receipt["schema"],
                "status": receipt["status"],
                "receipt_sha256": contract_v1.canonical_sha256(receipt),
                "output_written": args.output is not None,
                "cloud_executable": False,
                "launch_authorized": False,
                "cloud_mutated": False,
                "instances_created": False,
                "current_profile_changed": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
