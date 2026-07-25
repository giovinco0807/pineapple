"""Thin CLI for the two-phase fresh full-100 local preparation contract."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from ofc_regular import hu_m31_t3_step6d_full100_wave_plan_v2 as wave_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_run_prepare_v2 as subject
from ofc_regular import hu_m31_t3_step6d_full100_wave_science_registry_v2 as science_registry


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STARTUP = (
    REPO_ROOT / "scripts/startup_hu_m31_t3_step6d_full100_wave_v2.sh"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)

    phase_a = sub.add_parser("phase-a", help="create plan/package only")
    phase_a.add_argument("--output-dir", type=Path, required=True)
    phase_a.add_argument("--run-name", required=True)
    phase_a.add_argument("--scientific-package-dir", type=Path, required=True)
    phase_a.add_argument("--startup-script", type=Path)
    phase_a.add_argument("--wheelhouse-archive", type=Path, required=True)
    phase_a.add_argument("--wheelhouse-manifest", type=Path, required=True)
    phase_a.add_argument("--image-digest", required=True)
    phase_a.add_argument("--bucket", default=subject.BUCKET)
    phase_a.add_argument(
        "--expected-startup-sha256", default=None
    )
    phase_a.add_argument(
        "--full100-plan", type=Path, default=wave_v2.DEFAULT_FULL100_PLAN_PATH
    )
    phase_a.add_argument(
        "--startup-canary",
        action="store_true",
        help=(
            "select candidate-shard-00/a00 only; diagnostic output is never "
            "eligible for performance, merge, or training"
        ),
    )

    collect = sub.add_parser(
        "collect-absence", help="GET all planned instance/disk names"
    )
    collect.add_argument("--phase-a-dir", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--project", default=subject.PROJECT)
    collect.add_argument("--zone", default=subject.ZONE)

    phase_b = sub.add_parser(
        "phase-b", help="derive initial ledger/resume from fresh absence proof"
    )
    phase_b.add_argument("--phase-a-dir", type=Path, required=True)
    phase_b.add_argument("--absence-receipt", type=Path, required=True)
    phase_b.add_argument("--output-dir", type=Path, required=True)
    phase_b.add_argument("--current-time-utc")
    phase_b.add_argument(
        "--expected-startup-sha256", default=None
    )
    return parser


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.phase == "phase-a":
        identity_salt = os.environ.get(subject.IDENTITY_SALT_ENV)
        if identity_salt is None:
            raise PermissionError(
                f"fresh identity salt must be supplied only through "
                f"{subject.IDENTITY_SALT_ENV}"
            )
        frozen = wave_v2._read_frozen_plan(args.full100_plan)
        descriptor = science_registry.descriptor_for_plan(frozen)
        result = subject.prepare_phase_a(
            output_dir=args.output_dir,
            run_name=args.run_name,
            identity_salt=identity_salt,
            scientific_package_dir=args.scientific_package_dir,
            startup_script=(
                descriptor.resolved_startup_path()
                if args.startup_script is None
                else args.startup_script
            ),
            wheelhouse_archive=args.wheelhouse_archive,
            wheelhouse_manifest=args.wheelhouse_manifest,
            image_digest=args.image_digest,
            bucket=args.bucket,
            expected_startup_sha256=args.expected_startup_sha256,
            full100_plan_path=args.full100_plan,
            execution_scope=(
                wave_v2.STARTUP_CANARY_SCOPE
                if args.startup_canary
                else None
            ),
        )
    elif args.phase == "collect-absence":
        prepared = subject.validate_phase_a(args.phase_a_dir)
        result = subject.collect_all_owned_absence(
            wave_plan=prepared["plan"], project=args.project, zone=args.zone
        )
        subject.write_canonical_once(args.output, result)
    else:
        absence = subject.read_canonical_file(
            args.absence_receipt, "all-owned absence input"
        )
        result = subject.finalize_phase_b(
            phase_a_dir=args.phase_a_dir,
            output_dir=args.output_dir,
            absence_receipt=absence,
            current_time_utc=args.current_time_utc or _now_utc(),
            expected_startup_sha256=args.expected_startup_sha256,
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
