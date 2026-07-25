"""Command-line boundary for the two pure performance-lock-v4 finalizers.

The scientific merger and production bridge intentionally expose Python APIs
only.  This module supplies a narrow, local-only CLI so an operator does not
need to assemble their keyword arguments in an ad-hoc Python expression.

Neither command has a cloud client or imports the AI profile factory.  Both
delegate the create-only write and complete source replay to the underlying
scientific modules.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import hu_m31_t3_step6d_performance_lock_v4_production_bridge as production
from . import merge_hu_m31_t3_step6d_candidate02_performance_lock_v4 as pure


def _plain_file(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be an absolute non-symlink file")
    return path.resolve()


def _canonical_mapping(value: str | Path, label: str) -> dict[str, Any]:
    path = _plain_file(value, label)
    raw = path.read_bytes()
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(parsed, dict) or raw != pure.canonical_bytes(parsed):
        raise ValueError(f"{label} is not canonical LF JSON")
    return parsed


def merge_from_paths(
    *,
    candidate_done_paths: Sequence[str | Path],
    reference_done_paths: Sequence[str | Path],
    plan_path: str | Path,
    materialization_receipt_path: str | Path,
    root_seal_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Create and replay the pure v4 merge from explicit immutable paths."""

    candidate = [
        _plain_file(path, f"candidate DONE {index}")
        for index, path in enumerate(candidate_done_paths)
    ]
    reference = [
        _plain_file(path, f"reference DONE {index}")
        for index, path in enumerate(reference_done_paths)
    ]
    if not candidate or not reference:
        raise ValueError("candidate/reference DONE paths cannot be empty")
    if set(candidate) & set(reference):
        raise ValueError("candidate/reference DONE paths must be disjoint")
    return pure.merge_and_write_candidate02_performance_lock_v4(
        candidate_done_paths=candidate,
        reference_done_paths=reference,
        plan_value=_canonical_mapping(plan_path, "performance-lock-v4 plan"),
        materialization_value=_canonical_mapping(
            materialization_receipt_path, "v4 materialization receipt"
        ),
        root_seal_value=_canonical_mapping(root_seal_path, "v4 root seal"),
        output_path=Path(output_path),
    )


def finalize_from_paths(
    *,
    pure_merge_path: str | Path,
    performance_lock_plan_path: str | Path,
    materialization_receipt_path: str | Path,
    root_seal_path: str | Path,
    outer_package_path: str | Path,
    merge_view_manifest_path: str | Path,
    current_profile_registry_path: str | Path,
    expected_profile_sha256: str,
    wave_plan_path: str | Path,
    attempt_ledger_path: str | Path,
    accepted_results_snapshot_path: str | Path,
    validated_lifecycle_chain_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Create and replay the final production receipt from explicit paths."""

    merge_view = _canonical_mapping(
        merge_view_manifest_path, "wave-v2 merge-view manifest"
    )
    return production.write_performance_lock_v4_production_receipt(
        output_path=Path(output_path),
        expected_profile_sha256=expected_profile_sha256,
        pure_merge_path=_plain_file(pure_merge_path, "pure v4 merge"),
        performance_lock_plan_path=_plain_file(
            performance_lock_plan_path, "performance-lock-v4 plan"
        ),
        materialization_receipt_path=_plain_file(
            materialization_receipt_path, "v4 materialization receipt"
        ),
        root_seal_path=_plain_file(root_seal_path, "v4 root seal"),
        outer_package_path=Path(outer_package_path),
        merge_view_manifest_path=_plain_file(
            merge_view_manifest_path, "wave-v2 merge-view manifest"
        ),
        current_profile_registry_path=_plain_file(
            current_profile_registry_path, "current profile registry"
        ),
        wave_plan=_canonical_mapping(wave_plan_path, "wave plan"),
        attempt_ledger=_canonical_mapping(
            attempt_ledger_path, "final attempt ledger"
        ),
        accepted_results_snapshot=_canonical_mapping(
            accepted_results_snapshot_path, "accepted-results snapshot"
        ),
        validated_lifecycle_chain=_canonical_mapping(
            validated_lifecycle_chain_path, "validated lifecycle chain"
        ),
        merge_view_manifest=merge_view,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay and finalize the one-shot M3.1 performance lock v4."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    merge = commands.add_parser("merge", help="create the pure v4 scientific merge")
    merge.add_argument("--candidate-done", action="append", required=True)
    merge.add_argument("--reference-done", action="append", required=True)
    merge.add_argument("--plan", required=True)
    merge.add_argument("--materialization-receipt", required=True)
    merge.add_argument("--root-seal", required=True)
    merge.add_argument("--output", required=True)

    finalize = commands.add_parser(
        "finalize", help="create the transport-validated production receipt"
    )
    finalize.add_argument("--pure-merge", required=True)
    finalize.add_argument("--plan", required=True)
    finalize.add_argument("--materialization-receipt", required=True)
    finalize.add_argument("--root-seal", required=True)
    finalize.add_argument("--outer-package", required=True)
    finalize.add_argument("--merge-view-manifest", required=True)
    finalize.add_argument("--current-profile-registry", required=True)
    finalize.add_argument("--expected-profile-sha256", required=True)
    finalize.add_argument("--wave-plan", required=True)
    finalize.add_argument("--attempt-ledger", required=True)
    finalize.add_argument("--accepted-results-snapshot", required=True)
    finalize.add_argument("--validated-lifecycle-chain", required=True)
    finalize.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "merge":
        result = merge_from_paths(
            candidate_done_paths=args.candidate_done,
            reference_done_paths=args.reference_done,
            plan_path=args.plan,
            materialization_receipt_path=args.materialization_receipt,
            root_seal_path=args.root_seal,
            output_path=args.output,
        )
    else:
        result = finalize_from_paths(
            pure_merge_path=args.pure_merge,
            performance_lock_plan_path=args.plan,
            materialization_receipt_path=args.materialization_receipt,
            root_seal_path=args.root_seal,
            outer_package_path=args.outer_package,
            merge_view_manifest_path=args.merge_view_manifest,
            current_profile_registry_path=args.current_profile_registry,
            expected_profile_sha256=args.expected_profile_sha256,
            wave_plan_path=args.wave_plan,
            attempt_ledger_path=args.attempt_ledger,
            accepted_results_snapshot_path=args.accepted_results_snapshot,
            validated_lifecycle_chain_path=args.validated_lifecycle_chain,
            output_path=args.output,
        )
    print(pure.canonical_bytes(result).decode("ascii"), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "finalize_from_paths",
    "main",
    "merge_from_paths",
]
