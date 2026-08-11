#!/usr/bin/env python3
"""Run deterministic Python/Rust scalar HU RL parity against a prebuilt binary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from ofc_regular.hu_rl_scalar_parity import (
    HuRlScalarBinaryError,
    HuRlScalarParityError,
    build_scalar_parity_run_contract,
    load_scalar_parity_summary,
    pin_scalar_parity_binary,
    run_scalar_parity_suite,
    scalar_parity_run_contract_digest,
    validate_runtime_source_identity,
    write_scalar_parity_summary,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--pin-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--hands", type=int, default=4)
    parser.add_argument("--global-hands", type=int, required=True)
    parser.add_argument(
        "--hand-start",
        type=int,
        default=0,
        help="Global deterministic hand ordinal for restart-safe shards.",
    )
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent local validator processes; output order stays deterministic.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional new summary file; omitted means stdout only.",
    )
    parser.add_argument("--expected-run-contract-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    validate_runtime_source_identity(
        args.source_root,
        __file__,
        "scripts/run_hu_rl_scalar_parity.py",
    )
    pinned_binary = pin_scalar_parity_binary(args.binary, args.pin_dir)
    contract = build_scalar_parity_run_contract(
        pinned_binary,
        profile_path=args.profile,
        source_root=args.source_root,
        seed=args.seed,
        global_hands=args.global_hands,
    )
    contract_sha256 = scalar_parity_run_contract_digest(contract)
    if (
        args.expected_run_contract_sha256 is not None
        and args.expected_run_contract_sha256 != contract_sha256
    ):
        raise HuRlScalarParityError("frozen scalar parity run contract changed")
    if args.output is not None and args.output.exists():
        summary, _encoded = load_scalar_parity_summary(
            args.output,
            expected_hand_start=args.hand_start,
            expected_hands=args.hands,
            expected_run_contract_sha256=contract_sha256,
        )
        end_contract = build_scalar_parity_run_contract(
            pinned_binary,
            profile_path=args.profile,
            source_root=args.source_root,
            seed=args.seed,
            global_hands=args.global_hands,
        )
        if scalar_parity_run_contract_digest(end_contract) != contract_sha256:
            raise HuRlScalarParityError("pinned parity inputs changed during resume")
        print(_canonical_stdout(_stdout_receipt(summary)))
        return 0
    summary = run_scalar_parity_suite(
        pinned_binary,
        hands=args.hands,
        seed=args.seed,
        global_hands=args.global_hands,
        profile_path=args.profile,
        source_root=args.source_root,
        timeout_seconds=args.timeout_seconds,
        workers=args.workers,
        hand_start=args.hand_start,
        expected_run_contract_sha256=contract_sha256,
    )
    if args.output is not None:
        summary = write_scalar_parity_summary(args.output, summary)
    print(_canonical_stdout(_stdout_receipt(summary)))
    return 0


def _canonical_stdout(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _stdout_receipt(summary: dict[str, object]) -> dict[str, object]:
    """Keep reconstructable audit provenance out of ordinary process logs."""

    return {
        "schema": "regular_ofc_hu_rl_scalar_parity_run_receipt_v3",
        "status": summary["status"],
        "hands": summary["hands"],
        "hand_start": summary["hand_start"],
        "hand_end_exclusive": summary["hand_end_exclusive"],
        "total_decisions": summary["total_decisions"],
        "unique_request_count": summary["unique_request_count"],
        "run_contract_sha256": summary["run_contract_sha256"],
        "artifact_role": summary["artifact_role"],
        "contains_reconstructable_hidden_oracle_state": summary[
            "contains_reconstructable_hidden_oracle_state"
        ],
        "policy_input_eligible": summary["policy_input_eligible"],
        "replay_eligible": summary["replay_eligible"],
        "training_eligible": summary["training_eligible"],
        "artifact_written": summary["artifact_written"],
    }


def safe_main(argv: Sequence[str] | None = None) -> int:
    try:
        return main(argv)
    except (HuRlScalarBinaryError, HuRlScalarParityError, OSError):
        print(
            '{"error":"scalar parity run failed","status":"error"}',
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(safe_main())
