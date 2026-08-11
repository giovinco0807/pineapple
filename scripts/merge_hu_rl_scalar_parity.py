#!/usr/bin/env python3
"""Merge immutable RLB scalar-parity shards into one provenance receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from ofc_regular.hu_rl_scalar_parity_merge import (
    HuRlScalarParityMergeError,
    merge_scalar_parity_shards,
    validate_merge_runtime_entrypoint,
    write_merge_receipt,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--expected-seed", type=int, required=True)
    parser.add_argument("--expected-hands", type=int, required=True)
    parser.add_argument("--expected-run-contract-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    validate_merge_runtime_entrypoint(args.source_root, __file__)
    shards = sorted(args.shard_dir.glob("shard_*.json"))
    receipt = merge_scalar_parity_shards(
        shards,
        binary_path=args.binary,
        profile_path=args.profile,
        expected_seed=args.expected_seed,
        expected_hands=args.expected_hands,
        expected_run_contract_sha256=args.expected_run_contract_sha256,
        source_root=args.source_root,
    )
    write_merge_receipt(args.output, receipt)
    print(
        json.dumps(
            _stdout_receipt(receipt),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return 0


def _stdout_receipt(receipt: dict[str, object]) -> dict[str, object]:
    """Do not duplicate reconstructable audit provenance into ordinary logs."""

    return {
        "schema": "regular_ofc_hu_rl_scalar_parity_merge_receipt_stdout_v3",
        "status": receipt["status"],
        "hands": receipt["hands"],
        "shard_count": receipt["shard_count"],
        "total_decisions": receipt["total_decisions"],
        "unique_request_count": receipt["unique_request_count"],
        "run_contract_sha256": receipt["run_contract_sha256"],
        "self_sha256": receipt["self_sha256"],
        "artifact_role": receipt["artifact_role"],
        "contains_reconstructable_hidden_oracle_state": receipt[
            "contains_reconstructable_hidden_oracle_state"
        ],
        "policy_input_eligible": receipt["policy_input_eligible"],
        "replay_eligible": receipt["replay_eligible"],
        "training_eligible": receipt["training_eligible"],
    }


def safe_main(argv: Sequence[str] | None = None) -> int:
    try:
        return main(argv)
    except (HuRlScalarParityMergeError, OSError):
        print(
            '{"error":"scalar parity merge failed","status":"error"}',
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(safe_main())
