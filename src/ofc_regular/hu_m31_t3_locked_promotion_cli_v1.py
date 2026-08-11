"""Fail-closed CLI for M3.1 locked plan, merge, and gate artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from . import hu_m31_t3_step6d_locked_promotion_v1 as promotion


def _read_canonical(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != promotion.canonical_bytes(value):
        raise ValueError(f"{label} is not a canonical JSON object")
    return value


def _build_plan(args: argparse.Namespace) -> dict[str, Any]:
    plan = promotion.build_locked_promotion_plan(
        plan_id=args.plan_id,
        model_artifact_id=args.model_artifact_id,
        model_path=args.model,
        expected_model_sha256=args.expected_model_sha256,
        threshold_lock_path=args.threshold_lock,
        expected_threshold_lock_sha256=(
            args.expected_threshold_lock_sha256
        ),
        policy_registry_path=args.policy_registry,
        expected_policy_registry_sha256=(
            args.expected_policy_registry_sha256
        ),
        evaluation_runtime_closure_path=args.runtime_closure,
        expected_evaluation_runtime_closure_sha256=(
            args.expected_runtime_closure_sha256
        ),
    )
    return promotion.write_locked_promotion_plan(
        plan=plan, output_path=args.output
    )


def _merge(args: argparse.Namespace) -> dict[str, Any]:
    plan = _read_canonical(args.plan, "locked promotion plan")
    return promotion.write_locked_promotion_merge(
        plan=plan,
        shard_paths=args.shard,
        output_path=args.output,
    )


def _gate(args: argparse.Namespace) -> dict[str, Any]:
    plan = _read_canonical(args.plan, "locked promotion plan")
    merge = _read_canonical(args.merge, "locked promotion merge")
    return promotion.write_locked_promotion_gate(
        plan=plan,
        merge=merge,
        replay_sources=True,
        output_path=args.output,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M3.1 locked promotion plan/merge/gate"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("build-plan")
    plan.add_argument("--plan-id", required=True)
    plan.add_argument("--model-artifact-id", required=True)
    plan.add_argument("--model", required=True)
    plan.add_argument("--expected-model-sha256", required=True)
    plan.add_argument("--threshold-lock", required=True)
    plan.add_argument("--expected-threshold-lock-sha256", required=True)
    plan.add_argument("--policy-registry", required=True)
    plan.add_argument("--expected-policy-registry-sha256", required=True)
    plan.add_argument("--runtime-closure", required=True)
    plan.add_argument("--expected-runtime-closure-sha256", required=True)
    plan.add_argument("--output", required=True)
    plan.set_defaults(handler=_build_plan)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--plan", required=True)
    merge.add_argument("--shard", action="append", required=True)
    merge.add_argument("--output", required=True)
    merge.set_defaults(handler=_merge)
    gate = subparsers.add_parser("gate")
    gate.add_argument("--plan", required=True)
    gate.add_argument("--merge", required=True)
    gate.add_argument("--output", required=True)
    gate.set_defaults(handler=_gate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = args.handler(args)
    except Exception as exc:  # pragma: no cover - subprocess boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
