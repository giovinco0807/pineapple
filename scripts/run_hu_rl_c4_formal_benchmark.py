#!/usr/bin/env python3
"""Prepare, execute, or validate the local-only C4 HU RL benchmark gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

from ofc_regular.hu_rl_c4_formal_benchmark import (
    HuRlC4FormalBenchmarkError,
    build_c4_formal_benchmark_manifest,
    build_c4_machine_attestation,
    canonical_c4_json,
    load_canonical_c4_json,
    run_c4_formal_benchmark,
    validate_c4_formal_benchmark_manifest,
    validate_c4_formal_benchmark_result,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="Freeze a new byte-locked local package manifest."
    )
    prepare.add_argument("--output", type=Path, required=True)

    attest = subparsers.add_parser(
        "attest",
        help=(
            "Bind a canonical externally collected GCE observation to a manifest "
            "without performing a cloud lookup."
        ),
    )
    attest.add_argument("--manifest", type=Path, required=True)
    attest.add_argument("--external-observation", type=Path, required=True)
    attest.add_argument("--output", type=Path, required=True)

    execute = subparsers.add_parser(
        "execute", help="Run on an already provisioned and attested C4 host."
    )
    execute.add_argument("--manifest", type=Path, required=True)
    execute.add_argument("--machine-attestation", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)

    validate = subparsers.add_parser(
        "validate", help="Validate a manifest/result pair without mutation."
    )
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--result", type=Path, required=True)
    return parser


def _write_exclusive(path: Path, value: Mapping[str, object]) -> None:
    if path.exists():
        raise HuRlC4FormalBenchmarkError("output path already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_c4_json(value))
        handle.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        if args.output.exists():
            raise HuRlC4FormalBenchmarkError("output path already exists")
        manifest = build_c4_formal_benchmark_manifest()
        _write_exclusive(args.output, manifest)
        print(canonical_c4_json(manifest))
        return 0

    manifest = load_canonical_c4_json(args.manifest, context="C4 manifest")
    validate_c4_formal_benchmark_manifest(manifest)
    if args.command == "attest":
        if args.output.exists():
            raise HuRlC4FormalBenchmarkError("output path already exists")
        observation = load_canonical_c4_json(
            args.external_observation,
            context="C4 external GCE observation",
        )
        attestation = build_c4_machine_attestation(
            manifest_sha256=manifest["manifest_sha256"],
            external_observation=observation,
        )
        _write_exclusive(args.output, attestation)
        print(canonical_c4_json(attestation))
        return 0
    if args.command == "execute":
        if args.output.exists():
            raise HuRlC4FormalBenchmarkError("output path already exists")
        attestation = load_canonical_c4_json(
            args.machine_attestation,
            context="C4 machine attestation",
        )
        result = run_c4_formal_benchmark(
            manifest=manifest,
            machine_attestation=attestation,
        )
        _write_exclusive(args.output, result)
        print(canonical_c4_json(result))
        return 0 if result["gates"]["overall_pass"] else 2

    result = load_canonical_c4_json(args.result, context="C4 result")
    validate_c4_formal_benchmark_result(result, manifest=manifest)
    print(
        json.dumps(
            {
                "manifest_sha256": manifest["manifest_sha256"],
                "result_sha256": result["result_sha256"],
                "status": result["status"],
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )
    return 0 if result["gates"]["overall_pass"] else 2


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except HuRlC4FormalBenchmarkError as error:
        print(f"C4 formal benchmark failed closed: {error}", file=sys.stderr)
        raise SystemExit(1) from None
