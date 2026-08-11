"""Cloud-neutral preparation CLI for M3.1 dataset transport v2."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from ofc_regular import hu_m31_t3_dataset_gcp_provider_v2 as provider
from ofc_regular import hu_m31_t3_dataset_gcp_transport_v2 as bridge
from ofc_regular import hu_m31_t3_dataset_worker_sa_plan_v2 as sa_plan


def _read(path: str) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or raw != bridge.canonical_bytes(value):
        raise ValueError(f"not canonical JSON: {path}")
    return value


def _write(path: str, value: dict[str, Any]) -> None:
    target = Path(path)
    raw = bridge.canonical_bytes(value)
    if target.exists() or target.is_symlink():
        if (
            target.is_symlink()
            or not target.is_file()
            or target.read_bytes() != raw
        ):
            raise FileExistsError(f"immutable output conflicts: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _sources(args: argparse.Namespace) -> dict[str, str]:
    return {
        "dataset_plan": args.dataset_plan,
        "fresh_quality_gate": args.fresh_quality_gate,
        "smoke_gate": args.smoke_gate,
        "portable_authorization": args.portable_authorization,
        "smoke_shard_archive": args.smoke_shard_archive,
        "runtime_archive": args.runtime_archive,
        "wheelhouse_archive": args.wheelhouse_archive,
        "candidate_library": args.candidate_library,
    }


def _source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-plan", required=True)
    parser.add_argument("--fresh-quality-gate", required=True)
    parser.add_argument("--smoke-gate", required=True)
    parser.add_argument("--portable-authorization", required=True)
    parser.add_argument("--smoke-shard-directory", required=True)
    parser.add_argument("--smoke-shard-archive", required=True)
    parser.add_argument("--runtime-archive", required=True)
    parser.add_argument("--wheelhouse-archive", required=True)
    parser.add_argument("--candidate-library", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare immutable M3.1 parallel-20 transport artifacts"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    relocation = sub.add_parser("relocate-ext4")
    _source_arguments(relocation)
    relocation.add_argument("--destination-root", required=True)
    relocation.add_argument("--output", required=True)

    plan = sub.add_parser("build-plan")
    _source_arguments(plan)
    plan.add_argument("--run-name", required=True)
    plan.add_argument("--bucket", required=True)
    plan.add_argument("--source-relocation-receipt", required=True)
    plan.add_argument("--output", required=True)

    config = sub.add_parser("write-provider-config")
    config.add_argument("--image-self-link", required=True)
    config.add_argument("--image-id", required=True)
    config.add_argument("--guest-os-feature", action="append", required=True)
    config.add_argument("--output", required=True)

    service_accounts = sub.add_parser("write-sa-plan")
    service_accounts.add_argument("--output", required=True)
    service_accounts.add_argument("--observed-existing", action="append")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "relocate-ext4":
            value = bridge.relocate_sources_to_ext4(
                source_files=_sources(args),
                smoke_shard_directory=args.smoke_shard_directory,
                destination_root=args.destination_root,
            )
        elif args.command == "build-plan":
            value = bridge.build_transport_plan(
                source_relocation_receipt=_read(
                    args.source_relocation_receipt
                ),
                run_name=args.run_name,
                bucket=args.bucket,
                dataset_plan_path=args.dataset_plan,
                fresh_quality_gate_path=args.fresh_quality_gate,
                smoke_gate_path=args.smoke_gate,
                portable_authorization_path=args.portable_authorization,
                smoke_shard_directory=args.smoke_shard_directory,
                smoke_shard_archive_path=args.smoke_shard_archive,
                runtime_archive_path=args.runtime_archive,
                wheelhouse_archive_path=args.wheelhouse_archive,
                candidate_library_path=args.candidate_library,
            )
        elif args.command == "write-provider-config":
            value = {
                "schema": (
                    "hu_m31_t3_dataset_gcp_provider_config_v2"
                ),
                "image_self_link": args.image_self_link,
                "image_id": args.image_id,
                "guest_os_features": sorted(set(args.guest_os_feature)),
                "worker_service_accounts": list(
                    provider.EXPECTED_WORKER_SERVICE_ACCOUNTS
                ),
                "service_accounts_preexisting": True,
                "service_account_creation_authorized": False,
                "current_profile_changed": False,
            }
        elif args.command == "write-sa-plan":
            value = sa_plan.build_provisioning_plan(
                observed_existing_service_accounts=(
                    args.observed_existing
                    if args.observed_existing is not None
                    else sa_plan.KNOWN_EXISTING
                )
            )
        else:  # pragma: no cover
            raise RuntimeError("unknown preparation command")
        _write(args.output, value)
        print(bridge.canonical_bytes(value).decode("ascii"))
    except Exception as exc:  # pragma: no cover - CLI boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
