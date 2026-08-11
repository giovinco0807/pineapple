#!/usr/bin/env python3
"""Prepare and operate the exact-one-VM HU RL C4 GCP lifecycle."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

from ofc_regular.hu_rl_c4_gcp_lifecycle import (
    GcpRestTransport,
    HuRlC4GcpLifecycleError,
    build_execution_plan,
    build_operation_authorization,
    canonical_bytes,
    cleanup_owned_instance,
    collect_result,
    launch_or_recover,
    load_json,
    validate_execution_plan,
    validate_launch_receipt,
    write_json_once,
)
from ofc_regular.hu_rl_c4_formal_benchmark import HuRlC4FormalBenchmarkError


NONCE_ENV = "OFC_HU_RL_C4_GCP_OPERATION_NONCE"
TOKEN_ENV = "GOOGLE_OAUTH_ACCESS_TOKEN"
PRINCIPAL_ENV = "OFC_HU_RL_C4_GCP_CONTROLLER_PRINCIPAL"
LAUNCH_SENTINEL = "CREATE_EXACTLY_ONE_C4_SPOT_VM"
CLEANUP_SENTINEL = "DELETE_ONLY_THE_EXACT_OWNED_C4_VM"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Verify package _003 and freeze an exact-one-VM plan.")
    prepare.add_argument("--run-name", required=True)
    prepare.add_argument("--package-root", type=Path, required=True)
    prepare.add_argument("--archive", type=Path, required=True)
    prepare.add_argument("--archive-sha256", required=True)
    prepare.add_argument("--bucket", required=True)
    prepare.add_argument("--worker-service-account", required=True)
    prepare.add_argument("--source-image-self-link", required=True)
    prepare.add_argument("--image-id", required=True)
    prepare.add_argument("--guest-os-feature", action="append", required=True)
    prepare.add_argument("--output", type=Path, required=True)

    authorize = sub.add_parser("authorize", help="Create a short-lived local operation authorization.")
    authorize.add_argument("--plan", type=Path, required=True)
    authorize.add_argument("--operation", choices=("launch", "cleanup"), required=True)
    authorize.add_argument("--approve-exact-one-vm", action="store_true")
    authorize.add_argument("--output", type=Path, required=True)

    launch = sub.add_parser("launch", help="Stage, launch/recover, observe, attest, and return a receipt.")
    launch.add_argument("--plan", type=Path, required=True)
    launch.add_argument("--authorization", type=Path, required=True)
    launch.add_argument("--package-root", type=Path, required=True)
    launch.add_argument("--archive", type=Path, required=True)
    launch.add_argument("--execute-cloud-mutations")
    launch.add_argument("--output", type=Path, required=True)

    collect = sub.add_parser("collect", help="Read and validate the formal result; no cloud mutation.")
    collect.add_argument("--plan", type=Path, required=True)
    collect.add_argument("--launch-receipt", type=Path, required=True)
    collect.add_argument("--package-root", type=Path, required=True)
    collect.add_argument("--result-output", type=Path, required=True)
    collect.add_argument("--receipt-output", type=Path, required=True)

    cleanup = sub.add_parser("cleanup", help="Delete only the exact measured owned instance.")
    cleanup.add_argument("--plan", type=Path, required=True)
    cleanup.add_argument("--authorization", type=Path, required=True)
    cleanup.add_argument("--launch-receipt", type=Path)
    cleanup.add_argument("--collection-receipt", type=Path)
    cleanup.add_argument("--explicit-abort", action="store_true")
    cleanup.add_argument("--execute-cloud-mutations")
    cleanup.add_argument("--output", type=Path, required=True)
    return parser


def _nonce() -> str:
    value = os.environ.get(NONCE_ENV, "")
    if not value:
        raise HuRlC4GcpLifecycleError(f"missing {NONCE_ENV}")
    return value


def _transport() -> GcpRestTransport:
    token = os.environ.get(TOKEN_ENV, "")
    if not token:
        raise HuRlC4GcpLifecycleError(f"missing {TOKEN_ENV}")
    return GcpRestTransport(access_token=token)


def _load_plan(path: Path) -> dict:
    plan = load_json(path, "C4 lifecycle plan")
    validate_execution_plan(plan)
    return plan


def main(
    argv: Sequence[str] | None = None,
    *,
    transport_factory: Callable[[], object] = _transport,
    now_provider: Callable[[], int] = lambda: int(time.time()),
) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        plan = build_execution_plan(
            run_name=args.run_name,
            package_root=args.package_root,
            archive_path=args.archive,
            expected_archive_sha256=args.archive_sha256.lower(),
            bucket=args.bucket,
            worker_service_account=args.worker_service_account,
            source_image_self_link=args.source_image_self_link,
            image_id=args.image_id,
            guest_os_features=args.guest_os_feature,
        )
        write_json_once(args.output, plan)
        print(canonical_bytes(plan).decode("ascii").rstrip())
        return 0

    plan = _load_plan(args.plan)
    if args.command == "authorize":
        if args.approve_exact_one_vm is not True:
            raise HuRlC4GcpLifecycleError("authorization requires --approve-exact-one-vm")
        auth = build_operation_authorization(
            plan=plan,
            operation=args.operation,
            raw_nonce=_nonce(),
            now_unix_seconds=now_provider(),
        )
        write_json_once(args.output, auth)
        print(canonical_bytes(auth).decode("ascii").rstrip())
        return 0

    transport = transport_factory()
    if args.command == "launch":
        if args.execute_cloud_mutations != LAUNCH_SENTINEL:
            raise HuRlC4GcpLifecycleError(
                f"launch requires --execute-cloud-mutations {LAUNCH_SENTINEL}"
            )
        auth = load_json(args.authorization, "C4 launch authorization")
        principal = os.environ.get(PRINCIPAL_ENV, "")
        if not principal or any(ch.isspace() for ch in principal):
            raise HuRlC4GcpLifecycleError(f"missing or invalid {PRINCIPAL_ENV}")
        receipt = launch_or_recover(
            plan=plan,
            authorization=auth,
            raw_nonce=_nonce(),
            package_root=args.package_root,
            archive_path=args.archive,
            controller_principal_sha256=hashlib.sha256(principal.encode("utf-8")).hexdigest(),
            transport=transport,  # type: ignore[arg-type]
            now_unix_seconds=now_provider(),
        )
        write_json_once(args.output, receipt)
        print(canonical_bytes(receipt).decode("ascii").rstrip())
        return 0

    if args.command == "collect":
        launch_receipt = load_json(args.launch_receipt, "C4 launch receipt")
        validate_launch_receipt(launch_receipt, plan=plan)
        receipt, result_raw = collect_result(
            plan=plan,
            launch_receipt=launch_receipt,
            package_root=args.package_root,
            transport=transport,  # type: ignore[arg-type]
        )
        args.result_output.parent.mkdir(parents=True, exist_ok=True)
        with args.result_output.open("xb") as handle:
            handle.write(result_raw)
        write_json_once(args.receipt_output, receipt)
        print(canonical_bytes(receipt).decode("ascii").rstrip())
        return 0 if receipt["formal_gate_pass"] else 2

    if args.execute_cloud_mutations != CLEANUP_SENTINEL:
        raise HuRlC4GcpLifecycleError(
            f"cleanup requires --execute-cloud-mutations {CLEANUP_SENTINEL}"
        )
    auth = load_json(args.authorization, "C4 cleanup authorization")
    launch_receipt = (
        load_json(args.launch_receipt, "C4 launch receipt")
        if args.launch_receipt
        else None
    )
    if launch_receipt is not None:
        validate_launch_receipt(launch_receipt, plan=plan)
    collection = (
        load_json(args.collection_receipt, "C4 collection receipt")
        if args.collection_receipt
        else None
    )
    if launch_receipt is not None:
        package_generation = launch_receipt["package_object"]["generation"]
    else:
        metadata = transport.get_object_metadata(  # type: ignore[attr-defined]
            bucket=plan["bucket"], object_name=plan["objects"]["package"]
        )
        if metadata is None:
            raise HuRlC4GcpLifecycleError("abort cleanup cannot recover package generation")
        package_generation = str(metadata.get("generation"))
    receipt = cleanup_owned_instance(
        plan=plan,
        authorization=auth,
        raw_nonce=_nonce(),
        launch_receipt=launch_receipt,
        collection_receipt=collection,
        explicit_abort=args.explicit_abort,
        package_generation=package_generation,
        transport=transport,  # type: ignore[arg-type]
        now_unix_seconds=now_provider(),
    )
    write_json_once(args.output, receipt)
    print(canonical_bytes(receipt).decode("ascii").rstrip())
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except (HuRlC4GcpLifecycleError, HuRlC4FormalBenchmarkError) as error:
        print(f"C4 GCP lifecycle failed closed: {error}", file=sys.stderr)
        raise SystemExit(1) from None
