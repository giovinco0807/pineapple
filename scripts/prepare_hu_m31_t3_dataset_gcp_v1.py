"""Cloud-neutral preparation helpers for the selected M3.1 v1 transport."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from ofc_regular import hu_m31_t3_dataset_gcp_controller_v1 as controller
from ofc_regular import hu_m31_t3_dataset_gcp_provider_v1 as provider
from ofc_regular import hu_m31_t3_dataset_source_binding_v1 as source_binding
from ofc_regular import hu_m31_t3_dataset_supervisor_v1 as supervisor
from ofc_regular import hu_m31_t3_dataset_transport_selection_v1 as selection


EXPECTED_V1_WORKER_SERVICE_ACCOUNTS = tuple(
    f"ofc-f100-worker-{index:02d}@{provider.PROJECT}.iam.gserviceaccount.com"
    for index in range(8)
)


def _read(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    raw = source.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != controller.canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _self_digest(value: Mapping[str, Any], field: str) -> str:
    copied = deepcopy(dict(value))
    copied.pop(field, None)
    return controller.canonical_sha256(copied)


def _write_once(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    raw = controller.canonical_bytes(value)
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != raw:
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


def build_provider_config(
    *,
    image_self_link: str,
    image_id: str,
    guest_os_features: Sequence[str],
) -> dict[str, Any]:
    features = sorted(set(guest_os_features))
    value = {
        "schema": controller.PROVIDER_CONFIG_SCHEMA,
        "image_self_link": image_self_link,
        "image_id": image_id,
        "guest_os_features": features,
        "worker_service_accounts": list(EXPECTED_V1_WORKER_SERVICE_ACCOUNTS),
        "current_profile_changed": False,
    }
    return controller._provider_config(value)  # type: ignore[attr-defined]


def build_provider_config_from_fresh_quality_plan(
    path: str | Path,
) -> dict[str, Any]:
    plan = _read(path, "fresh-quality provider plan")
    if (
        plan.get("provider_plan_sha256") != _self_digest(plan, "provider_plan_sha256")
        or plan.get("schema") != "hu_m31_t3_step6d_fresh_quality_gcp_provider_plan_v1"
        or plan.get("status") != "provider_plan_ready_cloud_not_mutated"
        or plan.get("project") != provider.PROJECT
        or plan.get("region") != provider.REGION
        or plan.get("zone") != provider.ZONE
        or plan.get("cloud_mutated") is not False
        or plan.get("current_profile_changed") is not False
    ):
        raise ValueError("fresh-quality provider-plan evidence changed")
    image = plan.get("image")
    if not isinstance(image, Mapping):
        raise ValueError("fresh-quality provider plan omitted image evidence")
    features = image.get("guest_os_features")
    if not isinstance(features, list) or not all(
        isinstance(item, str) for item in features
    ):
        raise ValueError("fresh-quality image feature evidence changed")
    return build_provider_config(
        image_self_link=image.get("self_link"),
        image_id=image.get("id"),
        guest_os_features=features,
    )


def build_selection_from_fresh_quality_launch(
    *,
    launch_path: str | Path,
    observed_worker_service_accounts: Sequence[str],
) -> dict[str, Any]:
    launch = _read(launch_path, "fresh-quality launch receipt")
    unlisted = launch.get("unlisted_vm_created")
    if (
        launch.get("receipt_sha256") != _self_digest(launch, "receipt_sha256")
        or launch.get("schema") != "hu_m31_t3_step6d_fresh_quality_gcp_launch_v1"
        or launch.get("status") != "exact_selected_quality_wave_created"
        or launch.get("create_complete") is not True
        or not isinstance(unlisted, int)
        or isinstance(unlisted, bool)
        or unlisted != 0
        or launch.get("current_profile_changed") is not False
    ):
        raise ValueError("fresh-quality launch evidence changed")
    observed = tuple(observed_worker_service_accounts)
    if observed != EXPECTED_V1_WORKER_SERVICE_ACCOUNTS:
        raise ValueError("exact pre-existing v1 worker account inventory changed")
    quota = launch.get("quota_receipt")
    rows = quota.get("metrics") if isinstance(quota, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError("fresh-quality quota evidence is absent")
    metrics = {row.get("metric"): row for row in rows if isinstance(row, Mapping)}
    expected_metrics = {
        "c4_family_vcpus",
        "spot_vcpus",
        "global_vcpus",
    }
    if set(metrics) != expected_metrics:
        raise ValueError("fresh-quality quota metric set changed")
    source = supervisor.build_source_receipt()
    source_hashes = {
        row["relative_path"]: row["observed_sha256"] for row in source["files"]
    }
    evidence = Path(launch_path).resolve()
    evidence_sha256 = hashlib.sha256(evidence.read_bytes()).hexdigest()
    return selection.build_selection_receipt(
        c4_limit_vcpus=int(metrics["c4_family_vcpus"]["limit_vcpus"]),
        c4_usage_vcpus=int(metrics["c4_family_vcpus"]["usage_vcpus"]),
        spot_limit_vcpus=int(metrics["spot_vcpus"]["limit_vcpus"]),
        spot_usage_vcpus=int(metrics["spot_vcpus"]["usage_vcpus"]),
        global_limit_vcpus=int(metrics["global_vcpus"]["limit_vcpus"]),
        global_usage_vcpus=int(metrics["global_vcpus"]["usage_vcpus"]),
        observed_worker_service_account_count=len(observed),
        evidence_source=str(evidence),
        evidence_sha256=evidence_sha256,
        source_hashes=source_hashes,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare cloud-neutral M3.1 v1 8-VM artifacts"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    config = sub.add_parser("write-provider-config")
    config.add_argument("--image-self-link", required=True)
    config.add_argument("--image-id", required=True)
    config.add_argument("--guest-os-feature", action="append", required=True)
    config.add_argument("--output", required=True)
    frozen_config = sub.add_parser("write-provider-config-from-fq-plan")
    frozen_config.add_argument("--fresh-quality-provider-plan", required=True)
    frozen_config.add_argument("--output", required=True)
    selected = sub.add_parser("write-selection")
    selected.add_argument("--fresh-quality-launch", required=True)
    selected.add_argument(
        "--observed-worker-service-account", action="append", required=True
    )
    selected.add_argument("--output", required=True)
    bound = sub.add_parser("write-source-binding")
    bound.add_argument("--same-linux-closeout", required=True)
    bound.add_argument("--expected-closeout-file-sha256", required=True)
    bound.add_argument("--expected-fresh-quality-plan-file-sha256", required=True)
    bound.add_argument("--expected-smoke-gate-file-sha256", required=True)
    bound.add_argument("--expected-fresh-quality-run-name", required=True)
    bound.add_argument("--expected-dataset-run-name", required=True)
    bound.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "write-provider-config":
            value = build_provider_config(
                image_self_link=args.image_self_link,
                image_id=args.image_id,
                guest_os_features=args.guest_os_feature,
            )
        elif args.command == "write-provider-config-from-fq-plan":
            value = build_provider_config_from_fresh_quality_plan(
                args.fresh_quality_provider_plan
            )
        elif args.command == "write-selection":
            value = build_selection_from_fresh_quality_launch(
                launch_path=args.fresh_quality_launch,
                observed_worker_service_accounts=(args.observed_worker_service_account),
            )
        elif args.command == "write-source-binding":
            if not Path(args.output).is_absolute():
                raise ValueError("dataset source-binding output must be absolute")
            value = source_binding.build_source_binding(
                closeout_ready_path=args.same_linux_closeout,
                expected_closeout_file_sha256=(args.expected_closeout_file_sha256),
                expected_fresh_quality_plan_file_sha256=(
                    args.expected_fresh_quality_plan_file_sha256
                ),
                expected_smoke_gate_file_sha256=(args.expected_smoke_gate_file_sha256),
                expected_fresh_quality_run_name=(args.expected_fresh_quality_run_name),
                expected_dataset_run_name=args.expected_dataset_run_name,
            )
        else:  # pragma: no cover
            raise RuntimeError("unknown preparation command")
        _write_once(args.output, value)
        if args.command == "write-source-binding":
            source_binding.validate_source_binding_file(args.output)
        print(controller.canonical_bytes(value).decode("ascii"))
    except Exception as exc:  # pragma: no cover - CLI boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
