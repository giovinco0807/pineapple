"""Attempt10 binding for the immutable Attempt09 Spot lifecycle.

The lifecycle mechanics are shared deliberately: packaging, launch, status,
receive, and preflight finalization have no search-policy semantics of their
own.  This module binds that machinery to the frozen Attempt10 contract for
one synchronous call and restores every Attempt09 global afterwards.
"""

from __future__ import annotations

import argparse
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from . import hu_m43_attempt09_spot as _base
from .hu_m43_attempt10_contract import (
    AI_PROFILES_SHA256,
    ATTEMPT10_LAMBDA_MODEL_SHA256,
    M43_ATTEMPT10_PLAN_SHA256,
    M43_ATTEMPT10_PROFILES,
    enumerate_attempt10_seed_schedules,
    load_and_validate_attempt10_plan,
    validate_attempt10_artifact_bindings,
)
from .hu_m43_attempt10_teacher import (
    ATTEMPT10_TEACHER_SCHEMA,
    Attempt10TeacherConfig,
    validate_attempt10_teacher_output,
)
from .run_hu_m43_attempt10 import (
    ATTEMPT10_AUTHORIZATION_SCHEMA,
    ATTEMPT10_ROW_SCHEMA,
)


PACKAGE_SCHEMA = "hu_m43_attempt10_spot_package_v1"
LAUNCH_AUTHORIZATION_SCHEMA = "hu_m43_attempt10_spot_launch_authorization_v1"
DONE_SCHEMA = "hu_m43_attempt10_done_v1"
RECEIVE_SCHEMA = "hu_m43_attempt10_receive_v1"
SHARD_SCHEMA = "hu_m43_attempt10_spot_shard_v1"
LAUNCH_WAVE_SCHEMA = "hu_m43_attempt10_launch_wave_v1"
STATUS_SCHEMA = "hu_m43_attempt10_status_v1"
RECEIVED_SHARD_AUDIT_SCHEMA = "hu_m43_attempt10_received_shard_audit_v1"
PREFLIGHT_RESULT_SCHEMA = "hu_m43_attempt10_preflight_result_v1"
SOURCE_NAME = "ofc_regular_hu_m43_attempt10_source.zip"
SCHEDULE_NAME = _base.SCHEDULE_NAME
STARTUP_NAME = "startup_hu_m43_attempt10.sh"

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = _REPO_ROOT / "configs" / "hu_joint_policy_m43_attempt10.json"
DEFAULT_TEMPLATE_PACKAGE = _base.DEFAULT_TEMPLATE_PACKAGE
DEFAULT_STARTUP = _REPO_ROOT / "scripts" / STARTUP_NAME
PLAN_RELATIVE = "configs/hu_joint_policy_m43_attempt10.json"
GATE_RELATIVE = "artifacts/attempt10/preceding_gate.json"
OVERLAY_RELATIVES = (
    PLAN_RELATIVE,
    "src/ofc_regular/hu_m43_attempt10_contract.py",
    "src/ofc_regular/hu_m43_attempt10_teacher.py",
    "src/ofc_regular/run_hu_m43_attempt10.py",
    # The generic root persistence runner and its eager import closure are
    # required by run_hu_m43_attempt10 inside the immutable VM package.
    "src/ofc_regular/run_hu_m43_attempt09.py",
    "src/ofc_regular/hu_m43_attempt09_contract.py",
    "src/ofc_regular/hu_m43_attempt09_teacher.py",
    "outputs/hu_joint_policy/m43_attempt09_development/"
    "regular-hu-m43-attempt09-development200-20260715-052644/"
    "selector/decision.json",
    "outputs/hu_joint_policy/m43_attempt09_development/"
    "regular-hu-m43-attempt09-development200-20260715-052644/"
    "selector/decision_receipt.json",
)
EXPECTED_GATES = {
    "preflight": ("pass_local_correctness", "authorize_preflight_only"),
    "development": (
        "pass_correctness_preflight",
        "authorize_development200_package_only",
    ),
    "future_audit": ("go_freeze_attempt10_development", "go"),
}

_BIND_LOCK = threading.RLock()
_ATTEMPT10_BINDINGS: dict[str, Any] = {
    "AI_PROFILES_SHA256": AI_PROFILES_SHA256,
    "ATTEMPT09_LAMBDA_MODEL_SHA256": ATTEMPT10_LAMBDA_MODEL_SHA256,
    "M43_ATTEMPT09_PLAN_SHA256": M43_ATTEMPT10_PLAN_SHA256,
    "M43_ATTEMPT09_PROFILES": M43_ATTEMPT10_PROFILES,
    "enumerate_attempt09_seed_schedules": enumerate_attempt10_seed_schedules,
    "load_and_validate_attempt09_plan": load_and_validate_attempt10_plan,
    "validate_attempt09_artifact_bindings": validate_attempt10_artifact_bindings,
    "ATTEMPT09_TEACHER_SCHEMA": ATTEMPT10_TEACHER_SCHEMA,
    "Attempt09TeacherConfig": Attempt10TeacherConfig,
    "validate_attempt09_teacher_output": validate_attempt10_teacher_output,
    "ATTEMPT09_AUTHORIZATION_SCHEMA": ATTEMPT10_AUTHORIZATION_SCHEMA,
    "ATTEMPT09_ROW_SCHEMA": ATTEMPT10_ROW_SCHEMA,
    "PACKAGE_SCHEMA": PACKAGE_SCHEMA,
    "LAUNCH_AUTHORIZATION_SCHEMA": LAUNCH_AUTHORIZATION_SCHEMA,
    "DONE_SCHEMA": DONE_SCHEMA,
    "RECEIVE_SCHEMA": RECEIVE_SCHEMA,
    "SHARD_SCHEMA": SHARD_SCHEMA,
    "LAUNCH_WAVE_SCHEMA": LAUNCH_WAVE_SCHEMA,
    "STATUS_SCHEMA": STATUS_SCHEMA,
    "RECEIVED_SHARD_AUDIT_SCHEMA": RECEIVED_SHARD_AUDIT_SCHEMA,
    "PREFLIGHT_RESULT_SCHEMA": PREFLIGHT_RESULT_SCHEMA,
    "SOURCE_NAME": SOURCE_NAME,
    "STARTUP_NAME": STARTUP_NAME,
    "PLAN_RELATIVE": PLAN_RELATIVE,
    "GATE_RELATIVE": GATE_RELATIVE,
    "OVERLAY_RELATIVES": OVERLAY_RELATIVES,
    "EXPECTED_GATES": EXPECTED_GATES,
}


@contextmanager
def _attempt10_bindings() -> Iterator[None]:
    """Install Attempt10 globals for one call, then restore Attempt09 exactly."""

    with _BIND_LOCK:
        prior = {name: getattr(_base, name) for name in _ATTEMPT10_BINDINGS}
        try:
            for name, value in _ATTEMPT10_BINDINGS.items():
                setattr(_base, name, value)
            yield
        finally:
            for name, value in prior.items():
                setattr(_base, name, value)


def build_schedule(mode: str, run_name: str) -> list[dict[str, Any]]:
    with _attempt10_bindings():
        return _base.build_schedule(mode, run_name)


def package_attempt10(
    *,
    mode: str,
    run_name: str,
    run_dir: str | Path,
    repository_root: str | Path = _REPO_ROOT,
    template_package: str | Path = DEFAULT_TEMPLATE_PACKAGE,
    plan: str | Path = DEFAULT_PLAN,
    startup: str | Path = DEFAULT_STARTUP,
    preceding_gate: str | Path | None = None,
) -> dict[str, Any]:
    with _attempt10_bindings():
        return _base.package_attempt09(
            mode=mode,
            run_name=run_name,
            run_dir=run_dir,
            repository_root=repository_root,
            template_package=template_package,
            plan=plan,
            startup=startup,
            preceding_gate=preceding_gate,
        )


def validate_package(run_dir: str | Path) -> dict[str, Any]:
    with _attempt10_bindings():
        return _base.validate_package(run_dir)


def authorize_launch(
    *, run_dir: str | Path, output: str | Path | None = None
) -> dict[str, Any]:
    with _attempt10_bindings():
        return _base.authorize_launch(run_dir=run_dir, output=output)


def validate_launch(run_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with _attempt10_bindings():
        return _base.validate_launch(run_dir)


def launch_wave(
    *,
    run_dir: str | Path,
    project: str,
    bucket: str,
    zone: str,
    shards: Sequence[str],
    no_self_delete: bool = False,
) -> dict[str, Any]:
    with _attempt10_bindings():
        return _base.launch_wave(
            run_dir=run_dir,
            project=project,
            bucket=bucket,
            zone=zone,
            shards=shards,
            no_self_delete=no_self_delete,
        )


def run_status(
    *, run_dir: str | Path, project: str, bucket: str, zone: str
) -> dict[str, Any]:
    with _attempt10_bindings():
        return _base.run_status(
            run_dir=run_dir, project=project, bucket=bucket, zone=zone
        )


def receive_run(
    *,
    run_dir: str | Path,
    project: str,
    bucket: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    with _attempt10_bindings():
        return _base.receive_run(
            run_dir=run_dir,
            project=project,
            bucket=bucket,
            output_dir=output_dir,
        )


def finalize_preflight(
    *, received_dir: str | Path, output: str | Path
) -> dict[str, Any]:
    with _attempt10_bindings():
        return _base.finalize_preflight(received_dir=received_dir, output=output)


sha256_file = _base.sha256_file
canonical_json_bytes = _base.canonical_json_bytes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    package = commands.add_parser("package")
    package.add_argument(
        "--mode", choices=("preflight", "development", "future_audit"), required=True
    )
    package.add_argument("--run-name", required=True)
    package.add_argument("--run-dir", type=Path, required=True)
    package.add_argument("--repository-root", type=Path, default=_REPO_ROOT)
    package.add_argument("--template-package", type=Path, default=DEFAULT_TEMPLATE_PACKAGE)
    package.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    package.add_argument("--startup", type=Path, default=DEFAULT_STARTUP)
    package.add_argument("--preceding-gate", type=Path)
    authorize = commands.add_parser("authorize")
    authorize.add_argument("--run-dir", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--run-dir", type=Path, required=True)
    launch = commands.add_parser("launch")
    launch.add_argument("--run-dir", type=Path, required=True)
    launch.add_argument("--project", default="ofc-solver-485418")
    launch.add_argument("--bucket", default="pokerhu-ofc-solver-485418-training")
    launch.add_argument("--zone", default="asia-northeast1-b")
    launch.add_argument("--shards", action="append", required=True)
    launch.add_argument("--no-self-delete", action="store_true")
    status = commands.add_parser("status")
    status.add_argument("--run-dir", type=Path, required=True)
    status.add_argument("--project", default="ofc-solver-485418")
    status.add_argument("--bucket", default="pokerhu-ofc-solver-485418-training")
    status.add_argument("--zone", default="asia-northeast1-b")
    receive = commands.add_parser("receive")
    receive.add_argument("--run-dir", type=Path, required=True)
    receive.add_argument("--output-dir", type=Path, required=True)
    receive.add_argument("--project", default="ofc-solver-485418")
    receive.add_argument("--bucket", default="pokerhu-ofc-solver-485418-training")
    final = commands.add_parser("finalize-preflight")
    final.add_argument("--received-dir", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "package":
        result = package_attempt10(
            mode=args.mode,
            run_name=args.run_name,
            run_dir=args.run_dir,
            repository_root=args.repository_root,
            template_package=args.template_package,
            plan=args.plan,
            startup=args.startup,
            preceding_gate=args.preceding_gate,
        )
    elif args.command == "authorize":
        result = authorize_launch(run_dir=args.run_dir)
    elif args.command == "validate":
        manifest, authorization = validate_launch(args.run_dir)
        result = {
            "status": "valid",
            "manifest": manifest,
            "authorization": authorization,
        }
    elif args.command == "launch":
        result = launch_wave(
            run_dir=args.run_dir,
            project=args.project,
            bucket=args.bucket,
            zone=args.zone,
            shards=args.shards,
            no_self_delete=args.no_self_delete,
        )
    elif args.command == "status":
        result = run_status(
            run_dir=args.run_dir,
            project=args.project,
            bucket=args.bucket,
            zone=args.zone,
        )
    elif args.command == "receive":
        result = receive_run(
            run_dir=args.run_dir,
            project=args.project,
            bucket=args.bucket,
            output_dir=args.output_dir,
        )
    elif args.command == "finalize-preflight":
        result = finalize_preflight(
            received_dir=args.received_dir, output=args.output
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DONE_SCHEMA",
    "LAUNCH_AUTHORIZATION_SCHEMA",
    "PACKAGE_SCHEMA",
    "RECEIVE_SCHEMA",
    "authorize_launch",
    "build_schedule",
    "finalize_preflight",
    "launch_wave",
    "package_attempt10",
    "receive_run",
    "run_status",
    "validate_launch",
    "validate_package",
]
