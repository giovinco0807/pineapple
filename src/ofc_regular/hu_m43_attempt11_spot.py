"""Attempt11 binding for the immutable Attempt09 Spot lifecycle.

The lifecycle mechanics are shared deliberately: packaging, launch, status,
receive, and preflight finalization have no search-policy semantics of their
own.  This module binds that machinery to the frozen Attempt11 contract for
one synchronous call and restores every Attempt09 global afterwards.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from . import hu_m43_attempt09_spot as _base
from .hu_m43_attempt11_contract import (
    AI_PROFILES_SHA256,
    ATTEMPT11_LAMBDA_MODEL_SHA256,
    M43_ATTEMPT11_PLAN_SHA256,
    M43_ATTEMPT11_PROFILES,
    enumerate_attempt11_seed_schedules,
    load_and_validate_attempt11_plan,
    validate_attempt11_artifact_bindings,
)
from .hu_m43_attempt11_teacher import (
    ATTEMPT11_TEACHER_SCHEMA,
    Attempt11TeacherConfig,
    validate_attempt11_teacher_output,
)
from .run_hu_m43_attempt11 import (
    ATTEMPT11_AUTHORIZATION_SCHEMA,
    ATTEMPT11_ROW_SCHEMA,
)


PACKAGE_SCHEMA = "hu_m43_attempt11_spot_package_v1"
LAUNCH_AUTHORIZATION_SCHEMA = "hu_m43_attempt11_spot_launch_authorization_v1"
DONE_SCHEMA = "hu_m43_attempt11_done_v1"
RECEIVE_SCHEMA = "hu_m43_attempt11_receive_v1"
SHARD_SCHEMA = "hu_m43_attempt11_spot_shard_v1"
LAUNCH_WAVE_SCHEMA = "hu_m43_attempt11_launch_wave_v1"
STATUS_SCHEMA = "hu_m43_attempt11_status_v1"
RECEIVED_SHARD_AUDIT_SCHEMA = "hu_m43_attempt11_received_shard_audit_v1"
PREFLIGHT_RESULT_SCHEMA = "hu_m43_attempt11_preflight_result_v1"
SOURCE_NAME = "ofc_regular_hu_m43_attempt11_source.zip"
SCHEDULE_NAME = _base.SCHEDULE_NAME
STARTUP_NAME = "startup_hu_m43_attempt11.sh"

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = _REPO_ROOT / "configs" / "hu_joint_policy_m43_attempt11.json"
DEFAULT_TEMPLATE_PACKAGE = _base.DEFAULT_TEMPLATE_PACKAGE
DEFAULT_STARTUP = _REPO_ROOT / "scripts" / STARTUP_NAME
PLAN_RELATIVE = "configs/hu_joint_policy_m43_attempt11.json"
GATE_RELATIVE = "artifacts/attempt11/preceding_gate.json"
OVERLAY_RELATIVES = (
    PLAN_RELATIVE,
    "configs/hu_joint_policy_m43_attempt10.json",
    "src/ofc_regular/hu_m43_attempt11_contract.py",
    "src/ofc_regular/hu_m43_attempt11_teacher.py",
    "src/ofc_regular/run_hu_m43_attempt11.py",
    # The generic root persistence runner and its eager import closure are
    # required by run_hu_m43_attempt11 inside the immutable VM package.
    "src/ofc_regular/run_hu_m43_attempt09.py",
    "src/ofc_regular/hu_m43_attempt09_contract.py",
    "src/ofc_regular/hu_m43_attempt09_teacher.py",
)
EXPECTED_GATES = {
    "preflight": ("pass_local_correctness", "authorize_preflight_only"),
    "development": (
        "pass_correctness_preflight",
        "authorize_development200_package_only",
    ),
    "future_audit": ("go_freeze_attempt11_development", "go"),
}

_BIND_LOCK = threading.RLock()


def _build_attempt11_schedule(mode: str, run_name: str) -> list[dict[str, Any]]:
    if not _base._SAFE_RUN.fullmatch(run_name):
        raise ValueError("Attempt11 run_name is not a safe GCP identity")
    if mode == "preflight":
        specs = [
            (0, True, "root0_batch_a"),
            (0, True, "root0_batch_b"),
            (0, False, "root0_scalar"),
            (1, True, "root1_batch"),
            (2, True, "root2_batch"),
            (3, True, "root3_batch"),
            (4, True, "root4_batch"),
        ]
    elif mode == "development":
        specs = [(index, True, f"root{index:03d}") for index in range(200)]
    elif mode == "future_audit":
        specs = [(index, True, f"root{index:03d}") for index in range(200, 250)]
    else:
        raise ValueError("mode must be preflight, development, or future_audit")
    rows: list[dict[str, Any]] = []
    for shard, (root_index, batch, slot) in enumerate(specs):
        rows.append(
            {
                "schema": SHARD_SCHEMA,
                "run_name": run_name,
                "mode": mode,
                "shard": shard,
                "root_index": root_index,
                "root_profile": M43_ATTEMPT11_PROFILES[
                    root_index % len(M43_ATTEMPT11_PROFILES)
                ],
                "batch_child_selectors": batch,
                "native_batch_threads": 4,
                "run_id": f"{run_name}:{mode}:root={root_index}:"
                + ("parity" if mode == "preflight" else "search"),
                "output_prefix": f"shard-{shard:03d}-{slot}",
            }
        )
    return rows


_ATTEMPT11_BINDINGS: dict[str, Any] = {
    "AI_PROFILES_SHA256": AI_PROFILES_SHA256,
    "ATTEMPT09_LAMBDA_MODEL_SHA256": ATTEMPT11_LAMBDA_MODEL_SHA256,
    "M43_ATTEMPT09_PLAN_SHA256": M43_ATTEMPT11_PLAN_SHA256,
    "M43_ATTEMPT09_PROFILES": M43_ATTEMPT11_PROFILES,
    "enumerate_attempt09_seed_schedules": enumerate_attempt11_seed_schedules,
    "load_and_validate_attempt09_plan": load_and_validate_attempt11_plan,
    "validate_attempt09_artifact_bindings": validate_attempt11_artifact_bindings,
    "ATTEMPT09_TEACHER_SCHEMA": ATTEMPT11_TEACHER_SCHEMA,
    "Attempt09TeacherConfig": Attempt11TeacherConfig,
    "validate_attempt09_teacher_output": validate_attempt11_teacher_output,
    "ATTEMPT09_AUTHORIZATION_SCHEMA": ATTEMPT11_AUTHORIZATION_SCHEMA,
    "ATTEMPT09_ROW_SCHEMA": ATTEMPT11_ROW_SCHEMA,
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
    "build_schedule": _build_attempt11_schedule,
}


@contextmanager
def _attempt11_bindings() -> Iterator[None]:
    """Install Attempt11 globals for one call, then restore Attempt09 exactly."""

    with _BIND_LOCK:
        prior = {name: getattr(_base, name) for name in _ATTEMPT11_BINDINGS}
        try:
            for name, value in _ATTEMPT11_BINDINGS.items():
                setattr(_base, name, value)
            yield
        finally:
            for name, value in prior.items():
                setattr(_base, name, value)


def build_schedule(mode: str, run_name: str) -> list[dict[str, Any]]:
    with _attempt11_bindings():
        return _base.build_schedule(mode, run_name)


def package_attempt11(
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
    with _attempt11_bindings():
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
    with _attempt11_bindings():
        return _base.validate_package(run_dir)


def authorize_launch(
    *, run_dir: str | Path, output: str | Path | None = None
) -> dict[str, Any]:
    with _attempt11_bindings():
        return _base.authorize_launch(run_dir=run_dir, output=output)


def validate_launch(run_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with _attempt11_bindings():
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
    with _attempt11_bindings():
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
    with _attempt11_bindings():
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
    with _attempt11_bindings():
        return _base.receive_run(
            run_dir=run_dir,
            project=project,
            bucket=bucket,
            output_dir=output_dir,
        )


def _validate_root4_variable_candidate_teacher(teacher: dict[str, Any]) -> int:
    n = teacher.get("candidate_nonbaseline_count")
    if type(n) is not int or not 0 <= n <= 12:
        raise ValueError("Attempt11 root4 candidate count is outside 0..12")
    baseline = teacher.get("baseline_action_key")
    candidate_keys = teacher.get("learned_nonbaseline_action_keys")
    proposal = teacher.get("proposal_mapping")
    rerank = teacher.get("rerank")
    shortlist = teacher.get("shortlist")
    veto = teacher.get("veto")
    if not all(
        isinstance(value, dict) for value in (proposal, rerank, shortlist, veto)
    ) or not isinstance(candidate_keys, list):
        raise ValueError("Attempt11 root4 variable candidate mappings are missing")
    proposal_keys = proposal.get("action_keys")
    k = min(8, n)
    if (
        len(candidate_keys) != n
        or len(set(candidate_keys)) != n
        or baseline in candidate_keys
        or proposal.get("action_count") != n + 1
        or proposal_keys != [*candidate_keys, baseline]
        or len(set(proposal_keys or ())) != n + 1
        or rerank.get("action_count") != n + 1
        or rerank.get("action_keys") != proposal_keys
        or shortlist.get("nonbaseline_count") != k
        or shortlist.get("action_count") != k
        or len(set(shortlist.get("action_keys", ()))) != k
        or veto.get("action_count") != k + 1
        or veto.get("action_keys") != [*shortlist.get("action_keys", ()), baseline]
    ):
        raise ValueError("Attempt11 root4 variable candidate mapping changed")
    return n


def _finalize_attempt11_preflight(
    *, received_dir: str | Path, output: str | Path
) -> dict[str, Any]:
    root = Path(received_dir).resolve()
    receipt_path = root / "merged" / "receive_receipt.json"
    receipt = _base._load_canonical(receipt_path, "Attempt11 preflight receipt")
    expected_root_indices = [0, 0, 0, 1, 2, 3, 4]
    expected_profiles = [
        M43_ATTEMPT11_PROFILES[index % len(M43_ATTEMPT11_PROFILES)]
        for index in expected_root_indices
    ]
    if (
        receipt.get("schema") != RECEIVE_SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("mode") != "preflight"
        or receipt.get("roots") != 7
        or receipt.get("root_indices") != expected_root_indices
        or receipt.get("profiles") != expected_profiles
        or len(set(expected_profiles)) != 5
        or receipt.get("batch_boundary_validation_count") != 1
        or receipt.get("per_shard_boundary_revalidation_count") != 0
        or receipt.get("selector_executed") is not False
        or receipt.get("current_profile_mutated") is not False
        or receipt.get("runtime_policy_activated") is not False
    ):
        raise ValueError("Attempt11 preflight requires exact seven-slot full-profile proof")
    merged_path = root / "merged" / "teacher.jsonl"
    if _base.sha256_file(merged_path) != _base._require_sha256(
        receipt.get("merged_sha256"), "Attempt11 preflight merged"
    ):
        raise ValueError("Attempt11 preflight merged bytes changed")
    if {path.name for path in root.iterdir()} != {"shards", "audits", "merged"}:
        raise ValueError("Attempt11 preflight receive root changed")
    audit_paths = sorted((root / "audits").glob("shard-*.json"))
    expected_audits = [f"shard-{index:03d}.json" for index in range(7)]
    if [path.name for path in audit_paths] != expected_audits:
        raise ValueError("Attempt11 preflight audit exact set changed")
    audit_payloads = [
        _base._load_canonical(path, f"Attempt11 preflight audit {index}")
        for index, path in enumerate(audit_paths)
    ]
    if hashlib.sha256(_base._canonical_jsonl(audit_payloads)).hexdigest() != (
        _base._require_sha256(receipt.get("audit_sha256"), "Attempt11 audits")
    ):
        raise ValueError("Attempt11 preflight audit bytes changed")
    shard_rows: list[dict[str, Any]] = []
    for shard in range(7):
        directories = sorted((root / "shards").glob(f"shard-{shard:03d}-*"))
        if len(directories) != 1:
            raise ValueError(f"Attempt11 preflight shard {shard} is missing")
        shard_rows.append(json.loads((directories[0] / "teacher.jsonl").read_bytes()))
    teacher_a = shard_rows[0]["teacher"]
    teacher_b = shard_rows[1]["teacher"]
    deterministic = _base.canonical_json_bytes(teacher_a) == _base.canonical_json_bytes(
        teacher_b
    )
    if not deterministic:
        raise ValueError("Attempt11 repeated batch root is not byte-identical")
    batch = copy.deepcopy(teacher_a)
    scalar = copy.deepcopy(shard_rows[2]["teacher"])
    batch["search_config"]["batch_child_selectors"] = False
    scalar["search_config"]["batch_child_selectors"] = False
    scalar_batch = _base.canonical_json_bytes(batch) == _base.canonical_json_bytes(
        scalar
    )
    if not scalar_batch:
        raise ValueError("Attempt11 scalar/batch teacher payloads differ")
    root4_candidate_count = _validate_root4_variable_candidate_teacher(
        shard_rows[6]["teacher"]
    )
    inherited = (
        _REPO_ROOT
        / "outputs"
        / "hu_joint_policy"
        / "m43_attempt08_preflight"
        / "regular-hu-m43-attempt08-preflight-finalprop-20260714-213558"
        / "finalization.json"
    )
    inherited_hash = "681854a8cd37bec1abf6f0ff72e69a8e09e2ad257fe2ecfbea513d3f127dd1b3"
    if not inherited.is_file() or _base.sha256_file(inherited) != inherited_hash:
        raise ValueError("Attempt11 inherited exact-engine parity evidence changed")
    inherited_payload = _base._load_canonical(
        inherited, "Attempt08 exact parity finalization"
    )
    if inherited_payload.get("status") != (
        "pass_correctness_preflight_and_authorize_development200_only"
    ):
        raise ValueError("Attempt11 inherited exact-engine parity did not pass")
    elapsed = [float(row["provenance"]["elapsed_seconds"]) for row in shard_rows]
    result = {
        "schema": PREFLIGHT_RESULT_SCHEMA,
        "status": "pass_correctness_preflight",
        "decision": "authorize_development200_package_only",
        "received_receipt_sha256": _base.sha256_file(receipt_path),
        "deterministic_batch_repeat": deterministic,
        "scalar_batch_teacher_parity": scalar_batch,
        "full_profile_coverage": set(expected_profiles)
        == set(M43_ATTEMPT11_PROFILES),
        "root4_profile": expected_profiles[6],
        "root4_candidate_nonbaseline_count": root4_candidate_count,
        "root4_variable_candidate_contract_validated": True,
        "action_mapping_validated": True,
        "hidden_information_violations": 0,
        "rng_domain_overlap_violations": 0,
        "inherited_t3_t4_exact_parity_sha256": inherited_hash,
        "latency_seconds": elapsed,
        "latency_max_seconds": max(elapsed),
        "development_started": False,
        "future_audit_authorized": False,
        "fit_started": False,
        "runtime_policy_activated": False,
        "current_profile_mutated": False,
    }
    _base._write_once(Path(output).resolve(), _base.canonical_json_bytes(result))
    return result


def finalize_preflight(
    *, received_dir: str | Path, output: str | Path
) -> dict[str, Any]:
    with _attempt11_bindings():
        return _finalize_attempt11_preflight(
            received_dir=received_dir, output=output
        )


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
        result = package_attempt11(
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
    "package_attempt11",
    "receive_run",
    "run_status",
    "validate_launch",
    "validate_package",
]
