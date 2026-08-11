"""Resumable post-dataset controller for M3.1 locked promotion.

The controller deliberately reuses the already-reviewed training, runtime
closure, ABR, runner, merge, and gate implementations.  Its only scientific
responsibility is operational completeness:

* inspect the exact 260-item execution plan in source order;
* validate every existing shard before treating it as complete;
* run a bounded pending subset with one already-bound real runner;
* publish each shard atomically and resume from the remaining set;
* close out only the exact 260-shard directory;
* issue a dormant opt-in registration authorization only after a passing gate.

It does not talk to a cloud provider, mutate ``ai_profiles.py``, resolve
``current``, activate a runtime, or enable a full replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import hu_m31_t3_locked_promotion_execution_v1 as execution
from . import hu_m31_t3_locked_promotion_production_v1 as production
from . import hu_m31_t3_locked_promotion_provider_v1 as provider
from . import hu_m31_t3_step6d_locked_promotion_v1 as promotion
from . import hu_m31_t3_street_policy_runtime_v1 as policy_runtime


PROGRESS_SCHEMA = "hu_m31_t3_post_dataset_progress_v1"
BATCH_RECEIPT_SCHEMA = "hu_m31_t3_post_dataset_batch_receipt_v1"
OPT_IN_AUTHORIZATION_SCHEMA = (
    "hu_m31_t3_explicit_opt_in_registration_authorization_v1"
)


def _canonical_bytes(value: Any) -> bytes:
    return promotion.canonical_bytes(value)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_canonical(
    path: str | Path, label: str
) -> tuple[dict[str, Any], bytes]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value):
        raise ValueError(f"{label} is not a canonical JSON object")
    return value, raw


def _write_once_or_replay(path: Path, value: Mapping[str, Any]) -> None:
    raw = _canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if (
            path.is_symlink()
            or not path.is_file()
            or path.read_bytes() != raw
        ):
            raise FileExistsError(
                f"immutable post-dataset artifact changed: {path}"
            ) from None


def _atomic_publish_shard(
    shard_directory: Path,
    filename: str,
    shard: Mapping[str, Any],
) -> Path:
    """Publish create-only bytes without leaving partial files in shard root."""

    if Path(filename).name != filename:
        raise ValueError("locked-promotion shard filename is unsafe")
    shard_directory.mkdir(parents=True, exist_ok=True)
    if shard_directory.is_symlink() or not shard_directory.is_dir():
        raise ValueError("locked-promotion shard directory is unsafe")
    staging_directory = shard_directory.parent / (
        f".{shard_directory.name}.staging"
    )
    staging_directory.mkdir(parents=True, exist_ok=True)
    if staging_directory.is_symlink() or not staging_directory.is_dir():
        raise ValueError("locked-promotion staging directory is unsafe")
    destination = shard_directory / filename
    raw = _canonical_bytes(shard)
    temporary = staging_directory / f"{filename}.{os.getpid()}.tmp"
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"stale shard staging file exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if (
                destination.is_symlink()
                or not destination.is_file()
                or destination.read_bytes() != raw
            ):
                raise FileExistsError(
                    f"immutable locked-promotion shard changed: {destination}"
                ) from None
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination.resolve()


def inspect_evaluation_progress(
    *,
    promotion_plan: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    shard_directory: str | Path,
) -> dict[str, Any]:
    """Source-replay all existing files and report exact pending work IDs."""

    plan = promotion.validate_locked_promotion_plan(promotion_plan)
    frozen = execution.validate_execution_plan(
        execution_plan, promotion_plan=plan
    )
    directory = Path(shard_directory)
    if directory.exists() and (
        directory.is_symlink() or not directory.is_dir()
    ):
        raise ValueError("locked-promotion shard directory is unsafe")
    actual: dict[str, Path] = {}
    if directory.exists():
        for path in directory.iterdir():
            if path.is_symlink() or not path.is_file():
                raise ValueError(
                    "locked-promotion shard directory has unsafe entries"
                )
            actual[path.name] = path
    expected_names = {
        str(item["output_filename"]) for item in frozen["work_items"]
    }
    unknown = sorted(set(actual) - expected_names)
    if unknown:
        raise ValueError(
            f"locked-promotion shard directory has extra files: {unknown}"
        )
    complete: list[str] = []
    pending: list[str] = []
    shard_records: list[dict[str, Any]] = []
    for item in frozen["work_items"]:
        filename = str(item["output_filename"])
        path = actual.get(filename)
        if path is None:
            pending.append(str(item["work_id"]))
            continue
        shard, raw = _read_canonical(path, "locked-promotion work shard")
        validated = execution.validate_work_item_shard(
            work_item=item,
            shard=shard,
            promotion_plan=plan,
        )
        complete.append(str(item["work_id"]))
        shard_records.append(
            {
                "work_id": item["work_id"],
                "output_filename": filename,
                "file_sha256": _sha256(raw),
                "shard_sha256": promotion.canonical_sha256(validated),
                "row_count": validated["row_count"],
            }
        )
    identity = {
        "schema": PROGRESS_SCHEMA,
        "status": "complete" if not pending else "incomplete",
        "promotion_plan_sha256": promotion.canonical_sha256(plan),
        "execution_plan_sha256": promotion.canonical_sha256(frozen),
        "expected_work_item_count": execution.EXPECTED_WORK_ITEMS,
        "expected_row_count": execution.EXPECTED_ROWS,
        "complete_work_item_count": len(complete),
        "pending_work_item_count": len(pending),
        "complete_work_ids": complete,
        "pending_work_ids": pending,
        "shards": shard_records,
        "accepted_row_count": sum(
            int(record["row_count"]) for record in shard_records
        ),
        "named_profile_added": False,
        "current_profile_changed": False,
        "runtime_activated": False,
        "full_replacement_enabled": False,
    }
    return {
        **identity,
        "progress_sha256": promotion.canonical_sha256(identity),
    }


def run_pending_work_items(
    *,
    runner: Any,
    promotion_plan: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    shard_directory: str | Path,
    max_items: int,
    requested_work_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run at most ``max_items`` pending shards and return a resume receipt."""

    if (
        isinstance(max_items, bool)
        or not isinstance(max_items, int)
        or max_items <= 0
    ):
        raise ValueError("max_items must be a positive integer")
    plan = promotion.validate_locked_promotion_plan(promotion_plan)
    frozen = execution.validate_execution_plan(
        execution_plan, promotion_plan=plan
    )
    runner_plan = getattr(runner, "plan", None)
    run_shard = getattr(runner, "run_shard", None)
    if (
        not isinstance(runner_plan, Mapping)
        or promotion.validate_locked_promotion_plan(runner_plan) != plan
        or not callable(run_shard)
    ):
        raise ValueError("runner is not bound to this promotion plan")
    before = inspect_evaluation_progress(
        promotion_plan=plan,
        execution_plan=frozen,
        shard_directory=shard_directory,
    )
    pending = set(before["pending_work_ids"])
    by_id = {str(item["work_id"]): item for item in frozen["work_items"]}
    if requested_work_ids is None:
        selected = [
            str(item["work_id"])
            for item in frozen["work_items"]
            if item["work_id"] in pending
        ][:max_items]
    else:
        requested = list(requested_work_ids)
        if (
            not requested
            or len(requested) != len(set(requested))
            or any(work_id not in by_id for work_id in requested)
        ):
            raise ValueError("requested work IDs are empty, duplicated, or unknown")
        already_complete = [work_id for work_id in requested if work_id not in pending]
        if already_complete:
            raise ValueError(
                f"requested work IDs are already complete: {already_complete}"
            )
        selected = requested[:max_items]
    completed_now: list[dict[str, Any]] = []
    for work_id in selected:
        item = by_id[work_id]
        shard = run_shard(
            schedule=item["schedule"],
            entity_id=item["entity_id"],
            seed_indices=range(
                int(item["seed_index_start"]),
                int(item["seed_index_stop_exclusive"]),
            ),
            shard_id=item["work_id"],
            output_path=None,
        )
        validated = execution.validate_work_item_shard(
            work_item=item,
            shard=shard,
            promotion_plan=plan,
        )
        path = _atomic_publish_shard(
            Path(shard_directory),
            str(item["output_filename"]),
            validated,
        )
        completed_now.append(
            {
                "work_id": work_id,
                "output_filename": path.name,
                "file_sha256": _sha256(path.read_bytes()),
                "shard_sha256": promotion.canonical_sha256(validated),
                "row_count": validated["row_count"],
            }
        )
    after = inspect_evaluation_progress(
        promotion_plan=plan,
        execution_plan=frozen,
        shard_directory=shard_directory,
    )
    identity = {
        "schema": BATCH_RECEIPT_SCHEMA,
        "status": (
            "complete_all_260" if not after["pending_work_ids"] else "resume_required"
        ),
        "promotion_plan_sha256": promotion.canonical_sha256(plan),
        "execution_plan_sha256": promotion.canonical_sha256(frozen),
        "progress_sha256_before": before["progress_sha256"],
        "progress_sha256_after": after["progress_sha256"],
        "requested_max_items": max_items,
        "selected_work_ids": selected,
        "completed_now": completed_now,
        "complete_work_item_count": after["complete_work_item_count"],
        "pending_work_item_count": after["pending_work_item_count"],
        "pending_work_ids": after["pending_work_ids"],
        "named_profile_added": False,
        "current_profile_changed": False,
        "runtime_activated": False,
        "full_replacement_enabled": False,
    }
    return {
        **identity,
        "batch_receipt_sha256": promotion.canonical_sha256(identity),
    }


def build_opt_in_registration_authorization(
    *,
    promotion_plan: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    shard_directory: str | Path,
    merge: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Authorize, but do not apply, the one explicit opt-in registry change."""

    plan = promotion.validate_locked_promotion_plan(promotion_plan)
    frozen = execution.validate_execution_plan(
        execution_plan, promotion_plan=plan
    )
    progress = inspect_evaluation_progress(
        promotion_plan=plan,
        execution_plan=frozen,
        shard_directory=shard_directory,
    )
    if (
        progress["status"] != "complete"
        or progress["complete_work_item_count"] != execution.EXPECTED_WORK_ITEMS
        or progress["accepted_row_count"] != execution.EXPECTED_ROWS
    ):
        raise PermissionError(
            "opt-in registration requires the exact complete 260-shard grid"
        )
    replayed_merge, replayed_gate = execution.build_closeout(
        promotion_plan=plan,
        execution_plan=frozen,
        shard_directory=shard_directory,
    )
    if dict(merge) != replayed_merge or dict(gate) != replayed_gate:
        raise ValueError("opt-in registration evidence differs from closeout replay")
    validated_gate = promotion.validate_locked_promotion_gate(
        gate,
        plan=plan,
        merge=merge,
        replay_sources=True,
    )
    gates = validated_gate.get("gates")
    if (
        validated_gate.get("status") != "pass"
        or validated_gate.get("all_gates_passed") is not True
        or validated_gate.get("scientific_promotion_passed") is not True
        or validated_gate.get(
            "separate_opt_in_profile_candidate_authorized"
        )
        is not True
        or not isinstance(gates, Mapping)
        or not gates
        or any(value is not True for value in gates.values())
        or validated_gate.get("named_profile_added") is not False
        or validated_gate.get("current_profile_changed") is not False
        or validated_gate.get("runtime_activated") is not False
        or validated_gate.get("full_replacement_enabled") is not False
    ):
        raise PermissionError("locked promotion gate did not authorize opt-in")
    binding = plan["artifact_binding"]
    identity = {
        "schema": OPT_IN_AUTHORIZATION_SCHEMA,
        "status": "qualified_explicit_opt_in_registration_authorized_not_applied",
        "profile_id": policy_runtime.OPT_IN_PROFILE_CANDIDATE,
        "baseline_profile_id": policy_runtime.BASELINE_PROFILE,
        "runtime_factory_module": (
            "ofc_regular.hu_m31_t3_street_policy_runtime_v1"
        ),
        "runtime_factory_symbol": "build_opt_in_t3_policy_candidate",
        "promotion_plan_sha256": promotion.canonical_sha256(plan),
        "execution_plan_sha256": promotion.canonical_sha256(frozen),
        "merge_sha256": promotion.canonical_sha256(merge),
        "gate_sha256": promotion.canonical_sha256(validated_gate),
        "model_manifest_sha256": binding["model"]["sha256"],
        "compatibility_threshold_lock_sha256": binding["threshold_lock"][
            "sha256"
        ],
        "policy_registry_sha256_before": binding["policy_registry"]["sha256"],
        "evaluation_runtime_closure_sha256": binding[
            "evaluation_runtime_closure"
        ]["sha256"],
        "registration_scope": "one_explicit_named_profile_only",
        "implicit_current_resolution_allowed": False,
        "full_replacement_allowed": False,
        "registration_authorized": True,
        "registration_applied": False,
        "named_profile_added": False,
        "current_profile_changed": False,
        "runtime_activated": False,
        "full_replacement_enabled": False,
    }
    return {
        **identity,
        "authorization_sha256": promotion.canonical_sha256(identity),
    }


def _load_plan_and_execution(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan, _ = _read_canonical(args.promotion_plan, "locked promotion plan")
    frozen, _ = _read_canonical(
        args.execution_plan, "locked-promotion execution plan"
    )
    return plan, frozen


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M3.1 deterministic post-dataset controller"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    status = subparsers.add_parser("status")
    run = subparsers.add_parser("run-pending")
    closeout = subparsers.add_parser("closeout")
    authorize = subparsers.add_parser("authorize-opt-in")
    for command in (status, run, closeout, authorize):
        command.add_argument("--promotion-plan", required=True)
        command.add_argument("--execution-plan", required=True)
        command.add_argument("--shard-directory", required=True)
    run.add_argument("--closure-package", required=True)
    run.add_argument("--expected-closure-sha256", required=True)
    run.add_argument("--extraction-root", required=True)
    run.add_argument("--source-replay-root", required=True)
    run.add_argument("--compatibility-threshold-lock", required=True)
    run.add_argument("--policy-registry", required=True)
    run.add_argument("--abr-bundle-directory", required=True)
    run.add_argument("--expected-abr-bundle-sha256", required=True)
    run.add_argument(
        "--expected-abr-production-build-receipt-sha256",
        required=True,
    )
    run.add_argument("--max-items", type=int, required=True)
    run.add_argument("--work-id", action="append")
    run.add_argument("--output-receipt")
    closeout.add_argument("--merge-output", required=True)
    closeout.add_argument("--gate-output", required=True)
    authorize.add_argument("--merge", required=True)
    authorize.add_argument("--gate", required=True)
    authorize.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan, frozen = _load_plan_and_execution(args)
        if args.command == "status":
            result = inspect_evaluation_progress(
                promotion_plan=plan,
                execution_plan=frozen,
                shard_directory=args.shard_directory,
            )
        elif args.command == "run-pending":
            if not _is_sha256(args.expected_closure_sha256):
                raise ValueError("closure SHA-256 must be lowercase hex")
            import torch

            prepared = production.prepare_locked_execution_plan(
                plan_path=args.promotion_plan,
                closure_package_path=args.closure_package,
                expected_closure_package_sha256=(
                    args.expected_closure_sha256
                ),
                extraction_root=args.extraction_root,
                source_replay_root=args.source_replay_root,
                compatibility_threshold_lock_path=(
                    args.compatibility_threshold_lock
                ),
                policy_registry_path=args.policy_registry,
                require_host_target=True,
            )
            runner = provider.build_bound_runner(
                prepared=prepared,
                abr_bundle_directory=args.abr_bundle_directory,
                expected_abr_bundle_file_sha256=(
                    args.expected_abr_bundle_sha256
                ),
                expected_abr_production_build_receipt_sha256=(
                    args.expected_abr_production_build_receipt_sha256
                ),
                torch=torch,
            )
            result = run_pending_work_items(
                runner=runner,
                promotion_plan=plan,
                execution_plan=frozen,
                shard_directory=args.shard_directory,
                max_items=args.max_items,
                requested_work_ids=args.work_id,
            )
            if args.output_receipt:
                _write_once_or_replay(Path(args.output_receipt), result)
        elif args.command == "closeout":
            merge, gate = execution.write_closeout(
                promotion_plan=plan,
                execution_plan=frozen,
                shard_directory=args.shard_directory,
                merge_output_path=args.merge_output,
                gate_output_path=args.gate_output,
            )
            result = {
                "status": gate["status"],
                "merge_sha256": promotion.canonical_sha256(merge),
                "gate_sha256": promotion.canonical_sha256(gate),
                "scientific_promotion_passed": gate[
                    "scientific_promotion_passed"
                ],
                "named_profile_added": False,
                "current_profile_changed": False,
                "runtime_activated": False,
            }
        else:
            merge, _ = _read_canonical(args.merge, "locked promotion merge")
            gate, _ = _read_canonical(args.gate, "locked promotion gate")
            result = build_opt_in_registration_authorization(
                promotion_plan=plan,
                execution_plan=frozen,
                shard_directory=args.shard_directory,
                merge=merge,
                gate=gate,
            )
            _write_once_or_replay(Path(args.output), result)
    except Exception as exc:  # pragma: no cover - subprocess boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BATCH_RECEIPT_SCHEMA",
    "OPT_IN_AUTHORIZATION_SCHEMA",
    "PROGRESS_SCHEMA",
    "build_opt_in_registration_authorization",
    "inspect_evaluation_progress",
    "main",
    "run_pending_work_items",
]
