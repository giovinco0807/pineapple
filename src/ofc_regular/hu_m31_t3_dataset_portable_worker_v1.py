"""Portable, hash-pinned worker boundary for M3.1 dataset Spot shards.

Fresh-quality gates intentionally retain their controller-local source paths.
Replaying those Windows paths on a Linux Spot worker is impossible and must
not be faked.  This module performs the full source replay once on the trusted
controller, freezes the result in an externally staged receipt, and lets a
worker verify the exact gate bytes plus the fully replayable smoke shard.

The receipt authorizes only the already frozen post-smoke shard grid.  It does
not launch cloud resources and cannot change an AI profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import hu_m31_t3_dataset_contract_v1 as contract
from . import hu_m31_t3_dataset_executor_v1 as executor
from . import hu_m31_t3_step6d_fresh_quality_gate_v1 as quality_gate


PORTABLE_AUTHORIZATION_SCHEMA = (
    "hu_m31_t3_dataset_portable_fanout_authorization_v1"
)

_KEYS = frozenset(
    {
        "schema",
        "status",
        "plan_sha256",
        "fresh_quality_gate_schema",
        "fresh_quality_gate_file_sha256",
        "fresh_quality_gate_canonical_sha256",
        "fresh_quality_merge_sha256",
        "fresh_quality_decision",
        "dataset_smoke_gate_schema",
        "dataset_smoke_gate_file_sha256",
        "dataset_smoke_gate_canonical_sha256",
        "dataset_smoke_gate_decision",
        "smoke_shard_id",
        "smoke_shard_done_sha256",
        "smoke_shard_inventory",
        "smoke_shard_inventory_sha256",
        "current_profile_registry_sha256",
        "controller_source_replayed",
        "worker_replay_boundary",
        "full_9000_paired_fanout_authorized",
        "teacher_values_are_realized_match_ev",
        "current_profile_changed",
        "authorization_sha256",
    }
)
_INVENTORY_KEYS = frozenset({"path", "sha256", "bytes"})


def canonical_bytes(value: Any) -> bytes:
    return executor.canonical_bytes(value)


def canonical_sha256(value: Any) -> str:
    return executor.canonical_sha256(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_canonical(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    dataset_raw = canonical_bytes(value)
    if (
        not isinstance(value, dict)
        or raw not in (dataset_raw, dataset_raw.removesuffix(b"\n"))
    ):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite immutable artifact: {path}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _inventory(directory: Path) -> list[dict[str, Any]]:
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("smoke shard directory is missing or unsafe")
    records: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("smoke shard contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("smoke shard contains a non-regular entry")
        relative = path.relative_to(directory).as_posix()
        if relative.startswith("../") or relative.startswith("/"):
            raise ValueError("smoke shard inventory path escaped its root")
        records.append(
            {
                "path": relative,
                "sha256": _file_sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    if not records:
        raise ValueError("smoke shard inventory is empty")
    return records


def build_portable_fanout_authorization(
    *,
    plan: Mapping[str, Any],
    fresh_quality_gate_path: str | Path,
    smoke_gate_receipt_path: str | Path,
    smoke_shard_directory: str | Path,
) -> dict[str, Any]:
    """Replay every controller-local prerequisite and freeze portable facts."""

    checked_plan = contract.validate_dataset_plan(plan)
    fresh_gate, fresh_file_sha = executor._gate_value(  # type: ignore[attr-defined]
        fresh_quality_gate_path
    )
    smoke_directory = Path(smoke_shard_directory).resolve()
    executor._existing_smoke_authorization(  # type: ignore[attr-defined]
        plan=checked_plan,
        smoke_shard_directory=smoke_directory,
        fresh_quality_gate_path=fresh_quality_gate_path,
    )
    smoke_path = Path(smoke_gate_receipt_path).resolve()
    smoke_gate = contract.validate_smoke_gate_receipt(
        _read_canonical(smoke_path, "M3.1 smoke gate"),
        plan=checked_plan,
        smoke_shard_directory=smoke_directory,
    )
    if (
        fresh_gate["status"] != "pass"
        or fresh_gate["all_gates_passed"] is not True
        or smoke_gate["status"] != "pass"
        or smoke_gate["all_gates_passed"] is not True
        or smoke_gate["full_9000_paired_fanout_authorized"] is not True
    ):
        raise PermissionError("M3.1 controller gates do not authorize fanout")
    inventory = _inventory(smoke_directory)
    core = {
        "schema": PORTABLE_AUTHORIZATION_SCHEMA,
        "status": "controller_source_replayed_portable_fanout_authorized",
        "plan_sha256": canonical_sha256(checked_plan),
        "fresh_quality_gate_schema": fresh_gate["schema"],
        "fresh_quality_gate_file_sha256": fresh_file_sha,
        "fresh_quality_gate_canonical_sha256": canonical_sha256(fresh_gate),
        "fresh_quality_merge_sha256": fresh_gate["merge_sha256"],
        "fresh_quality_decision": fresh_gate["decision"],
        "dataset_smoke_gate_schema": smoke_gate["schema"],
        "dataset_smoke_gate_file_sha256": _file_sha256(smoke_path),
        "dataset_smoke_gate_canonical_sha256": canonical_sha256(smoke_gate),
        "dataset_smoke_gate_decision": smoke_gate["decision"],
        "smoke_shard_id": contract.SMOKE_SHARD_ID,
        "smoke_shard_done_sha256": smoke_gate["smoke_shard_done_sha256"],
        "smoke_shard_inventory": inventory,
        "smoke_shard_inventory_sha256": canonical_sha256(inventory),
        "current_profile_registry_sha256": executor._profile_sha256(),  # type: ignore[attr-defined]
        "controller_source_replayed": True,
        "worker_replay_boundary": (
            "exact_gate_file_hashes_plus_source_replayed_smoke_shard"
        ),
        "full_9000_paired_fanout_authorized": True,
        "teacher_values_are_realized_match_ev": False,
        "current_profile_changed": False,
    }
    return {**core, "authorization_sha256": canonical_sha256(core)}


def write_portable_fanout_authorization(
    path: str | Path, **kwargs: Any
) -> dict[str, Any]:
    value = build_portable_fanout_authorization(**kwargs)
    target = Path(path)
    _write_once(target, value)
    stored = _read_canonical(target, "portable fanout authorization")
    if stored != value:
        raise ValueError("stored portable fanout authorization changed")
    return stored


def validate_portable_fanout_authorization(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    fresh_quality_gate_path: str | Path,
    smoke_gate_receipt_path: str | Path,
    smoke_shard_directory: str | Path,
) -> dict[str, Any]:
    """Worker-side validation without pretending controller paths are local."""

    checked_plan = contract.validate_dataset_plan(plan)
    receipt = deepcopy(dict(value))
    if set(receipt) != _KEYS:
        raise ValueError("portable fanout authorization fields changed")
    core = dict(receipt)
    declared = core.pop("authorization_sha256")
    fresh_path = Path(fresh_quality_gate_path)
    fresh_gate = executor._read_quality_gate(  # type: ignore[attr-defined]
        fresh_path
    )
    smoke_path = Path(smoke_gate_receipt_path)
    smoke_directory = Path(smoke_shard_directory)
    smoke_gate = contract.validate_smoke_gate_receipt(
        _read_canonical(smoke_path, "staged M3.1 smoke gate"),
        plan=checked_plan,
        smoke_shard_directory=smoke_directory,
    )
    inventory = _inventory(smoke_directory)
    if (
        receipt["schema"] != PORTABLE_AUTHORIZATION_SCHEMA
        or receipt["status"]
        != "controller_source_replayed_portable_fanout_authorized"
        or declared != canonical_sha256(core)
        or receipt["plan_sha256"] != canonical_sha256(checked_plan)
        or receipt["fresh_quality_gate_schema"] != quality_gate.GATE_SCHEMA
        or receipt["fresh_quality_gate_file_sha256"]
        != _file_sha256(fresh_path)
        or receipt["fresh_quality_gate_canonical_sha256"]
        != canonical_sha256(fresh_gate)
        or receipt["fresh_quality_merge_sha256"]
        != fresh_gate.get("merge_sha256")
        or receipt["fresh_quality_decision"] != fresh_gate.get("decision")
        or fresh_gate.get("status") != "pass"
        or fresh_gate.get("all_gates_passed") is not True
        or fresh_gate.get("data_pilot_25_paired_authorized") is not True
        or fresh_gate.get("full_9000_paired_fanout_authorized") is not False
        or receipt["dataset_smoke_gate_schema"] != contract.SMOKE_GATE_SCHEMA
        or receipt["dataset_smoke_gate_file_sha256"] != _file_sha256(smoke_path)
        or receipt["dataset_smoke_gate_canonical_sha256"]
        != canonical_sha256(smoke_gate)
        or receipt["dataset_smoke_gate_decision"] != smoke_gate["decision"]
        or receipt["smoke_shard_id"] != contract.SMOKE_SHARD_ID
        or receipt["smoke_shard_done_sha256"]
        != smoke_gate["smoke_shard_done_sha256"]
        or receipt["smoke_shard_inventory"] != inventory
        or receipt["smoke_shard_inventory_sha256"]
        != canonical_sha256(inventory)
        or receipt["current_profile_registry_sha256"]
        != executor._profile_sha256()  # type: ignore[attr-defined]
        or receipt["controller_source_replayed"] is not True
        or receipt["worker_replay_boundary"]
        != "exact_gate_file_hashes_plus_source_replayed_smoke_shard"
        or receipt["full_9000_paired_fanout_authorized"] is not True
        or receipt["teacher_values_are_realized_match_ev"] is not False
        or receipt["current_profile_changed"] is not False
    ):
        raise PermissionError("portable fanout authorization binding changed")
    return receipt


def _shard_authorization(
    receipt: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    shard_id: str,
) -> dict[str, Any]:
    if shard_id == contract.SMOKE_SHARD_ID:
        raise PermissionError("portable fanout cannot rerun the smoke shard")
    descriptor = next(
        (row for row in plan["shards"] if row["shard_id"] == shard_id), None
    )
    if descriptor is None or descriptor["requires_smoke_gate_receipt"] is not True:
        raise KeyError(f"unknown post-smoke M3.1 shard: {shard_id}")
    return {
        "schema": executor.FULL_AUTHORIZATION_SCHEMA,
        "status": "qualified_smoke_gate_bound_to_full_dataset_shard",
        "plan_sha256": receipt["plan_sha256"],
        "shard_id": shard_id,
        "fresh_quality_gate_schema": receipt["fresh_quality_gate_schema"],
        "fresh_quality_gate_sha256": receipt[
            "fresh_quality_gate_file_sha256"
        ],
        "fresh_quality_merge_sha256": receipt[
            "fresh_quality_merge_sha256"
        ],
        "fresh_quality_decision": receipt["fresh_quality_decision"],
        "dataset_smoke_gate_schema": receipt["dataset_smoke_gate_schema"],
        "dataset_smoke_gate_sha256": receipt[
            "dataset_smoke_gate_file_sha256"
        ],
        "dataset_smoke_gate_decision": receipt[
            "dataset_smoke_gate_decision"
        ],
        "smoke_shard_id": receipt["smoke_shard_id"],
        "smoke_shard_done_sha256": receipt["smoke_shard_done_sha256"],
        "source_replayed": True,
        "data_pilot_25_paired_authorized": True,
        "full_9000_paired_fanout_authorized": True,
        "current_profile_registry_sha256": receipt[
            "current_profile_registry_sha256"
        ],
        "current_profile_changed": False,
    }


def run_portable_dataset_shard(
    *,
    plan: Mapping[str, Any],
    shard_id: str,
    shard_directory: str | Path,
    portable_authorization_path: str | Path,
    fresh_quality_gate_path: str | Path,
    smoke_gate_receipt_path: str | Path,
    smoke_shard_directory: str | Path,
    library_path: str | Path | None = None,
    root_generator: Any = None,
    search_adapter: Any = None,
    baseline_adapter: Any = None,
    bundle: Any = None,
    max_new_pairs: int | None = None,
) -> dict[str, Any]:
    checked_plan = contract.validate_dataset_plan(plan)
    portable = validate_portable_fanout_authorization(
        _read_canonical(
            Path(portable_authorization_path), "portable fanout authorization"
        ),
        plan=checked_plan,
        fresh_quality_gate_path=fresh_quality_gate_path,
        smoke_gate_receipt_path=smoke_gate_receipt_path,
        smoke_shard_directory=smoke_shard_directory,
    )
    authorization = _shard_authorization(
        portable, plan=checked_plan, shard_id=shard_id
    )
    directory = Path(shard_directory)
    directory.mkdir(parents=True, exist_ok=True)
    authorization_path = directory / executor.AUTHORIZATION_NAME
    if authorization_path.exists():
        if _read_canonical(
            authorization_path, "portable shard authorization"
        ) != authorization:
            raise PermissionError("portable shard authorization changed")
    else:
        _write_once(authorization_path, authorization)
    result = executor._run_authorized_shard(  # type: ignore[attr-defined]
        plan=checked_plan,
        shard_id=shard_id,
        shard_directory=directory,
        authorization=authorization,
        library_path=library_path,
        root_generator=root_generator,
        search_adapter=search_adapter,
        baseline_adapter=baseline_adapter,
        bundle=bundle,
        max_new_pairs=max_new_pairs,
    )
    return {
        **result,
        "portable_authorization_sha256": portable["authorization_sha256"],
        "cloud_worker_portable_boundary": True,
        "current_profile_changed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="M3.1 portable dataset worker")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--shard-id", required=True)
    run.add_argument("--shard-directory", required=True)
    run.add_argument("--portable-authorization", required=True)
    run.add_argument("--fresh-quality-gate", required=True)
    run.add_argument("--smoke-gate", required=True)
    run.add_argument("--smoke-shard-directory", required=True)
    run.add_argument("--library", required=True)
    run.add_argument("--max-new-pairs", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = contract.validate_dataset_plan(
        _read_canonical(Path(args.plan), "M3.1 dataset plan")
    )
    result = run_portable_dataset_shard(
        plan=plan,
        shard_id=args.shard_id,
        shard_directory=args.shard_directory,
        portable_authorization_path=args.portable_authorization,
        fresh_quality_gate_path=args.fresh_quality_gate,
        smoke_gate_receipt_path=args.smoke_gate,
        smoke_shard_directory=args.smoke_shard_directory,
        library_path=args.library,
        max_new_pairs=args.max_new_pairs,
    )
    print(canonical_bytes(result).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PORTABLE_AUTHORIZATION_SCHEMA",
    "build_portable_fanout_authorization",
    "canonical_bytes",
    "canonical_sha256",
    "main",
    "run_portable_dataset_shard",
    "validate_portable_fanout_authorization",
    "write_portable_fanout_authorization",
]
