"""Deterministic execution grid for the M3.1 locked population/ABR gate.

The scientific contract and metric implementation live in
``hu_m31_t3_step6d_locked_promotion_v1``.  This module fills the deliberately
separate operational gap: it freezes all and only the work items needed for
the five-opponent population and three-family ABR evaluation, dispatches one
validated work item to ``LockedPromotionRunner``, and closes out only an exact
shard directory.

It does not construct policies, train ABRs, talk to a cloud provider, register
a named profile, resolve ``current``, or activate a runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import hu_m31_t3_step6d_locked_promotion_v1 as promotion


EXECUTION_PLAN_SCHEMA = "hu_m31_t3_locked_promotion_execution_plan_v1"
WORK_ITEM_SCHEMA = "hu_m31_t3_locked_promotion_work_item_v1"
PAIRED_SEEDS_PER_SHARD = 25
EXPECTED_POPULATION_WORK_ITEMS = 200
EXPECTED_ABR_WORK_ITEMS = 60
EXPECTED_WORK_ITEMS = 260
EXPECTED_PAIRED_SEEDS = 6_500
EXPECTED_ROWS = 13_000

_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,127}$")
_WORK_ITEM_KEYS = frozenset(
    {
        "schema",
        "ordinal",
        "work_id",
        "schedule",
        "entity_id",
        "seed_index_start",
        "seed_index_stop_exclusive",
        "paired_seed_count",
        "row_count",
        "seats",
        "promotion_plan_sha256",
        "coverage_sha256",
        "output_filename",
    }
)
_EXECUTION_PLAN_KEYS = frozenset(
    {
        "schema",
        "status",
        "execution_id",
        "promotion_plan_sha256",
        "paired_seeds_per_shard",
        "work_items",
        "work_item_count",
        "population_work_item_count",
        "abr_work_item_count",
        "coverage",
        "work_item_aggregate_sha256",
        "cloud_execution_started",
        "promotion_decision_applied",
        "named_profile_added",
        "current_profile_changed",
        "runtime_activated",
        "full_replacement_enabled",
    }
)


def _exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields changed")


def _canonical_bytes(value: Any) -> bytes:
    return promotion.canonical_bytes(value)


def _read_canonical(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value):
        raise ValueError(f"{label} is not a canonical object")
    return value


def _write_once(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path)
    raw = _canonical_bytes(value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if (
            destination.is_symlink()
            or not destination.is_file()
            or destination.read_bytes() != raw
        ):
            raise FileExistsError(
                f"immutable locked-promotion execution artifact changed: "
                f"{destination}"
            ) from None
    return destination.resolve()


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        raise ValueError("locked-promotion entity id cannot form a work id")
    return slug


def _entities(schedule: str) -> tuple[str, ...]:
    if schedule == promotion.LOCKED_POPULATION:
        return tuple(
            str(value["opponent_id"])
            for value in promotion.OPPONENT_DESCRIPTORS
        )
    if schedule == promotion.LOCKED_ABR:
        return tuple(
            str(value["response_id"])
            for value in promotion.ABR_DESCRIPTORS
        )
    raise ValueError("locked-promotion execution schedule changed")


def _seed_count(schedule: str) -> int:
    if schedule == promotion.LOCKED_POPULATION:
        return promotion.POPULATION_SEED_COUNT
    if schedule == promotion.LOCKED_ABR:
        return promotion.ABR_SEED_COUNT
    raise ValueError("locked-promotion execution schedule changed")


def _coverage_keys(
    *,
    schedule: str,
    entity_id: str,
    start: int,
    stop: int,
) -> list[list[Any]]:
    return [
        [schedule, entity_id, index, seat]
        for index in range(start, stop)
        for seat in promotion.SEATS
    ]


def _work_item(
    *,
    ordinal: int,
    schedule: str,
    entity_id: str,
    start: int,
    stop: int,
    promotion_plan_sha256: str,
) -> dict[str, Any]:
    schedule_slug = (
        "population"
        if schedule == promotion.LOCKED_POPULATION
        else "abr"
    )
    work_id = (
        f"{schedule_slug}-{_slug(entity_id)}-"
        f"{start:04d}-{stop - 1:04d}"
    )
    coverage = _coverage_keys(
        schedule=schedule,
        entity_id=entity_id,
        start=start,
        stop=stop,
    )
    return {
        "schema": WORK_ITEM_SCHEMA,
        "ordinal": ordinal,
        "work_id": work_id,
        "schedule": schedule,
        "entity_id": entity_id,
        "seed_index_start": start,
        "seed_index_stop_exclusive": stop,
        "paired_seed_count": stop - start,
        "row_count": len(coverage),
        "seats": list(promotion.SEATS),
        "promotion_plan_sha256": promotion_plan_sha256,
        "coverage_sha256": promotion.canonical_sha256(coverage),
        "output_filename": f"{work_id}.json",
    }


def build_execution_plan(
    *, promotion_plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Freeze the exact 260-shard, 13,000-row evaluation work grid."""

    locked = promotion.validate_locked_promotion_plan(promotion_plan)
    promotion_plan_sha256 = promotion.canonical_sha256(locked)
    work_items: list[dict[str, Any]] = []
    for schedule in (
        promotion.LOCKED_POPULATION,
        promotion.LOCKED_ABR,
    ):
        count = _seed_count(schedule)
        if count % PAIRED_SEEDS_PER_SHARD:
            raise AssertionError(
                "frozen locked-promotion count is not shard divisible"
            )
        for entity_id in _entities(schedule):
            for start in range(0, count, PAIRED_SEEDS_PER_SHARD):
                work_items.append(
                    _work_item(
                        ordinal=len(work_items),
                        schedule=schedule,
                        entity_id=entity_id,
                        start=start,
                        stop=start + PAIRED_SEEDS_PER_SHARD,
                        promotion_plan_sha256=promotion_plan_sha256,
                    )
                )
    population_items = sum(
        item["schedule"] == promotion.LOCKED_POPULATION
        for item in work_items
    )
    abr_items = sum(
        item["schedule"] == promotion.LOCKED_ABR
        for item in work_items
    )
    all_coverage = [
        key
        for item in work_items
        for key in _coverage_keys(
            schedule=str(item["schedule"]),
            entity_id=str(item["entity_id"]),
            start=int(item["seed_index_start"]),
            stop=int(item["seed_index_stop_exclusive"]),
        )
    ]
    execution_id = (
        f"m31-t3-locked-promotion-{promotion_plan_sha256[:16]}-s"
        f"{PAIRED_SEEDS_PER_SHARD}"
    )
    value = {
        "schema": EXECUTION_PLAN_SCHEMA,
        "status": "frozen_grid_only_no_execution_or_promotion",
        "execution_id": execution_id,
        "promotion_plan_sha256": promotion_plan_sha256,
        "paired_seeds_per_shard": PAIRED_SEEDS_PER_SHARD,
        "work_items": work_items,
        "work_item_count": len(work_items),
        "population_work_item_count": population_items,
        "abr_work_item_count": abr_items,
        "coverage": {
            "population_opponents": len(
                promotion.OPPONENT_DESCRIPTORS
            ),
            "population_paired_seeds_per_opponent": (
                promotion.POPULATION_SEED_COUNT
            ),
            "abr_families": len(promotion.ABR_DESCRIPTORS),
            "abr_paired_seeds_per_response": promotion.ABR_SEED_COUNT,
            "paired_seed_count": len(all_coverage) // len(promotion.SEATS),
            "row_count": len(all_coverage),
            "first_rows": sum(
                key[3] == "first" for key in all_coverage
            ),
            "second_rows": sum(
                key[3] == "second" for key in all_coverage
            ),
            "coverage_sha256": promotion.canonical_sha256(all_coverage),
        },
        "work_item_aggregate_sha256": promotion.canonical_sha256(
            work_items
        ),
        "cloud_execution_started": False,
        "promotion_decision_applied": False,
        "named_profile_added": False,
        "current_profile_changed": False,
        "runtime_activated": False,
        "full_replacement_enabled": False,
    }
    if (
        len(work_items) != EXPECTED_WORK_ITEMS
        or population_items != EXPECTED_POPULATION_WORK_ITEMS
        or abr_items != EXPECTED_ABR_WORK_ITEMS
        or len(all_coverage) // len(promotion.SEATS)
        != EXPECTED_PAIRED_SEEDS
        or len(all_coverage) != EXPECTED_ROWS
        or len({tuple(key) for key in all_coverage}) != EXPECTED_ROWS
    ):
        raise AssertionError("locked-promotion execution grid changed")
    return value


def validate_execution_plan(
    value: Mapping[str, Any],
    *,
    promotion_plan: Mapping[str, Any],
) -> dict[str, Any]:
    plan = deepcopy(dict(value))
    _exact_keys(plan, _EXECUTION_PLAN_KEYS, "execution plan")
    items = plan.get("work_items")
    if not isinstance(items, list):
        raise ValueError("execution plan work items are missing")
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("execution plan work item is missing")
        _exact_keys(item, _WORK_ITEM_KEYS, "execution work item")
        if (
            _SAFE_ID.fullmatch(str(item.get("work_id"))) is None
            or Path(str(item.get("output_filename"))).name
            != item.get("output_filename")
        ):
            raise ValueError("execution work item path or id is unsafe")
    expected = build_execution_plan(promotion_plan=promotion_plan)
    if plan != expected:
        raise ValueError("execution plan differs from frozen source replay")
    return plan


def write_execution_plan(
    *,
    promotion_plan: Mapping[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    plan = build_execution_plan(promotion_plan=promotion_plan)
    _write_once(output_path, plan)
    return plan


def get_work_item(
    *,
    execution_plan: Mapping[str, Any],
    promotion_plan: Mapping[str, Any],
    work_id: str,
) -> dict[str, Any]:
    plan = validate_execution_plan(
        execution_plan, promotion_plan=promotion_plan
    )
    matches = [
        item for item in plan["work_items"] if item["work_id"] == work_id
    ]
    if len(matches) != 1:
        raise KeyError(f"unknown locked-promotion work item: {work_id}")
    return deepcopy(matches[0])


def validate_work_item_shard(
    *,
    work_item: Mapping[str, Any],
    shard: Mapping[str, Any],
    promotion_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove that a source-replayed shard covers exactly one work item."""

    locked = promotion.validate_locked_promotion_plan(promotion_plan)
    validated = promotion.validate_evaluation_shard(
        shard, plan=locked
    )
    item = dict(work_item)
    _exact_keys(item, _WORK_ITEM_KEYS, "execution work item")
    start = int(item["seed_index_start"])
    stop = int(item["seed_index_stop_exclusive"])
    expected_keys = [
        (
            item["schedule"],
            item["entity_id"],
            index,
            seat,
        )
        for index in range(start, stop)
        for seat in promotion.SEATS
    ]
    observed_keys = [
        (
            row["schedule"],
            row["entity_id"],
            row["seed_index"],
            row["seat"],
        )
        for row in validated["rows"]
    ]
    coverage = [
        [schedule, entity_id, index, seat]
        for schedule, entity_id, index, seat in expected_keys
    ]
    if (
        validated["shard_id"] != item["work_id"]
        or validated["row_count"] != item["row_count"]
        or observed_keys != expected_keys
        or item["paired_seed_count"] != stop - start
        or item["seats"] != list(promotion.SEATS)
        or item["promotion_plan_sha256"]
        != promotion.canonical_sha256(locked)
        or item["coverage_sha256"]
        != promotion.canonical_sha256(coverage)
    ):
        raise ValueError(
            "locked-promotion shard differs from its frozen work item"
        )
    return validated


def run_work_item(
    *,
    runner: Any,
    promotion_plan: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    work_id: str,
    shard_directory: str | Path,
) -> dict[str, Any]:
    """Dispatch exactly one frozen item to a pre-bound policy runner."""

    locked = promotion.validate_locked_promotion_plan(promotion_plan)
    item = get_work_item(
        execution_plan=execution_plan,
        promotion_plan=locked,
        work_id=work_id,
    )
    runner_plan = getattr(runner, "plan", None)
    run_shard = getattr(runner, "run_shard", None)
    if (
        not isinstance(runner_plan, Mapping)
        or promotion.validate_locked_promotion_plan(runner_plan) != locked
        or not callable(run_shard)
    ):
        raise ValueError(
            "runner is not bound to this locked-promotion plan"
        )
    directory = Path(shard_directory)
    if directory.is_symlink():
        raise ValueError("locked-promotion shard directory is unsafe")
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("locked-promotion shard directory is unsafe")
    output_path = directory / str(item["output_filename"])
    shard = run_shard(
        schedule=item["schedule"],
        entity_id=item["entity_id"],
        seed_indices=range(
            int(item["seed_index_start"]),
            int(item["seed_index_stop_exclusive"]),
        ),
        shard_id=item["work_id"],
        output_path=output_path,
    )
    return validate_work_item_shard(
        work_item=item,
        shard=shard,
        promotion_plan=locked,
    )


def collect_exact_shard_paths(
    *,
    promotion_plan: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    shard_directory: str | Path,
) -> list[Path]:
    """Return ordered shard paths only after exact-grid source replay."""

    locked = promotion.validate_locked_promotion_plan(promotion_plan)
    execution = validate_execution_plan(
        execution_plan, promotion_plan=locked
    )
    directory = Path(shard_directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("locked-promotion shard directory is unsafe")
    expected_names = {
        str(item["output_filename"]) for item in execution["work_items"]
    }
    actual_paths = list(directory.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in actual_paths):
        raise ValueError("locked-promotion shard directory has unsafe entries")
    actual_names = {path.name for path in actual_paths}
    if actual_names != expected_names:
        raise ValueError(
            "locked-promotion shard directory has missing or extra files"
        )
    result: list[Path] = []
    for item in execution["work_items"]:
        path = directory / str(item["output_filename"])
        shard = _read_canonical(path, "locked-promotion work shard")
        validate_work_item_shard(
            work_item=item,
            shard=shard,
            promotion_plan=locked,
        )
        result.append(path.resolve())
    return result


def build_closeout(
    *,
    promotion_plan: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    shard_directory: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay the exact source grid, then compute the existing merge and gate."""

    locked = promotion.validate_locked_promotion_plan(promotion_plan)
    paths = collect_exact_shard_paths(
        promotion_plan=locked,
        execution_plan=execution_plan,
        shard_directory=shard_directory,
    )
    merge = promotion.build_locked_promotion_merge(
        plan=locked, shard_paths=paths
    )
    gate = promotion.build_locked_promotion_gate(
        plan=locked, merge=merge, replay_sources=True
    )
    return merge, gate


def write_closeout(
    *,
    promotion_plan: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    shard_directory: str | Path,
    merge_output_path: str | Path,
    gate_output_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    merge, gate = build_closeout(
        promotion_plan=promotion_plan,
        execution_plan=execution_plan,
        shard_directory=shard_directory,
    )
    _write_once(merge_output_path, merge)
    _write_once(gate_output_path, gate)
    return merge, gate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare or source-replay the deterministic M3.1 locked "
            "population/ABR execution grid."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--promotion-plan", required=True)
    prepare.add_argument("--output", required=True)
    show = subparsers.add_parser("show-work-item")
    show.add_argument("--promotion-plan", required=True)
    show.add_argument("--execution-plan", required=True)
    show.add_argument("--work-id", required=True)
    closeout = subparsers.add_parser("closeout")
    closeout.add_argument("--promotion-plan", required=True)
    closeout.add_argument("--execution-plan", required=True)
    closeout.add_argument("--shard-directory", required=True)
    closeout.add_argument("--merge-output", required=True)
    closeout.add_argument("--gate-output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        promotion_plan = _read_canonical(
            args.promotion_plan, "locked-promotion plan"
        )
        if args.command == "prepare":
            value = write_execution_plan(
                promotion_plan=promotion_plan,
                output_path=args.output,
            )
        else:
            execution_plan = _read_canonical(
                args.execution_plan, "locked-promotion execution plan"
            )
            if args.command == "show-work-item":
                value = get_work_item(
                    execution_plan=execution_plan,
                    promotion_plan=promotion_plan,
                    work_id=args.work_id,
                )
            else:
                merge, gate = write_closeout(
                    promotion_plan=promotion_plan,
                    execution_plan=execution_plan,
                    shard_directory=args.shard_directory,
                    merge_output_path=args.merge_output,
                    gate_output_path=args.gate_output,
                )
                value = {
                    "merge_sha256": promotion.canonical_sha256(merge),
                    "gate_sha256": promotion.canonical_sha256(gate),
                    "status": gate["status"],
                    "scientific_promotion_passed": gate[
                        "scientific_promotion_passed"
                    ],
                    "named_profile_added": False,
                    "current_profile_changed": False,
                    "runtime_activated": False,
                }
    except Exception as exc:  # pragma: no cover - subprocess boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXECUTION_PLAN_SCHEMA",
    "EXPECTED_ABR_WORK_ITEMS",
    "EXPECTED_PAIRED_SEEDS",
    "EXPECTED_POPULATION_WORK_ITEMS",
    "EXPECTED_ROWS",
    "EXPECTED_WORK_ITEMS",
    "PAIRED_SEEDS_PER_SHARD",
    "WORK_ITEM_SCHEMA",
    "build_closeout",
    "build_execution_plan",
    "collect_exact_shard_paths",
    "get_work_item",
    "main",
    "run_work_item",
    "validate_execution_plan",
    "validate_work_item_shard",
    "write_closeout",
    "write_execution_plan",
]
