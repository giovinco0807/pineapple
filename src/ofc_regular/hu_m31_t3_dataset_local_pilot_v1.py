"""Create-only local driver for the first M3.1 dataset smoke shard.

This is the only combined entry point needed immediately after fresh quality:

1. replay the qualified fresh-quality gate;
2. create or replay the frozen dataset plan;
3. run/resume exactly ``train-0000`` (25 paired hands);
4. replay the completed shard into the smoke merge; and
5. create or replay the dataset smoke-gate receipt.

The driver never starts a post-smoke shard, writes a portable fanout
authorization, exports Parquet, trains a model, or changes an AI profile.
Parquet therefore remains optional and is not a prerequisite for the smoke
pass receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import hu_m31_t3_dataset_contract_v1 as contract
from . import hu_m31_t3_dataset_executor_v1 as executor


PILOT_RESULT_SCHEMA = "hu_m31_t3_dataset_local_pilot_result_v1"
PLAN_NAME = "dataset_plan.json"
SHARDS_DIRECTORY_NAME = "shards"
SMOKE_MERGE_NAME = "smoke_merge.json"
SMOKE_GATE_NAME = "dataset_smoke_gate.json"

_ALLOWED_ROOT_ENTRIES = frozenset(
    {
        PLAN_NAME,
        SHARDS_DIRECTORY_NAME,
        SMOKE_MERGE_NAME,
        SMOKE_GATE_NAME,
    }
)


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


def _is_link_or_junction(path: Path) -> bool:
    return path.is_symlink() or (
        hasattr(path, "is_junction") and path.is_junction()
    )


def _plain_file(path: str | Path, label: str) -> Path:
    target = Path(path)
    if (
        not target.is_absolute()
        or _is_link_or_junction(target)
        or not target.is_file()
    ):
        raise ValueError(f"{label} must be an absolute regular file")
    return target.resolve()


def _read_canonical(path: str | Path, label: str) -> dict[str, Any]:
    target = _plain_file(path, label)
    raw = target.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _safe_output_root(path: str | Path) -> Path:
    root = Path(path)
    if not root.is_absolute():
        raise ValueError("local dataset pilot root must be absolute")
    root = root.resolve()
    parent = root.parent
    if (
        _is_link_or_junction(parent)
        or not parent.is_dir()
        or any(_is_link_or_junction(item) for item in parent.parents)
    ):
        raise ValueError("local dataset pilot parent is unsafe")
    if root.exists():
        if _is_link_or_junction(root) or not root.is_dir():
            raise ValueError("local dataset pilot root is unsafe")
    else:
        root.mkdir()
    return root


def _native_library_suffix() -> str:
    if sys.platform == "win32":
        return ".dll"
    if sys.platform == "darwin":
        return ".dylib"
    return ".so"


def _validate_root_inventory(root: Path) -> None:
    if _is_link_or_junction(root) or not root.is_dir():
        raise ValueError("local dataset pilot root is unsafe")
    entries = list(root.iterdir())
    if any(
        _is_link_or_junction(entry)
        or entry.name not in _ALLOWED_ROOT_ENTRIES
        for entry in entries
    ):
        raise ValueError("local dataset pilot root has an unexpected entry")
    by_name = {entry.name: entry for entry in entries}
    for name in (PLAN_NAME, SMOKE_MERGE_NAME, SMOKE_GATE_NAME):
        entry = by_name.get(name)
        if entry is not None and not entry.is_file():
            raise ValueError(f"local dataset pilot {name} is not a file")
    shards = by_name.get(SHARDS_DIRECTORY_NAME)
    if shards is None:
        return
    if not shards.is_dir():
        raise ValueError("local dataset pilot shards entry is not a directory")
    shard_entries = list(shards.iterdir())
    if any(
        _is_link_or_junction(entry)
        or entry.name != contract.SMOKE_SHARD_ID
        or not entry.is_dir()
        for entry in shard_entries
    ):
        raise ValueError("local dataset pilot contains a non-smoke shard")


def _load_or_write_plan(path: Path) -> dict[str, Any]:
    if path.exists():
        return contract.validate_dataset_plan(
            _read_canonical(path, "M3.1 dataset plan")
        )
    return contract.write_dataset_plan(path)


def _load_or_write_smoke_merge(
    *,
    path: Path,
    plan: Mapping[str, Any],
    shard_directory: Path,
    fresh_quality_gate_path: Path,
) -> dict[str, Any]:
    if path.exists():
        return executor.validate_smoke_merge_value(
            _read_canonical(path, "M3.1 smoke merge"),
            plan=plan,
            shard_directory=shard_directory,
            fresh_quality_gate_path=fresh_quality_gate_path,
        )
    return executor.write_smoke_merge(
        plan=plan,
        shard_directory=shard_directory,
        fresh_quality_gate_path=fresh_quality_gate_path,
        output_path=path,
    )


def _load_or_write_smoke_gate(
    *,
    path: Path,
    plan: Mapping[str, Any],
    shard_directory: Path,
) -> dict[str, Any]:
    if path.exists():
        return contract.validate_smoke_gate_receipt(
            _read_canonical(path, "M3.1 dataset smoke gate"),
            plan=plan,
            smoke_shard_directory=shard_directory,
        )
    return contract.write_smoke_gate_receipt(
        plan=plan,
        smoke_shard_directory=shard_directory,
        output_path=path,
    )


def run_local_pilot(
    *,
    output_root: str | Path,
    fresh_quality_gate_path: str | Path,
    library_path: str | Path | None,
    root_generator: Any = None,
    search_adapter: Any = None,
    baseline_adapter: Any = None,
    bundle: Any = None,
) -> dict[str, Any]:
    """Run or source-replay the sole 25-paired local smoke pilot."""

    fresh_gate_path = _plain_file(
        fresh_quality_gate_path, "qualified fresh-quality gate"
    )
    fresh_gate, fresh_gate_file_sha256 = executor._gate_value(  # type: ignore[attr-defined]
        fresh_gate_path
    )
    executor._profile_sha256()  # type: ignore[attr-defined]
    native_path: Path | None = None
    if search_adapter is None:
        if library_path is None:
            raise ValueError("production local pilot requires --library")
        native_path = _plain_file(library_path, "accepted Candidate02 library")
        if (
            _file_sha256(native_path)
            != contract.ACCEPTED_CANDIDATE_LIBRARY_SHA256
        ):
            raise ValueError("local pilot Candidate02 library hash changed")
        if native_path.suffix.casefold() != _native_library_suffix():
            raise ValueError(
                "accepted Candidate02 library cannot load on this platform"
            )
    elif library_path is not None:
        native_path = _plain_file(library_path, "synthetic pilot library")

    proposed_root = Path(output_root)
    if not proposed_root.is_absolute():
        raise ValueError("local dataset pilot root must be absolute")
    proposed_root = proposed_root.resolve()
    if proposed_root in fresh_gate_path.parents or (
        native_path is not None and proposed_root in native_path.parents
    ):
        raise ValueError("local pilot output overlaps a pinned input")
    root = _safe_output_root(proposed_root)
    _validate_root_inventory(root)
    if (root / SMOKE_GATE_NAME).exists() and not (
        root / SMOKE_MERGE_NAME
    ).exists():
        raise ValueError("smoke gate exists before its source-replayed merge")

    plan_path = root / PLAN_NAME
    plan = _load_or_write_plan(plan_path)
    shards_root = root / SHARDS_DIRECTORY_NAME
    if shards_root.exists():
        if _is_link_or_junction(shards_root) or not shards_root.is_dir():
            raise ValueError("local dataset pilot shards root is unsafe")
    else:
        shards_root.mkdir()
    shard_directory = shards_root / contract.SMOKE_SHARD_ID

    run = executor.run_smoke_shard(
        plan=plan,
        shard_directory=shard_directory,
        fresh_quality_gate_path=fresh_gate_path,
        library_path=native_path,
        root_generator=root_generator,
        search_adapter=search_adapter,
        baseline_adapter=baseline_adapter,
        bundle=bundle,
        max_new_pairs=None,
    )
    if run.get("status") not in {
        "complete_done_published_last",
        "already_complete",
    }:
        raise RuntimeError(
            "local pilot stopped before all 25 paired hands completed"
        )
    done = contract.validate_completed_shard(
        plan=plan,
        shard_id=contract.SMOKE_SHARD_ID,
        shard_directory=shard_directory,
    )
    if (
        done["pair_count"] != contract.SHARD_PAIR_COUNT
        or done["root_count"] != contract.SHARD_PAIR_COUNT * 2
        or done["seat_counts"]
        != {
            "first": contract.SHARD_PAIR_COUNT,
            "second": contract.SHARD_PAIR_COUNT,
        }
    ):
        raise ValueError("local pilot completed another shard geometry")

    merge_path = root / SMOKE_MERGE_NAME
    merge = _load_or_write_smoke_merge(
        path=merge_path,
        plan=plan,
        shard_directory=shard_directory,
        fresh_quality_gate_path=fresh_gate_path,
    )
    smoke_gate_path = root / SMOKE_GATE_NAME
    smoke_gate = _load_or_write_smoke_gate(
        path=smoke_gate_path,
        plan=plan,
        shard_directory=shard_directory,
    )
    if (
        smoke_gate["status"] != "pass"
        or smoke_gate["all_gates_passed"] is not True
        or smoke_gate["full_9000_paired_fanout_authorized"] is not True
        or merge["data_pilot_25_paired_complete"] is not True
        or merge["full_9000_paired_fanout_authorized"] is not False
    ):
        raise PermissionError("local dataset smoke gate did not pass safely")
    _validate_root_inventory(root)
    executor._profile_sha256()  # type: ignore[attr-defined]

    done_path = shard_directory / "SHARD_DONE.json"
    return {
        "schema": PILOT_RESULT_SCHEMA,
        "status": "passed_source_replayed_25_paired_local_pilot",
        "output_root": str(root),
        "plan_path": str(plan_path),
        "plan_file_sha256": _file_sha256(plan_path),
        "plan_sha256": canonical_sha256(plan),
        "fresh_quality_gate_path": str(fresh_gate_path),
        "fresh_quality_gate_file_sha256": fresh_gate_file_sha256,
        "fresh_quality_gate_canonical_sha256": canonical_sha256(fresh_gate),
        "candidate_library_path": (
            str(native_path) if native_path is not None else None
        ),
        "candidate_library_sha256": (
            _file_sha256(native_path) if native_path is not None else None
        ),
        "smoke_shard_id": contract.SMOKE_SHARD_ID,
        "smoke_shard_directory": str(shard_directory.resolve()),
        "shard_done_path": str(done_path.resolve()),
        "shard_done_file_sha256": _file_sha256(done_path),
        "smoke_merge_path": str(merge_path.resolve()),
        "smoke_merge_file_sha256": _file_sha256(merge_path),
        "smoke_merge_sha256": canonical_sha256(merge),
        "smoke_gate_path": str(smoke_gate_path.resolve()),
        "smoke_gate_file_sha256": _file_sha256(smoke_gate_path),
        "smoke_gate_sha256": canonical_sha256(smoke_gate),
        "paired_hand_count": contract.SHARD_PAIR_COUNT,
        "root_count": contract.SHARD_PAIR_COUNT * 2,
        "source_replayed": True,
        "create_only_artifacts": True,
        "parquet_required_for_smoke_pass": False,
        "parquet_exported": False,
        "full_9000_paired_fanout_authorized": True,
        "full_fanout_started": False,
        "training_eligible": False,
        "teacher_values_are_realized_match_ev": False,
        "current_profile_changed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run/resume the create-only 25-paired M3.1 local dataset pilot."
        )
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--fresh-quality-gate", required=True)
    parser.add_argument("--library", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_local_pilot(
        output_root=args.output_root,
        fresh_quality_gate_path=args.fresh_quality_gate,
        library_path=args.library,
    )
    print(canonical_bytes(result).decode("ascii"), end="")
    return 0


__all__ = [
    "PILOT_RESULT_SCHEMA",
    "canonical_bytes",
    "canonical_sha256",
    "main",
    "run_local_pilot",
]


if __name__ == "__main__":
    raise SystemExit(main())
