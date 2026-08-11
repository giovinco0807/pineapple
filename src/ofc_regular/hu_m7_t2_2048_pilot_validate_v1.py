"""Validate one received M7 T2 2,048-particle shard for pilot Go/No-Go.

This is deliberately a *single-shard* gate.  It applies the frozen production
worker-plan contract and the same full position validator used by the 25,000
position postflight, then independently authenticates the receiver's local
``SHARD_DONE.json`` and generation-pinned complete-checkpoint copy.  A PASS is
evidence that one 143/144-position pilot shard is sound; it never substitutes
for, or relaxes, the paired full-corpus postflight.

The command is local and read-only apart from an optional JSON report file.  It
contains no cloud adapter, launch, profile, or runtime-policy operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any, Mapping, Sequence

from .hu_m31_label_gen_resume_v1 import COMPLETE_CHECKPOINT_SCHEMA
from .hu_m7_t2_2048_validate_v1 import (
    DEFAULT_CONTRACT,
    DONE_SCHEMA,
    T22048Contract,
    T22048ValidationError,
    _SHA256_RE,
    _canonical_bytes,
    _contract_digest,
    _fail,
    _read_json,
    _require_exact_keys,
    _require_int,
    _require_sha,
    _sha256,
    _validate_plan,
    _validate_position,
)


REPORT_SCHEMA = "hu_m7_t2_2048_pilot_shard_report_v1"
RECEIVE_EVIDENCE_SCHEMA = "hu_m31_label_gen_shard_receive_evidence_v2"
POSITION_GENERATION_MANIFEST_SCHEMA = (
    "hu_m31_label_gen_position_generation_manifest_v1"
)
_CHECKPOINT_FIELDS = {
    "schema",
    "checkpoint_kind",
    "plan_sha256",
    "shard_id",
    "attempt_id",
    "completed_position_count",
    "complete",
    "files",
    "checkpoint_published_after_files",
    "create_only",
    "checkpoint_sha256",
}
_CHECKPOINT_FILE_FIELDS = {
    "relative_path",
    "object_name",
    "sha256",
    "bytes",
}
_DONE_FIELDS = {"schema", "plan_sha256", "shard_id", "positions"}
_RECEIVE_EVIDENCE_FIELDS = {
    "schema",
    "run_name",
    "bucket",
    "worker_plan_sha256",
    "shard_id",
    "positions",
    "done_object",
    "complete_checkpoint_object",
    "checkpoint_sha256",
    "position_inventory_sha256",
    "position_generation_manifest",
    "generation_pinned_done_complete_and_all_positions",
}
_OBJECT_EVIDENCE_FIELDS = {"object_name", "generation", "sha256", "bytes"}
_GENERATION_REFERENCE_FIELDS = {"relative_path", "sha256", "positions"}
_GENERATION_MANIFEST_FIELDS = {
    "schema",
    "run_name",
    "bucket",
    "worker_plan_sha256",
    "shard_id",
    "positions",
    "files",
    "exact_prefix_inventory",
    "checkpoint_sha_and_bytes_matched",
    "all_position_gets_generation_pinned",
    "manifest_sha256",
}
_GENERATION_ROW_FIELDS = {
    "relative_path",
    "object_name",
    "generation",
    "bytes",
    "sha256",
}


def _expected_relative_paths(shard: Mapping[str, Any]) -> set[str]:
    start = int(shard["start"])
    count = int(shard["count"])
    return {"SHARD_DONE.json"} | {
        f"position_{offset:08d}.json" for offset in range(start, start + count)
    }


def _reject_link_chain(path: pathlib.Path, label: str) -> None:
    """Reject a symlink or junction at any component of an evidence path."""

    absolute = path.absolute()
    for component in (absolute, *absolute.parents):
        try:
            is_junction = getattr(component, "is_junction", lambda: False)()
            if component.is_symlink() or is_junction:
                _fail(f"{label} path chain contains a symlink or junction: {component}")
        except OSError as exc:
            _fail(f"cannot inspect {label} path component {component}: {exc}")


def _validate_checkpoint(
    checkpoint_path: pathlib.Path,
    *,
    plan_sha: str,
    shard: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]], str]:
    payload = _read_json(checkpoint_path, canonical=True)
    _require_exact_keys(payload, _CHECKPOINT_FIELDS, str(checkpoint_path))

    stated_digest = payload.get("checkpoint_sha256")
    unsigned = dict(payload)
    unsigned.pop("checkpoint_sha256", None)
    actual_digest = hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
    _require_sha(stated_digest, actual_digest, f"{checkpoint_path}.checkpoint_sha256")

    if payload.get("schema") != COMPLETE_CHECKPOINT_SCHEMA:
        _fail(f"{checkpoint_path} has unsupported complete-checkpoint schema")
    if payload.get("checkpoint_kind") != "complete":
        _fail(f"{checkpoint_path}.checkpoint_kind must be 'complete'")
    _require_sha(payload.get("plan_sha256"), plan_sha, f"{checkpoint_path}.plan_sha256")
    if payload.get("shard_id") != shard["shard_id"]:
        _fail(
            f"{checkpoint_path}.shard_id must be {shard['shard_id']!r}, got "
            f"{payload.get('shard_id')!r}"
        )
    attempt_id = payload.get("attempt_id")
    if not isinstance(attempt_id, str) or not attempt_id:
        _fail(f"{checkpoint_path}.attempt_id must be a non-empty string")
    _require_int(
        payload.get("completed_position_count"),
        int(shard["count"]),
        f"{checkpoint_path}.completed_position_count",
    )
    if payload.get("complete") is not True:
        _fail(f"{checkpoint_path}.complete must be true")
    if payload.get("checkpoint_published_after_files") is not True:
        _fail(f"{checkpoint_path} was not published after all files")
    if payload.get("create_only") is not True:
        _fail(f"{checkpoint_path}.create_only must be true")

    rows = payload.get("files")
    if not isinstance(rows, list):
        _fail(f"{checkpoint_path}.files must be a list")
    expected = _expected_relative_paths(shard)
    inventory: dict[str, Mapping[str, Any]] = {}
    prefixes: set[str] = set()
    for index, row in enumerate(rows):
        label = f"{checkpoint_path}.files[{index}]"
        if not isinstance(row, Mapping):
            _fail(f"{label} must be an object")
        _require_exact_keys(row, _CHECKPOINT_FILE_FIELDS, label)
        relative = row.get("relative_path")
        if not isinstance(relative, str) or relative not in expected:
            _fail(f"{label}.relative_path is not expected: {relative!r}")
        if relative in inventory:
            _fail(f"{checkpoint_path} duplicates {relative!r}")
        object_name = row.get("object_name")
        suffix = f"/files/{relative}"
        if not isinstance(object_name, str) or not object_name.endswith(suffix):
            _fail(f"{label}.object_name does not name {relative!r}")
        prefix = object_name[: -len(suffix)]
        if not prefix.endswith(f"/shards/{shard['shard_id']}"):
            _fail(f"{label}.object_name has the wrong shard prefix")
        prefixes.add(prefix)
        digest = row.get("sha256")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            _fail(f"{label}.sha256 is not a lowercase sha256")
        byte_count = row.get("bytes")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0:
            _fail(f"{label}.bytes must be a positive integer")
        inventory[relative] = row
    if set(inventory) != expected:
        _fail(
            f"{checkpoint_path} inventory differs: "
            f"missing={sorted(expected - set(inventory))[:3]}, "
            f"extra={sorted(set(inventory) - expected)[:3]}"
        )
    if len(prefixes) != 1:
        _fail(f"{checkpoint_path} inventory spans multiple object prefixes")
    return payload, inventory, next(iter(prefixes))


def _validate_local_file_against_inventory(
    path: pathlib.Path, row: Mapping[str, Any]
) -> None:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        _fail(f"cannot read {path}: {exc}")
    _require_int(row.get("bytes"), len(raw), f"{path} checkpoint byte count")
    _require_sha(row.get("sha256"), hashlib.sha256(raw).hexdigest(), f"{path} checkpoint")


def _validate_done(
    path: pathlib.Path,
    *,
    plan_sha: str,
    shard: Mapping[str, Any],
) -> None:
    marker = _read_json(path, canonical=True)
    _require_exact_keys(marker, _DONE_FIELDS, str(path))
    if marker.get("schema") != DONE_SCHEMA:
        _fail(f"{path} has unsupported SHARD_DONE schema")
    _require_sha(marker.get("plan_sha256"), plan_sha, f"{path}.plan_sha256")
    if marker.get("shard_id") != shard["shard_id"]:
        _fail(
            f"{path}.shard_id must be {shard['shard_id']!r}, got "
            f"{marker.get('shard_id')!r}"
        )
    _require_int(marker.get("positions"), int(shard["count"]), f"{path}.positions")


def _validate_object_evidence(
    value: Any,
    *,
    expected_name: str,
    expected_sha: str,
    expected_bytes: int,
    label: str,
) -> None:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be an object")
    _require_exact_keys(value, _OBJECT_EVIDENCE_FIELDS, label)
    if value.get("object_name") != expected_name:
        _fail(f"{label}.object_name is stale or foreign")
    generation = value.get("generation")
    if (
        not isinstance(generation, str)
        or not generation.isdigit()
        or int(generation) <= 0
    ):
        _fail(f"{label}.generation is invalid")
    _require_sha(value.get("sha256"), expected_sha, f"{label}.sha256")
    _require_int(value.get("bytes"), expected_bytes, f"{label}.bytes")


def _validate_receive_generation_evidence(
    *,
    receive_root: pathlib.Path,
    audit_directory: pathlib.Path,
    checkpoint_path: pathlib.Path,
    checkpoint: Mapping[str, Any],
    inventory: Mapping[str, Mapping[str, Any]],
    object_prefix: str,
    plan_sha: str,
    shard: Mapping[str, Any],
    done_path: pathlib.Path,
    position_inventory_sha256: str,
) -> tuple[str, str]:
    parts = object_prefix.split("/")
    if (
        len(parts) != 4
        or parts[0] != "labelgen"
        or not parts[1]
        or parts[2] != "shards"
        or parts[3] != shard["shard_id"]
    ):
        _fail("complete checkpoint has a non-production object prefix")
    run_name = parts[1]
    evidence_path = audit_directory / "receive_evidence.json"
    manifest_path = audit_directory / "position_generation_manifest.json"
    for path, label in (
        (evidence_path, "receive evidence"),
        (manifest_path, "position generation manifest"),
    ):
        _reject_link_chain(path, label)
        if not path.is_file():
            _fail(f"same-root {label} is missing: {path}")

    manifest = _read_json(manifest_path, canonical=True)
    _require_exact_keys(manifest, _GENERATION_MANIFEST_FIELDS, str(manifest_path))
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("manifest_sha256", None)
    manifest_self_sha = hashlib.sha256(
        _canonical_bytes(unsigned_manifest)
    ).hexdigest()
    _require_sha(
        manifest.get("manifest_sha256"),
        manifest_self_sha,
        f"{manifest_path}.manifest_sha256",
    )
    if (
        manifest.get("schema") != POSITION_GENERATION_MANIFEST_SCHEMA
        or manifest.get("run_name") != run_name
        or not isinstance(manifest.get("bucket"), str)
        or not manifest.get("bucket")
        or manifest.get("worker_plan_sha256") != plan_sha
        or manifest.get("shard_id") != shard["shard_id"]
        or manifest.get("exact_prefix_inventory") is not True
        or manifest.get("checkpoint_sha_and_bytes_matched") is not True
        or manifest.get("all_position_gets_generation_pinned") is not True
    ):
        _fail(f"{manifest_path} provenance or generation claims drifted")
    _require_int(
        manifest.get("positions"), int(shard["count"]), f"{manifest_path}.positions"
    )
    rows = manifest.get("files")
    if not isinstance(rows, list):
        _fail(f"{manifest_path}.files must be a list")
    expected_positions = {
        relative: row
        for relative, row in inventory.items()
        if relative.startswith("position_")
    }
    observed: set[str] = set()
    for index, row in enumerate(rows):
        label = f"{manifest_path}.files[{index}]"
        if not isinstance(row, Mapping):
            _fail(f"{label} must be an object")
        _require_exact_keys(row, _GENERATION_ROW_FIELDS, label)
        relative = row.get("relative_path")
        if not isinstance(relative, str) or relative not in expected_positions:
            _fail(f"{label}.relative_path is unexpected")
        if relative in observed:
            _fail(f"{manifest_path} duplicates {relative!r}")
        observed.add(relative)
        expected = expected_positions[relative]
        if row.get("object_name") != f"{object_prefix}/files/{relative}":
            _fail(f"{label}.object_name drifted")
        generation = row.get("generation")
        if (
            not isinstance(generation, str)
            or not generation.isdigit()
            or int(generation) <= 0
        ):
            _fail(f"{label}.generation is invalid")
        _require_int(row.get("bytes"), int(expected["bytes"]), f"{label}.bytes")
        _require_sha(row.get("sha256"), str(expected["sha256"]), f"{label}.sha256")
    if observed != set(expected_positions):
        _fail(f"{manifest_path} does not cover the checkpoint position inventory")

    evidence = _read_json(evidence_path, canonical=True)
    _require_exact_keys(evidence, _RECEIVE_EVIDENCE_FIELDS, str(evidence_path))
    if (
        evidence.get("schema") != RECEIVE_EVIDENCE_SCHEMA
        or evidence.get("run_name") != run_name
        or evidence.get("bucket") != manifest.get("bucket")
        or evidence.get("worker_plan_sha256") != plan_sha
        or evidence.get("shard_id") != shard["shard_id"]
        or evidence.get("checkpoint_sha256") != checkpoint["checkpoint_sha256"]
        or evidence.get("position_inventory_sha256")
        != position_inventory_sha256
        or evidence.get("generation_pinned_done_complete_and_all_positions")
        is not True
    ):
        _fail(f"{evidence_path} is stale or foreign")
    _require_int(
        evidence.get("positions"), int(shard["count"]), f"{evidence_path}.positions"
    )
    _validate_object_evidence(
        evidence.get("complete_checkpoint_object"),
        expected_name=f"{object_prefix}/checkpoints/complete.json",
        expected_sha=_sha256(checkpoint_path),
        expected_bytes=checkpoint_path.stat().st_size,
        label=f"{evidence_path}.complete_checkpoint_object",
    )
    _validate_object_evidence(
        evidence.get("done_object"),
        expected_name=f"{object_prefix}/files/SHARD_DONE.json",
        expected_sha=_sha256(done_path),
        expected_bytes=done_path.stat().st_size,
        label=f"{evidence_path}.done_object",
    )
    manifest_reference = evidence.get("position_generation_manifest")
    if not isinstance(manifest_reference, Mapping):
        _fail(f"{evidence_path}.position_generation_manifest must be an object")
    _require_exact_keys(
        manifest_reference,
        _GENERATION_REFERENCE_FIELDS,
        f"{evidence_path}.position_generation_manifest",
    )
    expected_relative = manifest_path.relative_to(receive_root).as_posix()
    if manifest_reference.get("relative_path") != expected_relative:
        _fail(f"{evidence_path} points outside the same-root generation manifest")
    _require_sha(
        manifest_reference.get("sha256"),
        _sha256(manifest_path),
        f"{evidence_path}.position_generation_manifest.sha256",
    )
    _require_int(
        manifest_reference.get("positions"),
        int(shard["count"]),
        f"{evidence_path}.position_generation_manifest.positions",
    )
    return _sha256(evidence_path), _sha256(manifest_path)


def validate_pilot_shard(
    *,
    worker_plan_path: pathlib.Path,
    shard_id: str,
    shard_directory: pathlib.Path,
    checkpoint_path: pathlib.Path,
    contract: T22048Contract = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    """Return a deterministic PASS report for one fully verified pilot shard."""

    plan = _read_json(worker_plan_path)
    seat = plan.get("seat")
    if seat not in ("first", "second"):
        _fail(f"worker plan seat must be 'first' or 'second', got {seat!r}")
    plan_result = _validate_plan(plan, str(seat), contract, "worker plan")
    plan_sha = _sha256(worker_plan_path)
    expected_plan_sha = (
        contract.first_plan_sha256 if seat == "first" else contract.second_plan_sha256
    )
    _require_sha(plan_sha, expected_plan_sha, f"{seat} worker plan")

    shard_rows = {
        str(row["shard_id"]): row for row in plan_result["shards"]
    }
    if shard_id not in shard_rows:
        _fail(f"worker plan has no shard {shard_id!r}")
    shard = shard_rows[shard_id]
    expected_shard_name = f"shard_{shard_id}"
    _reject_link_chain(shard_directory, "shard directory")
    if shard_directory.name != expected_shard_name or not shard_directory.is_dir():
        _fail(
            f"shard directory must be an existing {expected_shard_name!r} directory"
        )
    receive_root = shard_directory.absolute().parent
    audit_directory = receive_root / "_audit" / expected_shard_name
    expected_checkpoint = audit_directory / "complete.json"
    _reject_link_chain(checkpoint_path, "complete checkpoint")
    if checkpoint_path.absolute() != expected_checkpoint.absolute():
        _fail(
            "checkpoint must be bound to the shard's same receive root at "
            f"_audit/{expected_shard_name}/complete.json"
        )
    if not checkpoint_path.is_file():
        _fail(f"same-root complete checkpoint is missing: {checkpoint_path}")
    try:
        if checkpoint_path.resolve(strict=True) != expected_checkpoint.resolve(
            strict=True
        ):
            _fail("complete checkpoint resolved outside the shard receive root")
    except OSError as exc:
        _fail(f"cannot resolve same-root complete checkpoint: {exc}")

    expected_names = _expected_relative_paths(shard)
    entries = list(shard_directory.iterdir())
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            _fail(f"pilot shard contains a non-regular entry: {entry}")
    actual_names = {entry.name for entry in entries}
    if actual_names != expected_names:
        _fail(
            f"{shard_directory} contents differ: "
            f"missing={sorted(expected_names - actual_names)[:3]}, "
            f"extra={sorted(actual_names - expected_names)[:3]}"
        )

    checkpoint, inventory, object_prefix = _validate_checkpoint(
        checkpoint_path, plan_sha=plan_sha, shard=shard
    )
    done_path = shard_directory / "SHARD_DONE.json"
    _validate_local_file_against_inventory(done_path, inventory[done_path.name])
    _validate_done(done_path, plan_sha=plan_sha, shard=shard)

    start = int(shard["start"])
    count = int(shard["count"])
    action_counts: list[int] = []
    position_inventory: list[dict[str, Any]] = []
    for offset in range(start, start + count):
        relative = f"position_{offset:08d}.json"
        path = shard_directory / relative
        row = inventory[relative]
        _validate_local_file_against_inventory(path, row)
        action_counts.append(
            _validate_position(path, offset, plan_sha, str(seat), contract)
        )
        position_inventory.append(
            {
                "relative_path": relative,
                "sha256": row["sha256"],
                "bytes": row["bytes"],
            }
        )

    position_inventory_sha256 = hashlib.sha256(
        _canonical_bytes(position_inventory)
    ).hexdigest()
    receive_evidence_sha256, generation_manifest_sha256 = (
        _validate_receive_generation_evidence(
            receive_root=receive_root,
            audit_directory=audit_directory,
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            inventory=inventory,
            object_prefix=object_prefix,
            plan_sha=plan_sha,
            shard=shard,
            done_path=done_path,
            position_inventory_sha256=position_inventory_sha256,
        )
    )

    return {
        "schema": REPORT_SCHEMA,
        "status": "PASS",
        "scope": "single_shard_pilot_only",
        "contract_sha256": _contract_digest(contract),
        "seat": seat,
        "shard_id": shard_id,
        "shard_start": start,
        "positions": count,
        "worker_plan_sha256": plan_sha,
        "complete_checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "complete_checkpoint_attempt_id": checkpoint["attempt_id"],
        "checkpoint_object_prefix": object_prefix,
        "position_inventory_sha256": position_inventory_sha256,
        "receive_evidence_sha256": receive_evidence_sha256,
        "position_generation_manifest_sha256": generation_manifest_sha256,
        "action_count_min": min(action_counts),
        "action_count_max": max(action_counts),
        "total_action_scores": sum(action_counts),
        "canonical_done_verified": True,
        "complete_checkpoint_verified": True,
        "same_root_receive_evidence_verified": True,
        "all_position_generations_verified": True,
        "all_positions_fully_validated": True,
        "custom_fields_rejected": True,
        "hidden_truth_rejected": True,
        "full_25000_postflight_not_satisfied_by_this_report": True,
    }


def _emit(report: Mapping[str, Any], output: pathlib.Path | None) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


def main(
    argv: Sequence[str] | None = None,
    *,
    contract: T22048Contract = DEFAULT_CONTRACT,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-plan", type=pathlib.Path, required=True)
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--shard-directory", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        report = validate_pilot_shard(
            worker_plan_path=args.worker_plan,
            shard_id=args.shard_id,
            shard_directory=args.shard_directory,
            checkpoint_path=args.checkpoint,
            contract=contract,
        )
    except T22048ValidationError as exc:
        print(f"T2 2048 pilot validation failed: {exc}", file=sys.stderr)
        return 1
    _emit(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
