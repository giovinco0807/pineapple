"""Build and verify a fail-closed R0 reproducibility snapshot.

The R0 snapshot is provenance only.  It never imports the policy registry,
selects ``current``, activates a runtime, stages Git files, or contacts cloud
services.  A snapshot can be created only from an explicit, hash-pinned file
allowlist after a read-only audit passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .artifact_manifest import (
    MANIFEST_SCHEMA as RUN_MANIFEST_SCHEMA,
    build_run_manifest,
    collect_git_state,
    discover_repo_root,
    sha256_file,
    write_run_manifest,
)


SPEC_SCHEMA = "hu_joint_policy_r0_snapshot_spec_v1"
AUDIT_SCHEMA = "hu_joint_policy_r0_snapshot_audit_v1"
PAYLOAD_MANIFEST_SCHEMA = "hu_joint_policy_r0_payload_manifest_v1"
READY_SCHEMA = "hu_joint_policy_r0_snapshot_ready_v1"
VERIFY_SCHEMA = "hu_joint_policy_r0_snapshot_verification_v1"

SPEC_COPY_NAME = "snapshot_spec.json"
PAYLOAD_MANIFEST_NAME = "payload_manifest.json"
ARCHIVE_NAME = "payload.zip"
RUN_MANIFEST_NAME = "run_manifest.json"
READY_NAME = "SNAPSHOT_READY.json"

ARCHIVE_SPEC_NAME = "SNAPSHOT_SPEC.json"
ARCHIVE_MANIFEST_NAME = "PAYLOAD_MANIFEST.json"
_RESERVED_ARCHIVE_NAMES = frozenset({ARCHIVE_SPEC_NAME, ARCHIVE_MANIFEST_NAME})
_EXPECTED_OUTPUT_FILES = frozenset(
    {
        SPEC_COPY_NAME,
        PAYLOAD_MANIFEST_NAME,
        ARCHIVE_NAME,
        RUN_MANIFEST_NAME,
        READY_NAME,
    }
)
_ALLOWED_KINDS = frozenset(
    {"source", "config", "test", "model", "binary", "artifact", "toolchain"}
)
_EXPECTED_AUTHORIZATION = {
    "cloud_authorized": False,
    "current_profile_changed": False,
    "promotion_authorized": False,
    "quality_authorized": False,
    "runtime_activation_authorized": False,
    "training_authorized": False,
}
_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
_HEX_DIGITS = frozenset("0123456789abcdef")


class SnapshotContractError(ValueError):
    """The snapshot specification or bundle shape is ambiguous or unsafe."""


class SnapshotVerificationError(ValueError):
    """A completed snapshot no longer matches its pinned bytes."""


class SnapshotNoGoError(RuntimeError):
    """Creation was requested while the read-only R0 audit was No-Go."""

    def __init__(self, report: Mapping[str, Any]):
        self.report = dict(report)
        super().__init__("R0 snapshot audit is No-Go")


@dataclass(frozen=True)
class SnapshotEntry:
    role: str
    kind: str
    source_scope: str
    source: str
    archive_path: str
    sha256: str


@dataclass(frozen=True)
class SnapshotGate:
    gate_id: str
    purpose: str
    excluded_roles: tuple[str, ...]


@dataclass(frozen=True)
class SnapshotSpec:
    path: Path
    raw_bytes: bytes
    payload: dict[str, Any]
    repo_root: Path
    entries: tuple[SnapshotEntry, ...]
    gates: tuple[SnapshotGate, ...]
    default_gate_id: str


def canonical_json_bytes(value: Any) -> bytes:
    """Return a portable canonical JSON representation."""
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def load_snapshot_spec(
    spec_path: str | Path, *, repo_root: str | Path
) -> SnapshotSpec:
    """Load and structurally validate a snapshot specification."""
    path = Path(spec_path).resolve()
    raw = path.read_bytes()
    payload = _load_json_bytes(raw, label="R0 snapshot spec")
    if not isinstance(payload, dict):
        raise SnapshotContractError("R0 snapshot spec must be a JSON object")
    _require_exact_keys(
        payload,
        {
            "schema",
            "snapshot_id",
            "repository",
            "policy_registry",
            "profiles",
            "authorization",
            "gates",
            "entries",
            "notes",
        },
        "R0 snapshot spec",
    )
    if payload["schema"] != SPEC_SCHEMA:
        raise SnapshotContractError("R0 snapshot spec schema changed")
    snapshot_id = payload["snapshot_id"]
    if not isinstance(snapshot_id, str) or not snapshot_id.strip():
        raise SnapshotContractError("snapshot_id must be a non-empty string")

    repository = payload["repository"]
    if not isinstance(repository, dict):
        raise SnapshotContractError("repository contract must be an object")
    _require_exact_keys(
        repository,
        {
            "head_commit",
            "branch",
            "upstream",
            "upstream_commit",
            "require_dirty",
        },
        "repository contract",
    )
    for key in ("head_commit", "branch"):
        if not isinstance(repository[key], str) or not repository[key]:
            raise SnapshotContractError(f"repository {key} must be non-empty")
    for key in ("upstream", "upstream_commit"):
        if repository[key] is not None and not isinstance(repository[key], str):
            raise SnapshotContractError(f"repository {key} must be string or null")
    if type(repository["require_dirty"]) is not bool:
        raise SnapshotContractError("repository require_dirty must be boolean")

    profiles = payload["profiles"]
    if (
        not isinstance(profiles, list)
        or any(not isinstance(value, str) or not value for value in profiles)
        or len(set(profiles)) != len(profiles)
    ):
        raise SnapshotContractError("profiles must be unique non-empty strings")
    if "current" in profiles:
        raise SnapshotContractError("R0 profiles must be explicit; current is forbidden")

    authorization = payload["authorization"]
    if authorization != _EXPECTED_AUTHORIZATION:
        raise SnapshotContractError("R0 authorization must keep every action disabled")

    gates_contract = payload["gates"]
    if not isinstance(gates_contract, dict):
        raise SnapshotContractError("gates contract must be an object")
    _require_exact_keys(
        gates_contract,
        {"default_gate_id", "definitions"},
        "gates contract",
    )
    default_gate_id = gates_contract["default_gate_id"]
    raw_gates = gates_contract["definitions"]
    if (
        not isinstance(default_gate_id, str)
        or not default_gate_id
        or not isinstance(raw_gates, list)
        or not raw_gates
    ):
        raise SnapshotContractError("gates contract changed")
    gates: list[SnapshotGate] = []
    gate_ids: set[str] = set()
    for index, value in enumerate(raw_gates):
        if not isinstance(value, dict):
            raise SnapshotContractError(f"gate {index} must be an object")
        _require_exact_keys(
            value,
            {"gate_id", "purpose", "excluded_roles"},
            f"gate {index}",
        )
        gate_id = value["gate_id"]
        purpose = value["purpose"]
        excluded_roles = value["excluded_roles"]
        if (
            not isinstance(gate_id, str)
            or not gate_id
            or gate_id in gate_ids
            or not isinstance(purpose, str)
            or not purpose
            or not isinstance(excluded_roles, list)
            or any(not isinstance(role, str) or not role for role in excluded_roles)
            or len(excluded_roles) != len(set(excluded_roles))
        ):
            raise SnapshotContractError(f"gate {index} contract changed")
        gate_ids.add(gate_id)
        gates.append(
            SnapshotGate(
                gate_id=gate_id,
                purpose=purpose,
                excluded_roles=tuple(excluded_roles),
            )
        )
    if default_gate_id not in gate_ids:
        raise SnapshotContractError("default gate is absent from definitions")

    policy_registry = payload["policy_registry"]
    if not isinstance(policy_registry, dict):
        raise SnapshotContractError("policy_registry contract must be an object")
    _require_exact_keys(
        policy_registry,
        {"role", "sha256", "current_profile_changed"},
        "policy_registry contract",
    )
    if (
        policy_registry["current_profile_changed"] is not False
        or not _valid_sha256(policy_registry["sha256"])
        or not isinstance(policy_registry["role"], str)
        or not policy_registry["role"]
    ):
        raise SnapshotContractError("policy_registry contract changed")

    notes = payload["notes"]
    if not isinstance(notes, list) or any(not isinstance(note, str) for note in notes):
        raise SnapshotContractError("notes must be a string list")

    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise SnapshotContractError("entries must be a non-empty list")
    entries: list[SnapshotEntry] = []
    roles: set[str] = set()
    sources: set[tuple[str, str]] = set()
    archive_paths: set[str] = set()
    for index, value in enumerate(raw_entries):
        if not isinstance(value, dict):
            raise SnapshotContractError(f"entry {index} must be an object")
        _require_exact_keys(
            value,
            {"role", "kind", "source_scope", "source", "archive_path", "sha256"},
            f"entry {index}",
        )
        role = value["role"]
        kind = value["kind"]
        source_scope = value["source_scope"]
        source = value["source"]
        archive_path = value["archive_path"]
        digest = value["sha256"]
        if not isinstance(role, str) or not role:
            raise SnapshotContractError(f"entry {index} role is invalid")
        if kind not in _ALLOWED_KINDS:
            raise SnapshotContractError(f"entry {index} kind is invalid")
        if source_scope not in {"repo", "external"}:
            raise SnapshotContractError(f"entry {index} source_scope is invalid")
        if not isinstance(source, str) or not source:
            raise SnapshotContractError(f"entry {index} source is invalid")
        if not isinstance(archive_path, str):
            raise SnapshotContractError(f"entry {index} archive_path is invalid")
        _validate_archive_path(archive_path, label=f"entry {index} archive_path")
        if archive_path in _RESERVED_ARCHIVE_NAMES:
            raise SnapshotContractError(
                f"entry {index} uses a reserved archive path"
            )
        if not _valid_sha256(digest):
            raise SnapshotContractError(f"entry {index} SHA-256 is invalid")
        if source_scope == "repo":
            _validate_repo_relative_path(source, label=f"entry {index} source")
        else:
            _validate_external_path(source, label=f"entry {index} source")
        source_key = (source_scope, _source_identity(source_scope, source))
        if role in roles:
            raise SnapshotContractError(f"duplicate entry role: {role}")
        if source_key in sources:
            raise SnapshotContractError(f"duplicate entry source: {source}")
        if archive_path in archive_paths:
            raise SnapshotContractError(
                f"duplicate entry archive_path: {archive_path}"
            )
        roles.add(role)
        sources.add(source_key)
        archive_paths.add(archive_path)
        entries.append(
            SnapshotEntry(
                role=role,
                kind=kind,
                source_scope=source_scope,
                source=source,
                archive_path=archive_path,
                sha256=digest,
            )
        )

    registry_role = policy_registry["role"]
    registry_matches = [entry for entry in entries if entry.role == registry_role]
    if len(registry_matches) != 1:
        raise SnapshotContractError("policy_registry role is absent from entries")
    if registry_matches[0].sha256 != policy_registry["sha256"]:
        raise SnapshotContractError("policy_registry SHA differs from its entry")
    for gate in gates:
        unknown_roles = set(gate.excluded_roles) - roles
        if unknown_roles:
            raise SnapshotContractError(
                f"gate {gate.gate_id} excludes unknown roles: "
                + ", ".join(sorted(unknown_roles))
            )
        if registry_role in gate.excluded_roles:
            raise SnapshotContractError(
                f"gate {gate.gate_id} cannot exclude the policy registry"
            )
        if len(gate.excluded_roles) >= len(entries):
            raise SnapshotContractError(
                f"gate {gate.gate_id} must retain at least one entry"
            )

    return SnapshotSpec(
        path=path,
        raw_bytes=raw,
        payload=payload,
        repo_root=Path(repo_root).resolve(),
        entries=tuple(entries),
        gates=tuple(gates),
        default_gate_id=default_gate_id,
    )


def audit_snapshot(
    spec: SnapshotSpec, *, gate_id: str | None = None
) -> dict[str, Any]:
    """Read every pinned input and return a Go/No-Go report without writing."""
    selected_gate = _resolve_gate(spec, gate_id)
    state_before = collect_git_state(spec.repo_root)
    repository_issues: list[dict[str, Any]] = []
    expected_repo = spec.payload["repository"]
    for field in ("head_commit", "branch", "upstream", "upstream_commit"):
        if state_before.get(field) != expected_repo[field]:
            repository_issues.append(
                {
                    "code": "repository_pin_mismatch",
                    "field": field,
                    "expected": expected_repo[field],
                    "actual": state_before.get(field),
                }
            )
    if bool(state_before["dirty"]) != expected_repo["require_dirty"]:
        repository_issues.append(
            {
                "code": "repository_dirty_state_mismatch",
                "expected": expected_repo["require_dirty"],
                "actual": bool(state_before["dirty"]),
            }
        )

    verified_by_role: dict[str, dict[str, Any]] = {}
    entry_issues_by_role: dict[str, dict[str, Any]] = {}
    for entry in spec.entries:
        path = _entry_source_path(spec, entry)
        issue = _audit_entry(path, entry, repo_root=spec.repo_root)
        if issue is not None:
            entry_issues_by_role[entry.role] = issue
            continue
        verified_by_role[entry.role] = {
            "role": entry.role,
            "kind": entry.kind,
            "archive_path": entry.archive_path,
            "sha256": entry.sha256,
            "size_bytes": path.stat().st_size,
        }

    state_after = collect_git_state(spec.repo_root)
    if canonical_sha256(state_after) != canonical_sha256(state_before):
        repository_issues.append({"code": "repository_changed_during_audit"})

    registry_role = spec.payload["policy_registry"]["role"]
    registry_verified = verified_by_role.get(registry_role)
    full_gate_results = [
        _gate_audit_result(
            spec,
            gate,
            repository_issues=repository_issues,
            entry_issues_by_role=entry_issues_by_role,
            verified_by_role=verified_by_role,
        )
        for gate in spec.gates
    ]
    selected_result = next(
        result
        for result in full_gate_results
        if result["gate_id"] == selected_gate.gate_id
    )
    gate_results = [
        {
            key: value
            for key, value in result.items()
            if key != "verified_entries"
        }
        for result in full_gate_results
    ]
    repository = _repository_summary(state_before)
    return {
        "schema": AUDIT_SCHEMA,
        "status": selected_result["status"],
        "decision": (
            "r0_snapshot_inputs_verified"
            if selected_result["status"] == "go"
            else "r0_snapshot_blocked_fail_closed"
        ),
        "snapshot_id": spec.payload["snapshot_id"],
        "selected_gate_id": selected_gate.gate_id,
        "selected_gate_purpose": selected_gate.purpose,
        "spec_sha256": hashlib.sha256(spec.raw_bytes).hexdigest(),
        "repository": repository,
        "repository_state_sha256": canonical_sha256(state_before),
        "inventory_entry_count": len(spec.entries),
        "entry_count": selected_result["entry_count"],
        "verified_entry_count": selected_result["verified_entry_count"],
        "verified_entries": selected_result["verified_entries"],
        "issues": selected_result["issues"],
        "deferred_entry_issues": selected_result["deferred_entry_issues"],
        "gate_results": gate_results,
        "policy_registry_sha256": (
            registry_verified["sha256"] if registry_verified is not None else None
        ),
        "current_profile_changed": False,
        "authorization": dict(spec.payload["authorization"]),
    }


def _resolve_gate(spec: SnapshotSpec, gate_id: str | None) -> SnapshotGate:
    selected = spec.default_gate_id if gate_id is None else gate_id
    for gate in spec.gates:
        if gate.gate_id == selected:
            return gate
    raise SnapshotContractError(f"unknown R0 snapshot gate: {selected}")


def _entries_for_gate(
    spec: SnapshotSpec, gate: SnapshotGate
) -> tuple[SnapshotEntry, ...]:
    excluded = set(gate.excluded_roles)
    return tuple(entry for entry in spec.entries if entry.role not in excluded)


def _gate_audit_result(
    spec: SnapshotSpec,
    gate: SnapshotGate,
    *,
    repository_issues: Sequence[Mapping[str, Any]],
    entry_issues_by_role: Mapping[str, Mapping[str, Any]],
    verified_by_role: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    selected_entries = _entries_for_gate(spec, gate)
    selected_roles = {entry.role for entry in selected_entries}
    issues = [dict(issue) for issue in repository_issues]
    issues.extend(
        dict(entry_issues_by_role[entry.role])
        for entry in selected_entries
        if entry.role in entry_issues_by_role
    )
    deferred_entry_issues = [
        dict(entry_issues_by_role[entry.role])
        for entry in spec.entries
        if entry.role not in selected_roles and entry.role in entry_issues_by_role
    ]
    verified_entries = [
        dict(verified_by_role[entry.role])
        for entry in selected_entries
        if entry.role in verified_by_role
    ]
    return {
        "gate_id": gate.gate_id,
        "purpose": gate.purpose,
        "status": "go" if not issues else "no_go",
        "decision": (
            "gate_inputs_verified"
            if not issues
            else "gate_blocked_fail_closed"
        ),
        "entry_count": len(selected_entries),
        "verified_entry_count": len(verified_entries),
        "excluded_roles": list(gate.excluded_roles),
        "verified_entries": verified_entries,
        "issues": issues,
        "deferred_entry_issues": deferred_entry_issues,
    }


def create_snapshot(
    spec: SnapshotSpec,
    output_dir: str | Path,
    *,
    gate_id: str | None = None,
) -> dict[str, Any]:
    """Create a self-contained snapshot only after a clean audit."""
    selected_gate = _resolve_gate(spec, gate_id)
    selected_entries = _entries_for_gate(spec, selected_gate)
    output = _absolute_without_resolving(Path(output_dir))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"snapshot output already exists: {output}")
    _require_safe_output_location(spec.repo_root, output)
    audit = audit_snapshot(spec, gate_id=selected_gate.gate_id)
    if audit["status"] != "go":
        raise SnapshotNoGoError(audit)

    state_before = collect_git_state(spec.repo_root)
    if canonical_sha256(state_before) != audit["repository_state_sha256"]:
        raise SnapshotVerificationError(
            "repository changed after the R0 audit completed"
        )
    policy_path = _entry_source_path(
        spec,
        next(
            entry
            for entry in spec.entries
            if entry.role == spec.payload["policy_registry"]["role"]
        ),
    )
    policy_hash_before = sha256_file(policy_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent))
    )
    try:
        assert staging is not None
        spec_copy = staging / SPEC_COPY_NAME
        spec_copy.write_bytes(spec.raw_bytes)
        payload_manifest = _build_payload_manifest(
            spec,
            gate=selected_gate,
            entries=selected_entries,
        )
        payload_manifest_bytes = canonical_json_bytes(payload_manifest)
        payload_manifest_path = staging / PAYLOAD_MANIFEST_NAME
        payload_manifest_path.write_bytes(payload_manifest_bytes)
        archive_path = staging / ARCHIVE_NAME
        _write_payload_archive(
            spec,
            archive_path,
            entries=selected_entries,
            payload_manifest_bytes=payload_manifest_bytes,
        )
        _verify_archive(
            archive_path,
            spec_bytes=spec.raw_bytes,
            payload_manifest=payload_manifest,
            payload_manifest_bytes=payload_manifest_bytes,
        )

        run_manifest = build_run_manifest(
            repo_root=spec.repo_root,
            run_id=spec.payload["snapshot_id"],
            phase="r0_reproducible_snapshot",
            profiles=tuple(spec.payload["profiles"]),
            artifacts=(
                ("snapshot_spec", spec_copy),
                ("payload_manifest", payload_manifest_path),
                ("payload_archive", archive_path),
            ),
            command=(
                "python -m ofc_regular.freeze_hu_joint_policy_r0 create "
                f"--spec {spec.path} --gate {selected_gate.gate_id} "
                f"--output {output}"
            ),
            metadata={
                "archive_entry_count": str(len(selected_entries) + 2),
                "current_profile_changed": "false",
                "selected_gate_id": selected_gate.gate_id,
                "runtime_activation_authorized": "false",
            },
            notes=tuple(spec.payload["notes"]),
        )
        final_artifact_paths = {
            "snapshot_spec": output / SPEC_COPY_NAME,
            "payload_manifest": output / PAYLOAD_MANIFEST_NAME,
            "payload_archive": output / ARCHIVE_NAME,
        }
        for record in run_manifest["artifacts"]:
            display_path, repository_relative = _display_path(
                final_artifact_paths[record["role"]], spec.repo_root
            )
            record["path"] = display_path
            record["repository_relative"] = repository_relative
        run_manifest_path = staging / RUN_MANIFEST_NAME
        write_run_manifest(run_manifest_path, run_manifest)

        state_after = collect_git_state(spec.repo_root)
        policy_hash_after = sha256_file(policy_path)
        if canonical_sha256(state_after) != canonical_sha256(state_before):
            raise SnapshotVerificationError(
                "repository changed while creating R0 snapshot"
            )
        if policy_hash_after != policy_hash_before:
            raise SnapshotVerificationError(
                "policy registry changed while creating R0 snapshot"
            )

        ready = {
            "schema": READY_SCHEMA,
            "status": "r0_snapshot_ready_not_activated",
            "snapshot_id": spec.payload["snapshot_id"],
            "selected_gate_id": selected_gate.gate_id,
            "selected_gate_purpose": selected_gate.purpose,
            "spec_sha256": sha256_file(spec_copy),
            "payload_manifest_sha256": sha256_file(payload_manifest_path),
            "payload_archive_sha256": sha256_file(archive_path),
            "run_manifest_sha256": sha256_file(run_manifest_path),
            "repository_state_sha256": canonical_sha256(state_before),
            "policy_registry_sha256": policy_hash_before,
            "inventory_entry_count": len(spec.entries),
            "entry_count": len(selected_entries),
            "current_profile_changed": False,
            "authorization": dict(spec.payload["authorization"]),
        }
        write_run_manifest(staging / READY_NAME, ready)
        os.replace(staging, output)
        staging = None
        return verify_snapshot(output)
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


def verify_snapshot(snapshot_dir: str | Path) -> dict[str, Any]:
    """Verify every outer binding and every byte inside a completed snapshot."""
    root = Path(snapshot_dir)
    if root.is_symlink() or not root.is_dir():
        raise SnapshotVerificationError("snapshot directory is absent or symlinked")
    actual_files = {
        path.name for path in root.iterdir() if path.is_file() and not path.is_symlink()
    }
    if actual_files != _EXPECTED_OUTPUT_FILES or any(
        path.is_symlink() or not path.is_file() for path in root.iterdir()
    ):
        raise SnapshotVerificationError("snapshot output file set changed")

    spec_path = root / SPEC_COPY_NAME
    manifest_path = root / PAYLOAD_MANIFEST_NAME
    archive_path = root / ARCHIVE_NAME
    run_manifest_path = root / RUN_MANIFEST_NAME
    ready_path = root / READY_NAME
    ready = _load_json_file(ready_path, label="R0 ready receipt")
    if not isinstance(ready, dict):
        raise SnapshotVerificationError("R0 ready receipt must be an object")
    _require_exact_keys(
        ready,
        {
            "schema",
            "status",
            "snapshot_id",
            "selected_gate_id",
            "selected_gate_purpose",
            "spec_sha256",
            "payload_manifest_sha256",
            "payload_archive_sha256",
            "run_manifest_sha256",
            "repository_state_sha256",
            "policy_registry_sha256",
            "inventory_entry_count",
            "entry_count",
            "current_profile_changed",
            "authorization",
        },
        "R0 ready receipt",
        error_type=SnapshotVerificationError,
    )
    if (
        ready["schema"] != READY_SCHEMA
        or ready["status"] != "r0_snapshot_ready_not_activated"
        or not isinstance(ready["selected_gate_id"], str)
        or not ready["selected_gate_id"]
        or not isinstance(ready["selected_gate_purpose"], str)
        or not ready["selected_gate_purpose"]
        or type(ready["inventory_entry_count"]) is not int
        or ready["inventory_entry_count"] < 1
        or type(ready["entry_count"]) is not int
        or ready["entry_count"] < 1
        or ready["current_profile_changed"] is not False
        or ready["authorization"] != _EXPECTED_AUTHORIZATION
    ):
        raise SnapshotVerificationError("R0 ready receipt authorization changed")
    expected_outer_hashes = {
        "spec_sha256": spec_path,
        "payload_manifest_sha256": manifest_path,
        "payload_archive_sha256": archive_path,
        "run_manifest_sha256": run_manifest_path,
    }
    for field, path in expected_outer_hashes.items():
        if not _valid_sha256(ready.get(field)) or sha256_file(path) != ready[field]:
            raise SnapshotVerificationError(f"snapshot {field} binding changed")

    try:
        copied_spec = load_snapshot_spec(spec_path, repo_root=root)
        selected_gate = _resolve_gate(copied_spec, ready["selected_gate_id"])
    except SnapshotContractError as exc:
        raise SnapshotVerificationError("snapshot spec copy contract changed") from exc
    spec_payload = copied_spec.payload
    selected_entries = _entries_for_gate(copied_spec, selected_gate)
    expected_entry_identity = {
        entry.role: {
            "role": entry.role,
            "kind": entry.kind,
            "source_scope": entry.source_scope,
            "source": entry.source,
            "archive_path": entry.archive_path,
            "sha256": entry.sha256,
        }
        for entry in selected_entries
    }
    payload_manifest = _load_json_file(
        manifest_path, label="snapshot payload manifest"
    )
    _validate_payload_manifest(payload_manifest)
    if (
        spec_payload.get("schema") != SPEC_SCHEMA
        or spec_payload.get("snapshot_id") != ready["snapshot_id"]
        or selected_gate.purpose != ready["selected_gate_purpose"]
        or payload_manifest["snapshot_id"] != ready["snapshot_id"]
        or payload_manifest["selected_gate_id"] != ready["selected_gate_id"]
        or payload_manifest["selected_gate_purpose"]
        != ready["selected_gate_purpose"]
        or payload_manifest["spec_sha256"] != ready["spec_sha256"]
        or payload_manifest["entry_count"] != ready["entry_count"]
        or ready["inventory_entry_count"] != len(copied_spec.entries)
        or {
            entry["role"]: {
                key: entry[key]
                for key in (
                    "role",
                    "kind",
                    "source_scope",
                    "source",
                    "archive_path",
                    "sha256",
                )
            }
            for entry in payload_manifest["entries"]
        }
        != expected_entry_identity
    ):
        raise SnapshotVerificationError("snapshot identity chain changed")
    _verify_archive(
        archive_path,
        spec_bytes=spec_path.read_bytes(),
        payload_manifest=payload_manifest,
        payload_manifest_bytes=manifest_path.read_bytes(),
    )

    run_manifest = _load_json_file(run_manifest_path, label="R0 run manifest")
    if run_manifest.get("schema") != RUN_MANIFEST_SCHEMA:
        raise SnapshotVerificationError("R0 run manifest schema changed")
    artifact_hashes = {
        row.get("role"): row.get("sha256")
        for row in run_manifest.get("artifacts", [])
        if isinstance(row, dict)
    }
    if artifact_hashes != {
        "snapshot_spec": ready["spec_sha256"],
        "payload_manifest": ready["payload_manifest_sha256"],
        "payload_archive": ready["payload_archive_sha256"],
    }:
        raise SnapshotVerificationError("R0 run manifest artifact bindings changed")

    return {
        "schema": VERIFY_SCHEMA,
        "status": "pass",
        "decision": "r0_snapshot_bytes_and_bindings_verified",
        "snapshot_id": ready["snapshot_id"],
        "selected_gate_id": ready["selected_gate_id"],
        "selected_gate_purpose": ready["selected_gate_purpose"],
        "inventory_entry_count": ready["inventory_entry_count"],
        "entry_count": ready["entry_count"],
        "payload_archive_sha256": ready["payload_archive_sha256"],
        "run_manifest_sha256": ready["run_manifest_sha256"],
        "policy_registry_sha256": ready["policy_registry_sha256"],
        "current_profile_changed": False,
        "authorization": dict(ready["authorization"]),
    }


def _build_payload_manifest(
    spec: SnapshotSpec,
    *,
    gate: SnapshotGate,
    entries: Sequence[SnapshotEntry],
) -> dict[str, Any]:
    manifest_entries: list[dict[str, Any]] = []
    for entry in sorted(entries, key=lambda value: value.archive_path):
        path = _entry_source_path(spec, entry)
        if _audit_entry(path, entry, repo_root=spec.repo_root) is not None:
            raise SnapshotVerificationError(
                f"entry changed after audit: {entry.role}"
            )
        manifest_entries.append(
            {
                "role": entry.role,
                "kind": entry.kind,
                "source_scope": entry.source_scope,
                "source": entry.source,
                "archive_path": entry.archive_path,
                "size_bytes": path.stat().st_size,
                "sha256": entry.sha256,
            }
        )
    repository = spec.payload["repository"]
    return {
        "schema": PAYLOAD_MANIFEST_SCHEMA,
        "snapshot_id": spec.payload["snapshot_id"],
        "selected_gate_id": gate.gate_id,
        "selected_gate_purpose": gate.purpose,
        "spec_sha256": hashlib.sha256(spec.raw_bytes).hexdigest(),
        "base_repository": {
            "head_commit": repository["head_commit"],
            "branch": repository["branch"],
            "upstream": repository["upstream"],
            "upstream_commit": repository["upstream_commit"],
        },
        "restore_contract": "checkout_base_commit_then_overlay_payload_entries",
        "entry_count": len(manifest_entries),
        "entries": manifest_entries,
        "profiles": list(spec.payload["profiles"]),
        "current_profile_changed": False,
        "authorization": dict(spec.payload["authorization"]),
    }


def _write_payload_archive(
    spec: SnapshotSpec,
    archive_path: Path,
    *,
    entries: Sequence[SnapshotEntry],
    payload_manifest_bytes: bytes,
) -> None:
    records: list[tuple[str, bytes]] = [
        (ARCHIVE_SPEC_NAME, spec.raw_bytes),
        (ARCHIVE_MANIFEST_NAME, payload_manifest_bytes),
    ]
    for entry in entries:
        path = _entry_source_path(spec, entry)
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry.sha256:
            raise SnapshotVerificationError(
                f"entry pin changed while archiving: {entry.role}"
            )
        records.append((entry.archive_path, payload))
    records.sort(key=lambda value: value[0])
    with zipfile.ZipFile(
        archive_path,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for name, payload in records:
            info = zipfile.ZipInfo(name, date_time=_FIXED_ZIP_TIME)
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(
                info,
                payload,
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def _verify_archive(
    archive_path: Path,
    *,
    spec_bytes: bytes,
    payload_manifest: Mapping[str, Any],
    payload_manifest_bytes: bytes,
) -> None:
    _validate_payload_manifest(payload_manifest)
    expected_payloads: dict[str, tuple[int, str]] = {
        ARCHIVE_SPEC_NAME: (
            len(spec_bytes),
            hashlib.sha256(spec_bytes).hexdigest(),
        ),
        ARCHIVE_MANIFEST_NAME: (
            len(payload_manifest_bytes),
            hashlib.sha256(payload_manifest_bytes).hexdigest(),
        ),
    }
    for entry in payload_manifest["entries"]:
        expected_payloads[entry["archive_path"]] = (
            entry["size_bytes"],
            entry["sha256"],
        )
    expected_names = sorted(expected_payloads)
    try:
        with zipfile.ZipFile(archive_path, mode="r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if names != expected_names or len(names) != len(set(names)):
                raise SnapshotVerificationError(
                    "snapshot archive entry set or order changed"
                )
            for info in infos:
                _validate_archive_path(
                    info.filename,
                    label="snapshot archive member",
                    error_type=SnapshotVerificationError,
                )
                if (
                    info.is_dir()
                    or info.date_time != _FIXED_ZIP_TIME
                    or info.create_system != 3
                    or (info.external_attr >> 16) & 0o777 != 0o644
                ):
                    raise SnapshotVerificationError(
                        f"snapshot archive metadata changed: {info.filename}"
                    )
                payload = archive.read(info)
                expected_size, expected_hash = expected_payloads[info.filename]
                if (
                    len(payload) != expected_size
                    or hashlib.sha256(payload).hexdigest() != expected_hash
                ):
                    raise SnapshotVerificationError(
                        f"snapshot archive member changed: {info.filename}"
                    )
            if archive.read(ARCHIVE_SPEC_NAME) != spec_bytes:
                raise SnapshotVerificationError("archived snapshot spec changed")
            if archive.read(ARCHIVE_MANIFEST_NAME) != payload_manifest_bytes:
                raise SnapshotVerificationError("archived payload manifest changed")
    except (zipfile.BadZipFile, RuntimeError) as exc:
        raise SnapshotVerificationError("snapshot archive is corrupt") from exc


def _validate_payload_manifest(value: Any) -> None:
    if not isinstance(value, dict):
        raise SnapshotVerificationError("payload manifest must be an object")
    _require_exact_keys(
        value,
        {
            "schema",
            "snapshot_id",
            "selected_gate_id",
            "selected_gate_purpose",
            "spec_sha256",
            "base_repository",
            "restore_contract",
            "entry_count",
            "entries",
            "profiles",
            "current_profile_changed",
            "authorization",
        },
        "payload manifest",
        error_type=SnapshotVerificationError,
    )
    if (
        value["schema"] != PAYLOAD_MANIFEST_SCHEMA
        or value["restore_contract"]
        != "checkout_base_commit_then_overlay_payload_entries"
        or value["current_profile_changed"] is not False
        or value["authorization"] != _EXPECTED_AUTHORIZATION
        or not _valid_sha256(value["spec_sha256"])
        or type(value["entry_count"]) is not int
        or value["entry_count"] < 1
        or not isinstance(value["entries"], list)
        or len(value["entries"]) != value["entry_count"]
        or not isinstance(value["snapshot_id"], str)
        or not value["snapshot_id"]
        or not isinstance(value["selected_gate_id"], str)
        or not value["selected_gate_id"]
        or not isinstance(value["selected_gate_purpose"], str)
        or not value["selected_gate_purpose"]
        or not isinstance(value["profiles"], list)
        or len(value["profiles"]) != len(set(value["profiles"]))
        or "current" in value["profiles"]
    ):
        raise SnapshotVerificationError("payload manifest contract changed")
    base_repository = value["base_repository"]
    if not isinstance(base_repository, dict):
        raise SnapshotVerificationError("payload base_repository changed")
    _require_exact_keys(
        base_repository,
        {"head_commit", "branch", "upstream", "upstream_commit"},
        "payload base_repository",
        error_type=SnapshotVerificationError,
    )
    roles: set[str] = set()
    paths: set[str] = set()
    sources: set[tuple[str, str]] = set()
    for index, entry in enumerate(value["entries"]):
        if not isinstance(entry, dict):
            raise SnapshotVerificationError("payload entry must be an object")
        _require_exact_keys(
            entry,
            {
                "role",
                "kind",
                "source_scope",
                "source",
                "archive_path",
                "size_bytes",
                "sha256",
            },
            f"payload entry {index}",
            error_type=SnapshotVerificationError,
        )
        if (
            not isinstance(entry["role"], str)
            or not entry["role"]
            or not isinstance(entry["source"], str)
            or not entry["source"]
            or entry["kind"] not in _ALLOWED_KINDS
            or entry["source_scope"] not in {"repo", "external"}
            or type(entry["size_bytes"]) is not int
            or entry["size_bytes"] < 0
            or not _valid_sha256(entry["sha256"])
        ):
            raise SnapshotVerificationError("payload entry contract changed")
        _validate_archive_path(
            entry["archive_path"],
            label="payload archive path",
            error_type=SnapshotVerificationError,
        )
        if entry["archive_path"] in _RESERVED_ARCHIVE_NAMES:
            raise SnapshotVerificationError(
                "payload entry uses a reserved archive path"
            )
        source_key = (
            entry["source_scope"],
            _source_identity(entry["source_scope"], entry["source"]),
        )
        if (
            entry["role"] in roles
            or entry["archive_path"] in paths
            or source_key in sources
        ):
            raise SnapshotVerificationError("payload manifest contains duplicates")
        roles.add(entry["role"])
        paths.add(entry["archive_path"])
        sources.add(source_key)


def _audit_entry(
    path: Path, entry: SnapshotEntry, *, repo_root: Path
) -> dict[str, Any] | None:
    if _has_symlink_component(
        path, stop=repo_root if entry.source_scope == "repo" else None
    ):
        return {
            "code": "symlink_forbidden",
            "role": entry.role,
            "source": entry.source,
        }
    if not path.is_file():
        return {
            "code": "missing_required_entry",
            "role": entry.role,
            "source": entry.source,
            "expected_sha256": entry.sha256,
        }
    before = path.stat()
    actual = sha256_file(path)
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        return {
            "code": "entry_changed_during_audit",
            "role": entry.role,
            "source": entry.source,
        }
    if actual != entry.sha256:
        return {
            "code": "entry_pin_mismatch",
            "role": entry.role,
            "source": entry.source,
            "expected_sha256": entry.sha256,
            "actual_sha256": actual,
        }
    return None


def _entry_source_path(spec: SnapshotSpec, entry: SnapshotEntry) -> Path:
    if entry.source_scope == "repo":
        return spec.repo_root.joinpath(*entry.source.split("/"))
    return Path(entry.source)


def _repository_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "head_commit": state["head_commit"],
        "branch": state["branch"],
        "upstream": state["upstream"],
        "upstream_commit": state["upstream_commit"],
        "dirty": state["dirty"],
        "status_porcelain_sha256": state["status_porcelain_sha256"],
        "tracked_diff_sha256": state["tracked_diff_sha256"],
        "untracked_file_count": state["untracked_file_count"],
    }


def _display_path(path: Path, repo_root: Path) -> tuple[str, bool]:
    absolute = _absolute_without_resolving(path)
    try:
        return absolute.relative_to(repo_root).as_posix(), True
    except ValueError:
        return str(absolute), False


def _require_safe_output_location(repo_root: Path, output: Path) -> None:
    try:
        relative = output.relative_to(repo_root)
    except ValueError:
        return
    if relative == Path("."):
        raise SnapshotContractError("snapshot output cannot be the repository root")
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "check-ignore",
            "--quiet",
            "--",
            relative.as_posix(),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise SnapshotContractError(
            "snapshot output inside repository must be Git-ignored"
        )


def _validate_repo_relative_path(
    value: str,
    *,
    label: str,
    error_type: type[ValueError] = SnapshotContractError,
) -> None:
    if (
        "\\" in value
        or value.startswith("/")
        or ":" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise error_type(f"{label} must be a normalized repository-relative path")


def _validate_external_path(value: str, *, label: str) -> None:
    if "\x00" in value or any(part == ".." for part in Path(value).parts):
        raise SnapshotContractError(f"{label} contains traversal")
    if not Path(value).is_absolute():
        raise SnapshotContractError(f"{label} must be absolute for external scope")


def _validate_archive_path(
    value: str,
    *,
    label: str,
    error_type: type[ValueError] = SnapshotContractError,
) -> None:
    if not isinstance(value, str):
        raise error_type(f"{label} must be a string")
    pure = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or value.startswith("/")
        or ":" in value
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise error_type(f"{label} contains traversal or is not portable")


def _source_identity(scope: str, value: str) -> str:
    if scope == "repo":
        return value
    return os.path.normcase(os.path.abspath(value))


def _has_symlink_component(path: Path, *, stop: Path | None) -> bool:
    cursor = path
    stop_resolved = stop.resolve() if stop is not None else None
    while True:
        if cursor.is_symlink():
            return True
        if stop_resolved is not None and cursor == stop_resolved:
            return False
        parent = cursor.parent
        if parent == cursor:
            return False
        cursor = parent


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in _HEX_DIGITS for character in value)
    )


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
    *,
    error_type: type[ValueError] = SnapshotContractError,
) -> None:
    if set(value) != expected:
        raise error_type(f"{label} keys changed")


def _load_json_file(path: Path, *, label: str) -> Any:
    try:
        return _load_json_bytes(path.read_bytes(), label=label)
    except OSError as exc:
        raise SnapshotVerificationError(f"{label} cannot be read") from exc


def _load_json_bytes(payload: bytes, *, label: str) -> Any:
    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SnapshotContractError(f"{label} contains duplicate key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotContractError(f"{label} is not canonical UTF-8 JSON") from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "create"):
        child = subparsers.add_parser(command)
        child.add_argument("--spec", type=Path, required=True)
        child.add_argument("--repo-root", type=Path)
        child.add_argument("--gate")
        if command == "create":
            child.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--snapshot", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "verify":
            result = verify_snapshot(args.snapshot)
        else:
            repo_root = (
                args.repo_root.resolve()
                if args.repo_root is not None
                else discover_repo_root()
            )
            spec = load_snapshot_spec(args.spec, repo_root=repo_root)
            if args.command == "audit":
                result = audit_snapshot(spec, gate_id=args.gate)
            else:
                result = create_snapshot(spec, args.output, gate_id=args.gate)
    except SnapshotNoGoError as exc:
        result = exc.report
    except (
        FileExistsError,
        FileNotFoundError,
        SnapshotContractError,
        SnapshotVerificationError,
    ) as exc:
        result = {
            "schema": AUDIT_SCHEMA,
            "status": "error",
            "decision": "r0_snapshot_failed_closed",
            "error": str(exc),
            "current_profile_changed": False,
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("status") in {"go", "pass"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
