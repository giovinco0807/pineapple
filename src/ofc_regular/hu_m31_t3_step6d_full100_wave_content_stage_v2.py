"""Create-only content staging for the full-100 wave-v2 outer package.

This module is deliberately transport-agnostic.  The caller supplies a small
adapter that can list one exact prefix, create one object with generation-match
zero, and delete one object at an exact generation.  No live cloud client is
imported here.

The outer package remains the sole source of content truth.  Existing objects,
including byte-identical objects, are never adopted.  Cleanup is restricted to
the generations proven to have been created by a validated stage receipt.
"""

from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from . import hu_m31_t3_step6d_full100_wave_package_v2 as package_v2
from . import hu_m31_t3_step6d_full100_wave_plan_v2 as wave_v2


CONTENT_STAGE_PLAN_SCHEMA = "hu_m31_t3_step6d_full100_wave_content_stage_plan_v2"
CONTENT_PREFLIGHT_SCHEMA = "hu_m31_t3_step6d_full100_wave_content_preflight_v2"
CONTENT_STAGE_RECEIPT_SCHEMA = "hu_m31_t3_step6d_full100_wave_content_stage_receipt_v2"
CONTENT_CLEANUP_RECEIPT_SCHEMA = "hu_m31_t3_step6d_full100_wave_content_cleanup_receipt_v2"

_BUCKET = re.compile(r"^[a-z0-9][a-z0-9._-]{1,221}[a-z0-9]$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_GENERATION = re.compile(r"^[1-9][0-9]*$")
_UTC_SECONDS = re.compile(
    r"^(?:19|20)[0-9]{2}-(?:0[1-9]|1[0-2])-"
    r"(?:0[1-9]|[12][0-9]|3[01])T(?:[01][0-9]|2[0-3]):"
    r"[0-5][0-9]:[0-5][0-9]Z$"
)

_PLAN_ENTRY_KEYS = frozenset(
    {"relative_path", "object_name", "kind", "sha256", "bytes"}
)
_PLAN_KEYS = frozenset(
    {
        "schema", "status", "run_name", "execution_identity_sha256",
        "wave_plan_sha256", "outer_manifest_sha256",
        "content_payload_sha256", "bucket", "content_prefix", "entries",
        "entry_count", "content_create_only", "if_generation_match",
        "cloud_launch_authorized", "current_profile_changed", "plan_sha256",
    }
)
_PREFLIGHT_KEYS = frozenset(
    {
        "schema", "status", "plan_sha256", "run_name",
        "execution_identity_sha256", "wave_plan_sha256",
        "outer_manifest_sha256", "content_payload_sha256", "bucket",
        "content_prefix", "expected_entry_count", "expected_objects_sha256",
        "observed_object_count", "observed_objects", "listing_complete",
        "all_objects_absent", "preexisting_same_bytes_accepted",
        "create_authorized", "observed_at_utc", "current_profile_changed",
        "receipt_sha256",
    }
)
_STAGE_ROW_KEYS = frozenset(
    {"object_name", "generation", "sha256", "bytes"}
)
_STAGE_KEYS = frozenset(
    {
        "schema", "status", "plan_sha256", "preflight_receipt_sha256",
        "run_name", "execution_identity_sha256", "wave_plan_sha256",
        "outer_manifest_sha256", "content_payload_sha256", "bucket",
        "content_prefix", "expected_entry_count", "created_entry_count",
        "rows", "listing_complete", "stage_complete", "content_create_only",
        "if_generation_match", "preexisting_object_accepted",
        "observed_at_utc", "cloud_launch_authorized",
        "current_profile_changed", "receipt_sha256",
    }
)
_CLEANUP_ROW_KEYS = frozenset({"object_name", "generation"})
_CLEANUP_KEYS = frozenset(
    {
        "schema", "status", "plan_sha256", "stage_receipt_sha256",
        "run_name", "execution_identity_sha256", "wave_plan_sha256",
        "outer_manifest_sha256", "content_payload_sha256", "bucket",
        "content_prefix", "owned_created_count", "delete_attempt_count",
        "already_absent_count", "deleted_rows", "exact_generations_only",
        "wildcard_delete_used", "listing_complete", "all_objects_absent",
        "observed_at_utc", "additional_create_authorized",
        "current_profile_changed", "receipt_sha256",
    }
)
_LISTING_KEYS = frozenset({"bucket", "prefix", "complete", "objects"})
_REMOTE_OBJECT_KEYS = frozenset(
    {"bucket", "object_name", "generation", "sha256", "bytes"}
)
_CREATE_RESULT_KEYS = frozenset(
    {
        "bucket", "object_name", "generation", "sha256", "bytes",
        "created", "if_generation_match",
    }
)
_DELETE_RESULT_KEYS = frozenset(
    {"bucket", "object_name", "generation", "deleted", "if_generation_match"}
)


class ContentObjectBackend(Protocol):
    """Minimal injectable backend; implementations may be in-memory fakes."""

    def list_prefix(self, *, bucket: str, prefix: str) -> Mapping[str, Any]: ...

    def create_object_from_file(
        self,
        *,
        bucket: str,
        object_name: str,
        source_path: str,
        sha256: str,
        bytes: int,
        if_generation_match: int,
    ) -> Mapping[str, Any]: ...

    def delete_object(
        self,
        *,
        bucket: str,
        object_name: str,
        if_generation_match: str,
    ) -> Mapping[str, Any]: ...


class ContentStageIncompleteError(RuntimeError):
    """A create-only stage stopped after zero or more owned creations."""

    def __init__(
        self,
        message: str,
        *,
        partial_receipt: Mapping[str, Any] | None,
    ) -> None:
        super().__init__(message)
        self.partial_receipt = (
            None if partial_receipt is None else deepcopy(dict(partial_receipt))
        )


def canonical_bytes(value: Any) -> bytes:
    return package_v2.canonical_bytes(value)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields changed")


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _require_generation(value: Any, label: str) -> str:
    if not isinstance(value, str) or _GENERATION.fullmatch(value) is None:
        raise ValueError(f"{label} is not a positive decimal generation")
    return value


def _require_utc(value: Any, label: str) -> str:
    if not isinstance(value, str) or _UTC_SECONDS.fullmatch(value) is None:
        raise ValueError(f"{label} must be canonical UTC seconds")
    return value


def _require_bucket(value: Any) -> str:
    if not isinstance(value, str) or _BUCKET.fullmatch(value) is None:
        raise ValueError("content bucket is invalid")
    return value


def _safe_relative_path(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith("/")
        or "\\" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("content relative path is unsafe")
    return value


def _safe_prefix(value: Any, expected_payload_sha256: str) -> str:
    expected = f"{package_v2.CONTENT_PREFIX_ROOT}/{expected_payload_sha256}"
    if value != expected or value.startswith("/") or "\\" in value:
        raise ValueError("immutable content prefix changed or is unsafe")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("immutable content prefix is unsafe")
    return value


def _with_digest(core: Mapping[str, Any]) -> dict[str, Any]:
    payload = deepcopy(dict(core))
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _pop_digest(
    value: Mapping[str, Any], *, expected_keys: frozenset[str], label: str
) -> tuple[dict[str, Any], str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    payload = deepcopy(dict(value))
    _exact_keys(payload, expected_keys, label)
    digest = payload.pop("receipt_sha256", None)
    if digest != canonical_sha256(payload):
        raise ValueError(f"{label} digest changed")
    return payload, _require_sha(digest, f"{label} digest")


def _manifest_entries(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != 26:
        raise ValueError("outer package must contain exactly 26 content entries")
    result: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    seen_objects: set[str] = set()
    prefix = _safe_prefix(
        manifest.get("content_prefix"),
        _require_sha(manifest.get("content_payload_sha256"), "content payload"),
    )
    for raw in entries:
        if not isinstance(raw, Mapping):
            raise ValueError("outer content entry is not an object")
        relative = _safe_relative_path(raw.get("relative_path"))
        object_name = raw.get("object_name")
        if object_name != f"{prefix}/{relative}":
            raise ValueError("outer content object escaped its immutable prefix")
        if relative in seen_paths or object_name in seen_objects:
            raise ValueError("outer content entry is duplicated")
        seen_paths.add(relative)
        seen_objects.add(object_name)
        kind = raw.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ValueError("outer content kind is invalid")
        result.append(
            {
                "relative_path": relative,
                "object_name": object_name,
                "kind": kind,
                "sha256": _require_sha(raw.get("sha256"), "outer content entry"),
                "bytes": _require_positive_int(
                    raw.get("bytes"), "outer content entry bytes"
                ),
            }
        )
    return result


def build_content_stage_plan(
    *,
    package_dir: str | Path,
    wave_plan: Mapping[str, Any],
    expected_startup_sha256: str,
    bucket: str,
) -> dict[str, Any]:
    """Validate the local outer package, then freeze its 26 upload objects."""

    plan = wave_v2.validate_wave_plan(wave_plan)
    manifest = package_v2.validate_outer_package(
        package_dir,
        plan,
        expected_startup_sha256=expected_startup_sha256,
    )
    entries = _manifest_entries(manifest)
    core: dict[str, Any] = {
        "schema": CONTENT_STAGE_PLAN_SCHEMA,
        "status": "validated_outer_package_content_stage_not_started",
        "run_name": manifest["run_name"],
        "execution_identity_sha256": manifest["execution_identity_sha256"],
        "wave_plan_sha256": manifest["wave_plan_sha256"],
        "outer_manifest_sha256": manifest["manifest_sha256"],
        "content_payload_sha256": manifest["content_payload_sha256"],
        "bucket": _require_bucket(bucket),
        "content_prefix": manifest["content_prefix"],
        "entries": entries,
        "entry_count": len(entries),
        "content_create_only": True,
        "if_generation_match": 0,
        "cloud_launch_authorized": False,
        "current_profile_changed": False,
    }
    core["plan_sha256"] = canonical_sha256(core)
    return validate_content_stage_plan(manifest, core)


def _validate_stage_plan_self(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("content stage plan must be an object")
    payload = deepcopy(dict(value))
    _exact_keys(payload, _PLAN_KEYS, "content stage plan")
    digest = payload.pop("plan_sha256", None)
    if digest != canonical_sha256(payload):
        raise ValueError("content stage plan digest changed")
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != 26:
        raise ValueError("content stage plan must contain exactly 26 entries")
    content_sha = _require_sha(
        payload.get("content_payload_sha256"), "content payload"
    )
    prefix = _safe_prefix(payload.get("content_prefix"), content_sha)
    checked: list[dict[str, Any]] = []
    paths: set[str] = set()
    objects: set[str] = set()
    for raw in entries:
        if not isinstance(raw, Mapping):
            raise ValueError("content stage entry is not an object")
        row = deepcopy(dict(raw))
        _exact_keys(row, _PLAN_ENTRY_KEYS, "content stage entry")
        relative = _safe_relative_path(row.get("relative_path"))
        object_name = row.get("object_name")
        if (
            object_name != f"{prefix}/{relative}"
            or relative in paths
            or object_name in objects
        ):
            raise ValueError("content stage entry escaped or duplicated its prefix")
        paths.add(relative)
        objects.add(object_name)
        kind = row.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ValueError("content stage entry kind is invalid")
        _require_sha(row.get("sha256"), "content stage entry")
        _require_positive_int(row.get("bytes"), "content stage entry bytes")
        checked.append(row)
    for field in (
        "execution_identity_sha256", "wave_plan_sha256", "outer_manifest_sha256"
    ):
        _require_sha(payload.get(field), f"content stage {field}")
    if (
        payload.get("schema") != CONTENT_STAGE_PLAN_SCHEMA
        or payload.get("status")
        != "validated_outer_package_content_stage_not_started"
        or not isinstance(payload.get("run_name"), str)
        or not payload.get("run_name")
        or payload.get("entry_count") != 26
        or payload.get("content_create_only") is not True
        or payload.get("if_generation_match") != 0
        or payload.get("cloud_launch_authorized") is not False
        or payload.get("current_profile_changed") is not False
    ):
        raise ValueError("content stage plan boundary changed")
    _require_bucket(payload.get("bucket"))
    payload["entries"] = checked
    payload["plan_sha256"] = _require_sha(digest, "content stage plan")
    return payload


def validate_content_stage_plan(
    outer_manifest: Mapping[str, Any], value: Mapping[str, Any]
) -> dict[str, Any]:
    payload = _validate_stage_plan_self(value)
    manifest_entries = _manifest_entries(outer_manifest)
    entries = payload.get("entries")
    if (
        entries != manifest_entries
        or payload.get("run_name") != outer_manifest.get("run_name")
        or payload.get("execution_identity_sha256")
        != outer_manifest.get("execution_identity_sha256")
        or payload.get("wave_plan_sha256") != outer_manifest.get("wave_plan_sha256")
        or payload.get("outer_manifest_sha256")
        != outer_manifest.get("manifest_sha256")
        or payload.get("content_payload_sha256")
        != outer_manifest.get("content_payload_sha256")
        or payload.get("content_prefix") != outer_manifest.get("content_prefix")
    ):
        raise ValueError("content stage plan binding changed")
    return payload


def _validated_context(
    *,
    package_dir: str | Path,
    wave_plan: Mapping[str, Any],
    expected_startup_sha256: str,
    stage_plan: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = wave_v2.validate_wave_plan(wave_plan)
    manifest = package_v2.validate_outer_package(
        package_dir,
        plan,
        expected_startup_sha256=expected_startup_sha256,
    )
    return manifest, validate_content_stage_plan(manifest, stage_plan)


def _read_listing(
    backend: ContentObjectBackend, *, bucket: str, prefix: str
) -> list[dict[str, Any]]:
    raw = backend.list_prefix(bucket=bucket, prefix=prefix)
    if not isinstance(raw, Mapping):
        raise ValueError("content prefix listing must be an object")
    listing = deepcopy(dict(raw))
    _exact_keys(listing, _LISTING_KEYS, "content prefix listing")
    if (
        listing.get("bucket") != bucket
        or listing.get("prefix") != prefix
        or listing.get("complete") is not True
    ):
        raise ValueError("content prefix listing is incomplete or misbound")
    objects = listing.get("objects")
    if not isinstance(objects, list):
        raise ValueError("content prefix listing objects are missing")
    result: list[dict[str, Any]] = []
    names: set[str] = set()
    for raw_row in objects:
        if not isinstance(raw_row, Mapping):
            raise ValueError("remote content object is not an object")
        row = deepcopy(dict(raw_row))
        _exact_keys(row, _REMOTE_OBJECT_KEYS, "remote content object")
        name = row.get("object_name")
        if (
            row.get("bucket") != bucket
            or not isinstance(name, str)
            or not name.startswith(f"{prefix}/")
            or name in names
        ):
            raise ValueError("remote content object escaped or duplicated its prefix")
        names.add(name)
        _require_generation(row.get("generation"), "remote content object")
        _require_sha(row.get("sha256"), "remote content object")
        _require_positive_int(row.get("bytes"), "remote content object bytes")
        result.append(row)
    return result


def _base_binding(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "plan_sha256": plan["plan_sha256"],
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_plan_sha256": plan["wave_plan_sha256"],
        "outer_manifest_sha256": plan["outer_manifest_sha256"],
        "content_payload_sha256": plan["content_payload_sha256"],
        "bucket": plan["bucket"],
        "content_prefix": plan["content_prefix"],
    }


def build_preflight_absence_receipt(
    *,
    package_dir: str | Path,
    wave_plan: Mapping[str, Any],
    expected_startup_sha256: str,
    stage_plan: Mapping[str, Any],
    backend: ContentObjectBackend,
    observed_at_utc: str,
) -> dict[str, Any]:
    _, plan = _validated_context(
        package_dir=package_dir,
        wave_plan=wave_plan,
        expected_startup_sha256=expected_startup_sha256,
        stage_plan=stage_plan,
    )
    observed = _read_listing(
        backend, bucket=plan["bucket"], prefix=plan["content_prefix"]
    )
    if observed:
        raise FileExistsError(
            "immutable content prefix is not empty; existing bytes are never adopted"
        )
    core = {
        "schema": CONTENT_PREFLIGHT_SCHEMA,
        "status": "exact_content_prefix_absence_confirmed_create_only",
        **_base_binding(plan),
        "expected_entry_count": 26,
        "expected_objects_sha256": canonical_sha256(
            [row["object_name"] for row in plan["entries"]]
        ),
        "observed_object_count": 0,
        "observed_objects": [],
        "listing_complete": True,
        "all_objects_absent": True,
        "preexisting_same_bytes_accepted": False,
        "create_authorized": True,
        "observed_at_utc": _require_utc(observed_at_utc, "preflight time"),
        "current_profile_changed": False,
    }
    return validate_preflight_absence_receipt(plan, _with_digest(core))


def validate_preflight_absence_receipt(
    stage_plan: Mapping[str, Any], value: Mapping[str, Any]
) -> dict[str, Any]:
    plan = _validate_stage_plan_self(stage_plan)
    payload, digest = _pop_digest(
        value, expected_keys=_PREFLIGHT_KEYS, label="content preflight receipt"
    )
    expected = _base_binding(plan)
    if any(payload.get(key) != val for key, val in expected.items()):
        raise ValueError("content preflight receipt binding changed")
    if (
        payload.get("schema") != CONTENT_PREFLIGHT_SCHEMA
        or payload.get("status")
        != "exact_content_prefix_absence_confirmed_create_only"
        or payload.get("expected_entry_count") != 26
        or payload.get("expected_objects_sha256")
        != canonical_sha256([row["object_name"] for row in plan["entries"]])
        or payload.get("observed_object_count") != 0
        or payload.get("observed_objects") != []
        or payload.get("listing_complete") is not True
        or payload.get("all_objects_absent") is not True
        or payload.get("preexisting_same_bytes_accepted") is not False
        or payload.get("create_authorized") is not True
        or payload.get("current_profile_changed") is not False
    ):
        raise ValueError("content preflight absence contract changed")
    _require_utc(payload.get("observed_at_utc"), "preflight time")
    return {**payload, "receipt_sha256": digest}


def _validate_creation_result(
    raw: Mapping[str, Any], *, plan: Mapping[str, Any], entry: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("content creation result must be an object")
    result = deepcopy(dict(raw))
    _exact_keys(result, _CREATE_RESULT_KEYS, "content creation result")
    if (
        result.get("bucket") != plan["bucket"]
        or result.get("object_name") != entry["object_name"]
        or result.get("sha256") != entry["sha256"]
        or result.get("bytes") != entry["bytes"]
        or result.get("created") is not True
        or result.get("if_generation_match") != 0
    ):
        raise ValueError("content creation result changed")
    return {
        "object_name": result["object_name"],
        "generation": _require_generation(
            result.get("generation"), "content creation result"
        ),
        "sha256": result["sha256"],
        "bytes": result["bytes"],
    }


def _validate_stage_rows(
    plan: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError("content stage rows are missing")
    expected = {row["object_name"]: row for row in plan["entries"]}
    checked: list[dict[str, Any]] = []
    names: set[str] = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("content stage row is not an object")
        row = deepcopy(dict(raw))
        _exact_keys(row, _STAGE_ROW_KEYS, "content stage row")
        name = row.get("object_name")
        frozen = expected.get(name)
        if frozen is None or name in names:
            raise ValueError("content stage row is extra or duplicated")
        names.add(name)
        _require_generation(row.get("generation"), "content stage row")
        if row.get("sha256") != frozen["sha256"] or row.get("bytes") != frozen["bytes"]:
            raise ValueError("content stage row hash or bytes changed")
        checked.append(row)
    order = {row["object_name"]: i for i, row in enumerate(plan["entries"])}
    if [order[row["object_name"]] for row in checked] != sorted(
        order[row["object_name"]] for row in checked
    ):
        raise ValueError("content stage row order changed")
    return checked


def build_stage_readback_receipt(
    *,
    stage_plan: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
    created_rows: Sequence[Mapping[str, Any]],
    backend: ContentObjectBackend,
    observed_at_utc: str,
) -> dict[str, Any]:
    plan = _validate_stage_plan_self(stage_plan)
    preflight = validate_preflight_absence_receipt(plan, preflight_receipt)
    rows = _validate_stage_rows(plan, created_rows)
    observed = _read_listing(
        backend, bucket=plan["bucket"], prefix=plan["content_prefix"]
    )
    observed_by_name = {row["object_name"]: row for row in observed}
    if len(observed_by_name) != len(observed) or set(observed_by_name) != {
        row["object_name"] for row in rows
    }:
        raise ValueError("content stage readback has extra, missing, or unknown objects")
    for row in rows:
        remote = observed_by_name[row["object_name"]]
        if (
            remote["bucket"] != plan["bucket"]
            or remote["generation"] != row["generation"]
            or remote["sha256"] != row["sha256"]
            or remote["bytes"] != row["bytes"]
        ):
            raise ValueError("content stage readback generation/hash/bytes changed")
    complete = len(rows) == 26
    core = {
        "schema": CONTENT_STAGE_RECEIPT_SCHEMA,
        "status": (
            "all_26_content_objects_create_only_and_readback_verified"
            if complete
            else "partial_content_stage_owned_generations_readback_verified"
        ),
        **_base_binding(plan),
        "preflight_receipt_sha256": preflight["receipt_sha256"],
        "expected_entry_count": 26,
        "created_entry_count": len(rows),
        "rows": rows,
        "listing_complete": True,
        "stage_complete": complete,
        "content_create_only": True,
        "if_generation_match": 0,
        "preexisting_object_accepted": False,
        "observed_at_utc": _require_utc(observed_at_utc, "stage readback time"),
        "cloud_launch_authorized": False,
        "current_profile_changed": False,
    }
    return validate_stage_receipt(plan, preflight, _with_digest(core))


def validate_stage_receipt(
    stage_plan: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
    value: Mapping[str, Any],
) -> dict[str, Any]:
    plan = _validate_stage_plan_self(stage_plan)
    preflight = validate_preflight_absence_receipt(plan, preflight_receipt)
    payload, digest = _pop_digest(
        value, expected_keys=_STAGE_KEYS, label="content stage receipt"
    )
    expected = _base_binding(plan)
    if any(payload.get(key) != val for key, val in expected.items()):
        raise ValueError("content stage receipt binding changed")
    rows = _validate_stage_rows(plan, payload.get("rows"))
    complete = len(rows) == 26
    expected_status = (
        "all_26_content_objects_create_only_and_readback_verified"
        if complete
        else "partial_content_stage_owned_generations_readback_verified"
    )
    if (
        payload.get("schema") != CONTENT_STAGE_RECEIPT_SCHEMA
        or payload.get("status") != expected_status
        or payload.get("preflight_receipt_sha256") != preflight["receipt_sha256"]
        or payload.get("expected_entry_count") != 26
        or payload.get("created_entry_count") != len(rows)
        or payload.get("listing_complete") is not True
        or payload.get("stage_complete") is not complete
        or payload.get("content_create_only") is not True
        or payload.get("if_generation_match") != 0
        or payload.get("preexisting_object_accepted") is not False
        or payload.get("cloud_launch_authorized") is not False
        or payload.get("current_profile_changed") is not False
    ):
        raise ValueError("content stage receipt contract changed")
    _require_utc(payload.get("observed_at_utc"), "stage readback time")
    if payload["observed_at_utc"] < preflight["observed_at_utc"]:
        raise ValueError("content stage readback predates preflight")
    payload["rows"] = rows
    return {**payload, "receipt_sha256": digest}


def execute_content_stage(
    *,
    package_dir: str | Path,
    wave_plan: Mapping[str, Any],
    expected_startup_sha256: str,
    stage_plan: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
    backend: ContentObjectBackend,
    observed_at_utc: str,
) -> dict[str, Any]:
    manifest, plan = _validated_context(
        package_dir=package_dir,
        wave_plan=wave_plan,
        expected_startup_sha256=expected_startup_sha256,
        stage_plan=stage_plan,
    )
    del manifest
    preflight = validate_preflight_absence_receipt(plan, preflight_receipt)
    if _read_listing(backend, bucket=plan["bucket"], prefix=plan["content_prefix"]):
        raise FileExistsError("immutable content prefix changed after preflight")
    root = Path(package_dir)
    created: list[dict[str, Any]] = []
    try:
        for entry in plan["entries"]:
            source = package_v2._safe_file(root, entry["relative_path"])
            if (
                package_v2.sha256_file(source) != entry["sha256"]
                or source.stat().st_size != entry["bytes"]
            ):
                raise ValueError("outer package source changed before upload")
            raw = backend.create_object_from_file(
                bucket=plan["bucket"],
                object_name=entry["object_name"],
                source_path=str(source),
                sha256=entry["sha256"],
                bytes=entry["bytes"],
                if_generation_match=0,
            )
            row = _validate_creation_result(raw, plan=plan, entry=entry)
            created.append(row)
            if (
                package_v2.sha256_file(source) != entry["sha256"]
                or source.stat().st_size != entry["bytes"]
            ):
                raise ValueError("outer package source changed during upload")
    except Exception as exc:
        partial: dict[str, Any] | None = None
        try:
            partial = build_stage_readback_receipt(
                stage_plan=plan,
                preflight_receipt=preflight,
                created_rows=created,
                backend=backend,
                observed_at_utc=observed_at_utc,
            )
        except Exception:
            partial = None
        raise ContentStageIncompleteError(
            "create-only content stage did not complete",
            partial_receipt=partial,
        ) from exc
    return build_stage_readback_receipt(
        stage_plan=plan,
        preflight_receipt=preflight,
        created_rows=created,
        backend=backend,
        observed_at_utc=observed_at_utc,
    )


def _validate_delete_result(
    raw: Mapping[str, Any], *, bucket: str, row: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("content deletion result must be an object")
    result = deepcopy(dict(raw))
    _exact_keys(result, _DELETE_RESULT_KEYS, "content deletion result")
    if (
        result.get("bucket") != bucket
        or result.get("object_name") != row["object_name"]
        or result.get("generation") != row["generation"]
        or result.get("deleted") is not True
        or result.get("if_generation_match") != row["generation"]
    ):
        raise ValueError("content deletion result escaped exact generation")
    return {"object_name": row["object_name"], "generation": row["generation"]}


def cleanup_staged_content(
    *,
    stage_plan: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
    stage_receipt: Mapping[str, Any],
    backend: ContentObjectBackend,
    observed_at_utc: str,
) -> dict[str, Any]:
    plan = _validate_stage_plan_self(stage_plan)
    stage = validate_stage_receipt(plan, preflight_receipt, stage_receipt)
    owned = {row["object_name"]: row for row in stage["rows"]}
    observed = _read_listing(
        backend, bucket=plan["bucket"], prefix=plan["content_prefix"]
    )
    observed_by_name = {row["object_name"]: row for row in observed}
    if not set(observed_by_name).issubset(owned):
        raise ValueError("cleanup prefix contains an unowned object")
    for name, remote in observed_by_name.items():
        frozen = owned[name]
        if (
            remote["generation"] != frozen["generation"]
            or remote["sha256"] != frozen["sha256"]
            or remote["bytes"] != frozen["bytes"]
        ):
            raise ValueError("cleanup object no longer has the owned generation")
    deleted: list[dict[str, Any]] = []
    for row in stage["rows"]:
        if row["object_name"] not in observed_by_name:
            continue
        result = backend.delete_object(
            bucket=plan["bucket"],
            object_name=row["object_name"],
            if_generation_match=row["generation"],
        )
        deleted.append(_validate_delete_result(result, bucket=plan["bucket"], row=row))
    remaining = _read_listing(
        backend, bucket=plan["bucket"], prefix=plan["content_prefix"]
    )
    if remaining:
        raise ValueError("content cleanup absence readback is not empty")
    core = {
        "schema": CONTENT_CLEANUP_RECEIPT_SCHEMA,
        "status": "exact_owned_generations_deleted_content_prefix_absent",
        **_base_binding(plan),
        "stage_receipt_sha256": stage["receipt_sha256"],
        "owned_created_count": len(stage["rows"]),
        "delete_attempt_count": len(deleted),
        "already_absent_count": len(stage["rows"]) - len(deleted),
        "deleted_rows": deleted,
        "exact_generations_only": True,
        "wildcard_delete_used": False,
        "listing_complete": True,
        "all_objects_absent": True,
        "observed_at_utc": _require_utc(observed_at_utc, "cleanup time"),
        "additional_create_authorized": False,
        "current_profile_changed": False,
    }
    return validate_cleanup_receipt(
        plan, preflight_receipt, stage, _with_digest(core)
    )


def validate_cleanup_receipt(
    stage_plan: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
    stage_receipt: Mapping[str, Any],
    value: Mapping[str, Any],
) -> dict[str, Any]:
    plan = _validate_stage_plan_self(stage_plan)
    stage = validate_stage_receipt(plan, preflight_receipt, stage_receipt)
    payload, digest = _pop_digest(
        value, expected_keys=_CLEANUP_KEYS, label="content cleanup receipt"
    )
    expected = _base_binding(plan)
    if any(payload.get(key) != val for key, val in expected.items()):
        raise ValueError("content cleanup receipt binding changed")
    deleted = payload.get("deleted_rows")
    if not isinstance(deleted, list):
        raise ValueError("content cleanup deleted rows are missing")
    checked: list[dict[str, Any]] = []
    owned = {
        row["object_name"]: row["generation"]
        for row in stage.get("rows", [])
        if isinstance(row, Mapping)
    }
    seen: set[str] = set()
    for raw in deleted:
        if not isinstance(raw, Mapping):
            raise ValueError("content cleanup row is not an object")
        row = deepcopy(dict(raw))
        _exact_keys(row, _CLEANUP_ROW_KEYS, "content cleanup row")
        name = row.get("object_name")
        if name in seen or owned.get(name) != row.get("generation"):
            raise ValueError("content cleanup row is not an exact owned generation")
        seen.add(name)
        _require_generation(row.get("generation"), "content cleanup row")
        checked.append(row)
    owned_count = len(stage.get("rows", []))
    if (
        payload.get("schema") != CONTENT_CLEANUP_RECEIPT_SCHEMA
        or payload.get("status")
        != "exact_owned_generations_deleted_content_prefix_absent"
        or payload.get("stage_receipt_sha256") != stage.get("receipt_sha256")
        or payload.get("owned_created_count") != owned_count
        or payload.get("delete_attempt_count") != len(checked)
        or payload.get("already_absent_count") != owned_count - len(checked)
        or payload.get("exact_generations_only") is not True
        or payload.get("wildcard_delete_used") is not False
        or payload.get("listing_complete") is not True
        or payload.get("all_objects_absent") is not True
        or payload.get("additional_create_authorized") is not False
        or payload.get("current_profile_changed") is not False
    ):
        raise ValueError("content cleanup receipt contract changed")
    _require_utc(payload.get("observed_at_utc"), "cleanup time")
    if payload["observed_at_utc"] < stage["observed_at_utc"]:
        raise ValueError("content cleanup predates stage readback")
    payload["deleted_rows"] = checked
    return {**payload, "receipt_sha256": digest}


__all__ = [
    "CONTENT_CLEANUP_RECEIPT_SCHEMA",
    "CONTENT_PREFLIGHT_SCHEMA",
    "CONTENT_STAGE_PLAN_SCHEMA",
    "CONTENT_STAGE_RECEIPT_SCHEMA",
    "ContentObjectBackend",
    "ContentStageIncompleteError",
    "build_content_stage_plan",
    "build_preflight_absence_receipt",
    "build_stage_readback_receipt",
    "canonical_bytes",
    "canonical_sha256",
    "cleanup_staged_content",
    "execute_content_stage",
    "validate_cleanup_receipt",
    "validate_content_stage_plan",
    "validate_preflight_absence_receipt",
    "validate_stage_receipt",
]
