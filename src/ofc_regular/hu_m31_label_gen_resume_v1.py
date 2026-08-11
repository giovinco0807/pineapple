"""Pure resume and completion-contract helpers for GCS label shards.

Spot replacement starts with an empty boot disk.  A replacement therefore has
to restore every create-only position object before workers inspect the shard;
remembering only the object names is not enough because the final completion
inventory is built from the local shard directory.  The helpers in this module
keep that restore validation and the one authoritative complete checkpoint
small enough to exercise without a cloud credential.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
from typing import Any, Iterable, Mapping

from .hu_m31_label_gen_worker_v1 import (
    POSITION_SCHEMA,
    T0_VS_FL_KIND,
    T0_VS_FL_POSITION_SCHEMA,
    T1_VS_FL_KIND,
    T1_VS_FL_POSITION_SCHEMA,
    T2_VS_FL_KIND,
    T2_VS_FL_POSITION_SCHEMA,
    T3_VS_FL_KIND,
    T3_VS_FL_POSITION_SCHEMA,
    canonical_bytes,
)


COMPLETE_CHECKPOINT_SCHEMA = "hu_m31_label_gen_complete_checkpoint_v1"
COMPLETE_CHECKPOINT_BASENAME = "complete.json"


def expected_position_schema(plan: Mapping[str, Any]) -> str:
    """Return the only position schema valid for *plan*."""

    kind = plan.get("plan_kind")
    if kind is None:
        return POSITION_SCHEMA
    schemas = {
        T3_VS_FL_KIND: T3_VS_FL_POSITION_SCHEMA,
        T2_VS_FL_KIND: T2_VS_FL_POSITION_SCHEMA,
        T1_VS_FL_KIND: T1_VS_FL_POSITION_SCHEMA,
        T0_VS_FL_KIND: T0_VS_FL_POSITION_SCHEMA,
    }
    try:
        return schemas[kind]
    except KeyError as error:
        raise ValueError(f"unsupported cached-position plan_kind {kind!r}") from error


def validate_cached_position_object(
    raw: bytes,
    *,
    object_name: str,
    expected_object_name: str,
    generation: str,
    expected_bytes: int,
    expected_sha256: str,
    plan_sha256: str,
    offset: int,
    position_schema: str,
) -> str:
    """Validate one generation-pinned GCS position before caching it locally.

    ``expected_sha256`` is the immutable custom metadata written with the
    object.  Size and SHA are checked before JSON provenance so a truncated or
    substituted object never reaches a worker as an already-complete position.
    The returned digest is suitable for the final checkpoint inventory.
    """

    if object_name != expected_object_name:
        raise ValueError(
            f"cached object name drifted: {object_name!r} != {expected_object_name!r}"
        )
    if not isinstance(generation, str) or not generation.isdigit() or int(generation) <= 0:
        raise ValueError(f"cached object has invalid generation {generation!r}")
    if type(expected_bytes) is not int or expected_bytes <= 0:
        raise ValueError(f"cached object has invalid byte count {expected_bytes!r}")
    if len(raw) != expected_bytes:
        raise ValueError(
            f"cached object byte count drifted: {len(raw)} != {expected_bytes}"
        )
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in expected_sha256)
    ):
        raise ValueError("cached object has no canonical sha256 metadata")
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"cached object sha256 drifted: {actual_sha256} != {expected_sha256}"
        )
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("cached position is not JSON") from error
    if raw != canonical_bytes(payload):
        raise ValueError("cached position is not canonical JSON")
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != position_schema
        or payload.get("plan_sha256") != plan_sha256
        or type(payload.get("offset")) is not int
        or payload.get("offset") != offset
    ):
        raise ValueError(
            "cached position provenance drifted "
            f"for offset {offset}: schema/plan_sha256/offset mismatch"
        )
    return actual_sha256


def restore_cached_position_object(
    destination: pathlib.Path,
    raw: bytes,
    **validation: Any,
) -> str:
    """Validate and atomically create one local resume file.

    An already-present local file is accepted only byte-for-byte.  This covers
    package-provided resume directories without letting them override the
    generation-pinned object selected from GCS.
    """

    digest = validate_cached_position_object(raw, **validation)
    if destination.exists():
        if destination.read_bytes() != raw:
            raise ValueError(
                f"local resume copy disagrees with {validation['expected_object_name']}"
            )
        return digest
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return digest


def complete_checkpoint_object_name(object_prefix: str) -> str:
    return f"{object_prefix}/checkpoints/{COMPLETE_CHECKPOINT_BASENAME}"


def build_complete_checkpoint(
    *,
    plan_sha256: str,
    shard_id: str,
    attempt_id: str,
    shard_start: int,
    shard_count: int,
    files: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the sole completion witness for a shard.

    Progress is represented by heartbeats and immutable position objects.  No
    incomplete payload is ever written to ``complete.json``, so a full local
    count observed milliseconds before ``SHARD_DONE`` cannot poison the final
    create-only checkpoint name.
    """

    rows = [dict(row) for row in files]
    expected_relatives = {"SHARD_DONE.json"} | {
        f"position_{offset:08d}.json"
        for offset in range(shard_start, shard_start + shard_count)
    }
    relatives = [row.get("relative_path") for row in rows]
    if len(relatives) != len(set(relatives)) or set(relatives) != expected_relatives:
        raise ValueError("complete checkpoint does not inventory the whole shard")
    payload = {
        "schema": COMPLETE_CHECKPOINT_SCHEMA,
        "checkpoint_kind": "complete",
        "plan_sha256": plan_sha256,
        "shard_id": shard_id,
        "attempt_id": attempt_id,
        "completed_position_count": shard_count,
        "complete": True,
        "files": rows,
        "checkpoint_published_after_files": True,
        "create_only": True,
    }
    payload["checkpoint_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return payload
