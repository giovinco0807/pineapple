"""Receive and judge the perfdev-v2 ten-hand tail without cloud mutation.

The cloud collection receipt proves that both remote role trees were observed
and validated, but intentionally does not contain their timing/RSS payloads.
This module closes that boundary in two local-only steps:

* ``receive_pair_local`` copies the already-created GCS result objects through
  the receive-only transport into an immutable local tree.
* ``run_tail_gate`` revalidates every local byte from the receive receipt and
  invokes the frozen Candidate02 tail merger.  The resulting pass/no-go is
  performance-development authorization only; it is never quality, training,
  or runtime-promotion evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from . import (
    hu_m31_t3_step6d_performance_development_v2_cloud_execution_v1 as cloud,
)
from . import hu_m31_t3_step6d_performance_development_v2_contract as contract_v1
from . import merge_hu_m31_t3_step6d_candidate02_tail_v2 as merger
from . import run_hu_m31_t3_step6d_performance_v2 as runner


RECEIVE_SCHEMA = "hu_m31_t3_step6d_perfdev_v2_local_receive_receipt_v2"
RECEIVE_STATUS = "immutable_candidate_reference_pair_received_and_validated"
RECEIPT_NAME = "LOCAL_RECEIVE_RECEIPT.json"
CONTROL_DIR = "control"
SOURCE_DIR = "sources"

_SHA256 = frozenset("0123456789abcdef")
_COLLECTION_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "execution_plan_sha256",
        "authorization_sha256",
        "run_contract_digest",
        "run_contract_schema",
        "run_contract_variant",
        "selection_manifest_sha256",
        "tail_hand_indices",
        "roles",
        "instance_ownership",
        "portable_pair_complete",
        "artifact_validation_passed",
        "partial_result",
        "cleanup_authorized",
        "training_eligible",
        "quality_evidence",
        "promotion_evidence",
        "current_profile_changed",
        "receipt_content_sha256",
    }
)
_COLLECTION_ROLE_KEYS = frozenset(
    {
        "source_role",
        "result_manifest_object",
        "result_manifest_sha256",
        "artifact_count",
        "heartbeat_count",
        "checkpoint_object_count",
        "runner_validation_passed",
    }
)
_LOCAL_ARTIFACT_KEYS = frozenset({"path", "sha256", "bytes"})
_LOCAL_ROLE_KEYS = frozenset(
    {
        "source_role",
        "source_directory",
        "done_path",
        "result_manifest_path",
        "result_manifest_object",
        "result_manifest_sha256",
        "artifact_count",
        "artifacts",
        "heartbeat_count",
        "checkpoint_object_count",
        "runner_validation_passed",
    }
)
_RECEIVE_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "execution_plan_sha256",
        "authorization_sha256",
        "collection_receipt_sha256",
        "collection_receipt_path",
        "authorization_path",
        "source_roles",
        "tail_hand_indices",
        "run_contract_digest",
        "run_contract_schema",
        "run_contract_variant",
        "selection_manifest_sha256",
        "roles",
        "portable_pair_materialized",
        "runner_validation_passed",
        "remote_transport_mode",
        "remote_method_surface",
        "cloud_mutation_count",
        "performance_gate_evaluated",
        "teacher_quality_evaluated",
        "training_eligible",
        "quality_evidence",
        "promotion_evidence",
        "current_profile_changed",
        "receipt_content_sha256",
    }
)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.casefold()
        and set(value) <= _SHA256
    )


def _validate_tail_run_contract_bytes(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("received tail run contract is not canonical JSON") from exc
    if not isinstance(value, Mapping) or raw != cloud.canonical_bytes(value):
        raise ValueError("received tail run contract is not canonical JSON")
    return contract_v1.validate_tail_run_contract(value)


def _read_canonical(path: str | Path, label: str) -> dict[str, Any]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    raw = target.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != cloud.canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _write_once(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _safe_relative(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is invalid")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != value:
        raise ValueError(f"{label} escapes the receive tree")
    return pure


def _safe_local_path(root: Path, relative: Any, label: str) -> Path:
    pure = _safe_relative(relative, label)
    unresolved = root.joinpath(*pure.parts)
    if unresolved.is_symlink():
        raise ValueError(f"{label} is a symlink")
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes the receive tree") from exc
    return resolved


def _load_control_chain(
    *,
    cloud_package_dir: str | Path,
    collection_receipt: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    view = cloud._load_cloud_package(
        cloud_package_dir, now_unix_seconds=None, require_fresh_receipt=False
    )
    collection = cloud._validate_collection_receipt(collection_receipt)
    checked_authorization = cloud._validate_launch_authorization(
        authorization,
        execution_plan=view.plan,
        raw_nonce=None,
        now_unix_seconds=None,
        allow_runtime_nonce_hash_only=True,
    )
    if set(collection) != _COLLECTION_KEYS:
        raise ValueError("collection receipt fields changed")
    if (
        collection.get("run_name") != view.plan["run_name"]
        or collection.get("execution_plan_sha256") != view.plan_sha256
        or collection.get("authorization_sha256")
        != cloud.canonical_sha256(checked_authorization)
        or collection.get("instance_ownership") != view.plan["instances"]
        or collection.get("run_contract_digest")
        != contract_v1.TAIL_RUN_CONTRACT_DIGEST
        or collection.get("run_contract_schema")
        != contract_v1.TAIL_RUN_CONTRACT_SCHEMA
        or collection.get("run_contract_variant")
        != contract_v1.TAIL_RUN_CONTRACT_VARIANT
        or collection.get("selection_manifest_sha256")
        != contract_v1.TAIL_SELECTION_MANIFEST_SHA256
        or collection.get("tail_hand_indices")
        != list(contract_v1.TAIL_HAND_INDICES)
        or collection.get("portable_pair_complete") is not True
        or collection.get("artifact_validation_passed") is not True
        or collection.get("partial_result") is not False
        or collection.get("cleanup_authorized") is not True
        or any(
            collection.get(field) is not False
            for field in (
                "training_eligible",
                "quality_evidence",
                "promotion_evidence",
                "current_profile_changed",
            )
        )
    ):
        raise ValueError("collection receipt is not bound to this frozen pair")
    roles = collection.get("roles")
    if (
        not isinstance(roles, list)
        or len(roles) != len(cloud.SOURCE_ROLES)
        or [row.get("source_role") if isinstance(row, Mapping) else None for row in roles]
        != list(cloud.SOURCE_ROLES)
    ):
        raise ValueError("collection receipt role order/coverage changed")
    expected_artifacts = len(cloud._required_artifact_paths("candidate"))
    for role, raw in zip(cloud.SOURCE_ROLES, roles, strict=True):
        if not isinstance(raw, Mapping) or set(raw) != _COLLECTION_ROLE_KEYS:
            raise ValueError("collection role receipt fields changed")
        expected_manifest = (
            f"{view.plan['result_prefix']}results/{role}/RESULT_MANIFEST.json"
        )
        if (
            raw.get("source_role") != role
            or raw.get("result_manifest_object") != expected_manifest
            or not _is_sha256(raw.get("result_manifest_sha256"))
            or raw.get("artifact_count") != expected_artifacts
            or not isinstance(raw.get("heartbeat_count"), int)
            or isinstance(raw.get("heartbeat_count"), bool)
            or raw["heartbeat_count"] < 1
            or not isinstance(raw.get("checkpoint_object_count"), int)
            or isinstance(raw.get("checkpoint_object_count"), bool)
            or raw["checkpoint_object_count"] < 2
            or raw.get("runner_validation_passed") is not True
        ):
            raise ValueError("collection role receipt escaped complete tail")
    return view, collection, checked_authorization


def _control_record_path(role: str) -> str:
    return f"{CONTROL_DIR}/{role}_RESULT_MANIFEST.json"


def receive_pair_local(
    *,
    cloud_package_dir: str | Path,
    collection_receipt: Mapping[str, Any],
    authorization: Mapping[str, Any],
    output_dir: str | Path,
    transport: cloud.CloudTransport,
) -> dict[str, Any]:
    """Download the complete pair through GET/list only and seal it locally."""

    view, collection, checked_authorization = _load_control_chain(
        cloud_package_dir=cloud_package_dir,
        collection_receipt=collection_receipt,
        authorization=authorization,
    )
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("local receive destination is immutable")
    destination.parent.mkdir(parents=True, exist_ok=True)
    parent = destination.parent.resolve(strict=True)
    stage = parent / f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.staging"
    stage.mkdir()
    try:
        control = stage / CONTROL_DIR
        _write_once(
            control / "collection_receipt.json", cloud.canonical_bytes(collection)
        )
        _write_once(
            control / "launch_authorization.json",
            cloud.canonical_bytes(checked_authorization),
        )
        local_roles: list[dict[str, Any]] = []
        downloaded: dict[str, dict[str, bytes]] = {}
        for role, collection_role in zip(
            cloud.SOURCE_ROLES, collection["roles"], strict=True
        ):
            prefix = f"{view.plan['result_prefix']}results/{role}/"
            manifest_object = f"{prefix}RESULT_MANIFEST.json"
            raw_manifest = transport.get_object(object_name=manifest_object)
            if raw_manifest is None or hashlib.sha256(raw_manifest).hexdigest() != (
                collection_role["result_manifest_sha256"]
            ):
                raise ValueError(f"{role} result manifest differs from collection")
            manifest = cloud._strict_role_result_manifest(
                raw_manifest,
                role=role,
                view=view,
                authorization=checked_authorization,
            )
            expected_objects = sorted(
                [f"{prefix}{row['path']}" for row in manifest["artifacts"]]
                + [manifest_object]
            )
            if sorted(transport.list_objects(prefix=prefix)) != expected_objects:
                raise ValueError(f"{role} result object topology changed")
            heartbeats = sorted(
                transport.list_objects(
                    prefix=f"{view.plan['result_prefix']}heartbeats/{role}/"
                )
            )
            progress = sorted(
                transport.list_objects(
                    prefix=f"{view.plan['result_prefix']}progress/{role}/"
                )
            )
            if (
                len(heartbeats) != collection_role["heartbeat_count"]
                or len(progress) != collection_role["checkpoint_object_count"]
                or not any(name.endswith("run_contract.json") for name in progress)
                or not any(name.endswith("shard_manifest.json") for name in progress)
            ):
                raise ValueError(f"{role} heartbeat/checkpoint evidence changed")

            source_relative = f"{SOURCE_DIR}/{role}"
            source_root = stage / SOURCE_DIR / role
            values: dict[str, bytes] = {}
            artifact_records: list[dict[str, Any]] = []
            for row in manifest["artifacts"]:
                raw = transport.get_object(object_name=f"{prefix}{row['path']}")
                if (
                    raw is None
                    or len(raw) != row["bytes"]
                    or hashlib.sha256(raw).hexdigest() != row["sha256"]
                ):
                    raise ValueError(f"{role} artifact bytes changed: {row['path']}")
                pure = _safe_relative(row["path"], f"{role} artifact path")
                target = source_root.joinpath(*pure.parts)
                _write_once(target, raw)
                values[row["path"]] = raw
                artifact_records.append(
                    {
                        "path": f"{source_relative}/{row['path']}",
                        "sha256": row["sha256"],
                        "bytes": row["bytes"],
                    }
                )
            runner.validate_completed_output(source_root)
            manifest_relative = _control_record_path(role)
            _write_once(stage.joinpath(*PurePosixPath(manifest_relative).parts), raw_manifest)
            local_roles.append(
                {
                    "source_role": role,
                    "source_directory": source_relative,
                    "done_path": f"{source_relative}/DONE.json",
                    "result_manifest_path": manifest_relative,
                    "result_manifest_object": manifest_object,
                    "result_manifest_sha256": collection_role[
                        "result_manifest_sha256"
                    ],
                    "artifact_count": len(artifact_records),
                    "artifacts": artifact_records,
                    "heartbeat_count": len(heartbeats),
                    "checkpoint_object_count": len(progress),
                    "runner_validation_passed": True,
                }
            )
            downloaded[role] = values

        for index in contract_v1.TAIL_HAND_INDICES:
            relative = f"roots/hand_{index:03d}.json"
            if downloaded["candidate"][relative] != downloaded["reference"][relative]:
                raise ValueError("candidate/reference root bytes differ")
        if downloaded["candidate"]["run_contract.json"] != downloaded[
            "reference"
        ]["run_contract.json"]:
            raise ValueError("candidate/reference run contract bytes differ")
        tail_run_contract = _validate_tail_run_contract_bytes(
            downloaded["candidate"]["run_contract.json"]
        )

        unsigned = {
            "schema": RECEIVE_SCHEMA,
            "status": RECEIVE_STATUS,
            "run_name": view.plan["run_name"],
            "execution_plan_sha256": view.plan_sha256,
            "authorization_sha256": cloud.canonical_sha256(checked_authorization),
            "collection_receipt_sha256": cloud.canonical_sha256(collection),
            "collection_receipt_path": f"{CONTROL_DIR}/collection_receipt.json",
            "authorization_path": f"{CONTROL_DIR}/launch_authorization.json",
            "source_roles": list(cloud.SOURCE_ROLES),
            "tail_hand_indices": list(contract_v1.TAIL_HAND_INDICES),
            "run_contract_digest": contract_v1.TAIL_RUN_CONTRACT_DIGEST,
            "run_contract_schema": contract_v1.TAIL_RUN_CONTRACT_SCHEMA,
            "run_contract_variant": contract_v1.TAIL_RUN_CONTRACT_VARIANT,
            "selection_manifest_sha256": tail_run_contract[
                "selection_manifest_sha256"
            ],
            "roles": local_roles,
            "portable_pair_materialized": True,
            "runner_validation_passed": True,
            "remote_transport_mode": "receive",
            "remote_method_surface": ["GET", "LIST"],
            "cloud_mutation_count": 0,
            "performance_gate_evaluated": False,
            "teacher_quality_evaluated": False,
            "training_eligible": False,
            "quality_evidence": False,
            "promotion_evidence": False,
            "current_profile_changed": False,
        }
        receipt = {
            **unsigned,
            "receipt_content_sha256": cloud.canonical_sha256(unsigned),
        }
        _write_once(stage / RECEIPT_NAME, cloud.canonical_bytes(receipt))
        os.replace(stage, destination)
        return receipt
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _validate_receive_receipt(receive_root: Path) -> dict[str, Any]:
    receipt = _read_canonical(receive_root / RECEIPT_NAME, "local receive receipt")
    unsigned = dict(receipt)
    digest = unsigned.pop("receipt_content_sha256", None)
    if (
        set(receipt) != _RECEIVE_KEYS
        or digest != cloud.canonical_sha256(unsigned)
        or receipt.get("schema") != RECEIVE_SCHEMA
        or receipt.get("status") != RECEIVE_STATUS
        or receipt.get("source_roles") != list(cloud.SOURCE_ROLES)
        or receipt.get("tail_hand_indices") != list(contract_v1.TAIL_HAND_INDICES)
        or receipt.get("run_contract_digest")
        != contract_v1.TAIL_RUN_CONTRACT_DIGEST
        or receipt.get("run_contract_schema") != contract_v1.TAIL_RUN_CONTRACT_SCHEMA
        or receipt.get("run_contract_variant")
        != contract_v1.TAIL_RUN_CONTRACT_VARIANT
        or receipt.get("selection_manifest_sha256")
        != contract_v1.TAIL_SELECTION_MANIFEST_SHA256
        or receipt.get("portable_pair_materialized") is not True
        or receipt.get("runner_validation_passed") is not True
        or receipt.get("remote_transport_mode") != "receive"
        or receipt.get("remote_method_surface") != ["GET", "LIST"]
        or receipt.get("cloud_mutation_count") != 0
        or receipt.get("performance_gate_evaluated") is not False
        or receipt.get("teacher_quality_evaluated") is not False
        or any(
            receipt.get(field) is not False
            for field in (
                "training_eligible",
                "quality_evidence",
                "promotion_evidence",
                "current_profile_changed",
            )
        )
    ):
        raise ValueError("local receive receipt changed")
    return receipt


def validate_received_pair(
    *, receive_dir: str | Path, cloud_package_dir: str | Path
) -> tuple[list[Path], list[Path], dict[str, Any]]:
    """Rehash and semantically revalidate the immutable local pair."""

    receive_root = Path(receive_dir)
    if receive_root.is_symlink() or not receive_root.is_dir():
        raise ValueError("local receive directory is missing or unsafe")
    receive_root = receive_root.resolve(strict=True)
    receipt = _validate_receive_receipt(receive_root)
    collection_path = _safe_local_path(
        receive_root, receipt["collection_receipt_path"], "collection receipt path"
    )
    authorization_path = _safe_local_path(
        receive_root, receipt["authorization_path"], "authorization path"
    )
    collection_raw = _read_canonical(collection_path, "stored collection receipt")
    authorization_raw = _read_canonical(
        authorization_path, "stored launch authorization"
    )
    view, collection, authorization = _load_control_chain(
        cloud_package_dir=cloud_package_dir,
        collection_receipt=collection_raw,
        authorization=authorization_raw,
    )
    if (
        receipt["run_name"] != view.plan["run_name"]
        or receipt["execution_plan_sha256"] != view.plan_sha256
        or receipt["authorization_sha256"]
        != cloud.canonical_sha256(authorization)
        or receipt["collection_receipt_sha256"]
        != cloud.canonical_sha256(collection)
    ):
        raise ValueError("local receive control-chain binding changed")

    roles = receipt.get("roles")
    if (
        not isinstance(roles, list)
        or len(roles) != len(cloud.SOURCE_ROLES)
        or [row.get("source_role") if isinstance(row, Mapping) else None for row in roles]
        != list(cloud.SOURCE_ROLES)
    ):
        raise ValueError("local receive role order/coverage changed")
    done_by_role: dict[str, Path] = {}
    bytes_by_role: dict[str, dict[str, bytes]] = {}
    for role, raw_role, collection_role in zip(
        cloud.SOURCE_ROLES, roles, collection["roles"], strict=True
    ):
        if not isinstance(raw_role, Mapping) or set(raw_role) != _LOCAL_ROLE_KEYS:
            raise ValueError("local receive role fields changed")
        source_relative = f"{SOURCE_DIR}/{role}"
        expected_manifest_path = _control_record_path(role)
        if (
            raw_role.get("source_role") != role
            or raw_role.get("source_directory") != source_relative
            or raw_role.get("done_path") != f"{source_relative}/DONE.json"
            or raw_role.get("result_manifest_path") != expected_manifest_path
            or raw_role.get("result_manifest_object")
            != collection_role["result_manifest_object"]
            or raw_role.get("result_manifest_sha256")
            != collection_role["result_manifest_sha256"]
            or raw_role.get("artifact_count") != collection_role["artifact_count"]
            or raw_role.get("heartbeat_count") != collection_role["heartbeat_count"]
            or raw_role.get("checkpoint_object_count")
            != collection_role["checkpoint_object_count"]
            or raw_role.get("runner_validation_passed") is not True
        ):
            raise ValueError("local receive role binding changed")

        manifest_path = _safe_local_path(
            receive_root, raw_role["result_manifest_path"], "result manifest path"
        )
        raw_manifest = manifest_path.read_bytes()
        if hashlib.sha256(raw_manifest).hexdigest() != raw_role[
            "result_manifest_sha256"
        ]:
            raise ValueError("stored result manifest bytes changed")
        manifest = cloud._strict_role_result_manifest(
            raw_manifest, role=role, view=view, authorization=authorization
        )
        expected_local = [
            {
                "path": f"{source_relative}/{row['path']}",
                "sha256": row["sha256"],
                "bytes": row["bytes"],
            }
            for row in manifest["artifacts"]
        ]
        if raw_role.get("artifacts") != expected_local:
            raise ValueError("local receive artifact manifest changed")
        if len(expected_local) != raw_role["artifact_count"]:
            raise ValueError("local receive artifact count changed")

        values: dict[str, bytes] = {}
        for local_record, remote_record in zip(
            expected_local, manifest["artifacts"], strict=True
        ):
            if set(local_record) != _LOCAL_ARTIFACT_KEYS:
                raise ValueError("local receive artifact fields changed")
            path = _safe_local_path(
                receive_root, local_record["path"], "local received artifact"
            )
            if not path.is_file():
                raise ValueError("local received artifact is missing")
            raw = path.read_bytes()
            if (
                len(raw) != local_record["bytes"]
                or hashlib.sha256(raw).hexdigest() != local_record["sha256"]
            ):
                raise ValueError("local received artifact bytes changed")
            values[remote_record["path"]] = raw
        source_root = _safe_local_path(
            receive_root, raw_role["source_directory"], "local source directory"
        )
        if not source_root.is_dir():
            raise ValueError("local source directory is missing")
        expected_tree = {
            str(path.relative_to(source_root)).replace("\\", "/")
            for path in source_root.rglob("*")
            if path.is_file()
        }
        if expected_tree != set(cloud._required_artifact_paths(role)):
            raise ValueError("local received source tree topology changed")
        runner.validate_completed_output(source_root)
        done_path = _safe_local_path(
            receive_root, raw_role["done_path"], "local DONE path"
        )
        done_by_role[role] = done_path
        bytes_by_role[role] = values

    for index in contract_v1.TAIL_HAND_INDICES:
        relative = f"roots/hand_{index:03d}.json"
        if bytes_by_role["candidate"][relative] != bytes_by_role["reference"][
            relative
        ]:
            raise ValueError("stored candidate/reference root bytes differ")
    if bytes_by_role["candidate"]["run_contract.json"] != bytes_by_role[
        "reference"
    ]["run_contract.json"]:
        raise ValueError("stored candidate/reference run contract bytes differ")
    tail_run_contract = _validate_tail_run_contract_bytes(
        bytes_by_role["candidate"]["run_contract.json"]
    )
    if (
        receipt["run_contract_digest"] != runner.canonical_sha256(tail_run_contract)
        or receipt["selection_manifest_sha256"]
        != tail_run_contract["selection_manifest_sha256"]
    ):
        raise ValueError("stored tail run contract receipt binding changed")
    return (
        [done_by_role["candidate"]],
        [done_by_role["reference"]],
        receipt,
    )


def run_tail_gate(
    *,
    receive_dir: str | Path,
    cloud_package_dir: str | Path,
    summary_output_path: str | Path,
    validation_output_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recompute the frozen ten-hand performance gate from local source bytes."""

    candidate_done, reference_done, _receipt = validate_received_pair(
        receive_dir=receive_dir, cloud_package_dir=cloud_package_dir
    )
    return merger.merge_and_validate_candidate02_tail_v2(
        candidate_done_paths=candidate_done,
        reference_done_paths=reference_done,
        summary_output_path=Path(summary_output_path),
        validation_output_path=Path(validation_output_path),
    )


def verify_tail_gate(
    *,
    receive_dir: str | Path,
    cloud_package_dir: str | Path,
    summary_path: str | Path,
    validation_path: str | Path,
) -> dict[str, Any]:
    """Independently replay receive integrity and the stored gate aggregate."""

    validate_received_pair(
        receive_dir=receive_dir, cloud_package_dir=cloud_package_dir
    )
    expected = merger.validate_candidate02_tail_v2_merge(
        summary_path=Path(summary_path), output_path=None
    )
    observed = _read_canonical(validation_path, "tail gate validation")
    if observed != expected:
        raise ValueError("stored tail gate validation differs from source replay")
    return expected


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    receive = sub.add_parser("receive")
    receive.add_argument("--cloud-package", type=Path, required=True)
    receive.add_argument("--collection", type=Path, required=True)
    receive.add_argument("--authorization", type=Path, required=True)
    receive.add_argument("--bucket", required=True)
    receive.add_argument("--output-dir", type=Path, required=True)

    gate = sub.add_parser("gate")
    gate.add_argument("--cloud-package", type=Path, required=True)
    gate.add_argument("--receive-dir", type=Path, required=True)
    gate.add_argument("--summary-output", type=Path, required=True)
    gate.add_argument("--validation-output", type=Path, required=True)

    verify = sub.add_parser("verify")
    verify.add_argument("--cloud-package", type=Path, required=True)
    verify.add_argument("--receive-dir", type=Path, required=True)
    verify.add_argument("--summary", type=Path, required=True)
    verify.add_argument("--validation", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "receive":
        collection = _read_canonical(args.collection, "collection receipt")
        authorization = _read_canonical(args.authorization, "launch authorization")
        view = cloud._load_cloud_package(
            args.cloud_package, now_unix_seconds=None, require_fresh_receipt=False
        )
        transport = cloud.GcpHttpTransport(
            project=view.plan["project"],
            zone=view.plan["zone"],
            bucket=args.bucket,
            mode="receive",
            execution_plan=view.plan,
        )
        receipt = receive_pair_local(
            cloud_package_dir=args.cloud_package,
            collection_receipt=collection,
            authorization=authorization,
            output_dir=args.output_dir,
            transport=transport,
        )
        print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
        return 0
    if args.command == "gate":
        summary, validation = run_tail_gate(
            receive_dir=args.receive_dir,
            cloud_package_dir=args.cloud_package,
            summary_output_path=args.summary_output,
            validation_output_path=args.validation_output,
        )
        print(json.dumps(validation, sort_keys=True, separators=(",", ":")))
        return 0 if summary["status"] == "pass" else 2
    validation = verify_tail_gate(
        receive_dir=args.receive_dir,
        cloud_package_dir=args.cloud_package,
        summary_path=args.summary,
        validation_path=args.validation,
    )
    print(json.dumps(validation, sort_keys=True, separators=(",", ":")))
    return 0 if validation["status"] == "pass" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "RECEIPT_NAME",
    "RECEIVE_SCHEMA",
    "main",
    "receive_pair_local",
    "run_tail_gate",
    "validate_received_pair",
    "verify_tail_gate",
]
