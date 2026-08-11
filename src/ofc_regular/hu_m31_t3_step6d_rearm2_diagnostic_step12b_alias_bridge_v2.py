"""Pure VM-side result bridge for the authoritative Step12b deployment v2.

The nonce-bound ``step12b_deployment_contract_v2`` is the only deployment
authority.  The accepted v1 candidate/reference contracts remain immutable,
local-execution-only payload inputs.  This bridge downloads their accepted
outer package at exact generations, delegates old validation/execution to one
controlled same-process runtime, and publishes only to the deployment's
external direct-v2 result namespace.

The module is backend-neutral and performs no cloud operation at import time.
It does not call the old authorized-worker wrapper or old result publisher.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol, Sequence

from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as payload_transport,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_deployment_contract_v2
    as deployment_v2,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_pair_release_v2
    as pair_release_v2,
)


PUBLISH_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_direct_v2_publish_receipt_v1"
)
EXECUTION_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_alias_bridge_execution_receipt_v1"
)
PAYLOAD_RUN_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_same_process_payload_run_receipt_v1"
)

TREE_OBJECT_COUNT = 23
UPLOAD_COUNT = 10
HEARTBEAT_COUNT = 10
DONE_COUNT = 1
RESULT_OBJECT_COUNT = (
    TREE_OBJECT_COUNT + UPLOAD_COUNT + HEARTBEAT_COUNT + DONE_COUNT
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _safe_relative(value: Any) -> str:
    path = PurePosixPath(value) if isinstance(value, str) else None
    if (
        not isinstance(value, str)
        or not value
        or path is None
        or path.is_absolute()
        or "\\" in value
        or ":" in value
        or any(part in ("", ".", "..") for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError("Step12b bridge path escaped its root")
    return value


def _validated_inputs(
    *,
    deployment_contract: Mapping[str, Any],
    selected_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
    execution_job_id: str,
    package_generations: Mapping[str, int],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
]:
    payload = payload_transport.validate_job_contract(
        selected_payload_contract
    )
    deployment = deployment_v2.validate_role_runtime_view(
        deployment_contract,
        selected_payload_contract=payload,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
        external_job_id=execution_job_id,
    )
    inventory = payload["remote_layout"]["package_inventory"]
    if (
        deployment["remote_layout"]["package_prefix"]
        != inventory["package_prefix"]
        or deployment["payload_binding"][
            "package_inventory_records_sha256"
        ]
        != inventory["records_sha256"]
    ):
        raise ValueError("Step12b immutable package binding changed")
    expected = {row["uri"]: row for row in inventory["records"]}
    if (
        not isinstance(package_generations, Mapping)
        or len(expected) != 16
        or set(package_generations) != set(expected)
    ):
        raise ValueError("Step12b package generation set changed")
    pinned: list[dict[str, Any]] = []
    for inventory_record in inventory["records"]:
        uri = inventory_record["uri"]
        generation = package_generations[uri]
        if (
            type(generation) is not int
            or generation <= 0
        ):
            raise ValueError("Step12b package generation changed")
        pinned.append(
            {
                "path": _safe_relative(inventory_record["path"]),
                "uri": uri,
                "generation": generation,
                "sha256": inventory_record["sha256"],
                "bytes": inventory_record["bytes"],
            }
        )
    binding = _payload_binding(deployment, execution_job_id)
    if (
        payload["metadata_binding"]["job_id"]
        != binding["inner_job_id"]
        or payload["metadata_binding"]["source_role"]
        != binding["source_role"]
        or canonical_sha256(payload)
        != binding["payload_contract_sha256"]
    ):
        raise ValueError("Step12b selected immutable payload changed")
    return deployment, payload, pinned


def _external_job(
    deployment: Mapping[str, Any], execution_job_id: str
) -> dict[str, Any]:
    matches = [
        row
        for row in deployment["remote_layout"]["jobs"]
        if row["job_id"] == execution_job_id
    ]
    if len(matches) != 1:
        raise ValueError("Step12b execution job escaped exact pair")
    return matches[0]


def _payload_binding(
    deployment: Mapping[str, Any], execution_job_id: str
) -> dict[str, Any]:
    external = _external_job(deployment, execution_job_id)
    matches = [
        row
        for row in deployment["payload_binding"]["job_bindings"]
        if row["inner_job_id"] == external["inner_job_id"]
        and row["source_role"] == external["source_role"]
    ]
    if len(matches) != 1:
        raise ValueError("Step12b payload alias mapping changed")
    return matches[0]


def _instance(
    deployment: Mapping[str, Any], execution_job_id: str
) -> dict[str, Any]:
    matches = [
        row
        for row in deployment["instances"]
        if row["job_id"] == execution_job_id
    ]
    if len(matches) != 1:
        raise ValueError("Step12b instance alias mapping changed")
    return matches[0]


def _old_write_prefixes(
    payload: Mapping[str, Any],
) -> list[str]:
    inner_job_id = payload["metadata_binding"]["job_id"]
    preview_job = next(
        row
        for row in payload["adapter_preview"]["jobs"]
        if row["job_id"] == inner_job_id
    )
    legacy_job_prefix = str(preview_job["tree_prefix"]).removesuffix(
        "/tree"
    )
    values = [
        legacy_job_prefix,
        payload["remote_layout"]["stage_prefix"],
        payload["remote_layout"]["result_prefix"],
    ]
    if len(set(values)) != len(values):
        raise ValueError("Step12b old result-prefix audit changed")
    return values


class ImmutablePackageReader(Protocol):
    def generation_pinned_get(
        self, *, uri: str, generation: int
    ) -> bytes: ...


class ConditionalResultWriter(Protocol):
    def conditional_create(
        self, *, uri: str, content: bytes
    ) -> Mapping[str, Any]: ...


class PayloadRuntime(Protocol):
    def validate_and_run(
        self,
        *,
        payload_contract: Mapping[str, Any],
        deployment_contract: Mapping[str, Any],
        execution_job_id: str,
        outer_root: Path,
        work_root: Path,
    ) -> tuple[Path, Mapping[str, Any]]: ...


class ExactSelfDeleter(Protocol):
    def request_exact_self_delete(
        self,
        *,
        project: str,
        zone: str,
        instance_name: str,
        done_readback: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


def download_immutable_outer(
    *,
    pinned_records: Sequence[Mapping[str, Any]],
    reader: ImmutablePackageReader,
    destination: str | Path,
) -> dict[str, Any]:
    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Step12b outer destination must be fresh")
    if not target.parent.is_dir() or target.parent.is_symlink():
        raise ValueError("Step12b outer parent must be a real directory")
    if len(pinned_records) != 16:
        raise ValueError("Step12b immutable package count changed")
    target.mkdir()
    receipts: list[dict[str, Any]] = []
    for record in pinned_records:
        raw = reader.generation_pinned_get(
            uri=record["uri"], generation=record["generation"]
        )
        if (
            not isinstance(raw, bytes)
            or len(raw) != record["bytes"]
            or hashlib.sha256(raw).hexdigest()
            != record["sha256"]
        ):
            raise ValueError(
                "Step12b downloaded immutable package object changed"
            )
        local = target / PurePosixPath(record["path"])
        local.parent.mkdir(parents=True, exist_ok=True)
        with local.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        receipts.append(copy.deepcopy(dict(record)))
    return {
        "status": "immutable_outer_generation_pinned_download_complete",
        "object_count": len(receipts),
        "records": receipts,
        "records_sha256": canonical_sha256(receipts),
        "generation_pinned_get_only": True,
        "package_write_count": 0,
    }


def _output_file(
    root: Path, *, relative: str, uri: str
) -> dict[str, Any]:
    path = root / PurePosixPath(relative)
    if not path.is_file() or path.is_symlink():
        raise ValueError("Step12b payload output file is missing")
    raw = path.read_bytes()
    return {
        "path": relative,
        "uri": uri,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "content": raw,
    }


def _conditional_create(
    writer: ConditionalResultWriter,
    *,
    deployment_stage_prefix: str,
    forbidden_prefixes: Sequence[str],
    uri: str,
    content: bytes,
) -> dict[str, Any]:
    if (
        not uri.startswith(
            deployment_stage_prefix.rstrip("/") + "/"
        )
        or any(uri.startswith(prefix) for prefix in forbidden_prefixes)
    ):
        raise ValueError("Step12b result write escaped direct-v2")
    observed = dict(
        writer.conditional_create(uri=uri, content=content)
    )
    if (
        observed.get("uri") != uri
        or type(observed.get("generation")) is not int
        or observed["generation"] <= 0
        or observed.get("sha256")
        != hashlib.sha256(content).hexdigest()
        or observed.get("bytes") != len(content)
        or observed.get("created") is not True
    ):
        raise RuntimeError(
            "Step12b conditional-create readback changed"
        )
    return {
        "uri": uri,
        "generation": observed["generation"],
        "sha256": observed["sha256"],
        "bytes": observed["bytes"],
        "created": True,
    }


def publish_direct_v2_output(
    *,
    deployment_contract: Mapping[str, Any],
    execution_job_id: str,
    payload_contract: Mapping[str, Any],
    output_dir: str | Path,
    writer: ConditionalResultWriter,
) -> dict[str, Any]:
    deployment = dict(deployment_contract)
    job = _external_job(deployment, execution_job_id)
    binding = _payload_binding(deployment, execution_job_id)
    if (
        canonical_sha256(payload_contract)
        != binding["payload_contract_sha256"]
        or payload_contract["metadata_binding"]["job_id"]
        != binding["inner_job_id"]
    ):
        raise ValueError("Step12b publisher payload binding changed")
    output = Path(output_dir)
    if not output.is_dir() or output.is_symlink():
        raise ValueError("Step12b payload output root changed")
    tree_paths = list(binding["tree_paths"])
    if len(tree_paths) != TREE_OBJECT_COUNT:
        raise ValueError("Step12b tree path count changed")
    observed_paths = sorted(
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    if sorted(tree_paths) != observed_paths:
        raise ValueError("Step12b payload output tree changed")
    files = {
        relative: _output_file(
            output,
            relative=relative,
            uri=uri,
        )
        for relative, uri in zip(
            tree_paths, job["tree_object_uris"], strict=True
        )
    }
    forbidden = _old_write_prefixes(payload_contract)
    stage_prefix = deployment["remote_layout"]["stage_prefix"]
    all_readbacks: list[dict[str, Any]] = []
    tree_readbacks: list[dict[str, Any]] = []
    uploads: list[dict[str, Any]] = []
    heartbeats: list[dict[str, Any]] = []
    for relative in ("run_contract.json", "shard_manifest.json"):
        file = files[relative]
        readback = _conditional_create(
            writer,
            deployment_stage_prefix=stage_prefix,
            forbidden_prefixes=forbidden,
            uri=file["uri"],
            content=file["content"],
        )
        all_readbacks.append(readback)
        tree_readbacks.append({**readback, "path": relative})
    role = job["source_role"]
    for sequence, hand_index in enumerate(
        binding["work_hand_indices"], 1
    ):
        hand_tree: list[dict[str, Any]] = []
        for relative in (
            f"roots/hand_{hand_index:03d}.json",
            f"hands/{role}/hand_{hand_index:03d}.json",
        ):
            file = files[relative]
            readback = _conditional_create(
                writer,
                deployment_stage_prefix=stage_prefix,
                forbidden_prefixes=forbidden,
                uri=file["uri"],
                content=file["content"],
            )
            all_readbacks.append(readback)
            tree = {**readback, "path": relative}
            tree_readbacks.append(tree)
            hand_tree.append(tree)
        upload_body = {
            "schema": f"{PUBLISH_RECEIPT_SCHEMA}_upload_v1",
            "deployment_contract_sha256": deployment[
                "deployment_contract_sha256"
            ],
            "direct_stage_identity_sha256": deployment[
                "direct_stage_identity_sha256"
            ],
            "execution_job_id": execution_job_id,
            "inner_job_id": binding["inner_job_id"],
            "source_role": role,
            "sequence": sequence,
            "hand_index": hand_index,
            "tree_objects": hand_tree,
            "direct_v2_only": True,
        }
        upload = {
            **upload_body,
            "upload_sha256": canonical_sha256(upload_body),
        }
        upload_readback = _conditional_create(
            writer,
            deployment_stage_prefix=stage_prefix,
            forbidden_prefixes=forbidden,
            uri=job["upload_uris"][sequence - 1],
            content=canonical_bytes(upload),
        )
        all_readbacks.append(upload_readback)
        uploads.append(
            {
                "sequence": sequence,
                "hand_index": hand_index,
                "upload_sha256": upload["upload_sha256"],
                "readback": upload_readback,
            }
        )
        heartbeat_body = {
            "schema": f"{PUBLISH_RECEIPT_SCHEMA}_heartbeat_v1",
            "deployment_contract_sha256": deployment[
                "deployment_contract_sha256"
            ],
            "execution_job_id": execution_job_id,
            "source_role": role,
            "sequence": sequence,
            "completed_hand_indices": binding[
                "work_hand_indices"
            ][:sequence],
            "upload_sha256s": [
                row["upload_sha256"] for row in uploads
            ],
            "direct_v2_only": True,
        }
        heartbeat = {
            **heartbeat_body,
            "heartbeat_sha256": canonical_sha256(heartbeat_body),
        }
        heartbeat_readback = _conditional_create(
            writer,
            deployment_stage_prefix=stage_prefix,
            forbidden_prefixes=forbidden,
            uri=job["heartbeat_uris"][sequence - 1],
            content=canonical_bytes(heartbeat),
        )
        all_readbacks.append(heartbeat_readback)
        heartbeats.append(
            {
                "sequence": sequence,
                "heartbeat_sha256": heartbeat[
                    "heartbeat_sha256"
                ],
                "readback": heartbeat_readback,
            }
        )
    done_file = files["DONE.json"]
    done_tree_readback = _conditional_create(
        writer,
        deployment_stage_prefix=stage_prefix,
        forbidden_prefixes=forbidden,
        uri=done_file["uri"],
        content=done_file["content"],
    )
    all_readbacks.append(done_tree_readback)
    tree_readbacks.append(
        {**done_tree_readback, "path": "DONE.json"}
    )
    if (
        len(tree_readbacks) != TREE_OBJECT_COUNT
        or [row["path"] for row in tree_readbacks] != tree_paths
        or len(uploads) != UPLOAD_COUNT
        or len(heartbeats) != HEARTBEAT_COUNT
    ):
        raise RuntimeError("Step12b publication count changed")
    done_body = {
        "schema": f"{PUBLISH_RECEIPT_SCHEMA}_done_v1",
        "deployment_contract_sha256": deployment[
            "deployment_contract_sha256"
        ],
        "direct_stage_identity_sha256": deployment[
            "direct_stage_identity_sha256"
        ],
        "run_name": deployment["run_name"],
        "stage_id": deployment["stage_id"],
        "execution_job_id": execution_job_id,
        "inner_job_id": binding["inner_job_id"],
        "source_role": role,
        "attempt_index": 0,
        "completed_hand_indices": list(
            binding["work_hand_indices"]
        ),
        "tree_object_records": tree_readbacks,
        "tree_object_records_sha256": canonical_sha256(
            tree_readbacks
        ),
        "upload_sha256s": [
            row["upload_sha256"] for row in uploads
        ],
        "heartbeat_sha256s": [
            row["heartbeat_sha256"] for row in heartbeats
        ],
        "tree_object_count": len(tree_readbacks),
        "upload_count": len(uploads),
        "heartbeat_count": len(heartbeats),
        "old_result_write_count": 0,
        "direct_v2_only": True,
    }
    done = {**done_body, "done_sha256": canonical_sha256(done_body)}
    done_readback = _conditional_create(
        writer,
        deployment_stage_prefix=stage_prefix,
        forbidden_prefixes=forbidden,
        uri=job["done_uri"],
        content=canonical_bytes(done),
    )
    all_readbacks.append(done_readback)
    if len(all_readbacks) != RESULT_OBJECT_COUNT:
        raise RuntimeError("Step12b result-object count changed")
    body = {
        "schema": PUBLISH_RECEIPT_SCHEMA,
        "status": "direct_v2_done_readback_complete",
        "deployment_contract_sha256": deployment[
            "deployment_contract_sha256"
        ],
        "execution_job_id": execution_job_id,
        "inner_job_id": binding["inner_job_id"],
        "source_role": role,
        "stage_prefix": stage_prefix,
        "result_prefix": job["result_prefix"],
        "tree_object_count": len(tree_readbacks),
        "upload_count": len(uploads),
        "heartbeat_count": len(heartbeats),
        "done_count": 1,
        "result_object_count": len(all_readbacks),
        "readbacks": all_readbacks,
        "readbacks_sha256": canonical_sha256(all_readbacks),
        "done": done,
        "done_readback": done_readback,
        "done_published_last": True,
        "old_result_write_count": 0,
        "external_direct_v2_only": True,
        "old_publisher_called": False,
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def build_payload_run_receipt(
    *,
    deployment_contract: Mapping[str, Any],
    execution_job_id: str,
) -> dict[str, Any]:
    binding = _payload_binding(
        deployment_contract, execution_job_id
    )
    body = {
        "schema": PAYLOAD_RUN_RECEIPT_SCHEMA,
        "deployment_contract_sha256": deployment_contract[
            "deployment_contract_sha256"
        ],
        "execution_job_id": execution_job_id,
        "payload_contract_sha256": binding[
            "payload_contract_sha256"
        ],
        "old_validate_job_contract_called": True,
        "run_downloaded_worker_called": True,
        "same_process_controlled_context": True,
        "old_execute_authorized_worker_called": False,
        "old_publisher_called": False,
        "old_result_write_count": 0,
        "output_tree_validated": True,
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def _validate_payload_run_receipt(
    value: Mapping[str, Any],
    *,
    deployment: Mapping[str, Any],
    execution_job_id: str,
) -> dict[str, Any]:
    checked = copy.deepcopy(dict(value))
    digest = checked.pop("receipt_sha256", None)
    binding = _payload_binding(deployment, execution_job_id)
    if (
        digest != canonical_sha256(checked)
        or value.get("schema") != PAYLOAD_RUN_RECEIPT_SCHEMA
        or value.get("deployment_contract_sha256")
        != deployment["deployment_contract_sha256"]
        or value.get("execution_job_id") != execution_job_id
        or value.get("payload_contract_sha256")
        != binding["payload_contract_sha256"]
        or value.get("old_validate_job_contract_called") is not True
        or value.get("run_downloaded_worker_called") is not True
        or value.get("same_process_controlled_context") is not True
        or value.get("old_execute_authorized_worker_called") is not False
        or value.get("old_publisher_called") is not False
        or value.get("old_result_write_count") != 0
        or value.get("output_tree_validated") is not True
    ):
        raise ValueError("Step12b payload-run receipt changed")
    return copy.deepcopy(dict(value))


def execute_alias_bridge(
    *,
    deployment_contract: Mapping[str, Any],
    selected_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
    package_generations: Mapping[str, int],
    execution_job_id: str,
    role_runtime_approval: pair_release_v2.ValidatedRoleRuntimeApproval,
    pair_release_approval: pair_release_v2.PairReleaseApproval,
    package_reader: ImmutablePackageReader,
    payload_runtime: PayloadRuntime,
    result_writer: ConditionalResultWriter,
    self_deleter: ExactSelfDeleter,
    fresh_root: str | Path,
) -> dict[str, Any]:
    """Run the pure bridge after the separate metadata/signature gate."""

    deployment, payload, pinned = _validated_inputs(
        deployment_contract=deployment_contract,
        selected_payload_contract=selected_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
        execution_job_id=execution_job_id,
        package_generations=package_generations,
    )
    role_approval = pair_release_v2.require_role_runtime_approval(
        role_runtime_approval,
        deployment_contract_sha256=deployment[
            "deployment_contract_sha256"
        ],
        external_job_id=execution_job_id,
    )
    release_approval = pair_release_v2.require_pair_release_approval(
        pair_release_approval,
        deployment_contract_sha256=deployment[
            "deployment_contract_sha256"
        ],
        external_job_id=execution_job_id,
        role_approval=role_approval,
    )
    instance = _instance(deployment, execution_job_id)
    root = Path(fresh_root)
    if root.exists() or root.is_symlink():
        raise FileExistsError("Step12b bridge root must be fresh")
    if not root.parent.is_dir() or root.parent.is_symlink():
        raise ValueError("Step12b bridge parent must be a real directory")
    root.mkdir()
    package_receipt = download_immutable_outer(
        pinned_records=pinned,
        reader=package_reader,
        destination=root / "outer",
    )
    output_dir, raw_run_receipt = payload_runtime.validate_and_run(
        payload_contract=payload,
        deployment_contract=deployment,
        execution_job_id=execution_job_id,
        outer_root=root / "outer",
        work_root=root / "work",
    )
    run_receipt = _validate_payload_run_receipt(
        raw_run_receipt,
        deployment=deployment,
        execution_job_id=execution_job_id,
    )
    publish_receipt = publish_direct_v2_output(
        deployment_contract=deployment,
        execution_job_id=execution_job_id,
        payload_contract=payload,
        output_dir=output_dir,
        writer=result_writer,
    )
    deletion = dict(
        self_deleter.request_exact_self_delete(
            project=payload_transport.PROJECT,
            zone=payload_transport.ZONE,
            instance_name=instance["instance_name"],
            done_readback=publish_receipt["done_readback"],
        )
    )
    if (
        deletion.get("project") != payload_transport.PROJECT
        or deletion.get("zone") != payload_transport.ZONE
        or deletion.get("instance_name") != instance["instance_name"]
        or deletion.get("delete_requested") is not True
        or deletion.get("done_readback_sha256")
        != canonical_sha256(publish_receipt["done_readback"])
    ):
        raise RuntimeError("Step12b exact self-delete receipt changed")
    body = {
        "schema": EXECUTION_RECEIPT_SCHEMA,
        "status": "direct_v2_done_readback_then_exact_self_delete",
        "deployment_contract_sha256": deployment[
            "deployment_contract_sha256"
        ],
        "execution_job_id": execution_job_id,
        "inner_job_id": payload["metadata_binding"]["job_id"],
        "source_role": payload["metadata_binding"]["source_role"],
        "instance_name": instance["instance_name"],
        "role_runtime_approval_receipt_sha256": (
            role_approval.receipt_sha256
        ),
        "pair_release_approval_receipt_sha256": (
            release_approval.receipt_sha256
        ),
        "pair_release_sha256": release_approval.pair_release_sha256,
        "pair_release_verified": True,
        "package_download_receipt": package_receipt,
        "payload_run_receipt": run_receipt,
        "publish_receipt": publish_receipt,
        "self_delete_receipt": deletion,
        "tree_object_count": TREE_OBJECT_COUNT,
        "upload_count": UPLOAD_COUNT,
        "heartbeat_count": HEARTBEAT_COUNT,
        "done_count": DONE_COUNT,
        "result_object_count": RESULT_OBJECT_COUNT,
        "old_result_write_count": 0,
        "old_execute_authorized_worker_called": False,
        "old_publisher_called": False,
        "external_direct_v2_only": True,
        "current_profile_changed": False,
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


__all__ = [
    "DONE_COUNT",
    "EXECUTION_RECEIPT_SCHEMA",
    "HEARTBEAT_COUNT",
    "PAYLOAD_RUN_RECEIPT_SCHEMA",
    "PUBLISH_RECEIPT_SCHEMA",
    "RESULT_OBJECT_COUNT",
    "TREE_OBJECT_COUNT",
    "UPLOAD_COUNT",
    "build_payload_run_receipt",
    "canonical_bytes",
    "canonical_sha256",
    "download_immutable_outer",
    "execute_alias_bridge",
    "publish_direct_v2_output",
]
