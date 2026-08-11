"""Independent generation-pinned receiver for Step12b direct-v2 results."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Protocol, Sequence

from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as payload_transport,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_alias_bridge_v2
    as alias_bridge,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_deployment_contract_v2
    as deployment_v2,
)
from . import run_hu_m31_t3_step6d_performance_v2 as runner


SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_generation_pinned_result_receiver_v2"
)
PAIR_SCHEMA = f"{SCHEMA}_pair_v1"
MAX_DONE_WAIT_SECONDS = 4_500
POLL_SECONDS = 5.0
WORKER_FAILURE_MARKER_PREFIX = "OFC_STEP12N_WORKER_FAILURE_V1 "
WORKER_FAILURE_MARKER_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_worker_failure_marker_v1"
)
WORKER_FAILURE_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_worker_failure_receipt_v1"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_DIAGNOSTIC_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")


class ResultStore(Protocol):
    def read_current(
        self, *, uri: str, allow_missing: bool = False
    ) -> tuple[Mapping[str, Any], bytes] | None: ...

    def list_prefix(self, *, prefix: str) -> Sequence[Mapping[str, Any]]: ...

    def read_bytes(self, *, uri: str, generation: int) -> bytes: ...


class InstanceObserver(Protocol):
    def get_instance(
        self, *, instance_name: str
    ) -> Mapping[str, Any] | None: ...

    def get_serial_port_output(
        self, *, instance_name: str, start: int = -65_536
    ) -> Mapping[str, Any] | None: ...


class WorkerDiagnosticFailure(RuntimeError):
    """A secret-free worker failure recovered from the VM serial console."""

    def __init__(self, receipt: Mapping[str, Any]) -> None:
        super().__init__("worker reported sanitized pre-DONE failure")
        self.receipt = dict(receipt)


def canonical_bytes(value: Any) -> bytes:
    return alias_bridge.canonical_bytes(value)


def canonical_sha256(value: Any) -> str:
    return alias_bridge.canonical_sha256(value)


def _seal_worker_failure_receipt(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    body = dict(value)
    return {**body, "receipt_sha256": canonical_sha256(body)}


def _worker_failure_receipt_from_serial(
    *,
    serial: Mapping[str, Any],
    instance_name: str,
    job_id: str,
    source_role: str,
    provider_status: str | None,
) -> dict[str, Any] | None:
    contents = serial.get("contents")
    if (
        serial.get("instance_name") != instance_name
        or serial.get("port") != 1
        or type(serial.get("start")) is not int
        or type(serial.get("next")) is not int
        or not isinstance(contents, str)
    ):
        raise ValueError("worker serial diagnostic shape changed")
    marker: dict[str, Any] | None = None
    for line in reversed(contents.splitlines()):
        if not line.startswith(WORKER_FAILURE_MARKER_PREFIX):
            continue
        try:
            parsed = json.loads(line[len(WORKER_FAILURE_MARKER_PREFIX) :])
        except json.JSONDecodeError:
            raise ValueError("worker failure marker JSON changed") from None
        if not isinstance(parsed, dict):
            raise ValueError("worker failure marker shape changed")
        marker = parsed
        break
    if marker is None and provider_status not in {"STOPPING", "TERMINATED"}:
        return None
    if marker is None:
        marker = {
            "schema": WORKER_FAILURE_MARKER_SCHEMA,
            "status": "worker_failed_before_done",
            "stage": "serial_marker_missing",
            "exception_type": "UnknownWorkerFailure",
        }
    if (
        set(marker)
        != {"schema", "status", "stage", "exception_type"}
        or marker.get("schema") != WORKER_FAILURE_MARKER_SCHEMA
        or marker.get("status") != "worker_failed_before_done"
        or not isinstance(marker.get("stage"), str)
        or _SAFE_DIAGNOSTIC_NAME.fullmatch(marker["stage"]) is None
        or not isinstance(marker.get("exception_type"), str)
        or _SAFE_DIAGNOSTIC_NAME.fullmatch(marker["exception_type"])
        is None
    ):
        raise ValueError("worker failure marker semantics changed")
    return _seal_worker_failure_receipt(
        {
            "schema": WORKER_FAILURE_RECEIPT_SCHEMA,
            "status": "sanitized_worker_failure_recovered_before_done",
            "instance_name": instance_name,
            "job_id": job_id,
            "source_role": source_role,
            "provider_status": provider_status,
            "failure_stage": marker["stage"],
            "exception_type": marker["exception_type"],
            "serial_start": serial["start"],
            "serial_next": serial["next"],
            "serial_contents_sha256": hashlib.sha256(
                contents.encode("utf-8")
            ).hexdigest(),
            "serial_contents_stored": False,
            "exception_message_stored": False,
            "access_token_stored": False,
            "current_profile_changed": False,
        }
    )


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or _SHA256.fullmatch(value) is None
        or value == "0" * 64
    ):
        raise ValueError(f"{label} changed")
    return value


def _json(raw: bytes, label: str, *, canonical: bool) -> dict[str, Any]:
    if not isinstance(raw, bytes) or not raw or len(raw) > 8 * 1024 * 1024:
        raise ValueError(f"{label} bytes changed")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError(f"{label} JSON changed") from None
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON changed")
    if canonical and canonical_bytes(value) != raw:
        raise ValueError(f"{label} canonical bytes changed")
    return value


def _safe_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("result tree path changed")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("result tree path changed")
    return path.as_posix()


def _record(value: Mapping[str, Any]) -> dict[str, Any]:
    checked = dict(value)
    required = {
        "uri",
        "generation",
        "metageneration",
        "bytes",
        "sha256",
        "crc32c",
        "etag",
    }
    if (
        set(checked) != required
        or not isinstance(checked["uri"], str)
        or type(checked["generation"]) is not int
        or checked["generation"] <= 0
        or type(checked["metageneration"]) is not int
        or checked["metageneration"] <= 0
        or type(checked["bytes"]) is not int
        or checked["bytes"] <= 0
        or _SHA256.fullmatch(str(checked["sha256"])) is None
        or not isinstance(checked["crc32c"], str)
        or not checked["crc32c"]
        or not isinstance(checked["etag"], str)
        or not checked["etag"]
    ):
        raise ValueError("result object record changed")
    return checked


def _sealed(
    value: Mapping[str, Any], *, digest_field: str, label: str
) -> dict[str, Any]:
    checked = dict(value)
    supplied = _sha(checked.pop(digest_field, None), label)
    if canonical_sha256(checked) != supplied:
        raise ValueError(f"{label} digest changed")
    return {**checked, digest_field: supplied}


def _job(
    deployment: Mapping[str, Any], execution_job_id: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    layouts = [
        row
        for row in deployment["remote_layout"]["jobs"]
        if row["job_id"] == execution_job_id
    ]
    bindings = [
        row
        for row in deployment["payload_binding"]["job_bindings"]
        if row["source_role"] == layouts[0]["source_role"]
    ] if len(layouts) == 1 else []
    instances = [
        row
        for row in deployment["instances"]
        if row["job_id"] == execution_job_id
    ]
    if len(layouts) != 1 or len(bindings) != 1 or len(instances) != 1:
        raise ValueError("result role mapping changed")
    return dict(layouts[0]), dict(bindings[0]), dict(instances[0])


def _validate_done(
    value: Mapping[str, Any],
    *,
    deployment: Mapping[str, Any],
    layout: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    done = _sealed(value, digest_field="done_sha256", label="DONE envelope")
    expected_fields = {
        "schema",
        "deployment_contract_sha256",
        "direct_stage_identity_sha256",
        "run_name",
        "stage_id",
        "execution_job_id",
        "inner_job_id",
        "source_role",
        "attempt_index",
        "completed_hand_indices",
        "tree_object_records",
        "tree_object_records_sha256",
        "upload_sha256s",
        "heartbeat_sha256s",
        "tree_object_count",
        "upload_count",
        "heartbeat_count",
        "old_result_write_count",
        "direct_v2_only",
        "done_sha256",
    }
    if set(done) != expected_fields:
        raise ValueError("DONE envelope fields changed")
    tree_records = done["tree_object_records"]
    if (
        done["schema"] != f"{alias_bridge.PUBLISH_RECEIPT_SCHEMA}_done_v1"
        or done["deployment_contract_sha256"]
        != deployment["deployment_contract_sha256"]
        or done["direct_stage_identity_sha256"]
        != deployment["direct_stage_identity_sha256"]
        or done["run_name"] != deployment["run_name"]
        or done["stage_id"] != deployment["stage_id"]
        or done["execution_job_id"] != layout["job_id"]
        or done["inner_job_id"] != binding["inner_job_id"]
        or done["source_role"] != layout["source_role"]
        or done["attempt_index"] != 0
        or done["completed_hand_indices"]
        != binding["work_hand_indices"]
        or not isinstance(tree_records, list)
        or len(tree_records) != alias_bridge.TREE_OBJECT_COUNT
        or done["tree_object_records_sha256"]
        != canonical_sha256(tree_records)
        or not isinstance(done["upload_sha256s"], list)
        or len(done["upload_sha256s"]) != alias_bridge.UPLOAD_COUNT
        or not isinstance(done["heartbeat_sha256s"], list)
        or len(done["heartbeat_sha256s"])
        != alias_bridge.HEARTBEAT_COUNT
        or done["tree_object_count"] != alias_bridge.TREE_OBJECT_COUNT
        or done["upload_count"] != alias_bridge.UPLOAD_COUNT
        or done["heartbeat_count"] != alias_bridge.HEARTBEAT_COUNT
        or done["old_result_write_count"] != 0
        or done["direct_v2_only"] is not True
    ):
        raise ValueError("DONE envelope semantics changed")
    return done


def _validate_upload(
    value: Mapping[str, Any],
    *,
    deployment: Mapping[str, Any],
    layout: Mapping[str, Any],
    binding: Mapping[str, Any],
    sequence: int,
    expected_sha: str,
    tree_by_path: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    upload = _sealed(value, digest_field="upload_sha256", label="upload")
    hand_index = binding["work_hand_indices"][sequence - 1]
    paths = (
        f"roots/hand_{hand_index:03d}.json",
        f"hands/{layout['source_role']}/hand_{hand_index:03d}.json",
    )
    if (
        upload["upload_sha256"] != expected_sha
        or upload.get("schema")
        != f"{alias_bridge.PUBLISH_RECEIPT_SCHEMA}_upload_v1"
        or upload.get("deployment_contract_sha256")
        != deployment["deployment_contract_sha256"]
        or upload.get("direct_stage_identity_sha256")
        != deployment["direct_stage_identity_sha256"]
        or upload.get("execution_job_id") != layout["job_id"]
        or upload.get("inner_job_id") != binding["inner_job_id"]
        or upload.get("source_role") != layout["source_role"]
        or upload.get("sequence") != sequence
        or upload.get("hand_index") != hand_index
        or upload.get("tree_objects")
        != [tree_by_path[path] for path in paths]
        or upload.get("direct_v2_only") is not True
    ):
        raise ValueError("upload envelope changed")
    return upload


def _validate_heartbeat(
    value: Mapping[str, Any],
    *,
    deployment: Mapping[str, Any],
    layout: Mapping[str, Any],
    binding: Mapping[str, Any],
    sequence: int,
    expected_sha: str,
    upload_sha256s: Sequence[str],
) -> dict[str, Any]:
    heartbeat = _sealed(
        value, digest_field="heartbeat_sha256", label="heartbeat"
    )
    if (
        heartbeat["heartbeat_sha256"] != expected_sha
        or heartbeat.get("schema")
        != f"{alias_bridge.PUBLISH_RECEIPT_SCHEMA}_heartbeat_v1"
        or heartbeat.get("deployment_contract_sha256")
        != deployment["deployment_contract_sha256"]
        or heartbeat.get("execution_job_id") != layout["job_id"]
        or heartbeat.get("source_role") != layout["source_role"]
        or heartbeat.get("sequence") != sequence
        or heartbeat.get("completed_hand_indices")
        != binding["work_hand_indices"][:sequence]
        or heartbeat.get("upload_sha256s")
        != list(upload_sha256s[:sequence])
        or heartbeat.get("direct_v2_only") is not True
    ):
        raise ValueError("heartbeat envelope changed")
    return heartbeat


def receive_role(
    *,
    deployment_contract: Mapping[str, Any],
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
    execution_job_id: str,
    store: ResultStore,
    destination: str | Path,
) -> dict[str, Any]:
    deployment = deployment_v2.validate_deployment_contract(
        deployment_contract,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
    )
    layout, binding, _ = _job(deployment, execution_job_id)
    heartbeat_prefix = layout["heartbeat_uris"][0].rsplit("/", 1)[0]
    listed = [
        *[
            _record(row)
            for row in store.list_prefix(prefix=layout["result_prefix"])
        ],
        *[
            _record(row)
            for row in store.list_prefix(prefix=heartbeat_prefix)
        ],
    ]
    expected_uris = [
        *layout["tree_object_uris"],
        *layout["upload_uris"],
        *layout["heartbeat_uris"],
        layout["done_uri"],
    ]
    by_uri = {row["uri"]: row for row in listed}
    if (
        len(listed) != alias_bridge.RESULT_OBJECT_COUNT
        or len(by_uri) != len(listed)
        or set(by_uri) != set(expected_uris)
    ):
        raise ValueError("exact 44-object result inventory changed")
    raw_by_uri: dict[str, bytes] = {}
    for uri in expected_uris:
        record = by_uri[uri]
        raw = store.read_bytes(
            uri=uri, generation=record["generation"]
        )
        if (
            len(raw) != record["bytes"]
            or hashlib.sha256(raw).hexdigest() != record["sha256"]
        ):
            raise ValueError("generation-pinned result bytes changed")
        raw_by_uri[uri] = raw

    done = _validate_done(
        _json(raw_by_uri[layout["done_uri"]], "DONE envelope", canonical=True),
        deployment=deployment,
        layout=layout,
        binding=binding,
    )
    tree_by_path: dict[str, dict[str, Any]] = {}
    for row in done["tree_object_records"]:
        if not isinstance(row, Mapping) or set(row) != {
            "uri",
            "generation",
            "sha256",
            "bytes",
            "created",
            "path",
        }:
            raise ValueError("DONE tree record changed")
        path = _safe_path(row["path"])
        actual = by_uri.get(row["uri"])
        if (
            path in tree_by_path
            or actual is None
            or row["uri"] not in layout["tree_object_uris"]
            or row["generation"] != actual["generation"]
            or row["sha256"] != actual["sha256"]
            or row["bytes"] != actual["bytes"]
            or row["created"] is not True
        ):
            raise ValueError("DONE tree record disagrees with provider")
        tree_by_path[path] = dict(row)
    if list(tree_by_path) != binding["tree_paths"]:
        raise ValueError("DONE tree path order changed")

    uploads = []
    for sequence, (uri, expected_sha) in enumerate(
        zip(
            layout["upload_uris"],
            done["upload_sha256s"],
            strict=True,
        ),
        1,
    ):
        uploads.append(
            _validate_upload(
                _json(raw_by_uri[uri], "upload", canonical=True),
                deployment=deployment,
                layout=layout,
                binding=binding,
                sequence=sequence,
                expected_sha=expected_sha,
                tree_by_path=tree_by_path,
            )
        )
    heartbeats = []
    for sequence, (uri, expected_sha) in enumerate(
        zip(
            layout["heartbeat_uris"],
            done["heartbeat_sha256s"],
            strict=True,
        ),
        1,
    ):
        heartbeats.append(
            _validate_heartbeat(
                _json(raw_by_uri[uri], "heartbeat", canonical=True),
                deployment=deployment,
                layout=layout,
                binding=binding,
                sequence=sequence,
                expected_sha=expected_sha,
                upload_sha256s=done["upload_sha256s"],
            )
        )

    root = Path(destination).resolve()
    if root.exists() or root.is_symlink():
        raise FileExistsError("result receive destination must be fresh")
    if not root.parent.is_dir() or root.parent.is_symlink():
        raise ValueError("result receive parent changed")
    root.mkdir()
    try:
        for path, tree in tree_by_path.items():
            target = root.joinpath(*PurePosixPath(path).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            raw = raw_by_uri[tree["uri"]]
            with target.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
        runner.validate_completed_output(root)
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    body = {
        "schema": SCHEMA,
        "status": "exact_44_objects_generation_pinned_and_tree_validated",
        "deployment_contract_sha256": deployment[
            "deployment_contract_sha256"
        ],
        "execution_job_id": execution_job_id,
        "inner_job_id": binding["inner_job_id"],
        "source_role": layout["source_role"],
        "result_prefix": layout["result_prefix"],
        "done_uri": layout["done_uri"],
        "listed_records": listed,
        "listed_records_sha256": canonical_sha256(listed),
        "done_sha256": done["done_sha256"],
        "upload_sha256s": done["upload_sha256s"],
        "heartbeat_sha256s": done["heartbeat_sha256s"],
        "tree_object_count": alias_bridge.TREE_OBJECT_COUNT,
        "upload_count": alias_bridge.UPLOAD_COUNT,
        "heartbeat_count": alias_bridge.HEARTBEAT_COUNT,
        "done_count": alias_bridge.DONE_COUNT,
        "result_object_count": alias_bridge.RESULT_OBJECT_COUNT,
        "generation_pinned_read_count": alias_bridge.RESULT_OBJECT_COUNT,
        "materialized_tree_path": str(root),
        "runner_validate_completed_output_performed": True,
        "old_result_write_count": 0,
        "current_profile_changed": False,
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def wait_for_pair_and_receive(
    *,
    deployment_contract: Mapping[str, Any],
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
    store: ResultStore,
    instance_observer: InstanceObserver,
    destination_root: str | Path,
    timeout_seconds: int = MAX_DONE_WAIT_SECONDS,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if timeout_seconds != MAX_DONE_WAIT_SECONDS:
        raise ValueError("pair DONE timeout changed")
    deployment = deployment_v2.validate_deployment_contract(
        deployment_contract,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
    )
    destination = Path(destination_root).resolve()
    if (
        destination.exists()
        or destination.is_symlink()
        or not destination.parent.is_dir()
        or destination.parent.is_symlink()
    ):
        raise FileExistsError("pair receive root must be fresh")
    deadline = monotonic() + timeout_seconds
    done_records: dict[str, dict[str, Any]] = {}
    while len(done_records) != deployment_v2.VM_COUNT:
        for layout, instance in zip(
            deployment["remote_layout"]["jobs"],
            deployment["instances"],
            strict=True,
        ):
            job_id = layout["job_id"]
            if job_id in done_records:
                continue
            observed = store.read_current(
                uri=layout["done_uri"], allow_missing=True
            )
            if observed is not None:
                record, raw = observed
                _validate_done(
                    _json(raw, "DONE envelope", canonical=True),
                    deployment=deployment,
                    layout=layout,
                    binding=_job(deployment, job_id)[1],
                )
                done_records[job_id] = _record(record)
                continue
            provider = instance_observer.get_instance(
                instance_name=instance["instance_name"]
            )
            if provider is None:
                observed = store.read_current(
                    uri=layout["done_uri"], allow_missing=True
                )
                if observed is None:
                    raise RuntimeError(
                        "worker became absent before direct-v2 DONE"
                    )
                record, raw = observed
                _validate_done(
                    _json(raw, "DONE envelope", canonical=True),
                    deployment=deployment,
                    layout=layout,
                    binding=_job(deployment, job_id)[1],
                )
                done_records[job_id] = _record(record)
                continue
            serial_reader = getattr(
                instance_observer, "get_serial_port_output", None
            )
            if callable(serial_reader):
                serial = serial_reader(
                    instance_name=instance["instance_name"],
                    start=-65_536,
                )
                if serial is not None:
                    provider_status = provider.get("status")
                    if provider_status is not None and not isinstance(
                        provider_status, str
                    ):
                        raise ValueError("worker provider status changed")
                    failure = _worker_failure_receipt_from_serial(
                        serial=serial,
                        instance_name=instance["instance_name"],
                        job_id=job_id,
                        source_role=layout["source_role"],
                        provider_status=provider_status,
                    )
                    if failure is not None:
                        raise WorkerDiagnosticFailure(failure)
        if len(done_records) == deployment_v2.VM_COUNT:
            break
        if monotonic() >= deadline:
            raise TimeoutError("Step12b pair DONE timed out")
        sleep(POLL_SECONDS)

    destination.mkdir()
    role_receipts = []
    try:
        for job_id, role in zip(
            deployment["selected_job_ids"],
            deployment["source_roles"],
            strict=True,
        ):
            role_receipts.append(
                receive_role(
                    deployment_contract=deployment,
                    candidate_payload_contract=candidate_payload_contract,
                    reference_payload_contract=reference_payload_contract,
                    controller_public_key_record=controller_public_key_record,
                    run_nonce=run_nonce,
                    execution_job_id=job_id,
                    store=store,
                    destination=destination / role,
                )
            )
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    body = {
        "schema": PAIR_SCHEMA,
        "status": "candidate_and_reference_exact_44_object_trees_received",
        "deployment_contract_sha256": deployment[
            "deployment_contract_sha256"
        ],
        "selected_job_ids": list(deployment["selected_job_ids"]),
        "source_roles": list(deployment["source_roles"]),
        "done_records": [
            done_records[job_id]
            for job_id in deployment["selected_job_ids"]
        ],
        "role_receipts": role_receipts,
        "role_receipts_sha256": canonical_sha256(role_receipts),
        "pair_result_object_count": (
            alias_bridge.RESULT_OBJECT_COUNT * deployment_v2.VM_COUNT
        ),
        "both_trees_runner_validated": True,
        "current_profile_changed": False,
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


__all__ = [
    "MAX_DONE_WAIT_SECONDS",
    "PAIR_SCHEMA",
    "SCHEMA",
    "WORKER_FAILURE_MARKER_PREFIX",
    "WORKER_FAILURE_MARKER_SCHEMA",
    "WORKER_FAILURE_RECEIPT_SCHEMA",
    "WorkerDiagnosticFailure",
    "canonical_bytes",
    "canonical_sha256",
    "receive_role",
    "wait_for_pair_and_receive",
]
