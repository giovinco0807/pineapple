"""In-memory fault injection for the diagnostic native worker transport.

This module is intentionally *not* a cloud implementation.  It has no
filesystem, network, subprocess, gcloud, object-store, or VM API.  The only
operation that may inspect a package is ``worker_adapter.build_preview``.

The fake transport models the properties that a later real controller must
preserve:

* generation-match-zero immutable object creation;
* an exactly empty stage prefix before the first sentinel is created;
* separate, simulation-only authorization and claim records;
* attempts zero and one only;
* a trailing upload without its heartbeat is resumed by adding the heartbeat;
* a complete upload/heartbeat prefix without DONE is finalized without
  republishing any artifact;
* DONE is published last and read back before receive;
* receive and stage escalation require every worker VM to be absent;
* success asks the worker to self-delete while failure preserves it until a
  bounded controller shutdown.

Every result payload is produced by the adapter's transport fixture helpers.
It is non-scientific and is never quality, training, performance-lock, or
promotion evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_worker_adapter as adapter,
)
from . import hu_m31_t3_step6d_rearm2_diagnostic_canary_plan as plan


STAGE_SENTINEL_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_cloud_worker_fake_stage_sentinel_v1"
)
SIMULATION_AUTHORIZATION_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_cloud_worker_fake_authorization_v1"
)
SIMULATION_CLAIM_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_cloud_worker_fake_claim_v1"
)
WORKER_LIFECYCLE_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_cloud_worker_fake_lifecycle_v1"
)
CONTROLLER_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_cloud_worker_fake_controller_receipt_v1"
)
FAKE_SNAPSHOT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_cloud_worker_fake_snapshot_v1"
)
TREE_FIXTURE_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_cloud_worker_fake_tree_identity_v1"
)

MAX_ATTEMPTS = 2
MAX_CONTROLLER_SHUTDOWN_SECONDS = 120
_SENTINEL_RELATIVE = "control/fake-stage-prefix-sentinel.json"


def canonical_bytes(value: Any) -> bytes:
    return adapter.canonical_bytes(value)


def canonical_sha256(value: Any) -> str:
    return adapter.canonical_sha256(value)


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_bytes(value))


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{label} keys changed")


def _strict_bool(value: Any, expected: bool, label: str) -> bool:
    if value is not expected:
        raise ValueError(f"{label} must be the exact boolean {expected}")
    return value


def _strict_int(
    value: Any,
    label: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer, not boolean")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} is below its minimum")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} is above its maximum")
    return value


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        or value == "0" * 64
    ):
        raise ValueError(f"{label} must be a nonzero lowercase sha256")
    return value


def _join(prefix: str, relative: str) -> str:
    if not isinstance(prefix, str) or not prefix.startswith("gs://"):
        raise ValueError("fake remote prefix must be an exact gs:// identity")
    return f"{prefix.rstrip('/')}/{relative.lstrip('/')}"


@dataclass(frozen=True)
class PutResult:
    uri: str
    generation: int
    content_sha256: str
    created: bool


class InMemoryGenerationMatchStore:
    """Canonical JSON object store with if-generation-match=0 semantics."""

    def __init__(self) -> None:
        self._objects: dict[str, tuple[int, bytes]] = {}
        self._next_generation = 1
        self.events: list[dict[str, Any]] = []

    @staticmethod
    def _under_prefix(uri: str, prefix: str) -> bool:
        base = prefix.rstrip("/")
        return uri == base or uri.startswith(base + "/")

    def list_prefix(self, prefix: str) -> list[str]:
        return sorted(
            uri for uri in self._objects if self._under_prefix(uri, prefix)
        )

    def contains(self, uri: str) -> bool:
        return uri in self._objects

    def put_generation_match_zero(
        self, uri: str, content: Mapping[str, Any]
    ) -> PutResult:
        if not isinstance(uri, str) or not uri.startswith("gs://"):
            raise ValueError("fake remote object URI must start with gs://")
        if not isinstance(content, Mapping):
            raise ValueError("fake remote content must be an object")
        value = dict(content)
        raw = canonical_bytes(value)
        digest = canonical_sha256(value)
        prior = self._objects.get(uri)
        if prior is not None:
            generation, prior_raw = prior
            if prior_raw != raw:
                raise FileExistsError(
                    "if-generation-match=0 collision with different bytes"
                )
            self.events.append(
                {
                    "operation": "idempotent_readback",
                    "uri": uri,
                    "generation": generation,
                    "content_sha256": digest,
                }
            )
            return PutResult(uri, generation, digest, False)
        generation = self._next_generation
        self._next_generation += 1
        self._objects[uri] = (generation, raw)
        self.events.append(
            {
                "operation": "conditional_create_generation_match_zero",
                "uri": uri,
                "generation": generation,
                "content_sha256": digest,
            }
        )
        return PutResult(uri, generation, digest, True)

    # A short alias makes fault-injection tests easier to read.
    put_once = put_generation_match_zero

    def read(self, uri: str) -> dict[str, Any]:
        try:
            generation, raw = self._objects[uri]
        except KeyError as exc:
            raise FileNotFoundError(f"fake remote object is absent: {uri}") from exc
        value = json.loads(raw)
        if not isinstance(value, dict) or canonical_bytes(value) != raw:
            raise ValueError("fake remote object is not canonical JSON")
        self.events.append(
            {
                "operation": "readback",
                "uri": uri,
                "generation": generation,
                "content_sha256": canonical_sha256(value),
            }
        )
        return _copy_json(value)

    def record(self, uri: str) -> dict[str, Any]:
        value = self.read(uri)
        generation, _raw = self._objects[uri]
        return {
            "uri": uri,
            "generation": generation,
            "content_sha256": canonical_sha256(value),
            "content": value,
        }


@dataclass(frozen=True)
class ControllerReceiptProof:
    """Opaque proof that can only be issued by its completed source stage."""

    receipt_bytes: bytes
    _source_stage: Any
    _issuer_nonce: object

    @property
    def content(self) -> dict[str, Any]:
        value = json.loads(self.receipt_bytes)
        if not isinstance(value, dict) or canonical_bytes(value) != self.receipt_bytes:
            raise ValueError("fake controller receipt is not canonical")
        return value


class FakeCloudWorkerStage:
    """Controller/worker lifecycle backed only by an in-memory fake store."""

    def __init__(
        self,
        *,
        package_dir: str,
        stage_id: str,
        store: InMemoryGenerationMatchStore | None = None,
        stage1_proof: ControllerReceiptProof | None = None,
    ) -> None:
        self.store = store or InMemoryGenerationMatchStore()
        self._issuer_nonce = object()
        self._owned_objects: dict[str, str] = {}
        self._receive: dict[str, Any] | None = None
        self._controller_receipt: dict[str, Any] | None = None
        self._attempt_previews: dict[int, dict[str, Any]] = {}
        self._active_preview: dict[str, Any] | None = None
        self._vm: dict[str, dict[str, Any]] = {}
        self._attempt_claims: dict[tuple[str, int], dict[str, Any]] = {}
        self._opened_attempt_controls: set[int] = set()
        self._success_records: dict[str, dict[str, Any]] = {}
        self._failure_records: dict[str, dict[str, Any]] = {}

        prerequisite_preview: Mapping[str, Any] | None = None
        prerequisite_receive: Mapping[str, Any] | None = None
        if stage_id == plan.STAGE1_ID:
            if stage1_proof is not None:
                raise ValueError("stage1 cannot consume a stage1 receipt")
        elif stage_id == plan.STAGE2_ID:
            prerequisite_preview, prerequisite_receive = self._validate_stage1_proof(
                stage1_proof
            )
        else:
            raise ValueError("fake worker permits only diagnostic stage1/stage2")

        # This is the only call in this module allowed to read the package.
        preview = adapter.build_preview(
            package_dir=package_dir,
            stage_id=stage_id,
            attempt_index=0,
            prerequisite_stage1_preview=prerequisite_preview,
            prerequisite_stage1_receive=prerequisite_receive,
        )
        self._package_dir = package_dir
        self._stage1_preview = prerequisite_preview
        self._stage1_receive = prerequisite_receive
        self._validate_preview_boundary(preview)
        self._attempt_previews[0] = preview
        self._active_preview = preview
        self._stage_prefix = preview["remote_manifest"]["prefix"]
        self._assert_exact_empty_stage_prefix()
        self._publish_stage_sentinel()

    @property
    def preview(self) -> dict[str, Any]:
        if self._active_preview is None:
            raise RuntimeError("fake worker stage has no active preview")
        return _copy_json(self._active_preview)

    @property
    def attempt_index(self) -> int:
        return self.preview["attempt_index"]

    def _validate_stage1_proof(
        self, proof: ControllerReceiptProof | None
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        if (
            not isinstance(proof, ControllerReceiptProof)
            or not isinstance(proof._source_stage, FakeCloudWorkerStage)
            or proof._source_stage._issuer_nonce is not proof._issuer_nonce
        ):
            raise ValueError("stage2 requires an opaque stage1 controller proof")
        source = proof._source_stage
        if source._controller_receipt is None or source._receive is None:
            raise ValueError("stage1 controller proof was not issued")
        if canonical_bytes(source._controller_receipt) != proof.receipt_bytes:
            raise ValueError("stage1 controller proof bytes changed")
        receipt = proof.content
        inventory = source.assert_known_inventory()
        if inventory["all_vm_absent"] is not True:
            raise ValueError("stage1 controller proof lost VM absence")
        source._validate_controller_receipt(receipt)
        return source._active_preview, source._receive

    @staticmethod
    def _validate_preview_boundary(value: Mapping[str, Any]) -> None:
        preview = dict(value)
        if (
            preview.get("schema") != adapter.PREVIEW_SCHEMA
            or preview.get("adapter_schema") != adapter.ADAPTER_SCHEMA
            or preview.get("stage_id") not in (plan.STAGE1_ID, plan.STAGE2_ID)
            or type(preview.get("attempt_index")) is not int
            or preview["attempt_index"] not in (0, 1)
            or type(preview.get("max_attempts")) is not int
            or preview.get("max_attempts") != MAX_ATTEMPTS
            or not isinstance(preview.get("jobs"), list)
            or type(preview.get("vm_count")) is not int
            or preview.get("vm_count") != len(preview["jobs"])
            or preview.get("selected_job_ids")
            != [job.get("job_id") for job in preview["jobs"]]
        ):
            raise ValueError("diagnostic worker preview boundary changed")
        _sha(preview.get("package_manifest_sha256"), "package manifest")
        capabilities = preview.get("capabilities")
        if not isinstance(capabilities, Mapping):
            raise ValueError("diagnostic worker capabilities are missing")
        for field in (
            "launch_ready",
            "cloud_launch_authorized",
            "gcloud_invocation_authorized",
            "claim_write_authorized",
            "authorization_write_authorized",
            "object_write_authorized",
            "vm_create_authorized",
            "performance_lock_evidence",
            "quality_evidence",
            "training_eligible",
            "promotion_evidence",
        ):
            _strict_bool(capabilities.get(field), False, f"capability {field}")
        if preview["attempt_index"] == 0:
            resume = preview["stage_token"].get("resume_binding")
            if resume is not None:
                raise ValueError("attempt0 cannot contain a resume binding")

    def _assert_exact_empty_stage_prefix(self) -> None:
        objects = self.store.list_prefix(self._stage_prefix)
        if objects:
            raise FileExistsError(
                "diagnostic stage prefix must be exactly empty before sentinel"
            )

    def _sentinel_uri(self) -> str:
        return _join(self._stage_prefix, _SENTINEL_RELATIVE)

    def _authorization_uri(self, attempt_index: int) -> str:
        preview = self._attempt_previews[attempt_index]
        return _join(
            preview["remote_manifest"]["attempt_control_prefix"],
            "fake-authorization.json",
        )

    def _claim_uri(self, job_id: str, attempt_index: int) -> str:
        preview = self._attempt_previews[attempt_index]
        return _join(
            preview["remote_manifest"]["attempt_control_prefix"],
            f"fake-claims/{job_id}.json",
        )

    def _lifecycle_uri(self, job_id: str, attempt_index: int) -> str:
        preview = self._attempt_previews[attempt_index]
        return _join(
            preview["remote_manifest"]["attempt_control_prefix"],
            f"fake-lifecycle/{job_id}.json",
        )

    @staticmethod
    def _tree_fixture(record: Mapping[str, Any]) -> dict[str, Any]:
        row = dict(record)
        _exact(row, {"path", "uri", "sha256", "bytes"}, "tree object record")
        _sha(row["sha256"], "tree object record")
        _strict_int(row["bytes"], "tree object bytes", minimum=1)
        return {
            "schema": TREE_FIXTURE_SCHEMA,
            "path": row["path"],
            "declared_content_sha256": row["sha256"],
            "declared_content_bytes": row["bytes"],
            "identity_envelope_only": True,
            "content_bytes_embedded": False,
            "transport_fixture_only": True,
            "scientific_payload_present": False,
        }

    def _publish_tree_record(self, record: Mapping[str, Any]) -> PutResult:
        return self._publish_and_readback(
            str(record["uri"]), self._tree_fixture(record)
        )

    def _publish_and_readback(
        self, uri: str, content: Mapping[str, Any]
    ) -> PutResult:
        if self.store.contains(uri) and uri not in self._owned_objects:
            raise ValueError(
                "known fake remote URI was not created by this lifecycle"
            )
        result = self.store.put_generation_match_zero(uri, content)
        checked = self.store.read(uri)
        if checked != dict(content):
            raise ValueError("fake remote immutable readback changed")
        digest = canonical_sha256(checked)
        prior_digest = self._owned_objects.get(uri)
        if prior_digest is not None and prior_digest != digest:
            raise ValueError("owned fake remote object identity changed")
        self._owned_objects[uri] = digest
        return result

    def _publish_stage_sentinel(self) -> None:
        preview = self.preview
        value = {
            "schema": STAGE_SENTINEL_SCHEMA,
            "stage_id": preview["stage_id"],
            "run_name": preview["run_name"],
            "stage_prefix": self._stage_prefix,
            "package_manifest_sha256": preview["package_manifest_sha256"],
            "conditional_create_generation_match": 0,
            "prefix_was_exactly_empty": True,
            "simulation_only": True,
            "cloud_launch_authorized": False,
            "remote_write_performed": False,
        }
        self._validate_stage_sentinel(value)
        result = self._publish_and_readback(self._sentinel_uri(), value)
        if result.created is not True:
            raise AssertionError("new fake stage sentinel was not created")

    def _validate_stage_sentinel(self, value: Mapping[str, Any]) -> None:
        row = dict(value)
        _exact(
            row,
            {
                "schema",
                "stage_id",
                "run_name",
                "stage_prefix",
                "package_manifest_sha256",
                "conditional_create_generation_match",
                "prefix_was_exactly_empty",
                "simulation_only",
                "cloud_launch_authorized",
                "remote_write_performed",
            },
            "fake stage sentinel",
        )
        if (
            row["schema"] != STAGE_SENTINEL_SCHEMA
            or row["stage_id"] != self.preview["stage_id"]
            or row["run_name"] != self.preview["run_name"]
            or row["stage_prefix"] != self._stage_prefix
            or row["package_manifest_sha256"]
            != self.preview["package_manifest_sha256"]
        ):
            raise ValueError("fake stage sentinel identity changed")
        _strict_int(
            row["conditional_create_generation_match"],
            "sentinel generation match",
            minimum=0,
            maximum=0,
        )
        _strict_bool(row["prefix_was_exactly_empty"], True, "empty prefix")
        _strict_bool(row["simulation_only"], True, "sentinel simulation")
        _strict_bool(
            row["cloud_launch_authorized"], False, "sentinel authorization"
        )
        _strict_bool(row["remote_write_performed"], False, "sentinel remote write")

    def _authorization(self, attempt_index: int) -> dict[str, Any]:
        preview = self._attempt_previews[attempt_index]
        value = {
            "schema": SIMULATION_AUTHORIZATION_SCHEMA,
            "stage_id": preview["stage_id"],
            "run_name": preview["run_name"],
            "attempt_index": attempt_index,
            "preview_sha256": canonical_sha256(preview),
            "stage_sentinel_sha256": canonical_sha256(
                self.store.read(self._sentinel_uri())
            ),
            "simulation_only": True,
            "real_cloud_authorized": False,
            "gcloud_authorized": False,
            "vm_create_authorized": False,
            "object_write_authorized": False,
        }
        self._validate_authorization(value, attempt_index=attempt_index)
        return value

    def _validate_authorization(
        self, value: Mapping[str, Any], *, attempt_index: int
    ) -> None:
        row = dict(value)
        _exact(
            row,
            {
                "schema",
                "stage_id",
                "run_name",
                "attempt_index",
                "preview_sha256",
                "stage_sentinel_sha256",
                "simulation_only",
                "real_cloud_authorized",
                "gcloud_authorized",
                "vm_create_authorized",
                "object_write_authorized",
            },
            "fake simulation authorization",
        )
        preview = self._attempt_previews[attempt_index]
        if (
            row["schema"] != SIMULATION_AUTHORIZATION_SCHEMA
            or row["stage_id"] != preview["stage_id"]
            or row["run_name"] != preview["run_name"]
            or row["preview_sha256"] != canonical_sha256(preview)
            or row["stage_sentinel_sha256"]
            != canonical_sha256(self.store.read(self._sentinel_uri()))
        ):
            raise ValueError("fake simulation authorization identity changed")
        _strict_int(
            row["attempt_index"],
            "authorization attempt index",
            minimum=attempt_index,
            maximum=attempt_index,
        )
        _strict_bool(row["simulation_only"], True, "authorization simulation")
        for field in (
            "real_cloud_authorized",
            "gcloud_authorized",
            "vm_create_authorized",
            "object_write_authorized",
        ):
            _strict_bool(row[field], False, f"authorization {field}")

    def _job(self, job_id: str) -> dict[str, Any]:
        job = next(
            (row for row in self.preview["jobs"] if row["job_id"] == job_id),
            None,
        )
        if job is None:
            raise ValueError("fake worker job escaped exact stage set")
        return job

    def start_attempt(
        self, *, job_id: str, attempt_index: int
    ) -> dict[str, Any]:
        self._assert_no_unknown_objects()
        _strict_int(
            attempt_index,
            "attempt index",
            minimum=0,
            maximum=MAX_ATTEMPTS - 1,
        )
        if attempt_index != self.attempt_index:
            raise ValueError("attempt does not match active preview")
        job = self._job(job_id)
        key = (job_id, attempt_index)
        current = self._vm.get(job_id)
        if key in self._attempt_claims:
            raise ValueError("fake worker attempt was already claimed")
        if current is not None and current["state"] != "absent":
            raise ValueError("prior fake worker VM is not absent")
        if attempt_index == 1:
            prior = self._vm.get(job_id)
            if prior is None or prior["attempt_index"] != 0:
                raise ValueError("attempt1 requires an absent attempt0 VM")
        if attempt_index not in self._opened_attempt_controls:
            control_prefix = self._attempt_previews[attempt_index][
                "remote_manifest"
            ]["attempt_control_prefix"]
            if self.store.list_prefix(control_prefix):
                raise FileExistsError(
                    "attempt control prefix must be exactly empty before claim"
                )
            self._opened_attempt_controls.add(attempt_index)

        authorization = self._authorization(attempt_index)
        self._publish_and_readback(
            self._authorization_uri(attempt_index), authorization
        )
        claim = {
            "schema": SIMULATION_CLAIM_SCHEMA,
            "stage_id": self.preview["stage_id"],
            "run_name": self.preview["run_name"],
            "job_id": job_id,
            "attempt_index": attempt_index,
            "preview_sha256": canonical_sha256(self.preview),
            "authorization_sha256": canonical_sha256(authorization),
            "conditional_create_generation_match": 0,
            "simulation_only": True,
            "real_vm_claim": False,
            "cloud_launch_authorized": False,
        }
        self._validate_claim(claim, job_id=job_id, attempt_index=attempt_index)
        self._publish_and_readback(self._claim_uri(job_id, attempt_index), claim)
        self._attempt_claims[key] = claim
        self._vm[job_id] = {
            "job_id": job_id,
            "attempt_index": attempt_index,
            "state": "active",
            "self_delete_requested": False,
            "preserve_on_failure": False,
            "controller_shutdown_deadline_seconds": None,
        }
        # Runner control files are retry-invariant tree objects.  They are
        # seeded once; attempt1 reads the same generations and never rewrites
        # them.
        for record in job["tree_object_manifest"][:2]:
            if not self.store.contains(record["uri"]):
                self._publish_tree_record(
                    {
                        "path": record["path"],
                        "uri": record["uri"],
                        "sha256": record["sha256"],
                        "bytes": record["bytes"],
                    }
                )
        return _copy_json(claim)

    def _validate_claim(
        self,
        value: Mapping[str, Any],
        *,
        job_id: str,
        attempt_index: int,
    ) -> None:
        row = dict(value)
        _exact(
            row,
            {
                "schema",
                "stage_id",
                "run_name",
                "job_id",
                "attempt_index",
                "preview_sha256",
                "authorization_sha256",
                "conditional_create_generation_match",
                "simulation_only",
                "real_vm_claim",
                "cloud_launch_authorized",
            },
            "fake simulation claim",
        )
        authorization = self.store.read(self._authorization_uri(attempt_index))
        if (
            row["schema"] != SIMULATION_CLAIM_SCHEMA
            or row["stage_id"] != self.preview["stage_id"]
            or row["run_name"] != self.preview["run_name"]
            or row["job_id"] != job_id
            or row["preview_sha256"] != canonical_sha256(self.preview)
            or row["authorization_sha256"] != canonical_sha256(authorization)
        ):
            raise ValueError("fake simulation claim identity changed")
        _strict_int(
            row["attempt_index"],
            "claim attempt index",
            minimum=attempt_index,
            maximum=attempt_index,
        )
        _strict_int(
            row["conditional_create_generation_match"],
            "claim generation match",
            minimum=0,
            maximum=0,
        )
        _strict_bool(row["simulation_only"], True, "claim simulation")
        _strict_bool(row["real_vm_claim"], False, "claim real VM")
        _strict_bool(
            row["cloud_launch_authorized"], False, "claim authorization"
        )

    def start_initial_exact(self) -> list[dict[str, Any]]:
        if self.attempt_index != 0:
            raise ValueError("initial exact start requires attempt0")
        return [
            self.start_attempt(job_id=job_id, attempt_index=0)
            for job_id in self.preview["selected_job_ids"]
        ]

    def _object_content(self, uri: str) -> dict[str, Any] | None:
        return self.store.read(uri) if self.store.contains(uri) else None

    def _job_prefix_content(
        self, job: Mapping[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
        uploads: list[dict[str, Any]] = []
        heartbeats: list[dict[str, Any]] = []
        pending_upload = False
        for sequence, (upload_uri, heartbeat_uri) in enumerate(
            zip(job["upload_uris"], job["heartbeat_uris"], strict=True), 1
        ):
            upload = self._object_content(upload_uri)
            heartbeat = self._object_content(heartbeat_uri)
            if upload is None:
                if heartbeat is not None:
                    raise ValueError("heartbeat exists without its upload")
                later = [
                    *job["upload_uris"][sequence:],
                    *job["heartbeat_uris"][sequence:],
                ]
                if any(self.store.contains(uri) for uri in later):
                    raise ValueError("fake worker progress is non-contiguous")
                break
            expected_upload = adapter.build_fixture_upload(
                self.preview, job_id=job["job_id"], sequence=sequence
            )
            if upload != expected_upload:
                raise ValueError("fake worker upload content changed")
            uploads.append(expected_upload)
            if heartbeat is None:
                if sequence != len(uploads):
                    raise AssertionError("unreachable pending upload position")
                later = [
                    *job["upload_uris"][sequence:],
                    *job["heartbeat_uris"][sequence:],
                ]
                if any(self.store.contains(uri) for uri in later):
                    raise ValueError("pending upload must be the trailing object")
                pending_upload = True
                break
            expected_heartbeat = adapter.build_heartbeat(
                self.preview,
                job_id=job["job_id"],
                uploads=uploads,
            )
            if heartbeat != expected_heartbeat:
                raise ValueError("fake worker heartbeat content changed")
            heartbeats.append(expected_heartbeat)
        if len(uploads) - len(heartbeats) not in (0, 1):
            raise ValueError("fake worker has more than one pending upload")
        return uploads, heartbeats, pending_upload

    def publish_upload(
        self,
        *,
        job_id: str,
        sequence: int,
        publish_heartbeat: bool = True,
    ) -> dict[str, Any]:
        self._assert_no_unknown_objects()
        if type(publish_heartbeat) is not bool:
            raise ValueError("publish_heartbeat must be a strict boolean")
        job = self._job(job_id)
        vm = self._vm.get(job_id)
        if vm is None or vm["state"] != "active":
            raise ValueError("fake worker upload requires an active VM")
        _strict_int(
            sequence,
            "upload sequence",
            minimum=1,
            maximum=len(job["work_hand_indices"]),
        )
        uploads, heartbeats, pending = self._job_prefix_content(job)
        if pending:
            if sequence != len(uploads):
                raise ValueError("pending upload must be acknowledged first")
            upload = uploads[-1]
            upload_uri = job["upload_uris"][sequence - 1]
            checked = self.store.read(upload_uri)
            if (
                upload_uri not in self._owned_objects
                or checked != upload
                or self._owned_objects[upload_uri]
                != canonical_sha256(checked)
            ):
                raise ValueError("pending upload lacks lifecycle ownership")
            result = PutResult(
                uri=upload_uri,
                generation=self.store.record(upload_uri)["generation"],
                content_sha256=canonical_sha256(checked),
                created=False,
            )
        else:
            if sequence != len(uploads) + 1:
                raise ValueError("fake worker upload sequence is not contiguous")
            upload = adapter.build_fixture_upload(
                self.preview, job_id=job_id, sequence=sequence
            )
            for tree_record in upload["tree_object_records"]:
                if not self.store.contains(tree_record["uri"]):
                    self._publish_tree_record(tree_record)
            result = self._publish_and_readback(
                job["upload_uris"][sequence - 1], upload
            )
        if publish_heartbeat:
            committed_uploads = uploads if pending else [*uploads, upload]
            heartbeat = adapter.build_heartbeat(
                self.preview,
                job_id=job_id,
                uploads=committed_uploads,
            )
            self._publish_and_readback(
                job["heartbeat_uris"][sequence - 1], heartbeat
            )
        return {
            "upload": _copy_json(upload),
            "upload_created": result.created,
            "heartbeat_published": publish_heartbeat,
            "transport_fixture_only": True,
            "scientific_payload_present": False,
        }

    def publish_remaining(self, *, job_id: str) -> None:
        job = self._job(job_id)
        while True:
            uploads, heartbeats, pending = self._job_prefix_content(job)
            if pending:
                self.publish_upload(
                    job_id=job_id,
                    sequence=len(uploads),
                    publish_heartbeat=True,
                )
            elif len(uploads) < len(job["work_hand_indices"]):
                self.publish_upload(
                    job_id=job_id,
                    sequence=len(uploads) + 1,
                    publish_heartbeat=True,
                )
            else:
                if len(uploads) != len(heartbeats):
                    raise AssertionError("complete fake prefix lost a heartbeat")
                return

    def interrupt(self, *, job_id: str) -> dict[str, Any]:
        self._assert_no_unknown_objects()
        vm = self._vm.get(job_id)
        if vm is None or vm["state"] != "active":
            raise ValueError("only an active fake VM can be interrupted")
        vm["state"] = "interrupted_waiting_controller"
        return _copy_json(vm)

    def publish_failure(
        self, *, job_id: str, shutdown_seconds: int
    ) -> dict[str, Any]:
        self._assert_no_unknown_objects()
        _strict_int(
            shutdown_seconds,
            "controller shutdown seconds",
            minimum=1,
            maximum=MAX_CONTROLLER_SHUTDOWN_SECONDS,
        )
        vm = self._vm.get(job_id)
        if vm is None or vm["state"] != "active":
            raise ValueError("fake failure requires an active VM")
        if self.store.contains(self._job(job_id)["done_uri"]):
            raise ValueError("fake failure cannot follow DONE")
        record = {
            "schema": WORKER_LIFECYCLE_SCHEMA,
            "outcome": "failure",
            "stage_id": self.preview["stage_id"],
            "run_name": self.preview["run_name"],
            "job_id": job_id,
            "attempt_index": self.attempt_index,
            "self_delete_requested": False,
            "preserve_vm_on_failure": True,
            "controller_shutdown_bounded": True,
            "controller_shutdown_deadline_seconds": shutdown_seconds,
            "transport_fixture_only": True,
            "scientific_payload_present": False,
        }
        self._publish_and_readback(
            self._lifecycle_uri(job_id, self.attempt_index), record
        )
        vm.update(
            {
                "state": "failed_preserved",
                "self_delete_requested": False,
                "preserve_on_failure": True,
                "controller_shutdown_deadline_seconds": shutdown_seconds,
            }
        )
        self._failure_records[job_id] = record
        return _copy_json(record)

    def publish_done(self, *, job_id: str) -> dict[str, Any]:
        self._assert_no_unknown_objects()
        job = self._job(job_id)
        vm = self._vm.get(job_id)
        if vm is None or vm["state"] != "active":
            raise ValueError("fake DONE requires an active VM")
        uploads, heartbeats, pending = self._job_prefix_content(job)
        if (
            pending
            or len(uploads) != len(job["work_hand_indices"])
            or len(heartbeats) != len(uploads)
        ):
            raise ValueError("fake DONE requires full artifact/heartbeat readback")
        artifact_creates_before = sum(
            event["operation"] == "conditional_create_generation_match_zero"
            and event["uri"] in {*job["upload_uris"], *job["heartbeat_uris"]}
            for event in self.store.events
        )
        done = adapter.build_done(
            self.preview,
            job_id=job_id,
            uploads=uploads,
            heartbeats=heartbeats,
        )
        for tree_record in done["tree_object_records"]:
            if not self.store.contains(tree_record["uri"]):
                # At this point only the runner's DONE.json object may be
                # missing.  Any other gap means artifact publication was not
                # durable and the DONE envelope must not be created.
                if tree_record["path"] != "DONE.json":
                    raise ValueError("fake DONE requires every runner tree object")
                self._publish_tree_record(tree_record)
            expected_tree = self._tree_fixture(tree_record)
            if self.store.read(tree_record["uri"]) != expected_tree:
                raise ValueError("fake runner tree readback changed")
        self._publish_and_readback(job["done_uri"], done)
        if self.store.read(job["done_uri"]) != done:
            raise ValueError("fake DONE readback changed")
        artifact_creates_after = sum(
            event["operation"] == "conditional_create_generation_match_zero"
            and event["uri"] in {*job["upload_uris"], *job["heartbeat_uris"]}
            for event in self.store.events
        )
        if artifact_creates_after != artifact_creates_before:
            raise AssertionError("DONE finalization republished an artifact")
        lifecycle = {
            "schema": WORKER_LIFECYCLE_SCHEMA,
            "outcome": "success",
            "stage_id": self.preview["stage_id"],
            "run_name": self.preview["run_name"],
            "job_id": job_id,
            "attempt_index": self.attempt_index,
            "self_delete_requested": True,
            "preserve_vm_on_failure": False,
            "controller_shutdown_bounded": False,
            "controller_shutdown_deadline_seconds": 0,
            "transport_fixture_only": True,
            "scientific_payload_present": False,
        }
        self._publish_and_readback(
            self._lifecycle_uri(job_id, self.attempt_index), lifecycle
        )
        vm.update(
            {
                "state": "success_self_delete_requested",
                "self_delete_requested": True,
                "preserve_on_failure": False,
                "controller_shutdown_deadline_seconds": 0,
            }
        )
        self._success_records[job_id] = lifecycle
        return _copy_json(done)

    def controller_mark_vm_absent(self, *, job_id: str) -> dict[str, Any]:
        self._assert_no_unknown_objects()
        vm = self._vm.get(job_id)
        if vm is None:
            raise ValueError("unknown fake VM")
        if vm["state"] not in (
            "interrupted_waiting_controller",
            "failed_preserved",
            "success_self_delete_requested",
        ):
            raise ValueError("controller cannot remove this fake VM state")
        vm["state"] = "absent"
        return _copy_json(vm)

    def _adapter_snapshot(self, preview: Mapping[str, Any]) -> dict[str, Any]:
        snapshot = adapter.empty_snapshot(preview)
        allowed: list[str] = []
        for job in preview["jobs"]:
            allowed.extend(job["upload_uris"])
            allowed.extend(job["heartbeat_uris"])
            allowed.append(job["done_uri"])
        allowed.append(preview["remote_manifest"]["receive_uri"])
        snapshot["objects"] = [
            self.store.record(uri) for uri in allowed if self.store.contains(uri)
        ]
        return snapshot

    def resume_attempt1(self, selected_job_ids: Sequence[str]) -> dict[str, Any]:
        self._assert_no_unknown_objects()
        if self.attempt_index != 0:
            raise ValueError("attempt1 can only follow attempt0")
        prior_preview = self.preview
        prior_snapshot = self._adapter_snapshot(prior_preview)
        validation = adapter.validate_snapshot(prior_preview, prior_snapshot)
        exact_incomplete = [
            row["job_id"]
            for row in validation["job_states"]
            if row["state"] != "success_tree_ready"
        ]
        if list(selected_job_ids) != exact_incomplete or not exact_incomplete:
            raise ValueError("attempt1 must resume the exact incomplete job set")
        for job_id in prior_preview["selected_job_ids"]:
            vm = self._vm.get(job_id)
            if vm is None or vm["state"] != "absent":
                raise ValueError("attempt1 requires every attempt0 VM absent")
        for job_id in selected_job_ids:
            if self.store.contains(self._job(job_id)["done_uri"]):
                raise ValueError("successful job must not be retried")
        preview = adapter.build_preview(
            package_dir=self._package_dir,
            stage_id=prior_preview["stage_id"],
            attempt_index=1,
            prior_preview=prior_preview,
            prior_snapshot=prior_snapshot,
            prerequisite_stage1_preview=self._stage1_preview,
            prerequisite_stage1_receive=self._stage1_receive,
        )
        self._validate_preview_boundary(preview)
        self._attempt_previews[1] = preview
        self._active_preview = preview
        if preview["remote_manifest"]["prefix"] != self._stage_prefix:
            raise ValueError("attempt1 changed the stage-stable result prefix")
        for job_id in selected_job_ids:
            self.start_attempt(job_id=job_id, attempt_index=1)
        return self.preview

    def _allowed_uris(self) -> set[str]:
        preview = self.preview
        allowed = {
            self._sentinel_uri(),
            preview["remote_manifest"]["receive_uri"],
        }
        for attempt_index in self._attempt_previews:
            allowed.add(self._authorization_uri(attempt_index))
            for job_id in preview["selected_job_ids"]:
                allowed.add(self._claim_uri(job_id, attempt_index))
                allowed.add(self._lifecycle_uri(job_id, attempt_index))
        for job in preview["jobs"]:
            allowed.update(job["upload_uris"])
            allowed.update(job["heartbeat_uris"])
            allowed.add(job["done_uri"])
            allowed.update(
                row["uri"] for row in job["tree_object_manifest"]
            )
        return allowed

    def _assert_no_unknown_objects(self) -> None:
        allowed = self._allowed_uris()
        actual = set(self.store.list_prefix(self._stage_prefix))
        unknown = sorted(actual - allowed)
        if unknown:
            raise ValueError("unknown object under diagnostic stage prefix")
        unowned = sorted(actual - set(self._owned_objects))
        if unowned:
            raise ValueError(
                "known fake remote URI lacks lifecycle ownership"
            )
        for uri in sorted(actual):
            value = self.store.read(uri)
            if self._owned_objects[uri] != canonical_sha256(value):
                raise ValueError("owned fake remote object identity changed")
        self._validate_stage_sentinel(self.store.read(self._sentinel_uri()))

    def _validate_terminal_lifecycle(self, *, require_absent: bool) -> None:
        for job in self.preview["jobs"]:
            job_id = job["job_id"]
            vm = self._vm.get(job_id)
            done_present = self.store.contains(job["done_uri"])
            success = self._success_records.get(job_id)
            failure = self._failure_records.get(job_id)
            if done_present:
                if success is None:
                    raise ValueError(
                        "DONE requires publish_done success lifecycle provenance"
                    )
                lifecycle_uri = self._lifecycle_uri(
                    job_id, success["attempt_index"]
                )
                if (
                    not self.store.contains(lifecycle_uri)
                    or self.store.read(lifecycle_uri) != success
                    or success["outcome"] != "success"
                ):
                    raise ValueError("success lifecycle record changed")
                _strict_bool(
                    success["self_delete_requested"],
                    True,
                    "success self-delete request",
                )
                _strict_bool(
                    success["preserve_vm_on_failure"],
                    False,
                    "success preserve flag",
                )
            elif success is not None:
                raise ValueError("success lifecycle exists without DONE")
            if failure is not None:
                lifecycle_uri = self._lifecycle_uri(
                    job_id, failure["attempt_index"]
                )
                if (
                    not self.store.contains(lifecycle_uri)
                    or self.store.read(lifecycle_uri) != failure
                    or failure["outcome"] != "failure"
                ):
                    raise ValueError("failure lifecycle record changed")
                _strict_bool(
                    failure["self_delete_requested"],
                    False,
                    "failure self-delete request",
                )
                _strict_bool(
                    failure["preserve_vm_on_failure"],
                    True,
                    "failure preserve flag",
                )
                _strict_bool(
                    failure["controller_shutdown_bounded"],
                    True,
                    "failure bounded shutdown",
                )
                _strict_int(
                    failure["controller_shutdown_deadline_seconds"],
                    "failure shutdown deadline",
                    minimum=1,
                    maximum=MAX_CONTROLLER_SHUTDOWN_SECONDS,
                )
            if require_absent and (vm is None or vm["state"] != "absent"):
                raise ValueError("terminal lifecycle requires VM absence")

    def assert_known_inventory(self) -> dict[str, Any]:
        self._assert_no_unknown_objects()
        self._validate_terminal_lifecycle(require_absent=False)
        actual = set(self.store.list_prefix(self._stage_prefix))
        preview = self.preview
        return {
            "schema": FAKE_SNAPSHOT_SCHEMA,
            "stage_id": preview["stage_id"],
            "attempt_index": preview["attempt_index"],
            "known_object_count": len(actual),
            "unknown_object_count": 0,
            "all_vm_absent": bool(self._vm)
            and all(row["state"] == "absent" for row in self._vm.values()),
            "simulation_only": True,
            "cloud_query_performed": False,
            "cloud_write_performed": False,
            "transport_fixture_only": True,
            "scientific_payload_present": False,
        }

    def receive(self) -> dict[str, Any]:
        if self._receive is not None:
            raise ValueError("fake stage receive is write-once")
        inventory = self.assert_known_inventory()
        if inventory["all_vm_absent"] is not True:
            raise ValueError("receive requires controller-proven VM absence")
        self._validate_terminal_lifecycle(require_absent=True)
        done_records: list[dict[str, Any]] = []
        for job in self.preview["jobs"]:
            if not self.store.contains(job["done_uri"]):
                raise ValueError("receive requires exact DONE set")
            done_records.append(self.store.read(job["done_uri"]))
        receive = adapter.build_receive(self.preview, done_records=done_records)
        self._publish_and_readback(
            self.preview["remote_manifest"]["receive_uri"], receive
        )
        self._receive = receive
        return _copy_json(receive)

    def _validate_controller_receipt(
        self, value: Mapping[str, Any]
    ) -> dict[str, Any]:
        receipt = dict(value)
        _exact(
            receipt,
            {
                "schema",
                "stage_id",
                "run_name",
                "package_manifest_sha256",
                "attempt_index",
                "selected_job_ids",
                "preview_sha256",
                "receive_sha256",
                "receive",
                "success_lifecycle_records",
                "success_lifecycle_sha256s",
                "all_worker_vms_absent",
                "vm_absence_job_ids",
                "simulation_only",
                "cloud_receive_performed",
                "performance_lock_evidence",
                "quality_evidence",
                "training_eligible",
                "promotion_evidence",
            },
            "fake controller receipt",
        )
        if self._receive is None:
            raise ValueError("fake controller receipt has no receive")
        if (
            receipt["schema"] != CONTROLLER_RECEIPT_SCHEMA
            or receipt["stage_id"] != self.preview["stage_id"]
            or receipt["run_name"] != self.preview["run_name"]
            or receipt["package_manifest_sha256"]
            != self.preview["package_manifest_sha256"]
            or receipt["selected_job_ids"] != self.preview["selected_job_ids"]
            or receipt["preview_sha256"] != canonical_sha256(self.preview)
            or receipt["receive_sha256"] != canonical_sha256(self._receive)
            or receipt["receive"] != self._receive
            or receipt["success_lifecycle_records"]
            != [
                self._success_records[job_id]
                for job_id in self.preview["selected_job_ids"]
            ]
            or receipt["success_lifecycle_sha256s"]
            != [
                canonical_sha256(self._success_records[job_id])
                for job_id in self.preview["selected_job_ids"]
            ]
            or receipt["vm_absence_job_ids"] != self.preview["selected_job_ids"]
        ):
            raise ValueError("fake controller receipt identity changed")
        _strict_int(
            receipt["attempt_index"],
            "receipt attempt index",
            minimum=self.attempt_index,
            maximum=self.attempt_index,
        )
        _strict_bool(
            receipt["all_worker_vms_absent"], True, "receipt VM absence"
        )
        _strict_bool(receipt["simulation_only"], True, "receipt simulation")
        for field in (
            "cloud_receive_performed",
            "performance_lock_evidence",
            "quality_evidence",
            "training_eligible",
            "promotion_evidence",
        ):
            _strict_bool(receipt[field], False, f"receipt {field}")
        return receipt

    def controller_proof(self) -> ControllerReceiptProof:
        if self._receive is None:
            raise ValueError("controller proof requires receive")
        inventory = self.assert_known_inventory()
        if inventory["all_vm_absent"] is not True:
            raise ValueError("controller proof requires all worker VMs absent")
        receipt = {
            "schema": CONTROLLER_RECEIPT_SCHEMA,
            "stage_id": self.preview["stage_id"],
            "run_name": self.preview["run_name"],
            "package_manifest_sha256": self.preview["package_manifest_sha256"],
            "attempt_index": self.attempt_index,
            "selected_job_ids": self.preview["selected_job_ids"],
            "preview_sha256": canonical_sha256(self.preview),
            "receive_sha256": canonical_sha256(self._receive),
            "receive": self._receive,
            "success_lifecycle_records": [
                self._success_records[job_id]
                for job_id in self.preview["selected_job_ids"]
            ],
            "success_lifecycle_sha256s": [
                canonical_sha256(self._success_records[job_id])
                for job_id in self.preview["selected_job_ids"]
            ],
            "all_worker_vms_absent": True,
            "vm_absence_job_ids": self.preview["selected_job_ids"],
            "simulation_only": True,
            "cloud_receive_performed": False,
            "performance_lock_evidence": False,
            "quality_evidence": False,
            "training_eligible": False,
            "promotion_evidence": False,
        }
        self._validate_controller_receipt(receipt)
        self._controller_receipt = receipt
        return ControllerReceiptProof(
            receipt_bytes=canonical_bytes(receipt),
            _source_stage=self,
            _issuer_nonce=self._issuer_nonce,
        )


__all__ = [
    "CONTROLLER_RECEIPT_SCHEMA",
    "ControllerReceiptProof",
    "FakeCloudWorkerStage",
    "InMemoryGenerationMatchStore",
    "MAX_ATTEMPTS",
    "MAX_CONTROLLER_SHUTDOWN_SECONDS",
    "PutResult",
    "SIMULATION_AUTHORIZATION_SCHEMA",
    "SIMULATION_CLAIM_SCHEMA",
    "STAGE_SENTINEL_SCHEMA",
]
