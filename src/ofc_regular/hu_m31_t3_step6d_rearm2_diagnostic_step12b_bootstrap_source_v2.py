"""Fresh direct-v2 bootstrap-source provision contract for Step12b.

The large VM runtime overlay and the two immutable role payload contracts are
transparent canonical JSON objects in a fresh deployment-derived source
prefix.  They are never embedded in VM metadata and never written to the old
package or result namespaces.

This module is pure unless ``provision_bootstrap_sources`` is explicitly
called with injected prefix/read/write adapters.  No live adapter is provided
here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
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
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_bootstrap_source_content_v2
    as source_content_v2,
)


RUNTIME_SOURCE_BUNDLE_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_runtime_source_bundle_v2"
)
SOURCE_PLAN_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_bootstrap_source_plan_v2"
)
SOURCE_PROVISION_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_bootstrap_source_provision_receipt_v2"
)
ROLE_BOOTSTRAP_MANIFEST_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_role_bootstrap_manifest_v2"
)

SOURCE_NAMESPACE = f"{deployment_v2.DIRECT_NAMESPACE}/bootstrap-sources"
SOURCE_OBJECT_COUNT = 3
ROLE_MANIFEST_OBJECT_COUNT = 2
MAX_RUNTIME_SOURCE_FILES = 16
MAX_SOURCE_OBJECT_BYTES = 1_048_576
IF_GENERATION_MATCH = 0

RUNTIME_SOURCE_OBJECT_PATH = "runtime_source_bundle.json"
CANDIDATE_PAYLOAD_OBJECT_PATH = "candidate_payload_contract.json"
REFERENCE_PAYLOAD_OBJECT_PATH = "reference_payload_contract.json"

REQUIRED_RUNTIME_SOURCE_PATHS = frozenset(
    {
        (
            "ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_"
            "canary_gce_transport_10c2_v1.py"
        ),
        (
            "ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_"
            "step12b_vm_prebootstrap_v2.py"
        ),
        (
            "ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_"
            "step12b_bootstrap_source_v2.py"
        ),
        (
            "ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_"
            "step12b_bootstrap_source_content_v2.py"
        ),
        (
            "ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_"
            "step12b_alias_bridge_v2.py"
        ),
        (
            "ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_"
            "step12b_pair_release_v2.py"
        ),
        (
            "ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_"
            "step12b_vm_metadata_v2.py"
        ),
        (
            "ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_"
            "step12b_external_authorization_v2.py"
        ),
        (
            "ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_"
            "step12b_deployment_contract_v2.py"
        ),
        (
            "ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_"
            "step12b_phase2_iam_plan_v2.py"
        ),
        (
            "ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_"
            "step12b_run_scoped_controller_sa_v2.py"
        ),
        (
            "ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_"
            "step11_rest_iam_admin_v1.py"
        ),
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_PATH = re.compile(r"^ofc_regular/[a-z0-9_]+\.py$")
_GENERATION = re.compile(r"^[1-9][0-9]{0,31}$")
_PROVISION_SEAL = object()


class PrefixObserver(Protocol):
    def list_objects(self, *, prefix: str) -> Sequence[str]: ...


class ConditionalSourceWriter(Protocol):
    def conditional_create(
        self,
        *,
        uri: str,
        content: bytes,
        if_generation_match: int,
    ) -> Mapping[str, Any]: ...


class GenerationPinnedSourceReader(Protocol):
    def generation_pinned_get(
        self, *, uri: str, generation: int
    ) -> bytes: ...


@dataclass(frozen=True)
class ValidatedBootstrapSourceProvision:
    deployment_contract_sha256: str
    source_plan_sha256: str
    source_prefix: str
    provision_receipt_sha256: str
    records_sha256: str
    generations_sha256: str
    _receipt_bytes: bytes = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _PROVISION_SEAL:
            raise ValueError(
                "ValidatedBootstrapSourceProvision cannot be forged"
            )

    def receipt(self) -> dict[str, Any]:
        value = json.loads(self._receipt_bytes)
        if not isinstance(value, dict):
            raise AssertionError("sealed source receipt stopped being a map")
        return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _exact(value: Mapping[str, Any], fields: set[str], label: str) -> None:
    if set(value) != fields:
        raise ValueError(f"{label} fields changed")


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or _SHA256.fullmatch(value) is None
        or value == "0" * 64
    ):
        raise ValueError(f"{label} must be a nonzero lowercase SHA-256")
    return value


def _safe_source_path(value: Any) -> str:
    if (
        not isinstance(value, str)
        or _SOURCE_PATH.fullmatch(value) is None
        or "\\" in value
        or ".." in value.split("/")
        or "site-packages" in value.lower()
        or "__pycache__" in value.lower()
    ):
        raise ValueError("runtime source path escaped overlay")
    return value


def _source_text(value: Any, path: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or "\r" in value
    ):
        raise ValueError("runtime source text changed")
    raw = value.encode("utf-8")
    try:
        compile(value, path, "exec", dont_inherit=True)
    except (SyntaxError, ValueError) as error:
        raise ValueError("runtime source did not compile") from error
    return value


def build_runtime_source_bundle(
    source_files: Mapping[str, str],
) -> dict[str, Any]:
    """Build a transparent deterministic regular-file source bundle."""

    if (
        not isinstance(source_files, Mapping)
        or set(source_files) != REQUIRED_RUNTIME_SOURCE_PATHS
        or len(source_files) > MAX_RUNTIME_SOURCE_FILES
    ):
        raise ValueError("runtime source closure changed")
    records = []
    for raw_path in sorted(source_files):
        path = _safe_source_path(raw_path)
        source = _source_text(source_files[raw_path], path)
        raw = source.encode("utf-8")
        records.append(
            {
                "path": path,
                "kind": "regular_file",
                "mode": "0644",
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "source": source,
            }
        )
    body = {
        "schema": RUNTIME_SOURCE_BUNDLE_SCHEMA,
        "file_count": len(records),
        "paths": [row["path"] for row in records],
        "records": records,
        "records_sha256": canonical_sha256(records),
        "regular_files_only": True,
        "symlink_count": 0,
        "repo_import_permitted": False,
        "site_packages_import_permitted": False,
        "private_material_present": False,
    }
    result = {
        **body,
        "runtime_source_bundle_sha256": canonical_sha256(body),
    }
    if len(canonical_bytes(result)) > MAX_SOURCE_OBJECT_BYTES:
        raise ValueError("runtime source bundle escaped source object limit")
    return result


def validate_runtime_source_bundle(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    bundle = copy.deepcopy(dict(value))
    _exact(
        bundle,
        {
            "schema",
            "file_count",
            "paths",
            "records",
            "records_sha256",
            "regular_files_only",
            "symlink_count",
            "repo_import_permitted",
            "site_packages_import_permitted",
            "private_material_present",
            "runtime_source_bundle_sha256",
        },
        "runtime source bundle",
    )
    records = bundle["records"]
    if (
        bundle["schema"] != RUNTIME_SOURCE_BUNDLE_SCHEMA
        or bundle["file_count"] != len(REQUIRED_RUNTIME_SOURCE_PATHS)
        or not isinstance(records, list)
        or len(records) != bundle["file_count"]
        or bundle["paths"] != sorted(REQUIRED_RUNTIME_SOURCE_PATHS)
        or bundle["records_sha256"] != canonical_sha256(records)
        or bundle["regular_files_only"] is not True
        or bundle["symlink_count"] != 0
        or bundle["repo_import_permitted"] is not False
        or bundle["site_packages_import_permitted"] is not False
        or bundle["private_material_present"] is not False
    ):
        raise ValueError("runtime source bundle boundary changed")
    sources: dict[str, str] = {}
    for row in records:
        if not isinstance(row, Mapping):
            raise ValueError("runtime source record changed")
        _exact(
            row,
            {
                "path",
                "kind",
                "mode",
                "bytes",
                "sha256",
                "source",
            },
            "runtime source record",
        )
        path = _safe_source_path(row["path"])
        source = _source_text(row["source"], path)
        raw = source.encode("utf-8")
        if (
            path in sources
            or row["kind"] != "regular_file"
            or row["mode"] != "0644"
            or row["bytes"] != len(raw)
            or row["sha256"] != hashlib.sha256(raw).hexdigest()
        ):
            raise ValueError("runtime source record identity changed")
        sources[path] = source
    expected = build_runtime_source_bundle(sources)
    if bundle != expected:
        raise ValueError("runtime source bundle changed")
    return expected


def materialize_runtime_source_bundle(
    value: Mapping[str, Any],
    *,
    destination: str | Path,
) -> dict[str, Any]:
    """Write only validated regular files below one fresh real directory."""

    bundle = validate_runtime_source_bundle(value)
    root = Path(destination)
    if root.exists() or root.is_symlink():
        raise FileExistsError("runtime source destination must be fresh")
    if not root.parent.is_dir() or root.parent.is_symlink():
        raise ValueError("runtime source parent must be a real directory")
    root.mkdir()
    root_resolved = root.resolve()
    written = []
    for record in bundle["records"]:
        relative = PurePosixPath(record["path"])
        target = root.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        if (
            target.parent.is_symlink()
            or root_resolved not in target.resolve().parents
            or target.exists()
            or target.is_symlink()
        ):
            raise ValueError("runtime source materialization escaped root")
        raw = record["source"].encode("utf-8")
        with target.open("xb") as handle:
            handle.write(raw)
        if (
            target.is_symlink()
            or target.read_bytes() != raw
            or hashlib.sha256(target.read_bytes()).hexdigest()
            != record["sha256"]
        ):
            raise RuntimeError("runtime source materialization changed")
        written.append(
            {
                "path": record["path"],
                "bytes": record["bytes"],
                "sha256": record["sha256"],
                "regular_file": True,
            }
        )
    body = {
        "schema": RUNTIME_SOURCE_BUNDLE_SCHEMA,
        "status": "runtime_source_bundle_materialized",
        "runtime_source_bundle_sha256": bundle[
            "runtime_source_bundle_sha256"
        ],
        "file_count": len(written),
        "records": written,
        "records_sha256": canonical_sha256(written),
        "symlink_count": 0,
        "repo_import_permitted": False,
        "site_packages_import_permitted": False,
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def _validated_deployment(
    deployment_contract: Mapping[str, Any],
    *,
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
    runtime_source_bundle: Mapping[str, Any] | None = None,
    require_bootstrap_source_content: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    deployment = deployment_v2.validate_deployment_contract(
        deployment_contract,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
    )
    candidate = payload_transport.validate_job_contract(
        candidate_payload_contract
    )
    reference = payload_transport.validate_job_contract(
        reference_payload_contract
    )
    content_binding = deployment.get("bootstrap_source_content_binding")
    if require_bootstrap_source_content and content_binding is None:
        raise ValueError(
            "deployment bootstrap source content binding is required"
        )
    if content_binding is not None:
        source_content_v2.validate_bootstrap_source_content_binding(
            content_binding,
            candidate_payload_contract=candidate,
            reference_payload_contract=reference,
        )
        if runtime_source_bundle is not None:
            source_content_v2.validate_runtime_bundle_against_content_binding(
                runtime_source_bundle,
                content_binding=content_binding,
                candidate_payload_contract=candidate,
                reference_payload_contract=reference,
            )
    return deployment, candidate, reference


def _source_prefix(deployment: Mapping[str, Any]) -> str:
    return (
        f"gs://{payload_transport.BUCKET}/{SOURCE_NAMESPACE}/"
        f"{deployment['deployment_contract_sha256']}"
    )


def _object_record(
    *,
    source_prefix: str,
    kind: str,
    path: str,
    content: Mapping[str, Any],
) -> dict[str, Any]:
    raw = canonical_bytes(content)
    if len(raw) > MAX_SOURCE_OBJECT_BYTES:
        raise ValueError("bootstrap source object escaped size limit")
    return {
        "kind": kind,
        "path": path,
        "uri": f"{source_prefix}/{path}",
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "content": copy.deepcopy(dict(content)),
    }


def _build_plan(
    *,
    deployment: Mapping[str, Any],
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    runtime_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    source_prefix = _source_prefix(deployment)
    objects = [
        _object_record(
            source_prefix=source_prefix,
            kind="shared_runtime_source_bundle",
            path=RUNTIME_SOURCE_OBJECT_PATH,
            content=runtime_bundle,
        ),
        _object_record(
            source_prefix=source_prefix,
            kind="candidate_role_payload_contract",
            path=CANDIDATE_PAYLOAD_OBJECT_PATH,
            content=candidate,
        ),
        _object_record(
            source_prefix=source_prefix,
            kind="reference_role_payload_contract",
            path=REFERENCE_PAYLOAD_OBJECT_PATH,
            content=reference,
        ),
    ]
    body = {
        "schema": SOURCE_PLAN_SCHEMA,
        "deployment_contract_sha256": deployment[
            "deployment_contract_sha256"
        ],
        "run_identity_sha256": deployment["run_identity_sha256"],
        "bootstrap_source_content_binding_sha256": deployment[
            "bootstrap_source_content_binding"
        ]["bootstrap_source_content_binding_sha256"],
        "source_prefix": source_prefix,
        "object_count": SOURCE_OBJECT_COUNT,
        "objects": objects,
        "object_summaries_sha256": canonical_sha256(
            [
                {key: row[key] for key in row if key != "content"}
                for row in objects
            ]
        ),
        "runtime_source_bundle_sha256": runtime_bundle[
            "runtime_source_bundle_sha256"
        ],
        "if_generation_match": IF_GENERATION_MATCH,
        "empty_prefix_required_before_upload": True,
        "generation_bytes_sha_readback_required": True,
        "old_package_write_authorized": False,
        "old_result_write_authorized": False,
        "direct_v2_result_write_authorized": False,
        "source_objects_are_canonical_json": True,
        "cloud_mutation_performed": False,
        "current_profile_changed": False,
    }
    return {**body, "source_plan_sha256": canonical_sha256(body)}


def build_bootstrap_source_plan(
    *,
    deployment_contract: Mapping[str, Any],
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
    runtime_source_files: Mapping[str, str],
) -> dict[str, Any]:
    bundle = build_runtime_source_bundle(runtime_source_files)
    deployment, candidate, reference = _validated_deployment(
        deployment_contract,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
        runtime_source_bundle=bundle,
        require_bootstrap_source_content=True,
    )
    return _build_plan(
        deployment=deployment,
        candidate=candidate,
        reference=reference,
        runtime_bundle=bundle,
    )


def validate_bootstrap_source_plan(
    value: Mapping[str, Any],
    *,
    deployment_contract: Mapping[str, Any],
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
) -> dict[str, Any]:
    supplied = copy.deepcopy(dict(value))
    _exact(
        supplied,
        {
            "schema",
            "deployment_contract_sha256",
            "run_identity_sha256",
            "bootstrap_source_content_binding_sha256",
            "source_prefix",
            "object_count",
            "objects",
            "object_summaries_sha256",
            "runtime_source_bundle_sha256",
            "if_generation_match",
            "empty_prefix_required_before_upload",
            "generation_bytes_sha_readback_required",
            "old_package_write_authorized",
            "old_result_write_authorized",
            "direct_v2_result_write_authorized",
            "source_objects_are_canonical_json",
            "cloud_mutation_performed",
            "current_profile_changed",
            "source_plan_sha256",
        },
        "bootstrap source plan",
    )
    plan_sha = _sha(
        supplied.pop("source_plan_sha256", None),
        "bootstrap source plan",
    )
    if canonical_sha256(supplied) != plan_sha:
        raise ValueError("bootstrap source plan digest changed")
    objects = value.get("objects")
    if not isinstance(objects, list) or len(objects) != SOURCE_OBJECT_COUNT:
        raise ValueError("bootstrap source object count changed")
    runtime_bundle = validate_runtime_source_bundle(
        objects[0].get("content", {})
        if isinstance(objects[0], Mapping)
        else {}
    )
    deployment, candidate, reference = _validated_deployment(
        deployment_contract,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
        runtime_source_bundle=runtime_bundle,
        require_bootstrap_source_content=True,
    )
    expected = _build_plan(
        deployment=deployment,
        candidate=candidate,
        reference=reference,
        runtime_bundle=runtime_bundle,
    )
    if dict(value) != expected:
        raise ValueError("bootstrap source plan changed")
    return expected


def provision_bootstrap_sources(
    *,
    source_plan: Mapping[str, Any],
    deployment_contract: Mapping[str, Any],
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
    prefix_observer: PrefixObserver,
    writer: ConditionalSourceWriter,
    reader: GenerationPinnedSourceReader,
) -> ValidatedBootstrapSourceProvision:
    """Create exactly three objects and verify exact generation readback."""

    plan = validate_bootstrap_source_plan(
        source_plan,
        deployment_contract=deployment_contract,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
    )
    observed = prefix_observer.list_objects(prefix=plan["source_prefix"])
    if (
        isinstance(observed, (str, bytes))
        or not isinstance(observed, Sequence)
        or list(observed) != []
    ):
        raise ValueError("bootstrap source prefix was not empty")
    records = []
    for source in plan["objects"]:
        content = canonical_bytes(source["content"])
        created = dict(
            writer.conditional_create(
                uri=source["uri"],
                content=content,
                if_generation_match=IF_GENERATION_MATCH,
            )
        )
        _exact(
            created,
            {
                "uri",
                "generation",
                "sha256",
                "bytes",
                "created",
            },
            "bootstrap source create receipt",
        )
        generation = created["generation"]
        if (
            created["uri"] != source["uri"]
            or type(generation) is not int
            or generation <= 0
            or created["sha256"] != source["sha256"]
            or created["bytes"] != source["bytes"]
            or created["created"] is not True
        ):
            raise ValueError("bootstrap source conditional create changed")
        readback = reader.generation_pinned_get(
            uri=source["uri"], generation=generation
        )
        if (
            not isinstance(readback, bytes)
            or readback != content
            or hashlib.sha256(readback).hexdigest() != source["sha256"]
        ):
            raise ValueError("bootstrap source readback changed")
        records.append(
            {
                "kind": source["kind"],
                "path": source["path"],
                "uri": source["uri"],
                "bytes": source["bytes"],
                "sha256": source["sha256"],
                "generation": generation,
                "created": True,
                "readback_verified": True,
            }
        )
    generations = {row["uri"]: row["generation"] for row in records}
    body = {
        "schema": SOURCE_PROVISION_RECEIPT_SCHEMA,
        "status": "fresh_direct_v2_bootstrap_sources_created_and_read_back",
        "deployment_contract_sha256": plan[
            "deployment_contract_sha256"
        ],
        "source_plan_sha256": plan["source_plan_sha256"],
        "source_prefix": plan["source_prefix"],
        "prefix_empty_before_upload": True,
        "prefix_empty_observation_sha256": canonical_sha256([]),
        "if_generation_match": IF_GENERATION_MATCH,
        "object_count": SOURCE_OBJECT_COUNT,
        "records": records,
        "records_sha256": canonical_sha256(records),
        "source_generations": generations,
        "source_generations_sha256": canonical_sha256(generations),
        "all_objects_created_once": True,
        "all_generation_bound": True,
        "all_bytes_and_sha256_read_back": True,
        "old_package_write_count": 0,
        "old_result_write_count": 0,
        "direct_v2_result_write_count": 0,
        "cloud_mutation_performed": True,
        "current_profile_changed": False,
    }
    receipt = {**body, "receipt_sha256": canonical_sha256(body)}
    return validate_bootstrap_source_provision_receipt(
        receipt,
        source_plan=plan,
        deployment_contract=deployment_contract,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
    )


def validate_bootstrap_source_provision_receipt(
    value: Mapping[str, Any],
    *,
    source_plan: Mapping[str, Any],
    deployment_contract: Mapping[str, Any],
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
) -> ValidatedBootstrapSourceProvision:
    plan = validate_bootstrap_source_plan(
        source_plan,
        deployment_contract=deployment_contract,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
    )
    receipt = copy.deepcopy(dict(value))
    expected_fields = {
        "schema",
        "status",
        "deployment_contract_sha256",
        "source_plan_sha256",
        "source_prefix",
        "prefix_empty_before_upload",
        "prefix_empty_observation_sha256",
        "if_generation_match",
        "object_count",
        "records",
        "records_sha256",
        "source_generations",
        "source_generations_sha256",
        "all_objects_created_once",
        "all_generation_bound",
        "all_bytes_and_sha256_read_back",
        "old_package_write_count",
        "old_result_write_count",
        "direct_v2_result_write_count",
        "cloud_mutation_performed",
        "current_profile_changed",
        "receipt_sha256",
    }
    _exact(receipt, expected_fields, "bootstrap source provision receipt")
    receipt_sha = _sha(
        receipt.pop("receipt_sha256", None),
        "bootstrap source provision receipt",
    )
    if canonical_sha256(receipt) != receipt_sha:
        raise ValueError("bootstrap source provision receipt digest changed")
    records = receipt["records"]
    expected_summaries = [
        {key: row[key] for key in row if key != "content"}
        for row in plan["objects"]
    ]
    if not isinstance(records, list) or len(records) != SOURCE_OBJECT_COUNT:
        raise ValueError("bootstrap source provision record count changed")
    for source, row in zip(expected_summaries, records, strict=True):
        if not isinstance(row, Mapping):
            raise ValueError("bootstrap source provision record changed")
        _exact(
            row,
            {
                "kind",
                "path",
                "uri",
                "bytes",
                "sha256",
                "generation",
                "created",
                "readback_verified",
            },
            "bootstrap source provision record",
        )
        if (
            {key: row[key] for key in source} != source
            or row["created"] is not True
            or row["readback_verified"] is not True
            or type(row["generation"]) is not int
            or row["generation"] <= 0
        ):
            raise ValueError("bootstrap source provision record changed")
    generations = {row["uri"]: row["generation"] for row in records}
    if (
        receipt["schema"] != SOURCE_PROVISION_RECEIPT_SCHEMA
        or receipt["status"]
        != "fresh_direct_v2_bootstrap_sources_created_and_read_back"
        or receipt["deployment_contract_sha256"]
        != plan["deployment_contract_sha256"]
        or receipt["source_plan_sha256"] != plan["source_plan_sha256"]
        or receipt["source_prefix"] != plan["source_prefix"]
        or receipt["prefix_empty_before_upload"] is not True
        or receipt["prefix_empty_observation_sha256"]
        != canonical_sha256([])
        or receipt["if_generation_match"] != IF_GENERATION_MATCH
        or receipt["object_count"] != SOURCE_OBJECT_COUNT
        or receipt["records_sha256"] != canonical_sha256(records)
        or receipt["source_generations"] != generations
        or receipt["source_generations_sha256"]
        != canonical_sha256(generations)
        or receipt["all_objects_created_once"] is not True
        or receipt["all_generation_bound"] is not True
        or receipt["all_bytes_and_sha256_read_back"] is not True
        or receipt["old_package_write_count"] != 0
        or receipt["old_result_write_count"] != 0
        or receipt["direct_v2_result_write_count"] != 0
        or receipt["cloud_mutation_performed"] is not True
        or receipt["current_profile_changed"] is not False
    ):
        raise ValueError("bootstrap source provision receipt changed")
    sealed = {**receipt, "receipt_sha256": receipt_sha}
    return ValidatedBootstrapSourceProvision(
        deployment_contract_sha256=plan[
            "deployment_contract_sha256"
        ],
        source_plan_sha256=plan["source_plan_sha256"],
        source_prefix=plan["source_prefix"],
        provision_receipt_sha256=receipt_sha,
        records_sha256=receipt["records_sha256"],
        generations_sha256=receipt["source_generations_sha256"],
        _receipt_bytes=canonical_bytes(sealed),
        _seal=_PROVISION_SEAL,
    )


def require_validated_bootstrap_source_provision(
    value: ValidatedBootstrapSourceProvision,
    *,
    deployment_contract_sha256: str,
    source_plan_sha256: str,
) -> ValidatedBootstrapSourceProvision:
    if (
        not isinstance(value, ValidatedBootstrapSourceProvision)
        or value._seal is not _PROVISION_SEAL
        or value.deployment_contract_sha256
        != deployment_contract_sha256
        or value.source_plan_sha256 != source_plan_sha256
    ):
        raise ValueError("validated bootstrap source provision changed")
    return value


def build_role_bootstrap_manifest(
    *,
    source_plan: Mapping[str, Any],
    validated_provision: ValidatedBootstrapSourceProvision,
    deployment_contract: Mapping[str, Any],
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
    external_job_id: str,
) -> dict[str, Any]:
    plan = validate_bootstrap_source_plan(
        source_plan,
        deployment_contract=deployment_contract,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
    )
    provision = require_validated_bootstrap_source_provision(
        validated_provision,
        deployment_contract_sha256=plan[
            "deployment_contract_sha256"
        ],
        source_plan_sha256=plan["source_plan_sha256"],
    ).receipt()
    deployment, _, _ = _validated_deployment(
        deployment_contract,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
    )
    if external_job_id not in deployment["selected_job_ids"]:
        raise ValueError("role bootstrap manifest job escaped pair")
    position = deployment["selected_job_ids"].index(external_job_id)
    source_role = deployment["source_roles"][position]
    role_kind = f"{source_role}_role_payload_contract"
    selected = [
        row
        for row in provision["records"]
        if row["kind"] in {"shared_runtime_source_bundle", role_kind}
    ]
    if len(selected) != ROLE_MANIFEST_OBJECT_COUNT:
        raise ValueError("role bootstrap source selection changed")
    body = {
        "schema": ROLE_BOOTSTRAP_MANIFEST_SCHEMA,
        "deployment_contract_sha256": deployment[
            "deployment_contract_sha256"
        ],
        "source_plan_sha256": plan["source_plan_sha256"],
        "source_provision_receipt_sha256": (
            validated_provision.provision_receipt_sha256
        ),
        "source_prefix": plan["source_prefix"],
        "external_job_id": external_job_id,
        "inner_job_id": deployment["instances"][position][
            "inner_job_id"
        ],
        "source_role": source_role,
        "object_count": ROLE_MANIFEST_OBJECT_COUNT,
        "objects": selected,
        "objects_sha256": canonical_sha256(selected),
        "runtime_source_bundle_sha256": plan[
            "runtime_source_bundle_sha256"
        ],
        "all_generation_bound": True,
        "one_role_payload_only": True,
        "opponent_role_payload_present": False,
        "repo_import_permitted": False,
        "site_packages_import_permitted": False,
    }
    return {**body, "role_manifest_sha256": canonical_sha256(body)}


def validate_role_bootstrap_manifest(
    value: Mapping[str, Any],
    *,
    source_plan: Mapping[str, Any],
    validated_provision: ValidatedBootstrapSourceProvision,
    deployment_contract: Mapping[str, Any],
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
    external_job_id: str,
) -> dict[str, Any]:
    expected = build_role_bootstrap_manifest(
        source_plan=source_plan,
        validated_provision=validated_provision,
        deployment_contract=deployment_contract,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
        external_job_id=external_job_id,
    )
    if dict(value) != expected:
        raise ValueError("role bootstrap manifest changed")
    return expected


def validate_role_bootstrap_manifest_envelope(
    value: Mapping[str, Any],
    *,
    expected_deployment_contract_sha256: str | None = None,
    expected_external_job_id: str | None = None,
    expected_source_role: str | None = None,
) -> dict[str, Any]:
    """Validate the complete generation-pinned worker manifest envelope.

    Unlike :func:`validate_role_bootstrap_manifest`, this worker-side
    validator does not require the controller's source plan or opaque
    provision capability.  Trust in the envelope still comes from the signed
    authorization; this function only proves that the signed envelope is
    internally complete, role-local, and bound to the deployment-derived
    source prefix.
    """

    manifest = copy.deepcopy(dict(value))
    _exact(
        manifest,
        {
            "schema",
            "deployment_contract_sha256",
            "source_plan_sha256",
            "source_provision_receipt_sha256",
            "source_prefix",
            "external_job_id",
            "inner_job_id",
            "source_role",
            "object_count",
            "objects",
            "objects_sha256",
            "runtime_source_bundle_sha256",
            "all_generation_bound",
            "one_role_payload_only",
            "opponent_role_payload_present",
            "repo_import_permitted",
            "site_packages_import_permitted",
            "role_manifest_sha256",
        },
        "role bootstrap manifest envelope",
    )
    supplied_manifest_sha = _sha(
        manifest.pop("role_manifest_sha256", None),
        "role bootstrap manifest",
    )
    if canonical_sha256(manifest) != supplied_manifest_sha:
        raise ValueError("role bootstrap manifest digest changed")

    deployment_sha = _sha(
        manifest["deployment_contract_sha256"],
        "role bootstrap deployment",
    )
    source_role = manifest["source_role"]
    external_job_id = manifest["external_job_id"]
    inner_job_id = manifest["inner_job_id"]
    objects = manifest["objects"]
    if (
        manifest["schema"] != ROLE_BOOTSTRAP_MANIFEST_SCHEMA
        or (
            expected_deployment_contract_sha256 is not None
            and deployment_sha != expected_deployment_contract_sha256
        )
        or source_role not in {"candidate", "reference"}
        or (
            expected_source_role is not None
            and source_role != expected_source_role
        )
        or not isinstance(external_job_id, str)
        or not external_job_id
        or (
            expected_external_job_id is not None
            and external_job_id != expected_external_job_id
        )
        or not isinstance(inner_job_id, str)
        or not inner_job_id
        or manifest["source_prefix"]
        != (
            f"gs://{payload_transport.BUCKET}/{SOURCE_NAMESPACE}/"
            f"{deployment_sha}"
        )
        or manifest["object_count"] != ROLE_MANIFEST_OBJECT_COUNT
        or not isinstance(objects, list)
        or len(objects) != ROLE_MANIFEST_OBJECT_COUNT
        or manifest["objects_sha256"] != canonical_sha256(objects)
        or manifest["all_generation_bound"] is not True
        or manifest["one_role_payload_only"] is not True
        or manifest["opponent_role_payload_present"] is not False
        or manifest["repo_import_permitted"] is not False
        or manifest["site_packages_import_permitted"] is not False
    ):
        raise ValueError("role bootstrap manifest boundary changed")
    _sha(manifest["source_plan_sha256"], "bootstrap source plan")
    _sha(
        manifest["source_provision_receipt_sha256"],
        "bootstrap source provision receipt",
    )
    _sha(
        manifest["runtime_source_bundle_sha256"],
        "runtime source bundle",
    )

    expected_kinds_and_paths = [
        ("shared_runtime_source_bundle", RUNTIME_SOURCE_OBJECT_PATH),
        (
            f"{source_role}_role_payload_contract",
            (
                CANDIDATE_PAYLOAD_OBJECT_PATH
                if source_role == "candidate"
                else REFERENCE_PAYLOAD_OBJECT_PATH
            ),
        ),
    ]
    seen_uris: set[str] = set()
    for row, (expected_kind, expected_path) in zip(
        objects, expected_kinds_and_paths, strict=True
    ):
        if not isinstance(row, Mapping):
            raise ValueError("role bootstrap manifest object changed")
        _exact(
            row,
            {
                "kind",
                "path",
                "uri",
                "bytes",
                "sha256",
                "generation",
                "created",
                "readback_verified",
            },
            "role bootstrap manifest object",
        )
        expected_uri = f"{manifest['source_prefix']}/{expected_path}"
        if (
            row["kind"] != expected_kind
            or row["path"] != expected_path
            or row["uri"] != expected_uri
            or row["uri"] in seen_uris
            or type(row["bytes"]) is not int
            or not 1 <= row["bytes"] <= MAX_SOURCE_OBJECT_BYTES
            or type(row["generation"]) is not int
            or row["generation"] <= 0
            or row["created"] is not True
            or row["readback_verified"] is not True
        ):
            raise ValueError("role bootstrap manifest object changed")
        _sha(row["sha256"], "role bootstrap source object")
        seen_uris.add(row["uri"])
    return {**manifest, "role_manifest_sha256": supplied_manifest_sha}


__all__ = [
    "CANDIDATE_PAYLOAD_OBJECT_PATH",
    "ConditionalSourceWriter",
    "GenerationPinnedSourceReader",
    "IF_GENERATION_MATCH",
    "MAX_SOURCE_OBJECT_BYTES",
    "PrefixObserver",
    "REFERENCE_PAYLOAD_OBJECT_PATH",
    "REQUIRED_RUNTIME_SOURCE_PATHS",
    "ROLE_BOOTSTRAP_MANIFEST_SCHEMA",
    "ROLE_MANIFEST_OBJECT_COUNT",
    "RUNTIME_SOURCE_BUNDLE_SCHEMA",
    "RUNTIME_SOURCE_OBJECT_PATH",
    "SOURCE_NAMESPACE",
    "SOURCE_OBJECT_COUNT",
    "SOURCE_PLAN_SCHEMA",
    "SOURCE_PROVISION_RECEIPT_SCHEMA",
    "ValidatedBootstrapSourceProvision",
    "build_bootstrap_source_plan",
    "build_role_bootstrap_manifest",
    "build_runtime_source_bundle",
    "canonical_bytes",
    "canonical_sha256",
    "materialize_runtime_source_bundle",
    "provision_bootstrap_sources",
    "require_validated_bootstrap_source_provision",
    "validate_bootstrap_source_plan",
    "validate_bootstrap_source_provision_receipt",
    "validate_role_bootstrap_manifest",
    "validate_role_bootstrap_manifest_envelope",
    "validate_runtime_source_bundle",
]
