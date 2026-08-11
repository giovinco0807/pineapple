"""Fail-closed binding from same-Linux closeout to dataset production.

The fresh-quality run name is deliberately not inferred from a directory
name.  It is read from the immutable fresh-quality GCP plan retained by the
same-Linux closeout.  Likewise, the dataset run name is read from the
source-replayed transport plan.  A production binding is issued only when:

* the caller supplies the exact closeout, fresh-quality-plan, and smoke-gate
  file digests;
* the same-Linux closeout is terminal and cloud execution has not started;
* the retained fresh-quality plan has the explicitly expected run name;
* the 25-paired smoke gate source-replays as a pass; and
* the prepared controller and transport identities agree exactly.

This module performs no cloud mutation and never changes ``current``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from . import hu_m31_t3_dataset_contract_v1 as dataset
from . import hu_m31_t3_dataset_gcp_controller_v1 as controller
from . import hu_m31_t3_dataset_gcp_transport_v1 as transport
from . import hu_m31_t3_same_linux_closeout_v1 as closeout
from . import hu_m31_t3_step6d_fresh_quality_gate_v1 as quality_gate
from . import hu_m31_t3_step6d_fresh_quality_gcp_bridge_v1 as quality_bridge


SOURCE_BINDING_SCHEMA = "hu_m31_t3_dataset_source_binding_v1"
SOURCE_BINDING_STATUS = (
    "qualified_same_linux_closeout_and_25_pair_smoke_bound_cloud_not_started"
)

_SHA = re.compile(r"^[0-9a-f]{64}$")
_RUN = re.compile(r"^[a-z][a-z0-9-]{2,62}$")
_RECORD_KEYS = frozenset({"path", "sha256", "bytes"})
_READY_KEYS = frozenset(
    {
        "schema",
        "status",
        "output_root",
        "fresh_quality_final_receipt",
        "fresh_quality_gate",
        "dataset_smoke_gate",
        "portable_authorization",
        "fanout_bundle_manifest",
        "quality_job_count",
        "dataset_smoke_pair_count",
        "fanout_shard_count",
        "performance_receipt_validation_mode",
        "performance_source_paths_dereferenced",
        "fresh_quality_worker_vm_reuse_allowed",
        "dedicated_controller_context",
        "cloud_called",
        "cloud_execution_started",
        "full_fanout_started",
        "training_eligible",
        "current_profile_changed",
    }
)
_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "status",
        "required_restore_root",
        "restore_contract",
        "source_paths_all_within_restore_root",
        "portable_authorization_sha256",
        "transport_plan_sha256",
        "dataset_plan_sha256",
        "smoke_shard_id",
        "completed_smoke_pair_count",
        "fanout_shard_count",
        "bundle_files",
        "bundle_file_aggregate_sha256",
        "persistent_source_closure_files",
        "persistent_source_closure_sha256",
        "performance_receipt_validation",
        "performance_source_paths_dereferenced",
        "fresh_quality_worker_vm_reuse_allowed",
        "dedicated_controller_context_required",
        "controller_cloud_mutation_authorized",
        "controller_cleanup_condition",
        "hidden_information_field_count",
        "opponent_private_discards_used",
        "teacher_values_are_realized_match_ev",
        "training_eligible",
        "current_profile_changed",
    }
)
_BINDING_KEYS = frozenset(
    {
        "schema",
        "status",
        "fresh_quality_run_name",
        "dataset_run_name",
        "same_linux_closeout_path",
        "same_linux_closeout_file_sha256",
        "fresh_quality_plan_path",
        "fresh_quality_plan_file_sha256",
        "fresh_quality_plan_sha256",
        "dataset_smoke_gate_path",
        "dataset_smoke_gate_file_sha256",
        "dataset_smoke_gate_sha256",
        "controller_root",
        "controller_contract_sha256",
        "transport_plan_sha256",
        "execution_identity_sha256",
        "dataset_plan_sha256",
        "portable_authorization_sha256",
        "qualified_same_linux_closeout",
        "fresh_quality_gate_passed",
        "dataset_smoke_gate_passed",
        "dataset_smoke_pair_count",
        "full_fanout_authorized_after_smoke_only",
        "cloud_mutated",
        "cloud_execution_started",
        "full_fanout_started",
        "training_eligible",
        "current_profile_changed",
        "binding_sha256",
    }
)


def canonical_bytes(value: Any) -> bytes:
    return controller.canonical_bytes(value)


def canonical_sha256(value: Any) -> str:
    return controller.canonical_sha256(value)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _run(value: Any, label: str) -> str:
    if not isinstance(value, str) or _RUN.fullmatch(value) is None:
        raise ValueError(f"{label} is invalid")
    return value


def _plain_file(path: str | Path, label: str) -> Path:
    source = Path(path)
    if not source.is_absolute() or source.is_symlink() or not source.is_file():
        raise ValueError(f"{label} must be an absolute regular file")
    return source.resolve()


def _plain_directory(path: str | Path, label: str) -> Path:
    source = Path(path)
    if not source.is_absolute() or source.is_symlink() or not source.is_dir():
        raise ValueError(f"{label} must be an absolute regular directory")
    return source.resolve()


def _read_canonical(path: str | Path, label: str) -> dict[str, Any]:
    source = _plain_file(path, label)
    raw = source.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} field set changed")


def _record_file(base: Path, record: Mapping[str, Any], label: str) -> Path:
    _exact_keys(record, _RECORD_KEYS, f"{label} record")
    relative = record.get("path")
    if not isinstance(relative, str):
        raise ValueError(f"{label} path changed")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or "\\" in relative
        or pure.as_posix() != relative
    ):
        raise ValueError(f"{label} path is unsafe")
    source = base.joinpath(*pure.parts)
    if (
        source.is_symlink()
        or not source.is_file()
        or not source.resolve().is_relative_to(base)
        or source.stat().st_size != record.get("bytes")
        or _file_sha256(source) != _sha(record.get("sha256"), f"{label} SHA")
    ):
        raise ValueError(f"{label} file binding changed")
    return source.resolve()


def _record_by_schema(
    *,
    base: Path,
    records: list[Mapping[str, Any]],
    schema: str,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for index, record in enumerate(records):
        relative = record.get("path")
        if not isinstance(relative, str) or PurePosixPath(relative).suffix != ".json":
            continue
        path = _record_file(base, record, f"{label} candidate {index}")
        try:
            value = _read_canonical(path, f"{label} candidate {index}")
        except ValueError:
            continue
        if value.get("schema") == schema:
            matches.append((path, value))
    if len(matches) != 1:
        raise ValueError(f"{label} must occur exactly once")
    return matches[0]


def _content_path(plan: Mapping[str, Any], kind: str, closeout_root: Path) -> Path:
    rows = [
        row
        for row in plan.get("content_sources", [])
        if isinstance(row, Mapping) and row.get("kind") == kind
    ]
    if len(rows) != 1:
        raise ValueError(f"transport content kind changed: {kind}")
    path = _plain_file(rows[0].get("source_path", ""), kind)
    if not path.is_relative_to(closeout_root):
        raise ValueError(f"transport content escaped closeout: {kind}")
    if path.stat().st_size != rows[0].get("bytes") or _file_sha256(path) != rows[0].get(
        "sha256"
    ):
        raise ValueError(f"transport content digest changed: {kind}")
    return path


def _inspect_closeout(
    *,
    closeout_ready_path: str | Path,
    expected_closeout_file_sha256: str,
    expected_fresh_quality_plan_file_sha256: str,
    expected_smoke_gate_file_sha256: str,
    expected_fresh_quality_run_name: str,
    expected_dataset_run_name: str,
) -> dict[str, Any]:
    expected_closeout_sha = _sha(
        expected_closeout_file_sha256, "expected closeout file SHA"
    )
    expected_quality_plan_sha = _sha(
        expected_fresh_quality_plan_file_sha256,
        "expected fresh-quality plan file SHA",
    )
    expected_smoke_sha = _sha(
        expected_smoke_gate_file_sha256, "expected smoke-gate file SHA"
    )
    quality_run_name = _run(
        expected_fresh_quality_run_name, "expected fresh-quality run name"
    )
    dataset_run_name = _run(expected_dataset_run_name, "expected dataset run name")

    ready_path = _plain_file(closeout_ready_path, "same-Linux closeout")
    if _file_sha256(ready_path) != expected_closeout_sha:
        raise ValueError("same-Linux closeout file SHA changed")
    ready = _read_canonical(ready_path, "same-Linux closeout")
    _exact_keys(ready, _READY_KEYS, "same-Linux closeout")
    root = _plain_directory(ready_path.parent, "same-Linux closeout root")
    if (
        ready.get("schema") != closeout.CLOSEOUT_SCHEMA
        or ready.get("status")
        != "complete_same_linux_ready_359_shard_fanout_not_started"
        or ready.get("output_root") != str(root)
        or ready.get("quality_job_count") != 15
        or ready.get("dataset_smoke_pair_count") != dataset.SHARD_PAIR_COUNT
        or ready.get("fanout_shard_count") != transport.CLOUD_SHARD_COUNT
        or ready.get("performance_source_paths_dereferenced") is not False
        or ready.get("fresh_quality_worker_vm_reuse_allowed") is not False
        or ready.get("dedicated_controller_context") is not True
        or any(
            ready.get(field) is not False
            for field in (
                "cloud_called",
                "cloud_execution_started",
                "full_fanout_started",
                "training_eligible",
                "current_profile_changed",
            )
        )
    ):
        raise PermissionError("same-Linux closeout is not production-ready")

    manifest_path = _record_file(
        root,
        ready["fanout_bundle_manifest"],
        "fanout bundle manifest",
    )
    manifest = _read_canonical(manifest_path, "fanout bundle manifest")
    _exact_keys(manifest, _MANIFEST_KEYS, "fanout bundle manifest")
    if (
        manifest.get("schema") != closeout.FANOUT_BUNDLE_SCHEMA
        or manifest.get("status")
        != "portable_359_shard_authorization_ready_cloud_not_started"
        or manifest.get("restore_contract")
        != "same_persistent_disk_exact_absolute_mount_only"
        or manifest.get("source_paths_all_within_restore_root") is not True
        or manifest.get("smoke_shard_id") != dataset.SMOKE_SHARD_ID
        or manifest.get("completed_smoke_pair_count") != dataset.SHARD_PAIR_COUNT
        or manifest.get("fanout_shard_count") != transport.CLOUD_SHARD_COUNT
        or manifest.get("performance_source_paths_dereferenced") is not False
        or manifest.get("fresh_quality_worker_vm_reuse_allowed") is not False
        or manifest.get("dedicated_controller_context_required") is not True
        or manifest.get("controller_cloud_mutation_authorized") is not False
        or manifest.get("hidden_information_field_count") != 0
        or manifest.get("opponent_private_discards_used") is not False
        or manifest.get("teacher_values_are_realized_match_ev") is not False
        or manifest.get("training_eligible") is not False
        or manifest.get("current_profile_changed") is not False
    ):
        raise PermissionError("fanout bundle is not a safe same-Linux closeout")

    bundle_records = manifest.get("bundle_files")
    source_records = manifest.get("persistent_source_closure_files")
    if not isinstance(bundle_records, list) or not all(
        isinstance(row, Mapping) for row in bundle_records
    ):
        raise ValueError("fanout bundle file records changed")
    if not isinstance(source_records, list) or not all(
        isinstance(row, Mapping) for row in source_records
    ):
        raise ValueError("persistent source closure records changed")
    if canonical_sha256(bundle_records) != manifest.get(
        "bundle_file_aggregate_sha256"
    ) or canonical_sha256(source_records) != manifest.get(
        "persistent_source_closure_sha256"
    ):
        raise ValueError("same-Linux closeout aggregate digest changed")
    for index, record in enumerate(bundle_records):
        _record_file(root, record, f"fanout bundle file {index}")

    controller_filesystem = _plain_directory(
        manifest.get("required_restore_root", ""),
        "required same-Linux restore root",
    )
    if not root.is_relative_to(controller_filesystem):
        raise ValueError("closeout root escaped required restore root")
    for index, record in enumerate(source_records):
        _record_file(
            controller_filesystem,
            record,
            f"persistent source closure file {index}",
        )

    fresh_plan_path, fresh_plan = _record_by_schema(
        base=controller_filesystem,
        records=source_records,
        schema=quality_bridge.PLAN_SCHEMA,
        label="fresh-quality GCP plan",
    )
    if (
        _file_sha256(fresh_plan_path) != expected_quality_plan_sha
        or fresh_plan.get("plan_sha256")
        != quality_bridge._plan_digest(fresh_plan)  # type: ignore[attr-defined]
        or fresh_plan.get("status")
        != "immutable_quality_cloud_plan_ready_not_authorized"
        or fresh_plan.get("run_name") != quality_run_name
        or fresh_plan.get("cloud_launch_authorized") is not False
        or fresh_plan.get("cloud_execution_started") is not False
        or fresh_plan.get("training_eligible") is not False
        or fresh_plan.get("current_profile_changed") is not False
    ):
        raise PermissionError(
            "fresh-quality run identity or immutable plan digest changed"
        )

    transport_path, transport_raw = _record_by_schema(
        base=root,
        records=bundle_records,
        schema=transport.TRANSPORT_PLAN_SCHEMA,
        label="dataset transport plan",
    )
    transport_plan = transport.validate_transport_plan(
        transport_raw, replay_sources=True
    )
    if (
        transport_plan.get("run_name") != dataset_run_name
        or transport_plan.get("plan_sha256") != manifest.get("transport_plan_sha256")
        or transport_plan.get("cloud_execution_started") is not False
        or transport_plan.get("quality_and_smoke_source_replayed") is not True
        or transport_plan.get("cloud_launch_authorized") is not True
        or transport_plan.get("current_profile_changed") is not False
    ):
        raise PermissionError("dataset transport identity or gate changed")

    contract_path, contract_raw = _record_by_schema(
        base=root,
        records=bundle_records,
        schema=controller.CONTROLLER_CONTRACT_SCHEMA,
        label="dataset controller contract",
    )
    contract = controller._validate_contract(  # type: ignore[attr-defined]
        contract_raw, transport_plan=transport_plan
    )
    controller_root = _plain_directory(
        contract_path.parent, "prepared dataset controller root"
    )
    if (
        controller_root != transport_path.parent
        or controller_root != (root / "controller").resolve()
        or contract.get("run_name") != dataset_run_name
        or contract.get("quality_and_smoke_source_replayed") is not True
        or contract.get("full_fanout_authorized_after_smoke_only") is not True
        or contract.get("current_profile_changed") is not False
    ):
        raise PermissionError("prepared controller is not bound to this closeout")

    dataset_plan_path = _content_path(transport_plan, "dataset_plan", root)
    dataset_plan = dataset.validate_dataset_plan(
        _read_canonical(dataset_plan_path, "dataset plan")
    )
    smoke_path = _content_path(transport_plan, "smoke_gate", root)
    if (
        smoke_path
        != _record_file(root, ready["dataset_smoke_gate"], "dataset smoke gate")
        or _file_sha256(smoke_path) != expected_smoke_sha
    ):
        raise ValueError("expected 25-pair smoke-gate digest changed")
    smoke = _read_canonical(smoke_path, "dataset smoke gate")
    smoke_shard_directory = _plain_directory(
        transport_plan["source_paths"]["smoke_shard_directory"],
        "completed smoke shard",
    )
    if not smoke_shard_directory.is_relative_to(root):
        raise ValueError("completed smoke shard escaped closeout")
    dataset.validate_smoke_gate_receipt(
        smoke,
        plan=dataset_plan,
        smoke_shard_directory=smoke_shard_directory,
    )
    metrics = smoke.get("metrics")
    if (
        smoke.get("schema") != dataset.SMOKE_GATE_SCHEMA
        or smoke.get("status") != "pass"
        or smoke.get("decision") != "open_remaining_8975_paired_fanout"
        or smoke.get("all_gates_passed") is not True
        or smoke.get("full_9000_paired_fanout_authorized") is not True
        or smoke.get("current_profile_changed") is not False
        or not isinstance(metrics, Mapping)
        or metrics.get("paired_hand_count") != dataset.SHARD_PAIR_COUNT
        or metrics.get("root_count") != dataset.SHARD_PAIR_COUNT * 2
        or metrics.get("seat_counts")
        != {
            "first": dataset.SHARD_PAIR_COUNT,
            "second": dataset.SHARD_PAIR_COUNT,
        }
    ):
        raise PermissionError("25-paired smoke gate did not pass exactly")

    fresh_gate_path = _content_path(transport_plan, "fresh_quality_gate", root)
    if fresh_gate_path != _record_file(
        root, ready["fresh_quality_gate"], "fresh-quality gate"
    ):
        raise ValueError("fresh-quality gate record changed")
    fresh_gate = quality_gate.validate_fresh_quality_gate_value(
        _read_canonical(fresh_gate_path, "fresh-quality gate"),
        replay_sources=True,
    )
    if (
        fresh_gate.get("status") != "pass"
        or fresh_gate.get("all_gates_passed") is not True
        or fresh_gate.get("data_pilot_25_paired_authorized") is not True
        or fresh_gate.get("full_9000_paired_fanout_authorized") is not False
        or fresh_gate.get("current_profile_changed") is not False
    ):
        raise PermissionError("fresh-quality gate did not open only the smoke")

    final_path = _record_file(
        root,
        ready["fresh_quality_final_receipt"],
        "fresh-quality final receipt",
    )
    final_receipt = quality_bridge.validate_final_receipt(
        _read_canonical(final_path, "fresh-quality final receipt")
    )
    if (
        final_receipt.get("status") != "qualified"
        or final_receipt.get("accepted_job_count") != 15
        or final_receipt.get("data_pilot_25_paired_authorized") is not True
        or final_receipt.get("full_9000_paired_fanout_authorized") is not False
        or final_receipt.get("current_profile_changed") is not False
    ):
        raise PermissionError("fresh-quality final receipt is not qualified")

    portable_path = _content_path(transport_plan, "portable_authorization", root)
    if portable_path != _record_file(
        root, ready["portable_authorization"], "portable authorization"
    ):
        raise ValueError("portable authorization record changed")
    if (
        transport_plan["source_identity"]["dataset_plan_sha256"]
        != manifest["dataset_plan_sha256"]
        or transport_plan["source_identity"]["portable_authorization_sha256"]
        != manifest["portable_authorization_sha256"]
    ):
        raise ValueError("transport/manifest source identity changed")

    return {
        "schema": SOURCE_BINDING_SCHEMA,
        "status": SOURCE_BINDING_STATUS,
        "fresh_quality_run_name": quality_run_name,
        "dataset_run_name": dataset_run_name,
        "same_linux_closeout_path": str(ready_path),
        "same_linux_closeout_file_sha256": expected_closeout_sha,
        "fresh_quality_plan_path": str(fresh_plan_path),
        "fresh_quality_plan_file_sha256": expected_quality_plan_sha,
        "fresh_quality_plan_sha256": fresh_plan["plan_sha256"],
        "dataset_smoke_gate_path": str(smoke_path),
        "dataset_smoke_gate_file_sha256": expected_smoke_sha,
        "dataset_smoke_gate_sha256": dataset.canonical_sha256(smoke),
        "controller_root": str(controller_root),
        "controller_contract_sha256": contract["contract_sha256"],
        "transport_plan_sha256": transport_plan["plan_sha256"],
        "execution_identity_sha256": transport_plan["execution_identity_sha256"],
        "dataset_plan_sha256": manifest["dataset_plan_sha256"],
        "portable_authorization_sha256": manifest["portable_authorization_sha256"],
        "qualified_same_linux_closeout": True,
        "fresh_quality_gate_passed": True,
        "dataset_smoke_gate_passed": True,
        "dataset_smoke_pair_count": dataset.SHARD_PAIR_COUNT,
        "full_fanout_authorized_after_smoke_only": True,
        "cloud_mutated": False,
        "cloud_execution_started": False,
        "full_fanout_started": False,
        "training_eligible": False,
        "current_profile_changed": False,
    }


def build_source_binding(
    *,
    closeout_ready_path: str | Path,
    expected_closeout_file_sha256: str,
    expected_fresh_quality_plan_file_sha256: str,
    expected_smoke_gate_file_sha256: str,
    expected_fresh_quality_run_name: str,
    expected_dataset_run_name: str,
) -> dict[str, Any]:
    core = _inspect_closeout(
        closeout_ready_path=closeout_ready_path,
        expected_closeout_file_sha256=expected_closeout_file_sha256,
        expected_fresh_quality_plan_file_sha256=(
            expected_fresh_quality_plan_file_sha256
        ),
        expected_smoke_gate_file_sha256=expected_smoke_gate_file_sha256,
        expected_fresh_quality_run_name=expected_fresh_quality_run_name,
        expected_dataset_run_name=expected_dataset_run_name,
    )
    return {**core, "binding_sha256": canonical_sha256(core)}


def validate_source_binding(
    value: Mapping[str, Any],
    *,
    expected_dataset_run_name: str | None = None,
    expected_controller_root: str | Path | None = None,
) -> dict[str, Any]:
    binding = deepcopy(dict(value))
    _exact_keys(binding, _BINDING_KEYS, "dataset source binding")
    if (
        binding.get("schema") != SOURCE_BINDING_SCHEMA
        or binding.get("status") != SOURCE_BINDING_STATUS
        or binding.get("binding_sha256")
        != canonical_sha256(
            {key: value for key, value in binding.items() if key != "binding_sha256"}
        )
    ):
        raise ValueError("dataset source-binding digest or status changed")
    replayed = build_source_binding(
        closeout_ready_path=binding["same_linux_closeout_path"],
        expected_closeout_file_sha256=binding["same_linux_closeout_file_sha256"],
        expected_fresh_quality_plan_file_sha256=binding[
            "fresh_quality_plan_file_sha256"
        ],
        expected_smoke_gate_file_sha256=binding["dataset_smoke_gate_file_sha256"],
        expected_fresh_quality_run_name=binding["fresh_quality_run_name"],
        expected_dataset_run_name=binding["dataset_run_name"],
    )
    if replayed != binding:
        raise ValueError("dataset source binding differs from source replay")
    if (
        expected_dataset_run_name is not None
        and binding["dataset_run_name"] != expected_dataset_run_name
    ):
        raise PermissionError("dataset source-binding run name mismatch")
    if expected_controller_root is not None:
        expected_root = Path(expected_controller_root)
        if not expected_root.is_absolute() or expected_root.resolve() != Path(
            binding["controller_root"]
        ):
            raise PermissionError("dataset source-binding controller mismatch")
    return binding


def validate_source_binding_file(
    path: str | Path,
    *,
    expected_dataset_run_name: str | None = None,
    expected_controller_root: str | Path | None = None,
) -> dict[str, Any]:
    return validate_source_binding(
        _read_canonical(path, "dataset source binding"),
        expected_dataset_run_name=expected_dataset_run_name,
        expected_controller_root=expected_controller_root,
    )


def write_source_binding(
    *,
    output_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    binding = build_source_binding(**kwargs)
    target = Path(output_path)
    if not target.is_absolute():
        raise ValueError("dataset source-binding output must be absolute")
    raw = canonical_bytes(binding)
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != raw:
            raise FileExistsError(
                f"immutable dataset source binding conflicts: {target}"
            )
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return validate_source_binding_file(target)


__all__ = [
    "SOURCE_BINDING_SCHEMA",
    "SOURCE_BINDING_STATUS",
    "build_source_binding",
    "canonical_bytes",
    "canonical_sha256",
    "validate_source_binding",
    "validate_source_binding_file",
    "write_source_binding",
]
