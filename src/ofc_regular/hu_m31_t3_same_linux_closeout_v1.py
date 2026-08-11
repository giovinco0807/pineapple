"""Same-Linux closeout from fresh quality to the M3.1 dataset fanout.

This module owns no cloud mutation.  It is the local, resumable bridge that is
run on one dedicated Linux controller filesystem after both fresh-quality
waves have been cleaned up and received:

* replay all 15 accepted quality jobs;
* rebuild durable quality roots under the controller root;
* publish a source-replayable fresh-quality gate;
* run/resume the sole 25-paired ``train-0000`` dataset smoke shard;
* publish its smoke gate;
* build the content-addressed 359-shard transport/controller contract; and
* write ``SAME_LINUX_CLOSEOUT_READY.json`` last.

Every absolute path retained by the quality gate or dataset transport plan is
inside ``controller_filesystem_root``.  The exported controller filesystem
therefore remains valid only when its persistent disk is restored at the exact
same absolute mount.  Workers consume the existing content-addressed portable
authorization and never need the controller-local quality paths.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import zipfile
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from . import hu_m31_t3_dataset_contract_v1 as dataset
from . import hu_m31_t3_dataset_gcp_controller_v1 as dataset_controller
from . import hu_m31_t3_dataset_gcp_transport_v1 as dataset_transport
from . import hu_m31_t3_dataset_local_pilot_v1 as local_pilot
from . import hu_m31_t3_performance_lock_v4_portable_receipt_v1 as performance_portable
from . import hu_m31_t3_step6d_fresh_quality_gate_v1 as quality_gate
from . import hu_m31_t3_step6d_fresh_quality_gcp_bridge_v1 as quality_bridge
from . import hu_m31_t3_step6d_fresh_quality_local_staging_v1 as quality_staging
from . import hu_m31_t3_step6d_fresh_quality_transport_v1 as quality_transport
from . import hu_m31_t3_step6d_fresh_quality_v1 as quality


CLOSEOUT_SCHEMA = "hu_m31_t3_same_linux_closeout_v1"
FANOUT_BUNDLE_SCHEMA = "hu_m31_t3_fanout_controller_bundle_v1"
READY_NAME = "SAME_LINUX_CLOSEOUT_READY.json"
FANOUT_MANIFEST_NAME = "fanout_bundle_manifest.json"

_QUALITY_DIRECTORY = "quality_sources"
_DATASET_DIRECTORY = "dataset"
_FANOUT_DIRECTORY = "fanout_sources"
_CONTROLLER_DIRECTORY = "controller"
_LOCAL_SHARDS_DIRECTORY = "fanout_shards"

_ALLOWED_ROOT_ENTRIES = frozenset(
    {
        _QUALITY_DIRECTORY,
        _DATASET_DIRECTORY,
        _FANOUT_DIRECTORY,
        _CONTROLLER_DIRECTORY,
        _LOCAL_SHARDS_DIRECTORY,
        FANOUT_MANIFEST_NAME,
        READY_NAME,
    }
)
_REQUIRED_DATASET_RUNTIME_PATHS = frozenset(
    {
        "src/ofc_regular/hu_m31_t3_dataset_contract_v1.py",
        "src/ofc_regular/hu_m31_t3_dataset_executor_v1.py",
        "src/ofc_regular/hu_m31_t3_dataset_portable_worker_v1.py",
    }
)

PilotRunner = Callable[..., Mapping[str, Any]]
ControllerPrepare = Callable[..., Mapping[str, Any]]


def canonical_bytes(value: Any) -> bytes:
    return dataset.canonical_bytes(value)


def canonical_sha256(value: Any) -> str:
    return dataset.canonical_sha256(value)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_link_or_junction(path: Path) -> bool:
    return path.is_symlink() or (
        hasattr(path, "is_junction") and path.is_junction()
    )


def _plain_file(path: str | Path, label: str) -> Path:
    source = Path(path)
    if (
        not source.is_absolute()
        or _is_link_or_junction(source)
        or not source.is_file()
    ):
        raise ValueError(f"{label} must be an absolute regular file")
    return source.resolve()


def _plain_directory(path: str | Path, label: str) -> Path:
    source = Path(path)
    if (
        not source.is_absolute()
        or _is_link_or_junction(source)
        or not source.is_dir()
    ):
        raise ValueError(f"{label} must be an absolute regular directory")
    return source.resolve()


def _safe_root(path: str | Path) -> Path:
    root = Path(path)
    if not root.is_absolute():
        raise ValueError("same-Linux closeout root must be absolute")
    root = root.resolve()
    parent = root.parent
    if (
        _is_link_or_junction(parent)
        or not parent.is_dir()
        or any(_is_link_or_junction(item) for item in parent.parents)
    ):
        raise ValueError("same-Linux closeout parent is unsafe")
    if root.exists():
        if _is_link_or_junction(root) or not root.is_dir():
            raise ValueError("same-Linux closeout root is unsafe")
    else:
        root.mkdir()
    unknown = {
        entry.name
        for entry in root.iterdir()
        if _is_link_or_junction(entry)
        or entry.name not in _ALLOWED_ROOT_ENTRIES
    }
    if unknown:
        raise ValueError(
            f"same-Linux closeout root has unknown entries: {sorted(unknown)}"
        )
    return root


def _read_json(path: str | Path, label: str) -> dict[str, Any]:
    source = _plain_file(path, label)
    try:
        value = json.loads(source.read_bytes().decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _write_or_replay_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    encoder: Callable[[Any], bytes] = canonical_bytes,
) -> Path:
    raw = encoder(dict(value))
    if path.exists() or _is_link_or_junction(path):
        if (
            _is_link_or_junction(path)
            or not path.is_file()
            or path.read_bytes() != raw
        ):
            raise FileExistsError(f"immutable JSON conflicts: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _copy_or_replay(source: Path, destination: Path) -> Path:
    source = _plain_file(source, "closeout source file")
    if destination.exists() or _is_link_or_junction(destination):
        if (
            _is_link_or_junction(destination)
            or not destination.is_file()
            or _file_sha256(destination) != _file_sha256(source)
            or destination.stat().st_size != source.stat().st_size
        ):
            raise FileExistsError(f"immutable copy conflicts: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    try:
        with source.open("rb") as incoming, temporary.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _copy_tree(source: Path, destination: Path) -> None:
    source = _plain_directory(source, "closeout source tree")
    expected: set[str] = set()
    for item in sorted(source.rglob("*")):
        if _is_link_or_junction(item):
            raise ValueError("closeout source tree contains a link")
        if item.is_dir():
            continue
        if not item.is_file():
            raise ValueError("closeout source tree contains a non-file")
        relative = item.relative_to(source).as_posix()
        expected.add(relative)
        _copy_or_replay(item, destination.joinpath(*PurePosixPath(relative).parts))
    actual = {
        item.relative_to(destination).as_posix()
        for item in destination.rglob("*")
        if item.is_file() and not _is_link_or_junction(item)
    }
    if not expected or actual != expected:
        raise ValueError("closeout copied tree inventory changed")


def _relative_record(root: Path, path: Path) -> dict[str, Any]:
    source = _plain_file(path, "bundle file")
    try:
        relative = source.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("bundle file escaped closeout root") from exc
    return {
        "path": relative,
        "sha256": _file_sha256(source),
        "bytes": source.stat().st_size,
    }


def _write_tar(
    path: Path,
    *,
    archive_root: str,
    files: Sequence[tuple[str, Path | bytes]],
) -> None:
    expected = [
        {
            "path": relative,
            "sha256": (
                hashlib.sha256(source).hexdigest()
                if isinstance(source, bytes)
                else _file_sha256(source)
            ),
            "bytes": (
                len(source)
                if isinstance(source, bytes)
                else Path(source).stat().st_size
            ),
        }
        for relative, source in files
    ]
    if path.exists() or _is_link_or_junction(path):
        observed = dataset_transport._safe_tar_inventory(  # type: ignore[attr-defined]
            _plain_file(path, "existing closeout tar"),
            expected_root=archive_root,
        )
        if observed != expected:
            raise FileExistsError(f"immutable tar conflicts: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as raw:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, mtime=0
            ) as compressed:
                with tarfile.open(
                    fileobj=compressed,
                    mode="w",
                    format=tarfile.PAX_FORMAT,
                ) as archive:
                    for relative, source in files:
                        pure = PurePosixPath(relative)
                        if (
                            pure.is_absolute()
                            or ".." in pure.parts
                            or "\\" in relative
                        ):
                            raise ValueError("unsafe closeout tar path")
                        info = tarfile.TarInfo(
                            f"{archive_root}/{pure.as_posix()}"
                        )
                        info.mode = 0o644
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        info.mtime = 0
                        if isinstance(source, bytes):
                            info.size = len(source)
                            archive.addfile(info, io.BytesIO(source))
                        else:
                            source_path = _plain_file(
                                source, "closeout tar source"
                            )
                            info.size = source_path.stat().st_size
                            with source_path.open("rb") as stream:
                                archive.addfile(info, stream)
            raw.flush()
            os.fsync(raw.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    observed = dataset_transport._safe_tar_inventory(  # type: ignore[attr-defined]
        path, expected_root=archive_root
    )
    if observed != expected:
        raise ValueError("stored closeout tar inventory changed")


def _write_prefixed_zip(
    path: Path,
    *,
    archive_root: str,
    payloads: Mapping[str, bytes],
) -> None:
    expected = [
        {
            "path": name,
            "sha256": hashlib.sha256(payloads[name]).hexdigest(),
            "bytes": len(payloads[name]),
        }
        for name in sorted(payloads)
    ]
    if path.exists() or _is_link_or_junction(path):
        observed = dataset_transport._safe_zip_inventory(  # type: ignore[attr-defined]
            _plain_file(path, "existing closeout zip"),
            expected_root=archive_root,
        )
        if observed != expected:
            raise FileExistsError(f"immutable zip conflicts: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as raw:
            with zipfile.ZipFile(
                raw,
                "w",
                compression=zipfile.ZIP_STORED,
                strict_timestamps=True,
            ) as archive:
                for name in sorted(payloads):
                    pure = PurePosixPath(name)
                    if (
                        pure.is_absolute()
                        or ".." in pure.parts
                        or "\\" in name
                    ):
                        raise ValueError("unsafe closeout zip path")
                    info = zipfile.ZipInfo(
                        f"{archive_root}/{pure.as_posix()}",
                        date_time=(1980, 1, 1, 0, 0, 0),
                    )
                    info.compress_type = zipfile.ZIP_STORED
                    info.create_system = 3
                    info.external_attr = (0o100644 & 0xFFFF) << 16
                    archive.writestr(info, payloads[name])
            raw.flush()
            os.fsync(raw.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    observed = dataset_transport._safe_zip_inventory(  # type: ignore[attr-defined]
        path, expected_root=archive_root
    )
    if observed != expected:
        raise ValueError("stored closeout zip inventory changed")


def _durable_final_receipt(
    *,
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    merge: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    accepted_ids = [row["job_id"] for row in ledger["accepted_jobs"]]
    expected_ids = [
        job_id for ids in quality_bridge.WAVE_JOB_IDS for job_id in ids
    ]
    if accepted_ids != expected_ids:
        raise ValueError("fresh-quality accepted job order changed")
    passed = gate["quality_pilot_passed"] is True
    core = {
        "schema": quality_bridge.FINAL_SCHEMA,
        "status": "qualified" if passed else "no_go",
        "decision": (
            "fresh_quality_passed_open_25_paired_data_pilot_only"
            if passed
            else "fresh_quality_failed_no_data_fanout"
        ),
        "plan_sha256": plan["plan_sha256"],
        "ledger_sha256": ledger["ledger_sha256"],
        "accepted_job_count": len(accepted_ids),
        "accepted_job_ids": accepted_ids,
        "lifecycle_transition_sha256s": [
            row["transition_sha256"] for row in ledger["transitions"]
        ],
        "quality_merge": deepcopy(dict(merge)),
        "quality_merge_sha256": quality_gate.canonical_sha256(merge),
        "quality_gate": deepcopy(dict(gate)),
        "quality_gate_sha256": quality_gate.canonical_sha256(gate),
        "quality_pilot_passed": passed,
        "data_pilot_25_paired_authorized": passed,
        "full_9000_paired_fanout_authorized": False,
        "training_eligible": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
    }
    receipt = {
        **core,
        "receipt_sha256": quality_bridge.canonical_sha256(core),
    }
    return quality_bridge.validate_final_receipt(receipt)


def _paths_within(root: Path, values: Sequence[str], label: str) -> None:
    for raw in values:
        path = Path(raw)
        if (
            not path.is_absolute()
            or _is_link_or_junction(path)
            or not path.resolve().is_relative_to(root)
        ):
            raise ValueError(f"{label} escaped the controller filesystem")


def run_same_linux_closeout(
    *,
    controller_filesystem_root: str | Path,
    output_root: str | Path,
    repository_root: str | Path,
    fresh_quality_gcp_plan_path: str | Path,
    fresh_quality_ledger_path: str | Path,
    accepted_results_directory: str | Path,
    dataset_run_name: str,
    dataset_bucket: str,
    provider_config_path: str | Path,
    raw_controller_token: str,
    platform_name: str | None = None,
    pilot_runner: PilotRunner = local_pilot.run_local_pilot,
    controller_prepare: ControllerPrepare = dataset_controller.prepare_controller,
) -> dict[str, Any]:
    """Run or replay the complete local closeout without a cloud mutation."""

    platform_value = sys.platform if platform_name is None else platform_name
    if not platform_value.startswith("linux"):
        raise PermissionError(
            "M3.1 closeout requires one native Linux controller context"
        )
    controller_filesystem = _plain_directory(
        controller_filesystem_root, "controller filesystem root"
    )
    repository = _plain_directory(repository_root, "repository root")
    if pilot_runner is local_pilot.run_local_pilot and Path.cwd().resolve() != repository:
        raise ValueError("production closeout must run from the repository root")

    plan_raw = _read_json(
        fresh_quality_gcp_plan_path, "fresh-quality GCP plan"
    )
    plan = quality_bridge.validate_gcp_plan(plan_raw, replay_sources=True)
    ledger = quality_bridge.validate_attempt_ledger(
        plan,
        _read_json(fresh_quality_ledger_path, "fresh-quality attempt ledger"),
    )
    resume = quality_bridge.build_resume_plan(plan, ledger)
    expected_ids = [
        job_id for ids in quality_bridge.WAVE_JOB_IDS for job_id in ids
    ]
    if (
        resume["all_jobs_accepted"] is not True
        or [row["job_id"] for row in ledger["accepted_jobs"]]
        != expected_ids
    ):
        raise PermissionError(
            "same-Linux closeout requires all 15 quality jobs received"
        )
    if (
        plan["machine_contract"]["provisioning_model"] != "SPOT"
        or any(
            wave["requires_owned_compute_absent"] is not True
            or wave["requires_worker_iam_removed_before_receive"] is not True
            for wave in plan["waves"]
        )
    ):
        raise PermissionError("quality worker cleanup contract changed")

    accepted_source = _plain_directory(
        accepted_results_directory, "accepted quality results"
    )
    proposed_root = Path(output_root)
    if not proposed_root.is_absolute():
        raise ValueError("same-Linux closeout root must be absolute")
    proposed_root = proposed_root.resolve()
    if not proposed_root.is_relative_to(controller_filesystem):
        raise ValueError("closeout root escaped the controller filesystem")
    root = _safe_root(proposed_root)
    _paths_within(
        controller_filesystem,
        [
            str(repository),
            str(Path(fresh_quality_gcp_plan_path).resolve()),
            str(Path(fresh_quality_ledger_path).resolve()),
            str(accepted_source),
            str(Path(provider_config_path).resolve()),
            *[str(value) for value in plan["source_paths"].values()],
        ],
        "closeout prerequisite",
    )
    quality_root = root / _QUALITY_DIRECTORY
    quality_root.mkdir(exist_ok=True)

    staging = _plain_directory(
        plan["source_paths"]["staging_directory"],
        "fresh-quality staging directory",
    )
    launch = plan["launch_manifest"]
    staging_receipt = quality_staging.validate_local_staging_receipt(
        staging / quality_staging.READY_NAME
    )
    staged_paths = {
        field: quality_staging._file_from_record(  # type: ignore[attr-defined]
            staging, staging_receipt["artifacts"][field], field
        )
        for field in quality_staging.ARTIFACT_FIELDS
    }
    if (
        staged_paths["launch_manifest"].resolve()
        != Path(plan["source_paths"]["launch_manifest"]).resolve()
        or staged_paths["performance_receipt"].resolve()
        != Path(plan["source_paths"]["performance_receipt"]).resolve()
        or staged_paths["profile_registry"].resolve()
        != Path(plan["source_paths"]["profile_registry"]).resolve()
        or launch["quality_package"]["sha256"]
        != _file_sha256(staged_paths["quality_package"])
    ):
        raise ValueError("fresh-quality GCP/staging source binding changed")
    plan_path = staged_paths["plan"]
    performance_path = staged_paths["performance_receipt"]
    materialization_path = staged_paths["materialization"]
    seal_path = staged_paths["root_seal"]
    _performance_receipt, performance_audit = (
        performance_portable.load_preferred_or_pinned_receipt(
            performance_path,
            expected_profile_sha256=quality.CURRENT_PROFILE_REGISTRY_SHA256,
        )
    )
    if (
        performance_audit.get("mode")
        != "exact_pinned_portable_no_source_path_dereference"
        or performance_audit.get("receipt_file_sha256")
        != performance_portable.PINNED_RECEIPT_FILE_SHA256
        or performance_audit.get("receipt_sha256")
        != performance_portable.PINNED_RECEIPT_SHA256
        or performance_audit.get("source_paths_dereferenced") is not False
        or performance_audit.get("new_seed_authorized") is not False
        or performance_audit.get("replacement_binary_authorized") is not False
        or performance_audit.get("current_profile_changed") is not False
    ):
        raise PermissionError(
            "same-Linux closeout did not use the exact path-free "
            "performance authorization"
        )
    quality_plan = quality.validate_plan_authorization(
        _read_json(plan_path, "durable quality plan"),
        performance_receipt_path=performance_path,
    )
    materialization = quality.validate_materialization_receipt(
        _read_json(materialization_path, "durable quality materialization"),
        plan=quality_plan,
        replay_roots=True,
    )
    seal = quality.validate_root_seal(
        _read_json(seal_path, "durable quality root seal"),
        plan=quality_plan,
        materialization=materialization,
    )
    _paths_within(
        controller_filesystem,
        [
            str(plan_path),
            str(performance_path),
            str(materialization_path),
            str(seal_path),
            str(materialization["root_directory"]),
        ],
        "fresh-quality durable source",
    )

    merge = quality_gate.build_fresh_quality_merge(
        plan_path=plan_path,
        materialization_path=materialization_path,
        root_seal_path=seal_path,
        results_directory=accepted_source,
        performance_receipt_path=performance_path,
    )
    merge_path = _write_or_replay_json(
        quality_root / "fresh_quality_merge.json",
        merge,
        encoder=quality_gate.canonical_bytes,
    )
    gate = quality_gate.build_fresh_quality_gate(
        merge=merge, replay_sources=True
    )
    if (
        gate["status"] != "pass"
        or gate["all_gates_passed"] is not True
        or gate["data_pilot_25_paired_authorized"] is not True
        or gate["full_9000_paired_fanout_authorized"] is not False
    ):
        raise PermissionError("fresh quality did not open the 25-pair pilot")
    gate_path = _write_or_replay_json(
        quality_root / "fresh_quality_gate.json",
        gate,
        encoder=quality_gate.canonical_bytes,
    )
    quality_gate.validate_fresh_quality_gate_value(
        executor_gate := json.loads(gate_path.read_text(encoding="ascii")),
        replay_sources=True,
    )
    if executor_gate != gate:
        raise ValueError("stored durable fresh-quality gate changed")
    final_receipt = _durable_final_receipt(
        plan=plan, ledger=ledger, merge=merge, gate=gate
    )
    final_path = _write_or_replay_json(
        quality_root / "fresh_quality_final_receipt.json",
        final_receipt,
        encoder=quality_bridge.canonical_bytes,
    )

    runtime_source = staged_paths["runtime_source"]
    runtime_payloads = quality_transport._archive_payloads(  # type: ignore[attr-defined]
        runtime_source,
        expected_sha256=launch["runtime_source"]["sha256"],
    )
    if not _REQUIRED_DATASET_RUNTIME_PATHS.issubset(runtime_payloads):
        raise ValueError("runtime source predates the dataset worker")
    candidate_record = launch["candidate_library"]
    candidate_relative = str(candidate_record["path"])
    candidate_raw = runtime_payloads.get(candidate_relative)
    if (
        candidate_raw is None
        or hashlib.sha256(candidate_raw).hexdigest()
        != dataset.ACCEPTED_CANDIDATE_LIBRARY_SHA256
        or len(candidate_raw) != candidate_record["bytes"]
    ):
        raise ValueError("accepted Candidate02 payload changed")

    fanout_root = root / _FANOUT_DIRECTORY
    candidate_path = fanout_root / "libofc_hu_m3_engine.so"
    if candidate_path.exists() or _is_link_or_junction(candidate_path):
        if (
            _is_link_or_junction(candidate_path)
            or not candidate_path.is_file()
            or candidate_path.read_bytes() != candidate_raw
        ):
            raise FileExistsError("immutable Candidate02 copy conflicts")
    else:
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        with candidate_path.open("xb") as stream:
            stream.write(candidate_raw)
            stream.flush()
            os.fsync(stream.fileno())
    runtime_tar = fanout_root / "runtime.tar.gz"
    _write_tar(
        runtime_tar,
        archive_root="runtime",
        files=[(name, raw) for name, raw in sorted(runtime_payloads.items())],
    )
    wheelhouse_source = staged_paths["wheelhouse"]
    wheelhouse_payloads = quality_transport._archive_payloads(  # type: ignore[attr-defined]
        wheelhouse_source,
        expected_sha256=launch["wheelhouse"]["sha256"],
        require_deterministic_timestamp=False,
    )
    wheelhouse_path = fanout_root / "wheelhouse.zip"
    _write_prefixed_zip(
        wheelhouse_path,
        archive_root="wheelhouse",
        payloads=wheelhouse_payloads,
    )
    provider_copy = _copy_or_replay(
        Path(provider_config_path),
        fanout_root / "provider_config.json",
    )

    pilot = dict(
        pilot_runner(
            output_root=root / _DATASET_DIRECTORY,
            fresh_quality_gate_path=gate_path,
            library_path=candidate_path,
        )
    )
    if (
        pilot.get("status")
        != "passed_source_replayed_25_paired_local_pilot"
        or pilot.get("paired_hand_count") != dataset.SHARD_PAIR_COUNT
        or pilot.get("root_count") != dataset.SHARD_PAIR_COUNT * 2
        or pilot.get("full_fanout_started") is not False
        or pilot.get("current_profile_changed") is not False
    ):
        raise PermissionError("25-pair dataset smoke did not close safely")
    dataset_root = root / _DATASET_DIRECTORY
    dataset_plan_path = _plain_file(
        dataset_root / local_pilot.PLAN_NAME, "dataset plan"
    )
    smoke_shard_directory = _plain_directory(
        dataset_root
        / local_pilot.SHARDS_DIRECTORY_NAME
        / dataset.SMOKE_SHARD_ID,
        "completed smoke shard",
    )
    dataset.validate_completed_shard(
        plan=dataset.validate_dataset_plan(
            _read_json(dataset_plan_path, "dataset plan")
        ),
        shard_id=dataset.SMOKE_SHARD_ID,
        shard_directory=smoke_shard_directory,
    )
    smoke_gate_path = _plain_file(
        dataset_root / local_pilot.SMOKE_GATE_NAME,
        "dataset smoke gate",
    )
    smoke_files = [
        (path.relative_to(smoke_shard_directory).as_posix(), path)
        for path in sorted(smoke_shard_directory.rglob("*"))
        if path.is_file() and not _is_link_or_junction(path)
    ]
    smoke_archive = fanout_root / "smoke_shard.tar.gz"
    _write_tar(
        smoke_archive,
        archive_root="smoke_shard",
        files=smoke_files,
    )

    controller_root = root / _CONTROLLER_DIRECTORY
    local_shard_root = root / _LOCAL_SHARDS_DIRECTORY
    local_shard_root.mkdir(exist_ok=True)
    controller_contract = dict(
        controller_prepare(
            run_name=dataset_run_name,
            bucket=dataset_bucket,
            dataset_plan_path=dataset_plan_path,
            fresh_quality_gate_path=gate_path,
            smoke_gate_path=smoke_gate_path,
            smoke_shard_directory=smoke_shard_directory,
            smoke_shard_archive_path=smoke_archive,
            runtime_archive_path=runtime_tar,
            wheelhouse_archive_path=wheelhouse_path,
            candidate_library_path=candidate_path,
            provider_config_path=provider_copy,
            local_shard_root=local_shard_root,
            output_root=controller_root,
            raw_controller_token=raw_controller_token,
        )
    )
    if (
        controller_contract.get("status") != "prepared_cloud_not_started"
        or controller_contract.get("quality_and_smoke_source_replayed")
        is not True
        or controller_contract.get("current_profile_changed") is not False
    ):
        raise PermissionError("dataset controller did not prepare safely")
    portable_path = _plain_file(
        controller_root / "portable_authorization.json",
        "portable fanout authorization",
    )
    transport_path = _plain_file(
        controller_root / "transport_plan.json", "dataset transport plan"
    )
    portable_value = _read_json(
        portable_path, "portable fanout authorization"
    )
    transport_value = _read_json(transport_path, "dataset transport plan")
    if (
        portable_value.get("full_9000_paired_fanout_authorized") is not True
        or transport_value.get("cloud_shard_count") != 359
        or transport_value.get("precompleted_smoke_shard_id")
        != dataset.SMOKE_SHARD_ID
        or transport_value.get("cloud_execution_started") is not False
    ):
        raise PermissionError("portable 359-shard fanout boundary changed")

    quality_source_paths = list(merge["source_paths"].values())
    transport_source_paths = [
        str(record["source_path"])
        for record in transport_value["content_sources"]
    ] + [transport_value["source_paths"]["smoke_shard_directory"]]
    _paths_within(
        controller_filesystem,
        quality_source_paths,
        "fresh-quality source",
    )
    _paths_within(root, transport_source_paths, "dataset transport source")

    bundle_files = [
        _relative_record(root, path)
        for path in (
            merge_path,
            gate_path,
            final_path,
            dataset_plan_path,
            smoke_gate_path,
            smoke_archive,
            runtime_tar,
            wheelhouse_path,
            candidate_path,
            portable_path,
            transport_path,
            controller_root / "controller_contract.json",
        )
    ]
    accepted_source_files = [
        path
        for path in sorted(accepted_source.rglob("*"))
        if path.is_file() and not _is_link_or_junction(path)
    ]
    materialized_source_root = Path(materialization["root_directory"])
    materialized_source_files = [
        path
        for path in sorted(materialized_source_root.rglob("*"))
        if path.is_file() and not _is_link_or_junction(path)
    ]
    if (
        len(accepted_source_files) != 15
        or len(materialized_source_files) != 55
    ):
        raise ValueError("fresh-quality persistent source closure changed")
    source_closure_files = [
        _relative_record(controller_filesystem, path)
        for path in (
            Path(fresh_quality_gcp_plan_path),
            Path(fresh_quality_ledger_path),
            Path(provider_config_path),
            plan_path,
            performance_path,
            materialization_path,
            seal_path,
            *accepted_source_files,
            *materialized_source_files,
        )
    ]
    bundle = {
        "schema": FANOUT_BUNDLE_SCHEMA,
        "status": "portable_359_shard_authorization_ready_cloud_not_started",
        "required_restore_root": str(controller_filesystem),
        "restore_contract": (
            "same_persistent_disk_exact_absolute_mount_only"
        ),
        "source_paths_all_within_restore_root": True,
        "portable_authorization_sha256": portable_value[
            "authorization_sha256"
        ],
        "transport_plan_sha256": transport_value["plan_sha256"],
        "dataset_plan_sha256": dataset.canonical_sha256(
            dataset.validate_dataset_plan(
                _read_json(dataset_plan_path, "dataset plan")
            )
        ),
        "smoke_shard_id": dataset.SMOKE_SHARD_ID,
        "completed_smoke_pair_count": dataset.SHARD_PAIR_COUNT,
        "fanout_shard_count": 359,
        "bundle_files": bundle_files,
        "bundle_file_aggregate_sha256": canonical_sha256(bundle_files),
        "persistent_source_closure_files": source_closure_files,
        "persistent_source_closure_sha256": canonical_sha256(
            source_closure_files
        ),
        "performance_receipt_validation": performance_audit,
        "performance_source_paths_dereferenced": False,
        "fresh_quality_worker_vm_reuse_allowed": False,
        "dedicated_controller_context_required": True,
        "controller_cloud_mutation_authorized": False,
        "controller_cleanup_condition": (
            "dataset_final_receipt_complete_and_bundle_export_readback"
        ),
        "hidden_information_field_count": 0,
        "opponent_private_discards_used": False,
        "teacher_values_are_realized_match_ev": False,
        "training_eligible": False,
        "current_profile_changed": False,
    }
    manifest_path = _write_or_replay_json(
        root / FANOUT_MANIFEST_NAME, bundle
    )
    ready = {
        "schema": CLOSEOUT_SCHEMA,
        "status": "complete_same_linux_ready_359_shard_fanout_not_started",
        "output_root": str(root),
        "fresh_quality_final_receipt": _relative_record(root, final_path),
        "fresh_quality_gate": _relative_record(root, gate_path),
        "dataset_smoke_gate": _relative_record(root, smoke_gate_path),
        "portable_authorization": _relative_record(root, portable_path),
        "fanout_bundle_manifest": _relative_record(root, manifest_path),
        "quality_job_count": 15,
        "dataset_smoke_pair_count": 25,
        "fanout_shard_count": 359,
        "performance_receipt_validation_mode": performance_audit["mode"],
        "performance_source_paths_dereferenced": False,
        "fresh_quality_worker_vm_reuse_allowed": False,
        "dedicated_controller_context": True,
        "cloud_called": False,
        "cloud_execution_started": False,
        "full_fanout_started": False,
        "training_eligible": False,
        "current_profile_changed": False,
    }
    ready_path = _write_or_replay_json(root / READY_NAME, ready)
    return {
        **ready,
        "ready_path": str(ready_path),
        "ready_file_sha256": _file_sha256(ready_path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Close fresh quality and prepare the M3.1 359-shard fanout "
            "inside one persistent Linux controller filesystem."
        )
    )
    parser.add_argument("--controller-filesystem-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--fresh-quality-gcp-plan", required=True)
    parser.add_argument("--fresh-quality-ledger", required=True)
    parser.add_argument("--accepted-results-directory", required=True)
    parser.add_argument("--dataset-run-name", required=True)
    parser.add_argument("--dataset-bucket", required=True)
    parser.add_argument("--provider-config", required=True)
    parser.add_argument(
        "--controller-token-env",
        default=dataset_controller.CONTROLLER_TOKEN_ENV,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    token = os.environ.get(args.controller_token_env)
    if token is None:
        raise PermissionError(
            f"{args.controller_token_env} is required in memory"
        )
    result = run_same_linux_closeout(
        controller_filesystem_root=args.controller_filesystem_root,
        output_root=args.output_root,
        repository_root=args.repository_root,
        fresh_quality_gcp_plan_path=args.fresh_quality_gcp_plan,
        fresh_quality_ledger_path=args.fresh_quality_ledger,
        accepted_results_directory=args.accepted_results_directory,
        dataset_run_name=args.dataset_run_name,
        dataset_bucket=args.dataset_bucket,
        provider_config_path=args.provider_config,
        raw_controller_token=token,
    )
    print(canonical_bytes(result).decode("ascii"), end="")
    return 0


__all__ = [
    "CLOSEOUT_SCHEMA",
    "FANOUT_BUNDLE_SCHEMA",
    "FANOUT_MANIFEST_NAME",
    "READY_NAME",
    "canonical_bytes",
    "canonical_sha256",
    "main",
    "run_same_linux_closeout",
]


if __name__ == "__main__":
    raise SystemExit(main())
