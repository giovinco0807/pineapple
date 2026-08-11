"""Create-only local staging for the M3.1 T3 fresh-quality cloud run.

This module joins the already-reviewed scientific and transport APIs.  It has
no cloud client and performs no profile selection or activation.  A successful
run contains:

* the qualified performance-lock-v4 receipt and pinned profile registry;
* 55 paired fresh-quality roots (110 actor observations), seal, and 15 jobs;
* a deterministic quality package and hash-pinned runtime source archive;
* a network-free deterministic repackage of an accepted offline wheelhouse;
* the local 8+7 launch manifest consumed by the separate GCP bridge.

The output directory is create-only.  ``LOCAL_STAGING_READY.json`` is written
only after a full source replay; failures retain partial artifacts plus a
best-effort forensic marker and must be retried in a new directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import zipfile
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from . import hu_m31_t3_step6d_fresh_quality_transport_v1 as transport
from . import hu_m31_t3_step6d_fresh_quality_v1 as quality


SCHEMA = "hu_m31_t3_step6d_fresh_quality_local_staging_v1"
FAILURE_SCHEMA = "hu_m31_t3_step6d_fresh_quality_local_staging_failure_v1"
STATUS = "complete_local_staging_ready_cloud_not_authorized"
RUN_NAME_PREFIX = "regular-hu-m31-t3-fqv1-"
READY_NAME = "LOCAL_STAGING_READY.json"
FAILURE_NAME = "LOCAL_STAGING_FAILED.json"

PERFORMANCE_RELATIVE = "control/performance_lock_v4_production_receipt.json"
PROFILE_RELATIVE = "control/ai_profiles.py"
PLAN_RELATIVE = "control/fresh_quality_plan.json"
MATERIALIZATION_RELATIVE = "control/materialization.json"
SEAL_RELATIVE = "control/root_seal.json"
ROOTS_RELATIVE = "scientific/materialized_roots"
PACKAGE_DIRECTORY_RELATIVE = "scientific/package"
PACKAGE_ARCHIVE_RELATIVE = "artifacts/fresh_quality_package_v1.zip"
SOURCE_ARCHIVE_RELATIVE = f"runtime/{transport.SOURCE_ARCHIVE_NAME}"
SOURCE_MANIFEST_RELATIVE = f"runtime/{transport.SOURCE_MANIFEST_NAME}"
WHEELHOUSE_ARCHIVE_RELATIVE = "wheelhouse/wheelhouse.zip"
WHEELHOUSE_MANIFEST_RELATIVE = "wheelhouse/wheelhouse_manifest.json"
STARTUP_RELATIVE = "startup/startup_hu_m31_t3_step6d_fresh_quality_v1.sh"
LAUNCH_RELATIVE = "launch.json"

ARTIFACT_FIELDS = (
    "performance_receipt",
    "profile_registry",
    "plan",
    "materialization",
    "root_seal",
    "quality_package",
    "runtime_source",
    "runtime_source_manifest",
    "wheelhouse",
    "wheelhouse_manifest",
    "startup",
    "launch_manifest",
)
_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "output_directory",
        "artifacts",
        "artifact_aggregate_sha256",
        "paired_hand_count",
        "root_count",
        "job_count",
        "wave_job_counts",
        "profile_registry_sha256",
        "opponent_private_discards_used",
        "hidden_information_field_count",
        "cloud_called",
        "cloud_execution_started",
        "training_eligible",
        "promotion_evidence",
        "current_profile_changed",
        "receipt_sha256",
    }
)
_FILE_KEYS = frozenset({"path", "sha256", "bytes"})
_SHA = re.compile(r"^[0-9a-f]{64}$")
_RUN = re.compile(r"^[a-z][a-z0-9-]{2,62}$")
_WHEEL = re.compile(r"^[A-Za-z0-9_.+!-]+\.whl$")


def canonical_bytes(value: Any) -> bytes:
    return quality.canonical_bytes(value)


def canonical_sha256(value: Any) -> str:
    return quality.canonical_sha256(value)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} fields changed: "
            f"missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _plain_file(path: str | Path, label: str) -> Path:
    source = Path(path).resolve()
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    return source


def _safe_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} is unsafe")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != value
    ):
        raise ValueError(f"{label} is unsafe")
    return value


def _write_once(path: str | Path, payload: bytes) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return target


def _write_json_once(path: str | Path, value: Mapping[str, Any]) -> Path:
    return _write_once(path, canonical_bytes(dict(value)))


def _copy_once(source: str | Path, destination: str | Path) -> Path:
    source_path = _plain_file(source, "local staging source")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("rb") as reader, target.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    if (
        target.stat().st_size != source_path.stat().st_size
        or sha256_file(target) != sha256_file(source_path)
    ):
        raise ValueError("local staging copy readback changed")
    return target


def _read_canonical(path: str | Path, label: str) -> dict[str, Any]:
    raw = _plain_file(path, label).read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if (
        not isinstance(value, dict)
        or raw not in (canonical_bytes(value), canonical_bytes(value) + b"\n")
    ):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _file_record(root: Path, path: str | Path) -> dict[str, Any]:
    source = _plain_file(path, "staged artifact")
    try:
        relative = source.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("staged artifact escaped output root") from exc
    _safe_relative(relative, "staged artifact path")
    return {
        "path": relative,
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
    }


def _file_from_record(
    root: Path, record: Mapping[str, Any], label: str
) -> Path:
    _exact_keys(record, _FILE_KEYS, label)
    relative = _safe_relative(record.get("path"), f"{label} path")
    target = root.joinpath(*PurePosixPath(relative).parts).resolve()
    if (
        root not in target.parents
        or target.is_symlink()
        or not target.is_file()
        or not isinstance(record.get("sha256"), str)
        or _SHA.fullmatch(str(record["sha256"])) is None
        or sha256_file(target) != record["sha256"]
        or target.stat().st_size != record.get("bytes")
    ):
        raise ValueError(f"{label} hash/size/path changed")
    return target


def _package_offline_wheelhouse(
    *,
    input_archive: str | Path,
    input_manifest: str | Path,
    output_archive: str | Path,
    output_manifest: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_archive = _plain_file(input_archive, "input offline wheelhouse")
    source_manifest = _plain_file(
        input_manifest, "input offline wheelhouse manifest"
    )
    manifest, summary = transport._validate_wheelhouse(
        archive_path=source_archive,
        manifest_path=source_manifest,
    )
    entries = manifest["entries"]
    expected = {str(row["filename"]): row for row in entries}
    if len(expected) != len(entries):
        raise ValueError("offline wheelhouse entries are duplicated")
    destination = Path(output_archive)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_archive, "r") as source:
        if set(source.namelist()) != set(expected):
            raise ValueError("offline wheelhouse input inventory changed")
        with destination.open("xb") as raw:
            with zipfile.ZipFile(
                raw, "w", compression=zipfile.ZIP_STORED, strict_timestamps=True
            ) as archive:
                for name in sorted(expected):
                    if _WHEEL.fullmatch(name) is None:
                        raise ValueError("offline wheelhouse member name is unsafe")
                    payload = source.read(name)
                    record = expected[name]
                    if (
                        hashlib.sha256(payload).hexdigest() != record["sha256"]
                        or len(payload) != record["bytes"]
                    ):
                        raise ValueError("offline wheelhouse member changed")
                    info = zipfile.ZipInfo(
                        name, date_time=(1980, 1, 1, 0, 0, 0)
                    )
                    info.compress_type = zipfile.ZIP_STORED
                    info.create_system = 3
                    info.external_attr = (0o100600 & 0xFFFF) << 16
                    archive.writestr(info, payload)
            raw.flush()
            os.fsync(raw.fileno())
    # Retain the accepted manifest bytes; only the container is deterministically
    # rebuilt.  The transport validator replays every member again.
    _copy_once(source_manifest, output_manifest)
    observed, observed_summary = transport._validate_wheelhouse(
        archive_path=Path(output_archive),
        manifest_path=Path(output_manifest),
    )
    if observed != manifest or observed_summary != summary:
        raise ValueError("repackaged offline wheelhouse changed")
    return observed, observed_summary


def _failure_marker(
    *,
    output: Path,
    run_name: str,
    failed_stage: str,
    error: BaseException,
) -> None:
    if not output.is_dir() or (output / FAILURE_NAME).exists():
        return
    failure = {
        "schema": FAILURE_SCHEMA,
        "status": "partial_local_staging_preserved_not_authorized",
        "run_name": run_name,
        "output_directory": str(output),
        "failed_stage": failed_stage,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "cloud_called": False,
        "cloud_execution_started": False,
        "training_eligible": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
    }
    try:
        _write_json_once(output / FAILURE_NAME, failure)
    except BaseException:
        pass


def build_local_staging(
    *,
    repository_root: str | Path,
    performance_receipt_path: str | Path,
    run_name: str,
    candidate_library_path: str | Path,
    feature_encoder_path: str | Path,
    wheelhouse_archive_path: str | Path,
    wheelhouse_manifest_path: str | Path,
    startup_script_path: str | Path,
    profile_registry_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Build and fully replay one immutable, cloud-neutral staging directory."""

    repository = Path(repository_root).resolve()
    if repository.is_symlink() or not repository.is_dir():
        raise ValueError("repository root is missing or unsafe")
    if (
        not isinstance(run_name, str)
        or _RUN.fullmatch(run_name) is None
        or not run_name.startswith(RUN_NAME_PREFIX)
    ):
        raise ValueError("fresh-quality local run name is invalid")
    performance_source = _plain_file(
        performance_receipt_path, "qualified performance receipt"
    )
    candidate = _plain_file(candidate_library_path, "accepted candidate library")
    feature = _plain_file(feature_encoder_path, "accepted feature encoder")
    startup_source = _plain_file(startup_script_path, "fresh-quality startup")
    profile_source = _plain_file(profile_registry_path, "profile registry")
    if (
        sha256_file(candidate) != quality.ACCEPTED_CANDIDATE_LIBRARY_SHA256
        or sha256_file(feature) != quality.ACCEPTED_FEATURE_ENCODER_SHA256
        or sha256_file(profile_source) != quality.CURRENT_PROFILE_REGISTRY_SHA256
    ):
        raise PermissionError("fresh-quality pinned native/profile input changed")
    if startup_source.read_bytes().startswith(b"#!/usr/bin/env bash\n") is not True:
        raise ValueError("fresh-quality startup header changed")
    # Validate authorization and all large immutable wheel inputs before
    # creating a forensic output directory.
    expected_plan = quality.build_fresh_quality_plan(
        performance_receipt_path=performance_source
    )
    transport._validate_wheelhouse(
        archive_path=_plain_file(
            wheelhouse_archive_path, "input offline wheelhouse"
        ),
        manifest_path=_plain_file(
            wheelhouse_manifest_path, "input offline wheelhouse manifest"
        ),
    )

    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError("fresh-quality local staging is create-only")
    output.mkdir(parents=True, exist_ok=False)
    stage = "copy_immutable_inputs"
    try:
        staged_performance = _copy_once(
            performance_source, output / PERFORMANCE_RELATIVE
        )
        staged_profile = _copy_once(profile_source, output / PROFILE_RELATIVE)
        staged_startup = _copy_once(startup_source, output / STARTUP_RELATIVE)
        _package_offline_wheelhouse(
            input_archive=wheelhouse_archive_path,
            input_manifest=wheelhouse_manifest_path,
            output_archive=output / WHEELHOUSE_ARCHIVE_RELATIVE,
            output_manifest=output / WHEELHOUSE_MANIFEST_RELATIVE,
        )

        stage = "write_authorized_plan"
        plan = quality.write_fresh_quality_plan(
            performance_receipt_path=staged_performance,
            output_path=output / PLAN_RELATIVE,
        )
        if plan != expected_plan:
            raise ValueError("fresh-quality plan changed after staging")

        stage = "materialize_55_paired_roots"
        materialization, seal = quality.materialize_fresh_quality_roots(
            repository_root=repository,
            plan_path=output / PLAN_RELATIVE,
            performance_receipt_path=staged_performance,
            output_directory=output / ROOTS_RELATIVE,
            materialization_output=output / MATERIALIZATION_RELATIVE,
            seal_output=output / SEAL_RELATIVE,
            feature_encoder_path=feature,
        )

        stage = "create_15_job_quality_package"
        (output / PACKAGE_ARCHIVE_RELATIVE).parent.mkdir(
            parents=True, exist_ok=True
        )
        package = quality.create_fresh_quality_package(
            plan_path=output / PLAN_RELATIVE,
            materialization_path=output / MATERIALIZATION_RELATIVE,
            seal_path=output / SEAL_RELATIVE,
            performance_receipt_path=staged_performance,
            output_directory=output / PACKAGE_DIRECTORY_RELATIVE,
            archive_path=output / PACKAGE_ARCHIVE_RELATIVE,
        )

        stage = "create_runtime_source_archive"
        source_manifest = transport.create_runtime_source_archive(
            repository_root=repository,
            candidate_library_path=candidate,
            feature_encoder_path=feature,
            archive_path=output / SOURCE_ARCHIVE_RELATIVE,
            manifest_path=output / SOURCE_MANIFEST_RELATIVE,
        )

        stage = "build_local_launch_manifest"
        launch = transport.build_local_launch_manifest(
            run_name=run_name,
            staging_directory=output,
            quality_package_archive=output / PACKAGE_ARCHIVE_RELATIVE,
            runtime_source_archive=output / SOURCE_ARCHIVE_RELATIVE,
            runtime_source_manifest_path=output / SOURCE_MANIFEST_RELATIVE,
            wheelhouse_archive=output / WHEELHOUSE_ARCHIVE_RELATIVE,
            wheelhouse_manifest_path=output / WHEELHOUSE_MANIFEST_RELATIVE,
            startup_script=staged_startup,
            output_path=output / LAUNCH_RELATIVE,
        )

        artifacts = {
            "performance_receipt": _file_record(output, staged_performance),
            "profile_registry": _file_record(output, staged_profile),
            "plan": _file_record(output, output / PLAN_RELATIVE),
            "materialization": _file_record(
                output, output / MATERIALIZATION_RELATIVE
            ),
            "root_seal": _file_record(output, output / SEAL_RELATIVE),
            "quality_package": _file_record(
                output, output / PACKAGE_ARCHIVE_RELATIVE
            ),
            "runtime_source": _file_record(
                output, output / SOURCE_ARCHIVE_RELATIVE
            ),
            "runtime_source_manifest": _file_record(
                output, output / SOURCE_MANIFEST_RELATIVE
            ),
            "wheelhouse": _file_record(
                output, output / WHEELHOUSE_ARCHIVE_RELATIVE
            ),
            "wheelhouse_manifest": _file_record(
                output, output / WHEELHOUSE_MANIFEST_RELATIVE
            ),
            "startup": _file_record(output, staged_startup),
            "launch_manifest": _file_record(output, output / LAUNCH_RELATIVE),
        }
        if list(artifacts) != list(ARTIFACT_FIELDS):
            raise ValueError("local staging artifact ordering changed")
        core = {
            "schema": SCHEMA,
            "status": STATUS,
            "run_name": run_name,
            "output_directory": str(output),
            "artifacts": artifacts,
            "artifact_aggregate_sha256": canonical_sha256(artifacts),
            "paired_hand_count": materialization["paired_hand_count"],
            "root_count": materialization["root_count"],
            "job_count": package["manifest"]["job_count"],
            "wave_job_counts": launch["wave_job_counts"],
            "profile_registry_sha256": sha256_file(staged_profile),
            "opponent_private_discards_used": False,
            "hidden_information_field_count": materialization[
                "hidden_information_field_count"
            ],
            "cloud_called": False,
            "cloud_execution_started": False,
            "training_eligible": False,
            "promotion_evidence": False,
            "current_profile_changed": False,
        }
        receipt = {**core, "receipt_sha256": canonical_sha256(core)}
        stage = "full_local_source_replay"
        validate_local_staging_value(receipt, ready_must_exist=False)
        _write_json_once(output / READY_NAME, receipt)
        stored = validate_local_staging_receipt(output / READY_NAME)
        if stored != receipt or source_manifest != _read_canonical(
            output / SOURCE_MANIFEST_RELATIVE, "stored runtime source manifest"
        ):
            raise ValueError("stored local staging receipt changed")
        return receipt
    except BaseException as error:
        _failure_marker(
            output=output,
            run_name=run_name,
            failed_stage=stage,
            error=error,
        )
        raise


def validate_local_staging_value(
    value: Mapping[str, Any], *, ready_must_exist: bool
) -> dict[str, Any]:
    receipt = deepcopy(dict(value))
    _exact_keys(receipt, _RECEIPT_KEYS, "fresh-quality local staging receipt")
    payload = deepcopy(receipt)
    digest = payload.pop("receipt_sha256", None)
    if digest != canonical_sha256(payload):
        raise ValueError("fresh-quality local staging receipt digest changed")
    output = Path(str(receipt.get("output_directory", "")))
    if (
        not output.is_absolute()
        or output.is_symlink()
        or not output.is_dir()
        or str(output.resolve()) != str(output)
    ):
        raise ValueError("fresh-quality local staging directory changed")
    artifacts = receipt.get("artifacts")
    if (
        not isinstance(artifacts, Mapping)
        or set(artifacts) != set(ARTIFACT_FIELDS)
    ):
        raise ValueError("fresh-quality local staging artifacts changed")
    paths = {
        field: _file_from_record(output, artifacts[field], field)
        for field in ARTIFACT_FIELDS
    }
    if receipt.get("artifact_aggregate_sha256") != canonical_sha256(artifacts):
        raise ValueError("fresh-quality local staging aggregate changed")

    plan = quality.validate_plan_authorization(
        _read_canonical(paths["plan"], "staged fresh-quality plan"),
        performance_receipt_path=paths["performance_receipt"],
    )
    materialization = quality.validate_materialization_receipt(
        _read_canonical(
            paths["materialization"], "staged fresh-quality materialization"
        ),
        plan=plan,
        replay_roots=True,
    )
    seal = quality.validate_root_seal(
        _read_canonical(paths["root_seal"], "staged fresh-quality root seal"),
        plan=plan,
        materialization=materialization,
    )
    package = quality.validate_fresh_quality_package(
        package_directory=output / PACKAGE_DIRECTORY_RELATIVE,
        performance_receipt_path=paths["performance_receipt"],
        archive_path=paths["quality_package"],
    )
    source_manifest = transport.validate_runtime_source_manifest(
        _read_canonical(
            paths["runtime_source_manifest"], "staged runtime source manifest"
        ),
        archive_path=paths["runtime_source"],
        require_all_runtime_sources=True,
    )
    wheel_manifest, _wheel_summary = transport._validate_wheelhouse(
        archive_path=paths["wheelhouse"],
        manifest_path=paths["wheelhouse_manifest"],
    )
    launch = transport.validate_local_launch_manifest(
        _read_canonical(paths["launch_manifest"], "staged local launch manifest"),
        staging_directory=output,
    )
    expected_top = {
        "artifacts",
        "control",
        "runtime",
        "scientific",
        "startup",
        "wheelhouse",
        LAUNCH_RELATIVE,
    }
    if ready_must_exist:
        expected_top.add(READY_NAME)
    actual_top = {path.name for path in output.iterdir()}
    if actual_top != expected_top:
        raise ValueError("fresh-quality local staging top-level inventory changed")
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("status") != STATUS
        or not str(receipt.get("run_name", "")).startswith(RUN_NAME_PREFIX)
        or launch["run_name"] != receipt["run_name"]
        or receipt.get("paired_hand_count") != 55
        or receipt.get("root_count") != 110
        or receipt.get("job_count") != 15
        or receipt.get("wave_job_counts") != [8, 7]
        or receipt.get("profile_registry_sha256")
        != quality.CURRENT_PROFILE_REGISTRY_SHA256
        or sha256_file(paths["profile_registry"])
        != quality.CURRENT_PROFILE_REGISTRY_SHA256
        or package["manifest"]["job_count"] != 15
        or package["manifest"]["root_file_count"] != 55
        or source_manifest["profile_registry_sha256"]
        != quality.CURRENT_PROFILE_REGISTRY_SHA256
        or not wheel_manifest["entries"]
        or receipt.get("opponent_private_discards_used") is not False
        or receipt.get("hidden_information_field_count") != 0
        or any(
            receipt.get(field) is not False
            for field in (
                "cloud_called",
                "cloud_execution_started",
                "training_eligible",
                "promotion_evidence",
                "current_profile_changed",
            )
        )
        or plan["scientific_boundaries"]["opponent_private_discards_used"]
        is not False
        or seal["current_profile_changed"] is not False
        or launch["current_profile_changed"] is not False
    ):
        raise ValueError("fresh-quality local staging safety boundary changed")
    return receipt


def validate_local_staging_receipt(path: str | Path) -> dict[str, Any]:
    ready = _plain_file(path, "fresh-quality local staging receipt")
    if ready.name != READY_NAME:
        raise ValueError("fresh-quality local staging receipt filename changed")
    return validate_local_staging_value(
        _read_canonical(ready, "fresh-quality local staging receipt"),
        ready_must_exist=True,
    )


def _parser() -> argparse.ArgumentParser:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--performance-receipt", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--candidate-library", type=Path, required=True)
    parser.add_argument("--feature-encoder", type=Path, required=True)
    parser.add_argument("--wheelhouse-archive", type=Path, required=True)
    parser.add_argument("--wheelhouse-manifest", type=Path, required=True)
    parser.add_argument(
        "--startup",
        type=Path,
        default=repository
        / "scripts/startup_hu_m31_t3_step6d_fresh_quality_v1.sh",
    )
    parser.add_argument(
        "--profile-registry",
        type=Path,
        default=repository / "src/ofc_regular/ai_profiles.py",
    )
    parser.add_argument("--repository-root", type=Path, default=repository)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = build_local_staging(
        repository_root=args.repository_root,
        performance_receipt_path=args.performance_receipt,
        run_name=args.run_name,
        candidate_library_path=args.candidate_library,
        feature_encoder_path=args.feature_encoder,
        wheelhouse_archive_path=args.wheelhouse_archive,
        wheelhouse_manifest_path=args.wheelhouse_manifest,
        startup_script_path=args.startup,
        profile_registry_path=args.profile_registry,
        output_directory=args.output_directory,
    )
    print(
        canonical_bytes(
            {
                "status": receipt["status"],
                "run_name": receipt["run_name"],
                "receipt_sha256": receipt["receipt_sha256"],
                "paired_hand_count": receipt["paired_hand_count"],
                "root_count": receipt["root_count"],
                "job_count": receipt["job_count"],
                "wave_job_counts": receipt["wave_job_counts"],
                "cloud_called": False,
                "current_profile_changed": False,
            }
        ).decode("ascii")
    )
    return 0


__all__ = [
    "FAILURE_NAME",
    "READY_NAME",
    "RUN_NAME_PREFIX",
    "SCHEMA",
    "STATUS",
    "build_local_staging",
    "canonical_bytes",
    "canonical_sha256",
    "main",
    "sha256_file",
    "validate_local_staging_receipt",
    "validate_local_staging_value",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
