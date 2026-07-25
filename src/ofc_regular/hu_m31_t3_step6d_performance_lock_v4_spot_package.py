"""Content-addressed scientific source package for performance-lock v4.

The package is a transport facade only.  It replays the frozen v4 plan,
materialization receipt, root seal, all 100 canonical root files, accepted
native binaries, and feature encoder before writing a create-only directory.
It never invokes a cloud API or changes an AI profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from . import hu_m31_t3_step6d_candidate02_performance_lock_v4_plan as v4
from . import run_hu_m31_t3_step6d_performance as performance_v1
from . import run_hu_m31_t3_step6d_performance_v2 as runner


PACKAGE_SCHEMA = "hu_m31_t3_step6d_performance_lock_v4_spot_package_v1"
READY_SCHEMA = "hu_m31_t3_step6d_performance_lock_v4_spot_package_ready_v1"
PACKAGE_STATUS = "immutable_performance_lock_v4_source_ready_not_authorized"
READY_STATUS = "performance_lock_v4_package_validated_not_authorized"
RUN_NAME = "regular-hu-m31-c02-performance-lock-v4-20260723-001"

SOURCE_NAME = "ofc_regular_hu_m31_t3_step6d_performance_lock_v4_source.zip"
MANIFEST_NAME = "manifest.json"
READY_NAME = "PACKAGE_READY.json"
JOB_DIRECTORY = "jobs"

PLAN_ARCHIVE_PATH = "frozen/performance_lock_v4_plan.json"
CLAIM_ARCHIVE_PATH = "frozen/MATERIALIZATION_CLAIM.json"
MATERIALIZATION_ARCHIVE_PATH = "frozen/MATERIALIZATION_RECEIPT.json"
SEAL_ARCHIVE_PATH = "frozen/ROOT_SEAL.json"
ROOT_ARCHIVE_TEMPLATE = "frozen/roots/hand_{index:03d}.json"
CANDIDATE_ARCHIVE_PATH = "native/candidate/release/libofc_hu_m3_engine.so"
REFERENCE_ARCHIVE_PATH = "native/reference/release/libofc_hu_m3_engine.so"
FEATURE_ARCHIVE_PATH = "target/release/libofc_stage3_feature_encoder.so"

FEATURE_ENCODER_SHA256 = (
    "82510e563ee290cd536e7aebe6bcf0efefac061af5488851f0a582bb4ef83411"
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FROZEN_SOURCE_ROOT = (
    _REPO_ROOT
    / "outputs/gcp_runs/regular-hu-m31-c02-full100-dev-20260717-001/package_src"
)
DEFAULT_CANDIDATE_LIBRARY_PATH = (
    _FROZEN_SOURCE_ROOT / CANDIDATE_ARCHIVE_PATH
)
DEFAULT_REFERENCE_LIBRARY_PATH = (
    _FROZEN_SOURCE_ROOT / REFERENCE_ARCHIVE_PATH
)
DEFAULT_FEATURE_ENCODER_PATH = _FROZEN_SOURCE_ROOT / FEATURE_ARCHIVE_PATH
DEFAULT_PACKAGE_DIR = (
    _REPO_ROOT
    / "outputs/hu_joint_policy/m31_t3_step6d/performance_lock_v4/"
    "scientific_source_package_contract_repair_v1"
)
LEGACY_PACKAGE_DIR = (
    _REPO_ROOT
    / "outputs/hu_joint_policy/m31_t3_step6d/performance_lock_v4/"
    "scientific_source_package"
)

CONTRACT_RELATIVE_PATH = "configs/hu_joint_policy_m31_t3_step6d_contract.json"
PERFORMANCE_CONTRACT_SHA256 = (
    "1924295b18070432cf3126159311102d9285dba37c498a3ea7666b0d5b777775"
)
LEGACY_PACKAGE_MANIFEST_FILE_SHA256 = (
    "d01933aaf861ab714d13a21d8961ba04ed749631af24894d71b428820f099ddd"
)
LEGACY_PACKAGE_READY_FILE_SHA256 = (
    "3322d1e70e27b6f143e4f2be0f9b37649d1e961b0544669c1d24e52d1ded8d10"
)
LEGACY_SOURCE_SHA256 = (
    "acfd573fd4ea033bf3e154b37b325a7a61fc49188a77b522a6494a0a4c8dacf9"
)

REQUIRED_SOURCE_RELATIVE_PATHS = frozenset(
    {
        "pyproject.toml",
        CONTRACT_RELATIVE_PATH,
        "src/ofc_regular/__init__.py",
        "src/ofc_regular/hu_m31_t3_behavior_roots.py",
        "src/ofc_regular/hu_m31_t3_step6d_contract.py",
        (
            "src/ofc_regular/"
            "hu_m31_t3_step6d_candidate02_performance_lock_v4_plan.py"
        ),
        "src/ofc_regular/hu_m31_t3_step6d_performance_lock_v4_spot_package.py",
        "src/ofc_regular/run_hu_m31_t3_step6d_performance.py",
        "src/ofc_regular/run_hu_m31_t3_step6d_performance_v2.py",
    }
)

_SHA = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "source_name",
        "source_sha256",
        "source_bytes",
        "plan_sha256",
        "run_contract_digest",
        "run_contract",
        "candidate_variant",
        "allocation",
        "materialization_receipt_sha256",
        "root_seal_sha256",
        "claim_sha256",
        "root_file_count",
        "root_hash_aggregate_sha256",
        "observation_fingerprint_aggregate_sha256",
        "seed_set_sha256",
        "source_entries",
        "source_entry_count",
        "accepted_candidate",
        "accepted_reference",
        "feature_encoder",
        "job_manifests",
        "job_count",
        "content_addressed",
        "source_archive_executable",
        "gcloud_invoked",
        "spot_execution_authorized",
        "cloud_started",
        "performance_lock_authorized",
        "quality_pilot_authorized",
        "training_eligible",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
        "m31_complete",
        "manifest_sha256",
    }
)
_READY_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "package_manifest_sha256",
        "source_sha256",
        "source_bytes",
        "plan_sha256",
        "run_contract_digest",
        "root_seal_sha256",
        "job_count",
        "cloud_started",
        "current_profile_changed",
    }
)
_FILE_RECORD_KEYS = frozenset({"path", "sha256", "bytes"})
_JOB_RECORD_KEYS = frozenset(
    {
        "job_id",
        "path",
        "sha256",
        "bytes",
        "source_role",
        "shard_index",
        "work_hand_indices",
    }
)


def canonical_bytes(value: Any) -> bytes:
    return runner.canonical_bytes(value)


def canonical_sha256(value: Any) -> str:
    return runner.canonical_sha256(value)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} keys changed")


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ValueError(f"{label} is not lowercase SHA-256")
    return value


def _safe_file(path: str | Path, label: str) -> Path:
    target = Path(path).resolve()
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    return target


def _safe_directory(path: str | Path, label: str) -> Path:
    target = Path(path).resolve()
    if target.is_symlink() or not target.is_dir():
        raise ValueError(f"{label} is missing or unsafe")
    return target


def _read_canonical_file(path: str | Path, label: str) -> dict[str, Any]:
    target = _safe_file(path, label)
    raw = target.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _read_canonical_bytes(raw: bytes, label: str) -> dict[str, Any]:
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _read_json_object_bytes(raw: bytes, label: str) -> dict[str, Any]:
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _record(path: str, raw: bytes) -> dict[str, Any]:
    return {"path": path, "sha256": sha256_bytes(raw), "bytes": len(raw)}


def _write_once(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _repository_source_files(repository_root: Path) -> dict[str, Path]:
    root = repository_root.resolve()
    files: dict[str, Path] = {}
    pyproject = _safe_file(root / "pyproject.toml", "pyproject")
    files["pyproject.toml"] = pyproject
    # run_hu_m31_t3_step6d_performance_v2 reads this contract from the extracted
    # science root, so the archive has to carry it alongside the Python source.
    files[CONTRACT_RELATIVE_PATH] = _safe_file(
        root / CONTRACT_RELATIVE_PATH, "performance contract"
    )
    package_root = _safe_directory(root / "src/ofc_regular", "ofc_regular source")
    for path in sorted(package_root.rglob("*.py")):
        if path.is_symlink() or not path.is_file():
            raise ValueError("repository Python source contains an unsafe path")
        relative = path.relative_to(root).as_posix()
        files[relative] = path.resolve()
    if not REQUIRED_SOURCE_RELATIVE_PATHS.issubset(files):
        raise ValueError("required performance-lock-v4 source file is missing")
    return files


def _job_payloads(plan: Mapping[str, Any]) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    payloads: dict[str, bytes] = {}
    records: list[dict[str, Any]] = []
    for frozen in plan["jobs"]:
        manifest = {
            "schema": runner.CANDIDATE02_PERFORMANCE_LOCK_V4_SHARD_MANIFEST_SCHEMA,
            "run_contract": dict(plan["run_contract"]),
            "run_contract_digest": v4.RUN_CONTRACT_DIGEST,
            "source_role": frozen["source_role"],
            "work_hand_indices": list(frozen["work_hand_indices"]),
        }
        raw = canonical_bytes(manifest)
        if sha256_bytes(raw) != frozen["shard_manifest_sha256"]:
            raise ValueError("frozen v4 job manifest digest changed")
        relative = f"{JOB_DIRECTORY}/{frozen['job_id']}.json"
        payloads[relative] = raw
        records.append(
            {
                "job_id": frozen["job_id"],
                "path": relative,
                "sha256": sha256_bytes(raw),
                "bytes": len(raw),
                "source_role": frozen["source_role"],
                "shard_index": frozen["shard_index"],
                "work_hand_indices": list(frozen["work_hand_indices"]),
            }
        )
    if len(records) != 20 or len(payloads) != 20:
        raise ValueError("performance-lock-v4 job coverage changed")
    return payloads, records


def _load_frozen_evidence(
    *,
    plan_path: Path,
    materialization_receipt_path: Path,
    root_seal_path: Path,
    root_output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    plan = v4.validate_performance_lock_v4_plan(
        _read_canonical_file(plan_path, "performance-lock-v4 plan")
    )
    if sha256_file(plan_path) != v4.PLAN_SHA256:
        raise ValueError("performance-lock-v4 plan file hash changed")
    materialization = v4.validate_materialization_receipt(
        _read_canonical_file(
            materialization_receipt_path,
            "performance-lock-v4 materialization receipt",
        )
    )
    seal = v4.validate_root_seal(
        _read_canonical_file(root_seal_path, "performance-lock-v4 root seal")
    )
    output = root_output_dir.resolve()
    claim_path = output / v4.CLAIM_NAME
    claim = v4._validate_claim(
        _read_canonical_file(claim_path, "performance-lock-v4 claim"),
        output_dir=output,
    )
    if (
        materialization["root_output_directory"] != str(output)
        or seal["root_output_directory"] != str(output)
        or materialization["claim_sha256"] != canonical_sha256(claim)
        or seal["claim_sha256"] != canonical_sha256(claim)
        or seal["materialization_receipt_sha256"]
        != canonical_sha256(materialization)
    ):
        raise ValueError("performance-lock-v4 evidence lineage changed")
    roots = v4._load_root_files(output_dir=output, contract=plan["run_contract"])
    audit = v4._audit_roots(
        roots=roots,
        contract=plan["run_contract"],
        current_profile_unchanged=True,
        validate_artifacts=False,
    )
    for key in v4._AUDIT_KEYS:
        if materialization[key] != audit[key] or seal[key] != audit[key]:
            raise ValueError("performance-lock-v4 sealed roots changed")
    return plan, materialization, seal, claim, roots


def _source_payloads(
    *,
    repository_root: Path,
    plan_path: Path,
    materialization_receipt_path: Path,
    root_seal_path: Path,
    root_output_dir: Path,
    plan: Mapping[str, Any],
    materialization: Mapping[str, Any],
    seal: Mapping[str, Any],
    claim: Mapping[str, Any],
    roots: Sequence[Mapping[str, Any]],
    candidate_library_path: Path,
    reference_library_path: Path,
    feature_encoder_path: Path,
) -> tuple[dict[str, bytes], dict[str, Any], dict[str, Any], dict[str, Any]]:
    payloads = {
        relative: path.read_bytes()
        for relative, path in _repository_source_files(repository_root).items()
    }
    payloads[PLAN_ARCHIVE_PATH] = plan_path.read_bytes()
    payloads[CLAIM_ARCHIVE_PATH] = canonical_bytes(claim)
    payloads[MATERIALIZATION_ARCHIVE_PATH] = materialization_receipt_path.read_bytes()
    payloads[SEAL_ARCHIVE_PATH] = root_seal_path.read_bytes()
    for index, root in enumerate(roots):
        raw = canonical_bytes(root)
        if sha256_bytes(raw) != seal["root_hashes"][index]:
            raise ValueError("sealed root bytes changed before packaging")
        payloads[ROOT_ARCHIVE_TEMPLATE.format(index=index)] = raw

    candidate = _safe_file(candidate_library_path, "accepted candidate library")
    reference = _safe_file(reference_library_path, "accepted reference library")
    feature = _safe_file(feature_encoder_path, "feature encoder")
    binaries = (
        (CANDIDATE_ARCHIVE_PATH, candidate, v4.CANDIDATE_LIBRARY_SHA256),
        (REFERENCE_ARCHIVE_PATH, reference, v4.REFERENCE_LIBRARY_SHA256),
        (FEATURE_ARCHIVE_PATH, feature, FEATURE_ENCODER_SHA256),
    )
    records = []
    for relative, path, expected_sha in binaries:
        raw = path.read_bytes()
        if sha256_bytes(raw) != expected_sha:
            raise ValueError(f"frozen binary changed: {relative}")
        if relative in payloads:
            raise ValueError(f"duplicate archive entry: {relative}")
        payloads[relative] = raw
        records.append(_record(relative, raw))
    if len(payloads) != len(set(payloads)):
        raise ValueError("source archive paths are duplicated")
    return payloads, records[0], records[1], records[2]


def _zip_bytes(payloads: Mapping[str, bytes], output: Path) -> None:
    with output.open("xb") as raw_handle:
        with zipfile.ZipFile(
            raw_handle, mode="w", compression=zipfile.ZIP_STORED, strict_timestamps=True
        ) as archive:
            for name in sorted(payloads):
                pure = PurePosixPath(name)
                if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != name:
                    raise ValueError("unsafe source archive path")
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = (0o100644 & 0xFFFF) << 16
                archive.writestr(info, payloads[name])
        raw_handle.flush()
        os.fsync(raw_handle.fileno())


def _manifest_without_digest(
    *,
    source_path: Path,
    plan: Mapping[str, Any],
    materialization: Mapping[str, Any],
    seal: Mapping[str, Any],
    claim: Mapping[str, Any],
    source_entries: Mapping[str, Any],
    candidate_record: Mapping[str, Any],
    reference_record: Mapping[str, Any],
    feature_record: Mapping[str, Any],
    job_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": PACKAGE_SCHEMA,
        "status": PACKAGE_STATUS,
        "run_name": RUN_NAME,
        "source_name": SOURCE_NAME,
        "source_sha256": sha256_file(source_path),
        "source_bytes": source_path.stat().st_size,
        "plan_sha256": v4.PLAN_SHA256,
        "run_contract_digest": v4.RUN_CONTRACT_DIGEST,
        "run_contract": dict(plan["run_contract"]),
        "candidate_variant": runner.CANDIDATE02_PERFORMANCE_LOCK_V4_VARIANT,
        "allocation": dict(plan["allocation"]),
        "materialization_receipt_sha256": canonical_sha256(materialization),
        "root_seal_sha256": canonical_sha256(seal),
        "claim_sha256": canonical_sha256(claim),
        "root_file_count": seal["root_file_count"],
        "root_hash_aggregate_sha256": seal["root_hash_aggregate_sha256"],
        "observation_fingerprint_aggregate_sha256": seal[
            "observation_fingerprint_aggregate_sha256"
        ],
        "seed_set_sha256": seal["seed_set_sha256"],
        "source_entries": dict(source_entries),
        "source_entry_count": len(source_entries),
        "accepted_candidate": dict(candidate_record),
        "accepted_reference": dict(reference_record),
        "feature_encoder": dict(feature_record),
        "job_manifests": [dict(row) for row in job_records],
        "job_count": 20,
        "content_addressed": True,
        "source_archive_executable": False,
        "gcloud_invoked": False,
        "spot_execution_authorized": False,
        "cloud_started": False,
        "performance_lock_authorized": False,
        "quality_pilot_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "m31_complete": False,
    }


def _validate_manifest_value(
    value: Mapping[str, Any],
    *,
    required_source_paths: frozenset[str] = REQUIRED_SOURCE_RELATIVE_PATHS,
) -> dict[str, Any]:
    manifest = dict(value)
    _exact_keys(manifest, _MANIFEST_KEYS, "performance-lock-v4 package manifest")
    digest = manifest.pop("manifest_sha256", None)
    _require_sha(digest, "package manifest digest")
    if digest != canonical_sha256(manifest):
        raise ValueError("performance-lock-v4 package manifest digest changed")
    source_entries = manifest.get("source_entries")
    jobs = manifest.get("job_manifests")
    if not isinstance(source_entries, Mapping) or not isinstance(jobs, list):
        raise ValueError("performance-lock-v4 package inventory is missing")
    for path, record in source_entries.items():
        if (
            not isinstance(path, str)
            or not isinstance(record, Mapping)
            or set(record) != {"sha256", "bytes"}
            or record.get("bytes", 0) <= 0
        ):
            raise ValueError("performance-lock-v4 source entry changed")
        _require_sha(record.get("sha256"), f"source entry {path}")
    for record in jobs:
        if not isinstance(record, Mapping):
            raise ValueError("performance-lock-v4 job record is not an object")
        _exact_keys(record, _JOB_RECORD_KEYS, "performance-lock-v4 job record")
        _require_sha(record.get("sha256"), "job manifest")
    for field in ("accepted_candidate", "accepted_reference", "feature_encoder"):
        record = manifest.get(field)
        if not isinstance(record, Mapping):
            raise ValueError(f"{field} record is missing")
        _exact_keys(record, _FILE_RECORD_KEYS, field)
        _require_sha(record.get("sha256"), field)
    if (
        manifest.get("schema") != PACKAGE_SCHEMA
        or manifest.get("status") != PACKAGE_STATUS
        or manifest.get("run_name") != RUN_NAME
        or manifest.get("source_name") != SOURCE_NAME
        or not isinstance(manifest.get("source_bytes"), int)
        or manifest["source_bytes"] <= 0
        or _require_sha(manifest.get("source_sha256"), "source archive")
        != manifest["source_sha256"]
        or manifest.get("plan_sha256") != v4.PLAN_SHA256
        or manifest.get("run_contract_digest") != v4.RUN_CONTRACT_DIGEST
        or runner.validate_run_contract(manifest.get("run_contract", {}))
        != manifest.get("run_contract")
        or manifest.get("candidate_variant")
        != runner.CANDIDATE02_PERFORMANCE_LOCK_V4_VARIANT
        or manifest.get("allocation") != {"workers": 1, "rayon_threads_per_worker": 16}
        or manifest.get("root_file_count") != 100
        or manifest.get("seed_set_sha256")
        != runner.CANDIDATE02_PERFORMANCE_LOCK_V4_SEED_SET_SHA256
        or manifest.get("source_entry_count") != len(source_entries)
        or not required_source_paths.issubset(source_entries)
        or manifest["accepted_candidate"]["sha256"] != v4.CANDIDATE_LIBRARY_SHA256
        or manifest["accepted_reference"]["sha256"] != v4.REFERENCE_LIBRARY_SHA256
        or manifest["feature_encoder"]["sha256"] != FEATURE_ENCODER_SHA256
        or manifest.get("job_count") != 20
        or len(jobs) != 20
        or manifest.get("content_addressed") is not True
        or manifest.get("source_archive_executable") is not False
        or any(
            manifest.get(field) is not False
            for field in (
                "gcloud_invoked",
                "spot_execution_authorized",
                "cloud_started",
                "performance_lock_authorized",
                "quality_pilot_authorized",
                "training_eligible",
                "current_profile_changed",
                "named_profile_added",
                "runtime_policy_activated",
                "m31_complete",
            )
        )
    ):
        raise ValueError("performance-lock-v4 package boundary changed")
    manifest["manifest_sha256"] = digest
    return manifest


def _validate_ready(
    value: Mapping[str, Any], *, manifest: Mapping[str, Any], manifest_path: Path
) -> dict[str, Any]:
    ready = dict(value)
    _exact_keys(ready, _READY_KEYS, "performance-lock-v4 package ready")
    expected = {
        "schema": READY_SCHEMA,
        "status": READY_STATUS,
        "run_name": RUN_NAME,
        "package_manifest_sha256": sha256_file(manifest_path),
        "source_sha256": manifest["source_sha256"],
        "source_bytes": manifest["source_bytes"],
        "plan_sha256": v4.PLAN_SHA256,
        "run_contract_digest": v4.RUN_CONTRACT_DIGEST,
        "root_seal_sha256": manifest["root_seal_sha256"],
        "job_count": 20,
        "cloud_started": False,
        "current_profile_changed": False,
    }
    if ready != expected:
        raise ValueError("performance-lock-v4 package ready receipt changed")
    return ready


def _validate_archived_claim(value: Mapping[str, Any]) -> dict[str, Any]:
    claim = dict(value)
    if (
        set(claim) != v4._CLAIM_KEYS
        or claim.get("schema") != v4.CLAIM_SCHEMA
        or claim.get("status") != v4.CLAIM_STATUS
        or claim.get("plan_sha256") != v4.PLAN_SHA256
        or claim.get("run_contract_digest") != v4.RUN_CONTRACT_DIGEST
        or claim.get("authorizing_gate_receipt_file_sha256")
        != v4.RUN009_GATE_RECEIPT_FILE_SHA256
        or claim.get("authorizing_gate_receipt_sha256")
        != v4.RUN009_GATE_RECEIPT_SHA256
        or not isinstance(claim.get("root_output_directory"), str)
        or not claim["root_output_directory"]
        or claim.get("current_profile_registry_sha256")
        != v4.CURRENT_PROFILE_REGISTRY_SHA256
        or claim.get("root_content_opened_at_claim") is not False
        or claim.get("cloud_started") is not False
        or claim.get("training_authorized") is not False
        or claim.get("current_profile_changed") is not False
    ):
        raise ValueError("archived performance-lock-v4 claim changed")
    return claim


def _validate_archive(
    source_path: Path, manifest: Mapping[str, Any]
) -> dict[str, bytes]:
    expected_entries = manifest["source_entries"]
    payloads: dict[str, bytes] = {}
    try:
        with zipfile.ZipFile(source_path, mode="r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or set(names) != set(expected_entries):
                raise ValueError("source archive entry inventory changed")
            for info in infos:
                pure = PurePosixPath(info.filename)
                mode = info.external_attr >> 16
                if (
                    info.is_dir()
                    or pure.is_absolute()
                    or ".." in pure.parts
                    or pure.as_posix() != info.filename
                    or (mode and not stat.S_ISREG(mode))
                ):
                    raise ValueError("source archive contains an unsafe entry")
                raw = archive.read(info)
                expected = expected_entries[info.filename]
                if (
                    len(raw) != expected["bytes"]
                    or sha256_bytes(raw) != expected["sha256"]
                ):
                    raise ValueError("source archive entry bytes changed")
                payloads[info.filename] = raw
    except zipfile.BadZipFile as exc:
        raise ValueError("performance-lock-v4 source archive is invalid") from exc
    return payloads


def _validate_archived_science(
    payloads: Mapping[str, bytes],
    manifest: Mapping[str, Any],
    *,
    require_performance_contract: bool = True,
) -> dict[str, Any]:
    plan = v4.validate_performance_lock_v4_plan(
        _read_canonical_bytes(payloads[PLAN_ARCHIVE_PATH], "archived v4 plan")
    )
    materialization = v4.validate_materialization_receipt(
        _read_canonical_bytes(
            payloads[MATERIALIZATION_ARCHIVE_PATH],
            "archived v4 materialization",
        )
    )
    seal = v4.validate_root_seal(
        _read_canonical_bytes(payloads[SEAL_ARCHIVE_PATH], "archived v4 seal")
    )
    claim = _validate_archived_claim(
        _read_canonical_bytes(payloads[CLAIM_ARCHIVE_PATH], "archived v4 claim")
    )
    if (
        sha256_bytes(payloads[PLAN_ARCHIVE_PATH]) != v4.PLAN_SHA256
        or canonical_sha256(materialization)
        != manifest["materialization_receipt_sha256"]
        or canonical_sha256(seal) != manifest["root_seal_sha256"]
        or canonical_sha256(claim) != manifest["claim_sha256"]
        or materialization["claim_sha256"] != canonical_sha256(claim)
        or seal["claim_sha256"] != canonical_sha256(claim)
        or seal["materialization_receipt_sha256"]
        != canonical_sha256(materialization)
    ):
        raise ValueError("archived performance-lock-v4 evidence lineage changed")
    roots = []
    for index in range(100):
        relative = ROOT_ARCHIVE_TEMPLATE.format(index=index)
        raw = payloads.get(relative)
        if raw is None or sha256_bytes(raw) != seal["root_hashes"][index]:
            raise ValueError("archived performance-lock-v4 root bytes changed")
        root = _read_canonical_bytes(raw, f"archived v4 root {index}")
        roots.append(root)
    audit = v4._audit_roots(
        roots=roots,
        contract=plan["run_contract"],
        current_profile_unchanged=True,
    )
    for key in v4._AUDIT_KEYS:
        if materialization[key] != audit[key] or seal[key] != audit[key]:
            raise ValueError("archived performance-lock-v4 root seal changed")
    binary_records = (
        ("accepted_candidate", CANDIDATE_ARCHIVE_PATH, v4.CANDIDATE_LIBRARY_SHA256),
        ("accepted_reference", REFERENCE_ARCHIVE_PATH, v4.REFERENCE_LIBRARY_SHA256),
        ("feature_encoder", FEATURE_ARCHIVE_PATH, FEATURE_ENCODER_SHA256),
    )
    for field, relative, expected_sha in binary_records:
        raw = payloads.get(relative)
        record = manifest[field]
        if (
            raw is None
            or record["path"] != relative
            or record["sha256"] != expected_sha
            or record["bytes"] != len(raw)
            or sha256_bytes(raw) != expected_sha
        ):
            raise ValueError(f"archived {field} changed")
    profile_record = manifest["source_entries"].get(
        "src/ofc_regular/ai_profiles.py"
    )
    if (
        profile_record is None
        or profile_record["sha256"] != v4.CURRENT_PROFILE_REGISTRY_SHA256
    ):
        raise ValueError("archived current profile registry changed")
    if require_performance_contract:
        contract_raw = payloads.get(CONTRACT_RELATIVE_PATH)
        contract_record = manifest["source_entries"].get(CONTRACT_RELATIVE_PATH)
        if (
            contract_raw is None
            or contract_record is None
            or contract_record["sha256"] != PERFORMANCE_CONTRACT_SHA256
            or contract_record["bytes"] != len(contract_raw)
            or sha256_bytes(contract_raw) != PERFORMANCE_CONTRACT_SHA256
        ):
            raise ValueError("archived performance contract changed")
        _read_json_object_bytes(contract_raw, "archived performance contract")
    return plan


def _validate_job_files(
    root: Path, manifest: Mapping[str, Any], plan: Mapping[str, Any]
) -> None:
    frozen_jobs = {row["job_id"]: row for row in plan["jobs"]}
    expected_payloads, expected_records = _job_payloads(plan)
    expected_by_id = {row["job_id"]: row for row in expected_records}
    seen: set[str] = set()
    for record in manifest["job_manifests"]:
        job_id = str(record["job_id"])
        frozen = frozen_jobs.get(job_id)
        expected_record = expected_by_id.get(job_id)
        path = _safe_file(root / record["path"], f"job manifest {job_id}")
        raw = path.read_bytes()
        if (
            job_id in seen
            or frozen is None
            or expected_record is None
            or record != expected_record
            or raw != expected_payloads[record["path"]]
            or sha256_bytes(raw) != record["sha256"]
            or len(raw) != record["bytes"]
            or record["source_role"] != frozen["source_role"]
            or record["shard_index"] != frozen["shard_index"]
            or record["work_hand_indices"] != frozen["work_hand_indices"]
        ):
            raise ValueError("performance-lock-v4 job manifest changed")
        seen.add(job_id)
    if len(seen) != 20:
        raise ValueError("performance-lock-v4 job coverage changed")


def _validate_exact_tree(root: Path, manifest: Mapping[str, Any]) -> None:
    expected_files = {
        SOURCE_NAME,
        MANIFEST_NAME,
        READY_NAME,
        *(record["path"] for record in manifest["job_manifests"]),
    }
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("performance-lock-v4 package contains a symlink")
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            actual_directories.add(relative)
        elif path.is_file():
            actual_files.add(relative)
        else:
            raise ValueError("performance-lock-v4 package contains an unsafe node")
    if actual_files != expected_files or actual_directories != {JOB_DIRECTORY}:
        raise ValueError("performance-lock-v4 package tree changed")


def validate_package(run_dir: str | Path) -> dict[str, Any]:
    """Rehash and replay every immutable package input."""

    root = _safe_directory(run_dir, "performance-lock-v4 package")
    manifest_path = _safe_file(root / MANIFEST_NAME, "package manifest")
    manifest = _validate_manifest_value(
        _read_canonical_file(manifest_path, "package manifest")
    )
    ready = _validate_ready(
        _read_canonical_file(root / READY_NAME, "package ready"),
        manifest=manifest,
        manifest_path=manifest_path,
    )
    source = _safe_file(root / SOURCE_NAME, "source archive")
    if (
        sha256_file(source) != manifest["source_sha256"]
        or source.stat().st_size != manifest["source_bytes"]
        or ready["source_sha256"] != manifest["source_sha256"]
    ):
        raise ValueError("performance-lock-v4 source archive changed")
    _validate_exact_tree(root, manifest)
    payloads = _validate_archive(source, manifest)
    quick_plan = v4.validate_performance_lock_v4_plan(
        _read_canonical_bytes(payloads[PLAN_ARCHIVE_PATH], "archived v4 plan")
    )
    _validate_job_files(root, manifest, quick_plan)
    _validate_archived_science(payloads, manifest)
    return manifest


def _validate_legacy_package_for_contract_repair(
    legacy_package_dir: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, bytes]]:
    """Validate the pinned prelaunch package that omitted one runtime config."""

    root = _safe_directory(legacy_package_dir, "legacy performance-lock-v4 package")
    manifest_path = _safe_file(root / MANIFEST_NAME, "legacy package manifest")
    ready_path = _safe_file(root / READY_NAME, "legacy package ready")
    source_path = _safe_file(root / SOURCE_NAME, "legacy source archive")
    if (
        sha256_file(manifest_path) != LEGACY_PACKAGE_MANIFEST_FILE_SHA256
        or sha256_file(ready_path) != LEGACY_PACKAGE_READY_FILE_SHA256
        or sha256_file(source_path) != LEGACY_SOURCE_SHA256
    ):
        raise ValueError("legacy performance-lock-v4 package pin changed")
    legacy_required = frozenset(
        REQUIRED_SOURCE_RELATIVE_PATHS - {CONTRACT_RELATIVE_PATH}
    )
    manifest = _validate_manifest_value(
        _read_canonical_file(manifest_path, "legacy package manifest"),
        required_source_paths=legacy_required,
    )
    if (
        set(REQUIRED_SOURCE_RELATIVE_PATHS) - set(manifest["source_entries"])
        != {CONTRACT_RELATIVE_PATH}
        or CONTRACT_RELATIVE_PATH in manifest["source_entries"]
    ):
        raise ValueError("legacy package is not the one-file contract omission")
    _validate_ready(
        _read_canonical_file(ready_path, "legacy package ready"),
        manifest=manifest,
        manifest_path=manifest_path,
    )
    if (
        manifest["source_sha256"] != LEGACY_SOURCE_SHA256
        or source_path.stat().st_size != manifest["source_bytes"]
    ):
        raise ValueError("legacy source archive pin changed")
    _validate_exact_tree(root, manifest)
    payloads = _validate_archive(source_path, manifest)
    quick_plan = v4.validate_performance_lock_v4_plan(
        _read_canonical_bytes(payloads[PLAN_ARCHIVE_PATH], "legacy archived v4 plan")
    )
    _validate_job_files(root, manifest, quick_plan)
    _validate_archived_science(
        payloads,
        manifest,
        require_performance_contract=False,
    )
    return root, manifest, payloads


def repair_legacy_package_contract(
    *,
    destination: str | Path = DEFAULT_PACKAGE_DIR,
    legacy_package_dir: str | Path = LEGACY_PACKAGE_DIR,
    repository_root: str | Path = _REPO_ROOT,
    production_run_dir: str | Path,
) -> dict[str, Any]:
    """Create a new immutable package by adding only the pinned runtime config.

    This repair is prelaunch-only.  The caller must provide the intended
    production run directory, and creation fails if that directory already
    exists.  The legacy package remains untouched.
    """

    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError(
            f"performance-lock-v4 package already exists: {target}"
        )
    production = Path(production_run_dir).resolve()
    if production.exists():
        raise FileExistsError(
            f"performance-lock-v4 production run already exists: {production}"
        )
    repository = _safe_directory(repository_root, "repository root")
    profile_path = _safe_file(
        repository / "src/ofc_regular/ai_profiles.py",
        "current profile registry",
    )
    profile_before = sha256_file(profile_path)
    if profile_before != v4.CURRENT_PROFILE_REGISTRY_SHA256:
        raise ValueError("current profile registry changed before package repair")
    legacy_root, legacy_manifest, legacy_payloads = (
        _validate_legacy_package_for_contract_repair(legacy_package_dir)
    )
    contract_path = _safe_file(
        repository / CONTRACT_RELATIVE_PATH,
        "performance contract",
    )
    if sha256_file(contract_path) != PERFORMANCE_CONTRACT_SHA256:
        raise ValueError("performance contract pin changed before package repair")
    performance_v1.validate_performance_contract(contract_path)
    contract_raw = contract_path.read_bytes()
    _read_json_object_bytes(contract_raw, "performance contract")

    payloads = dict(legacy_payloads)
    if CONTRACT_RELATIVE_PATH in payloads:
        raise ValueError("legacy package unexpectedly contains performance contract")
    payloads[CONTRACT_RELATIVE_PATH] = contract_raw

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink():
        raise ValueError("performance-lock-v4 package parent is unsafe")
    target.mkdir()
    source_path = target / SOURCE_NAME
    _zip_bytes(payloads, source_path)
    for record in legacy_manifest["job_manifests"]:
        raw = _safe_file(
            legacy_root / record["path"],
            f"legacy job manifest {record['job_id']}",
        ).read_bytes()
        if (
            len(raw) != record["bytes"]
            or sha256_bytes(raw) != record["sha256"]
        ):
            raise ValueError("legacy job manifest changed during package repair")
        _write_once(target / record["path"], raw)

    manifest = {
        key: value
        for key, value in legacy_manifest.items()
        if key != "manifest_sha256"
    }
    source_entries = dict(manifest["source_entries"])
    source_entries[CONTRACT_RELATIVE_PATH] = {
        "sha256": PERFORMANCE_CONTRACT_SHA256,
        "bytes": len(contract_raw),
    }
    manifest["source_entries"] = dict(sorted(source_entries.items()))
    manifest["source_entry_count"] = len(source_entries)
    manifest["source_sha256"] = sha256_file(source_path)
    manifest["source_bytes"] = source_path.stat().st_size
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest = _validate_manifest_value(manifest)
    manifest_path = target / MANIFEST_NAME
    _write_once(manifest_path, canonical_bytes(manifest))
    ready = {
        "schema": READY_SCHEMA,
        "status": READY_STATUS,
        "run_name": RUN_NAME,
        "package_manifest_sha256": sha256_file(manifest_path),
        "source_sha256": manifest["source_sha256"],
        "source_bytes": manifest["source_bytes"],
        "plan_sha256": v4.PLAN_SHA256,
        "run_contract_digest": v4.RUN_CONTRACT_DIGEST,
        "root_seal_sha256": manifest["root_seal_sha256"],
        "job_count": 20,
        "cloud_started": False,
        "current_profile_changed": False,
    }
    _validate_ready(ready, manifest=manifest, manifest_path=manifest_path)
    _write_once(target / READY_NAME, canonical_bytes(ready))
    validated = validate_package(target)

    repaired_payloads = _validate_archive(source_path, validated)
    if (
        set(repaired_payloads) - set(legacy_payloads) != {CONTRACT_RELATIVE_PATH}
        or set(legacy_payloads) - set(repaired_payloads)
        or any(
            repaired_payloads[name] != raw
            for name, raw in legacy_payloads.items()
        )
        or repaired_payloads[CONTRACT_RELATIVE_PATH] != contract_raw
    ):
        raise ValueError("package repair changed more than the runtime contract")
    if sha256_file(profile_path) != profile_before:
        raise ValueError("current profile registry changed during package repair")
    return validated


def create_package(
    *,
    destination: str | Path = DEFAULT_PACKAGE_DIR,
    repository_root: str | Path = _REPO_ROOT,
    plan_path: str | Path = v4.DEFAULT_PLAN_PATH,
    materialization_receipt_path: str
    | Path = v4.DEFAULT_MATERIALIZATION_RECEIPT_PATH,
    root_seal_path: str | Path = v4.DEFAULT_ROOT_SEAL_PATH,
    root_output_dir: str | Path = v4.DEFAULT_ROOT_OUTPUT_DIR,
    candidate_library_path: str | Path = DEFAULT_CANDIDATE_LIBRARY_PATH,
    reference_library_path: str | Path = DEFAULT_REFERENCE_LIBRARY_PATH,
    feature_encoder_path: str | Path = DEFAULT_FEATURE_ENCODER_PATH,
) -> dict[str, Any]:
    """Create one immutable local package and validate it before returning."""

    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError(
            f"performance-lock-v4 package already exists: {target}"
        )
    repository = _safe_directory(repository_root, "repository root")
    plan_file = _safe_file(plan_path, "performance-lock-v4 plan")
    materialization_file = _safe_file(
        materialization_receipt_path,
        "performance-lock-v4 materialization receipt",
    )
    seal_file = _safe_file(root_seal_path, "performance-lock-v4 root seal")
    root_output = _safe_directory(root_output_dir, "performance-lock-v4 root output")
    profile_before = sha256_file(repository / "src/ofc_regular/ai_profiles.py")
    if profile_before != v4.CURRENT_PROFILE_REGISTRY_SHA256:
        raise ValueError("current profile registry changed before packaging")
    plan, materialization, seal, claim, roots = _load_frozen_evidence(
        plan_path=plan_file,
        materialization_receipt_path=materialization_file,
        root_seal_path=seal_file,
        root_output_dir=root_output,
    )
    payloads, candidate, reference, feature = _source_payloads(
        repository_root=repository,
        plan_path=plan_file,
        materialization_receipt_path=materialization_file,
        root_seal_path=seal_file,
        root_output_dir=root_output,
        plan=plan,
        materialization=materialization,
        seal=seal,
        claim=claim,
        roots=roots,
        candidate_library_path=Path(candidate_library_path),
        reference_library_path=Path(reference_library_path),
        feature_encoder_path=Path(feature_encoder_path),
    )
    profile_source = payloads.get("src/ofc_regular/ai_profiles.py")
    if (
        profile_source is None
        or sha256_bytes(profile_source) != v4.CURRENT_PROFILE_REGISTRY_SHA256
    ):
        raise ValueError("current profile registry changed in source inventory")
    source_entries = {
        name: {"sha256": sha256_bytes(raw), "bytes": len(raw)}
        for name, raw in sorted(payloads.items())
    }
    job_payloads, jobs = _job_payloads(plan)

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink():
        raise ValueError("performance-lock-v4 package parent is unsafe")
    target.mkdir()
    source_path = target / SOURCE_NAME
    _zip_bytes(payloads, source_path)
    for relative, raw in job_payloads.items():
        _write_once(target / relative, raw)
    manifest = _manifest_without_digest(
        source_path=source_path,
        plan=plan,
        materialization=materialization,
        seal=seal,
        claim=claim,
        source_entries=source_entries,
        candidate_record=candidate,
        reference_record=reference,
        feature_record=feature,
        job_records=jobs,
    )
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest = _validate_manifest_value(manifest)
    manifest_path = target / MANIFEST_NAME
    _write_once(manifest_path, canonical_bytes(manifest))
    ready = {
        "schema": READY_SCHEMA,
        "status": READY_STATUS,
        "run_name": RUN_NAME,
        "package_manifest_sha256": sha256_file(manifest_path),
        "source_sha256": manifest["source_sha256"],
        "source_bytes": manifest["source_bytes"],
        "plan_sha256": v4.PLAN_SHA256,
        "run_contract_digest": v4.RUN_CONTRACT_DIGEST,
        "root_seal_sha256": manifest["root_seal_sha256"],
        "job_count": 20,
        "cloud_started": False,
        "current_profile_changed": False,
    }
    _validate_ready(ready, manifest=manifest, manifest_path=manifest_path)
    _write_once(target / READY_NAME, canonical_bytes(ready))
    validated = validate_package(target)
    profile_after = sha256_file(repository / "src/ofc_regular/ai_profiles.py")
    if profile_after != profile_before:
        raise ValueError("current profile registry changed while packaging")
    return validated


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or validate the local performance-lock-v4 source package"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--destination", type=Path, default=DEFAULT_PACKAGE_DIR)
    create.add_argument("--repository-root", type=Path, default=_REPO_ROOT)
    create.add_argument("--plan", type=Path, default=v4.DEFAULT_PLAN_PATH)
    create.add_argument(
        "--materialization-receipt",
        type=Path,
        default=v4.DEFAULT_MATERIALIZATION_RECEIPT_PATH,
    )
    create.add_argument("--root-seal", type=Path, default=v4.DEFAULT_ROOT_SEAL_PATH)
    create.add_argument(
        "--root-output", type=Path, default=v4.DEFAULT_ROOT_OUTPUT_DIR
    )
    create.add_argument(
        "--candidate-library",
        type=Path,
        default=DEFAULT_CANDIDATE_LIBRARY_PATH,
    )
    create.add_argument(
        "--reference-library",
        type=Path,
        default=DEFAULT_REFERENCE_LIBRARY_PATH,
    )
    create.add_argument(
        "--feature-encoder",
        type=Path,
        default=DEFAULT_FEATURE_ENCODER_PATH,
    )
    repair = subparsers.add_parser("repair-contract")
    repair.add_argument("--destination", type=Path, default=DEFAULT_PACKAGE_DIR)
    repair.add_argument(
        "--legacy-package",
        type=Path,
        default=LEGACY_PACKAGE_DIR,
    )
    repair.add_argument("--repository-root", type=Path, default=_REPO_ROOT)
    repair.add_argument("--production-run-dir", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("package_dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "create":
        result = create_package(
            destination=args.destination,
            repository_root=args.repository_root,
            plan_path=args.plan,
            materialization_receipt_path=args.materialization_receipt,
            root_seal_path=args.root_seal,
            root_output_dir=args.root_output,
            candidate_library_path=args.candidate_library,
            reference_library_path=args.reference_library,
            feature_encoder_path=args.feature_encoder,
        )
    elif args.command == "repair-contract":
        result = repair_legacy_package_contract(
            destination=args.destination,
            legacy_package_dir=args.legacy_package,
            repository_root=args.repository_root,
            production_run_dir=args.production_run_dir,
        )
    else:
        result = validate_package(args.package_dir)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CANDIDATE_LIBRARY_PATH",
    "DEFAULT_FEATURE_ENCODER_PATH",
    "DEFAULT_PACKAGE_DIR",
    "DEFAULT_REFERENCE_LIBRARY_PATH",
    "LEGACY_PACKAGE_DIR",
    "MANIFEST_NAME",
    "PACKAGE_SCHEMA",
    "PERFORMANCE_CONTRACT_SHA256",
    "READY_NAME",
    "RUN_NAME",
    "SOURCE_NAME",
    "canonical_bytes",
    "canonical_sha256",
    "create_package",
    "main",
    "repair_legacy_package_contract",
    "sha256_file",
    "validate_package",
]
