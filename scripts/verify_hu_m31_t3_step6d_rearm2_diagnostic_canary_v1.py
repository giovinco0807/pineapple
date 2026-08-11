#!/usr/bin/env python3
"""Pre-content verifier for the bounded rearm2 diagnostic worker package.

The verifier intentionally validates every package identity needed to select a
diagnostic job before it opens a development-root member.  It is self-contained
so a VM startup script does not have to import untrusted archive code.
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


PACKAGE_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_cloud_worker_package_v1"
)
PACKAGE_STATUS = (
    "immutable_diagnostic_cloud_worker_package_ready_not_authorized"
)
JOB_SCHEMA = "hu_m31_t3_step6d_performance_shard_manifest_v2"
ROOT_PREFIX = (
    "outputs/hu_joint_policy/m31_t3_step6d/candidate02_development/"
    "tail_reselection_v2/roots/"
)
EXPECTED_JOBS = {
    "candidate-shard-00": {
        "stage_id": "stage1_lifecycle_one_candidate_vm",
        "run_name": "regular-hu-m31-r2diag-s1-20260718-001",
        "source_role": "candidate",
        "work_hand_indices": [0, 10, 13, 43, 49, 62, 66, 81, 82, 99],
    },
    "candidate-shard-01": {
        "stage_id": "stage2_candidate_reference_pair",
        "run_name": "regular-hu-m31-r2diag-s2-20260718-001",
        "source_role": "candidate",
        "work_hand_indices": [5, 6, 35, 39, 47, 53, 76, 83, 87, 89],
    },
    "reference-shard-01": {
        "stage_id": "stage2_candidate_reference_pair",
        "run_name": "regular-hu-m31-r2diag-s2-20260718-001",
        "source_role": "reference",
        "work_hand_indices": [5, 6, 35, 39, 47, 53, 76, 83, 87, 89],
    },
}
EXPECTED_BINARY_SHA256 = {
    "native/candidate/release/libofc_hu_m3_engine.so": (
        "4050e04b22d7943da9e1691402a2b34cf3d3781041a44557f35e2dfcf366d55d"
    ),
    "native/reference/release/libofc_hu_m3_engine.so": (
        "3f831615d9e751b8e1f266f14dd2009903b3ac2a4094f7e47616e0aa372a01b0"
    ),
    "target/release/libofc_stage3_feature_encoder.so": (
        "82510e563ee290cd536e7aebe6bcf0efefac061af5488851f0a582bb4ef83411"
    ),
}
EXPECTED_RUNTIME_CLOSURE_SHA256 = (
    "9894c508e028792d36238eced2a3370ea78acab03a00a1b467c92932f4f51ad6"
)
FORBIDDEN_CAPABILITIES = (
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
    "current_profile_changed",
    "runtime_policy_activated",
)
FORBIDDEN_KEY_PARTS = (
    "opponent_private_discard",
    "opponent_hidden",
    "hidden_truth",
    "realized_deck_tail",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase sha256")
    return value


def _read_canonical(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} must be a canonical JSON object")
    return value, raw


def _safe_zip_name(value: str) -> None:
    pure = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or value.startswith("/")
        or pure.is_absolute()
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise ValueError("diagnostic source contains an unsafe path")


def _is_regular_zip_member(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    return (
        not info.is_dir()
        and stat.S_IFMT(mode) == stat.S_IFREG
        and mode & 0o777 == 0o644
    )


def _scan_forbidden_keys(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).casefold()
            if any(part in key for part in FORBIDDEN_KEY_PARTS):
                raise ValueError(f"forbidden hidden field at {path}.{raw_key}")
            _scan_forbidden_keys(child, f"{path}.{raw_key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_forbidden_keys(child, f"{path}[{index}]")


def _read_member(
    archive: zipfile.ZipFile,
    name: str,
    *,
    poison_root_reads: bool,
) -> bytes:
    if name.startswith(ROOT_PREFIX):
        if poison_root_reads:
            raise AssertionError(
                "pre-content verifier attempted to open a development root"
            )
        raise AssertionError(
            "pre-content verifier implementation must never open root members"
        )
    return archive.read(name)


def _validate_elf(raw: bytes, label: str) -> None:
    header = raw[:20]
    if (
        len(header) != 20
        or header[:4] != b"\x7fELF"
        or header[4] != 2
        or header[5] != 1
        or header[16:18] != b"\x03\x00"
        or header[18:20] != b"\x3e\x00"
    ):
        raise ValueError(f"{label} is not a Linux x86_64 shared library")


def verify_package_precontent(
    *,
    source_path: str | Path,
    manifest_path: str | Path,
    job_path: str | Path,
    expected_source_sha256: str,
    expected_manifest_sha256: str,
    expected_job_sha256: str,
    expected_job_id: str,
    expected_stage_id: str,
    poison_root_reads: bool = False,
) -> dict[str, Any]:
    """Validate a single exact diagnostic job without opening any root bytes."""

    source = Path(source_path)
    manifest_file = Path(manifest_path)
    job_file = Path(job_path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("diagnostic source is missing or unsafe")
    expected_source = _sha(expected_source_sha256, "expected source")
    expected_manifest = _sha(expected_manifest_sha256, "expected manifest")
    expected_job = _sha(expected_job_sha256, "expected job")
    if sha256_file(source) != expected_source:
        raise ValueError("diagnostic source SHA-256 mismatch before content")
    manifest, manifest_raw = _read_canonical(
        manifest_file, "diagnostic package manifest"
    )
    job, job_raw = _read_canonical(job_file, "diagnostic job manifest")
    if hashlib.sha256(manifest_raw).hexdigest() != expected_manifest:
        raise ValueError("diagnostic manifest SHA-256 mismatch")
    if hashlib.sha256(job_raw).hexdigest() != expected_job:
        raise ValueError("diagnostic job SHA-256 mismatch")
    expected = EXPECTED_JOBS.get(expected_job_id)
    if expected is None or expected_stage_id != expected["stage_id"]:
        raise ValueError("diagnostic job escaped the exact three-job schedule")
    if (
        manifest.get("schema") != PACKAGE_SCHEMA
        or manifest.get("status") != PACKAGE_STATUS
        or manifest.get("source_sha256") != expected_source
        or manifest.get("source_name") != source.name
        or manifest.get("logical_job_count") != 3
        or manifest.get("development_root_count") != 20
        or manifest.get("runtime_closure_sha256")
        != EXPECTED_RUNTIME_CLOSURE_SHA256
        or manifest.get("cloud_executable") is not False
        or manifest.get("launch_ready") is not False
        or manifest.get("rearm2_production_package_reused") is not False
        or manifest.get("rearm2_locked_roots_used") is not False
        or any(manifest.get(key) is not False for key in FORBIDDEN_CAPABILITIES)
    ):
        raise ValueError("diagnostic package authorization boundary changed")
    _scan_forbidden_keys(manifest)
    records = manifest.get("job_manifests")
    record = next(
        (
            item
            for item in records
            if isinstance(item, dict) and item.get("job_id") == expected_job_id
        ),
        None,
    ) if isinstance(records, list) else None
    if (
        record is None
        or record.get("stage_id") != expected_stage_id
        or record.get("run_name") != expected["run_name"]
        or record.get("source_role") != expected["source_role"]
        or record.get("work_hand_indices") != expected["work_hand_indices"]
        or record.get("sha256") != expected_job
        or record.get("bytes") != len(job_raw)
        or record.get("path") != f"jobs/{expected_job_id}.json"
    ):
        raise ValueError("diagnostic job record changed")
    if (
        job.get("schema") != JOB_SCHEMA
        or job.get("source_role") != expected["source_role"]
        or job.get("work_hand_indices") != expected["work_hand_indices"]
        or job.get("run_contract_digest") != manifest.get("run_contract_digest")
        or job.get("run_contract") != manifest.get("run_contract")
        or any(
            job.get("run_contract", {}).get(key) is not False
            for key in (
                "current_profile_changed",
                "promotion_evidence",
                "quality_evidence",
                "training_eligible",
            )
        )
    ):
        raise ValueError("runner-compatible diagnostic job changed")
    _scan_forbidden_keys(job)

    source_entries = manifest.get("source_entries")
    if not isinstance(source_entries, dict):
        raise ValueError("diagnostic source entries are missing")
    closure = manifest.get("runtime_closure_records")
    python_allowlist = manifest.get("python_allowlist")
    if (
        not isinstance(closure, list)
        or not isinstance(python_allowlist, list)
        or len(closure) != 34
        or [row.get("path") for row in closure if isinstance(row, dict)]
        != [
            *python_allowlist,
            "configs/hu_joint_policy_m31_t3_step6d_contract.json",
        ]
        or any(
            not isinstance(row, dict)
            or set(row) != {"bytes", "path", "sha256"}
            or source_entries.get(row["path"]) is None
            or source_entries[row["path"]].get("bytes") != row["bytes"]
            or source_entries[row["path"]].get("sha256") != row["sha256"]
            for row in closure
        )
        or canonical_sha256(closure) != EXPECTED_RUNTIME_CLOSURE_SHA256
    ):
        raise ValueError("accepted runtime closure identity changed")
    expected_names = sorted(source_entries)
    root_names = sorted(
        f"{ROOT_PREFIX}hand_{index:03d}.json"
        for index in sorted(
            {
                hand
                for row in EXPECTED_JOBS.values()
                for hand in row["work_hand_indices"]
            }
        )
    )
    if (
        len(root_names) != 20
        or manifest.get("root_member_paths") != root_names
        or set(EXPECTED_BINARY_SHA256) - set(expected_names)
    ):
        raise ValueError("diagnostic root or binary file set changed")

    opened_members: list[str] = []
    with zipfile.ZipFile(source) as archive:
        infos = archive.infolist()
        if [info.filename for info in infos] != expected_names:
            raise ValueError("diagnostic source central-directory order changed")
        if len({info.filename for info in infos}) != len(infos):
            raise ValueError("diagnostic source has duplicate members")
        for info in infos:
            _safe_zip_name(info.filename)
            if not _is_regular_zip_member(info):
                raise ValueError("diagnostic source has a non-regular member")
            entry = source_entries.get(info.filename)
            if (
                not isinstance(entry, dict)
                or set(entry) != {"bytes", "kind", "sha256"}
                or entry.get("bytes") != info.file_size
            ):
                raise ValueError("diagnostic source entry metadata changed")
            if info.filename.startswith(ROOT_PREFIX):
                if info.filename not in root_names or entry.get("kind") != "root":
                    raise ValueError("diagnostic root central-directory entry changed")
                continue
            raw = _read_member(
                archive,
                info.filename,
                poison_root_reads=poison_root_reads,
            )
            opened_members.append(info.filename)
            if (
                len(raw) != entry["bytes"]
                or hashlib.sha256(raw).hexdigest() != entry["sha256"]
            ):
                raise ValueError("diagnostic non-root source member changed")
            if info.filename in EXPECTED_BINARY_SHA256:
                if entry["sha256"] != EXPECTED_BINARY_SHA256[info.filename]:
                    raise ValueError("accepted native artifact SHA-256 changed")
                _validate_elf(raw, info.filename)
    if any(name.startswith(ROOT_PREFIX) for name in opened_members):
        raise AssertionError("pre-content verifier opened a root member")
    return {
        "schema": (
            "hu_m31_t3_step6d_rearm2_diagnostic_precontent_verification_v1"
        ),
        "status": "accepted_before_any_development_root_member_open",
        "job_id": expected_job_id,
        "stage_id": expected_stage_id,
        "source_sha256": expected_source,
        "manifest_sha256": expected_manifest,
        "job_sha256": expected_job,
        "central_directory_member_count": len(expected_names),
        "root_member_count": len(root_names),
        "root_members_opened": 0,
        "non_root_members_verified": len(opened_members),
        "accepted_elf_count": len(EXPECTED_BINARY_SHA256),
        "poison_root_guard_enabled": poison_root_reads,
        "cloud_launch_authorized": False,
        "gcloud_invocation_authorized": False,
        "remote_write_performed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-job-sha256", required=True)
    parser.add_argument("--expected-job-id", required=True)
    parser.add_argument("--expected-stage-id", required=True)
    parser.add_argument(
        "--poison-root-reads",
        action="store_true",
        default=os.environ.get("OFC_DIAGNOSTIC_POISON_ROOT_READS") == "1",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = verify_package_precontent(
        source_path=args.source,
        manifest_path=args.manifest,
        job_path=args.job,
        expected_source_sha256=args.expected_source_sha256,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_job_sha256=args.expected_job_sha256,
        expected_job_id=args.expected_job_id,
        expected_stage_id=args.expected_stage_id,
        poison_root_reads=args.poison_root_reads,
    )
    print(
        json.dumps(
            result,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
