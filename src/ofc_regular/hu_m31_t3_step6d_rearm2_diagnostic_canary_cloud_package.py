"""Immutable worker package for the bounded rearm2 diagnostic canaries.

This module stops at local package construction, validation, and startup
pre-content smoke.  It deliberately has no launcher, claim, authorization,
object-store, or VM API.  The package uses accepted development artifacts and
the twenty roots frozen by the diagnostic contract; it does not copy or open
the production rearm2 source archive or locked roots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import types
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from . import hu_m31_t3_step6d_rearm2_diagnostic_canary_local as local
from . import hu_m31_t3_step6d_rearm2_diagnostic_canary_plan as plan
from . import hu_m31_t3_step6d_rearm2_diagnostic_canary_vm_adapter as vm_adapter
from . import run_hu_m31_t3_step6d_performance_v2 as runner


PACKAGE_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_cloud_worker_package_v1"
)
PACKAGE_STATUS = (
    "immutable_diagnostic_cloud_worker_package_ready_not_authorized"
)
READY_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_cloud_worker_package_ready_v1"
)
SMOKE_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_cloud_worker_startup_smoke_v1"
)

SOURCE_NAME = "hu_m31_t3_step6d_rearm2_diagnostic_worker_v1.zip"
MANIFEST_NAME = "manifest.json"
READY_NAME = "PACKAGE_READY.json"
STARTUP_NAME = (
    "startup_hu_m31_t3_step6d_rearm2_diagnostic_canary_v1.sh"
)
VERIFIER_NAME = (
    "verify_hu_m31_t3_step6d_rearm2_diagnostic_canary_v1.py"
)
ROOT_PREFIX = (
    "outputs/hu_joint_policy/m31_t3_step6d/candidate02_development/"
    "tail_reselection_v2/roots"
)
DEVELOPMENT_PLAN_RELATIVE = (
    "outputs/hu_joint_policy/m31_t3_step6d/candidate02_development/"
    "full100_plan_v1.json"
)
RUNTIME_REQUIREMENTS_RELATIVE = (
    "configs/hu_m31_t3_step6d_rearm2_diagnostic_runtime_requirements_v1.txt"
)

CANDIDATE_PACKAGE_PATH = (
    "native/candidate/release/libofc_hu_m3_engine.so"
)
REFERENCE_PACKAGE_PATH = (
    "native/reference/release/libofc_hu_m3_engine.so"
)
FEATURE_PACKAGE_PATH = "target/release/libofc_stage3_feature_encoder.so"
CANDIDATE_SOURCE_RELATIVE = (
    "outputs/hu_joint_policy/m31_t3_step6d/candidate02_development/"
    "candidate02_frozen/"
    "libofc_hu_m3_engine_candidate02_4050e04b22d7943d.so"
)
REFERENCE_SOURCE_RELATIVE = (
    "outputs/hu_joint_policy/m31_t3_step6a/linux-target/release/"
    "libofc_hu_m3_engine.so"
)
FEATURE_SOURCE_RELATIVE = (
    "outputs/hu_joint_policy/m31_t3_step6a/linux-feature-target/release/"
    "libofc_stage3_feature_encoder.so"
)

# Import-time and preseed-root runtime closure, independently audited against
# the accepted development package.  Training-only and lazy root-generation
# modules are intentionally excluded.
PYTHON_ALLOWLIST = (
    "src/ofc_regular/__init__.py",
    "src/ofc_regular/action_key.py",
    "src/ofc_regular/action_space.py",
    "src/ofc_regular/ai_profiles.py",
    "src/ofc_regular/cards.py",
    "src/ofc_regular/counter_rng.py",
    "src/ofc_regular/evaluator.py",
    "src/ofc_regular/hu_belief.py",
    "src/ofc_regular/hu_infoset.py",
    "src/ofc_regular/hu_late_street_teacher.py",
    "src/ofc_regular/hu_m31_t3_behavior_roots.py",
    "src/ofc_regular/hu_m31_t3_runtime.py",
    "src/ofc_regular/hu_m31_t3_step6d_contract.py",
    "src/ofc_regular/hu_m3_rust.py",
    "src/ofc_regular/hu_m3_t4_runtime.py",
    "src/ofc_regular/hu_turn0_candidate.py",
    "src/ofc_regular/hu_turn0_safe_selector.py",
    "src/ofc_regular/hu_turn1_safe_selector.py",
    "src/ofc_regular/hu_turn2_stage8_runtime.py",
    "src/ofc_regular/hu_turn3_gate_model.py",
    "src/ofc_regular/hu_turn3_joint_exact_teacher.py",
    "src/ofc_regular/hu_turn3_model.py",
    "src/ofc_regular/policy.py",
    "src/ofc_regular/rules.py",
    "src/ofc_regular/run_hu_m31_t3_step6d_performance.py",
    "src/ofc_regular/run_hu_m31_t3_step6d_performance_v2.py",
    "src/ofc_regular/state.py",
    "src/ofc_regular/teacher.py",
    "src/ofc_regular/train_hu_turn0_safe_override_selector.py",
    "src/ofc_regular/train_hu_turn1_safe_override_selector.py",
    "src/ofc_regular/train_torch_action_value.py",
    "src/ofc_regular/turn3_model.py",
    "src/ofc_regular/validate_hu_m31_t3_profile.py",
)
# Per-entry identity audited against the previously accepted development
# package.  A dirty source edit cannot silently become a new self-signed
# diagnostic package.
RUNTIME_CLOSURE_ACCEPTED = (
    ("src/ofc_regular/__init__.py", 1587, "3a7cc082996f24637a22bbfd6def94ea93177daa569ec21add6ac40bd2a4cfb9"),
    ("src/ofc_regular/action_key.py", 10381, "ff777a084fa1044d22f4f05ee6fa24a6a877e11f7cd2cd217e5f18e89e00ebf0"),
    ("src/ofc_regular/action_space.py", 2863, "bbdfbf6f531919b95a76b43a0e5eb2e5712019abc400a6dfee12f30402ad5fc0"),
    ("src/ofc_regular/ai_profiles.py", 35161, "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"),
    ("src/ofc_regular/cards.py", 1160, "a9a33617a395464df62443e49c44e20fa81a56150393de3fc130bb4de74dc510"),
    ("src/ofc_regular/counter_rng.py", 3633, "f713f881163046b5ba6d9ad3fbc4078c640799f8d58933f05eb7816e17252971"),
    ("src/ofc_regular/evaluator.py", 6130, "fc4bed0d7c5bb733edc5ef4a71e3f83e161e43b96f795cdbeb131432c63ba937"),
    ("src/ofc_regular/hu_belief.py", 18959, "9137531c6ec9e2c2a017070bd3407f69635117f7e9469fbfc556018c46af863d"),
    ("src/ofc_regular/hu_infoset.py", 33781, "f72624fff2e49564a95b462c94e74fff957c8aa19850d2a2e52b188c091cc490"),
    ("src/ofc_regular/hu_late_street_teacher.py", 26694, "a66bd8e2dda3a16bfc90f2f06cd0013ed406e704482599561c4e79b9ec68de81"),
    ("src/ofc_regular/hu_m31_t3_behavior_roots.py", 4188, "f17fc6b93a537ebbf53f18bdc6f388f7cddd9365ab2ff1006af57f274d981a55"),
    ("src/ofc_regular/hu_m31_t3_runtime.py", 41108, "d9230a82b8350dde3937f63988a90c96e3cb1b6fe6f0b83858f550897276998a"),
    ("src/ofc_regular/hu_m31_t3_step6d_contract.py", 10274, "85e290f9193cd2247b2d7619edc554601ef5c1f6e5ab96c55f198e2bc9a14a23"),
    ("src/ofc_regular/hu_m3_rust.py", 14492, "6de5bc5cf45d0088593b1c3d2eacbdd4fa2606a4c1063a128fc8c2a463514b78"),
    ("src/ofc_regular/hu_m3_t4_runtime.py", 26753, "f16dc52a3347a8cb47b083299c9b43cb180207b47a9b7ece819006c0dba116c9"),
    ("src/ofc_regular/hu_turn0_candidate.py", 1062, "b2eba591e0c3f66ce4e2937092cfe6049a5e5642500f80ec2892e5358475d5c2"),
    ("src/ofc_regular/hu_turn0_safe_selector.py", 1941, "1b8cec5dc3e992508b009c0aefe1e28126526bcf79bc12e275d9f66dc24d7e3a"),
    ("src/ofc_regular/hu_turn1_safe_selector.py", 2029, "d9fedc9e8b51d212f088ad0188e51b171f3453a93d65c1bbf0b13c1a44bacda2"),
    ("src/ofc_regular/hu_turn2_stage8_runtime.py", 18666, "6fca219d32241edfbd906e9c575d0d70dae83e7ff42dd20c9217415f56c09a88"),
    ("src/ofc_regular/hu_turn3_gate_model.py", 7386, "10a3a8277a5e65afe7e47be14744157901fb87537647e1fce439330670b8d902"),
    ("src/ofc_regular/hu_turn3_joint_exact_teacher.py", 41353, "c974744e52f310cb2ea9a87f4a5b10a12bff3076b0408ddb9a982310d2b9f4a1"),
    ("src/ofc_regular/hu_turn3_model.py", 41917, "ac6c246faf3ba2d806fa43afb6b871ef0e3a4e92808be5c48f867442d4064a57"),
    ("src/ofc_regular/policy.py", 59605, "6dec731cce97f914940c774214f6495dcb6feda60708b534f6b1316517e0f617"),
    ("src/ofc_regular/rules.py", 2613, "589ce9ebae0cd156a25058a214bfacee4eb0f986c42cc2ae8c829f92bcc843cd"),
    ("src/ofc_regular/run_hu_m31_t3_step6d_performance.py", 62156, "ac7e76d615894c3711faf7e29e7d8fff95159ab7d572e041cbd54440dea83133"),
    ("src/ofc_regular/run_hu_m31_t3_step6d_performance_v2.py", 99516, "02a4cbcc6773f88fa8fa936215a6e1602d9bf281ec7d6afc4f2b7d9d77834105"),
    ("src/ofc_regular/state.py", 1859, "3e3b1dd78726edb989043a2be7e996f6ef2fceb18e92bef8b95db3c74843d62b"),
    ("src/ofc_regular/teacher.py", 8455, "53a60bfca035542e09b9aec15cd17b068957a6a997292ff4b7c6111e16928887"),
    ("src/ofc_regular/train_hu_turn0_safe_override_selector.py", 22740, "57c0965785defc5fcb9d10355cd4424db92c06b7a4978d356fbc62993cc6a119"),
    ("src/ofc_regular/train_hu_turn1_safe_override_selector.py", 38346, "567d4c99acaa87d7cac3a91828aff01c9c2ea732657160f0c56b78d82e9c2589"),
    ("src/ofc_regular/train_torch_action_value.py", 11359, "655b42349f8001f706be3273b0424705e993ac2d7877b8654cb1856ea72a0075"),
    ("src/ofc_regular/turn3_model.py", 25134, "f977842868a95bc9ab4bfa5ee5a5c7999801400286aa2e23bfa6c12213fd933f"),
    ("src/ofc_regular/validate_hu_m31_t3_profile.py", 37921, "58999bcb3b24bfe46d1f239e335fccbc7612bd41be41ab37568004a41f2eaa14"),
    ("configs/hu_joint_policy_m31_t3_step6d_contract.json", 21794, "1924295b18070432cf3126159311102d9285dba37c498a3ea7666b0d5b777775"),
)
EXPECTED_RUNTIME_CLOSURE_SHA256 = (
    "9894c508e028792d36238eced2a3370ea78acab03a00a1b467c92932f4f51ad6"
)
CONFIG_ALLOWLIST = (
    "configs/hu_joint_policy_m31_t3_step6d_contract.json",
    plan.FROZEN_CONTRACT_RELATIVE,
    DEVELOPMENT_PLAN_RELATIVE,
    RUNTIME_REQUIREMENTS_RELATIVE,
)

EXPECTED_DEVELOPMENT_PLAN_SHA256 = plan.EXPECTED_DEVELOPMENT_PLAN_SHA256
EXPECTED_RUN_CONTRACT_DIGEST = (
    "92a87f1544a73104d5256db39b022094c8c7f7b11f77bd19dcf1ab1442581ebd"
)
EXPECTED_CANDIDATE_SHA256 = plan.EXPECTED_CANDIDATE_SHA256
EXPECTED_REFERENCE_SHA256 = plan.EXPECTED_REFERENCE_SHA256
EXPECTED_FEATURE_SHA256 = plan.EXPECTED_FEATURE_ENCODER_SHA256

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STARTUP = _REPO_ROOT / "scripts" / STARTUP_NAME
DEFAULT_VERIFIER = _REPO_ROOT / "scripts" / VERIFIER_NAME
DEFAULT_CANDIDATE = _REPO_ROOT / CANDIDATE_SOURCE_RELATIVE
DEFAULT_REFERENCE = _REPO_ROOT / REFERENCE_SOURCE_RELATIVE
DEFAULT_FEATURE = _REPO_ROOT / FEATURE_SOURCE_RELATIVE
_SHA256_CHARS = frozenset("0123456789abcdef")
_FORBIDDEN_KEY_PARTS = (
    "opponent_private_discard",
    "opponent_hidden",
    "hidden_truth",
    "realized_deck_tail",
)
_SAFE_FALSE_HIDDEN_GUARDS = frozenset({"opponent_private_discards_used"})
_FORBIDDEN_CAPABILITIES = (
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
_PACKAGE_KEYS = frozenset(
    {
        "schema",
        "status",
        "source_name",
        "source_sha256",
        "source_bytes",
        "startup_name",
        "startup_sha256",
        "verifier_name",
        "verifier_sha256",
        "diagnostic_contract_sha256",
        "development_plan_sha256",
        "run_contract",
        "run_contract_digest",
        "accepted_candidate",
        "accepted_reference",
        "feature_encoder",
        "python_allowlist",
        "config_allowlist",
        "runtime_closure_records",
        "runtime_closure_sha256",
        "source_entries",
        "source_entry_count",
        "root_member_paths",
        "development_root_count",
        "job_manifests",
        "logical_job_count",
        "stages",
        "allocation",
        "cloud_worker_payload_complete",
        "cloud_executable",
        "launch_ready",
        "host_prerequisites",
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
        "rearm2_production_package_reused",
        "rearm2_production_source_opened",
        "rearm2_locked_roots_used",
        "all20_launcher_reused",
    }
)
_READY_KEYS = frozenset(
    {
        "schema",
        "status",
        "package_manifest_sha256",
        "source_sha256",
        "startup_sha256",
        "verifier_sha256",
        "diagnostic_contract_sha256",
        "run_contract_digest",
        "logical_job_count",
        "development_root_count",
        "cloud_launch_authorized",
        "gcloud_invoked",
        "remote_write_performed",
        "current_profile_changed",
    }
)
_JOB_RECORD_KEYS = frozenset(
    {
        "job_id",
        "stage_id",
        "run_name",
        "source_role",
        "work_hand_indices",
        "path",
        "sha256",
        "bytes",
    }
)


def canonical_bytes(value: Any) -> bytes:
    return runner.canonical_bytes(value)


def canonical_sha256(value: Any) -> str:
    return runner.canonical_sha256(value)


def sha256_file(path: str | Path) -> str:
    return local.sha256_file(path)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_SHA256_CHARS)
    )


def _exact(value: Mapping[str, Any], keys: frozenset[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{label} keys changed")


def _reject_hidden(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).casefold()
            if any(part in key for part in _FORBIDDEN_KEY_PARTS):
                if key not in _SAFE_FALSE_HIDDEN_GUARDS or child is not False:
                    raise ValueError(f"forbidden hidden field at {path}.{raw_key}")
                continue
            _reject_hidden(child, f"{path}.{raw_key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_hidden(child, f"{path}[{index}]")


def _read_canonical(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} is missing or unsafe: {path}")
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} must be canonical JSON")
    return value


def _validate_elf(path: Path, expected_sha256: str, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} is missing or unsafe: {path}")
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} SHA-256 changed")
    with path.open("rb") as handle:
        header = handle.read(20)
    if (
        len(header) != 20
        or header[:4] != b"\x7fELF"
        or header[4] != 2
        or header[5] != 1
        or header[16:18] != b"\x03\x00"
        or header[18:20] != b"\x3e\x00"
    ):
        raise ValueError(f"{label} must be a Linux x86_64 shared library")
    return path.resolve()


def _write_once(path: Path, payload: bytes | Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable diagnostic package artifact exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = payload if isinstance(payload, bytes) else canonical_bytes(payload)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _zip(entries: Mapping[str, bytes], path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("immutable diagnostic source already exists")
    with zipfile.ZipFile(
        path,
        "x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in sorted(entries):
            pure = PurePosixPath(name)
            if (
                not name
                or "\\" in name
                or name.startswith("/")
                or pure.is_absolute()
                or any(part in ("", ".", "..") for part in pure.parts)
            ):
                raise ValueError("unsafe diagnostic source path")
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, entries[name])


def _read_source_file(root: Path, relative: str, label: str) -> bytes:
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} is missing or unsafe: {path}")
    return path.read_bytes()


def _accepted_runtime_closure_records() -> list[dict[str, Any]]:
    records = [
        {"path": path, "bytes": size, "sha256": digest}
        for path, size, digest in RUNTIME_CLOSURE_ACCEPTED
    ]
    if (
        [record["path"] for record in records]
        != [
            *PYTHON_ALLOWLIST,
            "configs/hu_joint_policy_m31_t3_step6d_contract.json",
        ]
        or canonical_sha256(records) != EXPECTED_RUNTIME_CLOSURE_SHA256
    ):
        raise AssertionError("accepted diagnostic runtime closure constant changed")
    return records


def _validate_runtime_closure(repository_root: Path) -> list[dict[str, Any]]:
    expected = _accepted_runtime_closure_records()
    actual: list[dict[str, Any]] = []
    for record in expected:
        raw = _read_source_file(
            repository_root,
            record["path"],
            "accepted diagnostic runtime closure member",
        )
        actual.append(
            {
                "path": record["path"],
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    if actual != expected or canonical_sha256(actual) != EXPECTED_RUNTIME_CLOSURE_SHA256:
        raise ValueError("accepted diagnostic runtime closure changed")
    return actual


def _stages(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "stage_id": stage["stage_id"],
            "run_name": stage["run_name"],
            "selected_job_ids": list(stage["selected_job_ids"]),
            "source_roles": list(stage["source_roles"]),
            "work_hand_indices": list(stage["hand_indices"]),
            "vm_count": stage["vm_count"],
            "diagnostic_only": True,
            "cloud_launch_authorized": False,
            "performance_lock_evidence": False,
            "quality_evidence": False,
            "training_eligible": False,
            "promotion_evidence": False,
        }
        for stage in contract["stages"]
    ]


def _host_prerequisites() -> dict[str, Any]:
    return {
        "architecture": "x86_64",
        "glibc_minimum": "2.34",
        "shell": "bash",
        "python_minimum": "3.10",
        "debian_packages": [
            "bash",
            "ca-certificates",
            "libgcc-s1",
            "libgomp1",
            "python3",
            "python3-venv",
        ],
        "dependency_install": (
            "versioned_numpy_requirement_requires_network_or_"
            "separately_vendored_wheel"
        ),
        "gce_metadata_bootstrap_included": False,
        "object_download_transport_included": False,
        "remote_upload_transport_included": False,
    }


def _job_specs(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for stage in contract["stages"]:
        for job_id, source_role in zip(
            stage["selected_job_ids"], stage["source_roles"], strict=True
        ):
            values.append(
                {
                    "job_id": job_id,
                    "stage_id": stage["stage_id"],
                    "run_name": stage["run_name"],
                    "source_role": source_role,
                    "work_hand_indices": list(stage["hand_indices"]),
                }
            )
    if (
        [row["job_id"] for row in values]
        != [
            "candidate-shard-00",
            "candidate-shard-01",
            "reference-shard-01",
        ]
        or len(values) != 3
        or values[1]["work_hand_indices"] != values[2]["work_hand_indices"]
        or set(values[0]["work_hand_indices"])
        & set(values[1]["work_hand_indices"])
    ):
        raise ValueError("diagnostic exact three-job stage separation changed")
    return values


def _development_plan(root: Path) -> dict[str, Any]:
    path = root / DEVELOPMENT_PLAN_RELATIVE
    if sha256_file(path) != EXPECTED_DEVELOPMENT_PLAN_SHA256:
        raise ValueError("accepted development plan SHA-256 changed")
    value = _read_canonical(path, "accepted development plan")
    run_contract = runner.validate_run_contract(value["run_contract"])
    if (
        value.get("run_contract_digest") != EXPECTED_RUN_CONTRACT_DIGEST
        or canonical_sha256(run_contract) != EXPECTED_RUN_CONTRACT_DIGEST
        or run_contract.get("candidate_variant") != runner.CANDIDATE02_VARIANT
        or run_contract.get("schedule") != runner.CANDIDATE02_SCHEDULE
        or run_contract.get("candidate_library_sha256")
        != EXPECTED_CANDIDATE_SHA256
        or run_contract.get("reference_library_sha256")
        != EXPECTED_REFERENCE_SHA256
        or any(
            run_contract.get(key) is not False
            for key in (
                "current_profile_changed",
                "promotion_evidence",
                "quality_evidence",
                "training_eligible",
            )
        )
    ):
        raise ValueError("diagnostic development run contract changed")
    return value


def _source_entries(
    *,
    repository_root: Path,
    contract: Mapping[str, Any],
    candidate: Path,
    reference: Path,
    feature: Path,
) -> tuple[dict[str, bytes], dict[str, dict[str, Any]], list[str]]:
    raw: dict[str, bytes] = {}
    kinds: dict[str, str] = {}
    for relative in PYTHON_ALLOWLIST:
        raw[relative] = _read_source_file(
            repository_root, relative, "allowlisted Python source"
        )
        kinds[relative] = "python"
    for relative in CONFIG_ALLOWLIST:
        raw[relative] = _read_source_file(
            repository_root, relative, "allowlisted diagnostic config"
        )
        kinds[relative] = (
            "requirements"
            if relative == RUNTIME_REQUIREMENTS_RELATIVE
            else "config"
        )
    binaries = {
        CANDIDATE_PACKAGE_PATH: (candidate, "native_candidate"),
        REFERENCE_PACKAGE_PATH: (reference, "native_reference"),
        FEATURE_PACKAGE_PATH: (feature, "native_feature_encoder"),
    }
    for relative, (path, kind) in binaries.items():
        raw[relative] = path.read_bytes()
        kinds[relative] = kind

    root_paths: list[str] = []
    seen: set[int] = set()
    for stage in contract["stages"]:
        for record in stage["root_records"]:
            index = int(record["hand_index"])
            if index in seen:
                continue
            seen.add(index)
            relative = f"{ROOT_PREFIX}/hand_{index:03d}.json"
            content = _read_source_file(
                repository_root, relative, "diagnostic development root"
            )
            if (
                hashlib.sha256(content).hexdigest() != record["sha256"]
                or len(content) != record["size_bytes"]
            ):
                raise ValueError(f"diagnostic development root changed: {index}")
            value = json.loads(content.decode("utf-8"))
            if (
                not isinstance(value, dict)
                or content != canonical_bytes(value)
                or value.get("hand_index") != index
                or value.get("schema") != runner.CANDIDATE02_ROOT_SCHEMA
                or value.get("training_eligible") is not False
                or value.get("current_profile_resolved") is not False
                or value.get("opponent_private_discards_used") is not False
            ):
                raise ValueError(f"diagnostic root safety boundary changed: {index}")
            _reject_hidden(value)
            raw[relative] = content
            kinds[relative] = "root"
            root_paths.append(relative)
    root_paths.sort()
    if len(root_paths) != 20 or len(seen) != 20:
        raise ValueError("diagnostic package requires exactly twenty roots")
    entries = {
        relative: {
            "bytes": len(content),
            "kind": kinds[relative],
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        for relative, content in sorted(raw.items())
    }
    return dict(sorted(raw.items())), entries, root_paths


def _job_records(
    *,
    stage_dir: Path,
    contract: Mapping[str, Any],
    run_contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for spec in _job_specs(contract):
        value = runner.build_shard_manifest(
            run_contract=run_contract,
            source_role=spec["source_role"],
            work_hand_indices=spec["work_hand_indices"],
        )
        runner.validate_shard_manifest(value)
        relative = f"jobs/{spec['job_id']}.json"
        path = stage_dir / relative
        _write_once(path, value)
        records.append(
            {
                **spec,
                "path": relative,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return records


def build_package(
    *,
    output_dir: str | Path,
    repository_root: str | Path = _REPO_ROOT,
    contract_path: str | Path = plan.DEFAULT_FROZEN_CONTRACT,
    candidate_library: str | Path = DEFAULT_CANDIDATE,
    reference_library: str | Path = DEFAULT_REFERENCE,
    feature_encoder: str | Path = DEFAULT_FEATURE,
    startup_script: str | Path = DEFAULT_STARTUP,
    precontent_verifier: str | Path = DEFAULT_VERIFIER,
) -> dict[str, Any]:
    """Build one immutable local worker package; never contact cloud services."""

    root = Path(repository_root).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("diagnostic cloud-worker package is immutable")
    frozen_contract_path = Path(contract_path).resolve()
    contract = plan.validate_frozen_contract(frozen_contract_path)
    if frozen_contract_path != (root / plan.FROZEN_CONTRACT_RELATIVE).resolve():
        raise ValueError("diagnostic package requires the fixed contract path")
    if sha256_file(plan.DEFAULT_CURRENT_PROFILE) != plan.EXPECTED_CURRENT_PROFILE_FILE_SHA256:
        raise ValueError("current profile file changed before diagnostic packaging")
    development = _development_plan(root)
    runtime_closure_records = _validate_runtime_closure(root)
    candidate = _validate_elf(
        Path(candidate_library).resolve(),
        EXPECTED_CANDIDATE_SHA256,
        "accepted candidate",
    )
    reference = _validate_elf(
        Path(reference_library).resolve(),
        EXPECTED_REFERENCE_SHA256,
        "accepted reference",
    )
    feature = _validate_elf(
        Path(feature_encoder).resolve(),
        EXPECTED_FEATURE_SHA256,
        "accepted feature encoder",
    )
    if len({candidate, reference, feature}) != 3:
        raise ValueError("accepted diagnostic native artifacts must be distinct")
    startup = Path(startup_script).resolve()
    verifier = Path(precontent_verifier).resolve()
    for path, label in (
        (startup, "diagnostic startup"),
        (verifier, "diagnostic pre-content verifier"),
    ):
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"{label} is missing or unsafe: {path}")

    staging = destination.with_name(f".{destination.name}.{os.getpid()}.staging")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError("stale diagnostic package staging exists")
    staging.mkdir(parents=True)
    try:
        content, entries, root_paths = _source_entries(
            repository_root=root,
            contract=contract,
            candidate=candidate,
            reference=reference,
            feature=feature,
        )
        source = staging / SOURCE_NAME
        _zip(content, source)
        shutil.copy2(startup, staging / STARTUP_NAME)
        shutil.copy2(verifier, staging / VERIFIER_NAME)
        jobs = _job_records(
            stage_dir=staging,
            contract=contract,
            run_contract=development["run_contract"],
        )
        manifest = {
            "schema": PACKAGE_SCHEMA,
            "status": PACKAGE_STATUS,
            "source_name": SOURCE_NAME,
            "source_sha256": sha256_file(source),
            "source_bytes": source.stat().st_size,
            "startup_name": STARTUP_NAME,
            "startup_sha256": sha256_file(staging / STARTUP_NAME),
            "verifier_name": VERIFIER_NAME,
            "verifier_sha256": sha256_file(staging / VERIFIER_NAME),
            "diagnostic_contract_sha256": canonical_sha256(contract),
            "development_plan_sha256": EXPECTED_DEVELOPMENT_PLAN_SHA256,
            "run_contract": development["run_contract"],
            "run_contract_digest": EXPECTED_RUN_CONTRACT_DIGEST,
            "accepted_candidate": {
                "package_path": CANDIDATE_PACKAGE_PATH,
                "sha256": EXPECTED_CANDIDATE_SHA256,
            },
            "accepted_reference": {
                "package_path": REFERENCE_PACKAGE_PATH,
                "sha256": EXPECTED_REFERENCE_SHA256,
            },
            "feature_encoder": {
                "package_path": FEATURE_PACKAGE_PATH,
                "sha256": EXPECTED_FEATURE_SHA256,
            },
            "python_allowlist": list(PYTHON_ALLOWLIST),
            "config_allowlist": list(CONFIG_ALLOWLIST),
            "runtime_closure_records": runtime_closure_records,
            "runtime_closure_sha256": EXPECTED_RUNTIME_CLOSURE_SHA256,
            "source_entries": entries,
            "source_entry_count": len(entries),
            "root_member_paths": root_paths,
            "development_root_count": len(root_paths),
            "job_manifests": jobs,
            "logical_job_count": len(jobs),
            "stages": _stages(contract),
            "allocation": {
                "machine_type": vm_adapter.MACHINE_TYPE,
                "process_count": 1,
                "rayon_threads_per_process": 16,
                "omp_threads": 1,
                "m3_batch_threads": 1,
            },
            "cloud_worker_payload_complete": True,
            "cloud_executable": False,
            "launch_ready": False,
            "host_prerequisites": _host_prerequisites(),
            "cloud_launch_authorized": False,
            "gcloud_invocation_authorized": False,
            "claim_write_authorized": False,
            "authorization_write_authorized": False,
            "object_write_authorized": False,
            "vm_create_authorized": False,
            "performance_lock_evidence": False,
            "quality_evidence": False,
            "training_eligible": False,
            "promotion_evidence": False,
            "current_profile_changed": False,
            "runtime_policy_activated": False,
            "rearm2_production_package_reused": False,
            "rearm2_production_source_opened": False,
            "rearm2_locked_roots_used": False,
            "all20_launcher_reused": False,
        }
        _write_once(staging / MANIFEST_NAME, manifest)
        ready = {
            "schema": READY_SCHEMA,
            "status": "immutable_local_package_complete_cloud_not_authorized",
            "package_manifest_sha256": sha256_file(staging / MANIFEST_NAME),
            "source_sha256": manifest["source_sha256"],
            "startup_sha256": manifest["startup_sha256"],
            "verifier_sha256": manifest["verifier_sha256"],
            "diagnostic_contract_sha256": manifest[
                "diagnostic_contract_sha256"
            ],
            "run_contract_digest": EXPECTED_RUN_CONTRACT_DIGEST,
            "logical_job_count": 3,
            "development_root_count": 20,
            "cloud_launch_authorized": False,
            "gcloud_invoked": False,
            "remote_write_performed": False,
            "current_profile_changed": False,
        }
        _write_once(staging / READY_NAME, ready)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_package(destination, contract_path=frozen_contract_path)


def _validate_archive(
    *,
    source: Path,
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    entries = manifest["source_entries"]
    expected_names = sorted(entries)
    with zipfile.ZipFile(source) as archive:
        infos = archive.infolist()
        if [info.filename for info in infos] != expected_names:
            raise ValueError("diagnostic source file set or order changed")
        if len({info.filename for info in infos}) != len(infos):
            raise ValueError("diagnostic source has duplicate members")
        roots: dict[int, dict[str, Any]] = {}
        for info in infos:
            mode = (info.external_attr >> 16) & 0xFFFF
            if (
                info.is_dir()
                or stat.S_IFMT(mode) != stat.S_IFREG
                or mode & 0o777 != 0o644
            ):
                raise ValueError("diagnostic source contains a symlink or unsafe member")
            record = entries.get(info.filename)
            if (
                not isinstance(record, dict)
                or set(record) != {"bytes", "kind", "sha256"}
                or record["bytes"] != info.file_size
                or not _is_sha256(record["sha256"])
            ):
                raise ValueError("diagnostic source entry metadata changed")
            raw = archive.read(info.filename)
            if (
                len(raw) != record["bytes"]
                or hashlib.sha256(raw).hexdigest() != record["sha256"]
            ):
                raise ValueError("diagnostic source entry content changed")
            if record["kind"] == "root":
                value = json.loads(raw.decode("utf-8"))
                if not isinstance(value, dict) or raw != canonical_bytes(value):
                    raise ValueError("diagnostic packaged root is not canonical")
                _reject_hidden(value)
                index = value.get("hand_index")
                if (
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or info.filename
                    != f"{ROOT_PREFIX}/hand_{index:03d}.json"
                    or value.get("schema") != runner.CANDIDATE02_ROOT_SCHEMA
                    or value.get("training_eligible") is not False
                    or value.get("current_profile_resolved") is not False
                    or value.get("opponent_private_discards_used") is not False
                ):
                    raise ValueError("diagnostic packaged root safety changed")
                roots[index] = value
        expected_indices = {
            int(record["hand_index"])
            for stage in contract["stages"]
            for record in stage["root_records"]
        }
        if set(roots) != expected_indices or len(roots) != 20:
            raise ValueError("diagnostic packaged root set changed")
        for relative, expected in (
            (CANDIDATE_PACKAGE_PATH, EXPECTED_CANDIDATE_SHA256),
            (REFERENCE_PACKAGE_PATH, EXPECTED_REFERENCE_SHA256),
            (FEATURE_PACKAGE_PATH, EXPECTED_FEATURE_SHA256),
        ):
            raw = archive.read(relative)
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError("accepted native artifact changed in archive")
            header = raw[:20]
            if (
                len(header) != 20
                or header[:4] != b"\x7fELF"
                or header[4] != 2
                or header[5] != 1
                or header[16:18] != b"\x03\x00"
                or header[18:20] != b"\x3e\x00"
            ):
                raise ValueError("packaged native artifact is not Linux x86_64 ELF")


def validate_package(
    package_dir: str | Path,
    *,
    contract_path: str | Path = plan.DEFAULT_FROZEN_CONTRACT,
) -> dict[str, Any]:
    target = Path(package_dir).resolve()
    if not target.is_dir() or target.is_symlink():
        raise FileNotFoundError("diagnostic cloud-worker package is missing or unsafe")
    contract = plan.validate_frozen_contract(contract_path)
    expected_files = {
        MANIFEST_NAME,
        READY_NAME,
        SOURCE_NAME,
        STARTUP_NAME,
        VERIFIER_NAME,
        "jobs/candidate-shard-00.json",
        "jobs/candidate-shard-01.json",
        "jobs/reference-shard-01.json",
    }
    actual_files: set[str] = set()
    for path in target.rglob("*"):
        if path.is_symlink():
            raise ValueError("diagnostic package contains a symlink")
        if path.is_file():
            actual_files.add(path.relative_to(target).as_posix())
    if actual_files != expected_files:
        raise ValueError("diagnostic package file set changed")
    manifest = _read_canonical(target / MANIFEST_NAME, "diagnostic package manifest")
    ready = _read_canonical(target / READY_NAME, "diagnostic package ready")
    _exact(manifest, _PACKAGE_KEYS, "diagnostic package manifest")
    _exact(ready, _READY_KEYS, "diagnostic package ready")
    _reject_hidden(manifest)
    source = target / SOURCE_NAME
    startup = target / STARTUP_NAME
    verifier = target / VERIFIER_NAME
    if any(path.is_symlink() or not path.is_file() for path in (source, startup, verifier)):
        raise ValueError("diagnostic package executable source is unsafe")
    entries = manifest.get("source_entries")
    if not isinstance(entries, dict):
        raise ValueError("diagnostic source entry manifest is missing")
    run_contract = runner.validate_run_contract(manifest["run_contract"])
    jobs = manifest.get("job_manifests")
    specs = _job_specs(contract)
    if not isinstance(jobs, list) or len(jobs) != 3:
        raise ValueError("diagnostic package requires exactly three jobs")
    for record, spec in zip(jobs, specs, strict=True):
        if not isinstance(record, dict):
            raise ValueError("diagnostic job record is not an object")
        _exact(record, _JOB_RECORD_KEYS, "diagnostic job record")
        path = target / record["path"]
        value = _read_canonical(path, f"diagnostic job {record.get('job_id')}")
        expected_manifest = runner.build_shard_manifest(
            run_contract=run_contract,
            source_role=spec["source_role"],
            work_hand_indices=spec["work_hand_indices"],
        )
        if (
            {key: record[key] for key in spec} != spec
            or record["path"] != f"jobs/{spec['job_id']}.json"
            or path.is_symlink()
            or record["sha256"] != sha256_file(path)
            or record["bytes"] != path.stat().st_size
            or value != expected_manifest
            or runner.validate_shard_manifest(value) != value
        ):
            raise ValueError("diagnostic runner-compatible job changed")
    expected_false = (
        *_FORBIDDEN_CAPABILITIES,
        "rearm2_production_package_reused",
        "rearm2_production_source_opened",
        "rearm2_locked_roots_used",
        "all20_launcher_reused",
    )
    expected_roots = sorted(
        f"{ROOT_PREFIX}/hand_{index:03d}.json"
        for index in {
            int(record["hand_index"])
            for stage in contract["stages"]
            for record in stage["root_records"]
        }
    )
    if (
        manifest.get("schema") != PACKAGE_SCHEMA
        or manifest.get("status") != PACKAGE_STATUS
        or manifest.get("source_name") != SOURCE_NAME
        or manifest.get("source_sha256") != sha256_file(source)
        or manifest.get("source_bytes") != source.stat().st_size
        or manifest.get("startup_name") != STARTUP_NAME
        or manifest.get("startup_sha256") != sha256_file(startup)
        or manifest.get("verifier_name") != VERIFIER_NAME
        or manifest.get("verifier_sha256") != sha256_file(verifier)
        or manifest.get("diagnostic_contract_sha256") != canonical_sha256(contract)
        or manifest.get("development_plan_sha256")
        != EXPECTED_DEVELOPMENT_PLAN_SHA256
        or canonical_sha256(run_contract) != EXPECTED_RUN_CONTRACT_DIGEST
        or manifest.get("run_contract_digest") != EXPECTED_RUN_CONTRACT_DIGEST
        or manifest.get("accepted_candidate")
        != {
            "package_path": CANDIDATE_PACKAGE_PATH,
            "sha256": EXPECTED_CANDIDATE_SHA256,
        }
        or manifest.get("accepted_reference")
        != {
            "package_path": REFERENCE_PACKAGE_PATH,
            "sha256": EXPECTED_REFERENCE_SHA256,
        }
        or manifest.get("feature_encoder")
        != {
            "package_path": FEATURE_PACKAGE_PATH,
            "sha256": EXPECTED_FEATURE_SHA256,
        }
        or manifest.get("python_allowlist") != list(PYTHON_ALLOWLIST)
        or manifest.get("config_allowlist") != list(CONFIG_ALLOWLIST)
        or manifest.get("runtime_closure_records")
        != _accepted_runtime_closure_records()
        or manifest.get("runtime_closure_sha256")
        != EXPECTED_RUNTIME_CLOSURE_SHA256
        or set(entries) != set(PYTHON_ALLOWLIST)
        | set(CONFIG_ALLOWLIST)
        | {
            CANDIDATE_PACKAGE_PATH,
            REFERENCE_PACKAGE_PATH,
            FEATURE_PACKAGE_PATH,
            *expected_roots,
        }
        or manifest.get("source_entry_count") != len(entries)
        or manifest.get("root_member_paths") != expected_roots
        or manifest.get("development_root_count") != 20
        or manifest.get("logical_job_count") != 3
        or manifest.get("stages") != _stages(contract)
        or manifest.get("allocation")
        != {
            "machine_type": vm_adapter.MACHINE_TYPE,
            "process_count": 1,
            "rayon_threads_per_process": 16,
            "omp_threads": 1,
            "m3_batch_threads": 1,
        }
        or manifest.get("cloud_worker_payload_complete") is not True
        or manifest.get("cloud_executable") is not False
        or manifest.get("launch_ready") is not False
        or manifest.get("host_prerequisites") != _host_prerequisites()
        or any(manifest.get(key) is not False for key in expected_false)
    ):
        raise ValueError("diagnostic package boundary changed")
    _validate_archive(source=source, manifest=manifest, contract=contract)
    expected_ready = {
        "schema": READY_SCHEMA,
        "status": "immutable_local_package_complete_cloud_not_authorized",
        "package_manifest_sha256": sha256_file(target / MANIFEST_NAME),
        "source_sha256": manifest["source_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "verifier_sha256": manifest["verifier_sha256"],
        "diagnostic_contract_sha256": manifest["diagnostic_contract_sha256"],
        "run_contract_digest": EXPECTED_RUN_CONTRACT_DIGEST,
        "logical_job_count": 3,
        "development_root_count": 20,
        "cloud_launch_authorized": False,
        "gcloud_invoked": False,
        "remote_write_performed": False,
        "current_profile_changed": False,
    }
    if ready != expected_ready:
        raise ValueError("diagnostic package-ready boundary changed")
    return manifest


def _load_verifier(path: Path) -> Any:
    module = types.ModuleType("_ofc_diag_precontent_verifier_v1")
    module.__file__ = str(path)
    code = compile(path.read_bytes(), str(path), "exec")
    exec(code, module.__dict__)
    return module


def _bash_prefix() -> list[str]:
    if os.name == "nt":
        executable = shutil.which("wsl.exe")
        if executable is None:
            raise RuntimeError("WSL bash is required for diagnostic startup smoke")
        return [executable, "-e", "bash"]
    executable = shutil.which("bash")
    if executable is None:
        raise RuntimeError("bash is required for diagnostic startup smoke")
    return [executable]


def _bash_path(path: Path) -> str:
    if os.name != "nt":
        return str(path)
    completed = subprocess.run(
        [*_bash_prefix()[:-1], "wslpath", "-a", str(path)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    value = completed.stdout.strip()
    if not value.startswith("/"):
        raise ValueError("could not translate diagnostic smoke path for WSL")
    return value


def startup_smoke(
    package_dir: str | Path,
    *,
    contract_path: str | Path = plan.DEFAULT_FROZEN_CONTRACT,
) -> dict[str, Any]:
    """Run local syntax and poisoned-root startup checks only."""

    target = Path(package_dir).resolve()
    manifest = validate_package(target, contract_path=contract_path)
    startup = target / STARTUP_NAME
    verifier = target / VERIFIER_NAME
    source = target / SOURCE_NAME
    manifest_path = target / MANIFEST_NAME
    subprocess.run(
        [*_bash_prefix(), "-n", STARTUP_NAME],
        check=True,
        cwd=target,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    verifier_module = _load_verifier(verifier)
    reports: list[dict[str, Any]] = []
    shell_reports: list[dict[str, Any]] = []
    seeded_root_count = 0
    for record in manifest["job_manifests"]:
        report = verifier_module.verify_package_precontent(
            source_path=source,
            manifest_path=manifest_path,
            job_path=target / record["path"],
            expected_source_sha256=manifest["source_sha256"],
            expected_manifest_sha256=sha256_file(manifest_path),
            expected_job_sha256=record["sha256"],
            expected_job_id=record["job_id"],
            expected_stage_id=record["stage_id"],
            poison_root_reads=True,
        )
        if (
            report.get("root_members_opened") != 0
            or report.get("poison_root_guard_enabled") is not True
            or report.get("accepted_elf_count") != 3
            or report.get("remote_write_performed") is not False
        ):
            raise ValueError("diagnostic poisoned-root pre-content smoke failed")
        reports.append(report)
        completed = subprocess.run(
            [
                *_bash_prefix(),
                "-c",
                (
                    'export OFC_DIAGNOSTIC_EXPECTED_SOURCE_SHA256="$1" '
                    'OFC_DIAGNOSTIC_EXPECTED_MANIFEST_SHA256="$2" '
                    'OFC_DIAGNOSTIC_EXPECTED_JOB_SHA256="$3" '
                    'OFC_DIAGNOSTIC_EXPECTED_JOB_ID="$4" '
                    'OFC_DIAGNOSTIC_EXPECTED_STAGE_ID="$5" '
                    "OFC_DIAGNOSTIC_POISON_ROOT_READS=1 "
                    "OFC_DIAGNOSTIC_PRECONTENT_ONLY=1; "
                    'exec bash "$6" "$7" "$8" "$9" "${10}" "${11}"'
                ),
                "ofc-r2diag-precontent",
                manifest["source_sha256"],
                sha256_file(manifest_path),
                record["sha256"],
                record["job_id"],
                record["stage_id"],
                STARTUP_NAME,
                VERIFIER_NAME,
                SOURCE_NAME,
                MANIFEST_NAME,
                record["path"],
                "unused-precontent-output",
            ],
            check=True,
            cwd=target,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        shell_report = json.loads(lines[-1]) if lines else None
        if (
            not isinstance(shell_report, dict)
            or shell_report.get("job_id") != record["job_id"]
            or shell_report.get("root_members_opened") != 0
            or shell_report.get("poison_root_guard_enabled") is not True
        ):
            raise ValueError("diagnostic startup shell pre-content smoke failed")
        shell_reports.append(shell_report)
        with tempfile.TemporaryDirectory(
            prefix=f"ofc-r2diag-rootseed-{record['job_id']}-"
        ) as seed_temporary:
            seed_output = Path(seed_temporary) / "output"
            seeded = subprocess.run(
                [
                    *_bash_prefix(),
                    "-c",
                    (
                        'export OFC_DIAGNOSTIC_EXPECTED_SOURCE_SHA256="$1" '
                        'OFC_DIAGNOSTIC_EXPECTED_MANIFEST_SHA256="$2" '
                        'OFC_DIAGNOSTIC_EXPECTED_JOB_SHA256="$3" '
                        'OFC_DIAGNOSTIC_EXPECTED_JOB_ID="$4" '
                        'OFC_DIAGNOSTIC_EXPECTED_STAGE_ID="$5" '
                        "OFC_DIAGNOSTIC_POISON_ROOT_READS=1 "
                        "OFC_DIAGNOSTIC_OFFLINE_WORKER_AUTHORIZED=1 "
                        "OFC_DIAGNOSTIC_ROOT_SEED_ONLY=1; "
                        'exec bash "$6" "$7" "$8" "$9" "${10}" "${11}"'
                    ),
                    "ofc-r2diag-rootseed",
                    manifest["source_sha256"],
                    sha256_file(manifest_path),
                    record["sha256"],
                    record["job_id"],
                    record["stage_id"],
                    STARTUP_NAME,
                    VERIFIER_NAME,
                    SOURCE_NAME,
                    MANIFEST_NAME,
                    record["path"],
                    _bash_path(seed_output),
                ],
                check=True,
                cwd=target,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            if seeded.returncode != 0:
                raise ValueError("diagnostic root-seed startup smoke failed")
            actual = sorted((seed_output / "roots").glob("hand_*.json"))
            if len(actual) != 10:
                raise ValueError("diagnostic root-seed smoke did not write ten roots")
            with zipfile.ZipFile(source) as archive:
                for index, path in zip(
                    record["work_hand_indices"], actual, strict=True
                ):
                    relative = f"{ROOT_PREFIX}/hand_{index:03d}.json"
                    if (
                        path.name != f"hand_{index:03d}.json"
                        or path.is_symlink()
                        or path.read_bytes() != archive.read(relative)
                    ):
                        raise ValueError("diagnostic seeded root content changed")
            seeded_root_count += len(actual)

    with tempfile.TemporaryDirectory(prefix="ofc-r2diag-pycompile-") as temporary:
        extracted = Path(temporary)
        temporary_verifier = extracted / VERIFIER_NAME
        temporary_verifier.write_bytes(verifier.read_bytes())
        with zipfile.ZipFile(source) as archive:
            for relative in PYTHON_ALLOWLIST:
                destination = extracted / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(relative))
        compile_paths = [str(temporary_verifier)] + [
            str(extracted / relative) for relative in PYTHON_ALLOWLIST
        ]
        subprocess.run(
            [sys.executable, "-m", "py_compile", *compile_paths],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        isolated_environment = dict(os.environ)
        isolated_environment["PYTHONDONTWRITEBYTECODE"] = "1"
        isolated_environment.pop("PYTHONPATH", None)
        isolated = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import pathlib,sys;"
                    f"sys.path.insert(0,{json.dumps(str(extracted / 'src'))});"
                    "import ofc_regular.run_hu_m31_t3_step6d_performance_v2 as m;"
                    f"assert pathlib.Path(m.__file__).resolve().is_relative_to("
                    f"pathlib.Path({json.dumps(str(extracted))}).resolve());"
                    "assert m.CANDIDATE02_VARIANT == "
                    "'candidate02_compact_scorer';"
                    "print('isolated-runtime-closure-import-pass')"
                ),
            ],
            check=False,
            cwd=extracted,
            env=isolated_environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if (
            isolated.returncode != 0
            or isolated.stdout.strip() != "isolated-runtime-closure-import-pass"
        ):
            raise ValueError(
                "diagnostic isolated runtime closure import failed: "
                + isolated.stderr.strip()
            )
    return {
        "schema": SMOKE_SCHEMA,
        "status": "pass_local_precontent_only_cloud_not_authorized",
        "package_manifest_sha256": sha256_file(manifest_path),
        "source_sha256": manifest["source_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "verifier_sha256": manifest["verifier_sha256"],
        "job_ids": [record["job_id"] for record in manifest["job_manifests"]],
        "job_count": len(reports),
        "startup_entrypoint_job_count": len(shell_reports),
        "root_seed_job_count": len(shell_reports),
        "seeded_root_file_count": seeded_root_count,
        "development_root_count": 20,
        "root_members_opened_by_precontent_verifier": 0,
        "poison_root_guard_passed": True,
        "accepted_elf_count": 3,
        "bash_syntax_valid": True,
        "python_allowlist_compiled": len(PYTHON_ALLOWLIST),
        "verifier_compiled": True,
        "isolated_runtime_closure_imported": True,
        "cloud_launch_authorized": False,
        "gcloud_invoked": False,
        "claim_or_authorization_written": False,
        "remote_write_performed": False,
        "vm_started": False,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    package = commands.add_parser("package")
    package.add_argument("--output-dir", type=Path, required=True)
    package.add_argument("--repository-root", type=Path, default=_REPO_ROOT)
    package.add_argument("--contract", type=Path, default=plan.DEFAULT_FROZEN_CONTRACT)
    package.add_argument("--candidate-library", type=Path, default=DEFAULT_CANDIDATE)
    package.add_argument("--reference-library", type=Path, default=DEFAULT_REFERENCE)
    package.add_argument("--feature-encoder", type=Path, default=DEFAULT_FEATURE)
    validate = commands.add_parser("validate")
    validate.add_argument("--package-dir", type=Path, required=True)
    validate.add_argument("--contract", type=Path, default=plan.DEFAULT_FROZEN_CONTRACT)
    smoke = commands.add_parser("startup-smoke")
    smoke.add_argument("--package-dir", type=Path, required=True)
    smoke.add_argument("--contract", type=Path, default=plan.DEFAULT_FROZEN_CONTRACT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "package":
        value = build_package(
            output_dir=args.output_dir,
            repository_root=args.repository_root,
            contract_path=args.contract,
            candidate_library=args.candidate_library,
            reference_library=args.reference_library,
            feature_encoder=args.feature_encoder,
        )
    elif args.command == "validate":
        value = validate_package(args.package_dir, contract_path=args.contract)
    else:
        value = startup_smoke(args.package_dir, contract_path=args.contract)
    print(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CONFIG_ALLOWLIST",
    "PACKAGE_SCHEMA",
    "PYTHON_ALLOWLIST",
    "READY_SCHEMA",
    "SMOKE_SCHEMA",
    "build_package",
    "main",
    "startup_smoke",
    "validate_package",
]
