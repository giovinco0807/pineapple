from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
import zlib
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_candidate02_performance_lock_rearm2_plan as rearm2_plan,
)
from ofc_regular import (
    hu_m31_t3_step6d_performance_lock_rearm1_spot as rearm1_spot,
)
from ofc_regular import (
    hu_m31_t3_step6d_performance_lock_rearm2_open as rearm2_open,
)
from ofc_regular import run_hu_m31_t3_step6d_performance_v2 as runner


REPO_ROOT = Path(__file__).resolve().parents[1]
STARTUP = REPO_ROOT / "scripts/startup_hu_m31_t3_step6d_full100_v1.sh"
PLAN = (
    REPO_ROOT
    / "outputs/hu_joint_policy/m31_t3_step6d/performance_lock_rearm2/"
    "precontent_plan_v1.json"
)
AI_PROFILES = REPO_ROOT / "src/ofc_regular/ai_profiles.py"
ROOT_PREFIX = "frozen/full100_roots"
PLAN_MEMBER = "frozen/full100_plan.json"
CLAIM_MEMBER = "frozen/performance_lock_open_claim.json"
SEAL_MEMBER = "frozen/performance_lock_root_seal.json"
MATERIALIZATION_MEMBER = "frozen/performance_lock_materialization.json"
GLOBAL_SPOT_CLAIM_SCHEMA = (
    "hu_m31_t3_step6d_performance_lock_rearm2_global_spot_claim_v1"
)
GLOBAL_SPOT_CLAIM_STATUS = (
    "global_rearm2_one_shot_spot_identity_claimed_after_exhaustive_"
    "smoke_before_authorization"
)
EXPECTED_OUTPUT = (
    "lock_rearm2|"
    "hu_m31_t3_step6d_candidate02_"
    "performance_lock_recovery_source_shard_done_v3"
)


def _canonical(value: Any) -> bytes:
    return runner.canonical_bytes(value)


def _digest(value: Any) -> str:
    return runner.canonical_sha256(value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_canonical(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value))


def _fixture(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    Path,
    list[tuple[str, Path, str]],
    list[str],
    Path,
    Path,
]:
    plan_raw = PLAN.read_bytes()
    assert hashlib.sha256(plan_raw).hexdigest() == rearm2_plan.PRECONTENT_PLAN_SHA256
    plan = json.loads(plan_raw)
    contract = runner.validate_run_contract(plan["run_contract"])
    assert _digest(contract) == rearm2_plan.LOCK_RUN_CONTRACT_DIGEST

    claim_path = tmp_path / "must-remain-absent" / "GLOBAL_REARM2_CLAIM.json"
    root_output = tmp_path / "must-remain-absent" / "roots"
    package_directory = tmp_path / "local-package"
    claim = {
        "schema": rearm2_open.CLAIM_SCHEMA,
        "status": rearm2_open.CLAIM_STATUS,
        "global_claim_path": str(claim_path.resolve()),
        "lock_output_directory": str(root_output.resolve()),
        "precontent_plan": {"sha256": rearm2_plan.PRECONTENT_PLAN_SHA256},
        "lock_run_contract_digest": rearm2_plan.LOCK_RUN_CONTRACT_DIGEST,
        "ai_profiles_current": {
            "sha256": rearm2_plan.CURRENT_PROFILE_REGISTRY_SHA256,
        },
        "restrictions": {
            "alternate_seed_allowed": False,
            "reseed_allowed": False,
            "cloud_authorized": False,
            "training_authorized": False,
            "promotion_authorized": False,
            "current_profile_resolution_allowed": False,
            "runtime_activation_allowed": False,
            "opponent_private_discards_allowed": False,
        },
    }

    root_names = [
        f"{ROOT_PREFIX}/hand_{index:03d}.json" for index in range(100)
    ]
    root_bytes = [
        f"unreadable-rearm2-root-{index:03d}\n".encode("ascii")
        for index in range(100)
    ]
    root_hashes = [hashlib.sha256(raw).hexdigest() for raw in root_bytes]
    materialization = {
        "schema": rearm2_open.MATERIALIZATION_SCHEMA,
        "status": rearm2_open.MATERIALIZATION_STATUS,
        "global_claim_sha256": _digest(claim),
        "plan_sha256": rearm2_plan.PRECONTENT_PLAN_SHA256,
        "run_contract_digest": rearm2_plan.LOCK_RUN_CONTRACT_DIGEST,
        "hand_indices": list(range(100)),
        "root_count": 100,
        "root_artifact_sha256": root_hashes,
        "aggregate_root_sha256": _digest(root_hashes),
        "same_identity_resume_only": True,
        "reseeded": False,
        "training_eligible": False,
        "current_profile_changed": False,
    }
    seal = {
        "schema": rearm2_open.SEAL_SCHEMA,
        "status": rearm2_open.SEAL_STATUS,
        "global_claim_sha256": _digest(claim),
        "materialization_sha256": _digest(materialization),
        "plan_sha256": rearm2_plan.PRECONTENT_PLAN_SHA256,
        "run_contract_digest": rearm2_plan.LOCK_RUN_CONTRACT_DIGEST,
        "hand_indices": list(range(100)),
        "root_count": 100,
        "observation_count": 200,
        "root_artifact_sha256": root_hashes,
        "aggregate_root_sha256": _digest(root_hashes),
        "development_comparison": {
            "lock_fingerprint_overlap_count": 0,
            "lock_root_hash_overlap_count": 0,
            "lock_seed_overlap_count": 0,
        },
        "training_eligible": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
    }
    controls = {
        PLAN_MEMBER: plan_raw,
        CLAIM_MEMBER: _canonical(claim),
        SEAL_MEMBER: _canonical(seal),
        MATERIALIZATION_MEMBER: _canonical(materialization),
    }
    source_entries = {
        name: {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
        for name, raw in controls.items()
    }
    source_entries.update(
        {
            name: {
                "sha256": root_hashes[index],
                "bytes": len(root_bytes[index]),
            }
            for index, name in enumerate(root_names)
        }
    )

    source = tmp_path / "source.zip"
    with zipfile.ZipFile(
        source,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for name, raw in controls.items():
            archive.writestr(name, raw)
        for name, raw in zip(root_names, root_bytes, strict=True):
            archive.writestr(name, raw)

    jobs: list[tuple[str, Path, str]] = []
    job_records: list[dict[str, Any]] = []
    for row in plan["jobs"]:
        job_id = row["job_id"]
        job = runner.build_shard_manifest(
            run_contract=contract,
            source_role=row["source_role"],
            work_hand_indices=row["work_hand_indices"],
        )
        raw = _canonical(job)
        job_sha256 = hashlib.sha256(raw).hexdigest()
        assert job_sha256 == row["shard_manifest_sha256"]
        job_path = tmp_path / "jobs" / f"{job_id}.json"
        job_path.parent.mkdir(parents=True, exist_ok=True)
        job_path.write_bytes(raw)
        jobs.append((job_id, job_path, job_sha256))
        job_records.append(
            {
                "job_id": job_id,
                "source_role": row["source_role"],
                "shard_index": row["shard_index"],
                "work_hand_indices": row["work_hand_indices"],
                "path": f"jobs/{job_id}.json",
                "output_prefix": f"jobs/{job_id}",
                "sha256": job_sha256,
                "bytes": len(raw),
            }
        )

    run_name = "rearm2-actual-startup-no-root-read-integration"
    manifest = {
        "schema": "hu_m31_t3_step6d_performance_lock_spot_package_v1",
        "status": "immutable_performance_lock_package_ready_not_authorized",
        "run_name": run_name,
        "source_sha256": _sha256(source),
        "startup_sha256": _sha256(STARTUP),
        "plan_sha256": rearm2_plan.PRECONTENT_PLAN_SHA256,
        "run_contract": contract,
        "run_contract_digest": rearm2_plan.LOCK_RUN_CONTRACT_DIGEST,
        "job_manifests": job_records,
        "source_entries": source_entries,
        "tail_qualification": {
            "summary_sha256": (
                "f587df6d037313e2111a5cbc3d474370106f6a341ef4ec130cee7516192e668f"
            ),
            "validation_sha256": (
                "da252b3cabfc361d89de2f2fe08931fe61302287b69b82bf9f812a4432baf3eb"
            ),
            "scientific_merge_sha256": (
                "29e6b3db5b372f9808b681fb70246368452eecf5c6bbbca2519b2882430369f7"
            ),
            "receive_receipt_sha256": (
                "2ba1b7434cda012d8a230109e9c37db6d9e9b41587e1138e79790a177df85cf6"
            ),
            "development_run_contract_digest": (
                "92a87f1544a73104d5256db39b022094c8c7f7b11f77bd19dcf1ab1442581ebd"
            ),
            "all_gates_passed": True,
            "performance_candidate_frozen": True,
            "performance_lock_authorized": True,
            "open_claim_package_path": CLAIM_MEMBER,
            "open_claim_sha256": _digest(claim),
            "root_seal_package_path": SEAL_MEMBER,
            "root_seal_sha256": _digest(seal),
            "materialization_package_path": MATERIALIZATION_MEMBER,
            "materialization_sha256": _digest(materialization),
            "lock_root_aggregate_sha256": _digest(root_hashes),
            "lock_development_overlap_count": 0,
        },
        "spot_execution_authorized": False,
        "performance_lock_authorized": False,
        "quality_pilot_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "m31_complete": False,
        "gcloud_invoked": False,
    }
    manifest_path = tmp_path / "manifest.json"
    _write_canonical(manifest_path, manifest)

    receipt_sha256 = hashlib.sha256(
        b"deterministic-local-preview-rearm2-smoke-receipt"
    ).hexdigest()
    authorized_job_ids = [job_id for job_id, _path, _sha in jobs]
    global_spot_claim = {
        "schema": GLOBAL_SPOT_CLAIM_SCHEMA,
        "status": GLOBAL_SPOT_CLAIM_STATUS,
        "global_root_claim_path": claim["global_claim_path"],
        "global_root_claim_sha256": _digest(claim),
        "lock_output_directory": claim["lock_output_directory"],
        "package_run_directory": str(package_directory.resolve()),
        "run_name": run_name,
        "package_manifest_sha256": _sha256(manifest_path),
        "source_sha256": manifest["source_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "precontent_plan_sha256": rearm2_plan.PRECONTENT_PLAN_SHA256,
        "root_seal_sha256": _digest(seal),
        "run_contract_digest": rearm2_plan.LOCK_RUN_CONTRACT_DIGEST,
        "authorized_job_ids": authorized_job_ids,
        "max_initial_jobs": 20,
        "max_resume_attempts": 1,
        "alternate_package_authorization_allowed": False,
        "quality_pilot_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "runtime_policy_activated": False,
        "claimed_unix_ns": 1,
        "preauthorize_smoke_receipt_sha256": receipt_sha256,
    }
    authorization = {
        "schema": "hu_m31_t3_step6d_performance_lock_launch_authorization_v1",
        "status": "explicit_one_shot_performance_lock_spot_authorization",
        "package_manifest_sha256": _sha256(manifest_path),
        "source_sha256": manifest["source_sha256"],
        "plan_sha256": rearm2_plan.PRECONTENT_PLAN_SHA256,
        "run_contract_digest": rearm2_plan.LOCK_RUN_CONTRACT_DIGEST,
        "global_spot_claim": global_spot_claim,
        "preauthorize_smoke_receipt_sha256": receipt_sha256,
        "performance_development_only": False,
        "spot_execution_authorized": True,
        "performance_lock_authorized": True,
        "quality_pilot_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
    }
    authorization_path = tmp_path / "preview-authorization.json"
    _write_canonical(authorization_path, authorization)

    poisoned_source = tmp_path / "poisoned-source.zip"
    assert (
        rearm1_spot._poison_temporary_root_payloads(
            source,
            poisoned_source,
        )
        == 100
    )
    return (
        poisoned_source,
        manifest_path,
        authorization_path,
        jobs,
        root_names,
        claim_path,
        root_output,
    )


def test_actual_startup_precontent_validates_all_rearm2_v3_jobs_without_root_read(
    tmp_path: Path,
) -> None:
    startup_before = _sha256(STARTUP)
    current_before = _sha256(AI_PROFILES)
    (
        poisoned_source,
        manifest,
        authorization,
        jobs,
        root_names,
        claim_path,
        root_output,
    ) = _fixture(tmp_path)
    assert not claim_path.exists()
    assert not root_output.exists()

    verifier = rearm1_spot._extract_precontent_verifier(STARTUP)
    observed = []
    for job_id, job_path, job_sha256 in jobs:
        completed = subprocess.run(
            [
                sys.executable,
                "-",
                str(poisoned_source),
                str(manifest),
                str(authorization),
                str(job_path),
                job_id,
                job_sha256,
            ],
            input=verifier,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert completed.returncode == 0, (
            job_id,
            completed.stdout,
            completed.stderr,
        )
        observed.append(completed.stdout.strip())
    assert observed == [EXPECTED_OUTPUT] * 20

    # Every temporary root is actually unreadable.  Since the real verifier
    # succeeded for all jobs, it could not have opened any root member.
    with zipfile.ZipFile(poisoned_source) as archive:
        for name in root_names:
            with pytest.raises(
                (zipfile.BadZipFile, zlib.error, RuntimeError, OSError)
            ):
                archive.read(name)

    assert not claim_path.exists()
    assert not root_output.exists()
    assert _sha256(STARTUP) == startup_before
    assert _sha256(AI_PROFILES) == current_before
    assert current_before == rearm2_plan.CURRENT_PROFILE_REGISTRY_SHA256
