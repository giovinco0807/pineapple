from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
STARTUP = REPO_ROOT / "scripts/startup_hu_m31_t3_step6d_full100_v1.sh"
PLAN = (
    REPO_ROOT / "outputs/hu_joint_policy/m31_t3_step6d/performance_lock/"
    "precontent_plan_v1.json"
)
PLAN_SHA256 = "d2c9985b574839cf7e1515c12ebffc813ac6b4fb4e6494ed8d1288ef582a4b6b"
RUN_DIGEST = "e73c2b06279c1f1e91c38b2887ee465acf85f8f1072afef636512283252c34a4"
CURRENT_SHA256 = "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _precontent_block() -> str:
    source = STARTUP.read_text(encoding="utf-8")
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", source, flags=re.DOTALL)
    matches = [block for block in blocks if "PRE-CONTENT ORDERING" in block]
    assert len(matches) == 1
    return matches[0]


def _worker_validation_block() -> str:
    source = STARTUP.read_text(encoding="utf-8")
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", source, flags=re.DOTALL)
    matches = [block for block in blocks if "development_authorization_keys" in block]
    assert len(matches) == 1
    return matches[0]


def _worker_global_spot_validator():
    tree = ast.parse(_worker_validation_block())
    key_sets = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "GLOBAL_SPOT_CLAIM_KEYS"
            for target in node.targets
        )
    ]
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "validate_global_spot_claim"
    ]
    assert len(key_sets) == 1
    assert len(matches) == 1
    module = ast.fix_missing_locations(
        ast.Module(body=[*key_sets, *matches], type_ignores=[])
    )

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise SystemExit(message)

    namespace = {"require": require, "digest": _digest}
    exec(compile(module, "<worker-global-spot-validator>", "exec"), namespace)
    return namespace["validate_global_spot_claim"]


def _write_canonical(path: Path, value: Any) -> None:
    path.write_bytes(_canonical(value))


def _lock_fixture(
    directory: Path, *, tamper_claim_status: bool = False
) -> tuple[Path, Path, Path]:
    plan_raw = PLAN.read_bytes()
    assert hashlib.sha256(plan_raw).hexdigest() == PLAN_SHA256
    run_name = "regular-hu-m31-lock-startup-test-001"
    global_root_claim_path = directory / "GLOBAL_PERFORMANCE_LOCK_CLAIM.json"
    lock_output_directory = directory / "lock-output"
    package_run_directory = directory / "package"

    claim = {
        "schema": "hu_m31_t3_step6d_performance_lock_global_claim_v1",
        "status": (
            "tampered"
            if tamper_claim_status
            else (
                "global_one_shot_claim_persisted_before_lock_root_touch_"
                "crash_consumes_claim"
            )
        ),
        "global_claim_path": str(global_root_claim_path.resolve()),
        "lock_output_directory": str(lock_output_directory.resolve()),
        "precontent_plan": {"sha256": PLAN_SHA256},
        "lock_run_contract_digest": RUN_DIGEST,
        "ai_profiles_current": {"sha256": CURRENT_SHA256},
        "restrictions": {
            "timing_used_for_root_selection": False,
            "q_used_for_root_selection": False,
            "ev_used_for_root_selection": False,
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
    root_bytes = [
        f"intentionally-not-json-root-{index}\n".encode() for index in range(100)
    ]
    root_hashes = [hashlib.sha256(raw).hexdigest() for raw in root_bytes]
    materialization = {
        "schema": "hu_m31_t3_step6d_performance_lock_root_materialization_v1",
        "status": "all_100_lock_roots_materialized_same_identity",
        "global_claim_sha256": _digest(claim),
        "plan_sha256": PLAN_SHA256,
        "run_contract_digest": RUN_DIGEST,
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
        "schema": "hu_m31_t3_step6d_performance_lock_root_seal_v1",
        "status": "sealed_100_disjoint_hidden_safe_performance_lock_roots",
        "global_claim_sha256": _digest(claim),
        "materialization_sha256": _digest(materialization),
        "plan_sha256": PLAN_SHA256,
        "run_contract_digest": RUN_DIGEST,
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
        "frozen/full100_plan.json": plan_raw,
        "frozen/performance_lock_open_claim.json": _canonical(claim),
        "frozen/performance_lock_root_seal.json": _canonical(seal),
        "frozen/performance_lock_materialization.json": _canonical(materialization),
    }
    entries = {
        name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        for name, raw in controls.items()
    }
    for index, raw in enumerate(root_bytes):
        entries[f"frozen/full100_roots/hand_{index:03d}.json"] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
    qualification = {
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
        "open_claim_package_path": "frozen/performance_lock_open_claim.json",
        "open_claim_sha256": _digest(claim),
        "root_seal_package_path": "frozen/performance_lock_root_seal.json",
        "root_seal_sha256": _digest(seal),
        "materialization_package_path": (
            "frozen/performance_lock_materialization.json"
        ),
        "materialization_sha256": _digest(materialization),
        "lock_root_aggregate_sha256": _digest(root_hashes),
        "lock_root_topology_sha256": "1" * 64,
        "lock_observation_fingerprint_sha256": "2" * 64,
        "lock_development_overlap_count": 0,
    }
    manifest = {
        "schema": "hu_m31_t3_step6d_performance_lock_spot_package_v1",
        "status": "immutable_performance_lock_package_ready_not_authorized",
        "run_name": run_name,
        "source_sha256": "a" * 64,
        "startup_sha256": "b" * 64,
        "plan_sha256": PLAN_SHA256,
        "run_contract_digest": RUN_DIGEST,
        "tail_qualification": qualification,
        "source_entries": entries,
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
    source = directory / "source.zip"
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in controls.items():
            archive.writestr(name, raw)
        for index, raw in enumerate(root_bytes):
            archive.writestr(f"frozen/full100_roots/hand_{index:03d}.json", raw)
    manifest_path = directory / "manifest.json"
    _write_canonical(manifest_path, manifest)
    job_ids = [
        f"{role}-shard-{index:02d}"
        for role in ("candidate", "reference")
        for index in range(10)
    ]
    global_spot_claim = {
        "schema": "hu_m31_t3_step6d_performance_lock_global_spot_claim_v1",
        "status": "global_one_shot_spot_identity_claimed_before_authorization",
        "global_root_claim_path": claim["global_claim_path"],
        "global_root_claim_sha256": _digest(claim),
        "lock_output_directory": claim["lock_output_directory"],
        "package_run_directory": str(package_run_directory.resolve()),
        "run_name": run_name,
        "package_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "source_sha256": manifest["source_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "precontent_plan_sha256": PLAN_SHA256,
        "root_seal_sha256": _digest(seal),
        "run_contract_digest": RUN_DIGEST,
        "authorized_job_ids": job_ids,
        "max_initial_jobs": 20,
        "max_resume_attempts": 1,
        "alternate_package_authorization_allowed": False,
        "quality_pilot_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "runtime_policy_activated": False,
        "claimed_unix_ns": 1,
    }
    authorization = {
        "schema": "hu_m31_t3_step6d_performance_lock_launch_authorization_v1",
        "status": "explicit_one_shot_performance_lock_spot_authorization",
        "package_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "source_sha256": manifest["source_sha256"],
        "plan_sha256": PLAN_SHA256,
        "run_contract_digest": RUN_DIGEST,
        "global_spot_claim": global_spot_claim,
        "performance_development_only": False,
        "spot_execution_authorized": True,
        "performance_lock_authorized": True,
        "quality_pilot_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
    }
    authorization_path = directory / "authorization.json"
    _write_canonical(authorization_path, authorization)
    return source, manifest_path, authorization_path


def test_lock_precontent_validator_does_not_open_root_members(tmp_path: Path) -> None:
    source, manifest, authorization = _lock_fixture(tmp_path)
    # Every root is deliberately invalid JSON.  A successful preflight proves
    # the pre-content block read only control artifacts, not root content.
    result = subprocess.run(
        [sys.executable, "-", str(source), str(manifest), str(authorization)],
        input=_precontent_block(),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "lock"


def test_lock_precontent_validator_rejects_claim_status_before_roots(
    tmp_path: Path,
) -> None:
    source, manifest, authorization = _lock_fixture(tmp_path, tamper_claim_status=True)
    result = subprocess.run(
        [sys.executable, "-", str(source), str(manifest), str(authorization)],
        input=_precontent_block(),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0
    assert "open claim changed" in result.stderr


def test_lock_precontent_validator_rejects_global_spot_claim_before_roots(
    tmp_path: Path,
) -> None:
    source, manifest, authorization = _lock_fixture(tmp_path)
    value = json.loads(authorization.read_text(encoding="utf-8"))
    value["global_spot_claim"]["global_root_claim_sha256"] = "0" * 64
    _write_canonical(authorization, value)
    result = subprocess.run(
        [sys.executable, "-", str(source), str(manifest), str(authorization)],
        input=_precontent_block(),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0
    assert "global root/Spot claim chain changed" in result.stderr


def test_lock_precontent_branch_precedes_unzip_and_root_copy() -> None:
    source = STARTUP.read_text(encoding="utf-8")
    precontent = source.index("PRE-CONTENT ORDERING")
    unzip = source.index("unzip -q /tmp/source.zip", precontent)
    root_copy = source.index(
        'cp "$WORK/frozen/full100_roots/hand_$hand_pad.json"', unzip
    )
    assert precontent < unzip < root_copy
    assert "archive.read(relative)" in _precontent_block()
    assert "archive.read(ROOT_PREFIX" not in _precontent_block()
    assert "Do not archive.read() a root here." in _precontent_block()


def test_lock_global_spot_claim_is_revalidated_in_normal_exact_key_stage() -> None:
    block = _worker_validation_block()
    assert "development_authorization_keys" in block
    assert '{"global_spot_claim"} if lock_mode else set()' in block
    assert 'a["global_spot_claim"]' in block
    assert 'work / "frozen/performance_lock_open_claim.json"' in block
    assert 'work / "frozen/performance_lock_root_seal.json"' in block
    assert "validate_global_spot_claim(" in block
    assert "global_root_claim_sha256" in block
    assert "max_initial_jobs" in block
    assert "max_resume_attempts" in block
    assert "claimed_unix_ns" in block


def test_lock_normal_stage_rejects_tampered_global_spot_claim(
    tmp_path: Path,
) -> None:
    source, manifest_path, authorization_path = _lock_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    with zipfile.ZipFile(source) as archive:
        root_claim = json.loads(archive.read("frozen/performance_lock_open_claim.json"))
        root_seal = json.loads(archive.read("frozen/performance_lock_root_seal.json"))
    validator = _worker_global_spot_validator()
    job_ids = [
        f"{role}-shard-{index:02d}"
        for role in ("candidate", "reference")
        for index in range(10)
    ]
    validator(
        authorization["global_spot_claim"],
        root_claim=root_claim,
        root_seal=root_seal,
        manifest=manifest,
        manifest_sha=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        job_ids=job_ids,
    )
    tampered = json.loads(json.dumps(authorization["global_spot_claim"]))
    tampered["alternate_package_authorization_allowed"] = True
    with pytest.raises(
        SystemExit,
        match="forbidden performance-lock global Spot flag",
    ):
        validator(
            tampered,
            root_claim=root_claim,
            root_seal=root_seal,
            manifest=manifest,
            manifest_sha=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            job_ids=job_ids,
        )


def test_lock_and_development_worker_contracts_are_both_preserved() -> None:
    source = STARTUP.read_text(encoding="utf-8")
    for value in (
        "hu_m31_t3_step6d_full100_spot_package_v1",
        "hu_m31_t3_step6d_performance_lock_spot_package_v1",
        "9ef14137b97db975bed683bcdd7e53b27414d2efab282c6f98390d7702944758",
        PLAN_SHA256,
        "92a87f1544a73104d5256db39b022094c8c7f7b11f77bd19dcf1ab1442581ebd",
        RUN_DIGEST,
        "hu_m31_t3_step6d_candidate02_performance_lock_run_contract_v1",
        "hu_m31_t3_step6d_candidate02_performance_lock_source_shard_done_v1",
        "hu_m31_t3_step6d_performance_lock_launch_claim_v1",
        "hu_m31_t3_step6d_performance_lock_resume_claim_v1",
        "hu_m31_t3_step6d_performance_lock_global_spot_claim_v1",
        "exclusive_performance_lock_attempt1_claim_before_remote_mutation",
    ):
        assert value in source
    assert 'a["performance_development_only"] is (not lock_mode)' in source
    assert 'a["performance_lock_authorized"] is lock_mode' in source
