from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

import ofc_regular.finalize_hu_m43_attempt08_preflight as finalizer
import ofc_regular.hu_m43_attempt08_spot as spot
from ofc_regular.hu_m43_attempt08_contract import (
    ATTEMPT08_LAMBDA_MODEL_SHA256,
    M43_ATTEMPT08_PROFILES,
    load_and_validate_attempt08_plan,
)
from ofc_regular.run_hu_m43_attempt08_development import (
    ATTEMPT08_CHECKPOINT_SCHEMA,
    ATTEMPT08_HEARTBEAT_SCHEMA,
    ATTEMPT08_SHARD_SUMMARY_SCHEMA,
    load_attempt08_development_open_bindings,
)
from ofc_regular.run_hu_m43_attempt08_preflight import ATTEMPT08_PREFLIGHT_SLOTS


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/hu_joint_policy_m43_attempt08.json"
PREFLIGHT_PLAN = ROOT / "configs/hu_joint_policy_m43_attempt08_preflight.json"
MODEL = (
    ROOT
    / "outputs/hu_joint_policy/m43_attempt05_old_dev900_architecture_once"
    / "lambda_rank_candidate.pkl"
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(spot.canonical_json_bytes(payload))


def _preflight_artifacts(tmp_path: Path) -> dict[str, object]:
    proof_paths: dict[str, Path] = {}
    for slot in ATTEMPT08_PREFLIGHT_SLOTS:
        path = tmp_path / f"proof-{slot}.json"
        _write_json(path, {"slot": slot})
        proof_paths[slot] = path
    proof_hashes = {slot: spot.sha256_file(path) for slot, path in proof_paths.items()}
    execution_evidence_payload = {
        "schema": finalizer.ATTEMPT08_PREFLIGHT_EXECUTION_EVIDENCE_SCHEMA,
        "status": "all_five_spot_jobs_received_and_validated",
        "run_name": "attempt08-preflight-test-run",
        "manifest_sha256": "a" * 64,
        "launch_authorization_sha256": "b" * 64,
        "local_evidence_sha256": "c" * 64,
        "done_sha256": {
            slot: f"{index + 1:x}" * 64
            for index, slot in enumerate(ATTEMPT08_PREFLIGHT_SLOTS)
        },
        "support_sha256": {
            slot: {
                "proof": "6" * 64,
                "checkpoint": "7" * 64,
                "heartbeat": "8" * 64,
                "summary": "9" * 64,
                "run_log": "a" * 64,
                "boot_image": "b" * 64,
            }
            for slot in ATTEMPT08_PREFLIGHT_SLOTS
        },
        "runtime": {
            "runtime_semantic_anchor_sha256": (
                spot.ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256
            ),
            "runtime_source_closure_sha256": (
                spot.ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256
            ),
            "runtime_fingerprint_sha256": (
                spot.ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256
            ),
            "runtime_requirements_sha256": (
                spot.ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256
            ),
            "gcp_image_name": spot.ATTEMPT08_GCP_IMAGE_NAME,
            "gcp_image_id": spot.ATTEMPT08_GCP_IMAGE_ID,
        },
        "all_five_done_before_payloads_opened": True,
        "proof_payloads_opened_after_all_done": True,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    execution_evidence = tmp_path / "preflight-execution-evidence.json"
    _write_json(execution_evidence, execution_evidence_payload)
    execution_evidence_sha = spot.sha256_file(execution_evidence)
    aggregate_payload = {
        "proof_file_sha256": proof_hashes,
        "proof_evidence_sha256": finalizer._sha256_value(proof_hashes),
        "proof_gates": {"all": True},
        "operational_gates": {"all": True},
        "spot_operational_evidence_sha256": execution_evidence_sha,
        "contract": {
            "runtime_semantic_anchor_sha256": (
                spot.ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256
            ),
            "runtime_source_closure_sha256": (
                spot.ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256
            ),
            "runtime_requirements_sha256": spot.ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
            "gcp_image_name": spot.ATTEMPT08_GCP_IMAGE_NAME,
            "gcp_image_id": spot.ATTEMPT08_GCP_IMAGE_ID,
        },
        "runtime_fingerprint_sha256": (
            spot.ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256
        ),
    }
    aggregate = tmp_path / "preflight-aggregate.json"
    _write_json(aggregate, aggregate_payload)
    aggregate_sha = spot.sha256_file(aggregate)
    authorization_payload = finalizer._authorization_payload(
        aggregate=aggregate_payload,
        aggregate_sha256=aggregate_sha,
    )
    authorization = tmp_path / "development-open.json"
    authorization.write_bytes(finalizer.canonical_json_bytes(authorization_payload))
    finalization_payload = {
        "schema": finalizer.ATTEMPT08_PREFLIGHT_FINALIZATION_SCHEMA,
        "milestone": "M4.3-attempt08",
        "status": finalizer.ATTEMPT08_PREFLIGHT_GO_STATUS,
        "decision": "authorize_exact_development_roots_0_through_199_only",
        "inputs": {
            "attempt08_plan_sha256": authorization_payload["target_plan"]["sha256"],
            "preflight_plan_sha256": authorization_payload["preflight_plan"]["sha256"],
            "preflight_result_schema": authorization_payload["preflight_result"]["schema"],
            "preflight_result_sha256": aggregate_sha,
            "proof_evidence_sha256": aggregate_payload["proof_evidence_sha256"],
            "runtime_semantic_anchor_sha256": spot.ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256,
            "runtime_source_closure_sha256": spot.ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256,
            "runtime_fingerprint_sha256": spot.ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
            "runtime_requirements_sha256": spot.ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
            "gcp_image_name": spot.ATTEMPT08_GCP_IMAGE_NAME,
            "gcp_image_id": spot.ATTEMPT08_GCP_IMAGE_ID,
            "preflight_execution_evidence_sha256": execution_evidence_sha,
        },
        "gate_digests": {
            "proof_gates_sha256": authorization_payload["evidence"]["proof_gates_sha256"],
            "operational_gates_sha256": authorization_payload["evidence"]["operational_gates_sha256"],
        },
        "authorization": {
            "emitted": True,
            "schema": finalizer.ATTEMPT08_DEVELOPMENT_OPEN_AUTHORIZATION_SCHEMA,
            "sha256": spot.sha256_file(authorization),
        },
        "science_boundary": {
            "future_audit_authorized": False,
            "fit_started": False,
            "threshold_selection_started": False,
            "runtime_policy_activated": False,
            "current_profile_changed": False,
            "full_replacement_enabled": False,
        },
        "next": "development_runner_and_spot_package_must_consume_exact_authorization_sha",
    }
    finalization_path = tmp_path / "preflight-finalization.json"
    _write_json(finalization_path, finalization_payload)
    source = tmp_path / "preflight-source.json"
    _write_json(source, {"source": "test"})

    spot_run = tmp_path / "preflight-spot-run"
    spot_run.mkdir()
    for filename in spot._PREFLIGHT_PACKAGE_FILES:
        (spot_run / filename).write_bytes(f"test:{filename}\n".encode("utf-8"))
    (spot_run / "package_src").mkdir()
    (spot_run / "package_src/placeholder.txt").write_text("test\n", encoding="utf-8")
    spot_authorization = tmp_path / "preflight-spot-authorization.json"
    _write_json(spot_authorization, {"authorization": "test"})
    spot_local_evidence = tmp_path / "preflight-spot-local-evidence.json"
    _write_json(spot_local_evidence, {"evidence": "test"})
    spot_jobs = tmp_path / "preflight-spot-jobs"
    spot_jobs.mkdir()
    for spec in spot.build_preflight_schedule():
        job = spot_jobs / spec["output_prefix"]
        job.mkdir()
        for filename in spot._PREFLIGHT_JOB_FILES:
            (job / filename).write_bytes(f"test:{filename}\n".encode("utf-8"))
    return {
        "proof_paths": proof_paths,
        "aggregate": aggregate,
        "aggregate_payload": aggregate_payload,
        "execution_evidence": execution_evidence,
        "execution_evidence_payload": execution_evidence_payload,
        "authorization": authorization,
        "finalization": finalization_path,
        "source": source,
        "spot_run": spot_run,
        "spot_authorization": spot_authorization,
        "spot_local_evidence": spot_local_evidence,
        "spot_jobs": spot_jobs,
    }


@pytest.fixture()
def package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    repo = tmp_path / "repo"
    source = repo / "src/ofc_regular"
    source.mkdir(parents=True)
    shutil.copyfile(ROOT / "src/ofc_regular/ai_profiles.py", source / "ai_profiles.py")
    (repo / "configs").mkdir()
    shutil.copyfile(
        ROOT / "configs/hu_m43_attempt08_runtime_requirements.txt",
        repo / "configs/hu_m43_attempt08_runtime_requirements.txt",
    )
    for relative in spot.ATTEMPT08_RUNTIME_SEMANTIC_FILES:
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    template = (
        repo / "outputs/gcp_runs" / spot.PINNED_TEMPLATE_RUN / "package_src"
    )
    (template / "models").mkdir(parents=True)
    (template / "native").mkdir(parents=True)
    (template / "models/fake.pkl").write_bytes(b"model")
    (template / "native/fake.so").write_bytes(b"native")
    model_manifest = {
        "schema": "test_model_manifest",
        "models": [
            {
                "path": "models/fake.pkl",
                "bytes": 5,
                "sha256": _sha(b"model"),
            }
        ],
    }
    native_manifest = {
        "schema": "test_native_manifest",
        "binaries": [
            {
                "path": "native/fake.so",
                "bytes": 6,
                "sha256": _sha(b"native"),
            }
        ],
    }
    _write_json(template / "source_model_manifest.json", model_manifest)
    _write_json(template / "source_native_manifest.json", native_manifest)
    monkeypatch.setattr(
        spot,
        "_verify_pinned_runtime_closure",
        lambda _path: (model_manifest, native_manifest),
    )
    monkeypatch.setattr(
        spot,
        "PINNED_MODEL_MANIFEST_SHA256",
        spot.sha256_file(template / "source_model_manifest.json"),
    )
    monkeypatch.setattr(
        spot,
        "PINNED_NATIVE_MANIFEST_SHA256",
        spot.sha256_file(template / "source_native_manifest.json"),
    )
    startup = repo / "scripts/startup_hu_m43_attempt08_development.sh"
    startup.parent.mkdir(parents=True, exist_ok=True)
    startup.write_text("#!/usr/bin/env bash\nset -euo pipefail\n", encoding="utf-8")
    preflight = _preflight_artifacts(tmp_path)
    authorization = preflight["authorization"]
    assert isinstance(authorization, Path)
    monkeypatch.setattr(
        spot,
        "load_and_validate_preflight_aggregate",
        lambda _path: preflight["aggregate_payload"],
    )
    monkeypatch.setattr(
        spot, "validate_preflight_aggregate_with_proofs", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        spot, "validate_runtime_semantic_anchor", lambda **kwargs: {"status": "ok"}
    )
    monkeypatch.setattr(
        spot,
        "validate_preflight_receive_bundle",
        lambda **kwargs: {
            "execution_evidence": preflight["execution_evidence_payload"],
            "proof_paths": preflight["proof_paths"],
            "done_rows": {},
        },
    )
    run_name = "attempt08-test-run"
    run_dir = repo / "outputs/gcp_runs" / run_name
    result = spot.package_attempt08_spot(
        repo_root=repo,
        run_dir=run_dir,
        run_name=run_name,
        plan_path=PLAN,
        preflight_plan_path=PREFLIGHT_PLAN,
        preflight_aggregate_path=preflight["aggregate"],
        preflight_execution_evidence_path=preflight["execution_evidence"],
        preflight_spot_run_dir=preflight["spot_run"],
        preflight_spot_launch_authorization_path=preflight[
            "spot_authorization"
        ],
        preflight_spot_local_evidence_path=preflight["spot_local_evidence"],
        preflight_spot_jobs_root=preflight["spot_jobs"],
        preflight_finalization_path=preflight["finalization"],
        preflight_proof_paths=preflight["proof_paths"],
        preflight_source_path=preflight["source"],
        development_open_authorization_path=authorization,
        model_path=MODEL,
        startup_path=startup,
    )
    assert result["fresh_root_opened"] is False
    launch = run_dir / "launch_authorization.json"
    spot.authorize_attempt08_launch(run_dir=run_dir, output=launch)
    return {
        "repo": repo,
        "run_dir": run_dir,
        "launch": launch,
        "authorization": authorization,
    }


def test_schedule_is_exact_balanced_development200() -> None:
    schedule = spot.build_attempt08_spot_schedule(
        load_and_validate_attempt08_plan(PLAN)
    )
    assert len(schedule) == 200
    assert [row["root_index"] for row in schedule] == list(range(200))
    assert {row["roots"] for row in schedule} == {1}
    assert {row["native_batch_threads"] for row in schedule} == {4}
    assert {row["machine_type"] for row in schedule} == {"c4-highmem-4"}
    assert {
        profile: sum(row["root_profile"] == profile for row in schedule)
        for profile in M43_ATTEMPT08_PROFILES
    } == {profile: 40 for profile in M43_ATTEMPT08_PROFILES}
    all_seeds = [seed for row in schedule for seed in row["seeds"].values()]
    assert len(all_seeds) == len(set(all_seeds)) == 1200


def test_package_rejects_noncanonical_startup_before_opening_inputs(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    alternate = repo / "scripts/alternate-startup.sh"
    alternate.parent.mkdir(parents=True)
    alternate.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    proofs = {
        slot: tmp_path / f"unused-{slot}.json"
        for slot in ATTEMPT08_PREFLIGHT_SLOTS
    }
    with pytest.raises(ValueError, match="canonical development startup"):
        spot.package_attempt08_spot(
            repo_root=repo,
            run_dir=repo / "outputs/gcp_runs/attempt08-alt-startup",
            run_name="attempt08-alt-startup",
            plan_path=tmp_path / "unused-plan.json",
            preflight_plan_path=tmp_path / "unused-preflight-plan.json",
            preflight_aggregate_path=tmp_path / "unused-aggregate.json",
            preflight_execution_evidence_path=tmp_path / "unused-evidence.json",
            preflight_spot_run_dir=tmp_path / "unused-preflight-run",
            preflight_spot_launch_authorization_path=tmp_path / "unused-launch.json",
            preflight_spot_local_evidence_path=tmp_path / "unused-local.json",
            preflight_spot_jobs_root=tmp_path / "unused-jobs",
            preflight_finalization_path=tmp_path / "unused-finalization.json",
            preflight_proof_paths=proofs,
            preflight_source_path=tmp_path / "unused-source.jsonl",
            development_open_authorization_path=tmp_path / "unused-open.json",
            model_path=tmp_path / "unused-model.pkl",
            startup_path=alternate,
        )


def test_package_then_separate_authorization_is_immutable(package: dict[str, Path]) -> None:
    manifest = spot.validate_package(package["run_dir"])
    authorization = spot.validate_launch_authorization(
        package["launch"], run_dir=package["run_dir"]
    )
    assert manifest["total_shards"] == 200
    assert manifest["recommended_wave_shards"] == 25
    assert authorization["spot_authorized"] is True
    assert authorization["package_frozen_before_authorization"] is True
    assert authorization["development_started"] is False
    for relative in (
        "scripts/HuM43Attempt08Spot.Common.ps1",
        "scripts/HuM43Attempt04Spot.Common.ps1",
    ):
        assert (package["run_dir"] / "package_src" / relative).read_bytes() == (
            package["repo"] / relative
        ).read_bytes()
    with pytest.raises(FileExistsError):
        spot.authorize_attempt08_launch(
            run_dir=package["run_dir"], output=package["launch"]
        )


def test_source_closure_tamper_is_rejected(package: dict[str, Path]) -> None:
    target = package["run_dir"] / "package_src/src/ofc_regular/ai_profiles.py"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="packaged source changed"):
        spot.validate_package(package["run_dir"])


@pytest.mark.parametrize(
    "relative",
    (
        "scripts/HuM43Attempt08Spot.Common.ps1",
        "scripts/HuM43Attempt04Spot.Common.ps1",
    ),
)
def test_transitive_common_script_tamper_is_rejected(
    package: dict[str, Path], relative: str
) -> None:
    target = package["run_dir"] / "package_src" / relative
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="packaged source changed"):
        spot.validate_package(package["run_dir"])


def test_frozen_preflight_execution_bundle_tamper_is_rejected(
    package: dict[str, Path],
) -> None:
    spec = spot.build_preflight_schedule()[0]
    target = (
        package["run_dir"]
        / "package_src/frozen/preflight_execution_bundle/jobs"
        / spec["output_prefix"]
        / "boot_image_evidence.json"
    )
    target.write_bytes(target.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="packaged source changed"):
        spot.validate_package(package["run_dir"])


def _done_payloads(
    package: dict[str, Path], received: Path
) -> tuple[Path, dict, dict[str, str]]:
    run_dir = package["run_dir"]
    manifest = spot.validate_package(run_dir)
    manifest_sha = spot.sha256_file(run_dir / "manifest.json")
    launch_sha = spot.sha256_file(package["launch"])
    schedule = spot.build_attempt08_spot_schedule(
        load_and_validate_attempt08_plan(run_dir / "hu_joint_policy_m43_attempt08.json")
    )
    bindings = load_attempt08_development_open_bindings(
        run_dir / "development_open_authorization.json",
        plan=run_dir / "hu_joint_policy_m43_attempt08.json",
        preflight_plan=run_dir / "hu_joint_policy_m43_attempt08_preflight.json",
    )
    shard_dir = received / "shard_000"
    shard_dir.mkdir(parents=True)
    teacher = {"root_index": 0, "placeholder": "validated-by-test-double"}
    _write_json(shard_dir / "teacher.jsonl", teacher)
    config_sha = "e" * 64
    generator_elapsed = 12.5
    common = {
        "config_sha256": config_sha,
        "plan_sha256": manifest["plan_sha256"],
        "ai_profiles_sha256": manifest["ai_profiles_sha256"],
        "model_sha256": manifest["model_sha256"],
        **bindings,
    }
    checkpoint = {
        "schema": ATTEMPT08_CHECKPOINT_SCHEMA,
        **common,
        "completed_roots": 1,
        "target_roots": 1,
        "root_index": 0,
        "partial_sha256": spot.sha256_file(shard_dir / "teacher.jsonl"),
        "updated_unix_seconds": 1.0,
        "generator_elapsed_seconds": generator_elapsed,
        "generator_peak_rss_bytes": 120000,
    }
    summary = {
        "schema": ATTEMPT08_SHARD_SUMMARY_SCHEMA,
        "status": "complete",
        "root_index": 0,
        "completed_roots": 1,
        "target_roots": 1,
        **common,
        "output_sha256": spot.sha256_file(shard_dir / "teacher.jsonl"),
        "elapsed_seconds": generator_elapsed,
        "generator_peak_rss_bytes": 120000,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    heartbeat = {
        **summary,
        "schema": ATTEMPT08_HEARTBEAT_SCHEMA,
        "summary_schema": ATTEMPT08_SHARD_SUMMARY_SCHEMA,
    }
    _write_json(shard_dir / "checkpoint.json", checkpoint)
    _write_json(shard_dir / "generator_summary.json", summary)
    _write_json(shard_dir / "heartbeat.json", heartbeat)
    (shard_dir / "run.log").write_bytes(b"test run\n")
    (shard_dir / "time.txt").write_text(
        "Elapsed (wall clock) time (h:mm:ss or m:ss): 0:13.00\n"
        "Maximum resident set size (kbytes): 121\n",
        encoding="utf-8",
    )
    resume_commit = {
        "schema": "hu_m43_attempt08_development_resume_commit_v1",
        "status": "committed_completed_root_pair",
        "run_name": manifest["run_name"],
        "shard": 0,
        "manifest_sha256": manifest_sha,
        "launch_authorization_sha256": launch_sha,
        "output_sha256": spot.sha256_file(shard_dir / "teacher.jsonl"),
        "checkpoint_sha256": spot.sha256_file(shard_dir / "checkpoint.json"),
        "time_report_sha256": spot.sha256_file(shard_dir / "time.txt"),
        "generator_elapsed_seconds": generator_elapsed,
        "process_elapsed_seconds": 13.0,
        "peak_rss_bytes": 130000,
    }
    _write_json(shard_dir / "resume_commit.json", resume_commit)
    _write_json(
        shard_dir / "boot_image_evidence.json",
        {
            "schema": "hu_m43_attempt08_development_boot_image_evidence_v1",
            "run_name": manifest["run_name"],
            "shard": 0,
            "instance_name": "attempt08-test-000",
            "disk_name": "attempt08-test-000",
            "source_image": spot.ATTEMPT08_GCP_IMAGE_SELF_LINK,
            "source_image_id": spot.ATTEMPT08_GCP_IMAGE_ID,
        },
    )
    global_claim = spot.build_global_claim(
        run_dir=run_dir, launch_authorization_path=package["launch"]
    )
    root_claim = spot.build_root_claim(
        run_dir=run_dir,
        launch_authorization_path=package["launch"],
        shard=0,
    )
    _write_json(shard_dir / "global_claim.json", global_claim)
    _write_json(shard_dir / "root_claim.json", root_claim)
    resume_commit.update(
        {
            "heartbeat_sha256": spot.sha256_file(shard_dir / "heartbeat.json"),
            "generator_summary_sha256": spot.sha256_file(
                shard_dir / "generator_summary.json"
            ),
            "run_log_sha256": spot.sha256_file(shard_dir / "run.log"),
            "boot_image_evidence_sha256": spot.sha256_file(
                shard_dir / "boot_image_evidence.json"
            ),
            "global_claim_sha256": spot.sha256_file(
                shard_dir / "global_claim.json"
            ),
            "root_claim_sha256": spot.sha256_file(shard_dir / "root_claim.json"),
        }
    )
    _write_json(shard_dir / "resume_commit.json", resume_commit)
    done_root = received / "done"
    done_root.mkdir()
    for shard, spec in enumerate(schedule):
        done = {
            "schema": spot.DONE_SCHEMA,
            "status": "complete",
            "run_name": manifest["run_name"],
            "run_id": f"{manifest['run_name']}:shard={shard}",
            "shard": shard,
            "root_index": shard,
            "root_profile": spec["root_profile"],
            "seeds": spec["seeds"],
            "output_prefix": spec["output_prefix"],
            "manifest_sha256": manifest_sha,
            "launch_authorization_sha256": launch_sha,
            "development_open_authorization_sha256": manifest[
                "development_open_authorization_sha256"
            ],
            "source_sha256": manifest["source_zip_sha256"],
            "startup_sha256": manifest["startup_sha256"],
            "schedule_sha256": manifest["schedule_sha256"],
            "plan_sha256": manifest["plan_sha256"],
            "model_sha256": manifest["model_sha256"],
            "ai_profiles_sha256": manifest["ai_profiles_sha256"],
            "source_closure_sha256": manifest["source_closure_sha256"],
            "source_model_manifest_sha256": manifest[
                "source_model_manifest_sha256"
            ],
            "source_native_manifest_sha256": manifest[
                "source_native_manifest_sha256"
            ],
            "preflight_aggregate_sha256": manifest["preflight_aggregate_sha256"],
            "preflight_finalization_sha256": manifest[
                "preflight_finalization_sha256"
            ],
            "preflight_proof_sha256": manifest["preflight_proof_sha256"],
            "runtime_semantic_anchor_sha256": manifest[
                "runtime_semantic_anchor_sha256"
            ],
            "preflight_execution_evidence_sha256": manifest[
                "preflight_execution_evidence_sha256"
            ],
            "runtime_source_closure_sha256": manifest[
                "runtime_source_closure_sha256"
            ],
            "runtime_fingerprint_sha256": manifest["runtime_fingerprint_sha256"],
            "runtime_requirements_sha256": manifest["runtime_requirements_sha256"],
            "gcp_image_name": manifest["gcp_image_name"],
            "gcp_image_id": manifest["gcp_image_id"],
            "gcp_image_self_link": manifest["gcp_image_self_link"],
            "global_claim_sha256": (
                spot.sha256_file(shard_dir / "global_claim.json")
                if shard == 0
                else "1" * 64
            ),
            "root_claim_sha256": (
                spot.sha256_file(shard_dir / "root_claim.json")
                if shard == 0
                else "2" * 64
            ),
            "output_sha256": (
                spot.sha256_file(shard_dir / "teacher.jsonl")
                if shard == 0
                else "3" * 64
            ),
            "checkpoint_sha256": (
                spot.sha256_file(shard_dir / "checkpoint.json")
                if shard == 0
                else "4" * 64
            ),
            "heartbeat_sha256": (
                spot.sha256_file(shard_dir / "heartbeat.json")
                if shard == 0
                else "5" * 64
            ),
            "generator_summary_sha256": (
                spot.sha256_file(shard_dir / "generator_summary.json")
                if shard == 0
                else "6" * 64
            ),
            "run_log_sha256": (
                spot.sha256_file(shard_dir / "run.log")
                if shard == 0
                else "7" * 64
            ),
            "boot_image_evidence_sha256": (
                spot.sha256_file(shard_dir / "boot_image_evidence.json")
                if shard == 0
                else "b" * 64
            ),
            "time_report_sha256": (
                spot.sha256_file(shard_dir / "time.txt")
                if shard == 0
                else "9" * 64
            ),
            "resume_commit_sha256": (
                spot.sha256_file(shard_dir / "resume_commit.json")
                if shard == 0
                else "a" * 64
            ),
            "config_sha256": config_sha if shard == 0 else "8" * 64,
            "teacher_generator_elapsed_seconds": generator_elapsed,
            "process_elapsed_seconds": 13.0,
            "peak_rss_bytes": 130000,
            "native_batch_threads": 4,
            "teacher_values_are_realized_match_ev": False,
            "selector_executed": False,
            "future_audit_authorized": False,
            "fit_performed": False,
            "threshold_selected": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        }
        _write_json(done_root / f"DONE-{shard:03d}.json", done)
        if shard == 0:
            _write_json(shard_dir / "DONE.json", done)
    return done_root, summary, bindings


def test_all_done_claim_then_local_remote_boundary_and_received_audit(
    package: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received = tmp_path / "received"
    done_root, summary, _ = _done_payloads(package, received)
    local_claim = tmp_path / "local-claim.json"
    spot.claim_complete_output(
        done_root=done_root,
        run_dir=package["run_dir"],
        launch_authorization_path=package["launch"],
        output=local_claim,
    )
    remote_claim = tmp_path / "remote-claim.json"
    shutil.copyfile(local_claim, remote_claim)
    monkeypatch.setattr(
        spot,
        "_validate_row",
        lambda row, **kwargs: {"config_sha256": "e" * 64},
    )
    audit = spot.audit_received_shard(
        directory=received / "shard_000",
        shard=0,
        done_root=done_root,
        run_dir=package["run_dir"],
        launch_authorization_path=package["launch"],
        consumption_claim_path=local_claim,
        remote_consumption_claim_path=remote_claim,
    )
    assert audit["status"] == "verified_after_local_and_remote_consumption_claims"
    completed = spot.validate_completed_shard_bundle(
        directory=received / "shard_000",
        shard=0,
        run_dir=package["run_dir"],
        launch_authorization_path=package["launch"],
    )
    assert completed["status"] == (
        "single_shard_bundle_fully_reopened_without_consumption_claim"
    )
    assert audit["teacher_generator_elapsed_seconds"] == summary["elapsed_seconds"]
    assert audit["peak_rss_bytes"] == 130000

    remote_claim.write_bytes(remote_claim.read_bytes() + b" ")
    with pytest.raises(ValueError, match="canonical|differ"):
        spot.audit_received_shard(
            directory=received / "shard_000",
            shard=0,
            done_root=done_root,
            run_dir=package["run_dir"],
            launch_authorization_path=package["launch"],
            consumption_claim_path=local_claim,
            remote_consumption_claim_path=remote_claim,
        )


def test_done_set_rejects_missing_marker_without_content_read(
    package: dict[str, Path], tmp_path: Path
) -> None:
    received = tmp_path / "received"
    done_root, _, _ = _done_payloads(package, received)
    (done_root / "DONE-199.json").unlink()
    with pytest.raises(ValueError, match="exact DONE-000..199"):
        spot.validate_complete_done_set(
            done_root=done_root,
            run_dir=package["run_dir"],
            launch_authorization_path=package["launch"],
        )


def test_done_set_rejects_extra_marker_without_content_read(
    package: dict[str, Path], tmp_path: Path
) -> None:
    received = tmp_path / "received"
    done_root, _, _ = _done_payloads(package, received)
    shutil.copyfile(done_root / "DONE-199.json", done_root / "DONE-200.json")
    with pytest.raises(ValueError, match="exact DONE-000..199"):
        spot.validate_complete_done_set(
            done_root=done_root,
            run_dir=package["run_dir"],
            launch_authorization_path=package["launch"],
        )


def test_run_global_selector_claim_blocks_alternate_output_and_double_run(
    package: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received = tmp_path / "received"
    done_root, _, _ = _done_payloads(package, received)
    paths = spot._selector_paths(package["run_dir"])
    paths["merged"].parent.mkdir(parents=True, exist_ok=True)
    paths["merged"].write_bytes(b"placeholder\n")
    local_consumption = tmp_path / "local-consumption.json"
    spot.claim_complete_output(
        done_root=done_root,
        run_dir=package["run_dir"],
        launch_authorization_path=package["launch"],
        output=local_consumption,
    )
    remote_consumption = tmp_path / "remote-consumption.json"
    shutil.copyfile(local_consumption, remote_consumption)
    manifest = spot.validate_package(package["run_dir"])
    merge_receipt = {
        "schema": spot.RECEIVE_MERGE_SCHEMA,
        "status": "complete_without_selection",
        "run_name": manifest["run_name"],
        "manifest_sha256": spot.sha256_file(package["run_dir"] / "manifest.json"),
        "launch_authorization_sha256": spot.sha256_file(package["launch"]),
        "local_consumption_claim_sha256": spot.sha256_file(local_consumption),
        "remote_consumption_claim_sha256": spot.sha256_file(remote_consumption),
        "schedule_sha256": manifest["schedule_sha256"],
        "runtime_semantic_anchor_sha256": manifest["runtime_semantic_anchor_sha256"],
        "preflight_execution_evidence_sha256": manifest[
            "preflight_execution_evidence_sha256"
        ],
        "runtime_source_closure_sha256": manifest["runtime_source_closure_sha256"],
        "runtime_fingerprint_sha256": manifest["runtime_fingerprint_sha256"],
        "runtime_requirements_sha256": manifest["runtime_requirements_sha256"],
        "gcp_image_name": manifest["gcp_image_name"],
        "gcp_image_id": manifest["gcp_image_id"],
        "gcp_image_self_link": manifest["gcp_image_self_link"],
        "roots": 200,
        "root_indices": "0..199",
        "profiles": {profile: 40 for profile in M43_ATTEMPT08_PROFILES},
        "received_audit_sha256": ["a" * 64] * 200,
        "merged_sha256": spot.sha256_file(paths["merged"]),
        "selector_command_required": "python -B -m ofc_regular.select_hu_m43_attempt08_development",
        "selector_executed": False,
        "selector_must_execute_exactly_once": True,
        "future_audit_authorized": False,
        "fit_performed": False,
        "threshold_selected": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    _write_json(paths["merge_receipt"], merge_receipt)
    spot.build_selector_claim(
        run_dir=package["run_dir"],
        launch_authorization_path=package["launch"],
        consumption_claim_path=local_consumption,
        remote_consumption_claim_path=remote_consumption,
        merged_path=paths["merged"],
        merge_receipt_path=paths["merge_receipt"],
        output=paths["claim"],
    )
    paths["remote_claim"].parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(paths["claim"], paths["remote_claim"])
    monkeypatch.setattr(
        spot,
        "select_attempt08_development",
        lambda **kwargs: {
            "decision": "go",
            "search_freeze_authorized": True,
            "decision_contract": {"gate_evaluation_count": 1},
            "source": {
                "run_name": package["run_dir"].name,
                "input_jsonl_sha256": spot.sha256_file(paths["merged"]),
                "selector_source_sha256": spot.sha256_file(
                    Path(spot.__file__).with_name(
                        "select_hu_m43_attempt08_development.py"
                    )
                ),
            },
            "science_boundary": {
                "current_profile_mutated": False,
                "runtime_policy_activated": False,
            },
        },
    )
    result = spot.execute_selector_once(
        run_dir=package["run_dir"],
        selector_claim_path=paths["claim"],
        remote_selector_claim_path=paths["remote_claim"],
        output=paths["decision"],
        receipt=paths["receipt"],
    )
    assert result["gate_evaluation_count"] == 1
    validated_completion = spot.validate_selector_completion(
        run_dir=package["run_dir"]
    )
    assert validated_completion["gate_evaluation_count"] == 1
    with pytest.raises((FileExistsError, ValueError)):
        spot.execute_selector_once(
            run_dir=package["run_dir"],
            selector_claim_path=paths["claim"],
            remote_selector_claim_path=paths["remote_claim"],
            output=paths["decision"],
            receipt=paths["receipt"],
        )
    with pytest.raises(ValueError, match="canonical path"):
        spot.execute_selector_once(
            run_dir=package["run_dir"],
            selector_claim_path=paths["claim"],
            remote_selector_claim_path=paths["remote_claim"],
            output=paths["decision"].with_name("decision-alt.json"),
            receipt=paths["receipt"],
        )
