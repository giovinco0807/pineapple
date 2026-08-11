from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import ofc_regular.finalize_hu_m43_attempt07_preflight as finalizer
from ofc_regular.aggregate_hu_m43_attempt07_preflight import (
    ATTEMPT07_PREFLIGHT_AGGREGATE_SCHEMA,
)
from ofc_regular.hu_m43_attempt06_teacher import ATTEMPT06_FROZEN_MODEL_SHA256
from ofc_regular.hu_m43_attempt06_teacher import (
    ATTEMPT06_SHARD_ROW_SCHEMA,
    ATTEMPT06_TEACHER_SCHEMA,
)
from ofc_regular.hu_m43_attempt06_spot import (
    PINNED_MODEL_MANIFEST_SHA256,
    PINNED_NATIVE_MANIFEST_SHA256,
)
from ofc_regular.hu_m43_attempt07_contract import (
    AI_PROFILES_SHA256,
    M43_ATTEMPT07_PLAN_SHA256,
    load_and_validate_attempt07_plan,
)
from ofc_regular.hu_m43_attempt07_preflight_spot import (
    ATTEMPT06_BASE_MANIFEST_SHA256,
    ATTEMPT07_MACHINE_TYPE,
    DONE_SCHEMA,
    LAUNCH_AUTHORIZATION_SCHEMA,
    PACKAGE_MANIFEST_SCHEMA,
    build_preflight_schedule,
)
from ofc_regular.hu_m43_attempt07_teacher import ATTEMPT07_TEACHER_SCHEMA
from ofc_regular.hu_m43_attempt07_spot import (
    PACKAGE_MANIFEST_SCHEMA as DEVELOPMENT_PACKAGE_MANIFEST_SCHEMA,
    build_attempt07_spot_schedule,
)
from ofc_regular.run_hu_m43_attempt07_preflight import (
    ATTEMPT06_SOURCE_PLAN_SHA256,
    ATTEMPT07_PREFLIGHT_PLAN_SCHEMA,
    ATTEMPT07_PREFLIGHT_SOURCE_SHA256,
    DEFAULT_ATTEMPT07_PLAN_PATH,
    DEFAULT_PREFLIGHT_PLAN_PATH,
    canonical_json_bytes,
)


SLOTS = (
    "root0_batch_a",
    "root0_batch_b",
    "root0_scalar",
    "root1_batch",
    "root2_batch",
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def _fixture(tmp_path: Path) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    schedule_rows = build_preflight_schedule()
    schedule = tmp_path / "schedule.jsonl"
    schedule.write_bytes(b"".join(canonical_json_bytes(row) for row in schedule_rows))
    schedule_sha = hashlib.sha256(schedule.read_bytes()).hexdigest()
    preflight_plan_sha = hashlib.sha256(DEFAULT_PREFLIGHT_PLAN_PATH.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest_payload = {
        "schema": PACKAGE_MANIFEST_SCHEMA,
        "status": "packaged_attempt06_copy_plus_overlay_without_execution",
        "run_name": "attempt07-preflight-test",
        "jobs": 5,
        "source_roots": [0, 1, 2],
        "machine_type": ATTEMPT07_MACHINE_TYPE,
        "native_batch_threads": 4,
        "base_attempt06_manifest_sha256": ATTEMPT06_BASE_MANIFEST_SHA256,
        "base_attempt06_package_tree_sha256": "5" * 64,
        "base_attempt06_source_zip_sha256": "6" * 64,
        "package_tree_sha256": "7" * 64,
        "overlay_closure_sha256": "8" * 64,
        "source_zip_sha256": "9" * 64,
        "source_zip_bytes": 12345,
        "startup_sha256": "a" * 64,
        "attempt07_plan_sha256": M43_ATTEMPT07_PLAN_SHA256,
        "preflight_plan_sha256": preflight_plan_sha,
        "schedule_sha256": schedule_sha,
        "source_merged_sha256": ATTEMPT07_PREFLIGHT_SOURCE_SHA256,
        "model_sha256": ATTEMPT06_FROZEN_MODEL_SHA256,
        "ai_profiles_sha256": AI_PROFILES_SHA256,
        "new_root_generated": False,
        "teacher_executed": False,
        "gcloud_invoked": False,
        "instances_created": False,
        "arm_selection_performed": False,
        "current_profile_resolved": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    _write(manifest, manifest_payload)
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    authorization = tmp_path / "preflight-launch-authorization.json"
    _write(
        authorization,
        {
            "schema": LAUNCH_AUTHORIZATION_SCHEMA,
            "status": "authorized_for_bounded_spot_preflight",
            "run_name": manifest_payload["run_name"],
            "manifest_sha256": manifest_sha,
            "preflight_plan_sha256": preflight_plan_sha,
            "attempt07_plan_sha256": M43_ATTEMPT07_PLAN_SHA256,
            "source_merged_sha256": ATTEMPT07_PREFLIGHT_SOURCE_SHA256,
            "model_sha256": ATTEMPT06_FROZEN_MODEL_SHA256,
            "ai_profiles_sha256": AI_PROFILES_SHA256,
            "machine_type": ATTEMPT07_MACHINE_TYPE,
            "native_batch_threads": 4,
            "jobs": 5,
            "source_roots": [0, 1, 2],
            "local_gates": {
                "correctness_smoke": "pass",
                "determinism": "pass",
                "scalar_batch_parity_test_harness": "pass",
                "package_closure": "pass",
            },
            "local_evidence": {
                "attempt07_pytest": {
                    "receipt_sha256": "b" * 64,
                    "passed": 99,
                    "failed": 0,
                },
                "rust_parity": {
                    "receipt_sha256": "c" * 64,
                    "passed": 9,
                    "failed": 0,
                },
                "package_tests": {
                    "receipt_sha256": "d" * 64,
                    "passed": 16,
                    "failed": 0,
                },
            },
            "actual_scalar_batch_result": (
                "pending_spot_preflight_receive_and_aggregate"
            ),
            "actual_operational_go_no_go": (
                "pending_spot_preflight_receive_and_aggregate"
            ),
            "spot_authorized": True,
            "new_root_generation_allowed": False,
            "arm_selection_allowed": False,
            "current_profile_resolved": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        },
    )
    authorization_sha = hashlib.sha256(authorization.read_bytes()).hexdigest()

    output_hashes = {
        "root0_batch_a": "a" * 64,
        "root0_batch_b": "a" * 64,
        "root0_scalar": "b" * 64,
        "root1_batch": "c" * 64,
        "root2_batch": "d" * 64,
    }
    times = {
        "root0_batch_a": 100.0,
        "root0_batch_b": 110.0,
        "root0_scalar": 300.0,
        "root1_batch": 120.0,
        "root2_batch": 130.0,
    }
    done_paths: dict[str, Path] = {}
    operational_jobs: dict[str, dict] = {}
    for spec, slot in zip(schedule_rows, SLOTS, strict=True):
        done = {
            "schema": DONE_SCHEMA,
            "status": "complete",
            "run_name": manifest_payload["run_name"],
            "job_index": spec["job_index"],
            "job_id": spec["job_id"],
            "source_root_index": spec["source_root_index"],
            "batch_child_selectors": spec["batch_child_selectors"],
            "native_batch_threads": 4,
            "output_prefix": spec["output_prefix"],
            "output_sha256": output_hashes[slot],
            "checkpoint_sha256": "1" * 64,
            "heartbeat_sha256": "2" * 64,
            "summary_sha256": "3" * 64,
            "run_log_sha256": "4" * 64,
            "manifest_sha256": manifest_sha,
            "authorization_sha256": authorization_sha,
            "schedule_sha256": schedule_sha,
            "attempt07_plan_sha256": M43_ATTEMPT07_PLAN_SHA256,
            "source_merged_sha256": ATTEMPT07_PREFLIGHT_SOURCE_SHA256,
            "model_sha256": ATTEMPT06_FROZEN_MODEL_SHA256,
            "ai_profiles_sha256": AI_PROFILES_SHA256,
            "elapsed_seconds": times[slot],
            "peak_rss_bytes": 1_000_000_000,
            "teacher_values_exported": False,
            "arm_selection_performed": False,
            "new_root_generated": False,
            "current_profile_resolved": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        }
        path = tmp_path / f"{slot}.DONE.json"
        _write(path, done)
        done_paths[slot] = path
        operational_jobs[slot] = {
            "elapsed_seconds": times[slot],
            "peak_rss_bytes": 1_000_000_000,
        }
    aggregate = tmp_path / "aggregate.json"
    _write(
        aggregate,
        {
            "schema": ATTEMPT07_PREFLIGHT_AGGREGATE_SCHEMA,
            "status": "complete",
            "decision": "go",
            "reasons": ["all_preflight_proof_gates_passed"],
            "proof_input_count": 5,
            "valid_proof_count": 5,
            "expected_slots": list(SLOTS),
            "root_coverage": [0, 1, 2],
            "proof_gates": {
                "frozen_context_valid": True,
                "five_proof_artifact_paths_distinct": True,
                "all_five_canonical_proofs_valid": True,
                "exact_root_and_mode_coverage": True,
                "root0_batch_a_b_canonical_rows_identical": True,
                "root0_batch_opaque_mode_determinism": True,
                "root0_scalar_batch_semantic_parity": True,
                "three_source_roots_covered": True,
                "teacher_values_absent": True,
                "arm_details_absent": True,
                "current_profile_unchanged": True,
            },
            "proof_file_sha256": output_hashes,
            "cross_mode_opaque_teacher_hash_compared": False,
            "science_boundary": {
                "proof_only_no_teacher_values_or_arm_details": True,
                "arm_selection_allowed": False,
                "fit_allowed": False,
                "threshold_selection_allowed": False,
                "runtime_activation_allowed": False,
                "current_profile_resolved": False,
                "current_profile_mutated": False,
                "fresh_seed_or_root_opened": False,
                "done_metadata_is_science_input": False,
            },
            "contract": {
                "preflight_plan_schema": ATTEMPT07_PREFLIGHT_PLAN_SCHEMA,
                "preflight_plan_sha256": preflight_plan_sha,
                "attempt06_source_plan_sha256": ATTEMPT06_SOURCE_PLAN_SHA256,
                "attempt06_merged_source_sha256": ATTEMPT07_PREFLIGHT_SOURCE_SHA256,
                "attempt06_wrapper_schema": ATTEMPT06_SHARD_ROW_SCHEMA,
                "attempt06_teacher_schema": ATTEMPT06_TEACHER_SCHEMA,
                "attempt07_plan_sha256": M43_ATTEMPT07_PLAN_SHA256,
                "attempt07_teacher_schema": ATTEMPT07_TEACHER_SCHEMA,
                "model_sha256": ATTEMPT06_FROZEN_MODEL_SHA256,
                "ai_profiles_sha256": AI_PROFILES_SHA256,
            },
            "operational_diagnostics": {
                "status": "ok",
                "science_decision_input": False,
                "allowed_fields": ["elapsed_seconds", "peak_rss_bytes"],
                "job_count": 5,
                "invalid_labels": [],
                "jobs": operational_jobs,
                "elapsed_seconds_sum": sum(times.values()),
                "elapsed_seconds_max": max(times.values()),
                "peak_rss_bytes_max": 1_000_000_000,
            },
        },
    )
    development_schedule_rows = build_attempt07_spot_schedule(
        load_and_validate_attempt07_plan(DEFAULT_ATTEMPT07_PLAN_PATH)
    )
    development_schedule = tmp_path / "development-schedule.jsonl"
    development_schedule.write_bytes(
        b"".join(canonical_json_bytes(row) for row in development_schedule_rows)
    )
    development_manifest = tmp_path / "development-manifest.json"
    _write(
        development_manifest,
        {
            "schema": DEVELOPMENT_PACKAGE_MANIFEST_SCHEMA,
            "status": "frozen_package_only_no_root_opened",
            "run_name": "attempt07-development-test",
            "plan_sha256": M43_ATTEMPT07_PLAN_SHA256,
            "status_sha256": "e" * 64,
            "model_sha256": ATTEMPT06_FROZEN_MODEL_SHA256,
            "ai_profiles_sha256": AI_PROFILES_SHA256,
            "preflight_aggregate_sha256": hashlib.sha256(
                aggregate.read_bytes()
            ).hexdigest(),
            "preflight_plan_sha256": preflight_plan_sha,
            "preflight_manifest_sha256": manifest_sha,
            "preflight_schedule_sha256": schedule_sha,
            "preflight_launch_authorization_sha256": authorization_sha,
            "preflight_done_sha256": {
                slot: hashlib.sha256(path.read_bytes()).hexdigest()
                for slot, path in done_paths.items()
            },
            "schedule_sha256": hashlib.sha256(
                development_schedule.read_bytes()
            ).hexdigest(),
            "source_closure_sha256": "f" * 64,
            "source_zip_sha256": "0" * 64,
            "startup_sha256": "1" * 64,
            "source_model_manifest_sha256": PINNED_MODEL_MANIFEST_SHA256,
            "source_native_manifest_sha256": PINNED_NATIVE_MANIFEST_SHA256,
            "total_roots": 100,
            "total_shards": 100,
            "roots_per_shard": 1,
            "root_profile_assignment": "root_index_mod_5_in_frozen_profile_order",
            "batch_child_selectors": True,
            "native_batch_threads": 4,
            "recommended_machine_type": "c4-standard-4",
            "recommended_wave_shards": 50,
            "fresh_root_opened": False,
            "teacher_executed": False,
            "gcloud_invoked": False,
            "instances_created": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        },
    )
    return {
        "aggregate": aggregate,
        "manifest": manifest,
        "schedule": schedule,
        "authorization": authorization,
        "done": done_paths,
        "development_manifest": development_manifest,
        "development_schedule": development_schedule,
    }


def _run(tmp_path: Path, fixture: dict, *, output: Path | None = None) -> dict:
    return finalizer.finalize_attempt07_preflight(
        proof_aggregate=fixture["aggregate"],
        preflight_manifest=fixture["manifest"],
        preflight_schedule=fixture["schedule"],
        preflight_launch_authorization=fixture["authorization"],
        development_manifest=fixture["development_manifest"],
        development_schedule=fixture["development_schedule"],
        done_paths=fixture["done"],
        output=output or tmp_path / "authorization.json",
    )


def _refresh_development_preflight_bindings(fixture: dict) -> None:
    manifest = json.loads(fixture["development_manifest"].read_bytes())
    manifest["preflight_aggregate_sha256"] = hashlib.sha256(
        fixture["aggregate"].read_bytes()
    ).hexdigest()
    manifest["preflight_done_sha256"] = {
        slot: hashlib.sha256(path.read_bytes()).hexdigest()
        for slot, path in fixture["done"].items()
    }
    _write(fixture["development_manifest"], manifest)


def test_passing_proofs_and_frozen_operational_gates_authorize(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    report = _run(tmp_path, fixture)
    assert report["schema"] == finalizer.AUTHORIZATION_SCHEMA
    assert report["status"] == finalizer.AUTHORIZED_STATUS
    assert report["spot_authorized"] is True
    assert report["all_gates_passed"] is True
    assert all(gate["passed"] for gate in report["operational_gates"].values())
    assert report["operational_metrics"][
        "scalar_to_root0_batch_median_speedup"
    ] == pytest.approx(300.0 / 105.0)
    assert report["development_started"] is False
    assert report["development_run_name"] == "attempt07-development-test"
    assert report["development_manifest_sha256"] == hashlib.sha256(
        fixture["development_manifest"].read_bytes()
    ).hexdigest()
    assert report["development_total_roots"] == 100
    assert report["development_package_frozen_before_authorization"] is True
    assert report["fresh_development_root_opened"] is False
    assert report["preflight_launch_authorization_sha256"] == hashlib.sha256(
        fixture["authorization"].read_bytes()
    ).hexdigest()
    assert report["current_profile_mutated"] is False
    assert (tmp_path / "authorization.json").read_bytes() == canonical_json_bytes(report)


def test_fixed_speed_failure_writes_no_go_without_reselection(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    scalar = json.loads(fixture["done"]["root0_scalar"].read_bytes())
    scalar["elapsed_seconds"] = 100.0
    _write(fixture["done"]["root0_scalar"], scalar)
    aggregate = json.loads(fixture["aggregate"].read_bytes())
    aggregate["operational_diagnostics"]["jobs"]["root0_scalar"][
        "elapsed_seconds"
    ] = 100.0
    aggregate["operational_diagnostics"]["elapsed_seconds_sum"] = 560.0
    aggregate["operational_diagnostics"]["elapsed_seconds_max"] = 130.0
    _write(fixture["aggregate"], aggregate)
    _refresh_development_preflight_bindings(fixture)
    report = _run(tmp_path, fixture)
    assert report["status"] == finalizer.NOT_AUTHORIZED_STATUS
    assert report["spot_authorized"] is False
    assert report["operational_gates"][
        "scalar_to_root0_batch_median_speedup"
    ]["passed"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda done: done.__setitem__("current_profile_mutated", True),
        lambda done: done.__setitem__("batch_child_selectors", 1),
        lambda done: done.__setitem__("elapsed_seconds", 0.0),
        lambda done: done.__setitem__("unexpected", "value"),
    ],
)
def test_done_identity_and_types_fail_closed(tmp_path: Path, mutation) -> None:
    fixture = _fixture(tmp_path)
    path = fixture["done"]["root1_batch"]
    done = json.loads(path.read_bytes())
    mutation(done)
    _write(path, done)
    with pytest.raises(ValueError):
        _run(tmp_path, fixture)


def test_aggregate_done_proof_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    aggregate = json.loads(fixture["aggregate"].read_bytes())
    aggregate["proof_file_sha256"]["root2_batch"] = "e" * 64
    _write(fixture["aggregate"], aggregate)
    with pytest.raises(ValueError, match="proof mismatch"):
        _run(tmp_path, fixture)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda aggregate: aggregate.__setitem__("proof_gates", {"all": True}),
        lambda aggregate: aggregate["science_boundary"].pop("fit_allowed"),
        lambda aggregate: aggregate["contract"].pop("attempt06_teacher_schema"),
        lambda aggregate: aggregate.__setitem__("valid_proof_count", 4),
        lambda aggregate: aggregate.__setitem__("reasons", []),
        lambda aggregate: aggregate["operational_diagnostics"].__setitem__(
            "elapsed_seconds_sum", 1.0
        ),
    ],
)
def test_truncated_or_rewritten_go_aggregate_fails_closed(
    tmp_path: Path, mutation
) -> None:
    fixture = _fixture(tmp_path)
    aggregate = json.loads(fixture["aggregate"].read_bytes())
    mutation(aggregate)
    _write(fixture["aggregate"], aggregate)
    with pytest.raises(ValueError):
        _run(tmp_path, fixture)


def test_manifest_authorization_and_schedule_require_exact_canonical_shape(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path / "manifest")
    manifest = json.loads(fixture["manifest"].read_bytes())
    manifest["unexpected"] = False
    _write(fixture["manifest"], manifest)
    with pytest.raises(ValueError, match="manifest changed"):
        _run(tmp_path / "manifest", fixture)

    fixture = _fixture(tmp_path / "authorization")
    authorization = json.loads(fixture["authorization"].read_bytes())
    authorization["unexpected"] = False
    _write(fixture["authorization"], authorization)
    with pytest.raises(ValueError, match="authorization fields"):
        _run(tmp_path / "authorization", fixture)

    fixture = _fixture(tmp_path / "schedule")
    fixture["schedule"].write_bytes(fixture["schedule"].read_bytes() + b"\n")
    with pytest.raises(ValueError, match="schedule changed"):
        _run(tmp_path / "schedule", fixture)


def test_authorization_is_bound_to_exact_development_package(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "manifest")
    manifest = json.loads(fixture["development_manifest"].read_bytes())
    manifest["source_zip_sha256"] = "2" * 64
    _write(fixture["development_manifest"], manifest)
    report = _run(tmp_path / "manifest", fixture)
    assert report["development_source_zip_sha256"] == "2" * 64
    assert report["development_manifest_sha256"] == hashlib.sha256(
        fixture["development_manifest"].read_bytes()
    ).hexdigest()

    fixture = _fixture(tmp_path / "schedule")
    rows = fixture["development_schedule"].read_bytes().splitlines(keepends=True)
    rows[0] = rows[0] + b"\n"
    fixture["development_schedule"].write_bytes(b"".join(rows))
    with pytest.raises(ValueError, match="development schedule changed"):
        _run(tmp_path / "schedule", fixture)

    fixture = _fixture(tmp_path / "shape")
    manifest = json.loads(fixture["development_manifest"].read_bytes())
    manifest["unexpected"] = False
    _write(fixture["development_manifest"], manifest)
    with pytest.raises(ValueError, match="development package manifest changed"):
        _run(tmp_path / "shape", fixture)


def test_output_is_no_clobber_and_may_not_alias_an_input(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "authorization.json"
    output.write_text("owned\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _run(tmp_path, fixture, output=output)
    assert output.read_text(encoding="utf-8") == "owned\n"
    with pytest.raises(ValueError, match="distinct"):
        _run(tmp_path, fixture, output=fixture["aggregate"])


def test_inputs_must_be_canonical_and_done_files_physically_distinct(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["aggregate"].write_bytes(fixture["aggregate"].read_bytes() + b"\n")
    with pytest.raises(ValueError, match="canonical"):
        _run(tmp_path, fixture)

    fixture = _fixture(tmp_path / "same")
    fixture["done"]["root0_batch_b"] = fixture["done"]["root0_batch_a"]
    with pytest.raises(ValueError, match="distinct"):
        _run(tmp_path / "same", fixture)
