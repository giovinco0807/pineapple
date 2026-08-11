from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import ofc_regular.finalize_hu_m43_attempt08_preflight as finalizer
import ofc_regular.run_hu_m43_attempt08_preflight as preflight
import ofc_regular.aggregate_hu_m43_attempt08_preflight as aggregate_module
import ofc_regular.hu_m43_attempt08_preflight_spot as preflight_spot
from ofc_regular.aggregate_hu_m43_attempt08_preflight import (
    ATTEMPT08_PREFLIGHT_AGGREGATE_SCHEMA,
    ATTEMPT08_PREFLIGHT_AGGREGATE_STATUS,
)
from ofc_regular.hu_m43_attempt08_contract import (
    AI_PROFILES_SHA256,
    ATTEMPT08_LAMBDA_MODEL_SHA256,
    M43_ATTEMPT08_PLAN_SCHEMA,
    M43_ATTEMPT08_PLAN_SHA256,
    M43_ATTEMPT08_PROFILES,
)
from ofc_regular.hu_m43_attempt06_teacher import (
    ATTEMPT06_SHARD_ROW_SCHEMA,
    ATTEMPT06_TEACHER_SCHEMA,
)
from ofc_regular.hu_m43_attempt08_teacher import (
    ATTEMPT08_SOLVER_ID,
    ATTEMPT08_TEACHER_SCHEMA,
)
from ofc_regular.hu_m43_attempt08_runtime_anchor_contract import (
    ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256,
    ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256,
)
from ofc_regular.hu_m43_attempt08_runtime_anchor import (
    ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
)
from ofc_regular.hu_m43_attempt08_runtime_identity import (
    ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
    ATTEMPT08_GCP_IMAGE_ID,
    ATTEMPT08_GCP_IMAGE_NAME,
)


@pytest.fixture(autouse=True)
def _stub_actual_bundle_revalidation(monkeypatch):
    """Keep finalizer unit tests focused; lifecycle tests exercise real bytes."""

    def validate(*, jobs_root, execution_evidence_path, **_kwargs):
        payload = json.loads(Path(execution_evidence_path).read_bytes())
        return {
            "execution_evidence": payload,
            "proof_paths": {
                slot: Path(jobs_root).parent / f"{slot}.json"
                for slot in preflight.ATTEMPT08_PREFLIGHT_SLOTS
            },
            "done_rows": {},
        }

    monkeypatch.setattr(
        preflight_spot, "validate_preflight_receive_bundle", validate
    )


def _execution_evidence(tmp_path: Path) -> Path:
    path = tmp_path / "preflight_execution_evidence.json"
    slots = tuple(preflight.ATTEMPT08_PREFLIGHT_SLOTS)
    payload = {
        "schema": finalizer.ATTEMPT08_PREFLIGHT_EXECUTION_EVIDENCE_SCHEMA,
        "status": "all_five_spot_jobs_received_and_validated",
        "run_name": "attempt08-test-run",
        "manifest_sha256": "1" * 64,
        "launch_authorization_sha256": "2" * 64,
        "local_evidence_sha256": "3" * 64,
        "done_sha256": {slot: "4" * 64 for slot in slots},
        "support_sha256": {
            slot: {
                "proof": "5" * 64,
                "checkpoint": "6" * 64,
                "heartbeat": "7" * 64,
                "summary": "8" * 64,
                "run_log": "9" * 64,
                "boot_image": "a" * 64,
            }
            for slot in slots
        },
        "runtime": {
            "runtime_semantic_anchor_sha256": ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256,
            "runtime_source_closure_sha256": ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256,
            "runtime_fingerprint_sha256": ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
            "runtime_requirements_sha256": ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
            "gcp_image_name": ATTEMPT08_GCP_IMAGE_NAME,
            "gcp_image_id": ATTEMPT08_GCP_IMAGE_ID,
        },
        "all_five_done_before_payloads_opened": True,
        "proof_payloads_opened_after_all_done": True,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    _write(path, payload)
    return path


def _spot_kwargs(tmp_path: Path) -> dict:
    return {
        "execution_evidence": tmp_path / "preflight_execution_evidence.json",
        "spot_run_dir": tmp_path / "spot-run",
        "spot_authorization": tmp_path / "spot-authorization.json",
        "spot_local_evidence": tmp_path / "local-evidence.json",
        "spot_jobs_root": tmp_path / "spot-jobs",
    }


def _aggregate(*, go: bool = True) -> dict:
    proof_hashes = {
        slot: character * 64
        for slot, character in zip(
            preflight.ATTEMPT08_PREFLIGHT_SLOTS,
            ("a", "b", "c", "d", "e"),
            strict=True,
        )
    }
    proof_gates = {
        "five_proof_artifact_paths_distinct": True,
        "all_five_canonical_proofs_valid": True,
        "source_root_coverage_exact_0_1_2": True,
        "root0_batch_a_b_exact_teacher_determinism": True,
        "root0_scalar_batch_semantic_parity": True,
        "exact_actionkey_reference_parity_each_run": True,
        "hidden_information_safety_each_run": True,
        "rng_domain_separation_each_run": True,
        "conditional_X_A_skip_contract_each_run": True,
        "runtime_fingerprint_identical_all_five": True,
    }
    if not go:
        proof_gates["root0_scalar_batch_semantic_parity"] = False
    operational_gates = {
        "teacher_elapsed_seconds_each_run_max_2400": True,
        "process_peak_rss_bytes_each_run_max_28GiB": True,
        "spot_operational_evidence_bound": True,
    }
    return {
        "schema": ATTEMPT08_PREFLIGHT_AGGREGATE_SCHEMA,
        "status": ATTEMPT08_PREFLIGHT_AGGREGATE_STATUS,
        "decision": "go" if go else "no_go",
        "reasons": (
            ["all_correctness_and_operational_preflight_gates_passed"]
            if go
            else ["gate_failed:root0_scalar_batch_semantic_parity"]
        ),
        "contract": {
            "attempt08_plan_schema": M43_ATTEMPT08_PLAN_SCHEMA,
            "attempt08_plan_sha256": M43_ATTEMPT08_PLAN_SHA256,
            "preflight_plan_schema": preflight.ATTEMPT08_PREFLIGHT_PLAN_SCHEMA,
            "preflight_plan_sha256": preflight.ATTEMPT08_PREFLIGHT_PLAN_SHA256,
            "teacher_schema": ATTEMPT08_TEACHER_SCHEMA,
            "model_sha256": ATTEMPT08_LAMBDA_MODEL_SHA256,
            "ai_profiles_sha256": AI_PROFILES_SHA256,
            "runtime_semantic_anchor_sha256": (
                ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256
            ),
            "runtime_source_closure_sha256": ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256,
            "runtime_requirements_sha256": ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
            "gcp_image_name": ATTEMPT08_GCP_IMAGE_NAME,
            "gcp_image_id": ATTEMPT08_GCP_IMAGE_ID,
        },
        "valid_proof_count": 5,
        "root_coverage": [0, 1, 2],
        "proof_file_sha256": proof_hashes,
        "proof_evidence_sha256": preflight._sha256_value(proof_hashes),
        "runtime_fingerprint_sha256": ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
        "spot_operational_evidence_sha256": "9" * 64,
        "proof_gates": proof_gates,
        "operational_gates": operational_gates,
        "operational_diagnostics": {
            "science_decision_input": False,
            "jobs": {
                "root0_batch_a": {
                    "teacher_elapsed_seconds": 100.0,
                    "process_peak_rss_bytes": 1000,
                },
                "root0_batch_b": {
                    "teacher_elapsed_seconds": 110.0,
                    "process_peak_rss_bytes": 1100,
                },
                "root0_scalar": {
                    "teacher_elapsed_seconds": 210.0,
                    "process_peak_rss_bytes": 1200,
                },
                "root1_batch": {
                    "teacher_elapsed_seconds": 90.0,
                    "process_peak_rss_bytes": 1300,
                },
                "root2_batch": {
                    "teacher_elapsed_seconds": 95.0,
                    "process_peak_rss_bytes": 1400,
                },
            },
            "teacher_elapsed_seconds_max": 210.0,
            "process_peak_rss_bytes_max": 1400,
            "root0_scalar_to_batch_median_speedup_diagnostic": 2.0,
        },
        "science_boundary": {
            "proof_only_not_policy_science": True,
            "opaque_teacher_hashes_exported": False,
            "teacher_action_or_value_details_exported": False,
            "development200_authorized": False,
            "future_audit_authorized": False,
            "fit_allowed": False,
            "threshold_selection_allowed": False,
            "runtime_activation_allowed": False,
            "current_profile_changed": False,
        },
    }


def _write(path: Path, payload: dict) -> None:
    path.write_bytes(preflight.canonical_json_bytes(payload))


def _proof(slot: str, *, opaque: str, semantic: str, elapsed: float) -> dict:
    root, batch = preflight.ATTEMPT08_PREFLIGHT_SLOTS[slot]
    source_row, observation = preflight.load_source_row(
        preflight.DEFAULT_SOURCE_PATH, root
    )
    seeds = preflight.attempt08_preflight_seeds(root)
    return {
        "schema": preflight.ATTEMPT08_PREFLIGHT_PROOF_SCHEMA,
        "status": preflight.ATTEMPT08_PREFLIGHT_PROOF_STATUS,
        "slot": slot,
        "source": {
            "merged_sha256": preflight.ATTEMPT07_PREFLIGHT_SOURCE_SHA256,
            "source_root_index": root,
            "source_row_sha256": preflight._sha256_value(source_row),
            "wrapper_schema": ATTEMPT06_SHARD_ROW_SCHEMA,
            "teacher_schema": ATTEMPT06_TEACHER_SCHEMA,
            "observation_fingerprint": observation.fingerprint(),
            "source_already_consumed": True,
            "new_root_generated": False,
        },
        "contract": {
            "attempt08_plan_schema": M43_ATTEMPT08_PLAN_SCHEMA,
            "attempt08_plan_sha256": M43_ATTEMPT08_PLAN_SHA256,
            "preflight_plan_schema": preflight.ATTEMPT08_PREFLIGHT_PLAN_SCHEMA,
            "preflight_plan_sha256": preflight.ATTEMPT08_PREFLIGHT_PLAN_SHA256,
            "model_sha256": ATTEMPT08_LAMBDA_MODEL_SHA256,
            "ai_profiles_sha256": AI_PROFILES_SHA256,
            "runtime_semantic_anchor_sha256": (
                ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256
            ),
            "runtime_source_closure_sha256": ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256,
            "runtime_requirements_sha256": ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
            "gcp_image_name": ATTEMPT08_GCP_IMAGE_NAME,
            "gcp_image_id": ATTEMPT08_GCP_IMAGE_ID,
            "t2_policy_id": "stage9f_p2",
            "t2_resolution": "explicit_profile_never_current",
            "seeds": seeds,
            "continuation_policy_seeds": {
                "first": seeds["child"],
                "second": seeds["child"] + 1,
            },
        },
        "execution": {
            "batch_child_selectors": batch,
            "native_batch_threads": 4,
            "run_id": (
                f"attempt08-preflight:source-root={root}:"
                f"obs={observation.fingerprint()}"
            ),
            "teacher_elapsed_seconds": elapsed,
            "process_peak_rss_bytes": 1_000_000 + root,
            "runtime_fingerprint_sha256": ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
            "measurement_scope": (
                "elapsed_is_teacher_call_only_rss_is_one_root_process_high_water"
            ),
        },
        "result_proof": {
            "teacher_schema": ATTEMPT08_TEACHER_SCHEMA,
            "solver_id": ATTEMPT08_SOLVER_ID,
            "opaque_teacher_sha256": opaque,
            "semantic_parity_sha256": semantic,
            "exact_actionkey_reference_parity_verified": True,
            "hidden_information_safety_verified": True,
            "rng_domain_separation_verified": True,
            "conditional_X_A_skip_contract_verified": True,
            "teacher_action_or_value_details_exported": False,
        },
        "science_boundary": {
            "proof_only_not_policy_science": True,
            "development200_authorized": False,
            "future_audit_authorized": False,
            "fit_allowed": False,
            "threshold_selection_allowed": False,
            "runtime_activation_allowed": False,
            "current_profile_resolved": False,
            "current_profile_mutated": False,
            "fresh_seed_or_root_opened": False,
        },
    }


def _go_bundle(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    proof_paths: dict[str, Path] = {}
    for index, slot in enumerate(preflight.ATTEMPT08_PREFLIGHT_SLOTS):
        path = tmp_path / f"{slot}.json"
        opaque = "a" * 64 if slot.startswith("root0_batch") else f"{index + 1:x}" * 64
        semantic = "b" * 64 if slot.startswith("root0") else f"{index + 6:x}" * 64
        _write(path, _proof(slot, opaque=opaque, semantic=semantic, elapsed=10.0 + index))
        proof_paths[slot] = path
    execution_evidence = _execution_evidence(tmp_path)
    aggregate = tmp_path / "aggregate.json"
    aggregate_module.aggregate_preflight_proofs(
        **proof_paths,
        output=aggregate,
        spot_operational_evidence_sha256=hashlib.sha256(
            execution_evidence.read_bytes()
        ).hexdigest(),
    )
    return aggregate, proof_paths


def test_go_emits_exact_immutable_development200_authorization(tmp_path: Path) -> None:
    aggregate, proof_paths = _go_bundle(tmp_path)
    report_path = tmp_path / "finalization.json"
    auth_path = tmp_path / "development-open-authorization.json"
    report = finalizer.finalize_preflight(
        aggregate=aggregate,
        output=report_path,
        authorization_output=auth_path,
        proof_paths=proof_paths,
        **_spot_kwargs(tmp_path),
    )
    assert report["status"] == finalizer.ATTEMPT08_PREFLIGHT_GO_STATUS
    assert report["authorization"]["emitted"] is True
    assert report["authorization"]["sha256"] == hashlib.sha256(
        auth_path.read_bytes()
    ).hexdigest()
    authorization = finalizer.load_and_validate_development_open_authorization(
        auth_path
    )
    assert authorization["schema"] == (
        finalizer.ATTEMPT08_DEVELOPMENT_OPEN_AUTHORIZATION_SCHEMA
    )
    assert authorization["target_plan"]["sha256"] == M43_ATTEMPT08_PLAN_SHA256
    assert authorization["preflight_plan"]["sha256"] == (
        preflight.ATTEMPT08_PREFLIGHT_PLAN_SHA256
    )
    assert authorization["preflight_result"]["sha256"] == hashlib.sha256(
        aggregate.read_bytes()
    ).hexdigest()
    population = authorization["development_population"]
    assert (
        population["roots"],
        population["root_index_first"],
        population["root_index_last"],
    ) == (
        200,
        0,
        199,
    )
    assert population["profiles"] == list(M43_ATTEMPT08_PROFILES)
    guards = authorization["authorization"]
    assert guards["development_generation_authorized"] is True
    assert guards["future_audit_authorized"] is False
    assert guards["fit_allowed"] is False
    assert guards["runtime_activation_allowed"] is False
    assert guards["current_profile_change_allowed"] is False
    assert report_path.read_bytes() == preflight.canonical_json_bytes(report)


def test_go_requires_authorization_output_and_writes_nothing_on_failure(
    tmp_path: Path,
) -> None:
    aggregate = tmp_path / "aggregate.json"
    _write(aggregate, _aggregate(go=True))
    report = tmp_path / "finalization.json"
    with pytest.raises(ValueError, match="requires authorization_output"):
        finalizer.finalize_preflight(aggregate=aggregate, output=report)
    assert not report.exists()


def test_no_go_never_emits_authorization(tmp_path: Path) -> None:
    aggregate = tmp_path / "aggregate.json"
    _write(aggregate, _aggregate(go=False))
    report_path = tmp_path / "finalization.json"
    auth_path = tmp_path / "must-not-exist.json"
    report = finalizer.finalize_preflight(
        aggregate=aggregate,
        output=report_path,
        authorization_output=auth_path,
    )
    assert report["status"] == finalizer.ATTEMPT08_PREFLIGHT_NO_GO_STATUS
    assert report["decision"] == "no_go_no_development_authorization"
    assert report["authorization"] == {
        "emitted": False,
        "schema": None,
        "sha256": None,
    }
    assert not auth_path.exists()


def test_no_go_refuses_stale_existing_authorization_path(tmp_path: Path) -> None:
    aggregate = tmp_path / "aggregate.json"
    _write(aggregate, _aggregate(go=False))
    auth = tmp_path / "authorization.json"
    auth.write_text("stale\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="No-Go"):
        finalizer.finalize_preflight(
            aggregate=aggregate,
            output=tmp_path / "finalization.json",
            authorization_output=auth,
        )
    assert auth.read_text(encoding="utf-8") == "stale\n"


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["target_plan"].__setitem__("sha256", "0" * 64),
        lambda value: value["preflight_result"].__setitem__("decision", "no_go"),
        lambda value: value["evidence"].__setitem__(
            "proof_evidence_sha256", "0" * 64
        ),
        lambda value: value["evidence"].__setitem__(
            "runtime_semantic_anchor_sha256", "f" * 64
        ),
        lambda value: value["development_population"].__setitem__(
            "root_index_last", 200
        ),
        lambda value: value["authorization"].__setitem__(
            "future_audit_authorized", True
        ),
        lambda value: value["authorization"].__setitem__(
            "current_profile_change_allowed", True
        ),
        lambda value: value.__setitem__("unexpected", False),
    ],
)
def test_authorization_tampering_fails_closed(tmp_path: Path, mutator) -> None:
    aggregate, proof_paths = _go_bundle(tmp_path)
    auth_path = tmp_path / "authorization.json"
    finalizer.finalize_preflight(
        aggregate=aggregate,
        output=tmp_path / "finalization.json",
        authorization_output=auth_path,
        proof_paths=proof_paths,
        **_spot_kwargs(tmp_path),
    )
    authorization = json.loads(auth_path.read_bytes())
    mutator(authorization)
    with pytest.raises(ValueError):
        finalizer.validate_development_open_authorization(authorization)


def test_tampered_or_noncanonical_aggregate_fails_before_outputs(tmp_path: Path) -> None:
    aggregate = tmp_path / "aggregate.json"
    payload = _aggregate(go=True)
    payload["proof_evidence_sha256"] = "0" * 64
    _write(aggregate, payload)
    report = tmp_path / "finalization.json"
    auth = tmp_path / "authorization.json"
    with pytest.raises(ValueError):
        finalizer.finalize_preflight(
            aggregate=aggregate, output=report, authorization_output=auth
        )
    assert not report.exists() and not auth.exists()

    _write(aggregate, _aggregate(go=True))
    aggregate.write_bytes(aggregate.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="canonical"):
        finalizer.finalize_preflight(
            aggregate=aggregate, output=report, authorization_output=auth
        )
    assert not report.exists() and not auth.exists()


def test_go_requires_exact_five_producer_proofs(tmp_path: Path) -> None:
    aggregate, proof_paths = _go_bundle(tmp_path)
    report = tmp_path / "finalization.json"
    auth = tmp_path / "authorization.json"
    with pytest.raises(ValueError, match="exact five proof paths"):
        finalizer.finalize_preflight(
            aggregate=aggregate,
            output=report,
            authorization_output=auth,
        )
    truncated = dict(proof_paths)
    truncated.pop("root2_batch")
    with pytest.raises(ValueError, match="exact five proof paths"):
        finalizer.finalize_preflight(
            aggregate=aggregate,
            output=report,
            authorization_output=auth,
            proof_paths=truncated,
        )
    assert not report.exists() and not auth.exists()


def test_forged_go_from_real_determinism_no_go_cannot_authorize(
    tmp_path: Path,
) -> None:
    aggregate, proof_paths = _go_bundle(tmp_path)
    batch_b = proof_paths["root0_batch_b"]
    proof = json.loads(batch_b.read_bytes())
    proof["result_proof"]["opaque_teacher_sha256"] = "f" * 64
    _write(batch_b, proof)
    aggregate.unlink()
    aggregate_module.aggregate_preflight_proofs(
        **proof_paths,
        output=aggregate,
        spot_operational_evidence_sha256=hashlib.sha256(
            (tmp_path / "preflight_execution_evidence.json").read_bytes()
        ).hexdigest(),
    )
    forged = json.loads(aggregate.read_bytes())
    assert forged["decision"] == "no_go"
    forged["proof_gates"]["root0_batch_a_b_exact_teacher_determinism"] = True
    forged["decision"] = "go"
    forged["reasons"] = ["all_correctness_and_operational_preflight_gates_passed"]
    _write(aggregate, forged)
    # The aggregate-only shape is internally self-consistent; the producer
    # proof revalidation is what must reject the forged promotion.
    aggregate_module.validate_preflight_aggregate(forged)
    report = tmp_path / "forged-finalization.json"
    auth = tmp_path / "forged-authorization.json"
    with pytest.raises(ValueError, match="producer proof evidence"):
        finalizer.finalize_preflight(
            aggregate=aggregate,
            output=report,
            authorization_output=auth,
            proof_paths=proof_paths,
            **_spot_kwargs(tmp_path),
        )
    assert not report.exists() and not auth.exists()


def test_finalizer_outputs_are_distinct_and_no_clobber(tmp_path: Path) -> None:
    aggregate, proof_paths = _go_bundle(tmp_path)
    report = tmp_path / "finalization.json"
    auth = tmp_path / "authorization.json"
    report.write_text("owned\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        finalizer.finalize_preflight(
            aggregate=aggregate,
            output=report,
            authorization_output=auth,
            proof_paths=proof_paths,
            **_spot_kwargs(tmp_path),
        )
    assert report.read_text(encoding="utf-8") == "owned\n"
    assert not auth.exists()

    report.unlink()
    auth.write_text("owned-auth\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        finalizer.finalize_preflight(
            aggregate=aggregate,
            output=report,
            authorization_output=auth,
            proof_paths=proof_paths,
            **_spot_kwargs(tmp_path),
        )
    assert not report.exists()
    assert auth.read_text(encoding="utf-8") == "owned-auth\n"
    with pytest.raises(ValueError, match="distinct"):
        finalizer.finalize_preflight(
            aggregate=aggregate,
            output=aggregate,
            authorization_output=tmp_path / "other.json",
            proof_paths=proof_paths,
            **_spot_kwargs(tmp_path),
        )
    with pytest.raises(ValueError, match="outside Spot inputs"):
        finalizer.finalize_preflight(
            aggregate=aggregate,
            output=tmp_path / "spot-jobs" / "forbidden.json",
            authorization_output=tmp_path / "other.json",
            proof_paths=proof_paths,
            **_spot_kwargs(tmp_path),
        )


def test_authorization_and_finalization_never_export_teacher_details(
    tmp_path: Path,
) -> None:
    aggregate, proof_paths = _go_bundle(tmp_path)
    report = tmp_path / "finalization.json"
    auth = tmp_path / "authorization.json"
    finalizer.finalize_preflight(
        aggregate=aggregate,
        output=report,
        authorization_output=auth,
        proof_paths=proof_paths,
        **_spot_kwargs(tmp_path),
    )
    for path in (report, auth):
        raw = path.read_bytes()
        for forbidden in (
            b'"opaque_teacher_sha256"',
            b'"semantic_parity_sha256"',
            b'"selected_action_key"',
            b'"raw_paired_deltas',
            b'"paired_delta_vs_baseline"',
        ):
            assert forbidden not in raw
