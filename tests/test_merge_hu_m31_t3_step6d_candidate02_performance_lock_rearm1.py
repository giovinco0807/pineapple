from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import hu_m31_t3_step6d_performance_lock_open as v1_open
from ofc_regular import (
    merge_hu_m31_t3_step6d_candidate02_performance_lock as v1_merge,
)
from ofc_regular import (
    merge_hu_m31_t3_step6d_candidate02_performance_lock_rearm1 as subject,
)


def _sha(index: int) -> str:
    return f"{index + 1:064x}"


def _inputs(tmp_path: Path) -> subject.lock_open.PerformanceLockRearm1Inputs:
    return subject.lock_open.PerformanceLockRearm1Inputs(
        repository_root=tmp_path / "repo",
        plan_path=tmp_path / "plan.json",
        lock_output_directory=tmp_path / "rearm-roots",
        candidate_library=tmp_path / "candidate.so",
        reference_library=tmp_path / "reference.so",
        feature_encoder=tmp_path / "feature.so",
        startup_source=tmp_path / "startup.sh",
        incident_receipt_path=tmp_path / "incident.json",
        old_v1_root_directory=tmp_path / "old-v1-roots",
        development_summary_path=tmp_path / "development-summary.json",
        development_validation_path=tmp_path / "development-validation.json",
        development_root_directory=tmp_path / "development-roots",
        old_v1_global_claim_path=tmp_path / "old-v1-claim.json",
        global_claim_path=tmp_path / "rearm-claim.json",
    )


def _receive_inputs(tmp_path: Path) -> subject.PerformanceLockReceiveInputs:
    return subject.PerformanceLockReceiveInputs(
        run_dir=tmp_path / "run",
        receive_dir=tmp_path / "receive",
        project="test-project",
        bucket="test-bucket",
    )


def _done_paths(tmp_path: Path) -> tuple[list[Path], list[Path]]:
    candidate = [
        tmp_path / "receive" / "jobs" / f"candidate-{index:03d}" / "DONE.json"
        for index in range(subject.lock_plan.SHARD_COUNT_PER_ROLE)
    ]
    reference = [
        tmp_path / "receive" / "jobs" / f"reference-{index:03d}" / "DONE.json"
        for index in range(subject.lock_plan.SHARD_COUNT_PER_ROLE)
    ]
    return candidate, reference


def _plan_and_claim(
    tmp_path: Path,
) -> tuple[
    subject.lock_open.PerformanceLockRearm1Inputs,
    dict[str, Any],
    dict[str, Any],
]:
    inputs = _inputs(tmp_path)
    seed_contract = {"seed_set_sha256": _sha(10)}
    source_identity = {
        "candidate_library_sha256": _sha(20),
        "reference_library_sha256": _sha(21),
        "feature_encoder_sha256": _sha(22),
        "current_profile_registry_sha256": _sha(23),
    }
    plan = {
        "schema": subject.lock_plan.PLAN_SCHEMA,
        "scope": subject.lock_plan.PLAN_SCOPE,
        "candidate_variant": (
            subject.runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT
        ),
        "run_contract": {"variant": "rearm1"},
        "run_contract_digest": subject.lock_plan.LOCK_RUN_CONTRACT_DIGEST,
        "source_identity": source_identity,
        "image": {"family": "c4"},
        "allocation": {"threads": 16},
        "root_contract": {"seed_contract": seed_contract},
    }
    inputs.plan_path.write_bytes(subject.lock_plan.canonical_bytes(plan))
    claim = {
        "schema": subject.lock_open.CLAIM_SCHEMA,
        "status": subject.lock_open.CLAIM_STATUS,
        "precontent_plan": {
            "path": str(inputs.plan_path.resolve()),
            "sha256": subject.lock_open.sha256_file(inputs.plan_path),
            "canonical_sha256": subject.lock_plan.canonical_sha256(plan),
            "schema": subject.lock_plan.PLAN_SCHEMA,
        },
        "global_claim_path": subject.lock_open._lexical_absolute(
            inputs.global_claim_path
        ),
        "lock_output_directory": subject.lock_open._lexical_absolute(
            inputs.lock_output_directory
        ),
        "lock_run_contract": plan["run_contract"],
        "lock_run_contract_digest": subject.lock_plan.LOCK_RUN_CONTRACT_DIGEST,
        "image": plan["image"],
        "allocation": plan["allocation"],
        "seed_contract": seed_contract,
        "accepted_binaries": {
            "candidate": {"sha256": source_identity["candidate_library_sha256"]},
            "reference": {"sha256": source_identity["reference_library_sha256"]},
            "feature_encoder": {
                "sha256": source_identity["feature_encoder_sha256"]
            },
        },
        "ai_profiles_current": {
            "sha256": source_identity["current_profile_registry_sha256"]
        },
        "old_v1_lock_evidence": {
            "global_claim": {
                "sha256": subject.lock_open.OLD_V1_GLOBAL_CLAIM_SHA256
            },
            "seal": {"sha256": subject.lock_open.OLD_V1_SEAL_SHA256},
            "attempt1_used": False,
            "root_reuse_authorized": False,
        },
        "rearm_guards": {
            "new_global_claim": True,
            "fresh_700_series_seed_schedule": True,
            "old_v1_attempt1_used": False,
            "old_v1_attempt1_reused": False,
            "old_v1_root_reused": False,
            "old_v1_package_reused": False,
        },
        "restrictions": {
            "old_v1_attempt1_allowed": False,
            "old_v1_root_reuse_allowed": False,
            "alternate_seed_allowed": False,
            "reseed_allowed": False,
            "post_claim_reseed_allowed": False,
            "cloud_authorized": False,
            "training_authorized": False,
            "quality_authorized": False,
            "promotion_authorized": False,
            "opponent_private_discards_allowed": False,
        },
    }
    return inputs, plan, claim


def _materialization_and_seal() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    claim = {
        "precontent_plan": {"sha256": _sha(200)},
        "identity": "rearm1",
    }
    hashes = [_sha(index) for index in range(100)]
    materialization = {
        "schema": subject.lock_open.MATERIALIZATION_SCHEMA,
        "status": subject.lock_open.MATERIALIZATION_STATUS,
        "global_claim_sha256": subject.lock_open.canonical_sha256(claim),
        "plan_sha256": claim["precontent_plan"]["sha256"],
        "run_contract_digest": subject.lock_plan.LOCK_RUN_CONTRACT_DIGEST,
        "hand_indices": list(subject.runner.CONTRACT_HAND_INDICES),
        "root_count": 100,
        "root_artifact_sha256": hashes,
        "aggregate_root_sha256": subject.lock_open.canonical_sha256(hashes),
        "same_identity_resume_only": True,
        "fresh_recovery_seed_schedule": True,
        "old_v1_attempt1_reused": False,
        "old_v1_root_reused": False,
        "reseeded": False,
        "training_eligible": False,
        "current_profile_changed": False,
    }
    seal = {
        "schema": subject.lock_open.SEAL_SCHEMA,
        "status": subject.lock_open.SEAL_STATUS,
        "global_claim_sha256": subject.lock_open.canonical_sha256(claim),
        "materialization_sha256": subject.lock_open.canonical_sha256(
            materialization
        ),
        "plan_sha256": materialization["plan_sha256"],
        "run_contract_digest": subject.lock_plan.LOCK_RUN_CONTRACT_DIGEST,
        "hand_indices": list(subject.runner.CONTRACT_HAND_INDICES),
        "root_count": 100,
        "observation_count": 200,
        "profile_counts": {
            profile: 20 for profile in subject.M31_T3_BEHAVIOR_PROFILES
        },
        "seat_counts": {"first": 100, "second": 100},
        "root_artifact_sha256": hashes,
        "aggregate_root_sha256": subject.lock_open.canonical_sha256(hashes),
        "root_topology_sha256": _sha(300),
        "observation_fingerprint_sha256": _sha(301),
        "root_artifact_unique": True,
        "observation_fingerprint_unique": True,
        "development_comparison": {
            "development_all100_root_sha256": _sha(302),
            "development_root_count": 100,
            "lock_fingerprint_overlap_count": 0,
            "lock_root_hash_overlap_count": 0,
            "lock_seed_overlap_count": 0,
        },
        "old_performance_lock_comparison": {
            "old_v1_global_claim_sha256": (
                subject.lock_open.OLD_V1_GLOBAL_CLAIM_SHA256
            ),
            "old_v1_seal_sha256": subject.lock_open.OLD_V1_SEAL_SHA256,
            "old_v1_root_count": 100,
            "rearm1_fingerprint_overlap_count": 0,
            "rearm1_root_hash_overlap_count": 0,
            "rearm1_seed_overlap_count": 0,
            "old_v1_attempt1_reused": False,
            "old_v1_root_reused": False,
        },
        "visibility": {
            "runner_validator_replayed_all_roots": True,
            "first_observation_count": 100,
            "second_observation_count": 100,
            "opponent_private_discards_used": False,
            "current_profile_resolved": False,
        },
        "selection_inputs": {
            "timing_used": False,
            "q_used": False,
            "ev_used": False,
            "all_100_preregistered_hands_used": True,
        },
        "rearm_guards": {
            "incident_receipt_sha256": (
                subject.lock_open.STARTUP_FAILURE_RECEIPT_SHA256
            ),
            "fresh_700_series_seed_schedule": True,
            "same_identity_resume_only": True,
            "old_v1_attempt1_reused": False,
            "old_v1_root_reused": False,
            "post_claim_reseeded": False,
        },
        "training_eligible": False,
        "quality_evidence": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
    }
    assert set(materialization) == subject.lock_open._MATERIALIZATION_KEYS
    assert set(seal) == subject.lock_open._SEAL_KEYS
    return claim, materialization, seal


def test_open_input_roundtrip_binds_rearm_only_paths(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    encoded = subject._serialize_open_inputs(inputs)
    assert set(encoded) == subject._OPEN_INPUT_KEYS
    assert encoded["incident_receipt_path"] == str(
        inputs.incident_receipt_path.resolve()
    )
    assert encoded["old_v1_root_directory"] != encoded["lock_output_directory"]
    assert subject._serialize_open_inputs(
        subject._deserialize_open_inputs(encoded)
    ) == encoded

    legacy = v1_open.PerformanceLockInputs(
        repository_root=tmp_path,
        plan_path=tmp_path / "legacy-plan.json",
        lock_output_directory=tmp_path / "legacy-roots",
        candidate_library=tmp_path / "candidate.so",
        reference_library=tmp_path / "reference.so",
        feature_encoder=tmp_path / "feature.so",
        startup_source=tmp_path / "startup.sh",
        development_summary_path=tmp_path / "summary.json",
        development_validation_path=tmp_path / "validation.json",
        development_root_directory=tmp_path / "development",
        global_claim_path=tmp_path / "legacy-claim.json",
    )
    with pytest.raises(TypeError, match="legacy v1 inputs are forbidden"):
        subject._serialize_open_inputs(legacy)


def test_context_selects_recovery_identity_and_restores_v1() -> None:
    before = {
        name: getattr(v1_merge, name)
        for name in (
            "lock_plan",
            "lock_open",
            "lock_spot",
            "runner",
            "MERGE_SCHEMA",
            "VALIDATION_SCHEMA",
            "SCOPE",
            "PASS_DECISION",
            "NO_GO_DECISION",
            "_serialize_open_inputs",
            "_deserialize_open_inputs",
            "_validate_claim_plan_lineage",
            "_validate_materialization_and_seal",
        )
    }
    with subject._rearm1_merge_context():
        assert v1_merge.lock_plan is subject.lock_plan
        assert v1_merge.lock_open is subject.lock_open
        assert v1_merge.MERGE_SCHEMA == subject.MERGE_SCHEMA
        assert (
            v1_merge.runner.CANDIDATE02_PERFORMANCE_LOCK_VARIANT
            == subject.runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT
        )
        assert (
            v1_merge.lock_spot.RESULT_OPEN_CLAIM_NAME
            == subject.spot_v1.RESULT_OPEN_CLAIM_NAME
        )
    for name, value in before.items():
        assert getattr(v1_merge, name) is value


def test_public_merge_delegates_under_rearm_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, reference = _done_paths(tmp_path)
    observed: dict[str, Any] = {}

    def fake_merge(**kwargs: Any) -> dict[str, Any]:
        observed["variant"] = (
            v1_merge.runner.CANDIDATE02_PERFORMANCE_LOCK_VARIANT
        )
        observed["schema"] = v1_merge.MERGE_SCHEMA
        observed["open"] = v1_merge._serialize_open_inputs(kwargs["lock_inputs"])
        return {"status": "pass"}

    monkeypatch.setattr(v1_merge, "merge_candidate02_performance_lock", fake_merge)
    result = subject.merge_candidate02_performance_lock_rearm1(
        candidate_done_paths=candidate,
        reference_done_paths=reference,
        lock_inputs=_inputs(tmp_path),
        receive_inputs=_receive_inputs(tmp_path),
    )
    assert result == {"status": "pass"}
    assert observed["variant"] == (
        subject.runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT
    )
    assert observed["schema"] == subject.MERGE_SCHEMA
    assert "old_v1_root_directory" in observed["open"]


@pytest.mark.parametrize("case", ("missing", "duplicate", "mixed_roles"))
def test_done_input_grid_fails_closed_before_replay(
    tmp_path: Path, case: str
) -> None:
    candidate, reference = _done_paths(tmp_path)
    if case == "missing":
        candidate.pop()
    elif case == "duplicate":
        candidate[-1] = candidate[0]
    else:
        reference[-1] = candidate[0]
    with pytest.raises(ValueError, match="DONE inputs"):
        subject.merge_candidate02_performance_lock_rearm1(
            candidate_done_paths=candidate,
            reference_done_paths=reference,
            lock_inputs=_inputs(tmp_path),
            receive_inputs=_receive_inputs(tmp_path),
        )


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("mixed_binary", "claim/plan lineage"),
        ("contract_digest", "claim/plan lineage"),
        ("old_claim", "claim/plan lineage"),
        ("old_root_path", "claim/plan lineage"),
    ),
)
def test_claim_chain_digest_and_binary_tamper_fail_closed(
    tmp_path: Path, case: str, message: str
) -> None:
    inputs, plan, claim = _plan_and_claim(tmp_path)
    tampered = deepcopy(claim)
    if case == "mixed_binary":
        tampered["accepted_binaries"]["candidate"]["sha256"] = _sha(999)
    elif case == "contract_digest":
        tampered["lock_run_contract_digest"] = _sha(998)
    elif case == "old_claim":
        tampered["old_v1_lock_evidence"]["global_claim"]["sha256"] = _sha(997)
    else:
        tampered["lock_output_directory"] = subject.lock_open._lexical_absolute(
            inputs.old_v1_root_directory
        )
    with pytest.raises(ValueError, match=message):
        subject._validate_claim_plan_lineage(
            inputs=inputs,
            plan=plan,
            claim=tampered,
        )


def test_claim_and_plan_lineage_accepts_exact_rearm_identity(tmp_path: Path) -> None:
    inputs, plan, claim = _plan_and_claim(tmp_path)
    subject._validate_claim_plan_lineage(
        inputs=inputs,
        plan=plan,
        claim=claim,
    )


@pytest.mark.parametrize(
    "case",
    (
        "duplicate_root",
        "missing_root",
        "contract_digest",
        "old_seed_overlap",
        "old_attempt1_reused",
        "post_claim_reseeded",
    ),
)
def test_root_chain_no_reuse_and_exact_coverage_fail_closed(case: str) -> None:
    claim, materialization, seal = _materialization_and_seal()
    broken_materialization = deepcopy(materialization)
    broken_seal = deepcopy(seal)
    if case == "duplicate_root":
        broken_materialization["root_artifact_sha256"][1] = (
            broken_materialization["root_artifact_sha256"][0]
        )
        broken_seal["root_artifact_sha256"][1] = (
            broken_seal["root_artifact_sha256"][0]
        )
        duplicate_hashes = broken_seal["root_artifact_sha256"]
        aggregate = subject.lock_open.canonical_sha256(duplicate_hashes)
        broken_materialization["aggregate_root_sha256"] = aggregate
        broken_seal["aggregate_root_sha256"] = aggregate
        broken_seal["materialization_sha256"] = (
            subject.lock_open.canonical_sha256(broken_materialization)
        )
    elif case == "missing_root":
        broken_materialization["root_artifact_sha256"].pop()
        broken_seal["root_artifact_sha256"].pop()
        missing_hashes = broken_seal["root_artifact_sha256"]
        aggregate = subject.lock_open.canonical_sha256(missing_hashes)
        broken_materialization["aggregate_root_sha256"] = aggregate
        broken_seal["aggregate_root_sha256"] = aggregate
        broken_seal["materialization_sha256"] = (
            subject.lock_open.canonical_sha256(broken_materialization)
        )
    elif case == "contract_digest":
        broken_materialization["run_contract_digest"] = _sha(900)
    elif case == "old_seed_overlap":
        broken_seal["old_performance_lock_comparison"][
            "rearm1_seed_overlap_count"
        ] = 1
    elif case == "old_attempt1_reused":
        broken_seal["old_performance_lock_comparison"][
            "old_v1_attempt1_reused"
        ] = True
    else:
        broken_seal["rearm_guards"]["post_claim_reseeded"] = True
    with pytest.raises(ValueError, match="materialization/root-seal lineage"):
        subject._validate_materialization_and_seal(
            plan={},
            claim=claim,
            materialization=broken_materialization,
            seal=broken_seal,
        )


def test_materialization_and_seal_accept_exact_fresh_rearm_identity() -> None:
    claim, materialization, seal = _materialization_and_seal()
    subject._validate_materialization_and_seal(
        plan={},
        claim=claim,
        materialization=materialization,
        seal=seal,
    )
