from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_performance_development_v2_contract as subject,
)


def test_contract_binds_accepted_development_and_exact_tail_topology() -> None:
    value = subject.build_contract()

    assert value["schema"] == subject.CONTRACT_SCHEMA
    assert value["accepted_provenance"] == {
        "full100_plan_sha256": subject.FULL100_PLAN_SHA256,
        "accepted_summary_sha256": subject.ACCEPTED_SUMMARY_SHA256,
        "accepted_validation_sha256": subject.ACCEPTED_VALIDATION_SHA256,
        "full100_plan_run_contract_digest": subject.FULL_RUN_CONTRACT_DIGEST,
        "candidate_library_sha256": subject.CANDIDATE_LIBRARY_SHA256,
        "reference_library_sha256": subject.REFERENCE_LIBRARY_SHA256,
        "root_set_sha256": subject.DEVELOPMENT_ROOT_SET_SHA256,
        "root_topology_sha256": subject.DEVELOPMENT_ROOT_TOPOLOGY_SHA256,
        "seed_set_sha256": subject.DEVELOPMENT_SEED_SET_SHA256,
        "files_rehashed_now": [
            "full100_plan",
            "accepted_merge_summary",
            "accepted_merge_validation",
        ],
        "candidate_library_file_rehashed_now": False,
        "reference_library_file_rehashed_now": False,
        "root_files_rehashed_now": False,
        "seed_material_recomputed_now": False,
        "binary_root_seed_values_are_stored_digest_bindings": True,
        "historical_result_used_for_candidate_freeze_only": True,
        "historical_timing_reused_as_new_evidence": False,
    }
    assert value["work"]["work_hand_indices"] == [0, 4, 5, 12, 14, 16, 17, 23, 41, 43]
    tail = value["tail_execution"]
    assert tail["run_contract_digest"] == subject.TAIL_RUN_CONTRACT_DIGEST
    assert tail["run_contract_schema"] == subject.TAIL_RUN_CONTRACT_SCHEMA
    assert tail["candidate_variant"] == subject.TAIL_RUN_CONTRACT_VARIANT
    assert tail["selection_manifest_sha256"] == subject.TAIL_SELECTION_MANIFEST_SHA256
    assert tail["selection_manifest_bytes"] == subject.TAIL_SELECTION_MANIFEST_BYTES
    assert tail["prior_mixed_geometry_artifacts_qualification_allowed"] is False
    assert tail["performance_only_remeasurement"] is True
    assert tail["quality_or_holdout_claim_allowed"] is False
    assert tail["run_contract"]["tail_hand_indices"] == list(subject.TAIL_HAND_INDICES)
    assert tail["run_contract"]["selection_manifest_sha256"] == (
        subject.TAIL_SELECTION_MANIFEST_SHA256
    )
    assert value["topology"] == {
        "machine_type": "c4-standard-16",
        "guest_vcpus_per_instance": 16,
        "instances": 2,
        "instances_per_source_role": 1,
        "workers_per_source_process": 1,
        "rayon_threads_per_worker": 16,
        "omp_threads": 1,
        "m3_batch_threads": 1,
        "execution_mode": "one_source_one_process_scalar_1x16",
        "minimum_required_c4_vcpus": 32,
    }
    assert value["cloud_executable"] is False
    assert value["launch_authorized"] is False
    assert value["cloud_mutated"] is False
    assert value["current_profile_changed"] is False


def test_future_package_launch_must_rehash_actual_bytes_before_and_after() -> None:
    requirements = subject.build_contract()["future_package_launch_requirements"]

    assert requirements["candidate_source_library_rehash_required"] is True
    assert requirements["candidate_packaged_library_rehash_required"] is True
    assert requirements["reference_source_library_rehash_required"] is True
    assert requirements["reference_packaged_library_rehash_required"] is True
    assert requirements["all_100_source_root_files_rehash_required"] is True
    assert requirements["all_100_packaged_root_files_rehash_required"] is True
    assert requirements["source_and_packaged_root_bytes_must_match"] is True
    assert requirements["tail_v2_selection_source_and_packaged_bytes_must_match"] is True
    assert requirements["tail_v2_selection_sha256_must_equal_bound_digest"] is True
    assert requirements["tail_execution_contract_must_not_reuse_full100_plan_contract"] is True
    assert requirements["development_seed_schedule_recompute_required"] is True
    assert requirements["rearm2_root_package_seed_overlap_must_be_zero"] is True
    assert requirements["prepackage_and_postpackage_checks_required"] is True
    assert requirements["launch_slice_must_remain_fail_closed_until_all_pass"] is True


def test_contract_denies_every_rearm2_resource_class() -> None:
    deny = subject.build_contract()["performance_lock_rearm2_denylist"]

    assert deny["claims_and_contracts"]["global_claim_sha256"] == (
        subject.REARM2_GLOBAL_CLAIM_SHA256
    )
    assert deny["roots"]["aggregate_root_sha256"] == (
        subject.REARM2_AGGREGATE_ROOT_SHA256
    )
    assert deny["packages"]["manifest_sha256"] == (
        subject.REARM2_PACKAGE_MANIFEST_SHA256
    )
    assert deny["seeds"]["seed_set_sha256"] == subject.REARM2_SEED_SET_SHA256
    assert deny["seeds"]["namespace_bases"] == subject.REARM2_NAMESPACE_BASES
    assert deny["root_reuse_allowed"] is False
    assert deny["package_reuse_allowed"] is False
    assert deny["seed_reuse_allowed"] is False
    assert deny["namespace_reuse_allowed"] is False


def test_contract_tamper_fails_closed() -> None:
    value = subject.build_contract()
    mutations = []

    topology = deepcopy(value)
    topology["topology"]["rayon_threads_per_worker"] = 8
    mutations.append(topology)

    tail = deepcopy(value)
    tail["work"]["work_hand_indices"][-1] = 51
    mutations.append(tail)

    rearm = deepcopy(value)
    rearm["performance_lock_rearm2_denylist"]["seed_reuse_allowed"] = True
    mutations.append(rearm)

    executable = deepcopy(value)
    executable["cloud_executable"] = True
    mutations.append(executable)

    for mutation in mutations:
        with pytest.raises(ValueError, match="contract changed"):
            subject.validate_contract(mutation)


def test_accepted_file_hash_is_recomputed(tmp_path: Path) -> None:
    summary = subject.DEFAULT_ACCEPTED_MERGE_DIR / "summary.json"
    tampered = tmp_path / "summary.json"
    tampered.write_bytes(summary.read_bytes() + b" ")

    with pytest.raises(ValueError, match="summary hash changed"):
        subject.build_contract(accepted_summary_path=tampered)


def test_tail_selection_bytes_and_old_execution_contract_fail_closed(
    tmp_path: Path,
) -> None:
    selection = tmp_path / "selection.json"
    selection.write_bytes(subject.DEFAULT_TAIL_SELECTION_MANIFEST_PATH.read_bytes() + b" ")
    with pytest.raises(ValueError, match="selection manifest bytes changed"):
        subject.build_contract(tail_selection_manifest_path=selection)

    old_contract = deepcopy(subject.build_tail_run_contract())
    old_contract["schema"] = "hu_m31_t3_step6d_candidate02_performance_run_contract_v1"
    old_contract["candidate_variant"] = "candidate02_compact_scorer"
    old_contract["tail_hand_indices"] = list(subject.FULL100_PLAN_TAIL_HAND_INDICES)
    old_contract.pop("selection_manifest_sha256")
    with pytest.raises(ValueError):
        subject.validate_tail_run_contract(old_contract)
