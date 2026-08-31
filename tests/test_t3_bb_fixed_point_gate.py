from __future__ import annotations

import copy

import pytest

from ai.tutor.promotion_gate_m3_full_card_strength import (
    REQUIRED_STRATA,
    SOLVER_METHOD,
    T3_BB_LIKELIHOOD_METHOD,
    T3_BB_LIKELIHOOD_SCHEMA,
    root_identity_commitment_sha256,
)
from ai.tutor.t3_bb_fixed_point_gate import (
    EVIDENCE_KIND,
    EVIDENCE_SCHEMA,
    GATE_ID,
    PARTITION_SCHEMA,
    ROOT_MANIFEST_SCHEMA,
    SCOPE,
    build_locked_t3_bb_fixed_point_gate_config,
    build_t3_bb_likelihood_binding,
    canonical_sha256,
    derive_t3_bb_fixed_point_metrics,
    root_commitment_sha256,
    self_hash,
    validate_t3_bb_fixed_point_evidence,
    verify_t3_bb_fixed_point_gate_result,
)


def _sha(label: str) -> str:
    return canonical_sha256({"fixture": label})


def _root_manifest(seeds: tuple[int, ...]) -> dict:
    roots = []
    for stratum in REQUIRED_STRATA:
        actor, joker_text = stratum.split("_joker")
        root = {
            "root_id": f"fixed-point-{stratum}",
            "stratum": stratum,
            "actor": actor,
            "visible_joker_count": int(joker_text),
            "root_identity_commitment_sha256": root_identity_commitment_sha256(
                f"fixed-point-{stratum}"
            ),
        }
        root["root_commitment_sha256"] = root_commitment_sha256(root)
        roots.append(root)
    manifest = {
        "schema": ROOT_MANIFEST_SCHEMA,
        "purpose": "independent_fixed_point_holdout",
        "locked_before_evaluation": True,
        "ruleset": "standard_ofc_pineapple_hu_joker2",
        "position_contract_version": "bb_first_v1",
        "evaluation_seeds": list(seeds),
        "roots": roots,
    }
    manifest["manifest_sha256"] = self_hash(manifest, "manifest_sha256")
    return manifest


def _partition(purpose: str, seed: int) -> dict:
    partition = {
        "schema": PARTITION_SCHEMA,
        "purpose": purpose,
        "root_identity_commitments": [
            root_identity_commitment_sha256(f"excluded-{purpose}")
        ],
        "solver_seeds": [seed],
    }
    partition["manifest_sha256"] = self_hash(partition, "manifest_sha256")
    return partition


def _build_valid(
    *, seeds: tuple[int, ...] = (101, 202), rounds: int = 2
) -> tuple[dict, dict]:
    solver_sha = _sha("solver-manifest")
    range_sha = _sha("range-builder-source")
    root_manifest = _root_manifest(seeds)
    partitions = {
        purpose: _partition(purpose, index + 1)
        for index, purpose in enumerate(("training", "calibration", "smoke"))
    }
    config = build_locked_t3_bb_fixed_point_gate_config(
        approved_solver_manifest_sha256=solver_sha,
        approved_range_builder_source_sha256=range_sha,
        approved_root_manifest_sha256=root_manifest["manifest_sha256"],
        approved_excluded_partition_sha256={
            purpose: partition["manifest_sha256"]
            for purpose, partition in partitions.items()
        },
        max_policy_tv=0.05,
        max_btn_posterior_weight_tv=0.05,
    )
    rows = []
    artifacts = [_sha("baseline-artifact")] + [
        _sha(f"candidate-artifact-round-{round_index}")
        for round_index in range(1, rounds + 1)
    ]
    checkpoints = [_sha("baseline-checkpoint")] + [
        _sha(f"candidate-checkpoint-round-{round_index}")
        for round_index in range(1, rounds + 1)
    ]
    for round_index in range(1, rounds + 1):
        if round_index == 1:
            previous_policy = {"place-a": 0.50, "place-b": 0.50}
            current_policy = {"place-a": 0.52, "place-b": 0.48}
            previous_weights = {"state-a": 50.0, "state-b": 50.0}
            current_weights = {"state-a": 51.0, "state-b": 49.0}
        else:
            previous_policy = {"place-a": 0.52, "place-b": 0.48}
            current_policy = {"place-a": 0.53, "place-b": 0.47}
            previous_weights = {"state-a": 51.0, "state-b": 49.0}
            current_weights = {"state-a": 51.5, "state-b": 48.5}
        for seed in seeds:
            for root in root_manifest["roots"]:
                is_btn = root["actor"] == "btn"
                rows.append(
                    {
                        "round_index": round_index,
                        "solver_seed": seed,
                        "root_id": root["root_id"],
                        "root_commitment_sha256": root[
                            "root_commitment_sha256"
                        ],
                        "root_identity_commitment_sha256": root[
                            "root_identity_commitment_sha256"
                        ],
                        "stratum": root["stratum"],
                        "actor": root["actor"],
                        "visible_joker_count": root["visible_joker_count"],
                        "previous_candidate_policy_artifact_sha256": artifacts[
                            round_index - 1
                        ],
                        "previous_candidate_policy_checkpoint_sha256": checkpoints[
                            round_index - 1
                        ],
                        "candidate_policy_artifact_sha256": artifacts[round_index],
                        "candidate_policy_checkpoint_sha256": checkpoints[
                            round_index
                        ],
                        "candidate_policy_method": SOLVER_METHOD,
                        "solver_manifest_sha256": solver_sha,
                        "range_builder_source_sha256": range_sha,
                        "exact_exploitability_computed": False,
                        "previous_policy_distribution": copy.deepcopy(
                            previous_policy
                        ),
                        "current_policy_distribution": copy.deepcopy(current_policy),
                        "previous_btn_posterior_weights": (
                            copy.deepcopy(previous_weights) if is_btn else None
                        ),
                        "current_btn_posterior_weights": (
                            copy.deepcopy(current_weights) if is_btn else None
                        ),
                    }
                )
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "gate_id": GATE_ID,
        "scope": SCOPE,
        "evidence_kind": EVIDENCE_KIND,
        "gate_config_sha256": config["gate_config_sha256"],
        "exact_exploitability_computed": False,
        "candidate_policy_method": SOLVER_METHOD,
        "solver_manifest_sha256": solver_sha,
        "range_builder_source_sha256": range_sha,
        "root_manifest": root_manifest,
        "excluded_partitions": partitions,
        "raw_iteration_rows": rows,
        "published_summary": {},
    }
    evidence["published_summary"] = derive_t3_bb_fixed_point_metrics(
        evidence, config=config
    )
    evidence["artifact_sha256"] = self_hash(evidence, "artifact_sha256")
    return evidence, config


def _rehash_evidence(evidence: dict, *, derive: bool, config: dict) -> None:
    if derive:
        evidence["published_summary"] = derive_t3_bb_fixed_point_metrics(
            evidence, config=config
        )
    evidence["artifact_sha256"] = self_hash(evidence, "artifact_sha256")


def _relock(evidence: dict, config: dict) -> dict:
    config = build_locked_t3_bb_fixed_point_gate_config(
        approved_solver_manifest_sha256=evidence["solver_manifest_sha256"],
        approved_range_builder_source_sha256=evidence[
            "range_builder_source_sha256"
        ],
        approved_root_manifest_sha256=evidence["root_manifest"][
            "manifest_sha256"
        ],
        approved_excluded_partition_sha256={
            purpose: partition["manifest_sha256"]
            for purpose, partition in evidence["excluded_partitions"].items()
        },
        max_policy_tv=config["thresholds"]["max_policy_tv"],
        max_btn_posterior_weight_tv=config["thresholds"][
            "max_btn_posterior_weight_tv"
        ],
        min_independent_seeds=config["thresholds"]["min_independent_seeds"],
        min_roots_per_stratum=config["thresholds"]["min_roots_per_stratum"],
        min_consecutive_converged_rounds=config["thresholds"][
            "min_consecutive_converged_rounds"
        ],
    )
    evidence["gate_config_sha256"] = config["gate_config_sha256"]
    _rehash_evidence(evidence, derive=True, config=config)
    return config


def test_two_seed_two_round_six_stratum_evidence_passes_and_builds_binding() -> None:
    evidence, config = _build_valid()
    result = validate_t3_bb_fixed_point_evidence(evidence, config=config)

    assert result["passed"] is True, result["failures"]
    assert result["promotion_eligible"] is True
    assert result["exact_exploitability_computed"] is False
    assert result["derived_metrics"]["final_consecutive_converged_rounds"] == 2
    assert set(result["derived_metrics"]["independent_seeds"]) == {101, 202}
    assert verify_t3_bb_fixed_point_gate_result(
        evidence, config=config, gate_result=result
    ) == result

    binding = build_t3_bb_likelihood_binding(
        evidence, config=config, gate_result=result
    )
    assert binding["schema"] == T3_BB_LIKELIHOOD_SCHEMA
    assert binding["method"] == T3_BB_LIKELIHOOD_METHOD
    assert binding["fixed_point_converged"] is True
    assert binding["fixed_point_evidence_sha256"] == evidence["artifact_sha256"]
    assert binding["fixed_point_gate_result_sha256"] == result[
        "gate_result_sha256"
    ]
    assert binding["solver_manifest_sha256"] == evidence[
        "solver_manifest_sha256"
    ]


def test_one_seed_fails_even_when_locked_and_self_hashed() -> None:
    evidence, config = _build_valid(seeds=(101,))
    result = validate_t3_bb_fixed_point_evidence(evidence, config=config)

    assert result["passed"] is False
    assert any("insufficient independent seeds" in item for item in result["failures"])


def test_one_converged_round_fails_and_cannot_build_binding() -> None:
    evidence, config = _build_valid(rounds=1)
    result = validate_t3_bb_fixed_point_evidence(evidence, config=config)

    assert result["passed"] is False
    assert any("insufficient consecutive converged rounds" in item for item in result["failures"])
    with pytest.raises(ValueError, match="did not pass"):
        build_t3_bb_likelihood_binding(
            evidence, config=config, gate_result=result
        )


def test_policy_tv_breach_is_rederived_and_fails() -> None:
    evidence, config = _build_valid()
    for row in evidence["raw_iteration_rows"]:
        if row["round_index"] == 2:
            row["current_policy_distribution"] = {
                "place-a": 0.90,
                "place-b": 0.10,
            }
    _rehash_evidence(evidence, derive=True, config=config)

    result = validate_t3_bb_fixed_point_evidence(evidence, config=config)
    assert result["passed"] is False
    assert result["derived_metrics"]["per_round"]["2"]["max_policy_tv"] > 0.05
    assert any("insufficient consecutive converged rounds" in item for item in result["failures"])


def test_btn_posterior_weight_tv_breach_is_rederived_and_fails() -> None:
    evidence, config = _build_valid()
    for row in evidence["raw_iteration_rows"]:
        if row["round_index"] == 2 and row["actor"] == "btn":
            row["current_btn_posterior_weights"] = {
                "state-a": 90.0,
                "state-b": 10.0,
            }
    _rehash_evidence(evidence, derive=True, config=config)

    result = validate_t3_bb_fixed_point_evidence(evidence, config=config)
    assert result["passed"] is False
    assert (
        result["derived_metrics"]["per_round"]["2"][
            "max_btn_posterior_weight_tv"
        ]
        > 0.05
    )
    assert any("insufficient consecutive converged rounds" in item for item in result["failures"])


def test_raw_tamper_fails_even_if_artifact_is_resigned() -> None:
    evidence, config = _build_valid()
    evidence["raw_iteration_rows"][0]["current_policy_distribution"] = {
        "place-a": 0.99,
        "place-b": 0.01,
    }
    # A malicious producer can recompute the envelope hash, but cannot keep a
    # stale aggregate or the next-round raw chain consistent.
    _rehash_evidence(evidence, derive=False, config=config)

    result = validate_t3_bb_fixed_point_evidence(evidence, config=config)
    assert result["passed"] is False
    assert any("published_summary: raw-derived mismatch" in item for item in result["failures"])
    assert any("policy chain mismatch" in item for item in result["failures"])


@pytest.mark.parametrize(
    ("field", "needle"),
    [
        ("solver_manifest_sha256", "solver_manifest_sha256"),
        ("range_builder_source_sha256", "range_builder_source_sha256"),
    ],
)
def test_source_or_solver_hash_drift_across_rows_fails(field: str, needle: str) -> None:
    evidence, config = _build_valid()
    evidence["raw_iteration_rows"][-1][field] = _sha(f"drift-{field}")
    _rehash_evidence(evidence, derive=False, config=config)

    result = validate_t3_bb_fixed_point_evidence(evidence, config=config)
    assert result["passed"] is False
    assert any(needle in item and "binding mismatch" in item for item in result["failures"])


def test_holdout_root_and_seed_overlap_with_excluded_partition_fail() -> None:
    evidence, config = _build_valid()
    smoke = evidence["excluded_partitions"]["smoke"]
    smoke["root_identity_commitments"].append(
        evidence["root_manifest"]["roots"][0][
            "root_identity_commitment_sha256"
        ]
    )
    smoke["root_identity_commitments"].sort()
    smoke["solver_seeds"].append(101)
    smoke["solver_seeds"].sort()
    smoke["manifest_sha256"] = self_hash(smoke, "manifest_sha256")
    config = _relock(evidence, config)

    result = validate_t3_bb_fixed_point_evidence(evidence, config=config)
    assert result["passed"] is False
    assert any("root overlap with smoke" in item for item in result["failures"])
    assert any("seed overlap with smoke" in item for item in result["failures"])


def test_fake_published_aggregate_cannot_override_raw_tv_breach() -> None:
    evidence, config = _build_valid()
    claimed_pass = copy.deepcopy(evidence["published_summary"])
    for row in evidence["raw_iteration_rows"]:
        if row["round_index"] == 2:
            row["current_policy_distribution"] = {
                "place-a": 0.95,
                "place-b": 0.05,
            }
    evidence["published_summary"] = claimed_pass
    _rehash_evidence(evidence, derive=False, config=config)

    result = validate_t3_bb_fixed_point_evidence(evidence, config=config)
    assert result["passed"] is False
    assert result["derived_metrics"]["per_round"]["2"]["converged"] is False
    assert any("published_summary: raw-derived mismatch" in item for item in result["failures"])


def test_prior_only_rows_fail_closed_even_when_all_tvs_are_small() -> None:
    evidence, config = _build_valid()
    evidence["candidate_policy_method"] = "frozen_teacher_ranking_prior_v1"
    for row in evidence["raw_iteration_rows"]:
        row["candidate_policy_method"] = "frozen_teacher_ranking_prior_v1"
    _rehash_evidence(evidence, derive=False, config=config)

    result = validate_t3_bb_fixed_point_evidence(evidence, config=config)
    assert result["passed"] is False
    assert any("candidate_policy_method" in item for item in result["failures"])


def test_resigned_fake_pass_result_is_rejected_by_binding_builder() -> None:
    evidence, config = _build_valid(rounds=1)
    result = validate_t3_bb_fixed_point_evidence(evidence, config=config)
    fake = copy.deepcopy(result)
    fake["passed"] = True
    fake["promotion_eligible"] = True
    fake["status"] = "t3_bb_fixed_point_ready"
    fake["gate_result_sha256"] = self_hash(fake, "gate_result_sha256")

    with pytest.raises(ValueError, match="does not match raw evidence"):
        build_t3_bb_likelihood_binding(evidence, config=config, gate_result=fake)
