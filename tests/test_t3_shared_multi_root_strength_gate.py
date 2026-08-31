from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

import ai.tutor.t3_shared_multi_root_strength_gate as gate
from ai.engine.action_space import get_turn_actions
from ai.engine.encoding import Board
from ai.tutor.exact_late import action_key
from ai.tutor.promotion_gate_m3_full_card_strength import (
    REQUIRED_AUDITS,
    REQUIRED_STRATA,
    T3_BB_LIKELIHOOD_METHOD,
    T3_BB_LIKELIHOOD_SCHEMA,
    canonical_sha256,
    root_identity_commitment_sha256,
    self_hash,
)
from ai.tutor.t3_hu_full_card_mccfr import serialize_strategy_profile
from ai.tutor.t3_hu_reduced_fixtures import compile_canonical_reduced_fixture


def _sha(label: str) -> str:
    return canonical_sha256({"label": label})


def _root_key(stratum: str):
    actor, joker_text = stratum.split("_joker")
    compiled = compile_canonical_reduced_fixture(actor, int(joker_text))
    return compiled.root.branches[0].child.state.infoset_key


def _legal_distribution(key):
    rows = key.board_bb if key.actor == "bb" else key.board_btn
    board = Board(top=list(rows[0]), middle=list(rows[1]), bottom=list(rows[2]))
    actions = sorted(
        action_key(action)
        for action in get_turn_actions(list(key.current_draw), board)
    )
    return {
        action_id: 1.0 if index == 0 else 0.0
        for index, action_id in enumerate(actions)
    }


def _fake_result(
    stratum: str,
    *,
    seed: int = 11,
    reused_observation_digest: str | None = None,
):
    key = _root_key(stratum)
    profile_text = serialize_strategy_profile({key: _legal_distribution(key)})
    profile_sha = hashlib.sha256(profile_text.encode("utf-8")).hexdigest()
    opaque_ids = (_sha(f"{stratum}-opaque-a"), _sha(f"{stratum}-opaque-b"))
    root_prior = {
        "schema": "ofc_multi_root_exact_prior/v1",
        "sampling_contract": gate.SUPER_ROOT_SAMPLING_CONTRACT,
        "roots": [
            {
                "root_id_sha256": opaque_id,
                "prior_mass_exact": "1/2",
                "observation_sha256": (
                    reused_observation_digest
                    if index == 0 and reused_observation_digest is not None
                    else _sha(f"{stratum}-{index}-training-observation")
                ),
                "conditional_particle_count": 3,
                "range_content_sha256": _sha(f"{stratum}-{index}-range-content"),
                "range_build_sha256": _sha(f"{stratum}-{index}-range-build"),
            }
            for index, opaque_id in enumerate(opaque_ids)
        ],
    }
    iterations = 10
    traversals = 2 * iterations
    metadata = {
        "method": gate.MULTI_ROOT_SOLVER_METHOD,
        "chance_super_root": True,
        "super_root_sampling_contract": gate.SUPER_ROOT_SAMPLING_CONTRACT,
        "root_prior_manifest": root_prior,
        "root_prior_manifest_sha256": canonical_sha256(root_prior),
        "root_prior_normalized_exact": True,
        "root_count": 2,
        "traverser_schedule": "bb_then_btn_each_iteration",
        "alternating_updates": True,
        "regret_matching_plus": True,
        "regret_clip_scope": "once_per_infoset_after_traversal",
        "linear_averaging": True,
        "root_prior_sampled_once_per_traversal": True,
        "conditional_posterior_sampled_once_per_traversal": True,
        "root_probability_multiplied_after_sampling": False,
        "posterior_probability_multiplied_after_sampling": False,
        "chance_probability_multiplied_after_sampling": False,
        "joint_particle_weight_used_after_sampling": False,
        "policy_identity_contract": gate.POLICY_IDENTITY_CONTRACT,
        "policy_table_shared_across_all_roots": True,
        "table_key_type": "InfoSetKey",
        "table_key_contains_root_id": False,
        "table_key_contains_private_type_id": False,
        "table_key_contains_particle_commitment": False,
        "table_key_contains_remaining_cards": False,
        "strategy_serialization_contains_root_id": False,
        "strategy_serialization_contains_hidden_particle": False,
        "strategy_fusion": False,
        "independent_per_root_solve": False,
        "shared_across_roots_infoset_count": 1,
        "compatible_full_card_adapters": True,
        "full_card": True,
        "full_card_policy_promoted": False,
        "promotion_eligible": False,
        "runtime_integrated": False,
        "hu_exact": False,
        "exact_exploitability_computed": False,
        "iterations": iterations,
        "traversals": traversals,
        "seed": seed,
        "max_infosets": 100_000,
        "average_strategy_sha256": profile_sha,
        "position_contract_version": "bb_first_v1",
    }
    sampling = {
        "traversals": traversals,
        "traversals_by_actor": {"bb": iterations, "btn": iterations},
        "super_root_samples": traversals,
        "conditional_root_posterior_samples": traversals,
        "root_samples_by_opaque_id": {
            opaque_ids[0]: iterations,
            opaque_ids[1]: iterations,
        },
        "distinct_root_adapters_sampled": 2,
        "infosets_created": 1,
    }
    return (
        SimpleNamespace(
            metadata=metadata,
            sampling_stats=sampling,
            average_strategy_json=profile_text,
            average_strategy_sha256=profile_sha,
        ),
        opaque_ids,
    )


def _fixed_point_binding(range_source_sha: str):
    binding = {
        "schema": T3_BB_LIKELIHOOD_SCHEMA,
        "route": "t3_bb",
        "consumer_root_actor": "btn",
        "consumer_phase": "t3_second",
        "method": T3_BB_LIKELIHOOD_METHOD,
        "promotion_eligible": True,
        "fixed_point_converged": True,
        "candidate_policy_artifact_sha256": _sha("fixed-point-candidate"),
        "candidate_policy_checkpoint_sha256": _sha("fixed-point-checkpoint"),
        "fixed_point_evidence_sha256": _sha("fixed-point-evidence"),
        "fixed_point_gate_result_sha256": _sha("fixed-point-result"),
        "fixed_point_config_sha256": _sha("fixed-point-config"),
        "range_builder_source_sha256": range_source_sha,
        "solver_manifest_sha256": _sha("fixed-point-solver"),
    }
    binding["binding_sha256"] = self_hash(binding, "binding_sha256")
    return binding


def _build_fixture(
    *,
    overlap=False,
    seed_overlap=False,
    observation_reuse=False,
    range_reuse=False,
    production_claim=True,
):
    range_source_sha = _sha("range-source")
    binding = _fixed_point_binding(range_source_sha)
    source_sha = _sha("multi-root-source")
    holdout_roots = [
        gate.build_holdout_root_record(
            f"holdout-{stratum}",
            stratum=stratum,
            observation_digest=_root_key(stratum).digest(),
            seat_swap_pair_id=f"holdout-pair-joker{stratum[-1]}",
        )
        for stratum in REQUIRED_STRATA
    ]

    strategies = {}
    training_identities = []
    for stratum_index, stratum in enumerate(REQUIRED_STRATA):
        result, opaque_ids = _fake_result(
            stratum,
            reused_observation_digest=(
                holdout_roots[0]["observation_digest"]
                if observation_reuse and stratum_index == 0
                else None
            ),
        )
        identities = {
            opaque_id: root_identity_commitment_sha256(
                f"training-{stratum}-{root_index}"
            )
            for root_index, opaque_id in enumerate(opaque_ids)
        }
        if overlap and stratum_index == 0:
            identities[opaque_ids[0]] = holdout_roots[0][
                "root_identity_commitment_sha256"
            ]
        training_identities.extend(identities.values())
        strategies[stratum] = gate.build_shared_multi_root_strategy_artifact(
            result,
            stratum=stratum,
            training_root_identity_by_opaque_id=identities,
            source_manifest_sha256=source_sha,
            range_builder_source_sha256=range_source_sha,
            t3_bb_likelihood_binding_sha256=binding["binding_sha256"],
        )
    bundle = gate.build_shared_multi_root_candidate_bundle(
        strategies, t3_bb_likelihood_binding=binding
    )
    partitions = {
        "training": gate.build_excluded_root_partition(
            "training", training_identities
        ),
        "calibration": gate.build_excluded_root_partition(
            "calibration",
            [
                root_identity_commitment_sha256("calibration-root-a"),
                root_identity_commitment_sha256("calibration-root-b"),
            ],
        ),
        "smoke": gate.build_excluded_root_partition(
            "smoke",
            [
                root_identity_commitment_sha256("smoke-root-a"),
                root_identity_commitment_sha256("smoke-root-b"),
            ],
        ),
    }
    partition_hashes = {
        purpose: partition["manifest_sha256"]
        for purpose, partition in partitions.items()
    }
    root_manifest = gate.build_holdout_root_manifest(
        holdout_roots,
        excluded_root_partition_sha256=partition_hashes,
    )
    evaluator = gate.build_holdout_evaluator_manifest(
        candidate_bundle_sha256=bundle["candidate_bundle_sha256"],
        root_manifest_sha256=root_manifest["root_manifest_sha256"],
        reference_policy_sha256=_sha("reference-policy"),
        source_manifest_sha256=_sha("holdout-evaluator-source"),
    )
    thresholds = {
        "min_independent_evaluation_seeds_per_root": 2,
        "min_roots_per_stratum": 1,
        "min_action_payoff_samples_per_action": 100,
        "max_action_payoff_standard_error": 0.1,
        "max_reference_action_payoff_standard_error": 0.1,
        "max_policy_reference_delta_standard_error": 0.1,
        "min_encountered_infoset_coverage": 1.0,
        "max_mean_ev_regret_score": 0.01,
        "max_p95_ev_regret_score": 0.01,
        "max_p99_ev_regret_score": 0.01,
        "paired_seat_swap_noninferiority_margin_score": 0.01,
        "max_runtime_ms_p95": 20.0,
        "max_runtime_ms_max": 20.0,
        "max_runtime_ms_total": 200.0,
    }
    config = gate.build_locked_shared_multi_root_strength_config(
        approved_candidate_bundle_sha256=bundle["candidate_bundle_sha256"],
        approved_holdout_root_manifest_sha256=root_manifest[
            "root_manifest_sha256"
        ],
        approved_evaluator_manifest_sha256=evaluator["manifest_sha256"],
        approved_excluded_root_partition_sha256=partition_hashes,
        thresholds=thresholds,
    )

    evaluation_seeds = (11, 202) if seed_overlap else (101, 202)
    rows = []
    for root in root_manifest["roots"]:
        artifact = bundle["strategies"][root["stratum"]]
        record = next(
            record
            for record in artifact["strategy_profile"]["records"]
            if record["infoset_digest"] == root["observation_digest"]
        )
        policy = {
            action["action_id"]: action["probability"]
            for action in record["actions"]
        }
        action_ids = sorted(policy)
        reference = {action_id: 1.0 / len(action_ids) for action_id in action_ids}
        payoffs = {
            action_id: 1.0 if index == 0 else 0.0
            for index, action_id in enumerate(action_ids)
        }
        reference_payoffs = {
            action_id: 0.5 if index == 0 else 0.0
            for index, action_id in enumerate(action_ids)
        }
        for seed_index, evaluation_seed in enumerate(evaluation_seeds):
            payoff_seed = 1_000 + 10 * root["visible_joker_count"] + seed_index
            policy_payoff = sum(
                policy[action_id] * payoffs[action_id] for action_id in action_ids
            )
            candidate_uniform_payoff = sum(
                reference[action_id] * payoffs[action_id]
                for action_id in action_ids
            )
            reference_payoff = sum(
                reference[action_id] * reference_payoffs[action_id]
                for action_id in action_ids
            )
            delta = policy_payoff - reference_payoff
            range_content_sha = _sha(f"holdout-{root['root_id']}-range-content")
            range_build_sha = _sha(f"holdout-{root['root_id']}-range-build")
            if range_reuse and root == root_manifest["roots"][0]:
                training_root = artifact["training_roots"][0]
                range_content_sha = training_root["range_content_sha256"]
                range_build_sha = training_root["range_build_sha256"]
            row = {
                "run_id": f"run-{root['root_id']}-{evaluation_seed}",
                "root_id": root["root_id"],
                "root_commitment_sha256": root["root_commitment_sha256"],
                "root_identity_commitment_sha256": root[
                    "root_identity_commitment_sha256"
                ],
                "stratum": root["stratum"],
                "actor": root["actor"],
                "visible_joker_count": root["visible_joker_count"],
                "seat_swap_pair_id": root["seat_swap_pair_id"],
                "evaluation_seed": evaluation_seed,
                "payoff_sample_seed": payoff_seed,
                "candidate_bundle_sha256": bundle["candidate_bundle_sha256"],
                "candidate_strategy_artifact_sha256": artifact["artifact_sha256"],
                "candidate_strategy_sha256": artifact["average_strategy_sha256"],
                "policy_infoset_digest": root["observation_digest"],
                "range_content_sha256": range_content_sha,
                "range_build_sha256": range_build_sha,
                "policy_action_distribution": policy,
                "reference_policy_sha256": evaluator["reference_policy_sha256"],
                "reference_action_distribution": reference,
                "action_payoff_estimates": payoffs,
                "action_payoff_sample_counts": {
                    action_id: 100 for action_id in action_ids
                },
                "action_payoff_aggregates": {
                    action_id: {
                        "count": 100,
                        "sum": 100.0 * payoffs[action_id],
                        "sum_squares": 100.0 * payoffs[action_id] ** 2,
                    }
                    for action_id in action_ids
                },
                "action_payoff_standard_errors": {
                    action_id: 0.0 for action_id in action_ids
                },
                "max_action_payoff_standard_error": 0.0,
                "reference_action_payoff_estimates": reference_payoffs,
                "reference_action_payoff_sample_counts": {
                    action_id: 100 for action_id in action_ids
                },
                "reference_action_payoff_aggregates": {
                    action_id: {
                        "count": 100,
                        "sum": 100.0 * reference_payoffs[action_id],
                        "sum_squares": 100.0 * reference_payoffs[action_id] ** 2,
                    }
                    for action_id in action_ids
                },
                "reference_action_payoff_standard_errors": {
                    action_id: 0.0 for action_id in action_ids
                },
                "max_reference_action_payoff_standard_error": 0.0,
                "policy_payoff_estimate": policy_payoff,
                "candidate_continuation_uniform_root_payoff_estimate": (
                    candidate_uniform_payoff
                ),
                "reference_payoff_estimate": reference_payoff,
                "policy_reference_paired_aggregate": {
                    "count": 100,
                    "sum": 100.0 * delta,
                    "sum_squares": 100.0 * delta**2,
                },
                "policy_reference_delta_estimate": delta,
                "policy_reference_delta_standard_error": 0.0,
                "best_action_payoff_estimate": max(payoffs.values()),
                "ev_regret_estimate": max(0.0, max(payoffs.values()) - policy_payoff),
                "eligible_infoset_digests": [root["observation_digest"]],
                "encountered_infoset_digests": [root["observation_digest"]],
                "eligible_infoset_count": 1,
                "encountered_infoset_count": 1,
                "encountered_infoset_coverage": 1.0,
                "audits": {field: 0 for field in REQUIRED_AUDITS},
                "runtime_ms": 10.0,
                "exact_exploitability_computed": False,
            }
            row["row_sha256"] = self_hash(row, "row_sha256")
            rows.append(row)
    evidence = gate.build_shared_multi_root_strength_evidence(
        config=config,
        candidate_bundle=bundle,
        root_manifest=root_manifest,
        excluded_root_partitions=partitions,
        evaluator_manifest=evaluator,
        raw_holdout_rows=rows,
        production_promotion_claim=production_claim,
    )
    return config, evidence


def _rehash_evidence(evidence):
    evidence["artifact_sha256"] = self_hash(evidence, "artifact_sha256")


def test_dynamic_result_adapter_and_six_stratum_bundle_are_nonpromoting():
    config, evidence = _build_fixture(production_claim=False)

    for stratum, artifact in evidence["candidate_bundle"]["strategies"].items():
        assert artifact["stratum"] == stratum
        assert artifact["producer_contract"]["policy_table_shared_across_all_roots"] is True
        assert artifact["producer_contract"]["strategy_fusion"] is False
        assert artifact["producer_contract"]["table_key_contains_root_id"] is False
        assert artifact["promotion_eligible"] is False
        assert artifact["exact_exploitability_computed"] is False
    result = gate.validate_shared_multi_root_strength_evidence(
        evidence, config=config
    )
    assert result["independent_holdout_replayed"] is True
    assert result["strategic_strength_evaluated"] is True
    assert result["exact_exploitability_computed"] is False
    assert result["passed"] is False
    assert result["promotion_eligible"] is False
    assert result["full_card_policy_promoted"] is False
    assert result["status"] == gate.ALGORITHM_VALIDATION_ONLY_STATUS
    assert result["failures"] == [
        "production promotion claim is false",
        gate.ALGORITHM_VALIDATION_ONLY_FAILURE,
    ]


def test_explicit_locked_evaluation_remains_algorithm_validation_only():
    config, evidence = _build_fixture()

    result = gate.validate_shared_multi_root_strength_evidence(
        evidence, config=config
    )

    assert result["passed"] is False
    assert result["promotion_eligible"] is False
    assert result["full_card_policy_promoted"] is False
    assert result["shared_multi_root_strength_promoted"] is False
    assert result["status"] == gate.ALGORITHM_VALIDATION_ONLY_STATUS
    assert result["exact_exploitability_computed"] is False
    assert result["failures"] == [gate.ALGORITHM_VALIDATION_ONLY_FAILURE]
    assert set(result["derived_metrics"]["strata"]) == set(REQUIRED_STRATA)
    assert result["derived_metrics"]["global"][
        "candidate_holdout_root_overlap_count"
    ] == 0
    assert result["derived_metrics"]["global"][
        "max_action_payoff_standard_error"
    ] == 0.0
    assert result["derived_metrics"]["global"][
        "max_reference_action_payoff_standard_error"
    ] == 0.0
    assert result["derived_metrics"]["global"][
        "max_policy_reference_delta_standard_error"
    ] == 0.0
    assert all(
        values["max_action_payoff_standard_error"] == 0.0
        for values in result["derived_metrics"]["strata"].values()
    )
    row = evidence["raw_holdout_rows"][0]
    assert row["reference_payoff_estimate"] != row[
        "candidate_continuation_uniform_root_payoff_estimate"
    ]


def test_point_estimates_without_raw_moments_fail_closed():
    config, evidence = _build_fixture()
    row = evidence["raw_holdout_rows"][0]
    del row["action_payoff_aggregates"]
    del row["action_payoff_standard_errors"]
    del row["max_action_payoff_standard_error"]
    row["row_sha256"] = self_hash(row, "row_sha256")
    _rehash_evidence(evidence)

    result = gate.validate_shared_multi_root_strength_evidence(
        evidence, config=config
    )

    assert result["passed"] is False
    assert result["promotion_eligible"] is False
    assert result["strategic_strength_evaluated"] is False
    assert any(
        "action_payoff_aggregates" in failure
        and "action_payoff_standard_errors" in failure
        for failure in result["failures"]
    )


def test_reference_point_estimate_without_raw_moments_fails_closed():
    config, evidence = _build_fixture()
    row = evidence["raw_holdout_rows"][0]
    for field in (
        "reference_action_payoff_aggregates",
        "reference_action_payoff_sample_counts",
        "reference_action_payoff_estimates",
        "reference_action_payoff_standard_errors",
        "max_reference_action_payoff_standard_error",
    ):
        del row[field]
    row["row_sha256"] = self_hash(row, "row_sha256")
    _rehash_evidence(evidence)

    result = gate.validate_shared_multi_root_strength_evidence(
        evidence, config=config
    )

    assert result["passed"] is False
    assert result["strategic_strength_evaluated"] is False
    assert any(
        "reference_action_payoff_aggregates" in failure
        and "reference_action_payoff_estimates" in failure
        for failure in result["failures"]
    )


def test_tampered_reference_raw_mean_fails_rederivation():
    config, evidence = _build_fixture()
    row = evidence["raw_holdout_rows"][0]
    action_id = sorted(row["reference_action_payoff_aggregates"])[0]
    aggregate = row["reference_action_payoff_aggregates"][action_id]
    old_sum = aggregate["sum"]
    shift = 0.01
    aggregate["sum"] += aggregate["count"] * shift
    aggregate["sum_squares"] += (
        2.0 * shift * old_sum + aggregate["count"] * shift**2
    )
    row["row_sha256"] = self_hash(row, "row_sha256")
    _rehash_evidence(evidence)

    result = gate.validate_shared_multi_root_strength_evidence(
        evidence, config=config
    )

    assert result["passed"] is False
    assert result["strategic_strength_evaluated"] is False
    assert any("mean/payoff mismatch" in failure for failure in result["failures"])


def test_tampered_paired_delta_moments_cannot_replace_policy_reference_delta():
    config, evidence = _build_fixture()
    row = evidence["raw_holdout_rows"][0]
    aggregate = row["policy_reference_paired_aggregate"]
    replacement_delta = row["policy_reference_delta_estimate"] + 0.01
    aggregate["sum"] = aggregate["count"] * replacement_delta
    aggregate["sum_squares"] = aggregate["count"] * replacement_delta**2
    row["policy_reference_delta_estimate"] = replacement_delta
    row["policy_reference_delta_standard_error"] = 0.0
    row["row_sha256"] = self_hash(row, "row_sha256")
    _rehash_evidence(evidence)

    result = gate.validate_shared_multi_root_strength_evidence(
        evidence, config=config
    )

    assert result["passed"] is False
    assert result["strategic_strength_evaluated"] is False
    assert any(
        "policy_reference_delta_estimate: paired raw-derived mismatch" in failure
        for failure in result["failures"]
    )


def test_excessive_raw_derived_payoff_uncertainty_fails_threshold():
    config, evidence = _build_fixture()
    row = evidence["raw_holdout_rows"][0]
    action_id = sorted(row["action_payoff_aggregates"])[0]
    aggregate = row["action_payoff_aggregates"][action_id]
    aggregate["sum_squares"] += 400.0
    standard_error = (400.0 / 99.0 / 100.0) ** 0.5
    row["action_payoff_standard_errors"][action_id] = standard_error
    row["max_action_payoff_standard_error"] = standard_error
    row["row_sha256"] = self_hash(row, "row_sha256")
    _rehash_evidence(evidence)

    result = gate.validate_shared_multi_root_strength_evidence(
        evidence, config=config
    )

    assert result["passed"] is False
    assert result["promotion_eligible"] is False
    assert result["strategic_strength_evaluated"] is True
    assert result["derived_metrics"]["global"][
        "max_action_payoff_standard_error"
    ] == pytest.approx(standard_error)
    assert any(
        failure
        == "derived_metrics.global.max_action_payoff_standard_error: threshold failed"
        for failure in result["failures"]
    )


def test_excessive_reference_and_paired_delta_uncertainty_fail_thresholds():
    config, evidence = _build_fixture()
    row = evidence["raw_holdout_rows"][0]
    action_id = sorted(row["reference_action_payoff_aggregates"])[0]
    row["reference_action_payoff_aggregates"][action_id]["sum_squares"] += 400.0
    reference_standard_error = (400.0 / 99.0 / 100.0) ** 0.5
    row["reference_action_payoff_standard_errors"][
        action_id
    ] = reference_standard_error
    row[
        "max_reference_action_payoff_standard_error"
    ] = reference_standard_error
    row["policy_reference_paired_aggregate"]["sum_squares"] += 400.0
    delta_standard_error = (400.0 / 99.0 / 100.0) ** 0.5
    row["policy_reference_delta_standard_error"] = delta_standard_error
    row["row_sha256"] = self_hash(row, "row_sha256")
    _rehash_evidence(evidence)

    result = gate.validate_shared_multi_root_strength_evidence(
        evidence, config=config
    )

    assert result["passed"] is False
    assert result["strategic_strength_evaluated"] is True
    assert result["derived_metrics"]["global"][
        "max_reference_action_payoff_standard_error"
    ] == pytest.approx(reference_standard_error)
    assert result["derived_metrics"]["global"][
        "max_policy_reference_delta_standard_error"
    ] == pytest.approx(delta_standard_error)
    assert any(
        failure
        == "derived_metrics.global.max_reference_action_payoff_standard_error: threshold failed"
        for failure in result["failures"]
    )
    assert any(
        failure
        == "derived_metrics.global.max_policy_reference_delta_standard_error: threshold failed"
        for failure in result["failures"]
    )


def test_candidate_training_and_holdout_root_reuse_fails_closed():
    config, evidence = _build_fixture(overlap=True)

    result = gate.validate_shared_multi_root_strength_evidence(
        evidence, config=config
    )

    assert result["passed"] is False
    assert any("reused by candidate training" in failure for failure in result["failures"])
    assert any("overlap training partition" in failure for failure in result["failures"])


def test_relabelled_training_observation_and_range_are_not_holdout():
    config, evidence = _build_fixture(observation_reuse=True)
    result = gate.validate_shared_multi_root_strength_evidence(
        evidence, config=config
    )
    assert result["passed"] is False
    assert any("observations relabelled" in failure for failure in result["failures"])

    config, evidence = _build_fixture(range_reuse=True)
    result = gate.validate_shared_multi_root_strength_evidence(
        evidence, config=config
    )
    assert result["passed"] is False
    assert any("training range" in failure for failure in result["failures"])


def test_strategy_training_content_must_match_root_prior():
    _config, evidence = _build_fixture()
    artifact = copy.deepcopy(
        evidence["candidate_bundle"]["strategies"]["bb_joker0"]
    )
    artifact["training_roots"][0]["range_content_sha256"] = _sha(
        "relabelled-training-range"
    )
    artifact["artifact_sha256"] = self_hash(artifact, "artifact_sha256")

    with pytest.raises(ValueError, match="root-prior content mismatch"):
        gate.verify_shared_multi_root_strategy_artifact(artifact)


def test_candidate_training_seed_cannot_be_reused_for_holdout():
    config, evidence = _build_fixture(seed_overlap=True)

    result = gate.validate_shared_multi_root_strength_evidence(
        evidence, config=config
    )

    assert result["passed"] is False
    assert any("training seed reused" in failure for failure in result["failures"])


def test_holdout_policy_row_must_equal_candidate_strategy_content():
    config, evidence = _build_fixture()
    row = evidence["raw_holdout_rows"][0]
    action_ids = sorted(row["policy_action_distribution"])
    row["policy_action_distribution"][action_ids[0]] = 0.0
    row["policy_action_distribution"][action_ids[1]] = 1.0
    row["row_sha256"] = self_hash(row, "row_sha256")
    _rehash_evidence(evidence)

    result = gate.validate_shared_multi_root_strength_evidence(
        evidence, config=config
    )

    assert result["passed"] is False
    assert any("candidate content mismatch" in failure for failure in result["failures"])


def test_strategy_fusion_or_hidden_root_policy_key_is_rejected():
    _config, evidence = _build_fixture()
    artifact = copy.deepcopy(
        evidence["candidate_bundle"]["strategies"]["bb_joker0"]
    )
    artifact["producer_contract"]["strategy_fusion"] = True
    artifact["artifact_sha256"] = self_hash(artifact, "artifact_sha256")
    with pytest.raises(ValueError, match="strategy_fusion: mismatch"):
        gate.verify_shared_multi_root_strategy_artifact(artifact)

    artifact = copy.deepcopy(
        evidence["candidate_bundle"]["strategies"]["bb_joker0"]
    )
    artifact["producer_contract"]["table_key_contains_root_id"] = True
    artifact["artifact_sha256"] = self_hash(artifact, "artifact_sha256")
    with pytest.raises(ValueError, match="table_key_contains_root_id: mismatch"):
        gate.verify_shared_multi_root_strategy_artifact(artifact)


def test_unpromoted_or_unconverged_fixed_point_binding_is_rejected():
    _config, evidence = _build_fixture()
    strategies = evidence["candidate_bundle"]["strategies"]
    binding = copy.deepcopy(evidence["candidate_bundle"]["t3_bb_likelihood_binding"])
    binding["promotion_eligible"] = False
    binding["fixed_point_converged"] = False
    binding["binding_sha256"] = self_hash(binding, "binding_sha256")

    with pytest.raises(ValueError, match="promotion_eligible"):
        gate.build_shared_multi_root_candidate_bundle(
            strategies, t3_bb_likelihood_binding=binding
        )


def test_exact_exploitability_claim_is_not_accepted_as_strength_proof():
    config, evidence = _build_fixture()
    evidence["exact_exploitability_computed"] = True
    _rehash_evidence(evidence)

    result = gate.validate_shared_multi_root_strength_evidence(
        evidence, config=config
    )

    assert result["passed"] is False
    assert result["exact_exploitability_computed"] is False
    assert any(
        "exact_exploitability_computed" in failure
        for failure in result["failures"]
    )


def test_missing_joker_role_stratum_is_rejected():
    _config, evidence = _build_fixture()
    bundle = copy.deepcopy(evidence["candidate_bundle"])
    del bundle["strategies"]["btn_joker2"]
    bundle["candidate_bundle_sha256"] = self_hash(
        bundle, "candidate_bundle_sha256"
    )

    with pytest.raises(ValueError, match="exact six strata required"):
        gate.verify_shared_multi_root_candidate_bundle(bundle)


def test_malformed_noncanonical_evidence_returns_failure_instead_of_throwing():
    config, _evidence = _build_fixture()
    malformed = {"schema": gate.EVIDENCE_SCHEMA, "not_json": {object()}}

    result = gate.validate_shared_multi_root_strength_evidence(
        malformed, config=config
    )

    assert result["passed"] is False
    assert result["promotion_eligible"] is False
    assert result["exact_exploitability_computed"] is False
    assert any("finite canonical" in failure for failure in result["failures"])
