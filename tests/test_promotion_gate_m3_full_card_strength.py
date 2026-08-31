from __future__ import annotations

from ai.tutor.promotion_gate_m3_full_card_strength import (
    BEHAVIOR_SCHEMA,
    BEHAVIOR_MODEL_TYPE,
    CHECKPOINT_SCHEMA,
    CONFIG_SCHEMA,
    EVIDENCE_KIND,
    EVIDENCE_SCHEMA,
    GATE_ID,
    POSITION_CONTRACT_VERSION,
    REQUIRED_STRATA,
    ROOT_MANIFEST_SCHEMA,
    ROOT_PARTITION_SCHEMA,
    RULESET,
    SCOPE,
    SOLVER_ADAPTER,
    SOLVER_METHOD,
    SOLVER_SCHEMA,
    SOURCE_SCHEMA,
    T3_BB_LIKELIHOOD_METHOD,
    T3_BB_LIKELIHOOD_SCHEMA,
    T3_FULL_CARD_RANGE_SOURCE_PATH,
    canonical_sha256,
    derive_strength_metrics,
    root_commitment_sha256,
    root_identity_commitment_sha256,
    self_hash,
    validate_promotion_evidence_m3_full_card_strength,
)


def _sha(label: str) -> str:
    return canonical_sha256({"label": label})


def _partition(name: str) -> dict:
    value = {
        "schema": ROOT_PARTITION_SCHEMA,
        "purpose": name,
        "root_commitments": sorted(
            [
                root_identity_commitment_sha256(f"{name}-root-0"),
                root_identity_commitment_sha256(f"{name}-root-1"),
            ]
        ),
    }
    value["manifest_sha256"] = self_hash(value, "manifest_sha256")
    return value


def _build_valid() -> tuple[dict, dict]:
    partitions = {name: _partition(name) for name in ("training", "calibration", "smoke")}
    partition_hashes = {
        name: value["manifest_sha256"] for name, value in partitions.items()
    }

    range_builder_sha = _sha("range-builder-source")
    source = {
        "schema": SOURCE_SCHEMA,
        "files": [
            {"path": "ai/engine/scoring.py", "sha256": _sha("scoring-source")},
            {
                "path": "ai/tutor/t3_hu_full_card_mccfr.py",
                "sha256": _sha("mccfr-source"),
            },
            {
                "path": T3_FULL_CARD_RANGE_SOURCE_PATH,
                "sha256": range_builder_sha,
            },
        ],
    }
    source_sha = canonical_sha256(source)
    solver = {
        "schema": SOLVER_SCHEMA,
        "method": SOLVER_METHOD,
        "adapter": SOLVER_ADAPTER,
        "ruleset": RULESET,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "turns": [3, 4],
        "actors": ["bb", "btn"],
        "physical_joker_ids": ["X1", "X2"],
        "information_model": "public_only_no_opponent_private_cards",
        "strategy_fusion": False,
        "full_card": True,
        "hu_exact": False,
        "exact_exploitability_computed": False,
        "source_manifest_sha256": source_sha,
    }
    solver_sha = canonical_sha256(solver)

    t3_bb_likelihood = {
        "schema": T3_BB_LIKELIHOOD_SCHEMA,
        "route": "t3_bb",
        "consumer_root_actor": "btn",
        "consumer_phase": "t3_second",
        "method": T3_BB_LIKELIHOOD_METHOD,
        "promotion_eligible": True,
        "fixed_point_converged": True,
        "candidate_policy_artifact_sha256": _sha("candidate-policy-artifact"),
        "candidate_policy_checkpoint_sha256": _sha("candidate-policy-checkpoint"),
        "fixed_point_evidence_sha256": _sha("fixed-point-evidence"),
        "fixed_point_gate_result_sha256": _sha("fixed-point-gate-result"),
        "fixed_point_config_sha256": _sha("fixed-point-config"),
        "range_builder_source_sha256": range_builder_sha,
        "solver_manifest_sha256": solver_sha,
    }
    t3_bb_likelihood["binding_sha256"] = self_hash(
        t3_bb_likelihood, "binding_sha256"
    )
    behavior = {
        "schema": BEHAVIOR_SCHEMA,
        "model_type": BEHAVIOR_MODEL_TYPE,
        "calibrated": True,
        "calibration_method": "t1_t2_observed_action_nll_temperature",
        "calibration_dataset_kind": "observed_full_trace_actions",
        "root_split": "root_disjoint_train_calibration_test",
        "promotion_eligible": True,
        "role_temperatures": {
            "t1_bb": "73/100",
            "t1_btn": "3/4",
            "t2_bb": "4/5",
            "t2_btn": "17/20",
        },
        "calibration_artifact_sha256": _sha("calibration-artifact"),
        "calibration_gate_config_sha256": _sha("calibration-gate-config"),
        "calibration_gate_result_sha256": _sha("calibration-gate-result"),
        "role_model_bindings": {
            role: {
                "checkpoint_sha256": _sha(f"{role}-checkpoint"),
                "model_sha256": _sha(f"{role}-model"),
                "row_extractor_sha256": _sha(f"{role}-extractor"),
                "adapter_source_sha256": _sha(f"{role}-adapter"),
            }
            for role in ("t1_bb", "t1_btn", "t2_bb", "t2_btn")
        },
        "t3_bb_likelihood_binding": t3_bb_likelihood,
        "training_root_partition_sha256": partition_hashes["training"],
        "calibration_root_partition_sha256": partition_hashes["calibration"],
    }
    behavior_sha = canonical_sha256(behavior)

    roots = []
    for name in REQUIRED_STRATA:
        actor, joker_text = name.split("_joker")
        root = {
            "root_id": f"root-{name}",
            "stratum": name,
            "actor": actor,
            "visible_joker_count": int(joker_text),
            "observation_digest": _sha(f"observation-{name}"),
            "seat_swap_pair_id": f"paired-joker{joker_text}",
        }
        root["root_identity_commitment_sha256"] = root_identity_commitment_sha256(
            root["root_id"]
        )
        root["root_commitment_sha256"] = root_commitment_sha256(root)
        roots.append(root)
    root_manifest = {
        "schema": ROOT_MANIFEST_SCHEMA,
        "purpose": "independent_promotion_holdout",
        "locked_before_evaluation": True,
        "ruleset": RULESET,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "physical_joker_ids": ["X1", "X2"],
        "excluded_root_partition_sha256": partition_hashes,
        "roots": roots,
    }
    root_manifest["root_manifest_sha256"] = self_hash(
        root_manifest, "root_manifest_sha256"
    )
    root_manifest_sha = root_manifest["root_manifest_sha256"]

    thresholds = {
        "min_independent_seeds_per_stratum": 2,
        "min_roots_per_stratum": 1,
        "min_encountered_infoset_coverage": 0.95,
        "max_policy_tv_drift": 0.05,
        "max_mean_ev_regret_score": 0.20,
        "max_p95_ev_regret_score": 0.20,
        "max_p99_ev_regret_score": 0.20,
        "paired_seat_swap_noninferiority_margin_score": 0.01,
        "max_runtime_ms_p95": 20.0,
        "max_runtime_ms_max": 20.0,
        "max_runtime_ms_total": 200.0,
    }
    config = {
        "schema": CONFIG_SCHEMA,
        "gate_id": GATE_ID,
        "approved_calibrated_behavior_sha256": behavior_sha,
        "approved_solver_manifest_sha256": solver_sha,
        "approved_source_manifest_sha256": source_sha,
        "approved_holdout_root_manifest_sha256": root_manifest_sha,
        "approved_excluded_root_partition_sha256": partition_hashes,
        "thresholds": thresholds,
    }
    config_sha = canonical_sha256(config)

    checkpoints: dict[str, dict] = {}
    checkpoint_by_key: dict[tuple[str, int], tuple[str, str]] = {}
    for name in REQUIRED_STRATA:
        for seed in (101, 202):
            checkpoint_sha = _sha(f"checkpoint-file-{name}-{seed}")
            manifest = {
                "schema": CHECKPOINT_SCHEMA,
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_manifest_sha256": "",
                "completed": True,
                "stratum": name,
                "solver_seed": seed,
                "iterations": 2_000,
                "solver_manifest_sha256": solver_sha,
                "source_manifest_sha256": source_sha,
                "calibrated_behavior_sha256": behavior_sha,
                "root_manifest_sha256": root_manifest_sha,
                "exact_exploitability_computed": False,
            }
            manifest["checkpoint_manifest_sha256"] = self_hash(
                manifest, "checkpoint_manifest_sha256"
            )
            checkpoints[checkpoint_sha] = manifest
            checkpoint_by_key[(name, seed)] = (
                checkpoint_sha,
                manifest["checkpoint_manifest_sha256"],
            )

    rows = []
    for root in roots:
        for seed_index, seed in enumerate((101, 202)):
            checkpoint_sha, checkpoint_manifest_sha = checkpoint_by_key[
                (root["stratum"], seed)
            ]
            policy_a = 0.90 - 0.02 * seed_index
            policy = {"place-a": policy_a, "place-b": 1.0 - policy_a}
            reference = {"place-a": 0.80, "place-b": 0.20}
            action_payoffs = {"place-a": 1.0, "place-b": 0.0}
            eligible = sorted([_sha(f"eligible-{root['root_id']}-0"), _sha(f"eligible-{root['root_id']}-1")])
            policy_payoff = policy_a
            reference_payoff = 0.8
            row = {
                "run_id": f"run-{root['root_id']}-{seed}",
                "root_id": root["root_id"],
                "root_commitment_sha256": root["root_commitment_sha256"],
                "root_identity_commitment_sha256": root[
                    "root_identity_commitment_sha256"
                ],
                "stratum": root["stratum"],
                "actor": root["actor"],
                "visible_joker_count": root["visible_joker_count"],
                "seat_swap_pair_id": root["seat_swap_pair_id"],
                "solver_seed": seed,
                "payoff_sample_seed": 1_000 + seed,
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_manifest_sha256": checkpoint_manifest_sha,
                "calibrated_behavior_sha256": behavior_sha,
                "reference_policy_sha256": behavior_sha,
                "source_manifest_sha256": source_sha,
                "solver_manifest_sha256": solver_sha,
                "root_manifest_sha256": root_manifest_sha,
                "exact_exploitability_computed": False,
                "audits": {
                    "legal_action_failures": 0,
                    "hidden_information_leak_failures": 0,
                    "opponent_private_card_policy_input_failures": 0,
                    "hidden_discard_policy_key_failures": 0,
                },
                "eligible_infoset_digests": eligible,
                "encountered_infoset_digests": list(eligible),
                "eligible_infoset_count": 2,
                "encountered_infoset_count": 2,
                "encountered_infoset_coverage": 1.0,
                "root_action_distribution": policy,
                "reference_action_distribution": reference,
                "action_payoff_estimates": action_payoffs,
                "policy_payoff_estimate": policy_payoff,
                "reference_payoff_estimate": reference_payoff,
                "best_action_payoff_estimate": 1.0,
                "ev_regret_estimate": 1.0 - policy_payoff,
                "runtime_ms": 10.0 + seed_index,
            }
            rows.append(row)

    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "gate_id": GATE_ID,
        "scope": SCOPE,
        "evidence_kind": EVIDENCE_KIND,
        "gate_config_sha256": config_sha,
        "exact_exploitability_computed": False,
        "calibrated_behavior_manifest": behavior,
        "calibrated_behavior_sha256": behavior_sha,
        "source_manifest": source,
        "source_manifest_sha256": source_sha,
        "solver_manifest": solver,
        "solver_manifest_sha256": solver_sha,
        "excluded_root_partitions": partitions,
        "root_manifest": root_manifest,
        "checkpoint_manifests": checkpoints,
        "raw_run_rows": rows,
        "published_summary": {},
    }
    evidence["published_summary"] = derive_strength_metrics(evidence)
    evidence["artifact_sha256"] = self_hash(evidence, "artifact_sha256")
    return evidence, config


def _rehash_artifact(evidence: dict) -> None:
    evidence["artifact_sha256"] = self_hash(evidence, "artifact_sha256")


def _rebind_config(evidence: dict, config: dict) -> None:
    evidence["gate_config_sha256"] = canonical_sha256(config)
    _rehash_artifact(evidence)


def test_synthetic_independent_strength_evidence_passes() -> None:
    evidence, config = _build_valid()
    result = validate_promotion_evidence_m3_full_card_strength(
        evidence, config=config
    )

    assert result["passed"] is True, result["failures"]
    assert result["m3_full_card_strength_promoted"] is True
    assert result["full_card_policy_promoted"] is True
    assert result["exact_exploitability_computed"] is False
    assert result["derived_metrics"]["global"]["policy_tv_comparison_count"] == 6
    assert result["derived_metrics"]["global"]["paired_seat_swap_comparison_count"] == 6


def test_tampered_raw_payoff_fails_even_when_artifact_hash_is_recomputed() -> None:
    evidence, config = _build_valid()
    evidence["raw_run_rows"][0]["policy_payoff_estimate"] = 123.0
    _rehash_artifact(evidence)

    result = validate_promotion_evidence_m3_full_card_strength(
        evidence, config=config
    )

    assert result["passed"] is False
    assert any("policy_payoff_estimate: raw-derived mismatch" in item for item in result["failures"])


def test_missing_stratum_root_and_rows_fail_closed() -> None:
    evidence, config = _build_valid()
    removed = "bb_joker2"
    evidence["root_manifest"]["roots"] = [
        root for root in evidence["root_manifest"]["roots"] if root["stratum"] != removed
    ]
    evidence["raw_run_rows"] = [
        row for row in evidence["raw_run_rows"] if row["stratum"] != removed
    ]
    evidence["checkpoint_manifests"] = {
        sha: manifest
        for sha, manifest in evidence["checkpoint_manifests"].items()
        if manifest["stratum"] != removed
    }
    evidence["root_manifest"]["root_manifest_sha256"] = self_hash(
        evidence["root_manifest"], "root_manifest_sha256"
    )
    _rehash_artifact(evidence)

    result = validate_promotion_evidence_m3_full_card_strength(
        evidence, config=config
    )

    assert result["passed"] is False
    assert any(f"{removed} needs" in item for item in result["failures"])


def test_holdout_overlap_with_smoke_partition_fails() -> None:
    evidence, config = _build_valid()
    holdout = evidence["root_manifest"]["roots"][0][
        "root_identity_commitment_sha256"
    ]
    evidence["excluded_root_partitions"]["smoke"]["root_commitments"][0] = holdout
    evidence["excluded_root_partitions"]["smoke"]["root_commitments"].sort()
    _rehash_artifact(evidence)

    result = validate_promotion_evidence_m3_full_card_strength(
        evidence, config=config
    )

    assert result["passed"] is False
    assert any("overlap smoke partition" in item for item in result["failures"])


def test_uncalibrated_behavior_fails() -> None:
    evidence, config = _build_valid()
    evidence["calibrated_behavior_manifest"]["calibrated"] = False
    _rehash_artifact(evidence)

    result = validate_promotion_evidence_m3_full_card_strength(
        evidence, config=config
    )

    assert result["passed"] is False
    assert any("calibrated_behavior_manifest.calibrated" in item for item in result["failures"])


def test_behavior_requires_all_four_role_temperatures() -> None:
    evidence, config = _build_valid()
    del evidence["calibrated_behavior_manifest"]["role_temperatures"]["t2_btn"]
    _rehash_artifact(evidence)

    result = validate_promotion_evidence_m3_full_card_strength(
        evidence, config=config
    )

    assert result["passed"] is False
    assert any("role_temperatures: exact" in item for item in result["failures"])


def test_behavior_requires_t3_bb_likelihood_route_for_btn_root() -> None:
    evidence, config = _build_valid()
    del evidence["calibrated_behavior_manifest"]["t3_bb_likelihood_binding"]
    _rehash_artifact(evidence)

    result = validate_promotion_evidence_m3_full_card_strength(
        evidence, config=config
    )

    assert result["passed"] is False
    assert any(
        "calibrated_behavior_manifest.t3_bb_likelihood_binding: object required"
        in item
        for item in result["failures"]
    )


def test_t3_bb_prior_or_unconverged_fixed_point_cannot_promote() -> None:
    for field, replacement, expected in (
        ("method", "frozen_teacher_ranking_prior_v1", ".method:"),
        ("fixed_point_converged", False, ".fixed_point_converged:"),
    ):
        evidence, config = _build_valid()
        binding = evidence["calibrated_behavior_manifest"][
            "t3_bb_likelihood_binding"
        ]
        binding[field] = replacement
        binding["binding_sha256"] = self_hash(binding, "binding_sha256")
        _rehash_artifact(evidence)

        result = validate_promotion_evidence_m3_full_card_strength(
            evidence, config=config
        )

        assert result["passed"] is False
        assert any(
            f"t3_bb_likelihood_binding{expected}" in item
            for item in result["failures"]
        )


def test_t3_bb_likelihood_must_bind_evaluated_solver_and_range_builder() -> None:
    for field, expected in (
        ("solver_manifest_sha256", "solver binding mismatch"),
        ("range_builder_source_sha256", "source manifest binding mismatch"),
    ):
        evidence, config = _build_valid()
        binding = evidence["calibrated_behavior_manifest"][
            "t3_bb_likelihood_binding"
        ]
        binding[field] = _sha(f"wrong-{field}")
        binding["binding_sha256"] = self_hash(binding, "binding_sha256")
        _rehash_artifact(evidence)

        result = validate_promotion_evidence_m3_full_card_strength(
            evidence, config=config
        )

        assert result["passed"] is False
        assert any(expected in item for item in result["failures"])


def test_behavior_rejects_noncanonical_temperature_and_incomplete_binding() -> None:
    evidence, config = _build_valid()
    behavior = evidence["calibrated_behavior_manifest"]
    behavior["role_temperatures"]["t1_bb"] = "730000/1000000"
    del behavior["role_model_bindings"]["t2_btn"]["adapter_source_sha256"]
    _rehash_artifact(evidence)

    result = validate_promotion_evidence_m3_full_card_strength(
        evidence, config=config
    )

    assert result["passed"] is False
    assert any("reduced positive rational required" in item for item in result["failures"])
    assert any("exact checkpoint/model/extractor/adapter hashes" in item for item in result["failures"])


def test_smoke_artifact_cannot_be_reused_for_strength_promotion() -> None:
    _evidence, config = _build_valid()
    smoke = {
        "schema": "ofc_m3_full_card_smoke_evidence/v1",
        "gate_id": "promotion_gate_m3_full_card_smoke",
        "pass_scope": "six_stratum_execution_smoke_only",
        "summary": {"execution_smoke_passed": True, "m3_promotion_passed": False},
    }

    result = validate_promotion_evidence_m3_full_card_strength(smoke, config=config)

    assert result["passed"] is False
    assert result["full_card_policy_promoted"] is False
    assert any("smoke artifacts are ineligible" in item for item in result["failures"])


def test_exact_exploitability_claim_is_rejected() -> None:
    evidence, config = _build_valid()
    evidence["exact_exploitability_computed"] = True
    _rehash_artifact(evidence)

    result = validate_promotion_evidence_m3_full_card_strength(
        evidence, config=config
    )

    assert result["passed"] is False
    assert any("exact_exploitability_computed: must be false" in item for item in result["failures"])


def test_empty_raw_rows_cannot_promote() -> None:
    evidence, config = _build_valid()
    evidence["raw_run_rows"] = []
    evidence["checkpoint_manifests"] = {}
    _rehash_artifact(evidence)

    result = validate_promotion_evidence_m3_full_card_strength(
        evidence, config=config
    )

    assert result["passed"] is False
    assert result["full_card_policy_promoted"] is False
    assert any("raw_run_rows: non-empty array required" in item for item in result["failures"])


def test_policy_tv_threshold_is_applied_to_raw_distributions() -> None:
    evidence, config = _build_valid()
    config["thresholds"]["max_policy_tv_drift"] = 0.01
    _rebind_config(evidence, config)

    result = validate_promotion_evidence_m3_full_card_strength(
        evidence, config=config
    )

    assert result["passed"] is False
    assert any("max_policy_tv_drift: threshold failed" in item for item in result["failures"])


def test_regret_threshold_is_applied_per_stratum() -> None:
    evidence, config = _build_valid()
    config["thresholds"]["max_mean_ev_regret_score"] = 0.05
    config["thresholds"]["max_p95_ev_regret_score"] = 0.05
    config["thresholds"]["max_p99_ev_regret_score"] = 0.05
    _rebind_config(evidence, config)

    result = validate_promotion_evidence_m3_full_card_strength(
        evidence, config=config
    )

    assert result["passed"] is False
    assert any("mean_ev_regret_score: threshold failed" in item for item in result["failures"])


def test_paired_seat_swap_noninferiority_is_raw_derived() -> None:
    evidence, config = _build_valid()
    row = evidence["raw_run_rows"][0]
    row["reference_action_distribution"] = {"place-a": 1.0, "place-b": 0.0}
    row["reference_payoff_estimate"] = 1.0
    evidence["published_summary"] = derive_strength_metrics(evidence)
    _rehash_artifact(evidence)

    result = validate_promotion_evidence_m3_full_card_strength(
        evidence, config=config
    )

    assert result["passed"] is False
    assert any(
        "paired_seat_swap_min_single_seat_delta_score: threshold failed" in item
        for item in result["failures"]
    )


def test_runtime_limits_are_applied_to_raw_rows() -> None:
    evidence, config = _build_valid()
    config["thresholds"]["max_runtime_ms_p95"] = 5.0
    config["thresholds"]["max_runtime_ms_max"] = 5.0
    config["thresholds"]["max_runtime_ms_total"] = 5.0
    _rebind_config(evidence, config)

    result = validate_promotion_evidence_m3_full_card_strength(
        evidence, config=config
    )

    assert result["passed"] is False
    assert any("runtime_ms_p95: threshold failed" in item for item in result["failures"])
