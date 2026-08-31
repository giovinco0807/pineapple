from __future__ import annotations

import copy
import hashlib
import json
import random
import shutil
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

import ai.tutor.t3_production_holdout_evaluator as evaluator
import ai.tutor.t3_shared_multi_root_strength_gate as gate
from ai.engine.action_space import get_turn_actions
from ai.engine.encoding import Board
from ai.tutor.exact_late import action_key
from ai.tutor.promotion_gate_m3_full_card_strength import (
    REQUIRED_STRATA,
    T3_BB_LIKELIHOOD_METHOD,
    T3_BB_LIKELIHOOD_SCHEMA,
    canonical_json,
    canonical_sha256,
    root_identity_commitment_sha256,
    self_hash,
)
from ai.tutor.t3_bb_range_evidence import (
    build_restricted_range_evidence,
    write_restricted_range_evidence,
)
from ai.tutor.t3_hu_full_card_mccfr import serialize_strategy_profile
from ai.tutor.t3_hu_full_card_mccfr import FullCardGenerativeAdapter
from ai.tutor.t3_hu_full_card_range import (
    UniformLegalBehaviorModel,
    build_history_weighted_full_card_range,
)
from ai.tutor.t3_hu_reduced_fixtures import compile_canonical_reduced_fixture


def _sha(label: str) -> str:
    return canonical_sha256({"label": label})


def _root_key(stratum: str):
    actor, joker_text = stratum.split("_joker")
    compiled = compile_canonical_reduced_fixture(actor, int(joker_text))
    return compiled.root.branches[0].child.state.infoset_key


def _distribution(key):
    rows = key.board_bb if key.actor == "bb" else key.board_btn
    board = Board(top=list(rows[0]), middle=list(rows[1]), bottom=list(rows[2]))
    actions = sorted(
        action_key(action) for action in get_turn_actions(list(key.current_draw), board)
    )
    return {
        action_id: 1.0 if index == 0 else 0.0
        for index, action_id in enumerate(actions)
    }


def _fake_result(stratum: str, *, seed: int, profile):
    key = _root_key(stratum)
    profile_text = serialize_strategy_profile(profile)
    profile_sha = hashlib.sha256(profile_text.encode()).hexdigest()
    opaque_ids = (_sha(f"{stratum}-a"), _sha(f"{stratum}-b"))
    prior = {
        "schema": "ofc_multi_root_exact_prior/v1",
        "sampling_contract": gate.SUPER_ROOT_SAMPLING_CONTRACT,
        "roots": [
            {
                "root_id_sha256": opaque,
                "prior_mass_exact": "1/2",
                "observation_sha256": _sha(f"{opaque}-training-observation"),
                "conditional_particle_count": 1,
                "range_content_sha256": _sha(f"{opaque}-content"),
                "range_build_sha256": _sha(f"{opaque}-build"),
            }
            for opaque in opaque_ids
        ],
    }
    iterations = 2
    traversals = 4
    metadata = {
        "method": gate.MULTI_ROOT_SOLVER_METHOD,
        "chance_super_root": True,
        "super_root_sampling_contract": gate.SUPER_ROOT_SAMPLING_CONTRACT,
        "root_prior_manifest": prior,
        "root_prior_manifest_sha256": canonical_sha256(prior),
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
        "max_infosets": 1000,
        "average_strategy_sha256": profile_sha,
        "position_contract_version": "bb_first_v1",
    }
    sampling = {
        "traversals": traversals,
        "traversals_by_actor": {"bb": iterations, "btn": iterations},
        "super_root_samples": traversals,
        "conditional_root_posterior_samples": traversals,
        "root_samples_by_opaque_id": {opaque: iterations for opaque in opaque_ids},
        "distinct_root_adapters_sampled": 2,
        "infosets_created": len(profile),
    }
    return SimpleNamespace(
        metadata=metadata,
        sampling_stats=sampling,
        average_strategy_json=profile_text,
        average_strategy_sha256=profile_sha,
    ), opaque_ids


def _candidate_bundle(profiles):
    range_source = _sha("range-source")
    binding = {
        "schema": T3_BB_LIKELIHOOD_SCHEMA,
        "route": "t3_bb",
        "consumer_root_actor": "btn",
        "consumer_phase": "t3_second",
        "method": T3_BB_LIKELIHOOD_METHOD,
        "promotion_eligible": True,
        "fixed_point_converged": True,
        "candidate_policy_artifact_sha256": _sha("fp-candidate"),
        "candidate_policy_checkpoint_sha256": _sha("fp-checkpoint"),
        "fixed_point_evidence_sha256": _sha("fp-evidence"),
        "fixed_point_gate_result_sha256": _sha("fp-result"),
        "fixed_point_config_sha256": _sha("fp-config"),
        "range_builder_source_sha256": range_source,
        "solver_manifest_sha256": _sha("fp-solver"),
    }
    binding["binding_sha256"] = self_hash(binding, "binding_sha256")
    strategies = {}
    for index, stratum in enumerate(REQUIRED_STRATA):
        result, opaque_ids = _fake_result(
            stratum, seed=10 + index, profile=profiles[stratum]
        )
        identities = {
            opaque: root_identity_commitment_sha256(f"training-{stratum}-{i}")
            for i, opaque in enumerate(opaque_ids)
        }
        strategies[stratum] = gate.build_shared_multi_root_strategy_artifact(
            result,
            stratum=stratum,
            training_root_identity_by_opaque_id=identities,
            source_manifest_sha256=_sha("multi-root-source"),
            range_builder_source_sha256=range_source,
            t3_bb_likelihood_binding_sha256=binding["binding_sha256"],
        )
    return gate.build_shared_multi_root_candidate_bundle(
        strategies, t3_bb_likelihood_binding=binding
    )


def _collect_candidate_profile(root, observation, full_range, payoff_values):
    profile = {observation: _distribution(observation)}
    adapter = FullCardGenerativeAdapter(observation, full_range)
    probe = adapter.sample_root_for_traversal(
        random.Random(
            evaluator._physical_entropy_seed(
                payoff_seed=payoff_values[0],
                seat_swap_pair_id=root["seat_swap_pair_id"],
                sample_index=0,
                stage="root_posterior",
                chance_index=0,
            )
        )
    )
    root_actions = tuple(action_id for action_id, _ in adapter.legal_actions(probe.state))
    for payoff_seed in payoff_values:
        for sample_index in range(2):
            for root_action in root_actions:
                adapter = FullCardGenerativeAdapter(observation, full_range)
                state = adapter.sample_root_for_traversal(
                    random.Random(
                        evaluator._physical_entropy_seed(
                            payoff_seed=payoff_seed,
                            seat_swap_pair_id=root["seat_swap_pair_id"],
                            sample_index=sample_index,
                            stage="root_posterior",
                            chance_index=0,
                        )
                    )
                ).state
                state = adapter.apply_action_id(state, root_action)
                chance_index = 0
                while state.__class__.__name__ != "PublicTreeTerminalState":
                    if state.__class__.__name__ == "PendingChanceState":
                        state = adapter.sample_next_draw(
                            state,
                            random.Random(
                                evaluator._physical_entropy_seed(
                                    payoff_seed=payoff_seed,
                                    seat_swap_pair_id=root["seat_swap_pair_id"],
                                    sample_index=sample_index,
                                    stage="future_draw",
                                    chance_index=chance_index,
                                )
                            ),
                        ).state
                        chance_index += 1
                        continue
                    key = adapter.information_key(state)
                    distribution = profile.setdefault(key, _distribution(key))
                    selected = next(
                        action_id
                        for action_id in sorted(distribution)
                        if distribution[action_id] == 1.0
                    )
                    state = adapter.apply_action_id(state, selected)
    return profile


def _fixture(tmp_path: Path):
    payoff_seeds = {
        f"pair-joker{joker}": {101: 1000 + joker * 10, 202: 1001 + joker * 10}
        for joker in range(3)
    }
    roots = [
        gate.build_holdout_root_record(
            f"holdout-{stratum}",
            stratum=stratum,
            observation_digest=_root_key(stratum).digest(),
            seat_swap_pair_id=f"pair-joker{stratum[-1]}",
        )
        for stratum in REQUIRED_STRATA
    ]
    range_paths = {}
    ranges = {}
    for index, root in enumerate(sorted(roots, key=lambda row: (row["stratum"], row["root_id"]))):
        key = _root_key(root["stratum"])
        seed = 300 + index
        full_range = build_history_weighted_full_card_range(
            key,
            UniformLegalBehaviorModel(),
            epsilon=Fraction(1, 1000),
            max_particles=1,
            seed=seed,
        )
        artifact = build_restricted_range_evidence(
            key,
            full_range,
            root_id=root["root_id"],
            root_commitment_sha256=root["root_commitment_sha256"],
            round_index=0,
            solver_seed=seed,
        )
        path = tmp_path / "source-ranges" / f"{root['root_id']}.json"
        write_restricted_range_evidence(path, artifact)
        range_paths[root["root_id"]] = path
        ranges[root["root_id"]] = full_range
    profiles = {
        root["stratum"]: _collect_candidate_profile(
            root,
            _root_key(root["stratum"]),
            ranges[root["root_id"]],
            list(payoff_seeds[root["seat_swap_pair_id"]].values()),
        )
        for root in roots
    }
    candidate = _candidate_bundle(profiles)
    training_roots = [
        row["root_identity_commitment_sha256"]
        for artifact in candidate["strategies"].values()
        for row in artifact["training_roots"]
    ]
    partitions = {
        "training": gate.build_excluded_root_partition("training", training_roots),
        "calibration": gate.build_excluded_root_partition(
            "calibration", [root_identity_commitment_sha256("calibration-a")]
        ),
        "smoke": gate.build_excluded_root_partition(
            "smoke", [root_identity_commitment_sha256("smoke-a")]
        ),
    }
    root_manifest = gate.build_holdout_root_manifest(
        roots,
        excluded_root_partition_sha256={
            purpose: partition["manifest_sha256"]
            for purpose, partition in partitions.items()
        },
    )
    root_bundle_dir = tmp_path / "root-bundle"
    root_bundle = evaluator.build_root_range_bundle(
        root_manifest=root_manifest,
        range_evidence_paths=range_paths,
        output_dir=root_bundle_dir,
    )
    candidate_path = tmp_path / "candidate.json"
    evaluator.write_candidate_bundle(candidate_path, candidate)
    return (
        root_bundle,
        root_bundle_dir,
        candidate,
        candidate_path,
        payoff_seeds,
        partitions,
    )


def _build_eval(tmp_path: Path, root_bundle_dir, candidate_path, payoff_seeds, name="eval"):
    output = tmp_path / name
    manifest = evaluator.build_evaluation_bundle(
        root_range_bundle_dir=root_bundle_dir,
        candidate_bundle_path=candidate_path,
        output_dir=output,
        evaluation_seeds=[101, 202],
        payoff_seeds=payoff_seeds,
        samples_per_action=2,
    )
    return output, manifest


def test_real_full_card_end_to_end_is_deterministic_and_nonpromoting(tmp_path):
    root_bundle, root_dir, candidate, candidate_path, payoff_seeds, partitions = _fixture(tmp_path)
    first_dir, first = _build_eval(tmp_path, root_dir, candidate_path, payoff_seeds, "eval-a")
    second_dir, second = _build_eval(tmp_path, root_dir, candidate_path, payoff_seeds, "eval-b")

    assert root_bundle["promotion_eligible"] is False
    assert candidate["algorithm_validation_only"] is True
    assert candidate["production_promotion_supported"] is False
    assert candidate["unseen_root_strength_supported"] is False
    assert candidate["candidate_scope_manifest_sha256"] == canonical_sha256(
        candidate["candidate_scope_manifest"]
    )
    assert first["promotion_eligible"] is False
    assert first["algorithm_validation_only"] is True
    assert first["production_promotion_supported"] is False
    assert first["exact_exploitability_computed"] is False
    assert len(first["shards"]) == 12
    first_rows = evaluator.build_strength_rows_from_evaluation_bundle(
        first_dir, root_range_bundle_dir=root_dir
    )
    second_rows = evaluator.build_strength_rows_from_evaluation_bundle(
        second_dir, root_range_bundle_dir=root_dir
    )
    assert len(first_rows) == 12
    for left, right in zip(first_rows, second_rows):
        for field in (
            "action_payoff_aggregates",
            "action_payoff_estimates",
            "action_payoff_standard_errors",
            "reference_action_payoff_aggregates",
            "reference_action_payoff_estimates",
            "reference_action_payoff_standard_errors",
            "policy_reference_paired_aggregate",
            "policy_reference_delta_estimate",
            "policy_reference_delta_standard_error",
            "policy_payoff_estimate",
            "reference_payoff_estimate",
            "ev_regret_estimate",
        ):
            assert left[field] == right[field]
        assert left["exact_exploitability_computed"] is False
    assert first["evaluation_seeds"] == second["evaluation_seeds"]
    raw_shards = [
        evaluator._read_canonical(first_dir / record["relative_path"])
        for record in first["shards"]
    ]
    assert any(
        len(shard["required_candidate_infoset_digests"]) > 1
        for shard in raw_shards
    )
    assert all(shard["candidate_missing_infoset_count"] == 0 for shard in raw_shards)
    assert all(shard["candidate_fallback_count"] == 0 for shard in raw_shards)
    with pytest.raises(TypeError, match="replay"):
        evaluator.verify_evaluation_bundle(
            first_dir, root_range_bundle_dir=root_dir, replay=False
        )

    thresholds = {
        "min_independent_evaluation_seeds_per_root": 2,
        "min_roots_per_stratum": 1,
        "min_action_payoff_samples_per_action": 2,
        "min_encountered_infoset_coverage": 1.0,
        "max_action_payoff_standard_error": 100.0,
        "max_reference_action_payoff_standard_error": 100.0,
        "max_policy_reference_delta_standard_error": 100.0,
        "max_mean_ev_regret_score": 100.0,
        "max_p95_ev_regret_score": 100.0,
        "max_p99_ev_regret_score": 100.0,
        "paired_seat_swap_noninferiority_margin_score": 100.0,
        "max_runtime_ms_p95": 1_000_000.0,
        "max_runtime_ms_max": 1_000_000.0,
        "max_runtime_ms_total": 10_000_000.0,
    }
    root_manifest = root_bundle["root_manifest"]
    gate_config = gate.build_locked_shared_multi_root_strength_config(
        approved_candidate_bundle_sha256=candidate["candidate_bundle_sha256"],
        approved_holdout_root_manifest_sha256=root_manifest["root_manifest_sha256"],
        approved_evaluator_manifest_sha256=first["holdout_evaluator_manifest"]["manifest_sha256"],
        approved_excluded_root_partition_sha256={
            purpose: partition["manifest_sha256"]
            for purpose, partition in partitions.items()
        },
        thresholds=thresholds,
    )
    evidence = gate.build_shared_multi_root_strength_evidence(
        config=gate_config,
        candidate_bundle=candidate,
        root_manifest=root_manifest,
        excluded_root_partitions=partitions,
        evaluator_manifest=first["holdout_evaluator_manifest"],
        raw_holdout_rows=first_rows,
        production_promotion_claim=False,
    )
    gate_result = gate.validate_shared_multi_root_strength_evidence(
        evidence, config=gate_config
    )
    assert gate_result["failures"] == [
        "production promotion claim is false",
        (
            "tabular exact-infoset evaluation is algorithm-validation only; "
            "production promotion is unsupported"
        ),
    ]


def test_resume_reuses_complete_shards_and_manifest_is_last(tmp_path, monkeypatch):
    _bundle, root_dir, _candidate, candidate_path, payoff_seeds, _partitions = _fixture(tmp_path)
    output, manifest = _build_eval(tmp_path, root_dir, candidate_path, payoff_seeds)
    missing_record = manifest["shards"][0]
    untouched_record = manifest["shards"][1]
    missing_path = output / missing_record["relative_path"]
    untouched_path = output / untouched_record["relative_path"]
    expected_missing = evaluator._read_canonical(missing_path)["action_payoff_aggregates"]
    untouched_bytes = untouched_path.read_bytes()
    (output / "evaluation-bundle.json").unlink()
    missing_path.unlink()

    resumed = evaluator.build_evaluation_bundle(
        root_range_bundle_dir=root_dir,
        candidate_bundle_path=candidate_path,
        output_dir=output,
        evaluation_seeds=[101, 202],
        payoff_seeds=payoff_seeds,
        samples_per_action=2,
    )
    assert untouched_path.read_bytes() == untouched_bytes
    assert evaluator._read_canonical(missing_path)["action_payoff_aggregates"] == expected_missing
    assert resumed["manifest_written_after_all_shards"] is True

    failed = tmp_path / "failed-eval"
    monkeypatch.setattr(evaluator, "_build_shard", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        evaluator.build_evaluation_bundle(
            root_range_bundle_dir=root_dir,
            candidate_bundle_path=candidate_path,
            output_dir=failed,
            evaluation_seeds=[101, 202],
            payoff_seeds=payoff_seeds,
            samples_per_action=2,
        )
    assert not (failed / "evaluation-bundle.json").exists()


def _write_canonical(path: Path, value):
    path.write_text(canonical_json(value) + "\n", encoding="utf-8", newline="\n")


def test_candidate_range_gap_orphan_and_rehashed_payoff_tamper_fail_closed(tmp_path):
    _bundle, root_dir, _candidate, candidate_path, payoff_seeds, _partitions = _fixture(tmp_path)
    output, manifest = _build_eval(tmp_path, root_dir, candidate_path, payoff_seeds)

    candidate_copy = tmp_path / "tamper-candidate"
    shutil.copytree(output, candidate_copy)
    with (candidate_copy / "inputs/candidate-bundle.json").open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(evaluator.ProductionHoldoutError):
        evaluator.verify_evaluation_bundle(candidate_copy, root_range_bundle_dir=root_dir)

    range_copy = tmp_path / "tamper-range"
    shutil.copytree(root_dir, range_copy)
    range_asset = root_dir / _bundle["entries"][0]["range_asset"]
    copied_asset = range_copy / range_asset.relative_to(root_dir)
    copied_asset.write_bytes(copied_asset.read_bytes() + b" ")
    with pytest.raises(evaluator.ProductionHoldoutError):
        evaluator.verify_root_range_bundle(range_copy)

    drift_dependencies = (
        "ai/engine/game_engine.py",
        "ai/mcts/rollout_evaluator.py",
        "ai/tutor/promotion_gate_m3_full_card_strength.py",
        "ai/tutor/t3_bb_fixed_point_gate_v2.py",
        "ai/tutor/t3_hu_public_cfr.py",
    )
    for dependency_index, dependency in enumerate(drift_dependencies):
        source_copy = tmp_path / f"tamper-source-{dependency_index}"
        shutil.copytree(root_dir, source_copy)
        source_manifest_path = source_copy / "root-range-bundle.json"
        source_bundle = evaluator._read_canonical(source_manifest_path)
        source_row = next(
            row
            for row in source_bundle["source_manifest"]["files"]
            if row["path"] == dependency
        )
        source_row["sha256"] = "0" * 64
        source_bundle["source_manifest"]["source_manifest_sha256"] = self_hash(
            source_bundle["source_manifest"], "source_manifest_sha256"
        )
        source_bundle["source_manifest_sha256"] = source_bundle["source_manifest"][
            "source_manifest_sha256"
        ]
        source_bundle["bundle_sha256"] = self_hash(source_bundle, "bundle_sha256")
        _write_canonical(source_manifest_path, source_bundle)
        with pytest.raises(evaluator.ProductionHoldoutError, match="stale source"):
            evaluator.verify_root_range_bundle(source_copy)

    gap_copy = tmp_path / "tamper-gap"
    shutil.copytree(output, gap_copy)
    (gap_copy / manifest["shards"][0]["relative_path"]).unlink()
    with pytest.raises(evaluator.ProductionHoldoutError, match="missing|gap|escapes"):
        evaluator.verify_evaluation_bundle(gap_copy, root_range_bundle_dir=root_dir)

    orphan_copy = tmp_path / "tamper-orphan"
    shutil.copytree(output, orphan_copy)
    (orphan_copy / "shards/orphan.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(evaluator.ProductionHoldoutError, match="orphan"):
        evaluator.verify_evaluation_bundle(orphan_copy, root_range_bundle_dir=root_dir)

    payoff_copy = tmp_path / "tamper-payoff"
    shutil.copytree(output, payoff_copy)
    tampered_manifest = evaluator._read_canonical(payoff_copy / "evaluation-bundle.json")
    record = tampered_manifest["shards"][0]
    shard_path = payoff_copy / record["relative_path"]
    shard = evaluator._read_canonical(shard_path)
    for prefix in ("", "reference_"):
        aggregates = shard[f"{prefix}action_payoff_aggregates"]
        estimates = shard[f"{prefix}action_payoff_estimates"]
        for action_id, moment in aggregates.items():
            old_sum = moment["sum"]
            count = moment["count"]
            moment["sum"] = old_sum + count
            moment["sum_squares"] += 2 * old_sum + count
            estimates[action_id] += 1.0
    policy = shard["policy_action_distribution"]
    reference = shard["reference_action_distribution"]
    payoffs = shard["action_payoff_estimates"]
    reference_payoffs = shard["reference_action_payoff_estimates"]
    shard["policy_payoff_estimate"] = sum(policy[key] * payoffs[key] for key in payoffs)
    shard["reference_payoff_estimate"] = sum(
        reference[key] * reference_payoffs[key] for key in reference_payoffs
    )
    shard["candidate_continuation_uniform_root_payoff_estimate"] = sum(
        reference[key] * payoffs[key] for key in payoffs
    )
    shard["best_action_payoff_estimate"] = max(payoffs.values())
    shard["ev_regret_estimate"] = max(0.0, shard["best_action_payoff_estimate"] - shard["policy_payoff_estimate"])
    shard["shard_sha256"] = self_hash(shard, "shard_sha256")
    _write_canonical(shard_path, shard)
    record["shard_sha256"] = shard["shard_sha256"]
    record["file_sha256"] = hashlib.sha256(shard_path.read_bytes()).hexdigest()
    tampered_manifest["evaluation_bundle_sha256"] = self_hash(
        tampered_manifest, "evaluation_bundle_sha256"
    )
    _write_canonical(payoff_copy / "evaluation-bundle.json", tampered_manifest)
    with pytest.raises(evaluator.ProductionHoldoutError, match="replay mismatch"):
        evaluator.verify_evaluation_bundle(payoff_copy, root_range_bundle_dir=root_dir)


def test_seed_overlap_and_missing_stratum_are_rejected(tmp_path):
    _bundle, root_dir, candidate, candidate_path, payoff_seeds, _partitions = _fixture(tmp_path)
    bad_payoff = copy.deepcopy(payoff_seeds)
    bad_payoff["pair-joker0"][101] = 101
    with pytest.raises(evaluator.ProductionHoldoutError, match="overlap"):
        evaluator.build_evaluation_bundle(
            root_range_bundle_dir=root_dir,
            candidate_bundle_path=candidate_path,
            output_dir=tmp_path / "bad-seeds",
            evaluation_seeds=[101, 202],
            payoff_seeds=bad_payoff,
            samples_per_action=2,
        )
    range_seed_payoff = copy.deepcopy(payoff_seeds)
    range_seed_payoff["pair-joker0"][101] = 300
    with pytest.raises(evaluator.ProductionHoldoutError, match="range_build.*payoff"):
        evaluator.build_evaluation_bundle(
            root_range_bundle_dir=root_dir,
            candidate_bundle_path=candidate_path,
            output_dir=tmp_path / "bad-range-seed",
            evaluation_seeds=[101, 202],
            payoff_seeds=range_seed_payoff,
            samples_per_action=2,
        )

    incomplete = copy.deepcopy(candidate)
    del incomplete["strategies"]["btn_joker2"]
    incomplete["candidate_bundle_sha256"] = self_hash(incomplete, "candidate_bundle_sha256")
    with pytest.raises(ValueError, match="six strata"):
        evaluator.write_candidate_bundle(tmp_path / "incomplete.json", incomplete)


def _rehashed_candidate_binding(candidate, *, prior_field, training_field, value):
    changed = copy.deepcopy(candidate)
    artifact = changed["strategies"]["bb_joker0"]
    artifact["root_prior_manifest"]["roots"][0][prior_field] = value
    artifact["training_roots"][0][training_field] = value
    artifact["root_prior_manifest_sha256"] = canonical_sha256(
        artifact["root_prior_manifest"]
    )
    artifact["artifact_sha256"] = self_hash(artifact, "artifact_sha256")
    return gate.build_shared_multi_root_candidate_bundle(
        changed["strategies"],
        t3_bb_likelihood_binding=changed["t3_bb_likelihood_binding"],
    )


def test_relabelled_training_observation_range_and_missing_continuation_fail(tmp_path):
    root_bundle, root_dir, candidate, _candidate_path, payoff_seeds, _partitions = _fixture(tmp_path)
    root = next(
        row for row in root_bundle["root_manifest"]["roots"] if row["stratum"] == "bb_joker0"
    )
    entry = next(
        row for row in root_bundle["entries"] if row["root"]["root_id"] == root["root_id"]
    )
    cases = (
        ("observation_sha256", "observation_digest", root["observation_digest"], "observation"),
        ("range_content_sha256", "range_content_sha256", entry["range_content_sha256"], "range content"),
        ("range_build_sha256", "range_build_sha256", entry["range_build_sha256"], "range build"),
    )
    for index, (prior_field, training_field, value, message) in enumerate(cases):
        relabelled = _rehashed_candidate_binding(
            candidate,
            prior_field=prior_field,
            training_field=training_field,
            value=value,
        )
        path = tmp_path / f"relabelled-{index}.json"
        evaluator.write_candidate_bundle(path, relabelled)
        with pytest.raises(evaluator.ProductionHoldoutError, match=message):
            evaluator.build_evaluation_bundle(
                root_range_bundle_dir=root_dir,
                candidate_bundle_path=path,
                output_dir=tmp_path / f"relabelled-eval-{index}",
                evaluation_seeds=[101, 202],
                payoff_seeds=payoff_seeds,
                samples_per_action=2,
            )

    missing = copy.deepcopy(candidate)
    artifact = missing["strategies"]["bb_joker0"]
    root_digest = root["observation_digest"]
    artifact["strategy_profile"]["records"] = [
        record
        for record in artifact["strategy_profile"]["records"]
        if record["infoset_digest"] == root_digest
    ]
    assert len(artifact["strategy_profile"]["records"]) == 1
    artifact["average_strategy_sha256"] = canonical_sha256(
        artifact["strategy_profile"]
    )
    artifact["producer_contract"]["shared_across_roots_infoset_count"] = 1
    artifact["sampling_audit"]["infosets_created"] = 1
    artifact["artifact_sha256"] = self_hash(artifact, "artifact_sha256")
    missing = gate.build_shared_multi_root_candidate_bundle(
        missing["strategies"],
        t3_bb_likelihood_binding=missing["t3_bb_likelihood_binding"],
    )
    missing_path = tmp_path / "missing-continuation.json"
    evaluator.write_candidate_bundle(missing_path, missing)
    with pytest.raises(evaluator.ProductionHoldoutError, match="missing exact InfoSetKey"):
        evaluator.build_evaluation_bundle(
            root_range_bundle_dir=root_dir,
            candidate_bundle_path=missing_path,
            output_dir=tmp_path / "missing-continuation-eval",
            evaluation_seeds=[101, 202],
            payoff_seeds=payoff_seeds,
            samples_per_action=2,
        )


def test_terminal_scorer_monkeypatch_and_wrong_root_semantics_fail_closed(monkeypatch):
    source = evaluator.build_source_manifest()
    monkeypatch.setattr(
        FullCardGenerativeAdapter,
        "terminal_metrics_bb",
        staticmethod(lambda _terminal: {"score": 999.0}),
    )
    with pytest.raises(evaluator.ProductionHoldoutError, match="terminal_metrics_bb"):
        evaluator.verify_source_manifest(source)

    key = _root_key("bb_joker1")
    wrong = gate.build_holdout_root_record(
        "wrong-semantic-root",
        stratum="btn_joker1",
        observation_digest=key.digest(),
        seat_swap_pair_id="wrong-pair",
    )
    with pytest.raises(evaluator.ProductionHoldoutError, match="actor"):
        evaluator._verify_root_observation_semantics(wrong, key)


def test_physical_crn_is_pair_common_and_policy_entropy_is_separate():
    left = evaluator._physical_entropy_seed(
        payoff_seed=1000,
        seat_swap_pair_id="pair-joker0",
        sample_index=3,
        stage="future_draw",
        chance_index=1,
    )
    right = evaluator._physical_entropy_seed(
        payoff_seed=1000,
        seat_swap_pair_id="pair-joker0",
        sample_index=3,
        stage="future_draw",
        chance_index=1,
    )
    assert left == right
    candidate_policy = evaluator._policy_entropy_seed(
        payoff_seed=1000,
        root_commitment="a" * 64,
        root_action_id="action-a",
        sample_index=3,
        policy_kind="candidate",
        decision_index=1,
        infoset_digest="b" * 64,
    )
    reference_policy = evaluator._policy_entropy_seed(
        payoff_seed=1000,
        root_commitment="a" * 64,
        root_action_id="action-a",
        sample_index=3,
        policy_kind="reference_uniform",
        decision_index=1,
        infoset_digest="b" * 64,
    )
    assert candidate_policy != reference_policy
    assert candidate_policy != left
