from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from ofc_regular.action_key import ACTION_KEY_SCHEMA, ActionKey
from ofc_regular import hu_m31_t3_step6d_locked_promotion_v1 as subject


def _threshold_lock(model_id: str, model_sha256: str) -> dict[str, object]:
    return {
        "schema": subject.THRESHOLD_LOCK_SCHEMA,
        "status": "locked_no_holdout_reselection",
        "model_artifact_id": model_id,
        "model_sha256": model_sha256,
        "state_action_input_schema_sha256": "b" * 64,
        "action_key_schema": ACTION_KEY_SCHEMA,
        "seat_thresholds": {"first": 0.80, "second": 0.82},
        "seat_enabled": {"first": True, "second": True},
        "source_training_threshold_lock_sha256": "c" * 64,
        "source_checkpoint_bundle_identity_sha256": "d" * 64,
        "locked_on_threshold_holdout_only": True,
        "model_change_requires_new_lock": True,
        "teacher_ev_lcb_is_runtime_gate": False,
        "top1_accuracy_is_promotion_gate": False,
        "current_profile_changed": False,
        "runtime_activated": False,
    }


def _make_plan(
    tmp_path: Path,
) -> tuple[dict[str, object], Path, Path, Path, Path]:
    model_id = "street-policy-net-v1-synthetic"
    model_path = tmp_path / "street_policy_net_v1.safetensors"
    model_path.write_bytes(b"immutable synthetic trained model")
    model_sha = subject.sha256_file(model_path)
    lock_path = tmp_path / "threshold_lock.json"
    lock_path.write_bytes(
        subject.canonical_bytes(_threshold_lock(model_id, model_sha))
    )
    policy_registry_path = tmp_path / "ai_profiles.py"
    policy_registry_path.write_bytes(b"synthetic locked policy registry")
    runtime_closure_path = tmp_path / "evaluation_runtime_closure.tar"
    runtime_closure_path.write_bytes(b"synthetic evaluator runtime closure")
    plan = subject.build_locked_promotion_plan(
        plan_id="m31-t3-locked-promotion-synthetic",
        model_artifact_id=model_id,
        model_path=model_path,
        expected_model_sha256=model_sha,
        threshold_lock_path=lock_path,
        expected_threshold_lock_sha256=subject.sha256_file(lock_path),
        policy_registry_path=policy_registry_path,
        expected_policy_registry_sha256=subject.sha256_file(
            policy_registry_path
        ),
        evaluation_runtime_closure_path=runtime_closure_path,
        expected_evaluation_runtime_closure_sha256=subject.sha256_file(
            runtime_closure_path
        ),
    )
    return (
        plan,
        model_path,
        lock_path,
        policy_registry_path,
        runtime_closure_path,
    )


def _hand_row(
    *,
    plan: dict[str, object],
    schedule: str,
    entity_id: str,
    seed_index: int,
    seat: str,
    candidate_score: float,
    baseline_score: float,
    override_fired: bool | None,
    override_log_valid: bool = True,
    nonfire_identical: bool = True,
) -> dict[str, object]:
    seeds = subject.seed_values(schedule, seed_index)
    is_nonfire = override_log_valid and override_fired is False
    baseline_action = ActionKey().to_token()
    action_is_identical = is_nonfire and nonfire_identical
    candidate_action = (
        baseline_action
        if action_is_identical
        else ActionKey(top_mask=1).to_token()
    )
    baseline_trajectory = "c" * 64
    candidate_trajectory = (
        baseline_trajectory
        if action_is_identical
        else "d" * 64
    )
    return {
        "schema": subject.ROW_SCHEMA,
        "schedule": schedule,
        "entity_id": entity_id,
        "seed_index": seed_index,
        "hand_seed": seeds["hand"],
        "actor_policy_seed": seeds["actor_policy"],
        "opponent_policy_seed": seeds["opponent_policy"],
        "evaluation_seed": seeds["evaluation"],
        "child_seed": seeds["child"],
        "confirmation_seed": seeds["confirmation"],
        "seat": seat,
        "action_key_schema": ACTION_KEY_SCHEMA,
        "candidate_action_key": candidate_action,
        "baseline_action_key": baseline_action,
        "legal_action_mapping_sha256": "e" * 64,
        "candidate_trajectory_sha256": candidate_trajectory,
        "baseline_trajectory_sha256": baseline_trajectory,
        "score_perspective": "candidate_hero_hu_score",
        "candidate_score": candidate_score,
        "baseline_score": baseline_score,
        "delta": candidate_score - baseline_score,
        "override_log_valid": override_log_valid,
        "override_fired": override_fired,
        "nonfire_action_key_identical": (
            nonfire_identical if is_nonfire else None
        ),
        "nonfire_trajectory_identical": (
            nonfire_identical if is_nonfire else None
        ),
        "nonfire_cancellation_valid": (
            nonfire_identical if is_nonfire else None
        ),
        "model_sha256": plan["artifact_binding"]["model"]["sha256"],
        "threshold_lock_sha256": (
            plan["artifact_binding"]["threshold_lock"]["sha256"]
        ),
        "policy_registry_sha256": (
            plan["artifact_binding"]["policy_registry"]["sha256"]
        ),
        "evaluation_runtime_closure_sha256": (
            plan["artifact_binding"]["evaluation_runtime_closure"]["sha256"]
        ),
        "counterfactual_basis": (
            "same_hand_role_policy_seeds_physical_seat_v1"
        ),
        "teacher_values_used": False,
        "opponent_private_discards_used": False,
        "current_profile_resolved": False,
        "current_profile_changed": False,
    }


def _passing_rows(
    plan: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    population: list[dict[str, object]] = []
    for descriptor in subject.OPPONENT_DESCRIPTORS:
        opponent_id = descriptor["opponent_id"]
        for index in range(subject.POPULATION_SEED_COUNT):
            fired = index < 40
            delta = 0.2 if fired else 0.0
            for seat in subject.SEATS:
                population.append(
                    _hand_row(
                        plan=plan,
                        schedule=subject.LOCKED_POPULATION,
                        entity_id=opponent_id,
                        seed_index=index,
                        seat=seat,
                        candidate_score=delta,
                        baseline_score=0.0,
                        override_fired=fired,
                    )
                )
    abr: list[dict[str, object]] = []
    for descriptor in subject.ABR_DESCRIPTORS:
        response_id = descriptor["response_id"]
        for index in range(subject.ABR_SEED_COUNT):
            for seat in subject.SEATS:
                abr.append(
                    _hand_row(
                        plan=plan,
                        schedule=subject.LOCKED_ABR,
                        entity_id=response_id,
                        seed_index=index,
                        seat=seat,
                        candidate_score=0.01,
                        baseline_score=0.01,
                        override_fired=True,
                    )
                )
    return (
        sorted(population, key=subject._row_key),
        sorted(abr, key=subject._row_key),
    )


@pytest.fixture(scope="module")
def locked_evaluation(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, object]:
    root = tmp_path_factory.mktemp("locked-promotion")
    (
        plan,
        model_path,
        lock_path,
        policy_registry_path,
        runtime_closure_path,
    ) = _make_plan(root)
    population, abr = _passing_rows(plan)
    population_path = root / "population.json"
    abr_path = root / "abr.json"
    subject.write_evaluation_shard(
        plan=plan,
        shard_id="population-all",
        rows=population,
        output_path=population_path,
    )
    subject.write_evaluation_shard(
        plan=plan,
        shard_id="abr-all",
        rows=abr,
        output_path=abr_path,
    )
    merge = subject.build_locked_promotion_merge(
        plan=plan, shard_paths=[population_path, abr_path]
    )
    return {
        "root": root,
        "plan": plan,
        "model_path": model_path,
        "lock_path": lock_path,
        "policy_registry_path": policy_registry_path,
        "runtime_closure_path": runtime_closure_path,
        "population_rows": population,
        "abr_rows": abr,
        "population_path": population_path,
        "abr_path": abr_path,
        "merge": merge,
    }


def test_plan_freezes_population_abr_seeds_and_no_activation(
    locked_evaluation: dict[str, object],
) -> None:
    plan = locked_evaluation["plan"]
    assert [row["opponent_id"] for row in plan["opponents"]] == [
        "stage19_p0",
        "stage9f_p2",
        "stage7_m5_r10",
        "stage3_baseline",
        "random_exact_final",
    ]
    assert len(plan["abr_families"]) == 3
    assert {
        row["response_id"] for row in plan["abr_families"]
    } == {
        "greedy_search_response",
        "foul_pressure_response",
        "royalty_denial_response",
    }
    assert plan["seed_contract"]["all_namespace_values_unique"] is True
    assert plan["seed_contract"]["population_abr_disjoint"] is True
    assert plan["seed_contract"]["paired_seat_swap_reuses_role_seeds"] is True
    assert plan["seed_contract"]["posthoc_extension_allowed"] is False
    assert plan["evaluation_contract"]["holdout_threshold_reselection_allowed"] is False
    assert plan["current_profile_changed"] is False
    assert plan["runtime_activated"] is False
    assert plan["full_replacement_enabled"] is False
    assert (
        plan["evaluation_contract"]["policy_registry_sha256"]
        == plan["artifact_binding"]["policy_registry"]["sha256"]
    )
    assert (
        plan["evaluation_contract"]["evaluation_runtime_closure_sha256"]
        == plan["artifact_binding"]["evaluation_runtime_closure"]["sha256"]
    )
    assert all(
        row.get("opponent_id") != "current" for row in plan["opponents"]
    )


def test_exact_synthetic_population_and_abr_pass_locked_gate(
    locked_evaluation: dict[str, object],
) -> None:
    plan = locked_evaluation["plan"]
    merge = locked_evaluation["merge"]
    gate = subject.build_locked_promotion_gate(
        plan=plan, merge=merge, replay_sources=True
    )
    assert merge["row_count"] == 13_000
    assert merge["coverage"]["population_rows"] == 10_000
    assert merge["coverage"]["abr_rows"] == 3_000
    assert merge["coverage"]["first_rows"] == 6_500
    assert merge["coverage"]["second_rows"] == 6_500
    assert merge["population"]["valid_overrides"] == 400
    assert merge["population"]["valid_overrides_by_seat"] == {
        "first": 200,
        "second": 200,
    }
    assert (
        merge["population"]["paired_seat_swap_delta_ev_per_hand"]["ci95_low"]
        > 0.0
    )
    assert (
        merge["population"]["realized_gain_per_override"]["ci95_low"] > 0.0
    )
    assert gate["all_gates_passed"] is True
    assert gate["scientific_promotion_passed"] is True
    assert gate["separate_opt_in_profile_candidate_authorized"] is True
    assert all(gate["gates"].values())
    assert gate["named_profile_added"] is False
    assert gate["current_profile_changed"] is False
    assert gate["runtime_activated"] is False
    assert gate["full_replacement_enabled"] is False


def test_combined_population_and_abr_failures_produce_no_go(
    locked_evaluation: dict[str, object],
) -> None:
    root = locked_evaluation["root"]
    plan = locked_evaluation["plan"]
    population = deepcopy(locked_evaluation["population_rows"])
    abr = deepcopy(locked_evaluation["abr_rows"])

    fired_seen = 0
    for row in population:
        if row["override_fired"] is True and fired_seen < 130:
            row["candidate_score"] = -60.0
            row["baseline_score"] = 0.0
            row["delta"] = -60.0
            fired_seen += 1
    nonfire = next(row for row in population if row["override_fired"] is False)
    nonfire["candidate_trajectory_sha256"] = "f" * 64
    nonfire["nonfire_trajectory_identical"] = False
    nonfire["nonfire_cancellation_valid"] = False
    for row in abr:
        if row["entity_id"] == "royalty_denial_response":
            row["candidate_score"] = -0.02
            row["baseline_score"] = -0.02
            row["delta"] = 0.0

    population_path = root / "population-no-go.json"
    abr_path = root / "abr-no-go.json"
    subject.write_evaluation_shard(
        plan=plan,
        shard_id="population-no-go",
        rows=population,
        output_path=population_path,
    )
    subject.write_evaluation_shard(
        plan=plan,
        shard_id="abr-no-go",
        rows=abr,
        output_path=abr_path,
    )
    merge = subject.build_locked_promotion_merge(
        plan=plan, shard_paths=[population_path, abr_path]
    )
    gate = subject.build_locked_promotion_gate(
        plan=plan, merge=merge, replay_sources=True
    )
    assert gate["decision"] == (
        "locked_promotion_no_go_no_holdout_reselection_or_extension"
    )
    assert gate["all_gates_passed"] is False
    assert gate["scientific_promotion_passed"] is False
    assert gate["separate_opt_in_profile_candidate_authorized"] is False
    assert gate["gates"]["invalid_counterfactuals_zero"] is False
    assert (
        gate["gates"]["nonfire_complete_cancellation_zero_mismatch"] is False
    )
    assert gate["gates"]["false_positive_rates_within_limits"] is False
    assert gate["gates"]["override_loss_tail_within_25_40_50"] is False
    assert gate["gates"]["each_opponent_mean_and_ci_floor"] is False
    assert gate["gates"][
        "direct_approximate_response_and_stress_family_floors"
    ] is False
    assert gate["current_profile_changed"] is False
    assert gate["runtime_activated"] is False


def test_row_contract_rejects_artifact_current_and_hidden_tampering(
    locked_evaluation: dict[str, object],
) -> None:
    plan = locked_evaluation["plan"]
    source = locked_evaluation["population_rows"][0]

    wrong_model = deepcopy(source)
    wrong_model["model_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="boundary changed"):
        subject.validate_hand_row(wrong_model, plan=plan)

    current = deepcopy(source)
    current["current_profile_resolved"] = True
    with pytest.raises(ValueError, match="boundary changed"):
        subject.validate_hand_row(current, plan=plan)

    false_action_claim = deepcopy(
        next(
            row
            for row in locked_evaluation["population_rows"]
            if row["override_fired"] is False
        )
    )
    false_action_claim["candidate_action_key"] = ActionKey(top_mask=1).to_token()
    with pytest.raises(ValueError, match="digest replay"):
        subject.validate_hand_row(false_action_claim, plan=plan)

    hidden = deepcopy(source)
    hidden["opponent_private_discards"] = ["As"]
    with pytest.raises(ValueError, match="fields changed"):
        subject.validate_hand_row(hidden, plan=plan)


def test_artifact_mutation_and_source_mutation_fail_closed(
    locked_evaluation: dict[str, object],
    tmp_path: Path,
) -> None:
    plan = locked_evaluation["plan"]
    model_copy = tmp_path / "model.safetensors"
    lock_copy = tmp_path / "threshold_lock.json"
    registry_copy = tmp_path / "ai_profiles.py"
    runtime_copy = tmp_path / "runtime_closure.tar"
    model_copy.write_bytes(locked_evaluation["model_path"].read_bytes())
    lock_copy.write_bytes(locked_evaluation["lock_path"].read_bytes())
    registry_copy.write_bytes(
        locked_evaluation["policy_registry_path"].read_bytes()
    )
    runtime_copy.write_bytes(
        locked_evaluation["runtime_closure_path"].read_bytes()
    )
    subject.validate_artifact_files(
        plan,
        model_path=model_copy,
        threshold_lock_path=lock_copy,
        policy_registry_path=registry_copy,
        evaluation_runtime_closure_path=runtime_copy,
    )
    model_copy.write_bytes(model_copy.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="artifact replay changed"):
        subject.validate_artifact_files(
            plan,
            model_path=model_copy,
            threshold_lock_path=lock_copy,
            policy_registry_path=registry_copy,
            evaluation_runtime_closure_path=runtime_copy,
        )
    model_copy.write_bytes(locked_evaluation["model_path"].read_bytes())
    registry_copy.write_bytes(registry_copy.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="artifact replay changed"):
        subject.validate_artifact_files(
            plan,
            model_path=model_copy,
            threshold_lock_path=lock_copy,
            policy_registry_path=registry_copy,
            evaluation_runtime_closure_path=runtime_copy,
        )

    population_copy = tmp_path / "population.json"
    abr_copy = tmp_path / "abr.json"
    population_copy.write_bytes(
        locked_evaluation["population_path"].read_bytes()
    )
    abr_copy.write_bytes(locked_evaluation["abr_path"].read_bytes())
    merge = subject.build_locked_promotion_merge(
        plan=plan, shard_paths=[population_copy, abr_copy]
    )
    population_copy.write_bytes(population_copy.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source shard changed"):
        subject.validate_locked_promotion_merge(
            merge, plan=plan, replay_sources=True
        )
    with pytest.raises(PermissionError, match="source replay"):
        subject.validate_locked_promotion_merge(
            locked_evaluation["merge"], plan=plan, replay_sources=False
        )


def test_write_once_refuses_changed_gate(
    locked_evaluation: dict[str, object],
    tmp_path: Path,
) -> None:
    plan = locked_evaluation["plan"]
    merge = locked_evaluation["merge"]
    output = tmp_path / "gate.json"
    first = subject.write_locked_promotion_gate(
        plan=plan,
        merge=merge,
        replay_sources=True,
        output_path=output,
    )
    assert first["status"] == "pass"
    changed = deepcopy(first)
    changed["status"] = "no_go"
    with pytest.raises(FileExistsError, match="output changed"):
        subject._write_once(output, changed)

    changed_plan = deepcopy(plan)
    changed_plan["artifact_binding"]["threshold_lock_content"][
        "seat_thresholds"
    ]["first"] = 0.01
    with pytest.raises(ValueError, match="plan boundary changed"):
        subject.validate_locked_promotion_plan(changed_plan)
