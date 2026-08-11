from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import hu_m31_t3_step6d_locked_promotion_v1 as promotion
from ofc_regular import hu_m31_t3_locked_promotion_runner_v1 as subject
from ofc_regular.action_key import (
    ACTION_KEY_SCHEMA,
    action_key,
    canonicalize_actions,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from ofc_regular.action_space import generate_actions
from ofc_regular.hu_m31_t3_street_policy_runtime_v1 import (
    BASELINE_PROFILE,
    EVALUATION_ONLY_CANDIDATE,
    EVALUATION_ONLY_WATERMARK,
    HU_M31_T3_DECISION_SCHEMA,
    HU_M31_T3_EVALUATION_RECEIPT_SCHEMA,
    HU_M31_T3_RUNTIME_SCHEMA,
    RUNTIME_SCOPE_EVALUATION_ONLY,
)


class _FirstLegalPolicy:
    def __init__(self, *, seat: str, policy_seed: int) -> None:
        self.seat = seat
        self.policy_seed = policy_seed

    def choose_action_observation(
        self,
        observation,
        *,
        hand_id=None,
        game_id=None,
        decision_seed=None,
    ):
        actions = canonicalize_actions(
            generate_actions(
                observation.hero_board, observation.dealt_cards
            )
        )
        return actions[0]


class _Receipt:
    def __init__(self, plan: dict[str, Any]) -> None:
        binding = plan["artifact_binding"]
        self.value = {
            "schema": HU_M31_T3_EVALUATION_RECEIPT_SCHEMA,
            "runtime_scope": RUNTIME_SCOPE_EVALUATION_ONLY,
            "candidate_id": EVALUATION_ONLY_CANDIDATE,
            "audit_watermark": EVALUATION_ONLY_WATERMARK,
            "plan_sha256": promotion.canonical_sha256(plan),
            "model_manifest_file_sha256": binding["model"]["sha256"],
            "compatibility_threshold_lock_file_sha256": binding[
                "threshold_lock"
            ]["sha256"],
            "policy_registry_file_sha256": binding["policy_registry"][
                "sha256"
            ],
            "evaluation_runtime_closure_file_sha256": binding[
                "evaluation_runtime_closure"
            ]["sha256"],
            "plan_and_artifacts_source_replayed": True,
            "evaluation_only": True,
            "scientific_promotion_passed": False,
            "separate_opt_in_profile_candidate_authorized": False,
            "named_profile_added": False,
            "current_profile_changed": False,
            "runtime_activated": False,
            "full_replacement_enabled": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self.value)


class _EvaluationCandidate(_FirstLegalPolicy):
    runtime_id = HU_M31_T3_RUNTIME_SCHEMA
    runtime_scope = RUNTIME_SCOPE_EVALUATION_ONLY
    evaluation_only = True
    profile_candidate = EVALUATION_ONLY_CANDIDATE

    def __init__(
        self,
        *,
        seat: str,
        policy_seed: int,
        plan: dict[str, Any],
        fire: bool,
    ) -> None:
        super().__init__(seat=seat, policy_seed=policy_seed)
        self.authorization_receipt = _Receipt(plan)
        self.fire = fire

    def choose_action_observation_with_audit(
        self,
        observation,
        *,
        hand_id=None,
        game_id=None,
        decision_seed=None,
    ):
        actions = canonicalize_actions(
            generate_actions(
                observation.hero_board, observation.dealt_cards
            )
        )
        baseline = actions[0]
        proposal = actions[1]
        final = proposal if self.fire else baseline
        canonical_mapping = ordered_action_mapping_digest(actions)
        decision = {
            "schema": HU_M31_T3_DECISION_SCHEMA,
            "runtime_schema": HU_M31_T3_RUNTIME_SCHEMA,
            "runtime_scope": RUNTIME_SCOPE_EVALUATION_ONLY,
            "evaluation_only": True,
            "audit_watermark": EVALUATION_ONLY_WATERMARK,
            "profile_candidate": EVALUATION_ONLY_CANDIDATE,
            "baseline_profile": BASELINE_PROFILE,
            "observation_fingerprint": observation.fingerprint(),
            "seat": observation.seat,
            "action_key_schema": ACTION_KEY_SCHEMA,
            "legal_action_count": len(actions),
            "legal_action_set_sha256": legal_action_set_digest(actions),
            "canonical_action_mapping_sha256": canonical_mapping,
            "baseline_action_key": action_key(baseline).to_token(),
            "candidate_action_key": action_key(proposal).to_token(),
            "final_action_key": action_key(final).to_token(),
            "override_fired": self.fire,
            "promotion_gate_file_sha256": None,
            "teacher_values_used": False,
            "opponent_private_discards_used": False,
            "current_profile_resolved": False,
            "runtime_activated": False,
        }
        return final, decision


def _factory():
    def make(*, policy_seed: int, seat: str):
        return _FirstLegalPolicy(seat=seat, policy_seed=policy_seed)

    return make


def _abr_factory(
    *, response_id: str, checkpoint: Path, checkpoint_sha: str
):
    def make(*, policy_seed: int, seat: str):
        policy = _FirstLegalPolicy(seat=seat, policy_seed=policy_seed)
        policy.abr_response_id = response_id
        policy.abr_policy_factory_id = subject.ABR_FACTORY_IDS[response_id]
        policy.abr_policy_artifact_sha256 = checkpoint_sha
        policy.opponent_private_discards_used = False
        policy.current_profile_resolved = False
        return policy

    make.factory_id = subject.ABR_FACTORY_IDS[response_id]
    make.response_id = response_id
    make.checkpoint_path = checkpoint
    make.checkpoint_sha256 = checkpoint_sha
    return make


def _candidate_factory(plan: dict[str, Any], *, fire: bool):
    def make(*, policy_seed: int, seat: str):
        return _EvaluationCandidate(
            seat=seat,
            policy_seed=policy_seed,
            plan=plan,
            fire=fire,
        )

    return make


def _threshold_lock(
    model_id: str, model_sha256: str
) -> dict[str, Any]:
    return {
        "schema": promotion.THRESHOLD_LOCK_SCHEMA,
        "status": "locked_no_holdout_reselection",
        "model_artifact_id": model_id,
        "model_sha256": model_sha256,
        "state_action_input_schema_sha256": "b" * 64,
        "action_key_schema": ACTION_KEY_SCHEMA,
        "seat_thresholds": {"first": 0.8, "second": 0.82},
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


def _make_plan(tmp_path: Path) -> dict[str, Any]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    model_id = "street-policy-net-v1-runner-test"
    model = tmp_path / "manifest.json"
    model.write_bytes(b"synthetic checkpoint bundle manifest")
    model_sha = promotion.sha256_file(model)
    threshold = tmp_path / "compatibility_threshold.json"
    threshold.write_bytes(
        promotion.canonical_bytes(_threshold_lock(model_id, model_sha))
    )
    registry = tmp_path / "ai_profiles.py"
    registry.write_bytes(b"synthetic explicit profile registry")
    runtime = tmp_path / "runtime.py"
    runtime.write_bytes(b"synthetic evaluation runtime")
    return promotion.build_locked_promotion_plan(
        plan_id="m31-t3-runner-test-plan",
        model_artifact_id=model_id,
        model_path=model,
        expected_model_sha256=model_sha,
        threshold_lock_path=threshold,
        expected_threshold_lock_sha256=promotion.sha256_file(
            threshold
        ),
        policy_registry_path=registry,
        expected_policy_registry_sha256=promotion.sha256_file(
            registry
        ),
        evaluation_runtime_closure_path=runtime,
        expected_evaluation_runtime_closure_sha256=(
            promotion.sha256_file(runtime)
        ),
    )


def _abr_manifest(
    tmp_path: Path,
    *,
    plan: dict[str, Any],
    response_id: str,
    locked_seed_training_allowed: bool = False,
) -> subject.AbrPolicyBinding:
    descriptor = next(
        row
        for row in promotion.ABR_DESCRIPTORS
        if row["response_id"] == response_id
    )
    checkpoint = tmp_path / f"{response_id}.zip"
    checkpoint.write_bytes(f"frozen {response_id}".encode("ascii"))
    checkpoint_sha = promotion.sha256_file(checkpoint)
    identity = {
        "schema": subject.ABR_POLICY_MANIFEST_SCHEMA,
        "status": "frozen_independent_abr_policy",
        "response_id": response_id,
        "family": descriptor["family"],
        "objective": descriptor["objective"],
        "candidate_plan_sha256": promotion.canonical_sha256(plan),
        "policy_checkpoint_filename": checkpoint.name,
        "policy_checkpoint_sha256": checkpoint_sha,
        "policy_checkpoint_bytes": checkpoint.stat().st_size,
        "policy_checkpoint_format": "street_policy_net_v1_checkpoint_zip",
        "policy_factory_id": subject.ABR_FACTORY_IDS[response_id],
        "development_schedule": "abr_development",
        "locked_evaluation_schedule": promotion.LOCKED_ABR,
        "locked_seed_training_allowed": locked_seed_training_allowed,
        "opponent_private_discards_used": False,
        "current_profile_resolved": False,
        "frozen_before_locked_evaluation": True,
    }
    manifest = dict(identity)
    manifest["manifest_identity_sha256"] = subject._canonical_sha256(
        identity
    )
    manifest_path = tmp_path / f"{response_id}.json"
    manifest_path.write_bytes(subject._canonical_bytes(manifest))
    return subject.AbrPolicyBinding(
        response_id=response_id,
        policy_factory=_abr_factory(
            response_id=response_id,
            checkpoint=checkpoint,
            checkpoint_sha=checkpoint_sha,
        ),
        manifest_path=manifest_path,
        expected_manifest_file_sha256=promotion.sha256_file(
            manifest_path
        ),
        checkpoint_path=checkpoint,
        expected_checkpoint_file_sha256=checkpoint_sha,
    )


def _runner(
    tmp_path: Path,
    *,
    fire: bool,
) -> tuple[subject.LockedPromotionRunner, dict[str, Any]]:
    plan = _make_plan(tmp_path)
    opponent_factories = {
        row["opponent_id"]: _factory()
        for row in promotion.OPPONENT_DESCRIPTORS
    }
    abr_bindings = {
        row["response_id"]: _abr_manifest(
            tmp_path,
            plan=plan,
            response_id=row["response_id"],
        )
        for row in promotion.ABR_DESCRIPTORS
    }
    return (
        subject.LockedPromotionRunner(
            plan=plan,
            candidate_policy_factory=_candidate_factory(
                plan, fire=fire
            ),
            baseline_policy_factory=_factory(),
            opponent_policy_factories=opponent_factories,
            abr_policy_bindings=abr_bindings,
        ),
        plan,
    )


@pytest.mark.parametrize("seat", ["first", "second"])
def test_real_hand_nonfire_cancels_action_trajectory_and_score(
    tmp_path: Path, seat: str
) -> None:
    runner, plan = _runner(tmp_path, fire=False)
    row = runner.generate_hand_row(
        schedule=promotion.LOCKED_POPULATION,
        entity_id="stage7_m5_r10",
        seed_index=0,
        seat=seat,
    )

    assert promotion.validate_hand_row(row, plan=plan) == row
    assert row["override_fired"] is False
    assert row["candidate_action_key"] == row["baseline_action_key"]
    assert (
        row["candidate_trajectory_sha256"]
        == row["baseline_trajectory_sha256"]
    )
    assert row["candidate_score"] == row["baseline_score"]
    assert row["delta"] == 0.0
    assert row["nonfire_action_key_identical"] is True
    assert row["nonfire_trajectory_identical"] is True
    assert row["nonfire_cancellation_valid"] is True
    assert row["opponent_private_discards_used"] is False
    assert row["current_profile_resolved"] is False


@pytest.mark.parametrize("seat", ["first", "second"])
def test_real_hand_fire_changes_semantic_t3_action(
    tmp_path: Path, seat: str
) -> None:
    runner, plan = _runner(tmp_path, fire=True)
    row = runner.generate_hand_row(
        schedule=promotion.LOCKED_POPULATION,
        entity_id="stage19_p0",
        seed_index=0,
        seat=seat,
    )

    assert promotion.validate_hand_row(row, plan=plan) == row
    assert row["override_fired"] is True
    assert row["candidate_action_key"] != row["baseline_action_key"]
    assert (
        row["candidate_trajectory_sha256"]
        != row["baseline_trajectory_sha256"]
    )
    assert row["nonfire_action_key_identical"] is None
    assert row["nonfire_trajectory_identical"] is None
    assert row["nonfire_cancellation_valid"] is None


def test_runner_covers_exact_five_opponents_and_three_frozen_abr(
    tmp_path: Path,
) -> None:
    runner, plan = _runner(tmp_path, fire=False)
    population_rows = [
        runner.generate_hand_row(
            schedule=promotion.LOCKED_POPULATION,
            entity_id=descriptor["opponent_id"],
            seed_index=0,
            seat="first",
        )
        for descriptor in promotion.OPPONENT_DESCRIPTORS
    ]
    abr_rows = [
        runner.generate_hand_row(
            schedule=promotion.LOCKED_ABR,
            entity_id=descriptor["response_id"],
            seed_index=0,
            seat="second",
        )
        for descriptor in promotion.ABR_DESCRIPTORS
    ]

    assert [row["entity_id"] for row in population_rows] == [
        "stage19_p0",
        "stage9f_p2",
        "stage7_m5_r10",
        "stage3_baseline",
        "random_exact_final",
    ]
    assert [row["entity_id"] for row in abr_rows] == [
        "greedy_search_response",
        "foul_pressure_response",
        "royalty_denial_response",
    ]
    for row in (*population_rows, *abr_rows):
        assert promotion.validate_hand_row(row, plan=plan) == row
        expected = promotion.seed_values(
            row["schedule"], row["seed_index"]
        )
        assert row["hand_seed"] == expected["hand"]
        assert row["actor_policy_seed"] == expected["actor_policy"]
        assert row["opponent_policy_seed"] == expected[
            "opponent_policy"
        ]


def test_shard_is_canonical_write_once_and_source_replayable(
    tmp_path: Path,
) -> None:
    runner, plan = _runner(tmp_path, fire=False)
    output = tmp_path / "population-shard.json"
    shard = runner.run_shard(
        schedule=promotion.LOCKED_POPULATION,
        entity_id="stage9f_p2",
        seed_indices=[0],
        shard_id="population-stage9f-p2-0000",
        output_path=output,
    )

    assert shard["row_count"] == 2
    assert [row["seat"] for row in shard["rows"]] == [
        "first",
        "second",
    ]
    assert promotion.validate_evaluation_shard(
        shard, plan=plan
    ) == shard
    assert output.read_bytes() == promotion.canonical_bytes(shard)

    same = runner.run_shard(
        schedule=promotion.LOCKED_POPULATION,
        entity_id="stage9f_p2",
        seed_indices=[0],
        shard_id="population-stage9f-p2-0000",
        output_path=output,
    )
    assert same == shard
    output.write_bytes(output.read_bytes() + b"\n")
    with pytest.raises(FileExistsError, match="shard changed"):
        runner.run_shard(
            schedule=promotion.LOCKED_POPULATION,
            entity_id="stage9f_p2",
            seed_indices=[0],
            shard_id="population-stage9f-p2-0000",
            output_path=output,
        )


def test_abr_locked_seed_training_and_candidate_promotion_claim_fail_closed(
    tmp_path: Path,
) -> None:
    plan = _make_plan(tmp_path)
    bad_binding = _abr_manifest(
        tmp_path,
        plan=plan,
        response_id="greedy_search_response",
        locked_seed_training_allowed=True,
    )
    with pytest.raises(ValueError, match="boundary changed"):
        subject.validate_abr_policy_binding(bad_binding, plan=plan)


def test_abr_arbitrary_callable_and_unbound_instance_fail_closed(
    tmp_path: Path,
) -> None:
    plan = _make_plan(tmp_path)
    binding = _abr_manifest(
        tmp_path,
        plan=plan,
        response_id="greedy_search_response",
    )
    with pytest.raises(
        subject.LockedPromotionRunnerError,
        match="checkpoint-aware factory",
    ):
        subject.validate_abr_policy_binding(
            replace(binding, policy_factory=_factory()), plan=plan
        )

    def unbound(*, policy_seed: int, seat: str):
        return _FirstLegalPolicy(seat=seat, policy_seed=policy_seed)

    for name in (
        "factory_id",
        "response_id",
        "checkpoint_path",
        "checkpoint_sha256",
    ):
        setattr(unbound, name, getattr(binding.policy_factory, name))
    validated = subject.validate_abr_policy_binding(
        replace(binding, policy_factory=unbound), plan=plan
    )
    with pytest.raises(
        subject.LockedPromotionRunnerError,
        match="policy instance is not bound",
    ):
        validated.make_policy(policy_seed=1, seat="first")

    runner, _plan = _runner(tmp_path / "runner", fire=False)
    original_factory = runner.candidate_policy_factory

    def promoted(*, policy_seed: int, seat: str):
        policy = original_factory(policy_seed=policy_seed, seat=seat)
        policy.authorization_receipt.value[
            "scientific_promotion_passed"
        ] = True
        return policy

    runner.candidate_policy_factory = promoted
    with pytest.raises(
        subject.LockedPromotionRunnerError,
        match="evaluation-only runtime",
    ):
        runner.generate_hand_row(
            schedule=promotion.LOCKED_POPULATION,
            entity_id="stage3_baseline",
            seed_index=0,
            seat="first",
        )


def test_registry_rejects_current_missing_or_reordered_families(
    tmp_path: Path,
) -> None:
    plan = _make_plan(tmp_path)
    opponents = {
        row["opponent_id"]: _factory()
        for row in promotion.OPPONENT_DESCRIPTORS
    }
    abr = {
        row["response_id"]: _abr_manifest(
            tmp_path,
            plan=plan,
            response_id=row["response_id"],
        )
        for row in promotion.ABR_DESCRIPTORS
    }
    with pytest.raises(ValueError, match="five-opponent order"):
        subject.LockedPromotionRunner(
            plan=plan,
            candidate_policy_factory=_candidate_factory(
                plan, fire=False
            ),
            baseline_policy_factory=_factory(),
            opponent_policy_factories={
                "current": _factory(),
                **dict(list(opponents.items())[:-1]),
            },
            abr_policy_bindings=abr,
        )
    with pytest.raises(ValueError, match="three-family order"):
        subject.LockedPromotionRunner(
            plan=plan,
            candidate_policy_factory=_candidate_factory(
                plan, fire=False
            ),
            baseline_policy_factory=_factory(),
            opponent_policy_factories=opponents,
            abr_policy_bindings=dict(reversed(list(abr.items()))),
        )
