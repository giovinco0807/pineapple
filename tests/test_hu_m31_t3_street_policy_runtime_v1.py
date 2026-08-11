from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ofc_regular import hu_m31_t3_street_policy_runtime_v1 as subject
from ofc_regular import hu_m31_t3_street_policy_training_v1 as training
from ofc_regular.action_key import action_key, canonicalize_actions
from ofc_regular.action_space import Action, generate_turn_actions
from ofc_regular.cards import ALL_CARDS
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.state import Board
from ofc_regular.street_policy_net_v1 import (
    FEATURE_SCHEMA_HASH,
    StreetPolicyNetV1Config,
    encode_street_policy_batch,
    model_state_sha256,
)


def _observation(seat: str = "first") -> ActorObservation:
    cards = list(ALL_CARDS)
    random.Random(91_000_003 + (seat == "second")).shuffle(cards)
    cursor = 0

    def take(count: int) -> tuple[str, ...]:
        nonlocal cursor
        value = tuple(cards[cursor : cursor + count])
        cursor += count
        return value

    hero = Board.from_rows(take(2), take(3), take(4))
    opponent = (
        Board.from_rows(take(2), take(3), take(4))
        if seat == "first"
        else Board.from_rows(take(2), take(4), take(5))
    )
    return ActorObservation(
        hero_board=hero,
        opponent_public_board=opponent,
        dealt_cards=take(3),
        hero_private_discards=take(2),
        seat=seat,  # type: ignore[arg-type]
        street="T3",
        to_act_order=seat,  # type: ignore[arg-type]
    )


class _Baseline:
    def __init__(self, *, seat: str, action: Action, seed: int = 818) -> None:
        self.seat = seat
        self.action = action
        self.rng = random.Random(seed)
        self.calls = 0

    def choose_action_observation(
        self,
        observation: ActorObservation,
        *,
        hand_id: str | int | None = None,
        game_id: str | int | None = None,
        decision_seed: int | None = None,
    ) -> Action:
        assert observation.seat == self.seat
        self.calls += 1
        self.rng.random()
        return self.action


def _training_config() -> training.StreetPolicyTrainingConfig:
    return training.StreetPolicyTrainingConfig(
        seed=441,
        ensemble_size=2,
        batch_size=4,
        core_epochs=1,
        risk_epochs=1,
        minimum_lock_fires_per_seat=1,
    )


def _model_config() -> StreetPolicyNetV1Config:
    return StreetPolicyNetV1Config(
        card_embedding_dim=4,
        zone_embedding_dim=2,
        token_hidden_dim=6,
        context_hidden_dim=4,
        seat_embedding_dim=2,
        street_embedding_dim=2,
        state_hidden_dim=8,
        action_hidden_dim=8,
    )


def _threshold_lock(
    *,
    config: training.StreetPolicyTrainingConfig,
    dataset_identity: str,
    model_hashes: list[str],
    threshold: float = 0.5,
) -> dict[str, Any]:
    metrics = {
        "fire_count": 1,
        "gain_sum": 1.0,
        "gain_mean": 1.0,
        "false_positive_count": 0,
        "false_positive_rate": 0.0,
        "loss_p95": 0.0,
        "loss_p99": 0.0,
        "loss_max": 0.0,
    }
    identity = {
        "schema": training.THRESHOLD_LOCK_SCHEMA,
        "source_split": "threshold-lock",
        "training_view_identity_sha256": dataset_identity,
        "training_config_sha256": config.identity_sha256,
        "model_state_sha256": model_hashes,
        "model_state_sha256_after_selection": model_hashes,
        "weight_update_count": 0,
        "seat_thresholds": {
            seat: {
                "enabled": True,
                "safe_probability_threshold": threshold,
                "eligible_count": 1,
                "metrics": dict(metrics),
            }
            for seat in training.SEATS
        },
        "selection_objective": (
            "max_gain_sum_then_mean_then_lower_false_positive_then_count_then_threshold"
        ),
        "gate_formula": (
            "predicted_delta-downside_multiplier*downside_p95-"
            "disagreement_multiplier*ensemble_disagreement>0"
        ),
        "candidate_must_differ_from_baseline": True,
        "teacher_values_are_realized_match_ev": False,
        "current_profile_changed": False,
    }
    result = dict(identity)
    result["threshold_lock_sha256"] = training._canonical_sha256(identity)
    return result


def _receipt(
    *,
    model_manifest_path: Path,
    checkpoint_bundle_identity_sha256: str = "7" * 64,
    training_threshold_lock_file_sha256: str = "8" * 64,
    threshold: float = 0.5,
) -> subject.QualifiedPromotionReceipt:
    return subject.QualifiedPromotionReceipt(
        schema=subject.HU_M31_T3_PROMOTION_RECEIPT_SCHEMA,
        profile_candidate=subject.OPT_IN_PROFILE_CANDIDATE,
        gate_file_sha256="1" * 64,
        plan_sha256="2" * 64,
        merge_sha256="3" * 64,
        model_manifest_file_sha256=subject._sha256_file(model_manifest_path),
        checkpoint_bundle_identity_sha256=(
            checkpoint_bundle_identity_sha256
        ),
        training_threshold_lock_file_sha256=(
            training_threshold_lock_file_sha256
        ),
        compatibility_threshold_lock_file_sha256="4" * 64,
        policy_registry_file_sha256="5" * 64,
        evaluation_runtime_closure_file_sha256="6" * 64,
        seat_safe_probability_thresholds={
            "first": threshold,
            "second": threshold,
        },
        seat_enabled={"first": True, "second": True},
        population_and_abr_source_replayed=True,
        scientific_promotion_passed=True,
        separate_opt_in_profile_candidate_authorized=True,
        named_profile_added=False,
        current_profile_changed=False,
        runtime_activated=False,
        full_replacement_enabled=False,
    )


def _evaluation_receipt(
    *,
    model_manifest_path: Path,
    checkpoint_bundle_identity_sha256: str,
    training_threshold_lock_file_sha256: str,
    threshold: float = 0.5,
) -> subject.LockedEvaluationReceipt:
    return subject.LockedEvaluationReceipt(
        schema=subject.HU_M31_T3_EVALUATION_RECEIPT_SCHEMA,
        runtime_scope=subject.RUNTIME_SCOPE_EVALUATION_ONLY,
        candidate_id=subject.EVALUATION_ONLY_CANDIDATE,
        audit_watermark=subject.EVALUATION_ONLY_WATERMARK,
        plan_file_sha256="9" * 64,
        plan_sha256="a" * 64,
        model_manifest_file_sha256=subject._sha256_file(model_manifest_path),
        checkpoint_bundle_identity_sha256=(
            checkpoint_bundle_identity_sha256
        ),
        training_threshold_lock_file_sha256=(
            training_threshold_lock_file_sha256
        ),
        compatibility_threshold_lock_file_sha256="b" * 64,
        policy_registry_file_sha256="c" * 64,
        evaluation_runtime_closure_file_sha256="d" * 64,
        seat_safe_probability_thresholds={
            "first": threshold,
            "second": threshold,
        },
        seat_enabled={"first": True, "second": True},
        plan_and_artifacts_source_replayed=True,
        evaluation_only=True,
        promotion_gate_required_for_evaluation=False,
        scientific_promotion_passed=False,
        separate_opt_in_profile_candidate_authorized=False,
        named_profile_added=False,
        current_profile_changed=False,
        runtime_activated=False,
        full_replacement_enabled=False,
    )


@pytest.fixture()
def locked_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[
    subject.HuM31T3StreetPolicyRuntime,
    _Baseline,
    ActorObservation,
    list[Action],
    Any,
]:
    torch = pytest.importorskip("torch")
    config = _training_config()
    dataset_identity = "a" * 64
    models = training.create_deterministic_ensemble(
        torch,
        training_config=config,
        model_config=_model_config(),
    )
    # Identical members make ensemble disagreement exactly zero.  Frozen risk
    # heads make downside negligible and safety high, while the randomly
    # initialized delta head still ranks semantic actions.
    models[1].load_state_dict(models[0].state_dict())
    with torch.no_grad():
        for model in models:
            model.uncertainty_head.weight.zero_()
            model.uncertainty_head.bias.fill_(-20.0)
            model.safe_head.weight.zero_()
            model.safe_head.bias.fill_(20.0)
    bundle = tmp_path / "bundle"
    manifest = training.write_ensemble_checkpoint_bundle(
        bundle,
        models,
        dataset=SimpleNamespace(identity_sha256=dataset_identity),
        training_config=config,
        stage="risk",
        completed_epoch=1,
    )
    model_hashes = [
        str(record["model_state_sha256"]) for record in manifest["models"]
    ]
    threshold = _threshold_lock(
        config=config,
        dataset_identity=dataset_identity,
        model_hashes=model_hashes,
    )
    threshold_path = tmp_path / "threshold.json"
    threshold_raw = training._canonical_bytes(threshold)
    threshold_path.write_bytes(threshold_raw)
    observation = _observation()
    actions = canonicalize_actions(
        generate_turn_actions(
            observation.hero_board, observation.dealt_cards
        )
    )
    baseline = _Baseline(seat="first", action=actions[0])

    def qualified(**kwargs: Any) -> subject.QualifiedPromotionReceipt:
        return _receipt(
            model_manifest_path=kwargs["model_manifest_path"],
            checkpoint_bundle_identity_sha256=kwargs[
                "checkpoint_bundle_identity_sha256"
            ],
            training_threshold_lock_file_sha256=kwargs[
                "training_threshold_lock_file_sha256"
            ],
        )

    monkeypatch.setattr(
        subject, "_load_qualified_promotion_receipt", qualified
    )
    runtime = subject.build_opt_in_t3_policy_candidate(
        baseline_policy=baseline,
        baseline_profile_id=subject.BASELINE_PROFILE,
        torch=torch,
        checkpoint_bundle_path=bundle,
        expected_dataset_identity_sha256=dataset_identity,
        training_config=config,
        expected_bundle_identity_sha256=manifest[
            "bundle_identity_sha256"
        ],
        threshold_lock_path=threshold_path,
        expected_threshold_lock_file_sha256=hashlib.sha256(
            threshold_raw
        ).hexdigest(),
        promotion_evidence=subject.PromotionEvidencePaths(
            plan_path=tmp_path / "unused-plan",
            merge_path=tmp_path / "unused-merge",
            gate_path=tmp_path / "unused-gate",
            expected_gate_file_sha256="f" * 64,
            compatibility_threshold_lock_path=tmp_path / "unused-compat",
            policy_registry_path=tmp_path / "unused-registry",
            evaluation_runtime_closure_path=Path(subject.__file__),
        ),
    )
    return runtime, baseline, observation, actions, torch


def _prediction_for_baseline(
    runtime: subject.HuM31T3StreetPolicyRuntime,
    observation: ActorObservation,
    actions: list[Action],
    baseline_index: int,
) -> training.GatePrediction:
    keys = tuple(action_key(action) for action in actions)
    encoded = encode_street_policy_batch(
        [observation],
        [keys],
        [keys[baseline_index]],
    )
    return runtime._predict(
        observation=observation,
        encoded=encoded,
        baseline_key=keys[baseline_index],
        baseline_index=baseline_index,
        legal_count=len(actions),
    )


def test_runtime_prediction_is_exactly_training_gate_semantics(
    locked_runtime: tuple[
        subject.HuM31T3StreetPolicyRuntime,
        _Baseline,
        ActorObservation,
        list[Action],
        Any,
    ],
) -> None:
    runtime, _baseline, observation, actions, torch = locked_runtime
    keys = tuple(action_key(action).to_token() for action in actions)
    baseline_index = len(actions) - 1
    runtime_prediction = _prediction_for_baseline(
        runtime, observation, actions, baseline_index
    )
    count = len(keys)
    example = training.PolicyTrainingExample(
        identity=observation.fingerprint(),
        split_role="diagnostic-holdout",
        seat="first",
        observation=observation.to_dict(),
        legal_action_keys=keys,
        baseline_action_key=keys[baseline_index],
        action_q=(0.0,) * count,
        baseline_delta=(0.0,) * count,
        teacher_policy=(1.0 / count,) * count,
        state_value=0.0,
    )
    training_prediction = training.predict_safe_gate(
        torch,
        runtime._models,
        [example],
        training_config=runtime._training_config,
    )[0]
    assert runtime_prediction == training_prediction


def test_nonfire_returns_same_baseline_object_and_consumes_no_extra_rng(
    locked_runtime: tuple[
        subject.HuM31T3StreetPolicyRuntime,
        _Baseline,
        ActorObservation,
        list[Action],
        Any,
    ],
) -> None:
    runtime, baseline, observation, actions, torch = locked_runtime
    first_prediction = _prediction_for_baseline(
        runtime, observation, actions, 0
    )
    baseline.action = actions[first_prediction.action_index]
    twin = _Baseline(seat="first", action=baseline.action)
    python_rng_before = random.getstate()
    torch_rng_before = torch.get_rng_state().clone()
    model_hashes_before = [
        model_state_sha256(model) for model in runtime._models
    ]

    returned, decision = runtime.choose_action_observation_with_audit(
        observation,
        hand_id=7,
        game_id=8,
        decision_seed=9,
    )
    direct = twin.choose_action_observation(
        observation,
        hand_id=7,
        game_id=8,
        decision_seed=9,
    )

    assert decision is not None
    assert runtime.evaluation_only is False
    assert (
        runtime.runtime_scope
        == subject.RUNTIME_SCOPE_QUALIFIED_OPT_IN
    )
    assert runtime.profile_candidate == subject.OPT_IN_PROFILE_CANDIDATE
    assert returned is baseline.action
    assert returned is direct
    assert decision.override_fired is False
    assert decision.evaluation_only is False
    assert decision.audit_watermark is None
    assert decision.promotion_gate_file_sha256 == "1" * 64
    assert decision.nonfire_reason == "candidate_matches_baseline"
    assert decision.final_action_key == decision.baseline_action_key
    assert baseline.calls == 1
    assert twin.calls == 1
    assert baseline.rng.getstate() == twin.rng.getstate()
    assert random.getstate() == python_rng_before
    assert torch.equal(torch.get_rng_state(), torch_rng_before)
    assert [
        model_state_sha256(model) for model in runtime._models
    ] == model_hashes_before


def test_spawn_with_baseline_shares_frozen_bundle_and_preserves_nonfire(
    locked_runtime: tuple[
        subject.HuM31T3StreetPolicyRuntime,
        _Baseline,
        ActorObservation,
        list[Action],
        Any,
    ],
) -> None:
    runtime, _baseline, observation, actions, _torch = locked_runtime
    prediction = _prediction_for_baseline(runtime, observation, actions, 0)
    fresh_baseline = _Baseline(
        seat="first",
        action=actions[prediction.action_index],
    )

    spawned = runtime.spawn_with_baseline(
        fresh_baseline,
        baseline_profile_id=subject.BASELINE_PROFILE,
    )
    returned, decision = spawned.choose_action_observation_with_audit(
        observation,
        hand_id=71,
        game_id=81,
        decision_seed=91,
    )

    assert spawned is not runtime
    assert all(
        spawned_model is source_model
        for spawned_model, source_model in zip(
            spawned._models,
            runtime._models,
            strict=True,
        )
    )
    assert (
        spawned.authorization_receipt
        is runtime.authorization_receipt
    )
    assert spawned.runtime_scope == runtime.runtime_scope
    assert returned is fresh_baseline.action
    assert decision is not None
    assert decision.override_fired is False
    assert decision.nonfire_reason == "candidate_matches_baseline"
    assert fresh_baseline.calls == 1

    with pytest.raises(ValueError, match="explicit stage7_m5_r10"):
        runtime.spawn_with_baseline(
            fresh_baseline,
            baseline_profile_id="current",
        )


def test_safe_gate_can_fire_and_resolves_canonical_action(
    locked_runtime: tuple[
        subject.HuM31T3StreetPolicyRuntime,
        _Baseline,
        ActorObservation,
        list[Action],
        Any,
    ],
) -> None:
    runtime, baseline, observation, actions, _torch = locked_runtime
    predictions = [
        _prediction_for_baseline(runtime, observation, actions, index)
        for index in range(len(actions))
    ]
    prediction = max(predictions, key=lambda value: value.predicted_delta)
    assert prediction.predicted_delta > 0.0
    assert prediction.action_index != prediction.baseline_index
    baseline.action = actions[prediction.baseline_index]

    returned, decision = runtime.choose_action_observation_with_audit(
        observation
    )

    assert decision is not None
    assert decision.override_fired is True
    assert decision.nonfire_reason is None
    assert action_key(returned).to_token() == decision.candidate_action_key
    assert returned is not baseline.action
    assert baseline.calls == 1


def test_runtime_prediction_is_card_order_invariant(
    locked_runtime: tuple[
        subject.HuM31T3StreetPolicyRuntime,
        _Baseline,
        ActorObservation,
        list[Action],
        Any,
    ],
) -> None:
    runtime, _baseline, observation, actions, _torch = locked_runtime
    permuted = ActorObservation(
        hero_board=Board.from_rows(
            tuple(reversed(observation.hero_board.top)),
            tuple(reversed(observation.hero_board.middle)),
            tuple(reversed(observation.hero_board.bottom)),
        ),
        opponent_public_board=Board.from_rows(
            tuple(reversed(observation.opponent_public_board.top)),
            tuple(reversed(observation.opponent_public_board.middle)),
            tuple(reversed(observation.opponent_public_board.bottom)),
        ),
        dealt_cards=tuple(reversed(observation.dealt_cards)),
        hero_private_discards=tuple(
            reversed(observation.hero_private_discards)
        ),
        seat=observation.seat,
        street=observation.street,
        to_act_order=observation.to_act_order,
        scoring=observation.scoring,
    )
    permuted_actions = canonicalize_actions(
        generate_turn_actions(
            permuted.hero_board, permuted.dealt_cards
        )
    )
    assert [action_key(action) for action in permuted_actions] == [
        action_key(action) for action in actions
    ]
    baseline_index = len(actions) // 2
    original_prediction = _prediction_for_baseline(
        runtime, observation, actions, baseline_index
    )
    permuted_prediction = _prediction_for_baseline(
        runtime, permuted, permuted_actions, baseline_index
    )
    assert original_prediction == permuted_prediction


def test_same_locked_runtime_handles_second_seat(
    locked_runtime: tuple[
        subject.HuM31T3StreetPolicyRuntime,
        _Baseline,
        ActorObservation,
        list[Action],
        Any,
    ],
) -> None:
    runtime, baseline, _first_observation, _first_actions, _torch = (
        locked_runtime
    )
    observation = _observation("second")
    actions = canonicalize_actions(
        generate_turn_actions(
            observation.hero_board, observation.dealt_cards
        )
    )
    baseline.seat = "second"
    prediction = _prediction_for_baseline(
        runtime, observation, actions, 0
    )
    baseline.action = actions[prediction.action_index]

    returned, decision = runtime.choose_action_observation_with_audit(
        observation
    )

    assert decision is not None
    assert decision.seat == "second"
    assert returned is baseline.action
    assert decision.override_fired is False
    assert decision.nonfire_reason == "candidate_matches_baseline"
    assert baseline.calls == 1


def test_evaluation_factory_is_watermarked_and_cannot_claim_promotion(
    locked_runtime: tuple[
        subject.HuM31T3StreetPolicyRuntime,
        _Baseline,
        ActorObservation,
        list[Action],
        Any,
    ],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualified, baseline, observation, actions, torch = locked_runtime
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(
        training._canonical_bytes(qualified._checkpoint_manifest)
    )
    threshold_file_sha = hashlib.sha256(
        qualified._threshold_lock_bytes
    ).hexdigest()
    loaded = subject._LoadedRuntimeArtifacts(
        models=qualified._models,
        checkpoint_manifest=qualified._checkpoint_manifest,
        threshold_lock=qualified._threshold_lock(),
        threshold_lock_bytes=qualified._threshold_lock_bytes,
        threshold_lock_file_sha256=threshold_file_sha,
        model_manifest_path=manifest_path,
    )
    receipt = _evaluation_receipt(
        model_manifest_path=manifest_path,
        checkpoint_bundle_identity_sha256=qualified._checkpoint_manifest[
            "bundle_identity_sha256"
        ],
        training_threshold_lock_file_sha256=threshold_file_sha,
    )
    monkeypatch.setattr(
        subject, "_load_runtime_artifacts", lambda **kwargs: loaded
    )
    monkeypatch.setattr(
        subject,
        "_load_locked_evaluation_receipt",
        lambda **kwargs: receipt,
    )
    evaluation = subject.build_locked_evaluation_t3_policy_candidate(
        baseline_policy=baseline,
        baseline_profile_id=subject.BASELINE_PROFILE,
        torch=torch,
        checkpoint_bundle_path=tmp_path / "unused-bundle",
        expected_dataset_identity_sha256="1" * 64,
        training_config=qualified._training_config,
        expected_bundle_identity_sha256="2" * 64,
        threshold_lock_path=tmp_path / "unused-threshold",
        expected_threshold_lock_file_sha256="3" * 64,
        evaluation_evidence=subject.LockedEvaluationEvidencePaths(
            plan_path=tmp_path / "unused-plan",
            expected_plan_file_sha256="4" * 64,
            compatibility_threshold_lock_path=tmp_path / "unused-compat",
            policy_registry_path=tmp_path / "unused-registry",
            evaluation_runtime_closure_path=Path(subject.__file__),
        ),
    )
    prediction = _prediction_for_baseline(
        evaluation, observation, actions, 0
    )
    baseline.action = actions[prediction.action_index]

    returned, decision = evaluation.choose_action_observation_with_audit(
        observation
    )

    assert evaluation.evaluation_only is True
    assert evaluation.runtime_scope == subject.RUNTIME_SCOPE_EVALUATION_ONLY
    assert evaluation.profile_candidate == subject.EVALUATION_ONLY_CANDIDATE
    assert evaluation.authorization_receipt is receipt
    with pytest.raises(PermissionError, match="no promotion-qualified"):
        _ = evaluation.promotion_receipt
    assert returned is baseline.action
    assert decision is not None
    assert decision.evaluation_only is True
    assert decision.audit_watermark == subject.EVALUATION_ONLY_WATERMARK
    assert decision.promotion_gate_file_sha256 is None
    assert decision.runtime_activated is False
    assert decision.current_profile_resolved is False


def test_strict_infoset_legacy_t3_and_model_tamper_fail_closed(
    locked_runtime: tuple[
        subject.HuM31T3StreetPolicyRuntime,
        _Baseline,
        ActorObservation,
        list[Action],
        Any,
    ],
) -> None:
    runtime, baseline, observation, _actions, torch = locked_runtime
    with pytest.raises(PermissionError, match="promotion-gated"):
        subject.HuM31T3StreetPolicyRuntime(
            baseline_policy=baseline,
            torch=torch,
            models=runtime._models,
            training_config=runtime._training_config,
            threshold_lock_bytes=runtime._threshold_lock_bytes,
            checkpoint_manifest=runtime._checkpoint_manifest,
            authorization_receipt=runtime.promotion_receipt,
            runtime_scope=subject.RUNTIME_SCOPE_QUALIFIED_OPT_IN,
        )
    hidden = observation.to_dict()
    hidden["opponent_private_discards"] = ["2h"]
    with pytest.raises(TypeError, match="requires ActorObservation"):
        runtime.choose_action_observation(hidden)  # type: ignore[arg-type]
    assert baseline.calls == 0

    with pytest.raises(subject.HuM31T3RuntimeError, match="requires"):
        runtime.choose_action(
            observation.hero_board, observation.dealt_cards
        )
    assert baseline.calls == 0

    with torch.no_grad():
        parameter = next(runtime._models[0].parameters())
        parameter.view(-1)[0].add_(1.0)
    with pytest.raises(subject.HuM31T3RuntimeError, match="model state"):
        runtime.choose_action_observation(observation)
    assert baseline.calls == 0


def test_qualified_receipt_requires_pinned_pass_and_current_runtime_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = tmp_path / "plan.json"
    merge_path = tmp_path / "merge.json"
    gate_path = tmp_path / "gate.json"
    model_path = tmp_path / "manifest.json"
    compatibility_path = tmp_path / "compat.json"
    registry_path = tmp_path / "ai_profiles.py"
    for path, value in (
        (plan_path, {"kind": "plan"}),
        (merge_path, {"kind": "merge"}),
        (gate_path, {"kind": "gate"}),
        (model_path, {"kind": "model"}),
        (compatibility_path, {"kind": "compatibility"}),
    ):
        path.write_bytes(subject._canonical_bytes(value))
    registry_path.write_bytes(b"unchanged registry")
    gate_sha = subject._sha256_file(gate_path)
    training_lock = {
        "seat_thresholds": {
            "first": {
                "safe_probability_threshold": 0.5,
                "enabled": True,
            },
            "second": {
                "safe_probability_threshold": 0.5,
                "enabled": True,
            },
        }
    }
    passing_gate = {
        "status": "pass",
        "all_gates_passed": True,
        "scientific_promotion_passed": True,
        "separate_opt_in_profile_candidate_authorized": True,
        "teacher_values_used": False,
        "gates": {"complete": True},
        "named_profile_added": False,
        "current_profile_changed": False,
        "runtime_activated": False,
        "full_replacement_enabled": False,
    }
    binding = {
        "model": {"sha256": subject._sha256_file(model_path)},
        "threshold_lock": {
            "sha256": subject._sha256_file(compatibility_path)
        },
        "threshold_lock_content": {
            "state_action_input_schema_sha256": FEATURE_SCHEMA_HASH,
            "seat_thresholds": {"first": 0.5, "second": 0.5},
            "seat_enabled": {"first": True, "second": True},
            "source_training_threshold_lock_sha256": "7" * 64,
            "source_checkpoint_bundle_identity_sha256": "8" * 64,
        },
        "policy_registry": {"sha256": subject._sha256_file(registry_path)},
        "evaluation_runtime_closure": {
            "sha256": subject._sha256_file(Path(subject.__file__))
        },
    }
    monkeypatch.setattr(
        subject.promotion,
        "validate_locked_promotion_gate",
        lambda *args, **kwargs: dict(passing_gate),
    )
    monkeypatch.setattr(
        subject.promotion,
        "validate_artifact_files",
        lambda *args, **kwargs: None,
    )
    validated_plan = {
        "artifact_binding": binding,
        "evaluation_contract": {
            "baseline_profile": subject.BASELINE_PROFILE
        },
        "named_profile_added": False,
        "current_profile_changed": False,
        "runtime_activated": False,
        "full_replacement_enabled": False,
        "cloud_execution_started": False,
    }
    monkeypatch.setattr(
        subject.promotion,
        "validate_locked_promotion_plan",
        lambda value: validated_plan,
    )
    monkeypatch.setattr(
        subject.promotion,
        "canonical_sha256",
        lambda value: hashlib.sha256(
            json.dumps(value, sort_keys=True).encode()
        ).hexdigest(),
    )
    def replay_fixture_closure(*, closure_path, validated_plan):
        if Path(closure_path).resolve() != Path(subject.__file__).resolve():
            raise subject.HuM31T3RuntimeError(
                "fixture runtime closure changed"
            )
        return {}

    monkeypatch.setattr(
        subject, "_replay_runtime_closure", replay_fixture_closure
    )
    evidence = subject.PromotionEvidencePaths(
        plan_path=plan_path,
        merge_path=merge_path,
        gate_path=gate_path,
        expected_gate_file_sha256=gate_sha,
        compatibility_threshold_lock_path=compatibility_path,
        policy_registry_path=registry_path,
        evaluation_runtime_closure_path=Path(subject.__file__),
    )
    evaluation_receipt = subject._load_locked_evaluation_receipt(
        evidence=subject.LockedEvaluationEvidencePaths(
            plan_path=plan_path,
            expected_plan_file_sha256=subject._sha256_file(plan_path),
            compatibility_threshold_lock_path=compatibility_path,
            policy_registry_path=registry_path,
            evaluation_runtime_closure_path=Path(subject.__file__),
        ),
        model_manifest_path=model_path,
        training_threshold_lock=training_lock,
        training_threshold_lock_file_sha256="7" * 64,
        checkpoint_bundle_identity_sha256="8" * 64,
    )
    assert evaluation_receipt.evaluation_only is True
    assert evaluation_receipt.scientific_promotion_passed is False
    assert (
        evaluation_receipt.separate_opt_in_profile_candidate_authorized
        is False
    )
    assert (
        evaluation_receipt.audit_watermark
        == subject.EVALUATION_ONLY_WATERMARK
    )

    receipt = subject._load_qualified_promotion_receipt(
        evidence=evidence,
        model_manifest_path=model_path,
        training_threshold_lock=training_lock,
        training_threshold_lock_file_sha256="7" * 64,
        checkpoint_bundle_identity_sha256="8" * 64,
    )
    assert receipt.population_and_abr_source_replayed is True
    assert receipt.separate_opt_in_profile_candidate_authorized is True

    no_go = dict(passing_gate)
    no_go["status"] = "no_go"
    no_go["all_gates_passed"] = False
    no_go["scientific_promotion_passed"] = False
    no_go["separate_opt_in_profile_candidate_authorized"] = False
    monkeypatch.setattr(
        subject.promotion,
        "validate_locked_promotion_gate",
        lambda *args, **kwargs: no_go,
    )
    with pytest.raises(PermissionError, match="activation remains closed"):
        subject._load_qualified_promotion_receipt(
            evidence=evidence,
            model_manifest_path=model_path,
            training_threshold_lock=training_lock,
            training_threshold_lock_file_sha256="7" * 64,
            checkpoint_bundle_identity_sha256="8" * 64,
        )

    monkeypatch.setattr(
        subject.promotion,
        "validate_locked_promotion_gate",
        lambda *args, **kwargs: dict(passing_gate),
    )
    other_runtime = tmp_path / "other-runtime.py"
    other_runtime.write_bytes(b"not this closure")
    wrong_runtime = subject.PromotionEvidencePaths(
        **{
            **evidence.__dict__,
            "evaluation_runtime_closure_path": other_runtime,
        }
    )
    with pytest.raises(subject.HuM31T3RuntimeError, match="runtime closure"):
        subject._load_qualified_promotion_receipt(
            evidence=wrong_runtime,
            model_manifest_path=model_path,
            training_threshold_lock=training_lock,
            training_threshold_lock_file_sha256="7" * 64,
            checkpoint_bundle_identity_sha256="8" * 64,
        )
