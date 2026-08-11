from __future__ import annotations

import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import hu_m31_t3_dataset_contract_v1 as dataset_contract
from ofc_regular import hu_m31_t3_street_policy_training_v1 as subject
from ofc_regular.action_key import action_key, canonicalize_actions
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.cards import ALL_CARDS
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.state import Board
from ofc_regular.street_policy_net_v1 import (
    StreetPolicyNetV1Config,
    model_state_sha256,
    parameter_names_for_update,
)


def _observation(index: int, seat: str) -> ActorObservation:
    cards = list(ALL_CARDS)
    random.Random(8_700_000 + index * 2 + (seat == "second")).shuffle(cards)
    cursor = 0

    def take(count: int) -> tuple[str, ...]:
        nonlocal cursor
        result = tuple(cards[cursor : cursor + count])
        cursor += count
        return result

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


def _example(
    split: str,
    index: int,
    seat: str,
    *,
    confirmed_gain: float = 1.0,
) -> subject.PolicyTrainingExample:
    observation = _observation(index, seat)
    actions = canonicalize_actions(
        generate_turn_actions(observation.hero_board, observation.dealt_cards)
    )
    keys = tuple(action_key(action).to_token() for action in actions)
    count = len(keys)
    baseline_index = count - 1
    q = tuple(
        1.0 - action_index / max(count - 1, 1)
        for action_index in range(count)
    )
    baseline = q[baseline_index]
    delta = tuple(value - baseline for value in q)
    logits = [pow(2.0, value) for value in q]
    mass = sum(logits)
    teacher_policy = tuple(value / mass for value in logits)
    confirmed = list(delta)
    confirmed[0] = confirmed_gain
    confirmed[baseline_index] = 0.0
    downside = tuple(
        max(0.0, primary - truth)
        for primary, truth in zip(delta, confirmed, strict=True)
    )
    safe = tuple(1.0 if truth > 0 else 0.0 for truth in confirmed)
    return subject.PolicyTrainingExample(
        identity=f"{split}:{seat}:{index:03d}",
        split_role=split,
        seat=seat,
        observation=observation.to_dict(),
        legal_action_keys=keys,
        baseline_action_key=keys[baseline_index],
        action_q=q,
        baseline_delta=delta,
        teacher_policy=teacher_policy,
        state_value=max(q),
        downside_p95=downside,
        safe=safe,
        confirmation_delta=tuple(confirmed),
    )


def _dataset() -> subject.PolicyTrainingDataset:
    examples = []
    cursor = 0
    for split in subject.SPLIT_ROLES:
        for seat in subject.SEATS:
            for local in range(2):
                gain = 1.0
                if split == "threshold-lock" and local == 1 and seat == "first":
                    gain = -0.2
                examples.append(_example(split, cursor, seat, confirmed_gain=gain))
                cursor += 1
    return subject.build_policy_training_dataset(
        examples,
        source_dataset_identity_sha256="a" * 64,
        synthetic_cpu_smoke=True,
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


def _config(*, core_epochs: int = 1) -> subject.StreetPolicyTrainingConfig:
    return subject.StreetPolicyTrainingConfig(
        seed=1234,
        ensemble_size=2,
        batch_size=4,
        core_epochs=core_epochs,
        risk_epochs=1,
        core_learning_rate=1e-3,
        risk_learning_rate=1e-3,
        disagreement_multiplier=0.01,
        downside_multiplier=0.01,
        maximum_false_positive_rate=0.5,
        minimum_lock_fires_per_seat=1,
    )


def _group_hash(model: Any, scope: str) -> str:
    parameters = dict(model.named_parameters())
    digest = hashlib.sha256()
    for name in parameter_names_for_update(model, scope):
        digest.update(name.encode("utf-8"))
        digest.update(
            parameters[name].detach().cpu().contiguous().numpy().tobytes()
        )
    return digest.hexdigest()


def test_training_view_is_immutable_and_rejects_hidden_truth() -> None:
    dataset = _dataset()
    dataset.validate()
    assert dataset.manifest["source_type"] == "synthetic_cpu_smoke"
    assert dataset.manifest["split_counts"] == {
        "train": 4,
        "safety-fit": 4,
        "threshold-lock": 4,
        "diagnostic-holdout": 4,
    }

    changed_manifest = dict(dataset.manifest)
    changed_manifest["split_counts"] = dict(changed_manifest["split_counts"])
    changed_manifest["split_counts"]["train"] = 5
    with pytest.raises(ValueError, match="training view changed"):
        subject.PolicyTrainingDataset(
            examples=dataset.examples, manifest=changed_manifest
        ).validate()

    original = dataset.examples[0]
    hidden = dict(original.observation)
    hidden["opponent_private_discards"] = ["2h"]
    with pytest.raises(ValueError, match="unknown fields"):
        replace(original, observation=hidden)


def test_production_split_view_requires_exact_pair_seat_confirmation_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset()
    monkeypatch.setattr(
        dataset_contract,
        "SPLIT_SPECS",
        tuple(
            {
                "split": split,
                "paired_hand_count": 2,
            }
            for split in subject.SPLIT_ROLES
        ),
    )
    monkeypatch.setattr(dataset_contract, "CONFIRMATION_MODULUS", 1)

    subject._validate_production_split_view(dataset.examples)
    missing_second_seat = tuple(
        example
        for example in dataset.examples
        if example.identity != "threshold-lock:second:011"
    )
    with pytest.raises(
        ValueError,
        match="threshold-lock training-view pair/seat/confirmation coverage changed",
    ):
        subject._validate_production_split_view(missing_second_seat)

    unconfirmed = list(dataset.examples)
    index = next(
        i
        for i, example in enumerate(unconfirmed)
        if example.split_role == "safety-fit"
    )
    unconfirmed[index] = replace(
        unconfirmed[index],
        downside_p95=None,
        safe=None,
        confirmation_delta=None,
    )
    with pytest.raises(
        ValueError,
        match="safety-fit training-view pair/seat/confirmation coverage changed",
    ):
        subject._validate_production_split_view(unconfirmed)


def test_train_updates_only_core_and_safety_updates_only_risk() -> None:
    torch = pytest.importorskip("torch")
    dataset = _dataset()
    config = _config()
    models = subject.create_deterministic_ensemble(
        torch, training_config=config, model_config=_model_config()
    )
    core_before = [_group_hash(model, "core") for model in models]
    risk_before = [_group_hash(model, "risk") for model in models]
    core_receipt = subject.fit_core_from_train(
        torch, models, dataset, training_config=config
    )
    assert core_receipt["split_role"] == "train"
    assert core_receipt["unauthorized_parameter_change_count"] == 0
    assert [_group_hash(model, "risk") for model in models] == risk_before
    assert [_group_hash(model, "core") for model in models] != core_before

    core_after = [_group_hash(model, "core") for model in models]
    risk_receipt = subject.fit_risk_from_safety(
        torch, models, dataset, training_config=config
    )
    assert risk_receipt["split_role"] == "safety-fit"
    assert [_group_hash(model, "core") for model in models] == core_after
    assert [_group_hash(model, "risk") for model in models] != risk_before

    with pytest.raises(PermissionError, match="not authorized"):
        subject._fit_scope(  # type: ignore[attr-defined]
            torch,
            models,
            dataset,
            training_config=config,
            split_role="threshold-lock",
            update_scope="core",
            start_epoch=0,
            end_epoch=1,
        )


def test_checkpoint_is_deterministic_and_epoch_resume_is_exact(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    dataset = _dataset()
    config = _config(core_epochs=2)

    direct = subject.create_deterministic_ensemble(
        torch, training_config=config, model_config=_model_config()
    )
    subject.fit_core_from_train(
        torch, direct, dataset, training_config=config, start_epoch=0, end_epoch=2
    )

    resumed = subject.create_deterministic_ensemble(
        torch, training_config=config, model_config=_model_config()
    )
    subject.fit_core_from_train(
        torch, resumed, dataset, training_config=config, start_epoch=0, end_epoch=1
    )
    first_bundle = tmp_path / "epoch1-a"
    manifest_a = subject.write_ensemble_checkpoint_bundle(
        first_bundle,
        resumed,
        dataset=dataset,
        training_config=config,
        stage="core",
        completed_epoch=1,
    )
    second_bundle = tmp_path / "epoch1-b"
    manifest_b = subject.write_ensemble_checkpoint_bundle(
        second_bundle,
        resumed,
        dataset=dataset,
        training_config=config,
        stage="core",
        completed_epoch=1,
    )
    assert manifest_a == manifest_b
    assert (first_bundle / "manifest.json").read_bytes() == (
        second_bundle / "manifest.json"
    ).read_bytes()
    for index in range(config.ensemble_size):
        assert (first_bundle / f"model_{index:02d}.zip").read_bytes() == (
            second_bundle / f"model_{index:02d}.zip"
        ).read_bytes()

    loaded, loaded_manifest = subject.load_ensemble_checkpoint_bundle(
        first_bundle,
        torch=torch,
        expected_dataset_identity_sha256=dataset.identity_sha256,
        expected_training_config=config,
        expected_stage="core",
        expected_bundle_identity_sha256=manifest_a[
            "bundle_identity_sha256"
        ],
    )
    assert loaded_manifest["completed_epoch"] == 1
    subject.fit_core_from_train(
        torch, loaded, dataset, training_config=config, start_epoch=1, end_epoch=2
    )
    assert [model_state_sha256(model) for model in loaded] == [
        model_state_sha256(model) for model in direct
    ]

    (first_bundle / "unknown.txt").write_text("tamper", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown files"):
        subject.load_ensemble_checkpoint_bundle(
            first_bundle,
            torch=torch,
            expected_dataset_identity_sha256=dataset.identity_sha256,
            expected_training_config=config,
            expected_stage="core",
            expected_bundle_identity_sha256=manifest_a[
                "bundle_identity_sha256"
            ],
        )


def test_threshold_lock_is_seat_specific_and_holdout_is_diagnostic_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    dataset = _dataset()
    config = _config()
    models = subject.create_deterministic_ensemble(
        torch, training_config=config, model_config=_model_config()
    )
    model_hashes = [model_state_sha256(model) for model in models]

    def fake_predictions(
        _torch: Any,
        _models: Any,
        examples: Any,
        *,
        training_config: Any,
        batch_size: Any = None,
    ) -> list[subject.GatePrediction]:
        del _torch, _models, training_config, batch_size
        result = []
        seat_seen = {"first": 0, "second": 0}
        for example in sorted(examples, key=lambda row: row.identity):
            local = seat_seen[example.seat]
            seat_seen[example.seat] += 1
            if example.split_role == "threshold-lock":
                probability = (
                    (0.90, 0.40)[local]
                    if example.seat == "first"
                    else (0.70, 0.60)[local]
                )
            else:
                probability = 0.95
            result.append(
                subject.GatePrediction(
                    example_identity=example.identity,
                    seat=example.seat,
                    action_key=example.legal_action_keys[0],
                    action_index=0,
                    baseline_action_key=example.baseline_action_key,
                    baseline_index=example.legal_action_keys.index(
                        example.baseline_action_key
                    ),
                    predicted_delta=1.0,
                    downside_p95=0.1,
                    ensemble_disagreement=0.05,
                    safe_probability=probability,
                    lower_bound=0.9985,
                )
            )
        return result

    monkeypatch.setattr(subject, "predict_safe_gate", fake_predictions)
    locked = subject.lock_seat_thresholds(
        torch, models, dataset, training_config=config
    )
    assert locked["weight_update_count"] == 0
    assert locked["seat_thresholds"]["first"]["enabled"] is True
    assert (
        locked["seat_thresholds"]["first"]["safe_probability_threshold"] == 0.90
    )
    assert locked["seat_thresholds"]["second"]["enabled"] is True
    assert (
        locked["seat_thresholds"]["second"]["safe_probability_threshold"] == 0.60
    )
    assert [model_state_sha256(model) for model in models] == model_hashes

    report = subject.report_diagnostic_holdout(
        torch,
        models,
        dataset,
        training_config=config,
        threshold_lock=locked,
    )
    assert report["diagnostic_only"] is True
    assert report["promotion_authorized"] is False
    assert report["threshold_research_performed"] is False
    assert report["weight_update_count"] == 0
    assert report["metrics"]["overall"]["fire_count"] == 4
    assert [model_state_sha256(model) for model in models] == model_hashes

    prediction = fake_predictions(
        torch,
        models,
        dataset.for_split("diagnostic-holdout"),
        training_config=config,
    )[0]
    assert subject.safe_override_decision(
        prediction, locked, training_config=config
    ) is True
    assert subject.safe_override_decision(
        replace(prediction, predicted_delta=0.0, lower_bound=-0.0015),
        locked,
        training_config=config,
    ) is False
    assert subject.safe_override_decision(
        replace(prediction, downside_p95=200.0, lower_bound=-1.0005),
        locked,
        training_config=config,
    ) is False


def test_actual_cpu_model_prediction_contains_disagreement_and_downside() -> None:
    torch = pytest.importorskip("torch")
    dataset = _dataset()
    config = _config()
    models = subject.create_deterministic_ensemble(
        torch, training_config=config, model_config=_model_config()
    )
    subject.fit_core_from_train(torch, models, dataset, training_config=config)
    subject.fit_risk_from_safety(torch, models, dataset, training_config=config)
    predictions = subject.predict_safe_gate(
        torch,
        models,
        dataset.for_split("threshold-lock"),
        training_config=config,
    )
    assert len(predictions) == 4
    assert all(prediction.downside_p95 >= 0 for prediction in predictions)
    assert all(
        prediction.ensemble_disagreement >= 0 for prediction in predictions
    )
    for prediction in predictions:
        assert prediction.lower_bound == pytest.approx(
            prediction.predicted_delta
            - config.downside_multiplier * prediction.downside_p95
            - config.disagreement_multiplier
            * prediction.ensemble_disagreement
        )
    model_hashes = [model_state_sha256(model) for model in models]
    locked = subject.lock_seat_thresholds(
        torch, models, dataset, training_config=config
    )
    report = subject.report_diagnostic_holdout(
        torch,
        models,
        dataset,
        training_config=config,
        threshold_lock=locked,
    )
    assert report["diagnostic_only"] is True
    assert report["promotion_authorized"] is False
    assert [model_state_sha256(model) for model in models] == model_hashes


def test_immutable_manifest_requires_external_digest_and_source_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = tmp_path / "plan.json"
    plan = dataset_contract.write_dataset_plan(plan_path)
    merge_path = tmp_path / "merge.json"
    merge = {
        "schema": dataset_contract.MERGE_SCHEMA,
        "synthetic_test_marker": True,
    }
    merge_path.write_bytes(dataset_contract.canonical_bytes(merge))
    merge_sha = hashlib.sha256(merge_path.read_bytes()).hexdigest()
    directories = {}
    for shard in plan["shards"]:
        directory = tmp_path / "shards" / shard["shard_id"]
        directory.mkdir(parents=True)
        directories[shard["shard_id"]] = directory

    with pytest.raises(ValueError, match="merge file SHA-256 changed"):
        subject.verify_immutable_dataset_manifest(
            plan_path=plan_path,
            merge_manifest_path=merge_path,
            shard_directories=directories,
            expected_merge_file_sha256="0" * 64,
        )

    called = {}

    def replay(
        value: Any,
        *,
        plan: Any,
        shard_directories: Any,
    ) -> dict[str, Any]:
        called["directory_count"] = len(shard_directories)
        assert value == merge
        return {
            "pair_record_aggregate_sha256": "b" * 64,
            "shard_records": [
                {
                    "shard_id": row["shard_id"],
                    "done_sha256": "c" * 64,
                }
                for row in plan["shards"]
            ],
            "split_counts": {
                "train": 6000,
                "safety-fit": 1000,
                "threshold-lock": 1000,
                "diagnostic-holdout": 1000,
            },
            "paired_hand_count": 9000,
            "root_count": 18000,
        }

    monkeypatch.setattr(dataset_contract, "validate_merge_manifest", replay)
    verified = subject.verify_immutable_dataset_manifest(
        plan_path=plan_path,
        merge_manifest_path=merge_path,
        shard_directories=directories,
        expected_merge_file_sha256=merge_sha,
    )
    assert verified.receipt["source_replayed"] is True
    assert called["directory_count"] == len(plan["shards"])
    assert verified.receipt["teacher_values_are_realized_match_ev"] is False
