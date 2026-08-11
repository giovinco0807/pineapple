import hashlib
import json
from pathlib import Path

import pytest

from ofc_regular import hu_m31_t3_locked_promotion_production_v1 as production
from ofc_regular import hu_m31_t3_promotion_runtime_closure_v1 as subject
from ofc_regular import hu_m31_t3_step6d_locked_promotion_v1 as promotion
from ofc_regular import hu_m31_t3_street_policy_training_v1 as training
from ofc_regular.action_key import ACTION_KEY_SCHEMA
from ofc_regular.street_policy_net_v1 import FEATURE_SCHEMA_HASH


def _write(path: Path, raw: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def _fixture_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, Path]:
    repository = tmp_path / "repository"
    for index, relative in enumerate(sorted(subject._REQUIRED_SOURCE_PATHS)):
        _write(
            repository / Path(relative),
            f"# fixture source {index}: {relative}\n".encode("ascii"),
        )
    monkeypatch.setattr(
        subject,
        "FROZEN_POLICY_REGISTRY_SHA256",
        subject.sha256_file(
            repository / "src/ofc_regular/ai_profiles.py"
        ),
    )
    for field, relative in subject.legacy_model_bindings().items():
        model_raw = f"fixture model {field}".encode("ascii")
        _write(
            repository / Path(relative),
            model_raw,
        )
        monkeypatch.setitem(
            subject.FROZEN_LEGACY_MODEL_SHA256_BY_FIELD,
            field,
            hashlib.sha256(model_raw).hexdigest(),
        )

    config = training.StreetPolicyTrainingConfig(
        ensemble_size=1,
        batch_size=1,
        core_epochs=1,
        risk_epochs=1,
        minimum_lock_fires_per_seat=1,
    )
    model_raw = b"tiny street policy checkpoint"
    identity = {
        "schema": training.CHECKPOINT_BUNDLE_SCHEMA,
        "training_view_identity_sha256": "1" * 64,
        "training_config": config.to_dict(),
        "training_config_sha256": config.identity_sha256,
        "stage": "risk",
        "completed_epoch": 1,
        "ensemble_size": 1,
        "feature_schema_hash": FEATURE_SCHEMA_HASH,
        "loss_schema_hash": subject.LOSS_SCHEMA_HASH,
        "epoch_optimizer": "fresh_adamw_per_epoch",
        "models": [
            {
                "model_index": 0,
                "path": "model_00.zip",
                "bytes": len(model_raw),
                "sha256": hashlib.sha256(model_raw).hexdigest(),
                "model_state_sha256": "2" * 64,
                "checkpoint_identity_sha256": "3" * 64,
            }
        ],
        "teacher_values_are_realized_match_ev": False,
        "current_profile_changed": False,
    }
    checkpoint_manifest = dict(identity)
    checkpoint_manifest["bundle_identity_sha256"] = (
        subject.canonical_sha256(identity)
    )
    bundle = tmp_path / "candidate_bundle"
    _write(bundle / "manifest.json", subject.canonical_bytes(checkpoint_manifest))
    _write(bundle / "model_00.zip", model_raw)
    threshold = _write(
        tmp_path / "training_threshold_lock.json",
        subject.canonical_bytes({}),
    )
    monkeypatch.setattr(
        subject.training,
        "_validate_threshold_lock",
        lambda value, **kwargs: dict(value),
    )
    t4_raw = b"tiny exact t4 library"
    t4 = _write(
        tmp_path / "native" / "release" / "hu_m3_t4_exact.dll",
        t4_raw,
    )
    monkeypatch.setitem(
        subject.ACCEPTED_EXACT_T4_LIBRARY_SHA256_BY_TARGET,
        "windows",
        hashlib.sha256(t4_raw).hexdigest(),
    )
    return repository, bundle, threshold, t4


def _create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str = "closure.zip"
) -> tuple[Path, dict]:
    repository, bundle, threshold, t4 = _fixture_tree(
        tmp_path, monkeypatch
    )
    package = tmp_path / name
    subject.create_runtime_closure_package(
        repository_root=repository,
        candidate_checkpoint_bundle=bundle,
        training_threshold_lock_path=threshold,
        exact_t4_native_library_path=t4,
        output_path=package,
    )
    manifest = subject.validate_runtime_closure_package(
        package,
        expected_sha256=subject.sha256_file(package),
        source_replay_root=repository,
    )
    return package, manifest


def test_closure_is_deterministic_semantic_and_source_replayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_a, manifest = _create(tmp_path / "a", monkeypatch, "a.zip")
    package_b, _ = _create(tmp_path / "b", monkeypatch, "b.zip")

    assert package_a.read_bytes() == package_b.read_bytes()
    assert manifest["factory_contract"]["current_profile_allowed"] is False
    assert [
        row["profile_id"]
        for row in manifest["factory_contract"]["population"]
    ] == list(subject.POPULATION_PROFILE_IDS)
    assert manifest["factory_contract"]["abr"] == [
        {
            "response_id": response_id,
            "factory_id": factory_id,
            "artifact_binding": "external_post_plan_manifest_and_checkpoint",
        }
        for response_id, factory_id in subject.ABR_FACTORY_IDS.items()
    ]
    assert manifest["schema_bindings"]["action_key_schema"] == ACTION_KEY_SCHEMA


def test_closure_rejects_package_hash_and_live_model_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package, manifest = _create(tmp_path, monkeypatch)
    with pytest.raises(subject.PromotionRuntimeClosureError):
        subject.validate_runtime_closure_package(
            package,
            expected_sha256="0" * 64,
            source_replay_root=tmp_path / "repository",
        )

    model_path = (
        tmp_path
        / "repository"
        / Path(manifest["legacy_model_bindings"]["turn3"])
    )
    model_path.write_bytes(b"drift")
    with pytest.raises(
        subject.PromotionRuntimeClosureError,
        match="live legacy model differs",
    ):
        subject.validate_runtime_closure_package(
            package,
            expected_sha256=subject.sha256_file(package),
            source_replay_root=tmp_path / "repository",
        )


def test_extract_is_create_only_and_byte_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package, manifest = _create(tmp_path, monkeypatch)
    destination = tmp_path / "extracted"
    receipt = subject.extract_runtime_closure_package(
        package,
        expected_sha256=subject.sha256_file(package),
        output_directory=destination,
        source_replay_root=tmp_path / "repository",
    )
    assert receipt["runtime_activated"] is False
    registry = destination / "src/ofc_regular/ai_profiles.py"
    assert subject.sha256_file(registry) == manifest["entries"][
        "src/ofc_regular/ai_profiles.py"
    ]["sha256"]
    with pytest.raises(FileExistsError):
        subject.extract_runtime_closure_package(
            package,
            expected_sha256=subject.sha256_file(package),
            output_directory=destination,
            source_replay_root=tmp_path / "repository",
        )


def test_production_prepare_stays_dormant_without_real_abrs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package, manifest = _create(tmp_path, monkeypatch)
    repository = tmp_path / "repository"
    model = tmp_path / "candidate_bundle" / "manifest.json"
    model_sha = promotion.sha256_file(model)
    compatibility = {
        "schema": promotion.THRESHOLD_LOCK_SCHEMA,
        "status": "locked_no_holdout_reselection",
        "model_artifact_id": "m31-fixture-model",
        "model_sha256": model_sha,
        "state_action_input_schema_sha256": FEATURE_SCHEMA_HASH,
        "action_key_schema": ACTION_KEY_SCHEMA,
        "seat_thresholds": {"first": 0.5, "second": 0.5},
        "seat_enabled": {"first": True, "second": True},
        "source_training_threshold_lock_sha256": "4" * 64,
        "source_checkpoint_bundle_identity_sha256": manifest[
            "candidate_artifacts"
        ]["checkpoint_bundle_identity_sha256"],
        "locked_on_threshold_holdout_only": True,
        "model_change_requires_new_lock": True,
        "teacher_ev_lcb_is_runtime_gate": False,
        "top1_accuracy_is_promotion_gate": False,
        "current_profile_changed": False,
        "runtime_activated": False,
    }
    compatibility_path = _write(
        tmp_path / "compatibility_threshold.json",
        promotion.canonical_bytes(compatibility),
    )
    registry = repository / "src/ofc_regular/ai_profiles.py"
    plan = promotion.build_locked_promotion_plan(
        plan_id="m31-fixture-plan",
        model_artifact_id="m31-fixture-model",
        model_path=model,
        expected_model_sha256=model_sha,
        threshold_lock_path=compatibility_path,
        expected_threshold_lock_sha256=promotion.sha256_file(
            compatibility_path
        ),
        policy_registry_path=registry,
        expected_policy_registry_sha256=promotion.sha256_file(registry),
        evaluation_runtime_closure_path=package,
        expected_evaluation_runtime_closure_sha256=(
            promotion.sha256_file(package)
        ),
    )
    plan_path = _write(
        tmp_path / "plan.json", promotion.canonical_bytes(plan)
    )
    prepared = production.prepare_locked_execution_plan(
        plan_path=plan_path,
        closure_package_path=package,
        expected_closure_package_sha256=promotion.sha256_file(package),
        extraction_root=tmp_path / "execution",
        source_replay_root=repository,
        compatibility_threshold_lock_path=compatibility_path,
        policy_registry_path=registry,
        require_host_target=False,
    )
    receipt = production.prepare_receipt(prepared)
    assert receipt["status"] == "dormant_waiting_for_frozen_abr_bindings"
    assert receipt["locked_execution_ready"] is False
    assert receipt["missing_abr_bindings"] == list(subject.ABR_FACTORY_IDS)
    assert receipt["current_profile_changed"] is False
