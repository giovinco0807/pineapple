from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import hu_m31_t3_locked_promotion_artifact_bridge_v1 as subject
from ofc_regular import hu_m31_t3_step6d_locked_promotion_v1 as promotion
from ofc_regular import hu_m31_t3_street_policy_training_v1 as training


def _training_fixture() -> Any:
    path = Path(__file__).with_name(
        "test_hu_m31_t3_street_policy_training_v1.py"
    )
    name = "_m31_training_bridge_fixture"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("training fixture cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_real_training_bundle_and_rich_lock_feed_promotion_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    torch = pytest.importorskip("torch")
    fixture = _training_fixture()
    dataset = fixture._dataset()
    config = fixture._config()
    models = training.create_deterministic_ensemble(
        torch,
        training_config=config,
        model_config=fixture._model_config(),
    )
    bundle = tmp_path / "risk-bundle"
    bundle_manifest = training.write_ensemble_checkpoint_bundle(
        bundle,
        models,
        dataset=dataset,
        training_config=config,
        stage="risk",
        completed_epoch=config.risk_epochs,
    )
    rich_lock = training.lock_seat_thresholds(
        torch, models, dataset, training_config=config
    )
    rich_lock_path = tmp_path / "rich-threshold-lock.json"
    rich_lock_path.write_bytes(training._canonical_bytes(rich_lock))

    compact_path = tmp_path / "promotion-threshold-lock.json"
    assert subject.main(
        [
            "--checkpoint-bundle-directory",
            str(bundle),
            "--training-threshold-lock",
            str(rich_lock_path),
            "--model-artifact-id",
            "street-policy-net-v1-test",
            "--output-threshold-lock",
            str(compact_path),
        ]
    ) == 0
    bridge = json.loads(capsys.readouterr().out)
    assert bridge["model_path"] == str((bundle / "manifest.json").resolve())
    assert (
        bridge["source_checkpoint_bundle_identity_sha256"]
        == bundle_manifest["bundle_identity_sha256"]
    )
    compact = subject._read_canonical(
        compact_path, "compact threshold lock"
    )[0]
    assert compact["seat_enabled"] == {
        seat: rich_lock["seat_thresholds"][seat]["enabled"]
        for seat in promotion.SEATS
    }
    assert compact["seat_thresholds"] == {
        seat: rich_lock["seat_thresholds"][seat][
            "safe_probability_threshold"
        ]
        for seat in promotion.SEATS
    }

    registry = tmp_path / "policy-registry.json"
    registry.write_bytes(b"immutable explicit policy registry")
    runtime = tmp_path / "runtime-closure.py"
    runtime.write_bytes(b"immutable runtime closure")
    plan = promotion.build_locked_promotion_plan(
        plan_id="m31-t3-training-bridge-test",
        model_artifact_id=bridge["model_artifact_id"],
        model_path=bridge["model_path"],
        expected_model_sha256=bridge["model_sha256"],
        threshold_lock_path=bridge["threshold_lock_path"],
        expected_threshold_lock_sha256=bridge["threshold_lock_sha256"],
        policy_registry_path=registry,
        expected_policy_registry_sha256=promotion.sha256_file(registry),
        evaluation_runtime_closure_path=runtime,
        expected_evaluation_runtime_closure_sha256=promotion.sha256_file(
            runtime
        ),
    )
    assert plan["artifact_binding"]["threshold_lock_content"] == compact
    assert plan["artifact_binding"]["model"]["filename"] == "manifest.json"
    assert plan["named_profile_added"] is False
    assert plan["current_profile_changed"] is False


def test_bridge_rejects_rich_lock_model_hash_drift(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    fixture = _training_fixture()
    dataset = fixture._dataset()
    config = fixture._config()
    models = training.create_deterministic_ensemble(
        torch,
        training_config=config,
        model_config=fixture._model_config(),
    )
    bundle = tmp_path / "risk-bundle"
    training.write_ensemble_checkpoint_bundle(
        bundle,
        models,
        dataset=dataset,
        training_config=config,
        stage="risk",
        completed_epoch=config.risk_epochs,
    )
    rich_lock = training.lock_seat_thresholds(
        torch, models, dataset, training_config=config
    )
    rich_lock["model_state_sha256"][0] = "0" * 64
    identity = dict(rich_lock)
    identity.pop("threshold_lock_sha256")
    rich_lock["threshold_lock_sha256"] = training._canonical_sha256(identity)
    path = tmp_path / "tampered-rich-lock.json"
    path.write_bytes(training._canonical_bytes(rich_lock))
    with pytest.raises(ValueError, match="threshold-lock|model binding"):
        subject.build_locked_promotion_artifacts(
            checkpoint_bundle_directory=bundle,
            training_threshold_lock_path=path,
            model_artifact_id="street-policy-net-v1-test",
        )
