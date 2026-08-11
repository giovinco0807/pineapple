from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import ofc_regular.evaluate_hu_m43_attempt04_locked as evaluator
from ofc_regular.evaluate_hu_m43_attempt04_locked import (
    ATTEMPT04_PROFILES,
    evaluate_attempt04_fixed_threshold,
    evaluate_attempt04_locked_once,
    validate_attempt04_locked_receipt,
)
from ofc_regular.hu_m43_attempt04_runtime import (
    M43_ATTEMPT04_LOCKED_MARKER_SCHEMA,
    M43_ATTEMPT04_LOCKED_RECEIPT_SCHEMA,
    file_sha256,
    self_digest,
)


class _FakeV6:
    safety_threshold = 0.7
    safety_enabled = True

    def predict_heads_sample(self, sample, *, baseline_index):
        assert baseline_index == 0
        return SimpleNamespace(
            proposal_index=1,
            proposal_risk_eligible=True,
            action_score=np.asarray([0.0, 1.0]),
            base_delta=np.asarray([0.0, 1.0]),
            delta_disagreement=np.asarray([0.0, 0.1]),
            risk_eligible_mask=np.asarray([False, True]),
        )

    def select_action_index(self, sample, *, baseline_index):
        fired = bool(sample["fire"])
        return SimpleNamespace(
            proposal_index=1,
            baseline_index=0,
            selected_index=1 if fired else 0,
            safety_probability=0.8 if fired else 0.6,
            override_fired=fired,
        )


def _rows_and_samples(*, bad_profile_raw: bool = False):
    rows = []
    samples = []
    for profile_index, profile in enumerate(ATTEMPT04_PROFILES):
        for offset in range(40):
            positive_cutoff = 11 if bad_profile_raw and profile_index == 0 else 16
            positive = offset < positive_cutoff
            fire = offset < 10
            delta = 1.0 if positive else -10.0
            rows.append({"provenance": {"root_profile": profile}})
            samples.append(
                SimpleNamespace(
                    baseline_index=0,
                    seat="second",
                    policy_sample={
                        "actions": [{"row": 0}, {"row": 1}],
                        "fire": fire,
                    },
                    teacher_paired_delta_mean=np.asarray([0.0, delta]),
                    downside_loss_p95=np.asarray([0.0, 5.0]),
                    downside_loss_p99=np.asarray([0.0, 10.0]),
                    downside_loss_max=np.asarray([0.0, 20.0]),
                )
            )
    return rows, samples


def test_locked_gates_allow_negative_raw_mean_for_selective_policy():
    rows, samples = _rows_and_samples()
    report = evaluate_attempt04_fixed_threshold(_FakeV6(), rows, samples)
    assert report["status"] == "go_locked200"
    assert report["all_gates_passed"] is True
    assert report["raw_proposal_metrics"]["positive_rate"] == 0.4
    assert report["raw_proposal_metrics"]["mean_delta"] < 0.0
    assert report["raw_proposal_metrics"]["mean_delta_lcb"]["lower_bound"] < 0.0
    assert report["fixed_threshold_metrics"]["fires"] == 50
    assert report["fixed_threshold_metrics"]["mean_delta_lcb"]["lower_bound"] > 0.0
    assert not any("raw_mean_delta" in name for name in report["gates"])


def test_locked_gates_require_raw_positive_rate_for_every_profile():
    rows, samples = _rows_and_samples(bad_profile_raw=True)
    report = evaluate_attempt04_fixed_threshold(_FakeV6(), rows, samples)
    assert report["status"] == "no_go_locked200"
    assert report["gates"][
        "raw_proposal_positive_rate_each_profile_at_least_0_30"
    ] is False


def _write(path: Path, payload: dict | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def test_locked_one_shot_claims_marker_before_audit_and_retry_fails(
    tmp_path, monkeypatch
):
    root = tmp_path / "repo"
    paths = {
        name: root / f"{name}.bin"
        for name in (
            "model",
            "training",
            "threshold",
            "attempt04_plan",
            "population_plan",
        )
    }
    for name, path in paths.items():
        _write(path, {} if name in {"training", "threshold"} else name)
    implementation = root / "src/ofc_regular/hu_m43_joint_model_v6.py"
    _write(implementation, "v6 implementation")
    marker = root / "outputs/M43_ATTEMPT04_LOCKED200_CONSUMED.json"
    receipt_path = root / "outputs/locked-receipt.json"
    locked_path = root / "sealed/locked200.jsonl"
    freeze = {
        "freeze_sha256": "f" * 64,
        "model": {"file_sha256": file_sha256(paths["model"])},
        "final_training_manifest": {"file_sha256": file_sha256(paths["training"])},
        "threshold_lock": {"file_sha256": file_sha256(paths["threshold"])},
        "attempt04_plan": {"file_sha256": file_sha256(paths["attempt04_plan"])},
        "population_plan": {"file_sha256": file_sha256(paths["population_plan"])},
        "v6_implementation": {"file_sha256": file_sha256(implementation)},
        "locked200": {
            "identity_sha256": "a" * 64,
            "ordered_shards": [
                {
                    "path": str(locked_path.relative_to(root)),
                    "file_sha256": "b" * 64,
                    "records": 200,
                    "bytes": 100,
                }
            ],
            "global_consumption_marker": str(marker.relative_to(root)),
        },
    }
    freeze_path = root / "runtime-freeze.json"
    _write(freeze_path, freeze)
    monkeypatch.setattr(evaluator, "validate_attempt04_runtime_freeze", lambda *a, **k: None)
    monkeypatch.setattr(
        evaluator,
        "validate_attempt04_runtime_artifact_files",
        lambda *_a, **_k: {
            "model": file_sha256(paths["model"]),
            "training": file_sha256(paths["training"]),
            "threshold": file_sha256(paths["threshold"]),
            "attempt04_plan": file_sha256(paths["attempt04_plan"]),
            "population_plan": file_sha256(paths["population_plan"]),
            "implementation": file_sha256(implementation),
        },
    )
    monkeypatch.setattr(evaluator, "load_bound_attempt04_v6_model", lambda *a, **k: _FakeV6())

    audited = {"called": 0}

    def _audit(shard_paths, *, shard_specs, expected_identity_sha256):
        assert marker.is_file(), "marker must exist before locked audit"
        marker_payload = json.loads(marker.read_text())
        assert marker_payload["status"] == "claimed_before_locked200_content_read"
        audited["called"] += 1
        return [], [], {
            "records": 200,
            "identity_sha256": expected_identity_sha256,
            "ordered_shards": [],
        }

    monkeypatch.setattr(evaluator, "_audit_and_read_locked200", _audit)
    diagnostic = {
        "schema": "hu_m43_attempt04_locked200_diagnostic_v1",
        "status": "go_locked200",
        "gates": {"test": True},
    }
    monkeypatch.setattr(evaluator, "evaluate_attempt04_fixed_threshold", lambda *a: diagnostic)
    result = evaluate_attempt04_locked_once(
        model_path=paths["model"],
        final_training_manifest_path=paths["training"],
        threshold_lock_path=paths["threshold"],
        runtime_freeze_path=freeze_path,
        attempt04_plan_path=paths["attempt04_plan"],
        population_plan_path=paths["population_plan"],
        repo_root=root,
        receipt_path=receipt_path,
    )
    assert audited["called"] == 1
    assert result["population_launch_allowed"] is True
    assert result["status"] == "go_locked200_population_launch_eligible"
    assert receipt_path.is_file()
    with pytest.raises(FileExistsError, match="receipt already exists"):
        evaluate_attempt04_locked_once(
            model_path=paths["model"],
            final_training_manifest_path=paths["training"],
            threshold_lock_path=paths["threshold"],
            runtime_freeze_path=freeze_path,
            attempt04_plan_path=paths["attempt04_plan"],
            population_plan_path=paths["population_plan"],
            repo_root=root,
            receipt_path=receipt_path,
        )


def test_locked_receipt_rejects_gate_status_tamper():
    marker = {
        "schema": M43_ATTEMPT04_LOCKED_MARKER_SCHEMA,
        "model_sha256": "a" * 64,
        "locked200_identity_sha256": "b" * 64,
        "marker_sha256": "c" * 64,
    }
    freeze = {"freeze_sha256": "d" * 64}
    receipt = {
        "schema": M43_ATTEMPT04_LOCKED_RECEIPT_SCHEMA,
        "status": "no_go_locked200",
        "promotion_status": "no_go_locked200",
        "population_launch_allowed": False,
        "requires_fresh_population_acceptance": False,
        "runtime_freeze_sha256": "d" * 64,
        "model_sha256": "a" * 64,
        "consumption_marker_canonical_sha256": "c" * 64,
        "locked200_identity_sha256": "b" * 64,
        "locked200_records": 200,
        "evaluation_pass_count": 1,
        "threshold_search_performed": False,
        "threshold_reselection_performed": False,
        "model_selection_performed": False,
        "feature_selection_performed": False,
        "algorithm_or_gate_change_after_result_allowed": False,
        "current_profile_resolved": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "policy_promoted": False,
        "locked_diagnostic": {
            "status": "no_go_locked200",
            "gates": {"failed": False},
        },
    }
    receipt["receipt_sha256"] = self_digest(receipt, "receipt_sha256")
    validate_attempt04_locked_receipt(receipt, runtime_freeze=freeze, marker=marker)
    changed = deepcopy(receipt)
    changed["population_launch_allowed"] = True
    changed["receipt_sha256"] = self_digest(changed, "receipt_sha256")
    with pytest.raises(ValueError, match="Go does not match"):
        validate_attempt04_locked_receipt(changed, runtime_freeze=freeze, marker=marker)
