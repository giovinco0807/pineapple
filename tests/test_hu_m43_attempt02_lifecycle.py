from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor

from ofc_regular import evaluate_hu_m43_attempt02_locked_holdout as evaluator
from ofc_regular import freeze_hu_m43_attempt02_model as freezer
from ofc_regular.hu_m43_attempt02_contract import (
    M43_ATTEMPT02_CALIBRATION_PARTITION_SCHEMA,
    M43_ATTEMPT02_DATA_CONTRACT_SCHEMA,
    M43_ATTEMPT02_TEACHER_SHARDS_SCHEMA,
)
from ofc_regular.hu_m43_attempt02_lifecycle import (
    M43_ATTEMPT02_LOCKED_RECEIPT_SCHEMA,
    M43_ATTEMPT02_MARKER_SCHEMA,
    build_calibration_role_binding,
    build_fresh_bindings,
    expected_training_data_binding,
    file_sha256,
    self_digest,
    validate_attempt02_freeze,
    validate_attempt02_locked_receipt,
    write_immutable_json,
)
from ofc_regular.hu_m43_joint_model_loader import load_hu_m43_joint_action_model
from ofc_regular.hu_m43_joint_model_v4 import (
    HU_M43_V4_CROSSFIT_SCHEMA,
    HU_M43_V4_MODEL_SCHEMA,
    HU_M43_V4_PRECAL_GATE_SCHEMA,
    HU_M43_V4_SAFETY_FIT_SCHEMA,
    HU_M43_V4_THRESHOLD_SCHEMA,
    HU_M43_V4_TRAINING_MANIFEST_SCHEMA,
    HuM43JointModelV4,
)
from ofc_regular.hu_m43_pilot_contract import (
    _identity_digest,
    _ordered_shard_binding,
    canonical_manifest_sha256,
)
from ofc_regular.hu_m4_joint_model import (
    HuM4JointActionModel,
    PairedDeltaRiskFoldEstimator,
)
from ofc_regular.hu_m4_t1_policy import HuM4T1SelectiveOverridePolicy
from ofc_regular.hu_turn3_model import HU_FEATURE_DIM
from ofc_regular.validate_hu_m43_attempt02_acceptance import (
    ATTEMPT02_POPULATION_PREFLIGHT_SCHEMA,
    build_attempt02_population_launch_preflight,
)


_PROFILES = (
    "stage19_p0",
    "stage9f_p2",
    "stage7_m5_r10",
    "stage3_baseline",
    "random_exact_final",
)


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(split: str, seed: int, profile: str) -> dict:
    return {
        "split": split,
        "hand_seed": seed,
        "observation_fingerprint": hashlib.sha256(
            f"{split}\0{seed}".encode()
        ).hexdigest(),
        "provenance": {"root_profile": profile, "current_profile_resolved": False},
    }


def _v4_model() -> HuM43JointModelV4:
    dimension = 4 * HU_FEATURE_DIM
    x = np.zeros((2, dimension), dtype=np.float32)

    def reg(value: float):
        return DummyRegressor(strategy="constant", constant=value).fit(
            x, np.full(2, value)
        )

    folds = tuple(
        PairedDeltaRiskFoldEstimator(
            delta_estimator=reg(float(index)),
            positive_gain_estimator=reg(0.7),
            downside_p95_estimator=reg(5.0),
            downside_p99_estimator=reg(10.0),
            downside_max_estimator=reg(20.0),
            paired_feature_dim=dimension,
            fold_index=index,
        )
        for index in range(5)
    )
    safety_x = np.zeros((2, 15), dtype=np.float32)
    safety = DummyClassifier(strategy="constant", constant=1).fit(
        safety_x, np.ones(2, dtype=np.int8)
    )
    return HuM43JointModelV4(
        paired_fold_estimators=folds,
        safety_estimator=safety,
        safety_threshold=0.5,
        safety_enabled=True,
        model_id="m43-attempt02-v4-test",
        manifest={"current_profile_mutated": False, "runtime_enabled": False},
    )


class _Baseline:
    def choose_action_observation(self, observation, **kwargs):  # pragma: no cover
        raise AssertionError("loader-only fixture should not choose an action")


@pytest.fixture
def lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    root = tmp_path.resolve()
    plan_path = root / "attempt02-plan.json"
    plan_path.write_text('{"synthetic":true}\n', encoding="utf-8")
    population_plan = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "hu_joint_policy_m43_population.json"
        ).read_text(encoding="utf-8")
    )
    population_plan["attempt02_training_plan"]["file_sha256"] = file_sha256(
        plan_path
    )
    population_plan_path = root / "population-plan.json"
    population_plan_path.write_text(
        json.dumps(population_plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    marker_relative = "global-lock/test-identity/M43_LOCKED_CONSUMED.json"
    plan = {
        "budget": {"roots_per_shard": 5},
        "fresh_splits": {"train": {"roots": 10}, "calibration": {"roots": 20}},
        "calibration_partition": {
            "schema": M43_ATTEMPT02_CALIBRATION_PARTITION_SCHEMA,
            "identity_hash_salt": "attempt02-lifecycle-test",
            "roots_per_profile_per_role": 2,
            "safety_fit_roots": 10,
            "threshold_lock_roots": 10,
            "minimum_threshold_lock_fires_for_positive_pilot_signal": 10,
        },
        "inherited_locked": {
            "global_consumption_marker": marker_relative,
        },
    }

    train_rows = [
        _row("train", 10_000 + index, _PROFILES[index % 5])
        for index in range(10)
    ]
    calibration_rows = [
        _row("calibration", 20_000 + 4 * profile_index + offset, profile)
        for profile_index, profile in enumerate(_PROFILES)
        for offset in range(4)
    ]
    train_paths = []
    calibration_paths = []
    for split, rows, paths in (
        ("train", train_rows, train_paths),
        ("calibration", calibration_rows, calibration_paths),
    ):
        for index in range(0, len(rows), 5):
            path = root / "fresh" / f"{split}-{index // 5:02d}.jsonl"
            _jsonl(path, rows[index : index + 5])
            paths.append(path)

    locked_path = root / "sealed" / "locked_holdout.jsonl"
    locked_rows = [
        _row("locked_holdout", 30_000 + index, _PROFILES[index % 5])
        for index in range(2)
    ]
    _jsonl(locked_path, locked_rows)
    locked_binding = _ordered_shard_binding(
        (locked_path,), repo_root=root, expected_split="locked_holdout"
    )
    locked_file = locked_binding["ordered_shards"][0]
    inherited = {
        "classification": "inherited_unopened",
        "path": locked_file["path"],
        "records": locked_binding["records"],
        "bytes": locked_file["bytes"],
        "file_sha256": locked_file["file_sha256"],
        "canonical_rows_sha256": locked_binding["canonical_rows_sha256"],
        "identity_sha256": locked_file["identity_sha256"],
        "ordered_shards_sha256": locked_binding["ordered_shards_sha256"],
        "content_parse_count": 0,
        "model_evaluation_count": 0,
    }
    plan["inherited_locked"].update(
        {key: value for key, value in inherited.items() if key not in {
            "content_parse_count", "model_evaluation_count"
        }}
    )

    fresh_bindings = build_fresh_bindings(
        repo_root=root, train=train_paths, calibration=calibration_paths
    )
    roles = build_calibration_role_binding(
        plan=plan, calibration=calibration_paths
    )
    teacher_shards = {
        "schema": M43_ATTEMPT02_TEACHER_SHARDS_SCHEMA,
        "splits": fresh_bindings,
    }
    teacher_shards["all_fresh_splits_sha256"] = canonical_manifest_sha256(
        {"schema": teacher_shards["schema"], "splits": fresh_bindings}
    )

    def identities(rows: list[dict]) -> str:
        return _identity_digest(
            (int(row["hand_seed"]), row["observation_fingerprint"])
            for row in rows
        )

    contract = {
        "schema": M43_ATTEMPT02_DATA_CONTRACT_SCHEMA,
        "status": "pass_fresh_train_calibration_sealed_inherited_locked_unopened",
        "plan_sha256": file_sha256(plan_path),
        "fresh_splits": {
            "train": {
                "records": 10,
                "shards": 2,
                "identity_sha256": identities(train_rows),
                "profile_counts": {profile: 2 for profile in _PROFILES},
            },
            "calibration": {
                "records": 20,
                "shards": 4,
                "identity_sha256": identities(calibration_rows),
                "profile_counts": {profile: 4 for profile in _PROFILES},
            },
        },
        "calibration_partition": roles,
        "teacher_shards": teacher_shards,
        "teacher_shards_all_fresh_splits_sha256": teacher_shards[
            "all_fresh_splits_sha256"
        ],
        "inherited_locked": inherited,
        "global_locked_consumption": {
            "identity_sha256": inherited["identity_sha256"],
            "canonical_marker_path": marker_relative,
            "matching_marker_count": 0,
            "status": "unconsumed_preflight",
            "claim_before_content_open_required": True,
            "claim_is_consuming_even_on_crash": True,
        },
        "freshness": {
            "train_calibration_identity_overlap": 0,
            "exclusion_hand_seed_overlap": 0,
            "exclusion_observation_fingerprint_overlap": 0,
            "inherited_locked_hand_seed_overlap": 0,
            "inherited_locked_observation_fingerprint_overlap": 0,
        },
        "runtime": {
            "current_profile_resolved": False,
            "current_profile_changed": False,
            "policy_activated": False,
            "full_replacement": False,
            "large_scale_authorized": False,
        },
    }
    contract["contract_sha256"] = self_digest(contract, "contract_sha256")
    contract_path = root / "data_contract.json"
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    model = replace(
        _v4_model(),
        manifest={
            "schema": "hu_m43_attempt02_v4_model_binding_v1",
            "data_contract_sha256": contract["contract_sha256"],
            "calibration_binding_schema": (
                "hu_m43_attempt02_v4_calibration_binding_v1"
            ),
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
            "full_replacement": False,
            "runtime_teacher_inputs": False,
        },
    )
    model_path = root / "model-v4.pkl"
    model_sha = model.save(model_path)
    binding = expected_training_data_binding(contract)
    manifest = {
        "schema": HU_M43_V4_TRAINING_MANIFEST_SCHEMA,
        "model_schema": HU_M43_V4_MODEL_SCHEMA,
        "model_id": model.model_id,
        "model_sha256": model_sha,
        "promotion_status": "candidate_ready_for_freeze",
        "baseline_training_rows_included": False,
        "baseline_runtime_score_exact_zero": True,
        "all_action_state_balanced_oof_safety": True,
        "runtime_teacher_inputs": False,
        "attempt02_data_contract": {
            "schema": M43_ATTEMPT02_DATA_CONTRACT_SCHEMA,
            "file_sha256": file_sha256(contract_path),
            "contract_sha256": contract["contract_sha256"],
        },
        "calibration_binding": {
            "schema": "hu_m43_attempt02_v4_calibration_binding_v1",
            "data_contract_file_sha256": file_sha256(contract_path),
            "data_contract_sha256": contract["contract_sha256"],
            "teacher_shards_all_fresh_splits_sha256": binding[
                "teacher_shards_all_fresh_splits_sha256"
            ],
            "roles": {
                role: {
                    **binding["calibration_roles"][role],
                    "profile_counts": roles[role]["profile_counts"],
                }
                for role in ("safety_fit", "threshold_lock")
            },
            "train_calibration_identity_overlap": 0,
            "inherited_holdout_input_accepted": False,
            "inherited_holdout_content_opened": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        },
        "crossfit": {
            "schema": HU_M43_V4_CROSSFIT_SCHEMA,
            "status": "pass",
            "states": 10,
            "folds": 5,
            "exact_outer_inner_job_grid": True,
            "calibration_opened": False,
            "precalibration_gate": {
                "schema": HU_M43_V4_PRECAL_GATE_SCHEMA,
                "status": "go",
                "calibration_opened": False,
                "runtime_teacher_inputs": False,
                "gates": {"all_required_checks": True},
            },
        },
        "safety_fit": {
            "schema": HU_M43_V4_SAFETY_FIT_SCHEMA,
            "status": "fit_threshold_unselected",
            "threshold_selected": False,
            "safety_enabled": False,
            "runtime_teacher_inputs": False,
            "sources": {
                "train_oof": {"states": 10},
                "calibration.safety_fit": {"states": 10},
                "calibration.threshold_lock": {"used": False, "labels_opened": False},
                "locked_holdout": {"used": False, "labels_opened": False},
            },
        },
        "threshold_selection": {
            "schema": HU_M43_V4_THRESHOLD_SCHEMA,
            "status": "go",
            "source": "fresh_calibration.threshold_lock_only",
            "states": 10,
            "proposal_rows": 10,
            "selected_threshold": 0.5,
            "selected_metrics": {
                "threshold": 0.5,
                "fires": 10,
                "false_positive_rate": 0.0,
                "p95_loss": 5.0,
                "p99_loss": 10.0,
                "max_loss": 20.0,
                "teacher_mean_delta_per_fire": 1.0,
            },
            "constraints": {
                "minimum_fires": 10,
                "maximum_false_positive_rate": 0.30,
                "maximum_p95_loss": 25.0,
                "maximum_p99_loss": 40.0,
                "maximum_max_loss": 50.0,
            },
            "identity_overlap_with_fit": {
                "seed_values": 0,
                "observation_fingerprints": 0,
            },
            "threshold_adaptation_after_selection": False,
            "safety_enabled": True,
            "locked_holdout": {"used": False, "labels_opened": False},
            "runtime_teacher_inputs": False,
        },
        "locked_holdout": {
            "status": "not_evaluated_pre_freeze",
            "labels_opened": False,
        },
        "model_artifact": {
            "sha256": model_sha,
            "model_id": model.model_id,
            "safety_enabled": True,
            "safety_threshold": 0.5,
        },
        "calibration_opened_after_precalibration_go": True,
        "inherited_holdout_input_accepted": False,
        "inherited_holdout_content_opened": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "full_replacement": False,
        "runtime": {
            "current_profile_mutated": False,
            "policy_activated": False,
            "full_replacement": False,
        },
    }
    manifest["manifest_sha256"] = self_digest(manifest, "manifest_sha256")
    manifest_path = root / "training_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    monkeypatch.setattr(freezer, "load_and_validate_attempt02_plan", lambda _: plan)
    freeze = freezer.build_attempt02_model_threshold_freeze(
        model_path=model_path,
        training_manifest_path=manifest_path,
        data_contract_path=contract_path,
        plan_path=plan_path,
        population_plan_path=population_plan_path,
        repo_root=root,
        train=train_paths,
        calibration=calibration_paths,
    )
    freeze_path = root / "freeze.json"
    write_immutable_json(freeze_path, freeze)
    return {
        "root": root,
        "plan": plan,
        "plan_path": plan_path,
        "population_plan_path": population_plan_path,
        "contract": contract,
        "contract_path": contract_path,
        "model_path": model_path,
        "model_sha": model_sha,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "train": train_paths,
        "calibration": calibration_paths,
        "locked": locked_path,
        "marker": root / marker_relative,
        "freeze": freeze,
        "freeze_path": freeze_path,
    }


def test_attempt02_freeze_and_schema_dispatch_v4_are_bound(lifecycle) -> None:
    freeze = lifecycle["freeze"]
    validate_attempt02_freeze(freeze, contract=lifecycle["contract"])
    assert freeze["precalibration_status"] == "go"
    assert freeze["threshold_lock_status"] == "go"
    assert freeze["locked_content_access_count_at_freeze"] == 0
    assert freeze["population_plan_file_sha256"] == file_sha256(
        lifecycle["population_plan_path"]
    )
    assert freeze["population_plan_resolved_path"] == str(
        lifecycle["population_plan_path"].resolve()
    )
    assert freeze["activation_guards"] == {
        "current_profile_changed": False,
        "current_profile_resolved": False,
        "runtime_policy_activated": False,
        "full_replacement_enabled": False,
    }
    loaded = load_hu_m43_joint_action_model(
        lifecycle["model_path"],
        expected_sha256=lifecycle["model_sha"],
        freeze_manifest=lifecycle["freeze_path"],
        training_manifest_path=lifecycle["manifest_path"],
    )
    assert isinstance(loaded, HuM43JointModelV4)

    policy = HuM4T1SelectiveOverridePolicy.from_model_paths(
        _Baseline(),
        action_value_model_path=lifecycle["model_path"],
        safety_model_path=lifecycle["model_path"],
        safety_probability_threshold=0.5,
        expected_joint_artifact_sha256=lifecycle["model_sha"],
        freeze_manifest_path=lifecycle["freeze_path"],
        training_manifest_path=lifecycle["manifest_path"],
    )
    assert policy.action_value_model is policy.safety_model
    assert isinstance(policy.action_value_model, HuM43JointModelV4)
    assert policy.runtime_binding_verified is True
    assert policy.model_load_failures == ()


def test_attempt02_v4_binding_is_all_or_nothing_and_tamper_fails_closed(
    lifecycle, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="requires expected SHA"):
        load_hu_m43_joint_action_model(
            lifecycle["model_path"],
            expected_sha256=lifecycle["model_sha"],
        )
    changed = tmp_path / "changed-training.json"
    changed.write_text('{}\n', encoding="utf-8")
    policy = HuM4T1SelectiveOverridePolicy.from_model_paths(
        _Baseline(),
        action_value_model_path=lifecycle["model_path"],
        safety_model_path=lifecycle["model_path"],
        safety_probability_threshold=0.5,
        expected_joint_artifact_sha256=lifecycle["model_sha"],
        freeze_manifest_path=lifecycle["freeze_path"],
        training_manifest_path=changed,
    )
    assert policy.action_value_model is None
    assert policy.safety_model is None
    assert policy.runtime_binding_verified is False
    assert policy.model_load_failures == (
        "frozen_joint_artifact_binding_failed:ValueError",
    )


def test_freeze_rejects_threshold_tail_and_shard_tamper(
    lifecycle, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        freezer, "load_and_validate_attempt02_plan", lambda _: lifecycle["plan"]
    )
    unsafe = copy.deepcopy(lifecycle["manifest"])
    unsafe["threshold_selection"]["selected_metrics"]["p99_loss"] = 41.0
    unsafe["manifest_sha256"] = self_digest(unsafe, "manifest_sha256")
    lifecycle["manifest_path"].write_text(
        json.dumps(unsafe, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="p99_loss exceeds"):
        freezer.build_attempt02_model_threshold_freeze(
            model_path=lifecycle["model_path"],
            training_manifest_path=lifecycle["manifest_path"],
            data_contract_path=lifecycle["contract_path"],
            plan_path=lifecycle["plan_path"],
            population_plan_path=lifecycle["population_plan_path"],
            repo_root=lifecycle["root"],
            train=lifecycle["train"],
            calibration=lifecycle["calibration"],
        )

    lifecycle["manifest_path"].write_text(
        json.dumps(lifecycle["manifest"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with lifecycle["train"][0].open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="train shard binding mismatch"):
        freezer.build_attempt02_model_threshold_freeze(
            model_path=lifecycle["model_path"],
            training_manifest_path=lifecycle["manifest_path"],
            data_contract_path=lifecycle["contract_path"],
            plan_path=lifecycle["plan_path"],
            population_plan_path=lifecycle["population_plan_path"],
            repo_root=lifecycle["root"],
            train=lifecycle["train"],
            calibration=lifecycle["calibration"],
        )


def test_one_shot_claims_canonical_marker_before_locked_read_and_rejects_duplicate(
    lifecycle, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        evaluator, "load_and_validate_attempt02_plan", lambda _: lifecycle["plan"]
    )
    original = evaluator._audit_and_read_locked_rows

    def guarded(path, *, repo_root, expected):
        assert lifecycle["marker"].is_file()
        marker = json.loads(lifecycle["marker"].read_text(encoding="utf-8"))
        assert marker["status"] == "claimed_before_inherited_locked_content_read"
        return original(path, repo_root=repo_root, expected=expected)

    monkeypatch.setattr(evaluator, "_audit_and_read_locked_rows", guarded)
    monkeypatch.setattr(evaluator, "prepare_teacher_samples", lambda rows: [])
    receipt_path = lifecycle["root"] / "attempt" / "locked-receipt.json"
    receipt = evaluator.evaluate_attempt02_locked_holdout_once(
        model_path=lifecycle["model_path"],
        freeze_manifest_path=lifecycle["freeze_path"],
        data_contract_path=lifecycle["contract_path"],
        plan_path=lifecycle["plan_path"],
        population_plan_path=lifecycle["population_plan_path"],
        repo_root=lifecycle["root"],
        receipt_path=receipt_path,
    )
    assert receipt["schema"] == M43_ATTEMPT02_LOCKED_RECEIPT_SCHEMA
    assert receipt["threshold_search_performed"] is False
    assert receipt["population_plan_file_sha256"] == lifecycle["freeze"][
        "population_plan_file_sha256"
    ]
    assert lifecycle["marker"].is_file()
    assert json.loads(lifecycle["marker"].read_text(encoding="utf-8"))[
        "schema"
    ] == M43_ATTEMPT02_MARKER_SCHEMA
    assert not (receipt_path.parent / "M43_LOCKED_CONSUMED.json").exists()
    assert "consumption_marker_path" not in inspect.signature(
        evaluator.evaluate_attempt02_locked_holdout_once
    ).parameters
    marker_payload = json.loads(lifecycle["marker"].read_text(encoding="utf-8"))
    assert marker_payload["population_plan_file_sha256"] == lifecycle["freeze"][
        "population_plan_file_sha256"
    ]
    tampered = copy.deepcopy(receipt)
    tampered["threshold_search_performed"] = True
    tampered["receipt_sha256"] = self_digest(tampered, "receipt_sha256")
    with pytest.raises(ValueError, match="unsafe flag"):
        validate_attempt02_locked_receipt(
            tampered,
            freeze=lifecycle["freeze"],
            contract=lifecycle["contract"],
            marker=marker_payload,
        )

    with pytest.raises(FileExistsError):
        evaluator.evaluate_attempt02_locked_holdout_once(
            model_path=lifecycle["model_path"],
            freeze_manifest_path=lifecycle["freeze_path"],
            data_contract_path=lifecycle["contract_path"],
            plan_path=lifecycle["plan_path"],
            population_plan_path=lifecycle["population_plan_path"],
            repo_root=lifecycle["root"],
            receipt_path=lifecycle["root"] / "another-attempt" / "receipt.json",
        )


def test_one_shot_rejects_population_plan_byte_change_before_claim(
    lifecycle, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        evaluator, "load_and_validate_attempt02_plan", lambda _: lifecycle["plan"]
    )
    # A semantic no-op still changes the frozen population-plan byte hash.
    with lifecycle["population_plan_path"].open("a", encoding="utf-8") as handle:
        handle.write("\n")
    receipt_path = lifecycle["root"] / "tampered-plan-receipt.json"
    with pytest.raises(ValueError, match="population plan disagrees with freeze"):
        evaluator.evaluate_attempt02_locked_holdout_once(
            model_path=lifecycle["model_path"],
            freeze_manifest_path=lifecycle["freeze_path"],
            data_contract_path=lifecycle["contract_path"],
            plan_path=lifecycle["plan_path"],
            population_plan_path=lifecycle["population_plan_path"],
            repo_root=lifecycle["root"],
            receipt_path=receipt_path,
        )
    assert not lifecycle["marker"].exists()
    assert not receipt_path.exists()


def test_attempt02_population_preflight_binds_real_v4_one_shot_chain(
    lifecycle, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        evaluator, "load_and_validate_attempt02_plan", lambda _: lifecycle["plan"]
    )
    monkeypatch.setattr(evaluator, "prepare_teacher_samples", lambda rows: [])
    receipt_path = lifecycle["root"] / "attempt" / "locked-receipt.json"
    evaluator.evaluate_attempt02_locked_holdout_once(
        model_path=lifecycle["model_path"],
        freeze_manifest_path=lifecycle["freeze_path"],
        data_contract_path=lifecycle["contract_path"],
        plan_path=lifecycle["plan_path"],
        population_plan_path=lifecycle["population_plan_path"],
        repo_root=lifecycle["root"],
        receipt_path=receipt_path,
    )
    population_plan_path = lifecycle["population_plan_path"]
    preflight = build_attempt02_population_launch_preflight(
        model_path=lifecycle["model_path"],
        training_manifest_path=lifecycle["manifest_path"],
        data_contract_path=lifecycle["contract_path"],
        freeze_manifest_path=lifecycle["freeze_path"],
        locked_receipt_path=receipt_path,
        consumption_marker_path=lifecycle["marker"],
        population_plan_path=population_plan_path,
    )
    assert preflight["schema"] == ATTEMPT02_POPULATION_PREFLIGHT_SCHEMA
    assert preflight["canonical_global_marker_verified"] is True
    assert preflight["teacher_calibration_locked_content_packaged"] is False
    assert preflight["model_schema"] == HU_M43_V4_MODEL_SCHEMA
    assert preflight["population_plan_file_sha256"] == lifecycle["freeze"][
        "population_plan_file_sha256"
    ]

    marker = json.loads(lifecycle["marker"].read_text(encoding="utf-8"))
    marker["model_sha256"] = "f" * 64
    marker["marker_sha256"] = self_digest(marker, "marker_sha256")
    lifecycle["marker"].write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="lifecycle binding mismatch"):
        build_attempt02_population_launch_preflight(
            model_path=lifecycle["model_path"],
            training_manifest_path=lifecycle["manifest_path"],
            data_contract_path=lifecycle["contract_path"],
            freeze_manifest_path=lifecycle["freeze_path"],
            locked_receipt_path=receipt_path,
            consumption_marker_path=lifecycle["marker"],
            population_plan_path=population_plan_path,
        )


def test_locked_tamper_after_freeze_consumes_marker_even_when_evaluation_crashes(
    lifecycle, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        evaluator, "load_and_validate_attempt02_plan", lambda _: lifecycle["plan"]
    )
    with lifecycle["locked"].open("a", encoding="utf-8") as handle:
        handle.write("\n")
    receipt = lifecycle["root"] / "receipt.json"
    with pytest.raises(ValueError, match="byte size mismatch"):
        evaluator.evaluate_attempt02_locked_holdout_once(
            model_path=lifecycle["model_path"],
            freeze_manifest_path=lifecycle["freeze_path"],
            data_contract_path=lifecycle["contract_path"],
            plan_path=lifecycle["plan_path"],
            population_plan_path=lifecycle["population_plan_path"],
            repo_root=lifecycle["root"],
            receipt_path=receipt,
        )
    assert lifecycle["marker"].is_file()
    assert not receipt.exists()
    with pytest.raises(FileExistsError):
        evaluator.evaluate_attempt02_locked_holdout_once(
            model_path=lifecycle["model_path"],
            freeze_manifest_path=lifecycle["freeze_path"],
            data_contract_path=lifecycle["contract_path"],
            plan_path=lifecycle["plan_path"],
            population_plan_path=lifecycle["population_plan_path"],
            repo_root=lifecycle["root"],
            receipt_path=receipt,
        )


def test_marker_publication_is_atomic_and_immutable(tmp_path: Path) -> None:
    marker = tmp_path / "global" / "M43_LOCKED_CONSUMED.json"

    def claim(value: int):
        try:
            write_immutable_json(marker, {"claim": value})
            return "created"
        except FileExistsError:
            return "exists"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, (1, 2)))
    assert sorted(results) == ["created", "exists"]
    assert json.loads(marker.read_text(encoding="utf-8"))["claim"] in {1, 2}
    with pytest.raises(FileExistsError):
        write_immutable_json(marker, {"claim": 3})


def test_dispatcher_keeps_existing_v3_loader_behavior(tmp_path: Path) -> None:
    x = np.zeros((2, HU_FEATURE_DIM), dtype=np.float32)
    safety_x = np.zeros((2, 18), dtype=np.float32)
    artifact = HuM4JointActionModel(
        policy_estimator=DummyClassifier(strategy="constant", constant=1).fit(
            x, np.ones(2, dtype=np.int8)
        ),
        value_estimator=DummyRegressor(strategy="constant", constant=0.0).fit(
            x, np.zeros(2)
        ),
        delta_estimator=DummyRegressor(strategy="constant", constant=0.0).fit(
            x, np.zeros(2)
        ),
        uncertainty_estimator=DummyRegressor(
            strategy="constant", constant=1.0
        ).fit(x, np.ones(2)),
        safety_estimator=DummyClassifier(strategy="constant", constant=1).fit(
            safety_x, np.ones(2, dtype=np.int8)
        ),
        safety_threshold=0.5,
        safety_enabled=True,
        model_id="v3-regression",
    )
    path = tmp_path / "v3.pkl"
    artifact.save(path)
    loaded = load_hu_m43_joint_action_model(path)
    assert isinstance(loaded, HuM4JointActionModel)
    assert loaded.model_id == "v3-regression"


def test_lifecycle_never_mutates_current_profile_source(lifecycle) -> None:
    profile_source = Path(__file__).resolve().parents[1] / "src" / "ofc_regular" / "ai_profiles.py"
    before = file_sha256(profile_source)
    validate_attempt02_freeze(lifecycle["freeze"], contract=lifecycle["contract"])
    assert file_sha256(profile_source) == before
    assert lifecycle["freeze"]["activation_guards"]["current_profile_changed"] is False
