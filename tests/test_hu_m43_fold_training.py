from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest

from ofc_regular.assemble_hu_m43_fold_training import verify_spot_run_manifest
from ofc_regular.hu_m43_fold_training import (
    M43_FOLD_CLOUD_CONTRACT_SCHEMA,
    M43_FOLD_ESTIMATOR_ARTIFACT_SCHEMA,
    M43_FOLD_JOB_DONE_SCHEMA,
    M43_FOLD_JOB_MANIFEST_SCHEMA,
    M43_FROZEN_DEPENDENCIES,
    M43_FROZEN_PROCESS_ENVIRONMENT,
    _validate_existing_done,
    _validate_predeclared_training_receipt,
    load_fold_artifact_provider,
)
from ofc_regular.hu_m43_pilot_contract import canonical_manifest_sha256
from ofc_regular.hu_m4_joint_model import (
    HU_M4_PAIRED_ACTION_FEATURE_SCHEMA,
    PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE,
    ConstantProbabilityEstimator,
    PairedDeltaRiskFoldEstimator,
)
from ofc_regular.hu_turn3_model import HU_FEATURE_DIM
from ofc_regular.train_hu_m4_joint_model import (
    M43FoldJobDefinition,
    M43FoldJobSpec,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _hyperparameters() -> dict:
    return {
        "cross_fit_folds": 5,
        "iterations": 150,
        "max_leaf_nodes": 31,
        "learning_rate": 0.05,
        "seed": 2026071801,
        "paired_se_floor": 0.5,
        "paired_huber_alpha": 0.9,
        "downside_quantile": 0.9,
        "positive_gain_score_weight": 0.25,
        "downside_risk_score_weight": 0.5,
        "ensemble_disagreement_score_weight": 0.25,
        "action_score_mode": PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE,
        "model_id": (
            "hu-m43-t1-second-regular-hu-m43-c2e16-pilot200-20260713-1810"
        ),
        "near_best_margin": 0.5,
        "minimum_safe_teacher_gain": 0.0,
        "l2_regularization": 1.0,
        "safety_calibrator_c": 0.25,
        "safety_fit_ratio": 0.5,
        "safety_split_seed": 2026071802,
        "minimum_safety_fit_samples": 30,
        "minimum_threshold_lock_samples": 30,
        "thresholds": [
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            0.95,
            0.975,
            0.99,
            1.0,
        ],
        "minimum_calibration_fires": 10,
        "maximum_false_positive_rate": 0.3,
        "maximum_p95_loss": 25.0,
        "maximum_p99_loss": 40.0,
        "maximum_max_loss": 50.0,
    }


def _receipt(config: dict | None = None) -> dict:
    return {
        "schema": "hu_m43_local_training_abort_receipt_v1",
        "status": "aborted_before_model_or_manifest_write",
        "model_written": False,
        "training_manifest_written": False,
        "locked_holdout_path_passed_to_trainer": False,
        "locked_holdout_opened_by_trainer": False,
        "locked_holdout_consumed": False,
        "safe_to_start_new_pre_holdout_training_attempt": True,
        "current_profile_changed": False,
        "runtime_policy_activated": False,
        "predeclared_training_configuration_for_equivalent_attempt": (
            config or _hyperparameters()
        ),
    }


def _spec(index: int) -> M43FoldJobSpec:
    outer = index // 6
    slot = index % 6
    inner = None if slot == 0 else slot - 1
    return M43FoldJobSpec(
        job_index=index,
        kind="outer_runtime" if inner is None else "inner_oof_safety",
        outer_fold=outer,
        inner_fold=inner,
        estimator_fold_index=outer if inner is None else inner,
        estimator_seed=2026071801 + index,
        fit_samples=80 if inner is None else 64,
        fit_identity_sha256=f"{index:064x}",
        outer_validation_samples=20,
        outer_validation_identity_sha256=f"{index + 100:064x}",
        inner_validation_samples=0 if inner is None else 16,
        inner_validation_identity_sha256=(
            None if inner is None else f"{index + 200:064x}"
        ),
        outer_assignment_sha256=SHA_A,
        inner_assignment_sha256=None if inner is None else SHA_B,
    )


def _projection() -> dict:
    inputs = {
        "train": [{"index": 0, "sha256": SHA_A, "bytes": 101, "rows": 100}],
        "calibration": [
            {"index": 0, "sha256": SHA_B, "bytes": 61, "rows": 60}
        ],
    }
    fold_plan = {
        "outer_folds": 5,
        "inner_folds_per_outer": 5,
        "total_jobs": 30,
        "train_identity_sha256": SHA_C,
        "jobs": [
            {**_spec(index).to_manifest(), "job_spec_sha256": _spec(index).sha256}
            for index in range(30)
        ],
    }
    fold_plan["fold_plan_sha256"] = canonical_manifest_sha256(fold_plan)
    unsigned = {
        "schema": M43_FOLD_CLOUD_CONTRACT_SCHEMA,
        "status": "frozen_cloud_safe",
        "predeclared_training_receipt_sha256": SHA_D,
        "inputs": inputs,
        "input_bundle_sha256": canonical_manifest_sha256({"inputs": inputs}),
        "fold_plan": fold_plan,
        "hyperparameters": _hyperparameters(),
        "dependencies": dict(M43_FROZEN_DEPENDENCIES),
        "process_environment": dict(M43_FROZEN_PROCESS_ENVIRONMENT),
        "cloud_safe": True,
    }
    return {**unsigned, "contract_sha256": canonical_manifest_sha256(unsigned)}


def _run_manifest(projection_path: Path, projection: dict) -> dict:
    run_name = "m43-fold-attempt02-test"
    bucket = "m43-test-bucket"
    inputs = {
        split: [
            {
                **entry,
                "uri": f"gs://{bucket}/runs/{run_name}/inputs/{split}-{i:03d}.jsonl",
            }
            for i, entry in enumerate(projection["inputs"][split])
        ]
        for split in ("train", "calibration")
    }
    inputs["input_bundle_sha256"] = projection["input_bundle_sha256"]
    jobs = []
    for index, projection_job in enumerate(projection["fold_plan"]["jobs"]):
        spec = dict(projection_job)
        digest = spec.pop("job_spec_sha256")
        prefix = f"gs://{bucket}/runs/{run_name}/results/job-{index:02d}"
        jobs.append(
            {
                "job_index": index,
                "job_kind": spec["kind"],
                "outer_fold": spec["outer_fold"],
                "inner_fold": spec["inner_fold"],
                "job_spec": spec,
                "job_spec_sha256": digest,
                "shard_index": index // 4,
                "result_prefix": prefix,
                "done_uri": f"{prefix}/DONE.json",
            }
        )
    return {
        "schema": "hu_m43_fold_spot_run_manifest_v1",
        "status": "frozen",
        "run_name": run_name,
        "project_id": "m43-test-project",
        "bucket": bucket,
        "job_count": 30,
        "source": {
            "uri": f"gs://{bucket}/runs/{run_name}/source/source.zip",
            "bytes": 1,
            "sha256": SHA_A,
            "fold_worker_sha256": SHA_B,
            "assembler_sha256": SHA_C,
        },
        "cloud_contract": {
            "uri": f"gs://{bucket}/runs/{run_name}/inputs/fold_cloud_contract.json",
            "bytes": projection_path.stat().st_size,
            "file_sha256": _sha256(projection_path),
            "contract_sha256": projection["contract_sha256"],
            "predeclared_training_receipt_sha256": projection[
                "predeclared_training_receipt_sha256"
            ],
            "train_identity_sha256": projection["fold_plan"][
                "train_identity_sha256"
            ],
            "fold_plan_sha256": projection["fold_plan"]["fold_plan_sha256"],
        },
        "inputs": inputs,
        "jobs": jobs,
        "training_config": projection["hyperparameters"],
        "training_config_sha256": canonical_manifest_sha256(
            projection["hyperparameters"]
        ),
        "dependencies": projection["dependencies"],
        "process_environment": projection["process_environment"],
        "compute": {
            "shard_count": 8,
            "jobs_per_shard": 4,
            "spot": True,
            "termination_action": "DELETE",
            "auto_delete": True,
        },
        "current_profile_mutated": False,
        "no_runtime_activation": True,
    }


def test_predeclared_receipt_requires_exact_safe_flags_and_configuration(tmp_path):
    path = tmp_path / "abort.json"
    _write_json(path, _receipt())
    validated = _validate_predeclared_training_receipt(
        path, expected_hyperparameters=_hyperparameters()
    )
    assert validated["locked_holdout_consumed"] is False

    changed_model = deepcopy(_hyperparameters())
    changed_model["model_id"] = "forged-model-id"
    with pytest.raises(ValueError, match="disagree with the predeclared"):
        _validate_predeclared_training_receipt(
            path, expected_hyperparameters=changed_model
        )

    unsafe = _receipt()
    unsafe["locked_holdout_opened_by_trainer"] = True
    _write_json(path, unsafe)
    with pytest.raises(ValueError, match="unsafe flag"):
        _validate_predeclared_training_receipt(
            path, expected_hyperparameters=_hyperparameters()
        )


def test_spot_run_manifest_exactly_binds_projection_specs_inputs_and_runtime(tmp_path):
    projection = _projection()
    projection_path = tmp_path / "fold_cloud_contract.json"
    _write_json(projection_path, projection)
    manifest = _run_manifest(projection_path, projection)
    manifest_path = tmp_path / "run_manifest.json"
    _write_json(manifest_path, manifest)
    result = verify_spot_run_manifest(
        run_manifest_path=manifest_path,
        fold_cloud_contract_path=projection_path,
        expected_run_name=manifest["run_name"],
        expected_run_manifest_sha256=_sha256(manifest_path),
        expected_source_sha256=SHA_A,
        hyperparameters=_hyperparameters(),
    )
    expected_spec = dict(projection["fold_plan"]["jobs"][29])
    expected_spec.pop("job_spec_sha256")
    assert result["jobs"][29]["job_spec"] == expected_spec
    assert result["inputs"]["train"][0]["rows"] == 100

    tampered = deepcopy(manifest)
    tampered["inputs"]["train"][0]["rows"] = 99
    _write_json(manifest_path, tampered)
    with pytest.raises(ValueError, match="provenance"):
        verify_spot_run_manifest(
            run_manifest_path=manifest_path,
            fold_cloud_contract_path=projection_path,
            expected_run_name=manifest["run_name"],
            expected_run_manifest_sha256=_sha256(manifest_path),
            expected_source_sha256=SHA_A,
            hyperparameters=_hyperparameters(),
        )

    tampered = deepcopy(manifest)
    tampered["jobs"][17]["job_spec"]["fit_samples"] += 1
    _write_json(manifest_path, tampered)
    with pytest.raises(ValueError, match="full job identity"):
        verify_spot_run_manifest(
            run_manifest_path=manifest_path,
            fold_cloud_contract_path=projection_path,
            expected_run_name=manifest["run_name"],
            expected_run_manifest_sha256=_sha256(manifest_path),
            expected_source_sha256=SHA_A,
            hyperparameters=_hyperparameters(),
        )


def _done_fixture(
    root: Path, definition: M43FoldJobDefinition
) -> tuple[dict, dict]:
    root.mkdir()
    artifact = root / "estimator.pkl"
    artifact.write_bytes(b"fold-estimator")
    spec = definition.spec
    manifest = {
        "schema": M43_FOLD_JOB_MANIFEST_SCHEMA,
        "status": "pass",
        "run_name": "attempt02",
        "job_index": spec.job_index,
        "job_kind": spec.kind,
        "outer_fold": spec.outer_fold,
        "inner_fold": spec.inner_fold,
        "job_spec": spec.to_manifest(),
        "job_spec_sha256": spec.sha256,
        "source_sha256": SHA_A,
        "run_manifest_sha256": SHA_B,
        "cloud_contract_sha256": SHA_C,
        "input_bundle_sha256": SHA_D,
        "artifact_sha256": _sha256(artifact),
        "current_profile_mutated": False,
        "no_runtime_activation": True,
    }
    _write_json(root / "job_manifest.json", manifest)
    done = {
        "schema": M43_FOLD_JOB_DONE_SCHEMA,
        "status": "complete",
        "run_name": "attempt02",
        "job_index": spec.job_index,
        "job_kind": spec.kind,
        "outer_fold": spec.outer_fold,
        "inner_fold": spec.inner_fold,
        "job_spec_sha256": spec.sha256,
        "source_sha256": SHA_A,
        "run_manifest_sha256": SHA_B,
        "cloud_contract_sha256": SHA_C,
        "input_bundle_sha256": SHA_D,
        "artifact_sha256": _sha256(artifact),
        "job_manifest_sha256": _sha256(root / "job_manifest.json"),
        "current_profile_mutated": False,
        "no_runtime_activation": True,
    }
    _write_json(root / "DONE.json", done)
    return manifest, done


def test_existing_done_is_idempotent_only_for_the_exact_run_context(tmp_path):
    definition = M43FoldJobDefinition(spec=_spec(1), fit_samples=())
    job_dir = tmp_path / "job-01"
    _done_fixture(job_dir, definition)
    done = _validate_existing_done(
        job_dir,
        expected_definition=definition,
        expected_cloud_contract_sha256=SHA_C,
        expected_run_name="attempt02",
        expected_source_sha256=SHA_A,
        expected_run_manifest_sha256=SHA_B,
        expected_input_bundle_sha256=SHA_D,
    )
    assert done["status"] == "complete"
    with pytest.raises(ValueError, match="hash/identity chain"):
        _validate_existing_done(
            job_dir,
            expected_definition=definition,
            expected_cloud_contract_sha256=SHA_C,
            expected_run_name="stale-other-run",
            expected_source_sha256=SHA_A,
            expected_run_manifest_sha256=SHA_B,
            expected_input_bundle_sha256=SHA_D,
        )

    manifest = json.loads((job_dir / "job_manifest.json").read_text())
    done_payload = json.loads((job_dir / "DONE.json").read_text())
    manifest["job_spec"]["fit_samples"] = float(
        manifest["job_spec"]["fit_samples"]
    )
    _write_json(job_dir / "job_manifest.json", manifest)
    done_payload["job_manifest_sha256"] = _sha256(
        job_dir / "job_manifest.json"
    )
    _write_json(job_dir / "DONE.json", done_payload)
    with pytest.raises(ValueError, match="hash/identity chain"):
        _validate_existing_done(
            job_dir,
            expected_definition=definition,
            expected_cloud_contract_sha256=SHA_C,
            expected_run_name="attempt02",
            expected_source_sha256=SHA_A,
            expected_run_manifest_sha256=SHA_B,
            expected_input_bundle_sha256=SHA_D,
        )

    manifest["job_spec"] = definition.spec.to_manifest()
    manifest["inner_fold"] = float(definition.spec.inner_fold)
    done_payload["inner_fold"] = float(definition.spec.inner_fold)
    _write_json(job_dir / "job_manifest.json", manifest)
    done_payload["job_manifest_sha256"] = _sha256(
        job_dir / "job_manifest.json"
    )
    _write_json(job_dir / "DONE.json", done_payload)
    with pytest.raises(ValueError, match="must be an integer"):
        _validate_existing_done(
            job_dir,
            expected_definition=definition,
            expected_cloud_contract_sha256=SHA_C,
            expected_run_name="attempt02",
            expected_source_sha256=SHA_A,
            expected_run_manifest_sha256=SHA_B,
            expected_input_bundle_sha256=SHA_D,
        )


def _dummy_estimator(fold_index: int) -> PairedDeltaRiskFoldEstimator:
    head = ConstantProbabilityEstimator(0.5)
    return PairedDeltaRiskFoldEstimator(
        delta_estimator=head,
        positive_gain_estimator=head,
        downside_p95_estimator=head,
        downside_p99_estimator=head,
        downside_max_estimator=head,
        paired_feature_dim=4 * HU_FEATURE_DIM,
        fold_index=fold_index,
        feature_schema=HU_M4_PAIRED_ACTION_FEATURE_SCHEMA,
    )


def test_artifact_provider_requires_canonical_d2_exact_coverage(tmp_path, monkeypatch):
    import ofc_regular.hu_m43_fold_training as module

    contract_path = tmp_path / "projection.json"
    contract_path.write_text("{}", encoding="utf-8")
    contract_sha = _sha256(contract_path)
    definitions = tuple(
        M43FoldJobDefinition(spec=_spec(index), fit_samples=())
        for index in range(30)
    )
    monkeypatch.setattr(
        module,
        "load_fold_cloud_contract",
        lambda _path: {"input_bundle_sha256": SHA_D},
    )
    monkeypatch.setattr(
        module,
        "_verify_cloud_inputs",
        lambda *_args, **_kwargs: ([], SimpleNamespace(jobs=definitions)),
    )
    root = tmp_path / "folds"
    root.mkdir()
    for definition in definitions:
        spec = definition.spec
        job_dir = root / f"job-{spec.job_index:02d}"
        job_dir.mkdir()
        artifact_path = job_dir / "estimator.pkl"
        artifact_path.write_bytes(
            pickle.dumps(
                {
                    "schema": M43_FOLD_ESTIMATOR_ARTIFACT_SCHEMA,
                    "job_spec_sha256": spec.sha256,
                    "estimator": _dummy_estimator(spec.estimator_fold_index),
                },
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        )
        manifest = {
            "schema": M43_FOLD_JOB_MANIFEST_SCHEMA,
            "status": "pass",
            "run_name": "attempt02",
            "job_index": spec.job_index,
            "job_kind": spec.kind,
            "outer_fold": spec.outer_fold,
            "inner_fold": spec.inner_fold,
            "job_spec": spec.to_manifest(),
            "job_spec_sha256": spec.sha256,
            "source_sha256": SHA_A,
            "run_manifest_sha256": SHA_B,
            "cloud_contract_sha256": contract_sha,
            "input_bundle_sha256": SHA_D,
            "artifact_sha256": _sha256(artifact_path),
            "current_profile_mutated": False,
            "no_runtime_activation": True,
        }
        _write_json(job_dir / "job_manifest.json", manifest)
        done = {
            key: value
            for key, value in manifest.items()
            if key not in {"job_spec"}
        }
        done.update(
            {
                "schema": M43_FOLD_JOB_DONE_SCHEMA,
                "status": "complete",
                "job_manifest_sha256": _sha256(job_dir / "job_manifest.json"),
            }
        )
        _write_json(job_dir / "DONE.json", done)
    provider = load_fold_artifact_provider(
        artifacts_dir=root,
        fold_cloud_contract_path=contract_path,
        train_path=tmp_path / "unused-train",
        calibration_path=tmp_path / "unused-calibration",
    )
    assert len(provider.estimators) == 30

    (root / "job-00").rename(root / "job-0000")
    with pytest.raises(ValueError, match="coverage mismatch"):
        load_fold_artifact_provider(
            artifacts_dir=root,
            fold_cloud_contract_path=contract_path,
            train_path=tmp_path / "unused-train",
            calibration_path=tmp_path / "unused-calibration",
        )
