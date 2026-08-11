from __future__ import annotations

from copy import deepcopy
import inspect
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest

from ofc_regular import hu_m43_attempt02_fold_training as fold
from ofc_regular.hu_m43_joint_model_v4 import (
    HU_M43_V4_MODEL_SCHEMA,
    HU_M43_V4_PROPOSAL_SCHEMA,
)
from ofc_regular.hu_m4_joint_model import (
    ConstantProbabilityEstimator,
    PairedDeltaRiskFoldEstimator,
)
from ofc_regular.hu_turn3_model import HU_FEATURE_DIM
from ofc_regular.train_hu_m43_attempt02_fold_job import parse_args
from ofc_regular.train_hu_m4_joint_model import (
    M43FoldJobDefinition,
    M43FoldJobSpec,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


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
        fit_samples=160 if inner is None else 128,
        fit_identity_sha256=f"{index + 1:064x}",
        outer_validation_samples=40,
        outer_validation_identity_sha256=f"{index + 101:064x}",
        inner_validation_samples=0 if inner is None else 32,
        inner_validation_identity_sha256=(
            None if inner is None else f"{index + 201:064x}"
        ),
        outer_assignment_sha256=SHA_A,
        inner_assignment_sha256=None if inner is None else SHA_B,
    )


def _estimator(fold_index: int) -> PairedDeltaRiskFoldEstimator:
    head = ConstantProbabilityEstimator(0.5)
    return PairedDeltaRiskFoldEstimator(
        delta_estimator=head,
        positive_gain_estimator=head,
        downside_p95_estimator=head,
        downside_p99_estimator=head,
        downside_max_estimator=head,
        paired_feature_dim=4 * HU_FEATURE_DIM,
        fold_index=fold_index,
    )


def _contract() -> dict:
    config = fold.Attempt02V4TrainingConfig()
    return {
        "input_bundle_sha256": SHA_A,
        "training_config": config.to_manifest(),
        "training_config_sha256": fold.canonical_manifest_sha256(
            config.to_manifest()
        ),
        "contract_sha256": SHA_B,
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def test_cloud_contract_contains_train_only_and_frozen_v4_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = []
    rows = []
    for index in range(20):
        path = tmp_path / f"train-{index:02d}.jsonl"
        path.write_text(f"shard-{index}\n", encoding="utf-8")
        paths.append(path)
        rows.append([{} for _ in range(10)])
    definitions = tuple(
        M43FoldJobDefinition(spec=_spec(index), fit_samples=())
        for index in range(30)
    )
    plan = SimpleNamespace(ordered_samples=(), jobs=definitions)
    monkeypatch.setattr(
        fold,
        "_read_train_and_plan",
        lambda *_args, **_kwargs: (tuple(paths), rows, [object()] * 200, plan),
    )
    monkeypatch.setattr(fold, "_m43_sample_identity_sha256", lambda _rows: SHA_C)

    first = fold.build_attempt02_fold_cloud_contract(train_path=paths)
    second = fold.build_attempt02_fold_cloud_contract(train_path=paths)

    assert first == second
    assert set(first["inputs"]) == {"train"}
    assert len(first["inputs"]["train"]) == 20
    assert all("path" not in row for row in first["inputs"]["train"])
    assert first["worker_input_boundary"] == {
        "fresh_train_only": True,
        "train_rows": 200,
        "nontrain_input_count": 0,
        "sealed_contract_content_count": 0,
    }
    assert first["fold_plan"]["total_jobs"] == 30
    assert first["training_config"]["model_schema"] == HU_M43_V4_MODEL_SCHEMA
    assert first["training_config"]["proposal_schema"] == HU_M43_V4_PROPOSAL_SCHEMA
    serialized = json.dumps(first, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert "locked_holdout_path" not in serialized
    contract_path = tmp_path / "fold-cloud-contract.json"
    _write_json(contract_path, first)
    assert fold.load_attempt02_fold_cloud_contract(contract_path) == first

    tampered = deepcopy(first["training_config"])
    tampered["threshold_selection"]["thresholds"] = [0.5]
    with pytest.raises(ValueError, match="threshold grid"):
        fold.Attempt02V4TrainingConfig.from_manifest(tampered)


def test_worker_cli_and_function_have_no_nontrain_input_surface() -> None:
    parameters = inspect.signature(fold.run_attempt02_fold_job).parameters
    assert set(parameters) == {
        "job_index",
        "train_path",
        "fold_cloud_contract_path",
        "output_dir",
        "run_name",
        "source_sha256",
        "run_manifest_sha256",
        "input_bundle_sha256",
        "job_spec_sha256",
    }
    valid = [
        "run",
        "--job-index",
        "0",
        "--train",
        "train.jsonl",
        "--fold-cloud-contract",
        "contract.json",
        "--output-dir",
        "job-00",
        "--run-name",
        "attempt02",
        "--source-sha256",
        SHA_A,
        "--run-manifest-sha256",
        SHA_B,
        "--input-bundle-sha256",
        SHA_A,
        "--job-spec-sha256",
        SHA_C,
    ]
    args = parse_args(valid)
    assert not hasattr(args, "calibration")
    assert not hasattr(args, "attempt02_data_contract")
    assert not hasattr(args, "locked_holdout")
    with pytest.raises(SystemExit):
        parse_args([*valid, "--calibration", "forbidden.jsonl"])


def test_fold_job_is_byte_deterministic_idempotent_and_detects_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract_path = tmp_path / "contract.json"
    contract_path.write_text("{}\n", encoding="utf-8")
    contract = _contract()
    definition = M43FoldJobDefinition(spec=_spec(0), fit_samples=())
    plan = SimpleNamespace(jobs=(definition,))
    monkeypatch.setattr(
        fold, "load_attempt02_fold_cloud_contract", lambda *_args, **_kwargs: contract
    )
    monkeypatch.setattr(
        fold,
        "_load_train_plan",
        lambda *_args, **_kwargs: ([], plan),
    )
    monkeypatch.setattr(
        fold,
        "fit_v4_fold_worker_compatible",
        lambda *_args, **_kwargs: _estimator(0),
    )
    common = {
        "job_index": 0,
        "train_path": tmp_path / "not-opened-by-mock.jsonl",
        "fold_cloud_contract_path": contract_path,
        "run_name": "attempt02",
        "source_sha256": SHA_A,
        "run_manifest_sha256": SHA_B,
        "input_bundle_sha256": SHA_A,
        "job_spec_sha256": definition.spec.sha256,
    }
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = fold.run_attempt02_fold_job(output_dir=first_dir, **common)
    second = fold.run_attempt02_fold_job(output_dir=second_dir, **common)
    assert first == second
    for name in ("estimator.pkl", "job_manifest.json", "DONE.json"):
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()
    assert fold.run_attempt02_fold_job(output_dir=first_dir, **common) == first

    with (first_dir / "estimator.pkl").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="manifest lineage"):
        fold.run_attempt02_fold_job(output_dir=first_dir, **common)


def test_artifact_provider_requires_exact_30_job_hash_lineage_and_tamper_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract_path = tmp_path / "contract.json"
    contract_path.write_text("cloud-contract\n", encoding="utf-8")
    contract = _contract()
    definitions = tuple(
        M43FoldJobDefinition(spec=_spec(index), fit_samples=())
        for index in range(30)
    )
    plan = SimpleNamespace(jobs=definitions)
    monkeypatch.setattr(
        fold, "load_attempt02_fold_cloud_contract", lambda *_args, **_kwargs: contract
    )
    monkeypatch.setattr(
        fold,
        "_load_train_plan",
        lambda *_args, **_kwargs: ([], plan),
    )
    root = tmp_path / "artifacts"
    for definition in definitions:
        spec = definition.spec
        job_dir = root / f"job-{spec.job_index:02d}"
        job_dir.mkdir(parents=True)
        artifact = job_dir / "estimator.pkl"
        artifact.write_bytes(
            pickle.dumps(
                {
                    "schema": fold.M43_ATTEMPT02_FOLD_ESTIMATOR_SCHEMA,
                    "model_schema": HU_M43_V4_MODEL_SCHEMA,
                    "proposal_schema": HU_M43_V4_PROPOSAL_SCHEMA,
                    "training_config_sha256": contract[
                        "training_config_sha256"
                    ],
                    "job_spec_sha256": spec.sha256,
                    "estimator": _estimator(spec.estimator_fold_index),
                },
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        )
        manifest = fold._job_manifest(
            spec,
            run_name="attempt02",
            source_sha256=SHA_A,
            run_manifest_sha256=SHA_B,
            contract_file_sha256=fold._file_sha256(contract_path),
            input_bundle_sha256=SHA_A,
            training_config_sha256=contract["training_config_sha256"],
            artifact_sha256=fold._file_sha256(artifact),
        )
        _write_json(job_dir / "job_manifest.json", manifest)
        done = {
            key: value
            for key, value in manifest.items()
            if key not in {"job_spec", "schema", "status"}
        }
        done.update(
            {
                "schema": fold.M43_ATTEMPT02_FOLD_DONE_SCHEMA,
                "status": "complete",
                "job_manifest_sha256": fold._file_sha256(
                    job_dir / "job_manifest.json"
                ),
            }
        )
        _write_json(job_dir / "DONE.json", done)

    provider = fold.load_attempt02_fold_artifact_provider(
        artifacts_dir=root,
        fold_cloud_contract_path=contract_path,
        train_path=tmp_path / "unused-train",
        expected_run_name="attempt02",
        expected_source_sha256=SHA_A,
        expected_run_manifest_sha256=SHA_B,
    )
    assert len(provider.estimators) == 30
    assert provider.assembly_receipt["exact_outer_inner_coverage"] is True

    with (root / "job-17" / "estimator.pkl").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="hash/spec chain"):
        fold.load_attempt02_fold_artifact_provider(
            artifacts_dir=root,
            fold_cloud_contract_path=contract_path,
            train_path=tmp_path / "unused-train",
            expected_run_name="attempt02",
            expected_source_sha256=SHA_A,
            expected_run_manifest_sha256=SHA_B,
        )
